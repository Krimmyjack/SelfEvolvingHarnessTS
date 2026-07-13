"""Append-only, hash-chained Final evaluation campaign ledger."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

if os.name == "nt":
    import msvcrt
else:  # pragma: no cover - exercised on POSIX runners
    import fcntl

LEDGER_SCHEMA = "benchmark-final-campaign/1"
REQUIRED_RESUME_FIELDS = (
    "campaign_id",
    "run_id",
    "entry_id",
    "method_code_sha",
    "runner_code_sha",
    "input_manifest_sha",
    "materialization_sha",
    "checkpoint_sha",
)


class CampaignStateError(RuntimeError):
    """The Final campaign transition or ledger bytes violate protocol."""


def _canonical_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    value = _canonical_string(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CampaignStateError("ledger value is not canonical JSON") from exc


@dataclass(frozen=True)
class CampaignEntry:
    entry_id: str
    method_code_sha: str
    order: int
    budget: int
    dry_run_sha: str | None = None
    confirmation_sha: str | None = None

    def __post_init__(self) -> None:
        _canonical_string(self.entry_id, "entry_id")
        _sha256(self.method_code_sha, "method_code_sha")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise ValueError("entry order must be a non-negative integer")
        if isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0:
            raise ValueError("entry budget must be a non-negative integer")
        if self.dry_run_sha is not None:
            _sha256(self.dry_run_sha, "dry_run_sha")
        if self.confirmation_sha is not None:
            _sha256(self.confirmation_sha, "confirmation_sha")


@dataclass(frozen=True)
class CampaignManifest:
    campaign_id: str
    benchmark_version: str
    input_manifest_sha: str
    materialization_sha: str
    runner_code_sha: str
    entries: tuple[CampaignEntry, ...]

    def __post_init__(self) -> None:
        _canonical_string(self.campaign_id, "campaign_id")
        _canonical_string(self.benchmark_version, "benchmark_version")
        _sha256(self.input_manifest_sha, "input_manifest_sha")
        _sha256(self.materialization_sha, "materialization_sha")
        _sha256(self.runner_code_sha, "runner_code_sha")
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ValueError("campaign roster must be a non-empty immutable tuple")
        if not all(isinstance(entry, CampaignEntry) for entry in self.entries):
            raise TypeError("campaign roster must contain CampaignEntry values")
        ids = [entry.entry_id for entry in self.entries]
        orders = [entry.order for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("campaign roster entry ids must be unique")
        if sorted(orders) != list(range(len(self.entries))):
            raise ValueError("campaign roster order must be contiguous from zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "benchmark_version": self.benchmark_version,
            "input_manifest_sha": self.input_manifest_sha,
            "materialization_sha": self.materialization_sha,
            "runner_code_sha": self.runner_code_sha,
            "entries": [asdict(entry) for entry in sorted(self.entries, key=lambda item: item.order)],
        }

    @property
    def manifest_sha(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def entry(self, entry_id: str) -> CampaignEntry:
        try:
            return next(entry for entry in self.entries if entry.entry_id == entry_id)
        except StopIteration:
            raise CampaignStateError(f"entry {entry_id!r} is not in the frozen roster") from None


class MethodResultStatus(str, Enum):
    COMPLETE = "complete"
    INVALID = "invalid"
    FAILED_TIMEOUT = "failed_timeout"
    INFRA_INTERRUPTED = "infra_interrupted"

    @property
    def terminal(self) -> bool:
        return self is not MethodResultStatus.INFRA_INTERRUPTED


@dataclass(frozen=True)
class ResumeBinding:
    campaign_id: str
    run_id: str
    entry_id: str
    method_code_sha: str
    runner_code_sha: str
    input_manifest_sha: str
    materialization_sha: str
    checkpoint_sha: str

    def __post_init__(self) -> None:
        for name in ("campaign_id", "run_id", "entry_id"):
            _canonical_string(getattr(self, name), name)
        for name in REQUIRED_RESUME_FIELDS[3:]:
            _sha256(getattr(self, name), name)


class _ExclusiveLedgerLock:
    def __init__(self, path: Path) -> None:
        self._fd: int | None = None
        fd = os.open(
            str(path),
            os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0),
        )
        try:
            if os.name == "nt":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise CampaignStateError("campaign ledger is already exclusively locked") from exc
        self._fd = fd

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:  # pragma: no cover
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            os.close(fd)


def compute_event_sha(event: Mapping[str, Any]) -> str:
    clean = {key: event[key] for key in event if key != "event_sha"}
    return hashlib.sha256(_canonical_json(clean).encode("utf-8")).hexdigest()


class CampaignLedger:
    """A single durable Final campaign bound to one immutable manifest."""

    def __init__(self, path: str | Path, manifest: CampaignManifest) -> None:
        if not isinstance(manifest, CampaignManifest):
            raise TypeError("manifest must be CampaignManifest")
        self._manifest = manifest
        self.path = Path(path)
        self._fd: int | None = None
        self._lock: _ExclusiveLedgerLock | None = None
        self._resource_closed = False
        self._events: list[dict[str, Any]] = []
        self._seq = 0
        self._tip = hashlib.sha256(
            f"benchmark-campaign-genesis|{manifest.manifest_sha}".encode("utf-8")
        ).hexdigest()
        self._unsealed = False
        self._campaign_closed = False
        self._accesses: dict[str, str] = {}
        self._results: dict[str, tuple[MethodResultStatus, str, ResumeBinding | None]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _ExclusiveLedgerLock(self.path.with_name(self.path.name + ".lock"))
        try:
            if self.path.exists() and self.path.stat().st_size:
                self._replay()
            self._fd = os.open(
                str(self.path),
                os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0),
            )
            if not self._events:
                self._commit("campaign_freeze", {})
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "CampaignLedger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        if self._resource_closed:
            return
        self._resource_closed = True
        fd, self._fd = self._fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.release()

    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._events))

    def _ensure_resource_open(self) -> None:
        if self._resource_closed or self._fd is None:
            raise CampaignStateError("campaign ledger resource is closed")

    def _ensure_campaign_open(self) -> None:
        self._ensure_resource_open()
        if self._campaign_closed:
            raise CampaignStateError("campaign is permanently closed")

    def unseal(self) -> str:
        self._ensure_campaign_open()
        if self._unsealed:
            raise CampaignStateError("Final campaign can be unsealed only once")
        return self._commit("unseal", {})

    def record_access(self, entry_id: str, run_id: str) -> str:
        self._ensure_campaign_open()
        entry = self._manifest.entry(entry_id)
        _canonical_string(run_id, "run_id")
        if not self._unsealed:
            raise CampaignStateError("Final campaign must be unsealed before access")
        existing_result = self._results.get(entry.entry_id)
        if existing_result is not None and existing_result[0].terminal:
            raise CampaignStateError(f"entry {entry.entry_id!r} already has a terminal result")
        if entry.entry_id in self._accesses:
            raise CampaignStateError("each roster entry receives exactly one access event")
        return self._commit("method_access", {"entry_id": entry.entry_id, "run_id": run_id})

    def record_result(
        self,
        entry_id: str,
        run_id: str,
        status: MethodResultStatus,
        result_digest: str,
        *,
        resume_binding: ResumeBinding | None = None,
    ) -> str:
        self._ensure_campaign_open()
        self._manifest.entry(entry_id)
        _canonical_string(run_id, "run_id")
        _sha256(result_digest, "result_digest")
        if not isinstance(status, MethodResultStatus):
            raise TypeError("status must be MethodResultStatus")
        if self._accesses.get(entry_id) != run_id:
            raise CampaignStateError("method result must match its durable access event")
        existing = self._results.get(entry_id)
        if existing is not None and existing[0].terminal:
            raise CampaignStateError("terminal method results cannot be overwritten")
        if existing is not None and existing[1] != run_id:
            raise CampaignStateError("infrastructure resume must retain the original run id")
        if status is MethodResultStatus.INFRA_INTERRUPTED:
            if resume_binding is None:
                raise CampaignStateError("infrastructure interruption requires a resume binding")
            self._validate_binding(resume_binding, entry_id=entry_id, run_id=run_id)
            if resume_binding.checkpoint_sha != result_digest:
                raise CampaignStateError("checkpoint digest disagrees with resume binding")
        elif resume_binding is not None:
            raise CampaignStateError("only infrastructure interruption accepts a resume binding")
        fields: dict[str, Any] = {
            "entry_id": entry_id,
            "run_id": run_id,
            "status": status.value,
            "result_digest": result_digest,
        }
        if resume_binding is not None:
            fields["resume_binding"] = asdict(resume_binding)
        return self._commit("method_result", fields)

    def resume(self, binding: ResumeBinding) -> str:
        self._ensure_campaign_open()
        if not isinstance(binding, ResumeBinding):
            raise TypeError("binding must be ResumeBinding")
        self._validate_binding(binding, entry_id=binding.entry_id, run_id=binding.run_id)
        existing = self._results.get(binding.entry_id)
        if existing is None or existing[0] is not MethodResultStatus.INFRA_INTERRUPTED:
            raise CampaignStateError("resume requires a recorded infrastructure interruption")
        if existing[2] != binding:
            raise CampaignStateError("resume binding differs from the durable interruption binding")
        return binding.run_id

    def close_campaign(self) -> str:
        self._ensure_campaign_open()
        terminal = {
            entry_id for entry_id, (status, _, _) in self._results.items() if status.terminal
        }
        roster = {entry.entry_id for entry in self._manifest.entries}
        if terminal != roster:
            raise CampaignStateError("campaign cannot close until the full roster is terminal")
        return self._commit("campaign_close", {})

    def _validate_binding(self, binding: ResumeBinding, *, entry_id: str, run_id: str) -> None:
        entry = self._manifest.entry(entry_id)
        expected = ResumeBinding(
            campaign_id=self._manifest.campaign_id,
            run_id=run_id,
            entry_id=entry_id,
            method_code_sha=entry.method_code_sha,
            runner_code_sha=self._manifest.runner_code_sha,
            input_manifest_sha=self._manifest.input_manifest_sha,
            materialization_sha=self._manifest.materialization_sha,
            checkpoint_sha=binding.checkpoint_sha,
        )
        if binding != expected:
            raise CampaignStateError("resume binding does not match the frozen campaign")

    def _commit(self, event: str, fields: Mapping[str, Any]) -> str:
        self._ensure_resource_open()
        item: dict[str, Any] = {
            "schema": LEDGER_SCHEMA,
            "seq": self._seq + 1,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "campaign_id": self._manifest.campaign_id,
            "campaign_manifest_sha": self._manifest.manifest_sha,
            "event": event,
            "prev_event_sha": self._tip,
        }
        overlap = set(item) & set(fields)
        if overlap:
            raise CampaignStateError(f"event fields conflict with headers: {sorted(overlap)}")
        item.update(copy.deepcopy(dict(fields)))
        item["event_sha"] = compute_event_sha(item)
        data = (_canonical_json(item) + "\n").encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(self._fd, view)
            view = view[written:]
        os.fsync(self._fd)
        self._apply(item)
        return item["event_sha"]

    def _apply(self, event: Mapping[str, Any]) -> None:
        kind = event["event"]
        if kind == "campaign_freeze":
            if self._events or event.get("campaign_manifest_sha") != self._manifest.manifest_sha:
                raise CampaignStateError("campaign freeze does not match the frozen manifest")
        elif kind == "unseal":
            if self._unsealed or self._campaign_closed:
                raise CampaignStateError("invalid repeated unseal event")
            self._unsealed = True
        elif kind == "method_access":
            if not self._unsealed or self._campaign_closed:
                raise CampaignStateError("method access occurred outside an open campaign")
            entry_id = str(event.get("entry_id"))
            self._manifest.entry(entry_id)
            if entry_id in self._accesses:
                raise CampaignStateError("duplicate method access event")
            self._accesses[entry_id] = _canonical_string(event.get("run_id"), "run_id")
        elif kind == "method_result":
            entry_id = str(event.get("entry_id"))
            self._manifest.entry(entry_id)
            run_id = _canonical_string(event.get("run_id"), "run_id")
            if self._accesses.get(entry_id) != run_id:
                raise CampaignStateError("method result is not linked to its access event")
            try:
                status = MethodResultStatus(event.get("status"))
            except ValueError as exc:
                raise CampaignStateError("unknown method result status") from exc
            _sha256(event.get("result_digest"), "result_digest")
            prior = self._results.get(entry_id)
            if prior is not None and prior[0].terminal:
                raise CampaignStateError("terminal result was overwritten in ledger")
            binding = None
            if status is MethodResultStatus.INFRA_INTERRUPTED:
                try:
                    binding = ResumeBinding(**dict(event.get("resume_binding") or {}))
                except (TypeError, ValueError) as exc:
                    raise CampaignStateError("invalid resume binding in ledger") from exc
                self._validate_binding(binding, entry_id=entry_id, run_id=run_id)
                if binding.checkpoint_sha != event.get("result_digest"):
                    raise CampaignStateError("resume checkpoint digest mismatch")
            elif event.get("resume_binding") is not None:
                raise CampaignStateError("terminal result contains a resume binding")
            self._results[entry_id] = (status, run_id, binding)
        elif kind == "campaign_close":
            roster = {entry.entry_id for entry in self._manifest.entries}
            terminal = {
                entry_id
                for entry_id, (status, _, _) in self._results.items()
                if status.terminal
            }
            if self._campaign_closed or terminal != roster:
                raise CampaignStateError("campaign close occurred before full roster completion")
            self._campaign_closed = True
        else:
            raise CampaignStateError(f"unknown campaign event {kind!r}")
        self._seq = int(event["seq"])
        self._tip = str(event["event_sha"])
        self._events.append(copy.deepcopy(dict(event)))

    def _replay(self) -> None:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CampaignStateError("campaign ledger cannot be read") from exc
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise CampaignStateError(f"campaign ledger line {line_number} is empty")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CampaignStateError("campaign ledger contains torn or invalid JSON") from exc
            required = {
                "schema",
                "seq",
                "ts",
                "campaign_id",
                "campaign_manifest_sha",
                "event",
                "prev_event_sha",
                "event_sha",
            }
            if not isinstance(event, dict) or not required <= set(event):
                raise CampaignStateError("campaign ledger event is missing required fields")
            if event["schema"] != LEDGER_SCHEMA:
                raise CampaignStateError("campaign ledger schema mismatch")
            if event["campaign_id"] != self._manifest.campaign_id:
                raise CampaignStateError("campaign id mismatch or tamper detected")
            if event["campaign_manifest_sha"] != self._manifest.manifest_sha:
                raise CampaignStateError("campaign manifest hash mismatch or tamper detected")
            if event["seq"] != self._seq + 1:
                raise CampaignStateError("campaign ledger sequence is not contiguous")
            if event["prev_event_sha"] != self._tip:
                raise CampaignStateError("campaign ledger hash chain is broken")
            if compute_event_sha(event) != event["event_sha"]:
                raise CampaignStateError("campaign ledger event hash indicates tamper")
            self._apply(event)
        if not self._events or self._events[0]["event"] != "campaign_freeze":
            raise CampaignStateError("campaign ledger lacks its freeze event")


__all__ = [
    "CampaignEntry",
    "CampaignLedger",
    "CampaignManifest",
    "CampaignStateError",
    "LEDGER_SCHEMA",
    "MethodResultStatus",
    "REQUIRED_RESUME_FIELDS",
    "ResumeBinding",
    "compute_event_sha",
]
