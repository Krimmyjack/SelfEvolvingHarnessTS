"""Durable access-before-read protocol for one-shot vNext resources.

The existing :mod:`benchmark.ledger` remains the frozen Final-campaign v1
implementation.  This module is the versioned, resource-level guard used by
SA-V, Support-B, and the vNext Final authorization bridge.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from ._canonical import canonical_json, require_sha, sha256

if os.name == "nt":  # pragma: no cover - CI and production are POSIX
    import msvcrt
else:  # pragma: no cover - branch selection is platform dependent
    import fcntl


ACCESS_SCHEMA = "vnext-one-shot-access/1"
RESOURCE_KINDS = frozenset({"sa_validation", "support_b", "final_query"})
TERMINAL_STATUSES = frozenset({
    "passed", "failed_gate", "invalid", "timeout", "budget_exceeded",
    "dependency_failure", "failed_infrastructure_terminal",
})


class OneShotAccessError(RuntimeError):
    """An access transition, receipt, or durable ledger is invalid."""


def _canonical_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string")
    return value


@dataclass(frozen=True)
class OneShotAccessManifestV1:
    resource_kind: str
    campaign_id: str
    prereg_sha: str
    authorization_sha: str
    resource_manifest_sha: str
    materialization_sha: str
    method_sha: str
    roster_sha: str
    runner_code_sha: str
    environment_sha: str
    budget_sha: str
    seed_book_sha: str
    initial_checkpoint_sha: str

    def __post_init__(self) -> None:
        if self.resource_kind not in RESOURCE_KINDS:
            raise ValueError(f"unknown one-shot resource {self.resource_kind!r}")
        _canonical_string(self.campaign_id, "campaign_id")
        for name in (
            "prereg_sha", "authorization_sha", "resource_manifest_sha",
            "materialization_sha", "method_sha", "roster_sha",
            "runner_code_sha", "environment_sha", "budget_sha",
            "seed_book_sha", "initial_checkpoint_sha",
        ):
            require_sha(getattr(self, name), name)

    @property
    def manifest_sha(self) -> str:
        return sha256(self)

    def ledger_path(self, ledger_root: Path | str) -> Path:
        """Return the only legal ledger path for this sealed resource.

        The path deliberately excludes campaign and method identifiers.  A second
        manifest for the same resource bytes therefore replays the first ledger and
        fails its manifest binding instead of receiving a new access opportunity.
        """
        return Path(ledger_root) / (
            f"{self.resource_kind}-{self.resource_manifest_sha}.jsonl"
        )


@dataclass(frozen=True)
class AccessReservationV1:
    resource_kind: str
    campaign_id: str
    manifest_sha: str
    resource_manifest_sha: str
    run_id: str
    reservation_event_sha: str
    checkpoint_sha: str

    def __post_init__(self) -> None:
        if self.resource_kind not in RESOURCE_KINDS:
            raise ValueError("reservation resource kind is invalid")
        for name in ("campaign_id", "run_id"):
            _canonical_string(getattr(self, name), name)
        for name in (
            "manifest_sha", "resource_manifest_sha", "reservation_event_sha",
            "checkpoint_sha",
        ):
            require_sha(getattr(self, name), name)

    @property
    def receipt_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class ResumeBindingV1:
    manifest_sha: str
    campaign_id: str
    run_id: str
    method_sha: str
    roster_sha: str
    runner_code_sha: str
    resource_manifest_sha: str
    materialization_sha: str
    environment_sha: str
    budget_sha: str
    seed_book_sha: str
    checkpoint_sha: str

    def __post_init__(self) -> None:
        for name in ("campaign_id", "run_id"):
            _canonical_string(getattr(self, name), name)
        for name in self.__dataclass_fields__:
            if name not in {"campaign_id", "run_id"}:
                require_sha(getattr(self, name), name)


@dataclass(frozen=True)
class AccessTerminalArtifactV1:
    resource_kind: str
    campaign_id: str
    manifest_sha: str
    reservation_sha: str
    terminal_status: str
    result_sha: str
    terminal_event_sha: str

    def __post_init__(self) -> None:
        if self.resource_kind not in RESOURCE_KINDS:
            raise ValueError("terminal artifact resource kind is invalid")
        if self.terminal_status not in TERMINAL_STATUSES:
            raise ValueError("terminal artifact status is invalid")
        _canonical_string(self.campaign_id, "campaign_id")
        for name in (
            "manifest_sha", "reservation_sha", "result_sha", "terminal_event_sha",
        ):
            require_sha(getattr(self, name), name)

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


class _ExclusiveLock:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":  # pragma: no cover
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._fd)
            raise OneShotAccessError("one-shot resource is already locked") from exc

    def close(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            if os.name != "nt":  # pragma: no cover
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _event_sha(event: Mapping[str, Any]) -> str:
    return sha256({key: value for key, value in event.items() if key != "event_sha"})


T = TypeVar("T")


class OneShotAccessControllerV1:
    """Append-only WAL whose reservation is a loader capability."""

    def __init__(self, ledger_root: Path | str, manifest: OneShotAccessManifestV1) -> None:
        if not isinstance(manifest, OneShotAccessManifestV1):
            raise TypeError("manifest must be OneShotAccessManifestV1")
        self.manifest = manifest
        self.path = manifest.ledger_path(ledger_root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _ExclusiveLock(self.path.with_suffix(self.path.suffix + ".lock"))
        self._fd: int | None = None
        self._closed = False
        self._events: list[dict[str, Any]] = []
        self._tip = hashlib.sha256(
            f"vnext-access-genesis|{manifest.resource_manifest_sha}".encode("utf-8")
        ).hexdigest()
        self._seq = 0
        self._authorized = False
        self._reservation: AccessReservationV1 | None = None
        self._running = False
        self._checkpoint_sha = manifest.initial_checkpoint_sha
        self._interrupted = False
        self._terminal: AccessTerminalArtifactV1 | None = None
        try:
            if self.path.exists() and self.path.stat().st_size:
                self._replay()
            self._fd = os.open(
                str(self.path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600,
            )
            if not self._events:
                self._commit("resource_freeze", {})
                self._fsync_directory()
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "OneShotAccessControllerV1":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._lock.close()

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(copy.deepcopy(self._events))

    @property
    def terminal_artifact(self) -> AccessTerminalArtifactV1 | None:
        return self._terminal

    def authorize(self, authorization_sha: str) -> str:
        require_sha(authorization_sha, "authorization_sha")
        if authorization_sha != self.manifest.authorization_sha:
            raise OneShotAccessError("authorization does not match the frozen manifest")
        if self._authorized:
            raise OneShotAccessError("resource is already authorized")
        if self._terminal is not None:
            raise OneShotAccessError("resource is already terminal")
        return self._commit("authorize", {"authorization_sha": authorization_sha})

    def reserve(self, run_id: str) -> AccessReservationV1:
        _canonical_string(run_id, "run_id")
        if not self._authorized:
            raise OneShotAccessError("resource must be authorized before reservation")
        if self._reservation is not None:
            raise OneShotAccessError("one-shot resource already has a reservation")
        event_sha = self._commit("access_reserved", {
            "run_id": run_id,
            "checkpoint_sha": self.manifest.initial_checkpoint_sha,
        })
        assert self._reservation is not None
        if self._reservation.reservation_event_sha != event_sha:
            raise AssertionError("reservation replay disagrees with committed event")
        return self._reservation

    def pending_reservation(self) -> AccessReservationV1 | None:
        if self._reservation is None or self._terminal is not None:
            return None
        return self._reservation

    def assert_load_allowed(self, receipt: AccessReservationV1) -> None:
        self._validate_receipt(receipt)
        if self._terminal is not None:
            raise OneShotAccessError("terminal reservation cannot load the resource")
        if not self._authorized:
            raise OneShotAccessError("resource is not authorized")

    def load(self, receipt: AccessReservationV1, loader: Callable[[], T]) -> T:
        """Validate the durable reservation before invoking the actual loader."""
        if not callable(loader):
            raise TypeError("loader must be callable")
        self.assert_load_allowed(receipt)
        if not self._running:
            self._commit("running", {"reservation_sha": receipt.receipt_sha})
        return loader()

    def checkpoint(self, receipt: AccessReservationV1, checkpoint_sha: str) -> str:
        self._validate_receipt(receipt)
        require_sha(checkpoint_sha, "checkpoint_sha")
        if self._terminal is not None:
            raise OneShotAccessError("terminal resource cannot checkpoint")
        return self._commit("checkpoint", {
            "reservation_sha": receipt.receipt_sha,
            "checkpoint_sha": checkpoint_sha,
        })

    def interrupt_infrastructure(
        self, receipt: AccessReservationV1, checkpoint_sha: str,
    ) -> str:
        self._validate_receipt(receipt)
        require_sha(checkpoint_sha, "checkpoint_sha")
        if self._terminal is not None:
            raise OneShotAccessError("terminal resource cannot be interrupted")
        return self._commit("infra_interrupted", {
            "reservation_sha": receipt.receipt_sha,
            "checkpoint_sha": checkpoint_sha,
        })

    def expected_resume_binding(self) -> ResumeBindingV1:
        if self._reservation is None:
            raise OneShotAccessError("no reservation exists")
        return ResumeBindingV1(
            manifest_sha=self.manifest.manifest_sha,
            campaign_id=self.manifest.campaign_id,
            run_id=self._reservation.run_id,
            method_sha=self.manifest.method_sha,
            roster_sha=self.manifest.roster_sha,
            runner_code_sha=self.manifest.runner_code_sha,
            resource_manifest_sha=self.manifest.resource_manifest_sha,
            materialization_sha=self.manifest.materialization_sha,
            environment_sha=self.manifest.environment_sha,
            budget_sha=self.manifest.budget_sha,
            seed_book_sha=self.manifest.seed_book_sha,
            checkpoint_sha=self._checkpoint_sha,
        )

    def resume_exact(self, binding: ResumeBindingV1) -> AccessReservationV1:
        if not isinstance(binding, ResumeBindingV1):
            raise TypeError("binding must be ResumeBindingV1")
        if self._terminal is not None or self._reservation is None:
            raise OneShotAccessError("resume requires a pending reservation")
        if binding != self.expected_resume_binding():
            raise OneShotAccessError("resume binding differs from durable state")
        self._commit("resume_exact", {
            "reservation_sha": self._reservation.receipt_sha,
            "binding": asdict(binding),
        })
        return self._reservation

    def record_terminal(
        self, receipt: AccessReservationV1, status: str, result_sha: str,
    ) -> AccessTerminalArtifactV1:
        self._validate_receipt(receipt)
        if status not in TERMINAL_STATUSES:
            raise OneShotAccessError(f"unknown terminal status {status!r}")
        require_sha(result_sha, "result_sha")
        if self._terminal is not None:
            if (
                self._terminal.terminal_status == status
                and self._terminal.result_sha == result_sha
            ):
                return self._terminal
            raise OneShotAccessError("terminal result cannot be overwritten")
        self._commit("terminal", {
            "reservation_sha": receipt.receipt_sha,
            "status": status,
            "result_sha": result_sha,
        })
        assert self._terminal is not None
        return self._terminal

    def _validate_receipt(self, receipt: AccessReservationV1) -> None:
        if not isinstance(receipt, AccessReservationV1):
            raise TypeError("receipt must be AccessReservationV1")
        if self._reservation != receipt:
            raise OneShotAccessError("reservation receipt does not match durable state")

    def _commit(self, kind: str, fields: Mapping[str, Any]) -> str:
        if self._closed or self._fd is None:
            raise OneShotAccessError("access controller is closed")
        event: dict[str, Any] = {
            "schema": ACCESS_SCHEMA,
            "seq": self._seq + 1,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": kind,
            "manifest_sha": self.manifest.manifest_sha,
            "resource_manifest_sha": self.manifest.resource_manifest_sha,
            "campaign_id": self.manifest.campaign_id,
            "prev_event_sha": self._tip,
        }
        if set(event) & set(fields):
            raise OneShotAccessError("event fields conflict with protocol headers")
        event.update(copy.deepcopy(dict(fields)))
        event["event_sha"] = _event_sha(event)
        payload = (canonical_json(event) + "\n").encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(self._fd, view)
            view = view[written:]
        os.fsync(self._fd)
        self._apply(event)
        return str(event["event_sha"])

    def _apply(self, event: Mapping[str, Any]) -> None:
        kind = event["event"]
        if event.get("manifest_sha") != self.manifest.manifest_sha:
            raise OneShotAccessError("ledger is bound to another manifest")
        if event.get("resource_manifest_sha") != self.manifest.resource_manifest_sha:
            raise OneShotAccessError("ledger is bound to another resource")
        if event.get("campaign_id") != self.manifest.campaign_id:
            raise OneShotAccessError("ledger is bound to another campaign")
        if kind == "resource_freeze":
            if self._events:
                raise OneShotAccessError("resource freeze must be the first event")
        elif kind == "authorize":
            if self._authorized or self._reservation is not None:
                raise OneShotAccessError("invalid authorize transition")
            if event.get("authorization_sha") != self.manifest.authorization_sha:
                raise OneShotAccessError("ledger authorization binding is invalid")
            self._authorized = True
        elif kind == "access_reserved":
            if not self._authorized or self._reservation is not None:
                raise OneShotAccessError("invalid reservation transition")
            checkpoint = str(event.get("checkpoint_sha"))
            require_sha(checkpoint, "checkpoint_sha")
            self._reservation = AccessReservationV1(
                resource_kind=self.manifest.resource_kind,
                campaign_id=self.manifest.campaign_id,
                manifest_sha=self.manifest.manifest_sha,
                resource_manifest_sha=self.manifest.resource_manifest_sha,
                run_id=_canonical_string(event.get("run_id"), "run_id"),
                reservation_event_sha=str(event["event_sha"]),
                checkpoint_sha=checkpoint,
            )
            self._checkpoint_sha = checkpoint
        elif kind in {"running", "checkpoint", "infra_interrupted", "resume_exact", "terminal"}:
            if self._reservation is None:
                raise OneShotAccessError(f"{kind} requires a reservation")
            if event.get("reservation_sha") != self._reservation.receipt_sha:
                raise OneShotAccessError(f"{kind} receipt binding is invalid")
            if kind == "running":
                if self._terminal is not None:
                    raise OneShotAccessError("terminal resource cannot run")
                self._running = True
            elif kind == "checkpoint":
                if self._terminal is not None:
                    raise OneShotAccessError("terminal resource cannot checkpoint")
                require_sha(str(event.get("checkpoint_sha")), "checkpoint_sha")
                self._checkpoint_sha = str(event["checkpoint_sha"])
            elif kind == "infra_interrupted":
                if self._terminal is not None:
                    raise OneShotAccessError("terminal resource cannot be interrupted")
                require_sha(str(event.get("checkpoint_sha")), "checkpoint_sha")
                self._checkpoint_sha = str(event["checkpoint_sha"])
                self._interrupted = True
                self._running = False
            elif kind == "resume_exact":
                try:
                    binding = ResumeBindingV1(**dict(event.get("binding") or {}))
                except (TypeError, ValueError) as exc:
                    raise OneShotAccessError("invalid resume binding in ledger") from exc
                if binding != self.expected_resume_binding():
                    raise OneShotAccessError("durable resume binding does not match state")
                self._interrupted = False
                self._running = True
            else:
                status = str(event.get("status"))
                if status not in TERMINAL_STATUSES or self._terminal is not None:
                    raise OneShotAccessError("invalid or repeated terminal result")
                result_sha = str(event.get("result_sha"))
                require_sha(result_sha, "result_sha")
                self._terminal = AccessTerminalArtifactV1(
                    resource_kind=self.manifest.resource_kind,
                    campaign_id=self.manifest.campaign_id,
                    manifest_sha=self.manifest.manifest_sha,
                    reservation_sha=self._reservation.receipt_sha,
                    terminal_status=status,
                    result_sha=result_sha,
                    terminal_event_sha=str(event["event_sha"]),
                )
                self._running = False
        else:
            raise OneShotAccessError(f"unknown ledger event {kind!r}")
        self._seq = int(event["seq"])
        self._tip = str(event["event_sha"])
        self._events.append(copy.deepcopy(dict(event)))

    def _replay(self) -> None:
        try:
            lines = self.path.read_text("utf-8").splitlines()
        except OSError as exc:
            raise OneShotAccessError("one-shot ledger cannot be read") from exc
        for line_number, line in enumerate(lines, 1):
            if not line:
                raise OneShotAccessError("one-shot ledger contains an empty line")
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OneShotAccessError("one-shot ledger contains torn JSON") from exc
            if event.get("schema") != ACCESS_SCHEMA:
                raise OneShotAccessError("one-shot ledger schema mismatch")
            if event.get("seq") != self._seq + 1:
                raise OneShotAccessError("one-shot ledger sequence is invalid")
            if event.get("prev_event_sha") != self._tip:
                raise OneShotAccessError("one-shot ledger hash chain is broken")
            if event.get("event_sha") != _event_sha(event):
                raise OneShotAccessError("one-shot ledger event hash is invalid")
            self._apply(event)

    def _fsync_directory(self) -> None:
        try:
            directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
        except OSError:  # pragma: no cover - unsupported filesystems
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

