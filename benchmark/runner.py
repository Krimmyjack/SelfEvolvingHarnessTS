"""Benchmark phase orchestration and the only Final-Query read gate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import BENCHMARK_VERSION
from .ledger import (
    CampaignEntry,
    CampaignLedger,
    CampaignManifest,
    MethodResultStatus,
    ResumeBinding,
)

FINAL_ROSTER_REQUIRED = frozenset(
    {"raw", "best_fixed", "h_ref", "oracle_transfer", "oracle_insample"}
)


class RunnerGateError(RuntimeError):
    """A benchmark phase was requested before its frozen prerequisites."""


class InfrastructureInterruption(RuntimeError):
    """Explicit evaluator/hardware interruption carrying exact checkpoint bytes."""

    def __init__(self, message: str, *, checkpoint: bytes) -> None:
        super().__init__(message)
        if not isinstance(checkpoint, bytes) or not checkpoint:
            raise ValueError("infrastructure interruption requires non-empty checkpoint bytes")
        self.checkpoint = checkpoint


def _require_sha(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def validate_campaign_roster(entries: Sequence[CampaignEntry]) -> None:
    entries = tuple(entries)
    if not entries or not all(isinstance(entry, CampaignEntry) for entry in entries):
        raise RunnerGateError("campaign roster must contain CampaignEntry values")
    ids = {entry.entry_id for entry in entries}
    missing = FINAL_ROSTER_REQUIRED - ids
    if missing:
        raise RunnerGateError(f"Final roster missing required entries: {sorted(missing)}")
    if len(ids) != len(entries):
        raise RunnerGateError("Final roster contains duplicate entry ids")
    for entry in entries:
        if entry.entry_id not in FINAL_ROSTER_REQUIRED and (
            entry.dry_run_sha is None or entry.confirmation_sha is None
        ):
            raise RunnerGateError(
                f"custom method {entry.entry_id!r} lacks Support-A dry-run or Support-B confirmation"
            )


def _artifact_digest(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, bytes):
        digest.update(value)
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    else:
        try:
            payload = json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            payload = repr(value)
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


class BenchmarkRunner:
    def __init__(
        self,
        *,
        final_store: Any,
        input_manifest_sha: str,
        materialization_sha: str,
        runner_code_sha: str,
        ledger: Any | None = None,
    ) -> None:
        if not hasattr(final_store, "load"):
            raise TypeError("final_store must expose load(entry_id)")
        self.final_store = final_store
        self.ledger = ledger
        self.input_manifest_sha = _require_sha(input_manifest_sha, "input_manifest_sha")
        self.materialization_sha = _require_sha(materialization_sha, "materialization_sha")
        self.runner_code_sha = _require_sha(runner_code_sha, "runner_code_sha")
        self.dev_discrimination_sha: str | None = None
        self.prepare_timeout_s: float | None = None
        self.trainer_timeout_s: float | None = None
        self.campaign_manifest: CampaignManifest | None = None

    def calibrate_timeouts(
        self,
        prepare_durations_s: Sequence[float],
        trainer_durations_s: Sequence[float],
    ) -> tuple[float, float]:
        def calibrated(values: Sequence[float], name: str) -> float:
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all() or (array <= 0).any():
                raise RunnerGateError(f"{name} timeout calibration needs positive finite Dev timings")
            return 2.0 * float(np.quantile(array, 0.95, method="linear"))

        self.prepare_timeout_s = calibrated(prepare_durations_s, "prepare")
        self.trainer_timeout_s = calibrated(trainer_durations_s, "trainer")
        return self.prepare_timeout_s, self.trainer_timeout_s

    def record_dev_discrimination(self, artifact: bytes | Mapping[str, Any]) -> str:
        self.dev_discrimination_sha = _artifact_digest(artifact)
        return self.dev_discrimination_sha

    def freeze_campaign(
        self,
        entries: Sequence[CampaignEntry],
        *,
        campaign_id: str,
        ledger_path: str | Path,
    ) -> CampaignManifest:
        if self.dev_discrimination_sha is None:
            raise RunnerGateError("Dev discrimination report must be frozen before Final")
        _require_sha(self.dev_discrimination_sha, "dev_discrimination_sha")
        if self.prepare_timeout_s is None or self.trainer_timeout_s is None:
            raise RunnerGateError("numeric prepare and trainer timeout values must be frozen")
        entries = tuple(entries)
        validate_campaign_roster(entries)
        manifest = CampaignManifest(
            campaign_id=campaign_id,
            benchmark_version=BENCHMARK_VERSION,
            input_manifest_sha=self.input_manifest_sha,
            materialization_sha=self.materialization_sha,
            runner_code_sha=self.runner_code_sha,
            entries=entries,
            prepare_timeout_s=self.prepare_timeout_s,
            trainer_timeout_s=self.trainer_timeout_s,
        )
        if self.ledger is not None and hasattr(self.ledger, "close"):
            self.ledger.close()
        self.ledger = CampaignLedger(ledger_path, manifest)
        self.campaign_manifest = manifest
        return manifest

    def load_final_for_entry(self, entry_id: str, run_id: str) -> Any:
        if self.ledger is None:
            raise RunnerGateError("Final campaign ledger is not available")
        # The WAL access event must be durable before this method invokes load().
        self.ledger.record_access(entry_id, run_id)
        return self.final_store.load(entry_id)

    def run_final_entry(
        self,
        entry_id: str,
        run_id: str,
        execute: Callable[[Any], Any],
    ) -> MethodResultStatus:
        data = self.load_final_for_entry(entry_id, run_id)
        try:
            result = execute(data)
        except InfrastructureInterruption as exc:
            if self.campaign_manifest is None:
                raise RunnerGateError("infrastructure resume requires a frozen campaign manifest") from exc
            checkpoint_sha = hashlib.sha256(exc.checkpoint).hexdigest()
            entry = self.campaign_manifest.entry(entry_id)
            binding = ResumeBinding(
                campaign_id=self.campaign_manifest.campaign_id,
                run_id=run_id,
                entry_id=entry_id,
                method_code_sha=entry.method_code_sha,
                runner_code_sha=self.runner_code_sha,
                input_manifest_sha=self.input_manifest_sha,
                materialization_sha=self.materialization_sha,
                checkpoint_sha=checkpoint_sha,
            )
            self.ledger.record_result(
                entry_id,
                run_id,
                MethodResultStatus.INFRA_INTERRUPTED,
                checkpoint_sha,
                resume_binding=binding,
            )
            raise
        except TimeoutError as exc:
            digest = _artifact_digest({"type": type(exc).__name__, "message": str(exc)})
            self.ledger.record_result(
                entry_id, run_id, MethodResultStatus.FAILED_TIMEOUT, digest
            )
            return MethodResultStatus.FAILED_TIMEOUT
        except Exception as exc:
            digest = _artifact_digest({"type": type(exc).__name__, "message": str(exc)})
            self.ledger.record_result(entry_id, run_id, MethodResultStatus.INVALID, digest)
            return MethodResultStatus.INVALID
        digest = _artifact_digest(result)
        self.ledger.record_result(entry_id, run_id, MethodResultStatus.COMPLETE, digest)
        return MethodResultStatus.COMPLETE

    def run_final(
        self,
        executors: Mapping[str, Callable[[Any], Any]],
        *,
        run_id_factory: Callable[[str], str],
    ) -> dict[str, MethodResultStatus]:
        if self.campaign_manifest is None or self.ledger is None:
            raise RunnerGateError("Final campaign must be frozen before evaluation")
        if set(executors) != {entry.entry_id for entry in self.campaign_manifest.entries}:
            raise RunnerGateError("Final executors must match the complete frozen roster")
        self.ledger.unseal()
        results: dict[str, MethodResultStatus] = {}
        for entry in sorted(self.campaign_manifest.entries, key=lambda item: item.order):
            results[entry.entry_id] = self.run_final_entry(
                entry.entry_id,
                run_id_factory(entry.entry_id),
                executors[entry.entry_id],
            )
        self.ledger.close_campaign()
        return results

    # Thin phase adapters keep orchestration in one place while allowing the
    # acquisition/materialization workflow to be injected and tested separately.
    def probe(self, operation: Callable[..., Any], *args, **kwargs) -> Any:
        return operation(*args, **kwargs)

    def freeze(self, operation: Callable[..., Any], *args, **kwargs) -> Any:
        return operation(*args, **kwargs)

    def dry_run(self, operation: Callable[..., Any], *args, **kwargs) -> Any:
        return operation(*args, **kwargs)

    def confirm(self, operation: Callable[..., Any], *args, **kwargs) -> Any:
        return operation(*args, **kwargs)

    def run_dev(self, operation: Callable[..., Any], *args, **kwargs) -> Any:
        result = operation(*args, **kwargs)
        self.record_dev_discrimination(result)
        return result


def default_cli_handlers() -> dict[str, Callable[[Any], int]]:
    def unavailable(args: Any) -> int:
        raise RunnerGateError(
            f"CLI phase {args.command!r} requires a configured benchmark workspace"
        )

    return {
        command: unavailable
        for command in (
            "acquire",
            "probe",
            "freeze",
            "dry-run",
            "confirm",
            "dev-eval",
            "campaign-freeze",
            "final-eval",
        )
    }


__all__ = [
    "BenchmarkRunner",
    "FINAL_ROSTER_REQUIRED",
    "InfrastructureInterruption",
    "RunnerGateError",
    "default_cli_handlers",
    "validate_campaign_roster",
]

