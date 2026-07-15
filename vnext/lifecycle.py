"""Receipt-driven vNext research lifecycle coordinator."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from ._canonical import canonical_json, require_sha, sha256
from .access import AccessTerminalArtifactV1

if os.name == "nt":  # pragma: no cover
    import msvcrt
else:  # pragma: no cover
    import fcntl


class LifecycleGateError(RuntimeError):
    """A vNext phase transition violated the frozen protocol."""


INITIAL_STATE: dict[str, Any] = {
    "schema_version": "vnext-lifecycle/3",
    "revision": 0,
    "m0": "blocked",
    "task_g": "sealed",
    "m2": "sealed",
    "h_base_sha": None,
    "h0_lineage_sha": None,
    "h0_method_sha": None,
    "h0_source": None,
    "m3_supplier_control": "sealed",
    "evolution": "sealed",
    "sa_validation": "sealed",
    "method_finalized": "sealed",
    "support_a_dry_run": "sealed",
    "dev": "sealed",
    "support_b": "sealed",
    "final": "sealed",
    "u": "sealed",
    "events": [],
}


class _LifecycleLock:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":  # pragma: no cover
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover
                fcntl.flock(self.fd, fcntl.LOCK_EX)
        except OSError:
            os.close(self.fd)
            raise

    def close(self) -> None:
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        try:
            if os.name != "nt":  # pragma: no cover
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> "_LifecycleLock":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VNextLifecycle:
    """Machine-enforced M0→Final ordering; holdouts require durable receipts."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LifecycleLock(self._lock_path):
            if not self.path.exists():
                self._write(copy.deepcopy(INITIAL_STATE))
            self._state = self._read()

    @property
    def _lock_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".lock")

    @property
    def state(self) -> Mapping[str, Any]:
        with _LifecycleLock(self._lock_path):
            self._state = self._read()
            return copy.deepcopy(self._state)

    @property
    def state_sha(self) -> str:
        return sha256(self.state)

    def _read(self) -> dict[str, Any]:
        try:
            state = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleGateError("lifecycle state is missing or torn") from exc
        if state.get("schema_version") != "vnext-lifecycle/3":
            raise LifecycleGateError("lifecycle schema mismatch")
        events = state.get("events")
        if not isinstance(events, list):
            raise LifecycleGateError("lifecycle events are invalid")
        tip = sha256("vnext-lifecycle-genesis")
        for index, event in enumerate(events, 1):
            if event.get("seq") != index or event.get("prev_event_sha") != tip:
                raise LifecycleGateError("lifecycle event chain is invalid")
            expected = sha256({key: value for key, value in event.items() if key != "event_sha"})
            if event.get("event_sha") != expected:
                raise LifecycleGateError("lifecycle event hash is invalid")
            tip = expected
        if state.get("revision") != len(events):
            raise LifecycleGateError("lifecycle revision disagrees with event chain")
        return state

    def _write(self, state: Mapping[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        fd = os.open(str(temporary), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        try:
            payload = (canonical_json(state) + "\n").encode("utf-8")
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, self.path)
        try:
            directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:  # pragma: no cover
            pass

    def _transition(
        self,
        kind: str,
        payload: Mapping[str, Any],
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        with _LifecycleLock(self._lock_path):
            state = self._read()
            mutate(state)
            prior = state["events"][-1]["event_sha"] if state["events"] else sha256(
                "vnext-lifecycle-genesis"
            )
            event = {
                "seq": len(state["events"]) + 1,
                "event": kind,
                "prev_event_sha": prior,
                **copy.deepcopy(dict(payload)),
            }
            event["event_sha"] = sha256(event)
            state["events"].append(event)
            state["revision"] = len(state["events"])
            self._write(state)
            self._state = state

    def record_m0_verdict(self, *, verdict_sha: str, passed: bool) -> None:
        require_sha(verdict_sha, "verdict_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["m0"] != "blocked":
                raise LifecycleGateError("M0 verdict is already recorded")
            state["m0"] = "passed" if passed else "failed_protocol_erratum_required"
            if passed:
                state["task_g"] = "authorized"

        self._transition("m0_verdict", {"verdict_sha": verdict_sha, "passed": passed}, mutate)

    def start_task_g(self, *, prereg_sha: str) -> None:
        require_sha(prereg_sha, "prereg_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["m0"] != "passed" or state["task_g"] != "authorized":
                raise LifecycleGateError("Task G requires M0_PASS and one authorization")
            state["task_g"] = "running"

        self._transition("task_g_start", {"prereg_sha": prereg_sha}, mutate)

    def record_task_g_terminal(self, *, result_sha: str, passed: bool) -> None:
        require_sha(result_sha, "result_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["task_g"] != "running":
                raise LifecycleGateError("Task G is not running")
            state["task_g"] = "passed" if passed else "capability_failed_terminal"
            if passed:
                state["m2"] = "authorized"

        self._transition("task_g_terminal", {"result_sha": result_sha, "passed": passed}, mutate)

    def record_m2_hbase(self, *, m2_result_sha: str, h_base_sha: str) -> None:
        require_sha(m2_result_sha, "m2_result_sha")
        require_sha(h_base_sha, "h_base_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["task_g"] != "passed" or state["m2"] != "authorized":
                raise LifecycleGateError("H_base requires Task G and M2 authorization")
            state["m2"] = "passed"
            state["h_base_sha"] = h_base_sha

        self._transition("m2_hbase", {
            "m2_result_sha": m2_result_sha, "h_base_sha": h_base_sha,
        }, mutate)

    def record_h0(self, *, lineage_sha: str, h0_method_sha: str) -> None:
        require_sha(lineage_sha, "lineage_sha")
        require_sha(h0_method_sha, "h0_method_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["m2"] != "passed" or state["h0_method_sha"] is not None:
                raise LifecycleGateError("formal Init-derived H0 requires M2 and is unique")
            state["h0_lineage_sha"] = lineage_sha
            state["h0_method_sha"] = h0_method_sha
            state["h0_source"] = "support_a_discovery_init_only"
            state["m3_supplier_control"] = "authorized"

        self._transition("h0_frozen", {
            "lineage_sha": lineage_sha, "h0_method_sha": h0_method_sha,
        }, mutate)

    def record_m3_supplier_control(
        self, *, result_sha: str, supplier_policy_sha: str,
    ) -> None:
        """Freeze runtime supply after H0; this phase cannot redefine H0."""
        require_sha(result_sha, "result_sha")
        require_sha(supplier_policy_sha, "supplier_policy_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["h0_source"] != "support_a_discovery_init_only":
                raise LifecycleGateError("M3 supplier control requires formal Init-derived H0")
            if state["m3_supplier_control"] != "authorized":
                raise LifecycleGateError("M3 supplier control is already closed")
            state["m3_supplier_control"] = "frozen"
            state["runtime_supplier_policy_sha"] = supplier_policy_sha
            state["evolution"] = "authorized"

        self._transition("m3_supplier_control", {
            "result_sha": result_sha, "supplier_policy_sha": supplier_policy_sha,
            "h0_unchanged": True,
        }, mutate)

    def precommit_evolution_candidate(self, *, candidate_sha: str, precommit_sha: str) -> None:
        require_sha(candidate_sha, "candidate_sha")
        require_sha(precommit_sha, "precommit_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["evolution"] != "authorized":
                raise LifecycleGateError("evolution candidate is not authorized")
            state["evolution"] = "candidate_precommitted"

        self._transition("evolution_precommit", {
            "candidate_sha": candidate_sha, "precommit_sha": precommit_sha,
        }, mutate)

    def skip_sa_validation_no_candidate(self, *, discovery_result_sha: str) -> None:
        require_sha(discovery_result_sha, "discovery_result_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["evolution"] != "authorized" or state["h0_method_sha"] is None:
                raise LifecycleGateError("SA-V abstention requires a frozen H0")
            state["evolution"] = "no_eligible_candidate"
            state["sa_validation"] = "skipped_no_eligible_candidate"

        self._transition("sa_validation_skipped", {
            "discovery_result_sha": discovery_result_sha,
        }, mutate)

    def record_sa_validation_terminal(
        self,
        terminal: AccessTerminalArtifactV1,
        *,
        promoted: bool,
    ) -> None:
        if not isinstance(terminal, AccessTerminalArtifactV1):
            raise TypeError("terminal must be AccessTerminalArtifactV1")
        if terminal.resource_kind != "sa_validation":
            raise LifecycleGateError("SA-V requires an SA-V access receipt")
        if promoted and terminal.terminal_status != "passed":
            raise LifecycleGateError("promotion requires a passing terminal receipt")

        def mutate(state: dict[str, Any]) -> None:
            if state["evolution"] != "candidate_precommitted":
                raise LifecycleGateError("SA-V requires one precommitted candidate")
            if state["sa_validation"] != "sealed":
                raise LifecycleGateError("SA-V terminal was already recorded")
            state["sa_validation"] = (
                "closed_promoted" if promoted else "closed_fallback_h0"
            )
            state["evolution"] = "closed"

        self._transition("sa_validation_terminal", {
            "terminal_artifact_sha": terminal.artifact_sha, "promoted": promoted,
        }, mutate)

    def finalize_method(self, *, method_sha: str, roster_sha: str, budget_sha: str) -> None:
        for name, value in (
            ("method_sha", method_sha), ("roster_sha", roster_sha), ("budget_sha", budget_sha),
        ):
            require_sha(value, name)

        def mutate(state: dict[str, Any]) -> None:
            allowed = {
                "closed_promoted", "closed_fallback_h0", "skipped_no_eligible_candidate",
            }
            if state["sa_validation"] not in allowed:
                raise LifecycleGateError("method freeze requires a closed SA-V decision")
            if state["method_finalized"] != "sealed":
                raise LifecycleGateError("method is already finalized")
            state["method_finalized"] = "frozen"
            state["frozen_method_sha"] = method_sha
            state["frozen_roster_sha"] = roster_sha
            state["frozen_budget_sha"] = budget_sha

        self._transition("method_finalized", {
            "method_sha": method_sha, "roster_sha": roster_sha, "budget_sha": budget_sha,
        }, mutate)

    def record_support_a_dry_run(self, *, dry_run_sha: str) -> None:
        require_sha(dry_run_sha, "dry_run_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["method_finalized"] != "frozen":
                raise LifecycleGateError("dry-run requires a frozen method")
            if state["support_a_dry_run"] != "sealed":
                raise LifecycleGateError("dry-run is already recorded")
            state["support_a_dry_run"] = "passed"

        self._transition("support_a_dry_run", {"dry_run_sha": dry_run_sha}, mutate)

    def record_dev_readonly(self, *, report_sha: str) -> None:
        require_sha(report_sha, "report_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["support_a_dry_run"] != "passed":
                raise LifecycleGateError("Dev requires the frozen Support-A dry-run")
            if state["dev"] != "sealed":
                raise LifecycleGateError("Dev is read-only and cannot become feedback")
            state["dev"] = "reported_readonly"
            state["support_b"] = "authorized"

        self._transition("dev_readonly", {"report_sha": report_sha}, mutate)

    def record_support_b_terminal(
        self,
        terminal: AccessTerminalArtifactV1,
        *,
        passed: bool,
    ) -> None:
        if not isinstance(terminal, AccessTerminalArtifactV1):
            raise TypeError("terminal must be AccessTerminalArtifactV1")
        if terminal.resource_kind != "support_b":
            raise LifecycleGateError("Support-B requires a Support-B access receipt")
        if passed and terminal.terminal_status != "passed":
            raise LifecycleGateError("Support-B pass requires a passing receipt")

        def mutate(state: dict[str, Any]) -> None:
            if state["dev"] != "reported_readonly" or state["support_b"] != "authorized":
                raise LifecycleGateError("Support-B follows Dev unconditionally exactly once")
            state["support_b"] = "passed" if passed else "failed_terminal"

        self._transition("support_b_terminal", {
            "terminal_artifact_sha": terminal.artifact_sha, "passed": passed,
        }, mutate)

    def authorize_final(self, *, campaign_manifest_sha: str, authorization_sha: str) -> None:
        require_sha(campaign_manifest_sha, "campaign_manifest_sha")
        require_sha(authorization_sha, "authorization_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["support_b"] != "passed" or state["final"] != "sealed":
                raise LifecycleGateError("Final requires passing Support-B and separate authorization")
            state["final"] = "authorized"

        self._transition("final_authorized", {
            "campaign_manifest_sha": campaign_manifest_sha,
            "authorization_sha": authorization_sha,
        }, mutate)

    def close_final(self, *, campaign_close_receipt_sha: str) -> None:
        require_sha(campaign_close_receipt_sha, "campaign_close_receipt_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["final"] != "authorized":
                raise LifecycleGateError("Final must be authorized before close")
            state["final"] = "closed_terminal"

        self._transition("final_closed", {
            "campaign_close_receipt_sha": campaign_close_receipt_sha,
        }, mutate)

    def record_u_directional(self, *, report_sha: str) -> None:
        require_sha(report_sha, "report_sha")

        def mutate(state: dict[str, Any]) -> None:
            if state["final"] != "closed_terminal" or state["u"] != "sealed":
                raise LifecycleGateError("U is directional and runs only after Final close")
            state["u"] = "reported_directional"

        self._transition("u_directional", {"report_sha": report_sha}, mutate)
