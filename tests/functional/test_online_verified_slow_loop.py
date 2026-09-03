from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner
import run_v1_sealed_a5_a3 as sealed

from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import WindowVerification


ORIGIN = 400
ALT_OP = "outlier_mad"
PATCH_ID = "replace-harmful-with-outlier-mad"
SKILL_ID = "verified_replacement"


def _series() -> np.ndarray:
    x = np.arange(1024, dtype=np.float64)
    return np.sin(x / 7.0) + 0.1 * np.sin(x / 3.0) + 5.0


def _verification(op: str) -> WindowVerification:
    result = WindowVerification(
        passed=True,
        checked_windows=1,
        window_modified_flags=(True,),
        window_identity_equivalent_flags=(False,),
    )
    result._program_supply_prepared_values = (
        np.asarray([float(sum(ord(ch) for ch in op))]),
    )
    return result


class _SignedExecutor:
    def __init__(self) -> None:
        self.verify_calls = 0
        self.evaluate_calls: list[tuple[str, int]] = []

    def verify(self, steps, origin):
        self.verify_calls += 1
        return _verification(str(steps[0][0]))

    def evaluate(self, steps, origin):
        op = str(steps[0][0]) if steps else "identity"
        self.evaluate_calls.append((op, int(origin)))
        gain = 0.10 if op == ALT_OP else -0.10
        return SimpleNamespace(
            verification=_verification(op),
            gain=gain,
            per_view_gain=(gain,),
            behavior_point_count=1,
        )


class _CapturingSlow:
    last_no_proposal_reason = None

    def __init__(self, manifest: EditManifest) -> None:
        self.manifest = manifest
        self.calls = 0
        self.seen_card = None
        self.seen_catalog = None

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        self.calls += 1
        self.seen_card = dict(card)
        self.seen_catalog = tuple(dict(item) for item in surface_catalog)
        preflight = kwargs.get("manifest_preflight")
        if preflight is not None:
            preflight(self.manifest)
        return self.manifest


def _manifest(snapshot) -> EditManifest:
    return EditManifest(
        edit_id="verified-slow-add",
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id="verified-natural-first-fault",
        target_surface_id="skill_library.entries/{skill_id}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": SKILL_ID,
            "skill_kind": "capability",
            "revision": 1,
            "body": "runtime binds the frozen program",
            "observable_applicability": {
                "all": [
                    {"feature": "task_kind", "op": "==", "value": "forecast"}
                ]
            },
            "allowed_tools": [ALT_OP],
            "risk_guards": {"single_surface_only": True},
        },
        observable_applicability={
            "all": [
                {"feature": "task_kind", "op": "==", "value": "forecast"}
            ]
        },
        predicted_agent_behavior_change=("retrieve_skill:" + SKILL_ID,),
        predicted_data_effect=("reduce_outlier_tail",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=PATCH_ID,
    )


def _card(_episode):
    return {
        "pattern_id": "verified-natural-first-fault",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
    }


def _options():
    return (
        {
            "patch_id": PATCH_ID,
            "program_steps": [{"op": ALT_OP, "params": {}}],
        },
    )


def _method(snapshot, series, backend):
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series[:ORIGIN], task_kind="forecast"),
    )
    return TTHAMethod(TTHAFastAgent(core), snapshot, ())


def test_verified_first_fault_reaches_one_surface_slow_and_next_fast(tmp_path):
    values = {"s0": _series()}
    series = values["s0"]
    snapshot = runner._h0_snapshot()
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = _method(
        snapshot,
        series,
        sealed.SealedProbeBackend(
            explore=True, operators=("winsorize",), force_pool=True
        ),
    )
    executor = _SignedExecutor()
    slow = _CapturingSlow(_manifest(snapshot))
    request = runner._a5_request(series, values, ORIGIN, "verified-slow-test")
    features = dict(
        extract_public_features(series[:ORIGIN], task_kind="forecast")
    )

    first = run_online_round(
        method,
        executor,
        request,
        values,
        origin=ORIGIN,
        slow_agent=slow,
        controller=controller,
        store=store,
        card_builder=_card,
        round_name="first_fault",
        budget=2,
        allow_slow=True,
        domain="verified-slow-test",
        period=24,
        fast_features=features,
        slow_typed_patch_options=_options(),
        program_supply_verifier=executor,
    )

    assert slow.calls == 1
    assert len(slow.seen_catalog) == 1
    assert slow.seen_catalog[0]["operation"] == "ADD"
    assert [item["patch_id"] for item in slow.seen_card[
        "typed_patch_options"
    ]] == [PATCH_ID]
    assert first._slow_event["stage"] == "pending", first._slow_event
    assert first._slow_event["verified_patch_ids"] == [PATCH_ID]
    assert first.target_support_receipts_used == 2
    assert first.slow_replay_receipts_used == 1
    assert first.program_supply_verifier_requests == 1
    assert first.program_supply_verifier_blocked == 0
    assert first.winner_program == [{"op": ALT_OP, "params": {}}]

    open_delayed(
        first,
        executor,
        delayed_origin=ORIGIN + 48,
        store=store,
    )
    assert first.approved_skill_id == "verified-slow-add"
    assert activate_approved(first, store) is True
    assert any(
        skill.skill_id == SKILL_ID for skill in method._active_snapshot().skills
    )

    replay_method = _method(
        method._active_snapshot(),
        series,
        sealed.SealedProbeBackend(
            explore=False,
            operators=(),
            force_pool=True,
            prefer_skill_in_select=True,
        ),
    )
    replay = run_online_round(
        replay_method,
        executor,
        request,
        values,
        origin=ORIGIN,
        slow_agent=None,
        controller=None,
        store=None,
        card_builder=lambda _episode: {},
        round_name="next_fast",
        budget=1,
        allow_slow=False,
        domain="verified-slow-test",
        period=24,
        fast_features=features,
    )
    trace = replay_method.last_trace
    assert SKILL_ID in tuple(trace.retrieved_skill_ids or ())
    assert "cand_skill_" + SKILL_ID in tuple(trace.candidate_ids or ())
    assert replay.winner_program == [{"op": ALT_OP, "params": {}}]


def test_verified_options_do_nothing_when_slow_is_disabled(tmp_path):
    values = {"s0": _series()}
    series = values["s0"]
    snapshot = runner._h0_snapshot()
    method = _method(
        snapshot,
        series,
        sealed.SealedProbeBackend(
            explore=True, operators=("winsorize",), force_pool=True
        ),
    )
    executor = _SignedExecutor()
    card_calls = 0

    def counted_card(episode):
        nonlocal card_calls
        card_calls += 1
        return _card(episode)

    result = run_online_round(
        method,
        executor,
        runner._a5_request(series, values, ORIGIN, "verified-slow-disabled"),
        values,
        origin=ORIGIN,
        slow_agent=None,
        controller=None,
        store=None,
        card_builder=counted_card,
        budget=2,
        allow_slow=False,
        slow_typed_patch_options=_options(),
        program_supply_verifier=executor,
    )

    assert result._slow_event is None
    assert executor.verify_calls == 0
    assert card_calls == 0
    assert result.target_support_receipts_used == 1
    assert result.pending_patch_id is None
    assert method._pending_update is None


def test_verified_path_budget_one_defers_before_card_verify_or_slow(tmp_path):
    values = {"s0": _series()}
    series = values["s0"]
    snapshot = runner._h0_snapshot()
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = _method(
        snapshot,
        series,
        sealed.SealedProbeBackend(
            explore=True, operators=("winsorize",), force_pool=True
        ),
    )
    executor = _SignedExecutor()
    slow = _CapturingSlow(_manifest(snapshot))
    card_calls = 0

    def counted_card(episode):
        nonlocal card_calls
        card_calls += 1
        return _card(episode)

    result = run_online_round(
        method,
        executor,
        runner._a5_request(series, values, ORIGIN, "verified-budget-one"),
        values,
        origin=ORIGIN,
        slow_agent=slow,
        controller=controller,
        store=store,
        card_builder=counted_card,
        budget=1,
        allow_slow=True,
        slow_typed_patch_options=_options(),
        program_supply_verifier=executor,
    )

    assert result._deferred_slow == "SLOW_UPDATE_DEFERRED_NO_TARGET_BUDGET"
    assert result.target_support_receipts_used == 1
    assert result.program_supply_verifier_requests == 0
    assert executor.verify_calls == 0
    assert card_calls == 0
    assert slow.calls == 0
    assert method._pending_update is None


def test_verified_path_probe_guard_abstains_before_exceeding_cap(tmp_path):
    values = {"s0": _series()}
    series = values["s0"]
    snapshot = runner._h0_snapshot()
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = _method(
        snapshot,
        series,
        sealed.SealedProbeBackend(
            explore=True, operators=("winsorize",), force_pool=True
        ),
    )
    executor = _SignedExecutor()
    slow = _CapturingSlow(_manifest(snapshot))

    result = run_online_round(
        method,
        executor,
        runner._a5_request(series, values, ORIGIN, "verified-probe-cap"),
        values,
        origin=ORIGIN,
        slow_agent=slow,
        controller=controller,
        store=store,
        card_builder=_card,
        budget=2,
        allow_slow=True,
        slow_typed_patch_options=_options(),
        program_supply_verifier=executor,
        program_supply_verifier_budget=0,
    )

    assert result._slow_event["stage"] == (
        "program_supply_verifier_budget_exhausted"
    )
    assert result.program_supply_verifier_requests == 0
    assert result.program_supply_verifier_blocked == 1
    assert executor.verify_calls == 0
    assert slow.calls == 0
    assert result.target_support_receipts_used == 1
    assert method._pending_update is None


def test_verified_path_missing_dependencies_fails_closed_before_verify():
    values = {"s0": _series()}
    series = values["s0"]
    snapshot = runner._h0_snapshot()
    method = _method(
        snapshot,
        series,
        sealed.SealedProbeBackend(
            explore=True, operators=("winsorize",), force_pool=True
        ),
    )
    executor = _SignedExecutor()

    result = run_online_round(
        method,
        executor,
        runner._a5_request(series, values, ORIGIN, "verified-no-deps"),
        values,
        origin=ORIGIN,
        slow_agent=None,
        controller=None,
        store=None,
        card_builder=_card,
        budget=2,
        allow_slow=True,
        slow_typed_patch_options=_options(),
        program_supply_verifier=executor,
    )

    assert result._slow_event["stage"] == "slow_dependencies_unavailable"
    assert sorted(result._slow_event["missing"]) == [
        "controller", "slow_agent", "store"
    ]
    assert executor.verify_calls == 0
    assert result.program_supply_verifier_requests == 0
    assert result.target_support_receipts_used == 1
    assert method._pending_update is None
