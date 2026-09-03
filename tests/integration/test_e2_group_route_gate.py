"""E-2 tests: group path only constructs an ADD manifest on a routed ADD Surface."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
    ProgramSupplyDecision,
    build_single_surface_catalog,
)

ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods/ttha/harness/h0"


@pytest.fixture
def store(tmp_path):
    return SnapshotStore(tmp_path / "store")


@pytest.fixture
def h0_materialized(store):
    return store.materialize(compile_snapshot(H0_ROOT, verify_lock=False))


@pytest.fixture
def controller(store):
    return EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )


def _decision(cause="SKILL_LIBRARY_GAP"):
    return ProgramSupplyDecision(
        "e2-case",
        cause,
        "EDITABLE_M0",
        ("skill_library.entries/{skill_id}",),
    )


def _group():
    ep = SimpleNamespace(
        episode_id="ep-1",
        context_summary={"support_origin": 400},
        support_response={"gain": -0.1},
    )
    return {"workflow": "winsorize", "sign": "NEGATIVE", "episodes": [ep]}


def _card(group, capsule):
    return {
        "pattern_id": "pattern-a1b2c3d4e5f6",
        "observable_signature": {"task_kind": "forecast"},
        "typed_patch_options": [
            {"patch_id": "p1",
             "program_steps": [{"op": "winsorize", "params": {}}]},
            {"patch_id": "p2",
             "program_steps": [{"op": "outlier_mad", "params": {}}]},
        ],
    }


class _Slow:
    def __init__(self, manifest=None):
        self.manifest = manifest
        self.calls = 0
        self.last_no_proposal_reason = None

    def propose_edit(self, *args, **kwargs):
        self.calls += 1
        return self.manifest


def test_group_feedback_requires_routed_decision_and_catalog_in_signature():
    sig = inspect.signature(TTHAMethod.handle_group_feedback)
    assert "route_decision" in sig.parameters
    assert "confirmed_cause" not in sig.parameters
    assert sig.parameters["surface_catalog"].default is inspect.Parameter.empty


def test_wrong_route_is_rejected_before_slow_or_manifest(
    h0_materialized, controller, store
):
    method = TTHAMethod(object(), h0_materialized.snapshot)
    slow = _Slow()
    catalog = build_single_surface_catalog(
        decision=_decision(),
        parent=h0_materialized,
        controller=controller,
    )
    wrong = _decision(cause="SKILL_CONTENT_GAP")
    ev = method.handle_group_feedback(
        _group(),
        {"workflow": "winsorize", "n_episodes": 1},
        slow_agent=slow,
        controller=controller,
        store=store,
        surface_catalog=catalog,
        route_decision=wrong,
        card_builder=_card,
        evaluator_group=lambda steps, ep: SimpleNamespace(
            gain=0.02, verification=SimpleNamespace(passed=True)
        ),
    )
    assert ev["stage"] == "route_not_add_only"
    assert "cause_not_skill_library_gap" in ev["reason"]
    assert slow.calls == 0
    assert method._pending_update is None


def test_valid_add_route_reaches_slow_and_records_choice_flags(
    h0_materialized, controller, store
):
    method = TTHAMethod(object(), h0_materialized.snapshot)
    slow = _Slow()
    catalog = build_single_surface_catalog(
        decision=_decision(),
        parent=h0_materialized,
        controller=controller,
    )
    ev = method.handle_group_feedback(
        _group(),
        {"workflow": "winsorize", "n_episodes": 1},
        slow_agent=slow,
        controller=controller,
        store=store,
        surface_catalog=catalog,
        route_decision=_decision(),
        card_builder=_card,
        evaluator_group=lambda steps, ep: SimpleNamespace(
            gain=0.02, verification=SimpleNamespace(passed=True)
        ),
        verified_choice_offered=True,
    )
    assert ev["stage"] == "abstained_by_agent"
    assert ev["route"]["cause_code"] == "SKILL_LIBRARY_GAP"
    assert ev["typed_option_count"] == 2
    assert ev["choice_offered"] is True
    assert "no_choice_offered" not in ev
    assert slow.calls == 1


def test_two_behaviorally_identical_options_are_not_choice_when_evidence_says_so(
    h0_materialized, controller, store
):
    method = TTHAMethod(object(), h0_materialized.snapshot)
    slow = _Slow()
    catalog = build_single_surface_catalog(
        decision=_decision(),
        parent=h0_materialized,
        controller=controller,
    )
    ev = method.handle_group_feedback(
        _group(),
        {"workflow": "winsorize", "n_episodes": 1},
        slow_agent=slow,
        controller=controller,
        store=store,
        surface_catalog=catalog,
        route_decision=_decision(),
        card_builder=_card,
        evaluator_group=lambda steps, ep: SimpleNamespace(
            gain=0.02, verification=SimpleNamespace(passed=True)
        ),
        verified_choice_offered=False,
    )
    assert ev["stage"] == "abstained_by_agent"
    assert ev["typed_option_count"] == 2
    assert ev["choice_offered"] is False
    assert ev["no_choice_offered"] is True
    assert ev["verified_choice_offered"] is False


def test_single_typed_option_is_recorded_as_no_choice_offered(
    h0_materialized, controller, store
):
    method = TTHAMethod(object(), h0_materialized.snapshot)
    slow = _Slow()

    def one_option_card(group, capsule):
        card = _card(group, capsule)
        card["typed_patch_options"] = card["typed_patch_options"][:1]
        return card

    catalog = build_single_surface_catalog(
        decision=_decision(),
        parent=h0_materialized,
        controller=controller,
    )
    ev = method.handle_group_feedback(
        _group(),
        {"workflow": "winsorize", "n_episodes": 1},
        slow_agent=slow,
        controller=controller,
        store=store,
        surface_catalog=catalog,
        route_decision=_decision(),
        card_builder=one_option_card,
        evaluator_group=lambda steps, ep: SimpleNamespace(
            gain=0.02, verification=SimpleNamespace(passed=True)
        ),
    )
    assert ev["stage"] == "abstained_by_agent"
    assert ev["typed_option_count"] == 1
    assert ev["choice_offered"] is False
    assert ev["no_choice_offered"] is True
