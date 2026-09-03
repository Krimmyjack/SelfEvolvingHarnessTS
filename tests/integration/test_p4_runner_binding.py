"""P4 exact program binding tests.

The binding unit is ``(patch_id, exact ordered program_steps)``.  The Card is
built once, E-1 is computed from that same Card, and Slow sees only the exact
verified steps.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.p4_runner import (
    bind_verified_program_options,
    run_p4_group_update,
)
from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
    ProgramSupplyDecision,
    ProgramSupplyFacts,
    ProgramSupplyVerification,
    VerifiedProgramAlternative,
    VerifiedProgramSupplyAssessment,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import WindowVerification
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

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


def _trace():
    return DecisionTrace(
        case_id="p4-case",
        public_observation_ids=(),
        inspected_regions=(),
        tool_calls=(),
        retrieved_skill_ids=(),
        retrieved_memory_ids=(),
        applicability_matches=(),
        candidate_ids=("identity",),
        candidate_program_shas=(None,),
        chosen_candidate_id="identity",
        compilation_status="OK",
        execution_status="OK",
        modified_indices=(),
        verification_actions=(),
        effect_equivalent_to_identity=True,
    )


def _view():
    return SimpleNamespace(skills=())


def _group():
    ep = SimpleNamespace(
        episode_id="ep-1",
        context_summary={"support_origin": 400},
        support_response={"gain": -0.1},
    )
    return {"workflow": "winsorize", "sign": "NEGATIVE", "episodes": [ep]}


def _capsule():
    return {"workflow": "winsorize", "n_episodes": 1, "sign": "NEGATIVE"}


def _card(group, capsule):
    return {
        "pattern_id": "pattern-a1b2c3d4e5f6",
        "observable_signature": {"task_kind": "forecast"},
        "typed_patch_options": [
            {"patch_id": "p1",
             "program_steps": [{"op": "winsorize", "params": {}}]},
            {"patch_id": "p2",
             "program_steps": [{"op": "outlier_mad", "params": {}}]},
            {"patch_id": "unverified",
             "program_steps": [{"op": "hampel_filter", "params": {}}]},
        ],
    }


def _verification(*, passed=True, checked=1, modified=1, identity=0,
                  prepared=((1.0,),)):
    result = WindowVerification(
        passed=passed,
        checked_windows=checked,
        window_modified_flags=(True,) * modified
        + (False,) * (checked - modified),
        window_identity_equivalent_flags=(True,) * identity
        + (False,) * (checked - identity),
    )
    result._program_supply_prepared_values = tuple(
        np.asarray(values, dtype=np.float64) for values in prepared
    )
    return result


class _SequenceVerifier:
    def __init__(self, results):
        self.results = list(results)
        self.evaluate_calls = 0

    def verify(self, steps, origin):
        if not self.results:
            raise AssertionError("more verify calls than expected")
        return self.results.pop(0)

    def evaluate(self, steps, origin):
        self.evaluate_calls += 1
        raise AssertionError("P4 binding must not call evaluate()")


def _assessment(*, alternatives):
    return VerifiedProgramSupplyAssessment(
        facts=ProgramSupplyFacts(
            case_id="p4-case",
            expressibility_status="PROVEN_EXPRESSIBLE",
            expressibility_cause=None,
            capability_skill_exists=False,
            skill_retrieved=False,
            constrained_proposal_succeeds=None,
        ),
        verification=ProgramSupplyVerification(
            alternatives=tuple(alternatives),
            choice_offered=False,
        ),
        decision=ProgramSupplyDecision(
            "p4-case",
            "SKILL_LIBRARY_GAP",
            "EDITABLE_M0",
            ("skill_library.entries/{skill_id}",),
        ),
    )


def _alternative(patch_id, op):
    return VerifiedProgramAlternative(
        patch_id=patch_id,
        steps=((op, {}),),
        verification=_verification(),
    )


class _CapturingSlow:
    last_no_proposal_reason = None

    def __init__(self, manifest=None):
        self.manifest = manifest
        self.calls = 0
        self.seen_card = None

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        self.calls += 1
        self.seen_card = dict(card)
        return self.manifest


def _run_with_verifier(results, *, h0_materialized, controller, store,
                       slow=None, manifest=None):
    method = TTHAMethod(object(), h0_materialized.snapshot)
    executor = _SequenceVerifier(results)
    slow_agent = slow if slow is not None else _CapturingSlow(manifest)
    ev = run_p4_group_update(
        method=method,
        group=_group(),
        capsule=_capsule(),
        trace=_trace(),
        episode=object(),
        view=_view(),
        executor=executor,
        origin=400,
        slow_agent=slow_agent,
        controller=controller,
        store=store,
        card_builder=_card,
        evaluator_group=lambda steps, ep: SimpleNamespace(
            gain=0.02, verification=SimpleNamespace(passed=True)
        ),
    )
    return executor, slow_agent, ev


def test_runner_builds_card_once_and_slow_sees_exact_verified_programs(
    h0_materialized, controller, store
):
    executor, slow, ev = _run_with_verifier(
        [
            _verification(prepared=((1.0,),)),
            _verification(prepared=((2.0,),)),
            _verification(passed=True, checked=0, modified=0),
        ],
        h0_materialized=h0_materialized,
        controller=controller,
        store=store,
    )
    assert ev["stage"] == "abstained_by_agent"
    assert [option["patch_id"] for option in slow.seen_card[
        "typed_patch_options"
    ]] == ["p1", "p2"]
    assert slow.seen_card["typed_patch_options"][0]["program_steps"] == [
        {"op": "winsorize", "params": {}}
    ]
    assert slow.seen_card["typed_patch_options"][1]["program_steps"] == [
        {"op": "outlier_mad", "params": {}}
    ]
    assert ev["choice_offered"] is True
    assert ev["verified_patch_ids"] == ["p1", "p2"]
    assert executor.evaluate_calls == 0


def test_same_id_different_steps_fails_before_slow():
    card = {
        "typed_patch_options": [
            {"patch_id": "p2",
             "program_steps": [{"op": "outlier_mad", "params": {}}]},
        ]
    }
    assessment = _assessment(
        alternatives=[_alternative("p2", "winsorize")]
    )
    filtered, ids, error = bind_verified_program_options(card, assessment)
    assert filtered is None
    assert ids == ()
    assert error["stage"] == "program_binding_mismatch"
    assert error["patch_id"] == "p2"
    assert error["verified_program_steps"] == [
        {"op": "winsorize", "params": {}}
    ]
    assert error["card_program_steps"] == [
        {"op": "outlier_mad", "params": {}}
    ]


def test_duplicate_patch_id_is_rejected():
    card = {
        "typed_patch_options": [
            {"patch_id": "p1",
             "program_steps": [{"op": "winsorize", "params": {}}]},
            {"patch_id": "p1",
             "program_steps": [{"op": "winsorize", "params": {}}]},
        ]
    }
    assessment = _assessment(
        alternatives=[_alternative("p1", "winsorize")]
    )
    _, _, error = bind_verified_program_options(card, assessment)
    assert error["stage"] == "duplicate_card_patch_id"
    assert error["patch_id"] == "p1"


def test_runner_rejects_slow_selection_outside_verified_ids(
    h0_materialized, controller, store
):
    manifest = EditManifest(
        edit_id="p4-unverified",
        base_harness_sha=h0_materialized.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id="skill_library.entries/{skill_id}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": "p4_unverified_skill",
            "skill_kind": "capability",
            "revision": 1,
            "body": "ignored",
            "observable_applicability": {"const": True},
            "allowed_tools": ["winsorize"],
            "risk_guards": {},
        },
        observable_applicability={"const": True},
        predicted_agent_behavior_change=("retrieve_skill:p4_unverified_skill",),
        predicted_data_effect=("local_improvement",),
        falsification_condition=("no_improvement",),
        patch_id="unverified",
    )
    _executor, _slow, ev = _run_with_verifier(
        [
            _verification(prepared=((1.0,),)),
            _verification(prepared=((2.0,),)),
            _verification(passed=True, checked=0, modified=0),
        ],
        h0_materialized=h0_materialized,
        controller=controller,
        store=store,
        manifest=manifest,
    )
    assert ev["stage"] == "verified_patch_binding_failed"
    assert ev["patch_id"] == "unverified"
    assert ev["verified_patch_ids"] == ["p1", "p2"]


def test_runner_abstains_when_no_verified_alternatives(
    h0_materialized, controller, store
):
    slow = _CapturingSlow()
    _executor, slow, ev = _run_with_verifier(
        [
            _verification(passed=True, checked=0, modified=0),
            _verification(passed=True, checked=0, modified=0),
            _verification(passed=True, checked=0, modified=0),
        ],
        h0_materialized=h0_materialized,
        controller=controller,
        store=store,
        slow=slow,
    )
    assert ev["stage"] == "no_verified_alternatives"
    assert slow.calls == 0


def test_runner_choice_uses_distinct_behavior_without_candidate_hashes(
    h0_materialized, controller, store
):
    _executor, _slow, ev = _run_with_verifier(
        [
            _verification(prepared=((1.0,),)),
            _verification(prepared=((2.0,),)),
            _verification(passed=True, checked=0, modified=0),
        ],
        h0_materialized=h0_materialized,
        controller=controller,
        store=store,
    )
    assert ev["stage"] == "abstained_by_agent"
    assert ev["choice_offered"] is True
    assert ev.get("no_choice_offered") is not True
    assert ev["verified_choice_offered"] is True
