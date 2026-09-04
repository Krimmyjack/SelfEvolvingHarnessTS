"""E-1 tests: verifier-only PROVEN_EXPRESSIBLE earning rules."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from SelfEvolvingHarnessTS.contracts.harness import SkillKind
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
    ProgramSupplyDecision,
    ProgramSupplyFacts,
    ProgramSupplyVerification,
    VerifiedProgramSupplyAssessment,
    retrieved_relevant_capability_skill_ids,
    route_online_program_supply_fault,
    route_verified_program_supply_fault,
    verify_program_supply_alternatives,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import WindowVerification
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

H0_ROOT = __import__("pathlib").Path(
    "methods/ttha/harness/h0"
).resolve()


class _FakeExecutor:
    def __init__(self, results):
        self.results = list(results)
        self.evaluate_calls = 0

    def verify(self, steps, origin):
        return self.results.pop(0)

    def evaluate(self, steps, origin):
        self.evaluate_calls += 1
        raise AssertionError("E-1 must never call evaluate()")


def _verification(*, checked=1, passed=True, modified=1, identity=0,
                  prepared=((1.0,),)):
    result = WindowVerification(
        passed=passed,
        checked_windows=checked,
        window_modified_flags=(True,) * modified + (False,) * (checked - modified),
        window_identity_equivalent_flags=(True,) * identity
        + (False,) * (checked - identity),
    )
    result._program_supply_prepared_values = tuple(
        np.asarray(values, dtype=np.float64) for values in prepared
    )
    return result


def _options():
    return [
        {"patch_id": "alt-a",
         "program_steps": [{"op": "winsorize", "params": {}}]},
        {"patch_id": "alt-b",
         "program_steps": [{"op": "outlier_mad", "params": {}}]},
    ]


def _trace():
    return DecisionTrace(
        case_id="e1-case",
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


def _view(skills=()):
    return SimpleNamespace(skills=tuple(skills))


def test_zero_training_windows_do_not_earn_proven_expressible():
    executor = _FakeExecutor([
        _verification(checked=0, modified=0),
    ])
    result = verify_program_supply_alternatives(
        executor=executor,
        typed_patch_options=[_options()[0]],
        origin=400,
    )
    assert result.alternatives == ()


def test_identity_equivalent_program_does_not_earn_proven_expressible():
    executor = _FakeExecutor([
        _verification(checked=2, modified=0, identity=2),
    ])
    result = verify_program_supply_alternatives(
        executor=executor,
        typed_patch_options=[_options()[0]],
        origin=400,
    )
    assert result.alternatives == ()


def test_verified_effect_earns_choice_when_programs_differ():
    executor = _FakeExecutor([
        _verification(prepared=((1.0,),)),
        _verification(prepared=((2.0,),)),
    ])
    result = verify_program_supply_alternatives(
        executor=executor,
        typed_patch_options=_options(),
        origin=400,
    )
    assert len(result.alternatives) == 2
    assert result.choice_offered is True
    assert result.behavior_distinct_pairs == (("alt-a", "alt-b"),)
    assert executor.evaluate_calls == 0


def test_duplicate_programs_under_two_ids_are_not_a_choice():
    executor = _FakeExecutor([
        _verification(prepared=((1.0,),)),
        _verification(prepared=((1.0,),)),
    ])
    duplicate_options = [
        _options()[0],
        {
            "patch_id": "alt-a-alias",
            "program_steps": [{"op": "winsorize", "params": {}}],
        },
    ]
    result = verify_program_supply_alternatives(
        executor=executor,
        typed_patch_options=duplicate_options,
        origin=400,
    )
    assert len(result.alternatives) == 2
    assert result.choice_offered is False


def test_patch_target_intersection_never_uses_unrelated_retrieved_skill():
    assessment = VerifiedProgramSupplyAssessment(
        facts=ProgramSupplyFacts(
            case_id="e1-case",
            expressibility_status="PROVEN_EXPRESSIBLE",
            expressibility_cause=None,
            capability_skill_exists=True,
            skill_retrieved=True,
            constrained_proposal_succeeds=False,
        ),
        verification=ProgramSupplyVerification(),
        decision=ProgramSupplyDecision(
            "e1-case",
            "SKILL_CONTENT_GAP",
            "EDITABLE_M0",
            ("skill_library.entries/{skill_id}.body",),
        ),
        relevant_capability_skill_ids=("related_a", "related_b"),
    )
    one = replace(
        _trace(), retrieved_skill_ids=("unrelated", "related_b")
    )
    assert retrieved_relevant_capability_skill_ids(
        assessment, one
    ) == ("related_b",)

    ambiguous = replace(
        _trace(), retrieved_skill_ids=("related_a", "related_b")
    )
    assert retrieved_relevant_capability_skill_ids(
        assessment, ambiguous
    ) == ("related_a", "related_b")


def test_verified_route_flips_to_skill_library_gap_on_empty_view():
    snapshot = compile_snapshot(H0_ROOT, verify_lock=False)
    view = resolve_harness_view(
        snapshot, {"task_kind": "forecast"}, role="fast"
    )
    executor = _FakeExecutor([
        _verification(prepared=((1.0,),)),
        _verification(prepared=((2.0,),)),
    ])
    assessment = route_verified_program_supply_fault(
        trace=_trace(),
        episode=object(),
        view=view,
        executor=executor,
        typed_patch_options=_options(),
        origin=400,
    )
    assert assessment.facts == ProgramSupplyFacts(
        case_id="e1-case",
        expressibility_status="PROVEN_EXPRESSIBLE",
        expressibility_cause=None,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    )
    assert assessment.decision.cause_code == "SKILL_LIBRARY_GAP"
    assert assessment.verification.choice_offered is True

    # Safety baseline: without verifier evidence the online route stays Unknown.
    assert route_online_program_supply_fault(
        _trace(), object(), view
    ).cause_code == "EXPRESSIBILITY_UNKNOWN"


def test_program_aware_capability_check_ignores_unrelated_skill():
    from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
        _relevant_capability_skill_ids,
    )

    verification = verify_program_supply_alternatives(
        executor=_FakeExecutor([
            _verification(prepared=((1.0,),)),
            _verification(prepared=((2.0,),)),
        ]),
        typed_patch_options=_options(),
        origin=400,
    )
    unrelated = SimpleNamespace(
        skill_id="impute_skill",
        skill_kind=SkillKind.CAPABILITY,
        body="Frozen program steps: [{\"op\": \"impute_linear\", "
             "\"params\": {}}]",
    )
    view = _view((unrelated,))
    executor = _FakeExecutor([
        _verification(prepared=((3.0,),)),
    ])
    assert _relevant_capability_skill_ids(
        view=view, executor=executor, origin=400, verification=verification
    ) == ()


def test_same_program_family_is_relevant_without_candidate_hashes():
    from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
        _relevant_capability_skill_ids,
    )

    verification = verify_program_supply_alternatives(
        executor=_FakeExecutor([
            _verification(),
        ]),
        typed_patch_options=[_options()[0]],
        origin=400,
    )
    skill = SimpleNamespace(
        skill_id="winsorize_skill",
        skill_kind=SkillKind.CAPABILITY,
        body="Frozen program steps: [{\"op\": \"winsorize\", "
             "\"params\": {}}]",
    )
    assert _relevant_capability_skill_ids(
        view=_view((skill,)), executor=_FakeExecutor([]), origin=400,
        verification=verification,
    ) == ("winsorize_skill",)


def test_different_family_with_exact_behavior_is_relevant_without_hashes():
    from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
        _relevant_capability_skill_ids,
    )

    verification = verify_program_supply_alternatives(
        executor=_FakeExecutor([
            _verification(prepared=((1.0, 2.0),)),
        ]),
        typed_patch_options=[_options()[0]],
        origin=400,
    )
    skill = SimpleNamespace(
        skill_id="equivalent_impute_skill",
        skill_kind=SkillKind.CAPABILITY,
        body="Frozen program steps: [{\"op\": \"impute_linear\", "
             "\"params\": {}}]",
    )
    executor = _FakeExecutor([
        _verification(prepared=((1.0, 2.0),)),
    ])
    assert _relevant_capability_skill_ids(
        view=_view((skill,)), executor=executor, origin=400,
        verification=verification,
    ) == ("equivalent_impute_skill",)


def test_real_scope_executor_cell_earns_route_without_evaluate():
    """The cheap controlled cell recommended in rev4: known legal typed
    alternatives, no matching capability skill."""
    values = {"s0": _series_with_outliers()}
    executor = _ScopeExecutorNoEvaluate(values)
    snapshot = compile_snapshot(H0_ROOT, verify_lock=False)
    view = resolve_harness_view(
        snapshot, {"task_kind": "forecast"}, role="fast"
    )
    assessment = route_verified_program_supply_fault(
        trace=_trace(),
        episode=object(),
        view=view,
        executor=executor,
        typed_patch_options=_options(),
        origin=400,
    )
    assert assessment.facts.expressibility_status == "PROVEN_EXPRESSIBLE"
    assert assessment.decision.cause_code == "SKILL_LIBRARY_GAP"
    assert assessment.verification.choice_offered is True
    assert all(
        alternative.verification.window_behavior_hashes == ()
        for alternative in assessment.verification.alternatives
    )
    assert executor.evaluate_calls == 0


def _series_with_outliers() -> np.ndarray:
    t = np.arange(1024, dtype=np.float64)
    series = np.sin(t / 7.0) + 5.0
    for idx in (300, 310, 320, 330, 340, 350, 360, 370):
        series[idx] += (-8.0 if idx % 2 else 12.0)
    return series


class _ScopeExecutorNoEvaluate:
    def __init__(self, values):
        from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (
            ScopeExecutor,
        )

        self.evaluate_calls = 0
        self._inner = ScopeExecutor(
            [{"series_uid": "s0", "role": "train"},
             {"series_uid": "s0", "role": "eval"}],
            values,
            {"anchors": [300]},
            evaluate_fn=lambda *a, **k: self._evaluate(*a, **k),
        )

    def _evaluate(self, roster, values, compiled, config, *, origin):
        self.evaluate_calls += 1
        raise AssertionError("E-1 must never call evaluate()")

    def verify(self, steps, origin):
        return self._inner.verify(steps, origin)

    def verify_without_behavior_hashes(self, steps, origin):
        return self._inner.verify_without_behavior_hashes(steps, origin)
