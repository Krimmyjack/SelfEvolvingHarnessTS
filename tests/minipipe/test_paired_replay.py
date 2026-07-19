from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.corpus.generate import build_core_corpus
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import CaseFacts, assess_case
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.patterns import mine_failure_patterns
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.paired import (
    EditVerdict,
    CaseRunReceipt,
    OutOfScopePair,
    ReplayEvaluationStatus,
    ReplayFacts,
    PairedReplayRunner,
    derive_verdict,
)
from SelfEvolvingHarnessTS.runtime.errors import InfrastructureError
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.risk_sets import AutomaticRiskSetBuilder


def _facts(
    *,
    prediction=False,
    behavior=False,
    target=False,
    risk=True,
    scope=True,
    evaluation="ok",
):
    if target is True:
        target_status = "FULL_RECOVERY"
    elif target == "partial":
        target_status = "PARTIAL_RECOVERY"
    else:
        target_status = "NO_GAIN"
    return ReplayFacts(
        evaluation_status=(
            ReplayEvaluationStatus.INFRASTRUCTURE_FAILURE
            if evaluation == "infrastructure_failure"
            else ReplayEvaluationStatus.OK
        ),
        prediction_verified=bool(prediction),
        behavior_change_status="CHANGED" if behavior else "UNCHANGED",
        target_outcome_status=target_status,
        risk_status="PASS" if risk else "FAIL",
        scope_status="PASS" if scope else "FAIL",
        target_recovery_fraction=1.0 if target is True else 0.5 if target == "partial" else 0.0,
        median_target_improvement=0.1 if target else 0.0,
        risk_set_miss=False,
    )


@pytest.mark.parametrize(
    "facts, expected",
    [
        (_facts(prediction=False, target=False), EditVerdict.DEAD_EDIT),
        (
            _facts(prediction=True, behavior=True, target=False),
            EditVerdict.BEHAVIOR_CHANGED_NO_GAIN,
        ),
        (
            _facts(prediction=True, behavior=True, target=True, risk=False),
            EditVerdict.TARGET_RECOVERED_WITH_HARM,
        ),
        (
            _facts(prediction=True, behavior=True, target="partial", risk=True, scope=True),
            EditVerdict.PARTIAL_RECOVERY,
        ),
        (
            _facts(prediction=True, behavior=True, target=True, risk=True, scope=True),
            EditVerdict.SUPPORTED_EDIT,
        ),
        (
            _facts(prediction=False, behavior=True, target=True, risk=True, scope=True),
            EditVerdict.UNEXPECTED_GAIN,
        ),
        (
            _facts(evaluation="infrastructure_failure"),
            EditVerdict.INCONCLUSIVE,
        ),
    ],
)
def test_verdict_truth_table(facts, expected):
    assert derive_verdict(facts) is expected


def test_out_of_scope_requires_equal_view_cache_reuse_and_behavior():
    pair = OutOfScopePair(
        case_id="m0-0001",
        new_skill_applicability_match=False,
        effective_view_equal=True,
        all_eligible_calls_reused=False,
        behavior_equal=True,
    )
    assert pair.scope_status == "FAIL"


def test_top_k_displacement_is_scope_failure_even_when_new_skill_does_not_match():
    pair = OutOfScopePair(
        case_id="m0-0001",
        new_skill_applicability_match=False,
        effective_view_equal=False,
        all_eligible_calls_reused=True,
        behavior_equal=False,
    )
    assert pair.new_skill_applicability_match is False
    assert pair.effective_view_equal is False
    assert pair.scope_status == "FAIL"


def test_rules_thresholds_drive_full_and_partial_recovery():
    root = Path(__file__).resolve().parents[2]
    rules = load_m0_rules(root / "evaluation/minipipe/config/m0_rules.json")
    assert rules["target_recovery_fraction"] == pytest.approx(0.67)
    full = ReplayFacts.from_target_improvements(
        [0.02, 0.03, 0.04],
        prediction_verified=True,
        behavior_changed=True,
        risk_pass=True,
        scope_pass=True,
        rules=rules,
    )
    partial = ReplayFacts.from_target_improvements(
        [0.02, 0.005, 0.005],
        prediction_verified=True,
        behavior_changed=True,
        risk_pass=True,
        scope_pass=True,
        rules=rules,
    )
    assert full.target_outcome_status == "FULL_RECOVERY"
    assert partial.target_outcome_status == "PARTIAL_RECOVERY"


def test_unexpected_gain_never_allows_promotion():
    facts = _facts(prediction=False, behavior=True, target=True)
    assert derive_verdict(facts) is EditVerdict.UNEXPECTED_GAIN
    assert derive_verdict(facts).promotion_eligible is False


def test_repeated_infrastructure_failure_is_inconclusive_after_one_retry():
    root = Path(__file__).resolve().parents[2]
    rules = load_m0_rules(root / "evaluation/minipipe/config/m0_rules.json")

    class FailingRunner:
        def __init__(self):
            self.calls = 0

        def run(self, snapshot, case, cache):
            self.calls += 1
            raise InfrastructureError("temporary evaluator failure")

    case_runner = FailingRunner()
    runner = PairedReplayRunner(case_runner, rules=rules, cache={})
    parent = SimpleNamespace(runtime_bundle_sha="1" * 64)
    candidate = SimpleNamespace(runtime_bundle_sha="2" * 64)
    applied = SimpleNamespace(
        parent_runtime_bundle_sha="1" * 64,
        candidate_runtime_bundle_sha="2" * 64,
        target_surface_id="skill_library.entries/example_v1",
    )
    manifest = SimpleNamespace(
        edit_id="example-edit-v1",
        predicted_agent_behavior_change=("identity_retained",),
    )
    report = runner.run(
        parent=parent,
        candidate=candidate,
        applied=applied,
        manifest=manifest,
        target_cases=(SimpleNamespace(case_id="m0-0001"),),
        risk_cases=(),
    )
    assert case_runner.calls == 2
    assert report.verdict is EditVerdict.INCONCLUSIVE
    assert report.promotion_eligible is False


def test_paired_runner_confirms_surface_only_after_predicted_behavior_and_gain():
    root = Path(__file__).resolve().parents[2]
    rules = load_m0_rules(root / "evaluation/minipipe/config/m0_rules.json")
    parent = SimpleNamespace(runtime_bundle_sha="1" * 64)
    candidate = SimpleNamespace(runtime_bundle_sha="2" * 64)

    class DeterministicRunner:
        def run(self, snapshot, case, cache):
            edited = snapshot is candidate
            return CaseRunReceipt(
                case_id=case.case_id,
                utility_u=-0.20 + (0.03 if edited else 0.0),
                effective_harness_view_sha=("b" if edited else "a") * 64,
                behavior_signature_sha=("d" if edited else "c") * 64,
                eligible_agent_calls=1,
                cache_hit_flags=(edited,),
                retrieved_skill_ids=("example_v1",) if edited else (),
                supplied_operator_ids=("hampel_filter",) if edited else (),
                supplied_effect_distinct=edited,
                chosen_candidate_kind="program" if edited else "identity",
                identity_retained=True,
                modified_fraction=0.01 if edited else 0.0,
            )

    runner = PairedReplayRunner(DeterministicRunner(), rules=rules, cache={})
    applied = SimpleNamespace(
        parent_runtime_bundle_sha="1" * 64,
        candidate_runtime_bundle_sha="2" * 64,
        target_surface_id="skill_library.entries/example_v1",
    )
    manifest = SimpleNamespace(
        edit_id="example-edit-v1",
        predicted_agent_behavior_change=(
            "retrieve_skill:example_v1",
            "supply_operator:hampel_filter",
            "supply_effect_distinct",
            "choose_candidate_kind:program",
            "identity_retained",
            "scope_modified_fraction<=0.05",
        ),
    )
    report = runner.run(
        parent=parent,
        candidate=candidate,
        applied=applied,
        manifest=manifest,
        target_cases=tuple(SimpleNamespace(case_id=f"m0-000{index}") for index in range(1, 4)),
        risk_cases=(),
    )
    assert report.verdict is EditVerdict.SUPPORTED_EDIT
    assert report.confirmed_surface == applied.target_surface_id


def test_automatic_risk_set_is_stable_five_part_and_deduplicated():
    root = Path(__file__).resolve().parents[2]
    rules = load_m0_rules(root / "evaluation/minipipe/config/m0_rules.json")
    corpus = build_core_corpus(rules)
    selected = [
        case for case in corpus.targets if case.private_family == "impulsive_outlier"
    ][:2]
    failures = []
    baseline = {}
    for case in corpus.all_cases:
        facts = replace(
            CaseFacts.passing(case_id=case.case_id),
            private_family=case.private_family,
            observable_features={"local_robust_z_peak": 5.0},
        )
        if case in selected:
            facts = replace(
                facts,
                candidate_utilities={"identity": -0.4},
                effect_distinct_candidate_ids=(),
                chosen_candidate_id="identity",
                capability_skill_exists=False,
                public_probe_gains={"clipping": 0.12},
            )
        feedback = assess_case(facts, rules=rules).feedback
        baseline[case.case_id] = feedback
        if case in selected:
            failures.append(feedback)
    pattern = mine_failure_patterns(failures)[0]
    first = AutomaticRiskSetBuilder().build(pattern, corpus.all_cases, baseline)
    second = AutomaticRiskSetBuilder().build(pattern, tuple(reversed(corpus.all_cases)), baseline)
    assert tuple(first.categories) == AutomaticRiskSetBuilder.CATEGORY_ORDER
    assert first.case_ids == second.case_ids
    assert len(first.case_ids) == len(set(first.case_ids))
