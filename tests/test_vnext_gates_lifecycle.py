from __future__ import annotations

import pytest

from SelfEvolvingHarnessTS.vnext.access import AccessTerminalArtifactV1
from SelfEvolvingHarnessTS.vnext.gates import (
    HarnessEvolutionPreregV2,
    JudgeGain,
    LLMFactorialSummaryV1,
    M3IdentityGatePreregV2,
    SAVPromotionInputV2,
    SupplierArmAggregateV2,
    llm_factorial_verdict,
    select_h0_supplier,
    support_a_validation_verdict_v2,
    task_g_verdict,
)
from SelfEvolvingHarnessTS.vnext.lifecycle import LifecycleGateError, VNextLifecycle


def _sha(character: str) -> str:
    return character * 64


def _terminal(resource: str, status: str = "passed") -> AccessTerminalArtifactV1:
    return AccessTerminalArtifactV1(
        resource_kind=resource, campaign_id=f"{resource}-campaign",
        manifest_sha=_sha("1"), reservation_sha=_sha("2"),
        terminal_status=status, result_sha=_sha("3"), terminal_event_sha=_sha("4"),
    )


def test_task_g_requires_two_positive_judges_and_no_deterministic_reversal():
    rows = [
        JudgeGain("closed", .03, .01, .05, ("a", "b")),
        JudgeGain("adam", .04, .01, .06, ("a", "c")),
        JudgeGain("lstm", -.01, -.03, .01, ("b", "c")),
    ]
    assert task_g_verdict(rows).passed
    rows[-1] = JudgeGain("lstm", -.03, -.05, -.01, ("b", "c"))
    assert not task_g_verdict(rows).passed


def _arm(arm_id: str, delta: float = 0.0, low: float = -0.01) -> SupplierArmAggregateV2:
    return SupplierArmAggregateV2(
        arm_id=arm_id, delta_vs_deterministic=delta,
        ci90_low_vs_deterministic=low,
        delta_vs_frozen_hbase=0.0, ci90_low_vs_frozen_hbase=-0.01,
        supply_ceiling_delta_vs_deterministic=max(0.0, delta),
        worst_readable_loss_regression=0.0, worst_readable_regression_ci90_low=-0.01,
        prepared_valid_fraction=1.0, cost_gate_passed=True, replay_gate_passed=True,
    )


def test_m3a_primary_is_unique_but_static_failure_does_not_block_llm_evolution():
    prereg = M3IdentityGatePreregV2()
    rows = [_arm(arm) for arm in prereg.roster]
    selection = select_h0_supplier(rows, prereg)
    assert selection.supplier_policy_id == "deterministic_b3"
    assert not selection.initial_runtime_efficacy

    evolution = llm_factorial_verdict(LLMFactorialSummaryV1(
        evolution_increment=.03, evolution_ci90_low=.01, evolution_worst_harm=0,
        mature_runtime_increment=.04, mature_runtime_ci90_low=.01,
        mature_runtime_worst_harm=0, complementarity_increment=.01,
        complementarity_ci90_low=-.01, cost_gate_passed=True, replay_gate_passed=True,
    ))
    assert evolution.evolution_llm_qualified
    assert evolution.mature_runtime_llm_qualified
    assert not evolution.complementarity_qualified
    assert HarnessEvolutionPreregV2().max_logical_candidates == 36


def test_sa_validation_v2_has_exact_h0_comparator_and_ci_based_harm():
    row = SAVPromotionInputV2(
        candidate_sha=_sha("a"), h0_sha=_sha("b"),
        ex_covid_gain_vs_h0=.03, ex_covid_ci90_low=.01,
        natural_regression_vs_h0=.06, natural_regression_ci90_low=-.01,
        controlled_regression_vs_h0=0, controlled_regression_ci90_low=-.01,
        worst_readable_loss_regression=0, worst_readable_regression_ci90_low=-.01,
        prepared_valid_fraction=1.0, unrecorded_fallback_count=0,
        dependency_masquerade_count=0, budget_violation_count=0,
    )
    assert support_a_validation_verdict_v2(row).passed
    harmful = type(row)(**{**row.__dict__, "natural_regression_ci90_low": .01})
    assert not support_a_validation_verdict_v2(harmful).passed


def test_receipt_driven_lifecycle_allows_sa_fallback_but_not_dev_feedback(tmp_path):
    lifecycle = VNextLifecycle(tmp_path / "state.json")
    lifecycle.record_m0_verdict(verdict_sha=_sha("a"), passed=True)
    lifecycle.start_task_g(prereg_sha=_sha("b"))
    lifecycle.record_task_g_terminal(result_sha=_sha("c"), passed=True)
    lifecycle.record_m2_hbase(m2_result_sha=_sha("d"), h_base_sha=_sha("e"))
    lifecycle.record_h0(lineage_sha=_sha("f"), h0_method_sha=_sha("1"))
    lifecycle.record_m3_supplier_control(result_sha=_sha("b"), supplier_policy_sha=_sha("c"))
    lifecycle.precommit_evolution_candidate(candidate_sha=_sha("2"), precommit_sha=_sha("3"))
    lifecycle.record_sa_validation_terminal(
        _terminal("sa_validation", "failed_gate"), promoted=False,
    )
    lifecycle.finalize_method(method_sha=_sha("1"), roster_sha=_sha("4"), budget_sha=_sha("5"))
    with pytest.raises(LifecycleGateError, match="dry-run"):
        lifecycle.record_dev_readonly(report_sha=_sha("6"))
    lifecycle.record_support_a_dry_run(dry_run_sha=_sha("7"))
    lifecycle.record_dev_readonly(report_sha=_sha("6"))
    with pytest.raises(LifecycleGateError, match="feedback"):
        lifecycle.record_dev_readonly(report_sha=_sha("6"))
    lifecycle.record_support_b_terminal(_terminal("support_b"), passed=True)
    reloaded = VNextLifecycle(tmp_path / "state.json")
    reloaded.authorize_final(campaign_manifest_sha=_sha("8"), authorization_sha=_sha("9"))
    reloaded.close_final(campaign_close_receipt_sha=_sha("a"))
    assert reloaded.state["sa_validation"] == "closed_fallback_h0"
    assert reloaded.state["final"] == "closed_terminal"
