"""Focused checks for the v1.2 P4 split release."""
from __future__ import annotations

import copy

from evaluation.main_protocol_p4 import run_p4


def _p3():
    return run_p4._read_object(run_p4.P3_REPORT)


def test_historical_p3_receipt_is_an_input_and_is_not_rewritten():
    before = run_p4.P3_REPORT.read_bytes()
    payload = run_p4.derive_split_gate(_p3())
    after = run_p4.P3_REPORT.read_bytes()
    assert after == before
    assert payload["p3_runner_invocations"] == 0
    assert payload["p3_report_writes"] == 0
    assert payload["historical_p3_verdict_preserved"] == (
        "P3_UNIFIED_VERTICAL_INTEGRATION_PASS__P4_HELD"
    )
    assert payload["historical_p3_release_p4_preserved"] is False


def test_rq3_not_exercised_does_not_block_h1_h2_collection():
    payload = run_p4.derive_split_gate(_p3())
    assert payload["p4_performance"]["status"] == "RELEASED"
    assert payload["p4_performance"]["release_by_task"] == {
        "forecast": True,
        "classification": True,
        "anomaly_detection": False,
    }
    assert payload["p4_evolution"]["status"] == "HELD"
    assert set(payload["p4_evolution"]["rq3_status_by_task"].values()) == {
        "RQ3_NOT_EXERCISED"
    }
    assert payload["p4_evolution"]["does_not_block_h1_h2"] is True


def test_ad_is_released_only_for_conditioning_and_safety():
    payload = run_p4.derive_split_gate(_p3())
    ad = payload["p4_ad"]
    assert ad["status"] == "RELEASED_CONDITIONING_AND_SAFETY_ONLY"
    assert ad["main_interpretation"] == "INVERTED_EFFECT_OBSERVED"
    assert ad["positive_performance_claim_authorized"] is False
    assert ad["online_revision_claim_authorized"] is False
    assert ad["consumer_metric_or_matching_change_authorized"] is False


def test_split_gate_fails_closed_on_p3_or_boundary_error():
    for mutation in ("incomplete", "protocol_error", "final_open"):
        p3 = copy.deepcopy(_p3())
        if mutation == "incomplete":
            p3["p3_integration_complete"] = False
        elif mutation == "protocol_error":
            p3["protocol_errors"]["natural_final_outcome_reads"] = 1
        else:
            p3["claim_boundaries"]["natural_final_release"] = True
        payload = run_p4.derive_split_gate(p3)
        assert payload["p4_performance"]["status"] == "HELD"
        assert payload["p4_performance"]["forecast_launch_authorized"] is False
        assert payload["p4_ad"]["status"] == "HELD"


def test_frozen_execution_plan_is_exact_and_final_stays_closed():
    payload = run_p4.derive_split_gate(_p3())
    plan = payload["execution_plan"]
    assert plan["episodes_per_task"] == 8
    assert plan["replicas"] == ["Forward", "Reverse", "Interleaved"]
    assert plan["arms"] == ["Static", "A3-reset", "K0-fixed", "A5-online"]
    assert plan["budget_scope"] == "FORECAST_P4_PERFORMANCE_ONLY"
    assert plan["full_support_budget"] == 8
    assert plan["support_a_budget"] == 7
    assert plan["support_b_budget"] == 1
    assert plan["cheap_probe_budget"] == 24
    assert plan["llm_call_budget"] == 8
    assert plan["token_budget"] == 60_000
    assert plan["accepted_update_budget"] == 1
    assert plan["wall_seconds_budget"] == 45 * 60
    assert plan["forecast_budget"] == {
        "operating_point": "B=8",
        "full_support_consumer_evaluations": 8,
        "support_a_max": 7,
        "support_b_max": 1,
        "cheap_probe_max": 24,
        "llm_call_max": 8,
        "token_max": 60_000,
        "accepted_update_max": 1,
        "wall_seconds_max": 45 * 60,
    }
    assert plan["classification_budget"] == {
        "operating_point": "B=4",
        "status": "UNCHANGED_BY_FORECAST_SPLIT_2",
    }
    assert plan["matched_baseline"] == "Parallel Best-of-N@8"
    assert plan["matched_budget"] is True
    assert plan["adaptive_arms_share_exact_budget_vector"] is True
    assert plan["a5_budget_exception"] is False
    assert plan["cell_llm_budget_exhaustion_action"] == (
        "ABSTAIN_TO_IDENTITY_AND_CONTINUE"
    )
    assert plan["cell_llm_budget_exhaustion_reason"] == (
        "LLM_CELL_BUDGET_EXHAUSTED"
    )
    assert plan["partial_cell_state_writeback"] is False
    assert plan["budget_exhaustion_rate_reported_by_arm"] is True
    assert plan["k0_a5_same_initial_knowledge"] is True
    assert plan["a5_only_arm_with_cross_unit_writeback"] is True
    assert payload["next_stage_release"] is False
    assert payload["natural_final_release"] is False
    assert payload["final_outcome_reads"] == 0
    assert run_p4.OUT_JSON.name == (
        "p4_split_gate_forecast_b8_llm8_20260830.json"
    )
