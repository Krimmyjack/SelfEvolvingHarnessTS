"""Focused checks for the v1.2.1 P3 unified integration gate."""
from __future__ import annotations

import copy

import pytest

from evaluation.main_protocol_p3 import run_p3


def _usage(**overrides):
    value = run_p3._usage(
        support_a=1,
        support_b=1,
        raw_a=24,
        raw_b=0,
        cache_a=0,
        cache_b=24,
    )
    value.update(overrides)
    return value


@pytest.fixture(scope="module")
def report():
    return run_p3.build_report()


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        ({"legal_update": False}, "NO_TREATMENT"),
        (
            {
                "legal_update": True,
                "retained_across_unit": False,
                "later_reencounter": False,
                "later_behavior_influenced": False,
            },
            "GENERATION_ONLY",
        ),
        (
            {
                "legal_update": True,
                "retained_across_unit": True,
                "later_reencounter": True,
                "later_behavior_influenced": True,
                "surviving_usable_skill": False,
            },
            "TERMINAL_RISK_CONTROL",
        ),
        (
            {
                "legal_update": True,
                "retained_across_unit": True,
                "later_reencounter": True,
                "later_behavior_influenced": True,
                "surviving_usable_skill": True,
                "revalidated_after_revision": True,
            },
            "NONTERMINAL_REVISION",
        ),
        (
            {
                "legal_update": False,
                "controlled_semantic_edit": True,
                "retained_across_unit": True,
                "later_reencounter": False,
                "later_same_surface_replay": True,
                "later_behavior_influenced": True,
                "surviving_usable_skill": True,
                "revalidated_after_revision": False,
            },
            "NO_TREATMENT",
        ),
    ],
)
def test_treatment_state_is_derived_from_ordered_facts(facts, expected):
    assert run_p3.derive_treatment_state(facts)["state"] == expected


def test_budget_uses_logical_faces_and_reports_raw_fits_separately():
    row = {
        "task": "anomaly_detection",
        "arm": "identity diagnostic",
        "target": "Yahoo-24",
        "usage": _usage(),
    }
    assert run_p3.budget_failures([row]) == []

    too_many = copy.deepcopy(row)
    too_many["usage"]["full_support_evaluations"]["support_a"] = 4
    assert "logical evaluation cap exceeded" in run_p3.budget_failures(
        [too_many]
    )[0]

    bad_tokens = copy.deepcopy(row)
    bad_tokens["usage"].update(
        {"input_tokens": 3, "output_tokens": 4, "tokens": 6}
    )
    assert "token arithmetic mismatch" in run_p3.budget_failures([bad_tokens])[0]


def test_upstream_p1_roster_and_metric_are_recomputed():
    roster_failures, roster = run_p3._sealed_roster_contract()
    assert roster_failures == []
    assert roster["classification"]["sealed_final"] == ["Adiac", "ArrowHead"]
    assert roster["anomaly_detection"]["structural_roster_count"] == 65
    assert roster["anomaly_detection"]["sealed_final_count"] == 41
    assert roster["anomaly_detection"]["sealed_csv_bytes_read"] == 0

    raw = run_p3._read_object(run_p3.P1_REPORT)
    failures, summary = run_p3._validate_p1(raw)
    assert failures == []
    assert summary["three_task_roster_exact"] is True
    assert summary["thirteen_methods_per_task"] is True
    assert summary["classification_metric"] == "Macro-F1"

    tampered = copy.deepcopy(raw)
    tampered["components"]["classification"]["methods"].pop()
    assert any("method roster changed" in item for item in run_p3._validate_p1(tampered)[0])


def test_forecast_p2_is_read_only_and_treatment_is_recomputed():
    raw = run_p3._read_object(run_p3.P2_REPORT)
    failures, summary, facts = run_p3._validate_p2(raw)
    assert failures == []
    assert summary["executed_by_p3"] is False
    assert facts["legal_update"] is True
    assert facts["later_behavior_influenced"] is True
    assert facts["surviving_usable_skill"] is False

    tampered = copy.deepcopy(raw)
    row = next(
        item
        for item in tampered["runs"]
        if item["decision_index"] == 2 and item["arm"] == "A5-online"
    )
    row["trace"]["controlled_supply_count"] = 1
    assert any("ordered Forecast" in item for item in run_p3._validate_p2(tampered)[0])


def test_classification_scope_policy_replay_is_real_metric_and_claim_bounded(report):
    component = report["task_components"]["classification"]
    replay = component["controlled_scope_edit_replay"]
    checks = replay["gate_checks"]
    assert component["integration_status"] == "PASS"
    assert replay["status"] == "CONTROLLED_SCOPE_POLICY_REPLAY_PASS"
    assert all(checks.values())
    assert replay["revision"]["changed_fields"] == ["observable_applicability"]
    assert replay["revision"]["body_unchanged"] is True
    assert replay["revision"]["skill_id_unchanged"] is True
    assert replay["revision"]["production_update_executed"] is False
    assert replay["revision"]["accepted_updates"] == 0
    assert replay["revision"]["pending_before_delayed"] is False
    assert replay["revision"]["independent_delayed_approval"] is False
    later = replay["later_policy_replay"]
    assert later["k0_supply"]["supplied"] is True
    assert later["a5_supply"]["supplied"] is False
    assert later["a5_minus_k0_utility"] >= run_p3.MATERIAL
    assert later["same_surface_replay"] is True
    assert later["independent_reencounter"] is False
    assert replay["source_learned"] is False
    assert replay["autonomous_failure_diagnosis"] is False
    assert replay["evidence_independence"]["fresh_replica"] is False
    assert replay["evidence_independence"]["generalization_claim"] is False
    assert component["rq3_status"] == "NOT_EXERCISED"
    assert report["derived_treatment_state"]["classification"]["state"] == (
        "NO_TREATMENT"
    )
    a5_row = next(
        row
        for row in component["cost_rows"]
        if row["arm"] == "controlled narrowed-policy same-surface replay"
    )
    assert a5_row["usage"]["full_support_evaluations"]["support_a"] == 1


def test_ad_vertical_slice_is_identity_only_and_method_gate_stays_closed(report):
    component = report["task_components"]["anomaly_detection"]
    behavior = component["behavior"]
    boundaries = component["boundaries"]
    usage = component["cost_rows"][0]["usage"]
    assert component["integration_status"] == "PASS"
    assert behavior["status"] == "NOT_EXERCISED"
    assert behavior["deployed_program"] == "identity"
    assert behavior["production_agent_executed_in_p3"] is False
    assert behavior["episodes_written"] == 0
    assert behavior["skills_written"] == 0
    assert boundaries["held_out_requests"] == 0
    assert boundaries["all_label_requests_before_heldout"] is True
    assert usage["raw_consumer_fits"] == {"support_a": 24, "support_b": 0}
    assert usage["cache_hits"] == {"support_a": 0, "support_b": 24}
    assert component["method_gate"]["status"] == "NOT_PASSED"
    assert component["release_p4_ad"] is False


def test_p3_pass_and_p4_release_are_independent_gates(report):
    assert report["p3_integration_complete"] is True
    assert report["blocking_failures"] == []
    concern = report["revision_concern_gate"]
    assert concern["status"] == (
        "PARTIAL_MECHANICAL_SCOPE_REPLAY__PRODUCTION_REVISION_PENDING"
    )
    assert concern["revoke_only_is_sufficient"] is False
    assert concern["production_update_observed"] is False
    assert concern["independent_delayed_approval_observed"] is False
    assert concern["independent_reencounter_observed"] is False
    assert concern["same_surface_policy_replay_observed"] is True
    assert report["claim_boundaries"][
        "classification_nonterminal_production_revision_claim"
    ] is False
    assert report["release_p4"] is False
    assert report["p4_gate"]["release_p4"] is False
    assert "AD #44a positive control is not passed" in report["p4_gate"][
        "blocking_failures"
    ]
    assert "Classification production revision reachability is not formed" in (
        report["p4_gate"]["blocking_failures"]
    )
    assert report["p4_executed"] is False
    assert report["live_outcome_release"] is False
    assert all(value == 0 for value in report["protocol_errors"].values())


def test_p4_gate_cannot_be_opened_by_integration_alone():
    nonterminal = {
        "state": "NONTERMINAL_REVISION",
        "claim_ceiling": "CONTROLLED_REVISION_MECHANISM",
    }
    terminal = {
        "state": "TERMINAL_RISK_CONTROL",
        "claim_ceiling": "RISK_CONTROL_ONLY",
    }
    none = {"state": "NO_TREATMENT", "claim_ceiling": "NO_ADAPTATION_CLAIM"}
    gate = run_p3.derive_p4_gate(
        p3_integration_complete=True,
        forecast_state=terminal,
        classification_state=nonterminal,
        anomaly_state=none,
        ad_method_gate_passed=False,
    )
    assert gate["release_p4"] is False
    assert gate["release_scope"] == "NONE"
    assert gate["p4_executed"] is False


def test_p3_has_one_persistent_report_surface():
    assert run_p3.OUT_JSON.name == "p3_unified_integration_gate_20260830.json"
    assert not run_p3.OUT_JSON.with_suffix(".md").exists()
