from __future__ import annotations

from collections.abc import Mapping

import pytest

from evaluation.main_protocol_p1 import anomaly_component
from evaluation.main_protocol_p1.common import normalize_component, validate_component


@pytest.fixture(scope="module")
def payload() -> dict:
    pytest.importorskip("sklearn")
    return anomaly_component.run(backend_mode="scripted")


def _methods(payload: Mapping) -> dict[str, Mapping]:
    return {str(row["method"]): row for row in payload["methods"]}


def _walk_keys(value):
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def test_ad_p1_component_is_a_complete_in_memory_contract_smoke(payload) -> None:
    assert payload["stage"] == "P1_COMMON_DSL_AND_CORE_BASELINE_SMOKE"
    assert payload["task_tranche"] == "anomaly_detection"
    assert payload["evidence_grade"] == "INFRASTRUCTURE"
    assert payload["status"] == "PASS"
    assert payload["component_pass"] is True
    assert payload["ad_component_pass"] is True
    assert payload["blocking_failures"] == []
    assert [row["method"] for row in payload["methods"]] == list(
        anomaly_component.MANDATORY_METHODS
    )
    assert all(row["status"] == "PASS" for row in payload["methods"])
    assert all(row["usage"]["within_caps"] for row in payload["methods"])


def test_ad_p1_component_passes_the_unified_normalized_gate(payload) -> None:
    normalized = normalize_component(payload)
    assert normalized["reported_component_pass"] is True
    assert validate_component(normalized) == []


def test_ad_p1_uses_only_mutually_exclusive_controlled_fixtures(payload) -> None:
    data = payload["data"]
    assert data["data_role"] == "CONTROLLED_EXPOSED_FIXTURE"
    assert set(data["target_series"]).isdisjoint(data["best_fixed_selection_series"])
    assert data["natural_final_outcome_reads"] == 0
    assert data["development_query_evaluations"] == 0
    assert data["yahoo_loader_available"] is False
    assert data["nab_loader_available"] is False
    assert data["sealed_ad_series_available"] is False
    assert payload["protocol_errors"]["natural_final_outcome_reads"] == 0
    assert payload["protocol_errors"]["development_query_evaluations"] == 0
    assert not any(payload["protocol_errors"].values())


def test_common_dsl_has_eleven_effect_programs_and_fixed_paths(payload) -> None:
    contract = payload["common_dsl_contract"]
    assert contract["status"] == "PASS"
    assert contract["effect_distinct_inventory_count_from_p0b"] == 11
    assert contract["eligible_operator_count"] == 11
    assert contract["eligible_operator_inventory_exact"] is True
    assert contract["identity_available"] is True
    assert contract["consumer_evaluations"] == 0
    assert contract["maximum_modified_fraction"] == pytest.approx(0.20)
    assert contract["fit_policy_extension"] is False
    assert contract["mandatory_fixed_programs_not_executable"] == []
    assert contract["mandatory_fixed_programs_not_verifier_approved"] == []
    by_program = {row["program"]: row for row in contract["rows"]}
    for _name, program in anomaly_component.FIXED_PROGRAMS:
        assert by_program[program]["compile"] == "PASS"
        assert by_program[program]["verifier"] == "PASS"
    assert contract["contract_overhead"]["charged_to_method_cell_b4"] is False


def test_best_fixed_is_frozen_off_target_and_covers_the_program_space(payload) -> None:
    row = _methods(payload)["Best Fixed Per-task"]
    details = row["details"]
    assert details["formal_evolution_winner_frozen"] is True
    assert details["selection_uses_target_support"] is False
    assert details["selection_disjoint_from_target"] is True
    assert details["program_space_coverage_complete"] is True
    assert details["cost_by_phase"]["evolution_selection"]["charged_to_target_b4"] is False
    assert details["cost_by_phase"]["target_frozen_diagnostic"]["charged_to_target_b4"] is True


def test_logical_b4_is_separate_from_raw_iforest_series_fits(payload) -> None:
    methods = _methods(payload)
    identity = methods["Identity"]
    assert identity["usage"]["full_support_evaluations"] == 2
    assert identity["usage"]["raw_consumer_fits"] == 2
    assert identity["details"]["raw_consumer_fits_by_face"] == {
        "support_a": 2,
        "support_b": 0,
    }
    parallel = methods["Parallel Best-of-N@4"]
    assert parallel["usage"]["full_support_evaluations"] == 4
    assert parallel["usage"]["raw_consumer_fits"] == 6
    assert parallel["usage"]["within_caps"] is True
    for name in ("Identity", *(name for name, _program in anomaly_component.FIXED_PROGRAMS)):
        assert methods[name]["details"]["same_adapter_support_a_b"] is True
        assert methods[name]["details"]["support_b_raw_series_fits"] == 0


def test_static_and_adaptive_lifecycle_contracts(payload) -> None:
    methods = _methods(payload)
    static = methods["Static"]
    assert static["details"]["identity_readings_equal"] is True
    assert static["details"]["prepare_calls"] == 0
    assert static["details"]["episode_writes"] == 0
    assert static["details"]["accepted_updates"] == 0

    assert payload["backend"]["production_format_exercised"] is True
    assert payload["backend"]["k0_a5_same_initial_state"] is True
    assert payload["backend"]["qualified_ad_history_skill_ids"] == []
    assert methods["K0-fixed"]["details"]["initial_skill_ids"] == methods["A5-online"]["details"]["initial_skill_ids"]
    for name in ("A3-reset", "K0-fixed", "A5-online"):
        row = methods[name]
        assert row["readings"]["abstained"] is True
        assert row["selected_program"] == "identity"
        assert row["details"]["chosen_program_steps"]
        details = row["details"]
        assert details["eligible_ad_history_skill_ids"] == []
        assert details["historical_skill_treatment"] == "NOT_EXERCISED"
        assert details["wrong_task_fail_closed"] is True
        assert not any(details["wrong_task_faces"].values())
    assert methods["A3-reset"]["details"]["writeback_channel"] is False
    assert methods["A3-reset"]["details"]["unit_state_discarded"] is True
    assert methods["K0-fixed"]["details"]["writeback_channel"] is False
    assert methods["K0-fixed"]["details"]["unit_state_discarded"] is True
    assert methods["A5-online"]["details"]["writeback_channel"] is True
    assert methods["A5-online"]["details"]["unit_state_discarded"] is False
    assert methods["A5-online"]["details"]["writeback_persisted_to_evolution_store"] is False


def test_contract_smoke_does_not_claim_the_unpassed_positive_control(payload) -> None:
    boundary = payload["positive_control_boundary"]
    assert boundary["p1_contract_smoke"] == "PASS"
    assert boundary["positive_control_44a"] == "NOT_PASSED"
    assert boundary["ad_method_release"] is False
    assert boundary["p2_p3_method_gate_released"] is False
    assert payload["performance_or_headroom_claim"] is False
    assert payload["treatment_or_capability_claim"] is False


def test_payload_adds_no_fingerprint_or_inventory_lock_fields(payload) -> None:
    forbidden_fragments = ("sha", "hash", "digest", "checksum", "manifest")
    assert not [
        key for key in _walk_keys(payload)
        if any(fragment in key.lower() for fragment in forbidden_fragments)
    ]
