from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from pathlib import Path

import numpy as np
import pytest

from evaluation.functional.consumers import cls_scope_adapter
from evaluation.main_protocol_p1 import classification_component as p1
from evaluation.main_protocol_p1 import common


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            found.add(str(key).lower())
            found.update(_keys(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            found.update(_keys(nested))
    return found


@pytest.fixture(scope="module")
def scripted_payload():
    return p1.run(backend_mode="scripted")


def test_train_only_fixtures_have_deterministic_disjoint_surfaces():
    target, selection, record = p1._load_exposed_cells()
    assert target.fixture_id == "Epilepsy2"
    assert target.split_counts() == {"fit": 40, "support_a": 20, "support_b": 20}
    assert {cell.fixture_id for cell in selection} == {"GunPoint", "PowerCons"}
    assert {cell.fixture_id: cell.split_counts() for cell in selection} == {
        "GunPoint": {"fit": 25, "support_a": 12, "support_b": 13},
        "PowerCons": {"fit": 90, "support_a": 44, "support_b": 46},
    }
    for cell in (target, *selection):
        fit = set(cell.fit_indices)
        support_a = set(cell.support_a_indices)
        support_b = set(cell.support_b_indices)
        assert fit.isdisjoint(support_a)
        assert fit.isdisjoint(support_b)
        assert support_a.isdisjoint(support_b)
        assert fit | support_a | support_b == set(range(cell.labels.size))
        assert cell.train_member.lower().endswith("_train.ts")
    assert record["best_fixed_selection_disjoint_from_target"] is True
    assert record["selection_uses_support_or_future_utility"] is False
    assert record["test_member_bytes_read"] == 0
    assert record["development_query_evaluations"] == 0
    assert record["natural_final_outcome_reads"] == 0


def test_adapter_reuses_the_frozen_raw_plus_difference_feature_map():
    assert p1.raw_plus_difference is cls_scope_adapter.raw_plus_difference
    values = np.asarray([[1.0, 3.0, 6.0], [2.0, 5.0, 9.0]])
    features = p1.raw_plus_difference(values)
    assert features.shape == (2, 5)
    np.testing.assert_array_equal(features[:, :3], values)
    np.testing.assert_array_equal(features[:, 3:], np.diff(values, axis=1))


def test_classification_adapter_supports_package_and_legacy_import_paths(monkeypatch):
    functional = p1.PROJECT_ROOT / "evaluation" / "functional"
    monkeypatch.syspath_prepend(str(functional))
    legacy = importlib.import_module("consumers.cls_scope_adapter")
    package = importlib.import_module(
        "evaluation.functional.consumers.cls_scope_adapter"
    )
    values = np.asarray([[1.0, 2.0, 4.0]])
    np.testing.assert_array_equal(
        legacy.raw_plus_difference(values), package.raw_plus_difference(values)
    )


def test_macro_f1_is_primary_and_not_laundered_into_accuracy():
    reading = p1._classification_metrics(
        truth=[0, 0, 0, 1], predicted=[0, 0, 0, 0], classes=(0, 1)
    )
    assert reading["accuracy"] == pytest.approx(0.75)
    assert reading["macro_f1"] == pytest.approx(3.0 / 7.0)
    assert reading["macro_f1"] != reading["accuracy"]
    assert reading["per_class_recall"] == {"0": 1.0, "1": 0.0}
    assert reading["worst_class_recall"] == 0.0
    spec, context = p1._task_contract()
    assert spec.metric.name == "Macro-F1"
    assert spec.metric.direction == "higher_is_better"
    assert context.deployment_constraints.maximum_candidates == 4
    assert context.deployment_constraints.maximum_modified_fraction == 0.10


def test_common_dsl_checks_current_inventory_without_consumer_use(scripted_payload):
    contract = scripted_payload["common_dsl_contract"]
    assert contract["status"] == "PASS"
    assert contract["eligible_operator_count"] == 19
    assert contract["identity_available"] is True
    assert contract["effect_distinct_inventory_count_from_p0b"] == 18
    assert contract["effect_aliases"] == {"resample_uniform": "identity"}
    assert contract["consumer_evaluations"] == 0
    assert contract["mandatory_fixed_programs_not_executable"] == []
    assert contract["compile_failures"] == []
    assert contract["contract_overhead"]["charged_to_method_cell_b4"] is False


def test_all_thirteen_methods_pass_the_normalized_contract(scripted_payload):
    assert scripted_payload["classification_component_pass"] is True
    assert scripted_payload["blocking_failures"] == []
    rows = scripted_payload["methods"]
    assert len(rows) == 13
    assert {row["method"] for row in rows} == set(common.MANDATORY_METHODS)
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["protocol_errors"] == [] for row in rows)
    normalized = common.normalize_component(scripted_payload)
    assert common.validate_component(normalized) == []
    for row in normalized["methods"]:
        assert set(row["surfaces"]) == {"support_a", "support_b"}
        assert row["usage"]["within_caps"] is True


def test_best_fixed_is_selected_only_on_disjoint_evolution_support_a(scripted_payload):
    row = next(
        row for row in scripted_payload["methods"]
        if row["method"] == "Best Fixed Per-task"
    )
    details = row["details"]
    assert details["formal_evolution_winner_frozen"] is True
    assert details["selection_uses_target_support"] is False
    assert details["selection_disjoint_from_target"] is True
    assert details["selection_fixture_ids"] == ["GunPoint", "PowerCons"]
    assert details["program_space_coverage_complete"] is True
    covered = (
        set(details["selection_programs"])
        | set(details["safe_rejected_programs"])
        | set(details["effect_aliases"])
    )
    assert len(covered) == 20  # identity + 19 syntactic operators
    phase = details["cost_by_phase"]["evolution_selection"]
    assert phase["charged_to_target_b4"] is False
    assert phase["full_support_evaluations"] == (
        2 * len(details["selection_programs"])
    )
    assert row["usage"]["full_support_evaluations"] == {
        "support_a": 1, "support_b": 1,
    }


def test_accuracy_history_and_wrong_task_history_are_withheld_fail_closed(scripted_payload):
    history = scripted_payload["backend"]["history_contract"]
    assert history["status"] == "PASS"
    assert history["historical_input_status"] == "WITHHELD_FAIL_CLOSED"
    assert history["accuracy_card_scope"] == {
        "task_kind": "classification",
        "consumer_id": "ridge-raw-plus-difference-v1",
        "metric": "accuracy",
    }
    assert history["target_scope"]["metric"] == "Macro-F1"
    assert history["accuracy_card_direct_numeric_comparison_allowed"] is False
    assert history["wrong_task_fail_closed"] is True
    assert history["accuracy_metric_fail_closed"] is True
    assert history["rq3_treatment"] == "NOT_EXERCISED"
    assert all(row["installed"] is False for row in history["withheld_history"])
    assert all(
        row[key] == 0
        for row in history["withheld_history"]
        for key in ("retrieval", "scope_match", "supply", "support_probe", "episode")
    )


def test_harness_arm_state_and_non_exercised_revision_are_explicit(scripted_payload):
    backend = scripted_payload["backend"]
    assert backend["k0_a5_same_initial_state"] is True
    assert backend["production_format_exercised"] is True
    assert backend["production_lifecycle_exercised"] is True
    assert backend["production_ttha_method_exercised"] is True
    assert backend["production_run_online_round_exercised"] is True
    assert backend["production_open_delayed_exercised"] is True
    assert backend["live_transport_exercised"] is False
    rows = {row["method"]: row for row in scripted_payload["methods"]}
    assert rows["A3-reset"]["details"]["writeback_channel"] is False
    assert rows["A3-reset"]["details"]["unit_state_discarded"] is True
    assert rows["K0-fixed"]["details"]["writeback_channel"] is False
    assert rows["K0-fixed"]["details"]["unit_state_discarded"] is True
    assert rows["A5-online"]["details"]["writeback_channel"] is True
    assert rows["A5-online"]["details"]["unit_state_discarded"] is False
    assert rows["A5-online"]["details"]["writeback_persisted_to_evolution_store"] is False
    assert rows["A5-online"]["details"]["writeback_treatment"] == "NOT_EXERCISED"
    assert rows["K0-fixed"]["details"]["initial_skill_ids"] == rows[
        "A5-online"
    ]["details"]["initial_skill_ids"]
    for name in ("A3-reset", "K0-fixed", "A5-online"):
        details = rows[name]["details"]
        assert details["prepare_calls"] == 1
        assert details["run_online_round_calls"] == 1
        assert details["open_delayed_calls"] == 1
        assert set(details["initial_skill_ids"]) == set(p1.H0_SKILL_IDS)
        assert details["accuracy_history_faces"] == {
            "retrieval": 0, "scope_match": 0, "supply": 0,
            "support_probe": 0, "episode": 0,
        }
        assert details["wrong_task_faces"] == {
            "retrieval": 0, "scope_match": 0, "supply": 0,
            "support_probe": 0, "episode": 0,
        }


def test_static_and_frozen_h0_are_independent_real_consumer_fits(scripted_payload):
    rows = {row["method"]: row for row in scripted_payload["methods"]}
    static = rows["Static"]
    frozen = rows["Frozen H0"]
    assert static["usage"]["full_support_evaluations"] == {
        "support_a": 1, "support_b": 1,
    }
    assert static["usage"]["raw_consumer_fits"] == {
        "support_a": 1, "support_b": 1,
    }
    assert static["details"]["prepare_calls"] == 0
    assert static["details"]["episode_writes"] == 0
    assert static["details"]["store_created"] is False
    assert frozen["usage"]["full_support_evaluations"] == {
        "support_a": 1, "support_b": 0,
    }
    assert frozen["usage"]["raw_consumer_fits"] == {
        "support_a": 1, "support_b": 0,
    }
    assert frozen["surfaces"]["support_b"]["evaluation_state"] == "NOT_EVALUATED"


def test_protocol_boundaries_and_no_new_integrity_fields(scripted_payload):
    assert scripted_payload["protocol_errors"] == {
        "natural_final_outcome_reads": 0,
        "development_query_evaluations": 0,
        "task_mismatch_execution": 0,
        "cross_task_skill_leakage": 0,
        "accuracy_skill_metric_leakage": 0,
        "historical_skill_bypassed_support": 0,
        "wrong_promotion": 0,
    }
    forbidden_fragments = ("sha", "hash", "digest", "checksum", "manifest")
    assert not [
        key for key in _keys(scripted_payload)
        if any(fragment in key for fragment in forbidden_fragments)
    ]
    assert "aegists_adapter" not in scripted_payload
    assert scripted_payload["overall_p1_complete"] is False
    assert scripted_payload["release_p2"] is False
    assert not hasattr(p1, "OUT_JSON")
    assert not hasattr(p1, "OUT_MD")


def test_non_scripted_backend_is_fail_closed():
    with pytest.raises(p1.P1Blocked, match="only the reproducible scripted"):
        p1.run(backend_mode="live")


def test_source_exposes_no_fresh_final_roster_loader():
    source = Path(p1.__file__).read_text(encoding="utf-8")
    assert "Adiac" not in source
    assert "ArrowHead" not in source
    assert "_load_pool" not in source
    assert "traffic._load_pool" not in source
    assert "vaults/held_out" not in source
