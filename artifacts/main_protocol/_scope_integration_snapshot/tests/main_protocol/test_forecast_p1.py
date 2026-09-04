from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.main_protocol_p1 import run_forecast_p1 as p1
from SelfEvolvingHarnessTS.methods.ttha import fast_agent as fast_agent_module


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
def scripted_payload(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("forecast_p1")
    original_json, original_md = p1.OUT_JSON, p1.OUT_MD
    p1.OUT_JSON = root / "report.json"
    p1.OUT_MD = root / "report.md"
    try:
        yield p1.run(backend_mode="scripted")
    finally:
        p1.OUT_JSON, p1.OUT_MD = original_json, original_md


def test_p0b_is_the_only_launch_precondition():
    release = p1._assert_p0_release()
    assert release["status"] == "PASS"
    assert release["p1_release"] is True


def test_exposed_kdd_cells_are_disjoint_20_20_without_query_or_final_loader():
    cell, selection, record = p1._load_exposed_cells()
    for fixture in (cell, selection):
        assert len(fixture.support_a) == 20
        assert len(fixture.support_b) == 20
        assert set(fixture.support_a).isdisjoint(fixture.support_b)
        assert set(fixture.values) == (
            set(fixture.support_a) | set(fixture.support_b)
        )
    assert set(cell.values).isdisjoint(selection.values)
    assert record["best_fixed_selection_disjoint_from_target"] is True
    assert record["selection_uses_support_or_future_utility"] is False
    assert record["development_query_evaluations"] == 0
    assert record["natural_final_outcome_reads"] == 0
    assert record["traffic_or_solar_loader_available"] is False


def test_common_dsl_contract_uses_current_inventory(scripted_payload):
    contract = scripted_payload["common_dsl_contract"]
    assert contract["status"] == "PASS"
    assert contract["identity_available"] is True
    assert contract["effect_distinct_inventory_count_from_p0b"] == 18
    assert contract["mandatory_fixed_programs_not_executable"] == []
    assert contract["consumer_evaluations"] == 0


def test_all_thirteen_core_methods_reach_a_bounded_contract(scripted_payload):
    rows = scripted_payload["methods"]
    assert len(rows) == 13
    assert {row["method"] for row in rows} == set(p1.MANDATORY_METHODS)
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["usage"]["within_caps"] for row in rows)
    assert all(row["usage"]["raw_consumer_fits"] <= p1.B_MAIN for row in rows)
    assert all(row["usage"]["llm_calls"] <= p1.MAX_LLM_CALLS for row in rows)


def test_k0_a5_share_safe_forecast_origin_and_wrong_task_is_silent(scripted_payload):
    backend = scripted_payload["backend"]
    assert backend["k0_a5_same_initial_state"] is True
    assert "sa1_supply_scope_v2" in backend["k0_a5_initial_skill_ids"]
    assert "s2a_forecast_supply_v0" in backend["k0_a5_initial_skill_ids"]
    contract = backend["k0_a5_forecast_supply_contract"]
    assert contract["requires_target_support"] is True
    assert contract["supplies_candidates"] is True
    assert contract["grants_execution"] is False
    assert contract["kdd_capability_claim"] is False
    rows = {row["method"]: row for row in scripted_payload["methods"]}
    for name in ("K0-fixed", "A5-online"):
        assert rows[name]["details"]["cross_task_faces"] == {
            "retrieval": 0,
            "scope_match": 0,
            "supply": 0,
            "episode": 0,
        }
        assert rows[name]["details"]["forecast_source_scope_fail_closed"] is True
        assert rows[name]["details"]["protocol_errors"] == []
    assert rows["K0-fixed"]["details"]["unit_state_discarded"] is True
    assert rows["K0-fixed"]["details"]["next_unit_base"] == "shared_initial_state"
    assert rows["K0-fixed"]["details"]["carried_episode_count"] == 0
    assert rows["K0-fixed"]["details"]["carried_new_skill_count"] == 0
    assert rows["A5-online"]["details"]["writeback_treatment"] in {
        "NOT_EXERCISED", "RETAINED_NEW_STATE"
    }


def test_scripted_run_passes_forecast_but_cannot_complete_p1_or_release_p2(
    scripted_payload,
):
    assert scripted_payload["forecast_component_pass"] is True
    assert scripted_payload["blocking_failures"] == []
    assert scripted_payload["backend"]["production_format_exercised"] is True
    assert scripted_payload["backend"]["live_transport_exercised"] is False
    assert scripted_payload["overall_p1_complete"] is False
    assert scripted_payload["release_p2"] is False
    assert scripted_payload["protocol_errors"] == {
        "natural_final_outcome_reads": 0,
        "development_query_evaluations": 0,
        "task_mismatch_execution": 0,
        "cross_task_skill_leakage": 0,
        "historical_skill_bypassed_support": 0,
        "wrong_promotion": 0,
    }


def test_best_fixed_is_frozen_before_target_on_disjoint_exposed_data(
    scripted_payload,
):
    row = next(
        row for row in scripted_payload["methods"]
        if row["method"] == "Best Fixed Per-task"
    )
    details = row["details"]
    assert details["formal_evolution_winner_frozen"] is True
    assert details["selection_uses_target_support"] is False
    assert details["selection_disjoint_from_target"] is True
    assert details["program_space_coverage_complete"] is True
    covered = (
        set(details["selection_programs"])
        | set(details["safe_rejected_programs"])
        | set(details["effect_aliases"])
    )
    assert len(covered) == 19
    assert details["effect_aliases"] == {"resample_uniform": "identity"}
    selection = details["cost_by_phase"]["evolution_selection"]
    assert selection["charged_to_target_b4"] is False
    assert selection["full_support_evaluations"] == len(
        details["selection_programs"]
    )
    assert row["usage"]["full_support_evaluations"] == 2
    assert scripted_payload["execution_order"][1].startswith("best_fixed_frozen")


def test_static_is_real_zero_lifecycle_with_independent_consumer_smoke(
    scripted_payload,
):
    rows = {row["method"]: row for row in scripted_payload["methods"]}
    static = rows["Static"]
    assert static["status"] == "PASS"
    assert static["selected_program"] == "identity"
    assert static["usage"]["full_support_evaluations"] == 2
    assert static["usage"]["raw_consumer_fits"] == 2
    assert static["details"] == {
        "prepare_calls": 0,
        "episode_writes": 0,
        "delayed_open_calls": 0,
        "accepted_updates": 0,
        "writeback_attempts": 0,
        "store_created": False,
        "protocol_errors": [],
    }
    assert static["readings"] == rows["Identity"]["readings"]


def test_face_dispatcher_caches_same_program_without_cross_face_reuse():
    receipt_a = SimpleNamespace(
        verification=SimpleNamespace(checked_windows=7)
    )
    receipt_b = SimpleNamespace(
        verification=SimpleNamespace(checked_windows=9)
    )

    class Face:
        def __init__(self, receipt):
            self.receipt = receipt
            self.calls = 0

        def evaluate(self, _steps, _origin):
            self.calls += 1
            return self.receipt

    face_a, face_b = Face(receipt_a), Face(receipt_b)
    dispatcher = p1._FaceExecutor(
        {p1.ORIGIN: face_a, p1.ORIGIN + 1: face_b},
        labels={p1.ORIGIN: "support_a", p1.ORIGIN + 1: "support_b"},
    )
    steps_a = (("winsorize", {"upper": 0.9, "lower": 0.1}),)
    steps_same = (("winsorize", {"lower": 0.1, "upper": 0.9}),)
    assert dispatcher.evaluate(steps_a, p1.ORIGIN) is receipt_a
    assert dispatcher.evaluate(steps_same, p1.ORIGIN) is receipt_a
    assert dispatcher.evaluate(steps_same, p1.ORIGIN + 1) is receipt_b
    assert face_a.calls == 1
    assert face_b.calls == 1
    accounting = dispatcher.accounting()
    assert accounting["requests_by_face"] == {"support_a": 2, "support_b": 1}
    assert accounting["unique_receipt_requests_by_face"] == {
        "support_a": 1, "support_b": 1,
    }
    assert accounting["cache_hits_by_face"] == {"support_a": 1, "support_b": 0}
    assert accounting["duplicate_requests"] == 1


def test_cheap_probe_accounting_uses_actual_requests(scripted_payload):
    rows = {row["method"]: row for row in scripted_payload["methods"]}
    frozen = rows["Frozen H0"]
    assert frozen["usage"]["cheap_probes"] == frozen["details"][
        "fast_candidate_verifier_requests"
    ]
    for name in ("A3-reset", "K0-fixed", "A5-online"):
        row = rows[name]
        expected = row["details"]["fast_candidate_verifier_requests"] + row[
            "details"
        ]["receipt_accounting"]["unique_candidate_verifier_requests"]
        assert row["usage"]["cheap_probes"] == expected
        assert expected <= p1.MAX_PROBES
    contract = scripted_payload["common_dsl_contract"]
    assert contract["contract_overhead"]["candidate_verifier_requests"] == (
        contract["eligible_operator_count"]
    )
    assert contract["contract_overhead"]["charged_to_method_cell_b4"] is False


def test_forecast_full_pool_skips_the_actionability_probe(monkeypatch, tmp_path):
    def forbidden_probe(*_args, **_kwargs):
        raise AssertionError("P1 full-pool path called the actionability probe")

    monkeypatch.setattr(
        fast_agent_module, "_actionable_operators", forbidden_probe
    )
    cell, _selection, _record = p1._load_exposed_cells()
    identity_budget = p1.FitBudget()
    identity = {
        face: p1._evaluate(cell, face, "identity", identity_budget)
        for face in ("support_a", "support_b")
    }
    eligible = p1._eligible_programs()
    spec, context = p1._task_contract(eligible)
    row = p1._frozen_h0(
        cell=cell,
        snapshot=p1.forecast_course._h0(),
        backend=p1.shared_harness._scripted_backend(p1.LIVE_GLOBAL_CALL_CAP),
        root=tmp_path,
        spec=spec,
        context=context,
        identity=identity,
    )
    assert row["status"] == "PASS"
    assert row["details"]["protocol_errors"] == []


def test_p1_report_adds_no_hash_or_revision_lock_fields(scripted_payload):
    forbidden = {
        "sha",
        "sha256",
        "checksum",
        "digest",
        "manifest_hash",
        "git_head",
        "execution_signature",
    }
    assert not (_keys(scripted_payload) & forbidden)
    assert scripted_payload["aegists_adapter"]["status"] == (
        "STRUCTURALLY_INCOMPATIBLE"
    )
    assert scripted_payload["aegists_adapter"]["blocking"] is False


def test_runner_source_has_no_final_dataset_loader():
    source = Path(p1.__file__).read_text(encoding="utf-8")
    assert "traffic._load_pool" not in source
    assert "_traffic_csv_path" not in source
    assert "solar-energy" not in source.lower()
    assert "vaults/held_out" not in source
