"""Pure contract checks for the Forecast P4-Performance runner.

These tests do not call ``run()``, load KDD values, invoke an Agent/backend, or
write a report.  They lock only the pre-registered plan and in-memory report
semantics.
"""
from __future__ import annotations

import copy
import inspect
import json
from types import SimpleNamespace

import pytest

from evaluation.main_protocol_p4 import run_forecast_p4_performance as p4


def test_eight_natural_origins_and_three_replica_orders_are_exact():
    assert p4.ORIGINS == (600, 648, 696, 744, 792, 840, 888, 936)
    assert p4.REPLICA_ORDERS == {
        "Forward": (600, 648, 696, 744, 792, 840, 888, 936),
        "Reverse": (936, 888, 840, 792, 744, 696, 648, 600),
        "Interleaved": (600, 792, 648, 840, 696, 888, 744, 936),
    }

    plan = p4.unit_plan()
    assert len(plan) == 8 * 3
    assert all(row["natural_episode"] is True for row in plan)
    assert all(row["horizon"] == p4.HORIZON for row in plan)
    assert {
        (row["replica"], row["sequence_index"], row["origin"])
        for row in plan
    } == {
        (replica, index, origin)
        for replica, origins in p4.REPLICA_ORDERS.items()
        for index, origin in enumerate(origins, start=1)
    }
    assert {
        (row["episode_id"], row["origin"])
        for row in plan
    } == {
        (f"E{index}", origin)
        for index, origin in enumerate(p4.ORIGINS, start=1)
    }


def test_four_core_arms_and_parallel_h2_comparator_are_separate():
    assert p4.CORE_ARMS == ("Static", "A3-reset", "K0-fixed", "A5-online")
    assert p4.ADAPTIVE_ARMS == ("A3-reset", "K0-fixed", "A5-online")
    assert p4.PARALLEL_COMPARATOR == "Parallel Best-of-N@8"
    assert p4.PARALLEL_COMPARATOR not in p4.CORE_ARMS
    assert p4.PARALLEL_PROGRAMS == (
        "impute_linear",
        "hampel_filter",
        "winsorize",
        "outlier_iqr",
        "impute_fft",
        "impute_ema",
        "period_complete",
    )

    payload = p4._initial_payload(
        release={"performance_status": "RELEASED"},
        replicas=tuple(p4.REPLICA_ORDERS),
        backend_mode="scripted",
    )
    assert payload["frozen_contract"]["core_arms"] == list(p4.CORE_ARMS)
    assert payload["frozen_contract"]["h2_comparator"] == (
        p4.PARALLEL_COMPARATOR
    )
    assert payload["expected"] == {
        "core_cells": 8 * 3 * 4,
        "h2_comparator_cells": 8 * 3,
    }

    comparator_source = inspect.getsource(p4._parallel_row)
    assert (
        '"support_a_full_evaluations": len(PARALLEL_PROGRAMS)'
        in comparator_source
    )
    assert '"support_b_full_evaluations": 1' in comparator_source
    assert '"independent_search_only": True' in comparator_source
    assert '"cross_unit_writeback": False' in comparator_source


def test_b8_is_a_per_cell_ceiling_with_seven_plus_one_face_split():
    contract = p4.budget_contract()
    per_cell = contract["per_method_cell"]
    assert contract["operating_point"] == "B=8"
    assert per_cell["full_support_consumer_evaluations"] == 8
    assert per_cell["support_a_max"] == 7
    assert per_cell["support_b_max"] == 1
    assert per_cell["cheap_probe_max"] == 24
    assert per_cell["llm_call_max"] == 8
    assert per_cell["token_max"] == 60_000
    assert contract["matched_ceiling_not_required_spend"] is True
    assert contract["adaptive_cell_count"] == 3 * 8 * 3
    assert contract["global_llm_call_cap"] == 3 * 8 * 3 * 8
    assert contract["core_cell_count"] == 3 * 8 * 4
    assert contract["h2_comparator_cell_count"] == 3 * 8

    valid = {
        "support_a_full_evaluations": 7,
        "support_b_full_evaluations": 1,
        "full_support_evaluations": 8,
        "raw_consumer_fits": 8,
        "cheap_probes": 24,
        "llm_calls": 8,
        "tokens": 60_000,
        "accepted_updates": 1,
        "wall_seconds": 45 * 60,
    }
    assert p4.validate_usage(valid) is True

    for field, value in (
        ("support_a_full_evaluations", 8),
        ("support_b_full_evaluations", 2),
        ("full_support_evaluations", 9),
        ("raw_consumer_fits", 9),
        ("cheap_probes", 25),
        ("llm_calls", 9),
        ("tokens", 60_001),
        ("accepted_updates", 2),
        ("wall_seconds", 45 * 60 + 0.001),
    ):
        over = dict(valid)
        over[field] = value
        assert p4.validate_usage(over) is False, field


def test_p4_task_context_raises_candidate_cap_without_changing_p1_default():
    eligible = p4.forecast_p1._eligible_programs()
    _spec, p4_context = p4.forecast_p1._task_contract(
        eligible, maximum_candidates=p4.B_MAIN
    )
    _spec, p1_context = p4.forecast_p1._task_contract(eligible)
    assert p4_context.deployment_constraints.maximum_candidates == 8
    assert p1_context.deployment_constraints.maximum_candidates == 4


def test_parallel_b8_evaluates_seven_support_a_candidates_then_one_winner(
    monkeypatch,
):
    calls: list[tuple[str, str]] = []
    support_a_smase = {
        op: float(index + 2)
        for index, op in enumerate(p4.PARALLEL_PROGRAMS)
    }
    support_a_smase["impute_ema"] = 0.5

    def fake_reading(_cell, face, steps, *, origin):
        assert origin == p4.ORIGINS[0]
        op = str(steps[0][0])
        calls.append((face, op))
        smase = support_a_smase[op] if face == "support_a" else 0.75
        return {
            "smase": smase,
            "utility": -smase,
            "median_series_smase": smase,
            "worst_series_smase": smase,
            "per_series_smase": [smase],
            "behavior_point_count": 1,
        }

    monkeypatch.setattr(p4, "_reading", fake_reading)
    row = p4._parallel_row(
        p4.unit_plan(("Forward",))[0],
        object(),
        p4.ORIGINS[0],
        {"support_b": {"utility": -1.0}},
    )

    assert calls[:7] == [("support_a", op) for op in p4.PARALLEL_PROGRAMS]
    assert calls[7:] == [("support_b", "impute_ema")]
    assert row["details"]["selected_program"] == "impute_ema"
    assert row["usage"]["support_a_full_evaluations"] == 7
    assert row["usage"]["support_b_full_evaluations"] == 1
    assert row["usage"]["full_support_evaluations"] == 8
    assert row["status"] == "PASS"


def test_release_fails_closed_if_any_forecast_b8_budget_field_drifts(
    monkeypatch, tmp_path
):
    gate = p4.split_release.derive_split_gate(
        p4.split_release._read_object(p4.split_release.P3_REPORT)
    )
    gate_path = tmp_path / "p4_split_gate_forecast_b8_llm8_20260830.json"
    monkeypatch.setattr(p4.split_release, "OUT_JSON", gate_path)
    monkeypatch.setattr(p4, "PROJECT_ROOT", tmp_path)

    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    assert all(p4._assert_release()["checks"].values())

    for field in tuple(gate["execution_plan"]["forecast_budget"]):
        drifted = copy.deepcopy(gate)
        old = drifted["execution_plan"]["forecast_budget"][field]
        drifted["execution_plan"]["forecast_budget"][field] = (
            "B=4" if isinstance(old, str) else old + 1
        )
        gate_path.write_text(json.dumps(drifted), encoding="utf-8")
        with pytest.raises(p4.ForecastP4Blocked, match="budget_vector"):
            p4._assert_release()


def test_k0_a5_share_initial_knowledge_and_only_a5_carries_across_units():
    payload = p4._initial_payload(
        release={},
        replicas=("Forward",),
        backend_mode="scripted",
    )
    frozen = payload["frozen_contract"]
    assert frozen["k0_a5_same_initial_knowledge"] is True
    assert frozen["a5_only_cross_unit_writeback"] is True

    run_source = inspect.getsource(p4.run)
    adaptive_source = inspect.getsource(p4._adaptive_row)
    assert "a5_snapshot = shared_initial" in run_source
    assert "a5_episodes: tuple[Any, ...] = ()" in run_source
    assert (
        'elif arm == "K0-fixed":\n                            '
        "base_snapshot = shared_initial"
    ) in run_source
    assert (
        "base_snapshot = a5_snapshot\n                            "
        "carried = a5_episodes"
    ) in run_source
    assert (
        'if arm == "A5-online":\n                            '
        "a5_snapshot = end_snapshot"
    ) in run_source
    assert '"cross_unit_writeback": arm == "A5-online"' in adaptive_source
    assert '"unit_state_discarded": arm != "A5-online"' in adaptive_source


def _exercise_adaptive_failure(monkeypatch, tmp_path, arm, failure):
    base_snapshot = SimpleNamespace(
        marker="base",
        skills=(SimpleNamespace(skill_id="base-skill"),),
    )
    partial_snapshot = SimpleNamespace(
        marker="partial",
        skills=(SimpleNamespace(skill_id="partial-skill"),),
    )
    snapshot_box = {"value": base_snapshot}
    carried_episodes = ("prior-episode",)
    method = SimpleNamespace(
        experience_episodes=list(carried_episodes),
        last_trace=None,
    )
    method._active_snapshot = lambda: snapshot_box["value"]

    class FakeBackend:
        def __init__(self):
            self.calls = 0
            self._shared = SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
            )
            self.scope = None
            self.maximum_calls = None

        def new_arm_backend(self, *, scope_id, maximum_calls):
            self.scope = scope_id
            self.maximum_calls = maximum_calls
            return object()

    backend = FakeBackend()

    class FakeCell:
        observation_block = [0.0]
        values = {"series": [0.0]}

        @staticmethod
        def roster(face):
            return [{"series_uid": face, "role": face}]

    class FakeExecutor:
        def __init__(self, *_args, **_kwargs):
            self._baseline_cache = {}
            self._per_view_cache = {}

        def evaluate(self, *_args, **_kwargs):  # pragma: no cover
            raise AssertionError("an exhausted prepare must not evaluate Support")

    def fail_round(*_args, **_kwargs):
        backend.calls = 8
        backend._shared.prompt_tokens = 120
        backend._shared.completion_tokens = 30
        snapshot_box["value"] = partial_snapshot
        method.experience_episodes.append(
            SimpleNamespace(
                episode_id="discarded-partial",
                support_response={"gain": 0.1},
                delayed_response={},
                context_summary={},
                relation="POSITIVE",
                local_status="pending",
            )
        )
        raise failure

    monkeypatch.setattr(p4.forecast_course, "_live_agent", lambda *_args: object())
    monkeypatch.setattr(
        p4.four_arms,
        "_new_state",
        lambda **_kwargs: {
            "method": method,
            "controller": object(),
            "store": object(),
        },
    )
    monkeypatch.setattr(p4, "ScopeExecutor", FakeExecutor)
    monkeypatch.setattr(p4, "_request", lambda **_kwargs: (object(), {}))
    monkeypatch.setattr(p4, "_config", lambda _origin: {})
    monkeypatch.setattr(
        p4.forecast_p1,
        "_snapshot_state_view",
        lambda snapshot: snapshot.marker,
    )
    monkeypatch.setattr(p4, "run_online_round", fail_round)

    identity_reading = {
        "smase": 1.0,
        "utility": -1.0,
        "median_series_smase": 1.0,
        "worst_series_smase": 1.0,
        "per_series_smase": [1.0],
        "behavior_point_count": 1,
    }
    result = p4._adaptive_row(
        unit=p4.unit_plan(("Forward",))[0],
        arm=arm,
        cell=FakeCell(),
        origin=p4.ORIGINS[0],
        base_snapshot=base_snapshot,
        carried_episodes=carried_episodes,
        backend=backend,
        temp_root=tmp_path,
        spec=object(),
        context=object(),
        identity={
            "support_a": identity_reading,
            "support_b": identity_reading,
        },
    )
    return result, base_snapshot, carried_episodes, backend


@pytest.mark.parametrize("arm", p4.ADAPTIVE_ARMS)
def test_cell_llm_exhaustion_is_identity_abstain_with_atomic_rollback(
    monkeypatch, tmp_path, arm
):
    result, base_snapshot, carried_episodes, backend = _exercise_adaptive_failure(
        monkeypatch,
        tmp_path,
        arm,
        p4.shared_harness.Stop(
            p4.CELL_LLM_EXHAUSTION_VERDICT,
            "ninth call blocked before backend",
        ),
    )
    row, end_snapshot, end_episodes = result

    assert row["status"] == "PASS"
    assert row["task_native"]["utility"] == -1.0
    assert row["delta_utility_vs_identity"] == 0.0
    assert row["usage"] == {
        "support_a_full_evaluations": 0,
        "support_b_full_evaluations": 0,
        "full_support_evaluations": 0,
        "raw_consumer_fits": 0,
        "cheap_probes": 0,
        "llm_calls": 8,
        "input_tokens": 120,
        "output_tokens": 30,
        "tokens": 150,
        "accepted_updates": 0,
        "wall_seconds": row["usage"]["wall_seconds"],
    }
    assert row["details"]["selected_program"] == "identity"
    assert row["details"]["abstained"] is True
    assert row["details"]["abstain_reason"] == (
        "LLM_CELL_BUDGET_EXHAUSTED"
    )
    assert row["details"]["llm_budget_exhausted"] is True
    assert row["details"]["episodes_written"] == []
    assert [
        episode["episode_id"]
        for episode in row["details"]["discarded_partial_episodes"]
    ] == ["discarded-partial"]
    assert row["details"]["discarded_partial_state_changed"] is True
    assert row["details"]["cross_unit_writeback"] is False
    assert row["details"]["unit_state_discarded"] is True
    assert end_snapshot is base_snapshot
    assert end_episodes == carried_episodes
    assert backend.maximum_calls == 8


@pytest.mark.parametrize(
    "failure, exception_type",
    [
        (
            p4.shared_harness.Stop(
                "LLM_BUDGET_EXCEEDED",
                "global cap exhausted",
            ),
            p4.shared_harness.Stop,
        ),
        (RuntimeError("transport failure"), RuntimeError),
    ],
)
def test_cell_fallback_does_not_swallow_global_or_transport_errors(
    monkeypatch, tmp_path, failure, exception_type
):
    with pytest.raises(exception_type):
        _exercise_adaptive_failure(
            monkeypatch,
            tmp_path,
            "A5-online",
            failure,
        )


def test_query_final_and_controlled_witnesses_are_excluded_from_performance():
    payload = p4._initial_payload(
        release={},
        replicas=tuple(p4.REPLICA_ORDERS),
        backend_mode="scripted",
    )
    scope = payload["scope"]
    assert scope["performance_hypotheses"] == ["H1", "H2"]
    assert scope["evolution_h3_status"] == "HELD__RQ3_NOT_EXERCISED"
    assert scope["controlled_treatment_rows"] == 0
    assert scope["injected_treatment_rows"] == 0
    assert scope["query_evaluations"] == 0
    assert scope["natural_final_outcome_reads"] == 0
    assert scope["ucr_test_outcome_reads"] == 0
    assert scope["sealed_ad_outcome_reads"] == 0
    assert all(row["natural_episode"] is True for row in payload["unit_plan"])

    run_source = inspect.getsource(p4.run)
    assert "_make_cell(" not in run_source
    assert "_controlled_card(" not in run_source
    assert '"controlled_or_injected_treatment": False' in run_source


def _synthetic_rows() -> list[dict[str, object]]:
    utility = {
        "Static": -1.00,
        "A3-reset": -0.90,
        "K0-fixed": -0.85,
        "A5-online": -0.80,
        p4.PARALLEL_COMPARATOR: -0.88,
    }
    rows: list[dict[str, object]] = []
    for unit in p4.unit_plan():
        for method, value in utility.items():
            rows.append(
                {
                    **unit,
                    "method": method,
                    "task_native": {"utility": value, "smase": -value},
                    "delta_utility_vs_identity": value - utility["Static"],
                    "material_harm_event": False,
                    "usage": {
                        "raw_consumer_fits": 1,
                        "llm_calls": 0,
                        "tokens": 0,
                        "wall_seconds": 0.0,
                    },
                }
            )
    return rows


def test_aggregation_keeps_h1_h2_confirmatory_and_k0_diagnostic_only():
    aggregated = p4.aggregate(_synthetic_rows())
    by_method = aggregated["by_method"]
    assert set(by_method) == {*p4.CORE_ARMS, p4.PARALLEL_COMPARATOR}
    assert all(record["n"] == 8 * 3 for record in by_method.values())

    contrasts = aggregated["confirmatory_performance_contrasts"]
    h1 = contrasts["H1_A5_minus_A3"]
    h2 = contrasts["H2_A5_minus_Parallel"]
    assert h1["left"] == "A5-online"
    assert h1["right"] == "A3-reset"
    assert h1["paired_n"] == 8 * 3
    assert h1["mean_delta_utility"] == pytest.approx(0.10)
    assert h2["left"] == "A5-online"
    assert h2["right"] == p4.PARALLEL_COMPARATOR
    assert h2["paired_n"] == 8 * 3
    assert h2["mean_delta_utility"] == pytest.approx(0.08)

    diagnostic = aggregated["evolution_diagnostic_only"]
    assert diagnostic["A5_minus_K0"]["paired_n"] == 8 * 3
    assert diagnostic["A5_minus_K0"]["mean_delta_utility"] == pytest.approx(
        0.05
    )
    assert diagnostic["claim_status"] == "RQ3_NOT_EXERCISED"
    assert diagnostic["independent_h3_gate_changed"] is False


def test_aggregation_reports_budget_exhaustion_as_arm_cost_without_row_drop():
    rows = _synthetic_rows()
    target_counts = {"A3-reset": 2, "K0-fixed": 1, "A5-online": 3}
    seen = {method: 0 for method in target_counts}
    for row in rows:
        method = row["method"]
        if method in target_counts:
            row["details"] = {
                "llm_budget_exhausted": seen[method] < target_counts[method]
            }
            seen[method] += 1

    aggregated = p4.aggregate(rows)
    by_method = aggregated["by_method"]
    for method, count in target_counts.items():
        assert by_method[method]["llm_budget_exhaustion_count"] == count
        assert by_method[method]["llm_budget_exhaustion_rate"] == pytest.approx(
            count / 24
        )
        assert by_method[method]["llm_budget_exhaustion_applicability"] == (
            "APPLICABLE"
        )

    for method in ("Static", p4.PARALLEL_COMPARATOR):
        assert by_method[method]["llm_budget_exhaustion_count"] == 0
        assert by_method[method]["llm_budget_exhaustion_rate"] is None
        assert by_method[method]["llm_budget_exhaustion_applicability"] == (
            "NOT_APPLICABLE__NO_LLM_BUDGET"
        )

    efficiency = aggregated["llm_budget_exhaustion_efficiency_by_arm"]
    assert set(efficiency) == set(p4.ADAPTIVE_ARMS)
    assert efficiency["A5-online"]["exhaustion_count"] == 3
    assert efficiency["A5-online"]["exhaustion_rate"] == pytest.approx(3 / 24)
    assert aggregated["confirmatory_performance_contrasts"][
        "H1_A5_minus_A3"
    ]["paired_n"] == 24
