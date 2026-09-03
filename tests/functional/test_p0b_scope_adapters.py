from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.functional.consumers.p0b_scope_adapters import (
    ForecastScopeAdapter,
    TrainingBlockScopeExecutor,
    WindowedIForestAdapter,
)


class Budget:
    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, count: int = 1) -> None:
        if self.used + count > self.cap:
            raise RuntimeError("fit budget exceeded")
        self.used += int(count)


def _compiled(*steps):
    program = SimpleNamespace(execution_steps=lambda: tuple(steps))
    return SimpleNamespace(candidate=SimpleNamespace(program=program))


def _forecast_fixture():
    t = np.arange(400, dtype=np.float64)
    values = {
        "train_a": np.sin(2 * np.pi * t / 24) + 0.002 * t,
        "train_b": 0.8 * np.sin(2 * np.pi * (t + 3) / 24) + 0.003 * t,
        "train_c": 1.2 * np.sin(2 * np.pi * (t + 7) / 24) - 0.001 * t,
        "eval_a": 0.9 * np.sin(2 * np.pi * (t + 1) / 24) + 0.0025 * t,
        "eval_b": 1.1 * np.sin(2 * np.pi * (t + 5) / 24) + 0.0015 * t,
    }
    roster = [
        {"series_uid": uid, "role": "train"}
        for uid in ("train_a", "train_b", "train_c")
    ] + [
        {"series_uid": uid, "role": "eval"}
        for uid in ("eval_a", "eval_b")
    ]
    return roster, values, {"anchors": (192, 240), "period": 24}


def _ad_fixture():
    t = np.arange(360, dtype=np.float64)
    values = np.sin(2 * np.pi * t / 30) + 0.0005 * t
    values[[202, 203, 204, 302, 303, 304]] += 8.0
    rows = {
        "series_a": {
            "values": values,
            "windows": {
                "r1": {
                    "train": [0, 180],
                    "support_a": [180, 260],
                    "support_b": [260, 340],
                }
            },
        }
    }
    roster = [{"series_uid": "series_a", "role": "train"}]
    value_map = {"series_a": values}
    events = {
        ("series_a", 180, 260): [list(range(202, 205))],
        ("series_a", 260, 340): [list(range(302, 305))],
    }
    return rows, roster, value_map, events


def _assert_common_reading(reading) -> None:
    assert set(("mean_smase", "per_view_smase", "behavior_point_count")) <= set(reading)
    assert np.isfinite(reading["mean_smase"])
    assert reading["per_view_smase"]
    assert np.isfinite(reading["per_view_smase"]).all()
    assert reading["behavior_point_count"] >= 0


def test_forecast_adapter_runs_the_frozen_pooled_ridge_and_charges_one_fit() -> None:
    from evaluation.functional import run_e2_autonomous_natural_workflow_generation as v6

    roster, values, config = _forecast_fixture()
    budget = Budget(2)
    adapter = ForecastScopeAdapter(
        frozen_evaluate=v6._evaluate,
        fit_budget=budget,
        phase_by_origin={300: "support_a"},
    )
    reading = adapter(roster, values, None, config, origin=300)

    _assert_common_reading(reading)
    assert reading["behavior_point_count"] == 0
    assert budget.used == 1
    assert adapter.calls == [{
        "phase": "support_a",
        "origin": 300,
        "logical_evaluations": 1,
        "raw_consumer_fits": 1,
    }]


def test_forecast_unknown_origin_fails_before_fit() -> None:
    called = []
    budget = Budget(1)
    adapter = ForecastScopeAdapter(
        frozen_evaluate=lambda *args, **kwargs: called.append(True),
        fit_budget=budget,
        phase_by_origin={300: "support_a"},
    )
    with pytest.raises(RuntimeError, match="not an open P0b surface"):
        adapter([], {}, None, {}, origin=301)
    assert budget.used == 0
    assert called == []


def test_training_block_executor_uses_each_series_frozen_geometry() -> None:
    rows, _, _, _ = _ad_fixture()
    executor = TrainingBlockScopeExecutor(
        rows=rows,
        round_name="r1",
        evaluate_fn=lambda *args, **kwargs: {
            "mean_smase": 0.0,
            "per_view_smase": [0.0],
            "behavior_point_count": 0,
        },
    )
    windows = executor.training_windows(12345)
    assert [(uid, lo, len(block)) for uid, lo, block in windows] == [
        ("series_a", 0, 180)
    ]


def test_training_block_executor_allows_narrower_task_bound() -> None:
    rows, _, _, _ = _ad_fixture()
    default_executor = TrainingBlockScopeExecutor(
        rows=rows,
        round_name="r1",
        evaluate_fn=lambda *args, **kwargs: {},
    )
    ad_executor = TrainingBlockScopeExecutor(
        rows=rows,
        round_name="r1",
        evaluate_fn=lambda *args, **kwargs: {},
        max_modified_fraction=0.20,
    )

    assert default_executor.max_modified_fraction == pytest.approx(0.35)
    assert ad_executor.max_modified_fraction == pytest.approx(0.20)


def test_real_iforest_adapter_reuses_fit_across_support_surfaces() -> None:
    pytest.importorskip("sklearn")
    from evaluation.functional.consumers import aegists_iforest_v1 as consumer

    rows, roster, values, events = _ad_fixture()
    requests = []

    def event_reader(uid, lo, hi):
        requests.append((uid, lo, hi))
        return events[(uid, lo, hi)]

    budget = Budget(2)
    adapter = WindowedIForestAdapter(
        consumer=consumer,
        rows=rows,
        round_name="r1",
        event_reader=event_reader,
        fit_budget=budget,
        phase_by_origin={10: "support_a", 20: "support_b"},
    )
    support_a = adapter(roster, values, None, {}, origin=10)
    support_b = adapter(roster, values, None, {}, origin=20)

    _assert_common_reading(support_a)
    _assert_common_reading(support_b)
    assert support_a["mean_smase"] == pytest.approx(-support_a["ad_macro_f1"])
    assert support_b["mean_smase"] == pytest.approx(-support_b["ad_macro_f1"])
    assert support_a["behavior_point_count"] == 0
    assert support_b["behavior_point_count"] == 0
    assert budget.used == 1
    assert adapter.calls[0]["raw_consumer_fits"] == 1
    assert adapter.calls[1]["raw_consumer_fits"] == 0
    assert adapter.calls[1]["model_cache_hits"] == 1
    assert requests == [
        ("series_a", 180, 260),
        ("series_a", 260, 340),
    ]


def test_ad_unknown_origin_never_reaches_label_reader() -> None:
    from evaluation.functional.consumers import aegists_iforest_v1 as consumer

    rows, roster, values, _ = _ad_fixture()
    requests = []
    adapter = WindowedIForestAdapter(
        consumer=consumer,
        rows=rows,
        round_name="r1",
        event_reader=lambda *args: requests.append(args),
        fit_budget=Budget(1),
        phase_by_origin={10: "support_a"},
    )
    with pytest.raises(RuntimeError, match="not an open P0b surface"):
        adapter(roster, values, None, {}, origin=999)
    assert requests == []


def test_ad_model_cache_identity_includes_operator_parameters() -> None:
    class FakeConsumer:
        def fit_series(self, block):
            return {"mean": float(np.mean(block))}

        def score_series(self, model, values, region, truth):
            return {
                "f1": 0.5,
                "truth_events": len(truth),
                "predicted_events": 1,
                "matched_events": int(bool(truth)),
            }

        def macro_f1(self, rows):
            return float(np.mean([row["f1"] for row in rows.values()]))

        def pooled_f1(self, rows):
            return self.macro_f1(rows)

    rows, roster, values, events = _ad_fixture()
    budget = Budget(4)
    adapter = WindowedIForestAdapter(
        consumer=FakeConsumer(),
        rows=rows,
        round_name="r1",
        event_reader=lambda uid, lo, hi: events[(uid, lo, hi)],
        fit_budget=budget,
        phase_by_origin={10: "support_a"},
    )
    first = _compiled(("winsorize", {"limits": 0.05}))
    second = _compiled(("winsorize", {"limits": 0.10}))
    reading_a = adapter(roster, values, first, {}, origin=10)
    reading_b = adapter(roster, values, second, {}, origin=10)

    _assert_common_reading(reading_a)
    _assert_common_reading(reading_b)
    assert budget.used == 2
    assert adapter.calls[0]["raw_consumer_fits"] == 1
    assert adapter.calls[1]["raw_consumer_fits"] == 1
