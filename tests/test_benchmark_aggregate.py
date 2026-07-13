from __future__ import annotations

import itertools

import pytest

from SelfEvolvingHarnessTS.benchmark.aggregate import (
    AggregationContractError,
    LossRow,
    aggregate_cells,
    bootstrap_ci90,
    collapse_uid_gains,
)
from SelfEvolvingHarnessTS.benchmark.corruption import CORRUPTION_GRID


def complete_loss_rows(
    *,
    model_seeds=(0, 1, 2),
    replicates=(0, 1),
    scenario_doses=CORRUPTION_GRID,
    uids=("u",),
    method_id="m",
):
    rows = []
    for uid, (scenario, dose), replicate, seed in itertools.product(
        uids, scenario_doses, replicates, model_seeds
    ):
        rows.append(
            LossRow(
                uid=uid,
                dataset_id="d1" if uid != "v" else "d2",
                regime="seasonal",
                method_id=method_id,
                scenario=scenario,
                dose=dose,
                corruption_replicate=replicate,
                model_seed=seed,
                reference_loss=2.0,
                method_loss=1.0 + 0.1 * seed,
            )
        )
    return rows


def test_collapse_requires_3_by_2_by_all_scenario_doses():
    rows = complete_loss_rows(model_seeds=(0, 1))
    with pytest.raises(AggregationContractError, match="model seeds"):
        collapse_uid_gains(rows)
    rows = complete_loss_rows(scenario_doses=CORRUPTION_GRID[:-1])
    with pytest.raises(AggregationContractError, match="scenario/dose"):
        collapse_uid_gains(rows)


def test_collapse_order_produces_exactly_one_row_per_uid():
    rows = complete_loss_rows(uids=("u", "v"))
    collapsed = collapse_uid_gains(rows)
    assert [row.uid for row in collapsed] == ["u", "v"]
    assert all(row.gain == pytest.approx(0.9) for row in collapsed)
    assert all(len(row.per_scenario_dose) == len(CORRUPTION_GRID) for row in collapsed)


def test_all_methods_must_use_identical_crn_coordinates():
    first = complete_loss_rows(method_id="a")
    second = complete_loss_rows(method_id="b")[:-1]
    with pytest.raises(AggregationContractError, match="CRN"):
        collapse_uid_gains(first + second)


def test_bootstrap_accepts_one_row_per_uid_only_and_replays():
    with pytest.raises(AggregationContractError, match="one row per uid"):
        bootstrap_ci90([("u", 0.1), ("u", 0.2)], b=20, seed=7)
    first = bootstrap_ci90({"u": 0.1, "v": 0.3}, b=100, seed=7)
    assert first == bootstrap_ci90({"u": 0.1, "v": 0.3}, b=100, seed=7)


def test_cell_then_dataset_macro_uses_equal_series_and_equal_datasets():
    rows = collapse_uid_gains(complete_loss_rows(uids=("u", "v")))
    report = aggregate_cells(rows)
    assert len(report.cells) == 2
    assert report.regimes[0].mean_gain == pytest.approx(0.9)

