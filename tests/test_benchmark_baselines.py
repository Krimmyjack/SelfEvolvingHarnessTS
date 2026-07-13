from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.baselines import (
    BaselineProtocolError,
    HRefBaseline,
    OracleInSample,
    OracleTransfer,
    ProgramLoss,
    RawBaseline,
    oracle_insample,
    oracle_transfer,
    select_best_fixed,
)
from SelfEvolvingHarnessTS.benchmark.method_api import BenchmarkMethod, MethodSeriesView
from SelfEvolvingHarnessTS.policy.task_spec import forecast_task_spec_v1


def test_oracles_are_not_public_methods():
    assert not isinstance(OracleTransfer(), BenchmarkMethod)
    assert not isinstance(OracleInSample(), BenchmarkMethod)


def test_raw_is_noop_before_canonical_ingestion():
    raw = RawBaseline()
    view = MethodSeriesView("u", np.array([1.0, np.nan, 3.0]))
    prepared = raw.prepare(view, forecast_task_spec_v1(horizon=48), {})
    assert prepared.operators == ()
    assert np.array_equal(prepared.values, view.degraded_inner_train, equal_nan=True)


def _loss(role, cell, program, uid, loss):
    return ProgramLoss(role, cell, program, uid, loss)


def test_best_fixed_is_support_a_only_and_cell_macro():
    rows = [
        _loss("support_a", "c1", "p", "u", 1.0),
        _loss("support_a", "c2", "p", "v", 3.0),
        _loss("support_a", "c1", "q", "u", 2.0),
        _loss("support_a", "c2", "q", "v", 2.5),
    ]
    selected = select_best_fixed(rows)
    assert selected.program_id == "p"
    assert selected.selection_role == "support_a"
    with pytest.raises(BaselineProtocolError, match="Support-A"):
        select_best_fixed([_loss("dev_query", "c", "p", "u", 1.0)])


def test_transfer_and_insample_oracles_have_distinct_selection_surfaces():
    support = [
        _loss("support_a", "c", "p", "s", 1.0),
        _loss("support_a", "c", "q", "s", 2.0),
    ]
    query = [
        _loss("dev_query", "c", "p", "u", 3.0),
        _loss("dev_query", "c", "q", "u", 1.0),
    ]
    assert [row.program_id for row in oracle_transfer(support, query)] == ["p"]
    assert [row.program_id for row in oracle_insample(query)] == ["q"]


@dataclass(frozen=True)
class _Candidate:
    ops: tuple[str, ...]

    def op_names(self):
        return self.ops


def test_h_ref_filters_target_space_programs_before_materialization():
    runner = lambda views, state, budget: {"u": _Candidate(("znorm",))}
    baseline = HRefBaseline(
        state=object(),
        budget=10,
        run_path=runner,
        materialize_choice=lambda choice, values: values,
    )
    view = MethodSeriesView("u", np.arange(12.0))
    with pytest.raises(BaselineProtocolError, match="target-space"):
        baseline.prepare(view, forecast_task_spec_v1(horizon=48), {})
