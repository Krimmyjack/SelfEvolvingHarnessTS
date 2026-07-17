from __future__ import annotations

from typing import Mapping

from ...benchmark.method_api import MethodSeriesView, PreparedSeries as BenchmarkPreparedSeries
from ...contracts.method import Method, PreparationRequest, PreparationStatus
from ...contracts.task import TaskSpec
from ...methods.h_ref_v02.config import default_state
from ...runtime.fast_path import run_fast_path


class BenchmarkMethodAdapter:
    def __init__(self, method: Method) -> None:
        self._method = method
        self.method_id = method.method_id

    def prepare(
        self,
        series_view: MethodSeriesView,
        task_spec: TaskSpec,
        observed_pattern_spec: Mapping[str, float],
    ) -> BenchmarkPreparedSeries:
        result = self._method.prepare(
            PreparationRequest(
                series_view.series_uid,
                series_view.degraded_inner_train,
                task_spec,
                observed_pattern_spec,
            )
        )
        if result.status is PreparationStatus.FAILED or result.prepared is None:
            raise RuntimeError(f"canonical method {self.method_id} failed: {result.receipt.error}")
        return BenchmarkPreparedSeries(
            series_uid=result.prepared.series_uid,
            values=result.prepared.values,
            operators=result.prepared.operators,
            units=result.prepared.units,
        )


def run_h_ref_batch(views):
    state = default_state()
    return run_fast_path(views, state, state.sampler.expected_total)


__all__ = ["BenchmarkMethodAdapter", "run_h_ref_batch"]
