import numpy as np

from SelfEvolvingHarnessTS.benchmark.method_api import MethodSeriesView
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest, PreparationStatus
from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.method_compat import BenchmarkMethodAdapter
from SelfEvolvingHarnessTS.methods.h_ref_v02.method import HRefV02Method


def test_canonical_h_ref_and_benchmark_adapter_are_equivalent():
    values = np.sin(np.arange(160, dtype=float) / 8.0)
    values[10:14] = np.nan
    task = forecast_task_spec_v1(horizon=12)
    method = HRefV02Method()

    canonical = method.prepare(PreparationRequest("u0", values, task, {}))
    benchmark = BenchmarkMethodAdapter(method).prepare(
        MethodSeriesView("u0", values), task, {}
    )

    assert canonical.status is PreparationStatus.PREPARED
    assert np.array_equal(canonical.prepared.values, benchmark.values, equal_nan=True)
    assert canonical.prepared.operators == benchmark.operators
    assert benchmark.units == "original_units"
