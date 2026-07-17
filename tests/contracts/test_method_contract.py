import numpy as np
import pytest

from SelfEvolvingHarnessTS.contracts.method import (
    ExecutionReceipt,
    PreparationRequest,
    PreparationResult,
    PreparationStatus,
    PreparedSeries,
)
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1


def test_program_identity_is_mapping_order_independent():
    left = Program.from_steps([("denoise_savgol", {"window": 11, "order": 3})], source="det")
    right = Program.from_steps([("denoise_savgol", {"order": 3, "window": 11})], source="det")
    assert left.sha() == right.sha()
    assert left.execution_steps() == [("denoise_savgol", {"order": 3, "window": 11})]


def test_request_and_result_own_array_copies():
    raw = np.array([1.0, np.nan, 3.0])
    request = PreparationRequest("u0", raw, forecast_task_spec_v1(horizon=1), {})
    raw[0] = 99.0
    assert request.values[0] == 1.0

    prepared = PreparedSeries("u0", np.array([1.0, 2.0, 3.0]), (), "original_units")
    result = PreparationResult(
        status=PreparationStatus.PREPARED,
        prepared=prepared,
        program=None,
        receipt=ExecutionReceipt(ok=True),
    )
    assert result.status is PreparationStatus.PREPARED


def test_failed_result_cannot_carry_prepared_series():
    prepared = PreparedSeries("u0", np.ones(3), (), "original_units")
    with pytest.raises(ValueError, match="FAILED"):
        PreparationResult(
            status=PreparationStatus.FAILED,
            prepared=prepared,
            program=None,
            receipt=ExecutionReceipt(ok=False, error="boom"),
        )
