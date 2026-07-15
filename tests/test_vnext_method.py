from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.method_api import MethodSeriesView, validate_prepared
from SelfEvolvingHarnessTS.policy.task_spec import forecast_task_spec_v1
from SelfEvolvingHarnessTS.vnext.method import (
    DeterministicSupplier, MethodTerminalInvalidInput, VNextBenchmarkMethod,
)


def test_minimum_vertical_slice_is_bit_deterministic_and_contract_valid():
    values = np.arange(96, dtype=float)
    values[10:15] = np.nan
    view = MethodSeriesView("audit-only-uid", values)
    task = forecast_task_spec_v1(horizon=48)
    first_method = VNextBenchmarkMethod()
    second_method = VNextBenchmarkMethod()
    first = first_method.prepare(view, task, {})
    second = second_method.prepare(view, task, {})
    assert first.operators == ()
    assert np.array_equal(first.values, second.values, equal_nan=True)
    assert first_method.audit_records[0].requested_program_sha == second_method.audit_records[0].requested_program_sha
    assert validate_prepared(first, expected_length=96).valid


def test_active_program_records_canonical_effective_operators():
    values = np.sin(np.arange(96) / 5.0)
    values[20:24] = np.nan
    from SelfEvolvingHarnessTS.vnext.grammar import ActionEligibilityManifestV1
    method = VNextBenchmarkMethod(
        supplier=DeterministicSupplier("v_ar"),
        eligibility=ActionEligibilityManifestV1.conservative(ar_reverse_test_passed=True),
    )
    prepared = method.prepare(
        MethodSeriesView("audit-uid", values), forecast_task_spec_v1(horizon=48), {},
    )
    assert prepared.operators == ("impute_linear",)
    audit = method.audit_records[-1]
    assert audit.fallback_stage == "selected_program"
    assert audit.operator_ledger == (
        ("impute_ar", "impute_linear", "insufficient_complete_lag_windows"),
    )
    assert validate_prepared(prepared, expected_length=96).valid


def test_uid_is_audit_only_and_cannot_change_method_semantics():
    values = np.sin(np.arange(96) / 7.0)
    task = forecast_task_spec_v1(horizon=48)
    first_method = VNextBenchmarkMethod()
    second_method = VNextBenchmarkMethod()
    first = first_method.prepare(MethodSeriesView("uid-a", values), task, {})
    second = second_method.prepare(MethodSeriesView("uid-b", values), task, {})
    assert np.array_equal(first.values, second.values, equal_nan=True)
    assert first.operators == second.operators
    assert first_method.audit_records[0].semantic_sha == second_method.audit_records[0].semantic_sha
    assert first_method.audit_records[0].sha256 != second_method.audit_records[0].sha256


def test_private_observed_fields_and_invalid_inputs_fail_loud():
    method = VNextBenchmarkMethod()
    task = forecast_task_spec_v1(horizon=48)
    with pytest.raises(ValueError, match="non-whitelisted"):
        method.prepare(MethodSeriesView("u", np.arange(96.0)), task, {"dataset_id": 1.0})
    for values in (np.array([]), np.array([np.nan]), np.array([np.inf])):
        with pytest.raises(MethodTerminalInvalidInput, match="METHOD_TERMINAL_INVALID_INPUT"):
            method.prepare(MethodSeriesView("u", values), task, {})
