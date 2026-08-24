"""#42k Part C -- the T6 behaviour paths carry one real TaskContext.

Zero LLM call, zero data file: the whole point of these assertions is that the
context wiring is checkable without opening a single series.  The T6 runner is
imported for its context factory only; no entry point is invoked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_t6_natural_a5_a3 as runner  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import (  # noqa: E402
    PreparationRequest,
)


def _series() -> np.ndarray:
    return np.linspace(1.0, 2.0, 64, dtype=np.float64)


def test_behaviour_context_is_one_object_bound_to_the_live_ad_spec() -> None:
    first = runner._target_task_context()
    second = runner._target_task_context()

    assert first is second, "the run must share one TaskContext object"
    assert first.task_spec == runner._target_task_spec()
    assert first.task_spec.task_type == "anomaly_detection"
    assert (first.task_spec.downstream_model_class
            == "aegists_iforest_v1")
    assert (first.deployment_constraints.fixed_downstream_model_id
            == "fixed:aegists_iforest_v1")
    # Part A: the adaptation window explores at most two non-identity probes.
    assert first.deployment_constraints.maximum_candidates == 2
    assert first.deployment_constraints.maximum_modified_fraction == 0.20


def test_behaviour_path_request_validates_and_passes_the_entry_check() -> None:
    context = runner._target_task_context()
    request = PreparationRequest(
        "t6-unit", _series(), runner._target_task_spec(), {},
        task_context=context)

    # PreparationRequest.__post_init__ is the real gate: it rejects a context
    # whose task_spec is not the request's spec.
    assert request.task_context is context
    runner._assert_behaviour_context(request)


def test_entry_check_rejects_a_missing_or_foreign_context() -> None:
    bare = PreparationRequest(
        "t6-unit-bare", _series(), runner._target_task_spec(), {})
    with pytest.raises(AssertionError):
        runner._assert_behaviour_context(bare)

    foreign_spec = runner.anomaly_task_spec_v1(
        downstream_model_class="some_other_consumer")
    foreign = PreparationRequest(
        "t6-unit-foreign", _series(), foreign_spec, {},
        task_context=runner.anomaly_task_context_v1(task_spec=foreign_spec))
    with pytest.raises(AssertionError):
        runner._assert_behaviour_context(foreign)
