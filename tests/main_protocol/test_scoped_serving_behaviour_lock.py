"""Lock the historical path so an additive Scope parameter cannot move it.

``ScopeExecutor.evaluate`` gained an optional ``serving_scope``.  The risk with
an additive parameter is that the default branch drifts, and every historical
number silently changes with it.  These tests pin the default: same call
signature, same injected evaluator, same receipt.  They also pin the two
properties that make abstention real -- an empty Scope reproduces Static
exactly, and a partial Scope leaves the declined series untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor

CONTEXT, HORIZON, PERIOD = 192, 48, 24
ORIGIN = 600
ANCHORS = (312, 372)


def _series(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index = np.arange(ORIGIN + HORIZON + 64, dtype=np.float64)
    values = (10.0 + 3.0 * np.sin(2 * np.pi * index / PERIOD)
              + rng.normal(0.0, 0.4, index.size))
    values[100 + seed] += 25.0          # one clear outlier for the program to find
    return values


@pytest.fixture(scope="module")
def cell() -> tuple[list[dict[str, str]], dict[str, np.ndarray], dict[str, object]]:
    roster = (
        [{"series_uid": "TR%d" % i, "role": "train"} for i in range(4)]
        + [{"series_uid": "EV%d" % i, "role": "eval"} for i in range(4)]
    )
    values = {
        str(row["series_uid"]): _series(index)
        for index, row in enumerate(roster)
    }
    config = {"anchors": list(ANCHORS), "period": PERIOD}
    return roster, values, config


def _executor(cell) -> ScopeExecutor:
    roster, values, config = cell
    return ScopeExecutor(
        roster, values, config,
        evaluate_fn=forecast_runtime._evaluate,
        max_modified_fraction=0.35,
    )


STEPS = (("winsorize", {}),)


def test_the_two_argument_call_still_works_and_uses_the_injected_evaluator(cell):
    # The historical signature is positional; a keyword-only addition that broke
    # it would break every existing call site in the online loop.
    roster, values, config = cell
    seen: list[int] = []

    def spy(r, v, compiled, c, *, origin):
        seen.append(int(origin))
        return forecast_runtime._evaluate(r, v, compiled, c, origin=origin)

    executor = ScopeExecutor(
        roster, values, config, evaluate_fn=spy, max_modified_fraction=0.35)
    receipt = executor.evaluate(STEPS, ORIGIN)
    assert receipt.gain is not None
    assert seen, "the default branch must go through the injected evaluate_fn"


def test_omitting_the_scope_equals_passing_none(cell):
    executor = _executor(cell)
    implicit = executor.evaluate(STEPS, ORIGIN)
    explicit = executor.evaluate(STEPS, ORIGIN, serving_scope=None)
    assert implicit.gain == explicit.gain
    assert implicit.per_view_gain == explicit.per_view_gain
    assert implicit.behavior_point_count == explicit.behavior_point_count


def test_an_empty_scope_is_exactly_static(cell):
    # Abstention has to be a real action: declining everything must return the
    # untreated numbers, not a treated one that happens to be close.
    executor = _executor(cell)
    receipt = executor.evaluate(STEPS, ORIGIN, serving_scope=frozenset())
    assert receipt.gain == 0.0
    assert receipt.per_view_gain == [0.0] * len(receipt.per_view_gain)


def test_a_partial_scope_leaves_the_declined_series_bit_identical(cell):
    roster, _values, _config = cell
    eval_uids = [str(r["series_uid"]) for r in roster if r["role"] == "eval"]
    executor = _executor(cell)
    receipt = executor.evaluate(
        STEPS, ORIGIN, serving_scope=frozenset(eval_uids[:2]))
    gains = np.asarray(receipt.per_view_gain, dtype=np.float64)
    assert np.array_equal(gains[2:], np.zeros(2)), "declined series must not move"


def test_a_scope_and_no_scope_are_different_readings(cell):
    # If they agreed, the serving-side pipeline would not be doing anything and
    # the whole contract would be decorative.
    roster, _values, _config = cell
    eval_uids = [str(r["series_uid"]) for r in roster if r["role"] == "eval"]
    executor = _executor(cell)
    unscoped = executor.evaluate(STEPS, ORIGIN)
    scoped = executor.evaluate(
        STEPS, ORIGIN, serving_scope=frozenset(eval_uids))
    assert unscoped.gain != scoped.gain


def test_a_refused_program_reports_the_verifier_not_a_scope_error(cell):
    executor = _executor(cell)
    receipt = executor.evaluate(
        (("smooth_ma", {}),), ORIGIN, serving_scope=frozenset({"EV0"}))
    if receipt.gain is None:
        assert "WINDOW_VERIFIER_REJECTED" in str(receipt.error)
