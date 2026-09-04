"""C39-r2: max_modified_fraction judged per window vs judged over the cohort.

The fix changes one thing and the tests are written to say exactly that thing.
Nothing here asserts a hard-coded modification count: the discriminating case
is *measured* first with the cap disabled, and the two scopes are then asked to
disagree about the same measured distribution.  An operator whose internals
drift will make the setup assertion fail loudly rather than make the semantic
assertion pass for the wrong reason.
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (
    COHORT_FRACTION_REJECTION_CODE,
    FRACTION_SCOPE_COHORT,
    FRACTION_SCOPE_PER_WINDOW,
    ScopeExecutor,
)

# outlier_iqr is spike-driven: a clean ramp gives it nothing to do, a ramp with
# a handful of extreme points gives it exactly those points.  That is what
# produces an uneven per-window distribution without hard-coding a count.
STEPS = (("outlier_iqr", {}),)
SHAPE_CHANGING_STEPS = (("sliding_window", {}),)


class _FixedWindowExecutor(ScopeExecutor):
    """A ScopeExecutor whose training windows are handed in directly."""

    def __init__(self, windows, **kwargs) -> None:
        roster = [{"series_uid": "w%d" % index, "role": "train"}
                  for index in range(len(windows))]
        values = {"w%d" % index: np.asarray(window, dtype=np.float64)
                  for index, window in enumerate(windows)}
        super().__init__(roster, values, {"anchors": []}, **kwargs)
        self._windows = [np.asarray(window, dtype=np.float64)
                         for window in windows]

    def training_windows(self, origin: int):
        return [("w%d" % index, 0, window)
                for index, window in enumerate(self._windows)]


def _uneven_windows() -> list[np.ndarray]:
    clean = np.arange(40, dtype=np.float64)
    spiked = np.arange(40, dtype=np.float64)
    spiked[[3, 8, 13, 18, 23, 28, 33, 38]] = 1.0e6
    return [clean, spiked]


@pytest.fixture(name="measured")
def _measured() -> dict:
    """The per-window distribution, taken with the fraction gate disabled."""
    probe = _FixedWindowExecutor(_uneven_windows(), max_modified_fraction=1.0)
    verification = probe.verify(STEPS, 0)
    assert verification.passed, "the cap-free probe must not reject anything"
    fractions = verification.window_modified_fractions
    assert len(fractions) == verification.checked_windows == 2
    assert max(fractions) > min(fractions), (
        "the fixture no longer discriminates: outlier_iqr modified both "
        "windows equally (%s)" % (fractions,))
    cohort = verification.cohort_modified_fraction
    assert min(fractions) < cohort < max(fractions), (
        "the cohort ratio must sit strictly between the per-window extremes "
        "for the two scopes to be able to disagree (%s vs %s)"
        % (fractions, cohort))
    return {"fractions": fractions, "cohort": cohort,
            "points": verification.cohort_modified_points,
            "total": verification.cohort_total_points}


def test_default_scope_is_per_window() -> None:
    """The historical semantics stay the default, so no caller moves."""
    executor = _FixedWindowExecutor(_uneven_windows())
    assert executor.modification_fraction_scope == FRACTION_SCOPE_PER_WINDOW
    assert executor.max_modified_fraction == 0.35


def test_unknown_scope_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        _FixedWindowExecutor(_uneven_windows(),
                             modification_fraction_scope="whatever")


def test_per_window_rejects_what_cohort_admits(measured: dict) -> None:
    """The one behaviour difference the fix is allowed to introduce."""
    cap = measured["cohort"]

    per_window = _FixedWindowExecutor(
        _uneven_windows(), max_modified_fraction=cap,
        modification_fraction_scope=FRACTION_SCOPE_PER_WINDOW)
    old = per_window.verify(STEPS, 0)
    assert old.passed is False
    assert [row["rejection_code"] for row in old.rejected_windows] == [
        "MODIFICATION_FRACTION_EXCEEDED"]

    cohort = _FixedWindowExecutor(
        _uneven_windows(), max_modified_fraction=cap,
        modification_fraction_scope=FRACTION_SCOPE_COHORT)
    new = cohort.verify(STEPS, 0)
    assert new.passed is True
    assert new.rejected_windows == []
    # the boundary is ``>``, matching verify_candidate: equal to the cap passes
    assert new.cohort_modified_fraction == pytest.approx(cap)


def test_cohort_rejects_when_the_aggregate_is_over(measured: dict) -> None:
    cap = measured["cohort"] / 2.0
    cohort = _FixedWindowExecutor(
        _uneven_windows(), max_modified_fraction=cap,
        modification_fraction_scope=FRACTION_SCOPE_COHORT)
    verification = cohort.verify(STEPS, 0)
    assert verification.passed is False
    assert [row["rejection_code"] for row in verification.rejected_windows] == [
        COHORT_FRACTION_REJECTION_CODE]
    assert verification.rejected_windows[0]["series_uid"] == "__cohort__"


def test_cohort_and_per_window_agree_when_no_window_is_over(
        measured: dict) -> None:
    """Above every per-window fraction the two scopes must not diverge."""
    cap = max(measured["fractions"]) + 0.01
    for scope in (FRACTION_SCOPE_PER_WINDOW, FRACTION_SCOPE_COHORT):
        executor = _FixedWindowExecutor(
            _uneven_windows(), max_modified_fraction=cap,
            modification_fraction_scope=scope)
        assert executor.verify(STEPS, 0).passed is True


def test_cohort_scope_keeps_every_non_fraction_gate(measured: dict) -> None:
    """Only the fraction gate moved; the rest still veto one window at a time."""
    for scope in (FRACTION_SCOPE_PER_WINDOW, FRACTION_SCOPE_COHORT):
        executor = _FixedWindowExecutor(
            _uneven_windows(), max_modified_fraction=1.0,
            modification_fraction_scope=scope)
        verification = executor.verify(SHAPE_CHANGING_STEPS, 0)
        assert verification.passed is False
        codes = {row["rejection_code"] for row in verification.rejected_windows}
        assert codes and COHORT_FRACTION_REJECTION_CODE not in codes
        assert "MODIFICATION_FRACTION_EXCEEDED" not in codes


def test_diagnostics_are_produced_under_both_scopes(measured: dict) -> None:
    """Switching the gate must not switch off the other scope's reading."""
    for scope in (FRACTION_SCOPE_PER_WINDOW, FRACTION_SCOPE_COHORT):
        executor = _FixedWindowExecutor(
            _uneven_windows(), max_modified_fraction=1.0,
            modification_fraction_scope=scope)
        verification = executor.verify(STEPS, 0)
        assert verification.window_modified_fractions == measured["fractions"]
        assert verification.cohort_modified_points == measured["points"]
        assert verification.cohort_total_points == measured["total"]
        assert verification.cohort_modified_fraction == pytest.approx(
            measured["cohort"])
        assert verification.modification_fraction_scope == scope


def test_windows_over_maximum_fraction_is_diagnostic_only(
        measured: dict) -> None:
    cap = measured["cohort"]
    executor = _FixedWindowExecutor(
        _uneven_windows(), max_modified_fraction=cap,
        modification_fraction_scope=FRACTION_SCOPE_COHORT)
    verification = executor.verify(STEPS, 0)
    assert verification.windows_over_maximum_fraction == 1
    assert verification.passed is True
