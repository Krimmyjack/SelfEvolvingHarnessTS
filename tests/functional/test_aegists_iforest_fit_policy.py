"""#42i Part B tests -- aegists_iforest_v1 contamination_mask_refit_v1 fit policy.

The Consumer-conditioned fit policy is implemented alongside
``aegists_iforest_v1`` rather than as a regular operator.  These tests pin
the r1 constraints:

  * window-level masking only (no point-level reprojection, no row deletion)
  * mask_fraction is HARD-capped at MASK_REFIT_FRACTION (0.01)
  * standardization constants are byte-for-byte identical across both fits
  * raw training block and Query are never mutated
  * one execution = two fits (refit flag set)
  * synthetic fixture only (no Yahoo reads)

The tests use a bare ``evaluation.functional.consumers.`` import path so they
run from inside the guidance-evolution checkout with ``PYTHONPATH=.``.
"""

import importlib

import numpy as np
import pytest

from evaluation.functional.consumers import aegists_iforest_v1 as consumer
from evaluation.functional.consumers.aegists_iforest_v1 import (
    CONTAMINATION_MASK_REFIT_ID,
    FOREST_KWARGS,
    MASK_REFIT_FRACTION,
    consumer_id_for,
    fit_series,
    fit_series_with_contamination_mask,
    standardization,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _synthetic_block(seed: int = 42, length: int = 200, spike: bool = True):
    """60+ point synthetic block; optional injected spike in the suspect region.

    Not Yahoo.  Deterministic for a given seed so tests are reproducible.
    The block is well above the WINDOW=20 minimum.
    """
    rng = np.random.default_rng(seed)
    block = rng.normal(loc=0.0, scale=1.0, size=length).astype(np.float64)
    if spike:
        # Inject one spike in the middle of the block so the first forest has
        # at least one clearly anomalous window to drop on the mask.
        block[length // 2] += 8.0
    return block


# ---------------------------------------------------------------------------
# Constants and resolver
# ---------------------------------------------------------------------------

def test_fit_policy_module_constants_are_pinned():
    """r1 caps the mask fraction at 1% of training windows."""
    assert MASK_REFIT_FRACTION == 0.01
    assert CONTAMINATION_MASK_REFIT_ID == "contamination_mask_refit_v1"


def test_consumer_id_for_resolves_only_fit_policies():
    assert consumer_id_for("contamination_mask_refit_v1") == CONTAMINATION_MASK_REFIT_ID
    # Identity and the four array operators are NOT fit policies.
    for name in ("identity", "outlier_iqr", "outlier_mad",
                 "hampel_filter", "winsorize"):
        assert consumer_id_for(name) is None, name
    # Bad inputs resolve to None (no KeyError).
    assert consumer_id_for("") is None
    assert consumer_id_for("bogus") is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_fit_policy_is_deterministic():
    """Same input -> same dropped-window set, same refit forest structure."""
    block = _synthetic_block(seed=11, length=200)
    a = fit_series_with_contamination_mask(block)
    b = fit_series_with_contamination_mask(block)
    assert a["drop_indices"] == b["drop_indices"]
    assert a["training_windows"] == b["training_windows"]
    assert a["dropped_windows"] == b["dropped_windows"]
    assert a["constants"] == b["constants"]
    # Two refits of the same IsolationForest hyperparameters must produce
    # trees with identical structural fingerprint.
    a_set = sorted(id(t) for t in a["forest"].estimators_)
    # Cannot rely on object id equality across runs, but estimators_ must
    # at least have the same shape and hyperparameters.
    assert len(a["forest"].estimators_) == len(b["forest"].estimators_)
    assert a["forest"].n_estimators == b["forest"].n_estimators
    assert a["forest"].random_state == b["forest"].random_state == FOREST_KWARGS.get("random_state")


def test_fit_policy_random_state_is_pinned():
    """The forest random_state must be the consumer's pinned value, not None."""
    block = _synthetic_block(seed=99, length=200)
    out = fit_series_with_contamination_mask(block)
    assert out["forest"].random_state == FOREST_KWARGS.get("random_state")


# ---------------------------------------------------------------------------
# Budget cap (mask fraction <= 1%)
# ---------------------------------------------------------------------------

def test_fit_policy_default_mask_fraction_is_one_percent():
    block = _synthetic_block(seed=1, length=200)
    out = fit_series_with_contamination_mask(block)
    assert out["mask_fraction_used"] <= MASK_REFIT_FRACTION
    assert out["mask_fraction_used"] == pytest.approx(MASK_REFIT_FRACTION)


def test_fit_policy_rejects_mask_fraction_above_one_percent():
    """r1 binds mask_fraction <= 1% as a hard upper bound."""
    import math

    block = _synthetic_block(seed=1, length=200)
    # 1.1% is above the cap; the policy must refuse.
    with pytest.raises(ValueError, match=r"mask_fraction"):
        fit_series_with_contamination_mask(block, mask_fraction=0.011)
    # 5% is also above the cap.
    with pytest.raises(ValueError, match=r"mask_fraction"):
        fit_series_with_contamination_mask(block, mask_fraction=0.05)
    # Negative fractions are out of range.
    with pytest.raises(ValueError, match=r"mask_fraction"):
        fit_series_with_contamination_mask(block, mask_fraction=-0.01)


def test_fit_policy_accepts_smaller_mask_fractions():
    """Caller may pick a smaller mask fraction; cap is upper bound."""
    block = _synthetic_block(seed=2, length=200)
    out_small = fit_series_with_contamination_mask(block, mask_fraction=0.0)
    assert out_small["dropped_windows"] == 0
    assert out_small["mask_fraction_used"] == 0.0
    assert out_small["training_windows"] == out_small["first_forest_windows"]


# ---------------------------------------------------------------------------
# Standardization constant invariance
# ---------------------------------------------------------------------------

def test_fit_policy_preserves_standardization_constants():
    """Constants must be byte-for-byte identical to the baseline's."""
    block = _synthetic_block(seed=3, length=200)
    baseline_constants = standardization(block)
    out = fit_series_with_contamination_mask(block)
    assert set(out["constants"].keys()) == set(baseline_constants.keys())
    for key, val in baseline_constants.items():
        assert out["constants"][key] == val, (key, val, out["constants"][key])
    # And the policy output's constants must equal the inline standardization.
    inline = standardization(block)
    assert out["constants"] == inline


# ---------------------------------------------------------------------------
# Input / Query zero-touch
# ---------------------------------------------------------------------------

def test_fit_policy_does_not_mutate_training_block():
    block = _synthetic_block(seed=4, length=200)
    snapshot = block.copy()
    fit_series_with_contamination_mask(block)
    # The training block must be unchanged: no NaN insertion, no row deletion.
    assert block.dtype == snapshot.dtype
    assert block.shape == snapshot.shape
    np.testing.assert_array_equal(block, snapshot)


def test_fit_policy_query_path_is_identical_to_baseline():
    """Query-side scoring (after fit) must use the same constants path."""
    block = _synthetic_block(seed=5, length=200)
    out = fit_series_with_contamination_mask(block)
    # The Query-side standardisation pipeline (`standardization`) on the
    # raw block must yield the same constants the model carries.  This
    # is the property that makes the fit policy a no-op on the Query side.
    live = standardization(block)
    assert live == out["constants"]


def test_fit_policy_baseline_fit_series_still_works():
    """The baseline fit_series path must continue to work unchanged.

    r1 forbids the policy from replacing fit_series; it lives alongside it.
    """
    block = _synthetic_block(seed=6, length=200)
    baseline = fit_series(block)
    assert "forest" in baseline
    assert baseline["forest"].n_estimators == FOREST_KWARGS.get("n_estimators")


# ---------------------------------------------------------------------------
# Window-level masking
# ---------------------------------------------------------------------------

def test_fit_policy_masks_at_most_one_percent_of_windows():
    """With a 200-point block, WINDOW=20, expect 181 training windows and a
    floor(0.01 * 181) = 1 drop.  The mask must NEVER exceed 1%."""
    block = _synthetic_block(seed=7, length=200)
    out = fit_series_with_contamination_mask(block)
    n_windows = out["first_forest_windows"]
    expected_drops = int(np.floor(MASK_REFIT_FRACTION * n_windows))
    assert out["dropped_windows"] == expected_drops
    assert out["training_windows"] == n_windows - expected_drops
    # Drops are capped at <=1% of total windows.
    assert out["dropped_windows"] <= int(np.ceil(MASK_REFIT_FRACTION * n_windows))


def test_fit_policy_drop_indices_are_within_window_range():
    """Drop indices must be valid integer indices into the window matrix."""
    block = _synthetic_block(seed=8, length=200)
    out = fit_series_with_contamination_mask(block)
    n_windows = out["first_forest_windows"]
    for idx in out["drop_indices"]:
        assert 0 <= idx < n_windows
    assert len(set(out["drop_indices"])) == len(out["drop_indices"])  # unique


def test_fit_policy_drops_highest_anomaly_score_windows():
    """The dropped windows must be the top-k highest-scoring windows."""
    block = _synthetic_block(seed=9, length=200)
    out = fit_series_with_contamination_mask(block)
    # Compute the same first-forest decision scores on the raw block, then
    # verify that the dropped indices correspond to the top-k highest
    # anomaly scores (-decision_function).
    constants = standardization(block)
    windows = consumer._windows(consumer._apply(block, constants))
    first = consumer.IsolationForest(**FOREST_KWARGS)
    first.fit(windows)
    decision = np.asarray(first.decision_function(windows), dtype=np.float64)
    anomaly = -decision
    expected_top = np.argsort(-anomaly, kind="stable")[:out["dropped_windows"]]
    np.testing.assert_array_equal(
        np.array(sorted(out["drop_indices"])),
        np.sort(expected_top),
    )


# ---------------------------------------------------------------------------
# Refit flag + bookkeeping
# ---------------------------------------------------------------------------

def test_fit_policy_returns_refit_true():
    block = _synthetic_block(seed=10, length=200)
    out = fit_series_with_contamination_mask(block)
    assert out["refit"] is True
    assert out["policy_id"] == CONTAMINATION_MASK_REFIT_ID


def test_fit_policy_two_fits_per_call():
    """One execution = two IsolationForest fits (first + refit).

    We patch the ``fit`` method on the IsolationForest class so the
    assertion does not depend on subclassing (sklearn's BaseEstimator
    refuses subclasses with varargs).  Each ``fit`` call increments a
    counter; the policy must reach exactly 2.
    """
    block = _synthetic_block(seed=11, length=200)
    counter = {"n": 0}
    real_fit = consumer.IsolationForest.fit

    def counting_fit(self, *args, **kwargs):
        counter["n"] += 1
        return real_fit(self, *args, **kwargs)

    consumer.IsolationForest.fit = counting_fit
    try:
        out = fit_series_with_contamination_mask(block)
    finally:
        consumer.IsolationForest.fit = real_fit
    # Two fits: first_forest.fit + refit_forest.fit.
    assert counter["n"] == 2
    # The returned forest is the refit forest (last one fitted).
    assert out["forest"] is not None
    assert out["refit"] is True
