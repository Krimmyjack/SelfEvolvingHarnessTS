"""The reversible-view algebra: closure, causality, and the affine screen.

These are the invariants the O1 contract rests on, and all of them are pure --
no Consumer, no data fixture.  The data-dependent gate (the identity view
reproducing the frozen evaluator to the bit) lives in
``preflight_representation_evaluator`` because it needs real windows and real
Consumer fits.
"""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.main_protocol_p4 import representation_view as views

CONTEXT, HORIZON = views.CONTEXT_LENGTH, views.HORIZON


def _seasonal_window(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index = np.arange(CONTEXT + HORIZON, dtype=np.float64)
    return (
        3.0 * np.sin(2 * np.pi * index / 24.0)
        + 0.05 * index
        + 10.0
        + rng.normal(0.0, 0.2, index.size)
    )


ADMISSIBLE = (
    views.ReversibleDetrend(),
    views.ReversibleSeasonalAdjust(),
    views.ReversibleDifference(),
)


@pytest.mark.parametrize("view", ADMISSIBLE, ids=lambda v: v.name)
def test_the_horizon_round_trips_through_the_view(view):
    window = _seasonal_window()
    params = view.fit(window[:CONTEXT])
    horizon = window[CONTEXT:]
    forward = view.forward(horizon, params, start=CONTEXT)
    back = view.inverse(forward, params, start=CONTEXT)
    assert np.max(np.abs(back - horizon)) < 1e-8


@pytest.mark.parametrize("view", ADMISSIBLE, ids=lambda v: v.name)
def test_parameters_cannot_see_the_horizon(view):
    # Fitting reads the context only, so replacing the future entirely must
    # leave the parameters untouched -- otherwise a view leaks the Outcome.
    window = _seasonal_window()
    polluted = window.copy()
    polluted[CONTEXT:] = 1e6
    assert repr(view.fit(window[:CONTEXT])) == repr(view.fit(polluted[:CONTEXT]))


def test_a_positive_affine_view_is_rejected_as_a_provable_no_op():
    # median and 1.4826*MAD are affine-equivariant, so the Consumer's own
    # standardisation erases any x -> a*x + b view exactly.  Measuring such a
    # view would spend Consumer fits to observe zero by construction.
    windows = [_seasonal_window(seed) for seed in range(4)]
    screen = views.affine_cancellation_screen(views.ReversibleScale(), windows)
    assert screen["cancelled_by_consumer_normalisation"] is True
    assert screen["max_standardised_difference"] < 1e-9


@pytest.mark.parametrize("view", ADMISSIBLE, ids=lambda v: v.name)
def test_admissible_views_survive_the_consumer_standardisation(view):
    windows = [_seasonal_window(seed) for seed in range(4)]
    screen = views.affine_cancellation_screen(view, windows)
    assert screen["cancelled_by_consumer_normalisation"] is False


def test_identity_is_reported_as_cancelled_because_it_is_a_no_op():
    windows = [_seasonal_window()]
    screen = views.affine_cancellation_screen(views.IdentityView(), windows)
    assert screen["cancelled_by_consumer_normalisation"] is True


def test_difference_anchors_the_horizon_on_the_last_context_value():
    # The anchor is what makes the inverse causal: reconstructing the horizon
    # must use a value the deployment could actually have seen.
    window = _seasonal_window()
    view = views.ReversibleDifference()
    params = view.fit(window[:CONTEXT])
    assert params["anchor"] == pytest.approx(window[CONTEXT - 1])
    forward = view.forward(window[CONTEXT:], params, start=CONTEXT)
    assert forward[0] == pytest.approx(window[CONTEXT] - window[CONTEXT - 1])


def test_detrend_removes_a_line_it_can_extend_over_the_horizon():
    index = np.arange(CONTEXT + HORIZON, dtype=np.float64)
    ramp = 2.0 + 0.5 * index
    view = views.ReversibleDetrend()
    params = view.fit(ramp[:CONTEXT])
    flattened = view.forward(ramp, params, start=0)
    # A pure ramp is entirely trend, so the whole window flattens -- including
    # the horizon block the fit never saw.
    assert np.max(np.abs(flattened)) < 1e-6


def test_seasonal_profile_is_centred_so_it_removes_shape_not_level():
    view = views.ReversibleSeasonalAdjust()
    params = view.fit(_seasonal_window()[:CONTEXT])
    assert float(np.mean(params["profile"])) == pytest.approx(0.0, abs=1e-12)
