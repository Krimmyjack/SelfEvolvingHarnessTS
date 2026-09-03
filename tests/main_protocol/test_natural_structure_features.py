"""X1 must be finite, causal, and actually responsive to the shape it names."""
from __future__ import annotations

import numpy as np
import pytest

from evaluation.main_protocol_p4 import natural_structure_features as x1
from evaluation.main_protocol_p4 import phase2_contract as contract

PERIOD = 24
SIZE = 192


def _index() -> np.ndarray:
    return np.arange(SIZE, dtype=np.float64)


def _noise(seed: int, scale: float = 0.1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, SIZE)


def test_the_frozen_contract_names_exactly_these_features():
    assert tuple(contract.X1_ADDITIONS) == x1.FEATURE_NAMES


@pytest.mark.parametrize(
    "window",
    [
        np.zeros(SIZE),
        np.ones(SIZE),
        _index(),
        np.sin(2 * np.pi * _index() / PERIOD),
        _noise(0, 5.0),
    ],
    ids=["zeros", "constant", "ramp", "pure_seasonal", "noise"],
)
def test_every_descriptor_is_finite_on_degenerate_input(window):
    # A NaN here would poison a tree split silently instead of failing loudly.
    values = x1.extract(window, period=PERIOD)
    assert set(values) == set(x1.FEATURE_NAMES)
    assert x1.is_finite_everywhere(values)


def test_gaps_do_not_produce_a_missing_descriptor():
    window = np.sin(2 * np.pi * _index() / PERIOD) + _noise(1)
    window[40:70] = np.nan
    values = x1.extract(window, period=PERIOD)
    assert x1.is_finite_everywhere(values)


def test_a_ramp_is_almost_all_trend():
    assert x1.extract(_index(), period=PERIOD)["trend_strength"] > 0.99


def test_noise_is_almost_no_trend():
    assert x1.extract(_noise(2, 1.0), period=PERIOD)["trend_strength"] < 0.4


def test_a_clean_cycle_is_almost_all_seasonal():
    window = 5.0 * np.sin(2 * np.pi * _index() / PERIOD)
    assert x1.extract(window, period=PERIOD)["seasonal_strength"] > 0.9


def test_spectral_entropy_separates_a_tone_from_noise():
    tone = np.sin(2 * np.pi * _index() / PERIOD)
    assert x1.extract(tone, period=PERIOD)["spectral_entropy"] < 0.4
    assert x1.extract(_noise(3, 1.0), period=PERIOD)["spectral_entropy"] > 0.8


def test_acf_at_period_sees_the_cycle_that_acf_lag_1_alone_would_miss():
    window = np.sin(2 * np.pi * _index() / PERIOD)
    values = x1.extract(window, period=PERIOD)
    assert values["acf_at_period"] > 0.9


def test_volatility_drift_is_signed_by_which_half_is_calmer():
    rng = np.random.default_rng(4)
    growing = np.concatenate([rng.normal(0, 0.1, SIZE // 2),
                              rng.normal(0, 2.0, SIZE // 2)])
    shrinking = growing[::-1].copy()
    assert x1.extract(growing, period=PERIOD)["volatility_drift"] > 1.0
    assert x1.extract(shrinking, period=PERIOD)["volatility_drift"] < -1.0


def test_descriptors_are_a_function_of_the_context_alone():
    # The extractor is only ever handed the pre-origin window; appending a wild
    # future must not be able to reach it, which is what makes X1 deployable.
    window = np.sin(2 * np.pi * _index() / PERIOD) + _noise(5)
    before = x1.extract(window, period=PERIOD)
    with_future = np.concatenate([window, np.full(48, 1e6)])
    after = x1.extract(with_future[:SIZE], period=PERIOD)
    assert before == after


def test_the_matrix_keeps_the_frozen_column_order():
    windows = [_index(), _noise(6)]
    values, names = x1.matrix(windows, period=PERIOD)
    assert names == x1.FEATURE_NAMES
    assert values.shape == (2, len(x1.FEATURE_NAMES))


def test_the_contract_reports_itself_frozen():
    state = contract.assert_frozen()
    assert state["frozen"], state["failures"]
    assert state["origins"] == [1176, 2136, 2376, 2616, 2856]
    assert "reversible_scale" not in state["views"]
    assert "identity" in state["views"]
