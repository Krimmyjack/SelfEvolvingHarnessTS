from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.metrics import (
    UndefinedSeasonalScale,
    gain,
    seasonal_scale,
    smase,
)


def test_smase_uses_only_observed_training_pairs():
    y = np.array([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    observed = np.array([1, 1, 1, 1, 0, 1], dtype=bool)
    scale = seasonal_scale(y, observed, period=3, min_pairs=2)
    assert scale == pytest.approx(1.0)
    assert smase([2.0, 5.0], [1.0, 3.0], scale=scale) == pytest.approx(1.5)


def test_undefined_or_degenerate_seasonal_scale_fails_loud():
    with pytest.raises(UndefinedSeasonalScale, match="pairs"):
        seasonal_scale(np.arange(6.0), np.ones(6, dtype=bool), period=4, min_pairs=3)
    with pytest.raises(UndefinedSeasonalScale, match="degenerate"):
        seasonal_scale(np.ones(40), np.ones(40, dtype=bool), period=7, min_pairs=20)


def test_smase_and_gain_require_finite_aligned_inputs():
    assert gain(2.0, 1.25) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="aligned"):
        smase([1.0], [1.0, 2.0], scale=1.0)
    with pytest.raises(ValueError, match="finite"):
        gain(float("nan"), 1.0)

