from __future__ import annotations

import numpy as np
import pytest
import torch

from SelfEvolvingHarnessTS.benchmark.trainers import (
    NormalizationState,
    build_windows,
    fit_closed_form,
    series_equal_batch_loss,
    series_equal_full_loss,
    window_order,
    window_weights,
)


def test_series_equal_full_objective():
    per_window = torch.tensor([1.0, 3.0, 9.0])
    series = np.array(["a", "a", "b"])
    assert series_equal_full_loss(per_window, series).item() == pytest.approx(5.5)


def test_batch_formula_handles_short_final_batch():
    losses = torch.tensor([2.0])
    weights = torch.tensor([0.5])
    assert series_equal_batch_loss(
        losses, weights, n_windows=3, n_series=2
    ).item() == pytest.approx(1.5)


def test_same_seed_replays_window_order():
    assert window_order(17, 11, 0) == window_order(17, 11, 0)
    assert sorted(window_order(17, 11, 0)) == list(range(17))
    assert window_order(17, 11, 0) != window_order(17, 11, 1)


def test_normalization_is_fit_from_finite_pre_method_observations():
    state = NormalizationState.fit(np.array([1.0, np.nan, 3.0]))
    assert state.mean == pytest.approx(2.0)
    assert state.std == pytest.approx(1.0)
    # Ingestion fills the middle point before this frozen state is applied.
    np.testing.assert_allclose(state.ingest_and_normalize([1.0, np.nan, 3.0]), [-1, 0, 1])


def test_build_windows_uses_stride_and_exact_series_weights():
    values = {
        "a": np.arange(10.0),
        "b": np.arange(8.0) + 100.0,
    }
    states = {uid: NormalizationState.fit(x) for uid, x in values.items()}
    batch = build_windows(
        values,
        states,
        lookback=3,
        horizon=2,
        stride=2,
    )
    assert batch.x.shape == (5, 3)
    assert batch.y.shape == (5, 2)
    assert batch.series_ids.tolist() == ["a", "a", "a", "b", "b"]
    np.testing.assert_allclose(batch.weights, [1 / 3, 1 / 3, 1 / 3, 1 / 2, 1 / 2])
    np.testing.assert_allclose(window_weights(batch.series_ids), batch.weights)


def test_closed_form_is_series_equal_and_predicts_requested_horizon():
    values = {
        "a": np.arange(18.0),
        "b": np.arange(14.0) * 2.0,
    }
    states = {uid: NormalizationState.fit(x) for uid, x in values.items()}
    batch = build_windows(values, states, lookback=4, horizon=2, stride=2)
    model = fit_closed_form(batch, lam=1e-3)
    prediction = model.predict(batch.x[:2])
    assert prediction.shape == (2, 2)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("bad", [[], [np.nan], [1.0, np.inf]])
def test_normalization_rejects_unusable_pre_method_values(bad):
    with pytest.raises(ValueError):
        NormalizationState.fit(np.asarray(bad, dtype=float))

