"""Benchmark-owned windowing, normalization, and downstream trainers.

The torch trainers deliberately use uniform, no-replacement shuffling plus an
explicit series-equal loss.  They never use a weighted sampler.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..evaluators._torch_models import DLinear, LSTMForecaster
from . import HEADLINE_HORIZON, HEADLINE_LOOKBACK
from .ingestion import IngestionInvalid, canonical_ingest

DEFAULT_STRIDE = 4
DEFAULT_EPOCHS = 120
DEFAULT_LR = 1e-2
DEFAULT_BATCH_SIZE = 256
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_RIDGE_LAMBDA = 1e-3
STD_FLOOR = 1e-8


@dataclass(frozen=True)
class NormalizationState:
    """Frozen benchmark-owned z-score state for one series."""

    mean: float
    std: float

    @classmethod
    def fit(cls, degraded_inner_train: Sequence[float] | np.ndarray) -> "NormalizationState":
        raw = np.asarray(degraded_inner_train, dtype=np.float64)
        if raw.ndim != 1 or raw.size == 0 or np.isinf(raw).any():
            raise ValueError("normalization source must be a non-empty finite-or-NaN vector")
        finite = raw[np.isfinite(raw)]
        if finite.size == 0:
            raise ValueError("normalization source has no finite observations")
        mean = float(finite.mean())
        std = max(float(finite.std(ddof=0)), STD_FLOOR)
        return cls(mean=mean, std=std)

    def normalize(self, finite_values: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(finite_values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("normalization input must already be finite")
        return (values - self.mean) / self.std

    def ingest_and_normalize(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        try:
            finite = canonical_ingest(np.asarray(values)).values
        except IngestionInvalid as exc:
            raise ValueError(str(exc)) from exc
        return self.normalize(finite)

    def denormalize(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.std + self.mean


@dataclass(frozen=True)
class WindowBatch:
    """One shared, normalized set of inner-train-only forecasting windows."""

    x: np.ndarray
    y: np.ndarray
    series_ids: np.ndarray
    weights: np.ndarray
    lookback: int
    horizon: int
    stride: int

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=np.float64).copy()
        y = np.asarray(self.y, dtype=np.float64).copy()
        series_ids = np.asarray(self.series_ids, dtype=str).copy()
        weights = np.asarray(self.weights, dtype=np.float64).copy()
        n = x.shape[0] if x.ndim == 2 else -1
        if (
            n < 1
            or y.ndim != 2
            or y.shape[0] != n
            or series_ids.shape != (n,)
            or weights.shape != (n,)
            or x.shape[1] != self.lookback
            or y.shape[1] != self.horizon
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isfinite(weights).all()
            or (weights <= 0).any()
        ):
            raise ValueError("invalid window batch")
        for array in (x, y, series_ids, weights):
            array.setflags(write=False)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "series_ids", series_ids)
        object.__setattr__(self, "weights", weights)

    @property
    def n_windows(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_series(self) -> int:
        return len(set(self.series_ids.tolist()))


def window_weights(series_ids: Sequence[str]) -> np.ndarray:
    ids = [str(item) for item in series_ids]
    if not ids:
        raise ValueError("at least one window is required")
    counts = Counter(ids)
    return np.asarray([1.0 / counts[item] for item in ids], dtype=np.float64)


def build_windows(
    values_by_uid: Mapping[str, Sequence[float] | np.ndarray],
    normalization_by_uid: Mapping[str, NormalizationState],
    *,
    lookback: int = HEADLINE_LOOKBACK,
    horizon: int = HEADLINE_HORIZON,
    stride: int = DEFAULT_STRIDE,
) -> WindowBatch:
    """Ingest, apply frozen normalization, then build deterministic windows."""
    if set(values_by_uid) != set(normalization_by_uid):
        raise ValueError("values/normalization uid sets differ")
    if lookback < 1 or horizon < 1 or stride < 1:
        raise ValueError("lookback, horizon, and stride must be positive")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ids: list[str] = []
    total = lookback + horizon
    for uid, raw_values in values_by_uid.items():
        values = normalization_by_uid[uid].ingest_and_normalize(raw_values)
        for start in range(0, values.size - total + 1, stride):
            xs.append(values[start : start + lookback])
            ys.append(values[start + lookback : start + total])
            ids.append(uid)
    if not xs:
        raise ValueError("no eligible inner-train windows")
    return WindowBatch(
        x=np.stack(xs),
        y=np.stack(ys),
        series_ids=np.asarray(ids),
        weights=window_weights(ids),
        lookback=lookback,
        horizon=horizon,
        stride=stride,
    )


def window_order(n_windows: int, batch_size: int, seed: int) -> list[int]:
    """Return a deterministic uniform permutation; batch size never affects sampling."""
    if n_windows < 1 or batch_size < 1:
        raise ValueError("n_windows and batch_size must be positive")
    return np.random.default_rng(int(seed)).permutation(n_windows).tolist()


def series_equal_full_loss(
    per_window_losses: torch.Tensor, series_ids: Sequence[str]
) -> torch.Tensor:
    if per_window_losses.ndim != 1 or per_window_losses.shape[0] != len(series_ids):
        raise ValueError("losses and series ids must be aligned one-dimensional arrays")
    n_series = len(set(str(item) for item in series_ids))
    if n_series < 1:
        raise ValueError("at least one series is required")
    weights = torch.as_tensor(
        window_weights(series_ids),
        dtype=per_window_losses.dtype,
        device=per_window_losses.device,
    )
    return torch.sum(weights * per_window_losses) / float(n_series)


def series_equal_batch_loss(
    losses: torch.Tensor,
    weights: torch.Tensor,
    *,
    n_windows: int,
    n_series: int,
) -> torch.Tensor:
    if losses.ndim != 1 or weights.shape != losses.shape:
        raise ValueError("losses and weights must be aligned one-dimensional tensors")
    batch_n = int(losses.shape[0])
    if batch_n < 1 or n_windows < batch_n or n_series < 1:
        raise ValueError("invalid batch dimensions")
    return (float(n_windows) / (batch_n * n_series)) * torch.sum(weights * losses)


def _moving_average_replicate(x: np.ndarray, kernel: int = 25) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), ((0, 0), (pad, pad)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=1)
    return windows.mean(axis=-1)[:, : x.shape[1]]


def _dlinear_features(x: np.ndarray) -> np.ndarray:
    trend = _moving_average_replicate(x)
    season = x - trend
    return np.concatenate([trend, season, np.ones((x.shape[0], 1))], axis=1)


@dataclass(frozen=True)
class ClosedFormDLinear:
    coefficients: np.ndarray
    lookback: int
    horizon: int
    lam: float

    def predict(self, x: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.lookback:
            raise ValueError("closed-form input has wrong shape")
        return _dlinear_features(values) @ self.coefficients


def fit_closed_form(
    batch: WindowBatch, *, lam: float = DEFAULT_RIDGE_LAMBDA
) -> ClosedFormDLinear:
    """Fit the DLinear feature map with the exact series-equal objective."""
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and non-negative")
    phi = _dlinear_features(batch.x)
    scaled_weights = batch.weights / float(batch.n_series)
    gram = phi.T @ (scaled_weights[:, None] * phi)
    cross = phi.T @ (scaled_weights[:, None] * batch.y)
    ridge = np.eye(phi.shape[1], dtype=np.float64)
    ridge[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + lam * ridge, cross)
    coefficients.setflags(write=False)
    return ClosedFormDLinear(coefficients, batch.lookback, batch.horizon, float(lam))


def _train_torch(
    model: torch.nn.Module,
    batch: WindowBatch,
    *,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> torch.nn.Module:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    torch.manual_seed(int(seed))
    model.to(torch.device("cpu")).train()
    x = torch.as_tensor(batch.x.copy(), dtype=torch.float32)
    y = torch.as_tensor(batch.y.copy(), dtype=torch.float32)
    weights = torch.as_tensor(batch.weights.copy(), dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(int(seed))
    for _ in range(epochs):
        order = rng.permutation(batch.n_windows)
        for start in range(0, batch.n_windows, batch_size):
            index = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            per_window = F.mse_loss(model(x[index]), y[index], reduction="none").mean(dim=1)
            loss = series_equal_batch_loss(
                per_window,
                weights[index],
                n_windows=batch.n_windows,
                n_series=batch.n_series,
            )
            loss.backward()
            optimizer.step()
    return model.eval()


def train_adam_dlinear(
    batch: WindowBatch,
    *,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    betas: tuple[float, float] = DEFAULT_BETAS,
    eps: float = DEFAULT_EPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> DLinear:
    torch.manual_seed(int(seed))
    model = DLinear(batch.lookback, batch.horizon)
    return _train_torch(
        model,
        batch,
        seed=seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def train_lstm_reporter(
    batch: WindowBatch,
    *,
    seed: int,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    betas: tuple[float, float] = DEFAULT_BETAS,
    eps: float = DEFAULT_EPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    hidden: int = 64,
) -> LSTMForecaster:
    torch.manual_seed(int(seed))
    model = LSTMForecaster(batch.lookback, batch.horizon, hidden=hidden)
    return _train_torch(
        model,
        batch,
        seed=seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


__all__ = [
    "ClosedFormDLinear",
    "NormalizationState",
    "WindowBatch",
    "build_windows",
    "fit_closed_form",
    "series_equal_batch_loss",
    "series_equal_full_loss",
    "train_adam_dlinear",
    "train_lstm_reporter",
    "window_order",
    "window_weights",
]
