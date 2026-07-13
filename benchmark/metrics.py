"""Unique benchmark loss and paired-gain semantics."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class UndefinedSeasonalScale(ValueError):
    """The clean inner-train series cannot support headline sMASE."""


def seasonal_scale(
    clean_inner_train: Sequence[float] | np.ndarray,
    observed_mask: Sequence[bool] | np.ndarray,
    *,
    period: int,
    min_pairs: int = 32,
) -> float:
    values = np.asarray(clean_inner_train, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    if values.ndim != 1 or observed.shape != values.shape or values.size == 0:
        raise ValueError("values and observed mask must be aligned non-empty vectors")
    if period < 1 or min_pairs < 1 or period >= values.size:
        raise ValueError("period and min_pairs must be valid positive integers")
    finite = np.isfinite(values)
    usable = observed[period:] & observed[:-period] & finite[period:] & finite[:-period]
    if int(usable.sum()) < min_pairs:
        raise UndefinedSeasonalScale(
            f"only {int(usable.sum())} observed seasonal pairs; need {min_pairs}"
        )
    differences = np.abs(values[period:] - values[:-period])[usable]
    scale = float(differences.mean())
    base = values[observed & finite]
    magnitude = float(np.mean(np.abs(base))) if base.size else 0.0
    floor = 1e-8 * max(1.0, magnitude)
    if not np.isfinite(scale) or scale <= floor:
        raise UndefinedSeasonalScale(
            f"degenerate seasonal scale {scale!r} at floor {floor!r}"
        )
    return scale


def smase(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    scale: float,
) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim != 1 or prediction.shape != truth.shape or truth.size == 0:
        raise ValueError("truth and prediction must be aligned non-empty vectors")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("truth and prediction must be finite")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return float(np.mean(np.abs(truth - prediction)) / scale)


def gain(reference_loss: float, method_loss: float) -> float:
    values = np.asarray([reference_loss, method_loss], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("gain inputs must be finite")
    return float(reference_loss - method_loss)


__all__ = ["UndefinedSeasonalScale", "gain", "seasonal_scale", "smase"]

