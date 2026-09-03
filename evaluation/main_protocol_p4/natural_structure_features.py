"""X1: six natural-structure descriptors, frozen as a hypothesis.

The existing PatternCard is degradation-oriented -- missingness, robust-z peaks,
level excursions, period reliability.  It answers "how broken is this window".
The Phase-2 hypothesis is that choosing a preparation program also needs "what
shape is this window", which is a different question: how much of the variance
is trend, how much is seasonal, how concentrated the spectrum is, how much
memory there is at lag 1 and at the period, and whether the variability is
drifting.

These are declared before the confirmation cohort is read.  They are **not** a
diagnosis: the closure document records that the binding gate is maximum
single-series harm and that the failure cannot be uniquely attributed to
features, model capacity or sample size.  X1 is one of the three axes being
tested, and a null on it closes this feature set, not the idea of features.

Every descriptor is a function of the pre-origin context alone, computed on the
linear-integrity completion of that context -- the same array the Consumer sees.
Each is guarded to return a finite float on degenerate input, because a NaN
here would silently poison a tree split rather than fail loudly.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)

FEATURE_NAMES = (
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "acf_lag_1",
    "acf_at_period",
    "volatility_drift",
)

_EPS = 1e-12


def _finite(value: float, *, default: float = 0.0) -> float:
    result = float(value)
    return result if np.isfinite(result) else default


def _variance(values: np.ndarray) -> float:
    return float(np.var(values)) if values.size > 1 else 0.0


def _strength(residual_variance: float, total_variance: float) -> float:
    """Hyndman-style strength: how much variance the component removed."""
    if total_variance <= _EPS:
        return 0.0
    return _finite(min(1.0, max(0.0, 1.0 - residual_variance / total_variance)))


def _detrended(values: np.ndarray) -> np.ndarray:
    index = np.arange(values.size, dtype=np.float64)
    slope, intercept = np.polyfit(index, values, 1)
    return values - (intercept + slope * index)


def _seasonally_adjusted(values: np.ndarray, period: int) -> np.ndarray:
    profile = np.zeros(period, dtype=np.float64)
    for phase in range(period):
        block = values[phase::period]
        profile[phase] = float(np.median(block)) if block.size else 0.0
    profile -= float(profile.mean())
    return values - profile[np.arange(values.size) % period]


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag < 1 or values.size <= lag + 1:
        return 0.0
    left = values[:-lag] - float(np.mean(values[:-lag]))
    right = values[lag:] - float(np.mean(values[lag:]))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return 0.0
    return _finite(float(np.dot(left, right) / denominator))


def _spectral_entropy(values: np.ndarray) -> float:
    """Normalised Shannon entropy of the periodogram: 0 = a pure tone, 1 = noise."""
    centred = values - float(np.mean(values))
    if _variance(centred) <= _EPS:
        return 0.0
    power = np.abs(np.fft.rfft(centred)) ** 2
    power = power[1:]  # drop the DC bin; the level is not shape
    total = float(power.sum())
    if total <= _EPS or power.size < 2:
        return 0.0
    density = power / total
    density = density[density > _EPS]
    entropy = -float(np.sum(density * np.log(density)))
    return _finite(entropy / float(np.log(power.size)))


def _volatility_drift(values: np.ndarray) -> float:
    """log ratio of the second half's spread to the first half's.

    Positive means the window is becoming more volatile, which is exactly the
    situation where a program tuned on the calm half can hurt.
    """
    half = values.size // 2
    if half < 4:
        return 0.0
    first = float(np.std(values[:half]))
    second = float(np.std(values[half:]))
    if first <= _EPS or second <= _EPS:
        return 0.0
    return _finite(float(np.log(second / first)))


def extract(context: np.ndarray, *, period: int) -> dict[str, float]:
    """The six descriptors for one pre-origin context window.

    ``context`` must be the window strictly before the origin; nothing in this
    module can reach a horizon value, which is what makes the set deployable.
    """
    values = np.asarray(context, dtype=np.float64).ravel()
    if values.size < 4:
        return {name: 0.0 for name in FEATURE_NAMES}
    try:
        values = np.asarray(
            forecast_runtime._linear_integrity(values), dtype=np.float64
        )
    except Exception:  # noqa: BLE001 - a context we cannot complete has no shape
        return {name: 0.0 for name in FEATURE_NAMES}
    if not np.isfinite(values).all():
        return {name: 0.0 for name in FEATURE_NAMES}

    total = _variance(values)
    detrended = _detrended(values)
    trend_strength = _strength(_variance(detrended), total)

    usable_period = int(period) if 2 <= int(period) < values.size else 0
    if usable_period:
        adjusted = _seasonally_adjusted(detrended, usable_period)
        seasonal_strength = _strength(_variance(adjusted), _variance(detrended))
        acf_at_period = _autocorrelation(values, usable_period)
    else:
        seasonal_strength = 0.0
        acf_at_period = 0.0

    return {
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "spectral_entropy": _spectral_entropy(values),
        "acf_lag_1": _autocorrelation(values, 1),
        "acf_at_period": acf_at_period,
        "volatility_drift": _volatility_drift(values),
    }


def matrix(contexts: "list[np.ndarray]", *, period: int
           ) -> tuple[np.ndarray, tuple[str, ...]]:
    """Feature matrix in the frozen column order."""
    rows = [extract(context, period=period) for context in contexts]
    return (
        np.array([[row[name] for name in FEATURE_NAMES] for row in rows],
                 dtype=np.float64),
        FEATURE_NAMES,
    )


def is_finite_everywhere(values: Mapping[str, float]) -> bool:
    return all(np.isfinite(float(value)) for value in values.values())
