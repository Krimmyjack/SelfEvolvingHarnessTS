"""Read-only structural census for clean inner-training series.

The probe exposes only input structure and sampling diagnostics.  It neither
imports outcome-scoring code nor calls a downstream trainer.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from statsmodels.tsa.seasonal import STL

from .registry import SeriesRecord

__all__ = ["ProbeError", "probe_registry", "probe_series"]


class ProbeError(ValueError):
    """The read-only probe received malformed clean-base input."""


def _sampling_diagnostics(
    timestamps: np.ndarray | None, length: int
) -> tuple[int, float]:
    if timestamps is None or length < 2:
        return 0, 0.0
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or len(raw) != length:
        raise ProbeError("timestamps must be one-dimensional and match values")
    try:
        times = raw.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ProbeError("timestamps must be datetime-like") from exc
    if np.isnat(times).any():
        raise ProbeError("timestamps must not contain NaT")
    deltas = np.diff(times.astype(np.int64))
    positive = deltas[deltas > 0]
    if positive.size == 0:
        count = int(deltas.size)
    else:
        unique, counts = np.unique(positive, return_counts=True)
        expected = unique[np.flatnonzero(counts == counts.max())[0]]
        count = int(np.count_nonzero(deltas != expected))
    return count, count / float(length - 1)


def _linear_fill(values: np.ndarray) -> np.ndarray:
    missing = np.isnan(values)
    if missing.all():
        raise ProbeError("clean inner-train cannot be all missing")
    result = values.copy()
    index = np.arange(len(result))
    result[missing] = np.interp(index[missing], index[~missing], result[~missing])
    return result


def _strength(component: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.var(component + residual))
    if not math.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return 0.0
    value = 1.0 - float(np.var(residual)) / denominator
    return float(np.clip(value, 0.0, 1.0))


def _spectral_entropy(values: np.ndarray) -> float:
    centered = values - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size:
        power = power[1:]
    total = float(power.sum())
    if power.size <= 1 or total <= np.finfo(float).eps:
        return 0.0
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.clip(entropy / math.log(power.size), 0.0, 1.0))


def probe_series(
    values: np.ndarray,
    *,
    period: int,
    inner_train_end: int | None = None,
    timestamps: np.ndarray | None = None,
    natural_missing_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    try:
        full = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ProbeError("values must be one-dimensional numeric data") from exc
    if full.size == 0 or np.isinf(full).any():
        raise ProbeError("values must be non-empty and contain no infinity")
    if isinstance(period, bool) or not isinstance(period, int) or period < 2:
        raise ProbeError("period must be an integer of at least 2")
    stop = len(full) if inner_train_end is None else inner_train_end
    if isinstance(stop, bool) or not isinstance(stop, int) or not 1 <= stop <= len(full):
        raise ProbeError("inner_train_end must select a non-empty prefix")
    if timestamps is not None:
        all_timestamps = np.asarray(timestamps)
        if all_timestamps.ndim != 1 or len(all_timestamps) != len(full):
            raise ProbeError("timestamps must be one-dimensional and match values")
        train_timestamps = all_timestamps[:stop]
    else:
        train_timestamps = None
    train = full[:stop].copy()
    actual_missing = np.isnan(train)
    if natural_missing_mask is not None:
        full_mask = np.asarray(natural_missing_mask, dtype=bool).reshape(-1)
        if len(full_mask) != len(full):
            raise ProbeError("natural_missing_mask must match values")
        if not np.array_equal(full_mask, np.isnan(full)):
            raise ProbeError("natural_missing_mask identity disagrees with values")
        mask = full_mask[:stop]
    else:
        mask = actual_missing
    irregular_count, irregular_rate = _sampling_diagnostics(
        train_timestamps, len(train)
    )
    clean = _linear_fill(train)
    if len(clean) < 2 * period + 1:
        raise ProbeError("clean inner-train is too short for the requested STL period")
    fitted = STL(clean, period=period, robust=True).fit()
    result: dict[str, float | int] = {
        "seasonal_strength": _strength(fitted.seasonal, fitted.resid),
        "trend_strength": _strength(fitted.trend, fitted.resid),
        "spectral_entropy": _spectral_entropy(clean),
        "natural_missing_count": int(mask.sum()),
        "natural_missing_rate": float(mask.mean()),
        "irregular_interval_count": irregular_count,
        "irregular_sampling_rate": irregular_rate,
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise ProbeError("probe produced a non-finite structural diagnostic")
    return result


_FREQUENCY_PERIOD = {"hourly": 24, "daily": 7, "monthly": 12}


def probe_registry(
    records: Sequence[SeriesRecord],
    values_by_uid: Mapping[str, np.ndarray],
    *,
    timestamps_by_uid: Mapping[str, np.ndarray] | None = None,
    period_by_uid: Mapping[str, int] | None = None,
    inner_train_end_by_uid: Mapping[str, int] | None = None,
) -> dict[str, dict[str, float | int]]:
    rows = list(records)
    if any(not isinstance(row, SeriesRecord) for row in rows):
        raise ProbeError("records must contain only SeriesRecord values")
    uids = [row.series_uid for row in rows]
    if len(uids) != len(set(uids)):
        raise ProbeError("registry contains duplicate series_uid values")
    missing_values = set(uids) - set(values_by_uid)
    extra_values = set(values_by_uid) - set(uids)
    if missing_values or extra_values:
        raise ProbeError(
            f"registry/value uid sets differ (missing={sorted(missing_values)}, "
            f"extra={sorted(extra_values)})"
        )
    report: dict[str, dict[str, float | int]] = {}
    for row in sorted(rows, key=lambda item: item.series_uid):
        if period_by_uid is not None and row.series_uid in period_by_uid:
            period = period_by_uid[row.series_uid]
        else:
            try:
                period = _FREQUENCY_PERIOD[row.frequency]
            except KeyError as exc:
                raise ProbeError(
                    f"no frozen period for frequency {row.frequency!r}"
                ) from exc
        timestamps = (
            None
            if timestamps_by_uid is None
            else timestamps_by_uid.get(row.series_uid)
        )
        try:
            row.verify_values(values_by_uid[row.series_uid], timestamps=timestamps)
        except ValueError as exc:
            raise ProbeError(
                f"registry identity check failed for {row.series_uid!r}: {exc}"
            ) from exc
        inner_stop = (
            None
            if inner_train_end_by_uid is None
            else inner_train_end_by_uid.get(row.series_uid)
        )
        report[row.series_uid] = probe_series(
            values_by_uid[row.series_uid],
            period=period,
            inner_train_end=inner_stop,
            timestamps=timestamps,
        )
    return report
