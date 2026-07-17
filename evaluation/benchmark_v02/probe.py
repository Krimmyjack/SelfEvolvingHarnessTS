"""Read-only structural census for clean inner-training series."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from statsmodels.tsa.seasonal import STL

from . import HEADLINE_HORIZON
from .registry import SeriesRecord

__all__ = ["ProbeError", "probe_registry", "probe_series"]


class ProbeError(ValueError):
    """The read-only probe received malformed clean-base input."""


def _strict_timestamps(timestamps: np.ndarray, length: int) -> np.ndarray:
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or len(raw) != length:
        raise ProbeError("timestamps must be one-dimensional and match values")
    try:
        times = raw.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise ProbeError("timestamps must be datetime-like") from exc
    if np.isnat(times).any():
        raise ProbeError("timestamps must not contain NaT")
    if length >= 2 and np.any(np.diff(times.astype(np.int64)) <= 0):
        raise ProbeError("timestamps must be strictly increasing")
    return times


def _sampling_diagnostics(
    timestamps: np.ndarray | None, length: int
) -> tuple[int, float]:
    if timestamps is None or length < 2:
        return 0, 0.0
    times = _strict_timestamps(timestamps, length)
    deltas = np.diff(times.astype(np.int64))
    unique, counts = np.unique(deltas, return_counts=True)
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
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ProbeError("values must be one-dimensional numeric data")
    try:
        full = raw.astype(np.float64, copy=True)
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
        all_timestamps = _strict_timestamps(timestamps, len(full))
        train_timestamps = all_timestamps[:stop]
    else:
        train_timestamps = None
    train = full[:stop].copy()
    actual_missing = np.isnan(train)
    if natural_missing_mask is not None:
        raw_mask = np.asarray(natural_missing_mask)
        if raw_mask.ndim != 1 or len(raw_mask) != len(full):
            raise ProbeError(
                "natural_missing_mask must be one-dimensional and match values"
            )
        full_mask = raw_mask.astype(bool, copy=False)
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


def _require_mapping_keys(
    value: Mapping[str, object] | None,
    expected: set[str],
    label: str,
) -> Mapping[str, object]:
    actual_mapping: Mapping[str, object] = {} if value is None else value
    if not isinstance(actual_mapping, Mapping):
        raise ProbeError(f"{label} mapping must be a mapping")
    if any(not isinstance(key, str) for key in actual_mapping):
        raise ProbeError(f"{label} mapping keys must be strings")
    actual = set(actual_mapping)
    if actual != expected:
        raise ProbeError(
            f"{label} uid sets differ (missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)})"
        )
    return actual_mapping


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
    if not isinstance(values_by_uid, Mapping):
        raise ProbeError("values_by_uid must be a mapping")
    uids = [row.series_uid for row in rows]
    if len(uids) != len(set(uids)):
        raise ProbeError("registry contains duplicate series_uid values")
    uid_set = set(uids)
    values_map = _require_mapping_keys(values_by_uid, uid_set, "registry/value")
    timestamp_uids = {row.series_uid for row in rows if row.timestamps_sha is not None}
    timestamp_map = _require_mapping_keys(
        timestamps_by_uid, timestamp_uids, "timestamp"
    )
    if period_by_uid is not None:
        period_map = _require_mapping_keys(period_by_uid, uid_set, "period")
    else:
        period_map = None
    if inner_train_end_by_uid is not None:
        inner_map = _require_mapping_keys(
            inner_train_end_by_uid, uid_set, "inner_train_end"
        )
    else:
        inner_map = None

    report: dict[str, dict[str, float | int]] = {}
    for row in sorted(rows, key=lambda item: item.series_uid):
        if period_map is not None:
            period = period_map[row.series_uid]
        else:
            try:
                period = _FREQUENCY_PERIOD[row.frequency]
            except KeyError as exc:
                raise ProbeError(
                    f"no frozen period for frequency {row.frequency!r}"
                ) from exc
        timestamps = timestamp_map.get(row.series_uid)
        try:
            row.verify_values(values_map[row.series_uid], timestamps=timestamps)
        except ValueError as exc:
            raise ProbeError(
                f"registry identity check failed for {row.series_uid!r}: {exc}"
            ) from exc
        frozen_stop = row.length - 2 * HEADLINE_HORIZON
        if frozen_stop < 1:
            raise ProbeError(
                f"series {row.series_uid!r} has no frozen inner-train prefix"
            )
        if inner_map is not None and inner_map[row.series_uid] != frozen_stop:
            raise ProbeError(
                f"inner_train_end for {row.series_uid!r} disagrees with frozen inner-train boundary"
            )
        features = probe_series(
            values_map[row.series_uid],
            period=period,
            inner_train_end=frozen_stop,
            timestamps=timestamps,
        )
        # Structure is intentionally inner-train-only. Provenance diagnostics
        # describe the full frozen clean base and must agree bit-for-bit with
        # SeriesRecord.with_probe_result even when missingness occurs later.
        features.update(
            {
                "natural_missing_count": row.natural_missing_count,
                "natural_missing_rate": row.natural_missing_rate,
                "irregular_interval_count": row.irregular_interval_count,
                "irregular_sampling_rate": row.irregular_sampling_rate,
            }
        )
        report[row.series_uid] = features
    return report
