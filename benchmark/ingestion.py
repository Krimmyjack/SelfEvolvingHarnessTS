"""The single benchmark-owned finite-value ingestion rule."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["IngestionInvalid", "IngestionResult", "canonical_ingest"]


class IngestionInvalid(ValueError):
    """Prepared values cannot be converted to the canonical finite context."""


@dataclass(frozen=True)
class IngestionResult:
    values: np.ndarray
    filled_count: int
    fill_rate: float
    dependency_flag: bool


def canonical_ingest(values: np.ndarray) -> IngestionResult:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise IngestionInvalid("input must be one-dimensional")
    try:
        source = raw.astype("<f8", copy=True)
    except (TypeError, ValueError) as exc:
        raise IngestionInvalid("input must be numeric") from exc
    if source.size == 0 or np.isinf(source).any() or np.isnan(source).all():
        raise IngestionInvalid(
            "input must contain finite values, contain no infinity, and not be empty"
        )
    missing = np.isnan(source)
    result = source.copy()
    if missing.any():
        index = np.arange(result.size)
        result[missing] = np.interp(
            index[missing], index[~missing], result[~missing]
        )
    if not np.isfinite(result).all():
        raise IngestionInvalid("canonical ingestion produced non-finite values")
    filled_count = int(missing.sum())
    fill_rate = filled_count / float(result.size)
    result.setflags(write=False)
    return IngestionResult(
        values=result,
        filled_count=filled_count,
        fill_rate=fill_rate,
        dependency_flag=fill_rate > 0.01,
    )
