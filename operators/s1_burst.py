"""operators/s1_burst.py — contiguous high-deviation segment repair.

CLS-4 Program Supply (2026-08-25).  Distinct from the two existing repair
families:

  * ``hampel_filter`` / ``outlier_mad`` replace or clip **points**.
  * ``repair_level_shift`` looks for a **step / two-boundary level** geometry.

This operator looks for a **contiguous burst** (|robust-z| above a frozen
threshold for a frozen minimum run length) and replaces the whole run by
linear interpolation between the adjacent intact endpoints.  No hit →
byte-identical identity.

The book freezes the two detection parameters.  They are not public and
must not be scanned.
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.contracts.observables import PUBLIC_ROBUST_Z_MAD_FLOOR

from ._common import MAD_TO_SIGMA, as_1d, interp_nan
from ._provenance import record

BURST_Z_THRESHOLD = 3.5
BURST_MIN_RUN = 8


def _robust_z(y: np.ndarray) -> np.ndarray:
    median = float(np.median(y))
    mad = float(np.median(np.abs(y - median)))
    scale = max(MAD_TO_SIGMA * mad, PUBLIC_ROBUST_Z_MAD_FLOOR)
    return (y - median) / scale


def detect_burst_segments(x) -> list[tuple[int, int]]:
    """Return half-open [start, end) runs with |robust-z| > 3.5 and length ≥ 8.

    Baseline is series-level median / MAD (the book allows this global
    simplification of a rolling robust-z).
    """
    y = interp_nan(as_1d(x))
    mask = np.abs(_robust_z(y)) > BURST_Z_THRESHOLD
    segments: list[tuple[int, int]] = []
    n = int(mask.size)
    index = 0
    while index < n:
        if not mask[index]:
            index += 1
            continue
        end = index + 1
        while end < n and mask[end]:
            end += 1
        if end - index >= BURST_MIN_RUN:
            segments.append((index, end))
        index = end
    return segments


def repair_burst_segment(x, **_) -> np.ndarray:
    """Replace each detected contiguous burst by endpoint linear interpolation."""
    y = interp_nan(as_1d(x))
    segments = detect_burst_segments(y)
    if not segments:
        record("repair_burst_segment", "repair_burst_segment", "no_burst_identity")
        return y
    marked = y.copy()
    for start, end in segments:
        marked[start:end] = np.nan
    if np.isnan(marked).all():
        record("repair_burst_segment", "repair_burst_segment", "full_span_identity")
        return y
    out = interp_nan(marked)
    n_replaced = int(sum(end - start for start, end in segments))
    record(
        "repair_burst_segment",
        "repair_burst_segment",
        "repaired_%d_points_in_%d_segments" % (n_replaced, len(segments)),
    )
    return out
