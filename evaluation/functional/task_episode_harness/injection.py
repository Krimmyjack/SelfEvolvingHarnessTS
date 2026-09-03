"""T0 corpus-level one-shot impulsive-outlier injection (Planner final ruling).

Final frozen recipe:

* injection_unit: corpus_level_one_shot
* position_pool: union of all training-label timestamps
  (raw[anchor : anchor + 48] for the frozen training anchors)
* scale: once per pristine series, nanstd(series[120:900])
* amplitude: 8.0
* count: 40 unique timestamps per faulty series
* seed: 7
* signs: random +/-1 from the same rng
* clean train series are copied pointwise unchanged

The injection is computed from pristine copies only; previously injected values
are never re-read for scale or as the base of a later delta.
"""
from __future__ import annotations

from typing import Any

import numpy as np

HORIZON = 48
TRAIN_ANCHORS = tuple(range(312, 853, 60))
PRISTINE_SCALE_REGION = (120, 900)


def label_touched_timestamp_pool(
    anchors: tuple[int, ...] | list[int] = TRAIN_ANCHORS,
    *,
    horizon: int = HORIZON,
) -> tuple[int, ...]:
    """Union of training-label timestamps; order-independent by construction."""
    return tuple(sorted({index for anchor in anchors for index in range(anchor, anchor + horizon)}))


def inject_label_touched_corpus(
    values: dict[str, Any],
    *,
    faulty_series: tuple[str, ...],
    clean_series: tuple[str, ...],
    amplitude: float = 8.0,
    count: int = 40,
    seed: int = 7,
    anchors: tuple[int, ...] | list[int] = TRAIN_ANCHORS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One-shot corpus injection; returns (fresh_values, private_ground_truth)."""
    if count <= 0:
        raise ValueError("count must be positive")
    if amplitude <= 0.0 or not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite and positive")
    pool = label_touched_timestamp_pool(tuple(anchors))
    if len(pool) < count:
        raise ValueError("training-label timestamp pool is smaller than count")

    injected = dict(values)
    for uid in clean_series:
        injected[uid] = np.asarray(injected[uid], dtype=np.float64).copy()

    ground_truth: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for uid in faulty_series:
        pristine = np.asarray(values[uid], dtype=np.float64)
        output = pristine.copy()
        scale = float(np.nanstd(pristine[PRISTINE_SCALE_REGION[0]:PRISTINE_SCALE_REGION[1]]))
        positions = np.sort(rng.choice(pool, size=count, replace=False))
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        deltas = signs * amplitude * scale
        for index, delta in zip(positions, deltas):
            output[int(index)] += float(delta)
        if len({int(index) for index in positions}) != count:
            raise RuntimeError("injection produced duplicate positions")
        injected[uid] = output
        ground_truth[uid] = {
            "family": "impulsive_outlier",
            "injection_unit": "corpus_level_one_shot",
            "pristine_scale": scale,
            "amplitude": float(amplitude),
            "count": int(count),
            "positions": [int(index) for index in positions],
            "signs": [float(sign) for sign in signs],
            "deltas": [float(delta) for delta in deltas],
            "delta_scale_dependence": (
                "every delta = sign * amplitude * nanstd(pristine[120:900]); "
                "no injected value is ever used as a scale source"
            ),
        }
    return injected, ground_truth

def inject_gap_corpus(
    values: dict[str, Any],
    *,
    faulty_series: tuple[str, ...],
    clean_series: tuple[str, ...],
    count: int = 80,
    seed: int = 11,
    anchors: tuple[int, ...] | list[int] = TRAIN_ANCHORS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One-shot gap injection on the training-label timestamp pool.

    Missingness is written into a copy of each pristine faulty series only;
    clean series are copied pointwise unchanged.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    pool = label_touched_timestamp_pool(tuple(anchors))
    if len(pool) < count:
        raise ValueError("training-label timestamp pool is smaller than count")
    injected = dict(values)
    for uid in clean_series:
        injected[uid] = np.asarray(injected[uid], dtype=np.float64).copy()
    ground_truth: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for uid in faulty_series:
        pristine = np.asarray(values[uid], dtype=np.float64)
        output = pristine.copy()
        positions = np.sort(rng.choice(pool, size=count, replace=False))
        for index in positions:
            output[int(index)] = np.nan
        injected[uid] = output
        ground_truth[uid] = {
            "family": "gap",
            "injection_unit": "corpus_level_one_shot",
            "count": int(count),
            "positions": [int(index) for index in positions],
        }
    return injected, ground_truth

