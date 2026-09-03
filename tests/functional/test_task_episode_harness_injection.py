"""Necessary T0 injection-contract test (final Planner ruling)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np

from evaluation.functional.task_episode_harness.injection import (
    TRAIN_ANCHORS,
    inject_label_touched_corpus,
)


def _values() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        uid: rng.normal(10.0, 2.0, size=1000)
        for uid in (
            "T117", "T118", "T119", "T12", "T120", "T121",
            "T122", "T123", "T124", "T125", "T126", "T127",
        )
    }


def test_final_injection_contracts() -> None:
    faulty = ("T117", "T118", "T119", "T12", "T120", "T121")
    clean = ("T122", "T123", "T124", "T125", "T126", "T127")
    values = _values()
    original = {uid: value.copy() for uid, value in values.items()}

    injected, ground_truth = inject_label_touched_corpus(
        values,
        faulty_series=faulty,
        clean_series=clean,
        amplitude=8.0,
        count=40,
        seed=7,
        anchors=TRAIN_ANCHORS,
    )

    # 1. exactly 40 unique positions per faulty series.
    for uid in faulty:
        positions = ground_truth[uid]["positions"]
        assert len(positions) == 40
        assert len(set(positions)) == 40
    assert set(ground_truth) == set(faulty)

    # 2. every delta depends only on the pristine scale.
    for uid in faulty:
        pristine = values[uid]
        scale = float(np.nanstd(pristine[120:900]))
        expected = [
            sign * 8.0 * scale for sign in ground_truth[uid]["signs"]
        ]
        assert np.allclose(ground_truth[uid]["deltas"], expected)

    # 3. anchor traversal order does not change the injection result.
    reversed_injected, reversed_truth = inject_label_touched_corpus(
        values,
        faulty_series=faulty,
        clean_series=clean,
        amplitude=8.0,
        count=40,
        seed=7,
        anchors=tuple(reversed(TRAIN_ANCHORS)),
    )
    for uid in faulty:
        assert np.array_equal(injected[uid], reversed_injected[uid])
        assert ground_truth[uid]["positions"] == reversed_truth[uid]["positions"]
        assert ground_truth[uid]["deltas"] == reversed_truth[uid]["deltas"]

    # 4. clean series are pointwise unchanged and inputs are not mutated.
    for uid in clean:
        assert np.array_equal(injected[uid], values[uid])
        assert np.array_equal(values[uid], original[uid])
    for uid in faulty:
        assert np.array_equal(values[uid], original[uid])
