"""#43 M0-C Part A -- the instrument gate for the low-rank reconstruction Consumer.

Synthetic fixture only; no Yahoo byte is read here.  The gate is what
authorises RANK and THRESHOLD_QUANTILE to be frozen before the Consumer
touches real data, and it asserts the three properties the book names:

  (i)   contamination injected into the *training* substrate moves the
        reconstruction boundary and the threshold measurably;
  (ii)  the Query bytes are never touched;
  (iii) the whole path is deterministic -- two runs are byte-identical.

Plus the rank justification (top-3 explained-variance ratio on a clean
substrate) and the proof that the event-scoring layer is literally the
in-service Consumer's, not a second implementation of it.

A miss here is INSTRUMENT_UNREADABLE for the round.  Re-picking the rank or
the threshold quantile on the spot to make it pass is exactly what the gate
exists to prevent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

pca = pytest.importorskip("consumers.pca_reconstruction_v1")
iforest = pytest.importorskip("consumers.aegists_iforest_v1")

SERIES_LENGTH = 600
TRAIN_CUT = 420
EVAL_REGION = (TRAIN_CUT, SERIES_LENGTH)
SPIKE_POSITIONS = (120, 121, 260, 261, 262, 330)
SPIKE_MAGNITUDE = 9.0
EXPLAINED_VARIANCE_FLOOR = 0.90
BOUNDARY_MOVE_FLOOR = 0.05  # relative threshold move the injection must cause


def _clean_series() -> np.ndarray:
    """A smooth quasi-periodic fixture: two incommensurate cycles plus noise."""
    t = np.arange(SERIES_LENGTH, dtype=np.float64)
    base = (10.0
            + 2.0 * np.sin(2.0 * np.pi * t / 24.0)
            + 0.7 * np.sin(2.0 * np.pi * t / 97.0))
    noise = np.random.default_rng(20260824).normal(0.0, 0.05, SERIES_LENGTH)
    return base + noise


def _contaminated_train(clean: np.ndarray) -> np.ndarray:
    block = clean[:TRAIN_CUT].copy()
    for position in SPIKE_POSITIONS:
        block[position] += SPIKE_MAGNITUDE
    return block


def _truth_windows() -> list[list[int]]:
    return [[500, 501, 502]]


def test_rank_three_carries_the_clean_window_family() -> None:
    """(rank basis) three modes explain the clean substrate's windows."""
    model = pca.fit_series(_clean_series()[:TRAIN_CUT])
    assert model["rank"] == pca.RANK == 3
    assert model["explained_variance_ratio"] >= EXPLAINED_VARIANCE_FLOOR


def test_training_time_alarm_budget_matches_the_iforest_parity_basis() -> None:
    """(threshold basis) the frozen quantile really is a 10% training budget."""
    model = pca.fit_series(_clean_series()[:TRAIN_CUT])
    share = model["training_flagged_windows"] / model["training_windows"]
    assert pca.THRESHOLD_QUANTILE == 0.90
    assert abs(share - (1.0 - pca.THRESHOLD_QUANTILE)) <= 0.01


def test_contamination_in_the_training_substrate_moves_the_boundary() -> None:
    """(i) injection on the training side is visible in threshold and subspace."""
    clean = _clean_series()
    clean_model = pca.fit_series(clean[:TRAIN_CUT])
    dirty_model = pca.fit_series(_contaminated_train(clean))

    relative_move = (abs(dirty_model["threshold"] - clean_model["threshold"])
                     / clean_model["threshold"])
    assert relative_move > BOUNDARY_MOVE_FLOOR

    # the fitted subspace itself moved, not just the cut point on it
    overlap = float(np.linalg.norm(
        np.asarray(clean_model["components"])
        @ np.asarray(dirty_model["components"]).T))
    assert overlap < float(np.sqrt(pca.RANK)) - 1e-6

    # and the move reaches the reading: the same Query scores differently
    clean_read = pca.score_region(clean_model, clean, *EVAL_REGION)
    dirty_read = pca.score_region(dirty_model, clean, *EVAL_REGION)
    assert not np.array_equal(clean_read["flags"], dirty_read["flags"])


def test_query_bytes_are_never_touched() -> None:
    """(ii) fit and score read the Query; they never write to it."""
    clean = _clean_series()
    before = clean.tobytes()
    model = pca.fit_series(_contaminated_train(clean))

    # a write of any kind would raise rather than pass silently
    clean.setflags(write=False)
    reading = pca.score_series(model, clean, EVAL_REGION, _truth_windows())
    clean.setflags(write=True)

    assert clean.tobytes() == before
    assert reading["scored_points"] == EVAL_REGION[1] - EVAL_REGION[0]


def test_two_runs_are_byte_identical() -> None:
    """(iii) determinism: no seed, no solver iteration, no ordering effect."""
    clean = _clean_series()
    block = _contaminated_train(clean)

    first = pca.fit_series(block)
    second = pca.fit_series(block)
    assert first["threshold"] == second["threshold"]
    assert first["rank"] == second["rank"]
    assert (np.asarray(first["center"]).tobytes()
            == np.asarray(second["center"]).tobytes())
    assert (np.asarray(first["components"]).tobytes()
            == np.asarray(second["components"]).tobytes())

    read_one = pca.score_series(first, clean, EVAL_REGION, _truth_windows())
    read_two = pca.score_series(second, clean, EVAL_REGION, _truth_windows())
    assert read_one == read_two

    region_one = pca.score_region(first, clean, *EVAL_REGION)
    region_two = pca.score_region(second, clean, *EVAL_REGION)
    assert (region_one["residuals"].tobytes()
            == region_two["residuals"].tobytes())
    assert region_one["flags"].tobytes() == region_two["flags"].tobytes()


def test_scoring_layer_is_the_in_service_one_not_a_copy() -> None:
    """One scoring semantics across the M0-C Consumers, by identity."""
    assert pca.merge_events is iforest.merge_events
    assert pca.match_events is iforest.match_events
    assert pca.event_f1 is iforest.event_f1
    assert pca.WINDOW == iforest.WINDOW == 20


def test_region_start_without_a_full_trailing_window_is_refused() -> None:
    clean = _clean_series()
    model = pca.fit_series(clean[:TRAIN_CUT])
    with pytest.raises(ValueError):
        pca.score_region(model, clean, 5, 100)
