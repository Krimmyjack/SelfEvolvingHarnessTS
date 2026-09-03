"""M0-C Part A -- the deterministic low-rank reconstruction Consumer, frozen.

Why it exists: the AD line has one in-service Consumer (a sliding-window
IsolationForest) and one supervised Consumer (a threshold head on the
task-native robust-z).  Neither is a *reconstruction* detector, and the
reconstruction family is the one the literature treats as benefiting from a
cleaned training substrate.  #43 M0-C needs a third Consumer whose inductive
bias is reconstruction so that "does the utility of a data-processing program
change sign when the Consumer protocol changes" can be read at all.

What this Consumer represents, and the limit of the claim it can carry:
it is a member of the **deterministic low-rank reconstruction family** and
nothing more.  It is not an autoencoder, it is not TimesNet, and no reading
taken with it may be extrapolated to a learned or iterative reconstruction
model.  The whole point of the frozen spec below is that every number it
produces is reproducible without a seed, an optimiser or a stopping rule.

The frozen specification (nothing below is tunable, and nothing below may be
scanned on Yahoo):

  * one model per series, fitted on that series' (possibly prepared)
    training substrate only;
  * feature = the 20-point window ending at and including t, [t-19, t] --
    the same geometry as the in-service IForest Consumer, so a difference
    between the two cannot be an artefact of differing window support;
  * scalar standardization: mean/std computed on the training substrate
    alone and applied unchanged to the untouched raw Query; a zero std is
    replaced by 1.  Identical function to the IForest Consumer's;
  * window-mean centering: the 20-vector column mean of the *training*
    window matrix; applied unchanged to the Query windows;
  * decomposition: ``numpy.linalg.svd`` -- the exact LAPACK factorisation of
    the centered training window matrix.  No randomized SVD, no truncated
    iterative solver, no power iteration, no seed;
  * rank: RANK = 3, frozen.  Basis: after column-centering a 20-point window
    of a locally smooth or quasi-periodic series, the window family is
    spanned by a small number of modes -- one level/slope-residual mode plus
    one periodic pair.  Three is the smallest rank that carries that family;
    the fixture gate asserts the top-3 explained-variance ratio clears 0.90
    on a clean synthetic substrate before the rank is used on any real data.
    The realized explained-variance ratio is reported per fit as a
    diagnostic; it is never used to re-pick the rank;
  * anomaly score of a window = the root-mean-square reconstruction residual
    ``rms(x - x_hat)`` where ``x_hat`` is the projection of x onto the frozen
    rank-3 affine subspace.  Higher = more anomalous;
  * threshold: the ``THRESHOLD_QUANTILE = 0.90`` quantile of the *training
    substrate's* residual distribution, frozen at fit time and never
    refitted on the Query.  Basis: the in-service IForest Consumer's
    ``contamination = 0.1`` fixes its training-time alarm budget at 10%;
    matching that budget here means a delta between the two Consumers cannot
    be an artefact of one of them simply alarming more often;
  * a Query point flags when its window residual is strictly greater than
    the frozen threshold.  The Query bytes are read and never written;
  * event merging, event matching, event F1, pointwise AUPRC and the
    background alarm rate are **not re-implemented here**.  They are the
    exact functions the in-service Consumer uses, imported from
    ``consumers.aegists_iforest_v1``, so the three Consumers of #43 M0-C
    score under one scoring semantics and their deltas are comparable.

Determinism: no RNG is constructed anywhere in this module, no solver
iterates, and no input depends on dictionary or file iteration order, so
re-running the same cell returns byte-identical output.

Dependency note: importing the shared scoring layer means this module
inherits ``aegists_iforest_v1``'s scikit-learn import, and therefore raises
``ConsumerDependencyUnavailable`` when scikit-learn is missing.  That
coupling is deliberate -- sharing the scoring code is worth more than
independence from a dependency the round already requires.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from consumers import aegists_iforest_v1 as _shared
from consumers.aegists_iforest_v1 import (  # noqa: F401  (re-exported)
    ConsumerDependencyUnavailable,
    auprc,
    background_alarm_rate,
    event_f1,
    macro_f1,
    match_events,
    merge_events,
    pooled_f1,
    standardization,
)

CONSUMER_ID = "pca_reconstruction_v1"
TASK = "anomaly_detection"

WINDOW = 20
RANK = 3
THRESHOLD_QUANTILE = 0.90
QUANTILE_METHOD = "linear"
MATERIAL_THRESHOLD = 0.005

CONSUMER_SPEC: dict[str, Any] = {
    "consumer_id": CONSUMER_ID,
    "task": TASK,
    "family": "deterministic low-rank reconstruction",
    "family_claim_cap": (
        "readings taken with this Consumer speak for the deterministic "
        "low-rank reconstruction family only; they do not extrapolate to "
        "autoencoders, TimesNet or any learned/iterative reconstruction model"
    ),
    "detector": "sliding-window low-rank projection, one model per series",
    "window": WINDOW,
    "window_geometry": "[t-19, t], the current point included",
    "standardization": (
        "scalar mean/std from the training substrate only, applied unchanged "
        "to the untouched Query; std == 0 is replaced by 1 -- the same "
        "function the in-service IForest Consumer uses"
    ),
    "centering": "20-vector column mean of the training window matrix",
    "decomposition": (
        "numpy.linalg.svd, the exact LAPACK factorisation; no randomized "
        "SVD, no truncated iterative solver, no power iteration, no seed"
    ),
    "rank": RANK,
    "rank_basis": (
        "a column-centered 20-point window of a locally smooth or "
        "quasi-periodic series is spanned by a level/slope-residual mode "
        "plus one periodic pair; rank 3 is the smallest rank carrying that "
        "family.  Fixture-gated at explained-variance ratio >= 0.90 on a "
        "clean synthetic substrate, then frozen -- never scanned on Yahoo"
    ),
    "score": "root-mean-square reconstruction residual; higher = more anomalous",
    "threshold": (
        "the %.2f quantile of the training substrate's residual "
        "distribution, frozen at fit time; never refitted on the Query"
        % THRESHOLD_QUANTILE
    ),
    "threshold_basis": (
        "parity with the in-service IForest Consumer's contamination = 0.1: "
        "both Consumers carry the same 10% frozen training-time alarm "
        "budget, so a delta between them is not an alarm-rate artefact"
    ),
    "query_rule": "residual > frozen threshold; the Query bytes are never written",
    "scoring_layer": (
        "merge_events / match_events / event_f1 / auprc / "
        "background_alarm_rate imported from consumers.aegists_iforest_v1 -- "
        "not re-implemented, so all three M0-C Consumers score identically"
    ),
    "primary_metric": "macro average of per-series event F1",
    "material_threshold": MATERIAL_THRESHOLD,
    "deterministic": True,
}


# --------------------------------------------------------------- windowing
def _windows(values: Any) -> np.ndarray:
    """Every 20-point window ending at each position from WINDOW-1 onward."""
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size < WINDOW:
        return np.empty((0, WINDOW), dtype=np.float64)
    count = array.size - WINDOW + 1
    strided = np.lib.stride_tricks.sliding_window_view(array, WINDOW)
    assert strided.shape == (count, WINDOW)
    return np.ascontiguousarray(strided, dtype=np.float64)


def _apply(values: Any, constants: Mapping[str, float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return (array - float(constants["mean"])) / float(constants["std"])


def _residuals(matrix: np.ndarray, center: np.ndarray,
               components: np.ndarray) -> np.ndarray:
    """RMS reconstruction residual of every row against the frozen subspace."""
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    centered = matrix - center
    if components.shape[0] == 0:
        reconstructed = np.zeros_like(centered)
    else:
        reconstructed = (centered @ components.T) @ components
    difference = centered - reconstructed
    return np.sqrt(np.mean(np.square(difference), axis=1))


# ------------------------------------------------------------------- model
def fit_series(train_block: Any) -> dict[str, Any]:
    """One model on one series' (possibly prepared) training substrate.

    Every quantity that survives into the Query path -- the scalar
    standardization constants, the window mean, the rank-3 basis and the
    threshold -- is estimated here and here only.
    """
    constants = standardization(train_block)
    matrix = _windows(_apply(train_block, constants))
    if matrix.shape[0] == 0:
        raise ValueError(
            "training block shorter than the %d-point window" % WINDOW)
    center = np.ascontiguousarray(np.mean(matrix, axis=0), dtype=np.float64)
    # Exact LAPACK SVD of the centered training window matrix.  full_matrices
    # is False only to skip building the m x m left factor that is never read;
    # the right singular vectors returned are the complete set for the row
    # space, which is what the projection uses.
    _u, singular, vt = np.linalg.svd(matrix - center, full_matrices=False)
    rank = int(min(RANK, vt.shape[0]))
    components = np.ascontiguousarray(vt[:rank], dtype=np.float64)
    spectrum = np.square(np.asarray(singular, dtype=np.float64))
    total = float(np.sum(spectrum))
    explained = float(np.sum(spectrum[:rank]) / total) if total > 0.0 else 1.0
    residuals = _residuals(matrix, center, components)
    threshold = float(np.quantile(
        residuals, THRESHOLD_QUANTILE, method=QUANTILE_METHOD))
    center.setflags(write=False)
    components.setflags(write=False)
    return {
        "constants": dict(constants),
        "center": center,
        "components": components,
        "rank": rank,
        "threshold": threshold,
        "explained_variance_ratio": explained,
        "training_windows": int(matrix.shape[0]),
        "training_residual_mean": float(np.mean(residuals)),
        "training_flagged_windows": int(
            np.count_nonzero(residuals > threshold)),
    }


def score_region(
    model: Mapping[str, Any],
    raw_series: Any,
    lo: int,
    hi: int,
) -> dict[str, Any]:
    """Score [lo, hi) of the *raw* series.

    The 19 points preceding ``lo`` are read so every region point has a full
    trailing window, and they are read from the raw series -- never from the
    prepared block -- for the same reason the IForest Consumer does it: a
    prepared trailing window would leak the program's effect into the Query
    features through the back door.  Nothing here writes to ``raw_series``.
    """
    array = np.asarray(raw_series, dtype=np.float64).ravel()
    lo, hi = int(lo), int(hi)
    if lo < WINDOW - 1:
        raise ValueError(
            "region start %d leaves no full trailing window" % lo)
    if hi > array.size:
        raise ValueError("region end %d beyond the series" % hi)
    fed = array[lo - (WINDOW - 1):hi]
    matrix = _windows(_apply(fed, model["constants"]))
    residuals = _residuals(
        matrix,
        np.asarray(model["center"], dtype=np.float64),
        np.asarray(model["components"], dtype=np.float64),
    )
    indices = np.arange(lo, hi, dtype=np.int64)
    assert residuals.size == indices.size
    flags = residuals > float(model["threshold"])
    return {
        "indices": indices,
        "residuals": residuals,
        # higher = more anomalous, for the ranking metric
        "anomaly_scores": residuals,
        "flags": flags,
        "flagged_points": int(np.count_nonzero(flags)),
        "scored_points": int(indices.size),
    }


def score_series(
    model: Mapping[str, Any],
    raw_series: Any,
    region: tuple[int, int],
    truth_windows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """One series' full reading on one Query region.

    Same signature, same truth representation and the same event arithmetic
    as ``aegists_iforest_v1.score_series``; the only difference is which
    detector produced the flags.
    """
    scored = score_region(model, raw_series, region[0], region[1])
    predicted = merge_events(scored["indices"], scored["flags"])
    lo, hi = int(region[0]), int(region[1])
    truth = [sorted(int(r) for r in rows) for rows in truth_windows
             if any(lo <= int(r) < hi for r in rows)]
    truth = [[r for r in rows if lo <= r < hi] for rows in truth]
    row = dict(event_f1(truth, predicted))
    labels = _shared._point_labels(scored["indices"], truth)
    row.update({
        "region": [int(region[0]), int(region[1])],
        "predicted_event_spans": [[int(s), int(e)] for s, e in predicted],
        "truth_event_rows": [[int(r) for r in rows] for rows in truth],
        "flagged_points": scored["flagged_points"],
        "scored_points": scored["scored_points"],
        "auprc": auprc(scored["anomaly_scores"], labels),
        "background_alarm_rate": background_alarm_rate(
            scored["indices"], scored["flags"], truth),
        "zero_scale": bool(model["constants"].get("zero_scale")),
        "rank": int(model["rank"]),
        "explained_variance_ratio": float(model["explained_variance_ratio"]),
        "threshold": float(model["threshold"]),
    })
    return row


def spec() -> dict[str, Any]:
    return dict(CONSUMER_SPEC)


__all__ = [
    "CONSUMER_ID",
    "CONSUMER_SPEC",
    "ConsumerDependencyUnavailable",
    "MATERIAL_THRESHOLD",
    "QUANTILE_METHOD",
    "RANK",
    "TASK",
    "THRESHOLD_QUANTILE",
    "WINDOW",
    "auprc",
    "background_alarm_rate",
    "event_f1",
    "fit_series",
    "macro_f1",
    "match_events",
    "merge_events",
    "pooled_f1",
    "score_region",
    "score_series",
    "spec",
    "standardization",
]
