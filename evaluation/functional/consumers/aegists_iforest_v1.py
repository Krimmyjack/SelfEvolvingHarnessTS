"""T6 / #42 Part B -- the natural AD Consumer, frozen.

Structure borrowed from a-evolve/AegisTS/Error_Detection/detectors/models/
IForest.py: a sliding window over the series, an IsolationForest over those
windows.  Two things are deliberately NOT borrowed:

  * the model-selection stack around it (no candidate detectors, no scoring
    of several configurations and keeping the best) -- there is exactly one
    frozen configuration here;
  * the preprocessing asymmetry.  The reference z-scores whatever matrix it
    is handed, so the training matrix and the prediction matrix each get
    their own transform.  A Consumer used to read whether *preparing the
    training data* helped cannot do that: it would fold the effect of the
    preparation into the query transform and report a change that never
    reached the query.  Here mean/std come only from P(B), and the very same
    numbers are applied to the untouched Query.

The frozen specification (nothing below is tunable):

  * one model per series, trained on that series' training block only;
  * feature = the 20-point window ending at and including t, [t-19, t];
  * standardization: mean/std computed on P(B) alone, applied unchanged to
    the raw Query; a zero std is replaced by 1;
  * IsolationForest(n_estimators=100, contamination=0.1, max_features=1.0,
    n_jobs=1, random_state=0);
  * a Query point is anomalous when the estimator's own decision_function
    is < 0 -- that is the threshold frozen at fit time by contamination, not
    a threshold refitted on the Query;
  * consecutive anomalous points merge into one predicted event;
  * predicted events are matched to the NAB official anomaly windows greedily,
    one-to-one, by overlap on rows (a window is the set of rows whose
    timestamp falls inside it -- #42a's row-order contract);
  * primary reading = macro average of per-series event F1;
  * secondary readings = pooled event F1, pointwise AUPRC, background alarm
    rate.

Edge cases, stated rather than left to emerge: a series with no true event
and no predicted event scores F1 = 1; no true event but at least one
predicted event scores F1 = 0; true events with nothing predicted scores 0.

Determinism: random_state is fixed, n_jobs=1, and no input depends on
iteration order, so a re-run of the same cell returns the identical numbers.

If scikit-learn is unavailable this module raises ConsumerDependencyUnavailable
at import time.  The caller reports CONSUMER_DEPENDENCY_UNAVAILABLE; swapping
in another model on the spot would silently change what the round measured.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class ConsumerDependencyUnavailable(RuntimeError):
    """scikit-learn is missing; the frozen Consumer cannot be built."""


try:
    from sklearn.ensemble import IsolationForest
except Exception as exc:  # noqa: BLE001
    raise ConsumerDependencyUnavailable(
        "scikit-learn is required for %s: %s" % (__name__, exc)
    ) from exc


CONSUMER_ID = "aegists_iforest_v1"
TASK = "anomaly_detection"

WINDOW = 20
FOREST_KWARGS: dict[str, Any] = {
    "n_estimators": 100,
    "contamination": 0.1,
    "max_features": 1.0,
    "n_jobs": 1,
    "random_state": 0,
}
DECISION_THRESHOLD = 0.0  # decision_function < 0 -> anomalous
MATERIAL_THRESHOLD = 0.005

CONSUMER_SPEC: dict[str, Any] = {
    "consumer_id": CONSUMER_ID,
    "task": TASK,
    "detector": "sliding-window IsolationForest, one model per series",
    "window": WINDOW,
    "window_geometry": "[t-19, t], the current point included",
    "standardization": (
        "scalar mean/std from P(B) only, applied unchanged to the untouched "
        "Query; std == 0 is replaced by 1"
    ),
    "forest": dict(FOREST_KWARGS),
    "query_rule": (
        "decision_function(window) < 0, the threshold frozen at fit time by "
        "contamination; never refitted on the Query"
    ),
    "event_rule": "consecutive anomalous points merge into one predicted event",
    "matching": ("greedy one-to-one by row overlap against NAB windows; a window is the set of rows whose timestamp falls inside it"),
    "primary_metric": "macro average of per-series event F1",
    "secondary_metrics": ["pooled_event_f1", "pointwise_auprc",
                          "background_alarm_rate"],
    "material_threshold": MATERIAL_THRESHOLD,
    "borrowed_from": (
        "a-evolve/AegisTS/Error_Detection/detectors/models/IForest.py -- "
        "window+IsolationForest structure only; not its model-selection "
        "stack, and not its train/predict normalization asymmetry"
    ),
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


def standardization(block: Any) -> dict[str, float]:
    """The transform, and it is the only one: computed on P(B), used on Q."""
    array = np.asarray(block, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"mean": 0.0, "std": 1.0, "zero_scale": True}
    mean = float(np.mean(finite))
    std = float(np.std(finite))
    if not np.isfinite(std) or std == 0.0:
        return {"mean": mean, "std": 1.0, "zero_scale": True}
    return {"mean": mean, "std": std, "zero_scale": False}


def _apply(values: Any, constants: Mapping[str, float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return (array - float(constants["mean"])) / float(constants["std"])


# ------------------------------------------------------------------- model
def fit_series(train_block: Any) -> dict[str, Any]:
    """One model on one series' (possibly prepared) training block."""
    constants = standardization(train_block)
    matrix = _windows(_apply(train_block, constants))
    if matrix.shape[0] == 0:
        raise ValueError(
            "training block shorter than the %d-point window" % WINDOW)
    forest = IsolationForest(**FOREST_KWARGS)
    forest.fit(matrix)
    return {
        "forest": forest,
        "constants": dict(constants),
        "training_windows": int(matrix.shape[0]),
    }


# #42i Part B (r1) -- Consumer-conditioned fit policy, NOT an operator.
# contamination_mask_refit_v1 is window-level masking, not point-level:
# the same standardization constants computed on P(B) are used for both
# fits; the first forest scores its own training windows; the <=1%
# highest-anomaly-score windows are dropped from the FIT matrix only;
# the raw series and the Query are never touched.  One execution = 2 fits.
# v1 rules (binding):
#   * no point-level inverse projection, no neighbourhood extension
#   * no NaN insertion, no row deletion on the raw series
#   * single iteration -- the refit forest does not mask again
#   * parameters fully fixed (MASK_REFIT_FRACTION, MASK_REFIT_SEED)
#   * only wired into the six-program census, NOT the Fast Agent menu
#   * waits for #42j to prove safe headroom before any formal entry
CONTAMINATION_MASK_REFIT_ID = "contamination_mask_refit_v1"
MASK_REFIT_FRACTION = 0.01
MASK_REFIT_SEED = 0  # forest random_state is already fixed in FOREST_KWARGS


def fit_series_with_contamination_mask(
    train_block: Any,
    *,
    mask_fraction: float = MASK_REFIT_FRACTION,
) -> dict[str, Any]:
    """Window-level contamination mask refit (single iteration).

    Returns a model dict with the same shape as ``fit_series`` plus bookkeeping
    (``dropped_windows``, ``mask_fraction_used``, ``refit``).  The returned
    ``constants`` are byte-for-byte the same as ``fit_series``'s, so the
    Query-side scoring path is unchanged.  ``train_block`` is never mutated
    -- no NaN insertion, no row deletion.
    """
    constants = standardization(train_block)  # computed once, reused
    matrix = _windows(_apply(train_block, constants))
    if matrix.shape[0] == 0:
        raise ValueError(
            "training block shorter than the %d-point window" % WINDOW)
    # First fit on the full window matrix.
    first_forest = IsolationForest(**FOREST_KWARGS)
    first_forest.fit(matrix)
    # Score the training windows with the first forest's own decision_function.
    # anomaly_scores convention: higher = more anomalous (see score_region).
    train_decision = np.asarray(
        first_forest.decision_function(matrix), dtype=np.float64)
    train_anomaly = -train_decision  # higher = more anomalous
    n_windows = int(matrix.shape[0])
    # mask_fraction is a HARD upper bound (r1: <=1%).  Enforced here.
    if mask_fraction < 0.0 or mask_fraction > MASK_REFIT_FRACTION:
        raise ValueError(
            "mask_fraction must be in [0, %s]; got %r"
            % (MASK_REFIT_FRACTION, mask_fraction))
    n_drop = int(math.floor(float(mask_fraction) * n_windows))
    if n_drop > 0:
        # Stable argpartition so ties are broken by original window order.
        # Highest anomaly scores first.
        top_idx = np.argsort(-train_anomaly, kind="stable")[:n_drop]
        drop_mask = np.zeros(n_windows, dtype=bool)
        drop_mask[top_idx] = True
        keep_mask = ~drop_mask
        reduced = matrix[keep_mask]
    else:
        drop_mask = np.zeros(n_windows, dtype=bool)
        keep_mask = np.ones(n_windows, dtype=bool)
        reduced = matrix
    if reduced.shape[0] == 0:
        # Pathological: every window would be masked.  Refuse to return a
        # zero-row forest -- the caller should not call this policy on a
        # block whose window count is below the masking floor.
        raise ValueError(
            "contamination mask would drop every window (n=%d, drop=%d)"
            % (n_windows, n_drop))
    refit_forest = IsolationForest(**FOREST_KWARGS)
    refit_forest.fit(reduced)
    return {
        "forest": refit_forest,
        "constants": dict(constants),
        "training_windows": int(reduced.shape[0]),
        "first_forest_windows": n_windows,
        "dropped_windows": int(n_drop),
        "mask_fraction_used": float(mask_fraction),
        "drop_indices": tuple(int(i) for i in np.flatnonzero(drop_mask)),
        "refit": True,
        "policy_id": CONTAMINATION_MASK_REFIT_ID,
    }


def consumer_id_for(program: str) -> str | None:
    """Resolve a runner-side program id to a Consumer-conditioned fit policy.

    Returns the policy id (``contamination_mask_refit_v1``) if the program
    name is recognised as a fit policy, otherwise None -- meaning the runner
    should treat ``program`` as an array-transformation operator and route
    it through ``fit_series`` on the prepared block.  This keeps the operator
    registry the single source of truth for array transformations; fit
    policies live next to the Consumer that owns the fit step.
    """
    raw = str(program or "").strip()
    if raw == CONTAMINATION_MASK_REFIT_ID:
        return CONTAMINATION_MASK_REFIT_ID
    return None



def score_region(
    model: Mapping[str, Any],
    raw_series: Any,
    lo: int,
    hi: int,
) -> dict[str, Any]:
    """Score [lo, hi) of the *raw* series.

    The 19 points preceding ``lo`` are read so every region point has a full
    trailing window.  Those points are read from the raw series, never from
    the prepared block: the Query is untouched by construction, and its
    trailing geometry has to be untouched too, or the program's effect would
    leak into the query features through the back door.
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
    scores = np.asarray(
        model["forest"].decision_function(matrix), dtype=np.float64)
    indices = np.arange(lo, hi, dtype=np.int64)
    assert scores.size == indices.size
    flags = scores < DECISION_THRESHOLD
    return {
        "indices": indices,
        "decision_scores": scores,
        # higher = more anomalous, for the ranking metric
        "anomaly_scores": -scores,
        "flags": flags,
        "flagged_points": int(np.count_nonzero(flags)),
        "scored_points": int(indices.size),
    }


# ------------------------------------------------------------------ events
def merge_events(indices: Any, flags: Any) -> list[tuple[int, int]]:
    """Consecutive flagged points become one predicted event [start, end]."""
    idx = np.asarray(indices, dtype=np.int64)
    flag = np.asarray(flags, dtype=bool)
    events: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for position, value in zip(idx.tolist(), flag.tolist()):
        if value:
            if start is None:
                start = position
            elif previous is not None and position != previous + 1:
                events.append((start, previous))
                start = position
            previous = position
        else:
            if start is not None and previous is not None:
                events.append((start, previous))
            start = None
            previous = None
    if start is not None and previous is not None:
        events.append((start, previous))
    return events


def match_events(
    truth: Sequence[Sequence[int]],
    predicted: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Greedy one-to-one matching by overlap on rows.

    A truth event is the set of rows whose timestamp falls inside one
    official window; a predicted event is a contiguous run of flagged rows.
    They overlap when the run contains at least one of the window's rows.
    Representing truth as a row *set* rather than a row span is what makes
    this survive a duplicated or backward timestamp: a window whose rows are
    not contiguous, or whose two rows share a stamp, still names exactly the
    rows it names.

    Predicted events are taken in start order; each takes the first still
    unmatched truth event it overlaps.  Greedy and order-fixed, so the result
    does not depend on iteration order.
    """
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    truth_sets = [frozenset(int(r) for r in rows) for rows in truth]
    for p_index, (p_start, p_end) in enumerate(
            sorted(predicted, key=lambda e: (e[0], e[1]))):
        span = range(int(p_start), int(p_end) + 1)
        for t_index, rows in enumerate(truth_sets):
            if t_index in used:
                continue
            if any(row in rows for row in span):
                used.add(t_index)
                pairs.append((p_index, t_index))
                break
    return pairs


def event_f1(
    truth: Sequence[Sequence[int]],
    predicted: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    matched = match_events(truth, predicted)
    n_truth, n_pred, n_hit = len(truth), len(predicted), len(matched)
    if n_truth == 0:
        # nothing to find: a silent detector is exactly right, a noisy one
        # is exactly wrong.  Stated, not left to a division by zero.
        f1 = 1.0 if n_pred == 0 else 0.0
        precision = 1.0 if n_pred == 0 else 0.0
        recall = 1.0
    elif n_pred == 0 or n_hit == 0:
        f1, precision, recall = 0.0, 0.0, 0.0
    else:
        precision = n_hit / n_pred
        recall = n_hit / n_truth
        f1 = (2.0 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "truth_events": n_truth,
        "predicted_events": n_pred,
        "matched_events": n_hit,
    }


# ----------------------------------------------------------------- metrics
def _point_labels(indices: Any,
                  truth: Sequence[Sequence[int]]) -> np.ndarray:
    """A row is positive when it is one of the rows an official window names.

    Membership, not range containment: with row order no longer assumed
    monotonic, a range test would sweep in rows whose timestamps sit outside
    the window entirely.
    """
    idx = np.asarray(indices, dtype=np.int64)
    positive: set[int] = set()
    for rows in truth:
        positive.update(int(r) for r in rows)
    if not positive:
        return np.zeros(idx.size, dtype=np.int64)
    return np.fromiter((1 if int(i) in positive else 0 for i in idx.tolist()),
                       dtype=np.int64, count=idx.size)


def auprc(scores: Any, labels: Any) -> float | None:
    """Pointwise average precision over the ranking, no sklearn metric call.

    Same step-wise summation the AD line already uses elsewhere, so the
    number is comparable to the other Consumers' AUPRC readings.
    """
    score = np.asarray(scores, dtype=np.float64)
    label = np.asarray(labels, dtype=np.int64)
    if score.size == 0 or int(np.sum(label)) == 0:
        return None
    order = np.argsort(-score, kind="stable")
    sorted_labels = label[order]
    cumulative_hits = np.cumsum(sorted_labels)
    ranks = np.arange(1, sorted_labels.size + 1, dtype=np.float64)
    precision = cumulative_hits / ranks
    recall = cumulative_hits / float(np.sum(label))
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    hits = sorted_labels.astype(bool)
    return float(np.sum((recall - previous_recall)[hits] * precision[hits]))


def background_alarm_rate(indices: Any, flags: Any,
                          truth: Sequence[Sequence[int]]) -> float | None:
    """Share of the points outside every truth window that were flagged.

    Denominator frozen by the #42a book: the scorable points that lie outside
    every official window.
    """
    labels = _point_labels(indices, truth)
    background = labels == 0
    total = int(np.count_nonzero(background))
    if total == 0:
        return None
    flagged = np.asarray(flags, dtype=bool)
    return float(np.count_nonzero(flagged & background) / total)


def score_series(
    model: Mapping[str, Any],
    raw_series: Any,
    region: tuple[int, int],
    truth_windows: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """One series' full reading on one Query region.

    ``truth_windows`` is a sequence of row-index sets, one per official
    anomaly window (see match_events).  A window is in scope for this region
    when at least one of its rows is scored here.
    """
    scored = score_region(model, raw_series, region[0], region[1])
    predicted = merge_events(scored["indices"], scored["flags"])
    lo, hi = int(region[0]), int(region[1])
    truth = [sorted(int(r) for r in rows) for rows in truth_windows
             if any(lo <= int(r) < hi for r in rows)]
    # only the part of a window that this region actually scores counts
    truth = [[r for r in rows if lo <= r < hi] for rows in truth]
    row = dict(event_f1(truth, predicted))
    labels = _point_labels(scored["indices"], truth)
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
    })
    return row


def macro_f1(per_series: Mapping[str, Mapping[str, Any]]) -> float | None:
    values = [float(row["f1"]) for row in per_series.values()
              if row.get("f1") is not None]
    if not values:
        return None
    return float(np.mean(values))


def pooled_f1(per_series: Mapping[str, Mapping[str, Any]]) -> float | None:
    truth = sum(int(row["truth_events"]) for row in per_series.values())
    pred = sum(int(row["predicted_events"]) for row in per_series.values())
    hit = sum(int(row["matched_events"]) for row in per_series.values())
    if truth == 0:
        return 1.0 if pred == 0 else 0.0
    if pred == 0 or hit == 0:
        return 0.0
    precision = hit / pred
    recall = hit / truth
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def spec() -> dict[str, Any]:
    return dict(CONSUMER_SPEC)


__all__ = [
    "CONSUMER_ID",
    "CONSUMER_SPEC",
    "CONTAMINATION_MASK_REFIT_ID",
    "ConsumerDependencyUnavailable",
    "MASK_REFIT_FRACTION",
    "MASK_REFIT_SEED",
    "MATERIAL_THRESHOLD",
    "TASK",
    "WINDOW",
    "auprc",
    "background_alarm_rate",
    "consumer_id_for",
    "event_f1",
    "fit_series",
    "fit_series_with_contamination_mask",
    "macro_f1",
    "match_events",
    "merge_events",
    "pooled_f1",
    "score_region",
    "score_series",
    "spec",
    "standardization",
]
