"""AD Consumer v2 (trainable): a weighted closed-form ridge event classifier.

The T1b question is whether the task flip survives when both downstream
Consumers *train* on the same P(training block) and are scored on a fixed,
independent, unprocessed Query.  The T0 detector (anomaly_detection_v1) fits
nothing, so it cannot answer a training-side question; this Consumer can.  It
stays deterministic end to end: closed form, no solver iteration, no seed.

The whole module is frozen instrument.  Every constant is pre-registered;
the only permitted movement is the single fallback in ``FALLBACK_SPEC``,
which may be taken at most once and is then frozen in the artifact for good.
The event-level scoring (run merging, greedy one-to-one matching at +-3,
F1) is not re-implemented here: it is imported from the frozen T0 scoring
parts in ``anomaly_detection_v1`` so the two Consumers differ only in how a
point score is produced.

What is deliberately fixed, and why
-----------------------------------
*Feature.*  For index ``t`` the feature is the strictly trailing 49-point
window ``x[t-49:t]`` -- history only, the same trailing geometry the T0
detector justified (a centred window lets a burst mask itself).

*Standardization.*  Per series, by the median and 1.4826 * MAD of that
series' own P(training block) -- deterministic, computed from the trained
bytes themselves.  Query features use the same constants: the Consumer may
carry to deployment only what it learned from the bytes it was trained on.

*Classifier.*  Weighted ridge, closed form, mirroring the repository's own
``evaluation/benchmark_v02/trainers.fit_closed_form`` gram/solve pattern
(same weighted normal equations, same unpenalized trailing intercept).
Positive-class weight = N_neg / N_pos computed on the pooled training set;
alpha = 1.0.  The score is the raw linear output; a point is flagged when
score > 0.5.

*AUPRC.*  Threshold-free secondary metric from the raw scores: the
right-continuous step area under the precision-recall curve, walking the
score ranking from the top (AP-style; ties broken deterministically by
index).  Reported per series and as the per-series mean; never a gate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from consumers import anomaly_detection_v1 as ad

CONSUMER_ID = "anomaly_detection_trainable_v1"
TASK = "anomaly_detection"

# ---- classifier, frozen -----------------------------------------------------
FEATURE_WINDOW = 49
ALPHA = 1.0
DECISION_THRESHOLD = 0.5
MAD_TO_SIGMA = 1.4826
CLASSIFIER_SPEC: dict[str, Any] = {
    "classifier": "weighted closed-form ridge over trailing lag windows",
    "feature": "x[t-49:t], strictly trailing, x_t excluded (the label point)",
    "feature_window": FEATURE_WINDOW,
    "standardization": (
        "per series: (x - median) / (1.4826 * MAD) with both constants taken "
        "from that series' own P(training block); the same constants "
        "standardize the Query features"
    ),
    "label": "1 at every ledger event point (all 3 points of a burst), else 0",
    "positive_class_weight": "N_neg / N_pos, computed on the pooled training set",
    "alpha": ALPHA,
    "solver": (
        "weighted normal equations, np.linalg.solve, mirroring "
        "evaluation/benchmark_v02/trainers.fit_closed_form (intercept column "
        "last and unpenalized)"
    ),
    "decision": "score > 0.5 flags the point",
    "deterministic": True,
}

# ---- the one pre-registered fallback ---------------------------------------
FALLBACK_FEATURE_WINDOW = 25
FALLBACK_SPEC: dict[str, Any] = {
    "feature_window": FALLBACK_FEATURE_WINDOW,
    "budget": (
        "may be taken at most once, and only after the primary spec fails the "
        "readability gate on the calibration Query"
    ),
    "on_use": "the chosen window freezes into the artifact; nothing else moves",
}

# ---- scoring, inherited frozen ----------------------------------------------
SCORE_SPEC: dict[str, Any] = {
    "unit": "event",
    "event_merging": "maximal runs of consecutive flagged points (T0)",
    "matching": "greedy nearest-distance one-to-one at +-3 (T0)",
    "main_metric": "Query event F1",
    "secondary_metric": "AUPRC from the raw scores, threshold-free",
    "precision_is_a_lower_bound": True,
    "scoring_parts": (
        "anomaly_detection_v1.predicted_events / match_events / score_events, "
        "imported, not re-implemented"
    ),
}


# ---- standardization --------------------------------------------------------
def standardization_constants(block: Any) -> dict[str, float]:
    """median / 1.4826*MAD of one series' P(training block) -- the trained
    bytes themselves.  A zero scale under both MAD and std is a spec defect,
    not a tuning opportunity."""
    array = np.asarray(block, dtype=np.float64).ravel()
    finite = array[np.isfinite(array)]
    centre = float(np.median(finite))
    scale = MAD_TO_SIGMA * float(np.median(np.abs(finite - centre)))
    source = "mad"
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(finite))
        source = "std"
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("standardization scale is zero under both MAD and std")
    return {"median": centre, "scale": float(scale), "source": source}


# ---- features ---------------------------------------------------------------
def features_for_range(
    values: Any,
    lo: int,
    hi: int,
    *,
    window: int,
    median: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardized trailing-window features for t in [lo, hi).

    ``values`` is the full array the range is read from (training block for
    training, the unprocessed Query copy for scoring); the window may reach
    back before ``lo`` inside that same array.  Returns (X, indices) with
    X[i] = standardized values[t-window:t].
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    lo, hi, window = int(lo), int(hi), int(window)
    if lo < window:
        raise ValueError("range start %d leaves no full trailing window" % lo)
    if hi > array.size:
        raise ValueError("range end %d beyond array" % hi)
    indices = np.arange(lo, hi, dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(array, window)
    # windows[i] = array[i : i + window]; the window for index t is windows[t-window].
    x = windows[indices - window]
    x = (x - float(median)) / float(scale)
    return np.asarray(x, dtype=np.float64), indices


# ---- the closed-form fit ----------------------------------------------------
def fit(X: Any, y: Any) -> dict[str, Any]:
    """Weighted ridge, closed form, mirroring trainers.fit_closed_form."""
    phi = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).ravel()
    if phi.ndim != 2 or phi.shape[0] != target.size or phi.shape[0] == 0:
        raise ValueError("feature/label shape mismatch or empty training set")
    n_pos = int(np.count_nonzero(target > 0.5))
    n_neg = int(target.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training set must carry both classes")
    positive_weight = float(n_neg) / float(n_pos)
    weights = np.where(target > 0.5, positive_weight, 1.0)
    design = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
    gram = design.T @ (weights[:, None] * design)
    # trainers.fit_closed_form writes weights[:, None] * batch.y, but batch.y is
    # 2-D there; with a 1-D point label that idiom broadcasts to (n, n).  Here
    # the target is 1-D, so the weighting is the plain elementwise product.
    cross = design.T @ (weights * target)
    ridge = np.eye(design.shape[1], dtype=np.float64)
    ridge[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + float(ALPHA) * ridge, cross)
    coefficients.setflags(write=False)
    return {
        "coefficients": coefficients,
        "positive_weight": positive_weight,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "alpha": float(ALPHA),
    }


def score(model: Mapping[str, Any], X: Any) -> np.ndarray:
    phi = np.asarray(X, dtype=np.float64)
    design = np.concatenate([phi, np.ones((phi.shape[0], 1))], axis=1)
    return design @ np.asarray(model["coefficients"], dtype=np.float64)


# ---- AUPRC, threshold-free --------------------------------------------------
def auprc(scores: Any, labels: Any) -> float | None:
    """Right-continuous step area under the precision-recall curve.

    Walk the score ranking from the top (ties broken by index, so the number
    does not depend on input order); each positive extends recall by 1/P and
    contributes precision at that step.  None when there are no positives.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(labels, dtype=np.int64).ravel()
    positives = int(np.count_nonzero(t))
    if positives == 0:
        return None
    order = np.lexsort((np.arange(s.size), -s))
    ranked = t[order]
    tp = np.cumsum(ranked > 0, dtype=np.float64)
    fp = np.cumsum(ranked <= 0, dtype=np.float64)
    precision = tp / (tp + fp)
    recall = tp / float(positives)
    hits = ranked > 0
    prev_recall = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev_recall)[hits] * precision[hits]))


# ---- one series, one Query --------------------------------------------------
def score_query_series(
    model: Mapping[str, Any],
    query_values: Any,
    region: tuple[int, int],
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    window: int,
    median: float,
    scale: float,
) -> dict[str, Any]:
    """Score one series' Query region; the Query bytes are never P-processed.

    Returns the T0 event-level scoring (merged runs, greedy +-3 matching, F1)
    plus the threshold-free AUPRC over the region's points.
    """
    lo, hi = int(region[0]), int(region[1])
    x, indices = features_for_range(
        query_values, lo, hi, window=window, median=median, scale=scale
    )
    point_scores = score(model, x)
    flags = point_scores > float(DECISION_THRESHOLD)
    predicted = ad.predicted_events(flags, offset=lo)
    truth = [
        {"start": int(row["index"]), "end": int(row["index"]) + int(row["points"])}
        for row in ledger_rows
    ]
    scored = ad.score_events(truth, predicted)
    label_vector = np.zeros(indices.size, dtype=np.int64)
    event_points = {
        int(point)
        for row in ledger_rows
        for point in range(int(row["index"]), int(row["index"]) + int(row["points"]))
    }
    for position, t in enumerate(indices):
        if int(t) in event_points:
            label_vector[position] = 1
    scored["auprc"] = auprc(point_scores, label_vector)
    scored["flagged_points"] = int(np.count_nonzero(flags))
    scored["scored_points"] = int(indices.size)
    scored["feature_window"] = int(window)
    return scored


def pooled_f1(per_series: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Event-level F1 over all series pooled (the 48-event granularity)."""
    truth = sum(int(row["ledger_events"]) for row in per_series.values())
    pred = sum(int(row["predicted_events"]) for row in per_series.values())
    hit = sum(int(row["matched_events"]) for row in per_series.values())
    precision = float(hit) / pred if pred else None
    recall = float(hit) / truth if truth else None
    if not truth:
        return None
    if not pred or (precision or 0.0) + (recall or 0.0) <= 0.0:
        return 0.0
    return 2.0 * float(precision) * float(recall) / (float(precision) + float(recall))


def spec() -> dict[str, Any]:
    """The whole frozen instrument, for the artifact."""
    return {
        "consumer_id": CONSUMER_ID,
        "task": TASK,
        "classifier": dict(CLASSIFIER_SPEC),
        "fallback": dict(FALLBACK_SPEC),
        "score": dict(SCORE_SPEC),
        "sited_at": "evaluation/functional/consumers/anomaly_detection_trainable_v1.py",
        "siting_rule": (
            "an experiment instrument, not a Harness part: nothing under "
            "methods/ttha imports this module and this module imports nothing "
            "from the Harness under study"
        ),
    }
