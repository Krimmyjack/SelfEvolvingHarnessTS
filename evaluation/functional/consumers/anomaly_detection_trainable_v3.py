"""AD Consumer, trainable v3: a threshold head learned on the task-native
sufficient statistic.

v3 exists by a main-line ruling after v1 and v2 both stopped at
AD_TRAINABLE_SPEC_DEFECT (t1b_training_flip_v1 / _v2): the raw trailing
window x linear-ridge specification family is closed by credible negative,
and the only permitted change now is the *feature family*.

- feature: the single value f1 = z_t, the trailing robust z-score, obtained
  in one pass through the very code path the T0 detector uses:
  ``anomaly_detection_v1.detect(values, window=49, threshold=3.5)["scores"]``.
  The 49/3.5 parameters are T0's frozen fallback parameters and are passed
  EXPLICITLY (the detect() file defaults are 25/4.0 and are never eaten).
  No median/MAD is re-implemented here; the feature is not re-standardized.
- head (unchanged from v1/v2): weighted closed-form ridge, alpha = 1.0,
  positive-class weight N_neg/N_pos over the rows that enter the fit,
  point flagged when the ridge output > 0.5; event merging and the greedy
  one-to-one +-3 matching come from the frozen T0 scoring parts.
- abstention (T0 semantics; a non-finite score is never converted to zero):
  * training -- a point whose z is undefined does not enter the fit; the
    exclusion is counted and attributed (warm_up / zero_scale / non_finite);
  * Query -- a point whose z is undefined is forced to not flag;
  * AUPRC -- computed over the finite-feature points only, with the
    undefined-point count reported;
  * a ledger event whose points are all unscoreable is simply missed --
    no special handling.
- metrics (v2 ruling): the primary reading is the macro average of the
  per-series event F1; pooled F1 and AUPRC are secondary readings.

What this Consumer is and is not (the two sentences the confirmed verdict
must carry): the AD Consumer is a threshold head learned ON the task-native
sufficient statistic (a learnable robust-z, in effect), so a flip verdict
speaks only for this Consumer family; and a confirmed flip only proves that
the training-data utility flip is readable by an instrument when the
task-native sufficient statistic is visible -- it does not prove the Harness
discovered that representation by itself, and it says nothing about
generalization to natural anomaly data.

The module stays deterministic end to end: one detect() pass, closed form,
no solver iteration, no seed.  There is NO fallback in v3: the readability
gate is single-shot, and a miss closes the supervised-AD positive-control
family (SUPERVISED_AD_PC_FAMILY_CLOSED).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from consumers import anomaly_detection_v1 as ad

CONSUMER_ID = "anomaly_detection_trainable_v3"
TASK = "anomaly_detection"

# ---- the feature path, frozen -----------------------------------------------
# T0's frozen fallback detector parameters.  Passed explicitly to detect();
# the detect() file defaults (25 / 4.0) are never used.
FEATURE_WINDOW = 49
FEATURE_THRESHOLD = 3.5
if not (
    int(ad.FALLBACK_WINDOW) == FEATURE_WINDOW
    and float(ad.FALLBACK_THRESHOLD) == FEATURE_THRESHOLD
):
    raise RuntimeError(
        "T0's frozen fallback parameters moved; v3's feature path is pinned "
        "to them and refuses to drift"
    )

# ---- head, frozen (unchanged from v1/v2) ------------------------------------
ALPHA = 1.0
DECISION_THRESHOLD = 0.5
CLASSIFIER_SPEC: dict[str, Any] = {
    "classifier": "weighted closed-form ridge over the single feature z_t",
    "feature": (
        "f1 = z_t = detect(values, window=49, threshold=3.5)['scores'], one "
        "pass through the T0 detector's own code path; explicit 49/3.5 (the "
        "frozen fallback parameters), never the 25/4.0 file defaults; no "
        "re-standardization"
    ),
    "feature_window": FEATURE_WINDOW,
    "standardization": "none -- the feature already is a robust z-score",
    "label": "1 at every ledger event point (all 3 points of a burst), else 0",
    "positive_class_weight": (
        "N_neg / N_pos, computed on the rows that enter the fit (undefined-z "
        "points are excluded first)"
    ),
    "alpha": ALPHA,
    "solver": (
        "weighted normal equations, np.linalg.solve, mirroring "
        "evaluation/benchmark_v02/trainers.fit_closed_form (intercept column "
        "last and unpenalized)"
    ),
    "decision": "score > 0.5 flags the point; an undefined z never flags",
    "deterministic": True,
    "fallback": "none -- v3 is single-shot; a gate miss closes the family",
}

# ---- abstention semantics (T0), frozen --------------------------------------
ABSTENTION_SPEC: dict[str, Any] = {
    "training": (
        "a point whose z is undefined does not enter the fit; exclusions are "
        "counted and attributed to warm_up / zero_scale / non_finite"
    ),
    "query": "a point whose z is undefined is forced to not flag",
    "auprc": (
        "computed over the finite-feature points only; the undefined-point "
        "count is reported"
    ),
    "ledger_events_all_unscoreable": (
        "naturally counted as missed detections; no special handling"
    ),
    "never": "a non-finite score is never converted to zero",
}

# ---- scoring, inherited frozen ----------------------------------------------
SCORE_SPEC: dict[str, Any] = {
    "unit": "event",
    "event_merging": "maximal runs of consecutive flagged points (T0)",
    "matching": "greedy nearest-distance one-to-one at +-3 (T0)",
    "main_metric": (
        "per-series Query event F1, macro-averaged across series (v2 "
        "execution-convention ruling); the pooled F1 is reported alongside "
        "as a secondary reading"
    ),
    "secondary_metric": "AUPRC from the ridge scores, threshold-free",
    "precision_is_a_lower_bound": True,
    "scoring_parts": (
        "anomaly_detection_v1.predicted_events / match_events / score_events, "
        "imported, not re-implemented"
    ),
}


# ---- the feature path ---------------------------------------------------------
def _attribute_undefined(
    values: np.ndarray, z: np.ndarray, window: int
) -> dict[str, int]:
    """Attribute every undefined z to warm_up / zero_scale / non_finite.

    Finiteness masks only -- no median/MAD is recomputed here.  A position
    beyond warm-up with finite inputs but an undefined z is exactly detect()'s
    zero-scale abstention.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    warm_up = min(int(window), int(array.size))
    warm_mask = np.zeros(array.size, dtype=bool)
    warm_mask[:warm_up] = True
    point_ok = np.isfinite(array)
    history_ok = np.ones(array.size, dtype=bool)
    if array.size >= int(window):
        windows = np.lib.stride_tricks.sliding_window_view(array, int(window))
        # windows[i] = array[i : i+window]; the history of index t is
        # windows[t-window], defined for t in [window, size).
        hist_finite = np.isfinite(windows).all(axis=1)
        history_ok[int(window):] = hist_finite[: array.size - int(window)]
    non_finite_mask = ~warm_mask & (~point_ok | ~history_ok)
    undefined = ~np.isfinite(z)
    zero_scale_mask = undefined & ~warm_mask & ~non_finite_mask
    return {
        "warm_up": int(np.count_nonzero(undefined & warm_mask)),
        "zero_scale": int(np.count_nonzero(zero_scale_mask)),
        "non_finite": int(np.count_nonzero(undefined & non_finite_mask)),
    }


def block_features(block: Any) -> dict[str, Any]:
    """z_t for every position of one series' P(training block), one detect()
    pass.  The block starts at its own index 0, so the first FEATURE_WINDOW
    positions are warm-up abstentions (the trailing geometry reads no bytes
    before the block here -- the same canon as v1/v2's training rows)."""
    array = np.asarray(block, dtype=np.float64).ravel()
    reading = ad.detect(
        array, window=FEATURE_WINDOW, threshold=FEATURE_THRESHOLD
    )
    z = reading["scores"]
    return {
        "z": z,
        "counts": _attribute_undefined(array, z, FEATURE_WINDOW),
        "detect_counts": {
            "unscoreable_warm_up": reading["unscoreable_warm_up"],
            "abstained_zero_scale": reading["abstained_zero_scale"],
            "abstained_non_finite": reading["abstained_non_finite"],
        },
    }


def query_features(
    query_values: Any, lo: int, hi: int
) -> dict[str, Any]:
    """z_t for t in [lo, hi) of one series' Query copy, one detect() pass.

    The trailing geometry reads the FEATURE_WINDOW pristine bytes preceding
    the region (the established canon; those bytes are never P-processed),
    so every region point has a full trailing window and only zero-scale or
    non-finite abstentions can leave a region point undefined.
    """
    lo, hi = int(lo), int(hi)
    array = np.asarray(query_values, dtype=np.float64).ravel()
    if lo < FEATURE_WINDOW:
        raise ValueError("region start %d leaves no full trailing window" % lo)
    if hi > array.size:
        raise ValueError("range end %d beyond array" % hi)
    fed = array[lo - FEATURE_WINDOW:hi]
    reading = ad.detect(
        fed, window=FEATURE_WINDOW, threshold=FEATURE_THRESHOLD
    )
    z_region = reading["scores"][FEATURE_WINDOW:]
    counts = _attribute_undefined(fed, reading["scores"], FEATURE_WINDOW)
    return {
        "z": z_region,
        "indices": np.arange(lo, hi, dtype=np.int64),
        # the region points themselves are never warm-up; only abstentions
        "counts": {
            "warm_up": 0,
            "zero_scale": counts["zero_scale"],
            "non_finite": counts["non_finite"],
        },
    }


# ---- the closed-form fit (unchanged head) -------------------------------------
def fit(X: Any, y: Any) -> dict[str, Any]:
    """Weighted ridge, closed form, mirroring trainers.fit_closed_form."""
    phi = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64).ravel()
    if phi.ndim != 2 or phi.shape[0] != target.size or phi.shape[0] == 0:
        raise ValueError("feature/label shape mismatch or empty training set")
    if not np.isfinite(phi).all():
        raise ValueError("v3 fits finite-feature rows only")
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


# ---- AUPRC, threshold-free ----------------------------------------------------
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


# ---- one series, one Query ----------------------------------------------------
def score_query_series(
    model: Mapping[str, Any],
    query_values: Any,
    region: tuple[int, int],
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    window: int,
    median: float | None,
    scale: float | None,
) -> dict[str, Any]:
    """Score one series' Query region; the Query bytes are never P-processed.

    ``window``/``median``/``scale`` are accepted for runner compatibility and
    are unused: the feature path is pinned to FEATURE_WINDOW and the z-score
    is never re-standardized.  Undefined-z points are forced to not flag and
    are excluded from the AUPRC ranking, with their count reported.
    """
    lo, hi = int(region[0]), int(region[1])
    feats = query_features(query_values, lo, hi)
    z = feats["z"]
    indices = feats["indices"]
    finite = np.isfinite(z)
    point_scores = np.full(z.size, np.nan, dtype=np.float64)
    if np.count_nonzero(finite):
        point_scores[finite] = score(model, z[finite, None])
    flags = np.zeros(z.size, dtype=bool)
    flags[finite] = point_scores[finite] > float(DECISION_THRESHOLD)
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
    scored["auprc"] = auprc(point_scores[finite], label_vector[finite])
    scored["flagged_points"] = int(np.count_nonzero(flags))
    scored["scored_points"] = int(np.count_nonzero(finite))
    scored["undefined_points"] = int(np.count_nonzero(~finite))
    scored["undefined_attribution"] = dict(feats["counts"])
    scored["feature_window"] = int(FEATURE_WINDOW)
    return scored


def pooled_f1(per_series: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Event-level F1 over all series pooled (the 48-event granularity).

    Secondary reading since the v2 ruling; the gate and the judgment read the
    macro average instead.
    """
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


def macro_f1(per_series: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Mean of the per-series event F1 -- the primary reading since the v2
    ruling.  Each series carries 4 events, so one event changing hands moves
    one series' recall by 1/4 and this aggregate by roughly 0.02."""
    values = [
        float(row["f1"]) for row in per_series.values() if row["f1"] is not None
    ]
    if not values:
        return None
    return float(np.mean(values))


def spec() -> dict[str, Any]:
    """The whole frozen instrument, for the artifact."""
    return {
        "consumer_id": CONSUMER_ID,
        "task": TASK,
        "classifier": dict(CLASSIFIER_SPEC),
        "abstention": dict(ABSTENTION_SPEC),
        "score": dict(SCORE_SPEC),
        "sited_at": "evaluation/functional/consumers/anomaly_detection_trainable_v3.py",
        "siting_rule": (
            "an experiment instrument, not a Harness part: nothing under "
            "methods/ttha imports this module and this module imports nothing "
            "from the Harness under study"
        ),
    }
