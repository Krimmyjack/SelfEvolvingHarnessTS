"""AD Consumer v1: a deterministic, zero-training anomaly detector and its score.

This is the second Consumer of the Phase T pair.  It exists so that the same
Program, acting once on the same block, can be read by two tasks whose quality
standards disagree -- the first-principles proposition the forecasting line has
never been able to test on its own.

The whole module is frozen instrument.  Every constant below is pre-registered:
none of them may be tuned against a reading, and the only permitted movement is
the single fallback in ``FALLBACK_SPEC``, which may be taken at most once and
is then frozen in the artifact for good.

What is deliberately fixed, and why
-----------------------------------
*Trailing, not centred.*  The window for index ``t`` is ``x[t-25:t]``: history
only.  A centred window would let a 3-point burst pull its own median and mask
itself, and a trailing window is also the only one a streaming detector could
compute.  The first ``WINDOW`` points of any array handed here are therefore
unscoreable and the caller must supply them as warm-up.

*Zero robust scale abstains.*  When ``1.4826 * MAD`` over the window is zero the
z-score is undefined, and calling that an anomaly is exactly the error the S0
census made on SMD's binary command channels, where every transition of a
constant channel read as an infinite z-peak.  An undefined scale is not
evidence.  The abstention is counted and reported, never silently dropped.

*Precision is a lower bound.*  It is computed against the injection ledger, so
a real, unlabelled anomaly that the detector correctly flags is counted in the
denominator as if it were a false alarm.  Every precision this module returns
carries that meaning and the artifact must say so.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CONSUMER_ID = "anomaly_detection_v1"
TASK = "anomaly_detection"

# ---- detector, frozen -------------------------------------------------------
WINDOW = 25
THRESHOLD = 4.0
MAD_TO_SIGMA = 1.4826
DETECTOR_SPEC: dict[str, Any] = {
    "detector": "trailing robust z-score",
    "window": WINDOW,
    "window_geometry": "x[t-25:t], strictly trailing, x_t excluded",
    "statistic": "z = |x_t - median(window)| / (1.4826 * MAD(window))",
    "threshold": THRESHOLD,
    "deterministic": True,
    "training": "none; the detector fits nothing and holds no state",
    "zero_scale_rule": (
        "1.4826 * MAD(window) == 0 -> abstain (no flag).  An undefined scale "
        "is not evidence of an anomaly; this is the S0 binary-channel lesson"
    ),
    "warm_up": (
        "the first WINDOW indices of the array handed to detect() have no "
        "full trailing window and are never flagged; callers pass warm-up"
    ),
}

# ---- the one pre-registered fallback ---------------------------------------
FALLBACK_WINDOW = 49
FALLBACK_THRESHOLD = 3.5
FALLBACK_SPEC: dict[str, Any] = {
    "window": FALLBACK_WINDOW,
    "threshold": FALLBACK_THRESHOLD,
    "budget": "may be taken at most once, and only after the primary spec fails",
    "on_use": "the chosen parameters freeze into the artifact; T1 may not move them",
}

# ---- scoring, frozen --------------------------------------------------------
TOLERANCE = 3
MATERIAL_THRESHOLD = 0.005
SCORE_SPEC: dict[str, Any] = {
    "unit": "event",
    "predicted_event": "a maximal run of consecutive over-threshold indices",
    "matching": "greedy nearest-distance, one-to-one, in both directions",
    "tolerance": TOLERANCE,
    "distance": (
        "min |p - g| over p in the predicted event's indices and g in the "
        "ledger event's indices; 0 when they overlap"
    ),
    "tie_break": "(distance, ledger event start, predicted event start)",
    "precision_denominator": (
        "every predicted event, including ones the ledger does not name"
    ),
    "precision_is_a_lower_bound": (
        "the substrate's natural anomalies carry no labels, so a correct flag "
        "on one of them is counted here as if it were a false alarm"
    ),
    "material_threshold": MATERIAL_THRESHOLD,
    "harm_threshold": -MATERIAL_THRESHOLD,
}


# ---- detector ---------------------------------------------------------------
def detect(
    values: Any,
    *,
    window: int = WINDOW,
    threshold: float = THRESHOLD,
) -> dict[str, Any]:
    """Flag every index whose trailing robust z-score exceeds ``threshold``.

    Returns the boolean flag vector plus the two diagnostics the acceptance
    reads: how many indices were unscoreable for want of a full window, and how
    many abstained because the robust scale was zero.
    """
    array = np.asarray(values, dtype=np.float64).ravel()
    flags = np.zeros(array.size, dtype=bool)
    scores = np.full(array.size, np.nan, dtype=np.float64)
    warm_up = min(int(window), int(array.size))
    zero_scale = 0
    non_finite = 0
    for index in range(warm_up, array.size):
        point = array[index]
        history = array[index - window:index]
        if not np.isfinite(point) or not np.isfinite(history).all():
            non_finite += 1
            continue
        centre = float(np.median(history))
        scale = MAD_TO_SIGMA * float(np.median(np.abs(history - centre)))
        if not np.isfinite(scale) or scale <= 0.0:
            zero_scale += 1
            continue
        z = abs(float(point) - centre) / scale
        scores[index] = z
        flags[index] = z > threshold
    return {
        "flags": flags,
        "scores": scores,
        "unscoreable_warm_up": int(warm_up),
        "abstained_zero_scale": int(zero_scale),
        "abstained_non_finite": int(non_finite),
        "window": int(window),
        "threshold": float(threshold),
    }


def predicted_events(flags: Any, *, offset: int = 0) -> list[dict[str, int]]:
    """Maximal runs of consecutive flagged indices, in global coordinates."""
    mask = np.asarray(flags, dtype=bool).ravel()
    events: list[dict[str, int]] = []
    start: int | None = None
    for index in range(mask.size):
        if mask[index] and start is None:
            start = index
        elif not mask[index] and start is not None:
            events.append({"start": int(start + offset), "end": int(index + offset)})
            start = None
    if start is not None:
        events.append({"start": int(start + offset), "end": int(mask.size + offset)})
    return events


# ---- scoring ----------------------------------------------------------------
def _span(event: Mapping[str, Any]) -> tuple[int, int]:
    return int(event["start"]), int(event["end"])


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    """Gap in steps between two index spans; 0 when they overlap."""
    a0, a1 = _span(left)
    b0, b1 = _span(right)
    if a0 < b1 and b0 < a1:
        return 0
    if b0 >= a1:
        return b0 - (a1 - 1)
    return a0 - (b1 - 1)


def match_events(
    truth: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    *,
    tolerance: int = TOLERANCE,
) -> list[dict[str, Any]]:
    """Greedy nearest-distance one-to-one matching, pinned by SCORE_SPEC.

    A predicted event matches at most one ledger event and a ledger event
    matches at most one predicted event.  Pairs are considered in order of
    distance, then ledger start, then predicted start, so the result does not
    depend on the order the caller built either list in.
    """
    pairs: list[tuple[int, int, int, int, int]] = []
    for t_index, t_event in enumerate(truth):
        for p_index, p_event in enumerate(predicted):
            distance = _distance(t_event, p_event)
            if distance <= tolerance:
                pairs.append((
                    int(distance),
                    int(t_event["start"]),
                    int(p_event["start"]),
                    int(t_index),
                    int(p_index),
                ))
    pairs.sort()
    taken_truth: set[int] = set()
    taken_pred: set[int] = set()
    matched: list[dict[str, Any]] = []
    for distance, _t_start, _p_start, t_index, p_index in pairs:
        if t_index in taken_truth or p_index in taken_pred:
            continue
        taken_truth.add(t_index)
        taken_pred.add(p_index)
        matched.append({
            "ledger_event_index": int(t_index),
            "predicted_event_index": int(p_index),
            "distance": int(distance),
        })
    matched.sort(key=lambda row: row["ledger_event_index"])
    return matched


def score_events(
    truth: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    *,
    tolerance: int = TOLERANCE,
) -> dict[str, Any]:
    """Event-level precision / recall / F1 against the ledger."""
    matched = match_events(truth, predicted, tolerance=tolerance)
    n_truth, n_pred, n_hit = len(truth), len(predicted), len(matched)
    precision = float(n_hit) / n_pred if n_pred else None
    recall = float(n_hit) / n_truth if n_truth else None
    if not n_truth:
        f1: float | None = None
    elif not n_pred:
        f1 = 0.0
    elif (precision or 0.0) + (recall or 0.0) <= 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * float(precision) * float(recall) / (float(precision) + float(recall))
    return {
        "ledger_events": n_truth,
        "predicted_events": n_pred,
        "matched_events": n_hit,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matched,
        "precision_is_a_lower_bound": True,
    }


def background_alarm_rate(flags: Any) -> dict[str, Any]:
    """Alarms per thousand scored points on a block carrying no injection.

    This is NOT a false-positive rate and the artifact must not call it one:
    the substrate's natural anomalies are unlabelled, so an alarm here may be
    correct.  It is a background level, nothing more.
    """
    mask = np.asarray(flags, dtype=bool).ravel()
    events = predicted_events(mask)
    scored = int(mask.size)
    return {
        "scored_points": scored,
        "alarm_events": len(events),
        "alarm_points": int(np.count_nonzero(mask)),
        "alarm_events_per_1000_points": (
            1000.0 * len(events) / scored if scored else None
        ),
        "not_a_false_positive_rate": (
            "the substrate carries unlabelled natural anomalies; an alarm on "
            "one of them is correct, so this level bounds nothing"
        ),
    }


# ---- the gain vector, in the forecasting line's field names -----------------
def gain_rows(
    identity_f1: Mapping[str, Any],
    candidate_f1: Mapping[str, Any],
    series_uids: Sequence[str],
) -> dict[str, Any]:
    """gain(P) = F1(P(B)) - F1(identity(B)), named as ``bch._gain_rows`` names it.

    The key names are the contract: ``per_eval_series_gain`` is what the guard
    grammar's ``min_per_series_gain`` statistic, the selector and RESCOPE all
    read, and they must reach this vector without one line of Harness change.
    Positive is better here as it is there -- F1 rises where sMASE falls -- so
    the material and harm lines keep the same sign convention.
    """
    per_series: dict[str, float] = {}
    undefined: list[str] = []
    for uid in series_uids:
        base, candidate = identity_f1.get(uid), candidate_f1.get(uid)
        if base is None or candidate is None:
            undefined.append(str(uid))
            continue
        per_series[str(uid)] = float(candidate) - float(base)
    harmed = {
        uid: value
        for uid, value in per_series.items()
        if value < -MATERIAL_THRESHOLD
    }
    return {
        "aggregate_gain": (
            float(np.mean(list(per_series.values()))) if per_series else None
        ),
        "per_origin_gain": None,
        "per_origin_gain_absent_because": (
            "the AD Consumer scores a block as one contiguous stream, so a "
            "block yields one F1 and not one per origin.  The key is named "
            "and nulled rather than dropped, the same way the compiler names "
            "it on an identity-routed projection"
        ),
        "per_eval_series_gain": per_series,
        "per_series_gain_undefined": undefined,
        "harmed_eval_series_count": len(harmed),
        "harmed_eval_series_total_harm": float(-sum(harmed.values())),
        "harmed_eval_series": sorted(harmed),
    }


def spec() -> dict[str, Any]:
    """The whole frozen instrument, for the artifact."""
    return {
        "consumer_id": CONSUMER_ID,
        "task": TASK,
        "detector": dict(DETECTOR_SPEC),
        "score": dict(SCORE_SPEC),
        "fallback": dict(FALLBACK_SPEC),
        "sited_at": "evaluation/functional/consumers/anomaly_detection_v1.py",
        "siting_rule": (
            "an experiment instrument, not a Harness part: nothing under "
            "methods/ttha imports this module and this module imports nothing "
            "from the Harness under study"
        ),
    }
