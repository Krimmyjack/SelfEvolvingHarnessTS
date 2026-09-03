"""Which refused candidate is the one worth a Slow call, decided by arithmetic.

A round can refuse several materially-positive candidates on the tail budget.
v2 handed Slow whichever came first in probe order, and probe order is the Fast
agent's, which knows nothing about repairability.  The live v2 run paid for that
exactly once and decisively: at origin 2136 the first refusal was
``hampel_filter``, and the pre-registered oracle bound (``p4y``) had already
established that this probe admits **no** feasible one-clause narrowing at that
origin, while the other refused probe at the same origin admits eleven.  Slow
was handed the impossible one, cleared three of the four lines anyway, and the
round was recorded as a failure to narrow.

The rule here is the distance sol specified: *how many served series would have
to be excluded before the four lines all hold?*  It is computed from the
Support-A per-series gain vector the refusal already carries -- so it costs
nothing, reads no Outcome the admission gate had not already read, and consults
no oracle.

What this is not
----------------
It does **not** say which predicate achieves the exclusion, which feature to
name, or which series to drop.  Those are precisely what ``p4y`` computed while
looking at outcomes, and they stay quarantined: the number below is a *ranking*
over refusals, and it never reaches the card.

It is also a Support-window number.  A Scope that could clear the lines here may
not clear them at origin+48, where the predicate is re-resolved against
different features -- that is the whole reason the delayed reading is the
endpoint.  So this ranks how *locally* repairable a refusal looks; it does not
promise a repair exists.

Why the greedy count is exact
-----------------------------
Dropping the k most-harmful series is simultaneously optimal for all four lines:
it raises the mean the most, removes a harmed series from the numerator and the
denominator at once, and lowers the worst single harm the fastest.  So the
smallest k that clears the lines under greedy removal is the smallest k that
clears them under any removal, and no search is needed.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import p4b_contract as bounded
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

MATERIAL = admission_policy.MATERIAL_THRESHOLD
MAX_HARMED = bounded.BOUNDED_MAX_HARMED_FRACTION
MAX_HARM = bounded.BOUNDED_MAX_SINGLE_SERIES_HARM

#: The coverage floor a surviving revision must still meet, declared in P4U-v2's
#: ``DELAYED_ADMISSION`` and unchanged here.  A count of exclusions that pushed
#: the treated set below it would not be a repair.
MIN_TREATED = 5

#: Returned instead of a count when no subset of the treated set clears the
#: lines while still meeting the coverage floor.
INFEASIBLE = None


def _lines_hold(values: np.ndarray) -> dict[str, bool]:
    if values.size == 0:
        return {"coverage_floor": False, "aggregate": False,
                "harmed_fraction": False, "single_series_harm": False}
    worst = float(-values.min()) if values.min() < 0 else 0.0
    return {
        "coverage_floor": int(values.size) >= MIN_TREATED,
        "aggregate": float(values.mean()) >= MATERIAL,
        "harmed_fraction": float((values < -MATERIAL).mean()) <= MAX_HARMED,
        "single_series_harm": worst <= MAX_HARM,
    }


def min_exclusions_to_clear(per_series_gain: Sequence[float]) -> int | None:
    """Fewest served series to exclude before every declared line holds.

    ``0`` means the lines already hold -- which for a *refused* candidate means
    the refusal came from a line this function does not model, and the caller
    should treat it as a reading, not a contradiction.  ``None`` means no
    exclusion clears them without breaching the coverage floor.
    """
    values = np.asarray(
        [float(v) for v in (per_series_gain or ())], dtype=np.float64)
    if values.size == 0:
        return INFEASIBLE
    ordered = np.sort(values)  # ascending: the most harmful first
    for dropped in range(0, max(values.size - MIN_TREATED, 0) + 1):
        remaining = ordered[dropped:]
        if all(_lines_hold(remaining).values()):
            return int(dropped)
    return INFEASIBLE


def _sort_key(row: Mapping[str, Any]) -> tuple:
    """Fewest exclusions first; infeasible last; ties by gain then by name.

    Every component is a total order over values already in the artifact, so
    two runs on the same readings select the same candidate.
    """
    required = row.get("min_exclusions_to_clear")
    return (
        1 if required is None else 0,          # feasible before infeasible
        required if required is not None else 0,
        -float(row.get("aggregate_gain") or 0.0),  # larger gain first
        str(row.get("candidate_id") or ""),
    )


def rank_risk_refusals(
    refusals: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Score every refusal of a round and order them, without choosing yet."""
    scored = []
    for index, refusal in enumerate(refusals or ()):
        vector = refusal.get("per_series_gain") or ()
        required = min_exclusions_to_clear(vector)
        scored.append({
            "probe_index": index,
            "candidate_id": str(refusal.get("candidate_id") or ""),
            "reason": str(refusal.get("reason") or ""),
            "aggregate_gain": refusal.get("aggregate_gain"),
            "treated_series": len(list(vector)),
            "min_exclusions_to_clear": required,
            "feasible_locally": required is not None,
        })
    scored.sort(key=_sort_key)
    for rank, row in enumerate(scored):
        row["rank"] = rank
    return scored


def select_risk_refusal(
    refusals: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """The one refusal this round's single Slow call is spent on.

    Returns the selection *and* the full ranking, because an artifact that
    records only the winner cannot answer whether the choice mattered.  One is
    always selected when a refusal exists -- including when every candidate is
    locally infeasible, in which case the record says so and the resulting Slow
    failure is attributable to the task rather than to Slow.
    """
    ranked = rank_risk_refusals(refusals)
    if not ranked:
        return None
    chosen = ranked[0]
    return {
        "selected_probe_index": chosen["probe_index"],
        "selected_candidate_id": chosen["candidate_id"],
        "selected_min_exclusions": chosen["min_exclusions_to_clear"],
        "all_candidates_locally_infeasible": not any(
            row["feasible_locally"] for row in ranked),
        "candidates_considered": len(ranked),
        "would_have_been_probe_order": ranked and min(
            row["probe_index"] for row in ranked),
        "rule": (
            "fewest served series that must be excluded before the coverage, "
            "aggregate, harmed-fraction and single-series lines all hold; "
            "ties by larger aggregate gain, then by candidate id"
        ),
        "computed_from": "the Support-A per-series gain vector of each refusal",
        "does_not_use": [
            "the oracle bound p4y", "any feature the oracle selected",
            "the identity of any series",
        ],
        "is_a_support_window_number": (
            "local repairability only; the delayed window re-resolves the "
            "predicate against different features and is the actual endpoint"
        ),
        "ranking": ranked,
    }


def selector(refusals_to_index: Sequence[Mapping[str, Any]]) -> int:
    """The injectable form ``run_online_round`` takes: refusals -> probe index."""
    choice = select_risk_refusal(refusals_to_index)
    return 0 if choice is None else int(choice["selected_probe_index"])
