"""Which Phase-T units can carry a curve point -- frozen, and cross-checked.

Three numbers that are not the same number
------------------------------------------
The course has **26 scheduled units**.  Of those, **23 are scoreable**: the
other three have no observed truth anywhere in their evaluation horizon, so the
missing-aware sMASE is undefined there and *no arm* can be scored on them.  A
``COMPLETE`` ordering therefore needs **19 valid paired curve points**, which is
``ceil(0.8 x 23)`` -- and the ceiling matters, because ``int(0.8 x 23)`` is 18
and 18/23 is 78.3%, which is not the 80% the contract says.

Conflating the three is how a run gets graded against a denominator it never
had.  Measuring completion against 26 would make a perfectly healthy course look
short by three; measuring the curve *over* 26 would silently add three zero
differences, because a missing evaluation reads as ``0.0`` through the ordinary
accessor and a zero difference is indistinguishable from "the arms tied here".

Why the list is frozen rather than recomputed per ordering
----------------------------------------------------------
The three orderings are permutations of one unit set, so their scoreable sets
are identical by construction.  Freezing the list makes that a checkable fact
instead of an assumption, and it removes the one route by which an ordering
could be scored against a denominator of its own -- which is exactly the shape
of error that a per-ordering recomputation invites.

The list is **declared** here and **derived** by
``preflight_hec1_evaluability``.  Neither is authoritative alone: the preflight
recomputes it from the data with zero fits and ``verify_against`` asserts the
two agree.  A disagreement means either the data changed or the declaration
drifted, and both must stop the run rather than be reconciled silently.

What "unscoreable" does not mean
--------------------------------
It does not mean the unit is skipped.  The Harness still probes it, the delayed
gate still decides, Episodes still enter the bank, and a Skill can still be
formed there.  Only the curve point is absent -- and it is absent for **every
arm identically**, because the cause is the data and not any policy.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

#: Units the frozen course schedules, per ordering (``p4ac``'s <=3816 caliber).
SCHEDULED_UNITS = 26

#: The completion fraction the contract requires of an ordering before it can
#: be read out rather than recorded ``HEC1_INCONCLUSIVE``.
COMPLETION_FRACTION = 0.8

#: (block, origin) pairs whose evaluation face carries no observed truth.
#: Derived by ``preflight_hec1_evaluability`` with 0 fits and 0 LLM, then frozen
#: here so all three orderings are scored against one list.
UNSCOREABLE_UNITS: tuple[tuple[str, int], ...] = (
    ("[0:40]", 2856),
    ("[40:80]", 2856),
    ("[120:160]", 1656),
)

#: Why each one cannot be scored.  One reason, and it is a property of the data.
UNSCOREABLE_REASON = "horizon contains no observed truth"

SCOREABLE_UNITS = SCHEDULED_UNITS - len(UNSCOREABLE_UNITS)

#: ``ceil``, not ``int``.  18/23 = 78.3% would not clear the 80% the contract
#: declares; 19/23 = 82.6% does.
MIN_PAIRED_CURVE_POINTS = math.ceil(COMPLETION_FRACTION * SCOREABLE_UNITS)


def is_scoreable(block: Any, origin: Any) -> bool:
    return (str(block), int(origin)) not in set(UNSCOREABLE_UNITS)


def unit_is_scoreable(unit: Mapping[str, Any] | None) -> bool:
    if not unit:
        return True
    return is_scoreable(unit.get("block"), unit.get("origin"))


def to_dict() -> dict[str, Any]:
    """The three numbers and the list, for a contract or artifact to embed."""
    return {
        "scheduled_units": SCHEDULED_UNITS,
        "scoreable_units": SCOREABLE_UNITS,
        "unscoreable_units": [{"block": block, "origin": origin,
                               "reason": UNSCOREABLE_REASON}
                              for block, origin in UNSCOREABLE_UNITS],
        "completion_fraction": COMPLETION_FRACTION,
        "min_paired_curve_points": MIN_PAIRED_CURVE_POINTS,
        "rounding": "ceil; int() would accept 18/23 = 78.3% as if it were 80%",
        "same_list_for_every_ordering": True,
        "unscoreable_units_still_learn": (
            "they are probed, gated and written to the bank; only the curve "
            "point is absent, and it is absent for every arm identically"
        ),
        "derived_by": "evaluation.main_protocol_p4.preflight_hec1_evaluability",
    }


def verify_against(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check the frozen declaration against a fresh 0-fit derivation.

    Mechanical both ways: the declaration cannot silently drift from the data,
    and a data change cannot silently pass as the declaration.  Returns the
    comparison rather than raising, so a caller can record the disagreement
    before stopping.
    """
    phase_t = dict(preflight.get("phase_t") or {})
    derived = {(str(row.get("block")), int(row.get("origin")))
               for row in (phase_t.get("dropped_units") or ())}
    declared = set(UNSCOREABLE_UNITS)
    scheduled_ok = int(phase_t.get("N_T") or 0) == SCHEDULED_UNITS
    scoreable_ok = int(phase_t.get("N_T_eff") or 0) == SCOREABLE_UNITS
    orderings_agree = bool(phase_t.get("orderings_agree"))
    missing = sorted(declared - derived)
    unexpected = sorted(derived - declared)
    return {
        "passed": (not missing and not unexpected and scheduled_ok
                   and scoreable_ok and orderings_agree),
        "declared": sorted(declared),
        "derived": sorted(derived),
        "declared_but_not_derived": missing,
        "derived_but_not_declared": unexpected,
        "scheduled_units": {"declared": SCHEDULED_UNITS,
                            "derived": phase_t.get("N_T"),
                            "agree": scheduled_ok},
        "scoreable_units": {"declared": SCOREABLE_UNITS,
                            "derived": phase_t.get("N_T_eff"),
                            "agree": scoreable_ok},
        "min_paired_curve_points": MIN_PAIRED_CURVE_POINTS,
        "orderings_agree_on_the_same_list": orderings_agree,
        "why": (
            "a disagreement means the data changed or the declaration drifted; "
            "either must stop the run rather than be reconciled silently"
        ),
    }


def paired_curve_points(rows: Sequence[Mapping[str, Any]], online: str,
                        frozen: str) -> list[Mapping[str, Any]]:
    """The units that can contribute a difference: both arms actually scored.

    A missing evaluation reads as ``0.0`` through the ordinary accessor, and a
    zero difference is indistinguishable from "the arms tied on this unit".
    Filtering here is what keeps an unscoreable unit out of the curve instead of
    contributing a fabricated tie.
    """
    paired = []
    for row in rows or ():
        arms = dict(row.get("arms") or {})
        left, right = arms.get(online), arms.get(frozen)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        if (left.get("aggregate_gain") is None
                or right.get("aggregate_gain") is None):
            continue
        paired.append(row)
    return paired
