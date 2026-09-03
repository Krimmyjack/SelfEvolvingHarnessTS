"""Slow names a direction; a tool calibrates the number on frozen bins.

Where this comes from
---------------------
Across the Source line, Slow revised a Scope five times and one revision
survived.  Every clause was legal, every clause named a deployment-visible
feature, and the preflight confirmed every one of them was a strict narrowing --
so the failures were not about legality.  They were about the *threshold*: the
model was asked to read twenty anonymous feature rows and eyeball a cut, and
``p4y`` had already shown that a feasible cut existed in six of seven Support
windows.  The task was solvable and the guess was the weak link.

The five-step chain sol specified splits that task at the seam where it is
actually hard:

    Slow proposes a semantic feature + direction
    -> this tool calibrates the threshold on the arm's own Episode bank
    -> the replay screen discards candidates that already failed somewhere
    -> a later, new held-in unit verifies it
    -> only then does it get execution rights

What this tool must not become
------------------------------
It does not choose the program.  It does not decide whether to act.  It does not
choose the feature.  It calibrates the number for the direction Slow named, on
threshold candidates that are the **frozen bin edges** already used for Scope
induction, and it refuses rather than inventing one when no edge is feasible.
A tool that picked the feature as well would be the open-loop tree Router that
``FEATURES_DO_NOT_BEAT_A_FIXED_CHOICE`` already closed -- which is exactly why
``best_stump`` below exists as a *shadow*: it is the measurement of how much of
the revision Slow actually contributed, and it is never deployed.

Widest, and coarser on ties
---------------------------
Among the feasible edges the tool takes the one that keeps the most series
inside the Scope.  A narrower feasible cut would trade coverage the evidence
did not ask it to trade.  Ties resolve toward the coarser box -- the more
permissive edge of the two -- because two edges that select the same rows *here*
do not select the same rows on the next window, and the coarser one is the one
whose behaviour the bins actually describe.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import scope_spec as scopes
from SelfEvolvingHarnessTS.contracts import observables
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

MATERIAL = admission_policy.MATERIAL_THRESHOLD
MAX_HARMED = bounded.BOUNDED_MAX_HARMED_FRACTION
MAX_HARM = bounded.BOUNDED_MAX_SINGLE_SERIES_HARM
MIN_TREATED = 5

DIRECTIONS = ("<=", ">=")

#: Recorded when Slow returns a number as well as a direction.  The number is
#: dropped and the tool's value is used, because a run in which the threshold
#: sometimes came from the model and sometimes from the tool would measure
#: neither.
LLM_THRESHOLD_IGNORED = "LLM_THRESHOLD_IGNORED"

#: The numeric part of the deployment-visible vocabulary, and the only feature
#: names a calibrated clause may carry.  Twelve names in this checkout; the
#: public card has twenty-one keys, and the nine non-numeric ones have no frozen
#: bins so there is nothing to calibrate a threshold against.
VOCABULARY = tuple(
    name for name, kind in observables.OBSERVABLE_FEATURES.items()
    if kind == "number"
)


class NoFeasibleThreshold(Exception):
    """No frozen bin edge for this (feature, direction) clears the budget.

    An exception rather than a sentinel: the caller's next move is to hand the
    refusal back to Slow for a different feature or direction, and a code path
    that could silently proceed with an uncalibrated clause is the failure mode
    this module exists to remove.
    """

    def __init__(self, feature: str, direction: str,
                 tried: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(
            "no frozen bin edge of %s %s clears the risk budget while keeping "
            "%d series in scope (%d edges tried)"
            % (feature, direction, MIN_TREATED, len(tried)))
        self.feature = str(feature)
        self.direction = str(direction)
        self.tried = [dict(row) for row in tried]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": "NO_FEASIBLE_THRESHOLD",
            "feature": self.feature,
            "direction": self.direction,
            "candidates_tried": self.tried,
        }


def frozen_bin_edges(feature: str) -> tuple[float, ...]:
    """The threshold candidates for one feature: its frozen bin edges.

    Read from ``contracts.observables`` rather than restated here.  The
    observable contract is a dependency SHA of every snapshot, so a second copy
    of these numbers could drift from the one Scope induction uses while both
    still claimed to describe the same evidence.
    """
    if feature not in observables.OBSERVABLE_FEATURES:
        raise scopes.ScopeError("unknown observable feature: %r" % feature)
    if observables.OBSERVABLE_FEATURES[feature] != "number":
        raise scopes.ScopeError(
            "%r is not numeric, so it has no bin edges to calibrate against"
            % feature)
    # ``observable_numeric_bin`` falls back to this tuple for every numeric
    # feature without an explicit entry; the fallback is part of the contract,
    # not a default this module chose.
    edges = observables._NUMERIC_BIN_EDGES.get(feature, (0.0, 1.0, 3.0, 6.0))
    return tuple(float(edge) for edge in edges)


def frozen_bins(vocabulary: Sequence[str] = VOCABULARY
                ) -> dict[str, tuple[float, ...]]:
    """Bin edges for a whole vocabulary, the shape ``best_stump`` takes."""
    return {str(name): frozen_bin_edges(str(name)) for name in vocabulary}


@dataclass(frozen=True)
class Policy:
    """The four lines a calibrated clause has to clear on the bank."""

    material: float = MATERIAL
    max_harmed_fraction: float = MAX_HARMED
    max_single_series_harm: float = MAX_HARM
    min_treated: int = MIN_TREATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "material": self.material,
            "max_harmed_fraction": self.max_harmed_fraction,
            "max_single_series_harm": self.max_single_series_harm,
            "min_treated": self.min_treated,
            "thresholds_changed": 0,
        }


BOUNDED_RISK_V1 = Policy()


def _edges_for(feature: str, bins: Any) -> tuple[float, ...]:
    if isinstance(bins, Mapping):
        edges = bins.get(feature)
        if edges is None:
            raise scopes.ScopeError(
                "no bin edges supplied for %r" % feature)
        return tuple(float(edge) for edge in edges)
    if bins is None:
        return frozen_bin_edges(feature)
    return tuple(float(edge) for edge in bins)


def _holds(op: str, value: float, threshold: float) -> bool:
    return value <= threshold if op == "<=" else value >= threshold


def _select(rows: Sequence[Mapping[str, Any]], feature: str, direction: str,
            threshold: float) -> list[Mapping[str, Any]]:
    kept = []
    for row in rows:
        card = dict(row.get("features") or {})
        if feature not in card:
            continue
        value = card[feature]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if _holds(direction, float(value), float(threshold)):
            kept.append(row)
    return kept


def _reading(selected: Sequence[Mapping[str, Any]],
             policy: Policy) -> dict[str, Any]:
    gains = [float(row.get("gain") or 0.0) for row in selected]
    if not gains:
        return {"treated": 0, "aggregate_gain": None, "harmed_fraction": None,
                "max_single_series_harm": None,
                "lines": {"coverage_floor": False, "aggregate": False,
                          "harmed_fraction": False, "single_series_harm": False},
                "feasible": False}
    aggregate = sum(gains) / len(gains)
    harmed = sum(1 for value in gains if value < -policy.material) / len(gains)
    worst = max(0.0, -min(gains))
    lines = {
        "coverage_floor": len(gains) >= policy.min_treated,
        "aggregate": aggregate >= policy.material,
        "harmed_fraction": harmed <= policy.max_harmed_fraction,
        "single_series_harm": worst <= policy.max_single_series_harm,
    }
    return {
        "treated": len(gains),
        "aggregate_gain": round(aggregate, 6),
        "harmed_fraction": round(harmed, 4),
        "max_single_series_harm": round(worst, 6),
        "lines": lines,
        "feasible": all(lines.values()),
    }


def _existing_survivors(rows: Sequence[Mapping[str, Any]],
                        existing: Sequence[Mapping[str, Any]] | None,
                        ) -> list[Mapping[str, Any]]:
    """The bank rows the Scope already selects, before the new clause.

    Passing the whole bank and the current predicate is safer than asking the
    caller to pre-filter: the tool then cannot be handed a row set that the
    Scope does not actually reach and calibrate a threshold against it.
    """
    kept = list(rows)
    for clause in existing or ():
        kept = _select(kept, str(clause["feature"]), str(clause["op"]),
                       float(clause["threshold"]))
    return kept


def calibrate(*, feature: str, direction: str,
              rows: Sequence[Mapping[str, Any]],
              bins: Any = None,
              policy: Policy = BOUNDED_RISK_V1,
              existing_clauses: Sequence[Mapping[str, Any]] | None = None,
              vocabulary: Sequence[str] = VOCABULARY,
              ) -> dict[str, Any]:
    """The widest frozen bin edge of ``feature`` that clears ``policy``.

    ``rows`` are the arm's own bank records for one program: each is
    ``{"features": {name: value}, "gain": float}`` plus whatever provenance the
    caller keeps.  ``existing_clauses`` is the predicate being narrowed, so the
    resolved set is the conjunction and not the new clause alone.

    Raises ``NoFeasibleThreshold`` when no edge works.
    """
    feature, direction = str(feature), str(direction)
    if feature not in set(str(name) for name in vocabulary):
        raise scopes.ScopeError(
            "%r is not in the frozen Scope vocabulary; the deployment could "
            "not read it at serving time" % feature)
    if direction not in DIRECTIONS:
        raise scopes.ScopeError(
            "direction must be one of %s, got %r" % (DIRECTIONS, direction))

    edges = _edges_for(feature, bins)
    base = _existing_survivors(rows, existing_clauses)
    tried: list[dict[str, Any]] = []
    for threshold in edges:
        selected = _select(base, feature, direction, threshold)
        reading = _reading(selected, policy)
        tried.append({"threshold": threshold, **reading})
    feasible = [row for row in tried if row["feasible"]]
    if not feasible:
        raise NoFeasibleThreshold(feature, direction, tried)

    widest = max(row["treated"] for row in feasible)
    tied = [row for row in feasible if row["treated"] == widest]
    # Coarser box on a tie: the more permissive edge.  ">=" is more permissive
    # the lower it sits, "<=" the higher.
    chosen = (min(tied, key=lambda row: row["threshold"]) if direction == ">="
              else max(tied, key=lambda row: row["threshold"]))
    clause = scopes.Clause(feature, direction, float(chosen["threshold"]))
    return {
        "outcome": "CALIBRATED",
        "clause": clause.to_dict(),
        "feature": feature,
        "direction": direction,
        "threshold": float(chosen["threshold"]),
        "threshold_is_a_frozen_bin_edge": True,
        "bin_edges": list(edges),
        "treated": chosen["treated"],
        "bank_rows_in_scope_before": len(base),
        "reading": {key: chosen[key] for key in (
            "aggregate_gain", "harmed_fraction", "max_single_series_harm",
            "lines")},
        "candidates_tried": tried,
        "tie_break": "coarser_box" if len(tied) > 1 else "unique_widest",
        "rule": (
            "candidate thresholds are the feature's frozen bin edges; take the "
            "widest edge clearing aggregate >= material, harmed fraction, "
            "single-series harm and the coverage floor; ties take the coarser "
            "box"
        ),
        "policy": policy.to_dict(),
    }


def best_stump(*, rows: Sequence[Mapping[str, Any]],
               bins: Any = None,
               policy: Policy = BOUNDED_RISK_V1,
               vocabulary: Sequence[str] = VOCABULARY,
               existing_clauses: Sequence[Mapping[str, Any]] | None = None,
               ) -> dict[str, Any]:
    """The ScopeFit-only shadow: the tool's own best (feature, direction).

    This is the control sol required.  If searching the whole vocabulary
    reaches the same new-unit performance as the clause Slow named, then the
    revision was not an LLM contribution and the paper may not report it as
    one.  It is recorded beside every Slow proposal and **never deployed**:
    nothing here returns execution rights, and no arm reads it.

    Objective among feasible stumps is the largest aggregate gain, which is the
    same objective ``calibrate`` is feasible against -- so the comparison is
    between who chose the feature, not between two different goals.
    """
    base = _existing_survivors(rows, existing_clauses)
    considered: list[dict[str, Any]] = []
    for feature in vocabulary:
        try:
            edges = _edges_for(str(feature), bins)
        except scopes.ScopeError:
            continue
        for direction in DIRECTIONS:
            for threshold in edges:
                reading = _reading(
                    _select(base, str(feature), direction, threshold), policy)
                considered.append({
                    "feature": str(feature), "direction": direction,
                    "threshold": float(threshold), **reading})
    feasible = [row for row in considered if row["feasible"]]
    if not feasible:
        return {
            "outcome": "NO_FEASIBLE_STUMP",
            "considered": len(considered),
            "vocabulary": [str(name) for name in vocabulary],
            "deployable": False,
        }
    best = max(feasible, key=lambda row: (
        row["aggregate_gain"], row["treated"], -abs(row["threshold"])))
    return {
        "outcome": "BEST_STUMP",
        "clause": scopes.Clause(
            best["feature"], best["direction"], best["threshold"]).to_dict(),
        "feature": best["feature"],
        "direction": best["direction"],
        "threshold": best["threshold"],
        "objective": best["aggregate_gain"],
        "treated": best["treated"],
        "reading": {key: best[key] for key in (
            "aggregate_gain", "harmed_fraction", "max_single_series_harm",
            "lines")},
        "considered": len(considered),
        "feasible_count": len(feasible),
        "vocabulary": [str(name) for name in vocabulary],
        #: Stated in the payload, not only in the docstring: any consumer that
        #: reads this record is told it may not act on it.
        "deployable": False,
        "why_not_deployable": (
            "the shadow measures how much of a Scope revision the LLM "
            "contributed; deploying it would delete the contrast it exists to "
            "provide"
        ),
    }


def clause_from_slow(payload: Mapping[str, Any] | None, *,
                     rows: Sequence[Mapping[str, Any]],
                     bins: Any = None,
                     policy: Policy = BOUNDED_RISK_V1,
                     existing_clauses: Sequence[Mapping[str, Any]] | None = None,
                     vocabulary: Sequence[str] = VOCABULARY,
                     ) -> dict[str, Any]:
    """Turn what Slow returned into a calibrated clause, plus its shadow.

    Slow's schema (``slow_scope_clause_v1``) still carries a ``threshold``
    field, and it is deliberately not removed: the schema is a dependency SHA
    of every snapshot, and rotating the lock to delete a field the Runtime can
    simply ignore would be a change to the frozen face for no method gain.  The
    field is dropped here instead, and the drop is recorded, so "Slow does not
    author the number" is enforced by this function rather than asserted by a
    schema.
    """
    clause = dict((payload or {}).get("scope_clause") or {})
    notes: list[str] = []
    if not clause:
        return {"outcome": "SLOW_ABSTAINED", "notes": notes,
                "rationale": (payload or {}).get("rationale")}
    if clause.get("threshold") is not None:
        notes.append(LLM_THRESHOLD_IGNORED)
    feature, direction = str(clause.get("feature")), str(clause.get("op"))
    shadow = best_stump(rows=rows, bins=bins, policy=policy,
                        vocabulary=vocabulary,
                        existing_clauses=existing_clauses)
    try:
        calibrated = calibrate(
            feature=feature, direction=direction, rows=rows, bins=bins,
            policy=policy, existing_clauses=existing_clauses,
            vocabulary=vocabulary)
    except NoFeasibleThreshold as exc:
        return {**exc.to_dict(), "notes": notes, "shadow": shadow,
                "slow_threshold_as_returned": clause.get("threshold"),
                "rationale": (payload or {}).get("rationale")}
    except scopes.ScopeError as exc:
        return {"outcome": "SLOW_CLAUSE_UNUSABLE", "why": str(exc),
                "notes": notes, "shadow": shadow,
                "feature": feature, "direction": direction,
                "rationale": (payload or {}).get("rationale")}
    return {
        **calibrated,
        "notes": notes,
        "slow_threshold_as_returned": clause.get("threshold"),
        "rationale": (payload or {}).get("rationale"),
        "shadow": shadow,
        "slow_and_shadow_agree": (
            shadow.get("feature") == feature
            and shadow.get("direction") == direction),
    }


def declared_rules() -> dict[str, Any]:
    """What this tool is allowed to decide, for the contract to embed."""
    return {
        "module": "evaluation.main_protocol_p4.scope_threshold_tool",
        "slow_authors": ["feature", "direction", "rationale"],
        "tool_authors": ["threshold"],
        "tool_may_not": [
            "choose the program",
            "decide whether to act",
            "choose the feature",
            "repair a clause it dislikes",
            "grant execution rights",
        ],
        "threshold_candidates": "the feature's frozen observable bin edges",
        "selection": "widest feasible edge; ties take the coarser box",
        "on_no_feasible_edge": (
            "NO_FEASIBLE_THRESHOLD is returned to Slow, which may change "
            "feature or direction at most twice, then abstains"
        ),
        "vocabulary_size": len(VOCABULARY),
        "vocabulary": list(VOCABULARY),
        "shadow": (
            "best_stump searches the whole vocabulary under the same "
            "feasibility and objective and is recorded beside every proposal; "
            "it is never deployed"
        ),
        "slow_numeric_threshold": (
            "ignored and recorded as %s; the schema keeps the field because "
            "removing it would rotate the snapshot lock for no method gain"
            % LLM_THRESHOLD_IGNORED
        ),
        "thresholds_changed": 0,
        "observation_features_added": 0,
    }
