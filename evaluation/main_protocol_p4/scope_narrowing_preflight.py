"""Prove a revised Scope really narrows -- nothing else in the chain does.

The route table's "monotone narrowing" is a *routing* constraint, not a content
one.  ``FaultRouter.authorize`` refuses ``SCOPE_OVERREACH`` unless the target
class is ``applicability``, and then stops: it never sees the proposed
predicate, so it cannot tell a narrowing from a widening.  ``RISK_GAP`` is not
in ``_APPLICABILITY_DIRECTION`` at all, so even that class gate does not apply
to it.  A Draft ADDed under ``RISK_GAP`` carrying a revised ``serving_scope``
therefore passes every existing check with a Scope that reaches *further* than
the one it replaced.  This module is what stops that.

Two checks, and the structural one is the load-bearing half
---------------------------------------------------------
A semantic check -- "the revised predicate resolves to a subset of the series
the original selected" -- is **cohort-local**.  It says nothing about the next
Target, where the same pair of predicates can resolve the other way round.  A
Skill exists to transfer, so a Scope that narrows only on the cohort it was
derived from is exactly the failure the ScopeSpec design was built to prevent.

The structural check is what transfers: ``ScopeSpec.resolve`` conjoins its
clauses, so a predicate whose clause set is a **superset** of the original's
selects a subset of the original's series on *every* cohort, by construction
rather than by measurement.  Both are required here -- structural because it
generalises, semantic because it confirms the structure was read correctly.

Strictness is required too.  A revision that excludes nobody on the very
evidence it was derived from has not responded to that evidence; it would
consume a Slow call and a Support receipt to restate the program that was
already refused.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evaluation.main_protocol_p4 import scope_spec as scopes

#: What a single Slow revision may add.  One clause, so the revision stays
#: readable and its effect attributable; more would make "which clause bought
#: the safety" unanswerable from the artifact.
MAX_ADDED_CLAUSES = 1

#: What the whole lifecycle may add, across both revisions (P4U-v3).  v2 gave a
#: Draft exactly one clause and one delayed reading, then destroyed it -- so a
#: revision that cleared three of the four lines and missed the fourth had no
#: second move, and the run could not distinguish "Slow cannot bound the tail"
#: from "one clause was not enough".  Two is the whole budget: it is what the
#: held-in geometry can support without the second reading becoming a search.
MAX_TOTAL_ADDED_CLAUSES = 2


@dataclass(frozen=True)
class NarrowingVerdict:
    """Why a revised Scope was accepted or refused, in a form the artifact keeps."""

    accepted: bool
    reason: str
    added_clauses: tuple[dict[str, Any], ...] = ()
    original_resolved: int | None = None
    proposed_resolved: int | None = None
    excluded_series: int | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    #: How many clauses this predicate has added since the frozen initialiser
    #: wrote it -- ``None`` when no root was supplied to compare against.
    total_added_since_root: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "added_clauses": [dict(clause) for clause in self.added_clauses],
            "original_resolved": self.original_resolved,
            "proposed_resolved": self.proposed_resolved,
            "excluded_series": self.excluded_series,
            "checks": dict(self.checks),
            "max_added_clauses": MAX_ADDED_CLAUSES,
            "total_added_since_root": self.total_added_since_root,
            "max_total_added_clauses": MAX_TOTAL_ADDED_CLAUSES,
        }


def _spec(payload: Mapping[str, Any]) -> scopes.ScopeSpec:
    return scopes.ScopeSpec.from_dict(dict(payload))


def _refuse(reason: str, **kwargs: Any) -> NarrowingVerdict:
    return NarrowingVerdict(accepted=False, reason=reason, **kwargs)


def validate_narrowing(
    original: Mapping[str, Any],
    proposed: Mapping[str, Any],
    *,
    features: Mapping[str, Mapping[str, float]] | None = None,
    available_features: Sequence[str] | None = None,
    root: Mapping[str, Any] | None = None,
) -> NarrowingVerdict:
    """Accept ``proposed`` only if it is a strict, structural narrowing.

    ``features`` are the served series' deployment-visible readings at the
    origin the revision is being judged on.  Supplying them turns on the
    semantic half; without them only the structural half can be checked, and
    the verdict says so rather than implying a confirmation it did not make.

    ``root`` is the predicate the frozen initialiser wrote, before any Slow
    revision.  It is what bounds the *lifecycle* rather than the step: without
    it, a chain of individually-legal one-clause narrowings could add clauses
    without end, and "at most one added clause" would be true of every step
    while being false of the Skill that resulted.  Supplied on the second
    revision, ``None`` on the first, where ``original`` is itself the root.
    """
    try:
        before, after = _spec(original), _spec(proposed)
    except (scopes.ScopeError, KeyError, TypeError) as exc:
        return _refuse("proposed_scope_is_not_a_legal_spec: %s" % exc)

    checks: dict[str, bool] = {}
    root_spec = None
    if root is not None:
        try:
            root_spec = _spec(root)
        except (scopes.ScopeError, KeyError, TypeError) as exc:
            return _refuse("root_scope_is_not_a_legal_spec: %s" % exc)

    # Abstention is a legitimate action, but it is not a narrowing: it deploys
    # nothing while passing every risk budget, so it must not reach the Draft
    # path by looking like a revision.
    if after.kind != "serving_series_predicate":
        return _refuse(
            "a revision must stay a predicate; %r deploys nothing and would "
            "clear the risk budget by treating no series" % after.kind,
            checks=checks)
    if before.kind != "serving_series_predicate":
        return _refuse(
            "the original Scope is %r, which has no clause set to narrow"
            % before.kind, checks=checks)

    original_clauses = set(before.clauses)
    proposed_clauses = set(after.clauses)

    # Structural: the guarantee that survives the next cohort.
    kept = original_clauses <= proposed_clauses
    checks["keeps_every_original_clause"] = kept
    if not kept:
        dropped = sorted(
            clause.describe() if hasattr(clause, "describe")
            else "%s %s %g" % (clause.feature, clause.op, clause.threshold)
            for clause in original_clauses - proposed_clauses)
        return _refuse(
            "a clause was dropped or rewritten (%s); that can widen the Scope "
            "on another cohort even where it narrows on this one"
            % ", ".join(dropped), checks=checks)

    added = tuple(
        clause for clause in after.clauses if clause not in original_clauses)
    within_budget = len(added) <= MAX_ADDED_CLAUSES
    checks["adds_at_most_one_clause"] = within_budget
    if not within_budget:
        return _refuse(
            "a revision may add at most %d clause, got %d"
            % (MAX_ADDED_CLAUSES, len(added)),
            added_clauses=tuple(c.to_dict() for c in added), checks=checks)
    if not added:
        return _refuse(
            "the revised Scope is the original one; a revision that changes "
            "nothing has not responded to the evidence that triggered it",
            checks=checks)

    # The lifecycle bound.  Checked against the initialiser's own predicate,
    # not against the previous revision, so it cannot be walked past one legal
    # step at a time.
    total_added: int | None = None
    if root_spec is not None:
        root_clauses = set(root_spec.clauses)
        kept_root = root_clauses <= proposed_clauses
        checks["keeps_every_root_clause"] = kept_root
        if not kept_root:
            return _refuse(
                "the revision no longer contains the predicate the frozen "
                "initialiser wrote, so it is not a narrowing of it",
                added_clauses=tuple(c.to_dict() for c in added), checks=checks)
        total_added = len(proposed_clauses - root_clauses)
        within_lifecycle = total_added <= MAX_TOTAL_ADDED_CLAUSES
        checks["within_lifecycle_clause_budget"] = within_lifecycle
        if not within_lifecycle:
            return _refuse(
                "this Scope has added %d clauses since the initialiser wrote "
                "it; the lifecycle allows %d"
                % (total_added, MAX_TOTAL_ADDED_CLAUSES),
                added_clauses=tuple(c.to_dict() for c in added), checks=checks,
                total_added_since_root=total_added)

    if available_features is not None:
        try:
            after.validate_against(list(available_features))
        except scopes.ScopeError as exc:
            return _refuse(
                "the added clause names something the deployment cannot "
                "observe: %s" % exc,
                added_clauses=tuple(c.to_dict() for c in added), checks=checks)
        checks["added_clause_is_deployment_visible"] = True

    added_dicts = tuple(clause.to_dict() for clause in added)
    if features is None:
        # Structure alone already guarantees the subset property, so this is an
        # acceptance -- but it is recorded as unconfirmed rather than dressed up
        # as a measurement that was never taken.
        return NarrowingVerdict(
            accepted=True,
            reason="structurally_narrower_semantics_unchecked",
            added_clauses=added_dicts, checks=checks,
            total_added_since_root=total_added)

    before_set = before.resolve(features)
    after_set = after.resolve(features)
    checks["resolves_to_a_subset"] = after_set <= before_set
    checks["excludes_at_least_one_series"] = after_set < before_set
    verdict_fields = {
        "added_clauses": added_dicts,
        "original_resolved": len(before_set),
        "proposed_resolved": len(after_set),
        "excluded_series": len(before_set - after_set),
        "checks": checks,
        "total_added_since_root": total_added,
    }
    # A conjunction cannot widen, so this failing means the two specs were not
    # what they claimed -- worth failing loudly rather than trusting structure.
    if not after_set <= before_set:
        return _refuse(
            "the revised predicate resolves outside the original Scope, which "
            "a conjunction cannot do; the specs disagree with their clauses",
            **verdict_fields)
    if not after_set < before_set:
        return _refuse(
            "the added clause excludes no series here, so the revision does "
            "not answer the refusal it was raised by", **verdict_fields)
    return NarrowingVerdict(
        accepted=True, reason="strictly_narrower", **verdict_fields)


def validate_program_frozen(
    original_steps: Sequence[Mapping[str, Any]],
    proposed_steps: Sequence[Mapping[str, Any]],
) -> NarrowingVerdict:
    """The Draft must carry the probe's program, operators and parameters alike.

    Slow is authorized to revise the Scope this round and nothing else.  A
    changed parameter would make the delayed reading measure a different
    program from the one the refusal was about.
    """
    def _canonical(steps: Sequence[Mapping[str, Any]]) -> list[tuple]:
        return [
            (str(step.get("op")),
             tuple(sorted((str(k), v) for k, v in dict(
                 step.get("params") or {}).items())))
            for step in steps
        ]

    before, after = _canonical(original_steps), _canonical(proposed_steps)
    if before == after:
        return NarrowingVerdict(
            accepted=True, reason="program_frozen",
            checks={"program_is_unchanged": True})
    return _refuse(
        "the Draft's program differs from the probed program; this round "
        "authorizes a Scope revision only",
        checks={"program_is_unchanged": False})
