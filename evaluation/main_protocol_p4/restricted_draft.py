"""A Draft that failed the delayed gate is restricted, not destroyed.

The geometry v2 actually ran
---------------------------
One Slow call, one added clause, one delayed reading, then the Draft was
dropped and the next origin started from nothing.  That geometry cannot produce
the evidence the protocol asks for.  AGENTS §3 permits an Episode or a
Target-local Draft formed in one round to enter the next, and §5's evidence of
evolution is precisely *local conflict -> bounded Scope revision -> independent
re-verification -> survival*.  A lifecycle that kills the Draft at "local
conflict" never reaches the second link.

The live reading is what makes this concrete rather than theoretical.  At origin
2136 a revision cleared three of the four delayed lines -- coverage 7/20,
aggregate +0.182, harmed fraction 0.05 -- and missed only the single-series line,
at 0.921 against a 0.30 budget, with exactly one served series over it.  Under
v2 that Draft was discarded.  It had no second move, and the run could not
distinguish "Slow cannot bound the tail with deployment-visible features" from
"one clause was not enough".

What a restriction is, and what it is not
-----------------------------------------
A restricted Draft is **not deployable**: it is not written to the active
snapshot, it is not retrieved, it is not counted as a Skill, and no reading is
taken through it.  It is a Runner-side record of a program and the predicate it
had reached, kept so that one more bounded revision can be attempted at the next
held-in origin.

What stays frozen across the restriction
----------------------------------------
The program, operators and parameters alike.  The risk thresholds.  The operator
set, the Observation vocabulary and the coverage floor.  The total LLM budget.
The only thing that may move is the predicate, and only by adding clauses --
at most one more, and at most two since the frozen initialiser wrote it, which
``scope_narrowing_preflight`` verifies against the root rather than against the
previous revision so the bound cannot be walked past one legal step at a time.

HEC-1 (W3): keeping a Draft is not the same as letting it keep narrowing
--------------------------------------------------------------------------
Source-v3 lost all three of its delayed-passing Drafts at the independent
re-encounter, and the three failures had three different mechanisms.  Narrowing
the Scope answers only one of them:

* the tail was carried entirely by series that had just entered the predicate
  -- the Scope really is too wide, and a clause can exclude them (``REVISABLE``);
* the predicate resolved to almost nobody, so the reading failed the coverage
  floor -- nothing is known to be wrong with the Skill and narrowing would only
  make the coverage worse (``WAITING``);
* series that were already inside the Scope, and had helped, now hurt -- the
  conditional effect itself moved, which is an Observation or Program signal.
  Narrowing here would repair the wrong surface (``FLAGGED``).

So the failed line decides the attribution, the attribution decides the state,
and the state decides which actions are available.  ``FLAGGED`` outranks
``REVISABLE`` outranks ``WAITING``: if continuing members dominate the damage,
that fact governs even when new entrants were also harmed.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing


def _plain(value: Any) -> Any:
    """A JSON-comparable copy.

    Scopes that came back out of an applied ``EditManifest`` are frozen --
    ``mappingproxy`` inside a tuple -- so two predicates that are equal can
    compare unequal, and ``json.dumps`` renders them as ``repr`` strings
    instead of objects.  Both bit the live v3 artifact.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value

#: How many Slow revisions one Draft's Scope may receive over its whole life.
#: Two: the first is the response to the Support-window refusal, the second is
#: the response to the delayed conflict.  A third would make the delayed reading
#: a search over the very window that is supposed to be the endpoint.
MAX_REVISIONS = 2

#: How many verification readings one Draft may be given, across every state.
#: Three: the reading that restricted it, and two more.  A Draft still open
#: after three windows is evidence about the mechanism, not a Skill candidate.
MAX_VERIFICATION_ATTEMPTS = 3

#: Prefix for the candidate id a restricted Draft re-enters under.  Deliberately
#: not ``cand_skill_``: it is not placed by a Skill card, and matching that
#: prefix would make it count as a supplied candidate in the exploration policy.
RESUPPLY_PREFIX = "resupplied_draft_"

WAITING = "WAITING"
REVISABLE = "REVISABLE"
FLAGGED = "FLAGGED"

#: Higher wins.  A reading can fail several lines at once, and the mechanism
#: that forbids narrowing has to be the one that decides.
STATE_PRIORITY = {WAITING: 0, REVISABLE: 1, FLAGGED: 2}

#: Why a Draft was archived, by the state it was archived from.
CLOSE_REASONS = {
    WAITING: "PATTERN_NOT_REENCOUNTERED",
    FLAGGED: "EFFECT_NONSTATIONARY",
    REVISABLE: "REVISION_BUDGET_EXHAUSTED",
}


def attribute(treated_prev: Sequence[str] | None,
              treated_now: Sequence[str] | None,
              per_series_gain: Mapping[str, float] | None,
              material: float) -> dict[str, Any]:
    """Split the treated set of one window into new / continuing / left.

    ``treated_prev`` is the set the previous verification face resolved to, so a
    series is "continuing" only if the Skill actually treated it before -- not
    merely if it existed.  ``per_series_gain`` is keyed by series id on *this*
    window; series it does not mention contribute nothing.

    No Outcome is read here that the gate did not already read: the gains are
    the ones the reading returned.
    """
    previous = {str(uid) for uid in (treated_prev or ())}
    current = {str(uid) for uid in (treated_now or ())}
    gains = {str(uid): float(value)
             for uid, value in dict(per_series_gain or {}).items()}
    new_entrant = sorted(current - previous)
    continuing = sorted(current & previous)
    left = sorted(previous - current)

    def _harm(members: Sequence[str]) -> float:
        return float(sum(-gains[uid] for uid in members
                         if uid in gains and gains[uid] < -material))

    harm_new, harm_continuing = _harm(new_entrant), _harm(continuing)
    harmed = sorted(uid for uid in current
                    if uid in gains and gains[uid] < -material)
    if harm_continuing > harm_new:
        dominant = "continuing"
    elif harm_new > harm_continuing:
        dominant = "new_entrant"
    else:
        # Includes the case where nothing was harmed at all: no side dominates,
        # and the caller must not read "tie" as "new entrants are to blame".
        dominant = "neither"
    return {
        "new_entrant": new_entrant,
        "continuing": continuing,
        "left": left,
        "harmed": harmed,
        "harm_from_new_entrant": round(harm_new, 6),
        "harm_from_continuing": round(harm_continuing, 6),
        "dominant": dominant,
        "sign_flipped_continuing": sorted(
            uid for uid in continuing
            if uid in gains and gains[uid] < -material),
        "treated_prev": sorted(previous),
        "treated_now": sorted(current),
    }


def classify_failure(*, failed_lines: Sequence[str],
                     per_series_gain: Mapping[str, float] | None,
                     treated_prev: Sequence[str] | None,
                     treated_now: Sequence[str] | None,
                     material: float) -> dict[str, Any]:
    """Which state a failed verification reading puts the Draft in, and why.

    Returns the state together with the attribution it was derived from, so the
    artifact records the reasoning rather than only the label.  An empty
    ``failed_lines`` is not a failure and returns ``state=None``.
    """
    lines = [str(name) for name in (failed_lines or ())]
    facts = attribute(treated_prev, treated_now, per_series_gain, material)
    if not lines:
        return {"state": None, "reason": "the reading passed every line",
                "failed_lines": [], "attribution": facts}
    if set(lines) == {"coverage_floor"}:
        return {
            "state": WAITING,
            "reason": (
                "only the coverage floor failed: the predicate resolved to too "
                "few series to take a reading on, which measures how prevalent "
                "the pattern is in this window and not whether the Skill works"
            ),
            "failed_lines": lines, "attribution": facts,
        }
    if facts["dominant"] == "continuing":
        return {
            "state": FLAGGED,
            "reason": (
                "the damage is dominated by series the Skill had already "
                "treated, so the conditional effect moved rather than the "
                "Scope reaching too far; narrowing would repair the wrong "
                "surface"
            ),
            "failed_lines": lines, "attribution": facts,
        }
    if facts["dominant"] == "new_entrant":
        return {
            "state": REVISABLE,
            "reason": (
                "every materially harmed series had just entered the "
                "predicate, so the Scope is too wide on this window and one "
                "more clause can answer the evidence"
            ),
            "failed_lines": lines, "attribution": facts,
        }
    return {
        # No harmed series at all, yet a line failed: the effect simply is not
        # material here any more.  That is the FLAGGED mechanism, and treating
        # it as REVISABLE would spend a clause on damage that does not exist.
        "state": FLAGGED,
        "reason": (
            "a line failed with no materially harmed series to attribute it "
            "to, so the effect is no longer material on this window rather "
            "than concentrated in a removable subset"
        ),
        "failed_lines": lines, "attribution": facts,
    }


@dataclass
class RestrictedDraft:
    """One program, the predicate it has reached, and why it is not deployed."""

    draft_id: str
    program_steps: tuple[tuple[str, dict[str, Any]], ...]
    root_scope: dict[str, Any]
    current_scope: dict[str, Any]
    revisions: int
    created_at_origin: int
    delayed_failures: list[dict[str, Any]] = field(default_factory=list)
    support_readings: list[dict[str, Any]] = field(default_factory=list)
    revision_history: list[dict[str, Any]] = field(default_factory=list)
    closed: str | None = None

    #: HEC-1 (W3).  ``None`` on a Draft that has not yet failed a classified
    #: verification face -- which is every Draft the Source-v3 geometry makes,
    #: so that runner's behaviour is unchanged by these fields existing.
    state: str | None = None
    verification_attempts: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    #: Task x Consumer x typed Program x root Scope (sol v1.1 A).  Carried on
    #: the Draft so a closed lineage can be recognised without re-deriving it,
    #: which is what stops a closed key reopening under a fresh shell with its
    #: revision and verification counters back at zero.
    census_key: str | None = None

    #: Never true in this class.  Present so that any consumer asking "may I
    #: deploy this?" gets an answer rather than having to infer one.
    deployable: bool = False

    def may_revise(self) -> bool:
        return self.closed is None and self.revisions < MAX_REVISIONS

    def may_add_clause(self) -> bool:
        """Whether Slow may be asked for another clause for this Draft.

        Stricter than ``may_revise``: a ``FLAGGED`` Draft has revisions left in
        the budget and is still explicitly forbidden to narrow, because the
        evidence points at the Observation or the Program rather than at the
        Scope.  A ``WAITING`` Draft is not forbidden -- it simply has no
        evidence to narrow on yet, and its automatic re-verification does not
        consume a revision.
        """
        return self.may_revise() and self.state != FLAGGED

    def may_verify(self) -> bool:
        return (self.closed is None
                and self.verification_attempts < MAX_VERIFICATION_ATTEMPTS)

    def clauses_added_so_far(self) -> int:
        root = [tuple(sorted(c.items())) for c in (self.root_scope.get("predicate") or ())]
        now = [tuple(sorted(c.items())) for c in (self.current_scope.get("predicate") or ())]
        return max(len(set(now) - set(root)), 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "program_steps": [{"op": op, "params": dict(params)}
                              for op, params in self.program_steps],
            "root_scope": dict(self.root_scope),
            "current_scope": dict(self.current_scope),
            "revisions": self.revisions,
            "clauses_added_since_root": self.clauses_added_so_far(),
            "max_revisions": MAX_REVISIONS,
            "max_total_added_clauses": narrowing.MAX_TOTAL_ADDED_CLAUSES,
            "created_at_origin": self.created_at_origin,
            "delayed_failures": [dict(row) for row in self.delayed_failures],
            "support_readings": [dict(row) for row in self.support_readings],
            "revision_history": [dict(row) for row in self.revision_history],
            "census_key": self.census_key,
            "state": self.state,
            "verification_attempts": self.verification_attempts,
            "max_verification_attempts": MAX_VERIFICATION_ATTEMPTS,
            "may_add_clause": self.may_add_clause(),
            "history": [dict(row) for row in self.history],
            "closed": self.closed,
            "deployable": self.deployable,
            "why_not_deployable": (
                "a Draft that failed the delayed gate is kept only so one more "
                "bounded Scope revision can be attempted; it is never written "
                "to the active snapshot and never serves a prediction"
            ),
        }


class DraftLedger:
    """The Runner's record of restricted Drafts, and the resupply it drives.

    Deliberately outside the harness snapshot.  Putting a restricted Draft into
    the snapshot -- even guarded -- would make it appear in the Skill inventory,
    and the whole claim this protocol is trying to establish is a statement
    about how many Skills survived.  A record that is not a Skill must not be
    counted as one, so it does not live where Skills live.
    """

    def __init__(self) -> None:
        self.drafts: list[RestrictedDraft] = []
        self._minted = 0

    # ---- creation ---------------------------------------------------------

    def restrict(self, *, program_steps: Sequence[tuple[str, Mapping[str, Any]]],
                 root_scope: Mapping[str, Any],
                 current_scope: Mapping[str, Any],
                 origin: int,
                 delayed_reading: Mapping[str, Any],
                 support_reading: Mapping[str, Any] | None = None,
                 revisions: int = 1) -> RestrictedDraft:
        self._minted += 1
        draft = RestrictedDraft(
            draft_id="%s%d" % (RESUPPLY_PREFIX, self._minted),
            program_steps=tuple((str(op), dict(params))
                                for op, params in program_steps),
            root_scope=dict(root_scope),
            current_scope=dict(current_scope),
            revisions=int(revisions),
            created_at_origin=int(origin),
        )
        draft.delayed_failures.append({
            "origin": int(origin),
            "delayed_origin": delayed_reading.get("delayed_origin"),
            "failed_lines": [name for name, ok in
                             (delayed_reading.get("lines") or {}).items()
                             if not ok],
            "reading": dict(delayed_reading),
        })
        if support_reading is not None:
            draft.support_readings.append(dict(support_reading))
        self.drafts.append(draft)
        return draft

    def lineage_keys(self) -> set[str]:
        """Every census key this ledger has ever opened -- open and closed.

        Closed included on purpose (sol v1.1 A): a closed key that could reopen
        would arrive with ``revisions=0`` and ``verification_attempts=0``, which
        is how a Draft walks past its bounds one legal-looking step at a time.
        """
        return {str(draft.census_key) for draft in self.drafts
                if draft.census_key}

    def by_census_key(self, key: str | None) -> RestrictedDraft | None:
        if not key:
            return None
        for draft in self.drafts:
            if draft.census_key == str(key):
                return draft
        return None

    def open_restricted(self, *,
                        program_steps: Sequence[tuple[str, Mapping[str, Any]]],
                        root_scope: Mapping[str, Any],
                        current_scope: Mapping[str, Any],
                        origin: int,
                        census_key: str | None = None,
                        provenance: Mapping[str, Any] | None = None,
                        ) -> RestrictedDraft:
        """A Draft opened by the outer loop, before any verification face.

        The difference from ``restrict`` is what it is *not*: there is no
        delayed failure to record, because this Draft was never deployed.  It
        came out of a census of already-processed units plus a replay screen,
        both of which are selection, not authorisation -- so it enters with no
        state, no verification attempts and no deployment rights, and it earns
        an ``Active`` only by clearing Support and delayed on a *new* unit.
        """
        if census_key and self.by_census_key(census_key) is not None:
            raise ValueError(
                "census key %r already has a lineage in this course; reopening "
                "it under a new shell would reset the revision and "
                "verification counters" % census_key)
        self._minted += 1
        draft = RestrictedDraft(
            draft_id="%s%d" % (RESUPPLY_PREFIX, self._minted),
            program_steps=tuple((str(op), dict(params))
                                for op, params in program_steps),
            root_scope=dict(root_scope),
            current_scope=dict(current_scope),
            revisions=0,
            created_at_origin=int(origin),
            census_key=str(census_key) if census_key else None,
        )
        draft.history.append({
            "event": "opened_by_outer_loop",
            "origin": int(origin),
            "provenance": dict(provenance) if provenance else None,
            "state_after": None,
        })
        self.drafts.append(draft)
        return draft

    # ---- verification faces ----------------------------------------------

    def record_verification(self, draft: RestrictedDraft, *,
                            window: int,
                            failed_lines: Sequence[str],
                            per_series_gain: Mapping[str, float] | None,
                            treated_prev: Sequence[str] | None,
                            treated_now: Sequence[str] | None,
                            material: float,
                            consumes_attempt: bool = True,
                            reading: Mapping[str, Any] | None = None,
                            ) -> dict[str, Any]:
        """One verification face for one Draft: classify, transition, record.

        ``consumes_attempt=False`` is the ``WAITING`` path: a Draft whose
        predicate finally resolves to enough series gets that reading for free,
        because the wait was never the Draft's fault.
        """
        verdict = classify_failure(
            failed_lines=failed_lines, per_series_gain=per_series_gain,
            treated_prev=treated_prev, treated_now=treated_now,
            material=material)
        if consumes_attempt:
            draft.verification_attempts += 1
        state = verdict["state"]
        if state is not None:
            # Priority, not recency: once continuing members have been shown to
            # dominate, a later window that happens to look revisable must not
            # unlock narrowing.
            current = STATE_PRIORITY.get(draft.state or "", -1)
            if STATE_PRIORITY[state] >= current:
                draft.state = state
        entry = {
            "event": "verification",
            "window": int(window),
            "treated_prev": verdict["attribution"]["treated_prev"],
            "treated_now": verdict["attribution"]["treated_now"],
            "attribution": verdict["attribution"],
            "per_series_gain": {
                str(uid): float(value)
                for uid, value in dict(per_series_gain or {}).items()},
            "failed_lines": verdict["failed_lines"],
            "classified_state": state,
            "classification_reason": verdict["reason"],
            "consumed_attempt": bool(consumes_attempt),
            "verification_attempts": draft.verification_attempts,
            "state_after": draft.state,
            "reading": dict(reading) if reading else None,
        }
        draft.history.append(entry)
        if state is None:
            return entry
        if not draft.may_verify():
            self.close(draft, CLOSE_REASONS.get(
                draft.state or REVISABLE, "REVISION_BUDGET_EXHAUSTED"))
            entry["closed"] = draft.closed
        return entry

    def close_unreencountered(self) -> list[dict[str, Any]]:
        """End of course: a Draft still ``WAITING`` never met its pattern again.

        Recorded as ``PATTERN_NOT_REENCOUNTERED`` rather than as a Skill
        failure -- nothing about the Skill was tested, and counting it as a
        failure would price pattern prevalence as if it were Skill quality.
        """
        closed = []
        for draft in list(self.drafts):
            if draft.closed is not None:
                continue
            reason = CLOSE_REASONS.get(draft.state or "", "OUT_OF_UNITS")
            if draft.state is None:
                reason = "NEVER_VERIFIED"
            self.close(draft, reason)
            closed.append({"draft_id": draft.draft_id, "state": draft.state,
                           "closed": reason})
        return closed

    # ---- resupply ---------------------------------------------------------

    def open_drafts(self) -> list[RestrictedDraft]:
        return [draft for draft in self.drafts if draft.may_revise()]

    def resupplied_programs(self) -> dict[str, tuple]:
        """What ``run_online_round`` appends to the probe pool, id -> steps."""
        return {draft.draft_id: draft.program_steps
                for draft in self.open_drafts()}

    def resupplied_scopes(self) -> dict[str, dict[str, Any]]:
        """The predicate each resupplied Draft is probed under: its current one.

        Not the root.  The second revision has to start from what the first one
        reached, or the Draft would be re-probed under a Scope it has already
        been shown to fail, and the added-clause budget would buy nothing.
        """
        return {draft.draft_id: dict(draft.current_scope)
                for draft in self.open_drafts()}

    # ---- HEC-1 resupply: by verification, not by revision budget ----------

    def verifiable_drafts(self) -> list[RestrictedDraft]:
        """Every open Draft that may still take a verification face.

        ``open_drafts`` keys resupply on ``may_revise``, which is the v3
        geometry: there the Draft's clause was written in the Support window
        and the next face verified it.  In HEC-1 clauses come from the outer
        loop, so a Draft whose second revision has just been written has spent
        its revision budget and still owes a verification -- keying resupply on
        ``may_revise`` would archive it as ``REVISION_BUDGET_EXHAUSTED`` without
        the second clause ever being read.  The v3 method is left untouched.
        """
        return [draft for draft in self.drafts if draft.may_verify()]

    def resupplied_programs_for_verification(self) -> dict[str, tuple]:
        return {draft.draft_id: draft.program_steps
                for draft in self.verifiable_drafts()}

    def resupplied_scopes_for_verification(self) -> dict[str, dict[str, Any]]:
        return {draft.draft_id: dict(draft.current_scope)
                for draft in self.verifiable_drafts()}

    def by_id(self, draft_id: str) -> RestrictedDraft | None:
        for draft in self.drafts:
            if draft.draft_id == str(draft_id):
                return draft
        return None

    def by_scope(self, scope: Mapping[str, Any] | None) -> RestrictedDraft | None:
        """The open Draft a winning policy *is*, found by what it deploys.

        A restricted Draft does not only come back through a second refusal.
        It can be re-probed at the next origin and simply be **admitted** --
        which is what happened live at origin 2376, where the Draft restricted
        at 2136 was re-probed under its own revised predicate, cleared the
        admission gate outright at +0.641, and went on to pass both promotion
        gates.  Looking the Draft up only by the selected refusal missed that
        case entirely, so the Draft stayed open after it had already become a
        Skill and was handed a further revision at 2856 that cost six LLM calls
        and produced a record in which one program was simultaneously an active
        Skill and an unresolved Draft.
        """
        if not scope:
            return None
        target = json.dumps(_plain(scope), sort_keys=True)
        for draft in self.drafts:
            if (draft.closed is None
                    and json.dumps(_plain(draft.current_scope), sort_keys=True)
                    == target):
                return draft
        return None

    def root_for_scope(self, scope: Mapping[str, Any] | None
                       ) -> Mapping[str, Any] | None:
        """The initialiser's predicate behind a Scope now under revision.

        The narrowing preflight is handed ``(original, proposed)`` and cannot
        know on its own whether ``original`` is itself a revision.  This is how
        the lifecycle budget reaches it: an ``original`` that matches an open
        Draft's current predicate resolves to that Draft's root, and the
        preflight then counts added clauses against the initialiser rather than
        against the previous step.
        """
        if not scope:
            return None
        for draft in self.drafts:
            if dict(draft.current_scope) == dict(scope):
                return dict(draft.root_scope)
        return None

    # ---- outcome ----------------------------------------------------------

    def record_revision(self, draft: RestrictedDraft, *,
                        origin: int, new_scope: Mapping[str, Any],
                        preflight: Mapping[str, Any] | None,
                        support: Mapping[str, Any] | None) -> None:
        if draft.state == FLAGGED:
            raise ValueError(
                "draft %s is FLAGGED: the damage was dominated by series it "
                "had already treated, so narrowing is not an available action"
                % draft.draft_id)
        draft.revision_history.append({
            "origin": int(origin),
            "from_scope": dict(draft.current_scope),
            "to_scope": dict(new_scope),
            "preflight": dict(preflight) if preflight else None,
            "support": dict(support) if support else None,
        })
        draft.current_scope = dict(new_scope)
        draft.revisions += 1

    def close(self, draft: RestrictedDraft, reason: str) -> None:
        draft.closed = str(reason)

    def to_dict(self) -> dict[str, Any]:
        by_state: dict[str, list[str]] = {}
        for draft in self.drafts:
            by_state.setdefault(str(draft.state), []).append(draft.draft_id)
        return {
            "drafts": [draft.to_dict() for draft in self.drafts],
            "open": [draft.draft_id for draft in self.open_drafts()],
            "by_state": by_state,
            "max_revisions": MAX_REVISIONS,
            "max_verification_attempts": MAX_VERIFICATION_ATTEMPTS,
            "held_outside_the_snapshot": (
                "restricted Drafts are Runner records, never Skill-library "
                "entries; the count of surviving Skills stays a statement "
                "about what actually reached the active snapshot"
            ),
        }
