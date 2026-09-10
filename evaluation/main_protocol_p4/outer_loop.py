"""Every k units, learn from the units already processed -- and only from those.

The bottleneck this answers
---------------------------
Until now the Harness decided whether an edit was any good by looking at the
unit that triggered it.  That is a sample of one, and it is also the same unit
the edit was derived from, so selection and verification shared a window.
``S3_EDIT_REJECTED`` and the Source line's zero survivors are both readings of
that geometry rather than of the edits.

The two loops are separated here.  The inner loop (``run_online_round``) probes,
passes the two gates, deploys inside the unit and **writes only Episodes**.  The
outer loop runs every k units and does four things, in this order:

    deterministic census   -- group the bank by task/consumer and behaviour
    Slow proposal          -- at most a couple of calls, semantics only (W2)
    replay screen          -- re-resolve and re-score on cells already processed
    restricted Draft       -- and nothing more; the next new unit decides

What the replay screen may and may not do
-----------------------------------------
It may eliminate.  A candidate that would have breached the risk budget on a
cell this arm has already seen does not need a fresh unit spent on it.  It may
**not** promote: replaying history is not evidence about the future, and a
candidate that survives the screen leaves here as a restricted Draft with no
deployment rights at all.  That boundary is the whole reason the two loops are
worth separating -- if replay could authorise, the outer loop would just be a
larger version of the same selection-equals-verification mistake.

What the bank may contain
-------------------------
Cells this arm, in this ordering, has already processed.  Not the evaluation
face (+144, which is scored and never fed back), not future units, not another
arm's records, and not held-out anything.  ``consolidate`` cannot check
provenance it was not given, so the runner assembles the bank and this module
records the declared boundary it was handed.

Revocation is recommended, never executed
-----------------------------------------
A repeatedly harmful Active Skill produces a revocation *recommendation* in the
step record.  Writing the active snapshot is the runner's move, taken through
the ordinary lifecycle, because an outer-loop module that could edit the active
set would be able to grant execution rights without any gate at all.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import scope_initializer as initializer
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from evaluation.main_protocol_p4 import scope_threshold_tool as tool_module
from SelfEvolvingHarnessTS.methods.ttha import experience_memory as memory

#: The only strings that are a *relation*.  Imported from the module that
#: defines them rather than restated, so the census and the Episode classifier
#: cannot drift into two vocabularies that share a name.
#:
#: M-R0: a bank row used to be read as ``row["relation"] or row["admission"]``,
#: and ``row["admission"]`` carries an ``AdmissionVerdict.reason`` -- a
#: *deployment-admission* verdict, not a relation.  Reasons such as
#: ``within_risk_budget`` are not in this tuple, so they are no longer mistaken
#: for one; a row that carries no usable relation falls through to this
#: module's own gain-based definition instead of silently answering with a
#: string that matches nothing the census counts.
RELATIONS = (
    memory.RELATION_POSITIVE,
    memory.RELATION_NEGATIVE,
    memory.RELATION_CONFLICT,
    memory.RELATION_ABSTAIN,
    memory.RELATION_NEUTRAL,
)

#: How many times a program must appear as POSITIVE in the bank, with no card
#: holding it, before the census proposes an ADD.
#:
#: **One** (sol, 2026-09-03 v1.1).  Two was the mainline's own guard against the
#: n=1 geometry, and it conflicted with the already-adjudicated ladder v2, which
#: prices a supply-tier evidence step at 1.  It was also the binding constraint
#: in Phase S-v1: across thirteen units *no* program was POSITIVE on two, so the
#: census never produced a candidate, Slow was never invoked, and K0 came back
#: empty without the mechanism having been exercised at all.
#:
#: One positive unit does not make a Skill here.  It makes a **restricted
#: Draft**, which carries no deployment rights and must still clear Support and
#: the delayed authoritative gate on a *later, independent* unit before it can
#: activate.  The n=1 worry is answered by that forward verification rather than
#: by refusing to look.
MIN_POSITIVE_UNITS_FOR_ADD = 1

#: How many times an Active Skill must come back CONFLICT or NEGATIVE before the
#: census proposes narrowing it, or recommends revoking it.
MIN_ADVERSE_UNITS_FOR_NARROWING = 2

CANDIDATE_KINDS = ("ADD", "REVISE", "NARROW", "REVOKE")

REPLAY_SCREEN_REJECTED = "REPLAY_SCREEN_REJECTED"


@dataclass
class OuterBudget:
    """What one outer step may spend.  Nothing here is a threshold on evidence."""

    outer_llm_per_step: int = 2
    replay_fits_remaining: int | None = None
    #: Slow may change feature or direction this many times after a
    #: NO_FEASIBLE_THRESHOLD before the candidate is abandoned.
    retries_per_candidate: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "outer_llm_per_step": self.outer_llm_per_step,
            "replay_fits_remaining": self.replay_fits_remaining,
            "retries_per_candidate": self.retries_per_candidate,
        }


@dataclass
class OuterStepRecord:
    """One outer step, in the form the artifact keeps it."""

    k_index: int
    units_in_bank: int = 0
    groups: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    drift_signals: list[dict[str, Any]] = field(default_factory=list)
    revocation_recommendations: list[dict[str, Any]] = field(
        default_factory=list)
    drafts_opened: list[str] = field(default_factory=list)
    drafts_revised: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    slow_calls: int = 0
    replay_fits: int = 0
    shadow_records: list[dict[str, Any]] = field(default_factory=list)
    wall_seconds: float = 0.0
    empty_reason: str | None = None
    bank_boundary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_index": self.k_index,
            "units_in_bank": self.units_in_bank,
            "groups": [dict(row) for row in self.groups],
            "candidate_count": len(self.candidates),
            "candidates": [dict(row) for row in self.candidates],
            "drift_signals": [dict(row) for row in self.drift_signals],
            "revocation_recommendations": [
                dict(row) for row in self.revocation_recommendations],
            "drafts_opened": list(self.drafts_opened),
            "drafts_revised": list(self.drafts_revised),
            "rejected": [dict(row) for row in self.rejected],
            "rejected_count": len(self.rejected),
            "slow_calls": self.slow_calls,
            "replay_fits": self.replay_fits,
            "shadow_records": [dict(row) for row in self.shadow_records],
            "wall_seconds": round(self.wall_seconds, 3),
            "empty_reason": self.empty_reason,
            "bank_boundary": dict(self.bank_boundary),
            "wrote_active": False,
            "why_no_active_write": (
                "the replay screen selects and the Draft carries no deployment "
                "rights; only a new unit's Support and delayed reading can "
                "produce an Active Skill"
            ),
        }


# ---------------------------------------------------------------------------
# the deterministic census
# ---------------------------------------------------------------------------

def _steps_tuple(steps: Any) -> tuple[tuple[str, dict], ...]:
    out: list[tuple[str, dict]] = []
    for step in steps or ():
        if isinstance(step, Mapping):
            out.append((str(step.get("op")), dict(step.get("params") or {})))
        elif isinstance(step, (tuple, list)) and step:
            params = step[1] if len(step) > 1 else {}
            out.append((str(step[0]), dict(params or {})))
    return tuple(out)


def _program_signature(steps: Any) -> str:
    return ">".join("%s(%s)" % (op, json.dumps(params, sort_keys=True))
                    for op, params in _steps_tuple(steps))


def _root_scope_signature(scope: Any) -> str:
    """The predicate a candidate started from, as part of the census key.

    sol's v1.1 ruling: the key is Task x Consumer x full typed Program x **root
    Scope**.  The same program under two different starting predicates treats
    two different sets of series, so merging them would average one card's
    evidence into another's and make "this program is POSITIVE here" a statement
    about no particular deployment.  Only the *root* enters the key -- a Draft's
    later narrowings are revisions of the same card, not a different one.

    Fault type deliberately stays out: it is stratified evidence, not identity.
    """
    if not scope:
        return "all_serving_series"
    kind = str(scope.get("scope_type") or "all_serving_series")
    clauses = sorted(
        "%s%s%g" % (str(clause.get("feature")), str(clause.get("op")),
                    float(clause.get("threshold")))
        for clause in (scope.get("predicate") or ())
    )
    return "%s[%s]" % (kind, ",".join(clauses))


def behaviour_fingerprint(per_series_gain: Mapping[str, float] | None,
                          digits: int = 6) -> str:
    """Dedupe programs by what they did, not by what they are called.

    ``AGENTS`` §5.1's methodology note: of 396 enumerated programs only a
    handful of distinct per-series gain vectors exist on gappy data, because
    four of the eighteen operators do nothing there.  Counting programs before
    deduplicating by effect would inflate every census number.
    """
    rows = sorted((str(uid), round(float(value), digits))
                  for uid, value in dict(per_series_gain or {}).items())
    return json.dumps(rows, separators=(",", ":"))


def exact_reading(per_series_gain: Mapping[str, float] | None) -> str:
    """The reading itself, at full precision, for deciding "is this the same".

    Deliberately **not** ``behaviour_fingerprint``.  That rounds to six
    decimals because its job is to collapse programs whose effects are
    indistinguishable at the resolution the alias rule cares about; borrowing
    it to decide whether two bank rows are the same observation would merge
    -0.3000001 with -0.2999999 and quietly drop a genuinely different reading.
    Two readings count as one observation only when the numbers are equal.

    ``repr`` of a float round-trips exactly, so this compares the values that
    were recorded rather than a summary of them.  No hash: the raw rows stay in
    the bank and this string is only a comparison key.
    """
    rows = sorted((str(uid), repr(float(value)))
                  for uid, value in dict(per_series_gain or {}).items())
    return json.dumps(rows, separators=(",", ":"))


def _relation(row: Mapping[str, Any], material: float) -> str:
    """The row's relation: the Episode's if it has one, else read off the gains.

    ``row["admission"]`` is deliberately *not* consulted.  It holds an
    ``AdmissionVerdict.reason``, which answers "may this deploy?" and not "what
    happened?"; reading it as a relation is what made ``positive_units`` and
    ``adverse_units`` identically zero for the whole of HEC-1.  A declared value
    is honoured only when it is one of the relations ``RELATIONS`` names.
    """
    declared = str(row.get("relation") or "").upper()
    if declared in RELATIONS:
        return declared
    gains = [float(value) for value in
             dict(row.get("per_series_gain") or {}).values()]
    if not gains:
        return "UNKNOWN"
    aggregate = sum(gains) / len(gains)
    harmed = any(value < -material for value in gains)
    if aggregate >= material:
        return "CONFLICT" if harmed else "POSITIVE"
    return "NEGATIVE"


def _unit_key(unit: Any) -> str:
    return json.dumps(unit, sort_keys=True, default=str)


def _alias_classes(groups: Mapping[tuple[str, str], Mapping[str, Any]]
                   ) -> dict[tuple[str, str], tuple[str, str]]:
    """Collapse programs that did the *same thing* on every unit they share.

    ``AGENTS`` §5.1's methodology note: of 396 enumerated programs only a
    handful of distinct per-series gain vectors exist on gappy data, because
    four of the eighteen operators do nothing there.  Two names for one effect
    must not become two census findings, two Slow calls and two Drafts.

    Aliasing is judged per unit and only on units both programs were read on.
    Programs that never met cannot be shown to be aliases, so they stay apart.
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {key: key for key in groups}

    def find(key: tuple[str, str]) -> tuple[str, str]:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    keys = sorted(groups)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            if left[0] != right[0]:  # different task/consumer: never aliases
                continue
            left_prints = groups[left]["fingerprints"]
            right_prints = groups[right]["fingerprints"]
            shared = set(left_prints) & set(right_prints)
            if not shared:
                continue
            if all(left_prints[unit] == right_prints[unit] for unit in shared):
                a, b = find(left), find(right)
                if a != b:
                    # The lexicographically smaller signature represents the
                    # class, so the choice does not depend on bank order.
                    parent[max(a, b)] = min(a, b)
    return {key: find(key) for key in keys}


def census(bank: Sequence[Mapping[str, Any]], *,
           material: float = tool_module.MATERIAL) -> list[dict[str, Any]]:
    """Group the bank by Task x Consumer x typed Program x root Scope.

    sol's v1.1 key.  ``task_consumer_key`` carries Task and Consumer, the
    program signature carries every operator with its order and parameters, and
    the root Scope signature carries the predicate the candidate started from --
    because the same program under two starting predicates is treating two
    different sets of series, and merging them would report one deployment's
    evidence as another's.

    Deterministic and search-free: it reads the gains the inner loop already
    recorded, does no fitting and makes no LLM call, so two runs over the same
    bank produce the same groups in the same order.

    The behaviour fingerprint is what collapses aliases (see ``_alias_classes``)
    and is deliberately *not* part of the key: per-series gains differ from unit
    to unit by construction, so keying on them would give every program one
    group per unit and no accumulation could ever be observed.
    """
    raw: dict[tuple[str, str], dict[str, Any]] = {}
    for row in bank or ():
        task = str(row.get("task_consumer_key") or "")
        signature = _program_signature(row.get("program_steps"))
        root = _root_scope_signature(row.get("serving_scope"))
        key = "%s@%s" % (signature, root)
        fingerprint = str(row.get("behavior_fingerprint")
                          or behaviour_fingerprint(row.get("per_series_gain")))
        unit = _unit_key(row.get("unit"))
        bucket = raw.setdefault((task, key), {
            "task_consumer_key": task,
            "program_signature": signature,
            "root_scope_signature": root,
            "census_key": "%s|%s" % (task, key),
            "program_steps": [
                {"op": op, "params": params}
                for op, params in _steps_tuple(row.get("program_steps"))],
            "root_scope": dict(row.get("serving_scope") or {}) or None,
            "units": [],
            "unit_keys": [],
            "relations": [],
            "readings": [],
            "rows": [],
            "source_skill_ids": [],
            "fingerprints": {},
            "seen_observations": set(),
            "row_tokens": set(),
            "bank_rows_seen": 0,
            "duplicate_observations_dropped": 0,
        })
        bucket["bank_rows_seen"] += 1
        relation = _relation(row, material)
        # One unit probing one program twice writes two bank rows.  If the two
        # readings are identical there is one observation, and counting it
        # twice inflates ``adverse_units`` against a threshold that is written
        # in units -- and doubles the weight of the same series in the evidence
        # the threshold tool searches.  Deduplication happens here, in the
        # census view; the bank keeps every row it was handed.
        #
        # Only *exact* repeats collapse: same unit, the same numbers, the same
        # relation.  Sameness is decided on ``exact_reading`` -- the recorded
        # per-series values -- and not on the six-decimal behaviour
        # fingerprint, which exists for the alias rule and would merge
        # -0.3000001 with -0.2999999.  Two different readings of one unit are
        # left as two entries and the unit is flagged instead, because
        # choosing between them (first, last, worst, or adverse-wins) is a
        # protocol question this module must not answer on its own.
        reading = exact_reading(row.get("per_series_gain"))
        observation = (unit, reading, relation)
        if observation in bucket["seen_observations"]:
            bucket["duplicate_observations_dropped"] += 1
            continue
        bucket["seen_observations"].add(observation)
        bucket["units"].append(row.get("unit"))
        bucket["unit_keys"].append(unit)
        bucket["fingerprints"][unit] = fingerprint
        bucket["relations"].append(relation)
        bucket["readings"].append(reading)
        skill_id = row.get("source_skill_id")
        if skill_id and skill_id not in bucket["source_skill_ids"]:
            bucket["source_skill_ids"].append(str(skill_id))
        features = dict(row.get("features") or {})
        for uid, gain in dict(row.get("per_series_gain") or {}).items():
            # Exact, for the same reason: a rounded token would drop a
            # genuinely different reading of one series out of the evidence
            # the threshold tool searches.
            token = (unit, str(uid), repr(float(gain)))  # exact, per unit
            if token in bucket["row_tokens"]:
                continue
            bucket["row_tokens"].add(token)
            bucket["rows"].append({
                "unit": row.get("unit"),
                "series": str(uid),
                "features": dict(features.get(str(uid)) or {}),
                "gain": float(gain),
            })

    representative = _alias_classes(raw)
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for key, bucket in sorted(raw.items()):
        target = representative[key]
        into = merged.get(target)
        if into is None:
            merged[target] = {**bucket, "aliases": []}
            continue
        if key != target:
            into["aliases"].append(bucket["program_signature"])
        # Rows: exact, so an alias that read one series differently keeps both
        # numbers in the evidence the threshold tool searches.
        for row in bucket["rows"]:
            token = (row["unit"] and _unit_key(row["unit"]), row["series"],
                     repr(float(row["gain"])))
            if token not in into["row_tokens"]:
                into["row_tokens"].add(token)
                into["rows"].append(row)
        # Observations: a **union**, not first-wins per unit.  Dropping an
        # alias's reading because the representative already had one for that
        # unit is how a unit that disagrees with itself came out of the merge
        # looking unanimous -- and then cast the vote its own evidence denies.
        # Only an exact repeat of an observation already held collapses, and
        # that collapse is counted like any other.
        for unit, unit_key, relation, reading in zip(
                bucket["units"], bucket["unit_keys"], bucket["relations"],
                bucket["readings"]):
            observation = (unit_key, reading, relation)
            if observation in into["seen_observations"]:
                into["duplicate_observations_dropped"] += 1
                continue
            into["seen_observations"].add(observation)
            into["units"].append(unit)
            into["unit_keys"].append(unit_key)
            into["relations"].append(relation)
            into["readings"].append(reading)
        for skill_id in bucket["source_skill_ids"]:
            if skill_id not in into["source_skill_ids"]:
                into["source_skill_ids"].append(skill_id)
        into["bank_rows_seen"] += bucket["bank_rows_seen"]
        into["duplicate_observations_dropped"] += bucket[
            "duplicate_observations_dropped"]

    ordered = sorted(merged.values(),
                     key=lambda g: (g["task_consumer_key"],
                                    g["program_signature"],
                                    g["root_scope_signature"]))
    for group in ordered:
        counts = {name: group["relations"].count(name)
                  for name in sorted(set(group["relations"]))}
        # Observations, kept as they were: how many readings this group holds.
        group["relation_counts"] = counts
        group["observations"] = len(group["relations"])

        # Votes, which is what the thresholds are written in.  One unit, one
        # vote, and only when the unit says one thing (R1, Fable's strict
        # consensus rule).  A unit whose readings disagree contributes to
        # neither ``positive_units`` nor ``adverse_units`` and is listed
        # instead: NEGATIVE beside CONFLICT is a disagreement like any other,
        # not an adverse majority, because "the effect is gone here" and "the
        # effect is still there but it hurts someone" are different findings
        # and picking one of them is a protocol decision, not a census one.
        # Derived from the observations the group actually holds, after any
        # alias merge, so nothing about a unit can be decided by which bucket
        # happened to represent its alias class.
        by_unit: dict[str, set[str]] = {}
        observations_by_unit: dict[str, set[tuple[str, str]]] = {}
        for unit_key, relation, reading in zip(
                group["unit_keys"], group["relations"], group["readings"]):
            by_unit.setdefault(unit_key, set()).add(relation)
            observations_by_unit.setdefault(unit_key, set()).add(
                (reading, relation))
        votes: dict[str, int] = {}
        split: list[str] = []
        for unit_key, relations in by_unit.items():
            if len(relations) == 1:
                name = next(iter(relations))
                votes[name] = votes.get(name, 0) + 1
            else:
                split.append(unit_key)
        group["unit_count"] = len(by_unit)
        group["unit_votes"] = dict(sorted(votes.items()))
        group["units_without_a_consistent_relation"] = sorted(split)
        group["positive_units"] = votes.get("POSITIVE", 0)
        group["adverse_units"] = (votes.get("CONFLICT", 0)
                                  + votes.get("NEGATIVE", 0))
        group["vote_rule"] = (
            "one unit, one vote, and only when every reading in that unit "
            "carries the same relation; a unit that disagrees with itself is "
            "counted in neither total and named in "
            "units_without_a_consistent_relation")
        group["aliases"] = sorted(group.get("aliases") or ())
        # The deduplication, stated rather than implied.  ``bank_rows_seen``
        # is what the bank handed over; the counts above are observations.
        group.pop("seen_observations", None)
        group.pop("row_tokens", None)
        group.pop("readings", None)
        group["units_with_more_than_one_distinct_observation"] = sorted(
            unit_key for unit_key, seen in observations_by_unit.items()
            if len(seen) > 1)
        group["multi_observation_units_are_not_aggregated_here"] = (
            "a unit that carries two different readings of one program keeps "
            "both entries; which one -- or which combination -- is the unit's "
            "evidence is a protocol question, not a census one")
        group["behavior_fingerprints"] = dict(group.pop("fingerprints"))
    return ordered


def census_key(task_consumer_key: Any, program_steps: Any,
               root_scope: Any) -> str:
    """The identity a lineage is kept under: Task x Consumer x Program x root Scope.

    The same string the census builds, exposed so a Draft, an Active card and a
    census group can all be compared on one key instead of on a program name.
    """
    return "%s|%s@%s" % (str(task_consumer_key or ""),
                        _program_signature(program_steps),
                        _root_scope_signature(root_scope))


def _revisable_now(draft: drafts.RestrictedDraft,
                   evidence: Sequence[Mapping[str, Any]]) -> bool:
    """The condition the Draft loop already uses to propose a ``REVISE``.

    Stated once so the ``NARROW`` fold cannot drift away from it: a lineage the
    Draft loop is about to propose must not also be proposed as a ``NARROW``,
    and a lineage it will not propose must not be folded onto silently.
    """
    verified_since_revision = bool(
        draft.history and draft.history[-1].get("event") == "verification")
    return bool(draft.state == drafts.REVISABLE and evidence
                and draft.may_add_clause() and verified_since_revision)


def _fold_narrow_onto(draft: drafts.RestrictedDraft,
                      base: Mapping[str, Any], *,
                      rows_by_signature: Mapping[str, Sequence[Mapping[str, Any]]],
                      ) -> dict[str, Any]:
    """What a ``NARROW`` becomes when its lineage already carries a Draft.

    The Active card's adverse evidence and the Draft are the same lineage: the
    Draft is the record of that card's own refused redeployment.  Opening a
    second shell for it would restart the revision and verification counters,
    so the mint refuses -- and it used to refuse only *after* Slow and the
    replay screen had been paid for.  So the decision moves here:

    * the Draft is open, ``REVISABLE`` and otherwise ready -> the ``NARROW``
      becomes a ``REVISE`` of that Draft, starting from its ``current_scope``
      and keeping its ``root_scope``, so the clause budget is still counted
      against the initialiser and the ancestry is unbroken.  Nothing is
      incremented here; ``record_revision`` still does that when a clause is
      actually calibrated, and the later verification still costs an attempt.
    * the Draft is in a state that forbids narrowing -> refused now, with the
      reason named, and Slow is never called.

    ``FLAGGED`` is untouched by this: it forbids narrowing for its own reason
    and continues to.
    """
    signature = str(base["program_signature"])
    if draft.closed is not None:
        return {"refused": True, "outcome": "LINEAGE_CLOSED",
                "why": ("this lineage's Draft is closed (%s); a narrowing "
                        "would need a new shell, and a new shell restarts the "
                        "revision and verification counters"
                        % draft.closed)}
    if draft.state == drafts.FLAGGED:
        return {"refused": True, "outcome": "REVISION_TARGET_FLAGGED",
                "why": ("the lineage's Draft is FLAGGED: the damage was "
                        "dominated by series the Skill had already treated, "
                        "so narrowing would repair the wrong surface")}
    if draft.state == drafts.WAITING:
        return {"refused": True, "outcome": "REVISION_TARGET_WAITING",
                "why": ("the lineage's Draft is WAITING: only the coverage "
                        "floor failed, and narrowing would reduce the "
                        "coverage further")}
    if not draft.may_add_clause():
        return {"refused": True, "outcome": "REVISION_BUDGET_EXHAUSTED",
                "why": ("the lineage's Draft has spent its revision budget "
                        "(%d of %d)" % (draft.revisions, drafts.MAX_REVISIONS))}
    if not (draft.history
            and draft.history[-1].get("event") == "verification"):
        return {"refused": True,
                "outcome": "AWAITING_VERIFICATION_OF_LAST_REVISION",
                "why": ("a second clause needs a second reading; this Draft "
                        "has not been verified since its last revision")}
    if _revisable_now(draft, rows_by_signature.get(signature) or []):
        # The Draft loop is going to propose exactly this lineage, from this
        # Draft's ``current_scope`` and against its ``root_scope``.  That *is*
        # the merged candidate: the narrowing's adverse evidence is carried
        # onto it, and no second candidate is opened beside it.
        return {"already_proposed": True}
    # Open, REVISABLE, in budget, verified -- and the Draft loop still will not
    # propose it, which can only mean the bank holds no row under this
    # program's signature.  A narrowing with no evidence to calibrate on is
    # refused rather than sent to Slow to invent one.
    return {"refused": True, "outcome": "NO_EVIDENCE_FOR_THE_LINEAGE",
            "why": ("the lineage is revisable but the bank holds no row under "
                    "%s to calibrate a clause on" % signature)}


def propose_candidates(groups: Sequence[Mapping[str, Any]], *,
                       ledger: drafts.DraftLedger,
                       held_lineage_keys: Sequence[str] = (),
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The three candidate classes, plus the drift-signal table.

    Returns ``(candidates, drift_signals)``.  ``FLAGGED`` Drafts never become
    revision candidates: their evidence points at the Observation or the
    Program, so they are listed as drift signals for the census to carry and
    for the report to count, and the outer loop does not try to narrow them.

    Deduplication is on the **full census key** and over the key's whole
    lineage in this course -- Active, open and closed alike (sol v1.1 A).  Two
    consequences, and both are the point:

    * a key that has already been closed cannot come back as a fresh ADD.  A new
      shell would carry its own ``revisions=0`` and ``verification_attempts=0``,
      which is precisely how a Draft walks past the two-revision and
      three-verification bounds one legal-looking step at a time.
    * the same program under a *different* root Scope is a different candidate,
      because it treats a different set of series.

    What is held is therefore read from the lineage itself, not inferred from
    which bank rows happen to carry a ``source_skill_id``: a card can hold a key
    without any row in this window naming it.

    Two sets, not one.  ``ADD`` is deduplicated against **every** lineage this
    course has opened -- Active cards and Drafts, open and closed -- for the
    reason spelled out above.  ``NARROW`` tests only the Active set the runner
    passed, because its own evidence sentence is "an Active Skill came back
    CONFLICT or NEGATIVE" and a Draft holds no deployment rights to narrow.
    Until ``restrict`` began recording a census key, ``ledger.lineage_keys()``
    was always empty and the two sets were indistinguishable; separating them
    keeps ``NARROW``'s target set exactly what it has always been rather than
    letting the repair hand it a new class of target.
    """
    active = {str(key) for key in held_lineage_keys}
    known = active | ledger.lineage_keys()
    candidates: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    #: Adverse evidence a ``NARROW`` folded into a Draft the loop below is
    #: already going to propose.  One lineage produces one candidate per step,
    #: so the evidence is carried onto that candidate rather than opening a
    #: second one beside it.
    folded_narrow_evidence: dict[str, dict[str, Any]] = {}

    rows_by_signature: dict[str, list[dict[str, Any]]] = {}
    for group in groups:
        rows_by_signature.setdefault(
            str(group["program_signature"]), []).extend(group["rows"])

    for group in groups:
        signature = str(group["program_signature"])
        key = str(group.get("census_key") or "")
        if (group["positive_units"] >= MIN_POSITIVE_UNITS_FOR_ADD
                and key not in known):
            candidates.append({
                "kind": "ADD",
                "task_consumer_key": group["task_consumer_key"],
                "program_signature": signature,
                "census_key": group.get("census_key"),
                "root_scope_signature": group.get("root_scope_signature"),
                "program_steps": list(group["program_steps"]),
                # The predicate the evidence was actually gathered under.  Not
                # re-derived from the initialiser: two groups can share a
                # program and differ only in their root Scope, and re-deriving
                # would silently give both the same one.
                "root_scope": (dict(group["root_scope"])
                               if group.get("root_scope") else None),
                "rows": list(group["rows"]),
                "evidence": {
                    "positive_units": group["positive_units"],
                    "unit_count": group["unit_count"],
                    "relation_counts": group["relation_counts"],
                    "units": list(group["units"]),
                },
                "needs_clause": False,
                "why": (
                    "the same behaviour was POSITIVE on %d already-processed "
                    "unit(s) and no card holds it; it becomes a restricted "
                    "Draft that a later independent unit must verify"
                    % group["positive_units"]),
            })
        if (group["adverse_units"] >= MIN_ADVERSE_UNITS_FOR_NARROWING
                and key in active):
            narrow_evidence = {
                "adverse_units": group["adverse_units"],
                "unit_votes": dict(group.get("unit_votes") or {}),
                "relation_counts": group["relation_counts"],
                "units": list(group["units"]),
            }
            base = {
                "kind": "NARROW",
                "task_consumer_key": group["task_consumer_key"],
                "program_signature": signature,
                "census_key": key,
                "root_scope_signature": group.get("root_scope_signature"),
                "root_scope": (dict(group["root_scope"])
                               if group.get("root_scope") else None),
                "program_steps": list(group["program_steps"]),
                "rows": list(group["rows"]),
                "skill_ids": list(group["source_skill_ids"]),
                "evidence": narrow_evidence,
                "why": (
                    "an Active Skill came back CONFLICT or NEGATIVE on %d "
                    "already-processed units" % group["adverse_units"]),
            }
            # The same identity test the mint uses.  Keying the lookup on the
            # census key alone missed a Draft that carries no key, or carries a
            # different one for the same (program, root Scope) -- and those are
            # exactly the Drafts ``restrict`` used to mint before it recorded a
            # key.  Missing one here put the narrowing back on the old road:
            # a clause calibrated, a replay screen paid for, and only then a
            # refusal at ``open_restricted``, which tests this pair.
            existing = (ledger.by_census_key(key)
                        or ledger.by_program_and_root(
                            group["program_steps"],
                            group.get("root_scope") or {}))
            if existing is None:
                # Nothing of this lineage on record: the original path, which
                # ends in a Draft of its own.
                candidates.append({**base, "needs_clause": True})
            else:
                fold = _fold_narrow_onto(
                    existing, base, rows_by_signature=rows_by_signature)
                if fold.get("refused"):
                    # Refused here, before Slow and before the replay screen.
                    # Paying for a clause and a screen only to be turned away
                    # at the mint is what the previous shape did.
                    candidates.append({
                        **base, "needs_clause": False,
                        "outcome_preset": fold["outcome"],
                        "why_refused": fold["why"],
                        "draft_id": existing.draft_id,
                        "draft_state": existing.state,
                    })
                else:
                    # The Draft loop below proposes this lineage from the
                    # Draft's own current predicate.  One candidate per lineage
                    # per step: the adverse evidence rides along on that one
                    # rather than opening a second beside it.
                    folded_narrow_evidence[existing.draft_id] = narrow_evidence

    for draft in ledger.open_drafts():
        signature = _program_signature(draft.program_steps)
        evidence = rows_by_signature.get(signature) or []
        if draft.state == drafts.FLAGGED:
            signals.append({
                "signal": "EFFECT_NONSTATIONARY_CANDIDATE",
                "draft_id": draft.draft_id,
                "program_signature": signature,
                "state": draft.state,
                "verification_attempts": draft.verification_attempts,
                "why_not_a_revision_candidate": (
                    "the damage was dominated by series the Skill had already "
                    "treated; narrowing would repair the wrong surface"
                ),
                "for": "Observation / Program drift, reported not repaired",
            })
            continue
        if draft.state == drafts.WAITING:
            signals.append({
                "signal": "AWAITING_PATTERN_REENCOUNTER",
                "draft_id": draft.draft_id,
                "program_signature": signature,
                "state": draft.state,
                "verification_attempts": draft.verification_attempts,
                "why_not_a_revision_candidate": (
                    "only the coverage floor failed; the predicate is waiting "
                    "for a window that resolves enough series, and narrowing "
                    "would reduce the coverage further"
                ),
            })
            continue
        # A second clause needs a second reading.  Without this, a Draft revised
        # at step k is proposed again at step k+1 on the very rows its first
        # clause was calibrated on, and both revisions are spent before any new
        # unit has verified either.  (``_revisable_now`` states the condition
        # once, so the NARROW fold above cannot drift away from it.)
        if _revisable_now(draft, evidence):
            folded = folded_narrow_evidence.get(draft.draft_id)
            candidates.append({
                "kind": "REVISE",
                "task_consumer_key": "",
                "program_signature": signature,
                "program_steps": [{"op": op, "params": dict(params)}
                                  for op, params in draft.program_steps],
                "rows": list(evidence),
                "draft_id": draft.draft_id,
                "base_scope": dict(draft.current_scope),
                "root_scope": dict(draft.root_scope),
                "evidence": {"bank_rows": len(evidence),
                             "revisions_so_far": draft.revisions,
                             **({"folded_from_narrow": folded}
                                if folded else {})},
                "needs_clause": True,
                "why": (
                    "a REVISABLE Draft has %d new bank rows to calibrate a "
                    "further clause on" % len(evidence)),
            })
    return candidates, signals


# ---------------------------------------------------------------------------
# the replay screen
# ---------------------------------------------------------------------------

NOT_APPLICABLE = "NOT_APPLICABLE"


def _applicable(cell: Mapping[str, Any], *, min_treated: int) -> bool:
    """Whether a replayed cell is a reading at all.

    A cell where the candidate's predicate resolves to fewer series than the
    coverage floor is Static by construction (``_policy_reading`` returns exact
    zeros), and a cell the evaluator could not read carries ``None``.  Neither
    is evidence for or against the candidate: the authoritative gate itself
    treats the coverage floor as "no reading can be taken", and the ``WAITING``
    state exists because a window without the pattern is not a Skill failure.
    Counting such cells as ``aggregate_not_material`` would eliminate exactly
    the narrowed candidates the outer loop exists to produce, on every processed
    unit where their pattern is rare -- H3 measured as a rejection.
    """
    if cell.get("aggregate_gain") is None:
        return False
    treated = cell.get("treated")
    if treated is None:
        return True  # a reading that did not report coverage is judged as read
    return int(treated) >= int(min_treated)


def _violations(cell: Mapping[str, Any], *, material: float,
                max_harmed: float, max_harm: float) -> list[str]:
    failed: list[str] = []
    aggregate = cell.get("aggregate_gain")
    if aggregate is None or float(aggregate) <= material:
        failed.append("aggregate_not_material")
    harmed = cell.get("harmed_fraction")
    if harmed is not None and float(harmed) > max_harmed:
        failed.append("harmed_fraction")
    worst = cell.get("max_single_series_harm")
    if worst is not None and float(worst) > max_harm:
        failed.append("single_series_harm")
    return failed


def screen(candidate: Mapping[str, Any], scope: Mapping[str, Any] | None, *,
           replay: Callable[..., Mapping[str, Any]],
           policy: tool_module.Policy = tool_module.BOUNDED_RISK_V1,
           ) -> dict[str, Any]:
    """Re-resolve and re-score one candidate on the cells already processed.

    ``replay(steps=..., scope=...)`` is injected: in the runner it re-runs
    ``scoped_evaluate`` on this arm's own processed cells and returns
    ``{"cells": [...], "fits": int}``.  Any single cell that breaches a risk
    line, or whose aggregate is not material, eliminates the candidate -- there
    is no averaging across cells, because a policy that is safe on average and
    unsafe on one window is the failure mode the tail budget exists for.
    """
    outcome = replay(steps=_steps_tuple(candidate.get("program_steps")),
                     scope=dict(scope) if scope else None)
    cells = list(outcome.get("cells") or ())
    fits = int(outcome.get("fits") or 0)
    rejected: list[dict[str, Any]] = []
    not_applicable: list[dict[str, Any]] = []
    for cell in cells:
        if not _applicable(cell, min_treated=policy.min_treated):
            not_applicable.append({"unit": cell.get("unit"),
                                   "treated": cell.get("treated"),
                                   "why": cell.get("unusable") or
                                   "predicate resolves below the coverage floor"})
            continue
        failed = _violations(
            cell, material=policy.material,
            max_harmed=policy.max_harmed_fraction,
            max_harm=policy.max_single_series_harm)
        if failed:
            rejected.append({"unit": cell.get("unit"), "violated": failed,
                             "reading": dict(cell)})
    applicable = len(cells) - len(not_applicable)
    return {
        "cells_replayed": len(cells),
        "cells_applicable": applicable,
        "cells_not_applicable": not_applicable,
        "replay_fits": fits,
        "passed": not rejected and applicable > 0,
        "violations": rejected,
        "reason": (None if (not rejected and applicable > 0) else
                   REPLAY_SCREEN_REJECTED if rejected
                   else NOT_APPLICABLE if cells
                   else "no already-processed cell reaches this candidate"),
        "does_not_grant": "deployment rights; a Draft only comes out of here",
    }


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

def _clause_for(candidate: Mapping[str, Any], *, slow: Callable[..., Any],
                tool: Any, budget: OuterBudget, record: OuterStepRecord,
                policy: tool_module.Policy) -> dict[str, Any] | None:
    """Ask Slow for a direction, let the tool calibrate it, keep the shadow."""
    existing = list(dict(candidate.get("base_scope") or {}).get("predicate")
                    or ())
    attempts = 0
    rejected_directions: list[dict[str, Any]] = []
    while attempts <= budget.retries_per_candidate:
        if record.slow_calls >= budget.outer_llm_per_step:
            return {"outcome": "OUTER_LLM_BUDGET_SPENT"}
        payload = slow(candidate=candidate, rejected=rejected_directions)
        record.slow_calls += 1
        attempts += 1
        if isinstance(payload, Mapping) and payload.get("outcome") == (
                "OUTER_LLM_BUDGET_SPENT"):
            return {"outcome": "OUTER_LLM_BUDGET_SPENT"}
        if payload is None:
            return {"outcome": "SLOW_ABSTAINED"}
        result = tool.clause_from_slow(
            payload, rows=candidate.get("rows") or (), policy=policy,
            existing_clauses=existing)
        record.shadow_records.append({
            "candidate": candidate.get("kind"),
            "program_signature": candidate.get("program_signature"),
            "slow": {"feature": result.get("feature"),
                     "direction": result.get("direction")},
            "shadow": result.get("shadow"),
            "agree": result.get("slow_and_shadow_agree"),
        })
        if result.get("outcome") == "CALIBRATED":
            return result
        rejected_directions.append({
            "feature": result.get("feature"),
            "direction": result.get("direction"),
            "outcome": result.get("outcome"),
        })
    return {"outcome": "NO_FEASIBLE_THRESHOLD_AFTER_RETRIES",
            "tried": rejected_directions}


def consolidate(*, bank: Sequence[Mapping[str, Any]],
                ledger: drafts.DraftLedger,
                k_index: int,
                slow: Callable[..., Any] | None = None,
                tool: Any = tool_module,
                replay: Callable[..., Mapping[str, Any]] | None = None,
                budget: OuterBudget | None = None,
                policy: tool_module.Policy = tool_module.BOUNDED_RISK_V1,
                held_lineage_keys: Sequence[str] = (),
                bank_boundary: Mapping[str, Any] | None = None,
                rng: Any = None,
                ) -> OuterStepRecord:
    """One outer step.  Costs nothing when the bank has nothing to say.

    ``rng`` is accepted and unused on purpose: every decision here is a total
    order over values already in the bank, so two runs on the same bank produce
    the same step.  A signature that took a seed would invite a future version
    to break that quietly.
    """
    import time  # noqa: PLC0415 - only the step duration needs it

    started = time.time()
    budget = budget or OuterBudget()
    record = OuterStepRecord(
        k_index=int(k_index),
        units_in_bank=len({
            json.dumps(row.get("unit"), sort_keys=True, default=str)
            for row in (bank or ())}),
        bank_boundary=dict(bank_boundary or {
            "contains": "cells this arm already processed, in this ordering",
            "excludes": ["the evaluation face (+144)", "future units",
                         "other arms", "held-out"],
            "declared_by": "the runner; this module records what it was handed",
        }),
    )
    if not bank:
        record.empty_reason = "the bank is empty"
        record.wall_seconds = time.time() - started
        return record

    groups = census(bank, material=policy.material)
    record.groups = [
        {key: group[key] for key in (
            "task_consumer_key", "program_signature", "root_scope_signature",
            "census_key", "aliases", "unit_count", "observations",
            "relation_counts", "unit_votes",
            "units_without_a_consistent_relation",
            "positive_units", "adverse_units", "bank_rows_seen",
            "duplicate_observations_dropped",
            "units_with_more_than_one_distinct_observation")}
        for group in groups]
    candidates, signals = propose_candidates(
        groups, ledger=ledger, held_lineage_keys=held_lineage_keys)
    record.drift_signals = signals
    if not candidates:
        record.empty_reason = "the census produced no candidate"
        record.wall_seconds = time.time() - started
        return record

    for candidate in candidates:
        entry: dict[str, Any] = {
            "kind": candidate["kind"],
            "program_signature": candidate["program_signature"],
            "why": candidate["why"],
            "evidence": candidate.get("evidence"),
        }
        if candidate["kind"] == "REVOKE":
            record.revocation_recommendations.append(entry)
            record.candidates.append({**entry, "outcome": "RECOMMENDED"})
            continue

        if candidate.get("outcome_preset"):
            # Decided in ``propose_candidates`` -- a narrowing whose lineage
            # already carries a Draft that cannot take a clause.  Recorded
            # here without calling Slow and without a replay screen: the old
            # shape paid for both and was then turned away at the mint.
            record.candidates.append({
                **entry,
                "outcome": candidate["outcome_preset"],
                "why_refused": candidate.get("why_refused"),
                "draft_id": candidate.get("draft_id"),
                "draft_state": candidate.get("draft_state"),
            })
            continue

        scope: Mapping[str, Any] | None
        if candidate["needs_clause"]:
            if slow is None:
                entry["outcome"] = "NO_SLOW_AGENT_AVAILABLE"
                record.candidates.append(entry)
                continue
            calibration = _clause_for(
                candidate, slow=slow, tool=tool, budget=budget,
                record=record, policy=policy)
            entry["calibration"] = calibration
            if not calibration or calibration.get("outcome") != "CALIBRATED":
                entry["outcome"] = (calibration or {}).get(
                    "outcome", "NO_CLAUSE")
                record.candidates.append(entry)
                continue
            base = dict(candidate.get("base_scope") or {})
            predicate = [dict(item) for item in (base.get("predicate") or ())]
            predicate.append(dict(calibration["clause"]))
            scope = {"scope_type": "serving_series_predicate",
                     "predicate": predicate}
            verdict = narrowing.validate_narrowing(
                base, scope, root=candidate.get("root_scope") or None)
            entry["narrowing_preflight"] = verdict.to_dict()
            if not verdict.accepted:
                entry["outcome"] = "NARROWING_PREFLIGHT_REFUSED"
                record.candidates.append(entry)
                continue
        else:
            # The predicate the evidence was gathered under, when the bank
            # recorded one.  Falling back to the initialiser is only correct
            # when the probe carried no Scope at all; re-deriving it whenever
            # the bank *does* carry one would give two groups that differ only
            # in their root Scope the same predicate, undoing the census key.
            scope = (candidate.get("root_scope")
                     or initializer.initialize(
                         candidate["program_steps"])["scope"])
        entry["scope"] = dict(scope) if scope else None
        entry["root_scope_signature"] = candidate.get("root_scope_signature")

        if replay is None:
            entry["outcome"] = "NO_REPLAY_SCREEN_AVAILABLE"
            record.candidates.append(entry)
            continue
        # Checked against what this screen *will* cost, not against whether
        # anything is left: one screen re-scores every processed cell, so a cap
        # tested only for "> 0" can be overshot by a whole screen and is not a
        # cap.  The estimate is published by the injected replay callable.
        estimate = int(getattr(replay, "estimated_fits_per_candidate", 0) or 0)
        if (budget.replay_fits_remaining is not None
                and estimate > budget.replay_fits_remaining):
            entry["outcome"] = "REPLAY_FITS_BUDGET_SPENT"
            entry["replay_estimate"] = estimate
            entry["replay_fits_remaining"] = budget.replay_fits_remaining
            record.candidates.append(entry)
            continue
        verdict = screen(candidate, scope, replay=replay, policy=policy)
        entry["replay"] = verdict
        record.replay_fits += int(verdict["replay_fits"])
        if budget.replay_fits_remaining is not None:
            budget.replay_fits_remaining -= int(verdict["replay_fits"])
        if not verdict["passed"]:
            entry["outcome"] = verdict["reason"]
            record.rejected.append({
                "kind": candidate["kind"],
                "program_signature": candidate["program_signature"],
                "reason": verdict["reason"],
                "violations": verdict["violations"],
            })
            record.candidates.append(entry)
            continue

        if candidate["kind"] == "REVISE":
            # The Draft that failed is the Draft that gets the clause.  Opening
            # a fresh shell here would leave the failed predicate open and
            # resupplied beside the revised one, and would start the revision
            # and verification counters again from zero -- the bound "two
            # revisions, three verifications" would then hold per shell and
            # not per Skill candidate, which is the only place it means
            # anything.
            draft = ledger.by_id(str(candidate["draft_id"]))
            if draft is None or not draft.may_add_clause():
                entry["outcome"] = "REVISION_TARGET_NOT_REVISABLE"
                record.candidates.append(entry)
                continue
            ledger.record_revision(
                draft, origin=int(k_index), new_scope=dict(scope or {}),
                preflight=entry.get("narrowing_preflight"), support=None)
            draft.history.append({
                "event": "revised_by_outer_loop",
                "outer_step": int(k_index),
                "revisions": draft.revisions,
                "replay_cells": verdict["cells_replayed"],
                "state_after": draft.state,
            })
            entry["outcome"] = "DRAFT_REVISED"
            entry["draft_id"] = draft.draft_id
            entry["revisions"] = draft.revisions
            record.drafts_revised.append(draft.draft_id)
            record.candidates.append(entry)
            continue

        try:
            draft = ledger.open_restricted(
                program_steps=_steps_tuple(candidate["program_steps"]),
                root_scope=dict(candidate.get("root_scope") or scope or {}),
                current_scope=dict(scope or {}),
                origin=int(k_index),
                census_key=candidate.get("census_key"),
                provenance={
                    "outer_step": int(k_index),
                    "kind": candidate["kind"],
                    "evidence": candidate.get("evidence"),
                    "replay_cells": verdict["cells_replayed"],
                })
        except ValueError as exc:
            # The ledger refuses a second shell for a key it already carries.
            # A ``NARROW`` on an Active card whose failed redeployment is
            # already recorded as a Draft lands here: opening one anyway is
            # what resets the revision and verification counters, and merging
            # the clause into the existing Draft would change what NARROW
            # produces.  Recorded as a refusal for the report to count; the
            # decision belongs to whoever rules on that question.
            entry["outcome"] = "LINEAGE_ALREADY_HAS_A_DRAFT"
            entry["why_refused"] = str(exc)[:240]
            record.candidates.append(entry)
            continue
        entry["outcome"] = "RESTRICTED_DRAFT_OPENED"
        entry["draft_id"] = draft.draft_id
        record.drafts_opened.append(draft.draft_id)
        record.candidates.append(entry)

    record.wall_seconds = time.time() - started
    return record


def declared_rules() -> dict[str, Any]:
    """The outer loop's boundary, for the contract to embed verbatim."""
    return {
        "module": "evaluation.main_protocol_p4.outer_loop",
        "period_k_units": "set by the contract",
        "reads": "this arm's own already-processed cells only",
        "does_not_read": [
            "the evaluation face (+144)", "future units", "other arms",
            "held-out", "any oracle",
        ],
        "census": (
            "Task x Consumer x full typed Program x root Scope (sol v1.1); "
            "programs whose per-series gain vectors agree on every shared unit "
            "are collapsed into one alias class.  Fault type is stratified "
            "evidence and does not enter the key"
        ),
        "candidate_classes": list(CANDIDATE_KINDS),
        "min_positive_units_for_add": MIN_POSITIVE_UNITS_FOR_ADD,
        "min_adverse_units_for_narrowing": MIN_ADVERSE_UNITS_FOR_NARROWING,
        "slow_gets": "one candidate at a time, semantics only (W2)",
        "replay_may": ["eliminate a candidate"],
        "replay_may_not": ["promote", "activate", "deploy"],
        "replay_not_applicable": (
            "a processed cell where the predicate resolves below the coverage "
            "floor, or that the evaluator could not read, is neither a pass nor "
            "a failure; at least one applicable cell is required"
        ),
        "revise": (
            "a REVISE candidate adds its clause to the existing Draft in place; "
            "revisions and verification attempts are counted per Draft, never "
            "per shell, and a further clause needs a verification after the last"
        ),
        "output": "restricted Drafts, plus revocation recommendations",
        "revocation": "recommended here, executed by the runner's lifecycle",
        "flagged_drafts": "never revision candidates; listed as drift signals",
        "deterministic": (
            "every ordering is over values already in the bank; the rng "
            "parameter is accepted and unused"
        ),
    }
