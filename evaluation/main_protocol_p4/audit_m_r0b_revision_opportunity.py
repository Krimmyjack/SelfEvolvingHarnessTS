"""M-R0b: how many revision opportunities the historical course actually held.

Read-only.  0 LLM, 0 Consumer fits, 0 held-out reads, 0 sealed reads; no
production code is changed and no existing artifact is written.

Run:  python -m evaluation.main_protocol_p4.audit_m_r0b_revision_opportunity

What an "opportunity" is here, and what it is not
-------------------------------------------------
For every one of the thirty recorded outer steps, and for every lineage the
bank held at that step, four questions are answered **from the state as it
stood at that step** -- never from the end-of-course state, because a Draft
that ended ``FLAGGED`` may have been ``REVISABLE`` when the step ran:

  A. was there an Active ancestor -- did this arm hold a card for the lineage?
  B. could the lifecycle receive a revision at that moment?
  C. was a Scope modification feasible at all, as a shadow over the frozen
     vocabulary and bin edges on the rows the bank already held?
  D. were there later evaluable units to verify it on, and did the pattern
     actually recur on them?

The reported intersection is ``B and C and D``, cross-tabulated by A.

Three things this is not:

* It is not a claim about what the repaired wiring would have produced.  No
  Slow call, no replay screen and no later verification is simulated; each of
  those is reported as ``NOT_EVALUATED`` wherever the course did not actually
  run it.  ``C`` says a clause exists in the search space, not that Slow would
  have named it, nor that it would have survived a gate.
* It is not the as-run candidate count.  As run, the census produced no ADD and
  no NARROW at all, for the reason M-R0 established.  The evidence conditions
  below are read with the relation the recorded per-series gains carry, because
  the question is what the *course* held, not what the wiring could see.
* It is not the 26/6 counterfactual.  That counted candidates a repaired census
  would emit; this counts opportunities, which is a strictly narrower object --
  it also requires a receiving lifecycle, a feasible clause and later units.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

ART = base.ART
PERIOD = base.PERIOD_K_UNITS
MATERIAL = base.MATERIAL
MIN_TREATED = base.MIN_TREATED
NOT_EVALUATED = "NOT_EVALUATED"
UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------
# the lifecycle, replayed to each step boundary
# --------------------------------------------------------------------------

def _draft_timeline(doc: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """Every Draft of this arm, with each recorded event placed at a position.

    The events carry a window and the lines that failed; the cell that produced
    them is the one whose delayed reading matches both.  M-R0 established that
    this match is unique for all thirty-three recorded events; where it is not,
    the event is placed at ``None`` and every state derived from it is UNKNOWN.
    """
    by_window: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in doc["cells"]:
        delayed = cell.get("delayed")
        if cell["arm"] == arm and delayed and cell.get("deployed"):
            by_window[int(delayed["origin"])].append(cell)

    timelines = []
    for draft in (doc["lifecycle"].get(arm) or {}).get("drafts", ()):
        events = []
        for entry in draft["history"]:
            pool = by_window.get(int(entry["window"]), [])
            matches = [cell for cell in pool
                       if (cell.get("delayed") or {}).get("gate", {}).get(
                           "failed_lines") == entry["failed_lines"]]
            cell = matches[0] if len(matches) == 1 else None
            events.append({
                "position": (int(cell["position"]) if cell else None),
                "state_after": entry["state_after"],
                "verification_attempts": entry["verification_attempts"],
                "cell_program": (base.prog(cell["deployed"]) if cell else None),
                "match": "unique" if cell else UNKNOWN,
            })
        timelines.append({
            "draft_id": draft["draft_id"],
            "program": base.prog(draft["program_steps"]),
            "root_scope": draft["root_scope"],
            "revisions": draft["revisions"],
            "max_verification_attempts": draft["max_verification_attempts"],
            "closed_at_end_of_course": draft["closed"],
            "events": events,
        })
    return timelines


def _draft_state_at(timeline: dict[str, Any], position: int) -> dict[str, Any]:
    """The Draft as it stood after every event at or before ``position``.

    ``None`` before its first event: the Draft did not exist yet, which is a
    different answer from "it existed and could not take a clause".
    """
    seen = [event for event in timeline["events"]
            if event["position"] is not None and event["position"] <= position]
    if not seen:
        unplaced = any(event["position"] is None
                       for event in timeline["events"])
        return {"exists": False,
                "why": (UNKNOWN if unplaced else
                        "no event of this Draft had happened yet")}
    last = seen[-1]
    attempts = int(last["verification_attempts"])
    state = last["state_after"]
    # ``record_verification`` closes a Draft the moment its attempts run out.
    closed = (drafts.CLOSE_REASONS.get(state or "", "REVISION_BUDGET_EXHAUSTED")
              if attempts >= int(timeline["max_verification_attempts"])
              else None)
    return {
        "exists": True,
        "draft_id": timeline["draft_id"],
        "state": state,
        "verification_attempts": attempts,
        "closed": closed,
        "revisions_at_end_of_course": timeline["revisions"],
        "created_at_position": seen[0]["position"],
    }


def _receivable(state: dict[str, Any]) -> dict[str, Any]:
    """Could the lifecycle take a Scope revision here, under the as-run rules?

    The conditions are the ones the outer loop applied at the time: a fresh
    lineage may be narrowed into a new Draft; an existing one may be revised
    only while it is open, ``REVISABLE`` and inside its revision budget.
    """
    if not state.get("exists"):
        if state.get("why") == UNKNOWN:
            return {"receivable": UNKNOWN,
                    "route": UNKNOWN,
                    "why": "an event of this lineage could not be placed"}
        return {"receivable": True, "route": "NARROW_OPENS_A_NEW_DRAFT",
                "why": "no Draft of this lineage existed at this step"}
    if state.get("closed"):
        return {"receivable": False, "route": None,
                "why": "the Draft was closed as %s" % state["closed"]}
    if state["state"] == drafts.FLAGGED:
        return {"receivable": False, "route": None,
                "why": ("FLAGGED: the damage was dominated by series already "
                        "treated, so narrowing repairs the wrong surface")}
    if state["state"] == drafts.WAITING:
        return {"receivable": False, "route": None,
                "why": ("WAITING: only the coverage floor failed, and "
                        "narrowing would reduce coverage further")}
    if state["state"] == drafts.REVISABLE:
        return {"receivable": True, "route": "REVISE_THE_EXISTING_DRAFT",
                "why": "REVISABLE and inside its revision budget"}
    return {"receivable": UNKNOWN, "route": UNKNOWN,
            "why": "unrecognised Draft state %r" % (state["state"],)}


# --------------------------------------------------------------------------
# the shadow: is any Scope modification feasible at all?
# --------------------------------------------------------------------------

def _rows_as_run(bank: Sequence[Mapping[str, Any]], group: Mapping[str, Any]
                 ) -> list[dict[str, Any]]:
    """The evidence rows the *live* run would have handed the shadow.

    Deliberately **not** ``group["rows"]``.  The census now drops an exact
    repeat of an observation, which is the right count -- but this is a
    reconstruction of what the course held, and the course's own shadow
    searched every row the bank carried.  Using the deduplicated set here would
    quietly answer a different question, and the recorded shadows would no
    longer reproduce.
    """
    rows: list[dict[str, Any]] = []
    for row in bank:
        key = outer_loop.census_key(row.get("task_consumer_key"),
                                    row.get("program_steps"),
                                    row.get("serving_scope"))
        if key != group["census_key"]:
            continue
        features = dict(row.get("features") or {})
        for uid, gain in dict(row.get("per_series_gain") or {}).items():
            rows.append({"unit": row.get("unit"), "series": str(uid),
                         "features": dict(features.get(str(uid)) or {}),
                         "gain": float(gain)})
    return rows


def _shadow(rows: list[dict[str, Any]],
            root_scope: Any) -> dict[str, Any]:
    """``best_stump`` over the rows the bank already held.  0 fits, 0 LLM.

    This is the same deterministic search the live run recorded beside each
    Slow proposal.  A ``BEST_STUMP`` here means a clause exists in the frozen
    vocabulary and bin edges that clears the four lines **on the already
    processed cells**.  It does not mean Slow would have named it, and it says
    nothing about a later unit -- both of those are NOT_EVALUATED.
    """
    if not rows:
        return {"feasible": False, "outcome": "NO_ROWS",
                "why": "the bank held no per-series row for this lineage"}
    existing = list((root_scope or {}).get("predicate") or ())
    try:
        found = tool.best_stump(rows=rows, existing_clauses=existing)
    except Exception as exc:  # noqa: BLE001 - reported, never swallowed
        return {"feasible": UNKNOWN, "outcome": UNKNOWN,
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
    feasible = found.get("outcome") == "BEST_STUMP"
    return {
        "feasible": feasible,
        "outcome": found.get("outcome"),
        "clause": found.get("clause"),
        "feasible_count": found.get("feasible_count"),
        "considered": found.get("considered"),
        "slow_would_have_named_it": NOT_EVALUATED,
        "would_pass_a_later_gate": NOT_EVALUATED,
    }


# --------------------------------------------------------------------------
# later units: verification and re-encounter
# --------------------------------------------------------------------------

def _later_units(doc: dict[str, Any], arm: str, after: int,
                 program: str) -> dict[str, Any]:
    """What the ordering still had to offer after this step.

    ``re_encountered`` counts later units on which this program was actually
    deployed and the predicate resolved at or above the coverage floor -- a
    measured lower bound.  Units where it was not deployed carry no reading of
    that predicate, so they are counted as UNKNOWN rather than as absent.
    """
    later = [cell for cell in base.cells(doc, arm)
             if int(cell["position"]) > after]
    evaluable = [cell for cell in later if cell.get("evaluation")]
    deployed = [cell for cell in later
                if cell.get("deployed")
                and base.prog(cell["deployed"]) == program
                and cell.get("delayed")]
    recurred = [cell for cell in deployed
                if int(cell["delayed"]["treated"]) >= MIN_TREATED]
    return {
        "later_units": len(later),
        "later_evaluable_units": len(evaluable),
        "independent_verification_available": len(evaluable) >= 1,
        "later_units_where_this_program_was_deployed": len(deployed),
        "re_encountered_at_or_above_the_coverage_floor": len(recurred),
        "later_units_with_no_reading_of_this_predicate": (
            len(later) - len(deployed)),
        "re_encounter_beyond_those": UNKNOWN,
        "verification_outcome": NOT_EVALUATED,
    }


# --------------------------------------------------------------------------
# the census, replayed to each step boundary
# --------------------------------------------------------------------------

#: (block, span, origin) -> the unit's served uids and their deployment-visible
#: features, rebuilt through the runner's own ``UnitContext``.  The course
#: artifacts record per-series *gains* but not per-series *features*, and the
#: shadow search needs both -- so the features are recomputed on the production
#: path rather than guessed.  Building a ``UnitContext`` reads the exposed KDD
#: development cache and fits nothing: 0 Consumer fits, 0 LLM.
_UNIT_CACHE: dict[tuple, tuple[list[str], dict[str, dict[str, float]]]] = {}
_CTX_CACHE: dict[tuple, Any] = {}


def _unit_ctx(unit: dict[str, Any]):
    """The runner's own ``UnitContext`` for this unit, built once and kept.

    Building one reads the exposed KDD development cache and fits nothing, so
    it is 0 Consumer fits and 0 LLM.  It carries the served uids, the
    deployment-visible features and the production Scope resolver, which is
    what lets a predicate be resolved at an arbitrary origin without
    re-implementing any of it here.
    """
    key = (str(unit["block"]), tuple(unit["span"]), int(unit["origin"]))
    if key not in _CTX_CACHE:
        from evaluation.main_protocol_p4 import run_hec1 as runner
        _CTX_CACHE[key] = runner.UnitContext(unit)
    return _CTX_CACHE[key]


def _unit_features(unit: dict[str, Any]):
    key = (str(unit["block"]), tuple(unit["span"]), int(unit["origin"]))
    if key not in _UNIT_CACHE:
        ctx = _unit_ctx(unit)
        _UNIT_CACHE[key] = (list(ctx.eval_uids),
                            {uid: dict(ctx.features.get(uid) or {})
                             for uid in ctx.eval_uids})
    return _UNIT_CACHE[key]


def _bank_row(cell: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """One bank row, rebuilt the way ``_bank_rows_from_round`` built it.

    The probe's per-series gains are positional over the unit's served uids,
    which is the alignment the runner itself uses.
    """
    uids, features = _unit_features(cell["unit"])
    gains = list(row["gains"])
    if len(gains) != len(uids):
        return {**row, "task_consumer_key": base.TASK_CONSUMER_KEY,
                "per_series_gain": {}, "features": {},
                "reconstruction": UNKNOWN}
    return {**row,
            "task_consumer_key": base.TASK_CONSUMER_KEY,
            "per_series_gain": dict(zip(uids, gains)),
            "features": {uid: dict(features.get(uid) or {}) for uid in uids}}


def _walk(doc: dict[str, Any], ordering: str, arm: str,
          k0_keys: list[str]) -> list[dict[str, Any]]:
    timelines = _draft_timeline(doc, arm)
    held: set[str] = set(k0_keys) if arm.startswith("A5") else set()
    bank: list[dict[str, Any]] = []
    snapshots: dict[int, dict[str, Any]] = {}
    for cell in base.cells(doc, arm):
        position = int(cell["position"])
        if cell.get("lineage_key"):
            held.add(str(cell["lineage_key"]))
        for row in base.bank_rows_for_cell(cell):
            # Features are not needed for the relation and are not recorded
            # per series on the probe, so the shadow searches the rows the
            # census carries; where a feature is absent the stump simply
            # cannot use it.  Recorded as a limit, not papered over.
            bank.append(_bank_row(cell, row))
        if (position + 1) % PERIOD == 0:
            snapshots[(position + 1) // PERIOD] = {
                "boundary_position": position,
                "held": set(held),
                "bank": list(bank),
            }

    out = []
    for step in [row for row in doc["outer_steps"] if row["arm"] == arm]:
        k = int(step["k_index"])
        snap = snapshots.get(k)
        if snap is None:
            out.append({"ordering": ordering, "arm": arm, "k_index": k,
                        "lineages": [], "note": UNKNOWN})
            continue
        groups = outer_loop.census(snap["bank"], material=MATERIAL)
        lineages = []
        for group in groups:
            key = str(group["census_key"])
            program = str(group["program_signature"]).split("(")[0]
            timeline = next(
                (row for row in timelines
                 if row["program"] == program
                 and outer_loop._root_scope_signature(row["root_scope"])
                 == group["root_scope_signature"]), None)
            state = (_draft_state_at(timeline, snap["boundary_position"])
                     if timeline else {"exists": False,
                                       "why": "no Draft of this lineage"})
            receivable = _receivable(state)
            as_run_rows = _rows_as_run(snap["bank"], group)
            shadow = _shadow(as_run_rows, group.get("root_scope"))
            shadow["rows_searched"] = len(as_run_rows)
            shadow["rows_after_exact_deduplication"] = len(group["rows"])
            later = _later_units(doc, arm, snap["boundary_position"], program)
            adverse_trigger = (
                group["adverse_units"] >= outer_loop.MIN_ADVERSE_UNITS_FOR_NARROWING)
            opportunity = bool(receivable["receivable"] is True
                               and shadow["feasible"] is True
                               and later["independent_verification_available"])
            reached = [row for row in step["candidates"]
                       if str(row.get("program_signature", "")).split("(")[0]
                       == program]
            lineages.append({
                "as_run_mechanism_reached_this_lineage": bool(reached),
                "as_run_outcome_for_this_lineage": [
                    row.get("outcome") for row in reached] or None,
                "census_key": key,
                "program": program,
                "active_ancestor_at_this_step": key in snap["held"],
                "adverse_units_at_this_step": group["adverse_units"],
                "positive_units_at_this_step": group["positive_units"],
                "unit_votes": group["unit_votes"],
                "observations": group["observations"],
                "units_without_a_consistent_relation": len(
                    group["units_without_a_consistent_relation"]),
                "reached_the_narrowing_trigger": adverse_trigger,
                "lifecycle_state": state,
                "lifecycle_can_receive": receivable,
                "scope_shadow": shadow,
                "later": later,
                "opportunity": opportunity,
                "opportunity_with_an_active_ancestor": bool(
                    opportunity and key in snap["held"]),
                "opportunity_at_the_narrowing_trigger": bool(
                    opportunity and adverse_trigger),
            })
        out.append({"ordering": ordering, "arm": arm, "k_index": k,
                    "boundary_position": snap["boundary_position"],
                    "recorded_candidate_kinds": [
                        row["kind"] for row in step["candidates"]],
                    "recorded_empty_reason": step["empty_reason"],
                    "lineages": lineages})
    return out


# --------------------------------------------------------------------------

def _validate_shadow(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """The three shadows the live course recorded, recomputed here.

    This is what makes the other twenty-seven trustworthy.  The features the
    search needs are not in the artifacts and had to be recomputed on the
    production path; if that reconstruction were wrong, these three would not
    come back with the same outcome, the same clause and the same feasible
    count as the run recorded.
    """
    rows = []
    for name in base.ORDERINGS:
        doc = base.load(name)
        bank, snaps = [], {}
        for cell in base.cells(doc, "A5-online"):
            position = int(cell["position"])
            for row in base.bank_rows_for_cell(cell):
                bank.append(_bank_row(cell, row))
            if (position + 1) % PERIOD == 0:
                snaps[(position + 1) // PERIOD] = list(bank)
        for step in [row for row in doc["outer_steps"]
                     if row["arm"] == "A5-online" and row["shadow_records"]]:
            k = int(step["k_index"])
            groups = outer_loop.census(snaps[k], material=MATERIAL)
            group = next(row for row in groups
                         if row["program_signature"] == "outlier_mad({})")
            found = tool.best_stump(
                rows=_rows_as_run(snaps[k], group),
                existing_clauses=list(
                    (group.get("root_scope") or {}).get("predicate") or ()))
            want = step["shadow_records"][0]["shadow"]
            rows.append({
                "ordering": name, "k_index": k,
                "recorded": {key: want.get(key)
                             for key in ("outcome", "clause",
                                         "feasible_count", "considered")},
                "recomputed": {key: found.get(key)
                               for key in ("outcome", "clause",
                                           "feasible_count", "considered")},
                "matches": (found.get("outcome") == want.get("outcome")
                            and found.get("clause") == want.get("clause")
                            and found.get("feasible_count")
                            == want.get("feasible_count")),
            })
    return {
        "why": ("the shadow search needs per-series features, which the course "
                "artifacts do not record; they are recomputed through the "
                "runner's own UnitContext, and these three are the only places "
                "the run left an answer to check that against"),
        "checks": rows,
        "all_match": all(row["matches"] for row in rows),
    }


def build() -> dict[str, Any]:
    k0 = json.loads(base.K0_RECEIPT.read_text(encoding="utf-8"))
    k0_keys = list(k0.get("lineage_keys") or ())

    steps: list[dict[str, Any]] = []
    for name in base.ORDERINGS:
        doc = base.load(name)
        for arm in base.ONLINE_ARMS:
            steps.extend(_walk(doc, name, arm, k0_keys))

    rows = [(step, lineage) for step in steps for lineage in step["lineages"]]
    opportunities = [(step, lineage) for step, lineage in rows
                     if lineage["opportunity"]]
    reasons = Counter()
    for _step, lineage in rows:
        if lineage["opportunity"]:
            continue
        if lineage["lifecycle_can_receive"]["receivable"] is not True:
            reasons["lifecycle_cannot_receive: %s"
                    % lineage["lifecycle_can_receive"]["why"][:60]] += 1
        elif lineage["scope_shadow"]["feasible"] is not True:
            reasons["no_feasible_scope_modification: %s"
                    % lineage["scope_shadow"]["outcome"]] += 1
        else:
            reasons["no_later_evaluable_unit"] += 1

    def _dedupe(pairs, key):
        return sorted({key(step, lineage) for step, lineage in pairs})

    return {
        "stage": "M_R0B_REVISION_OPPORTUNITY_CENSUS",
        "written_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "evidence_grade": "INSTRUMENT -- an inventory of the recorded course",
        "boundary": {"llm_calls": 0, "consumer_fits": 0, "held_out_reads": 0,
                     "sealed_reads": 0, "production_code_changed": 0,
                     "existing_artifacts_written": 0,
                     "new_sha_or_hash_introduced": 0},
        "version_discipline": base.version_check(),
        "shadow_reconstruction_validated": _validate_shadow(steps),
        "what_is_counted": {
            "state_used": ("the Active set and Draft lifecycle as they stood "
                           "at each step boundary, replayed forward from the "
                           "recorded cells; end-of-course state is never used"),
            "opportunity": ("lifecycle_can_receive AND a feasible Scope "
                            "modification in the frozen search space AND at "
                            "least one later evaluable unit"),
            "active_ancestor": "reported and cross-tabulated, not required",
            "not_simulated": [
                "whether Slow would have named the feasible clause",
                "whether a revision would have passed the replay screen",
                "whether a later unit would have verified it",
                "what the repaired wiring would have produced",
            ],
            "relations": ("read from the recorded per-series gains, because "
                          "the question is what the course held; as run, the "
                          "census saw none of it (M-R0)"),
        },
        "totals": {
            "outer_steps": len(steps),
            "lineage_readings_across_steps": len(rows),
            "step_level_opportunities": len(opportunities),
            "step_level_opportunities_with_an_active_ancestor": sum(
                1 for _s, l in opportunities
                if l["active_ancestor_at_this_step"]),
            "step_level_opportunities_at_the_narrowing_trigger": sum(
                1 for _s, l in opportunities
                if l["reached_the_narrowing_trigger"]),
            "step_level_opportunities_the_as_run_mechanism_reached": sum(
                1 for _s, l in opportunities
                if l["as_run_mechanism_reached_this_lineage"]),
            "step_level_opportunities_with_a_measured_re_encounter": sum(
                1 for _s, l in opportunities
                if l["later"]["re_encountered_at_or_above_the_coverage_floor"]
                >= 1),
            "distinct_lineages_with_a_measured_re_encounter": _dedupe(
                [(s_, l) for s_, l in opportunities
                 if l["later"]["re_encountered_at_or_above_the_coverage_floor"]
                 >= 1],
                lambda s_, l: "%s|%s|%s" % (s_["ordering"], s_["arm"],
                                            l["census_key"])),
            "re_encounter_beyond_the_measured_ones": UNKNOWN,
            "distinct_lineages_with_an_opportunity": _dedupe(
                opportunities,
                lambda s, l: "%s|%s|%s" % (s["ordering"], s["arm"],
                                           l["census_key"])),
            "distinct_lineage_identities_with_an_opportunity": _dedupe(
                opportunities, lambda s, l: l["census_key"]),
            "distinct_ordering_arm_pairs_with_an_opportunity": _dedupe(
                opportunities, lambda s, l: "%s|%s" % (s["ordering"], s["arm"])),
            "why_the_others_were_not_opportunities": dict(
                sorted(reasons.items(), key=lambda row: -row[1])),
        },
        "steps": steps,
    }


def main() -> int:
    report = build()
    out = ART / "m_r0b_revision_opportunity.json"
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8")).get("stage")
        except (OSError, ValueError):
            existing = None
        if existing != report["stage"]:
            sys.stderr.write("refusing to overwrite %s\n" % out)
            return 2
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("wrote %s" % out)
    for key, value in report["totals"].items():
        if isinstance(value, list):
            print("  %-58s %d" % (key, len(value)))
        elif isinstance(value, dict):
            print("  %s:" % key)
            for name, count in value.items():
                print("      %-56s %s" % (name[:56], count))
        else:
            print("  %-58s %s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
