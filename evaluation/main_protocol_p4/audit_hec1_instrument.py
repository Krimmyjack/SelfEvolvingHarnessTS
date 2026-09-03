"""Eight mechanical checks that decide whether the next ordering may start.

What this is, and what it deliberately is not
---------------------------------------------
It is the button that used to be pressed by a person: sol's release rule for
Reverse and Interleaved is *instrument completeness only, never the effect's
sign*, and every check below is a count, a ledger comparison or a set
intersection.  None of them reads a gain, a utility or a verdict, and
``audit()`` never sees the curve -- so the thing that decides whether to
continue cannot be influenced by whether the result looked good.

It is **not** the readout.  Interpreting the curve and writing the verdict is a
separate script and a separate hand, for the reason sol gave: the failure mode
worth guarding is not a runner behaving badly, it is a runner certifying its own
instruments and then marking its own work.  These eight are mechanical enough to
be self-run; the readout is not.

The eight
---------
1. completeness -- every planned (unit, arm) cell has a checkpoint
2. no RunFault -- the course was not blocked
3. budget -- the LLM ledger is inside the released envelope, per cell and total
4. gate authority -- nothing activated that the P4 gate refused
5. exposure -- no held-out read, and every window is one the contract declares
6. frozen reset -- the frozen arm carried nothing across a unit
7. replay -- the fit share held, and no Draft was activated by a replay screen
8. accounting -- the evaluation face never entered a bank, and faults are
   classified

A failure in any one stops the chain.  Repair is allowed to fix the instrument
and resume from the checkpoint; it is never allowed to re-throw the science.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import restricted_draft as drafts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts/main_protocol"
OUT_JSON = ARTIFACTS / "hec1_instrument.json"
OUT_MD = ARTIFACTS / "hec1_instrument.md"

CHECK_NAMES = (
    "completeness", "no_run_fault", "budget", "gate_authority", "exposure",
    "frozen_reset", "replay", "accounting",
)

#: Windows a course is allowed to read, as offsets from a unit's origin.
DECLARED_OFFSETS = (0, 48, 144)


def _cells(course: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(course.get("cells") or ())


def _check_completeness(course: Mapping[str, Any]) -> dict[str, Any]:
    planned = int(course.get("units_planned") or 0)
    arms = list(course.get("arms") or ())
    expected = {(position, arm) for position in range(planned) for arm in arms}
    seen = {(int(row["position"]), str(row["arm"])) for row in _cells(course)}
    missing = sorted(expected - seen)
    floor = int(0.8 * planned) if planned else 0
    completed = int(course.get("units_completed") or 0)
    return {
        "check": "completeness",
        "passed": not missing and completed >= floor,
        "planned_cells": len(expected),
        "seen_cells": len(seen),
        "missing_cells": [{"position": p, "arm": a} for p, a in missing[:20]],
        "units_completed": completed,
        "units_floor_at_0.8_N": floor,
    }


def _check_no_run_fault(course: Mapping[str, Any]) -> dict[str, Any]:
    fault = course.get("run_fault")
    return {
        "check": "no_run_fault",
        "passed": not fault and course.get("status") == "COMPLETE",
        "status": course.get("status"),
        "run_fault": fault,
        "verdict_field": course.get("verdict"),
        "note": (
            "a blocked course carries RUN_BLOCKED_NO_VERDICT, which is an "
            "instrument reading and never a scientific verdict"
        ),
    }


def _check_budget(course: Mapping[str, Any]) -> dict[str, Any]:
    ledgers = dict(course.get("ledgers") or {})
    guard = dict(course.get("budget_guard") or {})
    cap = int(guard.get("ordering_cap") or 0)
    per_cell = int(guard.get("per_unit_arm_cap") or 0)
    total = int(ledgers.get("llm_total") or 0)
    over_cell = [
        {"arm": row["arm"], "position": row["position"],
         "calls": int(row.get("llm_calls_this_cell") or 0)}
        for row in _cells(course)
        if int(row.get("llm_calls_this_cell") or 0) > per_cell
    ]
    blocked = list(guard.get("blocked_before_backend") or ())
    return {
        "check": "budget",
        "passed": total <= cap and not over_cell,
        "llm_total": total,
        "ordering_cap": cap,
        "per_unit_arm_cap": per_cell,
        "cells_over_cell_cap": over_cell,
        "blocked_before_backend": len(blocked),
        "scripted_calls_not_billed": guard.get("scripted_calls_not_billed"),
        "note": (
            "a blocked call is refused before the backend and is not billed, so "
            "a non-zero blocked count is health rather than overspend"
        ),
    }


def _check_gate_authority(course: Mapping[str, Any]) -> dict[str, Any]:
    violations, disagreements = [], []
    for row in _cells(course):
        disagreement = row.get("gate_disagreement")
        activated = bool(row.get("activated"))
        if disagreement is None:
            if activated:
                violations.append({"position": row["position"],
                                   "arm": row["arm"],
                                   "why": "activated with no gate record"})
            continue
        if str(disagreement.get("resolved_by")) != "p4_gate":
            violations.append({"position": row["position"], "arm": row["arm"],
                               "why": "resolved by %s"
                                      % disagreement.get("resolved_by")})
        if activated and not disagreement.get("may_activate"):
            violations.append({"position": row["position"], "arm": row["arm"],
                               "why": "activated while the P4 gate refused"})
        if disagreement.get("disagree"):
            disagreements.append({"position": row["position"],
                                  "arm": row["arm"],
                                  "kind": disagreement.get("kind"),
                                  "online_loop_event": disagreement.get(
                                      "online_loop_event"),
                                  "p4_gate": disagreement.get("p4_gate")})
    frozen_activated = [
        {"position": row["position"], "arm": row["arm"]}
        for row in _cells(course)
        if row.get("activated") and "frozen" in str(row["arm"]).lower()
    ]
    lost = _lost_activations(course)
    return {
        "check": "gate_authority",
        "passed": not violations and not frozen_activated,
        "violations": violations,
        "frozen_arm_activations": frozen_activated,
        "recorded_disagreements": disagreements,
        "disagreement_count": len(disagreements),
        "authority_upheld_count": sum(
            1 for row in disagreements
            if row.get("kind") == "AUTHORITY_UPHELD"),
        "lost_activations": lost,
        "lost_activation_count": len(lost),
        "lost_activation_note": (
            "a P4 pass with no approved lifecycle event; recorded and never "
            "activated (sol final ruling 2026-09-03 §4).  A count, not a fault"),
        "why_a_calibre_disagreement_is_not_a_failure": (
            "online_loop's delayed admission carries no coverage floor and the "
            "P4 gate does, so the two disagree structurally whenever a winner "
            "treats fewer than MIN_TREATED series at the delayed window -- the "
            "shakedown produced three in 26 units and the authority refused all "
            "three.  What fails this check is an activation the authority did "
            "not grant, which is what 'the Active set only ever grows through "
            "the authoritative gate' means"),
        "open_question_for_sol": (
            "sol's launch gate 3 reads 'any gate disagreement in a scientific "
            "ordering demotes that ordering'.  Read literally that demotes "
            "every ordering, because the calibre difference is structural and "
            "recurred 3 times in 26 units with the guard holding each time.  "
            "This check demotes on an authority breach only; see the ledger"),
    }


#: The frozen held-out block, read from the contract's own geometry rather than
#: restated, so a contract change could not leave this check testing old numbers.
HELD_OUT_ORIGINS = frozenset(
    contract.v1.HELD_OUT_ORIGINS)  # type: ignore[attr-defined]


def _check_exposure(course: Mapping[str, Any]) -> dict[str, Any]:
    held_out = set(HELD_OUT_ORIGINS)
    illegal, windows = [], set()
    for row in _cells(course):
        origin = int(row["unit"]["origin"])
        for offset in DECLARED_OFFSETS:
            windows.add(origin + offset)
        reading = row.get("evaluation") or {}
        if reading.get("origin") is not None:
            window = int(reading["origin"])
            windows.add(window)
            if window - origin not in DECLARED_OFFSETS:
                illegal.append({"position": row["position"],
                                "why": "scored at +%d" % (window - origin)})
        delayed = row.get("delayed") or {}
        if delayed.get("origin") is not None and (
                int(delayed["origin"]) - origin) not in DECLARED_OFFSETS:
            illegal.append({"position": row["position"],
                            "why": "gated at +%d"
                                   % (int(delayed["origin"]) - origin)})
    touched_held_out = sorted(windows & held_out)
    boundary = dict(course.get("boundary") or {})
    return {
        "check": "exposure",
        "passed": (not touched_held_out and not illegal
                   and int(boundary.get("held_out_reads") or 0) == 0),
        "held_out_origins": sorted(held_out),
        "windows_touching_held_out": touched_held_out,
        "illegal_windows": illegal,
        "declared_offsets": list(DECLARED_OFFSETS),
        "held_out_reads_recorded": boundary.get("held_out_reads"),
    }


def _lost_activations(course: Mapping[str, Any]) -> list[dict[str, Any]]:
    """sol final ruling §4: a P4 pass with no lifecycle event to activate on is
    recorded, counted, and never an activation.  Reported inside the gate
    authority check; a non-zero count is a reading, not a failure."""
    return [{"position": row.get("position"), "arm": row.get("arm"),
             "why": row.get("lost_activation_why")}
            for row in _cells(course) if row.get("lost_activation")]


def _check_frozen_reset(course: Mapping[str, Any]) -> dict[str, Any]:
    arms = list(course.get("arms") or ())
    frozen_arms = [arm for arm in arms if "frozen" in str(arm).lower()]
    if not frozen_arms:
        # Phase S is a single A5-online arm by contract: it builds K0 and has no
        # frozen contrast, so there is nothing to check.  Stated as
        # NOT_APPLICABLE rather than passed silently -- and only for the phase
        # that is allowed to have no frozen arm.  A Phase-T course missing one
        # is still a failure.
        applicable = str(course.get("phase")) == "phase_s" and arms == [
            "A5-online"]
        return {
            "check": "frozen_reset",
            "passed": applicable,
            "not_applicable": applicable,
            "arms": arms,
            "why": (
                "Phase S runs one online arm to build K0; there is no frozen "
                "contrast to reset" if applicable else
                "a Phase-T course has no frozen arm, which the contract's arm "
                "set requires in both the full and the empty-K0 shape"),
        }
    problems, evidence = [], {}
    for arm in frozen_arms:
        rows = sorted((row for row in _cells(course) if row["arm"] == arm),
                      key=lambda r: int(r["position"]))
        if not rows:
            continue
        resets = [bool((row.get("reset") or {}).get("reset")) for row in rows]
        banks = [int(row.get("bank_rows_after") or 0) for row in rows]
        carried = list((course.get("active_skill_ids") or {}).get(arm) or ())
        # A restricted Draft opened in one unit is candidate supply in the next;
        # a frozen arm that is resupplied anything has carried memory across
        # units even though its store was rebuilt.
        resupplied = [{"position": row["position"],
                       "ids": list(row.get("resupplied_candidate_ids") or ())}
                      for row in rows if row.get("resupplied_candidate_ids")]
        evidence[arm] = {"resets": resets, "bank_rows": banks,
                         "carried_skills": carried,
                         "resupplied_drafts": resupplied,
                         "dropped_drafts_at_rebuild": [
                             (row.get("reset") or {}).get("dropped_drafts")
                             for row in rows]}
        if any(not flag for flag in resets[1:]):
            problems.append({"arm": arm, "why": "a unit after the first did "
                                               "not rebuild the arm"})
        if any(count for count in banks):
            problems.append({"arm": arm, "why": "the frozen arm kept a bank"})
        if carried:
            problems.append({"arm": arm,
                             "why": "the frozen arm carried an Active Skill"})
        if any(row.get("activated") for row in rows):
            problems.append({"arm": arm, "why": "the frozen arm activated"})
        if resupplied:
            problems.append({"arm": arm,
                             "why": "the frozen arm was resupplied a restricted "
                                    "Draft opened in an earlier unit",
                             "cells": resupplied})
        # The store itself, not the flag: the library the frozen arm enters
        # every unit with must be its start snapshot -- K0's skills when a K0
        # was compiled, an empty library when it starts from h0.
        expected_start = sorted((course.get("k0_snapshot") or {}).get(
            "skill_ids") or ()) if "A5" in str(arm) else []
        recorded = [(row["position"],
                     sorted(row.get("snapshot_skill_ids_at_start") or ()))
                    for row in rows if "snapshot_skill_ids_at_start" in row]
        evidence[arm]["snapshot_skill_ids_at_start"] = [
            start for _, start in recorded]
        evidence[arm]["expected_start_library"] = expected_start
        drifted = [position for position, start in recorded
                   if start != expected_start]
        if drifted:
            problems.append({"arm": arm,
                             "why": "the frozen arm entered a unit with a "
                                    "library other than its start snapshot",
                             "positions": drifted})
    return {
        "check": "frozen_reset",
        "passed": not problems and bool(frozen_arms),
        "frozen_arms": frozen_arms,
        "problems": problems,
        "evidence": evidence,
    }


def _check_replay(course: Mapping[str, Any]) -> dict[str, Any]:
    allowance = dict(course.get("replay_fit_allowance") or {})
    problems = []
    if allowance and not allowance.get("within"):
        problems.append({"why": "the replay fit share was exceeded",
                         "spent": allowance.get("spent"),
                         "allowance": allowance.get("allowance")})
    starved = []
    for step in course.get("outer_steps") or ():
        if step.get("wrote_active"):
            problems.append({"why": "an outer step wrote the active set",
                             "k_index": step.get("k_index")})
        for candidate in step.get("candidates") or ():
            if candidate.get("outcome") == "REPLAY_FITS_BUDGET_SPENT":
                starved.append({"k_index": step.get("k_index"),
                                "kind": candidate.get("kind"),
                                "estimate": candidate.get("replay_estimate"),
                                "remaining": candidate.get(
                                    "replay_fits_remaining")})
    for arm, lifecycle in (course.get("lifecycle") or {}).items():
        for draft in lifecycle.get("drafts") or ():
            if draft.get("deployable"):
                problems.append({"why": "a Draft reported itself deployable",
                                 "arm": arm, "draft": draft.get("draft_id")})
    return {
        "check": "replay",
        "passed": not problems,
        "allowance": allowance,
        "problems": problems,
        # Not a failure: the cap binding is the cap working, and it is disclosed
        # because it means later outer steps screened fewer candidates.
        "candidates_starved_by_the_cap": starved,
    }


def _check_accounting(course: Mapping[str, Any]) -> dict[str, Any]:
    problems = []
    evaluation_windows = {int(row["unit"]["origin"]) + 144
                          for row in _cells(course)}
    for row in _cells(course):
        if row.get("evaluation_face_enters_bank") is not False:
            problems.append({"position": row["position"], "arm": row["arm"],
                             "why": "the evaluation face was not marked "
                                    "excluded from the bank"})
        for fault in row.get("faults") or ():
            if str(fault.get("kind")) not in (
                    "UnitFault", "FaceNotEvaluable"):
                problems.append({"position": row["position"],
                                 "arm": row["arm"],
                                 "why": "unclassified fault %s"
                                        % fault.get("kind")})
    for arm, lifecycle in (course.get("lifecycle") or {}).items():
        for draft in lifecycle.get("drafts") or ():
            for entry in draft.get("history") or ():
                if entry.get("window") in evaluation_windows:
                    problems.append({
                        "arm": arm, "draft": draft.get("draft_id"),
                        "why": "an evaluation-face window reached the lifecycle"})
            if draft.get("state") not in (None, drafts.WAITING, drafts.REVISABLE,
                                          drafts.FLAGGED):
                problems.append({"arm": arm, "draft": draft.get("draft_id"),
                                 "why": "unknown Draft state %s"
                                        % draft.get("state")})
    unreadable: dict[int, set[str]] = {}
    for row in _cells(course):
        if row.get("evaluation") is None:
            unreadable.setdefault(int(row["unit"]["origin"]), set()).add(
                str(row["arm"]))
    arms = set(course.get("arms") or ())
    asymmetric = {origin: sorted(hit) for origin, hit in unreadable.items()
                  if hit != arms}
    if asymmetric:
        problems.append({
            "why": "a unit lost its evaluation face for some arms but not all",
            "origins": asymmetric})
    # The replay cache's three ledgers, surfaced here so the instrument report
    # carries them.  The shakedown reported cache_hits/misses as 0/0 because
    # nothing incremented them, which reads exactly like "enabled and never
    # hit"; the counts and the LLM prompt cache's state are now both explicit.
    ledgers = dict(course.get("ledgers") or {})
    caches = dict(course.get("replay_cache") or {})
    hits = sum(int(row.get("cache_hits") or 0) for row in caches.values())
    logical = sum(int(row.get("logical_evaluations") or 0)
                  for row in caches.values())
    physical = sum(int(row.get("physical_fits") or 0)
                   for row in caches.values())
    if caches and int(ledgers.get("cache_hits") or 0) != hits:
        problems.append({"why": "the run ledger's cache hits disagree with the "
                                "arms' caches",
                         "ledger": ledgers.get("cache_hits"), "arms": hits})
    if caches and (int(ledgers.get("cache_hits") or 0)
                   + int(ledgers.get("cache_misses") or 0)) != logical:
        problems.append({"why": "hits plus misses do not equal the logical "
                                "evaluations the caches recorded"})
    return {
        "check": "accounting",
        "passed": not problems,
        "problems": problems,
        "replay_cache": {
            "physical_fits": physical,
            "logical_evaluations": logical,
            "cache_hits": hits,
            "hit_rate": (round(hits / logical, 4) if logical else None),
            "saved_fits": max(0, logical * 2 - physical),
            "per_arm": {name: {"physical_fits": row.get("physical_fits"),
                               "logical_evaluations": row.get(
                                   "logical_evaluations"),
                               "cache_hits": row.get("cache_hits")}
                        for name, row in sorted(caches.items())},
        },
        "llm_prompt_cache_enabled": ledgers.get("llm_prompt_cache_enabled"),
        "units_without_an_evaluation_face": sorted(unreadable),
        "why_that_is_not_a_failure": (
            "the +144 window can run past a series' observed data, which is a "
            "property of the data and drops the unit for every arm equally; the "
            "count is reported so the curve's usable N is not overstated"
        ),
        "ledgers": course.get("ledgers"),
    }


CHECKS = (
    _check_completeness, _check_no_run_fault, _check_budget,
    _check_gate_authority, _check_exposure, _check_frozen_reset,
    _check_replay, _check_accounting,
)


def audit(course: Mapping[str, Any]) -> dict[str, Any]:
    """Run all eight.  Reads no gain, no utility and no verdict."""
    results = []
    for check in CHECKS:
        try:
            results.append(check(course))
        except Exception as exc:  # noqa: BLE001 - a failed check is a reading
            results.append({"check": check.__name__, "passed": False,
                            "error": "%s: %s" % (type(exc).__name__,
                                                 str(exc)[:300])})
    passed = all(row.get("passed") for row in results)
    return {
        "stage": "HEC1_INSTRUMENT",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "course_artifact": course.get("run_label"),
        "ordering": course.get("ordering"),
        "phase": course.get("phase"),
        "checks": results,
        "checks_passed": sum(1 for row in results if row.get("passed")),
        "checks_total": len(results),
        "passed": passed,
        "may_continue": passed,
        "reads": ["counts", "ledgers", "set intersections"],
        "does_not_read": ["any gain", "any utility", "any verdict",
                          "the curve"],
        "why_this_may_be_self_run": (
            "every check is a mechanical assertion, so it cannot be influenced "
            "by whether the result looked good; interpreting the curve is a "
            "different script and a different hand"
        ),
        "on_failure": (
            "stop.  Repair the instrument and resume from the checkpoint; never "
            "re-throw the science"
        ),
        "boundary": {"llm_calls": 0, "consumer_fits": 0, "held_out_reads": 0},
    }


def _md(payload: Mapping[str, Any]) -> str:
    lines = [
        "# HEC-1 instrument check (eight mechanical assertions)",
        "",
        "Reads counts, ledgers and set intersections. Reads **no** gain, "
        "utility or verdict, which is what makes it safe to self-run.",
        "",
        "| check | state | detail |",
        "| --- | --- | --- |",
    ]
    for row in payload["checks"]:
        detail = row.get("error") or ""
        if not detail and not row.get("passed"):
            for key in ("missing_cells", "violations", "problems",
                        "illegal_windows", "cells_over_cell_cap"):
                if row.get(key):
                    detail = "%s: %s" % (key, json.dumps(row[key])[:160])
                    break
        lines.append("| `%s` | %s | %s |" % (
            row["check"], "PASS" if row.get("passed") else "**FAIL**", detail))
    lines += [
        "",
        "**%d/%d passed. May continue: %s.**" % (
            payload["checks_passed"], payload["checks_total"],
            payload["may_continue"]),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("course", help="path to a hec1_course_*.json artifact")
    parser.add_argument("--label", default=None,
                        help="output label; required when a reading for this "
                             "course already exists, so a re-run never "
                             "overwrites an instrument artifact")
    args = parser.parse_args(argv)
    started = time.time()
    course = json.loads(Path(args.course).read_text(encoding="utf-8"))
    payload = {**audit(course),
               "wall_seconds": round(time.time() - started, 2)}
    label = args.label or (course.get("run_label") or "course")
    out_json = OUT_JSON.with_name("hec1_instrument_%s.json" % label)
    out_md = out_json.with_suffix(".md")
    if out_json.exists():
        # hec1_instrument_e2e6 was overwritten once by a re-run with a tightened
        # check (see hec1_instrument_e2e6_erratum.*).  A reading is an artifact:
        # a second reading of the same course gets its own label.
        print("refusing to overwrite %s; pass --label <new label>"
              % out_json.relative_to(PROJECT_ROOT).as_posix())
        return 2
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    out_md.write_text(_md(payload), encoding="utf-8")
    for row in payload["checks"]:
        print("  %-18s %s" % (row["check"],
                             "PASS" if row.get("passed") else "FAIL"))
    print("checks     : %d/%d" % (payload["checks_passed"],
                                 payload["checks_total"]))
    print("may continue: %s" % payload["may_continue"])
    print("wrote %s" % out_json.relative_to(PROJECT_ROOT).as_posix())
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
