"""Which of the calibrated-feasible candidates could actually be verified?

Zero LLM, zero Consumer fits.  Scope resolution reads deployment-visible
feature cards through the production resolver and fits nothing, so the whole
finite space is checkable without spending anything.

This exists because M-R0g drew two wrong inferences from one screen refusal,
both corrected by non-author review:

1. The screen's condition is ``passed = not rejected and applicable > 0``
   (``outer_loop.py:895``) -- **one** applicable cell is enough, not five.  So
   "the pooled denominator makes narrowing structurally impossible" was not
   established.  What was established is narrower: *the one candidate tried*
   reached no applicable cell among the five.
2. ``violations == []`` was not a clean risk reading.  Every cell was skipped
   by ``_applicable`` **before** ``_violations`` ran, so nothing was checked.
   An empty violation list from a fully-skipped screen means "not measured",
   not "no harm".

So the question is re-asked over the whole finite candidate space rather than
the single best one, and in the terms that decide a candidate's fate:

* **screen eligibility** -- does at least one already-processed cell resolve at
  or above the coverage floor at its own origin, which is the face the screen
  replays on?
* **verification eligibility** -- is there a later cell whose delayed face
  resolves at or above the floor, so the authoritative gate could take a
  reading at all?
* **contrast** -- is there a later cell where the narrowed predicate selects a
  *different* series set than the ancestor, so a reading there could say
  anything about the narrowing rather than about the program?

The third is what p06 [0:40]@3576 failed for ``z_peak >= 6.0``: both predicates
select all 20 series there, so a verification on that unit measures the program
and is silent about the revision.  That fact is kept, not discarded -- it just
answers a different question.

What this does **not** do: it does not lower the coverage floor, change the
screen, change the candidate space, or fit anything.  A candidate that clears
all three columns here has *eligibility*, not benefit; whether narrowing helps
is a reading nobody has taken.

Run:  python -m evaluation.main_protocol_p4.audit_m_r0h_candidate_eligibility
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import audit_m_r0e_slow_input_truncation as e
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

ART = base.ART
FLOOR = int(contract.RISK["min_treated"])
DELAYED = int(runner.DELAYED_OFFSET)


def _candidate_rows() -> dict[str, Any]:
    """The k=1 REVISE candidate, captured off the real loop at zero cost."""
    doc = base.load(live.ORDERING)
    state = live._state_at_k1(doc)
    slow = e._CaptureSlow()
    outer_loop.consolidate(
        bank=state["bank"], ledger=state["ledger"], k_index=live.K_INDEX,
        slow=slow, replay=e._refusing_replay,
        budget=outer_loop.OuterBudget(
            outer_llm_per_step=int(contract.OUTER_LLM_PER_STEP),
            replay_fits_remaining=live.MAX_CONSUMER_FITS),
        held_lineage_keys=state["held"])
    return {"doc": doc, "state": state, "candidate": slow.captured[0]}


def _feasible(rows: Sequence[Mapping[str, Any]],
              existing: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every (feature, direction, edge) the calibration calls feasible."""
    surviving = tool._existing_survivors(rows, existing)
    out: list[dict[str, Any]] = []
    for feature in contract.SCOPE_CLASS["vocabulary"]:
        try:
            edges = tool._edges_for(str(feature), None)
        except Exception:  # noqa: BLE001 - not numeric in this vocabulary
            continue
        for direction in tool.DIRECTIONS:
            for edge in edges:
                reading = tool._reading(
                    tool._select(surviving, str(feature), direction, edge),
                    tool.BOUNDED_RISK_V1)
                if reading["feasible"]:
                    out.append({
                        "feature": str(feature), "direction": direction,
                        "threshold": float(edge),
                        "pooled_treated": reading["treated"],
                        "pooled_aggregate_gain": reading["aggregate_gain"]})
    return out


def _would_calibrate(feature: str, direction: str,
                     rows: Sequence[Mapping[str, Any]],
                     existing: Sequence[Mapping[str, Any]]) -> Any:
    """What ``calibrate`` actually returns for this pair -- the production pick.

    A (feature, direction) pair can have several feasible edges; only one of
    them is ever built, so the eligibility question is about that one.
    """
    try:
        got = tool.calibrate(feature=feature, direction=direction, rows=rows,
                             existing_clauses=existing)
    except tool.NoFeasibleThreshold:
        return None
    return got


def _cells(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(cell) for cell in base.cells(doc, live.ARM)]


def _coverage(ctx: Any, scope: Mapping[str, Any], origin: int) -> Any:
    try:
        return frozenset(ctx.resolve(scope, origin))
    except Exception as exc:  # noqa: BLE001 - recorded, never guessed
        return exc


def _row(ctx: Any, origin: int, scope, ancestor) -> dict[str, Any]:
    got = _coverage(ctx, scope, origin)
    anc = _coverage(ctx, ancestor, origin)
    if isinstance(got, Exception) or isinstance(anc, Exception):
        return {"origin": origin, "unresolvable":
                str(got if isinstance(got, Exception) else anc)[:160]}
    return {
        "origin": origin,
        "served": len(ctx.eval_uids),
        "treated": len(got),
        "ancestor_treated": len(anc),
        "at_or_above_floor": len(got) >= FLOOR,
        "differs_from_ancestor": got != anc,
        "series_only_the_ancestor_treats": len(anc - got),
    }


def build() -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    captured = _candidate_rows()
    doc, candidate = captured["doc"], captured["candidate"]
    rows = [dict(row) for row in (candidate.get("rows") or ())]
    existing = list(dict(candidate.get("base_scope") or {}).get("predicate")
                    or ())
    ancestor = dict(candidate.get("base_scope") or {})

    cells = _cells(doc)
    processed = [c for c in cells
                 if int(c["position"]) <= live.BOUNDARY_POSITION]
    later = [c for c in cells if int(c["position"]) > live.BOUNDARY_POSITION]
    ctxs = {int(c["position"]): opp._unit_ctx(dict(c["unit"])) for c in cells}

    triples = _feasible(rows, existing)
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for row in triples:
        pairs.setdefault((row["feature"], row["direction"]),
                         {"edges": []})["edges"].append(row)

    results: list[dict[str, Any]] = []
    for (feature, direction), group in sorted(pairs.items()):
        chosen = _would_calibrate(feature, direction, rows, existing)
        if chosen is None:
            results.append({"feature": feature, "direction": direction,
                            "status": "CALIBRATE_REFUSES_THE_PAIR",
                            "feasible_edges": group["edges"]})
            continue
        clause = dict(chosen["clause"])
        scope = {"scope_type": "serving_series_predicate",
                 "predicate": [dict(c) for c in existing] + [clause]}

        screen_rows = [{**_row(ctxs[int(c["position"])], ctxs[int(c["position"])].origin,
                               scope, ancestor),
                        "position": int(c["position"]),
                        "unit_origin": int(c["unit"]["origin"])}
                       for c in processed]
        applicable = [r for r in screen_rows if r.get("at_or_above_floor")]

        verify_rows = []
        for c in later:
            pos = int(c["position"])
            ctx = ctxs[pos]
            row = _row(ctx, ctx.face_origin(DELAYED), scope, ancestor)
            row.update({"position": pos,
                        "unit_origin": int(c["unit"]["origin"]),
                        "exposure": list(c["unit"].get("exposure") or ())})
            verify_rows.append(row)
        readable = [r for r in verify_rows if r.get("at_or_above_floor")]
        contrastive = [r for r in readable if r.get("differs_from_ancestor")]

        results.append({
            "feature": feature,
            "direction": direction,
            "calibrated_clause": clause,
            "calibrated_threshold_is_a_frozen_bin_edge": True,
            "pooled": {"treated": chosen["treated"],
                       "aggregate_gain": chosen["reading"]["aggregate_gain"],
                       "harmed_fraction": chosen["reading"]["harmed_fraction"],
                       "denominator": "the 100 pooled bank rows"},
            "feasible_edges_for_this_pair": group["edges"],
            "screen_eligibility": {
                "rule": ("outer_loop.screen: passed = not rejected and "
                         "applicable > 0 -- one applicable cell suffices"),
                "face": "each processed cell's own origin",
                "cells": screen_rows,
                "cells_at_or_above_floor": len(applicable),
                "eligible": bool(applicable),
            },
            "verification_eligibility": {
                "rule": ("the authoritative gate's coverage_floor line: a "
                         "delayed face resolving below %d yields no reading"
                         % FLOOR),
                "face": "each later cell's delayed face (+%d)" % DELAYED,
                "cells_at_or_above_floor": len(readable),
                "eligible": bool(readable),
                "first_readable": (readable[0] if readable else None),
            },
            "contrast_with_the_ancestor": {
                "rule": ("a later cell where the narrowed predicate selects a "
                         "different series set; without one, a reading there "
                         "measures the program and is silent about the "
                         "revision"),
                "readable_and_different": len(contrastive),
                "eligible": bool(contrastive),
                "first": (contrastive[0] if contrastive else None),
                "cells": verify_rows,
            },
            "all_three": bool(applicable) and bool(readable) and bool(contrastive),
        })

    complete = [r for r in results if r.get("all_three")]
    return {
        "stage": "M_R0H_CANDIDATE_ELIGIBILITY",
        "written_at": started.isoformat(),
        "cost": {"llm_calls": 0, "consumer_fits": 0,
                 "how": ("Scope resolution reads deployment-visible feature "
                         "cards through the production resolver and fits "
                         "nothing; no face is scored and no truth is read")},
        "corrects": [
            {"artifact": "m_r0g_deterministic_control__run1.md",
             "was": ("a narrowing across 5 cells effectively needs >=25 "
                     "treated series"),
             "is": ("the screen needs ONE applicable cell "
                    "(outer_loop.py:895); the established fact is only that "
                    "z_peak>=6.0 reached no applicable cell among the five"),
             "found_by": "non-author review"},
            {"artifact": "m_r0g_deterministic_control__run1.md",
             "was": "violations was empty, so the candidate harmed nothing",
             "is": ("all five cells were skipped by _applicable before "
                    "_violations ran; an empty list from a fully-skipped "
                    "screen is NOT_MEASURED, not no-harm"),
             "found_by": "non-author review"},
        ],
        "mechanism_as_now_stated": (
            "the calibrator searches for a good candidate on pooled evidence "
            "without asking whether that candidate could be validly verified "
            "on the units the protocol will actually read.  A pooled support "
            "gate and a per-unit coverage gate can both be right; what is "
            "unaligned is the selection objective and downstream verification "
            "eligibility.  This is also why the shadow's optimum is not the "
            "path's optimum."),
        "floor": FLOOR,
        "ancestor_scope": ancestor,
        "evidence": {"rows": len(rows), "units": len(processed)},
        "processed_positions": [int(c["position"]) for c in processed],
        "later_positions": [int(c["position"]) for c in later],
        "feasible_triples": len(triples),
        "distinct_feature_direction_pairs": len(pairs),
        "candidates": results,
        "summary": {
            "screen_eligible": sum(1 for r in results
                                   if (r.get("screen_eligibility") or {}).get(
                                       "eligible")),
            "verification_eligible": sum(
                1 for r in results
                if (r.get("verification_eligibility") or {}).get("eligible")),
            "contrastive": sum(1 for r in results
                               if (r.get("contrast_with_the_ancestor") or {}
                                   ).get("eligible")),
            "all_three": len(complete),
            "which": [{"clause": r["calibrated_clause"]} for r in complete],
        },
        "what_eligibility_is_not": (
            "benefit.  Clearing all three columns means a reading could be "
            "taken and could speak about the narrowing; whether the narrowing "
            "helps is a measurement nobody has made."),
        "code_state": base.version_check(),
    }


def main(argv: list[str] | None = None) -> int:
    report = build()
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "m_r0h_candidate_eligibility.json"
    if out.exists():
        print("refusing to overwrite %s" % out)
        return 2
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print("floor=%d  feasible triples=%d  pairs=%d"
          % (report["floor"], report["feasible_triples"],
             report["distinct_feature_direction_pairs"]))
    head = ("clause", "screen", "verify", "contrast", "all3")
    print("%-46s %6s %6s %8s %5s" % head)
    for row in report["candidates"]:
        if row.get("status"):
            print("%-46s %s" % ("%s %s" % (row["feature"], row["direction"]),
                                row["status"]))
            continue
        c = row["calibrated_clause"]
        print("%-46s %6s %6s %8s %5s" % (
            "%s %s %g" % (c["feature"], c["op"], c["threshold"]),
            "%d/%d" % (row["screen_eligibility"]["cells_at_or_above_floor"],
                       len(row["screen_eligibility"]["cells"])),
            "%d/%d" % (row["verification_eligibility"][
                "cells_at_or_above_floor"],
                len(row["contrast_with_the_ancestor"]["cells"])),
            row["contrast_with_the_ancestor"]["readable_and_different"],
            "YES" if row["all_three"] else "no"))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
