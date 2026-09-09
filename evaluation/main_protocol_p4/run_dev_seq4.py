"""DEV-SEQ-4 step 2: the same-start contrast.

DEV-SEQ-3 asked whether a Slow-written card reaches Fast.  It did, and Fast
changed its selection -- but the two arms had by then also accumulated
*different* Episode histories, and history reaches Fast through
``method.py:374`` -> ``fast_agent.py:804``.  So "the card changed the
selection" and "the divergent history changed the selection" were confounded,
and the exposed subset was n=3.

This runner removes the confound and adds repeats.  Six sequence-window
situations, two arms, three repeats -- 36 Fast decisions.  Every cell forks
from **the same legal pre-decision history**: the knowledge-frozen arm's own
history at that decision, rebuilt from the DEV-SEQ-3 checkpoints and handed to
*both* arms.  The only difference between the arms is whether the card is in
the snapshot.

Three situations are the ones DEV-SEQ-3 actually exposed.  The other three are
the closest decisions the card does *not* match, by a fixed rule (smallest
relative violation of the card's own clauses).  The card is not retrieved
there, so those cells measure what repeat-to-repeat sampling alone does on
comparable situations -- a noise floor read on this design rather than
borrowed from the population run.

This is a development diagnostic.  It does not replace the population utility
reading, it does not promote anything, and 36 decisions on 6 situations cannot
establish an effect size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_seq2 as seq2
from evaluation.main_protocol_p4 import run_dev_seq3 as seq3
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_seq4"

#: The DEV-SEQ-3 run this diagnostic forks from.  Both snapshots are read out
#: of its store by SHA -- no Slow call is made and no card is re-derived.
SOURCE_RUN = "pkg3"
SOURCE_GROUP = "g2"
STORE = base.ROOT / ".dev_seq3_runs" / SOURCE_RUN / SOURCE_GROUP / "store"
PARENT_SHA = "98dea3b037b790528cd745f8cbdf657da0285c25f91df53e9c572a0dfc87563b"
FORK_SHA = "c99dc92bbf89c1247313068720d5040f7513ce6516296d89edba1ba25f115f7c"
CARD_ID = "missingness_deviation_identity_guard"

#: Where the pre-decision histories are rebuilt from, oldest first.  Positions
#: 7-8 are DEV-SEQ-2's reused formation; 9-11 are DEV-SEQ-3's own checkpoints.
#: Only the knowledge-frozen arm's files are read: its history is the one that
#: exists without the edit, which is what the counterfactual needs.
HISTORY_CHAIN: tuple[tuple[int, Path], ...] = (
    (7, base.ROOT / "_scratch" / "dev_seq2" / "pkg2" / "g2" / "formation_u007.json"),
    (8, base.ROOT / "_scratch" / "dev_seq2" / "pkg2" / "g2" / "formation_u008.json"),
    (9, base.ROOT / "_scratch" / "dev_seq3" / "pkg3" / "g2"
        / "validation_knowledge_frozen.json"),
    (10, base.ROOT / "_scratch" / "dev_seq3" / "pkg3" / "g2"
         / "followup_u010_knowledge_frozen.json"),
    (11, base.ROOT / "_scratch" / "dev_seq3" / "pkg3" / "g2"
         / "followup_u011_knowledge_frozen.json"),
)

#: (position, series_uid, kind).  MATCHED = the card's own applicability holds,
#: these are exactly the three decisions DEV-SEQ-3 exposed.  NEAREST_MISS = the
#: three non-matching decisions with the smallest relative violation of a
#: clause, picked by that rule alone and before any of them was run here.
MATCHED = "matched"
NEAREST_MISS = "nearest_miss"
SITUATIONS: tuple[tuple[int, str, str], ...] = (
    (9, "T14", MATCHED),          # mf 0.5151  z 5.8358
    (10, "T14", MATCHED),         # mf 0.5272  z 5.2723
    (10, "T146", MATCHED),        # mf 0.4022  z 11.0087
    (11, "T146", NEAREST_MISS),   # mf 0.3910 -- short of 0.40 by 2.2%
    (10, "T158", NEAREST_MISS),   # mf 0.3820 -- short of 0.40 by 4.5%
    (11, "T14", NEAREST_MISS),    # z 4.7698 -- short of 5 by 4.6%
)
REPEATS = 3

ARM_A = seq2.ARM_A          # knowledge-frozen: the parent snapshot
ARM_B = seq2.ARM_B          # slow-candidate: the parent plus the card

MAX_FITS = 500
MAX_LLM = 600
MAX_WALL_SECONDS = 2 * 3600
PER_SEQUENCE_LLM_CAP = 24
CONCURRENCY = 4
REQUIRED_MODEL = "gpt-5.6-sol"


# ---------------------------------------------------------------------------
# the fixed pre-decision history
# ---------------------------------------------------------------------------

def history_before(position: int, uid: str) -> list[dict[str, Any]]:
    """The knowledge-frozen arm's Episodes for one sequence, before ``position``.

    Rebuilt from checkpoints, never re-run.  ``verify_histories`` checks the
    rebuild against the ``history_episodes_in`` that DEV-SEQ-3 recorded at the
    time, so a silent drift in this chain cannot pass unnoticed.
    """
    rows: list[dict[str, Any]] = []
    for pos, path in HISTORY_CHAIN:
        if pos >= int(position):
            break
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows.extend((saved.get("episodes") or {}).get(str(uid), []))
    return rows


def rehydrate(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415
        ExperienceEpisode,
    )
    out = []
    for row in rows:
        payload = dict(row)
        payload["evidence_refs"] = tuple(payload.get("evidence_refs") or ())
        out.append(ExperienceEpisode(**payload))
    return out


def verify_histories() -> dict[str, Any]:
    """Every rebuilt history must equal the count DEV-SEQ-3 recorded."""
    doc = json.loads(
        (ART / ("dev_seq3_guidance_delivery__%s.json" % SOURCE_GROUP)
         ).read_text(encoding="utf-8"))
    recorded: list[tuple[int, str, int]] = []
    for arm, block in ((doc.get("validation") or {}).get("units") or {}).items():
        if arm == ARM_A:
            for row in block.get("sequences") or ():
                recorded.append((9, str(row["series_uid"]),
                                 int(row.get("history_episodes_in") or 0)))
    for unit in doc.get("followup_units") or ():
        if str(unit.get("arm")) == ARM_A:
            for row in unit.get("sequences") or ():
                recorded.append((int(unit["position"]), str(row["series_uid"]),
                                 int(row.get("history_episodes_in") or 0)))
    mismatches = [{"position": pos, "series_uid": uid,
                   "rebuilt": len(history_before(pos, uid)), "recorded": want}
                  for pos, uid, want in recorded
                  if len(history_before(pos, uid)) != want]
    return {
        "decisions_checked": len(recorded),
        "exact": len(recorded) - len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
        "what_this_checks": (
            "that the history handed to both arms here is bit-for-bit the "
            "history the knowledge-frozen arm actually held at that decision "
            "in DEV-SEQ-3, not an approximation of it"),
    }


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def run_cell(*, situation: tuple[int, str, str], arm_name: str, repeat: int,
             snapshots: Mapping[str, Any], ctx: Any, readings: Any,
             machinery: Mapping[str, Any], budget: seq2.Budget,
             shared: seq3.SharedState, edit_probe: str,
             cache: Any) -> dict[str, Any]:
    """One (situation, arm, repeat): its own session, its own fixed history."""
    position, uid, kind = situation
    rows = history_before(position, uid)

    guard = seq3.SequenceGuard(ordering_cap=MAX_LLM,
                               per_unit_arm_cap=PER_SEQUENCE_LLM_CAP,
                               ledgers=budget.ledgers, lock=shared.ledger)

    def backend_factory(inner_guard: Any) -> Any:
        inner = machinery["agentic"]._default_backend_factory(
            PER_SEQUENCE_LLM_CAP)
        return runner._MeteredFastBackend(seq3.TransientRelayRetry(inner),
                                          guard=inner_guard, billable=True)

    # One Arm object per cell: nothing is shared between cells but the
    # prediction cache and the ledger, so the fixed history cannot race and
    # the Episodes this cell writes cannot leak into another cell.
    arm = seq2.Arm(arm_name, snapshots[arm_name], machinery=machinery,
                   backend_factory=backend_factory, guard=guard, cache=cache)
    arm.history[str(uid)] = rehydrate(rows)

    record = seq3.run_sequence(
        arm=arm, ctx=ctx, uid=uid, readings=readings, machinery=machinery,
        budget=budget, position=position, phase="contrast", guard=guard,
        shared=shared, edit_probe=edit_probe)
    record["situation_kind"] = kind
    record["repeat"] = int(repeat)
    record["history_episodes_fixed_at"] = len(rows)
    record["history_is_identical_across_arms"] = True
    return record


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def summarise(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What changed, split by whether the card could reach the decision."""
    by_kind: dict[str, Any] = {}
    for kind in (MATCHED, NEAREST_MISS):
        block: dict[str, Any] = {}
        rows = [r for r in records if r.get("situation_kind") == kind]
        for arm in (ARM_A, ARM_B):
            mine = [r for r in rows if r.get("arm") == arm]
            gains = [g for g in (_num(r.get("delayed_gain")) for r in mine)
                     if g is not None]
            block[arm] = {
                "decisions": len(mine),
                "card_in_view": sum(
                    1 for r in mine
                    if r.get("edited_text_is_in_this_sequences_view")),
                "card_retrieved": sum(
                    1 for r in mine
                    if CARD_ID in (r.get("retrieved_skill_ids") or ())),
                "select_identity": sum(
                    1 for r in mine
                    if r.get("chosen_candidate_id") == "identity"),
                "deployed": sum(1 for r in mine
                                if r.get("decision") == "DEPLOYED"),
                "candidates_total": sum(int(r.get("candidate_count") or 0)
                                        for r in mine),
                "tool_calls_total": sum(int(r.get("tool_calls") or 0)
                                        for r in mine),
                "llm_calls_total": sum(int(r.get("llm_calls_this_sequence") or 0)
                                       for r in mine),
                "delayed_mean": (round(sum(gains) / len(gains), 6)
                                 if gains else "UNKNOWN"),
                "delayed_readable": len(gains),
            }
        block["what_this_block_can_show"] = (
            "the card is retrievable here, so a difference between the arms "
            "may be the card"
            if kind == MATCHED else
            "the card is NOT retrievable here, so both arms hold the same "
            "effective knowledge and every difference is repeat-to-repeat "
            "sampling on comparable situations")
        by_kind[kind] = block

    per_situation = []
    for position, uid, kind in SITUATIONS:
        cell: dict[str, Any] = {"position": position, "series_uid": uid,
                                "kind": kind}
        for arm in (ARM_A, ARM_B):
            mine = sorted(
                (r for r in records
                 if r.get("arm") == arm and int(r.get("position", -1)) == position
                 and str(r.get("series_uid")) == uid),
                key=lambda r: int(r.get("repeat") or 0))
            cell[arm] = {
                "repeats": len(mine),
                "select": [r.get("chosen_candidate_id") for r in mine],
                "delivered": [r.get("deployed_label") for r in mine],
                "candidates": [r.get("candidate_count") for r in mine],
                "tool_calls": [r.get("tool_calls") for r in mine],
                "delayed": [r.get("delayed_gain") for r in mine],
                "card_retrieved": [CARD_ID in (r.get("retrieved_skill_ids") or ())
                                   for r in mine],
            }
        per_situation.append(cell)

    return {
        "by_situation_kind": by_kind,
        "per_situation": per_situation,
        "how_to_read_it": (
            "compare the two arms WITHIN a kind, and compare the matched "
            "block against the nearest-miss block.  A difference in the "
            "matched block that the nearest-miss block does not also show is "
            "the only thing here that can be the card.  Six situations and "
            "three repeats cannot size an effect; this separates 'the card "
            "did something' from 'the model varies between runs'"),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(*, run_id: str = "", concurrency: int | None = None
          ) -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    budget = seq2.Budget()
    seq2.MAX_FITS, seq2.MAX_LLM = MAX_FITS, MAX_LLM
    seq2.MAX_WALL_SECONDS = MAX_WALL_SECONDS
    seq2.GROUP_WALL_SECONDS = MAX_WALL_SECONDS
    budget.open_group()

    checkpoints = CHECKPOINTS / (run_id or "run")
    checkpoints.mkdir(parents=True, exist_ok=True)

    histories = verify_histories()
    if not histories["passed"]:
        return {"stage": "DEV_SEQ4_SAME_START_CONTRAST",
                "status": "BLOCKED_HISTORY_REBUILD_DISAGREES",
                "history_check": histories}

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)

    transport = seq3.transport_preflight(machinery)
    budget.spend_llm("preflight", int(transport.get("calls") or 0))
    if not transport.get("usable"):
        return {"stage": "DEV_SEQ4_SAME_START_CONTRAST",
                "status": "BLOCKED_BY_TRANSPORT", "transport": transport,
                "history_check": histories}

    snapshots = {
        ARM_A: machinery["compile_snapshot"](STORE / PARENT_SHA,
                                             verify_lock=False),
        ARM_B: machinery["compile_snapshot"](STORE / FORK_SHA,
                                             verify_lock=False),
    }
    identities = {
        ARM_A: snapshots[ARM_A].runtime_bundle_sha,
        ARM_B: snapshots[ARM_B].runtime_bundle_sha,
    }
    if identities[ARM_A] != PARENT_SHA or identities[ARM_B] != FORK_SHA:
        return {"stage": "DEV_SEQ4_SAME_START_CONTRAST",
                "status": "BLOCKED_SNAPSHOT_IDENTITY",
                "identities": identities}

    card = next((s for s in snapshots[ARM_B].skills
                 if str(s.skill_id) == CARD_ID), None)
    if card is None:
        return {"stage": "DEV_SEQ4_SAME_START_CONTRAST",
                "status": "BLOCKED_CARD_NOT_IN_FORK"}
    body = str(card.body)
    edit_probe = body[60:180]
    parent_ids = {str(s.skill_id) for s in snapshots[ARM_A].skills}

    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population_rows, _excluded = R._population(doc)
    by_position = {row["position"]: row for row in population_rows}

    positions = sorted({p for p, _u, _k in SITUATIONS})
    units = {p: R.opp._unit_ctx(by_position[p]["unit"]) for p in positions}
    cache = runner.ReplayPredictionCache("devseq4")
    shared = seq3.SharedState()
    unit_readings = {
        p: seq3.LockedReadings(
            ps.UnitReadings(cache=cache, ctx=units[p],
                            executor=units[p].executor, guard=budget.require),
            shared.fits)
        for p in positions}

    cells = [(situation, arm, repeat)
             for situation in SITUATIONS
             for repeat in range(1, REPEATS + 1)
             for arm in (ARM_A, ARM_B)]

    records: list[dict[str, Any]] = []
    stopped_at: dict[str, Any] | None = None
    width = int(concurrency or CONCURRENCY)
    with ThreadPoolExecutor(max_workers=width) as pool:
        futures = {
            pool.submit(run_cell, situation=situation, arm_name=arm,
                        repeat=repeat, snapshots=snapshots,
                        ctx=units[situation[0]], readings=unit_readings[situation[0]],
                        machinery=machinery, budget=budget, shared=shared,
                        edit_probe=edit_probe, cache=cache): (situation, arm, repeat)
            for situation, arm, repeat in cells}
        for future in futures:
            try:
                records.append(future.result())
            except seq2.PackageCeiling as exc:
                stopped_at = stopped_at or {"kind": "PACKAGE_CEILING",
                                            "why": str(exc)}
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                situation, arm, repeat = futures[future]
                records.append({
                    "phase": "contrast", "position": situation[0],
                    "series_uid": situation[1], "situation_kind": situation[2],
                    "arm": arm, "repeat": repeat,
                    "decision": "FAULT_NO_DECISION",
                    "support_gain": "UNKNOWN", "delayed_gain": "UNKNOWN",
                    "faults": [{"kind": type(exc).__name__,
                                "why": str(exc)[:300]}]})

    records.sort(key=lambda r: (int(r.get("position") or 0),
                                str(r.get("series_uid")),
                                str(r.get("arm")), int(r.get("repeat") or 0)))

    returned = sorted({m for r in records for m in (r.get("returned_models") or ())})
    return {
        "stage": "DEV_SEQ4_SAME_START_CONTRAST",
        "run_id": run_id or "run",
        "started": started.isoformat(),
        "finished": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "status": "STOPPED_AT_CEILING" if stopped_at else "COMPLETE",
        "stopped_at": stopped_at,
        "question": (
            "holding the pre-decision history fixed and repeating each cell "
            "three times, does the card change what Fast does?"),
        "design": {
            "situations": [{"position": p, "series_uid": u, "kind": k}
                           for p, u, k in SITUATIONS],
            "repeats": REPEATS,
            "arms": {ARM_A: PARENT_SHA, ARM_B: FORK_SHA},
            "cells": len(cells),
            "nearest_miss_rule": (
                "among the decisions the card does not match, the three with "
                "the smallest relative violation of one of its own clauses "
                "(missing_fraction >= 0.4, local_robust_z_peak >= 5).  Fixed "
                "before running and not revised afterwards"),
            "history_rule": (
                "both arms are handed the knowledge-frozen arm's own Episodes "
                "for that sequence at that decision, rebuilt from DEV-SEQ-3's "
                "checkpoints.  This is what removes the history confound that "
                "DEV-SEQ-3 could not separate from the card"),
        },
        "history_check": histories,
        "transport": transport,
        "snapshot_identities": identities,
        "card": {"skill_id": CARD_ID,
                 "in_parent": CARD_ID in parent_ids,
                 "probe_is_new_text": edit_probe not in json.dumps(
                     [str(s.body) for s in snapshots[ARM_A].skills],
                     ensure_ascii=False)},
        "records": records,
        "summary": summarise(records),
        "cost": budget.to_dict(),
        "returned_models": returned,
        "model_discipline": {
            "required": REQUIRED_MODEL,
            "requested_equals_returned": returned == [REQUIRED_MODEL],
        },
        "what_this_is_not": (
            "not a utility reading, not a population result, and not grounds "
            "to adopt or reject the card.  Six situations, three repeats: it "
            "can show that behaviour differs and roughly where, and it can "
            "show that the same-knowledge cells differ too.  It cannot size "
            "an effect and does not replace the population endpoint"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = build(run_id=args.run_id, concurrency=args.concurrency)
    out = Path(args.out) if args.out else (
        ART / "dev_seq4_same_start_contrast.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    print(out)
    print("status:", report.get("status"))
    summary = report.get("summary") or {}
    for kind, block in (summary.get("by_situation_kind") or {}).items():
        print(" ", kind)
        for arm in (ARM_A, ARM_B):
            row = block.get(arm) or {}
            print("   %-16s n=%s retrieved=%s select_identity=%s deployed=%s "
                  "cands=%s delayed=%s"
                  % (arm, row.get("decisions"), row.get("card_retrieved"),
                     row.get("select_identity"), row.get("deployed"),
                     row.get("candidates_total"), row.get("delayed_mean")))
    print("cost:", json.dumps(report.get("cost") or {}, default=str)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
