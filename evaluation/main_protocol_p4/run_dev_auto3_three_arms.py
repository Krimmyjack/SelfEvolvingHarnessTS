"""DEV-AUTO-3: splitting candidate supply from Skill update, three arms.

The question DEV-AUTO-2 could not answer
----------------------------------------
DEV-AUTO-2 opened two channels at once and the arm difference came mostly from
the first: a queued candidate is *offered* to later units and Fast may deploy
it before anything has been written back (+0.683 of the +0.798 total), while
the promoted card accounted for one cell (+0.142).  Two channels, one contrast,
no way to separate them.

Three arms, one difference each
-------------------------------
=========================  ==================  ==================  ============
arm                        Fast local          outer proposal      PATCH of the
                           adaptation          and candidate       target Skill
                                               supply              card
=========================  ==================  ==================  ============
A  control                 kept                no                  no
B  supply-only             kept                yes                 no
C  supply-and-update       kept                yes                 yes
=========================  ==================  ==================  ============

What is isolated is the PATCH of the K0 card's frozen program body, and only
that.  It is not a claim about every Memory mechanism.

The symmetry that makes B and C comparable
------------------------------------------
B must not become "the arm whose candidate is supplied forever".  In DEV-AUTO-2
a promoted Draft was closed and left the resupply pool, so an arm that never
promotes would keep re-offering the same candidate to every later unit while
the promoting arm would not -- a second mechanism difference smuggled in beside
the one under test.  Here the **verification event is identical in both arms**:
when a supplied candidate is deployed by Fast and the authoritative delayed
gate passes, both arms record it and both close the Draft.  C additionally
writes the program onto the card.  That single extra act is the contrast.

Everything else is held: the same proposer, the same Frame C queue and order,
the same calibration, the same admission rule, the same per-arm resource
allowance, the same course order.  Each arm keeps its own history, ledger,
prediction cache and proposal trajectory -- a candidate C found is never handed
to B, and the arms are never reset onto a shared history after they diverge.

Two groups
----------
The same course is run twice, with the arms executed in a different order in
each group so that whichever arm runs first does not systematically pay the
cold-cache cost.  Group 2 runs whatever group 1 showed.  This is a repeat-run
check, not an independent-data generalisation, and no significance is claimed.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_auto2_frame_c as D2
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_auto3"

ARM_A = "control"
ARM_B = "supply-only"
ARM_C = "supply-and-update"
SUPPLY_ARMS = (ARM_B, ARM_C)
UPDATING_ARMS = (ARM_C,)

MAX_FITS = 2000
MAX_LLM = 1000
MAX_WALL_SECONDS = 6 * 3600
PER_ARM_LLM = MAX_LLM // 3

#: Registered before the runs.  The arm order differs between groups so the
#: cold-cache cost does not always fall on the same arm; the seed is this
#: package's own RNG, and it is not a claim to control the model service.
GROUPS = {
    "g1": {"seed": 20260907, "arm_order": (ARM_A, ARM_B, ARM_C)},
    "g2": {"seed": 20260908, "arm_order": (ARM_C, ARM_B, ARM_A)},
}

K0_SKILL_ID = R.K0_SKILL_ID
ANCESTOR_BODY = None   # filled at build time from the K0 card


class Budget:
    """This package's ceilings, per-arm ledgers, same interface ``run_cell`` uses."""

    def __init__(self) -> None:
        self.started = time.time()
        self.fits_by_arm: dict[str, int] = {}
        self.llm_by_arm: dict[str, int] = {}

    def fits(self) -> int:
        return sum(self.fits_by_arm.values())

    def llm(self) -> int:
        return sum(self.llm_by_arm.values())

    def elapsed(self) -> float:
        return time.time() - self.started

    def spend_fits(self, arm: str, n: int) -> None:
        self.fits_by_arm[arm] = self.fits_by_arm.get(arm, 0) + int(n)

    def spend_llm(self, arm: str, n: int) -> None:
        self.llm_by_arm[arm] = self.llm_by_arm.get(arm, 0) + int(n)

    def room_for(self, fits: int) -> bool:
        return self.fits() + int(fits) <= MAX_FITS

    def llm_room(self, arm: str) -> int:
        mine = PER_ARM_LLM - int(self.llm_by_arm.get(arm, 0))
        return max(0, min(mine, MAX_LLM - self.llm()))

    def stop_reason(self) -> str | None:
        if self.fits() >= MAX_FITS:
            return "FIT_CEILING_REACHED"
        if self.llm() >= MAX_LLM:
            return "LLM_CEILING_REACHED"
        if self.elapsed() >= MAX_WALL_SECONDS:
            return "WALL_CLOCK_CEILING_REACHED"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_consumer_fits": self.fits(),
            "physical_consumer_fits_by_arm": dict(self.fits_by_arm),
            "fit_ceiling": MAX_FITS,
            "llm_calls": self.llm(),
            "llm_calls_by_arm": dict(self.llm_by_arm),
            "llm_ceiling": MAX_LLM,
            "llm_pre_allocated_per_arm": PER_ARM_LLM,
            "wall_seconds": round(self.elapsed(), 1),
            "wall_ceiling": MAX_WALL_SECONDS,
            "counting": ("every physical fit, including ones spent inside a "
                         "call that then raised; no allowance carried over; "
                         "no call is made to level the arms' cost"),
        }


def _card_body(arm: Any) -> str | None:
    return (R._skill_bodies(arm.active_snapshot()) or {}).get(K0_SKILL_ID)


def _verification_event(arm: Any, *, entry: Mapping[str, Any],
                        position: int, cell: Mapping[str, Any],
                        writes_back: bool) -> dict[str, Any]:
    """The event both supply arms record; only C then writes the card.

    Recorded identically on both sides, including the Draft's closure, so the
    candidate leaves the resupply pool at the same moment in both arms.  What
    differs is one act and its consequences, which is the whole contrast.
    """
    event: dict[str, Any] = {
        "arm": arm.spec.name,
        "program": cell.get("deployed_label"),
        "proposed_at_position": entry["proposed_at_position"],
        "verified_at_position": int(position),
        "draft_id": entry["draft_id"],
        "independent_unit": entry["proposed_at_position"] != int(position),
        "delayed_gate": cell.get("delayed_gate"),
        "deployed_via": cell.get("deployed_via"),
        # DEV-AUTO-3 section 4: when the candidate formed, against what.
        "parent_program_at_proposal": entry.get("parent_program_at_proposal"),
        "card_body_at_proposal": entry.get("card_body_at_proposal"),
        "card_body_before_the_event": _card_body(arm),
        "writes_back": bool(writes_back),
    }
    draft = arm.draft_ledger.by_id(entry["draft_id"])
    if writes_back:
        applied = rev.apply_workflow_revision(
            controller=arm._controller, store=arm._store,
            snapshot=arm.active_snapshot(), skill_id=K0_SKILL_ID,
            steps=entry["steps"],
            edit_id="dev-auto3-%s-u%03d" % (arm.spec.slug, position))
        event["write_back"] = {k: v for k, v in applied.items()
                               if k not in ("snapshot", "materialized")}
        if applied.get("applied"):
            arm._method._snapshot = applied["snapshot"]
            arm._store.set_active(applied["candidate_runtime_bundle_sha"])
            event["card_body_after_the_event"] = _card_body(arm)
            event["parent_program_at_write_back"] = rev.program_label(
                R.current_skill_steps(arm))
        closure = ("PROMOTED_TO_THE_SKILL_CARD" if applied.get("applied")
                   else "VERIFIED_BUT_THE_WRITE_BACK_FAILED")
    else:
        event["write_back"] = {
            "applied": False,
            "why": ("this arm does not write the target card; the "
                    "verification happened and is recorded, and the Draft is "
                    "closed exactly as it would be in the updating arm")}
        event["card_body_after_the_event"] = _card_body(arm)
        closure = "VERIFIED_BUT_NOT_WRITTEN_BACK"
    if draft is not None:
        arm.draft_ledger.close(draft, closure)
        event["draft_closed_as"] = draft.closed
    return event


def _knowledge_state(arm: Any, events: Sequence[Mapping[str, Any]],
                     position: int) -> dict[str, Any]:
    """What this call could have read: which updates had already landed.

    DEV-AUTO-2's two promotions were both proposed at u7, so nothing there
    showed a later proposal reacting to an earlier update.  This records the
    condition directly rather than leaving it to be inferred.
    """
    landed = [e for e in events
              if e.get("writes_back")
              and (e.get("write_back") or {}).get("applied")
              and int(e["verified_at_position"]) < int(position)]
    body = _card_body(arm)
    return {
        "card_body_now": body,
        "card_is_still_the_ancestor": body == ANCESTOR_BODY,
        "updates_that_had_already_landed": [
            {"program": e["program"], "verified_at_position":
                e["verified_at_position"]} for e in landed],
        "this_call_reads_an_updated_card": bool(landed),
        "parent_program_now": rev.program_label(R.current_skill_steps(arm)),
    }


def build(*, group: str = "g1", limit: int | None = None,
          label: str | None = None) -> dict[str, Any]:
    from evaluation.main_protocol_p4 import smoke_dev_auto3_three_arms as smoke

    global ANCESTOR_BODY
    started = datetime.now(timezone(timedelta(hours=8)))
    plan = GROUPS[group]
    label = label or group
    preflight = smoke.run()
    if not preflight["passed"]:
        return {"stage": "DEV_AUTO3_THREE_ARMS", "status": "BLOCKED_BY_SMOKE",
                "smoke": preflight}
    random.seed(int(plan["seed"]))
    np.random.seed(int(plan["seed"]) % (2 ** 32))

    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population, excluded = R._population(doc)
    index = D2._course_index(population)
    by_position = {row["position"]: row for row in population}
    after = [row for row in population if row["side"] == "after_the_boundary"]
    if limit is not None:
        after = after[:int(limit)]

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    snapshot_dir, snapshot_source = R._resolve_k0_snapshot(doc, state["k0"])
    if snapshot_dir is None:
        return {"stage": "DEV_AUTO3_THREE_ARMS", "status": "BLOCKED",
                "why": "no readable K0 snapshot"}
    start_snapshot = machinery["compile_snapshot"](snapshot_dir,
                                                   verify_lock=False)

    budget = Budget()
    ledgers = runner.Ledgers()
    guard = runner.BudgetGuard(ordering_cap=MAX_LLM,
                               per_unit_arm_cap=PER_ARM_LLM, ledgers=ledgers)
    root = base.ROOT / ".dev_auto3_runs" / label

    def backend_factory():
        inner = machinery["agentic"]._default_backend_factory(PER_ARM_LLM)
        return runner._MeteredFastBackend(inner, guard=guard, billable=True)

    arms: dict[str, Any] = {}
    for name in (ARM_A, ARM_B, ARM_C):
        spec = runner.ArmSpec(name, "k0", True, False, True)
        arm = runner.Arm(spec, root=root, machinery=machinery,
                         start_snapshot=start_snapshot, ledgers=ledgers,
                         guard=guard, backend_factory=backend_factory,
                         outer_slow_factory=None, offline=False)
        R._seed_arm(arm, state, fresh_ledger=True)
        arm._build("online")
        arm._store.materialize(arm.active_snapshot())
        arms[name] = arm
    ANCESTOR_BODY = _card_body(arms[ARM_A])

    outer_inner = machinery["agentic"]._default_backend_factory(MAX_LLM)
    outer_metered = runner._MeteredOuterBackend(outer_inner, guard=guard,
                                                billable=True)
    transport = machinery["agentic"].live_transport()
    outer_core = machinery["TTHAAgentCore"](
        outer_metered,
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=transport["model"], base_url=transport["base_url"])
    proposer = rev.RevisionProposer(
        outer_core, snapshot=start_snapshot,
        vocabulary=contract.SCOPE_CLASS["vocabulary"])

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    # Per arm, and never shared: a candidate one arm found is not handed to
    # another, and no arm is reset onto a common history after they diverge.
    steps_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    attempts_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    history_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    pending_by_arm: dict[str, dict[str, Any]] = {n: {} for n in arms}
    events_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    stopped_at = None
    faults_by_arm: dict[str, int] = {n: 0 for n in arms}
    completed: list[int] = []

    for row in after:
        reason = budget.stop_reason()
        if reason:
            stopped_at = {"position": row["position"], "why": reason}
            break
        ctx = R.opp._unit_ctx(row["unit"])
        cell_by_arm: dict[str, Any] = {}
        try:
            for name in plan["arm_order"]:
                cell = R.run_cell(arms[name], ctx, position=row["position"],
                                  ledgers=ledgers, guard=guard,
                                  machinery=machinery, budget=budget,
                                  seed=None)
                cell["group"] = group
                cell_by_arm[name] = cell
                cells.append(cell)
        except R.PackageCeiling as exc:
            stopped_at = {"position": row["position"],
                          "why": "FIT_CEILING_REACHED", "detail": str(exc)}
            break
        except runner.RunFault as exc:
            stopped_at = {"position": row["position"], "why": "RUN_FAULT",
                          "detail": str(exc)[:300]}
            break

        completed.append(row["position"])
        for name in plan["arm_order"]:
            cell = cell_by_arm[name]
            history_by_arm[name].append({
                "position": row["position"], "unit": row["unit"],
                "deployed_label": cell.get("deployed_label"),
                "deployed_scope": cell.get("deployed_scope"),
                "support_reading": cell.get("support_reading"),
                "delayed_reading": cell.get("delayed_reading"),
                "delayed_gate": cell.get("delayed_gate"),
            })

            # The verification event, identical in both supply arms.
            if name not in SUPPLY_ARMS:
                continue
            deployed = cell.get("deployed_label")
            if (deployed in pending_by_arm[name]
                    and bool((cell.get("delayed_gate") or {}).get("passes"))):
                entry = pending_by_arm[name].pop(deployed)
                events_by_arm[name].append(_verification_event(
                    arms[name], entry=entry, position=row["position"],
                    cell=cell, writes_back=name in UPDATING_ARMS))

        for name in plan["arm_order"]:
            if name not in SUPPLY_ARMS or budget.stop_reason():
                continue
            arm = arms[name]
            knowledge = _knowledge_state(arm, events_by_arm[name],
                                         row["position"])
            try:
                step = D2.revision_step(
                    arm, position=row["position"],
                    origin=int(row["unit"]["origin"]), proposer=proposer,
                    history=history_by_arm[name], attempts=attempts_by_arm[name],
                    ledgers=ledgers, budget=budget, metered=outer_metered,
                    pending=pending_by_arm[name], index=index,
                    by_position=by_position,
                    case_prefix="devauto3-%s-%s" % (group, arm.spec.slug))
            except R.PackageCeiling as exc:
                step = {"position": row["position"],
                        "outcome": "STOPPED_AT_THE_FIT_CEILING",
                        "why": str(exc)}
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                step = {"position": row["position"], "outcome": "STEP_FAULT",
                        "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
            step["arm"] = name
            step["group"] = group
            step["knowledge_state_at_this_call"] = knowledge
            steps_by_arm[name].append(step)
            # Carry the proposal-time context onto anything newly queued, so a
            # later write-back can say what the candidate was formed against.
            for label_, entry in pending_by_arm[name].items():
                entry.setdefault("parent_program_at_proposal",
                                 knowledge["parent_program_now"])
                entry.setdefault("card_body_at_proposal",
                                 knowledge["card_body_now"])
                entry.setdefault("proposed_when_an_update_had_landed",
                                 knowledge["this_call_reads_an_updated_card"])
            if step.get("outcome") in ("PROPOSER_FAULT", "STEP_FAULT"):
                faults_by_arm[name] += 1
            else:
                faults_by_arm[name] = 0
            if faults_by_arm[name] >= 3:
                stopped_at = {"position": row["position"], "arm": name,
                              "why": "PROPOSER_UNREACHABLE",
                              "detail": step.get("why")}
        if stopped_at:
            break

        (CHECKPOINTS / ("%s_progress.json" % label)).write_text(
            json.dumps(drafts._plain({
                "group": group, "completed_positions": completed,
                "cost": budget.to_dict(),
                "events": events_by_arm, "cells": cells}),
                ensure_ascii=False, default=str), encoding="utf-8")

    closed = {name: arms[name].draft_ledger.close_unreencountered()
              for name in arms}
    # First later use of a written-back program, filled from the cells.
    for name, events in events_by_arm.items():
        for event in events:
            later = [c for c in cells
                     if c["arm"] == name
                     and int(c["position"]) > int(event["verified_at_position"])
                     and c.get("deployed_label") == event["program"]]
            event["first_later_use"] = (
                {"position": later[0]["position"],
                 "deployed_via": later[0].get("deployed_via")}
                if later else None)

    common = sorted(set.intersection(*[
        {c["position"] for c in cells if c["arm"] == n} for n in arms]))
    scored = {n: [c for c in cells if c["arm"] == n and c["position"] in common]
              for n in arms}

    return {
        "stage": "DEV_AUTO3_THREE_ARMS",
        "package": ("DEV-AUTO-3: candidate supply versus Skill update, "
                    "three arms, development level"),
        "group": group,
        "registered_plan": {"seed": plan["seed"],
                            "arm_order": list(plan["arm_order"]),
                            "registered_before_the_run": True,
                            "what_the_seed_does": (
                                "this package's own RNG only; the model "
                                "service's sampling is not controlled and no "
                                "claim is made that it is")},
        "arms": {
            ARM_A: "control: Fast local adaptation, no outer supply, no PATCH",
            ARM_B: "supply-only: outer proposal and supply open, no PATCH",
            ARM_C: "supply-and-update: both open",
        },
        "what_is_isolated": (
            "the PATCH of %s's frozen program body, and only that; this is not "
            "a measurement of every Memory mechanism" % K0_SKILL_ID),
        "symmetry": (
            "the verification event is recorded and the Draft is closed "
            "identically in both supply arms, so a candidate leaves the "
            "resupply pool at the same moment whether or not it is written "
            "back; the arms share no history, no candidate and no cache"),
        "what_this_is_not": [
            "not the formal A3/A5 main experiment",
            "development level only: every unit here is already exposed",
            "no sealed data, no TARGET_HELD_IN unit and no +144 face was read",
            "two groups is a repeat-run check, not independent-data "
            "generalisation, and no significance is claimed",
        ],
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "smoke": preflight,
        "transport": {k: v for k, v in transport.items() if k != "api_key"},
        "admission_rule_in_force": bounded.BOUNDED_POLICY.to_dict(),
        "acceptance_rule": {"frame": "C", "rule": D2.FRAME_C_RULE,
                            "queue_order": D2.CANDIDATE_ORDER},
        "geometry": {
            "ordering": R.ORDERING, "source_arm": R.SOURCE_ARM,
            "units_planned": [r["position"] for r in after],
            "units_completed_by_every_arm": common,
            "excluded_units": [{"position": r["position"], "why": r["why"]}
                               for r in excluded],
            "k0_snapshot_source": snapshot_source,
            "ancestor_card_body": ANCESTOR_BODY,
        },
        "stopped_at": stopped_at,
        "per_arm": {n: R._arm_summary(rows) for n, rows in scored.items()},
        "cells": cells,
        "revision_steps": steps_by_arm,
        "attempt_log": attempts_by_arm,
        "verification_events": events_by_arm,
        "pending_never_verified": {
            n: {k: {kk: vv for kk, vv in v.items() if kk != "steps"}
                for k, v in pending_by_arm[n].items()} for n in arms},
        "drafts_closed_at_the_end": closed,
        "draft_ledgers": {n: arms[n].draft_ledger.to_dict() for n in arms},
        "actual_cost": budget.to_dict(),
        "guard": guard.to_dict(),
        "boundary": {
            "evaluation_face_reads": 0, "sealed_reads": 0,
            "target_held_in_reads": 0, "risk_constants_touched": 0,
            "scope_grid_touched": 0, "consumer_touched": 0,
            "known_winner_supplied_to_the_model": 0,
            "caches_shared_between_arms": 0,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or [])
    group = argv[0] if argv else "g1"
    limit = int(argv[1]) if len(argv) > 1 else None
    result = build(group=group, limit=limit)
    out = ART / ("dev_auto3_three_arms__%s.json" % group)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts._plain(result), ensure_ascii=False,
                              indent=1, default=str), encoding="utf-8")
    cost = result.get("actual_cost") or {}
    print("%s %s | fits %s/%s | llm %s/%s | %ss"
          % (result.get("status") or "OK", group,
             cost.get("physical_consumer_fits"), cost.get("fit_ceiling"),
             cost.get("llm_calls"), cost.get("llm_ceiling"),
             cost.get("wall_seconds")))
    for name, summary in (result.get("per_arm") or {}).items():
        face = summary["delayed_face"]
        print("  %-20s delayed=%-10s treated=%-4s harmed=%-3s gate=%s"
              % (name, face["mean_aggregate_gain_over_the_served_population"],
                 face["total_treated_series"], face["total_harmed_series"],
                 face.get("authoritative_gate_passes")))
    for name, events in (result.get("verification_events") or {}).items():
        if events:
            print("  %-20s verification events: %d (written back: %d)"
                  % (name, len(events),
                     sum(1 for e in events
                         if (e.get("write_back") or {}).get("applied"))))
    print("  wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
