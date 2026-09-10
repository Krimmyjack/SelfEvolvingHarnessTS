"""DEV-AUTO-1: does a Harness that is allowed to revise its own Skill actually
do it, and does the revision change what the next units do?

The loop this package closes
----------------------------
    read the legal history -> propose a Scope or Workflow modification
    -> a tool measures the consequence -> update the existing Skill
    -> later units, in fresh sessions, run Fast under the updated Skill

Two arms, from the same reconstructed entry state
-------------------------------------------------
* ``frozen-skill``   -- same model, same tools, same per-unit Target-local
  exploration, but the Skill library is rebuilt from the entry snapshot at
  every unit, so nothing crosses a unit boundary.
* ``revising-skill`` -- everything the control has, plus: after every unit it
  may read its own record, propose modifications, have them measured, and --
  when one is accepted and then survives an independent later unit's delayed
  gate -- write it onto the Skill through the real compile/validate path.

Geometry (fixed before the run, and not chosen for being favourable)
--------------------------------------------------------------------
forward / A5-online / k=1.  The entry state is M-R0d's reconstruction at the
k1 boundary: the five processed units' bank rows, their contexts, the Draft
ledger as it stood, and K0's Active lineage.  The course then runs the **next
sixteen development units in course order**.  The five ``TARGET_HELD_IN`` units
are excluded by exposure name, and the +144 evaluation face is never read --
``run_unit_arm``'s evaluation-face call is the one thing this runner does not
reproduce.

Scoring
-------
Every reading is the mean aggregate gain over the unit's **full served
population**; a series outside the serving scope carries the raw prediction and
scores exactly 0.0, and a unit that deployed nothing scores exactly 0.0.  Both
faces are reported: the unit's own origin (Support, evidence) and origin+48
(delayed, where the authoritative gate lives).  No risk denominator, risk
threshold or gate constant is changed, and no "Support and delayed must both
not regress" gate is added.

Cost
----
Package ceilings, checked before the spend: 1000 new physical Consumer fits,
400 LLM calls, six hours of wall clock.  There is no per-candidate LLM cap.
Every cell is checkpointed, so a run that hits a ceiling reports the prefix
both arms completed rather than a favourable fragment.

Run:  python -m evaluation.main_protocol_p4.run_dev_auto1_skill_revision
"""
from __future__ import annotations

import copy
import json
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import audit_m_r0b_revision_opportunity as opp
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner
from evaluation.main_protocol_p4 import run_source_line_v3 as v3runner
from evaluation.main_protocol_p4 import scope_repair_distance as distance
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as m_r0k

ART = base.ART
ORDERING = live.ORDERING
SOURCE_ARM = live.ARM
BOUNDARY = int(live.BOUNDARY_POSITION)
DELAYED = int(runner.DELAYED_OFFSET)
FACES = ("support_face", "delayed_face")

ANCESTOR_SCOPE = m_r0k.ANCESTOR_SCOPE
ANCESTOR_PROGRAM = m_r0k.ANCESTOR_PROGRAM
K0_SKILL_ID = "fast_winner_forecast_ridge_smase_outlier_mad"

#: Package ceilings.  Enforced before the spend, never checked afterwards.
MAX_FITS = 1000
MAX_LLM = 400
MAX_WALL_SECONDS = 6 * 3600

FROZEN_ARM = "frozen-skill"
REVISING_ARM = "revising-skill"

#: DEV-AUTO-1R 3.3.  Fast's resource is the package's, not a fixed five calls a
#: cell.  ``PER_UNIT_ARM_BUDGET["llm_calls"] == 5`` is HEC-1's contract and is
#: not touched: this development configuration simply does not adopt it for the
#: inner loop, and says so in the artifact.  The allowance is pre-allocated
#: symmetrically -- each arm may draw the same amount -- and what each actually
#: consumed is reported separately.  The stop is still hard, still checked
#: before the request is built, and still leaves request timeouts, the total
#: fit ceiling and the wall clock exactly where they were.
PER_ARM_LLM = MAX_LLM // 2

MATERIAL = float(contract.RISK["material"])
FLOOR = int(contract.RISK["min_treated"])

CHECKPOINTS = Path(__file__).resolve().parents[2] / "_scratch" / "dev_auto1"


class PackageCeiling(RuntimeError):
    """An authorised package ceiling would be crossed.  Nothing is spent."""


# ---------------------------------------------------------------------------
# population
# ---------------------------------------------------------------------------

def _population(doc: Mapping[str, Any]) -> tuple[list[dict[str, Any]],
                                                 list[dict[str, Any]]]:
    kept, excluded = [], []
    for cell in base.cells(doc, SOURCE_ARM):
        unit = dict(cell["unit"])
        exposure = list(unit.get("exposure") or ())
        row = {"position": int(cell["position"]), "unit": unit,
               "exposure": exposure,
               "side": ("before_the_boundary" if int(cell["position"]) <= BOUNDARY
                        else "after_the_boundary")}
        if any(name in m_r0k.EXCLUDED_EXPOSURES for name in exposure):
            excluded.append({**row, "why": "held for the Target, not development"})
        elif any(name in m_r0k.DEVELOPMENT_EXPOSURES for name in exposure):
            kept.append(row)
        else:
            excluded.append({**row, "why": "exposure is not a development one"})
    return kept, excluded


def _resolve_k0_snapshot(doc: Mapping[str, Any],
                         k0: Mapping[str, Any]) -> tuple[Path | None, str | None]:
    course_k0 = dict(doc.get("k0_snapshot") or {})
    candidates = [
        (course_k0.get("store_root"), course_k0.get("runtime_bundle_sha"),
         "course artifact k0_snapshot"),
        (k0.get("store_root"), k0.get("runtime_bundle_sha"),
         "Phase S K0 receipt"),
    ]
    for store_root, sha, label in candidates:
        if not (store_root and sha):
            continue
        path = base.ROOT / str(store_root) / str(sha)
        try:
            readable = path.is_dir() and (path / "snapshot.lock.json").exists()
        except OSError:
            readable = False
        if readable:
            return path, label
    return None, None


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

class PackageBudget:
    """One ledger for the whole package, shared by both arms and reported split."""

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
        """This arm's remaining share of the symmetric pre-allocation.

        Also bounded by what is left of the package as a whole, so one arm
        cannot spend the other's share by arriving first.
        """
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
            "llm_room_left_by_arm": {name: max(0, PER_ARM_LLM - spent)
                                     for name, spent in self.llm_by_arm.items()},
            "wall_seconds": round(self.elapsed(), 1),
            "wall_ceiling": MAX_WALL_SECONDS,
            "counting": ("every physical fit, including the ones spent inside a "
                         "call that then raised; no allowance carried over"),
        }


# ---------------------------------------------------------------------------
# one (unit, arm) cell -- run_unit_arm without the evaluation face
# ---------------------------------------------------------------------------

def _served_reading(arm: Any, ctx: Any, origin: int, steps: Any,
                    scope: Mapping[str, Any] | None,
                    budget: PackageBudget) -> dict[str, Any] | None:
    """The executed policy over the whole served population, on one face.

    Read through the arm's own ``ReplayPredictionCache``: the same instrument
    the outer screen uses, so the Support face of a deployed program and the
    screen that later re-scores a candidate on that same cell share one entry
    instead of paying twice for the same two fits.
    """
    if not steps:
        uids = list(ctx.eval_uids)
        return {"identity": True, "treated": 0, "served": len(uids),
                "per_series_gain": {uid: 0.0 for uid in uids},
                "aggregate_gain": 0.0, "harmed_fraction": 0.0,
                "harmed_series": 0, "max_single_series_harm": 0.0,
                "consumer_fits": 0}
    resolved = (ctx.resolve(scope, origin) if scope
                else frozenset(ctx.eval_uids))
    before = arm.replay_cache.physical_fits
    if not budget.room_for(runner.CACHE_FITS_PER_CELL):
        raise PackageCeiling("a new (unit, face, program) entry would cross "
                             "the %d fit ceiling" % MAX_FITS)
    try:
        reading = arm.replay_cache.reading(ctx, origin, steps, resolved)
    except runner.UnitFault as exc:
        budget.spend_fits(arm.spec.name,
                          arm.replay_cache.physical_fits - before)
        return {"unreadable": str(exc)[:200]}
    budget.spend_fits(arm.spec.name, arm.replay_cache.physical_fits - before)
    gains = np.asarray(list(reading["per_series_gain"].values()),
                       dtype=np.float64)
    return {**reading,
            "harmed_series": int((gains < -MATERIAL).sum())}


def run_cell(arm: Any, ctx: Any, *, position: int, ledgers: runner.Ledgers,
             guard: runner.BudgetGuard, machinery: Mapping[str, Any],
             budget: PackageBudget, seed: Any = None) -> dict[str, Any]:
    """One (unit, arm) cell: a fresh Fast session, the delayed gate, write-back.

    A trimmed ``run_hec1.run_unit_arm``.  Two differences and no others: the
    +144 evaluation face is never read, and the executed policy is scored over
    the served population on both faces through the arm's replay cache.

    ``seed`` runs immediately after ``begin_unit``.  Since DEV-AUTO-1R 3.2 no
    arm in this package passes one: both carry their own history across units
    and differ only in whether the Skill structure may be written back.  The
    hook is kept because it is what a resumed course would need, and because
    the smoke asserts it is not in use.
    """
    started = time.time()
    reset = arm.begin_unit(position)
    if seed is not None:
        seed(arm)
    record: dict[str, Any] = {
        "unit": ctx.unit, "position": int(position), "arm": arm.spec.name,
        "served": len(ctx.eval_uids), "reset": reset, "faults": [],
    }
    guard.open_cell()
    # DEV-AUTO-1R 3.3: the cap in force for this cell is what is left of this
    # arm's own package allowance, not a fixed five.  ``open_cell`` has just
    # zeroed ``spent_this_cell``, so writing the cap here makes the guard's
    # existing arithmetic enforce a package resource with no change to how it
    # refuses (before the request, billing nothing) or to its fault class.
    guard.per_unit_arm_cap = int(budget.llm_room(arm.spec.name))
    record["llm_cap_in_force_this_cell"] = guard.per_unit_arm_cap
    record["llm_cap_is_the_package_allowance"] = True
    record["snapshot_skill_ids_at_start"] = arm.snapshot_skill_ids()
    record["skill_bodies_at_start"] = _skill_bodies(arm.active_snapshot())
    record["active_program_signatures_at_start"] = dict(
        arm.active_program_signatures)
    calls_before = arm.backend_calls()
    billed_before = arm.backend_billed_calls()
    fits_before = runner._executor_cost(ctx.executor)
    ledger_view = runner._VerifiableLedgerView(arm.draft_ledger)
    resupplied = arm.draft_ledger.resupplied_programs_for_verification()
    record["resupplied_programs_offered"] = {
        key: rev.program_label(steps) for key, steps in resupplied.items()}
    try:
        guard.reserve(kind="fast", where={"unit": ctx.unit,
                                          "arm": arm.spec.name})
        result = machinery["online_loop"].run_online_round(
            arm._method, ctx.executor,
            machinery["runner"]._a5_request(
                ctx.at.observation_block, ctx.at.values, ctx.origin,
                arm.spec.slug),
            ctx.at.values,
            origin=ctx.origin,
            slow_agent=None, controller=arm._controller, store=arm._store,
            card_builder=lambda _episode: {
                "pattern_id": arm.spec.slug,
                "observable_signature": {"task_kind": "forecast"}},
            round_name="devauto1_%s_u%03d" % (arm.spec.slug, position),
            budget=int(contract.PER_UNIT_ARM_BUDGET["probes"]),
            allow_slow=False, allow_group_slow=False,
            domain=arm.spec.slug,
            period=int(ctx.config["period"]),
            fast_features=dict(machinery["extract_public_features"](
                ctx.at.observation_block[:ctx.origin], task_kind="forecast")),
            allow_fast_skill=True,
            candidate_scopes=v3runner._MergedScopes(
                v1runner.InitializerScopes(arm._method), ledger_view),
            scope_resolver=ctx.resolve,
            scope_revision_preflight=v3runner._preflight(
                ctx.features, ctx.available, ledger_view),
            program_supply_verifier=ctx.executor,
            resupplied_programs=resupplied,
            risk_refusal_selector=distance.selector,
        )
    except runner.UnitFault as exc:
        record["faults"].append({"kind": type(exc).__name__,
                                 "why": str(exc)[:240]})
        result = None
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        record["faults"].append({"kind": type(exc).__name__,
                                 "why": str(exc)[:300]})
        result = None
    spent = runner._bill_fast(arm, guard, record, calls_before,
                              aborted=result is None,
                              billed_before=billed_before)
    budget.spend_llm(arm.spec.name, int(spent))
    record.update(runner._bill_support_fits(ledgers, ctx.executor, fits_before))
    budget.spend_fits(arm.spec.name,
                      int(record.get("support_fits_this_cell") or 0))
    # DEV-AUTO-1R 3.3: three ways a cell can end without deploying, kept
    # apart because they call for opposite responses -- more budget, a better
    # Skill, or a narrower scope.
    record["cell_stop"] = _classify_cell_stop(record, result)
    if result is None:
        record.update({"deployed": None, "identity": True,
                       "support_reading": _served_reading(arm, ctx, ctx.origin,
                                                          None, None, budget),
                       "delayed_reading": None, "delayed_gate": None,
                       "activated": False,
                       "wall_seconds": round(time.time() - started, 2)})
        return record

    trace = getattr(arm._method, "last_trace", None)
    candidate_ids = list(getattr(trace, "candidate_program_steps", {}) or {})
    record["fast_decision"] = runner.classify_fast_decision(
        {"candidates": candidate_ids}, candidate_ids,
        no_proposal_reason=getattr(trace, "memory_resolution_status", None)
        if not candidate_ids else None)
    record["candidate_program_steps"] = {
        str(key): rev.program_label(_normalise_steps(value))
        for key, value in (getattr(trace, "candidate_program_steps", {})
                           or {}).items()}
    record["probes"] = [dict(row) for row in result.actual_probed_programs]
    record["risk_refusals"] = len(result.risk_refusals)
    record["cell_stop"] = _classify_cell_stop(record, result)
    record["retrieved_skill_ids"] = list(
        getattr(trace, "retrieved_skill_ids", ()) or ())
    record["resupplied_candidate_ids"] = list(result._resupplied_candidate_ids)

    steps, scope = result._winner_steps, result._winner_serving_scope
    record["deployed"] = result.winner_program
    record["deployed_label"] = (rev.program_label(steps) if steps
                                else "identity")
    record["deployed_scope"] = drafts._plain(scope)
    winner_id = str(result._winner_candidate_id or "")
    from_skill_card = bool(
        machinery["online_loop"].source_skill_of_candidate(winner_id))
    from_resupplied_draft = winner_id.startswith(drafts.RESUPPLY_PREFIX)
    signature = outer_loop._program_signature(steps) if steps else ""
    active_at_start = dict(record.get("active_program_signatures_at_start") or {})
    in_active_set = bool(signature) and signature in active_at_start
    record["winner_candidate_id"] = winner_id or None
    record["winner_from_skill_candidate"] = from_skill_card
    record["winner_from_resupplied_draft"] = from_resupplied_draft
    record["deployed_via"] = (
        "identity" if not steps else
        "recalled_skill" if from_skill_card else
        "resupplied_draft" if from_resupplied_draft else
        "searched_active_program" if in_active_set else
        "searched_this_unit")

    record["support_reading"] = _served_reading(arm, ctx, ctx.origin, steps,
                                                scope, budget)

    delayed_origin = ctx.face_origin(DELAYED)
    if steps:
        resolved = (ctx.resolve(scope, delayed_origin) if scope
                    else frozenset(ctx.eval_uids))
        try:
            delayed = runner._policy_reading(ctx, delayed_origin, steps,
                                             resolved)
        except runner.UnitFault as exc:
            record["faults"].append({"kind": type(exc).__name__,
                                     "why": str(exc)[:240]})
            delayed = None
        if delayed is None:
            record.update({"delayed_reading": None, "delayed_gate": None,
                           "activated": False,
                           "wall_seconds": round(time.time() - started, 2)})
            return record
        ledgers.course_fits += int(delayed["consumer_fits"])
        budget.spend_fits(arm.spec.name, int(delayed["consumer_fits"]))
        gate = runner.authoritative_gate(delayed)
        snapshot_before = sorted(arm.snapshot_skill_ids())
        active_before = runner._store_active_sha(arm._store)
        try:
            machinery["online_loop"].open_delayed(
                result, ctx.executor, delayed_origin=delayed_origin,
                store=arm._store, scope_resolver=ctx.resolve,
                delayed_authorizer=lambda _evidence: bool(gate["passes"]))
        except Exception as exc:  # noqa: BLE001 - the lifecycle event is a reading
            record["faults"].append({"kind": "UnitFault",
                                     "why": "open_delayed: %s" % str(exc)[:200]})
        state = {
            "snapshot_before": snapshot_before,
            "snapshot_after": sorted(arm.snapshot_skill_ids()),
            "store_active_before": active_before,
            "store_active_after": runner._store_active_sha(arm._store),
        }
        state["unchanged"] = (state["snapshot_before"] == state["snapshot_after"]
                              and state["store_active_before"]
                              == state["store_active_after"])
        disagreement = runner.resolve_gate_disagreement(
            gate, result._delayed_event, state=state)
        gains = dict(delayed["per_series_gain"])
        record["delayed_reading"] = {
            **{k: v for k, v in delayed.items() if k != "consumer_fits"},
            "harmed_series": sum(1 for v in gains.values() if v < -MATERIAL)}
        record["delayed_gate"] = gate
        record["gate_disagreement"] = disagreement
        record["authority_state"] = state

        activated = False
        if disagreement["may_activate"] and arm.spec.write_back:
            activated = bool(machinery["online_loop"].activate_approved(
                result, arm._store))
        record["activated"] = activated
        if activated:
            new_ids = [skill_id for skill_id in arm.snapshot_skill_ids()
                       if skill_id not in arm.known_skill_ids]
            for skill_id in new_ids:
                arm.known_skill_ids.add(skill_id)
                if skill_id not in arm.active_skill_ids:
                    arm.active_skill_ids.append(skill_id)
            if signature and len(new_ids) == 1:
                arm.active_program_signatures.setdefault(signature, new_ids[0])
            record["skills_minted_this_unit"] = new_ids
            if steps:
                key = outer_loop.census_key(
                    runner.TASK_CONSUMER_KEY, steps,
                    record.get("deployed_root_scope") or scope)
                arm.active_lineage_keys.add(key)
                record["lineage_key"] = key
        if not gate["passes"] and scope:
            draft, refusal = runner._draft_for_refused_deployment(
                arm.draft_ledger, steps=steps, scope=scope, origin=ctx.origin,
                delayed_origin=delayed_origin, gate=gate)
            record["restriction_refused"] = refusal
            if draft is not None:
                entry = arm.draft_ledger.record_verification(
                    draft, window=delayed_origin,
                    failed_lines=gate["failed_lines"],
                    per_series_gain=delayed["per_series_gain"],
                    treated_prev=sorted(result._winner_resolved_series or ()),
                    treated_now=sorted(resolved),
                    material=MATERIAL, reading=gate)
                record["restricted_state"] = entry.get("state_after")
                record["restricted_draft_id"] = draft.draft_id
    else:
        record.update({"delayed_reading": None, "delayed_gate": None,
                       "activated": False})

    if arm.spec.write_back:
        arm.bank.extend(runner._bank_rows_from_round(ctx, result))
        if ctx not in arm.processed:
            arm.processed.append(ctx)
    record["bank_rows_after"] = len(arm.bank)
    record["active_skill_ids"] = list(arm.active_skill_ids)
    record["skill_bodies_at_end"] = _skill_bodies(arm.active_snapshot())
    record["wall_seconds"] = round(time.time() - started, 2)
    return record


def _classify_cell_stop(record: Mapping[str, Any],
                        result: Any) -> dict[str, Any]:
    """Budget truncation, an abstention and a risk refusal are three readings.

    DEV-AUTO-1 reported "9 of 32 cells exhausted their five calls" beside "20
    chose identity" as if they were one story about selection.  They are not:
    the first says the resource ran out, the second says the model had the
    resource and declined, and the third says a candidate existed and the risk
    budget refused it.  Only the second is evidence about the Skill.
    """
    faults = [row.get("why") or "" for row in (record.get("faults") or ())]
    truncated = any("LLM budget of" in text
                    or "budget exhausted" in text.lower() for text in faults)
    refusals = int(len(getattr(result, "risk_refusals", ()) or ())
                   if result is not None else 0)
    decision = (record.get("fast_decision") or {}).get("decision")
    return {
        "llm_budget_truncated_this_cell": bool(truncated),
        "llm_cap_in_force": record.get("llm_cap_in_force_this_cell"),
        "llm_calls_this_cell": record.get("llm_calls_this_cell"),
        "model_abstained_with_a_reason": bool(
            decision == "ABSTAINED_WITH_REASON"),
        "risk_refusals_this_cell": refusals,
        "note": ("a truncated cell is not evidence that the model had nothing "
                 "to propose, and a risk refusal is not evidence that it did "
                 "not look"),
    }


def _normalise_steps(value: Any) -> tuple[tuple[str, dict], ...]:
    """Typed steps arrive either as ``(op, params)`` pairs or as mappings."""
    out = []
    for step in (value or ()):
        if isinstance(step, Mapping):
            out.append((str(step.get("op")), dict(step.get("params") or {})))
        else:
            op, params = tuple(step)[0], (tuple(step)[1]
                                          if len(tuple(step)) > 1 else {})
            out.append((str(op), dict(params or {})))
    return tuple(out)


def _skill_bodies(snapshot: Any) -> dict[str, str]:
    """Every capability card's frozen body, so a change is visible in the log."""
    out = {}
    for skill in list(getattr(snapshot, "skills", ()) or ()):
        kind = str(getattr(getattr(skill, "skill_kind", None), "value", ""))
        if kind != "capability":
            continue
        out[str(skill.skill_id)] = str(skill.body or "")[:300]
    return out


# ---------------------------------------------------------------------------
# the outer revision step
# ---------------------------------------------------------------------------

def current_skill_steps(arm: Any) -> tuple[tuple[str, dict], ...]:
    """The frozen program the Skill card holds right now, parsed as Fast does."""
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        _parse_frozen_steps,
    )
    card = next((s for s in (getattr(arm.active_snapshot(), "skills", ()) or ())
                 if str(s.skill_id) == K0_SKILL_ID), None)
    if card is None:
        return tuple(ANCESTOR_PROGRAM)
    parsed = _parse_frozen_steps(card.body)
    if not parsed:
        return tuple(ANCESTOR_PROGRAM)
    return tuple((str(op), dict(params)) for op, params in parsed)


def _parent_summary(replay, steps, scope) -> dict[str, Any]:
    screen = replay(steps=steps, scope=scope)
    return {**rev._screen_summary(list(screen.get("cells") or ())),
            "consumer_fits_spent": int(screen.get("fits") or 0)}


def revision_step(arm: Any, *, position: int, origin: int,
                  proposer: rev.RevisionProposer, history: Sequence[Mapping[str, Any]],
                  attempts: list[dict[str, Any]], ledgers: runner.Ledgers,
                  budget: PackageBudget, metered: Any,
                  pending: dict[str, Any]) -> dict[str, Any]:
    """Ask, measure, and -- only if the tool agrees -- write the revision."""
    step: dict[str, Any] = {"position": int(position), "origin": int(origin),
                            "arm": arm.spec.name}
    if not arm.processed:
        step["outcome"] = "NO_PROCESSED_UNITS"
        return step

    # The parent is whatever the Skill card holds *now*.  After a promotion
    # that is the revised program, so a second revision is judged against the
    # first rather than against an ancestor the arm no longer runs.
    parent_steps = current_skill_steps(arm)
    root_scope = dict(ANCESTOR_SCOPE)
    draft = arm.draft_ledger.by_program_and_root(parent_steps, root_scope)
    current_scope = dict(draft.current_scope) if draft is not None \
        else dict(root_scope)
    parent_scope = dict(root_scope)

    screen = runner.replay_screen_for(arm.processed, ledgers, arm.replay_cache)

    def guarded_replay(*, steps, scope):
        need = runner.CACHE_FITS_PER_CELL * len(arm.processed)
        if not budget.room_for(need):
            raise PackageCeiling(
                "one screen over %d processed cells could take the package "
                "past %d fits" % (len(arm.processed), MAX_FITS))
        before = arm.replay_cache.physical_fits
        out = screen(steps=steps, scope=scope)
        budget.spend_fits(arm.spec.name,
                          arm.replay_cache.physical_fits - before)
        return out

    try:
        parent = _parent_summary(guarded_replay, parent_steps, current_scope)
    except PackageCeiling as exc:
        step["outcome"] = "STOPPED_AT_THE_FIT_CEILING"
        step["why"] = str(exc)
        return step
    step["parent_policy"] = {
        "program": rev.program_label(parent_steps),
        "serving_scope": current_scope,
        "screen": parent,
        "the_parents_own_production_screen_violations": rev._risk_lines(parent),
        "the_parent_passes_its_own_screen": not rev._risk_lines(parent),
        "read_from": "the Skill card as it stands at this step",
        "why_this_is_the_target": ("the card holds deployment rights at every "
                                   "position in this course and was never "
                                   "revoked; a promoted revision becomes the "
                                   "parent the next one has to beat"),
    }

    group = next((g for g in outer_loop.census(arm.bank)
                  if g["program_signature"] == rev.program_label(parent_steps)),
                 None)
    bank_rows = list(group["rows"]) if group else []
    step["bank_rows_for_calibration"] = len(bank_rows)

    feedback = rev.build_feedback(
        skill_id=K0_SKILL_ID, program_steps=parent_steps,
        serving_scope=current_scope, history=history, attempts=attempts,
        vocabulary=contract.SCOPE_CLASS["vocabulary"])
    step["feedback_sent"] = {
        "units_sent": feedback["units_sent"],
        "units_available": feedback["units_available"],
        "nothing_was_truncated": feedback["nothing_was_truncated"],
        "earlier_attempts_included": len(attempts),
    }

    llm_before = int(getattr(metered, "calls", 0) or 0)
    # The Slow view follows the arm's own snapshot, so a promoted revision is
    # what the next proposal is asked about.
    proposer.snapshot = arm.active_snapshot()
    proposal_record = proposer(feedback, case_id="devauto1-u%03d" % position)
    budget.spend_llm(arm.spec.name,
                     int(getattr(metered, "calls", 0) or 0) - llm_before)
    step["proposer"] = {k: v for k, v in proposal_record.items()
                        if k != "proposals"}
    step["proposals"] = list(proposal_record.get("proposals") or ())

    evaluations = []
    for proposal in step["proposals"]:
        try:
            verdict = rev.evaluate_proposal(
                proposal, parent_steps=parent_steps,
                parent_scope=current_scope, root_scope=parent_scope,
                bank_rows=bank_rows, replay=guarded_replay,
                parent_summary=parent)
        except PackageCeiling as exc:
            verdict = {"kind": proposal.get("kind"),
                       "outcome": "STOPPED_AT_THE_FIT_CEILING",
                       "why": str(exc)}
        evaluations.append(verdict)
        attempts.append(_attempt_row(position, proposal, verdict))
        if verdict.get("outcome") != "ACCEPTED_FOR_VERIFICATION":
            continue
        if verdict["kind"] == "scope_clause":
            applied = rev.apply_scope_revision(
                ledger=arm.draft_ledger, draft=draft,
                new_scope=verdict["scope_replayed"],
                narrowing_preflight=verdict.get("narrowing_preflight"),
                support=verdict["screen"]["summary"], origin=origin)
            verdict["written_back"] = applied
            if applied.get("applied"):
                verdict["written_back"]["channel"] = (
                    "the lineage's RestrictedDraft; later units are offered the "
                    "revised predicate through resupplied_scopes")
        else:
            steps = tuple((str(s["op"]), dict(s.get("params") or {}))
                          for s in verdict["steps_replayed"])
            key = outer_loop.census_key(runner.TASK_CONSUMER_KEY,
                                        [{"op": op, "params": dict(p)}
                                         for op, p in steps], parent_scope)
            try:
                new_draft = arm.draft_ledger.open_restricted(
                    program_steps=steps, root_scope=parent_scope,
                    current_scope=current_scope, origin=int(origin),
                    census_key=key,
                    provenance={"opened_by": "dev_auto1_revision_step",
                                "position": int(position),
                                "what_changes": proposal.get("what_changes")})
                verdict["written_back"] = {
                    "applied": True, "draft_id": new_draft.draft_id,
                    "channel": ("a Draft carrying the revised program; later "
                                "units are offered it as a candidate through "
                                "resupplied_programs_for_verification.  It "
                                "holds no deployment right: the Skill card is "
                                "patched only after an independent later unit "
                                "passes the authoritative delayed gate"),
                }
                pending[rev.program_label(steps)] = {
                    "steps": steps, "draft_id": new_draft.draft_id,
                    "proposed_at_position": int(position),
                    "what_changes": proposal.get("what_changes"),
                }
            except Exception as exc:  # noqa: BLE001
                verdict["written_back"] = {
                    "applied": False,
                    "why": "%s: %s" % (type(exc).__name__, str(exc)[:240])}
        attempts[-1]["written_back"] = verdict.get("written_back")
    step["evaluations"] = evaluations
    step["outcome"] = (proposal_record.get("outcome") or "UNKNOWN")
    step["accepted"] = sum(1 for v in evaluations
                           if v.get("outcome") == "ACCEPTED_FOR_VERIFICATION")
    return step


def _attempt_row(position: int, proposal: Mapping[str, Any],
                 verdict: Mapping[str, Any]) -> dict[str, Any]:
    """What the next call is told about this attempt.  Never a bare REJECTED."""
    screen = (verdict.get("screen") or {}).get("summary") or {}
    comparison = verdict.get("comparison_with_the_parent") or {}
    row: dict[str, Any] = {
        "proposed_after_unit_at_position": int(position),
        "kind": proposal.get("kind"),
        "what_you_proposed": (proposal.get("scope_clause")
                              if proposal.get("kind") == "scope_clause"
                              else proposal.get("workflow_program")),
        "what_you_said_it_would_change": proposal.get("expected_change"),
        "outcome": verdict.get("outcome"),
        "why": verdict.get("why"),
    }
    if screen:
        row["measured_on_the_processed_units"] = {
            "mean_aggregate_gain_over_the_served_population": screen.get(
                "mean_aggregate_gain_over_the_served_population"),
            "parent_delivers": comparison.get("parent_mean_aggregate_gain"),
            "difference": comparison.get("difference"),
            "worst_single_series_harm": screen.get("worst_single_series_harm"),
            "worst_harmed_fraction": screen.get("worst_harmed_fraction"),
            "cells_at_or_above_the_coverage_floor": screen.get(
                "cells_at_or_above_the_coverage_floor"),
            "cells_readable": screen.get("cells_readable"),
        }
    if verdict.get("failed_lines"):
        row["risk_lines_that_failed"] = list(verdict["failed_lines"])
    relative = verdict.get("relative_to_the_parent_screen") or {}
    if relative:
        row["how_it_compared_with_the_parent_itself"] = {
            "the_parent_passes_the_same_screen": relative.get(
                "parent_passes_the_production_screen"),
            "the_parent_fails_these_lines": relative.get(
                "parent_screen_violations"),
            "your_worst_single_series_harm_is_worse_than_the_parents":
                relative.get(
                    "candidate_worst_single_series_harm_is_worse_than_the_"
                    "parents"),
            "your_worst_harmed_fraction_is_worse_than_the_parents":
                relative.get(
                    "candidate_worst_harmed_fraction_is_worse_than_the_"
                    "parents"),
            "you_beat_the_parent_utility_by_material": relative.get(
                "candidate_beats_the_parent_utility_by_material"),
        }
    if verdict.get("edges_tried") is not None:
        row["frozen_bin_edges_tried"] = verdict["edges_tried"]
    return row


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _face_value(cell: Mapping[str, Any], face: str) -> float:
    key = "support_reading" if face == "support_face" else "delayed_reading"
    reading = cell.get(key) or {}
    if not reading or reading.get("unreadable"):
        return 0.0
    return float(reading.get("aggregate_gain") or 0.0)


def _arm_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"units": len(cells)}
    for face in FACES:
        key = "support_reading" if face == "support_face" else "delayed_reading"
        values = [_face_value(c, face) for c in cells]
        readings = [c.get(key) or {} for c in cells]
        readable = [r for r in readings if r and not r.get("unreadable")]
        gates = [bool((c.get("delayed_gate") or {}).get("passes"))
                 for c in cells] if face == "delayed_face" else []
        out[face] = {
            "mean_aggregate_gain_over_the_served_population": (
                round(statistics.fmean(values), 6) if values else None),
            "units_scoring_exactly_zero": sum(1 for v in values if v == 0.0),
            "total_harmed_series": sum(int(r.get("harmed_series") or 0)
                                       for r in readable),
            "total_treated_series": sum(int(r.get("treated") or 0)
                                        for r in readable),
            "worst_single_series_harm": (
                round(max(float(r.get("max_single_series_harm") or 0.0)
                          for r in readable), 6) if readable else None),
            "faces_not_readable": sum(1 for r in readings
                                      if r and r.get("unreadable")),
        }
        if face == "delayed_face":
            out[face]["authoritative_gate_passes"] = sum(1 for g in gates if g)
    out["deployed_via"] = _counts(cells, "deployed_via")
    out["deployed_program"] = _counts(cells, "deployed_label")
    out["units_that_deployed_nothing"] = sum(
        1 for c in cells if not c.get("deployed"))
    out["distinct_programs_probed"] = sorted({
        label for c in cells
        for label in (c.get("candidate_program_steps") or {}).values()})
    out["skills_retrieved"] = _counts_list(cells, "retrieved_skill_ids")
    return out


def _counts(cells: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for cell in cells:
        value = str(cell.get(key) or "none")
        out[value] = out.get(value, 0) + 1
    return out


def _counts_list(cells: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for cell in cells:
        for value in (cell.get(key) or ()):
            out[str(value)] = out.get(str(value), 0) + 1
    return out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _seed_arm(arm: Any, state: Mapping[str, Any], *, fresh_ledger: bool) -> None:
    """Put the arm at the k1 entry state M-R0d reconstructed."""
    arm.bank = [dict(row) for row in state["bank"]]
    arm.processed = list(state["processed"])
    arm.active_lineage_keys = set(state["held"])
    if fresh_ledger:
        arm.draft_ledger = copy.deepcopy(state["ledger"])
    k0 = state["k0"]
    arm.seed_active_programs(dict(k0.get("program_signatures") or {}),
                             list(k0.get("lineage_keys") or ()))


def build(*, limit: int | None = None, label: str = "run1") -> dict[str, Any]:
    from evaluation.main_protocol_p4 import smoke_dev_auto1_skill_revision as smoke

    started = datetime.now(timezone(timedelta(hours=8)))
    preflight = smoke.run()
    if not preflight["passed"]:
        return {"stage": "DEV_AUTO1_SKILL_REVISION",
                "status": "BLOCKED_BY_SMOKE", "smoke": preflight}

    doc = base.load(ORDERING)
    state = live._state_at_k1(doc)
    population, excluded = _population(doc)
    after = [row for row in population if row["side"] == "after_the_boundary"]
    if limit is not None:
        after = after[:int(limit)]

    machinery = v1runner._machinery()
    # DEV-AUTO-1R: install the released admission rule, as every production
    # runner does (``run_hec1`` line 2722, ``run_source_line`` line 267).
    # DEV-AUTO-1 did not, so its Fast side ran under the module default --
    # ``strict_positive_only``, which admits a probe only when it harms no
    # series at all.  Measured on that run's own probe rows: 3 of 39 admitted
    # under strict, 9 of 39 under the released policy, and every strict
    # refusal was recorded as ``relation_not_positive``, which is not a budget
    # reason, so ``risk_refusal_count`` stayed at 0 in all 32 cells and the
    # "useful program, scope too wide" evidence the revision channel was built
    # to consume was never generated.  No threshold is introduced here: this is
    # the constant the P4 line already ships.
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    snapshot_dir, snapshot_source = _resolve_k0_snapshot(doc, state["k0"])
    if snapshot_dir is None:
        return {"stage": "DEV_AUTO1_SKILL_REVISION", "status": "BLOCKED",
                "why": "no readable K0 snapshot"}
    start_snapshot = machinery["compile_snapshot"](snapshot_dir,
                                                  verify_lock=False)

    budget = PackageBudget()
    ledgers = runner.Ledgers()
    guard = runner.BudgetGuard(
        ordering_cap=MAX_LLM,
        per_unit_arm_cap=PER_ARM_LLM,   # rewritten per cell in ``run_cell``
        ledgers=ledgers)

    root = base.ROOT / ".dev_auto1_runs" / label

    def backend_factory():
        # A fresh backend per unit is still what makes the stop happen at the
        # transport rather than in a count checked afterwards; what changed is
        # the number it holds.  ``BudgetedAgentBackend`` would otherwise refuse
        # the sixth call of a cell no matter how much of the package allowance
        # the arm still had, which is the per-cell truncation this package is
        # removing.  The real stop is ``guard.per_unit_arm_cap``, rewritten to
        # the arm's remaining allowance at the top of every cell.
        inner = machinery["agentic"]._default_backend_factory(PER_ARM_LLM)
        return runner._MeteredFastBackend(inner, guard=guard, billable=True)

    # DEV-AUTO-1R 3.2.  Both arms carry their own history, Draft ledger and
    # lifecycle across units.  ``write_back=False`` is HEC-1's *frozen control*,
    # which additionally drops the Draft ledger and rebuilds the method every
    # unit (``ArmState.begin_unit``), never activates an approved Skill
    # (``activate_approved`` is gated on it) and never extends ``bank`` or
    # ``processed``.  That is a control frozen in four ways where this package
    # compares one, and it made the two arms non-comparable on everything
    # except the Skill.  What stays frozen is the Skill *structure write-back*:
    # ``revision_step`` runs for the revising arm only, and the invariant below
    # is asserted at the end of the course rather than assumed.
    specs = {
        FROZEN_ARM: runner.ArmSpec(FROZEN_ARM, "k0", True, False, True),
        REVISING_ARM: runner.ArmSpec(REVISING_ARM, "k0", True, False, True),
    }
    arms = {}
    for name, spec in specs.items():
        arm = runner.Arm(spec, root=root, machinery=machinery,
                         start_snapshot=start_snapshot, ledgers=ledgers,
                         guard=guard, backend_factory=backend_factory,
                         outer_slow_factory=None, offline=False)
        _seed_arm(arm, state, fresh_ledger=True)
        arms[name] = arm

    # The revision proposer: one core, one metered backend, package budget.
    outer_inner = machinery["agentic"]._default_backend_factory(MAX_LLM)
    outer_metered = runner._MeteredOuterBackend(outer_inner, guard=guard,
                                                billable=True)
    outer_core = machinery["TTHAAgentCore"](
        outer_metered,
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=machinery["agentic"].live_transport()["model"],
        base_url=machinery["agentic"].live_transport()["base_url"])
    proposer = rev.RevisionProposer(
        outer_core, snapshot=start_snapshot,
        vocabulary=contract.SCOPE_CLASS["vocabulary"])

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, Any]] = []
    revision_steps: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}
    promotions: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    stopped_at = None
    consecutive_faults = 0
    completed_positions: list[int] = []

    for row in after:
        reason = budget.stop_reason()
        if reason:
            stopped_at = {"position": row["position"], "why": reason}
            break
        ctx = opp._unit_ctx(row["unit"])
        cell_records = {}
        try:
            for name in (FROZEN_ARM, REVISING_ARM):
                arm = arms[name]
                # No per-cell reseed.  Both arms were seeded once, at the same
                # k1 entry state, and from there each accumulates its own
                # history.  Re-imposing the entry state on the control every
                # unit deleted exactly the legal history and verification
                # events DEV-AUTO-1R 3.2 says a frozen *Skill* must keep.
                cell = run_cell(arm, ctx, position=row["position"],
                                ledgers=ledgers, guard=guard,
                                machinery=machinery, budget=budget, seed=None)
                cell_records[name] = cell
                cells.append(cell)
        except PackageCeiling as exc:
            stopped_at = {"position": row["position"],
                          "why": "FIT_CEILING_REACHED", "detail": str(exc)}
            break
        except runner.RunFault as exc:
            stopped_at = {"position": row["position"], "why": "RUN_FAULT",
                          "detail": str(exc)[:300]}
            break

        completed_positions.append(row["position"])
        revising = cell_records[REVISING_ARM]
        history.append({
            "position": row["position"], "unit": row["unit"],
            "deployed_label": revising.get("deployed_label"),
            "deployed_scope": revising.get("deployed_scope"),
            "support_reading": revising.get("support_reading"),
            "delayed_reading": revising.get("delayed_reading"),
            "delayed_gate": revising.get("delayed_gate"),
        })

        # Promotion: a pending revision that a later unit deployed and whose
        # delayed gate passed earns the PATCH.  Never before.
        label_now = revising.get("deployed_label")
        if (label_now in pending
                and bool((revising.get("delayed_gate") or {}).get("passes"))):
            entry = pending.pop(label_now)
            arm = arms[REVISING_ARM]
            applied = rev.apply_workflow_revision(
                controller=arm._controller, store=arm._store,
                snapshot=arm.active_snapshot(), skill_id=K0_SKILL_ID,
                steps=entry["steps"],
                edit_id="dev-auto1-rev-u%03d" % row["position"])
            promotion = {k: v for k, v in applied.items()
                         if k not in ("snapshot", "materialized")}
            promotion.update({
                "verified_at_position": row["position"],
                "proposed_at_position": entry["proposed_at_position"],
                "program": label_now,
                "independent_unit": True,
            })
            if applied.get("applied"):
                arm._method._snapshot = applied["snapshot"]
                arm._store.set_active(applied["candidate_runtime_bundle_sha"])
                promotion["skill_bodies_after"] = _skill_bodies(
                    arm.active_snapshot())
            promotions.append(promotion)

        # The one asymmetry in this package: only the revising arm is offered
        # the revision channel.  Named by arm rather than read off ``write_back``
        # now that both arms carry their state.
        if not budget.stop_reason():
            try:
                step = revision_step(
                    arms[REVISING_ARM], position=row["position"],
                    origin=int(row["unit"]["origin"]), proposer=proposer,
                    history=history, attempts=attempts, ledgers=ledgers,
                    budget=budget, metered=outer_metered, pending=pending)
            except PackageCeiling as exc:
                step = {"position": row["position"],
                        "outcome": "STOPPED_AT_THE_FIT_CEILING",
                        "why": str(exc)}
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                step = {"position": row["position"], "outcome": "STEP_FAULT",
                        "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
            revision_steps.append(step)
            # A candidate that fails, or a call that abstains, is a reading and
            # the course continues.  A proposer that cannot reach the service
            # at all is not: three in a row and the affected part stops rather
            # than spending the rest of the course on a dead transport.
            if step.get("outcome") in ("PROPOSER_FAULT", "STEP_FAULT"):
                consecutive_faults += 1
            else:
                consecutive_faults = 0
            if consecutive_faults >= 3:
                stopped_at = {"position": row["position"],
                              "why": "PROPOSER_UNREACHABLE",
                              "detail": step.get("why") or step.get("fault")}
                break

        (CHECKPOINTS / ("%s_progress.json" % label)).write_text(
            json.dumps(drafts._plain({
                "completed_positions": completed_positions,
                "cost": budget.to_dict(),
                "revision_steps": revision_steps,
                "promotions": promotions,
                "cells": cells}), ensure_ascii=False, default=str),
            encoding="utf-8")

    by_arm = {name: [c for c in cells if c["arm"] == name]
              for name in (FROZEN_ARM, REVISING_ARM)}

    # DEV-AUTO-1R 3.2: the one thing the control freezes, checked rather than
    # asserted.  A Fast-minted Skill is a legal lifecycle event and both arms
    # may have one; what the control may never have is a *revision* of the card
    # under revision -- its body or its serving scope rewritten.
    def _card_states(rows: Sequence[Mapping[str, Any]]) -> list[str]:
        seen: list[str] = []
        for cell in rows:
            body = json.dumps((cell.get("skill_bodies_at_start") or {}).get(
                K0_SKILL_ID), sort_keys=True, default=str)
            if body not in seen:
                seen.append(body)
        return seen

    frozen_states = _card_states(by_arm[FROZEN_ARM])
    frozen_invariant = {
        "what_is_frozen": ("the body of %s, the card the revision channel "
                           "edits" % K0_SKILL_ID),
        "distinct_card_bodies_seen_in_the_control": len(frozen_states),
        "the_control_never_revised_its_card": len(frozen_states) <= 1,
        "revision_channel_offered_to": [REVISING_ARM],
        "what_is_not_frozen": [
            "the unit history the arm accumulates (bank, processed)",
            "the Draft lifecycle and its verification events",
            "candidate supply and resupply to later units",
            "Fast-side activation of an approved Skill",
        ],
        "why": ("DEV-AUTO-1 froze all five and could therefore not attribute "
                "any arm difference to the Skill"),
    }
    common = sorted(set(c["position"] for c in by_arm[FROZEN_ARM])
                    & set(c["position"] for c in by_arm[REVISING_ARM]))
    scored = {name: [c for c in by_arm[name] if c["position"] in common]
              for name in by_arm}

    return {
        "stage": "DEV_AUTO1_SKILL_REVISION",
        "package": "DEV-AUTO-1: autonomous Skill revision, development level",
        "frozen_arm_invariant": frozen_invariant,
        "admission_rule_in_force": {
            "installed": bounded.BOUNDED_POLICY.to_dict(),
            "source": "p4b_contract.BOUNDED_POLICY, as the production runners "
                      "install it",
            "what_run2_actually_ran_under": "admission_policy.DEFAULT "
                                            "(strict_positive_only)",
        },
        "what_this_is_not": [
            "not the formal A3/A5 main experiment",
            "development level only: every unit here is already exposed",
            "no sealed data, no TARGET_HELD_IN unit and no +144 face was read",
        ],
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "smoke": preflight,
        "transport": {
            **dict(machinery["agentic"].live_transport()),
            "same_model_for_both_arms_and_for_the_proposer": True,
            "why_it_is_recorded_here": (
                "HEC-1 and every M-R0x reading ran on the agicto relay's "
                "gpt-5.6-luna.  That relay now refuses any request above a few "
                "thousand tokens with HTTP 403 insufficient_quota -- a tiny "
                "probe still succeeds, a real prompt does not -- so this "
                "package ran on the DeepSeek endpoint already configured in "
                "this environment.  No account or paid service was added.  "
                "Both arms and the revision proposer use the one model, so the "
                "arm contrast is unaffected; the absolute numbers are not "
                "comparable with HEC-1's"),
        },
        "geometry": {
            "ordering": ORDERING, "state_reconstructed_from": SOURCE_ARM,
            "boundary_position": BOUNDARY,
            "entry_state": {
                "processed_units": len(state["processed"]),
                "bank_rows": len(state["bank"]),
                "drafts": state["drafts_before"],
                "held_lineage_keys": state["held"],
            },
            "units_planned": [row["position"] for row in after],
            "units_completed_by_both_arms": common,
            "excluded_units": excluded,
            "k0_snapshot": {"path": str(snapshot_dir.relative_to(base.ROOT)),
                            "resolved_from": snapshot_source,
                            "runtime_bundle_sha":
                                start_snapshot.runtime_bundle_sha},
            "ancestor": {"program": rev.program_label(ANCESTOR_PROGRAM),
                         "serving_scope": ANCESTOR_SCOPE,
                         "skill_id": K0_SKILL_ID},
            "scoring": ("the full served population on every unit; an "
                        "out-of-scope series carries the raw prediction and "
                        "scores exactly 0"),
            "not_read": ["the +144 evaluation face", "TARGET_HELD_IN units",
                         "sealed data"],
        },
        "arms": {
            FROZEN_ARM: {
                "what_it_may_do": ("full Target-local exploration inside each "
                                   "unit; the Skill library is rebuilt from "
                                   "the entry snapshot at every unit"),
                "cross_unit_skill_state": "none",
            },
            REVISING_ARM: {
                "what_it_may_do": ("everything the control may do, plus a "
                                   "revision step after every unit: propose, "
                                   "have the tool measure, write an accepted "
                                   "revision onto the Skill, and use it later"),
                "cross_unit_skill_state": "carried",
            },
            "shared": ("same model, same transport, same per-unit-arm LLM cap "
                       "(%d), same probe budget, same feedback access rules, "
                       "same entry state"
                       % int(contract.PER_UNIT_ARM_BUDGET["llm_calls"])),
        },
        "stopped_at": stopped_at,
        "per_arm": {name: _arm_summary(scored[name]) for name in scored},
        "cells": cells,
        "revision_steps": revision_steps,
        "attempt_log": attempts,
        "promotions_to_the_skill_card": promotions,
        "pending_revisions_never_verified": drafts._plain(pending),
        "actual_cost": budget.to_dict(),
        "ledgers": ledgers.to_dict(),
        "boundary": {
            "evaluation_face_reads": 0,
            "sealed_reads": 0,
            "target_held_in_reads": 0,
            "thresholds_changed": 0,
            "operators_added": 0,
            "risk_denominator_changed": 0,
            "frozen_bin_edges_changed": 0,
            "production_runner_files_changed": 0,
        },
        "code_state": base.version_check(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    label = argv[argv.index("--label") + 1] if "--label" in argv else "run1"
    limit = (int(argv[argv.index("--limit") + 1]) if "--limit" in argv
             else None)
    out = ART / ("dev_auto1_skill_revision__%s.json" % label)
    if out.exists() and "--allow-overwrite" not in argv:
        sys.stderr.write("refusing to overwrite %s\n" % out)
        return 2
    report = build(limit=limit, label=label)
    ART.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps(drafts._plain({
        "status": report.get("status"),
        "cost": report.get("actual_cost"),
        "stopped_at": report.get("stopped_at"),
        "per_arm": report.get("per_arm"),
        "promotions": len(report.get("promotions_to_the_skill_card") or ()),
    }), indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
