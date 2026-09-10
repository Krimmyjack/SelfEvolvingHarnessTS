"""DEV-SEQ-1: the decision unit is one sequence, and shared knowledge is what
Slow updates at the batch boundary.

The loop this package restores
------------------------------
    each sequence's own Context
    -> the same Fast generates and executes a program for each of them
       separately
    -> legal feedback arrives and is recorded decision by decision
    -> similar failures are grouped, with their matched successes kept
    -> Slow proposes an update to the shared knowledge
    -> the update is validated on feedback that did not build it
    -> the next batch of Fast reads the updated knowledge, per sequence

Every earlier package in this line ran a different loop: one request built from
the block's first eval series, one program broadcast to twenty served series,
one card per unit.  Nothing below is a re-run of those; the readings are not
comparable to them and are not compared.

The three segments, registered before the run
---------------------------------------------
==================  ==========================  ==========================
segment             units (course positions)    what it decides
==================  ==========================  ==========================
formation           7, 8, 9  (origins 1176,     per-sequence Fast, real
                    1416, 1656)                 Support and delayed feedback
update boundary     built from 7 and 8 only;    grouping, real Slow, the
                    validated on 9              three registered lines
follow-up           10, 11   (origins 2136,     knowledge-frozen vs adopted-
                    2376) x 2 arms              update, same formation state
==================  ==========================  ==========================

The windows do not overlap: construction reads 1176/1416 and their delayed
faces 1224/1464, the withheld window is 1656/1704, and the follow-up is
2136/2184 and 2376/2424.  All five units are already-exposed development
units of the frozen course roster; ``TARGET_HELD_IN`` (positions 18-22), the
sealed data and the +144 evaluation face are never read.  The decision
population is the first ten eval series of the block in roster order -- a rule
fixed in ``per_sequence.DECISION_POPULATION_SIZE``, applied identically to
every unit and arm, and never chosen on a gain.

The two follow-up arms
----------------------
Both start from the **same formation state**: the same knowledge snapshot, the
same per-sequence histories, the same tools, the same model, the same probe
and LLM allowances, fresh sessions.  They differ in exactly one readable
thing -- whether the qualifying guidance card is in the library.  The
histories are identical in both arms, so they cannot produce a difference; the
card is the only thing that can.

If no qualifying update forms, that is the result.  No card is written by hand,
no extra run is added to find a positive number, and the follow-up segment then
measures nothing about knowledge -- which is recorded as ``NO_UPDATE_TREATMENT``
rather than dressed up.

What this is not
----------------
Development level.  This is a mechanism check about whether the design is
executed correctly and whether one real knowledge update helps the next batch.
It is not a generalisation claim, not an A5 result, and two follow-up units
with n=10 sequences is not a significance test.

Run:  python -m evaluation.main_protocol_p4.run_dev_seq1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq1_knowledge as know
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as m_r0k

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_seq1"

FORMATION_POSITIONS = (7, 8, 9)
#: The units the Slow input is built from.  Position 9 is deliberately absent.
CONSTRUCTION_POSITIONS = (7, 8)
WITHHELD_POSITION = 9
FOLLOWUP_POSITIONS = (10, 11)

ARM_FROZEN = "knowledge-frozen"
ARM_UPDATED = "adopted-update"
FOLLOWUP_ARMS = (ARM_FROZEN, ARM_UPDATED)

DOMAIN = "devseq1"
ANCESTOR_PROGRAM = m_r0k.ANCESTOR_PROGRAM

MAX_FITS = 2000
MAX_LLM = 2000
MAX_WALL_SECONDS = 6 * 3600

#: A runaway guard, not a cap on a normal flow.  One Fast session is three
#: stages plus at most one schema retry each; twelve leaves that untouched and
#: still stops a session that loops.  What was actually spent is reported.
PER_SEQUENCE_LLM_CAP = 12

#: Phase fit allowances, registered before the run, added to whatever the
#: package has already spent when the phase opens.  A phase refuses *before* it
#: spends, so crossing one is a recorded stop rather than a partial reading.
PHASE_ALLOWANCE = {
    "formation": 800,
    "boundary": 200,
    "followup": 900,
    "reference": 100,
}

SEED = 20260907


class PackageCeiling(RuntimeError):
    """An authorised ceiling would be crossed.  Nothing is spent."""


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------

class Budget:
    """One ledger for the package, reported split by arm and by phase."""

    def __init__(self) -> None:
        self.started = time.time()
        self.ledgers = runner.Ledgers()
        self.fits_by_arm: dict[str, int] = {}
        self.llm_by_arm: dict[str, int] = {}
        self.phase: str | None = None
        self.phase_floor = 0
        self.phase_allowance = 0

    def fits(self) -> int:
        return sum(self.fits_by_arm.values())

    def llm(self) -> int:
        return sum(self.llm_by_arm.values())

    def elapsed(self) -> float:
        return time.time() - self.started

    def spend_fits(self, arm: str, n: int) -> None:
        if int(n):
            self.fits_by_arm[arm] = self.fits_by_arm.get(arm, 0) + int(n)

    def spend_llm(self, arm: str, n: int) -> None:
        if int(n):
            self.llm_by_arm[arm] = self.llm_by_arm.get(arm, 0) + int(n)

    def open_phase(self, phase: str, allowance: int | None = None) -> None:
        self.phase = str(phase)
        self.phase_floor = self.fits()
        self.phase_allowance = int(
            PHASE_ALLOWANCE[phase] if allowance is None else allowance)

    def room_for(self, fits: int) -> bool:
        if self.fits() + int(fits) > MAX_FITS:
            return False
        if self.phase is None:
            return True
        return (self.fits() + int(fits)
                <= self.phase_floor + self.phase_allowance)

    def require(self, fits: int) -> None:
        if not self.room_for(fits):
            raise PackageCeiling(
                "a new reading would cross the %s allowance (%d) or the "
                "package ceiling (%d)" % (self.phase, self.phase_allowance,
                                          MAX_FITS))

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
            "llm_cap_per_sequence_session": PER_SEQUENCE_LLM_CAP,
            "llm_cap_is_a_runaway_guard_not_a_flow_cap": True,
            "phase_allowances": dict(PHASE_ALLOWANCE),
            "wall_seconds": round(self.elapsed(), 1),
            "wall_ceiling": MAX_WALL_SECONDS,
            "counting": ("every physical Consumer fit, including those spent "
                         "inside a call that then raised; nothing carried over"),
        }


# ---------------------------------------------------------------------------
# an arm
# ---------------------------------------------------------------------------

class Arm:
    """One knowledge version, its per-sequence histories and its own cache."""

    def __init__(self, name: str, snapshot: Any, *, machinery: Mapping[str, Any],
                 backend_factory: Any, guard: Any) -> None:
        self.name = str(name)
        self.slug = DOMAIN
        self.snapshot = snapshot
        self.m = machinery
        self.backend_factory = backend_factory
        self.guard = guard
        self.replay_cache = runner.ReplayPredictionCache(self.name)
        self.history: dict[str, list[Any]] = {}
        self.episodes: list[Any] = []
        self.backends: list[Any] = []

    def capability_cards(self) -> list[str]:
        from SelfEvolvingHarnessTS.contracts.harness import SkillKind
        return sorted(str(skill.skill_id) for skill in self.snapshot.skills
                      if skill.skill_kind is SkillKind.CAPABILITY)

    def fresh_method(self, uid: str) -> tuple[Any, Any]:
        """A new session for one sequence: its own backend, core and method.

        Isolation is structural rather than a flag: a per-sequence method holds
        only that sequence's history and the arm's frozen snapshot, so no
        candidate pool, trace or local state can reach another sequence.
        """
        backend = self.backend_factory()
        self.backends.append(backend)
        target = self.m["agentic"].live_transport()
        core = self.m["TTHAAgentCore"](
            backend, self.m["LocalPublicToolGateway"](
                np.zeros(8, dtype=np.float64), task_kind="forecast"),
            model=target["model"], base_url=target["base_url"])
        method = self.m["TTHAMethod"](self.m["TTHAFastAgent"](core),
                                      self.snapshot,
                                      tuple(self.history.get(str(uid), ())))
        return method, backend


# ---------------------------------------------------------------------------
# one sequence's decision
# ---------------------------------------------------------------------------

def run_sequence(*, arm: Arm, ctx: Any, uid: str, readings: ps.UnitReadings,
                 machinery: Mapping[str, Any], budget: Budget, position: int,
                 phase: str) -> dict[str, Any]:
    """One sequence: its own Fast session, its own program, its own feedback."""
    started = time.time()
    uid = str(uid)
    origin = int(ctx.origin)
    delayed_origin = ctx.face_origin(ps.DELAYED)
    record: dict[str, Any] = {
        "phase": phase, "position": int(position), "unit": ctx.unit,
        "arm": arm.name, "series_uid": uid, "origin": origin,
        "delayed_origin": int(delayed_origin), "faults": [],
    }
    features = ps.sequence_features(machinery, ctx, uid, origin=origin)
    request = ps.sequence_request(machinery, ctx, uid, origin=origin,
                                  domain=arm.slug)
    values = {uid: np.asarray(ctx.at.values[uid], dtype=np.float64)}
    record["public_features"] = features
    record["observed_pattern_keys"] = sorted(request.observed_pattern_spec)
    record["bound_series_uid"] = str(request.series_uid)
    record["bound_values_length"] = int(np.asarray(request.values).size)
    record["history_episodes_in"] = len(arm.history.get(uid, ()))
    record["knowledge_version"] = arm.snapshot.runtime_bundle_sha
    record["capability_cards_at_decision"] = arm.capability_cards()

    method, backend = arm.fresh_method(uid)
    view = ps.SequenceView(readings, uid)
    llm_before = int(budget.ledgers.llm_total())
    arm.guard.open_cell()
    arm.guard.per_unit_arm_cap = PER_SEQUENCE_LLM_CAP
    fits_before = int(arm.replay_cache.physical_fits)
    result = None
    try:
        budget.require(runner.CACHE_FITS_PER_CELL)
        arm.guard.reserve(kind="fast",
                          where={"unit": ctx.unit, "arm": arm.name,
                                 "series_uid": uid})
        result = machinery["online_loop"].run_online_round(
            method, view, request, values,
            origin=origin,
            slow_agent=None, controller=None, store=None,
            card_builder=lambda _episode: {
                "pattern_id": "%s_%s" % (arm.slug, uid),
                "observable_signature": {"task_kind": "forecast"}},
            round_name="devseq1_%s_u%03d_%s" % (phase, position, uid),
            budget=ps.PROBE_RECEIPTS_PER_SEQUENCE,
            allow_slow=False, allow_group_slow=False,
            domain=arm.slug,
            period=int(ctx.config["period"]),
            fast_features=dict(features),
            allow_fast_skill=False,
        )
    except PackageCeiling:
        raise
    except runner.UnitFault as exc:
        record["faults"].append({"kind": type(exc).__name__,
                                 "why": str(exc)[:240]})
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        record["faults"].append({"kind": type(exc).__name__,
                                 "why": str(exc)[:300]})

    spent_llm = int(budget.ledgers.llm_total()) - llm_before
    budget.spend_llm(arm.name, spent_llm)
    record["llm_calls_this_sequence"] = spent_llm
    record["llm_requests_sent"] = int(getattr(backend, "calls", 0) or 0)

    if result is None:
        budget.spend_fits(arm.name,
                          int(arm.replay_cache.physical_fits) - fits_before)
        record.update({"deployed": None, "deployed_label": "identity",
                       "decision": "FAULT_ABSTAINED_TO_IDENTITY",
                       "support_gain": None, "delayed_gain": "UNKNOWN",
                       "probes": [], "wall_seconds": round(
                           time.time() - started, 2)})
        return record

    trace = getattr(method, "last_trace", None)
    steps = result._winner_steps
    record["deployed_label"] = ps.program_label(steps)
    record["deployed"] = result.winner_program
    record["winner_candidate_id"] = str(result._winner_candidate_id or "") or None
    record["retrieved_skill_ids"] = list(
        getattr(trace, "retrieved_skill_ids", ()) or ())
    record["candidate_ids"] = list(
        getattr(trace, "candidate_program_steps", {}) or {})
    record["candidate_programs"] = {
        str(key): ps.program_label(ps.normalise_steps(value))
        for key, value in (getattr(trace, "candidate_program_steps", {})
                           or {}).items()}
    record["memory_resolution_status"] = str(
        getattr(trace, "memory_resolution_status", "") or "")
    record["probes"] = [
        {k: v for k, v in dict(row).items()
         if k in ("candidate_id", "kind", "gain", "passed", "relation",
                  "admission", "consumer_fits")}
        for row in result.actual_probed_programs]
    record["probe_programs"] = [
        {"candidate_id": str(row.get("candidate_id")),
         "program": ps.program_label(
             ps.normalise_steps(row.get("program_steps") or ())),
         "gain": row.get("gain")}
        for row in result.actual_probed_programs]
    record["risk_refusals"] = int(result.risk_refusal_count)
    winner_id = str(result._winner_candidate_id or "")
    from_card = bool(machinery["online_loop"].source_skill_of_candidate(winner_id))
    record["deployed_via"] = ("identity" if not steps
                              else "recalled_skill" if from_card
                              else "searched_this_sequence")
    record["decision"] = ("DEPLOYED" if steps else "ABSTAINED_TO_IDENTITY")

    support = None
    if steps:
        for row in result.actual_probed_programs:
            if (str(row.get("candidate_id")) == winner_id
                    and row.get("gain") is not None):
                support = float(row["gain"])
    record["support_gain"] = (round(support, 6) if support is not None
                              else (0.0 if not steps else None))
    record["support_credit_basis"] = (
        "this sequence's own gain at its own origin, with only this sequence "
        "treated; identity is 0.0 because the raw pipeline it takes is "
        "bit-identical to Static, not because a zero was filled in")

    # legal feedback: the delayed face of this sequence's own decision
    delayed_gain: Any = "UNKNOWN"
    if steps:
        try:
            machinery["online_loop"].open_delayed(
                result, view, delayed_origin=delayed_origin, store=None,
                scope_resolver=None, delayed_authorizer=None)
            delayed_gain = (round(float(result.delayed_utility), 6)
                            if result.delayed_utility is not None
                            else "UNKNOWN")
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            record["faults"].append({"kind": "UnitFault",
                                     "why": "open_delayed: %s" % str(exc)[:200]})
        event = result._fast_skill_event or {}
        if str(event.get("stage") or "") == "revocation_pending_no_store":
            record["would_have_revoked"] = {
                "skill_id": event.get("skill_id"),
                "delayed_gain": event.get("delayed_gain"),
                "not_acted_on_here": (
                    "a single sequence's delayed loss does not revoke shared "
                    "knowledge mid-batch; it is evidence for the boundary"),
            }
    else:
        delayed_gain = 0.0
    record["delayed_gain"] = delayed_gain
    record["delayed_credit_basis"] = (
        "this sequence's own delayed reading of the program it actually ran; "
        "never the group's mean, and UNKNOWN when it could not be read")

    # The Episodes as the lifecycle left them.  ``open_delayed`` writes the
    # delayed face by *replacing* the Episode in the method's list, so the
    # objects still held in ``result._episodes`` are the pre-delayed ones; the
    # method's list is the one that carries the feedback that just arrived.
    written = {str(episode_id) for episode_id in result.episode_ids}
    final = [ep for ep in method.experience_episodes
             if str(getattr(ep, "episode_id", "")) in written]
    record["episodes"] = [str(ep.episode_id) for ep in final]
    for episode in final:
        arm.history.setdefault(uid, []).append(episode)
        arm.episodes.append(episode)
    record["episode_delayed_evaluated"] = {
        str(ep.episode_id): bool(
            (getattr(ep, "delayed_response", {}) or {}).get("evaluated"))
        for ep in final}
    record["episode_local_status"] = {
        str(ep.episode_id): str(getattr(ep, "local_status", "") or "")
        for ep in final}

    spent_fits = int(arm.replay_cache.physical_fits) - fits_before
    budget.spend_fits(arm.name, spent_fits)
    record["consumer_fits_this_sequence"] = spent_fits
    record["verifier_calls_so_far_this_unit"] = readings.verifier_calls
    record["wall_seconds"] = round(time.time() - started, 2)
    return record


# ---------------------------------------------------------------------------
# one (unit, arm): every sequence, then the executed policy
# ---------------------------------------------------------------------------

def run_unit(*, arm: Arm, ctx: Any, position: int, phase: str,
             machinery: Mapping[str, Any], budget: Budget) -> dict[str, Any]:
    started = time.time()
    population = ps.decision_uids(ctx)
    readings = ps.UnitReadings(cache=arm.replay_cache, ctx=ctx,
                               executor=ctx.executor, guard=budget.require)
    rows: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None
    for uid in population:
        try:
            rows.append(run_sequence(arm=arm, ctx=ctx, uid=uid,
                                     readings=readings, machinery=machinery,
                                     budget=budget, position=position,
                                     phase=phase))
        except PackageCeiling as exc:
            stopped = {"series_uid": uid, "why": "FIT_CEILING_REACHED",
                       "detail": str(exc)[:200]}
            break
        except runner.RunFault as exc:
            stopped = {"series_uid": uid, "why": "ORDERING_LLM_CAP",
                       "detail": str(exc)[:200]}
            break
    decided = [row["series_uid"] for row in rows]
    assignment = {row["series_uid"]: ps.normalise_steps(row.get("deployed") or ())
                  for row in rows}
    fits_before = int(arm.replay_cache.physical_fits)
    support_reading, _f = readings.compose(assignment, ctx.origin, decided)
    delayed_reading, _f2 = readings.compose(assignment,
                                            ctx.face_origin(ps.DELAYED), decided)
    budget.spend_fits(arm.name,
                      int(arm.replay_cache.physical_fits) - fits_before)
    gate = None
    if delayed_reading.get("aggregate_gain") is not None:
        gate = runner.authoritative_gate({
            "treated": delayed_reading["treated"],
            "aggregate_gain": delayed_reading["aggregate_gain"],
            "harmed_fraction": delayed_reading["harmed_fraction"],
            "max_single_series_harm": delayed_reading["max_single_series_harm"],
        })
    return {
        "phase": phase, "position": int(position), "unit": ctx.unit,
        "arm": arm.name, "origin": int(ctx.origin),
        "decision_population": population,
        "decided": decided,
        "stopped_at": stopped,
        "sequences": rows,
        "assignment": {uid: ps.program_label_with_params(steps)
                       for uid, steps in assignment.items()},
        "assignment_by_operator_only": {uid: ps.program_label(steps)
                                        for uid, steps in assignment.items()},
        "distinct_programs_deployed": support_reading["distinct_programs"],
        "deployed_count": sum(1 for steps in assignment.values() if steps),
        "identity_count": sum(1 for steps in assignment.values() if not steps),
        "support_reading": support_reading,
        "delayed_reading": delayed_reading,
        "unit_authoritative_gate": gate,
        "gate_note": ("recorded, not acted on: this package writes knowledge "
                      "only at the batch boundary, so no unit gate activates "
                      "anything"),
        "cache": arm.replay_cache.to_dict(),
        "wall_seconds": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# reference readings
# ---------------------------------------------------------------------------

def reference_readings(*, ctx: Any, population: Sequence[str],
                       cache: Any, budget: Budget) -> dict[str, Any]:
    """Raw, and one fixed preparation, on the same population and faces."""
    readings = ps.UnitReadings(cache=cache, ctx=ctx, executor=ctx.executor,
                               guard=budget.require)
    out: dict[str, Any] = {}
    fits_before = int(cache.physical_fits)
    raw_support, _f = readings.compose({}, ctx.origin, population)
    raw_delayed, _f2 = readings.compose({}, ctx.face_origin(ps.DELAYED),
                                        population)
    out["raw"] = {"support": raw_support, "delayed": raw_delayed,
                  "what_it_is": ("no series prepared; every gain is exactly "
                                 "0.0 because this is Static by construction")}
    fixed = {uid: ANCESTOR_PROGRAM for uid in population}
    try:
        budget.require(2 * runner.CACHE_FITS_PER_CELL)
        support, _f3 = readings.compose(fixed, ctx.origin, population)
        delayed, _f4 = readings.compose(fixed, ctx.face_origin(ps.DELAYED),
                                        population)
        out["fixed_preparation"] = {
            "program": ps.program_label(ANCESTOR_PROGRAM),
            "support": support, "delayed": delayed,
            "what_it_is": ("the ancestor program given to every sequence in "
                           "the population -- the cohort-level answer this "
                           "package's per-sequence answer is read against")}
    except PackageCeiling as exc:
        out["fixed_preparation"] = {"unavailable": str(exc)[:200]}
    budget.spend_fits("reference", int(cache.physical_fits) - fits_before)
    return out


# ---------------------------------------------------------------------------
# the batch boundary
# ---------------------------------------------------------------------------

def knowledge_boundary(*, formation: Arm, units: Mapping[int, Any],
                       features: Mapping[int, Mapping[str, Mapping[str, Any]]],
                       machinery: Mapping[str, Any], budget: Budget,
                       outer_core: Any, snapshot: Any, store: Any,
                       controller: Any) -> dict[str, Any]:
    """Group, propose, compile, validate.  The only knowledge write there is."""
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent

    construction = [ep for ep in formation.episodes
                    if know.episode_origin(ep) in {
                        int(units[p].origin) for p in CONSTRUCTION_POSITIONS}]
    # DEFECT, found 2026-09-08: ``setdefault`` keys a visible-feature card by
    # series UID, so the first construction window wins and every u8 Episode
    # is described to Slow -- and scored by the validator -- with u7's
    # features.  Left in place so this package's receipts stay reproducible
    # from the code that wrote them; corrected for DEV-SEQ-2 in
    # ``dev_seq2_knowledge.FeatureBinding``, which keys a card by
    # (series_uid, origin) and reports a miss rather than borrowing.
    construction_features: dict[str, dict[str, Any]] = {}
    for position in CONSTRUCTION_POSITIONS:
        for uid, card in (features.get(position) or {}).items():
            construction_features.setdefault(str(uid), dict(card))
    grouping = know.group_failures(episodes=construction,
                                   features=construction_features)
    out: dict[str, Any] = {
        "construction_positions": list(CONSTRUCTION_POSITIONS),
        "withheld_position": WITHHELD_POSITION,
        "grouping": grouping,
        "episodes_in_the_slow_input": len(construction),
        "the_withheld_unit_is_not_in_the_slow_input": True,
    }
    if not grouping["comparability"]["comparable"]:
        out["outcome"] = "NOT_COMPARABLE_NO_GROUPING"
        return out
    if not grouping["groups"]:
        out["outcome"] = "NO_GROUP_FORMED"
        return out

    group = grouping["groups"][0]
    out["group_taken"] = group["group_id"]
    out["why_this_group"] = ("the largest group by member count, tie broken by "
                             "group id; not chosen on any gain")
    card = know.failure_card(
        group, pattern_id="devseq1-%s" % group["sign"].lower(),
        population_features=construction_features,
        vocabulary=contract.SCOPE_CLASS["vocabulary"])
    out["card"] = card
    catalog = know.build_catalog(controller=controller,
                                 parent=store.materialize(snapshot))
    out["surface_catalog"] = list(catalog)
    if not catalog:
        out["outcome"] = "NO_AUTHORISED_SURFACE"
        return out

    def slow_factory():
        """A fresh Slow session per attempt, on the same core and ledger."""
        return TTHASlowAgent(outer_core)

    llm_before = int(budget.ledgers.llm_total())
    proposal = know.propose_update(slow_factory=slow_factory, card=card,
                                   catalog=catalog, snapshot=snapshot)
    budget.spend_llm("slow", int(budget.ledgers.llm_total()) - llm_before)
    out["proposal"] = {k: v for k, v in proposal.items() if k != "manifest"}
    if not proposal.get("proposed"):
        out["outcome"] = "NO_PROPOSAL"
        return out

    applied = know.apply_update(controller=controller, store=store,
                                snapshot=snapshot,
                                manifest=proposal["manifest"])
    out["applied"] = {k: v for k, v in applied.items() if k != "snapshot"}
    if not applied.get("applied"):
        out["outcome"] = "PROPOSAL_DID_NOT_COMPILE"
        return out

    withheld_ctx = units[WITHHELD_POSITION]
    withheld_population = ps.decision_uids(withheld_ctx)
    withheld_features = dict(features.get(WITHHELD_POSITION) or {})
    reference_cache = runner.ReplayPredictionCache("boundary-validation")
    readings = ps.UnitReadings(cache=reference_cache, ctx=withheld_ctx,
                               executor=withheld_ctx.executor,
                               guard=budget.require)

    def withheld_reading(steps: Sequence[tuple], uids: Sequence[str],
                         face: str = "support") -> dict[str, float]:
        origin = (int(withheld_ctx.origin) if face == "support"
                  else int(withheld_ctx.face_origin(ps.DELAYED)))
        before = int(reference_cache.physical_fits)
        reading, _spent = readings.reading(steps, origin, uids)
        budget.spend_fits("boundary",
                          int(reference_cache.physical_fits) - before)
        return {str(uid): float(reading["per_series_gain"][str(uid)])
                for uid in uids}

    validation = know.validate_update(
        applicability=proposal["observable_applicability"],
        group=group,
        construction_features=construction_features,
        withheld_population=withheld_population,
        withheld_features=withheld_features,
        withheld_reading=withheld_reading)
    out["validation"] = validation
    out["outcome"] = ("UPDATE_ADOPTED" if validation["qualifies"]
                      else "UPDATE_REFUSED_BY_VALIDATION")
    if validation["qualifies"]:
        out["adopted_snapshot"] = applied["snapshot"]
        out["adopted_skill_id"] = proposal["skill_id"]
    return out


# ---------------------------------------------------------------------------
# the package
# ---------------------------------------------------------------------------

def build(*, run_id: str = "run1", formation_limit: int | None = None,
          followup_limit: int | None = None,
          population_size: int | None = None) -> dict[str, Any]:
    from evaluation.main_protocol_p4 import smoke_dev_seq1 as smoke

    if population_size is not None:
        ps.DECISION_POPULATION_SIZE = int(population_size)

    started = datetime.now(timezone(timedelta(hours=8)))
    budget = Budget()
    checkpoints = CHECKPOINTS / run_id
    checkpoints.mkdir(parents=True, exist_ok=True)

    preflight = smoke.run()
    if not preflight["passed"]:
        return {"stage": "DEV_SEQ1_PER_SEQUENCE", "status": "BLOCKED_BY_SMOKE",
                "smoke": preflight}

    np.random.seed(SEED % (2 ** 32))
    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population_rows, _excluded = R._population(doc)
    by_position = {row["position"]: row for row in population_rows}

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    snapshot_dir, snapshot_source = R._resolve_k0_snapshot(doc, state["k0"])
    if snapshot_dir is None:
        return {"stage": "DEV_SEQ1_PER_SEQUENCE", "status": "BLOCKED",
                "why": "no readable K0 snapshot"}
    start_snapshot = machinery["compile_snapshot"](snapshot_dir,
                                                   verify_lock=False)

    guard = runner.BudgetGuard(ordering_cap=MAX_LLM,
                               per_unit_arm_cap=PER_SEQUENCE_LLM_CAP,
                               ledgers=budget.ledgers)
    created_backends: list[Any] = []

    def backend_factory():
        inner = machinery["agentic"]._default_backend_factory(
            PER_SEQUENCE_LLM_CAP)
        created_backends.append(inner)
        return runner._MeteredFastBackend(inner, guard=guard, billable=True)

    outer_inner = machinery["agentic"]._default_backend_factory(64)
    created_backends.append(outer_inner)
    outer_metered = runner._MeteredOuterBackend(outer_inner, guard=guard,
                                                billable=True)
    transport = machinery["agentic"].live_transport()
    outer_core = machinery["TTHAAgentCore"](
        outer_metered, machinery["LocalPublicToolGateway"](
            np.zeros(8, dtype=np.float64), task_kind="forecast"),
        model=transport["model"], base_url=transport["base_url"])

    store = machinery["SnapshotStore"](
        base.ROOT / ".dev_seq1_runs" / run_id / "boundary_store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    store.materialize(start_snapshot)

    formation_positions = list(FORMATION_POSITIONS)
    if formation_limit is not None:
        formation_positions = formation_positions[: int(formation_limit)]
    followup_positions = list(FOLLOWUP_POSITIONS)
    if followup_limit is not None:
        followup_positions = followup_positions[: int(followup_limit)]

    units: dict[int, Any] = {}
    features: dict[int, dict[str, dict[str, Any]]] = {}
    for position in sorted(set(FORMATION_POSITIONS) | set(FOLLOWUP_POSITIONS)):
        ctx = R.opp._unit_ctx(by_position[position]["unit"])
        units[position] = ctx
        features[position] = {
            uid: ps.sequence_features(machinery, ctx, uid, origin=ctx.origin)
            for uid in ps.decision_uids(ctx)}

    # ---- formation -------------------------------------------------------
    formation = Arm("formation", start_snapshot, machinery=machinery,
                    backend_factory=backend_factory, guard=guard)
    budget.open_phase("formation")
    formation_units: list[dict[str, Any]] = []
    stopped_at: dict[str, Any] | None = None
    for position in formation_positions:
        if budget.stop_reason():
            stopped_at = {"phase": "formation", "position": position,
                          "why": budget.stop_reason()}
            break
        unit = run_unit(arm=formation, ctx=units[position], position=position,
                        phase="formation", machinery=machinery, budget=budget)
        formation_units.append(unit)
        (checkpoints / ("formation_u%03d.json" % position)).write_text(
            json.dumps(drafts._plain({"unit": unit,
                                      "cost": budget.to_dict()}),
                       ensure_ascii=False, default=str), encoding="utf-8")
        if unit["stopped_at"]:
            stopped_at = {"phase": "formation", "position": position,
                          **unit["stopped_at"]}
            break

    # ---- the batch boundary ---------------------------------------------
    budget.open_phase("boundary")
    boundary: dict[str, Any]
    if stopped_at:
        boundary = {"outcome": "NOT_REACHED", "why": stopped_at}
    else:
        try:
            boundary = knowledge_boundary(
                formation=formation, units=units, features=features,
                machinery=machinery, budget=budget, outer_core=outer_core,
                snapshot=start_snapshot, store=store, controller=controller)
        except PackageCeiling as exc:
            boundary = {"outcome": "STOPPED_AT_THE_FIT_CEILING",
                        "why": str(exc)[:240]}
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            boundary = {"outcome": "BOUNDARY_FAULT",
                        "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
    (checkpoints / "boundary.json").write_text(
        json.dumps(drafts._plain({k: v for k, v in boundary.items()
                                  if k != "adopted_snapshot"}),
                   ensure_ascii=False, default=str), encoding="utf-8")

    adopted = boundary.get("adopted_snapshot")
    treatment = "UPDATE_TREATMENT" if adopted is not None else "NO_UPDATE_TREATMENT"

    # ---- follow-up -------------------------------------------------------
    budget.open_phase("followup")
    arms: dict[str, Arm] = {}
    followup_units: list[dict[str, Any]] = []
    references: dict[str, Any] = {}
    if adopted is None:
        followup = {
            "ran": False,
            "why": ("no qualifying update formed, so the two arms would be "
                    "the same library and the comparison would measure "
                    "nothing about knowledge"),
            "no_card_was_written_by_hand": True,
        }
    else:
        for name, snapshot in ((ARM_FROZEN, start_snapshot),
                               (ARM_UPDATED, adopted)):
            arm = Arm(name, snapshot, machinery=machinery,
                      backend_factory=backend_factory, guard=guard)
            arm.history = {uid: list(rows)
                           for uid, rows in formation.history.items()}
            arms[name] = arm
        order = FOLLOWUP_ARMS
        for position in followup_positions:
            if budget.stop_reason():
                stopped_at = {"phase": "followup", "position": position,
                              "why": budget.stop_reason()}
                break
            reference_cache = runner.ReplayPredictionCache("reference")
            references[str(position)] = reference_readings(
                ctx=units[position],
                population=ps.decision_uids(units[position]),
                cache=reference_cache, budget=budget)
            for name in order:
                unit = run_unit(arm=arms[name], ctx=units[position],
                                position=position, phase="followup",
                                machinery=machinery, budget=budget)
                followup_units.append(unit)
                if unit["stopped_at"]:
                    stopped_at = {"phase": "followup", "position": position,
                                  "arm": name, **unit["stopped_at"]}
            (checkpoints / ("followup_u%03d.json" % position)).write_text(
                json.dumps(drafts._plain(
                    {"units": [u for u in followup_units
                               if u["position"] == position],
                     "reference": references[str(position)],
                     "cost": budget.to_dict()}),
                    ensure_ascii=False, default=str), encoding="utf-8")
            if stopped_at:
                break
        followup = {"ran": True, "arm_order": list(order)}

    finished = datetime.now(timezone(timedelta(hours=8)))
    return {
        "stage": "DEV_SEQ1_PER_SEQUENCE",
        "status": "COMPLETE" if stopped_at is None else "STOPPED",
        "run_id": run_id,
        "package": ("DEV-SEQ-1: per-sequence decision and execution, grouped "
                    "failures, one Slow knowledge update, development level"),
        "question": ("does the Harness observe, decide and execute per "
                     "sequence, and does one validated update to the shared "
                     "guidance change what the next batch of Fast does"),
        "registered_plan": {
            "formation_positions": formation_positions,
            "construction_positions": list(CONSTRUCTION_POSITIONS),
            "withheld_position": WITHHELD_POSITION,
            "followup_positions": followup_positions,
            "decision_population_rule": (
                "the first %d eval series of the block in roster order"
                % ps.DECISION_POPULATION_SIZE),
            "decision_population_size": ps.DECISION_POPULATION_SIZE,
            "arms": list(FOLLOWUP_ARMS),
            "registered_before_the_run": True,
            "not_chosen_on_a_gain": True,
        },
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "k0_snapshot": {"dir": str(snapshot_dir), "source": snapshot_source,
                        "runtime_bundle_sha": start_snapshot.runtime_bundle_sha,
                        "capability_cards": formation.capability_cards()},
        "transport": transport,
        "transport_actual_usage": {
            "requested_model": transport["model"],
            "returned_models": sorted({
                model for backend in created_backends
                for model in getattr(backend, "returned_models", set()) or ()}),
            "prompt_tokens": sum(int(getattr(b, "prompt_tokens", 0) or 0)
                                 for b in created_backends),
            "completion_tokens": sum(int(getattr(b, "completion_tokens", 0) or 0)
                                     for b in created_backends),
            "backends_created": len(created_backends),
        },
        "admission_policy": machinery["admission_policy"].active_policy().to_dict(),
        "admission_note": (
            "installed for continuity with the previous packages; with a "
            "one-series decision population strict and bounded_risk_v1 decide "
            "identically, because a single series cannot be both "
            "aggregate-positive and materially harmed"),
        "formation": {"units": formation_units,
                      "episodes_written": len(formation.episodes)},
        "boundary": {k: v for k, v in boundary.items()
                     if k != "adopted_snapshot"},
        "treatment": treatment,
        "followup": followup,
        "followup_units": followup_units,
        "reference_readings": references,
        "stopped_at": stopped_at,
        "cost": budget.to_dict(),
        "smoke": preflight,
        "what_this_is_not": (
            "development level: five already-exposed course units, ten "
            "sequences each, one course.  Not a generalisation claim, not an "
            "A5 result, and no significance is claimed.  The readings are on "
            "a per-sequence denominator and are not comparable to the "
            "cohort-broadcast numbers of DEV-AUTO-1/2/3 or DEV-KNOW-1."),
    }


def reference_pass(*, run_id: str = "run1") -> dict[str, Any]:
    """Raw and fixed-preparation reference readings for every registered unit.

    A separate 0-LLM pass rather than a second experiment.  The runner computes
    references inside the follow-up segment, which only opens when an update is
    adopted; this makes the same two reference readings available for every
    unit of the package whatever the boundary decided, so the per-sequence
    delivery always has something to be read against.  Its Consumer fits are
    reported here and are **not** folded into the package ledger of the run
    they annotate.
    """
    budget = Budget()
    budget.open_phase("reference", MAX_FITS)
    machinery = v1runner._machinery()
    doc = base.load(R.ORDERING)
    population_rows, _excluded = R._population(doc)
    by_position = {row["position"]: row for row in population_rows}
    out: dict[str, Any] = {}
    for position in sorted(set(FORMATION_POSITIONS) | set(FOLLOWUP_POSITIONS)):
        ctx = R.opp._unit_ctx(by_position[position]["unit"])
        cache = runner.ReplayPredictionCache("reference-u%03d" % position)
        out[str(position)] = reference_readings(
            ctx=ctx, population=ps.decision_uids(ctx), cache=cache,
            budget=budget)
    return {
        "stage": "DEV_SEQ1_REFERENCE_READINGS",
        "run_id": run_id,
        "what_it_is": ("the two reference deliveries for every registered "
                       "unit: raw (nothing prepared) and one fixed "
                       "preparation given to the whole decision population"),
        "fixed_program": ps.program_label(ANCESTOR_PROGRAM),
        "decision_population_size": ps.DECISION_POPULATION_SIZE,
        "readings": out,
        "cost": budget.to_dict(),
        "llm_calls": 0,
        "not_in_the_package_ledger": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--formation-limit", type=int, default=None)
    parser.add_argument("--followup-limit", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.references_only:
        reference = reference_pass(run_id=args.run_id)
        ART.mkdir(parents=True, exist_ok=True)
        path = Path(args.out) if args.out else (
            ART / ("dev_seq1_reference_readings__%s.json" % args.run_id))
        path.write_text(json.dumps(drafts._plain(reference),
                                   ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
        print(json.dumps({"stage": reference["stage"],
                          "cost": reference["cost"],
                          "artifact": str(path)},
                         ensure_ascii=False, indent=1, default=str))
        return 0

    result = build(run_id=args.run_id, formation_limit=args.formation_limit,
                   followup_limit=args.followup_limit,
                   population_size=args.population_size)
    ART.mkdir(parents=True, exist_ok=True)
    path = Path(args.out) if args.out else (
        ART / ("dev_seq1_per_sequence__%s.json" % args.run_id))
    path.write_text(json.dumps(drafts._plain(result), ensure_ascii=False,
                               indent=1, default=str), encoding="utf-8")
    print(json.dumps({
        "status": result.get("status"),
        "treatment": result.get("treatment"),
        "boundary": (result.get("boundary") or {}).get("outcome"),
        "cost": result.get("cost"),
        "stopped_at": result.get("stopped_at"),
        "artifact": str(path),
    }, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
