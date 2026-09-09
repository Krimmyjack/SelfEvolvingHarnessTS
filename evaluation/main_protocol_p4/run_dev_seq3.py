"""DEV-SEQ-3: does guidance that actually reaches Fast improve the next batch?

DEV-SEQ-2 answered a narrower question than it set out to: both groups' Slow
wrote a legal, compilable guidance card, and both cards then matched none of
the sequences they were written about, so neither arm ever saw one.  The block
was at *presentation*, and the utility question was never reached.

This package changes three things and nothing else about the design.

1.  The feature description handed to Slow now gives the formula and the
    window each feature is computed on, and says plainly that the feature
    window (the whole observed history), the serving window (its last 192
    points, the only thing the Consumer sees) and the region a program
    actually modifies are three different objects, only the first of which the
    vocabulary covers.  The unproven drift claims DEV-SEQ-2 asserted are gone.

2.  Every outcome record now carries **action facts**: how many points the
    chosen program moved on this series' own serving window, and how many it
    moved in the shared training material.  Both are replays of the
    evaluator's own ``_prepare`` against its own ``_linear_integrity``
    baseline and tolerance -- no refit, no Consumer time, no new schema.  The
    training count is a property of the shared training material and is
    labelled as such; a count is an action, never a causal contribution or a
    probability of harm; an unavailable count stays UNKNOWN.

3.  Fast sessions inside one batch run **in parallel** (default 4).  Batches
    stay serial and knowledge stays frozen within a batch.  Each sequence gets
    its own session, its own local state and its own budget guard; fits and
    shared-cache writes are serialised behind one lock; results are collected
    and applied in fixed roster order, so completion order cannot change any
    decision.

What is reused rather than re-paid
----------------------------------
The u7/u8 formation histories from DEV-SEQ-2's SOL run are loaded from their
checkpoints, Episodes and all.  That segment's cost was paid there and is
reported separately; this package does not call it free and does not present
itself as a fresh end-to-end run.  What is new here is the boundary (Slow sees
a corrected card over the same history) and everything after it.

Registered before the run
-------------------------
    formation      u7 / 1176, u8 / 1416   -- reused, 10 sequences, as before
    validation     u9 / 1656  (+48)       -- both arms
    follow-up      u10 / 2136, u11 / 2376 -- both arms
    arm population **the full 20 eval series** of each block, in roster order
    groups         g1 seed 20260908 arm order A->B
                   g2 seed 20260909 arm order B->A

The arm population changes from 6 to 20; the formation history stays at 10.
That is recorded, and DEV-SEQ-3 readings are **not** subtracted against
DEV-SEQ-2's.

Everything else is unchanged and unopened: no sealed material, no
TARGET_HELD_IN, no +144; Consumer, DSL, risk lines, scoring semantics and the
training/serving geometry are untouched; the model cannot edit its own
approval rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_seq2 as seq2
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as m_r0k

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_seq3"

FORMATION_POSITIONS = (7, 8)
VALIDATION_POSITION = 9
FOLLOWUP_POSITIONS = (10, 11)
ALL_POSITIONS = (*FORMATION_POSITIONS, VALIDATION_POSITION, *FOLLOWUP_POSITIONS)

ARM_A = seq2.ARM_A
ARM_B = seq2.ARM_B

#: Where DEV-SEQ-2's SOL formation checkpoints live, per group.
REUSED_FORMATION = {
    "g1": base.ROOT / "_scratch" / "dev_seq2" / "pkg" / "g1",
    "g2": base.ROOT / "_scratch" / "dev_seq2" / "pkg2" / "g2",
}

GROUPS = seq2.GROUPS
DOMAIN = "devseq3"
ANCESTOR_PROGRAM = m_r0k.ANCESTOR_PROGRAM

MAX_FITS = 2000
MAX_LLM = 2000
MAX_WALL_SECONDS = 6 * 3600

PER_SEQUENCE_LLM_CAP = 24
OUTER_LLM_CAP = 64

#: Formation history stays as DEV-SEQ-2 recorded it; the arms decide for the
#: full evaluation roster.  Both fixed before the run, neither chosen on a gain.
FORMATION_POPULATION = 10
ARM_POPULATION = 20

#: Fast sessions per batch.  Batches stay serial.
CONCURRENCY = 4

REQUIRED_MODEL = "gpt-5.6-sol"
MIN_GROUP = 2

RESUME_ONLY = False
EARLY_STOP = ""


def population_for(ctx: Any, phase: str) -> list[str]:
    size = FORMATION_POPULATION if phase == "formation" else ARM_POPULATION
    return ps.decision_uids(ctx, size=size)


# ---------------------------------------------------------------------------
# concurrency-safe shared state
# ---------------------------------------------------------------------------

class SharedState:
    """The three things a parallel batch shares, each behind its own lock.

    ``fits`` serialises every Consumer reading, so a cache miss is built once
    and the two physical fits it costs are never raced.  ``ledger`` serialises
    every counter move.  ``history`` serialises the Episode lists.  Nothing
    else is shared: each sequence has its own backend, core, method, view and
    budget guard.
    """

    def __init__(self) -> None:
        self.fits = threading.Lock()
        self.ledger = threading.Lock()
        self.history = threading.Lock()


class TransientRelayRetry:
    """Retry the relay's own transient refusals, and nothing else.

    The shared ``_RetryingTransport`` retries ``AgentTransportError`` only.
    This relay also refuses with a plain ``400`` carrying
    ``type: upstream_error`` and a message that literally asks the caller to
    retry ("分配账号异常，请重试"), and with a ``403`` carrying
    ``bad_response_status_code``.  Those are not request errors and not
    account decisions -- the same key answers the next call -- so they are
    ridden out here rather than allowed to fault a decision.

    Bounded and honest: at most ``attempts`` tries, short backoff, and each
    try goes through the budgeted backend underneath, so **every attempt is
    billed as the API call it is**.  A genuine account or permission signal is
    re-raised on sight and never retried.
    """

    def __init__(self, delegate: Any, *, attempts: int = 3,
                 backoff_seconds: float = 5.0) -> None:
        self.delegate = delegate
        self.attempts = int(attempts)
        self.backoff_seconds = float(backoff_seconds)
        self.relay_retries = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def complete(self, request: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return self.delegate.complete(request)
            except Exception as exc:  # noqa: BLE001 - classified, then re-raised
                detail = "%s: %s" % (type(exc).__name__, exc)
                if know._fault_kind(detail) != "TRANSPORT_TRANSIENT_FAULT":
                    raise
                last = exc
                self.relay_retries += 1
                if attempt + 1 < self.attempts:
                    time.sleep(self.backoff_seconds * (attempt + 1))
        raise last  # type: ignore[misc]


def transport_preflight(machinery: Mapping[str, Any]) -> dict[str, Any]:
    """DEV-SEQ-2's check, with the same bounded retry for a flaky relay."""
    row: dict[str, Any] = {}
    for attempt in range(3):
        row = seq2.transport_preflight(machinery)
        row["preflight_attempts"] = attempt + 1
        if row.get("usable"):
            return row
        if know._fault_kind(str(row.get("why") or "")) != "TRANSPORT_TRANSIENT_FAULT":
            return row
        time.sleep(5.0 * (attempt + 1))
    row["why_not_retried_further"] = (
        "three preflight attempts all met a transient relay refusal; this is "
        "reported as an instrument condition, not as a model result")
    return row


class SequenceGuard(runner.BudgetGuard):
    """One guard per sequence, sharing the package ledger under a lock.

    The per-cell counter is private, so two sequences cannot spend each
    other's allowance; the package ledger is shared, so the package cap still
    means the package.  Both ``reserve`` and ``spend`` hold the lock, so the
    check and the billing cannot interleave.
    """

    def __init__(self, *, ordering_cap: int, per_unit_arm_cap: int,
                 ledgers: Any, lock: threading.Lock) -> None:
        super().__init__(ordering_cap=int(ordering_cap),
                         per_unit_arm_cap=int(per_unit_arm_cap),
                         ledgers=ledgers)
        self._lock = lock

    def reserve(self, **kwargs: Any) -> None:
        with self._lock:
            super().reserve(**kwargs)

    def spend(self, **kwargs: Any) -> None:
        with self._lock:
            super().spend(**kwargs)


class LockedReadings:
    """``ps.UnitReadings`` with every reading serialised."""

    def __init__(self, readings: ps.UnitReadings, lock: threading.Lock) -> None:
        self._readings = readings
        self._lock = lock

    @property
    def cache(self) -> Any:
        return self._readings.cache

    @property
    def verifier_calls(self) -> int:
        return self._readings.verifier_calls

    def verification(self, steps: Sequence[tuple], origin: int) -> Any:
        with self._lock:
            return self._readings.verification(steps, origin)

    def reading(self, steps: Sequence[tuple], origin: int,
                uids: Sequence[str]) -> tuple[dict[str, Any], int]:
        with self._lock:
            return self._readings.reading(steps, origin, uids)

    def compose(self, assignment: Mapping[str, Any], origin: int,
                population: Sequence[str]) -> tuple[dict[str, Any], int]:
        with self._lock:
            return self._readings.compose(assignment, origin, population)


# ---------------------------------------------------------------------------
# one sequence's decision
# ---------------------------------------------------------------------------

def run_sequence(*, arm: seq2.Arm, ctx: Any, uid: str, readings: Any,
                 machinery: Mapping[str, Any], budget: seq2.Budget,
                 position: int, phase: str, guard: Any, shared: SharedState,
                 edit_probe: str | None = None) -> dict[str, Any]:
    """One sequence: its own session, its own program, its own feedback."""
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
    record["bound_series_uid"] = str(request.series_uid)
    record["bound_values_length"] = int(np.asarray(request.values).size)
    record["history_episodes_in"] = len(arm.history.get(uid, ()))
    record["knowledge_version"] = arm.snapshot.runtime_bundle_sha

    rendered = arm.rendered_view(features)
    record["fast_view_bytes"] = len(rendered)
    if edit_probe:
        record["edited_text_is_in_this_sequences_view"] = bool(
            edit_probe in rendered)

    backend = arm.backend_factory(guard)
    target = machinery["agentic"].live_transport()
    core = machinery["TTHAAgentCore"](
        backend, machinery["LocalPublicToolGateway"](
            np.zeros(8, dtype=np.float64), task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    with shared.history:
        own_history = tuple(arm.history.get(uid, ()))
    method = machinery["TTHAMethod"](machinery["TTHAFastAgent"](core),
                                     arm.snapshot, own_history)
    view = ps.SequenceView(readings, uid)
    guard.open_cell()
    fits_before = int(readings.cache.physical_fits)
    result = None
    try:
        budget.require(runner.CACHE_FITS_PER_CELL)
        guard.reserve(kind="fast", where={"unit": ctx.unit, "arm": arm.name,
                                          "series_uid": uid})
        result = machinery["online_loop"].run_online_round(
            method, view, request, values,
            origin=origin,
            slow_agent=None, controller=None, store=None,
            card_builder=lambda _episode: {
                "pattern_id": "%s_%s" % (arm.slug, uid),
                "observable_signature": {"task_kind": "forecast"}},
            round_name="devseq3_%s_u%03d_%s" % (phase, position, uid),
            budget=ps.PROBE_RECEIPTS_PER_SEQUENCE,
            allow_slow=False, allow_group_slow=False,
            domain=arm.slug, period=int(ctx.config["period"]),
            fast_features=dict(features), allow_fast_skill=False)
    except seq2.PackageCeiling:
        raise
    except runner.UnitFault as exc:
        record["faults"].append({"kind": type(exc).__name__,
                                 "why": str(exc)[:240]})
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        record["faults"].append({"kind": type(exc).__name__,
                                 "why": str(exc)[:300]})

    record["llm_requests_sent"] = int(getattr(backend, "calls", 0) or 0)
    record["returned_models"] = sorted(
        getattr(backend, "returned_models", set()) or set())
    record["llm_calls_this_sequence"] = record["llm_requests_sent"]

    if result is None:
        record.update({
            "deployed": None, "deployed_label": "identity",
            "decision": "FAULT_NO_DECISION",
            "support_gain": "UNKNOWN", "delayed_gain": "UNKNOWN",
            "why_unknown": ("the session faulted, so no program was executed "
                            "and no reading exists; this is not a zero"),
            "probes": [], "candidate_programs": {},
            "action_facts": {"support": "UNKNOWN", "delayed": "UNKNOWN"},
            "consumer_fits_this_sequence": int(readings.cache.physical_fits)
            - fits_before,
            "wall_seconds": round(time.time() - started, 2)})
        return record

    trace = getattr(method, "last_trace", None)
    steps = result._winner_steps
    record["deployed_label"] = ps.program_label_with_params(
        ps.normalise_steps(steps or ()))
    record["deployed"] = result.winner_program
    record["winner_candidate_id"] = str(result._winner_candidate_id or "") or None
    record["retrieved_skill_ids"] = list(
        getattr(trace, "retrieved_skill_ids", ()) or ())
    record["applicability_matches"] = list(
        getattr(trace, "applicability_matches", ()) or ())
    record["inspected_regions"] = [list(r) for r
                                   in (getattr(trace, "inspected_regions", ())
                                       or ())]
    record["tool_calls"] = len(getattr(trace, "tool_calls", ()) or ())
    record["candidate_programs"] = {
        str(key): ps.program_label_with_params(ps.normalise_steps(value))
        for key, value in (getattr(trace, "candidate_program_steps", {})
                           or {}).items()}
    record["candidate_count"] = len(record["candidate_programs"])
    record["chosen_candidate_id"] = str(
        getattr(trace, "chosen_candidate_id", "") or "")
    record["probes"] = [
        {k: v for k, v in dict(row).items()
         if k in ("candidate_id", "kind", "gain", "passed", "relation",
                  "admission")}
        for row in result.actual_probed_programs]
    record["probe_programs"] = [
        {"candidate_id": str(row.get("candidate_id")),
         "program": ps.program_label_with_params(
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
                              else (0.0 if not steps else "UNKNOWN"))
    record["support_credit_basis"] = (
        "this sequence's own gain at its own origin, with only this sequence "
        "treated.  A deliberate identity is 0.0 because the raw pipeline it "
        "falls back to actually ran; a fault is UNKNOWN because nothing ran")

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
    else:
        delayed_gain = 0.0
    record["delayed_gain"] = delayed_gain

    # ---- action facts, both faces, each bound to its own window ----------
    normalised = ps.normalise_steps(steps or ())
    facts: dict[str, Any] = {
        "support": know.action_facts(ctx, origin, normalised, uid)}
    if delayed_gain == "UNKNOWN" and steps:
        facts["delayed"] = {
            "status": "UNKNOWN",
            "why": ("the delayed reading did not legally arrive, so its "
                    "action facts are not recorded either")}
    else:
        facts["delayed"] = know.action_facts(ctx, delayed_origin, normalised,
                                             uid)
    facts["binding"] = (
        "Support and delayed are separate readings; each set of counts is "
        "bound to its own UID, window, program and parameters.  Delayed facts "
        "are recorded only once the delayed reading has legally arrived")
    record["action_facts"] = facts

    written = {str(episode_id) for episode_id in result.episode_ids}
    final = [ep for ep in method.experience_episodes
             if str(getattr(ep, "episode_id", "")) in written]
    record["episodes"] = [str(ep.episode_id) for ep in final]
    record["episode_id"] = record["episodes"][0] if final else None
    with shared.history:
        for episode in final:
            arm.history.setdefault(uid, []).append(episode)
            arm.episodes.append(episode)

    record["consumer_fits_this_sequence"] = (
        int(readings.cache.physical_fits) - fits_before)
    record["wall_seconds"] = round(time.time() - started, 2)
    return record


# ---------------------------------------------------------------------------
# one (unit, arm), sequences in parallel
# ---------------------------------------------------------------------------

def run_unit(*, arm: seq2.Arm, ctx: Any, position: int, phase: str,
             readings: Any, machinery: Mapping[str, Any], budget: seq2.Budget,
             shared: SharedState, edit_probe: str | None = None,
             concurrency: int | None = None) -> dict[str, Any]:
    started = time.time()
    population = population_for(ctx, phase)
    workers = max(1, int(concurrency if concurrency is not None else CONCURRENCY))
    rows_by_uid: dict[str, dict[str, Any]] = {}
    stopped: dict[str, Any] | None = None

    def one(uid: str) -> tuple[str, dict[str, Any] | None, Exception | None]:
        guard = SequenceGuard(ordering_cap=MAX_LLM,
                              per_unit_arm_cap=PER_SEQUENCE_LLM_CAP,
                              ledgers=budget.ledgers, lock=shared.ledger)
        try:
            return uid, run_sequence(
                arm=arm, ctx=ctx, uid=uid, readings=readings,
                machinery=machinery, budget=budget, position=position,
                phase=phase, guard=guard, shared=shared,
                edit_probe=edit_probe), None
        except Exception as exc:  # noqa: BLE001 - re-raised in roster order
            return uid, None, exc

    llm_before = int(budget.ledgers.llm_total())
    fits_before = int(readings.cache.physical_fits)
    if workers == 1:
        results = [one(uid) for uid in population]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, population))

    # Fixed roster order, never completion order.
    for uid, row, exc in sorted(results, key=lambda r: population.index(r[0])):
        if exc is not None:
            if stopped is None:
                stopped = {"series_uid": uid,
                           "why": type(exc).__name__,
                           "detail": str(exc)[:200]}
            continue
        rows_by_uid[uid] = row
    rows = [rows_by_uid[uid] for uid in population if uid in rows_by_uid]

    budget.spend_llm(arm.name, int(budget.ledgers.llm_total()) - llm_before)
    budget.spend_fits(arm.name,
                      int(readings.cache.physical_fits) - fits_before)
    budget.note_evaluation(arm.name, sum(1 + len(r.get("probes") or [])
                                         for r in rows))

    decided = [row["series_uid"] for row in rows]
    assignment = {row["series_uid"]: ps.normalise_steps(row.get("deployed") or ())
                  for row in rows}
    fits_mark = int(readings.cache.physical_fits)
    support_reading, _f = readings.compose(assignment, ctx.origin, decided)
    delayed_reading, _f2 = readings.compose(assignment,
                                            ctx.face_origin(ps.DELAYED), decided)
    budget.spend_fits(arm.name, int(readings.cache.physical_fits) - fits_mark)
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
        "knowledge_version": arm.snapshot.runtime_bundle_sha,
        "concurrency": workers,
        "decision_population": population,
        "decided": decided,
        "stopped_at": stopped,
        "sequences": rows,
        "assignment": {uid: ps.program_label_with_params(steps)
                       for uid, steps in assignment.items()},
        "distinct_programs_deployed": support_reading["distinct_programs"],
        "deployed_count": sum(1 for steps in assignment.values() if steps),
        "identity_count": sum(1 for steps in assignment.values() if not steps),
        "faulted_sequences": [r["series_uid"] for r in rows if r["faults"]],
        "support_reading": support_reading,
        "delayed_reading": delayed_reading,
        "unit_authoritative_gate": gate,
        "sequences_with_the_edited_text_in_view": sum(
            1 for row in rows
            if row.get("edited_text_is_in_this_sequences_view")),
        "aggregation_order": ("fixed roster order; completion order of the "
                              "parallel sessions changes nothing"),
        "wall_seconds": round(time.time() - started, 2),
    }


def run_or_resume(*, arm: seq2.Arm, ctx: Any, position: int, phase: str,
                  readings: Any, machinery: Mapping[str, Any],
                  budget: seq2.Budget, shared: SharedState, checkpoints: Path,
                  name: str, edit_probe: str | None = None,
                  concurrency: int | None = None) -> dict[str, Any] | None:
    path = checkpoints / ("%s.json" % name)
    if RESUME_ONLY and not path.is_file():
        return None
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            saved = None
        rows = (saved or {}).get("episodes")
        unit = (saved or {}).get("unit")
        if unit is not None and isinstance(rows, Mapping):
            restored = seq2._restore_episodes(arm, rows)
            unit = dict(unit)
            unit["resumed_from_checkpoint"] = {
                "path": name, "episodes_restored": restored,
                "not_re_billed": "already inside the carried-over budget"}
            return unit
    unit = run_unit(arm=arm, ctx=ctx, position=position, phase=phase,
                    readings=readings, machinery=machinery, budget=budget,
                    shared=shared, edit_probe=edit_probe,
                    concurrency=concurrency)
    path.write_text(json.dumps(drafts._plain(
        {"unit": unit, "episodes": seq2._episode_rows(arm, unit),
         "cost": budget.to_dict()}), ensure_ascii=False, default=str),
        encoding="utf-8")
    return unit


# ---------------------------------------------------------------------------
# reused formation history
# ---------------------------------------------------------------------------

def load_reused_formation(arm: seq2.Arm, group_name: str) -> dict[str, Any]:
    """DEV-SEQ-2's SOL formation, Episodes and all, from its checkpoints."""
    root = REUSED_FORMATION[group_name]
    units: list[dict[str, Any]] = []
    restored = 0
    for position in FORMATION_POSITIONS:
        path = root / ("formation_u%03d.json" % position)
        if not path.is_file():
            return {"available": False,
                    "why": "missing checkpoint %s" % path.name}
        saved = json.loads(path.read_text(encoding="utf-8"))
        rows = saved.get("episodes")
        if not isinstance(rows, Mapping):
            return {"available": False,
                    "why": ("checkpoint %s predates episode checkpointing and "
                            "cannot be reused" % path.name)}
        restored += seq2._restore_episodes(arm, rows)
        units.append(saved["unit"])
    return {
        "available": True,
        "source": str(root),
        "positions": list(FORMATION_POSITIONS),
        "units": units,
        "episodes_restored": restored,
        "cost_note": (
            "this segment's Fast decisions were paid for in DEV-SEQ-2 and are "
            "NOT re-run and NOT re-billed here.  Its cost is reported "
            "separately; reusing it does not make this package a fresh "
            "end-to-end run and the reused history is not free"),
    }


def backfill_action_facts(units: Sequence[Mapping[str, Any]],
                          contexts: Mapping[int, Any]) -> dict[str, Any]:
    """Add action facts to the reused records, without any Consumer time."""
    filled = 0
    unknown = 0
    for unit in units:
        ctx = contexts[int(unit["position"])]
        for row in unit.get("sequences") or ():
            steps = ps.normalise_steps(row.get("deployed") or ())
            facts = {"support": know.action_facts(ctx, int(row["origin"]),
                                                  steps, row["series_uid"])}
            if row.get("delayed_gain") == "UNKNOWN" and steps:
                facts["delayed"] = {"status": "UNKNOWN",
                                    "why": "the delayed reading never arrived"}
                unknown += 1
            else:
                facts["delayed"] = know.action_facts(
                    ctx, int(row["delayed_origin"]), steps, row["series_uid"])
            facts["binding"] = (
                "each set of counts is bound to its own UID, window, program "
                "and parameters")
            row["action_facts"] = facts
            filled += 1
    return {"records_backfilled": filled,
            "delayed_left_unknown": unknown,
            "how": ("replayed the evaluator's own _prepare on windows already "
                    "legally open at those origins; no refit, no Consumer "
                    "time, no new schema")}


# ---------------------------------------------------------------------------
# the package
# ---------------------------------------------------------------------------

def build(*, group_name: str, budget: seq2.Budget, run_id: str = "",
          skip_smoke: bool = False, concurrency: int | None = None
          ) -> dict[str, Any]:
    from evaluation.main_protocol_p4 import smoke_dev_seq3 as smoke

    # The boundary helpers live in the DEV-SEQ-2 module and read its globals;
    # point them at this package's registered population before using them.
    seq2.ARM_POPULATION = ARM_POPULATION
    seq2.FORMATION_POPULATION = FORMATION_POPULATION
    seq2.MIN_GROUP = MIN_GROUP
    seq2.MAX_FITS, seq2.MAX_LLM = MAX_FITS, MAX_LLM
    seq2.MAX_WALL_SECONDS = MAX_WALL_SECONDS
    seq2.GROUP_WALL_SECONDS = MAX_WALL_SECONDS

    plan = GROUPS[group_name]
    budget.open_group()
    started = datetime.now(timezone(timedelta(hours=8)))
    checkpoints = CHECKPOINTS / (run_id or "run") / group_name
    checkpoints.mkdir(parents=True, exist_ok=True)

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)

    transport = ({"requested_model": REQUIRED_MODEL, "usable": True,
                  "skipped": "resume-only close-out makes no model call",
                  "base_url": "", "source": "not_contacted",
                  "required_model": REQUIRED_MODEL}
                 if RESUME_ONLY else transport_preflight(machinery))
    budget.spend_llm("preflight", int(transport.get("calls") or 0))
    if not transport.get("usable"):
        return {"stage": "DEV_SEQ3_GUIDANCE_DELIVERY", "group": group_name,
                "status": "BLOCKED_BY_TRANSPORT", "transport": transport}

    preflight = ({"skipped": True} if (skip_smoke or RESUME_ONLY)
                 else smoke.run())
    if not (skip_smoke or RESUME_ONLY) and not preflight["passed"]:
        return {"stage": "DEV_SEQ3_GUIDANCE_DELIVERY", "group": group_name,
                "status": "BLOCKED_BY_SMOKE", "smoke": preflight,
                "transport": transport}
    budget.spend_fits("smoke", int(preflight.get("consumer_fits") or 0))

    np.random.seed(int(plan["seed"]) % (2 ** 32))
    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population_rows, _excluded = R._population(doc)
    by_position = {row["position"]: row for row in population_rows}

    snapshot_dir, snapshot_source = R._resolve_k0_snapshot(doc, state["k0"])
    parent_snapshot = machinery["compile_snapshot"](snapshot_dir,
                                                    verify_lock=False)

    shared = SharedState()
    created_backends: list[Any] = []

    def backend_factory(guard: Any) -> Any:
        inner = machinery["agentic"]._default_backend_factory(
            PER_SEQUENCE_LLM_CAP)
        with shared.ledger:
            created_backends.append(inner)
        return runner._MeteredFastBackend(TransientRelayRetry(inner),
                                          guard=guard, billable=True)

    outer_guard = SequenceGuard(ordering_cap=MAX_LLM,
                                per_unit_arm_cap=OUTER_LLM_CAP,
                                ledgers=budget.ledgers, lock=shared.ledger)
    outer_inner = machinery["agentic"]._default_backend_factory(OUTER_LLM_CAP)
    created_backends.append(outer_inner)
    outer_core = machinery["TTHAAgentCore"](
        runner._MeteredOuterBackend(outer_inner, guard=outer_guard,
                                    billable=True),
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=transport["requested_model"], base_url=transport["base_url"])

    store = machinery["SnapshotStore"](
        base.ROOT / ".dev_seq3_runs" / (run_id or "run") / group_name / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    store.materialize(parent_snapshot)

    units: dict[int, Any] = {}
    features_by_position: dict[int, dict[str, dict[str, Any]]] = {}
    origins: dict[int, int] = {}
    for position in ALL_POSITIONS:
        ctx = R.opp._unit_ctx(by_position[position]["unit"])
        units[position] = ctx
        origins[position] = int(ctx.origin)
        size = (FORMATION_POPULATION if position in FORMATION_POSITIONS
                else ARM_POPULATION)
        features_by_position[position] = {
            uid: ps.sequence_features(machinery, ctx, uid, origin=ctx.origin)
            for uid in ps.decision_uids(ctx, size=size)}
    binding = know.FeatureBinding(
        know.bind_window_features(features_by_position, origins))

    cache = runner.ReplayPredictionCache("devseq3-%s" % group_name)
    ledger = know.MeasurementLedger(cache, runner.forecast_p4._config)
    unit_readings: dict[int, Any] = {
        position: LockedReadings(
            ps.UnitReadings(cache=cache, ctx=units[position],
                            executor=units[position].executor,
                            guard=budget.require), shared.fits)
        for position in ALL_POSITIONS}

    stopped_at: dict[str, Any] | None = None

    # ---- reused formation ------------------------------------------------
    formation = seq2.Arm("formation", parent_snapshot, machinery=machinery,
                         backend_factory=None, guard=None, cache=cache)
    reuse = load_reused_formation(formation, group_name)
    if not reuse.get("available"):
        return {"stage": "DEV_SEQ3_GUIDANCE_DELIVERY", "group": group_name,
                "status": "BLOCKED_NO_REUSABLE_HISTORY", "reuse": reuse}
    formation_units = reuse["units"]
    backfill = backfill_action_facts(formation_units, units)
    formation_records = [row for unit in formation_units
                         for row in unit["sequences"]]
    for row in formation_records:
        for face, key in (("support", "origin"), ("delayed", "delayed_origin")):
            value = row.get("support_gain" if face == "support"
                            else "delayed_gain")
            if isinstance(value, float):
                ledger.note(unit=row["unit"], origin=row[key],
                            steps=ps.normalise_steps(row.get("deployed") or ()),
                            uid=row["series_uid"], value=value, face=face)

    # ---- the batch boundary ---------------------------------------------
    saved_boundary = seq2._load_boundary(checkpoints, machinery, store)
    if saved_boundary is not None:
        boundary = saved_boundary
    else:
        try:
            boundary = seq2.knowledge_boundary(
                formation=formation, units=units, binding=binding,
                machinery=machinery, budget=budget, outer_core=outer_core,
                snapshot=parent_snapshot, store=store, controller=controller,
                decision_records=formation_records, group_name=group_name)
        except seq2.PackageCeiling as exc:
            boundary = {"outcome": "STOPPED_AT_THE_CEILING",
                        "why": str(exc)[:240]}
        except Exception as exc:  # noqa: BLE001
            boundary = {"outcome": "BOUNDARY_FAULT",
                        "why": "%s: %s" % (type(exc).__name__, str(exc)[:400])}
        (checkpoints / "boundary.json").write_text(
            json.dumps(drafts._plain({k: v for k, v in boundary.items()
                                      if k != "candidate_snapshot"}),
                       ensure_ascii=False, default=str), encoding="utf-8")

    candidate = boundary.get("candidate_snapshot")
    edit_probe = boundary.get("edit_probe") or None
    never_exposed = (boundary.get("outcome")
                     == "CANDIDATE_COMPILED_BUT_NEVER_EXPOSED")
    if never_exposed:
        candidate = None
    treatment = ("UPDATE_TREATMENT" if candidate is not None
                 else "COMPILED_BUT_NEVER_EXPOSED__UTILITY_UNTESTED"
                 if never_exposed else "NO_UPDATE_TREATMENT")

    arms: dict[str, seq2.Arm] = {}
    validation: dict[str, Any] = {"ran": False}
    adoption: dict[str, Any] = {}
    followup_units: list[dict[str, Any]] = []
    references: dict[str, Any] = {}
    followup: dict[str, Any] = {"ran": False}

    if candidate is None:
        validation = {
            "ran": False,
            "why": (boundary.get("why_the_arms_are_not_run") if never_exposed
                    else "no legal candidate was compiled"),
            "no_candidate_was_written_by_hand": True,
        }
        followup = {"ran": False, "why": validation["why"]}
    else:
        for name, snap in ((ARM_A, parent_snapshot), (ARM_B, candidate)):
            arm = seq2.Arm(name, snap, machinery=machinery,
                           backend_factory=None, guard=None, cache=cache)
            arm.backend_factory = backend_factory
            arm.history = {uid: list(rows)
                           for uid, rows in formation.history.items()}
            arms[name] = arm
        order = tuple(plan["arm_order"])

        validation_units: dict[str, Any] = {}
        for name in order:
            if budget.stop_reason():
                stopped_at = {"phase": "validation", "arm": name,
                              "why": budget.stop_reason()}
                break
            unit = run_or_resume(
                arm=arms[name], ctx=units[VALIDATION_POSITION],
                position=VALIDATION_POSITION, phase="validation",
                readings=unit_readings[VALIDATION_POSITION],
                machinery=machinery, budget=budget, shared=shared,
                checkpoints=checkpoints,
                name="validation_%s" % seq2._slug(name),
                edit_probe=edit_probe, concurrency=concurrency)
            if unit is None:
                continue
            validation_units[name] = unit
            if unit["stopped_at"]:
                stopped_at = {"phase": "validation", "arm": name,
                              **unit["stopped_at"]}
        validation = {"ran": len(validation_units) == 2,
                      "arm_order": list(order), "units": validation_units}
        if len(validation_units) == 2:
            adoption = know.adoption_check(
                validation_a=validation_units[ARM_A],
                validation_b=validation_units[ARM_B])
            validation["adoption"] = adoption

        branch = ("ADOPTED_DEVELOPMENT_BRANCH" if adoption.get("adoptable")
                  else "UNADOPTED_CANDIDATE_DIAGNOSTIC_BRANCH")
        followup = {"ran": False, "arm_order": list(order),
                    "branch_status": branch}
        for position in FOLLOWUP_POSITIONS:
            if budget.stop_reason():
                stopped_at = {"phase": "followup", "position": position,
                              "why": budget.stop_reason()}
                break
            if not RESUME_ONLY:
                references[str(position)] = seq2.reference_readings(
                    ctx=units[position],
                    population=population_for(units[position], "followup"),
                    readings=unit_readings[position], budget=budget)
            for name in order:
                unit = run_or_resume(
                    arm=arms[name], ctx=units[position], position=position,
                    phase="followup", readings=unit_readings[position],
                    machinery=machinery, budget=budget, shared=shared,
                    checkpoints=checkpoints,
                    name="followup_u%03d_%s" % (position, seq2._slug(name)),
                    edit_probe=edit_probe, concurrency=concurrency)
                if unit is None:
                    continue
                followup_units.append(unit)
                if unit["stopped_at"]:
                    stopped_at = {"phase": "followup", "position": position,
                                  "arm": name, **unit["stopped_at"]}
            if stopped_at:
                break
        followup["ran"] = bool(followup_units)
        followup["positions"] = list(FOLLOWUP_POSITIONS)

    finished = datetime.now(timezone(timedelta(hours=8)))
    result = {
        "stage": "DEV_SEQ3_GUIDANCE_DELIVERY",
        "status": "COMPLETE" if stopped_at is None else "STOPPED",
        "group": group_name,
        "run_id": run_id or None,
        "question": ("when Slow is given accurate outcome and action facts, "
                     "and the guidance it writes actually reaches Fast, does "
                     "the next batch of per-sequence processing improve"),
        "registered_plan": {
            "formation_positions": list(FORMATION_POSITIONS),
            "formation_is_reused_not_rerun": True,
            "validation_position": VALIDATION_POSITION,
            "followup_positions": list(FOLLOWUP_POSITIONS),
            "origins": {str(p): origins[p] for p in ALL_POSITIONS},
            "seed": plan["seed"],
            "arm_order": list(plan["arm_order"]),
            "formation_population": FORMATION_POPULATION,
            "arm_population": ARM_POPULATION,
            "population_change_from_dev_seq_2": (
                "the arm population goes from 6 to the full 20 eval series; "
                "the reused formation history stays at 10.  Fixed before the "
                "run, in roster order, never on a gain.  DEV-SEQ-3 readings "
                "are NOT subtracted against DEV-SEQ-2's"),
            "concurrency": int(concurrency if concurrency is not None
                               else CONCURRENCY),
            "registered_before_the_run": True,
        },
        "started": started.isoformat(),
        "finished": finished.isoformat(),
        "transport": transport,
        "transport_actual_usage": {
            "requested_model": transport["requested_model"],
            "returned_models": sorted({
                model for backend in created_backends
                for model in getattr(backend, "returned_models", set()) or ()}),
            "prompt_tokens": sum(int(getattr(b, "prompt_tokens", 0) or 0)
                                 for b in created_backends),
            "completion_tokens": sum(int(getattr(b, "completion_tokens", 0) or 0)
                                     for b in created_backends),
            "backends_created": len(created_backends),
        },
        "k0_snapshot": {"dir": str(snapshot_dir), "source": snapshot_source,
                        "runtime_bundle_sha": parent_snapshot.runtime_bundle_sha},
        "reused_formation": {k: v for k, v in reuse.items() if k != "units"},
        "action_fact_backfill": backfill,
        "feature_binding": binding.to_dict(),
        "measurement_ledger": ledger.to_dict(),
        "formation": {"units": formation_units,
                      "episodes_restored": reuse["episodes_restored"]},
        "boundary": {k: v for k, v in boundary.items()
                     if k != "candidate_snapshot"},
        "treatment": treatment,
        "validation": validation,
        "adoption": adoption,
        "followup": followup,
        "followup_units": followup_units,
        "reference_readings": references,
        "stopped_at": stopped_at,
        "early_termination": ({"terminated_early": True, "reason": EARLY_STOP}
                              if EARLY_STOP else None),
        "cost": budget.to_dict(),
        "smoke": preflight,
        "what_this_is_not": (
            "development level, one course, already-exposed units.  Two groups "
            "are random repeats on the same data, not independent "
            "generalisation evidence.  No significance is claimed."),
    }
    (checkpoints / "result.json").write_text(
        json.dumps(drafts._plain(result), ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    global RESUME_ONLY, EARLY_STOP, CONCURRENCY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=sorted(GROUPS))
    parser.add_argument("--run", default="")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--resume-only", action="store_true")
    parser.add_argument("--early-stop", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.resume_only:
        RESUME_ONLY = True
    if args.early_stop:
        EARLY_STOP = str(args.early_stop)
    if args.concurrency is not None:
        CONCURRENCY = int(args.concurrency)

    groups = args.group or sorted(GROUPS)
    budget_path = CHECKPOINTS / (args.run or "run") / "package_budget.json"
    budget = (seq2.Budget.load(budget_path) if budget_path.is_file()
              else seq2.Budget())
    suffix = ("_" + args.suffix.strip("_")) if args.suffix else ""
    for group_name in groups:
        result = build(group_name=group_name, budget=budget, run_id=args.run,
                       skip_smoke=args.skip_smoke,
                       concurrency=args.concurrency)
        budget.save(budget_path)
        out = ART / ("dev_seq3_guidance_delivery__%s%s.json"
                     % (group_name, suffix))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(drafts._plain(result), ensure_ascii=False,
                                  indent=1, default=str), encoding="utf-8")
        cost = result.get("cost") or {}
        print("%s %s | fits %s | llm %s | %ss"
              % (result.get("status"), group_name,
                 cost.get("physical_consumer_fits"), cost.get("llm_calls"),
                 cost.get("wall_seconds")))
        boundary = result.get("boundary") or {}
        print("   boundary: %s | surface: %s | treatment: %s"
              % (boundary.get("outcome"),
                 (boundary.get("proposal") or {}).get("target_surface_id"),
                 result.get("treatment")))
        exposure = boundary.get("exposure") or {}
        if exposure:
            print("   exposure: %s/%s decisions"
                  % (exposure.get("decisions_the_edit_reaches"),
                     exposure.get("decisions_checked")))
        print("   wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
