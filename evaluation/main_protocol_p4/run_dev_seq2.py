"""DEV-SEQ-2: does one Slow edit to shared guidance change the next Fast?

The loop under test, end to end and in one package:

    per-sequence Fast on a common formation history
      -> one batch boundary: group the material, one real Slow edit on one
         authorised guidance surface, compiled into an isolated fork
      -> the SAME withheld unit run twice by real Fast: arm A on the parent
         knowledge, arm B on the candidate knowledge
      -> both arms carried into two further units, whether or not the
         candidate passed this package's adoption mark

What is registered before the run, and not chosen on any gain:

    formation      u7 / origin 1176, u8 / origin 1416   (+48 delayed face)
    validation     u9 / origin 1656                     (+48)
    follow-up      u10 / origin 2136, u11 / origin 2376 (+48)
    population     the first 10 eval series of each block, in roster order
    groups         g1 seed 20260908 arm order A->B
                   g2 seed 20260909 arm order B->A
    arms           A = knowledge-frozen (the parent snapshot)
                   B = slow-candidate   (the compiled fork)

Every position is already-exposed development.  Nothing sealed, nothing at
TARGET_HELD_IN and nothing at +144 is read.

Two arms, one difference
------------------------
A and B share the model, the tools, the DSL, the formation history, the
feedback permissions, the population and the per-sequence Fast budget.  The
only difference is the knowledge snapshot.  Each sequence gets a fresh Fast
session holding only that sequence's own history and the arm's frozen
snapshot: no Slow conversation, no reflection text and no raw cross-trajectory
Episode reaches Fast, in either arm.  After the fork the arms carry their own
trajectories and their own costs, and neither reads the other's probes.

Cost accounting
---------------
The Consumer reading is a deterministic function of (unit, origin, training
config, program, served set), so the two arms share one prediction cache and
a program both arms deploy is fitted once.  That is the physical cost, not a
discount, and it would otherwise make whichever arm ran first look expensive.
Physical fits are attributed to the arm whose call actually missed the cache;
each arm's *logical* evaluation count is reported separately alongside.

What this is not
----------------
A development-level mechanism test on one course.  Two groups are random
repeats on the same data, not two datasets and not independent generalisation
evidence.  No significance is claimed.  Readings are per-sequence and are not
comparable with the cohort-broadcast numbers of the earlier packages.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
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
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as m_r0k

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_seq2"

FORMATION_POSITIONS = (7, 8)
VALIDATION_POSITION = 9
FOLLOWUP_POSITIONS = (10, 11)
ALL_POSITIONS = (*FORMATION_POSITIONS, VALIDATION_POSITION, *FOLLOWUP_POSITIONS)

ARM_A = "knowledge-frozen"
ARM_B = "slow-candidate"

GROUPS: dict[str, dict[str, Any]] = {
    "g1": {"seed": 20260908, "arm_order": (ARM_A, ARM_B)},
    "g2": {"seed": 20260909, "arm_order": (ARM_B, ARM_A)},
}

DOMAIN = "devseq2"
ANCESTOR_PROGRAM = m_r0k.ANCESTOR_PROGRAM

#: Package ceilings, first to bind stops the package.  Suggested by the work
#: order; they limit runaway, they are not a quota to be spent.
MAX_FITS = 2000
MAX_LLM = 2000
MAX_WALL_SECONDS = 6 * 3600

#: A runaway guard on one sequence's Fast session, not a flow cap: a normal
#: inspect/propose/select flow with its schema retries is far below this.
PER_SEQUENCE_LLM_CAP = 24
OUTER_LLM_CAP = 64

#: Neither group may starve the other of the shared wall clock.  When the two
#: groups run as concurrent OS processes -- which is how this package runs
#: them, because a run process measured 466 MB resident and is blocked on the
#: relay rather than on CPU -- each process owns the whole six hours and half
#: of the count ceilings, so the package total is still 2000/2000/6h.  When
#: they run sequentially in one process this is halved instead.  Set by
#: ``--group-wall``; a group that runs out stops at a unit boundary and says
#: so rather than borrowing the other group's time.
GROUP_WALL_SECONDS = MAX_WALL_SECONDS

#: How many sequences each phase decides for, in the fixed roster order.
#:
#: These are not equal, and the reason is a measured cost rather than a
#: preference.  Two shakedowns on the fixed non-Flash model (deepseek-v4-pro)
#: measured about 260 wall seconds per per-sequence decision, range 142-379:
#: it is a reasoning model and spends about 2.8k completion tokens per stage.
#: At the DEV-SEQ-1 population of ten everywhere, the registered five-unit
#: plan is about 56 decisions and roughly four hours per group.
#:
#: What is protected, in this order.  (1) A group has to form at all, or the
#: boundary has nothing to propose from and the package tests nothing --
#: DEV-SEQ-1 saw about one material failure per four decisions, so the
#: formation population stays at ten over two units, as in DEV-SEQ-1.  (2)
#: Both registered follow-up units have to run, because the main readout is
#: the equal-weight mean over them.  (3) Whatever paired width fits then goes
#: to the arms: six sequences, the first six of the same fixed roster order.
#:
#: 20 + 12 + 24 = 56 decisions per group.  Fixed before the run, on measured
#: cost and never on a gain, and the two arms always decide for exactly the
#: same set.  The paired n is small and the report says so rather than
#: compensating for it.
FORMATION_POPULATION = 10
ARM_POPULATION = 6

#: Two comparable decisions before a group exists.  Test-only override.
MIN_GROUP = 2

#: Assemble a group's artifact from the checkpoints it already paid for and
#: run nothing new.  Used to close out a group stopped on purpose, so the
#: result is the work that actually happened rather than a lost run.
RESUME_ONLY = False
EARLY_STOP = ""

#: The model this package is fixed to.  A request name is not an identity:
#: the returned model string is checked against it, and against the Flash
#: family the work order excludes.
#:
#: The package opened on ``deepseek-v4-pro`` (requested name and returned name
#: matched, and it is not the ``deepseek-chat`` -> ``deepseek-v4-flash`` alias
#: DEV-SEQ-1 ran on).  That account then returned ``402 Insufficient
#: Balance`` at g1's batch boundary.  An account error is not a reason to
#: retry and not a reason to buy anything, so the package moved to the other
#: endpoint the work order authorises, ``gpt-5.6-sol`` on the already
#: configured relay, and **re-ran from the beginning**: one model for the
#: whole package, and the deepseek work is kept as an abandoned attempt
#: rather than merged.  Set with ``--model``.
REQUIRED_MODEL = "gpt-5.6-sol"
EXCLUDED_MODEL_MARKERS = ("flash",)

#: The work order excludes the Flash family from the package's main line, and
#: the preflight refuses it.  ``--allow-flash`` lifts that refusal for a
#: separate, explicitly labelled speed probe the user asked for.  It is never
#: silent: the run stamps ``line`` = "fast probe", its artifact carries its own
#: suffix, and its readings are never tabled with the main line's -- a
#: different model is a different experiment.
ALLOW_EXCLUDED_MODEL = False
LINE = "main"


def _episode_rows(arm: "Arm", unit: Mapping[str, Any]) -> dict[str, list[dict]]:
    """This unit's Episodes, by sequence, in the form they can be rebuilt from."""
    wanted = {str(eid) for row in unit.get("sequences") or ()
              for eid in (row.get("episodes") or ())}
    out: dict[str, list[dict]] = {}
    for uid, episodes in arm.history.items():
        rows = [ep.to_dict() for ep in episodes
                if str(getattr(ep, "episode_id", "")) in wanted]
        if rows:
            out[str(uid)] = rows
    return out


def _restore_episodes(arm: "Arm", rows: Mapping[str, Sequence[Mapping]]) -> int:
    """Put a checkpointed unit's Episodes back into the arm, in order."""
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415
        ExperienceEpisode,
    )
    restored = 0
    for uid in sorted(rows):
        for row in rows[uid]:
            payload = dict(row)
            payload["evidence_refs"] = tuple(payload.get("evidence_refs") or ())
            episode = ExperienceEpisode(**payload)
            arm.history.setdefault(str(uid), []).append(episode)
            arm.episodes.append(episode)
            restored += 1
    return restored


def run_or_resume(*, arm: "Arm", ctx: Any, position: int, phase: str,
                  readings: Any, machinery: Mapping[str, Any], budget: "Budget",
                  checkpoints: Any, name: str,
                  edit_probe: str | None = None) -> dict[str, Any] | None:
    """One (unit, arm), from its checkpoint when there is a complete one.

    A resumed unit is not re-billed and not re-called: its readings, its
    decisions and its Episodes are the ones the earlier process actually
    produced, and its cost is already inside the carried-over budget.  A
    checkpoint written before this mechanism existed carries no Episodes, so
    it cannot be resumed and the unit is run again rather than continued from
    a state that would silently be missing its feedback.
    """
    path = checkpoints / ("%s.json" % name)
    if RESUME_ONLY and not path.is_file():
        return None
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - an unreadable checkpoint is no checkpoint
            saved = None
        rows = (saved or {}).get("episodes")
        unit = (saved or {}).get("unit")
        if unit is not None and isinstance(rows, Mapping):
            restored = _restore_episodes(arm, rows)
            unit = dict(unit)
            unit["resumed_from_checkpoint"] = {
                "path": name, "episodes_restored": restored,
                "not_re_billed": ("this unit's cost is already inside the "
                                  "carried-over package budget; no call and "
                                  "no fit is repeated"),
            }
            return unit
    unit = run_unit(arm=arm, ctx=ctx, position=position, phase=phase,
                    readings=readings, machinery=machinery, budget=budget,
                    edit_probe=edit_probe)
    path.write_text(json.dumps(drafts._plain(
        {"unit": unit, "episodes": _episode_rows(arm, unit),
         "cost": budget.to_dict()}), ensure_ascii=False, default=str),
        encoding="utf-8")
    return unit


def population_for(ctx: Any, phase: str) -> list[str]:
    """The sequences this phase decides for, in the fixed roster order."""
    # Fetch the wider of the two rosters and slice per phase.  Fetching the
    # formation size first and slicing to the arm size silently capped the arm
    # population at the formation population whenever the arms were the wider
    # of the two -- which is exactly DEV-SEQ-3's configuration.
    uids = ps.decision_uids(ctx, size=max(int(FORMATION_POPULATION),
                                          int(ARM_POPULATION)))
    if phase == "formation":
        return uids[: int(FORMATION_POPULATION)]
    return uids[: int(ARM_POPULATION)]


class PackageCeiling(RuntimeError):
    """A package ceiling would be crossed by the next spend."""


class Budget:
    """Fits, LLM calls and wall clock, shared by every group of the package."""

    def __init__(self) -> None:
        self.started = time.time()
        self.ledgers = runner.Ledgers()
        self.fits_by_arm: dict[str, int] = {}
        self.llm_by_arm: dict[str, int] = {}
        self.logical_evaluations_by_arm: dict[str, int] = {}
        self.stopped: str | None = None
        #: Set at the start of each group, so one group cannot spend the
        #: other's share of the package wall clock.
        self.group_deadline: float | None = None

    # ---- spending ---------------------------------------------------------
    def fits(self) -> int:
        return sum(self.fits_by_arm.values())

    def llm(self) -> int:
        return int(self.ledgers.llm_total())

    def elapsed(self) -> float:
        return time.time() - self.started

    def spend_fits(self, arm: str, n: int) -> None:
        if n:
            self.fits_by_arm[arm] = self.fits_by_arm.get(arm, 0) + int(n)

    def spend_llm(self, arm: str, n: int) -> None:
        if n:
            self.llm_by_arm[arm] = self.llm_by_arm.get(arm, 0) + int(n)

    def note_evaluation(self, arm: str, n: int = 1) -> None:
        self.logical_evaluations_by_arm[arm] = (
            self.logical_evaluations_by_arm.get(arm, 0) + int(n))

    def room_for(self, fits: int) -> bool:
        return (self.fits() + int(fits) <= MAX_FITS
                and self.llm() <= MAX_LLM
                and self.elapsed() <= MAX_WALL_SECONDS)

    def require(self, fits: int) -> None:
        if not self.room_for(fits):
            raise PackageCeiling(
                "package ceiling: fits %d+%d/%d llm %d/%d wall %.0f/%d"
                % (self.fits(), int(fits), MAX_FITS, self.llm(), MAX_LLM,
                   self.elapsed(), MAX_WALL_SECONDS))

    def open_group(self) -> None:
        self.group_deadline = time.time() + GROUP_WALL_SECONDS

    def stop_reason(self) -> str | None:
        if self.fits() >= MAX_FITS:
            return "FIT_CEILING"
        if self.llm() >= MAX_LLM:
            return "LLM_CEILING"
        if self.elapsed() >= MAX_WALL_SECONDS:
            return "WALL_CLOCK_CEILING"
        if self.group_deadline is not None and time.time() >= self.group_deadline:
            return "GROUP_WALL_ALLOWANCE"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_consumer_fits": self.fits(),
            "fit_ceiling": MAX_FITS,
            "llm_calls": self.llm(),
            "llm_ceiling": MAX_LLM,
            "wall_seconds": round(self.elapsed(), 1),
            "wall_ceiling_seconds": MAX_WALL_SECONDS,
            "physical_fits_by_arm": dict(sorted(self.fits_by_arm.items())),
            "llm_calls_by_arm": dict(sorted(self.llm_by_arm.items())),
            "logical_evaluations_by_arm": dict(
                sorted(self.logical_evaluations_by_arm.items())),
            "cost_note": (
                "physical fits are attributed to the arm whose call missed "
                "the shared prediction cache; logical evaluations count every "
                "reading an arm asked for, hit or miss.  The two arms share "
                "one cache because the reading is a deterministic function of "
                "(unit, origin, config, program, served set), so whichever arm "
                "runs first would otherwise carry the whole cost"),
            "counting": ("every physical fit, including ones spent inside a "
                         "call that then raised"),
        }

    # ---- cross-process persistence ---------------------------------------
    def to_state(self) -> dict[str, Any]:
        return {"started": self.started,
                "fits_by_arm": dict(self.fits_by_arm),
                "llm_by_arm": dict(self.llm_by_arm),
                "logical_evaluations_by_arm": dict(
                    self.logical_evaluations_by_arm),
                "ledgers": dataclasses.asdict(self.ledgers)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "Budget":
        budget = cls()
        budget.started = float(state.get("started") or budget.started)
        budget.fits_by_arm = {str(k): int(v) for k, v
                              in (state.get("fits_by_arm") or {}).items()}
        budget.llm_by_arm = {str(k): int(v) for k, v
                             in (state.get("llm_by_arm") or {}).items()}
        budget.logical_evaluations_by_arm = {
            str(k): int(v) for k, v
            in (state.get("logical_evaluations_by_arm") or {}).items()}
        for field_name, value in (state.get("ledgers") or {}).items():
            if hasattr(budget.ledgers, field_name):
                setattr(budget.ledgers, field_name, value)
        return budget

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state(), ensure_ascii=False,
                                   default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Budget":
        return cls.from_state(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------------

def transport_preflight(machinery: Mapping[str, Any]) -> dict[str, Any]:
    """One real call, to check the identity of what answers -- not the alias.

    ``deepseek-chat`` is an alias that this relay answers with
    ``deepseek-v4-flash``; the previous package ran on it without noticing.
    So the requested name is compared with the *returned* name, and the Flash
    family is refused outright.  One billed call.
    """
    target = machinery["agentic"].live_transport()
    row: dict[str, Any] = {
        "requested_model": target["model"],
        "base_url": target["base_url"],
        "source": target["source"],
        "required_model": REQUIRED_MODEL,
    }
    if str(target["model"]) != REQUIRED_MODEL:
        row.update({"usable": False,
                    "why": "M0_AGENT_MODEL is not the model this package fixes on"})
        return row
    from SelfEvolvingHarnessTS.runtime.agent_backend import AgentRequest

    zero = "0" * 64
    backend = machinery["agentic"]._default_backend_factory(2)
    try:
        request = AgentRequest(
            case_id="devseq2-transport-preflight", role="fast",
            stage="inspect", call_index=0, replicate_id="preflight",
            messages=(
                {"role": "system", "content": "Answer with one short word."},
                {"role": "user", "content": "Reply with the single word: ready"},
            ),
            envelope_schema_sha=zero, tool_schema_sha=zero,
            tool_result_schema_sha=zero, stage_schema_sha=zero,
            public_case_view_sha=zero, effective_harness_view_sha=zero,
            tool_context_sha=zero,
            model=target["model"], base_url=target["base_url"])
        response = backend.complete(request)
        reply = getattr(response, "assistant_text", "") or ""
    except Exception as exc:  # noqa: BLE001 - a configuration receipt, not a crash
        row.update({"usable": False,
                    "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
        return row
    returned = sorted(getattr(backend, "returned_models", set()) or set())
    row.update({
        "returned_models": returned,
        "reply_nonempty": bool(str(reply or "").strip()),
        "calls": int(getattr(backend, "calls", 0) or 0),
    })
    excluded = any(marker in name.lower() for name in returned
                   for marker in EXCLUDED_MODEL_MARKERS)
    row["is_an_excluded_family"] = excluded
    row["excluded_family_allowed_explicitly"] = bool(
        excluded and ALLOW_EXCLUDED_MODEL)
    row["usable"] = bool(returned == [REQUIRED_MODEL]
                         and (not excluded or ALLOW_EXCLUDED_MODEL))
    if not row["usable"]:
        row["why"] = ("the endpoint answered as %s, which is not the fixed "
                      "non-Flash model" % (returned or "nothing"))
    return row


# ---------------------------------------------------------------------------
# one knowledge version and its sessions
# ---------------------------------------------------------------------------

class Arm:
    """One knowledge version, its per-sequence histories, its own trajectory."""

    def __init__(self, name: str, snapshot: Any, *, machinery: Mapping[str, Any],
                 backend_factory: Any, guard: Any, cache: Any) -> None:
        self.name = str(name)
        self.slug = DOMAIN
        self.snapshot = snapshot
        self.m = machinery
        self.backend_factory = backend_factory
        self.guard = guard
        self.cache = cache
        self.history: dict[str, list[Any]] = {}
        self.episodes: list[Any] = []
        self.backends: list[Any] = []

    def capability_cards(self) -> list[str]:
        from SelfEvolvingHarnessTS.contracts.harness import SkillKind
        return sorted(str(skill.skill_id) for skill in self.snapshot.skills
                      if skill.skill_kind is SkillKind.CAPABILITY)

    def fresh_method(self, uid: str) -> tuple[Any, Any]:
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

    def rendered_view(self, features: Mapping[str, Any]) -> str:
        """The Fast-visible knowledge for one sequence, as the bytes it sees."""
        from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: PLC0415
            resolve_harness_view,
        )
        view = resolve_harness_view(self.snapshot, dict(features), role="fast")
        return json.dumps({
            "instruction": view.instruction,
            "skills": [{"skill_id": s.skill_id, "body": s.body}
                       for s in view.skills],
            "controls": json.loads(json.dumps(view.controls, default=dict)),
        }, ensure_ascii=False, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# one sequence's decision
# ---------------------------------------------------------------------------

def run_sequence(*, arm: Arm, ctx: Any, uid: str, readings: ps.UnitReadings,
                 machinery: Mapping[str, Any], budget: Budget, position: int,
                 phase: str, edit_probe: str | None = None) -> dict[str, Any]:
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
    record["bound_series_uid"] = str(request.series_uid)
    record["bound_values_length"] = int(np.asarray(request.values).size)
    record["history_episodes_in"] = len(arm.history.get(uid, ()))
    record["knowledge_version"] = arm.snapshot.runtime_bundle_sha

    # behaviour chain, link 1: is the edited text in what this sequence reads?
    rendered = arm.rendered_view(features)
    record["fast_view_bytes"] = len(rendered)
    if edit_probe:
        record["edited_text_is_in_this_sequences_view"] = bool(
            edit_probe in rendered)

    method, backend = arm.fresh_method(uid)
    view = ps.SequenceView(readings, uid)
    llm_before = int(budget.ledgers.llm_total())
    arm.guard.open_cell()
    arm.guard.per_unit_arm_cap = PER_SEQUENCE_LLM_CAP
    fits_before = int(arm.cache.physical_fits)
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
            round_name="devseq2_%s_u%03d_%s" % (phase, position, uid),
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
    record["returned_models"] = sorted(
        getattr(backend, "returned_models", set()) or set())

    if result is None:
        budget.spend_fits(arm.name, int(arm.cache.physical_fits) - fits_before)
        record.update({
            "deployed": None, "deployed_label": "identity",
            "decision": "FAULT_NO_DECISION",
            "support_gain": "UNKNOWN", "delayed_gain": "UNKNOWN",
            "why_unknown": ("the session faulted, so no program was executed "
                            "and no reading exists; this is not a zero"),
            "probes": [], "candidate_programs": {},
            "wall_seconds": round(time.time() - started, 2)})
        return record

    trace = getattr(method, "last_trace", None)
    steps = result._winner_steps
    record["deployed_label"] = ps.program_label_with_params(
        ps.normalise_steps(steps or ()))
    record["deployed_label_by_operator"] = ps.program_label(steps)
    record["deployed"] = result.winner_program
    record["winner_candidate_id"] = str(result._winner_candidate_id or "") or None

    # behaviour chain, links 2-4: observation, candidates, selection
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
    record["memory_resolution_status"] = str(
        getattr(trace, "memory_resolution_status", "") or "")
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
        "falls back to actually ran and is bit-identical to Static; a fault "
        "is UNKNOWN because nothing ran")

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
    record["delayed_credit_basis"] = (
        "this sequence's own delayed reading of the program it actually ran; "
        "never a group mean, and UNKNOWN when it could not be read")

    written = {str(episode_id) for episode_id in result.episode_ids}
    final = [ep for ep in method.experience_episodes
             if str(getattr(ep, "episode_id", "")) in written]
    record["episodes"] = [str(ep.episode_id) for ep in final]
    record["episode_id"] = record["episodes"][0] if final else None
    for episode in final:
        arm.history.setdefault(uid, []).append(episode)
        arm.episodes.append(episode)

    spent_fits = int(arm.cache.physical_fits) - fits_before
    budget.spend_fits(arm.name, spent_fits)
    budget.note_evaluation(arm.name, 1 + len(record["probes"]))
    record["consumer_fits_this_sequence"] = spent_fits
    record["wall_seconds"] = round(time.time() - started, 2)
    return record


# ---------------------------------------------------------------------------
# one (unit, arm)
# ---------------------------------------------------------------------------

def run_unit(*, arm: Arm, ctx: Any, position: int, phase: str,
             readings: ps.UnitReadings, machinery: Mapping[str, Any],
             budget: Budget, edit_probe: str | None = None) -> dict[str, Any]:
    started = time.time()
    population = population_for(ctx, phase)
    rows: list[dict[str, Any]] = []
    stopped: dict[str, Any] | None = None
    for uid in population:
        try:
            rows.append(run_sequence(arm=arm, ctx=ctx, uid=uid,
                                     readings=readings, machinery=machinery,
                                     budget=budget, position=position,
                                     phase=phase, edit_probe=edit_probe))
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
    fits_before = int(readings.cache.physical_fits)
    support_reading, _f = readings.compose(assignment, ctx.origin, decided)
    delayed_reading, _f2 = readings.compose(assignment,
                                            ctx.face_origin(ps.DELAYED), decided)
    budget.spend_fits(arm.name,
                      int(readings.cache.physical_fits) - fits_before)
    budget.note_evaluation(arm.name, 2)
    gate = None
    if delayed_reading.get("aggregate_gain") is not None:
        gate = runner.authoritative_gate({
            "treated": delayed_reading["treated"],
            "aggregate_gain": delayed_reading["aggregate_gain"],
            "harmed_fraction": delayed_reading["harmed_fraction"],
            "max_single_series_harm": delayed_reading["max_single_series_harm"],
        })
    faulted = [row["series_uid"] for row in rows if row["faults"]]
    return {
        "phase": phase, "position": int(position), "unit": ctx.unit,
        "arm": arm.name, "origin": int(ctx.origin),
        "knowledge_version": arm.snapshot.runtime_bundle_sha,
        "decision_population": population,
        "decided": decided,
        "stopped_at": stopped,
        "sequences": rows,
        "assignment": {uid: ps.program_label_with_params(steps)
                       for uid, steps in assignment.items()},
        "distinct_programs_deployed": support_reading["distinct_programs"],
        "deployed_count": sum(1 for steps in assignment.values() if steps),
        "identity_count": sum(1 for steps in assignment.values() if not steps),
        "faulted_sequences": faulted,
        "support_reading": support_reading,
        "delayed_reading": delayed_reading,
        "unit_authoritative_gate": gate,
        "gate_note": ("recorded.  Knowledge is written only at the batch "
                      "boundary, so no unit gate activates anything here; the "
                      "validation unit's gate is one line of this package's "
                      "adoption mark"),
        "sequences_with_the_edited_text_in_view": sum(
            1 for row in rows
            if row.get("edited_text_is_in_this_sequences_view")),
        "wall_seconds": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# reference readings
# ---------------------------------------------------------------------------

def reference_readings(*, ctx: Any, population: Sequence[str],
                       readings: ps.UnitReadings, budget: Budget
                       ) -> dict[str, Any]:
    """Raw, and one fixed cohort-level preparation, on the same population."""
    out: dict[str, Any] = {}
    cache = readings.cache
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
            "what_it_is": ("the ancestor program given to every sequence -- "
                           "the cohort-level answer, kept as context only.  "
                           "The baseline this package compares against is arm "
                           "A, the policy that was actually executable")}
    except PackageCeiling as exc:
        out["fixed_preparation"] = {"unavailable": str(exc)[:200]}
    budget.spend_fits("reference", int(cache.physical_fits) - fits_before)
    return out


# ---------------------------------------------------------------------------
# the batch boundary: one Slow edit
# ---------------------------------------------------------------------------

def _edit_probe(*, operation: Any, skill_id: Any, written: Any,
                current: Any) -> tuple[str, str]:
    """A short string present in the candidate and absent from the parent."""
    if str(operation) == "ADD" and skill_id:
        return str(skill_id), "new_skill_id"
    text = written if isinstance(written, str) else json.dumps(
        written, ensure_ascii=False, sort_keys=True, default=str)
    old_text = current if isinstance(current, str) else json.dumps(
        current, ensure_ascii=False, sort_keys=True, default=str)
    text = " ".join(str(text).split())
    old_text = " ".join(str(old_text or "").split())
    width = 60
    shared = 0
    while (shared < len(text) and shared < len(old_text)
           and text[shared] == old_text[shared]):
        shared += 1
    for start in range(shared, max(len(text) - 1, 0) + 1):
        window = text[start:start + width]
        if len(window) >= 20 and window not in old_text:
            return window, "the_new_text_from_where_it_leaves_the_old_value"
    for start in range(0, max(len(text) - width, 0) + 1):
        window = text[start:start + width]
        if len(window) >= 20 and window not in old_text:
            return window, "first_window_of_new_text_absent_from_the_parent"
    return text[:width], "no_novel_window_found__probe_may_match_the_parent"


def knowledge_boundary(*, formation: Arm, units: Mapping[int, Any],
                       binding: know.FeatureBinding,
                       machinery: Mapping[str, Any], budget: Budget,
                       outer_core: Any, snapshot: Any, store: Any,
                       controller: Any, decision_records: Sequence[Mapping],
                       group_name: str) -> dict[str, Any]:
    """Group the material, let Slow write one edit, compile it into a fork."""
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent

    construction_origins = {int(units[p].origin) for p in FORMATION_POSITIONS}
    construction = [ep for ep in formation.episodes
                    if know.episode_origin(ep) in construction_origins]
    grouping = know.group_failures(episodes=construction, binding=binding,
                                   min_group=MIN_GROUP)
    out: dict[str, Any] = {
        "construction_positions": list(FORMATION_POSITIONS),
        "withheld_position": VALIDATION_POSITION,
        "episodes_in_the_slow_input": len(construction),
        "the_withheld_unit_is_not_in_the_slow_input": True,
        "raw_episodes_stay_on_the_slow_side": (
            "Fast in both arms reads the frozen snapshot and its own "
            "sequence's history only; this corpus never reaches it"),
        "grouping": grouping,
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
                             "group id; not chosen on any gain.  Slow is told "
                             "it may take a sub-group, disagree, or abstain")
    card = know.evidence_card(
        group, pattern_id="devseq2-%s-%s" % (group_name, group["sign"].lower()),
        binding=binding, vocabulary=contract.SCOPE_CLASS["vocabulary"],
        decision_records=decision_records)
    out["card"] = card

    parent = store.materialize(snapshot)
    catalog = know.build_catalog(controller=controller, parent=parent)
    out["surface_catalog"] = [{k: v for k, v in row.items()
                               if k != "current_value"} for row in catalog]
    out["surfaces_offered"] = [row["surface_id"] for row in catalog]
    out["surfaces_not_offered"] = {
        "skill_library.entries/{skill_id}.body": (
            "no existing *guidance* capability entry in this snapshot: the "
            "only capability card is a frozen program card, whose body is an "
            "execution body and is outside this package's whitelist"),
        "skill_library.entries/{skill_id}.observable_applicability": (
            "same reason"),
    }
    if not catalog:
        out["outcome"] = "NO_AUTHORISED_SURFACE"
        return out

    def slow_factory():
        return TTHASlowAgent(outer_core)

    llm_before = int(budget.ledgers.llm_total())
    proposal = know.propose_update(slow_factory=slow_factory, card=card,
                                   catalog=catalog, snapshot=snapshot)
    budget.spend_llm("slow", int(budget.ledgers.llm_total()) - llm_before)
    out["proposal"] = {k: v for k, v in proposal.items() if k != "manifest"}
    if not proposal.get("proposed"):
        out["outcome"] = {
            "ACCOUNT_OR_PERMISSION_FAULT": "BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT",
            "TRANSPORT_TRANSIENT_FAULT": "BOUNDARY_TRANSPORT_FAULT",
        }.get(str(proposal.get("why")), "NO_PROPOSAL")
        if out["outcome"] != "NO_PROPOSAL":
            out["why_this_is_not_an_abstention"] = (
                "the request never reached a model: this is a configuration "
                "receipt, not Slow declining to propose, and it is not "
                "retried and not substituted with another model")
        return out

    applied = know.apply_update(controller=controller, store=store,
                                snapshot=snapshot, manifest=proposal["manifest"],
                                catalog=catalog)
    out["applied"] = {k: v for k, v in applied.items() if k != "snapshot"}
    if not applied.get("applied"):
        out["outcome"] = "PROPOSAL_DID_NOT_COMPILE"
        return out

    withheld_ctx = units[VALIDATION_POSITION]
    withheld_population = population_for(withheld_ctx, "validation")
    withheld_features = {
        uid: binding.get(uid, withheld_ctx.origin)
        for uid in withheld_population}
    out["legacy_lines_diagnostic"] = know.legacy_lines(
        applicability=proposal.get("observable_applicability") or {},
        group=group, binding=binding,
        withheld_population=withheld_population,
        withheld_origin=int(withheld_ctx.origin),
        withheld_features=withheld_features)

    out["candidate_snapshot"] = applied["snapshot"]
    out["outcome"] = "CANDIDATE_COMPILED"
    written = proposal.get("written_text")
    current = next((row.get("current_value") for row in catalog
                    if row["surface_id"] == proposal["target_surface_id"]), None)
    probe, kind = _edit_probe(operation=proposal.get("operation"),
                              skill_id=proposal.get("skill_id"),
                              written=written, current=current)
    out["edit_probe"] = probe
    out["edit_probe_kind"] = kind
    out["why_the_probe_is_new_text_only"] = (
        "a PATCH usually keeps most of the value it replaces, so a probe cut "
        "from the front of the new text would also be found in the parent and "
        "the 'loaded in arm A' control would read as a false positive.  The "
        "probe is the first window of the new text that does not occur in the "
        "value it replaced")
    # Is the edit on screen for the decisions the arms are about to pay for?
    # Zero fits, zero LLM.  This is not a coverage gate -- it does not ask the
    # condition to explain most failures -- it asks only whether the thing
    # about to be measured is presented at all.  Paying for hours of paired
    # decisions that all run under the parent knowledge does not test the
    # guidance; it re-measures the parent.
    arm_windows = {
        str(position): {uid: binding.get(uid, units[position].origin)
                        for uid in population_for(units[position], "validation")}
        for position in (VALIDATION_POSITION, *FOLLOWUP_POSITIONS)}
    out["exposure"] = know.exposure_check(
        parent=snapshot, candidate=applied["snapshot"], probe=probe,
        windows=arm_windows, skill_id=proposal.get("skill_id") or "")
    if out["exposure"]["decisions_the_edit_reaches"] == 0:
        out["outcome"] = "CANDIDATE_COMPILED_BUT_NEVER_EXPOSED"
        out["why_the_arms_are_not_run"] = (
            "the compiled candidate reaches none of the %d registered arm "
            "decisions, so both arms would run under the parent knowledge and "
            "the comparison would re-measure the parent rather than test the "
            "guidance.  The candidate's utility is UNTESTED, not zero.  No "
            "threshold is adjusted to rescue it and no extra run is added"
            % out["exposure"]["decisions_checked"])
    out["parent_is_kept"] = {
        "parent_runtime_bundle_sha": applied["parent_runtime_bundle_sha"],
        "candidate_runtime_bundle_sha": applied["runtime_bundle_sha"],
        "old_knowledge_removed": [],
        "why": ("the candidate is a fork; the parent snapshot is untouched and "
                "arm A runs on it, so a failed candidate erases nothing"),
    }
    return out


# ---------------------------------------------------------------------------
# one group
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def _load_boundary(checkpoints: Any, machinery: Mapping[str, Any],
                   store: Any) -> dict[str, Any] | None:
    """A boundary already decided in an earlier process, with its fork.

    The candidate is re-compiled from the fork the earlier process wrote into
    this group's own snapshot store, so a resumed run uses the same knowledge
    version rather than asking Slow a second question.
    """
    path = checkpoints / "boundary.json"
    if not path.is_file():
        return None
    try:
        boundary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    boundary["resumed_from_checkpoint"] = True
    applied = boundary.get("applied") or {}
    sha = applied.get("runtime_bundle_sha")
    if not (boundary.get("outcome") == "CANDIDATE_COMPILED" and sha):
        return boundary
    for root in (store.root if hasattr(store, "root") else None,
                 getattr(store, "_root", None)):
        if root is None:
            continue
        candidate_dir = Path(root) / str(sha)
        if candidate_dir.is_dir():
            boundary["candidate_snapshot"] = machinery["compile_snapshot"](
                candidate_dir, verify_lock=False)
            return boundary
    boundary["outcome"] = "CANDIDATE_FORK_NOT_FOUND_ON_RESUME"
    boundary["why_resume_failed"] = (
        "the boundary was decided in an earlier process but its compiled fork "
        "is not in this group's snapshot store, so the candidate arm cannot "
        "be reconstructed without asking Slow a second question")
    return boundary


def build(*, group_name: str, budget: Budget, run_id: str = "",
          formation_limit: int | None = None,
          followup_limit: int | None = None,
          skip_smoke: bool = False) -> dict[str, Any]:
    from evaluation.main_protocol_p4 import smoke_dev_seq2 as smoke

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
        return {"stage": "DEV_SEQ2_SLOW_UPDATE_TO_FAST", "group": group_name,
                "status": "BLOCKED_BY_TRANSPORT", "transport": transport,
                "why": ("the fixed non-Flash model could not be confirmed; no "
                        "paid call is made and no substitute model is used")}

    preflight = ({"skipped": True} if (skip_smoke or RESUME_ONLY)
                 else smoke.run())
    if not (skip_smoke or RESUME_ONLY) and not preflight["passed"]:
        return {"stage": "DEV_SEQ2_SLOW_UPDATE_TO_FAST", "group": group_name,
                "status": "BLOCKED_BY_SMOKE", "smoke": preflight,
                "transport": transport}
    budget.spend_fits("smoke", int(preflight.get("consumer_fits") or 0))

    np.random.seed(int(plan["seed"]) % (2 ** 32))
    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population_rows, _excluded = R._population(doc)
    by_position = {row["position"]: row for row in population_rows}

    snapshot_dir, snapshot_source = R._resolve_k0_snapshot(doc, state["k0"])
    if snapshot_dir is None:
        return {"stage": "DEV_SEQ2_SLOW_UPDATE_TO_FAST", "group": group_name,
                "status": "BLOCKED", "why": "no readable K0 snapshot"}
    parent_snapshot = machinery["compile_snapshot"](snapshot_dir,
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

    outer_inner = machinery["agentic"]._default_backend_factory(OUTER_LLM_CAP)
    created_backends.append(outer_inner)
    outer_core = machinery["TTHAAgentCore"](
        runner._MeteredOuterBackend(outer_inner, guard=guard, billable=True),
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=transport["requested_model"], base_url=transport["base_url"])

    store = machinery["SnapshotStore"](
        base.ROOT / ".dev_seq2_runs" / (run_id or "run") / group_name / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    store.materialize(parent_snapshot)

    formation_positions = list(FORMATION_POSITIONS)
    if formation_limit is not None:
        formation_positions = formation_positions[: int(formation_limit)]
    followup_positions = list(FOLLOWUP_POSITIONS)
    if followup_limit is not None:
        followup_positions = followup_positions[: int(followup_limit)]

    units: dict[int, Any] = {}
    features_by_position: dict[int, dict[str, dict[str, Any]]] = {}
    origins: dict[int, int] = {}
    for position in ALL_POSITIONS:
        ctx = R.opp._unit_ctx(by_position[position]["unit"])
        units[position] = ctx
        origins[position] = int(ctx.origin)
        features_by_position[position] = {
            uid: ps.sequence_features(machinery, ctx, uid, origin=ctx.origin)
            for uid in ps.decision_uids(ctx, size=FORMATION_POPULATION)}
    binding = know.FeatureBinding(
        know.bind_window_features(features_by_position, origins))

    # one prediction cache for the whole group, shared by both arms
    cache = runner.ReplayPredictionCache("devseq2-%s" % group_name)
    ledger = know.MeasurementLedger(cache, runner.forecast_p4._config)
    unit_readings: dict[int, ps.UnitReadings] = {
        position: ps.UnitReadings(cache=cache, ctx=units[position],
                                  executor=units[position].executor,
                                  guard=budget.require)
        for position in ALL_POSITIONS}

    stopped_at: dict[str, Any] | None = None

    # ---- formation -------------------------------------------------------
    formation = Arm("formation", parent_snapshot, machinery=machinery,
                    backend_factory=backend_factory, guard=guard, cache=cache)
    formation_units: list[dict[str, Any]] = []
    for position in formation_positions:
        if budget.stop_reason():
            stopped_at = {"phase": "formation", "position": position,
                          "why": budget.stop_reason()}
            break
        unit = run_or_resume(arm=formation, ctx=units[position],
                             position=position, phase="formation",
                             readings=unit_readings[position],
                             machinery=machinery, budget=budget,
                             checkpoints=checkpoints,
                             name="formation_u%03d" % position)
        if unit is None:
            stopped_at = {"phase": "formation", "position": position,
                          "why": "NOT_RUN__RESUME_ONLY"}
            break
        formation_units.append(unit)
        if unit["stopped_at"]:
            stopped_at = {"phase": "formation", "position": position,
                          **unit["stopped_at"]}
            break

    formation_records = [row for unit in formation_units
                         for row in unit["sequences"]]

    # every measurement the formation segment actually produced, de-duplicated
    for row in formation_records:
        if isinstance(row.get("support_gain"), float):
            ledger.note(unit=row["unit"], origin=row["origin"],
                        steps=ps.normalise_steps(row.get("deployed") or ()),
                        uid=row["series_uid"], value=row["support_gain"],
                        face="support")
        if isinstance(row.get("delayed_gain"), float):
            ledger.note(unit=row["unit"], origin=row["delayed_origin"],
                        steps=ps.normalise_steps(row.get("deployed") or ()),
                        uid=row["series_uid"], value=row["delayed_gain"],
                        face="delayed")

    # ---- the batch boundary ---------------------------------------------
    saved_boundary = _load_boundary(checkpoints, machinery, store)
    if saved_boundary is not None:
        boundary = saved_boundary
    elif stopped_at:
        boundary = {"outcome": "NOT_REACHED", "why": stopped_at}
    else:
        try:
            boundary = knowledge_boundary(
                formation=formation, units=units, binding=binding,
                machinery=machinery, budget=budget, outer_core=outer_core,
                snapshot=parent_snapshot, store=store, controller=controller,
                decision_records=formation_records, group_name=group_name)
        except PackageCeiling as exc:
            boundary = {"outcome": "STOPPED_AT_THE_CEILING",
                        "why": str(exc)[:240]}
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            boundary = {"outcome": "BOUNDARY_FAULT",
                        "why": "%s: %s" % (type(exc).__name__, str(exc)[:400])}
    if saved_boundary is None:
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

    # ---- validation and follow-up, both arms ----------------------------
    arms: dict[str, Arm] = {}
    validation: dict[str, Any] = {"ran": False}
    adoption: dict[str, Any] = {}
    followup_units: list[dict[str, Any]] = []
    references: dict[str, Any] = {}

    if candidate is None:
        validation = {
            "ran": False,
            "why": (boundary.get("why_the_arms_are_not_run") if never_exposed
                    else "no legal candidate was compiled, so the two arms "
                    "would be the same snapshot and any difference between "
                    "them would be sampling noise, not a knowledge effect"),
            "no_candidate_was_written_by_hand": True,
            "exposure": (boundary.get("exposure") or {}).get(
                "decisions_the_edit_reaches") if never_exposed else None,
        }
        followup: dict[str, Any] = {
            "ran": False,
            "why": ("this group keeps its parent course: with no candidate "
                    "there is nothing to carry forward"),
        }
    else:
        for name, snap in ((ARM_A, parent_snapshot), (ARM_B, candidate)):
            arm = Arm(name, snap, machinery=machinery,
                      backend_factory=backend_factory, guard=guard, cache=cache)
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
            unit = run_or_resume(arm=arms[name], ctx=units[VALIDATION_POSITION],
                                 position=VALIDATION_POSITION,
                                 phase="validation",
                                 readings=unit_readings[VALIDATION_POSITION],
                                 machinery=machinery, budget=budget,
                                 checkpoints=checkpoints,
                                 name="validation_%s" % _slug(name),
                                 edit_probe=edit_probe)
            if unit is None:
                stopped_at = {"phase": "validation", "arm": name,
                              "why": "NOT_RUN__RESUME_ONLY"}
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

        branch_status = ("ADOPTED_DEVELOPMENT_BRANCH" if adoption.get("adoptable")
                         else "UNADOPTED_CANDIDATE_DIAGNOSTIC_BRANCH")
        followup = {"ran": False, "arm_order": list(order),
                    "branch_status": branch_status}
        for position in followup_positions:
            if budget.stop_reason():
                stopped_at = {"phase": "followup", "position": position,
                              "why": budget.stop_reason()}
                break
            if not RESUME_ONLY:
                references[str(position)] = reference_readings(
                    ctx=units[position],
                    population=population_for(units[position], "followup"),
                    readings=unit_readings[position], budget=budget)
            for name in order:
                unit = run_or_resume(
                    arm=arms[name], ctx=units[position], position=position,
                    phase="followup", readings=unit_readings[position],
                    machinery=machinery, budget=budget,
                    checkpoints=checkpoints,
                    name="followup_u%03d_%s" % (position, _slug(name)),
                    edit_probe=edit_probe)
                if unit is None:
                    stopped_at = {"phase": "followup", "position": position,
                                  "arm": name, "why": "NOT_RUN__RESUME_ONLY"}
                    continue
                followup_units.append(unit)
                if unit["stopped_at"]:
                    stopped_at = {"phase": "followup", "position": position,
                                  "arm": name, **unit["stopped_at"]}
            if str(position) in references:
                (checkpoints / ("reference_u%03d.json" % position)).write_text(
                    json.dumps(drafts._plain(
                        {"reference": references[str(position)],
                         "cost": budget.to_dict()}),
                        ensure_ascii=False, default=str), encoding="utf-8")
            if stopped_at:
                break
        followup["ran"] = bool(followup_units)
        followup["positions"] = followup_positions
        followup["why_it_ran_either_way"] = (
            "the candidate is carried into the follow-up units whether or not "
            "it passed the adoption mark, so 'the guidance has no value' can "
            "be told apart from 'the adoption rule did not pass it'.  An "
            "unadopted branch's readings are candidate diagnostics and are "
            "never counted as promotion or deployment gain")

    for row in followup_units:
        for seq in row["sequences"]:
            if isinstance(seq.get("delayed_gain"), float):
                ledger.note(unit=seq["unit"], origin=seq["delayed_origin"],
                            steps=ps.normalise_steps(seq.get("deployed") or ()),
                            uid=seq["series_uid"], value=seq["delayed_gain"],
                            face="delayed")

    finished = datetime.now(timezone(timedelta(hours=8)))
    result = {
        "stage": "DEV_SEQ2_SLOW_UPDATE_TO_FAST",
        "status": "COMPLETE" if stopped_at is None else "STOPPED",
        "line": LINE,
        "what_this_line_is": (
            "the package's main line, on the non-Flash model the work order "
            "requires" if LINE == "main" else
            "a separate speed probe on an excluded (Flash-family) model, run "
            "at the user's explicit request.  A different model is a "
            "different experiment: these readings are reported on their own "
            "and are never tabled with, subtracted from or averaged into the "
            "main line's"),
        "group": group_name,
        "run_id": run_id or None,
        "question": ("does one Slow edit to shared guidance, chosen by Slow "
                     "itself from real success/failure feedback, change what "
                     "the next batch of per-sequence Fast does -- and is the "
                     "change useful"),
        "registered_plan": {
            "formation_positions": formation_positions,
            "validation_position": VALIDATION_POSITION,
            "followup_positions": followup_positions,
            "origins": {str(p): origins[p] for p in ALL_POSITIONS},
            "delayed_offset": ps.DELAYED,
            "seed": plan["seed"],
            "arm_order": list(plan["arm_order"]),
            "arms": {ARM_A: "the parent snapshot, knowledge frozen",
                     ARM_B: "the compiled Slow candidate"},
            "decision_population_rule": (
                "the first N eval series of the block in fixed roster order: "
                "%d in formation, %d for each arm.  Fixed before the run on "
                "measured cost, never on a gain, and A and B always decide "
                "for exactly the same set"
                % (FORMATION_POPULATION, ARM_POPULATION)),
            "formation_population": FORMATION_POPULATION,
            "arm_population": ARM_POPULATION,
            "why_the_arm_population_is_smaller": (
                "the fixed non-Flash model costs 142-379 wall seconds per "
                "per-sequence decision, measured on a shakedown.  Ten "
                "sequences everywhere needs about eleven hours for the two "
                "groups and could not finish inside the six-hour ceiling.  "
                "The formation population is kept at ten so the boundary has "
                "enough material to group at all"),
            "group_wall_allowance_seconds": GROUP_WALL_SECONDS,
            "min_group": MIN_GROUP,
            "registered_before_the_run": True,
            "not_chosen_on_a_gain": True,
            "all_positions_are_already_exposed_development": True,
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
            "one_model_for_the_whole_package": True,
        },
        "k0_snapshot": {"dir": str(snapshot_dir), "source": snapshot_source,
                        "runtime_bundle_sha": parent_snapshot.runtime_bundle_sha,
                        "capability_cards": formation.capability_cards()},
        "admission_policy": machinery["admission_policy"].active_policy().to_dict(),
        "feature_binding": binding.to_dict(),
        "measurement_ledger": ledger.to_dict(),
        "formation": {"units": formation_units,
                      "episodes_written": len(formation.episodes)},
        "boundary": {k: v for k, v in boundary.items()
                     if k != "candidate_snapshot"},
        "treatment": treatment,
        "validation": validation,
        "adoption": adoption,
        "followup": followup,
        "followup_units": followup_units,
        "reference_readings": references,
        "stopped_at": stopped_at,
        "early_termination": ({
            "terminated_early": True,
            "reason": EARLY_STOP,
            "what_was_kept": (
                "every checkpoint the group actually paid for is in this "
                "artifact: the formation units, the batch boundary and any "
                "arm unit that completed.  Nothing is deleted and no cost is "
                "written off"),
            "what_was_not_run": (
                "the remaining paired arm decisions of the registered plan"),
            "this_is_not_a_complete_experiment": (
                "the guidance's utility is UNTESTED, not measured as zero.  "
                "This group must not be presented as a completed two-arm "
                "comparison, and a later package must not quote only the "
                "groups whose edit did reach Fast"),
        } if EARLY_STOP else None),
        "cost": budget.to_dict(),
        "smoke": preflight,
        "what_this_is_not": (
            "development level, one course, already-exposed units.  Two groups "
            "are random repeats on the same data, not independent "
            "generalisation evidence.  No significance is claimed and no "
            "production deployment permission is granted."),
    }
    (checkpoints / "result.json").write_text(
        json.dumps(drafts._plain(result), ensure_ascii=False, default=str),
        encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=sorted(GROUPS),
                        help="run one group; repeat for more (default: both)")
    parser.add_argument("--run", default="", help="run id for checkpoints")
    parser.add_argument("--formation-limit", type=int, default=None)
    parser.add_argument("--followup-limit", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None,
                        help="formation population override (testing only)")
    parser.add_argument("--arm-population", type=int, default=None,
                        help="arm population override (testing only)")
    parser.add_argument("--min-group", type=int, default=None,
                        help="minimum group size (testing only)")
    parser.add_argument("--fit-ceiling", type=int, default=None,
                        help="this process's share of the package fit ceiling")
    parser.add_argument("--llm-ceiling", type=int, default=None,
                        help="this process's share of the package LLM ceiling")
    parser.add_argument("--group-wall", type=int, default=None,
                        help="wall seconds allowed per group")
    parser.add_argument("--model", default=None,
                        help="the one model this package is fixed to")
    parser.add_argument("--resume-only", action="store_true",
                        help="assemble the artifact from existing checkpoints "
                             "and run nothing new (zero fits, zero LLM)")
    parser.add_argument("--early-stop", default="",
                        help="why this group was stopped before the plan ended")
    parser.add_argument("--allow-flash", action="store_true",
                        help="permit an excluded Flash-family model for an "
                             "explicitly labelled separate speed probe")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    global FORMATION_POPULATION, ARM_POPULATION, MIN_GROUP
    global MAX_FITS, MAX_LLM, GROUP_WALL_SECONDS, REQUIRED_MODEL
    global ALLOW_EXCLUDED_MODEL, LINE, RESUME_ONLY, EARLY_STOP
    if args.resume_only:
        RESUME_ONLY = True
    if args.early_stop:
        EARLY_STOP = str(args.early_stop)
    if args.model:
        REQUIRED_MODEL = str(args.model)
    if args.allow_flash:
        ALLOW_EXCLUDED_MODEL = True
        LINE = "fast_probe"
    if args.min_group is not None:
        MIN_GROUP = int(args.min_group)
    if args.fit_ceiling is not None:
        MAX_FITS = int(args.fit_ceiling)
    if args.llm_ceiling is not None:
        MAX_LLM = int(args.llm_ceiling)
    if args.group_wall is not None:
        GROUP_WALL_SECONDS = int(args.group_wall)
    if args.population_size is not None:
        FORMATION_POPULATION = int(args.population_size)
    if args.arm_population is not None:
        ARM_POPULATION = int(args.arm_population)
    ARM_POPULATION = min(ARM_POPULATION, FORMATION_POPULATION)
    groups = args.group or sorted(GROUPS)
    budget_path = (CHECKPOINTS / (args.run or "run") / "package_budget.json")
    if budget_path.is_file():
        budget = Budget.load(budget_path)
        print("resumed package budget from %s: fits=%d llm=%d wall=%.0fs"
              % (budget_path, budget.fits(), budget.llm(), budget.elapsed()))
    else:
        budget = Budget()

    suffix = ("_" + args.suffix.strip("_")) if args.suffix else ""
    for group_name in groups:
        result = build(group_name=group_name, budget=budget, run_id=args.run,
                       formation_limit=args.formation_limit,
                       followup_limit=args.followup_limit,
                       skip_smoke=args.skip_smoke)
        budget.save(budget_path)
        out = ART / ("dev_seq2_slow_update_to_fast__%s%s.json"
                     % (group_name, suffix))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(drafts._plain(result), ensure_ascii=False,
                                  indent=1, default=str), encoding="utf-8")
        cost = result.get("cost") or {}
        print("%s %s | fits %s/%s | llm %s/%s | %ss"
              % (result.get("status"), group_name,
                 cost.get("physical_consumer_fits"), cost.get("fit_ceiling"),
                 cost.get("llm_calls"), cost.get("llm_ceiling"),
                 cost.get("wall_seconds")))
        boundary = result.get("boundary") or {}
        print("   boundary: %s | surface: %s | treatment: %s"
              % (boundary.get("outcome"),
                 (boundary.get("proposal") or {}).get("target_surface_id"),
                 result.get("treatment")))
        adoption = result.get("adoption") or {}
        if adoption:
            print("   adoption: %s | failed: %s | paired delta: %s"
                  % (adoption.get("adoptable"), adoption.get("failed_lines"),
                     (adoption.get("paired") or {}).get(
                         "whole_population_difference")))
        print("   wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
