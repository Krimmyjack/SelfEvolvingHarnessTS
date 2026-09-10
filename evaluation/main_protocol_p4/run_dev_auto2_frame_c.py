"""DEV-AUTO-2: a live Frame C revision course on the repaired wiring.

The question
-----------
Can a modification a real LLM proposes get through *later verification*, update
the Skill, and change what a later Fast actually does and delivers?

DEV-AUTO-1 could not answer it: its comparison used two different denominators,
and three wiring defects (a control frozen in four extra ways, a five-call cap
that silently deleted four units from the arm's own history, and Fast running
under the strict admission rule) sat under every number.  DEV-AUTO-1R fixed the
comparison and the wiring but did not re-run anything.  This package runs it.

What is different from DEV-AUTO-1, and what is not
--------------------------------------------------
Different -- **the acceptance rule is Frame C**:

* legality and Scope calibration still gate, through the same tools;
* the historical replay is no longer an absolute veto.  It is evidence: it
  ranks the queue and it goes back to Slow as feedback.

Everything downstream is the production path, unchanged: a queued candidate
becomes a rights-less ``RestrictedDraft``, is *offered* to later units through
``resupplied_programs_for_verification``, and only Fast decides whether to probe
or deploy it.  Support admission, the authoritative delayed gate, verification
counting, closure and promotion are the same calls the HEC-1 runner makes.  The
Runner never selects a winner and never deploys one.

Not different: the risk lines, the risk denominator, the frozen Scope grid, the
Consumer, the data boundary, and the arms' symmetry -- the two arms differ in
exactly one thing, whether the structural revision channel is open, and each
arm's revision spending is billed to that arm.

DEV-AUTO-1R's findings are **not** fed to the model.  No known-good program is
supplied, named, or hinted at in the role instruction, the feedback or the
operator table.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_auto2"

PACKAGE = "DEV-AUTO-2"
FROZEN_ARM = R.FROZEN_ARM
REVISING_ARM = R.REVISING_ARM
K0_SKILL_ID = R.K0_SKILL_ID

#: Outcomes of ``rev.evaluate_proposal`` that Frame C still treats as refusals.
#: These are legality and calibration, which every frame shares.
SHARED_REFUSALS = ("MALFORMED", "UNKNOWN_KIND", "ILLEGAL_PROGRAM", "NO_CHANGE",
                   "SLOW_CLAUSE_UNUSABLE", "SLOW_ABSTAINED",
                   "NO_FEASIBLE_THRESHOLD", "NARROWING_REFUSED")

#: Outcomes that were an absolute veto under the historical screen and are
#: evidence-only here.  Kept as a list rather than "everything else" so a new
#: outcome class lands in the refusal branch by default.
DEMOTED_TO_EVIDENCE = ("RISK_LINE_FAILED", "NOT_BETTER_THAN_THE_PARENT",
                       "NOT_READABLE", "ACCEPTED_FOR_VERIFICATION")

FRAME_C_RULE = (
    "Frame C.  A proposal that is legal, and whose Scope clause the tool could "
    "calibrate, enters the verification queue.  The replay over the units this "
    "arm has already processed is measured and reported, but it does not "
    "refuse the candidate: it orders the queue and it comes back to you as "
    "feedback.  Entering the queue is not acceptance and carries no deployment "
    "right -- the candidate is only offered to later units, Fast decides "
    "whether to probe or deploy it, and the Skill is patched only after an "
    "independent later unit passes the authoritative delayed gate."
)

CANDIDATE_ORDER = (
    "Queued candidates are ordered by, in this order: (1) the common-"
    "denominator utility difference against the parent over the units already "
    "processed, larger first; (2) fewer cells where the pair is confirmed not "
    "executable and falls back to raw, because a program that cannot run on "
    "the cells already seen is less likely to run on the next one; (3) the "
    "position it was proposed at, then the order the model returned it.  Every "
    "input is a reading on evidence that has already arrived; no later unit, "
    "no later face and no future score enters the key."
)


class Ceiling(RuntimeError):
    """The package's own ceiling, refused before the spend."""


# ---------------------------------------------------------------------------
# the comparator, wired into the live step
# ---------------------------------------------------------------------------

def _course_index(population: Sequence[Mapping[str, Any]]) -> dict[tuple, int]:
    index = {(row["unit"]["block"], row["unit"]["origin"]): row["position"]
             for row in population}
    assert len(index) == len(population), "unit keys are not unique"
    return index


def _cells_by_position(cells: Sequence[Mapping[str, Any]],
                       index: Mapping[tuple, int]) -> dict[int, Any]:
    return {index[(cell["unit"]["block"], cell["unit"]["origin"])]: cell
            for cell in cells}


def compare_to_the_parent(candidate_cells: Sequence[Mapping[str, Any]],
                          parent_cells: Sequence[Mapping[str, Any]],
                          *, index: Mapping[tuple, int],
                          by_position: Mapping[int, Mapping[str, Any]],
                          candidate_label: str) -> dict[str, Any]:
    """One comparison function, over one declared unit set, three states kept.

    The declared set is every cell the screen ran over -- including the ones a
    side could not read, which the comparator scores as the raw fallback they
    actually deliver, and *not* including any cell nobody measured, which it
    marks UNKNOWN and refuses to average.
    """
    mine = _cells_by_position(candidate_cells, index)
    theirs = _cells_by_position(parent_cells, index)
    positions = sorted(set(mine) | set(theirs))
    units = [{"position": p, "unit": by_position[p]["unit"]} for p in positions]
    return rev.compare_on_a_common_denominator(
        units=units,
        candidate={"pos%d" % p: rev.classify_reading(mine.get(p))
                   for p in positions},
        reference={"pos%d" % p: rev.classify_reading(theirs.get(p))
                   for p in positions},
        face="support_face", candidate_label=candidate_label,
        reference_label="the parent program on the Skill card",
        comparison_kind=("parent PROGRAM replay over the units already "
                         "processed -- evidence, not a veto, under Frame C"))


def ordering_key(verdict: Mapping[str, Any]) -> tuple:
    """The declared order, computed only from evidence already in hand."""
    comparison = verdict.get("common_denominator_comparison") or {}
    gain = (comparison.get("mean_aggregate_gain") or {}).get("difference")
    states = comparison.get("candidate_cell_states") or {}
    return (-(float(gain) if gain is not None else -9.0),
            int(states.get(rev.RAW_FALLBACK, 0) or 0)
            + int(states.get(rev.UNKNOWN, 0) or 0))


def evaluate_frame_c(proposal: Mapping[str, Any], *, parent_steps, parent_scope,
                     root_scope, bank_rows, replay, parent_summary,
                     parent_cells, index, by_position) -> dict[str, Any]:
    """``rev.evaluate_proposal``'s legality and calibration, Frame C's verdict.

    The historical screen is still computed and still reported in full -- what
    it would have refused is recorded verbatim under
    ``the_absolute_screen_would_have_said``, so the frame change is auditable
    rather than merely asserted.  It just does not decide.
    """
    verdict = dict(rev.evaluate_proposal(
        proposal, parent_steps=parent_steps, parent_scope=parent_scope,
        root_scope=root_scope, bank_rows=bank_rows, replay=replay,
        parent_summary=parent_summary))
    outcome = str(verdict.get("outcome"))
    verdict["frame"] = "C"
    if outcome in SHARED_REFUSALS:
        verdict["frame_c_outcome"] = outcome
        verdict["why_frame_c_refused"] = (
            "legality and Scope calibration gate in every frame; Frame C "
            "changes only what the historical replay may do")
        return verdict
    if outcome not in DEMOTED_TO_EVIDENCE:
        verdict["frame_c_outcome"] = outcome        # a ceiling or a fault
        return verdict

    cells = (verdict.get("screen") or {}).get("per_cell") or ()
    verdict["common_denominator_comparison"] = compare_to_the_parent(
        cells, parent_cells, index=index, by_position=by_position,
        candidate_label=str(verdict.get("program_replayed")))
    verdict["the_absolute_screen_would_have_said"] = {
        "outcome": outcome,
        "failed_lines": verdict.get("failed_lines"),
        "why": verdict.get("why"),
        "note": "recorded, not acted on -- Frame C does not veto on it",
    }
    verdict["frame_c_outcome"] = "QUEUED_FOR_VERIFICATION"
    verdict["why"] = FRAME_C_RULE
    return verdict


def build_feedback_frame_c(**kwargs: Any) -> dict[str, Any]:
    """DEV-AUTO-1's whole-record feedback, with Frame C's rule in place of the
    absolute screen's, so what Slow is told matches what the runtime does."""
    feedback = rev.build_feedback(**kwargs)
    judged = dict(feedback["how_a_proposal_is_judged"])
    judged.update({
        "the_rule": FRAME_C_RULE,
        "queue_order": CANDIDATE_ORDER,
        "what_the_replay_does": ("it measures your proposal on the units this "
                                 "arm has already processed and reports the "
                                 "result to you; it does not refuse the "
                                 "proposal"),
        "what_still_refuses": ("an illegal program, a program identical to the "
                               "one already deployed, and a Scope clause the "
                               "calibration tool cannot place on a frozen bin "
                               "edge that clears the four lines"),
        "denominator": ("the full served population of every unit, over one "
                        "unit set for both sides.  A unit where the pair "
                        "cannot execute scores its raw fallback at exactly 0 "
                        "and stays in the denominator; a unit nobody measured "
                        "is reported as unknown and is not averaged"),
    })
    judged.pop("accepted_only_at", None)
    feedback["how_a_proposal_is_judged"] = judged
    return feedback


def attempt_row(position: int, proposal: Mapping[str, Any],
                verdict: Mapping[str, Any]) -> dict[str, Any]:
    """What the next call is told: the common-denominator reading, three states."""
    row = R._attempt_row(position, proposal, verdict)
    row["outcome"] = verdict.get("frame_c_outcome") or verdict.get("outcome")
    comparison = verdict.get("common_denominator_comparison")
    if comparison:
        gain = comparison["mean_aggregate_gain"]
        worst = comparison["cross_unit_worst"]
        row["measured_on_the_processed_units"] = {
            "denominator": comparison["denominator_rule"],
            "units_declared": comparison["units_declared"],
            "your_cells": comparison["candidate_cell_states"],
            "your_mean_gain": gain["candidate"],
            "the_parent_delivers": gain["reference"],
            "difference": gain["difference"],
            "your_worst_single_series_harm":
                worst["single_series_harm"]["candidate"],
            "the_parents": worst["single_series_harm"]["reference"],
            "your_worst_harmed_fraction":
                worst["harmed_fraction"]["candidate"],
            "the_parents_harmed_fraction":
                worst["harmed_fraction"]["reference"],
            "units_where_you_are_worse_than_the_parent":
                comparison["per_unit_risk"]["units_where_the_candidate_is_worse"],
            "reading_is_complete": comparison["verdict_is_complete"],
        }
        row["this_did_not_refuse_you"] = (
            "under this rule the replay orders the queue; it does not veto")
    return row


# ---------------------------------------------------------------------------
# one revision step
# ---------------------------------------------------------------------------

def revision_step(arm: Any, *, position: int, origin: int, proposer,
                  history: Sequence[Mapping[str, Any]],
                  attempts: list[dict[str, Any]], ledgers: runner.Ledgers,
                  budget: R.PackageBudget, metered: Any,
                  pending: dict[str, Any], index: Mapping[tuple, int],
                  by_position: Mapping[int, Mapping[str, Any]],
                  case_prefix: str = "devauto2",
                  ) -> dict[str, Any]:
    step: dict[str, Any] = {"position": int(position), "origin": int(origin),
                            "arm": arm.spec.name, "frame": "C"}
    if not arm.processed:
        step["outcome"] = "NO_PROCESSED_UNITS"
        return step

    parent_steps = R.current_skill_steps(arm)
    root_scope = dict(R.ANCESTOR_SCOPE)
    draft = arm.draft_ledger.by_program_and_root(parent_steps, root_scope)
    current_scope = dict(draft.current_scope) if draft is not None \
        else dict(root_scope)

    screen = runner.replay_screen_for(arm.processed, ledgers, arm.replay_cache)

    def guarded_replay(*, steps, scope):
        need = runner.CACHE_FITS_PER_CELL * len(arm.processed)
        if not budget.room_for(need):
            raise R.PackageCeiling(
                "one screen over %d processed cells could take the package "
                "past %d fits" % (len(arm.processed), R.MAX_FITS))
        before = arm.replay_cache.physical_fits
        out = screen(steps=steps, scope=scope)
        budget.spend_fits(arm.spec.name,
                          arm.replay_cache.physical_fits - before)
        return out

    try:
        parent_screen = guarded_replay(steps=parent_steps, scope=current_scope)
    except R.PackageCeiling as exc:
        step["outcome"] = "STOPPED_AT_THE_FIT_CEILING"
        step["why"] = str(exc)
        return step
    parent_cells = list(parent_screen.get("cells") or ())
    parent = {**rev._screen_summary(parent_cells),
              "consumer_fits_spent": int(parent_screen.get("fits") or 0)}
    step["parent_policy"] = {
        "program": rev.program_label(parent_steps),
        "serving_scope": current_scope,
        "screen": parent,
        "per_cell_states": {
            state: sum(1 for cell in parent_cells
                       if rev.classify_reading(cell)["state"] == state)
            for state in (rev.READ, rev.RAW_FALLBACK, rev.UNKNOWN)},
        "the_parents_own_production_screen_violations": rev._risk_lines(parent),
        "the_parent_passes_its_own_screen": not rev._risk_lines(parent),
        "read_from": "the Skill card as it stands at this step",
    }

    group = next((g for g in outer_loop.census(arm.bank)
                  if g["program_signature"] == rev.program_label(parent_steps)),
                 None)
    bank_rows = list(group["rows"]) if group else []
    step["bank_rows_for_calibration"] = len(bank_rows)

    feedback = build_feedback_frame_c(
        skill_id=K0_SKILL_ID, program_steps=parent_steps,
        serving_scope=current_scope, history=history, attempts=attempts,
        vocabulary=contract.SCOPE_CLASS["vocabulary"])
    step["feedback_sent"] = {
        "units_sent": feedback["units_sent"],
        "units_available": feedback["units_available"],
        "nothing_was_truncated": feedback["nothing_was_truncated"],
        "earlier_attempts_included": len(attempts),
        "carries_the_frame_c_rule": True,
    }

    llm_before = int(getattr(metered, "calls", 0) or 0)
    proposer.snapshot = arm.active_snapshot()
    # The case id separates sessions.  DEV-AUTO-3 gives each arm its own
    # prefix so two arms asking about the same unit are two conversations.
    record = proposer(feedback, case_id="%s-u%03d" % (case_prefix, position))
    budget.spend_llm(arm.spec.name,
                     int(getattr(metered, "calls", 0) or 0) - llm_before)
    step["proposer"] = {k: v for k, v in record.items() if k != "proposals"}
    step["proposals"] = list(record.get("proposals") or ())

    evaluations: list[dict[str, Any]] = []
    for proposal in step["proposals"]:
        try:
            verdict = evaluate_frame_c(
                proposal, parent_steps=parent_steps,
                parent_scope=current_scope, root_scope=root_scope,
                bank_rows=bank_rows, replay=guarded_replay,
                parent_summary=parent, parent_cells=parent_cells,
                index=index, by_position=by_position)
        except R.PackageCeiling as exc:
            verdict = {"kind": proposal.get("kind"), "frame": "C",
                       "outcome": "STOPPED_AT_THE_FIT_CEILING",
                       "frame_c_outcome": "STOPPED_AT_THE_FIT_CEILING",
                       "why": str(exc)}
        verdict["_proposal"] = proposal
        evaluations.append(verdict)

    # The declared order, applied.  Drafts are opened in this order, so it is
    # also the order later units are offered them in -- the resupply mapping
    # preserves insertion order.
    queued = [v for v in evaluations
              if v.get("frame_c_outcome") == "QUEUED_FOR_VERIFICATION"]
    queued.sort(key=ordering_key)
    step["queue_order_applied"] = [
        {"program": v.get("program_replayed"), "kind": v.get("kind"),
         "key": list(ordering_key(v))} for v in queued]

    for verdict in evaluations:
        attempts.append(attempt_row(position, verdict["_proposal"], verdict))
        verdict["attempt_row_index"] = len(attempts) - 1

    for verdict in queued:
        proposal = verdict["_proposal"]
        if verdict["kind"] == "scope_clause":
            applied = rev.apply_scope_revision(
                ledger=arm.draft_ledger, draft=draft,
                new_scope=verdict["scope_replayed"],
                narrowing_preflight=verdict.get("narrowing_preflight"),
                support=verdict["screen"]["summary"], origin=origin)
            verdict["written_back"] = applied
            if applied.get("applied"):
                applied["channel"] = ("the lineage's RestrictedDraft; later "
                                      "units are offered the revised "
                                      "predicate through resupplied_scopes")
        else:
            steps = tuple((str(s["op"]), dict(s.get("params") or {}))
                          for s in verdict["steps_replayed"])
            key = outer_loop.census_key(
                runner.TASK_CONSUMER_KEY,
                [{"op": op, "params": dict(p)} for op, p in steps], root_scope)
            try:
                new_draft = arm.draft_ledger.open_restricted(
                    program_steps=steps, root_scope=root_scope,
                    current_scope=current_scope, origin=int(origin),
                    census_key=key,
                    provenance={"opened_by": "dev_auto2_frame_c",
                                "position": int(position),
                                "queue_key": list(ordering_key(verdict)),
                                "what_changes": proposal.get("what_changes")})
                verdict["written_back"] = {
                    "applied": True, "draft_id": new_draft.draft_id,
                    "channel": ("a rights-less Draft; later units are offered "
                                "it through resupplied_programs_for_"
                                "verification and Fast decides"),
                }
                pending[rev.program_label(steps)] = {
                    "steps": steps, "draft_id": new_draft.draft_id,
                    "proposed_at_position": int(position),
                    "what_changes": proposal.get("what_changes"),
                }
            except Exception as exc:  # noqa: BLE001 - the lifecycle refusing
                verdict["written_back"] = {
                    "applied": False,
                    "why": "%s: %s" % (type(exc).__name__, str(exc)[:240]),
                    "note": ("the lineage already has a Draft; it is not "
                             "reopened and no counter is zeroed"),
                }
        attempts[verdict["attempt_row_index"]]["written_back"] = \
            verdict.get("written_back")

    for verdict in evaluations:
        verdict.pop("_proposal", None)
        verdict.pop("attempt_row_index", None)
    step["evaluations"] = evaluations
    step["outcome"] = (record.get("outcome") or "UNKNOWN")
    step["queued"] = len(queued)
    step["drafts_opened"] = sum(
        1 for v in queued if (v.get("written_back") or {}).get("applied"))
    return step


# ---------------------------------------------------------------------------
# the course
# ---------------------------------------------------------------------------

def build(*, limit: int | None = None, label: str = "run1") -> dict[str, Any]:
    from evaluation.main_protocol_p4 import smoke_dev_auto2_frame_c as smoke

    started = datetime.now(timezone(timedelta(hours=8)))
    preflight = smoke.run()
    if not preflight["passed"]:
        return {"stage": "DEV_AUTO2_FRAME_C", "status": "BLOCKED_BY_SMOKE",
                "smoke": preflight}

    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population, excluded = R._population(doc)
    index = _course_index(population)
    by_position = {row["position"]: row for row in population}
    after = [row for row in population if row["side"] == "after_the_boundary"]
    if limit is not None:
        after = after[:int(limit)]

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    snapshot_dir, snapshot_source = R._resolve_k0_snapshot(doc, state["k0"])
    if snapshot_dir is None:
        return {"stage": "DEV_AUTO2_FRAME_C", "status": "BLOCKED",
                "why": "no readable K0 snapshot"}
    start_snapshot = machinery["compile_snapshot"](snapshot_dir,
                                                   verify_lock=False)

    budget = R.PackageBudget()
    ledgers = runner.Ledgers()
    guard = runner.BudgetGuard(ordering_cap=R.MAX_LLM,
                               per_unit_arm_cap=R.PER_ARM_LLM, ledgers=ledgers)
    root = base.ROOT / ".dev_auto2_runs" / label

    def backend_factory():
        inner = machinery["agentic"]._default_backend_factory(R.PER_ARM_LLM)
        return runner._MeteredFastBackend(inner, guard=guard, billable=True)

    # One difference between the arms and no other: whether ``revision_step``
    # is offered.  Both carry their own history, Draft ledger and lifecycle.
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
        R._seed_arm(arm, state, fresh_ledger=True)
        # Bootstrap the arm's own store with the snapshot it starts on.
        #
        # ``Arm._build`` opens an empty ``SnapshotStore`` and hands the method a
        # snapshot compiled from a directory outside it, so the store has never
        # seen the bundle the arm is running.  ``activate_approved`` then calls
        # ``store.set_active(snap.runtime_bundle_sha)`` and the store refuses:
        # "cannot activate an unmaterialized runtime bundle".  That happens on
        # the ordinary path where Fast deploys the K0 card it already has and
        # the delayed face confirms it -- ``route="deployed_existing_skill"``.
        # DEV-AUTO-1 never reached it, because under the strict admission rule
        # almost nothing was deployed and no delayed event was ever approved.
        #
        # ``materialize`` writes the tree the store already implies and is
        # idempotent (a byte-different collision raises).  It grants nothing:
        # the snapshot being made activatable is the one the arm is already
        # running.  ``_build`` is hoisted out of the first ``begin_unit`` with
        # the same tag that call would use for a carrying arm, so the sequence
        # is otherwise unchanged.
        arm._build("online")
        arm._store.materialize(arm.active_snapshot())
        arms[name] = arm

    outer_inner = machinery["agentic"]._default_backend_factory(R.MAX_LLM)
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
    revision_steps: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    pending: dict[str, Any] = {}
    promotions: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    stopped_at = None
    consecutive_faults = 0
    completed: list[int] = []

    for row in after:
        reason = budget.stop_reason()
        if reason:
            stopped_at = {"position": row["position"], "why": reason}
            break
        ctx = R.opp._unit_ctx(row["unit"])
        by_arm_cell = {}
        try:
            for name in (FROZEN_ARM, REVISING_ARM):
                cell = R.run_cell(arms[name], ctx, position=row["position"],
                                  ledgers=ledgers, guard=guard,
                                  machinery=machinery, budget=budget,
                                  seed=None)
                by_arm_cell[name] = cell
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
        revising = by_arm_cell[REVISING_ARM]
        history.append({
            "position": row["position"], "unit": row["unit"],
            "deployed_label": revising.get("deployed_label"),
            "deployed_scope": revising.get("deployed_scope"),
            "support_reading": revising.get("support_reading"),
            "delayed_reading": revising.get("delayed_reading"),
            "delayed_gate": revising.get("delayed_gate"),
        })

        # Promotion.  Fast chose to deploy this program and an independent
        # later unit's authoritative delayed gate passed; only then is the
        # Skill card patched.  The Runner never forced the choice.
        label_now = revising.get("deployed_label")
        if (label_now in pending
                and bool((revising.get("delayed_gate") or {}).get("passes"))):
            entry = pending.pop(label_now)
            arm = arms[REVISING_ARM]
            applied = rev.apply_workflow_revision(
                controller=arm._controller, store=arm._store,
                snapshot=arm.active_snapshot(), skill_id=K0_SKILL_ID,
                steps=entry["steps"],
                edit_id="dev-auto2-rev-u%03d" % row["position"])
            promotion = {k: v for k, v in applied.items()
                         if k not in ("snapshot", "materialized")}
            promotion.update({
                "verified_at_position": row["position"],
                "proposed_at_position": entry["proposed_at_position"],
                "program": label_now, "independent_unit": True,
                "draft_id": entry["draft_id"],
                "skill_bodies_before": R._skill_bodies(arm.active_snapshot()),
            })
            if applied.get("applied"):
                arm._method._snapshot = applied["snapshot"]
                arm._store.set_active(applied["candidate_runtime_bundle_sha"])
                promotion["skill_bodies_after"] = R._skill_bodies(
                    arm.active_snapshot())
                draft = arm.draft_ledger.by_id(entry["draft_id"])
                if draft is not None:
                    arm.draft_ledger.close(draft,
                                           "PROMOTED_TO_THE_SKILL_CARD")
                    promotion["draft_closed_as"] = draft.closed
            promotions.append(promotion)

        if not budget.stop_reason():
            try:
                step = revision_step(
                    arms[REVISING_ARM], position=row["position"],
                    origin=int(row["unit"]["origin"]), proposer=proposer,
                    history=history, attempts=attempts, ledgers=ledgers,
                    budget=budget, metered=outer_metered, pending=pending,
                    index=index, by_position=by_position)
            except R.PackageCeiling as exc:
                step = {"position": row["position"],
                        "outcome": "STOPPED_AT_THE_FIT_CEILING",
                        "why": str(exc)}
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                step = {"position": row["position"], "outcome": "STEP_FAULT",
                        "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
            revision_steps.append(step)
            if step.get("outcome") in ("PROPOSER_FAULT", "STEP_FAULT"):
                consecutive_faults += 1
            else:
                consecutive_faults = 0
            if consecutive_faults >= 3:
                stopped_at = {"position": row["position"],
                              "why": "PROPOSER_UNREACHABLE",
                              "detail": step.get("why")}
                break

        (CHECKPOINTS / ("%s_progress.json" % label)).write_text(
            json.dumps(drafts._plain({
                "completed_positions": completed, "cost": budget.to_dict(),
                "revision_steps": revision_steps, "promotions": promotions,
                "cells": cells}), ensure_ascii=False, default=str),
            encoding="utf-8")

    closed = {name: arms[name].draft_ledger.close_unreencountered()
              for name in arms}

    by_arm = {name: [c for c in cells if c["arm"] == name] for name in arms}
    common = sorted(set(c["position"] for c in by_arm[FROZEN_ARM])
                    & set(c["position"] for c in by_arm[REVISING_ARM]))
    scored = {name: [c for c in by_arm[name] if c["position"] in common]
              for name in by_arm}

    return {
        "stage": "DEV_AUTO2_FRAME_C",
        "package": ("DEV-AUTO-2: a live Frame C revision course on the "
                    "repaired wiring, development level"),
        "what_this_is_not": [
            "not the formal A3/A5 main experiment",
            "development level only: every unit here is already exposed",
            "no sealed data, no TARGET_HELD_IN unit and no +144 face was read",
            "no known-good program was supplied to or hinted at the model",
        ],
        "acceptance_rule": {"frame": "C", "rule": FRAME_C_RULE,
                            "queue_order": CANDIDATE_ORDER,
                            "shared_refusals": list(SHARED_REFUSALS),
                            "demoted_to_evidence": list(DEMOTED_TO_EVIDENCE)},
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "smoke": preflight,
        "transport": {k: v for k, v in transport.items() if k != "api_key"},
        "admission_rule_in_force": bounded.BOUNDED_POLICY.to_dict(),
        "geometry": {
            "ordering": R.ORDERING, "source_arm": R.SOURCE_ARM,
            "units_planned": [r["position"] for r in after],
            "units_completed_by_both_arms": common,
            "excluded_units": [
                {"position": r["position"], "why": r["why"]} for r in excluded],
            "entry_state": "M-R0d k1 reconstruction, both arms, once",
            "k0_snapshot_source": snapshot_source,
        },
        "stopped_at": stopped_at,
        "per_arm": {name: R._arm_summary(rows) for name, rows in scored.items()},
        "cells": cells,
        "revision_steps": revision_steps,
        "attempt_log": attempts,
        "promotions_to_the_skill_card": promotions,
        "pending_never_verified": {label: {k: v for k, v in entry.items()
                                           if k != "steps"}
                                   for label, entry in pending.items()},
        "drafts_closed_at_the_end": closed,
        "draft_ledgers": {name: arms[name].draft_ledger.to_dict()
                          for name in arms},
        "actual_cost": budget.to_dict(),
        "ledgers": ledgers.to_dict() if hasattr(ledgers, "to_dict") else None,
        "guard": guard.to_dict(),
        "boundary": {
            "evaluation_face_reads": 0, "sealed_reads": 0,
            "target_held_in_reads": 0, "risk_constants_touched": 0,
            "scope_grid_touched": 0, "consumer_touched": 0,
            "known_winner_supplied_to_the_model": 0,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or [])
    label = argv[0] if argv else "run1"
    limit = int(argv[1]) if len(argv) > 1 else None
    result = build(limit=limit, label=label)
    out = ART / ("dev_auto2_frame_c__%s.json" % label)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(drafts._plain(result), ensure_ascii=False,
                              indent=1, default=str), encoding="utf-8")
    cost = result.get("actual_cost") or {}
    print("%s | fits %s/%s | llm %s/%s | %ss"
          % (result.get("status") or "OK",
             cost.get("physical_consumer_fits"), cost.get("fit_ceiling"),
             cost.get("llm_calls"), cost.get("llm_ceiling"),
             cost.get("wall_seconds")))
    print("  units both arms: %s" % (
        (result.get("geometry") or {}).get("units_completed_by_both_arms")))
    print("  queued: %s | drafts opened: %s | promotions: %s"
          % (sum(int(s.get("queued") or 0)
                 for s in result.get("revision_steps") or ()),
             sum(int(s.get("drafts_opened") or 0)
                 for s in result.get("revision_steps") or ()),
             len(result.get("promotions_to_the_skill_card") or ())))
    print("  wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
