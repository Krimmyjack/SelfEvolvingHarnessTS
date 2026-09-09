"""DEV-KNOW-1: is the Skill a formation segment accumulated worth anything later?

The question, exactly
---------------------
    Under the same subsequent adaptation budget, does the legal Skill an agent
    accumulated in an earlier segment let it choose, modify and verify programs
    better than an agent holding only the older knowledge?

Not "does it PATCH more often", and not "is the new card good on its own".  The
comparison is between two complete deliveries over the same later units.

Shape
-----
Per group, one formation trajectory and then two arms:

===========================  ===================  =====================
                             formation u5-u15     test u16,17,23,24,25
===========================  ===================  =====================
formation (one trajectory)   runs                 --
A  old-knowledge             --                   runs from the fork
B  accumulated-knowledge     --                   runs from the fork
===========================  ===================  =====================

The formation segment is DEV-AUTO-3's **arm B route**: Fast local adaptation,
outer proposal and candidate supply, the ordinary Fast minting lifecycle -- and
no PATCH of any card, ever.  A verified program becomes a card of its own beside
the parent (``dev_know1_retention`` states the repair and checks the two
behaviours it turns on).

The fork
--------
Both arms are built fresh from the **same** boundary and differ in one thing:

* **B** starts from the formation-end snapshot.
* **A** starts from the formation-end snapshot with the capability cards the
  formation segment added removed, through the same store round-trip a
  revocation uses.  Not from the *entry* snapshot: a card the formation segment
  legitimately lost stays lost in both arms, because a revoked ancestor is not
  resurrected to make a control look better.

Everything else is held equal and is stated rather than implied:

* Same model, same tools, same DSL, same editing route, same admission policy,
  same acceptance frame, same per-arm LLM allowance, same verification rules.
* **A keeps the whole outer loop.**  It can search, propose, have candidates
  measured, verify them and mint its own cards.  It is not an identity control.
* Neither arm inherits a candidate supply pool.  Every Draft still open at the
  fork is *closed* -- not deleted -- in both arms at the same instant, so
  ``resupplied_programs_for_verification`` is empty on both sides and the test
  cannot become "who was pre-loaded with more answers to try".
* Neither arm inherits the formation segment's raw Episodes.  Both methods are
  built with an empty experience list, so the only channel by which formation
  knowledge can reach the test segment is a Skill card.
* Both arms inherit the identical Draft ledger, with every risk record,
  revocation and counter intact.  A closed lineage stays closed for both; no
  record is deleted to restore a permission.
* A card verified in the formation segment carries no execution right into a new
  unit.  The minted cards keep ``requires_target_support``; both arms face the
  same current Support and the same authoritative delayed gate.
* After the fork the arms share nothing: separate stores, ledgers, caches,
  histories, proposal trajectories and model sessions.

What this is not
----------------
Development level.  All sixteen units here are already exposed; u18-u22,
``TARGET_HELD_IN``, the sealed data and the +144 evaluation face are not read.
The five test units are a mechanism-development judgement about a later course
situation -- not global time-increasing evidence and not an independent
generalisation proof.  Two groups is a repeat-run check, not independent data,
and no significance is claimed.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import dev_know1_retention as keep
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_auto2_frame_c as D2
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_know1"

FORMATION_ARM = "formation"
ARM_OLD = "old-knowledge"
ARM_ACC = "accumulated-knowledge"
TEST_ARMS = (ARM_OLD, ARM_ACC)

K0_SKILL_ID = R.K0_SKILL_ID

#: Registered before the run and not chosen on any program's score.  The split
#: follows the course's own sequence-group change; u18-u22 are the
#: ``TARGET_HELD_IN`` units the population excludes by exposure name.
FORMATION_POSITIONS = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
TEST_POSITIONS = (16, 17, 23, 24, 25)

MAX_FITS = 2000
MAX_LLM = 1000
MAX_WALL_SECONDS = 6 * 3600

#: Six arm-runs share the package's model allowance: formation, A and B, twice.
PER_ARM_LLM = MAX_LLM // 6

#: Per-phase fit allowances, registered before the run.  Four phases -- two
#: groups x (formation, test) -- each with the same room, and 4 x 500 is the
#: package ceiling exactly.  The allowance is added to whatever the package has
#: already spent when the phase opens, so the formation segment cannot eat the
#: budget the test segment needs and neither group can eat the other's.  A
#: phase refuses *before* it spends, so crossing one is a recorded stop rather
#: than a partial reading.
FORMATION_FIT_ALLOWANCE = 500
TEST_FIT_ALLOWANCE = 500

GROUPS: dict[str, dict[str, Any]] = {
    "g1": {"seed": 20260909, "test_arm_order": (ARM_OLD, ARM_ACC)},
    "g2": {"seed": 20260910, "test_arm_order": (ARM_ACC, ARM_OLD)},
}


class Budget:
    """The package's ceilings, with per-(group, arm) ledgers and a phase floor."""

    def __init__(self) -> None:
        self.started = time.time()
        self.fits_by_arm: dict[str, int] = {}
        self.llm_by_arm: dict[str, int] = {}
        self.group = "g1"
        self.phase = "formation"
        self.soft_fit_ceiling = MAX_FITS
        #: One course-wide cost ledger, shared by every arm and reported split.
        self.ledgers = runner.Ledgers()
        #: Every phase ceiling that was put in force, in the order it opened.
        self.phase_ceilings: list[dict[str, Any]] = []

    # ---- keys ------------------------------------------------------------
    def _key(self, arm: str) -> str:
        return "%s/%s" % (self.group, arm)

    # ---- readings --------------------------------------------------------
    def fits(self) -> int:
        return sum(self.fits_by_arm.values())

    def llm(self) -> int:
        return sum(self.llm_by_arm.values())

    def elapsed(self) -> float:
        return time.time() - self.started

    # ---- spending --------------------------------------------------------
    def spend_fits(self, arm: str, n: int) -> None:
        key = self._key(arm)
        self.fits_by_arm[key] = self.fits_by_arm.get(key, 0) + int(n)

    def spend_llm(self, arm: str, n: int) -> None:
        key = self._key(arm)
        self.llm_by_arm[key] = self.llm_by_arm.get(key, 0) + int(n)

    def open_phase(self, phase: str, allowance: int) -> None:
        """This phase may spend ``allowance`` fits on top of what is already
        spent, and never past the package ceiling."""
        self.phase = str(phase)
        self.soft_fit_ceiling = min(MAX_FITS, self.fits() + int(allowance))
        self.phase_ceilings.append(
            {"group": self.group, "phase": self.phase,
             "opened_at_package_fits": self.fits(),
             "allowance": int(allowance),
             "ceiling_in_force": self.soft_fit_ceiling})

    def room_for(self, fits: int) -> bool:
        return self.fits() + int(fits) <= self.soft_fit_ceiling

    def llm_room(self, arm: str) -> int:
        mine = PER_ARM_LLM - int(self.llm_by_arm.get(self._key(arm), 0))
        return max(0, min(mine, MAX_LLM - self.llm()))

    def stop_reason(self) -> str | None:
        if self.fits() >= self.soft_fit_ceiling:
            return "FIT_CEILING_REACHED"
        if self.llm() >= MAX_LLM:
            return "LLM_CEILING_REACHED"
        if self.elapsed() >= MAX_WALL_SECONDS:
            return "WALL_CLOCK_CEILING_REACHED"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "physical_consumer_fits": self.fits(),
            "physical_consumer_fits_by_group_and_arm": dict(self.fits_by_arm),
            "fit_ceiling": MAX_FITS,
            "llm_calls": self.llm(),
            "llm_calls_by_group_and_arm": dict(self.llm_by_arm),
            "llm_ceiling": MAX_LLM,
            "llm_pre_allocated_per_arm_run": PER_ARM_LLM,
            "wall_seconds": round(self.elapsed(), 1),
            "wall_ceiling": MAX_WALL_SECONDS,
            "phase_fit_ceilings_in_force": list(self.phase_ceilings),
            "counting": ("every physical fit, including ones spent inside a "
                         "call that then raised; no allowance is carried over "
                         "and no call is made to level the arms' cost"),
        }

    # ---- cross-process persistence ---------------------------------------
    # A package's fit/LLM/wall ceilings apply across every group.  When a
    # group is run in its own OS process -- to actually release memory back
    # to the platform between groups, which a single long-lived process does
    # not reliably do -- the *ledgers* have to survive the process boundary,
    # or the second process would silently start every cap over at zero and
    # the package ceiling would stop meaning the package.

    def to_state(self) -> dict[str, Any]:
        return {
            "started": self.started,
            "fits_by_arm": dict(self.fits_by_arm),
            "llm_by_arm": dict(self.llm_by_arm),
            "ledgers": dataclasses.asdict(self.ledgers),
            "phase_ceilings": list(self.phase_ceilings),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "Budget":
        budget = cls()
        budget.started = float(state.get("started") or budget.started)
        budget.fits_by_arm = {str(k): int(v)
                              for k, v in (state.get("fits_by_arm") or {}).items()}
        budget.llm_by_arm = {str(k): int(v)
                             for k, v in (state.get("llm_by_arm") or {}).items()}
        ledger_state = dict(state.get("ledgers") or {})
        known = {f.name for f in dataclasses.fields(budget.ledgers)}
        for key, value in ledger_state.items():
            if key in known:
                setattr(budget.ledgers, key, value)
        budget.phase_ceilings = list(state.get("phase_ceilings") or [])
        return budget

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state(), ensure_ascii=False,
                                   indent=1, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Budget":
        return cls.from_state(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# the outer step, with an honest reading of the card
# ---------------------------------------------------------------------------

def revision_step(arm: Any, **kwargs: Any) -> dict[str, Any]:
    """DEV-AUTO-2's Frame C step, refused outright when there is no card.

    ``R.current_skill_steps`` answers a missing card with ``ANCESTOR_PROGRAM``.
    That fallback would let a course whose card had been revoked keep proposing
    against the historical ancestor as though it were still the parent -- the
    ancestor standing in for a card that is not there.  The card is read here
    first, and a missing one ends the step as a fact instead.
    """
    card = keep.current_card(arm.active_snapshot(), K0_SKILL_ID)
    usable = bool(card["present"]) and not card.get("body_unparseable")
    library = sorted(keep.capability_cards(arm.active_snapshot()))
    if not usable:
        return {"position": int(kwargs.get("position") or 0),
                "arm": arm.spec.name, "frame": "C",
                "outcome": "NO_CURRENT_CARD",
                "current_card": {k: v for k, v in card.items()
                                 if k != "steps"},
                "capability_cards_now": library,
                "why": ("the target card is not in the active snapshot; the "
                        "historical ancestor is not substituted for it")}
    step = D2.revision_step(arm, **kwargs)
    step["current_card"] = {k: v for k, v in card.items() if k != "steps"}
    step["capability_cards_now"] = library
    step["the_parent_read_is_the_card_that_is_there"] = bool(
        (step.get("parent_policy") or {}).get("program")
        == rev.program_label(card["steps"] or ()))
    return step


def verification_event(arm: Any, *, entry: Mapping[str, Any], position: int,
                       cell: Mapping[str, Any]) -> dict[str, Any]:
    """A supplied candidate that Fast deployed and the delayed gate passed.

    DEV-AUTO-3's arm B event, with the write-back branch gone rather than
    switched off: this package never PATCHes a card.  What the run does instead
    is what ``run_cell`` already did on this cell -- ``activate_approved``
    mints the program as a card of its own, beside the parent.
    """
    event = {
        "arm": arm.spec.name,
        "program": cell.get("deployed_label"),
        "proposed_at_position": entry.get("proposed_at_position"),
        "verified_at_position": int(position),
        "draft_id": entry.get("draft_id"),
        "independent_unit": entry.get("proposed_at_position") != int(position),
        "delayed_gate": cell.get("delayed_gate"),
        "deployed_via": cell.get("deployed_via"),
        "parent_program_at_proposal": entry.get("parent_program_at_proposal"),
        "what_changes": entry.get("what_changes"),
        "minted_a_card_of_its_own": list(
            cell.get("skills_minted_this_unit") or ()),
        "activated": bool(cell.get("activated")),
        "cards_after_the_event": sorted(
            keep.capability_cards(arm.active_snapshot())),
        "no_card_was_patched": True,
    }
    draft = arm.draft_ledger.by_id(entry.get("draft_id"))
    if draft is not None:
        arm.draft_ledger.close(draft, "VERIFIED_AND_MINTED_AS_ITS_OWN_CARD"
                               if cell.get("skills_minted_this_unit")
                               else "VERIFIED_WITHOUT_A_NEW_CARD")
        event["draft_closed_as"] = draft.closed
    return event


# ---------------------------------------------------------------------------
# one segment of the course
# ---------------------------------------------------------------------------

def _run_segment(*, arms: Mapping[str, Any], order: Sequence[str],
                 rows: Sequence[Mapping[str, Any]], group: str, phase: str,
                 machinery: Mapping[str, Any], ledgers: Any, guard: Any,
                 budget: Budget, proposer: Any, metered: Any,
                 index: Mapping[tuple, int],
                 by_position: Mapping[int, Mapping[str, Any]],
                 checkpoints_dir: Path = CHECKPOINTS,
                 ) -> dict[str, Any]:
    """Every unit in ``rows``, every arm in ``order``, cells then outer steps."""
    cells: list[dict[str, Any]] = []
    steps_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    attempts_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    history_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    pending_by_arm: dict[str, dict[str, Any]] = {n: {} for n in arms}
    events_by_arm: dict[str, list[dict[str, Any]]] = {n: [] for n in arms}
    faults_by_arm: dict[str, int] = {n: 0 for n in arms}
    completed: list[int] = []
    stopped_at: dict[str, Any] | None = None

    for row in rows:
        reason = budget.stop_reason()
        if reason:
            stopped_at = {"position": row["position"], "why": reason,
                          "phase": phase}
            break
        ctx = R.opp._unit_ctx(row["unit"])
        cell_by_arm: dict[str, Any] = {}
        try:
            for name in order:
                fits_before, llm_before = budget.fits(), budget.llm()
                cell = R.run_cell(arms[name], ctx, position=row["position"],
                                  ledgers=ledgers, guard=guard,
                                  machinery=machinery, budget=budget,
                                  seed=None)
                cell["group"] = group
                cell["phase"] = phase
                # The package's own two meters, read across this cell alone, so
                # "how much feedback did it take to reach a qualifying
                # deployment" can be answered without re-deriving a cost.
                cell["package_fits_this_cell"] = budget.fits() - fits_before
                cell["package_llm_this_cell"] = budget.llm() - llm_before
                winner_id = str(cell.get("winner_candidate_id") or "")
                cell["deployed_skill_id"] = (
                    winner_id[len("cand_skill_"):]
                    if winner_id.startswith("cand_skill_") else None)
                cell["capability_cards_at_end"] = sorted(
                    keep.capability_cards(arms[name].active_snapshot()))
                cell_by_arm[name] = cell
                cells.append(cell)
        except R.PackageCeiling as exc:
            stopped_at = {"position": row["position"], "phase": phase,
                          "why": "FIT_CEILING_REACHED", "detail": str(exc)}
            break
        except runner.RunFault as exc:
            stopped_at = {"position": row["position"], "phase": phase,
                          "why": "RUN_FAULT", "detail": str(exc)[:300]}
            break

        completed.append(row["position"])
        for name in order:
            cell = cell_by_arm[name]
            history_by_arm[name].append({
                "position": row["position"], "unit": row["unit"],
                "deployed_label": cell.get("deployed_label"),
                "deployed_scope": cell.get("deployed_scope"),
                "support_reading": cell.get("support_reading"),
                "delayed_reading": cell.get("delayed_reading"),
                "delayed_gate": cell.get("delayed_gate"),
            })
            deployed = cell.get("deployed_label")
            if (deployed in pending_by_arm[name]
                    and bool((cell.get("delayed_gate") or {}).get("passes"))):
                entry = pending_by_arm[name].pop(deployed)
                events_by_arm[name].append(verification_event(
                    arms[name], entry=entry, position=row["position"],
                    cell=cell))

        for name in order:
            if budget.stop_reason():
                continue
            arm = arms[name]
            library_before = sorted(keep.capability_cards(arm.active_snapshot()))
            fits_before, llm_before = budget.fits(), budget.llm()
            try:
                step = revision_step(
                    arm, position=row["position"],
                    origin=int(row["unit"]["origin"]), proposer=proposer,
                    history=history_by_arm[name],
                    attempts=attempts_by_arm[name],
                    ledgers=ledgers, budget=budget, metered=metered,
                    pending=pending_by_arm[name], index=index,
                    by_position=by_position,
                    case_prefix="devknow1-%s-%s-%s" % (group, phase,
                                                       arm.spec.slug))
            except R.PackageCeiling as exc:
                step = {"position": row["position"],
                        "outcome": "STOPPED_AT_THE_FIT_CEILING",
                        "why": str(exc)}
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                step = {"position": row["position"], "outcome": "STEP_FAULT",
                        "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
            step["arm"] = name
            step["group"] = group
            step["phase"] = phase
            step["package_fits_this_step"] = budget.fits() - fits_before
            step["package_llm_this_step"] = budget.llm() - llm_before
            step.setdefault("capability_cards_now", library_before)
            steps_by_arm[name].append(step)
            for label, entry in pending_by_arm[name].items():
                entry.setdefault(
                    "parent_program_at_proposal",
                    (step.get("parent_policy") or {}).get("program"))
                entry.setdefault("cards_held_at_proposal", library_before)
            if step.get("outcome") in ("PROPOSER_FAULT", "STEP_FAULT"):
                faults_by_arm[name] += 1
            else:
                faults_by_arm[name] = 0
            if faults_by_arm[name] >= 3:
                stopped_at = {"position": row["position"], "arm": name,
                              "phase": phase, "why": "PROPOSER_UNREACHABLE",
                              "detail": step.get("why")}
        if stopped_at:
            break

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / ("%s_%s_progress.json" % (group, phase))).write_text(
            json.dumps(drafts._plain({
                "group": group, "phase": phase,
                "completed_positions": completed,
                "cost": budget.to_dict(), "cells": cells}),
                ensure_ascii=False, default=str), encoding="utf-8")

    return {"cells": cells, "revision_steps": steps_by_arm,
            "attempt_log": attempts_by_arm, "verification_events": events_by_arm,
            "pending_never_verified": {
                n: {k: {kk: vv for kk, vv in v.items() if kk != "steps"}
                    for k, v in pending_by_arm[n].items()} for n in arms},
            "completed_positions": completed, "stopped_at": stopped_at}


# ---------------------------------------------------------------------------
# the fork
# ---------------------------------------------------------------------------

def _copy_cache(source: Any, target: Any) -> int:
    """The formation segment's readings, handed to both arms identically.

    A cache entry is a *reading* of one (unit, face, Consumer, program), keyed
    by a program the asking arm has to name for itself, so inheriting the
    common prefix cannot tell either arm that a program exists.  Both arms get
    the same entries and each pays for everything it reads after the fork; the
    counters start at zero so the test segment's cost is the test segment's.
    """
    for key, value in source._entries.items():
        target._entries[(target.arm,) + tuple(key)[1:]] = value
    return len(target._entries)


def _fork(*, formation: Any, group: str, machinery: Mapping[str, Any],
          ledgers: Any, guard: Any, backend_factory: Any,
          entry_cards: Mapping[str, Any],
          minted_lineage_keys: Mapping[str, str],
          root: Path) -> dict[str, Any]:
    """Two arms from one boundary, differing only in the cards A cannot read."""
    end_snapshot = formation.active_snapshot()
    end_cards = keep.capability_cards(end_snapshot)
    formed = sorted(set(end_cards) - set(entry_cards))
    lost = sorted(set(entry_cards) - set(end_cards))

    # Every Draft still open at the boundary is closed -- in the one ledger
    # both arms are copied from, so the pool is empty on both sides at the same
    # instant and every counter, state and risk record survives the closure.
    carried_pool = []
    for draft in list(formation.draft_ledger.drafts):
        if draft.closed is None:
            carried_pool.append({
                "draft_id": draft.draft_id, "state": draft.state,
                "verification_attempts": draft.verification_attempts,
                "revisions": draft.revisions,
                "program": rev.program_label(draft.program_steps)})
            formation.draft_ledger.close(draft, keep.FORK_CLOSURE)

    removal: dict[str, Any] = {"removed": [], "not_found": [],
                               "files_unlinked": [], "unchanged": True}
    arms: dict[str, Any] = {}
    for name in TEST_ARMS:
        spec = runner.ArmSpec(name, "k0", True, False, True)
        if name == ARM_ACC:
            start = end_snapshot
        else:
            arm_store = machinery["SnapshotStore"](root / name / "prepare")
            removal = keep.remove_cards(store=arm_store,
                                        snapshot=end_snapshot,
                                        skill_ids=formed)
            start = removal["snapshot"]
        arm = runner.Arm(spec, root=root, machinery=machinery,
                         start_snapshot=start, ledgers=ledgers, guard=guard,
                         backend_factory=backend_factory,
                         outer_slow_factory=None, offline=False)
        # The runtime constraints, carried whole.  Deleting a record to give an
        # arm back a permission is the one move this package will not make.
        arm.draft_ledger = copy.deepcopy(formation.draft_ledger)
        arm.bank = [dict(row) for row in formation.bank]
        arm.processed = list(formation.processed)
        dropped = set(formed) if name == ARM_OLD else set()
        arm.active_skill_ids = [s for s in formation.active_skill_ids
                                if s not in dropped]
        arm.active_program_signatures = {
            sig: sid for sig, sid in formation.active_program_signatures.items()
            if sid not in dropped}
        arm.active_lineage_keys = {
            key for key in formation.active_lineage_keys
            if minted_lineage_keys.get(key) not in dropped}
        arm._build("online")
        arm._store.materialize(arm.active_snapshot())
        _copy_cache(formation.replay_cache, arm.replay_cache)
        arms[name] = arm

    return {
        "arms": arms,
        "cards_at_the_entry_state": sorted(entry_cards),
        "cards_at_the_boundary": sorted(end_cards),
        "formed_in_the_formation_segment": formed,
        "lost_during_the_formation_segment": lost,
        "removed_to_build_the_old_knowledge_arm": removal.get("removed"),
        "removal_files_unlinked": removal.get("files_unlinked"),
        "old_knowledge_arm_cards": sorted(
            keep.capability_cards(arms[ARM_OLD].active_snapshot())),
        "accumulated_arm_cards": sorted(
            keep.capability_cards(arms[ARM_ACC].active_snapshot())),
        "candidate_supply_pool_closed_at_the_fork": carried_pool,
        "treatment": ("NO_KNOWLEDGE_TREATMENT" if not formed
                      else "KNOWLEDGE_TREATMENT"),
        "how_the_old_knowledge_arm_was_built": (
            "the boundary snapshot with the formation segment's added cards "
            "removed by the same fork/unlink/compile/materialize round-trip a "
            "revocation performs -- not the entry snapshot, so a card the "
            "formation segment legitimately lost stays lost in both arms"),
        "what_is_not_carried": [
            "the formation segment's raw Episodes: both methods are built with "
            "an empty experience list",
            "the candidate supply pool: every open Draft was closed, in the "
            "one ledger both arms are copied from",
            "no hidden field and no candidate menu carries a formation "
            "program past the Skill boundary",
        ],
        "what_is_carried_identically_by_both_arms": [
            "the Draft ledger with every state, risk record, revocation and "
            "counter",
            "the bank rows and the processed unit contexts the outer screen "
            "reads",
            "the instrument's replay readings from the common prefix",
        ],
    }


# ---------------------------------------------------------------------------
# one group
# ---------------------------------------------------------------------------

def build(*, group: str, budget: Budget,
          formation_limit: int | None = None,
          test_limit: int | None = None,
          run_id: str = "") -> dict[str, Any]:
    from evaluation.main_protocol_p4 import smoke_dev_know1 as smoke

    # A run_id gives this attempt its own checkpoint directory and its own
    # ``.dev_know1_runs`` store root, so a retry after a crash (OOM or
    # otherwise) cannot overwrite a still-live checkpoint or the arm stores of
    # an attempt that has not been declared abandoned.  The empty default
    # keeps the historical, unnamespaced paths for callers that never asked
    # for isolation.
    checkpoints_dir = (CHECKPOINTS / run_id) if run_id else CHECKPOINTS
    run_root = (base.ROOT / ".dev_know1_runs" / run_id / group if run_id
               else base.ROOT / ".dev_know1_runs" / group)

    started = datetime.now(timezone(timedelta(hours=8)))
    plan = GROUPS[group]
    preflight = smoke.run()
    if not preflight["passed"]:
        return {"stage": "DEV_KNOW1_TWO_ARMS", "status": "BLOCKED_BY_SMOKE",
                "group": group, "smoke": preflight}
    random.seed(int(plan["seed"]))
    np.random.seed(int(plan["seed"]) % (2 ** 32))

    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population, excluded = R._population(doc)
    index = D2._course_index(population)
    by_position = {row["position"]: row for row in population}
    after = {row["position"]: row for row in population
             if row["side"] == "after_the_boundary"}
    formation_rows = [after[p] for p in FORMATION_POSITIONS if p in after]
    test_rows = [after[p] for p in TEST_POSITIONS if p in after]
    if formation_limit is not None:
        formation_rows = formation_rows[:int(formation_limit)]
    if test_limit is not None:
        test_rows = test_rows[:int(test_limit)]

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    snapshot_dir, snapshot_source = R._resolve_k0_snapshot(doc, state["k0"])
    if snapshot_dir is None:
        return {"stage": "DEV_KNOW1_TWO_ARMS", "status": "BLOCKED",
                "group": group, "why": "no readable K0 snapshot"}
    start_snapshot = machinery["compile_snapshot"](snapshot_dir,
                                                   verify_lock=False)
    entry_cards = keep.capability_cards(start_snapshot)

    guard = runner.BudgetGuard(ordering_cap=MAX_LLM,
                               per_unit_arm_cap=PER_ARM_LLM,
                               ledgers=budget.ledgers)

    # Every inner backend this group creates, so the requested/returned model
    # and the actual token usage can be read back after the fact without a
    # new counting surface -- ``BudgetedAgentBackend`` already tracks both.
    created_backends: list[Any] = []

    def backend_factory():
        inner = machinery["agentic"]._default_backend_factory(PER_ARM_LLM)
        created_backends.append(inner)
        return runner._MeteredFastBackend(inner, guard=guard, billable=True)

    outer_inner = machinery["agentic"]._default_backend_factory(MAX_LLM)
    created_backends.append(outer_inner)
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

    # ---- the formation segment ------------------------------------------
    budget.group = group
    budget.open_phase("formation", FORMATION_FIT_ALLOWANCE)
    formation = runner.Arm(runner.ArmSpec(FORMATION_ARM, "k0", True, False,
                                          True),
                           root=run_root, machinery=machinery,
                           start_snapshot=start_snapshot,
                           ledgers=budget.ledgers, guard=guard,
                           backend_factory=backend_factory,
                           outer_slow_factory=None, offline=False)
    R._seed_arm(formation, state, fresh_ledger=True)
    formation._build("online")
    formation._store.materialize(formation.active_snapshot())

    formation_out = _run_segment(
        arms={FORMATION_ARM: formation}, order=(FORMATION_ARM,),
        rows=formation_rows, group=group, phase="formation",
        machinery=machinery, ledgers=budget.ledgers, guard=guard,
        budget=budget, proposer=proposer, metered=outer_metered,
        index=index, by_position=by_position, checkpoints_dir=checkpoints_dir)

    # A key event, on disk the moment it is known -- not only inside the one
    # report file this whole build() writes at the very end.  If the process
    # dies at the fork or during the test segment, what the formation segment
    # actually did is not lost with it.
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / ("%s_formation_done.json" % group)).write_text(
        json.dumps(drafts._plain({
            "group": group, "formation": formation_out,
            "cost_at_formation_end": budget.to_dict()}),
            ensure_ascii=False, default=str), encoding="utf-8")

    minted_lineage_keys = {
        str(cell["lineage_key"]): (cell.get("skills_minted_this_unit")
                                   or [None])[0]
        for cell in formation_out["cells"] if cell.get("lineage_key")}

    # ---- the fork --------------------------------------------------------
    fork = _fork(formation=formation, group=group, machinery=machinery,
                 ledgers=budget.ledgers, guard=guard,
                 backend_factory=backend_factory,
                 entry_cards=entry_cards,
                 minted_lineage_keys=minted_lineage_keys, root=run_root)
    arms = fork.pop("arms")
    (checkpoints_dir / ("%s_fork.json" % group)).write_text(
        json.dumps(drafts._plain({"group": group, "fork": fork,
                                  "cost_at_the_fork": budget.to_dict()}),
                   ensure_ascii=False, default=str), encoding="utf-8")

    # ---- the test segment ------------------------------------------------
    budget.open_phase("test", TEST_FIT_ALLOWANCE)
    test_out = _run_segment(
        arms=arms, order=tuple(plan["test_arm_order"]), rows=test_rows,
        group=group, phase="test", machinery=machinery,
        ledgers=budget.ledgers, guard=guard, budget=budget,
        proposer=proposer, metered=outer_metered, index=index,
        by_position=by_position, checkpoints_dir=checkpoints_dir)

    closed = {name: arms[name].draft_ledger.close_unreencountered()
              for name in arms}
    closed[FORMATION_ARM] = formation.draft_ledger.close_unreencountered()

    common = sorted(set.intersection(*[
        {c["position"] for c in test_out["cells"] if c["arm"] == n}
        for n in arms]) if test_out["cells"] else set())
    scored = {n: [c for c in test_out["cells"]
                  if c["arm"] == n and c["position"] in common] for n in arms}

    return {
        "stage": "DEV_KNOW1_TWO_ARMS",
        "package": ("DEV-KNOW-1: does the Skill accumulated in a formation "
                    "segment help the next units, development level"),
        "group": group,
        "question": ("under the same subsequent adaptation budget, does the "
                     "legal Skill accumulated earlier let the agent choose, "
                     "modify and verify programs better than old knowledge "
                     "alone"),
        "registered_plan": {
            "seed": plan["seed"],
            "formation_positions": list(FORMATION_POSITIONS),
            "test_positions": list(TEST_POSITIONS),
            "test_arm_order": list(plan["test_arm_order"]),
            "formation_fit_allowance": FORMATION_FIT_ALLOWANCE,
            "test_fit_allowance": TEST_FIT_ALLOWANCE,
            "registered_before_the_run": True,
            "how_the_split_was_chosen": (
                "it follows the course's own sequence-group change, not any "
                "program's score; u18-u22 are the TARGET_HELD_IN units the "
                "population excludes by exposure name"),
            "what_the_seed_does": ("this package's own RNG only; the model "
                                   "service's sampling is not controlled and "
                                   "no claim is made that it is")},
        "arms": {
            FORMATION_ARM: ("one autonomous accumulation trajectory over "
                            "u5-u15; sees only the feedback legal at the time"),
            ARM_OLD: ("old knowledge: the boundary library minus the cards the "
                      "formation segment added; keeps search, verification, "
                      "minting and the whole outer loop"),
            ARM_ACC: ("accumulated knowledge: the boundary library as it "
                      "stands, including the cards formed legally in the "
                      "formation segment"),
        },
        "what_is_isolated": (
            "whether the formation segment's added Skill cards can be read; "
            "every other input, rule, allowance and route is the same"),
        "no_card_is_ever_patched": (
            "retention here is additive: a verified program becomes a card of "
            "its own through the ordinary Fast minting lifecycle, so a child "
            "that later fails takes only itself"),
        "what_this_is_not": [
            "not the formal A3/A5 main experiment",
            "development level only: every unit here is already exposed",
            "no sealed data, no TARGET_HELD_IN unit and no +144 face was read",
            "the five test units carry a mechanism-development judgement about "
            "a later course situation, not global time-increasing evidence and "
            "not an independent generalisation proof",
            "two groups is a repeat-run check, not independent data, and no "
            "significance is claimed",
        ],
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "smoke": preflight,
        "run_id": run_id or None,
        "transport": {k: v for k, v in transport.items() if k != "api_key"},
        "transport_actual_usage": {
            "requested_model": transport.get("model"),
            "returned_models": sorted(set().union(
                *(set(getattr(b, "returned_models", ()) or ())
                  for b in created_backends)) if created_backends else set()),
            "prompt_tokens": sum(int(getattr(b, "prompt_tokens", 0) or 0)
                                 for b in created_backends),
            "completion_tokens": sum(
                int(getattr(b, "completion_tokens", 0) or 0)
                for b in created_backends),
            "backend_instances_created": len(created_backends),
            "note": ("returned_models/token counts come straight off "
                     "BudgetedAgentBackend, which the relay populates from "
                     "each response's own usage/model fields; empty means "
                     "the relay never reported one, not that nothing ran"),
        },
        "transport_divergence": next(
            (row.get("acknowledged_divergence")
             for row in preflight.get("checks", [])
             if row.get("check")
             == "the_transport_is_the_one_the_previous_package_used"
             and not row.get("matches_the_previous_package")), None),
        "admission_rule_in_force": bounded.BOUNDED_POLICY.to_dict(),
        "acceptance_rule": {"frame": "C", "rule": D2.FRAME_C_RULE,
                            "queue_order": D2.CANDIDATE_ORDER},
        "geometry": {
            "ordering": R.ORDERING, "source_arm": R.SOURCE_ARM,
            "formation_units": [r["position"] for r in formation_rows],
            "test_units_planned": [r["position"] for r in test_rows],
            "test_units_completed_by_both_arms": common,
            "excluded_units": [{"position": r["position"], "why": r["why"]}
                               for r in excluded],
            "k0_snapshot_source": snapshot_source,
            "entry_capability_cards": sorted(entry_cards),
        },
        "formation": {
            "cells": formation_out["cells"],
            "revision_steps": formation_out["revision_steps"][FORMATION_ARM],
            "attempt_log": formation_out["attempt_log"][FORMATION_ARM],
            "verification_events":
                formation_out["verification_events"][FORMATION_ARM],
            "pending_never_verified":
                formation_out["pending_never_verified"][FORMATION_ARM],
            "completed_positions": formation_out["completed_positions"],
            "stopped_at": formation_out["stopped_at"],
            "summary": R._arm_summary(formation_out["cells"]),
        },
        "fork": fork,
        "test": {
            "cells": test_out["cells"],
            "revision_steps": test_out["revision_steps"],
            "attempt_log": test_out["attempt_log"],
            "verification_events": test_out["verification_events"],
            "pending_never_verified": test_out["pending_never_verified"],
            "completed_positions": test_out["completed_positions"],
            "stopped_at": test_out["stopped_at"],
        },
        "per_arm": {n: R._arm_summary(rows) for n, rows in scored.items()},
        "drafts_closed_at_the_end": closed,
        "draft_ledgers": {**{n: arms[n].draft_ledger.to_dict() for n in arms},
                          FORMATION_ARM: formation.draft_ledger.to_dict()},
        "actual_cost": budget.to_dict(),
        "guard": guard.to_dict(),
        "boundary": {
            "evaluation_face_reads": 0, "sealed_reads": 0,
            "target_held_in_reads": 0, "risk_constants_touched": 0,
            "scope_grid_touched": 0, "consumer_touched": 0,
            "known_winner_supplied_to_the_model": 0,
            "caches_shared_between_the_test_arms": 0,
            "cards_patched": 0,
            "core_files_edited": 0,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv or [])
    groups = [a for a in argv if a in GROUPS] or ["g1", "g2"]
    formation_limit = None
    test_limit = None
    # A run on a different transport is a different experiment, so it writes to
    # its own artifact rather than overwriting the comparable one.
    suffix = ""
    run_id = ""
    for arg in argv:
        if arg.startswith("formation="):
            formation_limit = int(arg.split("=", 1)[1])
        if arg.startswith("test="):
            test_limit = int(arg.split("=", 1)[1])
        if arg.startswith("suffix="):
            suffix = "_" + arg.split("=", 1)[1].strip("_")
        if arg.startswith("run="):
            run_id = arg.split("=", 1)[1].strip()

    # The package's ceilings are shared by every group.  When ``groups`` here
    # is only one of the two -- a caller running g1 and g2 as separate OS
    # processes so memory is actually released between them -- the ledgers
    # from any earlier process for this same run_id are loaded first, so the
    # second process's ceilings are the package's remaining room, not a fresh
    # 2000/1000 it has no right to.
    budget_path = ((CHECKPOINTS / run_id / "package_budget.json") if run_id
                   else None)
    if budget_path is not None and budget_path.is_file():
        budget = Budget.load(budget_path)
        print("resumed package budget from %s: fits=%d llm=%d wall=%.0fs"
              % (budget_path, budget.fits(), budget.llm(), budget.elapsed()))
    else:
        budget = Budget()

    for group in groups:
        result = build(group=group, budget=budget,
                       formation_limit=formation_limit, test_limit=test_limit,
                       run_id=run_id)
        if budget_path is not None:
            budget.save(budget_path)
        out = ART / ("dev_know1_two_arms__%s%s.json" % (group, suffix))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(drafts._plain(result), ensure_ascii=False,
                                  indent=1, default=str), encoding="utf-8")
        cost = result.get("actual_cost") or {}
        print("%s %s | fits %s/%s | llm %s/%s | %ss"
              % (result.get("status") or "OK", group,
                 cost.get("physical_consumer_fits"), cost.get("fit_ceiling"),
                 cost.get("llm_calls"), cost.get("llm_ceiling"),
                 cost.get("wall_seconds")))
        fork = result.get("fork") or {}
        print("   treatment: %s | formed: %s | removed for the control: %s"
              % (fork.get("treatment"),
                 fork.get("formed_in_the_formation_segment"),
                 fork.get("removed_to_build_the_old_knowledge_arm")))
        for name, summary in (result.get("per_arm") or {}).items():
            face = summary["delayed_face"]
            print("   %-24s delayed=%-11s treated=%-4s harmed=%-3s gate=%s"
                  % (name,
                     face["mean_aggregate_gain_over_the_served_population"],
                     face["total_treated_series"], face["total_harmed_series"],
                     face.get("authoritative_gate_passes")))
        print("   wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
