"""DEV-SEQ-4 steps 3 and 4: Slow revises its own guidance, then it is checked.

Step 3.  Slow is shown the outcome material it had before, plus the receipt
for the edit it wrote last time: what it predicted, and what step 2 actually
measured with the pre-decision history held fixed.  The snapshot it edits is
the one that already contains its card, so the catalog offers the card's own
body and applicability condition -- revising in place is available, and so is
leaving it alone.  Keeping the current guidance is a legitimate answer and is
recorded as ``NO_REVISION_TREATMENT``, not as a fault.

Step 4.  Whatever Slow writes is frozen and checked on later development
windows -- positions 12 and 13, which are not on the card it was shown.  The
full 20-sequence population is restored, the arms are the current knowledge
(the card as it stands) against the revised knowledge, and both arms start
from the same history.  A zero-exposure revision stops the arms and records
that its utility is untested; it is not rescued by moving a threshold.

What this package does not do: it does not adopt anything, it does not change
the Consumer, the risk rule, the scoring, the DSL or the training/serving
geometry, and it does not touch sealed or UNREAD windows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import dev_seq4_knowledge as k4
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_seq2 as seq2
from evaluation.main_protocol_p4 import run_dev_seq3 as seq3
from evaluation.main_protocol_p4 import run_dev_seq4 as seq4
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ROOT / "artifacts" / "main_protocol"
CHECKPOINTS = base.ROOT / "_scratch" / "dev_seq4_revision"

#: The windows whose outcomes Slow is shown.  These are the decisions both
#: DEV-SEQ-3 arms actually ran.
EVIDENCE_POSITIONS = (9, 10, 11)
#: The windows the revision is checked on.  Not on the card Slow is shown, and
#: not used to construct it.  Both are SPENT_DEV in the same block; no UNREAD
#: or sealed window is opened.
FOLLOWUP_POSITIONS = (12, 13)

ARM_A = seq2.ARM_A          # current knowledge: the DEV-SEQ-3 fork
ARM_B = seq2.ARM_B          # revised knowledge: whatever Slow writes now

ARM_POPULATION = 20
MIN_GROUP = 2
CONCURRENCY = 4
PER_SEQUENCE_LLM_CAP = 24
OUTER_LLM_CAP = 64
MAX_FITS = 800
MAX_LLM = 1200
MAX_WALL_SECONDS = 4 * 3600
REQUIRED_MODEL = "gpt-5.6-sol"

#: Arm B's own checkpoints: the history the fork's knowledge actually produced.
HISTORY_CHAIN: tuple[tuple[int, Path], ...] = (
    (7, base.ROOT / "_scratch" / "dev_seq2" / "pkg2" / "g2" / "formation_u007.json"),
    (8, base.ROOT / "_scratch" / "dev_seq2" / "pkg2" / "g2" / "formation_u008.json"),
    (9, base.ROOT / "_scratch" / "dev_seq3" / "pkg3" / "g2"
        / "validation_slow_candidate.json"),
    (10, base.ROOT / "_scratch" / "dev_seq3" / "pkg3" / "g2"
         / "followup_u010_slow_candidate.json"),
    (11, base.ROOT / "_scratch" / "dev_seq3" / "pkg3" / "g2"
         / "followup_u011_slow_candidate.json"),
)


def seeded_history() -> dict[str, list[dict[str, Any]]]:
    """Arm B's Episodes through position 11, by sequence, from checkpoints."""
    out: dict[str, list[dict[str, Any]]] = {}
    for _position, path in HISTORY_CHAIN:
        saved = json.loads(path.read_text(encoding="utf-8"))
        for uid, rows in (saved.get("episodes") or {}).items():
            out.setdefault(str(uid), []).extend(rows)
    return out


def evidence_episodes() -> list[Any]:
    """The DEV-SEQ-3 slow-candidate arm's Episodes at positions 9-11."""
    rows: list[dict[str, Any]] = []
    for position, path in HISTORY_CHAIN:
        if position not in EVIDENCE_POSITIONS:
            continue
        saved = json.loads(path.read_text(encoding="utf-8"))
        for _uid, entries in (saved.get("episodes") or {}).items():
            rows.extend(entries)
    return seq4.rehydrate(rows)


def decision_records() -> list[dict[str, Any]]:
    """That arm's per-decision receipts, so the card can show what ran."""
    doc = json.loads(
        (ART / "dev_seq3_guidance_delivery__g2.json").read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for arm, block in ((doc.get("validation") or {}).get("units") or {}).items():
        if arm == ARM_B:
            out.extend(block.get("sequences") or ())
    for unit in doc.get("followup_units") or ():
        if str(unit.get("arm")) == ARM_B:
            out.extend(unit.get("sequences") or ())
    return out


# ---------------------------------------------------------------------------
# step 3
# ---------------------------------------------------------------------------

def revision_boundary(*, machinery: Mapping[str, Any], budget: seq2.Budget,
                      outer_core: Any, fork: Any, store: Any, controller: Any,
                      binding: know.FeatureBinding,
                      contrast: Mapping[str, Any]) -> dict[str, Any]:
    """Let Slow review its own edit and write at most one more."""
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent

    episodes = evidence_episodes()
    grouping = know.group_failures(episodes=episodes, binding=binding,
                                   min_group=MIN_GROUP)
    out: dict[str, Any] = {
        "evidence_positions": list(EVIDENCE_POSITIONS),
        "followup_positions_not_in_the_slow_input": list(FOLLOWUP_POSITIONS),
        "episodes_in_the_slow_input": len(episodes),
        "whose_outcomes": (
            "the DEV-SEQ-3 slow-candidate arm: the arm whose knowledge is "
            "being revised, so the material is what happened WITH the "
            "guidance in place"),
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

    base_card = know.evidence_card(
        group, pattern_id="devseq4-revision",
        binding=binding, vocabulary=contract.SCOPE_CLASS["vocabulary"],
        decision_records=decision_records())

    previous = json.loads(
        (base.ROOT / "_scratch" / "dev_seq3" / "pkg3" / "g2" / "boundary.json"
         ).read_text(encoding="utf-8")).get("proposal") or {}
    card_entry = next((s for s in fork.snapshot.skills
                       if str(s.skill_id) == seq4.CARD_ID), None)
    if card_entry is None:
        out["outcome"] = "CARD_NOT_IN_PARENT_SNAPSHOT"
        return out

    card = k4.revision_card(
        base_card, pattern_id="devseq4-revision",
        previous_proposal=previous,
        current_body=str(card_entry.body),
        current_applicability=json.loads(json.dumps(
            card_entry.observable_applicability, default=dict)),
        contrast=contrast)
    out["card"] = card

    catalog = know.build_catalog(controller=controller, parent=fork)
    out["surface_catalog"] = [{k: v for k, v in row.items()
                               if k != "current_value"} for row in catalog]
    out["surfaces_offered"] = [row["surface_id"] for row in catalog]
    out["the_previous_card_is_editable_in_place"] = any(
        seq4.CARD_ID in row["surface_id"] for row in catalog)
    if not catalog:
        out["outcome"] = "NO_AUTHORISED_SURFACE"
        return out

    llm_before = int(budget.ledgers.llm_total())
    proposal = know.propose_update(slow_factory=lambda: TTHASlowAgent(outer_core),
                                   card=card, catalog=catalog,
                                   snapshot=fork.snapshot)
    budget.spend_llm("slow", int(budget.ledgers.llm_total()) - llm_before)
    out["proposal"] = {k: v for k, v in proposal.items() if k != "manifest"}
    if not proposal.get("proposed"):
        why = str(proposal.get("why"))
        out["outcome"] = {
            "ACCOUNT_OR_PERMISSION_FAULT": "BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT",
            "TRANSPORT_TRANSIENT_FAULT": "BOUNDARY_TRANSPORT_FAULT",
        }.get(why, "NO_PROPOSAL")
        if out["outcome"] == "NO_PROPOSAL":
            out.update(k4.unchanged_treatment(
                "Slow was shown the receipt for its own edit and did not "
                "propose a second one: %s" % why))
        else:
            out["why_this_is_not_an_abstention"] = (
                "the request never reached a model: a configuration receipt, "
                "not Slow declining to propose")
        return out

    current = next((row.get("current_value") for row in catalog
                    if row["surface_id"]
                    == str(proposal.get("target_surface_id"))), None)
    probe, kind = seq2._edit_probe(operation=proposal.get("operation"),
                                   skill_id=proposal.get("skill_id"),
                                   written=proposal.get("written_text")
                                   or proposal.get("observable_applicability"),
                                   current=current)
    out["edit_probe"], out["edit_probe_kind"] = probe, kind

    applied = know.apply_update(controller=controller, store=store,
                                snapshot=fork.snapshot,
                                manifest=proposal["manifest"], catalog=catalog)
    out["applied"] = {k: v for k, v in applied.items() if k != "snapshot"}
    if not applied.get("applied"):
        out["outcome"] = "PROPOSAL_DID_NOT_COMPILE"
        return out
    out["outcome"] = "REVISION_COMPILED"
    out["_snapshot"] = applied["snapshot"]
    return out


def resume_boundary(checkpoints: Path, machinery: Mapping[str, Any],
                    store_root: Path) -> tuple[dict[str, Any], Any] | None:
    """The boundary from a previous process, without a second Slow call.

    The Slow turn is the one irreplaceable thing in this package: it costs a
    real call and a second one would not return the same text, so a resumed
    run must never re-ask.  If the checkpoint says a revision compiled, the
    revised snapshot is recompiled from the store by its own SHA and the
    identity is checked against what the checkpoint recorded.
    """
    path = checkpoints / "boundary.json"
    if not path.is_file():
        return None
    boundary = json.loads(path.read_text(encoding="utf-8"))
    applied = boundary.get("applied") or {}
    sha = str(applied.get("runtime_bundle_sha") or "")
    if boundary.get("outcome") != "REVISION_COMPILED" or not sha:
        return boundary, None
    revised = machinery["compile_snapshot"](store_root / sha, verify_lock=False)
    if revised.runtime_bundle_sha != sha:
        raise RuntimeError(
            "the stored revision does not hash to what the boundary recorded")
    boundary["resumed_from_checkpoint"] = {
        "path": str(path), "runtime_bundle_sha": sha,
        "no_second_slow_call": (
            "the edit was written and compiled by the earlier process; it is "
            "reloaded by SHA, not re-proposed"),
    }
    return boundary, revised


def collect_units(checkpoints: Path) -> list[dict[str, Any]]:
    """Every follow-up unit any process has finished, tagged and in order."""
    units: list[dict[str, Any]] = []
    for path in sorted(checkpoints.glob("followup_u*.json")):
        stem = path.stem[len("followup_u"):]
        position = int(stem.split("_", 1)[0])
        arm = stem.split("_", 1)[1].replace("_", "-")
        saved = json.loads(path.read_text(encoding="utf-8"))
        unit = saved.get("unit")
        if not isinstance(unit, Mapping):
            continue
        unit = dict(unit)
        unit["position"], unit["arm"] = position, arm
        units.append(unit)
    return units


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(*, run_id: str = "", concurrency: int | None = None,
          positions: Sequence[int] = (), arms: Sequence[str] = (),
          budget: "seq2.Budget | None" = None) -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    wanted_positions = tuple(positions) or FOLLOWUP_POSITIONS
    wanted_arms = tuple(arms) or (ARM_A, ARM_B)
    for module in (seq2, seq3):
        module.MAX_FITS, module.MAX_LLM = MAX_FITS, MAX_LLM
        module.MAX_WALL_SECONDS = MAX_WALL_SECONDS
    seq2.GROUP_WALL_SECONDS = MAX_WALL_SECONDS
    seq3.ARM_POPULATION = seq2.ARM_POPULATION = ARM_POPULATION
    seq3.FORMATION_POPULATION = seq2.FORMATION_POPULATION = ARM_POPULATION
    seq3.CONCURRENCY = int(concurrency or CONCURRENCY)
    seq3.PER_SEQUENCE_LLM_CAP = PER_SEQUENCE_LLM_CAP

    budget = budget if budget is not None else seq2.Budget()
    budget.open_group()
    checkpoints = CHECKPOINTS / (run_id or "run")
    checkpoints.mkdir(parents=True, exist_ok=True)
    budget_path = checkpoints / "package_budget.json"

    def bank(where: str) -> None:
        """Write the spend to disk now, not at exit.

        A host OOM kill does not run ``finally``, so a budget that is only
        saved on the way out is lost exactly when the run failed -- and the
        next process then under-counts the package total.  Every call that
        can spend is followed by one of these.
        """
        try:
            budget.save(budget_path)
        except Exception:  # noqa: BLE001 - accounting must never end the run
            pass
        del where

    contrast_path = ART / "dev_seq4_same_start_contrast.json"
    if not contrast_path.is_file():
        return {"stage": "DEV_SEQ4_REVISION",
                "status": "BLOCKED_STEP_2_NOT_RUN",
                "why": "the same-start contrast is the input to step 3"}
    contrast = json.loads(contrast_path.read_text(encoding="utf-8"))

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    transport = seq3.transport_preflight(machinery)
    budget.spend_llm("preflight", int(transport.get("calls") or 0))
    bank("preflight")
    if not transport.get("usable"):
        return {"stage": "DEV_SEQ4_REVISION", "status": "BLOCKED_BY_TRANSPORT",
                "transport": transport}

    fork_snapshot = machinery["compile_snapshot"](seq4.STORE / seq4.FORK_SHA,
                                                  verify_lock=False)
    if fork_snapshot.runtime_bundle_sha != seq4.FORK_SHA:
        return {"stage": "DEV_SEQ4_REVISION",
                "status": "BLOCKED_SNAPSHOT_IDENTITY"}

    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    population_rows, _excluded = R._population(doc)
    by_position = {row["position"]: row for row in population_rows}
    for position in FOLLOWUP_POSITIONS:
        exposure = list((by_position[position]["unit"].get("exposure") or ()))
        if "UNREAD" in exposure:
            return {"stage": "DEV_SEQ4_REVISION",
                    "status": "BLOCKED_WINDOW_IS_UNREAD",
                    "position": position, "exposure": exposure}

    positions = (*EVIDENCE_POSITIONS, *FOLLOWUP_POSITIONS)
    units = {p: R.opp._unit_ctx(by_position[p]["unit"]) for p in positions}
    origins = {p: int(units[p].origin) for p in positions}
    features_by_position = {
        p: {uid: ps.sequence_features(machinery, units[p], uid,
                                      origin=units[p].origin)
            for uid in ps.decision_uids(units[p], size=ARM_POPULATION)}
        for p in positions}
    binding = know.FeatureBinding(
        know.bind_window_features(features_by_position, origins))

    store = machinery["SnapshotStore"](
        base.ROOT / ".dev_seq4_runs" / (run_id or "run") / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    fork = store.materialize(fork_snapshot)

    shared = seq3.SharedState()
    outer_guard = seq3.SequenceGuard(ordering_cap=MAX_LLM,
                                     per_unit_arm_cap=OUTER_LLM_CAP,
                                     ledgers=budget.ledgers, lock=shared.ledger)
    outer_inner = machinery["agentic"]._default_backend_factory(OUTER_LLM_CAP)
    outer_core = machinery["TTHAAgentCore"](
        runner._MeteredOuterBackend(outer_inner, guard=outer_guard,
                                    billable=True),
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=transport["requested_model"], base_url=transport["base_url"])

    resumed = resume_boundary(checkpoints, machinery, store.root)
    if resumed is not None:
        boundary, revised = resumed
    else:
        boundary = revision_boundary(machinery=machinery, budget=budget,
                                     outer_core=outer_core, fork=fork,
                                     store=store, controller=controller,
                                     binding=binding, contrast=contrast)
        revised = boundary.pop("_snapshot", None)
        bank("slow")
        (checkpoints / "boundary.json").write_text(
            json.dumps(boundary, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
    report: dict[str, Any] = {
        "stage": "DEV_SEQ4_REVISION",
        "run_id": run_id or "run",
        "started": started.isoformat(),
        "question": (
            "shown what its own guidance actually did, does Slow revise it -- "
            "and does the revision change anything on windows it was not "
            "shown?"),
        "step_2_input": k4.summarise_for_report(contrast),
        "boundary": boundary,
        "transport": transport,
    }
    if revised is None:
        report["status"] = "STOPPED_AT_BOUNDARY"
        report["followup"] = {
            "ran": False,
            "why": ("no revised knowledge exists, so there is nothing to "
                    "freeze and no follow-up contrast to run.  No utility "
                    "claim is made"),
        }
        report["cost"] = budget.to_dict()
        return report

    # ---- does the revision reach the follow-up decisions at all? ---------
    probe = str(boundary.get("edit_probe") or "")
    windows = {str(p): {uid: binding.get(uid, origins[p])
                        for uid in features_by_position[p]}
               for p in FOLLOWUP_POSITIONS}
    exposure = know.exposure_check(
        parent=fork_snapshot, candidate=revised, probe=probe,
        windows=windows,
        skill_id=str(boundary.get("proposal", {}).get("skill_id") or ""))
    report["exposure"] = exposure
    if not int(exposure.get("decisions_the_edit_reaches") or 0):
        report["status"] = "STOPPED_ZERO_EXPOSURE"
        report["followup"] = {
            "ran": False,
            "why": ("the revision reaches none of the follow-up decisions, so "
                    "running the arms would measure a guidance that is never "
                    "presented"),
            "utility": "UNTESTED_NOT_ZERO",
        }
        report["cost"] = budget.to_dict()
        return report

    # ---- step 4: the frozen contrast on windows Slow was not shown -------
    cache = runner.ReplayPredictionCache("devseq4rev")
    unit_readings = {
        p: seq3.LockedReadings(
            ps.UnitReadings(cache=cache, ctx=units[p],
                            executor=units[p].executor, guard=budget.require),
            shared.fits)
        for p in FOLLOWUP_POSITIONS}

    def backend_factory(guard: Any) -> Any:
        inner = machinery["agentic"]._default_backend_factory(
            PER_SEQUENCE_LLM_CAP)
        return runner._MeteredFastBackend(seq3.TransientRelayRetry(inner),
                                          guard=guard, billable=True)

    seed = seeded_history()
    arms = {}
    for name, snapshot in ((ARM_A, fork_snapshot), (ARM_B, revised)):
        arm = seq2.Arm(name, snapshot, machinery=machinery,
                       backend_factory=backend_factory, guard=None, cache=cache)
        for uid, rows in seed.items():
            arm.history[uid] = seq4.rehydrate(rows)
        arms[name] = arm

    stopped_at: dict[str, Any] | None = None
    for position in wanted_positions:
        for name in wanted_arms:
            try:
                unit = seq3.run_or_resume(
                    arm=arms[name], ctx=units[position], position=position,
                    phase="followup", readings=unit_readings[position],
                    machinery=machinery, budget=budget, shared=shared,
                    checkpoints=checkpoints,
                    name="followup_u%03d_%s" % (position, name.replace("-", "_")),
                    edit_probe=probe, concurrency=concurrency)
            except seq2.PackageCeiling as exc:
                stopped_at = {"kind": "PACKAGE_CEILING", "why": str(exc)}
                bank("ceiling")
                break
            bank("unit")
            del unit
        if stopped_at:
            break

    followup_units = collect_units(checkpoints)
    report["followup_units"] = followup_units
    finished = {(int(u["position"]), str(u["arm"])) for u in followup_units}
    expected = {(p, a) for p in FOLLOWUP_POSITIONS for a in (ARM_A, ARM_B)}
    report["followup"] = {
        "ran": True,
        "positions": list(FOLLOWUP_POSITIONS),
        "cells_finished": sorted("u%d/%s" % cell for cell in finished),
        "cells_missing": sorted("u%d/%s" % cell for cell in expected - finished),
        "complete": not (expected - finished),
        "this_process_ran": {"positions": list(wanted_positions),
                             "arms": list(wanted_arms)},
    }

    paired = {}
    for position in FOLLOWUP_POSITIONS:
        rows = {str(u["arm"]): u for u in followup_units
                if int(u["position"]) == position}
        if ARM_A in rows and ARM_B in rows:
            paired[str(position)] = know.paired_delayed(
                rows[ARM_A]["sequences"], rows[ARM_B]["sequences"])
    report["paired_delayed_by_unit"] = paired
    units_with_readings = [block for block in paired.values()
                           if isinstance(block.get("mean_difference"), (int, float))]
    report["main_readout"] = {
        "value": (round(sum(b["mean_difference"] for b in units_with_readings)
                        / len(units_with_readings), 6)
                  if units_with_readings else "UNKNOWN"),
        "units_averaged": len(units_with_readings),
        "rule": ("unit mean first, then equal weight over the units, on the "
                 "full restored population"),
        "what_it_is_not": (
            "not an adoption decision, and not separable from sampling unless "
            "the exposed subset is wide.  Read it beside the exposure count"),
    }
    report["status"] = ("STOPPED_AT_CEILING" if stopped_at
                        else "COMPLETE" if report["followup"]["complete"]
                        else "PARTIAL_MORE_CELLS_TO_RUN")
    report["stopped_at"] = stopped_at
    report["cost"] = budget.to_dict()
    report["returned_models"] = sorted(
        {m for unit in followup_units for row in unit.get("sequences") or ()
         for m in (row.get("returned_models") or ())})
    report["model_discipline"] = {
        "required": REQUIRED_MODEL,
        "requested_equals_returned":
            report["returned_models"] in ([], [REQUIRED_MODEL]),
    }
    report["what_this_is_not"] = (
        "not a promotion, not a population utility claim beyond the two units "
        "reported, and not evidence that a revision is the right response to "
        "a flat result.  The revision was written by Slow from the receipt; "
        "whether it helps is what the follow-up measures")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="run")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--positions", default="",
                        help="comma-separated follow-up positions this "
                             "process should run; default all")
    parser.add_argument("--arms", default="",
                        help="comma-separated arm names this process should "
                             "run; default both")
    parser.add_argument("--out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    positions = tuple(int(x) for x in args.positions.split(",") if x.strip())
    arms = tuple(x.strip() for x in args.arms.split(",") if x.strip())

    # One OS process per cell is how this package survives a host that runs
    # out of memory mid-run: the budget is the only thing that has to cross
    # the boundary, and it is carried in a file rather than re-counted.
    budget_path = CHECKPOINTS / (args.run_id or "run") / "package_budget.json"
    budget = (seq2.Budget.load(budget_path) if budget_path.is_file()
              else seq2.Budget())
    if budget_path.is_file():
        print("carried budget: fits %d llm %d wall %.0fs"
              % (budget.fits(), budget.llm(), budget.elapsed()))
    try:
        report = build(run_id=args.run_id, concurrency=args.concurrency,
                       positions=positions, arms=arms, budget=budget)
    finally:
        budget.save(budget_path)
    out = Path(args.out) if args.out else (ART / "dev_seq4_revision.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1,
                              default=str), encoding="utf-8")
    print(out)
    print("status:", report.get("status"))
    print("boundary:", (report.get("boundary") or {}).get("outcome"))
    print("surface:", ((report.get("boundary") or {}).get("proposal") or {}
                       ).get("target_surface_id"))
    print("exposure:", (report.get("exposure") or {}).get(
        "decisions_the_edit_reaches"), "/",
        (report.get("exposure") or {}).get("decisions_checked"))
    print("main readout:", (report.get("main_readout") or {}).get("value"))
    followup = report.get("followup") or {}
    if followup.get("cells_missing"):
        print("cells still to run:", followup["cells_missing"])
    print("cost:", json.dumps(report.get("cost") or {}, default=str)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
