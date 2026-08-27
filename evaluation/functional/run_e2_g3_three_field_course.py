"""G-3 -- three-field course: is the supply rung conditional, or does it just
repeat hampel everywhere?

W-1 wired ``supplies_candidates`` and the ps2p production runs converted 4/4
on GunPointOldVersusYoung at held-out +0.2127.  One field cannot separate
"experience supplies a candidate where its Scope says it applies, and Target
data decides" from "a card that fires everywhere and happened to be right".
Three fields, one question each:

* **field 1 (Scope matched, positive)** -- a *different* ROBUST hampel unit.
  Does the mechanism transfer off the single exam unit, and what does it cost
  against an A3 control?
* **field 2 (Scope mismatched, inert)** -- a unit whose deployment-visible
  pattern is outside the card's WHEN.  The supplied candidate must not be
  injected at all.  A leak here is a governance red light.
* **field 3 (veto)** -- a unit where hampel is legal and the card's WHEN
  matches, but the Target's own Support does not support it.  Support must
  draft nothing and the arm must deploy identity with zero harm.

Field selection is mechanical and happens before any live run
(``--select``): the card's 16-leaf machine applicability AST is evaluated
against each candidate unit's deployment-visible binned features, and the
sealed oracle is read as an exam key only, to write down what each field is
*expected* to show.  Nothing about the card, the thresholds or the
authorization rules is changed.

  python evaluation/functional/run_e2_g3_three_field_course.py --select
  python evaluation/functional/run_e2_g3_three_field_course.py --run
  python evaluation/functional/run_e2_g3_three_field_course.py --resume
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_ps0c_ps1 as ps0c  # noqa: E402
import run_e2_ps2_mechanical_supply as ps2  # noqa: E402
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import load_skill_entry  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    evaluate_applicability,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s1_oracle"
PS0B_JSON = E2 / "ps0b_confirmation_surface_audit.json"
PS2P_JSON = E2 / "ps2p_production_validation.json"
OUT_JSON = E2 / "g3_three_field_course.json"
OUT_MD = E2 / "g3_three_field_course.md"
CHECKPOINT = E2 / "g3_three_field_course.checkpoint.json"

PROTOCOL_VERSION = "g3_three_field_course_v1"
EVIDENCE_GRADE = "development-mechanism (pilot)"

SCOPED_SKILL_ID = ps2.SCOPED_SKILL_ID
TARGET_OPERATOR = ps2.TARGET_OPERATOR          # hampel_filter
MATERIAL = s1.MATERIAL
ROUNDS = s1.HELD_IN_ROUNDS
LLM_PER_RUN = 12
FIT_PER_RUN = 10
LLM_TOTAL_CAP = 150
FIT_TOTAL_CAP = 120
WALL_SECONDS_CAP = int(3 * 60 * 60)
REPLICATES = 4

ARM_A3 = ps2.ARM_A3
ARM_SCOPED = ps2.ARM_SCOPED

FIELD1 = "field1_scope_matched_positive"
FIELD2 = "field2_scope_mismatched_inert"
FIELD3 = "field3_veto"

# Candidate pools, declared before any machine evaluation.  Field 1 follows
# the task book's stated order of preference; the rest of each pool is scored
# and reported so the choice is auditable rather than asserted.
FIELD1_SHORTLIST = ("GunPointMaleVersusFemale__impulse_v2",
                    "PowerCons__burst_cls2")
FIELD1_POOL = FIELD1_SHORTLIST + ("PowerCons__impulse_v2",
                                  "GunPoint__impulse_v2")
FIELD2_POOL = ("ECGFiveDays__impulse_v2", "BeetleFly__impulse_v2",
               "Coffee__impulse_v2", "HouseTwenty__impulse_v2",
               "FreezerRegularTrain__impulse_v2", "MoteStrain__impulse_v2",
               "TwoLeadECG__impulse_v2", "ShapeletSim__impulse_v2",
               "DistalPhalanxOutlineCorrect__impulse_v2")
# The book named Wine (hampel legal, Support ~0, held-out class harm).  Wine
# turns out to miss the card's WHEN by one leaf, so the book's own fallback
# applies: take a unit the WHEN *does* match whose menu oracle is not hampel.
# The whole sealed roster is scanned for that, rather than a shortlist, so the
# choice cannot be steered.
FIELD3_POOL = ("Wine__impulse_v2", "Ham__impulse_v2")


def _all_sealed_unit_ids() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in ORACLE_DIR.glob("*.json")))
# GunPointOldVersusYoung is the ps2p exam unit and is excluded from field 1 by
# rule: the point of this field is that the mechanism is not specific to it.
EXCLUDED_FROM_FIELD1 = ("GunPointOldVersusYoung__impulse_v2",)


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


# =========================================================================== #
# Part 1 -- mechanical field selection (0 LLM, before any live run)
# =========================================================================== #
def _card_entry() -> Any:
    doc = json.loads(
        (E2 / "ps2_cards" / "ps2_card_scoped.json").read_text(encoding="utf-8"))
    return load_skill_entry(doc), doc


def _oracle(unit_id: str) -> dict[str, Any] | None:
    path = ORACLE_DIR / ("%s.json" % unit_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ps0b_rows() -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(PS0B_JSON.read_text(encoding="utf-8"))
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("pairs") or []:
        unit = "%s__%s" % (row["dataset"], row["injection"])
        out[(unit, str(row["program"]))] = row
    return out


def _leaf_report(ast: Mapping[str, Any],
                 features: Mapping[str, Any]) -> dict[str, Any]:
    """Which of the card's own leaves this unit meets, leaf by leaf.

    The verdict is ``evaluate_applicability`` -- the same call retrieval makes
    -- and the per-leaf list is only so a reader can see *why*.
    """
    leaves = list(ast.get("all") or [ast])
    met, missed = [], []
    for leaf in leaves:
        name = str(leaf["feature"])
        actual = features.get(name, "<absent>")
        ok, _score = evaluate_applicability(dict(leaf), features)
        (met if ok else missed).append(
            {"feature": name, "expected": leaf["value"], "actual": actual})
    matched, score = evaluate_applicability(dict(ast), features)
    return {"machine_match": bool(matched), "score": int(score),
            "leaves_total": len(leaves), "leaves_met": len(met),
            "leaves_missed": [row["feature"] for row in missed],
            "missed_detail": missed}


def _score_unit(unit_id: str, ast: Mapping[str, Any],
                ps0b: Mapping[tuple[str, str], Mapping[str, Any]]
                ) -> dict[str, Any] | None:
    oracle = _oracle(unit_id)
    if oracle is None:
        return None
    features = dict(oracle.get("public_features_binned") or {})
    report = _leaf_report(ast, features)
    row = ps0b.get((unit_id, TARGET_OPERATOR)) or {}
    programs = {str(entry.get("program")): entry
                for entry in (oracle.get("programs") or [])
                if isinstance(entry, Mapping)}
    hampel = programs.get(TARGET_OPERATOR) or {}
    return {
        "unit_id": unit_id,
        "dataset": oracle.get("dataset"),
        "injection": oracle.get("injection"),
        "series_length": oracle.get("series_length"),
        "when_axis": report,
        "oracle_set": list(oracle.get("oracle_set") or []),
        "menu_oracle_program": oracle.get("menu_oracle_program"),
        "menu_oracle_heldout_utility": oracle.get("menu_oracle_heldout_utility"),
        "hampel_in_oracle_set": TARGET_OPERATOR in (
            oracle.get("oracle_set") or []),
        "hampel_legal": bool(hampel.get("legal", hampel.get("verifier_ok"))),
        "hampel_heldin": hampel.get("heldin_utility", hampel.get("heldin")),
        "hampel_heldout": hampel.get("heldout_utility", hampel.get("heldout")),
        "hampel_modified_fraction": hampel.get("modified_fraction"),
        "confirmation_grade": row.get("grade"),
        "confirmation_margin": row.get("margin_multiplier"),
        "name_family": row.get("family"),
        "heldin_material_line": oracle.get("heldin_material_line"),
        "identity_heldout_accuracy": oracle.get("identity_heldout_accuracy"),
    }


def select_fields() -> dict[str, Any]:
    entry, doc = _card_entry()
    ast = dict(doc["observable_applicability"])
    ps0b = _ps0b_rows()

    # A unit the card was compiled *from* cannot test whether the card
    # transfers; the card names its own sources, so the exclusion is read off
    # the card rather than asserted here.
    card_sources = tuple(
        str(row["unit_id"]) for row in
        (doc["risk_guards"]["evidence"]["sources"] or ()))
    barred = set(card_sources) | set(EXCLUDED_FROM_FIELD1)

    scored = {name: [row for row in
                     (_score_unit(unit, ast, ps0b) for unit in pool)
                     if row is not None]
              for name, pool in (("field1", FIELD1_POOL),
                                 ("field2", FIELD2_POOL),
                                 ("field3_named_in_book", FIELD3_POOL),
                                 ("field3_sealed_roster_scan",
                                  _all_sealed_unit_ids()))}

    # --- field 1: matched AND ROBUST AND hampel is the oracle answer -------
    f1 = [row for row in scored["field1"]
          if row["unit_id"] in FIELD1_SHORTLIST
          and row["when_axis"]["machine_match"]
          and row["hampel_in_oracle_set"]
          and row["confirmation_grade"] == "ROBUST_LEARNABLE"
          and row["unit_id"] not in barred]
    f1.sort(key=lambda row: -float(row["confirmation_margin"] or 0.0))
    # --- field 2: the pattern axis is disjoint from the card's WHEN --------
    f2 = [row for row in scored["field2"]
          if not row["when_axis"]["machine_match"]]
    f2.sort(key=lambda row: (-len(row["when_axis"]["leaves_missed"]),
                             str(row["unit_id"])))
    # --- field 3: matched, but the Target's own answer is not hampel -------
    f3 = [row for row in scored["field3_sealed_roster_scan"]
          if row["when_axis"]["machine_match"]
          and not row["hampel_in_oracle_set"]
          and row["unit_id"] not in barred]
    # The most clearly wrong candidate first: lowest recorded held-out
    # reading for the supplied family, ties by unit id.
    f3.sort(key=lambda row: (float(row["hampel_heldout"])
                             if row["hampel_heldout"] is not None else 0.0,
                             str(row["unit_id"])))

    chosen = {
        FIELD1: f1[0] if f1 else None,
        FIELD2: f2[0] if f2 else None,
        FIELD3: f3[0] if f3 else None,
    }
    return {
        "card_skill_id": SCOPED_SKILL_ID,
        "card_machine_leaf_count": len(ast.get("all") or [ast]),
        "card_when_axis_source": (
            "artifacts/functional/e2/ps2_cards/ps2_card_scoped.json "
            "(unchanged; the W-1 dual-source card is not recompiled)"),
        "selection_rules": {
            FIELD1: ("from the task book's named shortlist: machine WHEN "
                     "match AND hampel in the sealed oracle set AND ps0b "
                     "grade ROBUST_LEARNABLE AND neither a card source nor "
                     "the ps2p exam unit; ties by the largest reproducibility "
                     "margin"),
            FIELD2: ("machine WHEN match is False -- the deployment-visible "
                     "pattern is outside the card's Scope; ranked by the most "
                     "missed leaves"),
            FIELD3: ("whole sealed roster: machine WHEN match AND hampel is "
                     "not the unit's oracle answer, so the Target's own "
                     "Support is expected to refuse it; ranked by the lowest "
                     "recorded held-out reading for the supplied family"),
        },
        "barred_units": sorted(barred),
        "card_source_units": list(card_sources),
        "excluded_from_field1": list(EXCLUDED_FROM_FIELD1),
        "when_axis_discrimination": {
            "sealed_units_scored": len(scored["field3_sealed_roster_scan"]),
            "sealed_units_matching_when": sum(
                1 for row in scored["field3_sealed_roster_scan"]
                if row["when_axis"]["machine_match"]),
            "of_those_hampel_is_the_oracle_answer": sum(
                1 for row in scored["field3_sealed_roster_scan"]
                if row["when_axis"]["machine_match"]
                and row["hampel_in_oracle_set"]),
            "note": (
                "a Scope readout, not a claim: the card's 16-leaf WHEN "
                "selects a small minority of the roster, and hampel is the "
                "sealed answer on most of what it selects."),
        },
        "scored_pools": scored,
        "chosen": chosen,
    }


def _unit_spec(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"unit_id": str(row["unit_id"]), "dataset": str(row["dataset"]),
            "injection": str(row["injection"]),
            "series_length": row.get("series_length")}


def _preregistration(selection: Mapping[str, Any]) -> dict[str, Any]:
    chosen = selection["chosen"]
    return {
        FIELD1: {
            "unit_id": (chosen[FIELD1] or {}).get("unit_id"),
            "arms": [ARM_A3, ARM_SCOPED],
            "injections_expected": "4/4 in A5-scoped, 0/4 in A3 (no card)",
            "conversions_expected": ">= 2/4 deployed hampel with positive "
                                    "held-out gain",
            "deployment_expected": "hampel_filter",
            "harm_expected": 0,
        },
        FIELD2: {
            "unit_id": (chosen[FIELD2] or {}).get("unit_id"),
            "arms": [ARM_SCOPED],
            "injections_expected": "0/4 -- the card is out of Scope and must "
                                   "not be retrieved or injected",
            "conversions_expected": 0,
            "deployment_expected": "identity",
            "harm_expected": 0,
        },
        FIELD3: {
            "unit_id": (chosen[FIELD3] or {}).get("unit_id"),
            "arms": [ARM_SCOPED],
            "injections_expected": "4/4 -- Scope matches, so the candidate is "
                                   "supplied and probed",
            "conversions_expected": "0/4 -- Support is not a material "
                                    "positive, so nothing drafts",
            "deployment_expected": "identity",
            "harm_expected": 0,
        },
    }


def _run_plan(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The sixteen runs, ordered governance-first.

    Run ids and their field/arm/unit bindings are exactly the book's; only
    the execution order is governance-first.  Field 2 (Scope leak) and field
    3 (veto failure) are the red lights that outrank everything else, so if
    the wall clock truncates this book it must truncate the field that is
    merely a positive reading, not the two that are safety checks.  Every run
    starts from a fresh arm state, so order is not an experimental variable.
    """
    chosen = selection["chosen"]
    field2 = [{"run_id": "g3_f2_%d" % (index + 1), "field": FIELD2,
               "arm": ARM_SCOPED, "replicate": index + 1,
               "unit_id": chosen[FIELD2]["unit_id"]}
              for index in range(REPLICATES)]
    field3 = [{"run_id": "g3_f3_%d" % (index + 1), "field": FIELD3,
               "arm": ARM_SCOPED, "replicate": index + 1,
               "unit_id": chosen[FIELD3]["unit_id"]}
              for index in range(REPLICATES)]
    field1_a5 = [{"run_id": "g3_f1_a5_%d" % (index + 1), "field": FIELD1,
                  "arm": ARM_SCOPED, "replicate": index + 1,
                  "unit_id": chosen[FIELD1]["unit_id"]}
                 for index in range(REPLICATES)]
    field1_a3 = [{"run_id": "g3_f1_a3_%d" % (index + 1), "field": FIELD1,
                  "arm": ARM_A3, "replicate": index + 1,
                  "unit_id": chosen[FIELD1]["unit_id"]}
                 for index in range(REPLICATES)]
    plan: list[dict[str, Any]] = []
    plan.extend(field2)
    plan.extend(field3)
    # Interleave the field-1 arms so a truncation keeps the control paired.
    for a5_row, a3_row in zip(field1_a5, field1_a3):
        plan.append(a5_row)
        plan.append(a3_row)
    return plan


# =========================================================================== #
# Part 2 -- the sixteen runs
# =========================================================================== #
def _score_run(plan: Mapping[str, Any], result: Mapping[str, Any],
               base_sha: str) -> dict[str, Any]:
    arm = str(plan["arm"])
    public = s1._public_unit_result(result)
    skill_ids = (SCOPED_SKILL_ID,) if arm == ARM_SCOPED else ()
    anatomies = [ps2._round_anatomy(record, skill_ids)
                 for record in public.get("rounds") or []]
    if arm == ARM_SCOPED:
        inject = ps2._inject_funnel(public, skill_id=SCOPED_SKILL_ID,
                                    operator=TARGET_OPERATOR)
    else:
        inject = {"injected_id": None, "entered_pool": False,
                  "selected_by_agent": False, "passed_verifier": False,
                  "support_material_positive": False,
                  "delayed_approved": False, "deployed": False,
                  "break_at": "no_card", "rows": []}
    card_seen = sorted({
        skill_id for record in public.get("rounds") or []
        for skill_id in record.get("retrieved_skill_ids") or []
        if skill_id == SCOPED_SKILL_ID})
    deployment = public.get("deployment") or {}
    deltas = deployment.get("heldout_recall_delta_by_class") or {}
    worst = min((float(v) for v in deltas.values()), default=0.0)
    probes = sum(len(record.get("probes") or [])
                 for record in public.get("rounds") or [])
    first_skill_round = next(
        (str(record["round"]) for record in public.get("rounds") or []
         if record.get("winner_delayed_approved")), None)
    return {
        "run_id": plan["run_id"],
        "field": plan["field"],
        "arm": arm,
        "unit_id": plan["unit_id"],
        "replicate": plan["replicate"],
        "base_runtime_bundle_sha": base_sha,
        "card_in_fast_view": card_seen,
        "inject_funnel": inject,
        "pool_anatomy": anatomies,
        "agent_proposal_families": sorted({
            family for anatomy in anatomies
            for family in anatomy["agent_families"]}),
        "exploration_slot_kept": all(
            anatomy["exploration_slot_kept"] for anatomy in anatomies)
        if anatomies else True,
        "supply_without_agent_program_rounds": sum(
            1 for anatomy in anatomies
            if anatomy.get("supply_without_agent_program")),
        "rounds_without_agent_program": sum(
            1 for anatomy in anatomies
            if int(anatomy.get("agent_program_count") or 0) == 0),
        "llm_calls": result.get("llm_calls"),
        "consumer_fits": result.get("consumer_fits"),
        "probes": probes,
        "first_approved_round": first_skill_round,
        "seconds": result.get("seconds"),
        "deployment": {
            "deploy_source": deployment.get("deploy_source"),
            "applied_program": deployment.get("applied_program"),
            "heldout_accuracy_gain": deployment.get("heldout_accuracy_gain"),
            "worst_class_delta": worst,
            "harm_event": bool(worst < -cls.HARM_BAR),
        },
        "rounds": public.get("rounds"),
    }


def _checkpoint(runs, ledger, base_shas, *, started=None) -> None:
    CHECKPOINT.write_text(
        json.dumps(ps0c.redact({
            "runs": list(runs), "ledger": dict(ledger),
            "base_shas": dict(base_shas),
            "completed_run_ids": [row["run_id"] for row in runs],
            "wall_seconds_used": (round(time.time() - started, 1)
                                  if started is not None else None),
        }), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_all(*, selection, h0, store_root: Path, ledger, started, runs):
    card = json.loads(
        (E2 / "ps2_cards" / "ps2_card_scoped.json").read_text(encoding="utf-8"))
    scoped_snapshot, _applied = s1._apply_entries(
        h0, [card], store_root=store_root / "bases", tag="a5_scoped")
    bases = {ARM_A3: h0, ARM_SCOPED: scoped_snapshot}
    base_shas = {ARM_A3: h0.runtime_bundle_sha,
                 ARM_SCOPED: scoped_snapshot.runtime_bundle_sha}
    cells: dict[str, Any] = {}
    done = {str(row["run_id"]) for row in runs}
    for plan in _run_plan(selection):
        if plan["run_id"] in done:
            print("skip %s (checkpoint)" % plan["run_id"], flush=True)
            continue
        if ledger["llm"] >= LLM_TOTAL_CAP or ledger["fit"] >= FIT_TOTAL_CAP:
            raise Stop("COMPUTE_BUDGET_EXCEEDED",
                       "book cap reached before %s" % plan["run_id"])
        if time.time() - started > WALL_SECONDS_CAP:
            raise Stop("COMPUTE_BUDGET_EXCEEDED",
                       "wall clock cap reached before %s" % plan["run_id"])
        unit_id = str(plan["unit_id"])
        if unit_id not in cells:
            row = next(entry for entry in selection["chosen"].values()
                       if entry and entry["unit_id"] == unit_id)
            cells[unit_id] = s1._build_cell(_unit_spec(row))
        arm = plan["arm"]
        backend = cls._live_backend(LLM_PER_RUN)
        row = next(entry for entry in selection["chosen"].values()
                   if entry and entry["unit_id"] == unit_id)
        result = ps2._run_unit_with_retry(
            unit=_unit_spec(row), cell=cells[unit_id], arm=arm,
            base_snapshot=bases[arm], backend=backend,
            store_root=store_root / plan["run_id"])
        ledger["llm"] += int(result.get("llm_calls") or 0)
        ledger["fit"] += int(result.get("consumer_fits") or 0)
        scored = _score_run(plan, result, base_shas[arm])
        runs.append(scored)
        _checkpoint(runs, ledger, base_shas, started=started)
        inject = scored["inject_funnel"]
        print("%-12s %-6s %-34s inject=%-5s support=%-5s deployed=%-5s "
              "gain=%+.4f worst=%+.4f llm=%s"
              % (plan["run_id"], arm, unit_id, inject.get("entered_pool"),
                 inject.get("support_material_positive"),
                 inject.get("deployed"),
                 float(scored["deployment"]["heldout_accuracy_gain"] or 0.0),
                 float(scored["deployment"]["worst_class_delta"] or 0.0),
                 scored.get("llm_calls")), flush=True)
    return base_shas


# =========================================================================== #
# Part 3 -- pre-registered verdicts
# =========================================================================== #
def _field_rows(runs, field, arm=None):
    return [row for row in runs if row["field"] == field
            and (arm is None or row["arm"] == arm)]


def _field_summary(runs, field, arm):
    rows = _field_rows(runs, field, arm)
    gains = [float(row["deployment"]["heldout_accuracy_gain"] or 0.0)
             for row in rows]
    return {
        "arm": arm, "runs": len(rows),
        "entered_pool": sum(1 for row in rows
                            if row["inject_funnel"]["entered_pool"]),
        "support_material_positive": sum(
            1 for row in rows
            if row["inject_funnel"]["support_material_positive"]),
        "deployed_target_operator": sum(
            1 for row in rows if row["inject_funnel"]["deployed"]),
        "deployed_identity": sum(
            1 for row in rows if not row["deployment"]["applied_program"]),
        "harm_runs": sum(1 for row in rows
                         if row["deployment"]["harm_event"]),
        "heldout_gains": gains,
        "mean_heldout_gain": (sum(gains) / len(gains)) if gains else 0.0,
        "worst_class_deltas": [
            float(row["deployment"]["worst_class_delta"] or 0.0)
            for row in rows],
        "card_in_fast_view_runs": sum(1 for row in rows
                                      if row["card_in_fast_view"]),
        "llm": sum(int(row["llm_calls"] or 0) for row in rows),
        "consumer_fits": sum(int(row["consumer_fits"] or 0) for row in rows),
        "probes": sum(int(row["probes"] or 0) for row in rows),
        "runs_reaching_an_approved_skill": sum(
            1 for row in rows if row["first_approved_round"]),
        "agent_families": sorted({family for row in rows
                                  for family in row["agent_proposal_families"]}),
    }


def _verdict(runs, *, stopped):
    if stopped:
        return {"verdict": stopped, "reason": "stopped before the full plan"}
    f1_a5 = _field_summary(runs, FIELD1, ARM_SCOPED)
    f1_a3 = _field_summary(runs, FIELD1, ARM_A3)
    f2 = _field_summary(runs, FIELD2, ARM_SCOPED)
    f3 = _field_summary(runs, FIELD3, ARM_SCOPED)
    facts = {"field1_a5": f1_a5, "field1_a3": f1_a3,
             "field2": f2, "field3": f3,
             "field1_gain_delta": round(
                 f1_a5["mean_heldout_gain"] - f1_a3["mean_heldout_gain"], 6)}
    if not (f1_a5["runs"] and f1_a3["runs"] and f2["runs"] and f3["runs"]):
        return {"verdict": "COMPUTE_BUDGET_EXCEEDED", "facts": facts,
                "reason": "not every field completed its replicates"}
    # Governance red lights first: they outrank the positive field.
    if f2["entered_pool"] > 0:
        return {"verdict": "FIELD2_SCOPE_LEAK", "facts": facts,
                "reason": (
                    "the card supplied a candidate on a unit outside its "
                    "machine Scope in %d/%d runs.  A Scope that does not hold "
                    "is a governance failure and outranks any positive "
                    "reading." % (f2["entered_pool"], f2["runs"]))}
    if f3["deployed_target_operator"] > 0 or f3["harm_runs"] > 0:
        return {"verdict": "FIELD3_VETO_FAILED", "facts": facts,
                "reason": (
                    "the veto field deployed the supplied family in %d/%d "
                    "runs (harm runs %d).  Target data did not get the last "
                    "word." % (f3["deployed_target_operator"], f3["runs"],
                               f3["harm_runs"]))}
    if f1_a5["deployed_target_operator"] < 2 or f1_a5["mean_heldout_gain"] <= 0:
        return {"verdict": "FIELD1_NO_CONVERSION", "facts": facts,
                "reason": (
                    "the Scope-matched field converted %d/%d with mean "
                    "held-out gain %+.4f; the mechanism did not reproduce off "
                    "the ps2p exam unit."
                    % (f1_a5["deployed_target_operator"], f1_a5["runs"],
                       f1_a5["mean_heldout_gain"]))}
    return {"verdict": "GATE3_PASSED", "facts": facts,
            "reason": (
                "Scope matched: %d/%d converted at mean held-out %+.4f "
                "against A3 %+.4f.  Scope mismatched: %d/%d injected -- the "
                "card stayed out.  Veto: %d/%d probed and %d/%d deployed the "
                "family, identity deployed, harm %d.  The supply rung is "
                "conditional on the card's own Scope and is adjudicated by "
                "Target data."
                % (f1_a5["deployed_target_operator"], f1_a5["runs"],
                   f1_a5["mean_heldout_gain"], f1_a3["mean_heldout_gain"],
                   f2["entered_pool"], f2["runs"],
                   f3["entered_pool"], f3["runs"],
                   f3["deployed_target_operator"], f3["runs"],
                   f3["harm_runs"]))}


def _markdown(payload) -> str:
    verdict = payload["verdict"]
    chosen = payload["selection"]["chosen"]
    lines = [
        "# G-3 -- three-field course for the supply rung",
        "",
        "protocol: `%s`  evidence grade: **%s**  git: `%s`"
        % (payload["protocol_version"], payload["evidence_grade"],
           payload["git_head"]),
        "", "**%s**" % verdict["verdict"], "", verdict.get("reason", ""), "",
        "> %s" % payload["semantic_discipline"], "",
        "## 1. Field selection (mechanical, before any live run)", "",
        "| field | unit | WHEN leaves met | machine match | hampel in oracle "
        "set | ps0b grade | margin | menu oracle |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for field in (FIELD1, FIELD2, FIELD3):
        row = chosen.get(field)
        if not row:
            lines.append("| %s | (none) | | | | | | |" % field)
            continue
        when = row["when_axis"]
        lines.append("| %s | `%s` | %d/%d | **%s** | %s | %s | %s | `%s` |" % (
            field, row["unit_id"], when["leaves_met"], when["leaves_total"],
            when["machine_match"], row["hampel_in_oracle_set"],
            row["confirmation_grade"], row["confirmation_margin"],
            row["menu_oracle_program"]))
    lines += ["", "Missed leaves for the mismatched field: `%s`"
              % ", ".join((chosen.get(FIELD2) or {}).get(
                  "when_axis", {}).get("leaves_missed") or ["-"]), ""]
    lines += ["## 2. Pre-registration", ""]
    for field, row in payload["preregistration"].items():
        lines.append("- **%s** (`%s`): inject %s; conversions %s; deploy %s; "
                     "harm %s" % (field, row["unit_id"],
                                  row["injections_expected"],
                                  row["conversions_expected"],
                                  row["deployment_expected"],
                                  row["harm_expected"]))
    lines += ["", "## 3. Sixteen-run protocol table", "",
              "| run | field | arm | unit | inject | Support+ | delayed | "
              "deployed | applied | held-out gain | worst class | LLM | fits |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in payload["runs"]:
        inject = row["inject_funnel"]
        applied = ",".join(step["op"] for step
                           in row["deployment"]["applied_program"]) or "identity"
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | `%s` | %+.4f | "
                     "%+.4f | %s | %s |" % (
                         row["run_id"], row["field"].split("_")[0], row["arm"],
                         row["unit_id"].split("__")[0],
                         inject["entered_pool"],
                         inject["support_material_positive"],
                         inject["delayed_approved"], inject["deployed"],
                         applied,
                         float(row["deployment"]["heldout_accuracy_gain"] or 0),
                         float(row["deployment"]["worst_class_delta"] or 0),
                         row["llm_calls"], row["consumer_fits"]))
    lines += ["", "## 4. Field summaries", "",
              "| field | arm | runs | entered | Support+ | deployed family | "
              "identity deploys | mean held-out | harm | LLM | fits | probes |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, summary in payload["field_summaries"].items():
        lines.append("| %s | %s | %d | %d | %d | %d | %d | %+.4f | %d | %d | "
                     "%d | %d |" % (
                         label, summary["arm"], summary["runs"],
                         summary["entered_pool"],
                         summary["support_material_positive"],
                         summary["deployed_target_operator"],
                         summary["deployed_identity"],
                         summary["mean_heldout_gain"], summary["harm_runs"],
                         summary["llm"], summary["consumer_fits"],
                         summary["probes"]))
    ledger = payload["ledger"]
    lines += ["", "## 5. Cost", "",
              "- LLM: %s / %s" % (ledger["llm"], ledger["llm_cap"]),
              "- Consumer fits: %s / %s" % (ledger["fit"], ledger["fit_cap"]),
              "- wall: %s s / %s s" % (ledger["wall_seconds"],
                                       ledger["wall_seconds_cap"]),
              "- downloads: 0", "", "## 6. Obligations", ""]
    for key, value in payload["obligations"].items():
        lines.append("- **%s**: %s" % (key, value))
    lines += ["", "## 7. Outside the book", ""]
    for note in payload["outside_book"]:
        lines.append("- %s" % note)
    return "\n".join(lines) + "\n"


def run(*, select_only: bool = False, resume: bool = False,
        finalize: bool = False) -> int:
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    selection = select_fields()
    missing = [field for field, row in selection["chosen"].items()
               if row is None]
    if missing:
        raise Stop("FIELD_SELECTION_EMPTY",
                   "no candidate satisfied: %s" % ", ".join(missing))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "ps2p_source": PS2P_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "ps0b_source": PS0B_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "selection": selection,
        "preregistration": _preregistration(selection),
        "run_plan": _run_plan(selection),
        "semantic_discipline": (
            "a conversion is experience supplying a candidate through the "
            "mechanical channel, adjudicated by Target feedback.  It is not "
            "evidence that the agent learned to propose the family, and a "
            "positive earned under the card counts zero toward cross-domain "
            "authorization for the Source Skill."),
    }
    if select_only:
        payload.update({"runs": [], "field_summaries": {},
                        "verdict": {"verdict": "SELECTION_ONLY",
                                    "reason": "fields selected; no live run"},
                        "ledger": {"llm": 0, "llm_cap": LLM_TOTAL_CAP,
                                   "fit": 0, "fit_cap": FIT_TOTAL_CAP,
                                   "wall_seconds": round(
                                       time.time() - started, 1),
                                   "wall_seconds_cap": WALL_SECONDS_CAP,
                                   "downloads": 0},
                        "obligations": {"select_only": True, "no_llm": True},
                        "outside_book": []})
        s1._dump(OUT_JSON, ps0c.redact(payload))
        OUT_MD.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps({"verdict": "SELECTION_ONLY",
                          "chosen": {field: (row or {}).get("unit_id")
                                     for field, row in
                                     selection["chosen"].items()}},
                         ensure_ascii=False, indent=1))
        return 0

    if finalize:
        # Render the artifact from whatever the checkpoint holds.  Used when
        # the wall-clock cap ends the book before the plan does: the runs are
        # already persisted, and nothing here re-runs or re-scores them.
        saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        runs = list(saved.get("runs") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm") or 0),
                  "fit": int((saved.get("ledger") or {}).get("fit") or 0)}
        planned = {row["run_id"] for row in payload["run_plan"]}
        done = {row["run_id"] for row in runs}
        stopped = None if planned <= done else "WALL_CLOCK_TRUNCATED"
        payload["truncation"] = {
            "planned": sorted(planned), "completed": sorted(done),
            "not_run": sorted(planned - done),
            "reason": ("the book's wall-clock cap ended the run; the plan was "
                       "ordered governance-first so the truncated tail is the "
                       "positive field, not a red-light field"),
        }
        return _finish(payload, runs, ledger,
                       base_shas=dict(saved.get("base_shas") or {}),
                       stopped=stopped,
                       started=time.time() - float(
                           saved.get("wall_seconds_used") or 0.0))

    install = ps0c.install_new_backend()
    payload["backend_install"] = {"host": install.get("host"),
                                  "model": install.get("model")}
    ledger = {"llm": 0, "fit": 0}
    store_root = Path(tempfile.gettempdir()) / "g3_arms"
    stopped: str | None = None
    runs: list[dict[str, Any]] = []
    base_shas: dict[str, str] = {}
    if resume and CHECKPOINT.is_file():
        saved = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        runs = list(saved.get("runs") or [])
        ledger = {"llm": int((saved.get("ledger") or {}).get("llm") or 0),
                  "fit": int((saved.get("ledger") or {}).get("fit") or 0)}
        base_shas = dict(saved.get("base_shas") or {})
        started = time.time() - float(saved.get("wall_seconds_used") or 0.0)
        payload["resumed_from_checkpoint"] = {
            "completed_run_ids": [row["run_id"] for row in runs],
            "ledger": dict(ledger)}
    elif store_root.exists():
        shutil.rmtree(store_root)
    try:
        probe = ps0c.probe_new_backend()
        payload["backend_probe"] = ps0c.redact(probe)
        if not probe.get("ok"):
            raise Stop("BACKEND_UNAVAILABLE",
                       "relay probe failed: %s" % probe.get("reason"))
        from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
            compile_snapshot,
        )
        h0 = compile_snapshot(
            PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
            verify_lock=False)
        base_shas = _run_all(selection=selection, h0=h0,
                             store_root=store_root, ledger=ledger,
                             started=started, runs=runs)
    except Stop as stop:
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except s1.Stop as stop:  # noqa: PERF203
        stopped = stop.verdict
        payload["stop"] = {"verdict": stop.verdict, "reason": stop.reason}
    except Exception as exc:  # noqa: BLE001
        import traceback
        stopped = "INSTRUMENT_UNREADABLE"
        payload["stop"] = {"verdict": stopped,
                           "reason": "%s: %s" % (type(exc).__name__, exc),
                           "traceback": traceback.format_exc()}
    return _finish(payload, runs, ledger, base_shas=base_shas,
                   stopped=stopped, started=started)


def _finish(payload, runs, ledger, *, base_shas, stopped, started) -> int:
    payload["runs"] = runs
    payload["field_summaries"] = {
        "field1 / A5-scoped": _field_summary(runs, FIELD1, ARM_SCOPED),
        "field1 / A3": _field_summary(runs, FIELD1, ARM_A3),
        "field2 / A5-scoped": _field_summary(runs, FIELD2, ARM_SCOPED),
        "field3 / A5-scoped": _field_summary(runs, FIELD3, ARM_SCOPED),
    }
    payload["verdict"] = _verdict(runs, stopped=stopped)
    payload["base_shas"] = base_shas
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
    }
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    payload["obligations"] = {
        "fields_selected_before_any_live_run": True,
        "card_unchanged_from_w1": True,
        "thresholds_and_authorization_unmodified": True,
        "runtime_contracts_operators_unmodified": True,
        "no_new_skill_class_or_permission_platform": True,
        "grants_execution_false": True,
        "oracle_read_as_exam_key_only": True,
        "oracle_not_loaded_into_harness": True,
        "guided_positive_counts_zero_toward_source_auth": True,
        "downloads": 0,
        "full_repo_pytest_not_run": True,
        "semantic_discipline": payload["semantic_discipline"],
    }
    payload["outside_book"] = payload.get("outside_book") or []
    s1._dump(OUT_JSON, ps0c.redact(payload))
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "reason": payload["verdict"].get("reason"),
                      "ledger": payload["ledger"],
                      "artifact": str(OUT_JSON)},
                     ensure_ascii=False, indent=1), flush=True)
    return 0 if payload["verdict"]["verdict"] in (
        "GATE3_PASSED", "FIELD1_NO_CONVERSION", "FIELD2_SCOPE_LEAK",
        "FIELD3_VETO_FAILED") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="G-3 three-field course")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--finalize", action="store_true",
                        help="render the artifact from the checkpoint, no "
                             "backend and no new runs")
    args = parser.parse_args()
    if not (args.select or args.run or args.resume or args.finalize):
        parser.error("pass --select, --run, --resume or --finalize")
    return run(select_only=args.select, resume=bool(args.resume),
               finalize=bool(args.finalize))


if __name__ == "__main__":
    raise SystemExit(main())
