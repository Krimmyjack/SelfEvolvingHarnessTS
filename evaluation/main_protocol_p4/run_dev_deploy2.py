"""DEV-DEPLOY-2: does feedback Slow already legally has improve frozen Fast?

See ``docs/DEV_DEPLOY2_FEEDBACK_TO_FROZEN_FAST_TASK_2026-09-09.md`` for the
full task book.  This module builds the pooled neutral formation table from
DEV-DEPLOY-1 Part B's 120 real decisions, runs six independent Slow revision
sessions (R/C/C-minus x g1/g2) from the same K0 parent snapshot, checks
whether each compiled candidate is exposed on u12/u13, runs the frozen
Fast-only comparison (F/R/C/C-minus x g1/g2 x u12/u13 x 20 series) and scores
it against raw and freshly-calibrated D-new-fixed/D-new-safe.

Reused rather than re-derived
------------------------------
``run_dev_deploy1.run_fastonly_decision`` is called verbatim for every Fast
decision in this package -- it already takes ``snapshot`` as a parameter, so
"which knowledge version" is the only thing that changes.  The Slow-side
primitives (``build_catalog``, ``propose_update``, ``apply_update``,
``exposure_check``, ``action_facts``, ``edit_preflight``) are
``dev_seq2_knowledge``'s, unmodified.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_deploy1 as deploy1
from evaluation.main_protocol_p4 import run_dev_seq2 as seq2
from evaluation.main_protocol_p4 import run_dev_seq3 as seq3
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ROOT / "artifacts" / "main_protocol"
SCRATCH = base.ROOT / "_scratch" / "dev_deploy2"
FORMATION_POSITIONS = (9, 10, 11)
REVIEW_POSITIONS = (12, 13)
ROSTER = ("T14", "T140", "T141", "T143", "T144", "T145", "T146", "T147",
         "T149", "T15", "T150", "T151", "T152", "T153", "T155", "T156",
         "T157", "T158", "T159", "T16")

REQUIRED_MODEL = deploy1.REQUIRED_MODEL
PER_SEQUENCE_LLM_CAP = deploy1.PER_SEQUENCE_LLM_CAP
SLOW_LLM_CAP = 16
PER_GROUP_ARM_LLM_CAP = 450
MAX_FITS = 2000
MAX_LLM = 4000
MAX_WALL_SECONDS = 8 * 3600
FORMATION_SUPPLEMENT_FIT_CAP = 300
CONCURRENCY = 2

GROUPS = ("g1", "g2")
GROUP_SEEDS = {"g1": 2026090903, "g2": 2026090904}
SLOW_ORDER = {"g1": ("R", "C", "C-minus"), "g2": ("C-minus", "C", "R")}
FAST_ARM_ORDER = {"g1": ("F", "R", "C", "C-minus"),
                  "g2": ("C-minus", "C", "R", "F")}
KNOWN_KNOWLEDGE_SHA = "98dea3b037b790528cd745f8cbdf657da0285c25f91df53e9c572a0dfc87563b"

PackageCeiling = deploy1.PackageCeiling
AccountFault = deploy1.AccountFault


# ---------------------------------------------------------------------------
# scorer over the formation (9/10/11) and review (12/13) positions
# ---------------------------------------------------------------------------

def _unit_dict(position: int, g2doc: Mapping[str, Any],
              review_units: Mapping[int, Any]) -> dict[str, Any]:
    if position in FORMATION_POSITIONS:
        return deploy1.unit_descriptor(position, g2doc)
    return review_units[position]


class Scorer:
    """The same replay-cache scoring surface as Part 1's, over this package's
    own five positions (9/10/11 formation, 12/13 review)."""

    def __init__(self) -> None:
        import threading

        self._g2doc = deploy1.load_g2()
        doc = base.load(R.ORDERING)
        rows, _excluded = R._population(doc)
        self._review_units = {int(row["position"]): row["unit"] for row in rows
                              if int(row["position"]) in REVIEW_POSITIONS}
        self.cache = runner.ReplayPredictionCache("dev_deploy2")
        self._lock = threading.Lock()
        self.ctx: dict[int, Any] = {}
        self.readings: dict[int, ps.UnitReadings] = {}
        for position in (*FORMATION_POSITIONS, *REVIEW_POSITIONS):
            unit = _unit_dict(position, self._g2doc, self._review_units)
            ctx = runner.UnitContext(unit)
            self.ctx[position] = ctx
            self.readings[position] = ps.UnitReadings(
                cache=self.cache, ctx=ctx, executor=ctx.executor)

    @property
    def physical_fits(self) -> int:
        return int(self.cache.physical_fits)

    def population(self, position: int) -> list[str]:
        return list(self.ctx[position].eval_uids)

    def sequence_reading(self, position: int, uid: str, steps: tuple,
                         origin: int) -> Any:
        with self._lock:
            return ps.SequenceView(self.readings[position], uid).evaluate(
                steps, int(origin))

    def population_reading(self, position: int, steps: tuple,
                           origin: int) -> dict[str, Any] | None:
        uids = self.population(position)
        if not steps:
            return {"identity": True, "treated": 0, "served": len(uids),
                    "per_series_gain": {uid: 0.0 for uid in uids},
                    "aggregate_gain": 0.0, "harmed_fraction": 0.0,
                    "max_single_series_harm": 0.0}
        try:
            with self._lock:
                return self.readings[position].reading(
                    steps, int(origin), uids)[0]
        except runner.UnitFault:
            return None


# ---------------------------------------------------------------------------
# the pooled neutral formation table (120 rows: g1+g2 x u9/u10/u11)
# ---------------------------------------------------------------------------

def _load_part1_checkpoint(group: str, position: int) -> list[dict[str, Any]]:
    path = (base.ROOT / "_scratch" / "dev_deploy1" / group
           / ("u%03d.json" % position))
    doc = json.loads(path.read_text(encoding="utf-8"))
    return list(doc["sequences"])


def _reconstruct_typed_steps(scorer: Scorer, row: Mapping[str, Any],
                             position: int) -> tuple[tuple | None, bool | None]:
    """The row's own deployed program, as typed steps -- self-checked, never
    trusted from the label alone (see run_dev_deploy1's own reconstruction
    method for why this is lossless for this project's typed DSL)."""
    label = row.get("deployed_program") or "identity"
    if label == "identity":
        return (), None
    try:
        steps = deploy1.parse_program_label(label)
    except ValueError:
        return None, False
    origin = int(row["origin"])
    uid = str(row["series_uid"])
    receipt = scorer.sequence_reading(position, uid, steps, origin)
    recorded = row.get("deployment_gain_at_origin")
    if not isinstance(recorded, (int, float)) or receipt.gain is None:
        return None, False
    verified = abs(float(receipt.gain) - float(recorded)) <= deploy1.TOL
    return (steps if verified else None), verified


def load_pooled_formation(scorer: Scorer) -> dict[str, Any]:
    rows_out: list[dict[str, Any]] = []
    unverified = 0
    n_invalid_mechanism = 0
    for group in GROUPS:
        for position in FORMATION_POSITIONS:
            for row in _load_part1_checkpoint(group, position):
                annotated = deploy1.annotate_recorded_decision(row)
                if not annotated.get("valid_mechanism_decision"):
                    # Do not reconstruct the historical identity/0 fallback
                    # as a real identity program or as utility evidence.
                    steps, verified = None, False
                    n_invalid_mechanism += 1
                else:
                    steps, verified = _reconstruct_typed_steps(
                        scorer, row, position)
                if steps is None:
                    unverified += 1
                uid = str(row["series_uid"])
                origin = int(row["origin"])
                facts_support = (know.action_facts(scorer.ctx[position], origin,
                                                   steps, uid)
                                 if steps is not None else "UNKNOWN")
                dedup_key = "|".join([
                    uid, str(origin),
                    ps.program_signature(steps) if steps is not None
                    else "UNRECONSTRUCTED",
                    "origin"])
                rows_out.append({
                    "source_group": group, "position": position,
                    "series_uid": uid, "origin": origin,
                    "delayed_origin": int(row["delayed_origin"]),
                    "public_features": row.get("public_features"),
                    "chosen_candidate_id": row.get("chosen_candidate_id"),
                    "candidate_programs": row.get("candidate_programs"),
                    "deploy_status": annotated.get("original_deploy_status")
                                     or row.get("deploy_status"),
                    "valid_mechanism_decision": annotated.get(
                        "valid_mechanism_decision"),
                    "original_record_semantics": annotated.get(
                        "original_record_semantics"),
                    "deployed_program": row.get("deployed_program"),
                    "deployed_program_steps_reconstructed": (
                        [{"op": o, "params": p} for o, p in steps]
                        if steps is not None else "UNKNOWN"),
                    "reconstruction_verified": verified,
                    "deployment_gain_at_origin": annotated.get(
                        "deployment_gain_at_origin"),
                    "fixed_program_gain_at_plus48": annotated.get(
                        "fixed_program_gain_at_plus48"),
                    "original_recorded_gain_at_origin": annotated.get(
                        "original_deployment_gain_at_origin"),
                    "original_recorded_gain_at_plus48": annotated.get(
                        "original_fixed_program_gain_at_plus48"),
                    "action_facts": facts_support,
                    "dedup_key": dedup_key,
                    "llm_requests_sent": row.get("llm_requests_sent"),
                })
    return {"rows": rows_out, "n_rows": len(rows_out),
           "n_unreconstructed": unverified,
           "n_invalid_mechanism_decision": n_invalid_mechanism,
           "note": ("120 rows: DEV-DEPLOY-1 Part B's g1+g2, u9/u10/u11, all "
                    "20 series each.  Typed steps for the delivered program "
                    "are reconstructed from the recorded label and "
                    "self-checked against the already-recorded origin gain "
                    "before being trusted (0 new LLM either way).  Historical "
                    "EMPTY_SELECTION_FALLBACK_TO_IDENTITY / "
                    "UNKNOWN_CANDIDATE_FALLBACK_TO_IDENTITY rows are marked "
                    "invalid and are not treated as identity/0 evidence.")}


# ---------------------------------------------------------------------------
# the mechanical public menu's readings at the same formation contexts
# ---------------------------------------------------------------------------

def menu_readings_at_formation(scorer: Scorer) -> dict[str, Any]:
    menu = deploy1.legal_forecast_menu()
    per_position: dict[int, Any] = {}
    for position in FORMATION_POSITIONS:
        ctx = scorer.ctx[position]
        origin = int(ctx.origin)
        period = int(ctx.config["period"])
        per_candidate: dict[str, Any] = {"identity": {
            "program_steps": [],
            "status": "READ", "aggregate_gain": 0.0, "harmed_fraction": 0.0,
            "max_single_series_harm": 0.0,
            "per_series_gain": {uid: 0.0 for uid in scorer.population(position)}}}
        for op in menu:
            steps = deploy1.candidate_steps(op, period)
            required = (deploy1.OPERATOR_METADATA[op].get(
                "public_parameter_schema") or {}).get("required") or []
            if any(k != "period" for k in required):
                per_candidate[op] = {"status": "UNAVAILABLE"}
                continue
            reading = scorer.population_reading(position, steps, origin)
            if reading is None:
                per_candidate[op] = {"status": "LEGALITY_REJECTED",
                                     "program_steps": [
                                         {"op": o, "params": p} for o, p in steps]}
                continue
            per_candidate[op] = {
                "program_steps": [{"op": o, "params": p} for o, p in steps],
                "status": "READ", "aggregate_gain": reading["aggregate_gain"],
                "harmed_fraction": reading["harmed_fraction"],
                "max_single_series_harm": reading["max_single_series_harm"],
                "per_series_gain": reading["per_series_gain"]}
        per_position[position] = per_candidate
    return {"menu": ["identity", *menu], "per_position": per_position,
           "note": ("the same 23-default-op + identity public menu Part 1 "
                    "used, re-read at the formation origins (u9/u10/u11); "
                    "unavailable/rejected candidates are listed, not "
                    "dropped.  Never constructed from an LLM winner list.")}


# ---------------------------------------------------------------------------
# D-new-fixed / D-new-safe: recalibrated on the formation table (9/10/11)
# ---------------------------------------------------------------------------

def select_new_fixed_baselines(menu_readings: Mapping[str, Any]
                               ) -> dict[str, Any]:
    menu = list(menu_readings["menu"])
    per_position = menu_readings["per_position"]
    calibrations = []
    for op in menu:
        per_unit = {}
        for position in FORMATION_POSITIONS:
            entry = per_position[position].get(op) or {"status": "UNAVAILABLE"}
            per_unit[position] = entry
        complete = all(per_unit[p]["status"] == "READ"
                      for p in FORMATION_POSITIONS)
        mean_gain = (float(np.mean([per_unit[p]["aggregate_gain"]
                                    for p in FORMATION_POSITIONS]))
                    if complete else None)
        gate_pass = None
        if complete:
            gates = []
            for p in FORMATION_POSITIONS:
                e = per_unit[p]
                treated = (0 if op == "identity"
                          else len(menu_readings["per_position"][p][op][
                              "per_series_gain"]))
                gates.append(runner.authoritative_gate({
                    "treated": treated, "aggregate_gain": e["aggregate_gain"],
                    "harmed_fraction": e["harmed_fraction"],
                    "max_single_series_harm": e["max_single_series_harm"]}))
            gate_pass = all(g["passes"] for g in gates)
        calibrations.append({"op": op, "complete": complete,
                             "mean_calibration_gain": mean_gain,
                             "passes_all_formation_gates": gate_pass})
    complete_ops = [c for c in calibrations if c["complete"] and c["op"] != "identity"]
    d_fixed = (max(complete_ops,
                  key=lambda c: (c["mean_calibration_gain"], -menu.index(c["op"])))
              if complete_ops else None)
    qualified = [c for c in complete_ops if c["passes_all_formation_gates"]]
    d_safe = (max(qualified,
                 key=lambda c: (c["mean_calibration_gain"], -menu.index(c["op"])))
             if qualified else None)
    return {
        "menu": menu, "calibrations": calibrations,
        "d_new_fixed": ({"op": d_fixed["op"],
                        "mean_calibration_gain": d_fixed["mean_calibration_gain"],
                        "status": "SELECTED"} if d_fixed else
                       {"op": "identity", "mean_calibration_gain": 0.0,
                        "status": "NO_COMPLETE_CANDIDATE_FROZEN_IDENTITY"}),
        "d_new_safe": ({"op": d_safe["op"],
                       "mean_calibration_gain": d_safe["mean_calibration_gain"],
                       "status": "SELECTED"} if d_safe else
                      {"op": "identity", "mean_calibration_gain": 0.0,
                       "status": "NO_QUALIFIED_FIXED_PROGRAM"}),
    }


def score_new_fixed_on_review(scorer: Scorer, selection: Mapping[str, Any],
                              ) -> dict[str, Any]:
    period = int(scorer.ctx[REVIEW_POSITIONS[0]].config["period"])
    fixed = {"raw": ()}
    for name, key in (("d_new_fixed", "d_new_fixed"), ("d_new_safe", "d_new_safe")):
        op = selection[key]["op"]
        fixed[name] = (() if op == "identity"
                       else deploy1.candidate_steps(op, period))
    out: dict[str, Any] = {}
    for name, steps in fixed.items():
        per_position = {}
        for position in REVIEW_POSITIONS:
            ctx = scorer.ctx[position]
            origin = int(ctx.origin)
            delayed_origin = ctx.face_origin(ps.DELAYED)
            origin_reading = scorer.population_reading(position, steps, origin)
            delayed_reading = scorer.population_reading(position, steps,
                                                         delayed_origin)
            per_position[position] = {
                "origin_reading": origin_reading,
                "delayed_reading": delayed_reading}
        complete = [p["origin_reading"]["aggregate_gain"]
                   for p in per_position.values() if p["origin_reading"]]
        out[name] = {
            "steps": [{"op": o, "params": p} for o, p in steps],
            "per_position": per_position,
            "equal_weighted_mean_origin_gain": (
                round(float(np.mean(complete)), 6)
                if len(complete) == len(REVIEW_POSITIONS) else None)}
    return out


# ---------------------------------------------------------------------------
# Slow material: same neutral facts for all three arms, effects only for R/C
# ---------------------------------------------------------------------------

DEPLOYMENT_MECHANISM_FACTS = (
    "The Fast you are writing for runs at deployment with no Support "
    "fallback and no admission gate: whatever candidate it selects (or "
    "identity, if it selects nothing or an unresolvable candidate) is what "
    "actually runs, checked only by a deterministic legality/window "
    "verifier before execution -- there is no downstream check that could "
    "reject a bad choice after the fact. A single series' realized loss "
    "comes from the full prepare-train-serve pipeline; a small "
    "serving-window action count does not imply a small training-side "
    "effect, and neither a serving nor a training action count is a causal "
    "explanation of realized utility -- a count is an action, not an "
    "effect.")

WHAT_YOU_MAY_NOT_WRITE = [
    "any surface not in writable_surface_catalog",
    "the string 'Frozen program steps:' anywhere",
    "candidate-supply authority, a serving scope, risk guards, verification "
    "rules, retrieval top_k or candidate slot counts",
    "an applicability condition naming a series, a dataset or a course "
    "position",
    "a concrete threshold, program name or 'winner' copied directly out of "
    "one example case -- state the general principle instead",
]

HOW_TO_FILL_THE_MANIFEST = {
    "one_surface_only": (
        "copy surface_id, operation, surface_precondition and "
        "dependency_precondition_shas verbatim from the one "
        "writable_surface_catalog entry you choose"),
    "ADD": {
        "new_value.skill_kind": "capability",
        "new_value.skill_id": ("a new lowercase id, not one in "
                               "existing_entry_inventory; put the same id "
                               "in target_surface_id"),
        "new_value.body": "the guidance prose",
        "new_value.allowed_tools": "[]",
        "observable_applicability": (
            "the condition, in applicability_grammar's closed leaf form; "
            "the same object in both edit_manifest.observable_applicability "
            "and new_value.observable_applicability"),
    },
    "PATCH": {
        "minimal_patch.value": (
            "the COMPLETE replacement value for that surface -- a PATCH "
            "replaces the whole field, so include everything you want to "
            "keep from current_value.  It must differ from current_value."),
        "observable_applicability": (
            "null for a text surface; for skill_library.entries/<id>."
            "observable_applicability the replacement condition object goes "
            "in minimal_patch.value"),
    },
    "you_may_also_abstain": (
        "return the no_proposal envelope if the evidence on this card "
        "supports no edit, or if the right answer is to keep the current "
        "value unchanged -- that is a legal, complete answer"),
}

_WHAT_YOU_MAY_WRITE = {
    "R": (
        "This is an open reflection task.  You have the complete real "
        "outcome (both the origin-face effect and the +48 persistence "
        "effect) of every decision in this formation batch.  Write ONE "
        "edit, on ONE surface from writable_surface_catalog, that would "
        "help a future frozen Fast -- reading only deployment-visible "
        "information, with no Support fallback -- make more valuable "
        "actual decisions.  You may compare programs, propose a condition, "
        "offer a counter-example to your own idea, or argue that nothing "
        "in the current behaviour should change.  Organise your reasoning "
        "however you find natural; you are not required to structure it "
        "as a success/failure comparison, and a simple, general reflection "
        "is a complete answer if that is what the evidence supports."),
    "C": (
        "You have the complete real outcome (both faces) of every decision "
        "in this formation batch, the same material given to R.  Write ONE "
        "edit, on ONE surface from writable_surface_catalog, organised as "
        "an explicit comparison: (1) name ONE specific observed difference "
        "in this batch that is worth explaining -- a pair or small group of "
        "decisions with different signs or different delivered behaviour; "
        "(2) say what deployment-visible evidence (features, typed "
        "candidates, action facts) could have distinguished them in "
        "advance; (3) the one change; (4) what behaviour and effect you "
        "expect to change, and what result would refute you.  Prefer a "
        "relative comparison where a paired reading exists in the material; "
        "state a plain hypothesis where it does not.  You are not required "
        "to make every edit adversarial or to force the busiest group into "
        "the explanation -- you may still conclude the current behaviour is "
        "fine."),
    "C-minus": (
        "You do NOT have this round's realized gains, losses, risk "
        "outcomes, or which candidate 'won' anywhere in this card -- those "
        "fields are deliberately absent, not zero.  What you do have is "
        "everything else: deployment-visible features and their formulas, "
        "the typed candidates each decision actually had, which one it "
        "chose, what was actually delivered (including legality fallbacks), "
        "and deterministic action-point counts (which describe what a "
        "program changed, not what effect it had).  Write ONE edit, on ONE "
        "surface from writable_surface_catalog, using only this neutral "
        "evidence -- for example, reorganising or clarifying existing "
        "guidance, or adding a condition justified by feature structure "
        "alone.  This tests whether reorganising the neutral evidence by "
        "itself is useful, absent any new outcome information.  Abstaining "
        "is a complete answer if nothing here supports an edit."),
}


def _representative_signature(pooled: Mapping[str, Any]) -> dict[str, Any]:
    row = pooled["rows"][0]
    return dict(row.get("public_features") or {})


#: ``know.action_facts`` repeats the same explanatory prose (baseline,
#: tolerance, "a count is an action not an effect", "whose_property") on
#: every row -- true and useful once, wasteful 120 times.  It is stated once
#: in the card (``HOW_TO_READ_ACTION_FACTS``) and each row keeps only the
#: numbers that actually vary: this is the lossless compression the task
#: book asks for before truncating anything (Sec 3.1), not a content change.
HOW_TO_READ_ACTION_FACTS = (
    "Every case's action_facts.serving_window is THIS series' own window "
    "under the program it actually ran (modified_points / total_points / "
    "modified_fraction); action_facts.training_material is a property of "
    "the SHARED training material for that (unit, face, program) -- every "
    "series a model like it serves shares the same number, so it is not a "
    "property of the evaluated series.  Both baselines are the evaluator's "
    "own linear-integrity baseline and isclose tolerance.  A count is an "
    "action, not an effect: it is not a causal contribution and not a "
    "probability of harm.  'UNKNOWN' means not measured, never zero.  "
    "identity's serving/training counts are 0 by construction (the raw "
    "pipeline ran) and training total_points is UNKNOWN (there is no "
    "program model).")


def _compact_action_facts(facts: Any) -> Any:
    if not isinstance(facts, Mapping):
        return facts
    serving = dict(facts.get("serving_window") or {})
    training = dict(facts.get("training_material") or {})
    return {
        "serving_window": {k: serving.get(k) for k in
                           ("modified_points", "total_points",
                            "modified_fraction") if k in serving},
        "training_material": {k: training.get(k) for k in
                              ("modified_points", "total_points")
                              if k in training},
    }


def _render_case(row: Mapping[str, Any], *, include_effects: bool
                 ) -> dict[str, Any]:
    annotated = deploy1.annotate_recorded_decision(row)
    case = {
        "series_uid": annotated["series_uid"],
        "window_origin": annotated.get("origin", row.get("origin")),
        "public_features_at_this_window": annotated.get(
            "public_features", row.get("public_features")),
        "candidates_proposed": annotated.get(
            "candidate_programs", row.get("candidate_programs")),
        "chosen_candidate_id": annotated.get("chosen_candidate_id"),
        "deploy_status": (annotated.get("original_deploy_status")
                          or annotated.get("deploy_status")),
        "valid_mechanism_decision": annotated.get("valid_mechanism_decision"),
        "original_record_semantics": annotated.get("original_record_semantics"),
        "actual_delivered_program": annotated.get(
            "deployed_program", row.get("deployed_program")),
        "actual_delivered_program_steps": row.get(
            "deployed_program_steps_reconstructed"),
        "action_facts": _compact_action_facts(row.get("action_facts")),
    }
    if include_effects:
        if annotated.get("valid_mechanism_decision"):
            case["deployment_gain_at_origin"] = annotated.get(
                "deployment_gain_at_origin")
            case["fixed_program_gain_at_plus48"] = annotated.get(
                "fixed_program_gain_at_plus48")
        else:
            case["deployment_gain_at_origin"] = "UNKNOWN"
            case["fixed_program_gain_at_plus48"] = "UNKNOWN"
            case["original_recorded_gain_at_origin"] = annotated.get(
                "original_deployment_gain_at_origin",
                row.get("original_recorded_gain_at_origin"))
            case["original_recorded_gain_is_not_valid_utility_evidence"] = True
    return case


def _render_menu(menu_readings: Mapping[str, Any], *, include_effects: bool
                 ) -> dict[str, Any]:
    """Same origin / UID / program readings, not a pair picker.

    When ``include_effects`` is true (R and C, identical material), each
    READ candidate keeps its existing ``per_series_gain`` map so a pair of
    programs with the same aggregate but opposite per-series signs stays
    distinguishable.  Unmeasured candidates stay UNAVAILABLE /
    LEGALITY_REJECTED without imputed zeros.  C-minus
    (``include_effects=False``) sees status and available typed steps:
    no gain, no per-series map, no winner labels.
    """
    out: dict[str, Any] = {}
    for position, per_candidate in menu_readings["per_position"].items():
        row: dict[str, Any] = {}
        for op, entry in per_candidate.items():
            if not include_effects:
                row[op] = {"status": entry["status"]}
                if "program_steps" in entry:
                    row[op]["program_steps"] = entry["program_steps"]
            else:
                row[op] = dict(entry)
        out[str(position)] = row
    return out


def build_card(*, arm: str, pooled: Mapping[str, Any],
              menu_readings: Mapping[str, Any],
              new_fixed_selection: Mapping[str, Any] | None,
              pattern_id: str) -> dict[str, Any]:
    include_effects = arm in ("R", "C")
    card: dict[str, Any] = {
        "pattern_id": pattern_id,
        "observable_signature": _representative_signature(pooled),
        "what_happened": (
            "%d per-sequence decisions (the pooled formation batch: two "
            "independent frozen Fast-only runs over the same 20-series "
            "roster at origins 1656/2136/2376, no Support fallback) are "
            "listed below in formation_cases, one entry per decision.  Each "
            "was decided for one sequence at one window from that "
            "sequence's own visible context; there is no shared cause "
            "implied by listing them together." % pooled["n_rows"]),
        "formation_cases": [
            _render_case(row, include_effects=include_effects)
            for row in pooled["rows"]],
        "how_to_read_action_facts": HOW_TO_READ_ACTION_FACTS,
        "public_menu_reference": _render_menu(menu_readings,
                                              include_effects=include_effects),
        "how_to_read_the_menu_reference": (
            "public_menu_reference lists, per formation origin, every "
            "candidate in the frozen default single-step menu plus "
            "identity, and whether it was READ / UNAVAILABLE (no "
            "deterministic parameter binding) / LEGALITY_REJECTED (failed "
            "the window verifier).  " + (
                "Where READ, aggregate_gain/harmed_fraction/"
                "max_single_series_harm/per_series_gain are this menu "
                "candidate's own reading at that origin, keyed by the "
                "same series UID, independent of what any decision "
                "actually chose.  per_series_gain is the paired "
                "same-position/UID/program map; it is not filled with "
                "zeros for UNAVAILABLE or LEGALITY_REJECTED candidates.  "
                "A true identity zero is the identity candidate's map."
                if include_effects else
                "Only availability and available typed program steps are "
                "shown here, not any gain, "
                "per_series_gain, or winner label: this arm does not see "
                "menu effect numbers.")),
        "observable_feature_vocabulary": list(contract.SCOPE_CLASS["vocabulary"]),
        "what_each_feature_is_measured_over": know.FEATURE_SEMANTICS,
        "how_to_use_the_feature_semantics": (
            "each entry gives the formula and the window it is computed "
            "on.  Three scopes are different objects: the feature window is "
            "the whole observed history; the serving window is its last "
            "192 points and is the only thing the Consumer sees; the "
            "modified region is whatever the program actually changes at "
            "execution time, and no feature here measures it."),
        "applicability_grammar": know.applicability_grammar(),
        "traceability_is_not_a_condition": (
            "series_uid and window_origin let you trace a case back.  They "
            "are NOT part of the observable feature vocabulary and an "
            "applicability condition cannot name an entity, a dataset or a "
            "time position -- only the public features listed above."),
        "deployment_mechanism_facts": DEPLOYMENT_MECHANISM_FACTS,
        "what_you_may_write": _WHAT_YOU_MAY_WRITE[arm],
        "what_you_may_not_write": WHAT_YOU_MAY_NOT_WRITE,
        "how_to_fill_the_manifest": HOW_TO_FILL_THE_MANIFEST,
    }
    if include_effects and new_fixed_selection is not None:
        card["fixed_reference_baselines"] = {
            "d_new_fixed": new_fixed_selection["d_new_fixed"],
            "d_new_safe": new_fixed_selection["d_new_safe"],
            "how_these_were_chosen": (
                "the same public menu, calibrated on the full 20-series "
                "population of this same formation batch, equal-weighted "
                "across the three formation origins, a fixed menu order "
                "breaking ties -- not chosen by any LLM.  d_new_safe "
                "additionally requires "
                "passing the existing four risk lines on every formation "
                "unit; if none qualifies it freezes to identity."),
        }
    return card


# ---------------------------------------------------------------------------
# one Slow session
# ---------------------------------------------------------------------------

def run_slow_branch(*, arm: str, group: str, parent_snapshot: Any,
                    machinery: Mapping[str, Any], pooled: Mapping[str, Any],
                    menu_readings: Mapping[str, Any],
                    new_fixed_selection: Mapping[str, Any], store: Any,
                    controller: Any, slow_factory: Any,
                    budget: seq2.Budget) -> dict[str, Any]:
    materialized_parent = store.materialize(parent_snapshot)
    catalog = know.build_catalog(controller=controller,
                                 parent=materialized_parent)
    out: dict[str, Any] = {
        "arm": arm, "group": group,
        "surfaces_offered": [row["surface_id"] for row in catalog]}
    if not catalog:
        out["outcome"] = "NO_AUTHORISED_SURFACE"
        return out

    card = build_card(arm=arm, pooled=pooled, menu_readings=menu_readings,
                      new_fixed_selection=new_fixed_selection,
                      pattern_id="devdeploy2-%s-%s" % (group, arm.lower()))
    out["card_pattern_id"] = card["pattern_id"]
    out["card_n_cases"] = len(card["formation_cases"])
    out["card_includes_effects"] = arm in ("R", "C")

    llm_before = int(budget.ledgers.llm_total())
    proposal = know.propose_update(slow_factory=slow_factory,
                                   card=card, catalog=catalog,
                                   snapshot=parent_snapshot, attempts=2)
    budget.spend_llm("slow_%s_%s" % (group, arm),
                     int(budget.ledgers.llm_total()) - llm_before)
    out["proposal"] = {k: v for k, v in proposal.items() if k != "manifest"}
    if not proposal.get("proposed"):
        out["outcome"] = {
            "ACCOUNT_OR_PERMISSION_FAULT": "BOUNDARY_ACCOUNT_OR_PERMISSION_FAULT",
            "TRANSPORT_TRANSIENT_FAULT": "BOUNDARY_TRANSPORT_FAULT",
            "SLOW_ABSTAINED": "NO_UPDATE",
        }.get(str(proposal.get("why")), "NO_PROPOSAL")
        return out

    applied = know.apply_update(controller=controller, store=store,
                                snapshot=parent_snapshot,
                                manifest=proposal["manifest"], catalog=catalog)
    out["applied"] = {k: v for k, v in applied.items() if k != "snapshot"}
    if not applied.get("applied"):
        out["outcome"] = "INVALID_EDIT"
        return out

    out["candidate_snapshot"] = applied["snapshot"]
    out["outcome"] = "CANDIDATE_COMPILED"
    written = proposal.get("written_text")
    current = next((row.get("current_value") for row in catalog
                    if row["surface_id"] == proposal["target_surface_id"]),
                   None)
    probe, kind = seq2._edit_probe(operation=proposal.get("operation"),
                                   skill_id=proposal.get("skill_id"),
                                   written=written, current=current)
    out["edit_probe"] = probe
    out["edit_probe_kind"] = kind
    return out


def _make_slow_factory(machinery: Mapping[str, Any], transport: Mapping[str, Any],
                       budget: seq2.Budget) -> Any:
    """A fresh outer core (its own backend + its own 16-call guard) per Slow
    branch -- ``knowledge_boundary``'s own pattern, one level up so each of
    the six branches is independently capped and independently a "new
    session", not three attempts sharing one branch's allowance across a
    whole group."""
    import threading

    outer_guard = seq3.SequenceGuard(ordering_cap=MAX_LLM,
                                     per_unit_arm_cap=SLOW_LLM_CAP,
                                     ledgers=budget.ledgers,
                                     lock=threading.Lock())
    outer_inner = machinery["agentic"]._default_backend_factory(SLOW_LLM_CAP)
    outer_core = machinery["TTHAAgentCore"](
        runner._MeteredOuterBackend(seq3.TransientRelayRetry(outer_inner),
                                    guard=outer_guard, billable=True),
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=transport["requested_model"], base_url=transport["base_url"])

    def factory() -> Any:
        from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent
        return TTHASlowAgent(outer_core)

    return factory


def run_group_slow(*, group: str, parent_snapshot: Any,
                   machinery: Mapping[str, Any], transport: Mapping[str, Any],
                   pooled: Mapping[str, Any], menu_readings: Mapping[str, Any],
                   new_fixed_selection: Mapping[str, Any],
                   run_id: str, budget: seq2.Budget) -> dict[str, Any]:
    store = machinery["SnapshotStore"](
        base.ROOT / ".dev_deploy2_runs" / run_id / group / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    store.materialize(parent_snapshot)

    branches: dict[str, Any] = {}
    for arm in SLOW_ORDER[group]:
        checkpoint = SCRATCH / run_id / group / ("slow_%s.json" % arm)
        if checkpoint.is_file():
            # Recovery never re-asks Slow (AGENTS 5.4(5)): a compiled
            # candidate is recompiled from the store by its own SHA and the
            # SHA is checked, rather than re-serialised as a live object --
            # a HarnessSnapshot does not round-trip through JSON.
            result = json.loads(checkpoint.read_text(encoding="utf-8"))
            sha = (result.get("applied") or {}).get("runtime_bundle_sha")
            if result.get("outcome") == "CANDIDATE_COMPILED" and sha:
                recompiled = machinery["compile_snapshot"](
                    store.root / sha, verify_lock=False)
                if recompiled.runtime_bundle_sha != sha:
                    raise RuntimeError(
                        "recompiled candidate SHA %s != checkpointed %s for "
                        "%s/%s" % (recompiled.runtime_bundle_sha, sha,
                                  group, arm))
                result["candidate_snapshot"] = recompiled
            branches[arm] = result
            continue
        slow_factory = _make_slow_factory(machinery, transport, budget)
        result = run_slow_branch(
            arm=arm, group=group, parent_snapshot=parent_snapshot,
            machinery=machinery, pooled=pooled, menu_readings=menu_readings,
            new_fixed_selection=new_fixed_selection, store=store,
            controller=controller, slow_factory=slow_factory, budget=budget)
        branches[arm] = result
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {k: v for k, v in result.items()
                       if k != "candidate_snapshot"}
        checkpoint.write_text(json.dumps(serialisable, ensure_ascii=False,
                                         indent=1, default=str),
                             encoding="utf-8")
    return {"group": group, "order": list(SLOW_ORDER[group]),
           "branches": branches}


# ---------------------------------------------------------------------------
# exposure check on u12/u13, 0 LLM / 0 fits
# ---------------------------------------------------------------------------

def review_windows(scorer: Scorer, machinery: Mapping[str, Any]
                   ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for position in REVIEW_POSITIONS:
        ctx = scorer.ctx[position]
        origin = int(ctx.origin)
        out[str(position)] = {
            uid: ps.sequence_features(machinery, ctx, uid, origin=origin)
            for uid in scorer.population(position)}
    return out


def check_exposure(branch: Mapping[str, Any], parent_snapshot: Any,
                   windows: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    if branch.get("outcome") != "CANDIDATE_COMPILED":
        return {"skipped": True, "why": branch.get("outcome")}
    return know.exposure_check(
        parent=parent_snapshot, candidate=branch["candidate_snapshot"],
        probe=branch["edit_probe"], windows=windows,
        skill_id=(branch.get("proposal") or {}).get("skill_id") or "")


# ---------------------------------------------------------------------------
# testability decision and the Fast run
# ---------------------------------------------------------------------------

def decide_testability(group: str, slow_result: Mapping[str, Any],
                       exposures: Mapping[str, Any]) -> dict[str, Any]:
    """F is always testable (it is the unmodified parent).  R/C/C-minus are
    testable only if Slow proposed something, it compiled, and it is
    exposed to at least one u12/u13 decision.  If nothing in a group is
    testable, the whole group (F included) is skipped -- running only F
    there would just re-measure the parent the other group's F already
    covers (task book Sec 4.2)."""
    status: dict[str, str] = {"F": "TESTABLE"}
    for arm in ("R", "C", "C-minus"):
        branch = slow_result["branches"].get(arm)
        if branch is None:
            status[arm] = "NOT_ATTEMPTED"
            continue
        outcome = branch.get("outcome")
        if outcome == "NO_PROPOSAL" or outcome == "NO_UPDATE":
            status[arm] = "NO_UPDATE"
        elif outcome == "INVALID_EDIT":
            status[arm] = "INVALID_EDIT"
        elif outcome != "CANDIDATE_COMPILED":
            status[arm] = outcome or "UNKNOWN_OUTCOME"
        else:
            exposure = exposures.get(arm) or {}
            if exposure.get("decisions_the_edit_reaches", 0) > 0:
                status[arm] = "TESTABLE"
            else:
                status[arm] = "COMPILED_BUT_NEVER_EXPOSED"
    any_update_testable = any(status[a] == "TESTABLE"
                              for a in ("R", "C", "C-minus"))
    if not any_update_testable:
        return {"group_status": "NO_TESTABLE_UPDATE", "arms": status}
    return {"group_status": "HAS_TESTABLE_UPDATE", "arms": status}


def run_fastonly_for_arm(*, group: str, arm: str, position: int,
                         snapshot: Any, scorer: Scorer,
                         machinery: Mapping[str, Any], budget: seq2.Budget,
                         run_id: str, concurrency: int,
                         limit: int | None = None) -> dict[str, Any]:
    """A thin call into ``run_dev_deploy1.run_fastonly_unit`` -- it already
    takes an arbitrary ``snapshot`` and an arbitrary ``group`` label for
    budget bookkeeping, so nothing about the per-decision mechanics (fresh
    session, ``experience_episodes=()``, legality-checked deployment, no
    Support search, checkpointed as it completes) needs to be rewritten.
    The two package-level constants it reads off ``run_dev_deploy1``'s own
    module globals are repointed at this package's own caps first --
    exactly how ``run_dev_seq3.build`` repoints ``run_dev_seq2``'s globals
    for its own package."""
    deploy1.MAX_LLM = MAX_LLM
    deploy1.PER_GROUP_LLM_CAP = PER_GROUP_ARM_LLM_CAP
    checkpoint_path = SCRATCH / run_id / group / arm / ("u%03d.json" % position)
    return deploy1.run_fastonly_unit(
        group="%s_%s" % (group, arm), position=position, scorer=scorer,
        snapshot=snapshot, machinery=machinery, budget=budget,
        checkpoint_path=checkpoint_path, concurrency=concurrency, limit=limit)


def run_fastonly_group(*, group: str, arm_snapshots: Mapping[str, Any],
                       testability: Mapping[str, Any], scorer: Scorer,
                       machinery: Mapping[str, Any], budget: seq2.Budget,
                       run_id: str, concurrency: int,
                       limit: int | None = None) -> dict[str, Any]:
    if testability["group_status"] == "NO_TESTABLE_UPDATE":
        return {"group": group, "status": "NO_TESTABLE_UPDATE",
               "arms": testability["arms"], "units": {}}
    units: dict[str, Any] = {}
    for arm in FAST_ARM_ORDER[group]:
        arm_status = testability["arms"].get(arm)
        if arm_status != "TESTABLE":
            units[arm] = {"status": arm_status, "positions": {}}
            continue
        snapshot = arm_snapshots[arm]
        positions_out = {}
        for position in REVIEW_POSITIONS:
            positions_out[position] = run_fastonly_for_arm(
                group=group, arm=arm, position=position, snapshot=snapshot,
                scorer=scorer, machinery=machinery, budget=budget,
                run_id=run_id, concurrency=concurrency, limit=limit)
        units[arm] = {"status": "RAN", "positions": positions_out}
    return {"group": group, "status": "RAN", "arms": testability["arms"],
           "units": units}


# ---------------------------------------------------------------------------
# run-state: the cheap, shared setup every fastonly process reads back
# ---------------------------------------------------------------------------

def _run_state_path(run_id: str) -> Path:
    return SCRATCH / run_id / "run_state.json"


def prepare_run(run_id: str) -> dict[str, Any]:
    """Formation table, menu, D-new-fixed/safe, the six Slow branches and
    their exposure checks -- everything cheap enough (0 LLM except six
    capped Slow calls) to do in one process before the many-hour Fast phase
    fans out one-arm-one-unit-one-process."""
    # Every ``--phase fastonly`` CLI call is a fresh process, so this
    # package's own caps (4000 LLM / 8h) must be reasserted on ``seq2``'s
    # module globals BEFORE the cache check -- ``Budget.room_for`` reads
    # them via normal closure scoping from wherever ``Budget`` is defined
    # (``run_dev_seq2.py``), not from this module.  Setting them only after
    # the cache-hit early return (as this function did until 2026-09-09)
    # left every already-prepared run silently enforcing run_dev_seq2.py's
    # own file defaults (2000 LLM / 6h) instead -- an unannounced smaller
    # ceiling that the backfill for this same run risked hitting mid-batch.
    seq2.MAX_FITS, seq2.MAX_LLM = MAX_FITS, MAX_LLM
    seq2.MAX_WALL_SECONDS = MAX_WALL_SECONDS

    state_path = _run_state_path(run_id)
    if state_path.is_file():
        return json.loads(state_path.read_text(encoding="utf-8"))

    scorer = Scorer()
    parent_snapshot, machinery = deploy1.parent_snapshot_and_machinery()
    saved_required = seq2.REQUIRED_MODEL
    seq2.REQUIRED_MODEL = REQUIRED_MODEL
    try:
        transport = seq3.transport_preflight(machinery)
    finally:
        seq2.REQUIRED_MODEL = saved_required
    if not transport.get("usable"):
        state = {"stage": "DEV_DEPLOY2", "status": "BLOCKED_BY_TRANSPORT",
                 "transport": transport}
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1,
                                         default=str), encoding="utf-8")
        return state

    pooled = load_pooled_formation(scorer)
    menu_readings = menu_readings_at_formation(scorer)
    new_fixed_selection = select_new_fixed_baselines(menu_readings)
    new_fixed_scoring = score_new_fixed_on_review(scorer, new_fixed_selection)

    budget_path = SCRATCH / run_id / "budget.json"
    budget = (seq2.Budget.load(budget_path) if budget_path.is_file()
             else seq2.Budget())
    budget.open_group()

    windows = review_windows(scorer, machinery)
    groups_out: dict[str, Any] = {}
    for group in GROUPS:
        slow_result = run_group_slow(
            group=group, parent_snapshot=parent_snapshot, machinery=machinery,
            transport=transport, pooled=pooled, menu_readings=menu_readings,
            new_fixed_selection=new_fixed_selection, run_id=run_id,
            budget=budget)
        budget.save(budget_path)
        exposures = {arm: check_exposure(branch, parent_snapshot, windows)
                    for arm, branch in slow_result["branches"].items()}
        testability = decide_testability(group, slow_result, exposures)
        arm_shas = {"F": parent_snapshot.runtime_bundle_sha}
        for arm in ("R", "C", "C-minus"):
            branch = slow_result["branches"].get(arm) or {}
            applied = branch.get("applied") or {}
            arm_shas[arm] = applied.get("runtime_bundle_sha")
        groups_out[group] = {
            "slow": {a: {k: v for k, v in b.items()
                        if k != "candidate_snapshot"}
                    for a, b in slow_result["branches"].items()},
            "slow_order": slow_result["order"],
            "exposures": exposures, "testability": testability,
            "arm_knowledge_sha": arm_shas,
        }

    state = {
        "stage": "DEV_DEPLOY2", "status": "PREPARED", "run_id": run_id,
        "parent_knowledge_sha": parent_snapshot.runtime_bundle_sha,
        "transport": transport,
        "pooled_formation": pooled, "menu_readings": menu_readings,
        "new_fixed_selection": new_fixed_selection,
        "new_fixed_scoring_on_review": new_fixed_scoring,
        "groups": groups_out,
        "cost_after_prepare": budget.to_dict(),
        "fits_from_scoring_after_prepare": scorer.physical_fits,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1,
                                     default=str), encoding="utf-8")
    return state


def _snapshot_for_arm(machinery: Mapping[str, Any], run_id: str, group: str,
                      arm: str, sha: str) -> Any:
    if arm == "F":
        doc = base.load(R.ORDERING)
        state = live._state_at_k1(doc)
        snapshot_dir, _source = R._resolve_k0_snapshot(doc, state["k0"])
        snapshot = machinery["compile_snapshot"](snapshot_dir, verify_lock=False)
        if snapshot.runtime_bundle_sha != sha:
            raise RuntimeError("F snapshot SHA drifted: %s != %s"
                              % (snapshot.runtime_bundle_sha, sha))
        return snapshot
    store = machinery["SnapshotStore"](
        base.ROOT / ".dev_deploy2_runs" / run_id / group / "store")
    snapshot = machinery["compile_snapshot"](store.root / sha, verify_lock=False)
    if snapshot.runtime_bundle_sha != sha:
        raise RuntimeError("%s/%s snapshot SHA drifted: %s != %s"
                          % (group, arm, snapshot.runtime_bundle_sha, sha))
    return snapshot


# ---------------------------------------------------------------------------
# aggregation: the comparator table, 0 new LLM / fits -- rereads outputs only
# ---------------------------------------------------------------------------

def _arm_review_mean(group: str, arm: str, run_id: str
                     ) -> dict[str, Any] | None:
    per_position = {}
    for position in REVIEW_POSITIONS:
        path = ART / ("dev_deploy2_fastonly__%s_%s_u%s.json"
                      % (group, arm, position))
        if not path.is_file():
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = [deploy1.annotate_recorded_decision(r)
                for r in doc.get("sequences", [])]
        stats = deploy1.mechanism_unit_stats(
            rows, population_n=len(doc.get("decision_population") or rows))
        # Old summaries include invalid empty-selection zeros.  Re-read rows
        # in memory; never rewrite the source artifact or trust its old mean.
        per_position[position] = {
            **doc, "sequences": rows,
            "deployment_gain_at_origin_mean": stats["origin_mean"],
            "fixed_program_gain_at_plus48_mean": stats["delayed_mean"],
            "mechanism_reading": stats,
        }
    origin = [d["deployment_gain_at_origin_mean"] for d in per_position.values()]
    if any(v is None for v in origin):
        return {"complete": False, "per_position": per_position}
    delayed_vals = [d["fixed_program_gain_at_plus48_mean"]
                    for d in per_position.values()]
    return {
        "complete": True,
        "equal_weighted_origin_mean": round(float(np.mean(origin)), 6),
        "equal_weighted_delayed_mean": (
            round(float(np.mean(delayed_vals)), 6)
            if all(v is not None for v in delayed_vals) else None),
        "per_position": per_position,
    }


def _paired_diff(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None,
                 key: str) -> Any:
    if not a or not b or not a.get("complete") or not b.get("complete"):
        return "UNKNOWN"
    if not isinstance(a.get(key), (int, float)) or not isinstance(b.get(key), (int, float)):
        return "UNKNOWN"
    return round(a[key] - b[key], 6)


def aggregate_run(run_id: str) -> dict[str, Any]:
    state = _run_state_path(run_id)
    if not state.is_file():
        return {"status": "NOT_PREPARED"}
    doc = json.loads(state.read_text(encoding="utf-8"))
    groups_out: dict[str, Any] = {}
    for group, blob in (doc.get("groups") or {}).items():
        testability = blob["testability"]
        arm_means: dict[str, Any] = {}
        for arm, arm_status in testability["arms"].items():
            if arm_status == "TESTABLE" or arm == "F":
                if (testability["group_status"] == "NO_TESTABLE_UPDATE"
                        or (arm != "F" and arm_status != "TESTABLE")):
                    arm_means[arm] = None
                    continue
                arm_means[arm] = _arm_review_mean(group, arm, run_id)
            else:
                arm_means[arm] = None
        # Named "diff_X_vs_Y" rather than "X_minus_Y": this package also has
        # a literal arm called "C-minus", and "C_minus_F" read as either
        # "C minus F" or "the C-minus arm vs F" -- exactly the ambiguity a
        # comparator table must not have.
        comparisons = {
            "diff_C_vs_F": _paired_diff(arm_means.get("C"), arm_means.get("F"),
                                        "equal_weighted_origin_mean"),
            "diff_R_vs_F": _paired_diff(arm_means.get("R"), arm_means.get("F"),
                                        "equal_weighted_origin_mean"),
            "diff_C_vs_R": _paired_diff(arm_means.get("C"), arm_means.get("R"),
                                        "equal_weighted_origin_mean"),
            "diff_C_vs_Cminus": _paired_diff(arm_means.get("C"),
                                             arm_means.get("C-minus"),
                                             "equal_weighted_origin_mean"),
        }
        groups_out[group] = {
            "testability": testability, "arm_means": arm_means,
            "comparisons_origin_face": comparisons,
        }
    return {
        "stage": "DEV_DEPLOY2_AGGREGATE", "run_id": run_id,
        "parent_knowledge_sha": doc.get("parent_knowledge_sha"),
        "pooled_formation_summary": {
            "n_rows": doc["pooled_formation"]["n_rows"],
            "n_unreconstructed": doc["pooled_formation"]["n_unreconstructed"]},
        "new_fixed_selection": doc.get("new_fixed_selection"),
        "new_fixed_scoring_on_review": doc.get("new_fixed_scoring_on_review"),
        "groups": groups_out,
        "cost_after_prepare": doc.get("cost_after_prepare"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fastonly", "aggregate"),
                        required=True)
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--arm", choices=("F", "R", "C", "C-minus"))
    parser.add_argument("--position", type=int, choices=list(REVIEW_POSITIONS))
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-loopback-relay", action="store_true",
                        help="opt-in: permit http:// to 127.0.0.1/localhost "
                             "for M0_AGENT_BASE_URL (runtime patch only, "
                             "never touches agent_backend.py on disk)")
    args = parser.parse_args(argv)

    if args.allow_loopback_relay:
        deploy1.allow_loopback_http_relay()

    if args.phase == "prepare":
        state = prepare_run(args.run_id)
        print("status", state.get("status"))
        for group, blob in (state.get("groups") or {}).items():
            print(" ", group, blob["testability"])
        return 0

    if args.phase == "fastonly":
        if not (args.group and args.arm and args.position):
            parser.error("--group --arm --position are required for fastonly")
        state = prepare_run(args.run_id)
        if state.get("status") != "PREPARED":
            print("not prepared:", state.get("status"))
            return 1
        group_state = state["groups"][args.group]
        arm_status = group_state["testability"]["arms"].get(args.arm)
        if group_state["testability"]["group_status"] == "NO_TESTABLE_UPDATE":
            print("group %s: NO_TESTABLE_UPDATE, not running" % args.group)
            return 0
        if arm_status != "TESTABLE":
            print("%s/%s not testable: %s" % (args.group, args.arm, arm_status))
            return 0
        _machinery_only = v1runner._machinery()
        sha = group_state["arm_knowledge_sha"][args.arm]
        snapshot = _snapshot_for_arm(_machinery_only, args.run_id, args.group,
                                     args.arm, sha)
        scorer = Scorer()
        budget_path = SCRATCH / args.run_id / "budget.json"
        budget = (seq2.Budget.load(budget_path) if budget_path.is_file()
                 else seq2.Budget())
        budget.open_group()
        result = run_fastonly_for_arm(
            group=args.group, arm=args.arm, position=args.position,
            snapshot=snapshot, scorer=scorer, machinery=_machinery_only,
            budget=budget, run_id=args.run_id, concurrency=args.concurrency,
            limit=args.limit)
        budget.save(budget_path)
        out_path = ART / ("dev_deploy2_fastonly__%s_%s_u%s.json"
                          % (args.group, args.arm, args.position))
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                       default=str), encoding="utf-8")
        print("wrote", out_path, "| n_scored", result.get("n_scored"),
             "/", result.get("n_total"))
        stopped = result.get("stopped_at")
        if stopped:
            # A fatal, cross-series condition (account/permission fault or
            # package ceiling) happened inside this unit.  The output file
            # above already says so honestly (n_scored short of n_total,
            # stopped_at populated) -- what was missing before this fix is
            # telling the ORCHESTRATING shell loop to stop dispatching
            # further (group, arm, position) jobs instead of ploughing
            # through all of them at zero balance.  Exit code 2 is this
            # module's own signal, not a project-wide convention.
            print("FATAL: unit stopped early:", stopped)
            return 2
        return 0

    if args.phase == "aggregate":
        result = aggregate_run(args.run_id)
        out_path = ART / "dev_deploy2_feedback_to_frozen_fast.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                       default=str), encoding="utf-8")
        print("wrote", out_path)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
