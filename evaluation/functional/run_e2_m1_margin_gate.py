"""M-1 -- causal half-slice protocol for confirmation-surface margin gating.

G3 (commit 19c6b22) froze FIELD1_NO_CONVERSION on GPMvF: A5-scoped inject
3/4 but material-positive 0/4 under the quarter protocol (ps0b margin
1.35x).  GPOvY at 4.15x converted 4/4.  The arbitration hypothesis is
that conversion is gated by confirmation-surface margin.

This book is the unique-variable test.  W-1 / G3 wiring is untouched
(methods / contracts / runtime / operators stay as they are).  The only
protocol change is held-in slice allocation: quarter four-slice
(r1s/r1d/r2s/r2d) becomes a dual-gate *half* -- one Support surface and
one delayed surface, each the role-concat of the two quarter slices,
run as a single held-in round.

Implementation choice (eval-layer parameterization of the existing cell
surfaces, not a methods edit):

* build the canonical quarter cell with ``s1._build_cell`` (same as G3)
* repack ``r1_support = concat(r1_support, r2_support)``  (n=21)
* repack ``r1_delayed = concat(r1_delayed, r2_delayed)``  (n=19)
* run ``rounds=("r1",)`` only

This is *not* the ps0b stored ``half_slices`` (same-round support+delayed
concat), which would collapse the dual gate.

Readings are margin-mechanism evidence.  They are not a capability
comparison against the G3 quarter baseline.

  python evaluation/functional/run_e2_m1_margin_gate.py --arith-only
  python evaluation/functional/run_e2_m1_margin_gate.py --probe-only
  python evaluation/functional/run_e2_m1_margin_gate.py --run
  python evaluation/functional/run_e2_m1_margin_gate.py --resume
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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_g3_three_field_course as g3  # noqa: E402
import run_e2_ps0c_ps1 as ps0c  # noqa: E402
import run_e2_ps2_mechanical_supply as ps2  # noqa: E402
import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
PS0B_JSON = E2 / "ps0b_confirmation_surface_audit.json"
G3_JSON = E2 / "g3_three_field_course.json"
OUT_JSON = E2 / "m1_margin_gate.json"
OUT_MD = E2 / "m1_margin_gate.md"
CHECKPOINT = E2 / "m1_margin_gate.checkpoint.json"

PROTOCOL_VERSION = "m1_margin_gate_v1"
EVIDENCE_GRADE = "development-mechanism (pilot)"

EXAM_UNIT = {
    "unit_id": "GunPointMaleVersusFemale__impulse_v2",
    "dataset": "GunPointMaleVersusFemale",
    "injection": "impulse_v2",
    "series_length": 150,
}
G3_CONTROL_COMMIT = "19c6b22"
UNIT_ID = EXAM_UNIT["unit_id"]
PROGRAM = "hampel_filter"
MARGIN_BAR = 2.0

ARM_A3 = ps2.ARM_A3
ARM_SCOPED = ps2.ARM_SCOPED
SCOPED_SKILL_ID = ps2.SCOPED_SKILL_ID
TARGET_OPERATOR = ps2.TARGET_OPERATOR
REPLICATES = 4
LLM_PER_RUN = g3.LLM_PER_RUN          # 12 -- same as G3 / W-1
FIT_PER_RUN = g3.FIT_PER_RUN          # 10
LLM_TOTAL_CAP = 100
FIT_TOTAL_CAP = 100
WALL_SECONDS_CAP = int(2 * 60 * 60)
HALF_ROUNDS = ("r1",)
QUARTER_ROUNDS = s1.HELD_IN_ROUNDS

RUN_PLAN = tuple(
    [{"run_id": "m1_a5_%d" % (index + 1), "arm": ARM_SCOPED,
      "replicate": index + 1} for index in range(REPLICATES)]
    + [{"run_id": "m1_a3_%d" % (index + 1), "arm": ARM_A3,
        "replicate": index + 1} for index in range(REPLICATES)]
)


class Stop(Exception):
    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__("%s: %s" % (verdict, reason))
        self.verdict = verdict
        self.reason = reason


def _same_rights() -> dict[str, Any]:
    """Everything except slice allocation is pinned to W-1 / G3."""
    return {
        "card": "artifacts/functional/e2/ps2_cards/ps2_card_scoped.json",
        "skill_id": SCOPED_SKILL_ID,
        "operator": TARGET_OPERATOR,
        "unit_id": UNIT_ID,
        "consumer": "ClassificationConsumerAdapter / ridge",
        "maximum_candidates": 3,
        "llm_per_run": LLM_PER_RUN,
        "fit_per_run": FIT_PER_RUN,
        "g3_llm_per_run": g3.LLM_PER_RUN,
        "g3_fit_per_run": g3.FIT_PER_RUN,
        "grants_execution": False,
        "requires_target_support": True,
        "methods_contracts_runtime_operators_unmodified": True,
        "unique_variable": (
            "held-in feedback slice allocation: quarter four-slice "
            "-> role-concat half, one Support + one delayed, one round"
        ),
        "assertions": {
            "per_run_caps_match_g3": (
                LLM_PER_RUN == g3.LLM_PER_RUN
                and FIT_PER_RUN == g3.FIT_PER_RUN),
            "skill_and_operator_match_g3": (
                SCOPED_SKILL_ID == g3.SCOPED_SKILL_ID
                and TARGET_OPERATOR == g3.TARGET_OPERATOR),
            "unit_is_g3_field1": UNIT_ID == (
                "GunPointMaleVersusFemale__impulse_v2"),
        },
    }


# =========================================================================== #
# Arithmetic precondition (0 fit): sealed ps0b counts, role-concat half
# =========================================================================== #
def _correct_count(n: int, accuracy: float) -> int:
    return int(round(float(n) * float(accuracy)))


def _slice_from_counts(*, n: int, identity_correct: int,
                       program_correct: int, composed_of: Sequence[str]
                       ) -> dict[str, Any]:
    identity_acc = identity_correct / n
    program_acc = program_correct / n
    reading = program_acc - identity_acc
    material = 1.0 / n
    margin = (reading / material) if material else 0.0
    return {
        "n": n,
        "identity_correct": identity_correct,
        "program_correct": program_correct,
        "identity_accuracy": identity_acc,
        "program_accuracy": program_acc,
        "reading": reading,
        "material_line": material,
        "margin_multiplier": margin,
        "meets_material": reading >= material,
        "meets_2x": margin + 1e-12 >= MARGIN_BAR,
        "composed_of": list(composed_of),
    }


def _ps0b_gpmvf_hampel() -> dict[str, Any]:
    payload = json.loads(PS0B_JSON.read_text(encoding="utf-8"))
    for row in payload.get("pairs") or []:
        if (str(row.get("unit_id")) == UNIT_ID
                and str(row.get("program")) == PROGRAM):
            return row
    raise Stop("INSTRUMENT_UNREADABLE",
               "ps0b has no sealed row for %s x %s" % (UNIT_ID, PROGRAM))


def arithmetic_precondition() -> dict[str, Any]:
    """0-fit.  Role-concat half from sealed per-slice accuracies.

    ps0b ``half_slices`` concat *same-round* support+delayed and would
    destroy the dual gate; those numbers are recorded as a rejected
    composition, not used as the live protocol.
    """
    sealed = _ps0b_gpmvf_hampel()
    slices = sealed["slices"]
    quarter: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for name, part in slices.items():
        n = int(part["n"])
        identity_n = _correct_count(n, part["identity_accuracy"])
        program_n = _correct_count(n, part["program_accuracy"])
        counts[name] = {"n": n, "identity_correct": identity_n,
                        "program_correct": program_n}
        reconstructed = program_n / n - identity_n / n
        if abs(reconstructed - float(part["reading"])) > 1e-9:
            raise Stop("INSTRUMENT_UNREADABLE",
                       "sealed %s reading %.12f != reconstructed %.12f"
                       % (name, part["reading"], reconstructed))
        quarter[name] = {
            "n": n,
            "identity_correct": identity_n,
            "program_correct": program_n,
            "identity_accuracy": part["identity_accuracy"],
            "program_accuracy": part["program_accuracy"],
            "reading": part["reading"],
            "material_line": part["material_line"],
            "meets_material": part["meets_material"],
        }
    support = _slice_from_counts(
        n=counts["r1_support"]["n"] + counts["r2_support"]["n"],
        identity_correct=(counts["r1_support"]["identity_correct"]
                          + counts["r2_support"]["identity_correct"]),
        program_correct=(counts["r1_support"]["program_correct"]
                         + counts["r2_support"]["program_correct"]),
        composed_of=("r1_support", "r2_support"))
    delayed = _slice_from_counts(
        n=counts["r1_delayed"]["n"] + counts["r2_delayed"]["n"],
        identity_correct=(counts["r1_delayed"]["identity_correct"]
                          + counts["r2_delayed"]["identity_correct"]),
        program_correct=(counts["r1_delayed"]["program_correct"]
                         + counts["r2_delayed"]["program_correct"]),
        composed_of=("r1_delayed", "r2_delayed"))
    rejected_same_round = sealed.get("half_slices") or {}
    expected_n = {
        "support": 21, "delayed": 19,
        "quarter": {"r1_support": 11, "r1_delayed": 10,
                    "r2_support": 10, "r2_delayed": 9},
    }
    if support["n"] != expected_n["support"] or delayed["n"] != expected_n["delayed"]:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "role-concat n %d/%d != sealed 21/19"
                   % (support["n"], delayed["n"]))
    both_ge_2x = bool(support["meets_2x"] and delayed["meets_2x"])
    next_unit = {
        "if_precondition_failed": (
            "local ROBUST pool above 2x is empty after GPMvF; do not "
            "spend LLM on another weak-margin GunPointFamily unit "
            "(GunPoint__impulse_v2 is 1.40x).  The remaining causal "
            "bracket is GPOvY down-shift (reverse: shrink a converting "
            "4.15x surface), not a second half-protocol on a sibling."
        ),
    }
    return {
        "source": PS0B_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "unit_id": UNIT_ID,
        "program": PROGRAM,
        "fit_spent": 0,
        "implementation_choice": {
            "kind": "one_round_role_concat",
            "rounds": list(HALF_ROUNDS),
            "support_slice": "concat(r1_support, r2_support)",
            "delayed_slice": "concat(r1_delayed, r2_delayed)",
            "support_n": support["n"],
            "delayed_n": delayed["n"],
            "dual_gate_preserved": True,
            "rejected_composition": (
                "ps0b.half_slices same-round support+delayed concat "
                "(collapses Support and delayed into one surface)"
            ),
            "cell_mechanism": (
                "eval-layer repack of s1._build_cell surfaces; "
                "s1a._r3_build_cell / methods untouched"
            ),
        },
        "quarter_sealed": {
            "slices": quarter,
            "margin_multiplier": sealed.get("margin_multiplier"),
            "reproducibility_margin_ge_2x": sealed.get(
                "reproducibility_margin_ge_2x"),
            "g3_control": "A5-scoped material-positive 0/4 (not re-run)",
        },
        "half_role_concat": {
            "support": support,
            "delayed": delayed,
            "min_margin_multiplier": min(support["margin_multiplier"],
                                         delayed["margin_multiplier"]),
            "both_meet_2x": both_ge_2x,
        },
        "ps0b_same_round_half_rejected_for_live": rejected_same_round,
        "expected_n": expected_n,
        "precondition_passed": both_ge_2x,
        "stop_if_failed": "MARGIN_PRECONDITION_FAILED",
        "next_unit_if_failed": next_unit,
    }


# =========================================================================== #
# Eval-layer cell repack (unique variable)
# =========================================================================== #
def _concat_surface(left: Sequence[Any], right: Sequence[Any]
                    ) -> tuple[Any, Any]:
    values = np.concatenate(
        [np.asarray(left[0], dtype=np.float64),
         np.asarray(right[0], dtype=np.float64)], axis=0)
    labels = np.concatenate(
        [np.asarray(left[1]), np.asarray(right[1])], axis=0)
    return (values, labels)


def _half_cell(quarter: Mapping[str, Any]) -> dict[str, Any]:
    surfaces = dict(quarter["surfaces"])
    support = _concat_surface(surfaces["r1_support"], surfaces["r2_support"])
    delayed = _concat_surface(surfaces["r1_delayed"], surfaces["r2_delayed"])
    half = dict(quarter)
    half["surfaces"] = {"r1_support": support, "r1_delayed": delayed}
    half["slice_rows"] = {
        "r1_support": int(support[0].shape[0]),
        "r1_delayed": int(delayed[0].shape[0]),
    }
    half["quarter_slice_rows"] = dict(quarter.get("slice_rows") or {})
    half["m1_protocol"] = "half_role_concat_one_round"
    if half["slice_rows"]["r1_support"] != 21:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "live support n=%s, sealed role-concat is 21"
                   % half["slice_rows"]["r1_support"])
    if half["slice_rows"]["r1_delayed"] != 19:
        raise Stop("INSTRUMENT_UNREADABLE",
                   "live delayed n=%s, sealed role-concat is 19"
                   % half["slice_rows"]["r1_delayed"])
    return half


# =========================================================================== #
# Scoring
# =========================================================================== #
def _g3_quarter_control() -> dict[str, Any]:
    if not G3_JSON.is_file():
        return {"available": False, "reason": "g3 artifact missing"}
    payload = json.loads(G3_JSON.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("runs") or []
            if row.get("unit_id") == UNIT_ID]
    summaries = payload.get("field_summaries") or {}
    verdict = payload.get("verdict") or {}
    return {
        "available": True,
        "commit": G3_CONTROL_COMMIT,
        "artifact": G3_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "verdict": verdict.get("verdict"),
        "note": (
            "quarter baseline, mechanism contrast only; not a capability "
            "ranking against the half protocol"
        ),
        "a5_scoped": summaries.get("field1 / A5-scoped"),
        "a3": summaries.get("field1 / A3"),
        "runs": [
            {"run_id": row["run_id"], "arm": row["arm"],
             "entered_pool": row["inject_funnel"]["entered_pool"],
             "support_material_positive": row["inject_funnel"][
                 "support_material_positive"],
             "delayed_approved": row["inject_funnel"]["delayed_approved"],
             "deployed": row["inject_funnel"]["deployed"],
             "applied": ",".join(
                 step["op"] for step
                 in (row.get("deployment") or {}).get("applied_program") or []
             ) or "identity",
             "heldout_accuracy_gain": (row.get("deployment") or {}).get(
                 "heldout_accuracy_gain"),
             "llm_calls": row.get("llm_calls"),
             "consumer_fits": row.get("consumer_fits")}
            for row in rows
        ],
    }


def _supply_converted(inject: Mapping[str, Any]) -> bool:
    """Supply candidate through both gates, not an agent-authored hampel."""
    return bool(
        inject.get("entered_pool")
        and inject.get("support_material_positive")
        and inject.get("delayed_approved")
        and inject.get("deployed"))


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
    applied = [str(step.get("op"))
               for step in deployment.get("applied_program") or []]
    first_skill_round = next(
        (str(record["round"]) for record in public.get("rounds") or []
         if record.get("winner_delayed_approved")), None)
    return {
        "run_id": plan["run_id"],
        "arm": arm,
        "unit_id": UNIT_ID,
        "replicate": plan["replicate"],
        "protocol": "half_role_concat_one_round",
        "rounds_run": list(HALF_ROUNDS),
        "base_runtime_bundle_sha": base_sha,
        "card_in_fast_view": card_seen,
        "inject_funnel": inject,
        "supply_dual_gate_converted": _supply_converted(inject),
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
            "applied_ops": applied,
            "heldout_accuracy_gain": deployment.get("heldout_accuracy_gain"),
            "worst_class_delta": worst,
            "harm_event": bool(worst < -cls.HARM_BAR),
        },
        "rounds": public.get("rounds"),
    }


def _arm_rows(runs: Sequence[Mapping[str, Any]], arm: str
              ) -> list[Mapping[str, Any]]:
    return [row for row in runs if row["arm"] == arm]


def _arm_summary(runs: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    rows = _arm_rows(runs, arm)
    gains = [float(row["deployment"]["heldout_accuracy_gain"] or 0.0)
             for row in rows]
    return {
        "arm": arm,
        "runs": len(rows),
        "entered_pool": sum(1 for row in rows
                            if row["inject_funnel"]["entered_pool"]),
        "passed_verifier": sum(
            1 for row in rows if row["inject_funnel"].get("passed_verifier")),
        "support_material_positive": sum(
            1 for row in rows
            if row["inject_funnel"]["support_material_positive"]),
        "delayed_approved": sum(
            1 for row in rows if row["inject_funnel"]["delayed_approved"]),
        "supply_dual_gate_converted": sum(
            1 for row in rows if row.get("supply_dual_gate_converted")),
        "deployed_target_operator": sum(
            1 for row in rows
            if TARGET_OPERATOR in (row["deployment"].get("applied_ops") or [])),
        "deployed_identity": sum(
            1 for row in rows if not row["deployment"]["applied_program"]),
        "harm_runs": sum(1 for row in rows
                         if row["deployment"]["harm_event"]),
        "heldout_gains": gains,
        "mean_heldout_gain": (sum(gains) / len(gains)) if gains else 0.0,
        "positive_heldout_deploys": sum(
            1 for row in rows
            if TARGET_OPERATOR in (row["deployment"].get("applied_ops") or [])
            and float(row["deployment"]["heldout_accuracy_gain"] or 0.0) > 0),
        "card_in_fast_view_runs": sum(1 for row in rows
                                      if row["card_in_fast_view"]),
        "llm": sum(int(row["llm_calls"] or 0) for row in rows),
        "consumer_fits": sum(int(row["consumer_fits"] or 0) for row in rows),
        "probes": sum(int(row["probes"] or 0) for row in rows),
        "agent_families": sorted({family for row in rows
                                  for family in row["agent_proposal_families"]}),
        "candidate_probed": sum(
            1 for row in rows
            if row["inject_funnel"].get("entered_pool")
            and (row["inject_funnel"].get("passed_verifier")
                 or row["inject_funnel"].get("support_material_positive")
                 or (row.get("probes") or 0) > 0)),
    }


def _gpovy_downshift_addendum() -> dict[str, Any]:
    """Design only.  Reverse bracket if this book lands AMBIGUOUS."""
    return {
        "status": "design_only_not_run",
        "purpose": (
            "close the causal bracket from the other side: a unit that "
            "already converts on a high-margin quarter surface should "
            "lose conversion when the confirmation surface is down-shifted"
        ),
        "unit_id": "GunPointOldVersusYoung__impulse_v2",
        "fixed": (
            "same W-1 dual-source hampel card, Scope, operator, Consumer, "
            "maximum_candidates=3, per-run LLM/fit caps"
        ),
        "unique_variable": (
            "held-in slice allocation only: GPOvY quarter (already 4/4 at "
            "4.15x, do not re-run as a capability race) vs a *finer* "
            "protocol that shrinks n until the coarsest dual-gate slice "
            "margin approaches the GPMvF quarter 1.35x band"
        ),
        "suggested_finer_protocol": (
            "single-round dual gate using only r1_support (n=11) + "
            "r1_delayed (n=10) is still 7.0x / 3.0x -- too coarse to "
            "reject.  Need an eighth-style split of the existing quarter "
            "parts (or a 1-vs-rest subsample of each role) until delayed "
            "margin is in [1.2x, 1.6x].  Stop at the first composition "
            "that is arithmetically below 2x on either gate -- that is "
            "the reject-side twin of this book's half-upshift."
        ),
        "arms": "A5-scoped x4 + A3 x4, fresh state, checkpoint+resume",
        "do_not_run_unless_extended": True,
        "label_discipline": (
            "down-shift readings are margin-mechanism evidence only; "
            "not a capability comparison against W-1 / G3"
        ),
    }


def _verdict(runs: Sequence[Mapping[str, Any]], arith: Mapping[str, Any],
             *, stopped: str | None) -> dict[str, Any]:
    if stopped == "MARGIN_PRECONDITION_FAILED":
        return {
            "verdict": "MARGIN_PRECONDITION_FAILED",
            "reason": (
                "role-concat half still below the 2x material line; "
                "half allocation cannot rescue this unit.  "
                + str((arith.get("next_unit_if_failed") or {}).get(
                    "if_precondition_failed") or "")
            ),
            "facts": {"arithmetic": arith.get("half_role_concat")},
        }
    if stopped == "ARITHMETIC_PRECONDITION_PASSED":
        half = arith.get("half_role_concat") or {}
        return {
            "verdict": "ARITHMETIC_PRECONDITION_PASSED",
            "reason": (
                "role-concat half Support %.2fx / delayed %.2fx, both "
                ">= 2x.  Live 8-run protocol is eligible; this flag "
                "spent 0 fit / 0 LLM."
                % ((half.get("support") or {}).get("margin_multiplier") or 0,
                   (half.get("delayed") or {}).get("margin_multiplier") or 0)
            ),
            "facts": {"arithmetic": half},
        }
    if stopped in ("COMPUTE_BUDGET_EXCEEDED", "BACKEND_UNAVAILABLE",
                   "INSTRUMENT_UNREADABLE"):
        return {"verdict": stopped,
                "reason": "stopped before the frozen 8-run table",
                "facts": {"completed": [row["run_id"] for row in runs]}}
    a5 = _arm_summary(runs, ARM_SCOPED)
    a3 = _arm_summary(runs, ARM_A3)
    facts = {"a5_scoped": a5, "a3": a3,
             "half_min_margin": (arith.get("half_role_concat") or {}).get(
                 "min_margin_multiplier")}
    converted = int(a5["supply_dual_gate_converted"])
    material = int(a5["support_material_positive"])
    probed = int(a5["candidate_probed"]) or int(a5["entered_pool"])
    heldout_ok = bool(a5["runs"] == REPLICATES and (
        a5["positive_heldout_deploys"] >= 2
        or (converted >= 2 and a5["mean_heldout_gain"] > 0)))
    harm0 = a5["harm_runs"] == 0 and a3["harm_runs"] == 0
    arith_ok = bool((arith.get("half_role_concat") or {}).get("both_meet_2x"))
    if a5["runs"] < REPLICATES:
        return {"verdict": stopped or "COMPUTE_BUDGET_EXCEEDED",
                "reason": "A5-scoped table incomplete (%d/4)" % a5["runs"],
                "facts": facts}
    if converted >= 2 and heldout_ok and harm0:
        return {
            "verdict": "MARGIN_GATING_CONFIRMED",
            "reason": (
                "A5-scoped supply candidate dual-gate converted %d/4; "
                "deployed held-out positive; harm 0.  Confirmation-surface "
                "margin gating stands; margin layering belongs in Gate 4."
                % converted),
            "facts": facts,
        }
    if converted == 1:
        return {
            "verdict": "AMBIGUOUS",
            "reason": (
                "A5-scoped supply dual-gate converted 1/4.  Stop.  "
                "GPOvY down-shift addendum attached for arbitration."
            ),
            "facts": facts,
            "gpovy_downshift_addendum": _gpovy_downshift_addendum(),
        }
    if arith_ok and probed and material == 0:
        return {
            "verdict": "MARGIN_GATING_REJECTED",
            "reason": (
                "arithmetic half margin >= 2x and the supplied candidate "
                "was probed, but material-positive Support is still 0/4.  "
                "The gating hypothesis is falsified.  Stop on the stop-loss "
                "line; do not try another allocation on this unit."
            ),
            "facts": facts,
        }
    if converted == 0 and material >= 1:
        return {
            "verdict": "AMBIGUOUS",
            "reason": (
                "Support opened on %d/4 but dual-gate conversion is 0/4.  "
                "Not a clean reject (material-positive is not 0/4) and not "
                "a confirm.  Stop.  GPOvY down-shift addendum attached."
                % material),
            "facts": facts,
            "gpovy_downshift_addendum": _gpovy_downshift_addendum(),
        }
    return {
        "verdict": stopped or "AMBIGUOUS",
        "reason": (
            "table completed without matching a frozen cell "
            "(converted=%d material+=%d probed=%d harm=%d)."
            % (converted, material, probed, a5["harm_runs"])),
        "facts": facts,
        "gpovy_downshift_addendum": _gpovy_downshift_addendum(),
    }


def _outside_book(runs: Sequence[Mapping[str, Any]],
                  a5: Mapping[str, Any],
                  a3: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    miss = [row["run_id"] for row in runs
            if row["arm"] == ARM_SCOPED
            and row.get("card_in_fast_view")
            and not row["inject_funnel"].get("entered_pool")]
    if miss:
        notes.append(
            "A5 inject=False on %s with the card still in Fast view "
            "(same prepare/identity-only miss seen in PS-2 / G3).  Those "
            "runs still deployed hampel via an agent-authored program; "
            "they are not supply conversions."
            % ",".join(miss))
    notes.append(
        "A5 funnel delayed_approved is 4/4 because _inject_funnel "
        "credits any delayed-approved hampel winner, including the "
        "agent path on inject=False runs.  The causal cell is "
        "supply_dual_gate_converted %d/4."
        % int(a5.get("supply_dual_gate_converted") or 0))
    notes.append(
        "A3 cold-proposed hampel and deployed it on %d/4 at the same "
        "held-out +0.1867 (a3_4 identity, LLM 2 / fit 1).  The half "
        "surface is readable for a cold proposal, not only the supply "
        "channel.  Mechanism-consistent with margin gating of the "
        "confirmation surface; not a capability ranking vs G3 A3 1/4."
        % int(a3.get("deployed_target_operator") or 0))
    notes.append(
        "Every converting deploy on this unit printed the same held-out "
        "+0.1867 (deterministic given hampel_filter).  Delayed half "
        "margin sits exactly on the 2.00x bar.")
    return notes


def _funnel_block(summary: Mapping[str, Any]) -> dict[str, Any]:
    n = int(summary.get("runs") or 0)
    return {
        "card_in_fast_view": "%d/%d" % (summary.get("card_in_fast_view_runs") or 0, n),
        "entered_pool": "%d/%d" % (summary.get("entered_pool") or 0, n),
        "passed_verifier": "%d/%d" % (summary.get("passed_verifier") or 0, n),
        "support_material_positive": "%d/%d" % (
            summary.get("support_material_positive") or 0, n),
        "delayed_approved": "%d/%d" % (summary.get("delayed_approved") or 0, n),
        "supply_dual_gate_converted": "%d/%d" % (
            summary.get("supply_dual_gate_converted") or 0, n),
        "deployed_target_operator_any_path": "%d/%d" % (
            summary.get("deployed_target_operator") or 0, n),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    arith = payload["arithmetic"]
    half = arith["half_role_concat"]
    quarter = arith["quarter_sealed"]["slices"]
    lines = [
        "# M-1 -- feedback-margin gating, GPMvF half protocol",
        "",
        "protocol: `%s`  evidence grade: **%s**  git: `%s`"
        % (payload["protocol_version"], payload["evidence_grade"],
           payload["git_head"]),
        "",
        "**%s**" % verdict["verdict"], "",
        verdict.get("reason", ""), "",
        "> %s" % payload["semantic_discipline"], "",
        "## 1. Unique variable and implementation",
        "",
        "- **Fixed**: W-1 dual-source hampel card, Scope, operator, "
        "Consumer, `maximum_candidates=3`, per-run LLM/fit caps, "
        "`GunPointMaleVersusFemale__impulse_v2` substrate and inject seed.",
        "- **Unique variable**: held-in slice allocation only.",
        "- **Implementation**: one held-in round; Support = concat("
        "r1_support, r2_support) n=21; delayed = concat(r1_delayed, "
        "r2_delayed) n=19.  Dual gate preserved.  Eval-layer repack of "
        "`s1._build_cell` surfaces; methods / cell builder untouched.",
        "- **Rejected composition**: ps0b stored `half_slices` "
        "(same-round support+delayed) -- that collapses the dual gate.",
        "- **Label discipline**: half-protocol readings are "
        "margin-mechanism evidence only.  Not a capability comparison "
        "against the G3 quarter baseline.  Pilot.  GunPointFamily "
        "same-family note.  Guided positives count zero toward Source "
        "cross-domain authorization.",
        "",
        "## 2. Arithmetic precondition (0 fit)",
        "",
        "| surface | composition | n | identity | program | reading | "
        "1/n | margin | meets 2x |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, part in quarter.items():
        lines.append(
            "| quarter `%s` | sealed | %d | %d/%d | %d/%d | %+.4f | "
            "%.4f | — | material=%s |" % (
                name, part["n"], part["identity_correct"], part["n"],
                part["program_correct"], part["n"], part["reading"],
                part["material_line"], part["meets_material"]))
    for label, part in (("half Support", half["support"]),
                        ("half delayed", half["delayed"])):
        lines.append(
            "| **%s** | `%s` | %d | %d/%d | %d/%d | %+.4f | %.4f | "
            "**%.2f×** | %s |" % (
                label, "+".join(part["composed_of"]), part["n"],
                part["identity_correct"], part["n"],
                part["program_correct"], part["n"], part["reading"],
                part["material_line"], part["margin_multiplier"],
                part["meets_2x"]))
    lines += [
        "",
        "Quarter G3 margin: **1.35×** (`reproducibility_margin_ge_2x` = "
        "false).  Role-concat half min margin: **%.2f×**.  Precondition: "
        "**%s**." % (
            half["min_margin_multiplier"],
            "PASS" if arith["precondition_passed"] else "FAIL"),
        "",
        "## 3. Eight-run protocol (fresh state, half, checkpoint+resume)",
        "",
        "| run | arm | inject | Support+ | delayed | supply dual-gate | "
        "applied | held-out | worst class | LLM | fits |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload.get("runs") or []:
        inject = row["inject_funnel"]
        applied_ops = row["deployment"].get("applied_ops") or []
        applied = ",".join(applied_ops) or "identity"
        # A3 has no inject funnel; delayed is the actual winner gate.
        delayed = (inject.get("delayed_approved") if row["arm"] == ARM_SCOPED
                   else bool(row.get("first_approved_round")
                             or TARGET_OPERATOR in applied_ops))
        lines.append(
            "| %s | %s | %s | %s | %s | %s | `%s` | %+.4f | %+.4f | %s | %s |"
            % (row["run_id"], row["arm"], inject.get("entered_pool"),
               inject.get("support_material_positive"),
               delayed,
               row.get("supply_dual_gate_converted"),
               applied,
               float(row["deployment"]["heldout_accuracy_gain"] or 0),
               float(row["deployment"]["worst_class_delta"] or 0),
               row.get("llm_calls"), row.get("consumer_fits")))
    a5 = payload.get("arm_summaries", {}).get("A5-scoped") or {}
    a3 = payload.get("arm_summaries", {}).get("A3") or {}
    funnel = payload.get("supply_funnel") or {}
    lines += [
        "", "## 4. Supply-candidate six-stage funnel (A5-scoped)", "",
        "| stage | A5-scoped half | G3 quarter control (not re-run) |",
        "|---|---|---|",
    ]
    g3c = ((payload.get("g3_quarter_control") or {}).get("a5_scoped") or {})
    g3_n = int(g3c.get("runs") or 0)
    pairs = (
        ("card in Fast view", funnel.get("card_in_fast_view"),
         "%d/%d" % (g3c.get("card_in_fast_view_runs") or 0, g3_n)),
        ("entered pool", funnel.get("entered_pool"),
         "%d/%d" % (g3c.get("entered_pool") or 0, g3_n)),
        ("Support material+", funnel.get("support_material_positive"),
         "%d/%d" % (g3c.get("support_material_positive") or 0, g3_n)),
        ("delayed approved (funnel)", funnel.get("delayed_approved"),
         "%d/%d" % (g3c.get("delayed_approved") or 0, g3_n)),
        ("supply dual-gate converted",
         funnel.get("supply_dual_gate_converted"), "0/4 (G3 material+ 0/4)"),
        ("any-path family deploy",
         funnel.get("deployed_target_operator_any_path"),
         "%d/%d (includes agent-authored a5_4)" % (
             g3c.get("deployed_target_operator") or 0, g3_n)),
    )
    for name, half_v, quarter_v in pairs:
        lines.append("| %s | %s | %s |" % (name, half_v, quarter_v))
    lines += [
        "",
        "G3 quarter is a **mechanism contrast**, not a capability ranking.",
        "",
        "## 5. A3 contrast (cold proposal on the readable half surface)",
        "",
        "- A3 runs: %d; family deploys: %d; mean held-out: %+.4f; harm: %d"
        % (a3.get("runs") or 0, a3.get("deployed_target_operator") or 0,
           float(a3.get("mean_heldout_gain") or 0), a3.get("harm_runs") or 0),
        "- A3 agent families: `%s`" % ",".join(a3.get("agent_families") or []),
        "- G3 quarter A3: one cold hampel deploy (a3_2, +0.1867); this "
        "book asks whether the half surface also lets a cold proposal "
        "convert.",
        "",
        "## 6. Cost",
        "",
    ]
    ledger = payload.get("ledger") or {}
    lines += [
        "- LLM: %s / %s" % (ledger.get("llm"), ledger.get("llm_cap")),
        "- Consumer fits: %s / %s" % (ledger.get("fit"), ledger.get("fit_cap")),
        "- wall: %s s / %s s" % (ledger.get("wall_seconds"),
                                 ledger.get("wall_seconds_cap")),
        "- downloads: 0",
        "",
        "## 7. Obligations",
        "",
    ]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    addendum = verdict.get("gpovy_downshift_addendum")
    if addendum:
        lines += ["", "## 8. GPOvY down-shift addendum (not run)", ""]
        for key, value in addendum.items():
            lines.append("- **%s**: %s" % (key, value))
    lines += ["", "## 9. Outside the book", ""]
    for note in payload.get("outside_book") or []:
        lines.append("- %s" % note)
    if not payload.get("outside_book"):
        lines.append("- (none yet)")
    return "\n".join(lines) + "\n"


# =========================================================================== #
# Live protocol
# =========================================================================== #
def _checkpoint(runs, ledger, base_shas, *, started=None) -> None:
    CHECKPOINT.write_text(
        json.dumps(ps0c.redact({
            "runs": list(runs), "ledger": dict(ledger),
            "base_shas": dict(base_shas),
            "completed_run_ids": [row["run_id"] for row in runs],
            "wall_seconds_used": (round(time.time() - started, 1)
                                  if started is not None else None),
        }), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_unit_with_retry(*, cell: Any, arm: str, base_snapshot: Any,
                         backend: Any, store_root: Path) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.runtime.agent_backend import AgentTransportError

    last: Exception | None = None
    for _attempt in range(2):
        try:
            return s1.run_unit(
                unit=EXAM_UNIT, cell=cell, arm=arm,
                base_snapshot=base_snapshot, carried_episodes=(),
                agent_factory=cls._live_agent, backend=backend,
                store_root=store_root, rounds=HALF_ROUNDS,
                fit_cap=FIT_PER_RUN)
        except AgentTransportError as exc:
            last = exc
            print("transport retry after %s on %s" % (exc, store_root.name),
                  flush=True)
            time.sleep(8)
            if store_root.exists():
                shutil.rmtree(store_root)
            backend = cls._live_backend(LLM_PER_RUN)
    raise s1.Stop("BACKEND_UNAVAILABLE",
                  "relay transport failed after unit retry: %s" % last)


def _run_all(*, h0, store_root: Path, ledger, started, runs
             ) -> dict[str, str]:
    card = json.loads(
        (E2 / "ps2_cards" / "ps2_card_scoped.json").read_text(encoding="utf-8"))
    scoped_snapshot, _applied = s1._apply_entries(
        h0, [card], store_root=store_root / "bases", tag="a5_scoped")
    bases = {ARM_A3: h0, ARM_SCOPED: scoped_snapshot}
    base_shas = {ARM_A3: h0.runtime_bundle_sha,
                 ARM_SCOPED: scoped_snapshot.runtime_bundle_sha}
    quarter = s1._build_cell(EXAM_UNIT)
    cell = _half_cell(quarter)
    done = {str(row["run_id"]) for row in runs}
    for plan in RUN_PLAN:
        if plan["run_id"] in done:
            print("skip %s (checkpoint)" % plan["run_id"], flush=True)
            continue
        if ledger["llm"] >= LLM_TOTAL_CAP or ledger["fit"] >= FIT_TOTAL_CAP:
            raise Stop("COMPUTE_BUDGET_EXCEEDED",
                       "book cap reached before %s" % plan["run_id"])
        if time.time() - started > WALL_SECONDS_CAP:
            raise Stop("COMPUTE_BUDGET_EXCEEDED",
                       "wall clock cap reached before %s" % plan["run_id"])
        arm = plan["arm"]
        backend = cls._live_backend(LLM_PER_RUN)
        result = _run_unit_with_retry(
            cell=cell, arm=arm, base_snapshot=bases[arm],
            backend=backend, store_root=store_root / plan["run_id"])
        ledger["llm"] += int(result.get("llm_calls") or 0)
        ledger["fit"] += int(result.get("consumer_fits") or 0)
        scored = _score_run(plan, result, base_shas[arm])
        runs.append(scored)
        _checkpoint(runs, ledger, base_shas, started=started)
        inject = scored["inject_funnel"]
        print("%-10s %-10s inject=%-5s support+=%-5s converted=%-5s "
              "applied=%s gain=%+.4f llm=%s fit=%s"
              % (plan["run_id"], arm, inject.get("entered_pool"),
                 inject.get("support_material_positive"),
                 scored.get("supply_dual_gate_converted"),
                 ",".join(scored["deployment"].get("applied_ops") or [])
                 or "identity",
                 float(scored["deployment"]["heldout_accuracy_gain"] or 0.0),
                 scored.get("llm_calls"), scored.get("consumer_fits")),
              flush=True)
    return base_shas


def _base_payload(arith: Mapping[str, Any]) -> dict[str, Any]:
    rights = _same_rights()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "git_head": s1._git("rev-parse", "HEAD"),
        "python": sys.version.split()[0],
        "ps0b_source": PS0B_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "g3_control_commit": G3_CONTROL_COMMIT,
        "exam_unit": EXAM_UNIT,
        "run_plan": list(RUN_PLAN),
        "same_rights": rights,
        "arithmetic": arith,
        "g3_quarter_control": _g3_quarter_control(),
        "semantic_discipline": (
            "a conversion is experience supplying a candidate through the "
            "mechanical channel, adjudicated by Target feedback on the "
            "half confirmation surface.  It is not evidence that the "
            "agent learned to propose the family.  Half-protocol readings "
            "are margin-mechanism evidence only and must not be ranked "
            "as capability against the G3 quarter baseline.  A guided "
            "positive counts zero toward Source cross-domain authorization. "
            "Pilot; GunPointFamily same-family note."
        ),
    }


def _finish(payload, runs, ledger, *, base_shas, stopped, started,
            arith) -> int:
    a5 = _arm_summary(runs, ARM_SCOPED)
    a3 = _arm_summary(runs, ARM_A3)
    payload["runs"] = runs
    payload["arm_summaries"] = {"A5-scoped": a5, "A3": a3}
    payload["supply_funnel"] = _funnel_block(a5)
    payload["verdict"] = _verdict(runs, arith, stopped=stopped)
    payload["base_shas"] = base_shas
    payload["ledger"] = {
        "llm": ledger["llm"], "llm_cap": LLM_TOTAL_CAP,
        "fit": ledger["fit"], "fit_cap": FIT_TOTAL_CAP,
        "wall_seconds": round(time.time() - started, 1),
        "wall_seconds_cap": WALL_SECONDS_CAP, "downloads": 0,
    }
    payload["oracle_isolation"] = s1._oracle_isolation_report()
    payload["obligations"] = {
        "unique_variable_is_slice_allocation_only": True,
        "methods_contracts_runtime_operators_unmodified": True,
        "w1_g3_wiring_unchanged": True,
        "same_card_scope_operator_consumer_caps": True,
        "maximum_candidates_3": True,
        "grants_execution_false": True,
        "g3_quarter_baseline_not_rerun": True,
        "half_readings_not_capability_ranked_vs_quarter": True,
        "oracle_sealed_grader_only": True,
        "oracle_not_loaded_into_harness": True,
        "guided_positive_counts_zero_toward_source_auth": True,
        "downloads": 0,
        "full_repo_pytest_not_run": True,
        "pilot_gunpoint_family_note": True,
        "semantic_discipline": payload["semantic_discipline"],
    }
    payload["outside_book"] = list(payload.get("outside_book") or [])
    payload["outside_book"].extend(
        note for note in _outside_book(runs, a5, a3)
        if note not in payload["outside_book"])
    s1._dump(OUT_JSON, ps0c.redact(payload))
    OUT_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"]["verdict"],
                      "reason": payload["verdict"].get("reason"),
                      "ledger": payload["ledger"],
                      "artifact": str(OUT_JSON)},
                     ensure_ascii=False, indent=1), flush=True)
    ok = payload["verdict"]["verdict"] in (
        "MARGIN_GATING_CONFIRMED", "MARGIN_GATING_REJECTED",
        "AMBIGUOUS", "MARGIN_PRECONDITION_FAILED",
        "ARITHMETIC_PRECONDITION_PASSED")
    return 0 if ok else 1


def run(*, arith_only: bool = False, probe_only: bool = False,
        resume: bool = False) -> int:
    started = time.time()
    s1._set_phase(s1.PHASE_SETUP)
    arith = arithmetic_precondition()
    payload = _base_payload(arith)
    if arith_only:
        stopped = ("ARITHMETIC_PRECONDITION_PASSED"
                   if arith["precondition_passed"]
                   else "MARGIN_PRECONDITION_FAILED")
        return _finish(payload, [], {"llm": 0, "fit": 0},
                       base_shas={}, stopped=stopped, started=started,
                       arith=arith)
    if not arith["precondition_passed"]:
        return _finish(payload, [], {"llm": 0, "fit": 0},
                       base_shas={},
                       stopped="MARGIN_PRECONDITION_FAILED",
                       started=started, arith=arith)

    install = ps0c.install_new_backend()
    payload["backend_install"] = {"host": install.get("host"),
                                  "model": install.get("model")}
    probe = ps0c.probe_new_backend()
    payload["backend_probe"] = ps0c.redact(probe)
    if probe_only:
        payload["verdict"] = {
            "verdict": "PROBE_ONLY" if probe.get("ok") else "BACKEND_UNAVAILABLE",
            "reason": "identity probe; no live run",
            "facts": {"ok": bool(probe.get("ok")),
                      "returned_model": probe.get("returned_model")},
        }
        payload["runs"] = []
        payload["arm_summaries"] = {}
        payload["supply_funnel"] = {}
        payload["ledger"] = {"llm": 0, "llm_cap": LLM_TOTAL_CAP,
                             "fit": 0, "fit_cap": FIT_TOTAL_CAP,
                             "wall_seconds": round(time.time() - started, 1),
                             "wall_seconds_cap": WALL_SECONDS_CAP,
                             "downloads": 0}
        payload["obligations"] = {"probe_only": True, "no_llm": True}
        payload["outside_book"] = []
        s1._dump(OUT_JSON, ps0c.redact(payload))
        OUT_MD.write_text(_markdown(payload), encoding="utf-8")
        print(json.dumps({"verdict": payload["verdict"]["verdict"],
                          "returned_model": probe.get("returned_model"),
                          "ok": probe.get("ok")},
                         ensure_ascii=False, indent=1), flush=True)
        return 0 if probe.get("ok") else 1
    if not probe.get("ok"):
        raise Stop("BACKEND_UNAVAILABLE",
                   "relay probe failed: %s" % probe.get("reason"))

    ledger = {"llm": 0, "fit": 0}
    store_root = Path(tempfile.gettempdir()) / "m1_margin_gate"
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
        from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
            compile_snapshot,
        )
        h0 = compile_snapshot(
            PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
            verify_lock=False)
        base_shas = _run_all(h0=h0, store_root=store_root, ledger=ledger,
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
                   stopped=stopped, started=started, arith=arith)


def main() -> int:
    parser = argparse.ArgumentParser(description="M-1 margin-gating half protocol")
    parser.add_argument("--arith-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not (args.arith_only or args.probe_only or args.run or args.resume):
        parser.error("pass --arith-only, --probe-only, --run or --resume")
    return run(arith_only=bool(args.arith_only),
               probe_only=bool(args.probe_only),
               resume=bool(args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
