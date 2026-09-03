"""The three arms that need no Agent, run first and sealed before A3 and A5.

Static, Global Best-Fixed and the open-loop tree are references, not the method.
They are run now because they cost no LLM call and because sealing them first is
what stops them from quietly informing the arms they are supposed to bound: the
contract forbids using these numbers to change A3 or A5.

All three go through the serving-side dual pipeline, so they are commensurable
with what the Harness will do:

* **Static** declines everything.  Its gain is zero by construction and its
  absolute sMASE is the reference every other arm is scored against.
* **Global Best-Fixed** treats every served series with the one program Phase 2
  froze -- the global program a selective Harness has to beat.
* **Open-loop Targeter** assigns a program per series from the frozen tree.
  Under the dual pipeline that assignment *is* a Scope, so this baseline is the
  closest non-Agent relative of the thing on trial: same action space, no
  feedback.  Its cost is reported because per-series assignment needs one
  program pipeline per distinct program chosen.

Cohort 3 has never been fitted, and the held-out block stays closed.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import audit_cross_fitted_targeting as targeting
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import natural_structure_features as x1
from evaluation.main_protocol_p4 import main_experiment_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_phase2_confirmation as phase2
from evaluation.main_protocol_p4 import run_phase2_development as dev
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4v_main_baselines.json"

FACES = contract.FACES
MATERIAL = 0.005


def _cell(uids: Sequence[str]) -> Any:
    variant = preflight.load_variant()
    support_a, support_b = tuple(uids[:20]), tuple(uids[20:40])
    values = {uid: variant[uid] for uid in uids}
    return forecast_p1.ForecastCell(
        values=values, support_a=support_a, support_b=support_b,
        observation_block=np.asarray(
            values[support_a[0]][:forecast_p1.ORIGIN], dtype=np.float64),
    ), variant


def _steps_of(entry: Mapping[str, Any],
              by_id: Mapping[str, Any]) -> tuple[tuple[str, dict], ...]:
    return tuple(
        (str(step["op"]), dict(step["params"]))
        for step in by_id[entry["program_id"]]["steps"]
    )


def _x1_design(variant: Mapping[str, np.ndarray], uids: Sequence[str],
               origin: int) -> np.ndarray:
    """X1 for one cell, in the frozen column order the tree was trained on."""
    block, _names = targeting.series_features(variant, list(uids), origin)
    contexts = [
        np.asarray(variant[uid][origin - contract.CONTEXT:origin],
                   dtype=np.float64)
        for uid in uids
    ]
    extra, _ = x1.matrix(contexts, period=forecast_p1.PERIOD)
    return np.hstack([block, extra])


def _try_scope(executor: Any, roster: Any, values: Any, config: Any,
               origin: int, steps: tuple, scope: frozenset[str],
               base: np.ndarray) -> tuple[np.ndarray, int, str | None]:
    """Run one (program, scope) pair, or say why it is illegal here.

    Two legality gates now exist, and neither may be silently softened: the
    window verifier on the training side, and a served context that the program
    flattens.  A refusal returns Static-equal gains so the arm is scored on what
    it could actually deploy, and the reason is recorded rather than averaged
    away.
    """
    if not executor.verify(steps, origin).passed:
        return np.zeros_like(base), 0, "WINDOW_VERIFIER_REJECTED"
    try:
        reading = scoped.scoped_evaluate(
            roster, values, executor._compiled(steps), config,
            origin=origin, scope=scope)
    except scoped.ServingContextDegenerate as exc:
        return np.zeros_like(base), 0, "SERVING_CONTEXT_DEGENERATE: %s" % (
            str(exc)[:120])
    except Exception as exc:  # noqa: BLE001 - an unusable pair is a reading
        return (np.zeros_like(base), 0,
                "%s: %s" % (type(exc).__name__, str(exc)[:100]))
    gains = base - np.asarray(reading["per_view_smase"], dtype=np.float64)
    return gains, int(reading["consumer_fits"]), None


def _budget(vector: np.ndarray) -> dict[str, Any]:
    finite = vector[np.isfinite(vector)]
    if finite.size == 0:
        return {"mean_gain": None, "clears_bounded_budget": False}
    harmed = float((finite < -MATERIAL).mean())
    worst = max(0.0, float(-finite.min()))
    mean = float(finite.mean())
    return {
        "mean_gain": round(mean, 6),
        "harmed_fraction": round(harmed, 4),
        "max_single_series_harm": round(worst, 6),
        "clears_bounded_budget": bool(
            mean >= MATERIAL
            and harmed <= bounded.BOUNDED_MAX_HARMED_FRACTION
            and worst <= bounded.BOUNDED_MAX_SINGLE_SERIES_HARM),
    }


def run() -> dict[str, Any]:
    started = time.time()
    state = contract.assert_frozen()
    if not state["frozen"]:
        raise RuntimeError("main experiment contract drifted: %s"
                           % state["failures"])
    groups = contract.cohorts()
    cell, variant = _cell(groups["target"])
    origins = list(contract.HELD_IN_ORIGINS)

    frozen = phase2.freeze_on_cohort_1()
    by_id = {p["program_id"]: p for p in dev.frozen_program_space()}
    best = frozen["best_fixed"]["O0"]
    best_steps = _steps_of(best, by_id)
    menu = frozen["menus"]["O0"]
    targeter = frozen["targeters"]["X1_O0"]["_model"]

    fits = 0
    per_arm: dict[str, list[np.ndarray]] = {
        "Static": [], "GlobalBestFixed": [], "OpenLoopTargeter": []}
    assignments: list[dict[str, Any]] = []
    refusals: list[dict[str, Any]] = []
    treated: dict[str, list[tuple[int, int]]] = {
        "Static": [], "GlobalBestFixed": [], "OpenLoopTargeter": []}
    reference_smase: list[float] = []

    for origin in origins:
        at = forecast_p4._cell_at(cell, int(origin))
        config = forecast_p4._config(int(origin))
        for face in FACES:
            roster = at.roster(face)
            eval_uids = [
                str(row["series_uid"]) for row in roster if row["role"] == "eval"]
            executor = p4c.ScopeExecutor(
                roster, at.values, config,
                evaluate_fn=views.forecast_runtime._evaluate,
                max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)

            static = scoped.scoped_evaluate(
                roster, at.values, None, config, origin=int(origin))
            fits += static["consumer_fits"]
            base = np.asarray(static["per_view_smase"], dtype=np.float64)
            reference_smase.append(float(static["mean_smase"]))
            per_arm["Static"].append(np.zeros_like(base))
            treated["Static"].append((0, len(eval_uids)))

            # Global Best-Fixed: treat everyone.
            gains, spent, refusal = _try_scope(
                executor, roster, at.values, config, int(origin),
                best_steps, frozenset(eval_uids), base)
            fits += spent
            per_arm["GlobalBestFixed"].append(gains)
            treated["GlobalBestFixed"].append(
                (0 if refusal else len(eval_uids), len(eval_uids)))
            if refusal:
                refusals.append({"arm": "GlobalBestFixed", "origin": int(origin),
                                 "face": face, "reason": refusal,
                                 "series_abstained": len(eval_uids)})

            # Open-loop Targeter: the frozen tree's per-series assignment is a
            # Scope, so it runs one program pipeline per distinct choice.
            design = _x1_design(variant, eval_uids, int(origin))
            picks = (targeter.predict(design).astype(int)
                     if targeter is not None
                     else np.zeros(len(eval_uids), dtype=int))
            chosen = base - base  # zeros: anything unassigned stays Static
            histogram: dict[str, int] = {}
            treated_here = 0
            for index in sorted(set(picks.tolist())):
                entry = menu[int(index)]
                histogram[entry["entry"]] = int((picks == index).sum())
                if entry["program_id"] == dev.EMPTY:
                    continue
                group = frozenset(
                    uid for uid, pick in zip(eval_uids, picks) if pick == index)
                gains, spent, refusal = _try_scope(
                    executor, roster, at.values, config, int(origin),
                    _steps_of(entry, by_id), group, base)
                fits += spent
                if refusal:
                    # Fail-closed: these series fall back to Static.  Correct
                    # behaviour, but abstention -- so they are not covered.
                    refusals.append({
                        "arm": "OpenLoopTargeter", "origin": int(origin),
                        "face": face, "entry": entry["entry"],
                        "reason": refusal, "series_abstained": len(group)})
                    continue
                treated_here += len(group)
                mask = np.array([uid in group for uid in eval_uids])
                chosen = np.where(mask, gains, chosen)
            per_arm["OpenLoopTargeter"].append(chosen)
            treated["OpenLoopTargeter"].append((treated_here, len(eval_uids)))
            assigned = int(sum(
                menu[int(p)]["program_id"] != dev.EMPTY for p in picks))
            assignments.append({
                "origin": int(origin), "face": face,
                "picks": histogram,
                "assigned": assigned,
                "treated": treated_here,
                "nominal_coverage": contract.deployment_coverage(
                    assigned, len(eval_uids)),
                "deployment_coverage": contract.deployment_coverage(
                    treated_here, len(eval_uids)),
            })
        print("  origin %d done (%d fits, %.1f min)" % (
            origin, fits, (time.time() - started) / 60), flush=True)

    arms = {
        name: {
            **_budget(np.concatenate(blocks)),
            "cells": len(blocks),
            "per_cell_clears": sum(
                1 for block in blocks
                if _budget(block)["clears_bounded_budget"]),
        }
        for name, blocks in per_arm.items()
    }
    for name, rows in treated.items():
        done = sum(count for count, _total in rows)
        served = sum(total for _count, total in rows)
        arms[name]["deployment_coverage"] = contract.deployment_coverage(
            done, served)
        arms[name]["series_treated"] = done
        arms[name]["series_served"] = served
        arms[name]["series_abstained"] = served - done
    arms["GlobalBestFixed"]["choice"] = best["entry"]
    return {
        "stage": "P4V_MAIN_BASELINES",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_NON_AGENT_BASELINES",
        "data_version": contract.DATA_VERSION,
        "frozen_contract": state,
        "cohort": "target readable[80:120] -- never fitted before",
        "origins": origins,
        "sealed_before": "A3 and A5; these numbers may not change them",
        "boundary": {
            **contract.BOUNDARY,
            "consumer_fits": fits,
            "llm_calls": 0,
            "held_out_reads": 0,
        },
        "reference_mean_smase": round(float(np.mean(reference_smase)), 6),
        "arms": arms,
        "targeter_assignments": assignments,
        "refusals": refusals,
        "coverage_semantics": contract.COVERAGE_SEMANTICS_SUMMARY,
        "corrections": [
            {
                "what": "deployment coverage was reported as 1.0 for both "
                        "treating arms even though some (program, scope) pairs "
                        "were refused and fell back to Static",
                "why_wrong": "a fail-closed abstention is correct behaviour but "
                             "is not coverage; counting it inflates an arm's "
                             "apparent reach",
                "now": "coverage counts treated series only; the assigned count "
                       "is kept beside it as nominal_coverage",
                "superseded_file":
                    "p4v_main_baselines.nominal_coverage_superseded.json",
            }
        ],
        "refusal_note": (
            "a refused (program, scope) pair scores as Static for the "
            "series it would have treated: the arm is credited with what "
            "it could actually deploy, not with what it proposed"
        ),
        "wall_seconds": round(time.time() - started, 1),
        "releases": "NONE",
    }


def main() -> int:
    report = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("%-20s %11s %9s %10s %9s %9s" % (
        "arm", "mean gain", "harmed", "max harm", "coverage", "cells ok"))
    for name in ("Static", "GlobalBestFixed", "OpenLoopTargeter"):
        arm = report["arms"][name]
        print("%-20s %+11.6f %9.2f %10.4f %9s %5d/%d" % (
            name, arm["mean_gain"], arm["harmed_fraction"],
            arm["max_single_series_harm"],
            arm["deployment_coverage"],
            arm["per_cell_clears"], arm["cells"]))
    print("reference sMASE : %.6f" % report["reference_mean_smase"])
    print("consumer fits   : %d in %.1f min" % (
        report["boundary"]["consumer_fits"], report["wall_seconds"] / 60))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
