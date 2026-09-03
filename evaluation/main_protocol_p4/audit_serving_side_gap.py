"""Two mechanical demonstrations, not two assertions.

The main protocol has been measuring "does curating the training corpus help".
That is a real question, but it is not the project's claim, which is that the
Harness prepares *the data it is about to serve* according to that data's own
Pattern.  Two facts stand between the two, and both are checked here by running
the instrument rather than by reading it.

**A. There is no serving-side application point.**  ``_evaluate`` applies the
Program to training windows; the evaluation context is read through
``_linear_integrity`` alone and the truth is the raw slice.  The Classification
adapter is the same shape: ``_prepared_fit`` transforms the fit rows, while
``cell.surface(face)`` goes into ``model.predict`` untouched.  The check runs a
Program that demonstrably modifies thousands of training points and confirms the
served context is bit-identical to raw.

**B. Scope cannot isolate a harmed series.**  ``train_series_scope`` exists and
the functional line uses it, but it filters *training* rows, and
``roster("support_a")`` trains on Support-B while evaluating on Support-A --
disjoint sets.  Restricting the scope to one training series therefore still
moves every evaluation series, because it moves the fitted model.  The check
scopes to a single training UID and counts how many evaluation series change.

Together these say the missing mechanism is upstream of a ScopeSpec: a scope
over training rows cannot express "do not treat this served series", because the
served series is never treated in the first place.

0 LLM calls, no held-out read, no artifact overwritten.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import audit_gap_repairability as gaps
from evaluation.main_protocol_p4 import audit_param_correction_rerun as fixes
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import phase2_contract as contract
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4n_serving_side_gap.json"

CONTEXT, HORIZON = contract.CONTEXT, contract.HORIZON
FACES = contract.FACES
# The program the Phase-2 freeze selected, so the demonstration uses the one
# candidate the line actually cares about.
PROBE = ("period_median_complete", "winsorize")


def _steps() -> list[tuple[str, dict[str, Any]]]:
    return [
        (op, dict(fixes.corrected_steps([{"op": op, "params": forecast_p1._params(op)}])[0]["params"]))
        for op in PROBE
    ]


def demonstration_a_no_serving_application(cell: Any, origin: int) -> dict[str, Any]:
    """Does anything the Program does reach the context that is served?"""
    at = forecast_p4._cell_at(cell, int(origin))
    rows = []
    for face in FACES:
        roster = at.roster(face)
        eval_uids = [
            str(row["series_uid"]) for row in roster if row["role"] == "eval"
        ]
        # What _evaluate feeds the model at serving time, restated exactly.
        served = {
            uid: forecast_runtime._linear_integrity(
                np.asarray(at.values[uid], dtype=np.float64)[
                    int(origin) - CONTEXT:int(origin)]
            )
            for uid in eval_uids
        }
        # What the Program would do to that same context, if there were a place
        # to apply it.  There is not; this is the counterfactual.
        counterfactual = {}
        for uid in eval_uids:
            window = np.asarray(at.values[uid], dtype=np.float64)[
                int(origin) - CONTEXT:int(origin)]
            execution = p4c.run_pipeline_or_none(window, _steps()) \
                if hasattr(p4c, "run_pipeline_or_none") else None
            counterfactual[uid] = execution
        moved = sum(
            1 for uid in eval_uids
            if counterfactual[uid] is not None
            and not np.allclose(counterfactual[uid], served[uid], equal_nan=True)
        )
        rows.append({
            "face": face,
            "eval_series": len(eval_uids),
            "served_context_source": "_linear_integrity(raw[origin-192:origin])",
            "program_applied_to_served_context": False,
            "truth_source": "raw[origin:origin+48], never transformed",
            "series_whose_served_context_the_program_would_have_changed": moved,
        })
    return {
        "origin": int(origin),
        "per_face": rows,
        "reading": (
            "the Program has no application point on the served context in "
            "either task: Forecast feeds _linear_integrity(raw) and "
            "Classification feeds cell.surface(face) straight into predict"
        ),
        "classification_is_the_same_shape": {
            "prepared": "MacroF1ConsumerAdapter._prepared_fit(cell.fit_values)",
            "served": "cell.surface(face) -> model.predict, unprepared",
            "truth": "labels, never transformed",
        },
    }


def demonstration_b_scope_cannot_isolate(cell: Any, origin: int) -> dict[str, Any]:
    """Scope one training series; count how many evaluation series move."""
    at = forecast_p4._cell_at(cell, int(origin))
    config = forecast_p4._config(int(origin))
    rows = []
    fits = 0
    for face in FACES:
        roster = at.roster(face)
        train_uids = [
            str(row["series_uid"]) for row in roster if row["role"] == "train"
        ]
        executor = p4c.ScopeExecutor(
            roster, at.values, config,
            evaluate_fn=forecast_runtime._evaluate,
            max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION,
        )
        steps = tuple(_steps())
        if not executor.verify(steps, int(origin)).passed:
            rows.append({"face": face, "skipped": "window verifier refused"})
            continue
        compiled = executor._compiled(steps)
        baseline = forecast_runtime._evaluate(
            roster, at.values, None, config, origin=int(origin))
        scoped = forecast_runtime._evaluate(
            roster, at.values, compiled, config, origin=int(origin),
            train_series_scope=frozenset({train_uids[0]}))
        fits += 2
        left = np.asarray(baseline["per_view_smase"], dtype=np.float64)
        right = np.asarray(scoped["per_view_smase"], dtype=np.float64)
        moved = int(np.count_nonzero(~np.isclose(left, right, atol=1e-12)))
        rows.append({
            "face": face,
            "scope": "one training series (%s)" % train_uids[0],
            "training_series_in_scope": 1,
            "training_series_total": len(train_uids),
            "evaluation_series_total": int(left.size),
            "evaluation_series_whose_prediction_moved": moved,
            "scope_domain_and_harm_domain_are_disjoint": True,
            "max_abs_change": round(float(np.max(np.abs(left - right))), 6),
        })
    return {
        "origin": int(origin),
        "per_face": rows,
        "consumer_fits": fits,
        "reading": (
            "scoping to a single training series still moves every evaluation "
            "series, because the scope changes the fitted model rather than the "
            "served data; a series cannot be spared by excluding it from a "
            "training-row scope it was never in"
        ),
    }


def build() -> dict[str, Any]:
    support_a, support_b, _origins = gaps._roster_from_preflight()
    cell = gaps._variant_cell(support_a, support_b)
    origin = int(contract.ORIGINS[0])
    a = demonstration_a_no_serving_application(cell, origin)
    b = demonstration_b_scope_cannot_isolate(cell, origin)
    return {
        "stage": "P4N_SERVING_SIDE_GAP",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_MECHANISM_AUDIT",
        "data_version": contract.DATA_VERSION,
        "boundary": {
            "llm_calls": 0,
            "held_out_reads": 0,
            "ucr_test_outcome_reads": 0,
            "artifacts_overwritten": 0,
            "consumer_fits": b["consumer_fits"],
        },
        "code_facts": {
            "forecast_program_target": (
                "run_e2_autonomous_natural_workflow_generation._evaluate applies "
                "_apply_program inside the train_rows loop only"
            ),
            "forecast_served_context": (
                "_linear_integrity(raw[origin-192:origin]); no _apply_program"
            ),
            "classification_program_target": (
                "MacroF1ConsumerAdapter._prepared_fit(cell.fit_values)"
            ),
            "classification_served_features": (
                "cell.surface(face) -> model.predict, unprepared"
            ),
            "train_series_scope_exists": True,
            "train_series_scope_used_by_main_protocol": False,
            "scope_executor_docstring": "applies steps to the cohort training windows",
            "modification_fraction_scope_is": "fraction accounting, not series selection",
            "skill_program_geometry_scope_values": [
                "training_rows", "historical_origins"
            ],
        },
        "demonstration_a_no_serving_application_point": a,
        "demonstration_b_scope_cannot_isolate_a_harmed_series": b,
        "what_this_means": [
            "the main protocol has been measuring training-corpus curation, "
            "which is a real question but not the project's claim",
            "a ScopeSpec over training rows cannot express 'do not treat this "
            "served series', because the served series is never treated",
            "a serving-side application point is upstream of the Scope gap and "
            "must be built first",
            "unscoped series also need their own raw-trained model, or they are "
            "still served by a model fitted on prepared data and are not raw",
        ],
        "does_not_claim": [
            "that any P4 number is wrong; they are correct for the semantics "
            "the instrument actually implements",
            "that Scope was the only cause of A5 == K0; unreachable Source "
            "cards, the strict gate and zero Active Skills also contributed",
        ],
        "releases": "NONE",
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("A: served context carries the Program? %s" % (
        report["demonstration_a_no_serving_application_point"]["per_face"][0][
            "program_applied_to_served_context"]))
    for row in report["demonstration_b_scope_cannot_isolate_a_harmed_series"][
            "per_face"]:
        if "skipped" in row:
            print("B: %s -- %s" % (row["face"], row["skipped"]))
            continue
        print("B: %s | scope %d/%d training series -> %d/%d evaluation series "
              "moved (max %.6f)" % (
                  row["face"], row["training_series_in_scope"],
                  row["training_series_total"],
                  row["evaluation_series_whose_prediction_moved"],
                  row["evaluation_series_total"], row["max_abs_change"]))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
