"""Development-only target-local probe replay on exposed Tourism Quarterly."""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import smase
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    _group_removal_predictions,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_tourism_quarterly_fresh_target import (
    _panel,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_tourism_workflow_executor_development import (
    _surface,
    _workflows,
)
from SelfEvolvingHarnessTS.methods.ttha.skill_acquisition import (
    attach_delayed_outcomes,
    plan_skill_card_support_only,
    plan_support_only,
    policy_adaptation_auc,
)
from SelfEvolvingHarnessTS.methods.ttha.workflow_execution import (
    replay_bound_workflow_prediction,
)
DATASET = "external_monash:tourism_quarterly"
PERIOD, CONTEXT, HORIZON = 4, 16, 8
HISTORICAL_CUTOFF, SUPPORT_CUTOFF, QUERY_CUTOFF = 48, 56, 64
HISTORICAL_ANCHORS = (24, 28, 32, 36, 40)
SUPPORT_ANCHORS = (32, 36, 40, 44, 48)
QUERY_ANCHORS = (40, 44, 48, 52, 56)
BLOCKS = ((0, 4), (4, 8))
WORKFLOWS = ("W_rowblock", "W_curation", "W_temporal_origin")
SKILL_PATH = "artifacts/functional/e2/historical_policy_episode_workflow_capability.json"
OLD_REPORT_PATH = "artifacts/functional/e2/tourism_quarterly_fresh_target_report.json"
PLAN_PATH = "artifacts/functional/e2/tourism_quarterly_target_local_probe_development_plan.json"
REPORT_PATH = "artifacts/functional/e2/tourism_quarterly_target_local_probe_development_report.json"
EXPERIMENT_ID = "E2.89-tourism-quarterly-target-local-probe-development"

def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def plan(root: Path) -> dict[str, Any]:
    train, unused, target, arrays = _panel(root, query_end=QUERY_CUTOFF)
    if [row.entity_id for row in unused] != ["T15", "T16", "T17"]:
        raise ValueError("frozen unused cohort changed")
    surface_args = {"period": PERIOD, "context_length": CONTEXT, "horizon": HORIZON}
    historical = _surface(
        train, target, arrays, cutoff=HISTORICAL_CUTOFF,
        anchors=HISTORICAL_ANCHORS, **surface_args,
    )
    historical_truth = np.asarray(
        [arrays[row.series_uid][HISTORICAL_CUTOFF:SUPPORT_CUTOFF] for row in target]
    )
    historical_workflows = _workflows(
        historical, train, HISTORICAL_ANCHORS, historical_truth, (0, 1, 2),
        horizon=HORIZON, blocks=BLOCKS,
    )
    support = _surface(
        train, target, arrays, cutoff=SUPPORT_CUTOFF,
        anchors=SUPPORT_ANCHORS, **surface_args,
    )
    support_truth = np.asarray(
        [arrays[row.series_uid][SUPPORT_CUTOFF:QUERY_CUTOFF] for row in target]
    )
    support_workflows = _workflows(
        support, train, SUPPORT_ANCHORS, support_truth, (0, 1, 2),
        horizon=HORIZON, blocks=BLOCKS,
    )
    query = _surface(
        train, target, arrays, cutoff=QUERY_CUTOFF,
        anchors=QUERY_ANCHORS, **surface_args,
    )
    historical_episode = {"workflows": {
        key: {"workflow_id": key, "support_gain": float(value["support_gain"])}
        for key, value in historical_workflows.items()
    }}
    support_only = {
        key: {"support_gain": float(value["support_gain"])}
        for key, value in support_workflows.items()
    }
    candidate = _read(root / SKILL_PATH)
    historical_card_status = str(candidate["status"])
    candidate["status"] = "CANDIDATE"
    a5 = plan_skill_card_support_only(
        candidate, lambda key: support_only[key],
        historical_episode=historical_episode, allow_candidate_replay=True,
    )
    a3 = [
        plan_support_only(
            WORKFLOWS, order, lambda key: support_only[key],
            control="stop_on_first_positive",
        )
        for order in permutations(WORKFLOWS)
    ]
    replay_calls = 0

    def group_predict(rows, target_block, removal_strength):
        nonlocal replay_calls
        replay_calls += 1
        return _group_removal_predictions(
            np, reference=query["reference"], selected_local_indices=rows,
            target_block=target_block, removal_strength=removal_strength,
        )

    predictions = {"IDENTITY": np.asarray(query["reference"]["baseline_prediction"])}
    predictions.update({
        key: replay_bound_workflow_prediction(
            query["reference"], value, group_predict=group_predict
        )
        for key, value in support_workflows.items()
    })
    roster = lambda rows: [row.entity_id for row in rows]
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset": DATASET,
        "data_role": "E2.88-exposed target cohort / development replay only",
        "roster": {"train": roster(train), "unused": roster(unused), "target": roster(target)},
        "geometry": {
            "period": PERIOD, "context": CONTEXT, "horizon": HORIZON,
            "historical_cutoff": HISTORICAL_CUTOFF,
            "support_cutoff": SUPPORT_CUTOFF, "query_cutoff": QUERY_CUTOFF,
            "historical_anchors": list(HISTORICAL_ANCHORS),
            "support_anchors": list(SUPPORT_ANCHORS),
            "query_anchors": list(QUERY_ANCHORS), "blocks": [list(row) for row in BLOCKS],
        },
        "composition": "COMPOSE_WORKFLOW(target_local_phase_aligned_probe)",
        "candidate_replay": {
            "capability_id": candidate["capability_id"], "status": "CANDIDATE",
            "historical_card_status": historical_card_status,
            "active_reader_used": False, "fast_path_changed": False,
        },
        "historical_workflow_gains": {
            key: float(value["support_gain"]) for key, value in historical_workflows.items()
        },
        "target_local_support_workflows": {
            key: {"decision": value["decision"],
                  "support_gain": float(value["support_gain"]),
                  "binding": value.get("bound_action", value.get("bound_groups"))}
            for key, value in support_workflows.items()
        },
        "A5_support_plan": a5, "A3_support_plans": a3,
        "query_predictions": {key: value.tolist() for key, value in predictions.items()},
        "query_normalization": query["metadata"],
        "information_boundary": {
            "context_exposure": "INSTANCE_SEEN_DEVELOPMENT",
            "outcome_exposure": "EXPOSED_BY_E2_88_BUT_NOT_READ_IN_PLAN",
            "historical_truth_half_open": [48, 56],
            "target_local_support_truth_half_open": [56, 64],
            "query_truth_half_open": [64, 72], "query_truth_read": False,
            "target_numeric_parse_end_exclusive": 64,
            "T15_T17_used_for_feedback_or_prediction": False,
            "query_predictions_use_support_binding_replay_only": True,
            "repair_truth_used": False,
        },
        "compute": {"ridge_reference_solve_count": 3,
                    "replay_small_matrix_solve_count": replay_calls,
                    "llm_api_call_count": 0},
    }


def evaluate(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    _, _, target, arrays = _panel(root, query_end=QUERY_CUTOFF + HORIZON)
    if frozen.get("experiment_id") != EXPERIMENT_ID or frozen["roster"]["target"] != [
        row.entity_id for row in target
    ]:
        raise ValueError("frozen development plan/target mismatch")
    truth = np.asarray(
        [arrays[row.series_uid][QUERY_CUTOFF:QUERY_CUTOFF + HORIZON] for row in target]
    )
    metadata, predictions = frozen["query_normalization"], frozen["query_predictions"]

    def losses(key: str) -> np.ndarray:
        normalized = np.asarray(predictions[key], dtype=np.float64)
        return np.asarray([
            smase(truth[index], normalized[index] * float(metadata[index]["scale"])
                  + float(metadata[index]["center"]),
                  scale=float(metadata[index]["seasonal_scale"]))
            for index in range(3)
        ])

    baseline = losses("IDENTITY")
    delayed = {key: float(np.mean(baseline - losses(key))) for key in WORKFLOWS}
    a5 = attach_delayed_outcomes(frozen["A5_support_plan"], delayed)
    a3_rows = [attach_delayed_outcomes(row, delayed) for row in frozen["A3_support_plans"]]
    a3_curve = [{
        "budget": budget,
        "fixed_query_gain": statistics.fmean(
            float(row["adaptation_curve"][budget]["fixed_query_gain"]) for row in a3_rows),
        "abstention_probability": statistics.fmean(
            float(row["adaptation_curve"][budget]["abstained"]) for row in a3_rows),
    } for budget in range(len(WORKFLOWS) + 1)]
    a5_auc, a3_auc = float(a5["adaptation_auc"]), policy_adaptation_auc(a3_curve)
    harm = sum(float(row["fixed_query_gain"]) < 0.0 for row in a5["adaptation_curve"][1:])
    nonidentity = any(row["selected_workflow"] != "IDENTITY"
                      for row in a5["adaptation_curve"][1:])
    old = _read(root / OLD_REPORT_PATH)["delayed_evaluator"]
    passed = bool(a5_auc > a3_auc and harm == 0 and nonidentity)
    kept = {key: value for key, value in frozen.items() if "prediction" not in key}
    return {
        **kept,
        "information_boundary": {
            **frozen["information_boundary"], "outcome_exposure":
            "ALREADY_EXPOSED_BY_E2_88_DEVELOPMENT_EVALUATE",
            "query_truth_read": True, "target_numeric_parse_end_exclusive": 72,
        },
        "development_evaluator": {
            "workflow_gains": delayed, "A5_curve": a5["adaptation_curve"],
            "A5_adaptation_auc": a5_auc, "A3_order_average_curve": a3_curve,
            "A3_adaptation_auc": a3_auc, "A5_minus_A3": a5_auc - a3_auc,
            "A5_harm_count": harm, "A5_nonidentity": nonidentity,
        },
        "E2_88_cross_series_support_diagnostic": {
            "old_A5_adaptation_auc": float(old["A5_adaptation_auc"]),
            "old_A5_harm_count": int(old["A5_harm_count"]),
            "old_selected_workflow": old["A5_curve"][1]["selected_workflow"],
            "old_selected_query_gain": float(old["A5_curve"][1]["fixed_query_gain"]),
            "target_local_minus_old_A5_auc": a5_auc - float(old["A5_adaptation_auc"]),
        },
        "gate": {"target_local_A5_strictly_above_equal_budget_A3": a5_auc > a3_auc,
                 "target_local_A5_harm_zero": harm == 0,
                 "target_local_A5_nonidentity": nonidentity, "passed": passed},
        "capability_or_memory_written": False, "fast_path_changed": False,
        "verdict": "TARGET_LOCAL_PROBE_DEVELOPMENT_PASS" if passed
                   else "TARGET_LOCAL_PROBE_DEVELOPMENT_FAIL",
        "claim_limit": (
            "Exposed Tourism Quarterly development diagnosis only; not fresh, Promotion, "
            "Fast-Path reactivation, or Memory evidence."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "evaluate"), required=True)
    args = parser.parse_args()
    path = root / (PLAN_PATH if args.phase == "plan" else REPORT_PATH)
    payload = plan(root) if args.phase == "plan" else evaluate(root, _read(root / PLAN_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    print(payload.get("verdict", "PLAN_FROZEN"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
