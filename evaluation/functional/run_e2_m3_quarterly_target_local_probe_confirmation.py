"""Plan/evaluate the frozen target-local Workflow probe on fresh M3 Quarterly."""

from __future__ import annotations

import argparse
import copy
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
    _read,
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
    read_active_skill_cards,
)
from SelfEvolvingHarnessTS.methods.ttha.workflow_execution import (
    replay_bound_workflow_prediction,
)

DATASET = "external_monash:m3_quarterly"
ARCHIVE, MEMBER, MIN_LENGTH = (
    "data/m3_quarterly_dataset.zip", "m3_quarterly_dataset.tsf", 72,
)
EXPECTED_METADATA = {"frequency": "quarterly", "horizon": "8", "missing": "false"}
PERIOD, CONTEXT, HORIZON = 4, 16, 8
HISTORICAL_CUTOFF, SUPPORT_CUTOFF, QUERY_CUTOFF = 48, 56, 64
HISTORICAL_ANCHORS = (24, 28, 32, 36, 40)
SUPPORT_ANCHORS = (32, 36, 40, 44, 48)
QUERY_ANCHORS = (40, 44, 48, 52, 56)
BLOCKS = ((0, 4), (4, 8))
WORKFLOWS = ("W_rowblock", "W_curation", "W_temporal_origin")
SKILL_PATH = "artifacts/functional/e2/historical_policy_episode_workflow_capability.json"
OVERLAY_PATH = "artifacts/functional/e2/historical_policy_episode_workflow_state_update_e288.json"
PLAN_PATH = "artifacts/functional/e2/m3_quarterly_target_local_probe_confirmation_plan.json"
REPORT_PATH = "artifacts/functional/e2/m3_quarterly_target_local_probe_confirmation_report.json"
EXPERIMENT_ID = "E2.90-m3-quarterly-target-local-probe-confirmation"


def _m3_panel(root: Path, *, query_end: int):
    return _panel(
        root, query_end=query_end, archive=ARCHIVE, member=MEMBER,
        min_length=MIN_LENGTH, expected_metadata=EXPECTED_METADATA,
    )


def plan(root: Path) -> dict[str, Any]:
    train, unused, target, arrays = _m3_panel(root, query_end=QUERY_CUTOFF)
    surface_args = {"period": PERIOD, "context_length": CONTEXT, "horizon": HORIZON}
    historical = _surface(
        train, target, arrays, cutoff=HISTORICAL_CUTOFF,
        anchors=HISTORICAL_ANCHORS, **surface_args,
    )
    historical_truth = np.asarray([
        arrays[row.series_uid][HISTORICAL_CUTOFF:SUPPORT_CUTOFF] for row in target
    ])
    historical_workflows = _workflows(
        historical, train, HISTORICAL_ANCHORS, historical_truth, (0, 1, 2),
        horizon=HORIZON, blocks=BLOCKS,
    )
    support = _surface(
        train, target, arrays, cutoff=SUPPORT_CUTOFF,
        anchors=SUPPORT_ANCHORS, **surface_args,
    )
    support_truth = np.asarray([
        arrays[row.series_uid][SUPPORT_CUTOFF:QUERY_CUTOFF] for row in target
    ])
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
    historical_card, restriction = _read(root / SKILL_PATH), _read(root / OVERLAY_PATH)
    active = read_active_skill_cards([historical_card], state_updates=[restriction])
    if active or restriction.get("status") != "RESTRICTED":
        raise ValueError("E2.88 restriction overlay is not active")
    candidate = copy.deepcopy(historical_card)
    candidate["status"] = "CANDIDATE"
    a5 = plan_skill_card_support_only(
        candidate, lambda key: support_only[key], historical_episode=historical_episode,
        allow_candidate_replay=True,
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
            query["reference"], value, group_predict=group_predict,
        )
        for key, value in support_workflows.items()
    })
    roster = lambda rows: [
        {"series_name": row.entity_id, "file_order_index_1based": row.file_order_index}
        for row in rows
    ]
    return {
        "experiment_id": EXPERIMENT_ID, "dataset": DATASET,
        "data_role": (
            "fresh external confirmation Target; aggregate metadata/length feasibility seen; "
            "Query outcome sealed until evaluate"
        ),
        "roster": {"train": roster(train), "unused": roster(unused), "target": roster(target)},
        "geometry": {
            "period": PERIOD, "context": CONTEXT, "horizon": HORIZON,
            "historical_cutoff": HISTORICAL_CUTOFF, "support_cutoff": SUPPORT_CUTOFF,
            "query_cutoff": QUERY_CUTOFF, "historical_anchors": list(HISTORICAL_ANCHORS),
            "support_anchors": list(SUPPORT_ANCHORS), "query_anchors": list(QUERY_ANCHORS),
            "blocks": [list(row) for row in BLOCKS],
        },
        "composition": "COMPOSE_WORKFLOW(target_local_phase_aligned_probe)",
        "candidate_replay": {
            "capability_id": candidate["capability_id"], "status": "CANDIDATE",
            "historical_card_status": historical_card["status"],
            "restriction_overlay_status": restriction["status"],
            "active_reader_result_count": len(active), "allow_candidate_replay": True,
            "fast_path_changed": False,
        },
        "historical_workflow_gains": {
            key: float(value["support_gain"]) for key, value in historical_workflows.items()
        },
        "target_local_support_workflows": {
            key: {"decision": value["decision"], "support_gain": float(value["support_gain"]),
                  "binding": value.get("bound_action", value.get("bound_groups"))}
            for key, value in support_workflows.items()
        },
        "A5_support_plan": a5, "A3_support_plans": a3,
        "query_predictions": {key: value.tolist() for key, value in predictions.items()},
        "query_normalization": query["metadata"],
        "information_boundary": {
            "pre_plan_context_exposure": "AGGREGATE_SEEN",
            "plan_context_exposure": "INSTANCE_SEEN_AFTER_COMPOSITION_FREEZE",
            "outcome_exposure": "LOGICALLY_SEALED_NOT_NUMERICALLY_MATERIALIZED",
            "historical_truth_half_open": [48, 56],
            "target_local_support_truth_half_open": [56, 64],
            "query_truth_half_open": [64, 72], "query_truth_read": False,
            "target_numeric_parse_end_exclusive": 64,
            "query_numeric_parser": "np.fromstring(count=64)",
            "query_truth_stored": False, "query_gain_stored": False,
            "storage_level_sealing": False,
            "unused_cohort_used_for_feedback_or_prediction": False,
            "query_predictions_use_support_binding_replay_only": True,
            "repair_truth_used": False,
        },
        "compute": {"ridge_reference_solve_count": 3,
                    "replay_small_matrix_solve_count": replay_calls, "llm_api_call_count": 0},
    }


def evaluate(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    train, unused, target, arrays = _m3_panel(root, query_end=QUERY_CUTOFF + HORIZON)
    roster = lambda rows: [row.entity_id for row in rows]
    expected = [roster(rows) for rows in (train, unused, target)]
    observed = [[row["series_name"] for row in frozen["roster"][key]]
                for key in ("train", "unused", "target")]
    if frozen.get("experiment_id") != EXPERIMENT_ID or observed != expected:
        raise ValueError("frozen M3 Quarterly plan/roster mismatch")
    truth = np.asarray([
        arrays[row.series_uid][QUERY_CUTOFF:QUERY_CUTOFF + HORIZON] for row in target
    ])
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
    passed = bool(a5_auc > a3_auc and harm == 0 and nonidentity)
    kept = {key: value for key, value in frozen.items() if "prediction" not in key}
    return {
        **kept,
        "information_boundary": {**frozen["information_boundary"],
                                 "outcome_exposure": "EXPOSED_ONCE_IN_EVALUATE",
                                 "query_truth_read": True,
                                 "target_numeric_parse_end_exclusive": 72,
                                 "query_numeric_parser": "np.fromstring(count=72)",
                                 "query_truth_stored": False,
                                 "query_gain_stored": True},
        "delayed_evaluator": {
            "workflow_gains": delayed, "A5_curve": a5["adaptation_curve"],
            "A5_adaptation_auc": a5_auc, "A3_order_average_curve": a3_curve,
            "A3_adaptation_auc": a3_auc, "A5_minus_A3": a5_auc - a3_auc,
            "A5_harm_count": harm, "A5_nonidentity": nonidentity,
        },
        "gate": {"A5_strictly_above_equal_budget_A3": a5_auc > a3_auc,
                 "A5_harm_zero": harm == 0, "A5_nonidentity": nonidentity,
                 "passed": passed},
        "capability_or_memory_written": False, "fast_path_changed": False,
        "verdict": "M3_QUARTERLY_TARGET_LOCAL_PROBE_CONFIRMATION_PASS" if passed
                   else "M3_QUARTERLY_TARGET_LOCAL_PROBE_CONFIRMATION_FAIL",
        "claim_limit": "Fresh M3 Quarterly confirmation evidence for this frozen roster only.",
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
