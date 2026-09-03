"""Development replay of the admitted natural Workflow Skill on Tourism Monthly."""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import seasonal_scale, smase
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    RIDGE_ALPHA,
    _group_removal_predictions,
    _ridge_reference_and_removal_predictions,
)
from SelfEvolvingHarnessTS.methods.ttha.skill_acquisition import (
    attach_delayed_outcomes,
    plan_skill_card_support_only,
    plan_support_only,
    policy_adaptation_auc,
    read_active_skill_cards,
)
from SelfEvolvingHarnessTS.methods.ttha.workflow_execution import (
    execute_rowblock_support_only,
    execute_whole_group_curation_support_only,
)


DATASET = "legacy_monash:tourism_monthly"
PERIOD, CONTEXT, HORIZON = 12, 48, 48
HISTORICAL_CUTOFF, CURRENT_CUTOFF = 156, 204
HISTORICAL_ANCHORS = (60, 72, 84, 96, 108)
CURRENT_ANCHORS = (108, 120, 132, 144, 156)
WORKFLOWS = ("W_rowblock", "W_curation", "W_temporal_origin")
BLOCKS = ((0, 12), (12, 24), (24, 36), (36, 48))
SKILL_PATH = "artifacts/functional/e2/historical_policy_episode_workflow_capability.json"
PLAN_PATH = "artifacts/functional/e2/tourism_workflow_executor_development_plan.json"
REPORT_PATH = "artifacts/functional/e2/tourism_workflow_executor_development_report.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _panel(root: Path) -> tuple[list[Any], list[Any], list[Any], dict[str, Any]]:
    rows = sorted(
        (
            row
            for row in read_registry_jsonl(
                root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
            )
            if row.dataset_id == DATASET
            and row.entity_id.startswith("T")
            and 3 <= int(row.entity_id[1:]) <= 20
        ),
        key=lambda row: int(row.entity_id[1:]),
    )
    if [row.entity_id for row in rows] != [f"T{index}" for index in range(3, 21)]:
        raise ValueError("the frozen numeric T3..T20 roster changed")
    wanted = {row.series_uid for row in rows}
    slots = {}
    for record_path in (root / "data/benchmark_v0_2/clean_base").glob("*/record.json"):
        record = _read(record_path)
        uid = str(record.get("series_uid", ""))
        if uid in wanted:
            slots[uid] = record_path.parent
    if set(slots) != wanted:
        raise ValueError("a frozen Tourism Monthly record is unavailable")
    arrays = {
        uid: np.load(slot / "values.npy", mmap_mode="r") for uid, slot in slots.items()
    }
    return rows[:12], rows[12:15], rows[15:18], arrays


def _center_scale(values: Any) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - center)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(array))
    if not np.isfinite(array).all() or not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("non-finite or scale-floor context")
    return center, scale


def _surface(
    train: list[Any],
    evaluation: list[Any],
    arrays: dict[str, Any],
    *,
    cutoff: int,
    anchors: tuple[int, ...],
    period: int = PERIOD,
    context_length: int = CONTEXT,
    horizon: int = HORIZON,
) -> dict[str, Any]:
    x_train, y_train = [], []
    for anchor in anchors:
        if anchor + horizon > cutoff:
            raise ValueError("training target crosses the visible cutoff")
        for row in train:
            visible = np.asarray(arrays[row.series_uid][:cutoff], dtype=np.float64)
            context = visible[anchor - context_length : anchor]
            target = visible[anchor : anchor + horizon]
            center, scale = _center_scale(context)
            if context.shape != (context_length,) or target.shape != (horizon,):
                raise ValueError("invalid training geometry")
            x_train.append(
                np.concatenate(((context - center) / scale, np.zeros(context_length)))
            )
            y_train.append((target - center) / scale)
    x_eval, metadata = [], []
    origin = cutoff
    for row in evaluation:
        visible = np.asarray(arrays[row.series_uid][:cutoff], dtype=np.float64)
        context = visible[origin - context_length : origin]
        center, scale = _center_scale(context)
        x_eval.append(
            np.concatenate(((context - center) / scale, np.zeros(context_length)))
        )
        metadata.append(
            {
                "entity_id": row.entity_id,
                "center": center,
                "scale": scale,
                "seasonal_scale": seasonal_scale(
                    visible[:origin],
                    np.isfinite(visible[:origin]),
                    period=period,
                    min_pairs=24,
                ),
            }
        )
    reference = _ridge_reference_and_removal_predictions(
        np,
        x_train=np.asarray(x_train),
        targets=np.asarray(y_train),
        x_eval=np.asarray(x_eval),
        candidate_rows=tuple(range(len(x_train))),
        target_block=(0, horizon),
        alpha=RIDGE_ALPHA,
    )
    return {"reference": reference, "metadata": metadata, "origin": origin}


def _score(
    prediction: Any,
    metadata: list[dict[str, Any]],
    truth: Any,
    indices: tuple[int, ...],
) -> np.ndarray:
    normalized = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    losses = []
    for truth_index, prediction_index in enumerate(indices):
        row = metadata[prediction_index]
        raw = normalized[prediction_index] * float(row["scale"]) + float(row["center"])
        losses.append(smase(actual[truth_index], raw, scale=float(row["seasonal_scale"])))
    return np.asarray(losses, dtype=np.float64)


def _workflows(
    surface: dict[str, Any],
    train: list[Any],
    anchors: tuple[int, ...],
    truth: Any,
    indices: tuple[int, ...],
    *,
    horizon: int = HORIZON,
    blocks: tuple[tuple[int, int], ...] = BLOCKS,
) -> dict[str, dict[str, Any]]:
    reference = surface["reference"]

    def score_support(prediction: Any) -> np.ndarray:
        return _score(prediction, surface["metadata"], truth, indices)

    def group_predict(
        selected_rows: tuple[int, ...],
        target_block: tuple[int, int],
        removal_strength: float,
    ) -> dict[str, Any]:
        return _group_removal_predictions(
            np,
            reference=reference,
            selected_local_indices=selected_rows,
            target_block=target_block,
            removal_strength=removal_strength,
        )

    rowblock = execute_rowblock_support_only(
        reference, blocks, score_support=score_support, group_predict=group_predict
    )
    doses = (
        {"action_id": "ATTENUATE", "removal_strength": 0.75},
        {"action_id": "EXCLUDE", "removal_strength": 1.0},
    )
    donor_groups = [
        {
            "group_id": row.entity_id,
            "selected_rows": tuple(
                anchor_index * len(train) + donor_index
                for anchor_index in range(len(anchors))
            ),
        }
        for donor_index, row in enumerate(train)
    ]
    curation = execute_whole_group_curation_support_only(
        reference,
        donor_groups,
        doses,
        target_block=(0, horizon),
        score_support=score_support,
        group_predict=group_predict,
        candidate_tiebreak=lambda row: (
            str(row["action_id"]) == "EXCLUDE",
            str(row["group_id"]),
        ),
    )
    origin_groups = [
        {
            "group_id": f"origin_{anchor:03d}",
            "selected_rows": tuple(
                range(anchor_index * len(train), (anchor_index + 1) * len(train))
            ),
        }
        for anchor_index, anchor in enumerate(anchors)
    ]
    temporal = execute_whole_group_curation_support_only(
        reference,
        origin_groups,
        doses,
        target_block=(0, horizon),
        score_support=score_support,
        group_predict=group_predict,
        candidate_tiebreak=lambda row: (
            str(row["action_id"]) == "EXCLUDE",
            str(row["group_id"]),
        ),
    )
    return {"W_rowblock": rowblock, "W_curation": curation, "W_temporal_origin": temporal}


def plan(root: Path) -> dict[str, Any]:
    train, support, holdout, arrays = _panel(root)
    skill = read_active_skill_cards([_read(root / SKILL_PATH)])[0]
    evaluation = [*support, *holdout]
    historical = _surface(
        train, evaluation, arrays, cutoff=HISTORICAL_CUTOFF, anchors=HISTORICAL_ANCHORS
    )
    historical_truth = np.asarray(
        [
            arrays[row.series_uid][historical["origin"] : CURRENT_CUTOFF]
            for row in evaluation
        ],
        dtype=np.float64,
    )
    historical_workflows = _workflows(
        historical, train, HISTORICAL_ANCHORS, historical_truth, tuple(range(6))
    )
    current = _surface(
        train, evaluation, arrays, cutoff=CURRENT_CUTOFF, anchors=CURRENT_ANCHORS
    )
    support_truth = np.asarray(
        [arrays[row.series_uid][CURRENT_CUTOFF : CURRENT_CUTOFF + HORIZON] for row in support],
        dtype=np.float64,
    )
    current_workflows = _workflows(
        current, train, CURRENT_ANCHORS, support_truth, (0, 1, 2)
    )
    historical_episode = {
        "workflows": {
            key: {"workflow_id": key, "support_gain": float(value["support_gain"])}
            for key, value in historical_workflows.items()
        }
    }
    support_only = {
        key: {"support_gain": float(value["support_gain"])}
        for key, value in current_workflows.items()
    }
    a5 = plan_skill_card_support_only(
        skill, lambda key: support_only[key], historical_episode=historical_episode
    )
    a3 = [
        plan_support_only(
            WORKFLOWS,
            order,
            lambda key: support_only[key],
            control="stop_on_first_positive",
        )
        for order in permutations(WORKFLOWS)
    ]
    return {
        "experiment_id": "E2.87-tourism-workflow-executor-development",
        "dataset": DATASET,
        "data_role": "confirmed_exposed / Support-A-only / development",
        "roster": {
            "train": [row.entity_id for row in train],
            "support": [row.entity_id for row in support],
            "development_holdout": [row.entity_id for row in holdout],
        },
        "geometry": {
            "period": PERIOD,
            "context": CONTEXT,
            "horizon": HORIZON,
            "historical_cutoff": HISTORICAL_CUTOFF,
            "historical_origin": HISTORICAL_CUTOFF,
            "historical_outcome_end": CURRENT_CUTOFF,
            "current_cutoff": CURRENT_CUTOFF,
        },
        "skill": {"capability_id": skill["capability_id"], "status": skill["status"]},
        "historical_workflow_gains": {
            key: float(value["support_gain"]) for key, value in historical_workflows.items()
        },
        "current_support_workflows": {
            key: {
                "decision": value["decision"],
                "support_gain": float(value["support_gain"]),
                "binding": value.get("bound_action", value.get("bound_groups")),
            }
            for key, value in current_workflows.items()
        },
        "A5_support_plan": a5,
        "A3_support_plans": a3,
        "development_holdout_predictions": {
            "IDENTITY": np.asarray(current["reference"]["baseline_prediction"])[3:].tolist(),
            **{
                key: np.asarray(value["prediction"])[3:].tolist()
                for key, value in current_workflows.items()
            },
        },
        "development_holdout_normalization": current["metadata"][3:],
        "information_boundary": {
            "holdout_future_read": False,
            "holdout_truth_or_gain_stored": False,
            "historical_outcomes_end_by_current_cutoff": True,
            "storage_level_sealing": False,
            "boundary": "deterministic slice boundary over memory-mapped arrays",
            "repair_truth_used": False,
        },
        "compute": {
            "ridge_reference_solve_count": 2,
            "grouped_small_matrix_solve_count": 12,
            "llm_api_call_count": 0,
        },
    }


def evaluate(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    _, _, holdout, arrays = _panel(root)
    truth = np.asarray(
        [arrays[row.series_uid][CURRENT_CUTOFF : CURRENT_CUTOFF + HORIZON] for row in holdout],
        dtype=np.float64,
    )
    metadata = frozen["development_holdout_normalization"]
    predictions = frozen["development_holdout_predictions"]

    def losses(key: str) -> np.ndarray:
        normalized = np.asarray(predictions[key], dtype=np.float64)
        return np.asarray(
            [
                smase(
                    truth[index],
                    normalized[index] * float(metadata[index]["scale"])
                    + float(metadata[index]["center"]),
                    scale=float(metadata[index]["seasonal_scale"]),
                )
                for index in range(3)
            ]
        )

    baseline = losses("IDENTITY")
    delayed = {
        key: float(np.mean(baseline - losses(key))) for key in WORKFLOWS
    }
    a5 = attach_delayed_outcomes(frozen["A5_support_plan"], delayed)
    a3_rows = [attach_delayed_outcomes(row, delayed) for row in frozen["A3_support_plans"]]
    a3_curve = [
        {
            "budget": budget,
            "fixed_query_gain": statistics.fmean(
                float(row["adaptation_curve"][budget]["fixed_query_gain"])
                for row in a3_rows
            ),
            "abstention_probability": statistics.fmean(
                float(row["adaptation_curve"][budget]["abstained"]) for row in a3_rows
            ),
        }
        for budget in range(4)
    ]
    a3_auc = policy_adaptation_auc(a3_curve)
    a5_auc = float(a5["adaptation_auc"])
    harm = sum(float(row["fixed_query_gain"]) < 0.0 for row in a5["adaptation_curve"][1:])
    nonidentity = any(
        row["selected_workflow"] != "IDENTITY" for row in a5["adaptation_curve"][1:]
    )
    passed = bool(a5_auc > a3_auc and harm == 0 and nonidentity)
    return {
        **{key: value for key, value in frozen.items() if "prediction" not in key},
        "information_boundary": {
            **frozen["information_boundary"],
            "holdout_future_read": True,
            "holdout_future_read_phase": "evaluate_only",
            "holdout_truth_or_gain_stored": True,
            "holdout_truth_stored": False,
            "holdout_gain_stored": True,
        },
        "development_evaluator": {
            "workflow_gains": delayed,
            "A5_curve": a5["adaptation_curve"],
            "A5_adaptation_auc": a5_auc,
            "A3_order_average_curve": a3_curve,
            "A3_adaptation_auc": a3_auc,
            "A5_minus_A3": a5_auc - a3_auc,
        },
        "gate": {
            "A5_strictly_above_A3": a5_auc > a3_auc,
            "A5_harm_zero": harm == 0,
            "A5_nonidentity": nonidentity,
            "passed": passed,
        },
        "outcome_exposure": "EXPOSED_ONCE_DEVELOPMENT_EVALUATION",
        "capability_or_memory_written": False,
        "verdict": (
            "TOURISM_WORKFLOW_EXECUTOR_DEVELOPMENT_PASS"
            if passed
            else "TOURISM_WORKFLOW_EXECUTOR_DEVELOPMENT_FAIL"
        ),
        "claim_limit": (
            "Confirmed-exposed Support-A development evidence only; not fresh, unseen, "
            "Target Query, Promotion, or Memory evidence."
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
