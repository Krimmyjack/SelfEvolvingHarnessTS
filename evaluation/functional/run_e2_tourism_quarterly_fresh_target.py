"""Plan/evaluate the admitted Workflow Skill on fresh Tourism Quarterly."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import smase
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
DATASET = "external_monash:tourism_quarterly"
ARCHIVE = "data/tourism_quarterly_dataset.zip"
MEMBER = "tourism_quarterly_dataset.tsf"
MIN_LENGTH = 80
EXPECTED_METADATA = {"frequency": "quarterly", "horizon": "8", "missing": "false"}
PERIOD, CONTEXT, HORIZON = 4, 16, 8
HISTORICAL_CUTOFF, CURRENT_CUTOFF = 56, 64
HISTORICAL_ANCHORS = (32, 36, 40, 44, 48)
CURRENT_ANCHORS = (40, 44, 48, 52, 56)
BLOCKS = ((0, 4), (4, 8))
WORKFLOWS = ("W_rowblock", "W_curation", "W_temporal_origin")
SKILL_PATH = "artifacts/functional/e2/historical_policy_episode_workflow_capability.json"
PLAN_PATH = "artifacts/functional/e2/tourism_quarterly_fresh_target_plan.json"
REPORT_PATH = "artifacts/functional/e2/tourism_quarterly_fresh_target_report.json"
EXPERIMENT_ID = "E2.88-tourism-quarterly-fresh-target"
@dataclass(frozen=True)
class _Row:
    entity_id: str
    series_uid: str
    file_order_index: int
def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value
def _panel(
    root: Path,
    *,
    query_end: int,
    archive: str = ARCHIVE,
    member: str = MEMBER,
    min_length: int = MIN_LENGTH,
    expected_metadata: dict[str, str] = EXPECTED_METADATA,
) -> tuple[list[_Row], list[_Row], list[_Row], dict[str, Any]]:
    if query_end not in (CURRENT_CUTOFF, CURRENT_CUTOFF + HORIZON):
        raise ValueError("query_end must be the plan or evaluate boundary")
    archive_path = root / archive
    headers: dict[str, str] = {}
    selected: list[_Row] = []
    with ZipFile(archive_path) as source, source.open(member) as stream:
        data_started = False
        file_index = 0
        for raw in stream:
            line = raw.decode("cp1252").strip()
            if not data_started:
                if line.startswith("@") and " " in line:
                    key, value = line[1:].split(" ", 1)
                    headers[key.lower()] = value.strip()
                data_started = line.lower() == "@data"
                continue
            if not line:
                continue
            file_index += 1
            fields = line.split(":", 2)
            if len(fields) != 3:
                raise ValueError("unexpected Tourism Quarterly TSF row geometry")
            if fields[2].count(",") + 1 >= min_length:
                selected.append(_Row(fields[0], f"tsf_row_{file_index:03d}", file_index))
                if len(selected) == 18:
                    break
    if any(
        headers.get(key) != value
        for key, value in expected_metadata.items()
        if key != "missing"
    ):
        raise ValueError("Tourism Quarterly metadata changed")
    if headers.get("missing") != expected_metadata.get("missing") or len(selected) != 18:
        raise ValueError("Tourism Quarterly frozen feasibility failed")
    by_index = {row.file_order_index: row for row in selected}
    arrays: dict[str, Any] = {}
    with ZipFile(archive_path) as source, source.open(member) as stream:
        data_started = False
        file_index = 0
        for raw in stream:
            line = raw.decode("cp1252").strip()
            if not data_started:
                data_started = line.lower() == "@data"
                continue
            if not line:
                continue
            file_index += 1
            row = by_index.get(file_index)
            if row is None:
                continue
            payload = line.split(":", 2)[2]
            selected_index = selected.index(row)
            limit = (
                CURRENT_CUTOFF
                if selected_index < 12
                else CURRENT_CUTOFF + HORIZON
                if selected_index < 15
                else query_end
            )
            values = np.fromstring(payload, dtype=np.float64, count=limit, sep=",")
            if values.shape != (limit,) or not np.isfinite(values).all():
                raise ValueError(f"invalid frozen row: {row.entity_id}")
            arrays[row.series_uid] = values
            if len(arrays) == 18:
                break
    if len(arrays) != 18:
        raise ValueError("Tourism Quarterly roster became unavailable")
    return selected[:12], selected[12:15], selected[15:], arrays
def plan(root: Path) -> dict[str, Any]:
    train, support, query, arrays = _panel(root, query_end=CURRENT_CUTOFF)
    skill = read_active_skill_cards([_read(root / SKILL_PATH)])[0]
    evaluation = [*support, *query]
    surface_args = {"period": PERIOD, "context_length": CONTEXT, "horizon": HORIZON}
    historical = _surface(
        train, evaluation, arrays, cutoff=HISTORICAL_CUTOFF,
        anchors=HISTORICAL_ANCHORS, **surface_args,
    )
    historical_truth = np.asarray(
        [arrays[row.series_uid][HISTORICAL_CUTOFF:CURRENT_CUTOFF] for row in evaluation]
    )
    historical_workflows = _workflows(
        historical, train, HISTORICAL_ANCHORS, historical_truth, tuple(range(6)),
        horizon=HORIZON, blocks=BLOCKS,
    )
    current = _surface(
        train, evaluation, arrays, cutoff=CURRENT_CUTOFF,
        anchors=CURRENT_ANCHORS, **surface_args,
    )
    support_truth = np.asarray(
        [arrays[row.series_uid][CURRENT_CUTOFF:CURRENT_CUTOFF + HORIZON] for row in support]
    )
    current_workflows = _workflows(
        current, train, CURRENT_ANCHORS, support_truth, (0, 1, 2),
        horizon=HORIZON, blocks=BLOCKS,
    )
    historical_episode = {"workflows": {
        key: {"workflow_id": key, "support_gain": float(value["support_gain"])}
        for key, value in historical_workflows.items()
    }}
    support_only = {
        key: {"support_gain": float(value["support_gain"])}
        for key, value in current_workflows.items()
    }
    a5 = plan_skill_card_support_only(
        skill, lambda key: support_only[key], historical_episode=historical_episode
    )
    a3 = [
        plan_support_only(WORKFLOWS, order, lambda key: support_only[key],
                          control="stop_on_first_positive")
        for order in permutations(WORKFLOWS)
    ]
    roster = lambda rows: [
        {"series_name": row.entity_id, "file_order_index_1based": row.file_order_index}
        for row in rows
    ]
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset": DATASET,
        "data_role": (
            "new external Target not used to develop the Skill; metadata/length "
            "feasibility aggregate seen; Query outcome sealed until evaluate"
        ),
        "roster": {"train": roster(train), "support": roster(support), "query": roster(query)},
        "geometry": {
            "period": PERIOD, "context": CONTEXT, "horizon": HORIZON,
            "historical_cutoff": HISTORICAL_CUTOFF, "current_cutoff": CURRENT_CUTOFF,
            "historical_anchors": list(HISTORICAL_ANCHORS),
            "current_anchors": list(CURRENT_ANCHORS), "blocks": [list(row) for row in BLOCKS],
        },
        "skill": {"capability_id": skill["capability_id"], "status": skill["status"]},
        "historical_workflow_gains": {
            key: float(value["support_gain"]) for key, value in historical_workflows.items()
        },
        "current_support_workflows": {
            key: {"decision": value["decision"], "support_gain": float(value["support_gain"]),
                  "binding": value.get("bound_action", value.get("bound_groups"))}
            for key, value in current_workflows.items()
        },
        "A5_support_plan": a5,
        "A3_support_plans": a3,
        "query_predictions": {
            "IDENTITY": np.asarray(current["reference"]["baseline_prediction"])[3:].tolist(),
            **{key: np.asarray(value["prediction"])[3:].tolist()
               for key, value in current_workflows.items()},
        },
        "query_normalization": current["metadata"][3:],
        "information_boundary": {
            "pre_plan_feasibility_context_exposure": "AGGREGATE_SEEN",
            "plan_context_exposure": "INSTANCE_SEEN_AFTER_SKILL_PROMOTION",
            "query_outcome_exposure": "LOGICALLY_SEALED_NOT_NUMERICALLY_MATERIALIZED",
            "historical_truth_half_open": [HISTORICAL_CUTOFF, CURRENT_CUTOFF],
            "current_support_truth_half_open": [CURRENT_CUTOFF, CURRENT_CUTOFF + HORIZON],
            "query_numeric_parse_end_exclusive": CURRENT_CUTOFF,
            "query_numeric_parser": "np.fromstring(count=64)",
            "query_truth_stored": False,
            "query_gain_stored": False,
            "repair_truth_used": False,
            "storage_level_sealing": False,
        },
        "compute": {"ridge_reference_solve_count": 2,
                    "grouped_small_matrix_solve_count": 8, "llm_api_call_count": 0},
    }
def evaluate(root: Path, frozen: dict[str, Any]) -> dict[str, Any]:
    train, support, query, arrays = _panel(root, query_end=CURRENT_CUTOFF + HORIZON)
    roster_names = lambda rows: [row.entity_id for row in rows]
    if frozen.get("experiment_id") != EXPERIMENT_ID or [
        [row["series_name"] for row in frozen["roster"][key]] for key in ("train", "support", "query")
    ] != [roster_names(train), roster_names(support), roster_names(query)]:
        raise ValueError("frozen Tourism Quarterly plan/roster mismatch")
    truth = np.asarray(
        [arrays[row.series_uid][CURRENT_CUTOFF:CURRENT_CUTOFF + HORIZON] for row in query]
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
    passed = bool(a5_auc > a3_auc and harm == 0 and nonidentity)
    kept = {key: value for key, value in frozen.items() if "prediction" not in key}
    return {
        **kept,
        "information_boundary": {**frozen["information_boundary"],
                                 "query_outcome_exposure": "EXPOSED_ONCE_IN_EVALUATE",
                                 "query_numeric_parse_end_exclusive": CURRENT_CUTOFF + HORIZON,
                                 "query_numeric_parser": "np.fromstring(count=72)",
                                 "query_truth_stored": False, "query_gain_stored": True},
        "delayed_evaluator": {
            "workflow_gains": delayed, "A5_curve": a5["adaptation_curve"],
            "A5_adaptation_auc": a5_auc, "A3_order_average_curve": a3_curve,
            "A3_adaptation_auc": a3_auc, "A5_minus_A3": a5_auc - a3_auc,
            "A5_harm_count": harm, "A5_nonidentity": nonidentity,
        },
        "gate": {"A5_strictly_above_A3": a5_auc > a3_auc,
                 "A5_harm_zero": harm == 0, "A5_nonidentity": nonidentity, "passed": passed},
        "capability_or_memory_written": False,
        "verdict": "TOURISM_QUARTERLY_FRESH_TARGET_PASS" if passed
                   else "TOURISM_QUARTERLY_FRESH_TARGET_FAIL",
        "claim_limit": "Fresh dataset-level Target transfer evidence for this frozen roster only.",
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
