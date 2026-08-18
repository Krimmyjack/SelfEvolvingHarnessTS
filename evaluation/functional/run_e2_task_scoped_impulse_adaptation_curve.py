"""Run an outcome-sealed controlled A3/A4/A5 adaptation curve.

W45-W46 compiled one task-scoped bound-repair Skill from Traffic/FRED and
transferred it to NN5/METR.  This runner keeps that Observer, Program and
Consumer frozen, binds the decision to the canonical TaskContext, and evaluates
equal-feedback adaptation on two local data sources unused by that experiment
family.  The injected task worlds are a mechanism control, not a natural defect
or paper-fresh Capability claim.
"""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile


SCHEMA_VERSION = "e2-task-scoped-impulse-adaptation-curve/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_task_scoped_impulse_adaptation_curve_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_scoped_impulse_adaptation_curve_report.json"
)
GEFCOM_PATH = "data/benchmark_v0/incoming/GEFCom2014.zip"
OPSD_PATH = "data/benchmark_v0/incoming/opsd-time_series-2020-10-06.zip"
GEFCOM_INNER = "GEFCom2014 Data/GEFCom2014-L_V2.zip"
GEFCOM_CSV = "Load/Task 1/L1-train.csv"
OPSD_CSV = "opsd-time_series-2020-10-06/time_series_60min_singleindex.csv"
SERIES_LENGTH = 1024
TRAIN_COUNT = 12
SUPPORT_COUNT = 4
QUERY_COUNT = 4
TRAIN_STOP = 928
TARGET_DATASETS = ("gefcom2014_weather", "opsd_hourly_load")
WORLDS = ("coarse_pattern", "event_evidence")
FEEDBACK_ORDERS = (
    ("coarse_pattern", "event_evidence"),
    ("event_evidence", "coarse_pattern"),
)
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_task_scoped_impulse_skill_transfer_report.json"
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _first_dense_start(np: Any, values: Any) -> int | None:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < SERIES_LENGTH:
        return None
    finite = np.isfinite(array).astype(np.int64)
    counts = np.convolve(finite, np.ones(SERIES_LENGTH, dtype=np.int64), mode="valid")
    matches = np.flatnonzero(counts == SERIES_LENGTH)
    return int(matches[0]) if matches.size else None


def _gefcom_frame(pd: Any, path: Path) -> Any:
    with ZipFile(path) as outer:
        nested = outer.read(GEFCOM_INNER)
    with ZipFile(BytesIO(nested)) as inner:
        return pd.read_csv(inner.open(GEFCOM_CSV))


def _opsd_frame(pd: Any, path: Path) -> Any:
    with ZipFile(path) as archive:
        header = pd.read_csv(archive.open(OPSD_CSV), nrows=0)
        columns = sorted(
            column
            for column in header.columns
            if column.endswith("_load_actual_entsoe_transparency")
        )
        return pd.read_csv(archive.open(OPSD_CSV), usecols=columns)


def _eligible_columns(np: Any, frame: Any, candidates: list[str]) -> list[dict[str, Any]]:
    eligible: list[dict[str, Any]] = []
    for column in candidates:
        start = _first_dense_start(np, frame[column].to_numpy(dtype="float64"))
        if start is not None:
            eligible.append({"column": column, "start": start})
    return eligible


def _assign_cohorts(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(selected) != TRAIN_COUNT + SUPPORT_COUNT + QUERY_COUNT:
        raise ValueError("controlled roster must contain exactly 20 series")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        if index < TRAIN_COUNT:
            cohort = "train"
        elif index < TRAIN_COUNT + SUPPORT_COUNT:
            cohort = "support"
        else:
            cohort = "query"
        rows.append({**item, "cohort": cohort})
    return rows


def build_plan(root: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    gefcom_path = root / GEFCOM_PATH
    opsd_path = root / OPSD_PATH
    if not gefcom_path.is_file() or not opsd_path.is_file():
        raise FileNotFoundError("W47 local incoming ZIP inputs are unavailable")

    gefcom = _gefcom_frame(pd, gefcom_path)
    weather_columns = sorted(
        (column for column in gefcom.columns if column.startswith("w") and column[1:].isdigit()),
        key=lambda column: int(column[1:]),
    )
    gefcom_eligible = _eligible_columns(np, gefcom, weather_columns)

    opsd = _opsd_frame(pd, opsd_path)
    opsd_columns = sorted(str(column) for column in opsd.columns)
    opsd_eligible = _eligible_columns(np, opsd, opsd_columns)
    if len(gefcom_eligible) < 20 or len(opsd_eligible) < 20:
        raise ValueError("fewer than 20 finite series are available for a target base")

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "context-only roster freeze before controlled target outcomes",
        "selection_used_program_or_consumer_outcome": False,
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "SEALED",
        "dataset_rosters": {
            "gefcom2014_weather": {
                "input_path": GEFCOM_PATH,
                "members": [GEFCOM_INNER, GEFCOM_CSV],
                "candidate_count": len(weather_columns),
                "eligible_count": len(gefcom_eligible),
                "selection_rule": "numeric weather-column order; first 20 with earliest 1024-point finite run",
                "series": _assign_cohorts(gefcom_eligible[:20]),
            },
            "opsd_hourly_load": {
                "input_path": OPSD_PATH,
                "members": [OPSD_CSV],
                "candidate_count": len(opsd_columns),
                "eligible_count": len(opsd_eligible),
                "selection_rule": "lexicographic load-column order; first 20 with earliest 1024-point finite run",
                "series": _assign_cohorts(opsd_eligible[:20]),
            },
        },
        "cohort_geometry": {
            "series_length": SERIES_LENGTH,
            "train_series": TRAIN_COUNT,
            "support_series": SUPPORT_COUNT,
            "query_series": QUERY_COUNT,
        },
        "method_frozen_before_outcome": {
            "observer": "class-conditioned-impulse-topology/1",
            "program": "bound-local-median-repair/1",
            "consumer": "nearest-centroid-tie-safe/1",
            "task_contracts": [
                "classification-global-coarse-quality-v1",
                "classification-local-event-quality-v1",
            ],
            "target_feedback_budgets": [0, 1, 2],
            "feedback_orders": [list(order) for order in FEEDBACK_ORDERS],
        },
        "claim_limit": (
            "The data sources were unused by W42-W46 and their Program outcomes are sealed, "
            "but parsing feasibility exposed instances; this is not paper-fresh natural evidence."
        ),
        "original_uci_target_query_opened": False,
    }


def _load_planned_values(np: Any, pd: Any, root: Path, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    frames = {
        "gefcom2014_weather": _gefcom_frame(pd, root / GEFCOM_PATH),
        "opsd_hourly_load": _opsd_frame(pd, root / OPSD_PATH),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for dataset_id in TARGET_DATASETS:
        rows = plan["dataset_rosters"][dataset_id]["series"]
        if [row["cohort"] for row in rows].count("train") != TRAIN_COUNT:
            raise ValueError(f"training cohort changed: {dataset_id}")
        if [row["cohort"] for row in rows].count("support") != SUPPORT_COUNT:
            raise ValueError(f"support cohort changed: {dataset_id}")
        if [row["cohort"] for row in rows].count("query") != QUERY_COUNT:
            raise ValueError(f"query cohort changed: {dataset_id}")
        frame = frames[dataset_id]
        loaded[dataset_id] = {}
        for row in rows:
            column = str(row["column"])
            start = int(row["start"])
            values = np.asarray(
                frame[column].to_numpy(dtype="float64")[start : start + SERIES_LENGTH],
                dtype=np.float64,
            )
            if values.shape != (SERIES_LENGTH,) or not np.isfinite(values).all():
                raise ValueError(f"planned finite window changed: {dataset_id}/{column}")
            loaded[dataset_id][column] = values
    return loaded


def _fit_dual_readout(
    np: Any,
    NearestCentroid: Any,
    classifier_features: Any,
    train_inputs: Any,
    train_labels: Any,
    support_inputs: Any,
    support_labels: Any,
    query_inputs: Any,
    query_labels: Any,
) -> dict[str, Any]:
    model = NearestCentroid(metric="euclidean", shrink_threshold=None)
    model.fit(classifier_features(np, train_inputs), train_labels)
    support_predictions = model.predict(classifier_features(np, support_inputs))
    query_predictions = model.predict(classifier_features(np, query_inputs))
    return {
        "support_accuracy": float(np.mean(support_predictions == support_labels)),
        "query_accuracy": float(np.mean(query_predictions == query_labels)),
        "centroid_max_abs_difference": float(
            np.max(np.abs(np.asarray(model.centroids_[0]) - np.asarray(model.centroids_[1])))
        ),
    }


def _context_decision(task_context: Any) -> str:
    harms = set(task_context.quality_contract.harms)
    if "event_erasure" in harms:
        return "ABSTAIN_KEEP_INCUMBENT"
    return "EXECUTE_BOUND_REPAIR"


def _policy_result(decisions: dict[str, str], worlds: dict[str, Any], split: str) -> dict[str, float]:
    accuracies: list[float] = []
    harms: list[float] = []
    for world in WORLDS:
        record = worlds[world]
        incumbent = float(record["incumbent"][f"{split}_accuracy"])
        action = float(record["action"][f"{split}_accuracy"])
        selected = action if decisions[world] == "EXECUTE_BOUND_REPAIR" else incumbent
        accuracies.append(selected)
        harms.append(max(0.0, incumbent - selected))
    return {
        "utility": sum(accuracies) / len(accuracies),
        "harm": sum(harms) / len(harms),
        "event_harm": harms[WORLDS.index("event_evidence")],
    }


def _adaptation_curve(
    *,
    initial: dict[str, str],
    order: tuple[str, str],
    worlds: dict[str, Any],
) -> list[dict[str, Any]]:
    state = dict(initial)
    curve: list[dict[str, Any]] = []
    for budget in range(3):
        result = _policy_result(state, worlds, "query")
        curve.append({"budget": budget, "decisions": dict(state), **result})
        if budget < 2:
            observed_world = order[budget]
            evidence = worlds[observed_world]
            support_gain = float(evidence["forced_support_gain"])
            state[observed_world] = (
                "EXECUTE_BOUND_REPAIR"
                if support_gain > 0.0
                else "ABSTAIN_KEEP_INCUMBENT"
            )
    return curve


def _average_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    averaged: list[dict[str, float]] = []
    for budget in range(3):
        averaged.append(
            {
                "budget": budget,
                "utility": sum(curve[budget]["utility"] for curve in curves) / len(curves),
                "harm": sum(curve[budget]["harm"] for curve in curves) / len(curves),
                "event_harm": sum(curve[budget]["event_harm"] for curve in curves) / len(curves),
            }
        )
    return averaged


def _adapt_auc(curve: list[dict[str, Any]]) -> float:
    return sum(float(point["utility"]) for point in curve) / len(curve)


def evaluate(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestCentroid

    from SelfEvolvingHarnessTS.contracts.task import (
        classification_global_coarse_task_quality_contract_v1,
        classification_local_event_task_quality_contract_v1,
        classification_task_context_v1,
        classification_task_spec_v1,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        _apply_bound_impulse_oracle,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_impulse_repair_control import (
        SPIKE_POSITIONS,
        WINDOW_LENGTH,
        _classifier_features,
        _normalize_background,
        _paired_world,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_scoped_impulse_skill_transfer import (
        _localization_evaluation,
        _observe_class_conditioned_impulse_topology,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
        ANCHORS,
    )

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("W47 plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("W47 outcomes were not sealed at plan time")
    source = _read_object(root / SOURCE_REPORT_PATH)
    if source.get("verdict") != "CONTROLLED_CROSS_BASE_TASK_SCOPED_SKILL_TRANSFER_PASS":
        raise ValueError("W46 source Skill evidence is not eligible")

    task_spec = classification_task_spec_v1(
        downstream_model_class="nearest_centroid_tie_safe"
    )
    contexts = {
        "coarse_pattern": classification_task_context_v1(
            task_spec=task_spec,
            quality_contract=classification_global_coarse_task_quality_contract_v1(),
        ),
        "event_evidence": classification_task_context_v1(
            task_spec=task_spec,
            quality_contract=classification_local_event_task_quality_contract_v1(),
        ),
    }
    source_decisions = {world: _context_decision(contexts[world]) for world in WORLDS}
    if source_decisions != {
        "coarse_pattern": "EXECUTE_BOUND_REPAIR",
        "event_evidence": "ABSTAIN_KEEP_INCUMBENT",
    }:
        raise AssertionError("canonical TaskContext no longer compiles to the W46 Skill")

    values = _load_planned_values(np, pd, root, plan)
    dataset_evidence: list[dict[str, Any]] = []
    fit_count = 0
    for dataset_id in TARGET_DATASETS:
        roster = plan["dataset_rosters"][dataset_id]["series"]
        train_columns = [row["column"] for row in roster if row["cohort"] == "train"]
        support_columns = [row["column"] for row in roster if row["cohort"] == "support"]
        query_columns = [row["column"] for row in roster if row["cohort"] == "query"]

        train_backgrounds = np.asarray(
            [
                _normalize_background(
                    np,
                    values[dataset_id][column][anchor - WINDOW_LENGTH : anchor],
                    identity=f"{dataset_id}/{column}@{anchor}",
                )
                for anchor in ANCHORS
                for column in train_columns
            ],
            dtype=np.float64,
        )
        support_backgrounds = np.asarray(
            [
                _normalize_background(
                    np,
                    values[dataset_id][column][TRAIN_STOP - WINDOW_LENGTH : TRAIN_STOP],
                    identity=f"{dataset_id}/{column}@support",
                )
                for column in support_columns
            ],
            dtype=np.float64,
        )
        query_backgrounds = np.asarray(
            [
                _normalize_background(
                    np,
                    values[dataset_id][column][TRAIN_STOP - WINDOW_LENGTH : TRAIN_STOP],
                    identity=f"{dataset_id}/{column}@query",
                )
                for column in query_columns
            ],
            dtype=np.float64,
        )
        if train_backgrounds.shape != (72, WINDOW_LENGTH):
            raise AssertionError(f"training background geometry changed: {dataset_id}")
        if support_backgrounds.shape != (4, WINDOW_LENGTH):
            raise AssertionError(f"support background geometry changed: {dataset_id}")
        if query_backgrounds.shape != (4, WINDOW_LENGTH):
            raise AssertionError(f"query background geometry changed: {dataset_id}")

        world_evidence: dict[str, Any] = {}
        for world in WORLDS:
            train_inputs, train_labels = _paired_world(
                np, train_backgrounds, world=world, training=True
            )
            support_inputs, support_labels = _paired_world(
                np, support_backgrounds, world=world, training=False
            )
            query_inputs, query_labels = _paired_world(
                np, query_backgrounds, world=world, training=False
            )
            observation = _observe_class_conditioned_impulse_topology(
                np, train_inputs, train_labels
            )
            localization = _localization_evaluation(
                [int(node) for node in observation["nodes"]], SPIKE_POSITIONS
            )
            repaired_inputs, modification = _apply_bound_impulse_oracle(
                np,
                train_inputs,
                positions=tuple(int(node) for node in observation["nodes"]),
                window_length=WINDOW_LENGTH,
            )
            incumbent = _fit_dual_readout(
                np,
                NearestCentroid,
                _classifier_features,
                train_inputs,
                train_labels,
                support_inputs,
                support_labels,
                query_inputs,
                query_labels,
            )
            action = _fit_dual_readout(
                np,
                NearestCentroid,
                _classifier_features,
                repaired_inputs,
                train_labels,
                support_inputs,
                support_labels,
                query_inputs,
                query_labels,
            )
            fit_count += 2
            world_evidence[world] = {
                "task_context": contexts[world].to_dict(),
                "compiled_source_decision": source_decisions[world],
                "observation": observation,
                "localization_evaluator": localization,
                "program_modification": modification,
                "incumbent": incumbent,
                "action": action,
                "forced_support_gain": action["support_accuracy"]
                - incumbent["support_accuracy"],
                "forced_query_gain": action["query_accuracy"]
                - incumbent["query_accuracy"],
            }

        blank = {world: "ABSTAIN_KEEP_INCUMBENT" for world in WORLDS}
        a3_orders = [
            _adaptation_curve(initial=blank, order=order, worlds=world_evidence)
            for order in FEEDBACK_ORDERS
        ]
        a5_orders = [
            _adaptation_curve(
                initial=source_decisions, order=order, worlds=world_evidence
            )
            for order in FEEDBACK_ORDERS
        ]
        a3 = _average_curves(a3_orders)
        a5 = _average_curves(a5_orders)
        a4_point = _policy_result(source_decisions, world_evidence, "query")
        a4 = [{"budget": budget, **a4_point} for budget in range(3)]
        unscoped_point = _policy_result(
            {world: "EXECUTE_BOUND_REPAIR" for world in WORLDS},
            world_evidence,
            "query",
        )
        unscoped = [{"budget": budget, **unscoped_point} for budget in range(3)]

        observer_pass = all(
            world_evidence[world]["localization_evaluator"]["exact_precision_recall_pass"]
            for world in WORLDS
        )
        direction_pass = bool(
            world_evidence["coarse_pattern"]["forced_query_gain"] > 0.0
            and world_evidence["event_evidence"]["forced_query_gain"] < 0.0
        )
        a3_auc = _adapt_auc(a3)
        a5_auc = _adapt_auc(a5)
        dataset_pass = bool(
            observer_pass
            and direction_pass
            and a5_auc > a3_auc
            and max(point["event_harm"] for point in a5) == 0.0
            and unscoped_point["event_harm"] > 0.0
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "context_or_decision_used_dataset_id": False,
                "task_worlds": world_evidence,
                "feedback_orders": [list(order) for order in FEEDBACK_ORDERS],
                "A3_target_only": {"per_order": a3_orders, "mean_curve": a3, "adapt_auc": a3_auc},
                "A4_source_only": {"mean_curve": a4, "adapt_auc": _adapt_auc(a4)},
                "A5_source_plus_target": {
                    "per_order": a5_orders,
                    "mean_curve": a5,
                    "adapt_auc": a5_auc,
                },
                "unscoped_source_skill": {
                    "mean_curve": unscoped,
                    "adapt_auc": _adapt_auc(unscoped),
                },
                "A5_minus_A3_adapt_auc": a5_auc - a3_auc,
                "A5_minus_A4_adapt_auc": a5_auc - _adapt_auc(a4),
                "observer_pass": observer_pass,
                "program_direction_pass": direction_pass,
                "dataset_gate_pass": dataset_pass,
            }
        )

    passing = sum(bool(row["dataset_gate_pass"]) for row in dataset_evidence)
    overall_pass = passing == len(TARGET_DATASETS)
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "A canonical TaskContext-conditioned Source Skill improves equal-budget "
            "Target adaptation AUC and avoids event-erasure harm relative to both "
            "target-only initialization and unscoped transfer."
        ),
        "scientific_role": "outcome-sealed controlled target adaptation mechanism",
        "plan_exposure": {
            "context_exposure": plan["context_exposure"],
            "outcome_exposure_before_evaluate": plan["outcome_exposure"],
            "outcome_exposure_after_evaluate": "EXPOSED",
        },
        "source_skill": {
            "source_evidence": "W45 Traffic/FRED and W46 NN5/METR controlled evidence",
            "task_context_decisions": source_decisions,
            "persistent_memory_built": False,
        },
        "dataset_evidence": dataset_evidence,
        "overall": {
            "dataset_count": len(TARGET_DATASETS),
            "passing_dataset_count": passing,
            "all_controlled_target_curves_pass": overall_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "CONTROLLED_TASK_SCOPED_A5_VS_A3_ADAPTATION_PASS"
            if overall_pass
            else "CONTROLLED_TASK_SCOPED_A5_VS_A3_ADAPTATION_FAIL"
        ),
        "controlled_w47_query_outcome_opened_once": True,
        "original_uci_target_query_opened": False,
        "uci_values_read": False,
        "formal_natural_capability_promotion": False,
        "paper_fresh_transfer_claim": False,
        "claim_limit": (
            "Two previously unused local data sources provide outcome-sealed controlled "
            "backgrounds. The task defect is injected and parsing contexts were inspected; "
            "this supports a mechanism, not paper-fresh natural Capability transfer."
        ),
        "next_step": (
            "Freeze the mechanism and seek one natural or separately acquired dataset "
            "Capability using the same Context-Skill-feedback loop."
            if overall_pass
            else "Localize the first failing surface; do not add Memory or tune all components."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output", type=Path)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path)
    evaluate_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "plan":
        payload = build_plan(root)
        output = args.output or root / DEFAULT_PLAN_PATH
    else:
        plan_path = args.plan or root / DEFAULT_PLAN_PATH
        payload = evaluate(root, _read_object(plan_path))
        output = args.output or root / DEFAULT_REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(payload.get("verdict", payload.get("outcome_exposure")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
