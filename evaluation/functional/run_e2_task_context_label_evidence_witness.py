"""Evaluate a non-oracle task-risk Witness on real labelled TS backgrounds.

The same generic classification quality contract is used in both controlled
worlds.  A class-conditioned local impulse is either present only in the fit
cohort (an acquisition artifact) or repeated in fit/support/query (task
evidence).  The decision input contains no world name, Dataset ID, hidden
positions, action outcome, or query statistic.  A frozen cohort-consistency
Witness decides whether the existing bound-median Skill should execute or
abstain before the official UCR test outcome is opened.

This is a controlled defect on real labelled time-series data.  It tests whether
the Harness can derive an instance risk from legal Context rather than receiving
the correct action in TaskContext; it is not a natural-defect promotion claim.
"""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile


SCHEMA_VERSION = "e2-task-context-label-evidence-witness/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_task_context_label_evidence_witness_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_context_label_evidence_witness_report.json"
)
DATA_DIR = "data/ucr_task_context"
DATASETS = ("Coffee", "ECG200", "FordA", "GunPoint")
CONDITIONS = ("fit_only_artifact", "stable_task_event")
SUPPORT_FRACTION = 0.30
MIN_SUPPORT_PER_CLASS = 3
SPIKE_FRACTIONS = (0.08, 0.20, 0.80, 0.92)
SPIKE_AMPLITUDE = 16.0
RIDGE_ALPHA = 1.0
STABLE_RATIO_MIN = 0.50
ARTIFACT_RATIO_MAX = 0.10
STABLE_DIRECTION_MIN = 0.75
MIN_ARTIFACT_MACRO_GAIN = 0.05
MIN_UNSCOPED_EVENT_HARM = 0.05


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_split(np: Any, archive_path: Path, dataset: str, split: str) -> tuple[Any, Any]:
    with ZipFile(archive_path) as archive:
        raw = archive.read(f"{dataset}_{split}.txt")
    table = np.loadtxt(BytesIO(raw), dtype=np.float64)
    if table.ndim != 2 or table.shape[1] < 8 or not np.isfinite(table).all():
        raise ValueError(f"invalid UCR table: {dataset}/{split}")
    raw_labels = table[:, 0]
    label_values = sorted(float(value) for value in np.unique(raw_labels))
    if len(label_values) != 2:
        raise ValueError(f"W48 requires a binary dataset: {dataset}")
    labels = np.asarray([label_values.index(float(value)) for value in raw_labels], dtype=np.int64)
    values = np.asarray(table[:, 1:], dtype=np.float64)
    scale = np.std(values, axis=1, keepdims=True)
    if bool(np.any(scale <= 1e-8)):
        raise ValueError(f"degenerate UCR row: {dataset}/{split}")
    values = (values - np.mean(values, axis=1, keepdims=True)) / scale
    return values, labels


def _split_fit_support(np: Any, labels: Any) -> tuple[Any, Any]:
    fit: list[int] = []
    support: list[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        count = max(MIN_SUPPORT_PER_CLASS, int(round(SUPPORT_FRACTION * len(indices))))
        if count >= len(indices):
            raise ValueError("insufficient examples for deterministic fit/support split")
        offsets = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
        selected = {int(indices[offset]) for offset in offsets}
        support.extend(sorted(selected))
        fit.extend(int(index) for index in indices if int(index) not in selected)
    return np.asarray(sorted(fit), dtype=np.int64), np.asarray(sorted(support), dtype=np.int64)


def _bound_positions(length: int) -> tuple[int, ...]:
    positions = tuple(sorted({int(round(fraction * (length - 1))) for fraction in SPIKE_FRACTIONS}))
    if len(positions) != len(SPIKE_FRACTIONS) or min(positions) < 3 or max(positions) >= length - 3:
        raise ValueError("relative impulse geometry is invalid")
    return positions


def _inject(np: Any, values: Any, labels: Any, positions: tuple[int, ...]) -> Any:
    template = np.zeros(values.shape[1], dtype=np.float64)
    for index, position in enumerate(positions):
        template[position] = (1.0 if index % 2 == 0 else -1.0) * SPIKE_AMPLITUDE
    signs = np.where(labels[:, None] == 1, 1.0, -1.0)
    return np.asarray(values, dtype=np.float64) + signs * template


def _features(np: Any, values: Any) -> Any:
    matrix = np.asarray(values, dtype=np.float64)
    return np.concatenate((matrix, np.diff(matrix, axis=1)), axis=1)


def _centroid_residual(np: Any, values: Any, labels: Any, rolling_median: Any) -> Any:
    delta = np.mean(values[labels == 1], axis=0) - np.mean(values[labels == 0], axis=0)
    return delta - rolling_median(np, delta)


def _build_witness(
    np: Any,
    fit_values: Any,
    fit_labels: Any,
    support_values: Any,
    support_labels: Any,
    nodes: tuple[int, ...],
    rolling_median: Any,
) -> dict[str, Any]:
    fit_residual = _centroid_residual(np, fit_values, fit_labels, rolling_median)
    support_residual = _centroid_residual(
        np, support_values, support_labels, rolling_median
    )
    fit_at_nodes = fit_residual[list(nodes)]
    support_at_nodes = support_residual[list(nodes)]
    fit_strength = float(np.median(np.abs(fit_at_nodes)))
    support_strength = float(np.median(np.abs(support_at_nodes)))
    ratio = support_strength / (fit_strength + 1e-8)
    direction_alignment = float(
        np.mean(np.sign(fit_at_nodes) == np.sign(support_at_nodes))
    )
    return {
        "witness_id": "cross-cohort-local-label-evidence/1",
        "inputs_used": [
            "fit_inputs",
            "fit_labels",
            "support_inputs",
            "support_labels",
            "candidate_program_nodes",
        ],
        "query_values_or_labels_used": False,
        "dataset_id_used": False,
        "program_outcome_used": False,
        "node_count": len(nodes),
        "normalized_node_positions": [
            float(node / (fit_values.shape[1] - 1)) for node in nodes
        ],
        "fit_node_median_strength": fit_strength,
        "support_node_median_strength": support_strength,
        "support_to_fit_strength_ratio": ratio,
        "direction_alignment": direction_alignment,
    }


def _decision_input(task_context: Any, witness: dict[str, Any]) -> str:
    quality = task_context.quality_contract
    text = "\n".join(
        (
            "TASK_CONTEXT",
            f"task_type={task_context.task_spec.task_type}",
            f"metric={task_context.task_spec.metric.name}",
            f"quality_objective={quality.objective}",
            f"preserve={','.join(quality.preserve)}",
            f"generic_harms={','.join(quality.harms)}",
            "CANDIDATE_PROGRAM",
            "program=bound_local_median_repair",
            "effect=replace_each_candidate_node_with_center_excluded_local_median",
            "APPLICABILITY_WITNESS",
            f"node_count={witness['node_count']}",
            "normalized_node_positions="
            + ",".join(f"{value:.6f}" for value in witness["normalized_node_positions"]),
            "support_to_fit_strength_ratio="
            + f"{witness['support_to_fit_strength_ratio']:.6f}",
            f"direction_alignment={witness['direction_alignment']:.6f}",
            "available_decisions=EXECUTE|ABSTAIN|REQUEST_MORE_OBSERVATION",
        )
    )
    forbidden = (
        "fit_only_artifact",
        "stable_task_event",
        "correct_action",
        "query_accuracy",
        "program_gain",
        "dataset_id",
    ) + tuple(dataset.lower() for dataset in DATASETS)
    lowered = text.lower()
    leaked = [token for token in forbidden if token.lower() in lowered]
    if leaked:
        raise AssertionError(f"decision input leaks private answer fields: {leaked}")
    return text


def _compile_decision(witness: dict[str, Any]) -> tuple[str, list[str]]:
    ratio = float(witness["support_to_fit_strength_ratio"])
    alignment = float(witness["direction_alignment"])
    if ratio >= STABLE_RATIO_MIN and alignment >= STABLE_DIRECTION_MIN:
        return "ABSTAIN_KEEP_INCUMBENT", ["local_label_evidence_repeats_across_cohorts"]
    if ratio <= ARTIFACT_RATIO_MAX:
        return "EXECUTE_BOUND_REPAIR", ["localized_fit_signal_not_reproduced_in_support"]
    return "REQUEST_MORE_OBSERVATION", ["cross_cohort_evidence_unresolved"]


def _fit_readout(
    np: Any,
    RidgeClassifier: Any,
    train_values: Any,
    train_labels: Any,
    support_values: Any,
    support_labels: Any,
    query_values: Any,
    query_labels: Any,
) -> dict[str, float]:
    model = RidgeClassifier(alpha=RIDGE_ALPHA)
    model.fit(_features(np, train_values), train_labels)
    return {
        "support_accuracy": float(
            np.mean(model.predict(_features(np, support_values)) == support_labels)
        ),
        "query_accuracy": float(
            np.mean(model.predict(_features(np, query_values)) == query_labels)
        ),
    }


def build_plan(root: Path) -> dict[str, Any]:
    import numpy as np

    datasets: list[dict[str, Any]] = []
    for dataset in DATASETS:
        path = root / DATA_DIR / f"{dataset}.zip"
        if not path.is_file():
            raise FileNotFoundError(path)
        train_values, train_labels = _load_split(np, path, dataset, "TRAIN")
        fit_indices, support_indices = _split_fit_support(np, train_labels)
        datasets.append(
            {
                "dataset": dataset,
                "archive": f"{DATA_DIR}/{dataset}.zip",
                "series_length": int(train_values.shape[1]),
                "official_train_count": int(train_values.shape[0]),
                "fit_count": int(fit_indices.size),
                "support_count": int(support_indices.size),
                "class_counts": {
                    str(label): int(np.count_nonzero(train_labels == label))
                    for label in (0, 1)
                },
                "split_rule": "per-class evenly-spaced round(30%) support; remainder fit",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "Cross-cohort repetition of Program-overlapping class evidence distinguishes "
            "a fit-only acquisition artifact from a stable task event without placing the "
            "instance answer in TaskContext."
        ),
        "scientific_role": "controlled real-label task-risk Witness",
        "context_exposure": "AGGREGATE_SEEN",
        "outcome_exposure": "SEALED",
        "datasets": datasets,
        "frozen_method": {
            "same_task_quality_contract_in_both_conditions": True,
            "quality_contract": "classification-local-event-quality-v1",
            "observer": "class-conditioned-impulse-topology/1",
            "program": "bound-local-median-repair/1",
            "consumer": "ridge-raw-plus-difference/1",
            "support_fraction": SUPPORT_FRACTION,
            "spike_fractions": list(SPIKE_FRACTIONS),
            "spike_amplitude": SPIKE_AMPLITUDE,
            "stable_ratio_min": STABLE_RATIO_MIN,
            "artifact_ratio_max": ARTIFACT_RATIO_MAX,
            "stable_direction_min": STABLE_DIRECTION_MIN,
        },
        "selection_used_program_or_consumer_outcome": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "UCR labels and backgrounds are real, but the local acquisition/event mechanism "
            "is injected; this is not natural-defect Capability promotion."
        ),
    }


def evaluate(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.contracts.task import (
        classification_local_event_task_quality_contract_v1,
        classification_task_context_v1,
        classification_task_spec_v1,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        _apply_bound_impulse_oracle,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_scoped_impulse_skill_transfer import (
        _localization_evaluation,
        _observe_class_conditioned_impulse_topology,
        _rolling_median,
    )

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("W48 plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("W48 query outcomes were not sealed at plan time")
    task_context = classification_task_context_v1(
        task_spec=classification_task_spec_v1(
            downstream_model_class="ridge-raw-plus-difference-v1"
        ),
        quality_contract=classification_local_event_task_quality_contract_v1(),
    )

    dataset_rows: list[dict[str, Any]] = []
    fit_count = 0
    for dataset in DATASETS:
        archive = root / DATA_DIR / f"{dataset}.zip"
        train_values, train_labels = _load_split(np, archive, dataset, "TRAIN")
        query_values, query_labels = _load_split(np, archive, dataset, "TEST")
        fit_indices, support_indices = _split_fit_support(np, train_labels)
        base_fit = train_values[fit_indices]
        fit_labels = train_labels[fit_indices]
        base_support = train_values[support_indices]
        support_labels = train_labels[support_indices]
        positions = _bound_positions(train_values.shape[1])
        condition_rows: dict[str, Any] = {}

        for condition in CONDITIONS:
            fit_values = _inject(np, base_fit, fit_labels, positions)
            if condition == "fit_only_artifact":
                support_values = base_support.copy()
                condition_query = query_values.copy()
            else:
                support_values = _inject(np, base_support, support_labels, positions)
                condition_query = _inject(np, query_values, query_labels, positions)

            observation = _observe_class_conditioned_impulse_topology(
                np, fit_values, fit_labels
            )
            nodes = tuple(int(node) for node in observation["nodes"])
            localization = _localization_evaluation(list(nodes), positions)
            repaired, modification = _apply_bound_impulse_oracle(
                np, fit_values, positions=nodes, window_length=train_values.shape[1]
            )
            witness = _build_witness(
                np,
                fit_values,
                fit_labels,
                support_values,
                support_labels,
                nodes,
                _rolling_median,
            )
            decision_input = _decision_input(task_context, witness)
            decision, reasons = _compile_decision(witness)
            incumbent = _fit_readout(
                np,
                RidgeClassifier,
                fit_values,
                fit_labels,
                support_values,
                support_labels,
                condition_query,
                query_labels,
            )
            action = _fit_readout(
                np,
                RidgeClassifier,
                repaired,
                fit_labels,
                support_values,
                support_labels,
                condition_query,
                query_labels,
            )
            fit_count += 2
            selected = action if decision == "EXECUTE_BOUND_REPAIR" else incumbent
            condition_rows[condition] = {
                "task_context": task_context.to_dict(),
                "decision_input_text": decision_input,
                "decision_input_contains_instance_answer": False,
                "observation": observation,
                "private_localization_evaluation_not_used_by_decision": localization,
                "witness": witness,
                "compiled_decision": decision,
                "decision_reasons": reasons,
                "program_modification": modification,
                "incumbent": incumbent,
                "forced_action": action,
                "forced_support_gain": action["support_accuracy"]
                - incumbent["support_accuracy"],
                "forced_query_gain": action["query_accuracy"]
                - incumbent["query_accuracy"],
                "scoped_policy": selected,
                "scoped_query_gain": selected["query_accuracy"]
                - incumbent["query_accuracy"],
                "scoped_query_harm": max(
                    0.0, incumbent["query_accuracy"] - selected["query_accuracy"]
                ),
            }

        artifact = condition_rows["fit_only_artifact"]
        event = condition_rows["stable_task_event"]
        witness_pass = (
            artifact["compiled_decision"] == "EXECUTE_BOUND_REPAIR"
            and event["compiled_decision"] == "ABSTAIN_KEEP_INCUMBENT"
        )
        dataset_rows.append(
            {
                "dataset": dataset,
                "condition_or_decision_used_dataset_id": False,
                "conditions": condition_rows,
                "witness_separation_pass": witness_pass,
            }
        )

    artifact_gains = [
        float(row["conditions"]["fit_only_artifact"]["forced_query_gain"])
        for row in dataset_rows
    ]
    event_harms = [
        max(
            0.0,
            -float(row["conditions"]["stable_task_event"]["forced_query_gain"]),
        )
        for row in dataset_rows
    ]
    scoped_event_harms = [
        float(row["conditions"]["stable_task_event"]["scoped_query_harm"])
        for row in dataset_rows
    ]
    artifact_macro_gain = sum(artifact_gains) / len(artifact_gains)
    event_macro_harm = sum(event_harms) / len(event_harms)
    all_witnesses = all(bool(row["witness_separation_pass"]) for row in dataset_rows)
    gain_pass = artifact_macro_gain >= MIN_ARTIFACT_MACRO_GAIN and sum(
        gain > 0.0 for gain in artifact_gains
    ) >= 3
    risk_pass = event_macro_harm >= MIN_UNSCOPED_EVENT_HARM and sum(
        harm > 0.0 for harm in event_harms
    ) >= 2
    scoped_safety_pass = max(scoped_event_harms) <= 1e-12
    overall_pass = all_witnesses and gain_pass and risk_pass and scoped_safety_pass

    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": plan["causal_hypothesis"],
        "scientific_role": plan["scientific_role"],
        "plan_exposure": {
            "context_exposure": plan["context_exposure"],
            "outcome_exposure_before_evaluate": plan["outcome_exposure"],
            "outcome_exposure_after_evaluate": "EXPOSED",
        },
        "shared_task_context_across_conditions": True,
        "dataset_evidence": dataset_rows,
        "overall": {
            "dataset_count": len(dataset_rows),
            "all_witnesses_separate_conditions": all_witnesses,
            "artifact_query_gains": artifact_gains,
            "artifact_macro_query_gain": artifact_macro_gain,
            "artifact_positive_dataset_count": sum(gain > 0.0 for gain in artifact_gains),
            "unscoped_event_query_harms": event_harms,
            "unscoped_event_macro_query_harm": event_macro_harm,
            "unscoped_event_harmful_dataset_count": sum(harm > 0.0 for harm in event_harms),
            "scoped_event_query_harms": scoped_event_harms,
            "all_gates_pass": overall_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "CONTROLLED_NONORACLE_TASK_RISK_WITNESS_PASS"
            if overall_pass
            else "CONTROLLED_NONORACLE_TASK_RISK_WITNESS_FAIL"
        ),
        "official_ucr_test_outcome_opened_once": True,
        "original_uci_target_query_opened": False,
        "formal_natural_capability_promotion": False,
        "paper_fresh_transfer_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": (
            "Use the same safe TaskContext/Witness input boundary on one natural or "
            "separately acquired defect family; do not build a Memory platform."
            if overall_pass
            else "Localize the first failed Witness, Program, or Consumer condition without tuning TEST."
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
