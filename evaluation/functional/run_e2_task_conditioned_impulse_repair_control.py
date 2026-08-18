"""Run one controlled task-conditioned Hampel training-data Program slice.

Traffic and FRED provide the same exposed natural background protocol as W42.
Every background is paired with both labels.  In the coarse task, label evidence
is a low-frequency ramp while sparse training-only acquisition spikes create a
train/evaluation mismatch.  In the event task, the identical sparse geometry is
legal label evidence in both train and evaluation.  The sole optional Program is
the repository Hampel operator with its literature defaults; it processes every
complete training input before the fixed classifier is refit from scratch.
Evaluation inputs are never processed.

This is a development mechanism control, not evidence of natural prevalence,
Capability promotion, or Source-to-Target transfer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-task-conditioned-impulse-repair-control/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_conditioned_impulse_repair_control_report.json"
)
DATASETS = ("monash:traffic_hourly", "legacy_monash:fred_md")
WINDOW_LENGTH = 192
HAMPEL_WINDOW = 7
HAMPEL_N_SIGMAS = 3.0
COARSE_SIGNAL_AMPLITUDE = 0.75
SPIKE_AMPLITUDE = 16.0
SPIKE_POSITIONS = (12, 36, 156, 180)
RIDGE_ALPHA = 1.0
MIN_COARSE_GAIN = 0.05
MAX_EVENT_GAIN = -0.05
MODIFIED_TOLERANCE = 1e-12


def _normalize_background(np: Any, values: Any, *, identity: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (WINDOW_LENGTH,) or not np.isfinite(array).all():
        raise ValueError(f"invalid natural background window: {identity}")
    scale = float(np.std(array))
    if scale <= 1e-8:
        raise ValueError(f"degenerate natural background window: {identity}")
    return (array - float(np.mean(array))) / scale


def _spike_template(np: Any) -> Any:
    """Frozen sparse geometry shared by acquisition noise and legal events."""

    template = np.zeros(WINDOW_LENGTH, dtype=np.float64)
    midpoint = (WINDOW_LENGTH - 1) / 2.0
    for position in SPIKE_POSITIONS:
        side = -1.0 if position < midpoint else 1.0
        template[position] = -side * SPIKE_AMPLITUDE
    return template


def _paired_world(
    np: Any,
    backgrounds: Any,
    *,
    world: str,
    training: bool,
) -> tuple[Any, Any]:
    """Create both labels over each background with a shared label ordering."""

    background_matrix = np.asarray(backgrounds, dtype=np.float64)
    time = np.linspace(-1.0, 1.0, WINDOW_LENGTH, dtype=np.float64)
    spikes = _spike_template(np)
    rows: list[Any] = []
    labels: list[int] = []
    for background in background_matrix:
        for label in (0, 1):
            sign = -1.0 if label == 0 else 1.0
            values = background.copy()
            if world == "coarse_pattern":
                values += sign * COARSE_SIGNAL_AMPLITUDE * time
                if training:
                    values += sign * spikes
            elif world == "event_evidence":
                values += sign * spikes
            else:
                raise ValueError(f"unknown task world: {world}")
            rows.append(values)
            labels.append(label)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _apply_hampel_program(
    np: Any,
    hampel_filter: Any,
    train_inputs: Any,
) -> tuple[Any, dict[str, object]]:
    """Apply the existing default Hampel operator to every complete training row."""

    matrix = np.asarray(train_inputs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW_LENGTH:
        raise ValueError("Hampel Program expects complete 192-point training inputs")
    processed = np.asarray(
        [
            hampel_filter(
                row,
                window=HAMPEL_WINDOW,
                n_sigmas=HAMPEL_N_SIGMAS,
            )
            for row in matrix
        ],
        dtype=np.float64,
    )
    if processed.shape != matrix.shape or not np.isfinite(processed).all():
        raise ValueError("Hampel Program returned invalid training inputs")
    changed = np.abs(processed - matrix) > MODIFIED_TOLERANCE
    modified_points = int(np.count_nonzero(changed))
    modified_rows = int(np.count_nonzero(np.any(changed, axis=1)))
    return processed, {
        "modified_point_count": modified_points,
        "total_point_count": int(matrix.size),
        "modified_point_fraction": modified_points / float(matrix.size),
        "modified_row_count": modified_rows,
        "total_row_count": int(matrix.shape[0]),
        "modified_row_fraction": modified_rows / float(matrix.shape[0]),
        "comparison_tolerance": MODIFIED_TOLERANCE,
    }


def _classifier_features(np: Any, rows: Any) -> Any:
    """W43's single preregistered Consumer view: raw sequence plus differences."""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW_LENGTH:
        raise ValueError("classifier expects aligned 192-point inputs")
    return np.concatenate((matrix, np.diff(matrix, axis=1)), axis=1)


def _fit_accuracy(
    np: Any,
    RidgeClassifier: Any,
    train_inputs: Any,
    train_labels: Any,
    eval_inputs: Any,
    eval_labels: Any,
) -> float:
    model = RidgeClassifier(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
        solver="svd",
    )
    model.fit(_classifier_features(np, train_inputs), train_labels)
    predictions = model.predict(_classifier_features(np, eval_inputs))
    return float(np.mean(np.asarray(predictions, dtype=np.int64) == eval_labels))


def run(root: Path) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
        read_registry_jsonl,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
        ANCHORS,
        J0_PLAN_PATH,
        SPECS,
        _read_object,
    )
    from SelfEvolvingHarnessTS.operators.s1_outlier import hampel_filter

    plan_roster = _read_object(root / J0_PLAN_PATH).get("roster")
    if not isinstance(plan_roster, list):
        raise ValueError("J0 Source roster is missing")
    roster = [row for row in plan_roster if row.get("dataset_id") in DATASETS]
    if (
        len(roster) != 40
        or len(roster) != len(plan_roster)
        or any(str(row.get("dataset_id", "")).startswith("uci") for row in roster)
    ):
        raise ValueError("expected the exposed Traffic/FRED Source roster only")

    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    selected_records = []
    for row in roster:
        uid = str(row["series_uid"])
        record = records.get(uid)
        if record is None or record.dataset_id not in DATASETS:
            raise ValueError(f"Source roster/registry mismatch: {uid}")
        selected_records.append(record)
    values = _load_values(
        selected_records,
        root / "data/benchmark_v0_2/clean_base",
    )

    fit_count = 0
    dataset_evidence: list[dict[str, object]] = []
    for dataset_id in DATASETS:
        train_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "train"
        ]
        eval_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "eval"
        ]
        if len(train_rows) != 12 or len(eval_rows) != 8:
            raise ValueError(f"Source roster geometry changed: {dataset_id}")

        train_backgrounds: list[Any] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                train_backgrounds.append(
                    _normalize_background(
                        np,
                        values[uid][anchor - WINDOW_LENGTH : anchor],
                        identity=f"{uid}@{anchor}",
                    )
                )
        train_stop = int(SPECS[dataset_id]["train_stop"])
        eval_backgrounds: list[Any] = []
        for row in eval_rows:
            uid = str(row["series_uid"])
            eval_backgrounds.append(
                _normalize_background(
                    np,
                    values[uid][train_stop - WINDOW_LENGTH : train_stop],
                    identity=f"{uid}@{train_stop}",
                )
            )
        train_background_array = np.asarray(train_backgrounds, dtype=np.float64)
        eval_background_array = np.asarray(eval_backgrounds, dtype=np.float64)
        if train_background_array.shape != (72, WINDOW_LENGTH):
            raise AssertionError(f"training background geometry changed: {dataset_id}")
        if eval_background_array.shape != (8, WINDOW_LENGTH):
            raise AssertionError(f"evaluation background geometry changed: {dataset_id}")

        world_evidence: dict[str, dict[str, object]] = {}
        labels_by_world: dict[str, tuple[Any, Any]] = {}
        for world in ("coarse_pattern", "event_evidence"):
            train_inputs, train_labels = _paired_world(
                np, train_background_array, world=world, training=True
            )
            eval_inputs, eval_labels = _paired_world(
                np, eval_background_array, world=world, training=False
            )
            labels_by_world[world] = (train_labels, eval_labels)
            program_inputs, modification = _apply_hampel_program(
                np, hampel_filter, train_inputs
            )
            incumbent_accuracy = _fit_accuracy(
                np,
                RidgeClassifier,
                train_inputs,
                train_labels,
                eval_inputs,
                eval_labels,
            )
            program_accuracy = _fit_accuracy(
                np,
                RidgeClassifier,
                program_inputs,
                train_labels,
                eval_inputs,
                eval_labels,
            )
            fit_count += 2
            gain = program_accuracy - incumbent_accuracy
            direction_pass = (
                gain >= MIN_COARSE_GAIN
                if world == "coarse_pattern"
                else gain <= MAX_EVENT_GAIN
            )
            mechanism_hit = int(modification["modified_point_count"]) > 0
            world_evidence[world] = {
                "task_context": (
                    {
                        "label_evidence": "global_low_frequency_ramp_direction",
                        "sparse_impulse_semantics": "training_only_acquisition_artifact",
                        "evaluation_condition": "clean",
                        "task_conditioned_action": "execute_hampel",
                    }
                    if world == "coarse_pattern"
                    else {
                        "label_evidence": "sparse_local_event_pattern",
                        "sparse_impulse_semantics": "legal_task_evidence",
                        "evaluation_condition": "matched_legal_event",
                        "task_conditioned_action": "abstain_keep_incumbent",
                    }
                ),
                "incumbent_accuracy": incumbent_accuracy,
                "program_accuracy": program_accuracy,
                "program_gain": gain,
                "fixed_direction_requirement": (
                    f"gain >= {MIN_COARSE_GAIN}"
                    if world == "coarse_pattern"
                    else f"gain <= {MAX_EVENT_GAIN}"
                ),
                "direction_pass": direction_pass,
                "program_mechanism_hit": mechanism_hit,
                "training_input_modification": modification,
                "train_instance_count": int(train_inputs.shape[0]),
                "eval_instance_count": int(eval_inputs.shape[0]),
                "evaluation_input_processed": False,
            }

        coarse_labels = labels_by_world["coarse_pattern"]
        event_labels = labels_by_world["event_evidence"]
        labels_paired = bool(
            np.array_equal(coarse_labels[0], event_labels[0])
            and np.array_equal(coarse_labels[1], event_labels[1])
        )
        all_mechanisms_hit = all(
            bool(world_evidence[world]["program_mechanism_hit"])
            for world in world_evidence
        )
        dataset_pass = bool(
            labels_paired
            and all_mechanisms_hit
            and world_evidence["coarse_pattern"]["direction_pass"]
            and world_evidence["event_evidence"]["direction_pass"]
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "natural_background_pairing": {
                    "same_backgrounds_and_labels_across_task_worlds": labels_paired,
                    "train_background_count": int(train_background_array.shape[0]),
                    "eval_background_count": int(eval_background_array.shape[0]),
                    "labels_per_background": 2,
                    "train_eval_series_disjoint": not bool(
                        {str(row["series_uid"]) for row in train_rows}
                        & {str(row["series_uid"]) for row in eval_rows}
                    ),
                },
                "shared_sparse_geometry": {
                    "positions": list(SPIKE_POSITIONS),
                    "amplitude": SPIKE_AMPLITUDE,
                    "same_geometry_across_task_worlds": True,
                },
                "task_worlds": world_evidence,
                "dataset_gate_pass": dataset_pass,
            }
        )

    expected_fits = len(DATASETS) * 2 * 2
    if fit_count != expected_fits:
        raise RuntimeError(f"expected {expected_fits} fits, observed {fit_count}")
    passing_dataset_count = sum(
        bool(row["dataset_gate_pass"]) for row in dataset_evidence
    )
    overall_pass = passing_dataset_count == len(DATASETS)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "controlled_task_conditioned_program_mechanism_calibration",
        "causal_hypothesis": (
            "With identical sparse geometry, Hampel should repair acquisition spikes "
            "for a coarse-pattern task but destroy legal sparse-event label evidence, "
            "so visible TaskSpec semantics must select execute versus abstain."
        ),
        "evidence_stage": "DEVELOPMENT_CONTROL",
        "configuration": {
            "datasets": list(DATASETS),
            "natural_base_series": True,
            "window_length": WINDOW_LENGTH,
            "training_anchors": list(ANCHORS),
            "labels_per_natural_background": 2,
            "program": (
                "operators.s1_outlier.hampel_filter over every complete training input; "
                "classifier refit from scratch; evaluation inputs untouched"
            ),
            "hampel_parameters": {
                "window": HAMPEL_WINDOW,
                "n_sigmas": HAMPEL_N_SIGMAS,
                "source": "repository_operator_literature_defaults",
            },
            "classifier": (
                f"RidgeClassifier(alpha={RIDGE_ALPHA}, fit_intercept=True, solver=svd)"
            ),
            "classifier_representation": (
                "W43 preregistered protocol: complete raw 192-point sequence plus "
                "191 first differences"
            ),
            "consumer_protocol_relation_to_w42": (
                "New Program family with a single full-resolution Consumer view; W42's "
                "landmark representation and moving-average outcomes are not reused."
            ),
            "coarse_signal_amplitude": COARSE_SIGNAL_AMPLITUDE,
            "shared_spike_amplitude": SPIKE_AMPLITUDE,
            "shared_spike_positions": list(SPIKE_POSITIONS),
            "frozen_direction_gate": {
                "minimum_coarse_gain": MIN_COARSE_GAIN,
                "maximum_event_gain": MAX_EVENT_GAIN,
                "program_must_modify_training_points_in_both_worlds": True,
                "all_datasets_required": True,
            },
            "parameter_search": False,
        },
        "dataset_evidence": dataset_evidence,
        "overall": {
            "dataset_count": len(DATASETS),
            "passing_dataset_count": passing_dataset_count,
            "all_dataset_task_contrasts_pass": overall_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "TASK_CONDITIONED_IMPULSE_REPAIR_CONTROL_PASS"
            if overall_pass
            else "TASK_CONDITIONED_IMPULSE_REPAIR_CONTROL_FAIL"
        ),
        "target_query_opened": False,
        "uci_values_read": False,
        "capability_promotion": False,
        "formal_transfer": False,
        "natural_prevalence_claim": False,
        "claim_limit": (
            "Controlled mechanism calibration on exposed Traffic/FRED Source natural "
            "backgrounds. It is not natural prevalence, Capability promotion, or "
            "Source-to-Target transfer evidence."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(root)
    output = args.output or root / DEFAULT_REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
