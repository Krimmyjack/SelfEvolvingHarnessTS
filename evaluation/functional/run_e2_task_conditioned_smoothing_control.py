"""Run one controlled task-conditioned training-input smoothing slice.

Traffic and FRED provide exposed natural background windows.  Each background
is paired with both labels in two task worlds.  The coarse-pattern world adds a
low-frequency class ramp plus a deterministic, training-only high-frequency
acquisition artifact.  The event-evidence world adds a legal short event to
both train and evaluation inputs.  The sole optional Program smooths every
training input before the fixed classifier is refit; evaluation inputs are
always left untouched.

This is development mechanism calibration.  It does not measure natural
prevalence, Capability promotion, or Source-to-Target transfer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-task-conditioned-smoothing-control/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_conditioned_smoothing_control_report.json"
)
DATASETS = ("monash:traffic_hourly", "legacy_monash:fred_md")
WINDOW_LENGTH = 192
SMOOTHING_WIDTH = 9
LANDMARK_OFFSET = 4
LANDMARK_STRIDE = 8
COARSE_SIGNAL_AMPLITUDE = 0.75
COARSE_ARTIFACT_MULTIPLIER = -2.5
EVENT_SIGNAL_AMPLITUDE = 3.0
RIDGE_ALPHA = 1.0
MIN_COARSE_GAIN = 0.05
MAX_EVENT_GAIN = -0.05


def _normalize_background(np: Any, values: Any, *, identity: str) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (WINDOW_LENGTH,) or not np.isfinite(array).all():
        raise ValueError(f"invalid natural background window: {identity}")
    scale = float(np.std(array))
    if scale <= 1e-8:
        raise ValueError(f"degenerate natural background window: {identity}")
    return (array - float(np.mean(array))) / scale


def _smooth_training_inputs(np: Any, rows: Any) -> Any:
    """Apply the one frozen Program to complete training rows only."""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW_LENGTH:
        raise ValueError("smoothing expects complete 192-point training inputs")
    radius = SMOOTHING_WIDTH // 2
    kernel = np.full(SMOOTHING_WIDTH, 1.0 / SMOOTHING_WIDTH, dtype=np.float64)
    return np.asarray(
        [
            np.convolve(np.pad(row, (radius, radius), mode="reflect"), kernel, mode="valid")
            for row in matrix
        ],
        dtype=np.float64,
    )


def _classifier_features(np: Any, rows: Any) -> Any:
    """Fixed TS-aware representation: values and changes at temporal landmarks."""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != WINDOW_LENGTH:
        raise ValueError("classifier expects aligned 192-point inputs")
    landmarks = np.arange(
        LANDMARK_OFFSET, WINDOW_LENGTH, LANDMARK_STRIDE, dtype=np.int64
    )
    sampled = matrix[:, landmarks]
    changes = np.diff(sampled, axis=1)
    return np.concatenate((sampled, changes), axis=1)


def _paired_world(
    np: Any,
    backgrounds: Any,
    *,
    world: str,
    training: bool,
) -> tuple[Any, Any]:
    """Create both labels over every background, identically across task worlds."""

    background_matrix = np.asarray(backgrounds, dtype=np.float64)
    time = np.linspace(-1.0, 1.0, WINDOW_LENGTH, dtype=np.float64)
    landmark_indices = np.arange(
        LANDMARK_OFFSET, WINDOW_LENGTH, LANDMARK_STRIDE, dtype=np.int64
    )
    rows: list[Any] = []
    labels: list[int] = []
    for background in background_matrix:
        for label in (0, 1):
            sign = -1.0 if label == 0 else 1.0
            values = background.copy()
            if world == "coarse_pattern":
                low_frequency_evidence = sign * COARSE_SIGNAL_AMPLITUDE * time
                values += low_frequency_evidence
                if training:
                    # Sparse one-step impulses are a controlled high-frequency
                    # acquisition artifact.  They are absent from clean evaluation.
                    artifact = np.zeros(WINDOW_LENGTH, dtype=np.float64)
                    artifact[landmark_indices] = (
                        COARSE_ARTIFACT_MULTIPLIER
                        * low_frequency_evidence[landmark_indices]
                    )
                    values += artifact
            elif world == "event_evidence":
                # The class is carried by a legal one-step local event.  The same
                # event geometry appears in training and evaluation.
                values[WINDOW_LENGTH // 2 + 4] += sign * EVENT_SIGNAL_AMPLITUDE
            else:
                raise ValueError(f"unknown task world: {world}")
            rows.append(values)
            labels.append(label)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


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
        train_background_ids: list[str] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                identity = f"{uid}@{anchor}"
                train_backgrounds.append(
                    _normalize_background(
                        np,
                        values[uid][anchor - WINDOW_LENGTH : anchor],
                        identity=identity,
                    )
                )
                train_background_ids.append(identity)
        train_stop = int(SPECS[dataset_id]["train_stop"])
        eval_backgrounds: list[Any] = []
        eval_background_ids: list[str] = []
        for row in eval_rows:
            uid = str(row["series_uid"])
            identity = f"{uid}@{train_stop}"
            eval_backgrounds.append(
                _normalize_background(
                    np,
                    values[uid][train_stop - WINDOW_LENGTH : train_stop],
                    identity=identity,
                )
            )
            eval_background_ids.append(identity)
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
            program_train_inputs = _smooth_training_inputs(np, train_inputs)
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
                program_train_inputs,
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
            world_evidence[world] = {
                "task_context": (
                    {
                        "label_evidence": "global_low_frequency_ramp_direction",
                        "training_condition": "deterministic_high_frequency_acquisition_noise",
                        "evaluation_condition": "clean",
                        "task_conditioned_action": "execute_smoothing",
                    }
                    if world == "coarse_pattern"
                    else {
                        "label_evidence": "localized_one_step_event_polarity",
                        "training_condition": "legal_matched_event",
                        "evaluation_condition": "legal_matched_event",
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
                "train_instance_count": int(train_inputs.shape[0]),
                "eval_instance_count": int(eval_inputs.shape[0]),
                "program_changed_training_input": bool(
                    not np.array_equal(train_inputs, program_train_inputs)
                ),
                "evaluation_input_processed": False,
            }
        coarse_labels = labels_by_world["coarse_pattern"]
        event_labels = labels_by_world["event_evidence"]
        labels_paired = bool(
            np.array_equal(coarse_labels[0], event_labels[0])
            and np.array_equal(coarse_labels[1], event_labels[1])
        )
        dataset_pass = bool(
            labels_paired
            and world_evidence["coarse_pattern"]["direction_pass"]
            and world_evidence["event_evidence"]["direction_pass"]
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "natural_background_pairing": {
                    "same_backgrounds_and_labels_across_task_worlds": labels_paired,
                    "train_background_count": len(train_background_ids),
                    "eval_background_count": len(eval_background_ids),
                    "labels_per_background": 2,
                    "train_eval_series_disjoint": not bool(
                        {str(row["series_uid"]) for row in train_rows}
                        & {str(row["series_uid"]) for row in eval_rows}
                    ),
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
            "With the same natural backgrounds and fixed classifier, training-input "
            "smoothing helps a coarse low-frequency task under high-frequency training "
            "noise, but should be abstained from when a legal local event carries the label."
        ),
        "evidence_stage": "DEVELOPMENT_CONTROL",
        "configuration": {
            "datasets": list(DATASETS),
            "natural_base_series": True,
            "window_length": WINDOW_LENGTH,
            "training_anchors": list(ANCHORS),
            "labels_per_natural_background": 2,
            "program": (
                "centered moving-average smoothing over every complete training input; "
                "classifier refit from scratch; evaluation inputs untouched"
            ),
            "smoothing_width": SMOOTHING_WIDTH,
            "classifier": (
                f"RidgeClassifier(alpha={RIDGE_ALPHA}, fit_intercept=True, solver=svd)"
            ),
            "classifier_representation": (
                "24 fixed temporal landmarks at offset 4/stride 8 plus 23 successive "
                "landmark differences"
            ),
            "coarse_signal_amplitude": COARSE_SIGNAL_AMPLITUDE,
            "coarse_training_artifact_multiplier": COARSE_ARTIFACT_MULTIPLIER,
            "event_signal_amplitude": EVENT_SIGNAL_AMPLITUDE,
            "frozen_direction_gate": {
                "minimum_coarse_gain": MIN_COARSE_GAIN,
                "maximum_event_gain": MAX_EVENT_GAIN,
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
            "TASK_CONDITIONED_SMOOTHING_CONTROL_PASS"
            if overall_pass
            else "TASK_CONDITIONED_SMOOTHING_CONTROL_FAIL"
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
