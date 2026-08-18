"""Run the private bound-node Program oracle for the W43 impulse control.

This runner reuses W43's exposed Traffic/FRED natural backgrounds, paired task
worlds, sparse impulse geometry, raw-plus-difference Ridge consumer, and frozen
direction gates.  The only change is Program Binding: a private development
oracle binds the four known Pattern Node positions and replaces each training
value with the median of its six neighboring values, excluding the center.
Evaluation inputs are never processed.

This is development headroom only.  No deployable Observation is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-task-conditioned-bound-impulse-oracle/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_conditioned_bound_impulse_oracle_report.json"
)
BOUND_WINDOW = 7
BOUND_RADIUS = 3
MODIFIED_TOLERANCE = 1e-12


def _apply_bound_impulse_oracle(
    np: Any,
    train_inputs: Any,
    *,
    positions: tuple[int, ...],
    window_length: int,
) -> tuple[Any, dict[str, object]]:
    """Replace only private bound nodes using a center-excluded local median."""

    matrix = np.asarray(train_inputs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != window_length:
        raise ValueError("bound-node Program expects complete training inputs")
    if BOUND_WINDOW != 2 * BOUND_RADIUS + 1:
        raise AssertionError("bound-node neighborhood geometry changed")
    if any(
        position - BOUND_RADIUS < 0
        or position + BOUND_RADIUS >= window_length
        for position in positions
    ):
        raise ValueError("bound position does not support the frozen neighborhood")

    processed = matrix.copy()
    for row_index, row in enumerate(matrix):
        for position in positions:
            neighbors = np.concatenate(
                (
                    row[position - BOUND_RADIUS : position],
                    row[position + 1 : position + BOUND_RADIUS + 1],
                )
            )
            if neighbors.shape != (6,) or not np.isfinite(neighbors).all():
                raise ValueError("invalid center-excluded bound-node neighborhood")
            processed[row_index, position] = float(np.median(neighbors))

    changed = np.abs(processed - matrix) > MODIFIED_TOLERANCE
    allowed = np.zeros(matrix.shape, dtype=bool)
    allowed[:, list(positions)] = True
    if bool(np.any(changed & ~allowed)):
        raise AssertionError("bound-node Program changed an unbound position")
    changed_positions_by_row = [
        tuple(int(index) for index in np.flatnonzero(row_changed))
        for row_changed in changed
    ]
    expected_positions = tuple(int(position) for position in positions)
    if any(row_positions != expected_positions for row_positions in changed_positions_by_row):
        raise AssertionError("not every bound node was materially replaced")
    modified_points = int(np.count_nonzero(changed))
    return processed, {
        "bound_positions": list(expected_positions),
        "exact_modified_positions": list(expected_positions),
        "same_exact_positions_on_every_training_row": True,
        "only_bound_positions_changed": True,
        "modified_points_per_row": len(expected_positions),
        "modified_point_count": modified_points,
        "total_point_count": int(matrix.size),
        "modified_point_fraction": modified_points / float(matrix.size),
        "modified_row_count": int(matrix.shape[0]),
        "total_row_count": int(matrix.shape[0]),
        "modified_row_fraction": 1.0,
        "comparison_tolerance": MODIFIED_TOLERANCE,
        "replacement": (
            "median(indices i-3:i concatenated with i+1:i+4); center excluded"
        ),
    }


def run(root: Path) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
        read_registry_jsonl,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_impulse_repair_control import (
        COARSE_SIGNAL_AMPLITUDE,
        DATASETS,
        MAX_EVENT_GAIN,
        MIN_COARSE_GAIN,
        RIDGE_ALPHA,
        SPIKE_AMPLITUDE,
        SPIKE_POSITIONS,
        WINDOW_LENGTH,
        _fit_accuracy,
        _normalize_background,
        _paired_world,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
        ANCHORS,
        J0_PLAN_PATH,
        SPECS,
        _read_object,
    )

    if BOUND_WINDOW != 7:
        raise AssertionError("W44 bound-node window must remain fixed at 7")
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
            program_inputs, modification = _apply_bound_impulse_oracle(
                np,
                train_inputs,
                positions=SPIKE_POSITIONS,
                window_length=WINDOW_LENGTH,
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
            world_evidence[world] = {
                "task_context": (
                    {
                        "label_evidence": "global_low_frequency_ramp_direction",
                        "bound_impulse_semantics": "training_only_acquisition_artifact",
                        "task_conditioned_action": "execute_bound_repair",
                    }
                    if world == "coarse_pattern"
                    else {
                        "label_evidence": "sparse_local_event_pattern",
                        "bound_impulse_semantics": "legal_task_evidence",
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
                    "train_background_count": int(train_background_array.shape[0]),
                    "eval_background_count": int(eval_background_array.shape[0]),
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
        "scientific_role": "private_bound_program_binding_headroom_control",
        "causal_hypothesis": (
            "If W43 failed because the global detector touched the wrong points, then "
            "an oracle bound to the four true impulse nodes should recover coarse-task "
            "utility while destroying matched legal event evidence."
        ),
        "evidence_stage": "DEVELOPMENT_CONTROL",
        "oracle_bound_nodes": True,
        "deployable_observation_available": False,
        "configuration": {
            "datasets": list(DATASETS),
            "natural_base_series": True,
            "window_length": WINDOW_LENGTH,
            "training_anchors": list(ANCHORS),
            "labels_per_natural_background": 2,
            "shared_spike_positions": list(SPIKE_POSITIONS),
            "shared_spike_amplitude": SPIKE_AMPLITUDE,
            "coarse_signal_amplitude": COARSE_SIGNAL_AMPLITUDE,
            "program_binding": {
                "binding": "private_oracle_four_pattern_nodes",
                "positions": list(SPIKE_POSITIONS),
                "window": BOUND_WINDOW,
                "replacement": (
                    "center-excluded median of i-3:i and i+1:i+4"
                ),
                "all_unbound_points_preserved_exactly": True,
                "evaluation_inputs_processed": False,
            },
            "consumer": (
                f"RidgeClassifier(alpha={RIDGE_ALPHA}, fit_intercept=True, solver=svd)"
            ),
            "classifier_representation": (
                "W43 frozen complete raw 192-point sequence plus 191 first differences"
            ),
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
            "TASK_CONDITIONED_BOUND_IMPULSE_ORACLE_PASS"
            if overall_pass
            else "TASK_CONDITIONED_BOUND_IMPULSE_ORACLE_FAIL"
        ),
        "next_step": (
            "Design one deployable visible Observation for the bound impulse nodes; "
            "do not build Witness or Memory."
            if overall_pass
            else "Close the task-conditioned impulse-repair family."
        ),
        "target_query_opened": False,
        "uci_values_read": False,
        "development_headroom_only": True,
        "capability_claim": False,
        "capability_promotion": False,
        "formal_transfer": False,
        "natural_prevalence_claim": False,
        "claim_limit": (
            "Private oracle-bound development headroom on exposed Traffic/FRED Source "
            "backgrounds. No deployable Observation exists; this is not Capability, "
            "promotion, natural prevalence, or transfer evidence."
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
