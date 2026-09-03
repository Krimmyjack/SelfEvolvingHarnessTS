"""Audit a tie-safe fixed readout on the unchanged W44 controlled protocol.

The Traffic/FRED natural backgrounds, paired task worlds, private bound-node
Program, spike/ramp construction, raw-plus-difference representation, splits,
and direction gates are reused unchanged.  The only changed instrument is the
Consumer readout: sklearn NearestCentroid with Euclidean distance and no class
centroid shrinkage.  Identical event-program class centroids must resolve to
balanced paired-evaluation chance accuracy rather than amplify floating zero.

This is an experimental-instrument audit.  It makes no Harness change and is
not method progress, Capability evidence, promotion, or transfer evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-task-conditioned-tie-safe-readout-audit/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_conditioned_tie_safe_readout_audit_report.json"
)
EXPECTED_TIE_ACCURACY = 0.5


def _fit_nearest_centroid(
    np: Any,
    NearestCentroid: Any,
    classifier_features: Any,
    train_inputs: Any,
    train_labels: Any,
    eval_inputs: Any,
    eval_labels: Any,
) -> dict[str, object]:
    """Fit the one fixed readout and expose its mechanical centroid behavior."""

    model = NearestCentroid(metric="euclidean", shrink_threshold=None)
    fixed_parameters = {"metric": "euclidean", "shrink_threshold": None}
    try:
        model.fit(classifier_features(np, train_inputs), train_labels)
        predictions = np.asarray(
            model.predict(classifier_features(np, eval_inputs)), dtype=np.int64
        )
        centroids = np.asarray(model.centroids_, dtype=np.float64)
        if centroids.ndim != 2 or centroids.shape[0] != 2:
            raise ValueError("tie-safe audit requires exactly two fitted centroids")
    except Exception as error:  # audit must STOP, never switch to a second model
        return {
            "fit_success": False,
            "error": f"{type(error).__name__}: {error}",
            "accuracy": None,
            "centroid_max_abs_difference": None,
            "class_centroids_exact_equal": False,
            "predicted_class_counts": None,
            "model_parameters": fixed_parameters,
        }
    difference = np.abs(centroids[0] - centroids[1])
    return {
        "fit_success": True,
        "error": None,
        "accuracy": float(np.mean(predictions == eval_labels)),
        "centroid_max_abs_difference": float(np.max(difference)),
        "class_centroids_exact_equal": bool(np.array_equal(centroids[0], centroids[1])),
        "predicted_class_counts": {
            str(label): int(np.count_nonzero(predictions == label)) for label in (0, 1)
        },
        "model_parameters": fixed_parameters,
    }


def run(root: Path) -> dict[str, object]:
    import numpy as np
    from sklearn.neighbors import NearestCentroid

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
        read_registry_jsonl,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        BOUND_WINDOW,
        _apply_bound_impulse_oracle,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_impulse_repair_control import (
        COARSE_SIGNAL_AMPLITUDE,
        DATASETS,
        MAX_EVENT_GAIN,
        MIN_COARSE_GAIN,
        SPIKE_AMPLITUDE,
        SPIKE_POSITIONS,
        WINDOW_LENGTH,
        _classifier_features,
        _normalize_background,
        _paired_world,
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

            incumbent = _fit_nearest_centroid(
                np,
                NearestCentroid,
                _classifier_features,
                train_inputs,
                train_labels,
                eval_inputs,
                eval_labels,
            )
            program = _fit_nearest_centroid(
                np,
                NearestCentroid,
                _classifier_features,
                program_inputs,
                train_labels,
                eval_inputs,
                eval_labels,
            )
            fit_count += 2
            fits_succeeded = bool(incumbent["fit_success"] and program["fit_success"])
            incumbent_accuracy = (
                float(incumbent["accuracy"]) if incumbent["accuracy"] is not None else None
            )
            program_accuracy = (
                float(program["accuracy"]) if program["accuracy"] is not None else None
            )
            gain = (
                program_accuracy - incumbent_accuracy
                if incumbent_accuracy is not None and program_accuracy is not None
                else None
            )
            direction_pass = bool(
                fits_succeeded
                and gain is not None
                and (
                    gain >= MIN_COARSE_GAIN
                    if world == "coarse_pattern"
                    else gain <= MAX_EVENT_GAIN
                )
            )
            row_pair_max_difference = float(
                np.max(np.abs(program_inputs[0::2] - program_inputs[1::2]))
            )
            event_tie_checks: dict[str, object] | None = None
            if world == "event_evidence":
                pairs_exact_equal = bool(
                    np.array_equal(program_inputs[0::2], program_inputs[1::2])
                )
                centroids_exact_equal = bool(program["class_centroids_exact_equal"])
                chance_exact = program_accuracy == EXPECTED_TIE_ACCURACY
                tie_safe = bool(
                    fits_succeeded
                    and pairs_exact_equal
                    and centroids_exact_equal
                    and chance_exact
                )
                event_tie_checks = {
                    "bound_program_training_pairs_exact_equal": pairs_exact_equal,
                    "training_pair_max_abs_difference": row_pair_max_difference,
                    "class_centroids_exact_equal": centroids_exact_equal,
                    "centroid_max_abs_difference": program[
                        "centroid_max_abs_difference"
                    ],
                    "expected_program_accuracy": EXPECTED_TIE_ACCURACY,
                    "program_accuracy_exact_chance": chance_exact,
                    "deterministic_tie_is_stable": tie_safe,
                }
            world_evidence[world] = {
                "incumbent_accuracy": incumbent_accuracy,
                "program_accuracy": program_accuracy,
                "program_gain": gain,
                "both_fits_succeeded": fits_succeeded,
                "fixed_direction_requirement": (
                    f"gain >= {MIN_COARSE_GAIN}"
                    if world == "coarse_pattern"
                    else f"gain <= {MAX_EVENT_GAIN}"
                ),
                "direction_pass": direction_pass,
                "incumbent_readout": incumbent,
                "program_readout": program,
                "event_tie_checks": event_tie_checks,
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
        event_checks = world_evidence["event_evidence"]["event_tie_checks"]
        if not isinstance(event_checks, dict):
            raise AssertionError("event tie checks were not produced")
        tie_safe = bool(event_checks["deterministic_tie_is_stable"])
        dataset_pass = bool(
            labels_paired
            and tie_safe
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
                "tie_safe_readout_pass": tie_safe,
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
        "scientific_role": "experimental_instrument_tie_safe_readout_audit",
        "evidence_stage": "DEVELOPMENT_CONTROL",
        "causal_hypothesis": (
            "A deterministic nearest-centroid readout should map identical class "
            "centroids to exact 0.5 paired-evaluation accuracy, making the unchanged "
            "bound Program's event-risk effect interpretable without floating-zero "
            "sign amplification."
        ),
        "configuration": {
            "datasets": list(DATASETS),
            "natural_base_series": True,
            "window_length": WINDOW_LENGTH,
            "training_anchors": list(ANCHORS),
            "labels_per_natural_background": 2,
            "shared_spike_positions": list(SPIKE_POSITIONS),
            "shared_spike_amplitude": SPIKE_AMPLITUDE,
            "coarse_signal_amplitude": COARSE_SIGNAL_AMPLITUDE,
            "bound_oracle_window": BOUND_WINDOW,
            "program": "unchanged W44 private four-node bound impulse oracle",
            "classifier_representation": (
                "unchanged W43/W44 raw 192-point sequence plus 191 first differences"
            ),
            "consumer_readout": {
                "class": "sklearn.neighbors.NearestCentroid",
                "metric": "euclidean",
                "shrink_threshold": None,
            },
            "expected_tie_accuracy": EXPECTED_TIE_ACCURACY,
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
            "all_dataset_task_contrasts_and_ties_pass": overall_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "TIE_SAFE_READOUT_AUDIT_PASS"
            if overall_pass
            else "TIE_SAFE_READOUT_AUDIT_FAIL"
        ),
        "protocol_decision": (
            "CONTINUE_FIXED_CLASSIFICATION_PROTOCOL"
            if overall_pass
            else "STOP_CURRENT_CLASSIFICATION_PROTOCOL"
        ),
        "allowed_protocol_decisions": [
            "CONTINUE_FIXED_CLASSIFICATION_PROTOCOL",
            "STOP_CURRENT_CLASSIFICATION_PROTOCOL",
        ],
        "harness_changes": 0,
        "method_progress": False,
        "target_query_opened": False,
        "uci_values_read": False,
        "capability_claim": False,
        "capability_promotion": False,
        "formal_transfer": False,
        "claim_limit": (
            "Experimental instrument audit on exposed Traffic/FRED Source controls. "
            "It changes no Harness surface and is not method progress, Capability, "
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
    print(report["protocol_decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
