"""Run the W58 development-only Pattern-Mass Multiplicity headroom check.

The sole hypothesis is that duplicating a TRAIN-visible high-roughness pattern
group biases the shared Consumer objective, while an inverse-multiplicity
sample-weight Program recovers the original objective.  The four official UCR
TEST splits used here were exposed by W48/W49, so this runner is diagnostic
development evidence only: it cannot promote a Capability or support a fresh
cross-dataset transfer claim.

No Target data, persistent Memory, proxy attribution, or Harness update is
used.  The fixed Consumer and raw-plus-difference features are reused from the
controlled classification experiments.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-pattern-mass-multiplicity-headroom/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_pattern_mass_multiplicity_headroom_report.json"
)
DATA_DIR = "data/ucr_task_context"
DATASETS = ("Coffee", "ECG200", "GunPoint", "BeetleFly")
MULTIPLICITY = 8
RIDGE_ALPHA = 1.0
PATTERN_FRACTION = 0.25
DECISION_TOLERANCE = 1e-8


def _roughness_scores(np: Any, values: Any) -> Any:
    """Compute the frozen TRAIN-visible, label-free local roughness score."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError("roughness observation requires finite complete series")
    center = np.median(matrix, axis=1, keepdims=True)
    mad_scale = 1.4826 * np.median(np.abs(matrix - center), axis=1)
    denominator = np.maximum(mad_scale, 1e-12)
    numerator = np.median(np.abs(np.diff(matrix, axis=1)), axis=1)
    scores = numerator / denominator
    if not np.isfinite(scores).all():
        raise ValueError("non-finite roughness score")
    return scores


def _top_quartile_indices(np: Any, scores: Any) -> Any:
    """Select the fixed-size top quartile using a deterministic stable order."""

    vector = np.asarray(scores, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("pattern-group selection requires at least one row")
    count = max(1, int(math.ceil(PATTERN_FRACTION * vector.size)))
    stable_order = np.argsort(vector, kind="stable")
    return np.asarray(stable_order[-count:], dtype=np.int64)


def _duplicate_pattern_group(
    np: Any,
    values: Any,
    labels: Any,
    selected: Any,
) -> tuple[Any, Any, Any, Any]:
    """Duplicate each selected original to total multiplicity eight."""

    matrix = np.asarray(values, dtype=np.float64)
    targets = np.asarray(labels)
    chosen = np.asarray(selected, dtype=np.int64)
    repeats = MULTIPLICITY - 1
    duplicate_indices = np.repeat(chosen, repeats)
    corrupt_values = np.concatenate((matrix, matrix[duplicate_indices]), axis=0)
    corrupt_labels = np.concatenate((targets, targets[duplicate_indices]), axis=0)
    origin_indices = np.concatenate(
        (np.arange(matrix.shape[0], dtype=np.int64), duplicate_indices), axis=0
    )
    weights = np.ones(corrupt_values.shape[0], dtype=np.float64)
    selected_mask = np.isin(origin_indices, chosen)
    weights[selected_mask] = 1.0 / MULTIPLICITY
    return corrupt_values, corrupt_labels, origin_indices, weights


def _class_counts(np: Any, labels: Any) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _fit_model(
    RidgeClassifier: Any,
    features: Any,
    np: Any,
    values: Any,
    labels: Any,
    sample_weight: Any = None,
) -> Any:
    model = RidgeClassifier(alpha=RIDGE_ALPHA)
    model.fit(features(np, values), labels, sample_weight=sample_weight)
    return model


def evaluate(root: Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _features,
        _load_split,
    )

    # Phase 1 is TRAIN-only: observe the Pattern group and compile the fixed
    # inverse-multiplicity Program before any TEST split is loaded.
    prepared: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        archive = root / DATA_DIR / f"{dataset}.zip"
        train_values, train_labels = _load_split(np, archive, dataset, "TRAIN")
        scores = _roughness_scores(np, train_values)
        selected = _top_quartile_indices(np, scores)
        corrupt_values, corrupt_labels, origins, weights = _duplicate_pattern_group(
            np, train_values, train_labels, selected
        )

        aggregate_weights = np.bincount(
            origins, weights=weights, minlength=train_values.shape[0]
        )
        selected_occurrences = np.isin(origins, selected)
        prepared[dataset] = {
            "archive": archive,
            "train_values": train_values,
            "train_labels": train_labels,
            "roughness_scores": scores,
            "selected": selected,
            "corrupt_values": corrupt_values,
            "corrupt_labels": corrupt_labels,
            "origins": origins,
            "weights": weights,
            "aggregate_weights": aggregate_weights,
            "selected_occurrences": selected_occurrences,
        }

    rows: list[dict[str, Any]] = []
    consumer_fit_count = 0
    test_split_load_count = 0
    for dataset in DATASETS:
        state = prepared[dataset]
        query_values, query_labels = _load_split(
            np, state["archive"], dataset, "TEST"
        )
        test_split_load_count += 1

        clean_model = _fit_model(
            RidgeClassifier,
            _features,
            np,
            state["train_values"],
            state["train_labels"],
        )
        corrupt_model = _fit_model(
            RidgeClassifier,
            _features,
            np,
            state["corrupt_values"],
            state["corrupt_labels"],
        )
        recovered_model = _fit_model(
            RidgeClassifier,
            _features,
            np,
            state["corrupt_values"],
            state["corrupt_labels"],
            sample_weight=state["weights"],
        )
        consumer_fit_count += 3

        query_features = _features(np, query_values)
        clean_predictions = clean_model.predict(query_features)
        corrupt_predictions = corrupt_model.predict(query_features)
        recovered_predictions = recovered_model.predict(query_features)
        clean_decision = np.asarray(clean_model.decision_function(query_features))
        recovered_decision = np.asarray(
            recovered_model.decision_function(query_features)
        )
        clean_accuracy = float(np.mean(clean_predictions == query_labels))
        corrupt_accuracy = float(np.mean(corrupt_predictions == query_labels))
        recovered_accuracy = float(
            np.mean(recovered_predictions == query_labels)
        )
        decision_max_abs_difference = float(
            np.max(np.abs(clean_decision - recovered_decision))
        )
        prediction_match = bool(
            np.array_equal(clean_predictions, recovered_predictions)
        )
        weight_recovery_max_abs_error = float(
            np.max(np.abs(state["aggregate_weights"] - 1.0))
        )
        p0_pass = bool(
            weight_recovery_max_abs_error <= 1e-12
            and prediction_match
            and abs(recovered_accuracy - clean_accuracy) <= 1e-12
            and decision_max_abs_difference <= DECISION_TOLERANCE
        )

        selected = state["selected"]
        selected_occurrences = state["selected_occurrences"]
        unselected_occurrences = ~selected_occurrences
        train_count = int(state["train_values"].shape[0])
        group_count = int(selected.size)
        rows.append(
            {
                "dataset": dataset,
                "official_train_count": train_count,
                "official_test_count": int(query_values.shape[0]),
                "series_length": int(query_values.shape[1]),
                "pattern_observation": {
                    "score": "median(abs(diff(x))) / max(1.4826*MAD(x), 1e-12)",
                    "selection": "stable-ranked top quartile of official TRAIN",
                    "label_or_test_used_for_selection": False,
                    "group_count": group_count,
                    "group_coverage": group_count / float(train_count),
                    "group_class_composition": _class_counts(
                        np, state["train_labels"][selected]
                    ),
                    "train_class_composition": _class_counts(
                        np, state["train_labels"]
                    ),
                    "selected_score_min": float(
                        np.min(state["roughness_scores"][selected])
                    ),
                    "selected_score_max": float(
                        np.max(state["roughness_scores"][selected])
                    ),
                },
                "program": {
                    "name": "inverse-multiplicity-sample-weight-v1",
                    "multiplicity": MULTIPLICITY,
                    "corrupt_train_row_count": int(
                        state["corrupt_values"].shape[0]
                    ),
                    "clean_total_weight_mass": float(train_count),
                    "corrupt_unweighted_total_mass": float(
                        state["corrupt_values"].shape[0]
                    ),
                    "corrupt_unweighted_pattern_group_mass": int(
                        np.count_nonzero(selected_occurrences)
                    ),
                    "recovered_total_weight_mass": float(
                        np.sum(state["weights"])
                    ),
                    "recovered_pattern_group_mass": float(
                        np.sum(state["weights"][selected_occurrences])
                    ),
                    "recovered_non_group_mass": float(
                        np.sum(state["weights"][unselected_occurrences])
                    ),
                    "aggregate_weight_per_original_min": float(
                        np.min(state["aggregate_weights"])
                    ),
                    "aggregate_weight_per_original_max": float(
                        np.max(state["aggregate_weights"])
                    ),
                    "aggregate_weight_max_abs_error_from_one": weight_recovery_max_abs_error,
                },
                "query_readout": {
                    "clean_accuracy": clean_accuracy,
                    "corrupt_unweighted_accuracy": corrupt_accuracy,
                    "recovered_weighted_accuracy": recovered_accuracy,
                    "recovered_minus_clean": recovered_accuracy - clean_accuracy,
                    "recovered_minus_corrupt": recovered_accuracy
                    - corrupt_accuracy,
                    "recovered_clean_prediction_match": prediction_match,
                    "recovered_clean_decision_max_abs_difference": decision_max_abs_difference,
                },
                "p0_exact_objective_recovery_pass": p0_pass,
                "consumer_fit_count": 3,
            }
        )

    gains = [float(row["query_readout"]["recovered_minus_corrupt"]) for row in rows]
    p0_all = all(bool(row["p0_exact_objective_recovery_pass"]) for row in rows)
    positive_count = sum(gain > 0.0 for gain in gains)
    macro_gain = sum(gains) / len(gains)
    gate_pass = bool(p0_all and positive_count >= 2 and macro_gain > 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "W58 development-only Pattern-Mass Program headroom",
        "causal_hypothesis": (
            "A TRAIN-visible high-local-roughness Pattern group duplicated to fixed "
            "multiplicity eight biases cohort mass, while inverse-multiplicity "
            "sample weighting recovers the clean objective and improves the corrupted "
            "unweighted Consumer on at least two exposed UCR backgrounds."
        ),
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "fresh_promotion_evidence": False,
        "datasets": rows,
        "overall": {
            "dataset_count": len(rows),
            "p0_exact_objective_recovery_all_datasets": p0_all,
            "positive_recovered_vs_corrupt_dataset_count": positive_count,
            "negative_recovered_vs_corrupt_dataset_count": sum(
                gain < 0.0 for gain in gains
            ),
            "macro_recovered_minus_corrupt_accuracy": macro_gain,
            "consumer_fit_count": consumer_fit_count,
            "test_split_load_count": test_split_load_count,
            "frozen_gate_pass": gate_pass,
        },
        "verdict": (
            "DEVELOPMENT_PATTERN_MASS_HEADROOM_PASS"
            if gate_pass
            else "DEVELOPMENT_PATTERN_MASS_HEADROOM_FAIL"
        ),
        "persistent_memory_built": False,
        "target_query_opened": False,
        "capability_promoted": False,
        "claim_limit": (
            "All four TEST splits were exposed before W58. This is controlled "
            "development headroom for one Pattern-conditioned reweighting Program, "
            "not fresh promotion, natural-defect evidence, or cross-dataset transfer."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or root / DEFAULT_REPORT_PATH
    payload = evaluate(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
