"""Run the W59 development-only class-mass-controlled Pattern diagnostic.

W58 duplicated the TRAIN-visible high-roughness top-quartile Pattern group and
showed recoverable Consumer headroom, but that group was class-skewed.  W59
keeps every label's total training weight equal to its clean count while the
selected Pattern remains eight-fold overrepresented relative to unselected
examples inside the same class.  This isolates within-class Pattern-mass bias
from a changed class prior.

All four UCR TEST splits were exposed before W59.  This is a development causal
diagnostic, not fresh evidence, Capability promotion, or Target transfer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-pattern-mass-class-control/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_pattern_mass_class_control_report.json"
)


def _class_mass_controlled_weights(
    np: Any,
    clean_labels: Any,
    corrupt_labels: Any,
) -> tuple[Any, dict[str, dict[str, float | int]]]:
    """Preserve clean label mass while retaining within-class multiplicity."""

    clean = np.asarray(clean_labels)
    corrupt = np.asarray(corrupt_labels)
    weights = np.zeros(corrupt.shape[0], dtype=np.float64)
    summary: dict[str, dict[str, float | int]] = {}
    clean_classes = np.unique(clean)
    if not np.array_equal(clean_classes, np.unique(corrupt)):
        raise ValueError("corruption changed the observed class set")
    for label in clean_classes:
        clean_count = int(np.count_nonzero(clean == label))
        corrupt_count = int(np.count_nonzero(corrupt == label))
        if clean_count <= 0 or corrupt_count < clean_count:
            raise ValueError("invalid class occurrence counts")
        scale = clean_count / float(corrupt_count)
        mask = corrupt == label
        weights[mask] = scale
        controlled_mass = float(np.sum(weights[mask]))
        summary[str(int(label))] = {
            "clean_mass": float(clean_count),
            "corrupt_occurrence_count": corrupt_count,
            "uniform_class_scale": scale,
            "controlled_corrupt_mass": controlled_mass,
            "mass_error": controlled_mass - float(clean_count),
        }
    if not np.isfinite(weights).all() or bool(np.any(weights <= 0.0)):
        raise ValueError("invalid class-mass-control weights")
    return weights, summary


def evaluate(root: Path) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_pattern_mass_multiplicity_headroom import (
        DATASETS,
        DATA_DIR,
        DECISION_TOLERANCE,
        MULTIPLICITY,
        PATTERN_FRACTION,
        RIDGE_ALPHA,
        _class_counts,
        _duplicate_pattern_group,
        _roughness_scores,
        _top_quartile_indices,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _features,
        _load_split,
    )

    def fit(values: Any, labels: Any, sample_weight: Any = None) -> Any:
        model = RidgeClassifier(alpha=RIDGE_ALPHA)
        model.fit(_features(np, values), labels, sample_weight=sample_weight)
        return model

    # Compile the frozen Pattern group and both weight Programs from TRAIN
    # before any exposed TEST split is opened.
    prepared: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        archive = root / DATA_DIR / f"{dataset}.zip"
        train_values, train_labels = _load_split(np, archive, dataset, "TRAIN")
        roughness_scores = _roughness_scores(np, train_values)
        selected = _top_quartile_indices(np, roughness_scores)
        corrupt_values, corrupt_labels, origins, recovered_weights = (
            _duplicate_pattern_group(
                np, train_values, train_labels, selected
            )
        )
        controlled_weights, class_mass = _class_mass_controlled_weights(
            np, train_labels, corrupt_labels
        )
        selected_occurrences = np.isin(origins, selected)
        for label_key, class_row in class_mass.items():
            label = int(label_key)
            class_occurrences = corrupt_labels == label
            selected_class_occurrences = class_occurrences & selected_occurrences
            unselected_class_occurrences = class_occurrences & ~selected_occurrences
            class_row.update(
                {
                    "selected_original_count": int(
                        np.count_nonzero(train_labels[selected] == label)
                    ),
                    "selected_occurrence_count": int(
                        np.count_nonzero(selected_class_occurrences)
                    ),
                    "unselected_occurrence_count": int(
                        np.count_nonzero(unselected_class_occurrences)
                    ),
                    "controlled_selected_mass": float(
                        np.sum(controlled_weights[selected_class_occurrences])
                    ),
                    "controlled_unselected_mass": float(
                        np.sum(controlled_weights[unselected_class_occurrences])
                    ),
                }
            )
        recovered_mass = np.bincount(
            origins,
            weights=recovered_weights,
            minlength=train_values.shape[0],
        )
        prepared[dataset] = {
            "archive": archive,
            "train_values": train_values,
            "train_labels": train_labels,
            "roughness_scores": roughness_scores,
            "selected": selected,
            "corrupt_values": corrupt_values,
            "corrupt_labels": corrupt_labels,
            "controlled_weights": controlled_weights,
            "class_mass": class_mass,
            "recovered_weights": recovered_weights,
            "recovered_mass": recovered_mass,
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

        clean_model = fit(state["train_values"], state["train_labels"])
        controlled_corrupt_model = fit(
            state["corrupt_values"],
            state["corrupt_labels"],
            state["controlled_weights"],
        )
        recovered_model = fit(
            state["corrupt_values"],
            state["corrupt_labels"],
            state["recovered_weights"],
        )
        consumer_fit_count += 3

        query_features = _features(np, query_values)
        clean_predictions = clean_model.predict(query_features)
        controlled_predictions = controlled_corrupt_model.predict(query_features)
        recovered_predictions = recovered_model.predict(query_features)
        clean_decision = np.asarray(clean_model.decision_function(query_features))
        recovered_decision = np.asarray(
            recovered_model.decision_function(query_features)
        )
        clean_accuracy = float(np.mean(clean_predictions == query_labels))
        controlled_accuracy = float(
            np.mean(controlled_predictions == query_labels)
        )
        recovered_accuracy = float(
            np.mean(recovered_predictions == query_labels)
        )
        decision_max_abs_difference = float(
            np.max(np.abs(clean_decision - recovered_decision))
        )
        recovered_prediction_match = bool(
            np.array_equal(clean_predictions, recovered_predictions)
        )
        recovered_mass_max_error = float(
            np.max(np.abs(state["recovered_mass"] - 1.0))
        )
        class_mass_max_error = max(
            abs(float(row["mass_error"])) for row in state["class_mass"].values()
        )
        p0_recovery = bool(
            recovered_mass_max_error <= 1e-12
            and recovered_prediction_match
            and abs(recovered_accuracy - clean_accuracy) <= 1e-12
            and decision_max_abs_difference <= DECISION_TOLERANCE
        )
        class_mass_preserved = bool(class_mass_max_error <= 1e-12)
        selected = state["selected"]
        train_count = int(state["train_values"].shape[0])
        rows.append(
            {
                "dataset": dataset,
                "official_train_count": train_count,
                "official_test_count": int(query_values.shape[0]),
                "series_length": int(query_values.shape[1]),
                "pattern_observation": {
                    "score": "median(abs(diff(x))) / max(1.4826*MAD(x), 1e-12)",
                    "selection": "stable-ranked top quartile of official TRAIN",
                    "pattern_fraction": PATTERN_FRACTION,
                    "label_or_test_used_for_selection": False,
                    "group_count": int(selected.size),
                    "group_coverage": float(selected.size) / train_count,
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
                "controlled_corrupt_program": {
                    "name": "within-class-pattern-mass-overrepresentation-v1",
                    "multiplicity": MULTIPLICITY,
                    "corrupt_train_row_count": int(
                        state["corrupt_values"].shape[0]
                    ),
                    "class_mass_control": state["class_mass"],
                    "class_mass_max_abs_error": class_mass_max_error,
                    "class_mass_preserved": class_mass_preserved,
                    "within_class_selected_to_unselected_relative_multiplicity": MULTIPLICITY,
                },
                "recovery_program": {
                    "name": "inverse-multiplicity-sample-weight-v1",
                    "aggregate_weight_per_original_min": float(
                        np.min(state["recovered_mass"])
                    ),
                    "aggregate_weight_per_original_max": float(
                        np.max(state["recovered_mass"])
                    ),
                    "aggregate_weight_max_abs_error_from_one": recovered_mass_max_error,
                },
                "query_readout": {
                    "clean_accuracy": clean_accuracy,
                    "class_mass_controlled_corrupt_accuracy": controlled_accuracy,
                    "recovered_weighted_accuracy": recovered_accuracy,
                    "recovered_minus_clean": recovered_accuracy - clean_accuracy,
                    "recovered_minus_class_mass_controlled_corrupt": (
                        recovered_accuracy - controlled_accuracy
                    ),
                    "recovered_clean_prediction_match": recovered_prediction_match,
                    "recovered_clean_decision_max_abs_difference": decision_max_abs_difference,
                },
                "class_mass_preservation_pass": class_mass_preserved,
                "p0_exact_objective_recovery_pass": p0_recovery,
                "consumer_fit_count": 3,
            }
        )

    gains = [
        float(
            row["query_readout"][
                "recovered_minus_class_mass_controlled_corrupt"
            ]
        )
        for row in rows
    ]
    class_mass_all = all(bool(row["class_mass_preservation_pass"]) for row in rows)
    p0_all = all(bool(row["p0_exact_objective_recovery_pass"]) for row in rows)
    positive_count = sum(gain > 0.0 for gain in gains)
    macro_gain = sum(gains) / len(gains)
    gate_pass = bool(
        class_mass_all and p0_all and positive_count >= 2 and macro_gain > 0.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "W59 development-only class-mass causal diagnostic",
        "causal_hypothesis": (
            "After every label's total training mass is held at its clean value, "
            "eight-fold within-class overrepresentation of the TRAIN-visible "
            "high-roughness Pattern group still harms the fixed Consumer, and the "
            "inverse-multiplicity Program recovers useful accuracy."
        ),
        "context_exposure": "INSTANCE_SEEN",
        "outcome_exposure": "EXPOSED",
        "fresh_promotion_evidence": False,
        "datasets": rows,
        "overall": {
            "dataset_count": len(rows),
            "class_mass_preservation_all_datasets": class_mass_all,
            "p0_exact_objective_recovery_all_datasets": p0_all,
            "positive_recovered_vs_controlled_corrupt_dataset_count": positive_count,
            "negative_recovered_vs_controlled_corrupt_dataset_count": sum(
                gain < 0.0 for gain in gains
            ),
            "macro_recovered_minus_class_mass_controlled_corrupt_accuracy": macro_gain,
            "consumer_fit_count": consumer_fit_count,
            "test_split_load_count": test_split_load_count,
            "frozen_gate_pass": gate_pass,
        },
        "verdict": (
            "DEVELOPMENT_PATTERN_MASS_CLASS_CONTROL_PASS"
            if gate_pass
            else "DEVELOPMENT_PATTERN_MASS_CLASS_CONTROL_FAIL"
        ),
        "persistent_memory_built": False,
        "target_query_opened": False,
        "capability_promoted": False,
        "claim_limit": (
            "The roster and TEST outcomes were already exposed. W59 only distinguishes "
            "within-class Pattern-mass headroom from class-prior effects; it is not "
            "fresh promotion, natural-defect evidence, or cross-dataset transfer."
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
