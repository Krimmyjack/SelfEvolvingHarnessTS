"""Measure whether fixed outlier corruption creates readable Consumer headroom.

This exposed-development Source diagnostic compares consumers trained on clean input
or the paired fixed four-spike corruption.  It evaluates no repair, Witness, Memory,
Target, Query, promotion, or transfer behavior.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    CONTEXT_LENGTH,
    EVAL_CONTEXT_BOUNDS,
    EVAL_FUTURE_BOUNDS,
    EVAL_SERIES_PER_DATASET,
    HORIZON,
    RIDGE_ALPHA,
    SOURCE_DATASETS,
    TRAIN_ANCHORS,
    TRAIN_SERIES_PER_DATASET,
    _center_scale,
    _evaluation_matrices,
    _load_roster_values,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_outlier_local_behavior_audit import (
    PREMISE_REPORT_RELATIVE_PATH,
    SPIKE_POSITIONS,
    RosterItem,
    _inject_spikes,
    _read_premise_roster,
)


SCHEMA_VERSION = "e2-source-outlier-consumer-headroom/1"
SCIENTIFIC_ROLE = "development_source_outlier_consumer_headroom_blocker_diagnostic"
POLICIES = ("clean_input_control", "corrupt_identity")
MATERIAL_DEGRADATION_MIN = 0.005
MIN_POSITIVE_DEGRADATION_COUNT = 5
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_outlier_consumer_headroom_report.json"
)


def _training_row(
    *, clean: np.ndarray, corrupt: np.ndarray, target: np.ndarray, policy: str
) -> tuple[np.ndarray, np.ndarray, str]:
    """Create one row with normalization derived only from the corrupt context."""

    clean_values = np.asarray(clean, dtype=np.float64)
    corrupt_values = np.asarray(corrupt, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if clean_values.shape != (CONTEXT_LENGTH,) or corrupt_values.shape != (
        CONTEXT_LENGTH,
    ):
        raise ValueError("headroom training contexts must have length 192")
    if target_values.shape != (HORIZON,):
        raise ValueError("headroom training target must have length 48")
    if not all(
        np.isfinite(values).all()
        for values in (clean_values, corrupt_values, target_values)
    ):
        raise ValueError("headroom training row must be finite")
    center, scale, scale_method = _center_scale(corrupt_values)
    if policy == "clean_input_control":
        raw = clean_values
    elif policy == "corrupt_identity":
        raw = corrupt_values
    else:
        raise ValueError(f"unknown outlier headroom policy: {policy}")
    features = np.concatenate(
        (
            (raw - center) / scale,
            np.zeros(CONTEXT_LENGTH, dtype=np.float64),
        )
    )
    normalized_target = (target_values - center) / scale
    if not np.isfinite(features).all() or not np.isfinite(normalized_target).all():
        raise ValueError("non-finite standardized headroom row")
    return features, normalized_target, scale_method


def _training_matrices(
    *,
    dataset_id: str,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
    policy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    scale_methods: dict[str, int] = {}
    for item in train_items:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        for anchor in TRAIN_ANCHORS:
            start = anchor - CONTEXT_LENGTH
            target_end = anchor + HORIZON
            if start < 0 or target_end > 928:
                raise AssertionError("headroom example crosses frozen train boundary")
            clean = np.asarray(values[start:anchor], dtype=np.float64).copy()
            target = np.asarray(values[anchor:target_end], dtype=np.float64).copy()
            corrupt, _hidden = _inject_spikes(
                clean,
                dataset_id=dataset_id,
                entity_id=item.record.entity_id,
                anchor=anchor,
            )
            features, normalized_target, scale_method = _training_row(
                clean=clean,
                corrupt=corrupt,
                target=target,
                policy=policy,
            )
            x_rows.append(features)
            y_rows.append(normalized_target)
            scale_methods[scale_method] = scale_methods.get(scale_method, 0) + 1
    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    if x.shape != (expected, 2 * CONTEXT_LENGTH) or y.shape != (expected, HORIZON):
        raise AssertionError("unexpected outlier headroom training matrix shape")
    mask_nonzero_count = int(np.count_nonzero(x[:, CONTEXT_LENGTH:]))
    if mask_nonzero_count != 0:
        raise AssertionError("all outlier headroom mask features must be zero")
    return x, y, {
        "scale_method_counts_from_corrupt_context": scale_methods,
        "center_scale_and_target_normalization_source": "paired_corrupt_context",
        "mask_feature_nonzero_count": mask_nonzero_count,
    }


def _dataset_evidence(
    *,
    dataset_id: str,
    policy_losses: dict[str, list[float]],
    eval_uids: list[str],
    training_diagnostics: dict[str, dict[str, object]],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    clean_losses = policy_losses["clean_input_control"]
    corrupt_losses = policy_losses["corrupt_identity"]
    if not (len(clean_losses) == len(corrupt_losses) == len(eval_uids)):
        raise AssertionError("paired headroom evidence lengths disagree")
    paired: list[dict[str, object]] = []
    for uid, clean_loss, corrupt_loss in zip(eval_uids, clean_losses, corrupt_losses):
        degradation = corrupt_loss - clean_loss
        paired.append(
            {
                "diagnostic_role": "paired_eval_series_diagnostic_not_individual_causal_evidence",
                "series_uid": uid,
                "clean_input_control_normalized_mae": clean_loss,
                "corrupt_identity_normalized_mae": corrupt_loss,
                "corruption_degradation": degradation,
                "positive_degradation": degradation > 0.0,
                "material_degradation": degradation >= MATERIAL_DEGRADATION_MIN,
            }
        )
    degradations = [float(row["corruption_degradation"]) for row in paired]
    mean_losses = {
        policy: statistics.fmean(losses) for policy, losses in policy_losses.items()
    }
    median_losses = {
        policy: statistics.median(losses) for policy, losses in policy_losses.items()
    }
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_exposed_development_premise_cohort",
        "dataset_id": dataset_id,
        "policy_mean_normalized_mae": mean_losses,
        "policy_median_normalized_mae": median_losses,
        "mean_corruption_degradation": statistics.fmean(degradations),
        "median_corruption_degradation": statistics.median(degradations),
        "positive_degradation_count": sum(
            bool(row["positive_degradation"]) for row in paired
        ),
        "material_degradation_count": sum(
            bool(row["material_degradation"]) for row in paired
        ),
        "material_degradation_threshold": MATERIAL_DEGRADATION_MIN,
        "train_cohort": {
            "series_count": TRAIN_SERIES_PER_DATASET,
            "anchor_count_per_series": len(TRAIN_ANCHORS),
            "example_count": TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS),
            "diagnostics_by_policy": training_diagnostics,
        },
        "consumer_spec": {
            "class": "sklearn.linear_model.Ridge",
            "alpha": RIDGE_ALPHA,
            "fit_intercept": True,
            "solver": "svd",
            "input_dimension": 2 * CONTEXT_LENGTH,
            "output_dimension": HORIZON,
            "random_training_or_tuning": False,
        },
        "eval_cohort": {
            "series_count": EVAL_SERIES_PER_DATASET,
            "context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "future_bounds": list(EVAL_FUTURE_BOUNDS),
            "clean_input_and_zero_mask_shared_across_both_models": True,
            "diagnostics": evaluation_diagnostics,
        },
        "paired_eval_series_diagnostics": paired,
    }


def run_e2_source_outlier_consumer_headroom(
    *, premise_report_path: Path, registry_path: Path, clean_root: Path
) -> dict[str, object]:
    roster, roster_report = _read_premise_roster(
        premise_report_path=premise_report_path,
        registry_path=registry_path,
    )
    values_by_uid = _load_roster_values(roster, clean_root)  # type: ignore[arg-type]
    evidence_rows: list[dict[str, object]] = []
    consumer_fit_count = 0
    for dataset_id in SOURCE_DATASETS:
        train_items = [
            item
            for item in roster
            if item.record.dataset_id == dataset_id and item.cohort == "train"
        ]
        eval_items = [
            item
            for item in roster
            if item.record.dataset_id == dataset_id and item.cohort == "eval"
        ]
        x_eval, y_eval, eval_uids, eval_diagnostics = _evaluation_matrices(
            eval_items, values_by_uid  # type: ignore[arg-type]
        )
        policy_losses: dict[str, list[float]] = {}
        training_diagnostics: dict[str, dict[str, object]] = {}
        reference_y_train: np.ndarray | None = None
        for policy in POLICIES:
            x_train, y_train, diagnostics = _training_matrices(
                dataset_id=dataset_id,
                train_items=train_items,
                values_by_uid=values_by_uid,
                policy=policy,
            )
            if reference_y_train is None:
                reference_y_train = y_train.copy()
            elif not np.array_equal(y_train, reference_y_train):
                raise AssertionError("paired policies must share normalized targets")
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
            model.fit(x_train, y_train)
            consumer_fit_count += 1
            prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
            if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}/{policy}")
            policy_losses[policy] = [
                float(loss) for loss in np.mean(np.abs(prediction - y_eval), axis=1)
            ]
            training_diagnostics[policy] = diagnostics
        evidence_rows.append(
            _dataset_evidence(
                dataset_id=dataset_id,
                policy_losses=policy_losses,
                eval_uids=eval_uids,
                training_diagnostics=training_diagnostics,
                evaluation_diagnostics=eval_diagnostics,
            )
        )

    if consumer_fit_count != 4:
        raise AssertionError("expected exactly four independent Consumer fits")
    gate_by_dataset: dict[str, dict[str, object]] = {}
    for row in evidence_rows:
        dataset_id = str(row["dataset_id"])
        mean_degradation = float(row["mean_corruption_degradation"])
        median_degradation = float(row["median_corruption_degradation"])
        positive_count = int(row["positive_degradation_count"])
        checks = {
            "mean_degradation_at_least_0_005": mean_degradation
            >= MATERIAL_DEGRADATION_MIN,
            "median_degradation_strictly_positive": median_degradation > 0.0,
            "positive_degradation_count_at_least_5_of_8": positive_count
            >= MIN_POSITIVE_DEGRADATION_COUNT,
        }
        gate_by_dataset[dataset_id] = {
            "mean_degradation": mean_degradation,
            "median_degradation": median_degradation,
            "positive_degradation_count": positive_count,
            "checks": checks,
            "pass": all(checks.values()),
        }
    passed = all(bool(row["pass"]) for row in gate_by_dataset.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "policies": list(POLICIES),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "train_anchors": list(TRAIN_ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "spike_positions_relative_to_context": list(SPIKE_POSITIONS),
            "corruption_reused_from": (
                "run_e2_source_outlier_local_behavior_audit._inject_spikes"
            ),
            "eval_context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "eval_future_bounds": list(EVAL_FUTURE_BOUNDS),
            "all_mask_features_zero": True,
            "training_targets_identical_across_policies": True,
            "normalization_contract": (
                "within each training example, both policies use center, scale, and "
                "target normalization derived from the paired corrupt context"
            ),
        },
        "roster": roster_report,
        "information_wall": {
            "global_registry_metadata_loaded": True,
            "only_exposed_premise_roster_source_values_loaded": True,
            "premise_policy_evidence_or_outcomes_consulted": False,
            "fresh_replay_report_read": False,
            "support_a_validation_values_or_context_or_future_read": False,
            "support_b_values_or_context_or_future_read": False,
            "uci_target_or_query_values_or_context_or_future_read": False,
            "eval_future_used_only_for_post_fit_fixed_premise_evaluation": True,
            "repair_witness_or_memory_evaluated": False,
        },
        "consumer_fit_count": consumer_fit_count,
        "expected_consumer_fit_count": 4,
        "chronos_judge_call_count": 0,
        "policy_intervention_evidence": evidence_rows,
        "headroom_gate": {
            "thresholds_frozen_before_execution": True,
            "exact_definition": (
                "pass iff each dataset has mean corruption degradation>=0.005, "
                "median corruption degradation>0, and at least 5 of 8 paired eval "
                "rows have strictly positive degradation"
            ),
            "material_degradation_threshold": MATERIAL_DEGRADATION_MIN,
            "minimum_positive_degradation_count": MIN_POSITIVE_DEGRADATION_COUNT,
            "dataset_results": gate_by_dataset,
            "pass": passed,
        },
        "verdict": (
            "OUTLIER_CONSUMER_HEADROOM_PRESENT"
            if passed
            else "OUTLIER_CONSUMER_HEADROOM_WEAK"
        ),
        "promotion": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "query": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most whether this fixed exposed-development Source Consumer/benchmark "
            "protocol has readable headroom under one frozen four-spike corruption; "
            "not evidence about any repair, Witness, Memory, individual causal effect, "
            "Capability, promotion, formal transfer, Target, or Query behavior."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--premise-report",
        type=Path,
        default=project_root / PREMISE_REPORT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/series_registry.jsonl",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument("--output", type=Path, default=project_root / OUTPUT_RELATIVE_PATH)
    args = parser.parse_args()
    report = run_e2_source_outlier_consumer_headroom(
        premise_report_path=args.premise_report,
        registry_path=args.registry,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
