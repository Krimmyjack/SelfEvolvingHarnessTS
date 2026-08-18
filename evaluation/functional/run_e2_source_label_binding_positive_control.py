"""Run a Source-only Consumer label-binding instrumentation positive control.

The incumbent deliberately binds every clean training input to another series'
normalized 48-step label.  The oracle uses only the frozen permutation manifest to
undo that binding.  This is an instrumentation sanity check, not a repair method or
evidence of a naturally occurring defect.
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    CONTEXT_LENGTH,
    DISCOVERY_SUBSPLIT,
    EVAL_CONTEXT_BOUNDS,
    EVAL_FUTURE_BOUNDS,
    EVAL_SERIES_PER_DATASET,
    HORIZON,
    RIDGE_ALPHA,
    ROBUST_SCALE_FLOOR,
    SOURCE_DATASETS,
    TRAIN_ANCHORS,
    TRAIN_SERIES_PER_DATASET,
    RosterItem,
    _center_scale,
    _evaluation_matrices,
    _load_roster_values,
    select_roster,
)


SCHEMA_VERSION = "e2-source-label-binding-positive-control/1"
SCIENTIFIC_ROLE = "source_only_consumer_instrumentation_positive_control"
POLICIES = ("deranged_label_incumbent", "inverse_manifest_oracle")
PERMUTATION_SALT = "e2-source-label-binding-positive-control-v1"
PERMUTATION_SHIFT = 1
MEAN_DEGRADATION_MIN = 0.005
MEDIAN_DEGRADATION_MIN_EXCLUSIVE = 0.0
POSITIVE_DEGRADATION_MIN_COUNT = 5
ORACLE_MAX_ABS_ERROR_MAX = 1e-12
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_label_binding_positive_control_report.json"
)


@dataclass(frozen=True)
class DatasetTrainingBundle:
    x_train: np.ndarray
    clean_y_train: np.ndarray
    deranged_y_train: np.ndarray
    oracle_y_train: np.ndarray
    diagnostics: dict[str, object]
    permutation_manifest: list[dict[str, object]]
    p0: dict[str, object]


def _permutation_order(dataset_id: str, anchor: int, uids: list[str]) -> list[str]:
    """Order UIDs with the one fixed, value-independent permutation recipe."""

    if len(uids) != len(set(uids)):
        raise ValueError("label-binding group contains duplicate UIDs")

    def key(uid: str) -> tuple[bytes, str]:
        material = f"{PERMUTATION_SALT}|{dataset_id}|{anchor}|{uid}".encode("utf-8")
        return hashlib.sha256(material).digest(), uid

    return sorted(uids, key=key)


def _row_multiset(values: np.ndarray) -> Counter[bytes]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("label multiset input must be a matrix")
    return Counter(np.ascontiguousarray(row).tobytes() for row in array)


def _clean_training_rows(
    *,
    dataset_id: str,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], dict[str, object]]:
    """Build clean normalized contexts, zero masks, and paired clean labels."""

    items = sorted(train_items, key=lambda item: item.record.series_uid)
    uids = [item.record.series_uid for item in items]
    if len(items) != TRAIN_SERIES_PER_DATASET or len(set(uids)) != len(items):
        raise ValueError(f"unexpected training roster for {dataset_id}")
    if any(item.record.dataset_id != dataset_id for item in items):
        raise ValueError("training roster crosses datasets")

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    row_keys: list[tuple[int, str]] = []
    scale_methods: dict[str, int] = {}
    for anchor in TRAIN_ANCHORS:
        context_start = anchor - CONTEXT_LENGTH
        target_end = anchor + HORIZON
        if context_start < 0 or target_end > 928:
            raise AssertionError("training example crosses the frozen train boundary")
        for item in items:
            uid = item.record.series_uid
            values = values_by_uid[uid]
            context = np.asarray(values[context_start:anchor], dtype=np.float64).copy()
            target = np.asarray(values[anchor:target_end], dtype=np.float64).copy()
            if context.shape != (CONTEXT_LENGTH,) or target.shape != (HORIZON,):
                raise ValueError(f"insufficient clean training window: {uid}/{anchor}")
            if not np.isfinite(context).all() or not np.isfinite(target).all():
                raise ValueError(f"non-finite clean training window: {uid}/{anchor}")
            center, scale, scale_method = _center_scale(context)
            features = np.concatenate(
                (
                    (context - center) / scale,
                    np.zeros(CONTEXT_LENGTH, dtype=np.float64),
                )
            )
            normalized_target = (target - center) / scale
            if not np.isfinite(features).all() or not np.isfinite(normalized_target).all():
                raise ValueError(f"non-finite normalized training row: {uid}/{anchor}")
            x_rows.append(features)
            y_rows.append(normalized_target)
            row_keys.append((anchor, uid))
            scale_methods[scale_method] = scale_methods.get(scale_method, 0) + 1

    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    x_train = np.asarray(x_rows, dtype=np.float64)
    y_train = np.asarray(y_rows, dtype=np.float64)
    if x_train.shape != (expected, 2 * CONTEXT_LENGTH):
        raise AssertionError("unexpected clean training input shape")
    if y_train.shape != (expected, HORIZON):
        raise AssertionError("unexpected clean training target shape")
    mask_nonzero_count = int(np.count_nonzero(x_train[:, CONTEXT_LENGTH:]))
    if mask_nonzero_count != 0:
        raise AssertionError("clean training masks must be all zero")
    return x_train, y_train, row_keys, {
        "scale_method_counts": scale_methods,
        "mask_feature_nonzero_count": mask_nonzero_count,
        "context_is_clean": True,
        "center_scale_source": "same clean 192-step context as the input row",
        "target_normalization": "paired clean 48-step target with that context center/scale",
    }


def _bind_and_inverse_labels(
    *,
    dataset_id: str,
    x_train: np.ndarray,
    clean_y_train: np.ndarray,
    row_keys: list[tuple[int, str]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], dict[str, object]]:
    """Derange labels within each anchor and recover them via the inverse manifest."""

    if len(row_keys) != x_train.shape[0] or len(row_keys) != clean_y_train.shape[0]:
        raise ValueError("training rows and label-binding keys disagree")
    index_by_key = {key: index for index, key in enumerate(row_keys)}
    if len(index_by_key) != len(row_keys):
        raise ValueError("duplicate label-binding row key")

    deranged = np.full_like(clean_y_train, np.nan)
    oracle = np.full_like(clean_y_train, np.nan)
    manifests: list[dict[str, object]] = []
    group_checks: list[dict[str, object]] = []
    for anchor in TRAIN_ANCHORS:
        group_uids = [uid for row_anchor, uid in row_keys if row_anchor == anchor]
        ordered = _permutation_order(dataset_id, anchor, group_uids)
        assignments: list[dict[str, str]] = []
        for rank, recipient_uid in enumerate(ordered):
            source_uid = ordered[(rank + PERMUTATION_SHIFT) % len(ordered)]
            recipient_index = index_by_key[(anchor, recipient_uid)]
            source_index = index_by_key[(anchor, source_uid)]
            deranged[recipient_index] = clean_y_train[source_index]
            assignments.append(
                {
                    "input_series_uid": recipient_uid,
                    "bound_label_series_uid": source_uid,
                }
            )

        # Invert recipient -> source: the label at each recipient is restored to its
        # source row, yielding labels aligned with the original clean input ordering.
        for assignment in assignments:
            recipient_uid = assignment["input_series_uid"]
            source_uid = assignment["bound_label_series_uid"]
            oracle[index_by_key[(anchor, source_uid)]] = deranged[
                index_by_key[(anchor, recipient_uid)]
            ]

        anchor_indices = [index_by_key[(anchor, uid)] for uid in group_uids]
        recipients = [row["input_series_uid"] for row in assignments]
        sources = [row["bound_label_series_uid"] for row in assignments]
        fixed_point_count = sum(a == b for a, b in zip(recipients, sources))
        bijection = (
            len(assignments) == TRAIN_SERIES_PER_DATASET
            and len(set(recipients)) == TRAIN_SERIES_PER_DATASET
            and len(set(sources)) == TRAIN_SERIES_PER_DATASET
            and set(recipients) == set(group_uids)
            and set(sources) == set(group_uids)
        )
        label_multiset_unchanged = _row_multiset(deranged[anchor_indices]) == _row_multiset(
            clean_y_train[anchor_indices]
        )
        displacements = [
            float(np.mean(np.abs(deranged[index] - clean_y_train[index])))
            for index in anchor_indices
        ]
        median_displacement = statistics.median(displacements)
        group_checks.append(
            {
                "anchor": anchor,
                "bijection": bijection,
                "fixed_point_count": fixed_point_count,
                "zero_fixed_points": fixed_point_count == 0,
                "label_multiset_unchanged": label_multiset_unchanged,
                "median_normalized_label_displacement": median_displacement,
                "median_normalized_label_displacement_positive": median_displacement > 0.0,
            }
        )
        manifests.append(
            {
                "anchor": anchor,
                "row_count": len(ordered),
                "circular_shift": PERMUTATION_SHIFT,
                "permutation_ordered_series_uids": ordered,
                "assignments": assignments,
            }
        )

    if not np.isfinite(deranged).all() or not np.isfinite(oracle).all():
        raise AssertionError("label binding left unassigned target rows")

    incumbent_x = x_train.copy()
    oracle_x = x_train.copy()
    x_unchanged = np.array_equal(incumbent_x, x_train) and np.array_equal(
        oracle_x, x_train
    )
    oracle_max_abs_error = float(np.max(np.abs(oracle - clean_y_train)))
    all_bijections = all(bool(row["bijection"]) for row in group_checks)
    all_zero_fixed_points = all(bool(row["zero_fixed_points"]) for row in group_checks)
    all_label_multisets_unchanged = all(
        bool(row["label_multiset_unchanged"]) for row in group_checks
    )
    all_anchor_displacements_positive = all(
        bool(row["median_normalized_label_displacement_positive"])
        for row in group_checks
    )
    p0_pass = (
        all_bijections
        and all_zero_fixed_points
        and x_unchanged
        and all_label_multisets_unchanged
        and oracle_max_abs_error <= ORACLE_MAX_ABS_ERROR_MAX
        and all_anchor_displacements_positive
    )
    return deranged, oracle, manifests, {
        "dataset_id": dataset_id,
        "checked_before_any_consumer_fit": True,
        "consumer_fit_count_at_check": 0,
        "group_checks": group_checks,
        "all_groups_are_bijections": all_bijections,
        "all_groups_have_zero_fixed_points": all_zero_fixed_points,
        "inputs_unchanged_across_policies": x_unchanged,
        "all_group_label_multisets_unchanged": all_label_multisets_unchanged,
        "oracle_max_abs_error": oracle_max_abs_error,
        "oracle_max_abs_error_max": ORACLE_MAX_ABS_ERROR_MAX,
        "all_anchor_median_normalized_label_displacements_positive": (
            all_anchor_displacements_positive
        ),
        "pass": p0_pass,
    }


def _training_bundle(
    *,
    dataset_id: str,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
) -> DatasetTrainingBundle:
    x_train, clean_y, row_keys, diagnostics = _clean_training_rows(
        dataset_id=dataset_id,
        train_items=train_items,
        values_by_uid=values_by_uid,
    )
    deranged_y, oracle_y, manifest, p0 = _bind_and_inverse_labels(
        dataset_id=dataset_id,
        x_train=x_train,
        clean_y_train=clean_y,
        row_keys=row_keys,
    )
    return DatasetTrainingBundle(
        x_train=x_train,
        clean_y_train=clean_y,
        deranged_y_train=deranged_y,
        oracle_y_train=oracle_y,
        diagnostics=diagnostics,
        permutation_manifest=manifest,
        p0=p0,
    )


def _dataset_evidence(
    *,
    dataset_id: str,
    policy_losses: dict[str, list[float]],
    eval_uids: list[str],
    training_diagnostics: dict[str, object],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    incumbent_losses = policy_losses["deranged_label_incumbent"]
    oracle_losses = policy_losses["inverse_manifest_oracle"]
    if not (len(incumbent_losses) == len(oracle_losses) == len(eval_uids)):
        raise AssertionError("paired label-binding evaluation lengths disagree")
    paired: list[dict[str, object]] = []
    for uid, incumbent_loss, oracle_loss in zip(
        eval_uids, incumbent_losses, oracle_losses
    ):
        degradation = incumbent_loss - oracle_loss
        paired.append(
            {
                "series_uid": uid,
                "deranged_label_incumbent_normalized_mae": incumbent_loss,
                "inverse_manifest_oracle_normalized_mae": oracle_loss,
                "normalized_mae_degradation": degradation,
                "positive_degradation": degradation > 0.0,
            }
        )
    degradations = [float(row["normalized_mae_degradation"]) for row in paired]
    mean_degradation = statistics.fmean(degradations)
    median_degradation = statistics.median(degradations)
    positive_count = sum(bool(row["positive_degradation"]) for row in paired)
    gate_pass = (
        mean_degradation >= MEAN_DEGRADATION_MIN
        and median_degradation > MEDIAN_DEGRADATION_MIN_EXCLUSIVE
        and positive_count >= POSITIVE_DEGRADATION_MIN_COUNT
    )
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "scientific_unit": "dataset_level_source_instrumentation_sanity_cohort",
        "dataset_id": dataset_id,
        "policy_contrast": "deranged_label_incumbent_minus_inverse_manifest_oracle",
        "train_cohort": {
            "series_count": TRAIN_SERIES_PER_DATASET,
            "anchor_count_per_series": len(TRAIN_ANCHORS),
            "example_count": TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS),
            "diagnostics": training_diagnostics,
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
            "clean_inputs_and_zero_masks_shared_across_policies": True,
            "diagnostics": evaluation_diagnostics,
        },
        "policy_mean_normalized_mae": {
            policy: statistics.fmean(losses) for policy, losses in policy_losses.items()
        },
        "policy_median_normalized_mae": {
            policy: statistics.median(losses) for policy, losses in policy_losses.items()
        },
        "mean_normalized_mae_degradation": mean_degradation,
        "median_normalized_mae_degradation": median_degradation,
        "positive_degradation_count": positive_count,
        "paired_eval_rows": paired,
        "p1_gate": {
            "mean_degradation_min": MEAN_DEGRADATION_MIN,
            "median_degradation_must_exceed": MEDIAN_DEGRADATION_MIN_EXCLUSIVE,
            "positive_degradation_min_count": POSITIVE_DEGRADATION_MIN_COUNT,
            "eval_series_count": EVAL_SERIES_PER_DATASET,
            "pass": gate_pass,
        },
    }


def _roster_report(
    roster: list[RosterItem], selection: dict[str, object]
) -> dict[str, object]:
    members = [
        {
            "dataset_id": item.record.dataset_id,
            "series_uid": item.record.series_uid,
            "entity_id": item.record.entity_id,
            "cohort": item.cohort,
            "split_role": item.assignment.role.value,
            "subsplit": DISCOVERY_SUBSPLIT,
        }
        for item in sorted(
            roster,
            key=lambda row: (row.record.dataset_id, row.cohort, row.record.series_uid),
        )
    ]
    return {"selection": selection, "members": members}


def run_e2_source_label_binding_positive_control(
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    roster, selection = select_roster(
        registry_path=registry_path,
        split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
    )
    values_by_uid = _load_roster_values(roster, clean_root)

    bundles: dict[str, DatasetTrainingBundle] = {}
    for dataset_id in sorted(SOURCE_DATASETS):
        train_items = [
            item
            for item in roster
            if item.record.dataset_id == dataset_id and item.cohort == "train"
        ]
        bundles[dataset_id] = _training_bundle(
            dataset_id=dataset_id,
            train_items=train_items,
            values_by_uid=values_by_uid,
        )

    # P0 is global and is enforced before either dataset is allowed to fit a model.
    p0_by_dataset = {
        dataset_id: bundle.p0 for dataset_id, bundle in bundles.items()
    }
    p0_pass = all(bool(row["pass"]) for row in p0_by_dataset.values())
    if not p0_pass:
        raise RuntimeError(
            "label-binding P0 failed before fitting; consumer_fit_count=0"
        )

    evidence_rows: list[dict[str, object]] = []
    consumer_fit_count = 0
    for dataset_id in sorted(SOURCE_DATASETS):
        bundle = bundles[dataset_id]
        eval_items = [
            item
            for item in roster
            if item.record.dataset_id == dataset_id and item.cohort == "eval"
        ]
        x_eval, y_eval, eval_uids, eval_diagnostics = _evaluation_matrices(
            eval_items, values_by_uid
        )
        policy_targets = {
            "deranged_label_incumbent": bundle.deranged_y_train,
            "inverse_manifest_oracle": bundle.oracle_y_train,
        }
        policy_losses: dict[str, list[float]] = {}
        for policy in POLICIES:
            model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="svd")
            model.fit(bundle.x_train, policy_targets[policy])
            consumer_fit_count += 1
            prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
            if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}/{policy}")
            policy_losses[policy] = [
                float(loss) for loss in np.mean(np.abs(prediction - y_eval), axis=1)
            ]
        evidence_rows.append(
            _dataset_evidence(
                dataset_id=dataset_id,
                policy_losses=policy_losses,
                eval_uids=eval_uids,
                training_diagnostics=bundle.diagnostics,
                evaluation_diagnostics=eval_diagnostics,
            )
        )

    if consumer_fit_count != 4:
        raise AssertionError("expected exactly four independent Ridge Consumer fits")
    p1_by_dataset = {
        str(row["dataset_id"]): row["p1_gate"] for row in evidence_rows
    }
    p1_pass = all(bool(gate["pass"]) for gate in p1_by_dataset.values())  # type: ignore[index]
    verdict = (
        "SOURCE_LABEL_BINDING_INSTRUMENTATION_POSITIVE_CONTROL_PASS"
        if p1_pass
        else "SOURCE_LABEL_BINDING_INSTRUMENTATION_POSITIVE_CONTROL_FAIL"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "configuration": {
            "datasets": sorted(SOURCE_DATASETS),
            "split": "support_a",
            "subsplit": DISCOVERY_SUBSPLIT,
            "policies": list(POLICIES),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "train_anchors": list(TRAIN_ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "training_input": "clean context normalized with its own center/scale plus all-zero mask",
            "training_target": "clean 48-step target normalized with the same clean-context center/scale",
            "standardization": {
                "center": "median of finite clean context values",
                "primary_scale": "1.4826 * median absolute deviation",
                "fallback": "population std when >=1e-6, otherwise 1e-6",
                "scale_floor": ROBUST_SCALE_FLOOR,
            },
            "label_binding": {
                "permutation_salt": PERMUTATION_SALT,
                "ordering": "sha256(salt|dataset_id|anchor|series_uid), then series_uid tie-break",
                "scope": "32 rows independently within each dataset and anchor",
                "circular_shift": PERMUTATION_SHIFT,
                "inverse_manifest_oracle_uses_values_or_outcomes_to_choose_mapping": False,
            },
            "ridge": {
                "alpha": RIDGE_ALPHA,
                "fit_intercept": True,
                "solver": "svd",
                "expected_fit_count": 4,
            },
            "agent_enabled": False,
            "memory_enabled": False,
            "promotion_enabled": False,
            "transfer_enabled": False,
        },
        "roster": _roster_report(roster, selection),
        "p0_pre_fit_integrity_gate": {
            "dataset_results": p0_by_dataset,
            "all_datasets_required": True,
            "failure_action": "hard stop before all Ridge fits",
            "pass": p0_pass,
        },
        "permutation_manifests": {
            dataset_id: bundle.permutation_manifest
            for dataset_id, bundle in bundles.items()
        },
        "policy_intervention_evidence": evidence_rows,
        "p1_positive_control_gate": {
            "degradation_definition": (
                "deranged_label_incumbent normalized MAE minus "
                "inverse_manifest_oracle normalized MAE"
            ),
            "dataset_results": p1_by_dataset,
            "conjunction_across_both_datasets": True,
            "pass": p1_pass,
        },
        "information_wall": {
            "series_roster_fixed_before_selected_value_loading": True,
            "source_only": True,
            "locked_source_datasets": sorted(SOURCE_DATASETS),
            "support_a_discovery_only": True,
            "train_eval_series_disjoint": True,
            "permutation_uses_dataset_anchor_uid_only": True,
            "uci_values_context_or_future_read": False,
            "support_b_values_context_or_future_read": False,
            "target_values_context_or_future_read": False,
            "query_values_context_or_future_read": False,
            "target_query_opened": False,
        },
        "consumer_fit_count": consumer_fit_count,
        "chronos_judge_call_count": 0,
        "agent_enabled": False,
        "memory_enabled": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "verdict": verdict,
        "claim_limit": (
            "Instrumentation sanity only: this synthetic Source label-binding contrast "
            "is not a deployable repair, evidence of a natural defect, Capability, "
            "Agent, Memory, promotion, Target performance, or transfer evidence."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/series_registry.jsonl",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/split_manifest.json",
    )
    parser.add_argument(
        "--support-a-subsplit",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / OUTPUT_RELATIVE_PATH,
    )
    args = parser.parse_args()

    report = run_e2_source_label_binding_positive_control(
        registry_path=args.registry,
        split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
