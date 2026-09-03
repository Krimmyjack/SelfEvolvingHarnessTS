"""Test whether cohort-level preparation can change a shared Source consumer.

This is a deliberately narrow premise experiment.  It trains one deterministic
multi-output Ridge consumer per Source dataset and preparation policy, then evaluates
the consumers on series-disjoint, clean Source contexts.  It neither assigns a causal
utility delta to an individual Pattern nor reads any Target/Query surface.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
    SeriesRecord,
    read_registry_jsonl,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.split import (
    SplitAssignment,
    SplitManifest,
    SplitRole,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    _execute_program,
    _global_features,
)


SCHEMA_VERSION = "e2-source-cohort-policy-premise/1"
SCIENTIFIC_ROLE = "source_cohort_policy_intervention_premise"
SOURCE_DATASETS = ("monash:traffic_hourly", "metr_la")
DISCOVERY_SUBSPLIT = "support_a_discovery"
SUPPORT_A_SUBSPLIT_SCHEMA = "benchmark-support-a-subsplit/2"
POLICIES = ("identity_minimal", "linear", "seasonal")
NONIDENTITY_POLICIES = ("linear", "seasonal")

TRAIN_SERIES_PER_DATASET = 32
EVAL_SERIES_PER_DATASET = 8
TRAIN_ANCHORS = (544, 592, 640, 688, 736, 784, 832, 880)
CONTEXT_LENGTH = 192
HORIZON = 48
GAP_BOUNDS = (156, 180)
EVAL_CONTEXT_BOUNDS = (736, 928)
EVAL_FUTURE_BOUNDS = (928, 976)

RIDGE_ALPHA = 1.0
ROBUST_SCALE_FLOOR = 1e-6
PER_DATASET_GAIN_FLOOR = -0.005
MEAN_GAIN_MIN = 0.005
HARM_GAIN_THRESHOLD = -0.005
HARM_RATE_MAX = 0.25
MATERIAL_GAIN_MIN = 0.005


@dataclass(frozen=True)
class RosterItem:
    record: SeriesRecord
    assignment: SplitAssignment
    cohort: str


def _read_discovery_uids(path: Path) -> set[str]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") != SUPPORT_A_SUBSPLIT_SCHEMA:
        raise ValueError("unsupported Support-A subsplit schema")
    members = payload.get("members")
    if not isinstance(members, dict):
        raise ValueError("Support-A subsplit members must be an object")
    raw = members.get(DISCOVERY_SUBSPLIT)
    if not isinstance(raw, list) or not all(isinstance(uid, str) and uid for uid in raw):
        raise ValueError("invalid support_a_discovery member list")
    if len(raw) != len(set(raw)):
        raise ValueError("duplicate support_a_discovery UID")
    counts = payload.get("counts")
    if not isinstance(counts, dict) or counts.get(DISCOVERY_SUBSPLIT) != len(raw):
        raise ValueError("support_a_discovery count disagrees with frozen metadata")
    return set(raw)


def _validate_candidate(record: SeriesRecord, assignment: SplitAssignment) -> None:
    uid = record.series_uid
    if record.dataset_id != assignment.dataset_id:
        raise ValueError(f"registry/split dataset mismatch: {uid}")
    if record.regime_tag != assignment.regime_tag:
        raise ValueError(f"registry/split regime mismatch: {uid}")
    if record.admission_reasons != ():
        raise ValueError(f"ineligible Source discovery record: {uid}")
    if SplitRole.SUPPORT_A.value not in record.roles_allowed:
        raise ValueError(f"record disallows Support-A: {uid}")
    if assignment.role is not SplitRole.SUPPORT_A:
        raise ValueError(f"discovery member is not Support-A: {uid}")
    boundaries = assignment.chronological_boundaries
    if boundaries is None:
        raise ValueError(f"missing chronological boundaries: {uid}")
    if tuple(boundaries.get("train", ())) != (0, 928):
        raise ValueError(f"unexpected train boundary: {uid}")
    if tuple(boundaries.get("validation", ())) != EVAL_FUTURE_BOUNDS:
        raise ValueError(f"unexpected validation boundary: {uid}")
    if tuple(boundaries.get("test", ())) != (976, 1024):
        raise ValueError(f"unexpected test boundary: {uid}")


def select_roster(
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
) -> tuple[list[RosterItem], dict[str, object]]:
    """Freeze train/eval UIDs from metadata before any selected values are loaded."""

    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    assignments = {row.series_uid: row for row in manifest.assignments}
    discovery_uids = _read_discovery_uids(support_a_subsplit_path)

    by_dataset: dict[str, list[tuple[SeriesRecord, SplitAssignment]]] = {
        dataset_id: [] for dataset_id in SOURCE_DATASETS
    }
    for uid in discovery_uids:
        assignment = assignments.get(uid)
        if assignment is None:
            raise ValueError(f"discovery UID absent from split manifest: {uid}")
        if assignment.dataset_id not in SOURCE_DATASETS:
            continue
        record = records.get(uid)
        if record is None:
            raise ValueError(f"discovery UID absent from registry: {uid}")
        _validate_candidate(record, assignment)
        by_dataset[assignment.dataset_id].append((record, assignment))

    roster: list[RosterItem] = []
    available_by_dataset: dict[str, int] = {}
    selected_by_dataset: dict[str, dict[str, list[str]]] = {}
    required = TRAIN_SERIES_PER_DATASET + EVAL_SERIES_PER_DATASET
    for dataset_id in sorted(SOURCE_DATASETS):
        candidates = sorted(by_dataset[dataset_id], key=lambda item: item[0].series_uid)
        available_by_dataset[dataset_id] = len(candidates)
        if len(candidates) < required:
            raise ValueError(f"fewer than {required} eligible discovery series: {dataset_id}")
        train = candidates[:TRAIN_SERIES_PER_DATASET]
        evaluate = candidates[TRAIN_SERIES_PER_DATASET:required]
        roster.extend(RosterItem(record, assignment, "train") for record, assignment in train)
        roster.extend(RosterItem(record, assignment, "eval") for record, assignment in evaluate)
        selected_by_dataset[dataset_id] = {
            "train": [record.series_uid for record, _ in train],
            "eval": [record.series_uid for record, _ in evaluate],
        }

    train_uids = {item.record.series_uid for item in roster if item.cohort == "train"}
    eval_uids = {item.record.series_uid for item in roster if item.cohort == "eval"}
    if len(train_uids) != len(SOURCE_DATASETS) * TRAIN_SERIES_PER_DATASET:
        raise AssertionError("training roster contains duplicate series")
    if len(eval_uids) != len(SOURCE_DATASETS) * EVAL_SERIES_PER_DATASET:
        raise AssertionError("evaluation roster contains duplicate series")
    if train_uids & eval_uids:
        raise AssertionError("training and evaluation series overlap")

    return roster, {
        "fixed_before_selected_value_loading": True,
        "selection_rule": (
            "within each frozen Source dataset, sort eligible support_a_discovery "
            "members by series_uid; first 32 train, next 8 eval"
        ),
        "selection_features": [
            "frozen_support_a_discovery_membership",
            "frozen_registry_eligibility",
            "dataset_id",
            "series_uid",
        ],
        "regime_feature_value_future_or_outcome_used": False,
        "available_by_dataset": available_by_dataset,
        "selected_by_dataset": selected_by_dataset,
        "train_eval_series_disjoint": True,
    }


def _center_scale(values: np.ndarray) -> tuple[float, float, str]:
    observed = np.asarray(values, dtype=np.float64)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        raise ValueError("cannot standardize a context with no observed values")
    center = float(np.median(observed))
    mad_scale = 1.4826 * float(np.median(np.abs(observed - center)))
    if np.isfinite(mad_scale) and mad_scale >= ROBUST_SCALE_FLOOR:
        return center, mad_scale, "median_mad"
    std_scale = float(np.std(observed))
    if np.isfinite(std_scale) and std_scale >= ROBUST_SCALE_FLOOR:
        return center, std_scale, "median_std_fallback"
    return center, ROBUST_SCALE_FLOOR, "scale_floor_fallback"


def _standardized_features(
    raw_context: np.ndarray,
    *,
    policy: str,
    center: float,
    scale: float,
) -> tuple[np.ndarray, int | None]:
    raw = np.asarray(raw_context, dtype=np.float64)
    if raw.shape != (CONTEXT_LENGTH,):
        raise ValueError("consumer context must have length 192")
    original_mask = ~np.isfinite(raw)
    observed_period: int | None = None
    if policy == "identity_minimal":
        normalized = (raw - center) / scale
        normalized[original_mask] = 0.0
    elif policy == "linear":
        prepared = _execute_program("linear", raw, observed_period=1)
        normalized = (prepared - center) / scale
    elif policy == "seasonal":
        _, observed_period = _global_features(raw)
        prepared = _execute_program("seasonal", raw, observed_period=observed_period)
        normalized = (prepared - center) / scale
    else:
        raise ValueError(f"unknown cohort preparation policy: {policy!r}")
    if not np.isfinite(normalized).all():
        raise ValueError(f"policy {policy!r} produced non-finite consumer features")
    features = np.concatenate((normalized, original_mask.astype(np.float64)))
    if features.shape != (2 * CONTEXT_LENGTH,):
        raise AssertionError("consumer input must have 384 features")
    return features, observed_period


def _load_roster_values(
    roster: list[RosterItem], clean_root: Path
) -> dict[str, np.ndarray]:
    """Load values only after the complete train/eval roster has been frozen."""

    return _load_values([item.record for item in roster], clean_root)


def _training_matrices(
    items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    scale_methods: dict[str, int] = {}
    observed_periods: list[int] = []
    for item in items:
        values = values_by_uid[item.record.series_uid]
        for context_end in TRAIN_ANCHORS:
            context_start = context_end - CONTEXT_LENGTH
            target_end = context_end + HORIZON
            if context_start < 0 or target_end > 928:
                raise AssertionError("training anchor crosses the frozen train boundary")
            clean_context = np.asarray(
                values[context_start:context_end], dtype=np.float64
            ).copy()
            clean_target = np.asarray(values[context_end:target_end], dtype=np.float64).copy()
            if clean_context.shape != (CONTEXT_LENGTH,) or clean_target.shape != (HORIZON,):
                raise ValueError(f"insufficient training window: {item.record.series_uid}")
            if not np.isfinite(clean_context).all() or not np.isfinite(clean_target).all():
                raise ValueError(
                    f"natural missingness enters training window: {item.record.series_uid}"
                )
            corrupt = clean_context.copy()
            corrupt[slice(*GAP_BOUNDS)] = np.nan
            if int(np.isnan(corrupt).sum()) != GAP_BOUNDS[1] - GAP_BOUNDS[0]:
                raise AssertionError("training corruption must inject exactly 24 NaNs")
            center, scale, scale_method = _center_scale(corrupt)
            features, observed_period = _standardized_features(
                corrupt, policy=policy, center=center, scale=scale
            )
            target = (clean_target - center) / scale
            if not np.isfinite(target).all():
                raise ValueError("standardized training target is non-finite")
            x_rows.append(features)
            y_rows.append(target)
            scale_methods[scale_method] = scale_methods.get(scale_method, 0) + 1
            if observed_period is not None:
                observed_periods.append(observed_period)
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    if x.shape != (expected, 384) or y.shape != (expected, HORIZON):
        raise AssertionError("unexpected cohort training matrix shape")
    diagnostics: dict[str, object] = {"scale_method_counts": scale_methods}
    if observed_periods:
        diagnostics["seasonal_observed_period_range"] = [
            min(observed_periods),
            max(observed_periods),
        ]
    return x, y, diagnostics


def _evaluation_matrices(
    items: list[RosterItem], values_by_uid: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, object]]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    uids: list[str] = []
    scale_methods: dict[str, int] = {}
    for item in items:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        context = np.asarray(values[slice(*EVAL_CONTEXT_BOUNDS)], dtype=np.float64).copy()
        future = np.asarray(values[slice(*EVAL_FUTURE_BOUNDS)], dtype=np.float64).copy()
        if context.shape != (CONTEXT_LENGTH,) or future.shape != (HORIZON,):
            raise ValueError(f"insufficient evaluation window: {uid}")
        if not np.isfinite(context).all() or not np.isfinite(future).all():
            raise ValueError(f"natural missingness enters evaluation window: {uid}")
        center, scale, scale_method = _center_scale(context)
        normalized = (context - center) / scale
        # Every trained policy receives this exact same clean context and all-zero mask.
        features = np.concatenate((normalized, np.zeros(CONTEXT_LENGTH, dtype=np.float64)))
        target = (future - center) / scale
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError(f"non-finite normalized evaluation data: {uid}")
        x_rows.append(features)
        y_rows.append(target)
        uids.append(uid)
        scale_methods[scale_method] = scale_methods.get(scale_method, 0) + 1
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    if x.shape != (EVAL_SERIES_PER_DATASET, 384) or y.shape != (
        EVAL_SERIES_PER_DATASET,
        HORIZON,
    ):
        raise AssertionError("unexpected cohort evaluation matrix shape")
    return x, y, uids, {"scale_method_counts": scale_methods}


def _policy_evidence(
    *,
    dataset_id: str,
    policy: str,
    losses: list[float],
    identity_losses: list[float],
    eval_uids: list[str],
    training_diagnostics: dict[str, object],
    evaluation_diagnostics: dict[str, object],
) -> dict[str, object]:
    if len(losses) != len(identity_losses) or len(losses) != len(eval_uids):
        raise AssertionError("paired cohort evidence lengths disagree")
    paired: list[dict[str, object]] = []
    for uid, loss, identity_loss in zip(eval_uids, losses, identity_losses):
        gain = identity_loss - loss
        paired.append(
            {
                "series_uid": uid,
                "normalized_mae": loss,
                "identity_minimal_normalized_mae": identity_loss,
                "gain_over_identity": gain,
                "harmed": gain < HARM_GAIN_THRESHOLD,
                "material_gain": gain >= MATERIAL_GAIN_MIN,
            }
        )
    mean_loss = statistics.fmean(losses)
    identity_mean = statistics.fmean(identity_losses)
    return {
        "evidence_type": "PolicyInterventionEvidence",
        "dataset_id": dataset_id,
        "policy": policy,
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
            "input_dimension": 384,
            "output_dimension": HORIZON,
            "random_training_or_tuning": False,
        },
        "eval_cohort": {
            "series_count": EVAL_SERIES_PER_DATASET,
            "context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "future_bounds": list(EVAL_FUTURE_BOUNDS),
            "clean_context_shared_across_policies": True,
            "diagnostics": evaluation_diagnostics,
        },
        "mean_normalized_mae": mean_loss,
        "identity_minimal_mean_normalized_mae": identity_mean,
        "gain_over_identity": identity_mean - mean_loss,
        "harm_rate": sum(bool(row["harmed"]) for row in paired) / len(paired),
        "paired_eval_series": paired,
    }


def run_e2_source_cohort_policy_premise(
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

    evidence_rows: list[dict[str, object]] = []
    consumer_fit_count = 0
    for dataset_id in sorted(SOURCE_DATASETS):
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
            eval_items, values_by_uid
        )
        policy_losses: dict[str, list[float]] = {}
        training_diagnostics: dict[str, dict[str, object]] = {}
        for policy in POLICIES:
            x_train, y_train, diagnostics = _training_matrices(
                train_items, values_by_uid, policy=policy
            )
            model = Ridge(
                alpha=RIDGE_ALPHA,
                fit_intercept=True,
                solver="svd",
            )
            model.fit(x_train, y_train)
            consumer_fit_count += 1
            prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
            if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}/{policy}")
            policy_losses[policy] = [
                float(value) for value in np.mean(np.abs(prediction - y_eval), axis=1)
            ]
            training_diagnostics[policy] = diagnostics

        identity_losses = policy_losses["identity_minimal"]
        for policy in POLICIES:
            evidence_rows.append(
                _policy_evidence(
                    dataset_id=dataset_id,
                    policy=policy,
                    losses=policy_losses[policy],
                    identity_losses=identity_losses,
                    eval_uids=eval_uids,
                    training_diagnostics=training_diagnostics[policy],
                    evaluation_diagnostics=eval_diagnostics,
                )
            )

    if consumer_fit_count != len(SOURCE_DATASETS) * len(POLICIES):
        raise AssertionError("expected exactly six independent Consumer fits")

    evidence_by_key = {
        (str(row["dataset_id"]), str(row["policy"])): row for row in evidence_rows
    }
    policy_gates: dict[str, dict[str, object]] = {}
    qualifying: list[str] = []
    for policy in NONIDENTITY_POLICIES:
        rows = [evidence_by_key[(dataset_id, policy)] for dataset_id in sorted(SOURCE_DATASETS)]
        dataset_gains = {
            str(row["dataset_id"]): float(row["gain_over_identity"]) for row in rows
        }
        mean_gain = statistics.fmean(dataset_gains.values())
        paired_rows = [
            paired
            for row in rows
            for paired in row["paired_eval_series"]  # type: ignore[union-attr]
        ]
        harm_rate = sum(bool(row["harmed"]) for row in paired_rows) / len(paired_rows)
        passed = (
            all(gain >= PER_DATASET_GAIN_FLOOR for gain in dataset_gains.values())
            and mean_gain >= MEAN_GAIN_MIN
            and harm_rate <= HARM_RATE_MAX
        )
        policy_gates[policy] = {
            "gain_by_dataset": dataset_gains,
            "per_dataset_gain_floor": PER_DATASET_GAIN_FLOOR,
            "dataset_equal_weight_mean_gain": mean_gain,
            "mean_gain_min": MEAN_GAIN_MIN,
            "pooled_eval_harm_rate": harm_rate,
            "harm_definition": f"gain < {HARM_GAIN_THRESHOLD}",
            "harm_rate_max": HARM_RATE_MAX,
            "pass": passed,
        }
        if passed:
            qualifying.append(policy)

    qualifying.sort(
        key=lambda policy: (
            -float(policy_gates[policy]["dataset_equal_weight_mean_gain"]),
            policy,
        )
    )
    selected_policy = qualifying[0] if qualifying else None
    verdict = (
        "COHORT_POLICY_PREMISE_PRESENT"
        if selected_policy is not None
        else "COHORT_POLICY_PREMISE_WEAK"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "configuration": {
            "datasets": sorted(SOURCE_DATASETS),
            "split": SplitRole.SUPPORT_A.value,
            "subsplit": DISCOVERY_SUBSPLIT,
            "policies": list(POLICIES),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "train_anchors": list(TRAIN_ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "train_gap_relative_to_context": list(GAP_BOUNDS),
            "eval_context_bounds": list(EVAL_CONTEXT_BOUNDS),
            "eval_future_bounds": list(EVAL_FUTURE_BOUNDS),
            "standardization": {
                "center": "median of original finite context values",
                "primary_scale": "1.4826 * median absolute deviation",
                "fallback": (
                    "if primary scale < 1e-6, use observed population std when >=1e-6; "
                    "otherwise use 1e-6"
                ),
                "scale_floor": ROBUST_SCALE_FLOOR,
            },
            "original_missing_mask_appended": True,
            "consumer_input_dimension": 384,
            "agent_enabled": False,
            "memory_enabled": False,
            "adaptation_enabled": False,
        },
        "roster_selection": selection,
        "information_wall": {
            "series_split_fixed_before_value_loading": True,
            "train_eval_series_disjoint": True,
            "train_targets_end_at_or_before_index": 928,
            "eval_future_loaded_only_after_complete_roster_freeze": True,
            "source_only": True,
            "support_b_values_context_or_future_read": False,
            "uci_values_context_or_future_read": False,
            "target_or_query_read": False,
            "target_query_opened": False,
        },
        "consumer_fit_count": consumer_fit_count,
        "chronos_judge_call_count": 0,
        "policy_intervention_evidence": evidence_rows,
        "premise_gate": {
            "policy_results": policy_gates,
            "qualifying_nonidentity_policies": qualifying,
            "selected_policy": selected_policy,
            "pass": selected_policy is not None,
        },
        "verdict": verdict,
        "promotion_eligible": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most evidence that a Source-only cohort preparation intervention changes "
            "a shared fixed Consumer under this frozen protocol; not per-series causal "
            "utility, Capability, Memory, Target performance, adaptation, promotion, or "
            "cross-dataset transfer evidence."
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
        default=project_root / "artifacts/functional/e2/source_cohort_policy_premise_report.json",
    )
    args = parser.parse_args()

    report = run_e2_source_cohort_policy_premise(
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
