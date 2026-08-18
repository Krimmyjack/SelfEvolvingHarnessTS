"""Audit local outlier observer/operator behavior on exposed Source contexts.

This zero-fit development screen applies one fixed public-feature binding and one
fixed sparse Hampel repair to synthetic impulsive spikes.  Hidden spike geometry and
clean truth are available only to the private grader.  The result is operator/observer
feasibility evidence, not downstream Consumer utility, Capability, promotion, Target,
Query, or transfer evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
    SeriesRecord,
    read_registry_jsonl,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    CONTEXT_LENGTH,
    EVAL_CONTEXT_BOUNDS,
    EVAL_FUTURE_BOUNDS,
    EVAL_SERIES_PER_DATASET,
    HORIZON,
    ROBUST_SCALE_FLOOR,
    SOURCE_DATASETS,
    TRAIN_ANCHORS,
    TRAIN_SERIES_PER_DATASET,
    _center_scale,
    _load_roster_values,
)
from SelfEvolvingHarnessTS.operators.s1_outlier import hampel_filter
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features


SCHEMA_VERSION = "e2-source-outlier-local-behavior-audit/2"
SCIENTIFIC_ROLE = "development_source_outlier_observer_operator_local_behavior"
PREMISE_SCHEMA_VERSION = "e2-source-cohort-policy-premise/1"
SPIKE_POSITIONS = (111, 128, 149, 166)
SPIKE_SIGN_PATTERN = (1, -1, 1, -1)
SPIKE_SCALE_MULTIPLIER = 10.0
SIGN_SALT = "e2-source-outlier-local-behavior-sign-v1"
HAMPEL_PARAMETERS = {"window": 7, "n_sigmas": 8.0, "global_z_min": 4.0}
PUBLIC_Z_MIN = 4.0
MAX_MODIFIED_FRACTION = 0.05
LOCAL_HARM_TOLERANCE = 1e-12
PREMISE_REPORT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_cohort_policy_premise_report.json"
)
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_outlier_local_behavior_audit_report.json"
)


@dataclass(frozen=True)
class RosterItem:
    record: SeriesRecord
    cohort: str


def _require_locked_configuration(configuration: object) -> None:
    if not isinstance(configuration, dict):
        raise ValueError("premise report lacks configuration")
    expected: dict[str, object] = {
        "datasets": sorted(SOURCE_DATASETS),
        "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
        "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
        "train_anchors": list(TRAIN_ANCHORS),
        "context_length": CONTEXT_LENGTH,
        "horizon": HORIZON,
        "eval_context_bounds": list(EVAL_CONTEXT_BOUNDS),
        "eval_future_bounds": list(EVAL_FUTURE_BOUNDS),
    }
    for name, value in expected.items():
        if configuration.get(name) != value:
            raise ValueError(f"premise configuration changed at {name}")


def _read_premise_roster(
    *, premise_report_path: Path, registry_path: Path
) -> tuple[list[RosterItem], dict[str, object]]:
    premise = json.loads(premise_report_path.read_text("utf-8"))
    if premise.get("schema_version") != PREMISE_SCHEMA_VERSION:
        raise ValueError("unsupported cohort-policy premise report schema")
    _require_locked_configuration(premise.get("configuration"))
    selection = premise.get("roster_selection")
    selected = selection.get("selected_by_dataset") if isinstance(selection, dict) else None
    if not isinstance(selected, dict) or set(selected) != set(SOURCE_DATASETS):
        raise ValueError("premise report lacks exactly the two locked Source rosters")

    selected_uids: dict[str, dict[str, list[str]]] = {}
    all_uids: list[str] = []
    for dataset_id in SOURCE_DATASETS:
        raw_dataset = selected.get(dataset_id)
        if not isinstance(raw_dataset, dict):
            raise ValueError(f"invalid premise roster for {dataset_id}")
        selected_uids[dataset_id] = {}
        for cohort, count in (
            ("train", TRAIN_SERIES_PER_DATASET),
            ("eval", EVAL_SERIES_PER_DATASET),
        ):
            raw = raw_dataset.get(cohort)
            if not isinstance(raw, list) or not all(
                isinstance(uid, str) and uid for uid in raw
            ):
                raise ValueError(f"invalid premise {cohort} UID list: {dataset_id}")
            if len(raw) != count or len(raw) != len(set(raw)):
                raise ValueError(f"wrong-sized or duplicate premise roster: {dataset_id}")
            selected_uids[dataset_id][cohort] = list(raw)
            all_uids.extend(raw)
    if len(all_uids) != len(set(all_uids)):
        raise ValueError("premise train/eval rosters overlap")

    records = {record.series_uid: record for record in read_registry_jsonl(registry_path)}
    roster: list[RosterItem] = []
    metadata_audit: dict[str, object] = {}
    for dataset_id in SOURCE_DATASETS:
        dataset_items: list[RosterItem] = []
        for cohort in ("train", "eval"):
            for uid in selected_uids[dataset_id][cohort]:
                record = records.get(uid)
                if record is None or record.dataset_id != dataset_id:
                    raise ValueError(f"premise UID missing or changed dataset: {uid}")
                dataset_items.append(RosterItem(record=record, cohort=cohort))
        train_entities = [
            item.record.entity_id for item in dataset_items if item.cohort == "train"
        ]
        if len(train_entities) != len(set(train_entities)):
            raise ValueError(f"duplicate premise train entity_id: {dataset_id}")
        roster.extend(dataset_items)
        metadata_audit[dataset_id] = {
            "train_uid_count": TRAIN_SERIES_PER_DATASET,
            "eval_uid_count_metadata_only": EVAL_SERIES_PER_DATASET,
            "train_entity_ids_unique": True,
            "records_match_locked_dataset_id": True,
        }
    return roster, {
        "source": str(premise_report_path),
        "premise_fields_consulted": [
            "schema_version",
            "configuration",
            "roster_selection.selected_by_dataset",
        ],
        "premise_outcomes_or_policy_evidence_consulted": False,
        "selected_by_dataset": selected_uids,
        "metadata_audit_by_dataset": metadata_audit,
        "train_eval_uid_disjoint": True,
    }


def _base_sign(dataset_id: str, entity_id: str, anchor: int) -> tuple[int, str]:
    payload = f"{SIGN_SALT}\0{dataset_id}\0{entity_id}\0{anchor}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (1 if int(digest[:2], 16) % 2 else -1), digest


def _inject_spikes(
    clean: np.ndarray, *, dataset_id: str, entity_id: str, anchor: int
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(clean, dtype=np.float64)
    if values.shape != (CONTEXT_LENGTH,) or not np.isfinite(values).all():
        raise ValueError("spike injection requires a finite length-192 context")
    _, scale, scale_method = _center_scale(values)
    base_sign, sign_sha256 = _base_sign(dataset_id, entity_id, anchor)
    signs = tuple(base_sign * direction for direction in SPIKE_SIGN_PATTERN)
    corrupt = values.copy()
    for position, sign in zip(SPIKE_POSITIONS, signs):
        corrupt[position] += sign * SPIKE_SCALE_MULTIPLIER * scale
    return corrupt, {
        "positions": list(SPIKE_POSITIONS),
        "base_sign": base_sign,
        "signs": list(signs),
        "clean_truth_at_positions": [float(values[position]) for position in SPIKE_POSITIONS],
        "clean_robust_scale": scale,
        "clean_robust_scale_method": scale_method,
        "sign_sha256": sign_sha256,
    }


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _compiled_policy(values: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the frozen public binding and risk guard without hidden information."""

    raw = np.asarray(values, dtype=np.float64)
    extraction = extract_public_features(raw)
    mapping = extraction.mapping
    public_outlier_indices = tuple(int(index) for index in extraction.outlier_indices)
    missing_fraction = mapping.get("missing_fraction")
    robust_z_peak = mapping.get("local_robust_z_peak")
    applicable = (
        _finite_number(missing_fraction)
        and float(missing_fraction) == 0.0
        and _finite_number(robust_z_peak)
        and float(robust_z_peak) >= PUBLIC_Z_MIN
    )
    public_view = {
        "missing_fraction": missing_fraction,
        "local_robust_z_peak": robust_z_peak,
    }
    if not applicable:
        return raw.copy(), {
            "activated": False,
            "candidate_executed": False,
            "risk_rollback": False,
            "reason": "public_applicability_not_satisfied",
            "public_mapping_view": public_view,
            "public_observer_outlier_indices": list(public_outlier_indices),
            "used_by_current_hampel_binding": False,
            "proposal_modified_fraction": 0.0,
            "final_modified_fraction": 0.0,
            "hidden_fields_consulted": False,
        }

    proposal = np.asarray(hampel_filter(raw, **HAMPEL_PARAMETERS), dtype=np.float64)
    if proposal.shape != raw.shape or not np.isfinite(proposal).all():
        raise RuntimeError("canonical Hampel candidate returned invalid output")
    proposal_modified = ~np.equal(proposal, raw)
    proposal_fraction = float(np.mean(proposal_modified))
    rollback = proposal_fraction > MAX_MODIFIED_FRACTION
    final = raw.copy() if rollback else proposal
    final_modified = ~np.equal(final, raw)
    return final, {
        "activated": True,
        "candidate_executed": True,
        "risk_rollback": rollback,
        "reason": (
            "proposal_exceeded_modified_fraction_risk_guard"
            if rollback
            else "public_applicability_and_risk_guard_passed"
        ),
        "public_mapping_view": public_view,
        "public_observer_outlier_indices": list(public_outlier_indices),
        "used_by_current_hampel_binding": False,
        "proposal_modified_fraction": proposal_fraction,
        "final_modified_fraction": float(np.mean(final_modified)),
        "hidden_fields_consulted": False,
    }


def _private_grader(
    *,
    clean: np.ndarray,
    corrupt: np.ndarray,
    repaired: np.ndarray,
    scale: float,
    hidden: dict[str, object],
    public_observer_outlier_indices: list[int],
) -> dict[str, object]:
    """Use hidden truth only after the compiled policy has returned its output."""

    modified_indices = tuple(int(index) for index in np.flatnonzero(~np.equal(repaired, corrupt)))
    injected = set(int(position) for position in hidden["positions"])  # type: ignore[arg-type]
    modified = set(modified_indices)
    true_positive = len(modified & injected)
    public_observer = set(int(index) for index in public_observer_outlier_indices)
    public_observer_true_positive = len(public_observer & injected)
    corrupt_error = float(np.mean(np.abs(corrupt - clean)) / scale)
    repair_error = float(np.mean(np.abs(repaired - clean)) / scale)
    reduction = corrupt_error - repair_error
    return {
        "diagnostic_role": "private_grader_only_not_policy_input",
        "hidden_positions": list(hidden["positions"]),  # type: ignore[arg-type]
        "hidden_base_sign": int(hidden["base_sign"]),
        "hidden_signs": list(hidden["signs"]),  # type: ignore[arg-type]
        "corrupt_normalized_mae_to_clean": corrupt_error,
        "repair_normalized_mae_to_clean": repair_error,
        "error_reduction": reduction,
        "recovery_fraction": reduction / corrupt_error if corrupt_error > 0.0 else None,
        "modified_indices": list(modified_indices),
        "modification_true_positive_count": true_positive,
        "modification_count": len(modified),
        "modification_precision": true_positive / len(modified) if modified else None,
        "injected_point_recall": true_positive / len(injected),
        "collateral_modification_count": len(modified - injected),
        "public_observer_true_positive_count": public_observer_true_positive,
        "public_observer_count": len(public_observer),
        "public_observer_precision": (
            public_observer_true_positive / len(public_observer)
            if public_observer
            else None
        ),
        "public_observer_injected_recall": public_observer_true_positive / len(injected),
        "public_observer_collateral_count": len(public_observer - injected),
        "public_observer_metrics_used_by_current_hampel_binding": False,
        "local_harm": repair_error > corrupt_error + LOCAL_HARM_TOLERANCE,
        "used_by_policy_or_rollback": False,
    }


def _audit_dataset(
    *,
    dataset_id: str,
    train_items: list[RosterItem],
    values_by_uid: dict[str, np.ndarray],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for item in train_items:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        for anchor in TRAIN_ANCHORS:
            start = anchor - CONTEXT_LENGTH
            if start < 0 or anchor > 928:
                raise AssertionError("audit context crosses frozen train boundary")
            clean = np.asarray(values[start:anchor], dtype=np.float64).copy()
            if clean.shape != (CONTEXT_LENGTH,) or not np.isfinite(clean).all():
                raise ValueError(f"invalid audit context: {uid}/{anchor}")
            corrupt, hidden = _inject_spikes(
                clean,
                dataset_id=dataset_id,
                entity_id=item.record.entity_id,
                anchor=anchor,
            )
            repaired, corrupt_policy = _compiled_policy(corrupt)
            # The identical compiled policy runs on clean input without hidden truth.
            clean_output, clean_policy = _compiled_policy(clean)
            grader = _private_grader(
                clean=clean,
                corrupt=corrupt,
                repaired=repaired,
                scale=float(hidden["clean_robust_scale"]),
                hidden=hidden,
                public_observer_outlier_indices=list(
                    corrupt_policy["public_observer_outlier_indices"]  # type: ignore[arg-type]
                ),
            )
            clean_modified = ~np.equal(clean_output, clean)
            clean_public_outlier_indices = list(
                clean_policy["public_observer_outlier_indices"]  # type: ignore[arg-type]
            )
            rows.append(
                {
                    "series_uid": uid,
                    "entity_id": item.record.entity_id,
                    "anchor": anchor,
                    "corrupt_policy": corrupt_policy,
                    "private_grader": grader,
                    "clean_risk": {
                        "activated": bool(clean_policy["activated"]),
                        "modified": bool(np.any(clean_modified)),
                        "modified_fraction": float(np.mean(clean_modified)),
                        "modified_point_count": int(np.count_nonzero(clean_modified)),
                        "public_observer_outlier_index_count": len(
                            clean_public_outlier_indices
                        ),
                        "public_observer_outlier_index_fraction": len(
                            clean_public_outlier_indices
                        )
                        / CONTEXT_LENGTH,
                        "risk_rollback": bool(clean_policy["risk_rollback"]),
                        "policy": clean_policy,
                    },
                    "sign_sha256": hidden["sign_sha256"],
                }
            )
    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    if len(rows) != expected:
        raise AssertionError("unexpected local-behavior example count")

    activation_count = sum(bool(row["corrupt_policy"]["activated"]) for row in rows)  # type: ignore[index]
    reductions = [float(row["private_grader"]["error_reduction"]) for row in rows]  # type: ignore[index]
    recoveries = [float(row["private_grader"]["recovery_fraction"]) for row in rows]  # type: ignore[index]
    total_modified = sum(int(row["private_grader"]["modification_count"]) for row in rows)  # type: ignore[index]
    total_true_positive = sum(
        int(row["private_grader"]["modification_true_positive_count"]) for row in rows  # type: ignore[index]
    )
    injected_total = expected * len(SPIKE_POSITIONS)
    micro_precision = total_true_positive / total_modified if total_modified else None
    micro_recall = total_true_positive / injected_total
    public_observer_count = sum(
        int(row["private_grader"]["public_observer_count"]) for row in rows  # type: ignore[index]
    )
    public_observer_true_positive = sum(
        int(row["private_grader"]["public_observer_true_positive_count"])  # type: ignore[index]
        for row in rows
    )
    public_observer_micro_precision = (
        public_observer_true_positive / public_observer_count
        if public_observer_count
        else None
    )
    public_observer_micro_recall = public_observer_true_positive / injected_total
    harm_rate = sum(bool(row["private_grader"]["local_harm"]) for row in rows) / expected  # type: ignore[index]
    clean_modified_count = sum(bool(row["clean_risk"]["modified"]) for row in rows)  # type: ignore[index]
    clean_public_outlier_example_count = sum(
        int(row["clean_risk"]["public_observer_outlier_index_count"]) > 0  # type: ignore[index]
        for row in rows
    )
    clean_public_outlier_point_count = sum(
        int(row["clean_risk"]["public_observer_outlier_index_count"])  # type: ignore[index]
        for row in rows
    )
    final_fractions = [
        float(row["corrupt_policy"]["final_modified_fraction"]) for row in rows  # type: ignore[index]
    ] + [float(row["clean_risk"]["modified_fraction"]) for row in rows]  # type: ignore[index]
    metrics = {
        "corrupt_activation_rate": activation_count / expected,
        "mean_error_reduction": statistics.fmean(reductions),
        "median_recovery_fraction": statistics.median(recoveries),
        "micro_modification_precision": micro_precision,
        "micro_injected_point_recall": micro_recall,
        "public_observer_micro_precision": public_observer_micro_precision,
        "public_observer_micro_injected_recall": public_observer_micro_recall,
        "local_harm_rate": harm_rate,
        "clean_modified_example_rate": clean_modified_count / expected,
        "clean_public_outlier_example_rate": clean_public_outlier_example_count
        / expected,
        "clean_public_outlier_point_rate": clean_public_outlier_point_count
        / (expected * CONTEXT_LENGTH),
        "maximum_final_modified_fraction": max(final_fractions),
    }
    checks = {
        "corrupt_activation_rate_at_least_0_90": metrics["corrupt_activation_rate"]
        >= 0.90,
        "mean_error_reduction_strictly_positive": metrics["mean_error_reduction"] > 0.0,
        "median_recovery_fraction_at_least_0_50": metrics["median_recovery_fraction"]
        >= 0.50,
        "micro_modification_precision_at_least_0_75": (
            micro_precision is not None and micro_precision >= 0.75
        ),
        "micro_injected_point_recall_at_least_0_75": micro_recall >= 0.75,
        "local_harm_rate_at_most_0_10": harm_rate <= 0.10,
        "clean_modified_example_rate_at_most_0_10": clean_modified_count / expected
        <= 0.10,
        "all_final_modified_fractions_at_most_0_05": max(final_fractions)
        <= MAX_MODIFIED_FRACTION,
    }
    corrupt_rollbacks = sum(bool(row["corrupt_policy"]["risk_rollback"]) for row in rows)  # type: ignore[index]
    clean_rollbacks = sum(bool(row["clean_risk"]["risk_rollback"]) for row in rows)  # type: ignore[index]
    return {
        "evidence_type": "LocalBehaviorEvidence",
        "scientific_unit": "dataset_level_exposed_development_train_contexts",
        "dataset_id": dataset_id,
        "context_count": expected,
        "metrics": metrics,
        "risk_rollbacks": {
            "corrupt_count": corrupt_rollbacks,
            "clean_count": clean_rollbacks,
        },
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
        "per_context_diagnostics": rows,
    }


def run_e2_source_outlier_local_behavior_audit(
    *, premise_report_path: Path, registry_path: Path, clean_root: Path
) -> dict[str, object]:
    roster, roster_report = _read_premise_roster(
        premise_report_path=premise_report_path,
        registry_path=registry_path,
    )
    train_roster = [item for item in roster if item.cohort == "train"]
    if len(train_roster) != len(SOURCE_DATASETS) * TRAIN_SERIES_PER_DATASET:
        raise AssertionError("expected exactly 64 premise training series")
    # Eval-series arrays are deliberately not passed to the value loader.
    values_by_uid = _load_roster_values(train_roster, clean_root)  # type: ignore[arg-type]
    evidence_rows = [
        _audit_dataset(
            dataset_id=dataset_id,
            train_items=[
                item for item in train_roster if item.record.dataset_id == dataset_id
            ],
            values_by_uid=values_by_uid,
        )
        for dataset_id in SOURCE_DATASETS
    ]
    passed = all(bool(row["gate_pass"]) for row in evidence_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": SCIENTIFIC_ROLE,
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "contexts_per_series": len(TRAIN_ANCHORS),
            "train_anchors": list(TRAIN_ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "spike_positions_relative_to_context": list(SPIKE_POSITIONS),
            "spike_scale_multiplier": SPIKE_SCALE_MULTIPLIER,
            "spike_scale_definition": (
                "10.0 times the clean-context robust scale from premise median/MAD "
                "with its fixed std and floor fallbacks"
            ),
            "spike_sign_pattern_times_hashed_base_sign": list(SPIKE_SIGN_PATTERN),
            "sign_salt": SIGN_SALT,
            "sign_hash_input_fields": ["dataset_id", "entity_id", "anchor"],
            "topology_and_dose_family": "existing_M0_severe_frozen_before_execution",
            "compiled_policy": {
                "source_skill_entry": "sparse_public_outlier_repair_v2",
                "public_applicability": {
                    "missing_fraction": "==0.0",
                    "local_robust_z_peak": ">=4.0",
                },
                "operator": "hampel_filter",
                "parameters": HAMPEL_PARAMETERS,
                "max_modified_fraction_risk_rollback": MAX_MODIFIED_FRACTION,
                "identity_always_retained": True,
            },
        },
        "roster": roster_report,
        "information_wall": {
            "global_registry_metadata_loaded": True,
            "only_premise_train_roster_source_values_loaded": True,
            "premise_eval_series_values_or_future_loaded": False,
            "premise_policy_evidence_or_outcomes_consulted": False,
            "fresh_replay_report_read": False,
            "support_a_validation_values_or_context_or_future_read": False,
            "support_b_values_or_context_or_future_read": False,
            "uci_target_or_query_values_or_context_or_future_read": False,
            "hidden_positions_signs_or_clean_truth_available_to_policy": False,
            "private_grader_diagnostics_used_by_policy_or_execution": False,
            "private_grader_metrics_used_only_by_predefined_audit_gate": True,
        },
        "gate": {
            "thresholds_frozen_before_execution": True,
            "exact_definition": (
                "pass only when both datasets independently satisfy activation>=0.90, "
                "mean error reduction>0, median recovery>=0.50, micro precision>=0.75, "
                "micro injected-point recall>=0.75, local harm<=0.10, clean modified-"
                "example rate<=0.10, and every final executed modified fraction<=0.05"
            ),
            "dataset_pass": {
                str(row["dataset_id"]): bool(row["gate_pass"]) for row in evidence_rows
            },
            "pass": passed,
        },
        "local_behavior_evidence": evidence_rows,
        "consumer_fit_count": 0,
        "chronos_judge_call_count": 0,
        "verdict": (
            "OUTLIER_LOCAL_BEHAVIOR_PROMISING"
            if passed
            else "OUTLIER_LOCAL_BEHAVIOR_WEAK"
        ),
        "promotion": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "query": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most exposed-development Source operator/observer feasibility under "
            "one frozen synthetic M0-severe impulsive-spike family, with public-observer "
            "fault-localization instrumentation used only by the private grader; not "
            "downstream Consumer utility, individual causal evidence, Capability, "
            "promotion, formal transfer, Target, or Query evidence."
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
    report = run_e2_source_outlier_local_behavior_audit(
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
