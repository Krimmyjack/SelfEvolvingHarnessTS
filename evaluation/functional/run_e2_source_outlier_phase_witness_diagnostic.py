"""Diagnose a human-designed phase witness for sparse Source outlier repair.

This zero-fit development diagnostic uses public outlier candidates and fixed hourly
phase donors to prototype a typed point-replacement program.  The prototype is not
registered in the Operator DSL.  Hidden injected positions and clean truth are used
only by the private grader, never by the witness or program.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    CONTEXT_LENGTH,
    SOURCE_DATASETS,
    TRAIN_ANCHORS,
    TRAIN_SERIES_PER_DATASET,
    _center_scale,
    _load_roster_values,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_outlier_local_behavior_audit import (
    MAX_MODIFIED_FRACTION,
    PREMISE_REPORT_RELATIVE_PATH,
    SPIKE_POSITIONS,
    RosterItem,
    _inject_spikes,
    _private_grader,
    _read_premise_roster,
)
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features


SCHEMA_VERSION = "e2-source-outlier-phase-witness-diagnostic/1"
SCIENTIFIC_ROLE = "development_source_outlier_phase_witness_local_behavior"
PERIOD = 24
PHASE_DONOR_LAGS = (24, 48, 72)
MIN_DONORS = 3
DONOR_MAD_TO_CONTEXT_SCALE_MAX = 1.0
PHASE_DEVIATION_MIN = 4.0
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_outlier_phase_witness_diagnostic_report.json"
)


def _phase_witness_plan(
    values: np.ndarray, public_outlier_indices: tuple[int, ...]
) -> tuple[dict[int, float], list[dict[str, object]], dict[str, int]]:
    """Build replacements using only the context and public candidate indices."""

    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (CONTEXT_LENGTH,) or not np.isfinite(raw).all():
        raise ValueError("phase witness requires one finite length-192 context")
    public_candidates = tuple(sorted(set(int(index) for index in public_outlier_indices)))
    if any(index < 0 or index >= CONTEXT_LENGTH for index in public_candidates):
        raise ValueError("public outlier candidate is outside the context")
    _, context_scale, context_scale_method = _center_scale(raw)
    excluded = set(public_candidates)
    replacements: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    reason_counts = {
        "eligible": 0,
        "donor_insufficiency": 0,
        "donor_disagreement": 0,
        "phase_deviation_rejection": 0,
    }
    for index in public_candidates:
        donor_indices = sorted(
            candidate
            for lag in PHASE_DONOR_LAGS
            for candidate in (index - lag, index + lag)
            if 0 <= candidate < CONTEXT_LENGTH and candidate not in excluded
        )
        row: dict[str, object] = {
            "candidate_index": index,
            "donor_indices": donor_indices,
            "donor_count": len(donor_indices),
            "context_scale": context_scale,
            "context_scale_method": context_scale_method,
        }
        if len(donor_indices) < MIN_DONORS:
            reason = "donor_insufficiency"
            row.update(
                {
                    "donor_median": None,
                    "donor_mad_to_context_scale": None,
                    "phase_deviation_to_context_scale": None,
                    "eligible": False,
                    "reason": reason,
                }
            )
        else:
            donor_values = raw[donor_indices]
            donor_median = float(np.median(donor_values))
            donor_mad = float(np.median(np.abs(donor_values - donor_median)))
            donor_mad_ratio = donor_mad / context_scale
            deviation = abs(float(raw[index]) - donor_median) / context_scale
            row.update(
                {
                    "donor_median": donor_median,
                    "donor_mad_to_context_scale": donor_mad_ratio,
                    "phase_deviation_to_context_scale": deviation,
                }
            )
            if donor_mad_ratio > DONOR_MAD_TO_CONTEXT_SCALE_MAX:
                reason = "donor_disagreement"
                row.update({"eligible": False, "reason": reason})
            elif deviation < PHASE_DEVIATION_MIN:
                reason = "phase_deviation_rejection"
                row.update({"eligible": False, "reason": reason})
            else:
                reason = "eligible"
                row.update({"eligible": True, "reason": reason})
                replacements[index] = donor_median
        reason_counts[reason] += 1
        rows.append(row)
    return replacements, rows, reason_counts


def _compiled_phase_prototype(values: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    """Observe, witness, and execute without hidden truth or injected geometry."""

    raw = np.asarray(values, dtype=np.float64)
    extraction = extract_public_features(raw)
    public_indices = tuple(int(index) for index in extraction.outlier_indices)
    replacements, candidate_rows, reason_counts = _phase_witness_plan(
        raw, public_indices
    )
    proposal = raw.copy()
    for index, replacement in replacements.items():
        proposal[index] = replacement
    proposal_modified = ~np.equal(proposal, raw)
    proposal_fraction = float(np.mean(proposal_modified))
    rollback = proposal_fraction > MAX_MODIFIED_FRACTION
    final = raw.copy() if rollback else proposal
    final_modified = ~np.equal(final, raw)
    return final, {
        "prototype_kind": "unregistered_typed_program",
        "activated": bool(replacements),
        "risk_rollback": rollback,
        "public_outlier_indices": list(public_indices),
        "public_candidate_count": len(public_indices),
        "eligible_candidate_count": len(replacements),
        # Per-candidate rows are intentionally not serialized: the aggregate reason
        # counts below are sufficient for this one-shot fault localization and keep
        # the development artifact compact.
        "reason_counts": reason_counts,
        "proposal_modified_fraction": proposal_fraction,
        "final_modified_fraction": float(np.mean(final_modified)),
        "hidden_positions_signs_or_clean_truth_consulted": False,
    }


def _sum_reason_counts(rows: list[dict[str, object]], role: str) -> dict[str, int]:
    result = {
        "eligible": 0,
        "donor_insufficiency": 0,
        "donor_disagreement": 0,
        "phase_deviation_rejection": 0,
    }
    for row in rows:
        diagnostics = row[role]
        if not isinstance(diagnostics, dict):
            raise TypeError("context row lacks phase witness diagnostics")
        counts = diagnostics["reason_counts"]
        if not isinstance(counts, dict):
            raise TypeError("phase witness diagnostics lack reason counts")
        for reason in result:
            result[reason] += int(counts[reason])
    return result


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
                raise AssertionError("phase-witness context crosses frozen train boundary")
            clean = np.asarray(values[start:anchor], dtype=np.float64).copy()
            if clean.shape != (CONTEXT_LENGTH,) or not np.isfinite(clean).all():
                raise ValueError(f"invalid phase-witness context: {uid}/{anchor}")
            corrupt, hidden = _inject_spikes(
                clean,
                dataset_id=dataset_id,
                entity_id=item.record.entity_id,
                anchor=anchor,
            )
            repaired, corrupt_witness = _compiled_phase_prototype(corrupt)
            clean_output, clean_witness = _compiled_phase_prototype(clean)
            grader = _private_grader(
                clean=clean,
                corrupt=corrupt,
                repaired=repaired,
                scale=float(hidden["clean_robust_scale"]),
                hidden=hidden,
                public_observer_outlier_indices=list(
                    corrupt_witness["public_outlier_indices"]  # type: ignore[arg-type]
                ),
            )
            clean_modified = ~np.equal(clean_output, clean)
            rows.append(
                {
                    "series_uid": uid,
                    "entity_id": item.record.entity_id,
                    "anchor": anchor,
                    "corrupt_phase_witness": corrupt_witness,
                    "private_grader": grader,
                    "clean_risk": {
                        "activated": bool(clean_witness["activated"]),
                        "modified": bool(np.any(clean_modified)),
                        "modified_fraction": float(np.mean(clean_modified)),
                        "modified_point_count": int(np.count_nonzero(clean_modified)),
                        "risk_rollback": bool(clean_witness["risk_rollback"]),
                        "phase_witness": clean_witness,
                    },
                    "sign_sha256": hidden["sign_sha256"],
                }
            )
    expected = TRAIN_SERIES_PER_DATASET * len(TRAIN_ANCHORS)
    if len(rows) != expected:
        raise AssertionError("unexpected phase-witness context count")

    activation_count = sum(
        bool(row["corrupt_phase_witness"]["activated"]) for row in rows  # type: ignore[index]
    )
    reductions = [
        float(row["private_grader"]["error_reduction"]) for row in rows  # type: ignore[index]
    ]
    recoveries = [
        float(row["private_grader"]["recovery_fraction"]) for row in rows  # type: ignore[index]
    ]
    total_modified = sum(
        int(row["private_grader"]["modification_count"]) for row in rows  # type: ignore[index]
    )
    total_true_positive = sum(
        int(row["private_grader"]["modification_true_positive_count"])  # type: ignore[index]
        for row in rows
    )
    injected_total = expected * len(SPIKE_POSITIONS)
    micro_precision = total_true_positive / total_modified if total_modified else None
    micro_recall = total_true_positive / injected_total
    harm_rate = sum(
        bool(row["private_grader"]["local_harm"]) for row in rows  # type: ignore[index]
    ) / expected
    clean_modified_count = sum(
        bool(row["clean_risk"]["modified"]) for row in rows  # type: ignore[index]
    )
    final_fractions = [
        float(row["corrupt_phase_witness"]["final_modified_fraction"])  # type: ignore[index]
        for row in rows
    ] + [
        float(row["clean_risk"]["modified_fraction"]) for row in rows  # type: ignore[index]
    ]
    metrics = {
        "corrupt_activation_rate": activation_count / expected,
        "mean_error_reduction": statistics.fmean(reductions),
        "median_recovery_fraction": statistics.median(recoveries),
        "micro_modification_precision": micro_precision,
        "micro_injected_point_recall": micro_recall,
        "local_harm_rate": harm_rate,
        "clean_modified_example_rate": clean_modified_count / expected,
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
    corrupt_candidate_count = sum(
        int(row["corrupt_phase_witness"]["public_candidate_count"])  # type: ignore[index]
        for row in rows
    )
    corrupt_eligible_count = sum(
        int(row["corrupt_phase_witness"]["eligible_candidate_count"])  # type: ignore[index]
        for row in rows
    )
    corrupt_candidate_contexts = sum(
        int(row["corrupt_phase_witness"]["public_candidate_count"]) > 0  # type: ignore[index]
        for row in rows
    )
    public_candidate_true_positive = sum(
        int(row["private_grader"]["public_observer_true_positive_count"])  # type: ignore[index]
        for row in rows
    )
    return {
        "evidence_type": "LocalBehaviorEvidence",
        "scientific_unit": "dataset_level_exposed_development_train_contexts",
        "dataset_id": dataset_id,
        "context_count": expected,
        "metrics": metrics,
        "witness_supply_diagnostics": {
            "public_candidate_count": corrupt_candidate_count,
            "candidate_context_coverage_rate": corrupt_candidate_contexts / expected,
            "candidate_point_fraction": corrupt_candidate_count
            / (expected * CONTEXT_LENGTH),
            "candidate_micro_precision_against_injected_positions": (
                public_candidate_true_positive / corrupt_candidate_count
                if corrupt_candidate_count
                else None
            ),
            "candidate_micro_injected_point_coverage": (
                public_candidate_true_positive / injected_total
            ),
            "eligible_candidate_count": corrupt_eligible_count,
            "eligibility_rate_among_public_candidates": (
                corrupt_eligible_count / corrupt_candidate_count
                if corrupt_candidate_count
                else None
            ),
            "corrupt_reason_counts": _sum_reason_counts(
                rows, "corrupt_phase_witness"
            ),
            "clean_reason_counts": _sum_reason_counts(
                [
                    {
                        "clean_phase_witness": row["clean_risk"]["phase_witness"]  # type: ignore[index]
                    }
                    for row in rows
                ],
                "clean_phase_witness",
            ),
        },
        "risk_rollbacks": {
            "corrupt_count": sum(
                bool(row["corrupt_phase_witness"]["risk_rollback"])  # type: ignore[index]
                for row in rows
            ),
            "clean_count": sum(
                bool(row["clean_risk"]["risk_rollback"]) for row in rows  # type: ignore[index]
            ),
        },
        "gate_checks": checks,
        "gate_pass": all(checks.values()),
        "per_context_diagnostics": rows,
    }


def run_e2_source_outlier_phase_witness_diagnostic(
    *, premise_report_path: Path, registry_path: Path, clean_root: Path
) -> dict[str, object]:
    roster, roster_report = _read_premise_roster(
        premise_report_path=premise_report_path,
        registry_path=registry_path,
    )
    train_roster = [item for item in roster if item.cohort == "train"]
    if len(train_roster) != len(SOURCE_DATASETS) * TRAIN_SERIES_PER_DATASET:
        raise AssertionError("expected exactly 64 premise training series")
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
            "corruption_reused_from": (
                "run_e2_source_outlier_local_behavior_audit._inject_spikes"
            ),
            "hourly_data_semantics_period": PERIOD,
            "phase_donor_lags": list(PHASE_DONOR_LAGS),
            "minimum_donor_count": MIN_DONORS,
            "donor_mad_to_context_scale_max": DONOR_MAD_TO_CONTEXT_SCALE_MAX,
            "phase_deviation_to_context_scale_min": PHASE_DEVIATION_MIN,
            "threshold_sweep_performed": False,
            "max_modified_fraction_risk_rollback": MAX_MODIFIED_FRACTION,
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
            "hidden_positions_signs_or_clean_truth_available_to_witness_or_program": False,
            "private_grader_metrics_used_only_by_predefined_local_behavior_gate": True,
        },
        "prototype_contract": {
            "status": "typed_program_prototype_not_registered_in_operator_dsl",
            "program": (
                "replace only eligible public outlier candidates with the median of "
                "legal unflagged same-phase donors; preserve every other point"
            ),
            "registration_decision": (
                "worth registering only if the frozen local-behavior gate passes"
            ),
            "human_designed_one_time_mechanistic_patch": True,
            "hampel_changed_or_tuned": False,
        },
        "gate": {
            "thresholds_frozen_before_execution": True,
            "same_eight_local_behavior_checks_as_outlier_audit": True,
            "dataset_pass": {
                str(row["dataset_id"]): bool(row["gate_pass"]) for row in evidence_rows
            },
            "pass": passed,
        },
        "local_behavior_evidence": evidence_rows,
        "consumer_fit_count": 0,
        "chronos_judge_call_count": 0,
        "verdict": (
            "PHASE_WITNESS_LOCAL_BEHAVIOR_PROMISING"
            if passed
            else "PHASE_WITNESS_LOCAL_BEHAVIOR_WEAK"
        ),
        "promotion": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "query": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most exposed-development feasibility evidence for one human-designed "
            "phase-witness typed-program prototype not registered in the Operator DSL; "
            "not downstream Consumer utility, Capability, Harness evolution, promotion, "
            "formal transfer, Target, or Query evidence."
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
    report = run_e2_source_outlier_phase_witness_diagnostic(
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
