"""Zero-fit donor-premise diagnostic on 24 frozen Source train series.

Public descriptors read only the corrupt artifact; clean truth is grader-only scoring.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import select_roster as select_scope_roster
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_coherent_missingness_positive_control import PHASE_CYCLES, PHASE_MIN_DONORS, RECENT_V2, DatasetSpec, _center_scale, _fixed_gap_mask
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_level_adjusted_scope_extension import SCOPE_DATASET_SPECS, TRAIN_SERIES_PER_DATASET, _scope_train_roster
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_recent_level_adjusted_supply_diagnostic import BETTER_THAN_BASELINE_MIN_COUNT, MEDIAN_RECOVERY_MIN, PROTOTYPE_ID, local_level_adjusted_period_median_complete_v0
from SelfEvolvingHarnessTS.operators.registry import get_operator


SCHEMA_VERSION = "e2-source-level-adjusted-donor-premise/1"
OUTPUT_RELATIVE_PATH = "artifacts/functional/e2/source_level_adjusted_donor_premise_report.json"
DESCRIPTORS = (
    "cycle_level_offset_mad_norm",
    "aligned_flank_residual_norm",
    "adjusted_donor_prediction_disagreement_norm",
)


def _public_gap_candidates(corrupt: np.ndarray, gap: tuple[int, int], *, period: int, scale: float) -> tuple[dict[str, object], list[tuple[int, np.ndarray]]]:
    """Build three prior-only adjusted donor predictions without clean truth."""
    start, stop = gap
    flank = np.arange(start - period, start, dtype=np.int64)
    cycles: list[dict[str, object]] = []
    candidates: list[tuple[int, np.ndarray]] = []
    offsets: list[float] = []
    residuals: list[float] = []
    for cycle in range(1, PHASE_CYCLES + 1):
        donor_flank = flank - cycle * period
        donor_gap = np.arange(start, stop, dtype=np.int64) - cycle * period
        pair_mask = np.isfinite(corrupt[flank]) & np.isfinite(corrupt[donor_flank])
        used_flank = flank[pair_mask]
        used_donor_flank = donor_flank[pair_mask]
        complete = bool(
            int(donor_flank[0]) >= 0
            and int(donor_gap[0]) >= 0
            and int(pair_mask.sum()) > 0
            and np.isfinite(corrupt[donor_gap]).all()
        )
        row: dict[str, object] = {
            "cycle": cycle,
            "complete": complete,
            "flank_reference_indices": used_flank.tolist(),
            "flank_donor_indices": used_donor_flank.tolist(),
            "gap_donor_indices": donor_gap.tolist(),
            "aligned_flank_observed_count": int(pair_mask.sum()),
            "gap_donor_count": int(np.isfinite(corrupt[donor_gap]).sum()),
        }
        if complete:
            offset = float(np.median(corrupt[used_flank] - corrupt[used_donor_flank]))
            prediction = np.asarray(corrupt[donor_gap] + offset, dtype=np.float64)
            residual = float(np.median(np.abs(corrupt[used_flank] - corrupt[used_donor_flank] - offset)) / scale)
            row.update({
                "level_offset": offset,
                "level_offset_norm": offset / scale,
                "aligned_flank_residual_norm": residual,
                "adjusted_gap_prediction": prediction.tolist(),
            })
            offsets.append(offset)
            residuals.append(residual)
            candidates.append((cycle, prediction))
        cycles.append(row)
    matrix = np.asarray([prediction for _, prediction in candidates], dtype=np.float64)
    offset_array = np.asarray(offsets, dtype=np.float64)
    offset_mad = 1.4826 * float(np.median(np.abs(offset_array - np.median(offset_array)))) / scale
    disagreement = float(
        np.median(1.4826 * np.median(np.abs(matrix - np.median(matrix, axis=0)), axis=0)) / scale
    )
    all_sources_pre_gap = all(
        max([*row["flank_reference_indices"], *row["flank_donor_indices"], *row["gap_donor_indices"]]) < start  # type: ignore[misc]
        for row in cycles
    )
    return {
        "gap_bounds": [start, stop],
        "complete_single_cycle_donor_count": len(candidates),
        "cycle_level_offset_mad_norm": offset_mad,
        "aligned_flank_residual_norm": statistics.median(residuals),
        "adjusted_donor_prediction_disagreement_norm": disagreement,
        "cycles": cycles,
        "all_sources_pre_gap": all_sources_pre_gap,
    }, candidates


def _series_result(*, spec: DatasetSpec, item: object, values: np.ndarray) -> dict[str, object]:
    record = item.record  # type: ignore[attr-defined]
    uid = record.series_uid
    if record.dataset_id != spec.dataset_id or record.frequency != spec.frequency:
        raise ValueError(f"registry frequency/DatasetSpec mismatch: {uid}")
    clean = np.asarray(values[: spec.train_stop], dtype=np.float64).copy()
    if clean.shape != (spec.train_stop,) or not np.isfinite(clean).all():
        raise ValueError(f"invalid private clean training artifact: {uid}")
    mask = _fixed_gap_mask(spec.train_stop, geometry=RECENT_V2)
    corrupt = clean.copy()
    corrupt[mask] = np.nan
    immutable = corrupt.copy()
    _, public_scale, public_scale_method = _center_scale(corrupt)
    _, scoring_scale, scoring_scale_method = _center_scale(clean)
    current = np.asarray(get_operator("period_median_complete")(
        corrupt, period=spec.period, cycles=PHASE_CYCLES, min_donors=PHASE_MIN_DONORS
    ), dtype=np.float64)
    proposed = local_level_adjusted_period_median_complete_v0(
        corrupt, period=spec.period, gap_bounds=RECENT_V2.gap_bounds
    )
    public_gaps: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    oracle = corrupt.copy()
    for gap in RECENT_V2.gap_bounds:
        public, candidates = _public_gap_candidates(corrupt, gap, period=spec.period, scale=public_scale)
        if len(candidates) != PHASE_CYCLES:
            raise AssertionError("frozen donor geometry lacks three complete candidates")
        start, stop = gap
        scored = [(float(np.mean(np.abs(p - clean[start:stop])) / scoring_scale), cycle, p) for cycle, p in candidates]
        best_error, best_cycle, best_prediction = min(scored, key=lambda row: row[0])
        oracle[start:stop] = best_prediction
        oracle_rows.append({
            "gap_bounds": [start, stop],
            "selected_cycle": best_cycle,
            "selected_clean_gap_nmae": best_error,
            "candidate_cycle_clean_gap_nmae": {str(cycle): error for error, cycle, _ in scored},
            "selection_uses_private_clean_truth": True,
        })
        public_gaps.append(public)
    if not np.array_equal(corrupt, immutable, equal_nan=True):
        raise AssertionError("Program/descriptor mutated the corrupt artifact")
    baseline = corrupt.copy()
    baseline[mask] = float(np.median(corrupt[np.isfinite(corrupt)]))
    nmae = lambda artifact: float(np.mean(np.abs(artifact[mask] - clean[mask])) / scoring_scale)
    current_nmae, proposed_nmae = nmae(current), nmae(proposed.artifact)
    oracle_nmae, baseline_nmae = nmae(oracle), nmae(baseline)
    mechanical = (
        np.array_equal(np.isnan(corrupt), mask)
        and all(bool(row["all_sources_pre_gap"]) for row in public_gaps)
        and all(int(row["complete_single_cycle_donor_count"]) == PHASE_CYCLES for row in public_gaps)
        and all(np.isfinite(artifact[mask]).all() for artifact in (current, proposed.artifact, oracle))
        and all(np.array_equal(artifact[~mask], corrupt[~mask]) for artifact in (current, proposed.artifact, oracle))
        and proposed.fill_count == RECENT_V2.gap_point_count
        and proposed.abstain_count == 0
        and proposed.observed_collateral_change_count == 0
    )
    return {
        "series_uid": uid,
        "dataset_id": spec.dataset_id,
        "public_observation": {
            "observed_robust_scale": public_scale,
            "observed_robust_scale_method": public_scale_method,
            "descriptor_medians": {
                name: statistics.median(float(row[name]) for row in public_gaps) for name in DESCRIPTORS
            },
            "gap_diagnostics": public_gaps,
            "outcome_or_clean_truth_used": False,
        },
        "private_scoring": {
            "scoring_scale": scoring_scale,
            "scoring_scale_method": scoring_scale_method,
            "current_gap_nmae": current_nmae,
            "proposed_gap_nmae": proposed_nmae,
            "oracle_best_coherent_donor_gap_nmae": oracle_nmae,
            "finite_median_baseline_gap_nmae": baseline_nmae,
            "oracle_recovery_fraction": (baseline_nmae - oracle_nmae) / baseline_nmae,
            "oracle_better_than_finite_median": oracle_nmae < baseline_nmae,
            "oracle_selection_by_gap": oracle_rows,
        },
        "mechanical_pass": bool(mechanical),
    }


def _dataset_result(spec: DatasetSpec, items: list[object], values: dict[str, np.ndarray]) -> dict[str, object]:
    rows = [
        _series_result(spec=spec, item=item, values=values[item.record.series_uid])  # type: ignore[attr-defined]
        for item in sorted(items, key=lambda row: row.record.series_uid)  # type: ignore[attr-defined]
    ]
    if len(rows) != TRAIN_SERIES_PER_DATASET:
        raise ValueError(f"expected 12 frozen train series: {spec.dataset_id}")
    private = [row["private_scoring"] for row in rows]
    metric_names = (
        "current_gap_nmae", "proposed_gap_nmae",
        "oracle_best_coherent_donor_gap_nmae", "finite_median_baseline_gap_nmae",
    )
    metrics = {
        name: statistics.median(float(row[name]) for row in private)  # type: ignore[index]
        for name in metric_names
    }
    descriptor_medians = {
        name: statistics.median(float(row["public_observation"]["descriptor_medians"][name]) for row in rows)  # type: ignore[index]
        for name in DESCRIPTORS
    }
    mechanical = all(bool(row["mechanical_pass"]) for row in rows)
    recovery = statistics.median(float(row["oracle_recovery_fraction"]) for row in private)  # type: ignore[index]
    better = sum(bool(row["oracle_better_than_finite_median"]) for row in private)  # type: ignore[index]
    no_worse = metrics["oracle_best_coherent_donor_gap_nmae"] <= metrics["current_gap_nmae"]
    passed = mechanical and recovery >= MEDIAN_RECOVERY_MIN and better >= BETTER_THAN_BASELINE_MIN_COUNT and no_worse
    return {
        "dataset_id": spec.dataset_id,
        "series_count": len(rows),
        "public_descriptor_medians": descriptor_medians,
        "private_metric_medians": metrics,
        "all_series_mechanical_pass": mechanical,
        "median_oracle_recovery_fraction": recovery,
        "median_oracle_recovery_min": MEDIAN_RECOVERY_MIN,
        "oracle_better_than_finite_median_count": better,
        "oracle_better_min_count": BETTER_THAN_BASELINE_MIN_COUNT,
        "oracle_median_no_worse_than_current": no_worse,
        "pass": passed,
        "series_results": rows,
    }


def run_e2_source_level_adjusted_donor_premise(*, registry_path: Path, split_path: Path, support_a_subsplit_path: Path, clean_root: Path) -> dict[str, object]:
    full_roster, selection = select_scope_roster(
        registry_path=registry_path, split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
    )
    roster = _scope_train_roster(full_roster)
    values = _load_values([item.record for item in roster], clean_root)
    results = {
        spec.dataset_id: _dataset_result(
            spec, [item for item in roster if item.record.dataset_id == spec.dataset_id], values
        ) for spec in SCOPE_DATASET_SPECS
    }
    passed = all(bool(result["pass"]) for result in results.values())
    decision = (
        "donor premise passed; only a later selector/binding observation may proceed"
        if passed else "stop this donor premise and ratio scope before P1"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "development_program_conditioned_donor_premise",
        "prototype_id": PROTOTYPE_ID,
        "prototype_not_registered": True,
        "oracle_best_coherent_donor_not_deployable": True,
        "configuration": {
            "datasets": [spec.dataset_id for spec in SCOPE_DATASET_SPECS],
            "geometry_id": RECENT_V2.geometry_id,
            "gap_bounds_absolute_half_open": [list(gap) for gap in RECENT_V2.gap_bounds],
            "gap_length_over_period": 0.25,
            "cycles": PHASE_CYCLES,
            "min_donors": PHASE_MIN_DONORS,
            "current_program": "registered:period_median_complete",
            "proposed_program": PROTOTYPE_ID,
            "oracle_candidate_binding": (
                "whole-gap past donor plus all available aligned pre-gap flank pairs"
            ),
            "oracle_candidate_set_matches_frozen_prototype_full_flank_rule": False,
            "thresholds_unchanged": {
                "median_recovery_min": MEDIAN_RECOVERY_MIN,
                "better_than_baseline_min_count": BETTER_THAN_BASELINE_MIN_COUNT,
                "oracle_median_nmae_must_not_exceed_current": True,
            },
            "period_binding_by_dataset": {
                spec.dataset_id: {
                    "registry_frequency": spec.frequency,
                    "period_steps": spec.period,
                    "period_unit": "time_steps_at_registry_frequency",
                    "effective_period": "24 hours",
                    "period_source": "frozen_DatasetSpec_checked_against_registry",
                } for spec in SCOPE_DATASET_SPECS
            },
            "descriptor_thresholds_selected": False,
            "applicability_witness_contract_implemented": False,
        },
        "roster": {
            "selection": selection,
            "selection_rule": "first 12 UIDs of each frozen 32-series train cohort",
            "loaded_source_discovery_train_series_count": len(roster),
            "eval_support_b_target_query_values_loaded": 0,
            "members": [
                {"dataset_id": item.record.dataset_id, "series_uid": item.record.series_uid}
                for item in sorted(roster, key=lambda row: (row.record.dataset_id, row.record.series_uid))
            ],
        },
        "dataset_results": results,
        "p0_oracle_donor_headroom_gate": {
            "datasets_conjunctive": True,
            "pass": passed,
            "decision": decision,
            "p1_consumer_evaluation_run": False,
            "current_operator_revival_allowed": False,
        },
        "pass": passed,
        "verdict": "LEVEL_ADJUSTED_DONOR_PREMISE_PASS" if passed else "LEVEL_ADJUSTED_DONOR_PREMISE_FAIL",
        "consumer_fit_count": 0,
        "chronos_judge_call_count": 0,
        "fresh_evidence": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "information_wall": {
            "source_support_a_discovery_train_values_only": True,
            "loaded_value_series_count": len(roster),
            "eval_values_loaded": False,
            "support_b_target_query_values_loaded": False,
            "public_descriptors_use_corrupt_visible_values_only": True,
            "private_clean_used_only_for_local_scoring_and_grader_oracle": True,
            "dataset_id_used_for_audit_stratification_only": True,
        },
        "claim_limit": (
            "Development-only zero-fit donor-premise diagnostic on Traffic and METR. "
            "The best-donor oracle is privileged and non-deployable. Not P1, fresh "
            "evidence, a registered Program, Capability, promotion, Target, Query, or transfer."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=root / "artifacts/frozen/benchmark_v02/series_registry.jsonl")
    parser.add_argument("--split", type=Path, default=root / "artifacts/frozen/benchmark_v02/split_manifest.json")
    parser.add_argument("--support-a-subsplit", type=Path, default=root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json")
    parser.add_argument("--clean-root", type=Path, default=root / "data/benchmark_v0_2/clean_base")
    parser.add_argument("--output", type=Path, default=root / OUTPUT_RELATIVE_PATH)
    args = parser.parse_args()
    report = run_e2_source_level_adjusted_donor_premise(
        registry_path=args.registry, split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit, clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
