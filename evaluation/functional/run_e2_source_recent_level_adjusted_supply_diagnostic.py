"""Evaluate one unregistered recent-gap Program Supply prototype locally.

This development-only diagnostic uses only the frozen Source discovery training roster.
It performs no Consumer fit, Chronos evaluation, promotion, or Query access.
"""
from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_coherent_missingness_positive_control import (
    DATASET_SPECS,
    PHASE_CYCLES,
    PHASE_MIN_DONORS,
    RECENT_V2,
    TRAIN_SERIES_PER_DATASET,
    DatasetSpec,
    RosterItem,
    _center_scale,
    _fixed_gap_mask,
    select_roster,
)
from SelfEvolvingHarnessTS.operators.registry import get_operator


SCHEMA_VERSION = "e2-source-recent-level-adjusted-supply-diagnostic/1"
PROTOTYPE_ID = "local_level_adjusted_period_median_complete_v0"
MEDIAN_RECOVERY_MIN = 0.50
BETTER_THAN_BASELINE_MIN_COUNT = 10
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/"
    "source_recent_level_adjusted_supply_diagnostic_report.json"
)


@dataclass(frozen=True)
class PrototypeResult:
    artifact: np.ndarray
    fill_count: int
    abstain_count: int
    observed_collateral_change_count: int
    source_indices: tuple[int, ...]
    gap_diagnostics: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class DatasetResult:
    series_rows: list[dict[str, object]]
    gate: dict[str, object]


def local_level_adjusted_period_median_complete_v0(
    corrupt: np.ndarray,
    *,
    period: int,
    gap_bounds: tuple[tuple[int, int], ...],
    min_donors: int = PHASE_MIN_DONORS,
) -> PrototypeResult:
    """Fill gaps from immutable prior-cycle donors with robust local level offsets."""

    raw = np.asarray(corrupt, dtype=np.float64).copy()
    raw.setflags(write=False)
    output = raw.copy()
    original_missing = ~np.isfinite(raw)
    all_sources: set[int] = set()
    gap_rows: list[dict[str, object]] = []
    fill_count = 0
    abstain_count = 0
    for gap_start, gap_stop in gap_bounds:
        if not np.isnan(raw[gap_start:gap_stop]).all():
            raise ValueError("prototype gap bounds do not match the corrupt artifact")
        flank = np.arange(gap_start - period, gap_start, dtype=np.int64)
        offsets: dict[int, float] = {}
        flank_sources: set[int] = set()
        if int(flank[0]) >= 0 and np.isfinite(raw[flank]).all():
            flank_sources.update(int(index) for index in flank)
            for cycle in range(1, PHASE_CYCLES + 1):
                donor_flank = flank - cycle * period
                if int(donor_flank[0]) < 0 or not np.isfinite(raw[donor_flank]).all():
                    continue
                offsets[cycle] = float(np.median(raw[flank] - raw[donor_flank]))
                flank_sources.update(int(index) for index in donor_flank)

        point_rows: list[dict[str, object]] = []
        gap_sources = set(flank_sources)
        for index in range(gap_start, gap_stop):
            adjusted: list[float] = []
            donor_indices: list[int] = []
            for cycle, offset in offsets.items():
                donor_index = index - cycle * period
                if donor_index < 0 or donor_index >= gap_start:
                    continue
                if np.isfinite(raw[donor_index]):
                    adjusted.append(float(raw[donor_index] + offset))
                    donor_indices.append(donor_index)
            if len(adjusted) >= min_donors:
                output[index] = float(np.median(adjusted))
                fill_count += 1
                abstained = False
            else:
                abstain_count += 1
                abstained = True
            gap_sources.update(donor_indices)
            point_rows.append(
                {
                    "index": index,
                    "donor_indices": donor_indices,
                    "donor_count": len(donor_indices),
                    "abstained": abstained,
                }
            )
        if gap_sources and max(gap_sources) >= gap_start:
            raise AssertionError("prototype read a source at or after the gap start")
        all_sources.update(gap_sources)
        gap_rows.append(
            {
                "gap_bounds": [gap_start, gap_stop],
                "cycle_level_offsets": {
                    str(cycle): offset for cycle, offset in sorted(offsets.items())
                },
                "source_indices": sorted(gap_sources),
                "max_source_index": max(gap_sources) if gap_sources else None,
                "all_source_indices_before_gap_start": all(
                    source < gap_start for source in gap_sources
                ),
                "points": point_rows,
            }
        )

    collateral = int(np.count_nonzero(output[~original_missing] != raw[~original_missing]))
    return PrototypeResult(
        artifact=output,
        fill_count=fill_count,
        abstain_count=abstain_count,
        observed_collateral_change_count=collateral,
        source_indices=tuple(sorted(all_sources)),
        gap_diagnostics=tuple(gap_rows),
    )


def _series_result(
    *, spec: DatasetSpec, item: RosterItem, values: np.ndarray
) -> dict[str, object]:
    uid = item.record.series_uid
    clean = np.asarray(values[: spec.train_stop], dtype=np.float64).copy()
    if clean.shape != (spec.train_stop,) or not np.isfinite(clean).all():
        raise ValueError(f"invalid private clean train artifact: {uid}")
    mask = _fixed_gap_mask(spec.train_stop, geometry=RECENT_V2)
    corrupt = clean.copy()
    corrupt[mask] = np.nan
    immutable_before = corrupt.copy()

    current = np.asarray(
        get_operator("period_median_complete")(
            corrupt,
            period=spec.period,
            cycles=PHASE_CYCLES,
            min_donors=PHASE_MIN_DONORS,
        ),
        dtype=np.float64,
    )
    proposed = local_level_adjusted_period_median_complete_v0(
        corrupt,
        period=spec.period,
        gap_bounds=RECENT_V2.gap_bounds,
    )
    if not np.array_equal(corrupt, immutable_before, equal_nan=True):
        raise AssertionError("a supply candidate mutated the raw corrupt artifact")

    baseline = corrupt.copy()
    baseline[mask] = float(np.median(corrupt[np.isfinite(corrupt)]))
    _, scale, scale_method = _center_scale(clean)

    def gap_nmae(artifact: np.ndarray) -> float:
        return float(np.mean(np.abs(artifact[mask] - clean[mask])) / scale)

    current_nmae = gap_nmae(current)
    proposed_nmae = gap_nmae(proposed.artifact)
    baseline_nmae = gap_nmae(baseline)
    if baseline_nmae <= 0.0:
        raise ValueError(f"finite-median baseline has zero gap NMAE: {uid}")
    all_sources_before = all(
        bool(row["all_source_indices_before_gap_start"])
        for row in proposed.gap_diagnostics
    )
    mechanical_pass = (
        proposed.fill_count == RECENT_V2.gap_point_count
        and proposed.abstain_count == 0
        and proposed.observed_collateral_change_count == 0
        and all_sources_before
    )
    return {
        "series_uid": uid,
        "dataset_id": spec.dataset_id,
        "period": spec.period,
        "clean_train_scale": scale,
        "clean_train_scale_method": scale_method,
        "gap_point_count": RECENT_V2.gap_point_count,
        "current_gap_nmae": current_nmae,
        "proposed_gap_nmae": proposed_nmae,
        "finite_median_baseline_gap_nmae": baseline_nmae,
        "current_recovery_fraction": (baseline_nmae - current_nmae) / baseline_nmae,
        "proposed_recovery_fraction": (baseline_nmae - proposed_nmae) / baseline_nmae,
        "proposed_minus_current_gap_nmae": proposed_nmae - current_nmae,
        "proposed_better_than_finite_median": proposed_nmae < baseline_nmae,
        "proposed_no_worse_than_current": proposed_nmae <= current_nmae,
        "proposed_fill_count": proposed.fill_count,
        "proposed_fill_rate": proposed.fill_count / RECENT_V2.gap_point_count,
        "proposed_abstain_count": proposed.abstain_count,
        "proposed_observed_collateral_change_count": (
            proposed.observed_collateral_change_count
        ),
        "all_source_indices_before_gap_start": all_sources_before,
        "no_recursive_filled_value_reads": True,
        "no_gap_after_or_future_reads": all_sources_before,
        "raw_corrupt_artifact_immutable": True,
        "mechanical_pass": mechanical_pass,
        "gap_diagnostics": list(proposed.gap_diagnostics),
    }


def _dataset_result(
    *, spec: DatasetSpec, items: list[RosterItem], values_by_uid: dict[str, np.ndarray]
) -> DatasetResult:
    rows = [
        _series_result(spec=spec, item=item, values=values_by_uid[item.record.series_uid])
        for item in sorted(items, key=lambda candidate: candidate.record.series_uid)
    ]
    if len(rows) != TRAIN_SERIES_PER_DATASET:
        raise ValueError(f"unexpected train roster size: {spec.dataset_id}")
    median_recovery = statistics.median(
        float(row["proposed_recovery_fraction"]) for row in rows
    )
    better_count = sum(bool(row["proposed_better_than_finite_median"]) for row in rows)
    current_median = statistics.median(float(row["current_gap_nmae"]) for row in rows)
    proposed_median = statistics.median(float(row["proposed_gap_nmae"]) for row in rows)
    mechanical = all(bool(row["mechanical_pass"]) for row in rows)
    passed = (
        mechanical
        and median_recovery >= MEDIAN_RECOVERY_MIN
        and better_count >= BETTER_THAN_BASELINE_MIN_COUNT
        and proposed_median <= current_median
    )
    return DatasetResult(
        series_rows=rows,
        gate={
            "dataset_id": spec.dataset_id,
            "all_series_mechanical_pass": mechanical,
            "median_proposed_recovery_fraction": median_recovery,
            "median_recovery_min": MEDIAN_RECOVERY_MIN,
            "proposed_better_than_finite_median_count": better_count,
            "better_than_finite_median_min_count": BETTER_THAN_BASELINE_MIN_COUNT,
            "current_median_gap_nmae": current_median,
            "proposed_median_gap_nmae": proposed_median,
            "proposed_median_no_worse_than_current": proposed_median <= current_median,
            "pass": passed,
        },
    )


def run_e2_source_recent_level_adjusted_supply_diagnostic(
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
    train_items = [item for item in roster if item.cohort == "train"]
    values_by_uid = _load_values([item.record for item in train_items], clean_root)
    results: dict[str, DatasetResult] = {}
    for spec in DATASET_SPECS:
        items = [item for item in train_items if item.record.dataset_id == spec.dataset_id]
        results[spec.dataset_id] = _dataset_result(
            spec=spec, items=items, values_by_uid=values_by_uid
        )
    passed = all(bool(result.gate["pass"]) for result in results.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "development_program_supply_local_diagnostic",
        "prototype_id": PROTOTYPE_ID,
        "prototype_not_registered": True,
        "configuration": {
            "datasets": [spec.dataset_id for spec in DATASET_SPECS],
            "geometry_id": RECENT_V2.geometry_id,
            "gap_bounds_absolute_half_open": [list(gap) for gap in RECENT_V2.gap_bounds],
            "gap_point_count": RECENT_V2.gap_point_count,
            "period_by_dataset": {spec.dataset_id: spec.period for spec in DATASET_SPECS},
            "cycles": PHASE_CYCLES,
            "min_donors": PHASE_MIN_DONORS,
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "compared_supplies": [
                "registered_period_median_complete",
                PROTOTYPE_ID,
                "full_artifact_finite_median",
                "private_clean_scoring_only",
            ],
        },
        "roster": {
            "selection": selection,
            "loaded_value_cohort": "train_only",
            "eval_value_series_loaded": 0,
            "train_members": [
                {
                    "dataset_id": item.record.dataset_id,
                    "series_uid": item.record.series_uid,
                    "cohort": item.cohort,
                }
                for item in sorted(
                    train_items,
                    key=lambda row: (row.record.dataset_id, row.record.series_uid),
                )
            ],
        },
        "per_series_results": {
            dataset_id: result.series_rows for dataset_id, result in results.items()
        },
        "dataset_gates": {
            dataset_id: result.gate for dataset_id, result in results.items()
        },
        "all_dataset_gates_conjunctive": True,
        "pass": passed,
        "verdict": (
            "LEVEL_ADJUSTED_SUPPLY_LOCAL_GATE_PASS"
            if passed
            else "LEVEL_ADJUSTED_SUPPLY_LOCAL_GATE_FAIL"
        ),
        "consumer_fit_count": 0,
        "chronos_judge_call_count": 0,
        "agent_enabled": False,
        "memory_enabled": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "fresh_evidence": False,
        "target_query_opened": False,
        "information_wall": {
            "source_support_a_discovery_train_values_only": True,
            "eval_values_loaded": False,
            "support_b_values_loaded": False,
            "target_or_query_values_loaded": False,
            "prototype_reads_private_clean_values_during_execution": False,
            "private_clean_used_for_local_scoring_only": True,
        },
        "claim_limit": (
            "Development-only same-family Program Supply local gate for an unregistered "
            "typed prototype. Not Consumer evidence, fresh evidence, Capability, "
            "promotion, Memory, Target, Query, or transfer evidence."
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
    report = run_e2_source_recent_level_adjusted_supply_diagnostic(
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
