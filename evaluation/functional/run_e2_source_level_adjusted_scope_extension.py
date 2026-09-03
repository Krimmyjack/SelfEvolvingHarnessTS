"""Run the zero-fit local scope extension for the level-adjusted prototype."""
from __future__ import annotations

import argparse
from pathlib import Path

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
    RosterItem,
    select_roster,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_coherent_missingness_positive_control import (
    PHASE_CYCLES,
    PHASE_MIN_DONORS,
    RECENT_V2,
    DatasetSpec,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_recent_level_adjusted_supply_diagnostic import (
    BETTER_THAN_BASELINE_MIN_COUNT,
    MEDIAN_RECOVERY_MIN,
    PROTOTYPE_ID,
    DatasetResult,
    _dataset_result,
)


SCHEMA_VERSION = "e2-source-level-adjusted-scope-extension/1"
TRAIN_SERIES_PER_DATASET = 12
SCOPE_DATASET_SPECS = (
    DatasetSpec("monash:traffic_hourly", 24, "hourly", 928, (928, 976)),
    DatasetSpec("metr_la", 24, "hourly", 928, (928, 976)),
)
OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/source_level_adjusted_scope_extension_report.json"
)


def _scope_train_roster(roster: list[RosterItem]) -> list[RosterItem]:
    """Take the first 12 UIDs from each already-frozen 32-series train cohort."""

    selected: list[RosterItem] = []
    for spec in SCOPE_DATASET_SPECS:
        frozen_train = sorted(
            (
                item
                for item in roster
                if item.cohort == "train" and item.record.dataset_id == spec.dataset_id
            ),
            key=lambda item: item.record.series_uid,
        )
        if len(frozen_train) != 32:
            raise ValueError(f"expected frozen 32-series train cohort: {spec.dataset_id}")
        if any(item.record.frequency != spec.frequency for item in frozen_train):
            raise ValueError(f"registry frequency disagrees with DatasetSpec: {spec.dataset_id}")
        selected.extend(frozen_train[:TRAIN_SERIES_PER_DATASET])
    uids = [item.record.series_uid for item in selected]
    if len(uids) != 24 or len(set(uids)) != 24:
        raise AssertionError("scope-extension roster must contain 24 unique train series")
    return selected


def run_e2_source_level_adjusted_scope_extension(
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    premise_roster, premise_selection = select_roster(
        registry_path=registry_path,
        split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
    )
    roster = _scope_train_roster(premise_roster)
    values_by_uid = _load_values([item.record for item in roster], clean_root)

    results: dict[str, DatasetResult] = {}
    for spec in SCOPE_DATASET_SPECS:
        items = [item for item in roster if item.record.dataset_id == spec.dataset_id]
        results[spec.dataset_id] = _dataset_result(
            spec=spec, items=items, values_by_uid=values_by_uid
        )
    passed = all(bool(result.gate["pass"]) for result in results.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "development_program_supply_scope_extension_local_gate",
        "scope_hypothesis_status": "PROPOSED",
        "prototype_id": PROTOTYPE_ID,
        "prototype_not_registered": True,
        "configuration": {
            "datasets": [spec.dataset_id for spec in SCOPE_DATASET_SPECS],
            "geometry_id": RECENT_V2.geometry_id,
            "gap_bounds_absolute_half_open": [list(gap) for gap in RECENT_V2.gap_bounds],
            "gap_point_count": RECENT_V2.gap_point_count,
            "cycles": PHASE_CYCLES,
            "min_donors": PHASE_MIN_DONORS,
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "period_binding_by_dataset": {
                spec.dataset_id: {
                    "registry_frequency": spec.frequency,
                    "period_steps": spec.period,
                    "effective_period": "24 hours",
                    "binding_source": "frozen_dataset_spec_checked_against_registry",
                }
                for spec in SCOPE_DATASET_SPECS
            },
            "thresholds_unchanged": {
                "median_recovery_min": MEDIAN_RECOVERY_MIN,
                "better_than_finite_median_min_count": BETTER_THAN_BASELINE_MIN_COUNT,
                "proposed_median_gap_nmae_must_not_exceed_current": True,
            },
        },
        "scope_hypothesis": {
            "predicate": "gap_length / period <= 0.5",
            "observable_inputs_only": ["gap_length", "period"],
            "gap_length": 6,
            "period": 24,
            "gap_length_over_period": 0.25,
            "threshold": 0.5,
            "predicate_satisfied": True,
            "dataset_id_used_as_predicate": False,
            "dataset_id_role": "audit_stratification_only",
        },
        "roster": {
            "premise_roster_selection": premise_selection,
            "selection_rule": (
                "within each frozen UID-sorted 32-series premise train cohort, "
                "take the first 12 UIDs"
            ),
            "fixed_before_value_loading": True,
            "loaded_value_series_count": len(roster),
            "eval_value_series_loaded": 0,
            "members": [
                {
                    "dataset_id": item.record.dataset_id,
                    "series_uid": item.record.series_uid,
                    "cohort": item.cohort,
                }
                for item in sorted(
                    roster,
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
            "LEVEL_ADJUSTED_SCOPE_EXTENSION_LOCAL_GATE_PASS"
            if passed
            else "LEVEL_ADJUSTED_SCOPE_EXTENSION_LOCAL_GATE_FAIL"
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
            "dataset_id_not_used_by_scope_predicate": True,
        },
        "claim_limit": (
            "Development-only local scope-extension diagnostic for a PROPOSED, "
            "unregistered Program Supply hypothesis. Not fresh evidence, Consumer "
            "evidence, Capability, promotion, Memory, Target, Query, or transfer evidence."
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
    report = run_e2_source_level_adjusted_scope_extension(
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
