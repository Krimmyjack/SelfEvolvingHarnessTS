"""Run the E2 natural Source-A evidence positive-control gate.

The sixteen-case roster is fixed from frozen visible metadata before selected
values are loaded and before any Judge outcome exists.  Discovery and validation
remain distinct population subsets; the menu oracle is diagnostic only.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

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
    GLOBAL_FEATURE_NAMES,
    LOCAL_EXTRA_FEATURE_NAMES,
    PROGRAM_IDS,
    _execute_program,
    _global_features,
    _local_features,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


SOURCE_DATASETS = ("monash:traffic_hourly", "metr_la")
SUBSPLITS = ("support_a_discovery", "support_a_validation")
SUPPORT_A_SUBSPLIT_SCHEMA = "benchmark-support-a-subsplit/2"
WINDOWS = {
    "support_a_discovery": {"context": (736, 928), "future": (928, 976)},
    "support_a_validation": {"context": (784, 976), "future": (976, 1024)},
}
GAP_BOUNDS = (156, 180)
DESIRED_STRATA = {
    "monash:traffic_hourly": {"seasonal_high": 2, "structured_mixed": 2},
    "metr_la": {"low_structure": 2, "structured_mixed": 2},
}
HEADROOM_MIN = 0.01
WINNER_MARGIN_MIN = 0.005
DISTINCT_ACTIONS_MIN = 2


class _Receipt(Protocol):
    loss_j: float


class _Valuator(Protocol):
    def evaluate(
        self,
        prepared_context: np.ndarray,
        clean_future: np.ndarray,
        *,
        scale_context: np.ndarray,
    ) -> _Receipt: ...


@dataclass(frozen=True)
class SourceRosterItem:
    record: SeriesRecord
    assignment: SplitAssignment
    subsplit: str


@dataclass(frozen=True)
class SourceCase:
    roster_item: SourceRosterItem
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray


def _read_subsplit(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") != SUPPORT_A_SUBSPLIT_SCHEMA:
        raise ValueError("unsupported Support-A subsplit schema")
    members = payload.get("members")
    if not isinstance(members, Mapping) or set(members) != set(SUBSPLITS):
        raise ValueError("Support-A subsplit members differ from the frozen schema")
    result: dict[str, set[str]] = {}
    for name in SUBSPLITS:
        raw_uids = members[name]
        if not isinstance(raw_uids, list) or not all(
            isinstance(uid, str) and uid for uid in raw_uids
        ):
            raise ValueError(f"invalid Support-A member list: {name}")
        if len(raw_uids) != len(set(raw_uids)):
            raise ValueError(f"duplicate Support-A member: {name}")
        result[name] = set(raw_uids)
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or any(
        counts.get(name) != len(result[name]) for name in SUBSPLITS
    ):
        raise ValueError("Support-A member counts disagree with frozen metadata")
    if result[SUBSPLITS[0]] & result[SUBSPLITS[1]]:
        raise ValueError("Support-A discovery and validation overlap")
    return result


def select_source_roster(
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
) -> tuple[list[SourceRosterItem], dict[str, object]]:
    """Fix 4 cases per dataset/subsplit using public metadata and UID only."""

    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    member_uids = _read_subsplit(support_a_subsplit_path)
    assignment_by_uid = {row.series_uid: row for row in manifest.assignments}
    for subsplit, uids in member_uids.items():
        for uid in uids:
            assignment = assignment_by_uid.get(uid)
            if assignment is None:
                raise ValueError(f"subsplit UID absent from split manifest: {uid}")
            if assignment.role is not SplitRole.SUPPORT_A:
                raise ValueError(f"subsplit member is not Support-A: {subsplit}/{uid}")

    selected: list[SourceRosterItem] = []
    cell_metadata: dict[str, dict[str, object]] = {}
    for dataset_id in SOURCE_DATASETS:
        desired = DESIRED_STRATA[dataset_id]
        for subsplit in SUBSPLITS:
            candidates: list[SourceRosterItem] = []
            for uid in member_uids[subsplit]:
                assignment = assignment_by_uid.get(uid)
                if assignment is None or assignment.dataset_id != dataset_id:
                    continue
                record = records.get(uid)
                if record is None:
                    raise ValueError(f"split UID absent from registry: {uid}")
                if record.dataset_id != assignment.dataset_id:
                    raise ValueError(f"registry/split dataset mismatch: {uid}")
                if record.regime_tag != assignment.regime_tag:
                    raise ValueError(f"registry/split regime mismatch: {uid}")
                if record.admission_reasons != ():
                    raise ValueError(f"ineligible Support-A record: {uid}")
                if SplitRole.SUPPORT_A.value not in record.roles_allowed:
                    raise ValueError(f"record disallows Support-A: {uid}")
                candidates.append(SourceRosterItem(record, assignment, subsplit))
            candidates.sort(key=lambda item: item.record.series_uid)

            cell_selected: list[SourceRosterItem] = []
            selected_uids: set[str] = set()
            deficits: dict[str, int] = {}
            for stratum, desired_count in desired.items():
                matches = [
                    item
                    for item in candidates
                    if item.assignment.regime_tag == stratum
                    and item.record.series_uid not in selected_uids
                ]
                chosen = matches[:desired_count]
                cell_selected.extend(chosen)
                selected_uids.update(item.record.series_uid for item in chosen)
                if len(chosen) < desired_count:
                    deficits[stratum] = desired_count - len(chosen)

            fallback_needed = 4 - len(cell_selected)
            if fallback_needed:
                fallback = [
                    item
                    for item in candidates
                    if item.record.series_uid not in selected_uids
                ][:fallback_needed]
                cell_selected.extend(fallback)
            if len(cell_selected) != 4:
                raise ValueError(f"fewer than four candidates: {dataset_id}/{subsplit}")
            selected.extend(cell_selected)

            actual = Counter(item.assignment.regime_tag for item in cell_selected)
            available = Counter(item.assignment.regime_tag for item in candidates)
            cell_metadata[f"{dataset_id}/{subsplit}"] = {
                "method": "visible_metadata_then_series_uid_ascending",
                "candidate_count": len(candidates),
                "available_strata": dict(sorted(available.items())),
                "desired_strata": dict(desired),
                "actual_strata": dict(sorted(actual.items())),
                "fallback_used": bool(deficits),
                "fallback_deficits": deficits,
            }

    selected_uids = [item.record.series_uid for item in selected]
    if len(selected) != 16 or len(selected_uids) != len(set(selected_uids)):
        raise AssertionError("source roster must contain sixteen unique cases")
    return selected, {
        "fixed_before_value_loading_and_judge": True,
        "cells": cell_metadata,
        "fallback_used": any(
            bool(metadata["fallback_used"]) for metadata in cell_metadata.values()
        ),
    }


def load_source_cases(
    roster: list[SourceRosterItem],
    clean_root: Path,
) -> list[SourceCase]:
    """Load and verify values for only the already-fixed sixteen-case roster."""

    values_by_uid = _load_values([item.record for item in roster], clean_root)
    cases: list[SourceCase] = []
    for item in roster:
        boundaries = item.assignment.chronological_boundaries
        if boundaries is None:
            raise ValueError(f"missing chronological boundaries: {item.record.series_uid}")
        if tuple(boundaries.get("train", ())) != (0, 928):
            raise ValueError(f"unexpected train boundary: {item.record.series_uid}")
        if tuple(boundaries.get("validation", ())) != (928, 976):
            raise ValueError(f"unexpected validation boundary: {item.record.series_uid}")
        if tuple(boundaries.get("test", ())) != (976, 1024):
            raise ValueError(f"unexpected test boundary: {item.record.series_uid}")

        windows = WINDOWS[item.subsplit]
        values = values_by_uid[item.record.series_uid]
        clean_context = values[slice(*windows["context"])].copy()
        clean_future = values[slice(*windows["future"])].copy()
        if clean_context.shape != (192,) or clean_future.shape != (48,):
            raise ValueError(f"insufficient fixed window: {item.record.series_uid}")
        if not np.isfinite(clean_context).all() or not np.isfinite(clean_future).all():
            raise ValueError(f"natural missingness enters fixed window: {item.record.series_uid}")
        corrupt_context = clean_context.copy()
        corrupt_context[slice(*GAP_BOUNDS)] = np.nan
        if int(np.isnan(corrupt_context).sum()) != 24:
            raise AssertionError("fixed corruption must inject exactly 24 NaNs")
        cases.append(SourceCase(item, clean_context, corrupt_context, clean_future))
    return cases


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty source cohort")
    fixed_mean_losses = {
        action: statistics.fmean(row["loss_by_action"][action] for row in rows)
        for action in PROGRAM_IDS
    }
    best_fixed_action = min(
        PROGRAM_IDS, key=lambda action: (fixed_mean_losses[action], action)
    )
    menu_oracle_mean = statistics.fmean(row["menu_oracle_loss"] for row in rows)
    qualifying_wins = {
        action: sum(
            row["winner"] == action and row["winner_margin"] >= WINNER_MARGIN_MIN
            for row in rows
        )
        for action in PROGRAM_IDS
    }
    return {
        "case_count": len(rows),
        "fixed_mean_loss_by_action": fixed_mean_losses,
        "best_fixed_action": best_fixed_action,
        "best_fixed_mean_loss": fixed_mean_losses[best_fixed_action],
        "menu_oracle_mean_loss": menu_oracle_mean,
        "routing_headroom": fixed_mean_losses[best_fixed_action] - menu_oracle_mean,
        "qualifying_wins_by_action": qualifying_wins,
    }


def run_e2_natural_source_evidence(
    valuator: _Valuator,
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    roster, selection = select_source_roster(
        registry_path, split_path, support_a_subsplit_path
    )
    cases = load_source_cases(roster, clean_root)

    rows: list[dict[str, Any]] = []
    judge_calls = 0
    for case in cases:
        global_features, observed_period = _global_features(case.corrupt_context)
        local_features = _local_features(case.corrupt_context, observed_period)
        global_feature_map = {
            name: float(global_features[name]) for name in GLOBAL_FEATURE_NAMES
        }
        local_feature_map = {
            name: float(local_features[name]) for name in LOCAL_EXTRA_FEATURE_NAMES
        }

        losses: dict[str, float] = {}
        for program_id in PROGRAM_IDS:
            prepared = _execute_program(
                program_id,
                case.corrupt_context,
                observed_period=observed_period,
            )
            receipt = valuator.evaluate(
                prepared,
                case.clean_future,
                scale_context=case.clean_context,
            )
            judge_calls += 1
            loss = float(receipt.loss_j)
            if not np.isfinite(loss):
                raise ValueError(
                    f"non-finite Judge loss: {case.roster_item.record.series_uid}/{program_id}"
                )
            losses[program_id] = loss

        ranked = sorted((loss, action) for action, loss in losses.items())
        winner_loss, winner = ranked[0]
        item = case.roster_item
        windows = WINDOWS[item.subsplit]
        rows.append(
            {
                "series_uid": item.record.series_uid,
                "dataset_id": item.record.dataset_id,
                "split": SplitRole.SUPPORT_A.value,
                "subsplit": item.subsplit,
                "regime_tag": item.assignment.regime_tag,
                "windows": {
                    "context": list(windows["context"]),
                    "future": list(windows["future"]),
                    "gap_relative_to_context": list(GAP_BOUNDS),
                },
                "observed_period": observed_period,
                "global_features": global_feature_map,
                "local_features": local_feature_map,
                "loss_by_action": losses,
                "winner": winner,
                "winner_margin": float(ranked[1][0] - winner_loss),
                "menu_oracle_loss": winner_loss,
            }
        )

    expected_calls = len(cases) * len(PROGRAM_IDS)
    if expected_calls != 48 or judge_calls != expected_calls:
        raise AssertionError(f"expected exactly 48 Judge calls, observed {judge_calls}")

    overall = _aggregate(rows)
    per_dataset = {
        dataset_id: _aggregate(
            [row for row in rows if row["dataset_id"] == dataset_id]
        )
        for dataset_id in SOURCE_DATASETS
    }
    heterogeneous_actions = [
        action
        for action in PROGRAM_IDS
        if overall["qualifying_wins_by_action"][action] >= 1
    ]
    dataset_non_identity_pass = {
        dataset_id: any(
            action != "identity" and count >= 1
            for action, count in aggregate["qualifying_wins_by_action"].items()
        )
        for dataset_id, aggregate in per_dataset.items()
    }
    headroom_pass = float(overall["routing_headroom"]) >= HEADROOM_MIN
    heterogeneity_pass = len(heterogeneous_actions) >= DISTINCT_ACTIONS_MIN
    source_action_pass = all(dataset_non_identity_pass.values())

    return {
        "schema_version": "e2-natural-source-evidence/1",
        "scientific_role": "natural_source_evidence_positive_control",
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "split": SplitRole.SUPPORT_A.value,
            "support_a_subsplit_schema": SUPPORT_A_SUBSPLIT_SCHEMA,
            "windows_by_subsplit": {
                name: {key: list(bounds) for key, bounds in WINDOWS[name].items()}
                for name in SUBSPLITS
            },
            "gap_relative_to_context": list(GAP_BOUNDS),
            "programs": list(PROGRAM_IDS),
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
            "local_feature_names": list(LOCAL_EXTRA_FEATURE_NAMES),
            "valuator": "FrozenChronosValuator",
            "agent_enabled": False,
            "memory_enabled": False,
            "adaptation_enabled": False,
        },
        "roster_selection_before_value_loading_and_judge": selection,
        "judge_call_count": judge_calls,
        "cases": rows,
        "aggregate": {"overall": overall, "per_dataset": per_dataset},
        "gates": {
            "overall_routing_headroom": {
                "threshold": HEADROOM_MIN,
                "value": overall["routing_headroom"],
                "pass": headroom_pass,
            },
            "overall_action_heterogeneity": {
                "winner_margin_min": WINNER_MARGIN_MIN,
                "distinct_actions_min": DISTINCT_ACTIONS_MIN,
                "qualifying_actions": heterogeneous_actions,
                "pass": heterogeneity_pass,
            },
            "per_dataset_non_identity_evidence": {
                "winner_margin_min": WINNER_MARGIN_MIN,
                "pass_by_dataset": dataset_non_identity_pass,
                "pass": source_action_pass,
            },
        },
        "all_gates_pass": headroom_pass and heterogeneity_pass and source_action_pass,
        "claim_limit": (
            "Natural source evidence positive control only; not evidence of transfer, "
            "target-query performance, Memory, adaptation, or promotion."
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
        default=project_root
        / "artifacts/functional/e2/natural_source_evidence_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_source_evidence(
        FrozenChronosValuator(),
        registry_path=args.registry,
        split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"all_gates_pass={report['all_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_source_cases",
    "run_e2_natural_source_evidence",
    "select_source_roster",
]
