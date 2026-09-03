"""Run an exposed-Source diagnostic of gap geometry and action response.

This development-only sweep reuses every Support-A discovery case from the
existing natural Source evidence report.  It varies only the length of a gap
ending at relative context index 180, re-extracts the public period for every
gap, and measures the fixed three-program menu.  The 24-point cell is replayed
from the existing report; only the 6- and 48-point cells incur new Judge calls.

The report can establish an action-heterogeneity premise on already exposed
Source data.  It cannot promote a Capability or support Memory/Target/transfer
claims.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    PROGRAM_IDS,
    _execute_program,
    _global_features,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_natural_source_evidence import (
    SOURCE_DATASETS,
    WINDOWS,
    SourceCase,
    load_source_cases,
    select_source_roster,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


DISCOVERY_SUBSPLIT = "support_a_discovery"
GAP_END = 180
GAP_LENGTHS = (6, 24, 48)
WINNER_MARGIN_MIN = 0.005
QUALIFIED_SWITCH_CASES_MIN = 2
SWITCH_DATASETS_MIN = 2
ROUTING_HEADROOM_MIN = 0.01
QUALIFYING_ACTIONS_PER_GAP_MIN = 2


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


def _read_discovery_cache(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = json.loads(path.read_text("utf-8"))
    if report.get("schema_version") != "e2-natural-source-evidence/1":
        raise ValueError("unsupported natural Source evidence report schema")
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("natural Source evidence report has no case list")
    discovery = [row for row in raw_cases if row.get("subsplit") == DISCOVERY_SUBSPLIT]
    if len(discovery) != 8:
        raise ValueError("expected all eight exposed Support-A discovery cases")
    by_uid: dict[str, dict[str, Any]] = {}
    for row in discovery:
        uid = row.get("series_uid")
        losses = row.get("loss_by_action")
        if not isinstance(uid, str) or not uid:
            raise ValueError("discovery cache contains an invalid series UID")
        if uid in by_uid:
            raise ValueError(f"duplicate discovery cache UID: {uid}")
        if not isinstance(losses, Mapping) or set(losses) != set(PROGRAM_IDS):
            raise ValueError(f"incomplete cached action menu: {uid}")
        if any(not np.isfinite(float(losses[action])) for action in PROGRAM_IDS):
            raise ValueError(f"non-finite cached Judge loss: {uid}")
        by_uid[uid] = row
    return by_uid, report


def _load_discovery_cases(
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
    cached_by_uid: Mapping[str, Mapping[str, Any]],
) -> tuple[list[SourceCase], dict[str, Any]]:
    roster, selection = select_source_roster(
        registry_path, split_path, support_a_subsplit_path
    )
    discovery_roster = [item for item in roster if item.subsplit == DISCOVERY_SUBSPLIT]
    if len(discovery_roster) != 8:
        raise AssertionError("fixed roster must contain eight discovery cases")
    cases = load_source_cases(discovery_roster, clean_root)
    loaded_uids = {case.roster_item.record.series_uid for case in cases}
    if loaded_uids != set(cached_by_uid):
        raise ValueError("loaded discovery roster differs from the exposed cached roster")

    declared_windows = {
        "context": list(WINDOWS[DISCOVERY_SUBSPLIT]["context"]),
        "future": list(WINDOWS[DISCOVERY_SUBSPLIT]["future"]),
        "gap_relative_to_context": [156, GAP_END],
    }
    for uid, row in cached_by_uid.items():
        if row.get("dataset_id") not in SOURCE_DATASETS:
            raise ValueError(f"unexpected discovery dataset: {uid}")
        if row.get("windows") != declared_windows:
            raise ValueError(f"cached windows differ from frozen discovery windows: {uid}")
    return cases, selection


def _rank_losses(losses: Mapping[str, float]) -> tuple[str, float, float, str | None]:
    ranked = sorted((float(losses[action]), action) for action in PROGRAM_IDS)
    best_loss, winner = ranked[0]
    margin = float(ranked[1][0] - best_loss)
    qualified = winner if margin >= WINNER_MARGIN_MIN else None
    return winner, best_loss, margin, qualified


def _aggregate_gap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed_mean_losses = {
        action: statistics.fmean(row["loss_by_action"][action] for row in rows)
        for action in PROGRAM_IDS
    }
    best_fixed_action = min(
        PROGRAM_IDS, key=lambda action: (fixed_mean_losses[action], action)
    )
    menu_oracle_mean_loss = statistics.fmean(row["menu_oracle_loss"] for row in rows)
    qualifying_wins = {
        action: sum(row["qualified_winner"] == action for row in rows)
        for action in PROGRAM_IDS
    }
    return {
        "case_count": len(rows),
        "fixed_mean_loss_by_action": fixed_mean_losses,
        "best_fixed_action": best_fixed_action,
        "best_fixed_mean_loss": fixed_mean_losses[best_fixed_action],
        "menu_oracle_mean_loss": menu_oracle_mean_loss,
        "routing_headroom": fixed_mean_losses[best_fixed_action]
        - menu_oracle_mean_loss,
        "qualifying_wins_by_action": qualifying_wins,
    }


def run_e2_natural_gap_geometry_sweep(
    valuator: _Valuator,
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
    source_evidence_report_path: Path,
) -> dict[str, object]:
    cached_by_uid, cached_report = _read_discovery_cache(source_evidence_report_path)
    cases, selection = _load_discovery_cases(
        registry_path=registry_path,
        split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
        clean_root=clean_root,
        cached_by_uid=cached_by_uid,
    )

    rows_by_gap: dict[int, list[dict[str, Any]]] = {gap: [] for gap in GAP_LENGTHS}
    judge_calls = 0
    cached_losses = 0
    for case in cases:
        item = case.roster_item
        uid = item.record.series_uid
        cached_row = cached_by_uid[uid]
        for gap_length in GAP_LENGTHS:
            gap_bounds = (GAP_END - gap_length, GAP_END)
            corrupt_context = case.clean_context.copy()
            corrupt_context[slice(*gap_bounds)] = np.nan
            if int(np.isnan(corrupt_context).sum()) != gap_length:
                raise AssertionError(f"gap injection failed: {uid}/g={gap_length}")

            _, observed_period = _global_features(corrupt_context)
            losses: dict[str, float] = {}
            provenance: dict[str, str] = {}
            for action in PROGRAM_IDS:
                prepared = _execute_program(
                    action,
                    corrupt_context,
                    observed_period=observed_period,
                )
                if gap_length == 24:
                    losses[action] = float(cached_row["loss_by_action"][action])
                    provenance[action] = "cached"
                    cached_losses += 1
                else:
                    receipt = valuator.evaluate(
                        prepared,
                        case.clean_future,
                        scale_context=case.clean_context,
                    )
                    judge_calls += 1
                    loss = float(receipt.loss_j)
                    if not np.isfinite(loss):
                        raise ValueError(f"non-finite Judge loss: {uid}/g={gap_length}/{action}")
                    losses[action] = loss
                    provenance[action] = "new"

            if gap_length == 24 and observed_period != int(cached_row["observed_period"]):
                raise ValueError(
                    f"24-point observed period no longer replays: {uid}: "
                    f"{observed_period} != {cached_row['observed_period']}"
                )
            winner, winner_loss, winner_margin, qualified_winner = _rank_losses(losses)
            rows_by_gap[gap_length].append(
                {
                    "series_uid": uid,
                    "dataset_id": item.record.dataset_id,
                    "split": "support_a",
                    "subsplit": DISCOVERY_SUBSPLIT,
                    "regime_tag": item.assignment.regime_tag,
                    "windows": {
                        "context": list(WINDOWS[DISCOVERY_SUBSPLIT]["context"]),
                        "future": list(WINDOWS[DISCOVERY_SUBSPLIT]["future"]),
                        "gap_relative_to_context": list(gap_bounds),
                    },
                    "gap_length": gap_length,
                    "observed_period": observed_period,
                    "gap_to_period_ratio": float(gap_length / observed_period),
                    "loss_by_action": losses,
                    "winner": winner,
                    "winner_margin": winner_margin,
                    "qualified_winner": qualified_winner,
                    "menu_oracle_loss": winner_loss,
                    "cached_or_new": provenance,
                }
            )

    if judge_calls != 48:
        raise AssertionError(f"expected exactly 48 new Judge calls, observed {judge_calls}")
    if cached_losses != 24:
        raise AssertionError(f"expected exactly 24 cached losses, observed {cached_losses}")

    aggregate_by_gap = {
        str(gap): _aggregate_gap(rows_by_gap[gap]) for gap in GAP_LENGTHS
    }
    across_gap_cases: list[dict[str, Any]] = []
    for case in cases:
        uid = case.roster_item.record.series_uid
        case_rows = [
            next(row for row in rows_by_gap[gap] if row["series_uid"] == uid)
            for gap in GAP_LENGTHS
        ]
        raw_winners = {str(row["gap_length"]): row["winner"] for row in case_rows}
        distinct_qualified = sorted(
            {
                str(row["qualified_winner"])
                for row in case_rows
                if row["qualified_winner"] is not None
            }
        )
        across_gap_cases.append(
            {
                "series_uid": uid,
                "dataset_id": case.roster_item.record.dataset_id,
                "raw_winner_by_gap": raw_winners,
                "raw_winner_switch": len(set(raw_winners.values())) >= 2,
                "distinct_qualified_actions": distinct_qualified,
                "qualified_action_switch": len(distinct_qualified) >= 2,
            }
        )

    qualified_switch_cases = sum(
        bool(row["qualified_action_switch"]) for row in across_gap_cases
    )
    switch_datasets = sorted(
        {
            str(row["dataset_id"])
            for row in across_gap_cases
            if row["qualified_action_switch"]
        }
    )
    highest_headroom_gap = max(
        GAP_LENGTHS,
        key=lambda gap: (float(aggregate_by_gap[str(gap)]["routing_headroom"]), -gap),
    )
    max_headroom = float(aggregate_by_gap[str(highest_headroom_gap)]["routing_headroom"])
    qualifying_action_count_by_gap = {
        str(gap): sum(
            count > 0
            for count in aggregate_by_gap[str(gap)]["qualifying_wins_by_action"].values()
        )
        for gap in GAP_LENGTHS
    }
    geometry_switch_pass = qualified_switch_cases >= QUALIFIED_SWITCH_CASES_MIN
    dataset_switch_pass = len(switch_datasets) >= SWITCH_DATASETS_MIN
    headroom_pass = max_headroom >= ROUTING_HEADROOM_MIN
    within_gap_heterogeneity_pass = (
        max(qualifying_action_count_by_gap.values()) >= QUALIFYING_ACTIONS_PER_GAP_MIN
    )
    all_gates_pass = (
        geometry_switch_pass
        and dataset_switch_pass
        and headroom_pass
        and within_gap_heterogeneity_pass
    )

    return {
        "schema_version": "e2-natural-gap-geometry-sweep/1",
        "scientific_role": "exposed_source_gap_geometry_premise_diagnostic",
        "promotion_eligible": False,
        "target_transfer_eligible": False,
        "interpretation_boundary": {
            "source_discovery_only": True,
            "target_or_query_read": False,
            "period_reestimated_for_each_gap": True,
            "seasonal_period_rebound_for_each_gap": True,
            "causal_limit": (
                "Winner switches describe the complete fixed-Harness response to changed "
                "gap geometry, including period re-estimation and Seasonal rebinding; they "
                "cannot be attributed to gap-to-period ratio alone."
            ),
        },
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "split": "support_a",
            "subsplit": DISCOVERY_SUBSPLIT,
            "gap_end_relative_to_context": GAP_END,
            "gap_lengths": list(GAP_LENGTHS),
            "programs": list(PROGRAM_IDS),
            "winner_margin_min": WINNER_MARGIN_MIN,
            "valuator": "FrozenChronosValuator",
            "agent_enabled": False,
            "memory_enabled": False,
            "adaptation_enabled": False,
        },
        "source_evidence_report": {
            "path": str(source_evidence_report_path),
            "schema_version": cached_report["schema_version"],
            "discovery_case_count": len(cached_by_uid),
            "g24_losses_reused": True,
        },
        "roster_selection_before_value_loading_and_judge": selection,
        "new_judge_call_count": judge_calls,
        "cached_loss_count": cached_losses,
        "cases_by_gap": {str(gap): rows_by_gap[gap] for gap in GAP_LENGTHS},
        "aggregate_by_gap": aggregate_by_gap,
        "across_gap_cases": across_gap_cases,
        "highest_routing_headroom_gap": highest_headroom_gap,
        "gates": {
            "qualified_action_switch_cases": {
                "threshold": QUALIFIED_SWITCH_CASES_MIN,
                "value": qualified_switch_cases,
                "pass": geometry_switch_pass,
            },
            "switch_datasets": {
                "threshold": SWITCH_DATASETS_MIN,
                "datasets": switch_datasets,
                "value": len(switch_datasets),
                "pass": dataset_switch_pass,
            },
            "maximum_routing_headroom": {
                "threshold": ROUTING_HEADROOM_MIN,
                "value": max_headroom,
                "gap_length": highest_headroom_gap,
                "pass": headroom_pass,
            },
            "within_gap_action_heterogeneity": {
                "distinct_actions_min": QUALIFYING_ACTIONS_PER_GAP_MIN,
                "qualifying_action_count_by_gap": qualifying_action_count_by_gap,
                "pass": within_gap_heterogeneity_pass,
            },
        },
        "all_gates_pass": all_gates_pass,
        "verdict": (
            "GEOMETRY_PREMISE_PRESENT" if all_gates_pass else "GEOMETRY_PREMISE_WEAK"
        ),
        "claim_limit": (
            "Already-exposed Source evidence may support only a gap-geometry/action-response "
            "heterogeneity premise; it is not Capability, promotion, Memory, Target, or "
            "transfer evidence."
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
        "--source-evidence-report",
        type=Path,
        default=project_root / "artifacts/functional/e2/natural_source_evidence_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_gap_geometry_sweep_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_gap_geometry_sweep(
        FrozenChronosValuator(),
        registry_path=args.registry,
        split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit,
        clean_root=args.clean_root,
        source_evidence_report_path=args.source_evidence_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"verdict={report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_e2_natural_gap_geometry_sweep"]
