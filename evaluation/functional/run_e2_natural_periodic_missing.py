"""Run the E2 natural UCI Support-B periodic-missing headroom stage.

The roster is fixed from public frozen metadata before any value is loaded or any
Judge outcome exists.  The menu oracle is a diagnostic ceiling, not a deployable
selector.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    PROGRAM_IDS,
    _execute_program,
    _global_features,
    _local_features,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


DATASET_ID = "uci_electricity_load_diagrams"
CONTEXT_BOUNDS = (736, 928)
FUTURE_BOUNDS = (928, 976)
GAP_BOUNDS = (156, 180)
DESIRED_STRATA = {"seasonal_high": 6, "structured_mixed": 2}
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
class NaturalCase:
    record: SeriesRecord
    assignment: SplitAssignment
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray


def select_support_b_roster(
    registry_path: Path,
    split_path: Path,
) -> tuple[list[tuple[SeriesRecord, SplitAssignment]], dict[str, object]]:
    """Fix the eight-case roster using frozen visible metadata only."""

    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    candidates: list[tuple[SeriesRecord, SplitAssignment]] = []
    for assignment in manifest.assignments:
        if assignment.dataset_id != DATASET_ID or assignment.role is not SplitRole.SUPPORT_B:
            continue
        record = records.get(assignment.series_uid)
        if record is None:
            raise ValueError(f"split UID absent from registry: {assignment.series_uid}")
        if record.dataset_id != assignment.dataset_id:
            raise ValueError(f"registry/split dataset mismatch: {assignment.series_uid}")
        if record.regime_tag != assignment.regime_tag:
            raise ValueError(f"registry/split regime mismatch: {assignment.series_uid}")
        if record.admission_reasons != ():
            raise ValueError(f"ineligible Support-B record: {assignment.series_uid}")
        if SplitRole.SUPPORT_B.value not in record.roles_allowed:
            raise ValueError(f"record disallows Support-B: {assignment.series_uid}")
        candidates.append((record, assignment))

    candidates.sort(key=lambda pair: pair[0].series_uid)
    selected: list[tuple[SeriesRecord, SplitAssignment]] = []
    selected_uids: set[str] = set()
    deficits: dict[str, int] = {}
    for stratum, desired_count in DESIRED_STRATA.items():
        matches = [
            pair
            for pair in candidates
            if pair[1].regime_tag == stratum and pair[0].series_uid not in selected_uids
        ]
        chosen = matches[:desired_count]
        selected.extend(chosen)
        selected_uids.update(pair[0].series_uid for pair in chosen)
        if len(chosen) < desired_count:
            deficits[stratum] = desired_count - len(chosen)

    fallback_needed = sum(DESIRED_STRATA.values()) - len(selected)
    if fallback_needed:
        fallback = [
            pair for pair in candidates if pair[0].series_uid not in selected_uids
        ][:fallback_needed]
        selected.extend(fallback)
        selected_uids.update(pair[0].series_uid for pair in fallback)
    if len(selected) != sum(DESIRED_STRATA.values()):
        raise ValueError("fewer than eight eligible UCI Support-B records")

    actual_counts = Counter(pair[1].regime_tag for pair in selected)
    selection = {
        "method": "visible_metadata_then_series_uid_ascending",
        "desired_strata": dict(DESIRED_STRATA),
        "actual_strata": dict(sorted(actual_counts.items())),
        "fallback_used": bool(deficits),
        "fallback_deficits": deficits,
        "candidate_count": len(candidates),
    }
    return selected, selection


def load_natural_cases(
    roster: list[tuple[SeriesRecord, SplitAssignment]],
    clean_root: Path,
) -> list[NaturalCase]:
    """Load and verify only the already-fixed Support-B roster."""

    values_by_uid = _load_values([record for record, _ in roster], clean_root)
    cases: list[NaturalCase] = []
    for record, assignment in roster:
        boundaries = assignment.chronological_boundaries
        if boundaries is None:
            raise ValueError(f"missing chronological boundaries: {record.series_uid}")
        if tuple(boundaries.get("train", ())) != (0, CONTEXT_BOUNDS[1]):
            raise ValueError(f"unexpected train boundary: {record.series_uid}")
        if tuple(boundaries.get("validation", ())) != FUTURE_BOUNDS:
            raise ValueError(f"unexpected validation boundary: {record.series_uid}")

        values = values_by_uid[record.series_uid]
        clean_context = values[slice(*CONTEXT_BOUNDS)].copy()
        clean_future = values[slice(*FUTURE_BOUNDS)].copy()
        if clean_context.shape != (192,) or clean_future.shape != (48,):
            raise ValueError(f"insufficient fixed window: {record.series_uid}")
        if not np.isfinite(clean_context).all() or not np.isfinite(clean_future).all():
            raise ValueError(f"natural missingness enters the fixed clean window: {record.series_uid}")
        corrupt_context = clean_context.copy()
        corrupt_context[slice(*GAP_BOUNDS)] = np.nan
        if int(np.isnan(corrupt_context).sum()) != 24:
            raise AssertionError("fixed corruption must inject exactly 24 NaNs")
        cases.append(
            NaturalCase(
                record=record,
                assignment=assignment,
                clean_context=clean_context,
                corrupt_context=corrupt_context,
                clean_future=clean_future,
            )
        )
    return cases


def run_e2_natural_periodic_missing(
    valuator: _Valuator,
    *,
    registry_path: Path,
    split_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    roster, selection = select_support_b_roster(registry_path, split_path)
    cases = load_natural_cases(roster, clean_root)

    rows: list[dict[str, object]] = []
    judge_calls = 0
    for case in cases:
        global_features, observed_period = _global_features(case.corrupt_context)
        local_features = _local_features(case.corrupt_context, observed_period)
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
                raise ValueError(f"non-finite Judge loss: {case.record.series_uid}/{program_id}")
            losses[program_id] = loss

        ranked = sorted((loss, action) for action, loss in losses.items())
        winner_loss, winner = ranked[0]
        rows.append(
            {
                "series_uid": case.record.series_uid,
                "dataset_id": DATASET_ID,
                "split": SplitRole.SUPPORT_B.value,
                "regime_tag": case.assignment.regime_tag,
                "windows": {
                    "context": list(CONTEXT_BOUNDS),
                    "future": list(FUTURE_BOUNDS),
                    "gap_relative_to_context": list(GAP_BOUNDS),
                },
                "public_features": {
                    "observed_period": observed_period,
                    "missing_fraction": global_features["missing_fraction"],
                    "phase_correlation": local_features["phase_correlation"],
                    "local_period_consistency": local_features[
                        "local_period_consistency"
                    ],
                },
                "loss_by_action": losses,
                "winner": winner,
                "winner_margin": float(ranked[1][0] - winner_loss),
                "menu_oracle_loss": winner_loss,
            }
        )

    expected_calls = len(cases) * len(PROGRAM_IDS)
    if expected_calls != 24 or judge_calls != expected_calls:
        raise AssertionError(f"expected exactly 24 Judge calls, observed {judge_calls}")

    fixed_mean_losses = {
        action: statistics.fmean(
            float(row["loss_by_action"][action])  # type: ignore[index]
            for row in rows
        )
        for action in PROGRAM_IDS
    }
    best_fixed_action = min(
        PROGRAM_IDS, key=lambda action: (fixed_mean_losses[action], action)
    )
    menu_oracle_mean = statistics.fmean(float(row["menu_oracle_loss"]) for row in rows)
    routing_headroom = fixed_mean_losses[best_fixed_action] - menu_oracle_mean
    qualifying_wins = {
        action: sum(
            row["winner"] == action
            and float(row["winner_margin"]) >= WINNER_MARGIN_MIN
            for row in rows
        )
        for action in PROGRAM_IDS
    }
    heterogeneous_actions = [
        action for action in PROGRAM_IDS if qualifying_wins[action] >= 1
    ]
    headroom_pass = routing_headroom >= HEADROOM_MIN
    heterogeneity_pass = len(heterogeneous_actions) >= DISTINCT_ACTIONS_MIN

    return {
        "schema_version": "e2-natural-periodic-missing-headroom/1",
        "scientific_role": "natural_uci_support_positive_control",
        "configuration": {
            "dataset_id": DATASET_ID,
            "split": SplitRole.SUPPORT_B.value,
            "programs": list(PROGRAM_IDS),
            "context": list(CONTEXT_BOUNDS),
            "future": list(FUTURE_BOUNDS),
            "gap_relative_to_context": list(GAP_BOUNDS),
            "agent_enabled": False,
            "memory_enabled": False,
            "adaptation_enabled": False,
        },
        "roster_selection_before_judge": selection,
        "judge_call_count": judge_calls,
        "cases": rows,
        "aggregate": {
            "fixed_mean_loss_by_action": fixed_mean_losses,
            "best_fixed_action": best_fixed_action,
            "best_fixed_mean_loss": fixed_mean_losses[best_fixed_action],
            "menu_oracle_mean_loss": menu_oracle_mean,
            "routing_headroom": routing_headroom,
            "identity_gain_to_menu_oracle": fixed_mean_losses["identity"]
            - menu_oracle_mean,
            "qualifying_wins_by_action": qualifying_wins,
            "heterogeneous_actions": heterogeneous_actions,
        },
        "gates": {
            "routing_headroom": {
                "threshold": HEADROOM_MIN,
                "value": routing_headroom,
                "pass": headroom_pass,
            },
            "action_heterogeneity": {
                "winner_margin_min": WINNER_MARGIN_MIN,
                "distinct_actions_min": DISTINCT_ACTIONS_MIN,
                "distinct_actions": len(heterogeneous_actions),
                "pass": heterogeneity_pass,
            },
        },
        "all_gates_pass": headroom_pass and heterogeneity_pass,
        "claim_limit": (
            "Natural UCI Support-B positive control only; not evidence of transfer, "
            "Memory, adaptation, promotion, or query performance."
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
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_periodic_missing_headroom_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_periodic_missing(
        FrozenChronosValuator(),
        registry_path=args.registry,
        split_path=args.split,
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
    "load_natural_cases",
    "run_e2_natural_periodic_missing",
    "select_support_b_roster",
]
