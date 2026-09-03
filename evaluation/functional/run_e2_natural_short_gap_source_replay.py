"""Run a fresh Source-only replay of the short-gap Linear candidate.

The exposed gap-geometry sweep is development evidence used only to compile a
small candidate template.  A fresh roster is then fixed from frozen
Support-A-discovery membership and public registry metadata before any selected
values, contexts, futures, or Judge outcomes are loaded.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
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
    _execute_program,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_natural_source_evidence import (
    SOURCE_DATASETS,
    _read_subsplit,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


SCHEMA_VERSION = "e2-natural-short-gap-source-replay/1"
DISCOVERY_SUBSPLIT = "support_a_discovery"
CONTEXT_BOUNDS = (736, 928)
FUTURE_BOUNDS = (928, 976)
GAP_BOUNDS = (174, 180)
GAP_LENGTH = 6
PER_DATASET_QUOTA = 4
COMPILE_N_MIN = 8
FRESH_N_MIN = 8
MEAN_GAIN_MIN = 0.005
HARM_GAIN_THRESHOLD = -0.005
HARM_RATE_MAX = 0.25
DATASET_MEAN_GAIN_MIN = -0.005
MATERIAL_GAIN_MIN = 0.005


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
class FreshRosterItem:
    record: SeriesRecord
    assignment: SplitAssignment


@dataclass(frozen=True)
class FreshCase:
    roster_item: FreshRosterItem
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray


def _finite_loss(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric loss: {label}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite loss: {label}")
    return result


def _effect(rows: list[dict[str, Any]]) -> dict[str, object]:
    gains = [float(row["gain_over_identity"]) for row in rows]
    if not gains:
        return {
            "n": 0,
            "mean_gain": None,
            "harm_count": 0,
            "harm_rate": None,
            "material_gain_count": 0,
        }
    return {
        "n": len(gains),
        "mean_gain": statistics.fmean(gains),
        "harm_count": sum(gain < HARM_GAIN_THRESHOLD for gain in gains),
        "harm_rate": sum(gain < HARM_GAIN_THRESHOLD for gain in gains) / len(gains),
        "material_gain_count": sum(gain >= MATERIAL_GAIN_MIN for gain in gains),
    }


def compile_short_gap_candidate(
    geometry_report_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compile a candidate from already-exposed g=6 discovery evidence."""

    report = json.loads(geometry_report_path.read_text("utf-8"))
    if report.get("schema_version") != "e2-natural-gap-geometry-sweep/1":
        raise ValueError("unsupported gap-geometry report schema")
    if report.get("verdict") != "GEOMETRY_PREMISE_PRESENT":
        raise ValueError("gap-geometry premise is not present")
    aggregate = report.get("aggregate_by_gap")
    rows_by_gap = report.get("cases_by_gap")
    if not isinstance(aggregate, dict) or not isinstance(rows_by_gap, dict):
        raise ValueError("gap-geometry report lacks aggregates or cases")
    g6_aggregate = aggregate.get(str(GAP_LENGTH))
    raw_rows = rows_by_gap.get(str(GAP_LENGTH))
    if not isinstance(g6_aggregate, dict) or not isinstance(raw_rows, list):
        raise ValueError("gap-geometry report lacks the g=6 cell")
    if g6_aggregate.get("best_fixed_action") != "linear":
        raise ValueError("frozen g=6 best fixed action is not Linear")
    if len(raw_rows) != COMPILE_N_MIN:
        raise ValueError("g=6 development cell must contain exactly eight cases")

    rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, dict):
            raise ValueError("invalid g=6 development case")
        uid = raw.get("series_uid")
        dataset_id = raw.get("dataset_id")
        losses = raw.get("loss_by_action")
        if not isinstance(uid, str) or not uid or uid in seen_uids:
            raise ValueError("g=6 UIDs must be unique strings")
        if dataset_id not in SOURCE_DATASETS:
            raise ValueError(f"non-Source dataset in g=6 cell: {uid}")
        if raw.get("subsplit") != DISCOVERY_SUBSPLIT or raw.get("gap_length") != GAP_LENGTH:
            raise ValueError(f"g=6 case violates frozen discovery geometry: {uid}")
        if not isinstance(losses, dict):
            raise ValueError(f"g=6 case lacks action losses: {uid}")
        identity_loss = _finite_loss(losses.get("identity"), label=f"{uid}/identity")
        linear_loss = _finite_loss(losses.get("linear"), label=f"{uid}/linear")
        gain = identity_loss - linear_loss
        rows.append(
            {
                "series_uid": uid,
                "dataset_id": dataset_id,
                "identity_loss": identity_loss,
                "linear_loss": linear_loss,
                "gain_over_identity": gain,
                "harmed": gain < HARM_GAIN_THRESHOLD,
                "material_gain": gain >= MATERIAL_GAIN_MIN,
            }
        )
        seen_uids.add(uid)

    overall = _effect(rows)
    per_dataset = {
        dataset_id: _effect([row for row in rows if row["dataset_id"] == dataset_id])
        for dataset_id in SOURCE_DATASETS
    }
    gates = {
        "n": {
            "threshold": COMPILE_N_MIN,
            "value": overall["n"],
            "pass": int(overall["n"]) >= COMPILE_N_MIN,
        },
        "mean_gain": {
            "threshold": MEAN_GAIN_MIN,
            "value": overall["mean_gain"],
            "pass": float(overall["mean_gain"]) >= MEAN_GAIN_MIN,
        },
        "harm_rate": {
            "threshold": HARM_RATE_MAX,
            "definition": f"gain < {HARM_GAIN_THRESHOLD}",
            "value": overall["harm_rate"],
            "pass": float(overall["harm_rate"]) <= HARM_RATE_MAX,
        },
        "per_dataset_mean_gain": {
            "threshold": DATASET_MEAN_GAIN_MIN,
            "values": {
                dataset_id: effect["mean_gain"]
                for dataset_id, effect in per_dataset.items()
            },
            "pass": all(
                int(effect["n"]) > 0
                and float(effect["mean_gain"]) >= DATASET_MEAN_GAIN_MIN
                for effect in per_dataset.values()
            ),
        },
    }
    all_gates_pass = all(bool(gate["pass"]) for gate in gates.values())
    compilation = {
        "development_evidence_only": True,
        "source_report_schema": report["schema_version"],
        "source_report_verdict": report["verdict"],
        "g6_best_fixed_action": g6_aggregate["best_fixed_action"],
        "cases": rows,
        "overall": overall,
        "per_dataset": per_dataset,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
    }
    if not all_gates_pass:
        raise ValueError("exposed g=6 development evidence does not compile the candidate")

    template = {
        "status": "DEVELOPMENT_CANDIDATE_NOT_FORMALLY_SUPPORTED",
        "phenomenon": "contiguous_internal_missing",
        "context_length": 192,
        "gap_relative_to_context": list(GAP_BOUNDS),
        "gap_length": GAP_LENGTH,
        "gap_fraction": GAP_LENGTH / 192,
        "program": "linear",
        "incumbent": "identity",
    }
    return template, compilation


def _excluded_uids(
    evidence_report_path: Path,
    promotion_report_path: Path,
) -> tuple[set[str], dict[str, object]]:
    evidence = json.loads(evidence_report_path.read_text("utf-8"))
    promotion = json.loads(promotion_report_path.read_text("utf-8"))
    if evidence.get("schema_version") != "e2-natural-source-evidence/1":
        raise ValueError("unsupported natural Source evidence report schema")
    if promotion.get("schema_version") != "e2-natural-source-promotion/1":
        raise ValueError("unsupported natural Source promotion report schema")
    evidence_cases = evidence.get("cases")
    promotion_cases = promotion.get("affected_cases")
    if not isinstance(evidence_cases, list) or len(evidence_cases) != 16:
        raise ValueError("natural Source evidence report must contain sixteen cases")
    if not isinstance(promotion_cases, list) or len(promotion_cases) != 4:
        raise ValueError("natural Source promotion report must contain four affected cases")
    evidence_uids = {row.get("series_uid") for row in evidence_cases if isinstance(row, dict)}
    promotion_uids = {row.get("series_uid") for row in promotion_cases if isinstance(row, dict)}
    if len(evidence_uids) != 16 or not all(isinstance(uid, str) and uid for uid in evidence_uids):
        raise ValueError("natural Source evidence UIDs are invalid or duplicated")
    if len(promotion_uids) != 4 or not all(isinstance(uid, str) and uid for uid in promotion_uids):
        raise ValueError("natural Source promotion UIDs are invalid or duplicated")
    excluded = evidence_uids | promotion_uids
    return excluded, {
        "natural_source_evidence_uid_count": len(evidence_uids),
        "natural_source_promotion_uid_count": len(promotion_uids),
        "unique_excluded_uid_count": len(excluded),
    }


def select_fresh_short_gap_roster(
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    evidence_report_path: Path,
    promotion_report_path: Path,
) -> tuple[list[FreshRosterItem], dict[str, object]]:
    """Fix eight Source-discovery cases without loading any series values."""

    excluded, exclusion = _excluded_uids(evidence_report_path, promotion_report_path)
    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    assignments = {row.series_uid: row for row in manifest.assignments}
    discovery_uids = _read_subsplit(support_a_subsplit_path)[DISCOVERY_SUBSPLIT]

    candidates_by_dataset: dict[str, list[FreshRosterItem]] = {
        dataset_id: [] for dataset_id in SOURCE_DATASETS
    }
    for uid in discovery_uids:
        assignment = assignments.get(uid)
        if assignment is None:
            raise ValueError(f"discovery UID absent from split manifest: {uid}")
        if assignment.role is not SplitRole.SUPPORT_A:
            raise ValueError(f"discovery member is not Support-A: {uid}")
        if assignment.dataset_id not in SOURCE_DATASETS or uid in excluded:
            continue
        record = records.get(uid)
        if record is None:
            raise ValueError(f"discovery UID absent from registry: {uid}")
        if record.dataset_id != assignment.dataset_id:
            raise ValueError(f"registry/split dataset mismatch: {uid}")
        if record.regime_tag != assignment.regime_tag:
            raise ValueError(f"registry/split regime mismatch: {uid}")
        if record.admission_reasons != ():
            raise ValueError(f"ineligible Source discovery record: {uid}")
        if SplitRole.SUPPORT_A.value not in record.roles_allowed:
            raise ValueError(f"record disallows Support-A: {uid}")
        boundaries = assignment.chronological_boundaries
        if boundaries is None:
            raise ValueError(f"missing chronological boundaries: {uid}")
        if tuple(boundaries.get("train", ())) != (0, 928):
            raise ValueError(f"unexpected train boundary: {uid}")
        if tuple(boundaries.get("validation", ())) != FUTURE_BOUNDS:
            raise ValueError(f"unexpected validation boundary: {uid}")
        if tuple(boundaries.get("test", ())) != (976, 1024):
            raise ValueError(f"unexpected test boundary: {uid}")
        candidates_by_dataset[assignment.dataset_id].append(
            FreshRosterItem(record=record, assignment=assignment)
        )

    selected: list[FreshRosterItem] = []
    available_by_dataset: dict[str, int] = {}
    for dataset_id in SOURCE_DATASETS:
        candidates = sorted(
            candidates_by_dataset[dataset_id], key=lambda item: item.record.series_uid
        )
        available_by_dataset[dataset_id] = len(candidates)
        if len(candidates) < PER_DATASET_QUOTA:
            raise ValueError(f"fewer than four fresh discovery cases: {dataset_id}")
        selected.extend(candidates[:PER_DATASET_QUOTA])

    selected_uids = [item.record.series_uid for item in selected]
    if len(selected) != 8 or len(selected_uids) != len(set(selected_uids)):
        raise AssertionError("fresh Source roster must contain eight unique cases")
    return selected, {
        "fixed_before_value_context_future_and_judge_loading": True,
        "selection_rule": (
            "within each frozen Source dataset, take the first four eligible, "
            "non-exposed support_a_discovery members by series_uid ascending"
        ),
        "selection_features": [
            "frozen_support_a_discovery_membership",
            "frozen_registry_eligibility",
            "dataset_id",
            "series_uid",
            "prior_exposure_uid_exclusion",
        ],
        "regime_feature_value_future_or_outcome_used": False,
        "available_after_exclusion_by_dataset": available_by_dataset,
        "selected_by_dataset": dict(Counter(item.record.dataset_id for item in selected)),
        "selected_uids": selected_uids,
        "exclusion": exclusion,
    }


def _load_fresh_cases(
    roster: list[FreshRosterItem],
    clean_root: Path,
) -> list[FreshCase]:
    """Load and verify values only after the complete fresh roster is fixed."""

    values_by_uid = _load_values([item.record for item in roster], clean_root)
    cases: list[FreshCase] = []
    for item in roster:
        uid = item.record.series_uid
        values = values_by_uid[uid]
        clean_context = np.asarray(values[slice(*CONTEXT_BOUNDS)], dtype=np.float64).copy()
        clean_future = np.asarray(values[slice(*FUTURE_BOUNDS)], dtype=np.float64).copy()
        if clean_context.shape != (192,) or clean_future.shape != (48,):
            raise ValueError(f"insufficient fixed discovery window: {uid}")
        if not np.isfinite(clean_context).all() or not np.isfinite(clean_future).all():
            raise ValueError(f"natural missingness enters fixed discovery window: {uid}")
        corrupt_context = clean_context.copy()
        corrupt_context[slice(*GAP_BOUNDS)] = np.nan
        if int(np.isnan(corrupt_context).sum()) != GAP_LENGTH:
            raise AssertionError(f"short-gap injection failed: {uid}")
        cases.append(FreshCase(item, clean_context, corrupt_context, clean_future))
    return cases


def run_e2_natural_short_gap_source_replay(
    valuator: _Valuator,
    *,
    geometry_report_path: Path,
    evidence_report_path: Path,
    promotion_report_path: Path,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    candidate_template, compilation = compile_short_gap_candidate(geometry_report_path)
    roster, selection = select_fresh_short_gap_roster(
        registry_path=registry_path,
        split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
        evidence_report_path=evidence_report_path,
        promotion_report_path=promotion_report_path,
    )
    cases = _load_fresh_cases(roster, clean_root)

    rows: list[dict[str, Any]] = []
    judge_calls = 0
    for case in cases:
        uid = case.roster_item.record.series_uid
        losses: dict[str, float] = {}
        for action in ("identity", "linear"):
            prepared = _execute_program(
                action,
                case.corrupt_context,
                # Neither member of this fixed menu binds a seasonal period.
                observed_period=1,
            )
            receipt = valuator.evaluate(
                prepared,
                case.clean_future,
                scale_context=case.clean_context,
            )
            judge_calls += 1
            losses[action] = _finite_loss(receipt.loss_j, label=f"{uid}/{action}")
        gain = losses["identity"] - losses["linear"]
        rows.append(
            {
                "series_uid": uid,
                "dataset_id": case.roster_item.record.dataset_id,
                "split": SplitRole.SUPPORT_A.value,
                "subsplit": DISCOVERY_SUBSPLIT,
                "windows": {
                    "context": list(CONTEXT_BOUNDS),
                    "future": list(FUTURE_BOUNDS),
                    "gap_relative_to_context": list(GAP_BOUNDS),
                },
                "identity_loss": losses["identity"],
                "linear_loss": losses["linear"],
                "gain_over_identity": gain,
                "harmed": gain < HARM_GAIN_THRESHOLD,
                "material_gain": gain >= MATERIAL_GAIN_MIN,
            }
        )

    if len(rows) != 8 or len({row["series_uid"] for row in rows}) != 8:
        raise AssertionError("fresh replay must contain eight unique cases")
    if judge_calls != 16:
        raise AssertionError(f"expected exactly 16 new Judge calls, observed {judge_calls}")

    overall = _effect(rows)
    per_dataset = {
        dataset_id: _effect([row for row in rows if row["dataset_id"] == dataset_id])
        for dataset_id in SOURCE_DATASETS
    }
    gates = {
        "n": {
            "threshold": FRESH_N_MIN,
            "value": overall["n"],
            "pass": int(overall["n"]) >= FRESH_N_MIN,
        },
        "mean_gain": {
            "threshold": MEAN_GAIN_MIN,
            "value": overall["mean_gain"],
            "pass": float(overall["mean_gain"]) >= MEAN_GAIN_MIN,
        },
        "harm_rate": {
            "threshold": HARM_RATE_MAX,
            "definition": f"gain < {HARM_GAIN_THRESHOLD}",
            "value": overall["harm_rate"],
            "pass": float(overall["harm_rate"]) <= HARM_RATE_MAX,
        },
        "per_dataset_mean_gain": {
            "threshold": DATASET_MEAN_GAIN_MIN,
            "values": {
                dataset_id: effect["mean_gain"]
                for dataset_id, effect in per_dataset.items()
            },
            "pass": all(
                int(effect["n"]) > 0
                and float(effect["mean_gain"]) >= DATASET_MEAN_GAIN_MIN
                for effect in per_dataset.values()
            ),
        },
        "per_dataset_material_gain_case_count": {
            "threshold": 1,
            "definition": f"gain >= {MATERIAL_GAIN_MIN}",
            "values": {
                dataset_id: effect["material_gain_count"]
                for dataset_id, effect in per_dataset.items()
            },
            "pass": all(
                int(effect["material_gain_count"]) >= 1
                for effect in per_dataset.values()
            ),
        },
    }
    all_gates_pass = all(bool(gate["pass"]) for gate in gates.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "fresh_source_short_gap_linear_replay_pilot",
        "candidate_capability_template": candidate_template,
        "development_compilation_evidence": compilation,
        "fresh_roster_selection": selection,
        "information_wall": {
            "roster_fixed_before_value_context_future_and_judge_loading": True,
            "fresh_uids": selection["selected_uids"],
            "source_discovery_only": True,
            "support_b_read": False,
            "target_or_query_read": False,
            "target_query_opened": False,
        },
        "configuration": {
            "datasets": list(SOURCE_DATASETS),
            "split": SplitRole.SUPPORT_A.value,
            "subsplit": DISCOVERY_SUBSPLIT,
            "context_bounds": list(CONTEXT_BOUNDS),
            "future_bounds": list(FUTURE_BOUNDS),
            "gap_relative_to_context": list(GAP_BOUNDS),
            "programs": ["identity", "linear"],
            "valuator": "FrozenChronosValuator",
            "agent_enabled": False,
            "memory_enabled": False,
            "adaptation_enabled": False,
        },
        "judge_call_count": judge_calls,
        "cases": rows,
        "intervention_effect": {
            "overall": overall,
            "per_dataset": per_dataset,
        },
        "fresh_gates": gates,
        "all_fresh_gates_pass": all_gates_pass,
        "verdict": (
            "FRESH_SOURCE_REPLAY_PASSED"
            if all_gates_pass
            else "FRESH_SOURCE_REPLAY_FAILED"
        ),
        "pilot_promotion_status": (
            "SOURCE_PILOT_PROMOTED" if all_gates_pass else "PROVISIONAL"
        ),
        "target_plan_eligible": all_gates_pass,
        "formal_cross_dataset_supported": False,
        "target_query_opened": False,
        "claim_limit": (
            "At most a fresh Source pilot replay of a short-gap Linear candidate; not "
            "evidence of Target performance, Memory benefit, adaptation, formal "
            "cross-dataset support, or transfer."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-report",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_gap_geometry_sweep_report.json",
    )
    parser.add_argument(
        "--evidence-report",
        type=Path,
        default=project_root / "artifacts/functional/e2/natural_source_evidence_report.json",
    )
    parser.add_argument(
        "--promotion-report",
        type=Path,
        default=project_root / "artifacts/functional/e2/natural_source_promotion_report.json",
    )
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
        / "artifacts/functional/e2/natural_short_gap_source_replay_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_short_gap_source_replay(
        FrozenChronosValuator(),
        geometry_report_path=args.geometry_report,
        evidence_report_path=args.evidence_report,
        promotion_report_path=args.promotion_report,
        registry_path=args.registry,
        split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"verdict={report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compile_short_gap_candidate",
    "run_e2_natural_short_gap_source_replay",
    "select_fresh_short_gap_roster",
]
