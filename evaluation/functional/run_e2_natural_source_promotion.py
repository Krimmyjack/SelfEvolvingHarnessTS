"""Compile and pilot a period-scoped natural-source promotion policy.

The prior sixteen examples are consumed development evidence.  They compile a
candidate policy, but their former Support-A subsplit is provenance only.  Fresh
roster selection uses visible context and frozen metadata; selected futures are
materialized only after that roster is fixed.
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
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _slot
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
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_natural_source_evidence import (
    SOURCE_DATASETS,
    _read_subsplit,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


CONTEXT_BOUNDS = (784, 976)
FUTURE_BOUNDS = (976, 1024)
GAP_BOUNDS = (156, 180)
CANDIDATE_ACTIONS = ("linear", "seasonal")
AFFECTED_QUOTAS = {"monash:traffic_hourly": 3, "metr_la": 1}
COMPILE_N_MIN = 4
MEAN_GAIN_MIN = 0.005
HARM_GAIN_THRESHOLD = -0.005
HARM_RATE_MAX = 0.25
METR_SINGLETON_GAIN_MIN = -0.005


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
class CompiledRule:
    period_bucket: str
    action: str


@dataclass(frozen=True)
class FreshContextCase:
    record: SeriesRecord
    assignment: SplitAssignment
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    observed_period: int
    period_bucket: str
    policy_action: str


def period_bucket(observed_period: int) -> str:
    if observed_period < 20:
        return "short"
    if observed_period < 25:
        return "daily"
    return "long"


def compile_period_policy(
    evidence_report_path: Path,
) -> tuple[list[CompiledRule], set[str], list[dict[str, Any]], dict[str, object]]:
    """Compile rules from the fully consumed sixteen-case development report."""

    report = json.loads(evidence_report_path.read_text("utf-8"))
    if report.get("schema_version") != "e2-natural-source-evidence/1":
        raise ValueError("unsupported source-evidence report schema")
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 16:
        raise ValueError("source-evidence report must contain sixteen cases")
    cases: list[dict[str, Any]] = []
    consumed_uids: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ValueError("source-evidence case must be an object")
        uid = raw.get("series_uid")
        dataset_id = raw.get("dataset_id")
        if not isinstance(uid, str) or not uid or uid in consumed_uids:
            raise ValueError("source-evidence UIDs must be unique strings")
        if dataset_id not in SOURCE_DATASETS:
            raise ValueError("source-evidence report contains a non-source dataset")
        if raw.get("split") != SplitRole.SUPPORT_A.value:
            raise ValueError("source-evidence case is not Support-A")
        observed_period = raw.get("observed_period")
        losses = raw.get("loss_by_action")
        if not isinstance(observed_period, int) or not isinstance(losses, dict):
            raise ValueError("source-evidence case lacks period or losses")
        for action in ("identity",) + CANDIDATE_ACTIONS:
            loss = losses.get(action)
            if not isinstance(loss, (int, float)) or not np.isfinite(float(loss)):
                raise ValueError(f"invalid source-evidence loss: {uid}/{action}")
        consumed_uids.add(uid)
        cases.append(raw)

    bucket_reports: dict[str, dict[str, object]] = {}
    compiled_rules: list[CompiledRule] = []
    for bucket in ("short", "daily", "long"):
        cohort = [
            row for row in cases if period_bucket(int(row["observed_period"])) == bucket
        ]
        action_stats: dict[str, dict[str, object]] = {}
        for action in CANDIDATE_ACTIONS:
            gains = [
                float(row["loss_by_action"]["identity"])
                - float(row["loss_by_action"][action])
                for row in cohort
            ]
            action_stats[action] = {
                "n": len(gains),
                "mean_gain_over_identity": statistics.fmean(gains) if gains else None,
                "harm_rate": (
                    sum(gain < HARM_GAIN_THRESHOLD for gain in gains) / len(gains)
                    if gains
                    else None
                ),
            }
        candidate_action = sorted(
            CANDIDATE_ACTIONS,
            key=lambda action: (
                -(
                    float(action_stats[action]["mean_gain_over_identity"])
                    if action_stats[action]["mean_gain_over_identity"] is not None
                    else float("-inf")
                ),
                action,
            ),
        )[0]
        candidate_stats = action_stats[candidate_action]
        compiled = (
            int(candidate_stats["n"]) >= COMPILE_N_MIN
            and float(candidate_stats["mean_gain_over_identity"]) >= MEAN_GAIN_MIN
            and float(candidate_stats["harm_rate"]) <= HARM_RATE_MAX
        )
        bucket_reports[bucket] = {
            "candidate_action": candidate_action,
            "candidate_compiled": compiled,
            "action_stats": action_stats,
        }
        if compiled:
            compiled_rules.append(CompiledRule(bucket, candidate_action))

    compilation = {
        "source_report_schema": report["schema_version"],
        "consumed_development_evidence_count": len(cases),
        "all_source_examples_consumed": True,
        "former_subsplit_semantics": "provenance_only_not_a_sealed_gate",
        "bucket_boundaries": {
            "short": "observed_period < 20",
            "daily": "20 <= observed_period < 25",
            "long": "observed_period >= 25",
        },
        "thresholds": {
            "n_min": COMPILE_N_MIN,
            "mean_gain_min": MEAN_GAIN_MIN,
            "harm_definition": f"gain < {HARM_GAIN_THRESHOLD}",
            "harm_rate_max": HARM_RATE_MAX,
        },
        "buckets": bucket_reports,
        "compiled_rules": [
            {"period_bucket": rule.period_bucket, "action": rule.action}
            for rule in compiled_rules
        ],
    }
    return compiled_rules, consumed_uids, cases, compilation


def _load_visible_context(
    record: SeriesRecord,
    clean_root: Path,
) -> np.ndarray:
    """Memory-map and copy only the visible context, never the future pages."""

    slot = _slot(record, clean_root)
    values = np.load(slot / "values.npy", allow_pickle=False, mmap_mode="r")
    if values.ndim != 1 or values.shape != (record.length,):
        raise ValueError(f"visible series shape disagrees with registry: {record.series_uid}")
    context = np.asarray(values[slice(*CONTEXT_BOUNDS)], dtype=np.float64).copy()
    if context.shape != (192,) or not np.isfinite(context).all():
        raise ValueError(f"invalid visible context: {record.series_uid}")
    return context


def _load_verified_selected_future(
    record: SeriesRecord,
    clean_root: Path,
) -> np.ndarray:
    """Materialize one future only after the affected roster has been fixed."""

    slot = _slot(record, clean_root)
    values = np.load(slot / "values.npy", allow_pickle=False)
    timestamps = (
        None
        if record.timestamps_sha is None
        else np.load(slot / "timestamps.npy", allow_pickle=False)
    )
    record.verify_values(values, timestamps=timestamps)
    future = np.asarray(values[slice(*FUTURE_BOUNDS)], dtype=np.float64).copy()
    if future.shape != (48,) or not np.isfinite(future).all():
        raise ValueError(f"invalid selected future: {record.series_uid}")
    return future


def select_fresh_context_roster(
    compiled_rules: list[CompiledRule],
    consumed_uids: set[str],
    *,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
) -> tuple[list[FreshContextCase], dict[str, object]]:
    """Scan fresh validation contexts and fix four affected cases without futures."""

    action_by_bucket = {rule.period_bucket: rule.action for rule in compiled_rules}
    if not action_by_bucket:
        raise ValueError("no compiled rule has an affected scope")
    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    validation_uids = _read_subsplit(support_a_subsplit_path)[
        "support_a_validation"
    ]

    candidates: list[FreshContextCase] = []
    for assignment in manifest.assignments:
        uid = assignment.series_uid
        if (
            uid not in validation_uids
            or uid in consumed_uids
            or assignment.dataset_id not in SOURCE_DATASETS
        ):
            continue
        if assignment.role is not SplitRole.SUPPORT_A:
            raise ValueError(f"fresh validation member is not Support-A: {uid}")
        record = records.get(uid)
        if record is None:
            raise ValueError(f"fresh split UID absent from registry: {uid}")
        if record.dataset_id != assignment.dataset_id:
            raise ValueError(f"registry/split dataset mismatch: {uid}")
        if record.regime_tag != assignment.regime_tag:
            raise ValueError(f"registry/split regime mismatch: {uid}")
        if record.admission_reasons != ():
            raise ValueError(f"ineligible fresh Support-A record: {uid}")
        boundaries = assignment.chronological_boundaries
        if boundaries is None or tuple(boundaries.get("test", ())) != FUTURE_BOUNDS:
            raise ValueError(f"unexpected fresh test boundary: {uid}")

        clean_context = _load_visible_context(record, clean_root)
        corrupt_context = clean_context.copy()
        corrupt_context[slice(*GAP_BOUNDS)] = np.nan
        _, observed_period = _global_features(corrupt_context)
        if int(np.isnan(corrupt_context).sum()) != 24:
            raise AssertionError("context scan must contain the fixed 24-value gap")
        bucket = period_bucket(observed_period)
        candidates.append(
            FreshContextCase(
                record=record,
                assignment=assignment,
                clean_context=clean_context,
                corrupt_context=corrupt_context,
                observed_period=observed_period,
                period_bucket=bucket,
                policy_action=action_by_bucket.get(bucket, "identity"),
            )
        )

    candidates.sort(
        key=lambda case: (SOURCE_DATASETS.index(case.record.dataset_id), case.record.series_uid)
    )
    affected_by_dataset = {
        dataset_id: [
            case
            for case in candidates
            if case.record.dataset_id == dataset_id and case.policy_action != "identity"
        ]
        for dataset_id in SOURCE_DATASETS
    }
    outside_by_dataset = {
        dataset_id: [
            case
            for case in candidates
            if case.record.dataset_id == dataset_id and case.policy_action == "identity"
        ]
        for dataset_id in SOURCE_DATASETS
    }

    selected: list[FreshContextCase] = []
    for dataset_id in SOURCE_DATASETS:
        selected.extend(affected_by_dataset[dataset_id][: AFFECTED_QUOTAS[dataset_id]])
    fallback_needed = sum(AFFECTED_QUOTAS.values()) - len(selected)
    fallback: list[FreshContextCase] = []
    if fallback_needed:
        selected_uids = {case.record.series_uid for case in selected}
        fallback_pool = [
            case
            for case in candidates
            if case.policy_action != "identity"
            and case.record.series_uid not in selected_uids
        ]
        fallback = fallback_pool[:fallback_needed]
        selected.extend(fallback)
    if len(selected) != sum(AFFECTED_QUOTAS.values()):
        raise ValueError("fewer than four fresh affected contexts are available")

    selection = {
        "fixed_before_selected_future_materialization_and_judge": True,
        "selection_sources": [
            "frozen_support_a_validation_membership",
            "frozen_registry_metadata",
            "series_uid",
            "visible_corrupt_context_observed_period",
        ],
        "future_or_judge_used_for_selection": False,
        "consumed_uid_exclusion_count": len(consumed_uids),
        "fresh_context_scan_count": len(candidates),
        "affected_available_by_dataset": {
            dataset_id: len(affected_by_dataset[dataset_id])
            for dataset_id in SOURCE_DATASETS
        },
        "fresh_out_of_scope_available_by_dataset": {
            dataset_id: len(outside_by_dataset[dataset_id])
            for dataset_id in SOURCE_DATASETS
        },
        "desired_affected_by_dataset": dict(AFFECTED_QUOTAS),
        "selected_affected_by_dataset": dict(
            Counter(case.record.dataset_id for case in selected)
        ),
        "fallback_used": bool(fallback),
        "fallback": [
            {"series_uid": case.record.series_uid, "dataset_id": case.record.dataset_id}
            for case in fallback
        ],
    }
    return selected, selection


def _unchanged_audit(
    consumed_cases: list[dict[str, Any]],
    compiled_rules: list[CompiledRule],
) -> dict[str, object]:
    action_by_bucket = {rule.period_bucket: rule.action for rule in compiled_rules}
    rows: list[dict[str, object]] = []
    for case in consumed_cases:
        if case.get("subsplit") != "support_a_validation":
            continue
        observed_period = int(case["observed_period"])
        bucket = period_bucket(observed_period)
        policy_action = action_by_bucket.get(bucket, "identity")
        if policy_action != "identity":
            continue
        rows.append(
            {
                "series_uid": case["series_uid"],
                "dataset_id": case["dataset_id"],
                "former_subsplit_provenance": case["subsplit"],
                "observed_period": observed_period,
                "period_bucket": bucket,
                "incumbent_action": "identity",
                "policy_action": "identity",
                "unchanged": True,
                "judge_calls": 0,
            }
        )
    rows.sort(key=lambda row: (SOURCE_DATASETS.index(str(row["dataset_id"])), str(row["series_uid"])))
    return {
        "contract": "outside compiled scope the complete policy remains identity",
        "former_validation_is_consumed_provenance_not_a_sealed_gate": True,
        "case_count": len(rows),
        "coverage_by_dataset": dict(Counter(str(row["dataset_id"]) for row in rows)),
        "judge_call_count": 0,
        "cases": rows,
    }


def _effect(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        return {"n": 0, "mean_gain": None, "harm_rate": None}
    gains = [float(row["gain_over_identity"]) for row in rows]
    return {
        "n": len(rows),
        "mean_gain": statistics.fmean(gains),
        "harm_rate": sum(gain < HARM_GAIN_THRESHOLD for gain in gains) / len(gains),
    }


def run_e2_natural_source_promotion(
    valuator: _Valuator,
    *,
    evidence_report_path: Path,
    registry_path: Path,
    split_path: Path,
    support_a_subsplit_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    compiled_rules, consumed_uids, consumed_cases, compilation = compile_period_policy(
        evidence_report_path
    )
    if compiled_rules != [CompiledRule("long", "seasonal")]:
        raise AssertionError(
            "frozen source compiler must produce exactly long -> seasonal"
        )
    roster, selection = select_fresh_context_roster(
        compiled_rules,
        consumed_uids,
        registry_path=registry_path,
        split_path=split_path,
        support_a_subsplit_path=support_a_subsplit_path,
        clean_root=clean_root,
    )
    unchanged = _unchanged_audit(consumed_cases, compiled_rules)

    rows: list[dict[str, Any]] = []
    judge_calls = 0
    for case in roster:
        clean_future = _load_verified_selected_future(case.record, clean_root)
        identity = _execute_program(
            "identity", case.corrupt_context, observed_period=case.observed_period
        )
        candidate = _execute_program(
            case.policy_action,
            case.corrupt_context,
            observed_period=case.observed_period,
        )
        identity_receipt = valuator.evaluate(
            identity, clean_future, scale_context=case.clean_context
        )
        judge_calls += 1
        candidate_receipt = valuator.evaluate(
            candidate, clean_future, scale_context=case.clean_context
        )
        judge_calls += 1
        identity_loss = float(identity_receipt.loss_j)
        candidate_loss = float(candidate_receipt.loss_j)
        if not np.isfinite(identity_loss) or not np.isfinite(candidate_loss):
            raise ValueError(f"non-finite promotion loss: {case.record.series_uid}")
        gain = identity_loss - candidate_loss
        rows.append(
            {
                "series_uid": case.record.series_uid,
                "dataset_id": case.record.dataset_id,
                "split": SplitRole.SUPPORT_A.value,
                "subsplit": "support_a_validation",
                "windows": {
                    "context": list(CONTEXT_BOUNDS),
                    "future": list(FUTURE_BOUNDS),
                    "gap_relative_to_context": list(GAP_BOUNDS),
                },
                "observed_period": case.observed_period,
                "period_bucket": case.period_bucket,
                "candidate_action": case.policy_action,
                "identity_loss": identity_loss,
                "candidate_loss": candidate_loss,
                "gain_over_identity": gain,
                "harmed": gain < HARM_GAIN_THRESHOLD,
            }
        )

    expected_calls = 2 * len(roster)
    if len(roster) != 4 or judge_calls != expected_calls or judge_calls != 8:
        raise AssertionError(f"expected exactly 8 Judge calls, observed {judge_calls}")

    affected_effect = _effect(rows)
    per_dataset = {
        dataset_id: _effect([row for row in rows if row["dataset_id"] == dataset_id])
        for dataset_id in SOURCE_DATASETS
    }
    metr_rows = [row for row in rows if row["dataset_id"] == "metr_la"]
    if len(metr_rows) != 1:
        raise AssertionError("promotion roster must contain one METR case")
    metr_singleton_gain = float(metr_rows[0]["gain_over_identity"])

    affected_n_pass = int(affected_effect["n"]) >= COMPILE_N_MIN
    mean_gain_pass = float(affected_effect["mean_gain"]) >= MEAN_GAIN_MIN
    harm_rate_pass = float(affected_effect["harm_rate"]) <= HARM_RATE_MAX
    metr_safety_pass = metr_singleton_gain >= METR_SINGLETON_GAIN_MIN
    all_gates_pass = (
        affected_n_pass and mean_gain_pass and harm_rate_pass and metr_safety_pass
    )
    baseline_policy = {bucket: "identity" for bucket in ("short", "daily", "long")}
    candidate_policy = dict(baseline_policy)
    candidate_policy.update(
        {rule.period_bucket: rule.action for rule in compiled_rules}
    )

    return {
        "schema_version": "e2-natural-source-promotion/1",
        "scientific_role": "source_capability_promotion_pilot",
        "policy_status": "PROMOTED" if all_gates_pass else "PROVISIONAL",
        "baseline_policy": baseline_policy,
        "candidate_policy": candidate_policy,
        "policy_compilation": compilation,
        "information_wall": {
            "source_evidence_examples_are_consumed_development_evidence": True,
            "former_subsplit_is_provenance_only": True,
            "fresh_selection_reads": ["frozen_metadata", "visible_corrupt_context"],
            "fresh_selection_reads_future_or_judge_outcome": False,
            "selected_future_materialized_after_roster_freeze": True,
        },
        "fresh_roster_selection": selection,
        "scope": {
            "compiled_rules": compilation["compiled_rules"],
            "default_action_outside_scope": "identity",
        },
        "judge_call_count": judge_calls,
        "affected_cases": rows,
        "intervention_effect": {
            "affected_cohort": affected_effect,
            "per_dataset": per_dataset,
            "affected_dataset_count": sum(
                int(effect["n"]) > 0 for effect in per_dataset.values()
            ),
        },
        "outside_scope_unchanged": unchanged,
        "promotion_gates": {
            "affected_n": {
                "threshold": COMPILE_N_MIN,
                "value": affected_effect["n"],
                "pass": affected_n_pass,
            },
            "affected_mean_gain": {
                "threshold": MEAN_GAIN_MIN,
                "value": affected_effect["mean_gain"],
                "pass": mean_gain_pass,
            },
            "affected_harm_rate": {
                "threshold": HARM_RATE_MAX,
                "value": affected_effect["harm_rate"],
                "pass": harm_rate_pass,
            },
            "metr_singleton_gain": {
                "threshold": METR_SINGLETON_GAIN_MIN,
                "value": metr_singleton_gain,
                "pass": metr_safety_pass,
            },
        },
        "all_promotion_gates_pass": all_gates_pass,
        "claim_limit": (
            "Source capability promotion pilot only; not evidence of target transfer, "
            "Memory, adaptation, or target performance."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-report",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_source_evidence_report.json",
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
        / "artifacts/functional/e2/natural_source_promotion_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_source_promotion(
        FrozenChronosValuator(),
        evidence_report_path=args.evidence_report,
        registry_path=args.registry,
        split_path=args.split,
        support_a_subsplit_path=args.support_a_subsplit,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"policy_status={report['policy_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compile_period_policy",
    "period_bucket",
    "run_e2_natural_source_promotion",
    "select_fresh_context_roster",
]
