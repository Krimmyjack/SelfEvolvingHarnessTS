"""Development-only Scope×Operator diagnostic on four exposed Source cases.

The roster and Identity/current single-cycle losses come from the existing
failed natural Source promotion report.  Exactly one new multi-cycle arm is
judged per case.  No Target or Query surface is accessible from this runner.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    _global_features,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_natural_source_promotion import (
    CONTEXT_BOUNDS,
    FUTURE_BOUNDS,
    GAP_BOUNDS,
    _load_verified_selected_future,
    _load_visible_context,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline


EXPECTED_DATASET_PERIODS = (
    ("monash:traffic_hourly", 25),
    ("monash:traffic_hourly", 25),
    ("monash:traffic_hourly", 48),
    ("metr_la", 47),
)
HARM_THRESHOLD = -0.005
MATERIAL_GAIN_THRESHOLD = 0.005
RATIO_SCOPE_MAX = 0.5


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


def _read_roster(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(path.read_text("utf-8"))
    if not isinstance(report, dict):
        raise TypeError("promotion report must be a JSON object")
    if report.get("schema_version") != "e2-natural-source-promotion/1":
        raise ValueError("unsupported promotion report schema")
    if report.get("policy_status") != "PROVISIONAL":
        raise ValueError("diagnostic requires the failed PROVISIONAL promotion")
    if report.get("all_promotion_gates_pass") is not False:
        raise ValueError("diagnostic requires explicitly failed promotion gates")
    raw_cases = report.get("affected_cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 4:
        raise ValueError("diagnostic requires exactly four affected cases")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise TypeError("affected case must be an object")
        row = dict(raw)
        uid = row.get("series_uid")
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError("affected case UIDs must be unique strings")
        if (
            row.get("split") != "support_a"
            or row.get("subsplit") != "support_a_validation"
            or row.get("candidate_action") != "seasonal"
        ):
            raise ValueError("diagnostic roster is not the exposed Source roster")
        for key in ("identity_loss", "candidate_loss"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"invalid cached loss: {uid}/{key}")
        seen.add(uid)
        rows.append(row)

    rows.sort(
        key=lambda row: (
            0 if row["dataset_id"] == "monash:traffic_hourly" else 1,
            int(row["observed_period"]),
            str(row["series_uid"]),
        )
    )
    actual = tuple((str(row["dataset_id"]), int(row["observed_period"])) for row in rows)
    if actual != EXPECTED_DATASET_PERIODS:
        raise ValueError(f"expected exposed roster {EXPECTED_DATASET_PERIODS}, observed {actual}")
    return report, rows


def _multi_cycle_context(corrupt_context: np.ndarray, period: int) -> np.ndarray:
    execution = run_pipeline(
        [
            (
                "period_median_complete",
                {"period": period, "cycles": 3, "min_donors": 2},
            )
        ],
        corrupt_context,
        source="e2_development_scope_operator_diagnostic",
    )
    if not execution.ok or execution.artifact is None:
        raise RuntimeError(f"multi-cycle operator failed: {execution.error}")
    artifact = np.asarray(execution.artifact, dtype=np.float64)
    if artifact.shape != corrupt_context.shape or not np.isfinite(artifact).all():
        raise RuntimeError("multi-cycle operator produced an invalid artifact")
    return artifact


def _policy_summary(
    case_rows: list[dict[str, Any]],
    *,
    policy_id: str,
    use_multi: bool,
    ratio_scope: bool,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for row in case_rows:
        activated = bool(row["ratio_scope_activated"]) if ratio_scope else True
        if not activated:
            loss = float(row["identity_loss"])
            action = "identity"
        elif use_multi:
            loss = float(row["multi_cycle_loss"])
            action = "period_median_complete"
        else:
            loss = float(row["current_single_cycle_loss"])
            action = "period_complete"
        gain = float(row["identity_loss"]) - loss
        cases.append(
            {
                "series_uid": row["series_uid"],
                "dataset_id": row["dataset_id"],
                "current_period": row["current_period"],
                "gap_to_period_ratio": row["gap_to_period_ratio"],
                "activated": activated,
                "action": action,
                "loss": loss,
                "gain_over_identity": gain,
                "harmed": gain < HARM_THRESHOLD,
                "material_gain": gain >= MATERIAL_GAIN_THRESHOLD,
            }
        )

    def _effect(selected: list[dict[str, object]]) -> dict[str, object]:
        gains = [float(row["gain_over_identity"]) for row in selected]
        return {
            "n": len(selected),
            "mean_gain_over_identity": statistics.fmean(gains) if gains else None,
            "harm_count": sum(bool(row["harmed"]) for row in selected),
            "harm_rate": (
                sum(bool(row["harmed"]) for row in selected) / len(selected)
                if selected
                else None
            ),
            "passes_material_no_harm_gate": bool(gains)
            and statistics.fmean(gains) >= MATERIAL_GAIN_THRESHOLD
            and not any(bool(row["harmed"]) for row in selected),
        }

    return {
        "policy_id": policy_id,
        "scope": "ratio_gap_over_period_le_0.5" if ratio_scope else "old_all_affected",
        "operator": "multi_cycle_median" if use_multi else "current_single_cycle",
        "cases": cases,
        "overall": _effect(cases),
        "activated": _effect([row for row in cases if bool(row["activated"])]),
    }


def _mechanism_verdict(
    case_rows: list[dict[str, Any]], policies: dict[str, dict[str, object]]
) -> dict[str, object]:
    p25 = [
        row
        for row in case_rows
        if row["dataset_id"] == "monash:traffic_hourly"
        and row["current_period"] == 25
    ]
    if len(p25) != 2:
        raise AssertionError("expected exactly two Monash p25 cases")
    recovery = [
        {
            "series_uid": row["series_uid"],
            "multi_gain_over_current": (
                float(row["current_single_cycle_loss"])
                - float(row["multi_cycle_loss"])
            ),
            "multi_gain_over_identity": (
                float(row["identity_loss"]) - float(row["multi_cycle_loss"])
            ),
        }
        for row in p25
    ]
    for row in recovery:
        row["recovered"] = (
            float(row["multi_gain_over_current"]) >= MATERIAL_GAIN_THRESHOLD
            and float(row["multi_gain_over_identity"]) >= HARM_THRESHOLD
        )
    recovered_count = sum(bool(row["recovered"]) for row in recovery)
    ratio_single_pass = bool(
        policies["ratio_scope_single"]["activated"]["passes_material_no_harm_gate"]
    )
    ratio_multi_pass = bool(
        policies["ratio_scope_multi"]["activated"]["passes_material_no_harm_gate"]
    )

    if recovered_count == 2:
        verdict = "OPERATOR_IMPLICATED"
    elif recovered_count == 0 and ratio_single_pass:
        verdict = "SCOPE_IMPLICATED"
    elif ratio_multi_pass and not ratio_single_pass:
        verdict = "SCOPE_OPERATOR_INTERACTION"
    elif recovered_count == 1:
        verdict = "MIXED"
    else:
        verdict = "NO_MATERIAL_REPAIR"
    return {
        "verdict": verdict,
        "p25_recovered_count": recovered_count,
        "p25_cases": recovery,
        "ratio_single_activated_gate_pass": ratio_single_pass,
        "ratio_multi_activated_gate_pass": ratio_multi_pass,
        "deterministic_exclusive_rule_order": [
            "OPERATOR_IMPLICATED: both p25 cases recover",
            "SCOPE_IMPLICATED: zero p25 recover and ratio-single activated gate passes",
            "SCOPE_OPERATOR_INTERACTION: ratio-multi passes, ratio-single fails, and operator rule did not match",
            "MIXED: exactly one p25 case recovers and interaction rule did not match",
            "NO_MATERIAL_REPAIR: none of the preceding rules match",
        ],
        "non_recovery_cases": "p48 and p47 enter policy summaries but not p25 recovery",
    }


def run_e2_natural_scope_operator_diagnostic(
    valuator: _Valuator,
    *,
    promotion_report_path: Path,
    registry_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    promotion_report, roster = _read_roster(promotion_report_path)
    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    case_rows: list[dict[str, Any]] = []
    judge_calls = 0

    for cached in roster:
        uid = str(cached["series_uid"])
        record = records.get(uid)
        if record is None or record.dataset_id != cached["dataset_id"]:
            raise ValueError(f"diagnostic UID is absent or mismatched: {uid}")
        if record.admission_reasons != ():
            raise ValueError(f"diagnostic UID is ineligible: {uid}")

        clean_context = _load_visible_context(record, clean_root)
        corrupt_context = clean_context.copy()
        corrupt_context[slice(*GAP_BOUNDS)] = np.nan
        _, replayed_period = _global_features(corrupt_context)
        current_period = int(cached["observed_period"])
        if replayed_period != current_period:
            raise ValueError(f"period replay mismatch: {uid}")
        clean_future = _load_verified_selected_future(record, clean_root)
        multi_context = _multi_cycle_context(corrupt_context, current_period)
        multi_loss = float(
            valuator.evaluate(
                multi_context, clean_future, scale_context=clean_context
            ).loss_j
        )
        judge_calls += 1
        if not math.isfinite(multi_loss):
            raise ValueError(f"non-finite diagnostic loss: {uid}")

        gap_length = GAP_BOUNDS[1] - GAP_BOUNDS[0]
        ratio = gap_length / current_period
        case_rows.append(
            {
                "series_uid": uid,
                "dataset_id": cached["dataset_id"],
                "split": "support_a",
                "subsplit": "support_a_validation_exposed",
                "windows": {
                    "context": list(CONTEXT_BOUNDS),
                    "future": list(FUTURE_BOUNDS),
                    "gap_relative_to_context": list(GAP_BOUNDS),
                },
                "current_period": current_period,
                "gap_length": gap_length,
                "gap_to_period_ratio": ratio,
                "ratio_scope_activated": ratio <= RATIO_SCOPE_MAX,
                "identity_loss": float(cached["identity_loss"]),
                "current_single_cycle_loss": float(cached["candidate_loss"]),
                "multi_cycle_loss": multi_loss,
                "multi_gain_over_identity": float(cached["identity_loss"]) - multi_loss,
                "multi_gain_over_current": float(cached["candidate_loss"]) - multi_loss,
            }
        )

    if judge_calls != 4:
        raise AssertionError(f"expected exactly 4 new Judge calls, observed {judge_calls}")
    policies = {
        policy_id: _policy_summary(
            case_rows,
            policy_id=policy_id,
            use_multi=use_multi,
            ratio_scope=ratio_scope,
        )
        for policy_id, use_multi, ratio_scope in (
            ("old_scope_single", False, False),
            ("ratio_scope_single", False, True),
            ("old_scope_multi", True, False),
            ("ratio_scope_multi", True, True),
        )
    }
    return {
        "schema_version": "e2-natural-scope-operator-diagnostic/1",
        "scientific_role": "exposed_source_scope_operator_fault_localization",
        "evidence_disposition": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "promotion_eligible": False,
        "target_transfer_eligible": False,
        "source_promotion_status_at_start": promotion_report["policy_status"],
        "configuration": {
            "roster_case_count": 4,
            "incumbent": "identity",
            "current_operator": "period_complete",
            "multi_cycle_operator": "period_median_complete",
            "multi_cycle_params": {"cycles": 3, "min_donors": 2},
            "ratio_scope_rule": "gap_length / current_period <= 0.5",
            "ratio_scope_is_human_development_hypothesis_not_fresh_evidence": True,
            "harm_threshold": HARM_THRESHOLD,
            "material_gain_threshold": MATERIAL_GAIN_THRESHOLD,
            "gain_definition": "identity_loss - policy_loss",
        },
        "information_wall": {
            "roster": "four already_exposed_source_promotion_affected_cases",
            "cached_identity_and_current_single_cycle_losses_reused": True,
            "new_multi_cycle_judgments_only": True,
            "new_source_or_target_case_selected": False,
            "target_context_read": False,
            "target_future_read": False,
            "uci_accessed": False,
            "query_accessed": False,
            "query_plan_generated": False,
        },
        "judge_call_count": judge_calls,
        "cases": case_rows,
        "policies": policies,
        "mechanism_verdict": _mechanism_verdict(case_rows, policies),
        "claim_limit": (
            "Development diagnostic on already-exposed Source cases only; not fresh "
            "Promotion evidence and inadmissible for Target, Query, or transfer claims."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_scope_operator_diagnostic_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_scope_operator_diagnostic(
        FrozenChronosValuator(),
        promotion_report_path=args.promotion_report,
        registry_path=args.registry,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"mechanism_verdict={report['mechanism_verdict']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_e2_natural_scope_operator_diagnostic"]
