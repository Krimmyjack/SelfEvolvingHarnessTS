"""Localize period-binding failure on already-exposed E2 source cases.

This is a development-only diagnostic, never fresh Promotion evidence.  The
roster is fixed by the failed natural Source promotion report.  For its three
exposed Monash hourly cases, cached Identity/current-seasonal losses are reused
and only two new arms are judged: Linear and the unchanged seasonal operator
bound to the deployment-visible 24-hour period.  No Target data is selectable.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    _execute_program,
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


DATASET_ID = "monash:traffic_hourly"
CALENDAR_PERIOD = 24
EXPECTED_PERIODS = (25, 25, 48)
HARM_MARGIN = 0.005
RECOVERY_MARGIN = 0.005


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


def _read_report(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
    if not isinstance(raw_cases, list):
        raise TypeError("promotion report lacks affected_cases")

    cases: list[dict[str, Any]] = []
    for raw in raw_cases:
        if not isinstance(raw, Mapping) or raw.get("dataset_id") != DATASET_ID:
            continue
        if (
            raw.get("split") != "support_a"
            or raw.get("subsplit") != "support_a_validation"
            or raw.get("candidate_action") != "seasonal"
        ):
            raise ValueError("diagnostic roster is not the exposed Source seasonal roster")
        row = dict(raw)
        for key in ("identity_loss", "candidate_loss"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"invalid cached loss: {key}")
        cases.append(row)
    cases.sort(key=lambda row: (int(row["observed_period"]), str(row["series_uid"])))
    periods = tuple(int(row["observed_period"]) for row in cases)
    if periods != EXPECTED_PERIODS:
        raise ValueError(f"expected exposed periods {EXPECTED_PERIODS}, observed {periods}")
    return report, cases


def _localize(rows: list[dict[str, Any]]) -> dict[str, object]:
    harmful_p25 = [
        row
        for row in rows
        if row["current_binding_period"] == 25
        and row["current_gain_over_identity"] < -HARM_MARGIN
    ]
    if len(harmful_p25) != 2:
        raise AssertionError("expected exactly two exposed harmful p=25 cases")
    recovered = [
        row
        for row in harmful_p25
        if row["calendar_gain_over_current"] >= RECOVERY_MARGIN
        and row["calendar_gain_over_identity"] >= -HARM_MARGIN
    ]
    if len(recovered) == 2:
        verdict = "BINDING_IMPLICATED"
        next_surface = "observation_and_binding"
    elif len(recovered) == 0:
        verdict = "SIMPLE_BINDING_NOT_SUFFICIENT"
        next_surface = "applicability_or_operator"
    else:
        verdict = "MIXED"
        next_surface = "binding_plus_applicability_or_operator"
    return {
        "verdict": verdict,
        "next_surface": next_surface,
        "harmful_p25_case_count": 2,
        "recovered_p25_case_count": len(recovered),
        "harm_margin": HARM_MARGIN,
        "recovery_margin": RECOVERY_MARGIN,
        "p48_role": "alias_probe_only_not_part_of_verdict",
    }


def run_e2_natural_binding_diagnostic(
    valuator: _Valuator,
    *,
    promotion_report_path: Path,
    registry_path: Path,
    clean_root: Path,
) -> dict[str, object]:
    promotion_report, exposed = _read_report(promotion_report_path)
    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    rows: list[dict[str, Any]] = []
    judge_calls = 0

    for cached in exposed:
        uid = str(cached["series_uid"])
        record = records.get(uid)
        if record is None or record.dataset_id != DATASET_ID or record.frequency != "hourly":
            raise ValueError(f"diagnostic UID is not an hourly Monash registry row: {uid}")
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

        linear_context = _execute_program(
            "linear", corrupt_context, observed_period=current_period
        )
        calendar_context = _execute_program(
            "seasonal", corrupt_context, observed_period=CALENDAR_PERIOD
        )
        linear_loss = float(
            valuator.evaluate(
                linear_context, clean_future, scale_context=clean_context
            ).loss_j
        )
        calendar_loss = float(
            valuator.evaluate(
                calendar_context, clean_future, scale_context=clean_context
            ).loss_j
        )
        judge_calls += 2
        if not math.isfinite(linear_loss) or not math.isfinite(calendar_loss):
            raise ValueError(f"non-finite diagnostic loss: {uid}")

        identity_loss = float(cached["identity_loss"])
        current_loss = float(cached["candidate_loss"])
        losses = {
            "identity_cached": identity_loss,
            "seasonal_current_binding_cached": current_loss,
            "linear_new": linear_loss,
            "seasonal_calendar_24_new": calendar_loss,
        }
        rows.append(
            {
                "series_uid": uid,
                "dataset_id": DATASET_ID,
                "split": "support_a",
                "subsplit": "support_a_validation_exposed",
                "current_binding_period": current_period,
                "calendar_binding_period": CALENDAR_PERIOD,
                "identity_loss": identity_loss,
                "current_loss": current_loss,
                "linear_loss": linear_loss,
                "calendar_loss": calendar_loss,
                "current_gain_over_identity": identity_loss - current_loss,
                "linear_gain_over_identity": identity_loss - linear_loss,
                "calendar_gain_over_identity": identity_loss - calendar_loss,
                "calendar_gain_over_current": current_loss - calendar_loss,
                "diagnostic_winner": min(
                    losses, key=lambda action: (losses[action], action)
                ),
            }
        )

    if judge_calls != 6:
        raise AssertionError(f"expected exactly 6 new Judge calls, observed {judge_calls}")
    return {
        "schema_version": "e2-natural-binding-diagnostic/1",
        "scientific_role": "exposed_source_case_fault_localization",
        "evidence_disposition": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "promotion_eligible": False,
        "target_transfer_eligible": False,
        "source_promotion_status_at_start": promotion_report["policy_status"],
        "configuration": {
            "dataset_id": DATASET_ID,
            "calendar_period": CALENDAR_PERIOD,
            "new_actions": ["linear", "seasonal_calendar_24"],
            "current_operator_implementation_unchanged": True,
            "cached_actions_rejudged": False,
        },
        "information_wall": {
            "roster": "already_exposed_source_promotion_cases",
            "new_source_or_target_case_selected": False,
            "target_context_read": False,
            "target_future_read": False,
            "query_plan_generated": False,
        },
        "judge_call_count": judge_calls,
        "cases": rows,
        "fault_localization": _localize(rows),
        "claim_limit": (
            "Development fault localization on already-exposed Source cases only; "
            "inadmissible for Promotion, Memory-transfer, or Target claims."
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
        default=project_root / "artifacts/functional/e2/natural_binding_diagnostic_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_binding_diagnostic(
        FrozenChronosValuator(),
        promotion_report_path=args.promotion_report,
        registry_path=args.registry,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"fault_localization={report['fault_localization']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_e2_natural_binding_diagnostic"]
