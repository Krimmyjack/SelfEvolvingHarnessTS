"""Test a pseudo-gap applicability observation without any new Judge calls.

This development diagnostic reuses twenty already-exposed Source outcomes.  It
asks whether reconstruction on four fixed, formerly observed windows separates
cases where the existing seasonal program helped from cases where it harmed.
Only each report-declared visible context is loaded; no future or Query surface
is available to this runner.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _slot
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
    SeriesRecord,
    read_registry_jsonl,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    _execute_program,
    _global_features,
)


GAP_BOUNDS = (156, 180)
PSEUDO_GAP_WINDOWS = ((48, 72), (72, 96), (96, 120), (120, 144))
HARM_THRESHOLD = -0.005
MATERIAL_GAIN_THRESHOLD = 0.005
MIN_SUPPORTED_CASES = 4
MIN_SUPPORTED_DATASETS = 2
MIN_HARM_RATE_REDUCTION = 0.10
MIN_MEAN_GAIN_IMPROVEMENT = 0.005


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _window(raw: object, *, name: str) -> tuple[int, int]:
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
    ):
        raise ValueError(f"{name} must be a two-integer list")
    start, end = int(raw[0]), int(raw[1])
    if start < 0 or end <= start:
        raise ValueError(f"invalid {name}: {raw}")
    return start, end


def _read_outcome_cases(
    evidence_report_path: Path,
    promotion_report_path: Path,
) -> list[dict[str, Any]]:
    evidence = json.loads(evidence_report_path.read_text("utf-8"))
    if evidence.get("schema_version") != "e2-natural-source-evidence/1":
        raise ValueError("unsupported source-evidence report schema")
    evidence_cases = evidence.get("cases")
    if not isinstance(evidence_cases, list) or len(evidence_cases) != 16:
        raise ValueError("source-evidence report must contain sixteen cases")

    promotion = json.loads(promotion_report_path.read_text("utf-8"))
    if promotion.get("schema_version") != "e2-natural-source-promotion/1":
        raise ValueError("unsupported source-promotion report schema")
    promotion_cases = promotion.get("affected_cases")
    if not isinstance(promotion_cases, list) or len(promotion_cases) != 4:
        raise ValueError("source-promotion report must contain four affected cases")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for origin, raw_cases in (
        ("source_evidence_exposed", evidence_cases),
        ("promotion_replay_exposed", promotion_cases),
    ):
        for raw in raw_cases:
            if not isinstance(raw, Mapping):
                raise TypeError("outcome case must be an object")
            uid = raw.get("series_uid")
            dataset_id = raw.get("dataset_id")
            observed_period = raw.get("observed_period")
            windows = raw.get("windows")
            if not isinstance(uid, str) or not uid or uid in seen:
                raise ValueError("the twenty exposed UIDs must be unique strings")
            if not isinstance(dataset_id, str) or not dataset_id:
                raise ValueError(f"invalid dataset ID: {uid}")
            if (
                isinstance(observed_period, bool)
                or not isinstance(observed_period, int)
                or observed_period < 1
            ):
                raise ValueError(f"invalid observed period: {uid}")
            if not isinstance(windows, Mapping):
                raise ValueError(f"missing declared windows: {uid}")
            context_bounds = _window(windows.get("context"), name=f"{uid}/context")
            if context_bounds[1] - context_bounds[0] != 192:
                raise ValueError(f"declared context must contain 192 values: {uid}")
            if _window(
                windows.get("gap_relative_to_context"), name=f"{uid}/gap"
            ) != GAP_BOUNDS:
                raise ValueError(f"unexpected real gap: {uid}")

            if origin == "source_evidence_exposed":
                losses = raw.get("loss_by_action")
                if not isinstance(losses, Mapping):
                    raise ValueError(f"missing cached source losses: {uid}")
                identity_loss = _finite_number(
                    losses.get("identity"), name=f"{uid}/identity_loss"
                )
                linear_loss = _finite_number(
                    losses.get("linear"), name=f"{uid}/linear_loss"
                )
                seasonal_loss = _finite_number(
                    losses.get("seasonal"), name=f"{uid}/seasonal_loss"
                )
                gain_over_linear: float | None = linear_loss - seasonal_loss
                gain_over_best_incumbent: float | None = (
                    min(identity_loss, linear_loss) - seasonal_loss
                )
            else:
                identity_loss = _finite_number(
                    raw.get("identity_loss"), name=f"{uid}/identity_loss"
                )
                seasonal_loss = _finite_number(
                    raw.get("candidate_loss"), name=f"{uid}/candidate_loss"
                )
                cached_gain = _finite_number(
                    raw.get("gain_over_identity"), name=f"{uid}/gain_over_identity"
                )
                if not math.isclose(
                    cached_gain,
                    identity_loss - seasonal_loss,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"cached promotion gain disagrees with losses: {uid}")
                linear_loss = None
                gain_over_linear = None
                gain_over_best_incumbent = None

            gain_over_identity = identity_loss - seasonal_loss
            seen.add(uid)
            rows.append(
                {
                    "series_uid": uid,
                    "dataset_id": dataset_id,
                    "origin": origin,
                    "context_bounds": context_bounds,
                    "report_observed_period": observed_period,
                    "identity_loss": identity_loss,
                    "linear_loss": linear_loss,
                    "seasonal_loss": seasonal_loss,
                    "actual_seasonal_gain_over_identity": gain_over_identity,
                    "actual_seasonal_gain_over_linear": gain_over_linear,
                    "actual_seasonal_gain_over_best_incumbent": gain_over_best_incumbent,
                    "actual_harm": gain_over_identity < HARM_THRESHOLD,
                    "actual_material_benefit": gain_over_identity
                    >= MATERIAL_GAIN_THRESHOLD,
                }
            )

    if len(rows) != 20 or len(seen) != 20:
        raise AssertionError("diagnostic roster must contain twenty unique exposed cases")
    return rows


def _load_declared_context(
    record: SeriesRecord,
    clean_root: Path,
    context_bounds: tuple[int, int],
) -> np.ndarray:
    """Materialize the declared context slice only; never index its future."""

    slot = _slot(record, clean_root)
    values = np.load(slot / "values.npy", allow_pickle=False, mmap_mode="r")
    if values.ndim != 1 or values.shape != (record.length,):
        raise ValueError(f"series shape disagrees with registry: {record.series_uid}")
    context = np.asarray(values[slice(*context_bounds)], dtype=np.float64).copy()
    if context.shape != (192,) or not np.isfinite(context).all():
        raise ValueError(f"invalid declared visible context: {record.series_uid}")
    return context


def _robust_scale(visible_values: np.ndarray) -> float:
    finite = np.asarray(visible_values[np.isfinite(visible_values)], dtype=np.float64)
    if finite.size == 0:
        raise ValueError("visible context has no finite values")
    center = float(np.median(finite))
    mad_scale = 1.4826 * float(np.median(np.abs(finite - center)))
    if mad_scale > 1e-8:
        return mad_scale
    # Discrete traffic series can have zero MAD despite real variation.  Avoid
    # meaningless billion-scale normalized errors while keeping MAD primary.
    return max(float(np.std(finite)), 1e-8)


def _probe_window(
    base_corrupt_context: np.ndarray,
    clean_context: np.ndarray,
    bounds: tuple[int, int],
    *,
    scale: float,
) -> dict[str, Any]:
    start, end = bounds
    row: dict[str, Any] = {"window": [start, end], "valid": False}
    truth = np.asarray(clean_context[start:end], dtype=np.float64)
    if truth.shape != (end - start,) or not np.isfinite(truth).all():
        row["failure_reason"] = "pseudo_window_truth_not_fully_observed"
        return row

    probe_context = np.asarray(base_corrupt_context, dtype=np.float64).copy()
    probe_context[start:end] = np.nan
    try:
        _, probe_period = _global_features(probe_context)
    except Exception as exc:  # record every fixed window rather than cherry-pick
        row["failure_reason"] = f"period_extraction_failed:{type(exc).__name__}"
        return row
    try:
        linear = _execute_program(
            "linear", probe_context, observed_period=probe_period
        )
    except Exception as exc:
        row["failure_reason"] = f"linear_execution_failed:{type(exc).__name__}"
        row["probe_observed_period"] = probe_period
        return row
    try:
        seasonal = _execute_program(
            "seasonal", probe_context, observed_period=probe_period
        )
    except Exception as exc:
        row["failure_reason"] = f"seasonal_execution_failed:{type(exc).__name__}"
        row["probe_observed_period"] = probe_period
        return row

    linear_values = np.asarray(linear[start:end], dtype=np.float64)
    seasonal_values = np.asarray(seasonal[start:end], dtype=np.float64)
    if not np.isfinite(linear_values).all():
        row["failure_reason"] = "linear_pseudo_window_non_finite"
        row["probe_observed_period"] = probe_period
        return row
    if not np.isfinite(seasonal_values).all():
        row["failure_reason"] = "seasonal_pseudo_window_non_finite"
        row["probe_observed_period"] = probe_period
        return row

    linear_nmae = float(np.mean(np.abs(linear_values - truth)) / scale)
    seasonal_nmae = float(np.mean(np.abs(seasonal_values - truth)) / scale)
    row.update(
        {
            "valid": True,
            "failure_reason": None,
            "probe_observed_period": probe_period,
            "linear_nmae": linear_nmae,
            "seasonal_nmae": seasonal_nmae,
            "margin": linear_nmae - seasonal_nmae,
        }
    )
    return row


def _effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "case_count": 0,
            "dataset_count": 0,
            "datasets": [],
            "mean_gain_over_identity": None,
            "harm_rate": None,
            "material_benefit_rate": None,
        }
    gains = [float(row["actual_seasonal_gain_over_identity"]) for row in rows]
    datasets = sorted({str(row["dataset_id"]) for row in rows})
    return {
        "case_count": len(rows),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "mean_gain_over_identity": statistics.fmean(gains),
        "harm_rate": sum(bool(row["actual_harm"]) for row in rows) / len(rows),
        "material_benefit_rate": sum(
            bool(row["actual_material_benefit"]) for row in rows
        )
        / len(rows),
    }


def _selector_summary(
    rows: list[dict[str, Any]], selector: str
) -> dict[str, Any]:
    selected = [row for row in rows if bool(row[selector])]
    rejected = [row for row in rows if not bool(row[selector])]
    return {
        "selector_field": selector,
        "supported": _effect(selected),
        "unsupported": _effect(rejected),
    }


def _incumbent_alignment_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_rows = [
        row
        for row in rows
        if row["actual_seasonal_gain_over_linear"] is not None
        and row["actual_seasonal_gain_over_best_incumbent"] is not None
    ]
    if not source_rows:
        return {
            "case_count": 0,
            "dataset_count": 0,
            "mean_gain_over_linear": None,
            "harm_rate_vs_linear": None,
            "mean_gain_over_best_incumbent": None,
            "harm_rate_vs_best_incumbent": None,
        }
    linear_gains = [
        float(row["actual_seasonal_gain_over_linear"]) for row in source_rows
    ]
    incumbent_gains = [
        float(row["actual_seasonal_gain_over_best_incumbent"])
        for row in source_rows
    ]
    return {
        "case_count": len(source_rows),
        "dataset_count": len({str(row["dataset_id"]) for row in source_rows}),
        "mean_gain_over_linear": statistics.fmean(linear_gains),
        "harm_rate_vs_linear": sum(
            gain < HARM_THRESHOLD for gain in linear_gains
        )
        / len(linear_gains),
        "mean_gain_over_best_incumbent": statistics.fmean(incumbent_gains),
        "harm_rate_vs_best_incumbent": sum(
            gain < HARM_THRESHOLD for gain in incumbent_gains
        )
        / len(incumbent_gains),
    }


def run_e2_natural_pseudogap_observation(
    *,
    registry_path: Path,
    clean_root: Path,
    evidence_report_path: Path,
    promotion_report_path: Path,
) -> dict[str, object]:
    outcome_cases = _read_outcome_cases(evidence_report_path, promotion_report_path)
    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    rows: list[dict[str, Any]] = []

    for outcome in outcome_cases:
        uid = str(outcome["series_uid"])
        record = records.get(uid)
        if record is None or record.dataset_id != outcome["dataset_id"]:
            raise ValueError(f"diagnostic UID is absent or mismatched: {uid}")
        clean_context = _load_declared_context(
            record, clean_root, outcome["context_bounds"]
        )
        base_corrupt_context = clean_context.copy()
        base_corrupt_context[slice(*GAP_BOUNDS)] = np.nan
        if int(np.isnan(base_corrupt_context).sum()) != GAP_BOUNDS[1] - GAP_BOUNDS[0]:
            raise AssertionError(f"real gap injection failed: {uid}")
        _, replayed_period = _global_features(base_corrupt_context)
        if replayed_period != int(outcome["report_observed_period"]):
            raise ValueError(f"base observed period replay mismatch: {uid}")
        scale = _robust_scale(base_corrupt_context)
        probes = [
            _probe_window(
                base_corrupt_context,
                clean_context,
                bounds,
                scale=scale,
            )
            for bounds in PSEUDO_GAP_WINDOWS
        ]
        valid_margins = [float(row["margin"]) for row in probes if row["valid"]]
        valid_count = len(valid_margins)
        median_margin = (
            float(statistics.median(valid_margins)) if valid_margins else None
        )
        positive_fraction = (
            sum(margin > 0.0 for margin in valid_margins) / valid_count
            if valid_count
            else None
        )
        probe_supported = (
            valid_count == len(PSEUDO_GAP_WINDOWS)
            and median_margin is not None
            and median_margin > 0.0
        )
        period = int(outcome["report_observed_period"])
        rows.append(
            {
                **{key: value for key, value in outcome.items() if key != "context_bounds"},
                "windows": {
                    "context": list(outcome["context_bounds"]),
                    "real_gap_relative_to_context": list(GAP_BOUNDS),
                },
                "base_observed_period": replayed_period,
                "robust_scale_from_base_visible_values": scale,
                "pseudo_windows": probes,
                "valid_window_count": valid_count,
                "median_margin": median_margin,
                "positive_window_fraction": positive_fraction,
                "probe_supported": probe_supported,
                "period_baseline_supported": period >= 25,
                "ratio_baseline_supported": (GAP_BOUNDS[1] - GAP_BOUNDS[0])
                / period
                <= 0.5,
            }
        )

    if len(rows) != 20:
        raise AssertionError("expected twenty pseudo-gap cases")
    if sum(len(row["pseudo_windows"]) for row in rows) != 80:
        raise AssertionError("expected exactly eighty fixed pseudo-gap windows")

    overall = _effect(rows)
    supported_rows = [row for row in rows if row["probe_supported"]]
    unsupported_rows = [row for row in rows if not row["probe_supported"]]
    supported = _effect(supported_rows)
    unsupported = _effect(unsupported_rows)
    if supported["case_count"]:
        harm_rate_reduction = float(overall["harm_rate"]) - float(
            supported["harm_rate"]
        )
        mean_gain_improvement = float(supported["mean_gain_over_identity"]) - float(
            overall["mean_gain_over_identity"]
        )
    else:
        harm_rate_reduction = None
        mean_gain_improvement = None

    gates = {
        "supported_case_count": {
            "threshold": MIN_SUPPORTED_CASES,
            "value": supported["case_count"],
            "pass": int(supported["case_count"]) >= MIN_SUPPORTED_CASES,
        },
        "supported_dataset_count": {
            "threshold": MIN_SUPPORTED_DATASETS,
            "value": supported["dataset_count"],
            "pass": int(supported["dataset_count"]) >= MIN_SUPPORTED_DATASETS,
        },
        "harm_rate_reduction_vs_all": {
            "threshold": MIN_HARM_RATE_REDUCTION,
            "value": harm_rate_reduction,
            "pass": harm_rate_reduction is not None
            and harm_rate_reduction >= MIN_HARM_RATE_REDUCTION,
        },
        "mean_gain_improvement_vs_all": {
            "threshold": MIN_MEAN_GAIN_IMPROVEMENT,
            "value": mean_gain_improvement,
            "pass": mean_gain_improvement is not None
            and mean_gain_improvement >= MIN_MEAN_GAIN_IMPROVEMENT,
        },
    }
    verdict = (
        "OBSERVATION_PROMISING"
        if all(bool(gate["pass"]) for gate in gates.values())
        else "OBSERVATION_NOT_YET_USEFUL"
    )
    promotion_rows = [
        row for row in rows if row["origin"] == "promotion_replay_exposed"
    ]
    harmful_p25 = [
        row
        for row in promotion_rows
        if row["dataset_id"] == "monash:traffic_hourly"
        and row["base_observed_period"] == 25
        and row["actual_harm"]
    ]
    if len(promotion_rows) != 4 or len(harmful_p25) != 2:
        raise AssertionError("expected four promotion replays and two harmful p25 cases")

    def _decision(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "series_uid": row["series_uid"],
            "dataset_id": row["dataset_id"],
            "base_observed_period": row["base_observed_period"],
            "actual_seasonal_gain_over_identity": row[
                "actual_seasonal_gain_over_identity"
            ],
            "actual_harm": row["actual_harm"],
            "median_margin": row["median_margin"],
            "positive_window_fraction": row["positive_window_fraction"],
            "valid_window_count": row["valid_window_count"],
            "probe_supported": row["probe_supported"],
            "period_baseline_supported": row["period_baseline_supported"],
            "ratio_baseline_supported": row["ratio_baseline_supported"],
        }

    return {
        "schema_version": "e2-natural-pseudogap-observation/1",
        "scientific_role": "program_specific_observation_feasibility_on_exposed_source_cases",
        "evidence_disposition": "DEVELOPMENT_DIAGNOSTIC_ONLY",
        "verdict": verdict,
        "promotion_eligible": False,
        "target_transfer_eligible": False,
        "configuration": {
            "real_gap_relative_to_context": list(GAP_BOUNDS),
            "fixed_pseudo_gap_windows": [list(bounds) for bounds in PSEUDO_GAP_WINDOWS],
            "program_contrast": "linear_vs_existing_single_cycle_seasonal",
            "error": (
                "MAE divided by one case-level MAD scale from base visible values; "
                "standard deviation is used only when MAD degenerates to zero"
            ),
            "margin": "linear_nmae - seasonal_nmae; positive supports seasonal",
            "probe_supported_rule": "all four windows valid and median_margin > 0",
            "outcome_harm_rule": f"seasonal gain over identity < {HARM_THRESHOLD}",
            "outcome_material_benefit_rule": (
                f"seasonal gain over identity >= {MATERIAL_GAIN_THRESHOLD}"
            ),
            "thresholds_are_frozen_not_fitted": True,
        },
        "information_wall": {
            "roster": "sixteen exposed source-evidence plus four exposed promotion cases",
            "report_cached_outcomes_are_the_only_outcome_labels": True,
            "only_report_declared_visible_context_slices_loaded": True,
            "actual_gap_remains_masked_during_every_probe": True,
            "pseudo_period_reestimated_after_each_pseudo_mask": True,
            "future_read": False,
            "target_or_query_read": False,
            "query_plan_generated": False,
        },
        "judge_call_count": 0,
        "case_count": len(rows),
        "pseudo_window_count": sum(len(row["pseudo_windows"]) for row in rows),
        "cases": rows,
        "probe_effect": {
            "all": overall,
            "supported": supported,
            "unsupported": unsupported,
        },
        "fixed_selector_comparison": {
            "pseudo_gap_probe": _selector_summary(rows, "probe_supported"),
            "period_ge_25": _selector_summary(rows, "period_baseline_supported"),
            "gap_over_period_le_0.5": _selector_summary(
                rows, "ratio_baseline_supported"
            ),
        },
        "secondary_incumbent_alignment_diagnostic": {
            "status": "POST_RESULT_DESCRIPTIVE",
            "scope": "sixteen exposed source-evidence cases with cached linear losses",
            "motivation": (
                "the probe contrasts seasonal with linear while the primary frozen "
                "outcome label and verdict contrast seasonal with identity"
            ),
            "changes_primary_verdict_or_gates": False,
            "harm_threshold": HARM_THRESHOLD,
            "all": _incumbent_alignment_effect(rows),
            "probe_supported": _incumbent_alignment_effect(supported_rows),
            "probe_unsupported": _incumbent_alignment_effect(unsupported_rows),
        },
        "promotion_replay_decisions": [_decision(row) for row in promotion_rows],
        "harmful_p25_decisions": [_decision(row) for row in harmful_p25],
        "verdict_gates": gates,
        "claim_limit": (
            "Self-supervised observation feasibility on twenty already-exposed Source "
            "cases only; not ApplicabilityWitness validation, Source Promotion, Target, "
            "Memory, adaptation, transfer, or causal evidence."
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
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument(
        "--evidence-report",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_source_evidence_report.json",
    )
    parser.add_argument(
        "--promotion-report",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_source_promotion_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_pseudogap_observation_report.json",
    )
    args = parser.parse_args()

    report = run_e2_natural_pseudogap_observation(
        registry_path=args.registry,
        clean_root=args.clean_root,
        evidence_report_path=args.evidence_report,
        promotion_report_path=args.promotion_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"verdict={report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_e2_natural_pseudogap_observation"]
