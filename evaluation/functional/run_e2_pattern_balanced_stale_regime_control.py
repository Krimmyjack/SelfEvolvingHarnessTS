"""Run one pattern-balanced stale-regime training-data control.

Traffic and FRED remain natural base series.  The controlled stale case reverses
the normalized context-to-target mapping for the first three training anchors;
the matched recurrent case keeps the same old rows valid.  The sole optional
Program excludes that old anchor prefix.  A recent-only fit is shared by both
cases, so each dataset needs only three Ridge fits.

This is a mechanism-identification control, not natural Capability evidence.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    ANCHORS,
    CONTEXT_LENGTH,
    HORIZON,
    J0_PLAN_PATH,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-pattern-balanced-stale-regime-control/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_pattern_balanced_stale_regime_control_report.json"
)
DATASETS = ("monash:traffic_hourly", "legacy_monash:fred_md")
OLD_ANCHORS = tuple(ANCHORS[:3])
RECENT_ANCHORS = tuple(ANCHORS[3:])
MIN_STALE_MEAN_GAIN = 0.05
MAX_RECURRENT_MEAN_GAIN = 0.01
MIN_EFFECT_CONTRAST = 0.05
MIN_POSITIVE_UIDS = 6


def _safe_correlation(np: Any, left: Any, right: Any) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.size < 2
        or left.shape != right.shape
        or float(np.std(left)) <= 1e-12
        or float(np.std(right)) <= 1e-12
    ):
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _mapping_observation(
    np: Any,
    targets: Any,
    provenance: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize visible old/recent target-shape agreement without eval outcomes."""

    array = np.asarray(targets, dtype=np.float64)
    old_by_uid: dict[str, list[Any]] = {}
    recent_by_uid: dict[str, list[Any]] = {}
    for index, row in enumerate(provenance):
        uid = str(row["series_uid"])
        anchor = int(row["anchor"])
        target = array[index]
        bucket = old_by_uid if anchor in OLD_ANCHORS else recent_by_uid
        bucket.setdefault(uid, []).append(target)
    correlations: list[float] = []
    distances: list[float] = []
    for uid in sorted(old_by_uid):
        old = np.mean(np.asarray(old_by_uid[uid], dtype=np.float64), axis=0)
        recent = np.mean(np.asarray(recent_by_uid[uid], dtype=np.float64), axis=0)
        correlations.append(_safe_correlation(np, old, recent))
        distances.append(float(np.sqrt(np.mean((old - recent) ** 2))))
    return {
        "series_count": len(correlations),
        "median_old_recent_target_shape_correlation": statistics.median(correlations),
        "mean_old_recent_target_shape_rms_distance": statistics.fmean(distances),
        "minimum_shape_correlation": min(correlations),
        "maximum_shape_correlation": max(correlations),
        "visibility": "training_context_and_target_only",
        "used_for_program_selection": False,
    }


def _effect_summary(gains: list[float]) -> dict[str, object]:
    return {
        "mean_gain": statistics.fmean(gains),
        "median_gain": statistics.median(gains),
        "positive_uid_count": sum(gain > 0.0 for gain in gains),
        "harmful_uid_count": sum(gain < 0.0 for gain in gains),
    }


def run(root: Path) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import Ridge

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
        read_registry_jsonl,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
        _center_scale,
    )

    roster = [
        row
        for row in _read_object(root / J0_PLAN_PATH)["roster"]
        if row["dataset_id"] in DATASETS
    ]
    if len(roster) != 40 or any(
        str(row["dataset_id"]).startswith("uci") for row in roster
    ):
        raise ValueError("expected exposed Traffic/FRED Source rosters only")
    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )

    fit_count = 0
    dataset_evidence: list[dict[str, object]] = []
    for dataset_id in DATASETS:
        spec = SPECS[dataset_id]
        period = int(spec["period"])
        train_stop = int(spec["train_stop"])
        train_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "train"
        ]
        eval_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "eval"
        ]
        if len(train_rows) != 12 or len(eval_rows) != 8:
            raise ValueError(f"Source roster geometry changed: {dataset_id}")

        x_rows: list[Any] = []
        y_rows: list[Any] = []
        provenance: list[dict[str, object]] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                raw = values[uid]
                context = np.asarray(
                    raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64
                )
                target = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
                center, scale, method = _center_scale(context)
                if (
                    context.shape != (CONTEXT_LENGTH,)
                    or target.shape != (HORIZON,)
                    or not np.isfinite(context).all()
                    or not np.isfinite(target).all()
                    or method == "scale_floor_fallback"
                ):
                    raise ValueError(f"invalid Source training window: {uid}/{anchor}")
                x_rows.append(
                    np.concatenate(
                        ((context - center) / scale, np.zeros(CONTEXT_LENGTH))
                    )
                )
                y_rows.append((target - center) / scale)
                provenance.append({"series_uid": uid, "anchor": anchor})
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_targets = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (72, 384) or clean_targets.shape != (72, HORIZON):
            raise AssertionError(f"training geometry changed: {dataset_id}")

        old_mask = np.asarray(
            [int(row["anchor"]) in OLD_ANCHORS for row in provenance], dtype=bool
        )
        recent_mask = ~old_mask
        stale_targets = clean_targets.copy()
        stale_targets[old_mask] *= -1.0
        recent_weights = recent_mask.astype(np.float64)
        if int(np.sum(old_mask)) != 36 or int(np.sum(recent_mask)) != 36:
            raise AssertionError("old/recent control is not pattern-balanced")

        x_eval_rows: list[Any] = []
        futures: list[Any] = []
        centers: list[float] = []
        scales: list[float] = []
        eval_uids: list[str] = []
        seasonal_by_uid: dict[str, float] = {}
        for row in eval_rows:
            uid = str(row["series_uid"])
            raw = values[uid]
            context = np.asarray(
                raw[train_stop - CONTEXT_LENGTH : train_stop], dtype=np.float64
            )
            future = np.asarray(raw[slice(*spec["future_bounds"])], dtype=np.float64)
            center, scale, method = _center_scale(context)
            if (
                context.shape != (CONTEXT_LENGTH,)
                or future.shape != (HORIZON,)
                or not np.isfinite(context).all()
                or not np.isfinite(future).all()
                or method == "scale_floor_fallback"
            ):
                raise ValueError(f"invalid Source evaluation window: {uid}")
            try:
                seasonal_by_uid[uid] = seasonal_scale(
                    np.asarray(raw[:train_stop], dtype=np.float64),
                    np.isfinite(raw[:train_stop]),
                    period=period,
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(f"invalid Source sMASE scale: {uid}") from error
            x_eval_rows.append(
                np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
            )
            futures.append(future)
            centers.append(center)
            scales.append(scale)
            eval_uids.append(uid)
        x_eval = np.asarray(x_eval_rows, dtype=np.float64)
        future_array = np.asarray(futures, dtype=np.float64)
        centers_array = np.asarray(centers, dtype=np.float64)
        scales_array = np.asarray(scales, dtype=np.float64)

        def fit_score(targets: Any, sample_weight: Any | None) -> list[float]:
            nonlocal fit_count
            model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
            model.fit(x_train, targets, sample_weight=sample_weight)
            fit_count += 1
            normalized = np.asarray(model.predict(x_eval), dtype=np.float64)
            predicted = normalized * scales_array[:, None] + centers_array[:, None]
            return [
                float(
                    smase(
                        future_array[index],
                        predicted[index],
                        scale=seasonal_by_uid[uid],
                    )
                )
                for index, uid in enumerate(eval_uids)
            ]

        clean_all_scores = fit_score(clean_targets, None)
        stale_all_scores = fit_score(stale_targets, None)
        recent_only_scores = fit_score(clean_targets, recent_weights)
        recurrent_gains = [
            baseline - program
            for baseline, program in zip(clean_all_scores, recent_only_scores)
        ]
        stale_gains = [
            baseline - program
            for baseline, program in zip(stale_all_scores, recent_only_scores)
        ]
        recurrent = _effect_summary(recurrent_gains)
        stale = _effect_summary(stale_gains)
        effect_contrast = float(stale["mean_gain"]) - float(recurrent["mean_gain"])
        stale_pass = (
            float(stale["mean_gain"]) >= MIN_STALE_MEAN_GAIN
            and float(stale["median_gain"]) > 0.0
            and int(stale["positive_uid_count"]) >= MIN_POSITIVE_UIDS
        )
        risk_pass = (
            float(recurrent["mean_gain"]) <= MAX_RECURRENT_MEAN_GAIN
            and effect_contrast >= MIN_EFFECT_CONTRAST
        )
        per_uid = []
        for index, uid in enumerate(eval_uids):
            per_uid.append(
                {
                    "series_uid": uid,
                    "clean_all_smase": clean_all_scores[index],
                    "stale_all_smase": stale_all_scores[index],
                    "recent_only_smase": recent_only_scores[index],
                    "stale_program_gain": stale_gains[index],
                    "recurrent_program_gain": recurrent_gains[index],
                }
            )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "stale_case": {
                    **stale,
                    "headroom_pass": stale_pass,
                    "visible_mapping_observation": _mapping_observation(
                        np, stale_targets, provenance
                    ),
                },
                "recurrent_matched_risk": {
                    **recurrent,
                    "risk_contrast_pass": risk_pass,
                    "visible_mapping_observation": _mapping_observation(
                        np, clean_targets, provenance
                    ),
                },
                "stale_minus_recurrent_mean_gain": effect_contrast,
                "dataset_gate_pass": stale_pass and risk_pass,
                "per_uid": per_uid,
            }
        )

    expected_fits = 3 * len(DATASETS)
    if fit_count != expected_fits:
        raise RuntimeError(f"expected {expected_fits} fits, observed {fit_count}")
    passed = sum(bool(row["dataset_gate_pass"]) for row in dataset_evidence)
    overall_pass = passed == len(DATASETS)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "controlled_context_conditioned_program_effect_calibration",
        "causal_hypothesis": (
            "Excluding a contiguous old training regime helps when its visible "
            "context-to-target mapping has become stale, but is not a universal "
            "improvement when the old mapping remains recurrent."
        ),
        "configuration": {
            "datasets": list(DATASETS),
            "natural_base_series": True,
            "old_anchors": list(OLD_ANCHORS),
            "recent_anchors": list(RECENT_ANCHORS),
            "stale_control": "multiply normalized targets at old anchors by -1",
            "matched_risk": "same natural rows with unchanged old targets",
            "program": "weight old-anchor rows 0 and recent-anchor rows 1",
            "consumer": "Ridge(alpha=1.0, fit_intercept=True, solver=svd)",
            "metric": "per-series sMASE",
            "thresholds": {
                "minimum_stale_mean_gain": MIN_STALE_MEAN_GAIN,
                "maximum_recurrent_mean_gain": MAX_RECURRENT_MEAN_GAIN,
                "minimum_stale_minus_recurrent_mean_gain": MIN_EFFECT_CONTRAST,
                "minimum_positive_uids": MIN_POSITIVE_UIDS,
            },
        },
        "dataset_evidence": dataset_evidence,
        "overall": {
            "dataset_count": len(DATASETS),
            "passing_dataset_count": passed,
            "all_dataset_context_contrast_pass": overall_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "PATTERN_BALANCED_STALE_REGIME_CONTROL_PASS"
            if overall_pass
            else "PATTERN_BALANCED_STALE_REGIME_CONTROL_FAIL"
        ),
        "next_step_if_pass": (
            "Freeze one deployable mapping-disagreement Observation and test whether "
            "it selects exclude versus abstain across held-out natural base datasets."
        ),
        "next_step_if_fail": (
            "Stop the current Ridge stale-regime control; do not build a Witness or "
            "Memory from an unreadable/non-heterogeneous Program effect."
        ),
        "target_query_opened": False,
        "capability_claim": False,
        "formal_transfer": False,
        "claim_limit": (
            "Controlled mapping-reversal calibration on exposed natural Source bases. "
            "It is not natural defect prevalence, promotion, or transfer evidence."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(root)
    output = args.output or root / DEFAULT_REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
