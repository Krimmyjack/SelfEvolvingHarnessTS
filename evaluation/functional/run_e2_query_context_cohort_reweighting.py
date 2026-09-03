"""Measure headroom of one query-context-aligned cohort reweighting Program.

This development-only Source experiment reuses the four exposed E2.2 datasets and
their 12-train/8-eval rosters.  A fixed, outcome-free time-series fingerprint binds
one rank-based training-window weighting Program to the visible evaluation-context
centroid.  Each dataset receives exactly one uniform and one weighted Ridge fit.

No Capability is promoted and no Target/UCI data is read.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
    _fresh_roster,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    ANCHORS,
    CONTEXT_LENGTH,
    FRESH_SPECS,
    HORIZON,
    J0_PLAN_PATH,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-query-context-cohort-reweighting/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_query_context_cohort_reweighting_report.json"
)
EXPECTED_DATASET_COUNT = 4
EXPECTED_FIT_COUNT = 2 * EXPECTED_DATASET_COUNT
MIN_POSITIVE_UIDS = 6


def _safe_correlation(np: Any, left: Any, right: Any) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if (
        left.size < 2
        or right.size != left.size
        or float(np.std(left)) <= 1e-12
        or float(np.std(right)) <= 1e-12
    ):
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _context_fingerprint(np: Any, standardized_context: Any, period: int) -> Any:
    """Return the frozen six-field visible-context fingerprint."""

    context = np.asarray(standardized_context, dtype=np.float64)
    if context.shape != (CONTEXT_LENGTH,) or not np.isfinite(context).all():
        raise ValueError("fingerprint requires one finite length-192 context")
    if period < 1 or period >= context.size:
        raise ValueError("known period must be inside the visible context")

    time = np.arange(context.size, dtype=np.float64)
    centered_time = time - float(np.mean(time))
    centered_context = context - float(np.mean(context))
    slope = float(
        np.dot(centered_time, centered_context) / np.dot(centered_time, centered_time)
    )
    seasonal_residual = context[period:] - context[:-period]
    fingerprint = np.asarray(
        [
            slope,
            float(np.std(np.diff(context))),
            float(np.mean(context[-period:]) - np.mean(context)),
            _safe_correlation(np, context[1:], context[:-1]),
            _safe_correlation(np, context[period:], context[:-period]),
            float(np.mean(np.abs(seasonal_residual))),
        ],
        dtype=np.float64,
    )
    if fingerprint.shape != (6,) or not np.isfinite(fingerprint).all():
        raise RuntimeError("invalid context fingerprint")
    return fingerprint


def _rank_weights(np: Any, train_fingerprints: Any, query_centroid: Any) -> tuple[Any, Any, Any]:
    """Bind the sole Program: nearest rows receive larger positive mean-one weights.

    Rank is zero-based.  Thus ``2 * (N - rank) / (N + 1)`` is strictly positive
    even for the farthest row, and its arithmetic mean is exactly one.
    """

    train = np.asarray(train_fingerprints, dtype=np.float64)
    query = np.asarray(query_centroid, dtype=np.float64)
    if train.ndim != 2 or train.shape[0] < 1 or query.shape != (train.shape[1],):
        raise ValueError("rank weighting requires an N x D train matrix and D-vector")
    if not np.isfinite(train).all() or not np.isfinite(query).all():
        raise ValueError("rank weighting inputs must be finite")

    distances = np.linalg.norm(train - query[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    ranks = np.empty(train.shape[0], dtype=np.int64)
    ranks[order] = np.arange(train.shape[0], dtype=np.int64)
    count = train.shape[0]
    weights = 2.0 * (count - ranks.astype(np.float64)) / (count + 1.0)
    if not np.all(weights > 0.0) or not np.isclose(np.mean(weights), 1.0):
        raise RuntimeError("rank Program violated positive mean-one weight contract")
    return weights, distances, ranks


def run(root: Path) -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import Ridge

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
        _center_scale,
    )

    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    roster = list(_read_object(root / J0_PLAN_PATH)["roster"])
    for dataset_id, spec in FRESH_SPECS.items():
        roster.extend(
            _fresh_roster(
                np,
                root=root,
                registry_rows=registry_rows,
                dataset_id=dataset_id,
                spec=spec,
            )
        )
    if any(str(row["dataset_id"]).startswith("uci") for row in roster):
        raise ValueError("UCI is forbidden in this Source experiment")

    specs = {**SPECS, **FRESH_SPECS}
    if len(specs) != EXPECTED_DATASET_COUNT or len(roster) != 20 * EXPECTED_DATASET_COUNT:
        raise ValueError("expected the four exposed 12-train/8-eval Source rosters")
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )

    fit_count = 0
    dataset_evidence: list[dict[str, object]] = []
    for dataset_id, spec in specs.items():
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
            raise ValueError(f"exposed roster geometry changed: {dataset_id}")
        period = int(spec["period"])

        x_train_rows: list[Any] = []
        y_train_rows: list[Any] = []
        train_fingerprints: list[Any] = []
        train_keys: list[dict[str, object]] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                raw = values[uid]
                context = np.asarray(
                    raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64
                )
                target = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
                if context.shape != (CONTEXT_LENGTH,) or target.shape != (HORIZON,):
                    raise ValueError(f"invalid training window: {uid}/{anchor}")
                if not np.isfinite(context).all() or not np.isfinite(target).all():
                    raise ValueError(f"non-finite training window: {uid}/{anchor}")
                center, scale, method = _center_scale(context)
                if method == "scale_floor_fallback":
                    raise ValueError(f"scale floor reached: {uid}/{anchor}")
                standardized = (context - center) / scale
                x_train_rows.append(
                    np.concatenate((standardized, np.zeros(CONTEXT_LENGTH)))
                )
                y_train_rows.append((target - center) / scale)
                train_fingerprints.append(_context_fingerprint(np, standardized, period))
                train_keys.append({"series_uid": uid, "anchor": anchor})

        x_train = np.asarray(x_train_rows, dtype=np.float64)
        y_train = np.asarray(y_train_rows, dtype=np.float64)
        raw_train_fingerprints = np.asarray(train_fingerprints, dtype=np.float64)
        if x_train.shape != (72, 384) or y_train.shape != (72, HORIZON):
            raise AssertionError(f"unexpected training geometry: {dataset_id}")

        x_eval_rows: list[Any] = []
        eval_fingerprints: list[Any] = []
        raw_future: list[Any] = []
        eval_uids: list[str] = []
        eval_centers: list[float] = []
        eval_scales: list[float] = []
        seasonal_by_uid: dict[str, float] = {}
        for row in eval_rows:
            uid = str(row["series_uid"])
            raw = values[uid]
            train_stop = int(spec["train_stop"])
            context = np.asarray(
                raw[train_stop - CONTEXT_LENGTH : train_stop], dtype=np.float64
            )
            future = np.asarray(raw[slice(*spec["future_bounds"])], dtype=np.float64)
            if context.shape != (CONTEXT_LENGTH,) or future.shape != (HORIZON,):
                raise ValueError(f"invalid evaluation window: {uid}")
            if not np.isfinite(context).all() or not np.isfinite(future).all():
                raise ValueError(f"non-finite evaluation window: {uid}")
            center, scale, method = _center_scale(context)
            if method == "scale_floor_fallback":
                raise ValueError(f"evaluation scale floor reached: {uid}")
            standardized = (context - center) / scale
            x_eval_rows.append(
                np.concatenate((standardized, np.zeros(CONTEXT_LENGTH)))
            )
            eval_fingerprints.append(_context_fingerprint(np, standardized, period))
            raw_future.append(future)
            eval_uids.append(uid)
            eval_centers.append(center)
            eval_scales.append(scale)
            try:
                seasonal_by_uid[uid] = seasonal_scale(
                    np.asarray(raw[:train_stop], dtype=np.float64),
                    np.isfinite(raw[:train_stop]),
                    period=period,
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(f"invalid evaluation sMASE scale: {uid}") from error

        fingerprint_mean = np.mean(raw_train_fingerprints, axis=0)
        fingerprint_std = np.std(raw_train_fingerprints, axis=0)
        fingerprint_scale = np.where(fingerprint_std > 1e-12, fingerprint_std, 1.0)
        standardized_train = (
            raw_train_fingerprints - fingerprint_mean[None, :]
        ) / fingerprint_scale[None, :]
        standardized_eval = (
            np.asarray(eval_fingerprints, dtype=np.float64) - fingerprint_mean[None, :]
        ) / fingerprint_scale[None, :]
        query_centroid = np.mean(standardized_eval, axis=0)
        weights, distances, ranks = _rank_weights(
            np, standardized_train, query_centroid
        )
        train_query_rms_shift = float(
            np.sqrt(np.mean((query_centroid - np.mean(standardized_train, axis=0)) ** 2))
        )

        x_eval = np.asarray(x_eval_rows, dtype=np.float64)
        future_array = np.asarray(raw_future, dtype=np.float64)
        centers = np.asarray(eval_centers, dtype=np.float64)
        scales = np.asarray(eval_scales, dtype=np.float64)

        def fit_score(sample_weight: Any | None) -> list[float]:
            nonlocal fit_count
            model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
            model.fit(x_train, y_train, sample_weight=sample_weight)
            fit_count += 1
            normalized = np.asarray(model.predict(x_eval), dtype=np.float64)
            if normalized.shape != (8, HORIZON) or not np.isfinite(normalized).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}")
            predicted = normalized * scales[:, None] + centers[:, None]
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

        baseline_scores = fit_score(None)
        weighted_scores = fit_score(weights)
        gains = [base - weighted for base, weighted in zip(baseline_scores, weighted_scores)]
        per_uid = [
            {
                "series_uid": uid,
                "baseline_smase": baseline_scores[index],
                "weighted_smase": weighted_scores[index],
                "gain": gains[index],
            }
            for index, uid in enumerate(eval_uids)
        ]
        mean_gain = statistics.fmean(gains)
        median_gain = statistics.median(gains)
        positive_count = sum(gain > 0.0 for gain in gains)
        headroom = mean_gain > 0.0 and median_gain > 0.0 and positive_count >= MIN_POSITIVE_UIDS
        weight_rows = [
            {
                **train_keys[index],
                "distance": float(distances[index]),
                "rank_zero_based": int(ranks[index]),
                "weight": float(weights[index]),
            }
            for index in range(len(train_keys))
        ]
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "per_uid": per_uid,
                "summary": {
                    "mean_gain": mean_gain,
                    "median_gain": median_gain,
                    "positive_uid_count": positive_count,
                    "headroom_pass": headroom,
                    "weight_effective_sample_size": float(
                        np.sum(weights) ** 2 / np.sum(weights**2)
                    ),
                    "weight_mean": float(np.mean(weights)),
                    "weight_min": float(np.min(weights)),
                    "weight_max": float(np.max(weights)),
                    "train_query_rms_shift": train_query_rms_shift,
                },
                "program_binding": {
                    "query_centroid": query_centroid.tolist(),
                    "train_window_weights": weight_rows,
                },
            }
        )

    if fit_count != EXPECTED_FIT_COUNT:
        raise RuntimeError(f"expected {EXPECTED_FIT_COUNT} fits, observed {fit_count}")
    passed_datasets = sum(
        bool(row["summary"]["headroom_pass"]) for row in dataset_evidence
    )
    overall_pass = passed_datasets >= 2
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_source_cohort_program_headroom",
        "causal_hypothesis": (
            "A fixed query-context-aligned rank reweighting Program improves complete "
            "evaluation-cohort Ridge utility relative to uniform training."
        ),
        "configuration": {
            "datasets": list(specs),
            "roster": "same exposed E2.2 12 train + 8 eval series per dataset",
            "anchors": list(ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "fingerprint_fields": [
                "linear_trend_slope",
                "first_difference_std",
                "recent_period_mean_minus_global_mean",
                "lag_1_correlation",
                "known_period_correlation",
                "mean_absolute_seasonal_residual",
            ],
            "fingerprint_standardization": "72 train fingerprints mean/std per dataset",
            "query_binding": "centroid of 8 visible evaluation-context fingerprints",
            "program": "zero-based ascending L2 rank weight = 2*(N-rank)/(N+1)",
            "consumer": "Ridge(alpha=1.0, fit_intercept=True, solver=svd)",
            "metric": "per-series sMASE",
            "baseline": "uniform training weights",
        },
        "dataset_evidence": dataset_evidence,
        "overall": {
            "headroom_dataset_count": passed_datasets,
            "required_headroom_dataset_count": 2,
            "headroom_pass": overall_pass,
        },
        "verdict": (
            "QUERY_CONTEXT_COHORT_REWEIGHTING_HEADROOM_PASS"
            if overall_pass
            else "QUERY_CONTEXT_COHORT_REWEIGHTING_HEADROOM_FAIL"
        ),
        "consumer_fit_count": fit_count,
        "capability_claim": False,
        "promotion_eligible": False,
        "memory_claim": False,
        "formal_transfer": False,
        "information_wall": {
            "fingerprint_uses_visible_context_only": True,
            "source_eval_future_read_for_judge_and_frozen_finite_admission": True,
            "source_eval_future_values_or_outcomes_enter_weights": False,
            "outcome_enters_fingerprint_or_weights": False,
            "opaque_dataset_identity_enters_fingerprint_or_weights": False,
            "known_period_semantics_enters_fingerprint": True,
            "uci_values_read": False,
            "target_values_read": False,
            "uci_target_query_context_read": False,
            "uci_target_query_outcome_read": False,
            "target_query_opened": False,
        },
        "target_query_opened": False,
        "claim_limit": (
            "Development-only exposed Source Program-headroom test. A pass is not "
            "Capability Promotion, Memory benefit, Target adaptation, or Transfer."
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
