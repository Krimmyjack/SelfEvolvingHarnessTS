"""Test exact temporal-offset rebind recoverability and Ridge headroom."""
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
    J0_REPORT_PATH,
    SEEDS,
    SELECTED_ROW_COUNT,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-source-temporal-offset-rebind-headroom/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_temporal_offset_rebind_headroom_report.json"
)
DATASETS = ("monash:traffic_hourly", "legacy_monash:fred_md")


def _temporal_offset_and_rebind(
    np: Any,
    *,
    clean_targets: Any,
    row_provenance: list[dict[str, object]],
    raw_values: dict[str, Any],
    selected_rows: set[int],
    offset: int,
    train_stop: int,
) -> tuple[Any, Any, list[list[int]]]:
    """Misbind selected targets to raw future slices, then exactly rebind them."""

    clean = np.asarray(clean_targets, dtype=np.float64)
    corrupt = clean.copy()
    rebound = clean.copy()
    source_intervals: list[list[int]] = []
    for row_index in sorted(selected_rows):
        row = row_provenance[row_index]
        uid = str(row["series_uid"])
        anchor = int(row["anchor"])
        center = float(row["center"])
        scale = float(row["scale"])
        offset_start = anchor + offset
        offset_stop = offset_start + HORIZON
        if offset_start < 0 or offset_stop > train_stop:
            raise ValueError("temporal-offset source interval crosses train_stop")
        raw = np.asarray(raw_values[uid], dtype=np.float64)
        wrong = np.asarray(raw[offset_start:offset_stop], dtype=np.float64)
        correct = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
        if (
            wrong.shape != (HORIZON,)
            or correct.shape != (HORIZON,)
            or not np.isfinite(wrong).all()
            or not np.isfinite(correct).all()
        ):
            raise ValueError("invalid temporal-offset raw source slice")
        corrupt[row_index] = (wrong - center) / scale
        rebound[row_index] = (correct - center) / scale
        source_intervals.append([offset_start, offset_stop])
    unselected = np.asarray(
        [index for index in range(clean.shape[0]) if index not in selected_rows],
        dtype=np.int64,
    )
    if unselected.size and not np.array_equal(rebound[unselected], corrupt[unselected]):
        raise AssertionError("exact rebind modified an unselected row")
    return corrupt, rebound, source_intervals


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

    j0_report = _read_object(root / J0_REPORT_PATH)
    if j0_report.get("target_query_opened") is not False:
        raise ValueError("J0 Target/Query boundary is not closed")
    roster = [
        row
        for row in _read_object(root / J0_PLAN_PATH)["roster"]
        if row["dataset_id"] in DATASETS
    ]
    if len(roster) != 40:
        raise ValueError("expected the exposed Traffic+FRED 12+8 rosters")
    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )

    actual_consumer_fit_count = 0
    logical_policy_evaluation_count = 0
    dataset_evidence: list[dict[str, object]] = []
    for dataset_id in DATASETS:
        spec = SPECS[dataset_id]
        train_stop = int(spec["train_stop"])
        period = int(spec["period"])
        offset = period // 2
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
                    np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
                )
                y_rows.append((target - center) / scale)
                provenance.append(
                    {
                        "series_uid": uid,
                        "anchor": anchor,
                        "center": center,
                        "scale": scale,
                    }
                )
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_y = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (72, 384) or clean_y.shape != (72, HORIZON):
            raise AssertionError(f"training geometry changed: {dataset_id}")

        x_eval: list[Any] = []
        raw_future: list[Any] = []
        centers: list[float] = []
        scales: list[float] = []
        eval_uids: list[str] = []
        seasonal_by_uid: dict[str, float] = {}
        for row in eval_rows:
            uid = str(row["series_uid"])
            raw = values[uid]
            future_bounds = tuple(int(value) for value in spec["future_bounds"])
            context = np.asarray(
                raw[train_stop - CONTEXT_LENGTH : train_stop], dtype=np.float64
            )
            future = np.asarray(raw[slice(*future_bounds)], dtype=np.float64)
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
            x_eval.append(
                np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
            )
            raw_future.append(future)
            centers.append(center)
            scales.append(scale)
            eval_uids.append(uid)
        x_eval_array = np.asarray(x_eval, dtype=np.float64)
        raw_future_array = np.asarray(raw_future, dtype=np.float64)
        centers_array = np.asarray(centers, dtype=np.float64)
        scales_array = np.asarray(scales, dtype=np.float64)

        def fit_and_score(targets: Any) -> tuple[Any, Any]:
            nonlocal actual_consumer_fit_count
            model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
            model.fit(x_train, targets)
            actual_consumer_fit_count += 1
            normalized = np.asarray(model.predict(x_eval_array), dtype=np.float64)
            original = normalized * scales_array[:, None] + centers_array[:, None]
            losses = np.asarray(
                [
                    smase(
                        raw_future_array[index],
                        original[index],
                        scale=seasonal_by_uid[uid],
                    )
                    for index, uid in enumerate(eval_uids)
                ],
                dtype=np.float64,
            )
            return normalized, losses

        clean_prediction, clean_losses = fit_and_score(clean_y)
        seed_evidence: list[dict[str, object]] = []
        for seed in SEEDS:
            selected = set(
                int(value)
                for value in np.random.default_rng(seed).choice(
                    len(clean_y), size=SELECTED_ROW_COUNT, replace=False
                )
            )
            offset_y, rebound_y, source_intervals = _temporal_offset_and_rebind(
                np,
                clean_targets=clean_y,
                row_provenance=provenance,
                raw_values=values,
                selected_rows=selected,
                offset=offset,
                train_stop=train_stop,
            )
            selected_indices = np.asarray(sorted(selected), dtype=np.int64)
            unselected_indices = np.asarray(
                [index for index in range(len(clean_y)) if index not in selected],
                dtype=np.int64,
            )
            p0_selected_exact = np.array_equal(
                rebound_y[selected_indices], clean_y[selected_indices]
            )
            p0_full_exact = np.array_equal(rebound_y, clean_y)
            collateral_count = int(
                np.count_nonzero(
                    rebound_y[unselected_indices] != offset_y[unselected_indices]
                )
            )
            if not p0_selected_exact or not p0_full_exact or collateral_count != 0:
                raise AssertionError(f"P0 exact rebind failed: {dataset_id}/{seed}")
            if max(stop for _, stop in source_intervals) > train_stop:
                raise AssertionError("offset source interval crossed train_stop")

            _, offset_losses = fit_and_score(offset_y)
            rebound_prediction = clean_prediction
            rebound_losses = clean_losses
            logical_policy_evaluation_count += 3
            per_uid_gain = offset_losses - rebound_losses
            seed_evidence.append(
                {
                    "seed": seed,
                    "selected_row_indices": sorted(selected),
                    "offset_source_intervals": source_intervals,
                    "p0_rebound_equals_clean": p0_full_exact,
                    "p0_unselected_collateral_change_count": collateral_count,
                    "clean_mean_smase": float(np.mean(clean_losses)),
                    "temporally_offset_mean_smase": float(np.mean(offset_losses)),
                    "exact_rebind_mean_smase": float(np.mean(rebound_losses)),
                    "offset_minus_rebind_mean_gain": float(np.mean(per_uid_gain)),
                    "positive_evaluation_uid_count": int(
                        np.count_nonzero(per_uid_gain > 0.0)
                    ),
                    "rebind_prediction_reuses_clean_prediction": (
                        rebound_prediction is clean_prediction
                    ),
                }
            )

        gains = [
            float(row["offset_minus_rebind_mean_gain"]) for row in seed_evidence
        ]
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "period": period,
                "offset": offset,
                "evaluation_uids": eval_uids,
                "seed_evidence": seed_evidence,
                "summary": {
                    "p0_all_seeds_exact": all(
                        bool(row["p0_rebound_equals_clean"]) for row in seed_evidence
                    ),
                    "p0_total_unselected_collateral_change_count": sum(
                        int(row["p0_unselected_collateral_change_count"])
                        for row in seed_evidence
                    ),
                    "mean_clean_smase": statistics.fmean(
                        float(row["clean_mean_smase"]) for row in seed_evidence
                    ),
                    "mean_temporally_offset_smase": statistics.fmean(
                        float(row["temporally_offset_mean_smase"])
                        for row in seed_evidence
                    ),
                    "mean_exact_rebind_smase": statistics.fmean(
                        float(row["exact_rebind_mean_smase"])
                        for row in seed_evidence
                    ),
                    "mean_offset_minus_rebind_gain": statistics.fmean(gains),
                    "positive_seed_count": sum(value > 0.0 for value in gains),
                    "mean_positive_evaluation_uid_count": statistics.fmean(
                        int(row["positive_evaluation_uid_count"])
                        for row in seed_evidence
                    ),
                },
            }
        )

    p0_pass = all(
        bool(row["summary"]["p0_all_seeds_exact"])
        and int(row["summary"]["p0_total_unselected_collateral_change_count"]) == 0
        for row in dataset_evidence
    )
    p1_pass = all(
        float(row["summary"]["mean_offset_minus_rebind_gain"]) > 0.0
        and int(row["summary"]["positive_seed_count"]) >= 2
        for row in dataset_evidence
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_source_temporal_offset_rebind_p0_p1",
        "configuration": {
            "datasets": list(DATASETS),
            "anchors": list(ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "seeds": list(SEEDS),
            "selected_rows_per_seed": SELECTED_ROW_COUNT,
            "offset_rule": "period // 2",
            "offset_source": (
                "raw same-series [anchor+k,anchor+k+48), standardized with original "
                "row context center/scale; never clean_y roll"
            ),
            "program": "exact_rebind from original row series_uid+anchor provenance",
            "consumer": "Ridge(alpha=1.0, fit_intercept=True, solver=svd)",
            "metric": "per-series sMASE",
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
        },
        "roster": roster,
        "dataset_evidence": dataset_evidence,
        "overall": {
            "p0_recoverability_pass": p0_pass,
            "p1_consumer_headroom_pass": p1_pass,
            "verdict": (
                "TEMPORAL_OFFSET_REBIND_P0_P1_PASS"
                if p0_pass and p1_pass
                else (
                    "TEMPORAL_OFFSET_REBIND_P0_PASS_P1_FAIL"
                    if p0_pass
                    else "TEMPORAL_OFFSET_REBIND_P0_FAIL"
                )
            ),
        },
        "compute_accounting": {
            "logical_policy_evaluation_count": logical_policy_evaluation_count,
            "actual_consumer_fit_count": actual_consumer_fit_count,
            "clean_fit_reused_across_seeds": True,
            "exact_rebind_prediction_reuses_clean_prediction": True,
        },
        "capability_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Exposed Source P0/P1 Program recoverability and headroom only; exact "
            "provenance rebind is not yet an Adaptive Capability."
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
    print(report["overall"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
