"""Diagnose per-intervention credit for the exposed flatline masking family.

This is a development-only bridge between a local time-series observation and the
batch-level Ridge judge.  Each intervention unit is one observed training-row flatline
interval.  We mask exactly one unit at a time, measure its marginal validation effect,
then use one fixed validation fold to choose mask/abstain and evaluate the combined
policy on the other fold.  Full-policy retraining remains the final judge.

No capability is promoted and no Target/UCI data is read by this runner.
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
    FRESH_SPECS,
    HORIZON,
    J0_PLAN_PATH,
    J0_REPORT_PATH,
    SEEDS,
    SELECTED_ROW_COUNT,
    SPECS,
    TARGET_BLOCK,
    _apply_stuck_value_censoring,
    _censor_flatline_interval_weights,
    _read_object,
    _selected_indices,
)


SCHEMA_VERSION = "e2-flatline-actionability-credit/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_flatline_actionability_credit_report.json"
)


def _alternating_folds(size: int) -> dict[str, tuple[int, ...]]:
    """Return two deterministic, equally sized cross-fitting folds."""

    if size < 2 or size % 2:
        raise ValueError("cross-fitting requires an even number of evaluation series")
    return {
        "fold_a": tuple(range(0, size, 2)),
        "fold_b": tuple(range(1, size, 2)),
    }


def _positive_credit_rows(
    unit_rows: list[dict[str, object]], credit_key: str
) -> tuple[int, ...]:
    """Compile the minimal actionability rule: mask iff support credit is positive."""

    return tuple(
        sorted(
            int(row["row_index"])
            for row in unit_rows
            if float(row[credit_key]) > 0.0
        )
    )


def _mean(values: Any, indices: tuple[int, ...]) -> float:
    return statistics.fmean(float(values[index]) for index in indices)


def _context_card(
    np: Any,
    *,
    row_key: tuple[str, int, int],
    context: Any,
    corrupt_target: Any,
    period: int,
) -> dict[str, object]:
    """Small public context card; none of these fields uses clean hidden targets."""

    start, stop = TARGET_BLOCK
    visible = np.asarray(context, dtype=np.float64)
    target = np.asarray(corrupt_target, dtype=np.float64)
    flatline_value = float(np.median(target[start:stop]))
    lagged = visible[:-period]
    advanced = visible[period:]
    lag_corr = (
        float(np.corrcoef(lagged, advanced)[0, 1])
        if lagged.size > 1
        and float(np.std(lagged)) > 0.0
        and float(np.std(advanced)) > 0.0
        else None
    )
    return {
        "series_uid": row_key[0],
        "anchor": row_key[1],
        "interval": list(TARGET_BLOCK),
        "flatline_length": stop - start,
        "known_sampling_period": period,
        "flatline_to_period_ratio": (stop - start) / period,
        "flatline_value_standardized": flatline_value,
        "left_boundary_jump_abs": abs(flatline_value - float(target[start - 1])),
        "right_boundary_jump_abs": abs(float(target[stop]) - flatline_value),
        "context_last_minus_median": float(visible[-1] - np.median(visible)),
        "context_lag_correlation": lag_corr,
        "role": "diagnostic_context_only_not_used_for_selection",
    }


def _fresh_roster(
    np: Any,
    *,
    root: Path,
    registry_rows: list[Any],
    dataset_id: str,
    spec: dict[str, object],
    excluded_uids: frozenset[str] = frozenset(),
) -> list[dict[str, object]]:
    """Reproduce the already exposed frozen-replay roster without outcome selection."""

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values

    required_stop = max(max(ANCHORS) + HORIZON, int(spec["future_bounds"][1]))
    candidates = sorted(
        (
            row
            for row in registry_rows
            if row.dataset_id == dataset_id
            and row.series_uid not in excluded_uids
            and int(row.length) >= required_stop
        ),
        key=lambda row: row.series_uid,
    )
    candidate_values = _load_values(
        candidates, root / "data/benchmark_v0_2/clean_base"
    )
    eligible = [
        row
        for row in candidates
        if np.isfinite(
            np.asarray(
                candidate_values[row.series_uid][48:required_stop], dtype=np.float64
            )
        ).all()
    ]
    if len(eligible) < 20:
        raise ValueError(f"insufficient exposed Source candidates: {dataset_id}")
    return [
        {
            "dataset_id": dataset_id,
            "series_uid": row.series_uid,
            "cohort": "train" if index < 12 else "eval",
        }
        for index, row in enumerate(eligible[:20])
    ]


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
    j0_plan = _read_object(root / J0_PLAN_PATH)
    j0_report = _read_object(root / J0_REPORT_PATH)
    if j0_report.get("target_query_opened") is not False:
        raise ValueError("J0 Target/Query boundary is not closed")

    roster = list(j0_plan["roster"])
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
        raise ValueError("UCI is forbidden in this Source diagnostic")

    selected_records = [records[str(row["series_uid"])] for row in roster]
    values = _load_values(selected_records, root / "data/benchmark_v0_2/clean_base")
    specs = {**SPECS, **FRESH_SPECS}
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
            raise ValueError(f"expected exposed 12+8 roster: {dataset_id}")

        x_rows: list[Any] = []
        y_rows: list[Any] = []
        row_keys: list[tuple[str, int, int]] = []
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
                    raise ValueError(f"invalid training window: {uid}/{anchor}")
                x_rows.append(
                    np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
                )
                y_rows.append((target - center) / scale)
                row_keys.append((uid, anchor, HORIZON))
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_y = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (72, 384) or clean_y.shape != (72, HORIZON):
            raise AssertionError(f"unexpected training geometry: {dataset_id}")
        row_index = {key: index for index, key in enumerate(row_keys)}

        x_eval: list[Any] = []
        raw_future: list[Any] = []
        eval_uids: list[str] = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal_by_uid: dict[str, float] = {}
        for row in eval_rows:
            uid = str(row["series_uid"])
            raw = values[uid]
            train_stop = int(spec["train_stop"])
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
                raise ValueError(f"invalid evaluation window: {uid}")
            try:
                seasonal_by_uid[uid] = seasonal_scale(
                    np.asarray(raw[:train_stop], dtype=np.float64),
                    np.isfinite(raw[:train_stop]),
                    period=int(spec["period"]),
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(f"invalid evaluation sMASE scale: {uid}") from error
            x_eval.append(
                np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
            )
            raw_future.append(future)
            eval_uids.append(uid)
            centers.append(center)
            scales.append(scale)

        x_eval_array = np.asarray(x_eval, dtype=np.float64)
        raw_future_array = np.asarray(raw_future, dtype=np.float64)
        centers_array = np.asarray(centers, dtype=np.float64)
        scales_array = np.asarray(scales, dtype=np.float64)
        folds = _alternating_folds(len(eval_uids))

        def score_predictions(normalized: Any) -> Any:
            prediction = np.asarray(normalized, dtype=np.float64)
            if prediction.shape != (8, HORIZON) or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}")
            original = prediction * scales_array[:, None] + centers_array[:, None]
            return np.asarray(
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

        def fit_full(targets: Any) -> tuple[Any, Any]:
            nonlocal fit_count
            model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
            model.fit(x_train, targets)
            fit_count += 1
            prediction = np.asarray(model.predict(x_eval_array), dtype=np.float64)
            return prediction, score_predictions(prediction)

        def fit_interval_mask(
            targets: Any,
            masked_rows: tuple[int, ...],
            baseline_prediction: Any,
        ) -> tuple[Any, Any]:
            nonlocal fit_count
            if not masked_rows:
                copied = np.asarray(baseline_prediction, dtype=np.float64).copy()
                return copied, score_predictions(copied)
            weights = np.ones(len(row_keys), dtype=np.float64)
            weights[list(masked_rows)] = 0.0
            start, stop = TARGET_BLOCK
            model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
            model.fit(x_train, targets[:, start:stop], sample_weight=weights)
            prediction = np.asarray(baseline_prediction, dtype=np.float64).copy()
            prediction[:, start:stop] = model.predict(x_eval_array)
            fit_count += 1
            return prediction, score_predictions(prediction)

        _, clean_losses = fit_full(clean_y)
        seed_evidence: list[dict[str, object]] = []
        for seed in SEEDS:
            selected_truth = (
                set(
                    int(value)
                    for value in np.random.default_rng(seed).choice(
                        len(row_keys), size=SELECTED_ROW_COUNT, replace=False
                    )
                )
                if dataset_id in FRESH_SPECS
                else _selected_indices(j0_report, dataset_id, seed, row_index)
            )
            corrupt, _, _ = _apply_stuck_value_censoring(
                np,
                clean_y,
                x_train[:, :CONTEXT_LENGTH],
                selected_truth,
                TARGET_BLOCK,
            )
            corrupt_prediction, corrupt_losses = fit_full(corrupt)
            observed_weights, observations = _censor_flatline_interval_weights(np, corrupt)
            candidate_rows = tuple(
                index
                for index, observation in enumerate(observations)
                if observation["status"] == "ACTIVATE"
            )
            if len(candidate_rows) != SELECTED_ROW_COUNT or set(candidate_rows) != selected_truth:
                raise AssertionError(
                    f"frozen flatline observation changed on exposed data: {dataset_id}/{seed}"
                )
            if any(
                observation["predicted_interval"] != list(TARGET_BLOCK)
                for observation in observations
                if observation["status"] == "ACTIVATE"
            ):
                raise AssertionError("actionability units must bind the frozen interval")
            if int(np.count_nonzero(observed_weights == 0.0)) != (
                SELECTED_ROW_COUNT * (TARGET_BLOCK[1] - TARGET_BLOCK[0])
            ):
                raise AssertionError("flatline compiler geometry changed")

            policy_cache: dict[tuple[int, ...], Any] = {(): corrupt_losses}
            _, all_mask_losses = fit_interval_mask(
                corrupt, candidate_rows, corrupt_prediction
            )
            policy_cache[candidate_rows] = all_mask_losses
            unit_rows: list[dict[str, object]] = []
            for candidate in candidate_rows:
                _, unit_losses = fit_interval_mask(
                    corrupt, (candidate,), corrupt_prediction
                )
                policy_cache[(candidate,)] = unit_losses
                fold_a_credit = _mean(corrupt_losses - unit_losses, folds["fold_a"])
                fold_b_credit = _mean(corrupt_losses - unit_losses, folds["fold_b"])
                unit_rows.append(
                    {
                        "row_index": candidate,
                        "row_key": list(row_keys[candidate]),
                        "context": _context_card(
                            np,
                            row_key=row_keys[candidate],
                            context=x_train[candidate, :CONTEXT_LENGTH],
                            corrupt_target=corrupt[candidate],
                            period=int(spec["period"]),
                        ),
                        "fold_a_marginal_credit": fold_a_credit,
                        "fold_b_marginal_credit": fold_b_credit,
                        "credit_sign_agreement": (fold_a_credit > 0.0)
                        == (fold_b_credit > 0.0),
                        "credit_semantics": (
                            "validation sMASE(keep all) - validation sMASE(mask only this "
                            "row x interval); positive favors masking"
                        ),
                    }
                )

            fold_evidence: list[dict[str, object]] = []
            fold_pairs = (
                ("a_to_b", "fold_a", "fold_b", "fold_a_marginal_credit"),
                ("b_to_a", "fold_b", "fold_a", "fold_b_marginal_credit"),
            )
            for name, support_name, holdout_name, credit_key in fold_pairs:
                support_indices = folds[support_name]
                holdout_indices = folds[holdout_name]
                chosen = _positive_credit_rows(unit_rows, credit_key)
                if chosen not in policy_cache:
                    _, policy_cache[chosen] = fit_interval_mask(
                        corrupt, chosen, corrupt_prediction
                    )
                guided_losses = policy_cache[chosen]
                selected_unit_rows = [
                    row for row in unit_rows if int(row["row_index"]) in chosen
                ]
                holdout_key = (
                    "fold_b_marginal_credit"
                    if support_name == "fold_a"
                    else "fold_a_marginal_credit"
                )
                fold_evidence.append(
                    {
                        "direction": name,
                        "support_uid_indices": list(support_indices),
                        "holdout_uid_indices": list(holdout_indices),
                        "selected_row_indices": list(chosen),
                        "selected_count": len(chosen),
                        "abstained_count": len(candidate_rows) - len(chosen),
                        "selected_unit_holdout_positive_fraction": (
                            sum(float(row[holdout_key]) > 0.0 for row in selected_unit_rows)
                            / len(selected_unit_rows)
                            if selected_unit_rows
                            else None
                        ),
                        "credit_guided_support_gain": _mean(
                            corrupt_losses - guided_losses, support_indices
                        ),
                        "credit_guided_holdout_gain": _mean(
                            corrupt_losses - guided_losses, holdout_indices
                        ),
                        "always_mask_support_gain": _mean(
                            corrupt_losses - all_mask_losses, support_indices
                        ),
                        "always_mask_holdout_gain": _mean(
                            corrupt_losses - all_mask_losses, holdout_indices
                        ),
                        "summed_selected_unit_support_credit": sum(
                            float(row[credit_key]) for row in selected_unit_rows
                        ),
                        "summed_selected_unit_holdout_credit": sum(
                            float(row[holdout_key]) for row in selected_unit_rows
                        ),
                    }
                )

            seed_evidence.append(
                {
                    "seed": seed,
                    "candidate_unit_count": len(candidate_rows),
                    "mean_corruption_degradation": float(
                        np.mean(corrupt_losses - clean_losses)
                    ),
                    "always_mask_full_eval_gain": float(
                        np.mean(corrupt_losses - all_mask_losses)
                    ),
                    "unit_credit": unit_rows,
                    "crossfit_policy": fold_evidence,
                }
            )

        all_units = [
            row for seed_row in seed_evidence for row in seed_row["unit_credit"]
        ]
        all_folds = [
            row for seed_row in seed_evidence for row in seed_row["crossfit_policy"]
        ]
        fold_a_credit = np.asarray(
            [float(row["fold_a_marginal_credit"]) for row in all_units],
            dtype=np.float64,
        )
        fold_b_credit = np.asarray(
            [float(row["fold_b_marginal_credit"]) for row in all_units],
            dtype=np.float64,
        )
        correlation = (
            float(np.corrcoef(fold_a_credit, fold_b_credit)[0, 1])
            if float(np.std(fold_a_credit)) > 0.0
            and float(np.std(fold_b_credit)) > 0.0
            else None
        )
        guided_gain = statistics.fmean(
            float(row["credit_guided_holdout_gain"]) for row in all_folds
        )
        all_mask_gain = statistics.fmean(
            float(row["always_mask_holdout_gain"]) for row in all_folds
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "evaluation_uids": eval_uids,
                "folds": {key: list(value) for key, value in folds.items()},
                "seed_evidence": seed_evidence,
                "summary": {
                    "unit_count": len(all_units),
                    "unit_credit_sign_agreement_rate": (
                        sum(bool(row["credit_sign_agreement"]) for row in all_units)
                        / len(all_units)
                    ),
                    "fold_credit_pearson": correlation,
                    "mean_selected_fraction": statistics.fmean(
                        int(row["selected_count"]) / SELECTED_ROW_COUNT
                        for row in all_folds
                    ),
                    "mean_credit_guided_holdout_gain": guided_gain,
                    "mean_always_mask_holdout_gain": all_mask_gain,
                    "credit_guided_harmful_fold_count": sum(
                        float(row["credit_guided_holdout_gain"]) < 0.0
                        for row in all_folds
                    ),
                    "always_mask_harmful_fold_count": sum(
                        float(row["always_mask_holdout_gain"]) < 0.0
                        for row in all_folds
                    ),
                },
            }
        )

    summaries = [row["summary"] for row in dataset_evidence]
    macro_guided_gain = statistics.fmean(
        float(row["mean_credit_guided_holdout_gain"]) for row in summaries
    )
    macro_all_mask_gain = statistics.fmean(
        float(row["mean_always_mask_holdout_gain"]) for row in summaries
    )
    macro_sign_agreement = statistics.fmean(
        float(row["unit_credit_sign_agreement_rate"]) for row in summaries
    )
    guided_harmful_datasets = sum(
        float(row["mean_credit_guided_holdout_gain"]) < 0.0 for row in summaries
    )
    all_mask_harmful_datasets = sum(
        float(row["mean_always_mask_holdout_gain"]) < 0.0 for row in summaries
    )
    actionability_signal = (
        macro_guided_gain > 0.0
        and macro_sign_agreement > 0.5
        and guided_harmful_datasets < all_mask_harmful_datasets
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_source_actionability_credit_diagnostic",
        "causal_hypothesis": (
            "A one-block marginal intervention credit measured on one validation fold "
            "predicts whether masking that block is useful on a disjoint validation fold; "
            "the compiled mask/abstain policy should reduce dataset-level harm relative "
            "to always masking every detected flatline."
        ),
        "intervention_unit": "one training row x observed flatline interval",
        "credit_definition": (
            "sMASE(keep all observed flatlines) - sMASE(mask only this unit), with all "
            "other training rows and intervals fixed"
        ),
        "configuration": {
            "datasets": list(specs),
            "consumer": "Ridge(alpha=1.0, fit_intercept=True, solver=svd)",
            "metric": "per-series sMASE, equal mean within validation fold",
            "seeds": list(SEEDS),
            "candidate_units_per_seed": SELECTED_ROW_COUNT,
            "crossfit": "alternating 4/4 evaluation-series split, both directions",
            "selection_rule": "mask unit iff support-fold marginal credit > 0; else abstain",
            "baselines": ["always_keep", "always_mask_all_detected_flatlines"],
        },
        "dataset_evidence": dataset_evidence,
        "overall": {
            "dataset_macro_credit_guided_holdout_gain": macro_guided_gain,
            "dataset_macro_always_mask_holdout_gain": macro_all_mask_gain,
            "dataset_macro_unit_credit_sign_agreement_rate": macro_sign_agreement,
            "credit_guided_harmful_dataset_count": guided_harmful_datasets,
            "always_mask_harmful_dataset_count": all_mask_harmful_datasets,
            "credit_guided_beats_always_mask_macro": macro_guided_gain > macro_all_mask_gain,
        },
        "verdict": (
            "ACTIONABILITY_CREDIT_SIGNAL_PRESENT"
            if actionability_signal
            else "ACTIONABILITY_CREDIT_NOT_STABLE"
        ),
        "next_step_if_present": (
            "Use these signed unit-level outcomes to test whether TS Context can predict "
            "mask/abstain across datasets; do not write Memory yet."
        ),
        "next_step_if_absent": (
            "Escalate once to small matched-group credit; if that also fails, close the "
            "flatline family under the fixed Ridge protocol."
        ),
        "consumer_fit_count": fit_count,
        "capability_claim": False,
        "memory_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Development-only exposed Source diagnostic. Marginal credit is conditional "
            "on this Ridge model, the other training rows, and the support fold; it is "
            "not intrinsic point quality or cross-dataset capability evidence."
        ),
    }


def run_action_value_guard(root: Path) -> dict[str, object]:
    """Extend the exposed report without rerunning legacy per-unit refits."""

    historical = _read_object(root / DEFAULT_REPORT_PATH)
    if historical.get("target_query_opened") is not False:
        raise ValueError("historical flatline report does not preserve Target boundary")
    if "dataset_evidence" not in historical or "consumer_fit_count" not in historical:
        raise ValueError("historical flatline report is unavailable for W29 replay")

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
        run as run_action_conditioned_proxy,
    )

    proxy_report = run_action_conditioned_proxy(root)
    diagnostic = proxy_report["action_value_guard_budget_diagnostic"]
    report = dict(historical)
    report["schema_version"] = "e2-flatline-actionability-credit/2"
    report["action_value_guard_budget_diagnostic"] = diagnostic
    report["w29_compute_accounting"] = diagnostic["compute_accounting"]
    report["w29_verdict"] = "EXPOSED_ACTION_VALUE_GUARD_BUDGET_DIAGNOSTIC_REPORTED"
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_action_value_guard(root)
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
