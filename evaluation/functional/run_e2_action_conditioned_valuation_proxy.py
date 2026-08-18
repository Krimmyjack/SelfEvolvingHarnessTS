"""Test a one-solve, action-conditioned Ridge attribution proxy.

This development-only instrument replays the already exposed flatline-masking
episodes and aligns against their historical one-at-a-time Consumer refits.  It
does not open Target/UCI data, promote the closed flatline family, or establish
utility support.  Exact Ridge row deletion is only a mechanical calibration;
the uncorrected first-order deletion is the TimeInf-like proxy under test.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
    DEFAULT_REPORT_PATH as HISTORICAL_CREDIT_REPORT_PATH,
    _alternating_folds,
    _fresh_roster,
    _mean,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    ANCHORS,
    CONTEXT_LENGTH,
    FRESH_SPECS,
    HORIZON,
    J0_PLAN_PATH,
    SEEDS,
    SELECTED_ROW_COUNT,
    SPECS,
    TARGET_BLOCK,
    _apply_stuck_value_censoring,
    _censor_flatline_interval_weights,
    _read_object,
)


SCHEMA_VERSION = "e2-action-conditioned-valuation-proxy/2"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_action_conditioned_valuation_proxy_report.json"
)
RIDGE_ALPHA = 1.0
EXACT_MAX_ABS_ERROR_THRESHOLD = 1e-8
ACTION_VALUE_FEEDBACK_BUDGET_SUPPORT_SPLITS = {
    1: tuple((index,) for index in range(8)),
    2: ((0, 1), (2, 3), (4, 5), (6, 7)),
    4: ((0, 2, 4, 6), (1, 3, 5, 7)),
}


def _ridge_reference_and_removal_predictions(
    np: Any,
    *,
    x_train: Any,
    targets: Any,
    x_eval: Any,
    candidate_rows: tuple[int, ...],
    target_block: tuple[int, int],
    alpha: float = RIDGE_ALPHA,
) -> dict[str, Any]:
    """Solve one Ridge system and return exact and first-order singleton removals.

    The final augmented-design coordinate is an unpenalized intercept.  One
    multi-right-hand-side solve obtains both the corrupt reference coefficients
    and ``A^-1 z_i`` for every candidate.  No per-action model fit occurs.
    """

    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    query = np.asarray(x_eval, dtype=np.float64)
    start, stop = target_block
    candidates = np.asarray(candidate_rows, dtype=np.int64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or query.ndim != 2
        or x.shape[0] != y.shape[0]
        or query.shape[1] != x.shape[1]
        or candidates.ndim != 1
        or candidates.size < 1
        or int(np.min(candidates)) < 0
        or int(np.max(candidates)) >= x.shape[0]
        or start < 0
        or stop > y.shape[1]
        or start >= stop
        or alpha <= 0.0
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(query).all()
    ):
        raise ValueError("invalid Ridge removal-proxy geometry")

    z_train = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    z_eval = np.column_stack((query, np.ones(query.shape[0], dtype=np.float64)))
    system = z_train.T @ z_train
    # P = diag(alpha, ..., alpha, 0): the intercept is intentionally unpenalized.
    system[:-1, :-1] += alpha * np.eye(x.shape[1], dtype=np.float64)
    coefficient_rhs = z_train.T @ y
    direction_rhs = z_train[candidates].T
    solved = np.linalg.solve(
        system, np.concatenate((coefficient_rhs, direction_rhs), axis=1)
    )
    beta = solved[:, : y.shape[1]]
    directions = solved[:, y.shape[1] :]
    baseline = z_eval @ beta
    candidate_z = z_train[candidates]
    leverage = np.sum(candidate_z * directions.T, axis=1)
    denominator = 1.0 - leverage
    if not np.isfinite(denominator).all() or np.any(denominator <= 1e-12):
        raise RuntimeError("Ridge singleton deletion has an unstable 1-h denominator")

    full_residual = y[candidates] - candidate_z @ beta
    residual = full_residual[:, start:stop]
    query_direction = z_eval @ directions
    exact = np.repeat(baseline[None, :, :], candidates.size, axis=0)
    first_order = exact.copy()
    for local_index in range(candidates.size):
        uncorrected_change = (
            query_direction[:, local_index, None]
            * residual[local_index, None, :]
        )
        # beta_minus = beta - A^-1 z_i e_i / (1-h_i).  Only the
        # action's TARGET_BLOCK is removed; all other outputs keep the corrupt
        # reference prediction exactly.
        exact[local_index, :, start:stop] -= (
            uncorrected_change / denominator[local_index]
        )
        first_order[local_index, :, start:stop] -= uncorrected_change

    if not (
        np.isfinite(baseline).all()
        and np.isfinite(exact).all()
        and np.isfinite(first_order).all()
    ):
        raise RuntimeError("non-finite Ridge attribution prediction")
    return {
        "baseline_prediction": baseline,
        "exact_removal_predictions": exact,
        "first_order_proxy_predictions": first_order,
        "leverage": leverage,
        "candidate_design": candidate_z,
        "candidate_directions": directions,
        "candidate_target_block_residual": residual,
        "candidate_full_residual": full_residual,
        "evaluation_design": z_eval,
    }


def _group_removal_predictions(
    np: Any,
    *,
    reference: dict[str, Any],
    selected_local_indices: tuple[int, ...],
    target_block: tuple[int, int],
    removal_strength: float = 1.0,
) -> dict[str, Any]:
    """Apply exact Woodbury and first-order downweighting for one candidate group."""

    baseline = np.asarray(reference["baseline_prediction"], dtype=np.float64)
    selected = np.asarray(selected_local_indices, dtype=np.int64)
    start, stop = target_block
    eta = float(removal_strength)
    if (
        selected.ndim != 1
        or start < 0
        or stop > baseline.shape[1]
        or start >= stop
        or not np.isfinite(eta)
        or eta <= 0.0
        or eta > 1.0
    ):
        raise ValueError("invalid grouped Ridge removal geometry")
    exact = baseline.copy()
    first_order = baseline.copy()
    if selected.size == 0:
        return {
            "exact_group_prediction": exact,
            "first_order_group_proxy_prediction": first_order,
            "selected_count": 0,
            "small_matrix_solve_count": 0,
            "middle_condition_number": None,
        }

    candidate_design = np.asarray(reference["candidate_design"], dtype=np.float64)
    directions = np.asarray(reference["candidate_directions"], dtype=np.float64)
    full_residual = np.asarray(reference["candidate_full_residual"], dtype=np.float64)
    residual = full_residual[:, start:stop]
    evaluation_design = np.asarray(reference["evaluation_design"], dtype=np.float64)
    if int(np.min(selected)) < 0 or int(np.max(selected)) >= candidate_design.shape[0]:
        raise ValueError("group selection contains an out-of-range candidate")
    group_design = candidate_design[selected]
    group_directions = directions[:, selected]
    group_residual = residual[selected]
    middle = (
        np.eye(selected.size, dtype=np.float64) / eta
        - group_design @ group_directions
    )
    correction = np.linalg.solve(middle, group_residual)
    query_direction = evaluation_design @ group_directions
    exact[:, start:stop] -= query_direction @ correction
    first_order[:, start:stop] -= eta * (query_direction @ group_residual)
    if not np.isfinite(exact).all() or not np.isfinite(first_order).all():
        raise RuntimeError("non-finite grouped Ridge attribution prediction")
    return {
        "exact_group_prediction": exact,
        "first_order_group_proxy_prediction": first_order,
        "selected_count": int(selected.size),
        "small_matrix_solve_count": 1,
        "middle_condition_number": float(np.linalg.cond(middle)),
    }


def _pearson(np: Any, actual: Any, predicted: Any) -> float | None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(predicted, dtype=np.float64)
    if (
        left.size < 2
        or float(np.std(left)) == 0.0
        or float(np.std(right)) == 0.0
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _sign_metrics(np: Any, actual: Any, predicted: Any) -> dict[str, object]:
    truth = np.asarray(actual, dtype=np.float64) > 0.0
    estimate = np.asarray(predicted, dtype=np.float64) > 0.0
    positive_count = int(np.count_nonzero(truth))
    nonpositive_count = int(truth.size - positive_count)
    accuracy = float(np.mean(truth == estimate))
    balanced_accuracy = None
    if positive_count and nonpositive_count:
        true_positive_rate = float(np.mean(estimate[truth]))
        true_negative_rate = float(np.mean(~estimate[~truth]))
        balanced_accuracy = (true_positive_rate + true_negative_rate) / 2.0
    return {
        "sign_agreement_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "actual_positive_count": positive_count,
        "actual_nonpositive_count": nonpositive_count,
        "actual_positive_fraction": positive_count / int(truth.size),
        "proxy_positive_count": int(np.count_nonzero(estimate)),
        "proxy_positive_fraction": float(np.mean(estimate)),
        "actual_label_majority_baseline_accuracy": (
            max(positive_count, nonpositive_count) / int(truth.size)
        ),
        "actual_label_majority_baseline_balanced_accuracy": (
            0.5 if positive_count and nonpositive_count else None
        ),
        "both_actual_sign_classes_present": bool(positive_count and nonpositive_count),
    }


def _credit_summary(np: Any, rows: list[dict[str, object]]) -> dict[str, object]:
    actual = np.asarray(
        [float(row["actual_refit_marginal_credit"]) for row in rows],
        dtype=np.float64,
    )
    exact = np.asarray(
        [float(row["exact_downdate_attribution_credit"]) for row in rows],
        dtype=np.float64,
    )
    proxy = np.asarray(
        [float(row["first_order_proxy_attribution_credit"]) for row in rows],
        dtype=np.float64,
    )
    exact_error = np.abs(exact - actual)
    proxy_error = np.abs(proxy - actual)
    selected = proxy > 0.0
    actual_mean = float(np.mean(actual))
    proxy_mean = float(np.mean(proxy))
    summary: dict[str, object] = {
        "fold_credit_count": len(rows),
        "exact_vs_actual_max_abs_error": float(np.max(exact_error)),
        "exact_vs_actual_mean_abs_error": float(np.mean(exact_error)),
        "first_order_vs_actual_mean_abs_error": float(np.mean(proxy_error)),
        "first_order_vs_actual_pearson": _pearson(np, actual, proxy),
        "actual_mean_marginal_credit": actual_mean,
        "first_order_proxy_mean_attribution_credit": proxy_mean,
        "actual_mean_direction": "POSITIVE" if actual_mean > 0.0 else "NONPOSITIVE",
        "first_order_proxy_mean_direction": (
            "POSITIVE" if proxy_mean > 0.0 else "NONPOSITIVE"
        ),
        "mean_credit_direction_agreement": (actual_mean > 0.0) == (proxy_mean > 0.0),
        "proxy_selected_fold_credit_count": int(np.count_nonzero(selected)),
        "proxy_selected_fold_credit_fraction": float(np.mean(selected)),
        "proxy_selected_actual_mean_marginal_credit": (
            float(np.mean(actual[selected])) if np.any(selected) else None
        ),
    }
    summary.update(_sign_metrics(np, actual, proxy))
    return summary


def _group_proxy_summary(
    np: Any, rows: list[dict[str, object]]
) -> dict[str, object]:
    exact = np.asarray(
        [float(row["exact_group_attribution_gain"]) for row in rows],
        dtype=np.float64,
    )
    proxy = np.asarray(
        [float(row["first_order_group_proxy_attribution_gain"]) for row in rows],
        dtype=np.float64,
    )
    summary: dict[str, object] = {
        "group_fold_credit_count": len(rows),
        "first_order_group_vs_exact_group_max_abs_error": float(
            np.max(np.abs(proxy - exact))
        ),
        "first_order_group_vs_exact_group_mean_abs_error": float(
            np.mean(np.abs(proxy - exact))
        ),
        "first_order_group_vs_exact_group_pearson": _pearson(np, exact, proxy),
    }
    sign = _sign_metrics(np, exact, proxy)
    summary.update(
        {
            "sign_agreement_accuracy": sign["sign_agreement_accuracy"],
            "balanced_accuracy": sign["balanced_accuracy"],
            "exact_group_positive_count": sign["actual_positive_count"],
            "exact_group_nonpositive_count": sign["actual_nonpositive_count"],
            "exact_group_positive_fraction": sign["actual_positive_fraction"],
            "first_order_group_proxy_positive_count": sign["proxy_positive_count"],
            "first_order_group_proxy_positive_fraction": sign[
                "proxy_positive_fraction"
            ],
            "exact_group_label_majority_baseline_accuracy": sign[
                "actual_label_majority_baseline_accuracy"
            ],
            "exact_group_label_majority_baseline_balanced_accuracy": sign[
                "actual_label_majority_baseline_balanced_accuracy"
            ],
            "both_exact_group_sign_classes_present": sign[
                "both_actual_sign_classes_present"
            ],
        }
    )
    return summary


def _nonadditivity_summary(
    np: Any, rows: list[dict[str, object]]
) -> dict[str, object]:
    exact_group = np.asarray(
        [float(row["exact_group_attribution_gain"]) for row in rows],
        dtype=np.float64,
    )
    naive_actual = np.asarray(
        [float(row["naive_sum_actual_singleton_credit"]) for row in rows],
        dtype=np.float64,
    )
    naive_exact = np.asarray(
        [float(row["naive_sum_exact_singleton_credit"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "group_fold_credit_count": len(rows),
        "naive_actual_singleton_sum_vs_exact_group_mean_abs_error": float(
            np.mean(np.abs(naive_actual - exact_group))
        ),
        "naive_actual_singleton_sum_vs_exact_group_max_abs_error": float(
            np.max(np.abs(naive_actual - exact_group))
        ),
        "naive_exact_singleton_sum_vs_exact_group_mean_abs_error": float(
            np.mean(np.abs(naive_exact - exact_group))
        ),
        "naive_exact_singleton_sum_vs_exact_group_max_abs_error": float(
            np.max(np.abs(naive_exact - exact_group))
        ),
    }


def _evaluate_action_value_guard_budget(
    np: Any,
    *,
    reference: dict[str, Any],
    baseline_losses: Any,
    score_predictions: Any,
    candidate_rows: tuple[int, ...],
    target_block: tuple[int, int],
    capture_support_action_response: bool = False,
    execution_removal_strength: float = 1.0,
) -> tuple[dict[str, dict[str, object]], int]:
    """Evaluate the frozen H0/H1 budget policies for one dataset episode."""

    baseline = np.asarray(reference["baseline_prediction"], dtype=np.float64)
    losses = np.asarray(baseline_losses, dtype=np.float64)
    exact_singleton = np.asarray(
        reference["exact_removal_predictions"], dtype=np.float64
    )
    proxy_singleton = np.asarray(
        reference["first_order_proxy_predictions"], dtype=np.float64
    )
    if (
        losses.shape != (8,)
        or len(candidate_rows) != proxy_singleton.shape[0]
        or exact_singleton.shape != proxy_singleton.shape
    ):
        raise ValueError("invalid frozen ActionValueGuard episode geometry")
    per_series_proxy_credit = [
        losses - score_predictions(proxy_singleton[local_index])
        for local_index in range(len(candidate_rows))
    ]
    per_series_exact_credit = (
        [
            losses - score_predictions(exact_singleton[local_index])
            for local_index in range(len(candidate_rows))
        ]
        if capture_support_action_response
        else None
    )
    h0_group = _group_removal_predictions(
        np,
        reference=reference,
        selected_local_indices=tuple(range(len(candidate_rows))),
        target_block=target_block,
        removal_strength=execution_removal_strength,
    )
    h0_losses = score_predictions(h0_group["exact_group_prediction"])
    h0_full_gain = float(np.mean(losses - h0_losses))
    evidence: dict[str, dict[str, object]] = {
        "0": {
            "support_series_budget": 0,
            "split_count": 1,
            "split_evidence": [
                {
                    "split_id": "b0_all_query",
                    "support_uid_indices": [],
                    "query_uid_indices": list(range(8)),
                    "h0_selected_row_indices": list(candidate_rows),
                    "h0_query_gain": h0_full_gain,
                    "h1_proposed_row_indices": [],
                    "h1_exact_grouped_support_gain": None,
                    "h1_guard_decision": "ABSTAIN_NO_FEEDBACK",
                    "h1_raw_query_gain": 0.0,
                    "h1_guarded_query_gain": 0.0,
                    "h1_grouped_small_matrix_solve_count": 0,
                }
            ],
        }
    }
    solve_count = 0
    for budget, support_splits in ACTION_VALUE_FEEDBACK_BUDGET_SUPPORT_SPLITS.items():
        split_evidence: list[dict[str, object]] = []
        for split_index, support_indices in enumerate(support_splits):
            query_indices = tuple(index for index in range(8) if index not in support_indices)
            selected_local = tuple(
                local_index
                for local_index, series_credit in enumerate(per_series_proxy_credit)
                if _mean(series_credit, support_indices) > 0.0
            )
            selected_rows = tuple(candidate_rows[index] for index in selected_local)
            h1_group = _group_removal_predictions(
                np,
                reference=reference,
                selected_local_indices=selected_local,
                target_block=target_block,
                removal_strength=execution_removal_strength,
            )
            group_solve_count = int(h1_group["small_matrix_solve_count"])
            solve_count += group_solve_count
            h1_losses = score_predictions(h1_group["exact_group_prediction"])
            support_gain = _mean(losses - h1_losses, support_indices)
            raw_query_gain = _mean(losses - h1_losses, query_indices)
            executes = support_gain > 0.0
            split_row: dict[str, object] = {
                "split_id": f"b{budget}_split_{split_index}",
                "support_uid_indices": list(support_indices),
                "query_uid_indices": list(query_indices),
                "h0_selected_row_indices": list(candidate_rows),
                "h0_query_gain": _mean(losses - h0_losses, query_indices),
                "h1_proposed_row_indices": list(selected_rows),
                "h1_proposed_count": len(selected_rows),
                "h1_proposed_fraction": len(selected_rows) / len(candidate_rows),
                "h1_exact_grouped_support_gain": support_gain,
                "h1_guard_threshold": 0.0,
                "h1_guard_decision": "EXECUTE" if executes else "ABSTAIN",
                "h1_raw_query_gain": raw_query_gain,
                "h1_guarded_query_gain": raw_query_gain if executes else 0.0,
                "h1_grouped_small_matrix_solve_count": group_solve_count,
                "h1_middle_condition_number": h1_group["middle_condition_number"],
            }
            if capture_support_action_response:
                if per_series_exact_credit is None:
                    raise RuntimeError("exact support Action-Response was not captured")
                selected_exact_support_means = [
                    _mean(per_series_exact_credit[local_index], support_indices)
                    for local_index in selected_local
                ]
                split_row[
                    "action__support_exact_singleton_sign_coherence"
                ] = (
                    statistics.fmean(
                        float(value > 0.0) for value in selected_exact_support_means
                    )
                    if selected_exact_support_means
                    else None
                )
            split_evidence.append(split_row)
        evidence[str(budget)] = {
            "support_series_budget": budget,
            "split_count": len(split_evidence),
            "split_evidence": split_evidence,
        }
    return evidence, solve_count


def _summarize_action_value_guard_budget(
    episodes: list[dict[str, dict[str, object]]],
) -> dict[str, object]:
    """Summarize frozen H0/H1 splits across repeated intervention episodes."""

    budget_summary: dict[str, dict[str, object]] = {}
    for budget in (0, 1, 2, 4):
        budget_key = str(budget)
        split_rows = [
            split
            for episode in episodes
            for split in episode[budget_key]["split_evidence"]
        ]
        h0_query_gains = [float(row["h0_query_gain"]) for row in split_rows]
        h1_query_gains = [float(row["h1_guarded_query_gain"]) for row in split_rows]
        h0_beneficial_pairs = [
            (h0_gain, h1_gain)
            for h0_gain, h1_gain in zip(h0_query_gains, h1_query_gains)
            if h0_gain > 0.0
        ]
        h0_beneficial_gain = sum(left for left, _ in h0_beneficial_pairs)
        retained_h1_beneficial_gain = sum(
            max(right, 0.0) for _, right in h0_beneficial_pairs
        )
        retained_split_count = sum(
            right > 0.0 for _, right in h0_beneficial_pairs
        )
        budget_summary[budget_key] = {
            "support_series_budget": budget,
            "split_count": len(split_rows),
            "h0_mean_query_gain": statistics.fmean(h0_query_gains),
            "h1_mean_guarded_query_gain": statistics.fmean(h1_query_gains),
            "h0_positive_split_count": sum(value > 0.0 for value in h0_query_gains),
            "h0_harmful_split_count": sum(value < 0.0 for value in h0_query_gains),
            "h0_abstained_split_count": 0,
            "h1_positive_split_count": sum(value > 0.0 for value in h1_query_gains),
            "h1_harmful_split_count": sum(value < 0.0 for value in h1_query_gains),
            "h1_abstained_split_count": sum(
                row["h1_guard_decision"] != "EXECUTE" for row in split_rows
            ),
            "h0_harmful_dataset": statistics.fmean(h0_query_gains) < 0.0,
            "h1_harmful_dataset": statistics.fmean(h1_query_gains) < 0.0,
            "h0_beneficial_gain_sum": h0_beneficial_gain,
            "h1_nonnegative_gain_on_h0_beneficial_splits": (
                retained_h1_beneficial_gain
            ),
            "beneficial_gain_retention_fraction": (
                retained_h1_beneficial_gain / h0_beneficial_gain
                if h0_beneficial_gain > 0.0
                else None
            ),
            "h0_beneficial_split_count": len(h0_beneficial_pairs),
            "retained_h0_beneficial_split_count": retained_split_count,
            "beneficial_split_retention_fraction": (
                retained_split_count / len(h0_beneficial_pairs)
                if h0_beneficial_pairs
                else None
            ),
            "mean_h1_proposed_fraction": (
                statistics.fmean(
                    float(row["h1_proposed_fraction"]) for row in split_rows
                )
                if budget > 0
                else 0.0
            ),
        }
    h0_auc = statistics.fmean(
        float(budget_summary[str(budget)]["h0_mean_query_gain"])
        for budget in (0, 1, 2, 4)
    )
    h1_auc = statistics.fmean(
        float(budget_summary[str(budget)]["h1_mean_guarded_query_gain"])
        for budget in (0, 1, 2, 4)
    )
    return {
        "budgets": budget_summary,
        "h0_adapt_auc_budget_grid_mean": h0_auc,
        "h1_adapt_auc_budget_grid_mean": h1_auc,
        "h1_minus_h0_adapt_auc": h1_auc - h0_auc,
    }


def run(
    root: Path,
    *,
    capture_support_action_response: bool = False,
    execution_removal_strength: float = 1.0,
) -> dict[str, object]:
    import numpy as np

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

    historical = _read_object(root / HISTORICAL_CREDIT_REPORT_PATH)
    if historical.get("target_query_opened") is not False:
        raise ValueError("historical diagnostic does not preserve the Target boundary")
    historical_fit_count = int(historical["consumer_fit_count"])
    historical_by_dataset = {
        str(row["dataset_id"]): row for row in historical["dataset_evidence"]
    }

    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    j0_plan = _read_object(root / J0_PLAN_PATH)
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
        raise ValueError("UCI is forbidden in this exposed Source diagnostic")

    selected_records = [records[str(row["series_uid"])] for row in roster]
    values = _load_values(selected_records, root / "data/benchmark_v0_2/clean_base")
    specs = {**SPECS, **FRESH_SPECS}
    if set(historical_by_dataset) != set(specs):
        raise ValueError("historical credit datasets do not match the exposed Source roster")

    reference_system_construction_count = 0
    reference_solve_count = 0
    grouped_small_matrix_solve_count = 0
    action_value_budget_grouped_small_matrix_solve_count = 0
    all_fold_rows: list[dict[str, object]] = []
    all_group_rows: list[dict[str, object]] = []
    all_proxy_guided_holdout_rows: list[dict[str, object]] = []
    all_group_history_abs_errors: list[float] = []
    all_leverage: list[float] = []
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

        historical_seed_rows = {
            int(row["seed"]): row
            for row in historical_by_dataset[dataset_id]["seed_evidence"]
        }
        if set(historical_seed_rows) != set(SEEDS):
            raise ValueError(f"historical seeds changed: {dataset_id}")
        seed_evidence: list[dict[str, object]] = []
        dataset_fold_rows: list[dict[str, object]] = []
        dataset_group_rows: list[dict[str, object]] = []
        dataset_proxy_guided_holdout_rows: list[dict[str, object]] = []
        dataset_group_history_abs_errors: list[float] = []
        dataset_leverage: list[float] = []

        for seed in SEEDS:
            historical_units = list(historical_seed_rows[seed]["unit_credit"])
            if len(historical_units) != SELECTED_ROW_COUNT:
                raise ValueError(f"historical candidate count changed: {dataset_id}/{seed}")
            historical_units_by_index = {
                int(row["row_index"]): row for row in historical_units
            }
            candidate_rows = tuple(sorted(historical_units_by_index))
            selected_truth = set(candidate_rows)
            corrupt, _, _ = _apply_stuck_value_censoring(
                np,
                clean_y,
                x_train[:, :CONTEXT_LENGTH],
                selected_truth,
                TARGET_BLOCK,
            )
            observed_weights, observations = _censor_flatline_interval_weights(np, corrupt)
            observed_candidates = tuple(
                index
                for index, observation in enumerate(observations)
                if observation["status"] == "ACTIVATE"
            )
            if observed_candidates != candidate_rows:
                raise AssertionError(
                    f"flatline candidates no longer align: {dataset_id}/{seed}"
                )
            if int(np.count_nonzero(observed_weights == 0.0)) != (
                SELECTED_ROW_COUNT * (TARGET_BLOCK[1] - TARGET_BLOCK[0])
            ):
                raise AssertionError("flatline compiler geometry changed")
            for candidate in candidate_rows:
                expected_key = list(row_keys[candidate])
                if historical_units_by_index[candidate]["row_key"] != expected_key:
                    raise AssertionError(
                        f"historical row key no longer aligns: {dataset_id}/{seed}/{candidate}"
                    )

            reference_system_construction_count += 1
            prediction_bundle = _ridge_reference_and_removal_predictions(
                np,
                x_train=x_train,
                targets=corrupt,
                x_eval=x_eval_array,
                candidate_rows=candidate_rows,
                target_block=TARGET_BLOCK,
            )
            reference_solve_count += 1
            baseline_losses = score_predictions(prediction_bundle["baseline_prediction"])
            leverages = np.asarray(prediction_bundle["leverage"], dtype=np.float64)
            dataset_leverage.extend(float(value) for value in leverages)
            all_leverage.extend(float(value) for value in leverages)
            action_rows: list[dict[str, object]] = []

            for local_index, candidate in enumerate(candidate_rows):
                exact_losses = score_predictions(
                    prediction_bundle["exact_removal_predictions"][local_index]
                )
                proxy_losses = score_predictions(
                    prediction_bundle["first_order_proxy_predictions"][local_index]
                )
                history_row = historical_units_by_index[candidate]
                action_row: dict[str, object] = {
                    "row_index": candidate,
                    "row_key": list(row_keys[candidate]),
                    "leverage": float(leverages[local_index]),
                    "one_minus_leverage": float(1.0 - leverages[local_index]),
                    "fold_attribution": {},
                }
                for fold_name, fold_indices in folds.items():
                    history_key = f"{fold_name}_marginal_credit"
                    actual_credit = float(history_row[history_key])
                    exact_credit = _mean(
                        baseline_losses - exact_losses, fold_indices
                    )
                    proxy_credit = _mean(
                        baseline_losses - proxy_losses, fold_indices
                    )
                    fold_row = {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "row_index": candidate,
                        "row_key": list(row_keys[candidate]),
                        "fold": fold_name,
                        "actual_refit_marginal_credit": actual_credit,
                        "exact_downdate_attribution_credit": exact_credit,
                        "first_order_proxy_attribution_credit": proxy_credit,
                    }
                    dataset_fold_rows.append(fold_row)
                    all_fold_rows.append(fold_row)
                    action_row["fold_attribution"][fold_name] = {
                        "actual_refit_marginal_credit": actual_credit,
                        "exact_downdate_attribution_credit": exact_credit,
                        "first_order_proxy_attribution_credit": proxy_credit,
                        "exact_vs_actual_abs_error": abs(exact_credit - actual_credit),
                        "first_order_vs_actual_abs_error": abs(proxy_credit - actual_credit),
                    }
                action_rows.append(action_row)

            historical_crossfit = {
                str(row["direction"]): row
                for row in historical_seed_rows[seed]["crossfit_policy"]
            }
            if set(historical_crossfit) != {"a_to_b", "b_to_a"}:
                raise ValueError(f"historical crossfit directions changed: {dataset_id}/{seed}")

            all_group = _group_removal_predictions(
                np,
                reference=prediction_bundle,
                selected_local_indices=tuple(range(len(candidate_rows))),
                target_block=TARGET_BLOCK,
            )
            grouped_small_matrix_solve_count += int(
                all_group["small_matrix_solve_count"]
            )
            all_group_exact_losses = score_predictions(
                all_group["exact_group_prediction"]
            )
            all_group_proxy_losses = score_predictions(
                all_group["first_order_group_proxy_prediction"]
            )
            exact_full_gain = float(np.mean(baseline_losses - all_group_exact_losses))
            historical_full_gain = float(
                historical_seed_rows[seed]["always_mask_full_eval_gain"]
            )
            full_history_error = abs(exact_full_gain - historical_full_gain)
            dataset_group_history_abs_errors.append(full_history_error)
            all_group_history_abs_errors.append(full_history_error)
            all_group_fold_evidence: list[dict[str, object]] = []
            historical_all_fold_gain = {
                "fold_a": float(
                    historical_crossfit["a_to_b"]["always_mask_support_gain"]
                ),
                "fold_b": float(
                    historical_crossfit["a_to_b"]["always_mask_holdout_gain"]
                ),
            }
            for fold_name, fold_indices in folds.items():
                exact_gain = _mean(
                    baseline_losses - all_group_exact_losses, fold_indices
                )
                proxy_gain = _mean(
                    baseline_losses - all_group_proxy_losses, fold_indices
                )
                naive_actual = sum(
                    float(row["fold_attribution"][fold_name]["actual_refit_marginal_credit"])
                    for row in action_rows
                )
                naive_exact = sum(
                    float(
                        row["fold_attribution"][fold_name][
                            "exact_downdate_attribution_credit"
                        ]
                    )
                    for row in action_rows
                )
                historical_gain = historical_all_fold_gain[fold_name]
                history_error = abs(exact_gain - historical_gain)
                dataset_group_history_abs_errors.append(history_error)
                all_group_history_abs_errors.append(history_error)
                group_row = {
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "group_kind": "all_detected_rows",
                    "evaluation_role": fold_name,
                    "selected_count": len(candidate_rows),
                    "exact_group_attribution_gain": exact_gain,
                    "first_order_group_proxy_attribution_gain": proxy_gain,
                    "naive_sum_actual_singleton_credit": naive_actual,
                    "naive_sum_exact_singleton_credit": naive_exact,
                }
                dataset_group_rows.append(group_row)
                all_group_rows.append(group_row)
                all_group_fold_evidence.append(
                    {
                        "fold": fold_name,
                        "exact_group_attribution_gain": exact_gain,
                        "first_order_group_proxy_attribution_gain": proxy_gain,
                        "historical_always_mask_refit_gain": historical_gain,
                        "exact_group_vs_historical_abs_error": history_error,
                        "naive_sum_actual_singleton_credit": naive_actual,
                        "naive_sum_exact_singleton_credit": naive_exact,
                        "naive_actual_sum_nonadditivity_error": naive_actual - exact_gain,
                        "naive_exact_sum_nonadditivity_error": naive_exact - exact_gain,
                    }
                )

            action_value_budget_evidence, budget_solve_count = (
                _evaluate_action_value_guard_budget(
                    np,
                    reference=prediction_bundle,
                    baseline_losses=baseline_losses,
                    score_predictions=score_predictions,
                    candidate_rows=candidate_rows,
                    target_block=TARGET_BLOCK,
                    capture_support_action_response=capture_support_action_response,
                    execution_removal_strength=execution_removal_strength,
                )
            )
            action_value_budget_grouped_small_matrix_solve_count += (
                budget_solve_count
            )

            guided_group_evidence: list[dict[str, object]] = []
            fold_pairs = (
                ("a_to_b", "fold_a", "fold_b"),
                ("b_to_a", "fold_b", "fold_a"),
            )
            for direction, support_name, holdout_name in fold_pairs:
                selected_local = tuple(
                    local_index
                    for local_index, row in enumerate(action_rows)
                    if float(
                        row["fold_attribution"][support_name][
                            "first_order_proxy_attribution_credit"
                        ]
                    )
                    > 0.0
                )
                selected_rows = tuple(candidate_rows[index] for index in selected_local)
                guided_group = _group_removal_predictions(
                    np,
                    reference=prediction_bundle,
                    selected_local_indices=selected_local,
                    target_block=TARGET_BLOCK,
                )
                grouped_small_matrix_solve_count += int(
                    guided_group["small_matrix_solve_count"]
                )
                guided_exact_losses = score_predictions(
                    guided_group["exact_group_prediction"]
                )
                guided_proxy_losses = score_predictions(
                    guided_group["first_order_group_proxy_prediction"]
                )
                by_role: dict[str, dict[str, object]] = {}
                for evaluation_role, fold_name in (
                    ("support", support_name),
                    ("holdout", holdout_name),
                ):
                    fold_indices = folds[fold_name]
                    exact_gain = _mean(
                        baseline_losses - guided_exact_losses, fold_indices
                    )
                    proxy_gain = _mean(
                        baseline_losses - guided_proxy_losses, fold_indices
                    )
                    selected_actions = [action_rows[index] for index in selected_local]
                    naive_actual = sum(
                        float(
                            row["fold_attribution"][fold_name][
                                "actual_refit_marginal_credit"
                            ]
                        )
                        for row in selected_actions
                    )
                    naive_exact = sum(
                        float(
                            row["fold_attribution"][fold_name][
                                "exact_downdate_attribution_credit"
                            ]
                        )
                        for row in selected_actions
                    )
                    group_row = {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "group_kind": "singleton_first_order_proxy_positive_cohort",
                        "direction": direction,
                        "evaluation_role": evaluation_role,
                        "fold": fold_name,
                        "selected_count": len(selected_local),
                        "exact_group_attribution_gain": exact_gain,
                        "first_order_group_proxy_attribution_gain": proxy_gain,
                        "naive_sum_actual_singleton_credit": naive_actual,
                        "naive_sum_exact_singleton_credit": naive_exact,
                    }
                    dataset_group_rows.append(group_row)
                    all_group_rows.append(group_row)
                    if evaluation_role == "holdout":
                        dataset_proxy_guided_holdout_rows.append(group_row)
                        all_proxy_guided_holdout_rows.append(group_row)
                    by_role[evaluation_role] = {
                        "fold": fold_name,
                        "exact_group_attribution_gain": exact_gain,
                        "first_order_group_proxy_attribution_gain": proxy_gain,
                        "naive_sum_actual_singleton_credit": naive_actual,
                        "naive_sum_exact_singleton_credit": naive_exact,
                        "naive_actual_sum_nonadditivity_error": naive_actual - exact_gain,
                        "naive_exact_sum_nonadditivity_error": naive_exact - exact_gain,
                    }
                old_policy = historical_crossfit[direction]
                guided_group_evidence.append(
                    {
                        "direction": direction,
                        "selection_source": support_name,
                        "selection_rule": (
                            "include iff singleton first-order proxy attribution credit > 0"
                        ),
                        "selected_row_indices": list(selected_rows),
                        "selected_count": len(selected_rows),
                        "empty_group_keeps_baseline": len(selected_rows) == 0,
                        "small_matrix_solve_count": int(
                            guided_group["small_matrix_solve_count"]
                        ),
                        "middle_condition_number": guided_group[
                            "middle_condition_number"
                        ],
                        "support": by_role["support"],
                        "holdout": by_role["holdout"],
                        "historical_exact_credit_guided_selected_count": int(
                            old_policy["selected_count"]
                        ),
                        "historical_exact_credit_guided_holdout_gain": float(
                            old_policy["credit_guided_holdout_gain"]
                        ),
                        "historical_always_mask_holdout_gain": float(
                            old_policy["always_mask_holdout_gain"]
                        ),
                    }
                )

            seed_evidence.append(
                {
                    "seed": seed,
                    "candidate_action_count": len(candidate_rows),
                    "reference_system_construction_count": 1,
                    "reference_solve_count": 1,
                    "per_action_refit_count": 0,
                    "actions": action_rows,
                    "action_value_guard_budget": action_value_budget_evidence,
                    "group_attribution": {
                        "all_detected_rows": {
                            "selected_count": len(candidate_rows),
                            "small_matrix_solve_count": int(
                                all_group["small_matrix_solve_count"]
                            ),
                            "middle_condition_number": all_group[
                                "middle_condition_number"
                            ],
                            "exact_full_eval_gain": exact_full_gain,
                            "historical_always_mask_full_eval_gain": historical_full_gain,
                            "exact_group_vs_historical_full_abs_error": full_history_error,
                            "folds": all_group_fold_evidence,
                        },
                        "proxy_guided_cohorts": guided_group_evidence,
                    },
                }
            )

        dataset_summary = _credit_summary(np, dataset_fold_rows)
        dataset_summary["dataset_mean_direction_agreement"] = dataset_summary.pop(
            "mean_credit_direction_agreement"
        )
        dataset_summary.update(
            {
                "action_count": len(dataset_fold_rows) // 2,
                "first_order_pearson_direction": (
                    "POSITIVE"
                    if dataset_summary["first_order_vs_actual_pearson"] is not None
                    and float(dataset_summary["first_order_vs_actual_pearson"]) > 0.0
                    else "NONPOSITIVE_OR_UNAVAILABLE"
                ),
                "max_leverage": max(dataset_leverage),
                "min_one_minus_leverage": min(1.0 - value for value in dataset_leverage),
            }
        )
        dataset_group_summary = {
            "exact_group_vs_historical_max_abs_error": max(
                dataset_group_history_abs_errors
            ),
            "exact_group_vs_historical_mean_abs_error": statistics.fmean(
                dataset_group_history_abs_errors
            ),
            "all_and_guided_group_proxy_quality": _group_proxy_summary(
                np, dataset_group_rows
            ),
            "all_and_guided_group_nonadditivity": _nonadditivity_summary(
                np, dataset_group_rows
            ),
            "proxy_guided_holdout_proxy_quality": _group_proxy_summary(
                np, dataset_proxy_guided_holdout_rows
            ),
            "proxy_guided_exact_holdout_mean_gain": statistics.fmean(
                float(row["exact_group_attribution_gain"])
                for row in dataset_proxy_guided_holdout_rows
            ),
            "proxy_guided_first_order_holdout_mean_gain": statistics.fmean(
                float(row["first_order_group_proxy_attribution_gain"])
                for row in dataset_proxy_guided_holdout_rows
            ),
            "historical_exact_credit_guided_holdout_mean_gain": statistics.fmean(
                float(direction["historical_exact_credit_guided_holdout_gain"])
                for seed_row in seed_evidence
                for direction in seed_row["group_attribution"][
                    "proxy_guided_cohorts"
                ]
            ),
            "historical_always_mask_holdout_mean_gain": statistics.fmean(
                float(direction["historical_always_mask_holdout_gain"])
                for seed_row in seed_evidence
                for direction in seed_row["group_attribution"][
                    "proxy_guided_cohorts"
                ]
            ),
            "mean_proxy_selected_fraction": statistics.fmean(
                int(direction["selected_count"]) / SELECTED_ROW_COUNT
                for seed_row in seed_evidence
                for direction in seed_row["group_attribution"][
                    "proxy_guided_cohorts"
                ]
            ),
            "empty_proxy_guided_group_count": sum(
                bool(direction["empty_group_keeps_baseline"])
                for seed_row in seed_evidence
                for direction in seed_row["group_attribution"][
                    "proxy_guided_cohorts"
                ]
            ),
        }
        dataset_action_value_summary = _summarize_action_value_guard_budget(
            [
                seed_row["action_value_guard_budget"]
                for seed_row in seed_evidence
            ]
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "evaluation_uids": eval_uids,
                "folds": {key: list(value) for key, value in folds.items()},
                "seed_evidence": seed_evidence,
                "attribution_summary": dataset_summary,
                "group_attribution_summary": dataset_group_summary,
                "action_value_guard_budget_diagnostic": (
                    dataset_action_value_summary
                ),
            }
        )

    overall_summary = _credit_summary(np, all_fold_rows)
    overall_summary["overall_mean_credit_direction_agreement"] = overall_summary.pop(
        "mean_credit_direction_agreement"
    )
    dataset_summaries = [row["attribution_summary"] for row in dataset_evidence]
    available_balanced = [
        float(row["balanced_accuracy"])
        for row in dataset_summaries
        if row["balanced_accuracy"] is not None
    ]
    positive_dataset_pearson_count = sum(
        row["first_order_vs_actual_pearson"] is not None
        and float(row["first_order_vs_actual_pearson"]) > 0.0
        for row in dataset_summaries
    )
    dataset_mean_direction_agreement_count = sum(
        bool(row["dataset_mean_direction_agreement"]) for row in dataset_summaries
    )
    exact_reconstructed = (
        float(overall_summary["exact_vs_actual_max_abs_error"])
        <= EXACT_MAX_ABS_ERROR_THRESHOLD
    )
    diagnostic_checks = {
        "overall_sign_accuracy_exceeds_actual_label_majority_baseline": (
            float(overall_summary["sign_agreement_accuracy"])
            > float(overall_summary["actual_label_majority_baseline_accuracy"])
        ),
        "dataset_macro_balanced_accuracy_above_chance": (
            bool(available_balanced)
            and statistics.fmean(available_balanced) > 0.5
        ),
        "overall_pearson_positive": (
            overall_summary["first_order_vs_actual_pearson"] is not None
            and float(overall_summary["first_order_vs_actual_pearson"]) > 0.0
        ),
        "at_least_three_of_four_dataset_pearsons_positive": (
            positive_dataset_pearson_count >= 3
        ),
        "proxy_selected_actual_marginal_credit_positive": (
            overall_summary["proxy_selected_actual_mean_marginal_credit"] is not None
            and float(overall_summary["proxy_selected_actual_mean_marginal_credit"])
            > 0.0
        ),
    }
    first_order_signal = exact_reconstructed and all(diagnostic_checks.values())
    overall_summary.update(
        {
            "action_count": len(all_fold_rows) // 2,
            "dataset_macro_balanced_accuracy": (
                statistics.fmean(available_balanced) if available_balanced else None
            ),
            "datasets_with_both_actual_sign_classes": len(available_balanced),
            "positive_dataset_pearson_count": positive_dataset_pearson_count,
            "dataset_mean_direction_agreement_count": (
                dataset_mean_direction_agreement_count
            ),
            "max_leverage": max(all_leverage),
            "min_one_minus_leverage": min(1.0 - value for value in all_leverage),
        }
    )

    dataset_group_summaries = [
        row["group_attribution_summary"] for row in dataset_evidence
    ]
    proxy_guided_exact_dataset_gains = [
        float(row["proxy_guided_exact_holdout_mean_gain"])
        for row in dataset_group_summaries
    ]
    old_credit_guided_dataset_gains = [
        float(row["historical_exact_credit_guided_holdout_mean_gain"])
        for row in dataset_group_summaries
    ]
    always_mask_dataset_gains = [
        float(row["historical_always_mask_holdout_mean_gain"])
        for row in dataset_group_summaries
    ]
    group_exact_reconstructed = (
        max(all_group_history_abs_errors) <= EXACT_MAX_ABS_ERROR_THRESHOLD
    )
    group_holdout_proxy_quality = _group_proxy_summary(
        np, all_proxy_guided_holdout_rows
    )
    group_first_order_checks = {
        "sign_accuracy_exceeds_exact_group_majority_baseline": (
            float(group_holdout_proxy_quality["sign_agreement_accuracy"])
            > float(
                group_holdout_proxy_quality[
                    "exact_group_label_majority_baseline_accuracy"
                ]
            )
        ),
        "balanced_accuracy_above_chance": (
            group_holdout_proxy_quality["balanced_accuracy"] is not None
            and float(group_holdout_proxy_quality["balanced_accuracy"]) > 0.5
        ),
        "pearson_positive": (
            group_holdout_proxy_quality[
                "first_order_group_vs_exact_group_pearson"
            ]
            is not None
            and float(
                group_holdout_proxy_quality[
                    "first_order_group_vs_exact_group_pearson"
                ]
            )
            > 0.0
        ),
    }
    group_first_order_signal = group_exact_reconstructed and all(
        group_first_order_checks.values()
    )
    group_overall = {
        "exact_group_vs_historical_comparison_count": len(
            all_group_history_abs_errors
        ),
        "exact_group_vs_historical_max_abs_error": max(
            all_group_history_abs_errors
        ),
        "exact_group_vs_historical_mean_abs_error": statistics.fmean(
            all_group_history_abs_errors
        ),
        "exact_group_reconstructs_historical_always_mask": group_exact_reconstructed,
        "all_and_guided_group_proxy_quality": _group_proxy_summary(
            np, all_group_rows
        ),
        "proxy_guided_holdout_proxy_quality": group_holdout_proxy_quality,
        "first_order_group_proxy_checks": group_first_order_checks,
        "first_order_group_proxy_has_signal": group_first_order_signal,
        "all_and_guided_group_nonadditivity": _nonadditivity_summary(
            np, all_group_rows
        ),
        "dataset_macro_proxy_guided_exact_holdout_gain": statistics.fmean(
            proxy_guided_exact_dataset_gains
        ),
        "dataset_macro_historical_exact_credit_guided_holdout_gain": (
            statistics.fmean(old_credit_guided_dataset_gains)
        ),
        "dataset_macro_historical_always_mask_holdout_gain": statistics.fmean(
            always_mask_dataset_gains
        ),
        "proxy_guided_exact_harmful_dataset_count": sum(
            value < 0.0 for value in proxy_guided_exact_dataset_gains
        ),
        "historical_exact_credit_guided_harmful_dataset_count": sum(
            value < 0.0 for value in old_credit_guided_dataset_gains
        ),
        "historical_always_mask_harmful_dataset_count": sum(
            value < 0.0 for value in always_mask_dataset_gains
        ),
        "proxy_guided_exact_beats_historical_exact_credit_guided_macro": (
            statistics.fmean(proxy_guided_exact_dataset_gains)
            > statistics.fmean(old_credit_guided_dataset_gains)
        ),
        "proxy_guided_exact_beats_historical_always_mask_macro": (
            statistics.fmean(proxy_guided_exact_dataset_gains)
            > statistics.fmean(always_mask_dataset_gains)
        ),
        "proxy_guided_exact_dataset_gain_positive_count": sum(
            value > 0.0 for value in proxy_guided_exact_dataset_gains
        ),
        "mean_proxy_selected_fraction": statistics.fmean(
            float(row["mean_proxy_selected_fraction"])
            for row in dataset_group_summaries
        ),
        "empty_proxy_guided_group_count": sum(
            int(row["empty_proxy_guided_group_count"])
            for row in dataset_group_summaries
        ),
    }
    group_verdict = (
        "EXACT_GROUP_RECONSTRUCTION_FAILED"
        if not group_exact_reconstructed
        else (
            "EXACT_GROUP_AVAILABLE_FIRST_ORDER_GROUP_PROXY_SIGNAL_PRESENT"
            if group_first_order_signal
            else "EXACT_GROUP_AVAILABLE_FIRST_ORDER_GROUP_PROXY_INSUFFICIENT"
        )
    )

    singleton_verdict = (
        "EXACT_DOWNDATE_RECONSTRUCTION_FAILED"
        if not exact_reconstructed
        else (
            "FIRST_ORDER_ATTRIBUTION_PROXY_SIGNAL_PRESENT_FOR_GROUP_COHORT_DIAGNOSTIC"
            if first_order_signal
            else "FIRST_ORDER_ATTRIBUTION_PROXY_SIGNAL_INSUFFICIENT"
        )
    )
    if not exact_reconstructed or not group_exact_reconstructed:
        verdict = "ACTION_CONDITIONED_VALUATION_MECHANICAL_RECONSTRUCTION_FAILED"
    elif first_order_signal and not group_first_order_signal:
        verdict = (
            "SINGLETON_PROXY_SELECTS_USEFUL_COHORT_EXACT_GROUP_FEEDBACK_REQUIRED"
        )
    elif first_order_signal and group_first_order_signal:
        verdict = "SINGLETON_AND_GROUP_FIRST_ORDER_PROXY_SIGNAL_PRESENT"
    else:
        verdict = "ACTION_CONDITIONED_FIRST_ORDER_PROXY_INSUFFICIENT"

    action_value_budget_overall: dict[str, dict[str, object]] = {}
    for budget in (0, 1, 2, 4):
        budget_key = str(budget)
        dataset_budget_rows = [
            row["action_value_guard_budget_diagnostic"]["budgets"][budget_key]
            for row in dataset_evidence
        ]
        h0_dataset_gains = [
            float(row["h0_mean_query_gain"]) for row in dataset_budget_rows
        ]
        h1_dataset_gains = [
            float(row["h1_mean_guarded_query_gain"])
            for row in dataset_budget_rows
        ]
        h0_beneficial_gain = sum(
            float(row["h0_beneficial_gain_sum"]) for row in dataset_budget_rows
        )
        retained_h1_beneficial_gain = sum(
            float(row["h1_nonnegative_gain_on_h0_beneficial_splits"])
            for row in dataset_budget_rows
        )
        h0_beneficial_split_count = sum(
            int(row["h0_beneficial_split_count"]) for row in dataset_budget_rows
        )
        retained_h0_beneficial_split_count = sum(
            int(row["retained_h0_beneficial_split_count"])
            for row in dataset_budget_rows
        )
        action_value_budget_overall[budget_key] = {
            "support_series_budget": budget,
            "dataset_macro_h0_query_gain": statistics.fmean(h0_dataset_gains),
            "dataset_macro_h1_guarded_query_gain": statistics.fmean(
                h1_dataset_gains
            ),
            "h1_minus_h0_dataset_macro_query_gain": statistics.fmean(
                h1_dataset_gains
            )
            - statistics.fmean(h0_dataset_gains),
            "h0_positive_split_count": sum(
                int(row["h0_positive_split_count"]) for row in dataset_budget_rows
            ),
            "h0_harmful_split_count": sum(
                int(row["h0_harmful_split_count"]) for row in dataset_budget_rows
            ),
            "h0_abstained_split_count": 0,
            "h1_positive_split_count": sum(
                int(row["h1_positive_split_count"]) for row in dataset_budget_rows
            ),
            "h1_harmful_split_count": sum(
                int(row["h1_harmful_split_count"]) for row in dataset_budget_rows
            ),
            "h1_abstained_split_count": sum(
                int(row["h1_abstained_split_count"]) for row in dataset_budget_rows
            ),
            "h0_harmful_dataset_ids": [
                str(dataset_evidence[index]["dataset_id"])
                for index, value in enumerate(h0_dataset_gains)
                if value < 0.0
            ],
            "h1_harmful_dataset_ids": [
                str(dataset_evidence[index]["dataset_id"])
                for index, value in enumerate(h1_dataset_gains)
                if value < 0.0
            ],
            "h0_beneficial_gain_sum": h0_beneficial_gain,
            "h1_nonnegative_gain_on_h0_beneficial_splits": (
                retained_h1_beneficial_gain
            ),
            "beneficial_gain_retention_fraction": (
                retained_h1_beneficial_gain / h0_beneficial_gain
                if h0_beneficial_gain > 0.0
                else None
            ),
            "h0_beneficial_split_count": h0_beneficial_split_count,
            "retained_h0_beneficial_split_count": (
                retained_h0_beneficial_split_count
            ),
            "beneficial_split_retention_fraction": (
                retained_h0_beneficial_split_count / h0_beneficial_split_count
                if h0_beneficial_split_count
                else None
            ),
        }
    h0_macro_adapt_auc = statistics.fmean(
        float(
            row["action_value_guard_budget_diagnostic"][
                "h0_adapt_auc_budget_grid_mean"
            ]
        )
        for row in dataset_evidence
    )
    h1_macro_adapt_auc = statistics.fmean(
        float(
            row["action_value_guard_budget_diagnostic"][
                "h1_adapt_auc_budget_grid_mean"
            ]
        )
        for row in dataset_evidence
    )
    retention_rows = [action_value_budget_overall[str(budget)] for budget in (1, 2, 4)]
    h0_beneficial_gain = sum(
        float(row["h0_beneficial_gain_sum"]) for row in retention_rows
    )
    h1_retained_beneficial_gain = sum(
        float(row["h1_nonnegative_gain_on_h0_beneficial_splits"])
        for row in retention_rows
    )
    h0_beneficial_split_count = sum(
        int(row["h0_beneficial_split_count"]) for row in retention_rows
    )
    h1_retained_beneficial_split_count = sum(
        int(row["retained_h0_beneficial_split_count"])
        for row in retention_rows
    )
    low_budget_h0_harmful_splits = sum(
        int(action_value_budget_overall[str(budget)]["h0_harmful_split_count"])
        for budget in (1, 2)
    )
    low_budget_h1_harmful_splits = sum(
        int(action_value_budget_overall[str(budget)]["h1_harmful_split_count"])
        for budget in (1, 2)
    )
    low_budget_h0_harmful_datasets = sum(
        len(action_value_budget_overall[str(budget)]["h0_harmful_dataset_ids"])
        for budget in (1, 2)
    )
    low_budget_h1_harmful_datasets = sum(
        len(action_value_budget_overall[str(budget)]["h1_harmful_dataset_ids"])
        for budget in (1, 2)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_source_development_attribution_instrument_feasibility",
        "question": (
            "Can an action-conditioned, first-order Ridge removal proxy preserve enough "
            "of the already exposed singleton marginal-credit signal to justify a later "
            "group/cohort attribution diagnostic without one Consumer refit per action?"
        ),
        "group_question": (
            "Can grouped action value account for non-additive row interactions, and "
            "does a cohort selected by support-fold singleton first-order signs retain "
            "positive exact grouped credit on the disjoint holdout fold?"
        ),
        "configuration": {
            "datasets": list(specs),
            "seeds": list(SEEDS),
            "training_geometry": [72, 384],
            "evaluation_row_count": 8,
            "candidate_actions_per_dataset_seed": SELECTED_ROW_COUNT,
            "target_block_half_open": list(TARGET_BLOCK),
            "consumer_reference": (
                "Ridge(alpha=1.0) augmented design z=[x,1], "
                "P=diag(alpha,...,alpha,0) with unpenalized intercept"
            ),
            "metric": "per-series sMASE, equal mean within alternating 4/4 fold",
            "exact_attribution": (
                "singleton row removal with 1/(1-h_i) leverage correction"
            ),
            "first_order_proxy_attribution": (
                "singleton row removal linearization without leverage correction"
            ),
            "exact_group_attribution": (
                "A^-1 Z_G^T (I - Z_G A^-1 Z_G^T)^-1 R_G; only TARGET_BLOCK changes"
            ),
            "first_order_group_proxy_attribution": (
                "A^-1 Z_G^T R_G without the grouped leverage correction"
            ),
            "proxy_guided_group_selection": (
                "within each direction include candidate iff support-fold singleton "
                "first-order proxy attribution credit > 0; empty group keeps baseline"
            ),
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
        },
        "diagnostic_thresholds": {
            "exact_reconstruction": (
                f"overall exact-vs-actual max absolute credit error <= "
                f"{EXACT_MAX_ABS_ERROR_THRESHOLD}"
            ),
            "first_order_signal_checks": [
                "overall sign accuracy > actual-label majority-sign baseline accuracy",
                "dataset-macro balanced accuracy > 0.5",
                "overall Pearson > 0",
                "at least 3 of 4 dataset Pearson values > 0",
                "mean actual marginal credit among proxy>0 fold-actions > 0",
            ],
            "group_exact_reconstruction": (
                f"all-14 exact grouped gain vs historical always-mask refit max "
                f"absolute error <= {EXACT_MAX_ABS_ERROR_THRESHOLD}"
            ),
            "group_sign_and_selection_threshold": 0.0,
            "role": (
                "relative development diagnostic only; this is not a Capability, "
                "Promotion, transfer, or utility-support gate"
            ),
        },
        "dataset_evidence": dataset_evidence,
        "overall_attribution_summary": overall_summary,
        "diagnostic_checks": diagnostic_checks,
        "exact_downdate_reconstructs_actual_action_credit": exact_reconstructed,
        "first_order_proxy_has_next_step_signal": first_order_signal,
        "singleton_verdict": singleton_verdict,
        "verdict": verdict,
        "group_attribution": {
            "overall": group_overall,
            "verdict": group_verdict,
            "claim": (
                "exposed fixed-Ridge grouped attribution instrument diagnostic only"
            ),
        },
        "action_value_guard_budget_diagnostic": {
            "scientific_role": (
                "exposed flatline Program-conditioned ActionValueGuard mechanism test; "
                "not Capability, Memory, Promotion, or transfer evidence"
            ),
            "hypotheses": {
                "h0": "always mask every flatline detected by the frozen Observation",
                "h1": (
                    "propose detected flatline actions with support singleton first-order "
                    "credit > 0; execute the exact grouped cohort iff exact support gain > 0"
                ),
            },
            "support_query_splits": {
                "B=0": "no support; query all 8 evaluation series",
                "B=1": "8 singleton supports; query the other 7 series",
                "B=2": (
                    "4 fixed supports (0,1),(2,3),(4,5),(6,7); query the other 6"
                ),
                "B=4": (
                    "2 alternating supports (0,2,4,6),(1,3,5,7); query the other 4"
                ),
            },
            "proxy_vs_historical_exact_singleton_credit": {
                "fold_credit_count": overall_summary["fold_credit_count"],
                "exact_vs_historical_max_abs_error": overall_summary[
                    "exact_vs_actual_max_abs_error"
                ],
                "first_order_vs_historical_sign_accuracy": overall_summary[
                    "sign_agreement_accuracy"
                ],
                "first_order_vs_historical_balanced_accuracy": overall_summary[
                    "balanced_accuracy"
                ],
                "dataset_macro_balanced_accuracy": overall_summary[
                    "dataset_macro_balanced_accuracy"
                ],
            },
            "dataset_evidence": [
                {
                    "dataset_id": row["dataset_id"],
                    **row["action_value_guard_budget_diagnostic"],
                    "split_evidence": {
                        str(budget): [
                            {
                                "seed": seed_row["seed"],
                                "split_id": split["split_id"],
                                "support_uid_indices": split[
                                    "support_uid_indices"
                                ],
                                "query_uid_indices": split["query_uid_indices"],
                                "h0_query_gain": split["h0_query_gain"],
                                "h1_proposed_count": split.get(
                                    "h1_proposed_count", 0
                                ),
                                "h1_exact_grouped_support_gain": split[
                                    "h1_exact_grouped_support_gain"
                                ],
                                "h1_guard_decision": split[
                                    "h1_guard_decision"
                                ],
                                "h1_guarded_query_gain": split[
                                    "h1_guarded_query_gain"
                                ],
                                **(
                                    {
                                        "action__support_exact_singleton_sign_coherence": (
                                            split[
                                                "action__support_exact_singleton_sign_coherence"
                                            ]
                                        )
                                    }
                                    if capture_support_action_response and budget > 0
                                    else {}
                                ),
                            }
                            for seed_row in row["seed_evidence"]
                            for split in seed_row["action_value_guard_budget"][
                                str(budget)
                            ]["split_evidence"]
                        ]
                        for budget in (0, 1, 2, 4)
                    },
                }
                for row in dataset_evidence
            ],
            "budget_summary": action_value_budget_overall,
            "dataset_macro_h0_adapt_auc_budget_grid_mean": h0_macro_adapt_auc,
            "dataset_macro_h1_adapt_auc_budget_grid_mean": h1_macro_adapt_auc,
            "h1_minus_h0_dataset_macro_adapt_auc": (
                h1_macro_adapt_auc - h0_macro_adapt_auc
            ),
            "beneficial_retention_budgets": [1, 2, 4],
            "h0_beneficial_gain_sum": h0_beneficial_gain,
            "h1_nonnegative_gain_on_h0_beneficial_splits": (
                h1_retained_beneficial_gain
            ),
            "beneficial_gain_retention_fraction": (
                h1_retained_beneficial_gain / h0_beneficial_gain
                if h0_beneficial_gain > 0.0
                else None
            ),
            "h0_beneficial_split_count": h0_beneficial_split_count,
            "retained_h0_beneficial_split_count": (
                h1_retained_beneficial_split_count
            ),
            "beneficial_split_retention_fraction": (
                h1_retained_beneficial_split_count / h0_beneficial_split_count
                if h0_beneficial_split_count
                else None
            ),
            "low_budget_h0_harmful_split_count": low_budget_h0_harmful_splits,
            "low_budget_h1_harmful_split_count": low_budget_h1_harmful_splits,
            "low_budget_h0_harmful_dataset_count_sum": (
                low_budget_h0_harmful_datasets
            ),
            "low_budget_h1_harmful_dataset_count_sum": (
                low_budget_h1_harmful_datasets
            ),
            "h1_reduces_low_budget_harmful_splits": (
                low_budget_h1_harmful_splits < low_budget_h0_harmful_splits
            ),
            "h1_reduces_low_budget_harmful_datasets": (
                low_budget_h1_harmful_datasets < low_budget_h0_harmful_datasets
            ),
            "h1_adapt_auc_not_lower_than_h0": h1_macro_adapt_auc >= h0_macro_adapt_auc,
            "compute_accounting": {
                "reference_system_construction_count": (
                    reference_system_construction_count
                ),
                "reference_solve_count": reference_solve_count,
                "h1_grouped_small_matrix_solve_count": (
                    action_value_budget_grouped_small_matrix_solve_count
                ),
                "per_action_consumer_refit_count": 0,
                "grouped_consumer_refit_count": 0,
            },
        },
        "compute_accounting": {
            "reference_system_construction_count": reference_system_construction_count,
            "reference_solve_count": reference_solve_count,
            "grouped_small_matrix_solve_count": grouped_small_matrix_solve_count,
            "action_value_budget_grouped_small_matrix_solve_count": (
                action_value_budget_grouped_small_matrix_solve_count
            ),
            "grouped_consumer_refit_count": 0,
            "per_action_refit_count": 0,
            "historical_action_credit_consumer_fit_count": historical_fit_count,
            "historical_action_count": len(all_fold_rows) // 2,
        },
        "capability_claim": False,
        "utility_supported": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Development-only, exposed Source attribution feasibility for one fixed "
            "Ridge system and flatline-mask actions. Exact singleton downdate and exact "
            "group Woodbury removal are mechanical calibrations. First-order credits "
            "are conditional attributions, not intrinsic row values; singleton credits "
            "cannot be added to infer joint policy value. This result does not establish "
            "general TimeInf behavior, utility support, Capability, Memory, or Transfer."
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
