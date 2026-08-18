"""Measure natural clean-data block-masking and soft-reweighting headroom.

This development-only census uses the four already exposed Source datasets and
one fixed Ridge reference per dataset.  It introduces no artificial corruption:
the sole Typed Program masks one training row from one fixed 12-step output
block.  Singleton first-order signs select cohorts on one evaluation fold;
exact grouped Ridge removal evaluates the combined four-block policy on the
disjoint fold.  Exact-support selection is reported only as an oracle diagnostic.

No Target/UCI data is read and no Capability, Memory, or transfer is claimed.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    RIDGE_ALPHA,
    _group_removal_predictions,
    _pearson,
    _ridge_reference_and_removal_predictions,
    _sign_metrics,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
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
    J0_REPORT_PATH,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-natural-block-action-value-headroom/3"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_natural_block_action_value_headroom_report.json"
)
BLOCKS = ((0, 12), (12, 24), (24, 36), (36, 48))
EXPECTED_TRAINING_ROWS = 72
EXPECTED_EVALUATION_ROWS = 8
SOFT_REMOVAL_STRENGTH = 0.25
FEEDBACK_BUDGET_SUPPORT_SPLITS = {
    1: tuple((index,) for index in range(EXPECTED_EVALUATION_ROWS)),
    2: ((0, 1), (2, 3), (4, 5), (6, 7)),
    4: ((0, 2, 4, 6), (1, 3, 5, 7)),
}


def _singleton_summary(np: Any, actions: list[dict[str, object]]) -> dict[str, object]:
    exact: list[float] = []
    proxy: list[float] = []
    exact_fold_agreement = 0
    proxy_fold_agreement = 0
    for action in actions:
        folds = action["fold_attribution"]
        exact_a = float(folds["fold_a"]["exact_singleton_attribution_credit"])
        exact_b = float(folds["fold_b"]["exact_singleton_attribution_credit"])
        proxy_a = float(
            folds["fold_a"]["first_order_proxy_attribution_credit"]
        )
        proxy_b = float(
            folds["fold_b"]["first_order_proxy_attribution_credit"]
        )
        exact.extend((exact_a, exact_b))
        proxy.extend((proxy_a, proxy_b))
        exact_fold_agreement += (exact_a > 0.0) == (exact_b > 0.0)
        proxy_fold_agreement += (proxy_a > 0.0) == (proxy_b > 0.0)
    sign = _sign_metrics(np, exact, proxy)
    exact_negative_count = sum(value < 0.0 for value in exact)
    exact_zero_count = sum(value == 0.0 for value in exact)
    return {
        "action_count": len(actions),
        "fold_credit_count": len(exact),
        "exact_positive_fold_credit_count": sign["actual_positive_count"],
        "exact_nonpositive_fold_credit_count": sign["actual_nonpositive_count"],
        "exact_negative_fold_credit_count": exact_negative_count,
        "exact_zero_fold_credit_count": exact_zero_count,
        "exact_positive_fold_credit_fraction": sign["actual_positive_fraction"],
        "both_exact_sign_classes_present": sign[
            "both_actual_sign_classes_present"
        ],
        "exact_action_fold_sign_agreement_rate": exact_fold_agreement / len(actions),
        "first_order_action_fold_sign_agreement_rate": (
            proxy_fold_agreement / len(actions)
        ),
        "first_order_vs_exact_pearson": _pearson(np, exact, proxy),
        "first_order_vs_exact_sign_accuracy": sign["sign_agreement_accuracy"],
        "first_order_vs_exact_balanced_accuracy": sign["balanced_accuracy"],
        "exact_label_majority_baseline_accuracy": sign[
            "actual_label_majority_baseline_accuracy"
        ],
        "first_order_positive_fold_credit_fraction": sign[
            "proxy_positive_fraction"
        ],
    }


def _combined_proxy_summary(
    np: Any, policies: list[dict[str, object]]
) -> dict[str, object]:
    exact = [float(row["exact_holdout_gain"]) for row in policies]
    proxy = [float(row["first_order_proxy_holdout_gain"]) for row in policies]
    sign = _sign_metrics(np, exact, proxy)
    return {
        "direction_count": len(policies),
        "first_order_combined_vs_exact_pearson": _pearson(np, exact, proxy),
        "first_order_combined_vs_exact_sign_accuracy": sign[
            "sign_agreement_accuracy"
        ],
        "first_order_combined_vs_exact_balanced_accuracy": sign[
            "balanced_accuracy"
        ],
        "exact_positive_direction_count": sign["actual_positive_count"],
        "exact_nonpositive_direction_count": sign["actual_nonpositive_count"],
        "exact_label_majority_baseline_accuracy": sign[
            "actual_label_majority_baseline_accuracy"
        ],
        "mean_abs_gain_error": statistics.fmean(
            abs(left - right) for left, right in zip(exact, proxy)
        ),
        "max_abs_gain_error": max(
            abs(left - right) for left, right in zip(exact, proxy)
        ),
    }


def run(root: Path) -> dict[str, object]:
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

    j0_report = _read_object(root / J0_REPORT_PATH)
    if j0_report.get("target_query_opened") is not False:
        raise ValueError("J0 Target/Query boundary is not closed")
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
        raise ValueError("UCI is forbidden in this exposed Source diagnostic")

    specs = {**SPECS, **FRESH_SPECS}
    if len(specs) != 4 or len(roster) != 20 * len(specs):
        raise ValueError("expected four exposed 12-train/8-eval Source rosters")
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )

    reference_solve_count = 0
    grouped_small_matrix_solve_count = 0
    feedback_budget_grouped_small_matrix_solve_count = 0
    soft_feedback_budget_grouped_small_matrix_solve_count = 0
    dataset_evidence: list[dict[str, object]] = []
    all_proxy_policies: list[dict[str, object]] = []
    all_oracle_policies: list[dict[str, object]] = []

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
        if len(train_rows) != 12 or len(eval_rows) != EXPECTED_EVALUATION_ROWS:
            raise ValueError(f"exposed roster geometry changed: {dataset_id}")

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
                    raise ValueError(f"invalid clean training window: {uid}/{anchor}")
                x_rows.append(
                    np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
                )
                y_rows.append((target - center) / scale)
                row_keys.append((uid, anchor, HORIZON))
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_y = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (EXPECTED_TRAINING_ROWS, 384) or clean_y.shape != (
            EXPECTED_TRAINING_ROWS,
            HORIZON,
        ):
            raise AssertionError(f"unexpected clean training geometry: {dataset_id}")

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
            if prediction.shape != (EXPECTED_EVALUATION_ROWS, HORIZON):
                raise RuntimeError(f"invalid Ridge prediction shape: {dataset_id}")
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"non-finite Ridge prediction: {dataset_id}")
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

        candidate_rows = tuple(range(EXPECTED_TRAINING_ROWS))
        reference = _ridge_reference_and_removal_predictions(
            np,
            x_train=x_train,
            targets=clean_y,
            x_eval=x_eval_array,
            candidate_rows=candidate_rows,
            target_block=(0, HORIZON),
            alpha=RIDGE_ALPHA,
        )
        reference_solve_count += 1
        baseline_prediction = np.asarray(
            reference["baseline_prediction"], dtype=np.float64
        )
        baseline_losses = score_predictions(baseline_prediction)
        leverage = np.asarray(reference["leverage"], dtype=np.float64)
        actions_by_block: dict[tuple[int, int], list[dict[str, object]]] = {}
        all_actions: list[dict[str, object]] = []

        for block in BLOCKS:
            start, stop = block
            block_actions: list[dict[str, object]] = []
            for row_index in candidate_rows:
                exact_prediction = baseline_prediction.copy()
                exact_prediction[:, start:stop] = reference[
                    "exact_removal_predictions"
                ][row_index, :, start:stop]
                proxy_prediction = baseline_prediction.copy()
                proxy_prediction[:, start:stop] = reference[
                    "first_order_proxy_predictions"
                ][row_index, :, start:stop]
                exact_losses = score_predictions(exact_prediction)
                proxy_losses = score_predictions(proxy_prediction)
                exact_series_credit = baseline_losses - exact_losses
                proxy_series_credit = baseline_losses - proxy_losses
                fold_attribution = {
                    fold_name: {
                        "exact_singleton_attribution_credit": _mean(
                            baseline_losses - exact_losses, fold_indices
                        ),
                        "first_order_proxy_attribution_credit": _mean(
                            baseline_losses - proxy_losses, fold_indices
                        ),
                    }
                    for fold_name, fold_indices in folds.items()
                }
                action = {
                    "row_index": row_index,
                    "row_key": list(row_keys[row_index]),
                    "block_half_open": list(block),
                    "leverage": float(leverage[row_index]),
                    "fold_attribution": fold_attribution,
                    "per_evaluation_series_exact_singleton_credit": [
                        float(value) for value in exact_series_credit
                    ],
                    "per_evaluation_series_first_order_proxy_credit": [
                        float(value) for value in proxy_series_credit
                    ],
                }
                block_actions.append(action)
                all_actions.append(action)
            actions_by_block[block] = block_actions

        direction_evidence: list[dict[str, object]] = []
        fold_pairs = (
            ("a_to_b", "fold_a", "fold_b"),
            ("b_to_a", "fold_b", "fold_a"),
        )
        for direction, support_name, holdout_name in fold_pairs:
            policies: dict[str, dict[str, object]] = {}
            for policy_name, support_credit_key in (
                ("singleton_first_order_proxy_guided", "first_order_proxy_attribution_credit"),
                ("singleton_exact_oracle_guided", "exact_singleton_attribution_credit"),
            ):
                exact_combined = baseline_prediction.copy()
                proxy_combined = baseline_prediction.copy()
                block_evidence: list[dict[str, object]] = []
                selected_total = 0
                guard_count = 0
                policy_group_solve_count = 0
                for block in BLOCKS:
                    block_actions = actions_by_block[block]
                    support_values = [
                        float(action["fold_attribution"][support_name][support_credit_key])
                        for action in block_actions
                    ]
                    selected = [
                        index for index, value in enumerate(support_values) if value > 0.0
                    ]
                    guard_triggered = len(selected) == EXPECTED_TRAINING_ROWS
                    guard_retained_row_index = None
                    if guard_triggered:
                        guard_retained_row_index = min(
                            range(EXPECTED_TRAINING_ROWS),
                            key=lambda index: (support_values[index], index),
                        )
                        selected.remove(guard_retained_row_index)
                        guard_count += 1
                    grouped = _group_removal_predictions(
                        np,
                        reference=reference,
                        selected_local_indices=tuple(selected),
                        target_block=block,
                    )
                    group_solves = int(grouped["small_matrix_solve_count"])
                    grouped_small_matrix_solve_count += group_solves
                    policy_group_solve_count += group_solves
                    selected_total += len(selected)
                    start, stop = block
                    exact_combined[:, start:stop] = grouped[
                        "exact_group_prediction"
                    ][:, start:stop]
                    proxy_combined[:, start:stop] = grouped[
                        "first_order_group_proxy_prediction"
                    ][:, start:stop]
                    block_exact_losses = score_predictions(
                        grouped["exact_group_prediction"]
                    )
                    block_proxy_losses = score_predictions(
                        grouped["first_order_group_proxy_prediction"]
                    )
                    block_evidence.append(
                        {
                            "block_half_open": list(block),
                            "selected_row_indices": selected,
                            "selected_count": len(selected),
                            "selected_fraction": len(selected) / EXPECTED_TRAINING_ROWS,
                            "empty_group_keeps_baseline": len(selected) == 0,
                            "all_rows_guard_triggered": guard_triggered,
                            "guard_retained_row_index": guard_retained_row_index,
                            "small_matrix_solve_count": group_solves,
                            "middle_condition_number": grouped[
                                "middle_condition_number"
                            ],
                            "exact_support_gain": _mean(
                                baseline_losses - block_exact_losses,
                                folds[support_name],
                            ),
                            "exact_holdout_gain": _mean(
                                baseline_losses - block_exact_losses,
                                folds[holdout_name],
                            ),
                            "first_order_proxy_support_gain": _mean(
                                baseline_losses - block_proxy_losses,
                                folds[support_name],
                            ),
                            "first_order_proxy_holdout_gain": _mean(
                                baseline_losses - block_proxy_losses,
                                folds[holdout_name],
                            ),
                        }
                    )
                exact_combined_losses = score_predictions(exact_combined)
                proxy_combined_losses = score_predictions(proxy_combined)
                policy = {
                    "policy": policy_name,
                    "support_credit_threshold": 0.0,
                    "support_credit_key": support_credit_key,
                    "selected_action_count": selected_total,
                    "selected_action_fraction": selected_total
                    / (EXPECTED_TRAINING_ROWS * len(BLOCKS)),
                    "empty_block_group_count": sum(
                        bool(row["empty_group_keeps_baseline"])
                        for row in block_evidence
                    ),
                    "all_rows_guard_trigger_count": guard_count,
                    "small_matrix_solve_count": policy_group_solve_count,
                    "exact_support_gain": _mean(
                        baseline_losses - exact_combined_losses,
                        folds[support_name],
                    ),
                    "exact_holdout_gain": _mean(
                        baseline_losses - exact_combined_losses,
                        folds[holdout_name],
                    ),
                    "first_order_proxy_support_gain": _mean(
                        baseline_losses - proxy_combined_losses,
                        folds[support_name],
                    ),
                    "first_order_proxy_holdout_gain": _mean(
                        baseline_losses - proxy_combined_losses,
                        folds[holdout_name],
                    ),
                    "block_attribution": block_evidence,
                }
                if policy_name == "singleton_first_order_proxy_guided":
                    exact_support_gain = float(policy["exact_support_gain"])
                    raw_exact_holdout_gain = float(policy["exact_holdout_gain"])
                    guard_executes = exact_support_gain > 0.0
                    policy["cohort_risk_guard"] = {
                        "evidence": "proxy-guided cohort exact grouped support gain",
                        "threshold": 0.0,
                        "decision": "EXECUTE" if guard_executes else "ABSTAIN",
                        "raw_exact_holdout_gain": raw_exact_holdout_gain,
                        "guarded_exact_holdout_gain": (
                            raw_exact_holdout_gain if guard_executes else 0.0
                        ),
                        "abstain_semantics": (
                            "keep the clean Ridge baseline on all four blocks"
                        ),
                    }
                policies[policy_name] = policy
                if policy_name == "singleton_first_order_proxy_guided":
                    all_proxy_policies.append(
                        {"dataset_id": dataset_id, "direction": direction, **policy}
                    )
                else:
                    all_oracle_policies.append(
                        {"dataset_id": dataset_id, "direction": direction, **policy}
                    )
            direction_evidence.append(
                {
                    "direction": direction,
                    "support_fold": support_name,
                    "holdout_fold": holdout_name,
                    "proxy_guided": policies["singleton_first_order_proxy_guided"],
                    "oracle_diagnostic": policies["singleton_exact_oracle_guided"],
                }
            )

        proxy_policies = [row["proxy_guided"] for row in direction_evidence]
        oracle_policies = [row["oracle_diagnostic"] for row in direction_evidence]
        singleton_summary = _singleton_summary(np, all_actions)
        dataset_summary = {
            "proxy_guided_exact_holdout_mean_gain": statistics.fmean(
                float(row["exact_holdout_gain"]) for row in proxy_policies
            ),
            "proxy_guided_exact_positive_direction_count": sum(
                float(row["exact_holdout_gain"]) > 0.0 for row in proxy_policies
            ),
            "proxy_guided_exact_harmful_direction_count": sum(
                float(row["exact_holdout_gain"]) < 0.0 for row in proxy_policies
            ),
            "proxy_guarded_exact_holdout_mean_gain": statistics.fmean(
                float(row["cohort_risk_guard"]["guarded_exact_holdout_gain"])
                for row in proxy_policies
            ),
            "proxy_guarded_exact_positive_direction_count": sum(
                float(row["cohort_risk_guard"]["guarded_exact_holdout_gain"])
                > 0.0
                for row in proxy_policies
            ),
            "proxy_guarded_exact_harmful_direction_count": sum(
                float(row["cohort_risk_guard"]["guarded_exact_holdout_gain"])
                < 0.0
                for row in proxy_policies
            ),
            "proxy_guarded_abstained_direction_count": sum(
                row["cohort_risk_guard"]["decision"] == "ABSTAIN"
                for row in proxy_policies
            ),
            "proxy_guided_mean_selected_action_fraction": statistics.fmean(
                float(row["selected_action_fraction"]) for row in proxy_policies
            ),
            "proxy_guided_all_rows_guard_trigger_count": sum(
                int(row["all_rows_guard_trigger_count"]) for row in proxy_policies
            ),
            "oracle_guided_exact_holdout_mean_gain": statistics.fmean(
                float(row["exact_holdout_gain"]) for row in oracle_policies
            ),
            "oracle_guided_exact_positive_direction_count": sum(
                float(row["exact_holdout_gain"]) > 0.0 for row in oracle_policies
            ),
            "oracle_guided_exact_harmful_direction_count": sum(
                float(row["exact_holdout_gain"]) < 0.0 for row in oracle_policies
            ),
            "oracle_guided_mean_selected_action_fraction": statistics.fmean(
                float(row["selected_action_fraction"]) for row in oracle_policies
            ),
            "oracle_guided_all_rows_guard_trigger_count": sum(
                int(row["all_rows_guard_trigger_count"]) for row in oracle_policies
            ),
            "proxy_combined_attribution_diagnostic": _combined_proxy_summary(
                np, proxy_policies
            ),
        }
        feedback_budget_evidence: dict[str, dict[str, object]] = {
            "0": {
                "support_series_budget": 0,
                "split_count": 1,
                "split_evidence": [
                    {
                        "split_id": "b0_baseline",
                        "support_uid_indices": [],
                        "query_uid_indices": list(range(EXPECTED_EVALUATION_ROWS)),
                        "selected_action_count": 0,
                        "selected_action_fraction": 0.0,
                        "exact_grouped_support_gain": None,
                        "guard_decision": "ABSTAIN_NO_FEEDBACK",
                        "raw_exact_grouped_query_gain": 0.0,
                        "guarded_exact_grouped_query_gain": 0.0,
                        "block_proposals": [],
                    }
                ],
                "summary": {
                    "mean_raw_exact_query_gain": 0.0,
                    "mean_guarded_exact_query_gain": 0.0,
                    "positive_split_count": 0,
                    "harmful_split_count": 0,
                    "abstained_split_count": 1,
                    "mean_selected_action_fraction": 0.0,
                    "all_rows_guard_trigger_count": 0,
                    "empty_block_group_count": 4,
                },
            }
        }
        soft_feedback_budget_evidence: dict[str, dict[str, object]] = {
            "0": {
                "support_series_budget": 0,
                "split_count": 1,
                "split_evidence": [
                    {
                        "split_id": "b0_baseline",
                        "support_uid_indices": [],
                        "query_uid_indices": list(range(EXPECTED_EVALUATION_ROWS)),
                        "selected_action_count": 0,
                        "selected_action_fraction": 0.0,
                        "exact_grouped_support_gain": None,
                        "guard_decision": "ABSTAIN_NO_FEEDBACK",
                        "raw_exact_grouped_query_gain": 0.0,
                        "guarded_exact_grouped_query_gain": 0.0,
                        "block_proposals": [],
                    }
                ],
                "summary": {
                    "mean_raw_exact_query_gain": 0.0,
                    "mean_guarded_exact_query_gain": 0.0,
                    "positive_split_count": 0,
                    "harmful_split_count": 0,
                    "abstained_split_count": 1,
                    "mean_selected_action_fraction": 0.0,
                    "all_rows_guard_trigger_count": 0,
                    "empty_block_group_count": 4,
                },
            }
        }
        dataset_budget_group_solve_count = 0
        dataset_soft_budget_group_solve_count = 0
        for budget, support_splits in FEEDBACK_BUDGET_SUPPORT_SPLITS.items():
            split_evidence: list[dict[str, object]] = []
            soft_split_evidence: list[dict[str, object]] = []
            for split_index, support_indices in enumerate(support_splits):
                query_indices = tuple(
                    index
                    for index in range(EXPECTED_EVALUATION_ROWS)
                    if index not in support_indices
                )
                if len(support_indices) != budget or len(query_indices) != (
                    EXPECTED_EVALUATION_ROWS - budget
                ):
                    raise AssertionError("feedback-budget split geometry changed")
                exact_combined = baseline_prediction.copy()
                soft_exact_combined = baseline_prediction.copy()
                selected_total = 0
                all_rows_guard_count = 0
                empty_block_group_count = 0
                split_group_solve_count = 0
                soft_split_group_solve_count = 0
                block_proposals: list[dict[str, object]] = []
                soft_block_proposals: list[dict[str, object]] = []
                for block in BLOCKS:
                    block_actions = actions_by_block[block]
                    support_proxy_credit = [
                        statistics.fmean(
                            float(
                                action[
                                    "per_evaluation_series_first_order_proxy_credit"
                                ][index]
                            )
                            for index in support_indices
                        )
                        for action in block_actions
                    ]
                    selected = [
                        index
                        for index, value in enumerate(support_proxy_credit)
                        if value > 0.0
                    ]
                    all_rows_guard_triggered = (
                        len(selected) == EXPECTED_TRAINING_ROWS
                    )
                    guard_retained_row_index = None
                    if all_rows_guard_triggered:
                        guard_retained_row_index = min(
                            range(EXPECTED_TRAINING_ROWS),
                            key=lambda index: (support_proxy_credit[index], index),
                        )
                        selected.remove(guard_retained_row_index)
                        all_rows_guard_count += 1
                    grouped = _group_removal_predictions(
                        np,
                        reference=reference,
                        selected_local_indices=tuple(selected),
                        target_block=block,
                    )
                    group_solve_count = int(grouped["small_matrix_solve_count"])
                    split_group_solve_count += group_solve_count
                    dataset_budget_group_solve_count += group_solve_count
                    feedback_budget_grouped_small_matrix_solve_count += (
                        group_solve_count
                    )
                    soft_grouped = _group_removal_predictions(
                        np,
                        reference=reference,
                        selected_local_indices=tuple(selected),
                        target_block=block,
                        removal_strength=SOFT_REMOVAL_STRENGTH,
                    )
                    soft_group_solve_count = int(
                        soft_grouped["small_matrix_solve_count"]
                    )
                    soft_split_group_solve_count += soft_group_solve_count
                    dataset_soft_budget_group_solve_count += soft_group_solve_count
                    soft_feedback_budget_grouped_small_matrix_solve_count += (
                        soft_group_solve_count
                    )
                    selected_total += len(selected)
                    empty_block_group_count += len(selected) == 0
                    start, stop = block
                    exact_combined[:, start:stop] = grouped[
                        "exact_group_prediction"
                    ][:, start:stop]
                    soft_exact_combined[:, start:stop] = soft_grouped[
                        "exact_group_prediction"
                    ][:, start:stop]
                    block_proposals.append(
                        {
                            "block_half_open": list(block),
                            "selected_row_indices": selected,
                            "selected_count": len(selected),
                            "selected_fraction": len(selected)
                            / EXPECTED_TRAINING_ROWS,
                            "empty_group_keeps_baseline": len(selected) == 0,
                            "all_rows_guard_triggered": all_rows_guard_triggered,
                            "guard_retained_row_index": guard_retained_row_index,
                            "small_matrix_solve_count": group_solve_count,
                            "middle_condition_number": grouped[
                                "middle_condition_number"
                            ],
                        }
                    )
                    soft_block_proposals.append(
                        {
                            "block_half_open": list(block),
                            "selected_row_indices": selected,
                            "selected_count": len(selected),
                            "selected_fraction": len(selected)
                            / EXPECTED_TRAINING_ROWS,
                            "empty_group_keeps_baseline": len(selected) == 0,
                            "all_rows_guard_triggered": all_rows_guard_triggered,
                            "guard_retained_row_index": guard_retained_row_index,
                            "small_matrix_solve_count": soft_group_solve_count,
                            "middle_condition_number": soft_grouped[
                                "middle_condition_number"
                            ],
                        }
                    )
                exact_combined_losses = score_predictions(exact_combined)
                soft_exact_combined_losses = score_predictions(soft_exact_combined)
                exact_support_gain = _mean(
                    baseline_losses - exact_combined_losses,
                    tuple(support_indices),
                )
                soft_exact_support_gain = _mean(
                    baseline_losses - soft_exact_combined_losses,
                    tuple(support_indices),
                )
                raw_query_gain = _mean(
                    baseline_losses - exact_combined_losses,
                    query_indices,
                )
                soft_raw_query_gain = _mean(
                    baseline_losses - soft_exact_combined_losses,
                    query_indices,
                )
                guard_executes = exact_support_gain > 0.0
                soft_guard_executes = soft_exact_support_gain > 0.0
                split_evidence.append(
                    {
                        "split_id": f"b{budget}_split_{split_index}",
                        "support_uid_indices": list(support_indices),
                        "query_uid_indices": list(query_indices),
                        "selected_action_count": selected_total,
                        "selected_action_fraction": selected_total
                        / (EXPECTED_TRAINING_ROWS * len(BLOCKS)),
                        "exact_grouped_support_gain": exact_support_gain,
                        "guard_threshold": 0.0,
                        "guard_decision": (
                            "EXECUTE" if guard_executes else "ABSTAIN"
                        ),
                        "raw_exact_grouped_query_gain": raw_query_gain,
                        "guarded_exact_grouped_query_gain": (
                            raw_query_gain if guard_executes else 0.0
                        ),
                        "grouped_small_matrix_solve_count": (
                            split_group_solve_count
                        ),
                        "all_rows_guard_trigger_count": all_rows_guard_count,
                        "empty_block_group_count": empty_block_group_count,
                        "block_proposals": block_proposals,
                    }
                )
                soft_split_evidence.append(
                    {
                        "split_id": f"b{budget}_split_{split_index}",
                        "support_uid_indices": list(support_indices),
                        "query_uid_indices": list(query_indices),
                        "selected_action_count": selected_total,
                        "selected_action_fraction": selected_total
                        / (EXPECTED_TRAINING_ROWS * len(BLOCKS)),
                        "exact_grouped_support_gain": soft_exact_support_gain,
                        "guard_threshold": 0.0,
                        "guard_decision": (
                            "EXECUTE" if soft_guard_executes else "ABSTAIN"
                        ),
                        "raw_exact_grouped_query_gain": soft_raw_query_gain,
                        "guarded_exact_grouped_query_gain": (
                            soft_raw_query_gain if soft_guard_executes else 0.0
                        ),
                        "grouped_small_matrix_solve_count": (
                            soft_split_group_solve_count
                        ),
                        "all_rows_guard_trigger_count": all_rows_guard_count,
                        "empty_block_group_count": empty_block_group_count,
                        "block_proposals": soft_block_proposals,
                    }
                )
            guarded_query_gain = [
                float(row["guarded_exact_grouped_query_gain"])
                for row in split_evidence
            ]
            feedback_budget_evidence[str(budget)] = {
                "support_series_budget": budget,
                "split_count": len(split_evidence),
                "split_evidence": split_evidence,
                "summary": {
                    "mean_raw_exact_query_gain": statistics.fmean(
                        float(row["raw_exact_grouped_query_gain"])
                        for row in split_evidence
                    ),
                    "mean_guarded_exact_query_gain": statistics.fmean(
                        guarded_query_gain
                    ),
                    "positive_split_count": sum(
                        value > 0.0 for value in guarded_query_gain
                    ),
                    "harmful_split_count": sum(
                        value < 0.0 for value in guarded_query_gain
                    ),
                    "abstained_split_count": sum(
                        row["guard_decision"] != "EXECUTE"
                        for row in split_evidence
                    ),
                    "mean_selected_action_fraction": statistics.fmean(
                        float(row["selected_action_fraction"])
                        for row in split_evidence
                    ),
                    "all_rows_guard_trigger_count": sum(
                        int(row["all_rows_guard_trigger_count"])
                        for row in split_evidence
                    ),
                    "empty_block_group_count": sum(
                        int(row["empty_block_group_count"])
                        for row in split_evidence
                    ),
                },
            }
            soft_guarded_query_gain = [
                float(row["guarded_exact_grouped_query_gain"])
                for row in soft_split_evidence
            ]
            soft_feedback_budget_evidence[str(budget)] = {
                "support_series_budget": budget,
                "split_count": len(soft_split_evidence),
                "split_evidence": soft_split_evidence,
                "summary": {
                    "mean_raw_exact_query_gain": statistics.fmean(
                        float(row["raw_exact_grouped_query_gain"])
                        for row in soft_split_evidence
                    ),
                    "mean_guarded_exact_query_gain": statistics.fmean(
                        soft_guarded_query_gain
                    ),
                    "positive_split_count": sum(
                        value > 0.0 for value in soft_guarded_query_gain
                    ),
                    "harmful_split_count": sum(
                        value < 0.0 for value in soft_guarded_query_gain
                    ),
                    "abstained_split_count": sum(
                        row["guard_decision"] != "EXECUTE"
                        for row in soft_split_evidence
                    ),
                    "mean_selected_action_fraction": statistics.fmean(
                        float(row["selected_action_fraction"])
                        for row in soft_split_evidence
                    ),
                    "all_rows_guard_trigger_count": sum(
                        int(row["all_rows_guard_trigger_count"])
                        for row in soft_split_evidence
                    ),
                    "empty_block_group_count": sum(
                        int(row["empty_block_group_count"])
                        for row in soft_split_evidence
                    ),
                },
            }
        budget_grid_gains = [
            float(
                feedback_budget_evidence[str(budget)]["summary"][
                    "mean_guarded_exact_query_gain"
                ]
            )
            for budget in (0, 1, 2, 4)
        ]
        feedback_budget_diagnostic = {
            "budgets": feedback_budget_evidence,
            "adapt_auc_budget_grid_mean": statistics.fmean(budget_grid_gains),
            "adapt_auc_semantics": (
                "unweighted mean of exposed target-only guarded exact query gains "
                "at B=0,1,2,4; not continuous-area or transfer evidence"
            ),
            "grouped_small_matrix_solve_count": dataset_budget_group_solve_count,
        }
        soft_budget_grid_gains = [
            float(
                soft_feedback_budget_evidence[str(budget)]["summary"][
                    "mean_guarded_exact_query_gain"
                ]
            )
            for budget in (0, 1, 2, 4)
        ]
        soft_feedback_budget_diagnostic = {
            "removal_strength": SOFT_REMOVAL_STRENGTH,
            "selected_row_weight_before": 1.0,
            "selected_row_weight_after": 1.0 - SOFT_REMOVAL_STRENGTH,
            "budgets": soft_feedback_budget_evidence,
            "adapt_auc_budget_grid_mean": statistics.fmean(
                soft_budget_grid_gains
            ),
            "adapt_auc_semantics": (
                "unweighted mean of exposed target-only guarded exact query gains "
                "at B=0,1,2,4; not continuous-area or transfer evidence"
            ),
            "grouped_small_matrix_solve_count": (
                dataset_soft_budget_group_solve_count
            ),
        }
        dataset_budget_comparison: dict[str, dict[str, object]] = {}
        for budget in (0, 1, 2, 4):
            budget_key = str(budget)
            hard_budget = feedback_budget_evidence[budget_key]
            soft_budget = soft_feedback_budget_evidence[budget_key]
            hard_split_rows = hard_budget["split_evidence"]
            soft_split_rows = soft_budget["split_evidence"]
            if [row["split_id"] for row in hard_split_rows] != [
                row["split_id"] for row in soft_split_rows
            ]:
                raise AssertionError("hard/soft feedback split identity changed")
            hard_positive_pairs = [
                (float(hard_row["guarded_exact_grouped_query_gain"]), soft_row)
                for hard_row, soft_row in zip(hard_split_rows, soft_split_rows)
                if float(hard_row["guarded_exact_grouped_query_gain"]) > 0.0
            ]
            hard_beneficial_gain = sum(value for value, _ in hard_positive_pairs)
            retained_soft_beneficial_gain = sum(
                max(float(row["guarded_exact_grouped_query_gain"]), 0.0)
                for _, row in hard_positive_pairs
            )
            dataset_budget_comparison[budget_key] = {
                "support_series_budget": budget,
                "hard_mean_guarded_exact_query_gain": hard_budget["summary"][
                    "mean_guarded_exact_query_gain"
                ],
                "soft_mean_guarded_exact_query_gain": soft_budget["summary"][
                    "mean_guarded_exact_query_gain"
                ],
                "hard_harmful_dataset": float(
                    hard_budget["summary"]["mean_guarded_exact_query_gain"]
                )
                < 0.0,
                "soft_harmful_dataset": float(
                    soft_budget["summary"]["mean_guarded_exact_query_gain"]
                )
                < 0.0,
                "hard_positive_split_count": hard_budget["summary"][
                    "positive_split_count"
                ],
                "soft_positive_split_count": soft_budget["summary"][
                    "positive_split_count"
                ],
                "hard_harmful_split_count": hard_budget["summary"][
                    "harmful_split_count"
                ],
                "soft_harmful_split_count": soft_budget["summary"][
                    "harmful_split_count"
                ],
                "hard_abstained_split_count": hard_budget["summary"][
                    "abstained_split_count"
                ],
                "soft_abstained_split_count": soft_budget["summary"][
                    "abstained_split_count"
                ],
                "hard_beneficial_gain_sum": hard_beneficial_gain,
                "soft_nonnegative_gain_on_hard_beneficial_splits": (
                    retained_soft_beneficial_gain
                ),
                "beneficial_gain_retention_fraction": (
                    retained_soft_beneficial_gain / hard_beneficial_gain
                    if hard_beneficial_gain > 0.0
                    else None
                ),
                "beneficial_split_retention_fraction": (
                    sum(
                        float(row["guarded_exact_grouped_query_gain"]) > 0.0
                        for _, row in hard_positive_pairs
                    )
                    / len(hard_positive_pairs)
                    if hard_positive_pairs
                    else None
                ),
            }
        hard_dataset_adapt_auc = float(
            feedback_budget_diagnostic["adapt_auc_budget_grid_mean"]
        )
        soft_dataset_adapt_auc = float(
            soft_feedback_budget_diagnostic["adapt_auc_budget_grid_mean"]
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "evaluation_uids": eval_uids,
                "folds": {key: list(value) for key, value in folds.items()},
                "reference_solve_count": 1,
                "per_action_refit_count": 0,
                "singleton_action_attribution": all_actions,
                "singleton_attribution_summary": singleton_summary,
                "crossfit_group_policy": direction_evidence,
                "summary": dataset_summary,
                "target_only_feedback_budget_diagnostic": (
                    feedback_budget_diagnostic
                ),
                "soft_temporal_block_reweighting_budget_diagnostic": (
                    soft_feedback_budget_diagnostic
                ),
                "hard_vs_soft_budget_comparison": {
                    "budgets": dataset_budget_comparison,
                    "hard_adapt_auc_budget_grid_mean": hard_dataset_adapt_auc,
                    "soft_adapt_auc_budget_grid_mean": soft_dataset_adapt_auc,
                    "soft_minus_hard_adapt_auc": (
                        soft_dataset_adapt_auc - hard_dataset_adapt_auc
                    ),
                },
            }
        )

    singleton_summaries = [row["singleton_attribution_summary"] for row in dataset_evidence]
    dataset_summaries = [row["summary"] for row in dataset_evidence]
    proxy_dataset_gains = [
        float(row["proxy_guided_exact_holdout_mean_gain"])
        for row in dataset_summaries
    ]
    guarded_proxy_dataset_gains = [
        float(row["proxy_guarded_exact_holdout_mean_gain"])
        for row in dataset_summaries
    ]
    oracle_dataset_gains = [
        float(row["oracle_guided_exact_holdout_mean_gain"])
        for row in dataset_summaries
    ]
    exact_positive_count = sum(
        int(row["exact_positive_fold_credit_count"]) for row in singleton_summaries
    )
    exact_nonpositive_count = sum(
        int(row["exact_nonpositive_fold_credit_count"]) for row in singleton_summaries
    )
    exact_negative_count = sum(
        int(row["exact_negative_fold_credit_count"]) for row in singleton_summaries
    )
    exact_zero_count = sum(
        int(row["exact_zero_fold_credit_count"]) for row in singleton_summaries
    )
    available_singleton_ba = [
        float(row["first_order_vs_exact_balanced_accuracy"])
        for row in singleton_summaries
        if row["first_order_vs_exact_balanced_accuracy"] is not None
    ]
    proxy_positive_dataset_count = sum(value > 0.0 for value in proxy_dataset_gains)
    proxy_macro_gain = statistics.fmean(proxy_dataset_gains)
    guarded_proxy_macro_gain = statistics.fmean(guarded_proxy_dataset_gains)
    guarded_proxy_positive_dataset_count = sum(
        value > 0.0 for value in guarded_proxy_dataset_gains
    )
    guarded_proxy_harmful_dataset_count = sum(
        value < 0.0 for value in guarded_proxy_dataset_gains
    )
    natural_heterogeneity = exact_positive_count > 0 and exact_negative_count > 0
    headroom_present = (
        natural_heterogeneity
        and proxy_macro_gain > 0.0
        and proxy_positive_dataset_count >= 2
    )
    guarded_headroom_present = (
        natural_heterogeneity
        and guarded_proxy_macro_gain > 0.0
        and guarded_proxy_positive_dataset_count >= 2
        and guarded_proxy_harmful_dataset_count == 0
    )
    feedback_budget_overall: dict[str, dict[str, object]] = {}
    for budget in (0, 1, 2, 4):
        budget_key = str(budget)
        dataset_budget_rows = [
            row["target_only_feedback_budget_diagnostic"]["budgets"][budget_key]
            for row in dataset_evidence
        ]
        dataset_query_gains = [
            float(row["summary"]["mean_guarded_exact_query_gain"])
            for row in dataset_budget_rows
        ]
        feedback_budget_overall[budget_key] = {
            "support_series_budget": budget,
            "dataset_macro_guarded_exact_query_gain": statistics.fmean(
                dataset_query_gains
            ),
            "positive_dataset_count": sum(
                value > 0.0 for value in dataset_query_gains
            ),
            "harmful_dataset_count": sum(
                value < 0.0 for value in dataset_query_gains
            ),
            "neutral_dataset_count": sum(
                value == 0.0 for value in dataset_query_gains
            ),
            "positive_split_count": sum(
                int(row["summary"]["positive_split_count"])
                for row in dataset_budget_rows
            ),
            "harmful_split_count": sum(
                int(row["summary"]["harmful_split_count"])
                for row in dataset_budget_rows
            ),
            "abstained_split_count": sum(
                int(row["summary"]["abstained_split_count"])
                for row in dataset_budget_rows
            ),
            "split_count": sum(
                int(row["split_count"]) for row in dataset_budget_rows
            ),
        }
    feedback_budget_adapt_auc = statistics.fmean(
        float(row["target_only_feedback_budget_diagnostic"][
            "adapt_auc_budget_grid_mean"
        ])
        for row in dataset_evidence
    )
    soft_feedback_budget_overall: dict[str, dict[str, object]] = {}
    hard_vs_soft_budget_overall: dict[str, dict[str, object]] = {}
    for budget in (0, 1, 2, 4):
        budget_key = str(budget)
        hard_dataset_rows = [
            row["target_only_feedback_budget_diagnostic"]["budgets"][budget_key]
            for row in dataset_evidence
        ]
        soft_dataset_rows = [
            row["soft_temporal_block_reweighting_budget_diagnostic"]["budgets"]
            [budget_key]
            for row in dataset_evidence
        ]
        hard_dataset_gains = [
            float(row["summary"]["mean_guarded_exact_query_gain"])
            for row in hard_dataset_rows
        ]
        soft_dataset_gains = [
            float(row["summary"]["mean_guarded_exact_query_gain"])
            for row in soft_dataset_rows
        ]
        soft_feedback_budget_overall[budget_key] = {
            "support_series_budget": budget,
            "dataset_macro_guarded_exact_query_gain": statistics.fmean(
                soft_dataset_gains
            ),
            "positive_dataset_count": sum(value > 0.0 for value in soft_dataset_gains),
            "harmful_dataset_count": sum(value < 0.0 for value in soft_dataset_gains),
            "neutral_dataset_count": sum(value == 0.0 for value in soft_dataset_gains),
            "positive_split_count": sum(
                int(row["summary"]["positive_split_count"])
                for row in soft_dataset_rows
            ),
            "harmful_split_count": sum(
                int(row["summary"]["harmful_split_count"])
                for row in soft_dataset_rows
            ),
            "abstained_split_count": sum(
                int(row["summary"]["abstained_split_count"])
                for row in soft_dataset_rows
            ),
            "split_count": sum(int(row["split_count"]) for row in soft_dataset_rows),
        }
        hard_soft_split_pairs = [
            (hard_split, soft_split)
            for hard_row, soft_row in zip(hard_dataset_rows, soft_dataset_rows)
            for hard_split, soft_split in zip(
                hard_row["split_evidence"], soft_row["split_evidence"]
            )
        ]
        if any(
            hard_row["split_id"] != soft_row["split_id"]
            for hard_row, soft_row in hard_soft_split_pairs
        ):
            raise AssertionError("hard/soft macro feedback split identity changed")
        hard_positive_pairs = [
            (hard_row, soft_row)
            for hard_row, soft_row in hard_soft_split_pairs
            if float(hard_row["guarded_exact_grouped_query_gain"]) > 0.0
        ]
        hard_beneficial_gain = sum(
            float(hard_row["guarded_exact_grouped_query_gain"])
            for hard_row, _ in hard_positive_pairs
        )
        retained_soft_beneficial_gain = sum(
            max(float(soft_row["guarded_exact_grouped_query_gain"]), 0.0)
            for _, soft_row in hard_positive_pairs
        )
        retained_beneficial_split_count = sum(
            float(soft_row["guarded_exact_grouped_query_gain"]) > 0.0
            for _, soft_row in hard_positive_pairs
        )
        hard_harmful_dataset_ids = [
            str(dataset_evidence[index]["dataset_id"])
            for index, value in enumerate(hard_dataset_gains)
            if value < 0.0
        ]
        soft_harmful_dataset_ids = [
            str(dataset_evidence[index]["dataset_id"])
            for index, value in enumerate(soft_dataset_gains)
            if value < 0.0
        ]
        hard_vs_soft_budget_overall[budget_key] = {
            "support_series_budget": budget,
            "hard_dataset_macro_guarded_exact_query_gain": statistics.fmean(
                hard_dataset_gains
            ),
            "soft_dataset_macro_guarded_exact_query_gain": statistics.fmean(
                soft_dataset_gains
            ),
            "soft_minus_hard_dataset_macro_query_gain": statistics.fmean(
                soft_dataset_gains
            )
            - statistics.fmean(hard_dataset_gains),
            "hard_positive_split_count": feedback_budget_overall[budget_key][
                "positive_split_count"
            ],
            "soft_positive_split_count": soft_feedback_budget_overall[budget_key][
                "positive_split_count"
            ],
            "hard_harmful_split_count": feedback_budget_overall[budget_key][
                "harmful_split_count"
            ],
            "soft_harmful_split_count": soft_feedback_budget_overall[budget_key][
                "harmful_split_count"
            ],
            "hard_harmful_split_gain_magnitude": -sum(
                min(float(row["guarded_exact_grouped_query_gain"]), 0.0)
                for row, _ in hard_soft_split_pairs
            ),
            "soft_harmful_split_gain_magnitude": -sum(
                min(float(row["guarded_exact_grouped_query_gain"]), 0.0)
                for _, row in hard_soft_split_pairs
            ),
            "hard_abstained_split_count": feedback_budget_overall[budget_key][
                "abstained_split_count"
            ],
            "soft_abstained_split_count": soft_feedback_budget_overall[budget_key][
                "abstained_split_count"
            ],
            "hard_harmful_dataset_ids": hard_harmful_dataset_ids,
            "soft_harmful_dataset_ids": soft_harmful_dataset_ids,
            "hard_beneficial_gain_sum": hard_beneficial_gain,
            "soft_nonnegative_gain_on_hard_beneficial_splits": (
                retained_soft_beneficial_gain
            ),
            "beneficial_gain_retention_fraction": (
                retained_soft_beneficial_gain / hard_beneficial_gain
                if hard_beneficial_gain > 0.0
                else None
            ),
            "hard_beneficial_split_count": len(hard_positive_pairs),
            "retained_hard_beneficial_split_count": (
                retained_beneficial_split_count
            ),
            "beneficial_split_retention_fraction": (
                retained_beneficial_split_count / len(hard_positive_pairs)
                if hard_positive_pairs
                else None
            ),
        }
    soft_feedback_budget_adapt_auc = statistics.fmean(
        float(row["soft_temporal_block_reweighting_budget_diagnostic"][
            "adapt_auc_budget_grid_mean"
        ])
        for row in dataset_evidence
    )
    soft_strictly_reduces_b1_b2_harm = all(
        int(hard_vs_soft_budget_overall[str(budget)]["soft_harmful_split_count"])
        < int(hard_vs_soft_budget_overall[str(budget)]["hard_harmful_split_count"])
        for budget in (1, 2)
    )
    soft_adapt_auc_not_lower = (
        soft_feedback_budget_adapt_auc >= feedback_budget_adapt_auc
    )
    retained_budgets = [hard_vs_soft_budget_overall[str(budget)] for budget in (1, 2, 4)]
    aggregate_hard_beneficial_gain = sum(
        float(row["hard_beneficial_gain_sum"]) for row in retained_budgets
    )
    aggregate_soft_retained_gain = sum(
        float(row["soft_nonnegative_gain_on_hard_beneficial_splits"])
        for row in retained_budgets
    )
    aggregate_hard_beneficial_split_count = sum(
        int(row["hard_beneficial_split_count"]) for row in retained_budgets
    )
    aggregate_retained_beneficial_split_count = sum(
        int(row["retained_hard_beneficial_split_count"]) for row in retained_budgets
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_source_natural_block_action_value_headroom",
        "question": (
            "Does clean natural Source training data contain signed row x block masking "
            "heterogeneity, and do singleton-proxy-selected cohorts have positive exact "
            "grouped holdout action value on at least two datasets?"
        ),
        "configuration": {
            "datasets": list(specs),
            "training_rows_per_dataset": EXPECTED_TRAINING_ROWS,
            "evaluation_rows_per_dataset": EXPECTED_EVALUATION_ROWS,
            "training_target_blocks_half_open": [list(block) for block in BLOCKS],
            "typed_program": (
                "mask one training row from the Ridge loss for one 12-step output block"
            ),
            "soft_typed_program": (
                "downweight selected training rows from 1.0 to 0.75 for each selected "
                "12-step output block"
            ),
            "soft_removal_strength": SOFT_REMOVAL_STRENGTH,
            "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE, equal mean within alternating 4/4 fold",
            "proxy_selection_rule": (
                "per block select rows with support-fold singleton first-order credit > 0"
            ),
            "cohort_risk_guard": (
                "execute the proxy-guided four-block cohort iff its exact grouped "
                "support gain > 0; otherwise abstain and keep the clean Ridge baseline"
            ),
            "oracle_diagnostic_rule": (
                "per block select rows with support-fold exact singleton credit > 0"
            ),
            "all_rows_guard": (
                "if all 72 rows are selected for a block, retain the row with minimum "
                "support credit; ties use lower row index"
            ),
            "empty_group_rule": "keep the clean Ridge baseline for that block",
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
        },
        "dataset_evidence": dataset_evidence,
        "overall": {
            "natural_exact_positive_fold_credit_count": exact_positive_count,
            "natural_exact_nonpositive_fold_credit_count": exact_nonpositive_count,
            "natural_exact_negative_fold_credit_count": exact_negative_count,
            "natural_exact_zero_fold_credit_count": exact_zero_count,
            "natural_exact_positive_fold_credit_fraction": exact_positive_count
            / (exact_positive_count + exact_nonpositive_count),
            "natural_action_sign_heterogeneity_present": natural_heterogeneity,
            "dataset_macro_exact_action_fold_sign_agreement_rate": statistics.fmean(
                float(row["exact_action_fold_sign_agreement_rate"])
                for row in singleton_summaries
            ),
            "dataset_macro_singleton_proxy_pearson": statistics.fmean(
                float(row["first_order_vs_exact_pearson"])
                for row in singleton_summaries
                if row["first_order_vs_exact_pearson"] is not None
            ),
            "dataset_macro_singleton_proxy_sign_accuracy": statistics.fmean(
                float(row["first_order_vs_exact_sign_accuracy"])
                for row in singleton_summaries
            ),
            "dataset_macro_singleton_proxy_balanced_accuracy": (
                statistics.fmean(available_singleton_ba)
                if available_singleton_ba
                else None
            ),
            "dataset_macro_proxy_guided_exact_holdout_gain": proxy_macro_gain,
            "proxy_guided_positive_dataset_count": proxy_positive_dataset_count,
            "proxy_guided_harmful_dataset_count": sum(
                value < 0.0 for value in proxy_dataset_gains
            ),
            "proxy_guided_positive_direction_count": sum(
                int(row["proxy_guided_exact_positive_direction_count"])
                for row in dataset_summaries
            ),
            "proxy_guided_harmful_direction_count": sum(
                int(row["proxy_guided_exact_harmful_direction_count"])
                for row in dataset_summaries
            ),
            "dataset_macro_proxy_guarded_exact_holdout_gain": (
                guarded_proxy_macro_gain
            ),
            "proxy_guarded_positive_dataset_count": (
                guarded_proxy_positive_dataset_count
            ),
            "proxy_guarded_harmful_dataset_count": (
                guarded_proxy_harmful_dataset_count
            ),
            "proxy_guarded_positive_direction_count": sum(
                int(row["proxy_guarded_exact_positive_direction_count"])
                for row in dataset_summaries
            ),
            "proxy_guarded_harmful_direction_count": sum(
                int(row["proxy_guarded_exact_harmful_direction_count"])
                for row in dataset_summaries
            ),
            "cohort_risk_guard_abstained_direction_count": sum(
                int(row["proxy_guarded_abstained_direction_count"])
                for row in dataset_summaries
            ),
            "datasets_with_any_cohort_risk_abstention_count": sum(
                int(row["proxy_guarded_abstained_direction_count"]) > 0
                for row in dataset_summaries
            ),
            "fully_abstained_dataset_count": sum(
                int(row["proxy_guarded_abstained_direction_count"]) == 2
                for row in dataset_summaries
            ),
            "dataset_macro_proxy_selected_action_fraction": statistics.fmean(
                float(row["proxy_guided_mean_selected_action_fraction"])
                for row in dataset_summaries
            ),
            "dataset_macro_oracle_guided_exact_holdout_gain": statistics.fmean(
                oracle_dataset_gains
            ),
            "oracle_guided_positive_dataset_count": sum(
                value > 0.0 for value in oracle_dataset_gains
            ),
            "oracle_guided_harmful_dataset_count": sum(
                value < 0.0 for value in oracle_dataset_gains
            ),
            "proxy_combined_attribution_diagnostic": _combined_proxy_summary(
                np, all_proxy_policies
            ),
            "oracle_combined_attribution_diagnostic": _combined_proxy_summary(
                np, all_oracle_policies
            ),
            "all_rows_guard_trigger_count": sum(
                int(row["proxy_guided_all_rows_guard_trigger_count"])
                + int(row["oracle_guided_all_rows_guard_trigger_count"])
                for row in dataset_summaries
            ),
        },
        "headroom_rule": {
            "natural_action_sign_heterogeneity_present": True,
            "dataset_macro_proxy_guided_exact_holdout_gain": "> 0",
            "proxy_guided_positive_dataset_count": ">= 2 of 4",
        },
        "cohort_risk_guard_rule": {
            "direction_decision": "EXECUTE iff exact grouped support gain > 0",
            "abstain_holdout_gain": 0.0,
            "guarded_headroom": [
                "natural action sign heterogeneity present",
                "dataset-macro guarded exact holdout gain > 0",
                "at least 2 of 4 guarded dataset means > 0",
                "0 guarded harmful datasets",
            ],
        },
        "cohort_risk_guard_verdict": (
            "GUARDED_NATURAL_BLOCK_ACTION_VALUE_HEADROOM_PRESENT"
            if guarded_headroom_present
            else "GUARDED_NATURAL_BLOCK_ACTION_VALUE_HEADROOM_NOT_ESTABLISHED"
        ),
        "target_only_feedback_budget_diagnostic": {
            "scientific_role": (
                "exposed Source datasets treated one at a time as target-only "
                "feedback-budget diagnostics; no UCI Target is opened"
            ),
            "support_query_splits": {
                "B=0": "no support; one baseline query over all 8 series",
                "B=1": "8 singleton supports; query is the other 7 series",
                "B=2": (
                    "4 fixed disjoint supports (0,1),(2,3),(4,5),(6,7); "
                    "query is the other 6 series"
                ),
                "B=4": (
                    "2 alternating supports (0,2,4,6) and (1,3,5,7); "
                    "query is the complementary 4 series"
                ),
            },
            "action_and_guard": (
                "support singleton first-order credit > 0 proposes each block cohort; "
                "execute the combined exact grouped cohort iff exact support gain > 0"
            ),
            "budget_summary": feedback_budget_overall,
            "dataset_macro_adapt_auc_budget_grid_mean": (
                feedback_budget_adapt_auc
            ),
            "adapt_auc_semantics": (
                "unweighted mean of B=0,1,2,4 guarded exact query gains on exposed "
                "datasets; not continuous-area, Capability, or transfer evidence"
            ),
            "verdict": "EXPOSED_TARGET_ONLY_FEEDBACK_BUDGET_DIAGNOSTIC_REPORTED",
        },
        "soft_temporal_block_reweighting_budget_diagnostic": {
            "scientific_role": (
                "exposed fixed-step Program-mechanism test on the same Source datasets, "
                "action proposals, and support/query splits as the hard diagnostic"
            ),
            "removal_strength": SOFT_REMOVAL_STRENGTH,
            "selected_row_weight_before": 1.0,
            "selected_row_weight_after": 1.0 - SOFT_REMOVAL_STRENGTH,
            "action_and_guard": (
                "hard singleton first-order deletion credit > 0 proposes the same rows; "
                "execute the four-block soft exact grouped cohort iff its soft exact "
                "support gain > 0"
            ),
            "budget_summary": soft_feedback_budget_overall,
            "dataset_macro_adapt_auc_budget_grid_mean": (
                soft_feedback_budget_adapt_auc
            ),
            "adapt_auc_semantics": (
                "unweighted mean of B=0,1,2,4 guarded exact query gains on exposed "
                "datasets; not continuous-area, Capability, or transfer evidence"
            ),
            "verdict": "EXPOSED_SOFT_REWEIGHTING_PROGRAM_DIAGNOSTIC_REPORTED",
        },
        "hard_vs_soft_feedback_budget_comparison": {
            "budget_comparison": hard_vs_soft_budget_overall,
            "hard_dataset_macro_adapt_auc_budget_grid_mean": (
                feedback_budget_adapt_auc
            ),
            "soft_dataset_macro_adapt_auc_budget_grid_mean": (
                soft_feedback_budget_adapt_auc
            ),
            "soft_minus_hard_dataset_macro_adapt_auc": (
                soft_feedback_budget_adapt_auc - feedback_budget_adapt_auc
            ),
            "beneficial_retention_budgets": [1, 2, 4],
            "hard_beneficial_gain_sum": aggregate_hard_beneficial_gain,
            "soft_nonnegative_gain_on_hard_beneficial_splits": (
                aggregate_soft_retained_gain
            ),
            "beneficial_gain_retention_fraction": (
                aggregate_soft_retained_gain / aggregate_hard_beneficial_gain
                if aggregate_hard_beneficial_gain > 0.0
                else None
            ),
            "hard_beneficial_split_count": (
                aggregate_hard_beneficial_split_count
            ),
            "retained_hard_beneficial_split_count": (
                aggregate_retained_beneficial_split_count
            ),
            "beneficial_split_retention_fraction": (
                aggregate_retained_beneficial_split_count
                / aggregate_hard_beneficial_split_count
                if aggregate_hard_beneficial_split_count
                else None
            ),
            "soft_strictly_reduces_harmful_split_count_at_b1_and_b2": (
                soft_strictly_reduces_b1_b2_harm
            ),
            "soft_adapt_auc_not_lower_than_hard": soft_adapt_auc_not_lower,
            "promotion_exit_condition_met": (
                soft_strictly_reduces_b1_b2_harm
                and soft_adapt_auc_not_lower
            ),
            "verdict": (
                "SOFT_PROGRAM_MEETS_PROMOTION_EXIT_CONDITION"
                if soft_strictly_reduces_b1_b2_harm
                and soft_adapt_auc_not_lower
                else "SOFT_PROGRAM_DOES_NOT_MEET_PROMOTION_EXIT_CONDITION"
            ),
        },
        "verdict": (
            "NATURAL_BLOCK_ACTION_VALUE_HEADROOM_PRESENT"
            if headroom_present
            else "NATURAL_BLOCK_ACTION_VALUE_HEADROOM_NOT_ESTABLISHED"
        ),
        "compute_accounting": {
            "reference_system_construction_count": reference_solve_count,
            "reference_solve_count": reference_solve_count,
            "grouped_small_matrix_solve_count": grouped_small_matrix_solve_count,
            "feedback_budget_grouped_small_matrix_solve_count": (
                feedback_budget_grouped_small_matrix_solve_count
            ),
            "soft_feedback_budget_grouped_small_matrix_solve_count": (
                soft_feedback_budget_grouped_small_matrix_solve_count
            ),
            "total_grouped_small_matrix_solve_count": (
                grouped_small_matrix_solve_count
                + feedback_budget_grouped_small_matrix_solve_count
            ),
            "hard_plus_soft_total_grouped_small_matrix_solve_count": (
                grouped_small_matrix_solve_count
                + feedback_budget_grouped_small_matrix_solve_count
                + soft_feedback_budget_grouped_small_matrix_solve_count
            ),
            "per_action_refit_count": 0,
            "consumer_refit_count": 0,
            "singleton_action_count": len(specs)
            * EXPECTED_TRAINING_ROWS
            * len(BLOCKS),
            "crossfit_combined_policy_count": len(all_proxy_policies)
            + len(all_oracle_policies),
        },
        "capability_claim": False,
        "utility_supported": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Exposed development comparison of hard masking and one fixed 0.25 "
            "row x block removal step under Ridge. Exact grouped query gain evaluates "
            "cohorts selected "
            "and risk-guarded using exposed feedback on a disjoint support fold; "
            "the B=0/1/2/4 curves treat each exposed Source as a target-only diagnostic "
            "and do not open the UCI Target. "
            "singleton credits are never summed as policy value. Proxy magnitudes are "
            "not averaged across datasets as intrinsic data value. This is not "
            "Capability, Promotion, Memory, or transfer evidence."
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
