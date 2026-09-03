"""Test a zero-fit ActionResponseSignature premise with exposed Source LODO.

Each episode is one cached B=1/2 candidate policy from the natural block action
report.  Its fixed signature contains only response available before query:
support budget/gain, action fractions, and grouped-solve condition numbers.
LODO top-3 retrieval predicts whether an A3-executed candidate should remain
executed.  No raw data is loaded and no Consumer outcome is recomputed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    _read_object,
)


SCHEMA_VERSION = "e2-action-response-signature-lodo-premise/1"
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_natural_block_action_value_headroom_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_action_response_signature_lodo_premise_report.json"
)
BUDGETS = (1, 2)
BLOCKS = ((0, 12), (12, 24), (24, 36), (36, 48))
TOP_K = 3
POLICIES = ("always_abstain", "a3_exact_support_guard", "a5_signature_premise")


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _episode_features(split: dict[str, Any], budget: int) -> dict[str, float]:
    proposals = split["block_proposals"]
    if len(proposals) != len(BLOCKS):
        raise ValueError("expected four block proposals")
    by_block = {
        tuple(int(value) for value in row["block_half_open"]): row
        for row in proposals
    }
    if set(by_block) != set(BLOCKS):
        raise ValueError("block proposal geometry changed")
    features = {
        "support_series_budget": float(budget),
        "exact_support_gain_signed_log1p": _signed_log1p(
            float(split["exact_grouped_support_gain"])
        ),
        "overall_selected_action_fraction": float(split["selected_action_fraction"]),
    }
    for index, block in enumerate(BLOCKS):
        proposal = by_block[block]
        condition = float(proposal["middle_condition_number"])
        if not math.isfinite(condition) or condition < 0.0:
            raise ValueError("middle condition number must be finite and nonnegative")
        features[f"block_{index}_selected_fraction"] = float(
            proposal["selected_fraction"]
        )
        features[f"block_{index}_middle_condition_log1p"] = math.log1p(condition)
    if len(features) != 11 or not all(math.isfinite(value) for value in features.values()):
        raise RuntimeError("invalid fixed ActionResponseSignature")
    return features


def _build_episodes(source_report: dict[str, Any]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    datasets = [str(row["dataset_id"]) for row in source_report["dataset_evidence"]]
    if len(datasets) != 4 or len(set(datasets)) != 4:
        raise ValueError("expected four exposed Source datasets")
    if any(dataset.startswith("uci") for dataset in datasets):
        raise ValueError("UCI is forbidden in this exposed Source diagnostic")
    for dataset in source_report["dataset_evidence"]:
        dataset_id = str(dataset["dataset_id"])
        budgets = dataset["target_only_feedback_budget_diagnostic"]["budgets"]
        for budget in BUDGETS:
            splits = budgets[str(budget)]["split_evidence"]
            expected_count = 8 if budget == 1 else 4
            if len(splits) != expected_count:
                raise ValueError(f"split geometry changed: {dataset_id}/B={budget}")
            for split in splits:
                support_gain = float(split["exact_grouped_support_gain"])
                raw_query_gain = float(split["raw_exact_grouped_query_gain"])
                a3_execute = support_gain > 0.0
                if (split["guard_decision"] == "EXECUTE") != a3_execute:
                    raise ValueError("cached A3 guard decision changed")
                episodes.append(
                    {
                        "episode_id": f"{dataset_id}|B={budget}|{split['split_id']}",
                        "dataset_id": dataset_id,
                        "budget": budget,
                        "split_id": str(split["split_id"]),
                        "features": _episode_features(split, budget),
                        "source_query_positive_label": raw_query_gain > 0.0,
                        "raw_exact_grouped_query_gain": raw_query_gain,
                        "a3_execute": a3_execute,
                    }
                )
    if len(episodes) != 48:
        raise ValueError("expected 48 B=1/2 candidate-policy episodes")
    return episodes


def _lodo_decisions(np: Any, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset_id"]) for row in episodes})
    for heldout in datasets:
        source = [row for row in episodes if row["dataset_id"] != heldout]
        target = [row for row in episodes if row["dataset_id"] == heldout]
        if len(source) != 36 or len(target) != 12:
            raise ValueError(f"unexpected LODO geometry: {heldout}")
        names = sorted(source[0]["features"])
        if any(sorted(row["features"]) != names for row in source + target):
            raise ValueError("ActionResponseSignature dimensions changed")
        source_matrix = np.asarray(
            [[float(row["features"][name]) for name in names] for row in source],
            dtype=np.float64,
        )
        center = np.mean(source_matrix, axis=0)
        scale = np.std(source_matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        source_z = (source_matrix - center[None, :]) / scale[None, :]
        for episode in target:
            target_vector = np.asarray(
                [float(episode["features"][name]) for name in names],
                dtype=np.float64,
            )
            target_z = (target_vector - center) / scale
            distances = np.linalg.norm(source_z - target_z[None, :], axis=1)
            ranked = sorted(
                range(len(source)),
                key=lambda index: (float(distances[index]), str(source[index]["episode_id"])),
            )
            neighbors = [source[index] for index in ranked[:TOP_K]]
            positive_vote_count = sum(
                bool(row["source_query_positive_label"]) for row in neighbors
            )
            a5_execute = bool(episode["a3_execute"]) and positive_vote_count > TOP_K // 2
            decisions.append(
                {
                    "episode_id": episode["episode_id"],
                    "dataset_id": heldout,
                    "budget": episode["budget"],
                    "split_id": episode["split_id"],
                    "raw_exact_grouped_query_gain": episode[
                        "raw_exact_grouped_query_gain"
                    ],
                    "a3_execute": episode["a3_execute"],
                    "a5_execute": a5_execute,
                    "source_neighbor_query_positive_vote_count": positive_vote_count,
                }
            )
    return decisions


def _policy_metrics(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    if policy == "always_abstain":
        execute = [False] * len(rows)
    elif policy == "a3_exact_support_guard":
        execute = [bool(row["a3_execute"]) for row in rows]
    elif policy == "a5_signature_premise":
        execute = [bool(row["a5_execute"]) for row in rows]
    else:
        raise ValueError(f"unknown policy: {policy}")
    gains = [
        float(row["raw_exact_grouped_query_gain"]) if decision else 0.0
        for row, decision in zip(rows, execute)
    ]
    return {
        "split_count": len(rows),
        "executed_split_count": sum(execute),
        "abstained_split_count": len(rows) - sum(execute),
        "positive_split_count": sum(value > 0.0 for value in gains),
        "harmful_split_count": sum(value < 0.0 for value in gains),
        "neutral_split_count": sum(value == 0.0 for value in gains),
        "mean_exact_query_gain": statistics.fmean(gains),
        "sum_exact_query_gain": sum(gains),
        "dataset_is_harmful": statistics.fmean(gains) < 0.0,
    }


def _beneficial_retention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    beneficial = [
        row
        for row in rows
        if bool(row["a3_execute"])
        and float(row["raw_exact_grouped_query_gain"]) > 0.0
    ]
    retained = [row for row in beneficial if bool(row["a5_execute"])]
    available_gain = sum(
        float(row["raw_exact_grouped_query_gain"]) for row in beneficial
    )
    retained_gain = sum(
        float(row["raw_exact_grouped_query_gain"]) for row in retained
    )
    return {
        "a3_beneficial_executed_split_count": len(beneficial),
        "a5_retained_beneficial_split_count": len(retained),
        "beneficial_split_retention_rate": (
            len(retained) / len(beneficial) if beneficial else None
        ),
        "a3_beneficial_exact_query_gain_sum": available_gain,
        "a5_retained_beneficial_exact_query_gain_sum": retained_gain,
        "beneficial_gain_retention_fraction": (
            retained_gain / available_gain if available_gain > 0.0 else None
        ),
    }


def _dataset_budget_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policies": {policy: _policy_metrics(rows, policy) for policy in POLICIES},
        "a5_vs_a3_beneficial_retention": _beneficial_retention(rows),
    }


def _overall_budget_results(
    rows: list[dict[str, Any]], dataset_ids: list[str]
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy in POLICIES:
        pooled = _policy_metrics(rows, policy)
        dataset_metrics = [
            _policy_metrics(
                [row for row in rows if row["dataset_id"] == dataset_id], policy
            )
            for dataset_id in dataset_ids
        ]
        pooled["dataset_macro_mean_exact_query_gain"] = statistics.fmean(
            float(row["mean_exact_query_gain"]) for row in dataset_metrics
        )
        pooled["harmful_dataset_count"] = sum(
            bool(row["dataset_is_harmful"]) for row in dataset_metrics
        )
        policies[policy] = pooled
    return {
        "policies": policies,
        "a5_vs_a3_beneficial_retention": _beneficial_retention(rows),
    }


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    source_report = _read_object(root / SOURCE_REPORT_PATH)
    if source_report.get("target_query_opened") is not False:
        raise ValueError("natural report did not preserve the Target/Query boundary")
    if int(source_report["compute_accounting"]["consumer_refit_count"]) != 0:
        raise ValueError("natural report unexpectedly contains Consumer refits")
    episodes = _build_episodes(source_report)
    decisions = _lodo_decisions(np, episodes)
    datasets = sorted({str(row["dataset_id"]) for row in decisions})

    per_dataset: dict[str, Any] = {}
    for dataset_id in datasets:
        dataset_rows = [row for row in decisions if row["dataset_id"] == dataset_id]
        budgets = {
            str(budget): _dataset_budget_results(
                [row for row in dataset_rows if int(row["budget"]) == budget]
            )
            for budget in BUDGETS
        }
        grid = {}
        for policy in POLICIES:
            b1 = float(budgets["1"]["policies"][policy]["mean_exact_query_gain"])
            b2 = float(budgets["2"]["policies"][policy]["mean_exact_query_gain"])
            grid[policy] = {
                "b0_exact_query_gain": 0.0,
                "b1_exact_query_gain": b1,
                "b2_exact_query_gain": b2,
                "b0_b1_b2_grid_adapt_auc": statistics.fmean((0.0, b1, b2)),
            }
        per_dataset[dataset_id] = {"budgets": budgets, "grid": grid}

    by_budget = {
        str(budget): _overall_budget_results(
            [row for row in decisions if int(row["budget"]) == budget], datasets
        )
        for budget in BUDGETS
    }
    combined_rows = decisions
    combined = _overall_budget_results(combined_rows, datasets)
    grid_by_policy: dict[str, Any] = {}
    for policy in POLICIES:
        dataset_auc = {
            dataset_id: float(
                per_dataset[dataset_id]["grid"][policy]["b0_b1_b2_grid_adapt_auc"]
            )
            for dataset_id in datasets
        }
        grid_by_policy[policy] = {
            "dataset_adapt_auc": dataset_auc,
            "dataset_macro_b0_b1_b2_grid_adapt_auc": statistics.fmean(
                dataset_auc.values()
            ),
            "grid_harmful_dataset_count": sum(value < 0.0 for value in dataset_auc.values()),
        }

    a3_combined = combined["policies"]["a3_exact_support_guard"]
    a5_combined = combined["policies"]["a5_signature_premise"]
    a3_grid = grid_by_policy["a3_exact_support_guard"]
    a5_grid = grid_by_policy["a5_signature_premise"]
    harmful_split_reduced = int(a5_combined["harmful_split_count"]) < int(
        a3_combined["harmful_split_count"]
    )
    harmful_dataset_reduced = int(a5_grid["grid_harmful_dataset_count"]) < int(
        a3_grid["grid_harmful_dataset_count"]
    )
    adapt_auc_not_lower = float(
        a5_grid["dataset_macro_b0_b1_b2_grid_adapt_auc"]
    ) >= float(a3_grid["dataset_macro_b0_b1_b2_grid_adapt_auc"])
    premise_pass = harmful_split_reduced and harmful_dataset_reduced and adapt_auc_not_lower

    feature_names = sorted(episodes[0]["features"])
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "zero_fit_exposed_action_response_signature_lodo_premise",
        "source_report": SOURCE_REPORT_PATH,
        "configuration": {
            "datasets": datasets,
            "episode_count": len(episodes),
            "budgets": list(BUDGETS),
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "diagnostic_only": True,
            "consumer_fit_count": 0,
            "raw_data_loaded": False,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "dataset_identity_used_as_feature": False,
            "query_gain_used_as_feature": False,
            "raw_context_used_as_feature": False,
            "source_label": "raw_exact_grouped_query_gain > 0",
            "retrieval": (
                "leave-one-dataset-out; source-bank z-score; Euclidean top-3; "
                "source query-positive majority vote"
            ),
            "top_k": TOP_K,
            "a3_rule": "EXECUTE iff exact_grouped_support_gain > 0",
            "a5_premise_rule": (
                "EXECUTE iff A3 executes and at least 2 of 3 source neighbors have "
                "positive exact query gain; otherwise ABSTAIN"
            ),
            "threshold_or_k_tuned": False,
            "adapt_auc_grid": [0, 1, 2],
            "b4_included_in_adapt_auc": False,
        },
        "per_dataset": per_dataset,
        "by_budget": by_budget,
        "combined_b1_b2": combined,
        "grid_b0_b1_b2": grid_by_policy,
        "gates": {
            "a3_harmful_split_count": a3_combined["harmful_split_count"],
            "a5_harmful_split_count": a5_combined["harmful_split_count"],
            "a5_strictly_reduces_harmful_splits": harmful_split_reduced,
            "a3_grid_harmful_dataset_count": a3_grid["grid_harmful_dataset_count"],
            "a5_grid_harmful_dataset_count": a5_grid["grid_harmful_dataset_count"],
            "a5_strictly_reduces_grid_harmful_datasets": harmful_dataset_reduced,
            "a3_dataset_macro_b0_b1_b2_grid_adapt_auc": a3_grid[
                "dataset_macro_b0_b1_b2_grid_adapt_auc"
            ],
            "a5_dataset_macro_b0_b1_b2_grid_adapt_auc": a5_grid[
                "dataset_macro_b0_b1_b2_grid_adapt_auc"
            ],
            "a5_grid_adapt_auc_not_lower_than_a3": adapt_auc_not_lower,
            "action_response_signature_premise_pass": premise_pass,
        },
        "verdict": (
            "ACTION_RESPONSE_SIGNATURE_SOURCE_MEMORY_PREMISE_PASS"
            if premise_pass
            else "ACTION_RESPONSE_SIGNATURE_SOURCE_MEMORY_PREMISE_FAIL"
        ),
        "next_step_if_fail": (
            "Close the current masking family's Source-Memory premise; do not add "
            "ActionResponseSignature fields against these exposed outcomes."
        ),
        "memory_claim": False,
        "router_claim": False,
        "capability_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Exposed Source-only zero-fit LODO premise over cached B=1/2 evidence. "
            "The signature uses only response available before query. This does not "
            "create formal Memory/Router state, fit a Consumer, open UCI, promote a "
            "Capability, or demonstrate Transfer."
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
