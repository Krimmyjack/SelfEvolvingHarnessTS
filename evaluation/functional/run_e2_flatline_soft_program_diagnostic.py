"""Compare the frozen hard flatline Program with one fixed soft alternative.

W37 changes only grouped execution strength: hard removal uses 1.0, while the
single soft candidate uses 0.25 (selected interval weight 0.75).  Observation,
hard first-order proxy proposal, support/query splits, exact support-gain Guard,
Ridge, and sMASE remain frozen.  The diagnostic consumes historical hard W29,
reconstructs hard W30, and reconstructs soft W29/W30 with exact Ridge algebra.
No Consumer is fit and no signed Context, Memory, Target, or UCI surface is used.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    run as run_action_conditioned_proxy,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_multiscale_context_guard_lodo import (
    BUDGETS,
    _build_episodes,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_signed_episode_supply_lodo import (
    W29_REPORT_PATH,
    _build_w30_episodes,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    _read_object,
)


SCHEMA_VERSION = "e2-flatline-soft-program-diagnostic/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_flatline_soft_program_diagnostic_report.json"
)
HARD_REMOVAL_STRENGTH = 1.0
SOFT_REMOVAL_STRENGTH = 0.25
MIN_HARD_BENEFICIAL_GAIN_RETENTION = 0.20


def _tag_w29(episodes: list[dict[str, Any]]) -> None:
    for episode in episodes:
        episode["cohort_id"] = "w29_development_exposed"
        episode["episode_id"] = f"w29|{episode['episode_id']}"


def _episode_map(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["episode_id"]): row for row in episodes}
    if len(result) != len(episodes):
        raise ValueError("flatline episode IDs must be unique")
    return result


def _budget_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("program summary requires episode rows")
    gains = [
        float(row["query_gain_if_locally_executed"])
        if bool(row["locally_executes"])
        else 0.0
        for row in rows
    ]
    return {
        "split_count": len(rows),
        "execution_count": sum(bool(row["locally_executes"]) for row in rows),
        "abstention_count": sum(not bool(row["locally_executes"]) for row in rows),
        "positive_execution_count": sum(value > 0.0 for value in gains),
        "harmful_execution_count": sum(value < 0.0 for value in gains),
        "mean_guarded_query_gain": statistics.fmean(gains),
        "sum_guarded_query_gain": sum(gains),
        "harmful_dataset": statistics.fmean(gains) < 0.0,
    }


def _program_summary(
    episodes: list[dict[str, Any]], datasets: list[str]
) -> dict[str, Any]:
    per_dataset: dict[str, Any] = {}
    for dataset_id in datasets:
        dataset_rows = [row for row in episodes if row["dataset_id"] == dataset_id]
        budgets = {
            str(budget): _budget_metrics(
                [row for row in dataset_rows if int(row["budget"]) == budget]
            )
            for budget in BUDGETS
        }
        adapt_auc = statistics.fmean(
            [0.0]
            + [
                float(budgets[str(budget)]["mean_guarded_query_gain"])
                for budget in BUDGETS
            ]
        )
        per_dataset[dataset_id] = {
            "budgets": budgets,
            "adapt_auc_budget_grid_mean": adapt_auc,
        }

    by_budget: dict[str, Any] = {}
    for budget in BUDGETS:
        rows = [per_dataset[name]["budgets"][str(budget)] for name in datasets]
        by_budget[str(budget)] = {
            "dataset_macro_mean_guarded_query_gain": statistics.fmean(
                float(row["mean_guarded_query_gain"]) for row in rows
            ),
            "harmful_execution_count": sum(
                int(row["harmful_execution_count"]) for row in rows
            ),
            "harmful_dataset_count": sum(bool(row["harmful_dataset"]) for row in rows),
            "execution_count": sum(int(row["execution_count"]) for row in rows),
            "split_count": sum(int(row["split_count"]) for row in rows),
        }
    low_budget_rows = [by_budget[str(budget)] for budget in (1, 2)]
    return {
        "per_dataset": per_dataset,
        "by_budget": by_budget,
        "dataset_macro_adapt_auc_budget_grid_mean": statistics.fmean(
            float(per_dataset[name]["adapt_auc_budget_grid_mean"])
            for name in datasets
        ),
        "b1_b2_harmful_execution_count": sum(
            int(row["harmful_execution_count"]) for row in low_budget_rows
        ),
        "b1_b2_harmful_dataset_count_sum": sum(
            int(row["harmful_dataset_count"]) for row in low_budget_rows
        ),
    }


def _hard_beneficial_retention(
    hard: dict[str, dict[str, Any]], soft: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    beneficial_ids = [
        episode_id
        for episode_id, row in hard.items()
        if bool(row["locally_executes"])
        and float(row["query_gain_if_locally_executed"]) > 0.0
    ]
    hard_gain = sum(
        float(hard[episode_id]["query_gain_if_locally_executed"])
        for episode_id in beneficial_ids
    )
    soft_gain = sum(
        max(float(soft[episode_id]["query_gain_if_locally_executed"]), 0.0)
        if bool(soft[episode_id]["locally_executes"])
        else 0.0
        for episode_id in beneficial_ids
    )
    return {
        "hard_beneficial_episode_count": len(beneficial_ids),
        "hard_beneficial_positive_gain_sum": hard_gain,
        "soft_retained_positive_gain_sum": soft_gain,
        "soft_retained_positive_gain_fraction": (
            soft_gain / hard_gain if hard_gain > 0.0 else None
        ),
    }


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    historical_hard_w29 = _read_object(root / W29_REPORT_PATH)
    if historical_hard_w29.get("target_query_opened") is not False:
        raise ValueError("historical hard W29 did not preserve the Target boundary")

    hard_w29 = _build_episodes(np, root=root, report=historical_hard_w29)
    _tag_w29(hard_w29)
    hard_w30, hard_w30_compute = _build_w30_episodes(
        np,
        root=root,
        execution_removal_strength=HARD_REMOVAL_STRENGTH,
    )
    hard_episodes = [*hard_w29, *hard_w30]

    soft_proxy_report = run_action_conditioned_proxy(
        root, execution_removal_strength=SOFT_REMOVAL_STRENGTH
    )
    if soft_proxy_report.get("target_query_opened") is not False:
        raise ValueError("soft W29 replay did not preserve the Target boundary")
    soft_w29_report = dict(historical_hard_w29)
    soft_w29_report["action_value_guard_budget_diagnostic"] = soft_proxy_report[
        "action_value_guard_budget_diagnostic"
    ]
    soft_w29 = _build_episodes(np, root=root, report=soft_w29_report)
    _tag_w29(soft_w29)
    soft_w30, soft_w30_compute = _build_w30_episodes(
        np,
        root=root,
        execution_removal_strength=SOFT_REMOVAL_STRENGTH,
    )
    soft_episodes = [*soft_w29, *soft_w30]

    hard = _episode_map(hard_episodes)
    soft = _episode_map(soft_episodes)
    if len(hard_w29) != 168 or len(hard_w30) != 126 or len(hard) != 294:
        raise ValueError("hard W29/W30 episode geometry changed")
    if len(soft_w29) != 168 or len(soft_w30) != 126 or len(soft) != 294:
        raise ValueError("soft W29/W30 episode geometry changed")
    if set(hard) != set(soft):
        raise ValueError("hard and soft episode IDs do not align")
    for episode_id in hard:
        hard_identity = (
            hard[episode_id]["dataset_id"],
            int(hard[episode_id]["seed"]),
            int(hard[episode_id]["budget"]),
            hard[episode_id]["split_id"],
        )
        soft_identity = (
            soft[episode_id]["dataset_id"],
            int(soft[episode_id]["seed"]),
            int(soft[episode_id]["budget"]),
            soft[episode_id]["split_id"],
        )
        if hard_identity != soft_identity:
            raise ValueError(f"hard/soft episode identity changed: {episode_id}")

    datasets = sorted({str(row["dataset_id"]) for row in hard_episodes})
    if len(datasets) != 5 or any(name.startswith("uci") for name in datasets):
        raise ValueError("W37 requires exactly five exposed non-UCI Source families")
    hard_summary = _program_summary(hard_episodes, datasets)
    soft_summary = _program_summary(soft_episodes, datasets)
    retention = _hard_beneficial_retention(hard, soft)

    checks = {
        "soft_dataset_macro_adapt_auc_strictly_above_hard": (
            float(soft_summary["dataset_macro_adapt_auc_budget_grid_mean"])
            > float(hard_summary["dataset_macro_adapt_auc_budget_grid_mean"])
        ),
        "soft_strictly_reduces_b1_b2_harmful_executions": (
            int(soft_summary["b1_b2_harmful_execution_count"])
            < int(hard_summary["b1_b2_harmful_execution_count"])
        ),
        "soft_b1_b2_harmful_dataset_count_not_worse": (
            int(soft_summary["b1_b2_harmful_dataset_count_sum"])
            <= int(hard_summary["b1_b2_harmful_dataset_count_sum"])
        ),
        "soft_retains_at_least_0_20_hard_beneficial_positive_gain": (
            retention["soft_retained_positive_gain_fraction"] is not None
            and float(retention["soft_retained_positive_gain_fraction"])
            >= MIN_HARD_BENEFICIAL_GAIN_RETENTION
        ),
    }
    passed = all(checks.values())

    soft_w29_action_compute = soft_proxy_report[
        "action_value_guard_budget_diagnostic"
    ]["compute_accounting"]
    soft_w29_other_group_solves = int(
        soft_proxy_report["compute_accounting"]["grouped_small_matrix_solve_count"]
    )
    soft_w29_action_group_solves = int(
        soft_w29_action_compute["h1_grouped_small_matrix_solve_count"]
    )
    hard_w30_group_solves = int(hard_w30_compute["grouped_small_matrix_solve_count"])
    soft_w30_group_solves = int(soft_w30_compute["grouped_small_matrix_solve_count"])
    soft_w29_h0_group_solves = int(
        soft_w29_action_compute["reference_solve_count"]
    )
    hard_w30_h0_group_solves = int(hard_w30_compute["reference_solve_count"])
    soft_w30_h0_group_solves = int(soft_w30_compute["reference_solve_count"])
    new_reference_solves = (
        int(hard_w30_compute["reference_solve_count"])
        + int(soft_w29_action_compute["reference_solve_count"])
        + int(soft_w30_compute["reference_solve_count"])
    )
    new_group_solves = (
        soft_w29_other_group_solves
        + soft_w29_action_group_solves
        + soft_w29_h0_group_solves
        + hard_w30_group_solves
        + hard_w30_h0_group_solves
        + soft_w30_group_solves
        + soft_w30_h0_group_solves
    )
    historical_w29_compute = historical_hard_w29.get("w29_compute_accounting", {})
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "zero_new_fit_exposed_flatline_program_supply_diagnostic",
        "causal_hypothesis": (
            "The hard flatline interval mask is over-aggressive; a fixed 0.25 "
            "removal strength should reduce cross-cohort harm while retaining "
            "utility under the frozen proposal and exact support Guard."
        ),
        "configuration": {
            "hard_removal_strength": HARD_REMOVAL_STRENGTH,
            "soft_removal_strength": SOFT_REMOVAL_STRENGTH,
            "soft_selected_interval_weight": 1.0 - SOFT_REMOVAL_STRENGTH,
            "removal_strength_grid_searched": False,
            "proxy_proposal": "frozen hard first-order singleton proxy > 0",
            "support_guard": "exact grouped support gain > 0",
            "support_query_splits_changed": False,
            "observation_changed": False,
            "consumer_or_metric_changed": False,
            "signed_context_or_memory_used": False,
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "datasets": datasets,
            "episode_count_per_program": len(hard),
            "consumer_fit_count": 0,
            "target_query_opened": False,
        },
        "compute_accounting": {
            "consumer_fit_count": 0,
            "new_reference_solve_count": new_reference_solves,
            "hard_w29_historical_new_reference_solve_count": 0,
            "hard_w29_historical_recorded_reference_solve_count": (
                historical_w29_compute.get("reference_solve_count")
            ),
            "hard_w30_reference_solve_count": int(
                hard_w30_compute["reference_solve_count"]
            ),
            "soft_w29_reference_solve_count": int(
                soft_w29_action_compute["reference_solve_count"]
            ),
            "soft_w30_reference_solve_count": int(
                soft_w30_compute["reference_solve_count"]
            ),
            "new_grouped_small_matrix_solve_count": new_group_solves,
            "soft_w29_attribution_grouped_small_matrix_solve_count": (
                soft_w29_other_group_solves
            ),
            "soft_w29_action_value_grouped_small_matrix_solve_count": (
                soft_w29_action_group_solves
            ),
            "soft_w29_h0_grouped_small_matrix_solve_count": (
                soft_w29_h0_group_solves
            ),
            "hard_w30_action_value_grouped_small_matrix_solve_count": (
                hard_w30_group_solves
            ),
            "hard_w30_h0_grouped_small_matrix_solve_count": (
                hard_w30_h0_group_solves
            ),
            "soft_w30_action_value_grouped_small_matrix_solve_count": (
                soft_w30_group_solves
            ),
            "soft_w30_h0_grouped_small_matrix_solve_count": (
                soft_w30_h0_group_solves
            ),
            "per_action_consumer_refit_count": 0,
            "grouped_consumer_refit_count": 0,
        },
        "program_results": {
            "hard": hard_summary,
            "soft": soft_summary,
        },
        "hard_beneficial_episode_retention": retention,
        "gates": {
            "minimum_hard_beneficial_positive_gain_retention": (
                MIN_HARD_BENEFICIAL_GAIN_RETENTION
            ),
            **checks,
            "primary_pass": passed,
        },
        "verdict": (
            "SOFT_FLATLINE_PROGRAM_DIAGNOSTIC_PASS"
            if passed
            else "SOFT_FLATLINE_PROGRAM_DIAGNOSTIC_FAIL"
        ),
        "next_step_if_pass": (
            "Freeze removal strength 0.25 and run one fresh Source replay; do not "
            "search additional weights."
        ),
        "next_step_if_fail": (
            "Close the entire flatline defect family. Do not add Context, Memory, "
            "or search another removal strength."
        ),
        "harness_update_applied": False,
        "capability_claim": False,
        "memory_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Exposed Source-only zero-new-fit Program diagnostic. PASS only permits "
            "one fresh Source replay; FAIL closes the flatline defect family."
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
