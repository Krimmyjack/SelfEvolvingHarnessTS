"""Test one cohort-context guard over the exposed flatline ActionValue skill.

The existing flatline skill proposes row-level interval masks from target-local
ActionValue and executes them when an exact grouped support check is positive.
This development-only runner asks whether a small, source-episode Context guard
can further distinguish when that locally supported action transfers to the
remaining query series.

Only already exposed Source outcomes are consumed.  Ridge, sMASE, the flatline
Observation, the keep/mask Program, and the target-local proposal are frozen.
No Consumer is fitted, no Target/UCI data is read, and no persistent Memory or
Capability is created.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_program_conditioned_action_value_episode_lodo import (
    _context_features,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    ANCHORS,
    CONTEXT_LENGTH,
    FRESH_SPECS,
    HORIZON,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-flatline-multiscale-context-guard-lodo/1"
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_flatline_actionability_credit_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_flatline_multiscale_context_guard_lodo_report.json"
)
BUDGETS = (1, 2, 4)
ADAPTATION_BUDGET_GRID = (0, 1, 2, 4)
VIEWS = ("local_only", "pooled_source", "global_context", "multiscale_context")
RETRIEVAL_VIEWS = ("global_context", "multiscale_context")
TOP_K_PER_SOURCE_DATASET = 3
MIN_BENEFICIAL_GAIN_RETENTION = 0.8


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _dispersion(np: Any, rows: Any) -> float:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("dispersion requires a non-empty matrix")
    center = np.mean(matrix, axis=0)
    return float(np.mean(np.linalg.norm(matrix - center[None, :], axis=1)))


def _mean_nearest_neighbor_distance(np: Any, rows: Any) -> float:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(
        matrix[:, None, :] - matrix[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    return float(np.mean(np.min(distances, axis=1)))


def _zscore(np: Any, rows: Any) -> Any:
    matrix = np.asarray(rows, dtype=np.float64)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - center[None, :]) / scale[None, :]


def _unit_context_vector(unit: dict[str, Any]) -> list[float]:
    context = unit["context"]
    return [
        float(context["context_lag_correlation"]),
        float(context["context_last_minus_median"]),
        float(context["flatline_value_standardized"]),
        float(context["left_boundary_jump_abs"]),
        float(context["right_boundary_jump_abs"]),
    ]


def _candidate_context(np: Any, seed_row: dict[str, Any]) -> dict[str, float]:
    units = list(seed_row["unit_credit"])
    vectors = np.asarray([_unit_context_vector(row) for row in units], dtype=np.float64)
    normalized = _zscore(np, vectors)
    unique_series = {str(row["row_key"][0]) for row in units}
    unique_anchors = {int(row["row_key"][1]) for row in units}
    periods = {int(row["context"]["known_sampling_period"]) for row in units}
    ratios = {float(row["context"]["flatline_to_period_ratio"]) for row in units}
    if len(periods) != 1 or len(ratios) != 1:
        raise ValueError("one seed episode must have fixed Program geometry")
    return {
        "program__known_period": float(next(iter(periods))),
        "program__flatline_to_period_ratio": float(next(iter(ratios))),
        "cohort__candidate_fraction": len(units) / 72.0,
        "cohort__candidate_unique_series_fraction": len(unique_series) / len(units),
        "cohort__candidate_anchor_coverage": len(unique_anchors) / len(ANCHORS),
        "cohort__candidate_context_dispersion": _dispersion(np, normalized),
        "cohort__candidate_context_nn_distance": (
            _mean_nearest_neighbor_distance(np, normalized)
        ),
        "cohort__candidate_lag_correlation_mean": _mean(
            [float(row["context"]["context_lag_correlation"]) for row in units]
        ),
        "cohort__candidate_boundary_jump_mean": _mean(
            [
                0.5
                * (
                    float(row["context"]["left_boundary_jump_abs"])
                    + float(row["context"]["right_boundary_jump_abs"])
                )
                for row in units
            ]
        ),
        "cohort__candidate_flatline_value_std": float(np.std(vectors[:, 2])),
    }


def _eval_context_matrix(
    np: Any,
    *,
    values: dict[str, Any],
    evaluation_uids: list[str],
    spec: dict[str, object],
) -> Any:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
        _center_scale,
    )

    rows: list[list[float]] = []
    period = int(spec["period"])
    train_stop = int(spec["train_stop"])
    for uid in evaluation_uids:
        raw = np.asarray(values[uid], dtype=np.float64)
        context = raw[train_stop - CONTEXT_LENGTH : train_stop]
        center, scale, method = _center_scale(context)
        if (
            context.shape != (CONTEXT_LENGTH,)
            or not np.isfinite(context).all()
            or method == "scale_floor_fallback"
        ):
            raise ValueError(f"invalid exposed eval context: {uid}")
        features = _context_features(np, (context - center) / scale, period)
        rows.append(
            [
                float(features["trend"]),
                float(features["last_minus_median"]),
                float(features["period_lag_correlation"]),
            ]
        )
    if len(rows) != 8:
        raise ValueError("flatline diagnostic requires eight evaluation series")
    return _zscore(np, rows)


def _split_context(
    np: Any,
    *,
    candidate: dict[str, float],
    eval_context: Any,
    split: dict[str, Any],
    budget: int,
) -> dict[str, dict[str, float]]:
    support_indices = tuple(int(value) for value in split["support_uid_indices"])
    query_indices = tuple(int(value) for value in split["query_uid_indices"])
    if len(support_indices) != budget or len(query_indices) != 8 - budget:
        raise ValueError("support/query geometry changed")
    support = np.asarray(eval_context[list(support_indices)], dtype=np.float64)
    query = np.asarray(eval_context[list(query_indices)], dtype=np.float64)
    support_center = np.mean(support, axis=0)
    query_center = np.mean(query, axis=0)
    support_query_distance = float(np.linalg.norm(support_center - query_center))
    global_context = {
        **candidate,
        "support__budget_fraction": budget / 8.0,
        "support__query_context_distance": support_query_distance,
        "support__context_dispersion": _dispersion(np, support),
        "query__context_dispersion": _dispersion(np, query),
    }
    support_gain = float(split["h1_exact_grouped_support_gain"])
    proposed_fraction = int(split["h1_proposed_count"]) / 14.0
    multiscale_context = {
        **global_context,
        "action__exact_grouped_support_gain": support_gain,
        "action__support_gain_per_proposed_fraction": (
            support_gain / proposed_fraction if proposed_fraction > 0.0 else 0.0
        ),
        "action__proposed_fraction": proposed_fraction,
    }
    return {
        "global_context": global_context,
        "multiscale_context": multiscale_context,
    }


def _build_episodes(
    np: Any,
    *,
    root: Path,
    report: dict[str, Any],
    capture_support_action_response: bool = False,
) -> list[dict[str, Any]]:
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl

    registry = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry}
    specs = {**SPECS, **FRESH_SPECS}
    historical = {str(row["dataset_id"]): row for row in report["dataset_evidence"]}
    diagnostic = {
        str(row["dataset_id"]): row
        for row in report["action_value_guard_budget_diagnostic"]["dataset_evidence"]
    }
    if set(historical) != set(specs) or set(diagnostic) != set(specs):
        raise ValueError("exposed flatline dataset set changed")
    all_eval_uids = [
        str(uid)
        for row in historical.values()
        for uid in row["evaluation_uids"]
    ]
    values = _load_values(
        [records[uid] for uid in all_eval_uids],
        root / "data/benchmark_v0_2/clean_base",
    )

    episodes: list[dict[str, Any]] = []
    for dataset_id, spec in specs.items():
        dataset_history = historical[dataset_id]
        eval_uids = [str(uid) for uid in dataset_history["evaluation_uids"]]
        eval_context = _eval_context_matrix(
            np,
            values=values,
            evaluation_uids=eval_uids,
            spec=spec,
        )
        seeds = {
            int(row["seed"]): row for row in dataset_history["seed_evidence"]
        }
        if set(seeds) != {0, 1, 2}:
            raise ValueError("exposed seed set changed")
        candidate_by_seed = {
            seed: _candidate_context(np, seed_row) for seed, seed_row in seeds.items()
        }
        split_rows = diagnostic[dataset_id]["split_evidence"]
        for budget in BUDGETS:
            for split in split_rows[str(budget)]:
                seed = int(split["seed"])
                contexts = _split_context(
                    np,
                    candidate=candidate_by_seed[seed],
                    eval_context=eval_context,
                    split=split,
                    budget=budget,
                )
                locally_executes = split["h1_guard_decision"] == "EXECUTE"
                query_gain = (
                    float(split["h1_guarded_query_gain"])
                    if locally_executes
                    else 0.0
                )
                episode = {
                    "episode_id": f"{dataset_id}|seed={seed}|{split['split_id']}",
                    "dataset_id": dataset_id,
                    "seed": seed,
                    "budget": budget,
                    "split_id": str(split["split_id"]),
                    "locally_executes": locally_executes,
                    "query_gain_if_locally_executed": query_gain,
                    "query_gain_observed": locally_executes,
                    "features": contexts,
                }
                if capture_support_action_response:
                    episode[
                        "action__support_exact_singleton_sign_coherence"
                    ] = split[
                        "action__support_exact_singleton_sign_coherence"
                    ]
                episodes.append(episode)
    return episodes


def _source_dataset_advice(
    np: Any,
    *,
    source: list[dict[str, Any]],
    target: dict[str, Any],
    view: str,
) -> dict[str, Any]:
    source_datasets = sorted({str(row["dataset_id"]) for row in source})
    if len(source_datasets) < 2:
        raise ValueError("LODO guard requires at least two source datasets")
    feature_names: list[str] = []
    source_matrix: Any = None
    target_vector: Any = None
    if view in RETRIEVAL_VIEWS:
        feature_names = sorted(source[0]["features"][view])
        if any(sorted(row["features"][view]) != feature_names for row in source):
            raise ValueError(f"source feature schema changed: {view}")
        if sorted(target["features"][view]) != feature_names:
            raise ValueError(f"target feature schema changed: {view}")
        source_matrix = np.asarray(
            [
                [float(row["features"][view][name]) for name in feature_names]
                for row in source
            ],
            dtype=np.float64,
        )
        center = np.mean(source_matrix, axis=0)
        scale = np.std(source_matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        source_matrix = (source_matrix - center[None, :]) / scale[None, :]
        target_vector = np.asarray(
            [float(target["features"][view][name]) for name in feature_names],
            dtype=np.float64,
        )
        target_vector = (target_vector - center) / scale

    dataset_rows: list[dict[str, Any]] = []
    for dataset_id in source_datasets:
        indices = [
            index
            for index, row in enumerate(source)
            if row["dataset_id"] == dataset_id
        ]
        if view == "pooled_source":
            selected_indices = indices
        else:
            ranked = sorted(
                indices,
                key=lambda index: (
                    float(np.linalg.norm(source_matrix[index] - target_vector)),
                    str(source[index]["episode_id"]),
                ),
            )
            selected_indices = ranked[:TOP_K_PER_SOURCE_DATASET]
        responses = [
            float(source[index]["query_gain_if_locally_executed"])
            for index in selected_indices
        ]
        positive_fraction = _mean([float(value > 0.0) for value in responses])
        response_mean = _mean(responses)
        supports = positive_fraction > 0.5 and response_mean > 0.0
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "selected_episode_count": len(selected_indices),
                "positive_fraction": positive_fraction,
                "response_mean": response_mean,
                "supports_execution": supports,
            }
        )
    supporting_datasets = sum(bool(row["supports_execution"]) for row in dataset_rows)
    dataset_macro_response = _mean(
        [float(row["response_mean"]) for row in dataset_rows]
    )
    execute = supporting_datasets >= math.ceil(len(dataset_rows) / 2) and (
        dataset_macro_response > 0.0
    )
    return {
        "execute": execute,
        "supporting_source_dataset_count": supporting_datasets,
        "source_dataset_count": len(dataset_rows),
        "dataset_macro_neighbor_response": dataset_macro_response,
        "dataset_advice": dataset_rows,
    }


def _evaluate_episode_bank(
    np: Any,
    *,
    source_episodes: list[dict[str, Any]],
    target_episodes: list[dict[str, Any]],
    heldout_dataset: str,
) -> dict[str, Any]:
    source = [
        row
        for row in source_episodes
        if row["dataset_id"] != heldout_dataset and row["locally_executes"]
    ]
    target = [
        row for row in target_episodes if row["dataset_id"] == heldout_dataset
    ]
    if not source or not target:
        raise ValueError(f"unexpected exposed LODO geometry: {heldout_dataset}")
    predictions: list[dict[str, Any]] = []
    for episode in target:
        local_execute = bool(episode["locally_executes"])
        decisions = {"local_only": local_execute}
        advice: dict[str, Any] = {}
        for view in ("pooled_source", *RETRIEVAL_VIEWS):
            if local_execute:
                source_advice = _source_dataset_advice(
                    np,
                    source=source,
                    target=episode,
                    view=view,
                )
                decisions[view] = bool(source_advice["execute"])
                advice[view] = source_advice
            else:
                decisions[view] = False
        predictions.append(
            {
                "episode_id": episode["episode_id"],
                "budget": episode["budget"],
                "locally_executes": local_execute,
                "query_gain_if_locally_executed": episode[
                    "query_gain_if_locally_executed"
                ],
                "decisions": decisions,
                "advice": advice,
            }
        )

    views: dict[str, Any] = {}
    for view in VIEWS:
        budget_rows: dict[str, Any] = {}
        for budget in BUDGETS:
            rows = [row for row in predictions if row["budget"] == budget]
            gains = [
                float(row["query_gain_if_locally_executed"])
                if bool(row["decisions"][view])
                else 0.0
                for row in rows
            ]
            local_positive_gain = sum(
                max(float(row["query_gain_if_locally_executed"]), 0.0)
                for row in rows
                if bool(row["decisions"]["local_only"])
            )
            retained_positive_gain = sum(
                max(float(row["query_gain_if_locally_executed"]), 0.0)
                for row in rows
                if bool(row["decisions"][view])
            )
            budget_rows[str(budget)] = {
                "split_count": len(rows),
                "mean_guarded_query_gain": _mean(gains),
                "execute_count": sum(bool(row["decisions"][view]) for row in rows),
                "harmful_execution_count": sum(
                    bool(row["decisions"][view])
                    and float(row["query_gain_if_locally_executed"]) < 0.0
                    for row in rows
                ),
                "positive_execution_count": sum(value > 0.0 for value in gains),
                "local_beneficial_gain_sum": local_positive_gain,
                "retained_beneficial_gain_sum": retained_positive_gain,
                "beneficial_gain_retention_fraction": (
                    retained_positive_gain / local_positive_gain
                    if local_positive_gain > 0.0
                    else 1.0
                ),
                "harmful_dataset": _mean(gains) < 0.0,
            }
        adapt_auc = _mean(
            [0.0]
            + [
                float(budget_rows[str(budget)]["mean_guarded_query_gain"])
                for budget in BUDGETS
            ]
        )
        views[view] = {
            "budgets": budget_rows,
            "adapt_auc_budget_grid_mean": adapt_auc,
        }
    return {
        "heldout_dataset": heldout_dataset,
        "source_datasets": sorted({str(row["dataset_id"]) for row in source}),
        "source_executed_episode_count": len(source),
        "target_episode_count": len(target),
        "views": views,
    }


def _evaluate_fold(
    np: Any,
    *,
    episodes: list[dict[str, Any]],
    heldout_dataset: str,
) -> dict[str, Any]:
    return _evaluate_episode_bank(
        np,
        source_episodes=episodes,
        target_episodes=episodes,
        heldout_dataset=heldout_dataset,
    )


def _macro(folds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view in VIEWS:
        auc = _mean(
            [float(fold["views"][view]["adapt_auc_budget_grid_mean"]) for fold in folds]
        )
        low_harm = 0
        low_harmful_datasets = 0
        execute_count = 0
        total_split_count = 0
        local_beneficial_gain = 0.0
        retained_beneficial_gain = 0.0
        for fold in folds:
            for budget in BUDGETS:
                row = fold["views"][view]["budgets"][str(budget)]
                execute_count += int(row["execute_count"])
                total_split_count += int(row["split_count"])
                local_beneficial_gain += float(row["local_beneficial_gain_sum"])
                retained_beneficial_gain += float(row["retained_beneficial_gain_sum"])
                if budget in (1, 2):
                    low_harm += int(row["harmful_execution_count"])
                    low_harmful_datasets += int(bool(row["harmful_dataset"]))
        result[view] = {
            "dataset_macro_adapt_auc_budget_grid_mean": auc,
            "b1_b2_harmful_execution_count": low_harm,
            "b1_b2_harmful_dataset_count_sum": low_harmful_datasets,
            "execution_fraction": execute_count / total_split_count,
            "local_beneficial_gain_sum": local_beneficial_gain,
            "retained_beneficial_gain_sum": retained_beneficial_gain,
            "beneficial_gain_retention_fraction": (
                retained_beneficial_gain / local_beneficial_gain
                if local_beneficial_gain > 0.0
                else 1.0
            ),
        }
    return result


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    source_report = _read_object(root / SOURCE_REPORT_PATH)
    if source_report.get("target_query_opened") is not False:
        raise ValueError("flatline report did not preserve the Target boundary")
    episodes = _build_episodes(np, root=root, report=source_report)
    datasets = sorted({str(row["dataset_id"]) for row in episodes})
    if len(episodes) != 168 or len(datasets) != 4:
        raise ValueError("expected 168 exposed split episodes across four datasets")
    if any(dataset.startswith("uci") for dataset in datasets):
        raise ValueError("UCI is forbidden in this exposed development runner")

    folds = [
        _evaluate_fold(np, episodes=episodes, heldout_dataset=dataset)
        for dataset in datasets
    ]
    episode_bank_summary: dict[str, Any] = {}
    for dataset in datasets:
        rows = [row for row in episodes if row["dataset_id"] == dataset]
        executed = [row for row in rows if row["locally_executes"]]
        responses = [
            float(row["query_gain_if_locally_executed"]) for row in executed
        ]
        episode_bank_summary[dataset] = {
            "episode_count": len(rows),
            "locally_executed_episode_count": len(executed),
            "locally_abstained_episode_count": len(rows) - len(executed),
            "executed_positive_response_fraction": _mean(
                [float(value > 0.0) for value in responses]
            ),
            "executed_response_mean": _mean(responses),
            "dataset_level_signed_advice": (
                "SUPPORT_EXECUTION"
                if _mean([float(value > 0.0) for value in responses]) > 0.5
                and _mean(responses) > 0.0
                else "CONTRAINDICATE_EXECUTION"
            ),
        }
    macro = _macro(folds)
    multi = macro["multiscale_context"]
    comparison_views = (
        macro["local_only"],
        macro["pooled_source"],
        macro["global_context"],
    )
    best_baseline_auc = max(
        float(row["dataset_macro_adapt_auc_budget_grid_mean"])
        for row in comparison_views
    )
    checks = {
        "multiscale_auc_strictly_above_every_baseline": (
            float(multi["dataset_macro_adapt_auc_budget_grid_mean"])
            > best_baseline_auc
        ),
        "multiscale_reduces_b1_b2_harm_vs_local_only": (
            int(multi["b1_b2_harmful_execution_count"])
            < int(macro["local_only"]["b1_b2_harmful_execution_count"])
        ),
        "multiscale_harmful_datasets_not_worse_than_local_only": (
            int(multi["b1_b2_harmful_dataset_count_sum"])
            <= int(macro["local_only"]["b1_b2_harmful_dataset_count_sum"])
        ),
        "multiscale_retains_at_least_0_8_local_beneficial_gain": (
            float(multi["beneficial_gain_retention_fraction"])
            >= MIN_BENEFICIAL_GAIN_RETENTION
        ),
        "multiscale_changes_behavior": (
            float(multi["execution_fraction"])
            != float(macro["local_only"]["execution_fraction"])
        ),
    }
    passed = all(checks.values())
    patch = {
        "operation": "ADD_GUARD",
        "target_skill": "ActionValueGuardedFlatlineSkill",
        "status": (
            "DEVELOPMENT_PATCH_REPLAY_SUPPORTED"
            if passed
            else "REJECTED_CONTEXT_HYPOTHESIS"
        ),
        "behavior_before": (
            "execute iff target-local exact grouped support gain is positive"
        ),
        "behavior_after_if_accepted": (
            "after the local support gate, execute only when a dataset-balanced "
            "retrieval of source Action-Response episodes also supports execution"
        ),
        "new_observation_surface": [
            "cohort candidate redundancy and dispersion",
            "candidate series/anchor coverage",
            "support-query context distance and dispersion",
            "proposed modification fraction",
            "exact grouped support gain",
        ],
        "fast_path": "compiled deterministic keep/mask/abstain guard",
        "persistent_memory_written": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": (
            "exposed_multiscale_context_guard_hypothesis_and_harness_update_gate"
        ),
        "causal_hypothesis": (
            "The frozen flatline skill failed because local flatline and ActionValue "
            "evidence omitted cohort redundancy, modification mass, and support-query "
            "composition; adding those observations should improve a source-episode "
            "risk guard without changing the Program or Consumer."
        ),
        "configuration": {
            "source_report": SOURCE_REPORT_PATH,
            "datasets": datasets,
            "episode_count": len(episodes),
            "budgets": list(ADAPTATION_BUDGET_GRID),
            "views": list(VIEWS),
            "top_k_per_source_dataset": TOP_K_PER_SOURCE_DATASET,
            "source_dataset_balanced_vote": True,
            "consumer": "frozen Ridge from exposed source report; zero new fits",
            "metric": "per-series sMASE gain from exposed source report",
            "program": "frozen keep/mask target interval",
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "dataset_identity_used_as_feature": False,
            "dataset_identity_used_only_for_lodo_and_evidence_balancing": True,
            "target_query_outcome_used_as_feature": False,
            "consumer_fit_count": 0,
            "feature_names": {
                view: sorted(episodes[0]["features"][view])
                for view in RETRIEVAL_VIEWS
            },
        },
        "lodo_folds": folds,
        "episode_bank_summary": episode_bank_summary,
        "dataset_macro": macro,
        "gates": {
            "minimum_beneficial_gain_retention": MIN_BENEFICIAL_GAIN_RETENTION,
            "best_baseline_adapt_auc": best_baseline_auc,
            **checks,
        },
        "harness_update": patch,
        "first_fault": (
            "DETAILED_SOURCE_EPISODE_BANK_HAS_NO_DATASET_LEVEL_CONTRAINDICATION"
            if not checks["multiscale_changes_behavior"]
            and all(
                row["dataset_level_signed_advice"] == "SUPPORT_EXECUTION"
                for row in episode_bank_summary.values()
            )
            else (
                "FROZEN_MULTISCALE_CONTEXT_DOES_NOT_IMPROVE_THE_GUARD"
                if not passed
                else None
            )
        ),
        "verdict": (
            "MULTISCALE_CONTEXT_GUARD_DEVELOPMENT_PASS"
            if passed
            else "MULTISCALE_CONTEXT_GUARD_DEVELOPMENT_FAIL"
        ),
        "next_step_if_pass": (
            "Freeze the compiled guard and run one new Source replay before writing "
            "persistent Memory or opening Target/UCI."
        ),
        "next_step_if_fail": (
            "Reject this cohort-context hypothesis. Do not tune thresholds or add "
            "embedding features against the same exposed outcomes; localize the next "
            "fault to a missing Observation, Program Supply, non-identifiability, or "
            "feedback readability."
        ),
        "capability_claim": False,
        "memory_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Development-only LODO replay over exposed Source episodes. A passing "
            "result supports one replayable Harness guard update, not a promoted "
            "Capability or unseen-target transfer claim."
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
