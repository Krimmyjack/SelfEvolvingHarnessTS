"""Replay the frozen W34 Context guard with exposed signed W30 episodes.

W34 had no behavior change because every detailed Source dataset advised
execution.  This runner changes only the supplied experience: it reconstructs
the already exposed W30 GEFCom/fresh-Traffic/fresh-METR split episodes with the
unchanged W34 Context Card, then compares a positive-only bank against the full
signed bank under dataset-family LODO.

No Context field, retrieval rule, Guard threshold, Program, Consumer, Metric,
fresh cohort, or Target/UCI surface is changed.  Ridge reference/group algebra
is replayed without any Consumer refit.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    RIDGE_ALPHA,
    _evaluate_action_value_guard_budget,
    _ridge_reference_and_removal_predictions,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_action_value_guard_fresh_replay import (
    FRESH_REPLAY_SPECS,
    _frozen_fresh_roster,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
    _context_card,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_multiscale_context_guard_lodo import (
    BUDGETS,
    MIN_BENEFICIAL_GAIN_RETENTION,
    RETRIEVAL_VIEWS,
    TOP_K_PER_SOURCE_DATASET,
    VIEWS,
    _build_episodes,
    _candidate_context,
    _eval_context_matrix,
    _evaluate_episode_bank,
    _macro,
    _split_context,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    ANCHORS,
    CONTEXT_LENGTH,
    HORIZON,
    SEEDS,
    SELECTED_ROW_COUNT,
    TARGET_BLOCK,
    _apply_stuck_value_censoring,
    _censor_flatline_interval_weights,
    _read_object,
)


SCHEMA_VERSION = "e2-flatline-signed-episode-supply-lodo/1"
W29_REPORT_PATH = (
    "artifacts/functional/e2/source_flatline_actionability_credit_report.json"
)
W30_REPORT_PATH = (
    "artifacts/functional/e2/source_flatline_action_value_guard_fresh_replay_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_flatline_signed_episode_supply_lodo_report.json"
)
BANKS = ("positive_only", "signed_full")


def _build_w30_episodes(
    np: Any,
    *,
    root: Path,
    capture_support_action_response: bool = False,
    execution_removal_strength: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
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

    historical = _read_object(root / W30_REPORT_PATH)
    if historical.get("target_query_opened") is not False:
        raise ValueError("W30 did not preserve the Target boundary")
    registry = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry}
    roster = _frozen_fresh_roster(np, root=root, registry_rows=registry)
    historical_roster = [
        (str(row["dataset_id"]), str(row["series_uid"]), str(row["cohort"]))
        for row in historical["roster"]
    ]
    replay_roster = [
        (str(row["dataset_id"]), str(row["series_uid"]), str(row["cohort"]))
        for row in roster
    ]
    if replay_roster != historical_roster:
        raise ValueError("W30 exposed roster changed")
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )

    episodes: list[dict[str, Any]] = []
    reference_solve_count = 0
    grouped_small_matrix_solve_count = 0
    for dataset_id, spec in FRESH_REPLAY_SPECS.items():
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
            raise ValueError(f"W30 12+8 geometry changed: {dataset_id}")

        x_rows: list[Any] = []
        y_rows: list[Any] = []
        row_keys: list[tuple[str, int, int]] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                raw = np.asarray(values[uid], dtype=np.float64)
                context = raw[anchor - CONTEXT_LENGTH : anchor]
                target = raw[anchor : anchor + HORIZON]
                center, scale, method = _center_scale(context)
                if (
                    context.shape != (CONTEXT_LENGTH,)
                    or target.shape != (HORIZON,)
                    or not np.isfinite(context).all()
                    or not np.isfinite(target).all()
                    or method == "scale_floor_fallback"
                ):
                    raise ValueError(f"invalid W30 train window: {uid}/{anchor}")
                x_rows.append(
                    np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
                )
                y_rows.append((target - center) / scale)
                row_keys.append((uid, anchor, HORIZON))
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_y = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (72, 384) or clean_y.shape != (72, HORIZON):
            raise ValueError(f"W30 train matrix changed: {dataset_id}")

        x_eval: list[Any] = []
        raw_future: list[Any] = []
        eval_uids: list[str] = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal_by_uid: dict[str, float] = {}
        train_stop = int(spec["train_stop"])
        future_bounds = tuple(int(value) for value in spec["future_bounds"])
        for row in eval_rows:
            uid = str(row["series_uid"])
            raw = np.asarray(values[uid], dtype=np.float64)
            context = raw[train_stop - CONTEXT_LENGTH : train_stop]
            future = raw[slice(*future_bounds)]
            center, scale, method = _center_scale(context)
            if (
                context.shape != (CONTEXT_LENGTH,)
                or future.shape != (HORIZON,)
                or not np.isfinite(context).all()
                or not np.isfinite(future).all()
                or method == "scale_floor_fallback"
            ):
                raise ValueError(f"invalid W30 eval window: {uid}")
            try:
                seasonal_by_uid[uid] = seasonal_scale(
                    raw[:train_stop],
                    np.isfinite(raw[:train_stop]),
                    period=int(spec["period"]),
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(f"invalid W30 seasonal scale: {uid}") from error
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
        eval_context = _eval_context_matrix(
            np,
            values=values,
            evaluation_uids=eval_uids,
            spec=spec,
        )

        def score_predictions(normalized: Any) -> Any:
            prediction = np.asarray(normalized, dtype=np.float64)
            if prediction.shape != (8, HORIZON) or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid W30 replay prediction: {dataset_id}")
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

        for seed in SEEDS:
            selected_truth = set(
                int(value)
                for value in np.random.default_rng(seed).choice(
                    len(x_train), size=SELECTED_ROW_COUNT, replace=False
                )
            )
            corrupt, _, _ = _apply_stuck_value_censoring(
                np,
                clean_y,
                x_train[:, :CONTEXT_LENGTH],
                selected_truth,
                TARGET_BLOCK,
            )
            observed_weights, observations = _censor_flatline_interval_weights(
                np, corrupt
            )
            candidate_rows = tuple(
                index
                for index, observation in enumerate(observations)
                if observation["status"] == "ACTIVATE"
            )
            if set(candidate_rows) != selected_truth or len(candidate_rows) != 14:
                raise ValueError(f"W30 flatline Observation changed: {dataset_id}/{seed}")
            if int(np.count_nonzero(observed_weights == 0.0)) != 14 * 12:
                raise ValueError("W30 flatline Program geometry changed")
            reference = _ridge_reference_and_removal_predictions(
                np,
                x_train=x_train,
                targets=corrupt,
                x_eval=x_eval_array,
                candidate_rows=candidate_rows,
                target_block=TARGET_BLOCK,
                alpha=RIDGE_ALPHA,
            )
            reference_solve_count += 1
            baseline_losses = score_predictions(reference["baseline_prediction"])
            budget_evidence, solve_count = _evaluate_action_value_guard_budget(
                np,
                reference=reference,
                baseline_losses=baseline_losses,
                score_predictions=score_predictions,
                candidate_rows=candidate_rows,
                target_block=TARGET_BLOCK,
                capture_support_action_response=capture_support_action_response,
                execution_removal_strength=execution_removal_strength,
            )
            grouped_small_matrix_solve_count += solve_count
            unit_rows = [
                {
                    "row_index": row_index,
                    "row_key": list(row_keys[row_index]),
                    "context": _context_card(
                        np,
                        row_key=row_keys[row_index],
                        context=x_train[row_index, :CONTEXT_LENGTH],
                        corrupt_target=corrupt[row_index],
                        period=int(spec["period"]),
                    ),
                }
                for row_index in candidate_rows
            ]
            candidate = _candidate_context(np, {"unit_credit": unit_rows})
            for budget in BUDGETS:
                for split in budget_evidence[str(budget)]["split_evidence"]:
                    contexts = _split_context(
                        np,
                        candidate=candidate,
                        eval_context=eval_context,
                        split=split,
                        budget=budget,
                    )
                    locally_executes = split["h1_guard_decision"] == "EXECUTE"
                    episode = {
                        "episode_id": (
                            f"w30|{dataset_id}|seed={seed}|{split['split_id']}"
                        ),
                        "cohort_id": "w30_fresh_replay_exposed",
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "budget": budget,
                        "split_id": str(split["split_id"]),
                        "locally_executes": locally_executes,
                        "query_gain_if_locally_executed": (
                            float(split["h1_raw_query_gain"])
                            if locally_executes
                            else 0.0
                        ),
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
    return episodes, {
        "consumer_fit_count": 0,
        "reference_solve_count": reference_solve_count,
        "grouped_small_matrix_solve_count": grouped_small_matrix_solve_count,
    }


def _bank_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in sorted({str(row["dataset_id"]) for row in episodes}):
        rows = [row for row in episodes if row["dataset_id"] == dataset]
        executed = [row for row in rows if row["locally_executes"]]
        responses = [
            float(row["query_gain_if_locally_executed"]) for row in executed
        ]
        result[dataset] = {
            "episode_count": len(rows),
            "executed_episode_count": len(executed),
            "executed_positive_response_fraction": (
                statistics.fmean(float(value > 0.0) for value in responses)
                if responses
                else 0.0
            ),
            "executed_response_mean": (
                statistics.fmean(responses) if responses else 0.0
            ),
            "cohort_ids": sorted(
                {str(row.get("cohort_id", "w29_development")) for row in rows}
            ),
        }
    return result


def _evaluate_bank(
    np: Any,
    *,
    source_bank: list[dict[str, Any]],
    target_pool: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    datasets = sorted({str(row["dataset_id"]) for row in target_pool})
    folds = [
        _evaluate_episode_bank(
            np,
            source_episodes=source_bank,
            target_episodes=target_pool,
            heldout_dataset=dataset,
        )
        for dataset in datasets
    ]
    return folds, _macro(folds)


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    w29_report = _read_object(root / W29_REPORT_PATH)
    if w29_report.get("target_query_opened") is not False:
        raise ValueError("W29 did not preserve the Target boundary")
    w29 = _build_episodes(np, root=root, report=w29_report)
    for row in w29:
        row["cohort_id"] = "w29_development_exposed"
        row["episode_id"] = f"w29|{row['episode_id']}"
    w30, compute = _build_w30_episodes(np, root=root)
    signed_full = [*w29, *w30]
    datasets = sorted({str(row["dataset_id"]) for row in signed_full})
    if len(w29) != 168 or len(w30) != 126 or len(signed_full) != 294:
        raise ValueError("signed episode geometry changed")
    if len(datasets) != 5 or any(dataset.startswith("uci") for dataset in datasets):
        raise ValueError("signed bank must contain five non-UCI dataset families")

    banks = {
        "positive_only": w29,
        "signed_full": signed_full,
    }
    evaluations: dict[str, Any] = {}
    for bank_name, source_bank in banks.items():
        folds, macro = _evaluate_bank(
            np,
            source_bank=source_bank,
            target_pool=signed_full,
        )
        evaluations[bank_name] = {"lodo_folds": folds, "dataset_macro": macro}

    positive_multi = evaluations["positive_only"]["dataset_macro"][
        "multiscale_context"
    ]
    signed_macro = evaluations["signed_full"]["dataset_macro"]
    signed_multi = signed_macro["multiscale_context"]
    signed_local = signed_macro["local_only"]
    best_signed_baseline_auc = max(
        float(signed_macro[view]["dataset_macro_adapt_auc_budget_grid_mean"])
        for view in ("local_only", "pooled_source", "global_context")
    )
    checks = {
        "signed_supply_changes_multiscale_behavior": (
            float(signed_multi["execution_fraction"])
            != float(positive_multi["execution_fraction"])
        ),
        "signed_multiscale_auc_above_positive_only_multiscale": (
            float(signed_multi["dataset_macro_adapt_auc_budget_grid_mean"])
            > float(positive_multi["dataset_macro_adapt_auc_budget_grid_mean"])
        ),
        "signed_multiscale_auc_above_every_signed_baseline": (
            float(signed_multi["dataset_macro_adapt_auc_budget_grid_mean"])
            > best_signed_baseline_auc
        ),
        "signed_multiscale_reduces_b1_b2_harm_vs_signed_local": (
            int(signed_multi["b1_b2_harmful_execution_count"])
            < int(signed_local["b1_b2_harmful_execution_count"])
        ),
        "signed_multiscale_harmful_datasets_not_worse_than_signed_local": (
            int(signed_multi["b1_b2_harmful_dataset_count_sum"])
            <= int(signed_local["b1_b2_harmful_dataset_count_sum"])
        ),
        "signed_multiscale_retains_at_least_0_8_beneficial_gain": (
            float(signed_multi["beneficial_gain_retention_fraction"])
            >= MIN_BENEFICIAL_GAIN_RETENTION
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": (
            "exposed_signed_episode_supply_ablation_for_frozen_context_guard"
        ),
        "causal_hypothesis": (
            "The W34 guard did not change behavior because its detailed Source bank "
            "contained no dataset-level contraindication; adding already exposed "
            "negative/conflict W30 episodes should improve the unchanged guard."
        ),
        "configuration": {
            "w29_report": W29_REPORT_PATH,
            "w30_report": W30_REPORT_PATH,
            "datasets": datasets,
            "positive_only_episode_count": len(w29),
            "signed_full_episode_count": len(signed_full),
            "banks": list(BANKS),
            "views": list(VIEWS),
            "retrieval_views": list(RETRIEVAL_VIEWS),
            "top_k_per_source_dataset": TOP_K_PER_SOURCE_DATASET,
            "dataset_family_lodo": True,
            "same_dataset_cohorts_held_out_together": True,
            "context_fields_changed_from_w34": False,
            "guard_changed_from_w34": False,
            "program_or_consumer_changed": False,
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "consumer_fit_count": 0,
            "target_query_opened": False,
        },
        "compute_accounting": compute,
        "episode_bank_summary": {
            "positive_only": _bank_summary(w29),
            "signed_full": _bank_summary(signed_full),
        },
        "bank_evaluations": evaluations,
        "gates": {
            "minimum_beneficial_gain_retention": MIN_BENEFICIAL_GAIN_RETENTION,
            "best_signed_baseline_adapt_auc": best_signed_baseline_auc,
            **checks,
        },
        "harness_update": {
            "operation": "WRITE_SIGNED_EPISODES",
            "target_skill": "ActionValueGuardedFlatlineSkill",
            "status": (
                "DEVELOPMENT_UPDATE_REPLAY_SUPPORTED"
                if passed
                else "REJECTED_SIGNED_EPISODE_SUPPLY"
            ),
            "context_or_guard_changed": False,
            "new_experience": (
                "W30 GEFCom/fresh-Traffic/fresh-METR negative and conflict episodes"
            ),
            "persistent_memory_written": False,
        },
        "verdict": (
            "SIGNED_EPISODE_SUPPLY_DEVELOPMENT_PASS"
            if passed
            else "SIGNED_EPISODE_SUPPLY_DEVELOPMENT_FAIL"
        ),
        "next_step_if_pass": (
            "Freeze the signed episode compiler and run one new Source replay before "
            "persistent Memory or Target/UCI."
        ),
        "next_step_if_fail": (
            "Reject the frozen Context plus Experience combination; do not add more "
            "Memory or tune retrieval. Localize the next fault to a missing Context "
            "Observation, Program Supply, or non-identifiability."
        ),
        "capability_claim": False,
        "memory_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Development-only ablation over exposed Source cohorts. Passing supports "
            "a signed-experience Harness update mechanism, not a promoted Capability "
            "or unseen-target transfer claim."
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
