"""Run the one-shot fresh Source replay of the frozen flatline ActionValueGuard."""
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
    _summarize_action_value_guard_budget,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
    _fresh_roster,
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


SCHEMA_VERSION = "e2-flatline-action-value-guard-fresh-replay/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_flatline_action_value_guard_fresh_replay_report.json"
)
FRESH_REPLAY_SPECS = {
    "gefcom2012_load": {
        "train_stop": 760,
        "future_bounds": (760, 808),
        "period": 24,
    },
    "monash:traffic_hourly": {
        "train_stop": 928,
        "future_bounds": (928, 976),
        "period": 24,
    },
    "metr_la": {
        "train_stop": 928,
        "future_bounds": (928, 976),
        "period": 24,
    },
}


def _frozen_fresh_roster(
    np: Any, *, root: Path, registry_rows: list[Any]
) -> list[dict[str, object]]:
    exposed_plan = _read_object(root / J0_PLAN_PATH)
    exposed_traffic = frozenset(
        str(row["series_uid"])
        for row in exposed_plan["roster"]
        if row["dataset_id"] == "monash:traffic_hourly"
    )
    exposed_metr = frozenset(
        str(row["series_uid"])
        for row in _fresh_roster(
            np,
            root=root,
            registry_rows=registry_rows,
            dataset_id="metr_la",
            spec=FRESH_SPECS["metr_la"],
        )
    )
    exclusions = {
        "gefcom2012_load": frozenset(),
        "monash:traffic_hourly": exposed_traffic,
        "metr_la": exposed_metr,
    }
    roster: list[dict[str, object]] = []
    for dataset_id, spec in FRESH_REPLAY_SPECS.items():
        roster.extend(
            _fresh_roster(
                np,
                root=root,
                registry_rows=registry_rows,
                dataset_id=dataset_id,
                spec=spec,
                excluded_uids=exclusions[dataset_id],
            )
        )
    if len(roster) != 60 or len({str(row["series_uid"]) for row in roster}) != 60:
        raise ValueError("fresh replay roster must contain 60 distinct series")
    if any(
        str(row["series_uid"]) in exclusions[str(row["dataset_id"])]
        for row in roster
    ):
        raise AssertionError("fresh replay roster overlaps the E2.9 roster")
    return roster


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

    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    roster = _frozen_fresh_roster(np, root=root, registry_rows=registry_rows)
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )

    reference_solve_count = 0
    h0_group_solve_count = 0
    h1_group_solve_count = 0
    dataset_evidence: list[dict[str, object]] = []
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
            raise ValueError(f"fresh 12+8 roster geometry changed: {dataset_id}")

        x_rows: list[Any] = []
        y_rows: list[Any] = []
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
                    raise ValueError(f"invalid fresh training window: {uid}/{anchor}")
                x_rows.append(
                    np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
                )
                y_rows.append((target - center) / scale)
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_y = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (72, 384) or clean_y.shape != (72, HORIZON):
            raise AssertionError(f"fresh training geometry changed: {dataset_id}")

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
                raise ValueError(f"invalid fresh evaluation window: {uid}")
            try:
                seasonal_by_uid[uid] = seasonal_scale(
                    np.asarray(raw[:train_stop], dtype=np.float64),
                    np.isfinite(raw[:train_stop]),
                    period=int(spec["period"]),
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(f"invalid fresh evaluation sMASE scale: {uid}") from error
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

        def score_predictions(normalized: Any) -> Any:
            prediction = np.asarray(normalized, dtype=np.float64)
            if prediction.shape != (8, HORIZON) or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid fresh Ridge prediction: {dataset_id}")
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

        episode_budgets: list[dict[str, dict[str, object]]] = []
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
            observed_weights, observations = _censor_flatline_interval_weights(np, corrupt)
            candidate_rows = tuple(
                index
                for index, observation in enumerate(observations)
                if observation["status"] == "ACTIVATE"
            )
            if set(candidate_rows) != selected_truth or len(candidate_rows) != 14:
                raise AssertionError(f"fresh flatline Observation changed: {dataset_id}/{seed}")
            if int(np.count_nonzero(observed_weights == 0.0)) != 14 * 12:
                raise AssertionError("fresh flatline Program geometry changed")
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
            episode, h1_solves = _evaluate_action_value_guard_budget(
                np,
                reference=reference,
                baseline_losses=baseline_losses,
                score_predictions=score_predictions,
                candidate_rows=candidate_rows,
                target_block=TARGET_BLOCK,
            )
            episode_budgets.append(episode)
            h0_group_solve_count += 1
            h1_group_solve_count += h1_solves

        dataset_summary = _summarize_action_value_guard_budget(episode_budgets)
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "train_uids": [str(row["series_uid"]) for row in train_rows],
                "evaluation_uids": eval_uids,
                **dataset_summary,
            }
        )

    overall_budgets: dict[str, dict[str, object]] = {}
    for budget in (0, 1, 2, 4):
        key = str(budget)
        rows = [row["budgets"][key] for row in dataset_evidence]
        overall_budgets[key] = {
            "support_series_budget": budget,
            "dataset_macro_h0_query_gain": statistics.fmean(
                float(row["h0_mean_query_gain"]) for row in rows
            ),
            "dataset_macro_h1_guarded_query_gain": statistics.fmean(
                float(row["h1_mean_guarded_query_gain"]) for row in rows
            ),
            "h0_harmful_split_count": sum(
                int(row["h0_harmful_split_count"]) for row in rows
            ),
            "h1_harmful_split_count": sum(
                int(row["h1_harmful_split_count"]) for row in rows
            ),
            "h1_abstained_split_count": sum(
                int(row["h1_abstained_split_count"]) for row in rows
            ),
            "h0_harmful_dataset_ids": [
                str(dataset_evidence[index]["dataset_id"])
                for index, row in enumerate(rows)
                if bool(row["h0_harmful_dataset"])
            ],
            "h1_harmful_dataset_ids": [
                str(dataset_evidence[index]["dataset_id"])
                for index, row in enumerate(rows)
                if bool(row["h1_harmful_dataset"])
            ],
            "h0_beneficial_gain_sum": sum(
                float(row["h0_beneficial_gain_sum"]) for row in rows
            ),
            "h1_nonnegative_gain_on_h0_beneficial_splits": sum(
                float(row["h1_nonnegative_gain_on_h0_beneficial_splits"])
                for row in rows
            ),
            "h0_beneficial_split_count": sum(
                int(row["h0_beneficial_split_count"]) for row in rows
            ),
            "retained_h0_beneficial_split_count": sum(
                int(row["retained_h0_beneficial_split_count"]) for row in rows
            ),
        }
    h0_auc = statistics.fmean(
        float(row["h0_adapt_auc_budget_grid_mean"]) for row in dataset_evidence
    )
    h1_auc = statistics.fmean(
        float(row["h1_adapt_auc_budget_grid_mean"]) for row in dataset_evidence
    )
    retention_rows = [overall_budgets[str(budget)] for budget in (1, 2, 4)]
    h0_beneficial_gain = sum(
        float(row["h0_beneficial_gain_sum"]) for row in retention_rows
    )
    retained_gain = sum(
        float(row["h1_nonnegative_gain_on_h0_beneficial_splits"])
        for row in retention_rows
    )
    gain_retention = retained_gain / h0_beneficial_gain
    h0_low_harm_splits = sum(
        int(overall_budgets[str(budget)]["h0_harmful_split_count"])
        for budget in (1, 2)
    )
    h1_low_harm_splits = sum(
        int(overall_budgets[str(budget)]["h1_harmful_split_count"])
        for budget in (1, 2)
    )
    h0_low_harm_datasets = sum(
        len(overall_budgets[str(budget)]["h0_harmful_dataset_ids"])
        for budget in (1, 2)
    )
    h1_low_harm_datasets = sum(
        len(overall_budgets[str(budget)]["h1_harmful_dataset_ids"])
        for budget in (1, 2)
    )
    checks = {
        "h1_reduces_b1_b2_harmful_splits": h1_low_harm_splits < h0_low_harm_splits,
        "h1_reduces_b1_b2_harmful_datasets": (
            h1_low_harm_datasets < h0_low_harm_datasets
        ),
        "h1_adapt_auc_strictly_above_h0": h1_auc > h0_auc,
        "beneficial_gain_retention_at_least_0_8": gain_retention >= 0.8,
        "every_dataset_h1_adapt_auc_nonnegative": all(
            float(row["h1_adapt_auc_budget_grid_mean"]) >= 0.0
            for row in dataset_evidence
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "one_shot_fresh_source_flatline_action_value_guard_replay",
        "configuration": {
            "specs": FRESH_REPLAY_SPECS,
            "anchors": list(ANCHORS),
            "seeds": list(SEEDS),
            "target_block_half_open": list(TARGET_BLOCK),
            "feedback_budgets": [0, 1, 2, 4],
            "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE",
            "context_exposure": (
                "PROGRAM_OUTCOME_FRESH_CONTEXT_NOT_GLOBALLY_AUDITED"
            ),
            "outcome_exposure": "EXPOSED",
        },
        "roster": roster,
        "dataset_evidence": dataset_evidence,
        "overall": {
            "budget_summary": overall_budgets,
            "h0_adapt_auc_budget_grid_mean": h0_auc,
            "h1_adapt_auc_budget_grid_mean": h1_auc,
            "beneficial_gain_retention_fraction": gain_retention,
            "b1_b2_h0_harmful_split_count": h0_low_harm_splits,
            "b1_b2_h1_harmful_split_count": h1_low_harm_splits,
            "b1_b2_h0_harmful_dataset_count_sum": h0_low_harm_datasets,
            "b1_b2_h1_harmful_dataset_count_sum": h1_low_harm_datasets,
            "exit_checks": checks,
            "verdict": (
                "SOURCE_PROVISIONAL_FOR_TARGET_PILOT"
                if passed
                else "FRESH_ACTION_VALUE_GUARD_REPLAY_NOT_PROMOTED"
            ),
        },
        "compute_accounting": {
            "reference_solve_count": reference_solve_count,
            "h0_grouped_small_matrix_solve_count": h0_group_solve_count,
            "h1_grouped_small_matrix_solve_count": h1_group_solve_count,
            "per_action_consumer_refit_count": 0,
            "grouped_consumer_refit_count": 0,
        },
        "capability_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "One-shot fresh Source replay of the frozen E2.9 algorithm; at most "
            "provisional evidence for a sealed Target pilot."
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
