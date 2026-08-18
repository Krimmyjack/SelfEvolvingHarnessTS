"""Test zero-fit LODO retrieval of natural block ActionValueEpisode signs.

The runner consumes the exposed natural block action report and reconstructs a
small visible feature set from the exposed Source raw data.  Source-bank labels
are the signs of two-fold mean singleton first-order proxy credits; held-out
dataset truth is the sign of the corresponding two-fold mean exact singleton
credits.  Retrieval is fixed to source-bank z-scoring, Euclidean distance, and
top-3 sign vote.

This is an exposed sign-premise diagnostic only.  It does not run a Consumer,
compile a group policy, open UCI/Target, or create Memory/Router/Capability state.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_actionability_context_gate import (
    _balanced_accuracy,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
    _fresh_roster,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    CONTEXT_LENGTH,
    FRESH_SPECS,
    HORIZON,
    J0_PLAN_PATH,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-program-conditioned-action-value-episode-lodo/1"
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_natural_block_action_value_headroom_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_program_conditioned_action_value_episode_lodo_report.json"
)
TOP_K = 3
MATERIAL_BALANCED_ACCURACY_DELTA = 0.02
VIEWS = ("pooled", "global", "program_conditioned")


def _trend(np: Any, values: Any) -> float:
    row = np.asarray(values, dtype=np.float64)
    time = np.linspace(-0.5, 0.5, row.size, dtype=np.float64)
    centered = row - float(np.mean(row))
    return float(np.dot(time, centered) / np.dot(time, time))


def _lag_correlation(np: Any, values: Any, period: int) -> float:
    row = np.asarray(values, dtype=np.float64)
    if period < 1 or period >= row.size:
        raise ValueError("period must lie inside the context")
    left = row[:-period]
    right = row[period:]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _context_features(np: Any, context: Any, period: int) -> dict[str, float]:
    row = np.asarray(context, dtype=np.float64)
    if row.shape != (CONTEXT_LENGTH,) or not np.isfinite(row).all():
        raise ValueError("context feature requires one finite standardized context")
    return {
        "trend": _trend(np, row),
        "last_minus_median": float(row[-1] - np.median(row)),
        "period_lag_correlation": _lag_correlation(np, row, period),
    }


def _block_features(
    np: Any,
    *,
    context: Any,
    target: Any,
    block: tuple[int, int],
    period: int,
) -> dict[str, float]:
    visible = np.asarray(context, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    start, stop = block
    if (
        visible.shape != (CONTEXT_LENGTH,)
        or values.shape != (HORIZON,)
        or start < 0
        or stop > HORIZON
        or stop - start != 12
        or not np.isfinite(visible).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid local target-block geometry")
    block_values = values[start:stop]
    left_value = float(visible[-1] if start == 0 else values[start - 1])
    phase_residual: list[float] = []
    for target_index in range(start, stop):
        donor_index = target_index - period
        donor = (
            float(values[donor_index])
            if donor_index >= 0
            else float(visible[CONTEXT_LENGTH + donor_index])
        )
        phase_residual.append(abs(float(values[target_index]) - donor))
    return {
        "mean": float(np.mean(block_values)),
        "std": float(np.std(block_values)),
        "trend": _trend(np, block_values),
        "left_boundary_jump_abs": abs(float(block_values[0]) - left_value),
        "period_aligned_residual_abs_mean": statistics.fmean(phase_residual),
    }


def _episode_features(
    *,
    row_context: dict[str, float],
    block: dict[str, float],
    eval_centroid: dict[str, float],
    block_start: int,
    period: int,
) -> dict[str, dict[str, float]]:
    global_features = {
        "program__known_period": float(period),
        "program__block_start_fraction": block_start / HORIZON,
        "program__block_length_to_period": 12.0 / period,
        "eval__trend_centroid": eval_centroid["trend"],
        "eval__last_minus_median_centroid": eval_centroid["last_minus_median"],
        "eval__period_lag_correlation_centroid": eval_centroid[
            "period_lag_correlation"
        ],
    }
    program_conditioned = {
        **global_features,
        "row__trend": row_context["trend"],
        "row__last_minus_median": row_context["last_minus_median"],
        "row__period_lag_correlation": row_context["period_lag_correlation"],
        "block__mean": block["mean"],
        "block__std": block["std"],
        "block__trend": block["trend"],
        "block__left_boundary_jump_abs": block["left_boundary_jump_abs"],
        "block__period_aligned_residual_abs_mean": block[
            "period_aligned_residual_abs_mean"
        ],
        "delta__row_eval_trend_abs": abs(
            row_context["trend"] - eval_centroid["trend"]
        ),
        "delta__row_eval_last_minus_median_abs": abs(
            row_context["last_minus_median"]
            - eval_centroid["last_minus_median"]
        ),
        "delta__row_eval_period_lag_correlation_abs": abs(
            row_context["period_lag_correlation"]
            - eval_centroid["period_lag_correlation"]
        ),
    }
    return {
        "global": global_features,
        "program_conditioned": program_conditioned,
    }


def _build_episodes(np: Any, *, root: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
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
        raise ValueError("UCI is forbidden in this exposed Source diagnostic")
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )
    specs = {**SPECS, **FRESH_SPECS}
    dataset_reports = {
        str(row["dataset_id"]): row for row in report["dataset_evidence"]
    }
    if set(dataset_reports) != set(specs):
        raise ValueError("natural action report datasets changed")

    episodes: list[dict[str, Any]] = []
    for dataset_id, spec in specs.items():
        dataset_report = dataset_reports[dataset_id]
        period = int(spec["period"])
        eval_context_features: list[dict[str, float]] = []
        for uid in dataset_report["evaluation_uids"]:
            raw = values[str(uid)]
            train_stop = int(spec["train_stop"])
            context = np.asarray(
                raw[train_stop - CONTEXT_LENGTH : train_stop], dtype=np.float64
            )
            center, scale, method = _center_scale(context)
            if method == "scale_floor_fallback":
                raise ValueError(f"invalid evaluation context scale: {dataset_id}/{uid}")
            standardized = (context - center) / scale
            eval_context_features.append(
                _context_features(np, standardized, period)
            )
        eval_centroid = {
            name: statistics.fmean(row[name] for row in eval_context_features)
            for name in ("trend", "last_minus_median", "period_lag_correlation")
        }

        row_cache: dict[tuple[str, int, int], tuple[Any, Any, dict[str, float]]] = {}
        for action in dataset_report["singleton_action_attribution"]:
            row_key = tuple(action["row_key"])
            if row_key not in row_cache:
                uid, anchor, horizon = str(row_key[0]), int(row_key[1]), int(row_key[2])
                if horizon != HORIZON:
                    raise ValueError("natural action row horizon changed")
                raw = values[uid]
                context = np.asarray(
                    raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64
                )
                target = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
                center, scale, method = _center_scale(context)
                if method == "scale_floor_fallback":
                    raise ValueError(f"invalid training context scale: {dataset_id}/{row_key}")
                standardized_context = (context - center) / scale
                standardized_target = (target - center) / scale
                row_cache[row_key] = (
                    standardized_context,
                    standardized_target,
                    _context_features(np, standardized_context, period),
                )
            context, target, row_context = row_cache[row_key]
            block = tuple(int(value) for value in action["block_half_open"])
            block_context = _block_features(
                np,
                context=context,
                target=target,
                block=block,
                period=period,
            )
            features = _episode_features(
                row_context=row_context,
                block=block_context,
                eval_centroid=eval_centroid,
                block_start=block[0],
                period=period,
            )
            folds = action["fold_attribution"]
            source_proxy_credit = statistics.fmean(
                float(folds[name]["first_order_proxy_attribution_credit"])
                for name in ("fold_a", "fold_b")
            )
            target_exact_credit = statistics.fmean(
                float(folds[name]["exact_singleton_attribution_credit"])
                for name in ("fold_a", "fold_b")
            )
            episode_id = (
                f"{dataset_id}|{row_key[0]}|anchor={row_key[1]}|block={block[0]}"
            )
            episodes.append(
                {
                    "episode_id": episode_id,
                    "dataset_id": dataset_id,
                    "row_key": list(row_key),
                    "block_half_open": list(block),
                    "program_id": "mask_training_row_from_12_step_output_block",
                    "source_proxy_credit": source_proxy_credit,
                    "source_proxy_positive": source_proxy_credit > 0.0,
                    "target_exact_credit": target_exact_credit,
                    "target_exact_positive": target_exact_credit > 0.0,
                    "features": features,
                }
            )
    return episodes


def _retrieve(
    np: Any,
    *,
    source: list[dict[str, Any]],
    target: dict[str, Any],
    view: str,
) -> dict[str, Any]:
    if view == "pooled":
        positive_count = sum(bool(row["source_proxy_positive"]) for row in source)
        predicted_positive = positive_count > (len(source) - positive_count)
        return {
            "predicted_positive": predicted_positive,
            "neighbor_episode_ids": [],
            "neighbor_source_proxy_credits": [],
            "neighbor_sign_vote_positive_count": None,
            "neighbor_mean_source_proxy_credit": statistics.fmean(
                float(row["source_proxy_credit"]) for row in source
            ),
            "pooled_source_positive_count": positive_count,
            "pooled_source_episode_count": len(source),
        }

    names = sorted(source[0]["features"][view])
    if any(sorted(row["features"][view]) != names for row in source):
        raise ValueError(f"source feature dimensions changed: {view}")
    if sorted(target["features"][view]) != names:
        raise ValueError(f"target feature dimensions changed: {view}")
    source_matrix = np.asarray(
        [[float(row["features"][view][name]) for name in names] for row in source],
        dtype=np.float64,
    )
    center = np.mean(source_matrix, axis=0)
    scale = np.std(source_matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    source_z = (source_matrix - center[None, :]) / scale[None, :]
    target_vector = np.asarray(
        [float(target["features"][view][name]) for name in names],
        dtype=np.float64,
    )
    target_z = (target_vector - center) / scale
    distances = np.linalg.norm(source_z - target_z[None, :], axis=1)
    ranked = sorted(
        range(len(source)),
        key=lambda index: (float(distances[index]), str(source[index]["episode_id"])),
    )
    neighbors = [source[index] for index in ranked[:TOP_K]]
    positive_count = sum(bool(row["source_proxy_positive"]) for row in neighbors)
    neighbor_mean = statistics.fmean(
        float(row["source_proxy_credit"]) for row in neighbors
    )
    return {
        "predicted_positive": positive_count > TOP_K // 2,
        "neighbor_episode_ids": [row["episode_id"] for row in neighbors],
        "neighbor_source_proxy_credits": [
            float(row["source_proxy_credit"]) for row in neighbors
        ],
        "neighbor_sign_vote_positive_count": positive_count,
        "neighbor_mean_source_proxy_credit": neighbor_mean,
        "neighbor_mean_credit_predicts_positive": neighbor_mean > 0.0,
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [bool(row["target_exact_positive"]) for row in rows]
    predicted = [bool(row["predicted_positive"]) for row in rows]
    exposed = sum(predicted)
    false_exposure_count = sum(
        estimate and not truth for estimate, truth in zip(predicted, actual)
    )
    selected_exact_credits = [
        float(row["target_exact_credit"])
        for row, estimate in zip(rows, predicted)
        if estimate
    ]
    selected_exact_credit_sum = sum(selected_exact_credits)
    selected_exact_credit_mean = (
        statistics.fmean(selected_exact_credits) if selected_exact_credits else None
    )
    mean_vote_credit_agreement = [
        bool(row["neighbor_mean_credit_predicts_positive"])
        == bool(row["predicted_positive"])
        for row in rows
        if row.get("neighbor_mean_credit_predicts_positive") is not None
    ]
    return {
        "episode_count": len(rows),
        "actual_positive_fraction": statistics.fmean(float(value) for value in actual),
        "predicted_exposure_fraction": exposed / len(rows),
        "sign_accuracy": statistics.fmean(
            float(left == right) for left, right in zip(actual, predicted)
        ),
        "balanced_accuracy": _balanced_accuracy(actual, predicted),
        "false_exposure_count": false_exposure_count,
        "false_exposure_fraction_of_all": false_exposure_count / len(rows),
        "false_exposure_fraction_among_exposed": (
            false_exposure_count / exposed if exposed else 0.0
        ),
        "selected_actual_marginal_credit_count": len(selected_exact_credits),
        "selected_actual_marginal_credit_mean": selected_exact_credit_mean,
        "selected_actual_marginal_credit_sum": selected_exact_credit_sum,
        "selected_actual_marginal_credit_is_not_group_utility": True,
        "exposure_is_harmful": selected_exact_credit_sum < 0.0,
        "sign_vote_vs_neighbor_mean_credit_sign_agreement": (
            statistics.fmean(float(value) for value in mean_vote_credit_agreement)
            if mean_vote_credit_agreement
            else None
        ),
    }


def _evaluate_lodo(
    np: Any, episodes: list[dict[str, Any]], heldout_dataset: str
) -> dict[str, Any]:
    source = [row for row in episodes if row["dataset_id"] != heldout_dataset]
    target = [row for row in episodes if row["dataset_id"] == heldout_dataset]
    if len(source) != 864 or len(target) != 288:
        raise ValueError(f"unexpected LODO geometry: {heldout_dataset}")
    views: dict[str, Any] = {}
    for view in VIEWS:
        predictions: list[dict[str, Any]] = []
        for episode in target:
            retrieval = _retrieve(np, source=source, target=episode, view=view)
            predictions.append(
                {
                    "episode_id": episode["episode_id"],
                    "row_key": episode["row_key"],
                    "block_half_open": episode["block_half_open"],
                    "target_exact_credit": episode["target_exact_credit"],
                    "target_exact_positive": episode["target_exact_positive"],
                    **retrieval,
                }
            )
        views[view] = {
            "source_episode_count": len(source),
            "metrics": _metrics(predictions),
        }
    return {
        "heldout_dataset": heldout_dataset,
        "source_datasets": sorted({str(row["dataset_id"]) for row in source}),
        "target_episode_count": len(target),
        "views": views,
    }


def _macro(folds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view in VIEWS:
        rows = [fold["views"][view]["metrics"] for fold in folds]
        balanced = [
            float(row["balanced_accuracy"])
            for row in rows
            if row["balanced_accuracy"] is not None
        ]
        result[view] = {
            "dataset_macro_balanced_accuracy": (
                statistics.fmean(balanced) if balanced else None
            ),
            "dataset_macro_sign_accuracy": statistics.fmean(
                float(row["sign_accuracy"]) for row in rows
            ),
            "dataset_macro_exposure_fraction": statistics.fmean(
                float(row["predicted_exposure_fraction"]) for row in rows
            ),
            "dataset_macro_false_exposure_fraction_of_all": statistics.fmean(
                float(row["false_exposure_fraction_of_all"]) for row in rows
            ),
            "dataset_macro_false_exposure_fraction_among_exposed": statistics.fmean(
                float(row["false_exposure_fraction_among_exposed"]) for row in rows
            ),
            "dataset_macro_selected_actual_marginal_credit_mean": statistics.fmean(
                float(row["selected_actual_marginal_credit_mean"])
                for row in rows
                if row["selected_actual_marginal_credit_mean"] is not None
            ),
            "dataset_macro_selected_actual_marginal_credit_sum": statistics.fmean(
                float(row["selected_actual_marginal_credit_sum"]) for row in rows
            ),
            "harmful_dataset_count": sum(
                bool(row["exposure_is_harmful"]) for row in rows
            ),
        }
    return result


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    source_report = _read_object(root / SOURCE_REPORT_PATH)
    if source_report.get("target_query_opened") is not False:
        raise ValueError("natural action report did not preserve Target/Query boundary")
    if int(source_report["compute_accounting"]["consumer_refit_count"]) != 0:
        raise ValueError("natural action report unexpectedly used Consumer refits")
    episodes = _build_episodes(np, root=root, report=source_report)
    datasets = sorted({str(row["dataset_id"]) for row in episodes})
    if len(episodes) != 1152 or len(datasets) != 4:
        raise ValueError("expected 1152 actions across four exposed Source datasets")
    if any(dataset.startswith("uci") for dataset in datasets):
        raise ValueError("UCI is forbidden in this exposed LODO diagnostic")

    folds = [_evaluate_lodo(np, episodes, dataset) for dataset in datasets]
    macro = _macro(folds)
    program = macro["program_conditioned"]
    baselines = [macro["pooled"], macro["global"]]
    best_baseline_ba = max(
        float(row["dataset_macro_balanced_accuracy"]) for row in baselines
    )
    ba_margin = (
        float(program["dataset_macro_balanced_accuracy"]) - best_baseline_ba
    )
    false_exposure_not_worse = float(
        program["dataset_macro_false_exposure_fraction_among_exposed"]
    ) <= min(
        float(row["dataset_macro_false_exposure_fraction_among_exposed"])
        for row in baselines
    )
    harm_not_worse = int(program["harmful_dataset_count"]) <= min(
        int(row["harmful_dataset_count"]) for row in baselines
    )
    premise_pass = (
        ba_margin > MATERIAL_BALANCED_ACCURACY_DELTA
        and false_exposure_not_worse
        and harm_not_worse
    )
    gates = {
        "material_balanced_accuracy_delta": MATERIAL_BALANCED_ACCURACY_DELTA,
        "program_conditioned_ba_margin_over_best_pooled_or_global": ba_margin,
        "program_conditioned_false_exposure_not_worse_than_both_baselines": (
            false_exposure_not_worse
        ),
        "program_conditioned_harmful_dataset_count_not_worse_than_both_baselines": (
            harm_not_worse
        ),
        "program_conditioned_sign_premise_pass": premise_pass,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": (
            "zero_new_consumer_fit_exposed_program_conditioned_action_value_sign_premise"
        ),
        "causal_hypothesis": (
            "The signed value of masking one training row x 12-step block is more "
            "retrievable across datasets from train-row Context, local block geometry, "
            "and visible eval Context than from pooled or global evidence."
        ),
        "source_report": SOURCE_REPORT_PATH,
        "configuration": {
            "datasets": datasets,
            "episode_count": len(episodes),
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "diagnostic_only": True,
            "program_id": "mask_training_row_from_12_step_output_block",
            "source_label": "sign(mean of fold-A/fold-B first-order singleton proxy credit)",
            "heldout_truth": "sign(mean of fold-A/fold-B exact singleton credit)",
            "views": list(VIEWS),
            "retrieval": (
                "leave-one-dataset-out; source-bank z-score; Euclidean top-3; "
                "source proxy-sign majority vote; neighbor proxy-credit mean is "
                "reported only as a fixed agreement diagnostic"
            ),
            "pooled_rule": (
                "majority source proxy sign across the LODO bank; ties abstain"
            ),
            "top_k": TOP_K,
            "dataset_identity_used_as_feature": False,
            "target_proxy_label_used": False,
            "target_outcome_derived_feature": False,
            "consumer_fit_count": 0,
            "feature_names": {
                view: sorted(episodes[0]["features"][view])
                for view in ("global", "program_conditioned")
            },
        },
        "lodo_folds": folds,
        "dataset_macro": macro,
        "gates": gates,
        "verdict": (
            "PROGRAM_CONDITIONED_ACTION_VALUE_SIGN_PREMISE_PASS"
            if premise_pass
            else "PROGRAM_CONDITIONED_ACTION_VALUE_SIGN_PREMISE_FAIL"
        ),
        "next_step_if_pass": (
            "Only then test whether the frozen Source prior improves B=1/2 exact-grouped "
            "query utility; do not write Memory or connect A5 yet."
        ),
        "next_step_if_fail": (
            "Close this frozen Context representation; do not add fields against the "
            "same exposed outcome without a new failure mechanism."
        ),
        "capability_claim": False,
        "memory_claim": False,
        "router_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Zero-new-fit LODO sign retrieval on fully exposed Source outcomes. Source "
            "proxy labels and held-out exact labels are development evidence. Retrieval "
            "does not execute a cohort, establish group utility, create Memory/Router "
            "state, promote a Capability, or demonstrate transfer. Reported selected "
            "actual marginal-credit means/sums are singleton diagnostics and must not "
            "be interpreted as grouped policy utility."
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
