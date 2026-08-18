"""Test whether cohort context makes flatline actionability evidence retrievable.

This development-only gate consumes the 168 already exposed intervention episodes from
``run_e2_flatline_actionability_credit``.  It performs leave-one-dataset-out retrieval
without fitting the downstream Consumer again.  The retrieval mechanism is intentionally
small and fixed: bank-only z-scoring, Euclidean distance, and top-3 mean signed credit.

The four views isolate what the Harness knows: no context, cohort/global context only,
local interval context with credit marginalized over observed cohort states, and local
plus current cohort/action-set context.  Retrieval exposes the one fixed target-cell
masking Program only when predicted signed credit is positive; it never chooses among
operators.  No capability is promoted and no Target/UCI data is read.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "e2-actionability-context-gate/1"
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_flatline_actionability_credit_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_actionability_context_gate_report.json"
)
TOP_K = 3
MATERIAL_BALANCED_ACCURACY_DELTA = 0.02

LOCAL_NUMERIC_FIELDS = (
    "anchor",
    "known_sampling_period",
    "flatline_to_period_ratio",
    "flatline_value_standardized",
    "left_boundary_jump_abs",
    "right_boundary_jump_abs",
    "context_last_minus_median",
    "context_lag_correlation",
    "context_lag_correlation_missing",
    "anchor_phase_fraction",
)
COHORT_DYNAMIC_FIELDS = (
    "anchor",
    "flatline_value_standardized",
    "left_boundary_jump_abs",
    "right_boundary_jump_abs",
    "context_last_minus_median",
    "context_lag_correlation",
)
VIEWS = (
    "pooled",
    "global_only",
    "local_episode",
    "local_expected",
    "local_plus_cohort",
)


def _mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("mean requires at least one value")
    return statistics.fmean(rows)


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    rows = [float(value) for value in values]
    center = _mean(rows)
    variance = _mean((value - center) ** 2 for value in rows)
    return center, math.sqrt(max(variance, 0.0))


def _local_features(card: dict[str, Any]) -> dict[str, float]:
    period = float(card["known_sampling_period"])
    anchor = float(card["anchor"])
    lag = card.get("context_lag_correlation")
    return {
        "anchor": anchor,
        "known_sampling_period": period,
        "flatline_to_period_ratio": float(card["flatline_to_period_ratio"]),
        "flatline_value_standardized": float(card["flatline_value_standardized"]),
        "left_boundary_jump_abs": float(card["left_boundary_jump_abs"]),
        "right_boundary_jump_abs": float(card["right_boundary_jump_abs"]),
        "context_last_minus_median": float(card["context_last_minus_median"]),
        "context_lag_correlation": float(lag) if lag is not None else 0.0,
        "context_lag_correlation_missing": float(lag is None),
        "anchor_phase_fraction": (anchor % period) / period,
    }


def _cohort_features(
    unit_rows: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[int, dict[str, float]]]:
    """Build outcome-free cohort and query-relative action-set geometry."""

    if not unit_rows:
        raise ValueError("empty intervention cohort")
    locals_by_index = {
        int(row["row_index"]): _local_features(dict(row["context"]))
        for row in unit_rows
    }
    cards = [dict(row["context"]) for row in unit_rows]
    local_rows = list(locals_by_index.values())
    count = len(unit_rows)
    series_counts = Counter(str(card["series_uid"]) for card in cards)
    anchor_counts = Counter(int(card["anchor"]) for card in cards)

    global_features: dict[str, float] = {
        "known_sampling_period": _mean(
            row["known_sampling_period"] for row in local_rows
        ),
        "flatline_to_period_ratio": _mean(
            row["flatline_to_period_ratio"] for row in local_rows
        ),
        "unique_series_fraction": len(series_counts) / count,
        "unique_anchor_fraction": len(anchor_counts) / count,
        "largest_series_share": max(series_counts.values()) / count,
        "largest_anchor_share": max(anchor_counts.values()) / count,
    }
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for field in COHORT_DYNAMIC_FIELDS:
        center, scale = _mean_sd(row[field] for row in local_rows)
        centers[field] = center
        scales[field] = scale
        global_features[f"{field}_mean"] = center
        global_features[f"{field}_sd"] = scale

    relative_by_index: dict[int, dict[str, float]] = {}
    for row in unit_rows:
        index = int(row["row_index"])
        card = dict(row["context"])
        local = locals_by_index[index]
        relative: dict[str, float] = {
            "same_series_peer_fraction": (
                (series_counts[str(card["series_uid"])] - 1) / max(count - 1, 1)
            ),
            "same_anchor_peer_fraction": (
                (anchor_counts[int(card["anchor"])] - 1) / max(count - 1, 1)
            ),
        }
        query_z: list[float] = []
        cohort_z: list[list[float]] = []
        for field in COHORT_DYNAMIC_FIELDS:
            scale = scales[field] if scales[field] > 1e-12 else 1.0
            value = (local[field] - centers[field]) / scale
            relative[f"{field}_within_cohort_z"] = value
            query_z.append(value)
        for other_index, other in locals_by_index.items():
            if other_index == index:
                continue
            cohort_z.append(
                [
                    (other[field] - centers[field])
                    / (scales[field] if scales[field] > 1e-12 else 1.0)
                    for field in COHORT_DYNAMIC_FIELDS
                ]
            )
        distances = [
            math.sqrt(sum((left - right) ** 2 for left, right in zip(query_z, other)))
            for other in cohort_z
        ]
        relative["nearest_peer_context_distance"] = min(distances) if distances else 0.0
        relative["mean_peer_context_distance"] = _mean(distances) if distances else 0.0
        relative_by_index[index] = relative
    return global_features, relative_by_index


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}{key}": float(value) for key, value in values.items()}


def _build_episodes(report: dict[str, Any]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for dataset in report["dataset_evidence"]:
        dataset_id = str(dataset["dataset_id"])
        for seed_row in dataset["seed_evidence"]:
            seed = int(seed_row["seed"])
            units = list(seed_row["unit_credit"])
            global_features, relative_by_index = _cohort_features(units)
            for unit in units:
                index = int(unit["row_index"])
                local = _local_features(dict(unit["context"]))
                combined = {
                    **_prefixed("local__", local),
                    **_prefixed("cohort__", global_features),
                    **_prefixed("relative__", relative_by_index[index]),
                }
                fold_a = float(unit["fold_a_marginal_credit"])
                fold_b = float(unit["fold_b_marginal_credit"])
                row_key = tuple(unit["row_key"])
                episodes.append(
                    {
                        "episode_id": f"{dataset_id}|seed={seed}|row={index}",
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "row_index": index,
                        "row_key": row_key,
                        "fold_a_credit": fold_a,
                        "fold_b_credit": fold_b,
                        "credit": (fold_a + fold_b) / 2.0,
                        "features": {
                            "global_only": _prefixed("cohort__", global_features),
                            "local_episode": _prefixed("local__", local),
                            "local_expected": _prefixed("local__", local),
                            "local_plus_cohort": combined,
                        },
                    }
                )
    return episodes


def _same_features(left: dict[str, float], right: dict[str, float]) -> bool:
    return set(left) == set(right) and all(
        math.isclose(left[key], right[key], rel_tol=0.0, abs_tol=1e-12)
        for key in left
    )


def _duplicate_context_audit(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[(episode["dataset_id"], episode["row_key"])].append(episode)
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    state_conflicts = 0
    mean_sign_conflicts = 0
    fold_specific_conflicts = 0
    expected_sign_matches = 0
    expected_sign_total = 0
    for rows in repeated:
        reference = rows[0]["features"]["local_expected"]
        if any(
            not _same_features(reference, row["features"]["local_expected"])
            for row in rows[1:]
        ):
            raise AssertionError("repeated row_key changed local Context across seeds")
        states = {
            (row["fold_a_credit"] > 0.0, row["fold_b_credit"] > 0.0)
            for row in rows
        }
        state_conflicts += len(states) > 1
        mean_signs = {row["credit"] > 0.0 for row in rows}
        mean_sign_conflicts += len(mean_signs) > 1
        for field in ("fold_a_credit", "fold_b_credit"):
            fold_specific_conflicts += len({row[field] > 0.0 for row in rows}) > 1
        expected_positive = _mean(row["credit"] for row in rows) > 0.0
        expected_sign_matches += sum(
            (row["credit"] > 0.0) == expected_positive for row in rows
        )
        expected_sign_total += len(rows)
    return {
        "repeated_local_context_group_count": len(repeated),
        "two_fold_state_conflict_group_count": state_conflicts,
        "two_fold_state_conflict_rate": state_conflicts / len(repeated) if repeated else 0.0,
        "mean_credit_sign_conflict_group_count": mean_sign_conflicts,
        "mean_credit_sign_conflict_rate": (
            mean_sign_conflicts / len(repeated) if repeated else 0.0
        ),
        "fold_specific_repeated_group_count": 2 * len(repeated),
        "fold_specific_sign_conflict_count": fold_specific_conflicts,
        "fold_specific_sign_conflict_rate": (
            fold_specific_conflicts / (2 * len(repeated)) if repeated else 0.0
        ),
        "episode_to_row_expected_sign_agreement": (
            expected_sign_matches / expected_sign_total if expected_sign_total else None
        ),
        "interpretation": (
            "Identical local Context can receive different credit when the concurrently "
            "corrupted action set changes; row-expected credit marginalizes those observed "
            "states, while local+cohort retrieval conditions on them."
        ),
    }


def _memory_entries(
    source: list[dict[str, Any]], view: str
) -> list[dict[str, Any]]:
    if view in {"local_episode", "local_plus_cohort"}:
        return [
            {
                "entry_id": episode["episode_id"],
                "features": episode["features"][view],
                "credit": episode["credit"],
            }
            for episode in source
        ]

    if view == "local_expected":
        grouped: dict[tuple[str, tuple[Any, ...]], list[dict[str, Any]]] = defaultdict(list)
        for episode in source:
            grouped[(episode["dataset_id"], episode["row_key"])].append(episode)
        entries: list[dict[str, Any]] = []
        for (dataset_id, row_key), rows in sorted(grouped.items(), key=lambda item: str(item[0])):
            reference = rows[0]["features"][view]
            if any(not _same_features(reference, row["features"][view]) for row in rows[1:]):
                raise AssertionError("local expected-credit group has changing features")
            entries.append(
                {
                    "entry_id": f"{dataset_id}|row_expected={row_key}",
                    "features": reference,
                    "credit": _mean(row["credit"] for row in rows),
                }
            )
        return entries

    if view == "global_only":
        grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for episode in source:
            grouped[(episode["dataset_id"], episode["seed"])].append(episode)
        entries = []
        for (dataset_id, seed), rows in sorted(grouped.items()):
            reference = rows[0]["features"][view]
            if any(not _same_features(reference, row["features"][view]) for row in rows[1:]):
                raise AssertionError("cohort summary changed within one action set")
            entries.append(
                {
                    "entry_id": f"{dataset_id}|cohort={seed}",
                    "features": reference,
                    "credit": _mean(row["credit"] for row in rows),
                }
            )
        return entries
    raise ValueError(f"unsupported memory view: {view}")


def _standardizer(entries: list[dict[str, Any]]) -> tuple[list[str], list[float], list[float]]:
    names = sorted(entries[0]["features"])
    if any(sorted(entry["features"]) != names for entry in entries):
        raise ValueError("retrieval feature dimensions changed")
    centers: list[float] = []
    scales: list[float] = []
    for name in names:
        center, scale = _mean_sd(entry["features"][name] for entry in entries)
        centers.append(center)
        scales.append(scale if scale > 1e-12 else 1.0)
    return names, centers, scales


def _zvector(
    features: dict[str, float], names: list[str], centers: list[float], scales: list[float]
) -> list[float]:
    return [
        (float(features[name]) - center) / scale
        for name, center, scale in zip(names, centers, scales)
    ]


def _retrieve_topk_mean(
    entries: list[dict[str, Any]], query: dict[str, float], k: int = TOP_K
) -> tuple[float, list[dict[str, Any]]]:
    """TIMECLAW-like bank-only z-scored L2 retrieval with fixed top-k."""

    if not entries:
        raise ValueError("empty source evidence bank")
    names, centers, scales = _standardizer(entries)
    query_z = _zvector(query, names, centers, scales)
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for entry in entries:
        vector = _zvector(entry["features"], names, centers, scales)
        distance = math.sqrt(
            sum((left - right) ** 2 for left, right in zip(query_z, vector))
        )
        ranked.append((distance, str(entry["entry_id"]), entry))
    neighbors = [row[2] for row in sorted(ranked, key=lambda row: (row[0], row[1]))[:k]]
    return _mean(row["credit"] for row in neighbors), neighbors


def _balanced_accuracy(actual: list[bool], predicted: list[bool]) -> float | None:
    positives = [index for index, value in enumerate(actual) if value]
    negatives = [index for index, value in enumerate(actual) if not value]
    if not positives or not negatives:
        return None
    true_positive_rate = _mean(float(predicted[index]) for index in positives)
    true_negative_rate = _mean(float(not predicted[index]) for index in negatives)
    return (true_positive_rate + true_negative_rate) / 2.0


def _score_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual_positive = [float(row["actual_credit"]) > 0.0 for row in rows]
    predicted_positive = [float(row["predicted_credit"]) > 0.0 for row in rows]
    exposed = sum(predicted_positive)
    false_exposures = sum(
        predicted and not actual
        for predicted, actual in zip(predicted_positive, actual_positive)
    )
    mixed_neighbors = sum(
        len({float(value) > 0.0 for value in row["neighbor_credits"]}) > 1
        for row in rows
        if row["neighbor_credits"]
    )
    with_neighbors = sum(bool(row["neighbor_credits"]) for row in rows)
    proxy_gain = _mean(
        float(row["actual_credit"]) if predicted else 0.0
        for row, predicted in zip(rows, predicted_positive)
    )
    return {
        "episode_count": len(rows),
        "actual_positive_fraction": _mean(float(value) for value in actual_positive),
        "predicted_exposure_fraction": exposed / len(rows),
        "sign_accuracy": _mean(
            float(actual == predicted)
            for actual, predicted in zip(actual_positive, predicted_positive)
        ),
        "balanced_accuracy": _balanced_accuracy(actual_positive, predicted_positive),
        "false_exposure_fraction_among_exposed": (
            false_exposures / exposed if exposed else 0.0
        ),
        "mixed_neighbor_sign_fraction": (
            mixed_neighbors / with_neighbors if with_neighbors else None
        ),
        "mean_absolute_credit_error": _mean(
            abs(float(row["actual_credit"]) - float(row["predicted_credit"]))
            for row in rows
        ),
        "marginal_credit_policy_proxy_gain": proxy_gain,
        "proxy_is_harmful": proxy_gain < 0.0,
    }


def _evaluate_fold(
    episodes: list[dict[str, Any]], heldout_dataset: str
) -> dict[str, Any]:
    source = [row for row in episodes if row["dataset_id"] != heldout_dataset]
    target = [row for row in episodes if row["dataset_id"] == heldout_dataset]
    if not source or not target:
        raise ValueError(f"invalid LODO split: {heldout_dataset}")

    source_dataset_means = [
        _mean(row["credit"] for row in source if row["dataset_id"] == dataset_id)
        for dataset_id in sorted({row["dataset_id"] for row in source})
    ]
    pooled_prediction = _mean(source_dataset_means)
    view_results: dict[str, Any] = {}
    for view in VIEWS:
        entries = None if view == "pooled" else _memory_entries(source, view)
        predictions: list[dict[str, Any]] = []
        for episode in target:
            if view == "pooled":
                predicted_credit = pooled_prediction
                neighbors: list[dict[str, Any]] = []
            else:
                predicted_credit, neighbors = _retrieve_topk_mean(
                    entries or [], episode["features"][view]
                )
            predictions.append(
                {
                    "episode_id": episode["episode_id"],
                    "actual_credit": episode["credit"],
                    "predicted_credit": predicted_credit,
                    "expose_program": predicted_credit > 0.0,
                    "neighbor_entry_ids": [row["entry_id"] for row in neighbors],
                    "neighbor_credits": [row["credit"] for row in neighbors],
                }
            )
        view_results[view] = {
            "source_memory_entry_count": 0 if entries is None else len(entries),
            "metrics": _score_predictions(predictions),
            "predictions": predictions,
        }
    return {
        "heldout_dataset": heldout_dataset,
        "source_datasets": sorted({row["dataset_id"] for row in source}),
        "target_episode_count": len(target),
        "views": view_results,
    }


def _macro_results(folds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for view in VIEWS:
        rows = [fold["views"][view]["metrics"] for fold in folds]
        balanced = [
            float(row["balanced_accuracy"])
            for row in rows
            if row["balanced_accuracy"] is not None
        ]
        result[view] = {
            "dataset_macro_sign_accuracy": _mean(row["sign_accuracy"] for row in rows),
            "dataset_macro_balanced_accuracy": _mean(balanced) if balanced else None,
            "dataset_macro_exposure_fraction": _mean(
                row["predicted_exposure_fraction"] for row in rows
            ),
            "dataset_macro_false_exposure_fraction": _mean(
                row["false_exposure_fraction_among_exposed"] for row in rows
            ),
            "dataset_macro_marginal_credit_policy_proxy_gain": _mean(
                row["marginal_credit_policy_proxy_gain"] for row in rows
            ),
            "harmful_dataset_count": sum(bool(row["proxy_is_harmful"]) for row in rows),
        }
    return result


def _route_verdict(macro: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pooled = macro["pooled"]
    global_only = macro["global_only"]
    local_episode = macro["local_episode"]
    local = macro["local_expected"]
    cohort = macro["local_plus_cohort"]

    def metric(row: dict[str, Any], name: str) -> float:
        value = row[name]
        return float(value) if value is not None else float("-inf")

    expected_accuracy_margin = metric(local, "dataset_macro_balanced_accuracy") - max(
        metric(pooled, "dataset_macro_balanced_accuracy"),
        metric(global_only, "dataset_macro_balanced_accuracy"),
    )
    expected_proxy_best = float(local["dataset_macro_marginal_credit_policy_proxy_gain"]) > max(
        float(pooled["dataset_macro_marginal_credit_policy_proxy_gain"]),
        float(global_only["dataset_macro_marginal_credit_policy_proxy_gain"]),
    )
    expected_harm_ok = int(local["harmful_dataset_count"]) <= min(
        int(pooled["harmful_dataset_count"]), int(global_only["harmful_dataset_count"])
    )
    expected_pass = (
        expected_accuracy_margin >= MATERIAL_BALANCED_ACCURACY_DELTA
        and expected_proxy_best
        and expected_harm_ok
    )

    cohort_accuracy_margin = metric(cohort, "dataset_macro_balanced_accuracy") - max(
        metric(pooled, "dataset_macro_balanced_accuracy"),
        metric(global_only, "dataset_macro_balanced_accuracy"),
        metric(local_episode, "dataset_macro_balanced_accuracy"),
        metric(local, "dataset_macro_balanced_accuracy"),
    )
    cohort_proxy_best = float(cohort["dataset_macro_marginal_credit_policy_proxy_gain"]) > max(
        float(pooled["dataset_macro_marginal_credit_policy_proxy_gain"]),
        float(global_only["dataset_macro_marginal_credit_policy_proxy_gain"]),
        float(local_episode["dataset_macro_marginal_credit_policy_proxy_gain"]),
        float(local["dataset_macro_marginal_credit_policy_proxy_gain"]),
    )
    cohort_harm_ok = int(cohort["harmful_dataset_count"]) <= min(
        int(pooled["harmful_dataset_count"]),
        int(global_only["harmful_dataset_count"]),
        int(local_episode["harmful_dataset_count"]),
        int(local["harmful_dataset_count"]),
    )
    cohort_pass = (
        cohort_accuracy_margin >= MATERIAL_BALANCED_ACCURACY_DELTA
        and cohort_proxy_best
        and cohort_harm_ok
    )
    gates = {
        "material_balanced_accuracy_delta": MATERIAL_BALANCED_ACCURACY_DELTA,
        "expected_credit_accuracy_margin": expected_accuracy_margin,
        "expected_credit_proxy_best": expected_proxy_best,
        "expected_credit_harm_not_worse": expected_harm_ok,
        "expected_credit_route_pass": expected_pass,
        "cohort_context_accuracy_margin": cohort_accuracy_margin,
        "cohort_context_proxy_best": cohort_proxy_best,
        "cohort_context_harm_not_worse": cohort_harm_ok,
        "cohort_context_route_pass": cohort_pass,
    }
    if cohort_pass:
        return "COHORT_CONTEXT_INCREMENTAL_SIGNAL", gates
    if expected_pass:
        return "ROW_EXPECTED_CREDIT_SIGNAL", gates
    return "CONTEXT_ROUTE_NOT_IDENTIFIED", gates


def run(root: Path) -> dict[str, Any]:
    source_path = root / SOURCE_REPORT_PATH
    source_report = json.loads(source_path.read_text(encoding="utf-8"))
    if source_report.get("target_query_opened") is not False:
        raise ValueError("Source credit report did not preserve the Target/Query boundary")
    if source_report.get("consumer_fit_count") != 219:
        raise ValueError("unexpected exposed actionability episode source")
    episodes = _build_episodes(source_report)
    if len(episodes) != 168:
        raise ValueError(f"expected 168 exposed episodes, got {len(episodes)}")
    datasets = sorted({row["dataset_id"] for row in episodes})
    if len(datasets) != 4 or any(dataset.startswith("uci") for dataset in datasets):
        raise ValueError("E2.2 requires exactly four non-Target Source datasets")

    folds = [_evaluate_fold(episodes, dataset) for dataset in datasets]
    macro = _macro_results(folds)
    verdict, gates = _route_verdict(macro)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_source_context_retrieval_feasibility",
        "causal_hypothesis": (
            "Outcome-free cohort and current action-set Context should make signed marginal "
            "credit more retrievable across datasets than pooled, global-only, or local-only "
            "evidence."
        ),
        "source_episode_report": SOURCE_REPORT_PATH,
        "configuration": {
            "task": "forecasting",
            "consumer": "Ridge(alpha=1.0, fit_intercept=True, solver=svd)",
            "program": "mask one observed training-target flatline interval",
            "episode_credit": "mean of fold-A and fold-B one-at-a-time marginal sMASE credit",
            "datasets": datasets,
            "episode_count": len(episodes),
            "retrieval": "source-bank z-score + Euclidean top-3 + mean signed credit",
            "evaluation": "leave one entire dataset out",
            "views": list(VIEWS),
            "dataset_identity_used_as_feature": False,
            "outcome_derived_query_feature": False,
            "consumer_fit_count": 0,
            "feature_names": {
                view: sorted(episodes[0]["features"][view])
                for view in VIEWS
                if view != "pooled"
            },
        },
        "duplicate_local_context_audit": _duplicate_context_audit(episodes),
        "lodo_folds": folds,
        "dataset_macro": macro,
        "gates": gates,
        "verdict": verdict,
        "next_step": (
            "Run one combined-policy replay with the frozen cohort-aware retrieval."
            if verdict == "COHORT_CONTEXT_INCREMENTAL_SIGNAL"
            else (
                "Represent Source evidence as row-expected signed credit with uncertainty; "
                "expose only stable positive evidence and otherwise abstain."
                if verdict == "ROW_EXPECTED_CREDIT_SIGNAL"
                else "Close this flatline family as a transferable Fast-Path Capability under the fixed Ridge protocol; retain marginal credit only for Slow-Path failure attribution."
            )
        ),
        "capability_claim": False,
        "memory_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "This is a zero-new-fit diagnostic on fully exposed Source outcomes. Top-k "
            "retrieval is a transparent feasibility probe, not a trained router or deployed "
            "Memory. The marginal-credit policy proxy is not additive full-policy utility."
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
