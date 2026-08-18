"""Test whether peer outcome contradiction localizes useful training-block removal.

This development-only, Source-only feasibility slice consumes W23's already
exposed singleton action credits as labels.  Its sole new Observation compares
one standardized training target block with the target-block consensus of the
five closest training rows under standardized 192-step history distance.

No Consumer is fit, no evaluation future is loaded, and no Target/UCI data is
opened.  The contradiction score is an Observation, not Utility, a Router, a
Capability, or Memory.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_natural_block_action_value_headroom import (
    BLOCKS,
    EXPECTED_TRAINING_ROWS,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    ANCHORS,
    CONTEXT_LENGTH,
    FRESH_SPECS,
    HORIZON,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-peer-outcome-contradiction-feasibility/1"
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_natural_block_action_value_headroom_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_peer_outcome_contradiction_feasibility_report.json"
)
PEER_K = 5
TOP_QUARTILE_COUNT_PER_BLOCK = math.ceil(EXPECTED_TRAINING_ROWS / 4)
DISPERSION_EPSILON = 1e-6

GATE_MIN_MACRO_AUROC = 0.60
GATE_MIN_DATASETS_AUROC_ABOVE_CHANCE = 3
GATE_MIN_DATASETS_TOP_MEAN_POSITIVE = 3
GATE_MIN_MACRO_POSITIVE_FRACTION_UPLIFT = 0.10


def _tie_aware_auroc(scores: list[float], labels: list[bool]) -> float | None:
    """Mann-Whitney AUROC with average ranks for tied scores."""

    if len(scores) != len(labels) or not scores:
        raise ValueError("AUROC requires aligned non-empty score and label arrays")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        stop = start + 1
        score = scores[order[start]]
        while stop < len(order) and scores[order[stop]] == score:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, labels) if label
    )
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _mean_exact_credit(action: dict[str, Any]) -> float:
    folds = action["fold_attribution"]
    return statistics.fmean(
        float(folds[name]["exact_singleton_attribution_credit"])
        for name in ("fold_a", "fold_b")
    )


def _metric_summary(
    rows: list[dict[str, Any]], *, selected_count: int
) -> dict[str, Any]:
    if not rows or not 0 < selected_count <= len(rows):
        raise ValueError("invalid metric-summary geometry")
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["contradiction_score"]),
            int(row["row_index"]),
        ),
    )
    selected = ranked[:selected_count]
    all_positive_fraction = statistics.fmean(
        float(row["exact_credit_positive"]) for row in rows
    )
    selected_positive_fraction = statistics.fmean(
        float(row["exact_credit_positive"]) for row in selected
    )
    return {
        "action_count": len(rows),
        "exact_positive_action_count": sum(
            bool(row["exact_credit_positive"]) for row in rows
        ),
        "tie_aware_auroc": _tie_aware_auroc(
            [float(row["contradiction_score"]) for row in rows],
            [bool(row["exact_credit_positive"]) for row in rows],
        ),
        "all_action_mean_exact_credit": statistics.fmean(
            float(row["mean_exact_singleton_credit"]) for row in rows
        ),
        "all_action_positive_fraction": all_positive_fraction,
        "top_quartile_action_count": selected_count,
        "top_quartile_mean_exact_credit": statistics.fmean(
            float(row["mean_exact_singleton_credit"]) for row in selected
        ),
        "top_quartile_positive_fraction": selected_positive_fraction,
        "top_quartile_positive_fraction_uplift": (
            selected_positive_fraction - all_positive_fraction
        ),
        "contradiction_score_min": min(
            float(row["contradiction_score"]) for row in rows
        ),
        "contradiction_score_median": statistics.median(
            float(row["contradiction_score"]) for row in rows
        ),
        "contradiction_score_max": max(
            float(row["contradiction_score"]) for row in rows
        ),
    }


def _load_training_windows(
    np: Any,
    *,
    root: Path,
    source_report: dict[str, Any],
) -> dict[str, dict[tuple[str, int, int], dict[str, Any]]]:
    """Load only the W23 training rows and reconstruct standardized windows."""

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
        _center_scale,
    )

    specs = {**SPECS, **FRESH_SPECS}
    report_by_dataset = {
        str(row["dataset_id"]): row for row in source_report["dataset_evidence"]
    }
    if set(report_by_dataset) != set(specs):
        raise ValueError("W23 Source dataset roster changed")
    if any(dataset_id.startswith("uci") for dataset_id in report_by_dataset):
        raise ValueError("UCI is forbidden in this Source feasibility slice")

    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    # W23's action row keys are the frozen training-row roster.  Reconstructing
    # the later two datasets through `_fresh_roster` would unnecessarily scan
    # the eval interval for finite-value admission, so this runner deliberately
    # validates the cached row binding against the registry and loads only those
    # training UIDs.
    train_uids_by_dataset: dict[str, set[str]] = {}
    for dataset_id, dataset_report in report_by_dataset.items():
        row_keys = {
            (
                str(action["row_key"][0]),
                int(action["row_key"][1]),
                int(action["row_key"][2]),
            )
            for action in dataset_report["singleton_action_attribution"]
        }
        train_uids = {key[0] for key in row_keys}
        if (
            len(row_keys) != EXPECTED_TRAINING_ROWS
            or len(train_uids) != 12
            or {key[1] for key in row_keys} != set(ANCHORS)
            or {key[2] for key in row_keys} != {HORIZON}
        ):
            raise ValueError(f"W23 frozen training-row roster changed: {dataset_id}")
        for uid in train_uids:
            record = records.get(uid)
            if record is None or record.dataset_id != dataset_id:
                raise ValueError(f"W23 training UID/registry mismatch: {dataset_id}/{uid}")
        train_uids_by_dataset[dataset_id] = train_uids

    all_train_uids = sorted(
        uid for uids in train_uids_by_dataset.values() for uid in uids
    )
    values = _load_values(
        [records[uid] for uid in all_train_uids],
        root / "data/benchmark_v0_2/clean_base",
    )

    windows_by_dataset: dict[
        str, dict[tuple[str, int, int], dict[str, Any]]
    ] = {}
    for dataset_id, dataset_report in report_by_dataset.items():
        row_keys = {
            (
                str(action["row_key"][0]),
                int(action["row_key"][1]),
                int(action["row_key"][2]),
            )
            for action in dataset_report["singleton_action_attribution"]
        }
        if len(row_keys) != EXPECTED_TRAINING_ROWS:
            raise ValueError(f"expected 72 W23 training rows: {dataset_id}")
        if {key[0] for key in row_keys} != train_uids_by_dataset[dataset_id]:
            raise ValueError(f"W23 action rows disagree with Source roster: {dataset_id}")
        cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        for uid, anchor, horizon in row_keys:
            if horizon != HORIZON:
                raise ValueError("W23 action horizon changed")
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
                raise ValueError(f"invalid W23 training window: {dataset_id}/{uid}/{anchor}")
            cache[(uid, anchor, horizon)] = {
                "series_uid": uid,
                "standardized_context": (context - center) / scale,
                "standardized_target": (target - center) / scale,
            }
        windows_by_dataset[dataset_id] = cache
    return windows_by_dataset


def _dataset_observations(
    np: Any,
    *,
    dataset_report: dict[str, Any],
    windows: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    actions = list(dataset_report["singleton_action_attribution"])
    if len(actions) != EXPECTED_TRAINING_ROWS * len(BLOCKS):
        raise ValueError("W23 action count changed")

    row_index_to_key: dict[int, tuple[str, int, int]] = {}
    actions_by_block: dict[tuple[int, int], dict[int, dict[str, Any]]] = {
        block: {} for block in BLOCKS
    }
    for action in actions:
        row_index = int(action["row_index"])
        row_key = (
            str(action["row_key"][0]),
            int(action["row_key"][1]),
            int(action["row_key"][2]),
        )
        block = tuple(int(value) for value in action["block_half_open"])
        if block not in actions_by_block or row_index not in range(EXPECTED_TRAINING_ROWS):
            raise ValueError("W23 action geometry changed")
        prior = row_index_to_key.setdefault(row_index, row_key)
        if prior != row_key or row_index in actions_by_block[block]:
            raise ValueError("W23 row binding is inconsistent")
        actions_by_block[block][row_index] = action
    if set(row_index_to_key) != set(range(EXPECTED_TRAINING_ROWS)):
        raise ValueError("W23 row indices changed")
    if any(
        set(block_actions) != set(range(EXPECTED_TRAINING_ROWS))
        for block_actions in actions_by_block.values()
    ):
        raise ValueError("W23 block action coverage changed")

    contexts = np.asarray(
        [windows[row_index_to_key[index]]["standardized_context"] for index in range(EXPECTED_TRAINING_ROWS)],
        dtype=np.float64,
    )
    targets = np.asarray(
        [windows[row_index_to_key[index]]["standardized_target"] for index in range(EXPECTED_TRAINING_ROWS)],
        dtype=np.float64,
    )
    series_uids = [row_index_to_key[index][0] for index in range(EXPECTED_TRAINING_ROWS)]
    if contexts.shape != (EXPECTED_TRAINING_ROWS, CONTEXT_LENGTH) or targets.shape != (
        EXPECTED_TRAINING_ROWS,
        HORIZON,
    ):
        raise ValueError("reconstructed W23 training geometry changed")
    distances = np.linalg.norm(contexts[:, None, :] - contexts[None, :, :], axis=2)

    block_evidence: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []
    for block in BLOCKS:
        start, stop = block
        observations: list[dict[str, Any]] = []
        for row_index in range(EXPECTED_TRAINING_ROWS):
            eligible = [
                peer_index
                for peer_index in range(EXPECTED_TRAINING_ROWS)
                if series_uids[peer_index] != series_uids[row_index]
            ]
            ranked = sorted(
                eligible,
                key=lambda peer_index: (float(distances[row_index, peer_index]), peer_index),
            )
            peer_indices = ranked[:PEER_K]
            if len(peer_indices) != PEER_K:
                raise ValueError("insufficient cross-series peer support")
            peer_blocks = targets[peer_indices, start:stop]
            consensus = np.median(peer_blocks, axis=0)
            pointwise_peer_absolute_residual_median = np.median(
                np.abs(peer_blocks - consensus[None, :]), axis=0
            )
            peer_dispersion = float(
                np.mean(pointwise_peer_absolute_residual_median)
            )
            candidate_residual = float(
                np.mean(np.abs(targets[row_index, start:stop] - consensus))
            )
            contradiction = candidate_residual / (
                peer_dispersion + DISPERSION_EPSILON
            )
            action = actions_by_block[block][row_index]
            mean_exact_credit = _mean_exact_credit(action)
            observation = {
                "row_index": row_index,
                "block_half_open": list(block),
                "peer_indices": peer_indices,
                "peer_count": PEER_K,
                "candidate_to_consensus_mean_absolute_residual": candidate_residual,
                "peer_dispersion": peer_dispersion,
                "contradiction_score": contradiction,
                "mean_exact_singleton_credit": mean_exact_credit,
                "exact_credit_positive": mean_exact_credit > 0.0,
            }
            observations.append(observation)
            all_observations.append(observation)
        block_evidence.append(
            {
                "block_half_open": list(block),
                "metrics": _metric_summary(
                    observations, selected_count=TOP_QUARTILE_COUNT_PER_BLOCK
                ),
            }
        )

    top_selected: list[dict[str, Any]] = []
    for block in BLOCKS:
        rows = [
            row for row in all_observations if tuple(row["block_half_open"]) == block
        ]
        top_selected.extend(
            sorted(
                rows,
                key=lambda row: (
                    -float(row["contradiction_score"]),
                    int(row["row_index"]),
                ),
            )[:TOP_QUARTILE_COUNT_PER_BLOCK]
        )
    all_positive_fraction = statistics.fmean(
        float(row["exact_credit_positive"]) for row in all_observations
    )
    top_positive_fraction = statistics.fmean(
        float(row["exact_credit_positive"]) for row in top_selected
    )
    dataset_metrics = {
        "action_count": len(all_observations),
        "tie_aware_auroc": _tie_aware_auroc(
            [float(row["contradiction_score"]) for row in all_observations],
            [bool(row["exact_credit_positive"]) for row in all_observations],
        ),
        "all_action_mean_exact_credit": statistics.fmean(
            float(row["mean_exact_singleton_credit"]) for row in all_observations
        ),
        "all_action_positive_fraction": all_positive_fraction,
        "top_quartile_action_count": len(top_selected),
        "top_quartile_mean_exact_credit": statistics.fmean(
            float(row["mean_exact_singleton_credit"]) for row in top_selected
        ),
        "top_quartile_positive_fraction": top_positive_fraction,
        "top_quartile_positive_fraction_uplift": (
            top_positive_fraction - all_positive_fraction
        ),
    }
    return {
        "dataset_id": str(dataset_report["dataset_id"]),
        "training_row_count": len(row_index_to_key),
        "action_count": len(all_observations),
        "per_block": block_evidence,
        "metrics": dataset_metrics,
    }


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    source_report = _read_object(root / SOURCE_REPORT_PATH)
    if source_report.get("target_query_opened") is not False:
        raise ValueError("W23 did not preserve the Target/Query boundary")
    source_datasets = [
        str(row["dataset_id"]) for row in source_report["dataset_evidence"]
    ]
    if len(source_datasets) != 4 or len(set(source_datasets)) != 4:
        raise ValueError("expected four exposed W23 Source datasets")
    if any(dataset_id.startswith("uci") for dataset_id in source_datasets):
        raise ValueError("UCI is forbidden in this Source feasibility slice")

    windows = _load_training_windows(np, root=root, source_report=source_report)
    dataset_evidence = [
        _dataset_observations(
            np,
            dataset_report=dataset_report,
            windows=windows[str(dataset_report["dataset_id"])],
        )
        for dataset_report in source_report["dataset_evidence"]
    ]
    dataset_aurocs = [row["metrics"]["tie_aware_auroc"] for row in dataset_evidence]
    macro_auroc = (
        statistics.fmean(float(value) for value in dataset_aurocs)
        if all(value is not None for value in dataset_aurocs)
        else None
    )
    macro_top_positive_uplift = statistics.fmean(
        float(row["metrics"]["top_quartile_positive_fraction_uplift"])
        for row in dataset_evidence
    )
    datasets_auroc_above_chance = sum(
        value is not None and float(value) > 0.5 for value in dataset_aurocs
    )
    datasets_top_mean_positive = sum(
        float(row["metrics"]["top_quartile_mean_exact_credit"]) > 0.0
        for row in dataset_evidence
    )
    gates = {
        "thresholds_frozen_before_run": True,
        "dataset_macro_tie_aware_auroc_at_least_0_60": {
            "threshold": GATE_MIN_MACRO_AUROC,
            "observed": macro_auroc,
            "pass": macro_auroc is not None and macro_auroc >= GATE_MIN_MACRO_AUROC,
        },
        "at_least_3_of_4_datasets_auroc_above_0_5": {
            "threshold": GATE_MIN_DATASETS_AUROC_ABOVE_CHANCE,
            "observed": datasets_auroc_above_chance,
            "pass": datasets_auroc_above_chance
            >= GATE_MIN_DATASETS_AUROC_ABOVE_CHANCE,
        },
        "at_least_3_of_4_datasets_top_quartile_mean_exact_credit_positive": {
            "threshold": GATE_MIN_DATASETS_TOP_MEAN_POSITIVE,
            "observed": datasets_top_mean_positive,
            "pass": datasets_top_mean_positive
            >= GATE_MIN_DATASETS_TOP_MEAN_POSITIVE,
        },
        "dataset_macro_top_quartile_positive_fraction_uplift_at_least_0_10": {
            "threshold": GATE_MIN_MACRO_POSITIVE_FRACTION_UPLIFT,
            "observed": macro_top_positive_uplift,
            "pass": macro_top_positive_uplift
            >= GATE_MIN_MACRO_POSITIVE_FRACTION_UPLIFT,
        },
    }
    premise_pass = all(bool(row["pass"]) for row in gates.values() if isinstance(row, dict))
    verdict = (
        "PEER_OUTCOME_CONTRADICTION_FEASIBILITY_PASS"
        if premise_pass
        else "PEER_OUTCOME_CONTRADICTION_FEASIBILITY_FAIL"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "development_exposed_program_specific_observation_feasibility",
        "question": (
            "Does disagreement between a training target block and targets of fixed-K "
            "peers with similar visible history rank actions whose removal is beneficial?"
        ),
        "causal_hypothesis": (
            "A target block that contradicts the outcome consensus of training peers "
            "with similar standardized history is more likely to be a beneficial "
            "row x block removal action."
        ),
        "source_report": SOURCE_REPORT_PATH,
        "configuration": {
            "datasets": source_datasets,
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "development_only": True,
            "program_id": "mask_one_training_row_from_ridge_loss_for_one_12_step_output_block",
            "training_rows_per_dataset": EXPECTED_TRAINING_ROWS,
            "blocks": [list(block) for block in BLOCKS],
            "peer_k": PEER_K,
            "peer_scope": (
                "same dataset and same output block; exclude every row sharing the "
                "candidate series_uid; rank remaining rows by 192-step standardized-"
                "context Euclidean distance"
            ),
            "peer_target_consensus_formula": (
                "pointwise median of the K standardized peer target blocks"
            ),
            "peer_dispersion_formula": (
                "mean across 12 target points of the pointwise median across K peers "
                "of abs(peer_target - pointwise_peer_consensus)"
            ),
            "candidate_residual_formula": (
                "mean across 12 target points of abs(candidate_target - "
                "pointwise_peer_consensus)"
            ),
            "contradiction_score_formula": (
                "candidate_residual / (peer_dispersion + 1e-6)"
            ),
            "ranking_scope": "within each dataset x 12-step block",
            "top_quartile_count_per_block": TOP_QUARTILE_COUNT_PER_BLOCK,
            "outcome_label": (
                "positive iff mean(fold_a, fold_b exact_singleton_attribution_credit) > 0"
            ),
            "score_reads_training_context": True,
            "score_reads_training_target": True,
            "score_reads_evaluation_future": False,
            "score_reads_exact_credit": False,
            "score_uses_dataset_id_as_feature": False,
            "consumer_fit_count": 0,
        },
        "dataset_evidence": dataset_evidence,
        "dataset_macro": {
            "tie_aware_auroc": macro_auroc,
            "datasets_auroc_above_0_5": datasets_auroc_above_chance,
            "datasets_top_quartile_mean_exact_credit_positive": datasets_top_mean_positive,
            "top_quartile_mean_exact_credit": statistics.fmean(
                float(row["metrics"]["top_quartile_mean_exact_credit"])
                for row in dataset_evidence
            ),
            "all_action_positive_fraction": statistics.fmean(
                float(row["metrics"]["all_action_positive_fraction"])
                for row in dataset_evidence
            ),
            "top_quartile_positive_fraction": statistics.fmean(
                float(row["metrics"]["top_quartile_positive_fraction"])
                for row in dataset_evidence
            ),
            "top_quartile_positive_fraction_uplift": macro_top_positive_uplift,
        },
        "gates": gates,
        "verdict": verdict,
        "observation_supported": premise_pass,
        "utility_claim": False,
        "capability_claim": False,
        "memory_claim": False,
        "router_claim": False,
        "formal_transfer": False,
        "consumer_fit_count": 0,
        "target_query_opened": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "This is an exposed Source-only ranking feasibility test. The score is a "
            "Program-specific TRAIN observation and cached W23 exact singleton credit "
            "is used only as the exposed outcome label. No cohort is executed, no "
            "Consumer is fit, and no Utility, Capability, Memory, Router, or transfer "
            "claim is established."
        ),
        "next_step": (
            "Freeze this Observation unchanged for one existing Source grouped-workflow "
            "test; do not build Memory or add score fields."
            if premise_pass
            else "Close peer outcome contradiction as an applicability Observation; do not add fields."
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
