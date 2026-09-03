"""Diagnose whether low-budget support contexts represent visible query contexts.

This exposed Source-only diagnostic consumes the cached B=1/2 split outcomes from
the natural block action-value report.  It reconstructs only the eight visible
evaluation contexts per dataset, applies the frozen six-field context fingerprint,
and asks whether a fixed context-distance risk ranks already-executed harmful
splits above safe ones.  No Consumer is fit and no Target/UCI outcome is opened.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_query_context_cohort_reweighting import (
    _context_fingerprint,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    CONTEXT_LENGTH,
    FRESH_SPECS,
    SPECS,
    _read_object,
)


SCHEMA_VERSION = "e2-support-context-representativeness-diagnostic/1"
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_natural_block_action_value_headroom_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_support_context_representativeness_diagnostic_report.json"
)
BUDGETS = (1, 2)
PRIMARY_RISK = "mean_query_nearest_support_distance"
SECONDARY_RISKS = (
    "support_query_centroid_distance",
    "max_query_nearest_support_distance",
)
DESCRIPTIVE_BASELINE = "exact_support_gain_magnitude"
PREMISE_MIN_AUROC = 0.75


def _rank_auroc(rows: list[dict[str, Any]], score_name: str) -> float | None:
    """Return Mann-Whitney AUROC with average ranks for tied scores."""

    labels = [bool(row["harmful"]) for row in rows]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][score_name]))
    ranks = [0.0] * len(rows)
    start = 0
    while start < len(order):
        stop = start + 1
        value = float(rows[order[start]][score_name])
        while stop < len(order) and float(rows[order[stop]][score_name]) == value:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _risk_summary(rows: list[dict[str, Any]], score_name: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("risk summary requires executed splits")
    count = len(rows)
    harmful_count = sum(bool(row["harmful"]) for row in rows)
    beneficial_count = sum(float(row["raw_exact_grouped_query_gain"]) > 0.0 for row in rows)
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row[score_name]),
            str(row["dataset_id"]),
            int(row["budget"]),
            str(row["split_id"]),
        ),
    )
    top_count = math.ceil(count / 4)
    top = ranked[:top_count]
    top_keys = {
        (str(row["dataset_id"]), int(row["budget"]), str(row["split_id"]))
        for row in top
    }
    harmful_rankings: list[dict[str, Any]] = []
    scores = [float(row[score_name]) for row in rows]
    for row in rows:
        if not bool(row["harmful"]):
            continue
        score = float(row[score_name])
        greater = sum(value > score for value in scores)
        equal = sum(value == score for value in scores)
        less = sum(value < score for value in scores)
        harmful_rankings.append(
            {
                "dataset_id": row["dataset_id"],
                "budget": row["budget"],
                "split_id": row["split_id"],
                "risk": score,
                "descending_risk_midrank": 1.0 + greater + (equal - 1) / 2.0,
                "empirical_cdf_risk_percentile": (less + equal) / count,
                "in_fixed_top_quartile": (
                    str(row["dataset_id"]),
                    int(row["budget"]),
                    str(row["split_id"]),
                )
                in top_keys,
            }
        )
    harmful_in_top = sum(bool(row["harmful"]) for row in top)
    beneficial_in_top = sum(
        float(row["raw_exact_grouped_query_gain"]) > 0.0 for row in top
    )
    positive_gain_foregone = sum(
        max(float(row["raw_exact_grouped_query_gain"]), 0.0) for row in top
    )
    mean_before = statistics.fmean(
        float(row["raw_exact_grouped_query_gain"]) for row in rows
    )
    top_identities = {id(row) for row in top}
    mean_after = statistics.fmean(
        0.0
        if id(row) in top_identities
        else float(row["raw_exact_grouped_query_gain"])
        for row in rows
    )
    return {
        "higher_score_means_higher_predicted_harm_risk": True,
        "executed_split_count": count,
        "harmful_split_count": harmful_count,
        "beneficial_split_count": beneficial_count,
        "rank_based_auroc": _rank_auroc(rows, score_name),
        "harmful_split_risk_rankings": harmful_rankings,
        "fixed_top_quartile": {
            "definition": "highest-risk ceil(executed_split_count / 4) splits",
            "abstained_split_count": top_count,
            "abstained_split_fraction": top_count / count,
            "harmful_split_recall": (
                harmful_in_top / harmful_count if harmful_count else None
            ),
            "beneficial_split_abstention_count": beneficial_in_top,
            "beneficial_split_abstention_rate": (
                beneficial_in_top / beneficial_count if beneficial_count else None
            ),
            "positive_raw_exact_query_gain_foregone_sum": positive_gain_foregone,
            "mean_raw_exact_query_gain_before_abstention": mean_before,
            "mean_raw_exact_query_gain_after_abstention": mean_after,
            "mean_gain_delta_after_abstention": mean_after - mean_before,
        },
    }


def _load_eval_fingerprints(
    np: Any,
    *,
    root: Path,
    source_report: dict[str, Any],
) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
        _center_scale,
    )

    specs = {**SPECS, **FRESH_SPECS}
    datasets = {
        str(row["dataset_id"]): row for row in source_report["dataset_evidence"]
    }
    if set(datasets) != set(specs) or any(name.startswith("uci") for name in datasets):
        raise ValueError("expected exactly four exposed non-UCI Source datasets")
    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    eval_uids = [
        str(uid)
        for dataset in datasets.values()
        for uid in dataset["evaluation_uids"]
    ]
    if len(eval_uids) != 32 or len(set(eval_uids)) != 32:
        raise ValueError("expected four disjoint eight-series evaluation rosters")
    values = _load_values(
        [records[uid] for uid in eval_uids],
        root / "data/benchmark_v0_2/clean_base",
    )

    result: dict[str, Any] = {}
    for dataset_id, dataset in datasets.items():
        spec = specs[dataset_id]
        period = int(spec["period"])
        train_stop = int(spec["train_stop"])
        fingerprints: list[Any] = []
        uids = [str(uid) for uid in dataset["evaluation_uids"]]
        for uid in uids:
            raw = values[uid]
            context = np.asarray(
                raw[train_stop - CONTEXT_LENGTH : train_stop], dtype=np.float64
            )
            if context.shape != (CONTEXT_LENGTH,) or not np.isfinite(context).all():
                raise ValueError(f"invalid visible evaluation context: {dataset_id}/{uid}")
            center, scale, method = _center_scale(context)
            if method == "scale_floor_fallback":
                raise ValueError(f"invalid evaluation context scale: {dataset_id}/{uid}")
            fingerprints.append(
                _context_fingerprint(np, (context - center) / scale, period)
            )
        raw_matrix = np.asarray(fingerprints, dtype=np.float64)
        center = np.mean(raw_matrix, axis=0)
        scale = np.std(raw_matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (raw_matrix - center[None, :]) / scale[None, :]
        if standardized.shape != (8, 6) or not np.isfinite(standardized).all():
            raise RuntimeError(f"invalid standardized fingerprints: {dataset_id}")
        result[dataset_id] = {"uids": uids, "fingerprints": standardized}
    return result


def _split_rows(
    np: Any,
    *,
    source_report: dict[str, Any],
    fingerprints: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    accounting: dict[str, Any] = {}
    for dataset in source_report["dataset_evidence"]:
        dataset_id = str(dataset["dataset_id"])
        matrix = np.asarray(
            fingerprints[dataset_id]["fingerprints"], dtype=np.float64
        )
        diagnostic = dataset["target_only_feedback_budget_diagnostic"]["budgets"]
        dataset_accounting: dict[str, Any] = {}
        for budget in BUDGETS:
            rows = diagnostic[str(budget)]["split_evidence"]
            execute_count = 0
            for split in rows:
                support_indices = [int(value) for value in split["support_uid_indices"]]
                query_indices = [int(value) for value in split["query_uid_indices"]]
                if (
                    len(support_indices) != budget
                    or len(query_indices) != 8 - budget
                    or sorted(support_indices + query_indices) != list(range(8))
                ):
                    raise ValueError(f"split geometry changed: {dataset_id}/B={budget}")
                support = matrix[support_indices]
                query = matrix[query_indices]
                distances = np.linalg.norm(
                    query[:, None, :] - support[None, :, :], axis=2
                )
                nearest = np.min(distances, axis=1)
                risks = {
                    PRIMARY_RISK: float(np.mean(nearest)),
                    SECONDARY_RISKS[0]: float(
                        np.linalg.norm(np.mean(query, axis=0) - np.mean(support, axis=0))
                    ),
                    SECONDARY_RISKS[1]: float(np.max(nearest)),
                    DESCRIPTIVE_BASELINE: abs(
                        float(split["exact_grouped_support_gain"])
                    ),
                }
                if split["guard_decision"] != "EXECUTE":
                    continue
                execute_count += 1
                query_gain = float(split["raw_exact_grouped_query_gain"])
                executed.append(
                    {
                        "dataset_id": dataset_id,
                        "budget": budget,
                        "split_id": str(split["split_id"]),
                        "support_uid_indices": support_indices,
                        "query_uid_indices": query_indices,
                        "exact_grouped_support_gain": float(
                            split["exact_grouped_support_gain"]
                        ),
                        "raw_exact_grouped_query_gain": query_gain,
                        "harmful": query_gain < 0.0,
                        **risks,
                    }
                )
            dataset_accounting[str(budget)] = {
                "input_split_count": len(rows),
                "executed_split_count": execute_count,
                "excluded_original_guard_abstention_count": len(rows) - execute_count,
            }
        accounting[dataset_id] = dataset_accounting
    return executed, accounting


def _scope_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "primary": {
            "risk_name": PRIMARY_RISK,
            **_risk_summary(rows, PRIMARY_RISK),
        },
        "secondary": {
            name: _risk_summary(rows, name) for name in SECONDARY_RISKS
        },
        "non_context_descriptive_baseline": {
            "score_name": DESCRIPTIVE_BASELINE,
            "fixed_direction": (
                "larger exact support-gain magnitude is treated as higher harm risk"
            ),
            **_risk_summary(rows, DESCRIPTIVE_BASELINE),
        },
    }


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    source_report = _read_object(root / SOURCE_REPORT_PATH)
    if source_report.get("target_query_opened") is not False:
        raise ValueError("natural report did not preserve the Target/Query boundary")
    if int(source_report["compute_accounting"]["consumer_refit_count"]) != 0:
        raise ValueError("source report unexpectedly contains Consumer refits")
    fingerprints = _load_eval_fingerprints(
        np, root=root, source_report=source_report
    )
    executed, accounting = _split_rows(
        np, source_report=source_report, fingerprints=fingerprints
    )
    by_budget = {
        str(budget): _scope_summary(
            [row for row in executed if int(row["budget"]) == budget]
        )
        for budget in BUDGETS
    }
    combined = _scope_summary(executed)
    primary = combined["primary"]
    primary_auroc = primary["rank_based_auroc"]
    harm_recall = primary["fixed_top_quartile"]["harmful_split_recall"]
    premise_pass = (
        primary_auroc is not None
        and float(primary_auroc) >= PREMISE_MIN_AUROC
        and harm_recall == 1.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exposed_support_context_representativeness_first_fault_diagnostic",
        "source_report": SOURCE_REPORT_PATH,
        "configuration": {
            "budgets": list(BUDGETS),
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "diagnostic_only": True,
            "consumer_fit_count": 0,
            "fingerprint_source": (
                "run_e2_query_context_cohort_reweighting._context_fingerprint"
            ),
            "fingerprint_field_count": 6,
            "fingerprint_standardization": (
                "within each dataset using all eight visible eval-context fingerprints"
            ),
            "future_or_outcome_used_in_context_risk": False,
            "primary_risk": PRIMARY_RISK,
            "secondary_risks": list(SECONDARY_RISKS),
            "non_context_descriptive_baseline": DESCRIPTIVE_BASELINE,
            "risk_threshold_tuned": False,
            "top_quartile_rule": "highest-risk ceil(N/4) executed splits",
            "harmful_label": (
                "guard_decision == EXECUTE and raw_exact_grouped_query_gain < 0"
            ),
        },
        "split_accounting": accounting,
        "executed_split_evidence": executed,
        "by_budget": by_budget,
        "combined_b1_b2": combined,
        "gates": {
            "primary_min_rank_based_auroc": PREMISE_MIN_AUROC,
            "primary_rank_based_auroc": primary_auroc,
            "primary_top_quartile_harm_recall": harm_recall,
            "support_representativeness_premise_pass": premise_pass,
        },
        "verdict": (
            "SUPPORT_CONTEXT_REPRESENTATIVENESS_PREMISE_PASS"
            if premise_pass
            else "SUPPORT_CONTEXT_REPRESENTATIVENESS_PREMISE_FAIL"
        ),
        "next_step_if_fail": (
            "Close this frozen support-representativeness Context; do not add fields "
            "against the same exposed outcomes."
        ),
        "capability_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Exposed Source-only first-fault diagnostic over cached B=1/2 outcomes. "
            "Context risk uses visible evaluation histories only. It does not change "
            "the guard, execute a new policy, fit a Consumer, open UCI, promote a "
            "Capability, or establish Transfer."
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
