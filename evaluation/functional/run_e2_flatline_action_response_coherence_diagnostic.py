"""Test one support-only Action-Response coherence Observation on W35 failures.

This exposed Source diagnostic reconstructs the W29 and W30 exact-algebra
episodes with one additional support-only scalar.  It then replays the frozen
W35 signed-bank multiscale decisions and asks whether low coherence strictly
separates the two correct harmful GEFCom abstentions from the three false
beneficial FRED/Traffic abstentions.  No Consumer is fit, no retrieval or Guard
is changed, and no Target/UCI surface is read.
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
    _build_episodes,
    _source_dataset_advice,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_signed_episode_supply_lodo import (
    W29_REPORT_PATH,
    _build_w30_episodes,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    _read_object,
)


SCHEMA_VERSION = "e2-flatline-action-response-coherence-diagnostic/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_flatline_action_response_coherence_diagnostic_report.json"
)
OBSERVATION = "action__support_exact_singleton_sign_coherence"
CORRECT_DATASET = "gefcom2012_load"
FALSE_DATASETS = frozenset(
    {"legacy_monash:fred_md", "monash:traffic_hourly"}
)
EXPECTED_CORRECT_COUNT = 2
EXPECTED_FALSE_COUNT = 3


def _rank_auroc(rows: list[dict[str, Any]]) -> float | None:
    """Return average-rank AUROC for low coherence as predicted harm risk."""

    labels = [bool(row["harmful"]) for row in rows]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    risks = [1.0 - float(row[OBSERVATION]) for row in rows]
    order = sorted(range(len(rows)), key=lambda index: risks[index])
    ranks = [0.0] * len(rows)
    start = 0
    while start < len(order):
        stop = start + 1
        value = risks[order[start]]
        while stop < len(order) and risks[order[stop]] == value:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average_rank
        start = stop
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _captured_coherence(episode: dict[str, Any]) -> float:
    value = episode.get(OBSERVATION)
    if value is None:
        raise ValueError(
            f"locally executed episode lacks captured support coherence: "
            f"{episode['episode_id']}"
        )
    coherence = float(value)
    if not 0.0 <= coherence <= 1.0:
        raise ValueError(f"invalid support coherence: {episode['episode_id']}")
    return coherence


def _decision_rows(np: Any, episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset_id"]) for row in episodes})
    for heldout in datasets:
        source = [
            row
            for row in episodes
            if row["dataset_id"] != heldout and row["locally_executes"]
        ]
        target = [
            row
            for row in episodes
            if row["dataset_id"] == heldout and row["locally_executes"]
        ]
        if not source or not target:
            raise ValueError(f"invalid W35 LODO geometry: {heldout}")
        for episode in target:
            advice = _source_dataset_advice(
                np,
                source=source,
                target=episode,
                view="multiscale_context",
            )
            gain = float(episode["query_gain_if_locally_executed"])
            rows.append(
                {
                    "episode_id": episode["episode_id"],
                    "cohort_id": episode.get("cohort_id", "w29_development_exposed"),
                    "dataset_id": heldout,
                    "seed": int(episode["seed"]),
                    "budget": int(episode["budget"]),
                    "split_id": str(episode["split_id"]),
                    OBSERVATION: _captured_coherence(episode),
                    "query_gain_if_locally_executed": gain,
                    "harmful": gain < 0.0,
                    "signed_multiscale_decision": (
                        "EXECUTE" if advice["execute"] else "ABSTAIN"
                    ),
                }
            )
    return rows


def _failure_card(row: dict[str, Any], failure_type: str) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "cohort_id": row["cohort_id"],
        "dataset_id": row["dataset_id"],
        "seed": row["seed"],
        "budget": row["budget"],
        "split_id": row["split_id"],
        OBSERVATION: row[OBSERVATION],
        "diagnostic_failure_type": failure_type,
        "query_gain_if_locally_executed_exposed_label": row[
            "query_gain_if_locally_executed"
        ],
        "local_decision": "EXECUTE",
        "signed_multiscale_decision": "ABSTAIN",
    }


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    historical_w29 = _read_object(root / W29_REPORT_PATH)
    if historical_w29.get("target_query_opened") is not False:
        raise ValueError("W29 did not preserve the Target boundary")
    proxy_report = run_action_conditioned_proxy(
        root, capture_support_action_response=True
    )
    if proxy_report.get("target_query_opened") is not False:
        raise ValueError("W29 proxy replay did not preserve the Target boundary")
    w29_report = dict(historical_w29)
    w29_report["action_value_guard_budget_diagnostic"] = proxy_report[
        "action_value_guard_budget_diagnostic"
    ]
    w29 = _build_episodes(
        np,
        root=root,
        report=w29_report,
        capture_support_action_response=True,
    )
    for episode in w29:
        episode["cohort_id"] = "w29_development_exposed"
        episode["episode_id"] = f"w29|{episode['episode_id']}"
    w30, w30_compute = _build_w30_episodes(
        np, root=root, capture_support_action_response=True
    )
    episodes = [*w29, *w30]
    datasets = sorted({str(row["dataset_id"]) for row in episodes})
    if len(w29) != 168 or len(w30) != 126 or len(episodes) != 294:
        raise ValueError("W35 signed episode geometry changed")
    if len(datasets) != 5 or any(name.startswith("uci") for name in datasets):
        raise ValueError("W36 requires exactly five exposed non-UCI Source families")

    decision_rows = _decision_rows(np, episodes)
    abstained = [
        row
        for row in decision_rows
        if row["signed_multiscale_decision"] == "ABSTAIN"
    ]
    correct = [
        row
        for row in abstained
        if row["dataset_id"] == CORRECT_DATASET and row["harmful"]
    ]
    false = [
        row
        for row in abstained
        if row["dataset_id"] in FALSE_DATASETS and not row["harmful"]
    ]
    failure_rows = [*correct, *false]
    strict_separation = (
        len(correct) == EXPECTED_CORRECT_COUNT
        and len(false) == EXPECTED_FALSE_COUNT
        and max(float(row[OBSERVATION]) for row in correct)
        < min(float(row[OBSERVATION]) for row in false)
    )
    locally_executed = decision_rows
    w29_compute = proxy_report["action_value_guard_budget_diagnostic"][
        "compute_accounting"
    ]
    w29_other_group_solves = int(
        proxy_report["compute_accounting"]["grouped_small_matrix_solve_count"]
    )
    w29_action_group_solves = int(
        w29_compute["h1_grouped_small_matrix_solve_count"]
    )
    w30_group_solves = int(w30_compute["grouped_small_matrix_solve_count"])
    reference_solve_count = int(w29_compute["reference_solve_count"]) + int(
        w30_compute["reference_solve_count"]
    )
    passed = strict_separation
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": (
            "zero_new_fit_exposed_program_conditioned_observation_diagnostic"
        ),
        "causal_hypothesis": (
            "Harmful GEFCom actions rejected by the W35 signed guard have lower "
            "support exact-singleton sign coherence than the beneficial FRED/Traffic "
            "actions that the same frozen guard rejected incorrectly."
        ),
        "configuration": {
            "observation": OBSERVATION,
            "observation_definition": (
                "Among the existing proxy-selected actions, compute each action's "
                "exact singleton gain averaged only over the current support indices; "
                "return the fraction whose exact mean gain is strictly positive."
            ),
            "singleton_gain_sum_used_as_group_utility": False,
            "query_response_used_in_observation": False,
            "query_response_role": "exposed diagnostic label only",
            "retrieval_or_guard_changed": False,
            "threshold_tuned": False,
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "datasets": datasets,
            "episode_count": len(episodes),
            "consumer_fit_count": 0,
            "target_query_opened": False,
        },
        "compute_accounting": {
            "consumer_fit_count": 0,
            "reference_solve_count": reference_solve_count,
            "w29_reference_solve_count": int(w29_compute["reference_solve_count"]),
            "w30_reference_solve_count": int(w30_compute["reference_solve_count"]),
            "grouped_small_matrix_solve_count": (
                w29_other_group_solves
                + w29_action_group_solves
                + w30_group_solves
            ),
            "w29_attribution_grouped_small_matrix_solve_count": (
                w29_other_group_solves
            ),
            "w29_action_value_grouped_small_matrix_solve_count": (
                w29_action_group_solves
            ),
            "w30_action_value_grouped_small_matrix_solve_count": w30_group_solves,
            "per_action_consumer_refit_count": 0,
            "grouped_consumer_refit_count": 0,
        },
        "failure_pattern_card": {
            "selection": (
                "local EXECUTE and frozen W35 signed multiscale ABSTAIN; correct is "
                "harmful GEFCom, false is beneficial FRED/Traffic"
            ),
            "correct_harmful_gefcom": [
                _failure_card(row, "CORRECT_HARMFUL_ABSTENTION") for row in correct
            ],
            "false_beneficial_fred_traffic": [
                _failure_card(row, "FALSE_BENEFICIAL_ABSTENTION") for row in false
            ],
        },
        "primary": {
            "expected_correct_harmful_count": EXPECTED_CORRECT_COUNT,
            "observed_correct_harmful_count": len(correct),
            "expected_false_beneficial_count": EXPECTED_FALSE_COUNT,
            "observed_false_beneficial_count": len(false),
            "max_correct_harmful_coherence": (
                max(float(row[OBSERVATION]) for row in correct) if correct else None
            ),
            "min_false_beneficial_coherence": (
                min(float(row[OBSERVATION]) for row in false) if false else None
            ),
            "strict_low_coherence_separation": strict_separation,
            "five_case_low_coherence_harm_risk_auroc": _rank_auroc(failure_rows),
        },
        "descriptive": {
            "locally_executed_episode_count": len(locally_executed),
            "locally_executed_harmful_episode_count": sum(
                bool(row["harmful"]) for row in locally_executed
            ),
            "all_locally_executed_low_coherence_harm_risk_auroc": (
                _rank_auroc(locally_executed)
            ),
            "mean_coherence_by_outcome": {
                "harmful": statistics.fmean(
                    float(row[OBSERVATION])
                    for row in locally_executed
                    if row["harmful"]
                ),
                "beneficial": statistics.fmean(
                    float(row[OBSERVATION])
                    for row in locally_executed
                    if not row["harmful"]
                ),
            },
        },
        "gates": {
            "primary_requires_exactly_two_correct_and_three_false_cases": True,
            "primary_requires_max_correct_below_min_false": True,
            "primary_pass": passed,
        },
        "verdict": (
            "ACTION_RESPONSE_COHERENCE_DIAGNOSTIC_PASS"
            if passed
            else "ACTION_RESPONSE_COHERENCE_DIAGNOSTIC_FAIL"
        ),
        "next_step_if_pass": (
            "Freeze this one Observation and compile one next fresh Source patch; "
            "do not tune a threshold on these exposed outcomes."
        ),
        "next_step_if_fail": (
            "Close the Flatline Context family. Do not add Context fields, tune "
            "retrieval, or add Memory against these exposed outcomes."
        ),
        "harness_update_applied": False,
        "capability_claim": False,
        "memory_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Exposed Source-only zero-new-fit diagnostic. PASS only permits one future "
            "frozen patch; FAIL closes the current Flatline Context family."
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
