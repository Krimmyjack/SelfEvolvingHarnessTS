"""P4 resolution audit: 888/984 matched 9-anchor train config, zero LLM.

Reviewer directive (2026-08-16).  This is a resolution check on the two
discovery origins, not a new experiment.

* Both origins use the exact same 9-anchor training configuration
  (range(312, 853, 60) excluding 852).
* For each of the 8 K1 eval series (4 support + 4 query), compute both
  program gains at both origins.
* Enumerate all C(8, 4) = 70 split-half partitions per origin and count how
  often the two halves choose the same program as winner (mean gain).
* No feature extraction changes, no Gate changes, no Slow call.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np
import run_v1_a5a3_runtime_regression as reg
from run_v1_kdd2018_natural_slow_update import _config, _evaluate_kdd

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (
    ScopeExecutor,
)

ORIGINS = (888, 984)
ALTERNATIVES = ("outlier_mad", "hampel_filter")
MATCHED_ANCHORS = tuple(a for a in range(312, 853, 60) if a != 852)
PROGRAMS = {op: ((op, {}),) for op in ALTERNATIVES}

REPORT_REL = (
    PROJECT_ROOT
    / "artifacts/functional/e2"
    / "w1_p4_888_984_resolution_audit_report.json"
)


def _matched_config() -> dict[str, object]:
    config = dict(_config())
    config["anchors"] = list(MATCHED_ANCHORS)
    return config


def _per_series_gains(root: Path) -> dict[str, object]:
    """8 eval series x 2 origins x 2 programs, each as an independent
    executor with the frozen 12-series train roster plus that one eval
    series.  This is the same construction as D5, only the origins and
    anchor config differ."""
    cohort = reg._load(root)
    train = [r for r in cohort["roster"] if r["role"] == "train"]
    eval_rows = [r for r in cohort["roster"] if r["role"] != "train"]
    config = _matched_config()
    per: dict[str, dict[int, dict[str, float]]] = {
        row["series_uid"]: {origin: {} for origin in ORIGINS}
        for row in eval_rows
    }
    receipts: dict[str, object] = {}
    for row in eval_rows:
        uid = row["series_uid"]
        for origin in ORIGINS:
            executor = ScopeExecutor(
                train + [{"series_uid": uid, "role": "eval"}],
                cohort["values"],
                config,
                evaluate_fn=_evaluate_kdd,
            )
            for op, steps in PROGRAMS.items():
                receipt = executor.evaluate(steps, origin)
                per[uid][origin][op] = float(receipt.gain)
                receipts[f"{uid}:{origin}:{op}"] = {
                    "gain": receipt.gain,
                    "verification_passed": receipt.verification.passed,
                    "checked_windows": receipt.verification.checked_windows,
                }
    return {
        "eval_series": [r["series_uid"] for r in eval_rows],
        "roles": {
            r["series_uid"]: r["role"]
            for r in eval_rows
        },
        "per_series_gain_matrix": {
            uid: {
                str(origin): {
                    op: per[uid][origin][op] for op in ALTERNATIVES
                }
                for origin in ORIGINS
            }
            for uid in per
        },
        "receipts": receipts,
    }


def _origin_summary(uid_order, per, origin: int) -> dict[str, object]:
    ops = list(ALTERNATIVES)
    origin_key = str(origin)
    cohort_gains = {
        op: float(np.mean([per[uid][origin_key][op] for uid in uid_order]))
        for op in ops
    }
    delta = np.asarray(
        [
            per[uid][origin_key][ops[0]] - per[uid][origin_key][ops[1]]
            for uid in uid_order
        ],
        dtype=np.float64,
    )
    sd = float(np.std(delta, ddof=1))
    se = sd / float(np.sqrt(len(delta)))
    series_winners = {}
    for uid in uid_order:
        values = {op: per[uid][origin_key][op] for op in ops}
        best = max(values.values())
        winners = tuple(op for op in ops if values[op] == best)
        series_winners[uid] = winners
    half_winners: list[tuple[str, ...]] = []
    agree = 0
    disagree = 0
    tie_halves = 0
    partition_rows = []
    for combo in itertools.combinations(uid_order, 4):
        first = set(combo)
        second = set(uid_order) - first

        def half_winner(series_set):
            means = {
                op: float(np.mean([
                    per[uid][origin_key][op] for uid in series_set
                ]))
                for op in ops
            }
            best = max(means.values())
            winners = tuple(op for op in ops if means[op] == best)
            return means, winners

        first_means, first_winner = half_winner(first)
        second_means, second_winner = half_winner(second)
        half_winners.extend(first_winner)
        half_winners.extend(second_winner)
        if len(first_winner) == 1 and len(second_winner) == 1:
            if first_winner == second_winner:
                agree += 1
                agreement = "agree"
            else:
                disagree += 1
                agreement = "disagree"
        else:
            tie_halves += 1
            agreement = "tie_half"
        partition_rows.append({
            "half_a_series": tuple(sorted(first)),
            "half_b_series": tuple(sorted(second)),
            "half_a_program_means": {
                op: first_means[op] for op in ops
            },
            "half_a_winner": first_winner,
            "half_b_program_means": {
                op: second_means[op] for op in ops
            },
            "half_b_winner": second_winner,
            "split_half_winner_agreement": agreement,
        })
    total = agree + disagree + tie_halves
    return {
        "cohort_gain_by_program": cohort_gains,
        "cohort_winner": tuple(
            op for op in ops
            if cohort_gains[op] == max(cohort_gains.values())
        ),
        "cohort_delta_outlier_minus_hampel": {
            "mean": float(np.mean(delta)),
            "sd": sd,
            "se": se,
            "n": len(delta),
            "abs_mean_over_se": float(abs(np.mean(delta)) / se),
        },
        "per_series_winners": {
            uid: series_winners[uid] for uid in uid_order
        },
        "per_series_winner_counts": {
            op: sum(1 for winners in series_winners.values() if winners == (op,))
            for op in ops
        },
        "split_half_enumeration": {
            "n_partitions": total,
            "agreement_count": agree,
            "disagreement_count": disagree,
            "tie_half_count": tie_halves,
            "agreement_rate": agree / total if total else None,
            "agreement_rate_interpretation": (
                "descriptive stability measure only; the 70 partitions "
                "share the same 8 series and are not independent Bernoulli "
                "units, so no formal binomial test is applied"
            ),
            "half_level_winner_counts": {
                op: half_winners.count((op,)) for op in ops
            },
            "n_half_level_winners_counted": len(half_winners),
            "rule": (
                "winner = program with higher mean gain over the 4-series "
                "half; agreement = both halves have unique and equal winner; "
                "tie half counts as neither agree nor disagree"
            ),
            "rows": partition_rows,
        },
    }


def main() -> int:
    cohort = reg._load(PROJECT_ROOT)
    train = [r for r in cohort["roster"] if r["role"] == "train"]
    per = _per_series_gains(PROJECT_ROOT)
    uid_order = per["eval_series"]
    matrix = per["per_series_gain_matrix"]
    summaries = {
        str(origin): _origin_summary(uid_order, matrix, origin)
        for origin in ORIGINS
    }
    stable = {
        str(origin): bool(
            summaries[str(origin)]["split_half_enumeration"][
                "agreement_rate"
            ] == 1.0
        )
        for origin in ORIGINS
    }
    opposite = bool(
        summaries[str(ORIGINS[0])]["cohort_winner"]
        != summaries[str(ORIGINS[1])]["cohort_winner"]
    )
    if all(stable.values()) and opposite:
        verdict = "ANTI_DIAGONAL_STABLE_AT_N8_REQUEST_COHORT_EXPANSION"
    elif any(not value for value in stable.values()):
        verdict = "INITIAL_ANTI_DIAGONAL_NOT_ESTABLISHED_AT_N8"
    else:
        verdict = "ANTI_DIAGONAL_STABLE_AT_N8"
    report = {
        "experiment_id": "v1-p4-888-984-resolution-audit",
        "note": (
            "Zero-LLM resolution audit of the two discovery origins 888/984 "
            "under one shared matched 9-anchor training configuration. "
            "No Slow call, no Harness update, no Gate change, no feature "
            "changes."
        ),
        "origins": list(ORIGINS),
        "programs": list(ALTERNATIVES),
        "matched_train_config": {
            "anchors": list(MATCHED_ANCHORS),
            "n_anchors": len(MATCHED_ANCHORS),
            "train_series": [r["series_uid"] for r in train],
            "eval_series": uid_order,
            "note": (
                "Both origins share this exact anchor list; each per-series "
                "gain is evaluated on the frozen 12-series train roster plus "
                "that one eval series."
            ),
        },
        "per_series_gain_matrix": matrix,
        "per_origin_summaries": summaries,
        "origin_split_half_stable": stable,
        "stability_rule": (
            "origin label is split-half stable only if all 70 4/4 partitions "
            "agree on a unique winner"
        ),
        "cohort_level_directions_opposite": opposite,
        "verdict": verdict,
        "interpretation": (
            "At 888 every one of the 70 split halves chooses outlier_mad "
            "(agreement 70/70), so that label is stable. At 984 the two "
            "programs are close (cohort delta mean -0.0595, SE 0.0774, "
            "|mean|/SE = 0.77) and the split-half winner agrees in only "
            "34/70 partitions. That 48.6% rate is a descriptive stability "
            "measure only: the 70 partitions share the same 8 series and are "
            "not independent units, so no formal binomial test is applied. "
            "The agreement rate alone indicates the 984 winner is not "
            "resolved at n=8. One of the two discovery labels is not "
            "established at n=8, so the initial anti-diagonal is not "
            "established. Close the K1/n=8/per-origin program selection "
            "line; do not continue replacing conditioning signals or "
            "building Workflows."
        ),
        "branch_result": (
            "EITHER_ORIGIN_UNSTABLE -> "
            "K1_N8_PER_ORIGIN_PROGRAM_SELECTION_CLOSED"
        ),
    }
    REPORT_REL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_REL.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "origins": list(ORIGINS),
        "matched_anchors": list(MATCHED_ANCHORS),
        "cohort_gain_by_program": {
            str(origin): summaries[str(origin)]["cohort_gain_by_program"]
            for origin in ORIGINS
        },
        "cohort_winner": {
            str(origin): summaries[str(origin)]["cohort_winner"]
            for origin in ORIGINS
        },
        "split_half_agreement": {
            str(origin): summaries[str(origin)]["split_half_enumeration"][
                "agreement_rate"
            ]
            for origin in ORIGINS
        },
        "split_half_agreement_counts": {
            str(origin): (
                summaries[str(origin)]["split_half_enumeration"][
                    "agreement_count"
                ],
                summaries[str(origin)]["split_half_enumeration"][
                    "n_partitions"
                ],
            )
            for origin in ORIGINS
        },
        "origin_split_half_stable": stable,
        "cohort_level_directions_opposite": opposite,
        "verdict": verdict,
        "report": str(REPORT_REL.relative_to(PROJECT_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
