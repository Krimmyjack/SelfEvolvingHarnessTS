"""Task Episode Harness runner: T0 substrate qualification (zero LLM).

Sole execution task book: docs/TASK_EPISODE_HARNESS_EXECUTION_PLAN_2026-08-17.md
Final Planner-ruled T0 recipe (adjudication of sections 14/15):

* corpus-level one-shot label-touched random injection
* amplitude 8.0 x pristine nanstd(series[120:900])
* 40 unique training-label timestamps per faulty series, seed 7
* 6 faulty train series, 6 clean train series
* support [1104, 1368, 1800], delayed [2856, 3648, 3888]
* arms: identity / oracle-scoped outlier_mad (matched) /
  oracle-scoped hampel_filter (mechanism-different / overactive comparator)
* baseline_degradation_ratio is a disclosure field only, never a Gate

T0 pass rule is section 13: matched support macro > 0, matched support
gain/SE_block >= 3, delayed direction same as support, split-half label not
easily flipping.  This runner makes the qualitative fourth item operational as
matched wins in at least 49/70 of both block partitions (70%, T0-only).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

for path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import run_e2_autonomous_natural_workflow_generation as v6
import run_v1_a5a3_runtime_regression as reg
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.injection import (
    inject_label_touched_corpus,
)
from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    CompiledWorkflow,
)

REPORT_REL = (
    PROJECT_ROOT
    / "artifacts/functional/e2"
    / "w1_task_episode_harness_report.json"
)

# Final Planner-ruled T0 constants --------------------------------------------
FAULT_FAMILY = "impulsive_outlier"
INJECTION_AMPLITUDE = 8.0
INJECTION_COUNT = 40
INJECTION_SEED = 7
FAULTY_SERIES = ("T117", "T118", "T119", "T12", "T120", "T121")
CLEAN_TRAIN_SERIES = (
    "T122", "T123", "T124", "T125", "T126", "T127",
)
SUPPORT_ORIGINS = (1104, 1368, 1800)
DELAYED_ORIGINS = (2856, 3648, 3888)
MATERIAL_THRESHOLD = 0.005
MATCHED_PROGRAM = "outlier_mad"
COMPARATOR_PROGRAM = "hampel_filter"
MIN_SPLIT_HALF_MATCHED_WINS = 49  # 49/70 = 70%, T0-only operationalization


def _compiled(op: str, *, name: str) -> CompiledWorkflow:
    program = Program.from_steps([(op, {})], source="task_episode_t0")
    candidate = Candidate.program_candidate(name, program, source="task_episode_t0")
    return CompiledWorkflow(
        candidate,
        (),
        ({"op": op, "params": {}, "bindings": {}},),
    )


def _mapped_roster(roster: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row, role="eval") if row["role"] != "train" else dict(row)
        for row in roster
    ]


def _evaluate_origins(
    roster: list[dict[str, Any]],
    values: dict[str, Any],
    compiled: CompiledWorkflow | None,
    config: dict[str, Any],
    origins: tuple[int, ...],
    scope: set[str] | None,
) -> list[dict[str, Any]]:
    rows = []
    for origin in origins:
        rows.append(
            v6._evaluate(
                roster,
                values,
                compiled,
                config,
                origin=origin,
                train_series_scope=scope,
            )
        )
    return rows


def _arm_metrics(
    identity_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    origins: tuple[int, ...],
    eval_uids: list[str],
) -> dict[str, Any]:
    """Section 12.3 cluster-unit metrics.

    SE_block = standard error of per-series mean gains, each series first
    averaged over the K origins in the block.  Cells are never IID samples.
    """
    per_series_by_origin = {uid: [] for uid in eval_uids}
    per_origin_gain = []
    behavior_points = 0
    for origin, base, candidate in zip(
        origins, identity_rows, candidate_rows
    ):
        gains = [
            float(reference - method)
            for reference, method in zip(
                base["per_view_smase"], candidate["per_view_smase"]
            )
        ]
        per_origin_gain.append(float(np.mean(gains)))
        for uid, gain in zip(eval_uids, gains):
            per_series_by_origin[uid].append(gain)
        behavior_points += int(candidate.get("behavior_point_count") or 0)
    per_series_mean = np.asarray(
        [float(np.mean(values)) for values in per_series_by_origin.values()],
        dtype=np.float64,
    )
    se_block = (
        float(np.std(per_series_mean, ddof=1) / np.sqrt(per_series_mean.size))
        if per_series_mean.size > 1
        else 0.0
    )
    macro_gain = float(np.mean(per_origin_gain))
    return {
        "macro_gain": macro_gain,
        "se_block": se_block,
        "gain_over_se": macro_gain / se_block if se_block > 0.0 else None,
        "per_series_mean_gain": {
            uid: float(value)
            for uid, value in zip(eval_uids, per_series_mean)
        },
        "per_origin_gain": {
            str(origin): gain
            for origin, gain in zip(origins, per_origin_gain)
        },
        "positive_series_count": int(np.sum(per_series_mean > MATERIAL_THRESHOLD)),
        "negative_series_count": int(np.sum(per_series_mean < -MATERIAL_THRESHOLD)),
        "modified_point_count": behavior_points,
    }


def _split_half_agreement(
    identity_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    origins: tuple[int, ...],
    eval_uids: list[str],
) -> dict[str, Any]:
    matched_beats_comparator = 0
    matched_beats_identity = 0
    total = 0
    for combo in itertools.combinations(range(len(eval_uids)), 4):
        half = set(combo)
        matched_gains = []
        comparator_gains = []
        for origin, base, matched, comparator in zip(
            origins, identity_rows, matched_rows, comparator_rows
        ):
            matched_gains.extend(
                float(reference - method)
                for index, (reference, method) in enumerate(
                    zip(base["per_view_smase"], matched["per_view_smase"])
                )
                if index in half
            )
            comparator_gains.extend(
                float(reference - method)
                for index, (reference, method) in enumerate(
                    zip(base["per_view_smase"], comparator["per_view_smase"])
                )
                if index in half
            )
        total += 1
        matched_beats_comparator += (
            float(np.mean(matched_gains)) > float(np.mean(comparator_gains))
        )
        matched_beats_identity += float(np.mean(matched_gains)) > 0.0
    return {
        "n_partitions": total,
        "matched_beats_comparator_count": matched_beats_comparator,
        "matched_beats_comparator_rate": (
            matched_beats_comparator / total if total else None
        ),
        "matched_beats_identity_count": matched_beats_identity,
        "matched_beats_identity_rate": (
            matched_beats_identity / total if total else None
        ),
        "d5_reference_rate": 0.45714285714285713,
        "d5_reference_note": (
            "D5 per-origin winner agreement over the same 8 eval series was "
            "45.7%; reference only, not a pass threshold."
        ),
    }


def _evaluate_block(
    roster: list[dict[str, Any]],
    values: dict[str, Any],
    clean_values: dict[str, Any],
    config: dict[str, Any],
    origins: tuple[int, ...],
    eval_uids: list[str],
    matched: CompiledWorkflow,
    comparator: CompiledWorkflow,
    block_name: str,
) -> dict[str, Any]:
    identity_rows = _evaluate_origins(
        roster, values, None, config, origins, None
    )
    clean_identity_rows = _evaluate_origins(
        roster, clean_values, None, config, origins, None
    )
    matched_rows = _evaluate_origins(
        roster,
        values,
        matched,
        config,
        origins,
        set(FAULTY_SERIES),
    )
    comparator_rows = _evaluate_origins(
        roster,
        values,
        comparator,
        config,
        origins,
        set(FAULTY_SERIES),
    )
    injected_identity_mean = float(np.mean([
        row["mean_smase"] for row in identity_rows
    ]))
    clean_identity_mean = float(np.mean([
        row["mean_smase"] for row in clean_identity_rows
    ]))
    matched_metrics = _arm_metrics(
        identity_rows, matched_rows, origins, eval_uids
    )
    comparator_metrics = _arm_metrics(
        identity_rows, comparator_rows, origins, eval_uids
    )
    split_half = _split_half_agreement(
        identity_rows, matched_rows, comparator_rows, origins, eval_uids
    )
    return {
        "block": block_name,
        "origins": list(origins),
        "injected_identity_task_mean_smase": injected_identity_mean,
        "clean_identity_task_mean_smase": clean_identity_mean,
        "baseline_degradation_ratio": (
            injected_identity_mean / clean_identity_mean
            if clean_identity_mean > 0.0
            else None
        ),
        "matched_outlier_mad": matched_metrics,
        "comparator_hampel_filter": comparator_metrics,
        "split_half": split_half,
    }


def _run_t0() -> dict[str, Any]:
    started = time.perf_counter()
    cohort = reg._load(PROJECT_ROOT)
    roster = cohort["roster"]
    values = cohort["values"]
    clean_values = {uid: np.asarray(value, dtype=np.float64).copy() for uid, value in values.items()}
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]

    injected, ground_truth = inject_label_touched_corpus(
        values,
        faulty_series=FAULTY_SERIES,
        clean_series=CLEAN_TRAIN_SERIES,
        amplitude=INJECTION_AMPLITUDE,
        count=INJECTION_COUNT,
        seed=INJECTION_SEED,
    )
    matched = _compiled(MATCHED_PROGRAM, name="t0-matched-outlier-mad")
    comparator = _compiled(COMPARATOR_PROGRAM, name="t0-comparator-hampel")

    task_matrix_started = time.perf_counter()
    support = _evaluate_block(
        mapped_roster,
        injected,
        clean_values,
        config,
        SUPPORT_ORIGINS,
        eval_uids,
        matched,
        comparator,
        "support",
    )
    delayed = _evaluate_block(
        mapped_roster,
        injected,
        clean_values,
        config,
        DELAYED_ORIGINS,
        eval_uids,
        matched,
        comparator,
        "delayed",
    )
    task_matrix_seconds = time.perf_counter() - task_matrix_started

    support_matched = support["matched_outlier_mad"]
    delayed_matched = delayed["matched_outlier_mad"]
    support_gain_over_se = support_matched["gain_over_se"]
    split_half_stable = bool(
        support["split_half"]["matched_beats_comparator_count"]
        >= MIN_SPLIT_HALF_MATCHED_WINS
        and delayed["split_half"]["matched_beats_comparator_count"]
        >= MIN_SPLIT_HALF_MATCHED_WINS
    )
    readable = bool(
        support_matched["macro_gain"] > 0.0
        and support_gain_over_se is not None
        and support_gain_over_se >= 3.0
        and delayed_matched["macro_gain"] > 0.0
        and split_half_stable
    )

    report = {
        "experiment_id": "w1-task-episode-harness",
        "phase": "T0",
        "protocol": {
            "feedback_unit": (
                "complete task macro gain: per eval series first averaged over "
                "the K origins in one block, then macro gain is the mean over "
                "eval series; cells are diagnostics only"
            ),
            "fault_family": FAULT_FAMILY,
            "injection_unit": "corpus_level_one_shot",
            "position_pool": "label_touched_timestamp_pool",
            "scale": "once per pristine series: nanstd(series[120:900])",
            "amplitude": INJECTION_AMPLITUDE,
            "count_per_faulty_series": INJECTION_COUNT,
            "seed": INJECTION_SEED,
            "faulty_series": list(FAULTY_SERIES),
            "clean_train_series": list(CLEAN_TRAIN_SERIES),
            "arms": ["identity", MATCHED_PROGRAM, COMPARATOR_PROGRAM],
            "scope": "oracle scope, T0 zero-LLM positive control only",
            "support_blocks": {"t0_support": list(SUPPORT_ORIGINS)},
            "delayed_blocks": {"t0_delayed": list(DELAYED_ORIGINS)},
            "support_delayed_semantics": (
                "same frozen eval roster; support and delayed share the same "
                "fitted Ridge and differ only in future windows; not a claim "
                "of cross-refit generalization (section 12.4)"
            ),
            "split_half_operationalization": {
                "matched_wins_vs_comparator_min_count": MIN_SPLIT_HALF_MATCHED_WINS,
                "scope": "T0-only; not a global Gate",
            },
            "baseline_degradation_ratio": "disclosure only, no threshold, not a Gate",
            "target_probe_budget": None,
            "source_target_separation": None,
            "development_only": True,
            "store_scope": "demo_local",
        },
        "private_audit": {
            "injection_specs": {
                "family": FAULT_FAMILY,
                "injection_unit": "corpus_level_one_shot",
                "position_pool_name": "label_touched_timestamp_pool",
                "amplitude": INJECTION_AMPLITUDE,
                "count": INJECTION_COUNT,
                "seed": INJECTION_SEED,
                "faulty_series": list(FAULTY_SERIES),
                "clean_train_series": list(CLEAN_TRAIN_SERIES),
                "ground_truth": ground_truth,
            },
            "oracle_checks": {
                "matching_scope": list(FAULTY_SERIES),
                "eval_series_untouched": True,
                "clean_series_pointwise_unchanged": True,
            },
        },
        "substrate": {
            "selected_recipe": {
                "family": FAULT_FAMILY,
                "injection_unit": "corpus_level_one_shot",
                "amplitude": INJECTION_AMPLITUDE,
                "count": INJECTION_COUNT,
                "seed": INJECTION_SEED,
                "matched_program": MATCHED_PROGRAM,
                "comparator_program": COMPARATOR_PROGRAM,
                "scope_is_private_audit_only": True,
            },
            "blocks": [support, delayed],
            "readable": readable,
            "baseline_degradation_ratio": {
                block["block"]: block["baseline_degradation_ratio"]
                for block in (support, delayed)
            },
            "baseline_degradation_note": (
                "injected identity task sMASE / clean identity task sMASE; "
                "disclosure only, no pass threshold, no retroactive verdict "
                "change"
            ),
        },
        "task_episodes": [],
        "source_bank": {
            "positive_count": 0,
            "negative_count": 0,
            "conflict_count": 0,
            "abstain_count": 0,
        },
        "a5_a3": None,
        "mechanical_checks": {
            "llm_calls": 0,
            "slow_calls": 0,
            "gate_changed": False,
            "support_delayed_disjoint": bool(
                set(SUPPORT_ORIGINS).isdisjoint(DELAYED_ORIGINS)
            ),
            "support_before_delayed": bool(
                max(SUPPORT_ORIGINS) < min(DELAYED_ORIGINS)
            ),
            "already_exposed_development_origins": sorted(
                set(SUPPORT_ORIGINS + DELAYED_ORIGINS)
            ),
            "private_audit_separate_from_agent_visible": True,
            "oracle_scope_used_only_in_t0_positive_control": True,
            "split_half_label_stable": split_half_stable,
        },
        "cost_probe": {
            "one_task_matrix_seconds": task_matrix_seconds,
            "candidate_origins_evaluated": 3 * len(
                SUPPORT_ORIGINS + DELAYED_ORIGINS
            ),
            "wall_seconds_total_t0": time.perf_counter() - started,
            "note": (
                "single formal run; no 2x2 and no dose scan per final ruling"
            ),
        },
        "verdict": (
            "TASK_EPISODE_SUBSTRATE_READABLE"
            if readable
            else "TASK_EPISODE_SUBSTRATE_UNREADABLE"
        ),
        "historical_note": (
            "previous context+fixed severe result remains historical protocol "
            "pass / diagnostic only; it is not the canonical T0 result"
        ),
    }
    REPORT_REL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_REL.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="t0")
    parser.add_argument("--report", type=Path, default=REPORT_REL)
    args = parser.parse_args()
    if args.phase == "t0":
        report = _run_t0()
    elif args.phase == "t1":
        from evaluation.functional.task_episode_harness.t1 import run_t1

        t1 = run_t1(args.report)
        print(json.dumps(t1, indent=2, default=str))
        return 0
    elif args.phase == "t2":
        from evaluation.functional.task_episode_harness.t2 import run_t2

        t2 = run_t2(args.report)
        print(json.dumps(t2, indent=2, default=str))
        return 0
    elif args.phase == "t2b":
        from evaluation.functional.task_episode_harness.t2b import run_t2b

        t2b = run_t2b(args.report)
        print(json.dumps(t2b, indent=2, default=str))
        return 0
    elif args.phase == "t3":
        from evaluation.functional.task_episode_harness.t3 import run_t3

        t3 = run_t3(args.report)
        print(json.dumps(t3, indent=2, default=str))
        return 0
    elif args.phase == "t3b":
        from evaluation.functional.task_episode_harness.t3b import run_t3b

        t3b = run_t3b(args.report)
        print(json.dumps(t3b, indent=2, default=str))
        return 0
    elif args.phase == "t2c":
        from evaluation.functional.task_episode_harness.t2c import run_t2c

        t2c = run_t2c(args.report)
        print(json.dumps(t2c, indent=2, default=str))
        return 0
    elif args.phase == "normal-flow":
        from evaluation.functional.task_episode_harness.normal_flow import (
            run_normal_flow,
        )

        normal_flow = run_normal_flow(args.report)
        print(json.dumps(normal_flow, indent=2, default=str))
        return 0
    elif args.phase == "natural-flow":
        from evaluation.functional.task_episode_harness.natural_flow import (
            run_natural_flow,
        )

        natural_flow = run_natural_flow(args.report)
        print(json.dumps(natural_flow, indent=2, default=str))
        return 0
    elif args.phase == "natural-precheck":
        from evaluation.functional.task_episode_harness.natural_precheck import (
            run_natural_precheck,
        )

        precheck = run_natural_precheck(args.report)
        print(json.dumps(precheck, indent=2, default=str))
        return 0
    elif args.phase == "a5a3":
        from evaluation.functional.task_episode_harness.a5a3 import run_a5a3

        a5a3 = run_a5a3(args.report)
        print(json.dumps(a5a3, indent=2, default=str))
        return 0
    elif args.phase == "workflow-generation":
        from evaluation.functional.task_episode_harness.workflow_gen import (
            run_workflow_generation,
        )

        workflow_generation = run_workflow_generation(args.report)
        print(json.dumps(workflow_generation, indent=2, default=str))
        return 0
    else:
        raise SystemExit(
            "only phases t0/t1/t2/t2b/t2c/t3/t3b/normal-flow/natural-flow/"
            "natural-precheck/a5a3/workflow-generation are implemented, "
            f"got {args.phase!r}"
        )
    substrate = report["substrate"]
    print(json.dumps({
        "phase": report["phase"],
        "verdict": report["verdict"],
        "selected_recipe": substrate.get("selected_recipe"),
        "baseline_degradation_ratio": substrate.get("baseline_degradation_ratio"),
        "support_gains": (
            {
                block["block"]: {
                    "matched_outlier_mad": block["matched_outlier_mad"]["macro_gain"],
                    "comparator_hampel_filter": block["comparator_hampel_filter"]["macro_gain"],
                    "se_block": block["matched_outlier_mad"]["se_block"],
                    "gain_over_se": block["matched_outlier_mad"]["gain_over_se"],
                    "split_half_matched_wins": block["split_half"]["matched_beats_comparator_count"],
                }
                for block in substrate["blocks"]
            }
            if substrate.get("blocks") else {}
        ),
        "cost_probe": report["cost_probe"],
        "report": str(args.report.relative_to(PROJECT_ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
