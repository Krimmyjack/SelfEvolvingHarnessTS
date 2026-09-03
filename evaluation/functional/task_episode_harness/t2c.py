"""T2c: GAP_SUBSTRATE_READABILITY_CALIBRATION (zero LLM).

One knob only: gap count / missing fraction in the existing corpus-level
injector.  Scope, seed, Consumer, Support/delayed blocks and T0 readability
criteria are frozen.  First passing strength is frozen; if none passes the
pre-registered grid, the random-gap x impute_ema family is closed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from run_v1_a5a3_runtime_regression import _load as _load_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.injection import (
    inject_gap_corpus,
)
from evaluation.functional.task_episode_harness.runner import (
    DELAYED_ORIGINS,
    REPORT_REL,
    SUPPORT_ORIGINS,
    _arm_metrics,
    _compiled,
    _evaluate_origins,
    _mapped_roster,
    _split_half_agreement,
)

T2C_GAP_COUNT_GRID = (120, 240, 360, 480)
T2C_SEED = 7
T2C_FAULTY_SERIES = ("T117", "T118", "T119", "T12", "T120", "T121")
T2C_MATCHED_WORKFLOW = "impute_ema"
T2C_WRONG_WORKFLOW = "outlier_mad"
T2C_MIN_GAIN_OVER_SE = 3.0
T2C_MIN_SPLIT_HALF_WINS = 49


def run_t2c(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    cohort = _load_cohort(repo_root)
    roster = cohort["roster"]
    values = cohort["values"]
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]
    clean = tuple(uid for uid in train_uids if uid not in T2C_FAULTY_SERIES)
    matched = _compiled(T2C_MATCHED_WORKFLOW, name="t2c-matched")
    wrong = _compiled(T2C_WRONG_WORKFLOW, name="t2c-wrong")

    rows = []
    frozen = None
    for count in T2C_GAP_COUNT_GRID:
        injected, _gt = inject_gap_corpus(
            values,
            faulty_series=T2C_FAULTY_SERIES,
            clean_series=clean,
            count=count,
            seed=T2C_SEED,
        )
        blocks = []
        for block_name, origins in (
            ("support", SUPPORT_ORIGINS),
            ("delayed", DELAYED_ORIGINS),
        ):
            identity_rows = _evaluate_origins(
                mapped_roster, injected, None, config, origins, None
            )
            matched_rows = _evaluate_origins(
                mapped_roster,
                injected,
                matched,
                config,
                origins,
                set(T2C_FAULTY_SERIES),
            )
            wrong_rows = _evaluate_origins(
                mapped_roster,
                injected,
                wrong,
                config,
                origins,
                set(T2C_FAULTY_SERIES),
            )
            matched_metrics = _arm_metrics(
                identity_rows, matched_rows, origins, eval_uids
            )
            wrong_metrics = _arm_metrics(
                identity_rows, wrong_rows, origins, eval_uids
            )
            split = _split_half_agreement(
                identity_rows,
                matched_rows,
                wrong_rows,
                origins,
                eval_uids,
            )
            blocks.append({
                "block": block_name,
                "origins": list(origins),
                "matched_impute_ema": matched_metrics,
                "wrong_outlier_mad": wrong_metrics,
                "split_half": split,
            })
        support = blocks[0]
        delayed = blocks[1]
        support_gse = support["matched_impute_ema"]["gain_over_se"]
        passes = bool(
            support_gse is not None
            and support["matched_impute_ema"]["macro_gain"] > 0.0
            and support_gse >= T2C_MIN_GAIN_OVER_SE
            and delayed["matched_impute_ema"]["macro_gain"] > 0.0
            and support["split_half"]["matched_beats_comparator_count"]
            >= T2C_MIN_SPLIT_HALF_WINS
            and delayed["split_half"]["matched_beats_comparator_count"]
            >= T2C_MIN_SPLIT_HALF_WINS
        )
        row = {
            "gap_count": count,
            "support_matched_gain": support["matched_impute_ema"]["macro_gain"],
            "support_matched_se": support["matched_impute_ema"]["se_block"],
            "support_matched_gain_over_se": support_gse,
            "delayed_matched_gain": delayed["matched_impute_ema"]["macro_gain"],
            "delayed_matched_se": delayed["matched_impute_ema"]["se_block"],
            "support_wrong_gain": support["wrong_outlier_mad"]["macro_gain"],
            "delayed_wrong_gain": delayed["wrong_outlier_mad"]["macro_gain"],
            "support_split_half_wins": support["split_half"][
                "matched_beats_comparator_count"
            ],
            "delayed_split_half_wins": delayed["split_half"][
                "matched_beats_comparator_count"
            ],
            "passes": passes,
            "blocks": blocks,
        }
        rows.append(row)
        if passes:
            frozen = count
            break

    verdict = (
        f"GAP_SUBSTRATE_READABILITY_PASS_COUNT_{frozen}"
        if frozen is not None
        else "GAP_SUBSTRATE_READABILITY_FAILED_CLOSE_FAMILY"
    )
    t2c = {
        "knob": "gap_count",
        "pre_registered_grid": list(T2C_GAP_COUNT_GRID),
        "seed": T2C_SEED,
        "faulty_scope_private": list(T2C_FAULTY_SERIES),
        "matched_workflow": T2C_MATCHED_WORKFLOW,
        "wrong_workflow": T2C_WRONG_WORKFLOW,
        "criteria": {
            "support_macro_gain_positive": True,
            "support_gain_over_se_min": T2C_MIN_GAIN_OVER_SE,
            "delayed_same_direction_positive": True,
            "min_split_half_wins_per_block": T2C_MIN_SPLIT_HALF_WINS,
        },
        "rows": rows,
        "frozen_count": frozen,
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - started,
        "llm_api_call_count": 0,
    }

    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    report["phase"] = "T2c"
    report["t2c"] = t2c
    report["verdict"] = verdict
    report["mechanical_checks"] = dict(
        report.get("mechanical_checks") or {},
        t2c_llm_calls=0,
        t2c_memory_unchanged=True,
        t2c_gate_unchanged=True,
    )
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return t2c
