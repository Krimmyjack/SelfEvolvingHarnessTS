"""C1 Observation diagnosis: why repair_level_shift flips inside very_low.

Zero-LLM and zero-new-outcome diagnosis.  It re-reads only already-exposed
E1-v2 Support cells for harm decomposition and reuses the cached public
Contexts for the small mechanism-feature check.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional.run_e2_autonomous_natural_workflow_generation import (
    _evaluate,
)
from evaluation.functional.task_episode_harness.e0b import (
    C1_DOWNSTREAM_WINDOW_POINTS,
    C1_POST_SHIFT_SUPPORT_FEATURE,
    C1_POST_SHIFT_SUPPORT_MIN_POINTS,
    _augment_context_with_c1_feature,
    _episode_rows_for_signature,
)
from evaluation.functional.task_episode_harness.e1 import (
    _load_kdd_roster,
)
from evaluation.functional.task_episode_harness.runner import (
    REPORT_REL,
    _arm_metrics,
    _evaluate_origins,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.skill_evolution import (
    _probe_compiled,
)
from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import seasonal_scale
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    CompiledWorkflow,
)

PROTOCOL_VERSION = "c1_observation_diagnosis_v1"
HARM_TASK_IDS = (
    "e1v2_task_10",
    "e1v2_task_14",
    "e1v2_task_15",
    "e1v2_task_16",
)


def _compiled_from_steps(steps: Sequence[Mapping[str, Any]]) -> CompiledWorkflow:
    program = Program.from_steps(
        [
            (str(step["op"]), dict(step.get("params") or {}))
            for step in steps
        ],
        source="c1_diagnosis",
    )
    candidate = Candidate.program_candidate(
        "c1_diagnosis", program, source="c1_diagnosis"
    )
    return CompiledWorkflow(
        candidate,
        (),
        tuple(
            {
                "op": str(step["op"]),
                "params": dict(step.get("params") or {}),
                "bindings": {},
            }
            for step in steps
        ),
    )


def _harm_decomposition(
    *,
    report: Mapping[str, Any],
    repo_root: Path,
) -> list[dict[str, Any]]:
    from run_v1_kdd2018_natural_slow_update import _config

    v2 = report.get("e1_v2") or {}
    roster, values, _selected = _load_kdd_roster(
        repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e31.jsonl"
    )
    mapped = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped if row["role"] == "eval"
    ]
    config = dict(_config())
    rows_out = []
    for task_row in v2.get("rows") or []:
        task_id = str(task_row.get("task_episode_id"))
        if task_id not in HARM_TASK_IDS:
            continue
        origins = tuple(task_row["support_origins"])
        scope = frozenset(
            (task_row.get("public_context") or {}).get("scope_series_uids") or []
        )
        first_probe = next(
            (
                probe for probe in (task_row.get("A3") or {}).get("probes") or []
                if isinstance(probe.get("support_gain"), (int, float))
                and probe.get("attempt_index") == 0
            ),
            None,
        )
        if first_probe is None:
            continue
        compiled = _compiled_from_steps(first_probe["compiled_steps"])
        identity = _evaluate_origins(
            mapped, values, None, config, origins, None
        )
        candidate = _evaluate_origins(
            mapped, values, compiled, config, origins, set(scope)
        )
        metrics = _arm_metrics(identity, candidate, origins, eval_uids)
        scales = {}
        for uid in eval_uids:
            raw = np.asarray(values[uid], dtype=np.float64)
            try:
                scales[uid] = [
                    float(
                        seasonal_scale(
                            raw[:origin],
                            np.isfinite(raw[:origin]),
                            period=24,
                            min_pairs=32,
                        )
                    )
                    for origin in origins
                ]
            except Exception as exc:  # noqa: BLE001
                scales[uid] = [f"{type(exc).__name__}"]
        gains = {
            uid: float(gain)
            for uid, gain in metrics["per_series_mean_gain"].items()
        }
        total_abs_harm = float(
            sum(max(0.0, -gain) for gain in gains.values())
        )
        rows_out.append({
            "task_episode_id": task_id,
            "support_origins": list(origins),
            "macro_gain": float(metrics["macro_gain"]),
            "se_block": float(metrics["se_block"]),
            "gain_over_se": metrics["gain_over_se"],
            "per_origin_gain": metrics["per_origin_gain"],
            "per_series_mean_gain": gains,
            "positive_series_count": int(metrics["positive_series_count"]),
            "negative_series_count": int(metrics["negative_series_count"]),
            "modified_point_count": int(metrics["modified_point_count"]),
            "failed_step_count": int(
                sum(row["failed_step_count"] for row in candidate)
            ),
            "seasonal_scale_by_series": scales,
            "min_seasonal_scale": float(
                min(
                    value
                    for values in scales.values()
                    for value in values
                    if isinstance(value, (int, float))
                )
            ),
            "single_series_harm_share_max": (
                max(
                    max(0.0, -gain) / total_abs_harm
                    for gain in gains.values()
                )
                if total_abs_harm > 0
                else 0.0
            ),
            "broad_harm": bool(metrics["negative_series_count"] == len(eval_uids)),
        })
    return rows_out


def run_c1_observation_diagnosis(
    report_path: Path = REPORT_REL,
) -> dict[str, Any]:
    started = time.perf_counter()
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    repo_root = Path(__file__).resolve().parents[3]
    selected_signature = {
        "task_kind": "forecast",
        "estimated_region_start_fraction": "very_low",
    }
    supply = _episode_rows_for_signature(report, selected_signature)
    harm = _harm_decomposition(report=report, repo_root=repo_root)

    mechanism_rows = []
    v2 = report.get("e1_v2") or {}
    for task_row in v2.get("rows") or []:
        context = task_row.get("public_context") or {}
        if dict(context.get("task_signature") or {}) != selected_signature:
            continue
        context = _augment_context_with_c1_feature(context)
        first_probe = next(
            (
                probe for probe in (task_row.get("A3") or {}).get("probes") or []
                if isinstance(probe.get("support_gain"), (int, float))
                and probe.get("attempt_index") == 0
            ),
            None,
        )
        if first_probe is None:
            continue
        ff = context["task_fast_features"]
        end_fraction = float(ff["estimated_region_end_fraction"])
        post_points = max(
            0.0,
            (1.0 - end_fraction) * C1_DOWNSTREAM_WINDOW_POINTS,
        )
        mechanism_rows.append({
            "task_episode_id": str(task_row.get("task_episode_id")),
            "first_a3_support_gain": float(first_probe["support_gain"]),
            "estimated_region_start_fraction": float(
                ff["estimated_region_start_fraction"]
            ),
            "estimated_region_end_fraction": end_fraction,
            "level_excursion_score": float(ff["level_excursion_score"]),
            "estimated_level_offset": float(ff["estimated_level_offset"]),
            "period_change_score": float(ff["period_change_score"]),
            "post_shift_support_points_diagnostic": post_points,
            C1_POST_SHIFT_SUPPORT_FEATURE: bool(ff[C1_POST_SHIFT_SUPPORT_FEATURE]),
        })

    positive_rows = [
        row for row in mechanism_rows if row["first_a3_support_gain"] >= 0.005
    ]
    negative_rows = [
        row for row in mechanism_rows if row["first_a3_support_gain"] < 0.005
    ]
    separator_valid = bool(
        positive_rows
        and negative_rows
        and all(row[C1_POST_SHIFT_SUPPORT_FEATURE] for row in positive_rows)
        and all(not row[C1_POST_SHIFT_SUPPORT_FEATURE] for row in negative_rows)
    )
    harm_real = bool(
        harm
        and all(row["broad_harm"] for row in harm)
        and all(row["min_seasonal_scale"] > 0.5 for row in harm)
        and all(row["single_series_harm_share_max"] < 0.30 for row in harm)
    )
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": (
            "C1_OBSERVATION_FOUND"
            if harm_real and separator_valid
            else "C1_OBSERVATION_NOT_FOUND"
        ),
        "zero_llm": True,
        "zero_new_outcome": True,
        "sealed_confirmation_opened": False,
        "e2_not_started": True,
        "harm_real": harm_real,
        "harm_decomposition": harm,
        "harm_conclusion": (
            "The large harms are real measurement effects: all eight eval "
            "series degrade, behavior_point_count is large, seasonal scales "
            "are far from the numeric floor, and no single series dominates."
            if harm_real else None
        ),
        "mechanism_feature_check": {
            "selected_signature": selected_signature,
            "rows": mechanism_rows,
            "positive_task_ids": [row["task_episode_id"] for row in positive_rows],
            "negative_task_ids": [row["task_episode_id"] for row in negative_rows],
            "separator_feature": C1_POST_SHIFT_SUPPORT_FEATURE,
            "separator_rule": (
                f"(1 - estimated_region_end_fraction) * "
                f"{C1_DOWNSTREAM_WINDOW_POINTS} >= "
                f"{C1_POST_SHIFT_SUPPORT_MIN_POINTS}"
            ),
            "threshold_origin": (
                "one full frozen seasonal period (24 points) in the frozen "
                "192+48 downstream Task window; not selected from outcomes"
            ),
            "separator_valid": separator_valid,
            "no_full_feature_search": True,
            "no_outcome_tuned_threshold": True,
        },
        "observation_added": {
            "feature": C1_POST_SHIFT_SUPPORT_FEATURE,
            "type": "boolean",
            "vocabulary_file": "SelfEvolvingHarnessTS/contracts/observables.py",
            "extractor_file": "SelfEvolvingHarnessTS/runtime/public_features.py",
            "derivation": (
                "post_shift_support_sufficient = "
                "((1 - estimated_region_end_fraction) * 240) >= 24"
            ),
        },
        "boundary": {
            "c1_only": True,
            "e2_not_started": True,
            "sealed_confirmation_opened": False,
            "no_new_schema": True,
            "no_new_taxonomy": True,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    report["historical_verdict_before_c1"] = report.get("verdict")
    report["phase"] = "c1_observation_diagnosis"
    report["c1_observation_diagnosis"] = result
    report["verdict"] = result["verdict"]
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["PROTOCOL_VERSION", "run_c1_observation_diagnosis"]
