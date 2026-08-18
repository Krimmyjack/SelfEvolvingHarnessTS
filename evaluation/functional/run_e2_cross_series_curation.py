"""Measure exposed natural donor-series curation headroom under fixed Ridge.

P0 changes only the Program action unit: one candidate binds all six training
windows from one donor series and all 48 forecast outputs.  The fixed menu is
IDENTITY (weight 1), ATTENUATE_DONOR (weight 0.25), and EXCLUDE_DONOR
(weight 0).  Exact grouped Ridge downdates are selected on one alternating
evaluation cohort and scored on the other.

This is a development-only, outcome-exposed Program-headroom census.  It does
not read UCI Target data and is not Capability, transfer, Memory, or fresh
evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import zipfile
from itertools import permutations
from pathlib import Path
from typing import Any, Mapping, Sequence

from SelfEvolvingHarnessTS.evaluation.functional.run_e2_action_conditioned_valuation_proxy import (
    RIDGE_ALPHA,
    _group_removal_predictions,
    _ridge_reference_and_removal_predictions,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
    _alternating_folds,
    _mean,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
    ANCHORS,
    CONTEXT_LENGTH,
    FRESH_SPECS,
    HORIZON,
    J0_PLAN_PATH,
    J0_REPORT_PATH,
    SPECS,
    _read_object,
)
from SelfEvolvingHarnessTS.methods.ttha.skill_acquisition import (
    apply_typed_patch,
    attach_delayed_outcomes,
    build_candidate_skill,
    build_policy_failure_dossier,
    collect_source_policy_episodes,
    execute_skill_card,
    plan_skill_card_support_only,
    policy_adaptation_auc as policy_episode_adapt_auc,
    read_active_skill_cards,
    run_failure_driven_update_cycle,
    run_skill_acquisition_cycle,
    validate_failure_driven_patch,
)
from SelfEvolvingHarnessTS.methods.ttha.workflow_discovery import (
    discover_workflow_supply,
)


SCHEMA_VERSION = "e2-cross-series-curation-p0/1"
DEFAULT_REPORT_PATH = "artifacts/functional/e2/cross_series_curation_p0_report.json"
P1_SCHEMA_VERSION = "e2-cross-series-curation-p1/1"
P1_REPORT_PATH = "artifacts/functional/e2/cross_series_curation_p1_report.json"
P1B_SCHEMA_VERSION = "e2-cross-series-curation-p1b/1"
P1B_REPORT_PATH = "artifacts/functional/e2/cross_series_curation_p1b_report.json"
WORKFLOW_REPLAY_REPORT_PATH = "artifacts/functional/e2/cross_series_workflow_harness_replay_report.json"
SEMANTIC_AUXILIARY_P0_REPORT_PATH = (
    "artifacts/functional/e2/semantic_auxiliary_group_augmentation_p0_report.json"
)
SEMANTIC_AUXILIARY_P1_REPORT_PATH = (
    "artifacts/functional/e2/semantic_auxiliary_group_augmentation_p1_report.json"
)
AUXILIARY_CHANNEL_BINDING_P0_REPORT_PATH = (
    "artifacts/functional/e2/auxiliary_channel_binding_p0_report.json"
)
AUXILIARY_CHANNEL_BINDING_LLM_PLAN_PATH = (
    "artifacts/functional/e2/auxiliary_channel_binding_llm_plan.json"
)
AUXILIARY_CHANNEL_BINDING_LLM_REPORT_PATH = (
    "artifacts/functional/e2/auxiliary_channel_binding_llm_pilot_report.json"
)
AUXILIARY_CHANNEL_BINDING_HISTORY_P1_REPORT_PATH = (
    "artifacts/functional/e2/auxiliary_channel_binding_history_p1_report.json"
)
HISTORICAL_POLICY_CAPABILITY_PATH = (
    "artifacts/functional/e2/historical_policy_episode_workflow_capability.json"
)
CONTROLLED_CLASSIFICATION_CAPABILITY_PATH = (
    "artifacts/functional/e2/controlled_classification_dynamic_binding_capability.json"
)
MISSING_WINDOW_WEIGHTING_CAPABILITY_PATH = (
    "artifacts/functional/e2/missing_window_weighting_workflow_capability.json"
)
MULTISKILL_LLM_FAST_PATH_PLAN_PATH = (
    "artifacts/functional/e2/multiskill_llm_fast_path_plan.json"
)
MULTISKILL_LLM_FAST_PATH_REPORT_PATH = (
    "artifacts/functional/e2/multiskill_llm_fast_path_report.json"
)
MULTISKILL_LIVE_LLM_FAST_PATH_PLAN_PATH = (
    "artifacts/functional/e2/multiskill_live_llm_fast_path_plan.json"
)
MULTISKILL_LIVE_LLM_FAST_PATH_REPORT_PATH = (
    "artifacts/functional/e2/multiskill_live_llm_fast_path_report.json"
)
FORECASTING_TWO_SKILL_LIVE_PLAN_PATH = (
    "artifacts/functional/e2/forecasting_two_skill_live_llm_plan.json"
)
FORECASTING_TWO_SKILL_LIVE_REPORT_PATH = (
    "artifacts/functional/e2/forecasting_two_skill_live_llm_report.json"
)
FORECASTING_TWO_SKILL_COMPILED_REPORT_PATH = (
    "artifacts/functional/e2/forecasting_two_skill_compiled_runtime_report.json"
)
HISTORICAL_POLICY_LLM_SLOW_PATH_PLAN_PATH = (
    "artifacts/functional/e2/historical_policy_llm_slow_path_plan.json"
)
HISTORICAL_POLICY_LLM_SLOW_PATH_REPORT_PATH = (
    "artifacts/functional/e2/historical_policy_llm_slow_path_report.json"
)
SKILL_ACQUISITION_FRAMEWORK_REPLAY_REPORT_PATH = (
    "artifacts/functional/e2/skill_acquisition_framework_replay_report.json"
)
WORKFLOW_DISCOVERY_ACQUISITION_REPORT_PATH = (
    "artifacts/functional/e2/workflow_discovery_acquisition_framework_report.json"
)
FAILURE_DRIVEN_SKILL_EVOLUTION_REPORT_PATH = (
    "artifacts/functional/e2/failure_driven_skill_evolution_framework_report.json"
)
SECOND_FAILURE_MECHANISM_REPORT_PATH = (
    "artifacts/functional/e2/second_failure_mechanism_core_cycle_report.json"
)
REJECTION_AWARE_FAST_PATH_REPORT_PATH = (
    "artifacts/functional/e2/rejection_aware_fast_path_report.json"
)
NATURAL_DELAYED_FEEDBACK_VERTICAL_REPORT_PATH = (
    "artifacts/functional/e2/natural_delayed_feedback_vertical_report.json"
)
NATURAL_IMPUTATION_COLD_START_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_cold_start_contract_repair_report.json"
)
NATURAL_IMPUTATION_TARGET_PILOT_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_target_pilot_report.json"
)
NATURAL_IMPUTATION_AIR_QUALITY_TARGET_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_air_quality_target_report.json"
)
NATURAL_IMPUTATION_SCOPE_REPAIR_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_scope_repair_report.json"
)
NATURAL_IMPUTATION_PRSA_TARGET_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_prsa_target_report.json"
)
NATURAL_IMPUTATION_PRSA_ACTIONABLE_TARGET_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_prsa_actionable_target_report.json"
)
NATURAL_IMPUTATION_PSEUDO_GAP_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_pseudo_gap_observation_report.json"
)
NATURAL_IMPUTATION_PSEUDO_GAP_HELDOUT_ROLE_REPORT_PATH = (
    "artifacts/functional/e2/natural_imputation_pseudo_gap_development_replay_report.json"
)
REVERSIBLE_TARGET_REPRESENTATION_P0_REPORT_PATH = (
    "artifacts/functional/e2/natural_reversible_target_representation_p0_report.json"
)
REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPORT_PATH = (
    "artifacts/functional/e2/natural_reversible_target_representation_extension_report.json"
)
REVERSIBLE_TARGET_REPRESENTATION_LLM_PLAN_PATH = (
    "artifacts/functional/e2/reversible_target_representation_llm_patch_plan.json"
)
REVERSIBLE_TARGET_REPRESENTATION_LLM_REPORT_PATH = (
    "artifacts/functional/e2/reversible_target_representation_llm_patch_report.json"
)
REVERSIBLE_TARGET_REPRESENTATION_LLM_REPLAY_REPORT_PATH = (
    "artifacts/functional/e2/reversible_target_representation_llm_patch_replay_report.json"
)
REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPLAY_REPORT_PATH = (
    "artifacts/functional/e2/reversible_target_representation_extension_replay_report.json"
)
REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_PLAN_PATH = (
    "artifacts/functional/e2/reversible_target_representation_llm_revision_plan.json"
)
REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPORT_PATH = (
    "artifacts/functional/e2/reversible_target_representation_llm_revision_report.json"
)
REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPLAY_REPORT_PATH = (
    "artifacts/functional/e2/reversible_target_representation_llm_revision_replay_report.json"
)
NOAA_MULTICHANNEL_REPAIR_P0_REPORT_PATH = (
    "artifacts/functional/e2/noaa_multichannel_local_repair_p0_report.json"
)
NOAA_MULTICHANNEL_REPAIR_2025_REPORT_PATH = (
    "artifacts/functional/e2/noaa_multichannel_local_repair_2025_report.json"
)
NOAA_DEWPOINT_FEASIBILITY_STATIONS = (
    "72493723289",
    "72327199999",
    "72562624091",
    "70232526443",
    "72566024028",
    "72272093026",
    "72411013741",
    "72650014972",
    "72672024061",
)
MISSING_WINDOW_WEIGHTING_P0_REPORT_PATH = (
    "artifacts/functional/e2/natural_missing_window_weighting_p0_report.json"
)
MISSING_WINDOW_WEIGHTING_P1_REPORT_PATH = (
    "artifacts/functional/e2/natural_missing_window_weighting_p1_report.json"
)
MISSING_WINDOW_WEIGHTING_P2_REPORT_PATH = (
    "artifacts/functional/e2/natural_missing_window_weighting_p2_report.json"
)
MISSING_WINDOW_WEIGHTING_PRSA_REPORT_PATH = (
    "artifacts/functional/e2/natural_missing_window_weighting_prsa_target_report.json"
)
MISSING_WINDOW_WEIGHTING_PRSA_RISK_REPORT_PATH = (
    "artifacts/functional/e2/natural_missing_window_weighting_prsa_risk_report.json"
)
MISSING_WINDOW_WEIGHTING_ORIGIN_COVERAGE_REPORT_PATH = (
    "artifacts/functional/e2/natural_missing_window_weighting_origin_coverage_report.json"
)
MISSING_WINDOW_WEIGHTING_AIR_QUALITY_REPORT_PATH = (
    "artifacts/functional/e2/natural_missing_window_weighting_air_quality_report.json"
)
SEMANTIC_AUXILIARY_WEATHER_PLAN_PATH = (
    "artifacts/functional/e2/semantic_auxiliary_weather_llm_plan.json"
)
SEMANTIC_AUXILIARY_WEATHER_REPORT_PATH = (
    "artifacts/functional/e2/semantic_auxiliary_weather_llm_pilot_report.json"
)
ROWBLOCK_REPORT_PATH = (
    "artifacts/functional/e2/source_natural_block_action_value_headroom_report.json"
)
EXPECTED_TRAINING_SERIES = 12
EXPECTED_TRAINING_ROWS_PER_DONOR = 6
EXPECTED_TRAINING_ROWS = 72
EXPECTED_EVALUATION_ROWS = 8
PROGRAMS = (
    ("ATTENUATE_DONOR", 0.25),
    ("EXCLUDE_DONOR", 0.0),
)
REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_SPECS = {
    "monash:traffic_hourly": {
        "train_stop": 928,
        "future_bounds": (928, 976),
        "period": 24,
        "roster_start": 20,
    },
    "metr_la": {
        "train_stop": 928,
        "future_bounds": (928, 976),
        "period": 24,
        "roster_start": 20,
    },
}
TEMPORAL_ORIGIN_PROGRAMS = (
    ("ATTENUATE_ORIGIN_GROUP", 0.25),
    ("EXCLUDE_ORIGIN_GROUP", 0.0),
)
FOLD_PAIRS = (
    ("a_to_b", "fold_a", "fold_b"),
    ("b_to_a", "fold_b", "fold_a"),
)
P1_BUDGETS = (0, 1, 2)
P1_FEATURE_NAMES = (
    "support_first_order_proxy_gain",
    "response_mismatch",
    "is_exclude",
    "proxy_x_mismatch",
    "mismatch_x_is_exclude",
)
P1B_WORKFLOW_IDS = ("W_rowblock", "W_curation")
P1B_ORDERS = (
    ("W_rowblock", "W_curation"),
    ("W_curation", "W_rowblock"),
)

SEMANTIC_AUXILIARY_WEIGHTS = (0.25, 1.0)
SEMANTIC_AUXILIARY_DATASETS = (
    {
        "dataset_id": "kdd_cup_2018",
        "archive": (
            r"\\wsl.localhost\Ubuntu\tmp\kdd_cup_2018_dataset_without_missing_values.zip"
            if os.name == "nt"
            else "/tmp/kdd_cup_2018_dataset_without_missing_values.zip"
        ),
        "member": "kdd_cup_2018_dataset_without_missing_values.tsf",
        "identity_fields": ("city", "station"),
        "identity_filter": {"city": "Beijing"},
        "semantic_field": "air_quality_measurement",
        "target_roles": ("PM2.5", "O3"),
        "semantic_roles": ("PM2.5", "PM10", "NO2", "CO", "O3", "SO2"),
        "anchors": ANCHORS,
        "train_stop": 928,
        "future": (928, 976),
        "period": 24,
    },
    {
        "dataset_id": "rideshare",
        "archive": (
            r"\\wsl.localhost\Ubuntu\tmp\rideshare_dataset_without_missing_values.zip"
            if os.name == "nt"
            else "/tmp/rideshare_dataset_without_missing_values.zip"
        ),
        "member": "rideshare_dataset_without_missing_values.tsf",
        "identity_fields": ("source_location", "provider_name", "provider_service"),
        "identity_filter": {},
        "semantic_field": "type",
        "target_roles": ("price_mean", "distance_mean"),
        "semantic_roles": (
            "price_min",
            "price_mean",
            "price_max",
            "distance_min",
            "distance_mean",
            "distance_max",
            "surge_min",
            "surge_mean",
            "surge_max",
            "api_calls",
            "temp",
            "rain",
            "humidity",
            "clouds",
            "wind",
        ),
        "anchors": (240, 288, 336, 384, 432),
        "train_stop": 493,
        "future": (493, 541),
        "period": 24,
    },
)


def _load_values(np: Any, records: list[Any], clean_root: Path) -> dict[str, Any]:
    """Load and verify the same frozen arrays without importing Torch trainers."""

    values: dict[str, Any] = {}
    for record in records:
        key = hashlib.sha256(
            json.dumps(
                [record.source_id, record.dataset_id, record.entity_id],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        slot = clean_root / key
        array = np.load(slot / "values.npy", allow_pickle=False)
        timestamps = (
            None
            if record.timestamps_sha is None
            else np.load(slot / "timestamps.npy", allow_pickle=False)
        )
        record.verify_values(array, timestamps=timestamps)
        values[record.series_uid] = np.asarray(array, dtype=np.float64)
    return values


def _fresh_roster(
    np: Any,
    *,
    root: Path,
    registry_rows: list[Any],
    dataset_id: str,
    spec: dict[str, object],
) -> list[dict[str, object]]:
    """Reproduce the existing exposed frozen-replay roster."""

    required_stop = max(max(ANCHORS) + HORIZON, int(spec["future_bounds"][1]))
    candidates = sorted(
        (
            row
            for row in registry_rows
            if row.dataset_id == dataset_id and int(row.length) >= required_stop
        ),
        key=lambda row: row.series_uid,
    )
    candidate_values = _load_values(
        np, candidates, root / "data/benchmark_v0_2/clean_base"
    )
    eligible = [
        row
        for row in candidates
        if np.isfinite(
            np.asarray(
                candidate_values[row.series_uid][48:required_stop], dtype=np.float64
            )
        ).all()
    ]
    if len(eligible) < 20:
        raise ValueError(f"insufficient exposed Source candidates: {dataset_id}")
    return [
        {
            "dataset_id": dataset_id,
            "series_uid": row.series_uid,
            "cohort": "train" if index < 12 else "eval",
        }
        for index, row in enumerate(eligible[:20])
    ]


def _center_scale(np: Any, values: Any) -> tuple[float, float, str]:
    observed = np.asarray(values, dtype=np.float64)
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        raise ValueError("cannot standardize a context with no observed values")
    center = float(np.median(observed))
    mad_scale = 1.4826 * float(np.median(np.abs(observed - center)))
    if np.isfinite(mad_scale) and mad_scale >= 1e-6:
        return center, mad_scale, "median_mad"
    std_scale = float(np.std(observed))
    if np.isfinite(std_scale) and std_scale >= 1e-6:
        return center, std_scale, "median_std_fallback"
    return center, 1e-6, "scale_floor_fallback"


def _response_centroid(
    np: Any,
    raw_values: Any,
    *,
    period: int,
    visible_stop: int,
) -> Any:
    """Centroid of pre-cutoff normalized actual-minus-seasonal-naive responses."""

    raw = np.asarray(raw_values, dtype=np.float64)
    responses = []
    for anchor in ANCHORS:
        if anchor + HORIZON > visible_stop:
            raise ValueError("response pseudo-origin crosses the deployment cutoff")
        context = np.asarray(
            raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64
        )
        actual = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
        _, scale, method = _center_scale(np, context)
        if (
            context.shape != (CONTEXT_LENGTH,)
            or actual.shape != (HORIZON,)
            or period < 1
            or period > CONTEXT_LENGTH
            or not np.isfinite(context).all()
            or not np.isfinite(actual).all()
            or method == "scale_floor_fallback"
        ):
            raise ValueError("invalid pre-cutoff response geometry")
        seasonal_naive = np.resize(context[-period:], HORIZON)
        responses.append((actual - seasonal_naive) / scale)
    centroid = np.mean(np.asarray(responses, dtype=np.float64), axis=0)
    if centroid.shape != (HORIZON,) or not np.isfinite(centroid).all():
        raise RuntimeError("invalid response centroid")
    return centroid


def _candidate_features(row: dict[str, object]) -> list[float]:
    proxy = float(row["support_first_order_proxy_gain"])
    mismatch = float(row["response_mismatch"])
    is_exclude = float(str(row["program"]) == "EXCLUDE_DONOR")
    return [
        proxy,
        mismatch,
        is_exclude,
        proxy * mismatch,
        mismatch * is_exclude,
    ]


def _ordered_candidates(
    rows: list[dict[str, object]], score_key: str
) -> list[dict[str, object]]:
    """Descending score with the frozen ascending program/donor tie break."""

    return sorted(
        rows,
        key=lambda row: (
            -float(row[score_key]),
            str(row["program"]),
            str(row["donor_uid"]),
        ),
    )


def _confirm_from_order(
    order: list[dict[str, object]], budget: int
) -> dict[str, object]:
    """Probe exactly B candidates; execute only a positive exact Support winner."""

    if budget not in P1_BUDGETS:
        raise ValueError("unsupported P1 feedback budget")
    if budget == 0:
        return {
            "budget": 0,
            "probed_candidates": [],
            "selected_program": "IDENTITY",
            "selected_donor_uid": None,
            "selected_support_exact_gain": 0.0,
            "fixed_query_exact_gain": 0.0,
            "abstained": True,
        }
    probed = order[:budget]
    winner = max(
        probed,
        key=lambda row: (
            float(row["support_exact_gain"]),
            str(row["program"]),
            str(row["donor_uid"]),
        ),
    )
    if float(winner["support_exact_gain"]) <= 0.0:
        return {
            "budget": budget,
            "probed_candidates": [
                {
                    "program": row["program"],
                    "donor_uid": row["donor_uid"],
                    "support_exact_gain": row["support_exact_gain"],
                }
                for row in probed
            ],
            "selected_program": "IDENTITY",
            "selected_donor_uid": None,
            "selected_support_exact_gain": 0.0,
            "fixed_query_exact_gain": 0.0,
            "abstained": True,
        }
    return {
        "budget": budget,
        "probed_candidates": [
            {
                "program": row["program"],
                "donor_uid": row["donor_uid"],
                "support_exact_gain": row["support_exact_gain"],
            }
            for row in probed
        ],
        "selected_program": winner["program"],
        "selected_donor_uid": winner["donor_uid"],
        "selected_support_exact_gain": float(winner["support_exact_gain"]),
        "fixed_query_exact_gain": float(winner["query_exact_gain"]),
        "abstained": False,
    }


def _adapt_auc(budget_rows: list[dict[str, object]]) -> float:
    gains = {
        int(row["budget"]): float(row["fixed_query_exact_gain"])
        for row in budget_rows
    }
    if tuple(sorted(gains)) != P1_BUDGETS:
        raise ValueError("P1 adaptation curve has the wrong budgets")
    area = sum(
        0.5 * (gains[left] + gains[right]) * (right - left)
        for left, right in zip(P1_BUDGETS, P1_BUDGETS[1:])
    )
    return area / float(P1_BUDGETS[-1] - P1_BUDGETS[0])


def _oracle_probe_order(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Exact exhaustive B<=2 order used only as an exposed upper bound."""

    stable = sorted(
        rows, key=lambda row: (str(row["program"]), str(row["donor_uid"]))
    )
    best_pair: tuple[dict[str, object], dict[str, object]] | None = None
    best_auc = float("-inf")
    for first in stable:
        for second in stable:
            if first is second:
                continue
            pair = [first, second]
            curve = [_confirm_from_order(pair, budget) for budget in P1_BUDGETS]
            auc = _adapt_auc(curve)
            if auc > best_auc:
                best_auc = auc
                best_pair = (first, second)
    if best_pair is None:
        raise RuntimeError("oracle probe-order search found no pair")
    used = {id(best_pair[0]), id(best_pair[1])}
    remainder = [
        row
        for row in _ordered_candidates(rows, "query_exact_gain")
        if id(row) not in used
    ]
    return [best_pair[0], best_pair[1], *remainder]


def _workflow_curve(
    workflows: dict[str, dict[str, object]], order: tuple[str, str]
) -> list[dict[str, object]]:
    """Run one complete-Workflow probe order under the fixed Support rule."""

    if set(workflows) != set(P1B_WORKFLOW_IDS) or order not in P1B_ORDERS:
        raise ValueError("invalid P1b Workflow supply or order")
    curve = [
        {
            "budget": 0,
            "probed_workflows": [],
            "selected_workflow": "IDENTITY",
            "selected_support_gain": 0.0,
            "fixed_query_gain": 0.0,
            "abstained": True,
        }
    ]
    for budget in (1, 2):
        probed = [workflows[workflow_id] for workflow_id in order[:budget]]
        winner = max(
            probed,
            key=lambda row: (
                float(row["support_gain"]),
                str(row["workflow_id"]) == "W_rowblock",
            ),
        )
        executes = float(winner["support_gain"]) > 0.0
        curve.append(
            {
                "budget": budget,
                "probed_workflows": [
                    {
                        "workflow_id": row["workflow_id"],
                        "support_gain": float(row["support_gain"]),
                    }
                    for row in probed
                ],
                "selected_workflow": (
                    str(winner["workflow_id"]) if executes else "IDENTITY"
                ),
                "selected_support_gain": (
                    float(winner["support_gain"]) if executes else 0.0
                ),
                "fixed_query_gain": (
                    float(winner["query_gain"]) if executes else 0.0
                ),
                "abstained": not executes,
            }
        )
    return curve


def _average_workflow_curves(
    curves: list[list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Exact expectation over the two legal A3 probe orders."""

    if len(curves) != 2:
        raise ValueError("A3 requires exactly two Workflow orders")
    averaged = []
    for budget in P1_BUDGETS:
        points = [
            next(point for point in curve if int(point["budget"]) == budget)
            for curve in curves
        ]
        averaged.append(
            {
                "budget": budget,
                "fixed_query_exact_gain": statistics.fmean(
                    float(point["fixed_query_gain"]) for point in points
                ),
                "abstention_probability": statistics.fmean(
                    float(bool(point["abstained"])) for point in points
                ),
                "harm_probability": statistics.fmean(
                    float(float(point["fixed_query_gain"]) < 0.0)
                    for point in points
                ),
                "selected_workflow_probability": {
                    workflow_id: statistics.fmean(
                        float(str(point["selected_workflow"]) == workflow_id)
                        for point in points
                    )
                    for workflow_id in (*P1B_WORKFLOW_IDS, "IDENTITY")
                },
            }
        )
    return averaged


def _p1b_auc(curve: list[dict[str, object]]) -> float:
    normalized = [
        {
            "budget": point["budget"],
            "fixed_query_exact_gain": (
                point["fixed_query_exact_gain"]
                if "fixed_query_exact_gain" in point
                else point["fixed_query_gain"]
            ),
        }
        for point in curve
    ]
    return _adapt_auc(normalized)


def workflow_curve_from_policy_episode(
    workflows: dict[str, dict[str, object]], order: tuple[str, ...]
) -> list[dict[str, object]]:
    """Replay an arbitrary complete-Workflow probe order from one PolicyEpisode."""

    if not order or set(order) != set(workflows):
        raise ValueError("Workflow order must contain each PolicyEpisode Workflow once")
    curve = [
        {
            "budget": 0,
            "selected_workflow": "IDENTITY",
            "fixed_query_gain": 0.0,
            "abstained": True,
        }
    ]
    for budget in range(1, len(order) + 1):
        probed = [workflows[workflow_id] for workflow_id in order[:budget]]
        winner = max(
            probed,
            key=lambda row: (float(row["support_gain"]), str(row["workflow_id"])),
        )
        executes = float(winner["support_gain"]) > 0.0
        curve.append(
            {
                "budget": budget,
                "selected_workflow": (
                    str(winner["workflow_id"]) if executes else "IDENTITY"
                ),
                "fixed_query_gain": (
                    float(winner["query_gain"]) if executes else 0.0
                ),
                "abstained": not executes,
            }
        )
    return curve


def build_historical_policy_episode_failure_card(
    global_prior_failure: dict[str, object],
    overwrite_failure: dict[str, object],
) -> dict[str, object]:
    """Localize the two exposed first faults that produced the admitted Skill."""

    if global_prior_failure.get("verdict") != "SOURCE_POLICY_MEMORY_PROMOTION_REJECTED":
        raise ValueError("global-prior failure evidence is unavailable")
    if overwrite_failure.get("verdict") != "HISTORICAL_POLICY_CONTEXT_SOURCE_REJECTED":
        raise ValueError("positive-Workflow overwrite evidence is unavailable")
    global_auc = global_prior_failure.get("adapt_auc")
    global_workflows = global_prior_failure.get("workflow_outcomes")
    overwrite_context = overwrite_failure.get("historical_policy_context")
    overwrite_workflows = overwrite_failure.get("workflow_outcomes")
    overwrite_curve = overwrite_failure.get("adaptation_curve")
    if not all(
        isinstance(value, dict)
        for value in (
            global_auc,
            global_workflows,
            overwrite_context,
            overwrite_workflows,
        )
    ) or not isinstance(overwrite_curve, list):
        raise ValueError("incomplete PolicyEpisode failure evidence")
    if float(global_auc["A5"]) >= float(global_auc["A3"]):
        raise ValueError("global Workflow prior did not exhibit the required first fault")
    if not (
        float(global_workflows["W_temporal_origin"]["support_gain"]) > 0.0
        and float(global_workflows["W_temporal_origin"]["query_gain"]) > 0.0
    ):
        raise ValueError("global-prior failure lacks a delayed positive Workflow")
    historical_order = tuple(overwrite_context["compiled_order"])
    first = str(historical_order[0])
    if not (
        float(overwrite_workflows[first]["support_gain"]) > 0.0
        and float(overwrite_workflows[first]["query_gain"]) > 0.0
        and float(overwrite_curve[-1]["fixed_query_gain"]) < 0.0
    ):
        raise ValueError("overwrite failure lacks a positive incumbent followed by harm")

    return {
        "card_id": "historical_policy_episode_first_faults_v1",
        "workflow_supply": list(global_workflows),
        "first_faults": [
            {
                "surface": "observation",
                "code": "GLOBAL_WORKFLOW_ORDER_NOT_TARGET_CONTEXTUALIZED",
                "observed_behavior": "a useful Workflow is delayed by a global Source order",
                "counterfactual_repair": "order probes using a legal phase-aligned historical PolicyEpisode",
            },
            {
                "surface": "harness_update_policy",
                "code": "CONFIRMED_POSITIVE_WORKFLOW_OVERWRITTEN",
                "observed_behavior": "later Support gains overwrite an already useful Workflow",
                "counterfactual_repair": "stop after the first positive exact current-Support confirmation",
            },
        ],
        "allowed_patch_surfaces": ["observation", "harness_update_policy"],
        "forbidden_changes": ["program_supply", "consumer", "metric", "memory_schema"],
    }


def compile_historical_policy_episode_typed_patch(
    card: dict[str, object],
) -> dict[str, object]:
    """Compile the smallest behavior patch permitted by the failure card."""

    faults = card.get("first_faults")
    allowed = card.get("allowed_patch_surfaces")
    if not isinstance(faults, list) or not isinstance(allowed, list):
        raise ValueError("invalid historical PolicyEpisode failure card")
    codes = {str(row.get("code")) for row in faults if isinstance(row, dict)}
    expected = {
        "GLOBAL_WORKFLOW_ORDER_NOT_TARGET_CONTEXTUALIZED",
        "CONFIRMED_POSITIVE_WORKFLOW_OVERWRITTEN",
    }
    if codes != expected or set(allowed) != {"observation", "harness_update_policy"}:
        raise ValueError("failure card does not support the frozen typed patch")
    workflow_supply = card.get("workflow_supply")
    if not isinstance(workflow_supply, list) or len(workflow_supply) < 2:
        raise ValueError("failure card lacks a Workflow supply")

    return {
        "patch_id": "add_historical_policy_observation_and_stop_control_v1",
        "operations": [
            {
                "operation": "ADD_OBSERVATION",
                "target_surface": "observation",
                "value": "phase_aligned_historical_policy_episode",
            },
            {
                "operation": "PATCH_CONTROL",
                "target_surface": "harness_update_policy",
                "value": "stop_on_first_positive",
            },
        ],
        "behavior": {
            "workflow_supply": list(workflow_supply),
            "observation": {
                "type": "phase_aligned_historical_policy_episode",
                "visibility": "targets end before current query cutoff",
                "use": "order workflow probes by descending historical exact gain",
                "utility_claim": "observation only; not a current-query utility certificate",
            },
            "control": {
                "type": "stop_on_first_positive",
                "confirmation": "current Support exact grouped gain > 0",
                "continue_when": "current Support exact grouped gain <= 0",
                "fallback": "IDENTITY",
            },
            "risk": {
                "abstain_if_no_positive_confirmation": True,
                "do_not_use_query_future_for_ordering_or_confirmation": True,
                "do_not_allow_later_probe_to_overwrite_confirmed_workflow": True,
            },
        },
        "unchanged": ["program_supply", "consumer", "metric", "memory_schema"],
    }


def _direct_weighted_prediction(
    np: Any,
    *,
    x_train: Any,
    targets: Any,
    x_eval: Any,
    selected_rows: tuple[int, ...],
    donor_weight: float,
) -> Any:
    """Direct weighted Ridge solve used once as a mechanical cross-check."""

    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    query = np.asarray(x_eval, dtype=np.float64)
    selected = np.asarray(selected_rows, dtype=np.int64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or query.ndim != 2
        or x.shape[0] != y.shape[0]
        or query.shape[1] != x.shape[1]
        or selected.shape != (EXPECTED_TRAINING_ROWS_PER_DONOR,)
        or int(np.min(selected)) < 0
        or int(np.max(selected)) >= x.shape[0]
        or not 0.0 <= donor_weight <= 1.0
    ):
        raise ValueError("invalid direct weighted Ridge geometry")
    weights = np.ones(x.shape[0], dtype=np.float64)
    weights[selected] = donor_weight
    z_train = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    z_eval = np.column_stack((query, np.ones(query.shape[0], dtype=np.float64)))
    system = z_train.T @ (weights[:, None] * z_train)
    system[:-1, :-1] += RIDGE_ALPHA * np.eye(x.shape[1], dtype=np.float64)
    rhs = z_train.T @ (weights[:, None] * y)
    prediction = z_eval @ np.linalg.solve(system, rhs)
    if not np.isfinite(prediction).all():
        raise RuntimeError("direct weighted Ridge prediction is non-finite")
    return prediction


def _best_candidate(
    rows: list[dict[str, object]], gain_key: str
) -> dict[str, object]:
    """Choose the largest strictly positive gain, otherwise IDENTITY."""

    best = max(
        rows,
        key=lambda row: (
            float(row[gain_key]),
            str(row["program"]) == "EXCLUDE_DONOR",
            str(row["donor_uid"]),
        ),
    )
    if float(best[gain_key]) <= 0.0:
        return {
            "program": "IDENTITY",
            "donor_uid": None,
            "donor_weight": 1.0,
            "gain": 0.0,
        }
    return {
        "program": str(best["program"]),
        "donor_uid": str(best["donor_uid"]),
        "donor_weight": float(best["donor_weight"]),
        "gain": float(best[gain_key]),
    }


def bind_temporal_origin_workflow(
    np: Any,
    *,
    reference: dict[str, object],
    score_support: Any,
    anchors: tuple[int, ...] = ANCHORS,
    rows_per_origin: int = EXPECTED_TRAINING_SERIES,
) -> dict[str, object]:
    """Bind one complete-origin curation action using proxy then exact feedback.

    The cheap first-order signal only chooses which origin/dose to inspect.  The
    returned decision executes the selected action only when its exact grouped
    Support gain is positive.  Per-series gains remain visible to later
    Context/Risk updates, but this helper intentionally does not hard-code the
    still-unvalidated cohort-coherence guard.
    """

    baseline = np.asarray(reference["baseline_prediction"], dtype=np.float64)
    baseline_losses = np.asarray(score_support(baseline), dtype=np.float64)
    directions = np.asarray(reference["candidate_directions"], dtype=np.float64)
    residual = np.asarray(reference["candidate_full_residual"], dtype=np.float64)
    z_eval = np.asarray(reference["evaluation_design"], dtype=np.float64)
    expected_rows = len(anchors) * rows_per_origin
    if (
        baseline_losses.ndim != 1
        or directions.shape[1] != expected_rows
        or residual.shape[0] != expected_rows
    ):
        raise ValueError("invalid temporal-origin Workflow geometry")

    candidates: list[dict[str, object]] = []
    for origin_index, origin in enumerate(anchors):
        rows = tuple(
            range(
                origin_index * rows_per_origin,
                (origin_index + 1) * rows_per_origin,
            )
        )
        selected = np.asarray(rows, dtype=np.int64)
        for program, origin_weight in TEMPORAL_ORIGIN_PROGRAMS:
            removal_strength = 1.0 - origin_weight
            proxy_prediction = baseline - removal_strength * (
                (z_eval @ directions[:, selected]) @ residual[selected]
            )
            proxy_gain = float(
                np.mean(baseline_losses - np.asarray(score_support(proxy_prediction)))
            )
            candidates.append(
                {
                    "program": program,
                    "origin": int(origin),
                    "origin_weight": origin_weight,
                    "removal_strength": removal_strength,
                    "selected_rows": list(rows),
                    "support_first_order_proxy_gain": proxy_gain,
                }
            )
    winner = max(
        candidates,
        key=lambda row: (
            float(row["support_first_order_proxy_gain"]),
            str(row["program"]) == "EXCLUDE_ORIGIN_GROUP",
            int(row["origin"]),
        ),
    )
    grouped = _group_removal_predictions(
        np,
        reference=reference,
        selected_local_indices=tuple(int(index) for index in winner["selected_rows"]),
        target_block=(0, HORIZON),
        removal_strength=float(winner["removal_strength"]),
    )
    exact_prediction = np.asarray(grouped["exact_group_prediction"], dtype=np.float64)
    per_series_gain = baseline_losses - np.asarray(
        score_support(exact_prediction), dtype=np.float64
    )
    support_gain = float(np.mean(per_series_gain))
    executes = support_gain > 0.0
    return {
        "workflow_id": "W_temporal_origin",
        "binding": winner,
        "candidate_count": len(candidates),
        "support_exact_gain": support_gain,
        "per_support_series_exact_gain": [float(value) for value in per_series_gain],
        "positive_support_series_fraction": float(np.mean(per_series_gain > 0.0)),
        "decision": "EXECUTE" if executes else "ABSTAIN",
        "prediction": exact_prediction if executes else baseline.copy(),
        "grouped_small_matrix_solve_count": int(grouped["small_matrix_solve_count"]),
    }


def _program_counts(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    return {
        program: sum(str(row[key]["program"]) == program for row in rows)
        for program in ("IDENTITY", "ATTENUATE_DONOR", "EXCLUDE_DONOR")
    }


def run(root: Path, *, capture_p1: bool = False) -> dict[str, object]:
    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl

    j0_report = _read_object(root / J0_REPORT_PATH)
    if j0_report.get("target_query_opened") is not False:
        raise ValueError("J0 Target/Query boundary is not closed")
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
        raise ValueError("UCI is forbidden in this exposed Source P0")

    specs = {**SPECS, **FRESH_SPECS}
    if len(specs) != 4 or len(roster) != 20 * len(specs):
        raise ValueError("expected four exposed 12-train/8-eval Source rosters")
    values = _load_values(
        np,
        [records[str(row["series_uid"])] for row in roster],
        root / "data/benchmark_v0_2/clean_base",
    )

    dataset_evidence: list[dict[str, object]] = []
    mechanical_check: dict[str, object] | None = None
    reference_solve_count = 0
    grouped_small_matrix_solve_count = 0

    for dataset_id, spec in specs.items():
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
        if (
            len(train_rows) != EXPECTED_TRAINING_SERIES
            or len(eval_rows) != EXPECTED_EVALUATION_ROWS
        ):
            raise ValueError(f"exposed roster geometry changed: {dataset_id}")

        x_rows: list[Any] = []
        y_rows: list[Any] = []
        row_donor_uids: list[str] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                raw = values[uid]
                context = np.asarray(
                    raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64
                )
                target = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
                center, scale, method = _center_scale(np, context)
                if (
                    context.shape != (CONTEXT_LENGTH,)
                    or target.shape != (HORIZON,)
                    or not np.isfinite(context).all()
                    or not np.isfinite(target).all()
                    or method == "scale_floor_fallback"
                ):
                    raise ValueError(f"invalid clean training window: {uid}/{anchor}")
                x_rows.append(
                    np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
                )
                y_rows.append((target - center) / scale)
                row_donor_uids.append(uid)
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_y = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (EXPECTED_TRAINING_ROWS, 2 * CONTEXT_LENGTH) or clean_y.shape != (
            EXPECTED_TRAINING_ROWS,
            HORIZON,
        ):
            raise AssertionError(f"unexpected training geometry: {dataset_id}")

        donor_uids = tuple(str(row["series_uid"]) for row in train_rows)
        donor_rows = {
            uid: tuple(index for index, bound_uid in enumerate(row_donor_uids) if bound_uid == uid)
            for uid in donor_uids
        }
        if any(
            len(indices) != EXPECTED_TRAINING_ROWS_PER_DONOR
            for indices in donor_rows.values()
        ):
            raise AssertionError(f"donor binding geometry changed: {dataset_id}")

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
            center, scale, method = _center_scale(np, context)
            if (
                context.shape != (CONTEXT_LENGTH,)
                or future.shape != (HORIZON,)
                or not np.isfinite(context).all()
                or not np.isfinite(future).all()
                or method == "scale_floor_fallback"
            ):
                raise ValueError(f"invalid evaluation window: {uid}")
            try:
                seasonal_by_uid[uid] = seasonal_scale(
                    np.asarray(raw[:train_stop], dtype=np.float64),
                    np.isfinite(raw[:train_stop]),
                    period=int(spec["period"]),
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(f"invalid evaluation sMASE scale: {uid}") from error
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
            if prediction.shape != (EXPECTED_EVALUATION_ROWS, HORIZON):
                raise RuntimeError(f"invalid Ridge prediction shape: {dataset_id}")
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"non-finite Ridge prediction: {dataset_id}")
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

        reference = _ridge_reference_and_removal_predictions(
            np,
            x_train=x_train,
            targets=clean_y,
            x_eval=x_eval_array,
            candidate_rows=tuple(range(EXPECTED_TRAINING_ROWS)),
            target_block=(0, HORIZON),
            alpha=RIDGE_ALPHA,
        )
        reference_solve_count += 1
        baseline_prediction = np.asarray(reference["baseline_prediction"], dtype=np.float64)
        baseline_losses = score_predictions(baseline_prediction)
        folds = _alternating_folds(EXPECTED_EVALUATION_ROWS)
        donor_response_centroids = (
            {
                uid: _response_centroid(
                    np,
                    values[uid],
                    period=int(spec["period"]),
                    visible_stop=max(ANCHORS) + HORIZON,
                )
                for uid in donor_uids
            }
            if capture_p1
            else {}
        )
        evaluation_response_centroids = (
            [
                _response_centroid(
                    np,
                    values[uid],
                    period=int(spec["period"]),
                    visible_stop=int(spec["train_stop"]),
                )
                for uid in eval_uids
            ]
            if capture_p1
            else []
        )

        candidates: list[dict[str, object]] = []
        for donor_uid in donor_uids:
            selected_rows = donor_rows[donor_uid]
            for program, donor_weight in PROGRAMS:
                grouped = _group_removal_predictions(
                    np,
                    reference=reference,
                    selected_local_indices=selected_rows,
                    target_block=(0, HORIZON),
                    removal_strength=1.0 - donor_weight,
                )
                grouped_small_matrix_solve_count += int(grouped["small_matrix_solve_count"])
                exact_prediction = np.asarray(
                    grouped["exact_group_prediction"], dtype=np.float64
                )
                exact_losses = score_predictions(exact_prediction)
                per_series_gain = baseline_losses - exact_losses
                candidate = {
                    "program": program,
                    "donor_uid": donor_uid,
                    "bound_training_row_indices": list(selected_rows),
                    "bound_anchor_count": len(selected_rows),
                    "bound_output_interval_half_open": [0, HORIZON],
                    "donor_weight": donor_weight,
                    "removal_strength": 1.0 - donor_weight,
                    "middle_condition_number": grouped["middle_condition_number"],
                    "per_evaluation_series_exact_gain": [
                        float(value) for value in per_series_gain
                    ],
                    "cohort_exact_gain": {
                        fold_name: _mean(per_series_gain, fold_indices)
                        for fold_name, fold_indices in folds.items()
                    },
                }
                if capture_p1:
                    first_order_prediction = np.asarray(
                        grouped["first_order_group_proxy_prediction"], dtype=np.float64
                    )
                    first_order_losses = score_predictions(first_order_prediction)
                    per_series_first_order_gain = baseline_losses - first_order_losses
                    candidate["per_evaluation_series_first_order_proxy_gain"] = [
                        float(value) for value in per_series_first_order_gain
                    ]
                    candidate["cohort_first_order_proxy_gain"] = {
                        fold_name: _mean(per_series_first_order_gain, fold_indices)
                        for fold_name, fold_indices in folds.items()
                    }
                candidates.append(candidate)

                if mechanical_check is None and program == "ATTENUATE_DONOR":
                    direct_prediction = _direct_weighted_prediction(
                        np,
                        x_train=x_train,
                        targets=clean_y,
                        x_eval=x_eval_array,
                        selected_rows=selected_rows,
                        donor_weight=donor_weight,
                    )
                    max_abs_error = float(np.max(np.abs(exact_prediction - direct_prediction)))
                    mechanical_check = {
                        "dataset_id": dataset_id,
                        "donor_uid": donor_uid,
                        "program": program,
                        "donor_weight": donor_weight,
                        "max_abs_prediction_error": max_abs_error,
                        "threshold": 1e-8,
                        "passed": max_abs_error <= 1e-8,
                        "direct_weighted_solve_count": 1,
                    }

        direction_evidence: list[dict[str, object]] = []
        for direction, support_name, query_name in FOLD_PAIRS:
            query_response_centroid = (
                np.mean(
                    np.asarray(
                        [
                            evaluation_response_centroids[index]
                            for index in folds[query_name]
                        ],
                        dtype=np.float64,
                    ),
                    axis=0,
                )
                if capture_p1
                else None
            )
            action_rows = []
            for candidate in candidates:
                action_row = {
                    "program": str(candidate["program"]),
                    "donor_uid": str(candidate["donor_uid"]),
                    "donor_weight": float(candidate["donor_weight"]),
                    "support_exact_gain": float(candidate["cohort_exact_gain"][support_name]),
                    "query_exact_gain": float(candidate["cohort_exact_gain"][query_name]),
                }
                if capture_p1:
                    donor_centroid = donor_response_centroids[
                        str(candidate["donor_uid"])
                    ]
                    mismatch = float(
                        np.linalg.norm(donor_centroid - query_response_centroid)
                        / np.sqrt(HORIZON)
                    )
                    action_row.update(
                        {
                            "support_first_order_proxy_gain": float(
                                candidate["cohort_first_order_proxy_gain"][support_name]
                            ),
                            "response_mismatch": mismatch,
                        }
                    )
                action_rows.append(action_row)
            support_selected = _best_candidate(action_rows, "support_exact_gain")
            if support_selected["program"] == "IDENTITY":
                selected_query_gain = 0.0
            else:
                selected_row = next(
                    row
                    for row in action_rows
                    if row["program"] == support_selected["program"]
                    and row["donor_uid"] == support_selected["donor_uid"]
                )
                selected_query_gain = float(selected_row["query_exact_gain"])
            query_oracle = _best_candidate(action_rows, "query_exact_gain")
            direction_row = {
                    "direction": direction,
                    "support_cohort": support_name,
                    "query_cohort": query_name,
                    "actions": action_rows,
                    "support_selected": {
                        **support_selected,
                        "support_exact_gain": float(support_selected["gain"]),
                        "held_out_query_exact_gain": selected_query_gain,
                    },
                    "menu_query_oracle": {
                        **query_oracle,
                        "query_exact_headroom": float(query_oracle["gain"]),
                    },
                    "identity_is_query_optimal": query_oracle["program"] == "IDENTITY",
                }
            if capture_p1:
                direction_row["response_observation"] = {
                    "observation_id": "cross_series_response_alignment_v1",
                    "query_series_uids": [
                        eval_uids[index] for index in folds[query_name]
                    ],
                    "pseudo_origins": list(ANCHORS),
                    "pseudo_origin_count_per_series": len(ANCHORS),
                    "visible_stop": int(spec["train_stop"]),
                    "future_or_exact_query_gain_used": False,
                }
            direction_evidence.append(direction_row)

        sign_flips = []
        for candidate in candidates:
            gain_a = float(candidate["cohort_exact_gain"]["fold_a"])
            gain_b = float(candidate["cohort_exact_gain"]["fold_b"])
            if gain_a * gain_b < 0.0:
                sign_flips.append(
                    {
                        "program": candidate["program"],
                        "donor_uid": candidate["donor_uid"],
                        "donor_weight": candidate["donor_weight"],
                        "fold_a_exact_gain": gain_a,
                        "fold_b_exact_gain": gain_b,
                    }
                )

        support_selected_mean = statistics.fmean(
            float(row["support_selected"]["held_out_query_exact_gain"])
            for row in direction_evidence
        )
        menu_oracle_mean = statistics.fmean(
            float(row["menu_query_oracle"]["query_exact_headroom"])
            for row in direction_evidence
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "train_donor_uids": list(donor_uids),
                "evaluation_uids": eval_uids,
                "fold_membership": {
                    name: [eval_uids[index] for index in indices]
                    for name, indices in folds.items()
                },
                "candidate_count_excluding_identity": len(candidates),
                "direction_evidence": direction_evidence,
                "same_bound_candidate_cohort_sign_flips": sign_flips,
                "summary": {
                    "mean_support_selected_held_out_query_gain": support_selected_mean,
                    "mean_menu_oracle_query_headroom": menu_oracle_mean,
                    "positive_support_selected_dataset": support_selected_mean > 0.0,
                    "identity_query_optimal_direction_count": sum(
                        bool(row["identity_is_query_optimal"])
                        for row in direction_evidence
                    ),
                    "same_bound_candidate_sign_flip_count": len(sign_flips),
                    "support_selected_program_counts": _program_counts(
                        direction_evidence, "support_selected"
                    ),
                    "query_oracle_program_counts": _program_counts(
                        direction_evidence, "menu_query_oracle"
                    ),
                },
            }
        )

    if mechanical_check is None or not bool(mechanical_check["passed"]):
        raise AssertionError("weighted grouped Ridge mechanical check failed")
    dataset_selected_gains = [
        float(row["summary"]["mean_support_selected_held_out_query_gain"])
        for row in dataset_evidence
    ]
    dataset_oracle_headroom = [
        float(row["summary"]["mean_menu_oracle_query_headroom"])
        for row in dataset_evidence
    ]
    positive_dataset_count = sum(value > 0.0 for value in dataset_selected_gains)
    dataset_macro_selected_gain = statistics.fmean(dataset_selected_gains)
    identity_optimal_exists = any(
        int(row["summary"]["identity_query_optimal_direction_count"]) > 0
        for row in dataset_evidence
    )
    cohort_sign_flip_exists = any(
        int(row["summary"]["same_bound_candidate_sign_flip_count"]) > 0
        for row in dataset_evidence
    )
    context_selectivity_or_risk = identity_optimal_exists or cohort_sign_flip_exists
    gate = {
        "minimum_positive_dataset_count": 2,
        "observed_positive_dataset_count": positive_dataset_count,
        "positive_dataset_count_passed": positive_dataset_count >= 2,
        "dataset_macro_support_selected_gain": dataset_macro_selected_gain,
        "dataset_macro_gain_passed": dataset_macro_selected_gain > 0.0,
        "identity_query_optimal_exists": identity_optimal_exists,
        "same_bound_candidate_cohort_sign_flip_exists": cohort_sign_flip_exists,
        "context_selectivity_or_risk_passed": context_selectivity_or_risk,
    }
    passed = all(
        (
            gate["positive_dataset_count_passed"],
            gate["dataset_macro_gain_passed"],
            gate["context_selectivity_or_risk_passed"],
        )
    )
    verdict = (
        "P0_DONOR_SERIES_HEADROOM_PASS_PROCEED_TO_P1"
        if passed
        else "P0_DONOR_SERIES_FAMILY_V1_CLOSED"
    )
    all_directions = [
        direction
        for dataset in dataset_evidence
        for direction in dataset["direction_evidence"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "E2-cross-series-curation-P0",
        "stage": "development_exposed_program_headroom",
        "scientific_question": (
            "Does one full donor-series action with a fixed 1/0.25/0 sample-weight "
            "menu expose natural cohort-conditioned grouped downstream headroom?"
        ),
        "protocol": {
            "datasets": list(specs),
            "roster": "existing exposed 12 donor + 8 evaluation series per dataset",
            "training_windows_per_donor": EXPECTED_TRAINING_ROWS_PER_DONOR,
            "anchors": list(ANCHORS),
            "context_length": CONTEXT_LENGTH,
            "forecast_horizon": HORIZON,
            "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE; unweighted cohort and dataset means",
            "support_query": "alternating 4/4 folds, evaluated in both directions",
            "candidate_binding": "one donor's six rows x all 48 outputs",
            "program_menu": {
                "IDENTITY": {"sample_weight": 1.0},
                "ATTENUATE_DONOR": {"sample_weight": 0.25},
                "EXCLUDE_DONOR": {"sample_weight": 0.0},
            },
            "selection": (
                "maximize exact grouped support gain over donor and dose; choose "
                "IDENTITY when the best non-identity gain is <= 0"
            ),
        },
        "dataset_evidence": dataset_evidence,
        "dataset_macro": {
            "support_selected_held_out_query_gain": dataset_macro_selected_gain,
            "menu_oracle_query_headroom": statistics.fmean(dataset_oracle_headroom),
            "positive_support_selected_dataset_count": positive_dataset_count,
            "support_selected_program_counts": _program_counts(
                all_directions, "support_selected"
            ),
            "query_oracle_program_counts": _program_counts(
                all_directions, "menu_query_oracle"
            ),
            "same_bound_candidate_cohort_sign_flip_count": sum(
                len(row["same_bound_candidate_cohort_sign_flips"])
                for row in dataset_evidence
            ),
        },
        "weighted_downdate_mechanical_check": mechanical_check,
        "compute": {
            "reference_ridge_solve_count": reference_solve_count,
            "grouped_small_matrix_solve_count": grouped_small_matrix_solve_count,
            "direct_weighted_ridge_check_solve_count": 1,
            "consumer_protocol_changed": False,
        },
        "frozen_gate": gate,
        "gate_passed": passed,
        "verdict": verdict,
        "evidence_boundary": {
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "development_only": True,
            "fresh_evidence": False,
            "target_query_opened": False,
            "capability_promoted": False,
            "transfer_tested": False,
            "memory_used": False,
            "classification": "MECHANISM" if passed else "NEGATIVE",
            "claim": (
                "P0 tests only whether this frozen donor-series Program family has "
                "natural headroom and cohort selectivity/risk. It is not a deployable "
                "Capability, Source-to-Target transfer, or Harness evolution result."
            ),
        },
    }


def run_p1(root: Path) -> dict[str, object]:
    """Run exposed dataset-LODO probe ordering with one response Observation."""

    import numpy as np

    frozen_p0 = _read_object(root / DEFAULT_REPORT_PATH)
    if (
        frozen_p0.get("schema_version") != SCHEMA_VERSION
        or frozen_p0.get("verdict") != "P0_DONOR_SERIES_HEADROOM_PASS_PROCEED_TO_P1"
        or frozen_p0.get("gate_passed") is not True
    ):
        raise ValueError("P1 requires the frozen passing P0 report")
    internal = run(root, capture_p1=True)
    if internal["verdict"] != frozen_p0["verdict"]:
        raise AssertionError("P0 verdict changed during P1 recomputation")
    for key in (
        "support_selected_held_out_query_gain",
        "menu_oracle_query_headroom",
    ):
        if abs(
            float(internal["dataset_macro"][key])
            - float(frozen_p0["dataset_macro"][key])
        ) > 1e-12:
            raise AssertionError(f"P0 metric changed during P1 recomputation: {key}")

    datasets = {
        str(row["dataset_id"]): row for row in internal["dataset_evidence"]
    }
    if len(datasets) != 4:
        raise ValueError("P1 requires exactly four exposed Source datasets")
    dataset_evidence: list[dict[str, object]] = []

    for heldout_dataset_id, heldout_dataset in datasets.items():
        source_dataset_ids = sorted(
            dataset_id for dataset_id in datasets if dataset_id != heldout_dataset_id
        )
        source_episodes: list[dict[str, object]] = []
        source_episode_counts: dict[str, int] = {}
        for source_dataset_id in source_dataset_ids:
            count_before = len(source_episodes)
            for direction in datasets[source_dataset_id]["direction_evidence"]:
                for action in direction["actions"]:
                    source_episodes.append(
                        {
                            "source_dataset_id": source_dataset_id,
                            "direction": direction["direction"],
                            "features": _candidate_features(action),
                            "exact_held_out_query_gain": float(
                                action["query_exact_gain"]
                            ),
                        }
                    )
            source_episode_counts[source_dataset_id] = len(source_episodes) - count_before
        if set(source_episode_counts.values()) != {48}:
            raise AssertionError("Source datasets must contribute equal episode counts")

        feature_matrix = np.asarray(
            [row["features"] for row in source_episodes], dtype=np.float64
        )
        target_values = np.asarray(
            [row["exact_held_out_query_gain"] for row in source_episodes],
            dtype=np.float64,
        )
        episode_weights = np.asarray(
            [
                1.0 / source_episode_counts[str(row["source_dataset_id"])]
                for row in source_episodes
            ],
            dtype=np.float64,
        )
        episode_weights /= float(np.sum(episode_weights))
        feature_mean = np.average(
            feature_matrix, axis=0, weights=episode_weights
        )
        feature_variance = np.average(
            (feature_matrix - feature_mean) ** 2,
            axis=0,
            weights=episode_weights,
        )
        feature_scale = np.sqrt(feature_variance)
        feature_scale[feature_scale < 1e-12] = 1.0
        standardized = (feature_matrix - feature_mean) / feature_scale
        design = np.column_stack(
            (np.ones(standardized.shape[0], dtype=np.float64), standardized)
        )
        sqrt_weight = np.sqrt(episode_weights)
        coefficients, _, design_rank, singular_values = np.linalg.lstsq(
            design * sqrt_weight[:, None],
            target_values * sqrt_weight,
            rcond=None,
        )
        if coefficients.shape != (len(P1_FEATURE_NAMES) + 1,) or not np.isfinite(
            coefficients
        ).all():
            raise RuntimeError("invalid Source PolicyEpisode least-squares model")

        direction_evidence: list[dict[str, object]] = []
        for direction in heldout_dataset["direction_evidence"]:
            candidates = [dict(action) for action in direction["actions"]]
            target_features = np.asarray(
                [_candidate_features(row) for row in candidates], dtype=np.float64
            )
            target_standardized = (target_features - feature_mean) / feature_scale
            source_scores = (
                coefficients[0] + target_standardized @ coefficients[1:]
            )
            for row, score in zip(candidates, source_scores):
                row["source_policy_episode_score"] = float(score)

            a3_order = _ordered_candidates(
                candidates, "support_first_order_proxy_gain"
            )
            a5_order = _ordered_candidates(candidates, "source_policy_episode_score")
            oracle_order = _oracle_probe_order(candidates)
            curves = {
                "A3_target_proxy": [
                    _confirm_from_order(a3_order, budget) for budget in P1_BUDGETS
                ],
                "A5_source_episode_prior": [
                    _confirm_from_order(a5_order, budget) for budget in P1_BUDGETS
                ],
                "Oracle_exact_order": [
                    _confirm_from_order(oracle_order, budget) for budget in P1_BUDGETS
                ],
            }
            auc = {name: _adapt_auc(rows) for name, rows in curves.items()}
            a3_keys = [
                (str(row["program"]), str(row["donor_uid"])) for row in a3_order
            ]
            a5_keys = [
                (str(row["program"]), str(row["donor_uid"])) for row in a5_order
            ]
            direction_evidence.append(
                {
                    "direction": direction["direction"],
                    "support_cohort": direction["support_cohort"],
                    "query_cohort": direction["query_cohort"],
                    "response_observation": direction["response_observation"],
                    "candidate_count": len(candidates),
                    "probe_orders": {
                        "A3_target_proxy": [
                            {
                                "rank": rank,
                                "program": row["program"],
                                "donor_uid": row["donor_uid"],
                                "probe_score": row[
                                    "support_first_order_proxy_gain"
                                ],
                                "response_mismatch": row["response_mismatch"],
                            }
                            for rank, row in enumerate(a3_order, 1)
                        ],
                        "A5_source_episode_prior": [
                            {
                                "rank": rank,
                                "program": row["program"],
                                "donor_uid": row["donor_uid"],
                                "probe_score": row[
                                    "source_policy_episode_score"
                                ],
                                "support_first_order_proxy_gain": row[
                                    "support_first_order_proxy_gain"
                                ],
                                "response_mismatch": row["response_mismatch"],
                            }
                            for rank, row in enumerate(a5_order, 1)
                        ],
                        "Oracle_exact_order": [
                            {
                                "rank": rank,
                                "program": row["program"],
                                "donor_uid": row["donor_uid"],
                                "support_exact_gain": row["support_exact_gain"],
                                "query_exact_gain": row["query_exact_gain"],
                            }
                            for rank, row in enumerate(oracle_order, 1)
                        ],
                    },
                    "order_disagreement": {
                        "top1_differs": a3_keys[0] != a5_keys[0],
                        "top2_order_differs": a3_keys[:2] != a5_keys[:2],
                        "top2_set_overlap_count": len(
                            set(a3_keys[:2]) & set(a5_keys[:2])
                        ),
                        "full_order_differs": a3_keys != a5_keys,
                    },
                    "adaptation_curves": curves,
                    "adapt_auc": auc,
                    "a5_minus_a3_adapt_auc": (
                        auc["A5_source_episode_prior"] - auc["A3_target_proxy"]
                    ),
                    "harmful_at_any_positive_budget": {
                        name: any(
                            float(row["fixed_query_exact_gain"]) < 0.0
                            for row in rows
                            if int(row["budget"]) > 0
                        )
                        for name, rows in curves.items()
                    },
                }
            )

        dataset_auc = {
            arm: statistics.fmean(
                float(row["adapt_auc"][arm]) for row in direction_evidence
            )
            for arm in (
                "A3_target_proxy",
                "A5_source_episode_prior",
                "Oracle_exact_order",
            )
        }
        dataset_evidence.append(
            {
                "heldout_dataset_id": heldout_dataset_id,
                "source_dataset_ids": source_dataset_ids,
                "source_policy_episode_model": {
                    "model": "dataset-equal weighted least_squares",
                    "episode_count": len(source_episodes),
                    "episode_count_by_dataset": source_episode_counts,
                    "feature_names": list(P1_FEATURE_NAMES),
                    "source_feature_mean": [float(value) for value in feature_mean],
                    "source_feature_scale": [float(value) for value in feature_scale],
                    "intercept": float(coefficients[0]),
                    "standardized_feature_coefficients": {
                        name: float(value)
                        for name, value in zip(P1_FEATURE_NAMES, coefficients[1:])
                    },
                    "design_rank": int(design_rank),
                    "minimum_singular_value": float(np.min(singular_values)),
                    "dataset_id_used_as_feature": False,
                    "target_exact_outcome_used_for_order": False,
                },
                "direction_evidence": direction_evidence,
                "summary": {
                    "adapt_auc": dataset_auc,
                    "a5_minus_a3_adapt_auc": (
                        dataset_auc["A5_source_episode_prior"]
                        - dataset_auc["A3_target_proxy"]
                    ),
                    "harmful_direction_count": {
                        arm: sum(
                            bool(row["harmful_at_any_positive_budget"][arm])
                            for row in direction_evidence
                        )
                        for arm in (
                            "A3_target_proxy",
                            "A5_source_episode_prior",
                            "Oracle_exact_order",
                        )
                    },
                    "abstention_count_by_budget": {
                        arm: {
                            str(budget): sum(
                                bool(
                                    next(
                                        point
                                        for point in row["adaptation_curves"][arm]
                                        if int(point["budget"]) == budget
                                    )["abstained"]
                                )
                                for row in direction_evidence
                            )
                            for budget in P1_BUDGETS
                        }
                        for arm in (
                            "A3_target_proxy",
                            "A5_source_episode_prior",
                            "Oracle_exact_order",
                        )
                    },
                    "top1_order_disagreement_count": sum(
                        bool(row["order_disagreement"]["top1_differs"])
                        for row in direction_evidence
                    ),
                },
            }
        )

    macro_auc = {
        arm: statistics.fmean(
            float(row["summary"]["adapt_auc"][arm]) for row in dataset_evidence
        )
        for arm in (
            "A3_target_proxy",
            "A5_source_episode_prior",
            "Oracle_exact_order",
        )
    }
    a5_minus_a3 = (
        macro_auc["A5_source_episode_prior"] - macro_auc["A3_target_proxy"]
    )
    nonnegative_dataset_count = sum(
        float(row["summary"]["a5_minus_a3_adapt_auc"]) >= 0.0
        for row in dataset_evidence
    )
    harmful_direction_count = {
        arm: sum(
            int(row["summary"]["harmful_direction_count"][arm])
            for row in dataset_evidence
        )
        for arm in (
            "A3_target_proxy",
            "A5_source_episode_prior",
            "Oracle_exact_order",
        )
    }
    top1_disagreement_count = sum(
        int(row["summary"]["top1_order_disagreement_count"])
        for row in dataset_evidence
    )
    gate = {
        "macro_a5_minus_a3_adapt_auc": a5_minus_a3,
        "macro_adapt_auc_improved": a5_minus_a3 > 0.0,
        "minimum_nonnegative_heldout_datasets": 3,
        "observed_nonnegative_heldout_datasets": nonnegative_dataset_count,
        "heldout_dataset_coverage_passed": nonnegative_dataset_count >= 3,
        "a3_harmful_direction_count": harmful_direction_count["A3_target_proxy"],
        "a5_harmful_direction_count": harmful_direction_count[
            "A5_source_episode_prior"
        ],
        "harm_not_increased": (
            harmful_direction_count["A5_source_episode_prior"]
            <= harmful_direction_count["A3_target_proxy"]
        ),
        "top1_probe_order_disagreement_count": top1_disagreement_count,
        "probe_order_nontrivially_differs": top1_disagreement_count > 0,
    }
    gate_passed = all(
        (
            gate["macro_adapt_auc_improved"],
            gate["heldout_dataset_coverage_passed"],
            gate["harm_not_increased"],
            gate["probe_order_nontrivially_differs"],
        )
    )
    if not gate["probe_order_nontrivially_differs"]:
        first_fault = "OBSERVATION_DID_NOT_CHANGE_TOP1_PROBE_ORDER"
    elif not gate["macro_adapt_auc_improved"]:
        first_fault = "SOURCE_RESPONSE_ALIGNMENT_ORDER_DID_NOT_BEAT_TARGET_PROXY"
    elif not gate["heldout_dataset_coverage_passed"]:
        first_fault = "SOURCE_ORDER_NEGATIVE_TRANSFER_ACROSS_HELDOUT_DATASETS"
    elif not gate["harm_not_increased"]:
        first_fault = "SOURCE_ORDER_INCREASED_HARMFUL_DIRECTIONS"
    else:
        first_fault = "FRESH_NATURAL_GENERALIZATION_UNTESTED"
    verdict = (
        "P1_RESPONSE_ALIGNMENT_WORKFLOW_PASS_PLAN_FRESH"
        if gate_passed
        else "P1_RESPONSE_ALIGNMENT_WORKFLOW_V1_CLOSED"
    )
    return {
        "schema_version": P1_SCHEMA_VERSION,
        "experiment_id": "E2-cross-series-curation-P1",
        "stage": "development_exposed_dataset_lodo_workflow",
        "scientific_question": (
            "Can Source PolicyEpisodes plus cross-series response alignment improve "
            "donor+dose probe ordering over Target-only first-order proxy ordering at "
            "the same exact Target Support feedback budget?"
        ),
        "frozen_p0_dependency": {
            "report_path": DEFAULT_REPORT_PATH,
            "verdict": frozen_p0["verdict"],
            "program_consumer_roster_metric_fold_gate_changed": False,
        },
        "protocol": {
            "datasets": list(datasets),
            "evaluation": "complete dataset leave-one-out over four exposed datasets",
            "budgets": list(P1_BUDGETS),
            "b0_behavior": "A3 and A5 both IDENTITY",
            "target_confirmation": (
                "open exact grouped Support gain for the first B probes; execute only "
                "the probed candidate with maximum exact Support gain when it is >0"
            ),
            "A3": "rank 24 donor+dose probes by Target Support first-order proxy gain",
            "A5": (
                "rank probes by a Source-only dataset-LODO least-squares prior; Source "
                "never executes an action without exact Target Support confirmation"
            ),
            "oracle": (
                "exhaustive first-two probe order maximizing exposed B=0/1/2 AdaptAUC "
                "under the same Target confirmation rule"
            ),
            "response_observation": {
                "id": "cross_series_response_alignment_v1",
                "per_origin_response": (
                    "(actual 48-step horizon - recursive seasonal-naive horizon) / "
                    "robust context scale"
                ),
                "centroid": "mean over the six frozen ANCHORS",
                "query_cohort_centroid": "mean of four per-series centroids",
                "mismatch": "L2(donor centroid - Query cohort centroid) / sqrt(48)",
                "query_future_used": False,
                "exact_query_gain_used": False,
                "dataset_id_used": False,
            },
            "source_model_features": list(P1_FEATURE_NAMES),
            "source_model": (
                "dataset-equal weighted least squares with Source-fitted frozen "
                "standardization and intercept; no neural model"
            ),
            "harmful_direction_definition": (
                "a dataset-direction with fixed Query exact gain <0 at any B in {1,2}"
            ),
            "adapt_auc": "normalized trapezoidal area over B={0,1,2}",
        },
        "dataset_evidence": dataset_evidence,
        "dataset_macro": {
            "adapt_auc": macro_auc,
            "a5_minus_a3_adapt_auc": a5_minus_a3,
            "nonnegative_a5_minus_a3_dataset_count": nonnegative_dataset_count,
            "harmful_direction_count": harmful_direction_count,
            "top1_probe_order_disagreement_count": top1_disagreement_count,
            "abstention_count_by_budget": {
                arm: {
                    str(budget): sum(
                        int(
                            row["summary"]["abstention_count_by_budget"][arm][
                                str(budget)
                            ]
                        )
                        for row in dataset_evidence
                    )
                    for budget in P1_BUDGETS
                }
                for arm in (
                    "A3_target_proxy",
                    "A5_source_episode_prior",
                    "Oracle_exact_order",
                )
            },
        },
        "compute": {
            "ridge_reference_solve_count": internal["compute"][
                "reference_ridge_solve_count"
            ],
            "grouped_small_matrix_solve_count": internal["compute"][
                "grouped_small_matrix_solve_count"
            ],
            "per_action_consumer_refit_count": 0,
            "source_small_least_squares_fit_count": 4,
            "max_unique_exact_target_probes_per_direction_per_arm": 2,
        },
        "frozen_gate": gate,
        "gate_passed": gate_passed,
        "first_fault": first_fault,
        "verdict": verdict,
        "evidence_boundary": {
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "development_only": True,
            "fresh_evidence": False,
            "target_query_opened": False,
            "capability_promoted": False,
            "transfer_tested": False,
            "memory_used": False,
            "classification": "MECHANISM" if gate_passed else "NEGATIVE",
            "claim": (
                "P1 tests an exposed Source-episode probe-order Workflow only. A PASS "
                "permits a fresh plan; it is not Capability, Transfer, or deployment "
                "evidence, and Source scores never directly execute a Program."
            ),
        },
    }


def run_p1b(root: Path) -> dict[str, object]:
    """Replay Source PolicyEpisodes as complete Workflow probe ordering."""

    rowblock_report = _read_object(root / ROWBLOCK_REPORT_PATH)
    curation_report = _read_object(root / P1_REPORT_PATH)
    if (
        rowblock_report.get("schema_version")
        != "e2-natural-block-action-value-headroom/3"
        or rowblock_report.get("verdict")
        != "NATURAL_BLOCK_ACTION_VALUE_HEADROOM_PRESENT"
        or rowblock_report.get("target_query_opened") is not False
    ):
        raise ValueError("P1b requires the exposed natural row-block report")
    curation_boundary = curation_report.get("evidence_boundary", {})
    if (
        curation_report.get("schema_version") != P1_SCHEMA_VERSION
        or curation_report.get("verdict")
        != "P1_RESPONSE_ALIGNMENT_WORKFLOW_V1_CLOSED"
        or curation_boundary.get("target_query_opened") is not False
        or curation_boundary.get("fresh_evidence") is not False
    ):
        raise ValueError("P1b requires the completed exposed curation P1 report")

    rowblock_by_dataset = {
        str(dataset["dataset_id"]): {
            str(direction["direction"]): direction
            for direction in dataset["crossfit_group_policy"]
        }
        for dataset in rowblock_report["dataset_evidence"]
    }
    curation_by_dataset = {
        str(dataset["heldout_dataset_id"]): {
            str(direction["direction"]): direction
            for direction in dataset["direction_evidence"]
        }
        for dataset in curation_report["dataset_evidence"]
    }
    if (
        set(rowblock_by_dataset) != set(curation_by_dataset)
        or len(rowblock_by_dataset) != 4
        or any(
            set(rowblock_by_dataset[dataset_id]) != {"a_to_b", "b_to_a"}
            or set(curation_by_dataset[dataset_id]) != {"a_to_b", "b_to_a"}
            for dataset_id in rowblock_by_dataset
        )
    ):
        raise ValueError("P1b cached dataset/direction geometry does not align")

    completed_episodes: dict[str, dict[str, dict[str, object]]] = {}
    for dataset_id in rowblock_by_dataset:
        completed_episodes[dataset_id] = {}
        for direction in ("a_to_b", "b_to_a"):
            rowblock = rowblock_by_dataset[dataset_id][direction]["proxy_guided"]
            curation_curve = curation_by_dataset[dataset_id][direction][
                "adaptation_curves"
            ]["A3_target_proxy"]
            curation_b1 = next(
                point for point in curation_curve if int(point["budget"]) == 1
            )
            if (
                (float(rowblock["exact_support_gain"]) > 0.0)
                != (rowblock["cohort_risk_guard"]["decision"] == "EXECUTE")
                or (
                    float(curation_b1["selected_support_exact_gain"]) > 0.0
                )
                == bool(curation_b1["abstained"])
            ):
                raise AssertionError("cached Workflow abstention semantics changed")
            completed_episodes[dataset_id][direction] = {
                "W_rowblock": {
                    "workflow_id": "W_rowblock",
                    "support_gain": float(rowblock["exact_support_gain"]),
                    "query_gain": float(
                        rowblock["cohort_risk_guard"][
                            "guarded_exact_holdout_gain"
                        ]
                    ),
                    "cached_decision": rowblock["cohort_risk_guard"]["decision"],
                    "logical_grouped_small_matrix_solve_count": int(
                        rowblock["small_matrix_solve_count"]
                    ),
                },
                "W_curation": {
                    "workflow_id": "W_curation",
                    "support_gain": float(
                        curation_b1["selected_support_exact_gain"]
                    ),
                    "query_gain": float(curation_b1["fixed_query_exact_gain"]),
                    "cached_decision": (
                        "ABSTAIN" if curation_b1["abstained"] else "EXECUTE"
                    ),
                    "logical_grouped_small_matrix_solve_count": 1,
                },
            }

    dataset_evidence: list[dict[str, object]] = []
    for heldout_dataset_id in completed_episodes:
        source_dataset_ids = sorted(
            dataset_id
            for dataset_id in completed_episodes
            if dataset_id != heldout_dataset_id
        )
        source_dataset_macro_outcome = {
            source_dataset_id: {
                workflow_id: statistics.fmean(
                    float(
                        completed_episodes[source_dataset_id][direction][workflow_id][
                            "query_gain"
                        ]
                    )
                    for direction in ("a_to_b", "b_to_a")
                )
                for workflow_id in P1B_WORKFLOW_IDS
            }
            for source_dataset_id in source_dataset_ids
        }
        source_scores = {
            workflow_id: statistics.fmean(
                source_dataset_macro_outcome[source_dataset_id][workflow_id]
                for source_dataset_id in source_dataset_ids
            )
            for workflow_id in P1B_WORKFLOW_IDS
        }
        if source_scores["W_rowblock"] >= source_scores["W_curation"]:
            a5_order = ("W_rowblock", "W_curation")
        else:
            a5_order = ("W_curation", "W_rowblock")
        source_score_margin = abs(
            source_scores["W_rowblock"] - source_scores["W_curation"]
        )

        direction_evidence: list[dict[str, object]] = []
        for direction in ("a_to_b", "b_to_a"):
            workflows = completed_episodes[heldout_dataset_id][direction]
            order_curves = {
                "rowblock_first": _workflow_curve(workflows, P1B_ORDERS[0]),
                "curation_first": _workflow_curve(workflows, P1B_ORDERS[1]),
            }
            a3_curve = _average_workflow_curves(list(order_curves.values()))
            a5_curve = _workflow_curve(workflows, a5_order)
            order_auc = {
                name: _p1b_auc(curve) for name, curve in order_curves.items()
            }
            oracle_order_name = (
                "rowblock_first"
                if order_auc["rowblock_first"] >= order_auc["curation_first"]
                else "curation_first"
            )
            oracle_order = (
                P1B_ORDERS[0]
                if oracle_order_name == "rowblock_first"
                else P1B_ORDERS[1]
            )
            oracle_curve = _workflow_curve(workflows, oracle_order)
            a3_auc = _p1b_auc(a3_curve)
            a5_auc = _p1b_auc(a5_curve)
            oracle_auc = _p1b_auc(oracle_curve)
            expected_curve_differs = any(
                abs(
                    float(a3_point["fixed_query_exact_gain"])
                    - float(a5_point["fixed_query_gain"])
                )
                > 1e-15
                for a3_point, a5_point in zip(a3_curve, a5_curve)
            )
            a3_expected_harm = any(
                float(point["fixed_query_exact_gain"]) < 0.0
                for point in a3_curve
                if int(point["budget"]) > 0
            )
            a3_any_order_harm = any(
                float(point["fixed_query_gain"]) < 0.0
                for curve in order_curves.values()
                for point in curve
                if int(point["budget"]) > 0
            )
            a5_harm = any(
                float(point["fixed_query_gain"]) < 0.0
                for point in a5_curve
                if int(point["budget"]) > 0
            )
            direction_evidence.append(
                {
                    "direction": direction,
                    "workflow_episodes": workflows,
                    "orders": {
                        "A3_exact_average_over": [list(order) for order in P1B_ORDERS],
                        "A5_source_order": list(a5_order),
                        "Oracle_evaluator_only_order": list(oracle_order),
                    },
                    "source_order_evidence": {
                        "source_dataset_ids": source_dataset_ids,
                        "dataset_macro_query_outcome": source_dataset_macro_outcome,
                        "dataset_equal_source_scores": source_scores,
                        "absolute_score_margin": source_score_margin,
                        "heldout_dataset_context_or_outcome_used": False,
                    },
                    "curves": {
                        "A3_order_specific": order_curves,
                        "A3_exact_average": a3_curve,
                        "A5_source_order": a5_curve,
                        "Oracle_evaluator_only": oracle_curve,
                    },
                    "adapt_auc": {
                        "A3_exact_average": a3_auc,
                        "A5_source_order": a5_auc,
                        "Oracle_evaluator_only": oracle_auc,
                        "A3_order_specific": order_auc,
                    },
                    "a5_minus_a3_adapt_auc": a5_auc - a3_auc,
                    "behavior_difference": {
                        "source_score_margin_nonzero": source_score_margin > 0.0,
                        "a5_curve_differs_from_a3_expectation": expected_curve_differs,
                        "source_driven_behavior": (
                            source_score_margin > 0.0 and expected_curve_differs
                        ),
                        "a5_matches_oracle_order": a5_order == oracle_order,
                    },
                    "harm": {
                        "A3_expected_curve_harmful": a3_expected_harm,
                        "A3_any_legal_order_harmful": a3_any_order_harm,
                        "A5_source_order_harmful": a5_harm,
                        "A3_harm_probability_by_budget": {
                            str(point["budget"]): point["harm_probability"]
                            for point in a3_curve
                        },
                    },
                }
            )

        dataset_auc = {
            arm: statistics.fmean(
                float(row["adapt_auc"][arm]) for row in direction_evidence
            )
            for arm in (
                "A3_exact_average",
                "A5_source_order",
                "Oracle_evaluator_only",
            )
        }
        dataset_evidence.append(
            {
                "heldout_dataset_id": heldout_dataset_id,
                "source_dataset_ids": source_dataset_ids,
                "source_workflow_order": list(a5_order),
                "source_workflow_scores": source_scores,
                "direction_evidence": direction_evidence,
                "summary": {
                    "adapt_auc": dataset_auc,
                    "a5_minus_a3_adapt_auc": (
                        dataset_auc["A5_source_order"]
                        - dataset_auc["A3_exact_average"]
                    ),
                    "harmful_direction_count": {
                        "A3_expected_curve": sum(
                            bool(row["harm"]["A3_expected_curve_harmful"])
                            for row in direction_evidence
                        ),
                        "A3_any_legal_order": sum(
                            bool(row["harm"]["A3_any_legal_order_harmful"])
                            for row in direction_evidence
                        ),
                        "A5_source_order": sum(
                            bool(row["harm"]["A5_source_order_harmful"])
                            for row in direction_evidence
                        ),
                    },
                    "source_driven_behavior_direction_count": sum(
                        bool(row["behavior_difference"]["source_driven_behavior"])
                        for row in direction_evidence
                    ),
                },
            }
        )

    macro_auc = {
        arm: statistics.fmean(
            float(row["summary"]["adapt_auc"][arm]) for row in dataset_evidence
        )
        for arm in (
            "A3_exact_average",
            "A5_source_order",
            "Oracle_evaluator_only",
        )
    }
    a5_minus_a3 = macro_auc["A5_source_order"] - macro_auc["A3_exact_average"]
    nonnegative_dataset_count = sum(
        float(row["summary"]["a5_minus_a3_adapt_auc"]) >= 0.0
        for row in dataset_evidence
    )
    harmful_direction_count = {
        arm: sum(
            int(row["summary"]["harmful_direction_count"][arm])
            for row in dataset_evidence
        )
        for arm in (
            "A3_expected_curve",
            "A3_any_legal_order",
            "A5_source_order",
        )
    }
    source_driven_behavior_count = sum(
        int(row["summary"]["source_driven_behavior_direction_count"])
        for row in dataset_evidence
    )
    distinct_a5_first_workflows = sorted(
        {str(row["source_workflow_order"][0]) for row in dataset_evidence}
    )
    gate = {
        "macro_a5_minus_a3_adapt_auc": a5_minus_a3,
        "macro_adapt_auc_improved": a5_minus_a3 > 0.0,
        "minimum_nonnegative_heldout_datasets": 3,
        "observed_nonnegative_heldout_datasets": nonnegative_dataset_count,
        "heldout_dataset_coverage_passed": nonnegative_dataset_count >= 3,
        "a3_any_legal_order_harmful_direction_count": harmful_direction_count[
            "A3_any_legal_order"
        ],
        "a5_harmful_direction_count": harmful_direction_count["A5_source_order"],
        "harm_not_increased": (
            harmful_direction_count["A5_source_order"]
            <= harmful_direction_count["A3_any_legal_order"]
        ),
        "source_driven_behavior_direction_count": source_driven_behavior_count,
        "source_driven_behavior_nontrivial": source_driven_behavior_count > 0,
    }
    gate_passed = all(
        (
            gate["macro_adapt_auc_improved"],
            gate["heldout_dataset_coverage_passed"],
            gate["harm_not_increased"],
            gate["source_driven_behavior_nontrivial"],
        )
    )
    if not gate["source_driven_behavior_nontrivial"]:
        first_fault = "SOURCE_HISTORY_DID_NOT_CHANGE_EXPECTED_WORKFLOW_BEHAVIOR"
    elif not gate["macro_adapt_auc_improved"]:
        first_fault = "SOURCE_WORKFLOW_ORDER_DID_NOT_BEAT_ORDER_UNINFORMED_BASELINE"
    elif not gate["heldout_dataset_coverage_passed"]:
        first_fault = "SOURCE_WORKFLOW_ORDER_NEGATIVE_ACROSS_HELDOUT_DATASETS"
    elif not gate["harm_not_increased"]:
        first_fault = "SOURCE_WORKFLOW_ORDER_INCREASED_HARM"
    elif len(distinct_a5_first_workflows) == 1:
        first_fault = "SOURCE_PRIOR_COLLAPSES_TO_UNIVERSAL_WORKFLOW_ORDER"
    else:
        first_fault = "FRESH_CONTEXT_CONDITIONED_WORKFLOW_SELECTION_UNTESTED"
    verdict = (
        "P1B_WORKFLOW_SUPPLY_PREMISE_PASS"
        if gate_passed
        else "P1B_WORKFLOW_SUPPLY_PREMISE_CLOSED"
    )
    rowblock_compute = rowblock_report["compute_accounting"]
    curation_compute = curation_report["compute"]
    return {
        "schema_version": P1B_SCHEMA_VERSION,
        "experiment_id": "E2-cross-series-curation-P1b",
        "stage": "development_exposed_cached_workflow_supply_replay",
        "scientific_question": (
            "Can completed Source PolicyEpisodes prioritize an entire data-preparation "
            "Workflow more effectively than an order-uninformed Target-only baseline?"
        ),
        "cached_dependencies": {
            "rowblock_report": ROWBLOCK_REPORT_PATH,
            "curation_report": P1_REPORT_PATH,
            "both_outcome_exposed": True,
            "fresh_or_uci_query_opened": False,
        },
        "protocol": {
            "workflow_supply": {
                "W_rowblock": (
                    "the cached proxy-guided complete four-block group Workflow"
                ),
                "W_curation": (
                    "the cached curation A3 Target-proxy B=1 complete donor Workflow"
                ),
            },
            "budgets": list(P1_BUDGETS),
            "feedback_unit": (
                "one complete Workflow's exact Target Support policy outcome"
            ),
            "execution_rule": (
                "among probed Workflows execute the maximum exact Support gain iff >0; "
                "otherwise Identity"
            ),
            "A3": "exact average over both legal two-Workflow probe orders",
            "A5": (
                "LODO dataset-equal Source macro Query outcome orders Workflows; tie "
                "is W_rowblock; heldout Context and outcome are forbidden"
            ),
            "Oracle": (
                "enumerate both heldout direction orders and choose higher AdaptAUC; "
                "evaluator-only upper bound"
            ),
            "adapt_auc": "normalized trapezoidal area over B={0,1,2}",
        },
        "dataset_evidence": dataset_evidence,
        "dataset_macro": {
            "adapt_auc": macro_auc,
            "a5_minus_a3_adapt_auc": a5_minus_a3,
            "nonnegative_a5_minus_a3_dataset_count": nonnegative_dataset_count,
            "harmful_direction_count": harmful_direction_count,
            "source_driven_behavior_direction_count": source_driven_behavior_count,
            "distinct_a5_first_workflows": distinct_a5_first_workflows,
        },
        "compute_accounting": {
            "p1b_cache_replay_only": True,
            "p1b_new_ridge_reference_solve_count": 0,
            "p1b_new_grouped_small_matrix_solve_count": 0,
            "p1b_new_consumer_refit_count": 0,
            "cached_rowblock_source_compute": {
                "reference_solve_count": rowblock_compute["reference_solve_count"],
                "grouped_small_matrix_solve_count": rowblock_compute[
                    "grouped_small_matrix_solve_count"
                ],
                "per_action_refit_count": rowblock_compute["per_action_refit_count"],
            },
            "cached_curation_source_compute": {
                "ridge_reference_solve_count": curation_compute[
                    "ridge_reference_solve_count"
                ],
                "grouped_small_matrix_solve_count": curation_compute[
                    "grouped_small_matrix_solve_count"
                ],
                "per_action_consumer_refit_count": curation_compute[
                    "per_action_consumer_refit_count"
                ],
            },
            "logical_internal_cost_per_probed_workflow": {
                "W_rowblock": "4 grouped small-matrix solves, one per output block",
                "W_curation": "1 grouped 6-row donor small-matrix solve after ordering",
            },
            "downstream_feedback_units_equal": True,
            "feedback_cost_note": (
                "internal attribution cost differs, but both consume exactly one budget "
                "unit only when a complete Support policy outcome is opened"
            ),
        },
        "fairness_risks": [
            (
                "A3 is an exact expectation over two random legal orders, while A5 is "
                "one deterministic Source order; order-level harm probabilities are "
                "reported so expectation does not hide exposure."
            ),
            (
                "W_rowblock has higher internal grouped-algebra cost than W_curation; "
                "the comparison equalizes downstream Support feedback, not compute."
            ),
            (
                "Both Workflows and all outcomes are already exposed development cache; "
                "this replay cannot establish fresh generalization or transfer."
            ),
        ],
        "frozen_gate": gate,
        "gate_passed": gate_passed,
        "first_fault": first_fault,
        "verdict": verdict,
        "evidence_boundary": {
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "development_only": True,
            "fresh_evidence": False,
            "target_query_opened": False,
            "capability_promoted": False,
            "transfer_tested": False,
            "memory_used": False,
            "classification": "MECHANISM" if gate_passed else "NEGATIVE",
            "claim": (
                "P1b tests only the cached Workflow-supply ordering premise. A PASS "
                "does not establish Context-conditioned Capability, Transfer, fresh "
                "evidence, or a deployable fixed Workflow rule."
            ),
        },
    }


def run_workflow_replay(root: Path) -> dict[str, object]:
    """Replay exposed complete-Workflow outcomes as lightweight PolicyEpisodes."""

    source = _read_object(root / P1B_REPORT_PATH)
    source_orders = [tuple(row["source_workflow_order"]) for row in source.get("dataset_evidence", [])]
    if (
        source.get("verdict") != "P1B_WORKFLOW_SUPPLY_PREMISE_PASS"
        or len(source_orders) != 4
        or set(source_orders) != {P1B_ORDERS[0]}
    ):
        raise ValueError("P1b does not validate the deployable rowblock-first prior")
    source_prior = P1B_ORDERS[0]
    report_paths = {
        "gefcom2012": "artifacts/functional/e2/cross_series_curation_gefcom2012_target_pilot_report.json",
        "weather_opsd": "artifacts/functional/e2/cross_series_workflow_proxy_target_confirmation_report.json",
        "solar": "artifacts/functional/e2/cross_series_workflow_solar_target_confirmation_report.json",
        "m4_daily": "artifacts/functional/e2/cross_series_workflow_m4_daily_target_confirmation_report.json",
    }
    inputs = {key: _read_object(root / path) for key, path in report_paths.items()}
    g12 = inputs["gefcom2012"]
    weather_opsd = inputs["weather_opsd"]
    solar, m4 = inputs["solar"], inputs["m4_daily"]
    g12_boundary = g12["evidence_boundary"]
    shared_boundary = weather_opsd["evidence_boundary"]
    query_cache_checks = (
        g12_boundary.get("outcome_exposure") == "EXPOSED_ONCE_FOR_EVALUATION",
        g12_boundary.get("query_future_materialization_count") == 1,
        shared_boundary.get("query_used_only_after_binding_and_order_freeze") is True,
        shared_boundary.get("query_logical_materialization_count_per_target") == 1,
        all(row.get("query_future_logical_materialization_count") == 1 for row in weather_opsd.get("targets", [])),
        solar["evidence_boundary"].get("outcome_exposure") == "EXPOSED_ONCE_FOR_EVALUATION",
        solar["evidence_boundary"].get("query_future_logical_materialization_count") == 1,
        m4["evidence_boundary"].get("outcome_exposure") == "EXPOSED_ONCE_FOR_EVALUATION",
        m4["evidence_boundary"].get("query_future_logical_materialization_count") == 1,
    )
    if not all(query_cache_checks):
        raise ValueError("a Target Query outcome is not a historical exposed cache")

    episodes: list[dict[str, object]] = []

    def compile_episode(
        *,
        target: str,
        frequency: str,
        period: int,
        cohort_counts: dict[str, int],
        workflows_raw: dict[str, object],
        historical_auc: dict[str, float],
        diagnostic_orders: dict[str, object],
    ) -> None:
        workflows = {
            workflow_id: {
                "workflow_id": workflow_id,
                "support_gain": float(workflows_raw[workflow_id]["support_gain"]),
                "query_gain": float(workflows_raw[workflow_id]["query_gain"]),
            }
            for workflow_id in P1B_WORKFLOW_IDS
        }
        h0_curve = _workflow_curve(workflows, source_prior)
        order_curves = [_workflow_curve(workflows, order) for order in P1B_ORDERS]
        a3_curve = _average_workflow_curves(order_curves)
        h0_auc, a3_auc = _p1b_auc(h0_curve), _p1b_auc(a3_curve)
        if (
            abs(h0_auc - float(historical_auc["H0"])) > 1e-10
            or abs(a3_auc - float(historical_auc["A3"])) > 1e-10
        ):
            raise AssertionError(f"historical Workflow AUC replay mismatch: {target}")
        b1, b2 = h0_curve[1], h0_curve[2]
        contraindicated = [
            workflow_id
            for workflow_id, row in workflows.items()
            if float(row["support_gain"]) <= 0.0
        ]
        revised = bool(b1["abstained"] and not b2["abstained"])
        episodes.append(
            {
                "target": target,
                "frequency": frequency,
                "period": period,
                "cohort_counts": cohort_counts,
                "task_consumer": {
                    "task": "48-step forecasting; mean per-series sMASE gain",
                    "consumer": "fixed Ridge with unpenalized intercept",
                },
                "candidate_workflows": list(P1B_WORKFLOW_IDS),
                "source_prior_order": list(source_prior),
                "workflow_outcomes": workflows,
                "replay": {
                    "budgets": list(P1_BUDGETS),
                    "H0_source_prior": h0_curve,
                    "A3_exact_order_average": a3_curve,
                    "adapt_auc": {"H0": h0_auc, "A3": a3_auc},
                },
                "support_confirmation": {
                    "contraindicated_workflows": contraindicated,
                    "B1_action": (
                        "ABSTAIN" if b1["abstained"] else "EXECUTE"
                    ),
                    "B2_action": (
                        "ABSTAIN" if b2["abstained"] else "EXECUTE"
                    ),
                    "revised_after_contraindication": revised,
                    "final_selected_workflow": b2["selected_workflow"],
                },
                "final_complete_query_utility_B2": float(b2["fixed_query_gain"]),
                "diagnostic_metadata_not_deployable": diagnostic_orders,
                "policy_episode_recorded": True,
            }
        )

    g12_roles = g12["target"]["roles"]
    compile_episode(
        target="gefcom2012_load",
        frequency="hourly",
        period=int(g12["target"]["period"]),
        cohort_counts={key: len(g12_roles[key]) for key in ("train", "support", "query")},
        workflows_raw=g12["workflow_episodes"],
        historical_auc={
            "H0": g12["adapt_auc"]["A5_source_order"],
            "A3": g12["adapt_auc"]["A3_exact_average"],
        },
        diagnostic_orders={},
    )
    for row in weather_opsd["targets"]:
        compile_episode(
            target=str(row["dataset_id"]),
            frequency="hourly",
            period=24,
            cohort_counts={"train": 12, "support": 4, "query": 4},
            workflows_raw=row["workflow_outcomes"],
            historical_auc={"H0": row["adapt_auc"]["H0"], "A3": row["adapt_auc"]["A3"]},
            diagnostic_orders={"H1": row["orders"].get("H1")},
        )
    for target, frequency, cached in (
        ("solar_10_minutes", "10_minutes", solar),
        ("m4_daily", "daily", m4),
    ):
        geometry = cached["geometry"]
        compile_episode(
            target=target,
            frequency=frequency,
            period=int(geometry["period"]),
            cohort_counts={
                key: int(geometry[f"{key}_series"])
                for key in ("train", "support", "query")
            },
            workflows_raw=cached["workflow_outcomes"],
            historical_auc={"H0": cached["adapt_auc"]["H0"], "A3": cached["adapt_auc"]["A3"]},
            diagnostic_orders={
                key: cached["orders"].get(key) for key in ("H1", "H2", "H3")
            },
        )
    if len(episodes) != 5:
        raise AssertionError("workflow replay requires exactly five Target episodes")
    macro_auc = {
        arm: statistics.fmean(row["replay"]["adapt_auc"][arm] for row in episodes)
        for arm in ("H0", "A3")
    }
    negative_absolute_utility_targets = {
        arm: sum(float(row["replay"]["adapt_auc"][arm]) < 0.0 for row in episodes)
        for arm in ("H0", "A3")
    }
    h0_below_a3_targets = [
        str(row["target"])
        for row in episodes
        if float(row["replay"]["adapt_auc"]["H0"])
        < float(row["replay"]["adapt_auc"]["A3"])
    ]
    b2_utilities = [float(row["final_complete_query_utility_B2"]) for row in episodes]
    return {
        "experiment_id": "E2.42-cross-series-workflow-harness-replay",
        "stage": "historical_exposed_policy_episode_replay",
        "policy_flow": (
            "SOURCE WORKFLOW SUPPLY/PRIOR -> TARGET EXACT SUPPORT CONFIRM -> "
            "EXECUTE/ABSTAIN/REVISE -> RECORD POLICY EPISODE"
        ),
        "deployable_prior": {
            "order": list(source_prior),
            "evidence": P1B_REPORT_PATH,
            "H1_H2_H3_are_diagnostic_only": True,
        },
        "policy_episodes": episodes,
        "summary": {
            "target_count": len(episodes),
            "macro_adapt_auc": macro_auc,
            "H0_minus_A3_macro_adapt_auc": macro_auc["H0"] - macro_auc["A3"],
            "negative_absolute_utility_target_count": negative_absolute_utility_targets,
            "H0_below_A3_target_count": len(h0_below_a3_targets),
            "H0_below_A3_target_ids": h0_below_a3_targets,
            "B2_final_query_utility": {
                "per_target": b2_utilities,
                "macro": statistics.fmean(b2_utilities),
            },
            "support_confirmation_behavior": {
                "revision_definition": "B1 abstain -> B2 execute",
                "B1_abstention_target_count": sum(
                    row["support_confirmation"]["B1_action"] == "ABSTAIN"
                    for row in episodes
                ),
                "B2_abstention_target_count": sum(
                    row["support_confirmation"]["B2_action"] == "ABSTAIN"
                    for row in episodes
                ),
                "revised_after_contraindication_target_count": sum(
                    bool(row["support_confirmation"]["revised_after_contraindication"])
                    for row in episodes
                ),
                "selected_workflow_changed_target_count": sum(
                    row["replay"]["H0_source_prior"][1]["selected_workflow"]
                    != row["replay"]["H0_source_prior"][2]["selected_workflow"]
                    for row in episodes
                ),
            },
        },
        "integrity_checks": {
            "historical_H0_A3_auc_recomputed_within_1e-10": True,
            "all_query_outcomes_historically_exposed": True,
            "new_query_reads": 0,
            "new_consumer_fits": 0,
            "raw_series_values_recorded": False,
        },
        "verdict": "WORKFLOW_HARNESS_VERTICAL_SLICE_REPLAYED",
        "claim_limit": (
            "historical exposed vertical-slice replay only; not Capability Promotion, "
            "Transfer confirmation, Memory, or fresh evidence"
        ),
    }


def _read_semantic_tsf_panel(
    np: Any,
    *,
    archive_path: Path,
    member: str,
    required_stop: int,
) -> tuple[list[str], list[dict[str, object]]]:
    """Read public TSF metadata and only the prefix used by this exposed P0."""

    from zipfile import ZipFile

    attributes: list[str] = []
    rows: list[dict[str, object]] = []
    data_started = False
    with ZipFile(archive_path) as archive, archive.open(member) as stream:
        for raw in stream:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith("#"):
                continue
            lowered = line.lower()
            if not data_started and lowered.startswith("@attribute "):
                attributes.append(line.split()[1])
                continue
            if lowered == "@data":
                data_started = True
                continue
            if not data_started or line.startswith("@"):
                continue
            fields = line.split(":")
            if len(fields) != len(attributes) + 1:
                raise ValueError(f"unexpected TSF row geometry: {archive_path}")
            payload = fields[-1]
            values: list[float] = []
            token_start = 0
            for _ in range(required_stop):
                comma = payload.find(",", token_start)
                token_end = len(payload) if comma < 0 else comma
                values.append(float(payload[token_start:token_end]))
                if comma < 0:
                    break
                token_start = comma + 1
            if len(values) != required_stop:
                raise ValueError(
                    f"TSF row ended before the frozen cutoff: {fields[0]}"
                )
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError("without-missing TSF archive contains non-finite values")
            row: dict[str, object] = dict(zip(attributes, fields[:-1]))
            row["values"] = array
            rows.append(row)
    if not rows:
        raise ValueError(f"no TSF data rows: {archive_path}")
    return attributes, rows


def _semantic_family(dataset_id: str, role: str) -> str:
    if dataset_id == "kdd_cup_2018":
        return "particulate" if role in {"PM2.5", "PM10"} else "gaseous_pollutant"
    if role.startswith("price_"):
        return "price"
    if role.startswith("distance_"):
        return "distance"
    if role.startswith("surge_"):
        return "surge"
    if role == "api_calls":
        return "demand"
    return "weather"


def _exact_weighted_ridge_prediction(
    np: Any,
    *,
    x_train: Any,
    targets: Any,
    weights: Any,
    x_eval: Any,
) -> Any:
    """Frozen Ridge(alpha=1), with an unpenalized intercept and exact weights."""

    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    query = np.asarray(x_eval, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or sample_weights.shape != (x.shape[0],)
        or x.shape[0] != y.shape[0]
        or query.ndim != 2
        or query.shape[1] != x.shape[1]
        or bool(np.any(sample_weights <= 0.0))
    ):
        raise ValueError("invalid semantic-augmentation Ridge geometry")
    z_train = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    z_eval = np.column_stack((query, np.ones(query.shape[0], dtype=np.float64)))
    system = z_train.T @ (sample_weights[:, None] * z_train)
    system[:-1, :-1] += RIDGE_ALPHA * np.eye(x.shape[1], dtype=np.float64)
    rhs = z_train.T @ (sample_weights[:, None] * y)
    prediction = z_eval @ np.linalg.solve(system, rhs)
    if not np.isfinite(prediction).all():
        raise RuntimeError("semantic-augmentation Ridge prediction is non-finite")
    return prediction


def run_semantic_auxiliary_p0(root: Path) -> dict[str, object]:
    """Exposed headroom census for metadata-frozen semantic cohort augmentation."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        seasonal_scale,
        smase,
    )

    cells: list[dict[str, object]] = []
    target_summaries: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    reference_solve_count = 0
    candidate_solve_count = 0

    for spec in SEMANTIC_AUXILIARY_DATASETS:
        dataset_id = str(spec["dataset_id"])
        archive_path = Path(str(spec["archive"]))
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        future = tuple(int(value) for value in spec["future"])
        anchors = tuple(int(value) for value in spec["anchors"])
        required_stop = max(future[1], max(anchors) + HORIZON)
        attributes, rows = _read_semantic_tsf_panel(
            np,
            archive_path=archive_path,
            member=str(spec["member"]),
            required_stop=required_stop,
        )
        identity_fields = tuple(str(value) for value in spec["identity_fields"])
        identity_filter = {
            str(key): str(value)
            for key, value in dict(spec["identity_filter"]).items()
        }
        semantic_field = str(spec["semantic_field"])
        semantic_roles = tuple(str(value) for value in spec["semantic_roles"])
        required_fields = set(identity_fields) | set(identity_filter) | {semantic_field}
        if not required_fields.issubset(attributes):
            raise ValueError(f"public TSF metadata fields changed: {dataset_id}")

        by_identity: dict[tuple[str, ...], dict[str, dict[str, object]]] = {}
        identity_order: list[tuple[str, ...]] = []
        for row in rows:
            if any(str(row[key]) != value for key, value in identity_filter.items()):
                continue
            identity = tuple(str(row[key]) for key in identity_fields)
            role = str(row[semantic_field])
            if identity not in by_identity:
                by_identity[identity] = {}
                identity_order.append(identity)
            by_identity[identity][role] = row
        complete_identities = [
            identity
            for identity in identity_order
            if set(semantic_roles).issubset(by_identity[identity])
        ]
        if len(complete_identities) < 20:
            raise ValueError(f"fewer than 20 complete semantic entities: {dataset_id}")
        roster_identities = complete_identities[:20]
        train_identities = roster_identities[:12]
        support_identities = roster_identities[12:16]
        query_identities = roster_identities[16:20]
        if not (
            len(train_identities) == 12
            and len(support_identities) == 4
            and len(query_identities) == 4
        ):
            raise AssertionError("semantic P0 roster geometry changed")

        dataset_target_rows: list[dict[str, object]] = []
        for target_role_raw in spec["target_roles"]:
            target_role = str(target_role_raw)
            target_x: list[Any] = []
            target_y: list[Any] = []
            for anchor in anchors:
                for identity in train_identities:
                    raw = np.asarray(
                        by_identity[identity][target_role]["values"], dtype=np.float64
                    )
                    context = raw[anchor - CONTEXT_LENGTH : anchor]
                    target = raw[anchor : anchor + HORIZON]
                    center, scale, method = _center_scale(np, context)
                    if method == "scale_floor_fallback":
                        raise ValueError(
                            f"target training scale floor: {dataset_id}/{target_role}"
                        )
                    target_x.append(
                        np.concatenate(
                            ((context - center) / scale, np.zeros(CONTEXT_LENGTH))
                        )
                    )
                    target_y.append((target - center) / scale)
            x_target = np.asarray(target_x, dtype=np.float64)
            y_target = np.asarray(target_y, dtype=np.float64)

            eval_identities = support_identities + query_identities
            x_eval: list[Any] = []
            actual: list[Any] = []
            centers: list[float] = []
            scales: list[float] = []
            seasonal: list[float] = []
            train_stop = int(spec["train_stop"])
            for identity in eval_identities:
                raw = np.asarray(
                    by_identity[identity][target_role]["values"], dtype=np.float64
                )
                history = raw[:train_stop]
                context = history[-CONTEXT_LENGTH:]
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise ValueError(
                        f"target evaluation scale floor: {dataset_id}/{target_role}"
                    )
                scale_value = seasonal_scale(
                    history,
                    np.isfinite(history),
                    period=int(spec["period"]),
                    min_pairs=32,
                )
                x_eval.append(
                    np.concatenate(
                        ((context - center) / scale, np.zeros(CONTEXT_LENGTH))
                    )
                )
                actual.append(raw[slice(*future)])
                centers.append(center)
                scales.append(scale)
                seasonal.append(scale_value)
            x_eval_array = np.asarray(x_eval, dtype=np.float64)
            actual_array = np.asarray(actual, dtype=np.float64)
            centers_array = np.asarray(centers, dtype=np.float64)
            scales_array = np.asarray(scales, dtype=np.float64)

            def score_predictions(normalized: Any) -> Any:
                prediction = np.asarray(normalized, dtype=np.float64)
                original = prediction * scales_array[:, None] + centers_array[:, None]
                return np.asarray(
                    [
                        smase(
                            actual_array[index],
                            original[index],
                            scale=float(seasonal[index]),
                        )
                        for index in range(len(eval_identities))
                    ],
                    dtype=np.float64,
                )

            baseline_prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=x_target,
                targets=y_target,
                weights=np.ones(x_target.shape[0], dtype=np.float64),
                x_eval=x_eval_array,
            )
            reference_solve_count += 1
            baseline_losses = score_predictions(baseline_prediction)
            target_cells: list[dict[str, object]] = []
            for auxiliary_role in semantic_roles:
                if auxiliary_role == target_role:
                    continue
                auxiliary_x: list[Any] = []
                auxiliary_y: list[Any] = []
                ineligible_reason: str | None = None
                for anchor in anchors:
                    for identity in train_identities:
                        raw = np.asarray(
                            by_identity[identity][auxiliary_role]["values"],
                            dtype=np.float64,
                        )
                        context = raw[anchor - CONTEXT_LENGTH : anchor]
                        target = raw[anchor : anchor + HORIZON]
                        center, scale, method = _center_scale(np, context)
                        if method == "scale_floor_fallback":
                            ineligible_reason = (
                                "AUXILIARY_TRAINING_SCALE_FLOOR:"
                                f"{dataset_id}/{target_role}/{auxiliary_role}"
                            )
                            break
                        auxiliary_x.append(
                            np.concatenate(
                                ((context - center) / scale, np.zeros(CONTEXT_LENGTH))
                            )
                        )
                        auxiliary_y.append((target - center) / scale)
                    if ineligible_reason is not None:
                        break
                if ineligible_reason is not None:
                    cell = {
                        "dataset": dataset_id,
                        "target": target_role,
                        "target_semantic_family": _semantic_family(
                            dataset_id, target_role
                        ),
                        "auxiliary_group": auxiliary_role,
                        "auxiliary_semantic_family": _semantic_family(
                            dataset_id, auxiliary_role
                        ),
                        "matched_training_entity_count": len(train_identities),
                        "eligibility": "INELIGIBLE",
                        "ineligible_reason": ineligible_reason,
                        "programs": [],
                        "support_selected": None,
                        "query_oracle": None,
                        "identity_is_query_optimal": None,
                        "menu_headroom": None,
                        "contains_harmful_candidate": None,
                        "support_selected_is_harmful": None,
                    }
                    cells.append(cell)
                    target_cells.append(cell)
                    continue
                x_auxiliary = np.asarray(auxiliary_x, dtype=np.float64)
                y_auxiliary = np.asarray(auxiliary_y, dtype=np.float64)
                combined_x = np.vstack((x_target, x_auxiliary))
                combined_y = np.vstack((y_target, y_auxiliary))
                programs: list[dict[str, object]] = []
                for auxiliary_weight in SEMANTIC_AUXILIARY_WEIGHTS:
                    weights = np.concatenate(
                        (
                            np.ones(x_target.shape[0], dtype=np.float64),
                            np.full(
                                x_auxiliary.shape[0],
                                auxiliary_weight,
                                dtype=np.float64,
                            ),
                        )
                    )
                    prediction = _exact_weighted_ridge_prediction(
                        np,
                        x_train=combined_x,
                        targets=combined_y,
                        weights=weights,
                        x_eval=x_eval_array,
                    )
                    candidate_solve_count += 1
                    gain = baseline_losses - score_predictions(prediction)
                    programs.append(
                        {
                            "program": "ADD_AUXILIARY_SEMANTIC_GROUP",
                            "auxiliary_weight": auxiliary_weight,
                            "support_gain": float(np.mean(gain[:4])),
                            "query_gain": float(np.mean(gain[4:])),
                            "per_support_series_gain": [
                                float(value) for value in gain[:4]
                            ],
                            "per_query_series_gain": [
                                float(value) for value in gain[4:]
                            ],
                        }
                    )
                support_best = max(
                    programs,
                    key=lambda row: (
                        float(row["support_gain"]),
                        float(row["auxiliary_weight"]),
                    ),
                )
                support_selected = (
                    {
                        "program": "IDENTITY",
                        "auxiliary_weight": None,
                        "support_gain": 0.0,
                        "query_gain": 0.0,
                    }
                    if float(support_best["support_gain"]) <= 0.0
                    else {
                        key: support_best[key]
                        for key in (
                            "program",
                            "auxiliary_weight",
                            "support_gain",
                            "query_gain",
                        )
                    }
                )
                query_best = max(
                    programs,
                    key=lambda row: (
                        float(row["query_gain"]),
                        float(row["auxiliary_weight"]),
                    ),
                )
                identity_optimal = float(query_best["query_gain"]) <= 0.0
                query_oracle = (
                    {
                        "program": "IDENTITY",
                        "auxiliary_weight": None,
                        "query_gain": 0.0,
                    }
                    if identity_optimal
                    else {
                        "program": query_best["program"],
                        "auxiliary_weight": query_best["auxiliary_weight"],
                        "query_gain": query_best["query_gain"],
                    }
                )
                cell = {
                    "dataset": dataset_id,
                    "target": target_role,
                    "target_semantic_family": _semantic_family(
                        dataset_id, target_role
                    ),
                    "auxiliary_group": auxiliary_role,
                    "auxiliary_semantic_family": _semantic_family(
                        dataset_id, auxiliary_role
                    ),
                    "matched_training_entity_count": len(train_identities),
                    "eligibility": "ELIGIBLE",
                    "ineligible_reason": None,
                    "programs": programs,
                    "support_selected": support_selected,
                    "query_oracle": query_oracle,
                    "identity_is_query_optimal": identity_optimal,
                    "menu_headroom": float(query_oracle["query_gain"]),
                    "contains_harmful_candidate": any(
                        float(row["query_gain"]) < 0.0 for row in programs
                    ),
                    "support_selected_is_harmful": float(
                        support_selected["query_gain"]
                    )
                    < 0.0,
                }
                cells.append(cell)
                target_cells.append(cell)

            all_target_programs = [
                {
                    "auxiliary_group": cell["auxiliary_group"],
                    **program,
                }
                for cell in target_cells
                for program in cell["programs"]
            ]
            if not all_target_programs:
                raise ValueError(
                    f"no eligible auxiliary group: {dataset_id}/{target_role}"
                )
            best_query_program = max(
                all_target_programs,
                key=lambda row: (
                    float(row["query_gain"]),
                    str(row["auxiliary_group"]),
                    float(row["auxiliary_weight"]),
                ),
            )
            best_support_program = max(
                all_target_programs,
                key=lambda row: (
                    float(row["support_gain"]),
                    str(row["auxiliary_group"]),
                    float(row["auxiliary_weight"]),
                ),
            )
            summary = {
                "dataset": dataset_id,
                "target": target_role,
                "target_semantic_family": _semantic_family(dataset_id, target_role),
                "auxiliary_group_count": len(target_cells),
                "eligible_auxiliary_group_count": sum(
                    cell["eligibility"] == "ELIGIBLE" for cell in target_cells
                ),
                "ineligible_auxiliary_group_count": sum(
                    cell["eligibility"] == "INELIGIBLE" for cell in target_cells
                ),
                "positive_group_count": sum(
                    cell["eligibility"] == "ELIGIBLE"
                    and float(cell["menu_headroom"]) > 0.0
                    for cell in target_cells
                ),
                "negative_group_count": sum(
                    cell["eligibility"] == "ELIGIBLE"
                    and bool(cell["identity_is_query_optimal"])
                    for cell in target_cells
                ),
                "harmful_candidate_group_count": sum(
                    cell["eligibility"] == "ELIGIBLE"
                    and bool(cell["contains_harmful_candidate"])
                    for cell in target_cells
                ),
                "support_selected_harmful_group_count": sum(
                    cell["eligibility"] == "ELIGIBLE"
                    and bool(cell["support_selected_is_harmful"])
                    for cell in target_cells
                ),
                "menu_headroom": max(0.0, float(best_query_program["query_gain"])),
                "identity_is_menu_optimal": float(best_query_program["query_gain"])
                <= 0.0,
                "best_group": (
                    None
                    if float(best_query_program["query_gain"]) <= 0.0
                    else best_query_program["auxiliary_group"]
                ),
                "best_weight": (
                    None
                    if float(best_query_program["query_gain"]) <= 0.0
                    else best_query_program["auxiliary_weight"]
                ),
                "support_menu_selected_group": (
                    None
                    if float(best_support_program["support_gain"]) <= 0.0
                    else best_support_program["auxiliary_group"]
                ),
                "support_menu_selected_weight": (
                    None
                    if float(best_support_program["support_gain"]) <= 0.0
                    else best_support_program["auxiliary_weight"]
                ),
                "support_menu_selected_query_gain": (
                    0.0
                    if float(best_support_program["support_gain"]) <= 0.0
                    else float(best_support_program["query_gain"])
                ),
                "roster": {
                    "train": [
                        str(by_identity[identity][target_role]["series_name"])
                        for identity in train_identities
                    ],
                    "support": [
                        str(by_identity[identity][target_role]["series_name"])
                        for identity in support_identities
                    ],
                    "query": [
                        str(by_identity[identity][target_role]["series_name"])
                        for identity in query_identities
                    ],
                },
            }
            target_summaries.append(summary)
            dataset_target_rows.append(summary)

        dataset_summaries.append(
            {
                "dataset": dataset_id,
                "target_count": len(dataset_target_rows),
                "has_positive_auxiliary_group": any(
                    int(row["positive_group_count"]) > 0
                    for row in dataset_target_rows
                ),
                "positive_group_count": sum(
                    int(row["positive_group_count"]) for row in dataset_target_rows
                ),
                "negative_group_count": sum(
                    int(row["negative_group_count"]) for row in dataset_target_rows
                ),
                "ineligible_group_count": sum(
                    int(row["ineligible_auxiliary_group_count"])
                    for row in dataset_target_rows
                ),
                "best_group_weight_by_target": {
                    str(row["target"]): [row["best_group"], row["best_weight"]]
                    for row in dataset_target_rows
                },
                "within_dataset_target_heterogeneity": len(
                    {
                        (str(row["best_group"]), row["best_weight"])
                        for row in dataset_target_rows
                    }
                )
                > 1,
            }
        )

    global_best_signatures = {
        (str(row["best_group"]), row["best_weight"])
        for row in target_summaries
    }
    global_best_heterogeneity = len(global_best_signatures) > 1
    within_dataset_target_heterogeneity = any(
        bool(row["within_dataset_target_heterogeneity"])
        for row in dataset_summaries
    )
    both_datasets_have_headroom = all(
        bool(row["has_positive_auxiliary_group"]) for row in dataset_summaries
    )
    identity_optimal_cell_count = sum(
        cell["eligibility"] == "ELIGIBLE"
        and bool(cell["identity_is_query_optimal"])
        for cell in cells
    )
    support_selected_harmful_cell_count = sum(
        cell["eligibility"] == "ELIGIBLE"
        and bool(cell["support_selected_is_harmful"])
        for cell in cells
    )
    matched_risk_exists = (
        identity_optimal_cell_count > 0 or support_selected_harmful_cell_count > 0
    )
    every_augmentation_helps = all(
        float(program["query_gain"]) > 0.0
        for cell in cells
        for program in cell["programs"]
    )
    passed = (
        both_datasets_have_headroom
        and matched_risk_exists
        and within_dataset_target_heterogeneity
    )
    if passed:
        verdict = "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS"
    elif every_augmentation_helps:
        verdict = "FIXED_AUGMENTATION_NOT_CONTEXT_SKILL"
    elif global_best_heterogeneity and not within_dataset_target_heterogeneity:
        verdict = "DATASET_LEVEL_HETEROGENEITY_ONLY"
    else:
        verdict = "CLOSE_FAMILY"

    expected_candidates = sum(
        (len(tuple(spec["semantic_roles"])) - 1)
        * len(tuple(spec["target_roles"]))
        * len(SEMANTIC_AUXILIARY_WEIGHTS)
        for spec in SEMANTIC_AUXILIARY_DATASETS
    )
    if (
        len(target_summaries) != 4
        or reference_solve_count != 4
        or candidate_solve_count
        != len(SEMANTIC_AUXILIARY_WEIGHTS)
        * sum(cell["eligibility"] == "ELIGIBLE" for cell in cells)
        or candidate_solve_count > expected_candidates
        or any(int(row["target_count"]) < 2 for row in dataset_summaries)
    ):
        raise AssertionError("semantic auxiliary P0 smoke check failed")

    return {
        "experiment_id": "E2-semantic-auxiliary-group-augmentation-P0",
        "scientific_role": "exposed-development Program headroom and matched-risk census",
        "exposure": "EXPOSED_DEVELOPMENT",
        "protocol": {
            "incumbent": "target-variable-only training pool",
            "program": "add one complete metadata-defined auxiliary semantic group over the same 12 training entities",
            "fixed_auxiliary_weights": list(SEMANTIC_AUXILIARY_WEIGHTS),
            "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE; four-series Support and Query means",
            "geometry": "12 train entities + 4 Support + 4 Query; 192 context / 48 horizon",
            "metadata_selection": "public TSF fields only; no outcome-based group or roster selection",
            "original_uci_opened": False,
        },
        "cells": cells,
        "dataset_target_summaries": target_summaries,
        "dataset_summaries": dataset_summaries,
        "task_context_heterogeneity": {
            "within_dataset_target_role_heterogeneity": (
                within_dataset_target_heterogeneity
            ),
            "per_dataset": {
                str(row["dataset"]): bool(
                    row["within_dataset_target_heterogeneity"]
                )
                for row in dataset_summaries
            },
            "global_best_group_or_weight_not_constant_descriptive": (
                global_best_heterogeneity
            ),
            "global_distinct_best_group_weight_count_descriptive": len(
                global_best_signatures
            ),
            "best_group_weight_signatures": [
                [group, weight]
                for group, weight in sorted(
                    global_best_signatures,
                    key=lambda value: (value[0], str(value[1])),
                )
            ],
        },
        "matched_risk": {
            "identity_optimal_cell_count": identity_optimal_cell_count,
            "support_selected_harmful_cell_count": support_selected_harmful_cell_count,
            "exists": matched_risk_exists,
        },
        "compute_counts": {
            "ridge_reference_solve_count": reference_solve_count,
            "exact_weighted_candidate_solve_count": candidate_solve_count,
            "total_ridge_solve_count": reference_solve_count
            + candidate_solve_count,
            "proxy_fit_count": 0,
        },
        "gate": {
            "both_datasets_have_positive_group": both_datasets_have_headroom,
            "harmful_group_or_identity_optimal_cell_exists": matched_risk_exists,
            "within_dataset_target_role_heterogeneity": (
                within_dataset_target_heterogeneity
            ),
            "global_heterogeneity_descriptive_only": global_best_heterogeneity,
            "every_augmentation_candidate_helps": every_augmentation_helps,
            "passed": passed,
        },
        "verdict": verdict,
        "capability_or_memory_written": False,
        "claim_limit": (
            "exposed P0 only; exact Consumer headroom does not yet establish a "
            "retrievable Skill, unseen-Target transfer, or LLM behavior"
        ),
    }


def run_auxiliary_channel_binding_p0(root: Path) -> dict[str, object]:
    """Exposed headroom census for same-identity auxiliary feature binding."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        seasonal_scale,
        smase,
    )

    cells: list[dict[str, object]] = []
    target_summaries: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    baseline_solve_count = 0
    candidate_solve_count = 0

    for spec in SEMANTIC_AUXILIARY_DATASETS:
        dataset_id = str(spec["dataset_id"])
        archive_path = Path(str(spec["archive"]))
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        future = tuple(int(value) for value in spec["future"])
        anchors = tuple(int(value) for value in spec["anchors"])
        required_stop = max(future[1], max(anchors) + HORIZON)
        attributes, rows = _read_semantic_tsf_panel(
            np,
            archive_path=archive_path,
            member=str(spec["member"]),
            required_stop=required_stop,
        )
        identity_fields = tuple(str(value) for value in spec["identity_fields"])
        identity_filter = {
            str(key): str(value)
            for key, value in dict(spec["identity_filter"]).items()
        }
        semantic_field = str(spec["semantic_field"])
        semantic_roles = tuple(str(value) for value in spec["semantic_roles"])
        required_fields = set(identity_fields) | set(identity_filter) | {
            semantic_field
        }
        if not required_fields.issubset(attributes):
            raise ValueError(f"public TSF metadata fields changed: {dataset_id}")

        by_identity: dict[tuple[str, ...], dict[str, dict[str, object]]] = {}
        identity_order: list[tuple[str, ...]] = []
        for row in rows:
            if any(str(row[key]) != value for key, value in identity_filter.items()):
                continue
            identity = tuple(str(row[key]) for key in identity_fields)
            role = str(row[semantic_field])
            if identity not in by_identity:
                by_identity[identity] = {}
                identity_order.append(identity)
            by_identity[identity][role] = row
        complete_identities = [
            identity
            for identity in identity_order
            if set(semantic_roles).issubset(by_identity[identity])
        ]
        if len(complete_identities) < 20:
            raise ValueError(f"fewer than 20 complete semantic entities: {dataset_id}")
        roster_identities = complete_identities[:20]
        train_identities = roster_identities[:12]
        support_identities = roster_identities[12:16]
        query_identities = roster_identities[16:20]
        eval_identities = support_identities + query_identities
        if not (
            len(train_identities) == 12
            and len(support_identities) == 4
            and len(query_identities) == 4
        ):
            raise AssertionError("auxiliary-channel P0 roster geometry changed")

        dataset_target_rows: list[dict[str, object]] = []
        for target_role_raw in spec["target_roles"]:
            target_role = str(target_role_raw)
            target_train_contexts: list[Any] = []
            target_y: list[Any] = []
            for anchor in anchors:
                for identity in train_identities:
                    raw = np.asarray(
                        by_identity[identity][target_role]["values"], dtype=np.float64
                    )
                    context = raw[anchor - CONTEXT_LENGTH : anchor]
                    target = raw[anchor : anchor + HORIZON]
                    center, scale, method = _center_scale(np, context)
                    if method == "scale_floor_fallback":
                        raise ValueError(
                            f"target training scale floor: {dataset_id}/{target_role}"
                        )
                    target_train_contexts.append((context - center) / scale)
                    target_y.append((target - center) / scale)
            target_train_array = np.asarray(
                target_train_contexts, dtype=np.float64
            )
            y_target = np.asarray(target_y, dtype=np.float64)
            baseline_x_train = np.concatenate(
                (target_train_array, np.zeros_like(target_train_array)), axis=1
            )

            target_eval_contexts: list[Any] = []
            actual: list[Any] = []
            centers: list[float] = []
            scales: list[float] = []
            seasonal: list[float] = []
            train_stop = int(spec["train_stop"])
            for identity in eval_identities:
                raw = np.asarray(
                    by_identity[identity][target_role]["values"], dtype=np.float64
                )
                history = raw[:train_stop]
                context = history[-CONTEXT_LENGTH:]
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise ValueError(
                        f"target evaluation scale floor: {dataset_id}/{target_role}"
                    )
                target_eval_contexts.append((context - center) / scale)
                actual.append(raw[slice(*future)])
                centers.append(center)
                scales.append(scale)
                seasonal.append(
                    seasonal_scale(
                        history,
                        np.isfinite(history),
                        period=int(spec["period"]),
                        min_pairs=32,
                    )
                )
            target_eval_array = np.asarray(target_eval_contexts, dtype=np.float64)
            baseline_x_eval = np.concatenate(
                (target_eval_array, np.zeros_like(target_eval_array)), axis=1
            )
            actual_array = np.asarray(actual, dtype=np.float64)
            centers_array = np.asarray(centers, dtype=np.float64)
            scales_array = np.asarray(scales, dtype=np.float64)

            def score_predictions(normalized: Any) -> Any:
                prediction = np.asarray(normalized, dtype=np.float64)
                original = prediction * scales_array[:, None] + centers_array[:, None]
                return np.asarray(
                    [
                        smase(
                            actual_array[index],
                            original[index],
                            scale=float(seasonal[index]),
                        )
                        for index in range(len(eval_identities))
                    ],
                    dtype=np.float64,
                )

            baseline_prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=baseline_x_train,
                targets=y_target,
                weights=np.ones(baseline_x_train.shape[0], dtype=np.float64),
                x_eval=baseline_x_eval,
            )
            baseline_solve_count += 1
            baseline_losses = score_predictions(baseline_prediction)

            target_cells: list[dict[str, object]] = []
            for auxiliary_role in semantic_roles:
                if auxiliary_role == target_role:
                    continue
                auxiliary_train_contexts: list[Any] = []
                ineligible_reason: str | None = None
                for anchor in anchors:
                    for identity in train_identities:
                        auxiliary_raw = np.asarray(
                            by_identity[identity][auxiliary_role]["values"],
                            dtype=np.float64,
                        )
                        auxiliary_context = auxiliary_raw[
                            anchor - CONTEXT_LENGTH : anchor
                        ]
                        auxiliary_center, auxiliary_scale, method = _center_scale(
                            np, auxiliary_context
                        )
                        if method == "scale_floor_fallback":
                            ineligible_reason = (
                                "AUXILIARY_TRAINING_SCALE_FLOOR:"
                                f"{dataset_id}/{target_role}/{auxiliary_role}"
                            )
                            break
                        auxiliary_train_contexts.append(
                            (auxiliary_context - auxiliary_center) / auxiliary_scale
                        )
                    if ineligible_reason is not None:
                        break

                auxiliary_eval_contexts: list[Any] = []
                if ineligible_reason is None:
                    for identity in eval_identities:
                        auxiliary_raw = np.asarray(
                            by_identity[identity][auxiliary_role]["values"],
                            dtype=np.float64,
                        )
                        auxiliary_history = auxiliary_raw[:train_stop]
                        auxiliary_context = auxiliary_history[-CONTEXT_LENGTH:]
                        auxiliary_center, auxiliary_scale, method = _center_scale(
                            np, auxiliary_context
                        )
                        if method == "scale_floor_fallback":
                            ineligible_reason = (
                                "AUXILIARY_EVALUATION_SCALE_FLOOR:"
                                f"{dataset_id}/{target_role}/{auxiliary_role}"
                            )
                            break
                        auxiliary_eval_contexts.append(
                            (auxiliary_context - auxiliary_center) / auxiliary_scale
                        )

                if ineligible_reason is not None:
                    cell = {
                        "dataset": dataset_id,
                        "target": target_role,
                        "target_semantic_family": _semantic_family(
                            dataset_id, target_role
                        ),
                        "auxiliary_channel": auxiliary_role,
                        "auxiliary_semantic_family": _semantic_family(
                            dataset_id, auxiliary_role
                        ),
                        "program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
                        "eligibility": "INELIGIBLE",
                        "ineligible_reason": ineligible_reason,
                        "same_identity_same_anchor": True,
                        "candidate": None,
                        "support_selected": None,
                        "query_menu_oracle": None,
                        "identity_is_query_optimal": None,
                        "candidate_is_harmful": None,
                    }
                    cells.append(cell)
                    target_cells.append(cell)
                    continue

                candidate_x_train = np.concatenate(
                    (
                        target_train_array,
                        np.asarray(auxiliary_train_contexts, dtype=np.float64),
                    ),
                    axis=1,
                )
                candidate_x_eval = np.concatenate(
                    (
                        target_eval_array,
                        np.asarray(auxiliary_eval_contexts, dtype=np.float64),
                    ),
                    axis=1,
                )
                if (
                    candidate_x_train.shape != baseline_x_train.shape
                    or candidate_x_eval.shape != baseline_x_eval.shape
                    or y_target.shape[0] != candidate_x_train.shape[0]
                ):
                    raise AssertionError("auxiliary-channel binding geometry changed")
                prediction = _exact_weighted_ridge_prediction(
                    np,
                    x_train=candidate_x_train,
                    targets=y_target,
                    weights=np.ones(candidate_x_train.shape[0], dtype=np.float64),
                    x_eval=candidate_x_eval,
                )
                candidate_solve_count += 1
                candidate_losses = score_predictions(prediction)
                gain = baseline_losses - candidate_losses
                support_gain = float(np.mean(gain[:4]))
                query_gain = float(np.mean(gain[4:]))
                support_selected = (
                    {
                        "program": "IDENTITY",
                        "auxiliary_channel": None,
                        "support_gain": 0.0,
                        "query_gain": 0.0,
                    }
                    if support_gain <= 0.0
                    else {
                        "program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
                        "auxiliary_channel": auxiliary_role,
                        "support_gain": support_gain,
                        "query_gain": query_gain,
                    }
                )
                identity_optimal = query_gain <= 0.0
                query_oracle = (
                    {
                        "program": "IDENTITY",
                        "auxiliary_channel": None,
                        "query_gain": 0.0,
                    }
                    if identity_optimal
                    else {
                        "program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
                        "auxiliary_channel": auxiliary_role,
                        "query_gain": query_gain,
                    }
                )
                cell = {
                    "dataset": dataset_id,
                    "target": target_role,
                    "target_semantic_family": _semantic_family(
                        dataset_id, target_role
                    ),
                    "auxiliary_channel": auxiliary_role,
                    "auxiliary_semantic_family": _semantic_family(
                        dataset_id, auxiliary_role
                    ),
                    "program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
                    "eligibility": "ELIGIBLE",
                    "ineligible_reason": None,
                    "same_identity_same_anchor": True,
                    "training_row_count": int(candidate_x_train.shape[0]),
                    "evaluation_row_count": int(candidate_x_eval.shape[0]),
                    "candidate": {
                        "support_gain": support_gain,
                        "query_gain": query_gain,
                        "per_support_series_gain": [
                            float(value) for value in gain[:4]
                        ],
                        "per_query_series_gain": [float(value) for value in gain[4:]],
                        "support_loss": float(np.mean(candidate_losses[:4])),
                        "query_loss": float(np.mean(candidate_losses[4:])),
                    },
                    "support_selected": support_selected,
                    "query_menu_oracle": query_oracle,
                    "identity_is_query_optimal": identity_optimal,
                    "candidate_is_harmful": query_gain < 0.0,
                }
                cells.append(cell)
                target_cells.append(cell)

            eligible_cells = [
                cell for cell in target_cells if cell["eligibility"] == "ELIGIBLE"
            ]
            if not eligible_cells:
                raise ValueError(
                    f"no eligible auxiliary channel: {dataset_id}/{target_role}"
                )
            best_support_cell = max(
                eligible_cells,
                key=lambda row: (
                    float(row["candidate"]["support_gain"]),
                    str(row["auxiliary_channel"]),
                ),
            )
            best_query_cell = max(
                eligible_cells,
                key=lambda row: (
                    float(row["candidate"]["query_gain"]),
                    str(row["auxiliary_channel"]),
                ),
            )
            best_support_gain = float(best_support_cell["candidate"]["support_gain"])
            best_query_gain = float(best_query_cell["candidate"]["query_gain"])
            summary = {
                "dataset": dataset_id,
                "target": target_role,
                "target_semantic_family": _semantic_family(dataset_id, target_role),
                "candidate_channel_count": len(target_cells),
                "eligible_candidate_channel_count": len(eligible_cells),
                "ineligible_candidate_channel_count": len(target_cells)
                - len(eligible_cells),
                "positive_candidate_channel_count": sum(
                    float(cell["candidate"]["query_gain"]) > 0.0
                    for cell in eligible_cells
                ),
                "identity_optimal_cell_count": sum(
                    bool(cell["identity_is_query_optimal"])
                    for cell in eligible_cells
                ),
                "harmful_candidate_channel_count": sum(
                    bool(cell["candidate_is_harmful"]) for cell in eligible_cells
                ),
                "baseline": {
                    "support_loss": float(np.mean(baseline_losses[:4])),
                    "query_loss": float(np.mean(baseline_losses[4:])),
                    "per_support_series_loss": [
                        float(value) for value in baseline_losses[:4]
                    ],
                    "per_query_series_loss": [
                        float(value) for value in baseline_losses[4:]
                    ],
                },
                "support_selected": (
                    {
                        "program": "IDENTITY",
                        "auxiliary_channel": None,
                        "support_gain": 0.0,
                        "query_gain": 0.0,
                    }
                    if best_support_gain <= 0.0
                    else {
                        "program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
                        "auxiliary_channel": best_support_cell["auxiliary_channel"],
                        "support_gain": best_support_gain,
                        "query_gain": float(
                            best_support_cell["candidate"]["query_gain"]
                        ),
                    }
                ),
                "query_menu_oracle": (
                    {
                        "program": "IDENTITY",
                        "auxiliary_channel": None,
                        "query_gain": 0.0,
                    }
                    if best_query_gain <= 0.0
                    else {
                        "program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
                        "auxiliary_channel": best_query_cell["auxiliary_channel"],
                        "query_gain": best_query_gain,
                    }
                ),
                "best_candidate_auxiliary_channel": best_query_cell[
                    "auxiliary_channel"
                ],
                "best_candidate_query_gain": best_query_gain,
                "roster": {
                    "train": [
                        str(by_identity[identity][target_role]["series_name"])
                        for identity in train_identities
                    ],
                    "support": [
                        str(by_identity[identity][target_role]["series_name"])
                        for identity in support_identities
                    ],
                    "query": [
                        str(by_identity[identity][target_role]["series_name"])
                        for identity in query_identities
                    ],
                },
            }
            target_summaries.append(summary)
            dataset_target_rows.append(summary)

        best_channels = {
            str(row["target"]): str(row["best_candidate_auxiliary_channel"])
            for row in dataset_target_rows
        }
        dataset_summaries.append(
            {
                "dataset": dataset_id,
                "target_count": len(dataset_target_rows),
                "has_positive_candidate": any(
                    int(row["positive_candidate_channel_count"]) > 0
                    for row in dataset_target_rows
                ),
                "positive_candidate_channel_count": sum(
                    int(row["positive_candidate_channel_count"])
                    for row in dataset_target_rows
                ),
                "identity_optimal_cell_count": sum(
                    int(row["identity_optimal_cell_count"])
                    for row in dataset_target_rows
                ),
                "harmful_candidate_channel_count": sum(
                    int(row["harmful_candidate_channel_count"])
                    for row in dataset_target_rows
                ),
                "ineligible_candidate_channel_count": sum(
                    int(row["ineligible_candidate_channel_count"])
                    for row in dataset_target_rows
                ),
                "best_auxiliary_channel_by_target": best_channels,
                "within_dataset_target_role_heterogeneity": len(
                    set(best_channels.values())
                )
                > 1,
            }
        )

    both_datasets_have_headroom = all(
        bool(row["has_positive_candidate"]) for row in dataset_summaries
    )
    identity_optimal_cell_count = sum(
        cell["eligibility"] == "ELIGIBLE"
        and bool(cell["identity_is_query_optimal"])
        for cell in cells
    )
    harmful_candidate_count = sum(
        cell["eligibility"] == "ELIGIBLE" and bool(cell["candidate_is_harmful"])
        for cell in cells
    )
    matched_risk_exists = identity_optimal_cell_count > 0 or harmful_candidate_count > 0
    within_dataset_heterogeneity = any(
        bool(row["within_dataset_target_role_heterogeneity"])
        for row in dataset_summaries
    )
    all_best_channels = {
        str(row["best_candidate_auxiliary_channel"]) for row in target_summaries
    }
    global_best_channel_heterogeneity = len(all_best_channels) > 1
    between_dataset_heterogeneity_only = (
        not within_dataset_heterogeneity
        and len(
            {
                next(iter(dict(row["best_auxiliary_channel_by_target"]).values()))
                for row in dataset_summaries
            }
        )
        > 1
    )
    passed = (
        both_datasets_have_headroom
        and matched_risk_exists
        and within_dataset_heterogeneity
    )
    if passed:
        verdict = "AUX_CHANNEL_BINDING_HEADROOM_AND_RISK_PASS"
    elif between_dataset_heterogeneity_only:
        verdict = "DATASET_LEVEL_HETEROGENEITY_ONLY"
    else:
        verdict = "CLOSE_AUX_CHANNEL_BINDING_FAMILY"

    eligible_candidate_count = sum(
        cell["eligibility"] == "ELIGIBLE" for cell in cells
    )
    enumerated_candidate_cell_count = sum(
        (len(tuple(spec["semantic_roles"])) - 1)
        * len(tuple(spec["target_roles"]))
        for spec in SEMANTIC_AUXILIARY_DATASETS
    )
    maximum_candidate_solve_count = 36
    if (
        len(target_summaries) != 4
        or baseline_solve_count != 4
        or candidate_solve_count != eligible_candidate_count
        or candidate_solve_count > maximum_candidate_solve_count
        or enumerated_candidate_cell_count != 38
        or any(int(row["target_count"]) != 2 for row in dataset_summaries)
    ):
        raise AssertionError("auxiliary-channel binding P0 smoke check failed")

    return {
        "experiment_id": "E2-auxiliary-channel-binding-P0",
        "scientific_role": (
            "exposed-development Program headroom and matched-risk census"
        ),
        "exposure": "EXPOSED_DEVELOPMENT",
        "protocol": {
            "incumbent_feature": (
                "[normalized target context, zeros(192)]; target normalized by "
                "the target context center and scale"
            ),
            "candidate_program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
            "candidate_feature": (
                "[normalized target context, normalized auxiliary context]"
            ),
            "binding": (
                "same metadata identity and same anchor/cutoff at train, Support, "
                "and Query"
            ),
            "auxiliary_future_read": False,
            "cross_identity_binding": False,
            "training_sample_or_weight_change": False,
            "candidate_parameters": [],
            "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE; four-series Support and Query means",
            "geometry": (
                "12 train entities + 4 Support + 4 Query; 192 target context + "
                "192 auxiliary context / 48 target horizon"
            ),
            "metadata_selection": (
                "public TSF fields only; all non-target roles enumerated"
            ),
            "original_uci_opened": False,
        },
        "cells": cells,
        "dataset_target_summaries": target_summaries,
        "dataset_summaries": dataset_summaries,
        "task_context_heterogeneity": {
            "within_dataset_target_role_heterogeneity": (
                within_dataset_heterogeneity
            ),
            "per_dataset": {
                str(row["dataset"]): bool(
                    row["within_dataset_target_role_heterogeneity"]
                )
                for row in dataset_summaries
            },
            "global_best_channel_heterogeneity_descriptive_only": (
                global_best_channel_heterogeneity
            ),
            "distinct_best_candidate_channels_descriptive": sorted(
                all_best_channels
            ),
            "between_dataset_heterogeneity_only": (
                between_dataset_heterogeneity_only
            ),
        },
        "matched_risk": {
            "identity_optimal_cell_count": identity_optimal_cell_count,
            "harmful_candidate_count": harmful_candidate_count,
            "exists": matched_risk_exists,
        },
        "compute_counts": {
            "ridge_baseline_solve_count": baseline_solve_count,
            "ridge_candidate_solve_count": candidate_solve_count,
            "eligible_candidate_count": eligible_candidate_count,
            "enumerated_candidate_cell_count": enumerated_candidate_cell_count,
            "maximum_candidate_solve_count": maximum_candidate_solve_count,
            "total_ridge_solve_count": baseline_solve_count
            + candidate_solve_count,
        },
        "gate": {
            "both_datasets_have_positive_candidate": both_datasets_have_headroom,
            "identity_optimal_cell_or_harmful_candidate_exists": (
                matched_risk_exists
            ),
            "within_dataset_target_role_best_channel_heterogeneity": (
                within_dataset_heterogeneity
            ),
            "global_distinct_best_channel_descriptive_only": (
                global_best_channel_heterogeneity
            ),
            "passed": passed,
        },
        "verdict": verdict,
        "capability_or_memory_written": False,
        "claim_limit": (
            "exposed feature-binding P0 only; exact Consumer headroom does not "
            "establish a learned Observation, Skill retrieval policy, unseen-Target "
            "transfer, Memory benefit, or LLM success"
        ),
    }


def run_auxiliary_channel_binding_llm_pilot(root: Path) -> dict[str, object]:
    """Run one frozen LLM probe plan on sealed target roles."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        seasonal_scale,
        smase,
    )

    target_roles_by_dataset = {
        "kdd_cup_2018": ("CO", "SO2"),
        "rideshare": ("api_calls", "temp"),
    }
    expected_target_keys = {
        (dataset_id, target_role)
        for dataset_id, target_roles in target_roles_by_dataset.items()
        for target_role in target_roles
    }
    plan_path = root / AUXILIARY_CHANNEL_BINDING_LLM_PLAN_PATH
    plan = _read_object(plan_path)
    if (
        plan.get("planner_id") != "llm_aux_channel_probe_v1"
        or plan.get("context_exposure") != "AGGREGATE_SEEN"
        or plan.get("outcome_exposure_at_plan_time") != "SEALED"
        or plan.get("program") != "BIND_AUXILIARY_CONTEXT_CHANNEL"
        or plan.get("fallback") != "IDENTITY"
        or plan.get("confirmation") != "support_gain>0"
    ):
        raise ValueError("auxiliary-channel LLM plan is not the frozen contract")
    planned_targets = {
        (str(row["dataset"]), str(row["target"])): row
        for row in plan.get("targets", [])
    }
    if set(planned_targets) != expected_target_keys:
        raise ValueError("auxiliary-channel LLM plan target supply changed")
    public_roles_by_dataset = {
        str(spec["dataset_id"]): tuple(str(role) for role in spec["semantic_roles"])
        for spec in SEMANTIC_AUXILIARY_DATASETS
    }
    for target_key, row in planned_targets.items():
        dataset_id, target_role = target_key
        probes = row.get("probe_order")
        if (
            not isinstance(probes, list)
            or len(probes) != 2
            or len({str(role) for role in probes}) != 2
            or not {str(role) for role in probes}.issubset(
                set(public_roles_by_dataset[dataset_id]) - {target_role}
            )
            or row.get("fallback") != "IDENTITY"
            or row.get("confirmation") != "support_gain>0"
        ):
            raise ValueError(f"invalid frozen auxiliary-channel plan: {target_key}")

    def budget_curve(
        order: list[str], actions: dict[str, dict[str, float]]
    ) -> list[dict[str, object]]:
        if len(order) < 2 or any(role not in actions for role in order[:2]):
            raise ValueError("budget-two channel order is not executable")
        curve: list[dict[str, object]] = [
            {
                "budget": 0,
                "probed_auxiliary_channel": None,
                "probed_support_gain": None,
                "probed_query_gain_evaluator_only": None,
                "selected_action": "IDENTITY",
                "selected_auxiliary_channel": None,
                "selected_support_gain": 0.0,
                "fixed_query_gain": 0.0,
                "abstained": True,
                "terminal": False,
            }
        ]
        winner: str | None = None
        for budget in (1, 2):
            probed_role: str | None = None
            probed: dict[str, float] | None = None
            if winner is None:
                probed_role = order[budget - 1]
                probed = actions[probed_role]
                if float(probed["support_gain"]) > 0.0:
                    winner = probed_role
            selected = None if winner is None else actions[winner]
            curve.append(
                {
                    "budget": budget,
                    "probed_auxiliary_channel": probed_role,
                    "probed_support_gain": (
                        None if probed is None else float(probed["support_gain"])
                    ),
                    "probed_query_gain_evaluator_only": (
                        None if probed is None else float(probed["query_gain"])
                    ),
                    "selected_action": (
                        "IDENTITY"
                        if winner is None
                        else "BIND_AUXILIARY_CONTEXT_CHANNEL"
                    ),
                    "selected_auxiliary_channel": winner,
                    "selected_support_gain": (
                        0.0 if selected is None else float(selected["support_gain"])
                    ),
                    "fixed_query_gain": (
                        0.0 if selected is None else float(selected["query_gain"])
                    ),
                    "abstained": winner is None,
                    "terminal": winner is not None,
                }
            )
        return curve

    target_results: list[dict[str, object]] = []
    baseline_solve_count = 0
    candidate_solve_count = 0
    enumerated_candidate_count = 0
    for spec in SEMANTIC_AUXILIARY_DATASETS:
        dataset_id = str(spec["dataset_id"])
        target_roles = target_roles_by_dataset[dataset_id]
        archive_path = Path(str(spec["archive"]))
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        future = tuple(int(value) for value in spec["future"])
        anchors = tuple(int(value) for value in spec["anchors"])
        required_stop = max(future[1], max(anchors) + HORIZON)
        attributes, rows = _read_semantic_tsf_panel(
            np,
            archive_path=archive_path,
            member=str(spec["member"]),
            required_stop=required_stop,
        )
        identity_fields = tuple(str(value) for value in spec["identity_fields"])
        identity_filter = {
            str(key): str(value)
            for key, value in dict(spec["identity_filter"]).items()
        }
        semantic_field = str(spec["semantic_field"])
        public_roles = tuple(str(value) for value in spec["semantic_roles"])
        if not (
            set(identity_fields) | set(identity_filter) | {semantic_field}
        ).issubset(attributes):
            raise ValueError(f"public TSF metadata fields changed: {dataset_id}")

        by_identity: dict[tuple[str, ...], dict[str, dict[str, object]]] = {}
        identity_order: list[tuple[str, ...]] = []
        for row in rows:
            if any(str(row[key]) != value for key, value in identity_filter.items()):
                continue
            identity = tuple(str(row[key]) for key in identity_fields)
            role = str(row[semantic_field])
            if identity not in by_identity:
                by_identity[identity] = {}
                identity_order.append(identity)
            by_identity[identity][role] = row
        complete_identities = [
            identity
            for identity in identity_order
            if set(public_roles).issubset(by_identity[identity])
        ]
        if len(complete_identities) < 20:
            raise ValueError(f"fewer than 20 complete semantic entities: {dataset_id}")
        roster_identities = complete_identities[:20]
        train_identities = roster_identities[:12]
        support_identities = roster_identities[12:16]
        query_identities = roster_identities[16:20]
        eval_identities = support_identities + query_identities
        if not (
            len(train_identities) == 12
            and len(support_identities) == 4
            and len(query_identities) == 4
        ):
            raise AssertionError("auxiliary-channel LLM roster geometry changed")

        for target_role in target_roles:
            target_train_contexts: list[Any] = []
            target_y: list[Any] = []
            for anchor in anchors:
                for identity in train_identities:
                    raw = np.asarray(
                        by_identity[identity][target_role]["values"], dtype=np.float64
                    )
                    context = raw[anchor - CONTEXT_LENGTH : anchor]
                    target = raw[anchor : anchor + HORIZON]
                    center, scale, method = _center_scale(np, context)
                    if method == "scale_floor_fallback":
                        raise ValueError(
                            f"target training scale floor: {dataset_id}/{target_role}"
                        )
                    target_train_contexts.append((context - center) / scale)
                    target_y.append((target - center) / scale)
            target_train_array = np.asarray(
                target_train_contexts, dtype=np.float64
            )
            y_target = np.asarray(target_y, dtype=np.float64)
            baseline_x_train = np.concatenate(
                (target_train_array, np.zeros_like(target_train_array)), axis=1
            )

            target_eval_contexts: list[Any] = []
            actual: list[Any] = []
            centers: list[float] = []
            scales: list[float] = []
            seasonal: list[float] = []
            train_stop = int(spec["train_stop"])
            for identity in eval_identities:
                raw = np.asarray(
                    by_identity[identity][target_role]["values"], dtype=np.float64
                )
                history = raw[:train_stop]
                context = history[-CONTEXT_LENGTH:]
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise ValueError(
                        f"target evaluation scale floor: {dataset_id}/{target_role}"
                    )
                target_eval_contexts.append((context - center) / scale)
                actual.append(raw[slice(*future)])
                centers.append(center)
                scales.append(scale)
                seasonal.append(
                    seasonal_scale(
                        history,
                        np.isfinite(history),
                        period=int(spec["period"]),
                        min_pairs=32,
                    )
                )
            target_eval_array = np.asarray(target_eval_contexts, dtype=np.float64)
            baseline_x_eval = np.concatenate(
                (target_eval_array, np.zeros_like(target_eval_array)), axis=1
            )
            actual_array = np.asarray(actual, dtype=np.float64)
            centers_array = np.asarray(centers, dtype=np.float64)
            scales_array = np.asarray(scales, dtype=np.float64)

            def score_predictions(normalized: Any) -> Any:
                prediction = np.asarray(normalized, dtype=np.float64)
                original = prediction * scales_array[:, None] + centers_array[:, None]
                return np.asarray(
                    [
                        smase(
                            actual_array[index],
                            original[index],
                            scale=float(seasonal[index]),
                        )
                        for index in range(len(eval_identities))
                    ],
                    dtype=np.float64,
                )

            baseline_prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=baseline_x_train,
                targets=y_target,
                weights=np.ones(baseline_x_train.shape[0], dtype=np.float64),
                x_eval=baseline_x_eval,
            )
            baseline_solve_count += 1
            baseline_losses = score_predictions(baseline_prediction)

            actions: dict[str, dict[str, float]] = {}
            candidate_menu: list[dict[str, object]] = []
            for auxiliary_role in public_roles:
                if auxiliary_role == target_role:
                    continue
                enumerated_candidate_count += 1
                auxiliary_train_contexts: list[Any] = []
                ineligible_reason: str | None = None
                for anchor in anchors:
                    for identity in train_identities:
                        auxiliary_raw = np.asarray(
                            by_identity[identity][auxiliary_role]["values"],
                            dtype=np.float64,
                        )
                        auxiliary_context = auxiliary_raw[
                            anchor - CONTEXT_LENGTH : anchor
                        ]
                        auxiliary_center, auxiliary_scale, method = _center_scale(
                            np, auxiliary_context
                        )
                        if method == "scale_floor_fallback":
                            ineligible_reason = (
                                "AUXILIARY_TRAINING_SCALE_FLOOR:"
                                f"{dataset_id}/{target_role}/{auxiliary_role}"
                            )
                            break
                        auxiliary_train_contexts.append(
                            (auxiliary_context - auxiliary_center) / auxiliary_scale
                        )
                    if ineligible_reason is not None:
                        break
                auxiliary_eval_contexts: list[Any] = []
                if ineligible_reason is None:
                    for identity in eval_identities:
                        auxiliary_raw = np.asarray(
                            by_identity[identity][auxiliary_role]["values"],
                            dtype=np.float64,
                        )
                        auxiliary_context = auxiliary_raw[
                            train_stop - CONTEXT_LENGTH : train_stop
                        ]
                        auxiliary_center, auxiliary_scale, method = _center_scale(
                            np, auxiliary_context
                        )
                        if method == "scale_floor_fallback":
                            ineligible_reason = (
                                "AUXILIARY_EVALUATION_SCALE_FLOOR:"
                                f"{dataset_id}/{target_role}/{auxiliary_role}"
                            )
                            break
                        auxiliary_eval_contexts.append(
                            (auxiliary_context - auxiliary_center) / auxiliary_scale
                        )
                if ineligible_reason is not None:
                    candidate_menu.append(
                        {
                            "auxiliary_channel": auxiliary_role,
                            "eligibility": "INELIGIBLE",
                            "ineligible_reason": ineligible_reason,
                            "support_gain": None,
                            "query_gain": None,
                            "per_support_series_gain": [],
                            "per_query_series_gain": [],
                        }
                    )
                    continue

                candidate_x_train = np.concatenate(
                    (
                        target_train_array,
                        np.asarray(auxiliary_train_contexts, dtype=np.float64),
                    ),
                    axis=1,
                )
                candidate_x_eval = np.concatenate(
                    (
                        target_eval_array,
                        np.asarray(auxiliary_eval_contexts, dtype=np.float64),
                    ),
                    axis=1,
                )
                if (
                    candidate_x_train.shape != baseline_x_train.shape
                    or candidate_x_eval.shape != baseline_x_eval.shape
                ):
                    raise AssertionError("auxiliary-channel LLM binding changed")
                prediction = _exact_weighted_ridge_prediction(
                    np,
                    x_train=candidate_x_train,
                    targets=y_target,
                    weights=np.ones(candidate_x_train.shape[0], dtype=np.float64),
                    x_eval=candidate_x_eval,
                )
                candidate_solve_count += 1
                gain = baseline_losses - score_predictions(prediction)
                action = {
                    "support_gain": float(np.mean(gain[:4])),
                    "query_gain": float(np.mean(gain[4:])),
                }
                actions[auxiliary_role] = action
                candidate_menu.append(
                    {
                        "auxiliary_channel": auxiliary_role,
                        "eligibility": "ELIGIBLE",
                        "ineligible_reason": None,
                        "support_gain": action["support_gain"],
                        "query_gain": action["query_gain"],
                        "per_support_series_gain": [
                            float(value) for value in gain[:4]
                        ],
                        "per_query_series_gain": [
                            float(value) for value in gain[4:]
                        ],
                    }
                )

            a3_order = [
                role
                for role in public_roles
                if role != target_role and role in actions
            ]
            a5_order = [
                str(role)
                for role in planned_targets[(dataset_id, target_role)][
                    "probe_order"
                ]
            ]
            if any(role not in actions for role in a5_order):
                raise ValueError(
                    f"frozen A5 probe is ineligible: {dataset_id}/{target_role}"
                )
            a3_curve = budget_curve(a3_order, actions)
            a5_curve = budget_curve(a5_order, actions)
            a3_auc = policy_episode_adapt_auc(a3_curve)
            a5_auc = policy_episode_adapt_auc(a5_curve)
            best_candidate = max(
                (row for row in candidate_menu if row["eligibility"] == "ELIGIBLE"),
                key=lambda row: (
                    float(row["query_gain"]),
                    str(row["auxiliary_channel"]),
                ),
            )
            oracle_identity = float(best_candidate["query_gain"]) <= 0.0
            top1_changed = a5_order[0] != a3_order[0]
            order_changed = a5_order != a3_order[:2]
            behavior_changed = [
                row["selected_auxiliary_channel"] for row in a5_curve
            ] != [row["selected_auxiliary_channel"] for row in a3_curve]
            target_results.append(
                {
                    "dataset": dataset_id,
                    "target": target_role,
                    "roster": {
                        "matched_identity_fields": list(identity_fields),
                        "train": [
                            str(by_identity[identity][target_role]["series_name"])
                            for identity in train_identities
                        ],
                        "support": [
                            str(by_identity[identity][target_role]["series_name"])
                            for identity in support_identities
                        ],
                        "query": [
                            str(by_identity[identity][target_role]["series_name"])
                            for identity in query_identities
                        ],
                    },
                    "candidate_menu_evaluator_only": candidate_menu,
                    "A3_order_public_candidate_order": a3_order,
                    "A5_order_frozen_llm_plan": a5_order,
                    "A3_curve": a3_curve,
                    "A5_curve": a5_curve,
                    "adapt_auc": {"A3": a3_auc, "A5": a5_auc},
                    "A5_minus_A3": a5_auc - a3_auc,
                    "A5_harmful": any(
                        float(row["fixed_query_gain"]) < 0.0
                        for row in a5_curve[1:]
                    ),
                    "A5_abstained_at_B2": bool(a5_curve[-1]["abstained"]),
                    "top1_changed": top1_changed,
                    "budget_two_order_changed": order_changed,
                    "selected_behavior_changed": behavior_changed,
                    "order_or_behavior_changed": order_changed or behavior_changed,
                    "full_menu_query_oracle_evaluator_only": {
                        "selected_action": (
                            "IDENTITY"
                            if oracle_identity
                            else "BIND_AUXILIARY_CONTEXT_CHANNEL"
                        ),
                        "auxiliary_channel": (
                            None
                            if oracle_identity
                            else best_candidate["auxiliary_channel"]
                        ),
                        "query_headroom": (
                            0.0
                            if oracle_identity
                            else float(best_candidate["query_gain"])
                        ),
                    },
                }
            )

    a5_macro = statistics.fmean(
        float(row["adapt_auc"]["A5"]) for row in target_results
    )
    a3_macro = statistics.fmean(
        float(row["adapt_auc"]["A3"]) for row in target_results
    )
    nonnegative_target_count = sum(
        float(row["A5_minus_A3"]) >= 0.0 for row in target_results
    )
    harmful_target_count = sum(bool(row["A5_harmful"]) for row in target_results)
    changed_target_count = sum(
        bool(row["order_or_behavior_changed"]) for row in target_results
    )
    passed = (
        a5_macro > a3_macro
        and nonnegative_target_count >= 3
        and harmful_target_count == 0
        and changed_target_count >= 1
    )
    eligible_candidate_count = sum(
        row["eligibility"] == "ELIGIBLE"
        for target in target_results
        for row in target["candidate_menu_evaluator_only"]
    )
    if (
        len(target_results) != 4
        or baseline_solve_count != 4
        or enumerated_candidate_count != 38
        or candidate_solve_count != eligible_candidate_count
        or candidate_solve_count > 36
        or any(
            [int(row["budget"]) for row in target["A3_curve"]] != [0, 1, 2]
            or [int(row["budget"]) for row in target["A5_curve"]] != [0, 1, 2]
            or len(target["A5_order_frozen_llm_plan"]) != 2
            for target in target_results
        )
    ):
        raise AssertionError("auxiliary-channel LLM pilot smoke check failed")

    return {
        "experiment_id": "E2-auxiliary-channel-binding-LLM-plan-pilot",
        "scientific_role": "outcome-sealed target-role workflow pilot",
        "context_exposure": "AGGREGATE_SEEN",
        "outcome_exposure": "SEALED_AT_PLAN_TIME_OPENED_ONCE_FOR_REPORT",
        "plan": {
            "planner_id": plan["planner_id"],
            "path": AUXILIARY_CHANNEL_BINDING_LLM_PLAN_PATH,
            "frozen_before_target_outcome": True,
            "target_count": len(planned_targets),
            "probe_count_per_target": 2,
        },
        "protocol": {
            "program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
            "incumbent": "[normalized target context, zeros(192)]",
            "candidate": (
                "[normalized target context, same-identity same-anchor normalized "
                "auxiliary context]"
            ),
            "feedback_budgets": [0, 1, 2],
            "confirmation": "strict Support gain > 0; stop first positive",
            "A3": "public candidate role order after INELIGIBLE filtering; no Source Memory",
            "A5": "exactly two probes from the frozen LLM plan",
            "query_use": "post-decision evaluator only",
            "auxiliary_future_read": False,
            "cross_identity_binding": False,
            "training_sample_or_weight_change": False,
            "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE; four-series Support and Query means",
            "roster_windows_consumer_metric_or_program_changed": False,
            "original_uci_opened": False,
        },
        "targets": target_results,
        "macro": {
            "dataset_target_macro_adapt_auc": {"A5": a5_macro, "A3": a3_macro},
            "A5_minus_A3": a5_macro - a3_macro,
            "nonnegative_target_count": nonnegative_target_count,
            "harmful_A5_target_count": harmful_target_count,
            "changed_order_or_behavior_target_count": changed_target_count,
            "A5_B2_abstain_target_count": sum(
                bool(row["A5_abstained_at_B2"]) for row in target_results
            ),
        },
        "compute_counts": {
            "ridge_baseline_solve_count": baseline_solve_count,
            "ridge_candidate_solve_count": candidate_solve_count,
            "eligible_candidate_count": eligible_candidate_count,
            "enumerated_candidate_count": enumerated_candidate_count,
            "maximum_candidate_solve_count": 36,
            "total_ridge_solve_count": baseline_solve_count
            + candidate_solve_count,
        },
        "gate": {
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_minus_A3_nonnegative_at_least_3_of_4": (
                nonnegative_target_count >= 3
            ),
            "A5_harmful_target_count_zero": harmful_target_count == 0,
            "A5_changes_top1_order_or_selected_behavior": (
                changed_target_count >= 1
            ),
            "passed": passed,
        },
        "verdict": (
            "LLM_AUX_CHANNEL_WORKFLOW_PILOT_PASS"
            if passed
            else "LLM_AUX_CHANNEL_WORKFLOW_PILOT_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "outcome-sealed target-role workflow pilot on two already-used datasets; "
            "not unseen-dataset Promotion, a Capability/Memory result, or a general "
            "LLM success claim"
        ),
    }


def run_auxiliary_channel_binding_history_p1(root: Path) -> dict[str, object]:
    """Order cached current probes with one phase-aligned historical episode."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        seasonal_scale,
        smase,
    )

    target_roles_by_dataset = {
        "kdd_cup_2018": ("CO", "SO2"),
        "rideshare": ("api_calls", "temp"),
    }
    historical_origin_by_dataset = {
        "kdd_cup_2018": 880,
        "rideshare": 445,
    }
    safe_anchors_by_dataset = {
        "kdd_cup_2018": tuple(ANCHORS),
        "rideshare": (240, 288, 336, 384),
    }
    pilot_path = root / AUXILIARY_CHANNEL_BINDING_LLM_REPORT_PATH
    pilot = _read_object(pilot_path)
    if (
        pilot.get("experiment_id")
        != "E2-auxiliary-channel-binding-LLM-plan-pilot"
        or pilot.get("outcome_exposure")
        != "SEALED_AT_PLAN_TIME_OPENED_ONCE_FOR_REPORT"
        or pilot.get("capability_or_memory_written") is not False
        or pilot.get("protocol", {}).get("program")
        != "BIND_AUXILIARY_CONTEXT_CHANNEL"
    ):
        raise ValueError("current auxiliary-channel pilot cache changed")
    expected_target_keys = {
        (dataset_id, target_role)
        for dataset_id, target_roles in target_roles_by_dataset.items()
        for target_role in target_roles
    }
    cached_targets = {
        (str(row["dataset"]), str(row["target"])): row
        for row in pilot.get("targets", [])
    }
    if set(cached_targets) != expected_target_keys:
        raise ValueError("current auxiliary-channel target cache changed")

    def replay_curve(
        order: list[str], actions: dict[str, dict[str, float]]
    ) -> list[dict[str, object]]:
        curve: list[dict[str, object]] = [
            {
                "budget": 0,
                "probed_auxiliary_channel": None,
                "probed_support_gain": None,
                "probed_query_gain_evaluator_only": None,
                "selected_action": "IDENTITY",
                "selected_auxiliary_channel": None,
                "selected_support_gain": 0.0,
                "fixed_query_gain": 0.0,
                "abstained": True,
                "terminal": False,
            }
        ]
        winner: str | None = None
        for budget in (1, 2):
            probed_role: str | None = None
            probed: dict[str, float] | None = None
            if winner is None and budget - 1 < len(order):
                probed_role = order[budget - 1]
                probed = actions[probed_role]
                if float(probed["support_gain"]) > 0.0:
                    winner = probed_role
            selected = None if winner is None else actions[winner]
            curve.append(
                {
                    "budget": budget,
                    "probed_auxiliary_channel": probed_role,
                    "probed_support_gain": (
                        None if probed is None else float(probed["support_gain"])
                    ),
                    "probed_query_gain_evaluator_only": (
                        None if probed is None else float(probed["query_gain"])
                    ),
                    "selected_action": (
                        "IDENTITY"
                        if winner is None
                        else "BIND_AUXILIARY_CONTEXT_CHANNEL"
                    ),
                    "selected_auxiliary_channel": winner,
                    "selected_support_gain": (
                        0.0 if selected is None else float(selected["support_gain"])
                    ),
                    "fixed_query_gain": (
                        0.0 if selected is None else float(selected["query_gain"])
                    ),
                    "abstained": winner is None,
                    "terminal": winner is not None,
                }
            )
        return curve

    target_results: list[dict[str, object]] = []
    historical_baseline_solve_count = 0
    historical_candidate_solve_count = 0
    historical_enumerated_candidate_count = 0
    for spec in SEMANTIC_AUXILIARY_DATASETS:
        dataset_id = str(spec["dataset_id"])
        target_roles = target_roles_by_dataset[dataset_id]
        historical_origin = historical_origin_by_dataset[dataset_id]
        safe_anchors = safe_anchors_by_dataset[dataset_id]
        current_cutoff = int(spec["train_stop"])
        current_future = tuple(int(value) for value in spec["future"])
        if (
            historical_origin + HORIZON != current_cutoff
            or current_future[0] != current_cutoff
            or any(anchor + HORIZON > historical_origin for anchor in safe_anchors)
        ):
            raise AssertionError("historical episode is not before current future")
        archive_path = Path(str(spec["archive"]))
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        required_stop = max(current_future[1], historical_origin + HORIZON)
        attributes, rows = _read_semantic_tsf_panel(
            np,
            archive_path=archive_path,
            member=str(spec["member"]),
            required_stop=required_stop,
        )
        identity_fields = tuple(str(value) for value in spec["identity_fields"])
        identity_filter = {
            str(key): str(value)
            for key, value in dict(spec["identity_filter"]).items()
        }
        semantic_field = str(spec["semantic_field"])
        public_roles = tuple(str(value) for value in spec["semantic_roles"])
        if not (
            set(identity_fields) | set(identity_filter) | {semantic_field}
        ).issubset(attributes):
            raise ValueError(f"public TSF metadata fields changed: {dataset_id}")

        by_identity: dict[tuple[str, ...], dict[str, dict[str, object]]] = {}
        identity_order: list[tuple[str, ...]] = []
        for row in rows:
            if any(str(row[key]) != value for key, value in identity_filter.items()):
                continue
            identity = tuple(str(row[key]) for key in identity_fields)
            role = str(row[semantic_field])
            if identity not in by_identity:
                by_identity[identity] = {}
                identity_order.append(identity)
            by_identity[identity][role] = row
        complete_identities = [
            identity
            for identity in identity_order
            if set(public_roles).issubset(by_identity[identity])
        ]
        if len(complete_identities) < 20:
            raise ValueError(f"fewer than 20 complete semantic entities: {dataset_id}")
        roster_identities = complete_identities[:20]
        train_identities = roster_identities[:12]
        eval_identities = roster_identities[12:20]
        if len(train_identities) != 12 or len(eval_identities) != 8:
            raise AssertionError("historical episode roster geometry changed")

        for target_role in target_roles:
            cached_target = cached_targets[(dataset_id, target_role)]
            current_menu = list(cached_target["candidate_menu_evaluator_only"])
            current_actions = {
                str(row["auxiliary_channel"]): {
                    "support_gain": float(row["support_gain"]),
                    "query_gain": float(row["query_gain"]),
                }
                for row in current_menu
                if row["eligibility"] == "ELIGIBLE"
            }
            a3_order = [
                role
                for role in public_roles
                if role != target_role and role in current_actions
            ]

            target_train_contexts: list[Any] = []
            target_y: list[Any] = []
            for anchor in safe_anchors:
                for identity in train_identities:
                    raw = np.asarray(
                        by_identity[identity][target_role]["values"], dtype=np.float64
                    )
                    context = raw[anchor - CONTEXT_LENGTH : anchor]
                    target = raw[anchor : anchor + HORIZON]
                    center, scale, method = _center_scale(np, context)
                    if method == "scale_floor_fallback":
                        raise ValueError(
                            f"historical target train scale floor: "
                            f"{dataset_id}/{target_role}"
                        )
                    target_train_contexts.append((context - center) / scale)
                    target_y.append((target - center) / scale)
            target_train_array = np.asarray(
                target_train_contexts, dtype=np.float64
            )
            y_target = np.asarray(target_y, dtype=np.float64)
            baseline_x_train = np.concatenate(
                (target_train_array, np.zeros_like(target_train_array)), axis=1
            )

            target_eval_contexts: list[Any] = []
            historical_actual: list[Any] = []
            centers: list[float] = []
            scales: list[float] = []
            seasonal: list[float] = []
            for identity in eval_identities:
                raw = np.asarray(
                    by_identity[identity][target_role]["values"], dtype=np.float64
                )
                history = raw[:historical_origin]
                context = history[-CONTEXT_LENGTH:]
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise ValueError(
                        f"historical target eval scale floor: "
                        f"{dataset_id}/{target_role}"
                    )
                target_eval_contexts.append((context - center) / scale)
                historical_actual.append(
                    raw[historical_origin : historical_origin + HORIZON]
                )
                centers.append(center)
                scales.append(scale)
                seasonal.append(
                    seasonal_scale(
                        history,
                        np.isfinite(history),
                        period=int(spec["period"]),
                        min_pairs=32,
                    )
                )
            target_eval_array = np.asarray(target_eval_contexts, dtype=np.float64)
            baseline_x_eval = np.concatenate(
                (target_eval_array, np.zeros_like(target_eval_array)), axis=1
            )
            actual_array = np.asarray(historical_actual, dtype=np.float64)
            centers_array = np.asarray(centers, dtype=np.float64)
            scales_array = np.asarray(scales, dtype=np.float64)

            def score_historical_predictions(normalized: Any) -> Any:
                prediction = np.asarray(normalized, dtype=np.float64)
                original = prediction * scales_array[:, None] + centers_array[:, None]
                return np.asarray(
                    [
                        smase(
                            actual_array[index],
                            original[index],
                            scale=float(seasonal[index]),
                        )
                        for index in range(len(eval_identities))
                    ],
                    dtype=np.float64,
                )

            baseline_prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=baseline_x_train,
                targets=y_target,
                weights=np.ones(baseline_x_train.shape[0], dtype=np.float64),
                x_eval=baseline_x_eval,
            )
            historical_baseline_solve_count += 1
            baseline_losses = score_historical_predictions(baseline_prediction)

            historical_menu: list[dict[str, object]] = []
            for auxiliary_role in public_roles:
                if auxiliary_role == target_role:
                    continue
                historical_enumerated_candidate_count += 1
                auxiliary_train_contexts: list[Any] = []
                ineligible_reason: str | None = None
                for anchor in safe_anchors:
                    for identity in train_identities:
                        raw = np.asarray(
                            by_identity[identity][auxiliary_role]["values"],
                            dtype=np.float64,
                        )
                        context = raw[anchor - CONTEXT_LENGTH : anchor]
                        center, scale, method = _center_scale(np, context)
                        if method == "scale_floor_fallback":
                            ineligible_reason = (
                                "HISTORICAL_AUXILIARY_TRAIN_SCALE_FLOOR:"
                                f"{dataset_id}/{target_role}/{auxiliary_role}"
                            )
                            break
                        auxiliary_train_contexts.append((context - center) / scale)
                    if ineligible_reason is not None:
                        break
                auxiliary_eval_contexts: list[Any] = []
                if ineligible_reason is None:
                    for identity in eval_identities:
                        raw = np.asarray(
                            by_identity[identity][auxiliary_role]["values"],
                            dtype=np.float64,
                        )
                        context = raw[
                            historical_origin - CONTEXT_LENGTH : historical_origin
                        ]
                        center, scale, method = _center_scale(np, context)
                        if method == "scale_floor_fallback":
                            ineligible_reason = (
                                "HISTORICAL_AUXILIARY_EVAL_SCALE_FLOOR:"
                                f"{dataset_id}/{target_role}/{auxiliary_role}"
                            )
                            break
                        auxiliary_eval_contexts.append((context - center) / scale)
                if ineligible_reason is not None:
                    historical_menu.append(
                        {
                            "auxiliary_channel": auxiliary_role,
                            "eligibility": "INELIGIBLE",
                            "ineligible_reason": ineligible_reason,
                            "historical_mean_gain": None,
                            "per_identity_historical_gain": [],
                        }
                    )
                    continue

                candidate_x_train = np.concatenate(
                    (
                        target_train_array,
                        np.asarray(auxiliary_train_contexts, dtype=np.float64),
                    ),
                    axis=1,
                )
                candidate_x_eval = np.concatenate(
                    (
                        target_eval_array,
                        np.asarray(auxiliary_eval_contexts, dtype=np.float64),
                    ),
                    axis=1,
                )
                if (
                    candidate_x_train.shape != baseline_x_train.shape
                    or candidate_x_eval.shape != baseline_x_eval.shape
                ):
                    raise AssertionError("historical channel binding geometry changed")
                prediction = _exact_weighted_ridge_prediction(
                    np,
                    x_train=candidate_x_train,
                    targets=y_target,
                    weights=np.ones(candidate_x_train.shape[0], dtype=np.float64),
                    x_eval=candidate_x_eval,
                )
                historical_candidate_solve_count += 1
                gain = baseline_losses - score_historical_predictions(prediction)
                historical_menu.append(
                    {
                        "auxiliary_channel": auxiliary_role,
                        "eligibility": "ELIGIBLE",
                        "ineligible_reason": None,
                        "historical_mean_gain": float(np.mean(gain)),
                        "per_identity_historical_gain": [
                            float(value) for value in gain
                        ],
                    }
                )

            a5_order = [
                str(row["auxiliary_channel"])
                for row in sorted(
                    (
                        row
                        for row in historical_menu
                        if row["eligibility"] == "ELIGIBLE"
                        and str(row["auxiliary_channel"]) in current_actions
                        and float(row["historical_mean_gain"]) > 0.0
                    ),
                    key=lambda row: (
                        -float(row["historical_mean_gain"]),
                        str(row["auxiliary_channel"]),
                    ),
                )
            ]
            a3_curve = replay_curve(a3_order, current_actions)
            a5_curve = replay_curve(a5_order, current_actions)
            llm_old_curve = list(cached_target["A5_curve"])
            a3_auc = policy_episode_adapt_auc(a3_curve)
            a5_auc = policy_episode_adapt_auc(a5_curve)
            llm_old_auc = policy_episode_adapt_auc(llm_old_curve)
            order_changed = a5_order[:2] != a3_order[:2]
            behavior_changed = [
                row["selected_auxiliary_channel"] for row in a5_curve
            ] != [row["selected_auxiliary_channel"] for row in a3_curve]
            target_results.append(
                {
                    "dataset": dataset_id,
                    "target": target_role,
                    "historical_observation": {
                        "origin": historical_origin,
                        "forecast_interval": [
                            historical_origin,
                            historical_origin + HORIZON,
                        ],
                        "current_query_future_start": current_future[0],
                        "strictly_before_current_query_future": (
                            historical_origin + HORIZON <= current_future[0]
                        ),
                        "safe_training_anchors": list(safe_anchors),
                        "evaluation_identity_count": len(eval_identities),
                        "historical_baseline_mean_loss": float(
                            np.mean(baseline_losses)
                        ),
                        "candidate_gains": historical_menu,
                        "positive_threshold": "historical_mean_gain > 0",
                        "reads_current_support_or_query_for_order": False,
                    },
                    "A3_order_public_candidate_order": a3_order,
                    "A5_order_positive_historical_gain": a5_order,
                    "frozen_semantic_LLM_order_descriptive": list(
                        cached_target["A5_order_frozen_llm_plan"]
                    ),
                    "A3_curve": a3_curve,
                    "A5_curve": a5_curve,
                    "frozen_semantic_LLM_curve_descriptive": llm_old_curve,
                    "adapt_auc": {
                        "A3": a3_auc,
                        "A5": a5_auc,
                        "frozen_semantic_LLM_descriptive": llm_old_auc,
                    },
                    "A5_minus_A3": a5_auc - a3_auc,
                    "A5_harmful": any(
                        float(row["fixed_query_gain"]) < 0.0
                        for row in a5_curve[1:]
                    ),
                    "A5_abstained_at_B2": bool(a5_curve[-1]["abstained"]),
                    "order_changed": order_changed,
                    "selected_behavior_changed": behavior_changed,
                    "order_or_behavior_changed": order_changed or behavior_changed,
                }
            )

    a5_macro = statistics.fmean(
        float(row["adapt_auc"]["A5"]) for row in target_results
    )
    a3_macro = statistics.fmean(
        float(row["adapt_auc"]["A3"]) for row in target_results
    )
    llm_old_macro = statistics.fmean(
        float(row["adapt_auc"]["frozen_semantic_LLM_descriptive"])
        for row in target_results
    )
    nonnegative_target_count = sum(
        float(row["A5_minus_A3"]) >= 0.0 for row in target_results
    )
    harmful_target_count = sum(bool(row["A5_harmful"]) for row in target_results)
    changed_target_count = sum(
        bool(row["order_or_behavior_changed"]) for row in target_results
    )
    passed = (
        a5_macro > a3_macro
        and nonnegative_target_count >= 3
        and harmful_target_count == 0
        and changed_target_count >= 1
    )
    historical_eligible_candidate_count = sum(
        row["eligibility"] == "ELIGIBLE"
        for target in target_results
        for row in target["historical_observation"]["candidate_gains"]
    )
    if (
        len(target_results) != 4
        or historical_baseline_solve_count != 4
        or historical_enumerated_candidate_count != 38
        or historical_candidate_solve_count != historical_eligible_candidate_count
        or historical_candidate_solve_count > 36
        or any(
            [int(row["budget"]) for row in target["A3_curve"]] != [0, 1, 2]
            or [int(row["budget"]) for row in target["A5_curve"]] != [0, 1, 2]
            or [int(row["budget"])
                for row in target["frozen_semantic_LLM_curve_descriptive"]]
            != [0, 1, 2]
            or not bool(
                target["historical_observation"][
                    "strictly_before_current_query_future"
                ]
            )
            for target in target_results
        )
    ):
        raise AssertionError("historical channel-binding P1 smoke check failed")

    return {
        "experiment_id": "E2-auxiliary-channel-binding-history-P1",
        "scientific_role": "single historical PolicyEpisode Observation premise",
        "exposure": "CURRENT_OUTCOMES_CACHED_EXPOSED__NO_INCREMENTAL_CURRENT_OUTCOME",
        "frozen_hypothesis": (
            "Phase-aligned historical full channel-binding PolicyEpisodes order "
            "Target-local exact Support probes better than public candidate order."
        ),
        "protocol": {
            "current_action_response_cache": (
                AUXILIARY_CHANNEL_BINDING_LLM_REPORT_PATH
            ),
            "candidate_program": "BIND_AUXILIARY_CONTEXT_CHANNEL",
            "observation": (
                "one phase-aligned historical exact full channel-binding gain on "
                "the same eight evaluation identities"
            ),
            "A3": "public candidate order",
            "A5": (
                "strictly positive historical mean gain, descending gain then "
                "public role name tie-break; empty order abstains"
            ),
            "frozen_semantic_LLM": "descriptive failed baseline only",
            "feedback_budgets": [0, 1, 2],
            "current_confirmation": "strict current Support gain > 0; stop first positive",
            "current_query_use": "post-decision evaluator only from cached pilot",
            "historical_order_reads_current_support_or_query": False,
            "consumer": "unchanged Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "unchanged per-series sMASE",
            "program_roster_current_outcomes_or_metric_changed": False,
            "original_uci_opened": False,
        },
        "targets": target_results,
        "macro": {
            "dataset_target_macro_adapt_auc": {
                "A5": a5_macro,
                "A3": a3_macro,
                "frozen_semantic_LLM_descriptive": llm_old_macro,
            },
            "A5_minus_A3": a5_macro - a3_macro,
            "nonnegative_target_count": nonnegative_target_count,
            "harmful_A5_target_count": harmful_target_count,
            "changed_order_or_behavior_target_count": changed_target_count,
            "A5_B2_abstain_target_count": sum(
                bool(row["A5_abstained_at_B2"]) for row in target_results
            ),
        },
        "compute_counts": {
            "historical_ridge_baseline_solve_count": (
                historical_baseline_solve_count
            ),
            "historical_ridge_candidate_solve_count": (
                historical_candidate_solve_count
            ),
            "historical_eligible_candidate_count": (
                historical_eligible_candidate_count
            ),
            "historical_enumerated_candidate_count": (
                historical_enumerated_candidate_count
            ),
            "historical_total_ridge_solve_count": (
                historical_baseline_solve_count
                + historical_candidate_solve_count
            ),
            "incremental_current_consumer_solve_count": 0,
            "cached_current_action_response_count": sum(
                len(
                    [
                        row
                        for row in cached_targets[key][
                            "candidate_menu_evaluator_only"
                        ]
                        if row["eligibility"] == "ELIGIBLE"
                    ]
                )
                for key in expected_target_keys
            ),
        },
        "gate": {
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_minus_A3_nonnegative_at_least_3_of_4": (
                nonnegative_target_count >= 3
            ),
            "A5_harmful_target_count_zero": harmful_target_count == 0,
            "A5_changes_order_or_behavior": changed_target_count >= 1,
            "passed": passed,
        },
        "verdict": (
            "HISTORICAL_CHANNEL_BINDING_WORKFLOW_PREMISE_PASS"
            if passed
            else "HISTORICAL_CHANNEL_BINDING_OBSERVATION_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "single exposed historical Observation premise using cached current "
            "outcomes; not a promoted Capability, Memory benefit, unseen-Target "
            "transfer result, or permission to add features or thresholds"
        ),
    }


def _multiskill_fast_path_contexts() -> list[dict[str, object]]:
    return [
        {
            "context_id": "forecast_regular_panel",
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "regular_panel": True,
            "known_seasonal_period": True,
            "current_support_feedback_available": True,
            "scale_valid_training_and_evaluation_contexts": True,
        },
        {
            "context_id": "forecast_scale_invalid",
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "regular_panel": True,
            "known_seasonal_period": True,
            "current_support_feedback_available": True,
            "scale_valid_training_and_evaluation_contexts": False,
        },
        {
            "context_id": "classification_fit_only_artifact",
            "task": "classification",
            "consumer": "ridge-raw-plus-difference-v1",
            "binary_class_evidence": True,
            "current_fit_observation_available": True,
            "fit_only_artifact_evidence": True,
            "stable_task_event_evidence": False,
        },
        {
            "context_id": "classification_stable_event",
            "task": "classification",
            "consumer": "ridge-raw-plus-difference-v1",
            "binary_class_evidence": True,
            "current_fit_observation_available": True,
            "fit_only_artifact_evidence": False,
            "stable_task_event_evidence": True,
        },
        {
            "context_id": "unsupported_anomaly",
            "task": "anomaly_detection",
            "consumer": "unsupported",
        },
    ]


def compile_multiskill_llm_fast_path(
    root: Path,
    *,
    plan_path: str = MULTISKILL_LLM_FAST_PATH_PLAN_PATH,
    expected_llm_api_integrated: bool = False,
    llm_api_call_count: int = 0,
) -> dict[str, object]:
    """Constrain a frozen typed LLM plan with two promoted capability contracts."""

    forecast = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    classification = _read_object(root / CONTROLLED_CLASSIFICATION_CAPABILITY_PATH)
    plan = _read_object(root / plan_path)
    forecast_id = "historical_policy_episode_workflow_v1"
    classification_id = "controlled_classification_local_event_dynamic_binding_v1"
    allowed_capability_ids = {forecast_id, classification_id}
    violations: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            violations.append(message)

    require(forecast.get("capability_id") == forecast_id, "forecast capability id")
    require(
        forecast.get("status") == "CROSS_DATASET_SUPPORTED",
        "forecast capability status",
    )
    require(
        forecast.get("task_context", {}).get("task") == "forecasting",
        "forecast task contract",
    )
    require(
        forecast.get("task_context", {}).get("consumer")
        == "shared frozen Ridge",
        "forecast consumer contract",
    )
    require(
        forecast.get("task_context", {}).get("requires_regular_panel") is True
        and forecast.get("task_context", {}).get("requires_known_seasonal_period")
        is True
        and forecast.get("task_context", {}).get(
            "requires_current_support_feedback"
        )
        is True
        and forecast.get("task_context", {}).get(
            "requires_scale_valid_training_and_evaluation_contexts"
        )
        is True,
        "forecast applicability contract",
    )
    require(
        forecast.get("workflow_supply")
        == ["W_rowblock", "W_curation", "W_temporal_origin"],
        "forecast workflow supply",
    )
    require(
        forecast.get("control", {}).get("type") == "stop_on_first_positive"
        and forecast.get("control", {}).get("fallback") == "IDENTITY"
        and forecast.get("risk", {}).get("abstain_if_no_positive_confirmation")
        is True
        and forecast.get("risk", {}).get(
            "do_not_use_query_future_for_ordering_or_confirmation"
        )
        is True
        and forecast.get("risk", {}).get(
            "do_not_allow_later_probe_to_overwrite_confirmed_workflow"
        )
        is True,
        "forecast control and risk contract",
    )

    require(
        classification.get("capability_id") == classification_id,
        "classification capability id",
    )
    require(
        classification.get("status") == "CONTROLLED_CROSS_DATASET_SUPPORTED",
        "classification capability status",
    )
    require(
        classification.get("task_context", {}).get("task") == "classification",
        "classification task contract",
    )
    require(
        classification.get("task_context", {}).get("consumer")
        == "ridge-raw-plus-difference-v1",
        "classification consumer contract",
    )
    require(
        classification.get("task_context", {}).get(
            "requires_binary_class_evidence"
        )
        is True
        and classification.get("task_context", {}).get(
            "requires_current_fit_observation"
        )
        is True,
        "classification applicability contract",
    )
    require(
        classification.get("workflow_supply")
        == [
            "OBSERVE_CROSS_COHORT_CLASS_IMPULSE_TOPOLOGY",
            "BIND_CURRENT_FIT_PATTERN_NODES",
            "CENTER_EXCLUDED_LOCAL_MEDIAN_REPAIR",
        ],
        "classification workflow supply",
    )
    require(
        classification.get("control", {}).get("eligible_action")
        == "execute bound center-excluded local median repair"
        and classification.get("control", {}).get("contraindicated_action")
        == "ABSTAIN_KEEP_INCUMBENT"
        and classification.get("risk", {}).get("abstain_on_stable_task_event")
        is True
        and classification.get("risk", {}).get("event_erasure_guard") is True
        and classification.get("risk", {}).get(
            "do_not_use_query_outcome_for_binding"
        )
        is True
        and classification.get("risk", {}).get(
            "do_not_reuse_source_node_indices"
        )
        is True,
        "classification control and risk contract",
    )

    contexts = _multiskill_fast_path_contexts()
    expected_steps = {
        "forecast_regular_panel": [
            "phase-aligned historical PolicyEpisode orders "
            "W_rowblock/W_curation/W_temporal_origin",
            "current exact Support probes in order",
            "stop on first positive",
            "otherwise IDENTITY",
        ],
        "forecast_scale_invalid": [],
        "classification_fit_only_artifact": [
            "observe cross-cohort class-conditioned impulse topology",
            "bind current-fit pattern nodes",
            "center-excluded local median repair",
        ],
        "classification_stable_event": ["ABSTAIN_KEEP_INCUMBENT"],
        "unsupported_anomaly": [],
    }
    deterministic = {
        "forecast_regular_panel": {
            "capability_id": forecast_id,
            "decision": "RETRIEVE_AND_PROBE",
            "risk_status": "CONTRACT_SATISFIED",
            "compiled_executable_behavior": {
                "operation": "PROBE_ORDERED_WORKFLOW_SUPPLY",
                "workflow_supply": list(forecast.get("workflow_supply", [])),
                "confirmation": forecast.get("control", {}).get("confirmation"),
                "stop_rule": forecast.get("control", {}).get("type"),
                "fallback": forecast.get("control", {}).get("fallback"),
            },
        },
        "forecast_scale_invalid": {
            "capability_id": None,
            "decision": "NO_APPLICABLE_SKILL",
            "risk_status": "APPLICABILITY_REJECTED_SAFE",
            "compiled_executable_behavior": {
                "operation": "NO_OP",
                "reason": "scale validity requirement failed",
                "fallback": "IDENTITY",
            },
        },
        "classification_fit_only_artifact": {
            "capability_id": classification_id,
            "decision": "RETRIEVE_AND_EXECUTE_BOUND_REPAIR",
            "risk_status": "CONTRACT_SATISFIED",
            "compiled_executable_behavior": {
                "operation": "EXECUTE_BOUND_REPAIR",
                "workflow_steps": list(
                    classification.get("workflow_supply", [])
                ),
                "action": classification.get("control", {}).get(
                    "eligible_action"
                ),
                "reuse_source_node_indices": False,
            },
        },
        "classification_stable_event": {
            "capability_id": classification_id,
            "decision": "RETRIEVE_AND_ABSTAIN",
            "risk_status": "RISK_GUARD_ABSTAINED_SAFE",
            "compiled_executable_behavior": {
                "operation": classification.get("control", {}).get(
                    "contraindicated_action"
                ),
                "guard": "event_erasure_guard",
                "fallback": classification.get("control", {}).get("fallback"),
            },
        },
        "unsupported_anomaly": {
            "capability_id": None,
            "decision": "NO_APPLICABLE_SKILL",
            "risk_status": "NO_PROMOTED_SKILL_SAFE",
            "compiled_executable_behavior": {
                "operation": "NO_OP",
                "reason": "unsupported task",
            },
        },
    }

    forbidden_plan_fields: list[str] = []

    def scan_plan_fields(value: object, prefix: str = "plan") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}"
                lowered = str(key).lower()
                if any(token in lowered for token in ("outcome", "utility", "loss")):
                    forbidden_plan_fields.append(path)
                scan_plan_fields(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan_plan_fields(child, f"{prefix}[{index}]")

    scan_plan_fields(plan)
    require(not forbidden_plan_fields, "plan contains outcome/utility/loss fields")
    require(
        plan.get("planner_id") == "llm_multiskill_fast_path_v1",
        "planner id",
    )
    require(
        plan.get("llm_api_integrated") is expected_llm_api_integrated,
        "LLM API integration flag",
    )
    plan_decisions_raw = list(plan.get("decisions", []))
    plan_decisions = {
        str(row.get("context_id")): row for row in plan_decisions_raw
    }
    expected_context_ids = {str(row["context_id"]) for row in contexts}
    require(
        len(plan_decisions_raw) == 5
        and len(plan_decisions) == 5
        and set(plan_decisions) == expected_context_ids,
        "five fixed plan contexts",
    )
    selected_capability_ids = {
        str(row.get("capability_id"))
        for row in plan_decisions_raw
        if row.get("capability_id") is not None
    }
    unknown_selected_ids = selected_capability_ids - allowed_capability_ids
    require(not unknown_selected_ids, "unpromoted capability selected")
    declared_unpromoted = list(plan.get("unpromoted_skills_selected", []))
    declared_invented = list(plan.get("new_skills_invented", []))
    require(not declared_unpromoted, "plan declares unpromoted Skill")
    require(not declared_invented, "plan declares invented Skill")

    decisions: list[dict[str, object]] = []
    behavior_match_count = 0
    for context in contexts:
        context_id = str(context["context_id"])
        planned = plan_decisions.get(context_id, {})
        compiled = deterministic[context_id]
        plan_capability_id = planned.get("capability_id")
        planned_steps = list(planned.get("workflow_steps", []))
        capability_match = plan_capability_id == compiled["capability_id"]
        decision_match = planned.get("decision") == compiled["decision"]
        steps_match = planned_steps == expected_steps[context_id]
        matched = capability_match and decision_match and steps_match
        behavior_match_count += matched
        if not capability_match:
            violations.append(f"{context_id}: capability mismatch")
        if not decision_match:
            violations.append(f"{context_id}: decision mismatch")
        if not steps_match:
            violations.append(f"{context_id}: workflow step mismatch")
        decisions.append(
            {
                "context_id": context_id,
                "input_summary": context,
                "retrieved_skill": plan_capability_id,
                "llm_decision": planned.get("decision"),
                "llm_workflow_steps": planned_steps,
                "deterministic_retrieved_skill": compiled["capability_id"],
                "deterministic_decision": compiled["decision"],
                "compiled_executable_behavior": compiled[
                    "compiled_executable_behavior"
                ],
                "risk_status": compiled["risk_status"],
                "capability_match": capability_match,
                "decision_match": decision_match,
                "workflow_steps_allowed": steps_match,
                "behavior_match": matched,
            }
        )

    unpromoted_or_invented_count = (
        len(unknown_selected_ids) + len(declared_unpromoted) + len(declared_invented)
    )
    passed = (
        behavior_match_count == 5
        and unpromoted_or_invented_count == 0
        and not forbidden_plan_fields
        and not violations
    )
    return {
        "experiment_id": "E2-multiskill-LLM-fast-path-runtime",
        "scientific_role": "typed plan constraint and executable behavior check",
        "inputs": {
            "capabilities": [
                HISTORICAL_POLICY_CAPABILITY_PATH,
                CONTROLLED_CLASSIFICATION_CAPABILITY_PATH,
            ],
            "frozen_plan": plan_path,
            "allowed_capability_ids": sorted(allowed_capability_ids),
        },
        "decisions": decisions,
        "validation": {
            "behavior_match_count": behavior_match_count,
            "expected_behavior_count": 5,
            "all_five_match": behavior_match_count == 5,
            "forbidden_plan_fields": forbidden_plan_fields,
            "unpromoted_or_invented_count": unpromoted_or_invented_count,
            "closed_semantic_or_channel_family_selected_count": 0,
            "contract_violations": violations,
        },
        "compute_counts": {
            "consumer_fit_count": 0,
            "data_load_count": 0,
            "llm_api_call_count": llm_api_call_count,
        },
        "llm_api_integrated": expected_llm_api_integrated,
        "gate": {
            "five_of_five_behaviors_match": behavior_match_count == 5,
            "no_unpromoted_or_invented_skill": (
                unpromoted_or_invented_count == 0
            ),
            "no_forbidden_plan_fields": not forbidden_plan_fields,
            "hard_contracts_satisfied": not violations,
            "passed": passed,
        },
        "verdict": (
            "MULTISKILL_LLM_FAST_PATH_BEHAVIOR_PASS"
            if passed
            else "MULTISKILL_LLM_FAST_PATH_BEHAVIOR_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "This proves only that a typed LLM plan can be constrained correctly "
            "by the two supplied capability and risk contracts. It does not prove "
            "that the LLM outperforms a deterministic retriever or that the "
            "controlled classification Skill is a natural second Skill."
        ),
    }


def run_live_multiskill_llm_fast_path(
    root: Path,
    *,
    model: str = "gpt-5.6-luna",
    base_url: str = "https://api.agicto.cn/v1",
) -> dict[str, object]:
    """Generate one live typed plan, then compile it through existing contracts."""

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")

    forecast = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    classification = _read_object(root / CONTROLLED_CLASSIFICATION_CAPABILITY_PATH)
    contexts = _multiskill_fast_path_contexts()
    capability_views = []
    for capability in (forecast, classification):
        capability_views.append(
            {
                "capability_id": capability.get("capability_id"),
                "status": capability.get("status"),
                "task_context": capability.get("task_context"),
                "workflow_supply": capability.get("workflow_supply"),
                "observation": capability.get("observation"),
                "control": capability.get("control"),
                "risk": capability.get("risk"),
                "claim_limit": capability.get("claim_limit"),
            }
        )

    user_payload = {
        "capabilities": capability_views,
        "contexts": contexts,
        "required_output": {
            "planner_id": "llm_multiskill_fast_path_v1",
            "planner_runtime": "project live OpenAI-compatible Chat Completions",
            "llm_api_integrated": True,
            "decisions": [
                {
                    "context_id": "one supplied context_id",
                    "capability_id": "one supplied capability_id or null",
                    "decision": (
                        "RETRIEVE_AND_PROBE | RETRIEVE_AND_EXECUTE_BOUND_REPAIR | "
                        "RETRIEVE_AND_ABSTAIN | NO_APPLICABLE_SKILL"
                    ),
                    "workflow_steps": ["exact allowed strings"],
                    "reason_codes": ["short reason codes"],
                }
            ],
            "unpromoted_skills_selected": [],
            "new_skills_invented": [],
        },
        "allowed_workflow_steps_by_context": {
            "forecast_regular_panel": [
                "phase-aligned historical PolicyEpisode orders W_rowblock/W_curation/W_temporal_origin",
                "current exact Support probes in order",
                "stop on first positive",
                "otherwise IDENTITY",
            ],
            "forecast_scale_invalid": [],
            "classification_fit_only_artifact": [
                "observe cross-cohort class-conditioned impulse topology",
                "bind current-fit pattern nodes",
                "center-excluded local median repair",
            ],
            "classification_stable_event": ["ABSTAIN_KEEP_INCUMBENT"],
            "unsupported_anomaly": [],
        },
    }
    system_prompt = (
        "You are the fast-path planner inside a time-series data adaptation Harness. "
        "Select only supplied promoted capabilities. Enforce task, consumer, "
        "applicability, binding and risk constraints. Never invent a capability, "
        "workflow or observation. If no capability is applicable, choose "
        "NO_APPLICABLE_SKILL. If a stable classification task event would be erased, "
        "choose RETRIEVE_AND_ABSTAIN. Return exactly one JSON object and no markdown. "
        "Include exactly one decision for every supplied context. Do not include any "
        "field whose name contains outcome, utility or loss. Copy workflow step strings "
        "exactly from allowed_workflow_steps_by_context."
    )

    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    )
    choice = completion.choices[0]
    assistant_text = choice.message.content or ""
    plan = json.loads(assistant_text)
    if not isinstance(plan, dict):
        raise ValueError("live planner must return one JSON object")
    plan["planner_id"] = "llm_multiskill_fast_path_v1"
    plan["planner_runtime"] = "project live OpenAI-compatible Chat Completions"
    plan["llm_api_integrated"] = True
    plan.setdefault("unpromoted_skills_selected", [])
    plan.setdefault("new_skills_invented", [])
    plan["provider"] = {
        "base_url": base_url,
        "requested_model": model,
        "returned_model": getattr(completion, "model", ""),
        "finish_reason": getattr(choice, "finish_reason", ""),
        "prompt_tokens": getattr(getattr(completion, "usage", None), "prompt_tokens", None),
        "completion_tokens": getattr(
            getattr(completion, "usage", None), "completion_tokens", None
        ),
    }

    plan_output = root / MULTISKILL_LIVE_LLM_FAST_PATH_PLAN_PATH
    plan_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = compile_multiskill_llm_fast_path(
        root,
        plan_path=MULTISKILL_LIVE_LLM_FAST_PATH_PLAN_PATH,
        expected_llm_api_integrated=True,
        llm_api_call_count=1,
    )
    report["provider"] = plan["provider"]
    report["experiment_id"] = "E2-multiskill-live-LLM-fast-path-runtime"
    report["verdict"] = (
        "MULTISKILL_LIVE_LLM_FAST_PATH_BEHAVIOR_PASS"
        if report["gate"]["passed"]
        else "MULTISKILL_LIVE_LLM_FAST_PATH_BEHAVIOR_FAIL"
    )
    report["claim_limit"] = (
        "This is a live project-side LLM planning and deterministic-compilation "
        "behavior result. It is not a new time-series Capability result and does "
        "not show that the LLM beats deterministic retrieval."
    )
    return report


def _forecasting_two_skill_contexts() -> list[dict[str, object]]:
    """Synthetic deployment-visible contexts; no dataset identity or outcome."""

    return [
        {
            "context_id": "forecast_general_regular_panel",
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "regular_panel": True,
            "known_seasonal_period": True,
            "current_support_feedback_available": True,
            "scale_valid": True,
            "observable_natural_missing_mask": False,
        },
        {
            "context_id": "forecast_missing_mixed_reliable_history",
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "regular_panel": True,
            "known_seasonal_period": True,
            "scale_valid": True,
            "observable_natural_missing_mask": True,
            "mixed_reliable_and_unreliable_training_windows": True,
            "distinct_phase_aligned_historical_origins": 3,
        },
        {
            "context_id": "forecast_missing_insufficient_history",
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "regular_panel": True,
            "known_seasonal_period": True,
            "scale_valid": True,
            "observable_natural_missing_mask": True,
            "mixed_reliable_and_unreliable_training_windows": True,
            "distinct_phase_aligned_historical_origins": 1,
        },
        {
            "context_id": "forecast_missing_no_actionable_geometry",
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "regular_panel": True,
            "known_seasonal_period": True,
            "scale_valid": True,
            "observable_natural_missing_mask": True,
            "mixed_reliable_and_unreliable_training_windows": False,
            "all_training_windows_unreliable": True,
            "current_support_feedback_available": False,
        },
        {
            "context_id": "classification_not_supported_by_forecast_skills",
            "task": "classification",
            "consumer": "ridge-raw-plus-difference-v1",
        },
    ]


def _compile_forecasting_two_skill_plan(
    root: Path,
    plan: dict[str, object],
) -> dict[str, object]:
    """Compile a live plan against two same-task Forecasting Skill contracts."""

    general = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    missing = _read_object(root / MISSING_WINDOW_WEIGHTING_CAPABILITY_PATH)
    general_id = "historical_policy_episode_workflow_v1"
    missing_id = "missing_window_weighting_workflow_v1"
    violations = []
    if (
        general.get("capability_id") != general_id
        or general.get("status") != "CROSS_DATASET_SUPPORTED"
    ):
        violations.append("general Forecasting Skill contract")
    if (
        missing.get("capability_id") != missing_id
        or missing.get("status") != "NATURAL_TARGET_PILOT_SUPPORTED"
        or missing.get("risk", {}).get("minimum_distinct_historical_origins") != 2
    ):
        violations.append("missing-window Skill contract")

    general_steps = [
        "retrieve phase-aligned historical PolicyEpisodes for W_rowblock/W_curation/W_temporal_origin",
        "probe current Support in retrieved order",
        "stop on first positive; otherwise IDENTITY",
    ]
    missing_steps = [
        "observe natural-missing training-window composition",
        "retrieve Source PolicyEpisode to order ATTENUATE/EXCLUDE probes",
        "confirm on at least two phase-aligned historical Target origins",
        "stop on first positive; otherwise KEEP_ALL",
    ]
    expected = {
        "forecast_general_regular_panel": (general_id, "RETRIEVE_AND_PROBE", general_steps),
        "forecast_missing_mixed_reliable_history": (
            missing_id,
            "RETRIEVE_AND_PROBE",
            missing_steps,
        ),
        "forecast_missing_insufficient_history": (
            missing_id,
            "RETRIEVE_AND_ABSTAIN",
            ["ABSTAIN_KEEP_ALL"],
        ),
        "forecast_missing_no_actionable_geometry": (
            None,
            "NO_APPLICABLE_SKILL",
            [],
        ),
        "classification_not_supported_by_forecast_skills": (
            None,
            "NO_APPLICABLE_SKILL",
            [],
        ),
    }
    forbidden_fields = []

    def scan(value: object, prefix: str = "plan") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}"
                if any(
                    token in str(key).lower() for token in ("outcome", "utility", "loss")
                ):
                    forbidden_fields.append(path)
                scan(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                scan(child, f"{prefix}[{index}]")

    scan(plan)
    decisions_raw = list(plan.get("decisions", []))
    decisions_by_id = {str(row.get("context_id")): row for row in decisions_raw}
    if len(decisions_raw) != 5 or set(decisions_by_id) != set(expected):
        violations.append("five fixed contexts")
    if plan.get("new_skills_invented") or plan.get("unadmitted_skills_selected"):
        violations.append("invented or unadmitted Skill")
    compiled_rows = []
    raw_matched = 0
    runtime_matched = 0
    for context in _forecasting_two_skill_contexts():
        context_id = str(context["context_id"])
        row = decisions_by_id.get(context_id, {})
        capability_id, decision, steps = expected[context_id]
        raw_match = (
            row.get("capability_id") == capability_id
            and row.get("decision") == decision
            and list(row.get("workflow_steps", [])) == steps
        )
        raw_matched += raw_match

        # The LLM proposes a plan; it does not own applicability or Risk.  The
        # runtime rejects an inapplicable Skill before any executable behavior
        # can be emitted.  Keep the raw mismatch visible instead of pretending
        # the planner itself was correct.
        if capability_id is None:
            runtime_capability_id = None
            runtime_decision = "NO_APPLICABLE_SKILL"
            runtime_steps: list[str] = []
            runtime_action = "REJECT_INAPPLICABLE_SKILL_AND_NO_OP"
        elif row.get("capability_id") != capability_id:
            runtime_capability_id = None
            runtime_decision = "NO_APPLICABLE_SKILL"
            runtime_steps = []
            runtime_action = "REJECT_NONMATCHING_SKILL_AND_NO_OP"
        elif decision == "RETRIEVE_AND_ABSTAIN":
            runtime_capability_id = capability_id
            runtime_decision = decision
            runtime_steps = list(steps)
            runtime_action = "COMPILE_RISK_ABSTENTION"
        elif (
            row.get("decision") == decision
            and list(row.get("workflow_steps", [])) == steps
        ):
            runtime_capability_id = capability_id
            runtime_decision = decision
            runtime_steps = list(steps)
            runtime_action = "COMPILE_ALLOWED_WORKFLOW"
        else:
            runtime_capability_id = capability_id
            runtime_decision = "RETRIEVE_AND_ABSTAIN"
            runtime_steps = ["ABSTAIN_KEEP_ALL"]
            runtime_action = "REJECT_INVALID_PLAN_AND_ABSTAIN"

        runtime_match = (
            runtime_capability_id == capability_id
            and runtime_decision == decision
            and runtime_steps == steps
        )
        runtime_matched += runtime_match
        if not runtime_match:
            violations.append(f"{context_id}: compiled runtime mismatch")
        compiled_rows.append(
            {
                "context_id": context_id,
                "input_summary": context,
                "llm_capability_id": row.get("capability_id"),
                "llm_decision": row.get("decision"),
                "llm_workflow_steps": list(row.get("workflow_steps", [])),
                "raw_planner_behavior_match": raw_match,
                "deterministic_capability_id": capability_id,
                "deterministic_decision": decision,
                "runtime_capability_id": runtime_capability_id,
                "runtime_decision": runtime_decision,
                "runtime_workflow_steps": runtime_steps,
                "runtime_action": runtime_action,
                "compiled_runtime_behavior_match": runtime_match,
            }
        )
    passed = runtime_matched == 5 and not forbidden_fields and not violations
    return {
        "experiment_id": "E2-live-LLM-same-task-two-Forecasting-Skills",
        "scientific_role": "same-task typed Skill retrieval and Risk compilation",
        "inputs": {
            "capabilities": [
                HISTORICAL_POLICY_CAPABILITY_PATH,
                MISSING_WINDOW_WEIGHTING_CAPABILITY_PATH,
            ],
            "synthetic_context_count": 5,
            "raw_time_series_or_experiment_outcomes_sent": False,
        },
        "decisions": compiled_rows,
        "validation": {
            "raw_planner_behavior_match_count": raw_matched,
            "compiled_runtime_behavior_match_count": runtime_matched,
            "forbidden_plan_fields": forbidden_fields,
            "contract_violations": violations,
        },
        "compute": {"consumer_fit_count": 0, "llm_api_call_count": 1},
        "gate": {
            "raw_planner_five_of_five": raw_matched == 5,
            "compiled_runtime_five_of_five": runtime_matched == 5,
            "no_forbidden_fields": not forbidden_fields,
            "hard_contracts_satisfied": not violations,
            "passed": passed,
        },
        "verdict": (
            "LIVE_LLM_SAME_TASK_TWO_SKILL_RUNTIME_COMPILE_PASS"
            if passed
            else "LIVE_LLM_SAME_TASK_TWO_SKILL_RUNTIME_COMPILE_FAIL"
        ),
        "claim_limit": (
            "Behavior-only live LLM test. The raw planner remains separately scored; "
            "a compiled PASS proves only that deterministic applicability and Risk "
            "contracts prevent an invalid plan from becoming executable behavior."
        ),
    }


def run_forecasting_two_skill_compile_replay(root: Path) -> dict[str, object]:
    """Compile the existing live plan without another external API call."""

    plan = _read_object(root / FORECASTING_TWO_SKILL_LIVE_PLAN_PATH)
    report = _compile_forecasting_two_skill_plan(root, plan)
    report["experiment_id"] = "E2-LLM-same-task-applicability-runtime-compile"
    report["compute"] = {"consumer_fit_count": 0, "llm_api_call_count": 0}
    report["source_live_plan"] = FORECASTING_TWO_SKILL_LIVE_PLAN_PATH
    return report


def _compile_historical_policy_llm_slow_path_plan(
    root: Path,
    plan: dict[str, object],
) -> dict[str, object]:
    """Compile one live LLM patch against the admitted natural Skill behavior."""

    violations: list[str] = []
    operations = list(plan.get("operations", []))
    normalized = {
        (
            str(row.get("operation")),
            str(row.get("target_surface")),
            str(row.get("value")),
        )
        for row in operations
        if isinstance(row, dict)
    }
    expected_operations = {
        (
            "ADD_OBSERVATION",
            "observation",
            "phase_aligned_historical_policy_episode",
        ),
        (
            "PATCH_CONTROL",
            "harness_update_policy",
            "stop_on_first_positive",
        ),
    }
    if len(operations) != 2 or normalized != expected_operations:
        violations.append("typed operations do not realize both first-fault repairs")
    if plan.get("new_programs"):
        violations.append("LLM invented a Program")
    if plan.get("consumer_changes") or plan.get("metric_changes"):
        violations.append("LLM changed the frozen Judge")
    if plan.get("memory_schema_changes"):
        violations.append("LLM changed the Memory schema")

    baseline = _read_object(
        root
        / "artifacts/functional/e2/historical_policy_episode_failure_to_capability_report.json"
    )
    admitted = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    if baseline.get("verdict") != "FAILURE_TO_TYPED_CAPABILITY_UPDATE_PASS":
        violations.append("deterministic Slow-Path baseline is unavailable")

    compiled_patch = None
    behavior_checks = {
        "workflow_supply_equivalent": False,
        "observation_equivalent": False,
        "control_equivalent": False,
        "risk_equivalent": False,
    }
    if not violations:
        deterministic_patch = baseline["typed_patch"]
        compiled_patch = {
            "patch_id": str(plan.get("patch_id", "llm_historical_policy_patch")),
            "operations": list(deterministic_patch["operations"]),
            "unchanged": list(deterministic_patch["unchanged"]),
        }
        behavior_checks = {
            "workflow_supply_equivalent": admitted.get("workflow_supply")
            == ["W_rowblock", "W_curation", "W_temporal_origin"],
            "observation_equivalent": admitted.get("observation", {}).get("type")
            == "phase_aligned_historical_policy_episode",
            "control_equivalent": admitted.get("control", {}).get("type")
            == "stop_on_first_positive",
            "risk_equivalent": bool(
                admitted.get("risk", {}).get(
                    "do_not_use_query_future_for_ordering_or_confirmation"
                )
            )
            and bool(
                admitted.get("risk", {}).get(
                    "do_not_allow_later_probe_to_overwrite_confirmed_workflow"
                )
            ),
        }
    passed = not violations and all(behavior_checks.values())
    return {
        "experiment_id": "E2-live-LLM-historical-policy-Slow-Path",
        "scientific_role": "natural FailurePatternCard to typed Harness patch",
        "inputs": {
            "initial_capability_memory_empty": True,
            "failure_count": 2,
            "raw_time_series_sent": False,
            "dataset_identities_sent": False,
            "effect_magnitudes_sent": False,
            "internal_report_sent": False,
        },
        "llm_plan": plan,
        "compiled_typed_patch": compiled_patch,
        "compiled_capability_behavior_checks": behavior_checks,
        "downstream_evidence_reused_without_refit": baseline.get(
            "downstream_evidence"
        ),
        "validation": {"contract_violations": violations},
        "compute": {"consumer_fit_count": 0, "llm_api_call_count": 1},
        "gate": {"passed": passed},
        "verdict": (
            "LIVE_LLM_FAILURE_TO_TYPED_CAPABILITY_UPDATE_PASS"
            if passed
            else "LIVE_LLM_FAILURE_TO_TYPED_CAPABILITY_UPDATE_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Development mechanism result on exposed categorical first faults. A PASS "
            "shows that the project-side LLM can propose the already validated typed "
            "Observation/Control update from empty Capability Memory; it does not make "
            "the old Target evidence fresh or prove autonomous invention on a new family."
        ),
    }


def run_live_historical_policy_llm_slow_path(
    root: Path,
    *,
    model: str,
    base_url: str,
) -> dict[str, object]:
    """Use one live LLM call to propose a bounded natural Harness patch."""

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    dossier = {
        "initial_capability_memory": [],
        "task_consumer": {
            "task": "forecasting",
            "consumer": "fixed shared forecasting Consumer",
        },
        "workflow_supply": ["W_rowblock", "W_curation", "W_temporal_origin"],
        "categorical_first_faults": [
            {
                "surface": "observation",
                "code": "GLOBAL_WORKFLOW_ORDER_NOT_TARGET_CONTEXTUALIZED",
                "observed_behavior": (
                    "a useful Workflow is delayed by a global Source order"
                ),
                "available_legal_observation": (
                    "phase-aligned historical PolicyEpisodes whose targets end before "
                    "the current Query cutoff"
                ),
            },
            {
                "surface": "harness_update_policy",
                "code": "CONFIRMED_POSITIVE_WORKFLOW_OVERWRITTEN",
                "observed_behavior": (
                    "later probes overwrite an already confirmed positive Workflow"
                ),
            },
        ],
        "allowed_patch_operations": {
            "observation": [
                {
                    "operation": "ADD_OBSERVATION",
                    "value": "phase_aligned_historical_policy_episode",
                }
            ],
            "harness_update_policy": [
                {
                    "operation": "PATCH_CONTROL",
                    "value": "stop_on_first_positive",
                }
            ],
        },
        "forbidden_changes": [
            "program_supply",
            "consumer",
            "metric",
            "memory_schema",
            "query_visibility",
        ],
        "privacy": {
            "raw_time_series_included": False,
            "dataset_identity_included": False,
            "effect_magnitude_included": False,
            "internal_report_included": False,
        },
    }
    payload = {
        "failure_dossier": dossier,
        "required_output": {
            "patch_id": "short identifier",
            "diagnosis": {
                "first_fault_codes": ["supplied codes"],
                "why_one_patch_needs_both_operations": "short reason",
            },
            "operations": [
                {
                    "operation": "ADD_OBSERVATION | PATCH_CONTROL",
                    "target_surface": "observation | harness_update_policy",
                    "value": "one supplied allowed value",
                }
            ],
            "expected_behavior_change": "short description",
            "falsification_condition": "short description",
            "new_programs": [],
            "consumer_changes": [],
            "metric_changes": [],
            "memory_schema_changes": [],
        },
    }
    system_prompt = (
        "You are the Slow-Path patch proposer inside a time-series data adaptation "
        "Harness. Convert the supplied categorical first faults into the smallest typed "
        "Harness patch. Use only supplied operations and values. Preserve Program supply, "
        "Consumer, metric, Memory schema and Query visibility. Do not invent data, "
        "thresholds, outcomes or Skills. Return exactly one JSON object and no markdown."
    )
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    choice = completion.choices[0]
    plan = json.loads(choice.message.content or "")
    if not isinstance(plan, dict):
        raise ValueError("Slow-Path planner must return one JSON object")
    plan["provider"] = {
        "base_url": base_url,
        "requested_model": model,
        "returned_model": getattr(completion, "model", ""),
        "prompt_tokens": getattr(getattr(completion, "usage", None), "prompt_tokens", None),
        "completion_tokens": getattr(
            getattr(completion, "usage", None), "completion_tokens", None
        ),
    }
    plan_path = root / HISTORICAL_POLICY_LLM_SLOW_PATH_PLAN_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = _compile_historical_policy_llm_slow_path_plan(root, plan)
    report["provider"] = plan["provider"]
    return report


def run_skill_acquisition_framework_replay(
    root: Path,
    *,
    workflow_supply_override: tuple[str, ...] | None = None,
    compiled_workflows_override: tuple[dict[str, object], ...] | None = None,
    typed_patch_override: dict[str, object] | None = None,
) -> dict[str, object]:
    """Replay the admitted natural Skill through the reusable acquisition core.

    This is framework-convergence evidence, not a new Capability experiment.  It
    starts from empty Capability Memory, compiles a Candidate from completed Source
    PolicyEpisodes, applies the already recorded live-LLM typed patch, replays the
    Candidate on three cached natural Targets, and delegates promotion to an explicit
    callback only after all cached target curves match their original evidence.
    """

    admitted = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    workflow_supply = (
        workflow_supply_override
        if workflow_supply_override is not None
        else tuple(str(value) for value in admitted["workflow_supply"])
    )
    compiled_workflows = (
        compiled_workflows_override
        if compiled_workflows_override is not None
        else tuple(
            {"workflow_id": workflow_id, "bindings": {}}
            for workflow_id in workflow_supply
        )
    )
    source_report_paths = (
        "artifacts/functional/e2/electricity_hourly_policy_memory_promotion_report.json",
        "artifacts/functional/e2/m4_hourly_historical_policy_context_promotion_report.json",
    )
    source_contexts = []
    for path in source_report_paths:
        report = _read_object(root / path)
        source_contexts.append(
            {
                "case_id": Path(path).stem,
                "workflow_outcomes": dict(report["workflow_outcomes"]),
            }
        )

    def evaluate_cached_source_workflow(
        context: dict[str, object],
        workflow_id: str,
        _bindings: dict[str, object],
    ) -> dict[str, object]:
        outcomes = context["workflow_outcomes"]
        if not isinstance(outcomes, dict):
            raise ValueError("cached Source context requires workflow_outcomes")
        response = outcomes[workflow_id]
        if not isinstance(response, dict):
            raise ValueError("cached Source Workflow response must be an object")
        return response

    source_episodes = collect_source_policy_episodes(
        source_contexts,
        compiled_workflows,
        evaluate_cached_source_workflow,
    )
    if typed_patch_override is None:
        llm_plan = _read_object(root / HISTORICAL_POLICY_LLM_SLOW_PATH_PLAN_PATH)
        patch = {
            "patch_id": str(llm_plan["patch_id"]),
            "operations": list(llm_plan["operations"]),
        }
    else:
        patch = {
            "patch_id": str(typed_patch_override["patch_id"]),
            "operations": list(typed_patch_override["operations"]),
        }
    target_paths = (
        "artifacts/functional/e2/pedestrian_counts_historical_policy_skill_confirmation_report.json",
        "artifacts/functional/e2/rideshare_historical_policy_skill_confirmation_report.json",
        "artifacts/functional/e2/kdd_historical_policy_skill_memory_target_report.json",
    )
    validation_cases: list[dict[str, object]] = []
    expected_curves: dict[str, list[dict[str, object]]] = {}
    source_paths: dict[str, str] = {}
    for path in target_paths:
        report = _read_object(root / path)
        case_id = Path(path).stem
        historical_gains = dict(report["historical_policy_context"]["workflow_gains"])
        historical_episode = {
            "workflows": {
                workflow_id: {
                    "workflow_id": workflow_id,
                    "support_gain": float(gain),
                }
                for workflow_id, gain in historical_gains.items()
            }
        }
        validation_cases.append(
            {
                "case_id": case_id,
                "historical_episode": historical_episode,
                "current_workflows": dict(report["workflow_outcomes"]),
            }
        )
        expected_curves[case_id] = list(report["adaptation_curve"])
        source_paths[case_id] = path

    def decide_promotion(
        _candidate: dict[str, object], replays: list[dict[str, object]]
    ) -> dict[str, object]:
        exact_matches = 0
        harms = 0
        for row in replays:
            case_id = str(row["case_id"])
            replay = dict(row["replay"])
            exact_matches += replay["adaptation_curve"] == expected_curves[case_id]
            harms += float(replay["adaptation_curve"][-1]["fixed_query_gain"]) < 0.0
        return {
            "status": (
                "CROSS_DATASET_SUPPORTED"
                if exact_matches == len(replays) and harms == 0
                else "REJECTED"
            ),
            "cached_natural_target_count": len(replays),
            "exact_curve_match_count": exact_matches,
            "harmful_target_count": harms,
        }

    cycle = run_skill_acquisition_cycle(
        [],
        source_episodes,
        validation_cases,
        capability_id="historical_policy_episode_workflow_reconstructed_v1",
        task_context=dict(admitted["task_context"]),
        workflow_supply=workflow_supply,
        typed_patch=patch,
        promotion=decide_promotion,
    )
    candidate = dict(cycle["candidate_before_patch"])
    patched_candidate = dict(cycle["candidate_after_patch"])
    resolved = dict(cycle["resolved_skill"])
    target_replays: list[dict[str, object]] = []
    for row in cycle["validation_replays"]:
        case_id = str(row["case_id"])
        replay = dict(row["replay"])
        curve_match = replay["adaptation_curve"] == expected_curves[case_id]
        final_gain = float(replay["adaptation_curve"][-1]["fixed_query_gain"])
        target_replays.append(
            {
                "source_report": source_paths[case_id],
                "probe_order": list(replay["probe_order"]),
                "selected_workflow": replay["selected_workflow"],
                "adaptation_auc": float(replay["adaptation_auc"]),
                "original_curve_exact_match": curve_match,
                "harmful": final_gain < 0.0,
            }
        )

    exact_match_count = sum(
        bool(row["original_curve_exact_match"]) for row in target_replays
    )
    harmful_target_count = sum(bool(row["harmful"]) for row in target_replays)
    replay_passed = exact_match_count == len(target_replays) and harmful_target_count == 0
    behavior_equivalent = (
        set(resolved["workflow_supply"]) == set(admitted["workflow_supply"])
        and resolved["observation"].get("type")
        == admitted["observation"].get("type")
        and resolved["control"].get("type") == admitted["control"].get("type")
        and resolved["control"].get("fallback")
        == admitted["control"].get("fallback")
        and resolved["risk"] == admitted["risk"]
    )
    passed = (
        candidate["status"] == "CANDIDATE"
        and patched_candidate["status"] == "CANDIDATE"
        and resolved["status"] == "CROSS_DATASET_SUPPORTED"
        and replay_passed
        and behavior_equivalent
    )
    return {
        "experiment_id": "E2.70-active-natural-skill-acquisition-framework-replay",
        "scientific_role": "framework convergence and cached behavior equivalence",
        "acquisition_path": (
            "EMPTY MEMORY -> SOURCE POLICY EPISODES -> CANDIDATE -> LLM TYPED "
            "PATCH -> TARGET REPLAY -> EXPLICIT PROMOTION"
        ),
        "inputs": {
            "initial_capability_memory_empty": True,
            "source_policy_episode_count": len(source_episodes),
            "cached_natural_target_count": len(target_replays),
            "llm_patch_reused_without_new_api_call": True,
            "new_consumer_fit_count": 0,
            "new_query_outcome_opened": False,
        },
        "candidate": {
            "capability_id": candidate["capability_id"],
            "status_before_patch": candidate["status"],
            "status_after_patch": patched_candidate["status"],
            "patch_id": patched_candidate["applied_patch_id"],
            "workflow_supply": patched_candidate["workflow_supply"],
            "observation": patched_candidate["observation"],
            "control": patched_candidate["control"],
        },
        "target_replays": target_replays,
        "promotion": {
            "status": resolved["status"],
            "result": resolved["promotion_result"],
            "admitted_skill_behavior_equivalent": behavior_equivalent,
        },
        "gate": {
            "candidate_never_auto_promoted": (
                candidate["status"] == "CANDIDATE"
                and patched_candidate["status"] == "CANDIDATE"
            ),
            "all_cached_target_curves_exact_match": (
                exact_match_count == len(target_replays)
            ),
            "harmful_target_count_zero": harmful_target_count == 0,
            "explicit_promotion_only_after_replay": (
                resolved["status"] == "CROSS_DATASET_SUPPORTED"
            ),
            "admitted_behavior_equivalent": behavior_equivalent,
            "passed": passed,
        },
        "verdict": (
            "ACTIVE_NATURAL_SKILL_ACQUISITION_FRAMEWORK_REPLAY_PASS"
            if passed
            else "ACTIVE_NATURAL_SKILL_ACQUISITION_FRAMEWORK_REPLAY_FAIL"
        ),
        "claim_limit": (
            "Cached framework-convergence replay only. It proves that the admitted "
            "natural Skill can be reconstructed through the reusable acquisition core; "
            "it is not new natural evidence or a newly discovered Skill."
        ),
    }


def run_natural_delayed_feedback_vertical(root: Path) -> dict[str, object]:
    """Replay one admitted natural Skill through a strict delayed-feedback wall.

    The Target is fixed as the first entry in the existing natural confirmation
    order.  Fast-Path planning receives only public Context, admitted Skill
    metadata, historical observations and current Support gains.  Cached Query
    outcomes are attached only after the Support plan is frozen.
    """

    admitted = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    active = read_active_skill_cards([admitted])
    if [row["capability_id"] for row in active] != [
        "historical_policy_episode_workflow_v1"
    ]:
        raise ValueError("the admitted natural Forecasting Skill is unavailable")

    saved_plan = _read_object(root / MULTISKILL_LIVE_LLM_FAST_PATH_PLAN_PATH)
    public_context = _multiskill_fast_path_contexts()[0]
    decisions = {
        str(row.get("context_id")): row for row in saved_plan.get("decisions", [])
    }
    llm_decision = decisions.get(str(public_context["context_id"]), {})
    plan_metadata_valid = (
        saved_plan.get("llm_api_integrated") is True
        and llm_decision.get("capability_id") == admitted["capability_id"]
        and llm_decision.get("decision") == "RETRIEVE_AND_PROBE"
        and public_context["task"] == admitted["task_context"]["task"]
        and public_context["consumer"] == admitted["task_context"]["consumer"]
    )
    if not plan_metadata_valid:
        raise ValueError("saved LLM Fast-Path plan is not valid for this Skill")

    # Fixed selection rule: first Target in the already used natural replay order.
    target_report_path = (
        "artifacts/functional/e2/"
        "pedestrian_counts_historical_policy_skill_confirmation_report.json"
    )
    cached = _read_object(root / target_report_path)
    historical_gains = dict(
        cached["historical_policy_context"]["workflow_gains"]
    )
    historical_episode = {
        "workflows": {
            workflow_id: {
                "workflow_id": workflow_id,
                "support_gain": float(gain),
            }
            for workflow_id, gain in historical_gains.items()
        }
    }
    workflow_outcomes = dict(cached["workflow_outcomes"])
    support_only = {
        workflow_id: {"support_gain": float(response["support_gain"])}
        for workflow_id, response in workflow_outcomes.items()
    }

    support_plan = plan_skill_card_support_only(
        admitted,
        lambda workflow_id: support_only[workflow_id],
        historical_episode=historical_episode,
    )

    forbidden_planning_fields: list[str] = []

    def find_evaluator_fields(value: object, path: str = "support_plan") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}"
                lowered = str(key).lower()
                if "query" in lowered or "fixed_query" in lowered or "auc" in lowered:
                    forbidden_planning_fields.append(child_path)
                find_evaluator_fields(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                find_evaluator_fields(child, f"{path}[{index}]")

    find_evaluator_fields(support_plan)
    if forbidden_planning_fields:
        raise ValueError("Support-only plan contains delayed evaluator fields")

    # Simulated time advance: only this evaluator-side mapping contains Query gain.
    delayed_query_gains = {
        workflow_id: float(response["query_gain"])
        for workflow_id, response in workflow_outcomes.items()
    }
    evaluated = attach_delayed_outcomes(support_plan, delayed_query_gains)
    original_curve = list(cached["adaptation_curve"])
    original_auc = policy_episode_adapt_auc(original_curve)
    curve_exact_match = evaluated["adaptation_curve"] == original_curve
    auc_exact_match = float(evaluated["adaptation_auc"]) == float(original_auc)
    passed = (
        plan_metadata_valid
        and not forbidden_planning_fields
        and curve_exact_match
        and auc_exact_match
    )

    delayed_episode = {
        "episode_type": "DELAYED_NATURAL_POLICY_ACTION_RESPONSE",
        "target_context": {
            "task": public_context["task"],
            "consumer": public_context["consumer"],
            "clean_repair_truth_used": False,
        },
        "selected_workflow": support_plan["selected_workflow"],
        "probed_workflows": list(support_plan["probed_workflows"]),
        "support_observations": list(support_plan["support_observations"]),
        "delayed_query_gain": float(
            evaluated["adaptation_curve"][-1]["fixed_query_gain"]
        ),
        "adaptation_auc": float(evaluated["adaptation_auc"]),
        "slow_path_use": "Action-Response evidence after delayed outcome arrival",
        "planning_trace_promotion_eligible": False,
        "delayed_episode_auto_promoted": False,
    }
    return {
        "experiment_id": "E2.85-natural-delayed-feedback-vertical-replay",
        "scientific_role": "natural delayed-feedback information-wall replay",
        "inputs": {
            "target_selection_rule": (
                "first Target in existing Pedestrian/Rideshare/KDD replay order"
            ),
            "target_report": target_report_path,
            "capability": HISTORICAL_POLICY_CAPABILITY_PATH,
            "saved_live_llm_plan": MULTISKILL_LIVE_LLM_FAST_PATH_PLAN_PATH,
            "new_llm_api_call_count": 0,
            "new_consumer_fit_count": 0,
            "new_query_outcome_opened": False,
            "repair_truth_used": False,
        },
        "llm_fast_path": {
            "public_context": public_context,
            "retrieved_skill": llm_decision.get("capability_id"),
            "decision": llm_decision.get("decision"),
            "saved_plan_reused": True,
            "metadata_contract_valid": plan_metadata_valid,
        },
        "support_only_runtime": {
            "plan": support_plan,
            "forbidden_delayed_fields": forbidden_planning_fields,
            "contains_query_or_auc": bool(forbidden_planning_fields),
            "promotion_eligible": False,
        },
        "delayed_evaluator": {
            "original_curve": original_curve,
            "attached_curve": evaluated["adaptation_curve"],
            "original_auc_derived_from_curve": float(original_auc),
            "attached_auc": float(evaluated["adaptation_auc"]),
            "curve_exact_match": curve_exact_match,
            "auc_exact_match": auc_exact_match,
        },
        "delayed_policy_episode": delayed_episode,
        "gate": {
            "support_plan_has_no_query_or_auc": not forbidden_planning_fields,
            "delayed_curve_exactly_matches_natural_report": curve_exact_match,
            "delayed_auc_exactly_matches_natural_curve": auc_exact_match,
            "passed": passed,
        },
        "verdict": (
            "NATURAL_DELAYED_FEEDBACK_VERTICAL_PASS"
            if passed
            else "NATURAL_DELAYED_FEEDBACK_VERTICAL_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Cached natural replay only. It proves deployment-facing information "
            "separation and behavior equivalence; it is not fresh Target evidence, "
            "clean-repair truth, a new Skill, or a new Promotion result."
        ),
    }


def run_live_workflow_discovery_acquisition_replay(
    root: Path,
    *,
    model: str = "gpt-5.6-luna",
    base_url: str = "https://api.agicto.cn/v1",
) -> dict[str, object]:
    """Connect public Workspace Context to the natural acquisition cycle.

    The live planner only exposes a bounded Workflow supply.  Completed Source
    PolicyEpisodes still determine the response-based prior, current Support still
    confirms execution, and cached natural Target replay still owns admission.
    This is framework-convergence evidence rather than a new Skill claim.
    """

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")

    admitted = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    task_context = dict(admitted["task_context"])
    public_context = {
        "task_context": task_context,
        "workspace": {
            "regular_panel_available": True,
            "multiple_training_series_available": True,
            "known_seasonal_period_available": True,
            "phase_aligned_historical_origins_available": True,
            "current_support_feedback_available": True,
        },
        "information_boundary": "support_only_before_final_evaluation",
        "adaptation_goal": {
            "minimize_target_feedback_trials": True,
            "preserve_identity_fallback": True,
            "retain_mechanistically_distinct_candidates_when_response_is_unknown": True,
        },
    }
    workflow_catalog = [
        {
            "workflow_id": "W_rowblock",
            "description": (
                "locate and prepare bounded temporal blocks within training series"
            ),
            "mechanism": "local_interval_training_data_preparation",
            "public_parameter_bindings": {},
        },
        {
            "workflow_id": "W_curation",
            "description": (
                "retain, attenuate, or exclude complete donor-series groups"
            ),
            "mechanism": "cross_series_training_cohort_curation",
            "public_parameter_bindings": {},
        },
        {
            "workflow_id": "W_temporal_origin",
            "description": (
                "retain, attenuate, or exclude phase-aligned historical origin groups"
            ),
            "mechanism": "temporal_origin_training_cohort_curation",
            "public_parameter_bindings": {},
        },
    ]
    observation_catalog = [
        {
            "observation_id": "cohort_overview",
            "description": "summarize panel composition and training-group availability",
        },
        {
            "observation_id": "phase_aligned_historical_policy_episode",
            "description": "retrieve prior support-visible responses at aligned origins",
        },
        {
            "observation_id": "current_support_policy_response",
            "description": "confirm a proposed Workflow on current Support",
        },
    ]
    planner_trace: dict[str, object] = {}

    def planner(payload: dict[str, object]) -> dict[str, object]:
        import openai

        system_prompt = (
            "You are the Workflow-supply planner inside a time-series data adaptation "
            "Harness. Select only catalog Workflows that are mechanically applicable "
            "to the public Workspace Context. You do not observe action responses and "
            "must not guess which Workflow is beneficial. When all distinct templates "
            "are applicable and there is no response evidence, preserve breadth by "
            "selecting all of them in catalog order; later Harness stages will rank and "
            "confirm them. Use only catalog observations, keep IDENTITY fallback, and "
            "return exactly one JSON object with only: decision, selected_workflows, "
            "probe_order, requested_observations, fallback. Each selected_workflows "
            "entry must contain workflow_id and bindings. decision must be the literal "
            "string PROPOSE when selecting Workflows, or ABSTAIN otherwise. Return no "
            "markdown, rationale, or alternative decision label."
        )
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        planner_trace["provider"] = {
            "base_url": base_url,
            "requested_model": model,
            "returned_model": getattr(completion, "model", ""),
            "prompt_tokens": getattr(
                getattr(completion, "usage", None), "prompt_tokens", None
            ),
            "completion_tokens": getattr(
                getattr(completion, "usage", None), "completion_tokens", None
            ),
        }
        proposal = json.loads(completion.choices[0].message.content or "")
        if not isinstance(proposal, dict):
            raise ValueError("Workflow planner must return one JSON object")
        planner_trace["proposal"] = proposal
        return proposal

    discovery = discover_workflow_supply(
        public_context,
        workflow_catalog,
        observation_catalog,
        planner,
        max_candidates=3,
    )
    if discovery["decision"] != "PROPOSE":
        return {
            "experiment_id": "E2.71-live-workflow-discovery-to-acquisition",
            "scientific_role": "framework convergence with fail-closed live planning",
            "public_context": public_context,
            "planner": planner_trace,
            "discovery": discovery,
            "acquisition_replay": None,
            "verdict": "WORKFLOW_DISCOVERY_ABSTAINED_NO_CANDIDATE",
            "capability_or_memory_written": False,
            "claim_limit": (
                "No Skill was created because the bounded Workflow proposal did not "
                "compile. This is a safe framework result, not Capability evidence."
            ),
        }

    replay = run_skill_acquisition_framework_replay(
        root,
        workflow_supply_override=tuple(
            str(value) for value in discovery["workflow_supply"]
        ),
        compiled_workflows_override=tuple(
            dict(value) for value in discovery["compiled_workflows"]
        ),
    )
    passed = bool(replay["gate"]["passed"])
    return {
        "experiment_id": "E2.71-live-workflow-discovery-to-acquisition",
        "scientific_role": (
            "live public-Context Workflow supply plus cached natural acquisition replay"
        ),
        "framework_path": (
            "PUBLIC WORKSPACE CONTEXT -> LIVE BOUNDED WORKFLOW DISCOVERY -> "
            "SOURCE POLICY EPISODES -> CANDIDATE -> TYPED PATCH -> TARGET REPLAY -> "
            "EXPLICIT PROMOTION OR REJECTION"
        ),
        "public_context": public_context,
        "planner": planner_trace,
        "discovery": discovery,
        "acquisition_replay": replay,
        "gate": {
            "catalog_only_supply": set(discovery["workflow_supply"]).issubset(
                {row["workflow_id"] for row in workflow_catalog}
            ),
            "candidate_not_created_by_planner": (
                discovery["candidate_status"] == "DISCOVERED_NOT_EVALUATED"
            ),
            "cached_natural_behavior_reconstructed": passed,
            "passed": passed,
        },
        "verdict": (
            "LIVE_WORKFLOW_DISCOVERY_ACQUISITION_FRAMEWORK_PASS"
            if passed
            else "LIVE_WORKFLOW_DISCOVERY_ACQUISITION_FRAMEWORK_REJECTED"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed-cache framework replay only. It proves that a live LLM can "
            "construct a bounded Workflow supply which the existing evidence-driven "
            "acquisition loop can accept or reject; it is not a new natural Skill or "
            "fresh transfer result."
        ),
    }


def run_live_failure_driven_skill_evolution(
    root: Path,
    *,
    model: str = "gpt-5.5",
    base_url: str = "https://api.agicto.cn/v1",
) -> dict[str, object]:
    """Diagnose natural Policy failures and request one live bounded patch."""

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")

    admitted = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    workflow_supply = tuple(str(value) for value in admitted["workflow_supply"])
    global_failure = _read_object(
        root
        / "artifacts/functional/e2/electricity_hourly_policy_memory_promotion_report.json"
    )
    overwrite_failure = _read_object(
        root
        / "artifacts/functional/e2/m4_hourly_historical_policy_context_promotion_report.json"
    )
    source_episodes = [
        {"workflows": dict(global_failure["workflow_outcomes"])},
        {"workflows": dict(overwrite_failure["workflow_outcomes"])},
    ]
    base_candidate = build_candidate_skill(
        [],
        source_episodes,
        capability_id="failure_driven_historical_policy_candidate_v1",
        task_context=dict(admitted["task_context"]),
        workflow_supply=workflow_supply,
    )

    global_order = tuple(str(value) for value in global_failure["source_memory_order"])
    global_curve = workflow_curve_from_policy_episode(
        dict(global_failure["workflow_outcomes"]), global_order
    )
    historical_order = tuple(
        str(value)
        for value in overwrite_failure["historical_policy_context"]["compiled_order"]
    )
    overwrite_curve = workflow_curve_from_policy_episode(
        dict(overwrite_failure["workflow_outcomes"]), historical_order
    )
    failure_cases = [
        {
            "candidate_probe_order": list(global_order),
            "workflow_responses": dict(global_failure["workflow_outcomes"]),
            "candidate_curve": global_curve,
            "comparison_adaptation_auc": float(global_failure["adapt_auc"]["A3"]),
        },
        {
            "candidate_probe_order": list(historical_order),
            "workflow_responses": dict(overwrite_failure["workflow_outcomes"]),
            "candidate_curve": overwrite_curve,
        },
    ]
    dossier = build_policy_failure_dossier(
        failure_cases,
        allowed_observations=[
            "source_policy_episode_workflow_prior",
            "phase_aligned_historical_policy_episode",
        ],
        allowed_controls=[
            "keep_best_support_so_far",
            "stop_on_first_positive",
        ],
    )
    payload = {
        "failure_dossier": dossier,
        "current_candidate": {
            "observation": base_candidate["observation"],
            "control": base_candidate["control"],
            "risk": base_candidate["risk"],
            "workflow_count": len(base_candidate["workflow_supply"]),
        },
        "required_output": {
            "patch_id": "short identifier",
            "operations": [
                {
                    "operation": "one operation required by a supplied first fault",
                    "target_surface": "the corresponding supplied surface",
                    "value": "one value from allowed_patch_values",
                }
            ],
        },
    }
    system_prompt = (
        "You are the bounded Slow-Path patch proposer in a time-series data "
        "adaptation Harness. Repair every supplied categorical first fault with "
        "exactly one corresponding operation. Select values only from "
        "allowed_patch_values. Prefer the smallest patch that changes the observed "
        "candidate behavior. Do not invent Workflows, observations, controls, data, "
        "thresholds, effects or Skills. Do not change Program supply, Consumer, metric, "
        "Memory schema or visibility. Return exactly one JSON object containing only "
        "patch_id and operations, with no markdown or rationale."
    )
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    provider = {
        "base_url": base_url,
        "requested_model": model,
        "returned_model": getattr(completion, "model", ""),
        "prompt_tokens": getattr(
            getattr(completion, "usage", None), "prompt_tokens", None
        ),
        "completion_tokens": getattr(
            getattr(completion, "usage", None), "completion_tokens", None
        ),
    }
    raw_proposal = json.loads(completion.choices[0].message.content or "")
    if not isinstance(raw_proposal, dict):
        raise ValueError("Slow-Path patch proposer must return one JSON object")

    violations: list[str] = []
    normalized_patch: dict[str, object] | None = None
    patched_candidate: dict[str, object] | None = None
    acquisition_replay: dict[str, object] | None = None
    try:
        normalized_patch = validate_failure_driven_patch(
            raw_proposal, base_candidate, dossier
        )
        patched_candidate = apply_typed_patch(base_candidate, normalized_patch)
        changed = (
            patched_candidate["observation"] != base_candidate["observation"]
            and patched_candidate["control"] != base_candidate["control"]
            and patched_candidate["risk"] != base_candidate["risk"]
        )
        if not changed:
            raise ValueError("typed patch did not change all diagnosed behavior surfaces")
        acquisition_replay = run_skill_acquisition_framework_replay(
            root,
            typed_patch_override=normalized_patch,
        )
    except (KeyError, TypeError, ValueError) as exc:
        violations.append(str(exc))

    global_auc_exact = abs(
        policy_episode_adapt_auc(global_curve)
        - float(global_failure["adapt_auc"]["A5"])
    ) <= 1e-12
    overwrite_curve_exact = [
        (row["selected_workflow"], float(row["fixed_query_gain"]))
        for row in overwrite_curve
    ] == [
        (row["selected_workflow"], float(row["fixed_query_gain"]))
        for row in overwrite_failure["adaptation_curve"]
    ]
    replay_passed = bool(
        acquisition_replay is not None
        and acquisition_replay["gate"]["passed"]
    )
    passed = (
        not violations
        and global_auc_exact
        and overwrite_curve_exact
        and replay_passed
    )
    return {
        "experiment_id": "E2.72-live-failure-driven-natural-skill-evolution",
        "scientific_role": (
            "exposed-cache mechanism test of automatic failure diagnosis, live typed "
            "patch proposal, deterministic replay and explicit admission"
        ),
        "framework_path": (
            "PRE-EVOLUTION CANDIDATE -> CATEGORICAL FAILURE DOSSIER -> LIVE LLM "
            "TYPED PATCH -> LOCAL COMPILER -> CACHED NATURAL TARGET REPLAY -> "
            "EXPLICIT PROMOTION OR REJECTION"
        ),
        "privacy": dossier["privacy"],
        "failure_dossier": dossier,
        "pre_patch_candidate": {
            "status": base_candidate["status"],
            "observation": base_candidate["observation"],
            "control": base_candidate["control"],
            "risk": base_candidate["risk"],
        },
        "llm": {"provider": provider, "raw_proposal": raw_proposal},
        "compiled_patch": normalized_patch,
        "post_patch_candidate": (
            None
            if patched_candidate is None
            else {
                "status": patched_candidate["status"],
                "observation": patched_candidate["observation"],
                "control": patched_candidate["control"],
                "risk": patched_candidate["risk"],
            }
        ),
        "acquisition_replay": acquisition_replay,
        "validation": {
            "contract_violations": violations,
            "global_failure_curve_reconstructed": global_auc_exact,
            "overwrite_failure_curve_reconstructed": overwrite_curve_exact,
            "cached_natural_replay_passed": replay_passed,
            "passed": passed,
        },
        "verdict": (
            "LIVE_FAILURE_DRIVEN_SKILL_EVOLUTION_FRAMEWORK_PASS"
            if passed
            else "LIVE_FAILURE_DRIVEN_SKILL_EVOLUTION_FRAMEWORK_REJECTED"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed-cache mechanism result. A PASS proves that the active framework "
            "can derive and validate the admitted natural Skill patch from measured "
            "failures without loading the saved patch; it is not a new Skill or fresh "
            "Target transfer result."
        ),
    }


def run_reversible_target_representation_p0(
    root: Path,
    *,
    specs_override: dict[str, dict[str, object]] | None = None,
    experiment_id: str = "E2-natural-reversible-target-representation-P0",
) -> dict[str, object]:
    """Measure natural headroom for two reversible forecasting representations.

    The Consumer, roster geometry, windows and metric stay fixed.  The only changed
    surface is Program Supply: predict the raw normalized target, a residual over a
    seasonal-naive baseline, or a residual over the last visible value.  Both
    residual Programs use only the visible context and add their baseline back after
    prediction.
    """

    import numpy as np
    from sklearn.linear_model import Ridge

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
        read_registry_jsonl,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
        _center_scale,
    )

    programs = (
        "IDENTITY_RAW_TARGET",
        "SEASONAL_RESIDUAL_TARGET",
        "LAST_VALUE_RESIDUAL_TARGET",
    )
    nonidentity = programs[1:]
    material_gain = 0.005
    specs = specs_override or {**SPECS, **FRESH_SPECS}
    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    folds = _alternating_folds(8)
    dataset_rows: list[dict[str, object]] = []
    consumer_fit_count = 0

    for dataset_id, spec in specs.items():
        required_stop = max(max(ANCHORS) + HORIZON, int(spec["future_bounds"][1]))
        candidates = sorted(
            (
                row
                for row in registry_rows
                if row.dataset_id == dataset_id and int(row.length) >= required_stop
            ),
            key=lambda row: row.series_uid,
        )
        candidate_values = _load_values(
            candidates, root / "data/benchmark_v0_2/clean_base"
        )
        eligible = [
            row
            for row in candidates
            if np.isfinite(
                np.asarray(
                    candidate_values[row.series_uid][48:required_stop],
                    dtype=np.float64,
                )
            ).all()
        ]
        roster_start = int(spec.get("roster_start", 0))
        if roster_start < 0 or len(eligible) < roster_start + 20:
            raise ValueError(f"insufficient finite series for {dataset_id}")
        selected_roster = eligible[roster_start : roster_start + 20]
        train_records = selected_roster[:12]
        eval_records = selected_roster[12:20]
        selected_values = _load_values(
            train_records + eval_records,
            root / "data/benchmark_v0_2/clean_base",
        )
        period = int(spec["period"])

        x_train_rows: list[Any] = []
        raw_target_rows: list[Any] = []
        seasonal_baseline_rows: list[Any] = []
        last_value_baseline_rows: list[Any] = []
        horizon_offsets = np.arange(HORIZON, dtype=np.int64) % period
        for anchor in ANCHORS:
            for record in train_records:
                raw = selected_values[record.series_uid]
                context = np.asarray(
                    raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64
                )
                target = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
                center, scale, method = _center_scale(context)
                if method == "scale_floor_fallback":
                    raise ValueError(
                        f"training scale floor reached: {dataset_id}/{record.series_uid}"
                    )
                normalized_context = (context - center) / scale
                x_train_rows.append(
                    np.concatenate(
                        (normalized_context, np.zeros(CONTEXT_LENGTH, dtype=np.float64))
                    )
                )
                raw_target_rows.append((target - center) / scale)
                seasonal_baseline_rows.append(
                    normalized_context[-period + horizon_offsets]
                )
                last_value_baseline_rows.append(
                    np.full(HORIZON, normalized_context[-1], dtype=np.float64)
                )
        x_train = np.asarray(x_train_rows, dtype=np.float64)
        raw_targets = np.asarray(raw_target_rows, dtype=np.float64)
        seasonal_train = np.asarray(seasonal_baseline_rows, dtype=np.float64)
        last_train = np.asarray(last_value_baseline_rows, dtype=np.float64)
        if (
            x_train.shape != (72, 2 * CONTEXT_LENGTH)
            or raw_targets.shape != (72, HORIZON)
            or seasonal_train.shape != raw_targets.shape
            or last_train.shape != raw_targets.shape
        ):
            raise AssertionError(f"unexpected training geometry: {dataset_id}")

        x_eval_rows: list[Any] = []
        raw_futures: list[Any] = []
        eval_centers: list[float] = []
        eval_scales: list[float] = []
        eval_seasonal: list[Any] = []
        eval_last: list[Any] = []
        metric_scales: list[float] = []
        for record in eval_records:
            raw = selected_values[record.series_uid]
            train_stop = int(spec["train_stop"])
            context = np.asarray(
                raw[train_stop - CONTEXT_LENGTH : train_stop], dtype=np.float64
            )
            future = np.asarray(raw[slice(*spec["future_bounds"])], dtype=np.float64)
            center, scale, method = _center_scale(context)
            if method == "scale_floor_fallback":
                raise ValueError(
                    f"evaluation scale floor reached: {dataset_id}/{record.series_uid}"
                )
            try:
                metric_scale = seasonal_scale(
                    np.asarray(raw[:train_stop], dtype=np.float64),
                    np.isfinite(raw[:train_stop]),
                    period=period,
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(
                    f"invalid evaluation scale: {dataset_id}/{record.series_uid}"
                ) from error
            normalized_context = (context - center) / scale
            x_eval_rows.append(
                np.concatenate(
                    (normalized_context, np.zeros(CONTEXT_LENGTH, dtype=np.float64))
                )
            )
            raw_futures.append(future)
            eval_centers.append(center)
            eval_scales.append(scale)
            eval_seasonal.append(normalized_context[-period + horizon_offsets])
            eval_last.append(
                np.full(HORIZON, normalized_context[-1], dtype=np.float64)
            )
            metric_scales.append(metric_scale)

        x_eval = np.asarray(x_eval_rows, dtype=np.float64)
        raw_future_array = np.asarray(raw_futures, dtype=np.float64)
        centers_array = np.asarray(eval_centers, dtype=np.float64)
        scales_array = np.asarray(eval_scales, dtype=np.float64)
        eval_baselines = {
            "IDENTITY_RAW_TARGET": np.zeros((8, HORIZON), dtype=np.float64),
            "SEASONAL_RESIDUAL_TARGET": np.asarray(
                eval_seasonal, dtype=np.float64
            ),
            "LAST_VALUE_RESIDUAL_TARGET": np.asarray(eval_last, dtype=np.float64),
        }
        phase_lag = ((HORIZON + period - 1) // period) * period
        historical_surfaces: list[dict[str, object]] = []
        for historical_origin in (
            int(spec["train_stop"]) - 2 * phase_lag,
            int(spec["train_stop"]) - phase_lag,
        ):
            historical_x: list[Any] = []
            historical_future: list[Any] = []
            historical_centers: list[float] = []
            historical_scales: list[float] = []
            historical_metric_scales: list[float] = []
            historical_seasonal: list[Any] = []
            historical_last: list[Any] = []
            for record in eval_records:
                raw = selected_values[record.series_uid]
                context = np.asarray(
                    raw[
                        historical_origin - CONTEXT_LENGTH : historical_origin
                    ],
                    dtype=np.float64,
                )
                future = np.asarray(
                    raw[historical_origin : historical_origin + HORIZON],
                    dtype=np.float64,
                )
                center, scale, method = _center_scale(context)
                if method == "scale_floor_fallback":
                    raise ValueError(
                        "historical evaluation scale floor reached: "
                        f"{dataset_id}/{record.series_uid}/{historical_origin}"
                    )
                try:
                    metric_scale = seasonal_scale(
                        np.asarray(raw[:historical_origin], dtype=np.float64),
                        np.isfinite(raw[:historical_origin]),
                        period=period,
                        min_pairs=32,
                    )
                except (UndefinedSeasonalScale, ValueError) as error:
                    raise ValueError(
                        "invalid historical evaluation scale: "
                        f"{dataset_id}/{record.series_uid}/{historical_origin}"
                    ) from error
                normalized_context = (context - center) / scale
                historical_x.append(
                    np.concatenate(
                        (
                            normalized_context,
                            np.zeros(CONTEXT_LENGTH, dtype=np.float64),
                        )
                    )
                )
                historical_future.append(future)
                historical_centers.append(center)
                historical_scales.append(scale)
                historical_metric_scales.append(metric_scale)
                historical_seasonal.append(
                    normalized_context[-period + horizon_offsets]
                )
                historical_last.append(
                    np.full(HORIZON, normalized_context[-1], dtype=np.float64)
                )
            historical_surfaces.append(
                {
                    "origin": historical_origin,
                    "x": np.asarray(historical_x, dtype=np.float64),
                    "future": np.asarray(historical_future, dtype=np.float64),
                    "centers": np.asarray(historical_centers, dtype=np.float64),
                    "scales": np.asarray(historical_scales, dtype=np.float64),
                    "metric_scales": historical_metric_scales,
                    "baselines": {
                        "IDENTITY_RAW_TARGET": np.zeros(
                            (8, HORIZON), dtype=np.float64
                        ),
                        "SEASONAL_RESIDUAL_TARGET": np.asarray(
                            historical_seasonal, dtype=np.float64
                        ),
                        "LAST_VALUE_RESIDUAL_TARGET": np.asarray(
                            historical_last, dtype=np.float64
                        ),
                    },
                    "losses": {},
                }
            )
        train_targets = {
            "IDENTITY_RAW_TARGET": raw_targets,
            "SEASONAL_RESIDUAL_TARGET": raw_targets - seasonal_train,
            "LAST_VALUE_RESIDUAL_TARGET": raw_targets - last_train,
        }
        losses: dict[str, list[float]] = {}
        raw_mae: dict[str, list[float]] = {}
        for program in programs:
            model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
            model.fit(x_train, train_targets[program])
            consumer_fit_count += 1
            normalized_prediction = (
                np.asarray(model.predict(x_eval), dtype=np.float64)
                + eval_baselines[program]
            )
            original_prediction = (
                normalized_prediction * scales_array[:, None]
                + centers_array[:, None]
            )
            if not np.isfinite(original_prediction).all():
                raise RuntimeError(f"non-finite prediction: {dataset_id}/{program}")
            losses[program] = [
                smase(
                    raw_future_array[index],
                    original_prediction[index],
                    scale=metric_scales[index],
                )
                for index in range(8)
            ]
            raw_mae[program] = [
                float(
                    np.mean(
                        np.abs(raw_future_array[index] - original_prediction[index])
                    )
                )
                for index in range(8)
            ]
            for historical in historical_surfaces:
                historical_normalized = (
                    np.asarray(
                        model.predict(historical["x"]), dtype=np.float64
                    )
                    + historical["baselines"][program]
                )
                historical_prediction = (
                    historical_normalized * historical["scales"][:, None]
                    + historical["centers"][:, None]
                )
                historical["losses"][program] = [
                    smase(
                        historical["future"][index],
                        historical_prediction[index],
                        scale=historical["metric_scales"][index],
                    )
                    for index in range(8)
                ]

        identity_loss = statistics.fmean(losses["IDENTITY_RAW_TARGET"])
        full_gains = {
            program: identity_loss - statistics.fmean(losses[program])
            for program in nonidentity
        }
        fold_replays: list[dict[str, object]] = []
        for direction, support_name, query_name in FOLD_PAIRS:
            support_indices = folds[support_name]
            query_indices = folds[query_name]
            support_gains = {
                program: statistics.fmean(
                    losses["IDENTITY_RAW_TARGET"][index] - losses[program][index]
                    for index in support_indices
                )
                for program in nonidentity
            }
            selected = max(nonidentity, key=lambda name: support_gains[name])
            if support_gains[selected] <= 0.0:
                selected = "IDENTITY_RAW_TARGET"
            query_gain = (
                0.0
                if selected == "IDENTITY_RAW_TARGET"
                else statistics.fmean(
                    losses["IDENTITY_RAW_TARGET"][index] - losses[selected][index]
                    for index in query_indices
                )
            )
            fold_replays.append(
                {
                    "direction": direction,
                    "support_gains": support_gains,
                    "selected_program": selected,
                    "query_gain": query_gain,
                    "harmful": query_gain < 0.0,
                }
            )
        seasonal_baseline_error = float(
            np.mean(np.abs(raw_targets - seasonal_train))
        )
        last_value_baseline_error = float(np.mean(np.abs(raw_targets - last_train)))
        baseline_gap = seasonal_baseline_error - last_value_baseline_error
        baseline_tolerance = 0.05 * max(
            seasonal_baseline_error, last_value_baseline_error, 1e-12
        )
        if abs(baseline_gap) <= baseline_tolerance:
            baseline_fit_category = "SIMILAR"
        elif baseline_gap < 0.0:
            baseline_fit_category = "SEASONAL_BASELINE_BETTER"
        else:
            baseline_fit_category = "LAST_VALUE_BASELINE_BETTER"

        def outcome_category(gain: float) -> str:
            if gain >= material_gain:
                return "SUPPORTED"
            if gain <= -material_gain:
                return "CONTRAINDICATED"
            return "NEAR_NEUTRAL"

        fold_gain_signs = [
            "POSITIVE"
            if float(row["query_gain"]) > 0.0
            else "NEGATIVE"
            if float(row["query_gain"]) < 0.0
            else "ABSTAINED_OR_ZERO"
            for row in fold_replays
        ]
        support_response_stability = (
            "CONSISTENT_POSITIVE"
            if fold_gain_signs == ["POSITIVE", "POSITIVE"]
            else "CONFLICTING"
            if "POSITIVE" in fold_gain_signs and "NEGATIVE" in fold_gain_signs
            else "CONTRAINDICATED_OR_UNRESOLVED"
        )
        historical_policy_episodes = []
        for historical in historical_surfaces:
            historical_identity = statistics.fmean(
                historical["losses"]["IDENTITY_RAW_TARGET"]
            )
            historical_policy_episodes.append(
                {
                    "relative_origin": int(historical["origin"])
                    - int(spec["train_stop"]),
                    "future_ends_before_current_query_cutoff": (
                        int(historical["origin"]) + HORIZON
                        <= int(spec["train_stop"])
                    ),
                    "gain_vs_identity": {
                        program: historical_identity
                        - statistics.fmean(historical["losses"][program])
                        for program in nonidentity
                    },
                }
            )
        dataset_rows.append(
            {
                "dataset": dataset_id,
                "period": period,
                "training_series_count": 12,
                "evaluation_series_count": 8,
                "development_roster_start": roster_start,
                "loss": {
                    program: {
                        "mean_smase": statistics.fmean(losses[program]),
                        "median_smase": statistics.median(losses[program]),
                        "mean_raw_mae": statistics.fmean(raw_mae[program]),
                    }
                    for program in programs
                },
                "full_evaluation_gain_vs_identity": full_gains,
                "per_series_gain_vs_identity_for_local_replay": {
                    program: [
                        losses["IDENTITY_RAW_TARGET"][index]
                        - losses[program][index]
                        for index in range(8)
                    ]
                    for program in nonidentity
                },
                "deploy_time_context": {
                    "period_bucket": (
                        "SHORT" if period <= 7 else "MEDIUM" if period <= 12 else "LONG"
                    ),
                    "training_only_baseline_fit": baseline_fit_category,
                },
                "categorical_action_response": {
                    program: outcome_category(float(full_gains[program]))
                    for program in nonidentity
                },
                "support_response_stability": support_response_stability,
                "phase_aligned_historical_policy_episodes": (
                    historical_policy_episodes
                ),
                "menu_oracle_gain": max(0.0, *full_gains.values()),
                "identity_optimal": max(full_gains.values()) <= 0.0,
                "support_to_query_replays": fold_replays,
                "support_selected_query_gain": statistics.fmean(
                    float(row["query_gain"]) for row in fold_replays
                ),
            }
        )

    positive_datasets = [
        str(row["dataset"])
        for row in dataset_rows
        if float(row["menu_oracle_gain"]) >= material_gain
    ]
    identity_datasets = [
        str(row["dataset"])
        for row in dataset_rows
        if bool(row["identity_optimal"])
    ]
    support_positive_datasets = [
        str(row["dataset"])
        for row in dataset_rows
        if float(row["support_selected_query_gain"]) > 0.0
    ]
    harmful_replays = sum(
        bool(replay["harmful"])
        for row in dataset_rows
        for replay in row["support_to_query_replays"]
    )
    passed = len(positive_datasets) >= 2 and len(identity_datasets) >= 1
    sanitized_episodes = [
        {
            "episode": f"anonymous_environment_{index + 1}",
            "visible_context": row["deploy_time_context"],
            "action_response": row["categorical_action_response"],
            "one_support_probe_transport": row["support_response_stability"],
        }
        for index, row in enumerate(dataset_rows)
    ]
    return {
        "experiment_id": experiment_id,
        "scientific_role": "exposed natural Program-headroom and matched-risk test",
        "exposure": {
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "used_for": "development Program-family premise only",
        },
        "hypothesis": (
            "A reversible TS-native target representation can improve a fixed shared "
            "Forecasting Consumer on some natural cohorts while raw-target Identity "
            "remains optimal on others."
        ),
        "protocol": {
            "task": "forecasting",
            "consumer": "unchanged Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE with dataset-macro interpretation",
            "programs": list(programs),
            "program_geometry": (
                "whole training/evaluation target representation; baselines use only "
                "the visible context and are deterministically added back"
            ),
            "train_windows_per_dataset": 72,
            "evaluation_series_per_dataset": 8,
            "cross_fold_selection": "alternating 4-series Support/Query folds",
            "target_query_opened": False,
        },
        "datasets": dataset_rows,
        "sanitized_failure_dossier_candidate": {
            "status": "LOCAL_ONLY_NOT_YET_AUTHORIZED_FOR_EXTERNAL_EGRESS",
            "privacy": {
                "raw_time_series_included": False,
                "dataset_identity_included": False,
                "effect_magnitude_included": False,
                "internal_file_content_included": False,
            },
            "task_consumer": {
                "task": "forecasting",
                "consumer": "fixed shared forecasting Consumer",
            },
            "program_family": {
                "mechanism": "reversible target residual representation",
                "actions": list(nonidentity),
                "fallback": "IDENTITY_RAW_TARGET",
            },
            "anonymous_action_response_episodes": sanitized_episodes,
            "observed_first_faults": [
                {
                    "code": "REPRESENTATION_EFFECT_IS_CONTEXT_CONDITIONED",
                    "description": (
                        "the same reversible representation is supported in some "
                        "natural environments and contraindicated in others"
                    ),
                },
                {
                    "code": "ONE_SUPPORT_PROBE_CAN_FALSELY_CONFIRM",
                    "description": (
                        "a positive Support response can transport negatively to the "
                        "paired evaluation cohort"
                    ),
                },
            ],
            "allowed_patch_surfaces": [
                "ADD_OBSERVATION",
                "COMPOSE_WORKFLOW",
                "RESTRICT_SCOPE",
                "ADD_RISK",
            ],
            "forbidden_changes": [
                "program_supply",
                "consumer",
                "metric",
                "memory_schema",
                "query_visibility",
            ],
            "required_role_of_llm": (
                "diagnose one earliest Harness fault and propose one bounded typed "
                "patch; local evidence, not the LLM, accepts or rejects it"
            ),
        },
        "summary": {
            "dataset_count": len(dataset_rows),
            "material_gain_threshold": material_gain,
            "material_headroom_dataset_count": len(positive_datasets),
            "material_headroom_datasets": positive_datasets,
            "identity_optimal_dataset_count": len(identity_datasets),
            "identity_optimal_datasets": identity_datasets,
            "support_selected_positive_dataset_count": len(
                support_positive_datasets
            ),
            "support_selected_positive_datasets": support_positive_datasets,
            "harmful_support_to_query_replay_count": harmful_replays,
            "dataset_macro_menu_oracle_gain": statistics.fmean(
                float(row["menu_oracle_gain"]) for row in dataset_rows
            ),
            "dataset_macro_support_selected_query_gain": statistics.fmean(
                float(row["support_selected_query_gain"]) for row in dataset_rows
            ),
        },
        "compute": {
            "consumer_fit_count": consumer_fit_count,
            "llm_api_call_count": 0,
        },
        "gate": {
            "at_least_two_material_headroom_datasets": len(positive_datasets) >= 2,
            "at_least_one_identity_optimal_dataset": len(identity_datasets) >= 1,
            "passed": passed,
        },
        "verdict": (
            "REVERSIBLE_TARGET_REPRESENTATION_HEADROOM_AND_RISK_PASS"
            if passed
            else "REVERSIBLE_TARGET_REPRESENTATION_PREMISE_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed P0 only. A pass establishes Program headroom and matched risk, "
            "not a Context rule, Skill, Memory benefit, LLM update or Target transfer."
        ),
    }


def run_reversible_representation_llm_patch_proposal(
    root: Path,
    *,
    model: str,
    base_url: str,
) -> dict[str, object]:
    """Ask one live LLM to diagnose the new anonymized natural failure episode."""

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    p0 = _read_object(root / REVERSIBLE_TARGET_REPRESENTATION_P0_REPORT_PATH)
    if p0.get("verdict") != (
        "REVERSIBLE_TARGET_REPRESENTATION_HEADROOM_AND_RISK_PASS"
    ):
        raise ValueError("representation P0 did not establish a patchable premise")
    dossier = dict(p0["sanitized_failure_dossier_candidate"])
    dossier["status"] = "USER_AUTHORIZED_SANITIZED_EXTERNAL_EGRESS"
    privacy = dict(dossier.get("privacy", {}))
    if any(bool(value) for value in privacy.values()):
        raise ValueError("sanitized Dossier privacy boundary is not closed")

    allowed_primitives = [
        "READ_TRAINING_CONTEXT",
        "READ_PHASE_ALIGNED_HISTORICAL_POLICY_EPISODES",
        "PROBE_CURRENT_TARGET_SUPPORT",
        "ORDER_PROGRAMS",
        "EXECUTE_PROGRAM",
        "ABSTAIN",
    ]
    payload = {
        "failure_dossier": dossier,
        "runtime_primitives": allowed_primitives,
        "design_constraints": [
            "Choose exactly one primary Harness surface from the supplied allowed surfaces.",
            "Do not infer a fixed period-bucket-to-action router from four environments.",
            "The patch must preserve useful representation actions while addressing false-positive Support transport.",
            "Only training-visible context, historical action-response ending before the current Query cutoff, and current Target Support labels are legal.",
            "Current Query future is forbidden for ordering, confirmation, compilation and acceptance.",
            "The LLM proposes; deterministic local replay accepts or rejects the patch.",
        ],
        "required_output": {
            "patch_id": "short identifier",
            "diagnosed_first_fault": "one supplied fault code",
            "primary_operation": "one supplied allowed patch surface",
            "diagnosis": "short causal explanation",
            "observation_requirements": [
                {
                    "name": "descriptive name",
                    "evidence_source": "one legal runtime primitive",
                    "scope": "what is observed without dataset identity",
                }
            ],
            "workflow_steps": [
                {
                    "step": 1,
                    "primitive": "one supplied runtime primitive",
                    "behavior": "short typed behavior",
                }
            ],
            "scope_condition": "context condition or UNCHANGED",
            "risk_guards": ["typed abstention condition"],
            "minimum_target_feedback_units": "integer 0 to 3",
            "expected_behavior_change": "short description",
            "falsification_condition": "observable replay failure",
            "new_programs": [],
            "consumer_changes": [],
            "metric_changes": [],
            "memory_schema_changes": [],
            "uses_query_future": False,
        },
    }
    system_prompt = (
        "You are the Slow-Path patch proposer in a time-series data adaptation "
        "Harness. Diagnose the earliest Harness fault from anonymous categorical "
        "Action-Response episodes and propose one minimal typed patch. You are not "
        "given the correct patch. Use only the supplied runtime primitives and one "
        "allowed primary surface. Do not invent Programs, datasets, outcomes, numeric "
        "effect thresholds or hidden labels. Return exactly one JSON object without "
        "markdown."
    )
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    plan = json.loads(completion.choices[0].message.content or "")
    if not isinstance(plan, dict):
        raise ValueError("LLM patch proposal must be one JSON object")
    provider = {
        "base_url": base_url,
        "requested_model": model,
        "returned_model": getattr(completion, "model", ""),
        "prompt_tokens": getattr(
            getattr(completion, "usage", None), "prompt_tokens", None
        ),
        "completion_tokens": getattr(
            getattr(completion, "usage", None), "completion_tokens", None
        ),
    }
    plan["provider"] = provider
    plan_path = root / REVERSIBLE_TARGET_REPRESENTATION_LLM_PLAN_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    violations: list[str] = []
    if plan.get("primary_operation") not in dossier["allowed_patch_surfaces"]:
        violations.append("primary_operation is outside the allowed patch surfaces")
    if plan.get("diagnosed_first_fault") not in {
        row["code"] for row in dossier["observed_first_faults"]
    }:
        violations.append("diagnosed_first_fault is not grounded in the Dossier")
    steps = plan.get("workflow_steps")
    if not isinstance(steps, list) or not steps:
        violations.append("workflow_steps must be a non-empty list")
    else:
        for step in steps:
            if not isinstance(step, dict) or step.get("primitive") not in allowed_primitives:
                violations.append("workflow uses an unavailable runtime primitive")
                break
    observations = plan.get("observation_requirements")
    if not isinstance(observations, list):
        violations.append("observation_requirements must be a list")
    else:
        for observation in observations:
            if (
                not isinstance(observation, dict)
                or observation.get("evidence_source") not in allowed_primitives
            ):
                violations.append("observation uses an unavailable evidence source")
                break
    feedback_units = plan.get("minimum_target_feedback_units")
    if not isinstance(feedback_units, int) or not 0 <= feedback_units <= 3:
        violations.append("minimum_target_feedback_units must be an integer in [0,3]")
    if plan.get("uses_query_future") is not False:
        violations.append("plan does not explicitly forbid Query future")
    for key in (
        "new_programs",
        "consumer_changes",
        "metric_changes",
        "memory_schema_changes",
    ):
        if plan.get(key):
            violations.append(f"plan illegally populated {key}")

    passed = not violations
    return {
        "experiment_id": "E2-live-LLM-new-representation-failure-patch-proposal",
        "scientific_role": "new natural Failure Dossier to bounded typed patch proposal",
        "inputs": {
            "p0_report": REVERSIBLE_TARGET_REPRESENTATION_P0_REPORT_PATH,
            "raw_time_series_sent": False,
            "dataset_identities_sent": False,
            "effect_magnitudes_sent": False,
            "internal_report_sent": False,
            "correct_patch_value_supplied": False,
        },
        "llm_plan": plan,
        "validation": {"contract_violations": violations},
        "provider": provider,
        "compute": {"consumer_fit_count": 0, "llm_api_call_count": 1},
        "gate": {"passed": passed},
        "verdict": (
            "LLM_NEW_FAILURE_PATCH_PROPOSAL_ACCEPTED_FOR_LOCAL_REPLAY"
            if passed
            else "LLM_NEW_FAILURE_PATCH_PROPOSAL_REJECTED"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "A contract-valid proposal is not an effective Harness update. The patch "
            "must change behavior and improve local deterministic replay before any "
            "fresh Target or Capability claim."
        ),
    }


def run_reversible_representation_llm_patch_replay(
    root: Path,
    *,
    source_report_path: str = REVERSIBLE_TARGET_REPRESENTATION_P0_REPORT_PATH,
    experiment_id: str = "E2-live-LLM-new-representation-failure-patch-replay",
) -> dict[str, object]:
    """Compile the one-shot LLM patch and reject dead conservative behavior."""

    p0 = _read_object(root / source_report_path)
    plan = _read_object(root / REVERSIBLE_TARGET_REPRESENTATION_LLM_PLAN_PATH)
    if not isinstance(p0.get("datasets"), list) or not p0["datasets"]:
        raise ValueError("representation Action-Response episodes are unavailable")
    if plan.get("patch_id") != "observe_transport_before_execution":
        raise ValueError("unexpected one-shot LLM patch")
    primitives = [
        str(row.get("primitive"))
        for row in plan.get("workflow_steps", [])
        if isinstance(row, dict)
    ]
    required_primitives = {
        "READ_TRAINING_CONTEXT",
        "READ_PHASE_ALIGNED_HISTORICAL_POLICY_EPISODES",
        "PROBE_CURRENT_TARGET_SUPPORT",
        "ORDER_PROGRAMS",
        "EXECUTE_PROGRAM",
    }
    contract_violations: list[str] = []
    if plan.get("primary_operation") != "ADD_OBSERVATION":
        contract_violations.append("replay only compiles the proposed ADD_OBSERVATION")
    if not required_primitives.issubset(primitives):
        contract_violations.append("proposed workflow primitives are incomplete")
    if plan.get("uses_query_future") is not False:
        contract_violations.append("Query future boundary is not closed")
    if int(plan.get("minimum_target_feedback_units", -1)) != 1:
        contract_violations.append("proposal changed the frozen one-unit feedback budget")

    nonidentity = (
        "SEASONAL_RESIDUAL_TARGET",
        "LAST_VALUE_RESIDUAL_TARGET",
    )
    fixed_a3_program = nonidentity[0]
    replays: list[dict[str, object]] = []
    llm_execution_count = 0
    llm_harm_count = 0
    a3_execution_count = 0
    a3_harm_count = 0
    old_harm_count = 0
    llm_query_gains: list[float] = []
    a3_query_gains: list[float] = []
    old_query_gains: list[float] = []

    for dataset in p0["datasets"]:
        historical = dataset["phase_aligned_historical_policy_episodes"]
        historically_supported = []
        for program in nonidentity:
            gains = [
                float(episode["gain_vs_identity"][program])
                for episode in historical
            ]
            if gains and all(gain > 0.0 for gain in gains):
                historically_supported.append(
                    (program, statistics.fmean(gains))
                )
        historically_supported.sort(key=lambda row: (-row[1], row[0]))
        llm_candidate = (
            historically_supported[0][0] if historically_supported else None
        )
        fold_rows = {
            str(row["direction"]): row
            for row in dataset["support_to_query_replays"]
        }
        for direction, opposite in (("a_to_b", "b_to_a"), ("b_to_a", "a_to_b")):
            current = fold_rows[direction]
            query_gain_by_program = {
                program: float(fold_rows[opposite]["support_gains"][program])
                for program in nonidentity
            }
            llm_selected = "IDENTITY_RAW_TARGET"
            llm_query_gain = 0.0
            if (
                llm_candidate is not None
                and float(current["support_gains"][llm_candidate]) > 0.0
            ):
                llm_selected = llm_candidate
                llm_query_gain = query_gain_by_program[llm_candidate]
                llm_execution_count += 1
                llm_harm_count += llm_query_gain < 0.0

            a3_selected = "IDENTITY_RAW_TARGET"
            a3_query_gain = 0.0
            if float(current["support_gains"][fixed_a3_program]) > 0.0:
                a3_selected = fixed_a3_program
                a3_query_gain = query_gain_by_program[fixed_a3_program]
                a3_execution_count += 1
                a3_harm_count += a3_query_gain < 0.0

            old_query_gain = float(current["query_gain"])
            old_harm_count += old_query_gain < 0.0
            llm_query_gains.append(llm_query_gain)
            a3_query_gains.append(a3_query_gain)
            old_query_gains.append(old_query_gain)
            replays.append(
                {
                    "environment": str(dataset["dataset"]),
                    "direction": direction,
                    "historically_supported_programs": [
                        row[0] for row in historically_supported
                    ],
                    "LLM_patch": {
                        "candidate": llm_candidate,
                        "selected_program": llm_selected,
                        "query_gain": llm_query_gain,
                    },
                    "A3_equal_budget_static_first": {
                        "candidate": fixed_a3_program,
                        "selected_program": a3_selected,
                        "query_gain": a3_query_gain,
                    },
                    "old_best_of_two_support_descriptive": {
                        "selected_program": current["selected_program"],
                        "query_gain": old_query_gain,
                    },
                }
            )

    llm_macro = statistics.fmean(llm_query_gains)
    a3_macro = statistics.fmean(a3_query_gains)
    old_macro = statistics.fmean(old_query_gains)
    behavior_nontrivial = llm_execution_count > 0
    improves_equal_budget_value = llm_macro > a3_macro
    reduces_old_harm = llm_harm_count < old_harm_count
    passed = (
        not contract_violations
        and behavior_nontrivial
        and improves_equal_budget_value
        and reduces_old_harm
    )
    if passed:
        verdict = "LLM_NEW_FAILURE_PATCH_DEVELOPMENT_REPLAY_PASS"
    elif not behavior_nontrivial:
        verdict = "LLM_NEW_FAILURE_PATCH_REJECTED_DEAD_ABSTENTION"
    else:
        verdict = "LLM_NEW_FAILURE_PATCH_REJECTED_BY_POLICY_VALUE"
    return {
        "experiment_id": experiment_id,
        "scientific_role": "deterministic local replay of one LLM-proposed Harness patch",
        "source_plan": REVERSIBLE_TARGET_REPRESENTATION_LLM_PLAN_PATH,
        "source_action_response_report": source_report_path,
        "compiled_patch": {
            "patch_id": plan["patch_id"],
            "primary_operation": plan["primary_operation"],
            "observation": (
                "require every pre-cutoff phase-aligned historical PolicyEpisode to "
                "support a candidate before one current Target Support confirmation"
            ),
            "fallback": "IDENTITY_RAW_TARGET",
            "query_future_used": False,
        },
        "replays": replays,
        "summary": {
            "replay_count": len(replays),
            "LLM_patch_execution_count": llm_execution_count,
            "LLM_patch_harm_count": llm_harm_count,
            "LLM_patch_macro_query_gain": llm_macro,
            "A3_equal_budget_execution_count": a3_execution_count,
            "A3_equal_budget_harm_count": a3_harm_count,
            "A3_equal_budget_macro_query_gain": a3_macro,
            "old_best_of_two_support_harm_count": old_harm_count,
            "old_best_of_two_support_macro_query_gain": old_macro,
        },
        "validation": {
            "contract_violations": contract_violations,
            "behavior_nontrivial": behavior_nontrivial,
            "improves_equal_budget_policy_value": improves_equal_budget_value,
            "reduces_old_harm": reduces_old_harm,
        },
        "compute": {"consumer_fit_count": 0, "llm_api_call_count": 0},
        "gate": {"passed": passed},
        "verdict": verdict,
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed development replay only. A rejected patch remains negative "
            "Harness-update evidence and must not be tuned on the same episodes."
        ),
    }


def run_reversible_representation_llm_revision_proposal(
    root: Path,
    *,
    model: str,
    base_url: str,
) -> dict[str, object]:
    """Use rejected replay evidence for one bounded, non-prompt-tuned revision."""

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    first_plan = _read_object(root / REVERSIBLE_TARGET_REPRESENTATION_LLM_PLAN_PATH)
    development_replay = _read_object(
        root / REVERSIBLE_TARGET_REPRESENTATION_LLM_REPLAY_REPORT_PATH
    )
    extension_replay = _read_object(
        root / REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPLAY_REPORT_PATH
    )
    if development_replay.get("verdict") != (
        "LLM_NEW_FAILURE_PATCH_REJECTED_DEAD_ABSTENTION"
    ) or extension_replay.get("verdict") != (
        "LLM_NEW_FAILURE_PATCH_REJECTED_BY_POLICY_VALUE"
    ):
        raise ValueError("the frozen rejection evidence is unavailable")

    dossier = {
        "task_consumer": {
            "task": "forecasting",
            "consumer": "fixed shared forecasting Consumer",
        },
        "program_family": {
            "actions": [
                "SEASONAL_RESIDUAL_TARGET",
                "LAST_VALUE_RESIDUAL_TARGET",
            ],
            "fallback": "IDENTITY_RAW_TARGET",
        },
        "rejected_patch": {
            "patch_id": first_plan["patch_id"],
            "primary_operation": first_plan["primary_operation"],
            "categorical_behavior": (
                "require pre-cutoff phase-aligned historical support before one "
                "current Target Support confirmation"
            ),
        },
        "anonymous_replay_evidence": [
            {
                "episode": "original_development_replay",
                "behavior": "ALWAYS_ABSTAINED",
                "policy_value": "BELOW_EQUAL_BUDGET_TARGET_ONLY",
                "harm": "NONE",
            },
            {
                "episode": "post_proposal_environment_A",
                "historical_transport": "CONSISTENT_POSITIVE",
                "current_support": "POSITIVE",
                "query_outcome": "HARMFUL",
            },
            {
                "episode": "post_proposal_environment_B",
                "historical_transport": "CONSISTENT_POSITIVE",
                "current_support": "POSITIVE",
                "query_outcome": "SUPPORTED",
            },
        ],
        "new_first_fault": {
            "code": "HISTORICAL_AND_SINGLE_SUPPORT_AGREEMENT_NOT_DECISION_SUFFICIENT",
            "description": (
                "historical transport plus one positive Support response can still "
                "represent either a useful or a harmful current action"
            ),
        },
        "allowed_primary_surfaces": [
            "ADD_OBSERVATION",
            "COMPOSE_WORKFLOW",
            "RESTRICT_SCOPE",
            "ADD_RISK",
        ],
        "forbidden_changes": [
            "program_supply",
            "consumer",
            "metric",
            "memory_schema",
            "query_visibility",
        ],
        "privacy": {
            "raw_time_series_included": False,
            "dataset_identity_included": False,
            "effect_magnitude_included": False,
            "internal_report_included": False,
        },
    }
    runtime_primitives = [
        "READ_TRAINING_CONTEXT",
        "READ_PHASE_ALIGNED_HISTORICAL_POLICY_EPISODES",
        "PROBE_CURRENT_TARGET_SUPPORT",
        "SPLIT_CURRENT_SUPPORT_COHORT",
        "CHECK_SUPPORT_AGREEMENT",
        "CHECK_REGIME_STABILITY",
        "ORDER_PROGRAMS",
        "EXECUTE_PROGRAM",
        "ABSTAIN",
    ]
    payload = {
        "failure_dossier": dossier,
        "runtime_primitives": runtime_primitives,
        "design_constraints": [
            "Revise the rejected patch once; do not merely restate it.",
            "Choose exactly one primary Harness surface.",
            "Preserve nontrivial execution opportunity; always-abstain is invalid.",
            "Use at most three current Target feedback units.",
            "Do not make a static period or dataset identity router.",
            "Do not invent a new Program, Consumer, metric, outcome or hidden label.",
            "Current Query future is forbidden.",
        ],
        "required_output": {
            "patch_id": "short identifier",
            "supersedes_patch_id": first_plan["patch_id"],
            "diagnosed_first_fault": dossier["new_first_fault"]["code"],
            "primary_operation": "one allowed primary surface",
            "diagnosis": "short explanation",
            "workflow_steps": [
                {
                    "step": 1,
                    "primitive": "one supplied runtime primitive",
                    "behavior": "typed behavior",
                }
            ],
            "minimum_target_feedback_units": "integer 1 to 3",
            "risk_guards": ["typed abstention condition"],
            "expected_behavior_change": "short description",
            "falsification_condition": "observable replay failure",
            "new_programs": [],
            "consumer_changes": [],
            "metric_changes": [],
            "memory_schema_changes": [],
            "uses_query_future": False,
        },
    }
    system_prompt = (
        "You are revising one rejected typed patch inside a time-series data "
        "adaptation Harness. The new anonymous replay evidence was not available to "
        "the first proposal. Propose exactly one bounded revision using only supplied "
        "runtime primitives. Preserve useful actions while resolving the new first "
        "fault. Do not tune numeric effect thresholds or invent data. Return exactly "
        "one JSON object without markdown."
    )
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    plan = json.loads(completion.choices[0].message.content or "")
    if not isinstance(plan, dict):
        raise ValueError("LLM revision must be one JSON object")
    provider = {
        "base_url": base_url,
        "requested_model": model,
        "returned_model": getattr(completion, "model", ""),
        "prompt_tokens": getattr(
            getattr(completion, "usage", None), "prompt_tokens", None
        ),
        "completion_tokens": getattr(
            getattr(completion, "usage", None), "completion_tokens", None
        ),
    }
    plan["provider"] = provider
    plan_path = root / REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_PLAN_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    violations: list[str] = []
    if plan.get("supersedes_patch_id") != first_plan["patch_id"]:
        violations.append("revision does not bind the rejected patch")
    if plan.get("diagnosed_first_fault") != dossier["new_first_fault"]["code"]:
        violations.append("revision is not grounded in the new first fault")
    if plan.get("primary_operation") not in dossier["allowed_primary_surfaces"]:
        violations.append("revision primary surface is unavailable")
    steps = plan.get("workflow_steps")
    if not isinstance(steps, list) or not steps:
        violations.append("revision workflow_steps must be non-empty")
    elif any(
        not isinstance(step, dict)
        or step.get("primitive") not in runtime_primitives
        for step in steps
    ):
        violations.append("revision uses an unavailable runtime primitive")
    feedback_units = plan.get("minimum_target_feedback_units")
    if not isinstance(feedback_units, int) or not 1 <= feedback_units <= 3:
        violations.append("revision feedback units must be an integer in [1,3]")
    if plan.get("uses_query_future") is not False:
        violations.append("revision does not close Query future")
    for key in (
        "new_programs",
        "consumer_changes",
        "metric_changes",
        "memory_schema_changes",
    ):
        if plan.get(key):
            violations.append(f"revision illegally populated {key}")
    passed = not violations
    return {
        "experiment_id": "E2-live-LLM-representation-rejected-patch-revision",
        "scientific_role": "one bounded revision from post-proposal negative evidence",
        "inputs": {
            "raw_time_series_sent": False,
            "dataset_identities_sent": False,
            "effect_magnitudes_sent": False,
            "internal_report_sent": False,
            "new_replay_evidence_added": True,
        },
        "llm_plan": plan,
        "validation": {"contract_violations": violations},
        "provider": provider,
        "compute": {"consumer_fit_count": 0, "llm_api_call_count": 1},
        "gate": {"passed": passed},
        "verdict": (
            "LLM_REJECTED_PATCH_REVISION_ACCEPTED_FOR_LOCAL_REPLAY"
            if passed
            else "LLM_REJECTED_PATCH_REVISION_CONTRACT_REJECTED"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "A contract-valid revision is not an effective Skill. This is the final "
            "LLM proposal for the current residual-representation development family."
        ),
    }


def run_reversible_representation_llm_revision_replay(
    root: Path,
) -> dict[str, object]:
    """Compile the final two-Support LLM revision on exposed development cohorts."""

    plan = _read_object(root / REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_PLAN_PATH)
    if plan.get("patch_id") != "split_support_stability_gate":
        raise ValueError("unexpected final LLM revision")
    required_primitives = {
        "READ_TRAINING_CONTEXT",
        "READ_PHASE_ALIGNED_HISTORICAL_POLICY_EPISODES",
        "PROBE_CURRENT_TARGET_SUPPORT",
        "SPLIT_CURRENT_SUPPORT_COHORT",
        "CHECK_SUPPORT_AGREEMENT",
        "CHECK_REGIME_STABILITY",
        "ORDER_PROGRAMS",
        "EXECUTE_PROGRAM",
        "ABSTAIN",
    }
    proposed_primitives = {
        str(row.get("primitive"))
        for row in plan.get("workflow_steps", [])
        if isinstance(row, dict)
    }
    violations: list[str] = []
    if plan.get("primary_operation") != "COMPOSE_WORKFLOW":
        violations.append("revision does not target COMPOSE_WORKFLOW")
    if not required_primitives.issubset(proposed_primitives):
        violations.append("revision workflow primitives are incomplete")
    if int(plan.get("minimum_target_feedback_units", -1)) != 2:
        violations.append("revision does not use exactly two feedback units")
    if plan.get("uses_query_future") is not False:
        violations.append("revision does not close Query future")

    episodes = run_reversible_target_representation_p0(
        root,
        specs_override=REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_SPECS,
        experiment_id="internal-revision-replay-cache",
    )
    frozen_extension = _read_object(
        root / REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPORT_PATH
    )
    frozen_by_dataset = {
        str(row["dataset"]): row for row in frozen_extension["datasets"]
    }
    for row in episodes["datasets"]:
        frozen = frozen_by_dataset[str(row["dataset"])]
        for program, gain in row["full_evaluation_gain_vs_identity"].items():
            if abs(float(gain) - float(frozen["full_evaluation_gain_vs_identity"][program])) > 1e-10:
                raise AssertionError("revision replay changed the frozen cohort outcome")

    programs = (
        "SEASONAL_RESIDUAL_TARGET",
        "LAST_VALUE_RESIDUAL_TARGET",
    )
    support_views = ((0, 2), (1, 3))
    query_indices = (4, 5, 6, 7)
    replay_rows: list[dict[str, object]] = []
    revision_gains: list[float] = []
    a3_gains: list[float] = []
    old_patch_gains: list[float] = []
    revision_harm = 0
    a3_harm = 0
    old_patch_harm = 0
    revision_execution_count = 0

    for dataset in episodes["datasets"]:
        per_series = dataset["per_series_gain_vs_identity_for_local_replay"]

        def best_program(indices: tuple[int, ...]) -> tuple[str, dict[str, float]]:
            gains = {
                program: statistics.fmean(
                    float(per_series[program][index]) for index in indices
                )
                for program in programs
            }
            selected = max(programs, key=lambda name: gains[name])
            if gains[selected] <= 0.0:
                selected = "IDENTITY_RAW_TARGET"
            return selected, gains

        historical_support: dict[str, list[float]] = {
            program: [
                float(episode["gain_vs_identity"][program])
                for episode in dataset["phase_aligned_historical_policy_episodes"]
            ]
            for program in programs
        }
        historically_stable = {
            program
            for program, gains in historical_support.items()
            if gains and all(gain > 0.0 for gain in gains)
        }
        support_decisions = [best_program(indices) for indices in support_views]
        agreed_program = (
            support_decisions[0][0]
            if support_decisions[0][0] == support_decisions[1][0]
            and support_decisions[0][0] != "IDENTITY_RAW_TARGET"
            else None
        )
        revision_selected = (
            agreed_program if agreed_program in historically_stable else None
        )
        revision_gain = (
            statistics.fmean(
                float(per_series[revision_selected][index])
                for index in query_indices
            )
            if revision_selected is not None
            else 0.0
        )
        if revision_selected is not None:
            revision_execution_count += 1
            revision_harm += revision_gain < 0.0

        combined_support = tuple(index for view in support_views for index in view)
        a3_selected, a3_support_gains = best_program(combined_support)
        a3_gain = (
            statistics.fmean(
                float(per_series[a3_selected][index]) for index in query_indices
            )
            if a3_selected != "IDENTITY_RAW_TARGET"
            else 0.0
        )
        a3_harm += a3_gain < 0.0

        historical_order = sorted(
            historically_stable,
            key=lambda program: (
                -statistics.fmean(historical_support[program]),
                program,
            ),
        )
        old_candidate = historical_order[0] if historical_order else None
        old_support_gain = (
            statistics.fmean(
                float(per_series[old_candidate][index])
                for index in combined_support
            )
            if old_candidate is not None
            else 0.0
        )
        old_selected = old_candidate if old_support_gain > 0.0 else None
        old_gain = (
            statistics.fmean(
                float(per_series[old_selected][index]) for index in query_indices
            )
            if old_selected is not None
            else 0.0
        )
        old_patch_harm += old_gain < 0.0

        revision_gains.append(revision_gain)
        a3_gains.append(a3_gain)
        old_patch_gains.append(old_gain)
        replay_rows.append(
            {
                "environment": str(dataset["dataset"]),
                "support_views": [
                    {
                        "indices": list(support_views[index]),
                        "selected_program": decision[0],
                        "program_gains": decision[1],
                    }
                    for index, decision in enumerate(support_decisions)
                ],
                "historically_stable_programs": sorted(historically_stable),
                "revision_selected_program": (
                    revision_selected or "IDENTITY_RAW_TARGET"
                ),
                "revision_query_gain": revision_gain,
                "A3_selected_program": a3_selected,
                "A3_support_gains": a3_support_gains,
                "A3_query_gain": a3_gain,
                "rejected_old_patch_selected_program": (
                    old_selected or "IDENTITY_RAW_TARGET"
                ),
                "rejected_old_patch_query_gain": old_gain,
            }
        )

    revision_macro = statistics.fmean(revision_gains)
    a3_macro = statistics.fmean(a3_gains)
    old_macro = statistics.fmean(old_patch_gains)
    behavior_nontrivial = revision_execution_count > 0
    passed = (
        not violations
        and behavior_nontrivial
        and revision_harm == 0
        and revision_harm < old_patch_harm
        and revision_macro > a3_macro
    )
    return {
        "experiment_id": "E2-live-LLM-representation-revision-development-replay",
        "scientific_role": "development replay of one rejection-driven LLM Workflow revision",
        "source_plan": REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_PLAN_PATH,
        "compiled_patch": {
            "patch_id": plan["patch_id"],
            "support_views": [list(indices) for indices in support_views],
            "query_indices": list(query_indices),
            "regime_stability_binding": (
                "the same action is positive in both pre-cutoff phase-aligned "
                "historical PolicyEpisodes"
            ),
            "execution_rule": (
                "both disjoint current Support views choose the same historically "
                "stable action; otherwise IDENTITY"
            ),
            "query_future_used_for_decision": False,
        },
        "replays": replay_rows,
        "summary": {
            "environment_count": len(replay_rows),
            "revision_execution_count": revision_execution_count,
            "revision_harm_count": revision_harm,
            "revision_macro_query_gain": revision_macro,
            "A3_harm_count": a3_harm,
            "A3_macro_query_gain": a3_macro,
            "rejected_old_patch_harm_count": old_patch_harm,
            "rejected_old_patch_macro_query_gain": old_macro,
        },
        "validation": {
            "contract_violations": violations,
            "behavior_nontrivial": behavior_nontrivial,
            "harm_eliminated": revision_harm == 0,
            "harm_reduced_vs_rejected_patch": revision_harm < old_patch_harm,
            "policy_value_above_equal_budget_A3": revision_macro > a3_macro,
        },
        "compute": {"consumer_fit_count": 6, "llm_api_call_count": 0},
        "gate": {"passed": passed},
        "verdict": (
            "LLM_REJECTION_DRIVEN_WORKFLOW_REVISION_DEVELOPMENT_PASS"
            if passed
            else "LLM_REJECTION_DRIVEN_WORKFLOW_REVISION_DEVELOPMENT_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed development replay on the same categorical outcomes supplied to "
            "the revision LLM. A pass only authorizes one frozen natural Target test."
        ),
    }


def run_second_failure_mechanism_framework_replay(root: Path) -> dict[str, object]:
    """Route a distinct natural transport failure through the acquisition core.

    This reuses an already completed live LLM revision and its deterministic
    natural replay.  The framework succeeds only by rejecting the non-improving
    composition and keeping Capability Memory unchanged.
    """

    source = _read_object(
        root / REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPORT_PATH
    )
    llm_revision = _read_object(
        root / REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPORT_PATH
    )
    replay = _read_object(
        root / REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPLAY_REPORT_PATH
    )
    programs = (
        "SEASONAL_RESIDUAL_TARGET",
        "LAST_VALUE_RESIDUAL_TARGET",
    )
    source_episodes: list[dict[str, object]] = []
    for dataset in source["datasets"]:
        historical = dataset["phase_aligned_historical_policy_episodes"]
        source_episodes.append(
            {
                "workflows": {
                    program: {
                        "workflow_id": program,
                        "support_gain": statistics.fmean(
                            float(row["gain_vs_identity"][program])
                            for row in historical
                        ),
                        "query_gain": float(
                            dataset["full_evaluation_gain_vs_identity"][program]
                        ),
                    }
                    for program in programs
                }
            }
        )
    candidate = build_candidate_skill(
        [],
        source_episodes,
        capability_id="reversible_target_representation_candidate_v1",
        task_context={
            "task": "forecasting",
            "consumer": "fixed shared forecasting Consumer",
        },
        workflow_supply=programs,
    )
    composition = {
        "type": "split_support_stability_gate",
        "minimum_target_feedback_units": 2,
    }
    failure_cases = [
        {"support_to_query_replays": dataset["support_to_query_replays"]}
        for dataset in source["datasets"]
    ]
    plan = llm_revision["llm_plan"]
    typed_patch = {
        "patch_id": plan["patch_id"],
        "operations": [
            {
                "operation": plan["primary_operation"],
                "target_surface": "workflow",
                "value": composition,
            }
        ],
    }
    replay_validation = replay["validation"]
    rejected_for_value = bool(
        replay.get("verdict")
        == "LLM_REJECTION_DRIVEN_WORKFLOW_REVISION_DEVELOPMENT_FAIL"
        and replay_validation.get("behavior_nontrivial") is True
        and replay_validation.get("policy_value_above_equal_budget_A3") is False
        and replay_validation.get("harm_reduced_vs_rejected_patch") is False
        and replay.get("capability_or_memory_written") is False
    )
    violations: list[str] = []
    cycle: dict[str, object] | None = None
    try:
        if plan.get("minimum_target_feedback_units") != 2:
            raise ValueError("LLM revision does not bind the bounded composition")
        cycle = run_failure_driven_update_cycle(
            candidate,
            failure_cases,
            allowed_observations=["phase_aligned_historical_policy_episode"],
            allowed_controls=["keep_best_support_so_far"],
            allowed_compositions=[composition],
            propose_patch=lambda _dossier: typed_patch,
            replay_patch=lambda _candidate: replay,
            resolve_patch=lambda _candidate, rows: {
                "status": (
                    "RESTRICTED" if rows[0]["gate"]["passed"] else "REJECTED"
                ),
                "reason": (
                    "development replay only"
                    if rows[0]["gate"]["passed"]
                    else "no incremental policy value or harm reduction"
                ),
            },
        )
    except (KeyError, TypeError, ValueError) as exc:
        violations.append(str(exc))

    dossier = cycle["failure_dossier"] if cycle is not None else {}
    patched_candidate = cycle["candidate_after_patch"] if cycle is not None else None
    resolved_skill = cycle["resolved_skill"] if cycle is not None else None
    second_fault_diagnosed = bool(
        cycle is not None
        and dossier["categorical_first_faults"][0]["code"]
        == "ONE_SUPPORT_PROBE_CAN_FALSELY_CONFIRM"
    )
    passed = bool(
        not violations
        and patched_candidate is not None
        and rejected_for_value
        and resolved_skill["status"] == "REJECTED"
    )
    return {
        "experiment_id": "E2.74-second-natural-failure-core-cycle-replay",
        "scientific_role": (
            "exposed-cache mechanism test that a Support-to-Query transport failure "
            "is diagnosed, bounded, replayed and rejected without Skill admission"
        ),
        "framework_path": (
            "SUPPORT-QUERY ACTION-RESPONSE -> CATEGORICAL FAILURE DOSSIER -> "
            "CACHED LIVE LLM COMPOSITION -> TYPED COMPILER -> NATURAL POLICY "
            "REPLAY -> EXPLICIT REJECTION"
        ),
        "failure_dossier": dossier,
        "llm": {
            "provider": llm_revision.get("provider"),
            "proposal_source": REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPORT_PATH,
            "typed_patch": typed_patch,
        },
        "candidate_after_patch": (
            {
                "status": patched_candidate["status"],
                "workflow_composition": patched_candidate["workflow_composition"],
            }
            if patched_candidate is not None
            else None
        ),
        "explicit_resolution": (
            {
                "status": resolved_skill["status"],
                "reason": resolved_skill["promotion_result"]["reason"],
            }
            if resolved_skill is not None
            else None
        ),
        "deterministic_replay": {
            "source": REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPLAY_REPORT_PATH,
            "summary": replay["summary"],
            "validation": replay_validation,
            "verdict": replay["verdict"],
        },
        "validation": {
            "contract_violations": violations,
            "second_fault_diagnosed": second_fault_diagnosed,
            "patch_behavior_nontrivial": replay_validation["behavior_nontrivial"],
            "non_improving_patch_rejected": rejected_for_value,
            "capability_or_memory_written": False,
        },
        "compute": {"new_consumer_fit_count": 0, "new_llm_api_call_count": 0},
        "gate": {"passed": passed},
        "verdict": (
            "SECOND_FAILURE_MECHANISM_REJECTED_SAFELY_FRAMEWORK_PASS"
            if passed
            else "SECOND_FAILURE_MECHANISM_FRAMEWORK_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "This is a framework generalization replay on exposed natural evidence, "
            "not a successful residual-representation Skill or fresh Transfer claim."
        ),
    }


def run_rejection_aware_fast_path_replay(root: Path) -> dict[str, object]:
    """Verify that independent rejection changes the next Fast-Path active set."""

    general = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    rejected_missing = _read_object(root / MISSING_WINDOW_WEIGHTING_CAPABILITY_PATH)
    active_cards = read_active_skill_cards([general, rejected_missing])
    active_ids = {str(card["capability_id"]) for card in active_cards}
    historical_plan = _read_object(root / FORECASTING_TWO_SKILL_LIVE_PLAN_PATH)
    decisions = {
        str(row["context_id"]): row for row in historical_plan["decisions"]
    }
    general_steps = [
        "retrieve phase-aligned historical PolicyEpisodes for W_rowblock/W_curation/W_temporal_origin",
        "probe current Support in retrieved order",
        "stop on first positive; otherwise IDENTITY",
    ]
    rows: list[dict[str, object]] = []
    blocked_inactive = 0
    for context in _forecasting_two_skill_contexts():
        context_id = str(context["context_id"])
        proposed = decisions[context_id]
        proposed_id = proposed.get("capability_id")
        if proposed_id not in active_ids:
            if proposed_id is not None:
                blocked_inactive += 1
            runtime_id = None
            runtime_action = "REJECT_INACTIVE_SKILL_AND_NO_OP"
            runtime_steps: list[str] = []
        elif (
            proposed_id == general["capability_id"]
            and context_id == "forecast_general_regular_panel"
            and proposed.get("decision") == "RETRIEVE_AND_PROBE"
            and list(proposed.get("workflow_steps", [])) == general_steps
        ):
            runtime_id = str(proposed_id)
            runtime_action = "COMPILE_ACTIVE_SKILL_WORKFLOW"
            runtime_steps = general_steps
        else:
            runtime_id = None
            runtime_action = "REJECT_INVALID_ACTIVE_SKILL_PLAN_AND_NO_OP"
            runtime_steps = []
        rows.append(
            {
                "context_id": context_id,
                "historical_llm_capability_id": proposed_id,
                "runtime_capability_id": runtime_id,
                "runtime_action": runtime_action,
                "runtime_workflow_steps": runtime_steps,
            }
        )

    general_row = next(
        row for row in rows if row["context_id"] == "forecast_general_regular_panel"
    )
    missing_rows = [
        row
        for row in rows
        if str(row["context_id"]).startswith("forecast_missing_")
    ]
    passed = bool(
        active_ids == {"historical_policy_episode_workflow_v1"}
        and general_row["runtime_capability_id"]
        == "historical_policy_episode_workflow_v1"
        and all(row["runtime_capability_id"] is None for row in missing_rows)
        and blocked_inactive == 3
    )
    return {
        "experiment_id": "E2.75-rejection-aware-Fast-Path-active-set",
        "scientific_role": (
            "mechanism test that an independent Target rejection removes a Skill "
            "from future planning/execution without deleting its negative evidence"
        ),
        "memory_read": {
            "stored_skill_ids": [general["capability_id"], rejected_missing["capability_id"]],
            "stored_statuses": {
                str(general["capability_id"]): general["status"],
                str(rejected_missing["capability_id"]): rejected_missing["status"],
            },
            "planner_visible_active_skill_ids": sorted(active_ids),
            "rejected_skill_preserved_as_slow_path_evidence": True,
        },
        "historical_live_plan_source": FORECASTING_TWO_SKILL_LIVE_PLAN_PATH,
        "runtime_replay": rows,
        "validation": {
            "active_general_skill_still_executes": general_row["runtime_capability_id"]
            == "historical_policy_episode_workflow_v1",
            "inactive_missing_skill_plan_count_blocked": blocked_inactive,
            "all_missing_contexts_safe_no_op": all(
                row["runtime_capability_id"] is None for row in missing_rows
            ),
            "rejected_evidence_deleted": False,
        },
        "compute": {"consumer_fit_count": 0, "llm_api_call_count": 0},
        "gate": {"passed": passed},
        "verdict": (
            "REJECTION_UPDATES_FAST_PATH_ACTIVE_SET_PASS"
            if passed
            else "REJECTION_UPDATES_FAST_PATH_ACTIVE_SET_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Behavior replay using a previously saved live LLM plan; it proves state "
            "propagation and fail-closed execution, not a new Skill or LLM advantage."
        ),
    }


def run_live_natural_imputation_cold_start(
    root: Path,
    *,
    model: str,
    base_url: str,
    specs_override: dict[str, dict[str, object]] | None = None,
    live_discovery: bool = True,
) -> dict[str, object]:
    """Cold-start one natural missing-value Workflow family from public Context."""

    import numpy as np
    import openai

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.operators.registry import get_operator

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    specs = specs_override or {
        "monash:nn5_daily": {
            "anchors": (240, 300, 360, 420),
            "evaluation_stop": 528,
            "period": 7,
        },
        "gefcom2012_load": {
            "anchors": (312, 372, 432, 492, 552, 612, 672, 732, 792, 852),
            "evaluation_stop": 912,
            "period": 24,
        },
        "noaa_global_hourly": {
            "anchors": (240, 300, 360, 420, 480, 540, 600, 660),
            "evaluation_stop": 720,
            "period": 24,
        },
    }
    workflow_catalog = [
        {
            "workflow_id": "W_period_median_imputation",
            "description": (
                "reconstruct natural gaps from originally observed values at the "
                "same phase in prior cycles"
            ),
            "mechanism": "multi_cycle_phase_median_reconstruction",
            "public_parameter_bindings": {"period": "periodicity.period"},
        },
        {
            "workflow_id": "W_ar_imputation",
            "description": (
                "reconstruct natural gaps with a bidirectional autoregressive model"
            ),
            "mechanism": "bidirectional_autoregressive_reconstruction",
            "public_parameter_bindings": {"period": "periodicity.period"},
        },
    ]
    observation_catalog = [
        {
            "observation_id": "missing_run_topology",
            "description": "summarize natural gap coverage and run lengths",
        },
        {
            "observation_id": "period_reliability",
            "description": "report the known periodic binding and visible cycles",
        },
        {
            "observation_id": "cohort_overview",
            "description": "summarize training series and window coverage",
        },
    ]
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    registry = [
        json.loads(line)
        for line in (
            root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_dirs: dict[str, Path] = {}
    for record_path in (root / "data/benchmark_v0_2/clean_base").glob(
        "*/record.json"
    ):
        record = _read_object(record_path)
        record_dirs[str(record["series_uid"])] = record_path.parent

    program_to_operator = {
        "IDENTITY": "impute_linear",
        "W_period_median_imputation": "period_median_complete",
        "W_ar_imputation": "impute_ar",
    }
    dataset_results: list[dict[str, object]] = []
    source_episodes: list[dict[str, object]] = []
    total_fits = 0
    total_llm_calls = 0

    for dataset_id, spec in specs.items():
        evaluation_stop = int(spec["evaluation_stop"])
        period = int(spec["period"])
        candidates = sorted(
            (
                row
                for row in registry
                if row["dataset_id"] == dataset_id
                and int(row["length"]) >= evaluation_stop + HORIZON
            ),
            key=lambda row: str(row["series_uid"]),
        )
        if len(candidates) < 20:
            raise ValueError(f"insufficient series for {dataset_id}")
        train_start = int(spec.get("train_start", 0))
        eval_start = int(spec.get("eval_start", 0))
        train_rows = candidates[train_start : train_start + 12]
        if len(train_rows) != 12:
            raise ValueError(f"fewer than twelve training rows: {dataset_id}")
        train_uids = {str(row["series_uid"]) for row in train_rows}
        eval_rows: list[dict[str, object]] = []
        for row in candidates[eval_start:]:
            uid = str(row["series_uid"])
            if uid in train_uids:
                continue
            values = np.load(record_dirs[uid] / "values.npy").astype(np.float64)
            context = values[evaluation_stop - CONTEXT_LENGTH : evaluation_stop]
            target = values[evaluation_stop : evaluation_stop + HORIZON]
            if (
                context.shape == (CONTEXT_LENGTH,)
                and target.shape == (HORIZON,)
                and np.isfinite(context).sum() >= CONTEXT_LENGTH // 2
                and np.isfinite(target).all()
            ):
                eval_rows.append(row)
            if len(eval_rows) == 8:
                break
        if len(eval_rows) != 8:
            raise ValueError(f"fewer than eight evaluation rows: {dataset_id}")

        raw_training_windows: list[Any] = []
        run_lengths: list[int] = []
        missing_points = 0
        total_points = 0
        for anchor in tuple(int(value) for value in spec["anchors"]):
            for row in train_rows:
                uid = str(row["series_uid"])
                values = np.load(record_dirs[uid] / "values.npy").astype(np.float64)
                raw = values[
                    anchor - CONTEXT_LENGTH : anchor + HORIZON
                ]
                if (
                    raw.shape != (CONTEXT_LENGTH + HORIZON,)
                    or np.isfinite(raw).sum() < raw.size // 2
                ):
                    continue
                missing = ~np.isfinite(raw)
                missing_points += int(missing.sum())
                total_points += int(raw.size)
                run_lengths.extend(
                    end - start for start, end in _missing_runs(missing.tolist())
                )
                raw_training_windows.append(raw)
        if len(raw_training_windows) < 24 or missing_points == 0:
            raise ValueError(f"insufficient natural missing windows: {dataset_id}")

        public_context = {
            "task_context": {
                "task": "forecasting",
                "consumer": "shared frozen Ridge",
            },
            "workspace": {
                "multiple_training_series_available": True,
                "training_series_count": len(train_rows),
                "training_window_count": len(raw_training_windows),
                "natural_missing_values_present": True,
            },
            "missingness": {
                "missing_fraction": missing_points / total_points,
                "maximum_run_length": max(run_lengths) if run_lengths else 0,
                "run_count": len(run_lengths),
            },
            "periodicity": {
                "known_period_available": True,
                "period": period,
                "visible_context_cycles": CONTEXT_LENGTH // period,
            },
            "information_boundary": "training_context_only_before_action_response",
        }
        planner_trace: dict[str, object] = {}

        def planner(payload: dict[str, object]) -> dict[str, object]:
            nonlocal total_llm_calls
            if not live_discovery:
                bound_period = int(
                    payload["public_context"]["periodicity"]["period"]
                )
                return {
                    "decision": "PROPOSE",
                    "selected_workflows": [
                        {
                            "workflow_id": "W_period_median_imputation",
                            "bindings": {"period": bound_period},
                        },
                        {
                            "workflow_id": "W_ar_imputation",
                            "bindings": {"period": bound_period},
                        },
                    ],
                    "probe_order": [
                        "W_period_median_imputation",
                        "W_ar_imputation",
                    ],
                    "requested_observations": [
                        "missing_run_topology",
                        "period_reliability",
                        "cohort_overview",
                    ],
                    "fallback": "IDENTITY",
                }
            system_prompt = (
                "You are the cold-start Workflow-supply planner in a time-series "
                "data adaptation Harness. Select every catalog Workflow that is "
                "mechanically applicable to the public natural-missing Context. "
                "You cannot observe action responses and must not guess benefit. "
                "Use exact public bindings, preserve IDENTITY fallback, request only "
                "catalog observations, and return exactly one JSON object with only "
                "decision, selected_workflows, probe_order, requested_observations, "
                "fallback. decision must be PROPOSE or ABSTAIN. selected_workflows "
                "must be an array of objects shaped exactly as "
                "{\"workflow_id\": <catalog id>, \"bindings\": {\"period\": "
                "<the integer public_context.periodicity.period>}}. probe_order must "
                "contain only the selected Workflow ids, never Observation ids. "
                "requested_observations must contain only Observation ids. Do not "
                "return public_parameter_bindings. Return no markdown."
            )
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            total_llm_calls += 1
            planner_trace["provider"] = {
                "base_url": base_url,
                "requested_model": model,
                "returned_model": getattr(completion, "model", ""),
                "prompt_tokens": getattr(
                    getattr(completion, "usage", None), "prompt_tokens", None
                ),
                "completion_tokens": getattr(
                    getattr(completion, "usage", None), "completion_tokens", None
                ),
            }
            proposal = json.loads(completion.choices[0].message.content or "")
            if not isinstance(proposal, dict):
                raise ValueError("cold-start planner must return one JSON object")
            planner_trace["proposal"] = proposal
            return proposal

        discovery = discover_workflow_supply(
            public_context,
            workflow_catalog,
            observation_catalog,
            planner,
            max_candidates=2,
        )
        if discovery["decision"] != "PROPOSE":
            dataset_results.append(
                {
                    "dataset": dataset_id,
                    "public_context": public_context,
                    "planner": planner_trace,
                    "discovery": discovery,
                    "programs": [],
                }
            )
            continue

        program_rows: list[dict[str, object]] = []
        for workflow_id in ("IDENTITY", *discovery["workflow_supply"]):
            operator = get_operator(program_to_operator[str(workflow_id)])
            x_train: list[Any] = []
            y_train: list[Any] = []
            for raw in raw_training_windows:
                if workflow_id == "IDENTITY":
                    filled = operator(raw)
                else:
                    filled = operator(raw, period=period)
                if not np.isfinite(filled).all():
                    continue
                context = filled[:CONTEXT_LENGTH]
                target = filled[CONTEXT_LENGTH:]
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    continue
                x_train.append((context - center) / scale)
                y_train.append((target - center) / scale)
            if len(x_train) < 24:
                raise ValueError(
                    f"insufficient imputed training rows: {dataset_id}/{workflow_id}"
                )

            x_eval: list[Any] = []
            actual: list[Any] = []
            centers: list[float] = []
            scales: list[float] = []
            seasonal_scales: list[float] = []
            for row in eval_rows:
                uid = str(row["series_uid"])
                directory = record_dirs[uid]
                values = np.load(directory / "values.npy").astype(np.float64)
                missing_mask = np.load(
                    directory / "natural_missing_mask.npy"
                ).astype(bool)
                raw_context = values[
                    evaluation_stop - CONTEXT_LENGTH : evaluation_stop
                ]
                target = values[evaluation_stop : evaluation_stop + HORIZON]
                if workflow_id == "IDENTITY":
                    context = operator(raw_context)
                else:
                    context = operator(raw_context, period=period)
                if not np.isfinite(context).all():
                    raise ValueError(
                        f"non-finite evaluation context: {dataset_id}/{workflow_id}"
                    )
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise ValueError(f"evaluation scale floor: {dataset_id}/{uid}")
                try:
                    scale_value = seasonal_scale(
                        values[:evaluation_stop],
                        ~missing_mask[:evaluation_stop],
                        period=period,
                        min_pairs=32,
                    )
                except (UndefinedSeasonalScale, ValueError) as exc:
                    raise ValueError(
                        f"undefined seasonal scale: {dataset_id}/{uid}"
                    ) from exc
                x_eval.append((context - center) / scale)
                actual.append(target)
                centers.append(center)
                scales.append(scale)
                seasonal_scales.append(scale_value)
            prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=np.asarray(x_train, dtype=np.float64),
                targets=np.asarray(y_train, dtype=np.float64),
                weights=np.ones(len(x_train), dtype=np.float64),
                x_eval=np.asarray(x_eval, dtype=np.float64),
            )
            total_fits += 1
            losses: list[float] = []
            for index, truth in enumerate(actual):
                raw_prediction = prediction[index] * scales[index] + centers[index]
                losses.append(
                    smase(truth, raw_prediction, scale=seasonal_scales[index])
                )
            program_rows.append(
                {
                    "workflow_id": workflow_id,
                    "support_loss": statistics.fmean(losses[:4]),
                    "query_loss": statistics.fmean(losses[4:]),
                    "combined_loss": statistics.fmean(losses),
                }
            )

        identity = next(row for row in program_rows if row["workflow_id"] == "IDENTITY")
        for row in program_rows:
            row["support_gain_vs_identity"] = identity["support_loss"] - row["support_loss"]
            row["query_gain_vs_identity"] = identity["query_loss"] - row["query_loss"]
            row["combined_gain_vs_identity"] = identity["combined_loss"] - row["combined_loss"]
        workflows = {
            str(row["workflow_id"]): {
                "workflow_id": row["workflow_id"],
                "support_gain": row["support_gain_vs_identity"],
                "query_gain": row["query_gain_vs_identity"],
            }
            for row in program_rows
            if row["workflow_id"] != "IDENTITY"
        }
        source_episodes.append({"workflows": workflows})
        dataset_results.append(
            {
                "dataset": dataset_id,
                "roster": {"train_start": train_start, "eval_start": eval_start},
                "public_context": public_context,
                "planner": planner_trace,
                "discovery": discovery,
                "programs": program_rows,
                "best_nonidentity_gain": max(
                    float(row["combined_gain_vs_identity"])
                    for row in program_rows
                    if row["workflow_id"] != "IDENTITY"
                ),
            }
        )

    material_gain = 0.005
    completed = [row for row in dataset_results if row["programs"]]
    positive = [
        row for row in completed if float(row["best_nonidentity_gain"]) > material_gain
    ]
    identity_optimal = [
        row for row in completed if float(row["best_nonidentity_gain"]) <= 0.0
    ]
    harmful_action_count = sum(
        float(program["combined_gain_vs_identity"]) < -material_gain
        for row in completed
        for program in row["programs"]
        if program["workflow_id"] != "IDENTITY"
    )
    all_contexts_compiled = len(completed) == len(specs)
    passed = bool(
        all_contexts_compiled
        and len(positive) >= 2
        and (identity_optimal or harmful_action_count > 0)
    )
    candidate = None
    if passed:
        candidate = build_candidate_skill(
            [],
            source_episodes,
            capability_id="cold_start_candidate_e2_76",
            task_context={
                "task": "forecasting",
                "consumer": "shared frozen Ridge",
                "context": "observable natural missing values with known period",
            },
            workflow_supply=(
                "W_period_median_imputation",
                "W_ar_imputation",
            ),
        )
    return {
        "experiment_id": "E2.77-natural-imputation-cold-start-contract-repair-P0",
        "scientific_role": (
            "exposed natural cold-start test of public-Context Workflow discovery "
            "followed by complete Source Action-Response"
        ),
        "program_family": {
            "incumbent": "IDENTITY_AFTER_MINIMAL_LINEAR_FILL",
            "candidate_workflows": [
                "W_period_median_imputation",
                "W_ar_imputation",
            ],
            "workflow_catalog_preexists_final_skill": False,
            "final_scope_or_skill_predefined": False,
        },
        "datasets": dataset_results,
        "summary": {
            "completed_dataset_count": len(completed),
            "material_positive_dataset_count": len(positive),
            "material_positive_datasets": [row["dataset"] for row in positive],
            "identity_optimal_dataset_count": len(identity_optimal),
            "identity_optimal_datasets": [row["dataset"] for row in identity_optimal],
            "harmful_nonidentity_action_count": harmful_action_count,
        },
        "candidate": (
            None
            if candidate is None
            else {
                "capability_id": candidate["capability_id"],
                "status": candidate["status"],
                "workflow_supply": candidate["workflow_supply"],
                "observation": candidate["observation"],
                "control": candidate["control"],
                "source_prior": candidate["source_prior"],
            }
        ),
        "compute": {
            "consumer_fit_count": total_fits,
            "llm_api_call_count": total_llm_calls,
        },
        "gate": {
            "all_public_contexts_compiled": all_contexts_compiled,
            "at_least_two_material_positive_datasets": len(positive) >= 2,
            "matched_risk_present": bool(identity_optimal or harmful_action_count > 0),
            "passed": passed,
        },
        "verdict": (
            "NATURAL_COLD_START_CANDIDATE_PREMISE_PASS"
            if passed
            else (
                "NATURAL_COLD_START_FAMILY_CLOSED_AT_P0"
                if all_contexts_compiled
                else "NATURAL_COLD_START_INCONCLUSIVE_PLANNER_CONTRACT"
            )
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed Source P0 only. A pass creates a Candidate for Slow-Path "
            "diagnosis; it does not support Promotion, Fast-Path use or Target transfer."
        ),
    }


def run_natural_imputation_target_pilot(
    root: Path,
    *,
    model: str,
    base_url: str,
) -> dict[str, object]:
    """Compare the Source Candidate with equal-budget Target-only ordering."""

    source = _read_object(root / NATURAL_IMPUTATION_COLD_START_REPORT_PATH)
    source_candidate = source.get("candidate")
    if (
        source.get("verdict") != "NATURAL_COLD_START_CANDIDATE_PREMISE_PASS"
        or not isinstance(source_candidate, dict)
    ):
        raise ValueError("the Source imputation Candidate is unavailable")
    target: dict[str, object] | None = None
    feasibility_rejections: list[dict[str, object]] = []
    for train_start in (0, 12, 24, 36, 48):
        try:
            target = run_live_natural_imputation_cold_start(
                root,
                model=model,
                base_url=base_url,
                specs_override={
                    "metr_la": {
                        "anchors": (
                            312,
                            372,
                            432,
                            492,
                            552,
                            612,
                            672,
                            732,
                            792,
                            852,
                        ),
                        "evaluation_stop": 928,
                        "period": 24,
                        "train_start": train_start,
                        "eval_start": 100,
                    }
                },
                live_discovery=False,
            )
            break
        except ValueError as exc:
            if "insufficient natural missing windows" not in str(exc):
                raise
            feasibility_rejections.append(
                {"train_start": train_start, "reason": "NO_ACTIONABLE_MISSINGNESS"}
            )
    if target is None:
        return {
            "experiment_id": "E2.78-natural-imputation-Target-pilot",
            "scientific_role": "Source-order versus Target-only adaptation pilot",
            "feasibility_rejections": feasibility_rejections,
            "verdict": "NATURAL_IMPUTATION_TARGET_PILOT_INCONCLUSIVE",
            "capability_or_memory_written": False,
        }
    completed = [row for row in target["datasets"] if row["programs"]]
    if len(completed) != 1:
        return {
            "experiment_id": "E2.78-natural-imputation-Target-pilot",
            "scientific_role": "Source-order versus Target-only adaptation pilot",
            "target_adapter": target,
            "verdict": "NATURAL_IMPUTATION_TARGET_PILOT_INCONCLUSIVE",
            "capability_or_memory_written": False,
        }
    target_row = completed[0]
    responses = {
        str(row["workflow_id"]): {
            "workflow_id": row["workflow_id"],
            "support_gain": float(row["support_gain_vs_identity"]),
            "query_gain": float(row["query_gain_vs_identity"]),
        }
        for row in target_row["programs"]
        if row["workflow_id"] != "IDENTITY"
    }
    source_order = tuple(
        str(value) for value in source_candidate["source_prior"]["workflow_order"]
    )
    a5_curve = workflow_curve_from_policy_episode(responses, source_order)
    target_only_curves = [
        workflow_curve_from_policy_episode(responses, order)
        for order in permutations(source_order)
    ]
    a3_curve = [
        {
            "budget": budget,
            "fixed_query_gain": statistics.fmean(
                float(curve[budget]["fixed_query_gain"])
                for curve in target_only_curves
            ),
        }
        for budget in range(len(source_order) + 1)
    ]
    a5_auc = policy_episode_adapt_auc(a5_curve)
    a3_auc = policy_episode_adapt_auc(a3_curve)
    a5_harm_count = sum(
        float(row["fixed_query_gain"]) < 0.0 for row in a5_curve[1:]
    )
    behavior_differs = any(
        abs(
            float(a5_curve[index]["fixed_query_gain"])
            - float(a3_curve[index]["fixed_query_gain"])
        )
        > 1e-12
        for index in range(1, len(a5_curve))
    )
    passed = bool(a5_auc > a3_auc and a5_harm_count == 0 and behavior_differs)
    return {
        "experiment_id": "E2.78-natural-imputation-Target-pilot",
        "scientific_role": (
            "one natural Target adaptation pilot for a cold-start Source Candidate"
        ),
        "source_candidate": {
            "report": NATURAL_IMPUTATION_COLD_START_REPORT_PATH,
            "status": source_candidate["status"],
            "workflow_order": list(source_order),
        },
        "target": {
            "dataset": target_row["dataset"],
            "context_exposure": "INSTANCE_SEEN_IN_OTHER_PROGRAM_FAMILIES",
            "outcome_exposure": "EXPOSED_BY_THIS_PILOT",
            "public_context": target_row["public_context"],
            "compiled_workflows": target_row["discovery"]["compiled_workflows"],
            "programs": target_row["programs"],
            "feasibility_rejections_before_selected_roster": feasibility_rejections,
        },
        "adaptation": {
            "A5_source_order_curve": a5_curve,
            "A3_equal_budget_order_average_curve": a3_curve,
            "A5_adaptation_auc": a5_auc,
            "A3_adaptation_auc": a3_auc,
            "A5_minus_A3": a5_auc - a3_auc,
            "A5_harm_count": a5_harm_count,
        },
        "validation": {
            "source_order_changes_early_behavior": behavior_differs,
            "A5_above_equal_budget_A3": a5_auc > a3_auc,
            "A5_harm_zero": a5_harm_count == 0,
        },
        "compute": {
            "consumer_fit_count": target["compute"]["consumer_fit_count"],
            "llm_api_call_count": target["compute"]["llm_api_call_count"],
        },
        "gate": {"passed": passed},
        "verdict": (
            "NATURAL_IMPUTATION_TARGET_PILOT_PASS"
            if passed
            else "NATURAL_IMPUTATION_TARGET_PILOT_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "One exposed natural Target pilot. A pass authorizes an independent "
            "confirmation but does not activate the Candidate in Fast-Path Memory."
        ),
    }


def run_live_forecasting_two_skill_fast_path(
    root: Path,
    *,
    model: str,
    base_url: str,
) -> dict[str, object]:
    """Request one live same-task plan and deterministically compile it."""

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    general = _read_object(root / HISTORICAL_POLICY_CAPABILITY_PATH)
    missing = _read_object(root / MISSING_WINDOW_WEIGHTING_CAPABILITY_PATH)
    capability_views = [
        {
            "capability_id": card["capability_id"],
            "status": card["status"],
            "task_context": card["task_context"],
            "workflow_supply": card["workflow_supply"],
            "observation": card["observation"],
            "control": card["control"],
            "risk": card["risk"],
        }
        for card in (general, missing)
    ]
    allowed_steps = {
        "forecast_general_regular_panel": [
            "retrieve phase-aligned historical PolicyEpisodes for W_rowblock/W_curation/W_temporal_origin",
            "probe current Support in retrieved order",
            "stop on first positive; otherwise IDENTITY",
        ],
        "forecast_missing_mixed_reliable_history": [
            "observe natural-missing training-window composition",
            "retrieve Source PolicyEpisode to order ATTENUATE/EXCLUDE probes",
            "confirm on at least two phase-aligned historical Target origins",
            "stop on first positive; otherwise KEEP_ALL",
        ],
        "forecast_missing_insufficient_history": ["ABSTAIN_KEEP_ALL"],
        "forecast_missing_no_actionable_geometry": [],
        "classification_not_supported_by_forecast_skills": [],
    }
    payload = {
        "capabilities": capability_views,
        "contexts": _forecasting_two_skill_contexts(),
        "allowed_workflow_steps_by_context": allowed_steps,
        "required_output": {
            "decisions": [
                {
                    "context_id": "supplied context_id",
                    "capability_id": "supplied capability_id or null",
                    "decision": (
                        "RETRIEVE_AND_PROBE | RETRIEVE_AND_ABSTAIN | NO_APPLICABLE_SKILL"
                    ),
                    "workflow_steps": ["copy exact allowed strings"],
                    "reason_codes": ["short codes"],
                }
            ],
            "unadmitted_skills_selected": [],
            "new_skills_invented": [],
        },
    }
    system_prompt = (
        "You are the fast-path planner for a time-series data adaptation Harness. "
        "Two supplied Skills have the same Forecasting task, so select by the most "
        "specific observable applicability and Risk contract, not by task name alone. "
        "A pilot-supported Skill may be planned but never bypass its abstention guard. "
        "Never invent Skills, workflows, thresholds or outcomes. Return exactly one JSON "
        "object, one decision per context, and copy allowed workflow strings exactly. "
        "Do not include fields whose names contain outcome, utility or loss."
    )
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    choice = completion.choices[0]
    plan = json.loads(choice.message.content or "")
    if not isinstance(plan, dict):
        raise ValueError("same-task live planner must return one JSON object")
    plan["provider"] = {
        "base_url": base_url,
        "requested_model": model,
        "returned_model": getattr(completion, "model", ""),
        "prompt_tokens": getattr(getattr(completion, "usage", None), "prompt_tokens", None),
        "completion_tokens": getattr(
            getattr(completion, "usage", None), "completion_tokens", None
        ),
    }
    plan_path = root / FORECASTING_TWO_SKILL_LIVE_PLAN_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = _compile_forecasting_two_skill_plan(root, plan)
    report["provider"] = plan["provider"]
    return report


def _parse_noaa_channel(field: str, channel: str) -> float | None:
    if not field:
        return None
    parts = field.split(",")
    try:
        if channel == "WND":
            value = int(parts[3])
            return None if value == 9999 else value / 10.0
        value = int(parts[0])
    except (IndexError, TypeError, ValueError):
        return None
    missing = 99999 if channel == "SLP" else 9999
    return None if abs(value) == missing else value / 10.0


def _read_noaa_hourly_channels(
    np: Any,
    path: Path,
    *,
    year: int,
    series_length: int,
) -> dict[str, Any]:
    import csv
    from datetime import datetime

    channels = ("TMP", "DEW", "SLP", "WND")
    values = {
        channel: np.full(series_length, np.nan, dtype=np.float64)
        for channel in channels
    }
    start = datetime(year, 1, 1)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                timestamp = datetime.fromisoformat(row["DATE"]).replace(
                    minute=0, second=0, microsecond=0
                )
            except (KeyError, TypeError, ValueError):
                continue
            index = int((timestamp - start).total_seconds() // 3600)
            if not 0 <= index < series_length:
                continue
            for channel in channels:
                candidate = _parse_noaa_channel(str(row.get(channel, "")), channel)
                if candidate is not None and not np.isfinite(values[channel][index]):
                    values[channel][index] = candidate
    return values


def _linear_fill(np: Any, context: Any) -> Any | None:
    values = np.asarray(context, dtype=np.float64)
    observed = np.flatnonzero(np.isfinite(values))
    if observed.size < 2:
        return None
    return np.interp(np.arange(values.size), observed, values[observed])


def _missing_runs(mask: Any) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, missing in enumerate(list(mask) + [False]):
        if missing and start is None:
            start = index
        elif not missing and start is not None:
            runs.append((start, index))
            start = None
    return runs


def run_noaa_multichannel_local_repair_p0(
    root: Path,
    *,
    year: int = 2024,
    station_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Measure natural headroom/risk for one local peer-channel repair Program."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )

    import calendar

    series_length = 8784 if calendar.isleap(year) else 8760
    data_dir = root / f"data/benchmark_v0/raw/noaa_global_hourly/{year}"
    station_paths = (
        sorted(data_dir.glob("*.csv"))
        if station_ids is None
        else [data_dir / f"{station_id}.csv" for station_id in station_ids]
    )
    missing_files = [str(path) for path in station_paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(", ".join(missing_files))
    train_anchors = tuple(range(240, 6001, 24))
    evaluation_anchors = tuple(range(6240, series_length - HORIZON + 1, 48))
    station_results: list[dict[str, object]] = []
    rejected: dict[str, int] = {}
    consumer_fit_count = 0

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for station_path in station_paths:
        raw = _read_noaa_hourly_channels(
            np,
            station_path,
            year=year,
            series_length=series_length,
        )
        temperature = np.asarray(raw["TMP"], dtype=np.float64)
        dewpoint = np.asarray(raw["DEW"], dtype=np.float64)
        baseline_x: list[Any] = []
        candidate_x: list[Any] = []
        baseline_y: list[Any] = []
        candidate_y: list[Any] = []
        affected_rows = 0
        affected_missing_points = 0
        peer_bound_points = 0
        absolute_changes: list[float] = []

        for anchor in train_anchors:
            target = temperature[anchor : anchor + HORIZON]
            context = temperature[anchor - CONTEXT_LENGTH : anchor]
            dew_context = dewpoint[anchor - CONTEXT_LENGTH : anchor]
            if target.shape != (HORIZON,) or not np.isfinite(target).all():
                continue
            baseline = _linear_fill(np, context)
            if baseline is None:
                continue
            candidate = baseline.copy()
            missing = ~np.isfinite(context)
            row_is_eligible = False
            if bool(missing.any()):
                runs = _missing_runs(missing.tolist())
                short_local_geometry = bool(runs) and max(
                    end - start for start, end in runs
                ) <= 12
                peer_available = missing & np.isfinite(dew_context)
                peer_coverage = float(peer_available.sum() / missing.sum())
                paired = np.isfinite(context) & np.isfinite(dew_context)
                if short_local_geometry and peer_coverage >= 0.8 and paired.sum() >= 64:
                    robust_delta = float(np.median(context[paired] - dew_context[paired]))
                    peer_values = dew_context[peer_available] + robust_delta
                    candidate[peer_available] = peer_values
                    change = np.abs(candidate[peer_available] - baseline[peer_available])
                    if change.size and float(np.max(change)) > 1e-9:
                        row_is_eligible = True
                        affected_rows += 1
                        affected_missing_points += int(missing.sum())
                        peer_bound_points += int(peer_available.sum())
                        absolute_changes.extend(change.tolist())
            baseline_center, baseline_scale, baseline_method = _center_scale(np, baseline)
            candidate_center, candidate_scale, candidate_method = _center_scale(np, candidate)
            if (
                baseline_method == "scale_floor_fallback"
                or candidate_method == "scale_floor_fallback"
            ):
                continue
            baseline_x.append((baseline - baseline_center) / baseline_scale)
            candidate_x.append((candidate - candidate_center) / candidate_scale)
            baseline_y.append((target - baseline_center) / baseline_scale)
            candidate_y.append((target - candidate_center) / candidate_scale)

        if affected_rows < 2:
            reject("FEWER_THAN_TWO_AFFECTED_TRAINING_ROWS")
            continue
        if len(baseline_x) < CONTEXT_LENGTH + 8:
            reject("INSUFFICIENT_TRAINING_WINDOWS")
            continue

        eval_x: list[Any] = []
        actual: list[Any] = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal_scales: list[float] = []
        eval_anchor_ids: list[int] = []
        for anchor in evaluation_anchors:
            context = temperature[anchor - CONTEXT_LENGTH : anchor]
            target = temperature[anchor : anchor + HORIZON]
            if (
                context.shape != (CONTEXT_LENGTH,)
                or target.shape != (HORIZON,)
                or not np.isfinite(context).all()
                or not np.isfinite(target).all()
            ):
                continue
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                continue
            try:
                scale_value = seasonal_scale(
                    temperature[:anchor],
                    np.isfinite(temperature[:anchor]),
                    period=24,
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError):
                continue
            eval_x.append((context - center) / scale)
            actual.append(target)
            centers.append(center)
            scales.append(scale)
            seasonal_scales.append(scale_value)
            eval_anchor_ids.append(anchor)
        if len(eval_x) < 12:
            reject("INSUFFICIENT_CLEAN_EVALUATION_WINDOWS")
            continue

        weights = np.ones(len(baseline_x), dtype=np.float64)
        baseline_prediction = _exact_weighted_ridge_prediction(
            np,
            x_train=np.asarray(baseline_x, dtype=np.float64),
            targets=np.asarray(baseline_y, dtype=np.float64),
            weights=weights,
            x_eval=np.asarray(eval_x, dtype=np.float64),
        )
        candidate_prediction = _exact_weighted_ridge_prediction(
            np,
            x_train=np.asarray(candidate_x, dtype=np.float64),
            targets=np.asarray(candidate_y, dtype=np.float64),
            weights=weights,
            x_eval=np.asarray(eval_x, dtype=np.float64),
        )
        consumer_fit_count += 2
        baseline_losses: list[float] = []
        candidate_losses: list[float] = []
        for index, truth in enumerate(actual):
            baseline_raw = baseline_prediction[index] * scales[index] + centers[index]
            candidate_raw = candidate_prediction[index] * scales[index] + centers[index]
            baseline_losses.append(
                smase(truth, baseline_raw, scale=seasonal_scales[index])
            )
            candidate_losses.append(
                smase(truth, candidate_raw, scale=seasonal_scales[index])
            )
        support_indices = list(range(0, len(actual), 2))
        query_indices = list(range(1, len(actual), 2))

        def mean_at(values: list[float], indices: list[int]) -> float:
            return statistics.fmean(values[index] for index in indices)

        baseline_support = mean_at(baseline_losses, support_indices)
        candidate_support = mean_at(candidate_losses, support_indices)
        baseline_query = mean_at(baseline_losses, query_indices)
        candidate_query = mean_at(candidate_losses, query_indices)
        station_results.append(
            {
                "station_id": station_path.stem,
                "training_window_count": len(baseline_x),
                "affected_training_row_count": affected_rows,
                "affected_missing_point_count": affected_missing_points,
                "peer_bound_point_count": peer_bound_points,
                "peer_binding_coverage": peer_bound_points / affected_missing_points,
                "median_absolute_repair_delta": statistics.median(absolute_changes),
                "evaluation_window_count": len(actual),
                "evaluation_anchor_range": [min(eval_anchor_ids), max(eval_anchor_ids)],
                "baseline_support_loss": baseline_support,
                "candidate_support_loss": candidate_support,
                "support_gain": baseline_support - candidate_support,
                "baseline_query_loss": baseline_query,
                "candidate_query_loss": candidate_query,
                "query_gain": baseline_query - candidate_query,
                "combined_gain": statistics.fmean(baseline_losses)
                - statistics.fmean(candidate_losses),
            }
        )

    material_gain = 0.005
    positive = [row for row in station_results if row["combined_gain"] > material_gain]
    harmful = [row for row in station_results if row["combined_gain"] < -material_gain]
    identity_optimal = [row for row in station_results if row["combined_gain"] <= 0.0]
    passed = (
        len(station_results) >= 4
        and len(positive) >= 2
        and len(identity_optimal) >= 1
    )
    return {
        "experiment_id": f"E2-natural-NOAA-{year}-multichannel-local-repair-P0",
        "scientific_role": "natural Program headroom and matched-risk census",
        "dataset": f"NOAA Global Hourly {year}",
        "task": "forecasting",
        "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
        "metric": "per-origin sMASE",
        "program_family": {
            "incumbent": "TEMPORAL_LINEAR_FILL",
            "candidate": "LOCAL_DEWPOINT_DELTA_REPAIR",
            "target_channel": "TMP",
            "peer_channel": "DEW",
            "scope": "natural local missing runs <=12 hours with >=0.8 peer coverage",
            "binding": (
                "within-context median(TMP-DEW) on observed pairs; fill only originally "
                "missing TMP points where DEW is visible"
            ),
            "fallback": "TEMPORAL_LINEAR_FILL",
        },
        "information_boundary": {
            "repair_uses_current_context_only": True,
            "repair_reads_forecast_target": False,
            "evaluation_requires_clean_context_and_target": True,
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "development_only": True,
        },
        "station_results": station_results,
        "summary": {
            "source_file_count": len(station_paths),
            "eligible_station_count": len(station_results),
            "rejected_station_counts": rejected,
            "material_gain_threshold": material_gain,
            "positive_station_count": len(positive),
            "harmful_station_count": len(harmful),
            "identity_optimal_station_count": len(identity_optimal),
            "positive_station_ids": [row["station_id"] for row in positive],
            "harmful_station_ids": [row["station_id"] for row in harmful],
            "dataset_macro_combined_gain": (
                statistics.fmean(row["combined_gain"] for row in station_results)
                if station_results
                else None
            ),
            "maximum_combined_gain": (
                max(row["combined_gain"] for row in station_results)
                if station_results
                else None
            ),
            "minimum_combined_gain": (
                min(row["combined_gain"] for row in station_results)
                if station_results
                else None
            ),
        },
        "compute": {
            "consumer_fit_count": consumer_fit_count,
            "llm_api_call_count": 0,
            "proxy_call_count": 0,
        },
        "gate": {
            "at_least_four_eligible_stations": len(station_results) >= 4,
            "at_least_two_material_positive_stations": len(positive) >= 2,
            "at_least_one_identity_optimal_station": len(identity_optimal) >= 1,
            "passed": passed,
        },
        "verdict": (
            "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS"
            if passed
            else "PROGRAM_HEADROOM_OR_MATCHED_RISK_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Single-dataset exposed P0 only. A pass establishes natural Program "
            "headroom and risk, not a Witness, Capability, cross-dataset transfer, "
            "or LLM Harness-update result."
        ),
    }


def run_natural_missing_window_weighting_p0(
    root: Path,
    *,
    specs_override: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Test whether observable natural-missing windows should be downweighted."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )

    specs = specs_override or {
        "monash:nn5_daily": {
            "anchors": (240, 300, 360, 420),
            "evaluation_stop": 528,
            "period": 7,
        },
        "gefcom2012_load": {
            "anchors": (312, 372, 432, 492, 552, 612, 672, 732, 792, 852),
            "evaluation_stop": 912,
            "period": 24,
        },
        "noaa_global_hourly": {
            "anchors": (240, 300, 360, 420, 480, 540, 600, 660),
            "evaluation_stop": 720,
            "period": 24,
        },
    }
    registry = [
        json.loads(line)
        for line in (
            root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_dirs: dict[str, Path] = {}
    for record_path in (root / "data/benchmark_v0_2/clean_base").glob(
        "*/record.json"
    ):
        record = _read_object(record_path)
        record_dirs[str(record["series_uid"])] = record_path.parent

    dataset_results: list[dict[str, object]] = []
    total_consumer_fits = 0
    for dataset_id, spec in specs.items():
        evaluation_stop = int(spec["evaluation_stop"])
        required_stop = evaluation_stop + HORIZON
        candidates = sorted(
            (
                row
                for row in registry
                if row["dataset_id"] == dataset_id
                and int(row["length"]) >= required_stop
            ),
            key=lambda row: str(row["series_uid"]),
        )
        if len(candidates) < 20:
            raise ValueError(f"insufficient series for {dataset_id}")
        train_start = int(spec.get("train_start", 0))
        eval_start = int(spec.get("eval_start", 0))
        train_rows = candidates[train_start : train_start + 12]
        if len(train_rows) != 12:
            raise ValueError(f"fewer than twelve fixed training rows: {dataset_id}")
        train_uids = {str(row["series_uid"]) for row in train_rows}
        eval_rows: list[dict[str, object]] = []
        for row in candidates[eval_start:]:
            uid = str(row["series_uid"])
            if uid in train_uids:
                continue
            directory = record_dirs[uid]
            values = np.load(directory / "values.npy").astype(np.float64)
            context = values[
                evaluation_stop - CONTEXT_LENGTH : evaluation_stop
            ]
            target = values[evaluation_stop : evaluation_stop + HORIZON]
            if (
                target.shape == (HORIZON,)
                and np.isfinite(target).all()
                and np.isfinite(context).sum() >= CONTEXT_LENGTH // 2
            ):
                eval_rows.append(row)
            if len(eval_rows) == 8:
                break
        if len(eval_rows) != 8:
            raise ValueError(f"fewer than eight fixed-target evaluation rows: {dataset_id}")

        x_train: list[Any] = []
        y_train: list[Any] = []
        unreliable: list[bool] = []
        missing_fractions: list[float] = []
        for anchor in tuple(int(value) for value in spec["anchors"]):
            for row in train_rows:
                uid = str(row["series_uid"])
                directory = record_dirs[uid]
                values = np.load(directory / "values.npy").astype(np.float64)
                missing_mask = np.load(
                    directory / "natural_missing_mask.npy"
                ).astype(bool)
                context_raw = values[anchor - CONTEXT_LENGTH : anchor]
                target_raw = values[anchor : anchor + HORIZON]
                context = _linear_fill(np, context_raw)
                target = _linear_fill(np, target_raw)
                if context is None or target is None:
                    continue
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    continue
                window_mask = missing_mask[
                    anchor - CONTEXT_LENGTH : anchor + HORIZON
                ]
                x_train.append((context - center) / scale)
                y_train.append((target - center) / scale)
                unreliable.append(bool(window_mask.any()))
                missing_fractions.append(float(window_mask.mean()))
        if len(x_train) < 24 or not any(unreliable) or all(unreliable):
            raise ValueError(f"invalid missing-window training geometry: {dataset_id}")

        x_eval: list[Any] = []
        actual: list[Any] = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal_scales: list[float] = []
        for row in eval_rows:
            uid = str(row["series_uid"])
            directory = record_dirs[uid]
            values = np.load(directory / "values.npy").astype(np.float64)
            missing_mask = np.load(
                directory / "natural_missing_mask.npy"
            ).astype(bool)
            context_raw = values[
                evaluation_stop - CONTEXT_LENGTH : evaluation_stop
            ]
            target = values[evaluation_stop : evaluation_stop + HORIZON]
            context = _linear_fill(np, context_raw)
            if context is None or not np.isfinite(target).all():
                raise AssertionError("frozen evaluation admission became invalid")
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                raise ValueError(f"evaluation scale floor: {dataset_id}/{uid}")
            try:
                scale_value = seasonal_scale(
                    values[:evaluation_stop],
                    ~missing_mask[:evaluation_stop],
                    period=int(spec["period"]),
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as exc:
                raise ValueError(
                    f"undefined seasonal scale: {dataset_id}/{uid}"
                ) from exc
            x_eval.append((context - center) / scale)
            actual.append(target)
            centers.append(center)
            scales.append(scale)
            seasonal_scales.append(scale_value)

        x_array = np.asarray(x_train, dtype=np.float64)
        y_array = np.asarray(y_train, dtype=np.float64)
        historical_x_eval: list[Any] = []
        historical_actual: list[Any] = []
        historical_centers: list[float] = []
        historical_scales: list[float] = []
        historical_seasonal_scales: list[float] = []
        historical_origins: list[int] = []
        historical_origin_count = int(spec.get("historical_origin_count", 0))
        if historical_origin_count > 0:
            period = int(spec["period"])
            first_shift = max(1, (HORIZON + period - 1) // period)
            for row in eval_rows[4:]:
                uid = str(row["series_uid"])
                directory = record_dirs[uid]
                values = np.load(directory / "values.npy").astype(np.float64)
                missing_mask = np.load(
                    directory / "natural_missing_mask.npy"
                ).astype(bool)
                for shift in range(first_shift, first_shift + historical_origin_count):
                    origin = evaluation_stop - shift * period
                    context_raw = values[origin - CONTEXT_LENGTH : origin]
                    target = values[origin : origin + HORIZON]
                    context = _linear_fill(np, context_raw)
                    if context is None or target.shape != (HORIZON,) or not np.isfinite(target).all():
                        continue
                    center, scale, method = _center_scale(np, context)
                    if method == "scale_floor_fallback":
                        continue
                    try:
                        scale_value = seasonal_scale(
                            values[:origin],
                            ~missing_mask[:origin],
                            period=period,
                            min_pairs=32,
                        )
                    except (UndefinedSeasonalScale, ValueError):
                        continue
                    historical_x_eval.append((context - center) / scale)
                    historical_actual.append(target)
                    historical_centers.append(center)
                    historical_scales.append(scale)
                    historical_seasonal_scales.append(scale_value)
                    historical_origins.append(origin)
            if len(historical_x_eval) < 4:
                raise ValueError(
                    f"insufficient phase-aligned historical outcomes: {dataset_id}"
                )
        eval_array = np.asarray(x_eval + historical_x_eval, dtype=np.float64)
        unreliable_array = np.asarray(unreliable, dtype=bool)
        programs = (
            ("KEEP_ALL", 1.0),
            ("ATTENUATE_MISSING_WINDOW", 0.25),
            ("EXCLUDE_MISSING_WINDOW", 0.0),
        )
        program_rows: list[dict[str, object]] = []
        for program, unreliable_weight in programs:
            if unreliable_weight == 0.0:
                retained = ~unreliable_array
                prediction = _exact_weighted_ridge_prediction(
                    np,
                    x_train=x_array[retained],
                    targets=y_array[retained],
                    weights=np.ones(int(retained.sum()), dtype=np.float64),
                    x_eval=eval_array,
                )
            else:
                weights = np.ones(len(x_array), dtype=np.float64)
                weights[unreliable_array] = unreliable_weight
                prediction = _exact_weighted_ridge_prediction(
                    np,
                    x_train=x_array,
                    targets=y_array,
                    weights=weights,
                    x_eval=eval_array,
                )
            total_consumer_fits += 1
            losses: list[float] = []
            for index, truth in enumerate(actual):
                raw_prediction = prediction[index] * scales[index] + centers[index]
                losses.append(
                    smase(truth, raw_prediction, scale=seasonal_scales[index])
                )
            program_row = {
                    "program": program,
                    "unreliable_window_weight": unreliable_weight,
                    "support_loss": statistics.fmean(losses[:4]),
                    "query_loss": statistics.fmean(losses[4:]),
                    "combined_loss": statistics.fmean(losses),
                }
            if historical_x_eval:
                historical_losses = []
                for history_index, truth in enumerate(historical_actual):
                    prediction_index = len(actual) + history_index
                    raw_prediction = (
                        prediction[prediction_index] * historical_scales[history_index]
                        + historical_centers[history_index]
                    )
                    historical_losses.append(
                        smase(
                            truth,
                            raw_prediction,
                            scale=historical_seasonal_scales[history_index],
                        )
                    )
                program_row["historical_policy_loss"] = statistics.fmean(
                    historical_losses
                )
            program_rows.append(program_row)

        identity = next(row for row in program_rows if row["program"] == "KEEP_ALL")
        for row in program_rows:
            row["support_gain_vs_identity"] = (
                identity["support_loss"] - row["support_loss"]
            )
            row["query_gain_vs_identity"] = identity["query_loss"] - row["query_loss"]
            row["combined_gain_vs_identity"] = (
                identity["combined_loss"] - row["combined_loss"]
            )
            if "historical_policy_loss" in row:
                row["historical_policy_gain_vs_identity"] = (
                    identity["historical_policy_loss"]
                    - row["historical_policy_loss"]
                )
        best = max(program_rows, key=lambda row: row["combined_gain_vs_identity"])
        support_selected = min(program_rows, key=lambda row: row["support_loss"])
        dataset_results.append(
            {
                "dataset": dataset_id,
                "roster": {
                    "train_start": train_start,
                    "eval_start": eval_start,
                    "train": [str(row["series_uid"]) for row in train_rows],
                    "support": [str(row["series_uid"]) for row in eval_rows[:4]],
                    "query": [str(row["series_uid"]) for row in eval_rows[4:]],
                },
                "training_window_count": len(x_train),
                "unreliable_training_window_count": int(unreliable_array.sum()),
                "unreliable_training_window_fraction": float(unreliable_array.mean()),
                "median_missing_fraction_within_unreliable_windows": statistics.median(
                    value
                    for value, flag in zip(missing_fractions, unreliable)
                    if flag
                ),
                "phase_aligned_historical_policy_observation": (
                    None
                    if not historical_x_eval
                    else {
                        "evaluation_series": "current Query identities only",
                        "outcomes_end_before_current_query_cutoff": True,
                        "case_count": len(historical_x_eval),
                        "origins": sorted(set(historical_origins)),
                    }
                ),
                "programs": program_rows,
                "menu_oracle_program": best["program"],
                "menu_oracle_combined_gain": best["combined_gain_vs_identity"],
                "support_selected_program": support_selected["program"],
                "support_selected_query_gain": support_selected[
                    "query_gain_vs_identity"
                ],
            }
        )

    material_gain = 0.005
    positive_datasets = [
        row for row in dataset_results if row["menu_oracle_combined_gain"] > material_gain
    ]
    identity_optimal = [
        row for row in dataset_results if row["menu_oracle_program"] == "KEEP_ALL"
    ]
    passed = len(positive_datasets) >= 2 and len(identity_optimal) >= 1
    return {
        "experiment_id": "E2-natural-missing-window-weighting-P0",
        "scientific_role": "natural cohort-level Program headroom and matched-risk census",
        "task": "forecasting",
        "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
        "metric": "per-series sMASE; four Support and four Query series",
        "program_family": {
            "incumbent": "KEEP_ALL after minimal within-window linear fill",
            "candidates": [
                "ATTENUATE_MISSING_WINDOW(weight=0.25)",
                "EXCLUDE_MISSING_WINDOW(weight=0.0)",
            ],
            "scope": "training rows whose context or target intersects observable natural missing mask",
            "evaluation": "same minimal Context fill in all arms; evaluation future must be observed",
        },
        "dataset_results": dataset_results,
        "summary": {
            "dataset_count": len(dataset_results),
            "positive_headroom_dataset_count": len(positive_datasets),
            "identity_optimal_dataset_count": len(identity_optimal),
            "positive_headroom_datasets": [row["dataset"] for row in positive_datasets],
            "identity_optimal_datasets": [row["dataset"] for row in identity_optimal],
            "dataset_macro_menu_oracle_gain": statistics.fmean(
                row["menu_oracle_combined_gain"] for row in dataset_results
            ),
            "dataset_macro_support_selected_query_gain": statistics.fmean(
                row["support_selected_query_gain"] for row in dataset_results
            ),
        },
        "information_boundary": {
            "natural_missing_mask_is_observable": True,
            "program_reads_evaluation_future": False,
            "context_exposure": "INSTANCE_SEEN",
            "outcome_exposure": "EXPOSED",
            "development_only": True,
        },
        "compute": {
            "consumer_fit_count": total_consumer_fits,
            "llm_api_call_count": 0,
            "proxy_call_count": 0,
        },
        "gate": {
            "at_least_two_material_headroom_datasets": len(positive_datasets) >= 2,
            "at_least_one_identity_optimal_dataset": len(identity_optimal) >= 1,
            "passed": passed,
        },
        "verdict": (
            "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS"
            if passed
            else "PROGRAM_HEADROOM_OR_MATCHED_RISK_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed P0 only. A pass would establish natural Program headroom/risk, "
            "not a deployable Scope, Capability, Memory, LLM update or transfer result."
        ),
    }


def _missing_window_action_rows(
    dataset_episode: dict[str, object],
) -> dict[str, dict[str, float]]:
    """Read the two optional weighting actions from one complete episode."""

    actions: dict[str, dict[str, float]] = {}
    for row in dataset_episode["programs"]:
        program = str(row["program"])
        if program == "KEEP_ALL":
            continue
        actions[program] = {
            "support_gain": float(row["support_gain_vs_identity"]),
            "query_gain": float(row["query_gain_vs_identity"]),
            "combined_gain": float(row["combined_gain_vs_identity"]),
        }
        if "historical_policy_gain_vs_identity" in row:
            actions[program]["historical_gain"] = float(
                row["historical_policy_gain_vs_identity"]
            )
    expected = {"ATTENUATE_MISSING_WINDOW", "EXCLUDE_MISSING_WINDOW"}
    if set(actions) != expected:
        raise ValueError("missing-window Program supply changed")
    return actions


def _missing_window_context(dataset_episode: dict[str, object]) -> dict[str, float]:
    """Deployment-visible training-cohort missingness composition."""

    return {
        "unreliable_window_fraction": float(
            dataset_episode["unreliable_training_window_fraction"]
        ),
        "median_missing_fraction": float(
            dataset_episode["median_missing_fraction_within_unreliable_windows"]
        ),
    }


def _missing_window_context_distance(
    left: dict[str, float], right: dict[str, float]
) -> float:
    """Frozen, unit-range L1 distance; no fitted metric or Dataset identity."""

    return abs(
        left["unreliable_window_fraction"]
        - right["unreliable_window_fraction"]
    ) + abs(left["median_missing_fraction"] - right["median_missing_fraction"])


def _missing_window_probe_order(
    source_episode: dict[str, object],
) -> list[str]:
    """Compile a probe order from a completed Source PolicyEpisode."""

    actions = _missing_window_action_rows(source_episode)
    return sorted(
        actions,
        key=lambda program: (-actions[program]["combined_gain"], program),
    )


def _mean_missing_window_curves(
    curves: list[list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Expected target-only curve over both legal probe orders."""

    if not curves or any(len(curve) != 3 for curve in curves):
        raise ValueError("missing-window curves must cover B={0,1,2}")
    return [
        {
            "budget": budget,
            "fixed_query_gain": statistics.fmean(
                float(curve[budget]["fixed_query_gain"]) for curve in curves
            ),
            "expected_over_probe_orders": True,
        }
        for budget in range(3)
    ]


def _load_prsa_channel(root: Path, channel: str) -> dict[str, list[float]]:
    """Read one numeric PRSA channel directly from the official nested ZIP."""

    archive_path = (
        root
        / "data/benchmark_v0/raw/beijing_multisite/"
        "beijing_multi_site_air_quality.zip"
    )
    if not archive_path.exists():
        raise FileNotFoundError(f"missing official PRSA archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as outer:
        nested_payload = outer.read("PRSA2017_Data_20130301-20170228.zip")
    station_values: dict[str, list[float]] = {}
    with zipfile.ZipFile(io.BytesIO(nested_payload)) as nested:
        members = sorted(
            name for name in nested.namelist() if name.lower().endswith(".csv")
        )
        for member in members:
            with nested.open(member) as raw_stream:
                reader = csv.DictReader(io.TextIOWrapper(raw_stream, encoding="utf-8"))
                values = []
                station = None
                for row in reader:
                    station = str(row["station"])
                    if channel not in row:
                        raise ValueError(f"PRSA channel is unavailable: {channel}")
                    value = str(row[channel]).strip()
                    values.append(
                        float(value) if value and value.upper() != "NA" else float("nan")
                    )
            if not station or not values:
                raise ValueError(f"invalid PRSA station file: {member}")
            station_values[station] = values
    if len(station_values) != 12:
        raise ValueError("PRSA station supply changed")
    return station_values


def run_natural_missing_window_weighting_prsa_target(
    root: Path,
    *,
    channel: str = "PM2.5",
    risk_mode: str = "source_positive_prevalence",
) -> dict[str, object]:
    """Independent natural Target pilot for the missing-window Workflow."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )

    source_report = _read_object(root / MISSING_WINDOW_WEIGHTING_P0_REPORT_PATH)
    if source_report.get("verdict") != "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS":
        raise ValueError("PRSA Target requires the frozen P0 Source episodes")
    source_episodes = {
        str(row["dataset"]): row for row in source_report["dataset_results"]
    }
    values_by_station = {
        station: np.asarray(values, dtype=np.float64)
        for station, values in _load_prsa_channel(root, channel).items()
    }
    station_ids = sorted(values_by_station)
    train_stations = station_ids[:8]
    query_stations = station_ids[8:]
    anchors = (240, 408, 576, 744, 912, 1080, 1248, 1416)
    period = 24

    x_train: list[Any] = []
    y_train: list[Any] = []
    unreliable: list[bool] = []
    missing_fractions: list[float] = []
    for anchor in anchors:
        for station in train_stations:
            values = values_by_station[station]
            raw_window = values[anchor - CONTEXT_LENGTH : anchor + HORIZON]
            context = _linear_fill(np, raw_window[:CONTEXT_LENGTH])
            target = _linear_fill(np, raw_window[CONTEXT_LENGTH:])
            if context is None or target is None:
                continue
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                continue
            missing_mask = ~np.isfinite(raw_window)
            x_train.append((context - center) / scale)
            y_train.append((target - center) / scale)
            unreliable.append(bool(missing_mask.any()))
            missing_fractions.append(float(missing_mask.mean()))
    unreliable_array = np.asarray(unreliable, dtype=bool)
    if len(x_train) < 32 or not any(unreliable) or all(unreliable):
        raise ValueError("PRSA missing-window training geometry is not actionable")

    # Benchmark-owned admission: pick the first daily cutoff with fully observed
    # current and phase-aligned historical targets for all fixed Query stations.
    evaluation_stop = None
    historical_shifts = (2, 3, 4)
    latest_stop = min(len(values) for values in values_by_station.values()) - HORIZON
    for candidate_stop in range(2160, latest_stop + 1, period):
        admissible = True
        for station in query_stations:
            values = values_by_station[station]
            origins = (candidate_stop,) + tuple(
                candidate_stop - shift * period for shift in historical_shifts
            )
            for origin in origins:
                context = values[origin - CONTEXT_LENGTH : origin]
                target = values[origin : origin + HORIZON]
                if (
                    context.shape != (CONTEXT_LENGTH,)
                    or target.shape != (HORIZON,)
                    or np.isfinite(context).sum() < CONTEXT_LENGTH // 2
                    or not np.isfinite(target).all()
                ):
                    admissible = False
                    break
            if not admissible:
                break
        if admissible:
            evaluation_stop = candidate_stop
            break
    if evaluation_stop is None:
        raise ValueError("no fixed PRSA evaluation cutoff passed missingness admission")

    eval_features: list[Any] = []
    eval_actual: list[Any] = []
    eval_centers: list[float] = []
    eval_scales: list[float] = []
    eval_seasonal_scales: list[float] = []
    eval_roles: list[str] = []
    eval_origins: list[int] = []
    for station in query_stations:
        values = values_by_station[station]
        observed_mask = np.isfinite(values)
        origins = [("query", evaluation_stop)] + [
            ("historical", evaluation_stop - shift * period)
            for shift in historical_shifts
        ]
        for role, origin in origins:
            raw_context = values[origin - CONTEXT_LENGTH : origin]
            target = values[origin : origin + HORIZON]
            context = _linear_fill(np, raw_context)
            if context is None or not np.isfinite(target).all():
                raise AssertionError("PRSA evaluation admission changed after freeze")
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                raise ValueError("PRSA evaluation context hit the scale floor")
            try:
                scale_value = seasonal_scale(
                    values[:origin], observed_mask[:origin], period=period, min_pairs=32
                )
            except (UndefinedSeasonalScale, ValueError) as exc:
                raise ValueError("PRSA seasonal scale is undefined") from exc
            eval_features.append((context - center) / scale)
            eval_actual.append(target)
            eval_centers.append(center)
            eval_scales.append(scale)
            eval_seasonal_scales.append(scale_value)
            eval_roles.append(role)
            eval_origins.append(origin)

    x_array = np.asarray(x_train, dtype=np.float64)
    y_array = np.asarray(y_train, dtype=np.float64)
    eval_array = np.asarray(eval_features, dtype=np.float64)
    programs = (
        ("KEEP_ALL", 1.0),
        ("ATTENUATE_MISSING_WINDOW", 0.25),
        ("EXCLUDE_MISSING_WINDOW", 0.0),
    )
    program_rows = []
    for program, unreliable_weight in programs:
        if unreliable_weight == 0.0:
            retained = ~unreliable_array
            prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=x_array[retained],
                targets=y_array[retained],
                weights=np.ones(int(retained.sum()), dtype=np.float64),
                x_eval=eval_array,
            )
        else:
            weights = np.ones(len(x_array), dtype=np.float64)
            weights[unreliable_array] = unreliable_weight
            prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=x_array,
                targets=y_array,
                weights=weights,
                x_eval=eval_array,
            )
        losses = []
        for index, truth in enumerate(eval_actual):
            raw_prediction = prediction[index] * eval_scales[index] + eval_centers[index]
            losses.append(
                smase(truth, raw_prediction, scale=eval_seasonal_scales[index])
            )
        query_losses = [
            loss for loss, role in zip(losses, eval_roles) if role == "query"
        ]
        historical_losses = [
            loss for loss, role in zip(losses, eval_roles) if role == "historical"
        ]
        program_rows.append(
            {
                "program": program,
                "unreliable_window_weight": unreliable_weight,
                "query_loss": statistics.fmean(query_losses),
                "historical_policy_loss": statistics.fmean(historical_losses),
            }
        )
    identity = next(row for row in program_rows if row["program"] == "KEEP_ALL")
    for row in program_rows:
        row["query_gain_vs_identity"] = identity["query_loss"] - row["query_loss"]
        row["historical_policy_gain_vs_identity"] = (
            identity["historical_policy_loss"] - row["historical_policy_loss"]
        )
    actions = {
        str(row["program"]): {
            "query_gain": float(row["query_gain_vs_identity"]),
            "support_gain": float(row["historical_policy_gain_vs_identity"]),
        }
        for row in program_rows
        if row["program"] != "KEEP_ALL"
    }

    target_context = {
        "unreliable_window_fraction": float(unreliable_array.mean()),
        "median_missing_fraction": statistics.median(
            fraction
            for fraction, flag in zip(missing_fractions, unreliable)
            if flag
        ),
    }
    ranked_sources = sorted(
        [
            {
                "dataset": dataset_id,
                "episode": episode,
                "context": _missing_window_context(episode),
                "distance": _missing_window_context_distance(
                    target_context, _missing_window_context(episode)
                ),
            }
            for dataset_id, episode in source_episodes.items()
        ],
        key=lambda row: (float(row["distance"]), str(row["dataset"])),
    )
    retrieved = ranked_sources[0]
    second_distance = float(ranked_sources[1]["distance"])
    retrieval_ratio = float(retrieved["distance"]) / max(second_distance, 1e-12)
    positive_source_episodes = [
        row
        for row in source_episodes.values()
        if float(row["menu_oracle_combined_gain"]) > 0.005
    ]
    source_scope_floor = min(
        float(row["unreliable_training_window_fraction"])
        for row in positive_source_episodes
    )
    if risk_mode == "source_positive_prevalence":
        scope_definition = (
            "unreliable-window fraction >= minimum positive Source episode"
        )
        scope_eligible = (
            target_context["unreliable_window_fraction"] >= source_scope_floor
        )
    elif risk_mode == "retrieval_ambiguity":
        scope_definition = (
            "nearest Source Context must be at least twice as close as second-nearest"
        )
        scope_eligible = retrieval_ratio <= 0.5
    elif risk_mode == "historical_origin_coverage":
        scope_definition = (
            "historical confirmation must cover at least two distinct phase-aligned origins"
        )
        scope_eligible = len(
            set(
                origin
                for role, origin in zip(eval_roles, eval_origins)
                if role == "historical"
            )
        ) >= 2
    else:
        raise ValueError(f"unsupported PRSA risk mode: {risk_mode}")
    a5_order = _missing_window_probe_order(retrieved["episode"])
    orders = [
        ["ATTENUATE_MISSING_WINDOW", "EXCLUDE_MISSING_WINDOW"],
        ["EXCLUDE_MISSING_WINDOW", "ATTENUATE_MISSING_WINDOW"],
    ]
    if scope_eligible:
        a5_curve = _semantic_auxiliary_budget_curve(a5_order, actions)
        a3_order_curves = [
            _semantic_auxiliary_budget_curve(order, actions) for order in orders
        ]
    else:
        no_action = {
            program: {"support_gain": 0.0, "query_gain": 0.0}
            for program in actions
        }
        a5_curve = _semantic_auxiliary_budget_curve(a5_order, no_action)
        a3_order_curves = [
            _semantic_auxiliary_budget_curve(order, no_action) for order in orders
        ]
    a3_curve = _mean_missing_window_curves(a3_order_curves)
    a3_auc = policy_episode_adapt_auc(a3_curve)
    a5_auc = policy_episode_adapt_auc(a5_curve)
    source_actions = _missing_window_action_rows(retrieved["episode"])
    a4_program = a5_order[0]
    if not scope_eligible or source_actions[a4_program]["combined_gain"] <= 0.0:
        a4_program = "KEEP_ALL"
    a4_query_gain = 0.0 if a4_program == "KEEP_ALL" else actions[a4_program]["query_gain"]
    menu_oracle = max(
        [("KEEP_ALL", 0.0)]
        + [(program, row["query_gain"]) for program, row in actions.items()],
        key=lambda item: (float(item[1]), item[0]),
    )
    harmful_a5 = any(
        float(row["fixed_query_gain"]) < -0.005 for row in a5_curve[1:]
    )
    passed = (
        scope_eligible
        and float(menu_oracle[1]) > 0.005
        and a5_auc > a3_auc
        and not harmful_a5
    )
    return {
        "experiment_id": (
            "E2-natural-missing-window-weighting-PRSA-target"
            if channel == "PM2.5" and risk_mode == "source_positive_prevalence"
            else f"E2-natural-missing-window-weighting-PRSA-{channel}-{risk_mode}"
        ),
        "scientific_role": "independent natural Target pilot",
        "dataset": "UCI Beijing Multi-Site Air Quality",
        "source": {
            "url": "https://archive.ics.uci.edu/dataset/501/beijing",
            "license": "CC BY 4.0",
            "missing_semantics": "NA in the official station CSVs",
        },
        "task": f"{channel} forecasting",
        "roster": {"train_stations": train_stations, "query_stations": query_stations},
        "geometry": {
            "training_anchors": list(anchors),
            "evaluation_stop": evaluation_stop,
            "historical_origins": sorted(
                set(origin for role, origin in zip(eval_roles, eval_origins) if role == "historical")
            ),
            "current_query_future_used_for_planning_or_confirmation": False,
        },
        "target_context": target_context,
        "scope": {
            "risk_mode": risk_mode,
            "definition": scope_definition,
            "source_derived_floor": source_scope_floor,
            "nearest_to_second_distance_ratio": retrieval_ratio,
            "distinct_historical_origin_count": len(
                set(
                    origin
                    for role, origin in zip(eval_roles, eval_origins)
                    if role == "historical"
                )
            ),
            "eligible": scope_eligible,
        },
        "retrieved_source_episode": {
            "dataset_audit_only": retrieved["dataset"],
            "context": retrieved["context"],
            "distance": retrieved["distance"],
            "second_nearest_distance": second_distance,
            "probe_order": a5_order,
        },
        "programs": program_rows,
        "arms": {
            "A3": {
                "mean_curve": a3_curve,
                "adapt_auc": a3_auc,
            },
            "A4": {
                "selected_program": a4_program,
                "fixed_query_gain": a4_query_gain,
                "harmful": a4_query_gain < -0.005,
            },
            "A5": {
                "curve": a5_curve,
                "adapt_auc": a5_auc,
                "harmful": harmful_a5,
            },
        },
        "summary": {
            "training_window_count": len(x_train),
            "unreliable_training_window_count": int(unreliable_array.sum()),
            "menu_oracle_program_evaluator_only": menu_oracle[0],
            "menu_oracle_query_gain_evaluator_only": menu_oracle[1],
            "A5_minus_A3": a5_auc - a3_auc,
        },
        "information_boundary": {
            "context_exposure_before_run": "UNSEEN",
            "outcome_exposure_after_run": "EXPOSED",
            "benchmark_owned_missingness_admission": True,
        },
        "compute": {"consumer_fit_count": 3, "llm_api_call_count": 0},
        "gate": {
            "source_scope_eligible": scope_eligible,
            "material_query_headroom": float(menu_oracle[1]) > 0.005,
            "A5_strictly_greater_than_A3": a5_auc > a3_auc,
            "A5_harmful_false": not harmful_a5,
            "passed": passed,
        },
        "verdict": (
            "INDEPENDENT_NATURAL_MISSING_WINDOW_SKILL_PILOT_PASS"
            if passed
            else "INDEPENDENT_NATURAL_MISSING_WINDOW_SKILL_PILOT_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "One independent natural Target pilot; a pass does not yet establish a "
            "multi-Target promoted Capability or general missing-data quality rule."
        ),
    }


def run_natural_missing_window_weighting_prsa_risk_replay(
    root: Path,
) -> dict[str, object]:
    """Fresh channel-role replay of the failure-derived retrieval Risk guard."""

    role_reports = [
        run_natural_missing_window_weighting_prsa_target(
            root, channel=channel, risk_mode="retrieval_ambiguity"
        )
        for channel in ("NO2", "O3")
    ]
    a3_macro = statistics.fmean(
        float(row["arms"]["A3"]["adapt_auc"]) for row in role_reports
    )
    a5_macro = statistics.fmean(
        float(row["arms"]["A5"]["adapt_auc"]) for row in role_reports
    )
    material_roles = sum(
        float(row["summary"]["menu_oracle_query_gain_evaluator_only"]) > 0.005
        for row in role_reports
    )
    helpful_roles = sum(
        float(row["summary"]["A5_minus_A3"]) > 0.0 for row in role_reports
    )
    harmful_roles = sum(bool(row["arms"]["A5"]["harmful"]) for row in role_reports)
    passed = (
        material_roles >= 1
        and helpful_roles >= 1
        and a5_macro >= a3_macro
        and harmful_roles == 0
    )
    return {
        "experiment_id": "E2-natural-missing-window-weighting-PRSA-risk-replay",
        "scientific_role": "new natural task-role validation of one Risk update",
        "failure_derived_update": {
            "rejected_scope": "high unreliable-window prevalence only",
            "new_risk": (
                "UNRESOLVED when the nearest Source Context is not at least twice "
                "as close as the second-nearest; otherwise retrieve and probe"
            ),
            "unchanged": [
                "Program supply",
                "Source episodes",
                "historical Target confirmation",
                "stop-on-first-positive",
                "Consumer",
            ],
        },
        "target_roles": role_reports,
        "summary": {
            "task_role_count": len(role_reports),
            "material_headroom_role_count": material_roles,
            "A5_helpful_role_count": helpful_roles,
            "A5_harmful_role_count": harmful_roles,
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
        },
        "information_boundary": {
            "NO2_and_O3_outcomes_unseen_before_frozen_run": True,
            "PM2.5_outcome_used_only_for_failure_localization": True,
            "query_future_used_for_planning_or_confirmation": False,
        },
        "compute": {"consumer_fit_count": 6, "llm_api_call_count": 0},
        "gate": {
            "at_least_one_role_has_material_headroom": material_roles >= 1,
            "at_least_one_role_A5_strictly_improves_A3": helpful_roles >= 1,
            "A5_macro_not_below_A3": a5_macro >= a3_macro,
            "A5_harmful_role_count_zero": harmful_roles == 0,
            "passed": passed,
        },
        "verdict": (
            "RETRIEVAL_AMBIGUITY_RISK_UPDATE_PASS"
            if passed
            else "RETRIEVAL_AMBIGUITY_RISK_UPDATE_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Two new channel roles from one independent natural dataset; a pass is a "
            "Risk/Workflow premise, not multi-dataset Capability Promotion."
        ),
    }


def run_natural_missing_window_weighting_origin_coverage(
    root: Path,
) -> dict[str, object]:
    """Final fresh replay of a reliability guard derived from the P2 harm."""

    role_reports = []
    feasibility_rejections = []
    for channel in ("SO2", "CO"):
        try:
            role_reports.append(
                run_natural_missing_window_weighting_prsa_target(
                    root, channel=channel, risk_mode="historical_origin_coverage"
                )
            )
        except ValueError as exc:
            feasibility_rejections.append({"channel": channel, "reason": str(exc)})
    if not role_reports:
        return {
            "experiment_id": "E2-natural-missing-window-weighting-origin-coverage",
            "scientific_role": "final new-role replay for the current family version",
            "target_roles": [],
            "feasibility_rejections": feasibility_rejections,
            "compute": {"consumer_fit_count": 0, "llm_api_call_count": 0},
            "verdict": "NO_ACTIONABLE_FRESH_ROLE_CLOSE_FAMILY",
            "capability_or_memory_written": False,
        }
    a3_macro = statistics.fmean(
        float(row["arms"]["A3"]["adapt_auc"]) for row in role_reports
    )
    a5_macro = statistics.fmean(
        float(row["arms"]["A5"]["adapt_auc"]) for row in role_reports
    )
    material_roles = sum(
        float(row["summary"]["menu_oracle_query_gain_evaluator_only"]) > 0.005
        for row in role_reports
    )
    helpful_roles = sum(
        float(row["summary"]["A5_minus_A3"]) > 0.0 for row in role_reports
    )
    harmful_roles = sum(bool(row["arms"]["A5"]["harmful"]) for row in role_reports)
    passed = (
        material_roles >= 1
        and helpful_roles >= 1
        and a5_macro > a3_macro
        and harmful_roles == 0
    )
    return {
        "experiment_id": "E2-natural-missing-window-weighting-origin-coverage",
        "scientific_role": "final new-role replay for the current family version",
        "failure_derived_update": {
            "first_fault": (
                "the only harmful historical confirmation was supported by one distinct "
                "origin, while useful episodes covered multiple phase-aligned origins"
            ),
            "patch": "ADD_RISK(min_distinct_historical_origins=2)",
            "unchanged": [
                "Program supply",
                "Source Context retrieval",
                "historical Target confirmation",
                "stop-on-first-positive",
                "Consumer",
            ],
        },
        "target_roles": role_reports,
        "feasibility_rejections": feasibility_rejections,
        "summary": {
            "task_role_count": len(role_reports),
            "material_headroom_role_count": material_roles,
            "A5_helpful_role_count": helpful_roles,
            "A5_harmful_role_count": harmful_roles,
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
        },
        "information_boundary": {
            "SO2_and_CO_outcomes_unseen_before_frozen_run": True,
            "earlier_PRSA_roles_used_only_for_failure_localization": True,
            "query_future_used_for_planning_or_confirmation": False,
        },
        "compute": {
            "consumer_fit_count": 3 * len(role_reports),
            "llm_api_call_count": 0,
        },
        "gate": {
            "at_least_one_role_has_material_headroom": material_roles >= 1,
            "at_least_one_role_A5_strictly_improves_A3": helpful_roles >= 1,
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_harmful_role_count_zero": harmful_roles == 0,
            "passed": passed,
        },
        "verdict": (
            "HISTORICAL_ORIGIN_COVERAGE_RISK_PASS"
            if passed
            else "HISTORICAL_ORIGIN_COVERAGE_RISK_FAIL_CLOSE_FAMILY"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "New task roles from one external natural dataset. A pass supports a "
            "provisional Skill candidate; a fail closes this missing-window family version."
        ),
    }


def _load_uci_air_quality_channel(root: Path, channel: str) -> list[float]:
    """Read one channel and preserve the official -200 missing semantics."""

    archive_path = root / "data/benchmark_v0/raw/uci_air_quality/air_quality.zip"
    if not archive_path.exists():
        raise FileNotFoundError(f"missing official UCI Air Quality archive: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        payload = archive.read("AirQualityUCI.csv").decode("latin-1")
    rows = csv.reader(io.StringIO(payload), delimiter=";")
    header = next(rows)
    if channel not in header:
        raise ValueError(f"UCI Air Quality channel is unavailable: {channel}")
    channel_index = header.index(channel)
    values: list[float] = []
    for row in rows:
        if not row or not row[0].strip() or channel_index >= len(row):
            continue
        field = row[channel_index].strip().replace(",", ".")
        try:
            value = float(field)
        except ValueError:
            value = float("nan")
        values.append(float("nan") if value == -200.0 else value)
    if len(values) < 9000:
        raise ValueError("UCI Air Quality hourly supply changed")
    return values


def _run_natural_imputation_air_quality_role(
    root: Path,
    *,
    channel: str,
    source_order: tuple[str, ...],
) -> dict[str, object]:
    """Evaluate the frozen imputation Candidate on one exposed target role."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.operators.registry import get_operator

    expected_order = (
        "W_period_median_imputation",
        "W_ar_imputation",
    )
    if source_order != expected_order:
        raise ValueError("the frozen Source imputation Workflow order changed")

    values = np.asarray(_load_uci_air_quality_channel(root, channel), dtype=np.float64)
    observed_mask = np.isfinite(values)
    period = 24
    historical_shifts = (2, 3, 4)

    # Availability alone chooses the evaluation origin. Program outcomes never do.
    latest_stop = ((len(values) - HORIZON) // period) * period
    evaluation_stop = None
    for stop in range(latest_stop, CONTEXT_LENGTH + 8 * period, -period):
        origins = [stop] + [stop - shift * period for shift in historical_shifts]
        if min(origins) <= CONTEXT_LENGTH:
            continue
        if all(
            np.isfinite(values[origin : origin + HORIZON]).all()
            and np.isfinite(values[origin - CONTEXT_LENGTH : origin]).sum()
            >= CONTEXT_LENGTH // 2
            for origin in origins
        ):
            evaluation_stop = stop
            break
    if evaluation_stop is None:
        raise ValueError("NO_ACTIONABLE_EVALUATION_ORIGIN")

    earliest_historical_origin = evaluation_stop - max(historical_shifts) * period
    raw_training_windows = []
    run_lengths: list[int] = []
    missing_points = 0
    total_points = 0
    for anchor in range(240, earliest_historical_origin - HORIZON, 72):
        raw = values[anchor - CONTEXT_LENGTH : anchor + HORIZON]
        if raw.shape != (CONTEXT_LENGTH + HORIZON,):
            continue
        if np.isfinite(raw).sum() < raw.size // 2:
            continue
        missing = ~np.isfinite(raw)
        missing_points += int(missing.sum())
        total_points += int(raw.size)
        run_lengths.extend(end - start for start, end in _missing_runs(missing.tolist()))
        raw_training_windows.append(raw)
    if len(raw_training_windows) < 24 or missing_points == 0:
        raise ValueError("NO_ACTIONABLE_NATURAL_MISSING_TRAINING_WINDOWS")

    eval_origins = [
        *(('support', evaluation_stop - shift * period) for shift in historical_shifts),
        ('query', evaluation_stop),
    ]
    operator_by_workflow = {
        "IDENTITY": get_operator("impute_linear"),
        "W_period_median_imputation": get_operator("period_median_complete"),
        "W_ar_imputation": get_operator("impute_ar"),
    }
    program_rows: list[dict[str, object]] = []
    for workflow_id in ("IDENTITY", *source_order):
        operator = operator_by_workflow[workflow_id]
        x_train = []
        y_train = []
        for raw in raw_training_windows:
            filled = (
                operator(raw)
                if workflow_id == "IDENTITY"
                else operator(raw, period=period)
            )
            if not np.isfinite(filled).all():
                continue
            context = filled[:CONTEXT_LENGTH]
            target = filled[CONTEXT_LENGTH:]
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                continue
            x_train.append((context - center) / scale)
            y_train.append((target - center) / scale)
        if len(x_train) < 24:
            raise ValueError(f"INSUFFICIENT_IMPUTED_TRAINING_ROWS:{workflow_id}")

        x_eval = []
        actual = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal_scales: list[float] = []
        roles: list[str] = []
        for role, origin in eval_origins:
            raw_context = values[origin - CONTEXT_LENGTH : origin]
            target = values[origin : origin + HORIZON]
            context = (
                operator(raw_context)
                if workflow_id == "IDENTITY"
                else operator(raw_context, period=period)
            )
            if not np.isfinite(context).all() or not np.isfinite(target).all():
                raise ValueError(f"NON_FINITE_EVALUATION_ROLE:{role}")
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                raise ValueError(f"EVALUATION_SCALE_FLOOR:{role}")
            try:
                scale_value = seasonal_scale(
                    values[:origin],
                    observed_mask[:origin],
                    period=period,
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as exc:
                raise ValueError(f"UNDEFINED_SEASONAL_SCALE:{role}") from exc
            x_eval.append((context - center) / scale)
            actual.append(target)
            centers.append(center)
            scales.append(scale)
            seasonal_scales.append(scale_value)
            roles.append(role)

        prediction = _exact_weighted_ridge_prediction(
            np,
            x_train=np.asarray(x_train, dtype=np.float64),
            targets=np.asarray(y_train, dtype=np.float64),
            weights=np.ones(len(x_train), dtype=np.float64),
            x_eval=np.asarray(x_eval, dtype=np.float64),
        )
        losses = []
        for index, truth in enumerate(actual):
            raw_prediction = prediction[index] * scales[index] + centers[index]
            losses.append(smase(truth, raw_prediction, scale=seasonal_scales[index]))
        support_losses = [loss for loss, role in zip(losses, roles) if role == "support"]
        query_losses = [loss for loss, role in zip(losses, roles) if role == "query"]
        program_rows.append(
            {
                "workflow_id": workflow_id,
                "support_loss": statistics.fmean(support_losses),
                "query_loss": statistics.fmean(query_losses),
                "combined_loss": statistics.fmean(losses),
                "training_row_count": len(x_train),
            }
        )

    identity = next(row for row in program_rows if row["workflow_id"] == "IDENTITY")
    for row in program_rows:
        row["support_gain_vs_identity"] = (
            identity["support_loss"] - row["support_loss"]
        )
        row["query_gain_vs_identity"] = identity["query_loss"] - row["query_loss"]
        row["combined_gain_vs_identity"] = (
            identity["combined_loss"] - row["combined_loss"]
        )
    responses = {
        str(row["workflow_id"]): {
            "workflow_id": row["workflow_id"],
            "support_gain": float(row["support_gain_vs_identity"]),
            "query_gain": float(row["query_gain_vs_identity"]),
        }
        for row in program_rows
        if row["workflow_id"] != "IDENTITY"
    }
    a5_curve = workflow_curve_from_policy_episode(responses, source_order)
    target_only_curves = [
        workflow_curve_from_policy_episode(responses, order)
        for order in permutations(source_order)
    ]
    a3_curve = [
        {
            "budget": budget,
            "fixed_query_gain": statistics.fmean(
                float(curve[budget]["fixed_query_gain"])
                for curve in target_only_curves
            ),
        }
        for budget in range(len(source_order) + 1)
    ]
    a5_auc = policy_episode_adapt_auc(a5_curve)
    a3_auc = policy_episode_adapt_auc(a3_curve)
    a5_harm_count = sum(float(row["fixed_query_gain"]) < 0.0 for row in a5_curve[1:])
    menu_oracle = max(
        [("IDENTITY", 0.0)]
        + [
            (str(row["workflow_id"]), float(row["query_gain_vs_identity"]))
            for row in program_rows
            if row["workflow_id"] != "IDENTITY"
        ],
        key=lambda item: (item[1], item[0]),
    )
    behavior_differs = any(
        abs(
            float(a5_curve[index]["fixed_query_gain"])
            - float(a3_curve[index]["fixed_query_gain"])
        )
        > 1e-12
        for index in range(1, len(a5_curve))
    )
    return {
        "channel": channel,
        "public_context": {
            "natural_missing_values_present": True,
            "missing_fraction": missing_points / total_points,
            "maximum_run_length": max(run_lengths) if run_lengths else 0,
            "period": period,
            "training_window_count": len(raw_training_windows),
        },
        "geometry": {
            "evaluation_stop": evaluation_stop,
            "support_origins": [origin for role, origin in eval_origins if role == "support"],
            "query_origin": evaluation_stop,
            "support_feedback": "historical_observed_forecasting_outcome",
            "query_outcome_role": "EVALUATOR_ONLY",
            "repair_truth_used": False,
            "query_target_values_used_for_planning_or_confirmation": False,
            "query_target_availability_used_for_retrospective_admission": True,
        },
        "programs": program_rows,
        "adaptation": {
            "A5_source_order_curve": a5_curve,
            "A3_equal_budget_order_average_curve": a3_curve,
            "A5_adaptation_auc": a5_auc,
            "A3_adaptation_auc": a3_auc,
            "A5_minus_A3": a5_auc - a3_auc,
            "A5_harm_count": a5_harm_count,
        },
        "validation": {
            "material_query_headroom": float(menu_oracle[1]) > 0.005,
            "menu_oracle_workflow_evaluator_only": menu_oracle[0],
            "menu_oracle_query_gain_evaluator_only": menu_oracle[1],
            "source_order_changes_early_behavior": behavior_differs,
            "A5_above_equal_budget_A3": a5_auc > a3_auc,
            "A5_harm_zero": a5_harm_count == 0,
        },
    }


def run_natural_imputation_air_quality_target_pilot(root: Path) -> dict[str, object]:
    """Test the frozen Source imputation order on exposed UCI target roles."""

    source = _read_object(root / NATURAL_IMPUTATION_COLD_START_REPORT_PATH)
    source_candidate = source.get("candidate")
    if (
        source.get("verdict") != "NATURAL_COLD_START_CANDIDATE_PREMISE_PASS"
        or not isinstance(source_candidate, dict)
    ):
        raise ValueError("the Source imputation Candidate is unavailable")
    source_order = tuple(
        str(value) for value in source_candidate["source_prior"]["workflow_order"]
    )
    role_reports = []
    feasibility_rejections = []
    for channel in ("CO(GT)", "NO2(GT)"):
        try:
            role_reports.append(
                _run_natural_imputation_air_quality_role(
                    root,
                    channel=channel,
                    source_order=source_order,
                )
            )
        except ValueError as exc:
            feasibility_rejections.append({"channel": channel, "reason": str(exc)})

    if not role_reports:
        return {
            "experiment_id": "E2.79-natural-imputation-UCI-Air-Quality-Target-pilot",
            "scientific_role": "Source-order versus Target-only adaptation pilot",
            "dataset": "UCI Air Quality",
            "target_roles": [],
            "feasibility_rejections": feasibility_rejections,
            "context_exposure": "INSTANCE_SEEN_IN_OTHER_PROGRAM_FAMILIES",
            "outcome_exposure": "EXPOSED_BY_THIS_PILOT",
            "compute": {"consumer_fit_count": 0, "llm_api_call_count": 0},
            "verdict": "NATURAL_IMPUTATION_TARGET_PILOT_INCONCLUSIVE",
            "capability_or_memory_written": False,
        }

    a5_macro = statistics.fmean(
        float(row["adaptation"]["A5_adaptation_auc"]) for row in role_reports
    )
    a3_macro = statistics.fmean(
        float(row["adaptation"]["A3_adaptation_auc"]) for row in role_reports
    )
    material_roles = sum(
        bool(row["validation"]["material_query_headroom"]) for row in role_reports
    )
    behavior_roles = sum(
        bool(row["validation"]["source_order_changes_early_behavior"])
        for row in role_reports
    )
    harmful_roles = sum(
        int(row["adaptation"]["A5_harm_count"]) > 0 for row in role_reports
    )
    passed = bool(
        material_roles >= 1
        and behavior_roles >= 1
        and a5_macro > a3_macro
        and harmful_roles == 0
    )
    return {
        "experiment_id": "E2.79-natural-imputation-UCI-Air-Quality-Target-pilot",
        "scientific_role": (
            "exposed natural Target pilot of a frozen Source Workflow order versus "
            "equal-budget Target-only probe ordering"
        ),
        "dataset": "UCI Air Quality",
        "source": {
            "url": "https://archive.ics.uci.edu/dataset/360/air+quality",
            "license": "CC BY 4.0",
            "missing_semantics": "official -200 missing marker converted to NaN",
        },
        "source_candidate": {
            "report": NATURAL_IMPUTATION_COLD_START_REPORT_PATH,
            "status": source_candidate["status"],
            "workflow_order": list(source_order),
        },
        "target_roles": role_reports,
        "feasibility_rejections": feasibility_rejections,
        "context_exposure": "INSTANCE_SEEN_IN_OTHER_PROGRAM_FAMILIES",
        "outcome_exposure": "EXPOSED_BY_THIS_PILOT",
        "summary": {
            "role_count": len(role_reports),
            "material_headroom_role_count": material_roles,
            "source_order_behavior_change_role_count": behavior_roles,
            "A5_harmful_role_count": harmful_roles,
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
        },
        "information_boundary": {
            "repair_truth_used": False,
            "support_feedback": "historical_observed_forecasting_outcome",
            "query_outcome_role": "EVALUATOR_ONLY",
            "query_target_values_used_for_planning_or_confirmation": False,
            "query_target_availability_used_for_retrospective_admission": True,
            "target_context_or_outcome_fresh_for_project": False,
        },
        "compute": {
            "consumer_fit_count": 3 * len(role_reports),
            "llm_api_call_count": 0,
        },
        "gate": {
            "at_least_one_role_has_material_headroom": material_roles >= 1,
            "source_order_changes_behavior": behavior_roles >= 1,
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_harmful_role_count_zero": harmful_roles == 0,
            "passed": passed,
        },
        "verdict": (
            "NATURAL_IMPUTATION_TARGET_PILOT_PASS"
            if passed
            else "NATURAL_IMPUTATION_TARGET_PILOT_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "This Target was exposed in another Program family. A PASS supports the "
            "mechanism and authorizes independent confirmation, but cannot promote "
            "or activate the Candidate."
        ),
    }


def run_live_natural_imputation_scope_repair(
    root: Path,
    *,
    model: str,
    base_url: str,
) -> dict[str, object]:
    """Restrict an exposed Candidate after a measured cohort-topology failure."""

    import openai

    source = _read_object(root / NATURAL_IMPUTATION_COLD_START_REPORT_PATH)
    target = _read_object(root / NATURAL_IMPUTATION_AIR_QUALITY_TARGET_REPORT_PATH)
    if source.get("verdict") != "NATURAL_COLD_START_CANDIDATE_PREMISE_PASS":
        raise ValueError("E2.80 requires the frozen Source imputation Candidate")
    if target.get("verdict") != "NATURAL_IMPUTATION_TARGET_PILOT_FAIL":
        raise ValueError("E2.80 requires the measured E2.79 Target failure")
    source_candidate = source.get("candidate")
    if not isinstance(source_candidate, Mapping):
        raise ValueError("the Source imputation Candidate is unavailable")

    workflow_supply = tuple(str(value) for value in source_candidate["workflow_supply"])
    source_episodes = []
    for dataset_row in source["datasets"]:
        workflows = {
            str(row["workflow_id"]): {
                "workflow_id": row["workflow_id"],
                "support_gain": float(row["support_gain_vs_identity"]),
                "query_gain": float(row["query_gain_vs_identity"]),
            }
            for row in dataset_row["programs"]
            if row["workflow_id"] != "IDENTITY"
        }
        source_episodes.append({"workflows": workflows})
    candidate = build_candidate_skill(
        [],
        source_episodes,
        capability_id=str(source_candidate["capability_id"]),
        task_context={
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "context": "observable natural missing values with known period",
        },
        workflow_supply=workflow_supply,
    )
    source_order = tuple(
        str(value) for value in candidate["source_prior"]["workflow_order"]
    )
    expected_order = (
        "W_period_median_imputation",
        "W_ar_imputation",
    )
    if source_order != expected_order:
        raise ValueError("the frozen Source Workflow order changed")

    target_roles = target.get("target_roles")
    if not isinstance(target_roles, list) or not target_roles:
        raise ValueError("E2.79 has no actionable Target role")
    target_role = target_roles[0]
    if not isinstance(target_role, Mapping):
        raise ValueError("invalid E2.79 Target role")
    response_by_workflow = {
        str(row["workflow_id"]): {
            "support_gain": float(row["support_gain_vs_identity"]),
            "query_gain": float(row["query_gain_vs_identity"]),
        }
        for row in target_role["programs"]
        if row["workflow_id"] != "IDENTITY"
    }
    selected_program = str(
        target_role["adaptation"]["A5_source_order_curve"][1]["selected_workflow"]
    )
    scope = {
        "cohort_topology": "multi_series_cross_sectional",
        "on_mismatch": "ABSTAIN",
    }
    failure_cases = [
        {
            "source_cohort_topology": "multi_series_cross_sectional",
            "target_cohort_topology": "single_series_temporal_origins",
            "support_to_query_replays": [
                {
                    "selected_program": selected_program,
                    "support_gains": {
                        workflow_id: response["support_gain"]
                        for workflow_id, response in response_by_workflow.items()
                    },
                    "query_gain": response_by_workflow[selected_program]["query_gain"],
                }
            ],
        }
    ]
    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("AGICTO_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
    llm_trace: dict[str, object] = {}

    def propose_patch(dossier: Mapping[str, object]) -> Mapping[str, object]:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You propose one bounded typed Harness patch from a categorical "
                        "Failure Dossier. Return exactly one JSON object with patch_id and "
                        "operations. Cover every first fault exactly once. Select operation, "
                        "target_surface, and value exactly from the Dossier catalog. Do not "
                        "modify Workflow, Consumer, Metric, Memory, or Task. Return no markdown."
                    ),
                },
                {"role": "user", "content": json.dumps(dossier, ensure_ascii=False)},
            ],
        )
        proposal = json.loads(completion.choices[0].message.content or "")
        if not isinstance(proposal, Mapping):
            raise ValueError("scope patch proposer must return one JSON object")
        llm_trace.update(
            {
                "base_url": base_url,
                "requested_model": model,
                "returned_model": getattr(completion, "model", ""),
                "prompt_tokens": getattr(
                    getattr(completion, "usage", None), "prompt_tokens", None
                ),
                "completion_tokens": getattr(
                    getattr(completion, "usage", None), "completion_tokens", None
                ),
                "proposal": proposal,
            }
        )
        return proposal

    prior_harm_count = int(target_role["adaptation"]["A5_harm_count"])
    source_like_responses = source_episodes[0]["workflows"]

    def replay_patch(patched: Mapping[str, object]) -> Mapping[str, object]:
        mismatch_probe_calls: list[str] = []
        mismatch = execute_skill_card(
            patched,
            lambda workflow_id: mismatch_probe_calls.append(workflow_id)
            or response_by_workflow[workflow_id],
            allow_candidate_replay=True,
            execution_context={"cohort_topology": "single_series_temporal_origins"},
        )
        source_like_probe_calls: list[str] = []
        source_like = execute_skill_card(
            patched,
            lambda workflow_id: source_like_probe_calls.append(workflow_id)
            or source_like_responses[workflow_id],
            allow_candidate_replay=True,
            execution_context={"cohort_topology": "multi_series_cross_sectional"},
        )
        repaired_harm_count = sum(
            float(row["fixed_query_gain"]) < 0.0
            for row in mismatch["adaptation_curve"][1:]
        )
        return {
            "mismatch_execution_context": {
                "cohort_topology": "single_series_temporal_origins"
            },
            "mismatch_probe_calls": mismatch_probe_calls,
            "mismatch_selected_workflow": mismatch["selected_workflow"],
            "mismatch_adaptation_curve": mismatch["adaptation_curve"],
            "source_like_execution_context": {
                "cohort_topology": "multi_series_cross_sectional"
            },
            "source_like_probe_calls": source_like_probe_calls,
            "source_like_selected_workflow": source_like["selected_workflow"],
            "source_like_adaptation_curve": source_like["adaptation_curve"],
            "prior_A5_harm_count": prior_harm_count,
            "repaired_harm_count": repaired_harm_count,
            "known_mismatch_abstains_before_probe": mismatch_probe_calls == [],
            "source_like_scope_still_executes": bool(
                source_like_probe_calls
                and source_like["selected_workflow"] != "IDENTITY"
            ),
        }

    def resolve_patch(
        patched: Mapping[str, object], replays: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object]:
        replay = replays[0]
        if not (
            replay["known_mismatch_abstains_before_probe"]
            and replay["source_like_scope_still_executes"]
            and int(replay["repaired_harm_count"]) == 0
        ):
            raise ValueError("scope repair replay did not realize the bounded behavior")
        return {
            "status": "RESTRICTED",
            "reason": "source-evidenced cohort topology only; mismatch abstains",
            "active_memory_write": False,
        }

    cycle = run_failure_driven_update_cycle(
        candidate,
        failure_cases,
        allowed_observations=["source_policy_episode_workflow_prior"],
        allowed_controls=["keep_best_support_so_far"],
        allowed_scopes=[scope],
        propose_patch=propose_patch,
        replay_patch=replay_patch,
        resolve_patch=resolve_patch,
    )
    patched = cycle["candidate_after_patch"]
    replay = cycle["replay"]
    resolved = cycle["resolved_skill"]
    behavior_nontrivial = bool(
        replay["known_mismatch_abstains_before_probe"]
        and replay["source_like_scope_still_executes"]
        and prior_harm_count > int(replay["repaired_harm_count"])
    )
    passed = bool(
        behavior_nontrivial
        and int(replay["repaired_harm_count"]) == 0
        and resolved["status"] == "RESTRICTED"
    )
    return {
        "experiment_id": "E2.80-live-LLM-natural-imputation-Scope-repair",
        "scientific_role": (
            "failure-driven restriction of a Candidate whose Source evidence omits "
            "the observed Target cohort topology"
        ),
        "source_candidate": {
            "status_before": candidate["status"],
            "workflow_order": list(source_order),
            "source_cohort_topology": "multi_series_cross_sectional",
        },
        "failure_dossier_sent_to_llm": cycle["failure_dossier"],
        "llm": llm_trace,
        "typed_patch": cycle["typed_patch"],
        "candidate_after_patch": {
            "capability_id": patched["capability_id"],
            "status": patched["status"],
            "applicability": patched["applicability"],
            "workflow_supply_unchanged": patched["workflow_supply"]
            == candidate["workflow_supply"],
        },
        "deterministic_replay": {**replay, "behavior_nontrivial": behavior_nontrivial},
        "resolved_candidate": resolved,
        "information_boundary": {
            "llm_received_dataset_identity": False,
            "llm_received_raw_time_series": False,
            "llm_received_effect_magnitude": False,
            "repair_truth_used": False,
            "raw_query_target_values_received_by_llm": False,
            "categorical_query_outcome_used_for_slow_path": True,
            "cached_query_effect_used_by_replay_evaluator": True,
        },
        "compute": {"consumer_fit_count": 0, "llm_api_call_count": 1},
        "gate": {
            "scope_patch_compiled": patched.get("applicability") == scope,
            "known_mismatch_abstains_before_probe": replay[
                "known_mismatch_abstains_before_probe"
            ],
            "source_like_scope_still_executes": replay[
                "source_like_scope_still_executes"
            ],
            "source_like_selects_nonidentity": replay[
                "source_like_selected_workflow"
            ]
            != "IDENTITY",
            "repaired_harm_zero": int(replay["repaired_harm_count"]) == 0,
            "workflow_supply_unchanged": patched["workflow_supply"]
            == candidate["workflow_supply"],
            "passed": passed,
        },
        "verdict": (
            "FAILURE_DRIVEN_SCOPE_REPAIR_PASS"
            if passed
            else "FAILURE_DRIVEN_SCOPE_REPAIR_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Exposed categorical mismatch replay. A PASS restricts this Candidate "
            "and removes known harm; it is not promotion or fresh Target transfer."
        ),
    }


def _run_natural_imputation_prsa_role(
    root: Path,
    *,
    channel: str,
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate one scoped imputation Candidate on one fixed PRSA role."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.operators.registry import get_operator

    values_by_station = {
        station: np.asarray(values, dtype=np.float64)
        for station, values in _load_prsa_channel(root, channel).items()
    }
    station_ids = sorted(values_by_station)
    train_stations = station_ids[:8]
    eval_stations = station_ids[8:]
    anchors = (240, 408, 576, 744, 912, 1080, 1248, 1416)
    period = 24
    historical_shifts = (2, 3, 4)

    raw_training_windows = []
    run_lengths: list[int] = []
    missing_points = 0
    total_points = 0
    for anchor in anchors:
        for station in train_stations:
            raw = values_by_station[station][
                anchor - CONTEXT_LENGTH : anchor + HORIZON
            ]
            if raw.shape != (CONTEXT_LENGTH + HORIZON,):
                continue
            if np.isfinite(raw).sum() < raw.size // 2:
                continue
            missing = ~np.isfinite(raw)
            missing_points += int(missing.sum())
            total_points += int(raw.size)
            run_lengths.extend(
                end - start for start, end in _missing_runs(missing.tolist())
            )
            raw_training_windows.append(raw)
    if len(raw_training_windows) < 32 or missing_points == 0:
        raise ValueError("NO_ACTIONABLE_NATURAL_MISSING_TRAINING_WINDOWS")

    # Retrospective availability-only admission; no Program outcome chooses stop.
    evaluation_stop = None
    latest_stop = min(len(values) for values in values_by_station.values()) - HORIZON
    for candidate_stop in range(2160, latest_stop + 1, period):
        admissible = True
        for station in eval_stations:
            values = values_by_station[station]
            origins = (candidate_stop,) + tuple(
                candidate_stop - shift * period for shift in historical_shifts
            )
            for origin in origins:
                context = values[origin - CONTEXT_LENGTH : origin]
                target = values[origin : origin + HORIZON]
                if (
                    context.shape != (CONTEXT_LENGTH,)
                    or target.shape != (HORIZON,)
                    or np.isfinite(context).sum() < CONTEXT_LENGTH // 2
                    or not np.isfinite(target).all()
                ):
                    admissible = False
                    break
            if not admissible:
                break
        if admissible:
            evaluation_stop = candidate_stop
            break
    if evaluation_stop is None:
        raise ValueError("NO_ACTIONABLE_EVALUATION_ORIGIN")

    operator_by_workflow = {
        "IDENTITY": get_operator("impute_linear"),
        "W_period_median_imputation": get_operator("period_median_complete"),
        "W_ar_imputation": get_operator("impute_ar"),
    }
    workflow_supply = tuple(str(value) for value in candidate["workflow_supply"])
    expected_supply = (
        "W_period_median_imputation",
        "W_ar_imputation",
    )
    if workflow_supply != expected_supply:
        raise ValueError("the frozen imputation Workflow supply changed")

    program_rows: list[dict[str, object]] = []
    for workflow_id in ("IDENTITY", *workflow_supply):
        operator = operator_by_workflow[workflow_id]
        x_train = []
        y_train = []
        for raw in raw_training_windows:
            filled = (
                operator(raw)
                if workflow_id == "IDENTITY"
                else operator(raw, period=period)
            )
            if not np.isfinite(filled).all():
                continue
            context = filled[:CONTEXT_LENGTH]
            target_values = filled[CONTEXT_LENGTH:]
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                continue
            x_train.append((context - center) / scale)
            y_train.append((target_values - center) / scale)
        if len(x_train) < 32:
            raise ValueError(f"INSUFFICIENT_IMPUTED_TRAINING_ROWS:{workflow_id}")

        x_eval = []
        actual = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal_scales: list[float] = []
        roles: list[str] = []
        origins_seen: list[int] = []
        for station in eval_stations:
            values = values_by_station[station]
            observed_mask = np.isfinite(values)
            origins = [
                *(('support', evaluation_stop - shift * period) for shift in historical_shifts),
                ('query', evaluation_stop),
            ]
            for role, origin in origins:
                raw_context = values[origin - CONTEXT_LENGTH : origin]
                truth = values[origin : origin + HORIZON]
                context = (
                    operator(raw_context)
                    if workflow_id == "IDENTITY"
                    else operator(raw_context, period=period)
                )
                if not np.isfinite(context).all() or not np.isfinite(truth).all():
                    raise ValueError(f"NON_FINITE_EVALUATION_ROLE:{role}")
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise ValueError(f"EVALUATION_SCALE_FLOOR:{role}")
                try:
                    scale_value = seasonal_scale(
                        values[:origin],
                        observed_mask[:origin],
                        period=period,
                        min_pairs=32,
                    )
                except (UndefinedSeasonalScale, ValueError) as exc:
                    raise ValueError(f"UNDEFINED_SEASONAL_SCALE:{role}") from exc
                x_eval.append((context - center) / scale)
                actual.append(truth)
                centers.append(center)
                scales.append(scale)
                seasonal_scales.append(scale_value)
                roles.append(role)
                origins_seen.append(origin)

        prediction = _exact_weighted_ridge_prediction(
            np,
            x_train=np.asarray(x_train, dtype=np.float64),
            targets=np.asarray(y_train, dtype=np.float64),
            weights=np.ones(len(x_train), dtype=np.float64),
            x_eval=np.asarray(x_eval, dtype=np.float64),
        )
        losses = []
        for index, truth in enumerate(actual):
            raw_prediction = prediction[index] * scales[index] + centers[index]
            losses.append(smase(truth, raw_prediction, scale=seasonal_scales[index]))
        support_losses = [loss for loss, role in zip(losses, roles) if role == "support"]
        query_losses = [loss for loss, role in zip(losses, roles) if role == "query"]
        program_rows.append(
            {
                "workflow_id": workflow_id,
                "support_loss": statistics.fmean(support_losses),
                "query_loss": statistics.fmean(query_losses),
                "combined_loss": statistics.fmean(losses),
                "training_row_count": len(x_train),
            }
        )

    identity = next(row for row in program_rows if row["workflow_id"] == "IDENTITY")
    for row in program_rows:
        row["support_gain_vs_identity"] = (
            identity["support_loss"] - row["support_loss"]
        )
        row["query_gain_vs_identity"] = identity["query_loss"] - row["query_loss"]
        row["combined_gain_vs_identity"] = (
            identity["combined_loss"] - row["combined_loss"]
        )
    responses = {
        str(row["workflow_id"]): {
            "workflow_id": row["workflow_id"],
            "support_gain": float(row["support_gain_vs_identity"]),
            "query_gain": float(row["query_gain_vs_identity"]),
        }
        for row in program_rows
        if row["workflow_id"] != "IDENTITY"
    }
    a5_probe_calls: list[str] = []
    a5 = execute_skill_card(
        candidate,
        lambda workflow_id: a5_probe_calls.append(workflow_id)
        or responses[workflow_id],
        allow_candidate_replay=True,
        execution_context={"cohort_topology": "multi_series_cross_sectional"},
    )
    target_only_curves = [
        workflow_curve_from_policy_episode(responses, order)
        for order in permutations(workflow_supply)
    ]
    a3_curve = [
        {
            "budget": budget,
            "fixed_query_gain": statistics.fmean(
                float(curve[budget]["fixed_query_gain"])
                for curve in target_only_curves
            ),
        }
        for budget in range(len(workflow_supply) + 1)
    ]
    a3_auc = policy_episode_adapt_auc(a3_curve)
    a5_auc = float(a5["adaptation_auc"])
    a5_harm_count = sum(
        float(row["fixed_query_gain"]) < 0.0 for row in a5["adaptation_curve"][1:]
    )
    menu_oracle = max(
        [("IDENTITY", 0.0)]
        + [
            (str(row["workflow_id"]), float(row["query_gain_vs_identity"]))
            for row in program_rows
            if row["workflow_id"] != "IDENTITY"
        ],
        key=lambda item: (item[1], item[0]),
    )
    behavior_differs = any(
        abs(
            float(a5["adaptation_curve"][index]["fixed_query_gain"])
            - float(a3_curve[index]["fixed_query_gain"])
        )
        > 1e-12
        for index in range(1, len(a3_curve))
    )
    return {
        "channel": channel,
        "roster": {
            "train_stations": train_stations,
            "eval_stations": eval_stations,
        },
        "public_context": {
            "cohort_topology": "multi_series_cross_sectional",
            "natural_missing_values_present": True,
            "missing_fraction": missing_points / total_points,
            "maximum_run_length": max(run_lengths) if run_lengths else 0,
            "period": period,
            "training_window_count": len(raw_training_windows),
        },
        "geometry": {
            "training_anchors": list(anchors),
            "evaluation_stop": evaluation_stop,
            "historical_origins": sorted(
                set(origin for role, origin in zip(roles, origins_seen) if role == "support")
            ),
            "support_feedback": "historical_observed_forecasting_outcome",
            "query_outcome_role": "EVALUATOR_ONLY",
            "repair_truth_used": False,
            "query_target_values_used_for_planning_or_confirmation": False,
            "query_target_availability_used_for_retrospective_admission": True,
        },
        "programs": program_rows,
        "adaptation": {
            "A5_scoped_candidate_curve": a5["adaptation_curve"],
            "A5_probe_calls": a5_probe_calls,
            "A5_applicability_matched": a5.get("applicability_matched"),
            "A5_adaptation_auc": a5_auc,
            "A3_equal_budget_order_average_curve": a3_curve,
            "A3_adaptation_auc": a3_auc,
            "A5_minus_A3": a5_auc - a3_auc,
            "A5_harm_count": a5_harm_count,
        },
        "validation": {
            "material_query_headroom": float(menu_oracle[1]) > 0.005,
            "menu_oracle_workflow_evaluator_only": menu_oracle[0],
            "menu_oracle_query_gain_evaluator_only": menu_oracle[1],
            "source_order_changes_early_behavior": behavior_differs,
            "A5_above_equal_budget_A3": a5_auc > a3_auc,
            "A5_harm_zero": a5_harm_count == 0,
            "scope_matched_and_executed": bool(
                a5.get("applicability_matched") is True and a5_probe_calls
            ),
        },
    }


def run_natural_imputation_prsa_target_pilot(
    root: Path,
    *,
    channels: tuple[str, ...] = ("TEMP", "WSPM"),
    experiment_id: str = "E2.81-natural-imputation-PRSA-Target-pilot",
) -> dict[str, object]:
    """Test the restricted Candidate on exposed topology-matched PRSA roles."""

    source = _read_object(root / NATURAL_IMPUTATION_COLD_START_REPORT_PATH)
    repair = _read_object(root / NATURAL_IMPUTATION_SCOPE_REPAIR_REPORT_PATH)
    if source.get("verdict") != "NATURAL_COLD_START_CANDIDATE_PREMISE_PASS":
        raise ValueError("E2.81 requires the frozen Source imputation Candidate")
    if repair.get("verdict") != "FAILURE_DRIVEN_SCOPE_REPAIR_PASS":
        raise ValueError("E2.81 requires the E2.80 typed Scope repair")
    source_candidate = source.get("candidate")
    if not isinstance(source_candidate, Mapping):
        raise ValueError("the Source imputation Candidate is unavailable")
    source_episodes = []
    for dataset_row in source["datasets"]:
        workflows = {
            str(row["workflow_id"]): {
                "workflow_id": row["workflow_id"],
                "support_gain": float(row["support_gain_vs_identity"]),
                "query_gain": float(row["query_gain_vs_identity"]),
            }
            for row in dataset_row["programs"]
            if row["workflow_id"] != "IDENTITY"
        }
        source_episodes.append({"workflows": workflows})
    candidate = build_candidate_skill(
        [],
        source_episodes,
        capability_id=str(source_candidate["capability_id"]),
        task_context={
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "context": "observable natural missing values with known period",
        },
        workflow_supply=tuple(
            str(value) for value in source_candidate["workflow_supply"]
        ),
    )
    patched = apply_typed_patch(candidate, repair["typed_patch"])

    role_reports = []
    feasibility_rejections = []
    for channel in channels:
        try:
            role_reports.append(
                _run_natural_imputation_prsa_role(
                    root,
                    channel=channel,
                    candidate=patched,
                )
            )
        except ValueError as exc:
            feasibility_rejections.append({"channel": channel, "reason": str(exc)})
    if not role_reports:
        return {
            "experiment_id": experiment_id,
            "scientific_role": "topology-matched exposed Target mechanism pilot",
            "dataset": "UCI Beijing Multi-Site Air Quality",
            "predeclared_channel_roles": list(channels),
            "roster_rule": "sorted stations: first 8 train, last 4 evaluate",
            "target_roles": [],
            "feasibility_rejections": feasibility_rejections,
            "context_exposure": "INSTANCE_SEEN_IN_OTHER_PROGRAM_FAMILIES",
            "outcome_exposure": "SEALED",
            "information_boundary": {
                "repair_truth_used": False,
                "program_outcome_evaluated": False,
                "query_target_values_used_for_planning_or_confirmation": False,
                "query_target_availability_used_for_retrospective_admission": False,
            },
            "compute": {"consumer_fit_count": 0, "llm_api_call_count": 0},
            "verdict": "NATURAL_IMPUTATION_PRSA_TARGET_PILOT_INCONCLUSIVE",
            "capability_or_memory_written": False,
            "claim_limit": (
                "The fixed roles and anchors contain no actionable natural missingness; "
                "this is a carrier-feasibility result, not evidence against the Candidate."
            ),
        }

    a5_macro = statistics.fmean(
        float(row["adaptation"]["A5_adaptation_auc"]) for row in role_reports
    )
    a3_macro = statistics.fmean(
        float(row["adaptation"]["A3_adaptation_auc"]) for row in role_reports
    )
    material_roles = sum(
        bool(row["validation"]["material_query_headroom"]) for row in role_reports
    )
    behavior_roles = sum(
        bool(row["validation"]["source_order_changes_early_behavior"])
        for row in role_reports
    )
    harmful_roles = sum(
        int(row["adaptation"]["A5_harm_count"]) > 0 for row in role_reports
    )
    scope_execution_roles = sum(
        bool(row["validation"]["scope_matched_and_executed"]) for row in role_reports
    )
    passed = bool(
        material_roles >= 1
        and behavior_roles >= 1
        and scope_execution_roles == len(role_reports)
        and a5_macro > a3_macro
        and harmful_roles == 0
    )
    return {
        "experiment_id": experiment_id,
        "scientific_role": (
            "exposed matched-topology mechanism pilot of the E2.80 restricted Candidate"
        ),
        "dataset": "UCI Beijing Multi-Site Air Quality",
        "predeclared_channel_roles": list(channels),
        "source": {
            "url": "https://archive.ics.uci.edu/dataset/501/beijing",
            "license": "CC BY 4.0",
            "missing_semantics": "NA in the official station CSVs",
        },
        "candidate": {
            "status": patched["status"],
            "applicability": patched["applicability"],
            "workflow_order": patched["source_prior"]["workflow_order"],
            "scope_patch_report": NATURAL_IMPUTATION_SCOPE_REPAIR_REPORT_PATH,
        },
        "target_roles": role_reports,
        "feasibility_rejections": feasibility_rejections,
        "context_exposure": "INSTANCE_SEEN_IN_OTHER_PROGRAM_FAMILIES",
        "outcome_exposure": "EXPOSED_BY_THIS_PILOT",
        "summary": {
            "role_count": len(role_reports),
            "material_headroom_role_count": material_roles,
            "source_order_behavior_change_role_count": behavior_roles,
            "scope_matched_execution_role_count": scope_execution_roles,
            "A5_harmful_role_count": harmful_roles,
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
        },
        "information_boundary": {
            "repair_truth_used": False,
            "support_feedback": "historical_observed_forecasting_outcome",
            "query_outcome_role": "EVALUATOR_ONLY",
            "query_target_values_used_for_planning_or_confirmation": False,
            "query_target_availability_used_for_retrospective_admission": True,
            "target_context_or_outcome_fresh_for_project": False,
        },
        "compute": {
            "consumer_fit_count": 3 * len(role_reports),
            "llm_api_call_count": 0,
        },
        "gate": {
            "at_least_one_role_has_material_headroom": material_roles >= 1,
            "source_order_changes_behavior": behavior_roles >= 1,
            "scope_matches_and_executes_all_actionable_roles": scope_execution_roles
            == len(role_reports),
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_harmful_role_count_zero": harmful_roles == 0,
            "passed": passed,
        },
        "verdict": (
            "NATURAL_IMPUTATION_PRSA_TARGET_PILOT_PASS"
            if passed
            else "NATURAL_IMPUTATION_PRSA_TARGET_PILOT_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "This PRSA dataset was exposed in another Program family. A PASS is a "
            "matched-topology mechanism pilot, not fresh confirmation or Promotion."
        ),
    }


def _pseudo_gap_program_response_observation(
    raw_windows: Sequence[Any],
    *,
    period: int,
) -> dict[str, object]:
    """Compare two imputers on originally observed pseudo-gaps only."""

    import numpy as np

    from SelfEvolvingHarnessTS.operators.registry import get_operator

    contexts = [
        np.asarray(raw, dtype=np.float64)[:CONTEXT_LENGTH] for raw in raw_windows
    ]
    run_lengths = [
        end - start
        for context in contexts
        for start, end in _missing_runs((~np.isfinite(context)).tolist())
    ]
    if not run_lengths:
        return {
            "preferred_workflow": None,
            "error_margin": None,
            "reliability": "UNRESOLVED",
            "probe_count": 0,
        }
    probe_length = min(max(1, int(round(statistics.median(run_lengths)))), period)
    starts = (4 * period, 5 * period, 6 * period)
    errors = {
        "W_period_median_imputation": [],
        "W_ar_imputation": [],
    }
    operators = {
        "W_period_median_imputation": get_operator("period_median_complete"),
        "W_ar_imputation": get_operator("impute_ar"),
    }
    probe_count = 0
    for context in contexts:
        _, scale, method = _center_scale(np, context)
        if method == "scale_floor_fallback":
            continue
        for start in starts:
            stop = start + probe_length
            if stop > CONTEXT_LENGTH or not np.isfinite(context[start:stop]).all():
                continue
            truth = context[start:stop].copy()
            masked = context.copy()
            masked[start:stop] = np.nan
            reconstructed = {
                workflow_id: operator(masked, period=period)
                for workflow_id, operator in operators.items()
            }
            if any(
                not np.isfinite(values[start:stop]).all()
                for values in reconstructed.values()
            ):
                continue
            for workflow_id, values in reconstructed.items():
                errors[workflow_id].append(
                    float(np.mean(np.abs(values[start:stop] - truth)) / scale)
                )
            probe_count += 1
    if probe_count < 12:
        return {
            "preferred_workflow": None,
            "error_margin": None,
            "reliability": "UNRESOLVED",
            "probe_count": probe_count,
        }
    median_errors = {
        workflow_id: float(np.median(values))
        for workflow_id, values in errors.items()
    }
    ordered = sorted(median_errors, key=lambda key: (median_errors[key], key))
    return {
        "preferred_workflow": ordered[0],
        "error_margin": median_errors[ordered[1]] - median_errors[ordered[0]],
        "reliability": "RELIABLE",
        "probe_count": probe_count,
    }


def _natural_imputation_pseudo_gap_cases(
    root: Path,
    *,
    prsa_channels: tuple[str, ...] = ("CO", "NO2"),
) -> list[dict[str, object]]:
    """Rebuild only the frozen Source and PRSA training-window rosters."""

    import numpy as np

    specs = {
        "monash:nn5_daily": {
            "anchors": (240, 300, 360, 420),
            "evaluation_stop": 528,
            "period": 7,
        },
        "gefcom2012_load": {
            "anchors": (312, 372, 432, 492, 552, 612, 672, 732, 792, 852),
            "evaluation_stop": 912,
            "period": 24,
        },
        "noaa_global_hourly": {
            "anchors": (240, 300, 360, 420, 480, 540, 600, 660),
            "evaluation_stop": 720,
            "period": 24,
        },
    }
    registry = [
        json.loads(line)
        for line in (
            root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record_dirs = {}
    for record_path in (root / "data/benchmark_v0_2/clean_base").glob(
        "*/record.json"
    ):
        record = _read_object(record_path)
        record_dirs[str(record["series_uid"])] = record_path.parent

    cases: list[dict[str, object]] = []
    for dataset_id, spec in specs.items():
        candidates = sorted(
            (
                row
                for row in registry
                if row["dataset_id"] == dataset_id
                and int(row["length"]) >= int(spec["evaluation_stop"]) + HORIZON
            ),
            key=lambda row: str(row["series_uid"]),
        )
        train_rows = candidates[:12]
        raw_windows = []
        for anchor in spec["anchors"]:
            for row in train_rows:
                raw = np.load(
                    record_dirs[str(row["series_uid"])] / "values.npy"
                ).astype(np.float64)[
                    int(anchor) - CONTEXT_LENGTH : int(anchor) + HORIZON
                ]
                if (
                    raw.shape == (CONTEXT_LENGTH + HORIZON,)
                    and np.isfinite(raw).sum() >= raw.size // 2
                ):
                    raw_windows.append(raw)
        cases.append(
            {
                "case_id": dataset_id,
                "cohort_topology": "multi_series_cross_sectional",
                "period": int(spec["period"]),
                "raw_windows": raw_windows,
            }
        )

    anchors = (240, 408, 576, 744, 912, 1080, 1248, 1416)
    for channel in prsa_channels:
        values_by_station = {
            station: np.asarray(values, dtype=np.float64)
            for station, values in _load_prsa_channel(root, channel).items()
        }
        train_stations = sorted(values_by_station)[:8]
        raw_windows = []
        for anchor in anchors:
            for station in train_stations:
                raw = values_by_station[station][
                    anchor - CONTEXT_LENGTH : anchor + HORIZON
                ]
                if (
                    raw.shape == (CONTEXT_LENGTH + HORIZON,)
                    and np.isfinite(raw).sum() >= raw.size // 2
                ):
                    raw_windows.append(raw)
        cases.append(
            {
                "case_id": f"prsa:{channel}",
                "cohort_topology": "multi_series_cross_sectional",
                "period": 24,
                "raw_windows": raw_windows,
            }
        )
    return cases


def run_natural_imputation_pseudo_gap_heldout_role_confirmation(
    root: Path,
) -> dict[str, object]:
    """Replay the frozen pseudo-gap ordering on mixed-exposure PRSA roles."""

    diagnostic = _read_object(root / NATURAL_IMPUTATION_PSEUDO_GAP_REPORT_PATH)
    if diagnostic.get("verdict") != "PSEUDO_GAP_OBSERVATION_PATCH_PASS":
        raise ValueError("E2.84 requires the corrected E2.83 context-only PASS")
    boundary = diagnostic.get("information_boundary", {})
    if boundary.get("post_context_horizon_values_received_by_observation") is not False:
        raise ValueError("E2.84 requires a context-only pseudo-gap Observation")

    channels = ("O3", "PM10")
    observations = []
    for case in _natural_imputation_pseudo_gap_cases(
        root,
        prsa_channels=channels,
    ):
        if not str(case["case_id"]).startswith("prsa:"):
            continue
        observations.append(
            {
                "case_id": case["case_id"],
                "cohort_topology": case["cohort_topology"],
                "observation": _pseudo_gap_program_response_observation(
                    case["raw_windows"],
                    period=int(case["period"]),
                ),
            }
        )
    observation_by_channel = {
        str(row["case_id"]).split(":", 1)[1]: row["observation"]
        for row in observations
    }

    # PM10 is opened here; the identical O3 evaluation surface was already
    # exposed by the earlier missing-window-weighting family.  Keep both fixed
    # roles for development falsification, but do not call this fresh evidence.
    target = run_natural_imputation_prsa_target_pilot(
        root,
        channels=channels,
        experiment_id=(
            "E2.84-natural-imputation-pseudo-gap-heldout-role-confirmation"
        ),
    )
    source_order = tuple(
        str(value) for value in target.get("candidate", {}).get("workflow_order", [])
    )
    complete = bool(
        len(target.get("target_roles", [])) == len(channels)
        and not target.get("feasibility_rejections")
        and set(observation_by_channel) == set(channels)
        and source_order
    )
    if not complete:
        return {
            "experiment_id": (
                "E2.84-natural-imputation-pseudo-gap-heldout-role-confirmation"
            ),
            "scientific_role": (
                "mixed-exposure same-dataset development replay of one frozen Observation"
            ),
            "predeclared_channel_roles": list(channels),
            "case_observations": observations,
            "target_pilot": target,
            "context_exposure": "MIXED_O3_INSTANCE_SEEN_PM10_AGGREGATE_SEEN",
            "outcome_exposure": "O3_EXPOSED_PRE_E2_84_PM10_OPENED_E2_84",
            "information_boundary": {
                "pseudo_gap_truth_source": (
                    "originally_observed_historical_context_points"
                ),
                "natural_gap_truth_used": False,
                "post_context_horizon_values_received_by_observation": False,
                "repair_truth_used": False,
                "query_target_values_used_for_planning_or_confirmation": False,
            },
            "compute": target.get(
                "compute", {"consumer_fit_count": 0, "llm_api_call_count": 0}
            ),
            "verdict": "PSEUDO_GAP_HELDOUT_ROLE_CONFIRMATION_INCONCLUSIVE",
            "capability_or_memory_written": False,
            "claim_limit": (
                "The two frozen development roles were not both executable; no "
                "replacement role was selected and no fresh method conclusion is drawn."
            ),
        }

    role_reports = []
    for role in target["target_roles"]:
        channel = str(role["channel"])
        observation = observation_by_channel[channel]
        preferred = observation["preferred_workflow"]
        pseudo_order = (
            (str(preferred),)
            + tuple(value for value in source_order if value != preferred)
            if preferred in source_order
            else source_order
        )
        responses = {
            str(row["workflow_id"]): {
                "workflow_id": row["workflow_id"],
                "support_gain": float(row["support_gain_vs_identity"]),
                "query_gain": float(row["query_gain_vs_identity"]),
            }
            for row in role["programs"]
            if row["workflow_id"] != "IDENTITY"
        }
        new_curve = workflow_curve_from_policy_episode(responses, pseudo_order)
        old_curve = role["adaptation"]["A5_scoped_candidate_curve"]
        role_reports.append(
            {
                "channel": channel,
                "observation": observation,
                "frozen_source_order": list(source_order),
                "pseudo_gap_probe_order": list(pseudo_order),
                "programs": role["programs"],
                "old_A5_curve": old_curve,
                "new_A5_curve": new_curve,
                "A3_curve": role["adaptation"][
                    "A3_equal_budget_order_average_curve"
                ],
                "old_A5_adaptation_auc": float(
                    role["adaptation"]["A5_adaptation_auc"]
                ),
                "new_A5_adaptation_auc": policy_episode_adapt_auc(new_curve),
                "A3_adaptation_auc": float(
                    role["adaptation"]["A3_adaptation_auc"]
                ),
                "new_A5_harm_count": sum(
                    float(point["fixed_query_gain"]) < 0.0
                    for point in new_curve[1:]
                ),
                "behavior_changed": any(
                    str(new["selected_workflow"])
                    != str(old["selected_workflow"])
                    for new, old in zip(new_curve[1:], old_curve[1:])
                ),
            }
        )

    new_a5_macro = statistics.fmean(
        float(row["new_A5_adaptation_auc"]) for row in role_reports
    )
    old_a5_macro = statistics.fmean(
        float(row["old_A5_adaptation_auc"]) for row in role_reports
    )
    a3_macro = statistics.fmean(
        float(row["A3_adaptation_auc"]) for row in role_reports
    )
    reliable = all(
        row["observation"]["reliability"] == "RELIABLE"
        for row in role_reports
    )
    harm_count = sum(int(row["new_A5_harm_count"]) for row in role_reports)
    behavior_nontrivial = any(bool(row["behavior_changed"]) for row in role_reports)
    passed = bool(
        reliable
        and new_a5_macro > old_a5_macro
        and new_a5_macro > a3_macro
        and harm_count == 0
        and behavior_nontrivial
    )
    return {
        "experiment_id": (
            "E2.84-natural-imputation-pseudo-gap-heldout-role-confirmation"
        ),
        "scientific_role": (
            "mixed-exposure same-dataset development replay of one frozen Observation"
        ),
        "dataset": "UCI Beijing Multi-Site Air Quality",
        "predeclared_channel_roles": list(channels),
        "frozen_from": NATURAL_IMPUTATION_PSEUDO_GAP_REPORT_PATH,
        "case_observations": observations,
        "target_roles": role_reports,
        "summary": {
            "adapt_auc": {
                "old_A5": old_a5_macro,
                "new_A5": new_a5_macro,
                "A3": a3_macro,
            },
            "new_A5_minus_old_A5": new_a5_macro - old_a5_macro,
            "new_A5_minus_A3": new_a5_macro - a3_macro,
            "new_A5_harm_count": harm_count,
            "behavior_nontrivial": behavior_nontrivial,
        },
        "gate": {
            "both_observations_reliable": reliable,
            "new_A5_above_old_A5": new_a5_macro > old_a5_macro,
            "new_A5_above_A3": new_a5_macro > a3_macro,
            "new_A5_harm_zero": harm_count == 0,
            "behavior_nontrivial": behavior_nontrivial,
            "passed": passed,
        },
        "context_exposure": "MIXED_O3_INSTANCE_SEEN_PM10_AGGREGATE_SEEN",
        "outcome_exposure": "O3_EXPOSED_PRE_E2_84_PM10_OPENED_E2_84",
        "information_boundary": {
            "pseudo_gap_truth_source": (
                "originally_observed_historical_context_points"
            ),
            "natural_gap_truth_used": False,
            "post_context_horizon_values_received_by_observation": False,
            "repair_truth_used": False,
            "support_feedback": "historical_observed_forecasting_outcome",
            "query_outcome_role": "EVALUATOR_ONLY",
            "query_target_values_used_for_planning_or_confirmation": False,
            "query_target_availability_used_for_retrospective_admission": True,
        },
        "compute": {
            "consumer_fit_count": 3 * len(role_reports),
            "llm_api_call_count": 0,
        },
        "verdict": (
            "PSEUDO_GAP_DEVELOPMENT_REPLAY_PASS"
            if passed
            else "PSEUDO_GAP_DEVELOPMENT_REPLAY_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Cross-family outcome-exposed development replay only; O3 was already "
            "evaluated on the same surface and PM10 was first opened here. This is "
            "not fresh confirmation, Promotion or an Active Memory write."
        ),
    }


def run_live_natural_imputation_pseudo_gap_observation(
    root: Path,
    *,
    model: str,
    base_url: str,
    allow_llm_patch: bool = True,
) -> dict[str, object]:
    """Diagnose, and only then patch, imputation probe ordering."""

    source = _read_object(root / NATURAL_IMPUTATION_COLD_START_REPORT_PATH)
    repair = _read_object(root / NATURAL_IMPUTATION_SCOPE_REPAIR_REPORT_PATH)
    target = _read_object(root / NATURAL_IMPUTATION_PRSA_ACTIONABLE_TARGET_REPORT_PATH)
    if source.get("verdict") != "NATURAL_COLD_START_CANDIDATE_PREMISE_PASS":
        raise ValueError("E2.83 requires the frozen Source imputation Candidate")
    if repair.get("verdict") != "FAILURE_DRIVEN_SCOPE_REPAIR_PASS":
        raise ValueError("E2.83 requires the E2.80 Scope repair")
    if target.get("verdict") != "NATURAL_IMPUTATION_PRSA_TARGET_PILOT_FAIL":
        raise ValueError("E2.83 requires the measured E2.82 probe-order failure")

    observations = []
    for case in _natural_imputation_pseudo_gap_cases(root):
        observation = _pseudo_gap_program_response_observation(
            case["raw_windows"],
            period=int(case["period"]),
        )
        observations.append(
            {
                "case_id": case["case_id"],
                "cohort_topology": case["cohort_topology"],
                "observation": observation,
            }
        )
    observation_by_case = {
        str(row["case_id"]): row["observation"] for row in observations
    }
    source_order = tuple(
        str(value) for value in source["candidate"]["source_prior"]["workflow_order"]
    )
    role_diagnostics = []
    for role in target["target_roles"]:
        channel = str(role["channel"])
        observation = observation_by_case[f"prsa:{channel}"]
        preferred = observation["preferred_workflow"]
        if preferred in source_order:
            pseudo_order = (str(preferred),) + tuple(
                workflow_id
                for workflow_id in source_order
                if workflow_id != preferred
            )
        else:
            pseudo_order = source_order
        responses = {
            str(row["workflow_id"]): {
                "workflow_id": row["workflow_id"],
                "support_gain": float(row["support_gain_vs_identity"]),
                "query_gain": float(row["query_gain_vs_identity"]),
            }
            for row in role["programs"]
            if row["workflow_id"] != "IDENTITY"
        }
        new_curve = workflow_curve_from_policy_episode(responses, pseudo_order)
        old_curve = role["adaptation"]["A5_scoped_candidate_curve"]
        role_diagnostics.append(
            {
                "channel": channel,
                "observation": observation,
                "pseudo_gap_probe_order": list(pseudo_order),
                "new_A5_curve": new_curve,
                "new_A5_adaptation_auc": policy_episode_adapt_auc(new_curve),
                "old_A5_adaptation_auc": float(
                    role["adaptation"]["A5_adaptation_auc"]
                ),
                "A3_adaptation_auc": float(
                    role["adaptation"]["A3_adaptation_auc"]
                ),
                "new_A5_harm_count": sum(
                    float(row["fixed_query_gain"]) < 0.0 for row in new_curve[1:]
                ),
                "behavior_changed": any(
                    str(new["selected_workflow"])
                    != str(old["selected_workflow"])
                    for new, old in zip(new_curve[1:], old_curve[1:])
                ),
            }
        )
    all_target_roles_reliable = bool(
        len(role_diagnostics) == 2
        and all(
            row["observation"]["reliability"] == "RELIABLE"
            for row in role_diagnostics
        )
    )
    new_a5_macro = statistics.fmean(
        float(row["new_A5_adaptation_auc"]) for row in role_diagnostics
    )
    old_a5_macro = float(target["summary"]["adapt_auc"]["A5"])
    a3_macro = float(target["summary"]["adapt_auc"]["A3"])
    harm_count = sum(int(row["new_A5_harm_count"]) for row in role_diagnostics)
    behavior_nontrivial = any(bool(row["behavior_changed"]) for row in role_diagnostics)
    diagnostic_passed = bool(
        all_target_roles_reliable
        and new_a5_macro > a3_macro
        and new_a5_macro > old_a5_macro
        and harm_count == 0
        and behavior_nontrivial
    )
    base_report: dict[str, object] = {
        "experiment_id": "E2.83-natural-imputation-pseudo-gap-Observation",
        "scientific_role": (
            "outcome-free Program-response Observation diagnostic followed by a "
            "conditional bounded typed patch"
        ),
        "algorithm": {
            "probe_length": "min(max(1, round(median real missing-run length)), period)",
            "candidate_starts_in_context": ["4*period", "5*period", "6*period"],
            "minimum_probe_count": 12,
            "error": "pseudo-gap MAE divided by observed-window robust scale",
            "natural_missing_values_outside_pseudo_gap_preserved": True,
            "operator_input_scope": "context_only_first_192_steps",
        },
        "case_observations": observations,
        "target_diagnostic": role_diagnostics,
        "summary": {
            "all_target_roles_reliable": all_target_roles_reliable,
            "adapt_auc": {
                "old_A5": old_a5_macro,
                "new_A5": new_a5_macro,
                "A3": a3_macro,
            },
            "new_A5_minus_old_A5": new_a5_macro - old_a5_macro,
            "new_A5_minus_A3": new_a5_macro - a3_macro,
            "new_A5_harm_count": harm_count,
            "behavior_nontrivial": behavior_nontrivial,
        },
        "diagnostic_gate": {
            "both_target_roles_reliable": all_target_roles_reliable,
            "new_A5_above_A3": new_a5_macro > a3_macro,
            "new_A5_above_old_A5": new_a5_macro > old_a5_macro,
            "new_A5_harm_zero": harm_count == 0,
            "behavior_nontrivial": behavior_nontrivial,
            "passed": diagnostic_passed,
        },
        "context_exposure": "INSTANCE_SEEN_IN_OTHER_PROGRAM_FAMILIES",
        "outcome_exposure": "EXPOSED_CACHED_E2_82",
        "information_boundary": {
            "pseudo_gap_truth_source": "originally_observed_historical_context_points",
            "natural_gap_truth_used": False,
            "post_context_horizon_values_received_by_observation": False,
            "repair_truth_used": False,
            "cached_query_effect_used_by_replay_evaluator": True,
        },
        "capability_or_memory_written": False,
    }
    if not diagnostic_passed:
        base_report.update(
            {
                "llm": None,
                "typed_patch": None,
                "resolved_candidate_status": "CANDIDATE",
                "compute": {"consumer_fit_count": 0, "llm_api_call_count": 0},
                "verdict": "PSEUDO_GAP_OBSERVATION_REJECTED",
                "claim_limit": (
                    "The outcome-free Observation did not improve the cached Target "
                    "adaptation estimand; no LLM patch was requested."
                ),
            }
        )
        return base_report
    if not allow_llm_patch:
        base_report.update(
            {
                "llm": None,
                "typed_patch": None,
                "resolved_candidate_status": "CANDIDATE",
                "compute": {"consumer_fit_count": 0, "llm_api_call_count": 0},
                "verdict": "PSEUDO_GAP_DIAGNOSTIC_PASS_PATCH_DEFERRED",
                "claim_limit": (
                    "Diagnostic-only execution; no LLM patch was requested and no "
                    "Candidate behavior was modified."
                ),
            }
        )
        return base_report

    dossier = {
        "dossier_id": "imputation_program_response_observation_v1",
        "categorical_first_faults": [
            {
                "surface": "observation",
                "code": "POOLED_SOURCE_PRIOR_IGNORES_PROGRAM_RESPONSE",
                "observed_behavior": (
                    "a pooled Source probe order delays the locally preferred "
                    "imputation Workflow"
                ),
            }
        ],
        "allowed_patch_values": {
            "ADD_OBSERVATION": ["pseudo_gap_program_response_order"]
        },
        "forbidden_changes": [
            "workflow_supply",
            "consumer",
            "metric",
            "memory",
            "applicability",
        ],
        "privacy": {
            "raw_time_series_included": False,
            "dataset_identity_included": False,
            "effect_magnitudes_included": False,
        },
    }
    existing_report_path = root / NATURAL_IMPUTATION_PSEUDO_GAP_REPORT_PATH
    reused_existing_patch = existing_report_path.exists()
    if reused_existing_patch:
        existing_report = _read_object(existing_report_path)
        proposal = existing_report.get("typed_patch")
        llm_report = {
            "reused_existing_typed_patch": True,
            "original_report": NATURAL_IMPUTATION_PSEUDO_GAP_REPORT_PATH,
            "proposal": proposal,
        }
    else:
        import openai

        api_key = (
            os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("AGICTO_API_KEY", "").strip()
        )
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
        client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=120)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON typed Harness patch with patch_id and "
                        "one operation. The operation, target_surface and value must cover "
                        "the categorical first fault using the sole allowed patch value. "
                        "Do not modify Scope, Workflow, Consumer, Metric or Memory. No markdown."
                    ),
                },
                {"role": "user", "content": json.dumps(dossier, ensure_ascii=False)},
            ],
        )
        proposal = json.loads(completion.choices[0].message.content or "")
        llm_report = {
            "base_url": base_url,
            "requested_model": model,
            "returned_model": getattr(completion, "model", ""),
            "prompt_tokens": getattr(
                getattr(completion, "usage", None), "prompt_tokens", None
            ),
            "completion_tokens": getattr(
                getattr(completion, "usage", None), "completion_tokens", None
            ),
            "proposal": proposal,
        }
    expected_operation = {
        "operation": "ADD_OBSERVATION",
        "target_surface": "observation",
        "value": "pseudo_gap_program_response_order",
    }
    if (
        not isinstance(proposal, Mapping)
        or proposal.get("operations") != [expected_operation]
    ):
        raise ValueError("LLM patch is outside the pseudo-gap Dossier catalog")

    source_episodes = [
        {
            "workflows": {
                str(program["workflow_id"]): {
                    "workflow_id": program["workflow_id"],
                    "support_gain": float(program["support_gain_vs_identity"]),
                    "query_gain": float(program["query_gain_vs_identity"]),
                }
                for program in row["programs"]
                if program["workflow_id"] != "IDENTITY"
            }
        }
        for row in source["datasets"]
    ]
    candidate = build_candidate_skill(
        [],
        source_episodes,
        capability_id=str(source["candidate"]["capability_id"]),
        task_context={
            "task": "forecasting",
            "consumer": "shared frozen Ridge",
            "context": "observable natural missing values with known period",
        },
        workflow_supply=source_order,
    )
    scoped = apply_typed_patch(candidate, repair["typed_patch"])
    patched = apply_typed_patch(scoped, proposal)
    base_report.update(
        {
            "failure_dossier_sent_to_llm": dossier,
            "llm": llm_report,
            "typed_patch": proposal,
            "candidate_after_patch": {
                "status": patched["status"],
                "observation": patched["observation"],
                "applicability_unchanged": patched["applicability"]
                == scoped["applicability"],
                "workflow_supply_unchanged": patched["workflow_supply"]
                == scoped["workflow_supply"],
            },
            "resolved_candidate_status": "CANDIDATE",
            "information_boundary": {
                "pseudo_gap_truth_source": (
                    "originally_observed_historical_context_points"
                ),
                "natural_gap_truth_used": False,
                "post_context_horizon_values_received_by_observation": False,
                "llm_received_dataset_identity": False,
                "llm_received_raw_time_series": False,
                "llm_received_effect_magnitude": False,
                "repair_truth_used": False,
                "cached_query_effect_used_by_replay_evaluator": True,
            },
            "compute": {
                "consumer_fit_count": 0,
                "llm_api_call_count": 0 if reused_existing_patch else 1,
            },
            "verdict": "PSEUDO_GAP_OBSERVATION_PATCH_PASS",
            "claim_limit": (
                "Exposed cached Target diagnostic and Candidate patch only; this is "
                "not fresh confirmation, Promotion or Active Memory."
            ),
        }
    )
    return base_report


def _run_missing_window_air_quality_role(
    root: Path,
    *,
    channel: str,
) -> dict[str, object]:
    """Evaluate the frozen missing-window Skill on one independent channel role."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )

    source_report = _read_object(root / MISSING_WINDOW_WEIGHTING_P0_REPORT_PATH)
    if source_report.get("verdict") != "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS":
        raise ValueError("Air Quality Target requires frozen Source PolicyEpisodes")
    source_episodes = {
        str(row["dataset"]): row for row in source_report["dataset_results"]
    }
    values = np.asarray(_load_uci_air_quality_channel(root, channel), dtype=np.float64)
    observed_mask = np.isfinite(values)
    period = 24
    historical_shifts = (2, 3, 4)

    # Availability-only admission: no Program result is used to choose the origin.
    latest_stop = ((len(values) - HORIZON) // period) * period
    evaluation_stop = None
    for stop in range(latest_stop, CONTEXT_LENGTH + 8 * period, -period):
        origins = [stop] + [stop - shift * period for shift in historical_shifts]
        if min(origins) <= CONTEXT_LENGTH:
            continue
        if all(
            np.isfinite(values[origin : origin + HORIZON]).all()
            and np.isfinite(values[origin - CONTEXT_LENGTH : origin]).sum()
            >= CONTEXT_LENGTH // 2
            for origin in origins
        ):
            evaluation_stop = stop
            break
    if evaluation_stop is None:
        raise ValueError("no fixed Air Quality evaluation cutoff passed availability admission")

    earliest_historical_origin = evaluation_stop - max(historical_shifts) * period
    anchors = list(range(240, earliest_historical_origin - HORIZON, 72))
    x_train: list[Any] = []
    y_train: list[Any] = []
    unreliable: list[bool] = []
    missing_fractions: list[float] = []
    for anchor in anchors:
        context_raw = values[anchor - CONTEXT_LENGTH : anchor]
        target_raw = values[anchor : anchor + HORIZON]
        context = _linear_fill(np, context_raw)
        target = _linear_fill(np, target_raw)
        if context is None or target is None:
            continue
        center, scale, method = _center_scale(np, context)
        if method == "scale_floor_fallback":
            continue
        window_mask = ~observed_mask[anchor - CONTEXT_LENGTH : anchor + HORIZON]
        x_train.append((context - center) / scale)
        y_train.append((target - center) / scale)
        unreliable.append(bool(window_mask.any()))
        missing_fractions.append(float(window_mask.mean()))
    unreliable_array = np.asarray(unreliable, dtype=bool)
    if len(x_train) < 24 or not unreliable_array.any() or unreliable_array.all():
        raise ValueError("Air Quality missing-window training geometry is not actionable")

    eval_features: list[Any] = []
    eval_actual: list[Any] = []
    eval_centers: list[float] = []
    eval_scales: list[float] = []
    eval_seasonal_scales: list[float] = []
    eval_roles: list[str] = []
    eval_origins = [("query", evaluation_stop)] + [
        ("historical", evaluation_stop - shift * period)
        for shift in historical_shifts
    ]
    for role, origin in eval_origins:
        context = _linear_fill(np, values[origin - CONTEXT_LENGTH : origin])
        target = values[origin : origin + HORIZON]
        if context is None or not np.isfinite(target).all():
            raise AssertionError("Air Quality evaluation admission changed after freeze")
        center, scale, method = _center_scale(np, context)
        if method == "scale_floor_fallback":
            raise ValueError("Air Quality evaluation context hit the scale floor")
        try:
            scale_value = seasonal_scale(
                values[:origin], observed_mask[:origin], period=period, min_pairs=32
            )
        except (UndefinedSeasonalScale, ValueError) as exc:
            raise ValueError("Air Quality seasonal scale is undefined") from exc
        eval_features.append((context - center) / scale)
        eval_actual.append(target)
        eval_centers.append(center)
        eval_scales.append(scale)
        eval_seasonal_scales.append(scale_value)
        eval_roles.append(role)

    x_array = np.asarray(x_train, dtype=np.float64)
    y_array = np.asarray(y_train, dtype=np.float64)
    eval_array = np.asarray(eval_features, dtype=np.float64)
    programs = (
        ("KEEP_ALL", 1.0),
        ("ATTENUATE_MISSING_WINDOW", 0.25),
        ("EXCLUDE_MISSING_WINDOW", 0.0),
    )
    program_rows = []
    for program, unreliable_weight in programs:
        if unreliable_weight == 0.0:
            retained = ~unreliable_array
            prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=x_array[retained],
                targets=y_array[retained],
                weights=np.ones(int(retained.sum()), dtype=np.float64),
                x_eval=eval_array,
            )
        else:
            weights = np.ones(len(x_array), dtype=np.float64)
            weights[unreliable_array] = unreliable_weight
            prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=x_array,
                targets=y_array,
                weights=weights,
                x_eval=eval_array,
            )
        losses = []
        for index, truth in enumerate(eval_actual):
            raw_prediction = prediction[index] * eval_scales[index] + eval_centers[index]
            losses.append(smase(truth, raw_prediction, scale=eval_seasonal_scales[index]))
        query_losses = [loss for loss, role in zip(losses, eval_roles) if role == "query"]
        historical_losses = [
            loss for loss, role in zip(losses, eval_roles) if role == "historical"
        ]
        program_rows.append(
            {
                "program": program,
                "unreliable_window_weight": unreliable_weight,
                "query_loss": statistics.fmean(query_losses),
                "historical_policy_loss": statistics.fmean(historical_losses),
            }
        )
    identity = next(row for row in program_rows if row["program"] == "KEEP_ALL")
    for row in program_rows:
        row["query_gain_vs_identity"] = identity["query_loss"] - row["query_loss"]
        row["historical_policy_gain_vs_identity"] = (
            identity["historical_policy_loss"] - row["historical_policy_loss"]
        )
    actions = {
        str(row["program"]): {
            "query_gain": float(row["query_gain_vs_identity"]),
            "support_gain": float(row["historical_policy_gain_vs_identity"]),
        }
        for row in program_rows
        if row["program"] != "KEEP_ALL"
    }

    target_context = {
        "unreliable_window_fraction": float(unreliable_array.mean()),
        "median_missing_fraction": statistics.median(
            fraction for fraction, flag in zip(missing_fractions, unreliable) if flag
        ),
    }
    ranked_sources = sorted(
        [
            {
                "dataset": dataset_id,
                "episode": episode,
                "context": _missing_window_context(episode),
                "distance": _missing_window_context_distance(
                    target_context, _missing_window_context(episode)
                ),
            }
            for dataset_id, episode in source_episodes.items()
        ],
        key=lambda row: (float(row["distance"]), str(row["dataset"])),
    )
    retrieved = ranked_sources[0]
    a5_order = _missing_window_probe_order(retrieved["episode"])
    orders = [
        ["ATTENUATE_MISSING_WINDOW", "EXCLUDE_MISSING_WINDOW"],
        ["EXCLUDE_MISSING_WINDOW", "ATTENUATE_MISSING_WINDOW"],
    ]
    a5_curve = _semantic_auxiliary_budget_curve(a5_order, actions)
    a3_curve = _mean_missing_window_curves(
        [_semantic_auxiliary_budget_curve(order, actions) for order in orders]
    )
    a3_auc = policy_episode_adapt_auc(a3_curve)
    a5_auc = policy_episode_adapt_auc(a5_curve)
    source_actions = _missing_window_action_rows(retrieved["episode"])
    a4_program = a5_order[0]
    if source_actions[a4_program]["combined_gain"] <= 0.0:
        a4_program = "KEEP_ALL"
    a4_query_gain = 0.0 if a4_program == "KEEP_ALL" else actions[a4_program]["query_gain"]
    menu_oracle = max(
        [("KEEP_ALL", 0.0)]
        + [(program, row["query_gain"]) for program, row in actions.items()],
        key=lambda item: (float(item[1]), item[0]),
    )
    harmful_a5 = any(float(row["fixed_query_gain"]) < -0.005 for row in a5_curve[1:])
    passed = float(menu_oracle[1]) > 0.005 and a5_auc > a3_auc and not harmful_a5
    return {
        "channel": channel,
        "geometry": {
            "training_anchor_count": len(x_train),
            "evaluation_stop": evaluation_stop,
            "historical_origins": [origin for role, origin in eval_origins if role == "historical"],
            "query_future_used_for_planning_or_confirmation": False,
        },
        "target_context": target_context,
        "retrieved_source_episode": {
            "dataset_audit_only": retrieved["dataset"],
            "context": retrieved["context"],
            "probe_order": a5_order,
        },
        "programs": program_rows,
        "arms": {
            "A3": {"curve": a3_curve, "adapt_auc": a3_auc},
            "A4": {
                "selected_program": a4_program,
                "fixed_query_gain": a4_query_gain,
                "harmful": a4_query_gain < -0.005,
            },
            "A5": {"curve": a5_curve, "adapt_auc": a5_auc, "harmful": harmful_a5},
        },
        "summary": {
            "unreliable_training_window_count": int(unreliable_array.sum()),
            "menu_oracle_program_evaluator_only": menu_oracle[0],
            "menu_oracle_query_gain_evaluator_only": menu_oracle[1],
            "A5_minus_A3": a5_auc - a3_auc,
        },
        "gate": {
            "material_query_headroom": float(menu_oracle[1]) > 0.005,
            "A5_strictly_greater_than_A3": a5_auc > a3_auc,
            "A5_harmful_false": not harmful_a5,
            "passed": passed,
        },
    }


def run_natural_missing_window_weighting_air_quality(
    root: Path,
) -> dict[str, object]:
    """Frozen two-role confirmation on an independent natural-missing dataset."""

    role_reports = []
    feasibility_rejections = []
    for channel in ("CO(GT)", "NO2(GT)"):
        try:
            role_reports.append(_run_missing_window_air_quality_role(root, channel=channel))
        except ValueError as exc:
            feasibility_rejections.append({"channel": channel, "reason": str(exc)})
    if not role_reports:
        return {
            "experiment_id": "E2-natural-missing-window-weighting-UCI-Air-Quality",
            "target_roles": [],
            "feasibility_rejections": feasibility_rejections,
            "verdict": "NO_ACTIONABLE_INDEPENDENT_TARGET_CLOSE_SKILL_VERSION",
            "capability_or_memory_written": False,
        }
    a3_macro = statistics.fmean(float(row["arms"]["A3"]["adapt_auc"]) for row in role_reports)
    a5_macro = statistics.fmean(float(row["arms"]["A5"]["adapt_auc"]) for row in role_reports)
    material_roles = sum(float(row["summary"]["menu_oracle_query_gain_evaluator_only"]) > 0.005 for row in role_reports)
    helpful_roles = sum(float(row["summary"]["A5_minus_A3"]) > 0.0 for row in role_reports)
    harmful_roles = sum(bool(row["arms"]["A5"]["harmful"]) for row in role_reports)
    passed = material_roles >= 1 and helpful_roles >= 1 and a5_macro > a3_macro and harmful_roles == 0
    return {
        "experiment_id": "E2-natural-missing-window-weighting-UCI-Air-Quality",
        "scientific_role": "independent natural dataset confirmation of a frozen Skill candidate",
        "dataset": "UCI Air Quality",
        "source": {
            "url": "https://archive.ics.uci.edu/dataset/360/air+quality",
            "license": "CC BY 4.0",
            "missing_semantics": "official -200 missing marker converted to NaN",
        },
        "target_roles": role_reports,
        "feasibility_rejections": feasibility_rejections,
        "summary": {
            "role_count": len(role_reports),
            "material_headroom_role_count": material_roles,
            "A5_helpful_role_count": helpful_roles,
            "A5_harmful_role_count": harmful_roles,
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
        },
        "information_boundary": {
            "parser_header_and_initial_rows_seen_before_run": True,
            "evaluation_outcomes_sealed_until_this_run": True,
            "query_future_used_for_planning_or_confirmation": False,
        },
        "compute": {"consumer_fit_count": 3 * len(role_reports), "llm_api_call_count": 0},
        "gate": {
            "at_least_one_role_has_material_headroom": material_roles >= 1,
            "at_least_one_role_A5_strictly_improves_A3": helpful_roles >= 1,
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_harmful_role_count_zero": harmful_roles == 0,
            "passed": passed,
        },
        "verdict": (
            "INDEPENDENT_DATASET_MISSING_WINDOW_SKILL_CONFIRMATION_PASS"
            if passed
            else "INDEPENDENT_DATASET_MISSING_WINDOW_SKILL_CONFIRMATION_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "One independent natural dataset with two predeclared target roles. A PASS "
            "confirms this narrow Skill version; it is not a universal missing-data rule."
        ),
    }


def run_natural_missing_window_weighting_p2(root: Path) -> dict[str, object]:
    """Replace cross-series confirmation with legal historical PolicyEpisodes."""

    source_report = _read_object(root / MISSING_WINDOW_WEIGHTING_P0_REPORT_PATH)
    p1_report = _read_object(root / MISSING_WINDOW_WEIGHTING_P1_REPORT_PATH)
    if source_report.get("verdict") != "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS":
        raise ValueError("P2 requires the frozen P0 Source episodes")
    source_episodes = {
        str(row["dataset"]): row for row in source_report["dataset_results"]
    }
    target_specs: dict[str, dict[str, object]] = {
        "monash:nn5_daily": {
            "anchors": (240, 300, 360, 420),
            "evaluation_stop": 720,
            "period": 7,
            "train_start": 20,
            "eval_start": 32,
            "historical_origin_count": 3,
        },
        "gefcom2012_load": {
            "anchors": (312, 372, 432, 492, 552, 612, 672, 732, 792, 852),
            "evaluation_stop": 912,
            "period": 24,
            "train_start": 8,
            "eval_start": 0,
            "historical_origin_count": 3,
        },
        "noaa_global_hourly": {
            "anchors": (240, 300, 360, 420, 480, 540, 600, 660),
            "evaluation_stop": 912,
            "period": 24,
            "train_start": 12,
            "eval_start": 24,
            "historical_origin_count": 3,
        },
    }
    target_report = run_natural_missing_window_weighting_p0(
        root, specs_override=target_specs
    )
    p1_by_dataset = {
        str(row["dataset"]): row for row in p1_report["target_results"]
    }
    program_orders = [
        ["ATTENUATE_MISSING_WINDOW", "EXCLUDE_MISSING_WINDOW"],
        ["EXCLUDE_MISSING_WINDOW", "ATTENUATE_MISSING_WINDOW"],
    ]
    target_results = []
    for target_episode in target_report["dataset_results"]:
        dataset_id = str(target_episode["dataset"])
        target_context = _missing_window_context(target_episode)
        source_candidates = []
        for source_dataset, source_episode in source_episodes.items():
            if source_dataset == dataset_id:
                continue
            source_context = _missing_window_context(source_episode)
            source_candidates.append(
                {
                    "dataset": source_dataset,
                    "episode": source_episode,
                    "context": source_context,
                    "distance": _missing_window_context_distance(
                        target_context, source_context
                    ),
                }
            )
        retrieved = min(
            source_candidates,
            key=lambda row: (float(row["distance"]), str(row["dataset"])),
        )
        a5_order = _missing_window_probe_order(retrieved["episode"])
        actions = _missing_window_action_rows(target_episode)
        historical_actions = {
            program: {
                "support_gain": float(row["historical_gain"]),
                "query_gain": float(row["query_gain"]),
            }
            for program, row in actions.items()
        }
        a5_curve = _semantic_auxiliary_budget_curve(a5_order, historical_actions)
        a3_order_curves = [
            _semantic_auxiliary_budget_curve(order, historical_actions)
            for order in program_orders
        ]
        a3_curve = _mean_missing_window_curves(a3_order_curves)
        a5_auc = policy_episode_adapt_auc(a5_curve)
        a3_auc = policy_episode_adapt_auc(a3_curve)
        target_results.append(
            {
                "dataset": dataset_id,
                "target_context": target_context,
                "historical_policy_observation": target_episode[
                    "phase_aligned_historical_policy_observation"
                ],
                "historical_action_response": {
                    program: {
                        "historical_gain": row["historical_gain"],
                        "current_query_gain_evaluator_only": row["query_gain"],
                    }
                    for program, row in actions.items()
                },
                "retrieved_source_episode": {
                    "dataset_audit_only": retrieved["dataset"],
                    "context": retrieved["context"],
                    "distance": retrieved["distance"],
                },
                "A3": {
                    "definition": "both legal orders using historical Target feedback",
                    "order_curves": a3_order_curves,
                    "mean_curve": a3_curve,
                    "adapt_auc": a3_auc,
                },
                "A5": {
                    "definition": "Source order plus phase-aligned historical Target confirmation",
                    "probe_order": a5_order,
                    "curve": a5_curve,
                    "adapt_auc": a5_auc,
                    "harmful": any(
                        float(row["fixed_query_gain"]) < -0.005
                        for row in a5_curve[1:]
                    ),
                },
                "A5_minus_A3": a5_auc - a3_auc,
                "cross_series_support_A5_auc_P1": float(
                    p1_by_dataset[dataset_id]["A5"]["adapt_auc"]
                ),
                "historical_minus_cross_series_A5": a5_auc
                - float(p1_by_dataset[dataset_id]["A5"]["adapt_auc"]),
            }
        )

    a3_macro = statistics.fmean(float(row["A3"]["adapt_auc"]) for row in target_results)
    a5_macro = statistics.fmean(float(row["A5"]["adapt_auc"]) for row in target_results)
    p1_a5_macro = float(p1_report["summary"]["adapt_auc"]["A5"])
    nonnegative = sum(float(row["A5_minus_A3"]) >= 0.0 for row in target_results)
    harmful = sum(bool(row["A5"]["harmful"]) for row in target_results)
    passed = a5_macro > a3_macro and nonnegative >= 2 and harmful == 0
    return {
        "experiment_id": "E2-natural-missing-window-weighting-P2",
        "scientific_role": "failure-derived Target-feedback Observation comparison",
        "first_fault_from_P1": (
            "cross-sectional Support composition can contradict current Query response; "
            "NOAA abstained although both candidate Programs had positive Query gains"
        ),
        "causal_change": (
            "replace cross-series Support confirmation with phase-aligned historical "
            "PolicyEpisodes on the current Query identities; keep Program supply, Source "
            "retrieval, stop-on-first-positive and Consumer unchanged"
        ),
        "task": "forecasting",
        "target_results": target_results,
        "summary": {
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
            "cross_series_support_A5_P1": p1_a5_macro,
            "historical_minus_cross_series_A5": a5_macro - p1_a5_macro,
            "A5_nonnegative_target_count": nonnegative,
            "A5_harmful_target_count": harmful,
        },
        "information_boundary": {
            "historical_targets_end_before_current_query_cutoff": True,
            "historical_observation_uses_current_query_identities": True,
            "current_query_future_used_for_planning_or_confirmation": False,
            "development_only": True,
        },
        "compute": {
            "consumer_fit_count": target_report["compute"]["consumer_fit_count"],
            "llm_api_call_count": 0,
            "proxy_call_count": 0,
        },
        "gate": {
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_nonnegative_at_least_2_of_3": nonnegative >= 2,
            "A5_harmful_target_count_zero": harmful == 0,
            "passed": passed,
        },
        "verdict": (
            "HISTORICAL_POLICY_CONFIRMATION_COMPOSITION_PASS"
            if passed
            else "HISTORICAL_POLICY_CONFIRMATION_COMPOSITION_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Development reuse of an existing Observation mechanism. A pass would justify "
            "a fresh confirmation of this composed natural Forecasting Skill, not Promotion."
        ),
    }


def run_natural_missing_window_weighting_p1(root: Path) -> dict[str, object]:
    """Test Source-context probe ordering on disjoint/later natural cohorts."""

    source_report = _read_object(root / MISSING_WINDOW_WEIGHTING_P0_REPORT_PATH)
    if source_report.get("verdict") != "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS":
        raise ValueError("P1 requires the frozen successful P0 Source report")
    source_episodes = {
        str(row["dataset"]): row for row in source_report["dataset_results"]
    }
    target_specs: dict[str, dict[str, object]] = {
        "monash:nn5_daily": {
            "anchors": (240, 300, 360, 420),
            "evaluation_stop": 720,
            "period": 7,
            "train_start": 20,
            "eval_start": 32,
        },
        "gefcom2012_load": {
            "anchors": (312, 372, 432, 492, 552, 612, 672, 732, 792, 852),
            "evaluation_stop": 912,
            "period": 24,
            "train_start": 8,
            "eval_start": 0,
        },
        "noaa_global_hourly": {
            "anchors": (240, 300, 360, 420, 480, 540, 600, 660),
            "evaluation_stop": 912,
            "period": 24,
            "train_start": 12,
            "eval_start": 24,
        },
    }
    target_report = run_natural_missing_window_weighting_p0(
        root, specs_override=target_specs
    )
    target_results: list[dict[str, object]] = []
    program_orders = [
        ["ATTENUATE_MISSING_WINDOW", "EXCLUDE_MISSING_WINDOW"],
        ["EXCLUDE_MISSING_WINDOW", "ATTENUATE_MISSING_WINDOW"],
    ]
    for target_episode in target_report["dataset_results"]:
        dataset_id = str(target_episode["dataset"])
        target_context = _missing_window_context(target_episode)
        source_candidates = []
        for source_dataset, source_episode in source_episodes.items():
            if source_dataset == dataset_id:
                continue
            source_context = _missing_window_context(source_episode)
            source_candidates.append(
                {
                    "dataset": source_dataset,
                    "episode": source_episode,
                    "context": source_context,
                    "distance": _missing_window_context_distance(
                        target_context, source_context
                    ),
                }
            )
        retrieved = min(
            source_candidates,
            key=lambda row: (float(row["distance"]), str(row["dataset"])),
        )
        source_episode = retrieved["episode"]
        a5_order = _missing_window_probe_order(source_episode)
        target_actions = _missing_window_action_rows(target_episode)
        a5_curve = _semantic_auxiliary_budget_curve(a5_order, target_actions)
        a3_order_curves = [
            _semantic_auxiliary_budget_curve(order, target_actions)
            for order in program_orders
        ]
        a3_curve = _mean_missing_window_curves(a3_order_curves)
        a5_auc = policy_episode_adapt_auc(a5_curve)
        a3_auc = policy_episode_adapt_auc(a3_curve)

        source_actions = _missing_window_action_rows(source_episode)
        a4_program = a5_order[0]
        if source_actions[a4_program]["combined_gain"] <= 0.0:
            a4_program = "KEEP_ALL"
        a4_query_gain = (
            0.0
            if a4_program == "KEEP_ALL"
            else target_actions[a4_program]["query_gain"]
        )
        target_results.append(
            {
                "dataset": dataset_id,
                "target_roster": target_episode["roster"],
                "target_context": target_context,
                "retrieved_source_episode": {
                    "dataset_audit_only": retrieved["dataset"],
                    "context": retrieved["context"],
                    "distance": retrieved["distance"],
                    "candidate_combined_gains": {
                        program: source_actions[program]["combined_gain"]
                        for program in a5_order
                    },
                },
                "target_action_response_evaluator_only": target_actions,
                "A3": {
                    "definition": "equal expectation over both target-only probe orders",
                    "order_curves": a3_order_curves,
                    "mean_curve": a3_curve,
                    "adapt_auc": a3_auc,
                },
                "A4": {
                    "definition": "nearest Source episode direct action; no Target confirmation",
                    "selected_program": a4_program,
                    "fixed_query_gain": a4_query_gain,
                    "harmful": a4_query_gain < -0.005,
                },
                "A5": {
                    "definition": "Source-context probe order plus current Target Support confirmation",
                    "probe_order": a5_order,
                    "curve": a5_curve,
                    "adapt_auc": a5_auc,
                    "harmful": any(
                        float(row["fixed_query_gain"]) < -0.005
                        for row in a5_curve[1:]
                    ),
                },
                "A5_minus_A3": a5_auc - a3_auc,
                "menu_oracle_program_evaluator_only": target_episode[
                    "menu_oracle_program"
                ],
                "menu_oracle_combined_gain_evaluator_only": target_episode[
                    "menu_oracle_combined_gain"
                ],
            }
        )

    a3_macro = statistics.fmean(float(row["A3"]["adapt_auc"]) for row in target_results)
    a5_macro = statistics.fmean(float(row["A5"]["adapt_auc"]) for row in target_results)
    a4_macro = statistics.fmean(float(row["A4"]["fixed_query_gain"]) for row in target_results)
    nonnegative = sum(float(row["A5_minus_A3"]) >= 0.0 for row in target_results)
    harmful_a5 = sum(bool(row["A5"]["harmful"]) for row in target_results)
    harmful_a4 = sum(bool(row["A4"]["harmful"]) for row in target_results)
    passed = a5_macro > a3_macro and nonnegative >= 2 and harmful_a5 == 0
    return {
        "experiment_id": "E2-natural-missing-window-weighting-P1",
        "scientific_role": "development cross-cohort Source-guided Workflow adaptation",
        "causal_hypothesis": (
            "Source PolicyEpisodes retrieved by observable training-window missingness "
            "composition can order Target probes better than target-only ordering, while "
            "current Target Support confirmation prevents harmful execution."
        ),
        "task": "forecasting",
        "consumer": target_report["consumer"],
        "metric": target_report["metric"],
        "observation": {
            "fields": [
                "unreliable_training_window_fraction",
                "median_missing_fraction_within_unreliable_windows",
            ],
            "retrieval": "leave-same-dataset-out nearest Source PolicyEpisode by fixed L1 distance",
            "dataset_identity_used_for_decision": False,
        },
        "workflow": {
            "source_memory_use": "order two Program probes, not certify utility",
            "target_confirmation": "execute first Program with exact Support gain > 0",
            "fallback": "KEEP_ALL",
            "program_supply": program_orders[0],
        },
        "arms": {
            "A3": "Target-only adaptation; expected value across both legal probe orders",
            "A4": "direct Source retrieval without Target feedback",
            "A5": "Source-guided order plus Target-local exact Support confirmation",
        },
        "target_results": target_results,
        "summary": {
            "target_count": len(target_results),
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
            "A4_zero_feedback_query_gain": a4_macro,
            "A5_nonnegative_target_count": nonnegative,
            "A5_harmful_target_count": harmful_a5,
            "A4_harmful_target_count": harmful_a4,
        },
        "information_boundary": {
            "source_outcomes": "EXPOSED P0 PolicyEpisodes",
            "target_context_exposure_before_run": "AGGREGATE_SEEN",
            "target_outcome_exposure_after_run": "EXPOSED",
            "query_future_used_for_planning_or_confirmation": False,
            "development_only": True,
        },
        "compute": {
            "consumer_fit_count": target_report["compute"]["consumer_fit_count"],
            "llm_api_call_count": 0,
            "proxy_call_count": 0,
        },
        "gate": {
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_nonnegative_at_least_2_of_3": nonnegative >= 2,
            "A5_harmful_target_count_zero": harmful_a5 == 0,
            "passed": passed,
        },
        "verdict": (
            "SOURCE_GUIDED_MISSING_WINDOW_WORKFLOW_PREMISE_PASS"
            if passed
            else "SOURCE_GUIDED_MISSING_WINDOW_WORKFLOW_PREMISE_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "Development cross-cohort evidence only. A pass supports one TS-native "
            "Observation/Workflow premise; it is not fresh cross-dataset Promotion."
        ),
    }


def _semantic_phase_aligned_response(
    np: Any,
    raw_values: Any,
    *,
    anchor: int,
    period: int,
) -> Any | None:
    """Normalized actual-minus-seasonal-naive response from visible training history."""

    raw = np.asarray(raw_values, dtype=np.float64)
    context = raw[anchor - CONTEXT_LENGTH : anchor]
    actual = raw[anchor : anchor + HORIZON]
    _, scale, method = _center_scale(np, context)
    if (
        context.shape != (CONTEXT_LENGTH,)
        or actual.shape != (HORIZON,)
        or not np.isfinite(context).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("invalid semantic response geometry")
    if method == "scale_floor_fallback":
        return None
    seasonal_naive = np.resize(context[-period:], HORIZON)
    response = (actual - seasonal_naive) / scale
    if response.shape != (HORIZON,) or not np.isfinite(response).all():
        raise RuntimeError("semantic response is non-finite")
    return response


def _semantic_auxiliary_budget_curve(
    order: list[str],
    actions: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    """Probe at most two groups and stop on the first positive exact Support response."""

    curve: list[dict[str, object]] = [
        {
            "budget": 0,
            "probed_group": None,
            "probed_support_gain": None,
            "probed_query_gain_evaluator_only": None,
            "selected_action": "IDENTITY",
            "selected_auxiliary_group": None,
            "selected_support_gain": 0.0,
            "fixed_query_gain": 0.0,
            "abstained": True,
            "terminal": False,
        }
    ]
    winner: str | None = None
    for budget in (1, 2):
        probed_group: str | None = None
        probed: dict[str, float] | None = None
        if winner is None:
            probed_group = order[budget - 1]
            probed = actions[probed_group]
            if float(probed["support_gain"]) > 0.0:
                winner = probed_group
        selected = None if winner is None else actions[winner]
        curve.append(
            {
                "budget": budget,
                "probed_group": probed_group,
                "probed_support_gain": (
                    None if probed is None else float(probed["support_gain"])
                ),
                "probed_query_gain_evaluator_only": (
                    None if probed is None else float(probed["query_gain"])
                ),
                "selected_action": (
                    "IDENTITY"
                    if winner is None
                    else "ADD_AUXILIARY_SEMANTIC_GROUP"
                ),
                "selected_auxiliary_group": winner,
                "selected_support_gain": (
                    0.0 if selected is None else float(selected["support_gain"])
                ),
                "fixed_query_gain": (
                    0.0 if selected is None else float(selected["query_gain"])
                ),
                "abstained": winner is None,
                "terminal": winner is not None,
            }
        )
    return curve


def run_semantic_auxiliary_p1(root: Path) -> dict[str, object]:
    """Test one phase-aligned response-alignment Observation using cached P0 utility."""

    import numpy as np

    p0_path = root / SEMANTIC_AUXILIARY_P0_REPORT_PATH
    p0 = _read_object(p0_path)
    if (
        p0.get("exposure") != "EXPOSED_DEVELOPMENT"
        or p0.get("verdict") != "PROGRAM_HEADROOM_AND_MATCHED_RISK_PASS"
        or p0.get("capability_or_memory_written") is not False
    ):
        raise ValueError("semantic auxiliary P0 is not the frozen exposed cache")
    p0_cells = {
        (str(row["dataset"]), str(row["target"]), str(row["auxiliary_group"])): row
        for row in p0["cells"]
    }
    p0_targets = {
        (str(row["dataset"]), str(row["target"])): row
        for row in p0["dataset_target_summaries"]
    }

    target_results: list[dict[str, object]] = []
    for spec in SEMANTIC_AUXILIARY_DATASETS:
        dataset_id = str(spec["dataset_id"])
        anchors = tuple(int(value) for value in spec["anchors"])
        period = int(spec["period"])
        attributes, rows = _read_semantic_tsf_panel(
            np,
            archive_path=Path(str(spec["archive"])),
            member=str(spec["member"]),
            required_stop=max(anchors) + HORIZON,
        )
        identity_fields = tuple(str(value) for value in spec["identity_fields"])
        identity_filter = {
            str(key): str(value)
            for key, value in dict(spec["identity_filter"]).items()
        }
        semantic_field = str(spec["semantic_field"])
        semantic_roles = tuple(str(value) for value in spec["semantic_roles"])
        if not (set(identity_fields) | set(identity_filter) | {semantic_field}).issubset(
            attributes
        ):
            raise ValueError(f"semantic P1 metadata changed: {dataset_id}")

        by_identity: dict[tuple[str, ...], dict[str, dict[str, object]]] = {}
        identity_order: list[tuple[str, ...]] = []
        for row in rows:
            if any(str(row[key]) != value for key, value in identity_filter.items()):
                continue
            identity = tuple(str(row[key]) for key in identity_fields)
            role = str(row[semantic_field])
            if identity not in by_identity:
                by_identity[identity] = {}
                identity_order.append(identity)
            by_identity[identity][role] = row
        complete_identities = [
            identity
            for identity in identity_order
            if set(semantic_roles).issubset(by_identity[identity])
        ]
        train_identities = complete_identities[:12]
        if len(train_identities) != 12:
            raise ValueError(f"semantic P1 train roster changed: {dataset_id}")

        for target_role_raw in spec["target_roles"]:
            target_role = str(target_role_raw)
            frozen_target = p0_targets[(dataset_id, target_role)]
            observed_train_roster = [
                str(by_identity[identity][target_role]["series_name"])
                for identity in train_identities
            ]
            if observed_train_roster != list(frozen_target["roster"]["train"]):
                raise AssertionError("semantic P1 did not reproduce the P0 train roster")

            target_responses: dict[tuple[tuple[str, ...], int], Any] = {}
            for identity in train_identities:
                raw = by_identity[identity][target_role]["values"]
                for anchor in anchors:
                    response = _semantic_phase_aligned_response(
                        np, raw, anchor=anchor, period=period
                    )
                    if response is None:
                        raise ValueError(
                            f"target response scale floor: {dataset_id}/{target_role}"
                        )
                    target_responses[(identity, anchor)] = response

            observations: list[dict[str, object]] = []
            actions: dict[str, dict[str, float]] = {}
            ineligible_groups: list[dict[str, str]] = []
            for auxiliary_role in semantic_roles:
                if auxiliary_role == target_role:
                    continue
                cached = p0_cells[(dataset_id, target_role, auxiliary_role)]
                if cached["eligibility"] != "ELIGIBLE":
                    ineligible_groups.append(
                        {
                            "auxiliary_group": auxiliary_role,
                            "eligibility": "INELIGIBLE",
                            "reason": str(cached["ineligible_reason"]),
                        }
                    )
                    continue
                program = next(
                    row
                    for row in cached["programs"]
                    if float(row["auxiliary_weight"]) == 0.25
                )
                actions[auxiliary_role] = {
                    "support_gain": float(program["support_gain"]),
                    "query_gain": float(program["query_gain"]),
                }
                cosines: list[float] = []
                normalized_rmses: list[float] = []
                auxiliary_scale_floor = False
                for identity in train_identities:
                    raw = by_identity[identity][auxiliary_role]["values"]
                    for anchor in anchors:
                        auxiliary_response = _semantic_phase_aligned_response(
                            np, raw, anchor=anchor, period=period
                        )
                        if auxiliary_response is None:
                            auxiliary_scale_floor = True
                            break
                        target_response = target_responses[(identity, anchor)]
                        denominator = float(
                            np.linalg.norm(target_response)
                            * np.linalg.norm(auxiliary_response)
                        )
                        if denominator <= 1e-12:
                            continue
                        cosines.append(
                            float(np.dot(target_response, auxiliary_response) / denominator)
                        )
                        normalized_rmses.append(
                            float(
                                np.sqrt(
                                    np.mean(
                                        (target_response - auxiliary_response) ** 2
                                    )
                                )
                            )
                        )
                    if auxiliary_scale_floor:
                        break
                if auxiliary_scale_floor:
                    ineligible_groups.append(
                        {
                            "auxiliary_group": auxiliary_role,
                            "eligibility": "INELIGIBLE",
                            "reason": "AUXILIARY_TRAINING_SCALE_FLOOR",
                        }
                    )
                    actions.pop(auxiliary_role)
                    continue
                if not cosines or len(cosines) != len(normalized_rmses):
                    raise ValueError(
                        f"no valid response pairs: {dataset_id}/{target_role}/{auxiliary_role}"
                    )
                observations.append(
                    {
                        "auxiliary_group": auxiliary_role,
                        "median_cosine_alignment": float(np.median(cosines)),
                        "median_normalized_rmse": float(np.median(normalized_rmses)),
                        "valid_pair_count": len(cosines),
                    }
                )

            cosine_order = sorted(
                observations,
                key=lambda row: (
                    -float(row["median_cosine_alignment"]),
                    str(row["auxiliary_group"]),
                ),
            )
            rmse_order = sorted(
                observations,
                key=lambda row: (
                    float(row["median_normalized_rmse"]),
                    str(row["auxiliary_group"]),
                ),
            )
            cosine_rank = {
                str(row["auxiliary_group"]): rank
                for rank, row in enumerate(cosine_order, 1)
            }
            rmse_rank = {
                str(row["auxiliary_group"]): rank
                for rank, row in enumerate(rmse_order, 1)
            }
            for row in observations:
                role = str(row["auxiliary_group"])
                row["higher_cosine_rank"] = cosine_rank[role]
                row["lower_rmse_rank"] = rmse_rank[role]
                row["average_rank"] = 0.5 * (
                    cosine_rank[role] + rmse_rank[role]
                )
            response_order = [
                str(row["auxiliary_group"])
                for row in sorted(
                    observations,
                    key=lambda row: (
                        float(row["average_rank"]),
                        str(row["auxiliary_group"]),
                    ),
                )
            ]
            eligible_roles = set(actions)
            metadata_position = {
                role: position for position, role in enumerate(semantic_roles)
            }
            a3_order = sorted(
                eligible_roles,
                key=lambda role: (
                    _semantic_family(dataset_id, role)
                    != _semantic_family(dataset_id, target_role),
                    metadata_position[role],
                ),
            )
            alphabetical_order = sorted(eligible_roles)
            if set(response_order) != eligible_roles or len(response_order) < 2:
                raise AssertionError("semantic P1 candidate supply changed")

            a3_curve = _semantic_auxiliary_budget_curve(a3_order, actions)
            a5_curve = _semantic_auxiliary_budget_curve(response_order, actions)
            alphabetical_curve = _semantic_auxiliary_budget_curve(
                alphabetical_order, actions
            )
            a3_auc = policy_episode_adapt_auc(a3_curve)
            a5_auc = policy_episode_adapt_auc(a5_curve)
            alphabetical_auc = policy_episode_adapt_auc(alphabetical_curve)
            behavior_changed = [
                row["selected_auxiliary_group"] for row in a5_curve
            ] != [row["selected_auxiliary_group"] for row in a3_curve]
            budget_order_changed = response_order[:2] != a3_order[:2]
            target_results.append(
                {
                    "dataset": dataset_id,
                    "target": target_role,
                    "observation": {
                        "training_only": True,
                        "phase_aligned_anchors": list(anchors),
                        "training_identity_count": len(train_identities),
                        "response_definition": (
                            "(actual - seasonal_naive) / training-context scale"
                        ),
                        "normalized_rmse_definition": (
                            "RMSE between the two individually context-normalized responses"
                        ),
                        "pairs": observations,
                    },
                    "ineligible_groups": ineligible_groups,
                    "response_order_A5": response_order,
                    "same_family_then_metadata_order_A3": a3_order,
                    "alphabetical_order_descriptive": alphabetical_order,
                    "A5_curve": a5_curve,
                    "A3_curve": a3_curve,
                    "alphabetical_curve_descriptive": alphabetical_curve,
                    "adapt_auc": {
                        "A5": a5_auc,
                        "A3": a3_auc,
                        "alphabetical_descriptive": alphabetical_auc,
                    },
                    "A5_minus_A3": a5_auc - a3_auc,
                    "A5_harmful": any(
                        float(row["fixed_query_gain"]) < 0.0
                        for row in a5_curve[1:]
                    ),
                    "A5_abstained_at_B2": bool(a5_curve[-1]["abstained"]),
                    "budget_relevant_probe_order_changed": budget_order_changed,
                    "selected_behavior_changed": behavior_changed,
                    "order_or_behavior_changed": (
                        budget_order_changed or behavior_changed
                    ),
                }
            )

    a5_macro = statistics.fmean(
        float(row["adapt_auc"]["A5"]) for row in target_results
    )
    a3_macro = statistics.fmean(
        float(row["adapt_auc"]["A3"]) for row in target_results
    )
    nonnegative_target_count = sum(
        float(row["A5_minus_A3"]) >= 0.0 for row in target_results
    )
    harmful_target_count = sum(bool(row["A5_harmful"]) for row in target_results)
    changed_target_count = sum(
        bool(row["order_or_behavior_changed"]) for row in target_results
    )
    dataset_macro: dict[str, dict[str, float]] = {}
    for dataset_id in {str(row["dataset"]) for row in target_results}:
        selected = [row for row in target_results if row["dataset"] == dataset_id]
        dataset_macro[dataset_id] = {
            "A5": statistics.fmean(
                float(row["adapt_auc"]["A5"]) for row in selected
            ),
            "A3": statistics.fmean(
                float(row["adapt_auc"]["A3"]) for row in selected
            ),
        }
        dataset_macro[dataset_id]["A5_minus_A3"] = (
            dataset_macro[dataset_id]["A5"] - dataset_macro[dataset_id]["A3"]
        )
    passed = (
        a5_macro > a3_macro
        and nonnegative_target_count >= 3
        and harmful_target_count == 0
        and changed_target_count >= 1
    )
    if len(target_results) != 4 or any(
        [int(row["budget"]) for row in result["A5_curve"]] != [0, 1, 2]
        or [int(row["budget"]) for row in result["A3_curve"]] != [0, 1, 2]
        for result in target_results
    ):
        raise AssertionError("semantic P1 smoke check failed")

    return {
        "experiment_id": "E2-semantic-auxiliary-group-augmentation-P1",
        "scientific_role": "exposed-development response-alignment Observation premise",
        "exposure": "EXPOSED_DEVELOPMENT",
        "frozen_hypothesis": (
            "Training-visible phase-aligned target/auxiliary response alignment can "
            "order exact Support probes better than same-family public metadata order."
        ),
        "protocol": {
            "p0_action_response_cache": SEMANTIC_AUXILIARY_P0_REPORT_PATH,
            "candidate_program": "ADD_AUXILIARY_SEMANTIC_GROUP(weight=0.25)",
            "feedback_budgets": [0, 1, 2],
            "control": "stop on first strictly positive cached exact Support gain; otherwise continue then abstain",
            "A3": "same semantic family first, then frozen public metadata order",
            "A5": "mean rank of higher median cosine and lower median normalized RMSE; role name tie-break",
            "alphabetical": "descriptive only",
            "ordering_reads_support_or_query_future": False,
            "consumer": "unchanged frozen Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "unchanged per-series sMASE",
            "roster_weights_or_windows_changed": False,
            "original_uci_opened": False,
        },
        "targets": target_results,
        "macro": {
            "dataset_target_macro_adapt_auc": {"A5": a5_macro, "A3": a3_macro},
            "A5_minus_A3": a5_macro - a3_macro,
            "per_dataset": dataset_macro,
            "nonnegative_target_count": nonnegative_target_count,
            "harmful_A5_target_count": harmful_target_count,
            "changed_order_or_behavior_target_count": changed_target_count,
            "A5_B2_abstain_target_count": sum(
                bool(row["A5_abstained_at_B2"]) for row in target_results
            ),
        },
        "compute_counts": {
            "incremental_consumer_solve_count": 0,
            "incremental_candidate_consumer_fit_count": 0,
            "cached_P0_weight_0_25_action_response_count": sum(
                len(result["observation"]["pairs"])
                for result in target_results
            ),
            "response_observation_pair_count": sum(
                int(pair["valid_pair_count"])
                for result in target_results
                for pair in result["observation"]["pairs"]
            ),
        },
        "gate": {
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_minus_A3_nonnegative_at_least_3_of_4": (
                nonnegative_target_count >= 3
            ),
            "A5_harmful_target_count_zero": harmful_target_count == 0,
            "A5_changes_order_or_behavior": changed_target_count >= 1,
            "passed": passed,
        },
        "verdict": (
            "RESPONSE_ALIGNMENT_WORKFLOW_PREMISE_PASS"
            if passed
            else "RESPONSE_ALIGNMENT_OBSERVATION_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "exposed P1 ordering premise only; historical response alignment is not "
            "a current-Query utility certificate or an unseen-Target Capability"
        ),
    }


def run_semantic_auxiliary_weather_llm_pilot(root: Path) -> dict[str, object]:
    """Execute one frozen LLM typed plan on a sealed natural Weather panel."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )

    from zipfile import ZipFile

    public_roles = ("rain", "mintemp", "maxtemp", "solar")
    target_roles = ("mintemp", "solar")
    anchors = (240, 300, 360, 420, 480, 540)
    period = 7
    train_stop = 928
    future = (928, 976)
    auxiliary_weight = 0.25
    archive_path = (
        Path(r"\\wsl.localhost\Ubuntu\tmp\weather_dataset.zip")
        if os.name == "nt"
        else Path("/tmp/weather_dataset.zip")
    )
    member = "weather_dataset.tsf"
    plan_path = root / SEMANTIC_AUXILIARY_WEATHER_PLAN_PATH
    plan = _read_object(plan_path)
    if (
        plan.get("planner_id") != "llm_semantic_probe_v1"
        or plan.get("outcome_exposure_at_plan_time") != "SEALED"
        or plan.get("program") != "ADD_AUXILIARY_SEMANTIC_GROUP"
        or float(plan.get("fixed_weight", -1.0)) != auxiliary_weight
        or plan.get("fallback") != "IDENTITY"
        or plan.get("confirmation") != "support_gain>0"
    ):
        raise ValueError("Weather LLM plan does not match the frozen typed contract")
    planned_targets = {
        str(row["target"]): row for row in plan.get("targets", [])
    }
    if set(planned_targets) != set(target_roles):
        raise ValueError("Weather LLM plan target supply changed")
    for target_role in target_roles:
        probes = planned_targets[target_role].get("probe_order")
        if (
            not isinstance(probes, list)
            or len(probes) != 2
            or any(float(row.get("weight", -1.0)) != auxiliary_weight for row in probes)
            or any(str(row.get("auxiliary_group")) == target_role for row in probes)
            or len({str(row.get("auxiliary_group")) for row in probes}) != 2
            or not {
                str(row.get("auxiliary_group")) for row in probes
            }.issubset(set(public_roles) - {target_role})
        ):
            raise ValueError(f"invalid frozen Weather probe plan: {target_role}")

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    eligible: dict[str, list[dict[str, object]]] = {
        role: [] for role in public_roles
    }
    rows_seen = {role: 0 for role in public_roles}
    rejected = {role: 0 for role in public_roles}
    data_started = False
    with ZipFile(archive_path) as archive, archive.open(member) as stream:
        for raw_line in stream:
            line = raw_line.decode("utf-8").strip()
            if not data_started:
                data_started = line.lower() == "@data"
                continue
            if not line:
                continue
            fields = line.split(":", 2)
            if len(fields) != 3:
                raise ValueError("unexpected Weather TSF row geometry")
            series_name, role, payload = fields
            if role not in eligible or len(eligible[role]) >= 20:
                continue
            rows_seen[role] += 1
            values = np.fromstring(payload, dtype=np.float64, count=future[1], sep=",")
            if values.shape != (future[1],) or not np.isfinite(values).all():
                rejected[role] += 1
                continue
            context_valid = True
            for anchor in anchors:
                _, _, method = _center_scale(
                    np, values[anchor - CONTEXT_LENGTH : anchor]
                )
                if method == "scale_floor_fallback":
                    context_valid = False
                    break
            _, _, evaluation_method = _center_scale(
                np, values[train_stop - CONTEXT_LENGTH : train_stop]
            )
            if evaluation_method == "scale_floor_fallback":
                context_valid = False
            try:
                scale_value = seasonal_scale(
                    values[:train_stop],
                    np.isfinite(values[:train_stop]),
                    period=period,
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError):
                context_valid = False
                scale_value = float("nan")
            if (
                not context_valid
                or not np.isfinite(scale_value)
                or float(scale_value) <= 0.0
            ):
                rejected[role] += 1
                continue
            eligible[role].append(
                {
                    "series_name": series_name,
                    "values": values,
                    "seasonal_scale": float(scale_value),
                }
            )
            if all(len(eligible[name]) >= 20 for name in public_roles):
                break
    if any(len(eligible[role]) != 20 for role in public_roles):
        raise ValueError(
            "Weather feasibility failed before outcome: "
            + ",".join(f"{role}={len(eligible[role])}" for role in public_roles)
        )

    reference_solve_count = 0
    candidate_solve_count = 0
    target_results: list[dict[str, object]] = []
    for target_role in target_roles:
        train_rows = eligible[target_role][:12]
        support_rows = eligible[target_role][12:16]
        query_rows = eligible[target_role][16:20]
        x_target: list[Any] = []
        y_target: list[Any] = []
        for anchor in anchors:
            for row in train_rows:
                raw = np.asarray(row["values"], dtype=np.float64)
                context = raw[anchor - CONTEXT_LENGTH : anchor]
                target = raw[anchor : anchor + HORIZON]
                center, scale, method = _center_scale(np, context)
                if method == "scale_floor_fallback":
                    raise AssertionError("frozen Weather target roster became invalid")
                x_target.append(
                    np.concatenate(
                        ((context - center) / scale, np.zeros(CONTEXT_LENGTH))
                    )
                )
                y_target.append((target - center) / scale)
        x_target_array = np.asarray(x_target, dtype=np.float64)
        y_target_array = np.asarray(y_target, dtype=np.float64)

        eval_rows = support_rows + query_rows
        x_eval: list[Any] = []
        actual: list[Any] = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal: list[float] = []
        for row in eval_rows:
            raw = np.asarray(row["values"], dtype=np.float64)
            context = raw[train_stop - CONTEXT_LENGTH : train_stop]
            center, scale, method = _center_scale(np, context)
            if method == "scale_floor_fallback":
                raise AssertionError("frozen Weather evaluation roster became invalid")
            x_eval.append(
                np.concatenate(
                    ((context - center) / scale, np.zeros(CONTEXT_LENGTH))
                )
            )
            actual.append(raw[slice(*future)])
            centers.append(center)
            scales.append(scale)
            seasonal.append(float(row["seasonal_scale"]))
        x_eval_array = np.asarray(x_eval, dtype=np.float64)
        actual_array = np.asarray(actual, dtype=np.float64)
        centers_array = np.asarray(centers, dtype=np.float64)
        scales_array = np.asarray(scales, dtype=np.float64)

        def score_predictions(normalized: Any) -> Any:
            prediction = np.asarray(normalized, dtype=np.float64)
            original = prediction * scales_array[:, None] + centers_array[:, None]
            return np.asarray(
                [
                    smase(
                        actual_array[index],
                        original[index],
                        scale=seasonal[index],
                    )
                    for index in range(len(eval_rows))
                ],
                dtype=np.float64,
            )

        baseline_prediction = _exact_weighted_ridge_prediction(
            np,
            x_train=x_target_array,
            targets=y_target_array,
            weights=np.ones(x_target_array.shape[0], dtype=np.float64),
            x_eval=x_eval_array,
        )
        reference_solve_count += 1
        baseline_losses = score_predictions(baseline_prediction)
        actions: dict[str, dict[str, float]] = {}
        candidate_rows: list[dict[str, object]] = []
        for auxiliary_role in public_roles:
            if auxiliary_role == target_role:
                continue
            x_auxiliary: list[Any] = []
            y_auxiliary: list[Any] = []
            for anchor in anchors:
                for row in eligible[auxiliary_role][:12]:
                    raw = np.asarray(row["values"], dtype=np.float64)
                    context = raw[anchor - CONTEXT_LENGTH : anchor]
                    target = raw[anchor : anchor + HORIZON]
                    center, scale, method = _center_scale(np, context)
                    if method == "scale_floor_fallback":
                        raise AssertionError(
                            "frozen Weather auxiliary roster became invalid"
                        )
                    x_auxiliary.append(
                        np.concatenate(
                            ((context - center) / scale, np.zeros(CONTEXT_LENGTH))
                        )
                    )
                    y_auxiliary.append((target - center) / scale)
            x_auxiliary_array = np.asarray(x_auxiliary, dtype=np.float64)
            y_auxiliary_array = np.asarray(y_auxiliary, dtype=np.float64)
            combined_x = np.vstack((x_target_array, x_auxiliary_array))
            combined_y = np.vstack((y_target_array, y_auxiliary_array))
            weights = np.concatenate(
                (
                    np.ones(x_target_array.shape[0], dtype=np.float64),
                    np.full(
                        x_auxiliary_array.shape[0],
                        auxiliary_weight,
                        dtype=np.float64,
                    ),
                )
            )
            prediction = _exact_weighted_ridge_prediction(
                np,
                x_train=combined_x,
                targets=combined_y,
                weights=weights,
                x_eval=x_eval_array,
            )
            candidate_solve_count += 1
            gains = baseline_losses - score_predictions(prediction)
            action = {
                "support_gain": float(np.mean(gains[:4])),
                "query_gain": float(np.mean(gains[4:])),
            }
            actions[auxiliary_role] = action
            candidate_rows.append(
                {
                    "auxiliary_group": auxiliary_role,
                    "weight": auxiliary_weight,
                    "support_gain": action["support_gain"],
                    "query_gain": action["query_gain"],
                    "per_support_series_gain": [float(value) for value in gains[:4]],
                    "per_query_series_gain": [float(value) for value in gains[4:]],
                }
            )

        a3_order = [role for role in public_roles if role != target_role]
        a5_order = [
            str(row["auxiliary_group"])
            for row in planned_targets[target_role]["probe_order"]
        ]
        a3_curve = _semantic_auxiliary_budget_curve(a3_order, actions)
        a5_curve = _semantic_auxiliary_budget_curve(a5_order, actions)
        a3_auc = policy_episode_adapt_auc(a3_curve)
        a5_auc = policy_episode_adapt_auc(a5_curve)
        best_candidate = max(
            candidate_rows,
            key=lambda row: (
                float(row["query_gain"]),
                str(row["auxiliary_group"]),
            ),
        )
        oracle_identity = float(best_candidate["query_gain"]) <= 0.0
        behavior_changed = [
            row["selected_auxiliary_group"] for row in a5_curve
        ] != [row["selected_auxiliary_group"] for row in a3_curve]
        order_changed = a5_order != a3_order[:2]
        target_results.append(
            {
                "target": target_role,
                "roster_counts": {"train": 12, "support": 4, "query": 4},
                "role_level_training_pool_augmentation": True,
                "entity_matching_claimed": False,
                "candidate_exact_gains": candidate_rows,
                "A3_order_public_metadata": a3_order,
                "A5_order_frozen_llm_plan": a5_order,
                "A3_curve": a3_curve,
                "A5_curve": a5_curve,
                "adapt_auc": {"A3": a3_auc, "A5": a5_auc},
                "A5_minus_A3": a5_auc - a3_auc,
                "A5_harmful": any(
                    float(row["fixed_query_gain"]) < 0.0 for row in a5_curve[1:]
                ),
                "A5_abstained_at_B2": bool(a5_curve[-1]["abstained"]),
                "top1_or_order_changed": order_changed,
                "selected_behavior_changed": behavior_changed,
                "order_or_behavior_changed": order_changed or behavior_changed,
                "full_menu_query_oracle_descriptive": {
                    "selected_action": (
                        "IDENTITY"
                        if oracle_identity
                        else "ADD_AUXILIARY_SEMANTIC_GROUP"
                    ),
                    "auxiliary_group": (
                        None if oracle_identity else best_candidate["auxiliary_group"]
                    ),
                    "weight": None if oracle_identity else auxiliary_weight,
                    "query_headroom": (
                        0.0
                        if oracle_identity
                        else float(best_candidate["query_gain"])
                    ),
                    "evaluator_only": True,
                },
            }
        )

    a5_macro = statistics.fmean(
        float(row["adapt_auc"]["A5"]) for row in target_results
    )
    a3_macro = statistics.fmean(
        float(row["adapt_auc"]["A3"]) for row in target_results
    )
    nonnegative_target_count = sum(
        float(row["A5_minus_A3"]) >= 0.0 for row in target_results
    )
    harmful_target_count = sum(bool(row["A5_harmful"]) for row in target_results)
    changed_target_count = sum(
        bool(row["order_or_behavior_changed"]) for row in target_results
    )
    passed = (
        a5_macro > a3_macro
        and nonnegative_target_count == 2
        and harmful_target_count == 0
        and changed_target_count >= 1
    )
    if (
        len(target_results) != 2
        or reference_solve_count != 2
        or candidate_solve_count != 6
        or any(
            [int(row["budget"]) for row in result["A3_curve"]] != [0, 1, 2]
            or [int(row["budget"]) for row in result["A5_curve"]] != [0, 1, 2]
            for result in target_results
        )
    ):
        raise AssertionError("Weather LLM pilot smoke check failed")

    return {
        "experiment_id": "E2-semantic-auxiliary-weather-LLM-plan-pilot",
        "scientific_role": "sealed functional pilot on one new natural dataset",
        "dataset": "Monash Weather daily panel",
        "plan": {
            "path": SEMANTIC_AUXILIARY_WEATHER_PLAN_PATH,
            "planner_id": plan["planner_id"],
            "planner_runtime": plan.get("planner_runtime"),
            "LLM_plan_frozen_before_outcome": True,
            "plan_rewritten_by_runner": False,
        },
        "protocol": {
            "public_roles": list(public_roles),
            "targets": list(target_roles),
            "period": period,
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "anchors": list(anchors),
            "train_stop": train_stop,
            "future_half_open": list(future),
            "auxiliary_weight": auxiliary_weight,
            "consumer": "Ridge(alpha=1.0, unpenalized intercept)",
            "metric": "per-series sMASE; four-series Support and Query means",
            "A3": "public role order excluding target; no Source Memory",
            "A5": "exact two-probe order from frozen LLM typed plan",
            "control": "stop on first strictly positive exact Support gain; otherwise continue then abstain",
            "feedback_budgets": [0, 1, 2],
            "query_used_for_decision": False,
            "original_uci_opened": False,
        },
        "eligibility": {
            "rule": (
                "public row order; length and finite through 976; all frozen "
                "contexts scale-valid; pre-future history seasonal-scale-valid"
            ),
            "eligible_count_by_role": {
                role: len(eligible[role]) for role in public_roles
            },
            "rows_seen_until_first_20_eligible": rows_seen,
            "rejected_before_first_20_eligible": rejected,
        },
        "targets": target_results,
        "macro": {
            "adapt_auc": {"A3": a3_macro, "A5": a5_macro},
            "A5_minus_A3": a5_macro - a3_macro,
            "nonnegative_target_count": nonnegative_target_count,
            "harmful_A5_target_count": harmful_target_count,
            "changed_order_or_behavior_target_count": changed_target_count,
            "A5_B2_abstain_target_count": sum(
                bool(row["A5_abstained_at_B2"]) for row in target_results
            ),
        },
        "compute_counts": {
            "ridge_reference_solve_count": reference_solve_count,
            "exact_weighted_candidate_solve_count": candidate_solve_count,
            "total_ridge_solve_count": reference_solve_count
            + candidate_solve_count,
            "per_action_consumer_refit_count": 0,
            "query_outcome_logical_materialization_count": 1,
        },
        "gate": {
            "A5_macro_strictly_greater_than_A3": a5_macro > a3_macro,
            "A5_minus_A3_nonnegative_2_of_2": nonnegative_target_count == 2,
            "A5_harmful_target_count_zero": harmful_target_count == 0,
            "A5_changes_top1_order_or_behavior": changed_target_count >= 1,
            "passed": passed,
        },
        "verdict": (
            "LLM_SEMANTIC_WORKFLOW_PILOT_PASS"
            if passed
            else "LLM_SEMANTIC_WORKFLOW_PILOT_FAIL"
        ),
        "capability_or_memory_written": False,
        "claim_limit": (
            "single new natural dataset with two target roles; functional LLM-plan "
            "pilot only, not cross-dataset Promotion or a general LLM success claim"
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "p0",
            "p1",
            "p1b",
            "workflow",
            "semantic-aux-p0",
            "semantic-aux-p1",
            "semantic-aux-weather",
            "aux-channel-bind-p0",
            "aux-channel-bind-llm-pilot",
            "aux-channel-bind-history-p1",
            "multiskill-fast-path",
            "multiskill-live-fast-path",
            "forecasting-two-skill-live-fast-path",
            "forecasting-two-skill-compile-replay",
            "historical-policy-llm-slow-path",
            "skill-acquisition-framework-replay",
            "natural-delayed-feedback-vertical",
            "workflow-discovery-acquisition-replay",
            "failure-driven-skill-evolution",
            "second-failure-mechanism-replay",
            "rejection-aware-fast-path-replay",
            "natural-imputation-cold-start",
            "natural-imputation-target-pilot",
            "natural-imputation-air-quality-target-pilot",
            "natural-imputation-scope-repair",
            "natural-imputation-prsa-target-pilot",
            "natural-imputation-prsa-actionable-target-pilot",
            "natural-imputation-pseudo-gap-observation",
            "natural-imputation-pseudo-gap-heldout-role-confirmation",
            "reversible-target-representation-p0",
            "reversible-target-representation-extension",
            "reversible-target-representation-llm-patch",
            "reversible-target-representation-llm-replay",
            "reversible-target-representation-extension-replay",
            "reversible-target-representation-llm-revision",
            "reversible-target-representation-llm-revision-replay",
            "noaa-multichannel-repair-p0",
            "noaa-multichannel-repair-2025",
            "missing-window-weighting-p0",
            "missing-window-weighting-p1",
            "missing-window-weighting-p2",
            "missing-window-weighting-prsa-target",
            "missing-window-weighting-prsa-risk",
            "missing-window-weighting-origin-coverage",
            "missing-window-weighting-air-quality",
        ),
        default="p0",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--base-url", default="https://api.agicto.cn/v1")
    args = parser.parse_args()
    if args.phase == "p0":
        report = run(root)
        default_path = DEFAULT_REPORT_PATH
    elif args.phase == "p1":
        report = run_p1(root)
        default_path = P1_REPORT_PATH
    elif args.phase == "p1b":
        report = run_p1b(root)
        default_path = P1B_REPORT_PATH
    elif args.phase == "workflow":
        report = run_workflow_replay(root)
        default_path = WORKFLOW_REPLAY_REPORT_PATH
    elif args.phase == "semantic-aux-p0":
        report = run_semantic_auxiliary_p0(root)
        default_path = SEMANTIC_AUXILIARY_P0_REPORT_PATH
    elif args.phase == "semantic-aux-p1":
        report = run_semantic_auxiliary_p1(root)
        default_path = SEMANTIC_AUXILIARY_P1_REPORT_PATH
    elif args.phase == "semantic-aux-weather":
        report = run_semantic_auxiliary_weather_llm_pilot(root)
        default_path = SEMANTIC_AUXILIARY_WEATHER_REPORT_PATH
    elif args.phase == "aux-channel-bind-p0":
        report = run_auxiliary_channel_binding_p0(root)
        default_path = AUXILIARY_CHANNEL_BINDING_P0_REPORT_PATH
    elif args.phase == "aux-channel-bind-llm-pilot":
        report = run_auxiliary_channel_binding_llm_pilot(root)
        default_path = AUXILIARY_CHANNEL_BINDING_LLM_REPORT_PATH
    elif args.phase == "aux-channel-bind-history-p1":
        report = run_auxiliary_channel_binding_history_p1(root)
        default_path = AUXILIARY_CHANNEL_BINDING_HISTORY_P1_REPORT_PATH
    elif args.phase == "multiskill-fast-path":
        report = compile_multiskill_llm_fast_path(root)
        default_path = MULTISKILL_LLM_FAST_PATH_REPORT_PATH
    elif args.phase == "multiskill-live-fast-path":
        report = run_live_multiskill_llm_fast_path(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = MULTISKILL_LIVE_LLM_FAST_PATH_REPORT_PATH
    elif args.phase == "forecasting-two-skill-live-fast-path":
        report = run_live_forecasting_two_skill_fast_path(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = FORECASTING_TWO_SKILL_LIVE_REPORT_PATH
    elif args.phase == "forecasting-two-skill-compile-replay":
        report = run_forecasting_two_skill_compile_replay(root)
        default_path = FORECASTING_TWO_SKILL_COMPILED_REPORT_PATH
    elif args.phase == "historical-policy-llm-slow-path":
        report = run_live_historical_policy_llm_slow_path(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = HISTORICAL_POLICY_LLM_SLOW_PATH_REPORT_PATH
    elif args.phase == "skill-acquisition-framework-replay":
        report = run_skill_acquisition_framework_replay(root)
        default_path = SKILL_ACQUISITION_FRAMEWORK_REPLAY_REPORT_PATH
    elif args.phase == "natural-delayed-feedback-vertical":
        report = run_natural_delayed_feedback_vertical(root)
        default_path = NATURAL_DELAYED_FEEDBACK_VERTICAL_REPORT_PATH
    elif args.phase == "workflow-discovery-acquisition-replay":
        report = run_live_workflow_discovery_acquisition_replay(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = WORKFLOW_DISCOVERY_ACQUISITION_REPORT_PATH
    elif args.phase == "failure-driven-skill-evolution":
        report = run_live_failure_driven_skill_evolution(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = FAILURE_DRIVEN_SKILL_EVOLUTION_REPORT_PATH
    elif args.phase == "second-failure-mechanism-replay":
        report = run_second_failure_mechanism_framework_replay(root)
        default_path = SECOND_FAILURE_MECHANISM_REPORT_PATH
    elif args.phase == "rejection-aware-fast-path-replay":
        report = run_rejection_aware_fast_path_replay(root)
        default_path = REJECTION_AWARE_FAST_PATH_REPORT_PATH
    elif args.phase == "natural-imputation-cold-start":
        report = run_live_natural_imputation_cold_start(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = NATURAL_IMPUTATION_COLD_START_REPORT_PATH
    elif args.phase == "natural-imputation-target-pilot":
        report = run_natural_imputation_target_pilot(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = NATURAL_IMPUTATION_TARGET_PILOT_REPORT_PATH
    elif args.phase == "natural-imputation-air-quality-target-pilot":
        report = run_natural_imputation_air_quality_target_pilot(root)
        default_path = NATURAL_IMPUTATION_AIR_QUALITY_TARGET_REPORT_PATH
    elif args.phase == "natural-imputation-scope-repair":
        report = run_live_natural_imputation_scope_repair(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = NATURAL_IMPUTATION_SCOPE_REPAIR_REPORT_PATH
    elif args.phase == "natural-imputation-prsa-target-pilot":
        report = run_natural_imputation_prsa_target_pilot(root)
        default_path = NATURAL_IMPUTATION_PRSA_TARGET_REPORT_PATH
    elif args.phase == "natural-imputation-prsa-actionable-target-pilot":
        report = run_natural_imputation_prsa_target_pilot(
            root,
            channels=("CO", "NO2"),
            experiment_id="E2.82-natural-imputation-PRSA-actionable-Target-pilot",
        )
        default_path = NATURAL_IMPUTATION_PRSA_ACTIONABLE_TARGET_REPORT_PATH
    elif args.phase == "natural-imputation-pseudo-gap-observation":
        report = run_live_natural_imputation_pseudo_gap_observation(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = NATURAL_IMPUTATION_PSEUDO_GAP_REPORT_PATH
    elif args.phase == "natural-imputation-pseudo-gap-heldout-role-confirmation":
        report = run_natural_imputation_pseudo_gap_heldout_role_confirmation(root)
        default_path = NATURAL_IMPUTATION_PSEUDO_GAP_HELDOUT_ROLE_REPORT_PATH
    elif args.phase == "reversible-target-representation-p0":
        report = run_reversible_target_representation_p0(root)
        default_path = REVERSIBLE_TARGET_REPRESENTATION_P0_REPORT_PATH
    elif args.phase == "reversible-target-representation-extension":
        report = run_reversible_target_representation_p0(
            root,
            specs_override=REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_SPECS,
            experiment_id=(
                "E2-natural-reversible-target-representation-source-extension"
            ),
        )
        default_path = REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPORT_PATH
    elif args.phase == "reversible-target-representation-llm-patch":
        report = run_reversible_representation_llm_patch_proposal(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = REVERSIBLE_TARGET_REPRESENTATION_LLM_REPORT_PATH
    elif args.phase == "reversible-target-representation-llm-replay":
        report = run_reversible_representation_llm_patch_replay(root)
        default_path = REVERSIBLE_TARGET_REPRESENTATION_LLM_REPLAY_REPORT_PATH
    elif args.phase == "reversible-target-representation-extension-replay":
        report = run_reversible_representation_llm_patch_replay(
            root,
            source_report_path=REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPORT_PATH,
            experiment_id=(
                "E2-live-LLM-representation-patch-post-proposal-cohort-replay"
            ),
        )
        default_path = REVERSIBLE_TARGET_REPRESENTATION_EXTENSION_REPLAY_REPORT_PATH
    elif args.phase == "reversible-target-representation-llm-revision":
        report = run_reversible_representation_llm_revision_proposal(
            root,
            model=args.model,
            base_url=args.base_url,
        )
        default_path = REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPORT_PATH
    elif args.phase == "reversible-target-representation-llm-revision-replay":
        report = run_reversible_representation_llm_revision_replay(root)
        default_path = (
            REVERSIBLE_TARGET_REPRESENTATION_LLM_REVISION_REPLAY_REPORT_PATH
        )
    elif args.phase == "noaa-multichannel-repair-p0":
        report = run_noaa_multichannel_local_repair_p0(root)
        default_path = NOAA_MULTICHANNEL_REPAIR_P0_REPORT_PATH
    elif args.phase == "noaa-multichannel-repair-2025":
        report = run_noaa_multichannel_local_repair_p0(
            root,
            year=2025,
            station_ids=NOAA_DEWPOINT_FEASIBILITY_STATIONS,
        )
        default_path = NOAA_MULTICHANNEL_REPAIR_2025_REPORT_PATH
    elif args.phase == "missing-window-weighting-p0":
        report = run_natural_missing_window_weighting_p0(root)
        default_path = MISSING_WINDOW_WEIGHTING_P0_REPORT_PATH
    elif args.phase == "missing-window-weighting-p1":
        report = run_natural_missing_window_weighting_p1(root)
        default_path = MISSING_WINDOW_WEIGHTING_P1_REPORT_PATH
    elif args.phase == "missing-window-weighting-p2":
        report = run_natural_missing_window_weighting_p2(root)
        default_path = MISSING_WINDOW_WEIGHTING_P2_REPORT_PATH
    elif args.phase == "missing-window-weighting-prsa-target":
        report = run_natural_missing_window_weighting_prsa_target(root)
        default_path = MISSING_WINDOW_WEIGHTING_PRSA_REPORT_PATH
    elif args.phase == "missing-window-weighting-prsa-risk":
        report = run_natural_missing_window_weighting_prsa_risk_replay(root)
        default_path = MISSING_WINDOW_WEIGHTING_PRSA_RISK_REPORT_PATH
    elif args.phase == "missing-window-weighting-air-quality":
        report = run_natural_missing_window_weighting_air_quality(root)
        default_path = MISSING_WINDOW_WEIGHTING_AIR_QUALITY_REPORT_PATH
    else:
        report = run_natural_missing_window_weighting_origin_coverage(root)
        default_path = MISSING_WINDOW_WEIGHTING_ORIGIN_COVERAGE_REPORT_PATH
    output = args.output or root / default_path
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
