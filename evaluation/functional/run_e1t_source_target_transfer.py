"""Run the minimum Source-to-Target equal-feedback-budget transfer pilot.

This runner consumes the checked-in E1-P report as fixed source evidence and never
re-runs its source cases.  The target is a controlled cross-generator pilot composed of
triangle-seasonal, seasonal-AR, and pulse-train processes.  Every family includes fixed
stable and contraindication variants without reusing the E1-P sinusoidal equations.  It
is not a natural-dataset or cross-domain confirmation experiment.

The adaptation plans for every frozen query case are produced before query outcomes are
judged.  Selection sees only corrupt-context global/local features plus the complete
three-program outcomes of the revealed support prefix.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    CONTEXT_LENGTH,
    FUTURE_LENGTH,
    GLOBAL_FEATURE_NAMES,
    LOCAL_EXTRA_FEATURE_NAMES,
    PROGRAM_IDS,
    _execute_program,
    _global_features,
    _local_features,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


BUDGETS = (0, 1, 2, 4, 8)
TARGET_FAMILIES = (
    "triangle_seasonal",
    "seasonal_ar",
    "pulse_train_regime",
)
SUPPORT_SPECS = (
    ("triangle_seasonal", "stable", 6101),
    ("seasonal_ar", "stable", 6201),
    ("pulse_train_regime", "stable", 6301),
    ("triangle_seasonal", "phase_slip", 6102),
    ("seasonal_ar", "lag_break", 6202),
    ("pulse_train_regime", "irregular_regime", 6302),
    ("triangle_seasonal", "stable", 6103),
    ("seasonal_ar", "stable", 6203),
)
QUERY_SPECS = tuple(
    (family, variant, seed)
    for family in TARGET_FAMILIES
    for variant, seed in (
        ("stable", 7101),
        ("stable", 7102),
        (
            {
                "triangle_seasonal": "phase_slip",
                "seasonal_ar": "lag_break",
                "pulse_train_regime": "irregular_regime",
            }[family],
            7103,
        ),
        (
            {
                "triangle_seasonal": "phase_slip",
                "seasonal_ar": "lag_break",
                "pulse_train_regime": "irregular_regime",
            }[family],
            7104,
        ),
    )
)
ARMS = ("a3_target_only", "a4_source_only", "a5_source_plus_target")
HARM_MARGIN = 0.005
NEAREST_EVIDENCE_K = 5
TOTAL_LENGTH = CONTEXT_LENGTH + FUTURE_LENGTH


@dataclass(frozen=True)
class TargetCase:
    case_id: str
    split: str
    family: str
    variant: str
    seed: int
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray


def _triangle_seasonal(seed: int, variant: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(TOTAL_LENGTH, dtype=np.float64)
    period = (20, 24, 28)[seed % 3]
    phase = (t / period + 0.07 * (seed % 5)) % 1.0
    if variant == "phase_slip":
        phase = (phase + np.where(t >= 168.0, 0.33, 0.0)) % 1.0
    elif variant != "stable":
        raise ValueError(f"unknown triangle variant: {variant!r}")
    triangle = 1.0 - 4.0 * np.abs(phase - 0.5)
    values = 0.0012 * t + triangle
    return values + rng.normal(0.0, 0.022, size=TOTAL_LENGTH)


def _seasonal_ar(seed: int, variant: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    period = (18, 22, 26)[seed % 3]
    innovations = rng.normal(0.0, 0.055, size=TOTAL_LENGTH)
    values = np.empty(TOTAL_LENGTH, dtype=np.float64)
    values[:period] = rng.normal(0.0, 0.65, size=period)
    for index in range(period, TOTAL_LENGTH):
        active_lag = period
        if variant == "lag_break" and index >= 166:
            active_lag = max(8, period - 5)
        elif variant != "stable" and variant != "lag_break":
            raise ValueError(f"unknown seasonal-AR variant: {variant!r}")
        values[index] = 0.94 * values[index - active_lag] + innovations[index]
    values += 0.001 * np.arange(TOTAL_LENGTH, dtype=np.float64)
    return values


def _pulse_train_regime(seed: int, variant: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(TOTAL_LENGTH, dtype=np.float64)
    period = (21, 25, 29)[seed % 3]
    width = 5 + seed % 4
    phase = (t + seed % 7) % period
    values = np.where(phase < width, 1.1, -0.25)
    if variant == "irregular_regime":
        changed_period = max(12, period - 7)
        changed_phase = (t + 3 + seed % 5) % changed_period
        values = np.where(
            t >= 168.0,
            np.where(changed_phase < width + 3, 0.65, -0.70),
            values,
        )
    elif variant != "stable":
        raise ValueError(f"unknown pulse-train variant: {variant!r}")
    values += 0.0008 * t
    return values + rng.normal(0.0, 0.018, size=TOTAL_LENGTH)


def _clean_target_series(family: str, variant: str, seed: int) -> np.ndarray:
    if family == "triangle_seasonal":
        values = _triangle_seasonal(seed, variant)
    elif family == "seasonal_ar":
        values = _seasonal_ar(seed, variant)
    elif family == "pulse_train_regime":
        values = _pulse_train_regime(seed, variant)
    else:
        raise ValueError(f"unknown E1-T target family: {family!r}")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (TOTAL_LENGTH,) or not np.isfinite(values).all():
        raise AssertionError("target generator produced an invalid clean trajectory")
    return values


def build_target_cases() -> tuple[TargetCase, ...]:
    cases: list[TargetCase] = []
    for split, specs in (("support", SUPPORT_SPECS), ("query", QUERY_SPECS)):
        for position, (family, variant, seed) in enumerate(specs):
            clean = _clean_target_series(family, variant, seed)
            corrupt = clean[:CONTEXT_LENGTH].copy()
            corrupt[156:180] = np.nan
            cases.append(
                TargetCase(
                    case_id=f"e1t-{split}-{position:02d}-{family}-{variant}-{seed}",
                    split=split,
                    family=family,
                    variant=variant,
                    seed=seed,
                    clean_context=clean[:CONTEXT_LENGTH].copy(),
                    corrupt_context=corrupt,
                    clean_future=clean[CONTEXT_LENGTH:].copy(),
                )
            )
    return tuple(cases)


def _public_descriptor(case: TargetCase) -> dict[str, object]:
    global_features, observed_period = _global_features(case.corrupt_context)
    local_features = _local_features(case.corrupt_context, observed_period)
    return {
        "case_id": case.case_id,
        "split": case.split,
        # Private report-only metadata. It is never read by _select_action.
        "target_family_report_only": case.family,
        "target_variant_report_only": case.variant,
        "seed_report_only": case.seed,
        "observed_period_parameter": observed_period,
        "global_features": global_features,
        "local_features": local_features,
    }


def _judge_case(
    case: TargetCase,
    descriptor: Mapping[str, object],
    valuator: FrozenChronosValuator,
) -> dict[str, object]:
    arms: dict[str, dict[str, object]] = {}
    for program_id in PROGRAM_IDS:
        prepared = _execute_program(
            program_id,
            case.corrupt_context,
            observed_period=int(descriptor["observed_period_parameter"]),
        )
        receipt = valuator.evaluate(
            prepared,
            case.clean_future,
            scale_context=case.clean_context,
        )
        arms[program_id] = {
            "loss_j": float(receipt.loss_j),
            "utility_u": float(receipt.utility_u),
            "status": str(receipt.status),
        }
    losses = {program: float(arms[program]["loss_j"]) for program in PROGRAM_IDS}
    winner = min(PROGRAM_IDS, key=lambda program: (losses[program], program))
    return {
        **descriptor,
        "arms": arms,
        "grader_winner": winner,
        "menu_oracle_loss": losses[winner],
    }


def _load_source_evidence(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "e1p-periodic-missing-pilot/1":
        raise ValueError("source report is not the frozen E1-P periodic-missing pilot")
    if not payload.get("all_pilot_gates_pass"):
        raise ValueError("source E1-P premise gates did not pass")
    source: list[dict[str, object]] = []
    for raw in payload.get("cases", []):
        if not isinstance(raw, Mapping):
            raise TypeError("source report contains a non-mapping case")
        global_features = raw.get("global_features")
        local_features = raw.get("local_features")
        arms = raw.get("arms")
        if not isinstance(global_features, Mapping) or not isinstance(local_features, Mapping):
            raise ValueError("source case is missing public feature evidence")
        if not isinstance(arms, Mapping) or any(program not in arms for program in PROGRAM_IDS):
            raise ValueError("source case is missing complete three-program outcomes")
        # Deliberately omit source archetype, cohort, seed, and grader winner. Selection
        # receives only public features and complete judge outcomes.
        source.append(
            {
                "global_features": {
                    name: float(global_features[name]) for name in GLOBAL_FEATURE_NAMES
                },
                "local_features": {
                    name: float(local_features[name]) for name in LOCAL_EXTRA_FEATURE_NAMES
                },
                "arms": {
                    program: {"loss_j": float(arms[program]["loss_j"])}
                    for program in PROGRAM_IDS
                },
            }
        )
    if not source:
        raise ValueError("source E1-P report contains no cases")
    metadata = {
        "path": str(path.resolve()),
        "schema_version": payload["schema_version"],
        "case_count": len(source),
        "all_pilot_gates_pass": True,
        "risk_corpus_coverage_status": payload.get("risk_corpus_coverage_status"),
    }
    return source, metadata


def _squash_feature(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"non-finite public feature {name!r}")
    if name in {
        "missing_fraction",
        "longest_missing_run_fraction",
        "estimated_region_start_fraction",
        "estimated_region_end_fraction",
        "period_reliability",
        "acf_spectral_consistency",
        "amplitude_stability",
        "local_period_consistency",
    }:
        return float(np.clip(value, -1.0, 1.0))
    if name in {"pre_period", "post_period"}:
        return float(np.clip(value / 48.0, 0.0, 1.5))
    if name == "observed_cycles":
        return float(np.clip(value / 8.0, 0.0, 2.0))
    if name in {"phase_correlation"}:
        return float(np.clip(value, -1.0, 1.0))
    if name == "amplitude_ratio":
        return float(np.clip(math.log(max(value, 1e-8)) / 3.0, -2.0, 2.0))
    return float(np.clip(np.arcsinh(value) / 3.0, -2.0, 2.0))


def _feature_vector(row: Mapping[str, object]) -> np.ndarray:
    global_features = row.get("global_features")
    local_features = row.get("local_features")
    if not isinstance(global_features, Mapping) or not isinstance(local_features, Mapping):
        raise TypeError("selection row lacks public feature mappings")
    values = [
        _squash_feature(name, float(global_features[name])) for name in GLOBAL_FEATURE_NAMES
    ]
    values.extend(
        _squash_feature(name, float(local_features[name]))
        for name in LOCAL_EXTRA_FEATURE_NAMES
    )
    return np.asarray(values, dtype=np.float64)


def _normalized_regret(row: Mapping[str, object], action: str) -> float:
    arms = row.get("arms")
    if not isinstance(arms, Mapping):
        raise TypeError("evidence row lacks revealed judge outcomes")
    losses = [float(arms[program]["loss_j"]) for program in PROGRAM_IDS]
    floor = min(losses)
    scale = max(max(losses) - floor, HARM_MARGIN)
    return (float(arms[action]["loss_j"]) - floor) / scale


def _selection_view(
    row: Mapping[str, object], *, include_revealed_outcomes: bool
) -> dict[str, object]:
    """Strip report/private metadata before a row crosses the selector boundary."""

    global_features = row.get("global_features")
    local_features = row.get("local_features")
    if not isinstance(global_features, Mapping) or not isinstance(local_features, Mapping):
        raise TypeError("selection view requires public feature mappings")
    view: dict[str, object] = {
        "global_features": dict(global_features),
        "local_features": dict(local_features),
    }
    if include_revealed_outcomes:
        arms = row.get("arms")
        if not isinstance(arms, Mapping):
            raise TypeError("revealed evidence view requires arm outcomes")
        view["arms"] = {
            program: {"loss_j": float(arms[program]["loss_j"])}
            for program in PROGRAM_IDS
        }
    return view


def _select_action(
    evidence: Sequence[Mapping[str, object]], query: Mapping[str, object]
) -> str:
    """Choose from public features and revealed evidence outcomes only."""

    if not evidence:
        return "identity"
    query_vector = _feature_vector(query)
    distances = [
        (float(np.linalg.norm(_feature_vector(row) - query_vector)), row)
        for row in evidence
    ]
    cutoff = sorted(distance for distance, _ in distances)[
        min(NEAREST_EVIDENCE_K, len(distances)) - 1
    ]
    # Include all exact boundary ties so source file order cannot act as a hidden label.
    nearest = [(distance, row) for distance, row in distances if distance <= cutoff + 1e-12]
    estimates: dict[str, float] = {}
    for action in PROGRAM_IDS:
        weighted = 0.0
        weight_sum = 0.0
        for distance, row in nearest:
            weight = 1.0 / (distance + 0.05)
            weighted += weight * _normalized_regret(row, action)
            weight_sum += weight
        estimates[action] = weighted / weight_sum
    return min(PROGRAM_IDS, key=lambda action: (estimates[action], action))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty value sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _group_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    groups = {"all": list(rows)}
    for family in TARGET_FAMILIES:
        groups[family] = [
            row for row in rows if row["target_family_report_only"] == family
        ]
    return groups


def _selection_summary(
    rows: Sequence[Mapping[str, object]], choices: Mapping[str, str]
) -> dict[str, object]:
    result: dict[str, object] = {}
    for group_name, group in _group_rows(rows).items():
        selected_losses = [
            float(row["arms"][choices[str(row["case_id"])]]["loss_j"]) for row in group
        ]
        oracle_losses = [float(row["menu_oracle_loss"]) for row in group]
        result[group_name] = {
            "n_query_cases": len(group),
            "mean_query_loss": _mean(selected_losses),
            "mean_adaptation_regret": _mean(
                [loss - oracle for loss, oracle in zip(selected_losses, oracle_losses)]
            ),
            "action_counts": {
                program: sum(choices[str(row["case_id"])] == program for row in group)
                for program in PROGRAM_IDS
            },
        }
    return result


def _fixed_action_from_evidence(evidence: Sequence[Mapping[str, object]]) -> str:
    mean_regret = {
        program: _mean([_normalized_regret(row, program) for row in evidence])
        for program in PROGRAM_IDS
    }
    return min(PROGRAM_IDS, key=lambda program: (mean_regret[program], program))


def _diagnostics(
    rows: Sequence[Mapping[str, object]], *, source_fixed_action: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for group_name, group in _group_rows(rows).items():
        fixed_losses = {
            program: _mean([float(row["arms"][program]["loss_j"]) for row in group])
            for program in PROGRAM_IDS
        }
        best_fixed = min(PROGRAM_IDS, key=lambda item: (fixed_losses[item], item))
        result[group_name] = {
            "identity_mean_loss": fixed_losses["identity"],
            "fixed_mean_loss_by_program": fixed_losses,
            "source_selected_fixed_action": source_fixed_action,
            "source_selected_fixed_mean_loss": fixed_losses[source_fixed_action],
            "posthoc_best_fixed_action": best_fixed,
            "posthoc_best_fixed_mean_loss": fixed_losses[best_fixed],
            "menu_oracle_mean_loss": _mean(
                [float(row["menu_oracle_loss"]) for row in group]
            ),
        }
    return result


def _curve_auc(values_by_budget: Mapping[int, float]) -> float:
    area = 0.0
    for left, right in zip(BUDGETS[:-1], BUDGETS[1:]):
        area += 0.5 * (values_by_budget[left] + values_by_budget[right]) * (right - left)
    return float(area / BUDGETS[-1])


def _harm_summary(
    rows: Sequence[Mapping[str, object]],
    left_choices: Mapping[str, str],
    right_choices: Mapping[str, str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for group_name, group in _group_rows(rows).items():
        deltas = []
        for row in group:
            case_id = str(row["case_id"])
            left_loss = float(row["arms"][left_choices[case_id]]["loss_j"])
            right_loss = float(row["arms"][right_choices[case_id]]["loss_j"])
            deltas.append(left_loss - right_loss)
        result[group_name] = {
            "mean_loss_delta": _mean(deltas),
            "harm_case_count_at_margin": sum(delta > HARM_MARGIN for delta in deltas),
            "harm_case_rate_at_margin": sum(delta > HARM_MARGIN for delta in deltas)
            / len(deltas),
            "margin": HARM_MARGIN,
        }
    return result


def run_e1t_source_target_transfer(
    valuator: FrozenChronosValuator,
    *,
    source_report: Path,
) -> dict[str, object]:
    source_evidence, source_metadata = _load_source_evidence(source_report)
    cases = build_target_cases()
    support_cases = [case for case in cases if case.split == "support"]
    query_cases = [case for case in cases if case.split == "query"]

    # Target support outcomes are the only newly revealed adaptation feedback.
    support_rows = [
        _judge_case(case, _public_descriptor(case), valuator) for case in support_cases
    ]
    query_public = [_public_descriptor(case) for case in query_cases]
    source_selection_evidence = [
        _selection_view(row, include_revealed_outcomes=True) for row in source_evidence
    ]
    target_selection_evidence = [
        _selection_view(row, include_revealed_outcomes=True) for row in support_rows
    ]
    query_selection_views = [
        _selection_view(row, include_revealed_outcomes=False) for row in query_public
    ]

    # Freeze every query action plan before any query case is passed to the judge.
    plans: dict[int, dict[str, dict[str, str]]] = {}
    for budget in BUDGETS:
        revealed_target = target_selection_evidence[:budget]
        plans[budget] = {arm: {} for arm in ARMS}
        for descriptor, query in zip(query_public, query_selection_views):
            case_id = str(descriptor["case_id"])
            plans[budget]["a3_target_only"][case_id] = _select_action(
                revealed_target, query
            )
            plans[budget]["a4_source_only"][case_id] = _select_action(
                source_selection_evidence, query
            )
            plans[budget]["a5_source_plus_target"][case_id] = _select_action(
                [*source_selection_evidence, *revealed_target], query
            )

    # Query outcomes are grader/report data only and cannot affect the frozen plans.
    query_rows = [
        _judge_case(case, descriptor, valuator)
        for case, descriptor in zip(query_cases, query_public)
    ]

    budget_results: dict[str, object] = {}
    for budget in BUDGETS:
        budget_results[str(budget)] = {
            "revealed_support_case_ids": [
                str(row["case_id"]) for row in support_rows[:budget]
            ],
            "arms": {
                arm: _selection_summary(query_rows, plans[budget][arm]) for arm in ARMS
            },
            "harm": {
                "a5_minus_a3": _harm_summary(
                    query_rows,
                    plans[budget]["a5_source_plus_target"],
                    plans[budget]["a3_target_only"],
                ),
                "a5_minus_a4": _harm_summary(
                    query_rows,
                    plans[budget]["a5_source_plus_target"],
                    plans[budget]["a4_source_only"],
                ),
            },
        }

    regret_auc: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for group_name in ("all", *TARGET_FAMILIES):
            curve = {
                budget: float(
                    budget_results[str(budget)]["arms"][arm][group_name][
                        "mean_adaptation_regret"
                    ]
                )
                for budget in BUDGETS
            }
            regret_auc[arm][group_name] = _curve_auc(curve)

    a3_auc = regret_auc["a3_target_only"]["all"]
    a5_auc = regret_auc["a5_source_plus_target"]["all"]
    target_diagnostics = _diagnostics(
        query_rows,
        source_fixed_action=_fixed_action_from_evidence(source_selection_evidence),
    )
    target_best_fixed_loss = float(
        target_diagnostics["all"]["posthoc_best_fixed_mean_loss"]
    )
    target_menu_oracle_loss = float(target_diagnostics["all"]["menu_oracle_mean_loss"])
    target_routing_headroom = target_best_fixed_loss - target_menu_oracle_loss
    a4_final_loss = float(
        budget_results[str(BUDGETS[-1])]["arms"]["a4_source_only"]["all"][
            "mean_query_loss"
        ]
    )
    a5_final_loss = float(
        budget_results[str(BUDGETS[-1])]["arms"]["a5_source_plus_target"]["all"][
            "mean_query_loss"
        ]
    )
    final_harm = budget_results[str(BUDGETS[-1])]["harm"]["a5_minus_a3"]["all"]
    gates = {
        "protocol_integrity": {
            "gate_status": "structural_pilot_gate",
            "support_prefix_matches_budget": True,
            "query_plans_frozen_before_query_judging": True,
            "query_outcome_visible_to_adaptation": False,
            "selection_feature_sources": ["corrupt_context", "missing_mask"],
            "selection_outcome_sources": [
                "fixed_source_e1p_case_outcomes",
                "revealed_target_support_prefix_outcomes",
            ],
            "pass": True,
        },
        "source_plus_target_regret_auc": {
            "gate_status": "cross_generator_pilot_gate_not_confirmation",
            "threshold": "A3 regret AUC - A5 regret AUC >= 0.005",
            "a3_target_only_regret_auc": a3_auc,
            "a5_source_plus_target_regret_auc": a5_auc,
            "a3_minus_a5_regret_auc": a3_auc - a5_auc,
            "minimum_improvement": HARM_MARGIN,
            "pass": a3_auc - a5_auc >= HARM_MARGIN,
        },
        "target_routing_headroom": {
            "gate_status": "cross_generator_pilot_gate_not_confirmation",
            "threshold": "best-fixed mean loss - menu-oracle mean loss >= 0.01",
            "best_fixed_mean_loss": target_best_fixed_loss,
            "menu_oracle_mean_loss": target_menu_oracle_loss,
            "best_fixed_minus_menu_oracle_mean_loss": target_routing_headroom,
            "minimum_headroom": 0.01,
            "pass": target_routing_headroom >= 0.01,
        },
        "final_budget_source_plus_target": {
            "gate_status": "cross_generator_pilot_gate_not_confirmation",
            "threshold": "A5 query loss at B=8 <= A4 query loss + 0.005",
            "a4_source_only_query_loss": a4_final_loss,
            "a5_source_plus_target_query_loss": a5_final_loss,
            "pass": a5_final_loss <= a4_final_loss + HARM_MARGIN,
        },
        "negative_transfer_control": {
            "gate_status": "cross_generator_pilot_gate_not_confirmation",
            "comparison": "A5 minus A3 at B=8; positive is source-induced harm",
            "mean_loss_delta_max": HARM_MARGIN,
            "harm_case_rate_max": 0.25,
            **final_harm,
            "pass": float(final_harm["mean_loss_delta"]) <= HARM_MARGIN
            and float(final_harm["harm_case_rate_at_margin"]) <= 0.25,
        },
    }

    return {
        "schema_version": "e1t-source-target-transfer-pilot/1",
        "scientific_role": "controlled_cross_generator_equal_feedback_budget_pilot",
        "source_evidence": source_metadata,
        "consumer": {
            "valuator": "FrozenChronosValuator",
            "model_id": str(valuator.manifest["model_id"]),
            "revision": str(valuator.manifest["revision"]),
            "device": str(valuator.manifest["device"]),
            "context_length": int(valuator.manifest["context_length"]),
            "prediction_length": int(valuator.manifest["prediction_length"]),
            "ingestion_policy_id": valuator.ingestion_policy_id,
        },
        "configuration": {
            "budgets": list(BUDGETS),
            "budget_unit": "one target support case complete three-program judge outcome",
            "target_families": list(TARGET_FAMILIES),
            "support_specs_in_fixed_reveal_order": [list(spec) for spec in SUPPORT_SPECS],
            "query_specs_frozen": [list(spec) for spec in QUERY_SPECS],
            "programs": list(PROGRAM_IDS),
            "selector": {
                "kind": "deterministic_nearest_evidence_normalized_regret",
                "nearest_k_with_all_boundary_ties": NEAREST_EVIDENCE_K,
                "feature_normalization": "fixed_featurewise_squashing_no_dataset_fit",
                "empty_target_evidence_fallback": "identity",
            },
            "agent_enabled": False,
            "memory_enabled": False,
            "promotion_enabled": False,
        },
        "information_wall": {
            "target_support_and_query_split_fixed_before_judging": True,
            "query_action_plans_frozen_before_query_judging": True,
            "query_outcomes_used_only_by_grader_and_report": True,
            "selection_does_not_receive": [
                "target_family",
                "target_variant",
                "source_archetype",
                "domain_label",
                "semantic_case_id",
                "clean_context",
                "clean_future",
                "query_outcome",
            ],
            "selection_receives": [
                "corrupt_context_global_features",
                "corrupt_context_gap_local_features",
                "revealed_evidence_three_program_outcomes",
            ],
        },
        "support_cases": support_rows,
        "query_cases": query_rows,
        "query_action_plans": {
            str(budget): plans[budget] for budget in BUDGETS
        },
        "diagnostics": target_diagnostics,
        "budget_results": budget_results,
        "adaptation_regret_auc": regret_auc,
        "gates": gates,
        "all_pilot_gates_pass": all(bool(gate["pass"]) for gate in gates.values()),
        "limitations": [
            "Target families are controlled synthetic generators, not natural datasets.",
            "This supports only a cross-generator pilot claim, not cross-domain transfer.",
            "The source corpus is E1-P periodic missingness and has PARTIAL risk coverage.",
            "Stable and contraindication target variants are fixed before judging and are report-only metadata.",
            "The deterministic nearest-evidence selector is a mechanism probe, not Memory.",
            "Query has four cases per target family; uncertainty estimates are descriptive.",
            "Generator definitions and thresholds are fixed before the formal judge run and must not be tuned to the outcome.",
        ],
        "claim_limit": (
            "This run can measure fixed-source reuse, target-feedback adaptation, and negative "
            "transfer across stable and contraindication variants of three controlled "
            "non-sinusoidal generator families only."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the controlled E1-T Source-to-Target transfer pilot."
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=project_root / "artifacts/functional/e1p/periodic_missing_report.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    valuator = FrozenChronosValuator()
    report = run_e1t_source_target_transfer(
        valuator,
        source_report=args.source_report.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={output}")
    print(f"all_pilot_gates_pass={report['all_pilot_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_target_cases", "run_e1t_source_target_transfer"]
