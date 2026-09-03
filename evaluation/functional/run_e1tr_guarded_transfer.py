"""Run the minimum guarded repair of the failed E1-T transfer selector.

E1-TR reuses the frozen E1-P source evidence and only the eight frozen E1-T support
outcomes.  It judges a new, predeclared Query once.  The repair is a deterministic
tri-state guard (supported / contradicted / unresolved), not model training, Memory,
promotion, or confirmation evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    CONTEXT_LENGTH,
    PROGRAM_IDS,
    _correlation,
    _primary_gap,
    build_pilot_cases,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e1t_source_target_transfer import (
    BUDGETS,
    TARGET_FAMILIES,
    TargetCase,
    build_target_cases,
    _clean_target_series,
    _curve_auc,
    _diagnostics,
    _feature_vector,
    _fixed_action_from_evidence,
    _judge_case,
    _load_source_evidence,
    _mean,
    _public_descriptor,
    _select_action as _naive_union_select_action,
    _selection_view,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


GAIN_MIN = 0.01
HARM_MARGIN = 0.005
MAX_HARM_SHARE = 0.25
MIN_SUPPORT_SHARE = 0.60
MIN_EVIDENCE = 3
NEAREST_K = 5
PHASE_BRIDGE_ERROR_MAX = 0.75
NON_IDENTITY_ACTIONS = ("linear", "seasonal")
REPAIR_ARMS = (
    "a3_target_only_guarded",
    "a4_source_only_guarded",
    "a5_guarded_overlay",
    "a5_no_bridge_guarded_ablation",
    "a5_naive_union_ablation",
)
REPAIR_QUERY_SPECS = tuple(
    (family, variant, seed)
    for family in TARGET_FAMILIES
    for variant, seed in (
        ("stable", 8101),
        ("stable", 8102),
        (
            {
                "triangle_seasonal": "phase_slip",
                "seasonal_ar": "lag_break",
                "pulse_train_regime": "irregular_regime",
            }[family],
            8103,
        ),
        (
            {
                "triangle_seasonal": "phase_slip",
                "seasonal_ar": "lag_break",
                "pulse_train_regime": "irregular_regime",
            }[family],
            8104,
        ),
    )
)


def build_repair_query_cases() -> tuple[TargetCase, ...]:
    cases: list[TargetCase] = []
    for position, (family, variant, seed) in enumerate(REPAIR_QUERY_SPECS):
        clean = _clean_target_series(family, variant, seed)
        corrupt = clean[:CONTEXT_LENGTH].copy()
        corrupt[156:180] = np.nan
        cases.append(
            TargetCase(
                case_id=f"e1tr-query-{position:02d}-{family}-{variant}-{seed}",
                split="query",
                family=family,
                variant=variant,
                seed=seed,
                clean_context=clean[:CONTEXT_LENGTH].copy(),
                corrupt_context=corrupt,
                clean_future=clean[CONTEXT_LENGTH:].copy(),
            )
        )
    return tuple(cases)


def _phase_bridge_observation(
    corrupt_context: np.ndarray, observed_period: int
) -> dict[str, object]:
    """Compare visible post-gap points with their nearest finite pre-gap phase peers."""

    values = np.asarray(corrupt_context, dtype=np.float64)
    start, end = _primary_gap(values)
    period = max(1, int(observed_period))
    right_visible = [
        index for index in range(end, values.size) if np.isfinite(values[index])
    ]
    right_values: list[float] = []
    reference_values: list[float] = []
    for index in right_visible:
        reference = index - period
        while reference >= start:
            reference -= period
        while reference >= 0 and not np.isfinite(values[reference]):
            reference -= period
        if reference >= 0:
            right_values.append(float(values[index]))
            reference_values.append(float(values[reference]))

    finite = values[np.isfinite(values)]
    center = float(np.median(finite))
    robust_scale = 1.4826 * float(np.median(np.abs(finite - center)))
    if robust_scale <= 1e-8:
        robust_scale = max(float(np.std(finite)), 1e-8)
    if right_values:
        right_array = np.asarray(right_values, dtype=np.float64)
        reference_array = np.asarray(reference_values, dtype=np.float64)
        bridge_error = float(np.median(np.abs(right_array - reference_array)) / robust_scale)
        bridge_correlation = _correlation(right_array, reference_array)
    else:
        bridge_error = 0.0
        bridge_correlation = 0.0
    coverage = len(right_values) / len(right_visible) if right_visible else 0.0
    return {
        "gap_interval": [start, end],
        "observed_period": period,
        "right_visible_point_count": len(right_visible),
        "matched_phase_pair_count": len(right_values),
        "coverage": float(coverage),
        "robust_normalized_median_absolute_error": bridge_error,
        "correlation": float(bridge_correlation),
    }


def _bridge_selection_view(
    row: Mapping[str, object], *, include_revealed_outcomes: bool
) -> dict[str, object]:
    view = _selection_view(
        row, include_revealed_outcomes=include_revealed_outcomes
    )
    bridge = row.get("phase_bridge")
    if not isinstance(bridge, Mapping):
        raise TypeError("bridge selection row is missing phase_bridge observation")
    view["phase_bridge"] = dict(bridge)
    return view


def _bridge_coverage_receipt(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    coverages = [float(row["phase_bridge"]["coverage"]) for row in rows]
    return {
        "case_count": len(coverages),
        "minimum": min(coverages),
        "maximum": max(coverages),
        "all_complete": all(value == 1.0 for value in coverages),
    }


def _enrich_frozen_source_evidence(
    source_rows: Sequence[Mapping[str, object]], source_report: Path
) -> list[dict[str, object]]:
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    raw_rows = payload.get("cases")
    generated = build_pilot_cases()
    if not isinstance(raw_rows, list) or len(raw_rows) != len(generated):
        raise ValueError("frozen E1-P cases do not match deterministic reconstruction")
    if len(source_rows) != len(generated):
        raise ValueError("loaded E1-P evidence count changed before bridge enrichment")
    enriched: list[dict[str, object]] = []
    for loaded, raw, case in zip(source_rows, raw_rows, generated):
        if str(raw["case_id"]) != case.case_id:
            raise ValueError("frozen E1-P case ID/order differs from reconstruction")
        row = dict(loaded)
        row["phase_bridge"] = _phase_bridge_observation(
            case.corrupt_context, int(raw["observed_period_parameter"])
        )
        enriched.append(row)
    return enriched


def _load_frozen_target_support(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "e1t-source-target-transfer-pilot/1":
        raise ValueError("target-support report is not the frozen E1-T pilot")
    raw_support = payload.get("support_cases")
    if not isinstance(raw_support, list) or len(raw_support) != 8:
        raise ValueError("frozen E1-T report must contain exactly eight support cases")
    reconstructed = [case for case in build_target_cases() if case.split == "support"]
    if len(reconstructed) != len(raw_support):
        raise ValueError("frozen E1-T support count differs from reconstruction")
    support: list[dict[str, object]] = []
    for raw, case in zip(raw_support, reconstructed):
        if str(raw["case_id"]) != case.case_id:
            raise ValueError("frozen E1-T support ID/order differs from reconstruction")
        enriched = dict(raw)
        enriched["phase_bridge"] = _phase_bridge_observation(
            case.corrupt_context, int(raw["observed_period_parameter"])
        )
        support.append(
            _bridge_selection_view(enriched, include_revealed_outcomes=True)
        )
    return support, {
        "path": str(path.resolve()),
        "schema_version": payload["schema_version"],
        "support_case_count": len(support),
        "old_query_case_count_ignored": len(payload.get("query_cases", [])),
        "old_query_outcomes_used": False,
        "old_all_pilot_gates_pass": bool(payload.get("all_pilot_gates_pass")),
        "deterministic_context_id_order_check": "PASS",
        "judge_outcomes_recomputed": False,
        "phase_bridge_coverage_receipt": _bridge_coverage_receipt(support),
    }


def _guard_feature_vector(
    row: Mapping[str, object], *, use_phase_bridge: bool
) -> np.ndarray:
    base = _feature_vector(row)
    if not use_phase_bridge:
        return base
    bridge = row.get("phase_bridge")
    if not isinstance(bridge, Mapping):
        raise TypeError("phase-bridge selector row lacks bridge observation")
    error = float(bridge["robust_normalized_median_absolute_error"])
    correlation = float(bridge["correlation"])
    extras = np.asarray(
        [np.clip(np.arcsinh(error) / 3.0, 0.0, 2.0), np.clip(correlation, -1.0, 1.0)],
        dtype=np.float64,
    )
    return np.concatenate((base, extras))


def _nearest_evidence(
    evidence: Sequence[Mapping[str, object]],
    query: Mapping[str, object],
    *,
    use_phase_bridge: bool,
) -> list[Mapping[str, object]]:
    if not evidence:
        return []
    query_vector = _guard_feature_vector(query, use_phase_bridge=use_phase_bridge)
    distances = [
        (
            float(
                np.linalg.norm(
                    _guard_feature_vector(row, use_phase_bridge=use_phase_bridge)
                    - query_vector
                )
            ),
            row,
        )
        for row in evidence
    ]
    cutoff = sorted(distance for distance, _ in distances)[
        min(NEAREST_K, len(distances)) - 1
    ]
    return [row for distance, row in distances if distance <= cutoff + 1e-12]


def _action_verdict(
    evidence: Sequence[Mapping[str, object]], action: str
) -> dict[str, object]:
    gains = [
        float(row["arms"]["identity"]["loss_j"])
        - float(row["arms"][action]["loss_j"])
        for row in evidence
    ]
    count = len(gains)
    if count == 0:
        return {
            "verdict": "unresolved",
            "evidence_count": 0,
            "mean_action_vs_identity_gain": None,
            "support_share": 0.0,
            "harm_share": 0.0,
        }
    mean_gain = _mean(gains)
    support_share = sum(gain >= GAIN_MIN for gain in gains) / count
    harm_share = sum(gain <= -HARM_MARGIN for gain in gains) / count
    if (
        count >= MIN_EVIDENCE
        and mean_gain >= GAIN_MIN
        and support_share >= MIN_SUPPORT_SHARE
        and harm_share <= MAX_HARM_SHARE
    ):
        verdict = "supported"
    elif count >= MIN_EVIDENCE and (
        mean_gain <= -HARM_MARGIN or harm_share > MAX_HARM_SHARE
    ):
        verdict = "contradicted"
    else:
        verdict = "unresolved"
    return {
        "verdict": verdict,
        "evidence_count": count,
        "mean_action_vs_identity_gain": mean_gain,
        "support_share": support_share,
        "harm_share": harm_share,
    }


def _guarded_decision(
    evidence: Sequence[Mapping[str, object]],
    query: Mapping[str, object],
    *,
    use_phase_bridge: bool,
) -> dict[str, object]:
    nearest = _nearest_evidence(
        evidence, query, use_phase_bridge=use_phase_bridge
    )
    verdicts = {
        action: _action_verdict(nearest, action) for action in NON_IDENTITY_ACTIONS
    }
    bridge = query.get("phase_bridge") if use_phase_bridge else None
    if use_phase_bridge and not isinstance(bridge, Mapping):
        raise TypeError("phase-bridge guard requires a query bridge observation")
    bridge_error = (
        float(bridge["robust_normalized_median_absolute_error"])
        if isinstance(bridge, Mapping)
        else None
    )
    bridge_coverage = (
        float(bridge["coverage"]) if isinstance(bridge, Mapping) else None
    )
    seasonal_bridge_override = bool(
        use_phase_bridge
        and bridge_coverage is not None
        and bridge_coverage > 0.0
        and bridge_error is not None
        and bridge_error > PHASE_BRIDGE_ERROR_MAX
    )
    if seasonal_bridge_override:
        verdicts["seasonal"] = {
            **verdicts["seasonal"],
            "verdict_before_phase_bridge_guard": verdicts["seasonal"]["verdict"],
            "verdict": "contradicted",
            "phase_bridge_risk_override": True,
        }
    supported = [
        action
        for action in NON_IDENTITY_ACTIONS
        if verdicts[action]["verdict"] == "supported"
    ]
    if supported:
        action = max(
            supported,
            key=lambda item: (
                float(verdicts[item]["mean_action_vs_identity_gain"]),
                -PROGRAM_IDS.index(item),
            ),
        )
        resolved = True
        resolution = "supported_non_identity"
    elif all(
        verdicts[action]["verdict"] == "contradicted"
        for action in NON_IDENTITY_ACTIONS
    ):
        action = "identity"
        resolved = True
        resolution = "all_non_identity_contradicted"
    else:
        action = "identity"
        resolved = False
        resolution = "unresolved_abstain"
    return {
        "action": action,
        "resolved": resolved,
        "resolution": resolution,
        "nearest_evidence_count": len(nearest),
        "action_verdicts": verdicts,
        "phase_bridge_enabled": use_phase_bridge,
        "phase_bridge_error": bridge_error,
        "phase_bridge_coverage": bridge_coverage,
        "seasonal_phase_bridge_risk_override": seasonal_bridge_override,
    }


def _overlay_decision(
    target_guard: Mapping[str, object], source_guard: Mapping[str, object]
) -> dict[str, str]:
    """Apply target verdicts to the action actually proposed by the source guard."""

    if target_guard["resolution"] == "supported_non_identity":
        return {
            "action": str(target_guard["action"]),
            "resolution_source": "target_supported_action",
        }
    source_action = str(source_guard["action"])
    target_verdicts = target_guard["action_verdicts"]
    if (
        source_action != "identity"
        and target_verdicts[source_action]["verdict"] == "contradicted"
    ):
        return {
            "action": "identity",
            "resolution_source": "target_contradicts_source_action",
        }
    if target_guard["resolution"] == "all_non_identity_contradicted":
        return {
            "action": "identity",
            "resolution_source": "all_target_actions_contradicted",
        }
    return {
        "action": source_action,
        "resolution_source": "source_guard_fallback",
    }


def _selection_summary(
    rows: Sequence[Mapping[str, object]], choices: Mapping[str, str]
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {"all": list(rows)}
    groups.update(
        {
            family: [
                row for row in rows if row["target_family_report_only"] == family
            ]
            for family in TARGET_FAMILIES
        }
    )
    result: dict[str, object] = {}
    for group_name, group in groups.items():
        losses = [
            float(row["arms"][choices[str(row["case_id"])]]["loss_j"])
            for row in group
        ]
        oracle = [float(row["menu_oracle_loss"]) for row in group]
        result[group_name] = {
            "n_query_cases": len(group),
            "mean_query_loss": _mean(losses),
            "mean_adaptation_regret": _mean(
                [loss - oracle_loss for loss, oracle_loss in zip(losses, oracle)]
            ),
            "action_counts": {
                action: sum(choices[str(row["case_id"])] == action for row in group)
                for action in PROGRAM_IDS
            },
        }
    return result


def _harm_summary(
    rows: Sequence[Mapping[str, object]],
    left: Mapping[str, str],
    right: Mapping[str, str],
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = {"all": list(rows)}
    groups.update(
        {
            family: [
                row for row in rows if row["target_family_report_only"] == family
            ]
            for family in TARGET_FAMILIES
        }
    )
    result: dict[str, object] = {}
    for group_name, group in groups.items():
        deltas = []
        for row in group:
            case_id = str(row["case_id"])
            deltas.append(
                float(row["arms"][left[case_id]]["loss_j"])
                - float(row["arms"][right[case_id]]["loss_j"])
            )
        result[group_name] = {
            "mean_loss_delta": _mean(deltas),
            "harm_case_count_at_margin": sum(delta > HARM_MARGIN for delta in deltas),
            "harm_case_rate_at_margin": sum(delta > HARM_MARGIN for delta in deltas)
            / len(deltas),
            "margin": HARM_MARGIN,
        }
    return result


def run_e1tr_guarded_transfer(
    valuator: FrozenChronosValuator,
    *,
    source_report: Path,
    failed_e1t_report: Path,
) -> dict[str, object]:
    source_rows, source_metadata = _load_source_evidence(source_report)
    enriched_source_rows = _enrich_frozen_source_evidence(
        source_rows, source_report
    )
    source_metadata["deterministic_context_id_order_check"] = "PASS"
    source_metadata["judge_outcomes_recomputed"] = False
    source_metadata["phase_bridge_coverage_receipt"] = _bridge_coverage_receipt(
        enriched_source_rows
    )
    source_evidence = [
        _bridge_selection_view(row, include_revealed_outcomes=True)
        for row in enriched_source_rows
    ]
    target_support, target_support_metadata = _load_frozen_target_support(
        failed_e1t_report
    )
    cases = build_repair_query_cases()
    query_public = []
    for case in cases:
        descriptor = _public_descriptor(case)
        descriptor["phase_bridge"] = _phase_bridge_observation(
            case.corrupt_context, int(descriptor["observed_period_parameter"])
        )
        query_public.append(descriptor)
    query_selection = [
        _bridge_selection_view(row, include_revealed_outcomes=False)
        for row in query_public
    ]

    # Freeze all equal-budget actions and tri-state receipts before Query judging.
    plans: dict[int, dict[str, dict[str, object]]] = {}
    choices: dict[int, dict[str, dict[str, str]]] = {}
    for budget in BUDGETS:
        revealed_target = target_support[:budget]
        plans[budget] = {}
        choices[budget] = {arm: {} for arm in REPAIR_ARMS}
        for descriptor, query in zip(query_public, query_selection):
            case_id = str(descriptor["case_id"])
            target_guard = _guarded_decision(
                revealed_target, query, use_phase_bridge=True
            )
            source_guard = _guarded_decision(
                source_evidence, query, use_phase_bridge=True
            )
            overlay = _overlay_decision(target_guard, source_guard)
            overlay_action = overlay["action"]
            target_guard_no_bridge = _guarded_decision(
                revealed_target, query, use_phase_bridge=False
            )
            source_guard_no_bridge = _guarded_decision(
                source_evidence, query, use_phase_bridge=False
            )
            no_bridge_overlay = _overlay_decision(
                target_guard_no_bridge, source_guard_no_bridge
            )
            naive_action = _naive_union_select_action(
                [*source_evidence, *revealed_target], query
            )
            choices[budget]["a3_target_only_guarded"][case_id] = str(
                target_guard["action"]
            )
            choices[budget]["a4_source_only_guarded"][case_id] = str(
                source_guard["action"]
            )
            choices[budget]["a5_guarded_overlay"][case_id] = overlay_action
            choices[budget]["a5_no_bridge_guarded_ablation"][case_id] = (
                no_bridge_overlay["action"]
            )
            choices[budget]["a5_naive_union_ablation"][case_id] = naive_action
            plans[budget][case_id] = {
                "target_guard": target_guard,
                "source_guard": source_guard,
                "a5_overlay_resolution_source": overlay["resolution_source"],
                "no_bridge_ablation": {
                    "target_guard": target_guard_no_bridge,
                    "source_guard": source_guard_no_bridge,
                    "a5_overlay_resolution_source": no_bridge_overlay[
                        "resolution_source"
                    ],
                },
                "actions": {
                    arm: choices[budget][arm][case_id] for arm in REPAIR_ARMS
                },
            }

    query_rows = [
        _judge_case(case, descriptor, valuator)
        for case, descriptor in zip(cases, query_public)
    ]

    budget_results: dict[str, object] = {}
    for budget in BUDGETS:
        budget_results[str(budget)] = {
            "revealed_frozen_support_count": budget,
            "arms": {
                arm: _selection_summary(query_rows, choices[budget][arm])
                for arm in REPAIR_ARMS
            },
            "harm": {
                "guarded_a5_minus_a3": _harm_summary(
                    query_rows,
                    choices[budget]["a5_guarded_overlay"],
                    choices[budget]["a3_target_only_guarded"],
                ),
                "guarded_a5_minus_a4": _harm_summary(
                    query_rows,
                    choices[budget]["a5_guarded_overlay"],
                    choices[budget]["a4_source_only_guarded"],
                ),
                "guarded_a5_minus_naive_union": _harm_summary(
                    query_rows,
                    choices[budget]["a5_guarded_overlay"],
                    choices[budget]["a5_naive_union_ablation"],
                ),
                "bridge_guarded_a5_minus_no_bridge_guarded": _harm_summary(
                    query_rows,
                    choices[budget]["a5_guarded_overlay"],
                    choices[budget]["a5_no_bridge_guarded_ablation"],
                ),
            },
        }

    regret_auc: dict[str, dict[str, float]] = {arm: {} for arm in REPAIR_ARMS}
    for arm in REPAIR_ARMS:
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

    diagnostics = _diagnostics(
        query_rows,
        source_fixed_action=_fixed_action_from_evidence(source_evidence),
    )
    best_fixed = float(diagnostics["all"]["posthoc_best_fixed_mean_loss"])
    menu_oracle = float(diagnostics["all"]["menu_oracle_mean_loss"])
    a3_auc = regret_auc["a3_target_only_guarded"]["all"]
    a5_auc = regret_auc["a5_guarded_overlay"]["all"]
    naive_auc = regret_auc["a5_naive_union_ablation"]["all"]
    no_bridge_auc = regret_auc["a5_no_bridge_guarded_ablation"]["all"]
    b0_harm = budget_results["0"]["harm"]["guarded_a5_minus_a3"]["all"]
    b8_harm = budget_results["8"]["harm"]["guarded_a5_minus_a3"]["all"]
    b8_a4_loss = float(
        budget_results["8"]["arms"]["a4_source_only_guarded"]["all"][
            "mean_query_loss"
        ]
    )
    b8_a5_loss = float(
        budget_results["8"]["arms"]["a5_guarded_overlay"]["all"]["mean_query_loss"]
    )
    gates = {
        "protocol_integrity": {
            "gate_status": "structural_repair_gate",
            "new_query_plans_frozen_before_query_judging": True,
            "old_e1t_query_outcomes_used": False,
            "old_e1t_query_role": "failure_diagnosis_and_repair_design_only",
            "selector_private_fields_visible": False,
            "pass": True,
        },
        "target_routing_headroom": {
            "gate_status": "controlled_repair_pilot_gate_not_confirmation",
            "best_fixed_mean_loss": best_fixed,
            "menu_oracle_mean_loss": menu_oracle,
            "best_fixed_minus_menu_oracle_mean_loss": best_fixed - menu_oracle,
            "minimum_headroom": GAIN_MIN,
            "pass": best_fixed - menu_oracle >= GAIN_MIN,
        },
        "guarded_source_plus_target_regret_auc": {
            "gate_status": "controlled_repair_pilot_gate_not_confirmation",
            "a3_target_only_guarded_regret_auc": a3_auc,
            "a5_guarded_overlay_regret_auc": a5_auc,
            "a3_minus_a5_regret_auc": a3_auc - a5_auc,
            "minimum_improvement": HARM_MARGIN,
            "pass": a3_auc - a5_auc >= HARM_MARGIN,
        },
        "guarded_vs_naive_union": {
            "gate_status": "controlled_repair_pilot_gate_not_confirmation",
            "guarded_regret_auc": a5_auc,
            "naive_union_regret_auc": naive_auc,
            "naive_minus_guarded_regret_auc": naive_auc - a5_auc,
            "pass": a5_auc <= naive_auc,
        },
        "phase_bridge_vs_no_bridge_guard": {
            "gate_status": "controlled_repair_pilot_gate_not_confirmation",
            "phase_bridge_guarded_regret_auc": a5_auc,
            "no_bridge_guarded_regret_auc": no_bridge_auc,
            "no_bridge_minus_phase_bridge_regret_auc": no_bridge_auc - a5_auc,
            "pass": a5_auc <= no_bridge_auc,
        },
        "b0_b8_negative_transfer_control": {
            "gate_status": "controlled_repair_pilot_gate_not_confirmation",
            "comparison": "guarded A5 minus guarded A3; B0 A3 is identity abstain",
            "maximum_mean_harm": HARM_MARGIN,
            "maximum_harm_case_share": MAX_HARM_SHARE,
            "b0": b0_harm,
            "b8": b8_harm,
            "pass": all(
                float(item["mean_loss_delta"]) <= HARM_MARGIN
                and float(item["harm_case_rate_at_margin"]) <= MAX_HARM_SHARE
                for item in (b0_harm, b8_harm)
            ),
        },
        "final_budget_overlay_vs_source": {
            "gate_status": "controlled_repair_pilot_gate_not_confirmation",
            "a4_source_only_guarded_query_loss": b8_a4_loss,
            "a5_guarded_overlay_query_loss": b8_a5_loss,
            "pass": b8_a5_loss <= b8_a4_loss + HARM_MARGIN,
        },
    }

    return {
        "schema_version": "e1tr-guarded-transfer-repair-pilot/1",
        "scientific_role": "controlled_guarded_selector_repair_pilot",
        "frozen_evidence": {
            "source_e1p": source_metadata,
            "target_support_e1t": target_support_metadata,
        },
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
            "budget_unit": "one frozen target support case complete three-program outcome",
            "repair_query_specs_frozen": [list(spec) for spec in REPAIR_QUERY_SPECS],
            "programs": list(PROGRAM_IDS),
            "guard_thresholds_fixed_before_new_query_judging": {
                "gain_min": GAIN_MIN,
                "harm_margin": HARM_MARGIN,
                "max_harm_share": MAX_HARM_SHARE,
                "min_support_share": MIN_SUPPORT_SHARE,
                "min_evidence": MIN_EVIDENCE,
                "nearest_k_with_all_boundary_ties": NEAREST_K,
                "seasonal_phase_bridge_error_max": PHASE_BRIDGE_ERROR_MAX,
            },
            "phase_bridge_observation": {
                "source": "corrupt_context_only",
                "binding": "primary_gap_interval_and_public_observed_period",
                "distance_features": [
                    "robust_normalized_median_absolute_error",
                    "correlation",
                ],
                "coverage_role": "reliability_receipt",
                "threshold_provenance": (
                    "single failed-E1T development diagnostic after failure and before "
                    "the unjudged E1-TR Query; no parameter search"
                ),
            },
            "a5_overlay_rule": (
                "target supported action overrides; target contradiction of the proposed "
                "source action abstains to identity; otherwise unresolved target evidence "
                "falls back to the source guard"
            ),
            "agent_enabled": False,
            "memory_enabled": False,
            "model_training_enabled": False,
            "parameter_search_enabled": False,
        },
        "information_wall": {
            "query_plans_frozen_before_query_judging": True,
            "old_e1t_query_outcomes_used": False,
            "selection_receives": [
                "corrupt_context_global_features",
                "corrupt_context_gap_local_features",
                "corrupt_context_phase_bridge_error_and_correlation",
                "frozen_or_revealed_evidence_action_losses",
            ],
            "selection_does_not_receive": [
                "target_family",
                "target_variant",
                "semantic_case_id",
                "clean_context",
                "clean_future",
                "new_query_outcome",
                "old_e1t_query_outcome",
            ],
        },
        "query_cases": query_rows,
        "repair_design_provenance": {
            "failed_e1t_old_query_used_for": "read-only failure diagnosis and counterfactual design validation",
            "failed_e1t_old_query_used_by_runtime_selector": False,
            "failed_e1t_old_query_outcomes_in_new_query_metrics": False,
            "independent_confirmation_claimed": False,
        },
        "query_action_plans_and_guard_receipts": {
            str(budget): plans[budget] for budget in BUDGETS
        },
        "diagnostics": diagnostics,
        "budget_results": budget_results,
        "adaptation_regret_auc": regret_auc,
        "gates": gates,
        "all_repair_pilot_gates_pass": all(
            bool(gate["pass"]) for gate in gates.values()
        ),
        "limitations": [
            "This is a controlled selector repair pilot, not confirmation or a domain claim.",
            "Frozen E1-P source evidence retains PARTIAL risk coverage.",
            "Only eight previously judged target support cases are available.",
            "The tri-state constants are protocol constants, not fitted thresholds.",
            "The 0.75 bridge threshold is a single post-failure development diagnostic fixed before the new Query, not a searched threshold.",
            "Old E1-T Query outcomes informed failure diagnosis, so this repair pilot is not independent confirmation.",
            "The no-bridge guarded and naive-union arms are no-extra-judge-cost ablations.",
            "New Query seeds and generators are frozen before the sealed run and must not be tuned to its outcomes.",
        ],
        "claim_limit": (
            "E1-TR can only test whether a fixed tri-state guard repairs the observed "
            "controlled cross-generator selection failure on a fresh Query."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the controlled E1-TR guarded transfer repair pilot."
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=project_root / "artifacts/functional/e1p/periodic_missing_report.json",
    )
    parser.add_argument(
        "--failed-e1t-report",
        type=Path,
        default=project_root
        / "artifacts/functional/e1t/source_target_transfer_report.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = run_e1tr_guarded_transfer(
        FrozenChronosValuator(),
        source_report=args.source_report.resolve(),
        failed_e1t_report=args.failed_e1t_report.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={output}")
    print(f"all_repair_pilot_gates_pass={report['all_repair_pilot_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_repair_query_cases", "run_e1tr_guarded_transfer"]
