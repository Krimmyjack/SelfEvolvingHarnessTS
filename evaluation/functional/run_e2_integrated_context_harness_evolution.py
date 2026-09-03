"""Run one integrated controlled Context-conditioned Harness evolution test.

The Source failure sequence is frozen before these targets are opened.  H0 sees
localized class-conditioned nodes but has no cross-cohort applicability
observation, so forced repair can erase stable task evidence.  The Source update
adds one Program-specific observation -- whether the same local label evidence
repeats in Support -- and compiles two scopes: contraindicated task evidence, or
an artifact candidate that requires one full Target Support confirmation.

Six new, reasonably sized binary UCR TRAIN splits freeze all target contexts.
``evaluate`` opens their official TEST splits once and compares conservative
target-only adaptation (A3), the Source scope without feedback (A4), and the
Source scope plus equal-budget Target confirmation (A5).  No valuation proxy,
Dataset-ID rule, persistent Memory, UCI target, or natural-defect claim is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-integrated-context-harness-evolution/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_integrated_context_harness_evolution_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_integrated_context_harness_evolution_report.json"
)
W48_REPORT_PATH = (
    "artifacts/functional/e2/source_task_context_label_evidence_witness_report.json"
)
W50B_REPORT_PATH = (
    "artifacts/functional/e2/source_task_risk_confirmation_adaptation_report.json"
)
DATA_DIR = "data/ucr_task_context"
TARGET_DATASETS = (
    "Computers",
    "PowerCons",
    "Yoga",
    "SemgHandGenderCh2",
    "WormsTwoClass",
    "HandOutlines",
)
CONDITIONS = ("fit_only_artifact", "stable_task_event")
FEEDBACK_ORDERS = (
    ("fit_only_artifact", "stable_task_event"),
    ("stable_task_event", "fit_only_artifact"),
)
EXECUTE = "EXECUTE_BOUND_REPAIR"
ABSTAIN = "ABSTAIN_KEEP_INCUMBENT"
REQUEST = "REQUEST_FULL_CONFIRMATION"
ELIGIBLE = "ELIGIBLE_REQUEST_CONFIRMATION"
CONTRAINDICATED = "CONTRAINDICATED_ABSTAIN"
UNRESOLVED = "UNRESOLVED_ABSTAIN"
MIN_POSITIVE_ARTIFACT_TARGETS = 4
MIN_UNSCOPED_EVENT_HARMFUL_TARGETS = 3


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _compile_scope(risk_decision: str) -> tuple[str, list[str]]:
    """Compile Source evidence into applicability, not unconfirmed execution."""

    if risk_decision == EXECUTE:
        return ELIGIBLE, ["fit_local_evidence_not_reproduced_in_support"]
    if risk_decision == ABSTAIN:
        return CONTRAINDICATED, ["stable_local_task_evidence_repeats_in_support"]
    return UNRESOLVED, ["cross_cohort_local_evidence_is_insufficient"]


def _initial_state(scope: str) -> str:
    return REQUEST if scope == ELIGIBLE else ABSTAIN


def _build_target_train_plan(root: Path, dataset: str) -> dict[str, Any]:
    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _condition_inputs,
        _helpers,
    )

    task_context, helpers = _helpers()
    archive = root / DATA_DIR / f"{dataset}.zip"
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    fit_indices, support_indices = helpers["split"](np, train_labels)
    base_fit = train_values[fit_indices]
    fit_labels = train_labels[fit_indices]
    base_support = train_values[support_indices]
    support_labels = train_labels[support_indices]
    positions = helpers["positions"](train_values.shape[1])
    conditions: dict[str, Any] = {}
    for condition_name in CONDITIONS:
        fit_values, support_values = _condition_inputs(
            np,
            base_fit=base_fit,
            fit_labels=fit_labels,
            base_support=base_support,
            support_labels=support_labels,
            positions=positions,
            condition=condition_name,
            inject=helpers["inject"],
        )
        observation = helpers["observe"](np, fit_values, fit_labels)
        nodes = tuple(int(node) for node in observation["nodes"])
        witness = helpers["witness"](
            np,
            fit_values,
            fit_labels,
            support_values,
            support_labels,
            nodes,
            helpers["rolling_median"],
        )
        risk_decision, risk_reasons = helpers["risk_decision"](witness)
        scope, scope_reasons = _compile_scope(risk_decision)
        conditions[condition_name] = {
            "observation": observation,
            "witness": witness,
            "decision_input_text": helpers["decision_text"](task_context, witness),
            "decision_input_contains_instance_answer": False,
            "legacy_risk_decision": risk_decision,
            "legacy_risk_reasons": risk_reasons,
            "evolved_scope": scope,
            "evolved_scope_reasons": scope_reasons,
            "initial_policy_state": _initial_state(scope),
        }
    priority_order = sorted(
        CONDITIONS,
        key=lambda name: (
            0 if conditions[name]["evolved_scope"] == ELIGIBLE else 1,
            name,
        ),
    )
    return {
        "dataset": dataset,
        "archive": f"{DATA_DIR}/{dataset}.zip",
        "official_train_count": int(train_values.shape[0]),
        "series_length": int(train_values.shape[1]),
        "fit_count": int(fit_indices.size),
        "support_count": int(support_indices.size),
        "class_counts": {
            str(label): int(np.count_nonzero(train_labels == label))
            for label in (0, 1)
        },
        "conditions": conditions,
        "source_priority_order": priority_order,
    }


def build_plan(root: Path) -> dict[str, Any]:
    w48 = _read_object(root / W48_REPORT_PATH)
    w50b = _read_object(root / W50B_REPORT_PATH)
    if w48.get("verdict") != "CONTROLLED_NONORACLE_TASK_RISK_WITNESS_PASS":
        raise ValueError("W48 Source Observation/Scope evidence is unavailable")
    if w50b.get("verdict") != "CONTROLLED_RISK_CONFIRMATION_A5_VS_A3_FAIL":
        raise ValueError("W50b must have rejected unconfirmed eligible execution")
    targets = [_build_target_train_plan(root, dataset) for dataset in TARGET_DATASETS]
    for row in targets:
        conditions = row["conditions"]
        if conditions["fit_only_artifact"]["evolved_scope"] != ELIGIBLE:
            raise ValueError(f"artifact applicability unresolved: {row['dataset']}")
        if conditions["stable_task_event"]["evolved_scope"] != CONTRAINDICATED:
            raise ValueError(f"event risk unresolved: {row['dataset']}")
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "A Source failure-driven cross-cohort Observation and applicability Scope "
            "reduce equal-budget Target adaptation regret and event-erasure risk relative "
            "to conservative Target-only adaptation and an unscoped H0 Harness."
        ),
        "scientific_role": "integrated controlled Harness evolution confirmation",
        "context_exposure": "INSTANCE_SEEN_TRAIN_ONLY",
        "outcome_exposure": "SEALED",
        "source_failure_pattern_card": {
            "H0_failure": {
                "surface": "observation_and_scope",
                "unscoped_event_macro_harm": float(
                    w48["overall"]["unscoped_event_macro_query_harm"]
                ),
                "unscoped_event_harmful_dataset_count": int(
                    w48["overall"]["unscoped_event_harmful_dataset_count"]
                ),
                "diagnosis": (
                    "fit-local node strength alone cannot distinguish acquisition artifact "
                    "from stable class evidence"
                ),
            },
            "source_patch": {
                "target_surface": "observation_plus_scope",
                "operation": "ADD_CROSS_COHORT_OBSERVATION_AND_RESTRICT_SCOPE",
                "new_observation": "support_to_fit_node_strength_and_direction_alignment",
                "eligible_behavior": "request full Target confirmation",
                "contraindicated_behavior": "abstain",
                "program_changed": False,
                "consumer_changed": False,
                "proxy_used": False,
                "patch_origin": "deterministic human-mechanistic compiler from exposed Source failures",
            },
            "why_confirmation_is_required": {
                "source_evidence": w50b["verdict"],
                "negative_transfer_target_count": int(
                    w50b["overall"]["negative_transfer_target_count"]
                ),
            },
        },
        "target_train_plans": targets,
        "frozen_policies": {
            "H0_unscoped": "execute both localized conditions without cross-cohort scope",
            "A3_target_only": (
                "conservative incumbent; mean over both condition-confirmation orders"
            ),
            "A4_source_only": (
                "Source eligibility/contraindication states; no unconfirmed execution"
            ),
            "A5_source_plus_target": (
                "confirm Source-eligible condition first; exact Support feedback controls execution"
            ),
            "full_confirmation_rule": "execute iff Support accuracy gain > 0",
            "feedback_budgets": [0, 1, 2],
        },
        "feedback_budget_contract": {
            "budget_unit": (
                "one complete Target Support Consumer refit/outcome that may change "
                "the execution state of one condition"
            ),
            "free_target_context": (
                "official Target TRAIN inputs and labels, including the fixed fit/support "
                "split used to construct observations"
            ),
            "shared_visibility": (
                "A3 and A5 receive the same Target TRAIN inputs and labels; A5 additionally "
                "receives the Source-compiled Observation/Scope prior"
            ),
            "A3_scope": (
                "order-symmetrized conservative target-only adaptation, not an optimal "
                "or unrestricted target-only search"
            ),
        },
        "frozen_success_gate": {
            "observer_and_scope_realization_all_targets": True,
            "positive_artifact_target_count_min": MIN_POSITIVE_ARTIFACT_TARGETS,
            "macro_A5_adapt_auc_strictly_greater_than_A3": True,
            "negative_transfer_target_count": 0,
            "A5_event_harm_max": 0.0,
            "unscoped_event_harmful_target_count_min": MIN_UNSCOPED_EVENT_HARMFUL_TARGETS,
        },
        "target_test_values_or_labels_read": False,
        "selection_used_target_program_or_consumer_outcome": False,
        "persistent_memory_built": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "Real UCR backgrounds and labels are used, but the local artifact/event mechanism "
            "is controlled. This can confirm Harness evolution mechanics, not natural-defect transfer."
        ),
    }


def _policy_result(
    states: dict[str, str], conditions: dict[str, Any]
) -> dict[str, float]:
    utilities: list[float] = []
    harms: list[float] = []
    for condition_name in CONDITIONS:
        row = conditions[condition_name]
        incumbent = float(row["incumbent"]["query_accuracy"])
        action = float(row["action"]["query_accuracy"])
        selected = action if states[condition_name] == EXECUTE else incumbent
        utilities.append(selected)
        harms.append(max(0.0, incumbent - selected))
    return {
        "utility": sum(utilities) / len(utilities),
        "harm": sum(harms) / len(harms),
        "event_harm": harms[CONDITIONS.index("stable_task_event")],
    }


def _adaptation_curve(
    *,
    initial: dict[str, str],
    order: tuple[str, str] | list[str],
    conditions: dict[str, Any],
    locked_contraindications: set[str],
) -> list[dict[str, Any]]:
    states = dict(initial)
    curve: list[dict[str, Any]] = []
    for budget in range(3):
        curve.append(
            {
                "budget": budget,
                "states": dict(states),
                **_policy_result(states, conditions),
            }
        )
        if budget < 2:
            observed = str(order[budget])
            support_gain = float(conditions[observed]["forced_support_gain"])
            if observed in locked_contraindications:
                states[observed] = ABSTAIN
                rule = "source_contraindication_preserved"
            else:
                states[observed] = EXECUTE if support_gain > 0.0 else ABSTAIN
                rule = "full_support_confirmation"
            curve[-1]["next_feedback"] = {
                "condition": observed,
                "support_gain": support_gain,
                "compiled_state": states[observed],
                "rule": rule,
            }
    return curve


def _average_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    return [
        {
            "budget": budget,
            "utility": sum(curve[budget]["utility"] for curve in curves)
            / len(curves),
            "harm": sum(curve[budget]["harm"] for curve in curves) / len(curves),
            "event_harm": sum(curve[budget]["event_harm"] for curve in curves)
            / len(curves),
        }
        for budget in range(3)
    ]


def _adapt_auc(curve: list[dict[str, Any]]) -> float:
    return sum(float(point["utility"]) for point in curve) / len(curve)


def evaluate(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _condition_inputs,
        _fit_readout,
        _helpers,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_scoped_impulse_skill_transfer import (
        _localization_evaluation,
    )

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("integrated Harness plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("integrated Target outcomes were not sealed")
    target_plans = plan["target_train_plans"]
    planned = {str(row["dataset"]): row for row in target_plans}
    if (
        len(target_plans) != len(TARGET_DATASETS)
        or len(planned) != len(TARGET_DATASETS)
        or set(planned) != set(TARGET_DATASETS)
    ):
        raise ValueError("integrated Target roster is not the frozen unique roster")
    task_context, helpers = _helpers()
    del task_context
    rows: list[dict[str, Any]] = []
    fit_count = 0

    for dataset in TARGET_DATASETS:
        current = _build_target_train_plan(root, dataset)
        for condition_name in CONDITIONS:
            old = planned[dataset]["conditions"][condition_name]
            new = current["conditions"][condition_name]
            if (
                old["evolved_scope"] != new["evolved_scope"]
                or old["observation"]["nodes"] != new["observation"]["nodes"]
            ):
                raise ValueError(f"planned TRAIN context changed: {dataset}/{condition_name}")
        if planned[dataset]["source_priority_order"] != current["source_priority_order"]:
            raise ValueError(f"planned TRAIN priority changed: {dataset}")

        archive = root / DATA_DIR / f"{dataset}.zip"
        train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
        query_values, query_labels = helpers["load"](np, archive, dataset, "TEST")
        fit_indices, support_indices = helpers["split"](np, train_labels)
        base_fit = train_values[fit_indices]
        fit_labels = train_labels[fit_indices]
        base_support = train_values[support_indices]
        support_labels = train_labels[support_indices]
        positions = helpers["positions"](train_values.shape[1])
        conditions: dict[str, Any] = {}

        for condition_name in CONDITIONS:
            fit_values, support_values = _condition_inputs(
                np,
                base_fit=base_fit,
                fit_labels=fit_labels,
                base_support=base_support,
                support_labels=support_labels,
                positions=positions,
                condition=condition_name,
                inject=helpers["inject"],
            )
            condition_query = (
                query_values.copy()
                if condition_name == "fit_only_artifact"
                else helpers["inject"](np, query_values, query_labels, positions)
            )
            condition_plan = planned[dataset]["conditions"][condition_name]
            nodes = tuple(int(node) for node in condition_plan["observation"]["nodes"])
            repaired, modification = helpers["apply_program"](
                np,
                fit_values,
                positions=nodes,
                window_length=train_values.shape[1],
            )
            incumbent = _fit_readout(
                np,
                RidgeClassifier,
                helpers["features"],
                fit_values,
                fit_labels,
                support_values,
                support_labels,
                condition_query,
                query_labels,
            )
            action = _fit_readout(
                np,
                RidgeClassifier,
                helpers["features"],
                repaired,
                fit_labels,
                support_values,
                support_labels,
                condition_query,
                query_labels,
            )
            fit_count += 2
            conditions[condition_name] = {
                "evolved_scope": condition_plan["evolved_scope"],
                "witness": condition_plan["witness"],
                "localization": _localization_evaluation(list(nodes), positions),
                "program_modification": modification,
                "incumbent": incumbent,
                "action": action,
                "forced_support_gain": action["support_accuracy"]
                - incumbent["support_accuracy"],
                "forced_query_gain": action["query_accuracy"]
                - incumbent["query_accuracy"],
            }

        conservative = {condition: REQUEST for condition in CONDITIONS}
        source_initial = {
            condition: _initial_state(conditions[condition]["evolved_scope"])
            for condition in CONDITIONS
        }
        a3_orders = [
            _adaptation_curve(
                initial=conservative,
                order=order,
                conditions=conditions,
                locked_contraindications=set(),
            )
            for order in FEEDBACK_ORDERS
        ]
        a3 = _average_curves(a3_orders)
        source_order = list(planned[dataset]["source_priority_order"])
        contraindicated = {
            condition
            for condition in CONDITIONS
            if conditions[condition]["evolved_scope"] == CONTRAINDICATED
        }
        a5 = _adaptation_curve(
            initial=source_initial,
            order=source_order,
            conditions=conditions,
            locked_contraindications=contraindicated,
        )
        a4_point = _policy_result(source_initial, conditions)
        a4 = [{"budget": budget, **a4_point} for budget in range(3)]
        h0_point = _policy_result(
            {condition: EXECUTE for condition in CONDITIONS}, conditions
        )
        h0 = [{"budget": budget, **h0_point} for budget in range(3)]
        a3_auc = _adapt_auc(a3)
        a5_auc = _adapt_auc(a5)
        scope_realized = bool(
            conditions["fit_only_artifact"]["evolved_scope"] == ELIGIBLE
            and conditions["stable_task_event"]["evolved_scope"] == CONTRAINDICATED
            and all(
                conditions[name]["localization"]["exact_precision_recall_pass"]
                for name in CONDITIONS
            )
        )
        rows.append(
            {
                "dataset": dataset,
                "context_or_decision_used_dataset_id": False,
                "conditions": conditions,
                "source_priority_order": source_order,
                "H0_unscoped": {"mean_curve": h0, "adapt_auc": _adapt_auc(h0)},
                "A3_target_only": {
                    "per_order": a3_orders,
                    "mean_curve": a3,
                    "adapt_auc": a3_auc,
                },
                "A4_source_only": {"mean_curve": a4, "adapt_auc": _adapt_auc(a4)},
                "A5_source_plus_target": {"mean_curve": a5, "adapt_auc": a5_auc},
                "A5_minus_A3_adapt_auc": a5_auc - a3_auc,
                "scope_update_realized": scope_realized,
            }
        )

    a3_macro = sum(row["A3_target_only"]["adapt_auc"] for row in rows) / len(rows)
    a5_macro = sum(row["A5_source_plus_target"]["adapt_auc"] for row in rows) / len(rows)
    negative_count = sum(row["A5_minus_A3_adapt_auc"] < -1e-12 for row in rows)
    artifact_positive_count = sum(
        row["conditions"]["fit_only_artifact"]["forced_query_gain"] > 0.0
        for row in rows
    )
    a5_event_harm_max = max(
        point["event_harm"]
        for row in rows
        for point in row["A5_source_plus_target"]["mean_curve"]
    )
    h0_event_harmful_count = sum(
        row["conditions"]["stable_task_event"]["forced_query_gain"] < 0.0
        for row in rows
    )
    h0_event_harm_macro = sum(
        max(
            0.0,
            -float(row["conditions"]["stable_task_event"]["forced_query_gain"]),
        )
        for row in rows
    ) / len(rows)
    realization_all = all(bool(row["scope_update_realized"]) for row in rows)
    gate_pass = bool(
        realization_all
        and artifact_positive_count >= MIN_POSITIVE_ARTIFACT_TARGETS
        and a5_macro > a3_macro
        and negative_count == 0
        and a5_event_harm_max <= 1e-12
        and h0_event_harmful_count >= MIN_UNSCOPED_EVENT_HARMFUL_TARGETS
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": plan["causal_hypothesis"],
        "scientific_role": plan["scientific_role"],
        "source_failure_pattern_card": plan["source_failure_pattern_card"],
        "feedback_budget_contract": plan["feedback_budget_contract"],
        "plan_exposure": {
            "context_exposure": plan["context_exposure"],
            "outcome_exposure_before_evaluate": plan["outcome_exposure"],
            "outcome_exposure_after_evaluate": "EXPOSED",
        },
        "dataset_evidence": rows,
        "overall": {
            "target_dataset_count": len(rows),
            "scope_update_realized_all_targets": realization_all,
            "positive_artifact_target_count": artifact_positive_count,
            "A3_macro_adapt_auc": a3_macro,
            "A5_macro_adapt_auc": a5_macro,
            "A5_minus_A3_macro_adapt_auc": a5_macro - a3_macro,
            "A4_macro_adapt_auc": sum(
                row["A4_source_only"]["adapt_auc"] for row in rows
            )
            / len(rows),
            "A5_minus_A4_macro_adapt_auc": a5_macro
            - sum(row["A4_source_only"]["adapt_auc"] for row in rows) / len(rows),
            "negative_transfer_target_count": negative_count,
            "A5_event_harm_max": a5_event_harm_max,
            "H0_unscoped_event_harmful_target_count": h0_event_harmful_count,
            "H0_unscoped_event_harm_macro": h0_event_harm_macro,
            "frozen_gate_pass": gate_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "CONTROLLED_INTEGRATED_HARNESS_EVOLUTION_PASS"
            if gate_pass
            else "CONTROLLED_INTEGRATED_HARNESS_EVOLUTION_FAIL"
        ),
        "official_target_test_outcome_opened_once": True,
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
        "formal_natural_capability_promotion": False,
        "paper_fresh_natural_transfer_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": (
            "Treat the controlled Harness-evolution mechanism as established and move to one "
            "natural Workflow family; do not add another controlled Proxy diagnostic."
            if gate_pass
            else "Localize the first failed gate once, then close or materially redesign this "
            "controlled family without tuning these target outcomes."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output", type=Path)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path)
    evaluate_parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        payload = build_plan(root)
        output = args.output or root / DEFAULT_PLAN_PATH
    else:
        plan_path = args.plan or root / DEFAULT_PLAN_PATH
        payload = evaluate(root, _read_object(plan_path))
        output = args.output or root / DEFAULT_REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(payload.get("verdict", payload.get("outcome_exposure")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
