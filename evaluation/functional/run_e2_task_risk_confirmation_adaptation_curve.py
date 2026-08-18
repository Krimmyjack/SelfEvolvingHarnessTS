"""Test Source proposal credit plus Target full-confirmation feedback.

W48 retained a useful Program-specific risk Witness.  W49 showed that a
first-order Ridge action-credit signal is useful for prioritization but unsafe
as a permanent veto, and W50a rejected a numerically brittle curvature repair.
This runner therefore changes only the feedback policy: positive Source credit
may execute at budget zero, while non-positive credit requests a full Support
counterfactual instead of permanently suppressing the Program.

Four new official UCR TRAIN splits are used to freeze the Source decisions.
Their TEST splits are opened once by ``evaluate``.  A3 and A5 receive the same
ordered full-Support feedback budget.  This is a controlled injected mechanism,
not natural-defect promotion, and the original UCI target remains unopened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-task-risk-confirmation-adaptation-curve/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_task_risk_confirmation_adaptation_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_risk_confirmation_adaptation_report.json"
)
W48_REPORT_PATH = (
    "artifacts/functional/e2/source_task_context_label_evidence_witness_report.json"
)
W49_REPORT_PATH = (
    "artifacts/functional/e2/source_task_risk_action_credit_transfer_report.json"
)
W50A_REPORT_PATH = (
    "artifacts/functional/e2/source_curvature_corrected_action_credit_report.json"
)
DATA_DIR = "data/ucr_task_context"
TARGET_DATASETS = ("Herring", "Ham", "FreezerSmallTrain", "Strawberry")
CONDITIONS = ("fit_only_artifact", "stable_task_event")
FEEDBACK_ORDERS = (
    ("fit_only_artifact", "stable_task_event"),
    ("stable_task_event", "fit_only_artifact"),
)
EXECUTE = "EXECUTE_BOUND_REPAIR"
ABSTAIN = "ABSTAIN_KEEP_INCUMBENT"
REQUEST = "REQUEST_FULL_CONFIRMATION"
PROXY_THRESHOLD = 0.0


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _source_state(condition: dict[str, Any]) -> tuple[str, list[str]]:
    """Compile the frozen W48/W49 evidence into a three-way fast-path state."""

    if condition["risk_decision"] != EXECUTE:
        return ABSTAIN, ["risk_witness_contraindicates_program"]
    credit = float(condition["action_credit"]["group"]["proxy_credit"])
    if credit > PROXY_THRESHOLD:
        return EXECUTE, ["eligible_and_positive_proxy_priority"]
    return REQUEST, ["eligible_but_proxy_requires_full_confirmation"]


def _policy_result(
    states: dict[str, str], conditions: dict[str, Any], split: str
) -> dict[str, float]:
    utilities: list[float] = []
    harms: list[float] = []
    for condition_name in CONDITIONS:
        row = conditions[condition_name]
        incumbent = float(row["incumbent"][f"{split}_accuracy"])
        action = float(row["action"][f"{split}_accuracy"])
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
    order: tuple[str, str],
    conditions: dict[str, Any],
) -> list[dict[str, Any]]:
    states = dict(initial)
    curve: list[dict[str, Any]] = []
    for budget in range(3):
        curve.append(
            {
                "budget": budget,
                "states": dict(states),
                **_policy_result(states, conditions, "query"),
            }
        )
        if budget < 2:
            observed = order[budget]
            support_gain = float(conditions[observed]["forced_support_gain"])
            states[observed] = EXECUTE if support_gain > 0.0 else ABSTAIN
            curve[-1]["next_full_confirmation"] = {
                "condition": observed,
                "support_gain": support_gain,
                "compiled_state": states[observed],
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


def _build_target_plan(root: Path, dataset: str) -> dict[str, Any]:
    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _build_train_plan,
        _helpers,
    )

    task_context, helpers = _helpers()
    row = _build_train_plan(
        np,
        root=root,
        dataset=dataset,
        task_context=task_context,
        helpers=helpers,
    )
    for condition_name in CONDITIONS:
        state, reasons = _source_state(row["conditions"][condition_name])
        row["conditions"][condition_name]["source_initial_state"] = state
        row["conditions"][condition_name]["source_state_reasons"] = reasons
    return row


def build_plan(root: Path) -> dict[str, Any]:
    w48 = _read_object(root / W48_REPORT_PATH)
    w49 = _read_object(root / W49_REPORT_PATH)
    w50a = _read_object(root / W50A_REPORT_PATH)
    if w48.get("verdict") != "CONTROLLED_NONORACLE_TASK_RISK_WITNESS_PASS":
        raise ValueError("W48 risk Witness is unavailable")
    if w49.get("verdict") != "CONTROLLED_ACTION_CREDIT_TRANSFER_FAIL":
        raise ValueError("W49 must have rejected first-order hard veto")
    if w50a.get("verdict") != "LOW_RANK_CURVATURE_CALIBRATION_FAIL":
        raise ValueError("W50a must have closed the curvature-proxy branch")

    plans = [_build_target_plan(root, dataset) for dataset in TARGET_DATASETS]
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "A Source-compiled risk-plus-priority Skill lowers equal-budget Target "
            "adaptation regret when uncertain actions request full Support confirmation "
            "instead of being permanently vetoed by a local valuation proxy."
        ),
        "scientific_role": "controlled Source-initialized confirmation adaptation",
        "context_exposure": "INSTANCE_SEEN_TRAIN_ONLY",
        "outcome_exposure": "SEALED",
        "frozen_branch_evidence": {
            "risk_witness": w48["verdict"],
            "first_order_hard_veto": w49["verdict"],
            "curvature_proxy": w50a["verdict"],
        },
        "target_train_plans": plans,
        "feedback_policy": {
            "A3": "blank states; full Support confirmation updates one condition per budget",
            "A4": "fixed Source risk-plus-priority states; no Target revision",
            "A5": "Source states; same ordered full Support confirmations as A3",
            "proxy_positive": EXECUTE,
            "proxy_nonpositive": REQUEST,
            "risk_contraindicated": ABSTAIN,
            "full_confirmation_rule": "execute iff full-refit Support accuracy gain > 0",
            "budgets": [0, 1, 2],
            "orders": [list(order) for order in FEEDBACK_ORDERS],
        },
        "frozen_success_gate": {
            "macro_A5_adapt_auc_strictly_greater_than_A3": True,
            "negative_transfer_target_count": 0,
            "A5_event_harm_max": 0.0,
            "unscoped_event_harm_strictly_positive": True,
        },
        "target_test_values_or_labels_read": False,
        "selection_used_target_program_or_consumer_outcome": False,
        "persistent_memory_built": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "The target backgrounds and labels are real UCR data, but the local mechanism "
            "is injected. This tests Harness feedback roles, not natural Capability promotion."
        ),
    }


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
        raise ValueError("W50b plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("W50b target TEST outcomes were not sealed")
    planned = {str(row["dataset"]): row for row in plan["target_train_plans"]}
    task_context, helpers = _helpers()
    dataset_rows: list[dict[str, Any]] = []
    fit_count = 0

    for dataset in TARGET_DATASETS:
        current = _build_target_plan(root, dataset)
        for condition_name in CONDITIONS:
            old = planned[dataset]["conditions"][condition_name]
            new = current["conditions"][condition_name]
            if (
                old["source_initial_state"] != new["source_initial_state"]
                or old["observation"]["nodes"] != new["observation"]["nodes"]
            ):
                raise ValueError(f"planned TRAIN state changed: {dataset}/{condition_name}")

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
            localization = _localization_evaluation(list(nodes), positions)
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
                "source_initial_state": condition_plan["source_initial_state"],
                "source_state_reasons": condition_plan["source_state_reasons"],
                "risk_decision": condition_plan["risk_decision"],
                "first_order_proxy_credit": float(
                    condition_plan["action_credit"]["group"]["proxy_credit"]
                ),
                "localization": localization,
                "program_modification": modification,
                "incumbent": incumbent,
                "action": action,
                "forced_support_gain": action["support_accuracy"]
                - incumbent["support_accuracy"],
                "forced_query_gain": action["query_accuracy"]
                - incumbent["query_accuracy"],
            }

        blank = {condition: REQUEST for condition in CONDITIONS}
        source = {
            condition: str(conditions[condition]["source_initial_state"])
            for condition in CONDITIONS
        }
        a3_orders = [
            _adaptation_curve(initial=blank, order=order, conditions=conditions)
            for order in FEEDBACK_ORDERS
        ]
        a5_orders = [
            _adaptation_curve(initial=source, order=order, conditions=conditions)
            for order in FEEDBACK_ORDERS
        ]
        a3 = _average_curves(a3_orders)
        a5 = _average_curves(a5_orders)
        a4_point = _policy_result(source, conditions, "query")
        a4 = [{"budget": budget, **a4_point} for budget in range(3)]
        risk_only_states = {
            condition: (
                EXECUTE if conditions[condition]["risk_decision"] == EXECUTE else ABSTAIN
            )
            for condition in CONDITIONS
        }
        risk_only_point = _policy_result(risk_only_states, conditions, "query")
        unscoped_point = _policy_result(
            {condition: EXECUTE for condition in CONDITIONS}, conditions, "query"
        )
        a3_auc = _adapt_auc(a3)
        a5_auc = _adapt_auc(a5)
        dataset_rows.append(
            {
                "dataset": dataset,
                "context_or_decision_used_dataset_id": False,
                "conditions": conditions,
                "feedback_orders": [list(order) for order in FEEDBACK_ORDERS],
                "A3_target_only": {
                    "per_order": a3_orders,
                    "mean_curve": a3,
                    "adapt_auc": a3_auc,
                },
                "A4_source_only": {
                    "mean_curve": a4,
                    "adapt_auc": _adapt_auc(a4),
                },
                "A5_source_plus_target": {
                    "per_order": a5_orders,
                    "mean_curve": a5,
                    "adapt_auc": a5_auc,
                },
                "risk_only_source": risk_only_point,
                "unscoped_forced_execution": unscoped_point,
                "A5_minus_A3_adapt_auc": a5_auc - a3_auc,
            }
        )

    a3_macro = sum(row["A3_target_only"]["adapt_auc"] for row in dataset_rows) / len(
        dataset_rows
    )
    a5_macro = sum(
        row["A5_source_plus_target"]["adapt_auc"] for row in dataset_rows
    ) / len(dataset_rows)
    negative_count = sum(row["A5_minus_A3_adapt_auc"] < -1e-12 for row in dataset_rows)
    a5_event_harm_max = max(
        point["event_harm"]
        for row in dataset_rows
        for point in row["A5_source_plus_target"]["mean_curve"]
    )
    unscoped_event_harm = sum(
        row["unscoped_forced_execution"]["event_harm"] for row in dataset_rows
    ) / len(dataset_rows)
    gate_pass = bool(
        a5_macro > a3_macro
        and negative_count == 0
        and a5_event_harm_max <= 1e-12
        and unscoped_event_harm > 0.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": plan["causal_hypothesis"],
        "scientific_role": plan["scientific_role"],
        "plan_exposure": {
            "context_exposure": plan["context_exposure"],
            "outcome_exposure_before_evaluate": plan["outcome_exposure"],
            "outcome_exposure_after_evaluate": "EXPOSED",
        },
        "dataset_evidence": dataset_rows,
        "overall": {
            "target_dataset_count": len(dataset_rows),
            "A3_macro_adapt_auc": a3_macro,
            "A5_macro_adapt_auc": a5_macro,
            "A5_minus_A3_macro_adapt_auc": a5_macro - a3_macro,
            "negative_transfer_target_count": negative_count,
            "A5_event_harm_max": a5_event_harm_max,
            "unscoped_event_harm_macro": unscoped_event_harm,
            "frozen_gate_pass": gate_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "CONTROLLED_RISK_CONFIRMATION_A5_VS_A3_PASS"
            if gate_pass
            else "CONTROLLED_RISK_CONFIRMATION_A5_VS_A3_FAIL"
        ),
        "official_target_test_outcome_opened_once": True,
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
        "formal_natural_capability_promotion": False,
        "paper_fresh_transfer_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": (
            "Retain the hierarchical proxy role and seek one natural Context-conditioned "
            "Capability; do not build a valuation or Memory platform."
            if gate_pass
            else "Demote the proxy to proposal ranking only and localize whether Source "
            "initialization or Support feedback caused the failed adaptation curve."
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
