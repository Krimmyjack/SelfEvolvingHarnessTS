"""Transfer the W48 Task-risk Skill with a one-solve action-credit proxy.

W48 showed that a legal fit/support Witness can distinguish a fit-only artifact
from stable task evidence without receiving the instance answer in TaskContext.
It also retained one small harmful artifact action on FordA.  This runner uses
the exposed W48 episodes only to freeze a zero-threshold, first-order Ridge
support-loss credit.  It then plans decisions from TRAIN data on four new UCR
targets and opens their official TEST splits once for full-refit confirmation.

The proxy is proposal/credit evidence, not final Utility.  No persistent Memory,
Dataset-ID rule, UCI target access, or natural-defect promotion is introduced.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-task-risk-action-credit-transfer/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_task_risk_action_credit_transfer_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_risk_action_credit_transfer_report.json"
)
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_task_context_label_evidence_witness_report.json"
)
SOURCE_DATASETS = ("Coffee", "ECG200", "FordA", "GunPoint")
TARGET_DATASETS = ("Wafer", "ECGFiveDays", "TwoLeadECG", "BeetleFly")
CONDITIONS = ("fit_only_artifact", "stable_task_event")
RIDGE_ALPHA = 1.0
PROXY_EXECUTION_THRESHOLD = 0.0
MIN_TARGET_SIGN_AGREEMENT = 0.75
MIN_POSITIVE_GAIN_RETENTION = 0.75


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _inverse_product(np: Any, design: Any, right: Any, alpha: float) -> Any:
    """Return ``(Z'Z + alpha I)^-1 right`` using the smaller system."""

    z = np.asarray(design, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    rows, columns = z.shape
    if columns <= rows:
        system = z.T @ z + alpha * np.eye(columns, dtype=np.float64)
        return np.linalg.solve(system, rhs)
    dual = z @ z.T + alpha * np.eye(rows, dtype=np.float64)
    return (rhs - z.T @ np.linalg.solve(dual, z @ rhs)) / alpha


def _first_order_action_credits(
    np: Any,
    *,
    fit_values: Any,
    fit_labels: Any,
    support_values: Any,
    support_labels: Any,
    action_values: list[Any],
    features: Any,
) -> list[dict[str, float]]:
    """Estimate multiple actions from one reference and one multi-RHS solve."""

    fit_features = features(np, fit_values)
    support_features = features(np, support_values)
    design = np.column_stack(
        (fit_features, np.ones(fit_features.shape[0], dtype=np.float64))
    )
    support_design = np.column_stack(
        (support_features, np.ones(support_features.shape[0], dtype=np.float64))
    )
    targets = np.where(np.asarray(fit_labels) == 1, 1.0, -1.0)
    support_targets = np.where(np.asarray(support_labels) == 1, 1.0, -1.0)
    coefficient = _inverse_product(
        np, design, design.T @ targets, RIDGE_ALPHA
    )
    residual = design @ coefficient - targets
    baseline_score = support_design @ coefficient
    baseline_loss = float(np.mean((baseline_score - support_targets) ** 2))
    normal_changes: list[Any] = []
    for values in action_values:
        action_features = features(np, values)
        delta_design = np.column_stack(
            (
                action_features - fit_features,
                np.zeros(fit_features.shape[0], dtype=np.float64),
            )
        )
        normal_changes.append(
            delta_design.T @ residual + design.T @ (delta_design @ coefficient)
        )
    changes = -_inverse_product(
        np, design, np.column_stack(normal_changes), RIDGE_ALPHA
    )
    results: list[dict[str, float]] = []
    for index in range(len(action_values)):
        coefficient_change = changes[:, index]
        proxy_action_score = support_design @ (coefficient + coefficient_change)
        proxy_action_loss = float(
            np.mean((proxy_action_score - support_targets) ** 2)
        )
        if not all(
            np.isfinite(value)
            for value in (
                baseline_loss,
                proxy_action_loss,
                float(np.linalg.norm(coefficient_change)),
            )
        ):
            raise RuntimeError("non-finite first-order action credit")
        results.append(
            {
                "proxy_credit": baseline_loss - proxy_action_loss,
                "baseline_support_squared_loss": baseline_loss,
                "proxy_action_support_squared_loss": proxy_action_loss,
                "coefficient_change_l2": float(np.linalg.norm(coefficient_change)),
                "linear_system_solve_count_for_panel": 2,
                "action_model_refit_count": 0,
            }
        )
    return results


def _proxy_panel(
    np: Any,
    *,
    fit_values: Any,
    fit_labels: Any,
    support_values: Any,
    support_labels: Any,
    nodes: tuple[int, ...],
    apply_program: Any,
    features: Any,
) -> tuple[Any, dict[str, Any]]:
    repaired, modification = apply_program(
        np,
        fit_values,
        positions=nodes,
        window_length=fit_values.shape[1],
    )
    action_panel = [repaired]
    for node in nodes:
        node_values, _ = apply_program(
            np,
            fit_values,
            positions=(node,),
            window_length=fit_values.shape[1],
        )
        action_panel.append(node_values)
    credits = _first_order_action_credits(
        np,
        fit_values=fit_values,
        fit_labels=fit_labels,
        support_values=support_values,
        support_labels=support_labels,
        action_values=action_panel,
        features=features,
    )
    group = credits[0]
    node_credits = [
        {"node": node, "proxy_credit": credit["proxy_credit"]}
        for node, credit in zip(nodes, credits[1:])
    ]
    return repaired, {
        "proxy_id": "first-order-ridge-support-loss-action-credit/1",
        "consumer_conditioned": True,
        "final_utility": False,
        "query_values_or_labels_used": False,
        "group": group,
        "per_node": node_credits,
        "program_modification": modification,
    }


def _credit_decision(risk_decision: str, proxy: dict[str, Any]) -> tuple[str, list[str]]:
    if risk_decision != "EXECUTE_BOUND_REPAIR":
        return "ABSTAIN_KEEP_INCUMBENT", ["risk_witness_blocks_program"]
    credit = float(proxy["group"]["proxy_credit"])
    if credit > PROXY_EXECUTION_THRESHOLD:
        return "EXECUTE_BOUND_REPAIR", ["eligible_and_positive_action_credit"]
    return "ABSTAIN_KEEP_INCUMBENT", ["eligible_but_nonpositive_action_credit"]


def _credit_augmented_text(base_text: str, proxy: dict[str, Any]) -> str:
    node_rows = proxy["per_node"]
    text = "\n".join(
        (
            base_text,
            "ACTION_CREDIT_CONTEXT",
            "proxy=first_order_ridge_support_squared_loss",
            f"group_proxy_credit={proxy['group']['proxy_credit']:.12f}",
            "per_node_proxy_credit="
            + ",".join(
                f"{int(row['node'])}:{float(row['proxy_credit']):.12f}"
                for row in node_rows
            ),
            "proxy_semantics=proposal_credit_not_final_utility",
        )
    )
    forbidden = (
        "query_accuracy",
        "query_gain",
        "correct_action",
        "target_dataset",
        "fit_only_artifact",
        "stable_task_event",
    ) + tuple(name.lower() for name in SOURCE_DATASETS + TARGET_DATASETS)
    lowered = text.lower()
    leaked = [token for token in forbidden if token.lower() in lowered]
    if leaked:
        raise AssertionError(f"credit decision input leaks outcome fields: {leaked}")
    return text


def _condition_inputs(
    np: Any,
    *,
    base_fit: Any,
    fit_labels: Any,
    base_support: Any,
    support_labels: Any,
    positions: tuple[int, ...],
    condition: str,
    inject: Any,
) -> tuple[Any, Any]:
    fit_values = inject(np, base_fit, fit_labels, positions)
    support_values = (
        base_support.copy()
        if condition == "fit_only_artifact"
        else inject(np, base_support, support_labels, positions)
    )
    return fit_values, support_values


def _build_train_plan(
    np: Any,
    *,
    root: Path,
    dataset: str,
    task_context: Any,
    helpers: dict[str, Any],
) -> dict[str, Any]:
    archive = root / "data/ucr_task_context" / f"{dataset}.zip"
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    fit_indices, support_indices = helpers["split"](np, train_labels)
    base_fit = train_values[fit_indices]
    fit_labels = train_labels[fit_indices]
    base_support = train_values[support_indices]
    support_labels = train_labels[support_indices]
    positions = helpers["positions"](train_values.shape[1])
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        fit_values, support_values = _condition_inputs(
            np,
            base_fit=base_fit,
            fit_labels=fit_labels,
            base_support=base_support,
            support_labels=support_labels,
            positions=positions,
            condition=condition,
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
        repaired, proxy = _proxy_panel(
            np,
            fit_values=fit_values,
            fit_labels=fit_labels,
            support_values=support_values,
            support_labels=support_labels,
            nodes=nodes,
            apply_program=helpers["apply_program"],
            features=helpers["features"],
        )
        del repaired
        final_decision, final_reasons = _credit_decision(risk_decision, proxy)
        base_text = helpers["decision_text"](task_context, witness)
        conditions[condition] = {
            "observation": observation,
            "witness": witness,
            "risk_decision": risk_decision,
            "risk_reasons": risk_reasons,
            "action_credit": proxy,
            "decision_input_text": _credit_augmented_text(base_text, proxy),
            "decision_input_contains_instance_answer": False,
            "final_decision": final_decision,
            "final_reasons": final_reasons,
        }
    return {
        "dataset": dataset,
        "archive": f"data/ucr_task_context/{dataset}.zip",
        "official_train_count": int(train_values.shape[0]),
        "series_length": int(train_values.shape[1]),
        "fit_count": int(fit_indices.size),
        "support_count": int(support_indices.size),
        "conditions": conditions,
    }


def _helpers() -> tuple[Any, dict[str, Any]]:
    from SelfEvolvingHarnessTS.contracts.task import (
        classification_local_event_task_quality_contract_v1,
        classification_task_context_v1,
        classification_task_spec_v1,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        _apply_bound_impulse_oracle,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _bound_positions,
        _build_witness,
        _compile_decision,
        _decision_input,
        _features,
        _inject,
        _load_split,
        _split_fit_support,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_scoped_impulse_skill_transfer import (
        _observe_class_conditioned_impulse_topology,
        _rolling_median,
    )

    task_context = classification_task_context_v1(
        task_spec=classification_task_spec_v1(
            downstream_model_class="ridge-raw-plus-difference-v1"
        ),
        quality_contract=classification_local_event_task_quality_contract_v1(),
    )
    return task_context, {
        "load": _load_split,
        "split": _split_fit_support,
        "positions": _bound_positions,
        "inject": _inject,
        "features": _features,
        "witness": _build_witness,
        "risk_decision": _compile_decision,
        "decision_text": _decision_input,
        "observe": _observe_class_conditioned_impulse_topology,
        "rolling_median": _rolling_median,
        "apply_program": _apply_bound_impulse_oracle,
    }


def build_plan(root: Path) -> dict[str, Any]:
    import numpy as np

    source_report = _read_object(root / SOURCE_REPORT_PATH)
    if source_report.get("verdict") != "CONTROLLED_NONORACLE_TASK_RISK_WITNESS_PASS":
        raise ValueError("W48 Source risk-Witness evidence is unavailable")
    source_query_gain = {
        str(row["dataset"]): float(
            row["conditions"]["fit_only_artifact"]["forced_query_gain"]
        )
        for row in source_report["dataset_evidence"]
    }
    task_context, helpers = _helpers()
    source_calibration: list[dict[str, Any]] = []
    for dataset in SOURCE_DATASETS:
        row = _build_train_plan(
            np, root=root, dataset=dataset, task_context=task_context, helpers=helpers
        )
        credit = float(
            row["conditions"]["fit_only_artifact"]["action_credit"]["group"][
                "proxy_credit"
            ]
        )
        actual = source_query_gain[dataset]
        source_calibration.append(
            {
                "dataset": dataset,
                "proxy_credit": credit,
                "exposed_w48_query_gain": actual,
                "sign_agreement": (credit > 0.0) == (actual > 0.0),
            }
        )
    if not all(bool(row["sign_agreement"]) for row in source_calibration):
        raise ValueError("W48 Source proxy calibration does not support transfer")

    target_plans = [
        _build_train_plan(
            np, root=root, dataset=dataset, task_context=task_context, helpers=helpers
        )
        for dataset in TARGET_DATASETS
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "After a Program-specific risk Witness establishes eligibility, a one-solve "
            "Consumer-conditioned action-credit proxy reduces harmful execution on new "
            "datasets while retaining most positive full-refit utility."
        ),
        "scientific_role": "controlled Source-calibrated action-credit transfer",
        "context_exposure": "INSTANCE_SEEN_TRAIN_ONLY",
        "outcome_exposure": "SEALED",
        "source_calibration": source_calibration,
        "source_sign_agreement": 1.0,
        "frozen_proxy_threshold": PROXY_EXECUTION_THRESHOLD,
        "target_train_plans": target_plans,
        "target_test_values_or_labels_read": False,
        "selection_used_target_program_outcome": False,
        "persistent_memory_built": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "The Source and Target local mechanisms are controlled injections on real UCR "
            "backgrounds; this validates proxy feedback behavior, not natural Capability promotion."
        ),
    }


def _fit_readout(
    np: Any,
    RidgeClassifier: Any,
    features: Any,
    train_values: Any,
    train_labels: Any,
    support_values: Any,
    support_labels: Any,
    query_values: Any,
    query_labels: Any,
) -> dict[str, float]:
    model = RidgeClassifier(alpha=RIDGE_ALPHA)
    model.fit(features(np, train_values), train_labels)
    return {
        "support_accuracy": float(
            np.mean(model.predict(features(np, support_values)) == support_labels)
        ),
        "query_accuracy": float(
            np.mean(model.predict(features(np, query_values)) == query_labels)
        ),
    }


def evaluate(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("W49 plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("W49 target TEST outcomes were not sealed")
    task_context, helpers = _helpers()
    planned = {str(row["dataset"]): row for row in plan["target_train_plans"]}
    dataset_rows: list[dict[str, Any]] = []
    fit_count = 0
    for dataset in TARGET_DATASETS:
        current = _build_train_plan(
            np, root=root, dataset=dataset, task_context=task_context, helpers=helpers
        )
        for condition in CONDITIONS:
            if (
                current["conditions"][condition]["final_decision"]
                != planned[dataset]["conditions"][condition]["final_decision"]
                or current["conditions"][condition]["observation"]["nodes"]
                != planned[dataset]["conditions"][condition]["observation"]["nodes"]
            ):
                raise ValueError(f"planned TRAIN decision changed: {dataset}/{condition}")

        archive = root / "data/ucr_task_context" / f"{dataset}.zip"
        train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
        query_values, query_labels = helpers["load"](np, archive, dataset, "TEST")
        fit_indices, support_indices = helpers["split"](np, train_labels)
        base_fit = train_values[fit_indices]
        fit_labels = train_labels[fit_indices]
        base_support = train_values[support_indices]
        support_labels = train_labels[support_indices]
        positions = helpers["positions"](train_values.shape[1])
        conditions: dict[str, Any] = {}
        for condition in CONDITIONS:
            fit_values, support_values = _condition_inputs(
                np,
                base_fit=base_fit,
                fit_labels=fit_labels,
                base_support=base_support,
                support_labels=support_labels,
                positions=positions,
                condition=condition,
                inject=helpers["inject"],
            )
            condition_query = (
                query_values.copy()
                if condition == "fit_only_artifact"
                else helpers["inject"](np, query_values, query_labels, positions)
            )
            nodes = tuple(
                int(node)
                for node in planned[dataset]["conditions"][condition]["observation"][
                    "nodes"
                ]
            )
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
            actual_gain = action["query_accuracy"] - incumbent["query_accuracy"]
            risk_decision = planned[dataset]["conditions"][condition]["risk_decision"]
            final_decision = planned[dataset]["conditions"][condition]["final_decision"]
            risk_selected = action if risk_decision == "EXECUTE_BOUND_REPAIR" else incumbent
            credit_selected = action if final_decision == "EXECUTE_BOUND_REPAIR" else incumbent
            proxy_credit = float(
                planned[dataset]["conditions"][condition]["action_credit"]["group"][
                    "proxy_credit"
                ]
            )
            conditions[condition] = {
                "planned_risk_decision": risk_decision,
                "planned_final_decision": final_decision,
                "planned_proxy_credit": proxy_credit,
                "proxy_predicts_positive_action": proxy_credit > 0.0,
                "program_modification": modification,
                "incumbent": incumbent,
                "forced_action": action,
                "actual_query_gain": actual_gain,
                "proxy_actual_sign_agreement": (proxy_credit > 0.0)
                == (actual_gain > 0.0),
                "risk_only_policy_query_gain": risk_selected["query_accuracy"]
                - incumbent["query_accuracy"],
                "credit_scoped_policy_query_gain": credit_selected["query_accuracy"]
                - incumbent["query_accuracy"],
                "credit_scoped_policy_query_harm": max(
                    0.0,
                    incumbent["query_accuracy"] - credit_selected["query_accuracy"],
                ),
            }
        dataset_rows.append({"dataset": dataset, "conditions": conditions})

    artifact = [row["conditions"]["fit_only_artifact"] for row in dataset_rows]
    event = [row["conditions"]["stable_task_event"] for row in dataset_rows]
    actual = [float(row["actual_query_gain"]) for row in artifact]
    proxy = [float(row["planned_proxy_credit"]) for row in artifact]
    risk_gains = [float(row["risk_only_policy_query_gain"]) for row in artifact]
    credit_gains = [float(row["credit_scoped_policy_query_gain"]) for row in artifact]
    risk_harmful = sum(gain < 0.0 for gain in risk_gains)
    credit_harmful = sum(gain < 0.0 for gain in credit_gains)
    sign_agreement = sum((p > 0.0) == (a > 0.0) for p, a in zip(proxy, actual)) / len(actual)
    positive_total = sum(max(0.0, gain) for gain in actual)
    retained = (
        sum(max(0.0, gain) for gain in credit_gains) / positive_total
        if positive_total > 0.0
        else 0.0
    )
    event_harm = [float(row["credit_scoped_policy_query_harm"]) for row in event]
    target_pass = (
        sign_agreement >= MIN_TARGET_SIGN_AGREEMENT
        and credit_harmful <= risk_harmful
        and retained >= MIN_POSITIVE_GAIN_RETENTION
        and max(event_harm) <= 1e-12
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
        "source_calibration": plan["source_calibration"],
        "dataset_evidence": dataset_rows,
        "overall": {
            "target_dataset_count": len(dataset_rows),
            "artifact_actual_query_gains": actual,
            "artifact_proxy_credits": proxy,
            "target_proxy_actual_sign_agreement": sign_agreement,
            "risk_only_artifact_macro_gain": sum(risk_gains) / len(risk_gains),
            "credit_scoped_artifact_macro_gain": sum(credit_gains) / len(credit_gains),
            "risk_only_harmful_artifact_count": risk_harmful,
            "credit_scoped_harmful_artifact_count": credit_harmful,
            "positive_artifact_gain_retention": retained,
            "credit_scoped_event_harms": event_harm,
            "target_gate_pass": target_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "CONTROLLED_ACTION_CREDIT_TRANSFER_PASS"
            if target_pass
            else "CONTROLLED_ACTION_CREDIT_TRANSFER_FAIL"
        ),
        "official_target_test_outcome_opened_once": True,
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
        "formal_natural_capability_promotion": False,
        "paper_fresh_transfer_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": (
            "Integrate the verified risk-plus-credit Context into an equal-budget target "
            "adaptation slice or a natural defect family; do not build a Memory platform."
            if target_pass
            else "Preserve the target failures and localize proxy, Program, or support-to-query mismatch without tuning these TEST splits."
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
