"""Test first-order action credit as proposal ordering, never final Utility.

W50b showed that even a positive first-order credit can cause harmful B0
execution, while one exact Support counterfactual removes that mistake.  This
runner freezes a safer role on four new UCR targets: the existing risk Witness
defines the eligible condition, first-order credits rank one group and four
single-node Workflow candidates, and full Support refits alone select an action.

A3 is the exact expectation over target-only random candidate orders.  A5 uses
the Source credit order with the same number of full confirmations.  Official
TEST data are opened once by ``evaluate``.  No UCI target, persistent Memory,
new proxy, or natural-defect claim is involved.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-action-credit-candidate-ordering/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_action_credit_candidate_ordering_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_action_credit_candidate_ordering_report.json"
)
W50B_REPORT_PATH = (
    "artifacts/functional/e2/source_task_risk_confirmation_adaptation_report.json"
)
DATA_DIR = "data/ucr_task_context"
TARGET_DATASETS = (
    "ShapeletSim",
    "Wine",
    "Earthquakes",
    "DistalPhalanxOutlineCorrect",
)
ARTIFACT = "fit_only_artifact"
EVENT = "stable_task_event"
EXECUTE = "EXECUTE_BOUND_REPAIR"
ABSTAIN = "ABSTAIN_KEEP_INCUMBENT"


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _candidate_rows(condition: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [int(node) for node in condition["observation"]["nodes"]]
    proxy = condition["action_credit"]
    node_credit = {
        int(row["node"]): float(row["proxy_credit"])
        for row in proxy["per_node"]
    }
    if set(node_credit) != set(nodes):
        raise ValueError("proxy nodes do not match the observed Program nodes")
    rows = [
        {
            "candidate_id": "group",
            "nodes": nodes,
            "first_order_proxy_credit": float(proxy["group"]["proxy_credit"]),
        }
    ]
    rows.extend(
        {
            "candidate_id": f"node:{node}",
            "nodes": [node],
            "first_order_proxy_credit": node_credit[node],
        }
        for node in nodes
    )
    return rows


def _source_order(candidates: list[dict[str, Any]]) -> list[str]:
    return [
        str(row["candidate_id"])
        for row in sorted(
            candidates,
            key=lambda row: (
                -float(row["first_order_proxy_credit"]),
                str(row["candidate_id"]),
            ),
        )
    ]


def _build_target_plan(root: Path, dataset: str) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_confirmation_adaptation_curve import (
        _build_target_plan as _w50b_train_plan,
    )

    row = _w50b_train_plan(root, dataset)
    artifact = row["conditions"][ARTIFACT]
    event = row["conditions"][EVENT]
    candidates = _candidate_rows(artifact)
    row["candidate_search"] = {
        "risk_eligible_condition": ARTIFACT,
        "candidate_rows": candidates,
        "source_priority_order": _source_order(candidates),
        "target_only_reference": "uniform expectation over all 5! candidate orders",
        "all_actions_require_full_support_confirmation": True,
    }
    row["risk_scope_audit"] = {
        "artifact_decision": artifact["risk_decision"],
        "event_decision": event["risk_decision"],
    }
    return row


def build_plan(root: Path) -> dict[str, Any]:
    w50b = _read_object(root / W50B_REPORT_PATH)
    if w50b.get("verdict") != "CONTROLLED_RISK_CONFIRMATION_A5_VS_A3_FAIL":
        raise ValueError("W50b must have rejected unconfirmed Source execution")
    rows = [_build_target_plan(root, dataset) for dataset in TARGET_DATASETS]
    for row in rows:
        audit = row["risk_scope_audit"]
        if audit["artifact_decision"] != EXECUTE or audit["event_decision"] == EXECUTE:
            raise ValueError(f"risk scope is unresolved on {row['dataset']}")
        if len(row["candidate_search"]["candidate_rows"]) != 5:
            raise ValueError("W51 requires one group and four single-node candidates")
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "A Source action-credit proxy reduces the number of exact Target confirmations "
            "needed to find a useful Workflow when it ranks candidates but never directly "
            "authorizes execution."
        ),
        "scientific_role": "controlled Source-prioritized Workflow search",
        "context_exposure": "INSTANCE_SEEN_TRAIN_ONLY",
        "outcome_exposure": "SEALED",
        "frozen_failure_input": w50b["verdict"],
        "target_train_plans": rows,
        "frozen_search_protocol": {
            "candidate_menu": "group repair plus four single-node repairs",
            "A3": "risk-scoped target-only uniform random-order expectation",
            "A5": "same risk scope and confirmations; Source proxy-descending order",
            "selection_after_feedback": (
                "highest positive full-refit Support accuracy gain among confirmed candidates; "
                "candidate_id tie break"
            ),
            "budgets": [0, 1, 2, 3, 4, 5],
            "final_utility": "official TEST accuracy after selected full Consumer retraining",
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
            "Real UCR backgrounds support an injected local mechanism. This isolates the "
            "cost-reduction role of proxy feedback, not natural Capability transfer."
        ),
    }


def _best_confirmed(
    confirmed: list[str], candidates: dict[str, dict[str, Any]]
) -> str | None:
    positive = [
        candidate_id
        for candidate_id in confirmed
        if float(candidates[candidate_id]["forced_support_gain"]) > 0.0
    ]
    if not positive:
        return None
    return sorted(
        positive,
        key=lambda candidate_id: (
            -float(candidates[candidate_id]["forced_support_gain"]),
            candidate_id,
        ),
    )[0]


def _search_curve(
    order: tuple[str, ...] | list[str],
    candidates: dict[str, dict[str, Any]],
    *,
    artifact_incumbent: float,
    event_incumbent: float,
) -> list[dict[str, Any]]:
    confirmed: list[str] = []
    curve: list[dict[str, Any]] = []
    for budget in range(len(order) + 1):
        selected = _best_confirmed(confirmed, candidates)
        artifact_utility = (
            artifact_incumbent
            if selected is None
            else float(candidates[selected]["query_accuracy"])
        )
        artifact_harm = max(0.0, artifact_incumbent - artifact_utility)
        curve.append(
            {
                "budget": budget,
                "confirmed_candidates": list(confirmed),
                "selected_candidate": selected or "incumbent",
                "utility": (artifact_utility + event_incumbent) / 2.0,
                "harm": artifact_harm / 2.0,
                "event_harm": 0.0,
            }
        )
        if budget < len(order):
            confirmed.append(str(order[budget]))
    return curve


def _average_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    return [
        {
            "budget": budget,
            "utility": sum(curve[budget]["utility"] for curve in curves)
            / len(curves),
            "harm": sum(curve[budget]["harm"] for curve in curves) / len(curves),
            "event_harm": 0.0,
        }
        for budget in range(6)
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
        raise ValueError("W51 plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("W51 target TEST outcomes were not sealed")
    planned = {str(row["dataset"]): row for row in plan["target_train_plans"]}
    task_context, helpers = _helpers()
    del task_context
    dataset_rows: list[dict[str, Any]] = []
    fit_count = 0

    for dataset in TARGET_DATASETS:
        current = _build_target_plan(root, dataset)
        if (
            current["candidate_search"]["source_priority_order"]
            != planned[dataset]["candidate_search"]["source_priority_order"]
        ):
            raise ValueError(f"planned Source candidate order changed: {dataset}")
        archive = root / DATA_DIR / f"{dataset}.zip"
        train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
        query_values, query_labels = helpers["load"](np, archive, dataset, "TEST")
        fit_indices, support_indices = helpers["split"](np, train_labels)
        base_fit = train_values[fit_indices]
        fit_labels = train_labels[fit_indices]
        base_support = train_values[support_indices]
        support_labels = train_labels[support_indices]
        positions = helpers["positions"](train_values.shape[1])

        artifact_fit, artifact_support = _condition_inputs(
            np,
            base_fit=base_fit,
            fit_labels=fit_labels,
            base_support=base_support,
            support_labels=support_labels,
            positions=positions,
            condition=ARTIFACT,
            inject=helpers["inject"],
        )
        artifact_query = query_values.copy()
        artifact_incumbent = _fit_readout(
            np,
            RidgeClassifier,
            helpers["features"],
            artifact_fit,
            fit_labels,
            artifact_support,
            support_labels,
            artifact_query,
            query_labels,
        )
        fit_count += 1
        candidates: dict[str, dict[str, Any]] = {}
        planned_candidates = planned[dataset]["candidate_search"]["candidate_rows"]
        for candidate_plan in planned_candidates:
            candidate_id = str(candidate_plan["candidate_id"])
            nodes = tuple(int(node) for node in candidate_plan["nodes"])
            repaired, modification = helpers["apply_program"](
                np,
                artifact_fit,
                positions=nodes,
                window_length=train_values.shape[1],
            )
            result = _fit_readout(
                np,
                RidgeClassifier,
                helpers["features"],
                repaired,
                fit_labels,
                artifact_support,
                support_labels,
                artifact_query,
                query_labels,
            )
            fit_count += 1
            candidates[candidate_id] = {
                "nodes": list(nodes),
                "first_order_proxy_credit": float(
                    candidate_plan["first_order_proxy_credit"]
                ),
                "support_accuracy": result["support_accuracy"],
                "query_accuracy": result["query_accuracy"],
                "forced_support_gain": result["support_accuracy"]
                - artifact_incumbent["support_accuracy"],
                "forced_query_gain": result["query_accuracy"]
                - artifact_incumbent["query_accuracy"],
                "program_modification": modification,
            }

        event_fit, event_support = _condition_inputs(
            np,
            base_fit=base_fit,
            fit_labels=fit_labels,
            base_support=base_support,
            support_labels=support_labels,
            positions=positions,
            condition=EVENT,
            inject=helpers["inject"],
        )
        event_query = helpers["inject"](np, query_values, query_labels, positions)
        event_incumbent = _fit_readout(
            np,
            RidgeClassifier,
            helpers["features"],
            event_fit,
            fit_labels,
            event_support,
            support_labels,
            event_query,
            query_labels,
        )
        event_repaired, event_modification = helpers["apply_program"](
            np,
            event_fit,
            positions=tuple(int(node) for node in positions),
            window_length=train_values.shape[1],
        )
        event_action = _fit_readout(
            np,
            RidgeClassifier,
            helpers["features"],
            event_repaired,
            fit_labels,
            event_support,
            support_labels,
            event_query,
            query_labels,
        )
        fit_count += 2

        candidate_ids = tuple(sorted(candidates))
        all_orders = list(itertools.permutations(candidate_ids))
        a3_orders = [
            _search_curve(
                order,
                candidates,
                artifact_incumbent=artifact_incumbent["query_accuracy"],
                event_incumbent=event_incumbent["query_accuracy"],
            )
            for order in all_orders
        ]
        source_order = list(
            planned[dataset]["candidate_search"]["source_priority_order"]
        )
        a5_curve = _search_curve(
            source_order,
            candidates,
            artifact_incumbent=artifact_incumbent["query_accuracy"],
            event_incumbent=event_incumbent["query_accuracy"],
        )
        a3_curve = _average_curves(a3_orders)
        a3_auc = _adapt_auc(a3_curve)
        a5_auc = _adapt_auc(a5_curve)
        direct_id = source_order[0]
        direct_query = float(candidates[direct_id]["query_accuracy"])
        direct_harm = max(
            0.0, artifact_incumbent["query_accuracy"] - direct_query
        ) / 2.0
        unscoped_event_harm = max(
            0.0,
            event_incumbent["query_accuracy"] - event_action["query_accuracy"],
        )
        dataset_rows.append(
            {
                "dataset": dataset,
                "context_or_decision_used_dataset_id": False,
                "localization": _localization_evaluation(
                    [int(node) for node in positions], positions
                ),
                "artifact_incumbent": artifact_incumbent,
                "event_incumbent": event_incumbent,
                "candidate_evidence": candidates,
                "source_priority_order": source_order,
                "A3_target_only_random_order": {
                    "order_count": len(all_orders),
                    "mean_curve": a3_curve,
                    "adapt_auc": a3_auc,
                },
                "A4_unconfirmed_top_proxy": {
                    "candidate_id": direct_id,
                    "utility": (direct_query + event_incumbent["query_accuracy"])
                    / 2.0,
                    "harm": direct_harm,
                },
                "A5_source_order_plus_target_confirmation": {
                    "curve": a5_curve,
                    "adapt_auc": a5_auc,
                },
                "unscoped_event_group_repair": {
                    "query_gain": event_action["query_accuracy"]
                    - event_incumbent["query_accuracy"],
                    "query_harm": unscoped_event_harm,
                    "program_modification": event_modification,
                },
                "A5_minus_A3_adapt_auc": a5_auc - a3_auc,
            }
        )

    a3_macro = sum(
        row["A3_target_only_random_order"]["adapt_auc"] for row in dataset_rows
    ) / len(dataset_rows)
    a5_macro = sum(
        row["A5_source_order_plus_target_confirmation"]["adapt_auc"]
        for row in dataset_rows
    ) / len(dataset_rows)
    negative_count = sum(row["A5_minus_A3_adapt_auc"] < -1e-12 for row in dataset_rows)
    a5_event_harm_max = max(
        point["event_harm"]
        for row in dataset_rows
        for point in row["A5_source_order_plus_target_confirmation"]["curve"]
    )
    unscoped_event_harm = sum(
        row["unscoped_event_group_repair"]["query_harm"] for row in dataset_rows
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
            "CONTROLLED_ACTION_CREDIT_ORDERING_A5_VS_A3_PASS"
            if gate_pass
            else "CONTROLLED_ACTION_CREDIT_ORDERING_A5_VS_A3_FAIL"
        ),
        "official_target_test_outcome_opened_once": True,
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
        "formal_natural_capability_promotion": False,
        "paper_fresh_transfer_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": (
            "Keep valuation as Workflow proposal ordering and move this Context-Skill-feedback "
            "role into one natural candidate family."
            if gate_pass
            else "Close first-order proxy as a cross-dataset ordering mechanism for this "
            "Program family; retain exact Target confirmation and the risk Witness."
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
