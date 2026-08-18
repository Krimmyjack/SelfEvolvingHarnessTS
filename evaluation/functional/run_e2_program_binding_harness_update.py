"""Evaluate one controlled Program-Binding Harness update on new UCR targets.

After W52 established an Observation-plus-Scope update, this runner tests a
distinct controlled first fault: an H0 compiler that reuses a Source Program
instance with absolute positions.  It freezes one blind typed patch: bind the
unchanged center-excluded local-median Program to the current fit-series
observer nodes.  ``plan`` reads only official TRAIN splits;
``evaluate`` replays the frozen TRAIN plan before opening each official TEST
split once.  The artifact is controlled on real UCR backgrounds.  No persistent
Memory, valuation proxy, Dataset-ID decision, natural-defect claim, or original
UCI target is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-program-binding-harness-update/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_program_binding_harness_update_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_program_binding_harness_update_report.json"
)
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_integrated_context_harness_evolution_report.json"
)
DATA_DIR = "data/ucr_task_context"
TARGET_DATASETS = (
    "FordB",
    "Lightning2",
    "MoteStrain",
    "SonyAIBORobotSurface1",
    "ProximalPhalanxOutlineCorrect",
    "MiddlePhalanxOutlineCorrect",
)
H0_ABSOLUTE_POSITIONS = (12, 36, 156, 180)
CANDIDATE_FIXED = "H0_fixed_binding"
CANDIDATE_DYNAMIC = "H1_dynamic_binding"
CANDIDATES = (CANDIDATE_FIXED, CANDIDATE_DYNAMIC)
TARGET_ONLY_ORDERS = (
    (CANDIDATE_FIXED, CANDIDATE_DYNAMIC),
    (CANDIDATE_DYNAMIC, CANDIDATE_FIXED),
)
SOURCE_ORDER = (CANDIDATE_DYNAMIC, CANDIDATE_FIXED)
ELIGIBLE = "ELIGIBLE_REQUEST_CONFIRMATION"
CONTRAINDICATED = "CONTRAINDICATED_ABSTAIN"

LLM_PROPOSAL: dict[str, Any] = {
    "target_surface": "Program Binding",
    "operation": "PATCH_BINDING",
    "diagnosed_failure": "The Source program instance hard-codes absolute positions, so compilation ignores the current series' observer-localized nodes and edits unrelated locations when series length changes.",
    "proposed_change": "Compile each program instance by binding its replacement positions directly to the H0 observer's localized node set for the current visible-fit series, preserving the existing center-excluded local-median operation and all other surfaces.",
    "required_observables": [
        "current visible-fit series identity within the evaluation boundary",
        "current series length",
        "H0 localized node indices for that series",
    ],
    "predicted_behavior_change": "For every compiled instance, executed_positions equals the in-range H0 localized node set; node_overlap becomes complete and no position outside that observed scope is modified, including for lengths 286 and 96.",
    "non_target_invariant": "Observation logic, local-median repair semantics, scope policy, guards, Consumer, Metric, and outcome-access policy remain unchanged; neither Dataset ID nor Query outcome is used.",
    "falsification_condition": "On a fresh series of a different length, the frozen compiler receives valid H0 node indices but executes any different position, omits an in-range localized node, or changes any position outside the localized node set.",
}


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _in_range_h0_positions(length: int) -> tuple[int, ...]:
    """Compile the old Source instance without changing its absolute binding."""

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        BOUND_RADIUS,
    )

    return tuple(
        position
        for position in H0_ABSOLUTE_POSITIONS
        if position - BOUND_RADIUS >= 0 and position + BOUND_RADIUS < length
    )


def _binding_diagnostic(
    observer_nodes: tuple[int, ...], executed_positions: tuple[int, ...]
) -> dict[str, Any]:
    observed = set(observer_nodes)
    executed = set(executed_positions)
    overlap = observed & executed
    return {
        "observer_localized_nodes": list(observer_nodes),
        "executed_positions": list(executed_positions),
        "node_overlap_count": len(overlap),
        "observer_node_count": len(observed),
        "executed_position_count": len(executed),
        "missed_observer_nodes": sorted(observed - executed),
        "out_of_scope_executed_positions": sorted(executed - observed),
        "exact_binding": executed == observed,
    }


def _apply_bound_program_allow_noop(
    np: Any,
    train_inputs: Any,
    *,
    positions: tuple[int, ...],
    window_length: int,
) -> tuple[Any, dict[str, Any]]:
    """Run the unchanged local-median Program without treating a numeric no-op as failure."""

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        BOUND_RADIUS,
        BOUND_WINDOW,
        MODIFIED_TOLERANCE,
    )

    matrix = np.asarray(train_inputs, dtype=np.float64)
    attempted = tuple(int(position) for position in positions)
    if matrix.ndim != 2 or matrix.shape[1] != window_length:
        raise ValueError("bound-node Program expects complete training inputs")
    if BOUND_WINDOW != 2 * BOUND_RADIUS + 1:
        raise AssertionError("bound-node neighborhood geometry changed")
    if any(
        position - BOUND_RADIUS < 0
        or position + BOUND_RADIUS >= window_length
        for position in attempted
    ):
        raise ValueError("bound position does not support the frozen neighborhood")

    processed = matrix.copy()
    for row_index, row in enumerate(matrix):
        for position in attempted:
            neighbors = np.concatenate(
                (
                    row[position - BOUND_RADIUS : position],
                    row[position + 1 : position + BOUND_RADIUS + 1],
                )
            )
            if neighbors.shape != (6,) or not np.isfinite(neighbors).all():
                raise ValueError("invalid center-excluded bound-node neighborhood")
            processed[row_index, position] = float(np.median(neighbors))

    changed = np.abs(processed - matrix) > MODIFIED_TOLERANCE
    allowed = np.zeros(matrix.shape, dtype=bool)
    allowed[:, list(attempted)] = True
    if bool(np.any(changed & ~allowed)):
        raise AssertionError("bound-node Program changed an unbound position")
    modified_any = tuple(int(index) for index in np.flatnonzero(np.any(changed, axis=0)))
    modified_every = tuple(int(index) for index in np.flatnonzero(np.all(changed, axis=0)))
    modified_points = int(np.count_nonzero(changed))
    return processed, {
        "attempted_bound_positions": list(attempted),
        "materially_modified_positions": list(modified_any),
        "materially_modified_positions_on_every_row": list(modified_every),
        "numeric_noop_possible": True,
        "only_bound_positions_changed": True,
        "all_attempted_positions_modified_on_every_row": set(modified_every) == set(attempted),
        "modified_point_count": modified_points,
        "total_point_count": int(matrix.size),
        "modified_point_fraction": modified_points / float(matrix.size),
        "replacement": (
            "median(indices i-3:i concatenated with i+1:i+4); center excluded"
        ),
        "comparison_tolerance": MODIFIED_TOLERANCE,
    }


def _scope_from_legacy_decision(decision: str) -> str:
    if decision == "EXECUTE_BOUND_REPAIR":
        return ELIGIBLE
    if decision == "ABSTAIN_KEEP_INCUMBENT":
        return CONTRAINDICATED
    return "UNRESOLVED_ABSTAIN"


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
    true_positions = tuple(int(position) for position in helpers["positions"](train_values.shape[1]))

    fit_artifact, support_artifact = _condition_inputs(
        np,
        base_fit=base_fit,
        fit_labels=fit_labels,
        base_support=base_support,
        support_labels=support_labels,
        positions=true_positions,
        condition="fit_only_artifact",
        inject=helpers["inject"],
    )
    artifact_observation = helpers["observe"](np, fit_artifact, fit_labels)
    observer_nodes = tuple(int(node) for node in artifact_observation["nodes"])
    artifact_witness = helpers["witness"](
        np,
        fit_artifact,
        fit_labels,
        support_artifact,
        support_labels,
        observer_nodes,
        helpers["rolling_median"],
    )
    artifact_decision, artifact_reasons = helpers["risk_decision"](artifact_witness)
    artifact_scope = _scope_from_legacy_decision(artifact_decision)

    fit_event, support_event = _condition_inputs(
        np,
        base_fit=base_fit,
        fit_labels=fit_labels,
        base_support=base_support,
        support_labels=support_labels,
        positions=true_positions,
        condition="stable_task_event",
        inject=helpers["inject"],
    )
    event_observation = helpers["observe"](np, fit_event, fit_labels)
    event_nodes = tuple(int(node) for node in event_observation["nodes"])
    event_witness = helpers["witness"](
        np,
        fit_event,
        fit_labels,
        support_event,
        support_labels,
        event_nodes,
        helpers["rolling_median"],
    )
    event_decision, event_reasons = helpers["risk_decision"](event_witness)
    event_scope = _scope_from_legacy_decision(event_decision)

    h0_positions = _in_range_h0_positions(train_values.shape[1])
    h1_positions = observer_nodes
    h0_binding = _binding_diagnostic(observer_nodes, h0_positions)
    h1_binding = _binding_diagnostic(observer_nodes, h1_positions)
    if len(observer_nodes) != 4 or set(observer_nodes) != set(true_positions):
        raise ValueError(f"observer did not recover four injected nodes: {dataset}")
    if artifact_scope != ELIGIBLE:
        raise ValueError(f"artifact scope changed: {dataset}")
    if event_scope != CONTRAINDICATED or set(event_nodes) != set(true_positions):
        raise ValueError(f"event scope changed: {dataset}")
    if h0_binding["node_overlap_count"] >= 4 or h0_binding["exact_binding"]:
        raise ValueError(f"H0 binding is not a mismatch: {dataset}")
    if not h1_binding["exact_binding"]:
        raise ValueError(f"H1 binding did not bind observer nodes: {dataset}")

    return {
        "dataset": dataset,
        "archive": f"{DATA_DIR}/{dataset}.zip",
        "official_train_count": int(train_values.shape[0]),
        "series_length": int(train_values.shape[1]),
        "fit_count": int(fit_indices.size),
        "support_count": int(support_indices.size),
        "class_counts": {
            str(label): int(np.count_nonzero(train_labels == label)) for label in (0, 1)
        },
        "controlled_artifact": {
            "private_evaluator_true_positions": list(true_positions),
            "observation": artifact_observation,
            "witness": artifact_witness,
            "decision_input_text": helpers["decision_text"](task_context, artifact_witness),
            "scope": artifact_scope,
            "scope_reasons": artifact_reasons,
        },
        "stable_task_event": {
            "observation": event_observation,
            "witness": event_witness,
            "scope": event_scope,
            "scope_reasons": event_reasons,
            "compiled_policy": "ABSTAIN_KEEP_INCUMBENT",
        },
        "bindings": {
            CANDIDATE_FIXED: h0_binding,
            CANDIDATE_DYNAMIC: h1_binding,
        },
    }


def build_plan(root: Path) -> dict[str, Any]:
    source_report = _read_object(root / SOURCE_REPORT_PATH)
    if source_report.get("verdict") != "CONTROLLED_INTEGRATED_HARNESS_EVOLUTION_FAIL":
        raise ValueError("W52 prior Harness evidence is unavailable")
    if LLM_PROPOSAL.get("operation") != "PATCH_BINDING":
        raise ValueError("blind proposal did not select Program Binding")
    targets = [_build_target_train_plan(root, dataset) for dataset in TARGET_DATASETS]
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "After Observation and Scope are fixed, binding the unchanged Program schema "
            "to current observer-localized nodes reduces adaptation regret relative to "
            "reusing a Source instance with absolute positions."
        ),
        "scientific_role": "controlled Program-Binding Harness update",
        "context_exposure": "INSTANCE_SEEN_TRAIN_ONLY",
        "outcome_exposure": "SEALED",
        "prior_harness_evidence": {
            "W52_verdict": source_report["verdict"],
            "established_surfaces": ["Observation", "Scope"],
            "W52_used_dynamic_observer_binding": True,
        },
        "controlled_binding_failure_card": {
            "evidence_role": "development_controlled_binding_failure",
            "controlled_first_actionable_fault": "program_binding",
            "H0_program_instance_positions": list(H0_ABSOLUTE_POSITIONS),
            "example_lengths_in_blind_proposal": [286, 96],
            "not_attributed_to_W52": True,
        },
        "blind_llm_patch_proposal": LLM_PROPOSAL,
        "frozen_surfaces": {
            "changed": "Program Binding only",
            "task": "shared classification-local-event-quality-v1",
            "observation": "class-conditioned-impulse-topology/1",
            "scope": "W52 cross-cohort evidence scope, unchanged",
            "program": "center-excluded local median, unchanged",
            "risk": "stable task event abstains, unchanged",
            "consumer": "ridge-raw-plus-difference-v1",
            "metric": "classification accuracy",
            "proxy": "none",
        },
        "target_train_plans": targets,
        "frozen_policies": {
            "H0_direct": "execute unchanged Program with in-range Source absolute positions",
            "A3_target_only": "incumbent then exact mean over both candidate-confirmation orders",
            "A4_source_only": "incumbent because binding correctness is not Utility evidence",
            "A5_source_plus_target": "confirm H1 dynamic binding, then H0 fixed binding",
            "candidate_selection": (
                "highest Support accuracy among incumbent and confirmed candidates; "
                "ties prefer incumbent, then H0_fixed_binding"
            ),
            "feedback_budgets": [0, 1, 2],
            "budget_unit": "one complete Target Support Consumer refit outcome",
        },
        "frozen_success_gate": {
            "H1_binding_exact_all_targets": True,
            "H0_binding_mismatch_all_targets": True,
            "H1_query_accuracy_strictly_greater_than_H0_count_min": 5,
            "H1_positive_query_gain_count_min": 4,
            "macro_A5_adapt_auc_strictly_greater_than_A3": True,
            "negative_A5_vs_A3_target_count": 0,
            "event_scope_unchanged_all_targets": True,
        },
        "target_test_values_or_labels_read": False,
        "selection_used_target_program_or_consumer_outcome": False,
        "persistent_memory_built": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "The backgrounds and labels are real UCR data, but the artifact is controlled. "
            "This tests Program-Binding Harness evolution, not natural defects, persistent "
            "Memory, autonomous promotion, or general safety."
        ),
    }


def _best_confirmed(
    confirmed: list[str], candidates: dict[str, dict[str, Any]], incumbent: dict[str, float]
) -> str | None:
    """Return the frozen Support winner; exact incumbent ties abstain."""

    incumbent_support = float(incumbent["support_accuracy"])
    useful = [
        candidate
        for candidate in confirmed
        if float(candidates[candidate]["support_accuracy"]) > incumbent_support
    ]
    if not useful:
        return None
    tie_rank = {CANDIDATE_FIXED: 0, CANDIDATE_DYNAMIC: 1}
    return min(
        useful,
        key=lambda candidate: (
            -float(candidates[candidate]["support_accuracy"]),
            tie_rank[candidate],
        ),
    )


def _adaptation_curve(
    order: tuple[str, str],
    candidates: dict[str, dict[str, Any]],
    incumbent: dict[str, float],
) -> list[dict[str, Any]]:
    confirmed: list[str] = []
    curve: list[dict[str, Any]] = []
    for budget in range(3):
        selected = _best_confirmed(confirmed, candidates, incumbent)
        selected_readout = incumbent if selected is None else candidates[selected]
        point: dict[str, Any] = {
            "budget": budget,
            "confirmed_candidates": list(confirmed),
            "selected_candidate": selected or "incumbent",
            "utility": float(selected_readout["query_accuracy"]),
        }
        if budget < 2:
            revealed = order[budget]
            confirmed.append(revealed)
            point["next_feedback"] = {
                "candidate": revealed,
                "support_accuracy": float(candidates[revealed]["support_accuracy"]),
                "support_gain": float(candidates[revealed]["support_accuracy"])
                - float(incumbent["support_accuracy"]),
                "budget_cost": 1,
            }
        curve.append(point)
    return curve


def _average_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    return [
        {
            "budget": budget,
            "utility": sum(float(curve[budget]["utility"]) for curve in curves)
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

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Program-Binding plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("Program-Binding Target outcomes were not sealed")
    if plan.get("blind_llm_patch_proposal") != LLM_PROPOSAL:
        raise ValueError("blind Program-Binding proposal changed")
    target_plans = plan.get("target_train_plans")
    if not isinstance(target_plans, list):
        raise ValueError("missing frozen Target TRAIN plans")
    planned = {str(row["dataset"]): row for row in target_plans}
    if (
        len(target_plans) != len(TARGET_DATASETS)
        or len(planned) != len(TARGET_DATASETS)
        or set(planned) != set(TARGET_DATASETS)
    ):
        raise ValueError("Program-Binding Target roster is not frozen and unique")

    _, helpers = _helpers()
    rows: list[dict[str, Any]] = []
    consumer_fit_count = 0
    for dataset in TARGET_DATASETS:
        current = _build_target_train_plan(root, dataset)
        frozen = planned[dataset]
        if current != frozen:
            raise ValueError(f"planned TRAIN context changed: {dataset}")

        archive = root / DATA_DIR / f"{dataset}.zip"
        train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
        fit_indices, support_indices = helpers["split"](np, train_labels)
        base_fit = train_values[fit_indices]
        fit_labels = train_labels[fit_indices]
        base_support = train_values[support_indices]
        support_labels = train_labels[support_indices]
        true_positions = tuple(int(position) for position in helpers["positions"](train_values.shape[1]))
        fit_values, support_values = _condition_inputs(
            np,
            base_fit=base_fit,
            fit_labels=fit_labels,
            base_support=base_support,
            support_labels=support_labels,
            positions=true_positions,
            condition="fit_only_artifact",
            inject=helpers["inject"],
        )
        h0_positions = tuple(
            int(position)
            for position in frozen["bindings"][CANDIDATE_FIXED]["executed_positions"]
        )
        h1_positions = tuple(
            int(position)
            for position in frozen["bindings"][CANDIDATE_DYNAMIC]["executed_positions"]
        )
        h0_values, h0_modification = _apply_bound_program_allow_noop(
            np,
            fit_values,
            positions=h0_positions,
            window_length=train_values.shape[1],
        )
        h1_values, h1_modification = _apply_bound_program_allow_noop(
            np,
            fit_values,
            positions=h1_positions,
            window_length=train_values.shape[1],
        )
        if set(h1_modification["materially_modified_positions_on_every_row"]) != set(
            h1_positions
        ):
            raise AssertionError(
                f"H1 did not materially repair every localized injected node: {dataset}"
            )
        observer_nodes = tuple(
            int(node) for node in frozen["controlled_artifact"]["observation"]["nodes"]
        )
        actual_execution_binding = {
            CANDIDATE_FIXED: _binding_diagnostic(
                observer_nodes,
                tuple(int(position) for position in h0_modification["attempted_bound_positions"]),
            ),
            CANDIDATE_DYNAMIC: _binding_diagnostic(
                observer_nodes,
                tuple(int(position) for position in h1_modification["attempted_bound_positions"]),
            ),
        }
        # Keep all execution/binding assertions TRAIN-only.  TEST is opened only
        # after those checks, immediately before the first Consumer readout.
        query_values, query_labels = helpers["load"](np, archive, dataset, "TEST")
        incumbent = _fit_readout(
            np,
            RidgeClassifier,
            helpers["features"],
            fit_values,
            fit_labels,
            support_values,
            support_labels,
            query_values,
            query_labels,
        )
        h0 = _fit_readout(
            np,
            RidgeClassifier,
            helpers["features"],
            h0_values,
            fit_labels,
            support_values,
            support_labels,
            query_values,
            query_labels,
        )
        h1 = _fit_readout(
            np,
            RidgeClassifier,
            helpers["features"],
            h1_values,
            fit_labels,
            support_values,
            support_labels,
            query_values,
            query_labels,
        )
        consumer_fit_count += 3
        candidates = {CANDIDATE_FIXED: h0, CANDIDATE_DYNAMIC: h1}
        a3_per_order = [
            _adaptation_curve(order, candidates, incumbent) for order in TARGET_ONLY_ORDERS
        ]
        a3_curve = _average_curves(a3_per_order)
        a5_curve = _adaptation_curve(SOURCE_ORDER, candidates, incumbent)
        a4_curve = [
            {
                "budget": budget,
                "utility": float(incumbent["query_accuracy"]),
                "selected_candidate": "incumbent",
            }
            for budget in range(3)
        ]
        a3_auc = _adapt_auc(a3_curve)
        a5_auc = _adapt_auc(a5_curve)
        rows.append(
            {
                "dataset": dataset,
                "series_length": int(train_values.shape[1]),
                "controlled_artifact_true_positions": list(true_positions),
                "observer_localized_nodes": frozen["controlled_artifact"]["observation"]["nodes"],
                "bindings": frozen["bindings"],
                "actual_execution_binding": actual_execution_binding,
                "program_modification": {
                    CANDIDATE_FIXED: h0_modification,
                    CANDIDATE_DYNAMIC: h1_modification,
                },
                "readouts": {
                    "incumbent": incumbent,
                    CANDIDATE_FIXED: {
                        **h0,
                        "support_gain": float(h0["support_accuracy"])
                        - float(incumbent["support_accuracy"]),
                        "query_gain": float(h0["query_accuracy"])
                        - float(incumbent["query_accuracy"]),
                    },
                    CANDIDATE_DYNAMIC: {
                        **h1,
                        "support_gain": float(h1["support_accuracy"])
                        - float(incumbent["support_accuracy"]),
                        "query_gain": float(h1["query_accuracy"])
                        - float(incumbent["query_accuracy"]),
                    },
                },
                "H0_direct_query_accuracy": float(h0["query_accuracy"]),
                "A3_target_only": {
                    "per_order": a3_per_order,
                    "mean_curve": a3_curve,
                    "adapt_auc": a3_auc,
                },
                "A4_source_only": {
                    "mean_curve": a4_curve,
                    "adapt_auc": _adapt_auc(a4_curve),
                },
                "A5_source_plus_target": {
                    "confirmation_order": list(SOURCE_ORDER),
                    "curve": a5_curve,
                    "adapt_auc": a5_auc,
                },
                "A5_minus_A3_adapt_auc": a5_auc - a3_auc,
                "stable_event_policy": {
                    "scope": frozen["stable_task_event"]["scope"],
                    "selected_action": "ABSTAIN_KEEP_INCUMBENT",
                    "policy_event_harm": 0.0,
                    "forced_action_harm_not_claimed": True,
                },
                "context_or_decision_used_dataset_id": False,
            }
        )

    binding_exact_all = all(
        bool(row["actual_execution_binding"][CANDIDATE_DYNAMIC]["exact_binding"])
        for row in rows
    )
    h0_mismatch_all = all(
        not bool(row["actual_execution_binding"][CANDIDATE_FIXED]["exact_binding"])
        for row in rows
    )
    event_scope_unchanged_all = all(
        row["stable_event_policy"]["scope"] == CONTRAINDICATED for row in rows
    )
    h1_better_than_h0_count = sum(
        float(row["readouts"][CANDIDATE_DYNAMIC]["query_accuracy"])
        > float(row["readouts"][CANDIDATE_FIXED]["query_accuracy"])
        for row in rows
    )
    h1_positive_count = sum(
        float(row["readouts"][CANDIDATE_DYNAMIC]["query_gain"]) > 0.0 for row in rows
    )
    a3_macro = sum(float(row["A3_target_only"]["adapt_auc"]) for row in rows) / len(rows)
    a5_macro = sum(float(row["A5_source_plus_target"]["adapt_auc"]) for row in rows) / len(rows)
    negative_count = sum(float(row["A5_minus_A3_adapt_auc"]) < -1e-12 for row in rows)
    gate_pass = bool(
        binding_exact_all
        and h0_mismatch_all
        and h1_better_than_h0_count >= 5
        and h1_positive_count >= 4
        and a5_macro > a3_macro
        and negative_count == 0
        and event_scope_unchanged_all
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": plan["causal_hypothesis"],
        "scientific_role": plan["scientific_role"],
        "prior_harness_evidence": plan["prior_harness_evidence"],
        "controlled_binding_failure_card": plan["controlled_binding_failure_card"],
        "blind_llm_patch_proposal": plan["blind_llm_patch_proposal"],
        "frozen_surfaces": plan["frozen_surfaces"],
        "frozen_policies": plan["frozen_policies"],
        "frozen_success_gate": plan["frozen_success_gate"],
        "plan_exposure": {
            "context_exposure": plan["context_exposure"],
            "outcome_exposure_before_evaluate": plan["outcome_exposure"],
            "outcome_exposure_after_evaluate": "EXPOSED",
        },
        "dataset_evidence": rows,
        "overall": {
            "target_dataset_count": len(rows),
            "H1_binding_exact_all_targets": binding_exact_all,
            "H0_binding_mismatch_all_targets": h0_mismatch_all,
            "H1_query_accuracy_greater_than_H0_count": h1_better_than_h0_count,
            "H1_positive_query_gain_count": h1_positive_count,
            "A3_macro_adapt_auc": a3_macro,
            "A5_macro_adapt_auc": a5_macro,
            "A5_minus_A3_macro_adapt_auc": a5_macro - a3_macro,
            "negative_A5_vs_A3_target_count": negative_count,
            "event_scope_unchanged_all_targets": event_scope_unchanged_all,
            "A5_policy_event_harm_max": max(
                float(row["stable_event_policy"]["policy_event_harm"]) for row in rows
            ),
            "frozen_gate_pass": gate_pass,
        },
        "consumer_fit_count": consumer_fit_count,
        "execution_history": {
            "evaluate_attempt_count": 2,
            "first_evaluate_attempt": {
                "status": "FAILED_BEFORE_UTILITY",
                "failed_dataset": "FordB",
                "failure_stage": "after_first_TEST_load_before_any_consumer_fit",
                "failure": "not every bound node was materially replaced",
                "test_split_load_count": 1,
                "consumer_fit_count": 0,
                "utility_output_count": 0,
            },
            "retry": {
                "status": "OUTCOME_COMPUTED",
                "test_split_load_count": len(TARGET_DATASETS),
                "outcome_computed_once": True,
            },
            "total_test_split_load_count_across_attempts": 1 + len(TARGET_DATASETS),
        },
        "verdict": (
            "CONTROLLED_PROGRAM_BINDING_HARNESS_UPDATE_PASS"
            if gate_pass
            else "CONTROLLED_PROGRAM_BINDING_HARNESS_UPDATE_FAIL"
        ),
        "official_target_test_outcome_opened_once": False,
        "outcome_computed_once": True,
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
        "formal_natural_capability_promotion": False,
        "general_safety_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": (
            "Treat Program Binding as a validated evolvable Harness surface and move to a "
            "natural Workflow family without adding another controlled binding variant."
            if gate_pass
            else "Localize the first failed frozen gate once; do not tune this roster or gate."
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
