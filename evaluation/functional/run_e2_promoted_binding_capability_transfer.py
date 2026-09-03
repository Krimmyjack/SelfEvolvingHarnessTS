"""Run W56 controlled transfer of the promoted W55 Program-Binding Capability.

The admitted Source evidence compiles one frozen Context-conditioned Capability:
classification-local-event-quality TaskContext, the W52/W55 cross-cohort
Observation and Scope, W55 dynamic observer-node binding, and the unchanged
center-excluded local-median Program.  ``plan`` reads only official UCR TRAIN
splits.  ``evaluate`` first replays every TRAIN plan and execution assertion;
only then does it open each official TEST split once, immediately before that
dataset's Consumer readouts.

This is a controlled injected artifact on real UCR backgrounds.  It is not a
natural-defect result, persistent Memory, an autonomous LLM update, or a general
safety claim.  The original UCI Target remains unopened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-promoted-binding-capability-transfer/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_promoted_binding_capability_transfer_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_promoted_binding_capability_transfer_report.json"
)
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_program_binding_harness_update_report.json"
)
DATA_DIR = "data/ucr_task_context"
TARGET_DATASETS = (
    "BirdChicken",
    "HouseTwenty",
    "ToeSegmentation1",
    "PhalangesOutlinesCorrect",
    "SonyAIBORobotSurface2",
    "GunPointAgeSpan",
)
CANDIDATE_FIXED = "H0_fixed_binding"
CANDIDATE_DYNAMIC = "H1_dynamic_binding"
TARGET_ONLY_ORDERS = (
    (CANDIDATE_FIXED, CANDIDATE_DYNAMIC),
    (CANDIDATE_DYNAMIC, CANDIDATE_FIXED),
)
SOURCE_ORDER = (CANDIDATE_DYNAMIC, CANDIDATE_FIXED)
ELIGIBLE = "ELIGIBLE_REQUEST_CONFIRMATION"
CONTRAINDICATED = "CONTRAINDICATED_ABSTAIN"


PROMOTED_CAPABILITY: dict[str, Any] = {
    "capability_id": "controlled-classification-local-event-dynamic-binding-v1",
    "task_context": "classification-local-event-quality",
    "observation": "W52/W55 cross-cohort class-conditioned impulse topology",
    "applicability_scope": {
        "fit_only_artifact": ELIGIBLE,
        "stable_task_event": CONTRAINDICATED,
    },
    "risk_policy": "stable task event abstains",
    "program_binding": "W55 dynamic current-fit observer-node binding",
    "program": "center-excluded local median",
    "program_changed_from_W55": False,
    "consumer": "ridge-raw-plus-difference-v1",
    "metric": "classification accuracy",
}


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _admit_source_report(source_report: dict[str, Any]) -> dict[str, Any]:
    """Validate exactly the W55 evidence required to compile the Capability."""

    overall = source_report.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("W55 Source report has no overall evidence")
    checks = {
        "verdict": source_report.get("verdict")
        == "CONTROLLED_PROGRAM_BINDING_HARNESS_UPDATE_PASS",
        "H1_binding_exact_all_targets": overall.get(
            "H1_binding_exact_all_targets"
        )
        is True,
        "H1_positive_query_gain_count": overall.get(
            "H1_positive_query_gain_count"
        )
        == 6,
        "negative_A5_vs_A3_target_count": overall.get(
            "negative_A5_vs_A3_target_count"
        )
        == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"W55 Source Capability is not admitted: {failed}")
    return {
        "source_report": SOURCE_REPORT_PATH,
        "source_admitted": True,
        "verdict": source_report["verdict"],
        "H1_binding_exact_all_targets": True,
        "H1_positive_query_gain_count": 6,
        "negative_A5_vs_A3_target_count": 0,
        "evidence_role": "lightweight admitted Source evidence; not persistent Memory",
    }


def _build_target_train_plan(root: Path, dataset: str) -> dict[str, Any]:
    """Reuse the W55 generic TRAIN-only Observation/Scope/Binding planner."""

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_program_binding_harness_update import (
        _build_target_train_plan as _build_w55_target_train_plan,
    )

    return _build_w55_target_train_plan(root, dataset)


def build_plan(root: Path) -> dict[str, Any]:
    source_evidence = _admit_source_report(_read_object(root / SOURCE_REPORT_PATH))
    targets = [_build_target_train_plan(root, dataset) for dataset in TARGET_DATASETS]
    for row in targets:
        dataset = str(row["dataset"])
        if row["controlled_artifact"]["scope"] != ELIGIBLE:
            raise ValueError(f"artifact eligibility changed: {dataset}")
        if row["stable_task_event"]["scope"] != CONTRAINDICATED:
            raise ValueError(f"event contraindication changed: {dataset}")
        if not row["bindings"][CANDIDATE_DYNAMIC]["exact_binding"]:
            raise ValueError(f"H1 binding is not exact: {dataset}")
        if row["bindings"][CANDIDATE_FIXED]["exact_binding"]:
            raise ValueError(f"H0 binding is not a mismatch: {dataset}")
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "At the same Target downstream-feedback budget, the admitted Source "
            "Capability with Scope/Risk/Evidence and dynamic Program Binding adapts "
            "to unseen controlled UCR targets faster and no less safely than Target-only "
            "adaptation from scratch."
        ),
        "scientific_role": "W56 controlled Source-Capability transfer",
        "context_exposure": "INSTANCE_SEEN_TRAIN_ONLY",
        "outcome_exposure": "SEALED",
        "source_evidence": source_evidence,
        "compiled_source_capability": PROMOTED_CAPABILITY,
        "target_train_plans": targets,
        "frozen_policies": {
            "incumbent": "leave optional fit-only artifact untreated",
            "candidate_order_tie_break": CANDIDATE_FIXED,
            "selection_rule": (
                "select the highest-Support-accuracy confirmed candidate only when "
                "strictly above incumbent; exact incumbent ties abstain; equal candidate "
                "Support accuracy prefers H0_fixed_binding"
            ),
            "A3_target_only": (
                "B0 incumbent; exact mean over H0->H1 and H1->H0 confirmation orders "
                "at B1/B2"
            ),
            "A4_source_only": (
                "execute promoted H1 directly at B0/B1/B2 without Target feedback"
            ),
            "A5_source_plus_target": (
                "B0 execute promoted H1; confirm H1 first and keep or rollback at B1; "
                "confirm H0 second and select among incumbent and confirmed candidates at B2"
            ),
            "feedback_budgets": [0, 1, 2],
            "budget_unit": "one complete Target Support Consumer refit outcome",
        },
        "frozen_surfaces": {
            "task_context": "classification-local-event-quality, unchanged",
            "observation": "W52/W55 cross-cohort Observation, unchanged",
            "scope": "fit-only artifact eligible; stable event contraindicated, unchanged",
            "program_binding": "W55 dynamic observer-node Binding for H1",
            "program": "center-excluded local median, unchanged",
            "consumer": "ridge-raw-plus-difference-v1, unchanged",
            "metric": "classification accuracy, unchanged",
            "proxy": "none",
        },
        "frozen_success_gate": {
            "source_admitted": True,
            "H1_binding_exact_target_count": 6,
            "H0_binding_mismatch_target_count": 6,
            "event_scope_unchanged_target_count": 6,
            "A4_zero_shot_macro_query_gain_strictly_greater_than": 0.0,
            "A4_positive_query_gain_count_min": 4,
            "macro_A5_adapt_auc_strictly_greater_than_A3": True,
            "negative_A5_vs_A3_target_count": 0,
            "A5_policy_event_harm_max": 0.0,
        },
        "target_test_values_or_labels_read": False,
        "selection_used_target_query_outcome": False,
        "persistent_memory_built": False,
        "autonomous_llm_update": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "Controlled injected artifact on real UCR backgrounds; not a natural defect, "
            "persistent Memory, autonomous LLM update, or general safety conclusion. "
            "The original UCI Target is unopened."
        ),
    }


def _best_confirmed(
    confirmed: list[str],
    candidates: dict[str, dict[str, float]],
    incumbent: dict[str, float],
) -> str | None:
    """Reuse W55's frozen strict-improvement and candidate-tie rule."""

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_program_binding_harness_update import (
        _best_confirmed as _best_w55_confirmed,
    )

    return _best_w55_confirmed(confirmed, candidates, incumbent)


def _target_only_curve(
    order: tuple[str, str],
    candidates: dict[str, dict[str, float]],
    incumbent: dict[str, float],
) -> list[dict[str, Any]]:
    """Reuse W55's exact Target-only confirmation curve."""

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_program_binding_harness_update import (
        _adaptation_curve as _w55_target_only_curve,
    )

    return _w55_target_only_curve(order, candidates, incumbent)


def _source_only_curve(
    candidates: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    return [
        {
            "budget": budget,
            "confirmed_candidates": [],
            "selected_candidate": CANDIDATE_DYNAMIC,
            "selection_basis": "promoted_source_capability_no_target_feedback",
            "utility": float(candidates[CANDIDATE_DYNAMIC]["query_accuracy"]),
        }
        for budget in range(3)
    ]


def _source_plus_target_curve(
    candidates: dict[str, dict[str, float]],
    incumbent: dict[str, float],
) -> list[dict[str, Any]]:
    """Execute H1 at B0, then revise it with equal-budget Target feedback."""

    confirmed: list[str] = []
    curve: list[dict[str, Any]] = []
    for budget in range(3):
        if budget == 0:
            selected: str | None = CANDIDATE_DYNAMIC
            basis = "promoted_source_capability_zero_shot"
        else:
            selected = _best_confirmed(confirmed, candidates, incumbent)
            basis = "target_support_keep_or_rollback"
        selected_name = selected or "incumbent"
        readout = incumbent if selected is None else candidates[selected]
        point: dict[str, Any] = {
            "budget": budget,
            "confirmed_candidates": list(confirmed),
            "selected_candidate": selected_name,
            "selection_basis": basis,
            "utility": float(readout["query_accuracy"]),
        }
        if budget < 2:
            revealed = SOURCE_ORDER[budget]
            point["next_feedback"] = {
                "candidate": revealed,
                "support_accuracy": float(candidates[revealed]["support_accuracy"]),
                "support_gain": float(candidates[revealed]["support_accuracy"])
                - float(incumbent["support_accuracy"]),
                "budget_cost": 1,
            }
            confirmed.append(revealed)
        curve.append(point)
    return curve


def _average_curves(curves: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_program_binding_harness_update import (
        _average_curves as _average_w55_curves,
    )

    return _average_w55_curves(curves)


def _adapt_auc(curve: list[dict[str, Any]]) -> float:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_program_binding_harness_update import (
        _adapt_auc as _w55_adapt_auc,
    )

    return _w55_adapt_auc(curve)


def _prepare_train_execution(
    root: Path, dataset: str, frozen: dict[str, Any]
) -> dict[str, Any]:
    """Execute and assert the frozen candidates without reading TEST."""

    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_program_binding_harness_update import (
        _apply_bound_program_allow_noop,
        _binding_diagnostic,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _condition_inputs,
        _helpers,
    )

    _, helpers = _helpers()
    archive = root / DATA_DIR / f"{dataset}.zip"
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    fit_indices, support_indices = helpers["split"](np, train_labels)
    base_fit = train_values[fit_indices]
    fit_labels = train_labels[fit_indices]
    base_support = train_values[support_indices]
    support_labels = train_labels[support_indices]
    true_positions = tuple(
        int(position) for position in helpers["positions"](train_values.shape[1])
    )
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
    positions = {
        candidate: tuple(
            int(position)
            for position in frozen["bindings"][candidate]["executed_positions"]
        )
        for candidate in (CANDIDATE_FIXED, CANDIDATE_DYNAMIC)
    }
    repaired: dict[str, Any] = {}
    modifications: dict[str, dict[str, Any]] = {}
    for candidate in (CANDIDATE_FIXED, CANDIDATE_DYNAMIC):
        repaired[candidate], modifications[candidate] = _apply_bound_program_allow_noop(
            np,
            fit_values,
            positions=positions[candidate],
            window_length=train_values.shape[1],
        )

    observer_nodes = tuple(
        int(node) for node in frozen["controlled_artifact"]["observation"]["nodes"]
    )
    actual_bindings = {
        candidate: _binding_diagnostic(
            observer_nodes,
            tuple(
                int(position)
                for position in modifications[candidate]["attempted_bound_positions"]
            ),
        )
        for candidate in (CANDIDATE_FIXED, CANDIDATE_DYNAMIC)
    }
    if frozen["controlled_artifact"]["scope"] != ELIGIBLE:
        raise AssertionError(f"artifact scope changed during execution: {dataset}")
    if frozen["stable_task_event"]["scope"] != CONTRAINDICATED:
        raise AssertionError(f"event scope changed during execution: {dataset}")
    if not actual_bindings[CANDIDATE_DYNAMIC]["exact_binding"]:
        raise AssertionError(f"H1 execution binding is not exact: {dataset}")
    if actual_bindings[CANDIDATE_FIXED]["exact_binding"]:
        raise AssertionError(f"H0 execution binding is not a mismatch: {dataset}")
    h1_every_row = set(
        modifications[CANDIDATE_DYNAMIC][
            "materially_modified_positions_on_every_row"
        ]
    )
    if len(observer_nodes) != 4 or h1_every_row != set(observer_nodes):
        raise AssertionError(
            f"H1 did not materially repair all four nodes on every row: {dataset}"
        )
    return {
        "archive": archive,
        "train_values": train_values,
        "fit_values": fit_values,
        "fit_labels": fit_labels,
        "support_values": support_values,
        "support_labels": support_labels,
        "repaired": repaired,
        "true_positions": true_positions,
        "observer_nodes": observer_nodes,
        "actual_bindings": actual_bindings,
        "modifications": modifications,
        "features": helpers["features"],
        "load": helpers["load"],
    }


def _macro_curve(rows: list[dict[str, Any]], policy: str) -> list[dict[str, float]]:
    return [
        {
            "budget": budget,
            "utility": sum(
                float(row[policy]["mean_curve"][budget]["utility"]) for row in rows
            )
            / len(rows),
        }
        for budget in range(3)
    ]


def evaluate(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _fit_readout,
    )

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("promoted Capability transfer plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("W56 Target outcomes were not sealed")
    if plan.get("compiled_source_capability") != PROMOTED_CAPABILITY:
        raise ValueError("compiled Source Capability changed")
    current_source = _admit_source_report(_read_object(root / SOURCE_REPORT_PATH))
    if plan.get("source_evidence") != current_source:
        raise ValueError("admitted Source evidence changed")
    target_plans = plan.get("target_train_plans")
    if not isinstance(target_plans, list):
        raise ValueError("missing frozen Target TRAIN plans")
    planned = {str(row["dataset"]): row for row in target_plans}
    if (
        len(target_plans) != len(TARGET_DATASETS)
        or len(planned) != len(TARGET_DATASETS)
        or tuple(str(row["dataset"]) for row in target_plans) != TARGET_DATASETS
    ):
        raise ValueError("W56 Target roster or order is not frozen and unique")

    # Phase 1 is entirely TRAIN-only.  No TEST split may be opened until all six
    # plans, executions, bindings, scopes, and material-repair assertions pass.
    prepared: dict[str, dict[str, Any]] = {}
    for dataset in TARGET_DATASETS:
        current = _build_target_train_plan(root, dataset)
        frozen = planned[dataset]
        if current != frozen:
            raise ValueError(f"planned TRAIN context changed: {dataset}")
        prepared[dataset] = _prepare_train_execution(root, dataset, frozen)

    # Phase 2 opens each TEST split once, immediately before its three shared
    # Consumer readouts.  Query outcomes never participate in selection.
    rows: list[dict[str, Any]] = []
    consumer_fit_count = 0
    test_split_load_count = 0
    for dataset in TARGET_DATASETS:
        state = prepared[dataset]
        query_values, query_labels = state["load"](
            np, state["archive"], dataset, "TEST"
        )
        test_split_load_count += 1
        incumbent = _fit_readout(
            np,
            RidgeClassifier,
            state["features"],
            state["fit_values"],
            state["fit_labels"],
            state["support_values"],
            state["support_labels"],
            query_values,
            query_labels,
        )
        h0 = _fit_readout(
            np,
            RidgeClassifier,
            state["features"],
            state["repaired"][CANDIDATE_FIXED],
            state["fit_labels"],
            state["support_values"],
            state["support_labels"],
            query_values,
            query_labels,
        )
        h1 = _fit_readout(
            np,
            RidgeClassifier,
            state["features"],
            state["repaired"][CANDIDATE_DYNAMIC],
            state["fit_labels"],
            state["support_values"],
            state["support_labels"],
            query_values,
            query_labels,
        )
        consumer_fit_count += 3
        candidates = {CANDIDATE_FIXED: h0, CANDIDATE_DYNAMIC: h1}
        a3_per_order = [
            _target_only_curve(order, candidates, incumbent)
            for order in TARGET_ONLY_ORDERS
        ]
        a3_curve = _average_curves(a3_per_order)
        a4_curve = _source_only_curve(candidates)
        a5_curve = _source_plus_target_curve(candidates, incumbent)
        a3_auc = _adapt_auc(a3_curve)
        a4_auc = _adapt_auc(a4_curve)
        a5_auc = _adapt_auc(a5_curve)
        h0_readout = {
            **h0,
            "support_gain": float(h0["support_accuracy"])
            - float(incumbent["support_accuracy"]),
            "query_gain": float(h0["query_accuracy"])
            - float(incumbent["query_accuracy"]),
        }
        h1_readout = {
            **h1,
            "support_gain": float(h1["support_accuracy"])
            - float(incumbent["support_accuracy"]),
            "query_gain": float(h1["query_accuracy"])
            - float(incumbent["query_accuracy"]),
        }
        rows.append(
            {
                "dataset": dataset,
                "series_length": int(state["train_values"].shape[1]),
                "controlled_artifact_true_positions": list(state["true_positions"]),
                "binding_metadata": {
                    "observer_localized_nodes": list(state["observer_nodes"]),
                    "planned": planned[dataset]["bindings"],
                    "actual_execution": state["actual_bindings"],
                },
                "program_modification": state["modifications"],
                "readouts": {
                    "incumbent": incumbent,
                    CANDIDATE_FIXED: h0_readout,
                    CANDIDATE_DYNAMIC: h1_readout,
                },
                "A3_target_only": {
                    "per_order": a3_per_order,
                    "mean_curve": a3_curve,
                    "selection_path_per_order": [
                        [point["selected_candidate"] for point in curve]
                        for curve in a3_per_order
                    ],
                    "adapt_auc": a3_auc,
                },
                "A4_source_only": {
                    "mean_curve": a4_curve,
                    "selection_path": [
                        point["selected_candidate"] for point in a4_curve
                    ],
                    "adapt_auc": a4_auc,
                },
                "A5_source_plus_target": {
                    "confirmation_order": list(SOURCE_ORDER),
                    "mean_curve": a5_curve,
                    "selection_path": [
                        point["selected_candidate"] for point in a5_curve
                    ],
                    "adapt_auc": a5_auc,
                },
                "comparisons": {
                    "A5_minus_A3_adapt_auc": a5_auc - a3_auc,
                    "A4_minus_incumbent_query_accuracy": float(h1["query_accuracy"])
                    - float(incumbent["query_accuracy"]),
                    "A5_minus_A4_adapt_auc": a5_auc - a4_auc,
                },
                "stable_event_policy": {
                    "scope": planned[dataset]["stable_task_event"]["scope"],
                    "selected_action": "ABSTAIN_KEEP_INCUMBENT",
                    "policy_event_harm": 0.0,
                },
                "consumer_fit_count": 3,
                "context_or_selection_used_dataset_id": False,
                "selection_used_target_query_outcome": False,
            }
        )

    h1_exact_count = sum(
        bool(row["binding_metadata"]["actual_execution"][CANDIDATE_DYNAMIC]["exact_binding"])
        for row in rows
    )
    h0_mismatch_count = sum(
        not bool(row["binding_metadata"]["actual_execution"][CANDIDATE_FIXED]["exact_binding"])
        for row in rows
    )
    event_scope_count = sum(
        row["stable_event_policy"]["scope"] == CONTRAINDICATED for row in rows
    )
    a4_zero_shot_gains = [
        float(row["comparisons"]["A4_minus_incumbent_query_accuracy"])
        for row in rows
    ]
    a4_zero_shot_macro_gain = sum(a4_zero_shot_gains) / len(rows)
    a4_positive_count = sum(gain > 0.0 for gain in a4_zero_shot_gains)
    a3_macro_auc = sum(float(row["A3_target_only"]["adapt_auc"]) for row in rows) / len(rows)
    a4_macro_auc = sum(float(row["A4_source_only"]["adapt_auc"]) for row in rows) / len(rows)
    a5_macro_auc = sum(float(row["A5_source_plus_target"]["adapt_auc"]) for row in rows) / len(rows)
    negative_a5_a3_count = sum(
        float(row["comparisons"]["A5_minus_A3_adapt_auc"]) < -1e-12
        for row in rows
    )
    a5_event_harm_max = max(
        float(row["stable_event_policy"]["policy_event_harm"]) for row in rows
    )
    source_admitted = bool(current_source["source_admitted"])
    gate_pass = bool(
        source_admitted
        and h1_exact_count == 6
        and h0_mismatch_count == 6
        and event_scope_count == 6
        and a4_zero_shot_macro_gain > 0.0
        and a4_positive_count >= 4
        and a5_macro_auc > a3_macro_auc
        and negative_a5_a3_count == 0
        and a5_event_harm_max <= 1e-12
    )
    macro_curves = {
        policy: _macro_curve(rows, policy)
        for policy in ("A3_target_only", "A4_source_only", "A5_source_plus_target")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": plan["causal_hypothesis"],
        "scientific_role": plan["scientific_role"],
        "source_evidence": current_source,
        "compiled_source_capability": plan["compiled_source_capability"],
        "frozen_surfaces": plan["frozen_surfaces"],
        "frozen_policies": plan["frozen_policies"],
        "frozen_success_gate": plan["frozen_success_gate"],
        "plan_exposure": {
            "context_exposure": plan["context_exposure"],
            "outcome_exposure_before_evaluate": plan["outcome_exposure"],
            "outcome_exposure_after_evaluate": "EXPOSED",
        },
        "dataset_evidence": rows,
        "macro_curves": macro_curves,
        "overall": {
            "target_dataset_count": len(rows),
            "source_admitted": source_admitted,
            "H1_binding_exact_target_count": h1_exact_count,
            "H0_binding_mismatch_target_count": h0_mismatch_count,
            "event_scope_unchanged_target_count": event_scope_count,
            "A4_zero_shot_macro_query_gain": a4_zero_shot_macro_gain,
            "A4_positive_query_gain_count": a4_positive_count,
            "A3_macro_adapt_auc": a3_macro_auc,
            "A4_macro_adapt_auc": a4_macro_auc,
            "A5_macro_adapt_auc": a5_macro_auc,
            "A5_minus_A3_macro_adapt_auc": a5_macro_auc - a3_macro_auc,
            "A4_minus_incumbent_macro_query_accuracy": a4_zero_shot_macro_gain,
            "A5_minus_A4_macro_adapt_auc": a5_macro_auc - a4_macro_auc,
            "negative_A5_vs_A3_target_count": negative_a5_a3_count,
            "A5_policy_event_harm_max": a5_event_harm_max,
            "frozen_gate_pass": gate_pass,
        },
        "consumer_fit_count": consumer_fit_count,
        "target_test_split_load_count": test_split_load_count,
        "official_target_test_outcome_opened_once_per_dataset": True,
        "outcome_computed_once": True,
        "verdict": (
            "CONTROLLED_PROMOTED_BINDING_CAPABILITY_TRANSFER_PASS"
            if gate_pass
            else "CONTROLLED_PROMOTED_BINDING_CAPABILITY_TRANSFER_FAIL"
        ),
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
        "autonomous_llm_update": False,
        "natural_defect_claim": False,
        "general_safety_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": (
            "If the frozen gate passes, use this controlled result only as evidence for "
            "Source-Capability transfer behavior; do not generalize it to natural defects."
            if gate_pass
            else "Close or redesign this controlled transfer slice without tuning the opened roster."
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
