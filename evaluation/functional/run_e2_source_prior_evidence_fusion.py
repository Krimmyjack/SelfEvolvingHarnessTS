"""Run W57 controlled Source-prior Evidence-Fusion transfer.

W56 promoted one Source H1 Capability and found that weak Target Support ties
could erase that useful prior.  This runner changes only the Harness Update /
Evidence Fusion rule: retain promoted H1 on an incumbent or H0 Support tie;
rollback only on strictly negative evidence, or replace it with a strictly
better H0.  Observation, Scope/Risk, Program, Binding, Consumer, Metric, data
roles, and feedback budgets are unchanged.

``plan`` reads the admitted W56 report and official UCR TRAIN splits only.
``evaluate`` first rebuilds and executes every TRAIN plan, then opens each
official TEST split once, immediately before that dataset's shared readouts.
Query outcomes never participate in selection.

This is a controlled injected artifact on real UCR backgrounds.  Four Targets
include partially related dataset families.  It is not evidence for natural
defects, persistent Memory, a fully autonomous LLM, or general safety.  The
original UCI Target remains unopened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-source-prior-evidence-fusion/1"
DEFAULT_PLAN_PATH = (
    "artifacts/functional/e2/source_prior_evidence_fusion_plan.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_prior_evidence_fusion_report.json"
)
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_promoted_binding_capability_transfer_report.json"
)
TARGET_DATASETS = (
    "FreezerRegularTrain",
    "ToeSegmentation2",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
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
NEXT_STEP = "controlled impulse family closed, no threshold/roster tuning."


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _promoted_capability() -> dict[str, Any]:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        PROMOTED_CAPABILITY,
    )

    return PROMOTED_CAPABILITY


def _admit_source_report(source_report: dict[str, Any]) -> dict[str, Any]:
    """Validate only the W56 evidence needed to admit the Source prior."""

    overall = source_report.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("W56 Source report has no overall evidence")
    checks = {
        "verdict_PASS": source_report.get("verdict")
        == "CONTROLLED_PROMOTED_BINDING_CAPABILITY_TRANSFER_PASS",
        "A4_positive_query_gain_count_6": overall.get(
            "A4_positive_query_gain_count"
        )
        == 6,
        "A4_zero_shot_macro_query_gain_positive": float(
            overall.get("A4_zero_shot_macro_query_gain", float("-inf"))
        )
        > 0.0,
        "A5_minus_A3_macro_adapt_auc_positive": float(
            overall.get("A5_minus_A3_macro_adapt_auc", float("-inf"))
        )
        > 0.0,
        "negative_A5_vs_A3_target_count_0": overall.get(
            "negative_A5_vs_A3_target_count"
        )
        == 0,
        "A5_minus_A4_macro_adapt_auc_negative": float(
            overall.get("A5_minus_A4_macro_adapt_auc", float("inf"))
        )
        < 0.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"W56 Source prior is not admitted: {failed}")
    return {
        "source_report": SOURCE_REPORT_PATH,
        "source_admitted": True,
        "verdict": source_report["verdict"],
        "A4_positive_query_gain_count": 6,
        "A4_zero_shot_macro_query_gain": float(
            overall["A4_zero_shot_macro_query_gain"]
        ),
        "A5_minus_A3_macro_adapt_auc": float(
            overall["A5_minus_A3_macro_adapt_auc"]
        ),
        "negative_A5_vs_A3_target_count": 0,
        "A5_minus_A4_macro_adapt_auc": float(
            overall["A5_minus_A4_macro_adapt_auc"]
        ),
        "evidence_role": "lightweight admitted Source evidence; not persistent Memory",
    }


def _build_target_train_plan(root: Path, dataset: str) -> dict[str, Any]:
    """Reuse the W56/W55 generic TRAIN-only planner unchanged."""

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        _build_target_train_plan as _build_w56_target_train_plan,
    )

    return _build_w56_target_train_plan(root, dataset)


def _assert_target_plan(row: dict[str, Any]) -> None:
    dataset = str(row["dataset"])
    if row["controlled_artifact"]["scope"] != ELIGIBLE:
        raise ValueError(f"artifact eligibility changed: {dataset}")
    if row["stable_task_event"]["scope"] != CONTRAINDICATED:
        raise ValueError(f"event contraindication changed: {dataset}")
    if not row["bindings"][CANDIDATE_DYNAMIC]["exact_binding"]:
        raise ValueError(f"H1 binding is not exact: {dataset}")
    if row["bindings"][CANDIDATE_FIXED]["exact_binding"]:
        raise ValueError(f"H0 binding is not a mismatch: {dataset}")


def build_plan(root: Path) -> dict[str, Any]:
    source_evidence = _admit_source_report(_read_object(root / SOURCE_REPORT_PATH))
    targets = [_build_target_train_plan(root, dataset) for dataset in TARGET_DATASETS]
    for row in targets:
        _assert_target_plan(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "At the same Target downstream-feedback budget, retaining an admitted "
            "Source H1 prior under non-negative Target evidence prevents weak Support "
            "ties from erasing useful transferred behavior, while strict counterevidence "
            "still permits rollback or replacement."
        ),
        "scientific_role": "W57 controlled Harness Update / Evidence Fusion",
        "context_exposure": "INSTANCE_SEEN_TRAIN_ONLY",
        "outcome_exposure": "SEALED",
        "source_evidence": source_evidence,
        "compiled_source_capability": _promoted_capability(),
        "target_train_plans": targets,
        "frozen_policies": {
            "incumbent": "leave optional fit-only artifact untreated",
            "A3_target_only": (
                "W56 unchanged: B0 incumbent; exact mean over H0->H1 and H1->H0; "
                "a confirmed candidate must strictly beat incumbent; candidate ties prefer H0"
            ),
            "A4_source_only": (
                "W56 unchanged: execute promoted H1 at B0/B1/B2 without Target feedback"
            ),
            "A5_old": "W56 source-plus-target curve unchanged",
            "A5_new": (
                "B0 H1; after B1 H1 confirmation retain H1 when Support >= incumbent, "
                "otherwise rollback; after B2 H0 confirmation, H1 remains eligible at "
                ">= incumbent while H0 requires > incumbent, select highest Support, "
                "and prefer Source H1 on an H0/H1 tie"
            ),
            "confirmation_order": list(SOURCE_ORDER),
            "feedback_budgets": [0, 1, 2],
            "budget_unit": "one complete Target Support Consumer refit outcome",
            "selection_used_target_query_outcome": False,
        },
        "one_allowed_change": (
            "Harness Update / Evidence Fusion tie semantics for promoted Source H1"
        ),
        "frozen_surfaces": {
            "task_context": "classification-local-event-quality, unchanged",
            "observation": "W52/W55 cross-cohort Observation, unchanged",
            "scope": "fit-only artifact eligible; stable event contraindicated, unchanged",
            "program_binding": "W55 dynamic observer-node Binding for H1, unchanged",
            "program": "center-excluded local median, unchanged",
            "consumer": "ridge-raw-plus-difference-v1, unchanged",
            "metric": "classification accuracy, unchanged",
            "target_roster": list(TARGET_DATASETS),
            "proxy": "none",
        },
        "frozen_success_gate": {
            "source_admitted": True,
            "target_dataset_count": 4,
            "H1_binding_exact_target_count": 4,
            "H0_binding_mismatch_target_count": 4,
            "event_scope_unchanged_target_count": 4,
            "A5_new_vs_A5_old_behavior_difference_target_count_min": 1,
            "macro_A5_new_adapt_auc_strictly_greater_than_A5_old": True,
            "macro_A5_new_adapt_auc_strictly_greater_than_A3": True,
            "negative_A5_new_vs_A3_target_count": 0,
            "negative_A5_new_vs_A4_target_count": 0,
            "A5_new_policy_event_harm_max": 0.0,
        },
        "target_test_values_or_labels_read": False,
        "selection_used_target_query_outcome": False,
        "persistent_memory_built": False,
        "fully_autonomous_llm_update": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "Real UCR backgrounds with a controlled injected artifact; four Targets, "
            "including partially related dataset families. Not a natural-defect result, "
            "persistent Memory, fully autonomous LLM update, or general safety claim. "
            "The original UCI Target remains unopened."
        ),
        "next_step": NEXT_STEP,
    }


def _target_only_curve(
    order: tuple[str, str],
    candidates: dict[str, dict[str, float]],
    incumbent: dict[str, float],
) -> list[dict[str, Any]]:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        _target_only_curve as _w56_target_only_curve,
    )

    return _w56_target_only_curve(order, candidates, incumbent)


def _source_only_curve(
    candidates: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        _source_only_curve as _w56_source_only_curve,
    )

    return _w56_source_only_curve(candidates)


def _source_plus_target_old_curve(
    candidates: dict[str, dict[str, float]],
    incumbent: dict[str, float],
) -> list[dict[str, Any]]:
    """Call W56's old Source-plus-Target curve exactly."""

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        _source_plus_target_curve as _w56_source_plus_target_curve,
    )

    return _w56_source_plus_target_curve(candidates, incumbent)


def _source_prior_selection(
    confirmed: list[str],
    candidates: dict[str, dict[str, float]],
    incumbent: dict[str, float],
) -> str | None:
    """Fuse Source prior and Target evidence without erasing H1 on a tie."""

    incumbent_support = float(incumbent["support_accuracy"])
    h1_eligible = (
        CANDIDATE_DYNAMIC in confirmed
        and float(candidates[CANDIDATE_DYNAMIC]["support_accuracy"])
        >= incumbent_support
    )
    h0_eligible = (
        CANDIDATE_FIXED in confirmed
        and float(candidates[CANDIDATE_FIXED]["support_accuracy"])
        > incumbent_support
    )
    eligible = [
        candidate
        for candidate, admitted in (
            (CANDIDATE_DYNAMIC, h1_eligible),
            (CANDIDATE_FIXED, h0_eligible),
        )
        if admitted
    ]
    if not eligible:
        return None
    tie_rank = {CANDIDATE_DYNAMIC: 0, CANDIDATE_FIXED: 1}
    return min(
        eligible,
        key=lambda candidate: (
            -float(candidates[candidate]["support_accuracy"]),
            tie_rank[candidate],
        ),
    )


def _source_plus_target_new_curve(
    candidates: dict[str, dict[str, float]],
    incumbent: dict[str, float],
) -> list[dict[str, Any]]:
    """Execute H1 at B0 and revise it only under strict Target evidence."""

    confirmed: list[str] = []
    curve: list[dict[str, Any]] = []
    for budget in range(3):
        if budget == 0:
            selected: str | None = CANDIDATE_DYNAMIC
            basis = "promoted_source_capability_zero_shot"
        else:
            selected = _source_prior_selection(confirmed, candidates, incumbent)
            basis = "source_prior_strict_counterevidence_fusion"
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
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        _average_curves as _w56_average_curves,
    )

    return _w56_average_curves(curves)


def _adapt_auc(curve: list[dict[str, Any]]) -> float:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        _adapt_auc as _w56_adapt_auc,
    )

    return _w56_adapt_auc(curve)


def _prepare_train_execution(
    root: Path, dataset: str, frozen: dict[str, Any]
) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_promoted_binding_capability_transfer import (
        _prepare_train_execution as _prepare_w56_train_execution,
    )

    return _prepare_w56_train_execution(root, dataset, frozen)


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
        raise ValueError("Source-prior Evidence-Fusion plan revision mismatch")
    if plan.get("outcome_exposure") != "SEALED":
        raise ValueError("W57 Target outcomes were not sealed")
    if plan.get("compiled_source_capability") != _promoted_capability():
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
        raise ValueError("W57 Target roster or order is not frozen and unique")

    # Phase 1 is TRAIN-only for all Targets. No TEST split is opened until every
    # rebuilt plan and all TRAIN execution/scope/binding assertions pass.
    prepared: dict[str, dict[str, Any]] = {}
    for dataset in TARGET_DATASETS:
        current = _build_target_train_plan(root, dataset)
        frozen = planned[dataset]
        if current != frozen:
            raise ValueError(f"planned TRAIN context changed: {dataset}")
        _assert_target_plan(frozen)
        prepared[dataset] = _prepare_train_execution(root, dataset, frozen)

    # Phase 2 loads each TEST once, immediately before its three shared Ridge
    # readouts. Selection below uses Support accuracy only.
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
        a5_old_curve = _source_plus_target_old_curve(candidates, incumbent)
        a5_new_curve = _source_plus_target_new_curve(candidates, incumbent)
        a3_auc = _adapt_auc(a3_curve)
        a4_auc = _adapt_auc(a4_curve)
        a5_old_auc = _adapt_auc(a5_old_curve)
        a5_new_auc = _adapt_auc(a5_new_curve)
        behavior_difference_budgets = [
            budget
            for budget in range(3)
            if a5_old_curve[budget]["selected_candidate"]
            != a5_new_curve[budget]["selected_candidate"]
        ]
        readouts = {
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
                "readouts": readouts,
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
                "A5_old": {
                    "confirmation_order": list(SOURCE_ORDER),
                    "mean_curve": a5_old_curve,
                    "selection_path": [
                        point["selected_candidate"] for point in a5_old_curve
                    ],
                    "adapt_auc": a5_old_auc,
                },
                "A5_new": {
                    "confirmation_order": list(SOURCE_ORDER),
                    "mean_curve": a5_new_curve,
                    "selection_path": [
                        point["selected_candidate"] for point in a5_new_curve
                    ],
                    "adapt_auc": a5_new_auc,
                },
                "comparisons": {
                    "A5_new_minus_A5_old_adapt_auc": a5_new_auc - a5_old_auc,
                    "A5_new_minus_A3_adapt_auc": a5_new_auc - a3_auc,
                    "A5_new_minus_A4_adapt_auc": a5_new_auc - a4_auc,
                    "A4_minus_incumbent_query_accuracy": float(h1["query_accuracy"])
                    - float(incumbent["query_accuracy"]),
                },
                "behavior_difference": {
                    "differs_from_A5_old": bool(behavior_difference_budgets),
                    "budgets": behavior_difference_budgets,
                    "old_path": [
                        point["selected_candidate"] for point in a5_old_curve
                    ],
                    "new_path": [
                        point["selected_candidate"] for point in a5_new_curve
                    ],
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
    behavior_difference_targets = [
        str(row["dataset"])
        for row in rows
        if row["behavior_difference"]["differs_from_A5_old"]
    ]
    behavior_difference_budgets_by_target = {
        str(row["dataset"]): list(row["behavior_difference"]["budgets"])
        for row in rows
        if row["behavior_difference"]["differs_from_A5_old"]
    }
    behavior_difference_budget_count = sum(
        len(row["behavior_difference"]["budgets"]) for row in rows
    )
    a4_query_gains = [
        float(row["comparisons"]["A4_minus_incumbent_query_accuracy"])
        for row in rows
    ]
    a4_positive_count = sum(gain > 0.0 for gain in a4_query_gains)
    a4_macro_query_gain = sum(a4_query_gains) / len(rows)
    a3_macro_auc = sum(float(row["A3_target_only"]["adapt_auc"]) for row in rows) / len(rows)
    a4_macro_auc = sum(float(row["A4_source_only"]["adapt_auc"]) for row in rows) / len(rows)
    a5_old_macro_auc = sum(float(row["A5_old"]["adapt_auc"]) for row in rows) / len(rows)
    a5_new_macro_auc = sum(float(row["A5_new"]["adapt_auc"]) for row in rows) / len(rows)
    negative_new_a3_count = sum(
        float(row["comparisons"]["A5_new_minus_A3_adapt_auc"]) < -1e-12
        for row in rows
    )
    negative_new_a4_count = sum(
        float(row["comparisons"]["A5_new_minus_A4_adapt_auc"]) < -1e-12
        for row in rows
    )
    event_harm_max = max(
        float(row["stable_event_policy"]["policy_event_harm"]) for row in rows
    )
    source_admitted = bool(current_source["source_admitted"])
    gate_pass = bool(
        source_admitted
        and len(rows) == 4
        and h1_exact_count == 4
        and h0_mismatch_count == 4
        and event_scope_count == 4
        and len(behavior_difference_targets) >= 1
        and a5_new_macro_auc > a5_old_macro_auc
        and a5_new_macro_auc > a3_macro_auc
        and negative_new_a3_count == 0
        and negative_new_a4_count == 0
        and event_harm_max <= 1e-12
    )
    macro_curves = {
        policy: _macro_curve(rows, policy)
        for policy in ("A3_target_only", "A4_source_only", "A5_old", "A5_new")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": plan["causal_hypothesis"],
        "scientific_role": plan["scientific_role"],
        "source_evidence": current_source,
        "compiled_source_capability": plan["compiled_source_capability"],
        "one_allowed_change": plan["one_allowed_change"],
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
            "A4_positive_query_gain_count": a4_positive_count,
            "A4_zero_shot_macro_query_gain": a4_macro_query_gain,
            "A3_macro_adapt_auc": a3_macro_auc,
            "A4_macro_adapt_auc": a4_macro_auc,
            "A5_old_macro_adapt_auc": a5_old_macro_auc,
            "A5_new_macro_adapt_auc": a5_new_macro_auc,
            "A5_new_minus_A5_old_macro_adapt_auc": a5_new_macro_auc
            - a5_old_macro_auc,
            "A5_new_minus_A3_macro_adapt_auc": a5_new_macro_auc - a3_macro_auc,
            "A5_new_minus_A4_macro_adapt_auc": a5_new_macro_auc - a4_macro_auc,
            "A5_new_vs_A5_old_behavior_difference_target_count": len(
                behavior_difference_targets
            ),
            "A5_new_vs_A5_old_behavior_difference_targets": behavior_difference_targets,
            "A5_new_vs_A5_old_behavior_difference_budget_count": behavior_difference_budget_count,
            "A5_new_vs_A5_old_behavior_difference_budgets_by_target": behavior_difference_budgets_by_target,
            "negative_A5_new_vs_A3_target_count": negative_new_a3_count,
            "negative_A5_new_vs_A4_target_count": negative_new_a4_count,
            "A5_new_policy_event_harm_max": event_harm_max,
            "frozen_gate_pass": gate_pass,
        },
        "consumer_fit_count": consumer_fit_count,
        "target_test_split_load_count": test_split_load_count,
        "official_target_test_outcome_opened_once_per_dataset": True,
        "outcome_computed_once": True,
        "verdict": (
            "CONTROLLED_SOURCE_PRIOR_EVIDENCE_FUSION_PASS"
            if gate_pass
            else "CONTROLLED_SOURCE_PRIOR_EVIDENCE_FUSION_FAIL"
        ),
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
        "fully_autonomous_llm_update": False,
        "natural_defect_claim": False,
        "general_safety_claim": False,
        "claim_limit": plan["claim_limit"],
        "next_step": NEXT_STEP,
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
