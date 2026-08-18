"""Calibrate an exact low-rank Ridge action-credit update on exposed episodes.

W49 showed that a first-order support-loss proxy can falsely veto a large group
feature edit because it omits the Program-induced Gram update.  The bound-node
Program changes only a few raw/difference feature columns, so that Gram update
is low rank.  This exposed-data instrument checks a Woodbury correction against
a direct counterfactual Ridge solve and compares both credit signals with the
already opened W48/W49 Query outcomes.

No new Query is opened and this mechanical calibration is not a Capability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-curvature-corrected-action-credit/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_curvature_corrected_action_credit_report.json"
)
W48_REPORT_PATH = (
    "artifacts/functional/e2/source_task_context_label_evidence_witness_report.json"
)
W49_PLAN_PATH = (
    "artifacts/functional/e2/source_task_risk_action_credit_transfer_plan.json"
)
W49_REPORT_PATH = (
    "artifacts/functional/e2/source_task_risk_action_credit_transfer_report.json"
)
DATA_DIR = "data/ucr_task_context"
DATASETS = (
    "Coffee",
    "ECG200",
    "FordA",
    "GunPoint",
    "Wafer",
    "ECGFiveDays",
    "TwoLeadECG",
    "BeetleFly",
)
RIDGE_ALPHA = 1.0
MODIFIED_COLUMN_TOLERANCE = 1e-12
MECHANICAL_MAX_ABS_ERROR = 1e-8


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _curvature_corrected_coefficient(
    np: Any,
    *,
    reference_design: Any,
    action_design: Any,
    targets: Any,
    inverse_product: Any,
    alpha: float = RIDGE_ALPHA,
) -> dict[str, Any]:
    """Apply the exact low-rank Gram and right-hand-side update."""

    z = np.asarray(reference_design, dtype=np.float64)
    action = np.asarray(action_design, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if z.shape != action.shape or y.shape != (z.shape[0],):
        raise ValueError("invalid low-rank action geometry")
    delta = action - z
    modified_columns = np.flatnonzero(
        np.any(np.abs(delta) > MODIFIED_COLUMN_TOLERANCE, axis=0)
    )
    if modified_columns.size < 1:
        raise ValueError("action has no modified design columns")
    residual_columns = delta[:, modified_columns]
    selector = np.eye(z.shape[1], dtype=np.float64)[:, modified_columns]
    cross = z.T @ residual_columns
    local_gram = residual_columns.T @ residual_columns
    basis = np.column_stack((cross, selector))
    count = modified_columns.size
    identity = np.eye(count, dtype=np.float64)
    zero = np.zeros((count, count), dtype=np.float64)
    middle_factor = np.block([[zero, identity], [identity, local_gram]])
    middle_factor_inverse = np.block(
        [[-local_gram, identity], [identity, zero]]
    )
    factor_inverse_error = float(
        np.max(
            np.abs(
                middle_factor @ middle_factor_inverse
                - np.eye(2 * count, dtype=np.float64)
            )
        )
    )
    reference_rhs = z.T @ y
    action_rhs = reference_rhs + selector @ (residual_columns.T @ y)
    solved = inverse_product(
        np,
        z,
        np.column_stack((action_rhs, basis)),
        alpha,
    )
    inverse_action_rhs = solved[:, 0]
    inverse_basis = solved[:, 1:]
    correction_system = (
        middle_factor_inverse + basis.T @ inverse_basis
    )
    correction = np.linalg.solve(
        correction_system, basis.T @ inverse_action_rhs
    )
    coefficient = inverse_action_rhs - inverse_basis @ correction
    if not np.isfinite(coefficient).all():
        raise RuntimeError("non-finite curvature-corrected coefficient")
    return {
        "coefficient": coefficient,
        "modified_design_columns": [int(index) for index in modified_columns],
        "modified_design_column_count": int(count),
        "woodbury_rank_bound": int(2 * count),
        "small_system_dimension": int(2 * count),
        "small_system_condition_number": float(np.linalg.cond(correction_system)),
        "middle_factor_inverse_max_abs_error": factor_inverse_error,
        "reference_inverse_multi_rhs_solve_count": 1,
        "counterfactual_full_design_solve_count": 0,
    }


def _actual_and_proxy_maps(root: Path) -> tuple[dict[str, float], dict[str, float]]:
    w48 = _read_object(root / W48_REPORT_PATH)
    w49_plan = _read_object(root / W49_PLAN_PATH)
    w49 = _read_object(root / W49_REPORT_PATH)
    actual = {
        str(row["dataset"]): float(
            row["conditions"]["fit_only_artifact"]["forced_query_gain"]
        )
        for row in w48["dataset_evidence"]
    }
    actual.update(
        {
            str(row["dataset"]): float(
                row["conditions"]["fit_only_artifact"]["actual_query_gain"]
            )
            for row in w49["dataset_evidence"]
        }
    )
    first_order = {
        str(row["dataset"]): float(row["proxy_credit"])
        for row in w49_plan["source_calibration"]
    }
    first_order.update(
        {
            str(row["dataset"]): float(
                row["conditions"]["fit_only_artifact"]["action_credit"]["group"][
                    "proxy_credit"
                ]
            )
            for row in w49_plan["target_train_plans"]
        }
    )
    if set(actual) != set(DATASETS) or set(first_order) != set(DATASETS):
        raise ValueError("exposed W48/W49 episode set changed")
    return actual, first_order


def _retention(actual: dict[str, float], credit: dict[str, float]) -> float:
    available = sum(max(0.0, value) for value in actual.values())
    retained = sum(
        max(0.0, actual[dataset])
        for dataset in DATASETS
        if credit[dataset] > 0.0
    )
    return retained / available if available > 0.0 else 0.0


def run(root: Path) -> dict[str, Any]:
    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        _apply_bound_impulse_oracle,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _bound_positions,
        _features,
        _inject,
        _load_split,
        _split_fit_support,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_risk_action_credit_transfer import (
        _inverse_product,
    )

    actual, first_order = _actual_and_proxy_maps(root)
    rows: list[dict[str, Any]] = []
    curvature_credit: dict[str, float] = {}
    for dataset in DATASETS:
        archive = root / DATA_DIR / f"{dataset}.zip"
        values, labels = _load_split(np, archive, dataset, "TRAIN")
        fit_indices, support_indices = _split_fit_support(np, labels)
        positions = _bound_positions(values.shape[1])
        fit_values = _inject(np, values[fit_indices], labels[fit_indices], positions)
        support_values = values[support_indices]
        fit_labels = labels[fit_indices]
        support_labels = labels[support_indices]
        repaired, _ = _apply_bound_impulse_oracle(
            np,
            fit_values,
            positions=positions,
            window_length=values.shape[1],
        )
        reference_design = np.column_stack(
            (_features(np, fit_values), np.ones(fit_values.shape[0], dtype=np.float64))
        )
        action_design = np.column_stack(
            (_features(np, repaired), np.ones(repaired.shape[0], dtype=np.float64))
        )
        support_design = np.column_stack(
            (
                _features(np, support_values),
                np.ones(support_values.shape[0], dtype=np.float64),
            )
        )
        fit_targets = np.where(fit_labels == 1, 1.0, -1.0)
        support_targets = np.where(support_labels == 1, 1.0, -1.0)
        reference_coefficient = _inverse_product(
            np,
            reference_design,
            reference_design.T @ fit_targets,
            RIDGE_ALPHA,
        )
        corrected = _curvature_corrected_coefficient(
            np,
            reference_design=reference_design,
            action_design=action_design,
            targets=fit_targets,
            inverse_product=_inverse_product,
        )
        direct_coefficient = _inverse_product(
            np,
            action_design,
            action_design.T @ fit_targets,
            RIDGE_ALPHA,
        )
        reference_prediction = support_design @ reference_coefficient
        corrected_prediction = support_design @ corrected["coefficient"]
        direct_prediction = support_design @ direct_coefficient
        baseline_loss = float(
            np.mean((reference_prediction - support_targets) ** 2)
        )
        corrected_loss = float(
            np.mean((corrected_prediction - support_targets) ** 2)
        )
        direct_loss = float(np.mean((direct_prediction - support_targets) ** 2))
        credit = baseline_loss - corrected_loss
        curvature_credit[dataset] = credit
        prediction_error = float(
            np.max(np.abs(corrected_prediction - direct_prediction))
        )
        coefficient_error = float(
            np.max(np.abs(corrected["coefficient"] - direct_coefficient))
        )
        rows.append(
            {
                "dataset": dataset,
                "fit_count": int(fit_indices.size),
                "support_count": int(support_indices.size),
                "first_order_proxy_credit": first_order[dataset],
                "curvature_corrected_credit": credit,
                "direct_action_solve_credit": baseline_loss - direct_loss,
                "exposed_query_gain": actual[dataset],
                "first_order_query_sign_agreement": (
                    first_order[dataset] > 0.0
                )
                == (actual[dataset] > 0.0),
                "curvature_query_sign_agreement": (credit > 0.0)
                == (actual[dataset] > 0.0),
                "corrected_vs_direct_prediction_max_abs_error": prediction_error,
                "corrected_vs_direct_coefficient_max_abs_error": coefficient_error,
                "mechanical_equivalence_pass": (
                    prediction_error <= MECHANICAL_MAX_ABS_ERROR
                    and coefficient_error <= MECHANICAL_MAX_ABS_ERROR
                ),
                "low_rank_update": {
                    key: value
                    for key, value in corrected.items()
                    if key != "coefficient"
                },
            }
        )

    all_mechanical = all(bool(row["mechanical_equivalence_pass"]) for row in rows)
    first_sign = sum(bool(row["first_order_query_sign_agreement"]) for row in rows) / len(rows)
    curvature_sign = sum(bool(row["curvature_query_sign_agreement"]) for row in rows) / len(rows)
    first_retention = _retention(actual, first_order)
    curvature_retention = _retention(actual, curvature_credit)
    harmful_first = sum(
        actual[dataset] < 0.0 and first_order[dataset] > 0.0
        for dataset in DATASETS
    )
    harmful_curvature = sum(
        actual[dataset] < 0.0 and curvature_credit[dataset] > 0.0
        for dataset in DATASETS
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "causal_hypothesis": (
            "For the sparse bound-node Program, retaining the exact low-rank Gram "
            "update mechanically recovers direct Ridge action credit and avoids the "
            "large-edit failure of a first-order approximation."
        ),
        "scientific_role": "exposed action-credit instrument calibration",
        "dataset_evidence": rows,
        "overall": {
            "dataset_count": len(rows),
            "all_low_rank_updates_match_direct_solve": all_mechanical,
            "maximum_prediction_abs_error": max(
                float(row["corrected_vs_direct_prediction_max_abs_error"])
                for row in rows
            ),
            "maximum_coefficient_abs_error": max(
                float(row["corrected_vs_direct_coefficient_max_abs_error"])
                for row in rows
            ),
            "first_order_query_sign_agreement": first_sign,
            "curvature_query_sign_agreement": curvature_sign,
            "first_order_positive_gain_retention": first_retention,
            "curvature_positive_gain_retention": curvature_retention,
            "first_order_harmful_execution_count": harmful_first,
            "curvature_harmful_execution_count": harmful_curvature,
        },
        "new_consumer_fit_count": 0,
        "new_query_opened": False,
        "original_uci_target_query_opened": False,
        "formal_capability_claim": False,
        "verdict": (
            "LOW_RANK_CURVATURE_CALIBRATION_PASS"
            if all_mechanical
            else "LOW_RANK_CURVATURE_CALIBRATION_FAIL"
        ),
        "claim_limit": (
            "This exposed calibration validates Ridge mechanics and retrospective credit "
            "behavior only; it is not fresh transfer or final Utility evidence."
        ),
        "next_step": (
            "Freeze a new-target budgeted EXECUTE versus REQUEST_FULL_CONFIRMATION slice."
            if all_mechanical
            else "Stop curvature correction and use full feedback for unresolved actions."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(root)
    output = args.output or root / DEFAULT_REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(payload["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
