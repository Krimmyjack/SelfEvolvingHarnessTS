"""Small support-only executors for the admitted natural Workflow templates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


ScoreSupport = Callable[[Any], Any]
GroupPredict = Callable[[tuple[int, ...], tuple[int, int], float], Mapping[str, Any]]
CandidateTieBreak = Callable[[Mapping[str, Any]], tuple[Any, ...]]


def _reference_arrays(reference: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = (
        "baseline_prediction",
        "candidate_directions",
        "candidate_full_residual",
        "evaluation_design",
        "first_order_proxy_predictions",
    )
    if not isinstance(reference, Mapping) or any(key not in reference for key in required):
        raise ValueError("incomplete Ridge reference")
    arrays = {key: np.asarray(reference[key], dtype=np.float64) for key in required}
    baseline = arrays["baseline_prediction"]
    directions = arrays["candidate_directions"]
    residual = arrays["candidate_full_residual"]
    design = arrays["evaluation_design"]
    singleton = arrays["first_order_proxy_predictions"]
    if (
        baseline.ndim != 2
        or directions.ndim != 2
        or residual.ndim != 2
        or design.ndim != 2
        or singleton.ndim != 3
        or design.shape[0] != baseline.shape[0]
        or directions.shape[1] != residual.shape[0]
        or design.shape[1] != directions.shape[0]
        or residual.shape[1] != baseline.shape[1]
        or singleton.shape != (residual.shape[0], *baseline.shape)
        or not all(np.isfinite(value).all() for value in arrays.values())
    ):
        raise ValueError("invalid Ridge reference geometry")
    return arrays


def _losses(score_support: ScoreSupport, prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(score_support(prediction), dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("score_support must return finite per-support losses")
    return values


def _exact_prediction(
    group_predict: GroupPredict,
    rows: tuple[int, ...],
    target_block: tuple[int, int],
    removal_strength: float,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, Mapping[str, Any]]:
    result = group_predict(rows, target_block, removal_strength)
    if not isinstance(result, Mapping) or "exact_group_prediction" not in result:
        raise ValueError("group_predict must return exact_group_prediction")
    prediction = np.asarray(result["exact_group_prediction"], dtype=np.float64)
    if prediction.shape != expected_shape or not np.isfinite(prediction).all():
        raise ValueError("group_predict returned invalid prediction geometry")
    return prediction, result


def execute_whole_group_curation_support_only(
    reference: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    doses: Sequence[Mapping[str, Any]],
    *,
    target_block: tuple[int, int],
    score_support: ScoreSupport,
    group_predict: GroupPredict,
    candidate_tiebreak: CandidateTieBreak | None = None,
) -> dict[str, Any]:
    """Proxy-rank group×dose actions and exact-confirm only the top action."""

    arrays = _reference_arrays(reference)
    baseline = arrays["baseline_prediction"]
    baseline_losses = _losses(score_support, baseline)
    start, stop = target_block
    if start < 0 or stop > baseline.shape[1] or start >= stop:
        raise ValueError("invalid target_block")
    candidates = []
    for group in groups:
        group_id = str(group.get("group_id", ""))
        rows = tuple(int(value) for value in group.get("selected_rows", ()))
        if not group_id or not rows or min(rows) < 0 or max(rows) >= arrays[
            "candidate_full_residual"
        ].shape[0]:
            raise ValueError("invalid whole-group binding")
        selected = np.asarray(rows, dtype=np.int64)
        for dose in doses:
            action_id = str(dose.get("action_id", ""))
            strength = float(dose.get("removal_strength", float("nan")))
            if not action_id or not 0.0 < strength <= 1.0:
                raise ValueError("invalid curation dose")
            proxy = baseline.copy()
            proxy[:, start:stop] -= strength * (
                (arrays["evaluation_design"] @ arrays["candidate_directions"][:, selected])
                @ arrays["candidate_full_residual"][selected, start:stop]
            )
            gain = float(np.mean(baseline_losses - _losses(score_support, proxy)))
            candidates.append(
                {
                    "group_id": group_id,
                    "action_id": action_id,
                    "selected_rows": rows,
                    "removal_strength": strength,
                    "proxy_gain": gain,
                }
            )
    if not candidates:
        raise ValueError("whole-group curation requires candidates")
    winner = max(
        candidates,
        key=(
            (lambda row: float(row["proxy_gain"]))
            if candidate_tiebreak is None
            else lambda row: (
                float(row["proxy_gain"]),
                *candidate_tiebreak(row),
            )
        ),
    )
    exact, exact_meta = _exact_prediction(
        group_predict,
        winner["selected_rows"],
        target_block,
        float(winner["removal_strength"]),
        baseline.shape,
    )
    support_gain = float(np.mean(baseline_losses - _losses(score_support, exact)))
    executes = support_gain > 0.0
    return {
        "decision": "EXECUTE" if executes else "ABSTAIN",
        "support_gain": support_gain,
        "bound_action": {**winner, "selected_rows": list(winner["selected_rows"])},
        "prediction": exact if executes else baseline.copy(),
        "diagnostics": {
            "candidate_count": len(candidates),
            "exact_confirmation_count": 1,
            "small_matrix_solve_count": int(
                exact_meta.get("small_matrix_solve_count", 0)
            ),
        },
    }


def execute_rowblock_support_only(
    reference: Mapping[str, Any],
    blocks: Sequence[tuple[int, int]],
    *,
    score_support: ScoreSupport,
    group_predict: GroupPredict,
) -> dict[str, Any]:
    """Compose proxy-positive singleton rows per block, then confirm once."""

    arrays = _reference_arrays(reference)
    baseline = arrays["baseline_prediction"]
    baseline_losses = _losses(score_support, baseline)
    candidate_count = arrays["first_order_proxy_predictions"].shape[0]
    composed = baseline.copy()
    bindings = []
    solve_count = 0
    for raw_block in blocks:
        start, stop = (int(raw_block[0]), int(raw_block[1]))
        if start < 0 or stop > baseline.shape[1] or start >= stop:
            raise ValueError("invalid rowblock interval")
        gains = []
        for row_index in range(candidate_count):
            proxy = baseline.copy()
            proxy[:, start:stop] = arrays["first_order_proxy_predictions"][
                row_index, :, start:stop
            ]
            gains.append(
                float(np.mean(baseline_losses - _losses(score_support, proxy)))
            )
        selected = [index for index, gain in enumerate(gains) if gain > 0.0]
        retained = None
        if len(selected) == candidate_count:
            retained = min(range(candidate_count), key=lambda index: (gains[index], index))
            selected.remove(retained)
        exact, exact_meta = _exact_prediction(
            group_predict,
            tuple(selected),
            (start, stop),
            1.0,
            baseline.shape,
        )
        composed[:, start:stop] = exact[:, start:stop]
        solve_count += int(exact_meta.get("small_matrix_solve_count", 0))
        bindings.append(
            {
                "block_half_open": [start, stop],
                "selected_rows": selected,
                "all_rows_guard_retained_index": retained,
            }
        )
    if not bindings:
        raise ValueError("rowblock requires at least one block")
    support_gain = float(
        np.mean(baseline_losses - _losses(score_support, composed))
    )
    executes = support_gain > 0.0
    return {
        "decision": "EXECUTE" if executes else "ABSTAIN",
        "support_gain": support_gain,
        "bound_groups": bindings,
        "prediction": composed if executes else baseline.copy(),
        "diagnostics": {
            "block_count": len(bindings),
            "exact_confirmation_count": len(bindings),
            "small_matrix_solve_count": solve_count,
        },
    }


def replay_bound_workflow_prediction(
    reference: Mapping[str, Any],
    support_result: Mapping[str, Any],
    *,
    group_predict: GroupPredict,
) -> np.ndarray:
    """Replay a Support-bound action on a later Ridge reference without labels."""

    arrays = _reference_arrays(reference)
    baseline = arrays["baseline_prediction"]
    if not isinstance(support_result, Mapping):
        raise ValueError("support_result must be an object")
    decision = support_result.get("decision")
    if decision == "ABSTAIN":
        return baseline.copy()
    if decision != "EXECUTE":
        raise ValueError("support_result decision must be EXECUTE or ABSTAIN")

    bound_action = support_result.get("bound_action")
    bound_groups = support_result.get("bound_groups")
    if isinstance(bound_action, Mapping) == isinstance(bound_groups, Sequence):
        raise ValueError("executed support_result requires one binding form")
    if isinstance(bound_action, Mapping):
        rows = tuple(int(value) for value in bound_action.get("selected_rows", ()))
        strength = float(bound_action.get("removal_strength", float("nan")))
        prediction, _ = _exact_prediction(
            group_predict,
            rows,
            (0, baseline.shape[1]),
            strength,
            baseline.shape,
        )
        return prediction

    if not isinstance(bound_groups, Sequence) or isinstance(bound_groups, (str, bytes)):
        raise ValueError("rowblock replay requires bound_groups")
    composed = baseline.copy()
    for binding in bound_groups:
        if not isinstance(binding, Mapping):
            raise ValueError("rowblock bindings must be objects")
        raw_block = binding.get("block_half_open")
        if not isinstance(raw_block, Sequence) or len(raw_block) != 2:
            raise ValueError("rowblock binding requires block_half_open")
        block = (int(raw_block[0]), int(raw_block[1]))
        if block[0] < 0 or block[0] >= block[1] or block[1] > baseline.shape[1]:
            raise ValueError("invalid rowblock replay interval")
        rows = tuple(int(value) for value in binding.get("selected_rows", ()))
        prediction, _ = _exact_prediction(
            group_predict, rows, block, 1.0, baseline.shape
        )
        composed[:, block[0] : block[1]] = prediction[:, block[0] : block[1]]
    if not bound_groups:
        raise ValueError("rowblock replay requires at least one binding")
    return composed


__all__ = [
    "execute_rowblock_support_only",
    "execute_whole_group_curation_support_only",
    "replay_bound_workflow_prediction",
]
