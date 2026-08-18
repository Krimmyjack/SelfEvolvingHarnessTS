"""Measure oracle headroom for training-target reliability weighting.

This development-only experiment reuses the exposed E2-J0 Source roster and its
already selected endpoint corruption rows.  ``row`` mode excludes a whole unreliable
training window; ``target_cell`` mode excludes only its corrupted target interval and
keeps the other horizons.  Hidden injection labels are used only to test Program
headroom; they are not a deployable observation or Witness.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-training-target-reliability-weighting-headroom/1"
J0_PLAN_PATH = "artifacts/functional/e2/source_judge_readability_calibration_plan.json"
J0_REPORT_PATH = "artifacts/functional/e2/source_judge_readability_calibration_report.json"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_reliability_weighting_headroom_report.json"
)
TARGET_CELL_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_reliability_weighting_target_cell_headroom_report.json"
)
OBSERVED_INTERVAL_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_reliability_weighting_observed_interval_report.json"
)
PHASE_RESIDUAL_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_reliability_weighting_phase_residual_report.json"
)
RAW_PHASE_AGREEMENT_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_reliability_weighting_raw_phase_agreement_report.json"
)
CENSOR_ORACLE_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_censoring_oracle_headroom_report.json"
)
CENSOR_MASK_ORACLE_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_censoring_mask_headroom_report.json"
)
CENSOR_FLATLINE_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_censoring_flatline_observation_report.json"
)
FRESH_CENSOR_FLATLINE_REPORT_PATH = (
    "artifacts/functional/e2/"
    "source_training_target_censoring_flatline_frozen_replay_report.json"
)
CONTEXT_LENGTH = 192
HORIZON = 48
ANCHORS = (240, 300, 360, 420, 480, 540)
SEEDS = (0, 1, 2)
SELECTED_ROW_COUNT = 14
TARGET_BLOCK = (18, 30)
TARGET_OFFSET = 2.0
EXPECTED_FITS = 14
MIN_POSITIVE_UIDS = 6
MIN_RECOVERY_FRACTION = 0.5
OBSERVED_INTERVAL_LENGTH = 12
OBSERVED_FLANK_LENGTH = 6
OBSERVED_STARTS = tuple(range(6, 31))
OBSERVED_SCORE_THRESHOLD = 1.0
MIN_LOCALIZATION_PRECISION = 0.75
MIN_LOCALIZATION_RECALL = 0.75
SPECS = {
    "monash:traffic_hourly": {
        "train_stop": 928,
        "future_bounds": (928, 976),
        "period": 24,
    },
    "legacy_monash:fred_md": {
        "train_stop": 632,
        "future_bounds": (632, 680),
        "period": 12,
    },
}
FRESH_SPECS = {
    "legacy_monash:nn5_daily": {
        "train_stop": 632,
        "future_bounds": (632, 680),
        "period": 7,
    },
    "metr_la": {
        "train_stop": 928,
        "future_bounds": (928, 976),
        "period": 24,
    },
}


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _oracle_row_weights(np: Any, row_count: int, selected: set[int]) -> Any:
    """Return the fixed Program output: zero unreliable rows, retain all others."""

    if row_count < 1 or not selected or min(selected) < 0 or max(selected) >= row_count:
        raise ValueError("selected row indices must be a non-empty in-range set")
    weights = np.ones(row_count, dtype=np.float64)
    weights[sorted(selected)] = 0.0
    return weights


def _oracle_target_cell_weights(
    np: Any,
    row_count: int,
    horizon: int,
    selected: set[int],
    target_block: tuple[int, int],
) -> Any:
    """Zero only the selected-row cells inside the corrupted target interval."""

    start, stop = target_block
    if (
        row_count < 1
        or horizon < 1
        or not selected
        or min(selected) < 0
        or max(selected) >= row_count
        or start < 0
        or stop > horizon
        or start >= stop
    ):
        raise ValueError("selected cells must form a non-empty in-range rectangle")
    weights = np.ones((row_count, horizon), dtype=np.float64)
    weights[sorted(selected), start:stop] = 0.0
    return weights


def _apply_stuck_value_censoring(
    np: Any,
    clean_targets: Any,
    contexts: Any,
    selected: set[int],
    target_block: tuple[int, int],
) -> tuple[Any, Any, int]:
    """Create one legal flatline defect and its exact hidden-label restoration.

    The selected target block is replaced by the last visible context value, a
    stuck-sensor/last-value censoring mechanism.  The second output restores only
    those cells from the hidden clean copy and must equal the clean artifact.
    """

    clean = np.asarray(clean_targets, dtype=np.float64)
    visible = np.asarray(contexts, dtype=np.float64)
    start, stop = target_block
    if (
        clean.ndim != 2
        or visible.ndim != 2
        or clean.shape[0] != visible.shape[0]
        or not selected
        or min(selected) < 0
        or max(selected) >= clean.shape[0]
        or start < 0
        or stop > clean.shape[1]
        or start >= stop
        or not np.isfinite(clean).all()
        or not np.isfinite(visible).all()
    ):
        raise ValueError("invalid stuck-value censoring geometry")
    rows = sorted(selected)
    corrupted = clean.copy()
    corrupted[rows, start:stop] = visible[rows, -1, None]
    changed_cell_count = int(
        np.count_nonzero(corrupted[rows, start:stop] != clean[rows, start:stop])
    )
    if changed_cell_count < 1:
        raise ValueError("stuck-value censoring changed no target cells")
    restored = corrupted.copy()
    restored[rows, start:stop] = clean[rows, start:stop]
    if not np.array_equal(restored, clean):
        raise AssertionError("exact censoring restoration did not recover clean targets")
    return corrupted, restored, changed_cell_count


def _observe_interval(np: Any, target: Any) -> dict[str, object]:
    """Locate one fixed-length two-sided excursion without injection labels."""

    values = np.asarray(target, dtype=np.float64)
    if values.shape != (HORIZON,) or not np.isfinite(values).all():
        raise ValueError("observer requires one finite standardized target of length 48")
    best_start: int | None = None
    best_score = -1.0
    for start in OBSERVED_STARTS:
        stop = start + OBSERVED_INTERVAL_LENGTH
        block_median = float(np.median(values[start:stop]))
        left_delta = block_median - float(
            np.median(values[start - OBSERVED_FLANK_LENGTH : start])
        )
        right_delta = block_median - float(
            np.median(values[stop : stop + OBSERVED_FLANK_LENGTH])
        )
        score = (
            min(abs(left_delta), abs(right_delta))
            if left_delta * right_delta > 0.0
            else 0.0
        )
        # Strictly greater preserves the frozen earlier-start tie break.
        if score > best_score:
            best_start, best_score = start, score
    activated = best_start is not None and best_score >= OBSERVED_SCORE_THRESHOLD
    return {
        "status": "ACTIVATE" if activated else "ABSTAIN",
        "predicted_interval": (
            [best_start, best_start + OBSERVED_INTERVAL_LENGTH] if activated else None
        ),
        "score": best_score,
        "threshold": OBSERVED_SCORE_THRESHOLD,
    }


def _observe_flatline_interval(np: Any, target: Any) -> dict[str, object]:
    """Locate the first exact-length flatline using target values only."""

    values = np.asarray(target, dtype=np.float64)
    if values.shape != (HORIZON,) or not np.isfinite(values).all():
        raise ValueError("flatline observer requires one finite target of length 48")
    tolerance = 1e-12
    for start in range(0, HORIZON - OBSERVED_INTERVAL_LENGTH + 1):
        stop = start + OBSERVED_INTERVAL_LENGTH
        value_range = float(np.max(values[start:stop]) - np.min(values[start:stop]))
        if value_range <= tolerance:
            return {
                "status": "ACTIVATE",
                "predicted_interval": [start, stop],
                "value_range": value_range,
                "tolerance": tolerance,
            }
    return {
        "status": "ABSTAIN",
        "predicted_interval": None,
        "value_range": None,
        "tolerance": tolerance,
    }


def _censor_flatline_interval_weights(
    np: Any, targets: Any
) -> tuple[Any, list[dict[str, object]]]:
    values = np.asarray(targets, dtype=np.float64)
    if values.shape != (72, HORIZON):
        raise ValueError("flatline compiler requires a 72x48 target matrix")
    weights = np.ones(values.shape, dtype=np.float64)
    observations: list[dict[str, object]] = []
    for row_index, target in enumerate(values):
        observation = _observe_flatline_interval(np, target)
        interval = observation["predicted_interval"]
        if interval is not None:
            start, stop = int(interval[0]), int(interval[1])
            weights[row_index, start:stop] = 0.0
        observations.append(observation)
    return weights, observations


def _interval_iou(
    predicted: tuple[int, int] | list[int] | None,
    truth: tuple[int, int] | list[int],
) -> float:
    if predicted is None:
        return 0.0
    p_start, p_stop = int(predicted[0]), int(predicted[1])
    t_start, t_stop = int(truth[0]), int(truth[1])
    intersection = max(0, min(p_stop, t_stop) - max(p_start, t_start))
    union = max(p_stop, t_stop) - min(p_start, t_start)
    return intersection / union if union > 0 else 0.0


def _observed_interval_weights(
    np: Any, targets: Any
) -> tuple[Any, list[dict[str, object]]]:
    """Compile label-free row observations into a target-cell weight matrix."""

    values = np.asarray(targets, dtype=np.float64)
    if values.shape != (72, HORIZON):
        raise ValueError("observed interval compiler requires a 72x48 target matrix")
    weights = np.ones(values.shape, dtype=np.float64)
    observations: list[dict[str, object]] = []
    for row_index, target in enumerate(values):
        observation = _observe_interval(np, target)
        interval = observation["predicted_interval"]
        if interval is not None:
            start, stop = int(interval[0]), int(interval[1])
            weights[row_index, start:stop] = 0.0
        observations.append(observation)
    return weights, observations


def _phase_residual_observation(
    np: Any, context: Any, target: Any, *, period: int
) -> dict[str, object]:
    """Run the frozen scan on context-only nearest same-phase residuals."""

    visible_context = np.asarray(context, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    if (
        visible_context.shape != (CONTEXT_LENGTH,)
        or values.shape != (HORIZON,)
        or not np.isfinite(visible_context).all()
        or not np.isfinite(values).all()
        or period < 1
        or period > CONTEXT_LENGTH
    ):
        raise ValueError("phase residual requires finite context/target and valid period")
    residual = np.empty(HORIZON, dtype=np.float64)
    for target_index in range(HORIZON):
        cycles_back = target_index // period + 1
        donor_index = CONTEXT_LENGTH + target_index - cycles_back * period
        residual[target_index] = values[target_index] - visible_context[donor_index]
    observation = _observe_interval(np, residual)
    observation["donor_semantics"] = "context-only nearest same-phase donor"
    return observation


def _phase_residual_interval_weights(
    np: Any, contexts: Any, targets: Any, *, period: int
) -> tuple[Any, list[dict[str, object]]]:
    visible_contexts = np.asarray(contexts, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    if visible_contexts.shape != (72, CONTEXT_LENGTH) or values.shape != (72, HORIZON):
        raise ValueError("phase residual compiler requires 72 aligned context/target rows")
    weights = np.ones(values.shape, dtype=np.float64)
    observations: list[dict[str, object]] = []
    for row_index, (context, target) in enumerate(zip(visible_contexts, values)):
        observation = _phase_residual_observation(
            np, context, target, period=period
        )
        interval = observation["predicted_interval"]
        if interval is not None:
            start, stop = int(interval[0]), int(interval[1])
            weights[row_index, start:stop] = 0.0
        observations.append(observation)
    return weights, observations


def _agree_raw_phase_observations(
    raw: dict[str, object], phase: dict[str, object]
) -> dict[str, object]:
    raw_interval = raw.get("predicted_interval")
    phase_interval = phase.get("predicted_interval")
    agreement_iou = (
        _interval_iou(raw_interval, phase_interval)
        if raw_interval is not None and phase_interval is not None
        else 0.0
    )
    activated = (
        raw.get("status") == "ACTIVATE"
        and phase.get("status") == "ACTIVATE"
        and agreement_iou >= 0.5
    )
    return {
        "raw_status": raw.get("status"),
        "raw_interval": raw_interval,
        "raw_score": raw.get("score"),
        "phase_status": phase.get("status"),
        "phase_interval": phase_interval,
        "phase_score": phase.get("score"),
        "agreement_iou": agreement_iou,
        "status": "ACTIVATE" if activated else "ABSTAIN",
        "predicted_interval": phase_interval if activated else None,
        "score": phase.get("score"),
    }


def _raw_phase_agreement_weights(
    np: Any, contexts: Any, targets: Any, *, period: int
) -> tuple[Any, list[dict[str, object]]]:
    visible_contexts = np.asarray(contexts, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    if visible_contexts.shape != (72, CONTEXT_LENGTH) or values.shape != (72, HORIZON):
        raise ValueError("raw-phase agreement requires 72 aligned context/target rows")
    weights = np.ones(values.shape, dtype=np.float64)
    observations: list[dict[str, object]] = []
    for row_index, (context, target) in enumerate(zip(visible_contexts, values)):
        raw = _observe_interval(np, target)
        phase = _phase_residual_observation(np, context, target, period=period)
        observation = _agree_raw_phase_observations(raw, phase)
        interval = observation["predicted_interval"]
        if interval is not None:
            start, stop = int(interval[0]), int(interval[1])
            weights[row_index, start:stop] = 0.0
        observations.append(observation)
    return weights, observations


def _observation_program_classification(
    utility_classification: str,
    *,
    precision: float,
    recall: float,
) -> tuple[str, dict[str, bool]]:
    localization = {
        "row_precision_at_least_0_75": precision >= MIN_LOCALIZATION_PRECISION,
        "row_recall_at_least_0_75": recall >= MIN_LOCALIZATION_RECALL,
    }
    if not all(localization.values()):
        return "LOCALIZATION_FAIL", localization
    if utility_classification != "HEADROOM_PASS":
        return "LOCALIZATION_PASS_UTILITY_FAIL", localization
    return "OBSERVATION_PROGRAM_PASS", localization


def _dataset_classification(
    *,
    corruption_mean: float,
    corruption_positive_uids: int,
    recovery_mean: float,
    recovery_positive_uids: int,
    recovery_fraction: float | None,
) -> tuple[str, dict[str, bool]]:
    """Apply the frozen, deliberately small development decision rule."""

    readable = corruption_mean > 0.0 and corruption_positive_uids >= MIN_POSITIVE_UIDS
    gates = {
        "corruption_readable": readable,
        "oracle_mean_recovery_positive": recovery_mean > 0.0,
        "oracle_positive_uid_count_at_least_6": recovery_positive_uids >= MIN_POSITIVE_UIDS,
        "oracle_recovery_fraction_at_least_0_5": (
            recovery_fraction is not None
            and recovery_fraction >= MIN_RECOVERY_FRACTION
        ),
    }
    if not readable:
        return "CORRUPTION_UNREADABLE", gates
    if all(gates.values()):
        return "HEADROOM_PASS", gates
    return "READABLE_BUT_NO_ORACLE_RECOVERY", gates


def _selected_indices(
    j0_report: dict[str, Any],
    dataset_id: str,
    seed: int,
    row_index: dict[tuple[str, int, int], int],
) -> set[int]:
    rows = j0_report["p0_pre_fit_gate"]["dataset_results"][dataset_id][
        "intervention_checks"
    ]
    matches = [
        row
        for row in rows
        if int(row["seed"]) == seed
        and int(row["selected_row_count"]) == SELECTED_ROW_COUNT
    ]
    if len(matches) != 1:
        raise ValueError(f"missing unique J0 endpoint selection: {dataset_id}/{seed}")
    selected: set[int] = set()
    for raw_key in matches[0]["selected_row_keys"]:
        # J0 key is [historical dataset identity, series_uid, anchor, horizon].
        key = (str(raw_key[1]), int(raw_key[2]), int(raw_key[3]))
        if key not in row_index:
            raise ValueError(f"J0 endpoint row absent from reused roster: {key}")
        selected.add(row_index[key])
    if len(selected) != SELECTED_ROW_COUNT:
        raise ValueError(f"J0 endpoint selection is not {SELECTED_ROW_COUNT} unique rows")
    return selected


def run(root: Path, *, mode: str = "row") -> dict[str, object]:
    import numpy as np
    from sklearn.linear_model import Ridge

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        UndefinedSeasonalScale,
        seasonal_scale,
        smase,
    )
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import (
        _center_scale,
    )

    if mode not in {
        "row",
        "target_cell",
        "observed_interval",
        "phase_residual_interval",
        "raw_phase_agreement",
        "censor_oracle",
        "censor_mask_oracle",
        "censor_flatline_observed",
        "censor_flatline_fresh",
    }:
        raise ValueError(f"unsupported weighting mode: {mode}")

    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    records = {row.series_uid: row for row in registry_rows}
    fresh_replay = mode == "censor_flatline_fresh"
    specs = FRESH_SPECS if fresh_replay else SPECS
    j0_plan: dict[str, Any] | None = None
    j0_report: dict[str, Any] | None = None
    if fresh_replay:
        roster: list[dict[str, object]] = []
        for dataset_id, spec in specs.items():
            required_stop = max(max(ANCHORS) + HORIZON, int(spec["future_bounds"][1]))
            candidates = sorted(
                (
                    row
                    for row in registry_rows
                    if row.dataset_id == dataset_id and int(row.length) >= required_stop
                ),
                key=lambda row: row.series_uid,
            )
            candidate_values = _load_values(
                candidates, root / "data/benchmark_v0_2/clean_base"
            )
            # Integrity-only admission: every value used by the frozen train/eval
            # geometry must be finite.  No Consumer score or Program outcome enters.
            eligible = [
                row
                for row in candidates
                if np.isfinite(
                    np.asarray(
                        candidate_values[row.series_uid][48:required_stop], dtype=np.float64
                    )
                ).all()
            ]
            if len(eligible) < 20:
                raise ValueError(f"insufficient frozen Source candidates: {dataset_id}")
            for index, row in enumerate(eligible[:20]):
                roster.append(
                    {
                        "dataset_id": dataset_id,
                        "series_uid": row.series_uid,
                        "cohort": "train" if index < 12 else "eval",
                    }
                )
    else:
        j0_plan = _read_object(root / J0_PLAN_PATH)
        j0_report = _read_object(root / J0_REPORT_PATH)
        if j0_plan.get("schema_version") != "e2-j0-judge-readability-calibration/1":
            raise ValueError("unexpected J0 plan")
        if j0_report.get("classification") != (
            "READABLE_AT_INJECTED_DOSE_BUT_UNDERPOWERED_FOR_EPSILON"
        ):
            raise ValueError("J0 did not establish strong-effect readability")
        if j0_report.get("target_query_opened") is not False:
            raise ValueError("J0 Target/Query boundary is not closed")
        raw_roster = j0_plan.get("roster")
        if not isinstance(raw_roster, list) or len(raw_roster) != 40:
            raise ValueError("expected the exposed J0 12+8 roster for two datasets")
        roster = raw_roster
    if any(str(row.get("dataset_id", "")).startswith("uci") for row in roster):
        raise ValueError("UCI is forbidden in this Source experiment")

    selected_records = [records[str(row["series_uid"])] for row in roster]
    values = _load_values(selected_records, root / "data/benchmark_v0_2/clean_base")
    fit_count = 0
    scalar_subfit_count = 0
    dataset_evidence: list[dict[str, object]] = []

    for dataset_id, spec in specs.items():
        train_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "train"
        ]
        eval_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "eval"
        ]
        if len(train_rows) != 12 or len(eval_rows) != 8:
            raise ValueError(f"J0 roster geometry changed: {dataset_id}")

        x_rows: list[Any] = []
        y_rows: list[Any] = []
        row_keys: list[tuple[str, int, int]] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                raw = values[uid]
                context = np.asarray(
                    raw[anchor - CONTEXT_LENGTH : anchor], dtype=np.float64
                )
                target = np.asarray(raw[anchor : anchor + HORIZON], dtype=np.float64)
                if context.shape != (CONTEXT_LENGTH,) or target.shape != (HORIZON,):
                    raise ValueError(f"invalid training window: {uid}/{anchor}")
                if not np.isfinite(context).all() or not np.isfinite(target).all():
                    raise ValueError(f"non-finite training window: {uid}/{anchor}")
                center, scale, method = _center_scale(context)
                if method == "scale_floor_fallback":
                    raise ValueError(f"scale floor reached: {uid}/{anchor}")
                x_rows.append(
                    np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
                )
                y_rows.append((target - center) / scale)
                row_keys.append((uid, anchor, HORIZON))
        x_train = np.asarray(x_rows, dtype=np.float64)
        clean_y = np.asarray(y_rows, dtype=np.float64)
        if x_train.shape != (72, 384) or clean_y.shape != (72, HORIZON):
            raise AssertionError(f"unexpected training geometry: {dataset_id}")
        row_index = {key: index for index, key in enumerate(row_keys)}

        x_eval: list[Any] = []
        raw_future: list[Any] = []
        eval_uids: list[str] = []
        centers: list[float] = []
        scales: list[float] = []
        seasonal_by_uid: dict[str, float] = {}
        for row in eval_rows:
            uid = str(row["series_uid"])
            raw = values[uid]
            context = np.asarray(
                raw[spec["train_stop"] - CONTEXT_LENGTH : spec["train_stop"]],
                dtype=np.float64,
            )
            future = np.asarray(raw[slice(*spec["future_bounds"])], dtype=np.float64)
            if context.shape != (CONTEXT_LENGTH,) or future.shape != (HORIZON,):
                raise ValueError(f"invalid evaluation window: {uid}")
            if not np.isfinite(context).all() or not np.isfinite(future).all():
                raise ValueError(f"non-finite evaluation window: {uid}")
            center, scale, method = _center_scale(context)
            if method == "scale_floor_fallback":
                raise ValueError(f"evaluation scale floor reached: {uid}")
            try:
                seasonal_by_uid[uid] = seasonal_scale(
                    np.asarray(raw[: spec["train_stop"]], dtype=np.float64),
                    np.isfinite(raw[: spec["train_stop"]]),
                    period=int(spec["period"]),
                    min_pairs=32,
                )
            except (UndefinedSeasonalScale, ValueError) as error:
                raise ValueError(f"invalid evaluation sMASE scale: {uid}") from error
            x_eval.append(
                np.concatenate(((context - center) / scale, np.zeros(CONTEXT_LENGTH)))
            )
            raw_future.append(future)
            eval_uids.append(uid)
            centers.append(center)
            scales.append(scale)
        x_eval_array = np.asarray(x_eval, dtype=np.float64)
        raw_future_array = np.asarray(raw_future, dtype=np.float64)
        centers_array = np.asarray(centers, dtype=np.float64)
        scales_array = np.asarray(scales, dtype=np.float64)

        def score_predictions(normalized: Any) -> list[dict[str, object]]:
            normalized = np.asarray(normalized, dtype=np.float64)
            if normalized.shape != (8, HORIZON) or not np.isfinite(normalized).all():
                raise RuntimeError(f"invalid Ridge prediction: {dataset_id}")
            original = normalized * scales_array[:, None] + centers_array[:, None]
            return [
                {
                    "series_uid": uid,
                    "smase": smase(
                        raw_future_array[index],
                        original[index],
                        scale=seasonal_by_uid[uid],
                    ),
                    "original_unit_mae": float(
                        np.mean(np.abs(raw_future_array[index] - original[index]))
                    ),
                }
                for index, uid in enumerate(eval_uids)
            ]

        def fit_score(
            targets: Any, weights: Any | None = None
        ) -> tuple[list[dict[str, object]], Any]:
            nonlocal fit_count
            model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
            model.fit(x_train, targets, sample_weight=weights)
            fit_count += 1
            normalized = np.asarray(model.predict(x_eval_array), dtype=np.float64)
            return score_predictions(normalized), normalized

        clean_scores, _ = fit_score(clean_y)
        clean_by_uid = {str(row["series_uid"]): row for row in clean_scores}
        seed_rows: list[dict[str, object]] = []
        localization_rows: list[dict[str, object]] = []
        corrupt_smase_by_uid: dict[str, list[float]] = {uid: [] for uid in eval_uids}
        oracle_smase_by_uid: dict[str, list[float]] = {uid: [] for uid in eval_uids}
        for seed in SEEDS:
            selected = (
                set(
                    int(value)
                    for value in np.random.default_rng(seed).choice(
                        len(row_keys), size=SELECTED_ROW_COUNT, replace=False
                    )
                )
                if fresh_replay
                else _selected_indices(j0_report, dataset_id, seed, row_index)
            )
            selected_list = sorted(selected)
            restored_targets = None
            modified_target_cell_count = 0
            if mode in {
                "censor_oracle", "censor_mask_oracle", "censor_flatline_observed",
                "censor_flatline_fresh"
            }:
                corrupt, restored_targets, modified_target_cell_count = (
                    _apply_stuck_value_censoring(
                        np,
                        clean_y,
                        x_train[:, :CONTEXT_LENGTH],
                        selected,
                        TARGET_BLOCK,
                    )
                )
            else:
                corrupt = clean_y.copy()
                corrupt[selected_list, TARGET_BLOCK[0] : TARGET_BLOCK[1]] += TARGET_OFFSET
                if not np.array_equal(
                    corrupt[selected_list, TARGET_BLOCK[0] : TARGET_BLOCK[1]],
                    clean_y[selected_list, TARGET_BLOCK[0] : TARGET_BLOCK[1]] + TARGET_OFFSET,
                ):
                    raise AssertionError("endpoint corruption differs from J0 mechanism")
                modified_target_cell_count = SELECTED_ROW_COUNT * (
                    TARGET_BLOCK[1] - TARGET_BLOCK[0]
                )
            corrupt_scores, corrupt_predictions = fit_score(corrupt)
            if mode == "censor_oracle":
                if restored_targets is None:
                    raise AssertionError("missing exact restored targets")
                oracle_scores, _ = fit_score(restored_targets)
                zero_weight_count = 0
                positive_weight_count = int(clean_y.size)
                public_observations = None
                private_localization = None
            elif mode == "row":
                weights = _oracle_row_weights(np, len(row_keys), selected)
                if int(np.count_nonzero(weights == 0.0)) != SELECTED_ROW_COUNT:
                    raise AssertionError("oracle Program did not zero exactly 14 rows")
                oracle_scores, _ = fit_score(corrupt, weights)
                zero_weight_count = int(np.count_nonzero(weights == 0.0))
                positive_weight_count = int(np.count_nonzero(weights == 1.0))
                public_observations = None
                private_localization = None
            else:
                if mode in {"target_cell", "censor_mask_oracle"}:
                    cell_weights = _oracle_target_cell_weights(
                        np, len(row_keys), HORIZON, selected, TARGET_BLOCK
                    )
                    expected_zero_cells = SELECTED_ROW_COUNT * (
                        TARGET_BLOCK[1] - TARGET_BLOCK[0]
                    )
                    if int(np.count_nonzero(cell_weights == 0.0)) != expected_zero_cells:
                        raise AssertionError("oracle Program target-cell geometry is wrong")
                    public_observations = None
                    private_localization = None
                else:
                    # Labels enter only after this observer and mask have been compiled.
                    if mode in {"censor_flatline_observed", "censor_flatline_fresh"}:
                        cell_weights, observations = _censor_flatline_interval_weights(
                            np, corrupt
                        )
                    elif mode == "phase_residual_interval":
                        cell_weights, observations = _phase_residual_interval_weights(
                            np,
                            x_train[:, :CONTEXT_LENGTH],
                            corrupt,
                            period=int(spec["period"]),
                        )
                    elif mode == "raw_phase_agreement":
                        cell_weights, observations = _raw_phase_agreement_weights(
                            np,
                            x_train[:, :CONTEXT_LENGTH],
                            corrupt,
                            period=int(spec["period"]),
                        )
                    else:
                        cell_weights, observations = _observed_interval_weights(np, corrupt)
                    public_observations = []
                    private_localization = []
                    for row_number, (key, observation) in enumerate(
                        zip(row_keys, observations)
                    ):
                        predicted = observation["predicted_interval"]
                        selected_truth = row_number in selected
                        iou = (
                            _interval_iou(predicted, TARGET_BLOCK)
                            if selected_truth
                            else None
                        )
                        activated = predicted is not None
                        tp = bool(selected_truth and activated and iou >= 0.5)
                        fp = bool(
                            (not selected_truth and activated)
                            or (selected_truth and activated and iou < 0.5)
                        )
                        fn = bool(selected_truth and (not activated or iou < 0.5))
                        public_row = {"row_key": list(key), **observation}
                        public_observations.append(public_row)
                        grade = {
                            "true_selected": selected_truth,
                            "iou": iou,
                            "exact_bound_match": bool(
                                selected_truth and predicted == list(TARGET_BLOCK)
                            ),
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "unselected_false_activation": bool(
                                not selected_truth and activated
                            ),
                        }
                        private_localization.append(grade)
                        localization_rows.append(grade)
                oracle_predictions = corrupt_predictions.copy()
                affected_columns = [
                    column
                    for column in range(HORIZON)
                    if bool(np.any(cell_weights[:, column] == 0.0))
                ]
                for target_index in affected_columns:
                    scalar_model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
                    scalar_model.fit(
                        x_train,
                        corrupt[:, target_index],
                        sample_weight=cell_weights[:, target_index],
                    )
                    oracle_predictions[:, target_index] = scalar_model.predict(x_eval_array)
                    scalar_subfit_count += 1
                # The twelve scalar refits jointly implement one logical oracle policy fit.
                fit_count += 1
                oracle_scores = score_predictions(oracle_predictions)
                zero_weight_count = int(np.count_nonzero(cell_weights == 0.0))
                positive_weight_count = int(np.count_nonzero(cell_weights == 1.0))
            per_uid: list[dict[str, object]] = []
            for corrupt_row, oracle_row in zip(corrupt_scores, oracle_scores):
                uid = str(corrupt_row["series_uid"])
                corrupt_loss = float(corrupt_row["smase"])
                oracle_loss = float(oracle_row["smase"])
                clean_loss = float(clean_by_uid[uid]["smase"])
                corrupt_smase_by_uid[uid].append(corrupt_loss)
                oracle_smase_by_uid[uid].append(oracle_loss)
                per_uid.append(
                    {
                        "series_uid": uid,
                        "clean_smase": clean_loss,
                        "corrupt_smase": corrupt_loss,
                        "oracle_weighted_smase": oracle_loss,
                        "corruption_degradation": corrupt_loss - clean_loss,
                        "oracle_recovery": corrupt_loss - oracle_loss,
                    }
                )
            seed_rows.append(
                {
                    "seed": seed,
                    "selected_row_count": len(selected),
                    "modified_target_cell_count": modified_target_cell_count,
                    "zero_weight_count": zero_weight_count,
                    "positive_weight_count": positive_weight_count,
                    "public_observations": public_observations,
                    "private_localization_grader": private_localization,
                    "per_uid": per_uid,
                }
            )

        per_uid_summary: list[dict[str, object]] = []
        for uid in eval_uids:
            clean_loss = float(clean_by_uid[uid]["smase"])
            corrupt_mean = statistics.fmean(corrupt_smase_by_uid[uid])
            oracle_mean = statistics.fmean(oracle_smase_by_uid[uid])
            per_uid_summary.append(
                {
                    "series_uid": uid,
                    "clean_smase": clean_loss,
                    "seed_mean_corrupt_smase": corrupt_mean,
                    "seed_mean_oracle_weighted_smase": oracle_mean,
                    "seed_mean_corruption_degradation": corrupt_mean - clean_loss,
                    "seed_mean_oracle_recovery": corrupt_mean - oracle_mean,
                }
            )
        corruption_mean = statistics.fmean(
            float(row["seed_mean_corruption_degradation"]) for row in per_uid_summary
        )
        recovery_mean = statistics.fmean(
            float(row["seed_mean_oracle_recovery"]) for row in per_uid_summary
        )
        corruption_positive = sum(
            float(row["seed_mean_corruption_degradation"]) > 0.0
            for row in per_uid_summary
        )
        recovery_positive = sum(
            float(row["seed_mean_oracle_recovery"]) > 0.0 for row in per_uid_summary
        )
        recovery_fraction = (
            recovery_mean / corruption_mean if corruption_mean > 0.0 else None
        )
        utility_classification, gates = _dataset_classification(
            corruption_mean=corruption_mean,
            corruption_positive_uids=corruption_positive,
            recovery_mean=recovery_mean,
            recovery_positive_uids=recovery_positive,
            recovery_fraction=recovery_fraction,
        )
        localization_evidence = None
        classification = utility_classification
        if mode in {
            "observed_interval", "phase_residual_interval", "raw_phase_agreement",
            "censor_flatline_observed", "censor_flatline_fresh"
        }:
            tp = sum(bool(row["tp"]) for row in localization_rows)
            fp = sum(bool(row["fp"]) for row in localization_rows)
            fn = sum(bool(row["fn"]) for row in localization_rows)
            unselected = [row for row in localization_rows if not row["true_selected"]]
            selected_truth = [row for row in localization_rows if row["true_selected"]]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            classification, localization_gate = _observation_program_classification(
                utility_classification, precision=precision, recall=recall
            )
            localization_evidence = {
                "true_positive_iou_threshold": 0.5,
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "row_precision": precision,
                "row_recall": recall,
                "unselected_false_activation_rate": (
                    sum(bool(row["unselected_false_activation"]) for row in unselected)
                    / len(unselected)
                ),
                "selected_mean_iou": statistics.fmean(
                    float(row["iou"]) for row in selected_truth
                ),
                "selected_exact_bound_match_rate_diagnostic_only": (
                    sum(bool(row["exact_bound_match"]) for row in selected_truth)
                    / len(selected_truth)
                ),
                "gate": localization_gate,
            }
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "clean_reference_scores": clean_scores,
                "seed_evidence": seed_rows,
                "seed_averaged_per_uid": per_uid_summary,
                "mean_corruption_degradation": corruption_mean,
                "corruption_positive_uid_count": corruption_positive,
                "mean_oracle_recovery": recovery_mean,
                "oracle_positive_uid_count": recovery_positive,
                "oracle_recovery_fraction": recovery_fraction,
                "gate": gates,
                "utility_classification": utility_classification,
                "localization_evidence": localization_evidence,
                "classification": classification,
            }
        )

    if fit_count != EXPECTED_FITS:
        raise AssertionError(f"expected exactly {EXPECTED_FITS} Ridge fits, got {fit_count}")
    expected_scalar_subfits = (
        len(specs) * len(SEEDS) * (TARGET_BLOCK[1] - TARGET_BLOCK[0])
        if mode in {"target_cell", "censor_mask_oracle"}
        else 0 if mode in {"row", "censor_oracle"} else None
    )
    if expected_scalar_subfits is not None and scalar_subfit_count != expected_scalar_subfits:
        raise AssertionError(
            f"expected {expected_scalar_subfits} scalar subfits, got {scalar_subfit_count}"
        )
    classifications = [str(row["classification"]) for row in dataset_evidence]
    if mode in {
        "observed_interval", "phase_residual_interval", "raw_phase_agreement",
        "censor_flatline_observed", "censor_flatline_fresh"
    }:
        if all(value == "OBSERVATION_PROGRAM_PASS" for value in classifications):
            verdict = (
                "TRAINING_TARGET_FLATLINE_SOURCE_REPLAY_PASS"
                if fresh_replay
                else "TRAINING_TARGET_OBSERVATION_PROGRAM_PASS"
            )
            overall = "OBSERVATION_PROGRAM_PASS"
        elif any(value == "LOCALIZATION_FAIL" for value in classifications):
            verdict = (
                "TRAINING_TARGET_FLATLINE_SOURCE_REPLAY_LOCALIZATION_FAIL"
                if fresh_replay
                else "TRAINING_TARGET_OBSERVATION_LOCALIZATION_FAIL"
            )
            overall = "LOCALIZATION_FAIL"
        else:
            verdict = (
                "TRAINING_TARGET_FLATLINE_SOURCE_REPLAY_UTILITY_FAIL"
                if fresh_replay
                else "TRAINING_TARGET_OBSERVATION_UTILITY_FAIL"
            )
            overall = "LOCALIZATION_PASS_UTILITY_FAIL"
    elif all(value == "HEADROOM_PASS" for value in classifications):
        verdict = (
            "TRAINING_TARGET_CENSORING_MASK_HEADROOM_PASS"
            if mode == "censor_mask_oracle"
            else "TRAINING_TARGET_CENSORING_ORACLE_HEADROOM_PASS"
            if mode == "censor_oracle"
            else "TRAINING_TARGET_RELIABILITY_WEIGHTING_HEADROOM_PASS"
        )
        overall = "HEADROOM_PASS"
    elif any(value == "CORRUPTION_UNREADABLE" for value in classifications):
        verdict = (
            "TRAINING_TARGET_CENSORING_MASK_HEADROOM_UNAVAILABLE"
            if mode == "censor_mask_oracle"
            else "TRAINING_TARGET_CENSORING_ORACLE_HEADROOM_UNAVAILABLE"
            if mode == "censor_oracle"
            else "TRAINING_TARGET_RELIABILITY_WEIGHTING_HEADROOM_UNAVAILABLE"
        )
        overall = "UNAVAILABLE"
    else:
        verdict = (
            "TRAINING_TARGET_CENSORING_MASK_HEADROOM_FAIL"
            if mode == "censor_mask_oracle"
            else "TRAINING_TARGET_CENSORING_ORACLE_HEADROOM_FAIL"
            if mode == "censor_oracle"
            else "TRAINING_TARGET_RELIABILITY_WEIGHTING_HEADROOM_FAIL"
        )
        overall = "READABLE_BUT_NO_RECOVERY"

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": (
            "frozen_source_replay"
            if fresh_replay
            else "development_mechanism_only_program_headroom"
        ),
        "family": (
            "training-target stuck-value censoring"
            if mode in {
                "censor_oracle", "censor_mask_oracle", "censor_flatline_observed",
                "censor_flatline_fresh"
            }
            else "training-target reliability weighting"
        ),
        "program": {
            "mode": mode,
            "mechanism": (
                "target-cell masking over a known stuck-value interval"
                if mode == "censor_mask_oracle"
                else "exact restoration of a stuck-value censored training-target interval"
                if mode == "censor_oracle"
                else "Ridge sample_weight over training rows"
                if mode == "row"
                else "per-target-cell Ridge sample_weight over the affected interval"
            ),
            "oracle_rule": (
                "known-censored row x target[18:30) cells weight=0; all other cells=1"
                if mode == "censor_mask_oracle"
                else "replace only known-censored cells from the hidden clean copy"
                if mode == "censor_oracle"
                else "known-corrupted row weight=0; all other row weights=1"
                if mode == "row"
                else (
                    "known-corrupted row x target[18:30) cells weight=0; all other cells=1"
                    if mode == "target_cell"
                    else "observer-predicted interval cells weight=0; all other cells=1"
                )
            ),
            "deployment_witness_available": False,
            "observer_uses_injection_labels": False,
        },
        "configuration": {
            "datasets": list(specs),
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "anchors": list(ANCHORS),
            "seeds": list(SEEDS),
            "selected_rows_per_seed": SELECTED_ROW_COUNT,
            "target_block_half_open": list(TARGET_BLOCK),
            "corruption": (
                "selected target[18:30) replaced by the last visible standardized context value"
                if mode in {
                    "censor_oracle", "censor_mask_oracle", "censor_flatline_observed",
                    "censor_flatline_fresh"
                }
                else f"selected target[18:30) additive standardized offset {TARGET_OFFSET}"
            ),
            "consumer": "Ridge(alpha=1.0, fit_intercept=True, solver=svd)",
            "metric": "per-series sMASE; dataset-macro development decision",
            "expected_fit_count": EXPECTED_FITS,
            "observer": (
                {
                    "window_length": 12,
                    "candidate_starts_inclusive": [0, 36],
                    "absolute_value_range_tolerance": 1e-12,
                    "tie_break": "earlier start",
                    "input": "raw standardized target flatline topology",
                    "boundary_claim": "exact 12-cell flatline interval",
                }
                if mode in {"censor_flatline_observed", "censor_flatline_fresh"}
                else
                {
                    "window_length": 12,
                    "candidate_starts_inclusive": [6, 30],
                    "flank_length": 6,
                    "score_threshold": 1.0,
                    "tie_break": "earlier start",
                    "boundary_claim": "useful overlap only; exact match is diagnostic",
                    "input": (
                        "context-only nearest same-phase residual"
                        if mode == "phase_residual_interval"
                        else "raw-target and phase-residual interval agreement"
                        if mode == "raw_phase_agreement"
                        else "raw standardized target"
                    ),
                    "period_by_dataset": (
                        {dataset: int(spec["period"]) for dataset, spec in SPECS.items()}
                        if mode in {"phase_residual_interval", "raw_phase_agreement"}
                        else None
                    ),
                    "agreement_iou_threshold": (
                        0.5 if mode == "raw_phase_agreement" else None
                    ),
                }
                if mode in {
                    "observed_interval", "phase_residual_interval", "raw_phase_agreement"
                }
                else None
            ),
        },
        "source_reuse": (
            {
                "roster": "deterministic first 12 train + next 8 eval eligible series per dataset",
                "endpoint_rows": "seeded 14 of 72 training rows, frozen before outcomes",
                "context_exposure_before_run": "AGGREGATE_SEEN",
                "family_outcome_exposure_before_run": "SEALED",
                "family_outcome_exposure_after_run": "EXPOSED",
            }
            if fresh_replay
            else {
                "j0_plan": J0_PLAN_PATH,
                "j0_report": J0_REPORT_PATH,
                "roster": "same exposed J0 12 train + 8 eval per dataset",
                "endpoint_rows": "same already-recorded J0 14-row endpoint per seed",
                "context_exposure": "INSTANCE_SEEN",
                "outcome_exposure": "EXPOSED",
            }
        ),
        "dataset_evidence": dataset_evidence,
        "consumer_fit_count": fit_count,
        "consumer_fit_count_semantics": (
            "logical policy fits; one target-cell oracle policy contains twelve scalar subfits"
        ),
        "scalar_subfit_count": scalar_subfit_count,
        "expected_scalar_subfit_count": expected_scalar_subfits,
        "overall_classification": overall,
        "verdict": verdict,
        "capability_claim": False,
        "pattern_claim": False,
        "witness_claim": False,
        "memory_claim": False,
        "promotion_eligible": False,
        "formal_transfer": False,
        "information_wall": {
            "exposed_source_values_read": True,
            "support_b_values_read": False,
            "uci_values_read": False,
            "target_values_read": False,
            "query_values_read": False,
            "target_query_opened": False,
        },
        "target_query_opened": False,
        "claim_limit": (
            "Frozen two-dataset controlled Source replay only; dataset aggregates existed "
            "before this family. A pass is SOURCE_PROVISIONAL, not formal Promotion, "
            "Memory benefit, Target adaptation, or transfer evidence."
            if fresh_replay
            else "Development mechanism-only oracle headroom. Hidden corruption labels "
            "are unavailable at deployment; no Capability, Pattern, Witness, Memory, "
            "Promotion, Transfer, Target, or Query claim."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "row",
            "target_cell",
            "observed_interval",
            "phase_residual_interval",
            "raw_phase_agreement",
            "censor_oracle",
            "censor_mask_oracle",
            "censor_flatline_observed",
            "censor_flatline_fresh",
        ),
        default="row",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(root, mode=args.mode)
    default_path = (
        TARGET_CELL_REPORT_PATH
        if args.mode == "target_cell"
        else OBSERVED_INTERVAL_REPORT_PATH
        if args.mode == "observed_interval"
        else PHASE_RESIDUAL_REPORT_PATH
        if args.mode == "phase_residual_interval"
        else RAW_PHASE_AGREEMENT_REPORT_PATH
        if args.mode == "raw_phase_agreement"
        else CENSOR_ORACLE_REPORT_PATH
        if args.mode == "censor_oracle"
        else CENSOR_MASK_ORACLE_REPORT_PATH
        if args.mode == "censor_mask_oracle"
        else CENSOR_FLATLINE_REPORT_PATH
        if args.mode == "censor_flatline_observed"
        else FRESH_CENSOR_FLATLINE_REPORT_PATH
        if args.mode == "censor_flatline_fresh"
        else DEFAULT_REPORT_PATH
    )
    output = args.output or root / default_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
