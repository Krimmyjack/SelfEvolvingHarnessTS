"""Run the W60 development-only temporary-excursion Skill headroom check.

The one frozen hypothesis is that a visible two-boundary return geometry can
distinguish a temporary level excursion (rollback is eligible) from a natural
persistent tail regime (rollback must abstain).  ECG200 and GunPoint are
already-exposed UCR development backgrounds.  This runner does not promote a
Capability, build Memory, inspect the original UCI target, or tune the fixed
Ridge/accuracy judge.

``--smoke-only`` exercises the observer, scope, and explicit interval+offset
operator on synthetic arrays without reading any dataset or fitting a model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-temporary-excursion-skill-headroom/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/temporary_excursion_skill_headroom_report.json"
)
DATA_DIR = "data/ucr_task_context"
DATASETS = ("ECG200", "GunPoint")
RIDGE_ALPHA = 1.0

# Frozen before any W60 UCR outcome is read.  Row-index parity makes the
# corruption Consumer-readable without consulting labels, Dataset ID, or
# outcomes; a common shift could be absorbed by the Ridge intercept.
EXCURSION_START_FRACTION = 0.40
EXCURSION_END_FRACTION = 0.60
EXCURSION_OFFSET_SCALE = 12.0

# A broad geometry scan is used instead of the hidden injection interval.
SHOULDER_FRACTION = 0.04
MIN_INTERNAL_START_FRACTION = 0.15
MAX_INTERNAL_END_FRACTION = 0.85
MIN_EXCURSION_LENGTH_FRACTION = 0.10
MAX_EXCURSION_LENGTH_FRACTION = 0.35
TEMPORARY_EDGE_STRENGTH_MIN = 5.0
TEMPORARY_EDGE_AGREEMENT_MIN = 0.50
TEMPORARY_SHOULDER_DRIFT_MAX = 0.75

# Natural matched-risk candidates are visible TRAIN-only tail changes.  A
# fixed top quartile is forced only for the diagnostic harmful-action control;
# the actual Scope always abstains because there is no observed return.
TAIL_ONSET_MIN_FRACTION = 0.50
TAIL_ONSET_MAX_FRACTION = 0.80
TAIL_STRENGTH_MIN = 1.0
NATURAL_TOP_FRACTION = 0.25

MODIFICATION_TOLERANCE = 1e-10
P0_CORRUPTION_GAP_MIN = 0.01
P0_ORACLE_RECOVERY_MIN = 0.80
P0_ORACLE_CLEAN_TOLERANCE = 0.005
P1_TEMPORARY_ACT_RATE_MIN = 0.90
P1_OBSERVABLE_RECOVERY_MIN = 0.50
P1_NATURAL_ABSTAIN_RATE_MIN = 0.90


def _series_scale(np: Any, values: Any) -> float:
    row = np.asarray(values, dtype=np.float64)
    median = float(np.median(row))
    mad = 1.4826 * float(np.median(np.abs(row - median)))
    standard = float(np.std(row))
    return max(mad, 0.25 * standard, 1e-6)


def _geometry(length: int) -> tuple[int, int]:
    start = min(length - 1, max(0, int(math.floor(EXCURSION_START_FRACTION * length))))
    end = min(length, max(start + 1, int(math.ceil(EXCURSION_END_FRACTION * length))))
    if start < 3 or end > length - 3:
        raise ValueError("series is too short for the frozen internal excursion")
    return start, end


def _inject_temporary(
    np: Any,
    values: Any,
    original_fit_indices: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    matrix = np.asarray(values, dtype=np.float64)
    origins = np.asarray(original_fit_indices, dtype=np.int64)
    if matrix.ndim != 2 or origins.shape != (matrix.shape[0],):
        raise ValueError("temporary injection requires a 2D fit cohort and row origins")
    start, end = _geometry(matrix.shape[1])
    corrupt = matrix.copy()
    hidden: list[dict[str, Any]] = []
    for row_index, row in enumerate(matrix):
        scale = _series_scale(np, row)
        sign = 1.0 if int(origins[row_index]) % 2 == 0 else -1.0
        offset = sign * EXCURSION_OFFSET_SCALE * scale
        corrupt[row_index, start:end] += offset
        hidden.append(
            {
                "start": start,
                "end": end,
                "offset": offset,
                "scale": scale,
                "sign_source": "official_train_row_index_parity",
            }
        )
    return corrupt, hidden


def _best_internal_return(np: Any, values: Any, scale: float) -> dict[str, Any]:
    row = np.asarray(values, dtype=np.float64)
    length = int(row.size)
    shoulder = max(3, int(round(SHOULDER_FRACTION * length)))
    start_min = max(shoulder, int(math.ceil(MIN_INTERNAL_START_FRACTION * length)))
    end_max = min(length - shoulder, int(math.floor(MAX_INTERNAL_END_FRACTION * length)))
    span_min = max(2, int(math.ceil(MIN_EXCURSION_LENGTH_FRACTION * length)))
    span_max = max(span_min, int(math.floor(MAX_EXCURSION_LENGTH_FRACTION * length)))
    best: dict[str, Any] | None = None
    for start in range(start_min, end_max - span_min + 1):
        for end in range(start + span_min, min(end_max, start + span_max) + 1):
            left_out = float(np.median(row[start - shoulder : start]))
            left_in = float(np.median(row[start : start + shoulder]))
            right_in = float(np.median(row[end - shoulder : end]))
            right_out = float(np.median(row[end : end + shoulder]))
            left_jump = left_in - left_out
            right_return = right_in - right_out
            same_direction = left_jump * right_return > 0.0
            min_edge = min(abs(left_jump), abs(right_return))
            max_edge = max(abs(left_jump), abs(right_return), 1e-12)
            agreement = 1.0 - abs(left_jump - right_return) / max_edge
            offset = 0.5 * (left_jump + right_return)
            shoulder_drift_ratio = abs(left_out - right_out) / max(abs(offset), 1e-12)
            strength = min_edge / scale
            score = strength * max(0.0, agreement) / (1.0 + shoulder_drift_ratio)
            candidate = {
                "start": start,
                "end": end,
                "start_fraction": start / float(length),
                "end_fraction": end / float(length),
                "estimated_offset": offset,
                "left_edge_strength": abs(left_jump) / scale,
                "right_edge_strength": abs(right_return) / scale,
                "edge_direction_same": same_direction,
                "edge_offset_agreement": agreement,
                "left_right_shoulder_drift_ratio": shoulder_drift_ratio,
                "inside_shift_stability": agreement,
                "return_evidence": bool(same_direction),
                "score": score,
            }
            key = (score, strength, -start, -end)
            if best is None or key > best["_key"]:
                candidate["_key"] = key
                best = candidate
    if best is None:
        raise ValueError("no legal internal-return candidate")
    best.pop("_key")
    return best


def _best_persistent_tail(np: Any, values: Any, scale: float) -> dict[str, Any]:
    row = np.asarray(values, dtype=np.float64)
    length = int(row.size)
    shoulder = max(3, int(round(SHOULDER_FRACTION * length)))
    onset_min = max(shoulder, int(math.ceil(TAIL_ONSET_MIN_FRACTION * length)))
    onset_max = min(length - shoulder, int(math.floor(TAIL_ONSET_MAX_FRACTION * length)))
    best: dict[str, Any] | None = None
    for onset in range(onset_min, onset_max + 1):
        tail_length = length - onset
        before_start = max(0, onset - tail_length)
        before = float(np.median(row[before_start:onset]))
        after = float(np.median(row[onset:]))
        local_before = float(np.median(row[onset - shoulder : onset]))
        local_after = float(np.median(row[onset : onset + shoulder]))
        global_offset = after - before
        local_jump = local_after - local_before
        direction_agreement = global_offset * local_jump > 0.0
        strength = min(abs(global_offset), abs(local_jump)) / scale
        score = strength if direction_agreement else 0.0
        candidate = {
            "onset": onset,
            "onset_fraction": onset / float(length),
            "end_fraction": 1.0,
            "estimated_offset": global_offset,
            "local_jump": local_jump,
            "strength": strength,
            "direction_agreement": direction_agreement,
            "right_return_available": False,
            "score": score,
        }
        key = (score, strength, -onset)
        if best is None or key > best["_key"]:
            candidate["_key"] = key
            best = candidate
    if best is None:
        raise ValueError("no legal persistent-tail candidate")
    best.pop("_key")
    return best


def _observe_and_scope(np: Any, values: Any) -> dict[str, Any]:
    """Build one label/outcome-free observation and compile ACT or ABSTAIN."""

    row = np.asarray(values, dtype=np.float64)
    if row.ndim != 1 or row.size < 24 or not np.isfinite(row).all():
        raise ValueError("excursion observation requires one finite complete series")
    scale = _series_scale(np, row)
    temporary = _best_internal_return(np, row, scale)
    tail = _best_persistent_tail(np, row, scale)
    temporary_eligible = bool(
        temporary["edge_direction_same"]
        and float(temporary["left_edge_strength"]) >= TEMPORARY_EDGE_STRENGTH_MIN
        and float(temporary["right_edge_strength"]) >= TEMPORARY_EDGE_STRENGTH_MIN
        and float(temporary["edge_offset_agreement"]) >= TEMPORARY_EDGE_AGREEMENT_MIN
        and float(temporary["left_right_shoulder_drift_ratio"])
        <= TEMPORARY_SHOULDER_DRIFT_MAX
    )
    tail_eligible = bool(
        tail["direction_agreement"]
        and float(tail["strength"]) >= TAIL_STRENGTH_MIN
    )
    if temporary_eligible:
        decision = "ACT_ROLLBACK_TEMPORARY_EXCURSION"
        reasons = ["two_internal_edges_same_offset_direction", "visible_right_side_return"]
    elif tail_eligible:
        decision = "ABSTAIN_PERSISTENT_TAIL_NO_RETURN"
        reasons = ["one_visible_onset", "no_observed_right_side_return"]
    else:
        decision = "ABSTAIN_UNRESOLVED"
        reasons = ["insufficient_program_specific_geometry"]
    return {
        "observer": "label-free-two-edge-return-geometry-v1",
        "inputs_used": ["one_observed_train_series", "time_index"],
        "label_used": False,
        "dataset_id_used": False,
        "test_or_consumer_outcome_used": False,
        "injection_metadata_used": False,
        "scale": scale,
        "temporary_candidate": temporary,
        "persistent_tail_candidate": tail,
        "decision": decision,
        "reason_codes": reasons,
    }


def _explicit_repair(
    np: Any,
    values: Any,
    *,
    start_fraction: float,
    end_fraction: float,
    offset: float,
) -> Any:
    from SelfEvolvingHarnessTS.operators.s1_structural import repair_level_shift

    repaired = np.asarray(
        repair_level_shift(
            np.asarray(values, dtype=np.float64),
            region_start_fraction=float(start_fraction),
            region_end_fraction=float(end_fraction),
            estimated_offset=float(offset),
        ),
        dtype=np.float64,
    )
    if repaired.shape != np.asarray(values).shape or not np.isfinite(repaired).all():
        raise RuntimeError("explicit interval+offset level repair returned invalid output")
    return repaired


def _oracle_repair_cohort(np: Any, corrupt: Any, hidden: list[dict[str, Any]]) -> Any:
    output = np.asarray(corrupt, dtype=np.float64).copy()
    length = output.shape[1]
    for index, specification in enumerate(hidden):
        output[index] = _explicit_repair(
            np,
            output[index],
            start_fraction=int(specification["start"]) / float(length),
            end_fraction=int(specification["end"]) / float(length),
            offset=float(specification["offset"]),
        )
    return output


def _observable_repair_cohort(
    np: Any, corrupt: Any, observations: list[dict[str, Any]]
) -> Any:
    output = np.asarray(corrupt, dtype=np.float64).copy()
    for index, observation in enumerate(observations):
        if observation["decision"] != "ACT_ROLLBACK_TEMPORARY_EXCURSION":
            continue
        candidate = observation["temporary_candidate"]
        output[index] = _explicit_repair(
            np,
            output[index],
            start_fraction=float(candidate["start_fraction"]),
            end_fraction=float(candidate["end_fraction"]),
            offset=float(candidate["estimated_offset"]),
        )
    return output


def _interval_iou(start: int, end: int, expected_start: int, expected_end: int) -> float:
    intersection = max(0, min(end, expected_end) - max(start, expected_start))
    union = max(end, expected_end) - min(start, expected_start)
    return intersection / float(union) if union else 1.0


def _select_natural_candidates(
    np: Any, observations: list[dict[str, Any]]
) -> list[int]:
    """Select persistent-tail risks from geometry, independently of Scope output."""

    eligible = [
        index
        for index, observation in enumerate(observations)
        if bool(observation["persistent_tail_candidate"]["direction_agreement"])
        and float(observation["persistent_tail_candidate"]["strength"])
        >= TAIL_STRENGTH_MIN
    ]
    if not eligible:
        return []
    count = max(1, int(math.ceil(NATURAL_TOP_FRACTION * len(observations))))
    ranked = sorted(
        eligible,
        key=lambda index: (
            float(observations[index]["persistent_tail_candidate"]["score"]),
            -index,
        ),
        reverse=True,
    )
    return ranked[:count]


def _force_persistent_rollback(
    np: Any,
    clean: Any,
    observations: list[dict[str, Any]],
    selected: list[int],
) -> Any:
    output = np.asarray(clean, dtype=np.float64).copy()
    for index in selected:
        tail = observations[index]["persistent_tail_candidate"]
        output[index] = _explicit_repair(
            np,
            output[index],
            start_fraction=float(tail["onset_fraction"]),
            end_fraction=1.0,
            offset=float(tail["estimated_offset"]),
        )
    return output


def _fit_accuracy(
    np: Any,
    RidgeClassifier: Any,
    features: Any,
    train_values: Any,
    train_labels: Any,
    query_values: Any,
    query_labels: Any,
) -> float:
    model = RidgeClassifier(alpha=RIDGE_ALPHA)
    model.fit(features(np, train_values), train_labels)
    return float(np.mean(model.predict(features(np, query_values)) == query_labels))


def _preflight(root: Path) -> tuple[Any, dict[str, dict[str, Any]]]:
    import numpy as np

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _load_split,
        _split_fit_support,
    )

    prepared: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        archive = root / DATA_DIR / f"{dataset}.zip"
        train_values, train_labels = _load_split(np, archive, dataset, "TRAIN")
        fit_indices, support_indices = _split_fit_support(np, train_labels)
        clean_fit = train_values[fit_indices]
        fit_labels = train_labels[fit_indices]
        corrupt_fit, hidden = _inject_temporary(np, clean_fit, fit_indices)
        temporary_observations = [
            _observe_and_scope(np, row) for row in corrupt_fit
        ]
        observable_fit = _observable_repair_cohort(
            np, corrupt_fit, temporary_observations
        )
        oracle_fit = _oracle_repair_cohort(np, corrupt_fit, hidden)

        natural_observations = [
            _observe_and_scope(np, row) for row in clean_fit
        ]
        natural_selected = _select_natural_candidates(np, natural_observations)
        forced_fit = _force_persistent_rollback(
            np, clean_fit, natural_observations, natural_selected
        )
        prepared[dataset] = {
            "archive": archive,
            "train_count": int(train_values.shape[0]),
            "series_length": int(train_values.shape[1]),
            "clean_fit": clean_fit,
            "fit_labels": fit_labels,
            "support_values": train_values[support_indices],
            "support_labels": train_labels[support_indices],
            "corrupt_fit": corrupt_fit,
            "hidden": hidden,
            "temporary_observations": temporary_observations,
            "observable_fit": observable_fit,
            "oracle_fit": oracle_fit,
            "natural_observations": natural_observations,
            "natural_selected": natural_selected,
            "forced_fit": forced_fit,
        }
    return np, prepared


def evaluate(root: Path) -> dict[str, Any]:
    from sklearn.linear_model import RidgeClassifier

    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_context_label_evidence_witness import (
        _features,
        _load_split,
    )

    np, prepared = _preflight(root)
    missing = [
        dataset
        for dataset, state in prepared.items()
        if not state["natural_selected"]
    ]
    if missing:
        return {
            "schema_version": SCHEMA_VERSION,
            "scientific_role": "W60 development-only temporary-excursion Skill headroom",
            "verdict": "DEVELOPMENT_TEMPORARY_EXCURSION_FEASIBILITY_FAIL",
            "failed_datasets": missing,
            "reason": "no frozen label-free persistent-tail candidate before any Consumer fit",
            "consumer_fit_count": 0,
            "test_split_load_count": 0,
            "fresh_promotion_evidence": False,
            "capability_promoted": False,
            "persistent_memory_built": False,
            "original_uci_target_query_opened": False,
        }

    rows: list[dict[str, Any]] = []
    consumer_fit_count = 0
    test_split_load_count = 0
    for dataset in DATASETS:
        state = prepared[dataset]
        query_values, query_labels = _load_split(
            np, state["archive"], dataset, "TEST"
        )
        test_split_load_count += 1
        accuracies = {
            "clean_incumbent": _fit_accuracy(
                np, RidgeClassifier, _features, state["clean_fit"], state["fit_labels"],
                query_values, query_labels,
            ),
            "temporary_corrupt": _fit_accuracy(
                np, RidgeClassifier, _features, state["corrupt_fit"], state["fit_labels"],
                query_values, query_labels,
            ),
            "oracle_bound_rollback": _fit_accuracy(
                np, RidgeClassifier, _features, state["oracle_fit"], state["fit_labels"],
                query_values, query_labels,
            ),
            "observable_bound_rollback": _fit_accuracy(
                np, RidgeClassifier, _features, state["observable_fit"], state["fit_labels"],
                query_values, query_labels,
            ),
            "natural_forced_blind_rollback": _fit_accuracy(
                np, RidgeClassifier, _features, state["forced_fit"], state["fit_labels"],
                query_values, query_labels,
            ),
        }
        consumer_fit_count += 5

        clean = state["clean_fit"]
        corrupt = state["corrupt_fit"]
        oracle = state["oracle_fit"]
        hidden = state["hidden"]
        observations = state["temporary_observations"]
        start, end = _geometry(state["series_length"])
        allowed = np.zeros_like(clean, dtype=bool)
        allowed[:, start:end] = True
        oracle_change = np.abs(oracle - corrupt) > MODIFICATION_TOLERANCE
        oracle_exact_max_error = float(np.max(np.abs(oracle - clean)))
        oracle_only_interval = not bool(np.any(oracle_change & ~allowed))
        observable_act_rate = sum(
            observation["decision"] == "ACT_ROLLBACK_TEMPORARY_EXCURSION"
            for observation in observations
        ) / float(len(observations))
        interval_ious = [
            _interval_iou(
                int(observation["temporary_candidate"]["start"]),
                int(observation["temporary_candidate"]["end"]),
                start,
                end,
            )
            for observation in observations
        ]
        relative_offset_errors = [
            abs(
                float(observation["temporary_candidate"]["estimated_offset"])
                - float(specification["offset"])
            )
            / max(abs(float(specification["offset"])), 1e-12)
            for observation, specification in zip(observations, hidden)
        ]
        natural_selected = state["natural_selected"]
        natural_abstain_rate = sum(
            state["natural_observations"][index]["decision"].startswith("ABSTAIN")
            for index in natural_selected
        ) / float(len(natural_selected))

        clean_accuracy = accuracies["clean_incumbent"]
        corrupt_accuracy = accuracies["temporary_corrupt"]
        oracle_accuracy = accuracies["oracle_bound_rollback"]
        observable_accuracy = accuracies["observable_bound_rollback"]
        corruption_gap = clean_accuracy - corrupt_accuracy
        oracle_recovery = (
            (oracle_accuracy - corrupt_accuracy) / corruption_gap
            if corruption_gap > 0.0
            else 0.0
        )
        observable_recovery = (
            (observable_accuracy - corrupt_accuracy) / corruption_gap
            if corruption_gap > 0.0
            else 0.0
        )
        forced_gain = accuracies["natural_forced_blind_rollback"] - clean_accuracy
        p0_pass = bool(
            oracle_exact_max_error <= MODIFICATION_TOLERANCE
            and oracle_only_interval
            and corruption_gap >= P0_CORRUPTION_GAP_MIN
            and oracle_recovery >= P0_ORACLE_RECOVERY_MIN
            and oracle_accuracy >= clean_accuracy - P0_ORACLE_CLEAN_TOLERANCE
            and oracle_accuracy >= corrupt_accuracy
        )
        p1_temporary_pass = bool(
            observable_act_rate >= P1_TEMPORARY_ACT_RATE_MIN
            and observable_accuracy >= corrupt_accuracy
            and observable_recovery >= P1_OBSERVABLE_RECOVERY_MIN
        )
        p1_natural_scope_pass = bool(
            natural_selected and natural_abstain_rate >= P1_NATURAL_ABSTAIN_RATE_MIN
        )
        rows.append(
            {
                "dataset": dataset,
                "official_train_count": state["train_count"],
                "fit_count": int(clean.shape[0]),
                "support_count": int(state["support_values"].shape[0]),
                "official_test_count": int(query_values.shape[0]),
                "series_length": state["series_length"],
                "temporary_control": {
                    "injection_interval": [start, end],
                    "injection_interval_fraction": [
                        EXCURSION_START_FRACTION,
                        EXCURSION_END_FRACTION,
                    ],
                    "offset_scale_multiplier": EXCURSION_OFFSET_SCALE,
                    "sign_rule": "official TRAIN row-index parity; label/dataset/outcome blind",
                    "oracle_exact_artifact_recovery_max_abs_error": oracle_exact_max_error,
                    "oracle_changed_only_in_injected_interval": oracle_only_interval,
                    "observer_act_rate": observable_act_rate,
                    "private_interval_iou_median": float(np.median(interval_ious)),
                    "private_relative_offset_error_median": float(
                        np.median(relative_offset_errors)
                    ),
                    "private_diagnostics_feed_decision": False,
                },
                "natural_matched_risk": {
                    "candidate_rule": (
                        "top quartile of fit rows among visible persistent-tail geometry "
                        "candidates; selection is independent of the Scope decision"
                    ),
                    "candidate_count": len(natural_selected),
                    "candidate_fit_fraction": len(natural_selected) / float(clean.shape[0]),
                    "scope_abstain_rate": natural_abstain_rate,
                    "right_return_available": False,
                    "forced_action": "subtract observed tail offset despite frozen Scope ABSTAIN",
                    "forced_minus_clean_accuracy": forced_gain,
                    "selection_used_label_dataset_or_outcome": False,
                },
                "query_accuracy": accuracies,
                "corruption_gap": corruption_gap,
                "oracle_recovery_fraction": oracle_recovery,
                "observable_recovery_fraction": observable_recovery,
                "p0_recoverability_and_readability_pass": p0_pass,
                "p1_temporary_observer_scope_pass": p1_temporary_pass,
                "p1_natural_scope_pass": p1_natural_scope_pass,
                "consumer_fit_count": 5,
            }
        )

    forced_gains = [
        float(row["natural_matched_risk"]["forced_minus_clean_accuracy"])
        for row in rows
    ]
    macro_forced_gain = sum(forced_gains) / len(forced_gains)
    p0_all = all(bool(row["p0_recoverability_and_readability_pass"]) for row in rows)
    p1_temporary_all = all(
        bool(row["p1_temporary_observer_scope_pass"]) for row in rows
    )
    p1_natural_all = all(bool(row["p1_natural_scope_pass"]) for row in rows)
    forced_harm = any(gain < 0.0 for gain in forced_gains) and macro_forced_gain < 0.0
    gate_pass = bool(p0_all and p1_temporary_all and p1_natural_all and forced_harm)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "W60 development-only temporary-excursion Skill headroom",
        "causal_hypothesis": (
            "Visible two-boundary return geometry makes explicit interval+offset rollback "
            "applicable to temporary excursions, while a natural persistent tail without "
            "a visible return is a matched contraindication that must abstain."
        ),
        "frozen_surfaces": {
            "task": "binary UCR classification",
            "consumer": "ridge-raw-plus-difference-v1",
            "metric": "classification accuracy",
            "program": "operators.s1_structural.repair_level_shift explicit interval+offset",
            "only_tested_change": "Program-specific local excursion Observation plus Scope",
            "proxy": "none",
        },
        "information_boundary": {
            "temporary_observer_inputs": ["one observed fit series", "time index"],
            "natural_candidate_inputs": ["original fit series", "time index"],
            "observer_forbidden_inputs": [
                "label",
                "dataset name",
                "TEST",
                "Consumer outcome",
                "injection interval",
                "injection offset",
            ],
            "private_interval_and_offset_grading_used_for_decision": False,
            "test_loaded_after_all_train_observation_and_scope_decisions": True,
        },
        "datasets": rows,
        "overall": {
            "dataset_count": len(rows),
            "p0_all_datasets": p0_all,
            "p1_temporary_all_datasets": p1_temporary_all,
            "p1_natural_scope_all_datasets": p1_natural_all,
            "forced_rollback_harmful_dataset_count": sum(
                gain < 0.0 for gain in forced_gains
            ),
            "macro_forced_minus_clean_accuracy": macro_forced_gain,
            "consumer_fit_count": consumer_fit_count,
            "test_split_load_count": test_split_load_count,
            "frozen_gate_pass": gate_pass,
        },
        "verdict": (
            "DEVELOPMENT_TEMPORARY_EXCURSION_SKILL_HEADROOM_PASS"
            if gate_pass
            else "DEVELOPMENT_TEMPORARY_EXCURSION_SKILL_HEADROOM_FAIL_FAMILY_CLOSED"
        ),
        "fresh_promotion_evidence": False,
        "capability_promoted": False,
        "persistent_memory_built": False,
        "original_uci_target_query_opened": False,
        "claim_limit": (
            "ECG200 and GunPoint are exposed development backgrounds and the temporary "
            "excursion is controlled.  A pass supports only headroom and visible matched-risk "
            "discrimination for this Skill; it is not fresh promotion or natural transfer."
        ),
    }


def synthetic_smoke() -> dict[str, Any]:
    import numpy as np

    length = 120
    time = np.linspace(0.0, 4.0 * np.pi, length)
    clean = 0.20 * np.sin(time) + 0.002 * np.arange(length)
    corrupt, hidden = _inject_temporary(
        np, clean[None, :], np.asarray([2], dtype=np.int64)
    )
    observation = _observe_and_scope(np, corrupt[0])
    oracle = _oracle_repair_cohort(np, corrupt, hidden)
    observable = _observable_repair_cohort(np, corrupt, [observation])
    persistent = clean.copy()
    persistent[int(0.65 * length) :] += 4.0
    persistent_observation = _observe_and_scope(np, persistent)
    passed = bool(
        observation["decision"] == "ACT_ROLLBACK_TEMPORARY_EXCURSION"
        and persistent_observation["decision"] == "ABSTAIN_PERSISTENT_TAIL_NO_RETURN"
        and float(np.max(np.abs(oracle[0] - clean))) <= MODIFICATION_TOLERANCE
        and float(np.mean(np.abs(observable[0] - clean)))
        < float(np.mean(np.abs(corrupt[0] - clean)))
    )
    return {
        "passed": passed,
        "temporary_decision": observation["decision"],
        "persistent_decision": persistent_observation["decision"],
        "oracle_max_abs_error": float(np.max(np.abs(oracle[0] - clean))),
        "observable_mae": float(np.mean(np.abs(observable[0] - clean))),
        "corrupt_mae": float(np.mean(np.abs(corrupt[0] - clean))),
        "dataset_read": False,
        "consumer_fit_count": 0,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only:
        payload = synthetic_smoke()
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["passed"] else 1

    output = args.output or root / DEFAULT_REPORT_PATH
    payload = evaluate(root)
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
