"""Run the minimum zero-Agent E1-P periodic-missing premise pilot.

The pilot asks three questions before any Memory or adaptation machinery is built:

1. does the fixed program menu have forecasting headroom over identity;
2. does the best action vary between stable seasonal cases and matched risks; and
3. do gap-local observations predict realized seasonal-copy contraindications better
   than the current global public feature context.

All deployment-side features and policy choices are functions of ``corrupt_context``
only.  Clean context, clean future, archetype names, and risk labels remain private
grader inputs.  The in-sample menu oracle is reported as a ceiling, never as a
deployable selector.
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline
from SelfEvolvingHarnessTS.runtime.public_features import extract_public_features


CONTEXT_LENGTH = 192
FUTURE_LENGTH = 48
TOTAL_LENGTH = CONTEXT_LENGTH + FUTURE_LENGTH
SEEDS = (1103, 2207, 3301, 4409, 5519)
PROGRAM_IDS = ("identity", "linear", "seasonal")

POSITIVE_ARCHETYPES = (
    "stationary_seasonal",
    "slow_trend_seasonal",
    "harmonic_stable_seasonal",
)
RISK_ARCHETYPES = (
    "amplitude_drift",
    "phase_drift",
    "period_change",
    "gap_crossing_regime_boundary",
    "insufficient_cycles",
)

PILOT_THRESHOLDS = {
    "positive_archetype_oracle_gain_min": 0.01,
    "positive_archetypes_required": 2,
    "risk_winner_margin_min": 0.005,
    "local_risk_auc_min": 0.75,
    "local_over_global_auc_min": 0.10,
}

GLOBAL_FEATURE_NAMES = (
    "missing_fraction",
    "longest_missing_run_fraction",
    "local_robust_z_peak",
    "estimated_region_start_fraction",
    "estimated_region_end_fraction",
    "level_excursion_score",
    "estimated_level_offset",
    "period_change_score",
    "period_reliability",
    "pre_period",
    "post_period",
    "acf_spectral_consistency",
)
LOCAL_EXTRA_FEATURE_NAMES = (
    "observed_cycles",
    "phase_correlation",
    "amplitude_ratio",
    "amplitude_stability",
    "local_period_consistency",
    "boundary_proximity",
)


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    seed: int
    archetype: str
    cohort: str
    generator_period: int
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray


def _period_for_seed(seed: int) -> int:
    return (24, 26, 28)[SEEDS.index(seed) % 3]


def _continuous_angle(periods: np.ndarray, phase: float) -> np.ndarray:
    increments = 2.0 * np.pi / np.asarray(periods, dtype=np.float64)
    return phase + np.cumsum(np.concatenate(([0.0], increments[:-1])))


def _clean_series(seed: int, archetype: str) -> tuple[np.ndarray, int]:
    """Create one private clean trajectory; no missingness is introduced here."""

    period = _period_for_seed(seed)
    rng = np.random.default_rng(seed)
    t = np.arange(TOTAL_LENGTH, dtype=np.float64)
    phase = 0.15 + 0.07 * (SEEDS.index(seed) % 5)
    base_angle = 2.0 * np.pi * t / period + phase
    noise = rng.normal(0.0, 0.025, size=TOTAL_LENGTH)
    trend = 0.0015 * t

    if archetype == "stationary_seasonal":
        values = trend + np.sin(base_angle) + 0.18 * np.sin(2.0 * base_angle + 0.4)
    elif archetype == "slow_trend_seasonal":
        values = 0.0045 * t + np.sin(base_angle) + 0.15 * np.sin(2.0 * base_angle)
    elif archetype == "harmonic_stable_seasonal":
        values = (
            trend
            + 0.75 * np.sin(base_angle)
            + 0.48 * np.sin(2.0 * base_angle + 0.5)
            + 0.20 * np.sin(3.0 * base_angle - 0.2)
        )
    elif archetype == "amplitude_drift":
        amplitude = 0.65 + 1.25 * np.clip((t - 108.0) / 84.0, 0.0, 1.0)
        values = trend + amplitude * np.sin(base_angle) + 0.15 * np.sin(2.0 * base_angle)
    elif archetype == "phase_drift":
        phase_drift = 1.5 * np.pi * np.clip((t - 108.0) / 84.0, 0.0, 1.0)
        angle = base_angle + phase_drift
        values = trend + np.sin(angle) + 0.16 * np.sin(2.0 * angle + 0.4)
    elif archetype == "period_change":
        periods = np.full(TOTAL_LENGTH, float(period), dtype=np.float64)
        periods[144:] = max(8.0, 0.75 * period)
        angle = _continuous_angle(periods, phase)
        values = trend + np.sin(angle) + 0.16 * np.sin(2.0 * angle + 0.4)
    elif archetype == "gap_crossing_regime_boundary":
        values = trend + np.sin(base_angle) + 0.18 * np.sin(2.0 * base_angle + 0.4)
        values = values + np.where(t >= 168.0, 1.35, 0.0)
    elif archetype == "insufficient_cycles":
        values = trend + np.sin(base_angle) + 0.18 * np.sin(2.0 * base_angle + 0.4)
    else:
        raise ValueError(f"unknown E1-P archetype: {archetype!r}")
    return np.asarray(values + noise, dtype=np.float64), period


def build_pilot_cases() -> tuple[PilotCase, ...]:
    """Build five matched seeds across the three positives and five risks."""

    cases: list[PilotCase] = []
    for seed in SEEDS:
        for cohort, archetypes in (
            ("positive", POSITIVE_ARCHETYPES),
            ("risk", RISK_ARCHETYPES),
        ):
            for archetype in archetypes:
                clean, period = _clean_series(seed, archetype)
                clean_context = clean[:CONTEXT_LENGTH].copy()
                corrupt_context = clean_context.copy()
                # All cases share the same forecast-relevant primary gap.  The regime
                # boundary is deliberately hidden inside it for the matched boundary risk.
                corrupt_context[156:180] = np.nan
                if archetype == "insufficient_cycles":
                    # Preserve the exact 192-point consumer input while exposing fewer
                    # than two complete observed cycles before the primary gap.
                    corrupt_context[:132] = np.nan
                cases.append(
                    PilotCase(
                        case_id=f"e1p-{seed}-{archetype}",
                        seed=seed,
                        archetype=archetype,
                        cohort=cohort,
                        generator_period=period,
                        clean_context=clean_context,
                        corrupt_context=corrupt_context,
                        clean_future=clean[CONTEXT_LENGTH:].copy(),
                    )
                )
    return tuple(cases)


def _missing_runs(values: np.ndarray) -> tuple[tuple[int, int], ...]:
    missing = np.flatnonzero(~np.isfinite(values))
    if not missing.size:
        return ()
    groups = np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1)
    return tuple((int(group[0]), int(group[-1] + 1)) for group in groups if group.size)


def _primary_gap(values: np.ndarray) -> tuple[int, int]:
    runs = _missing_runs(values)
    bounded = [
        (start, end)
        for start, end in runs
        if np.any(np.isfinite(values[:start])) and np.any(np.isfinite(values[end:]))
    ]
    candidates = bounded or list(runs)
    if not candidates:
        raise ValueError("E1-P case has no observable missing run")
    return max(candidates, key=lambda item: (item[1] - item[0], item[0]))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 4:
        return 0.0
    x = np.asarray(left[mask], dtype=np.float64)
    y = np.asarray(right[mask], dtype=np.float64)
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.clip((x @ y) / denominator, -1.0, 1.0)) if denominator > 1e-12 else 0.0


def _lag_correlation(values: np.ndarray, lag: int) -> float:
    if lag < 1 or lag >= values.size:
        return 0.0
    return _correlation(values[:-lag], values[lag:])


def _contiguous_finite_left(values: np.ndarray, stop: int) -> int:
    count = 0
    for index in range(stop - 1, -1, -1):
        if not np.isfinite(values[index]):
            break
        count += 1
    return count


def _contiguous_finite_right(values: np.ndarray, start: int) -> int:
    count = 0
    for index in range(start, values.size):
        if not np.isfinite(values[index]):
            break
        count += 1
    return count


def _global_features(values: np.ndarray) -> tuple[dict[str, float], int]:
    public = extract_public_features(values, task_kind="forecast")
    mapping = public.mapping
    features = {
        "missing_fraction": float(mapping["missing_fraction"]),
        "longest_missing_run_fraction": float(mapping["longest_missing_run_fraction"]),
        "local_robust_z_peak": float(mapping["local_robust_z_peak"]),
        "estimated_region_start_fraction": float(mapping["estimated_region_start_fraction"]),
        "estimated_region_end_fraction": float(mapping["estimated_region_end_fraction"]),
        "level_excursion_score": float(mapping["level_excursion_score"]),
        "estimated_level_offset": float(mapping["estimated_level_offset"]),
        "period_change_score": float(mapping["period_change_score"]),
        "period_reliability": float(mapping["period_reliability"]),
        "pre_period": float(public.pre_period),
        "post_period": float(public.post_period),
        "acf_spectral_consistency": float(public.acf_spectral_consistency),
    }
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("global public feature extraction produced a non-finite value")
    observed_period = max(4, min(48, int(public.pre_period)))
    return features, observed_period


def _local_features(values: np.ndarray, observed_period: int) -> dict[str, float]:
    """Extract gap-local, deployment-visible evidence from corrupt values only."""

    start, end = _primary_gap(values)
    period = max(4, int(observed_period))
    left_run = _contiguous_finite_left(values, start)
    right_run = _contiguous_finite_right(values, end)
    observed_cycles = float(left_run // period + right_run // period)

    local_start = max(0, start - 3 * period)
    local_end = min(values.size, end + 2 * period)
    local = np.asarray(values[local_start:local_end], dtype=np.float64)
    phase_correlation = _lag_correlation(local, period)

    recent = values[max(0, start - period) : start]
    previous = values[max(0, start - 2 * period) : max(0, start - period)]
    recent_finite = recent[np.isfinite(recent)]
    previous_finite = previous[np.isfinite(previous)]
    recent_scale = float(np.std(recent_finite)) if recent_finite.size >= 4 else 0.0
    previous_scale = float(np.std(previous_finite)) if previous_finite.size >= 4 else 0.0
    scale_floor = 1e-8
    amplitude_ratio = max(recent_scale, previous_scale) / max(
        min(recent_scale, previous_scale), scale_floor
    )
    amplitude_ratio = float(min(amplitude_ratio, 20.0))
    amplitude_stability = float(math.exp(-abs(math.log(max(amplitude_ratio, scale_floor)))))

    lag_radius = max(2, int(round(0.30 * period)))
    candidates = range(max(4, period - lag_radius), min(48, period + lag_radius) + 1)
    lag_scores = [(lag, _lag_correlation(local, lag)) for lag in candidates]
    best_lag, best_score = max(lag_scores, key=lambda item: (item[1], -abs(item[0] - period)))
    lag_agreement = max(0.0, 1.0 - abs(best_lag - period) / max(period, 1))
    local_period_consistency = float(lag_agreement * max(0.0, best_score))

    boundary_width = max(6, period // 2)
    left_boundary = values[max(0, start - boundary_width) : start]
    right_boundary = values[end : min(values.size, end + boundary_width)]
    left_boundary = left_boundary[np.isfinite(left_boundary)]
    right_boundary = right_boundary[np.isfinite(right_boundary)]
    finite = values[np.isfinite(values)]
    robust_center = float(np.median(finite))
    robust_scale = max(1.4826 * float(np.median(np.abs(finite - robust_center))), 1e-8)
    if left_boundary.size and right_boundary.size:
        boundary_proximity = abs(
            float(np.median(right_boundary)) - float(np.median(left_boundary))
        ) / robust_scale
    else:
        boundary_proximity = 0.0

    features = {
        "observed_cycles": observed_cycles,
        "phase_correlation": phase_correlation,
        "amplitude_ratio": amplitude_ratio,
        "amplitude_stability": amplitude_stability,
        "local_period_consistency": local_period_consistency,
        "boundary_proximity": float(min(boundary_proximity, 20.0)),
    }
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("local feature extraction produced a non-finite value")
    return features


def _execute_program(
    program_id: str,
    corrupt_context: np.ndarray,
    *,
    observed_period: int,
) -> np.ndarray:
    if program_id == "identity":
        return np.asarray(corrupt_context, dtype=np.float64).copy()
    if program_id == "linear":
        steps = [("impute_linear", {})]
    elif program_id == "seasonal":
        steps = [("period_complete", {"period": int(observed_period)})]
    else:
        raise ValueError(f"unknown E1-P program: {program_id!r}")
    execution = run_pipeline(steps, corrupt_context, source="e1p_fixed_program")
    if not execution.ok or execution.artifact is None:
        raise RuntimeError(f"E1-P program {program_id!r} failed: {execution.error}")
    artifact = np.asarray(execution.artifact, dtype=np.float64)
    if artifact.shape != corrupt_context.shape or not np.isfinite(artifact).all():
        raise RuntimeError(f"E1-P program {program_id!r} produced an invalid artifact")
    return artifact


def _feature_vector(row: Mapping[str, object], *, local: bool) -> np.ndarray:
    global_features = row["global_features"]
    local_features = row["local_features"]
    if not isinstance(global_features, Mapping) or not isinstance(local_features, Mapping):
        raise TypeError("case row is missing feature mappings")
    names = GLOBAL_FEATURE_NAMES + (LOCAL_EXTRA_FEATURE_NAMES if local else ())
    sources = {**global_features, **local_features}
    return np.asarray([float(sources[name]) for name in names], dtype=np.float64)


def _standardization(rows: Sequence[Mapping[str, object]], *, local: bool) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.stack([_feature_vector(row, local=local) for row in rows])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def _distance(vector: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(vector - centroid))))


def _loso_risk_scores(rows: Sequence[Mapping[str, object]], *, local: bool) -> dict[str, float]:
    """Predict realized Seasonal contraindication from other-seed outcomes only."""

    scores: dict[str, float] = {}
    for seed in SEEDS:
        train = [row for row in rows if int(row["seed"]) != seed]
        test = [row for row in rows if int(row["seed"]) == seed]
        mean, scale = _standardization(train, local=local)
        train_vectors = [(_feature_vector(row, local=local) - mean) / scale for row in train]
        safe = np.stack(
            [
                vector
                for row, vector in zip(train, train_vectors)
                if int(row["seasonal_contraindication_label"]) == 0
            ]
        ).mean(axis=0)
        contraindicated = np.stack(
            [
                vector
                for row, vector in zip(train, train_vectors)
                if int(row["seasonal_contraindication_label"]) == 1
            ]
        ).mean(axis=0)
        for row in test:
            vector = (_feature_vector(row, local=local) - mean) / scale
            # Higher means closer to the realized-contraindication centroid. The held-out
            # row's own outcome label is not used while producing this score.
            scores[str(row["case_id"])] = _distance(vector, safe) - _distance(
                vector, contraindicated
            )
    return scores


def _loso_action_choices(
    rows: Sequence[Mapping[str, object]], *, local: bool
) -> dict[str, str]:
    choices: dict[str, str] = {}
    for seed in SEEDS:
        train = [row for row in rows if int(row["seed"]) != seed]
        test = [row for row in rows if int(row["seed"]) == seed]
        mean, scale = _standardization(train, local=local)
        by_action: dict[str, list[np.ndarray]] = {program: [] for program in PROGRAM_IDS}
        for row in train:
            vector = (_feature_vector(row, local=local) - mean) / scale
            by_action[str(row["grader_winner"])].append(vector)
        centroids = {
            action: np.stack(vectors).mean(axis=0)
            for action, vectors in by_action.items()
            if vectors
        }
        for row in test:
            vector = (_feature_vector(row, local=local) - mean) / scale
            choices[str(row["case_id"])] = min(
                centroids,
                key=lambda action: (_distance(vector, centroids[action]), action),
            )
    return choices


def _rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise ValueError("rank AUC requires positive and negative examples")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty value sequence")
    return float(statistics.median(float(value) for value in values))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty value sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _cohort_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    summaries: dict[str, object] = {}
    groups = {
        "all": list(rows),
        "positive": [row for row in rows if row["cohort"] == "positive"],
        "risk": [row for row in rows if row["cohort"] == "risk"],
    }
    for archetype in POSITIVE_ARCHETYPES + RISK_ARCHETYPES:
        groups[f"archetype:{archetype}"] = [row for row in rows if row["archetype"] == archetype]
    for name, group in groups.items():
        arm_losses = {
            program: [float(row["arms"][program]["loss_j"]) for row in group]
            for program in PROGRAM_IDS
        }
        summaries[name] = {
            "n_cases": len(group),
            "mean_loss_by_program": {
                program: _mean(losses) for program, losses in arm_losses.items()
            },
            "median_loss_by_program": {
                program: _median(losses) for program, losses in arm_losses.items()
            },
            "mean_menu_oracle_loss": _mean([float(row["menu_oracle_loss"]) for row in group]),
            "mean_oracle_gain_over_identity": _mean(
                [float(row["oracle_gain_over_identity"]) for row in group]
            ),
            "winner_counts": {
                program: sum(row["grader_winner"] == program for row in group)
                for program in PROGRAM_IDS
            },
        }
    return summaries


def _policy_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for cohort in ("all", "positive", "risk"):
        group = list(rows) if cohort == "all" else [row for row in rows if row["cohort"] == cohort]
        fixed_losses = {
            program: _mean([float(row["arms"][program]["loss_j"]) for row in group])
            for program in PROGRAM_IDS
        }
        best_fixed = min(fixed_losses, key=lambda program: (fixed_losses[program], program))
        oracle_losses = [float(row["menu_oracle_loss"]) for row in group]
        global_losses = [
            float(row["arms"][str(row["global_policy_action"])]["loss_j"]) for row in group
        ]
        local_losses = [
            float(row["arms"][str(row["local_policy_action"])]["loss_j"]) for row in group
        ]
        result[cohort] = {
            "n_cases": len(group),
            "best_fixed_action": best_fixed,
            "best_fixed_mean_loss": fixed_losses[best_fixed],
            "menu_oracle_mean_loss": _mean(oracle_losses),
            "global_loso_policy_mean_loss": _mean(global_losses),
            "local_loso_policy_mean_loss": _mean(local_losses),
            "global_loso_policy_mean_regret": _mean(
                [loss - oracle for loss, oracle in zip(global_losses, oracle_losses)]
            ),
            "local_loso_policy_mean_regret": _mean(
                [loss - oracle for loss, oracle in zip(local_losses, oracle_losses)]
            ),
            "global_action_counts": {
                program: sum(row["global_policy_action"] == program for row in group)
                for program in PROGRAM_IDS
            },
            "local_action_counts": {
                program: sum(row["local_policy_action"] == program for row in group)
                for program in PROGRAM_IDS
            },
        }
    return result


def run_e1p_periodic_missing(valuator: FrozenChronosValuator) -> dict[str, object]:
    cases = build_pilot_cases()
    rows: list[dict[str, object]] = []
    for case in cases:
        if (
            case.clean_context.shape != (CONTEXT_LENGTH,)
            or case.corrupt_context.shape != (CONTEXT_LENGTH,)
            or case.clean_future.shape != (FUTURE_LENGTH,)
        ):
            raise AssertionError("E1-P case violates the frozen consumer lengths")

        # Information wall: both feature families and the executable period parameter
        # are resolved before, and without, passing any private clean value.
        global_features, observed_period = _global_features(case.corrupt_context)
        local_features = _local_features(case.corrupt_context, observed_period)

        arms: dict[str, dict[str, object]] = {}
        for program_id in PROGRAM_IDS:
            prepared = _execute_program(
                program_id,
                case.corrupt_context,
                observed_period=observed_period,
            )
            receipt = valuator.evaluate(
                prepared,
                case.clean_future,
                scale_context=case.clean_context,
            )
            arms[program_id] = {
                "loss_j": float(receipt.loss_j),
                "utility_u": float(receipt.utility_u),
                "status": receipt.status,
            }

        ranked = sorted(
            ((float(arms[program]["loss_j"]), program) for program in PROGRAM_IDS),
            key=lambda item: (item[0], item[1]),
        )
        winner_loss, winner = ranked[0]
        margin = float(ranked[1][0] - winner_loss)
        identity_loss = float(arms["identity"]["loss_j"])
        seasonal_contraindication_margin = float(
            arms["seasonal"]["loss_j"]
        ) - min(float(arms["identity"]["loss_j"]), float(arms["linear"]["loss_j"]))
        rows.append(
            {
                "case_id": case.case_id,
                "seed": case.seed,
                # Private grader/report fields. They never enter feature extraction or
                # held-out policy prediction for this row.
                "cohort": case.cohort,
                "archetype": case.archetype,
                "generator_period_private": case.generator_period,
                "archetype_risk_label_descriptive_only": 1 if case.cohort == "risk" else 0,
                # Deployment-visible evidence.
                "observed_period_parameter": observed_period,
                "missing_runs": [list(run) for run in _missing_runs(case.corrupt_context)],
                "global_features": global_features,
                "local_features": local_features,
                # Grader outcomes and the non-deployable menu ceiling.
                "arms": arms,
                "grader_winner": winner,
                "winner_margin": margin,
                "menu_oracle_loss": winner_loss,
                "oracle_gain_over_identity": identity_loss - winner_loss,
                "seasonal_contraindication_margin": seasonal_contraindication_margin,
                "seasonal_contraindication_label": int(
                    seasonal_contraindication_margin
                    >= PILOT_THRESHOLDS["risk_winner_margin_min"]
                ),
            }
        )

    global_risk_scores = _loso_risk_scores(rows, local=False)
    local_risk_scores = _loso_risk_scores(rows, local=True)
    global_actions = _loso_action_choices(rows, local=False)
    local_actions = _loso_action_choices(rows, local=True)
    for row in rows:
        case_id = str(row["case_id"])
        row["global_loso_risk_score"] = global_risk_scores[case_id]
        row["local_loso_risk_score"] = local_risk_scores[case_id]
        row["global_policy_action"] = global_actions[case_id]
        row["local_policy_action"] = local_actions[case_id]

    labels = [int(row["seasonal_contraindication_label"]) for row in rows]
    global_auc = _rank_auc(labels, [global_risk_scores[str(row["case_id"])] for row in rows])
    local_auc = _rank_auc(labels, [local_risk_scores[str(row["case_id"])] for row in rows])

    positive_gain_by_archetype = {
        archetype: _median(
            [
                float(row["oracle_gain_over_identity"])
                for row in rows
                if row["archetype"] == archetype
            ]
        )
        for archetype in POSITIVE_ARCHETYPES
    }
    passing_positive_archetypes = [
        archetype
        for archetype, gain in positive_gain_by_archetype.items()
        if gain >= PILOT_THRESHOLDS["positive_archetype_oracle_gain_min"]
    ]
    positive_seasonal_winners = [
        str(row["case_id"])
        for row in rows
        if row["cohort"] == "positive"
        and row["grader_winner"] == "seasonal"
        and float(row["winner_margin"]) >= PILOT_THRESHOLDS["risk_winner_margin_min"]
    ]
    other_risk_winners = [
        str(row["case_id"])
        for row in rows
        if row["cohort"] == "risk"
        and row["grader_winner"] != "seasonal"
        and float(row["winner_margin"]) >= PILOT_THRESHOLDS["risk_winner_margin_min"]
    ]
    other_risk_winner_archetypes = sorted(
        {
            str(row["archetype"])
            for row in rows
            if str(row["case_id"]) in set(other_risk_winners)
        }
    )
    fixed_mean_losses = {
        program: _mean([float(row["arms"][program]["loss_j"]) for row in rows])
        for program in PROGRAM_IDS
    }
    best_fixed_action = min(
        fixed_mean_losses, key=lambda program: (fixed_mean_losses[program], program)
    )
    menu_oracle_mean_loss = _mean([float(row["menu_oracle_loss"]) for row in rows])
    routing_headroom = fixed_mean_losses[best_fixed_action] - menu_oracle_mean_loss
    risk_contraindication_by_archetype = {
        archetype: {
            "case_count": sum(row["archetype"] == archetype for row in rows),
            "realized_contraindication_count": sum(
                row["archetype"] == archetype
                and int(row["seasonal_contraindication_label"]) == 1
                for row in rows
            ),
        }
        for archetype in RISK_ARCHETYPES
    }
    for values in risk_contraindication_by_archetype.values():
        values["realized_contraindication_rate"] = (
            int(values["realized_contraindication_count"]) / int(values["case_count"])
        )
    covered_risk_archetypes = sorted(
        archetype
        for archetype, values in risk_contraindication_by_archetype.items()
        if int(values["realized_contraindication_count"]) > 0
    )
    corpus_risk_coverage_status = (
        "COMPLETE"
        if len(covered_risk_archetypes) == len(RISK_ARCHETYPES)
        else "PARTIAL"
    )

    gates = {
        "headroom": {
            "gate_status": "pilot_gate_not_confirmation",
            "thresholds": {
                "seed_median_oracle_gain_min": PILOT_THRESHOLDS[
                    "positive_archetype_oracle_gain_min"
                ],
                "positive_archetypes_required": PILOT_THRESHOLDS[
                    "positive_archetypes_required"
                ],
            },
            "seed_median_oracle_gain_by_positive_archetype": positive_gain_by_archetype,
            "passing_positive_archetypes": passing_positive_archetypes,
            "pass": len(passing_positive_archetypes)
            >= PILOT_THRESHOLDS["positive_archetypes_required"],
        },
        "action_heterogeneity": {
            "gate_status": "pilot_gate_not_confirmation",
            "thresholds": {
                "positive_seasonal_winner_margin_min": PILOT_THRESHOLDS[
                    "risk_winner_margin_min"
                ],
                "positive_seasonal_winner_count_min": 2,
                "other_action_risk_winner_margin_min": PILOT_THRESHOLDS[
                    "risk_winner_margin_min"
                ],
                "other_action_risk_winner_archetype_count_min": 2,
                "best_fixed_over_menu_oracle_mean_loss_min": 0.01,
            },
            "positive_seasonal_winner_case_ids": positive_seasonal_winners,
            "other_action_margin_winner_risk_case_ids": other_risk_winners,
            "other_action_margin_winner_risk_archetypes": other_risk_winner_archetypes,
            "best_fixed_action": best_fixed_action,
            "best_fixed_mean_loss": fixed_mean_losses[best_fixed_action],
            "menu_oracle_mean_loss": menu_oracle_mean_loss,
            "best_fixed_minus_menu_oracle_mean_loss": routing_headroom,
            "risk_archetype_realized_contraindication": risk_contraindication_by_archetype,
            "risk_corpus_coverage_status": corpus_risk_coverage_status,
            "risk_archetypes_with_realized_contraindication": covered_risk_archetypes,
            "pass": len(positive_seasonal_winners) >= 2
            and len(other_risk_winner_archetypes) >= 2
            and routing_headroom >= 0.01,
        },
        "local_vs_global_seasonal_contraindication_prediction": {
            "gate_status": "pilot_gate_not_confirmation",
            "outcome_definition": (
                "seasonal_loss - min(identity_loss, linear_loss) >= 0.005"
            ),
            "protocol": (
                "outcome_defined_seasonal_contraindication; leave_one_seed_out source "
                "centroids; heldout target score uses corrupt-context features only; "
                "heldout outcome label is used only by grader rank AUC"
            ),
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
            "local_extra_feature_names": list(LOCAL_EXTRA_FEATURE_NAMES),
            "global_auc": global_auc,
            "local_auc": local_auc,
            "local_minus_global_auc": local_auc - global_auc,
            "realized_contraindication_count": sum(labels),
            "realized_safe_count": len(labels) - sum(labels),
            "thresholds": {
                "local_auc_min": PILOT_THRESHOLDS["local_risk_auc_min"],
                "local_minus_global_auc_min": PILOT_THRESHOLDS[
                    "local_over_global_auc_min"
                ],
            },
            "pass": local_auc >= PILOT_THRESHOLDS["local_risk_auc_min"]
            and local_auc - global_auc
            >= PILOT_THRESHOLDS["local_over_global_auc_min"],
        },
    }

    return {
        "schema_version": "e1p-periodic-missing-pilot/1",
        "scientific_role": "premise_pilot",
        "gate_interpretation": "pilot_gate_not_confirmation",
        "consumer": {
            "valuator": "FrozenChronosValuator",
            "model_id": str(valuator.manifest["model_id"]),
            "revision": str(valuator.manifest["revision"]),
            "device": str(valuator.manifest["device"]),
            "context_length": int(valuator.manifest["context_length"]),
            "prediction_length": int(valuator.manifest["prediction_length"]),
            "ingestion_policy_id": valuator.ingestion_policy_id,
        },
        "configuration": {
            "seeds": list(SEEDS),
            "positive_archetypes": list(POSITIVE_ARCHETYPES),
            "matched_risk_archetypes": list(RISK_ARCHETYPES),
            "programs": {
                "identity": [],
                "linear": [["impute_linear", {}]],
                "seasonal": [["period_complete", {"period_from": "public_pre_period"}]],
            },
            "thresholds": dict(PILOT_THRESHOLDS),
            "agent_enabled": False,
            "memory_enabled": False,
            "promotion_enabled": False,
        },
        "information_wall": {
            "target_feature_sources": ["corrupt_context", "missing_mask"],
            "target_action_selection_sources": [
                "corrupt_context_features",
                "other_seed_source_outcome_centroids",
            ],
            "target_clean_completion_visible_to_features_or_policy": False,
            "target_clean_future_visible_to_features_or_policy": False,
            "private_fields_used_only_by_grader_and_report": [
                "clean_context",
                "clean_future",
                "archetype",
                "risk_label",
                "seasonal_contraindication_label",
                "menu_oracle",
            ],
        },
        "risk_archetype_realized_contraindication": risk_contraindication_by_archetype,
        "risk_corpus_coverage_status": corpus_risk_coverage_status,
        "cases": rows,
        "cohort_aggregate": _cohort_summary(rows),
        "policy_aggregate": _policy_summary(rows),
        "gates": gates,
        "all_pilot_gates_pass": all(bool(gate["pass"]) for gate in gates.values()),
        "claim_limit": (
            "This deterministic synthetic run can pass or fail E1-P premises only. It is not "
            "cross-domain transfer, adaptation, Memory, promotion, or confirmation evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the zero-Agent E1-P periodic-missing premise pilot."
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    valuator = FrozenChronosValuator()
    report = run_e1p_periodic_missing(valuator)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={output}")
    print(f"all_pilot_gates_pass={report['all_pilot_gates_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_pilot_cases", "run_e1p_periodic_missing"]
