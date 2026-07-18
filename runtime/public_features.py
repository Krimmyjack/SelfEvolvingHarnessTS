from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.observables import (
    OBSERVABLE_FEATURES,
    PERIOD_RELIABILITY_MIN,
)


_MAD_TO_SIGMA = 1.4826
_MAD_FLOOR = 1e-8
_OUTLIER_Z_THRESHOLD = 4.0
_LEVEL_BOUNDARY_THRESHOLD = 0.85
_LEVEL_MIN_WIDTH = 40
_LEVEL_MAX_WIDTH = 96
_LEVEL_WINDOW = 24
_LEVEL_SCORE_SCALE = 3.0
_LEVEL_TAIL_REFINEMENT = 8
_LEVEL_MIN_POST = 12


def _array_sha(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").copy()
    canonical[np.isnan(canonical)] = np.nan
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _fill(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    indices = np.arange(values.size, dtype=np.float64)
    return np.interp(indices, indices[finite], values[finite])


def _longest_run(mask: np.ndarray) -> int:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 0
    boundaries = np.flatnonzero(np.diff(indices) > 1) + 1
    return max(len(group) for group in np.split(indices, boundaries))


def _longest_observed_segment(values: np.ndarray) -> np.ndarray:
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return np.asarray([], dtype=np.float64)
    boundaries = np.flatnonzero(np.diff(finite_indices) > 1) + 1
    groups = np.split(finite_indices, boundaries)
    best = max(groups, key=len)
    return np.asarray(values[best], dtype=np.float64)


def _expand(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    expanded = mask.copy()
    for shift in range(1, radius + 1):
        expanded[shift:] |= mask[:-shift]
        expanded[:-shift] |= mask[shift:]
    return expanded


def _dominant_acf_period(values: np.ndarray) -> int:
    centered = values - float(np.mean(values))
    lags = tuple(range(4, min(48, values.size - 2) + 1))
    scores: list[float] = []
    for lag in lags:
        left = centered[:-lag]
        right = centered[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        scores.append(float(left @ right) / denominator if denominator > 1e-12 else -1.0)
    return int(lags[int(np.argmax(scores))]) if scores else 4


def _dominant_spectral_period(values: np.ndarray) -> float:
    centered = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(values.size)
    eligible = (frequencies >= 1.0 / 48.0) & (frequencies <= 1.0 / 4.0)
    eligible[0] = False
    if not np.any(eligible):
        return 4.0
    indices = np.flatnonzero(eligible)
    best = int(indices[int(np.argmax(spectrum[indices]))])
    return float(1.0 / frequencies[best])


def _period_summary(values: np.ndarray) -> tuple[int, int, float, float, float, str]:
    pre_raw = values[:80]
    post_raw = values[-80:]
    pre = _longest_observed_segment(pre_raw)
    post = _longest_observed_segment(post_raw)
    if pre.size < 16 or post.size < 16:
        return 4, 4, 0.0, 0.0, 0.0, "UNKNOWN"
    pre_period = _dominant_acf_period(pre)
    post_period = _dominant_acf_period(post)
    score = abs(post_period - pre_period) / max(pre_period, 1)
    spectral = (_dominant_spectral_period(pre), _dominant_spectral_period(post))
    agreements = (
        1.0 - min(abs(pre_period - spectral[0]) / max(pre_period, 1), 1.0),
        1.0 - min(abs(post_period - spectral[1]) / max(post_period, 1), 1.0),
    )
    coverage = min(pre.size / pre_raw.size, post.size / post_raw.size)
    longest_gap = max(
        _longest_run(~np.isfinite(pre_raw)),
        _longest_run(~np.isfinite(post_raw)),
    )
    gap_penalty = max(
        0.0,
        1.0 - longest_gap / max(min(pre_period, post_period), 4),
    )
    reliability = float(np.clip(coverage * gap_penalty, 0.0, 1.0))
    status = "OK" if reliability >= PERIOD_RELIABILITY_MIN else "UNKNOWN"
    return (
        pre_period,
        post_period,
        float(score) if status == "OK" else 0.0,
        float(np.mean(agreements)) if status == "OK" else 0.0,
        reliability,
        status,
    )


def _level_candidate(
    filled: np.ndarray,
    scale: float,
) -> tuple[float, np.ndarray, float]:
    splits = np.arange(
        _LEVEL_WINDOW,
        filled.size - _LEVEL_WINDOW + 1,
        dtype=int,
    )
    differences = np.asarray(
        [
            np.median(filled[split : split + _LEVEL_WINDOW])
            - np.median(filled[split - _LEVEL_WINDOW : split])
            for split in splits
        ],
        dtype=np.float64,
    )
    mask = np.zeros(filled.size, dtype=bool)
    if differences.size == 0 or scale <= 0:
        return 0.0, mask, 0.0
    best: tuple[float, float, int, int] | None = None
    for first_index, start in enumerate(splits):
        for second_index in range(first_index + 1, len(splits)):
            end = int(splits[second_index])
            width = end - int(start)
            if width < _LEVEL_MIN_WIDTH or width > _LEVEL_MAX_WIDTH:
                continue
            first = float(differences[first_index])
            second = float(differences[second_index])
            if first * second >= 0.0:
                continue
            boundary = min(abs(first), abs(second)) / scale
            mean_boundary = (abs(first) + abs(second)) / (2.0 * scale)
            candidate = (boundary, mean_boundary, int(start), end)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None or best[0] < _LEVEL_BOUNDARY_THRESHOLD:
        return 0.0, mask, 0.0
    boundary, _mean_boundary, start, end = best
    if end - start >= 55:
        refined: list[tuple[float, int, float]] = []
        maximum_end = min(
            filled.size - _LEVEL_MIN_POST,
            end + _LEVEL_TAIL_REFINEMENT,
        )
        first_direction = float(
            np.median(filled[start : start + _LEVEL_WINDOW])
            - np.median(filled[start - _LEVEL_WINDOW : start])
        )
        for candidate_end in range(end, maximum_end + 1):
            post = filled[candidate_end : min(filled.size, candidate_end + _LEVEL_WINDOW)]
            if post.size < _LEVEL_MIN_POST:
                continue
            difference = float(
                np.median(post)
                - np.median(filled[candidate_end - _LEVEL_WINDOW : candidate_end])
            )
            if first_direction * difference < 0.0:
                refined.append((abs(difference), candidate_end, difference))
        if refined:
            _strength, end, refined_difference = max(
                refined, key=lambda item: (item[0], -item[1])
            )
            boundary = min(abs(first_direction), abs(refined_difference)) / scale
    mask[start:end] = True
    left = filled[max(0, start - _LEVEL_WINDOW) : start]
    right = filled[end : min(filled.size, end + _LEVEL_WINDOW)]
    references = [float(np.median(block)) for block in (left, right) if block.size]
    reference = float(np.mean(references)) if references else float(np.median(filled))
    offset = float(np.median(filled[mask]) - reference)
    return float(boundary * _LEVEL_SCORE_SCALE), mask, offset


@dataclass(frozen=True)
class PublicFeatureExtraction:
    mapping: Mapping[str, object]
    feature_context_sha: str
    missing_indices: tuple[int, ...]
    outlier_indices: tuple[int, ...]
    region_mask: np.ndarray
    level_mask: np.ndarray
    estimated_excursion_offset: float
    pre_period: int
    post_period: int
    acf_spectral_consistency: float
    period_reliability: float
    period_evidence_status: str

    def with_probe_directions(
        self,
        directions: Mapping[str, str],
    ) -> "PublicFeatureExtraction":
        allowed = {"positive", "flat", "overdose_collapse", "negative", "unknown"}
        keys = {
            "imputation": "imputation_probe_direction",
            "clipping": "clipping_probe_direction",
            "denoising": "denoising_probe_direction",
            "level_correction": "level_probe_direction",
        }
        updated = dict(self.mapping)
        for probe, feature in keys.items():
            value = str(directions.get(probe, "unknown"))
            if value not in allowed:
                raise ValueError(f"invalid probe direction: {value}")
            updated[feature] = value
        digest = canonical_sha256(
            {
                "schema_version": "public-feature-context/2",
                "base_feature_context_sha": self.feature_context_sha,
                "mapping": updated,
            }
        )
        return replace(self, mapping=MappingProxyType(updated), feature_context_sha=digest)


def extract_public_features(
    series: object,
    *,
    task_kind: str = "forecast",
) -> PublicFeatureExtraction:
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("public feature extractor requires one non-empty series")
    missing = ~np.isfinite(values)
    filled = _fill(values)
    median = float(np.median(filled))
    mad = float(np.median(np.abs(filled - median)))
    scale = max(_MAD_TO_SIGMA * mad, _MAD_FLOOR)
    robust_z = np.abs(filled - median) / scale
    outliers = robust_z >= _OUTLIER_Z_THRESHOLD
    outlier_region = _expand(outliers)
    missing_region = _expand(missing)
    if np.any(missing):
        level_score = 0.0
        level_mask = np.zeros(values.size, dtype=bool)
        offset = 0.0
    else:
        level_score, level_mask, offset = _level_candidate(filled, scale)
    union = missing_region | outlier_region | level_mask
    selected = np.flatnonzero(union)
    if selected.size:
        start_fraction = float(selected[0] / values.size)
        end_fraction = float((selected[-1] + 1) / values.size)
    else:
        start_fraction = end_fraction = 0.0
    period_values = values.copy()
    if np.any(level_mask):
        period_values[level_mask] = period_values[level_mask] - offset
    (
        pre_period,
        post_period,
        period_score,
        consistency,
        period_reliability,
        period_evidence_status,
    ) = _period_summary(period_values)
    mapping: dict[str, object] = {
        "task_kind": str(task_kind),
        "missing_fraction": float(np.mean(missing)),
        "longest_missing_run_fraction": _longest_run(missing) / values.size,
        "local_robust_z_peak": float(np.max(robust_z)),
        "estimated_region_start_fraction": start_fraction,
        "estimated_region_end_fraction": end_fraction,
        "level_excursion_score": level_score,
        "estimated_level_offset": offset,
        "period_change_score": period_score,
        "period_reliability": period_reliability,
        "period_evidence_status": period_evidence_status,
        "period_repair_available": False,
    }
    if not set(mapping) <= set(OBSERVABLE_FEATURES):
        raise AssertionError("public extractor emitted a feature outside the closed vocabulary")
    feature_context_sha = canonical_sha256(
        {
            "schema_version": "public-feature-context/2",
            "input_float64_sha": _array_sha(values),
            "mapping": mapping,
            "region_mask_sha": hashlib.sha256(union.tobytes()).hexdigest(),
        }
    )
    union.setflags(write=False)
    level_mask.setflags(write=False)
    return PublicFeatureExtraction(
        mapping=MappingProxyType(mapping),
        feature_context_sha=feature_context_sha,
        missing_indices=tuple(int(index) for index in np.flatnonzero(missing)),
        outlier_indices=tuple(int(index) for index in np.flatnonzero(outliers)),
        region_mask=union,
        level_mask=level_mask,
        estimated_excursion_offset=offset,
        pre_period=pre_period,
        post_period=post_period,
        acf_spectral_consistency=consistency,
        period_reliability=period_reliability,
        period_evidence_status=period_evidence_status,
    )


__all__ = ["PublicFeatureExtraction", "extract_public_features"]
