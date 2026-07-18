from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TOTAL_LENGTH = 240
CONTEXT_LENGTH = 192
FUTURE_LENGTH = 48


@dataclass(frozen=True)
class InjectionResult:
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray
    affected_indices: tuple[int, ...]


def generate_base_series(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(TOTAL_LENGTH, dtype=np.float64)
    period = 24.0 + (seed % 3) * 2.0
    return (
        0.0025 * t
        + 0.8 * np.sin(2.0 * np.pi * t / period)
        + 0.25 * np.sin(2.0 * np.pi * t / 7.0 + 0.3)
        + rng.normal(0.0, 0.04, size=t.size)
    ).astype(np.float64)


def _scale(clean_context: np.ndarray) -> float:
    return max(float(np.std(clean_context)), 1e-8)


def inject_target(seed: int, family: str, severity: str) -> InjectionResult:
    if severity not in {"mild", "severe"}:
        raise ValueError("target severity must be mild or severe")
    base = generate_base_series(seed)
    clean_context = base[:CONTEXT_LENGTH].copy()
    corrupt = clean_context.copy()
    clean_future = base[CONTEXT_LENGTH:].copy()
    scale = _scale(clean_context)
    if family == "missing":
        start, length = (108, 12) if severity == "mild" else (102, 30)
        affected = tuple(range(start, start + length))
        corrupt[list(affected)] = np.nan
    elif family == "impulsive_outlier":
        positions = (119, 147) if severity == "mild" else (111, 128, 149, 166)
        amplitude = 6.0 if severity == "mild" else 10.0
        affected = tuple(positions)
        for offset, index in enumerate(positions):
            corrupt[index] += (1.0 if offset % 2 == 0 else -1.0) * amplitude * scale
    elif family == "level_shift":
        start, end, amplitude = (
            (128, 168, 1.5) if severity == "mild" else (112, 176, 3.0)
        )
        affected = tuple(range(start, end))
        corrupt[start:end] += amplitude * scale
    elif family == "period_change":
        start, end = 96, CONTEXT_LENGTH
        period = 24.0 + (seed % 3) * 2.0
        factor = 0.75 if severity == "mild" else 0.55
        indices = np.arange(start, end, dtype=np.float64)
        old_primary = 0.8 * np.sin(2.0 * np.pi * indices / period)
        new_primary = 0.8 * np.sin(2.0 * np.pi * indices / (factor * period))
        corrupt[start:end] += new_primary - old_primary
        affected = tuple(range(start, end))
    else:
        raise ValueError(f"unknown target family: {family}")
    return InjectionResult(clean_context, corrupt, clean_future, affected)


def build_risk_series(
    seed: int,
    family: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    base = generate_base_series(seed)
    scale = _scale(base[:CONTEXT_LENGTH])
    if family == "missing":
        return base[:CONTEXT_LENGTH].copy(), base[CONTEXT_LENGTH:].copy(), "clean"
    if family == "impulsive_outlier":
        event = base.copy()
        event[186:188] += np.asarray([2.0, 3.0]) * scale
        event[192:196] += np.asarray([2.6, 1.8, 1.0, 0.4]) * scale
        return event[:CONTEXT_LENGTH], event[CONTEXT_LENGTH:], "genuine"
    if family == "level_shift":
        event = base.copy()
        event[156:] += 2.0 * scale
        return event[:CONTEXT_LENGTH], event[CONTEXT_LENGTH:], "genuine"
    if family == "period_change":
        event = base.copy()
        start, end = 120, TOTAL_LENGTH
        period = 24.0 + (seed % 3) * 2.0
        indices = np.arange(start, end, dtype=np.float64)
        old_primary = 0.8 * np.sin(2.0 * np.pi * indices / period)
        new_primary = 0.8 * np.sin(2.0 * np.pi * indices / (0.75 * period))
        event[start:end] += new_primary - old_primary
        return event[:CONTEXT_LENGTH], event[CONTEXT_LENGTH:], "genuine"
    raise ValueError(f"unknown risk family: {family}")


__all__ = [
    "CONTEXT_LENGTH",
    "FUTURE_LENGTH",
    "InjectionResult",
    "build_risk_series",
    "generate_base_series",
    "inject_target",
]
