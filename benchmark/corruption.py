"""Content-keyed benchmark corruption with common-random-number pairing."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

from . import BENCHMARK_VERSION, CORRUPTION_REPLICATES

CORRUPTION_GRID = (
    ("block", 0.12),
    ("block", 0.24),
    ("scattered", 0.12),
)

__all__ = [
    "CORRUPTION_GRID",
    "apply_corruption",
    "corruption_seed",
    "materialize_corruptions",
]


def _require_grid(scenario: str, dose: float) -> tuple[str, float]:
    if not isinstance(scenario, str) or not scenario or scenario != scenario.strip():
        raise ValueError("scenario must be a canonical non-empty string")
    if isinstance(dose, bool) or not isinstance(dose, (int, float)):
        raise ValueError("dose must be finite numeric data")
    value = float(dose)
    if not math.isfinite(value) or (scenario, value) not in CORRUPTION_GRID:
        raise ValueError("scenario/dose is not in the frozen corruption grid")
    return scenario, value


def _require_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("content_sha must be a lowercase SHA256 digest")
    return value


def corruption_seed(
    version: str,
    content_sha: str,
    scenario: str,
    dose: float,
    replicate_idx: int,
) -> int:
    if version != BENCHMARK_VERSION:
        raise ValueError(f"benchmark version must be exactly {BENCHMARK_VERSION!r}")
    _require_sha256(content_sha)
    scenario, dose_value = _require_grid(scenario, dose)
    if (
        isinstance(replicate_idx, bool)
        or not isinstance(replicate_idx, int)
        or replicate_idx not in CORRUPTION_REPLICATES
    ):
        raise ValueError(
            f"replicate_idx must be one of {CORRUPTION_REPLICATES!r}"
        )
    payload = json.dumps(
        [version, content_sha, scenario, format(dose_value, ".17g"), replicate_idx],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _canonical_values(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError("values must be one-dimensional")
    try:
        result = raw.astype("<f8", copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be numeric") from exc
    if result.size == 0 or np.isinf(result).any():
        raise ValueError("values must be non-empty and contain no infinity")
    result[np.isnan(result)] = np.nan
    return result


def apply_corruption(
    values: np.ndarray, *, scenario: str, dose: float, seed: int
) -> np.ndarray:
    scenario, dose_value = _require_grid(scenario, dose)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise ValueError("seed must be a uint64 integer")
    result = _canonical_values(values)
    count = max(1, int(round(result.size * dose_value)))
    if count > result.size:
        raise ValueError("dose masks more values than the series contains")
    rng = np.random.default_rng(seed)
    if scenario == "block":
        start = int(rng.integers(0, result.size - count + 1))
        result[start : start + count] = np.nan
    elif scenario == "scattered":
        indices = rng.choice(result.size, size=count, replace=False)
        result[indices] = np.nan
    else:  # guarded by the frozen grid
        raise AssertionError(scenario)
    result.setflags(write=False)
    return result


def materialize_corruptions(
    values_by_uid: Mapping[str, np.ndarray],
    content_sha_by_uid: Mapping[str, str],
    scenario: str,
    dose: float,
    replicate_idx: int,
    benchmark_version: str,
) -> dict[str, np.ndarray]:
    if not isinstance(values_by_uid, Mapping) or not isinstance(
        content_sha_by_uid, Mapping
    ):
        raise TypeError("values and content hashes must be mappings")
    if set(values_by_uid) != set(content_sha_by_uid):
        raise ValueError("values/content-sha uid sets differ")
    result: dict[str, np.ndarray] = {}
    for uid, values in values_by_uid.items():
        if not isinstance(uid, str) or not uid or uid != uid.strip():
            raise ValueError("uid must be a canonical non-empty string")
        canonical = _canonical_values(values)
        content_sha = _require_sha256(content_sha_by_uid[uid])
        actual_sha = hashlib.sha256(canonical.tobytes()).hexdigest()
        if actual_sha != content_sha:
            raise ValueError(f"content_sha disagrees with values for uid {uid!r}")
        seed = corruption_seed(
            benchmark_version,
            content_sha,
            scenario,
            dose,
            replicate_idx,
        )
        result[uid] = apply_corruption(
            canonical, scenario=scenario, dose=dose, seed=seed
        )
    return result
