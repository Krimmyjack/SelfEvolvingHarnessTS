"""Content-keyed benchmark corruption with common-random-number pairing.

The v0.1 grid is pre-registered in full *before* any method is run against it, so
that a later "the benchmark could not tell methods apart" result cannot be repaired
by quietly adding defect types that happen to favour some method.  It has three
lanes:

``natural`` (dose 0)
    No synthetic corruption at all.  The series keeps exactly its natural
    missingness, which for several sources (NOAA ISD, UCI, GEFCom) is substantial
    and is the defect the readiness pipeline actually meets in the wild.  This lane
    is deterministic in the corruption RNG, so it carries exactly ONE replicate;
    every stochastic lane carries two.  It is the honest floor: a method that helps
    only under synthetic damage is not a data-readiness method.

``Controlled v0`` (block / scattered missingness)
    The three cells inherited unchanged from benchmark-v0, kept so v0's Dev results
    remain comparable at the cell level.  Note the *seeds* differ from v0's because
    the benchmark version is part of the corruption key -- the cells are the same
    question, drawn fresh.

``Controlled v0.1`` (new defect families)
    Outliers (``spike``), structural break (``level_shift``), additive noise
    (``gaussian``), and timestamp disorder (``local_permutation``).  These probe
    failure modes that pure missingness cannot: a pipeline that only imputes will
    score identically to Raw on all four.

`dose` is a scenario-specific intensity parameter, not one universal unit:

  * block / scattered / spike / level_shift -- fraction of observations affected;
  * gaussian -- additive noise sigma as a multiple of the series' robust scale;
  * natural -- unused, pinned to 0.0.

Scale-relative scenarios (spike, level_shift, gaussian) derive their magnitude from
a robust scale (MAD) of the series' own finite observations, so a defect means the
same thing on a 10-kW meter and a 10,000-vehicle traffic count.  The robust scale is
computed from the values handed to `apply_corruption` -- the clean pre-method
history -- never from held-out truth.

Timestamp disorder is realized in the value domain (`local_permutation`) rather than
by perturbing a timestamp array, because a large part of the roster (all legacy
Monash series) carries no timestamps at all.  Permuting values inside a short window
is exactly what a jittered/re-sorted timestamp index does to the observed series, and
it applies uniformly to every series regardless of timestamp availability.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

from . import BENCHMARK_VERSION, CORRUPTION_REPLICATES, KNOWN_BENCHMARK_VERSIONS

NATURAL_LANE = (("natural", 0.0),)

CONTROLLED_V0_LANE = (
    ("block", 0.12),
    ("block", 0.24),
    ("scattered", 0.12),
)

CONTROLLED_V0_1_LANE = (
    ("spike", 0.01),
    ("spike", 0.03),
    ("level_shift", 0.05),
    ("gaussian", 0.50),
    ("local_permutation", 0.05),
)

CORRUPTION_GRID = NATURAL_LANE + CONTROLLED_V0_LANE + CONTROLLED_V0_1_LANE

LANE_OF_SCENARIO = {
    "natural": "natural",
    "block": "controlled_v0",
    "scattered": "controlled_v0",
    "spike": "controlled_v0_1",
    "level_shift": "controlled_v0_1",
    "gaussian": "controlled_v0_1",
    "local_permutation": "controlled_v0_1",
}

DETERMINISTIC_SCENARIOS = frozenset({"natural"})

SPIKE_MAGNITUDE_ROBUST_SIGMA = 6.0
LEVEL_SHIFT_MAGNITUDE_ROBUST_SIGMA = 2.0
LOCAL_PERMUTATION_WINDOW = 6
_MAD_TO_SIGMA = 1.4826
_ROBUST_SCALE_FLOOR = 1e-8

__all__ = [
    "CONTROLLED_V0_1_LANE",
    "CONTROLLED_V0_LANE",
    "CORRUPTION_GRID",
    "DETERMINISTIC_SCENARIOS",
    "LANE_OF_SCENARIO",
    "NATURAL_LANE",
    "apply_corruption",
    "corruption_seed",
    "materialize_corruptions",
    "replicates_for",
    "robust_scale",
]


def replicates_for(scenario: str) -> tuple[int, ...]:
    """Deterministic lanes carry one replicate; stochastic lanes carry the frozen two."""

    if scenario not in LANE_OF_SCENARIO:
        raise ValueError(f"unknown corruption scenario: {scenario!r}")
    if scenario in DETERMINISTIC_SCENARIOS:
        return (CORRUPTION_REPLICATES[0],)
    return tuple(CORRUPTION_REPLICATES)


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
    if version not in KNOWN_BENCHMARK_VERSIONS:
        raise ValueError(
            f"benchmark version must be one of {KNOWN_BENCHMARK_VERSIONS!r}"
        )
    _require_sha256(content_sha)
    scenario, dose_value = _require_grid(scenario, dose)
    if (
        isinstance(replicate_idx, bool)
        or not isinstance(replicate_idx, int)
        or replicate_idx not in replicates_for(scenario)
    ):
        raise ValueError(
            f"replicate_idx must be one of {replicates_for(scenario)!r} for {scenario!r}"
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


def robust_scale(values: np.ndarray) -> float:
    """MAD-based sigma of the finite observations; floored so constant series stay usable."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot scale a series with no finite observations")
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = _MAD_TO_SIGMA * mad
    if scale <= _ROBUST_SCALE_FLOOR:
        # A near-constant series has no usable MAD; fall back to the standard
        # deviation, then to a unit scale, so scale-relative defects stay defined.
        scale = float(np.std(finite))
    return max(scale, _ROBUST_SCALE_FLOOR)


def _affected_count(size: int, dose: float) -> int:
    count = max(1, int(round(size * dose)))
    if count > size:
        raise ValueError("dose affects more values than the series contains")
    return count


def apply_corruption(
    values: np.ndarray, *, scenario: str, dose: float, seed: int
) -> np.ndarray:
    scenario, dose_value = _require_grid(scenario, dose)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 64:
        raise ValueError("seed must be a uint64 integer")
    result = _canonical_values(values)
    rng = np.random.default_rng(seed)

    if scenario == "natural":
        # The identity lane: natural missingness only. Deliberately does not touch rng.
        pass
    elif scenario == "block":
        count = _affected_count(result.size, dose_value)
        start = int(rng.integers(0, result.size - count + 1))
        result[start : start + count] = np.nan
    elif scenario == "scattered":
        count = _affected_count(result.size, dose_value)
        indices = rng.choice(result.size, size=count, replace=False)
        result[indices] = np.nan
    elif scenario == "spike":
        count = _affected_count(result.size, dose_value)
        scale = robust_scale(result)
        observed = np.flatnonzero(np.isfinite(result))
        if observed.size == 0:
            raise ValueError("spike corruption needs at least one observed value")
        count = min(count, observed.size)
        indices = rng.choice(observed, size=count, replace=False)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=count)
        result[indices] += signs * SPIKE_MAGNITUDE_ROBUST_SIGMA * scale
    elif scenario == "level_shift":
        count = _affected_count(result.size, dose_value)
        scale = robust_scale(result)
        start = int(rng.integers(0, result.size - count + 1))
        sign = float(rng.choice(np.asarray([-1.0, 1.0])))
        segment = result[start : start + count]
        shifted = segment + sign * LEVEL_SHIFT_MAGNITUDE_ROBUST_SIGMA * scale
        # NaNs stay NaN: a level shift moves observations, it does not create them.
        result[start : start + count] = np.where(
            np.isfinite(segment), shifted, segment
        )
    elif scenario == "gaussian":
        scale = robust_scale(result)
        noise = rng.normal(0.0, dose_value * scale, size=result.size)
        observed = np.isfinite(result)
        result[observed] += noise[observed]
    elif scenario == "local_permutation":
        # Timestamp disorder in the value domain: shuffle observations inside disjoint
        # short windows, which is what a jittered timestamp index does to the series.
        # The windows tile the series so they cannot overlap; an overlapping draw would
        # re-shuffle already-shuffled values and make the affected count a lie.
        count = _affected_count(result.size, dose_value)
        n_tiles = result.size // LOCAL_PERMUTATION_WINDOW
        if n_tiles >= 1:
            n_windows = min(
                max(1, count // LOCAL_PERMUTATION_WINDOW),
                n_tiles,
            )
            tiles = rng.choice(n_tiles, size=n_windows, replace=False)
            for tile in np.sort(tiles):
                start = int(tile) * LOCAL_PERMUTATION_WINDOW
                stop = start + LOCAL_PERMUTATION_WINDOW
                window = result[start:stop]
                result[start:stop] = window[rng.permutation(window.size)]
    else:  # guarded by the frozen grid
        raise AssertionError(scenario)

    if np.isinf(result).any():
        raise ValueError("corruption produced a non-finite value")
    result.setflags(write=False)
    return result


def materialize_corruptions(
    values_by_uid: Mapping[str, np.ndarray],
    content_sha_by_uid: Mapping[str, str],
    scenario: str,
    dose: float,
    replicate_idx: int,
    benchmark_version: str = BENCHMARK_VERSION,
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
