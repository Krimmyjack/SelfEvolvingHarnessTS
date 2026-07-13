from __future__ import annotations

import hashlib

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark import BENCHMARK_VERSION
from SelfEvolvingHarnessTS.benchmark.corruption import (
    CONTROLLED_V0_LANE,
    CORRUPTION_GRID,
    LANE_OF_SCENARIO,
    LOCAL_PERMUTATION_WINDOW,
    NATURAL_LANE,
    apply_corruption,
    corruption_seed,
    materialize_corruptions,
    replicates_for,
    robust_scale,
)


def test_frozen_v0_1_corruption_grid():
    # The three v0 missingness cells survive unchanged, so v0 cells stay comparable.
    assert CONTROLLED_V0_LANE == (
        ("block", 0.12),
        ("block", 0.24),
        ("scattered", 0.12),
    )
    assert NATURAL_LANE == (("natural", 0.0),)
    assert CORRUPTION_GRID[0] == ("natural", 0.0)
    assert len(CORRUPTION_GRID) == 9
    assert len(set(CORRUPTION_GRID)) == 9
    assert all(scenario in LANE_OF_SCENARIO for scenario, _ in CORRUPTION_GRID)


def test_natural_lane_is_deterministic_and_carries_one_replicate():
    assert replicates_for("natural") == (0,)
    assert replicates_for("block") == (0, 1)

    values = np.arange(200.0)
    values[7] = np.nan  # a natural gap the source really shipped with
    seed = corruption_seed(BENCHMARK_VERSION, "a" * 64, "natural", 0.0, 0)
    got = apply_corruption(values, scenario="natural", dose=0.0, seed=seed)
    # The identity lane injects nothing: only the source's own missingness remains.
    assert np.array_equal(got, values, equal_nan=True)


def test_natural_lane_rejects_a_second_replicate():
    with pytest.raises(ValueError, match="replicate_idx"):
        corruption_seed(BENCHMARK_VERSION, "a" * 64, "natural", 0.0, 1)


def test_spike_injects_scaled_outliers_without_creating_missingness():
    rng = np.random.default_rng(0)
    values = rng.normal(100.0, 5.0, size=400)
    seed = corruption_seed(BENCHMARK_VERSION, "b" * 64, "spike", 0.03, 0)
    got = apply_corruption(values, scenario="spike", dose=0.03, seed=seed)
    assert not np.isnan(got).any(), "spike is an outlier defect, not a missingness defect"
    changed = np.flatnonzero(~np.isclose(got, values))
    assert len(changed) == 12  # 3% of 400
    # Every spike is a large multiple of the robust scale, not a nudge.
    scale = robust_scale(values)
    assert np.all(np.abs(got[changed] - values[changed]) > 4.0 * scale)


def test_level_shift_moves_a_contiguous_segment():
    values = np.full(400, 50.0) + np.random.default_rng(1).normal(0.0, 1.0, size=400)
    seed = corruption_seed(BENCHMARK_VERSION, "c" * 64, "level_shift", 0.05, 0)
    got = apply_corruption(values, scenario="level_shift", dose=0.05, seed=seed)
    changed = np.flatnonzero(~np.isclose(got, values))
    assert len(changed) == 20  # 5% of 400
    # Contiguous: a level shift is a structural break, not scattered noise.
    assert np.array_equal(changed, np.arange(changed[0], changed[0] + len(changed)))


def test_gaussian_perturbs_every_observation_but_preserves_gaps():
    values = np.linspace(0.0, 100.0, 300)
    values[10:20] = np.nan
    seed = corruption_seed(BENCHMARK_VERSION, "d" * 64, "gaussian", 0.50, 0)
    got = apply_corruption(values, scenario="gaussian", dose=0.50, seed=seed)
    assert np.array_equal(np.isnan(got), np.isnan(values)), "noise must not fill or create gaps"
    observed = np.isfinite(values)
    assert not np.allclose(got[observed], values[observed])


def test_local_permutation_preserves_the_multiset_of_values():
    values = np.arange(300.0)
    seed = corruption_seed(BENCHMARK_VERSION, "e" * 64, "local_permutation", 0.05, 0)
    got = apply_corruption(values, scenario="local_permutation", dose=0.05, seed=seed)
    # Timestamp disorder reorders observations; it neither invents nor destroys them.
    assert sorted(got.tolist()) == sorted(values.tolist())
    assert not np.array_equal(got, values)


def test_local_permutation_windows_are_disjoint_so_disorder_stays_local():
    values = np.arange(300.0)
    seed = corruption_seed(BENCHMARK_VERSION, "e" * 64, "local_permutation", 0.05, 0)
    got = apply_corruption(values, scenario="local_permutation", dose=0.05, seed=seed)

    # Overlapping windows would re-shuffle already-shuffled values, letting an observation
    # drift further than one window and making the affected count meaningless.
    moved = np.flatnonzero(got != values)
    for index in moved:
        displacement = abs(int(got[index]) - int(index))
        assert displacement < LOCAL_PERMUTATION_WINDOW

    # Disjoint tiles => every touched tile is fully inside one window boundary.
    touched_tiles = {int(index) // LOCAL_PERMUTATION_WINDOW for index in moved}
    assert len(moved) <= len(touched_tiles) * LOCAL_PERMUTATION_WINDOW


def test_every_grid_cell_is_realizable_on_a_plain_series():
    values = np.linspace(1.0, 2.0, 400)
    for scenario, dose in CORRUPTION_GRID:
        for replicate in replicates_for(scenario):
            seed = corruption_seed(BENCHMARK_VERSION, "f" * 64, scenario, dose, replicate)
            got = apply_corruption(values, scenario=scenario, dose=dose, seed=seed)
            assert got.shape == values.shape
            assert not np.isinf(got).any()


def test_corruption_is_reorder_and_subset_invariant():
    values = {
        "a": np.linspace(0.0, 1.0, 240),
        "b": np.linspace(1.0, 2.0, 240),
        "c": np.linspace(2.0, 3.0, 240),
    }
    hashes = {
        uid: hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()
        for uid, x in values.items()
    }
    full = materialize_corruptions(values, hashes, "block", 0.12, 0, BENCHMARK_VERSION)
    reversed_values = dict(reversed(list(values.items())))
    reversed_hashes = {uid: hashes[uid] for uid in reversed_values}
    rev = materialize_corruptions(
        reversed_values, reversed_hashes, "block", 0.12, 0, BENCHMARK_VERSION
    )
    sub = materialize_corruptions(
        {"b": values["b"]}, {"b": hashes["b"]}, "block", 0.12, 0, BENCHMARK_VERSION
    )
    assert np.array_equal(full["b"], rev["b"], equal_nan=True)
    assert np.array_equal(full["b"], sub["b"], equal_nan=True)


def test_seed_uses_all_coordinates_and_corruption_is_deterministic():
    base = corruption_seed(BENCHMARK_VERSION, "a" * 64, "block", 0.12, 0)
    assert base == corruption_seed(BENCHMARK_VERSION, "a" * 64, "block", 0.12, 0)
    assert base != corruption_seed(BENCHMARK_VERSION, "a" * 64, "block", 0.12, 1)
    assert base != corruption_seed(BENCHMARK_VERSION, "a" * 64, "block", 0.24, 0)
    values = np.arange(100.0)
    got = apply_corruption(values, scenario="scattered", dose=0.12, seed=base)
    assert np.isnan(got).sum() == 12
    assert not np.isnan(values).any()


def test_corruption_rejects_unknown_grid_or_invalid_identity():
    with pytest.raises(ValueError, match="frozen corruption grid"):
        apply_corruption(np.arange(100.0), scenario="block", dose=0.13, seed=1)
    with pytest.raises(ValueError, match="benchmark version"):
        corruption_seed("benchmark-vX", "a" * 64, "block", 0.12, 0)
    with pytest.raises(ValueError, match="content_sha"):
        corruption_seed(BENCHMARK_VERSION, "bad", "block", 0.12, 0)
    with pytest.raises(ValueError, match="uid sets"):
        materialize_corruptions(
            {"a": np.arange(100.0)}, {}, "block", 0.12, 0, BENCHMARK_VERSION
        )
