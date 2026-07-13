from __future__ import annotations

import hashlib

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark import BENCHMARK_VERSION
from SelfEvolvingHarnessTS.benchmark.corruption import (
    CORRUPTION_GRID,
    apply_corruption,
    corruption_seed,
    materialize_corruptions,
)


def test_frozen_v0_corruption_grid():
    assert CORRUPTION_GRID == (
        ("block", 0.12),
        ("block", 0.24),
        ("scattered", 0.12),
    )


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
