"""Spatial blocking must make co-located sensors inseparable and stay deterministic."""
from __future__ import annotations

import math

import pytest

from SelfEvolvingHarnessTS.benchmark.spatial import (
    SITE_MERGE_RADIUS_KM,
    block_diagnostics,
    build_spatial_blocks,
    haversine_km,
    merge_colocated_sites,
)


def _grid_coordinates(n: int = 40) -> dict[str, tuple[float, float]]:
    """A line of sensors ~1 km apart, each paired with a twin ~20 m away."""
    coordinates: dict[str, tuple[float, float]] = {}
    for index in range(n):
        latitude = 34.0 + index * 0.009
        coordinates[f"s{index:03d}a"] = (latitude, -118.0)
        coordinates[f"s{index:03d}b"] = (latitude + 0.0002, -118.0)
    return coordinates


def test_haversine_matches_known_distance() -> None:
    # One degree of latitude is ~111.2 km anywhere on the globe.
    assert haversine_km((0.0, 0.0), (1.0, 0.0)) == pytest.approx(111.19, abs=0.1)


def test_colocated_twins_merge_into_one_site() -> None:
    coordinates = _grid_coordinates()
    sites = merge_colocated_sites(coordinates, radius_km=SITE_MERGE_RADIUS_KM)
    # Each ~20 m twin pair is one site; the 1 km spacing must NOT chain them together.
    assert len(set(sites.values())) == 40
    for index in range(40):
        assert sites[f"s{index:03d}a"] == sites[f"s{index:03d}b"]


def test_colocated_twins_never_straddle_a_block() -> None:
    coordinates = _grid_coordinates()
    blocking = build_spatial_blocks(coordinates, n_blocks=8)
    for index in range(40):
        assert blocking[f"s{index:03d}a"] == blocking[f"s{index:03d}b"], (
            "a co-located twin pair landed in two different blocks, which is exactly "
            "the near-duplicate leakage blocking exists to prevent"
        )


def test_min_cross_block_distance_is_at_least_the_merge_radius() -> None:
    coordinates = _grid_coordinates()
    blocking = build_spatial_blocks(coordinates, n_blocks=8)
    diagnostics = block_diagnostics(coordinates, blocking)
    assert diagnostics["min_cross_block_distance_km"] >= SITE_MERGE_RADIUS_KM


def test_blocking_is_invariant_to_input_order() -> None:
    coordinates = _grid_coordinates()
    forward = build_spatial_blocks(coordinates, n_blocks=8)
    reversed_input = dict(reversed(list(coordinates.items())))
    backward = build_spatial_blocks(reversed_input, n_blocks=8)
    assert dict(forward) == dict(backward)


def test_every_requested_block_is_populated() -> None:
    coordinates = _grid_coordinates()
    blocking = build_spatial_blocks(coordinates, n_blocks=8)
    assert sorted(set(blocking.values())) == list(range(8))
    assert len(blocking) == len(coordinates)


def test_blocks_cannot_outnumber_merged_sites() -> None:
    coordinates = _grid_coordinates()
    # 40 twin pairs collapse to 40 sites, so 41 blocks is unsatisfiable.
    with pytest.raises(ValueError, match="cannot exceed the number of merged sites"):
        build_spatial_blocks(coordinates, n_blocks=41)


def test_rejects_coordinates_off_the_globe() -> None:
    with pytest.raises(ValueError, match="not on the globe"):
        build_spatial_blocks({"a": (91.0, 0.0), "b": (0.0, 0.0)}, n_blocks=1)


def test_blocks_are_geographically_compact() -> None:
    coordinates = _grid_coordinates()
    blocking = build_spatial_blocks(coordinates, n_blocks=8)
    diagnostics = block_diagnostics(coordinates, blocking)
    # 40 km of sensors cut into 8 blocks: no block may span the whole line.
    assert diagnostics["max_within_block_distance_km"] < 10.0
    assert math.isfinite(diagnostics["min_cross_block_distance_km"])
