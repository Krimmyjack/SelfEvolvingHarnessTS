"""Deterministic geographic blocking for sensor-network sources.

A road-network source such as METR-LA is 207 sensors on one freeway graph, not 207
independent series.  Sensors a mile apart see the same cars; if one lands in Support
and its neighbour in Final-Query, the Final number is measuring memorization of a
neighbour rather than generalization.  Splitting per sensor therefore leaks.

The atomic split unit is instead a *spatial block*, built in two stages:

1. **Site merge.**  METR-LA's sensor geometry is not uniform: a quarter of its sensors
   sit within 27 m of another sensor -- these are co-located detectors (opposite
   directions at one milepost), effectively duplicate measurements of one location.
   Single-linkage clustering at `SITE_MERGE_RADIUS_KM` collapses them into one *site*.
   The radius is chosen to capture co-location without chaining along a freeway: on
   METR-LA, 200 m yields sites of at most 4 sensors, whereas 1 km chains 54 sensors
   into one blob.  A duplicate pair can therefore never straddle a role boundary.

2. **Block.**  Recursive median bisection (a KD-tree) over the site centroids groups
   sites into geographically compact blocks.  No RNG, no data, no free parameters
   beyond the block count, so the blocking is a pure function of the pinned coordinate
   file and is reproducible from it.

The block is what the split treats as atomic, so an entire block always travels to the
same role.

Two honest limits, both reported in the data card rather than hidden:

  * Blocking is a mitigation, not a proof.  Sites on opposite sides of a block
    boundary are still neighbours, so residual cross-role correlation survives.  The
    manifest reports the minimum cross-block distance so the size of that residual is
    on the record.
  * Blocking bounds the claim to "unseen block within THIS road network".  It does
    not license a cross-network claim.  Cross-network generalization has to come from
    holding two independent networks (METR-LA in Los Angeles, Monash traffic_hourly
    in the San Francisco Bay Area) and testing each against the other.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "SITE_MERGE_RADIUS_KM",
    "SpatialBlocking",
    "block_diagnostics",
    "build_spatial_blocks",
    "haversine_km",
    "merge_colocated_sites",
]

EARTH_RADIUS_KM = 6371.0088

SITE_MERGE_RADIUS_KM = 0.2


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (latitude, longitude) points."""

    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


class SpatialBlocking(dict):
    """Mapping entity_id -> block index, plus the parameters that produced it."""

    def __init__(
        self,
        assignment: Mapping[str, int],
        *,
        n_blocks: int,
        sites: Mapping[str, int],
        site_merge_radius_km: float,
    ) -> None:
        super().__init__(sorted(assignment.items()))
        self.n_blocks = int(n_blocks)
        self.sites = dict(sorted(sites.items()))
        self.site_merge_radius_km = float(site_merge_radius_km)

    @property
    def n_sites(self) -> int:
        return len(set(self.sites.values()))


def merge_colocated_sites(
    coordinates: Mapping[str, tuple[float, float]],
    *,
    radius_km: float = SITE_MERGE_RADIUS_KM,
) -> dict[str, int]:
    """Single-linkage merge of sensors within `radius_km` into physical sites.

    Co-located detectors measure one location twice; treating them as independent
    series would let a near-duplicate straddle Support and Query.
    """
    if not isinstance(coordinates, Mapping) or not coordinates:
        raise ValueError("coordinates must be a non-empty mapping")
    if not math.isfinite(radius_km) or radius_km < 0:
        raise ValueError("site merge radius must be finite and non-negative")

    entities = sorted(coordinates)
    parent = {entity: entity for entity in entities}

    def find(entity: str) -> str:
        while parent[entity] != entity:
            parent[entity] = parent[parent[entity]]
            entity = parent[entity]
        return entity

    for i, left in enumerate(entities):
        for right in entities[i + 1 :]:
            if haversine_km(coordinates[left], coordinates[right]) <= radius_km:
                root_left, root_right = find(left), find(right)
                if root_left != root_right:
                    # Union by canonical name keeps the result order-independent.
                    low, high = sorted((root_left, root_right))
                    parent[high] = low

    roots = sorted({find(entity) for entity in entities})
    index_of_root = {root: index for index, root in enumerate(roots)}
    return {entity: index_of_root[find(entity)] for entity in entities}


def _projected(
    coordinates: Mapping[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """Project degrees to an approximately isotropic local km frame.

    Longitude degrees shrink with latitude; without the cos(lat) factor the KD-tree
    would think the map is wider than it is and would prefer the wrong split axis.
    """
    mean_lat = sum(lat for lat, _ in coordinates.values()) / len(coordinates)
    lon_scale = math.cos(math.radians(mean_lat))
    return {
        entity: (lat * 111.32, lon * 111.32 * lon_scale)
        for entity, (lat, lon) in coordinates.items()
    }


def _bisect(
    entities: Sequence[str],
    projected: Mapping[str, tuple[float, float]],
    n_blocks: int,
    assignment: dict[str, int],
    next_index: list[int],
) -> None:
    if n_blocks <= 1 or len(entities) <= 1:
        block = next_index[0]
        next_index[0] += 1
        for entity in entities:
            assignment[entity] = block
        return

    ys = [projected[entity][0] for entity in entities]
    xs = [projected[entity][1] for entity in entities]
    axis = 0 if (max(ys) - min(ys)) >= (max(xs) - min(xs)) else 1
    # entity_id breaks coordinate ties, so the order never depends on input order.
    ordered = sorted(entities, key=lambda entity: (projected[entity][axis], entity))

    left_blocks = n_blocks // 2
    right_blocks = n_blocks - left_blocks
    cut = round(len(ordered) * left_blocks / n_blocks)
    cut = max(left_blocks, min(cut, len(ordered) - right_blocks))

    _bisect(ordered[:cut], projected, left_blocks, assignment, next_index)
    _bisect(ordered[cut:], projected, right_blocks, assignment, next_index)


def build_spatial_blocks(
    coordinates: Mapping[str, tuple[float, float]],
    *,
    n_blocks: int,
    site_merge_radius_km: float = SITE_MERGE_RADIUS_KM,
) -> SpatialBlocking:
    """Merge co-located sensors into sites, then bisect the sites into compact blocks."""

    if not isinstance(coordinates, Mapping) or not coordinates:
        raise ValueError("coordinates must be a non-empty mapping")
    if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or n_blocks < 1:
        raise ValueError("n_blocks must be a positive integer")
    for entity, point in coordinates.items():
        if not isinstance(entity, str) or not entity or entity != entity.strip():
            raise ValueError("sensor identifiers must be canonical non-empty strings")
        if len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
            raise ValueError(f"sensor {entity!r} has a non-finite coordinate")
        latitude, longitude = float(point[0]), float(point[1])
        if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
            raise ValueError(f"sensor {entity!r} is not on the globe")

    sites = merge_colocated_sites(coordinates, radius_km=site_merge_radius_km)
    site_members: dict[int, list[str]] = {}
    for entity, site in sites.items():
        site_members.setdefault(site, []).append(entity)
    if n_blocks > len(site_members):
        raise ValueError(
            f"n_blocks ({n_blocks}) cannot exceed the number of merged sites "
            f"({len(site_members)})"
        )

    # Bisect over site centroids so a site is indivisible by construction.
    centroids = {
        f"site_{site:04d}": (
            sum(coordinates[entity][0] for entity in members) / len(members),
            sum(coordinates[entity][1] for entity in members) / len(members),
        )
        for site, members in site_members.items()
    }
    projected = _projected(centroids)
    site_blocks: dict[str, int] = {}
    _bisect(sorted(centroids), projected, n_blocks, site_blocks, [0])
    if len(site_blocks) != len(centroids):
        raise RuntimeError("spatial bisection lost sites")
    if len(set(site_blocks.values())) != n_blocks:
        raise RuntimeError("spatial bisection did not produce the requested block count")

    assignment = {
        entity: site_blocks[f"site_{site:04d}"] for entity, site in sites.items()
    }
    return SpatialBlocking(
        assignment,
        n_blocks=n_blocks,
        sites=sites,
        site_merge_radius_km=site_merge_radius_km,
    )


def block_diagnostics(
    coordinates: Mapping[str, tuple[float, float]], blocking: Mapping[str, int]
) -> dict[str, object]:
    """Report block sizes, within-block spread, and the residual cross-block adjacency."""

    members: dict[int, list[str]] = {}
    for entity, block in blocking.items():
        members.setdefault(int(block), []).append(entity)

    max_within = 0.0
    for group in members.values():
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                max_within = max(
                    max_within, haversine_km(coordinates[left], coordinates[right])
                )

    min_across = math.inf
    entities = sorted(blocking)
    for i, left in enumerate(entities):
        for right in entities[i + 1 :]:
            if blocking[left] == blocking[right]:
                continue
            min_across = min(
                min_across, haversine_km(coordinates[left], coordinates[right])
            )

    sizes = sorted(len(group) for group in members.values())
    diagnostics: dict[str, object] = {
        "n_blocks": len(members),
        "n_sensors": len(blocking),
        "block_size_min": sizes[0],
        "block_size_max": sizes[-1],
        "max_within_block_distance_km": round(max_within, 4),
        "min_cross_block_distance_km": (
            round(min_across, 4) if math.isfinite(min_across) else None
        ),
        "residual_leakage_note": (
            "Sites on opposite sides of a block boundary remain neighbours. "
            "min_cross_block_distance_km is the worst-case residual: blocking bounds "
            "adjacent-sensor leakage, it does not eliminate it."
        ),
    }
    if isinstance(blocking, SpatialBlocking):
        site_sizes = sorted(
            len([entity for entity, site in blocking.sites.items() if site == target])
            for target in set(blocking.sites.values())
        )
        diagnostics["n_sites"] = blocking.n_sites
        diagnostics["site_merge_radius_km"] = blocking.site_merge_radius_km
        diagnostics["site_size_max"] = site_sizes[-1]
        diagnostics["n_sensors_in_merged_sites"] = sum(
            size for size in site_sizes if size > 1
        )
    return diagnostics
