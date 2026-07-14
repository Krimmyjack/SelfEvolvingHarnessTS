"""Cluster bootstrap, effect size, and detectability -- the panel that says whether a cell
can be read at all.

v0.1 reported 17 cells and read a number off every one of them.  Most of those numbers were
noise: Dev-Query's 299 series spread across 17 cells left many with a handful of members, and
nothing in the report said so.  A benchmark that reports an unreadable cell as if it were a
finding is a benchmark that will eventually turn a null into a discovery.

Two corrections live here, and they are protocol, not presentation:

**The resampling unit is the overlap group, not the series.**  METR-LA's 207 sensors are not
207 independent draws -- they are 20 spatial blocks, and sensors inside a block sit on the
same stretch of freeway.  Resampling series IID pretends the sample is ten times larger than
it is and shrinks every confidence interval accordingly.  The cluster bootstrap draws whole
groups with replacement.  For datasets whose overlap group is the series itself this reduces
exactly to the IID bootstrap, so it is safe to apply everywhere.

The honesty limit is disclosed rather than hidden: `monash:traffic_hourly`'s 862 sensors are
one Bay Area road network, and the pinned Monash release ships no sensor coordinates, so its
groups are singletons and its intervals remain optimistic.  METR-LA is the spatially clean
traffic read; the two are cross-checks on each other, not independent confirmations.

**A cell that cannot resolve the effect says so.**  `mde_80` is the smallest true effect this
cell could detect at conventional power. When it exceeds the material threshold, "no gain
here" is not evidence of saturation -- the instrument simply cannot see a gain that size, and
the cell is marked `diagnostic_unavailable`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from . import BOOTSTRAP_B, BOOTSTRAP_MASTER_SEED, SATURATION_GAP
from .aggregate import AggregationContractError, bootstrap_subseed

__all__ = [
    "PowerRow",
    "cluster_bootstrap_ci90",
    "power_panel",
]

# Two-sided alpha=0.05, power=0.80: z_{0.975} + z_{0.80} = 1.959964 + 0.841621.
_MDE_Z = 2.801585


def cluster_bootstrap_ci90(
    paired_gain: Mapping[str, float],
    cluster_of_uid: Mapping[str, str],
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_MASTER_SEED,
) -> tuple[float, float, float, int]:
    """Resample overlap groups with replacement. Returns (low, high, se, n_clusters).

    Every member of a drawn group comes with it, and a group drawn twice contributes twice.
    That is what makes the interval reflect the number of *independent* units rather than
    the number of rows.
    """
    if not paired_gain:
        raise AggregationContractError("cluster bootstrap requires at least one uid")
    missing = sorted(set(paired_gain) - set(cluster_of_uid))
    if missing:
        raise AggregationContractError(
            f"every uid needs a cluster; {len(missing)} missing, e.g. {missing[:3]}"
        )
    if not isinstance(b, int) or isinstance(b, bool) or b < 1:
        raise ValueError("bootstrap B must be a positive integer")

    by_cluster: dict[str, list[float]] = {}
    for uid, value in sorted(paired_gain.items()):
        if not np.isfinite(value):
            raise AggregationContractError("paired gains must be finite")
        by_cluster.setdefault(cluster_of_uid[uid], []).append(float(value))

    clusters = sorted(by_cluster)
    n_clusters = len(clusters)
    # The mean is series-equal within the cell, so a resample must carry each drawn group's
    # sum and count together rather than averaging the group first.
    sums = np.asarray([float(np.sum(by_cluster[key])) for key in clusters])
    counts = np.asarray([len(by_cluster[key]) for key in clusters], dtype=np.float64)

    if n_clusters < 2:
        only = float(sums.sum() / counts.sum())
        return only, only, float("inf"), n_clusters

    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, n_clusters, size=(int(b), n_clusters))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.quantile(means, [0.05, 0.95], method="linear")
    return float(low), float(high), float(np.std(means, ddof=1)), n_clusters


@dataclass(frozen=True)
class PowerRow:
    key: str
    n_series: int
    n_clusters: int
    effect: float
    ci90_low: float
    ci90_high: float
    standard_error: float
    mde_80: float
    material_threshold: float
    material_uid_fraction: float
    is_material: bool
    is_detectable: bool
    diagnostic_unavailable: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def power_panel(
    paired_gain_by_key: Mapping[str, Mapping[str, float]],
    cluster_of_uid: Mapping[str, str],
    *,
    material_threshold: float = SATURATION_GAP,
    scale_warning_keys: Sequence[str] = (),
    pool_cannot_act_keys: Sequence[str] = (),
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_MASTER_SEED,
) -> dict[str, object]:
    """Build the detectability panel for every reporting key (cell, or dataset x scenario).

    `diagnostic_unavailable` is the load-bearing column.  A cell carrying a reason there
    contributes nothing to a saturation claim: the instrument could not have seen the effect
    even if it were there.  Ranking cells by effect size while quietly including unreadable
    ones is how a benchmark manufactures a result.
    """
    scale_warned = set(scale_warning_keys)
    dead_pool = set(pool_cannot_act_keys)
    rows: list[PowerRow] = []

    for key in sorted(paired_gain_by_key):
        gains = dict(paired_gain_by_key[key])
        if not gains:
            continue
        low, high, se, n_clusters = cluster_bootstrap_ci90(
            gains,
            cluster_of_uid,
            b=b,
            seed=bootstrap_subseed(seed, "power_panel", key),
        )
        effect = float(np.mean(list(gains.values())))
        mde = float(_MDE_Z * se) if np.isfinite(se) else float("inf")
        material_fraction = float(
            np.mean([1.0 if value > material_threshold else 0.0 for value in gains.values()])
        )

        detected = bool(np.isfinite(mde) and abs(effect) > mde)

        # `diagnostic_unavailable` guards the NULL direction only, which is what the
        # preregistration says it is for: "absence of a detected effect is not evidence of
        # absence when mde_80 > epsilon". A cell that DID detect its effect is informative
        # regardless of how small an effect it could not have seen -- asking whether it
        # could have resolved 0.02 when it resolved 1.18 is not a question. Applying the
        # rule in both directions makes `detectable` and `readable` contradict each other,
        # and would throw away every cell with a large, cleanly measured effect.
        reason: str | None = None
        if key in dead_pool:
            reason = "pool_cannot_act -- every program scores identically on this defect"
        elif n_clusters < 2:
            reason = "n_clusters<2 -- a single independent unit admits no interval"
        elif not detected and (not np.isfinite(mde) or mde > material_threshold):
            reason = (
                f"no effect detected (|{effect:.4f}| <= mde_80={mde:.4f}), and mde_80 "
                f"exceeds the material threshold {material_threshold:.4f} -- a material "
                "effect could be present and this cell could not have seen it, so this "
                "cell cannot support a saturation claim either way"
            )
        if key in scale_warned:
            note = "seasonal_scale_warning -- sMASE denominator is tiny, magnitudes inflated"
            reason = note if reason is None else f"{reason}; {note}"

        rows.append(
            PowerRow(
                key=key,
                n_series=len(gains),
                n_clusters=n_clusters,
                effect=effect,
                ci90_low=low,
                ci90_high=high,
                standard_error=se,
                mde_80=mde,
                material_threshold=float(material_threshold),
                material_uid_fraction=material_fraction,
                is_material=bool(effect > material_threshold),
                is_detectable=detected,
                diagnostic_unavailable=reason,
            )
        )

    readable = [row for row in rows if row.diagnostic_unavailable is None]
    return {
        "definitions": {
            "resampling_unit": (
                "overlap_group, not series. METR-LA's 207 sensors are 20 spatial blocks; "
                "treating them as 207 independent draws shrinks every interval by roughly "
                "sqrt(10)."
            ),
            "mde_80": (
                "smallest true effect detectable at alpha=0.05 two-sided with power 0.80, "
                "computed as 2.8016 * the cluster-bootstrap standard error"
            ),
            "material_threshold": (
                "the frozen SATURATION_GAP epsilon. A gain below it is not worth an "
                "expensive method, whether or not it is statistically resolvable."
            ),
            "diagnostic_unavailable": (
                "a reason here means the cell cannot support a saturation claim -- absence "
                "of a detected effect is not evidence of absence when mde_80 > epsilon"
            ),
            "known_optimism": (
                "monash:traffic_hourly has singleton clusters because the pinned Monash "
                "release ships no sensor coordinates, though its 862 sensors are one road "
                "network. Its intervals are an optimistic bound; METR-LA is the spatially "
                "clean traffic read."
            ),
        },
        "rows": [row.to_dict() for row in rows],
        "n_keys": len(rows),
        "n_readable": len(readable),
        "n_diagnostic_unavailable": len(rows) - len(readable),
        "readable_and_material": sorted(
            row.key for row in readable if row.is_material and row.is_detectable
        ),
    }
