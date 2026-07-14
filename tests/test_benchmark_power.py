"""Guards for the detectability panel and the cluster bootstrap."""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.aggregate import AggregationContractError, bootstrap_ci90
from SelfEvolvingHarnessTS.benchmark.power import cluster_bootstrap_ci90, power_panel


def test_clustered_interval_is_wider_than_the_iid_one_it_replaces():
    """The METR-LA correction, stated as a number.

    Sensors inside a spatial block share a stretch of freeway, so their gains move together.
    An IID bootstrap over 200 series treats that as 200 independent draws and reports an
    interval roughly sqrt(n_per_cluster) too narrow -- which is exactly how a cell that
    cannot resolve anything ends up reporting a significant effect.
    """
    rng = np.random.default_rng(7)
    gains: dict[str, float] = {}
    cluster_of_uid: dict[str, str] = {}
    for block in range(20):
        block_effect = rng.normal(0.0, 1.0)  # the whole block moves together
        for sensor in range(10):
            uid = f"s{block}_{sensor}"
            gains[uid] = block_effect + rng.normal(0.0, 0.05)
            cluster_of_uid[uid] = f"block{block}"

    iid_low, iid_high = bootstrap_ci90(gains)
    low, high, se, n_clusters = cluster_bootstrap_ci90(gains, cluster_of_uid)

    assert n_clusters == 20
    assert (high - low) > 3.0 * (iid_high - iid_low)
    assert se > 0


def test_singleton_clusters_reduce_to_the_iid_bootstrap():
    # Most datasets have one series per overlap group. The clustered estimator must not
    # change their numbers, or adopting it everywhere would silently move old results.
    rng = np.random.default_rng(3)
    gains = {f"u{i}": float(rng.normal(0.5, 1.0)) for i in range(80)}
    cluster_of_uid = {uid: uid for uid in gains}

    low, high, se, n_clusters = cluster_bootstrap_ci90(gains, cluster_of_uid, b=4000)
    iid_low, iid_high = bootstrap_ci90(gains, b=4000)

    assert n_clusters == 80
    assert low == pytest.approx(iid_low, abs=0.05)
    assert high == pytest.approx(iid_high, abs=0.05)


def test_a_cell_that_cannot_resolve_a_material_effect_is_marked_unavailable():
    # Three noisy series. The point estimate is near zero, but the cell could not have seen
    # a material effect either -- so it must not be counted as evidence of saturation.
    gains = {"a": 0.30, "b": -0.28, "c": 0.02}
    clusters = {uid: uid for uid in gains}
    panel = power_panel({"thin|cell": gains}, clusters, material_threshold=0.02)
    row = panel["rows"][0]

    assert row["n_series"] == 3
    assert row["mde_80"] > 0.02
    assert row["diagnostic_unavailable"] is not None
    assert "mde_80" in row["diagnostic_unavailable"]
    assert panel["n_readable"] == 0
    assert panel["readable_and_material"] == []


def test_a_dead_pool_cell_is_never_read_as_saturated():
    # Every program scored identically, so every paired gain is exactly zero. Without the
    # flag this reads as "nothing to gain here"; with it, it reads as "the pool has no
    # action for this defect", which is the true statement.
    gains = {f"u{i}": 0.0 for i in range(40)}
    clusters = {uid: uid for uid in gains}
    panel = power_panel(
        {"ds|level_shift": gains},
        clusters,
        pool_cannot_act_keys=["ds|level_shift"],
    )
    row = panel["rows"][0]
    assert row["diagnostic_unavailable"].startswith("pool_cannot_act")
    assert panel["n_readable"] == 0


def test_scale_warning_is_appended_not_swallowed():
    rng = np.random.default_rng(11)
    gains = {f"u{i}": float(rng.normal(1.2, 0.05)) for i in range(60)}
    clusters = {uid: uid for uid in gains}
    panel = power_panel({"covid|trend_high": gains}, clusters, scale_warning_keys=["covid|trend_high"])
    row = panel["rows"][0]
    # The effect is huge and perfectly resolvable, but the denominator is suspect, so the
    # cell still must not be reported as a clean finding.
    assert row["is_material"] is True
    assert row["is_detectable"] is True
    assert "seasonal_scale_warning" in row["diagnostic_unavailable"]
    assert panel["readable_and_material"] == []


def test_a_cell_that_detected_its_effect_is_readable_however_coarse_the_instrument():
    """The null guard must not fire in the positive direction.

    A cell with a large, cleanly separated effect is informative even when its mde_80 sits
    well above epsilon: asking whether it could have resolved 0.02 when it just resolved
    1.18 is not a question. Applying the saturation guard both ways makes `is_detectable`
    and `diagnostic_unavailable` contradict each other and discards every cell with a big
    measured effect -- which inverts the verdict.
    """
    rng = np.random.default_rng(5)
    gains = {f"u{i}": float(rng.normal(1.18, 0.9)) for i in range(40)}
    clusters = {uid: uid for uid in gains}
    panel = power_panel({"lane": gains}, clusters, material_threshold=0.02)
    row = panel["rows"][0]

    assert row["mde_80"] > 0.02, "instrument is coarse relative to epsilon -- the point here"
    assert row["is_detectable"] is True
    assert row["is_material"] is True
    assert row["diagnostic_unavailable"] is None
    assert panel["readable_and_material"] == ["lane"]


def test_bootstrap_refuses_a_uid_with_no_cluster():
    with pytest.raises(AggregationContractError, match="needs a cluster"):
        cluster_bootstrap_ci90({"a": 1.0}, {})
