from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.probe import probe_registry, probe_series
from SelfEvolvingHarnessTS.benchmark.registry import SeriesRecord
from SelfEvolvingHarnessTS.benchmark.sources import SOURCE_SPECS


def _record(
    values: np.ndarray,
    *,
    entity_id: str = "station",
    timestamps: np.ndarray | None = None,
) -> SeriesRecord:
    spec = SOURCE_SPECS["noaa_global_hourly"]
    return SeriesRecord.from_values(
        dataset_id="noaa_global_hourly",
        entity_id=entity_id,
        values=values,
        source_id=spec.source_id,
        source_asset_sha256="a" * 64,
        source_revision=spec.source_revision,
        license_id=spec.license_id,
        overlap_family=spec.overlap_family,
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group=f"noaa:{entity_id}",
        overlap_status="resolved",
        overlap_evidence_sha256="b" * 64,
        timestamps=timestamps,
    )


def test_probe_returns_structure_without_loss_fields():
    result = probe_series(
        np.sin(np.arange(240) * 2 * np.pi / 24), period=24
    )
    assert set(result) >= {
        "seasonal_strength",
        "trend_strength",
        "spectral_entropy",
        "natural_missing_count",
        "natural_missing_rate",
        "irregular_interval_count",
        "irregular_sampling_rate",
    }
    assert not ({"loss", "utility", "gain"} & set(result))


def test_probe_uses_only_clean_inner_train_and_is_deterministic():
    prefix = np.sin(np.arange(240) * 2 * np.pi / 24)
    a = np.concatenate([prefix, np.full(48, 1_000.0)])
    b = np.concatenate([prefix, np.full(48, -1_000.0)])
    got_a = probe_series(a, period=24, inner_train_end=240)
    got_b = probe_series(b, period=24, inner_train_end=240)
    assert got_a == got_b


def test_probe_registry_derives_frozen_inner_train_boundary_and_rejects_overrides():
    prefix = np.sin(np.arange(240) * 2 * np.pi / 24)
    a = np.concatenate([prefix, np.full(96, 1_000.0)])
    b = np.concatenate([prefix, np.full(96, -1_000.0)])
    row_a = _record(a)
    row_b = _record(b)
    got_a = probe_registry([row_a], {row_a.series_uid: a})[row_a.series_uid]
    got_b = probe_registry([row_b], {row_b.series_uid: b})[row_b.series_uid]
    assert got_a == got_b
    with pytest.raises(ValueError, match="inner_train_end.*uid sets differ"):
        probe_registry(
            [row_a],
            {row_a.series_uid: a},
            inner_train_end_by_uid={},
        )
    with pytest.raises(ValueError, match="frozen inner-train"):
        probe_registry(
            [row_a],
            {row_a.series_uid: a},
            inner_train_end_by_uid={row_a.series_uid: 241},
        )


def test_probe_registry_enforces_exact_auxiliary_uid_sets():
    values = np.sin(np.arange(240) * 2 * np.pi / 24)
    row = _record(values)
    with pytest.raises(ValueError, match="period.*uid sets differ"):
        probe_registry([row], {row.series_uid: values}, period_by_uid={})
    with pytest.raises(ValueError, match="timestamp.*uid sets differ"):
        probe_registry(
            [row],
            {row.series_uid: values},
            timestamps_by_uid={row.series_uid: np.arange(240)},
        )

    timestamped = _record(
        values,
        entity_id="timestamped",
        timestamps=np.arange(240, dtype="timedelta64[h]")
        + np.datetime64("2020-01-01T00"),
    )
    with pytest.raises(ValueError, match="timestamp.*uid sets differ"):
        probe_registry([timestamped], {timestamped.series_uid: values})


def test_probe_rejects_multidimensional_values_and_nonincreasing_timestamps():
    with pytest.raises(ValueError, match="one-dimensional"):
        probe_series(np.ones((2, 120)), period=24)
    for timestamps in (
        np.array(["2020-01-01T00", "2020-01-01T00", "2020-01-01T01"]),
        np.array(["2020-01-01T00", "2019-12-31T23", "2020-01-01T01"]),
    ):
        with pytest.raises(ValueError, match="strictly increasing"):
            probe_series(np.arange(3.0), period=2, timestamps=timestamps)
    with pytest.raises(ValueError, match="one-dimensional"):
        probe_series(
            np.arange(240.0),
            period=24,
            natural_missing_mask=np.zeros((2, 120), dtype=bool),
        )


def test_probe_preserves_noaa_natural_missing_and_irregular_diagnostics():
    timestamps = np.arange(240, dtype="timedelta64[h]") + np.datetime64(
        "2020-01-01T00"
    )
    timestamps[120:] += np.timedelta64(1, "h")
    values = np.sin(np.arange(240) * 2 * np.pi / 24)
    values[[5, 30, 100]] = np.nan
    got = probe_series(values, period=24, timestamps=timestamps)
    assert got["natural_missing_count"] == 3
    assert got["natural_missing_rate"] == 3 / 240
    assert got["irregular_interval_count"] == 1
    assert got["irregular_sampling_rate"] == 1 / 239
    assert np.isfinite(got["seasonal_strength"])


def test_probe_registry_keeps_every_uid_and_contains_no_outcome_fields():
    values = np.sin(np.arange(240) * 2 * np.pi / 24)
    row = _record(values)
    report = probe_registry([row], {row.series_uid: values})
    assert set(report) == {row.series_uid}
    assert not ({"loss", "utility", "gain"} & set(report[row.series_uid]))


def test_probe_module_has_read_only_import_boundary():
    import SelfEvolvingHarnessTS.benchmark.probe as module

    tree = ast.parse(inspect.getsource(module))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(
        name.endswith(("trainers", "metrics", "baselines")) for name in imported
    )
