from __future__ import annotations

import ast
import inspect

import numpy as np

from SelfEvolvingHarnessTS.benchmark.probe import probe_registry, probe_series
from SelfEvolvingHarnessTS.benchmark.registry import SeriesRecord


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
    row = SeriesRecord.from_values(
        dataset_id="noaa_global_hourly",
        entity_id="station",
        values=values,
        source_revision="2020",
        license_id="us-public-domain",
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group="noaa:station",
        regime_tag="candidate_u",
    )
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

