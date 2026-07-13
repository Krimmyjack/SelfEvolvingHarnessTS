from __future__ import annotations

import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest

import SelfEvolvingHarnessTS.benchmark.materialize as materialize_module

from SelfEvolvingHarnessTS.benchmark.materialize import (
    ParsedSeries,
    RawMutationError,
    materialize_clean_base,
    parse_metr_la_hdf,
    parse_monash_parquet,
    parse_noaa_global_hourly,
    parse_uci_electricity_zip,
    promote_download,
    read_clean_base,
    resample_hourly,
    select_benchmark_span,
    verify_raw_asset,
    write_raw_once,
)
from SelfEvolvingHarnessTS.benchmark.sources import SOURCE_SPECS


def test_raw_write_is_immutable_and_verifiable(tmp_path):
    asset = write_raw_once(tmp_path / "raw.bin", b"abc", source_revision="r1")
    assert asset.sha256 == hashlib.sha256(b"abc").hexdigest()
    assert verify_raw_asset(asset) == asset
    assert write_raw_once(tmp_path / "raw.bin", b"abc", source_revision="r1") == asset
    with pytest.raises(RawMutationError):
        write_raw_once(tmp_path / "raw.bin", b"abd", source_revision="r1")
    with pytest.raises(RawMutationError, match="revision"):
        write_raw_once(tmp_path / "raw.bin", b"abc", source_revision="r2")


def test_hourly_mean_keeps_raw_missing_mask():
    index = pd.date_range("2020-01-01", periods=8, freq="15min", tz="UTC")
    values = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 7.0, 9.0, 11.0], index=index)
    out, mask = resample_hourly(values)
    assert out.tolist() == [pytest.approx(8 / 3), 8.0]
    assert mask.tolist() == [True, False]
    assert str(out.index.tz) == "UTC"


def test_download_promote_checks_hash_before_atomic_raw_write(tmp_path):
    part = tmp_path / "asset.part"
    part.write_bytes(b"downloaded")
    expected = hashlib.sha256(b"downloaded").hexdigest()
    with pytest.raises(RawMutationError, match="SHA256 mismatch"):
        promote_download(
            part,
            tmp_path / "raw" / "asset.bin",
            source_revision="rev",
            expected_sha256="0" * 64,
        )
    assert part.exists()
    asset = promote_download(
        part,
        tmp_path / "raw" / "asset.bin",
        source_revision="rev",
        expected_sha256=expected,
    )
    assert not part.exists()
    assert verify_raw_asset(asset).sha256 == expected


def test_hourly_resample_rejects_naive_or_nonmonotonic_time():
    naive = pd.Series([1.0, 2.0], index=pd.date_range("2020", periods=2, freq="h"))
    with pytest.raises(ValueError, match="timezone"):
        resample_hourly(naive)
    reversed_series = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2020-01-01T01:00Z", "2020-01-01T00:00Z"]),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        resample_hourly(reversed_series)


def test_clean_base_writes_values_timestamps_and_mask_separately(tmp_path):
    spec = SOURCE_SPECS["noaa_global_hourly"]
    raw = write_raw_once(tmp_path / "raw" / "station.csv", b"source", source_revision=spec.source_revision)
    timestamps = np.arange(240, dtype="timedelta64[h]") + np.datetime64("2020-01-01T00")
    values = np.sin(np.arange(240) * 2 * np.pi / 24)
    values[[3, 10]] = np.nan
    materialized = materialize_clean_base(
        tmp_path / "clean_base",
        dataset_id="noaa_global_hourly",
        entity_id="station",
        values=values,
        timestamps=timestamps,
        source_id=spec.source_id,
        source_asset_sha256=raw.sha256,
        source_revision=spec.source_revision,
        license_id=spec.license_id,
        overlap_family=spec.overlap_family,
        exposure_class="certified_virgin",
        frequency="hourly",
        overlap_group="noaa:station",
        overlap_status="resolved",
        overlap_evidence_sha256="b" * 64,
    )
    assert materialized.values_path != materialized.mask_path
    assert materialized.timestamps_path is not None
    loaded_values, loaded_timestamps, loaded_mask = read_clean_base(materialized)
    assert np.array_equal(loaded_values, values, equal_nan=True)
    assert np.array_equal(loaded_timestamps, timestamps.astype("datetime64[ns]"))
    assert loaded_mask.tolist() == np.isnan(values).tolist()
    materialized.record.verify_values(loaded_values, timestamps=loaded_timestamps)

    with pytest.raises(RawMutationError):
        materialize_clean_base(
            tmp_path / "clean_base",
            dataset_id="noaa_global_hourly",
            entity_id="station",
            values=values + 1.0,
            timestamps=timestamps,
            source_id=spec.source_id,
            source_asset_sha256=raw.sha256,
            source_revision=spec.source_revision,
            license_id=spec.license_id,
            overlap_family=spec.overlap_family,
            exposure_class="certified_virgin",
            frequency="hourly",
            overlap_group="noaa:station",
            overlap_status="resolved",
            overlap_evidence_sha256="b" * 64,
        )


def test_span_selection_keeps_latest_finite_future_and_natural_missingness():
    values = np.arange(12, dtype=float)
    values[9] = np.nan
    timestamps = np.arange(12, dtype="timedelta64[h]") + np.datetime64("2024-01-01")
    selected_values, selected_timestamps = select_benchmark_span(
        values,
        timestamps,
        horizon=2,
        min_length=6,
        max_length=8,
    )
    assert selected_values.tolist()[:3] == [4.0, 5.0, 6.0]
    assert np.isnan(selected_values[-3])
    assert selected_values[-2:].tolist() == [10.0, 11.0]
    assert len(selected_timestamps) == 8


def test_uci_zip_parser_resamples_clients_hourly_and_preserves_missing_bins(tmp_path):
    archive = tmp_path / "eld.zip"
    body = (
        "\"\";\"MT_001\"\n"
        "2014-01-01 00:00:00;1,0\n"
        "2014-01-01 00:15:00;\n"
        "2014-01-01 00:30:00;3,0\n"
        "2014-01-01 00:45:00;4,0\n"
        "2014-01-01 01:00:00;5,0\n"
        "2014-01-01 01:15:00;6,0\n"
        "2014-01-01 01:30:00;7,0\n"
        "2014-01-01 01:45:00;8,0\n"
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("LD2011_2014.txt", body)
    rows = parse_uci_electricity_zip(archive, min_length=2, horizon=1, max_length=8)
    assert len(rows) == 1
    assert isinstance(rows[0], ParsedSeries)
    assert rows[0].entity_id == "MT_001"
    assert rows[0].frequency == "hourly"
    assert np.isnan(rows[0].values[0])
    assert rows[0].natural_missing_mask.tolist() == [True, False]


def test_noaa_parser_decodes_tenths_and_materializes_empty_hours(tmp_path):
    path = tmp_path / "station.csv"
    path.write_text(
        "STATION,DATE,TMP\n"
        '01234599999,2024-01-01T00:10:00,"+0123,1"\n'
        '01234599999,2024-01-01T02:20:00,"+0200,1"\n',
        encoding="utf-8",
    )
    row = parse_noaa_global_hourly(path, min_length=3, horizon=1, max_length=8)
    assert row.entity_id == "01234599999"
    assert row.values.tolist()[:1] == [pytest.approx(12.3)]
    assert row.natural_missing_mask.tolist() == [False, True, False]


def test_monash_parquet_parser_uses_frozen_config_frequency(tmp_path):
    path = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "item_id": ["series-1"],
            "start": [pd.Timestamp("2024-01-01")],
            "target": [[1.0, np.nan, 3.0, 4.0, 5.0, 6.0]],
        }
    ).to_parquet(path)
    rows = parse_monash_parquet(
        path,
        config="nn5_daily",
        min_length=5,
        horizon=2,
        max_length=8,
    )
    assert len(rows) == 1
    assert rows[0].entity_id == "series-1"
    assert rows[0].frequency == "daily"
    assert np.isnan(rows[0].values[1])
    assert np.all(np.diff(rows[0].timestamps) == np.timedelta64(1, "D"))


def test_metr_hdf_parser_resamples_each_sensor_without_pooling(tmp_path):
    path = tmp_path / "metr.h5"
    index = pd.date_range("2024-01-01", periods=12, freq="5min")
    pd.DataFrame(
        {"sensor-a": np.arange(12, dtype=float), "sensor-b": np.arange(12, dtype=float) + 10},
        index=index,
    ).to_hdf(path, key="df")
    rows = parse_metr_la_hdf(path, min_length=1, horizon=1, max_length=8)
    assert [row.entity_id for row in rows] == ["sensor-a", "sensor-b"]
    assert rows[0].frequency == "hourly"
    assert rows[0].values.tolist() == [pytest.approx(5.5)]


def test_source_timezone_localization_freezes_dst_ambiguity_to_standard_time():
    local = pd.DatetimeIndex(["2011-10-30 00:45", "2011-10-30 01:00", "2011-10-30 01:15"])
    utc = materialize_module._localize_index(local, "Europe/Lisbon")
    assert str(utc.tz) == "UTC"
    assert utc.is_monotonic_increasing
