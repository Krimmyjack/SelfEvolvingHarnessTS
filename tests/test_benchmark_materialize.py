from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from SelfEvolvingHarnessTS.benchmark.materialize import (
    RawMutationError,
    materialize_clean_base,
    promote_download,
    read_clean_base,
    resample_hourly,
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
