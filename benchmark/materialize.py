"""Immutable raw storage and deterministic clean-base materialization."""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .registry import SeriesRecord

__all__ = [
    "CleanBaseAsset",
    "ParsedSeries",
    "RawAsset",
    "RawMutationError",
    "materialize_clean_base",
    "parse_gefcom2012_load_zip",
    "parse_noaa_global_hourly",
    "parse_metr_la_hdf",
    "parse_metr_la_sensor_locations",
    "parse_monash_parquet",
    "parse_uci_electricity_zip",
    "promote_download",
    "read_clean_base",
    "resample_hourly",
    "select_benchmark_span",
    "verify_raw_asset",
    "write_raw_once",
]


class RawMutationError(RuntimeError):
    """An immutable raw or clean-base path disagrees with frozen bytes."""


@dataclass(frozen=True)
class ParsedSeries:
    entity_id: str
    values: np.ndarray
    timestamps: np.ndarray
    natural_missing_mask: np.ndarray
    frequency: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        timestamps = np.asarray(self.timestamps).astype("datetime64[ns]")
        mask = np.asarray(self.natural_missing_mask, dtype=bool)
        if values.ndim != 1 or timestamps.ndim != 1 or mask.ndim != 1:
            raise ValueError("parsed series arrays must be one-dimensional")
        if not (len(values) == len(timestamps) == len(mask)) or not len(values):
            raise ValueError("parsed series arrays must be non-empty and aligned")
        if np.isinf(values).any() or np.isnat(timestamps).any():
            raise ValueError("parsed series cannot contain infinity or NaT")
        if len(timestamps) > 1 and np.any(np.diff(timestamps.astype(np.int64)) <= 0):
            raise ValueError("parsed timestamps must be strictly increasing")
        if not isinstance(self.entity_id, str) or not self.entity_id.strip():
            raise ValueError("parsed entity_id must be non-empty")
        if not isinstance(self.frequency, str) or not self.frequency.strip():
            raise ValueError("parsed frequency must be non-empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "natural_missing_mask", mask)


def select_benchmark_span(
    values: np.ndarray,
    timestamps: np.ndarray,
    *,
    horizon: int,
    min_length: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the latest bounded span whose held-out future is fully observed."""

    array = np.asarray(values, dtype=np.float64)
    times = np.asarray(timestamps).astype("datetime64[ns]")
    if array.ndim != 1 or times.ndim != 1 or len(array) != len(times):
        raise ValueError("values and timestamps must be aligned one-dimensional arrays")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (horizon, min_length, max_length)):
        raise ValueError("span dimensions must be positive integers")
    if min_length < horizon or max_length < min_length:
        raise ValueError("span dimensions are inconsistent")
    for stop in range(len(array), min_length - 1, -1):
        if np.isfinite(array[stop - horizon : stop]).all():
            start = max(0, stop - max_length)
            if stop - start >= min_length:
                return array[start:stop].copy(), times[start:stop].copy()
    raise ValueError("no admissible span has a fully observed test future")


def _localize_index(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    if index.tz is None:
        # The official UCI/METR tables contain no offset bit for isolated DST
        # fallback timestamps. Freeze those to standard time (ambiguous=False).
        index = index.tz_localize(timezone, ambiguous=False, nonexistent="shift_forward")
    return index.tz_convert("UTC")


def parse_uci_electricity_zip(
    path: Path | str,
    *,
    min_length: int,
    horizon: int,
    max_length: int,
) -> tuple[ParsedSeries, ...]:
    """Parse the official UCI ELD archive and conservatively resample to UTC hours."""

    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.lower().endswith(".txt") and not name.startswith("__MACOSX/")
        )
        if len(names) != 1:
            raise ValueError("UCI ELD archive must contain exactly one data text file")
        with archive.open(names[0]) as handle:
            frame = pd.read_csv(handle, sep=";", decimal=",")
    if frame.shape[1] < 2:
        raise ValueError("UCI ELD table has no client columns")
    timestamps = pd.to_datetime(frame.iloc[:, 0], errors="raise")
    index = _localize_index(pd.DatetimeIndex(timestamps), "Europe/Lisbon")
    rows: list[ParsedSeries] = []
    for column in frame.columns[1:]:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        series = pd.Series(numeric.to_numpy(dtype=float), index=index).sort_index()
        if series.index.has_duplicates:
            series = series.groupby(level=0).mean()
        hourly, missing = resample_hourly(series)
        conservative = hourly.to_numpy(dtype=float)
        mask = missing.to_numpy(dtype=bool)
        conservative[mask] = np.nan
        try:
            selected, selected_times = select_benchmark_span(
                conservative,
                hourly.index.to_numpy(dtype="datetime64[ns]"),
                horizon=horizon,
                min_length=min_length,
                max_length=max_length,
            )
        except ValueError:
            continue
        rows.append(
            ParsedSeries(
                entity_id=str(column),
                values=selected,
                timestamps=selected_times,
                natural_missing_mask=np.isnan(selected),
                frequency="hourly",
            )
        )
    return tuple(rows)


def parse_gefcom2012_load_zip(
    path: Path | str,
    *,
    min_length: int,
    horizon: int,
    max_length: int,
) -> tuple[ParsedSeries, ...]:
    """Parse the official GEFCom2012 load track into one hourly series per zone.

    The competition history deliberately blanks evaluation blocks.  Those cells
    are restored only from the official ``Load_solution.csv`` ground truth; any
    other missing observations remain natural NaNs.  The source does not encode
    a timezone, so its documented calendar/hour convention is frozen as UTC,
    with h1=01:00 and h24=00:00 on the following day.
    """

    hour_columns = [f"h{hour}" for hour in range(1, 25)]
    key_columns = ["zone_id", "year", "month", "day"]
    with zipfile.ZipFile(path) as archive:
        history_names = [
            name for name in archive.namelist()
            if name.replace("\\", "/").endswith("/Load/Load_history.csv")
        ]
        solution_names = [
            name for name in archive.namelist()
            if name.replace("\\", "/").endswith("/Load/Load_solution.csv")
        ]
        if len(history_names) != 1 or len(solution_names) != 1:
            raise ValueError(
                "GEFCom2012 archive must contain one Load_history.csv and one Load_solution.csv"
            )
        history = pd.read_csv(archive.open(history_names[0]), thousands=",")
        solution = pd.read_csv(archive.open(solution_names[0]), thousands=",")

    required = set(key_columns + hour_columns)
    if not required <= set(history.columns) or not required <= set(solution.columns):
        raise ValueError("GEFCom2012 load tables lack required zone/date/hour columns")
    history = history[key_columns + hour_columns].copy()
    solution = solution[key_columns + hour_columns].copy()
    for column in key_columns:
        history[column] = pd.to_numeric(history[column], errors="raise").astype(int)
        solution[column] = pd.to_numeric(solution[column], errors="raise").astype(int)
    if history.duplicated(key_columns).any() or solution.duplicated(key_columns).any():
        raise ValueError("GEFCom2012 load tables contain duplicate zone/day keys")

    history_index = pd.MultiIndex.from_frame(history[key_columns])
    solution_indexed = solution.set_index(key_columns)
    for column in hour_columns:
        values = pd.to_numeric(history[column], errors="coerce")
        truth = pd.to_numeric(solution_indexed[column], errors="coerce").reindex(
            history_index
        )
        truth.index = history.index
        history[column] = values.where(values.notna(), truth)

    rows: list[ParsedSeries] = []
    for zone_id, zone in history.groupby("zone_id", sort=True):
        zone = zone.sort_values(["year", "month", "day"])
        days = pd.to_datetime(
            zone[["year", "month", "day"]], errors="raise", utc=True
        )
        timestamps = pd.DatetimeIndex(days.repeat(24)) + pd.to_timedelta(
            np.tile(np.arange(1, 25), len(zone)), unit="h"
        )
        values = zone[hour_columns].to_numpy(dtype=np.float64).reshape(-1)
        series = pd.Series(values, index=timestamps).sort_index()
        if series.index.has_duplicates:
            raise ValueError("GEFCom2012 zone expands to duplicate hourly timestamps")
        complete_index = pd.date_range(
            series.index.min(), series.index.max(), freq="1h", tz="UTC"
        )
        series = series.reindex(complete_index)
        try:
            selected, selected_times = select_benchmark_span(
                series.to_numpy(dtype=np.float64),
                series.index.to_numpy(dtype="datetime64[ns]"),
                horizon=horizon,
                min_length=min_length,
                max_length=max_length,
            )
        except ValueError:
            continue
        rows.append(
            ParsedSeries(
                entity_id=f"zone_{int(zone_id)}",
                values=selected,
                timestamps=selected_times,
                natural_missing_mask=np.isnan(selected),
                frequency="hourly",
            )
        )
    return tuple(rows)


def parse_noaa_global_hourly(
    path: Path | str,
    *,
    min_length: int,
    horizon: int,
    max_length: int,
) -> ParsedSeries:
    """Decode NOAA Global Hourly TMP tenths-Celsius observations into UTC hours."""

    frame = pd.read_csv(path, dtype={"STATION": str, "TMP": str})
    if not {"STATION", "DATE", "TMP"} <= set(frame.columns) or frame.empty:
        raise ValueError("NOAA file lacks STATION, DATE, or TMP")
    stations = frame["STATION"].dropna().astype(str).unique()
    if len(stations) != 1:
        raise ValueError("NOAA file must contain exactly one station")
    index = pd.DatetimeIndex(pd.to_datetime(frame["DATE"], utc=True, errors="raise"))
    decoded: list[float] = []
    for raw in frame["TMP"].astype(str):
        token = raw.split(",", 1)[0].strip()
        try:
            value = int(token)
        except ValueError:
            value = 9999
        decoded.append(np.nan if abs(value) == 9999 else value / 10.0)
    series = pd.Series(decoded, index=index).sort_index()
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    hourly, missing = resample_hourly(series)
    values = hourly.to_numpy(dtype=float)
    values[missing.to_numpy(dtype=bool)] = np.nan
    selected, selected_times = select_benchmark_span(
        values,
        hourly.index.to_numpy(dtype="datetime64[ns]"),
        horizon=horizon,
        min_length=min_length,
        max_length=max_length,
    )
    return ParsedSeries(
        entity_id=str(stations[0]),
        values=selected,
        timestamps=selected_times,
        natural_missing_mask=np.isnan(selected),
        frequency="hourly",
    )


_MONASH_CONFIG_FREQUENCY = {
    "nn5_daily": ("daily", "1D"),
    "covid_deaths": ("daily", "1D"),
    "traffic_hourly": ("hourly", "1h"),
    "electricity_hourly": ("hourly", "1h"),
}


def parse_monash_parquet(
    path: Path | str,
    *,
    config: str,
    min_length: int,
    horizon: int,
    max_length: int,
) -> tuple[ParsedSeries, ...]:
    """Parse a pinned Hugging Face Monash shard without filling natural NaNs."""

    try:
        frequency, pandas_frequency = _MONASH_CONFIG_FREQUENCY[config]
    except KeyError as exc:
        raise ValueError(f"unsupported Monash benchmark config: {config!r}") from exc
    frame = pd.read_parquet(path)
    target_column = "target" if "target" in frame else "series_value" if "series_value" in frame else None
    if target_column is None:
        raise ValueError("Monash parquet lacks target/series_value")
    rows: list[ParsedSeries] = []
    for position, row in frame.iterrows():
        values = np.asarray(row[target_column], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or np.isinf(values).any():
            continue
        entity_id = str(row.get("item_id", position))
        start = pd.Timestamp(row.get("start", "1970-01-01"))
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        timestamps = pd.date_range(start, periods=len(values), freq=pandas_frequency).to_numpy(
            dtype="datetime64[ns]"
        )
        try:
            selected, selected_times = select_benchmark_span(
                values,
                timestamps,
                horizon=horizon,
                min_length=min_length,
                max_length=max_length,
            )
        except ValueError:
            continue
        rows.append(
            ParsedSeries(
                entity_id=entity_id,
                values=selected,
                timestamps=selected_times,
                natural_missing_mask=np.isnan(selected),
                frequency=frequency,
            )
        )
    return tuple(rows)


def parse_metr_la_sensor_locations(path: Path | str) -> dict[str, tuple[float, float]]:
    """Read the pinned DCRNN sensor coordinate table as entity_id -> (lat, lon)."""

    frame = pd.read_csv(path, dtype={"sensor_id": str})
    required = {"sensor_id", "latitude", "longitude"}
    if not required <= set(frame.columns):
        raise ValueError("METR-LA sensor locations lack sensor_id/latitude/longitude")
    coordinates: dict[str, tuple[float, float]] = {}
    for row in frame.itertuples():
        entity = str(row.sensor_id).strip()
        latitude, longitude = float(row.latitude), float(row.longitude)
        if not entity:
            raise ValueError("METR-LA sensor locations contain a blank sensor_id")
        if entity in coordinates:
            raise ValueError(f"METR-LA sensor locations repeat sensor {entity!r}")
        if not np.isfinite([latitude, longitude]).all():
            raise ValueError(f"METR-LA sensor {entity!r} has a non-finite coordinate")
        coordinates[entity] = (latitude, longitude)
    if not coordinates:
        raise ValueError("METR-LA sensor locations are empty")
    return coordinates


def parse_metr_la_hdf(
    path: Path | str,
    *,
    min_length: int,
    horizon: int,
    max_length: int,
) -> tuple[ParsedSeries, ...]:
    """Parse the DCRNN METR-LA matrix and retain sensors as independent series."""

    frame = pd.read_hdf(path)
    if not isinstance(frame, pd.DataFrame) or frame.empty or frame.shape[1] < 1:
        raise ValueError("METR-LA HDF must contain a non-empty data frame")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"))
    index = _localize_index(index, "America/Los_Angeles")
    rows: list[ParsedSeries] = []
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        series = pd.Series(numeric.to_numpy(dtype=float), index=index).sort_index()
        if series.index.has_duplicates:
            series = series.groupby(level=0).mean()
        hourly, missing = resample_hourly(series)
        values = hourly.to_numpy(dtype=float)
        values[missing.to_numpy(dtype=bool)] = np.nan
        try:
            selected, selected_times = select_benchmark_span(
                values,
                hourly.index.to_numpy(dtype="datetime64[ns]"),
                horizon=horizon,
                min_length=min_length,
                max_length=max_length,
            )
        except ValueError:
            continue
        rows.append(
            ParsedSeries(
                entity_id=str(column),
                values=selected,
                timestamps=selected_times,
                natural_missing_mask=np.isnan(selected),
                frequency="hourly",
            )
        )
    return tuple(rows)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metadata_path(path: Path) -> Path:
    return path.with_name(path.name + ".asset.json")


@dataclass(frozen=True)
class RawAsset:
    path: Path
    sha256: str
    source_revision: str
    size: int


def _canonical_metadata(asset: RawAsset) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": "benchmark-raw-asset/1",
                "sha256": asset.sha256,
                "size": asset.size,
                "source_revision": asset.source_revision,
            },
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_raw_once(
    path: Path | str, payload: bytes, *, source_revision: str
) -> RawAsset:
    """Write bytes once and bind them to a revision in a durable sidecar."""

    target = Path(path)
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if (
        not isinstance(source_revision, str)
        or not source_revision
        or source_revision != source_revision.strip()
    ):
        raise ValueError("source_revision must be a canonical non-empty string")
    candidate = RawAsset(target, _sha256(payload), source_revision, len(payload))
    metadata = _metadata_path(target)
    if target.exists() or metadata.exists():
        if not target.is_file() or not metadata.is_file():
            raise RawMutationError(f"incomplete immutable asset: {target}")
        try:
            stored = json.loads(metadata.read_text("utf-8"))
            current_binding = RawAsset(
                target,
                stored["sha256"],
                stored["source_revision"],
                stored["size"],
            )
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RawMutationError(f"invalid immutable asset metadata: {target}") from exc
        current = verify_raw_asset(current_binding)
        if current.sha256 != candidate.sha256 or current.size != candidate.size:
            raise RawMutationError(f"immutable raw asset mismatch: {target}")
        if current.source_revision != source_revision:
            raise RawMutationError(f"immutable raw asset revision mismatch: {target}")
        return current
    _write_exclusive(target, payload)
    try:
        _write_exclusive(metadata, _canonical_metadata(candidate))
    except Exception:
        # A payload without its sidecar is intentionally fail-loud on the next access.
        raise
    return candidate


def verify_raw_asset(asset: RawAsset) -> RawAsset:
    if not isinstance(asset, RawAsset):
        raise TypeError("asset must be RawAsset")
    path = Path(asset.path)
    metadata_path = _metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        raise RawMutationError(f"immutable asset is missing: {path}")
    payload = path.read_bytes()
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RawMutationError(f"invalid immutable asset metadata: {path}") from exc
    expected_keys = {"schema_version", "sha256", "size", "source_revision"}
    if set(metadata) != expected_keys or metadata.get("schema_version") != "benchmark-raw-asset/1":
        raise RawMutationError(f"invalid immutable asset metadata: {path}")
    actual_sha = _sha256(payload)
    actual_size = len(payload)
    if metadata["sha256"] != actual_sha or metadata["size"] != actual_size:
        raise RawMutationError(f"immutable asset bytes disagree with metadata: {path}")
    current = RawAsset(path, actual_sha, metadata["source_revision"], actual_size)
    if (
        asset.sha256 != current.sha256
        or asset.size != current.size
        or asset.source_revision != current.source_revision
    ):
        raise RawMutationError(f"immutable asset binding mismatch: {path}")
    return current


def promote_download(
    temporary_path: Path | str,
    destination: Path | str,
    *,
    source_revision: str,
    expected_sha256: str | None = None,
) -> RawAsset:
    """Hash-check a resumable temporary file and atomically promote it to raw."""

    temporary = Path(temporary_path)
    target = Path(destination)
    if not temporary.is_file():
        raise FileNotFoundError(temporary)
    digest = hashlib.sha256()
    size = 0
    with temporary.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    actual_sha = digest.hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise RawMutationError(
            f"download SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    if target.exists() or _metadata_path(target).exists():
        payload = temporary.read_bytes()
        current = write_raw_once(target, payload, source_revision=source_revision)
        temporary.unlink()
        return current
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, target)
    asset = RawAsset(target, actual_sha, source_revision, size)
    try:
        _write_exclusive(_metadata_path(target), _canonical_metadata(asset))
    except Exception:
        # The promoted file remains deliberately unusable without its durable binding.
        raise
    return verify_raw_asset(asset)


def resample_hourly(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Hourly mean plus a mask marking any natural missingness in each bin."""

    if not isinstance(values, pd.Series) or not isinstance(values.index, pd.DatetimeIndex):
        raise TypeError("values must be a pandas Series with DatetimeIndex")
    if values.empty:
        raise ValueError("values must not be empty")
    if values.index.tz is None:
        raise ValueError("timestamps must carry a timezone")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("timestamps must be strictly increasing")
    try:
        numeric = pd.to_numeric(values, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be numeric") from exc
    numeric.index = numeric.index.tz_convert("UTC")
    hourly = numeric.resample("1h").mean()
    counts = numeric.resample("1h").size()
    any_missing = numeric.isna().resample("1h").max().astype(bool)
    mask = (any_missing | counts.eq(0) | hourly.isna()).astype(bool)
    return hourly, mask


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


@dataclass(frozen=True)
class CleanBaseAsset:
    record: SeriesRecord
    values_asset: RawAsset
    timestamps_asset: RawAsset | None
    mask_asset: RawAsset
    record_asset: RawAsset

    @property
    def values_path(self) -> Path:
        return self.values_asset.path

    @property
    def timestamps_path(self) -> Path | None:
        return None if self.timestamps_asset is None else self.timestamps_asset.path

    @property
    def mask_path(self) -> Path:
        return self.mask_asset.path


def materialize_clean_base(
    root: Path | str,
    *,
    dataset_id: str,
    entity_id: str,
    values: np.ndarray,
    source_id: str,
    source_asset_sha256: str,
    source_revision: str,
    license_id: str,
    overlap_family: str,
    exposure_class: str,
    frequency: str,
    overlap_group: str,
    overlap_status: str,
    overlap_evidence_sha256: str,
    timestamps: np.ndarray | None = None,
) -> CleanBaseAsset:
    """Materialize a stable entity slot; changed bytes fail instead of versioning silently."""

    record = SeriesRecord.from_values(
        dataset_id=dataset_id,
        entity_id=entity_id,
        values=values,
        source_id=source_id,
        source_asset_sha256=source_asset_sha256,
        source_revision=source_revision,
        license_id=license_id,
        overlap_family=overlap_family,
        exposure_class=exposure_class,
        frequency=frequency,
        overlap_group=overlap_group,
        overlap_status=overlap_status,
        overlap_evidence_sha256=overlap_evidence_sha256,
        timestamps=timestamps,
    )
    slot_key = _sha256(
        json.dumps(
            [source_id, dataset_id, entity_id],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    slot = Path(root) / slot_key
    revision = f"clean-base-v1:{record.content_sha}"
    canonical_values = np.asarray(values).astype("<f8", copy=True)
    canonical_values[np.isnan(canonical_values)] = np.nan
    values_asset = write_raw_once(
        slot / "values.npy", _npy_bytes(canonical_values), source_revision=revision
    )
    mask_asset = write_raw_once(
        slot / "natural_missing_mask.npy",
        _npy_bytes(np.isnan(canonical_values).astype(bool)),
        source_revision=revision,
    )
    timestamps_asset = None
    if timestamps is not None:
        canonical_timestamps = np.asarray(timestamps).astype("datetime64[ns]")
        timestamps_asset = write_raw_once(
            slot / "timestamps.npy",
            _npy_bytes(canonical_timestamps),
            source_revision=revision,
        )
    record_payload = (
        json.dumps(
            record.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")
    record_asset = write_raw_once(
        slot / "record.json", record_payload, source_revision=revision
    )
    return CleanBaseAsset(
        record=record,
        values_asset=values_asset,
        timestamps_asset=timestamps_asset,
        mask_asset=mask_asset,
        record_asset=record_asset,
    )


def read_clean_base(
    asset: CleanBaseAsset,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if not isinstance(asset, CleanBaseAsset):
        raise TypeError("asset must be CleanBaseAsset")
    verify_raw_asset(asset.values_asset)
    verify_raw_asset(asset.mask_asset)
    verify_raw_asset(asset.record_asset)
    values = np.load(asset.values_path, allow_pickle=False)
    mask = np.load(asset.mask_path, allow_pickle=False)
    timestamps = None
    if asset.timestamps_asset is not None:
        verify_raw_asset(asset.timestamps_asset)
        timestamps = np.load(asset.timestamps_path, allow_pickle=False)
    if values.ndim != 1 or mask.ndim != 1 or len(values) != len(mask):
        raise RawMutationError("clean-base arrays have invalid shape")
    if not np.array_equal(mask.astype(bool), np.isnan(values)):
        raise RawMutationError("clean-base natural missing mask disagrees with values")
    asset.record.verify_values(values, timestamps=timestamps)
    return values, timestamps, mask.astype(bool)
