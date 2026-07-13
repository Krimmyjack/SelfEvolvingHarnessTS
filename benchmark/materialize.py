"""Immutable raw storage and deterministic clean-base materialization."""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .registry import SeriesRecord

__all__ = [
    "CleanBaseAsset",
    "RawAsset",
    "RawMutationError",
    "materialize_clean_base",
    "promote_download",
    "read_clean_base",
    "resample_hourly",
    "verify_raw_asset",
    "write_raw_once",
]


class RawMutationError(RuntimeError):
    """An immutable raw or clean-base path disagrees with frozen bytes."""


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
