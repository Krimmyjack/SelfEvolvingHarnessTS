"""Canonical per-series registry, provenance, and admission decisions."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from .sources import SOURCE_SPECS
from .split import SplitCandidate, SplitRole

__all__ = [
    "Admission",
    "RegistryError",
    "SeriesRecord",
    "admit_series",
    "import_legacy_inventory",
    "read_registry_jsonl",
    "write_registry_jsonl",
]


class RegistryError(ValueError):
    """Registry bytes or metadata violate the frozen schema."""


_EXPOSURE_CLASSES = frozenset(
    {
        "certified_virgin",
        "confirmed_exposed",
        "uncertain_legacy_exposure",
        "probe_consumed",
    }
)
_SUPPORT_A_ONLY = frozenset(
    {"confirmed_exposed", "uncertain_legacy_exposure", "probe_consumed"}
)
_FRESH_ROLES = tuple(role.value for role in SplitRole)
_UNRESOLVED_FRESH_ROLES = (
    SplitRole.SUPPORT_A.value,
    SplitRole.DEV_QUERY.value,
)
_OVERLAP_STATUSES = frozenset({"unresolved", "resolved"})
_PROBE_FEATURE_KEYS = frozenset(
    {
        "seasonal_strength",
        "trend_strength",
        "spectral_entropy",
        "natural_missing_count",
        "natural_missing_rate",
        "irregular_interval_count",
        "irregular_sampling_rate",
    }
)
_SCHEMA_VERSION = "benchmark-series-registry/2"
_INTERNAL_SOURCES = {
    "legacy_internal_monash_clean": {
        "source_revision": "legacy-monash-clean-artifact-v1",
        "license_id": "legacy-project-artifact",
        "overlap_family": "legacy_monash_clean",
    }
}


def _canonical_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RegistryError(f"{name} must be a canonical non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    value = _canonical_string(value, name)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RegistryError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite_rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite rate in [0,1]")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite rate in [0,1]") from exc
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be a finite rate in [0,1]")
    return converted


def _canonical_values(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise RegistryError("values must be a one-dimensional numeric array")
    try:
        array = raw.astype("<f8", copy=True)
    except (TypeError, ValueError) as exc:
        raise RegistryError("values must be a one-dimensional numeric array") from exc
    if array.size == 0:
        raise RegistryError("values must not be empty")
    if np.isinf(array).any():
        raise RegistryError("values must not contain infinity")
    # Canonicalize every NaN payload before hashing.
    array[np.isnan(array)] = np.nan
    return array


def _mask_sha(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    payload = len(mask).to_bytes(8, "big") + packed.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _timestamp_diagnostics(
    timestamps: np.ndarray | None, length: int
) -> tuple[str | None, int, float]:
    if timestamps is None:
        return None, 0, 0.0
    raw = np.asarray(timestamps)
    if raw.ndim != 1 or len(raw) != length:
        raise RegistryError("timestamps must be one-dimensional and match values")
    try:
        times = raw.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise RegistryError("timestamps must be datetime-like") from exc
    if np.isnat(times).any():
        raise RegistryError("timestamps must not contain NaT")
    ints = times.astype("<i8", copy=False)
    timestamps_sha = hashlib.sha256(ints.tobytes()).hexdigest()
    if length < 2:
        return timestamps_sha, 0, 0.0
    deltas = np.diff(ints)
    if np.any(deltas <= 0):
        raise RegistryError("timestamps must be strictly increasing")
    unique, counts = np.unique(deltas, return_counts=True)
    expected = unique[np.flatnonzero(counts == counts.max())[0]]
    irregular_count = int(np.count_nonzero(deltas != expected))
    return timestamps_sha, irregular_count, irregular_count / float(length - 1)


def _freeze_probe_features(
    value: Mapping[str, float | int] | None,
) -> Mapping[str, float | int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not value:
        raise RegistryError("probe_features must be a non-empty mapping")
    if set(value) != _PROBE_FEATURE_KEYS:
        raise RegistryError("probe_features fields differ from canonical probe schema")
    frozen: dict[str, float | int] = {}
    for key, feature in value.items():
        _canonical_string(key, "probe feature name")
        if isinstance(feature, bool) or not isinstance(feature, (int, float)):
            raise RegistryError("probe feature values must be finite numeric data")
        try:
            converted = float(feature)
        except (OverflowError, TypeError, ValueError) as exc:
            raise RegistryError(
                "probe feature values must be finite numeric data"
            ) from exc
        if not math.isfinite(converted):
            raise RegistryError("probe feature values must be finite numeric data")
        frozen[key] = feature
    for key in ("natural_missing_count", "irregular_interval_count"):
        feature = frozen[key]
        if not isinstance(feature, int) or feature < 0:
            raise RegistryError(f"{key} probe feature must be a non-negative integer")
    for key in _PROBE_FEATURE_KEYS - {
        "natural_missing_count",
        "irregular_interval_count",
    }:
        feature = float(frozen[key])
        if not 0.0 <= feature <= 1.0:
            raise RegistryError(f"{key} probe feature must be in [0,1]")
    return MappingProxyType(frozen)


def _expected_roles(exposure_class: str, overlap_status: str) -> tuple[str, ...]:
    if exposure_class in _SUPPORT_A_ONLY:
        return (SplitRole.SUPPORT_A.value,)
    if overlap_status == "unresolved":
        return _UNRESOLVED_FRESH_ROLES
    return _FRESH_ROLES


def _validate_source_binding(
    *,
    source_id: str,
    source_asset_sha256: str,
    source_revision: str,
    license_id: str,
    overlap_family: str,
) -> None:
    if source_id in SOURCE_SPECS:
        spec = SOURCE_SPECS[source_id]
        for name in ("source_revision", "license_id", "overlap_family"):
            if locals()[name] != getattr(spec, name):
                raise RegistryError(f"{name} disagrees with registered SourceSpec")
        try:
            spec.validate_asset_sha256(source_asset_sha256)
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc
        return
    try:
        binding = _INTERNAL_SOURCES[source_id]
    except KeyError as exc:
        raise RegistryError(f"source_id is not registered: {source_id!r}") from exc
    for name in ("source_revision", "license_id", "overlap_family"):
        if locals()[name] != binding[name]:
            raise RegistryError(f"{name} disagrees with internal source binding")


@dataclass(frozen=True)
class Admission:
    eligible: bool
    reasons: tuple[str, ...]
    natural_missing_rate: float
    irregular_sampling_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean")
        if not isinstance(self.reasons, tuple):
            raise ValueError("reasons must be a tuple")
        for reason in self.reasons:
            _canonical_string(reason, "admission reason")
        if len(set(self.reasons)) != len(self.reasons):
            raise ValueError("admission reasons must be unique")
        if self.eligible != (not self.reasons):
            raise ValueError("eligible must agree with admission reasons")
        _finite_rate(self.natural_missing_rate, "natural_missing_rate")
        _finite_rate(self.irregular_sampling_rate, "irregular_sampling_rate")


@dataclass(frozen=True)
class SeriesRecord:
    schema_version: str
    series_uid: str
    dataset_id: str
    entity_id: str
    content_sha: str
    source_id: str
    source_asset_sha256: str
    source_revision: str
    license_id: str
    frequency: str
    length: int
    natural_missing_count: int
    natural_missing_rate: float
    natural_missing_mask_sha: str
    timestamps_sha: str | None
    irregular_interval_count: int
    irregular_sampling_rate: float
    overlap_family: str
    overlap_group: str
    overlap_status: str
    overlap_evidence_sha256: str
    exposure_class: str
    probe_features: Mapping[str, float | int] | None
    regime_tag: str | None
    admission_reasons: tuple[str, ...] | None
    roles_allowed: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise RegistryError("unsupported registry schema version")
        for name in (
            "series_uid",
            "dataset_id",
            "entity_id",
            "content_sha",
            "source_id",
            "source_asset_sha256",
            "source_revision",
            "license_id",
            "frequency",
            "natural_missing_mask_sha",
            "overlap_family",
            "overlap_group",
            "overlap_status",
            "overlap_evidence_sha256",
            "exposure_class",
        ):
            _canonical_string(getattr(self, name), name)
        for value, name in (
            (self.series_uid, "series_uid"),
            (self.content_sha, "content_sha"),
            (self.source_asset_sha256, "source_asset_sha256"),
            (self.natural_missing_mask_sha, "natural_missing_mask_sha"),
            (self.overlap_evidence_sha256, "overlap_evidence_sha256"),
        ):
            _require_sha256(value, name)
        if self.timestamps_sha is not None:
            _require_sha256(self.timestamps_sha, "timestamps_sha")
        if self.exposure_class not in _EXPOSURE_CLASSES:
            raise RegistryError("exposure_class is not a frozen exposure class")
        if self.overlap_status not in _OVERLAP_STATUSES:
            raise RegistryError("overlap_status is not a frozen status")
        _validate_source_binding(
            source_id=self.source_id,
            source_asset_sha256=self.source_asset_sha256,
            source_revision=self.source_revision,
            license_id=self.license_id,
            overlap_family=self.overlap_family,
        )
        if isinstance(self.length, bool) or not isinstance(self.length, int) or self.length < 1:
            raise RegistryError("length must be a positive integer")
        for name, maximum in (
            ("natural_missing_count", self.length),
            ("irregular_interval_count", max(0, self.length - 1)),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise RegistryError(f"{name} is outside its valid range")
        expected_missing_rate = self.natural_missing_count / float(self.length)
        actual_missing_rate = _finite_rate(
            self.natural_missing_rate, "natural_missing_rate"
        )
        if not math.isclose(actual_missing_rate, expected_missing_rate, abs_tol=1e-15):
            raise RegistryError("natural_missing_rate disagrees with count")
        denominator = max(1, self.length - 1)
        expected_irregular_rate = self.irregular_interval_count / float(denominator)
        actual_irregular_rate = _finite_rate(
            self.irregular_sampling_rate, "irregular_sampling_rate"
        )
        if not math.isclose(actual_irregular_rate, expected_irregular_rate, abs_tol=1e-15):
            raise RegistryError("irregular_sampling_rate disagrees with count")
        finalized = (
            self.probe_features is not None,
            self.regime_tag is not None,
            self.admission_reasons is not None,
        )
        if len(set(finalized)) != 1:
            raise RegistryError(
                "probe_features, regime_tag, and admission_reasons finalize together"
            )
        if finalized[0]:
            object.__setattr__(self, "probe_features", _freeze_probe_features(self.probe_features))
            _canonical_string(self.regime_tag, "regime_tag")
            if not isinstance(self.admission_reasons, tuple):
                raise RegistryError("admission_reasons must be a tuple")
            for reason in self.admission_reasons:
                _canonical_string(reason, "admission reason")
            if len(set(self.admission_reasons)) != len(self.admission_reasons):
                raise RegistryError("admission_reasons must be unique")
        expected_roles = _expected_roles(self.exposure_class, self.overlap_status)
        if not isinstance(self.roles_allowed, tuple):
            raise RegistryError("roles_allowed must be an immutable tuple")
        if tuple(self.roles_allowed) != expected_roles:
            raise RegistryError("roles_allowed disagrees with exposure/overlap status")

    @classmethod
    def from_values(
        cls,
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
    ) -> "SeriesRecord":
        for value, name in (
            (dataset_id, "dataset_id"),
            (entity_id, "entity_id"),
            (source_id, "source_id"),
            (source_revision, "source_revision"),
            (license_id, "license_id"),
            (overlap_family, "overlap_family"),
            (frequency, "frequency"),
            (overlap_group, "overlap_group"),
            (overlap_status, "overlap_status"),
        ):
            _canonical_string(value, name)
        _require_sha256(source_asset_sha256, "source_asset_sha256")
        _require_sha256(overlap_evidence_sha256, "overlap_evidence_sha256")
        if exposure_class not in _EXPOSURE_CLASSES:
            raise RegistryError("exposure_class is not a frozen exposure class")
        if overlap_status not in _OVERLAP_STATUSES:
            raise RegistryError("overlap_status is not a frozen status")
        _validate_source_binding(
            source_id=source_id,
            source_asset_sha256=source_asset_sha256,
            source_revision=source_revision,
            license_id=license_id,
            overlap_family=overlap_family,
        )
        array = _canonical_values(values)
        missing = np.isnan(array)
        content_sha = hashlib.sha256(array.tobytes()).hexdigest()
        timestamp_sha, irregular_count, irregular_rate = _timestamp_diagnostics(
            timestamps, len(array)
        )
        identity = json.dumps(
            [
                source_id,
                source_asset_sha256,
                dataset_id,
                entity_id,
                source_revision,
                content_sha,
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        series_uid = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return cls(
            schema_version=_SCHEMA_VERSION,
            series_uid=series_uid,
            dataset_id=dataset_id,
            entity_id=entity_id,
            content_sha=content_sha,
            source_id=source_id,
            source_asset_sha256=source_asset_sha256,
            source_revision=source_revision,
            license_id=license_id,
            frequency=frequency,
            length=len(array),
            natural_missing_count=int(missing.sum()),
            natural_missing_rate=float(missing.mean()),
            natural_missing_mask_sha=_mask_sha(missing),
            timestamps_sha=timestamp_sha,
            irregular_interval_count=irregular_count,
            irregular_sampling_rate=irregular_rate,
            overlap_family=overlap_family,
            overlap_group=overlap_group,
            overlap_status=overlap_status,
            overlap_evidence_sha256=overlap_evidence_sha256,
            exposure_class=exposure_class,
            probe_features=None,
            regime_tag=None,
            admission_reasons=None,
            roles_allowed=_expected_roles(exposure_class, overlap_status),
        )

    def with_probe_result(
        self,
        *,
        probe_features: Mapping[str, float | int],
        regime_tag: str,
        admission: Admission,
    ) -> "SeriesRecord":
        """Atomically finalize structural features, regime, and admission once."""

        if self.probe_features is not None:
            raise RegistryError("probe result is already finalized")
        if not isinstance(admission, Admission):
            raise TypeError("admission must be Admission")
        if not math.isclose(
            admission.natural_missing_rate,
            self.natural_missing_rate,
            abs_tol=1e-15,
        ):
            raise RegistryError("admission natural_missing_rate disagrees with registry")
        if not math.isclose(
            admission.irregular_sampling_rate,
            self.irregular_sampling_rate,
            abs_tol=1e-15,
        ):
            raise RegistryError("admission irregular_sampling_rate disagrees with registry")
        frozen_features = _freeze_probe_features(probe_features)
        assert frozen_features is not None
        expected_probe_values = {
            "natural_missing_count": self.natural_missing_count,
            "natural_missing_rate": self.natural_missing_rate,
            "irregular_interval_count": self.irregular_interval_count,
            "irregular_sampling_rate": self.irregular_sampling_rate,
        }
        for name, expected in expected_probe_values.items():
            actual = frozen_features[name]
            if isinstance(expected, int):
                agrees = actual == expected
            else:
                agrees = math.isclose(float(actual), float(expected), abs_tol=1e-15)
            if not agrees:
                raise RegistryError(f"probe feature {name} disagrees with registry")
        return replace(
            self,
            probe_features=frozen_features,
            regime_tag=_canonical_string(regime_tag, "regime_tag"),
            admission_reasons=tuple(admission.reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        payload["roles_allowed"] = list(self.roles_allowed)
        payload["admission_reasons"] = (
            None if self.admission_reasons is None else list(self.admission_reasons)
        )
        payload["probe_features"] = (
            None if self.probe_features is None else dict(self.probe_features)
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SeriesRecord":
        if not isinstance(payload, Mapping):
            raise RegistryError("registry row must be a JSON object")
        expected = {field.name for field in fields(cls)}
        if set(payload) != expected:
            raise RegistryError("registry row fields differ from canonical schema")
        data = dict(payload)
        roles = data.get("roles_allowed")
        if not isinstance(roles, list) or not all(isinstance(item, str) for item in roles):
            raise RegistryError("roles_allowed must be a JSON string array")
        data["roles_allowed"] = tuple(roles)
        reasons = data.get("admission_reasons")
        if reasons is not None:
            if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
                raise RegistryError("admission_reasons must be null or a JSON string array")
            data["admission_reasons"] = tuple(reasons)
        features_value = data.get("probe_features")
        if features_value is not None and not isinstance(features_value, Mapping):
            raise RegistryError("probe_features must be null or a JSON object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise RegistryError("registry row contains invalid field types") from exc

    def to_split_candidate(self) -> SplitCandidate:
        if self.overlap_status != "resolved":
            raise RegistryError("overlap must be resolved before split allocation")
        if (
            self.probe_features is None
            or self.regime_tag is None
            or self.admission_reasons is None
        ):
            raise RegistryError("probe/admission must be finalized before split allocation")
        if self.admission_reasons:
            raise RegistryError("admission rejected this series from split allocation")
        return SplitCandidate(
            series_uid=self.series_uid,
            dataset_id=self.dataset_id,
            regime_tag=self.regime_tag,
            overlap_group=self.overlap_group,
            exposure_class=self.exposure_class,
            length=self.length,
        )

    def verify_values(
        self, values: np.ndarray, *, timestamps: np.ndarray | None = None
    ) -> None:
        """Fail closed unless local clean-base bytes match this registry row."""

        array = _canonical_values(values)
        if len(array) != self.length:
            raise RegistryError("clean values length disagrees with registry")
        if hashlib.sha256(array.tobytes()).hexdigest() != self.content_sha:
            raise RegistryError("clean values content_sha disagrees with registry")
        if _mask_sha(np.isnan(array)) != self.natural_missing_mask_sha:
            raise RegistryError("natural missing mask identity disagrees with registry")
        timestamp_sha, irregular_count, irregular_rate = _timestamp_diagnostics(
            timestamps, len(array)
        )
        if timestamp_sha != self.timestamps_sha:
            raise RegistryError("timestamp identity disagrees with registry")
        if (
            irregular_count != self.irregular_interval_count
            or not math.isclose(
                irregular_rate, self.irregular_sampling_rate, abs_tol=1e-15
            )
        ):
            raise RegistryError("sampling diagnostics disagree with registry")


def admit_series(
    record: SeriesRecord,
    *,
    min_len: int,
    allowed_frequencies: set[str] | frozenset[str],
    max_natural_missing_rate: float,
    max_irregular_sampling_rate: float,
) -> Admission:
    if not isinstance(record, SeriesRecord):
        raise TypeError("record must be SeriesRecord")
    if isinstance(min_len, bool) or not isinstance(min_len, int) or min_len < 1:
        raise ValueError("min_len must be a positive integer")
    if not isinstance(allowed_frequencies, (set, frozenset)):
        raise ValueError("allowed_frequencies must be a set or frozenset")
    for frequency in allowed_frequencies:
        _canonical_string(frequency, "allowed frequency")
    missing_limit = _finite_rate(
        max_natural_missing_rate, "max_natural_missing_rate"
    )
    irregular_limit = _finite_rate(
        max_irregular_sampling_rate, "max_irregular_sampling_rate"
    )
    reasons: list[str] = []
    if record.length < min_len:
        reasons.append("below_min_length")
    if record.frequency not in allowed_frequencies:
        reasons.append("frequency_not_allowed")
    if record.natural_missing_count == record.length:
        reasons.append("no_finite_observations")
    if record.natural_missing_rate > missing_limit:
        reasons.append("excessive_natural_missingness")
    if record.irregular_sampling_rate > irregular_limit:
        reasons.append("excessive_irregular_sampling")
    return Admission(
        eligible=not reasons,
        reasons=tuple(reasons),
        natural_missing_rate=record.natural_missing_rate,
        irregular_sampling_rate=record.irregular_sampling_rate,
    )


def write_registry_jsonl(path: Path | str, records: Iterable[SeriesRecord]) -> None:
    target = Path(path)
    rows = list(records)
    if any(not isinstance(row, SeriesRecord) for row in rows):
        raise RegistryError("registry contains a non-SeriesRecord row")
    if any(
        row.exposure_class == "certified_virgin"
        and (
            row.probe_features is None
            or row.regime_tag is None
            or row.admission_reasons is None
        )
        for row in rows
    ):
        raise RegistryError(
            "certified-virgin registry rows require finalized probe/admission"
        )
    uids = [row.series_uid for row in rows]
    if len(uids) != len(set(uids)):
        raise RegistryError("registry contains duplicate series_uid values")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":")
                )
                + "\n"
            )


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RegistryError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def read_registry_jsonl(path: Path | str) -> list[SeriesRecord]:
    rows: list[SeriesRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text("utf-8").splitlines(), 1):
        if not line:
            raise RegistryError(f"blank registry line at {line_number}")
        try:
            payload = json.loads(line, object_pairs_hook=_reject_duplicate_object_keys)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"invalid JSON at registry line {line_number}") from exc
        row = SeriesRecord.from_dict(payload)
        if row.series_uid in seen:
            raise RegistryError(f"duplicate series_uid at registry line {line_number}")
        seen.add(row.series_uid)
        rows.append(row)
    return rows


_LEGACY_FREQUENCY = {
    "nn5_daily": "daily",
    "fred_md": "monthly",
    "tourism_monthly": "monthly",
    "covid_deaths": "daily",
    "us_births": "daily",
    "saugeenday": "daily",
    "sunspot": "daily",
}


def _legacy_bundle_sha(metadata_bytes: bytes, values_bytes: bytes) -> str:
    payload = (
        b"benchmark-legacy-bundle-v1\0"
        + len(metadata_bytes).to_bytes(8, "big")
        + metadata_bytes
        + len(values_bytes).to_bytes(8, "big")
        + values_bytes
    )
    return hashlib.sha256(payload).hexdigest()


def import_legacy_inventory(metadata_path: Path | str) -> list[SeriesRecord]:
    path = Path(metadata_path)
    metadata_bytes = path.read_bytes()
    metadata = [
        json.loads(line, object_pairs_hook=_reject_duplicate_object_keys)
        for line in metadata_bytes.decode("utf-8").splitlines()
        if line
    ]
    values_path = path.with_name(path.name.replace(".meta.jsonl", ".npz"))
    if not values_path.is_file():
        raise RegistryError(f"legacy values artifact is missing: {values_path}")
    values_bytes = values_path.read_bytes()
    source_asset_sha256 = _legacy_bundle_sha(metadata_bytes, values_bytes)
    with np.load(values_path, allow_pickle=True) as archive:
        if "clean" not in archive:
            raise RegistryError("legacy values artifact has no clean array")
        clean_values = list(archive["clean"])
    if len(metadata) != 83 or len(clean_values) != 83:
        raise RegistryError("legacy inventory must contain exactly 83 series")
    expected_counts = Counter(
        {
            "nn5_daily": 20,
            "fred_md": 20,
            "tourism_monthly": 20,
            "covid_deaths": 20,
            "us_births": 1,
            "saugeenday": 1,
            "sunspot": 1,
        }
    )
    actual_counts = Counter(item.get("config") for item in metadata if isinstance(item, dict))
    if actual_counts != expected_counts:
        raise RegistryError("legacy inventory config counts differ from frozen 83")
    records: list[SeriesRecord] = []
    for index, (item, values) in enumerate(zip(metadata, clean_values)):
        if not isinstance(item, dict):
            raise RegistryError(f"legacy metadata row {index} is not an object")
        config = _canonical_string(item.get("config"), "config")
        item_id = _canonical_string(item.get("item_id"), "item_id")
        frequency = _LEGACY_FREQUENCY.get(config)
        if frequency is None:
            raise RegistryError(f"unknown legacy config: {config!r}")
        records.append(
            SeriesRecord.from_values(
                dataset_id=f"legacy_monash:{config}",
                entity_id=item_id,
                values=np.asarray(values),
                source_id="legacy_internal_monash_clean",
                source_asset_sha256=source_asset_sha256,
                source_revision="legacy-monash-clean-artifact-v1",
                license_id="legacy-project-artifact",
                overlap_family="legacy_monash_clean",
                exposure_class="confirmed_exposed",
                frequency=frequency,
                overlap_group=f"legacy:{config}:{item_id}",
                overlap_status="resolved",
                overlap_evidence_sha256=source_asset_sha256,
            )
        )
    if len({row.series_uid for row in records}) != 83:
        raise RegistryError("legacy inventory series identities are not unique")
    return records
