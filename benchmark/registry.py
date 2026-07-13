"""Canonical per-series registry, provenance, and admission decisions."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

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
_SCHEMA_VERSION = "benchmark-series-registry/1"


def _canonical_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RegistryError(f"{name} must be a canonical non-empty string")
    return value


def _canonical_values(values: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(values, dtype="<f8").reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise RegistryError("values must be a one-dimensional numeric array") from exc
    if array.size == 0:
        raise RegistryError("values must not be empty")
    if np.isinf(array).any():
        raise RegistryError("values must not contain infinity")
    array[np.isnan(array)] = np.nan
    return array


def _mask_sha(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    payload = len(mask).to_bytes(8, "big") + packed.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RegistryError(f"{name} must be a lowercase SHA256 digest")


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
    positive = deltas[deltas > 0]
    if positive.size == 0:
        irregular_count = int(deltas.size)
    else:
        unique, counts = np.unique(positive, return_counts=True)
        expected = unique[np.flatnonzero(counts == counts.max())[0]]
        irregular_count = int(np.count_nonzero(deltas != expected))
    return timestamps_sha, irregular_count, irregular_count / float(length - 1)


@dataclass(frozen=True)
class SeriesRecord:
    schema_version: str
    series_uid: str
    dataset_id: str
    entity_id: str
    content_sha: str
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
    overlap_group: str
    exposure_class: str
    regime_tag: str
    roles_allowed: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise RegistryError("unsupported registry schema version")
        for name in (
            "series_uid",
            "dataset_id",
            "entity_id",
            "content_sha",
            "source_revision",
            "license_id",
            "frequency",
            "natural_missing_mask_sha",
            "overlap_group",
            "exposure_class",
            "regime_tag",
        ):
            _canonical_string(getattr(self, name), name)
        if self.timestamps_sha is not None:
            _canonical_string(self.timestamps_sha, "timestamps_sha")
            _require_sha256(self.timestamps_sha, "timestamps_sha")
        _require_sha256(self.content_sha, "content_sha")
        _require_sha256(self.natural_missing_mask_sha, "natural_missing_mask_sha")
        if self.exposure_class not in _EXPOSURE_CLASSES:
            raise RegistryError("exposure_class is not a frozen exposure class")
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
        if not math.isclose(self.natural_missing_rate, expected_missing_rate, abs_tol=1e-15):
            raise RegistryError("natural_missing_rate disagrees with count")
        denominator = max(1, self.length - 1)
        expected_irregular_rate = self.irregular_interval_count / float(denominator)
        if not math.isclose(self.irregular_sampling_rate, expected_irregular_rate, abs_tol=1e-15):
            raise RegistryError("irregular_sampling_rate disagrees with count")
        expected_roles = (
            (SplitRole.SUPPORT_A.value,)
            if self.exposure_class in _SUPPORT_A_ONLY
            else _FRESH_ROLES
        )
        if tuple(self.roles_allowed) != expected_roles:
            raise RegistryError("roles_allowed disagrees with exposure_class")

    @classmethod
    def from_values(
        cls,
        *,
        dataset_id: str,
        entity_id: str,
        values: np.ndarray,
        source_revision: str,
        license_id: str,
        exposure_class: str,
        frequency: str = "unknown",
        overlap_group: str,
        regime_tag: str,
        timestamps: np.ndarray | None = None,
    ) -> "SeriesRecord":
        for value, name in (
            (dataset_id, "dataset_id"),
            (entity_id, "entity_id"),
            (source_revision, "source_revision"),
            (license_id, "license_id"),
            (frequency, "frequency"),
            (overlap_group, "overlap_group"),
            (regime_tag, "regime_tag"),
        ):
            _canonical_string(value, name)
        if exposure_class not in _EXPOSURE_CLASSES:
            raise RegistryError("exposure_class is not a frozen exposure class")
        array = _canonical_values(values)
        missing = np.isnan(array)
        content_sha = hashlib.sha256(array.tobytes()).hexdigest()
        timestamp_sha, irregular_count, irregular_rate = _timestamp_diagnostics(
            timestamps, len(array)
        )
        identity = json.dumps(
            [dataset_id, entity_id, source_revision, content_sha],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        series_uid = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        roles = (
            (SplitRole.SUPPORT_A.value,)
            if exposure_class in _SUPPORT_A_ONLY
            else _FRESH_ROLES
        )
        return cls(
            schema_version=_SCHEMA_VERSION,
            series_uid=series_uid,
            dataset_id=dataset_id,
            entity_id=entity_id,
            content_sha=content_sha,
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
            overlap_group=overlap_group,
            exposure_class=exposure_class,
            regime_tag=regime_tag,
            roles_allowed=roles,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["roles_allowed"] = list(self.roles_allowed)
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
        try:
            return cls(**data)
        except TypeError as exc:
            raise RegistryError("registry row contains invalid field types") from exc

    def to_split_candidate(self) -> SplitCandidate:
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


@dataclass(frozen=True)
class Admission:
    eligible: bool
    reasons: tuple[str, ...]
    natural_missing_rate: float
    irregular_sampling_rate: float


def admit_series(
    record: SeriesRecord,
    *,
    min_len: int,
    allowed_frequencies: set[str] | frozenset[str],
) -> Admission:
    if not isinstance(record, SeriesRecord):
        raise TypeError("record must be SeriesRecord")
    if isinstance(min_len, bool) or not isinstance(min_len, int) or min_len < 1:
        raise ValueError("min_len must be a positive integer")
    reasons: list[str] = []
    if record.length < min_len:
        reasons.append("below_min_length")
    if record.frequency not in allowed_frequencies:
        reasons.append("frequency_not_allowed")
    if record.natural_missing_count == record.length:
        reasons.append("no_finite_observations")
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


def read_registry_jsonl(path: Path | str) -> list[SeriesRecord]:
    rows: list[SeriesRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text("utf-8").splitlines(), 1):
        if not line:
            raise RegistryError(f"blank registry line at {line_number}")
        try:
            payload = json.loads(line)
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


def import_legacy_inventory(metadata_path: Path | str) -> list[SeriesRecord]:
    path = Path(metadata_path)
    metadata = [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line
    ]
    values_path = path.with_name(path.name.replace(".meta.jsonl", ".npz"))
    if not values_path.is_file():
        raise RegistryError(f"legacy values artifact is missing: {values_path}")
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
                values=np.asarray(values, dtype=np.float64),
                source_revision="legacy-monash-clean-artifact-v1",
                license_id="legacy-project-artifact",
                exposure_class="confirmed_exposed",
                frequency=frequency,
                overlap_group=f"legacy:{config}:{item_id}",
                regime_tag="legacy_support_a",
            )
        )
    if len({row.series_uid for row in records}) != 83:
        raise RegistryError("legacy inventory series identities are not unique")
    return records
