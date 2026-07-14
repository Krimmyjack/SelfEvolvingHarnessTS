"""Deterministic outer-role allocation and manifest validation.

The split is an immutable protocol decision rather than an RNG draw.  Every
atomic overlap group receives its role from a canonical SHA256 value unless
exposure or the frozen U selection forces the whole group into a safer role.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from . import (
    BENCHMARK_VERSION,
    DESIGN_COMMIT,
    EXTERNAL_ADDENDUM_SHA256,
    HEADLINE_HORIZON,
    HEADLINE_LOOKBACK,
    HEADLINE_MIN_LENGTH,
    KNOWN_BENCHMARK_VERSIONS,
)

__all__ = [
    "SUPPORT_A_DISCOVERY_FRACTION",
    "SUPPORT_A_SUBSPLIT_SALT",
    "SplitAssignment",
    "SplitCandidate",
    "SplitManifest",
    "SplitManifestError",
    "SplitRole",
    "build_split_manifest",
    "build_support_a_subsplit",
    "group_hash_value",
    "role_from_unit_interval",
    "support_a_partition",
    "validate_split_manifest",
]

# Support-A is one role but two jobs, and conflating them is how a development loop
# quietly overfits: you search for a program on the same series you then use to decide
# whether the program earned promotion.  The partition below is at the OVERLAP-GROUP
# level (not the series level), so a group's members never land on both sides.
#
# This is an entirely different axis from the chronological train/validation/test
# boundaries inside each series -- those slice one series in time; this slices the
# Support-A population.  The names are kept deliberately unalike so a reader can never
# mistake one for the other.
SUPPORT_A_SUBSPLIT_SALT = "benchmark-support-a-subsplit-v1"
SUPPORT_A_DISCOVERY_FRACTION = 0.70


class SplitManifestError(ValueError):
    """A candidate set or split manifest violates the frozen protocol."""


class SplitRole(str, Enum):
    SUPPORT_A = "support_a"
    SUPPORT_B = "support_b"
    DEV_QUERY = "dev_query"
    FINAL_QUERY = "final_query"
    U = "u"


_SCHEMA_VERSION = "benchmark-split-manifest/1"
_EXPOSURE_CLASSES = frozenset(
    {
        "certified_virgin",
        "confirmed_exposed",
        "uncertain_legacy_exposure",
        "probe_consumed",
    }
)
_FORCED_SUPPORT_A_EXPOSURES = frozenset(
    {"confirmed_exposed", "uncertain_legacy_exposure", "probe_consumed"}
)

_INNER_SPLIT = {
    "indexing": "zero_based_half_open",
    "lookback": HEADLINE_LOOKBACK,
    "horizon": HEADLINE_HORIZON,
    "validation_size": HEADLINE_HORIZON,
    "test_size": HEADLINE_HORIZON,
    "minimum_length": HEADLINE_MIN_LENGTH,
}

_ROLE_POLICIES: Mapping[SplitRole, Mapping[str, bool]] = {
    SplitRole.SUPPORT_A: {
        "repeatable": True,
        "utility_visible": True,
        "may_select_best_fixed": True,
        "may_train_oracle_transfer": True,
        "may_confirm_method": False,
        "final_eligible": False,
    },
    SplitRole.SUPPORT_B: {
        "repeatable": False,
        "utility_visible": True,
        "may_select_best_fixed": False,
        "may_train_oracle_transfer": False,
        "may_confirm_method": True,
        "final_eligible": False,
    },
    SplitRole.DEV_QUERY: {
        "repeatable": True,
        "utility_visible": True,
        "may_select_best_fixed": False,
        "may_train_oracle_transfer": False,
        "may_confirm_method": False,
        "final_eligible": False,
    },
    SplitRole.FINAL_QUERY: {
        "repeatable": False,
        "utility_visible": True,
        "may_select_best_fixed": False,
        "may_train_oracle_transfer": False,
        "may_confirm_method": False,
        "final_eligible": True,
    },
    SplitRole.U: {
        "repeatable": False,
        "utility_visible": True,
        "may_select_best_fixed": False,
        "may_train_oracle_transfer": False,
        "may_confirm_method": False,
        "final_eligible": False,
    },
}


class _ImmutableSequence(tuple):
    """Tuple storage that retains JSON-list equality for legacy callers."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = tuple.__hash__


def _freeze_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_nested(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return _ImmutableSequence(_freeze_nested(item) for item in value)
    return value


def _require_canonical_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SplitManifestError(
            f"{field_name} must be a non-empty string without boundary whitespace"
        )
    return value


def _require_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SplitManifestError(f"{field_name} must be finite numeric data")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise SplitManifestError(
            f"{field_name} must be finite numeric data"
        ) from exc
    if not math.isfinite(converted):
        raise SplitManifestError(f"{field_name} must be finite numeric data")
    return converted


def _require_exact_keys(
    value: Any, expected: set[str], context: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SplitManifestError(f"{context} must be an object")
    keys = list(value.keys())
    if any(not isinstance(key, str) for key in keys):
        raise SplitManifestError(f"{context} keys must be strings")
    actual = set(keys)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SplitManifestError(
            f"{context} fields are malformed (missing={missing}, extra={extra})"
        )
    return value


def role_from_unit_interval(u: float) -> SplitRole:
    """Map a canonical hash fraction to the frozen fresh-data role bins."""

    try:
        value = float(u)
    except (TypeError, ValueError) as exc:
        raise ValueError("u must be in [0,1)") from exc
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("u must be in [0,1)")
    if value < 0.25:
        return SplitRole.SUPPORT_A
    if value < 0.45:
        return SplitRole.SUPPORT_B
    if value < 0.65:
        return SplitRole.DEV_QUERY
    return SplitRole.FINAL_QUERY


def group_hash_value(version: str, salt: str, group_key: str) -> float:
    """Return the canonical big-endian uint64 SHA256 fraction for a group."""

    for value, name in (
        (version, "benchmark version"),
        (salt, "split salt"),
        (group_key, "group key"),
    ):
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
        ):
            raise ValueError(
                f"{name} must be a non-empty string without boundary whitespace"
            )
    raw = hashlib.sha256(
        f"{version}|outer|{salt}|{group_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(raw[:8], "big") / float(1 << 64)


@dataclass(frozen=True)
class SplitCandidate:
    """Registry projection needed to freeze one series into an outer role."""

    series_uid: str
    dataset_id: str
    regime_tag: str
    overlap_group: str | None
    exposure_class: str
    length: int | None = None

    @property
    def group_key(self) -> str:
        return self.overlap_group or self.series_uid


@dataclass(frozen=True)
class SplitAssignment:
    """One auditable series placement in a frozen split manifest."""

    series_uid: str
    dataset_id: str
    regime_tag: str
    overlap_group: str | None
    exposure_class: str
    length: int | None
    group_key: str
    group_hash_value: float
    role: SplitRole
    forced_by: str | None
    chronological_boundaries: Mapping[str, tuple[int, int]] | None

    def __post_init__(self) -> None:
        if self.chronological_boundaries is not None:
            object.__setattr__(
                self,
                "chronological_boundaries",
                _freeze_nested(self.chronological_boundaries),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "series_uid": self.series_uid,
            "dataset_id": self.dataset_id,
            "regime_tag": self.regime_tag,
            "overlap_group": self.overlap_group,
            "exposure_class": self.exposure_class,
            "length": self.length,
            "group_key": self.group_key,
            "group_hash_value": self.group_hash_value,
            "role": self.role.value,
            "forced_by": self.forced_by,
            "chronological_boundaries": (
                None
                if self.chronological_boundaries is None
                else {
                    name: list(bounds)
                    for name, bounds in self.chronological_boundaries.items()
                }
            ),
        }


@dataclass(frozen=True)
class SplitManifest:
    """Canonical split result plus every rule needed to validate it later."""

    benchmark_version: str
    split_salt: str
    assignments: tuple[SplitAssignment, ...]
    u_selected_uids: tuple[str, ...]
    inner_split: Mapping[str, Any]
    policies: Mapping[str, Mapping[str, bool]]
    provenance: Mapping[str, str]
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", tuple(self.assignments))
        object.__setattr__(self, "u_selected_uids", tuple(self.u_selected_uids))
        object.__setattr__(self, "inner_split", _freeze_nested(self.inner_split))
        object.__setattr__(self, "policies", _freeze_nested(self.policies))
        object.__setattr__(self, "provenance", _freeze_nested(self.provenance))

    @staticmethod
    def role_policies() -> dict[SplitRole, dict[str, bool]]:
        """Return defensive copies of all role-capability policies."""

        return {
            role: dict(policy) for role, policy in _ROLE_POLICIES.items()
        }

    def policy(self, role: SplitRole) -> dict[str, bool]:
        try:
            return dict(self.policies[SplitRole(role).value])
        except (KeyError, ValueError) as exc:
            raise SplitManifestError(f"manifest has no policy for role {role!r}") from exc

    @property
    def dev_query_policy(self) -> dict[str, bool]:
        return self.policy(SplitRole.DEV_QUERY)

    def assignment(self, series_uid: str) -> SplitAssignment:
        for assignment in self.assignments:
            if assignment.series_uid == series_uid:
                return assignment
        raise KeyError(series_uid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_version": self.benchmark_version,
            "split_salt": self.split_salt,
            "assignments": [row.to_dict() for row in self.assignments],
            "u_selected_uids": list(self.u_selected_uids),
            "inner_split": dict(self.inner_split),
            "role_policies": {
                role: dict(policy)
                for role, policy in sorted(self.policies.items())
            },
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitManifest":
        """Rebuild and validate the one canonical immutable persisted form."""

        data = _require_exact_keys(
            payload,
            {
                "schema_version",
                "benchmark_version",
                "split_salt",
                "assignments",
                "u_selected_uids",
                "inner_split",
                "role_policies",
                "provenance",
            },
            "split manifest",
        )
        schema_version = _require_canonical_string(
            data["schema_version"], "schema_version"
        )
        benchmark_version = _require_canonical_string(
            data["benchmark_version"], "benchmark version"
        )
        split_salt = _require_canonical_string(data["split_salt"], "split salt")

        assignment_values = data["assignments"]
        if not isinstance(assignment_values, list):
            raise SplitManifestError("assignments must be a JSON array")
        assignments = tuple(
            _assignment_from_dict(value, index)
            for index, value in enumerate(assignment_values)
        )

        selected_values = data["u_selected_uids"]
        if not isinstance(selected_values, list):
            raise SplitManifestError("u_selected_uids must be a JSON array")
        selected = tuple(
            _require_canonical_string(value, f"U-selected uid {index}")
            for index, value in enumerate(selected_values)
        )

        inner_split = data["inner_split"]
        if not isinstance(inner_split, Mapping):
            raise SplitManifestError("inner_split must be an object")
        policies = data["role_policies"]
        if not isinstance(policies, Mapping):
            raise SplitManifestError("role_policies must be an object")
        for role_name, policy in policies.items():
            _require_canonical_string(role_name, "role policy name")
            if not isinstance(policy, Mapping):
                raise SplitManifestError(
                    f"role policy {role_name!r} must be an object"
                )
        provenance = data["provenance"]
        if not isinstance(provenance, Mapping):
            raise SplitManifestError("provenance must be an object")

        manifest = cls(
            schema_version=schema_version,
            benchmark_version=benchmark_version,
            split_salt=split_salt,
            assignments=assignments,
            u_selected_uids=selected,
            inner_split=dict(inner_split),
            policies={key: dict(value) for key, value in policies.items()},
            provenance=dict(provenance),
        )
        validate_split_manifest(manifest)
        return manifest

    @property
    def manifest_sha(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assignment_from_dict(value: Any, index: int) -> SplitAssignment:
    data = _require_exact_keys(
        value,
        {
            "series_uid",
            "dataset_id",
            "regime_tag",
            "overlap_group",
            "exposure_class",
            "length",
            "group_key",
            "group_hash_value",
            "role",
            "forced_by",
            "chronological_boundaries",
        },
        f"assignment {index}",
    )
    series_uid = _require_canonical_string(data["series_uid"], "series_uid")
    dataset_id = _require_canonical_string(data["dataset_id"], "dataset_id")
    regime_tag = _require_canonical_string(data["regime_tag"], "regime_tag")
    exposure_class = _require_canonical_string(
        data["exposure_class"], "exposure_class"
    )

    overlap_group = data["overlap_group"]
    if overlap_group is not None:
        overlap_group = _require_canonical_string(overlap_group, "overlap_group")
    length = data["length"]
    if length is not None and (
        isinstance(length, bool) or not isinstance(length, int)
    ):
        raise SplitManifestError(f"assignment {index} length must be an integer")
    group_key = _require_canonical_string(data["group_key"], "group_key")

    hash_value = _require_finite_float(
        data["group_hash_value"], f"assignment {index} group_hash_value"
    )

    role_value = _require_canonical_string(data["role"], "role")
    try:
        role = SplitRole(role_value)
    except ValueError as exc:
        raise SplitManifestError(
            f"assignment {index} has unknown role {role_value!r}"
        ) from exc

    forced_by = data["forced_by"]
    if forced_by is not None:
        forced_by = _require_canonical_string(forced_by, "forced_by")

    raw_boundaries = data["chronological_boundaries"]
    if raw_boundaries is None:
        boundaries = None
    else:
        boundary_data = _require_exact_keys(
            raw_boundaries,
            {"train", "validation", "test"},
            f"assignment {index} chronological_boundaries",
        )
        parsed: dict[str, tuple[int, int]] = {}
        for name in ("train", "validation", "test"):
            bounds = boundary_data[name]
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(isinstance(item, bool) or not isinstance(item, int) for item in bounds)
            ):
                raise SplitManifestError(
                    f"assignment {index} boundary {name!r} must contain two integers"
                )
            parsed[name] = (bounds[0], bounds[1])
        boundaries = parsed

    return SplitAssignment(
        series_uid=series_uid,
        dataset_id=dataset_id,
        regime_tag=regime_tag,
        overlap_group=overlap_group,
        exposure_class=exposure_class,
        length=length,
        group_key=group_key,
        group_hash_value=hash_value,
        role=role,
        forced_by=forced_by,
        chronological_boundaries=boundaries,
    )


def _candidate_boundaries(
    length: int | None,
) -> Mapping[str, tuple[int, int]] | None:
    if length is None:
        return None
    train_stop = length - 2 * HEADLINE_HORIZON
    validation_stop = length - HEADLINE_HORIZON
    return _freeze_nested(
        {
            "train": (0, train_stop),
            "validation": (train_stop, validation_stop),
            "test": (validation_stop, length),
        }
    )


def _validate_candidate(candidate: SplitCandidate, index: int) -> None:
    if not isinstance(candidate, SplitCandidate):
        raise SplitManifestError(
            f"candidate {index} must be SplitCandidate, got {type(candidate).__name__}"
        )
    for field_name in ("series_uid", "dataset_id", "regime_tag", "exposure_class"):
        value = getattr(candidate, field_name)
        try:
            _require_canonical_string(value, field_name)
        except SplitManifestError as exc:
            raise SplitManifestError(f"candidate {index}: {exc}") from exc
    if candidate.exposure_class not in _EXPOSURE_CLASSES:
        raise SplitManifestError(
            f"candidate {index} exposure_class is not a frozen exposure class"
        )
    if candidate.overlap_group is not None:
        try:
            _require_canonical_string(candidate.overlap_group, "overlap_group")
        except SplitManifestError as exc:
            raise SplitManifestError(
                f"candidate {candidate.series_uid!r}: {exc}"
            ) from exc
    if candidate.length is not None:
        if isinstance(candidate.length, bool) or not isinstance(candidate.length, int):
            raise SplitManifestError(
                f"candidate {candidate.series_uid!r} length must be an integer"
            )
        if candidate.length < HEADLINE_MIN_LENGTH:
            raise SplitManifestError(
                f"candidate {candidate.series_uid!r} length is below headline minimum "
                f"{HEADLINE_MIN_LENGTH}"
            )


def build_split_manifest(
    candidates: Iterable[SplitCandidate],
    benchmark_version: str,
    split_salt: str,
    u_selected_uids: Iterable[str],
) -> SplitManifest:
    """Build the deterministic five-role outer split and validate it fully."""

    _require_canonical_string(benchmark_version, "benchmark version")
    if benchmark_version != BENCHMARK_VERSION:
        raise SplitManifestError(
            "benchmark version must be exactly the current builder version "
            f"{BENCHMARK_VERSION!r}; older known versions are readable but not buildable"
        )
    _require_canonical_string(split_salt, "split salt")

    try:
        rows = list(candidates)
    except TypeError as exc:
        raise SplitManifestError("candidates must be an iterable") from exc
    if not rows:
        raise SplitManifestError("candidate set must contain at least one series")
    seen: set[str] = set()
    by_group: dict[str, list[SplitCandidate]] = {}
    for index, row in enumerate(rows):
        _validate_candidate(row, index)
        if row.series_uid in seen:
            raise SplitManifestError(f"duplicate series_uid: {row.series_uid!r}")
        seen.add(row.series_uid)
        by_group.setdefault(row.group_key, []).append(row)

    try:
        raw_selected = list(u_selected_uids)
    except TypeError as exc:
        raise SplitManifestError("U-selected uid values must be an iterable") from exc
    for index, uid in enumerate(raw_selected):
        try:
            _require_canonical_string(uid, f"U-selected uid {index}")
        except SplitManifestError as exc:
            raise SplitManifestError(f"invalid U-selected uid: {exc}") from exc
    selected = tuple(sorted(set(raw_selected)))
    unknown_u = sorted(set(selected) - seen)
    if unknown_u:
        raise SplitManifestError(f"unknown U-selected series_uid values: {unknown_u}")

    selected_set = set(selected)
    assignments: list[SplitAssignment] = []
    for group_key in sorted(by_group):
        members = by_group[group_key]
        forced_support = any(
            row.exposure_class in _FORCED_SUPPORT_A_EXPOSURES for row in members
        )
        forced_u = any(row.series_uid in selected_set for row in members)
        if forced_support and forced_u:
            raise SplitManifestError(
                f"overlap group {group_key!r} is forced to both Support-A and U"
            )

        hash_value = group_hash_value(benchmark_version, split_salt, group_key)
        if forced_support:
            role = SplitRole.SUPPORT_A
            forced_by = "exposure"
        elif forced_u:
            role = SplitRole.U
            forced_by = "u_selection"
        else:
            role = role_from_unit_interval(hash_value)
            forced_by = None

        for row in members:
            assignments.append(
                SplitAssignment(
                    series_uid=row.series_uid,
                    dataset_id=row.dataset_id,
                    regime_tag=row.regime_tag,
                    overlap_group=row.overlap_group,
                    exposure_class=row.exposure_class,
                    length=row.length,
                    group_key=group_key,
                    group_hash_value=hash_value,
                    role=role,
                    forced_by=forced_by,
                    chronological_boundaries=_candidate_boundaries(row.length),
                )
            )

    policies = {
        role.value: dict(policy) for role, policy in _ROLE_POLICIES.items()
    }
    manifest = SplitManifest(
        benchmark_version=benchmark_version,
        split_salt=split_salt,
        assignments=tuple(sorted(assignments, key=lambda row: row.series_uid)),
        u_selected_uids=selected,
        inner_split=dict(_INNER_SPLIT),
        policies=policies,
        provenance={
            "external_addendum_sha256": EXTERNAL_ADDENDUM_SHA256,
            "design_commit": DESIGN_COMMIT,
        },
    )
    validate_split_manifest(manifest)
    return manifest


def support_a_partition(
    version: str, group_key: str, *, discovery_fraction: float = SUPPORT_A_DISCOVERY_FRACTION
) -> str:
    """Deterministically place one Support-A overlap group in discovery or validation."""

    for value, name in ((version, "benchmark version"), (group_key, "group key")):
        _require_canonical_string(value, name)
    if not 0.0 < discovery_fraction < 1.0:
        raise SplitManifestError("discovery fraction must lie strictly inside (0,1)")
    raw = hashlib.sha256(
        f"{version}|support_a_subsplit|{SUPPORT_A_SUBSPLIT_SALT}|{group_key}".encode("utf-8")
    ).digest()
    value = int.from_bytes(raw[:8], "big") / float(1 << 64)
    return "support_a_discovery" if value < discovery_fraction else "support_a_validation"


def build_support_a_subsplit(
    manifest: SplitManifest,
    *,
    discovery_fraction: float = SUPPORT_A_DISCOVERY_FRACTION,
) -> dict[str, object]:
    """Partition Support-A into a search half and a promotion-gate half.

    Development promotion decisions must be made on Support-A series that were not used
    to search for the candidate in the first place.  Support-B is one-shot confirmation
    after code freeze and cannot serve as a development gate; Dev-Query is the arena's
    query side.  This partition is what a development loop is allowed to iterate on.

    From v0.2 the hash draw is overridden in one direction: any overlap group holding a
    series that is not `certified_virgin` is forced into discovery.  Those series -- the
    136 that make up the Init Harness, and every other legacy or probe-consumed row -- have
    already fed the incumbent H_ref.  Validating an update to that harness on data which
    helped form it is a closed loop: the gate would be asking the harness to be judged by
    its own training experience.  So A-validation is `certified_virgin` only, and the
    price -- a smaller, slightly less representative validation half -- is the right one to
    pay.  The exposure classes drive this, not a hard-coded uid list, so it stays correct
    if the roster changes.
    """
    if not isinstance(manifest, SplitManifest):
        raise SplitManifestError("manifest must be a SplitManifest")

    exposed_groups: set[str] = {
        row.group_key
        for row in manifest.assignments
        if row.role is SplitRole.SUPPORT_A
        and row.exposure_class in _FORCED_SUPPORT_A_EXPOSURES
    }

    partition_of_group: dict[str, str] = {}
    forced_groups: list[str] = []
    members: dict[str, list[str]] = {
        "support_a_discovery": [],
        "support_a_validation": [],
    }
    for row in manifest.assignments:
        if row.role is not SplitRole.SUPPORT_A:
            continue
        group_key = row.group_key
        if group_key not in partition_of_group:
            if group_key in exposed_groups:
                partition_of_group[group_key] = "support_a_discovery"
                forced_groups.append(group_key)
            else:
                partition_of_group[group_key] = support_a_partition(
                    manifest.benchmark_version,
                    group_key,
                    discovery_fraction=discovery_fraction,
                )
        members[partition_of_group[group_key]].append(row.series_uid)

    if not partition_of_group:
        raise SplitManifestError("split manifest has no Support-A groups to partition")
    for name, uids in members.items():
        if not uids:
            raise SplitManifestError(f"Support-A partition {name!r} came out empty")

    virgin_uids = {
        row.series_uid
        for row in manifest.assignments
        if row.role is SplitRole.SUPPORT_A and row.exposure_class == "certified_virgin"
    }
    contaminated = sorted(set(members["support_a_validation"]) - virgin_uids)
    if contaminated:
        raise SplitManifestError(
            "A-validation must hold certified_virgin series only; found "
            f"{len(contaminated)} exposed: {contaminated[:5]}"
        )

    return {
        "schema_version": "benchmark-support-a-subsplit/2",
        "benchmark_version": manifest.benchmark_version,
        "subsplit_salt": SUPPORT_A_SUBSPLIT_SALT,
        "discovery_fraction": float(discovery_fraction),
        "unit": "overlap_group",
        "distinct_from": (
            "the chronological train/validation/test boundaries inside each series; "
            "this partitions the Support-A population, not any series' timeline"
        ),
        "usage": {
            "support_a_discovery": "search, propose, fit; iterate freely",
            "support_a_validation": (
                "development promotion gate; never used to select the candidate it judges"
            ),
        },
        "validation_exposure_rule": (
            "certified_virgin only. Any overlap group holding a confirmed_exposed, "
            "uncertain_legacy_exposure, or probe_consumed series is forced to discovery, "
            "because those series already fed the incumbent harness and cannot be used to "
            "validate an update to it."
        ),
        "n_groups": len(partition_of_group),
        "n_groups_forced_to_discovery_by_exposure": len(forced_groups),
        "groups_forced_to_discovery_by_exposure": sorted(forced_groups),
        "counts": {name: len(uids) for name, uids in sorted(members.items())},
        "members": {name: sorted(uids) for name, uids in sorted(members.items())},
    }


def validate_split_manifest(manifest: SplitManifest) -> None:
    """Reject tampering or any placement inconsistent with frozen rules."""

    if not isinstance(manifest, SplitManifest):
        raise SplitManifestError("manifest must be a SplitManifest")
    if manifest.schema_version != _SCHEMA_VERSION:
        raise SplitManifestError(f"unsupported schema version {manifest.schema_version!r}")
    _require_canonical_string(manifest.benchmark_version, "benchmark version")
    # Readers accept any known version so the sealed, never-opened benchmark-v0 Final
    # split stays auditable after v0.1 supersedes it. Builders still emit only current.
    if manifest.benchmark_version not in KNOWN_BENCHMARK_VERSIONS:
        raise SplitManifestError(
            f"benchmark version must be one of {KNOWN_BENCHMARK_VERSIONS!r}"
        )
    _require_canonical_string(manifest.split_salt, "split salt")
    if not isinstance(manifest.inner_split, Mapping):
        raise SplitManifestError("inner_split must be an object")
    if dict(manifest.inner_split) != _INNER_SPLIT:
        raise SplitManifestError("chronological inner-split rule differs from benchmark-v0")
    expected_policies = {
        role.value: dict(policy) for role, policy in _ROLE_POLICIES.items()
    }
    if not isinstance(manifest.policies, Mapping):
        raise SplitManifestError("role policies must be an object")
    actual_policies: dict[str, dict[str, Any]] = {}
    for key, value in manifest.policies.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise SplitManifestError("role policies contain malformed entries")
        actual_policies[key] = dict(value)
    if actual_policies != expected_policies:
        raise SplitManifestError("role policies differ from the frozen protocol")
    if not isinstance(manifest.provenance, Mapping):
        raise SplitManifestError("design provenance must be an object")
    if dict(manifest.provenance) != {
        "external_addendum_sha256": EXTERNAL_ADDENDUM_SHA256,
        "design_commit": DESIGN_COMMIT,
    }:
        raise SplitManifestError("design provenance differs from the frozen protocol")

    rows = list(manifest.assignments)
    if not rows:
        raise SplitManifestError("manifest must contain at least one assignment")
    for index, row in enumerate(rows):
        if not isinstance(row, SplitAssignment):
            raise SplitManifestError(
                f"assignment {index} must be a SplitAssignment, "
                f"got {type(row).__name__}"
            )
        _validate_candidate(
            SplitCandidate(
                row.series_uid,
                row.dataset_id,
                row.regime_tag,
                row.overlap_group,
                row.exposure_class,
                row.length,
            ),
            index,
        )
    uids = [row.series_uid for row in rows]
    if len(uids) != len(set(uids)):
        raise SplitManifestError("manifest contains duplicate series_uid values")
    if uids != sorted(uids):
        raise SplitManifestError("manifest assignments are not in canonical uid order")
    for index, uid in enumerate(manifest.u_selected_uids):
        try:
            _require_canonical_string(uid, f"U-selected uid {index}")
        except SplitManifestError as exc:
            raise SplitManifestError(f"invalid U-selected uid: {exc}") from exc
    selected = tuple(sorted(set(manifest.u_selected_uids)))
    if selected != manifest.u_selected_uids:
        raise SplitManifestError("U-selected uid list is not canonical and unique")
    unknown_u = sorted(set(selected) - set(uids))
    if unknown_u:
        raise SplitManifestError(f"manifest has unknown U-selected uid values: {unknown_u}")

    by_group: dict[str, list[SplitAssignment]] = {}
    for row in rows:
        candidate = SplitCandidate(
            row.series_uid,
            row.dataset_id,
            row.regime_tag,
            row.overlap_group,
            row.exposure_class,
            row.length,
        )
        _validate_candidate(candidate, 0)
        _require_canonical_string(row.group_key, "group_key")
        if row.group_key != candidate.group_key:
            raise SplitManifestError(
                f"assignment {row.series_uid!r} has a non-canonical group key"
            )
        _require_finite_float(
            row.group_hash_value,
            f"assignment {row.series_uid!r} group hash value",
        )
        if not isinstance(row.role, SplitRole):
            raise SplitManifestError(
                f"assignment {row.series_uid!r} has an invalid role"
            )
        if row.forced_by is not None:
            _require_canonical_string(row.forced_by, "forced_by")
        if row.chronological_boundaries != _candidate_boundaries(row.length):
            raise SplitManifestError(
                f"assignment {row.series_uid!r} has invalid chronological boundaries"
            )
        by_group.setdefault(row.group_key, []).append(row)

    selected_set = set(selected)
    for group_key, members in by_group.items():
        roles = {row.role for row in members}
        hash_values = {
            _require_finite_float(row.group_hash_value, "group hash value")
            for row in members
        }
        forced_values = {row.forced_by for row in members}
        if len(roles) != 1 or len(hash_values) != 1 or len(forced_values) != 1:
            raise SplitManifestError(f"overlap group {group_key!r} is not atomic")

        forced_support = any(
            row.exposure_class in _FORCED_SUPPORT_A_EXPOSURES for row in members
        )
        forced_u = any(row.series_uid in selected_set for row in members)
        if forced_support and forced_u:
            raise SplitManifestError(
                f"overlap group {group_key!r} is forced to both Support-A and U"
            )
        expected_hash = group_hash_value(
            manifest.benchmark_version, manifest.split_salt, group_key
        )
        actual_hash = _require_finite_float(
            members[0].group_hash_value, "group hash value"
        )
        if actual_hash != expected_hash:
            raise SplitManifestError(f"overlap group {group_key!r} hash value is invalid")
        if forced_support:
            expected_role, expected_forced = SplitRole.SUPPORT_A, "exposure"
        elif forced_u:
            expected_role, expected_forced = SplitRole.U, "u_selection"
        else:
            expected_role = role_from_unit_interval(expected_hash)
            expected_forced = None
        if members[0].role is not expected_role or members[0].forced_by != expected_forced:
            raise SplitManifestError(
                f"overlap group {group_key!r} placement differs from frozen rules"
            )
