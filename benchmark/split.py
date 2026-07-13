"""Deterministic outer-role allocation and manifest validation.

The split is an immutable protocol decision rather than an RNG draw.  Every
atomic overlap group receives its role from a canonical SHA256 value unless
exposure or the frozen U selection forces the whole group into a safer role.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from . import (
    DESIGN_COMMIT,
    EXTERNAL_ADDENDUM_SHA256,
    HEADLINE_HORIZON,
    HEADLINE_LOOKBACK,
    HEADLINE_MIN_LENGTH,
)

__all__ = [
    "SplitAssignment",
    "SplitCandidate",
    "SplitManifest",
    "SplitManifestError",
    "SplitRole",
    "build_split_manifest",
    "group_hash_value",
    "role_from_unit_interval",
    "validate_split_manifest",
]


class SplitManifestError(ValueError):
    """A candidate set or split manifest violates the frozen protocol."""


class SplitRole(str, Enum):
    SUPPORT_A = "support_a"
    SUPPORT_B = "support_b"
    DEV_QUERY = "dev_query"
    FINAL_QUERY = "final_query"
    U = "u"


_FORCED_SUPPORT_A_EXPOSURES = frozenset(
    {"confirmed_exposed", "uncertain_legacy_exposure"}
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

    if not isinstance(version, str) or not version:
        raise ValueError("benchmark version must be a non-empty string")
    if not isinstance(salt, str) or not salt:
        raise ValueError("split salt must be a non-empty string")
    if not isinstance(group_key, str) or not group_key:
        raise ValueError("group key must be a non-empty string")
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
    chronological_boundaries: Mapping[str, list[int]] | None

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
    schema_version: str = "benchmark-split-manifest/1"

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

    @property
    def manifest_sha(self) -> str:
        canonical = json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _candidate_boundaries(length: int | None) -> Mapping[str, list[int]] | None:
    if length is None:
        return None
    train_stop = length - 2 * HEADLINE_HORIZON
    validation_stop = length - HEADLINE_HORIZON
    return {
        "train": [0, train_stop],
        "validation": [train_stop, validation_stop],
        "test": [validation_stop, length],
    }


def _validate_candidate(candidate: SplitCandidate, index: int) -> None:
    if not isinstance(candidate, SplitCandidate):
        raise SplitManifestError(
            f"candidate {index} must be SplitCandidate, got {type(candidate).__name__}"
        )
    for field_name in ("series_uid", "dataset_id", "regime_tag", "exposure_class"):
        value = getattr(candidate, field_name)
        if not isinstance(value, str) or not value.strip():
            raise SplitManifestError(
                f"candidate {index} has empty or invalid {field_name}"
            )
    if candidate.overlap_group is not None and (
        not isinstance(candidate.overlap_group, str)
        or not candidate.overlap_group.strip()
    ):
        raise SplitManifestError(
            f"candidate {candidate.series_uid!r} has an invalid overlap_group"
        )
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

    if not isinstance(benchmark_version, str) or not benchmark_version:
        raise SplitManifestError("benchmark version must be a non-empty string")
    if not isinstance(split_salt, str) or not split_salt:
        raise SplitManifestError("split salt must be a non-empty string")

    rows = list(candidates)
    seen: set[str] = set()
    by_group: dict[str, list[SplitCandidate]] = {}
    for index, row in enumerate(rows):
        _validate_candidate(row, index)
        if row.series_uid in seen:
            raise SplitManifestError(f"duplicate series_uid: {row.series_uid!r}")
        seen.add(row.series_uid)
        by_group.setdefault(row.group_key, []).append(row)

    selected = tuple(sorted(set(u_selected_uids)))
    if any(not isinstance(uid, str) or not uid for uid in selected):
        raise SplitManifestError("U-selected uid values must be non-empty strings")
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


def validate_split_manifest(manifest: SplitManifest) -> None:
    """Reject tampering or any placement inconsistent with frozen rules."""

    if not isinstance(manifest, SplitManifest):
        raise SplitManifestError("manifest must be a SplitManifest")
    if manifest.schema_version != "benchmark-split-manifest/1":
        raise SplitManifestError(f"unsupported schema version {manifest.schema_version!r}")
    if not manifest.benchmark_version or not manifest.split_salt:
        raise SplitManifestError("manifest version and split salt must be non-empty")
    if dict(manifest.inner_split) != _INNER_SPLIT:
        raise SplitManifestError("chronological inner-split rule differs from benchmark-v0")
    expected_policies = {
        role.value: dict(policy) for role, policy in _ROLE_POLICIES.items()
    }
    if {key: dict(value) for key, value in manifest.policies.items()} != expected_policies:
        raise SplitManifestError("role policies differ from the frozen protocol")
    if dict(manifest.provenance) != {
        "external_addendum_sha256": EXTERNAL_ADDENDUM_SHA256,
        "design_commit": DESIGN_COMMIT,
    }:
        raise SplitManifestError("design provenance differs from the frozen protocol")

    rows = list(manifest.assignments)
    uids = [row.series_uid for row in rows]
    if len(uids) != len(set(uids)):
        raise SplitManifestError("manifest contains duplicate series_uid values")
    if uids != sorted(uids):
        raise SplitManifestError("manifest assignments are not in canonical uid order")
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
        if row.group_key != candidate.group_key:
            raise SplitManifestError(
                f"assignment {row.series_uid!r} has a non-canonical group key"
            )
        if row.chronological_boundaries != _candidate_boundaries(row.length):
            raise SplitManifestError(
                f"assignment {row.series_uid!r} has invalid chronological boundaries"
            )
        by_group.setdefault(row.group_key, []).append(row)

    selected_set = set(selected)
    for group_key, members in by_group.items():
        roles = {row.role for row in members}
        hash_values = {row.group_hash_value for row in members}
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
        actual_hash = members[0].group_hash_value
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
