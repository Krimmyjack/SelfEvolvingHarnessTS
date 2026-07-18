from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.observables import OBSERVABLE_FEATURES


_PUBLIC_CASE_ID = re.compile(r"^m0-[0-9]{4}$")
_PRIVATE_FAMILIES = frozenset(
    {"missing", "impulsive_outlier", "level_shift", "period_change"}
)


def _readonly_array(values: object, *, length: int, field: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != length:
        raise ValueError(f"{field} must be a one-dimensional array of length {length}")
    result = np.asarray(array, dtype="<f8").copy()
    result.setflags(write=False)
    return result


def _array_sha(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _missing_runs(values: np.ndarray) -> list[list[int]]:
    indices = np.flatnonzero(~np.isfinite(values))
    if indices.size == 0:
        return []
    runs: list[list[int]] = []
    start = previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            runs.append([start, previous + 1])
            start = index
        previous = index
    runs.append([start, previous + 1])
    return runs


def serialize_series(values: np.ndarray) -> dict[str, object]:
    return {
        "length": int(values.size),
        "finite_values": [
            [int(index), float(value)]
            for index, value in enumerate(values)
            if math.isfinite(float(value))
        ],
        "absent_runs": _missing_runs(values),
        "float64_sha": _array_sha(values),
    }


def _validate_public_features(features: Mapping[str, object]) -> None:
    for key, value in features.items():
        if key not in OBSERVABLE_FEATURES:
            raise ValueError(f"unknown observable feature: {key}")
        kind = OBSERVABLE_FEATURES[key]
        if kind == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"observable feature {key} must be a finite number")
        elif kind == "boolean" and not isinstance(value, bool):
            raise ValueError(f"observable feature {key} must be boolean")
        elif kind == "string" and not isinstance(value, str):
            raise ValueError(f"observable feature {key} must be string")


def _public_sha_payload(
    *,
    case_id: str,
    values: np.ndarray,
    task_kind: str,
    public_features: Mapping[str, object],
    public_probe_panel: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": "public-case-view/1",
        "case_id": case_id,
        "values_float64_sha": _array_sha(values),
        "task_kind": task_kind,
        "public_features": _plain(public_features),
        "public_probe_panel": _plain(public_probe_panel),
    }


class CasePurpose(str, Enum):
    TARGET = "target"
    RISK_CLEAN = "risk_clean"
    RISK_GENUINE_EVENT = "risk_genuine_event"


@dataclass(frozen=True)
class PublicCaseView:
    schema_version: str
    case_id: str
    values: np.ndarray
    task_kind: str
    public_features: Mapping[str, object]
    public_probe_panel: Mapping[str, object] | None
    public_case_view_sha: str

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        values: object,
        task_kind: str,
        public_features: Mapping[str, object],
    ) -> "PublicCaseView":
        if not _PUBLIC_CASE_ID.fullmatch(case_id):
            raise ValueError("public case_id must be an opaque m0-XXXX identifier")
        array = _readonly_array(values, length=192, field="public values")
        _validate_public_features(public_features)
        features = _freeze_json(public_features)
        digest = canonical_sha256(
            _public_sha_payload(
                case_id=case_id,
                values=array,
                task_kind=task_kind,
                public_features=features,
                public_probe_panel=None,
            )
        )
        return cls(
            schema_version="public-case-view/1",
            case_id=case_id,
            values=array,
            task_kind=task_kind,
            public_features=features,
            public_probe_panel=None,
            public_case_view_sha=digest,
        )

    def with_features(self, features: Mapping[str, object]) -> "PublicCaseView":
        _validate_public_features(features)
        frozen = _freeze_json(features)
        digest = canonical_sha256(
            _public_sha_payload(
                case_id=self.case_id,
                values=self.values,
                task_kind=self.task_kind,
                public_features=frozen,
                public_probe_panel=self.public_probe_panel,
            )
        )
        return replace(self, public_features=frozen, public_case_view_sha=digest)

    def with_probe_panel(self, public_receipt: object) -> "PublicCaseView":
        receipt_type = type(public_receipt).__name__.lower()
        if "private" in receipt_type or not hasattr(public_receipt, "to_public_dict"):
            raise TypeError("with_probe_panel accepts a strict public probe receipt only")
        panel = public_receipt.to_public_dict()
        if not isinstance(panel, Mapping) or panel.get("schema_version") != "public-probe-panel/1":
            raise TypeError("invalid public probe receipt serializer")
        frozen = _freeze_json(panel)
        digest = canonical_sha256(
            _public_sha_payload(
                case_id=self.case_id,
                values=self.values,
                task_kind=self.task_kind,
                public_features=self.public_features,
                public_probe_panel=frozen,
            )
        )
        return replace(self, public_probe_panel=frozen, public_case_view_sha=digest)

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "series": serialize_series(self.values),
            "task_kind": self.task_kind,
            "public_features": _plain(self.public_features),
            "public_probe_panel": _plain(self.public_probe_panel),
            "public_case_view_sha": self.public_case_view_sha,
        }


@dataclass(frozen=True)
class PrivateSyntheticCase:
    case_id: str
    seed: int
    purpose: CasePurpose
    private_family: str
    private_severity: str
    clean_context: np.ndarray
    corrupt_context: np.ndarray
    clean_future: np.ndarray
    oracle_affected_indices: tuple[int, ...]
    observable_counterpart_id: str | None
    private_sha: str

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        seed: int,
        purpose: CasePurpose,
        private_family: str,
        private_severity: str,
        clean_context: object,
        corrupt_context: object,
        clean_future: object,
        oracle_affected_indices: Sequence[int],
        observable_counterpart_id: str | None,
    ) -> "PrivateSyntheticCase":
        if not _PUBLIC_CASE_ID.fullmatch(case_id):
            raise ValueError("private case_id must use the opaque public ID")
        if private_family not in _PRIVATE_FAMILIES:
            raise ValueError("unknown private synthetic family")
        clean = _readonly_array(clean_context, length=192, field="clean_context")
        corrupt = _readonly_array(corrupt_context, length=192, field="corrupt_context")
        future = _readonly_array(clean_future, length=48, field="clean_future")
        affected = tuple(int(index) for index in oracle_affected_indices)
        if affected != tuple(sorted(set(affected))) or any(
            index < 0 or index >= 192 for index in affected
        ):
            raise ValueError("oracle affected indices must be sorted unique context indices")
        if purpose is not CasePurpose.TARGET and affected:
            raise ValueError("risk cases cannot carry a repair target")
        metadata = {
            "schema_version": "private-synthetic-case/1",
            "case_id": case_id,
            "seed": seed,
            "purpose": purpose.value,
            "private_family": private_family,
            "private_severity": private_severity,
            "clean_context_sha": _array_sha(clean),
            "corrupt_context_sha": _array_sha(corrupt),
            "clean_future_sha": _array_sha(future),
            "oracle_affected_indices": list(affected),
            "observable_counterpart_id": observable_counterpart_id,
        }
        return cls(
            case_id=case_id,
            seed=int(seed),
            purpose=purpose,
            private_family=private_family,
            private_severity=private_severity,
            clean_context=clean,
            corrupt_context=corrupt,
            clean_future=future,
            oracle_affected_indices=affected,
            observable_counterpart_id=observable_counterpart_id,
            private_sha=canonical_sha256(metadata),
        )

    def to_public_view(self) -> PublicCaseView:
        return PublicCaseView.create(
            case_id=self.case_id,
            values=self.corrupt_context,
            task_kind="forecast",
            public_features={},
        )

    def to_private_json(self) -> dict[str, object]:
        return {
            "schema_version": "private-synthetic-case/1",
            "case_id": self.case_id,
            "seed": self.seed,
            "purpose": self.purpose.value,
            "private_family": self.private_family,
            "private_severity": self.private_severity,
            "clean_context": serialize_series(self.clean_context),
            "corrupt_context": serialize_series(self.corrupt_context),
            "clean_future": serialize_series(self.clean_future),
            "oracle_affected_indices": list(self.oracle_affected_indices),
            "observable_counterpart_id": self.observable_counterpart_id,
            "private_sha": self.private_sha,
        }


@dataclass(frozen=True)
class ArtifactRoots:
    public: Path
    private: Path

    @classmethod
    def create(cls, run_root: Path) -> "ArtifactRoots":
        artifact_root = Path(run_root).resolve() / "artifacts"
        public = artifact_root / "public"
        private = artifact_root / "private"
        public.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        return cls(public=public, private=private)


__all__ = [
    "ArtifactRoots",
    "CasePurpose",
    "PrivateSyntheticCase",
    "PublicCaseView",
    "serialize_series",
]
