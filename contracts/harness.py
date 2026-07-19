from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from .canonical import canonical_json_bytes
from .observables import validate_applicability


_CANONICAL_ID = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_DEPLOYABLE_FIELDS = frozenset(
    {
        "case_id",
        "injection_type",
        "injection_indices",
        "D",
        "G",
        "J",
        "pattern_id",
        "private_receipt",
    }
)


def _require_canonical_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _CANONICAL_ID.fullmatch(value):
        raise ValueError(f"{field} must be a canonical identifier")
    return value


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _reject_forbidden_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in _FORBIDDEN_DEPLOYABLE_FIELDS:
                raise ValueError(f"forbidden deployable field: {key}")
            _reject_forbidden_fields(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_forbidden_fields(nested)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _require_exact_fields(
    payload: Mapping[str, object],
    expected: frozenset[str],
    *,
    artifact: str,
) -> None:
    missing = expected - set(payload)
    extra = set(payload) - expected
    if missing:
        raise ValueError(f"{artifact} missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{artifact} has unexpected fields: {sorted(extra)}")


def _require_body(value: object, *, artifact: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{artifact} body must be non-empty UTF-8 text without NUL")
    return value


def _require_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("revision must be a positive integer")
    return value


def _require_string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or item != item.strip():
            raise ValueError(f"{field} entries must be canonical non-empty strings")
        result.append(item)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} entries must be unique")
    return tuple(result)


class SkillKind(str, Enum):
    BOOTSTRAP_PROCEDURE = "bootstrap_procedure"
    CAPABILITY = "capability"
    SAFETY = "safety"


@dataclass(frozen=True)
class SkillEntry:
    schema_version: str
    skill_id: str
    skill_kind: SkillKind
    revision: int
    body: str
    observable_applicability: Mapping[str, object]
    allowed_tools: tuple[str, ...]
    risk_guards: Mapping[str, object]


@dataclass(frozen=True)
class MemoryEntry:
    schema_version: str
    memory_id: str
    revision: int
    body: str
    observable_applicability: Mapping[str, object]
    risk_guards: Mapping[str, object]


@dataclass(frozen=True)
class HarnessSnapshot:
    schema_version: str
    instruction: str
    skills: tuple[SkillEntry, ...]
    memories: tuple[MemoryEntry, ...]
    retrieval: Mapping[str, object]
    candidate_policy: Mapping[str, object]
    verification: Mapping[str, object]
    dependency_shas: Mapping[str, str]
    harness_content_sha: str
    runtime_bundle_sha: str

    def __post_init__(self) -> None:
        if self.schema_version != "harness-snapshot/1":
            raise ValueError("HarnessSnapshot schema_version must be harness-snapshot/1")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("HarnessSnapshot instruction must be non-empty")
        if not isinstance(self.skills, tuple) or not all(
            isinstance(skill, SkillEntry) for skill in self.skills
        ):
            raise ValueError("HarnessSnapshot skills must be a tuple of SkillEntry")
        if not isinstance(self.memories, tuple) or not all(
            isinstance(memory, MemoryEntry) for memory in self.memories
        ):
            raise ValueError("HarnessSnapshot memories must be a tuple of MemoryEntry")
        skill_ids = [skill.skill_id for skill in self.skills]
        memory_ids = [memory.memory_id for memory in self.memories]
        if len(skill_ids) != len(set(skill_ids)) or len(memory_ids) != len(set(memory_ids)):
            raise ValueError("HarnessSnapshot entry IDs must be unique")
        _require_sha(self.harness_content_sha, field="harness_content_sha")
        _require_sha(self.runtime_bundle_sha, field="runtime_bundle_sha")
        for key, digest in self.dependency_shas.items():
            if not isinstance(key, str) or not key:
                raise ValueError("dependency SHA names must be non-empty strings")
            _require_sha(digest, field=f"dependency_shas[{key}]")
        object.__setattr__(self, "retrieval", _freeze_json(self.retrieval))
        object.__setattr__(self, "candidate_policy", _freeze_json(self.candidate_policy))
        object.__setattr__(self, "verification", _freeze_json(self.verification))
        object.__setattr__(self, "dependency_shas", _freeze_json(self.dependency_shas))


class EditOperation(str, Enum):
    PATCH = "PATCH"
    ADD = "ADD"


@dataclass(frozen=True)
class EditManifest:
    edit_id: str
    base_harness_sha: str
    target_pattern_id: str
    target_surface_id: str
    operation: EditOperation
    surface_precondition: Mapping[str, object]
    dependency_precondition_shas: Mapping[str, str]
    minimal_patch: Mapping[str, object] | None = None
    new_value: Mapping[str, object] | None = None
    observable_applicability: Mapping[str, object] | None = None
    predicted_agent_behavior_change: tuple[str, ...] = ()
    predicted_data_effect: tuple[str, ...] = ()
    automatically_selected_risk_cases: tuple[str, ...] = ()
    falsification_condition: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_canonical_id(self.edit_id, field="edit_id")
        _require_sha(self.base_harness_sha, field="base_harness_sha")
        _require_canonical_id(self.target_pattern_id, field="target_pattern_id")
        if (
            not isinstance(self.target_surface_id, str)
            or not self.target_surface_id
            or self.target_surface_id != self.target_surface_id.strip()
        ):
            raise ValueError("target_surface_id must be canonical")
        operation = self.operation
        if isinstance(operation, str) and not isinstance(operation, EditOperation):
            try:
                operation = EditOperation(operation)
            except ValueError as exc:
                raise ValueError("unsupported edit operation") from exc
            object.__setattr__(self, "operation", operation)
        if not isinstance(self.surface_precondition, Mapping):
            raise ValueError("surface_precondition must be an object")
        precondition_kind = self.surface_precondition.get("kind")
        if operation is EditOperation.ADD:
            if precondition_kind != "ABSENT" or set(self.surface_precondition) != {"kind"}:
                raise ValueError("ADD edit requires ABSENT surface precondition")
            if self.new_value is None or self.minimal_patch is not None:
                raise ValueError("ADD edit requires new_value only")
        elif operation is EditOperation.PATCH:
            if precondition_kind != "SHA":
                raise ValueError("PATCH edit requires SHA surface precondition")
            _require_sha(self.surface_precondition.get("sha"), field="surface_precondition.sha")
            if self.minimal_patch is None or self.new_value is not None:
                raise ValueError("PATCH edit requires minimal_patch only")
        else:
            raise ValueError("unsupported edit operation")
        for key, digest in self.dependency_precondition_shas.items():
            if not isinstance(key, str) or not key:
                raise ValueError("dependency precondition names must be non-empty strings")
            _require_sha(digest, field=f"dependency_precondition_shas[{key}]")
        deployable = self.new_value if self.new_value is not None else self.minimal_patch
        _reject_forbidden_fields(deployable)
        canonical_json_bytes(deployable)
        if self.observable_applicability is not None:
            validate_applicability(self.observable_applicability)
        for field in (
            "predicted_agent_behavior_change",
            "predicted_data_effect",
            "automatically_selected_risk_cases",
            "falsification_condition",
        ):
            value = _require_string_tuple(getattr(self, field), field=field)
            object.__setattr__(self, field, value)
        object.__setattr__(self, "surface_precondition", _freeze_json(self.surface_precondition))
        object.__setattr__(
            self,
            "dependency_precondition_shas",
            _freeze_json(self.dependency_precondition_shas),
        )
        if self.minimal_patch is not None:
            object.__setattr__(self, "minimal_patch", _freeze_json(self.minimal_patch))
        if self.new_value is not None:
            object.__setattr__(self, "new_value", _freeze_json(self.new_value))
        if self.observable_applicability is not None:
            object.__setattr__(
                self,
                "observable_applicability",
                _freeze_json(self.observable_applicability),
            )


_SKILL_FIELDS = frozenset(
    {
        "schema_version",
        "skill_id",
        "skill_kind",
        "revision",
        "body",
        "observable_applicability",
        "allowed_tools",
        "risk_guards",
    }
)
_MEMORY_FIELDS = frozenset(
    {
        "schema_version",
        "memory_id",
        "revision",
        "body",
        "observable_applicability",
        "risk_guards",
    }
)


def load_skill_entry(payload: Mapping[str, object]) -> SkillEntry:
    if not isinstance(payload, Mapping):
        raise ValueError("SkillEntry must be an object")
    _reject_forbidden_fields(payload)
    _require_exact_fields(payload, _SKILL_FIELDS, artifact="SkillEntry")
    if payload["schema_version"] != "skill-entry/1":
        raise ValueError("SkillEntry schema_version must be skill-entry/1")
    skill_id = _require_canonical_id(payload["skill_id"], field="skill_id")
    try:
        kind = SkillKind(payload["skill_kind"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown skill_kind") from exc
    revision = _require_revision(payload["revision"])
    body = _require_body(payload["body"], artifact="SkillEntry")
    applicability = payload["observable_applicability"]
    if not isinstance(applicability, Mapping):
        raise ValueError("observable_applicability must be an object")
    validate_applicability(applicability)
    allowed_tools = _require_string_tuple(payload["allowed_tools"], field="allowed_tools")
    for tool in allowed_tools:
        _require_canonical_id(tool, field="allowed_tools entry")
    risk_guards = payload["risk_guards"]
    if not isinstance(risk_guards, Mapping):
        raise ValueError("risk_guards must be an object")
    canonical_json_bytes(risk_guards)
    return SkillEntry(
        schema_version="skill-entry/1",
        skill_id=skill_id,
        skill_kind=kind,
        revision=revision,
        body=body,
        observable_applicability=_freeze_json(applicability),
        allowed_tools=allowed_tools,
        risk_guards=_freeze_json(risk_guards),
    )


def load_learned_skill_entry(payload: Mapping[str, object]) -> SkillEntry:
    skill = load_skill_entry(payload)
    if skill.skill_kind is not SkillKind.CAPABILITY:
        raise ValueError("learned SkillEntry must have skill_kind=capability")
    return skill


def load_memory_entry(payload: Mapping[str, object]) -> MemoryEntry:
    if not isinstance(payload, Mapping):
        raise ValueError("MemoryEntry must be an object")
    _reject_forbidden_fields(payload)
    _require_exact_fields(payload, _MEMORY_FIELDS, artifact="MemoryEntry")
    if payload["schema_version"] != "memory-entry/1":
        raise ValueError("MemoryEntry schema_version must be memory-entry/1")
    memory_id = _require_canonical_id(payload["memory_id"], field="memory_id")
    revision = _require_revision(payload["revision"])
    body = _require_body(payload["body"], artifact="MemoryEntry")
    applicability = payload["observable_applicability"]
    if not isinstance(applicability, Mapping):
        raise ValueError("observable_applicability must be an object")
    validate_applicability(applicability)
    risk_guards = payload["risk_guards"]
    if not isinstance(risk_guards, Mapping):
        raise ValueError("risk_guards must be an object")
    canonical_json_bytes(risk_guards)
    return MemoryEntry(
        schema_version="memory-entry/1",
        memory_id=memory_id,
        revision=revision,
        body=body,
        observable_applicability=_freeze_json(applicability),
        risk_guards=_freeze_json(risk_guards),
    )


__all__ = [
    "EditManifest",
    "EditOperation",
    "HarnessSnapshot",
    "MemoryEntry",
    "SkillEntry",
    "SkillKind",
    "load_learned_skill_entry",
    "load_memory_entry",
    "load_skill_entry",
]
