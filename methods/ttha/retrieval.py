from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.contracts.harness import (
    HarnessSnapshot,
    MemoryEntry,
    SkillEntry,
    SkillKind,
)
from SelfEvolvingHarnessTS.contracts.observables import (
    observable_numeric_bin,
    validate_applicability,
)


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
    return value


_NUMERIC_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


def _evaluate(
    ast: Mapping[str, object],
    public_features: Mapping[str, object],
) -> tuple[bool | None, int]:
    if set(ast) == {"const"}:
        return bool(ast["const"]), 0
    if set(ast) in ({"all"}, {"any"}):
        key = next(iter(ast))
        results = [_evaluate(child, public_features) for child in ast[key]]
        score = sum(nested_score for _, nested_score in results)
        states = [state for state, _ in results]
        if key == "all":
            if False in states:
                return False, score
            return (None if None in states else True), score
        if True in states:
            return True, score
        return (None if None in states else False), score
    if set(ast) == {"not"}:
        state, score = _evaluate(ast["not"], public_features)
        return (None if state is None else not state), score
    feature = ast["feature"]
    if feature not in public_features:
        return None, 0
    actual = public_features[feature]
    expected = ast["value"]
    operation = ast["op"]
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if isinstance(expected, str):
            actual = observable_numeric_bin(str(feature), float(actual))
        elif operation == "in" and isinstance(expected, Sequence):
            actual = observable_numeric_bin(str(feature), float(actual))
    if operation == "in":
        return bool(actual in expected), 1
    if operation in _NUMERIC_OPERATORS:
        try:
            return bool(_NUMERIC_OPERATORS[operation](actual, expected)), 1
        except TypeError:
            return False, 1
    return False, 1


def evaluate_applicability(
    ast: Mapping[str, object],
    public_features: Mapping[str, object],
) -> tuple[bool, int]:
    validate_applicability(ast)
    matched, score = _evaluate(ast, public_features)
    return matched is True, score


def _skill_payload(skill: SkillEntry) -> dict[str, object]:
    return {
        "schema_version": skill.schema_version,
        "skill_id": skill.skill_id,
        "skill_kind": skill.skill_kind.value,
        "revision": skill.revision,
        "body": skill.body,
        "observable_applicability": _plain(skill.observable_applicability),
        "allowed_tools": list(skill.allowed_tools),
        "risk_guards": _plain(skill.risk_guards),
    }


def _memory_payload(memory: MemoryEntry) -> dict[str, object]:
    return {
        "schema_version": memory.schema_version,
        "memory_id": memory.memory_id,
        "revision": memory.revision,
        "body": memory.body,
        "observable_applicability": _plain(memory.observable_applicability),
        "risk_guards": _plain(memory.risk_guards),
    }


@dataclass(frozen=True)
class EffectiveHarnessView:
    instruction: str
    skills: tuple[SkillEntry, ...]
    memories: tuple[MemoryEntry, ...]
    controls: Mapping[str, object]
    effective_harness_view_sha: str

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(skill.skill_id for skill in self.skills)

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(memory.memory_id for memory in self.memories)


def resolve_harness_view(
    snapshot: HarnessSnapshot,
    public_features: Mapping[str, object],
    *,
    role: str = "fast",
) -> EffectiveHarnessView:
    if role not in {"fast", "slow"}:
        raise ValueError("role must be fast or slow")
    bootstrap = sorted(
        (
            skill
            for skill in snapshot.skills
            if skill.skill_kind is SkillKind.BOOTSTRAP_PROCEDURE
        ),
        key=lambda skill: skill.skill_id,
    )
    capabilities: list[tuple[int, SkillEntry]] = []
    all_capabilities: list[SkillEntry] = []
    safety: list[SkillEntry] = []
    for skill in snapshot.skills:
        if skill.skill_kind is SkillKind.BOOTSTRAP_PROCEDURE:
            continue
        if skill.skill_kind is SkillKind.CAPABILITY:
            all_capabilities.append(skill)
        matched, score = evaluate_applicability(
            skill.observable_applicability, public_features
        )
        if not matched:
            continue
        if skill.skill_kind is SkillKind.CAPABILITY:
            capabilities.append((score, skill))
        else:
            safety.append(skill)
    retrieval = snapshot.retrieval
    capability_rule = retrieval.get("capability", {})
    top_k = capability_rule.get("top_k", 0) if isinstance(capability_rule, Mapping) else 0
    ranked_capabilities = (
        sorted(all_capabilities, key=lambda skill: skill.skill_id)
        if role == "slow"
        else [
            skill
            for _, skill in sorted(
                capabilities,
                key=lambda item: (-item[0], item[1].skill_id),
            )[: int(top_k)]
        ]
    )
    selected_skills = tuple(
        [*bootstrap, *ranked_capabilities, *sorted(safety, key=lambda skill: skill.skill_id)]
    )
    selected_memories = tuple(
        memory
        for memory in sorted(snapshot.memories, key=lambda item: item.memory_id)
        if role == "slow"
        or evaluate_applicability(memory.observable_applicability, public_features)[0]
    )
    if role == "fast":
        controls = {
            "role": role,
            "candidate_policy": _plain(snapshot.candidate_policy),
            "verification": _plain(snapshot.verification),
        }
    else:
        controls = {
            "role": role,
            "verification": _plain(snapshot.verification),
            "edit_policy": {
                "single_surface_only": True,
                "observable_applicability_only": True,
            },
        }
    payload = {
        "schema_version": "effective-harness-view/1",
        "instruction": snapshot.instruction,
        "skills": [_skill_payload(skill) for skill in selected_skills],
        "memories": [_memory_payload(memory) for memory in selected_memories],
        "controls": controls,
    }
    return EffectiveHarnessView(
        instruction=snapshot.instruction,
        skills=selected_skills,
        memories=selected_memories,
        controls=_freeze_json(controls),
        effective_harness_view_sha=canonical_sha256(payload),
    )


__all__ = [
    "EffectiveHarnessView",
    "evaluate_applicability",
    "resolve_harness_view",
]
