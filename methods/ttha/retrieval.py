from __future__ import annotations

import operator
import re
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


# Set on ``risk_guards`` when Target feedback disconfirms a Skill.  Kept here
# rather than imported so the retrieval layer has no dependency on the
# evaluation package that writes it.
_RESTRICTED_GUARD = "restricted_by_target_feedback"

# The literal a Slow consolidation must write into TRY when the deterministic
# authorization audit handed it no operator to name.  Kept local for the same
# reason as the guard above.
_TRY_ABSTAIN = "NO_AUTHORIZED_ACTIVE_RECOMMENDATION"
_TRY_SENTINEL = re.compile(r"NO_[A-Z0-9_]+")

# What a deprioritizing carrier writes to state how many distinct Tasks the
# harm repeated across, and the smallest count that is about a Program family
# rather than about one instance.
_RISK_EVIDENCE_COUNT = "evidence_distinct_task_count"
_RISK_MIN_DISTINCT_TASKS = 2


def _experience_card_sections(skill: SkillEntry) -> Mapping[str, object] | None:
    """The six named sections of a consolidated experience card, if any.

    Only a card a Slow stage compiled out of Source evidence carries
    ``risk_guards.sections`` with a TRY clause in it.  A bootstrap procedure
    and a Target-local Skill carrying a frozen Workflow have no such field, so
    neither is an experience card here.
    """
    sections = (skill.risk_guards or {}).get("sections")
    if isinstance(sections, Mapping) and "TRY" in sections:
        return sections
    return None


def _scopes_beyond_task_kind(ast: Mapping[str, object]) -> bool:
    """Whether the applicability AST constrains anything but ``task_kind``.

    ``task_kind == classification`` selects every Task in the exam, so it is
    an eligibility gate rather than a Context Scope a deprioritization could
    be executed against.
    """
    if "feature" in ast:
        return str(ast["feature"]) != "task_kind"
    for key in ("all", "any"):
        children = ast.get(key)
        if isinstance(children, Sequence):
            return any(
                _scopes_beyond_task_kind(child)
                for child in children
                if isinstance(child, Mapping)
            )
    negated = ast.get("not")
    if isinstance(negated, Mapping):
        return _scopes_beyond_task_kind(negated)
    return False


def _is_inert_experience_card(skill: SkillEntry) -> bool:
    """An experience card that authorizes no action on any surface.

    C40 established the cost of serving one anyway.  A5 received
    ``source_investigation_cls_v1``: TRY abstained, so the execution layer
    correctly withheld every candidate, but its RISK prose still named
    ``repair_level_shift``, the proposal stage read it, and both rounds went
    to the level-shift family and were rejected while A3 reached
    ``hampel_filter`` (A5 - A3 = -0.269).  Knowledge that qualifies for no
    action has to be inert on every surface that shapes behaviour, and the
    proposal context is one of those surfaces.

    A card is inert when both clause kinds come up empty:

    * no authorized TRY -- the clause is blank, or the abstention sentinel the
      Slow stage writes when the audit authorized no operator;
    * no repeated scoped RISK -- a deprioritization is executable only with a
      Context Scope finer than the eligibility gate *and* an explicit count of
      the distinct Tasks the harm repeated across.  Free prose satisfies
      neither, so it authorizes nothing.

    Everything else stays: a card naming an authorized operator in TRY, a card
    whose RISK is scoped and counted, and every entry that is not an
    experience card at all.
    """
    sections = _experience_card_sections(skill)
    if sections is None:
        return False
    try_text = str(sections.get("TRY") or "").strip()
    authorized_try = bool(
        try_text
        and try_text != _TRY_ABSTAIN
        and not _TRY_SENTINEL.fullmatch(try_text)
    )
    count = (skill.risk_guards or {}).get(_RISK_EVIDENCE_COUNT)
    repeated = (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= _RISK_MIN_DISTINCT_TASKS
    )
    scoped_risk = repeated and _scopes_beyond_task_kind(
        skill.observable_applicability
    )
    return not authorized_try and not scoped_risk


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
        # A Skill its own Target later disconfirmed stops being retrieved, in
        # both roles.  Leaving it in the active snapshot -- which is what used
        # to happen once delayed re-validation failed -- let a claim the
        # Domain had already refuted keep arriving alongside fresh evidence.
        # Restriction is written by a PATCH to this Skill's own risk_guards,
        # so the snapshot still records that it existed and why it stopped.
        if bool((skill.risk_guards or {}).get(_RESTRICTED_GUARD)):
            continue
        # An experience card with nothing to authorize is withheld from the
        # proposal view only.  Slow still resolves it, and the store still
        # holds it unchanged, so what it said stays auditable.
        if role == "fast" and _is_inert_experience_card(skill):
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
