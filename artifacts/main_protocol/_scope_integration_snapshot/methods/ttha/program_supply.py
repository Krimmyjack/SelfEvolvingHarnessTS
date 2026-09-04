"""P1/P2 minimum Program Supply adapter (plan rev3, 2026-08-16).

This module is the online counterpart of the shared Program Supply route
in ``contracts.program_supply``.

Hard rules:

- The online adapter never constructs a partially-filled ``CaseFacts`` and
  never calls ``assess_case``.  The controlled ten-stage fold and the online
  Program Supply route are different instruments with different evidence.
- Every field read by ``route_program_supply_fault`` is assigned explicitly
  in ``ProgramSupplyFacts`` (the dataclass has no defaults).
- Online has no constrained-proposal experiment in P1, so the tri-state field
  is ``None`` (Unknown) and the router ABSTAINs.  It is never silently fixed
  to ``False``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.harness import SkillKind
from SelfEvolvingHarnessTS.contracts.program_supply import (
    PROGRAM_SUPPLY_ROUTE_FIELDS,
    route_program_supply_fault,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import MaterializedSnapshot
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

_ONLINE_EXPRESSIBILITY_STATUS = "EXPRESSIBILITY_UNKNOWN"


@dataclass(frozen=True)
class ProgramSupplyFacts:
    """Only the fields the Program Supply route is allowed to read, plus case_id.

    No field has a default: a partially-filled facts object cannot exist.
    """

    case_id: str
    expressibility_status: str
    expressibility_cause: str | None
    capability_skill_exists: bool
    skill_retrieved: bool
    constrained_proposal_succeeds: bool | None

    def route_kwargs(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in PROGRAM_SUPPLY_ROUTE_FIELDS}


@dataclass(frozen=True)
class ProgramSupplyDecision:
    case_id: str
    cause_code: str
    actionability: str
    surface_templates: tuple[str, ...]


def _capability_skills(view: object) -> tuple[Any, ...]:
    skills = getattr(view, "skills", ()) or ()
    out: list[Any] = []
    for skill in skills:
        kind = getattr(skill, "skill_kind", None)
        kind_value = getattr(kind, "value", kind)
        if kind is SkillKind.CAPABILITY or kind_value == SkillKind.CAPABILITY.value:
            out.append(skill)
    return tuple(out)


def _capability_skill_ids(view: object) -> tuple[str, ...]:
    return tuple(str(getattr(skill, "skill_id", "")) for skill in _capability_skills(view))


def build_program_supply_facts(
    trace: DecisionTrace,
    episode: object,
    view: object,
) -> ProgramSupplyFacts:
    """Pure ``(trace, episode, view) -> ProgramSupplyFacts`` adapter.

    P1 deliberately has no online expressibility oracle and no constrained
    proposal experiment yet.  The route inputs are therefore assigned
    explicitly from the only online evidence available:

    - ``expressibility_status``: ``EXPRESSIBILITY_UNKNOWN`` (never a fixture
      default such as ``PROVEN_EXPRESSIBLE``);
    - ``expressibility_cause``: ``None``;
    - ``capability_skill_exists`` / ``skill_retrieved``: current view and
      ``trace.retrieved_skill_ids``;
    - ``constrained_proposal_succeeds``: ``None`` (Unknown) -> ABSTAIN.

    ``episode`` is accepted for the documented P1 signature and future
    episode-level evidence; the current version does not mine private or
    oracle fields out of it.
    """
    if not isinstance(trace, DecisionTrace):
        raise TypeError("trace must be a DecisionTrace")
    if not trace.case_id:
        raise ValueError("trace.case_id must be non-empty")
    capability_ids = set(_capability_skill_ids(view))
    retrieved_ids = {str(value) for value in (trace.retrieved_skill_ids or ())}
    return ProgramSupplyFacts(
        case_id=trace.case_id,
        expressibility_status=_ONLINE_EXPRESSIBILITY_STATUS,
        expressibility_cause=None,
        capability_skill_exists=bool(capability_ids),
        skill_retrieved=bool(capability_ids & retrieved_ids),
        constrained_proposal_succeeds=None,
    )


def route_online_program_supply_fault(
    trace: DecisionTrace,
    episode: object,
    view: object,
) -> ProgramSupplyDecision:
    """Build online facts and route them without touching ``CaseFacts``."""
    facts = build_program_supply_facts(trace, episode, view)
    cause, actionability, templates = route_program_supply_fault(**facts.route_kwargs())
    return ProgramSupplyDecision(
        case_id=facts.case_id,
        cause_code=cause,
        actionability=actionability,
        surface_templates=templates,
    )


def build_single_surface_catalog(
    *,
    decision: ProgramSupplyDecision,
    parent: MaterializedSnapshot,
    controller: Any,
    retrieved_capability_skill_ids: Sequence[str] = (),
) -> tuple[dict[str, object], ...]:
    """P2: turn an attribution into exactly one authorized surface, or ().

    ABSTAIN is mechanical: non-EDITABLE routes and routes whose only surface
    does not exist in the current snapshot return the empty catalog.  Callers
    must record ``abstained_by_route`` and must not invoke Slow.
    """
    if not isinstance(parent, MaterializedSnapshot):
        raise TypeError("parent must be a MaterializedSnapshot")
    if decision.actionability != "EDITABLE_M0" or not decision.surface_templates:
        return ()
    router = getattr(controller, "router", None)
    if router is None:
        return ()
    try:
        authorization = router.allowed_targets(decision.cause_code)
    except KeyError:
        return ()
    if not authorization.allowed_operations:
        return ()

    for template in decision.surface_templates:
        definition = next(
            (
                item
                for item in controller.surfaces.definitions
                if item.surface_template_id == template
            ),
            None,
        )
        if definition is None:
            continue
        operation = next(
            (
                op
                for op in authorization.allowed_operations
                if op in definition.allowed_operations
            ),
            None,
        )
        if operation is None:
            continue
        if "{skill_id}" not in template:
            surface_id = template
        else:
            skills = parent.snapshot.skills
            if definition.target_class == "capability":
                candidates = tuple(
                    skill
                    for skill in skills
                    if skill.skill_kind is SkillKind.CAPABILITY
                )
            elif definition.target_class == "bootstrap_procedure":
                candidates = tuple(
                    skill
                    for skill in skills
                    if skill.skill_kind is SkillKind.BOOTSTRAP_PROCEDURE
                )
            else:
                candidates = ()
            if definition.precondition == "ABSENT":
                surface_id = template
            else:
                if retrieved_capability_skill_ids:
                    wanted = {str(value) for value in retrieved_capability_skill_ids}
                    candidates = tuple(
                        skill for skill in candidates if skill.skill_id in wanted
                    )
                candidates = tuple(
                    sorted(candidates, key=lambda skill: skill.skill_id)
                )
                if not candidates:
                    return ()
                surface_id = template.format(skill_id=candidates[0].skill_id)
        precondition: dict[str, object] = {"kind": definition.precondition}
        if definition.precondition == "SHA":
            precondition["sha"] = controller.surface_precondition_sha(
                parent, surface_id
            )
        dependency_preconditions = {
            key: parent.snapshot.dependency_shas[key]
            for key in definition.required_dependency_keys
            if key in parent.snapshot.dependency_shas
        }
        return (
            {
                "surface_id": surface_id,
                "surface_template_id": definition.surface_template_id,
                "target_class": definition.target_class,
                "surface_type": definition.surface_type,
                "operation": operation,
                "allowed_operations": [operation],
                "surface_precondition": precondition,
                "required_dependency_keys": list(definition.required_dependency_keys),
                "dependency_precondition_shas": dependency_preconditions,
            },
        )
    return ()


@dataclass(frozen=True)
class VerifiedProgramAlternative:
    patch_id: str
    steps: tuple[tuple[str, dict[str, object]], ...]
    verification: Any


@dataclass(frozen=True)
class ProgramSupplyVerification:
    alternatives: tuple[VerifiedProgramAlternative, ...] = ()
    choice_offered: bool = False
    behavior_distinct_pairs: tuple[tuple[str, str], ...] = ()
    invalid_option_count: int = 0
    relevant_capability_skill_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedProgramSupplyAssessment:
    facts: ProgramSupplyFacts
    verification: ProgramSupplyVerification
    decision: ProgramSupplyDecision
    relevant_capability_skill_ids: tuple[str, ...] = ()


def _ordered_option_steps(
    option: Mapping[str, object],
) -> tuple[tuple[str, dict[str, object]], ...] | None:
    raw_steps = option.get("program_steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        return None
    steps: list[tuple[str, dict[str, object]]] = []
    for item in raw_steps:
        if not isinstance(item, Mapping) or not isinstance(item.get("op"), str):
            return None
        params = item.get("params")
        if not isinstance(params, Mapping):
            return None
        steps.append((str(item["op"]), dict(params)))
    return tuple(steps)


def _canonical_steps(
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    return [{"op": op, "params": dict(params)} for op, params in steps]


def bind_verified_program_options(
    card: Mapping[str, object],
    assessment: VerifiedProgramSupplyAssessment,
) -> tuple[dict[str, object] | None, tuple[str, ...], dict[str, Any] | None]:
    """Bind one Card to verifier-earned ``(patch_id, ordered steps)`` pairs."""
    alternatives = assessment.verification.alternatives
    if not alternatives:
        return None, (), {
            "stage": "no_verified_alternatives",
            "case_id": assessment.facts.case_id,
        }

    verified_by_id: dict[str, tuple[tuple[str, dict[str, object]], ...]] = {}
    for alternative in alternatives:
        if alternative.patch_id in verified_by_id:
            return None, (), {
                "stage": "duplicate_verified_patch_id",
                "patch_id": alternative.patch_id,
            }
        verified_by_id[alternative.patch_id] = alternative.steps

    raw_options = [
        option for option in (card.get("typed_patch_options") or [])
        if isinstance(option, Mapping)
    ]
    seen_ids: set[str] = set()
    filtered_options: list[dict[str, object]] = []
    for option in raw_options:
        patch_id = str(option.get("patch_id") or "")
        if patch_id in seen_ids:
            return None, (), {
                "stage": "duplicate_card_patch_id",
                "patch_id": patch_id,
            }
        seen_ids.add(patch_id)
        if patch_id not in verified_by_id:
            continue
        card_steps = _ordered_option_steps(option)
        verified_steps = verified_by_id[patch_id]
        if card_steps is None or card_steps != verified_steps:
            return None, (), {
                "stage": "program_binding_mismatch",
                "patch_id": patch_id,
                "verified_program_steps": _canonical_steps(verified_steps),
                "card_program_steps": (
                    _canonical_steps(card_steps)
                    if card_steps is not None else None
                ),
            }
        canonical = dict(option)
        canonical["program_steps"] = _canonical_steps(verified_steps)
        filtered_options.append(canonical)

    verified_ids = tuple(verified_by_id)
    if not filtered_options:
        return None, verified_ids, {
            "stage": "no_verified_options",
            "case_id": assessment.facts.case_id,
            "verified_patch_ids": list(verified_ids),
        }
    filtered_card = dict(card)
    filtered_card["typed_patch_options"] = filtered_options
    return filtered_card, verified_ids, None


def retrieved_relevant_capability_skill_ids(
    assessment: VerifiedProgramSupplyAssessment,
    trace: DecisionTrace,
) -> tuple[str, ...]:
    """Exact relevant/retrieved intersection used for Skill-body PATCH."""
    retrieved = {str(value) for value in tuple(trace.retrieved_skill_ids or ())}
    return tuple(
        skill_id
        for skill_id in assessment.relevant_capability_skill_ids
        if skill_id in retrieved
    )


def _typed_option_steps(option: Mapping[str, object]) -> tuple[tuple[str, dict[str, object]], ...]:
    patch_id = option.get("patch_id")
    if not isinstance(patch_id, str) or not patch_id:
        raise ValueError("typed_patch_options entry requires patch_id")
    raw_steps = option.get("program_steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise TypeError("typed_patch_options entry requires program_steps")
    steps: list[tuple[str, dict[str, object]]] = []
    for item in raw_steps:
        if not isinstance(item, Mapping) or not isinstance(item.get("op"), str):
            raise TypeError("typed program step requires string op")
        params = item.get("params")
        if not isinstance(params, Mapping):
            raise TypeError("typed program step requires params object")
        steps.append((item["op"], dict(params)))
    if not steps:
        raise ValueError("typed_patch_options entry requires non-empty program_steps")
    return tuple(steps)


def _verification_proves_effect(verification: Any) -> bool:
    """E-1 mechanical gate, reading only ``ScopeExecutor.verify()`` products.

    - checked_windows > 0（零训练窗通过不算证据）；
    - 至少一个窗口实际修改已有观测；
    - 程序整体不得与 identity 等效（至少一个窗口非 identity-equivalent）。
    """
    checked = int(getattr(verification, "checked_windows", 0))
    modified = int(getattr(verification, "modified_windows", 0))
    identity_equivalent = int(
        getattr(verification, "identity_equivalent_windows", checked)
    )
    return (
        bool(getattr(verification, "passed", False))
        and checked > 0
        and modified > 0
        and identity_equivalent < checked
    )


def _prepared_values_for(verification: Any) -> tuple[Any, ...]:
    return tuple(
        getattr(verification, "_program_supply_prepared_values", ())
    )


def _prepared_value_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return bool(
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _same_window_behavior(left: Any, right: Any) -> bool:
    left_values = _prepared_values_for(left)
    right_values = _prepared_values_for(right)
    return bool(
        left_values
        and len(left_values) == len(right_values)
        and all(
            _prepared_value_equal(a, b)
            for a, b in zip(left_values, right_values)
        )
    )
def verify_program_supply_alternatives(
    *,
    executor: Any,
    typed_patch_options: Sequence[Mapping[str, object]],
    origin: int,
) -> ProgramSupplyVerification:
    """E-1: earn PROVEN_EXPRESSIBLE from ``executor.verify()`` only.

    Never calls ``executor.evaluate()``; therefore this consumes zero Support
    budget and leaks zero downstream Outcome into routing.
    """
    alternatives: list[VerifiedProgramAlternative] = []
    invalid_option_count = 0
    for option in typed_patch_options:
        if not isinstance(option, Mapping):
            invalid_option_count += 1
            continue
        try:
            steps = _typed_option_steps(option)
        except (TypeError, ValueError):
            invalid_option_count += 1
            continue
        verifier = getattr(executor, "verify_without_behavior_hashes", None)
        if not callable(verifier):
            verifier = executor.verify
        verification = verifier(steps, origin)
        if not _verification_proves_effect(verification):
            continue
        alternatives.append(
            VerifiedProgramAlternative(
                patch_id=str(option["patch_id"]),
                steps=steps,
                verification=verification,
            )
        )
    pairs: list[tuple[str, str]] = []
    for i, left in enumerate(alternatives):
        for right in alternatives[i + 1:]:
            if not _same_window_behavior(
                left.verification, right.verification
            ):
                pairs.append((left.patch_id, right.patch_id))
    return ProgramSupplyVerification(
        alternatives=tuple(alternatives),
        choice_offered=len(alternatives) >= 2 and bool(pairs),
        behavior_distinct_pairs=tuple(pairs),
        invalid_option_count=invalid_option_count,
    )


def _steps_family_signature(steps: Sequence[tuple[str, Mapping[str, object]]]) -> str:
    from .experience_memory import workflow_signature_of

    return workflow_signature_of([
        {"op": op, "params": dict(params)} for op, params in steps
    ])


def _relevant_capability_skill_ids(
    *,
    view: object,
    executor: Any,
    origin: int,
    verification: ProgramSupplyVerification,
) -> tuple[str, ...]:
    """Program-aware ``capability_skill_exists``.

    A capability Skill only counts when it is usable in the current view AND
    supplies the same program family (operator-sequence signature) as a
    verified alternative, or has exactly the same ordered prepared values on
    every aligned window. Program-supply routing deliberately creates no
    per-candidate hashes and persists none of these transient values.
    """
    if not verification.alternatives:
        return ()
    from .fast_agent import _parse_frozen_steps

    family_signatures = {
        _steps_family_signature(alternative.steps)
        for alternative in verification.alternatives
    }
    alternative_verifications = tuple(
        alternative.verification
        for alternative in verification.alternatives
    )
    relevant: list[str] = []
    for skill in getattr(view, "skills", ()) or ():
        kind = getattr(skill, "skill_kind", None)
        kind_value = getattr(kind, "value", kind)
        if not (kind is SkillKind.CAPABILITY
                or kind_value == SkillKind.CAPABILITY.value):
            continue
        steps = _parse_frozen_steps(getattr(skill, "body", ""))
        if steps is None:
            continue
        if _steps_family_signature(steps) in family_signatures:
            relevant.append(str(skill.skill_id))
            continue
        verifier = getattr(executor, "verify_without_behavior_hashes", None)
        if not callable(verifier):
            verifier = executor.verify
        try:
            skill_verification = verifier(steps, origin)
        except Exception as exc:  # noqa: BLE001 -- verifier failure = unrelated
            if getattr(exc, "program_supply_budget_exhausted", False):
                raise
            continue
        if any(
            _same_window_behavior(skill_verification, alternative)
            for alternative in alternative_verifications
        ):
            relevant.append(str(skill.skill_id))
    return tuple(dict.fromkeys(relevant))


def controlled_add_only_group_decision(*, case_id: str) -> ProgramSupplyDecision:
    """Controlled positive-control route for legacy/dev ADD-only group callers.

    The facts are explicit (PROVEN_EXPRESSIBLE + no capability skill); the
    cause string is produced by ``route_program_supply_fault``, not hardcoded
    at the call site.  P4 natural flow must use the verifier-earned router.
    """
    cause, actionability, templates = route_program_supply_fault(
        expressibility_status="PROVEN_EXPRESSIBLE",
        expressibility_cause=None,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    )
    return ProgramSupplyDecision(case_id, cause, actionability, templates)


def controlled_add_only_group_catalog() -> tuple[dict[str, object], ...]:
    """Static catalog for the controlled ADD-only group helper above."""
    return (
        {
            "surface_id": "skill_library.entries/{skill_id}",
            "surface_template_id": "skill_library.entries/{skill_id}",
            "target_class": "capability",
            "surface_type": "structured_entry",
            "operation": "ADD",
            "allowed_operations": ["ADD"],
            "surface_precondition": {"kind": "ABSENT"},
            "required_dependency_keys": [],
            "dependency_precondition_shas": {},
        },
    )


def build_verified_program_supply_facts(
    *,
    trace: DecisionTrace,
    episode: object,
    view: object,
    executor: Any,
    typed_patch_options: Sequence[Mapping[str, object]],
    origin: int,
    constrained_proposal_succeeds: bool | None = None,
) -> tuple[ProgramSupplyFacts, ProgramSupplyVerification]:
    """E-1 online facts: PROVEN_EXPRESSIBLE only when earned by verify()."""
    verification = verify_program_supply_alternatives(
        executor=executor,
        typed_patch_options=typed_patch_options,
        origin=origin,
    )
    relevant_ids: tuple[str, ...] = ()
    if verification.alternatives:
        expressibility_status = "PROVEN_EXPRESSIBLE"
        expressibility_cause = None
        relevant_ids = _relevant_capability_skill_ids(
            view=view,
            executor=executor,
            origin=origin,
            verification=verification,
        )
        verification = ProgramSupplyVerification(
            alternatives=verification.alternatives,
            choice_offered=verification.choice_offered,
            behavior_distinct_pairs=verification.behavior_distinct_pairs,
            invalid_option_count=verification.invalid_option_count,
            relevant_capability_skill_ids=relevant_ids,
        )
        capability_skill_exists = bool(relevant_ids)
    else:
        expressibility_status = "EXPRESSIBILITY_UNKNOWN"
        expressibility_cause = None
        capability_skill_exists = False
    retrieved_ids = {str(value) for value in (trace.retrieved_skill_ids or ())}
    facts = ProgramSupplyFacts(
        case_id=trace.case_id,
        expressibility_status=expressibility_status,
        expressibility_cause=expressibility_cause,
        capability_skill_exists=capability_skill_exists,
        skill_retrieved=bool(set(relevant_ids) & retrieved_ids),
        constrained_proposal_succeeds=constrained_proposal_succeeds,
    )
    return facts, verification


def route_verified_program_supply_fault(
    *,
    trace: DecisionTrace,
    episode: object,
    view: object,
    executor: Any,
    typed_patch_options: Sequence[Mapping[str, object]],
    origin: int,
    constrained_proposal_succeeds: bool | None = None,
) -> VerifiedProgramSupplyAssessment:
    """E-1 route: earned PROVEN_EXPRESSIBLE + program-aware skill check."""
    facts, verification = build_verified_program_supply_facts(
        trace=trace,
        episode=episode,
        view=view,
        executor=executor,
        typed_patch_options=typed_patch_options,
        origin=origin,
        constrained_proposal_succeeds=constrained_proposal_succeeds,
    )
    cause, actionability, templates = route_program_supply_fault(
        **facts.route_kwargs()
    )
    return VerifiedProgramSupplyAssessment(
        facts=facts,
        verification=verification,
        decision=ProgramSupplyDecision(
            case_id=facts.case_id,
            cause_code=cause,
            actionability=actionability,
            surface_templates=templates,
        ),
        relevant_capability_skill_ids=(
            verification.relevant_capability_skill_ids
        ),
    )


__all__ = [
    "ProgramSupplyDecision",
    "ProgramSupplyFacts",
    "ProgramSupplyVerification",
    "VerifiedProgramAlternative",
    "VerifiedProgramSupplyAssessment",
    "bind_verified_program_options",
    "build_program_supply_facts",
    "build_single_surface_catalog",
    "build_verified_program_supply_facts",
    "controlled_add_only_group_catalog",
    "controlled_add_only_group_decision",
    "route_online_program_supply_fault",
    "route_verified_program_supply_fault",
    "retrieved_relevant_capability_skill_ids",
    "verify_program_supply_alternatives",
]
