"""Equal-budget B3 supplier arms with ITT no-op handling and effect deduplication."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..policy.program_edit import ProgramSpecV1, spec_v1_from_dict
from ..policy.task_spec import TaskSpec
from .grammar import CandidateGrammarV1
from .pattern import PatternCard
from ._canonical import sha256


B3 = 3


def semantic_supplier_seed(
    *, global_seed: int, fold_id: str, slot_namespace: str, pattern: PatternCard,
) -> int:
    """UID-free seed bound only to prereg context and visible numeric semantics."""
    digest = sha256({
        "global_seed": int(global_seed),
        "fold_id": str(fold_id),
        "slot_namespace": str(slot_namespace),
        "pattern_semantic_sha": pattern.sha256,
    })
    return int(digest[:16], 16)


@dataclass(frozen=True)
class CandidateBatch:
    arm_id: str
    programs: tuple[ProgramSpecV1, ...]
    statuses: tuple[str, ...]
    request_count: int

    def __post_init__(self) -> None:
        if len(self.programs) != B3 or len(self.statuses) != B3:
            raise ValueError("all identity-gate arms must spend exactly three candidate slots")
        if self.request_count < 0:
            raise ValueError("request_count cannot be negative")


def _active_ids(grammar: CandidateGrammarV1) -> list[str]:
    return [
        action_id for action_id in sorted(grammar.menu.actions)
        if grammar.eligibility.state(action_id) == "active"
    ]


def _dedupe_itt(
    arm_id: str,
    programs: Sequence[ProgramSpecV1],
    statuses: Sequence[str],
    grammar: CandidateGrammarV1,
    task_spec: TaskSpec,
    *,
    request_count: int = 0,
) -> CandidateBatch:
    output: list[ProgramSpecV1] = []
    final_status: list[str] = []
    effects = set()
    for index in range(B3):
        program = programs[index] if index < len(programs) else grammar.noop(task_spec)
        status = statuses[index] if index < len(statuses) else "missing_itt_noop"
        try:
            grammar.validate(program, task_spec)
            effect = grammar.effect_signature(program)
        except Exception:
            program, effect, status = grammar.noop(task_spec), (), "invalid_itt_noop"
        if effect and effect in effects:
            program, effect, status = grammar.noop(task_spec), (), "duplicate_effect_itt_noop"
        if effect:
            effects.add(effect)
        output.append(program)
        final_status.append(status)
    return CandidateBatch(arm_id, tuple(output), tuple(final_status), request_count)


def deterministic_b3(
    pattern: PatternCard, task_spec: TaskSpec, grammar: CandidateGrammarV1,
) -> CandidateBatch:
    """Deterministic preset enumeration; ranking uses no UID or private metadata."""
    values = pattern.values
    preferred = []
    if values.get("missing_rate", 0.0) > 0:
        preferred.extend(("v_ar", "v_none"))
    if values.get("outlier_density", 0.0) > 0.01:
        preferred.extend(("v_hampel", "v_winsor"))
    preferred.extend(_active_ids(grammar))
    ids = list(dict.fromkeys(action_id for action_id in preferred if action_id in _active_ids(grammar)))
    programs = [grammar.from_action(action_id, task_spec) for action_id in ids[:B3]]
    return _dedupe_itt(
        "deterministic_b3", programs, ["valid"] * len(programs), grammar, task_spec,
    )


def random_valid_b3(
    pattern: PatternCard, task_spec: TaskSpec, grammar: CandidateGrammarV1, *, seed: int,
) -> CandidateBatch:
    del pattern
    ids = _active_ids(grammar)
    rng = np.random.default_rng(int(seed))
    selected = [ids[index] for index in rng.permutation(len(ids))[:B3]] if ids else []
    programs = [grammar.from_action(action_id, task_spec) for action_id in selected]
    return _dedupe_itt("random_valid_b3", programs, ["valid"] * len(programs), grammar, task_spec)


def llm_direct_b3(
    pattern: PatternCard,
    task_spec: TaskSpec,
    grammar: CandidateGrammarV1,
    *,
    propose: Callable[[str, int], Sequence[Mapping[str, Any]]],
) -> CandidateBatch:
    """One API request; malformed/error slots become ITT no-op and are never replaced."""
    programs: list[ProgramSpecV1] = []
    statuses: list[str] = []
    try:
        payloads = list(propose(pattern.summary, B3))
    except Exception:
        payloads = []
        statuses = ["api_error_itt_noop"] * B3
    for payload in payloads[:B3]:
        try:
            programs.append(spec_v1_from_dict(payload))
            statuses.append("valid")
        except Exception:
            programs.append(grammar.noop(task_spec))
            statuses.append("malformed_itt_noop")
    return _dedupe_itt(
        "llm_direct_b3", programs, statuses, grammar, task_spec, request_count=1,
    )


def llm_plan_compiler_b3(
    pattern: PatternCard,
    task_spec: TaskSpec,
    grammar: CandidateGrammarV1,
    *,
    plan: Callable[[str, int], Sequence[str]],
) -> CandidateBatch:
    """Readiness plan -> deterministic menu compiler, the preregistered main LLM arm."""
    programs: list[ProgramSpecV1] = []
    statuses: list[str] = []
    try:
        action_ids = list(plan(pattern.summary, B3))
    except Exception:
        action_ids = []
        statuses = ["api_error_itt_noop"] * B3
    for action_id in action_ids[:B3]:
        try:
            programs.append(grammar.from_action(str(action_id), task_spec))
            statuses.append("valid")
        except Exception:
            programs.append(grammar.noop(task_spec))
            statuses.append("malformed_itt_noop")
    return _dedupe_itt(
        "llm_plan_compiler_b3", programs, statuses, grammar, task_spec, request_count=1,
    )


def hybrid_escalation_b3(
    pattern: PatternCard,
    task_spec: TaskSpec,
    grammar: CandidateGrammarV1,
    *,
    low_support_or_same_effect: bool,
    plan: Callable[[str, int], Sequence[str]],
) -> CandidateBatch:
    deterministic = deterministic_b3(pattern, task_spec, grammar)
    if not low_support_or_same_effect:
        return CandidateBatch(
            "hybrid_escalation_b3", deterministic.programs,
            deterministic.statuses, request_count=0,
        )
    llm = llm_plan_compiler_b3(pattern, task_spec, grammar, plan=plan)
    combined = [deterministic.programs[0], *llm.programs[:2]]
    statuses = [deterministic.statuses[0], *llm.statuses[:2]]
    return _dedupe_itt(
        "hybrid_escalation_b3", combined, statuses, grammar, task_spec, request_count=1,
    )
