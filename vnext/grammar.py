"""CandidateGrammarV1: menu-v2 presets expressed as ProgramSpecV1 values."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..operators.registry import OPERATOR_METADATA, canonicalize
from ..policy.action_spec import ActionSpec, action_menu_v2
from ..policy.program_edit import ProgramSpecV1
from ..policy.task_spec import TaskSpec
from ._canonical import require_sha, sha256


@dataclass(frozen=True)
class ActionEligibilityManifestV1:
    states: tuple[tuple[str, str], ...]
    natural_harm_delta: float = 0.05
    source: str = "support_a_discovery_init"

    ALLOWED = frozenset({"active", "disabled", "experimental"})

    def __post_init__(self) -> None:
        menu_ids = set(action_menu_v2().actions)
        if {name for name, _ in self.states} != menu_ids:
            raise ValueError("eligibility manifest must cover action_menu_v2 exactly")
        if any(state not in self.ALLOWED for _, state in self.states):
            raise ValueError("eligibility manifest contains an unknown state")

    @classmethod
    def conservative(
        cls, *, ar_reverse_test_passed: bool = False,
        hampel_reverse_test_passed: bool = False,
        enable_levelshift: bool = False,
    ) -> "ActionEligibilityManifestV1":
        states = []
        for action_id in sorted(action_menu_v2().actions):
            if action_id == "v_ssm":
                state = "disabled"
            elif action_id == "v_ar":
                state = "active" if ar_reverse_test_passed else "experimental"
            elif action_id == "v_hampel":
                state = "active" if hampel_reverse_test_passed else "experimental"
            elif action_id == "v_levelshift":
                state = "active" if enable_levelshift else "experimental"
            else:
                state = "active"
            states.append((action_id, state))
        return cls(tuple(states))

    def state(self, action_id: str) -> str:
        return dict(self.states)[action_id]

    @property
    def sha256(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class ActionEligibilityEvidenceV1:
    action_id: str
    state: str
    evidence_sha: str
    operator_implementation_sha: str
    dependency_environment_sha: str

    def __post_init__(self) -> None:
        if self.state not in ActionEligibilityManifestV1.ALLOWED:
            raise ValueError("eligibility evidence state is invalid")
        for name in (
            "evidence_sha", "operator_implementation_sha", "dependency_environment_sha",
        ):
            require_sha(getattr(self, name), name)


@dataclass(frozen=True)
class ActionEligibilityManifestV2:
    entries: tuple[ActionEligibilityEvidenceV1, ...]
    action_menu_sha: str
    init_view_sha: str
    natural_harm_delta: float = 0.05
    source: str = "support_a_discovery_init_only"
    schema_version: str = "vnext-action-eligibility/2"

    def __post_init__(self) -> None:
        require_sha(self.action_menu_sha, "action_menu_sha")
        require_sha(self.init_view_sha, "init_view_sha")
        menu_ids = set(action_menu_v2().actions)
        if {entry.action_id for entry in self.entries} != menu_ids:
            raise ValueError("V2 eligibility evidence must cover action_menu_v2 exactly")
        if self.action_menu_sha != action_menu_v2().sha256:
            raise ValueError("V2 eligibility is bound to another action menu")
        if self.state("v_ssm") != "disabled":
            raise ValueError("v_ssm is disabled in the MVP grammar")

    @property
    def states(self) -> tuple[tuple[str, str], ...]:
        return tuple((entry.action_id, entry.state) for entry in self.entries)

    def state(self, action_id: str) -> str:
        try:
            return next(entry.state for entry in self.entries if entry.action_id == action_id)
        except StopIteration:
            raise KeyError(action_id) from None

    @property
    def sha256(self) -> str:
        return sha256(self)


class CandidateGrammarV1:
    max_steps = 3

    def __init__(
        self, eligibility: ActionEligibilityManifestV1 | ActionEligibilityManifestV2,
    ) -> None:
        self.eligibility = eligibility
        self.menu = action_menu_v2()

    @property
    def sha256(self) -> str:
        payload = {
            "version": "candidate-grammar-v1",
            "menu_sha": self.menu.sha256,
            "eligibility_sha": self.eligibility.sha256,
            "max_steps": self.max_steps,
            "parameter_domain": "action_menu_v2_frozen_presets_only",
            "allows_noop": True,
        }
        return sha256(payload)

    def from_action(self, action_id: str, task_spec: TaskSpec) -> ProgramSpecV1:
        if action_id not in self.menu.actions:
            raise ValueError(f"unknown menu-v2 action {action_id!r}")
        if self.eligibility.state(action_id) != "active":
            raise ValueError(f"action {action_id!r} is not active")
        action = self.menu.actions[action_id]
        spec = ProgramSpecV1(
            steps=tuple((step.op, tuple(sorted(dict(step.params).items()))) for step in action.steps),
            scope=("global",), task_type=task_spec.task_type,
            risk_budget_beta=1.0, max_modified_fraction=1.0,
            provenance={"source": "candidate_grammar_v1", "menu_action_id": action_id},
        )
        self.validate(spec, task_spec)
        return spec

    def noop(self, task_spec: TaskSpec) -> ProgramSpecV1:
        return ProgramSpecV1(
            steps=(), scope=("global",), task_type=task_spec.task_type,
            risk_budget_beta=0.0, max_modified_fraction=0.0,
            provenance={"source": "candidate_grammar_v1", "menu_action_id": "__noop__"},
        )

    def validate(self, spec: ProgramSpecV1, task_spec: TaskSpec) -> None:
        if spec.task_type != task_spec.task_type:
            raise ValueError("candidate task_type disagrees with TaskSpec")
        if len(spec.steps) > self.max_steps:
            raise ValueError("candidate exceeds the three-step grammar budget")
        if not spec.steps:
            return
        matching: list[ActionSpec] = []
        requested = tuple((op, dict(params)) for op, params in spec.steps)
        for action_id, action in self.menu.actions.items():
            expected = tuple((step.op, dict(step.params)) for step in action.steps)
            if requested == expected and self.eligibility.state(action_id) == "active":
                matching.append(action)
        if not matching:
            raise ValueError("candidate is not an active action_menu_v2 frozen preset")
        for op, _ in spec.steps:
            canonical = canonicalize(op)
            metadata = OPERATOR_METADATA.get(canonical)
            if metadata is None or task_spec.task_type not in metadata.get("allowed_tasks", ()):
                raise ValueError(f"operator {op!r} is unavailable for this task")
            if metadata.get("changes_target_space"):
                raise ValueError(f"operator {op!r} changes target space")
            if task_spec.is_op_forbidden(canonical):
                raise ValueError(f"operator {op!r} is forbidden by TaskSpec")

    def canonical_program_sha(self, spec: ProgramSpecV1) -> str:
        return sha256({"steps": [(op, dict(params)) for op, params in spec.steps]})

    def effect_signature(self, spec: ProgramSpecV1) -> tuple[tuple[str, tuple[tuple[str, object], ...]], ...]:
        return tuple((canonicalize(op), tuple(sorted(params))) for op, params in spec.steps)
