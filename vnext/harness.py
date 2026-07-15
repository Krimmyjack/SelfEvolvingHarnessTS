"""Minimal immutable Harness evolution surface, separate from legacy HarnessState/P6."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from ._canonical import require_sha, sha256


EDITABLE_SURFACES = frozenset({
    "action_risk_rule", "selector_threshold", "supplier_mix",
    "retrieval_support_threshold", "llm_instruction", "llm_skill_applicability",
    "llm_retrieval_config",
})
LLM_SURFACES = frozenset({
    "llm_instruction", "llm_skill_applicability", "llm_retrieval_config",
})


@dataclass(frozen=True)
class AtomicHarnessEdit:
    surface: str
    value: Any
    rationale: str

    def __post_init__(self) -> None:
        if self.surface not in EDITABLE_SURFACES:
            raise ValueError(f"protected or unknown harness surface {self.surface!r}")
        if not self.rationale:
            raise ValueError("atomic edit requires an audit rationale")

    @property
    def semantic_sha(self) -> str:
        return sha256({"surface": self.surface, "value": self.value})


@dataclass(frozen=True)
class HarnessArtifactV1:
    action_risk_rule: Mapping[str, Any]
    selector_threshold: float
    supplier_mix: Mapping[str, int]
    retrieval_support_threshold: int
    llm_instruction: str | None = None
    llm_skill_applicability: Mapping[str, Any] | None = None
    llm_retrieval_config: Mapping[str, Any] | None = None
    llm_runtime_qualified: bool = False
    parent_sha: str | None = None
    atomic_edit: AtomicHarnessEdit | None = None

    def __post_init__(self) -> None:
        if not 0 <= float(self.selector_threshold) <= 1:
            raise ValueError("selector_threshold must be in [0,1]")
        if self.retrieval_support_threshold < 1:
            raise ValueError("retrieval support threshold must be positive")
        if any(value < 0 for value in self.supplier_mix.values()):
            raise ValueError("supplier allocation cannot be negative")
        if not self.llm_runtime_qualified and any(
            value is not None for value in (
                self.llm_instruction, self.llm_skill_applicability, self.llm_retrieval_config,
            )
        ):
            raise ValueError("LLM surfaces remain locked until the M3 qualification gate")

    @classmethod
    def h0(cls) -> "HarnessArtifactV1":
        return cls(
            action_risk_rule={}, selector_threshold=0.5,
            supplier_mix={"deterministic": 1, "random": 0, "llm": 0},
            retrieval_support_threshold=8,
        )

    @property
    def semantic_payload(self) -> dict[str, Any]:
        return {
            "action_risk_rule": dict(self.action_risk_rule),
            "selector_threshold": self.selector_threshold,
            "supplier_mix": dict(self.supplier_mix),
            "retrieval_support_threshold": self.retrieval_support_threshold,
            "llm_instruction": self.llm_instruction,
            "llm_skill_applicability": self.llm_skill_applicability,
            "llm_retrieval_config": self.llm_retrieval_config,
            "llm_runtime_qualified": self.llm_runtime_qualified,
        }

    @property
    def semantic_sha(self) -> str:
        return sha256(self.semantic_payload)

    @property
    def artifact_sha(self) -> str:
        return sha256({
            **self.semantic_payload, "parent_sha": self.parent_sha,
            "atomic_edit": self.atomic_edit,
        })

    def apply_atomic(self, edit: AtomicHarnessEdit) -> "HarnessArtifactV1":
        if edit.surface in LLM_SURFACES and not self.llm_runtime_qualified:
            raise ValueError("M3 did not unlock LLM-editable surfaces")
        if not hasattr(self, edit.surface):
            raise ValueError(f"unknown surface {edit.surface!r}")
        return replace(
            self, **{edit.surface: edit.value},
            parent_sha=self.artifact_sha, atomic_edit=edit,
        )


SELECTOR_GRID = (0.25, 0.35, 0.5, 0.65, 0.75)
RETRIEVAL_GRID = (4, 8, 12, 16)


@dataclass(frozen=True)
class HarnessEditAuthorizationArtifactV1:
    """Technical authorization for edit suppliers, separate from runtime efficacy."""

    h0_sha: str
    evolution_prereg_sha: str
    llm_trial_authorization_sha: str | None
    allowed_surfaces: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha(self.h0_sha, "h0_sha")
        require_sha(self.evolution_prereg_sha, "evolution_prereg_sha")
        if self.llm_trial_authorization_sha is not None:
            require_sha(self.llm_trial_authorization_sha, "llm_trial_authorization_sha")
        if any(surface not in EDITABLE_SURFACES for surface in self.allowed_surfaces):
            raise ValueError("edit authorization contains an unknown surface")
        if any(surface in LLM_SURFACES for surface in self.allowed_surfaces):
            if self.llm_trial_authorization_sha is None:
                raise ValueError("LLM edit surfaces require technical trial authorization")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class HarnessArtifactV2:
    """Supplier-neutral Harness semantics with typed, bounded atomic edits.

    Evidence flags are intentionally absent from the semantic payload.  Runtime LLM
    permission is carried by ``LLMQualificationArtifactV2``; discovery edit
    permission is carried by ``HarnessEditAuthorizationArtifactV1``.
    """

    action_risk_rule: Mapping[str, Any]
    selector_threshold: float
    supplier_mix: Mapping[str, int]
    retrieval_support_threshold: int
    llm_instruction: str | None = None
    llm_skill_applicability: Mapping[str, Any] | None = None
    llm_retrieval_config: Mapping[str, Any] | None = None
    parent_sha: str | None = None
    atomic_edit: AtomicHarnessEdit | None = None

    def __post_init__(self) -> None:
        if float(self.selector_threshold) not in SELECTOR_GRID:
            raise ValueError("selector threshold must use the frozen grid")
        if self.retrieval_support_threshold not in RETRIEVAL_GRID:
            raise ValueError("retrieval threshold must use the frozen grid")
        keys = set(self.supplier_mix)
        if keys != {"deterministic", "random", "llm"}:
            raise ValueError("supplier mix must contain exactly det/random/llm")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in self.supplier_mix.values()):
            raise ValueError("supplier allocations must be non-negative integers")
        if sum(self.supplier_mix.values()) != 3:
            raise ValueError("vNext supplier mix has exactly three slots")
        predicates = tuple(self.action_risk_rule.get("predicates", ()))
        if len(predicates) > 2 or len(predicates) != len(set(predicates)):
            raise ValueError("risk rule supports at most two unique predicates")
        if (self.parent_sha is None) != (self.atomic_edit is None):
            raise ValueError("parent_sha and atomic_edit must appear together")
        if self.parent_sha is not None:
            require_sha(self.parent_sha, "parent_sha")

    @classmethod
    def engineering_default(cls) -> "HarnessArtifactV2":
        """An Init-safe engineering value; it is explicitly not the formal H0."""
        return cls(
            action_risk_rule={"predicates": ()}, selector_threshold=0.5,
            supplier_mix={"deterministic": 3, "random": 0, "llm": 0},
            retrieval_support_threshold=8,
        )

    @property
    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "vnext-harness/2",
            "action_risk_rule": dict(self.action_risk_rule),
            "selector_threshold": self.selector_threshold,
            "supplier_mix": dict(self.supplier_mix),
            "retrieval_support_threshold": self.retrieval_support_threshold,
            "llm_instruction": self.llm_instruction,
            "llm_skill_applicability": self.llm_skill_applicability,
            "llm_retrieval_config": self.llm_retrieval_config,
        }

    @property
    def semantic_sha(self) -> str:
        return sha256(self.semantic_payload)

    @property
    def artifact_sha(self) -> str:
        return sha256({
            **self.semantic_payload,
            "parent_sha": self.parent_sha,
            "atomic_edit": self.atomic_edit,
        })

    def apply_atomic(
        self,
        edit: AtomicHarnessEdit,
        authorization: HarnessEditAuthorizationArtifactV1,
    ) -> "HarnessArtifactV2":
        if not isinstance(edit, AtomicHarnessEdit):
            raise TypeError("edit must be AtomicHarnessEdit")
        if not isinstance(authorization, HarnessEditAuthorizationArtifactV1):
            raise TypeError("authorization must be HarnessEditAuthorizationArtifactV1")
        if edit.surface not in authorization.allowed_surfaces:
            raise ValueError("edit surface was not preregistered")
        if edit.surface in LLM_SURFACES and authorization.llm_trial_authorization_sha is None:
            raise ValueError("LLM edit lacks technical trial authorization")
        self._validate_transition(edit)
        return replace(
            self,
            **{edit.surface: edit.value},
            parent_sha=self.artifact_sha,
            atomic_edit=edit,
        )

    def _validate_transition(self, edit: AtomicHarnessEdit) -> None:
        old = getattr(self, edit.surface)
        if edit.surface == "selector_threshold":
            if edit.value not in SELECTOR_GRID:
                raise ValueError("selector edit is outside the frozen grid")
            if abs(SELECTOR_GRID.index(float(old)) - SELECTOR_GRID.index(float(edit.value))) != 1:
                raise ValueError("selector edit must move exactly one grid step")
        elif edit.surface == "retrieval_support_threshold":
            if edit.value not in RETRIEVAL_GRID:
                raise ValueError("retrieval edit is outside the frozen grid")
            if abs(RETRIEVAL_GRID.index(int(old)) - RETRIEVAL_GRID.index(int(edit.value))) != 1:
                raise ValueError("retrieval edit must move exactly one grid step")
        elif edit.surface == "supplier_mix":
            new = dict(edit.value)
            if set(new) != set(old) or sum(new.values()) != 3 or any(v < 0 for v in new.values()):
                raise ValueError("supplier edit must preserve the frozen three-slot budget")
            if sum(abs(int(new[key]) - int(old[key])) for key in old) != 2:
                raise ValueError("supplier edit must move exactly one slot")
        elif edit.surface == "action_risk_rule":
            old_pred = set(dict(old).get("predicates", ()))
            new_pred = set(dict(edit.value).get("predicates", ()))
            if len(new_pred) > 2 or len(old_pred ^ new_pred) != 1:
                raise ValueError("risk edit must add or remove one predicate")
        elif edit.surface in LLM_SURFACES:
            if edit.value == old:
                raise ValueError("LLM edit must change one authorized surface")
