"""Versioned, immutable artifact contracts for the benchmark-native vNext track."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._canonical import require_sha, sha256


@dataclass(frozen=True)
class VNextMethodArtifact:
    method_id: str
    benchmark_version: str
    pattern_binding_sha: str
    action_menu_sha: str
    candidate_grammar_sha: str
    action_eligibility_sha: str
    harness_sha: str
    supplier_kind: str
    selector_sha: str
    agent_model_id: str | None
    prompt_sha: str | None
    decoding_sha: str | None
    dependency_fingerprint: Mapping[str, str]
    budget_sha: str
    parent_sha: str | None = None
    atomic_edit: Mapping[str, Any] | None = None
    semantic_sha: str = ""

    def __post_init__(self) -> None:
        if not self.method_id or not self.benchmark_version or not self.supplier_kind:
            raise ValueError("method, benchmark, and supplier identifiers are required")
        for name in (
            "pattern_binding_sha", "action_menu_sha", "candidate_grammar_sha",
            "action_eligibility_sha", "harness_sha", "selector_sha", "budget_sha",
        ):
            require_sha(getattr(self, name), name)
        for name in ("prompt_sha", "decoding_sha", "parent_sha"):
            value = getattr(self, name)
            if value is not None:
                require_sha(value, name)
        expected = sha256(self.semantic_payload())
        if self.semantic_sha and self.semantic_sha != expected:
            raise ValueError("semantic_sha does not match the method semantic payload")
        if not self.semantic_sha:
            object.__setattr__(self, "semantic_sha", expected)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "benchmark_version": self.benchmark_version,
            "pattern_binding_sha": self.pattern_binding_sha,
            "action_menu_sha": self.action_menu_sha,
            "candidate_grammar_sha": self.candidate_grammar_sha,
            "action_eligibility_sha": self.action_eligibility_sha,
            "harness_sha": self.harness_sha,
            "supplier_kind": self.supplier_kind,
            "selector_sha": self.selector_sha,
            "agent_model_id": self.agent_model_id,
            "prompt_sha": self.prompt_sha,
            "decoding_sha": self.decoding_sha,
            "dependency_fingerprint": dict(self.dependency_fingerprint),
            "budget_sha": self.budget_sha,
        }

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class VNextMethodArtifactV2:
    """Replay-complete method identity used by the hardened vNext protocol.

    V1 is retained for compatibility.  V2 adds the supplier policy, input
    contract, environment, and comparison identity that V1 could not express.
    """

    method_id: str
    benchmark_version: str
    pattern_binding_sha: str
    action_menu_sha: str
    candidate_grammar_sha: str
    action_eligibility_sha: str
    harness_sha: str
    supplier_kind: str
    supplier_policy_sha: str
    selector_sha: str
    method_input_contract_sha: str
    environment_sha: str
    seed_book_sha: str
    agent_model_id: str | None
    prompt_sha: str | None
    decoding_sha: str | None
    dependency_fingerprint: Mapping[str, str]
    budget_sha: str
    comparison_base_sha: str | None = None
    parent_sha: str | None = None
    atomic_edit: Mapping[str, Any] | None = None
    semantic_sha: str = ""

    def __post_init__(self) -> None:
        if not self.method_id or not self.benchmark_version or not self.supplier_kind:
            raise ValueError("method, benchmark, and supplier identifiers are required")
        for name in (
            "pattern_binding_sha", "action_menu_sha", "candidate_grammar_sha",
            "action_eligibility_sha", "harness_sha", "supplier_policy_sha",
            "selector_sha", "method_input_contract_sha", "environment_sha",
            "seed_book_sha", "budget_sha",
        ):
            require_sha(getattr(self, name), name)
        for name in (
            "prompt_sha", "decoding_sha", "comparison_base_sha", "parent_sha",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha(value, name)
        if self.parent_sha is None and self.atomic_edit is not None:
            raise ValueError("atomic_edit requires a real parent_sha")
        if self.parent_sha is not None and self.atomic_edit is None:
            raise ValueError("parent_sha is reserved for real atomic edits")
        expected = sha256(self.semantic_payload())
        if self.semantic_sha and self.semantic_sha != expected:
            raise ValueError("semantic_sha does not match the V2 method payload")
        if not self.semantic_sha:
            object.__setattr__(self, "semantic_sha", expected)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "vnext-method-artifact/2",
            "benchmark_version": self.benchmark_version,
            "pattern_binding_sha": self.pattern_binding_sha,
            "action_menu_sha": self.action_menu_sha,
            "candidate_grammar_sha": self.candidate_grammar_sha,
            "action_eligibility_sha": self.action_eligibility_sha,
            "harness_sha": self.harness_sha,
            "supplier_kind": self.supplier_kind,
            "supplier_policy_sha": self.supplier_policy_sha,
            "selector_sha": self.selector_sha,
            "method_input_contract_sha": self.method_input_contract_sha,
            "environment_sha": self.environment_sha,
            "seed_book_sha": self.seed_book_sha,
            "agent_model_id": self.agent_model_id,
            "prompt_sha": self.prompt_sha,
            "decoding_sha": self.decoding_sha,
            "dependency_fingerprint": dict(self.dependency_fingerprint),
            "budget_sha": self.budget_sha,
        }

    @property
    def artifact_sha(self) -> str:
        return sha256({
            **self.semantic_payload(),
            "method_id": self.method_id,
            "comparison_base_sha": self.comparison_base_sha,
            "parent_sha": self.parent_sha,
            "atomic_edit": self.atomic_edit,
        })


@dataclass(frozen=True)
class MethodEvaluationArtifact:
    method_sha: str
    data_sha: str
    judge_sha: str
    effect_mode: str
    result_sha: str
    cost_sha: str
    estimand: str = "published_joint_v02"

    def __post_init__(self) -> None:
        for name in ("method_sha", "data_sha", "judge_sha", "result_sha", "cost_sha"):
            require_sha(getattr(self, name), name)
        if self.estimand != "published_joint_v02":
            raise ValueError("vNext headline estimand must remain published_joint_v02")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class ConfirmationArtifactV2:
    method_sha: str
    dry_run_sha: str
    support_b_result_sha: str
    terminal_status: str
    roster_sha: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    TERMINAL = frozenset({"passed", "failed_gate", "invalid", "timeout", "infrastructure_error"})

    def __post_init__(self) -> None:
        for name in ("method_sha", "dry_run_sha", "support_b_result_sha", "roster_sha"):
            require_sha(getattr(self, name), name)
        if self.terminal_status not in self.TERMINAL:
            raise ValueError(f"unknown terminal confirmation status {self.terminal_status!r}")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)


@dataclass(frozen=True)
class ConfirmationArtifactV3:
    """Support-B confirmation bound to an access-before-read terminal receipt."""

    method_sha: str
    dry_run_sha: str
    roster_sha: str
    access_manifest_sha: str
    reservation_sha: str
    terminal_event_sha: str
    support_b_result_sha: str
    terminal_status: str
    environment_sha: str
    budget_sha: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    TERMINAL = frozenset({
        "passed", "failed_gate", "invalid", "timeout", "budget_exceeded",
        "dependency_failure", "failed_infrastructure_terminal",
    })

    def __post_init__(self) -> None:
        for name in (
            "method_sha", "dry_run_sha", "roster_sha", "access_manifest_sha",
            "reservation_sha", "terminal_event_sha", "support_b_result_sha",
            "environment_sha", "budget_sha",
        ):
            require_sha(getattr(self, name), name)
        if self.terminal_status not in self.TERMINAL:
            raise ValueError(f"unknown V3 confirmation status {self.terminal_status!r}")

    @property
    def artifact_sha(self) -> str:
        return sha256(self)
