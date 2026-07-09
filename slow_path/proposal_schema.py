"""Typed slow-path proposal schema for deployment evidence promotion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..policy.edits import MemoryWrite as MemoryWriteEdit
from ..policy.edits import bundle_v0
from ..policy.risk_policy import RiskRule

ALLOWED_PROPOSAL_KINDS = frozenset({
    "ProposeRiskRule",
    "ProposeSkillSpec",
    "MemoryWrite",
    "ProposePatternSpecEdit",
    "PolicyBundlePatch",
})


@dataclass(frozen=True)
class SlowProposal:
    kind: str
    scope: str
    payload: Mapping[str, Any]
    evidence_refs: Sequence[str] = field(default_factory=tuple)
    support: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scope": self.scope,
            "payload": dict(self.payload),
            "evidence_refs": list(self.evidence_refs),
            "support": dict(self.support),
            "provenance": dict(self.provenance),
        }


def _validate_risk_rule_payload(proposal: SlowProposal) -> str | None:
    payload = proposal.payload
    required = ("rule_id", "when", "then", "scope")
    missing = [key for key in required if key not in payload]
    if missing:
        return f"risk rule invalid: missing fields {missing}"
    if payload["scope"] != proposal.scope:
        return "risk rule invalid: payload scope must match proposal scope"
    if not isinstance(payload["when"], Mapping):
        return "risk rule invalid: when must be a mapping"
    if not isinstance(payload["then"], Mapping):
        return "risk rule invalid: then must be a mapping"
    provenance = dict(proposal.provenance)
    provenance.update(dict(payload.get("provenance", {})) if isinstance(payload.get("provenance"), Mapping) else {})
    rule = RiskRule(
        rule_id=str(payload["rule_id"]),
        when=dict(payload["when"]),
        then=dict(payload["then"]),
        scope=str(payload["scope"]),
        provenance=provenance,
    )
    reason = rule.validate()
    if reason:
        return f"risk rule invalid: {reason}"
    return None


def _validate_memory_write_payload(proposal: SlowProposal) -> str | None:
    payload = dict(proposal.payload)
    if payload.get("scope") != proposal.scope:
        return "memory write invalid: payload scope must match proposal scope"
    reason = MemoryWriteEdit(payload).validate(bundle_v0())
    if reason:
        return f"memory write invalid: {reason}"
    return None


def validate_slow_proposal(proposal: SlowProposal) -> str | None:
    """Return None when a proposal is well-formed and deployment-consumable."""
    if proposal.kind not in ALLOWED_PROPOSAL_KINDS:
        return f"unknown proposal kind: {proposal.kind}"
    if not proposal.scope:
        return "proposal requires non-empty scope"
    if not isinstance(proposal.payload, Mapping) or not proposal.payload:
        return "proposal requires non-empty payload"
    if not proposal.evidence_refs:
        return "proposal requires evidence_refs"
    if proposal.kind == "ProposeRiskRule":
        return _validate_risk_rule_payload(proposal)
    if proposal.kind == "MemoryWrite":
        return _validate_memory_write_payload(proposal)
    return None
