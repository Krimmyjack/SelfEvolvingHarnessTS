"""Promotion gate for slow-path proposals.

This module validates mined/proposed slow-path knowledge and compiles the
subset that is deployment-consumable into existing PolicyBundle EditOps. It does
not apply the edits; validators/promoters decide when an EditOp enters a bundle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..policy.edits import AddRiskRule, EditOp, MemoryWrite
from ..policy.risk_policy import RiskRule
from .proposal_schema import SlowProposal, validate_slow_proposal


@dataclass(frozen=True)
class ProposalValidationOutcome:
    proposal: SlowProposal
    accepted: bool
    reason: str | None = None
    edit_op: EditOp | None = None


def _support_n(proposal: SlowProposal) -> int:
    support = proposal.support or {}
    value = support.get("n_unique_cases", support.get("n", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _risk_rule_from_proposal(proposal: SlowProposal) -> RiskRule:
    payload = proposal.payload
    provenance: dict[str, Any] = dict(proposal.provenance)
    payload_provenance = payload.get("provenance")
    if isinstance(payload_provenance, dict):
        provenance.update(payload_provenance)
    return RiskRule(
        rule_id=str(payload["rule_id"]),
        when=dict(payload["when"]),
        then=dict(payload["then"]),
        scope=str(payload["scope"]),
        provenance=provenance,
    )


def compile_slow_proposal_to_edit(proposal: SlowProposal) -> EditOp | None:
    """Compile supported slow proposals into existing deployment EditOps."""
    if proposal.kind == "MemoryWrite":
        return MemoryWrite(dict(proposal.payload))
    if proposal.kind == "ProposeRiskRule":
        return AddRiskRule(_risk_rule_from_proposal(proposal))
    return None


class PromotionGate:
    """Validate slow proposals before a separate validator/promoter can apply them."""

    def __init__(self, *, min_support: int = 1):
        self.min_support = int(min_support)

    def validate(self, proposal: SlowProposal) -> ProposalValidationOutcome:
        reason = validate_slow_proposal(proposal)
        if reason:
            return ProposalValidationOutcome(proposal=proposal, accepted=False, reason=reason)
        support_n = _support_n(proposal)
        if support_n < self.min_support:
            return ProposalValidationOutcome(
                proposal=proposal,
                accepted=False,
                reason=f"insufficient support: {support_n} < {self.min_support}",
            )
        edit_op = compile_slow_proposal_to_edit(proposal)
        if edit_op is None:
            return ProposalValidationOutcome(
                proposal=proposal,
                accepted=False,
                reason=f"unsupported proposal kind for promotion: {proposal.kind}",
            )
        return ProposalValidationOutcome(proposal=proposal, accepted=True, edit_op=edit_op)

