"""Utility-bound memory evidence rows for Skill+Memory composition."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

MEMORY_EVIDENCE_SCHEMA = "memory_evidence_v1"
MEMORY_EVIDENCE_V2_SCHEMA = "memory_evidence_v2"
MemoryType = Literal["case", "utility", "risk", "contrast", "strategy", "skill"]
MemoryRole = Literal["diagnostic", "recommend", "warn", "ban", "abstain", "contrast"]
MEMORY_PACKET_BUCKETS = (
    "case_memory",
    "utility_memory",
    "risk_memory",
    "contrast_memory",
    "strategy_memory",
    "skill_evidence",
)
_FORBIDDEN = {
    "L_test",
    "raw_loss",
    "selected_loss",
    "oracle",
    "oracle_action",
    "oracle_loss",
    "future",
    "label",
    "target",
    "history",
    "series",
    "raw_series",
    "arms",
    "X_t",
}


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items() if str(k) not in _FORBIDDEN}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _delta(raw_loss: float | None, selected_loss: float | None) -> tuple[float | None, float | None]:
    if raw_loss is None or selected_loss is None:
        return None, None
    utility = round(float(raw_loss) - float(selected_loss), 12)
    harm = round(max(0.0, float(selected_loss) - float(raw_loss)), 12)
    return utility, harm


@dataclass(frozen=True)
class MemoryEvidence:
    task: str
    pattern_region: str
    skill_id: str | None = None
    action_id: str | None = None
    program: Mapping[str, Any] = field(default_factory=dict)
    utility_delta_vs_raw: float | None = None
    harm_delta_vs_raw: float | None = None
    support: Mapping[str, Any] = field(default_factory=dict)
    subgroup: str = ""
    validator_result: Mapping[str, Any] = field(default_factory=dict)
    failure_signature: str | None = None
    source_domain: str = ""
    version: str = MEMORY_EVIDENCE_SCHEMA
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_packet_row(self) -> dict[str, Any]:
        """Return the LLM-safe memory row; raw losses and labels are intentionally omitted."""

        return {
            "schema": self.version,
            "task": self.task,
            "pattern_region": self.pattern_region,
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "program": _clean(dict(self.program)),
            "utility_delta_vs_raw": self.utility_delta_vs_raw,
            "harm_delta_vs_raw": self.harm_delta_vs_raw,
            "support": _clean(dict(self.support)),
            "subgroup": self.subgroup,
            "validator_result": _clean(dict(self.validator_result)),
            "failure_signature": self.failure_signature,
            "source_domain": self.source_domain,
            "provenance": _clean(dict(self.provenance)),
        }


@dataclass(frozen=True)
class MemoryEvidenceV2:
    """Typed, role-aware memory row for deployment evidence packets."""

    task: str
    pattern_region: str
    memory_type: MemoryType
    role: MemoryRole
    scope: Mapping[str, Any] = field(default_factory=dict)
    skill_id: str | None = None
    action_id: str | None = None
    program: Mapping[str, Any] = field(default_factory=dict)
    utility_delta_vs_raw: float | None = None
    harm_delta_vs_raw: float | None = None
    confidence: Mapping[str, Any] = field(default_factory=dict)
    support: Mapping[str, Any] = field(default_factory=dict)
    retrieval: Mapping[str, Any] = field(default_factory=dict)
    conflict: Mapping[str, Any] = field(default_factory=dict)
    validator_result: Mapping[str, Any] = field(default_factory=dict)
    failure_signature: str | None = None
    source_domain: str = ""
    lifecycle_status: str = "candidate"
    evidence_refs: tuple[str, ...] = ()
    version: str = MEMORY_EVIDENCE_V2_SCHEMA
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_packet_row(self) -> dict[str, Any]:
        return {
            "schema": self.version,
            "task": self.task,
            "pattern_region": self.pattern_region,
            "memory_type": self.memory_type,
            "role": self.role,
            "scope": _clean(dict(self.scope)),
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "program": _clean(dict(self.program)),
            "utility_delta_vs_raw": self.utility_delta_vs_raw,
            "harm_delta_vs_raw": self.harm_delta_vs_raw,
            "confidence": _clean(dict(self.confidence)),
            "support": _clean(dict(self.support)),
            "retrieval": _clean(dict(self.retrieval)),
            "conflict": _clean(dict(self.conflict)),
            "validator_result": _clean(dict(self.validator_result)),
            "failure_signature": self.failure_signature,
            "source_domain": self.source_domain,
            "lifecycle_status": self.lifecycle_status,
            "evidence_refs": list(self.evidence_refs),
            "provenance": _clean(dict(self.provenance)),
        }


def _infer_role(
    *,
    memory_type: MemoryType,
    role: MemoryRole | None,
    utility: float | None,
    harm: float | None,
) -> MemoryRole:
    if role is not None:
        return role
    if memory_type == "risk" or (harm is not None and harm > 0.0):
        return "warn"
    if memory_type == "contrast":
        return "contrast"
    if memory_type in ("case", "strategy"):
        return "diagnostic"
    if utility is not None and utility > 0.0:
        return "recommend"
    return "diagnostic"


def build_memory_evidence(
    *,
    task: str,
    pattern_region: str,
    skill_id: str | None = None,
    action_id: str | None = None,
    program: Mapping[str, Any] | None = None,
    raw_loss: float | None = None,
    selected_loss: float | None = None,
    support: Mapping[str, Any] | None = None,
    subgroup: str = "",
    validator_result: Mapping[str, Any] | None = None,
    failure_signature: str | None = None,
    source_domain: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> MemoryEvidence:
    """Build a memory item after validation has made utility evidence available."""

    utility, harm = _delta(raw_loss, selected_loss)
    result = dict(validator_result or {})
    sig = failure_signature if failure_signature is not None else result.get("failure_signature")
    return MemoryEvidence(
        task=str(task),
        pattern_region=str(pattern_region),
        skill_id=skill_id,
        action_id=action_id,
        program=dict(program or {}),
        utility_delta_vs_raw=utility,
        harm_delta_vs_raw=harm,
        support=dict(support or {}),
        subgroup=str(subgroup or ""),
        validator_result=result,
        failure_signature=sig,
        source_domain=str(source_domain or ""),
        provenance=dict(provenance or {}),
    )


def build_memory_evidence_v2(
    *,
    task: str,
    pattern_region: str,
    memory_type: MemoryType = "utility",
    role: MemoryRole | None = None,
    scope: Mapping[str, Any] | None = None,
    skill_id: str | None = None,
    action_id: str | None = None,
    program: Mapping[str, Any] | None = None,
    raw_loss: float | None = None,
    selected_loss: float | None = None,
    utility_delta_vs_raw: float | None = None,
    harm_delta_vs_raw: float | None = None,
    confidence: Mapping[str, Any] | None = None,
    support: Mapping[str, Any] | None = None,
    retrieval: Mapping[str, Any] | None = None,
    conflict: Mapping[str, Any] | None = None,
    validator_result: Mapping[str, Any] | None = None,
    failure_signature: str | None = None,
    source_domain: str = "",
    lifecycle_status: str = "candidate",
    evidence_refs: tuple[str, ...] | list[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> MemoryEvidenceV2:
    """Build a typed V2 row, deriving utility/harm when raw losses are present."""

    utility, harm = _delta(raw_loss, selected_loss)
    if utility_delta_vs_raw is not None:
        utility = float(utility_delta_vs_raw)
    if harm_delta_vs_raw is not None:
        harm = float(harm_delta_vs_raw)
    result = dict(validator_result or {})
    sig = failure_signature if failure_signature is not None else result.get("failure_signature")
    inferred_role = _infer_role(memory_type=memory_type, role=role, utility=utility, harm=harm)
    return MemoryEvidenceV2(
        task=str(task),
        pattern_region=str(pattern_region),
        memory_type=memory_type,
        role=inferred_role,
        scope=dict(scope or {}),
        skill_id=skill_id,
        action_id=action_id,
        program=dict(program or {}),
        utility_delta_vs_raw=utility,
        harm_delta_vs_raw=harm,
        confidence=dict(confidence or {}),
        support=dict(support or {}),
        retrieval=dict(retrieval or {}),
        conflict=dict(conflict or {}),
        validator_result=result,
        failure_signature=sig,
        source_domain=str(source_domain or ""),
        lifecycle_status=str(lifecycle_status or "candidate"),
        evidence_refs=tuple(str(v) for v in evidence_refs),
        provenance=dict(provenance or {}),
    )


def memory_packet_bucket(row: Mapping[str, Any]) -> str:
    """Return the EvidencePacket bucket for a sanitized memory row."""

    memory_type = str(row.get("memory_type") or "")
    role = str(row.get("role") or "")
    if memory_type == "risk" or role in {"warn", "ban", "abstain"}:
        return "risk_memory"
    if memory_type == "contrast" or role == "contrast":
        return "contrast_memory"
    if memory_type == "strategy":
        return "strategy_memory"
    if memory_type == "skill":
        return "skill_evidence"
    if memory_type == "case" or role == "diagnostic":
        return "case_memory"
    return "utility_memory"
