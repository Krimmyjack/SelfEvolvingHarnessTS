"""Utility-bound memory evidence rows for Skill+Memory composition."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

MEMORY_EVIDENCE_SCHEMA = "memory_evidence_v1"
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