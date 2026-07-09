"""Typed Skill+Memory composer output parsing and validation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class TypedCandidate:
    skill_id: str | None = None
    action_id: str | None = None
    program_spec: Mapping[str, Any] = field(default_factory=dict)
    risk_rule: Mapping[str, Any] = field(default_factory=dict)
    abstain_to_raw: bool = False
    rationale: str = ""
    evidence_refs: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "action_id": self.action_id,
            "ProgramSpec": dict(self.program_spec),
            "risk_rule": dict(self.risk_rule),
            "abstain_to_raw": bool(self.abstain_to_raw),
            "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs),
        }


def _extract_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        from ..llm import extract_json

        return extract_json(raw)
    except Exception:
        return None


def _allowed_actions(packet: Mapping[str, Any]) -> set[str]:
    menu = packet.get("action_menu") or {}
    if isinstance(menu, Mapping):
        allowed = menu.get("allowed_actions")
        if allowed is not None:
            return {str(v) for v in allowed}
        actions = menu.get("actions")
        if isinstance(actions, Mapping):
            return {str(k) for k in actions}
        if actions is not None:
            return {str(v) for v in actions}
    return set()


def _skill_actions(packet: Mapping[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for card in packet.get("skills") or []:
        if not isinstance(card, Mapping) or "name" not in card:
            continue
        name = str(card["name"])
        allowed = card.get("allowed_actions")
        if allowed is not None:
            out[name] = {str(v) for v in allowed}
            continue
        actions = card.get("actions")
        if isinstance(actions, Mapping):
            out[name] = {str(v) for v in actions.values()}
        else:
            out[name] = set()
    return out


def parse_typed_candidate(raw: str | Mapping[str, Any], packet: Mapping[str, Any]) -> TypedCandidate | None:
    """Parse and validate a typed candidate against the evidence packet surface."""

    spec = dict(raw) if isinstance(raw, Mapping) else _extract_json(str(raw))
    if not isinstance(spec, Mapping):
        return None
    skill_id = spec.get("skill_id")
    action_id = spec.get("action_id")
    program_spec = spec.get("ProgramSpec", spec.get("program_spec", {})) or {}
    risk_rule = spec.get("risk_rule") or {}
    abstain_to_raw = bool(spec.get("abstain_to_raw", False))
    rationale = str(spec.get("rationale", ""))[:400]
    evidence_refs = [str(v) for v in spec.get("evidence_refs", [])]

    if not (action_id or program_spec or risk_rule or abstain_to_raw):
        return None
    skill_to_actions = _skill_actions(packet)
    if skill_id is not None and str(skill_id) not in skill_to_actions:
        return None
    allowed_actions = _allowed_actions(packet)
    if action_id is not None:
        action_id = str(action_id)
        if allowed_actions and action_id not in allowed_actions:
            return None
        if skill_id is not None:
            skill_allowed = skill_to_actions.get(str(skill_id), set())
            if skill_allowed and action_id not in skill_allowed:
                return None
    if not isinstance(program_spec, Mapping) or not isinstance(risk_rule, Mapping):
        return None
    return TypedCandidate(
        skill_id=str(skill_id) if skill_id is not None else None,
        action_id=action_id,
        program_spec=dict(program_spec),
        risk_rule=dict(risk_rule),
        abstain_to_raw=abstain_to_raw,
        rationale=rationale,
        evidence_refs=tuple(evidence_refs),
    )


def compose_skill_memory_candidate(
    packet: Mapping[str, Any],
    llm: Callable[[str, str], str],
    *,
    nonce: int = 0,
) -> TypedCandidate | None:
    """Ask an LLM-like callable for one typed candidate and validate the result."""

    system = (
        "You are a time-series Skill-Memory composer. Output JSON only. "
        "Use only the evidence packet and return a typed_candidate_v1 object."
    )
    user = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    try:
        raw = llm(system, user, nonce=nonce)
    except TypeError:
        raw = llm(system, user)
    return parse_typed_candidate(raw, packet)