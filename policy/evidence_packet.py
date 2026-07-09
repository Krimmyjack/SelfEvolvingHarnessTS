"""Leakage-safe Skill + Memory evidence packet for fast-path LLM composition."""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from ..e32_policy import P_FEATS
from .skills import SkillSpec

PACKET_SCHEMA = "skill_memory_evidence_packet_v1"
FORBIDDEN_KEYS = {
    "L_test",
    "loss",
    "losses",
    "oracle",
    "oracle_action",
    "oracle_loss",
    "arms",
    "X_t",
    "history",
    "series",
    "raw_series",
    "target",
    "y",
    "label",
    "future",
}
DEFAULT_CANDIDATE_SCHEMA = {
    "type": "typed_candidate_v1",
    "allowed_fields": [
        "skill_id",
        "action_id",
        "ProgramSpec",
        "risk_rule",
        "abstain_to_raw",
        "rationale",
        "evidence_refs",
    ],
    "required_any": ["action_id", "ProgramSpec", "risk_rule", "abstain_to_raw"],
    "output_contract": "candidate must be compiled and checked by ActionCompiler plus RiskPolicy/SafetyGate before execution",
}


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _safe_scalar(value.item())
    return value


def _sanitize(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _sanitize(v) for k, v in value.items() if str(k) not in FORBIDDEN_KEYS}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return _safe_scalar(value)


def _pattern_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    x_p = list(record.get("X_p") or [])
    if len(x_p) != len(P_FEATS):
        raise ValueError(f"record {record.get('uid')!r} has X_p length {len(x_p)}, expected {len(P_FEATS)}")
    struct_feats = {name: float(value) for name, value in zip(P_FEATS, x_p)}
    return {
        "cell": str(record.get("cell") or ""),
        "snr": float(record["snr"]),
        "missing_rate": float(record["miss_rate"]),
        "struct_feats": struct_feats,
    }


def _skill_cards(skills: Mapping[str, SkillSpec] | Sequence[SkillSpec] | Sequence[Mapping[str, Any]], max_skills: int) -> list[dict[str, Any]]:
    values = list(skills.values()) if isinstance(skills, Mapping) else list(skills)
    cards = []
    for idx, skill in enumerate(values[:max_skills], start=1):
        if isinstance(skill, Mapping):
            card = _sanitize(dict(skill))
            card.setdefault("rank", idx)
            cards.append(card)
            continue
        cards.append(
            {
                "rank": idx,
                "name": skill.name,
                "actions": dict(skill.actions),
                "allowed_actions": list(skill.actions.values()),
                "applicability": skill.applicability,
                "risk": skill.risk,
                "fallback": skill.fallback,
                "version": skill.version,
            }
        )
    return cards


def _memory_summary(memory_rows: Any, max_memory: int) -> dict[str, list[Any]]:
    if memory_rows is None:
        return {"prior_fragments": [], "failure_warnings": []}
    if isinstance(memory_rows, Mapping):
        prior = memory_rows.get("prior_fragments") or memory_rows.get("successes") or []
        failures = memory_rows.get("failure_warnings") or memory_rows.get("failures") or []
    else:
        prior = memory_rows
        failures = []
    return {
        "prior_fragments": [_sanitize(row) for row in list(prior)[:max_memory]],
        "failure_warnings": [_sanitize(row) for row in list(failures)[:max_memory]],
    }


def build_evidence_packet(
    record: Mapping[str, Any],
    *,
    skills: Mapping[str, SkillSpec] | Sequence[SkillSpec] | Sequence[Mapping[str, Any]],
    memory_rows: Any,
    action_menu_meta: Mapping[str, Any],
    support_stats: Mapping[str, Any] | None = None,
    harm_stats: Mapping[str, Any] | None = None,
    risk_constraints: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    incumbent_decision: Mapping[str, Any] | None = None,
    candidate_schema: Mapping[str, Any] | None = None,
    max_skills: int = 8,
    max_memory: int = 5,
) -> dict[str, Any]:
    """Build the bounded evidence object intended to be serialized into an LLM prompt."""

    return {
        "schema": PACKET_SCHEMA,
        "task": {"type": "forecast"},
        "pattern": _pattern_summary(record),
        "skills": _skill_cards(skills, max_skills),
        "memory": _memory_summary(memory_rows, max_memory),
        "action_menu": _sanitize(dict(action_menu_meta)),
        "support": _sanitize(dict(support_stats or {})),
        "harm_stats": _sanitize(dict(harm_stats or {})),
        "risk_constraints": _sanitize(risk_constraints or []),
        "incumbent_decision": _sanitize(dict(incumbent_decision or {})),
        "candidate_schema": _sanitize(dict(candidate_schema or DEFAULT_CANDIDATE_SCHEMA)),
        "safety_constraints": [
            "Use only observable pattern features, skill cards, memory summaries, support stats, action menu metadata, and risk constraints.",
            "Do not infer from held-out losses, oracle actions, arm picks, true degradation labels, or raw full series.",
            "Return only a typed candidate that can be compiled and checked by the existing safety gate before execution.",
        ],
        "provenance": {
            "source_uid": str(record.get("uid") or ""),
            "leakage_excluded": sorted(FORBIDDEN_KEYS),
        },
    }