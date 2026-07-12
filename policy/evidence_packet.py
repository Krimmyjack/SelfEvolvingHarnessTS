"""Leakage-safe Skill + Memory evidence packet for fast-path composition."""
from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from ..e32_policy import P_FEATS
from ..memory.evidence_schema import MEMORY_PACKET_BUCKETS, memory_packet_bucket
from .skills import SkillSpec
from .task_spec import TaskSpec, forecast_task_spec_v1

PACKET_SCHEMA = "skill_memory_evidence_packet_v1"
PACKET_SCHEMA_V2 = "skill_memory_evidence_packet_v2"
FORBIDDEN_KEYS = {
    "L_test",
    "loss",
    "losses",
    "raw_loss",
    "selected_loss",
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


def _packet_row(row: Any) -> Any:
    if hasattr(row, "to_packet_row"):
        return row.to_packet_row()
    if isinstance(row, Mapping):
        return dict(row)
    return row


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


def _skill_cards(
    skills: Mapping[str, SkillSpec] | Sequence[SkillSpec] | Sequence[Mapping[str, Any]],
    max_skills: int,
) -> list[dict[str, Any]]:
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


def _empty_memory_summary() -> dict[str, list[Any]]:
    summary = {bucket: [] for bucket in MEMORY_PACKET_BUCKETS}
    summary["prior_fragments"] = []
    summary["failure_warnings"] = []
    return summary


def _add_memory_row(summary: dict[str, list[Any]], row: Any, *, legacy_bucket: str | None = None) -> None:
    packet_row = _sanitize(_packet_row(row))
    if legacy_bucket is not None:
        summary[legacy_bucket].append(packet_row)
    if isinstance(packet_row, Mapping):
        bucket = memory_packet_bucket(packet_row)
        summary[bucket].append(packet_row)


def _memory_summary(memory_rows: Any, max_memory: int) -> dict[str, list[Any]]:
    summary = _empty_memory_summary()
    if memory_rows is None:
        return summary

    if isinstance(memory_rows, Mapping):
        for legacy_key, fallback_key in (("prior_fragments", "successes"), ("failure_warnings", "failures")):
            rows = memory_rows.get(legacy_key)
            if rows is None:
                rows = memory_rows.get(fallback_key) or []
            for row in list(rows)[:max_memory]:
                _add_memory_row(summary, row, legacy_bucket=legacy_key)
        for bucket in MEMORY_PACKET_BUCKETS:
            for row in list(memory_rows.get(bucket) or [])[:max_memory]:
                _add_memory_row(summary, row)
    else:
        for row in list(memory_rows)[:max_memory]:
            packet_row = _packet_row(row)
            if isinstance(packet_row, Mapping) and (packet_row.get("memory_type") or packet_row.get("role")):
                _add_memory_row(summary, packet_row)
            else:
                _add_memory_row(summary, packet_row, legacy_bucket="prior_fragments")

    return {key: rows[:max_memory] for key, rows in summary.items()}


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
    task_spec: TaskSpec | None = None,
    max_skills: int = 8,
    max_memory: int = 5,
) -> dict[str, Any]:
    """Build the bounded evidence object intended to be serialized into an LLM prompt.

    P0（Final_Plan_CodeAgentFirst_2026-07-09）：task 契约显式化——不传 task_spec 时默认
    forecast_task_spec_v1()（与历史隐式 forecast 口径一致），packet["task"] 保留 legacy
    `type` 键并携带完整 TaskSpec 字段；provenance 记 task_spec_sha。
    """
    spec = task_spec if task_spec is not None else forecast_task_spec_v1()

    return {
        "schema": PACKET_SCHEMA,
        "task": spec.to_packet_dict(),
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
            "Treat risk_memory and contrast_memory as first-class counter-evidence, not as action recommendations.",
            "Return only a typed candidate that can be compiled and checked by the existing safety gate before execution.",
        ],
        "provenance": {
            "source_uid": str(record.get("uid") or ""),
            "task_spec_sha": spec.sha(),
            "leakage_excluded": [key for key in sorted(FORBIDDEN_KEYS) if key not in {"raw_loss", "selected_loss"}],
            "memory_schema": "memory_evidence_v2_compatible",
        },
    }


# ═════════════════ Packet v2（P1：连续证据 + trace + allowed_grammar）═════════════════
#
# R1（slice v2 教训）：喂给 code agent 的证据必须是连续特征化 + 结构化 trace——二值读数
# 使 LLM 蒸馏出过泛化规则（B+ 赢在消费 17 维连续证据）。v2 在 v1 之上加三个通道并对
# continuous_evidence 做数值性 fail-loud 校验。

def _validate_continuous_evidence(ce: Mapping[str, Any]) -> None:
    for action_id, stats in ce.items():
        if not isinstance(stats, Mapping):
            raise ValueError(f"continuous_evidence[{action_id!r}] 须为 stats object，得到 {stats!r}")
        for key, value in stats.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"continuous_evidence[{action_id!r}][{key!r}] 须为数值"
                    f"（R1：连续证据，禁二值读数），得到 {value!r}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"continuous_evidence[{action_id!r}][{key!r}] 须为有限数，得到 {value!r}")


def default_allowed_grammar() -> dict[str, Any]:
    """告诉 code agent 它被允许产出什么（ProgramSpec v1 语法面；单一真源=program_edit 常量）。"""
    from .program_edit import (
        DENOISERS, IMPUTERS, OUTLIERS, WINDOWED, WINDOW_GRID,
        _GUARD_COMPARATORS, _guard_feats,
    )
    return {
        "grammar": "program_spec_v1",
        "max_steps": 3,
        "imputers": list(IMPUTERS),
        "outliers": list(OUTLIERS),
        "denoisers": list(DENOISERS),
        "windowed_ops": sorted(WINDOWED),
        "window_grid": list(WINDOW_GRID),
        "beta_range": [0.0, 1.0],
        "guard_features": sorted(_guard_feats()),
        "guard_comparators": list(_GUARD_COMPARATORS),
        "fallback_semantics": ["v_raw_identity", "v_impute_linear"],
        "output_contract": "emit exactly one JSON ProgramSpec v1 object; "
                           "invalid or empty output counts as no-op under ITT",
    }


def build_evidence_packet_v2(
    record: Mapping[str, Any],
    *,
    continuous_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    trace_summaries: Sequence[Mapping[str, Any]] | None = None,
    allowed_grammar: Mapping[str, Any] | None = None,
    max_traces: int = 16,
    **kwargs: Any,
) -> dict[str, Any]:
    """v2 = v1 + continuous_evidence（数值 fail-loud）+ trace_summaries（泄漏 lint）+ allowed_grammar。"""
    packet = build_evidence_packet(record, **kwargs)
    ce = dict(continuous_evidence or {})
    _validate_continuous_evidence(ce)
    packet["schema"] = PACKET_SCHEMA_V2
    packet["continuous_evidence"] = _sanitize(ce)
    packet["trace_summaries"] = _sanitize(list(trace_summaries or [])[:max_traces])
    packet["allowed_grammar"] = dict(allowed_grammar) if allowed_grammar else default_allowed_grammar()
    packet["provenance"]["packet_schema"] = PACKET_SCHEMA_V2
    return packet
