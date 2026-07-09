"""Escalation-style fast-path decision layer.

This module wires the deployment-facing evidence surface without executing the
chosen action. It keeps the LLM as an optional composer over retrieved skills,
memory evidence, and explicit constraints; the final action is guarded by a
small SafetyGate before any downstream compiler/overlay may consume it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..e32_policy import P_FEATS
from ..fast_path.execute import execute
from ..fast_path.verify import role_b_score
from ..memory import EvidenceRecord, EvidenceStore
from .action_spec import ActionCompiler, ActionMenu
from .evidence_packet import build_evidence_packet
from .skill_memory_composer import TypedCandidate, parse_typed_candidate
from .skill_retriever import retrieve_skill_cards


Composer = Callable[[Mapping[str, Any]], TypedCandidate | Mapping[str, Any] | str | None]


@dataclass(frozen=True)
class EscalationConfig:
    raw_action: str = "v_none"
    top_k_skills: int = 5
    max_memory: int = 5
    min_deterministic_skill_score: float = 0.55
    max_support_score: float | None = None
    max_harm_rate: float | None = None


@dataclass(frozen=True)
class SafetyGateDecision:
    accepted: bool
    serve_action_id: str
    fallback_raw: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EscalationDecision:
    route: str
    proposal_route: str
    action_id: str
    candidate: TypedCandidate
    packet: Mapping[str, Any]
    safety: SafetyGateDecision
    composer_called: bool = False


@dataclass(frozen=True)
class CompiledFastPathDecision:
    action_id: str
    compiled: bool
    program: Any | None
    conditioning_key: Mapping[str, Any] | None = None
    reason: str = ""


@dataclass(frozen=True)
class ExecutedFastPathDecision:
    status: str
    artifact: Any
    compiled: CompiledFastPathDecision
    execution_ok: bool
    execution_result: Any | None = None
    decision: EscalationDecision | None = None


@dataclass(frozen=True)
class DownstreamValidationResult:
    passed: bool
    result: Mapping[str, Any]
    role_b_score: float
    failure_signature: str | None = None


def _allowed_actions_from_menu(action_menu_meta: Mapping[str, Any]) -> set[str]:
    values = action_menu_meta.get("allowed_actions")
    if values is not None:
        return {str(v) for v in values}
    actions = action_menu_meta.get("actions")
    if isinstance(actions, Mapping):
        return {str(k) for k in actions}
    if actions is not None:
        return {str(v) for v in actions}
    return set()


def _packet_skill_actions(packet: Mapping[str, Any]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for card in packet.get("skills") or []:
        if not isinstance(card, Mapping):
            continue
        name = card.get("name")
        if name is None:
            continue
        allowed = card.get("allowed_actions")
        if allowed is not None:
            out[str(name)] = {str(v) for v in allowed}
            continue
        actions = card.get("actions")
        if isinstance(actions, Mapping):
            out[str(name)] = {str(v) for v in actions.values()}
    return out


def _support_score(support_stats: Mapping[str, Any]) -> float | None:
    for key in ("support_score", "support_distance", "distance", "score"):
        if key not in support_stats:
            continue
        try:
            return float(support_stats[key])
        except (TypeError, ValueError):
            return None
    return None


def _weak_support(support_stats: Mapping[str, Any], config: EscalationConfig) -> bool:
    if bool(support_stats.get("out_of_support")):
        return True
    if config.max_support_score is None:
        return False
    score = _support_score(support_stats)
    if score is None:
        return True
    return score > float(config.max_support_score)


def _needs_composer(support_stats: Mapping[str, Any], config: EscalationConfig) -> bool:
    return (
        _weak_support(support_stats, config)
        or bool(support_stats.get("evidence_conflict"))
        or bool(support_stats.get("needs_composition"))
    )


def _memory_for_packet(memory_rows: Any) -> dict[str, list[Any]]:
    if memory_rows is None:
        return {"prior_fragments": [], "failure_warnings": []}
    if isinstance(memory_rows, Mapping):
        prior = memory_rows.get("prior_fragments") or memory_rows.get("successes") or []
        failures = memory_rows.get("failure_warnings") or memory_rows.get("failures") or []
    else:
        prior = memory_rows
        failures = []
    return {
        "prior_fragments": [_packet_row(row) for row in list(prior)],
        "failure_warnings": [_packet_row(row) for row in list(failures)],
    }


def _packet_row(row: Any) -> Any:
    if hasattr(row, "to_packet_row"):
        return row.to_packet_row()
    if isinstance(row, Mapping):
        return dict(row)
    return row


def _deterministic_candidate(
    skill_cards: Sequence[Mapping[str, Any]],
    config: EscalationConfig,
) -> TypedCandidate | None:
    if not skill_cards:
        return None
    top = skill_cards[0]
    try:
        score = float(top.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if score < config.min_deterministic_skill_score:
        return None
    actions = list(top.get("allowed_actions") or [])
    if not actions:
        return None
    return TypedCandidate(
        skill_id=str(top.get("name")) if top.get("name") is not None else None,
        action_id=str(actions[0]),
        rationale="deterministic_high_support_skill_match",
    )


def _raw_candidate(config: EscalationConfig, reason: str) -> TypedCandidate:
    return TypedCandidate(
        skill_id="identity",
        action_id=config.raw_action,
        abstain_to_raw=True,
        rationale=reason,
    )


def _coerce_candidate(raw: Any, packet: Mapping[str, Any]) -> TypedCandidate | None:
    if raw is None:
        return None
    if isinstance(raw, TypedCandidate):
        return raw
    if isinstance(raw, str):
        return parse_typed_candidate(raw, packet)
    if isinstance(raw, Mapping):
        return TypedCandidate(
            skill_id=str(raw["skill_id"]) if raw.get("skill_id") is not None else None,
            action_id=str(raw["action_id"]) if raw.get("action_id") is not None else None,
            program_spec=dict(raw.get("ProgramSpec", raw.get("program_spec", {})) or {}),
            risk_rule=dict(raw.get("risk_rule", {}) or {}),
            abstain_to_raw=bool(raw.get("abstain_to_raw", False)),
            rationale=str(raw.get("rationale", ""))[:400],
            evidence_refs=tuple(str(v) for v in raw.get("evidence_refs", ()) or ()),
        )
    return None


def safety_gate_candidate(
    candidate: TypedCandidate,
    packet: Mapping[str, Any],
    *,
    support_stats: Mapping[str, Any] | None = None,
    harm_stats: Mapping[str, Any] | None = None,
    config: EscalationConfig | None = None,
) -> SafetyGateDecision:
    cfg = config or EscalationConfig()
    support = dict(support_stats or {})
    harms = dict(harm_stats or {})
    reasons: list[str] = []
    action = candidate.action_id

    if candidate.abstain_to_raw:
        reasons.append("candidate_abstain_to_raw")
    if action is None:
        reasons.append("missing_action")

    allowed = _allowed_actions_from_menu(packet.get("action_menu") or {})
    if action is not None and allowed and action not in allowed:
        reasons.append("action_not_in_menu")

    skill_actions = _packet_skill_actions(packet)
    if candidate.skill_id is not None:
        allowed_for_skill = skill_actions.get(str(candidate.skill_id))
        if allowed_for_skill is None:
            reasons.append("unknown_skill")
        elif action is not None and action not in allowed_for_skill:
            reasons.append("skill_action_mismatch")

    if action != cfg.raw_action and _weak_support(support, cfg):
        reasons.append("weak_support")

    if cfg.max_harm_rate is not None and action != cfg.raw_action:
        try:
            harm_rate = float(harms.get("harm_rate", 0.0))
        except (TypeError, ValueError):
            harm_rate = 0.0
        if harm_rate > float(cfg.max_harm_rate):
            reasons.append("harm_rate_exceeds_policy")

    if reasons:
        return SafetyGateDecision(
            accepted=False,
            serve_action_id=cfg.raw_action,
            fallback_raw=True,
            reasons=tuple(dict.fromkeys(reasons)),
        )
    return SafetyGateDecision(
        accepted=True,
        serve_action_id=action or cfg.raw_action,
        fallback_raw=False,
        reasons=(),
    )


def _task_type(record: Mapping[str, Any], default: str) -> str:
    task = record.get("task")
    if isinstance(task, Mapping):
        return str(task.get("type") or default)
    if task is not None:
        return str(task)
    return default


def conditioning_key_from_record(
    record: Mapping[str, Any],
    *,
    task_type: str = "forecast",
) -> dict[str, Any]:
    """Build the minimal ActionCompiler conditioning key from a packet record."""

    x_p = list(record.get("X_p") or [])
    if len(x_p) != len(P_FEATS):
        raise ValueError(f"record {record.get('uid')!r} has X_p length {len(x_p)}, expected {len(P_FEATS)}")
    struct = {
        "SNR": float(record.get("snr", record.get("SNR", 0.0))),
        "missing_rate": float(record.get("miss_rate", record.get("missing_rate", 0.0))),
    }
    struct.update({name: float(value) for name, value in zip(P_FEATS, x_p)})
    task = _task_type(record, task_type)
    return {
        "pattern": {"struct_feats": struct, "quality_profile": {"problem_types": []}},
        "task": {"type": task},
        "cell_id": str(record.get("cell") or ""),
        "pattern_bin": str(record.get("pattern_bin") or record.get("cell") or ""),
    }


def compile_fast_path_decision(
    decision: EscalationDecision,
    record: Mapping[str, Any],
    action_menu: ActionMenu,
    *,
    compiler: ActionCompiler | None = None,
    task_type: str = "forecast",
    compile_raw_fallback: bool = False,
) -> CompiledFastPathDecision:
    """Compile an accepted fast-path decision into the existing Program contract."""

    if decision.route == "raw_fallback" and not compile_raw_fallback:
        return CompiledFastPathDecision(
            action_id=decision.action_id,
            compiled=False,
            program=None,
            reason="raw_fallback_not_compiled",
        )
    if not decision.safety.accepted and not compile_raw_fallback:
        return CompiledFastPathDecision(
            action_id=decision.action_id,
            compiled=False,
            program=None,
            reason="safety_rejected_not_compiled",
        )
    if decision.action_id not in action_menu:
        raise ValueError(f"action {decision.action_id!r} not in ActionMenu {action_menu.version!r}")
    key = conditioning_key_from_record(record, task_type=task_type)
    comp = compiler or ActionCompiler()
    program = comp.to_program(action_menu.actions[decision.action_id], key)
    return CompiledFastPathDecision(
        action_id=decision.action_id,
        compiled=True,
        program=program,
        conditioning_key=key,
        reason="compiled",
    )


def execute_fast_path_decision(
    decision: EscalationDecision,
    record: Mapping[str, Any],
    action_menu: ActionMenu,
    x: Any,
    *,
    compiler: ActionCompiler | None = None,
    task_type: str = "forecast",
    compile_raw_fallback: bool = False,
) -> ExecutedFastPathDecision:
    """Compile and execute an accepted fast-path decision, otherwise return raw."""

    raw = np.asarray(x, dtype=float).ravel()
    compiled = compile_fast_path_decision(
        decision,
        record,
        action_menu,
        compiler=compiler,
        task_type=task_type,
        compile_raw_fallback=compile_raw_fallback,
    )
    if not compiled.compiled or compiled.program is None:
        return ExecutedFastPathDecision(
            status=compiled.reason or "raw_fallback_not_compiled",
            artifact=raw.copy(),
            compiled=compiled,
            execution_ok=False,
            execution_result=None,
            decision=decision,
        )

    try:
        result = execute(compiled.program, raw)
    except Exception as exc:
        return ExecutedFastPathDecision(
            status="execution_exception_raw_fallback",
            artifact=raw.copy(),
            compiled=compiled,
            execution_ok=False,
            execution_result={"error": f"{type(exc).__name__}: {exc}"},
            decision=decision,
        )

    if not result.ok or result.artifact is None:
        return ExecutedFastPathDecision(
            status="execution_failed_raw_fallback",
            artifact=raw.copy(),
            compiled=compiled,
            execution_ok=False,
            execution_result=result,
            decision=decision,
        )
    artifact = np.asarray(result.artifact, dtype=float).ravel()
    if artifact.shape != raw.shape or not np.all(np.isfinite(artifact)):
        return ExecutedFastPathDecision(
            status="invalid_artifact_raw_fallback",
            artifact=raw.copy(),
            compiled=compiled,
            execution_ok=False,
            execution_result=result,
            decision=decision,
        )
    return ExecutedFastPathDecision(
        status="executed",
        artifact=artifact,
        compiled=compiled,
        execution_ok=True,
        execution_result=result,
        decision=decision,
    )

def _validation_failure_signature(executed: ExecutedFastPathDecision, passed: bool) -> str | None:
    if passed:
        return None
    reasons = []
    if executed.status:
        reasons.append(executed.status)
    decision = executed.decision
    if decision is not None:
        reasons.extend(str(reason) for reason in decision.safety.reasons)
    return ":".join(reasons) if reasons else "downstream_failed"


def validate_fast_path_output(
    raw: Any,
    executed: ExecutedFastPathDecision,
    *,
    task_type: str = "forecast",
    validator: Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> DownstreamValidationResult:
    """Validate the executed fast-path artifact before EvidenceStore writeback."""

    raw_arr = np.asarray(raw, dtype=float).ravel()
    artifact = np.asarray(executed.artifact, dtype=float).ravel()
    rb = role_b_score(raw_arr, artifact, task_type)
    context = {
        "task_type": task_type,
        "executed": executed,
        "compiled": executed.compiled,
        "decision": executed.decision,
    }
    if validator is None:
        result: dict[str, Any] = {
            "validator": "role_b_proxy",
            "passed": bool(executed.execution_ok),
            "role_b_score": rb,
            "output_status": executed.status,
        }
    else:
        payload = dict(validator(raw_arr, artifact, context) or {})
        payload.setdefault("validator", getattr(validator, "__name__", "custom_downstream"))
        payload.setdefault("passed", bool(executed.execution_ok))
        payload.setdefault("role_b_score", rb)
        payload.setdefault("output_status", executed.status)
        result = payload
    passed = bool(result.get("passed", False)) and bool(executed.execution_ok)
    result["passed"] = passed
    failure = result.get("failure_signature") or _validation_failure_signature(executed, passed)
    return DownstreamValidationResult(
        passed=passed,
        result=result,
        role_b_score=rb,
        failure_signature=str(failure) if failure is not None else None,
    )


def _program_for_evidence(executed: ExecutedFastPathDecision) -> Mapping[str, Any]:
    program = executed.compiled.program
    if executed.compiled.compiled and program is not None:
        return program.to_dict()
    return {"source": "raw_fallback", "steps": [], "note": executed.status}


def _execution_trace_for_evidence(executed: ExecutedFastPathDecision) -> list[Mapping[str, Any]]:
    result = executed.execution_result
    trace = getattr(result, "trace", None)
    if trace is None:
        return []
    return [dict(row) for row in trace]


def _routing_for_evidence(decision: EscalationDecision) -> dict[str, Any]:
    packet = decision.packet or {}
    return {
        "route": decision.route,
        "proposal_route": decision.proposal_route,
        "selected_action": decision.action_id,
        "composer_called": decision.composer_called,
        "candidate": decision.candidate.to_dict(),
        "safety": {
            "accepted": decision.safety.accepted,
            "serve_action_id": decision.safety.serve_action_id,
            "fallback_raw": decision.safety.fallback_raw,
            "reasons": list(decision.safety.reasons),
        },
        "packet": {
            "schema": packet.get("schema"),
            "provenance": packet.get("provenance", {}),
        },
        "support": packet.get("support", {}),
        "harm_stats": packet.get("harm_stats", {}),
    }


def emit_fast_path_evidence(
    source_record: Mapping[str, Any],
    decision: EscalationDecision,
    executed: ExecutedFastPathDecision,
    validation: DownstreamValidationResult,
    *,
    store: EvidenceStore | None = None,
    batch_id: str = "",
    harness_version: int = 0,
) -> EvidenceRecord:
    """Build and optionally write an EvidenceRecord for the escalation fast path."""

    key = dict(executed.compiled.conditioning_key or conditioning_key_from_record(source_record))
    cell_id = str(key.get("cell_id") or source_record.get("cell") or "")
    verification_result = {
        "passed": bool(validation.passed),
        "gate_results": [
            {
                "name": "safety_gate",
                "passed": bool(decision.safety.accepted),
                "detail": ",".join(decision.safety.reasons),
            },
            {
                "name": "downstream_validator",
                "passed": bool(validation.passed),
                "detail": validation.failure_signature or "",
            },
        ],
        "failure_signature": validation.failure_signature,
        "role_b_score": validation.role_b_score,
        "output_status": executed.status,
        "downstream": dict(validation.result),
    }
    evidence = EvidenceRecord(
        conditioning_key=key,
        cell_id=cell_id,
        harness_version=int(harness_version),
        program=_program_for_evidence(executed),
        execution_trace=_execution_trace_for_evidence(executed),
        verification_result=verification_result,
        batch_id=str(batch_id or ""),
        routing=_routing_for_evidence(decision),
    )
    if store is not None:
        store.write(evidence)
    return evidence
def decide_fast_path(
    record: Mapping[str, Any],
    *,
    action_menu_meta: Mapping[str, Any],
    memory_rows: Any = None,
    support_stats: Mapping[str, Any] | None = None,
    harm_stats: Mapping[str, Any] | None = None,
    risk_constraints: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    incumbent_decision: Mapping[str, Any] | None = None,
    config: EscalationConfig | None = None,
    composer: Composer | None = None,
    skill_cards_override: Sequence[Mapping[str, Any]] | None = None,
) -> EscalationDecision:
    """Return a typed fast-path decision plus the evidence packet it used.

    The function intentionally stops before execution unless the caller passes
    the decision to execute_fast_path_decision.
    """

    cfg = config or EscalationConfig()
    support = dict(support_stats or {})
    harms = dict(harm_stats or {})
    if skill_cards_override is None:
        skill_cards = retrieve_skill_cards(
            record,
            action_menu=action_menu_meta,
            top_k=cfg.top_k_skills,
        )
    else:
        skill_cards = [dict(card) for card in skill_cards_override]
    deterministic = _deterministic_candidate(skill_cards, cfg)
    packet = build_evidence_packet(
        record,
        skills=skill_cards,
        memory_rows=_memory_for_packet(memory_rows),
        action_menu_meta=action_menu_meta,
        support_stats=support,
        harm_stats=harms,
        risk_constraints=risk_constraints,
        incumbent_decision=incumbent_decision
        or {
            "route": "deterministic" if deterministic is not None else "raw",
            "action_id": deterministic.action_id if deterministic is not None else cfg.raw_action,
        },
        max_skills=cfg.top_k_skills,
        max_memory=cfg.max_memory,
    )

    composer_called = False
    if deterministic is not None and not _needs_composer(support, cfg):
        candidate = deterministic
        proposal_route = "deterministic"
    elif composer is not None:
        composer_called = True
        candidate = _coerce_candidate(composer(packet), packet)
        proposal_route = "llm_composer"
    elif deterministic is not None:
        candidate = deterministic
        proposal_route = "deterministic"
    else:
        candidate = _raw_candidate(cfg, "no_safe_candidate")
        proposal_route = "raw_fallback"

    if candidate is None:
        candidate = _raw_candidate(cfg, "composer_no_candidate")
        safety = SafetyGateDecision(
            accepted=False,
            serve_action_id=cfg.raw_action,
            fallback_raw=True,
            reasons=("composer_no_candidate",),
        )
    else:
        safety = safety_gate_candidate(
            candidate,
            packet,
            support_stats=support,
            harm_stats=harms,
            config=cfg,
        )

    route = proposal_route if safety.accepted else "raw_fallback"
    return EscalationDecision(
        route=route,
        proposal_route=proposal_route,
        action_id=safety.serve_action_id,
        candidate=candidate,
        packet=packet,
        safety=safety,
        composer_called=composer_called,
    )

