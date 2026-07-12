"""Deterministic fast-path ablation runner.

The runner executes named arms over the same records and series. It is intended
for local, reproducible checks before any real LLM/API composer is introduced.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..memory import EvidenceRecord, EvidenceStore, MEMORY_PACKET_BUCKETS, memory_packet_bucket
from ..policy.action_spec import ActionMenu
from ..policy.escalation import (
    DownstreamValidationResult,
    EscalationConfig,
    EscalationDecision,
    ExecutedFastPathDecision,
    SafetyGateDecision,
    decide_fast_path,
    emit_fast_path_evidence,
    execute_fast_path_decision,
    validate_fast_path_output,
)
from ..policy.evidence_packet import build_evidence_packet
from ..policy.skill_memory_composer import TypedCandidate


Composer = Callable[[Mapping[str, Any]], TypedCandidate | Mapping[str, Any] | str | None]
DownstreamValidator = Callable[[Any, Any, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class FastPathAblationArm:
    name: str
    kind: str = "escalation"
    use_skills: bool = True
    use_memory: bool = True
    use_composer: bool = False
    use_safety: bool = True
    composer: Composer | None = None
    support_stats: Mapping[str, Any] = field(default_factory=dict)
    harm_stats: Mapping[str, Any] = field(default_factory=dict)
    config: EscalationConfig | None = None
    memory_mode: str | None = None

    @classmethod
    def raw(cls, *, raw_action: str = "v_none") -> "FastPathAblationArm":
        return cls(
            name="raw",
            kind="raw",
            use_skills=False,
            use_memory=False,
            use_composer=False,
            use_safety=True,
            config=EscalationConfig(raw_action=raw_action),
        )


@dataclass(frozen=True)
class FastPathAblationResult:
    arm_name: str
    uid: str
    decision: EscalationDecision
    executed: ExecutedFastPathDecision
    validation: DownstreamValidationResult
    evidence: EvidenceRecord


def phase4_fast_path_ablation_arms(composer: Composer | None = None) -> list[FastPathAblationArm]:
    """Return the Stage-4 no-API ablation matrix for MemoryV2 attribution."""
    needs_composition = {"needs_composition": True, "support_score": 0.1}
    return [
        FastPathAblationArm.raw(),
        FastPathAblationArm(name="deterministic_router", use_memory=False),
        FastPathAblationArm(name="skill_only_deterministic", use_memory=False),
        FastPathAblationArm(
            name="positive_memory_only",
            use_skills=False,
            use_memory=True,
            use_composer=True,
            use_safety=False,
            composer=composer,
            support_stats=needs_composition,
            memory_mode="positive",
        ),
        FastPathAblationArm(
            name="risk_memory_only",
            use_skills=False,
            use_memory=True,
            use_composer=True,
            use_safety=True,
            composer=composer,
            support_stats=needs_composition,
            memory_mode="risk",
        ),
        FastPathAblationArm(
            name="positive_risk_memory",
            use_skills=False,
            use_memory=True,
            use_composer=True,
            use_safety=True,
            composer=composer,
            support_stats=needs_composition,
            memory_mode="positive_risk",
        ),
        FastPathAblationArm(name="skill_memory_deterministic", use_memory=True, memory_mode="positive_risk"),
        FastPathAblationArm(
            name="composer_skill",
            use_memory=False,
            use_composer=True,
            composer=composer,
            support_stats=needs_composition,
        ),
        FastPathAblationArm(
            name="composer_skill_memory",
            use_memory=True,
            use_composer=True,
            use_safety=False,
            composer=composer,
            support_stats=needs_composition,
            memory_mode="positive_risk",
        ),
        FastPathAblationArm(
            name="composer_skill_memory_safety",
            use_memory=True,
            use_composer=True,
            use_safety=True,
            composer=composer,
            support_stats=needs_composition,
            memory_mode="positive_risk",
        ),
    ]


def standard_fast_path_ablation_arms(composer: Composer | None = None) -> list[FastPathAblationArm]:
    """Return the current no-API ablation matrix using an injected stub composer."""
    return phase4_fast_path_ablation_arms(composer=composer)


def _raw_decision(
    record: Mapping[str, Any],
    *,
    action_menu_meta: Mapping[str, Any],
    memory_rows: Any,
    config: EscalationConfig,
) -> EscalationDecision:
    candidate = TypedCandidate(
        skill_id="identity",
        action_id=config.raw_action,
        rationale="raw_ablation_baseline",
    )
    packet = build_evidence_packet(
        record,
        skills=[],
        memory_rows=memory_rows,
        action_menu_meta=action_menu_meta,
        support_stats={},
        harm_stats={},
        incumbent_decision={"route": "raw", "action_id": config.raw_action},
    )
    safety = SafetyGateDecision(
        accepted=True,
        serve_action_id=config.raw_action,
        fallback_raw=False,
        reasons=(),
    )
    return EscalationDecision(
        route="raw",
        proposal_route="raw_baseline",
        action_id=config.raw_action,
        candidate=candidate,
        packet=packet,
        safety=safety,
        composer_called=False,
    )


def _without_safety(decision: EscalationDecision) -> EscalationDecision:
    action = decision.candidate.action_id or decision.action_id
    safety = SafetyGateDecision(accepted=True, serve_action_id=action, fallback_raw=False, reasons=())
    return EscalationDecision(
        route=decision.proposal_route,
        proposal_route=decision.proposal_route,
        action_id=action,
        candidate=decision.candidate,
        packet=decision.packet,
        safety=safety,
        composer_called=decision.composer_called,
        program_action=decision.program_action,
    )


def _merged_stats(base_by_uid: Mapping[str, Mapping[str, Any]] | None, uid: str, arm_stats: Mapping[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    if base_by_uid and uid in base_by_uid:
        stats.update(dict(base_by_uid[uid]))
    stats.update(dict(arm_stats or {}))
    return stats


def _packet_memory_row(row: Any) -> Any:
    if hasattr(row, "to_packet_row"):
        return row.to_packet_row()
    if isinstance(row, Mapping):
        return dict(row)
    return row


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _inferred_memory_bucket(row: Any) -> str:
    if not isinstance(row, Mapping):
        return "case_memory"
    if row.get("memory_type") or row.get("role"):
        return memory_packet_bucket(row)
    harm = _float_or_none(row.get("harm_delta_vs_raw"))
    if harm is not None and harm > 0.0:
        return "risk_memory"
    if row.get("failure_signature"):
        return "risk_memory"
    utility = _float_or_none(row.get("utility_delta_vs_raw"))
    if utility is not None and utility > 0.0:
        return "utility_memory"
    return "case_memory"


def _row_with_inferred_v2_role(row: Any) -> Any:
    packet_row = _packet_memory_row(row)
    if not isinstance(packet_row, Mapping):
        return packet_row
    if packet_row.get("memory_type") or packet_row.get("role"):
        return packet_row
    bucket = _inferred_memory_bucket(packet_row)
    typed = dict(packet_row)
    if bucket == "risk_memory":
        typed.setdefault("memory_type", "risk")
        typed.setdefault("role", "warn")
    elif bucket == "utility_memory":
        typed.setdefault("memory_type", "utility")
        typed.setdefault("role", "recommend")
    else:
        typed.setdefault("memory_type", "case")
        typed.setdefault("role", "diagnostic")
    return typed


def _iter_memory_rows(memory_rows: Any) -> list[Any]:
    if memory_rows is None:
        return []
    if isinstance(memory_rows, Mapping):
        rows: list[Any] = []
        for key in ("prior_fragments", "successes", "failure_warnings", "failures", *MEMORY_PACKET_BUCKETS):
            value = memory_rows.get(key)
            if isinstance(value, (list, tuple)):
                rows.extend(value)
        return rows
    return list(memory_rows)


def _filter_memory_rows(memory_rows: Any, memory_mode: str | None) -> Any:
    if memory_rows is None or memory_mode in (None, "all"):
        return memory_rows
    allowed_by_mode = {
        "positive": {"utility_memory"},
        "risk": {"risk_memory"},
        "positive_risk": {"utility_memory", "risk_memory"},
        "contrast": {"contrast_memory"},
    }
    allowed = allowed_by_mode.get(str(memory_mode))
    if allowed is None:
        raise ValueError(f"unknown memory_mode {memory_mode!r}")
    rows = []
    for row in _iter_memory_rows(memory_rows):
        typed = _row_with_inferred_v2_role(row)
        if _inferred_memory_bucket(typed) in allowed:
            rows.append(typed)
    return rows


def run_fast_path_ablation(
    records: Sequence[Mapping[str, Any]],
    series_by_uid: Mapping[str, Any],
    *,
    arms: Sequence[FastPathAblationArm],
    action_menu: ActionMenu,
    memory_by_uid: Mapping[str, Any] | None = None,
    support_by_uid: Mapping[str, Mapping[str, Any]] | None = None,
    harm_by_uid: Mapping[str, Mapping[str, Any]] | None = None,
    validator: DownstreamValidator | None = None,
    store: EvidenceStore | None = None,
    task_type: str = "forecast",
    harness_version: int = 0,
) -> list[FastPathAblationResult]:
    """Run named fast-path ablation arms and write comparable EvidenceRecords."""

    evidence_store = store if store is not None else EvidenceStore()
    menu_meta = action_menu.to_dict()
    results: list[FastPathAblationResult] = []
    for index, record in enumerate(records):
        uid = str(record.get("uid") or index)
        if uid not in series_by_uid:
            raise KeyError(f"missing series for uid {uid!r}")
        x = series_by_uid[uid]
        for arm in arms:
            config = arm.config or EscalationConfig()
            raw_memory_rows = (memory_by_uid or {}).get(uid) if arm.use_memory else None
            memory_rows = _filter_memory_rows(raw_memory_rows, arm.memory_mode) if arm.use_memory else None
            if arm.kind == "raw":
                decision = _raw_decision(
                    record,
                    action_menu_meta=menu_meta,
                    memory_rows=memory_rows,
                    config=config,
                )
            else:
                decision = decide_fast_path(
                    record,
                    action_menu_meta=menu_meta,
                    memory_rows=memory_rows,
                    support_stats=_merged_stats(support_by_uid, uid, arm.support_stats),
                    harm_stats=_merged_stats(harm_by_uid, uid, arm.harm_stats),
                    config=config,
                    composer=arm.composer if arm.use_composer else None,
                    skill_cards_override=None if arm.use_skills else [],
                )
                if not arm.use_safety:
                    decision = _without_safety(decision)
            executed = execute_fast_path_decision(decision, record, action_menu, x, task_type=task_type)
            validation = validate_fast_path_output(x, executed, task_type=task_type, validator=validator)
            evidence = emit_fast_path_evidence(
                record,
                decision,
                executed,
                validation,
                store=evidence_store,
                batch_id=f"ablation:{arm.name}",
                harness_version=harness_version,
            )
            evidence.routing["ablation_arm"] = arm.name
            evidence.routing["ablation_flags"] = {
                "use_skills": arm.use_skills,
                "use_memory": arm.use_memory,
                "use_composer": arm.use_composer,
                "use_safety": arm.use_safety,
                "memory_mode": arm.memory_mode,
            }
            results.append(
                FastPathAblationResult(
                    arm_name=arm.name,
                    uid=uid,
                    decision=decision,
                    executed=executed,
                    validation=validation,
                    evidence=evidence,
                )
            )
    return results


def _json_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return _json_float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _mean(values: Sequence[float]) -> float | None:
    vals = []
    for value in values:
        numeric = _json_float(value)
        if numeric is not None:
            vals.append(numeric)
    if not vals:
        return None
    return round(sum(vals) / len(vals), 12)


def _fraction(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(float(count) / float(total), 12)


def _result_to_record(result: FastPathAblationResult) -> dict[str, Any]:
    downstream = result.evidence.verification_result.get("downstream") or {}
    safety = result.decision.safety
    return {
        "arm_name": result.arm_name,
        "uid": result.uid,
        "route": result.decision.route,
        "proposal_route": result.decision.proposal_route,
        "action_id": result.decision.action_id,
        "candidate": result.decision.candidate.to_dict(),
        "composer_called": bool(result.decision.composer_called),
        "safety_accepted": bool(safety.accepted),
        "safety_reasons": list(safety.reasons),
        "safety_evidence_refs": list(safety.evidence_refs),
        "status": result.executed.status,
        "execution_ok": bool(result.executed.execution_ok),
        "passed": bool(result.validation.passed),
        "role_b_score": _json_float(result.validation.role_b_score),
        "failure_signature": result.validation.failure_signature,
        "downstream": _json_safe(dict(downstream)),
        "cell_id": result.evidence.cell_id,
        "batch_id": result.evidence.batch_id,
    }


def _downstream_metric(result: FastPathAblationResult, key: str) -> float | None:
    downstream = result.evidence.verification_result.get("downstream") or {}
    if key not in downstream:
        return None
    try:
        numeric = float(downstream[key])
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def summarize_fast_path_ablation_results(results: Sequence[FastPathAblationResult]) -> dict[str, Any]:
    """Aggregate comparable per-arm metrics for a fast-path ablation run."""
    arm_order: list[str] = []
    grouped: dict[str, list[FastPathAblationResult]] = {}
    for result in results:
        if result.arm_name not in grouped:
            arm_order.append(result.arm_name)
            grouped[result.arm_name] = []
        grouped[result.arm_name].append(result)

    reference_arm = "raw"
    raw_utility_by_uid: dict[str, float] = {}
    for row in grouped.get(reference_arm, []):
        utility = _downstream_metric(row, "utility_delta_vs_raw")
        if utility is not None and row.uid not in raw_utility_by_uid:
            raw_utility_by_uid[row.uid] = utility

    arms: dict[str, Any] = {}
    for arm_name in arm_order:
        rows = grouped[arm_name]
        action_counts = Counter(row.decision.action_id for row in rows)
        status_counts = Counter(row.executed.status for row in rows)
        safety_reasons: Counter = Counter()
        safety_refs: Counter = Counter()
        utilities: list[float] = []
        harms: list[float] = []
        scores: list[float] = []
        lifts_vs_raw_arm: list[float] = []
        cell_utilities: dict[str, list[float]] = {}
        for row in rows:
            safety_reasons.update(row.decision.safety.reasons)
            safety_refs.update(row.decision.safety.evidence_refs)
            score = _json_float(row.validation.role_b_score)
            if score is not None:
                scores.append(score)
            utility = _downstream_metric(row, "utility_delta_vs_raw")
            if utility is not None:
                utilities.append(utility)
                cell_utilities.setdefault(str(row.evidence.cell_id), []).append(utility)
                if row.uid in raw_utility_by_uid:
                    lifts_vs_raw_arm.append(utility - raw_utility_by_uid[row.uid])
            harm = _downstream_metric(row, "harm_delta_vs_raw")
            if harm is not None:
                harms.append(harm)
        executed_count = sum(1 for row in rows if row.executed.status == "executed")
        fallback_count = sum(1 for row in rows if row.executed.status != "executed")
        rejected_count = sum(1 for row in rows if not row.decision.safety.accepted)
        abstain_count = sum(1 for row in rows if row.decision.candidate.abstain_to_raw)
        worst_cell = None
        if cell_utilities:
            cell_means = [_mean(values) for values in cell_utilities.values()]
            finite_cell_means = [value for value in cell_means if value is not None]
            worst_cell = min(finite_cell_means) if finite_cell_means else None
        arms[arm_name] = {
            "n": len(rows),
            "passed": sum(1 for row in rows if row.validation.passed),
            "executed": executed_count,
            "raw_fallback": fallback_count,
            "serve_fraction": _fraction(executed_count, len(rows)),
            "fallback_fraction": _fraction(fallback_count, len(rows)),
            "composer_called": sum(1 for row in rows if row.decision.composer_called),
            "safety_rejected": rejected_count,
            "safety_rejection_fraction": _fraction(rejected_count, len(rows)),
            "abstain_to_raw": abstain_count,
            "abstain_fraction": _fraction(abstain_count, len(rows)),
            "action_counts": dict(sorted(action_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "safety_reason_counts": dict(sorted(safety_reasons.items())),
            "safety_evidence_ref_count": sum(safety_refs.values()),
            "safety_evidence_ref_counts": dict(sorted(safety_refs.items())),
            "mean_role_b_score": _mean(scores),
            "mean_utility_delta_vs_raw": _mean(utilities),
            "mean_harm_delta_vs_raw": _mean(harms),
            "mean_lift_vs_raw_arm": _mean(lifts_vs_raw_arm),
            "cell_count": len(cell_utilities),
            "worst_cell_mean_utility_delta_vs_raw": worst_cell,
        }
    return {
        "schema": "fast_path_ablation_summary_v2",
        "reference_arm": reference_arm,
        "reference_metric": "utility_delta_vs_raw",
        "lift_metric": "utility_delta_vs_raw - raw_arm.utility_delta_vs_raw for same uid",
        "n_results": len(results),
        "arm_order": arm_order,
        "arms": arms,
    }


def write_fast_path_ablation_report(
    results: Sequence[FastPathAblationResult],
    out_dir: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write report.json and records.jsonl for a fast-path ablation run."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    records = [_result_to_record(result) for result in results]
    report = {
        "schema": "fast_path_ablation_report_v1",
        "metadata": dict(metadata or {}),
        "summary": summarize_fast_path_ablation_results(results),
    }
    (path / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with (path / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return report

