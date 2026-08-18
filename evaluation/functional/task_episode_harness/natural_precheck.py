"""Outcome-blind A5/A3 precheck on the natural development bank.

No Target outcome is opened.  A3 sees no Source Experience; A5 sees every
actual natural trajectory (success, ineffective, low-reliability and abstain),
not only the winsorize winner.  Because Target would reuse the same K1
series/origin pool, the only valid memory label here is TARGET_HISTORY_MEMORY.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from evaluation.functional.task_episode_harness.natural_flow import (
    NATURAL_POOL,
)
from evaluation.functional.task_episode_harness.normal_flow import (
    _nf_call,
)
from evaluation.functional.task_episode_harness.runner import REPORT_REL

PRECHECK_TARGET_CONTEXT = {
    "task_episode_id": "natural_precheck_target",
    "support_origins": (1800, 2856, 3648),
    "scope_feature": "local_robust_z_peak",
    "scope_bin": "high",
    "candidate_pool": list(NATURAL_POOL),
    "budget": 3,
}
HYPOTHETICAL_FIRST_PROBE = {
    "support_gain": 0.020,
    "support_se_block": 0.018,
    "support_gain_over_se": 0.020 / 0.018,
    "note": "hypothetical low-confidence positive probe; no real outcome opened",
}


def _natural_trajectory_summaries(report: dict[str, Any]) -> list[dict[str, Any]]:
    natural = report.get("natural_flow") or {}
    bank = report.get("natural_bank") or []
    episodes_by_id = {
        str(ep.get("episode_id")): ep for ep in bank if isinstance(ep, dict)
    }
    summaries = []
    for task in natural.get("episodes", []):
        for probe in task.get("probes", []):
            ep = episodes_by_id.get(str(probe.get("episode", {}).get("episode_id")))
            if not isinstance(ep, dict):
                continue
            summaries.append({
                "task_episode_id": task.get("task_episode_id"),
                "program": probe.get("program"),
                "support_gain": probe.get("support_gain"),
                "support_se_block": probe.get("support_se_block"),
                "support_gain_over_se": probe.get("support_gain_over_se"),
                "agent_decision": (probe.get("agent_decision") or {}).get(
                    "decision"
                ),
                "mechanical_gate": probe.get("mechanical_gate"),
                "relation": ep.get("relation"),
                "local_status": ep.get("local_status"),
                "delayed_gain": (ep.get("delayed_response") or {}).get("gain"),
                "delayed_se_block": (ep.get("delayed_response") or {}).get(
                    "se_block"
                ),
                "delayed_gain_over_se": (ep.get("delayed_response") or {}).get(
                    "gain_over_se"
                ),
                "evidence_level": ep.get("evidence_level"),
            })
    return summaries


def _call(payload: dict[str, Any], system: str) -> dict[str, Any]:
    api_key = next(
        (
            os.environ.get(name, "").strip()
            for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
            if os.environ.get(name, "").strip()
        ),
        None,
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    return _nf_call([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])


def _proposal(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "target_context": PRECHECK_TARGET_CONTEXT,
        "source_experiences": summaries,
        "allowed_programs": list(NATURAL_POOL),
    }
    system = (
        "You have no actual Target probe yet. Return an ordered candidate "
        "list of one to three Workflows to probe, using only allowed_programs. "
        "Return JSON: {'program_order': ['outlier_mad', 'hampel_filter', "
        "'winsorize'], 'reason': '...'}. Source experiences are directions "
        "with uncertainty; low gain_over_se is weak evidence."
    )
    response = _call(payload, system)
    order = response.get("program_order")
    if not isinstance(order, list) or not 1 <= len(order) <= 3:
        raise RuntimeError(f"invalid program_order: {order!r}")
    if any(op not in NATURAL_POOL for op in order) or len(set(order)) != len(order):
        raise RuntimeError(f"illegal/duplicate program_order: {order!r}")
    return {
        "program_order": [str(x) for x in order],
        "reason": response.get("reason"),
        "raw": response,
    }


def _decision(
    summaries: list[dict[str, Any]],
    first_workflow: str,
) -> dict[str, Any]:
    payload = {
        "target_context": PRECHECK_TARGET_CONTEXT,
        "source_experiences": summaries,
        "hypothetical_first_probe": {
            **HYPOTHETICAL_FIRST_PROBE,
            "program": first_workflow,
        },
    }
    system = (
        "This is a hypothetical weak positive first probe; no real Target "
        "outcome has been opened. Decide TRUST_DRAFT, CONTINUE, ABSTAIN or "
        "REQUEST_OBSERVATION. Return JSON: {'decision': 'ABSTAIN', "
        "'reason': '...'}."
    )
    response = _call(payload, system)
    decision = response.get("decision")
    if decision not in {"TRUST_DRAFT", "CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"}:
        raise RuntimeError(f"invalid decision: {decision!r}")
    return {
        "decision": decision,
        "reason": response.get("reason"),
        "raw": response,
    }


def run_natural_precheck(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    summaries = _natural_trajectory_summaries(report)
    a3_proposal = _proposal([])
    a5_proposal = _proposal(summaries)
    a3_decision = _decision([], a3_proposal["program_order"][0])
    a5_decision = _decision(summaries, a5_proposal["program_order"][0])

    same_order = a3_proposal["program_order"] == a5_proposal["program_order"]
    same_first = a3_proposal["program_order"][0] == a5_proposal["program_order"][0]
    same_decision = a3_decision["decision"] == a5_decision["decision"]
    inert = same_order and same_first and same_decision
    verdict = (
        "NATURAL_MEMORY_DECISION_INERT"
        if inert
        else "NATURAL_MEMORY_CHANGES_DECISION_BEHAVIOR"
    )
    precheck = {
        "target_context": PRECHECK_TARGET_CONTEXT,
        "target_outcome_opened": False,
        "natural_trajectory_count": len(summaries),
        "trajectory_independence_note": (
            "12 probe trajectories belong to 4 Task Episodes and are highly "
            "overlapping; they are not 12 independent Source Episodes."
        ),
        "winner_reliability_note": (
            "natural_k1_04 winsorize became LOCAL_ACTIVE mechanically, but "
            "Support g/SE=1.19 and delayed g/SE=0.55; this is lifecycle "
            "closure, not a reliable natural Capability."
        ),
        "memory_scope": "TARGET_HISTORY_MEMORY",
        "memory_scope_note": (
            "Any reuse here is on the same K1 series/origin pool; it cannot "
            "be called Source-to-Target. Formal A5/A3 requires a "
            "non-overlapping Target cohort/domain."
        ),
        "a3": {
            "proposal": a3_proposal,
            "decision": a3_decision,
        },
        "a5": {
            "proposal": a5_proposal,
            "decision": a5_decision,
        },
        "comparison": {
            "same_ordered_candidates": same_order,
            "same_first_workflow": same_first,
            "same_decision": same_decision,
            "summary": (
                "排序不同" if not same_order else "排序相同"
            ) + (
                "、首选相同" if same_first else "、首选不同"
            ) + (
                "、弱证据决策不同" if not same_decision else "、弱证据决策相同"
            ),
        },
        "verdict": verdict,
        "llm_api_call_count": 4,
        "wall_seconds": time.perf_counter() - started,
    }
    report["phase"] = "natural_precheck"
    report["natural_precheck"] = precheck
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return precheck
