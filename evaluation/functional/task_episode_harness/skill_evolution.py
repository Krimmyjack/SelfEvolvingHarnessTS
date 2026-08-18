"""E0: natural Experience -> Slow ADD Skill -> Runtime-owned Program lifecycle.

Sole authority: docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md §4.

One A5-shaped Source-Evidence path only.  Target is the already-exposed
natural_k1_03; Source is natural_k1_01/02/04 bounded by SignedEpisodeRetriever.
No virgin cohort, no A3 arm, no E1/E2 behavior is introduced here.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.natural_flow import (
    NATURAL_EPISODES,
    NATURAL_POOL,
)
from evaluation.functional.task_episode_harness.normal_flow import (
    NF_BASE_URL,
    NF_MODEL,
    _FastAgentStub,
)
from evaluation.functional.task_episode_harness.public_context import (
    C0_MATCHING_TASK_ID,
    C0_NON_MATCHING_TASK_ID,
    PUBLIC_CONTEXT_PROJECTION_FEATURE,
    build_task_public_context,
    run_context_census,
)
from evaluation.functional.task_episode_harness.runner import (
    MATERIAL_THRESHOLD,
    REPORT_REL,
    _arm_metrics,
    _evaluate_origins,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import TASK_CONSUMER_KEY
from SelfEvolvingHarnessTS.contracts.harness import HarnessSnapshot
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_DELAYED,
    EVIDENCE_SUPPORT,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    STATUS_RESTRICTED,
    SignedEpisodeRetriever,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _parse_frozen_steps
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    CandidateCompilationError,
    CompiledWorkflow,
    build_public_operator_inventory,
    compile_workflow_proposal,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import (
    TTHAMethod,
    _applicability_is_wide,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (
    evaluate_applicability,
)

E0_TARGET_TASK_ID = C0_MATCHING_TASK_ID
E0_SOURCE_TASK_IDS = ("natural_k1_01", "natural_k1_02", "natural_k1_04")
E0_DOMAIN = "kdd2018-natural-development"
E0_CAUSE = "SKILL_LIBRARY_GAP"
E0_MAX_ATTEMPTS = 3
E0_STORE_REL = ".e0_skill_evolution_store"

_SKILL_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")

_E0_SYSTEM_PROMPT = (
    "You are the Slow Skill author for one already-exposed natural forecast "
    "Task Episode. Target Support and an independent delayed block decide "
    "whether your proposal is accepted; you never approve your own Skill. "
    "Use target_public_context as the only Context features. "
    "target_history_feedback contains the Target's already-exposed fixed-pool "
    "Support outcomes. source_evidence contains at most one most-similar "
    "POSITIVE, one NEGATIVE and one CONFLICT episode from other natural Task "
    "Episodes; missing relations are null and must not be invented. "
    "Propose one to four EXECUTABLE operators from operator_inventory; "
    "reusing a Source Workflow is legal when evidence justifies it, and "
    "novelty is not required. Bind dynamic parameters only through the "
    "declared public bindings. Never replay numeric parameters from "
    "source_evidence onto the Target. "
    "Return JSON only. "
    "ADD: {'decision':'ADD','skill_id':'target_local_skill_v1',"
    "'workflow':{'steps':[{'op':'canonical_operator','params':{},"
    "'bindings':{}}],'requested_observations':[],'fallback':'IDENTITY',"
    "'experience_use':['episode_id']},'scope_rationale':'...',"
    "'risk_rationale':'...'} "
    "ABSTAIN: {'decision':'ABSTAIN','reason':'...','experience_use':[]}."
)


class _E0Receipt:
    def __init__(self, gain: float | None, *, passed: bool = True) -> None:
        self.gain = gain
        self.verification = type("V", (), {"passed": passed})()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(nested) for nested in value]
    return value


def _target_spec() -> dict[str, Any]:
    for spec in NATURAL_EPISODES:
        if spec["task_episode_id"] == E0_TARGET_TASK_ID:
            return dict(spec)
    raise RuntimeError(f"missing E0 target task: {E0_TARGET_TASK_ID}")


def _safe_workflow_signature(steps: Sequence[tuple[str, Mapping[str, object]]]) -> str:
    names = [str(op) for op, _params in steps]
    signature = "e0_" + "_".join(names)
    if not _SKILL_ID_RE.fullmatch(signature):
        signature = "e0_" + "_".join(
            re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
            for name in names
        )
    if not _SKILL_ID_RE.fullmatch(signature):
        raise ValueError(f"workflow signature is not a canonical id: {signature!r}")
    return signature


def _plain_steps(steps: Sequence[tuple[str, Mapping[str, object]]]) -> list[dict[str, Any]]:
    return [{"op": op, "params": dict(params)} for op, params in steps]


def _steps_equal(
    left: Sequence[tuple[str, Mapping[str, object]]],
    right: Sequence[tuple[str, Mapping[str, object]]],
) -> bool:
    return tuple(
        (str(op), dict(params)) for op, params in left
    ) == tuple(
        (str(op), dict(params)) for op, params in right
    )


def _target_context(
    values: Mapping[str, Any],
    train_uids: Sequence[str],
) -> dict[str, Any]:
    spec = _target_spec()
    return build_task_public_context(
        values,
        train_uids,
        observation_cutoff=int(spec["support_origins"][0]),
    )


def _non_matching_context(
    values: Mapping[str, Any],
    train_uids: Sequence[str],
) -> dict[str, Any]:
    spec = next(
        spec for spec in NATURAL_EPISODES
        if spec["task_episode_id"] == C0_NON_MATCHING_TASK_ID
    )
    return build_task_public_context(
        values,
        train_uids,
        observation_cutoff=int(spec["support_origins"][0]),
    )


def _target_history_feedback(report: dict[str, Any]) -> list[dict[str, Any]]:
    natural = report.get("natural_flow") or {}
    target = next(
        (
            episode
            for episode in natural.get("episodes", [])
            if episode.get("task_episode_id") == E0_TARGET_TASK_ID
        ),
        None,
    )
    if target is None:
        raise RuntimeError("natural_flow report has no E0 target episode")
    rows = []
    for probe in target.get("probes", []):
        rows.append({
            "program": probe.get("program"),
            "support_gain": probe.get("support_gain"),
            "support_se_block": probe.get("support_se_block"),
            "support_gain_over_se": probe.get("support_gain_over_se"),
            "mechanical_gate": probe.get("mechanical_gate"),
            "agent_decision": (probe.get("agent_decision") or {}).get(
                "decision"
            ),
        })
    return rows


def _episode_from_report_row(row: Mapping[str, Any]) -> Any:
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
        ExperienceEpisode,
    )

    return ExperienceEpisode(
        episode_id=str(row["episode_id"]),
        schema_version=str(row["schema_version"]),
        task_consumer_key=str(row["task_consumer_key"]),
        domain_namespace=str(row["domain_namespace"]),
        context_summary=dict(row.get("context_summary") or {}),
        workflow_signature=str(row.get("workflow_signature") or ""),
        support_response=dict(row.get("support_response") or {}),
        delayed_response=dict(row.get("delayed_response") or {}),
        relation=str(row["relation"]),
        evidence_level=str(row["evidence_level"]),
        response_validity=str(row.get("response_validity") or "VALID"),
        local_status=str(row["local_status"]),
        pattern_view=str(row.get("pattern_view") or "default"),
        evidence_refs=tuple(row.get("evidence_refs") or ()),
    )


def _source_episodes(report: dict[str, Any]) -> tuple[Any, ...]:
    bank = report.get("natural_bank") or []
    episodes = []
    for row in bank:
        if not isinstance(row, Mapping):
            continue
        context = row.get("context_summary") or {}
        if not isinstance(context, Mapping):
            continue
        if str(context.get("task_episode_id")) == E0_TARGET_TASK_ID:
            continue
        try:
            episodes.append(_episode_from_report_row(row))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(episodes)


def _source_summary(episode: Any) -> dict[str, Any]:
    context = dict(episode.context_summary or {})
    return {
        "episode_id": episode.episode_id,
        "task_episode_id": context.get("task_episode_id"),
        "workflow": episode.workflow_signature,
        "relation": episode.relation,
        "evidence_level": episode.evidence_level,
        "local_status": episode.local_status,
        "support": dict(episode.support_response or {}),
        "delayed": dict(episode.delayed_response or {}),
    }


def _target_query_context(public_context: Mapping[str, Any]) -> dict[str, Any]:
    scope = frozenset(public_context["scope_series_uids"])
    return {
        "task_episode_id": E0_TARGET_TASK_ID,
        "cohort": {"training_series_count": 12, "evaluation_series_count": 8},
        "local_pattern": {
            "scope_observation_bin": public_context["scope_bin"],
            "task_projection_bin": public_context["task_signature"].get(
                PUBLIC_CONTEXT_PROJECTION_FEATURE
            ),
        },
        "program_geometry": {
            "scope": "training_series_subset",
            "program_steps": [],
            "training_series_count": len(scope),
            "training_series_uids": sorted(scope),
        },
    }


def _retrieve_source_bundle(
    report: dict[str, Any],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    retriever = SignedEpisodeRetriever(
        _source_episodes(report),
        task_consumer_key=TASK_CONSUMER_KEY,
        allowed_operators=NATURAL_POOL,
    )
    pack = retriever.retrieve(
        _target_query_context(public_context),
        E0_DOMAIN,
    )
    bundle = {
        "positive": _source_summary(pack.positive) if pack.positive else None,
        "negative": _source_summary(pack.negative) if pack.negative else None,
        "conflict": _source_summary(pack.conflict) if pack.conflict else None,
        "evidence_sufficient": bool(pack.evidence_sufficient),
        "retrieval_note": pack.retrieval_note,
    }
    bundle["non_empty"] = any(
        bundle[key] is not None for key in ("positive", "negative", "conflict")
    )
    return bundle


def _known_headroom(report: dict[str, Any]) -> dict[str, Any]:
    rows = _target_history_feedback(report)
    positive = [
        row for row in rows
        if row.get("mechanical_gate") == "PASS"
        and isinstance(row.get("support_gain"), (int, float))
        and math.isfinite(float(row["support_gain"]))
        and float(row["support_gain"]) >= MATERIAL_THRESHOLD
    ]
    return {
        "pass": bool(positive),
        "known_positive_candidates": [
            {
                "program": row["program"],
                "support_gain": float(row["support_gain"]),
            }
            for row in positive
        ],
    }


def _slow_payload(
    *,
    public_context: Mapping[str, Any],
    target_history_feedback: Sequence[Mapping[str, Any]],
    source_bundle: Mapping[str, Any],
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    scope = frozenset(public_context["scope_series_uids"])
    return {
        "target_task_episode_id": E0_TARGET_TASK_ID,
        "target_public_context": {
            "task_kind": public_context["task_kind"],
            "observation_cutoff": int(public_context["observation_cutoff"]),
            "task_signature": dict(public_context["task_signature"]),
            "scope_policy": {
                "feature": public_context["scope_feature"],
                "bin": public_context["scope_bin"],
                "selected_series_count": len(scope),
            },
            "scope_series_uids": sorted(scope),
            "representative_series_uid": public_context["representative_uid"],
            "representative_features": dict(
                public_context["representative_features"]
            ),
        },
        "target_history_feedback": [dict(row) for row in target_history_feedback],
        "source_evidence": {
            key: source_bundle.get(key)
            for key in ("positive", "negative", "conflict")
        },
        "operator_inventory": [dict(row) for row in inventory],
        "material_threshold": MATERIAL_THRESHOLD,
    }


def _repair_raw_string_newlines(text: str) -> str:
    """Repair one observed LLM JSON failure mode: raw newline/tab bytes
    inside a JSON string.  Escaped backslash sequences are preserved."""
    out: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
            elif char == "\\":
                out.append(char)
                escaped = True
            elif char == '"':
                out.append(char)
                in_string = False
            elif char == "\n":
                out.append("\\n")
            elif char == "\r":
                continue
            elif char == "\t":
                out.append("\\t")
            else:
                out.append(char)
        else:
            if char == '"':
                in_string = True
            out.append(char)
    return "".join(out)


def _parse_json_response(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"non-JSON LLM response: {text[:200]!r}")
    segment = text[start : end + 1]
    candidates = [segment, _repair_raw_string_newlines(segment)]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
    raise RuntimeError(
        f"invalid JSON LLM response: {last_error}; "
        f"raw={text[:500]!r}"
    )


def _e0_slow_call(payload: Mapping[str, Any]) -> dict[str, Any]:
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
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=NF_BASE_URL, timeout=180)
    completion = client.chat.completions.create(
        model=NF_MODEL,
        messages=[
            {"role": "system", "content": _E0_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
    )
    return _parse_json_response(str(completion.choices[0].message.content or ""))


def _compile_slow_add(
    response: Mapping[str, Any],
    inventory: Sequence[Mapping[str, object]],
    public_context: Mapping[str, Any],
    *,
    attempt_index: int,
) -> tuple[dict[str, Any], CompiledWorkflow]:
    workflow = response.get("workflow")
    if not isinstance(workflow, Mapping):
        raise CandidateCompilationError(
            "PROPOSAL_INVALID", "ADD workflow must be an object"
        )
    proposal = {
        "decision": "PROPOSE",
        "steps": workflow.get("steps"),
        "requested_observations": workflow.get("requested_observations", []),
        "fallback": workflow.get("fallback", "IDENTITY"),
        "experience_use": workflow.get("experience_use", []),
    }
    compiled = compile_workflow_proposal(
        proposal,
        inventory,
        public_context["representative_features"],
        generation=attempt_index,
    )
    return dict(proposal), compiled


def _probe_compiled(
    roster: list[dict[str, Any]],
    values: dict[str, Any],
    config: dict[str, Any],
    origins: tuple[int, ...],
    eval_uids: list[str],
    compiled: CompiledWorkflow,
    scope: frozenset[str],
) -> dict[str, Any]:
    identity_rows = _evaluate_origins(
        roster, values, None, config, origins, None
    )
    candidate_rows = _evaluate_origins(
        roster, values, compiled, config, origins, set(scope)
    )
    metrics = _arm_metrics(
        identity_rows, candidate_rows, origins, eval_uids
    )
    metrics["program_steps"] = _plain_steps(
        compiled.candidate.program.execution_steps()
    )
    return metrics


def _make_e0_episode(
    *,
    attempt_index: int,
    compiled: CompiledWorkflow,
    workflow_signature: str,
    scope: frozenset[str],
    probe: Mapping[str, Any],
    support_origins: tuple[int, ...],
    source_evidence_used: Sequence[str],
    public_context: Mapping[str, Any],
) -> Any:
    steps = compiled.candidate.program.execution_steps()
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    return build_episode(
        episode_id=f"e0_slow_add_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace=E0_DOMAIN,
        context_summary={
            "task_episode_id": E0_TARGET_TASK_ID,
            "attempt_index": attempt_index,
            "observations_used": [
                "task_kind",
                PUBLIC_CONTEXT_PROJECTION_FEATURE,
            ],
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {
                "scope_observation_bin": public_context["scope_bin"],
                "task_projection_bin": public_context["task_signature"].get(
                    PUBLIC_CONTEXT_PROJECTION_FEATURE
                ),
            },
            "program_geometry": {
                "scope": "training_series_subset",
                "program_steps": _plain_steps(steps),
            },
            "source_evidence_used": list(source_evidence_used),
        },
        workflow_signature=workflow_signature,
        support_response={
            "gain": gain,
            "se_block": float(probe["se_block"]),
            "gain_over_se": probe["gain_over_se"],
            "accepted": positive,
            "block_origins": list(support_origins),
        },
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=RELATION_POSITIVE if positive else RELATION_NEGATIVE,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT if positive else STATUS_EPISODE_ONLY,
        evidence_refs=["task_episode_harness_skill_evolution_e0"],
    )


def _update_e0_episode_delayed(
    episode: Any,
    delayed_probe: Mapping[str, Any],
    *,
    delayed_origins: tuple[int, ...],
) -> Any:
    support_gain = float(episode.support_response.get("gain") or 0.0)
    delayed_gain = float(delayed_probe["macro_gain"])
    support_positive = support_gain >= MATERIAL_THRESHOLD
    delayed_approved = delayed_gain >= -MATERIAL_THRESHOLD
    if support_positive and delayed_approved:
        status, relation = STATUS_LOCAL_ACTIVE, RELATION_POSITIVE
    elif support_positive and not delayed_approved:
        status, relation = STATUS_RESTRICTED, RELATION_CONFLICT
    else:
        status, relation = STATUS_EPISODE_ONLY, RELATION_NEGATIVE
    return dataclasses.replace(
        episode,
        delayed_response={
            "evaluated": True,
            "gain": delayed_gain,
            "se_block": float(delayed_probe["se_block"]),
            "gain_over_se": delayed_probe["gain_over_se"],
            "block_origins": list(delayed_origins),
        },
        evidence_level=EVIDENCE_DELAYED,
        local_status=status,
        relation=relation,
    )


def _skill_steps_from_snapshot(
    snapshot: HarnessSnapshot,
    skill_id: str,
) -> tuple[tuple[str, Mapping[str, object]], ...] | None:
    skill = next(
        (entry for entry in snapshot.skills if entry.skill_id == skill_id),
        None,
    )
    if skill is None:
        return None
    return _parse_frozen_steps(skill.body)


def _snapshot_sha(snapshot: HarnessSnapshot) -> str:
    return str(snapshot.harness_content_sha)


def _run_e0_attempt(
    *,
    attempt_index: int,
    repo_root: Path,
    report: dict[str, Any],
    values: dict[str, Any],
    roster: list[dict[str, Any]],
    mapped_roster: list[dict[str, Any]],
    config: dict[str, Any],
    eval_uids: list[str],
    train_uids: list[str],
    target_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    non_matching_context: Mapping[str, Any],
    source_bundle: Mapping[str, Any],
    inventory: Sequence[Mapping[str, object]],
    payload: Mapping[str, Any],
    llm_counter: list[int],
    set_active: bool,
) -> dict[str, Any]:
    support_origins = tuple(target_spec["support_origins"])
    delayed_origins = tuple(target_spec["delayed_origins"])
    scope = frozenset(public_context["scope_series_uids"])
    record: dict[str, Any] = {
        "attempt_index": attempt_index,
        "branch": "SLOW_ADD_ABSTAINED",
        "complete_path": False,
        "retryable_llm_failure": False,
        "slow_response": None,
        "compile_proposal": None,
        "compiled_steps": None,
        "support": None,
        "lifecycle": {
            "method_event": {"stage": "no_winner"},
            "delayed_event": {"stage": "no_winner"},
        },
        "episode": None,
        "active_snapshot_changed": None,
    }
    try:
        response = _e0_slow_call(payload)
    except RuntimeError as exc:
        llm_counter[0] += 1
        message = str(exc)
        retryable = "LLM response" in message
        record["slow_response"] = None
        record["compile_error"] = f"{type(exc).__name__}: {exc}"
        record["branch"] = (
            "SLOW_ADD_COMPILATION_FAILED"
            if retryable
            else "E0_MECHANICAL_ERROR"
        )
        record["retryable_llm_failure"] = retryable
        return record
    llm_counter[0] += 1
    record["slow_response"] = response
    if not isinstance(response, Mapping) or response.get("decision") != "ADD":
        record["branch"] = "SLOW_ADD_ABSTAINED"
        record["retryable_llm_failure"] = True
        record["abstain_reason"] = (
            response.get("reason") if isinstance(response, Mapping) else None
        )
        return record
    skill_id = response.get("skill_id")
    if not isinstance(skill_id, str) or not _SKILL_ID_RE.fullmatch(skill_id):
        record["branch"] = "SLOW_ADD_COMPILATION_FAILED"
        record["compile_error"] = "skill_id is not a canonical surface id"
        record["retryable_llm_failure"] = True
        return record
    try:
        proposal, compiled = _compile_slow_add(
            response, inventory, public_context, attempt_index=attempt_index
        )
    except (CandidateCompilationError, ValueError) as exc:
        record["branch"] = "SLOW_ADD_COMPILATION_FAILED"
        record["compile_error"] = f"{type(exc).__name__}: {exc}"
        record["retryable_llm_failure"] = True
        return record
    steps = compiled.candidate.program.execution_steps()
    workflow_signature = _safe_workflow_signature(steps)
    record["compile_proposal"] = proposal
    record["compiled_steps"] = _plain_steps(steps)
    try:
        support = _probe_compiled(
            mapped_roster,
            values,
            config,
            support_origins,
            eval_uids,
            compiled,
            scope,
        )
    except Exception as exc:  # noqa: BLE001
        record["branch"] = "E0_MECHANICAL_ERROR"
        record["compile_error"] = f"SUPPORT_INSTRUMENT_FAILED:{type(exc).__name__}"
        record["retryable_llm_failure"] = False
        return record
    record["support"] = support
    support_gain = float(support["macro_gain"])
    episode = _make_e0_episode(
        attempt_index=attempt_index,
        compiled=compiled,
        workflow_signature=workflow_signature,
        scope=scope,
        probe=support,
        support_origins=support_origins,
        source_evidence_used=[
            str(item)
            for item in (response.get("workflow", {}) or {}).get(
                "experience_use", []
            )
        ],
        public_context=public_context,
    )
    record["episode"] = episode.to_dict()

    baseline = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    baseline_sha = _snapshot_sha(baseline)
    store = SnapshotStore(repo_root / E0_STORE_REL)
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(_FastAgentStub(), baseline, experience_episodes=())
    method.append_experience_episode(episode)
    card = {
        "pattern_id": "e0-natural-experience-skill-add",
        "failure_family": "natural_readiness_observation",
        "observable_signature": dict(public_context["task_signature"]),
        "workflow": {
            "steps": _plain_steps(steps),
        },
    }
    method_event = method.handle_fast_winner(
        episode,
        steps,
        controller=controller,
        store=store,
        card=card,
        evaluator=lambda _s, _m: _E0Receipt(None),
        fast_features=dict(public_context["task_fast_features"]),
        support_gain=support_gain,
        confirmed_cause=E0_CAUSE,
    )
    record["lifecycle"]["method_event"] = method_event
    record["active_snapshot_changed"] = (
        _snapshot_sha(method._active_snapshot()) != baseline_sha
    )
    if method_event.get("stage") == "support_rejected":
        record["branch"] = "SLOW_ADD_SUPPORT_REJECTED"
        record["retryable_llm_failure"] = True
        return record
    if method_event.get("stage") != "pending":
        record["branch"] = "E0_MECHANICAL_ERROR"
        record["lifecycle_error"] = method_event.get("stage")
        record["retryable_llm_failure"] = False
        return record

    skill_id_final = f"fast_winner_{workflow_signature}"
    pending_receipt = method._pending_update["receipt"]
    candidate_snapshot = pending_receipt.candidate_snapshot.snapshot
    pending_steps = _skill_steps_from_snapshot(candidate_snapshot, skill_id_final)
    body_bound = pending_steps is not None and _steps_equal(
        pending_steps, steps
    )
    record["body_binding"] = {
        "skill_id": skill_id_final,
        "matches_compiled_steps": body_bound,
        "candidate_snapshot_sha": _snapshot_sha(candidate_snapshot),
    }
    if not body_bound:
        record["branch"] = "E0_MECHANICAL_ERROR"
        record["retryable_llm_failure"] = False
        return record

    delayed_holder: dict[str, Any] = {}

    def delayed_evaluator(_steps: Any, _mode: int) -> _E0Receipt:
        delayed_probe = _probe_compiled(
            mapped_roster,
            values,
            config,
            delayed_origins,
            eval_uids,
            compiled,
            scope,
        )
        delayed_holder["probe"] = delayed_probe
        return _E0Receipt(float(delayed_probe["macro_gain"]))

    delayed_event = method.handle_feedback_delayed(
        delayed_evaluator, episode_id=episode.episode_id
    )
    delayed_probe = delayed_holder.get("probe")
    record["delayed"] = delayed_probe
    record["lifecycle"]["delayed_event"] = delayed_event
    if isinstance(delayed_probe, Mapping):
        updated_episode = _update_e0_episode_delayed(
            episode,
            delayed_probe,
            delayed_origins=delayed_origins,
        )
        method.update_experience_episode(updated_episode)
        record["episode"] = updated_episode.to_dict()

    if delayed_event.get("stage") == "approved":
        active = method._active_snapshot()
        active_steps = _skill_steps_from_snapshot(active, skill_id_final)
        record["active_snapshot"] = {
            "harness_content_sha": _snapshot_sha(active),
            "skill_ids": [skill.skill_id for skill in active.skills],
        }
        if active_steps is None or not _steps_equal(active_steps, steps):
            record["branch"] = "E0_MECHANICAL_ERROR"
            record["retryable_llm_failure"] = False
            return record
        skill = next(
            entry for entry in active.skills if entry.skill_id == skill_id_final
        )
        applicability = skill.observable_applicability
        matching_match, _ = evaluate_applicability(
            applicability, public_context["task_fast_features"]
        )
        non_matching_match, _ = evaluate_applicability(
            applicability, non_matching_context["task_fast_features"]
        )
        guards = dict(skill.risk_guards or {})
        is_narrow = not _applicability_is_wide(applicability)
        retrieval = {
            "skill_id": skill_id_final,
            "observable_applicability": _plain(applicability),
            "risk_guards": guards,
            "matching_context_match": matching_match,
            "non_matching_context_match": non_matching_match,
            "scope_type": "narrow" if is_narrow else "wide",
        }
        if is_narrow and matching_match and not non_matching_match:
            retrieval["scope_label"] = "E0_NARROW_SCOPE_RETRIEVAL_PASS"
        elif not is_narrow and guards.get("requires_target_support") is True:
            retrieval["scope_label"] = "E0_WIDE_SCOPE_GUARDED_REUSE_PASS"
        else:
            retrieval["scope_label"] = "E0_SCOPE_RETRIEVAL_FAIL"
        record["retrieval"] = retrieval
        record["branch"] = "EXPERIENCE_TO_SKILL_ADD_MECHANISM_PASS"
        record["complete_path"] = True
        record["retryable_llm_failure"] = False
        record["active_snapshot_changed"] = True
        if set_active:
            store.set_active(active.runtime_bundle_sha)
        return record

    record["branch"] = "SLOW_ADD_DELAYED_RESTRICTED"
    record["retryable_llm_failure"] = False
    record["active_snapshot_changed"] = (
        _snapshot_sha(method._active_snapshot()) != baseline_sha
    )
    record["active_snapshot_after_delayed_rejection"] = {
        "harness_content_sha": _snapshot_sha(method._active_snapshot()),
        "skill_ids": [skill.skill_id for skill in method._active_snapshot().skills],
    }
    return record


def run_skill_evolution_e0(
    report_path: Path = REPORT_REL,
    *,
    set_active: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    cohort = _load_cohort(repo_root)
    roster = cohort["roster"]
    values = {
        uid: np.asarray(value, dtype=np.float64).copy()
        for uid, value in cohort["values"].items()
    }
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]

    target_spec = _target_spec()
    public_context = _target_context(values, train_uids)
    non_matching_context = _non_matching_context(values, train_uids)
    context_census = run_context_census(
        {
            E0_TARGET_TASK_ID: public_context,
            C0_NON_MATCHING_TASK_ID: non_matching_context,
        }
    )
    if context_census["verdict"] != "TASK_CONTEXT_INLET_BINDING_PASS":
        return {
            "verdict": context_census["verdict"],
            "llm_api_call_count": 0,
        }

    headroom = _known_headroom(report)
    if not headroom["pass"]:
        result = {
            "verdict": "E0_NO_KNOWN_PROGRAM_HEADROOM",
            "headroom": headroom,
            "llm_api_call_count": 0,
        }
        report["historical_verdict_before_e0"] = report.get("verdict")
        report["phase"] = "skill_evolution_e0"
        report["skill_evolution_e0"] = result
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    source_bundle = _retrieve_source_bundle(report, public_context)
    target_history = _target_history_feedback(report)
    inventory = build_public_operator_inventory(
        public_context["task_kind"],
        public_context["representative_features"],
    )
    payload = _slow_payload(
        public_context=public_context,
        target_history_feedback=target_history,
        source_bundle=source_bundle,
        inventory=inventory,
    )
    llm_counter = [0]
    attempts = []
    for attempt_index in range(1, E0_MAX_ATTEMPTS + 1):
        try:
            attempt = _run_e0_attempt(
                attempt_index=attempt_index,
                repo_root=repo_root,
                report=report,
                values=values,
                roster=roster,
                mapped_roster=mapped_roster,
                config=config,
                eval_uids=eval_uids,
                train_uids=train_uids,
                target_spec=target_spec,
                public_context=public_context,
                non_matching_context=non_matching_context,
                source_bundle=source_bundle,
                inventory=inventory,
                payload=payload,
                llm_counter=llm_counter,
                set_active=set_active,
            )
        except Exception as exc:  # noqa: BLE001
            attempt = {
                "attempt_index": attempt_index,
                "branch": "E0_MECHANICAL_ERROR",
                "complete_path": False,
                "retryable_llm_failure": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        attempts.append(attempt)
        if attempt["complete_path"] or not attempt["retryable_llm_failure"]:
            break

    complete_paths = [attempt for attempt in attempts if attempt["complete_path"]]
    first_verdict = attempts[0]["branch"]
    if complete_paths:
        verdict = "EXPERIENCE_TO_SKILL_ADD_MECHANISM_PASS"
    else:
        verdict = attempts[-1]["branch"]

    result = {
        "protocol_version": "skill_evolution_e0_dev_v1",
        "question": (
            "Can existing natural Experience be distilled by the Slow Agent "
            "into a real Target-local Skill Card that survives Support, "
            "delayed feedback and changes next-Context retrieval?"
        ),
        "slow_input_audit": {
            "target_episode_included": True,
            "target_task_episode_id": E0_TARGET_TASK_ID,
            "target_history_feedback_entries": [
                row.get("program") for row in target_history
            ],
            "source_bundle_non_empty": bool(source_bundle.get("non_empty")),
            "source_evidence": {
                relation: (
                    source_bundle[relation].get("episode_id")
                    if source_bundle.get(relation)
                    else None
                )
                for relation in ("positive", "negative", "conflict")
            },
            "operator_inventory_entries": len(inventory),
        },
        "verdict": verdict,
        "first_attempt_verdict": first_verdict,
        "scope_sub_label": (
            attempts[-1].get("retrieval", {}).get("scope_label")
            if attempts else None
        ),
        "complete_path_count": len(complete_paths),
        "attempts_count": len(attempts),
        "repeatable_dev": (
            "SLOW_ADD_REPEATABLE_DEV"
            if len(complete_paths) >= 2
            else None
        ),
        "target": {
            "task_episode_id": E0_TARGET_TASK_ID,
            "support_origins": list(target_spec["support_origins"]),
            "delayed_origins": list(target_spec["delayed_origins"]),
            "scope": sorted(public_context["scope_series_uids"]),
            "task_signature": dict(public_context["task_signature"]),
        },
        "source": {
            "task_episode_ids": list(E0_SOURCE_TASK_IDS),
            "bundle": source_bundle,
        },
        "headroom_assertion": headroom,
        "context_census": context_census,
        "attempts": attempts,
        "llm_api_call_count": llm_counter[0],
        "wall_seconds": time.perf_counter() - started,
        "boundary": {
            "e0_only": True,
            "e1_not_started": True,
            "e2_not_started": True,
            "virgin_data_opened": False,
        },
    }
    report["historical_verdict_before_e0"] = report.get("verdict")
    report["phase"] = "skill_evolution_e0"
    report["skill_evolution_e0"] = result
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result
