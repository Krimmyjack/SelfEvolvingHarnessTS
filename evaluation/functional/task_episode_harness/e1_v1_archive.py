"""E1: paired A5/A3 development pilot on a non-overlapping KDD Target cohort.

Frozen authority: docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md §5.
One development dataset, one sealed dataset, N0=12 paired Task Episodes, and a
pre-frozen extension roster up to N=30.  A5 reads the E0 Source Card plus a
bounded Source contrast bundle; A3 reads none.  Both arms share the same Target
history ledger so the only A5/A3 input difference is the Source prior block.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_k1  # noqa: F401
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.normal_flow import (
    NF_BASE_URL,
    NF_MODEL,
    _FastAgentStub,
)
from evaluation.functional.task_episode_harness.public_context import (
    PUBLIC_CONTEXT_PROJECTION_FEATURE,
    build_task_public_context,
)
from evaluation.functional.task_episode_harness.runner import (
    MATERIAL_THRESHOLD,
    REPORT_REL,
    _arm_metrics,
    _evaluate_origins,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.skill_evolution import (
    _parse_json_response,
    _plain_steps,
    _probe_compiled,
    _safe_workflow_signature,
)
from evaluation.functional.task_episode_harness.t1 import TASK_CONSUMER_KEY
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
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    CandidateCompilationError,
    CompiledWorkflow,
    build_public_operator_inventory,
    compile_workflow_proposal,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

DEVELOPMENT_DATASET = "kdd2018_frozen_cohort_e31"
SEALED_CONFIRMATION_DATASET = "noaa_global_hourly"
CALIBRATION_DATASET = "kdd2018_frozen_cohort_e1"
E1_DOMAIN = "kdd2018-e31-development"
E1_CAUSE = "SKILL_LIBRARY_GAP"
B = 3
N0 = 12
MAX_N = 30
E1_STORE_A3 = ".e1_a3_store"
E1_STORE_A5 = ".e1_a5_store"

K1_SERIES = {
    "T117", "T118", "T119", "T12", "T120", "T121", "T122", "T123",
    "T124", "T125", "T126", "T127", "T128", "T129", "T13", "T130",
    "T131", "T132", "T133", "T134",
}

_CALIBRATION_ORIGINS = (1104, 1128, 1152)


# Two infrastructure-interrupted pre-runs opened a small number of formal
# cells without writing a report.  The official roster starts ten task-blocks
# later so every official outcome cell is fresh and unique.
_ORIGIN_SHIFT_TASKS = 10


def _task_spec(index: int) -> dict[str, Any]:
    base = 1104 + _ORIGIN_SHIFT_TASKS * 48 + index * 48
    return {
        "task_episode_id": f"e1_task_{index + 1:02d}",
        "arm_order": "A3_A5" if index % 2 == 0 else "A5_A3",
        "support_origins": (base, base + 6, base + 12),
        "delayed_origins": (base + 18, base + 24, base + 30),
    }


def _frozen_task_roster(n: int = MAX_N) -> tuple[dict[str, Any], ...]:
    return tuple(_task_spec(index) for index in range(n))


def _load_kdd_roster(
    repo_root: Path,
    cohort_rel: str,
    *,
    train_count: int = 12,
    eval_count: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows = [
        json.loads(line)
        for line in (repo_root / cohort_rel).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [str(row["series_name"]) for row in rows][: train_count + eval_count]
    cache = np.load(repo_root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(name) for name in cache["names"]]
    values = {
        uid: np.asarray(cache["values"][names.index(uid)], dtype=np.float64)
        for uid in selected
    }
    roster = [
        {"series_uid": uid, "role": "train"} for uid in selected[:train_count]
    ] + [
        {"series_uid": uid, "role": "eval"} for uid in selected[train_count:]
    ]
    return roster, values, selected


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


def _source_card_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    e0 = report.get("skill_evolution_e0") or {}
    attempts = e0.get("attempts") or []
    for attempt in reversed(attempts):
        retrieval = attempt.get("retrieval") or {}
        if not retrieval.get("skill_id"):
            continue
        if attempt.get("compiled_steps"):
            return {
                "skill_id": retrieval["skill_id"],
                "workflow_steps": list(attempt["compiled_steps"]),
                "observable_applicability": retrieval.get(
                    "observable_applicability"
                ),
                "risk_guards": retrieval.get("risk_guards") or {},
                "local_status": "LOCAL_ACTIVE",
                "evidence_ref": "skill_evolution_e0",
            }
    return {}


def _source_bundle_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    e0 = report.get("skill_evolution_e0") or {}
    bundle = (e0.get("source") or {}).get("bundle") or {}
    return {
        "positive": bundle.get("positive"),
        "negative": bundle.get("negative"),
        "conflict": bundle.get("conflict"),
        "non_empty": any(
            bundle.get(key) is not None
            for key in ("positive", "negative", "conflict")
        ),
    }


def _inventory_rows(public_context: Mapping[str, Any]) -> tuple[dict[str, object], ...]:
    return build_public_operator_inventory(
        public_context["task_kind"],
        public_context["representative_features"],
    )


def _proposal_params(row: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    declared = dict(row.get("public_parameter_bindings") or {})
    for parameter in row.get("runtime_parameters") or []:
        name = parameter.get("name")
        if not isinstance(name, str) or name in declared:
            continue
        if name == "period":
            params[name] = 24
        elif parameter.get("default") is not None:
            params[name] = parameter["default"]
    return params


def _single_step_proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    step: dict[str, Any] = {
        "op": row["name"],
        "params": _proposal_params(row),
    }
    declared = dict(row.get("public_parameter_bindings") or {})
    if declared:
        step["bindings"] = dict(declared)
    return {
        "decision": "PROPOSE",
        "steps": [step],
        "requested_observations": [],
        "fallback": "IDENTITY",
        "experience_use": [],
    }


class _Receipt:
    def __init__(self, gain: float | None, *, passed: bool = True) -> None:
        self.gain = gain
        self.verification = type("V", (), {"passed": passed})()


def _evaluate_reachability(
    card: Mapping[str, Any],
    fast_features: Mapping[str, Any],
) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.method import (
        _applicability_from_card,
        _applicability_reachable,
    )

    applicability = _applicability_from_card(card)
    reachable, reason = _applicability_reachable(
        card, applicability, fast_features
    )
    return {
        "applicability": applicability,
        "reachable": reachable,
        "reason": reason,
    }


def _e1_slow_call(messages: list[dict[str, str]]) -> dict[str, Any]:
    import os

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
        messages=messages,
    )
    return _parse_json_response(str(completion.choices[0].message.content or ""))


_E1_PROPOSAL_SYSTEM = (
    "You are the Slow proposal stage for one paired A5/A3 Target Task Episode. "
    "Return an ordered list of one to three Workflow proposals to probe. "
    "Each proposal has one to four EXECUTABLE operators from operator_inventory. "
    "Bind dynamic parameters only through declared public bindings; never "
    "replay numeric parameters from source_prior or target_experiences. "
    "Reusing a Source Workflow is legal when the evidence justifies it; "
    "novelty is not required. You do not approve proposals. "
    "Return JSON only: "
    "{'decision':'PROPOSE','proposals':[{'steps':[{'op':'canonical_operator',"
    "'params':{},'bindings':{}}],'requested_observations':[],"
    "'fallback':'IDENTITY','experience_use':[]}],'reason':'...'} "
    "or {'decision':'ABSTAIN','reason':'...'}."
)


def _proposal_payload(
    *,
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    target_memories: Sequence[Mapping[str, Any]],
    source_prior: Mapping[str, Any] | None,
    inventory: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    scope = frozenset(public_context["scope_series_uids"])
    payload: dict[str, Any] = {
        "task": TASK_CONSUMER_KEY,
        "target_task_episode_id": task_spec["task_episode_id"],
        "target_public_context": {
            "task_kind": public_context["task_kind"],
            "observation_cutoff": int(public_context["observation_cutoff"]),
            "task_signature": dict(public_context["task_signature"]),
            "scope_policy": {
                "feature": public_context["scope_feature"],
                "bin": public_context["scope_bin"],
                "selected_series_count": len(scope),
            },
            "representative_series_uid": public_context["representative_uid"],
            "representative_features": dict(
                public_context["representative_features"]
            ),
        },
        "operator_inventory": [dict(row) for row in inventory],
        "probe_budget": B,
        "material_threshold": MATERIAL_THRESHOLD,
        "target_experiences": [dict(row) for row in target_memories],
    }
    if source_prior is None:
        payload["source_prior"] = None
    else:
        payload["source_prior"] = dict(source_prior)
    return payload


def _normalized_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    normalized = {
        key: value for key, value in payload.items()
        if key != "source_prior"
    }
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _initial_proposals(
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    target_memories: Sequence[Mapping[str, Any]],
    source_prior: Mapping[str, Any] | None,
    inventory: Sequence[Mapping[str, object]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _proposal_payload(
        task_spec=task_spec,
        public_context=public_context,
        target_memories=target_memories,
        source_prior=source_prior,
        inventory=inventory,
    )
    response = _e1_slow_call([
        {"role": "system", "content": _E1_PROPOSAL_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    proposals = response.get("proposals")
    if response.get("decision") == "ABSTAIN":
        return payload, {
            "decision": "ABSTAIN",
            "reason": response.get("reason"),
            "proposals": [],
            "raw": response,
        }
    if (
        response.get("decision") != "PROPOSE"
        or not isinstance(proposals, list)
        or not 1 <= len(proposals) <= B
    ):
        raise RuntimeError(f"invalid Slow proposal response: {response!r}")
    return payload, {
        "decision": "PROPOSE",
        "reason": response.get("reason"),
        "proposals": proposals[:B],
        "raw": response,
    }


def _compile_proposal(
    proposal: Mapping[str, Any],
    inventory: Sequence[Mapping[str, object]],
    public_context: Mapping[str, Any],
    *,
    generation: int,
) -> tuple[CompiledWorkflow, dict[str, Any]]:
    normalized = {
        "decision": "PROPOSE",
        "steps": proposal.get("steps"),
        "requested_observations": proposal.get("requested_observations", []),
        "fallback": proposal.get("fallback", "IDENTITY"),
        "experience_use": proposal.get("experience_use", []),
    }
    compiled = compile_workflow_proposal(
        normalized,
        inventory,
        public_context["representative_features"],
        generation=generation,
    )
    return compiled, normalized


def _make_episode(
    *,
    arm: str,
    task_episode_id: str,
    attempt_index: int,
    compiled: CompiledWorkflow,
    workflow_signature: str,
    scope: frozenset[str],
    probe: Mapping[str, Any],
    support_origins: tuple[int, ...],
    public_context: Mapping[str, Any],
) -> Any:
    steps = compiled.candidate.program.execution_steps()
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    return build_episode(
        episode_id=f"e1_{arm}_{task_episode_id}_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace=E1_DOMAIN,
        context_summary={
            "task_episode_id": task_episode_id,
            "arm": arm,
            "attempt_index": attempt_index,
            "observation_cutoff": int(public_context["observation_cutoff"]),
            "task_signature": dict(public_context["task_signature"]),
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
        evidence_refs=["task_episode_harness_e1"],
    )


def _update_delayed(
    episode: Any,
    delayed_probe: Mapping[str, Any],
    delayed_origins: tuple[int, ...],
) -> Any:
    support_gain = float(episode.support_response.get("gain") or 0.0)
    delayed_gain = float(delayed_probe["macro_gain"])
    support_positive = support_gain >= MATERIAL_THRESHOLD
    delayed_ok = delayed_gain >= -MATERIAL_THRESHOLD
    if support_positive and delayed_ok:
        status, relation = STATUS_LOCAL_ACTIVE, RELATION_POSITIVE
    elif support_positive and not delayed_ok:
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


def _memory_summary(episode: Any) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "workflow": episode.workflow_signature,
        "support_gain": (episode.support_response or {}).get("gain"),
        "support_se_block": (episode.support_response or {}).get("se_block"),
        "support_gain_over_se": (episode.support_response or {}).get(
            "gain_over_se"
        ),
        "delayed_gain": (episode.delayed_response or {}).get("gain"),
        "delayed_se_block": (episode.delayed_response or {}).get("se_block"),
        "delayed_gain_over_se": (episode.delayed_response or {}).get(
            "gain_over_se"
        ),
        "relation": episode.relation,
        "local_status": episode.local_status,
    }


def _sync_memory(memories: list[dict[str, Any]], episode: Any) -> None:
    summary = _memory_summary(episode)
    for index, memory in enumerate(memories):
        if memory.get("episode_id") == summary["episode_id"]:
            memories[index] = summary
            return
    memories.append(summary)


def _merge_target_memories(
    shared: list[dict[str, Any]],
    arm_memories: Sequence[Mapping[str, Any]],
) -> None:
    existing = {str(memory.get("episode_id")) for memory in shared}
    for memory in arm_memories:
        episode_id = str(memory.get("episode_id"))
        if episode_id not in existing:
            shared.append(dict(memory))
            existing.add(episode_id)


def _decision_payload(
    *,
    workflow: str,
    gain: float,
    se: float,
    gain_over_se: float | None,
    remaining: list[str],
    above_threshold: bool,
    target_memories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allowed = (
        ["TRUST_DRAFT", "CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
        if above_threshold
        else ["CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
    )
    return {
        "last_probe": {
            "workflow": workflow,
            "support_gain": gain,
            "support_se_block": se,
            "support_gain_over_se": gain_over_se,
        },
        "remaining_workflows": list(remaining),
        "material_threshold": MATERIAL_THRESHOLD,
        "allowed_decisions": allowed,
        "target_experiences": [dict(row) for row in target_memories],
    }


_DECISION_SYSTEM = (
    "You are deciding what to do after one real Target Support probe. "
    "Use gain, se_block and gain_over_se as evidence; direction labels are "
    "not confidence. TRUST_DRAFT passes the candidate to the mechanical Gate; "
    "CONTINUE probes the next remaining workflow; ABSTAIN stops with no winner; "
    "REQUEST_OBSERVATION stops and records an observation gap. "
    "Return JSON: {'decision': one of allowed_decisions, 'reason': '...'}."
)


def _agent_decision(
    *,
    workflow: str,
    gain: float,
    se: float,
    gain_over_se: float | None,
    remaining: list[str],
    above_threshold: bool,
    target_memories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _decision_payload(
        workflow=workflow,
        gain=gain,
        se=se,
        gain_over_se=gain_over_se,
        remaining=remaining,
        above_threshold=above_threshold,
        target_memories=target_memories,
    )
    response = _e1_slow_call([
        {"role": "system", "content": _DECISION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    decision = response.get("decision")
    if decision not in payload["allowed_decisions"]:
        raise RuntimeError(
            f"invalid decision {decision!r}; allowed={payload['allowed_decisions']}"
        )
    return {
        "decision": decision,
        "reason": response.get("reason"),
        "raw": response,
        "decision_input": payload,
    }


def _lifecycle(
    *,
    repo_root: Path,
    arm: str,
    winner: Any,
    compiled: CompiledWorkflow,
    workflow_signature: str,
    scope: frozenset[str],
    values: Mapping[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: Mapping[str, Any],
    eval_uids: list[str],
    delayed_origins: tuple[int, ...],
    public_context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any] | None]:
    baseline = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(repo_root / (E1_STORE_A5 if arm == "A5" else E1_STORE_A3))
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(_FastAgentStub(), baseline, experience_episodes=())
    method.append_experience_episode(winner)
    steps = compiled.candidate.program.execution_steps()
    card = {
        "pattern_id": "e1-paired-target-episode",
        "failure_family": "natural_readiness_observation",
        "observable_signature": dict(public_context["task_signature"]),
        "workflow": {"steps": _plain_steps(steps)},
    }
    method_event = method.handle_fast_winner(
        winner,
        steps,
        controller=controller,
        store=store,
        card=card,
        evaluator=lambda _s, _m: _Receipt(None),
        fast_features=dict(public_context["task_fast_features"]),
        support_gain=float(winner.support_response["gain"]),
        confirmed_cause=E1_CAUSE,
    )
    delayed_event: dict[str, Any] = {"stage": "no_pending"}
    delayed_probe: dict[str, Any] | None = None
    if method_event.get("stage") == "pending":
        holder: dict[str, Any] = {}

        def delayed_evaluator(_steps: Any, _mode: int) -> _Receipt:
            probe = _probe_compiled(
                mapped_roster,
                values,
                config,
                delayed_origins,
                eval_uids,
                compiled,
                scope,
            )
            holder["probe"] = probe
            return _Receipt(float(probe["macro_gain"]))

        delayed_event = method.handle_feedback_delayed(
            delayed_evaluator, episode_id=winner.episode_id
        )
        delayed_probe = holder.get("probe")
        if isinstance(delayed_probe, Mapping):
            winner = _update_delayed(winner, delayed_probe, delayed_origins)
            method.update_experience_episode(winner)
    return method_event, delayed_event, winner, delayed_probe


def _run_arm(
    *,
    repo_root: Path,
    arm: str,
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    source_prior: Mapping[str, Any] | None,
    target_memories: list[dict[str, Any]],
    inventory: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: Mapping[str, Any],
    eval_uids: list[str],
    llm_counter: list[int],
) -> dict[str, Any]:
    support_origins = tuple(task_spec["support_origins"])
    delayed_origins = tuple(task_spec["delayed_origins"])
    scope = frozenset(public_context["scope_series_uids"])
    try:
        payload, initial = _initial_proposals(
            task_spec,
            public_context,
            target_memories,
            source_prior,
            inventory,
        )
        llm_counter[0] += 1
    except RuntimeError as exc:
        payload = _proposal_payload(
            task_spec=task_spec,
            public_context=public_context,
            target_memories=target_memories,
            source_prior=source_prior,
            inventory=inventory,
        )
        initial = {
            "decision": "ABSTAIN",
            "reason": f"proposal protocol error: {exc}",
            "proposals": [],
            "raw": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    proposals = list(initial["proposals"])
    probes = []
    winner = None
    winner_compiled = None
    stop_reason = "NO_DRAFT_IN_BUDGET"
    if initial["decision"] == "ABSTAIN":
        stop_reason = "AGENT_ABSTAIN"
    compiled_proposals: list[tuple[int, CompiledWorkflow, str]] = []
    for attempt_index, proposal in enumerate(proposals):
        generation = (
            int(task_spec["task_episode_id"].split("_")[-1]) * 10
            + attempt_index
        )
        try:
            compiled, _normalized_proposal = _compile_proposal(
                proposal,
                inventory,
                public_context,
                generation=generation,
            )
            workflow = _safe_workflow_signature(
                compiled.candidate.program.execution_steps()
            )
        except (CandidateCompilationError, ValueError) as exc:
            record = {
                "attempt_index": attempt_index,
                "status": "COMPILATION_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            probes.append(record)
            continue
        compiled_proposals.append((attempt_index, compiled, workflow))
    for attempt_index, compiled, workflow in compiled_proposals:
        steps = compiled.candidate.program.execution_steps()
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
            record = {
                "attempt_index": attempt_index,
                "status": "INSTRUMENT_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            probes.append(record)
            continue
        episode = _make_episode(
            arm=arm,
            task_episode_id=task_spec["task_episode_id"],
            attempt_index=attempt_index,
            compiled=compiled,
            workflow_signature=workflow,
            scope=scope,
            probe=support,
            support_origins=support_origins,
            public_context=public_context,
        )
        gain = float(support["macro_gain"])
        se = float(support["se_block"])
        gse = support["gain_over_se"]
        remaining = [
            workflow_name
            for _index, _compiled, workflow_name in compiled_proposals
            if _index > attempt_index
        ]
        above = gain >= MATERIAL_THRESHOLD
        if above or remaining:
            try:
                decision = _agent_decision(
                    workflow=workflow,
                    gain=gain,
                    se=se,
                    gain_over_se=gse,
                    remaining=remaining,
                    above_threshold=above,
                    target_memories=target_memories,
                )
                llm_counter[0] += 1
            except RuntimeError as exc:
                decision = {
                    "decision": "ABSTAIN",
                    "reason": f"promotion protocol error: {exc}",
                    "raw": None,
                    "decision_input": None,
                }
        else:
            decision = {
                "decision": "ABSTAIN",
                "reason": "budget exhausted without an acceptable candidate",
                "raw": None,
                "decision_input": None,
            }
        record = {
            "attempt_index": attempt_index,
            "workflow": workflow,
            "compiled_steps": _plain_steps(steps),
            "support_gain": gain,
            "support_se_block": se,
            "support_gain_over_se": gse,
            "agent_decision": decision,
            "mechanical_gate": "PASS" if above else "REJECT",
            "episode": episode.to_dict(),
        }
        probes.append(record)
        _sync_memory(target_memories, episode)
        action = decision["decision"]
        if action == "TRUST_DRAFT" and above:
            winner = episode
            winner_compiled = compiled
            stop_reason = "TRUST_DRAFT_GATE_PASS"
            break
        if action == "TRUST_DRAFT" and not above:
            record["mechanical_gate"] = "REJECT_TRUST_BELOW_THRESHOLD"
            if remaining:
                continue
            stop_reason = "NO_DRAFT_IN_BUDGET"
            break
        if action == "CONTINUE":
            if remaining:
                continue
            stop_reason = "NO_DRAFT_IN_BUDGET"
            break
        if action == "ABSTAIN":
            stop_reason = "AGENT_ABSTAIN"
            break
        if action == "REQUEST_OBSERVATION":
            stop_reason = "REQUEST_OBSERVATION"
            break

    lifecycle = {"method_event": {"stage": "no_winner"},
                 "delayed_event": {"stage": "no_winner"}}
    delayed_probe = None
    if winner is not None and winner_compiled is not None:
        method_event, delayed_event, updated, delayed_probe = _lifecycle(
            repo_root=repo_root,
            arm=arm,
            winner=winner,
            compiled=winner_compiled,
            workflow_signature=winner.workflow_signature,
            scope=scope,
            values=values,
            mapped_roster=mapped_roster,
            config=config,
            eval_uids=eval_uids,
            delayed_origins=delayed_origins,
            public_context=public_context,
        )
        lifecycle = {"method_event": method_event,
                     "delayed_event": delayed_event}
        for probe in probes:
            if probe.get("episode", {}).get("episode_id") == winner.episode_id:
                probe["episode"] = updated.to_dict()
        winner = updated
        _sync_memory(target_memories, winner)

    valid_probes = [
        probe for probe in probes
        if isinstance(probe.get("support_gain"), (int, float))
    ]
    actual_probe_count = len(valid_probes)
    local_active = bool(
        winner is not None and winner.local_status == STATUS_LOCAL_ACTIVE
    )
    task_probe_cost = actual_probe_count if local_active else B + 1
    return {
        "arm": arm,
        "payload": payload,
        "initial": initial,
        "probes": probes,
        "stop_reason": stop_reason,
        "winner": (
            {
                "episode_id": winner.episode_id,
                "workflow": winner.workflow_signature,
                "local_status": winner.local_status,
                "delayed_gain": winner.delayed_response.get("gain"),
                "delayed_se_block": winner.delayed_response.get("se_block"),
                "delayed_gain_over_se": winner.delayed_response.get("gain_over_se"),
            }
            if winner is not None else None
        ),
        "delayed": delayed_probe,
        "lifecycle": lifecycle,
        "target_memories_after": [dict(row) for row in target_memories],
        "metrics": {
            "task_probe_cost": task_probe_cost,
            "harmful_probe_count": sum(
                1 for probe in valid_probes
                if probe["support_gain"] < -MATERIAL_THRESHOLD
            ),
            "cumulative_support_harm": float(sum(
                -probe["support_gain"]
                for probe in valid_probes
                if probe["support_gain"] < -MATERIAL_THRESHOLD
            )),
            "task_local_active": int(local_active),
            "task_delayed_utility": (
                winner.delayed_response.get("gain")
                if winner is not None and winner.delayed_response.get("evaluated")
                else None
            ),
            "abstention": int(stop_reason in {"AGENT_ABSTAIN", "REQUEST_OBSERVATION"}),
        },
    }


def _calibration_headroom(
    *,
    repo_root: Path,
    calibration_roster: list[dict[str, Any]],
    calibration_values: Mapping[str, Any],
    train_uids: list[str],
    inventory: Sequence[Mapping[str, object]],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(_config())
    config["support_origin"] = _CALIBRATION_ORIGINS[0]
    mapped = _mapped_roster(calibration_roster)
    eval_uids = [row["series_uid"] for row in mapped if row["role"] == "eval"]
    scope = frozenset(public_context["scope_series_uids"])
    evaluated = []
    first_positive = None
    for index, row in enumerate(inventory):
        if row.get("availability") != "EXECUTABLE":
            continue
        try:
            compiled = compile_workflow_proposal(
                _single_step_proposal(row),
                inventory,
                public_context["representative_features"],
                generation=index + 1,
            )
            identity_rows = _evaluate_origins(
                mapped, calibration_values, None, config,
                _CALIBRATION_ORIGINS, None,
            )
            candidate_rows = _evaluate_origins(
                mapped, calibration_values, compiled, config,
                _CALIBRATION_ORIGINS, set(scope),
            )
            metrics = _arm_metrics(
                identity_rows, candidate_rows, _CALIBRATION_ORIGINS, eval_uids
            )
        except Exception as exc:  # noqa: BLE001
            evaluated.append({
                "operator": row["name"],
                "status": "INSTRUMENT_INVALID",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        record = {
            "operator": row["name"],
            "status": "EVALUATED",
            "macro_gain": metrics["macro_gain"],
            "se_block": metrics["se_block"],
            "gain_over_se": metrics["gain_over_se"],
            "positive_series_count": metrics["positive_series_count"],
            "negative_series_count": metrics["negative_series_count"],
        }
        evaluated.append(record)
        if first_positive is None and metrics["macro_gain"] >= MATERIAL_THRESHOLD:
            first_positive = record
            break
    return {
        "calibration_dataset": CALIBRATION_DATASET,
        "calibration_origins": list(_CALIBRATION_ORIGINS),
        "inventory_order": "canonical operator registry order",
        "single_step_only": True,
        "combination_search": False,
        "first_positive": first_positive,
        "pass": first_positive is not None,
        "evaluated": evaluated,
        "private_audit_only": True,
    }


def _run_preflight(
    repo_root: Path,
    *,
    target_roster: list[dict[str, Any]],
    target_values: Mapping[str, Any],
    target_train_uids: list[str],
    task_roster: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = sorted({row["series_uid"] for row in target_roster})
    overlap_with_k1 = sorted(set(selected) & K1_SERIES)
    target_contexts = {
        spec["task_episode_id"]: build_task_public_context(
            target_values,
            target_train_uids,
            observation_cutoff=int(spec["support_origins"][0]),
        )
        for spec in task_roster
    }
    signatures = [
        dict(context["task_signature"])
        for context in target_contexts.values()
    ]
    distinct = []
    for signature in signatures:
        if signature not in distinct:
            distinct.append(signature)
    context_pass = len(distinct) >= 2

    cal_roster, cal_values, cal_selected = _load_kdd_roster(
        repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e1.jsonl"
    )
    cal_train = [row["series_uid"] for row in cal_roster if row["role"] == "train"]
    cal_context = build_task_public_context(
        cal_values, cal_train, _CALIBRATION_ORIGINS[0]
    )
    inventory = _inventory_rows(cal_context)
    headroom = _calibration_headroom(
        repo_root=repo_root,
        calibration_roster=cal_roster,
        calibration_values=cal_values,
        train_uids=cal_train,
        inventory=inventory,
        public_context=cal_context,
    )
    checks = {
        "development_dataset": DEVELOPMENT_DATASET,
        "sealed_confirmation_dataset": SEALED_CONFIRMATION_DATASET,
        "sealed_dataset_read": False,
        "target_base_series_overlap_with_k1_source": overlap_with_k1,
        "target_base_series_non_overlap": not overlap_with_k1,
        "paired_task_count": len(task_roster),
        "paired_task_count_at_least_12": len(task_roster) >= N0,
        "calibration_slice_isolated": {
            "calibration_dataset": CALIBRATION_DATASET,
            "calibration_series_disjoint_from_target": bool(
                set(cal_selected).isdisjoint(selected)
            ),
            "calibration_origin_blocks_disjoint_from_target": bool(
                max(_CALIBRATION_ORIGINS)
                < min(int(spec["support_origins"][0]) for spec in task_roster)
            ),
        },
        "context_census": {
            "task_count": len(signatures),
            "distinct_signature_count": len(distinct),
            "distinct_signatures": distinct,
            "pass": context_pass,
        },
        "calibration_headroom": headroom,
        "frozen": {
            "B": B,
            "N0": N0,
            "max_N": MAX_N,
            "llm_model": NF_MODEL,
            "llm_base_url": NF_BASE_URL,
            "arm_order_rule": "A3_A5 on even task index, A5_A3 on odd task index",
            "task_origin_rule": "base=1104+10*48+i*48 (shifted past two discarded partial-cell pre-runs); support=base,+6,+12; delayed=base+18,+24,+30",
            "candidate_pool": "full canonical operator inventory, 1-4 steps per proposal",
            "post_probe_promotion": "Slow decision, mechanical Gate",
        },
    }
    checks["preflight_pass"] = bool(
        checks["target_base_series_non_overlap"]
        and checks["paired_task_count_at_least_12"]
        and checks["calibration_slice_isolated"]["calibration_series_disjoint_from_target"]
        and checks["calibration_slice_isolated"]["calibration_origin_blocks_disjoint_from_target"]
        and checks["context_census"]["pass"]
        and checks["calibration_headroom"]["pass"]
    )
    return checks


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def metric(arm: str, key: str) -> list[float]:
        return [float(row[arm]["metrics"][key]) for row in rows]

    a3_cost = metric("A3", "task_probe_cost")
    a5_cost = metric("A5", "task_probe_cost")
    probe_diff = [a5 - a3 for a3, a5 in zip(a3_cost, a5_cost)]
    n = len(probe_diff)
    probe_mean = float(np.mean(probe_diff)) if n else 0.0
    probe_sd = float(np.std(probe_diff, ddof=1)) if n > 1 else 0.0
    probe_se = probe_sd / math.sqrt(n) if n else 0.0

    def diff(arm_key: str) -> tuple[list[float], float]:
        a3 = metric("A3", arm_key)
        a5 = metric("A5", arm_key)
        values = [a5v - a3v for a3v, a5v in zip(a3, a5)]
        return values, float(np.mean(values)) if values else 0.0

    harm_count_diff, harm_count_mean = diff("harmful_probe_count")
    harm_sum_diff, harm_sum_mean = diff("cumulative_support_harm")
    a3_active = sum(row["A3"]["metrics"]["task_local_active"] for row in rows)
    a5_active = sum(row["A5"]["metrics"]["task_local_active"] for row in rows)
    drafts = []
    for row in rows:
        a3_winner = row["A3"].get("winner")
        a5_winner = row["A5"].get("winner")
        if a3_winner is not None and a5_winner is not None:
            drafts.append({
                "task_episode_id": row["task_episode_id"],
                "A3_delayed_utility": a3_winner.get("delayed_gain"),
                "A5_delayed_utility": a5_winner.get("delayed_gain"),
            })
    q = len(drafts) / n if n else 0.0
    paired_delayed_diff = [
        float(row["A5_delayed_utility"]) - float(row["A3_delayed_utility"])
        for row in drafts
        if isinstance(row.get("A3_delayed_utility"), (int, float))
        and isinstance(row.get("A5_delayed_utility"), (int, float))
    ]
    delayed_utility_mean = (
        float(np.mean(paired_delayed_diff)) if paired_delayed_diff else None
    )
    a3_draft_count = sum(
        1 for row in rows if row["A3"].get("winner") is not None
    )
    a5_draft_count = sum(
        1 for row in rows if row["A5"].get("winner") is not None
    )
    a3_survival = sum(
        1 for row in rows
        if row["A3"].get("winner") is not None
        and row["A3"]["winner"].get("local_status") == STATUS_LOCAL_ACTIVE
    )
    a5_survival = sum(
        1 for row in rows
        if row["A5"].get("winner") is not None
        and row["A5"]["winner"].get("local_status") == STATUS_LOCAL_ACTIVE
    )
    return {
        "n": n,
        "probe_diff": probe_diff,
        "probe_paired_mean": probe_mean,
        "probe_paired_sd": probe_sd,
        "probe_paired_se": probe_se,
        "probe_ci95_upper": probe_mean + 1.96 * probe_se if n else None,
        "harmful_probe_count_diff": harm_count_diff,
        "harmful_probe_count_paired_mean": harm_count_mean,
        "cumulative_support_harm_diff": harm_sum_diff,
        "cumulative_support_harm_paired_mean": harm_sum_mean,
        "a3_local_active_count": a3_active,
        "a5_local_active_count": a5_active,
        "paired_draft_count": len(drafts),
        "q": q,
        "paired_delayed_utility_diff": paired_delayed_diff,
        "paired_delayed_utility_mean": delayed_utility_mean,
        "a3_draft_count": a3_draft_count,
        "a5_draft_count": a5_draft_count,
        "a3_delayed_survival_rate": (
            a3_survival / a3_draft_count if a3_draft_count else None
        ),
        "a5_delayed_survival_rate": (
            a5_survival / a5_draft_count if a5_draft_count else None
        ),
    }


def _sample_plan(summary: Mapping[str, Any]) -> dict[str, Any]:
    n = summary["n"]
    s = summary["probe_paired_sd"]
    delta = 1.0
    n_req = math.ceil(7.84 * s * s / (delta * delta)) if s > 0 else 0
    q = summary["q"]
    n_draft_req = math.ceil(8 / q) if q and q > 0 else None
    values = [12, n_req] + ([n_draft_req] if n_draft_req is not None else [])
    n_final = max(values)
    return {
        "delta_probe": delta,
        "paired_probe_sd": s,
        "N_req": n_req,
        "q": q,
        "N_draft_req": n_draft_req,
        "N_final": n_final,
        "N_final_within_cap": n_final <= MAX_N,
        "extension_count": max(0, n_final - n),
        "note": (
            "sample formula is development capacity planning only; "
            "prequential Task pairs are not independent replicates"
        ),
    }


def _verdict(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_behavior_changed: bool,
) -> str:
    if plan["N_final"] > MAX_N:
        return "E1_PRACTICAL_RESOLUTION_INSUFFICIENT"
    paired_drafts_readable = summary["paired_draft_count"] >= 8
    if not source_behavior_changed:
        return "A5_A3_SKILL_INPUT_INERT"
    harm_count_ci_lower = summary["harmful_probe_count_paired_mean"] - (
        1.96 * (
            float(np.std(summary["harmful_probe_count_diff"], ddof=1))
            / math.sqrt(summary["n"])
        )
        if summary["n"] > 1 and summary["harmful_probe_count_diff"]
        else 0.0
    )
    harm_sum_ci_lower = summary["cumulative_support_harm_paired_mean"] - (
        1.96 * (
            float(np.std(summary["cumulative_support_harm_diff"], ddof=1))
            / math.sqrt(summary["n"])
        )
        if summary["n"] > 1 and summary["cumulative_support_harm_diff"]
        else 0.0
    )
    negative_transfer = bool(
        harm_count_ci_lower > 0
        or harm_sum_ci_lower > 0
        or (
            paired_drafts_readable
            and summary["paired_delayed_utility_mean"] is not None
            and summary["paired_delayed_utility_mean"] < 0
        )
    )
    support_efficiency = bool(
        summary["probe_paired_mean"] <= -1
        and summary["probe_ci95_upper"] is not None
        and summary["probe_ci95_upper"] < 0
        and summary["harmful_probe_count_paired_mean"] <= 0
        and summary["cumulative_support_harm_paired_mean"] <= 0
        and summary["a5_local_active_count"] >= summary["a3_local_active_count"]
    )
    if negative_transfer:
        return "A5_SKILL_CARD_NEGATIVE_TRANSFER_DEV"
    if not paired_drafts_readable:
        if support_efficiency:
            return "A5_SUPPORT_EFFICIENCY_DEV_SIGNAL / DELAYED_UNREADABLE"
        return "A5_SKILL_CARD_NO_BENEFIT_DEV"
    delayed_ok = bool(
        summary["paired_delayed_utility_mean"] is not None
        and summary["paired_delayed_utility_mean"] >= 0
        and (
            summary["a5_delayed_survival_rate"] is None
            or summary["a3_delayed_survival_rate"] is None
            or summary["a5_delayed_survival_rate"]
            >= summary["a3_delayed_survival_rate"]
        )
    )
    if support_efficiency and delayed_ok:
        return "A5_SKILL_CARD_WARM_START_DEV_SIGNAL"
    return "A5_SKILL_CARD_NO_BENEFIT_DEV"


def run_e1(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    source_card = _source_card_from_report(report)
    if not source_card:
        result = {
            "verdict": "E1_SOURCE_CARD_UNAVAILABLE",
            "llm_api_call_count": 0,
        }
        report["e1"] = result
        report["phase"] = "e1"
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result
    source_bundle = _source_bundle_from_report(report)

    target_roster, target_values, target_selected = _load_kdd_roster(
        repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e31.jsonl"
    )
    target_train_uids = [
        row["series_uid"] for row in target_roster if row["role"] == "train"
    ]
    target_eval_uids = [
        row["series_uid"] for row in target_roster if row["role"] == "eval"
    ]
    mapped_roster = _mapped_roster(target_roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    assert target_eval_uids == eval_uids
    config = dict(_config())
    task_roster = _frozen_task_roster(MAX_N)

    preflight = _run_preflight(
        repo_root,
        target_roster=target_roster,
        target_values=target_values,
        target_train_uids=target_train_uids,
        task_roster=task_roster,
    )
    if not preflight["preflight_pass"]:
        if preflight["context_census"]["pass"] is False:
            verdict = "E1_TARGET_CONTEXTS_INERT"
        elif preflight["calibration_headroom"]["pass"] is False:
            verdict = "E1_DEVELOPMENT_SUBSTRATE_NO_KNOWN_HEADROOM"
        elif not preflight["target_base_series_non_overlap"]:
            verdict = "E1_TARGET_SOURCE_OVERLAP"
        else:
            verdict = "E1_PREFLIGHT_FAILED"
        result = {
            "verdict": verdict,
            "preflight": preflight,
            "llm_api_call_count": 0,
            "boundary": {"e2_not_started": True, "sealed_confirmation_opened": False},
        }
        report["e1"] = result
        report["phase"] = "e1_preflight"
        report["verdict"] = verdict
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    # Freeze before first paired outcome.
    preregistration = {
        "development_dataset": DEVELOPMENT_DATASET,
        "sealed_confirmation_dataset": SEALED_CONFIRMATION_DATASET,
        "development_target_series": target_selected,
        "target_train_series": target_train_uids,
        "target_eval_series": target_eval_uids,
        "task_roster": list(task_roster),
        "B": B,
        "N0": N0,
        "max_N": MAX_N,
        "llm_settings": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "preflight": preflight,
        "source_card": source_card,
        "source_bundle": source_bundle,
    }

    llm_counter = [0]
    target_memories: list[dict[str, Any]] = []
    rows = []
    a3_source_prior = None
    a5_source_prior = {
        "source_card": source_card,
        "source_evidence": source_bundle,
    }
    inventory = None
    for task_index, spec in enumerate(task_roster[:N0]):
        print(f"E1_TASK_START {spec['task_episode_id']}", flush=True)
        public_context = build_task_public_context(
            target_values,
            target_train_uids,
            observation_cutoff=int(spec["support_origins"][0]),
        )
        inventory = _inventory_rows(public_context)
        arm_order = [("A3", a3_source_prior), ("A5", a5_source_prior)]
        if spec["arm_order"] == "A5_A3":
            arm_order = list(reversed(arm_order))
        shared_base_memories = [dict(row) for row in target_memories]
        arm_rows = {}
        for arm, source_prior in arm_order:
            arm_rows[arm] = _run_arm(
                repo_root=repo_root,
                arm=arm,
                task_spec=spec,
                public_context=public_context,
                source_prior=source_prior,
                target_memories=[dict(row) for row in shared_base_memories],
                inventory=inventory,
                values=target_values,
                mapped_roster=mapped_roster,
                config=config,
                eval_uids=eval_uids,
                llm_counter=llm_counter,
            )
        for arm in ("A3", "A5"):
            _merge_target_memories(
                target_memories, arm_rows[arm]["target_memories_after"]
            )
        # A3/A5 proposal payloads are identical except the Source prior block.
        non_source_identical = (
            _normalized_payload_fingerprint(arm_rows["A3"]["payload"])
            == _normalized_payload_fingerprint(arm_rows["A5"]["payload"])
        )
        rows.append({
            "task_episode_id": spec["task_episode_id"],
            "support_origins": list(spec["support_origins"]),
            "delayed_origins": list(spec["delayed_origins"]),
            "arm_order": spec["arm_order"],
            "public_context": public_context,
            "A3": arm_rows["A3"],
            "A5": arm_rows["A5"],
            "non_source_payload_identical": non_source_identical,
        })
        print(
            f"E1_TASK_DONE {spec['task_episode_id']} "
            f"A3={arm_rows['A3']['stop_reason']} "
            f"A5={arm_rows['A5']['stop_reason']}",
            flush=True,
        )

    summary = _paired_summary(rows)
    plan = _sample_plan(summary)
    if plan["N_final_within_cap"] and plan["extension_count"] > 0:
        for spec in task_roster[N0 : N0 + plan["extension_count"]]:
            print(f"E1_EXT_START {spec['task_episode_id']}", flush=True)
            public_context = build_task_public_context(
                target_values,
                target_train_uids,
                observation_cutoff=int(spec["support_origins"][0]),
            )
            inventory = _inventory_rows(public_context)
            arm_order = [("A3", a3_source_prior), ("A5", a5_source_prior)]
            if spec["arm_order"] == "A5_A3":
                arm_order = list(reversed(arm_order))
            shared_base_memories = [dict(row) for row in target_memories]
            arm_rows = {}
            for arm, source_prior in arm_order:
                arm_rows[arm] = _run_arm(
                    repo_root=repo_root,
                    arm=arm,
                    task_spec=spec,
                    public_context=public_context,
                    source_prior=source_prior,
                    target_memories=[dict(row) for row in shared_base_memories],
                    inventory=inventory,
                    values=target_values,
                    mapped_roster=mapped_roster,
                    config=config,
                    eval_uids=eval_uids,
                    llm_counter=llm_counter,
                )
            for arm in ("A3", "A5"):
                _merge_target_memories(
                    target_memories, arm_rows[arm]["target_memories_after"]
                )
            rows.append({
                "task_episode_id": spec["task_episode_id"],
                "support_origins": list(spec["support_origins"]),
                "delayed_origins": list(spec["delayed_origins"]),
                "arm_order": spec["arm_order"],
                "public_context": public_context,
                "A3": arm_rows["A3"],
                "A5": arm_rows["A5"],
                "non_source_payload_identical": (
                    _normalized_payload_fingerprint(arm_rows["A3"]["payload"])
                    == _normalized_payload_fingerprint(arm_rows["A5"]["payload"])
                ),
            })
        summary = _paired_summary(rows)
        plan = _sample_plan(summary)

    behavior_changed = any(
        row["A3"]["initial"].get("proposals") != row["A5"]["initial"].get("proposals")
        for row in rows
    )
    verdict = _verdict(rows, summary, plan, behavior_changed)

    result = {
        "protocol_version": "e1_skill_card_warm_start_dev_v1",
        "question": (
            "Can one Source-domain Skill Card plus bounded contrast Experience "
            "shorten Target cold-start without increasing Support harm?"
        ),
        "verdict": verdict,
        "preregistration": preregistration,
        "rows": rows,
        "summary": summary,
        "sample_plan": plan,
        "source_behavior_changed": behavior_changed,
        "delayed_comparison": (
            "readable" if summary["paired_draft_count"] >= 8
            else "DELAYED_COMPARISON_UNREADABLE"
        ),
        "llm_api_call_count": llm_counter[0],
        "wall_seconds": time.perf_counter() - started,
        "claim_scope": (
            "development paired pilot on one prequential Target trajectory; "
            "Task pairs are not independent replicates. No sealed confirmation "
            "was opened."
        ),
        "boundary": {
            "e1_only": True,
            "e2_not_started": True,
            "sealed_confirmation_opened": False,
        },
    }
    report["historical_verdict_before_e1"] = report.get("verdict")
    report["phase"] = "e1"
    report["e1"] = result
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result
