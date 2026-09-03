"""Workflow Generation A5/A3 development slice (pre-registered §17).

Question: can Source Experience help the LLM generate a more effective data
adaptation Workflow than the same LLM using only Target Context/feedback?

Target = natural_k1_03 (already exposed; fixed-pool candidates ended in
AGENT_ABSTAIN).  A3 and A5 each make exactly one catalog-free generation call,
compiled by the existing generative Workflow compiler.  Target Support decides
Draft by the existing material threshold; delayed feedback and Skill activation
stay inside TTHAMethod.  Source has no approval authority.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_natural_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.natural_flow import (
    NATURAL_EPISODES,
)
from evaluation.functional.task_episode_harness.natural_precheck import (
    _natural_trajectory_summaries,
)
from evaluation.functional.task_episode_harness.normal_flow import (
    _FastAgentStub,
    _nf_call,
)
from evaluation.functional.task_episode_harness.runner import (
    MATERIAL_THRESHOLD,
    REPORT_REL,
    _arm_metrics,
    _evaluate_origins,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import (
    T1_SCOPE_FEATURE,
    TASK_CONSUMER_KEY,
    _public_scope_proposal,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_DELAYED,
    EVIDENCE_SUPPORT,
    RELATION_ABSTAIN,
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
    CompiledWorkflow,
    build_public_operator_inventory,
    compile_workflow_proposal,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features,
)

WORKFLOW_GEN_TARGET_TASK = "natural_k1_03"
WORKFLOW_GEN_SOURCE_TASKS = (
    "natural_k1_01",
    "natural_k1_02",
    "natural_k1_04",
)
WORKFLOW_GEN_GENERATION = 1
WORKFLOW_GEN_SCOPE_CUTOFF = 1104
WORKFLOW_GEN_CAUSE = "SKILL_LIBRARY_GAP"

_GENERATION_SYSTEM_PROMPT = (
    "You are the Slow Workflow generation stage for one already-exposed "
    "natural forecast Task Episode. Propose exactly one Typed Workflow by "
    "composing one to four executable operators from operator_inventory, or "
    "return decision ABSTAIN if no legal, evidence-justified candidate "
    "exists. Do not propose identity. Use only operators whose availability "
    "is EXECUTABLE. Bind dynamic parameters only through the bindings declared "
    "in each operator contract, using binding_context feature values; never "
    "replay numerical parameters from source_experiences onto this Target. "
    "Source experiences are direction evidence only and may inform mechanism "
    "choice, but Target Support decides whether the Workflow becomes a Draft. "
    "Return JSON only. PROPOSE: "
    "{'decision':'PROPOSE','steps':[{'op':'canonical_operator','params':{},"
    "'bindings':{}}],'requested_observations':[],'fallback':'IDENTITY',"
    "'experience_use':['episode_id']}. ABSTAIN: {'decision':'ABSTAIN',"
    "'steps':[],'requested_observations':[],'fallback':'IDENTITY',"
    "'experience_use':[]}."
)


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _target_spec() -> dict[str, Any]:
    for spec in NATURAL_EPISODES:
        if spec["task_episode_id"] == WORKFLOW_GEN_TARGET_TASK:
            return dict(spec)
    raise RuntimeError(f"target task not in NATURAL_EPISODES: {WORKFLOW_GEN_TARGET_TASK}")


def _generated_workflow_signature(steps: list[tuple[str, Any]]) -> str:
    names = [str(op) for op, _params in steps]
    return "generated_" + "_".join(names) if names else "generated_unknown"


def _generated_probe(
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
    metrics["program_steps"] = [
        {"op": op, "params": dict(params)}
        for op, params in compiled.candidate.program.execution_steps()
    ]
    return metrics


def _public_context_for_generation(
    values: dict[str, Any],
    train_uids: list[str],
) -> dict[str, Any]:
    proposal = _public_scope_proposal(values, train_uids)
    scope = frozenset(proposal["scope"])
    observations = {
        str(uid): {
            key: value
            for key, value in dict(proposal["observations"][uid]).items()
            if key in {
                "local_robust_z_peak",
                "local_robust_z_peak_bin",
                "missing_fraction",
                "level_excursion_score",
            }
        }
        for uid in scope
    }
    representative_uid = max(
        scope,
        key=lambda uid: float(observations[uid]["local_robust_z_peak"]),
    )
    representative_features = dict(
        extract_public_features(
            np.asarray(values[representative_uid], dtype=np.float64)[
                :WORKFLOW_GEN_SCOPE_CUTOFF
            ],
            task_kind="forecast",
        )
    )
    return {
        "scope": scope,
        "observations": observations,
        "representative_uid": representative_uid,
        "representative_features": representative_features,
    }


def _target_fixed_feedback(report: dict[str, Any]) -> list[dict[str, Any]]:
    natural = report.get("natural_flow") or {}
    target = None
    for episode in natural.get("episodes", []):
        if episode.get("task_episode_id") == WORKFLOW_GEN_TARGET_TASK:
            target = episode
            break
    if target is None:
        raise RuntimeError(
            f"natural_flow report has no target episode {WORKFLOW_GEN_TARGET_TASK}"
        )
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


def _source_experiences(report: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = _natural_trajectory_summaries(report)
    return [
        summary
        for summary in summaries
        if summary.get("task_episode_id") in WORKFLOW_GEN_SOURCE_TASKS
    ]


def _generation_payload(
    *,
    target_spec: dict[str, Any],
    public_context: dict[str, Any],
    inventory: tuple[dict[str, object], ...],
    target_feedback: list[dict[str, Any]],
    source_experiences: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = sorted(
        (
            {"series_uid": uid, **features}
            for uid, features in public_context["observations"].items()
        ),
        key=lambda row: str(row["series_uid"]),
    )
    return {
        "task": TASK_CONSUMER_KEY,
        "target_task_episode_id": target_spec["task_episode_id"],
        "target_support_origins": list(target_spec["support_origins"]),
        "scope_policy": {
            "feature": T1_SCOPE_FEATURE,
            "bin": "high",
            "selected_series_count": len(public_context["scope"]),
        },
        "target_series_observations": observations,
        "binding_context": {
            "series_uid": public_context["representative_uid"],
            "features": public_context["representative_features"],
        },
        "operator_inventory": [dict(row) for row in inventory],
        "target_fixed_pool_feedback": target_feedback,
        "source_experiences": source_experiences,
    }


def _generation_call(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], CompiledWorkflow | None, str]:
    """One LLM generation call; compile via the existing Runtime compiler."""
    response = _nf_call([
        {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    if response.get("decision") == "ABSTAIN":
        return response, None, "GENERATION_ABSTAIN"
    try:
        compiled = compile_workflow_proposal(
            response,
            tuple(payload["operator_inventory"]),
            payload["binding_context"]["features"],
            generation=WORKFLOW_GEN_GENERATION,
        )
        return response, compiled, "COMPILED"
    except Exception as exc:  # noqa: BLE001
        return response, None, f"COMPILATION_FAILED:{type(exc).__name__}"


def _make_generated_episode(
    *,
    arm: str,
    task_episode_id: str,
    compiled: CompiledWorkflow,
    scope: frozenset[str],
    probe: dict[str, Any],
    support_origins: tuple[int, ...],
    experience_use: tuple[str, ...],
) -> Any:
    steps = compiled.candidate.program.execution_steps()
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    signature = _generated_workflow_signature(steps)
    return build_episode(
        episode_id=(
            f"workflow_gen_{arm}_{task_episode_id}_"
            f"generation_{WORKFLOW_GEN_GENERATION}"
        ),
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-natural-development",
        context_summary={
            "task_episode_id": task_episode_id,
            "arm": arm,
            "generation_index": WORKFLOW_GEN_GENERATION,
            "observations_used": [T1_SCOPE_FEATURE],
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {"scope_observation_bin": "high"},
            "program_geometry": {
                "scope": "training_series_subset",
                "program_steps": [
                    {"op": op, "params": dict(params)}
                    for op, params in steps
                ],
            },
            "generation": {
                "experience_use": list(experience_use),
                "candidate_id": compiled.candidate.candidate_id,
                "program_sha": compiled.candidate.program.sha(),
            },
        },
        workflow_signature=signature,
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
        evidence_refs=["task_episode_harness_workflow_generation"],
    )


def _update_generated_episode_delayed(
    episode: Any,
    delayed_gain: float,
    *,
    delayed_se_block: float,
    delayed_gain_over_se: float | None,
    delayed_origins: tuple[int, ...],
) -> Any:
    support_gain = float(episode.support_response.get("gain") or 0.0)
    m = MATERIAL_THRESHOLD
    support_positive = support_gain >= m
    delayed_positive = delayed_gain >= m
    if support_positive and delayed_positive:
        status, relation = STATUS_LOCAL_ACTIVE, RELATION_POSITIVE
    elif support_positive and not delayed_positive:
        status, relation = STATUS_RESTRICTED, RELATION_CONFLICT
    elif abs(support_gain) < m and abs(delayed_gain) < m:
        status, relation = STATUS_EPISODE_ONLY, RELATION_ABSTAIN
    elif not support_positive and not delayed_positive:
        status, relation = STATUS_EPISODE_ONLY, RELATION_NEGATIVE
    else:
        status, relation = STATUS_EPISODE_ONLY, RELATION_CONFLICT
    return dataclasses.replace(
        episode,
        delayed_response={
            "evaluated": True,
            "gain": delayed_gain,
            "se_block": delayed_se_block,
            "gain_over_se": delayed_gain_over_se,
            "block_origins": list(delayed_origins),
        },
        evidence_level=EVIDENCE_DELAYED,
        local_status=status,
        relation=relation,
    )


def _generated_lifecycle(
    *,
    repo_root: Path,
    arm: str,
    winner: Any,
    compiled: CompiledWorkflow,
    scope: frozenset[str],
    values: dict[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: dict[str, Any],
    eval_uids: list[str],
    delayed_origins: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any] | None]:
    baseline = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(repo_root / f".workflow_gen_{arm}_store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(_FastAgentStub(), baseline, experience_episodes=())
    method.append_experience_episode(winner)
    steps = compiled.candidate.program.execution_steps()
    card = {
        "pattern_id": "workflow-gen-target-episode",
        "failure_family": "natural_readiness_observation",
        "observable_signature": {
            "task_kind": "forecast",
            T1_SCOPE_FEATURE: "high",
        },
        "workflow": {
            "steps": [
                {"op": op, "params": dict(params)} for op, params in steps
            ],
        },
    }
    method_event = method.handle_fast_winner(
        winner,
        steps,
        controller=controller,
        store=store,
        card=card,
        evaluator=lambda _s, _m: type("R", (), {
            "gain": None, "verification": type("V", (), {"passed": True})(),
        })(),
        fast_features={"task_kind": "forecast", T1_SCOPE_FEATURE: "high"},
        support_gain=float(winner.support_response["gain"]),
        confirmed_cause=WORKFLOW_GEN_CAUSE,
    )
    delayed_event: dict[str, Any] = {"stage": "no_pending"}
    delayed_probe: dict[str, Any] | None = None
    if method_event.get("stage") == "pending":
        holder: dict[str, Any] = {}

        def delayed_evaluator(_steps: Any, _mode: int) -> Any:
            probe = _generated_probe(
                mapped_roster,
                values,
                config,
                delayed_origins,
                eval_uids,
                compiled,
                scope,
            )
            holder["probe"] = probe
            return type("R", (), {
                "gain": probe["macro_gain"],
                "verification": type("V", (), {"passed": True})(),
            })()

        delayed_event = method.handle_feedback_delayed(
            delayed_evaluator, episode_id=winner.episode_id
        )
        delayed_probe = holder.get("probe")
        if isinstance(delayed_probe, dict):
            winner = _update_generated_episode_delayed(
                winner,
                float(delayed_probe["macro_gain"]),
                delayed_se_block=float(delayed_probe["se_block"]),
                delayed_gain_over_se=delayed_probe["gain_over_se"],
                delayed_origins=delayed_origins,
            )
            method.update_experience_episode(winner)
    return method_event, delayed_event, winner, delayed_probe


def _run_generation_arm(
    *,
    arm: str,
    repo_root: Path,
    target_spec: dict[str, Any],
    scope: frozenset[str],
    values: dict[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: dict[str, Any],
    eval_uids: list[str],
    payload: dict[str, Any],
    llm_counter: list[int],
) -> dict[str, Any]:
    response, compiled, status = _generation_call(payload)
    llm_counter[0] += 1
    record: dict[str, Any] = {
        "arm": arm,
        "generation_status": status,
        "proposal": response,
        "probe": None,
        "winner": None,
        "delayed": None,
        "lifecycle": {
            "method_event": {"stage": "no_winner"},
            "delayed_event": {"stage": "no_winner"},
        },
    }
    if compiled is None:
        return record
    steps = compiled.candidate.program.execution_steps()
    try:
        probe = _generated_probe(
            mapped_roster,
            values,
            config,
            target_spec["support_origins"],
            eval_uids,
            compiled,
            scope,
        )
    except Exception as exc:  # noqa: BLE001
        record["generation_status"] = f"INSTRUMENT_FAILED:{type(exc).__name__}"
        return record
    episode = _make_generated_episode(
        arm=arm,
        task_episode_id=target_spec["task_episode_id"],
        compiled=compiled,
        scope=scope,
        probe=probe,
        support_origins=target_spec["support_origins"],
        experience_use=tuple(
            str(item) for item in (response.get("experience_use") or [])
        ),
    )
    draft_formed = float(probe["macro_gain"]) >= MATERIAL_THRESHOLD
    record["probe"] = {
        "macro_gain": probe["macro_gain"],
        "se_block": probe["se_block"],
        "gain_over_se": probe["gain_over_se"],
        "program_steps": [
            {"op": op, "params": dict(params)} for op, params in steps
        ],
        "draft_formed": draft_formed,
        "episode": episode.to_dict(),
    }
    if not draft_formed:
        record["winner"] = None
        return record
    method_event, delayed_event, updated, delayed_probe = _generated_lifecycle(
        repo_root=repo_root,
        arm=arm,
        winner=episode,
        compiled=compiled,
        scope=scope,
        values=values,
        mapped_roster=mapped_roster,
        config=config,
        eval_uids=eval_uids,
        delayed_origins=target_spec["delayed_origins"],
    )
    record["lifecycle"] = {
        "method_event": method_event,
        "delayed_event": delayed_event,
    }
    record["probe"]["episode"] = updated.to_dict()
    record["delayed"] = delayed_probe
    record["winner"] = {
        "episode_id": updated.episode_id,
        "workflow": updated.workflow_signature,
        "program_steps": [
            {"op": op, "params": dict(params)} for op, params in steps
        ],
        "local_status": updated.local_status,
        "delayed_gain": updated.delayed_response.get("gain"),
        "delayed_se_block": updated.delayed_response.get("se_block"),
        "delayed_gain_over_se": updated.delayed_response.get("gain_over_se"),
    }
    return record


def run_workflow_generation(
    report_path: Path = REPORT_REL,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    target_spec = _target_spec()
    target_feedback = _target_fixed_feedback(report)
    source_experiences = _source_experiences(report)
    cohort = _load_natural_cohort(repo_root)
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

    public_context = _public_context_for_generation(values, train_uids)
    inventory = build_public_operator_inventory(
        "forecast", public_context["representative_features"]
    )

    # Frozen-scope pre-assertion: use the exact scope already recorded for the
    # exposed natural episode, not a fresh observation result.
    natural_target = next(
        episode
        for episode in report.get("natural_flow", {}).get("episodes", [])
        if episode.get("task_episode_id") == WORKFLOW_GEN_TARGET_TASK
    )
    frozen_scope = frozenset(natural_target.get("agent_scope") or [])
    scope_matches_frozen = frozen_scope == public_context["scope"]
    if not scope_matches_frozen:
        raise RuntimeError(
            "workflow generation pre-assertion failed: recomputed public "
            "scope differs from the frozen natural_k1_03 agent_scope"
        )

    payload_a3 = _generation_payload(
        target_spec=target_spec,
        public_context=public_context,
        inventory=inventory,
        target_feedback=target_feedback,
        source_experiences=[],
    )
    payload_a5 = _generation_payload(
        target_spec=target_spec,
        public_context=public_context,
        inventory=inventory,
        target_feedback=target_feedback,
        source_experiences=source_experiences,
    )

    a3_without_source = {
        key: value for key, value in payload_a3.items()
        if key != "source_experiences"
    }
    a5_without_source = {
        key: value for key, value in payload_a5.items()
        if key != "source_experiences"
    }
    input_check = {
        "same_except_source_experiences": (
            _canonical_fingerprint(a3_without_source)
            == _canonical_fingerprint(a5_without_source)
        ),
        "a3_source_experience_count": len(payload_a3["source_experiences"]),
        "a5_source_experience_count": len(payload_a5["source_experiences"]),
    }

    llm_counter = [0]
    a3 = _run_generation_arm(
        arm="A3",
        repo_root=repo_root,
        target_spec=target_spec,
        scope=public_context["scope"],
        values=values,
        mapped_roster=mapped_roster,
        config=config,
        eval_uids=eval_uids,
        payload=payload_a3,
        llm_counter=llm_counter,
    )
    a5 = _run_generation_arm(
        arm="A5",
        repo_root=repo_root,
        target_spec=target_spec,
        scope=public_context["scope"],
        values=values,
        mapped_roster=mapped_roster,
        config=config,
        eval_uids=eval_uids,
        payload=payload_a5,
        llm_counter=llm_counter,
    )

    fixed_gains = [
        float(row["support_gain"])
        for row in target_feedback
        if isinstance(row.get("support_gain"), (int, float))
    ]
    fixed_best_gain = float(max(fixed_gains)) if fixed_gains else 0.0
    fixed_programs = {
        str(row["program"])
        for row in target_feedback
        if isinstance(row.get("program"), str)
    }

    def effective(arm_row: dict[str, Any]) -> bool:
        return bool(arm_row.get("winner") is not None)

    a3_effective, a5_effective = effective(a3), effective(a5)
    if not a3_effective and not a5_effective:
        verdict = "WORKFLOW_GENERATION_NO_EFFECTIVE_CANDIDATE_DEV"
    elif a5_effective and not a3_effective:
        verdict = "A5_SOURCE_GENERATION_ADVANTAGE_DEV_SIGNAL"
    elif a3_effective and not a5_effective:
        verdict = "A3_TARGET_ONLY_GENERATION_ADVANTAGE_DEV_SIGNAL"
    else:
        verdict = "WORKFLOW_GENERATION_BOTH_EFFECTIVE_DEV"

    def arm_summary(arm_row: dict[str, Any]) -> dict[str, Any]:
        probe = arm_row.get("probe") or {}
        generated_gain = probe.get("macro_gain")
        steps = probe.get("program_steps") or []
        op_names = [
            str(step.get("op"))
            for step in steps
            if isinstance(step, dict) and step.get("op")
        ]
        fixed_pool_reuse = (
            len(op_names) == 1 and op_names[0] in fixed_programs
        )
        support_gain_matches_fixed = (
            generated_gain is not None
            and any(
                abs(float(generated_gain) - gain) <= 1e-12
                for gain in fixed_gains
            )
        )
        return {
            "generation_status": arm_row["generation_status"],
            "generated_support_gain": generated_gain,
            "generated_support_se_block": probe.get("se_block"),
            "generated_support_gain_over_se": probe.get("gain_over_se"),
            "draft_formed": bool(probe.get("draft_formed")),
            "beats_fixed_best_support": (
                generated_gain is not None
                and float(generated_gain) > fixed_best_gain
            ),
            "fixed_pool_reuse": fixed_pool_reuse,
            "support_gain_matches_fixed_feedback": support_gain_matches_fixed,
            "winner": arm_row.get("winner"),
        }

    workflow_generation = {
        "protocol_version": "workflow_generation_dev_v1",
        "question": (
            "Can Source Experience help the LLM generate a more effective "
            "data adaptation Workflow than Target Context/feedback alone?"
        ),
        "target": {
            "task_episode_id": WORKFLOW_GEN_TARGET_TASK,
            "support_origins": list(target_spec["support_origins"]),
            "delayed_origins": list(target_spec["delayed_origins"]),
            "agent_scope": sorted(public_context["scope"]),
            "scope_matches_frozen_natural_episode": scope_matches_frozen,
            "representative_binding_series": (
                public_context["representative_uid"]
            ),
        },
        "source": {
            "task_episode_ids": list(WORKFLOW_GEN_SOURCE_TASKS),
            "trajectories": len(source_experiences),
            "target_episode_excluded": True,
        },
        "fixed_pool_baseline": {
            "trajectories": target_feedback,
            "best_support_gain": fixed_best_gain,
        },
        "generation_input_check": input_check,
        "arms": {"A3": a3, "A5": a5},
        "summaries": {
            "A3": arm_summary(a3),
            "A5": arm_summary(a5),
        },
        "verdict": verdict,
        "claim_scope": (
            "development diagnostic on one already-exposed natural Task "
            "Episode with one stochastic generation trajectory per arm. "
            "Source and Target share the same K1 cohort and overlapping "
            "origin blocks, so no transfer claim is made. A positive or "
            "negative A5/A3 difference here is a signal for the next "
            "development step, not a reproducible result."
        ),
        "fixed_pool_reuse_note": (
            "fixed_pool_reuse=true means the generated proposal is a single "
            "operator already probed in the target fixed-pool feedback; "
            "support_gain_matches_fixed_feedback=true then means this run "
            "reused an already-exposed Support outcome and only opened a new "
            "delayed outcome when a Draft was formed."
        ),
        "llm_api_call_count": llm_counter[0],
        "wall_seconds": time.perf_counter() - started,
    }
    report["historical_verdict_a5a3_permission_replay"] = report.get(
        "verdict"
    )
    report["phase"] = "workflow_generation"
    report["workflow_generation"] = workflow_generation
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return workflow_generation
