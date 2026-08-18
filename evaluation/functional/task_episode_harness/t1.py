"""T1: one development Target Task Episode longitudinal slice (zero LLM).

Mechanical vertical slice under docs/TASK_EPISODE_HARNESS_EXECUTION_PLAN_2026-08-17.md
section T1:

public train-series Observation -> deterministic Agent scope/program proposal
-> complete Task Support probe (3-origin block) -> actual ExperienceEpisode
write -> method-owned LOCAL_DRAFT -> independent delayed block ->
TTHAMethod.handle_feedback_delayed approves or rejects.

Closure from the latest adjudication:

* Agent scope uses the frozen observable bin local_robust_z_peak == high
  (no free numeric threshold);
* the activated Skill carries the same observable applicability;
* every actual probe writes an Episode immediately, failed attempts included;
* the delayed result updates the same Episode in place (DELAYED evidence and
  LOCAL_ACTIVE / RESTRICTED status).

The development Agent is deterministic and never sees injection recipes or
the oracle scope. Live A5/A3 Agent comparison is T3, not T1.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.injection import (
    inject_label_touched_corpus,
)
from evaluation.functional.task_episode_harness.runner import (
    CLEAN_TRAIN_SERIES,
    COMPARATOR_PROGRAM,
    DELAYED_ORIGINS,
    FAULTY_SERIES,
    INJECTION_AMPLITUDE,
    INJECTION_COUNT,
    INJECTION_SEED,
    MATCHED_PROGRAM,
    MATERIAL_THRESHOLD,
    REPORT_REL,
    SUPPORT_ORIGINS,
    _arm_metrics,
    _compiled,
    _evaluate_origins,
    _mapped_roster,
)
from SelfEvolvingHarnessTS.contracts.observables import observable_numeric_bin
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
    SignedEpisodeRetriever,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import evaluate_applicability

T1_OBSERVATION_CUTOFF = 1104
T1_SCOPE_BIN = "high"
T1_SCOPE_FEATURE = "local_robust_z_peak"
T1_MAX_PROBES = 3
T1_CAUSE = "SKILL_LIBRARY_GAP"
TASK_CONSUMER_KEY = "forecast|ridge|sMASE"


class _FastAgentStub:
    """T1 never calls prepare(); handle_fast_winner only needs the snapshot."""

    core = None


@dataclasses.dataclass(frozen=True)
class _TaskReceipt:
    gain: float | None
    verification: Any


def _public_scope_proposal(values: dict[str, Any], train_uids: list[str]) -> dict[str, Any]:
    """Deterministic public-observation Agent.

    Observes local_robust_z_peak on each train series prefix up to the first
    Support origin (1104) and selects the frozen bin ``high``.  The oracle
    injection scope is never read here.
    """
    observations = {}
    selected = set()
    for uid in train_uids:
        prefix = np.asarray(values[uid], dtype=np.float64)[:T1_OBSERVATION_CUTOFF]
        features = dict(extract_public_features(prefix, task_kind="forecast"))
        z_peak = float(features[T1_SCOPE_FEATURE])
        z_bin = observable_numeric_bin(T1_SCOPE_FEATURE, z_peak)
        observations[uid] = {
            "local_robust_z_peak": z_peak,
            "local_robust_z_peak_bin": z_bin,
            "missing_fraction": float(features["missing_fraction"]),
            "level_excursion_score": float(features["level_excursion_score"]),
        }
        if z_bin == T1_SCOPE_BIN:
            selected.add(uid)
    return {
        "scope": frozenset(selected),
        "program_order": [MATCHED_PROGRAM, COMPARATOR_PROGRAM] if selected else [],
        "observations": observations,
        "rule": f"{T1_SCOPE_FEATURE} == {T1_SCOPE_BIN}",
    }


def _task_probe(
    roster: list[dict[str, Any]],
    values: dict[str, Any],
    config: dict[str, Any],
    origins: tuple[int, ...],
    eval_uids: list[str],
    program: str,
    scope: frozenset[str],
) -> dict[str, Any]:
    compiled = _compiled(program, name=f"t1-probe-{program}")
    identity_rows = _evaluate_origins(
        roster, values, None, config, origins, None
    )
    candidate_rows = _evaluate_origins(
        roster, values, compiled, config, origins, set(scope)
    )
    metrics = _arm_metrics(
        identity_rows, candidate_rows, origins, eval_uids
    )
    metrics["program"] = program
    metrics["scope"] = sorted(scope)
    return metrics


def _make_episode(
    *,
    attempt_index: int,
    program: str,
    scope: frozenset[str],
    observations: dict[str, Any],
    probe: dict[str, Any],
) -> Any:
    gain = float(probe["macro_gain"])
    status = STATUS_LOCAL_DRAFT if gain >= MATERIAL_THRESHOLD else STATUS_EPISODE_ONLY
    relation = RELATION_POSITIVE if gain >= MATERIAL_THRESHOLD else RELATION_NEGATIVE
    return build_episode(
        episode_id=f"t1_task_episode_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-injected-development",
        context_summary={
            "task_episode_id": "t1-development-target-task",
            "attempt_index": attempt_index,
            "observations_used": [T1_SCOPE_FEATURE],
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {
                "scope_observation_bin": T1_SCOPE_BIN,
                "scope_observation_mean_z": float(np.mean([
                    float(observations[uid]["local_robust_z_peak"])
                    for uid in scope
                ])) if scope else 0.0,
            },
            "program_geometry": {
                "scope": "training_series_subset",
                "program_steps": [{"op": program, "params": {}}],
            },
        },
        workflow_signature=program,
        support_response={
            "gain": gain,
            "se_block": float(probe["se_block"]),
            "gain_over_se": probe["gain_over_se"],
            "accepted": gain >= MATERIAL_THRESHOLD,
            "block_origins": list(SUPPORT_ORIGINS),
        },
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=relation,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=status,
        evidence_refs=["task_episode_harness_t1"],
    )


def _update_episode_delayed(
    episode: Any,
    delayed_gain: float,
    *,
    delayed_se_block: float | None = None,
    delayed_gain_over_se: float | None = None,
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
            "block_origins": list(DELAYED_ORIGINS),
        },
        evidence_level=EVIDENCE_DELAYED,
        local_status=status,
        relation=relation,
    )


def _fast_features() -> dict[str, Any]:
    return {
        "task_kind": "forecast",
        T1_SCOPE_FEATURE: T1_SCOPE_BIN,
    }


def run_t1(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    cohort = _load_cohort(repo_root)
    roster = cohort["roster"]
    values = cohort["values"]
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]

    injected, _ground_truth = inject_label_touched_corpus(
        values,
        faulty_series=FAULTY_SERIES,
        clean_series=CLEAN_TRAIN_SERIES,
        amplitude=INJECTION_AMPLITUDE,
        count=INJECTION_COUNT,
        seed=INJECTION_SEED,
    )

    agent = _public_scope_proposal(injected, train_uids)
    scope = agent["scope"]

    # Method-owned lifecycle is created before probing so every actual probe
    # can write its Episode immediately (failed attempts included).
    baseline = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    baseline_skill_ids = [skill.skill_id for skill in baseline.skills]
    store = SnapshotStore(repo_root / ".t1_task_episode_store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(
        _FastAgentStub(),  # type: ignore[arg-type]
        baseline,
        experience_episodes=(),
    )

    attempts = []
    winner = None
    for attempt_index, program in enumerate(agent["program_order"][:T1_MAX_PROBES]):
        probe = _task_probe(
            mapped_roster,
            injected,
            config,
            SUPPORT_ORIGINS,
            eval_uids,
            program,
            scope,
        )
        episode = _make_episode(
            attempt_index=attempt_index,
            program=program,
            scope=scope,
            observations=agent["observations"],
            probe=probe,
        )
        method.append_experience_episode(episode)
        attempts.append({
            "attempt_index": attempt_index,
            "program": program,
            "scope": sorted(scope),
            "support": probe,
            "episode": episode.to_dict(),
        })
        if probe["macro_gain"] >= MATERIAL_THRESHOLD:
            winner = episode
            break

    method_event: dict[str, Any] = {"stage": "no_winner"}
    delayed_event: dict[str, Any] = {"stage": "no_pending"}
    delayed_gain: float | None = None
    if winner is not None:
        card = {
            "pattern_id": "t1-task-episode",
            "failure_family": "impulsive_outlier_readiness",
            "observable_signature": {
                "task_kind": "forecast",
                T1_SCOPE_FEATURE: T1_SCOPE_BIN,
            },
            "workflow": {
                "steps": [{"op": winner.workflow_signature, "params": {}}]
            },
        }
        method_event = method.handle_fast_winner(
            winner,
            [(winner.workflow_signature, {})],
            controller=controller,
            store=store,
            card=card,
            evaluator=lambda _steps, _mode: _TaskReceipt(None, type(
                "V", (), {"passed": True})()),
            fast_features=_fast_features(),
            support_gain=float(winner.support_response["gain"]),
            confirmed_cause=T1_CAUSE,
        )
        if method_event.get("stage") == "pending":
            delayed_probe_holder: dict[str, Any] = {}

            def delayed_evaluator(_steps: Any, _mode: int) -> _TaskReceipt:
                delayed_probe = _task_probe(
                    mapped_roster,
                    injected,
                    config,
                    DELAYED_ORIGINS,
                    eval_uids,
                    winner.workflow_signature,
                    scope,
                )
                delayed_probe_holder["probe"] = delayed_probe
                return _TaskReceipt(
                    delayed_probe["macro_gain"],
                    type("V", (), {"passed": True})(),
                )

            delayed_event = method.handle_feedback_delayed(
                delayed_evaluator,
                episode_id=winner.episode_id,
            )
            delayed_gain = delayed_event.get("delayed_gain")
            if isinstance(delayed_gain, (int, float)):
                delayed_probe = delayed_probe_holder.get("probe")
                updated_winner = _update_episode_delayed(
                    winner,
                    float(delayed_gain),
                    delayed_se_block=(
                        float(delayed_probe["se_block"])
                        if isinstance(delayed_probe, dict)
                        else None
                    ),
                    delayed_gain_over_se=(
                        delayed_probe.get("gain_over_se")
                        if isinstance(delayed_probe, dict)
                        else None
                    ),
                )
                method.update_experience_episode(updated_winner)
                winner = updated_winner
                for attempt in attempts:
                    if attempt["attempt_index"] == 0 and attempt["program"] == updated_winner.workflow_signature:
                        attempt["episode"] = updated_winner.to_dict()

    active_snapshot = method._active_snapshot()
    active_skill_ids_after_approval = [
        skill.skill_id for skill in active_snapshot.skills
    ]

    # Zero-outcome scope-binding check: the activated Skill must match high
    # contexts and must not match non-high contexts.
    scope_binding_check: dict[str, Any] = {"checked": False}
    if delayed_event.get("stage") == "approved":
        skill = next(
            (s for s in active_snapshot.skills if s.skill_id == "fast_winner_outlier_mad"),
            None,
        )
        if skill is not None:
            high_match, _ = evaluate_applicability(
                skill.observable_applicability,
                {"task_kind": "forecast", T1_SCOPE_FEATURE: "high"},
            )
            non_high_match, _ = evaluate_applicability(
                skill.observable_applicability,
                {"task_kind": "forecast", T1_SCOPE_FEATURE: "medium"},
            )
            scope_binding_check = {
                "checked": True,
                "skill_id": skill.skill_id,
                "observable_applicability": skill.observable_applicability,
                "high_context_matches": high_match,
                "non_high_context_matches": non_high_match,
                "pass": bool(high_match and not non_high_match),
            }

    # Memory self-check: the delayed-updated winner Episode must be retrievable
    # by the existing signed retriever and must carry DELAYED evidence.
    memory_self_check: dict[str, Any] = {"checked": False}
    if winner is not None:
        retriever = SignedEpisodeRetriever(
            method._experience_episodes,
            task_consumer_key=TASK_CONSUMER_KEY,
            allowed_operators=(winner.workflow_signature,),
        )
        pack = retriever.retrieve(
            winner.context_summary,
            winner.domain_namespace,
        )
        memory_self_check = {
            "checked": True,
            "positive_episode_id": (
                pack.positive.episode_id if pack.positive else None
            ),
            "winner_retrieved_as_positive": bool(
                pack.positive is not None
                and pack.positive.episode_id == winner.episode_id
            ),
            "winner_evidence_level": winner.evidence_level,
            "winner_local_status": winner.local_status,
            "winner_delayed_evaluated": bool(
                winner.delayed_response.get("evaluated")
            ),
            "pass": bool(
                pack.positive is not None
                and pack.positive.episode_id == winner.episode_id
                and winner.evidence_level == EVIDENCE_DELAYED
                and winner.local_status == STATUS_LOCAL_ACTIVE
            ),
        }

    # Controlled revocation check: use the existing revoke_deployed_skill
    # path with a synthetic delayed-harm result to verify behavior restoration
    # to the baseline skill set.  This is a mechanism control, not a claim
    # that the actual delayed block harmed the approved Skill.
    revocation_control: dict[str, Any] = {"attempted": False}
    active_skill_ids_after_revocation = active_skill_ids_after_approval
    if delayed_event.get("stage") == "approved":
        from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
            revoke_deployed_skill,
        )

        class _RevokeResult:
            def __init__(self, method_obj: Any) -> None:
                self.deployed_skill_id = "fast_winner_outlier_mad"
                self._method = method_obj
                self.delayed_utility = -0.05
                self.revoked_skill_id = None
                self.revoked_runtime_bundle_sha = None
                self._fast_skill_event = None

        revoke_result = _RevokeResult(method)
        revocation_control["attempted"] = True
        revocation_control["revoked"] = revoke_deployed_skill(
            revoke_result, store
        )
        active_skill_ids_after_revocation = [
            skill.skill_id
            for skill in method._active_snapshot().skills
        ]
        revocation_control["revoked_skill_id"] = revoke_result.revoked_skill_id
        revocation_control["behavior_restored_to_baseline_skill_ids"] = bool(
            set(active_skill_ids_after_revocation) == set(baseline_skill_ids)
        )

    closures_pass = bool(
        method_event.get("stage") == "pending"
        and delayed_event.get("stage") == "approved"
        and scope_binding_check.get("pass") is True
        and memory_self_check.get("pass") is True
        and revocation_control.get(
            "behavior_restored_to_baseline_skill_ids", False
        )
    )

    t1 = {
        "agent": {
            "kind": "deterministic_public_observation_scope_selector",
            "rule": agent["rule"],
            "observation_cutoff": T1_OBSERVATION_CUTOFF,
            "selected_scope": sorted(scope),
            "program_order": agent["program_order"],
            "private_oracle_scope": list(FAULTY_SERIES),
            "scope_intersection_count": len(scope & set(FAULTY_SERIES)),
        },
        "attempts": attempts,
        "support_winner": (
            {"episode_id": winner.episode_id, "workflow": winner.workflow_signature}
            if winner is not None else None
        ),
        "method_event": method_event,
        "delayed_event": delayed_event,
        "scope_binding_check": scope_binding_check,
        "memory_self_check": memory_self_check,
        "active_skill_ids_after_approval": active_skill_ids_after_approval,
        "baseline_skill_ids": [skill.skill_id for skill in baseline.skills],
        "revocation_control": revocation_control,
        "active_skill_ids_after_revocation_control": active_skill_ids_after_revocation,
        "wall_seconds": time.perf_counter() - started,
        "verdict": (
            "TASK_EPISODE_TARGET_LOCAL_LOOP_DEV_PASS"
            if closures_pass
            else "TASK_EPISODE_LOOP_INCONCLUSIVE"
        ),
        "llm_api_call_count": 0,
    }

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}
    report["phase"] = "T1"
    report["t0_verdict"] = (
        "TASK_EPISODE_SUBSTRATE_READABLE"
        if report.get("substrate", {}).get("readable")
        else "TASK_EPISODE_SUBSTRATE_UNREADABLE"
    )
    report["task_episodes"] = attempts
    report["t1"] = t1
    report["verdict"] = t1["verdict"]
    report["mechanical_checks"] = dict(
        report.get("mechanical_checks") or {},
        t1_llm_calls=0, t1_slow_calls=0, t1_oracle_scope_not_given_to_agent=True, t1_delayed_after_support_freeze=True, t1_scope_binding_pass=bool(scope_binding_check.get("pass")), t1_memory_self_check_pass=bool(memory_self_check.get("pass")),
    )
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return t1
