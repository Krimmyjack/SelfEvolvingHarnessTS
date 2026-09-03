"""Natural Task Episode longitudinal slice (development K1 data, no injection).

Source Memory starts empty.  Each pre-registered Task Episode aggregates
multiple origins in its Support block and opens delayed only when an actual
Draft is formed.  No A5/A3 and no Memory claim is made in this phase.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.normal_flow import (
    _FastAgentStub,
    _nf_agent_decision,
    _nf_call,
)
from evaluation.functional.task_episode_harness.public_context import (
    PUBLIC_CONTEXT_PROJECTION_FEATURE,
    build_task_public_context,
    run_context_census,
)
from evaluation.functional.task_episode_harness.runner import (
    MATERIAL_THRESHOLD,
    REPORT_REL,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import (
    T1_MAX_PROBES,
    TASK_CONSUMER_KEY,
    _task_probe,
    _update_episode_delayed,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_SUPPORT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_DRAFT,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

NATURAL_POOL = (
    "outlier_mad",
    "hampel_filter",
    "impute_fft",
    "impute_ema",
    "period_complete",
    "winsorize",
)
NATURAL_EPISODES = (
    {
        "task_episode_id": "natural_k1_01",
        "support_origins": (888, 984, 1104),
        "delayed_origins": (1368, 1800, 2856),
    },
    {
        "task_episode_id": "natural_k1_02",
        "support_origins": (984, 1104, 1368),
        "delayed_origins": (1800, 2856, 3648),
    },
    {
        "task_episode_id": "natural_k1_03",
        "support_origins": (1104, 1368, 1800),
        "delayed_origins": (2856, 3648, 3888),
    },
    {
        "task_episode_id": "natural_k1_04",
        "support_origins": (1368, 1800, 2856),
        "delayed_origins": (3648, 3888),
    },
)


def _natural_initial_order(
    scope: frozenset[str],
    public_context: dict[str, Any],
) -> dict[str, Any]:
    task_context = {
        "task_kind": public_context["task_kind"],
        "task_signature": dict(public_context["task_signature"]),
        "observation_cutoff": int(public_context["observation_cutoff"]),
        "representative_series_uid": public_context["representative_uid"],
        "representative_features": dict(
            public_context["representative_features"]
        ),
    }
    payload = {
        "task": TASK_CONSUMER_KEY,
        "task_context": task_context,
        "scope_policy": {
            "feature": public_context["scope_feature"],
            "bin": public_context["scope_bin"],
            "selected_series_count": len(scope),
        },
        "allowed_programs": [{"op": op, "params": {}} for op in NATURAL_POOL],
        "source_experiences": [],
    }
    system = (
        "Choose up to three Workflows to probe, in order, for this natural "
        "Target forecasting task. The task_context field contains the "
        "deployment-visible Target observations; do not invent additional "
        "observations. You have no Source Memory yet. Return JSON: "
        '{"program_order": ["outlier_mad", "hampel_filter", "impute_ema"]}. '
        "Use exactly names from allowed_programs, no duplicates."
    )
    response = _nf_call([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    order = response.get("program_order")
    if not isinstance(order, list) or not 1 <= len(order) <= T1_MAX_PROBES:
        raise RuntimeError(f"invalid program_order: {order!r}")
    if any(op not in NATURAL_POOL for op in order) or len(set(order)) != len(order):
        raise RuntimeError(f"program_order contains illegal/duplicate names: {order!r}")
    return {"program_order": [str(x) for x in order[:T1_MAX_PROBES]], "raw": response}


def _make_natural_episode(
    *,
    task_episode_id: str,
    attempt_index: int,
    program: str,
    scope: frozenset[str],
    probe: dict[str, Any],
    support_origins: tuple[int, ...],
    public_context: dict[str, Any],
) -> Any:
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    projection_bin = public_context["task_signature"].get(
        PUBLIC_CONTEXT_PROJECTION_FEATURE
    )
    return build_episode(
        episode_id=f"{task_episode_id}_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-natural-development",
        context_summary={
            "task_episode_id": task_episode_id,
            "attempt_index": attempt_index,
            "observation_cutoff": int(
                public_context["observation_cutoff"]
            ),
            "observations_used": [
                public_context["scope_feature"],
                PUBLIC_CONTEXT_PROJECTION_FEATURE,
            ],
            "task_signature": dict(public_context["task_signature"]),
            "representative_series_uid": public_context[
                "representative_uid"
            ],
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "source_memory": "empty",
            "development_evidence": True,
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {
                "scope_observation_bin": public_context["scope_bin"],
                "task_projection_bin": projection_bin,
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
            "accepted": positive,
            "block_origins": list(support_origins),
        },
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=RELATION_POSITIVE if positive else RELATION_NEGATIVE,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT if positive else STATUS_EPISODE_ONLY,
        evidence_refs=["task_episode_harness_natural_flow"],
    )


def _winner_lifecycle(
    *,
    repo_root: Path,
    winner: Any,
    scope: frozenset[str],
    values: dict[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: dict[str, Any],
    eval_uids: list[str],
    support_origins: tuple[int, ...],
    delayed_origins: tuple[int, ...],
    public_context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any] | None]:
    baseline = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(repo_root / ".natural_flow_store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(_FastAgentStub(), baseline, experience_episodes=())
    method.append_experience_episode(winner)
    card = {
        "pattern_id": "natural-task-episode",
        "failure_family": "natural_readiness_observation",
        "observable_signature": dict(public_context["task_signature"]),
        "workflow": {"steps": [{"op": winner.workflow_signature, "params": {}}]},
    }
    method_event = method.handle_fast_winner(
        winner,
        [(winner.workflow_signature, {})],
        controller=controller,
        store=store,
        card=card,
        evaluator=lambda _s, _m: type("R", (), {
            "gain": None, "verification": type("V", (), {"passed": True})(),
        })(),
        fast_features=dict(public_context["task_fast_features"]),
        support_gain=float(winner.support_response["gain"]),
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    delayed_event: dict[str, Any] = {"stage": "no_pending"}
    delayed_probe: dict[str, Any] | None = None
    if method_event.get("stage") == "pending":
        holder: dict[str, Any] = {}

        def delayed_evaluator(_steps: Any, _mode: int) -> Any:
            probe = _task_probe(
                mapped_roster,
                values,
                config,
                delayed_origins,
                eval_uids,
                winner.workflow_signature,
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
            winner = _update_episode_delayed(
                winner,
                float(delayed_probe["macro_gain"]),
                delayed_se_block=float(delayed_probe["se_block"]),
                delayed_gain_over_se=delayed_probe["gain_over_se"],
            )
            method.update_experience_episode(winner)
    return method_event, delayed_event, winner, delayed_probe


def run_natural_flow(report_path: Path = REPORT_REL) -> dict[str, Any]:
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

    # C0 inlet binding: compute every Task Context from that Episode's public
    # prefix before any LLM call.  The census is zero-outcome and must show at
    # least two distinct task-level signatures with a frozen matching /
    # non-matching pair before the Runner may continue.
    task_contexts = {
        spec["task_episode_id"]: build_task_public_context(
            values,
            train_uids,
            observation_cutoff=int(spec["support_origins"][0]),
        )
        for spec in NATURAL_EPISODES
    }
    context_census = run_context_census(task_contexts)
    if context_census["verdict"] != "TASK_CONTEXT_INLET_BINDING_PASS":
        raise RuntimeError(context_census["verdict"])

    episodes = []
    natural_bank = []
    llm_calls = 0
    for spec in NATURAL_EPISODES:
        support_origins = spec["support_origins"]
        delayed_origins = spec["delayed_origins"]
        public_context = task_contexts[spec["task_episode_id"]]
        scope = frozenset(public_context["scope_series_uids"])
        record: dict[str, Any] = {
            "task_episode_id": spec["task_episode_id"],
            "support_origins": list(support_origins),
            "delayed_origins": list(delayed_origins),
            "agent_scope": sorted(scope),
            "public_context": public_context,
        }
        if not scope:
            record["interpretation"] = "PROGRAM_SUPPLY_INTERFACE_BLOCKER"
            record["probes"] = []
            record["winner"] = None
            episodes.append(record)
            continue
        try:
            initial = _natural_initial_order(scope, public_context)
            llm_calls += 1
        except Exception as exc:  # noqa: BLE001
            record["interpretation"] = "PROGRAM_SUPPLY_INTERFACE_BLOCKER"
            record["error"] = f"{type(exc).__name__}: {exc}"
            episodes.append(record)
            continue
        order = list(initial["program_order"])
        probes = []
        winner = None
        stop_reason = None
        observation_gap = None
        for attempt_index, program in enumerate(order):
            probe = _task_probe(
                mapped_roster,
                values,
                config,
                support_origins,
                eval_uids,
                program,
                scope,
            )
            episode = _make_natural_episode(
                task_episode_id=spec["task_episode_id"],
                attempt_index=attempt_index,
                program=program,
                scope=scope,
                probe=probe,
                support_origins=support_origins,
                public_context=public_context,
            )
            gain = float(probe["macro_gain"])
            se = float(probe["se_block"])
            gse = probe["gain_over_se"]
            remaining = order[attempt_index + 1:]
            above = gain >= MATERIAL_THRESHOLD
            if above or remaining:
                try:
                    decision = _nf_agent_decision(
                        program=program,
                        gain=gain,
                        se=se,
                        gain_over_se=gse,
                        remaining=remaining,
                        above_threshold=above,
                    )
                    llm_calls += 1
                except Exception as exc:  # noqa: BLE001
                    decision = {
                        "decision": "ABSTAIN",
                        "reason": f"LLM unavailable, fail-safe abstain: {exc}",
                        "raw": None,
                    }
            else:
                decision = {
                    "decision": "ABSTAIN",
                    "reason": "budget exhausted without an acceptable candidate",
                    "raw": None,
                }
            record_probe = {
                "attempt_index": attempt_index,
                "program": program,
                "support_gain": gain,
                "support_se_block": se,
                "support_gain_over_se": gse,
                "agent_decision": decision,
                "mechanical_gate": "PASS" if above else "REJECT",
                "episode": episode.to_dict(),
            }
            probes.append(record_probe)
            natural_bank.append(episode.to_dict())
            action = decision["decision"]
            if action == "TRUST_DRAFT" and above:
                winner = episode
                stop_reason = "TRUST_DRAFT_GATE_PASS"
                break
            if action == "TRUST_DRAFT" and not above:
                record_probe["mechanical_gate"] = "REJECT_TRUST_BELOW_THRESHOLD"
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
                observation_gap = {
                    "program": program,
                    "reason": decision.get("reason"),
                }
                stop_reason = "REQUEST_OBSERVATION"
                break

        record["initial_order"] = initial
        record["probes"] = probes
        record["stop_reason"] = stop_reason
        record["observation_gap"] = observation_gap

        lifecycle = {
            "method_event": {"stage": "no_winner"},
            "delayed_event": {"stage": "no_winner"},
        }
        delayed_probe = None
        if winner is not None:
            method_event, delayed_event, updated, delayed_probe = _winner_lifecycle(
                repo_root=repo_root,
                winner=winner,
                scope=scope,
                values=values,
                mapped_roster=mapped_roster,
                config=config,
                eval_uids=eval_uids,
                support_origins=support_origins,
                delayed_origins=delayed_origins,
                public_context=public_context,
            )
            lifecycle = {
                "method_event": method_event,
                "delayed_event": delayed_event,
            }
            for probe in probes:
                if probe["episode"]["episode_id"] == winner.episode_id:
                    probe["episode"] = updated.to_dict()
            winner = updated
            # replace support episode in natural bank with delayed-updated one
            for idx, ep in enumerate(natural_bank):
                if ep.get("episode_id") == winner.episode_id:
                    natural_bank[idx] = winner.to_dict()

        support_positive = any(
            p["support_gain"] >= MATERIAL_THRESHOLD for p in probes
        )
        delayed_negative = bool(
            winner is not None
            and winner.delayed_response.get("evaluated")
            and float(winner.delayed_response.get("gain") or 0.0)
            < -MATERIAL_THRESHOLD
        )
        if winner is None and stop_reason == "NO_DRAFT_IN_BUDGET":
            interpretation = "NO_DRAFT_IN_BUDGET"
        elif winner is None and stop_reason == "REQUEST_OBSERVATION":
            interpretation = "REQUEST_OBSERVATION"
        elif winner is None and stop_reason == "AGENT_ABSTAIN":
            interpretation = "AGENT_ABSTAIN"
        elif winner is None:
            interpretation = "NO_DRAFT_IN_BUDGET"
        elif support_positive and delayed_negative:
            interpretation = "SCOPE_RISK_CONFLICT"
        elif winner.local_status == "LOCAL_ACTIVE":
            interpretation = "NATURAL_TARGET_LOCAL_SKILL_PASS"
        else:
            interpretation = "DRAFT_NOT_ACTIVATED"
        record["winner"] = (
            {
                "episode_id": winner.episode_id,
                "workflow": winner.workflow_signature,
                "local_status": winner.local_status,
                "delayed_gain": winner.delayed_response.get("gain"),
                "delayed_se_block": winner.delayed_response.get("se_block"),
                "delayed_gain_over_se": winner.delayed_response.get(
                    "gain_over_se"
                ),
            }
            if winner is not None else None
        )
        record["delayed"] = delayed_probe
        record["lifecycle"] = lifecycle
        record["interpretation"] = interpretation
        episodes.append(record)

    natural_flow = {
        "development_evidence": True,
        "source_memory_start": "empty",
        "context_inlet_binding": {
            "verdict": context_census["verdict"],
            "census": context_census,
        },
        "episodes": episodes,
        "natural_bank": natural_bank,
        "llm_api_call_count": llm_calls,
        "wall_seconds": time.perf_counter() - started,
        "verdict": "NATURAL_FLOW_RECORDED",
        "memory_claim": (
            "No Memory claim is made in this phase; Source Memory was empty "
            "and only natural development Experience was generated."
        ),
    }
    report["phase"] = "natural_flow"
    report["natural_flow"] = natural_flow
    report["natural_bank"] = natural_bank
    report["verdict"] = natural_flow["verdict"]
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return natural_flow
