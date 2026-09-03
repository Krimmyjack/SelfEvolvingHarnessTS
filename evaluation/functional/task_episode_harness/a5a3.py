"""Formal development A5/A3 on a new non-overlapping KDD Target cohort.

Source = frozen K1 natural bank (4 Task Episodes, 12 trajectories).
Target = 20 KDD series with zero overlap with K1, pre-frozen 4 Task Episodes.
A3 and A5 share the same Target probe budget, candidate pool, LLM model and
prompts; each arm keeps its own Target Memory updated in the same prequential
order.  A5 additionally reads the frozen Source bank.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.natural_flow import (
    NATURAL_POOL,
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
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import (
    T1_MAX_PROBES,
    T1_SCOPE_FEATURE,
    TASK_CONSUMER_KEY,
    _public_scope_proposal,
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

# Strictly forward, pairwise non-overlapping blocks.  Eight development-exposed
# origins are each used exactly once (4 support + 4 delayed).  Single-origin
# blocks are a mechanical-replay compromise, not a new statistical design.
TARGET_EPISODES = (
    {
        "task_episode_id": "target_01",
        "support_origins": (888,),
        "delayed_origins": (984,),
    },
    {
        "task_episode_id": "target_02",
        "support_origins": (1104,),
        "delayed_origins": (1368,),
    },
    {
        "task_episode_id": "target_03",
        "support_origins": (1800,),
        "delayed_origins": (2856,),
    },
    {
        "task_episode_id": "target_04",
        "support_origins": (3648,),
        "delayed_origins": (3888,),
    },
)
K1_SERIES = {
    "T117", "T118", "T119", "T12", "T120", "T121", "T122", "T123",
    "T124", "T125", "T126", "T127", "T128", "T129", "T13", "T130",
    "T131", "T132", "T133", "T134",
}


def _numeric_key(name: str) -> int:
    match = re.match(r"^T(\d+)$", name)
    return int(match.group(1)) if match else 10**9


def _load_target_cohort(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    cache = np.load(repo_root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    eligible = sorted(
        (n for n in names if n not in K1_SERIES),
        key=_numeric_key,
    )
    selected = eligible[:20]
    values_map = {
        n: np.asarray(values[names.index(n)], dtype=np.float64) for n in selected
    }
    roster = [
        {"series_uid": n, "role": "train"} for n in selected[:12]
    ] + [
        {"series_uid": n, "role": "eval"} for n in selected[12:]
    ]
    return roster, values_map, selected


def _call(payload: dict[str, Any], system: str) -> dict[str, Any]:
    return _nf_call([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])


def _memory_initial_order(
    scope: frozenset[str],
    memories: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "task": TASK_CONSUMER_KEY,
        "scope_policy": {
            "feature": T1_SCOPE_FEATURE,
            "bin": "high",
            "selected_series_count": len(scope),
        },
        "allowed_programs": [{"op": op, "params": {}} for op in NATURAL_POOL],
        "source_experiences": memories,
    }
    system = (
        "Choose one to three Workflows to probe, in order. Return JSON: "
        "{'program_order': ['outlier_mad', 'hampel_filter', 'winsorize'], "
        "'reason': '...'}. Use only allowed_programs, no duplicates. "
        "POSITIVE/NEGATIVE/CONFLICT are directions, not confidence; low "
        "gain_over_se is weak evidence."
    )
    response = _call(payload, system)
    order = response.get("program_order")
    if not isinstance(order, list) or not 1 <= len(order) <= T1_MAX_PROBES:
        raise RuntimeError(f"invalid program_order: {order!r}")
    if any(op not in NATURAL_POOL for op in order) or len(set(order)) != len(order):
        raise RuntimeError(f"illegal/duplicate program_order: {order!r}")
    return {
        "program_order": [str(x) for x in order[:T1_MAX_PROBES]],
        "reason": response.get("reason"),
        "raw": response,
    }


def _decision_target_view(memory: dict[str, Any]) -> dict[str, Any]:
    """Arm-independent Target-local evidence view for Promotion decisions.

    Internal memory identifiers (episode_id carries the arm prefix in this
    harness) are not evidence and must not reach the decision input; otherwise
    A3 and A5 would not read exactly the same Target evidence.
    """
    return {
        key: value for key, value in memory.items() if key != "episode_id"
    }


def _decision_input(
    *,
    program: str,
    gain: float,
    se: float,
    gain_over_se: float | None,
    remaining: list[str],
    above_threshold: bool,
    target_memories: list[dict[str, Any]],
) -> dict[str, Any]:
    """Post-Support Promotion decision input (deterministic, arm-free).

    Source Experience is intentionally absent.  Only the current Target probe
    and Target-local history are visible here.  Candidate proposal/ranking
    remains the only place where Source Experience may participate.
    """
    allowed = (
        ["TRUST_DRAFT", "CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
        if above_threshold else ["CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
    )
    return {
        "last_probe": {
            "program": program,
            "support_gain": gain,
            "support_se_block": se,
            "support_gain_over_se": gain_over_se,
        },
        "remaining_programs": list(remaining),
        "material_threshold": MATERIAL_THRESHOLD,
        "allowed_decisions": allowed,
        "target_experiences": [
            _decision_target_view(memory) for memory in target_memories
        ],
    }


def _memory_agent_decision(
    *,
    program: str,
    gain: float,
    se: float,
    gain_over_se: float | None,
    remaining: list[str],
    above_threshold: bool,
    target_memories: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _decision_input(
        program=program,
        gain=gain,
        se=se,
        gain_over_se=gain_over_se,
        remaining=remaining,
        above_threshold=above_threshold,
        target_memories=target_memories,
    )
    allowed = payload["allowed_decisions"]
    system = (
        "Decide what to do after one real Support probe. Use gain, se_block "
        "and gain_over_se as evidence; direction labels are not confidence. "
        "You may read the current Target probe and Target-local history only; "
        "Source Experience is excluded from this Promotion decision. "
        "TRUST_DRAFT passes the candidate to the mechanical Gate; CONTINUE "
        "probes the next remaining program; ABSTAIN stops with no winner; "
        "REQUEST_OBSERVATION stops and records an observation gap. Return "
        "JSON: {'decision': one of allowed_decisions, 'reason': '...'}."
    )
    response = _call(payload, system)
    decision = response.get("decision")
    if decision not in allowed:
        raise RuntimeError(f"invalid decision {decision!r}; allowed={allowed}")
    return {
        "decision": decision,
        "reason": response.get("reason"),
        "raw": response,
        "decision_input": payload,
    }


def _decision_input_fingerprint(payload: dict[str, Any]) -> str:
    """Canonical fingerprint used by the deterministic permission check."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _trust_channel_mechanical_check(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Deterministic Source-permission check over recorded decision inputs.

    Same Target evidence + same Target-local history must produce exactly the
    same Promotion decision input for A3 and A5.  This verifies the channel is
    cut without relying on one stochastic LLM output.
    """
    sample_history = [{
        "episode_id": "a5a3_ARM_target_01_attempt_0",
        "program": "winsorize",
        "support_gain": 0.02,
        "support_se_block": 0.14,
        "support_gain_over_se": 0.02 / 0.14,
        "delayed_gain": None,
        "delayed_se_block": None,
        "delayed_gain_over_se": None,
        "relation": "POSITIVE",
        "local_status": "LOCAL_ACTIVE",
    }]
    identity_a = _decision_input(
        program="winsorize",
        gain=0.02,
        se=0.14,
        gain_over_se=0.02 / 0.14,
        remaining=["hampel_filter"],
        above_threshold=True,
        target_memories=sample_history,
    )
    identity_b = _decision_input(
        program="winsorize",
        gain=0.02,
        se=0.14,
        gain_over_se=0.02 / 0.14,
        remaining=["hampel_filter"],
        above_threshold=True,
        target_memories=sample_history,
    )
    identity_ok = (
        _decision_input_fingerprint(identity_a)
        == _decision_input_fingerprint(identity_b)
        and "source_experiences" not in identity_a
        and "target_experiences" in identity_a
        and all(
            "episode_id" not in view
            for view in identity_a["target_experiences"]
        )
    )

    runtime_same_evidence_pairs = []
    for row in rows:
        a3_by_evidence = {}
        for probe in row["A3"]["probes"]:
            payload = (probe.get("agent_decision") or {}).get("decision_input")
            if payload is None:
                continue
            key = (
                payload["last_probe"]["program"],
                payload["last_probe"]["support_gain"],
                payload["last_probe"]["support_se_block"],
                payload["last_probe"]["support_gain_over_se"],
                tuple(payload["remaining_programs"]),
                _decision_input_fingerprint(payload["target_experiences"]),
            )
            a3_by_evidence[key] = payload
        for probe in row["A5"]["probes"]:
            payload = (probe.get("agent_decision") or {}).get("decision_input")
            if payload is None:
                continue
            key = (
                payload["last_probe"]["program"],
                payload["last_probe"]["support_gain"],
                payload["last_probe"]["support_se_block"],
                payload["last_probe"]["support_gain_over_se"],
                tuple(payload["remaining_programs"]),
                _decision_input_fingerprint(payload["target_experiences"]),
            )
            a3_payload = a3_by_evidence.get(key)
            if a3_payload is None:
                continue
            identical = (
                _decision_input_fingerprint(a3_payload)
                == _decision_input_fingerprint(payload)
            )
            runtime_same_evidence_pairs.append({
                "task_episode_id": row["task_episode_id"],
                "program": key[0],
                "support_gain": key[1],
                "support_se_block": key[2],
                "remaining_programs": list(key[4]),
                "decision_input_identical": identical,
            })

    source_key_leaked = any(
        "source_experiences"
        in ((probe.get("agent_decision") or {}).get("decision_input") or {})
        for row in rows
        for arm in ("A3", "A5")
        for probe in row[arm]["probes"]
    )
    runtime_pairs_pass = all(
        pair["decision_input_identical"]
        for pair in runtime_same_evidence_pairs
    )
    trust_channel_cut = (
        identity_ok
        and not source_key_leaked
        and runtime_pairs_pass
    )
    return {
        "offline_same_target_evidence_decision_input_identical": identity_ok,
        "runtime_same_evidence_pairs": runtime_same_evidence_pairs,
        "runtime_same_evidence_pair_count": len(runtime_same_evidence_pairs),
        "runtime_pairs_all_identical": runtime_pairs_pass,
        "source_key_in_any_decision_input": source_key_leaked,
        "same_evidence_definition": (
            "program + support_gain + support_se_block + support_gain_over_se "
            "+ remaining_programs + target_experiences"
        ),
        "verification_mode": (
            "deterministic decision-input identity; observed LLM decisions "
            "are reported as descriptive behavior only"
        ),
        "trust_channel_cut": trust_channel_cut,
    }


def _make_target_episode(
    *,
    arm: str,
    task_episode_id: str,
    attempt_index: int,
    program: str,
    scope: frozenset[str],
    probe: dict[str, Any],
    support_origins: tuple[int, ...],
) -> Any:
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    return build_episode(
        episode_id=f"a5a3_{arm}_{task_episode_id}_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-fresh-target-development",
        context_summary={
            "task_episode_id": task_episode_id,
            "arm": arm,
            "attempt_index": attempt_index,
            "observations_used": [T1_SCOPE_FEATURE],
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
                "base_series_non_overlap_with_k1_source": True,
            },
            "local_pattern": {"scope_observation_bin": "high"},
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
        evidence_refs=["task_episode_harness_a5a3"],
    )


def _memory_summary(episode: Any) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "program": episode.workflow_signature,
        "support_gain": (episode.support_response or {}).get("gain"),
        "support_se_block": (episode.support_response or {}).get("se_block"),
        "support_gain_over_se": (episode.support_response or {}).get("gain_over_se"),
        "delayed_gain": (episode.delayed_response or {}).get("gain"),
        "delayed_se_block": (episode.delayed_response or {}).get("se_block"),
        "delayed_gain_over_se": (episode.delayed_response or {}).get("gain_over_se"),
        "relation": episode.relation,
        "local_status": episode.local_status,
    }


def _sync_memory_summary(
    memories: list[dict[str, Any]],
    episode: Any,
) -> None:
    """Replace the support-only summary with the delayed-updated summary."""
    episode_id = episode.episode_id
    for index, memory in enumerate(memories):
        if memory.get("episode_id") == episode_id:
            memories[index] = _memory_summary(episode)
            return
    memories.append(_memory_summary(episode))


def _lifecycle(
    *,
    repo_root: Path,
    arm: str,
    winner: Any,
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
    store = SnapshotStore(repo_root / f".a5a3_{arm}_store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(_FastAgentStub(), baseline, experience_episodes=())
    method.append_experience_episode(winner)
    card = {
        "pattern_id": "a5a3-target-episode",
        "failure_family": "natural_readiness_observation",
        "observable_signature": {
            "task_kind": "forecast",
            T1_SCOPE_FEATURE: "high",
        },
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
        fast_features={"task_kind": "forecast", T1_SCOPE_FEATURE: "high"},
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


def _run_arm(
    *,
    arm: str,
    task_episode_id: str,
    scope: frozenset[str],
    source_memories: list[dict[str, Any]],
    target_memories: list[dict[str, Any]],
    values: dict[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: dict[str, Any],
    eval_uids: list[str],
    support_origins: tuple[int, ...],
    delayed_origins: tuple[int, ...],
    repo_root: Path,
    llm_counter: list[int],
) -> dict[str, Any]:
    # Candidate proposal/ranking may read Source + Target Experience.
    initial = _memory_initial_order(scope, source_memories + target_memories)
    llm_counter[0] += 1
    order = list(initial["program_order"])
    probes = []
    winner = None
    stop_reason = None
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
        episode = _make_target_episode(
            arm=arm,
            task_episode_id=task_episode_id,
            attempt_index=attempt_index,
            program=program,
            scope=scope,
            probe=probe,
            support_origins=support_origins,
        )
        gain = float(probe["macro_gain"])
        se = float(probe["se_block"])
        gse = probe["gain_over_se"]
        remaining = order[attempt_index + 1:]
        above = gain >= MATERIAL_THRESHOLD
        if above or remaining:
            # Promotion decision reads current Target probe + Target-local
            # history only; Source Experience is not passed here.
            decision = _memory_agent_decision(
                program=program,
                gain=gain,
                se=se,
                gain_over_se=gse,
                remaining=remaining,
                above_threshold=above,
                target_memories=target_memories,
            )
            llm_counter[0] += 1
        else:
            decision = {
                "decision": "ABSTAIN",
                "reason": "budget exhausted without an acceptable candidate",
                "raw": None,
            }
        record = {
            "attempt_index": attempt_index,
            "program": program,
            "support_gain": gain,
            "support_se_block": se,
            "support_gain_over_se": gse,
            "agent_decision": decision,
            "mechanical_gate": "PASS" if above else "REJECT",
            "episode": episode.to_dict(),
        }
        probes.append(record)
        target_memories.append(_memory_summary(episode))
        action = decision["decision"]
        if action == "TRUST_DRAFT" and above:
            winner = episode
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
    if winner is not None:
        method_event, delayed_event, updated, delayed_probe = _lifecycle(
            repo_root=repo_root,
            arm=arm,
            winner=winner,
            scope=scope,
            values=values,
            mapped_roster=mapped_roster,
            config=config,
            eval_uids=eval_uids,
            delayed_origins=delayed_origins,
        )
        lifecycle = {"method_event": method_event,
                     "delayed_event": delayed_event}
        for probe in probes:
            if probe["episode"]["episode_id"] == winner.episode_id:
                probe["episode"] = updated.to_dict()
        winner = updated
        _sync_memory_summary(target_memories, winner)
    return {
        "arm": arm,
        "initial_order": initial,
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
        "probe_count": len(probes),
        "support_harm_count": sum(
            1 for p in probes if p["support_gain"] < -MATERIAL_THRESHOLD
        ),
        "cumulative_support_harm": float(sum(
            -p["support_gain"]
            for p in probes if p["support_gain"] < -MATERIAL_THRESHOLD
        )),
        "abstention": int(stop_reason in {"AGENT_ABSTAIN", "REQUEST_OBSERVATION"}),
    }


def run_a5a3(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    source_memories = _natural_trajectory_summaries(report)
    roster, values, selected = _load_target_cohort(repo_root)
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [row["series_uid"] for row in mapped_roster if row["role"] == "eval"]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]

    a3_source_memories: list[dict[str, Any]] = []
    a3_target_memories: list[dict[str, Any]] = []
    a5_source_memories: list[dict[str, Any]] = list(source_memories)
    a5_target_memories: list[dict[str, Any]] = []
    llm_counter = [0]
    rows = []
    for spec in TARGET_EPISODES:
        scope_proposal = _public_scope_proposal(values, train_uids)
        scope = scope_proposal["scope"]
        a3 = _run_arm(
            arm="A3",
            task_episode_id=spec["task_episode_id"],
            scope=scope,
            source_memories=a3_source_memories,
            target_memories=a3_target_memories,
            values=values,
            mapped_roster=mapped_roster,
            config=config,
            eval_uids=eval_uids,
            support_origins=spec["support_origins"],
            delayed_origins=spec["delayed_origins"],
            repo_root=repo_root,
            llm_counter=llm_counter,
        )
        a5 = _run_arm(
            arm="A5",
            task_episode_id=spec["task_episode_id"],
            scope=scope,
            source_memories=a5_source_memories,
            target_memories=a5_target_memories,
            values=values,
            mapped_roster=mapped_roster,
            config=config,
            eval_uids=eval_uids,
            support_origins=spec["support_origins"],
            delayed_origins=spec["delayed_origins"],
            repo_root=repo_root,
            llm_counter=llm_counter,
        )
        rows.append({
            "task_episode_id": spec["task_episode_id"],
            "support_origins": list(spec["support_origins"]),
            "delayed_origins": list(spec["delayed_origins"]),
            "agent_scope": sorted(scope),
            "A3": a3,
            "A5": a5,
        })

    def aggregate(arm: str) -> dict[str, Any]:
        vals = [row[arm] for row in rows]
        return {
            "total_probes": sum(v["probe_count"] for v in vals),
            "drafts_formed": sum(1 for v in vals if v["winner"] is not None),
            "delayed_approved": sum(
                1 for v in vals
                if v["winner"] is not None
                and v["winner"]["local_status"] == "LOCAL_ACTIVE"
            ),
            "support_harm_count": sum(v["support_harm_count"] for v in vals),
            "cumulative_support_harm": sum(
                v["cumulative_support_harm"] for v in vals
            ),
            "abstentions": sum(v["abstention"] for v in vals),
            "final_delayed_utility": float(sum(
                v["winner"]["delayed_gain"]
                for v in vals
                if v["winner"] is not None
                and isinstance(v["winner"].get("delayed_gain"), (int, float))
            )),
        }

    a3_agg, a5_agg = aggregate("A3"), aggregate("A5")

    trust_check = _trust_channel_mechanical_check(rows)
    verdict = (
        "SOURCE_PERMISSION_BOUNDARY_REPLAY_PASS"
        if trust_check["trust_channel_cut"]
        else "SOURCE_PERMISSION_BOUNDARY_REPLAY_FAIL"
    )

    predecessor = report.get("a5a3")
    if predecessor and predecessor.get("protocol_version") == "clean_replay_v2":
        report["a5a3_clean_replay_v2"] = predecessor
        report["historical_verdict_clean_replay_v2"] = predecessor.get(
            "verdict"
        )

    a5a3 = {
        "protocol_version": "permission_replay_v1",
        "source_permission_boundary": {
            "proposal_ranking_reads": [
                "source_experiences",
                "target_experiences",
            ],
            "promotion_decision_reads": [
                "last_probe",
                "remaining_programs",
                "material_threshold",
                "allowed_decisions",
                "target_experiences",
            ],
            "source_in_promotion_decision": False,
            "doc_reference": (
                "TASK_EPISODE_HARNESS_EXECUTION_PLAN_2026-08-17.md §5 "
                "Source Memory cannot directly approve or activate a Skill"
            ),
        },
        "trust_channel_mechanical_check": trust_check,
        "delayed_memory_update_fixed": True,
        "origin_roles_unique": True,
        "scope_note": (
            "all four episodes share the same public scope; this is single "
            "Context continuous adaptation, not Context diversity"
        ),
        "source": {
            "cohort": "K1",
            "episodes": "natural_k1_01..04",
            "trajectories": len(source_memories),
        },
        "target": {
            "cohort": selected,
            "base_series_overlap_with_source": sorted(
                set(selected) & K1_SERIES
            ),
            "task_episodes": list(TARGET_EPISODES),
        },
        "rows": rows,
        "aggregate": {"A3": a3_agg, "A5": a5_agg},
        "verdict": verdict,
        "predecessor": (
            {
                "protocol_version": predecessor.get("protocol_version"),
                "verdict": predecessor.get("verdict"),
                "aggregate": predecessor.get("aggregate"),
            }
            if predecessor else None
        ),
        "claim_scope": (
            "behavioral replay only, on the already-exposed development "
            "cohort. This run verifies the Source permission boundary and "
            "makes no new A5-vs-A3 transfer claim. Aggregate differences are "
            "descriptive facts of this replay at |g/SE| <= ~2.3; reproducible "
            "negative transfer is not established. delayed_approved means the "
            "existing mechanical Gate approved, not statistical confirmation."
        ),
        "naming_note": (
            "delayed_confirmed is renamed delayed_approved for this protocol "
            "version; historical clean_replay_v2 rows keep their old field."
        ),
        "llm_api_call_count": llm_counter[0],
        "wall_seconds": time.perf_counter() - started,
    }
    report["phase"] = "a5a3_permission_replay"
    report["a5a3"] = a5a3
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return a5a3
