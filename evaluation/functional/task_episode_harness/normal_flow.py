"""Small normal Task Episode flow with Agent-decision-to-runtime closure.

Mechanism recheck (clean + one spike only):

* Agent TRUST_DRAFT -> mechanical Gate decides; pass -> delayed + activation.
* Agent CONTINUE -> probe next Workflow.
* Agent ABSTAIN -> stop; no winner; no delayed; no Skill activation.
* Agent REQUEST_OBSERVATION -> stop and record observation gap.
* gain / se_block / gain_over_se are evidence only; no Runner threshold of 3.
"""
from __future__ import annotations

import json
import os
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
    DELAYED_ORIGINS,
    INJECTION_AMPLITUDE,
    INJECTION_COUNT,
    MATERIAL_THRESHOLD,
    REPORT_REL,
    SUPPORT_ORIGINS,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import (
    T1_MAX_PROBES,
    T1_SCOPE_FEATURE,
    TASK_CONSUMER_KEY,
    _make_episode,
    _public_scope_proposal,
    _task_probe,
    _update_episode_delayed,
)
from evaluation.functional.task_episode_harness.t3 import _source_summaries
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

NF_MODEL = "gpt-5.6-luna"
NF_BASE_URL = "https://api.agicto.cn/v1"
NF_POOL = ("outlier_mad", "hampel_filter")
NF_TASKS = (
    {
        "task_id": "nf_clean_01",
        "faulty": (),
        "seed": 0,
    },
    {
        "task_id": "nf_spike_01",
        "faulty": ("T117", "T118", "T119", "T12", "T120", "T121"),
        "seed": 7,
    },
)


class _FastAgentStub:
    core = None


def _nf_call(messages: list[dict[str, str]]) -> dict[str, Any]:
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

    client = openai.OpenAI(api_key=api_key, base_url=NF_BASE_URL, timeout=120)
    completion = client.chat.completions.create(model=NF_MODEL, messages=messages)
    text = str(completion.choices[0].message.content or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"non-JSON LLM response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _nf_initial_order(
    scope: frozenset[str],
    source_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "task": TASK_CONSUMER_KEY,
        "scope_policy": {
            "feature": T1_SCOPE_FEATURE,
            "bin": "high",
            "selected_series_count": len(scope),
        },
        "allowed_programs": [{"op": op, "params": {}} for op in NF_POOL],
        "source_experiences": source_summaries,
    }
    system = (
        "Choose an initial probe order for this Target task. Return JSON: "
        '{"program_order": ["outlier_mad", "hampel_filter"]}. '
        "POSITIVE/NEGATIVE/CONFLICT are directions, not confidence. "
        "Prefer programs whose prior episodes have higher gain_over_se."
    )
    response = _nf_call([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    order = response.get("program_order")
    if not isinstance(order, list) or sorted(order) != sorted(NF_POOL):
        raise RuntimeError(f"invalid initial order: {order!r}")
    return {"program_order": [str(x) for x in order], "raw": response}


def _nf_agent_decision(
    *,
    program: str,
    gain: float,
    se: float,
    gain_over_se: float | None,
    remaining: list[str],
    above_threshold: bool,
) -> dict[str, Any]:
    allowed = (
        ["TRUST_DRAFT", "CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
        if above_threshold
        else ["CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
    )
    payload = {
        "last_probe": {
            "program": program,
            "support_gain": gain,
            "support_se_block": se,
            "support_gain_over_se": gain_over_se,
        },
        "remaining_programs": remaining,
        "material_threshold": MATERIAL_THRESHOLD,
        "allowed_decisions": allowed,
    }
    system = (
        "You are deciding what to do after one real Support probe. "
        "Direction labels are not confidence. Use gain, se_block and "
        "gain_over_se as evidence. Return JSON: "
        "{'decision': one of allowed_decisions, 'reason': '...'}. "
        "TRUST_DRAFT means pass the candidate to the mechanical Gate; "
        "CONTINUE probes the next remaining program; ABSTAIN stops with no "
        "winner; REQUEST_OBSERVATION stops and records an observation gap. "
        "Do not invent programs or observations."
    )
    response = _nf_call([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    decision = response.get("decision")
    if decision not in allowed:
        raise RuntimeError(f"invalid decision {decision!r}; allowed={allowed}")
    return {"decision": decision, "reason": response.get("reason"), "raw": response}


def _winner_lifecycle(
    *,
    repo_root: Path,
    winner: Any,
    scope: frozenset[str],
    injected: dict[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: dict[str, Any],
    eval_uids: list[str],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    baseline = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(repo_root / ".nf_task_episode_store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(_FastAgentStub(), baseline, experience_episodes=())
    method.append_experience_episode(winner)
    card = {
        "pattern_id": "nf-task-episode",
        "failure_family": "impulsive_outlier_readiness",
        "observable_signature": {
            "task_kind": "forecast",
            T1_SCOPE_FEATURE: "high",
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
        evaluator=lambda _s, _m: type("R", (), {
            "gain": None, "verification": type("V", (), {"passed": True})(),
        })(),
        fast_features={"task_kind": "forecast", T1_SCOPE_FEATURE: "high"},
        support_gain=float(winner.support_response["gain"]),
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    delayed_event: dict[str, Any] = {"stage": "no_pending"}
    if method_event.get("stage") == "pending":
        holder: dict[str, Any] = {}

        def delayed_evaluator(_steps: Any, _mode: int) -> Any:
            probe = _task_probe(
                mapped_roster,
                injected,
                config,
                DELAYED_ORIGINS,
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
        probe = holder.get("probe")
        if isinstance(probe, dict):
            winner = _update_episode_delayed(
                winner,
                float(probe["macro_gain"]),
                delayed_se_block=float(probe["se_block"]),
                delayed_gain_over_se=probe["gain_over_se"],
            )
            method.update_experience_episode(winner)
    return method_event, delayed_event, winner


def run_normal_flow(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    source_summaries = _source_summaries(report)
    cohort = _load_cohort(repo_root)
    roster = cohort["roster"]
    values = cohort["values"]
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]

    tasks = []
    llm_calls = 0
    for task in NF_TASKS:
        if task["faulty"]:
            clean = tuple(uid for uid in train_uids if uid not in task["faulty"])
            injected, _gt = inject_label_touched_corpus(
                values,
                faulty_series=task["faulty"],
                clean_series=clean,
                amplitude=INJECTION_AMPLITUDE,
                count=INJECTION_COUNT,
                seed=task["seed"],
            )
        else:
            injected = {
                uid: np.asarray(value, dtype=np.float64).copy()
                for uid, value in values.items()
            }
        agent = _public_scope_proposal(injected, train_uids)
        scope = agent["scope"]
        try:
            initial = _nf_initial_order(scope, source_summaries)
            llm_calls += 1
        except Exception as exc:  # noqa: BLE001
            tasks.append({
                "task_id": task["task_id"],
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        order = list(initial["program_order"])
        probes = []
        winner = None
        stop_reason = None
        observation_gap = None
        for attempt_index, program in enumerate(order[:T1_MAX_PROBES]):
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
            action = decision["decision"]
            if action == "TRUST_DRAFT":
                if above:
                    winner = episode
                    stop_reason = "TRUST_DRAFT_GATE_PASS"
                    break
                record["mechanical_gate"] = "REJECT_TRUST_BELOW_THRESHOLD"
                if remaining:
                    continue
                stop_reason = "TRUST_DRAFT_GATE_REJECT_NO_REMAINING"
                break
            if action == "CONTINUE":
                if remaining:
                    continue
                stop_reason = "CONTINUE_WITHOUT_REMAINING_PROGRAMS"
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

        lifecycle = {"method_event": {"stage": "no_winner"},
                     "delayed_event": {"stage": "no_winner"}}
        if winner is not None:
            method_event, delayed_event, updated = _winner_lifecycle(
                repo_root=repo_root,
                winner=winner,
                scope=scope,
                injected=injected,
                mapped_roster=mapped_roster,
                config=config,
                eval_uids=eval_uids,
            )
            lifecycle = {
                "method_event": method_event,
                "delayed_event": delayed_event,
            }
            for probe in probes:
                if probe["episode"]["episode_id"] == winner.episode_id:
                    probe["episode"] = updated.to_dict()
            winner = updated
        tasks.append({
            "task_id": task["task_id"],
            "agent_scope": sorted(scope),
            "initial_order": initial,
            "probes": probes,
            "stop_reason": stop_reason,
            "observation_gap": observation_gap,
            "winner": (
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
            ),
            "lifecycle": lifecycle,
        })

    clean_task = tasks[0]
    spike_task = tasks[1]
    clean_no_activation = bool(
        clean_task.get("winner") is None
        and clean_task.get("stop_reason") in {
            "AGENT_ABSTAIN", "REQUEST_OBSERVATION",
            "CONTINUE_WITHOUT_REMAINING_PROGRAMS",
        }
        and clean_task["lifecycle"]["method_event"].get("stage") == "no_winner"
    )
    spike_lifecycle_pass = bool(
        spike_task.get("stop_reason") == "TRUST_DRAFT_GATE_PASS"
        and spike_task["lifecycle"]["method_event"].get("stage") == "pending"
        and spike_task["lifecycle"]["delayed_event"].get("stage") == "approved"
    )
    closure_pass = clean_no_activation and spike_lifecycle_pass
    normal_flow = {
        "pre_registered_tasks": len(NF_TASKS),
        "tasks": tasks,
        "abstain_branch_exercised": any(
            probe.get("agent_decision", {}).get("decision") == "ABSTAIN"
            for probe in clean_task.get("probes", [])
        ),
        "closure_pass": closure_pass,
        "verdict": (
            "AGENT_DECISION_RUNTIME_CLOSURE_PASS"
            if closure_pass else "AGENT_DECISION_RUNTIME_CLOSURE_FAIL"
        ),
        "llm_api_call_count": llm_calls,
        "wall_seconds": time.perf_counter() - started,
    }
    report["phase"] = "normal_flow"
    report["normal_flow"] = normal_flow
    report["verdict"] = normal_flow["verdict"]
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return normal_flow
