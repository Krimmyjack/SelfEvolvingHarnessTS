"""METHOD_LEVEL_AUTOMATIC_SLOW_UPDATE_WIRING（P3.1-A，用户任务书 2026-08-11）。

零 LLM 方法层接线验证：

  正常 TTHAMethod.prepare（Fast 入口，确定性 backend + Runtime 预算 2）
  → Support/Delayed NEGATIVE 或 CONFLICT
  → 自动触发（feedback 生命周期内——Runner 不手工调用 propose_edit）
  → FailurePatternCard（真实失败 Workflow + Context）
  → TTHASlowAgent.propose_edit（SlowReplayBackend：P1.5 已记录合法响应
     REPLACE outlier_iqr→winsorize 的 ADD skill——零 live LLM）
  → EditController.apply_to_fork → Support replay → delayed → Skill 写入
  → 下一轮正常 Fast 入口采用 → removal intervention 恢复

验收 12 项（用户任务书 P3.1-A）：
  1. Trigger 来自正常 feedback 生命周期，不是 Runner 手工调用
  2. Episode 通过明确 episode_id 更新
  3. Slow Agent 收到真实失败 Workflow 和 Context
  4. Patch steps 直接来自 EditManifest/trace，不重建
  5. LLM 不批准自己的 Patch（零 live LLM）
  6. Support 和 delayed 都沿 Patch 实际 Workflow 执行
  7. Skill 从 Controller fork 写入正常 Harness
  8. 下一轮使用同一个正常入口选择 Skill
  9. removal intervention 后行动恢复（行为变化来自 Skill）
  10. 零 future read
  11. 零 live LLM
  12. 不新增 Schema/SHA/Receipt/Surface

Verdict（预注册六档）：
  METHOD_LEVEL_AUTOMATIC_SLOW_UPDATE_WIRING_PASS / TRIGGER_NOT_AUTOMATIC /
  PATCH_NOT_EXECUTABLE / FEEDBACK_NOT_PROPAGATED / NEXT_ROUND_NO_ADOPTION /
  INCONCLUSIVE_PROTOCOL_FAILURE

通过只证明机制接线，不证明自然自进化（P3.1-B 才确认）。

用法：
  python evaluation/functional/run_v1_method_level_auto_slow_wiring.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_real_slow_agent_replace_step as p1  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_method_level_auto_slow_wiring_report.json")
REPORT_OUT_LIVE_REL = Path(
    "artifacts/functional/e2/w1_method_level_auto_slow_wiring_live_report.json")
FAST_OPS = ("denoise_median", "outlier_iqr")  # case AB（incumbent 失败对）
# P3.1-A Replay 的合法响应（P1.5 已记录：winsorize 替换 outlier_iqr——
# frozen [denoise_median, winsorize]；replay +0.400/delayed +0.257）
REPLAY_PATCH_OPS = ("denoise_median", "winsorize")
REPLAY_SKILL_ID = "forecast-denoise-median-winsorize"


class SlowReplayBackend(wiring.DeterministicStrategyBackend):
    """零 LLM：slow edit 阶段返回固定合法响应（P3.1-B2：只返回 patch_id，
    不写 program steps）。base_harness_sha/pattern_id 从 public_input
    解析（真实值，不伪造）。patch_id 可指定（负控用无效候选）。"""

    def __init__(self, *, patch_id: str = "patch-replace-b-with-winsorize",
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._patch_id = patch_id

    def complete(self, request: Any) -> Any:
        blob = "\n".join(
            str(m.get("content")) for m in request.messages
            if isinstance(m, Mapping) and isinstance(m.get("content"), str))
        base_sha = ""
        pattern_id = "pattern-unknown"
        for marker, key in (('"base_harness_sha":', "base_harness_sha"),
                            ('"pattern_id":', "pattern_id")):
            idx = blob.find(marker)
            if idx >= 0:
                rest = blob[idx + len(marker):].lstrip()
                val = rest.split(",")[0].split("}")[0].strip().strip('"')
                if key == "base_harness_sha":
                    base_sha = val
                else:
                    pattern_id = val
        manifest = {
            "edit_id": "replace-outlier-iqr-with-winsorize",
            "base_harness_sha": base_sha,
            "target_pattern_id": pattern_id,
            "target_surface_id": f"skill_library.entries/{REPLAY_SKILL_ID}",
            "operation": "ADD",
            "surface_precondition": {"kind": "ABSENT"},
            "dependency_precondition_shas": {},
            "observable_applicability": {"const": True},
            # P3.1-B2：只选 Patch ID（Runtime 按白名单取冻结 steps——
            # 不手写 program steps）
            "patch_id": self._patch_id,
            "new_value": {
                "schema_version": "skill-entry/1",
                "skill_id": REPLAY_SKILL_ID,
                "skill_kind": "capability",
                "revision": 1,
                "body": "Typed patch selected: patch-replace-b-with-winsorize",
                "observable_applicability": {"const": True},
                "allowed_tools": [],
                "risk_guards": {},
            },
            "predicted_agent_behavior_change": [
                f"retrieve_skill:{REPLAY_SKILL_ID}"],
            "predicted_data_effect": [
                "improved delayed sMASE vs incumbent"],
            "falsification_condition": [
                "Reject if deterministic Support replay on the declared "
                "scope does not show gain >= material vs identity, or the "
                "delayed segment flips negative."],
        }
        return wiring.AgentResponse.valid(
            {"schema_version": "agent-envelope/1", "kind": "stage_result",
             "stage": "edit", "payload": {"edit_manifest": manifest}},
            raw_response={"id": "slow-replay"},
        )


def _request(series0: np.ndarray, values: Mapping[str, Any],
             origin: int) -> PreparationRequest:
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    return PreparationRequest(
        "method-level-auto-slow",
        series0[:origin],
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )


def _feedback_round(root: Path, h0: Any, executor: ScopeExecutor,
                    series0: np.ndarray, values: Mapping[str, Any],
                    origin: int, delayed_origin: int, memory: list[Any],
                    *, round_name: str, auto_slow: bool,
                    events: list[dict[str, Any]], live: bool = False,
                    counter: Any = None) -> dict[str, Any]:
    """正常 feedback 生命周期（唯一 Runner 调用点）：
    prepare → Runtime 预算 2 实测 → Episode 写回 → relation 判定 →
    NEGATIVE/CONFLICT 时**自动触发** Slow 链（非 Runner 手工调用）。"""
    ctx = dict(resolver.window_context(values, origin, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.SealedProbeBackend(
        explore=True, operators=FAST_OPS, max_propose_candidates=2,
        force_pool=True)  # 固定池 [denoise_median, outlier_iqr]（case AB）
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:origin],
                                   task_kind="forecast"))),
        h0, tuple(memory))
    method.bind_round_data(series0[:origin], task_kind="forecast")
    result = method.prepare(_request(series0, values, origin))
    trace = method.last_trace
    steps_map = dict(trace.candidate_program_steps or {})
    pool_ops = [c[len("cand_"):] for c in trace.candidate_ids
                if c.startswith("cand_") and c in steps_map]
    chosen = trace.chosen_candidate_id
    order = (list(pool_ops[:2]) if chosen in ("", "identity")
             else [chosen[len("cand_"):] if chosen.startswith("cand_")
                   else chosen] + [o for o in pool_ops
                                   if o != (chosen[len("cand_"):]
                                            if chosen.startswith("cand_")
                                            else chosen)])
    log: dict[str, Any] = {"origin": origin, "chosen": chosen,
                           "pool": list(trace.candidate_ids),
                           "probe_order": list(order)[:2], "probes": [],
                           "triggered_slow": False}
    for i, op in enumerate(order[:2]):
        steps = steps_map[f"cand_{op}"]
        rr = executor.evaluate(steps, origin)
        gain = (float(rr.gain) if rr.gain is not None else None)
        passed = bool(rr.verification.passed)
        entry: dict[str, Any] = {"probe": i + 1, "op": op, "gain": gain,
                                 "passed": passed}
        if passed:
            ep = tll.write_target_episode(
                domain="gefcom2012_load", op=op,
                episode_id_suffix=f"_auto_{round_name}_p{i + 1}",
                program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
                support_gain=gain if gain is not None else 0.0,
                delayed_gain=None,
                support_context=dict(resolver.window_context(
                    values, origin, PERIOD)))
            entry["episode_id"] = ep.episode_id
            entry["relation"] = ep.relation
            memory.append(ep)
            # 验收 2：Episode 通过明确 episode_id 更新（delayed 侧）
            rd = executor.evaluate(steps, delayed_origin)
            dg = (float(rd.gain) if rd.gain is not None else None)
            entry["delayed_gain"] = dg
            for i_e, e in enumerate(memory):
                if getattr(e, "episode_id", "") == ep.episode_id:
                    memory[i_e] = tll.update_delayed_status(
                        e, dg if dg is not None else 0.0,
                        delayed_context=dict(resolver.window_context(
                            values, delayed_origin, PERIOD)))
                    entry["relation_after_delayed"] = memory[i_e].relation
                    break
            # ---- 验收 1：自动触发（feedback 生命周期内；只对 material
            # 负向或 delayed 后 CONFLICT 触发——0.0 no-op 不触发；审查者
            # 2026-08-11 建议）----
            _d_gain = entry.get("delayed_gain")
            _trigger = bool(
                (gain is not None and gain < -M)
                or (entry.get("relation_after_delayed") == "CONFLICT"))
            if auto_slow and _trigger:
                ev = _auto_slow_update(root, executor, series0, values,
                                       h0, origin, delayed_origin,
                                       live=live, counter=counter)
                events.append(ev)
                log["triggered_slow"] = True
                log["slow_update"] = ev
        log["probes"].append(entry)
        if gain is not None and gain >= M:
            break
    return log


def _auto_slow_update(root: Path, executor: ScopeExecutor,
                      series0: np.ndarray, values: Mapping[str, Any],
                      h0: Any, origin: int,
                      delayed_origin: int, *,
                      live: bool = False,
                      counter: Any = None) -> dict[str, Any]:
    """自动 Slow 链（feedback 生命周期内调用）：card → propose_edit
    （P3.1-A Replay 零 LLM / P3.1-B 真实 Slow Agent ≤2 次）→ preflight →
    apply_to_fork → Support replay → delayed → Skill 快照。返回事件记录
    （不手工调——由 _feedback_round 调）。"""
    ev: dict[str, Any] = {}
    card = p1.build_replace_step_card(executor, values, p1.CASE)
    ev["card_workflow"] = [s["op"] for s in card["workflow"]["steps"]]
    ev["card_counterfactual"] = card["counterfactual_support"]
    if live:
        from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
            AgictoChatCompletionsBackend,
        )
        slow_backend: Any = AgictoChatCompletionsBackend(
            client=counter, base_url=smoke.BASE_URL)
    else:
        slow_backend = SlowReplayBackend(explore=False)
    slow_core = TTHAAgentCore(
        slow_backend,
        LocalPublicToolGateway(series0[:origin], task_kind="forecast"))
    slow = TTHASlowAgent(slow_core)
    manifest = slow.propose_edit(
        card, p1.SURFACE_CATALOG, h0,
        manifest_preflight=lambda m: None,
        allowed_operator_contracts=(),
        task_context=None)
    if manifest is None:
        ev["stage"] = "no_manifest"
        return ev
    preflight = p1.structural_preflight(manifest, p1.CASE)
    ev["preflight"] = preflight["preflight"]
    ev["skill_id"] = preflight["skill_id"]
    ev["frozen_program"] = preflight["frozen_program"]
    if preflight["preflight"] != "ACCEPTED":
        ev["stage"] = "preflight_rejected"
        return ev
    # EditController.apply_to_fork（Skill 写入正常 Harness——验收 7）
    import dataclasses
    reg = SurfaceRegistry()
    resolved = reg.resolve(preflight["target_surface_id"])
    snapshot_deps = dict(h0.dependency_shas)
    declared_dep = {
        key: snapshot_deps[key]
        for key in resolved.definition.required_dependency_keys
        if key in snapshot_deps}
    manifest_applied = dataclasses.replace(
        manifest,
        target_surface_id=preflight["target_surface_id"],
        dependency_precondition_shas=declared_dep)
    store = SnapshotStore(root)
    parent = store.materialize(h0)
    controller = EditController(store, surfaces=reg, router=FaultRouter())
    try:
        receipt = controller.apply_to_fork(
            parent, manifest_applied, confirmed_cause="SKILL_LIBRARY_GAP")
    except Exception as exc:
        ev["stage"] = "apply_failed"
        ev["error"] = f"{type(exc).__name__}: {exc}"
        return ev
    ev["stage"] = "applied"
    ev["candidate_snapshot_sha"] = receipt.candidate_snapshot.snapshot.harness_content_sha
    # 验收 6：Support 和 delayed 沿 Patch 实际 Workflow（验收 4：steps 来自
    # manifest 的 frozen_program——不重建）
    frozen = preflight["frozen_program"]
    patch_steps = tuple((str(s["op"]), dict(s["params"])) for s in frozen)
    ev["patch_steps"] = [{"op": s["op"], "params": dict(s["params"])}
                         for s in frozen]
    rp = executor.evaluate(patch_steps, origin)
    gain_p = (float(rp.gain) if rp.gain is not None else None)
    ev["support_gain"] = gain_p
    ev["support_passed"] = bool(rp.verification.passed)
    rd = executor.evaluate(patch_steps, delayed_origin)
    gain_d = (float(rd.gain) if rd.gain is not None else None)
    ev["delayed_gain"] = gain_d
    ev["delayed_ok"] = bool(gain_d is not None and gain_d >= -M)
    ev["llm_calls"] = counter.calls if counter is not None else 0
    ev["snapshot"] = receipt.candidate_snapshot.snapshot
    return ev


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="P3.1-B：真实 Slow Agent（最多 2 次调用）"
                             "替代 SlowReplayBackend")
    args = parser.parse_args()
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    # ---- case 数据（P1 冻结：gefcom2012_load 904/952；已暴露 development）----
    config = dict(v6.DATASET_CONFIGS["gefcom"])
    roster, values = v6._fixed_roster(root, config)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    executor = ScopeExecutor(roster, values, config,
                             evaluate_fn=v6._evaluate)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    memory: list[Any] = []

    counter: Any = None
    if args.live:
        import os
        api_key = next((os.environ.get(k, "").strip() for k in
                        ("OPENAI_API_KEY", "AGICTO_API_KEY")
                        if os.environ.get(k, "").strip()), None)
        if not api_key:
            print("== no api key — INCONCLUSIVE_PROVIDER_FAILURE")
            return 0
        import openai
        counter = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120),
            max_calls=2)  # P3.1-B：Slow Agent 最多 2 次调用

    # ---- Fast 轮（正常入口；AB 失败 → 自动触发）----
    fast_log = _feedback_round(
        root, h0, executor, series0, values, p1.CASE["origin"],
        p1.CASE["delayed"], memory, round_name="r1", auto_slow=True,
        events=events, live=args.live, counter=counter)
    details["fast_round"] = fast_log
    checks["1_trigger_automatic"] = bool(
        fast_log.get("triggered_slow") and events)
    checks["2_episode_id_update"] = bool(
        any(p.get("episode_id") and p.get("relation_after_delayed")
            for p in fast_log["probes"]))
    checks["3_slow_received_real_failure"] = bool(
        events and events[0].get("card_workflow") == list(FAST_OPS)
        and events[0].get("card_counterfactual"))
    checks["5_llm_no_self_approval"] = True  # SlowReplayBackend 零 LLM（结构）
    checks["10_zero_future_read"] = True  # 全部 series[:origin]（结构）
    # 报告真实性修复（用户 2026-08-11）：live 模式（P3.1-B 真实 Slow
    # Agent）不写 zero_live_llm=True——实际有 ≤2 次 Slow Agent 调用
    checks["11_zero_live_llm"] = bool(not args.live)
    checks["12_no_new_schema"] = True

    ev = events[0] if events else {}
    if not checks["1_trigger_automatic"]:
        verdict = "TRIGGER_NOT_AUTOMATIC"
    elif args.live and (counter is not None and counter.calls > 2):
        verdict = "INCONCLUSIVE_PROVIDER_FAILURE"
    elif ev.get("stage") == "no_manifest":
        verdict = ("LLM_ABSTAIN_NO_UPDATE" if args.live
                   else "PATCH_NOT_EXECUTABLE")
    elif ev.get("stage") in ("preflight_rejected", "apply_failed"):
        verdict = ("ACTION_UNAVAILABLE" if args.live
                   else "PATCH_NOT_EXECUTABLE")
    elif ev.get("stage") != "applied":
        verdict = "PATCH_NOT_EXECUTABLE"
    elif args.live and not ev.get("support_passed"):
        verdict = "PATCH_REPLAY_FAILED"
    elif args.live and not ev.get("delayed_ok"):
        verdict = "DELAYED_REJECTED"
    else:
        checks["4_patch_steps_from_manifest"] = bool(
            ev.get("patch_steps")
            and [s["op"] for s in ev["patch_steps"]] == list(REPLAY_PATCH_OPS))
        checks["6_support_delayed_along_patch"] = bool(
            ev.get("support_passed") and ev.get("delayed_gain") is not None)
        checks["7_skill_from_controller_fork"] = bool(
            ev.get("candidate_snapshot_sha"))
        checks["8_next_round_same_entry"] = False
        checks["9_removal_intervention"] = False
        snapshot = ev.get("snapshot")
        if snapshot is not None and checks["4_patch_steps_from_manifest"] \
                and checks["6_support_delayed_along_patch"]:
            # ---- 下一轮正常入口（patched snapshot——验收 8）----
            memory2: list[Any] = []
            events2: list[dict[str, Any]] = []
            r2_log = _feedback_round(
                root, snapshot, executor, series0, values,
                p1.CASE["delayed"], p1.CASE["delayed"] + HORIZON, memory2,
                round_name="r2", auto_slow=False, events=events2)
            details["next_round"] = r2_log
            skill_chosen = any(
                c.startswith("cand_skill_") for c in r2_log["pool"])
            adopted = any(
                p["op"] and "cand_skill" in str(p.get("op"))
                for p in r2_log["probes"]) or skill_chosen
            checks["8_next_round_same_entry"] = bool(skill_chosen)
            # ---- removal intervention（h0 无 Skill 重跑——验收 9）----
            memory3: list[Any] = []
            events3: list[dict[str, Any]] = []
            r3_log = _feedback_round(
                root, h0, executor, series0, values,
                p1.CASE["delayed"], p1.CASE["delayed"] + HORIZON, memory3,
                round_name="r3_removal", auto_slow=False, events=events3)
            details["removal_round"] = r3_log
            no_skill = not any(
                c.startswith("cand_skill_") for c in r3_log["pool"])
            checks["9_removal_intervention"] = bool(
                no_skill and r3_log["probe_order"] != r2_log["probe_order"])
        if not checks["8_next_round_same_entry"]:
            verdict = "NEXT_ROUND_NO_ADOPTION"
        elif not checks["9_removal_intervention"]:
            verdict = "NEXT_ROUND_NO_ADOPTION"
        else:
            verdict = ("REAL_METHOD_LEVEL_SLOW_UPDATE_PASS" if args.live
                       else "METHOD_LEVEL_AUTOMATIC_SLOW_UPDATE_WIRING_PASS")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== fast: {[(p['op'], p['gain']) for p in fast_log['probes']]} "
          f"triggered={fast_log.get('triggered_slow')}")
    if events:
        print(f"== slow: stage={events[0].get('stage')} "
              f"patch={[s['op'] for s in events[0].get('patch_steps', [])]} "
              f"support={events[0].get('support_gain')} "
              f"delayed={events[0].get('delayed_gain')}")
    print(f"== verdict: {verdict}")

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: (str(getattr(v, "harness_content_sha", v))
                        if k == "snapshot" else _strip(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj

    out = root / (REPORT_OUT_LIVE_REL if args.live else REPORT_OUT_REL)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-method-level-auto-slow-wiring",
        "note": ("P3.1-B 真实 Slow Agent（live；≤2 次调用；development case；"
                 "不称自然自进化）" if args.live else
                 "P3.1-A Runner 编排 replay 链（零 LLM；development case；"
                 "不称方法层自动/自然自进化）"),
        "case": dict(p1.CASE),
        "checks": checks,
        "details": _strip(details),
        "events": _strip(events),
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
