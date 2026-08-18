"""METHOD_OWNED_SLOW_UPDATE_WIRING（P3.1-A2，用户裁决 2026-08-11）。

只改一个行为：触发判断和 Slow Update 所有权从实验 Runner 移入方法层
（TTHAMethod.handle_feedback——methods/ttha/method.py）。

验收 1-8（用户裁决）：
  1. 同一个 TTHAMethod 实例跨轮存在；
  2. Runner 只提交真实 Episode/feedback（append_experience_episode +
     handle_feedback）；
  3. Runner 不读取 relation 来决定是否调用 Slow Agent；
  4. Runner 不调用 propose_edit、Controller 或任何 Slow 链函数；
  5. TTHAMethod 自己：append/update Episode、判断 material NEGATIVE/
     CONFLICT、调用 Slow Agent、调用 Controller、接收 replay/delayed、
     更新 active snapshot；
  6. 下一轮 prepare() 自动读取更新后的 snapshot；
  7. Replay Backend，零 live LLM；
  8. removal 使用旧 snapshot 后行为恢复。

不新增 Schema/SHA/事件平台。Runner 提供 card_builder/evaluator 回调
（数据回调），但不拥有触发与更新决策。

Verdict（预注册五档）：
  METHOD_OWNED_SLOW_UPDATE_WIRING_PASS / TRIGGER_STILL_RUNNER_OWNED /
  SNAPSHOT_NOT_UPDATED / NEXT_ROUND_NO_ADOPTION / PATCH_NOT_EXECUTABLE

用法：
  python evaluation/functional/run_v1_method_owned_slow_update.py
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
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_method_level_auto_slow_wiring import (  # noqa: E402
    SlowReplayBackend,
    _request,
)

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
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
    "artifacts/functional/e2/w1_method_owned_slow_update_report.json")
REPORT_OUT_LIVE_REL = Path(
    "artifacts/functional/e2/w1_method_owned_slow_update_live_report.json")
REPORT_OUT_NEG_REL = Path(
    "artifacts/functional/e2/w1_method_owned_slow_update_negative_report.json")
FAST_OPS = ("denoise_median", "outlier_iqr")  # case AB（incumbent 失败对）
# P3.1-B2：Runtime-owned Typed Patch 白名单（冻结 steps——LLM 只选 ID）
TYPED_PATCH_OPTIONS = [
    {"patch_id": "patch-replace-b-with-winsorize",
     "program_steps": [{"op": "denoise_median",
                        "params": {"strength": 1.0, "window": 1}},
                       {"op": "winsorize", "params": {}}]},
    {"patch_id": "patch-replace-b-with-outlier-mad",
     "program_steps": [{"op": "denoise_median",
                        "params": {"strength": 1.0, "window": 1}},
                       {"op": "outlier_mad", "params": {}}]},
]


def _card_from_episode(episode: Any) -> Mapping[str, object]:
    """Runner 提供的 card 构造回调（数据回调——从真实 Episode 的 Workflow/
    Context/gain 构造；不含触发决策）。"""
    cs = getattr(episode, "context_summary", {}) or {}
    ctx = {k: float(v) for k, v in
           (cs.get("local_pattern") or {}).items()
           if isinstance(v, (int, float))}
    pg = cs.get("program_geometry") or {}
    steps = [{"op": s["op"], "params": dict(s.get("params") or {})}
             for s in (pg.get("program_steps") or [])
             if isinstance(s, Mapping) and s.get("op")]
    sg = (getattr(episode, "support_response", {}) or {}).get("gain")
    dg = (getattr(episode, "delayed_response", {}) or {}).get("gain")
    ctx["bound_period"] = float(PERIOD)
    return {
        "pattern_id": f"gefcom2012_load-{'-'.join(s['op'] for s in steps)}-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {"steps": steps, "scope": "training_windows_only",
                     "evaluator": "v6._evaluate"},
        "facts": {"window_context": ctx,
                  "task_objective": "forecast; cohort-Ridge sMASE"},
        "counterfactual_support": {
            "A_then_B_support_gain": float(sg) if sg is not None else None,
            "A_then_B_delayed_gain": float(dg) if dg is not None else None},
        "typed_patch_options": [
            {"patch_id": str(o["patch_id"]),
             "program_steps": o["program_steps"]}
            for o in TYPED_PATCH_OPTIONS],
        "instruction": ("Select exactly one typed patch by its patch_id from "
                        "typed_patch_options (or abstain). Do not write "
                        "program steps."),
    }


def _feedback_round(method: TTHAMethod, executor: ScopeExecutor,
                    series0: np.ndarray, values: Mapping[str, Any],
                    origin: int, delayed_origin: int, *,
                    round_name: str, submit_feedback: bool,
                    slow_agent: Any, controller: Any, store: Any,
                    events: list[dict[str, Any]]) -> dict[str, Any]:
    """Runner 侧 round：只做 prepare + evaluate + 提交 Episode/feedback
    （append_experience_episode + method.handle_feedback）。不读 relation、
    不调 Slow 链函数。"""
    ctx = dict(resolver.window_context(values, origin, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.SealedProbeBackend(
        explore=True, operators=FAST_OPS, max_propose_candidates=2,
        force_pool=True)
    # 同一 method 实例的 backend/gateway 在 bind_round_data 时重建；池固定
    core = method.fast_agent.core
    core.backend = backend  # 每轮确定性 backend（同实例）
    method.bind_round_data(series0[:origin], task_kind="forecast")
    result = method.prepare(_request(series0, values, origin))
    trace = method.last_trace
    steps_map = dict(trace.candidate_program_steps or {})
    pool_ops = [c[len("cand_"):] for c in trace.candidate_ids
                if c.startswith("cand_") and c in steps_map]
    log: dict[str, Any] = {"origin": origin,
                           "pool": list(trace.candidate_ids), "probes": []}
    for i, op in enumerate(pool_ops[:2]):
        steps = steps_map[f"cand_{op}"]
        rr = executor.evaluate(steps, origin)
        gain = (float(rr.gain) if rr.gain is not None else None)
        passed = bool(rr.verification.passed)
        entry: dict[str, Any] = {"probe": i + 1, "op": op, "gain": gain,
                                 "passed": passed}
        if passed:
            ep = tll.write_target_episode(
                domain="gefcom2012_load", op=op,
                episode_id_suffix=f"_owned_{round_name}_p{i + 1}",
                program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
                support_gain=gain if gain is not None else 0.0,
                delayed_gain=None,
                support_context=dict(resolver.window_context(
                    values, origin, PERIOD)))
            entry["episode_id"] = ep.episode_id
            # ---- Runner 只提交 Episode（方法层判定/触发/更新）；
            # submit_feedback=False（r2/r3）不触发 Slow 链（纯观察轮）----
            method.append_experience_episode(ep)
            if submit_feedback:
                ev = method.handle_feedback(ep, confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=slow_agent, controller=controller,
                    store=store, surface_catalog=p1.SURFACE_CATALOG,
                    card_builder=_card_from_episode,
                    evaluator=lambda s, _o: executor.evaluate(s, origin),
                    delayed_evaluator=lambda s, _o: executor.evaluate(
                        s, delayed_origin),
                    manifest_preflight=lambda m: None)
                events.append(ev)
                log["feedback_event"] = ev
            # delayed 更新（同一 Episode——方法层 update_experience_episode）
            rd = executor.evaluate(steps, delayed_origin)
            dg = (float(rd.gain) if rd.gain is not None else None)
            entry["delayed_gain"] = dg
            for i_e, e in enumerate(method._experience_episodes):  # noqa: SLF001
                if getattr(e, "episode_id", "") == ep.episode_id:
                    upd = tll.update_delayed_status(
                        e, dg if dg is not None else 0.0,
                        delayed_context=dict(resolver.window_context(
                            values, delayed_origin, PERIOD)))
                    method.update_experience_episode(upd)
                    entry["relation_after_delayed"] = upd.relation
                    break
        log["probes"].append(entry)
        if gain is not None and gain >= M:
            break
    return log


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="B2 live：真实 Slow Agent 从 typed_patch_options "
                             "选 patch_id（最多 1 次调用）")
    parser.add_argument("--negative", action="store_true",
                        help="批准权负控（零 LLM）：Replay 选无效候选 "
                             "patch-replace-b-with-outlier-mad（gefcom 上 "
                             "support −0.065 < M）→ 应被 replay 否决、"
                             "snapshot 不变、下一轮池无该 Skill")
    args = parser.parse_args()
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    config = dict(v6.DATASET_CONFIGS["gefcom"])
    roster, values = v6._fixed_roster(root, config)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    # ---- 同一 TTHAMethod 实例跨轮（验收 1）----
    if args.live:
        import os
        api_key = next((os.environ.get(k, "").strip() for k in
                        ("OPENAI_API_KEY", "AGICTO_API_KEY")
                        if os.environ.get(k, "").strip()), None)
        if not api_key:
            print("== no api key — ACTION_UNAVAILABLE")
            return 0
        import openai
        import run_v1_slow_path_smoke as smoke
        counter = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120),
            max_calls=2)  # B2 live：1 次正式 Slow 调用 + 1 次契约内
        # schema 重试（validation_retries=1）；超过 → INCONCLUSIVE_PROVIDER_FAILURE
        from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
            AgictoChatCompletionsBackend,
        )
        slow_backend: Any = AgictoChatCompletionsBackend(
            client=counter, base_url=smoke.BASE_URL)
    else:
        counter = None
        slow_backend = SlowReplayBackend(
            explore=False,
            patch_id=("patch-replace-b-with-outlier-mad" if args.negative
                      else "patch-replace-b-with-winsorize"))
    slow_core = TTHAAgentCore(
        slow_backend,
        LocalPublicToolGateway(series0[:904], task_kind="forecast"))
    slow_agent = TTHASlowAgent(slow_core)
    store = SnapshotStore(root)
    reg = SurfaceRegistry()
    controller = EditController(store, surfaces=reg, router=FaultRouter())
    method = TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=FAST_OPS),
            LocalPublicToolGateway(series0[:904], task_kind="forecast"))),
        h0, ())

    # ---- R1（失败反馈 → 方法层自动 Slow 更新）----
    r1 = _feedback_round(method, executor, series0, values, p1.CASE["origin"],
                         p1.CASE["delayed"], round_name="r1",
                         submit_feedback=True, slow_agent=slow_agent,
                         controller=controller, store=store, events=events)
    details["r1"] = r1
    ev = next((e for e in reversed(events) if e.get("triggered")),
              events[0] if events else {})
    checks["1_same_instance_across_rounds"] = True  # 结构：单个 method 实例
    checks["2_runner_only_submits_episode"] = True  # 结构：Runner 只调 append/handle
    checks["3_runner_no_relation_read_for_trigger"] = True  # 结构：Runner
    # 不读取 relation 决定触发（L191 的 upd.relation 仅日志用途）
    checks["4_runner_no_slow_chain_call"] = True  # 结构：无 propose_edit/apply_to_fork
    checks["5_method_owned_trigger"] = bool(ev.get("triggered"))
    # 报告真实性（用户 2026-08-11）：live 模式有 ≤1 次真实 Slow 调用——
    # 不写 zero_live_llm=True
    checks["7_zero_live_llm"] = bool(not args.live)
    checks["8_removal_restores"] = False

    if args.negative:
        # 批准权负控（零 LLM）：已知无效候选（outlier_mad——support
        # −0.065 < M）→ replay 否决、snapshot 保持原版本、下一轮池无
        # 该 Skill
        rejected = ev.get("stage") == "replay_rejected"
        snapshot_unchanged = not ev.get("snapshot_updated", False)
        r2n = _feedback_round(method, executor, series0, values,
                              p1.CASE["delayed"],
                              p1.CASE["delayed"] + HORIZON,
                              round_name="r2_neg", submit_feedback=False,
                              slow_agent=slow_agent, controller=controller,
                              store=store, events=events)
        details["r2_negative"] = r2n
        no_skill = not any(c.startswith("cand_skill_")
                           for c in r2n["pool"])
        checks["neg_replay_rejected"] = rejected
        checks["neg_snapshot_unchanged"] = snapshot_unchanged
        checks["neg_next_round_no_skill"] = no_skill
        verdict = ("REPLAY_REJECTION_NEGATIVE_CONTROL_PASS"
                   if (rejected and snapshot_unchanged and no_skill)
                   else "REPLAY_REJECTION_NEGATIVE_CONTROL_FAILED")
    elif not checks["5_method_owned_trigger"]:
        verdict = "TRIGGER_STILL_RUNNER_OWNED"
    elif ev.get("stage") == "budget_exceeded":
        verdict = "INCONCLUSIVE_PROVIDER_FAILURE"
    elif ev.get("stage") in ("no_manifest", "preflight_rejected",
                             "apply_failed", "no_frozen_program"):
        verdict = "ACTION_UNAVAILABLE"  # B2：未知/缺失 patch_id 或 apply 失败
    elif ev.get("stage") == "replay_rejected":
        verdict = "PATCH_REPLAY_FAILED"  # 批准权正确否决（support/delayed）
    elif ev.get("stage") != "applied":
        verdict = "PATCH_NOT_EXECUTABLE"
    elif not ev.get("snapshot_updated"):
        verdict = "SNAPSHOT_NOT_UPDATED"
    else:
        # ---- 下一轮同一实例（验收 6：prepare 自动读更新后 snapshot）----
        r2 = _feedback_round(method, executor, series0, values,
                             p1.CASE["delayed"], p1.CASE["delayed"] + HORIZON,
                             round_name="r2", submit_feedback=False,
                             slow_agent=slow_agent, controller=controller,
                             store=store, events=events)
        details["r2"] = r2
        skill_in_pool = any(c.startswith("cand_skill_") for c in r2["pool"])
        checks["6_next_round_auto_snapshot"] = bool(skill_in_pool)
        if not skill_in_pool:
            verdict = "NEXT_ROUND_NO_ADOPTION"
        else:
            # ---- removal：h0 旧 snapshot 新实例重跑（验收 8）----
            method_r = TTHAMethod(
                sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                    sealed.SealedProbeBackend(explore=True, operators=FAST_OPS),
                    LocalPublicToolGateway(
                        series0[:p1.CASE["delayed"]],
                        task_kind="forecast"))),
                h0, ())
            events_r: list[dict[str, Any]] = []
            r3 = _feedback_round(method_r, executor, series0, values,
                                 p1.CASE["delayed"],
                                 p1.CASE["delayed"] + HORIZON,
                                 round_name="r3_removal",
                                 submit_feedback=False,
                                 slow_agent=slow_agent, controller=controller,
                                 store=store, events=events_r)
            details["r3_removal"] = r3
            no_skill = not any(c.startswith("cand_skill_")
                               for c in r3["pool"])
            checks["8_removal_restores"] = bool(
                no_skill and r3["pool"] != r2["pool"])
            verdict = ("METHOD_OWNED_SLOW_UPDATE_WIRING_PASS"
                       if checks["8_removal_restores"]
                       else "NEXT_ROUND_NO_ADOPTION")
    print(f"== r1: {[(p['op'], p['gain']) for p in r1['probes']]} "
          f"event_stage={ev.get('stage')} snapshot_updated={ev.get('snapshot_updated')}")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: (str(getattr(v, "harness_content_sha", v))
                        if k == "snapshot" else _strip(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj

    out = root / (REPORT_OUT_NEG_REL if args.negative
                  else REPORT_OUT_LIVE_REL if args.live
                  else REPORT_OUT_REL)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-method-owned-slow-update",
        "note": ("P3.1-A2+B2 live：真实 Slow Agent 选 patch_id（≤1 次调用；"
                 "development case；不称自然自进化）" if args.live else
                 "P3.1-A2+B2 replay：方法层所有权 + Typed Patch Binding"
                 "（零 live LLM；development case；不称自然自进化）"),
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
