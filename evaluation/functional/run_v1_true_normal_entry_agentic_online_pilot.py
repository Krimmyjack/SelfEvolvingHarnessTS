"""TRUE_NORMAL_ENTRY_AGENTIC_ONLINE_PILOT（任务书 P2，2026-08-10）。

真实正常入口 3 轮在线轨迹（traffic offset=240 新 certified-virgin cohort）：

  轮 1 origin=648 delayed=696
  轮 2 origin=744 delayed=792
  轮 3 origin=840 delayed=888

每轮（同一 Harness/Memory 生命周期——探索状态跨轮继承、Memory 累积
进下一轮 prepare、同一 LLM client）：
  真实 TTHAMethod.prepare → 真实 Fast Agent inspect/propose/select
  （inspect/propose 确定性、select 真实 LLM）→ Agent 自主 Typed Workflow
  → 最多 2 个 Target Support receipts → 每次 Support 后立即写 Episode →
  delayed 到达后更新本轮 Episode → 下一轮同一生命周期。

Runner 不固定两步组合、不直接调用 _combos()、不手工选算子、运行中不
Slow Path 修改。abstain/reject 如实记录。

因果检查（keep vs remove last update）：本轮正式 prepare 之前，用
memory[:-1] + 相同探索状态/Context/预算/模型做一次只观察的对照 prepare；
比较 pool/chosen/abstention 是否回退。真实 LLM 不稳定（temp=0 下
同输入不同输出）无法归因时 → INCONCLUSIVE_LLM_VARIANCE，不靠投票。

checkpoint 恢复（后台任务 ~5-8 分钟被杀）：每轮结束写
artifacts/functional/e2/p2_online_checkpoint.json；被杀后
--resume 从断点继续（memory/探索状态/轮日志全恢复）。

轨迹结束后定位第一个自然 fault（执行 gain < −MATERIAL）→ 反事实 →
headroom 三条件记录（P3 触发判定，P2 不调用 Slow 链）。

Verdict（预注册）：
  NORMAL_ONLINE_ADAPTATION_PASS / MEMORY_WRITTEN_NOT_USED /
  AGENT_PROGRAM_SUPPLY_STALLED / NO_ACTIONABLE_FEEDBACK /
  NEGATIVE_TRANSFER / INCONCLUSIVE_LLM_VARIANCE / NO_ELIGIBLE_VIRGIN_COHORT

用法：
  python evaluation/functional/run_v1_true_normal_entry_agentic_online_pilot.py [--resume]
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import ExperienceEpisode  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

DOMAIN = "monash:traffic_hourly"
OFFSET = 240  # 下一组 virgin（offset=40 实验2、120 Pilot/hybrid 已消费）
PERIOD = 24
HORIZON = 48
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005
ROUNDS = [(648, 696), (744, 792), (840, 888)]
BUDGET = 2  # 每轮最多 2 次 prepare（→ ≤2 Support receipts）
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_true_normal_entry_agentic_online_pilot_report.json")
CHECKPOINT_REL = Path("artifacts/functional/e2/p2_online_checkpoint.json")

OPS_ALL = tuple(o for o in (
    "denoise_median", "hampel_filter", "impute_ar", "impute_ema",
    "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
    "outlier_mad", "period_complete", "period_median_complete",
    "repair_level_shift", "resample_uniform", "winsorize"))


def _episode_to_dict(ep: Any) -> dict[str, Any]:
    return dataclasses.asdict(ep)


def _episode_from_dict(d: Mapping[str, Any]) -> ExperienceEpisode:
    return ExperienceEpisode(**{k: v for k, v in d.items()
                                if k in ExperienceEpisode.__dataclass_fields__})


def _make_method(backend: Any, snapshot: Any, memory: Sequence[Any],
                 series0: np.ndarray, origin: int) -> TTHAMethod:
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:origin], task_kind="forecast"))),
        snapshot,
        tuple(memory))
    method.bind_round_data(series0[:origin], task_kind="forecast")
    return method


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoint 恢复（被杀后续跑）")
    parser.add_argument("--offset", type=int, default=OFFSET,
                        help="virgin cohort 偏移（P2=240；P2-V2=360）")
    parser.add_argument("--domain", default=DOMAIN,
                        help="dataset_id（如 uci_electricity_load_diagrams）")
    parser.add_argument("--report-suffix", default="",
                        help="报告文件名后缀（如 _v2 避免覆盖 replay 报告）")
    args = parser.parse_args()
    offset = args.offset
    domain = args.domain

    root = PROJECT_ROOT
    sealed._set_domain(domain)
    config = sealed._config()
    try:
        (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(
            root, offset=offset)
    except AssertionError as exc:
        print(f"== NO_ELIGIBLE_VIRGIN_COHORT: {exc}")
        return 0
    tgt_series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=sealed.v6._evaluate)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    # 预注册可行动 Context 检查（CONTEXT_ACTIONABLE_NATURAL_OPERATION_PILOT，
    # 2026-08-10）：只按部署可见 Context（window_context，不读 gain）确认
    # 至少一个决策点满足 Operator 前提——防止误选干净 cohort。
    _actionable_points = []
    for _o, _ in ROUNDS:
        _ctx = dict(resolver.window_context(tgt_values, _o, PERIOD))
        _cd = _ctx.get("change.median_robust_center_delta")
        _sr = _ctx.get("recent.median_normalized_seasonal_residual")
        _cov = _ctx.get("recent.coverage", 1.0)
        _level_shift = _cd is not None and abs(float(_cd)) > 1.0
        _outlier = _sr is not None and float(_sr) > 1.5
        _missing = float(_cov) < 1.0
        _actionable_points.append(
            {"origin": _o, "level_shift": _level_shift,
             "outlier_signal": _outlier, "missing": _missing,
             "center_delta": _cd, "seasonal_residual": _sr,
             "coverage": _cov})
    actionable_cohort = any(
        p["level_shift"] or p["outlier_signal"] or p["missing"]
        for p in _actionable_points)
    print(f"== pre-registered actionable context: "
          f"{'OK' if actionable_cohort else 'NONE'} "
          f"{json.dumps(_actionable_points, indent=1)}")

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=40)

    # ---- checkpoint 恢复（按 cohort offset 区分文件）----
    ckpt_path = root / CHECKPOINT_REL.with_name(
        f"{CHECKPOINT_REL.stem}_o{offset}{CHECKPOINT_REL.suffix}")
    memory: list[Any] = []
    shared_explored: list[str] = []
    shared_deprioritized: list[str] = []
    rounds_log: list[dict[str, Any]] = []
    prev_round_state: dict[str, Any] | None = None
    start_round = 0
    if args.resume and ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        memory = [_episode_from_dict(e) for e in ck.get("memory", [])]
        shared_explored = list(ck.get("explored", []))
        shared_deprioritized = list(ck.get("deprioritized", []))
        rounds_log = list(ck.get("rounds_log", []))
        prev_round_state = ck.get("prev_round_state")
        start_round = int(ck.get("rounds_done", 0))
        print(f"== resume: {start_round} rounds done, "
              f"{len(memory)} episodes, explored={len(shared_explored)}")

    def _checkpoint(rounds_done: int) -> None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(json.dumps({
            "rounds_done": rounds_done,
            "memory": [_episode_to_dict(e) for e in memory],
            "explored": shared_explored,
            "deprioritized": shared_deprioritized,
            "rounds_log": rounds_log,
            "prev_round_state": prev_round_state,
        }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    first_fault: dict[str, Any] | None = None
    behavior_changed: bool = False
    remove_ctrl_flipped: bool = False

    for round_i in range(start_round, len(ROUNDS)):
        origin, delayed_origin = ROUNDS[round_i]
        r_idx = round_i + 1  # 1-based
        ctx = dict(resolver.window_context(tgt_values, origin, PERIOD))
        ctx["bound_period"] = float(PERIOD)
        round_log: dict[str, Any] = {"round": r_idx, "origin": origin,
                                     "delayed_origin": delayed_origin,
                                     "attempts": []}

        # ---- remove-last 对照（memory 有上轮新写入时；只观察不执行）----
        if prev_round_state is not None and memory:
            ctrl_backend = sealed.LLMSelectBackend(
                explore=True, operators=OPS_ALL, client=counter,
                context_plain=dict(ctx))
            ctrl_backend._explored = list(shared_explored)
            ctrl_backend._deprioritized = list(shared_deprioritized)
            ctrl_backend._pending_op = None
            ctrl_method = _make_method(ctrl_backend, h0, memory[:-1],
                                       tgt_series0, origin)
            ctrl_req = sealed._request(tgt_series0, tgt_values, origin)
            try:
                ctrl_method.prepare(ctrl_req)
            except Exception as exc:
                round_log["ctrl_exception"] = f"{type(exc).__name__}: {exc}"
            else:
                ctrl = ctrl_method.last_trace
                ctrl_chosen = ctrl.chosen_candidate_id
                ctrl_pool = list(ctrl.candidate_ids)
                round_log["ctrl_remove_last"] = {
                    "pool": ctrl_pool,
                    "chosen": ctrl_chosen,
                    "abstained": ctrl_chosen is None}
                # 归因审查：flip 只在 ctrl 有候选时算数——无候选的 identity
                # 是机制空转（探索卡在 no-op 过滤），不是 memory 效应
                if (prev_round_state.get("chosen") != ctrl_chosen
                        and len(ctrl_pool) > 1):
                    remove_ctrl_flipped = True
                    round_log["ctrl_remove_last"]["flip_vs_prev"] = True

        # ---- 正式轮次（≤2 次 prepare 行动）----
        backend = sealed.LLMSelectBackend(
            explore=True, operators=OPS_ALL, client=counter,
            context_plain=dict(ctx))
        backend._explored = list(shared_explored)
        backend._deprioritized = list(shared_deprioritized)
        backend._pending_op = None
        receipts = 0
        for attempt in range(BUDGET):
            method = _make_method(backend, h0, memory, tgt_series0, origin)
            req = sealed._request(tgt_series0, tgt_values, origin)
            try:
                result = method.prepare(req)
            except Exception as exc:
                round_log["attempts"].append(
                    {"attempt": attempt + 1, "kind": "exception",
                     "detail": f"{type(exc).__name__}: {exc}"})
                break
            trace = method.last_trace
            chosen = trace.chosen_candidate_id
            entry: dict[str, Any] = {
                "attempt": attempt + 1,
                "pool": list(trace.candidate_ids),
                "chosen": chosen,
                "program": None,
                "verifier_passed": None,
                "gain": None,
                "episode_id": None,
                "relation": None,
            }
            # LLM select 决策记录（rationale 供审查核实 abstain 是否合理）
            if getattr(backend, "_select_logs", None):
                last = backend._select_logs[-1]
                entry["llm_rationale"] = (last.get("raw") or "")[:300]
            if chosen is None or result.program is None:
                entry["kind"] = "abstain"
                round_log["attempts"].append(entry)
                continue  # abstain 不消耗 Support receipt；预算内可再试
            steps = tuple(result.program.execution_steps())
            entry["program"] = [{"op": op, "params": dict(pr)}
                                for op, pr in steps]
            rr = executor.evaluate(steps, origin)
            entry["verifier_passed"] = bool(rr.verification.passed)
            gain = (float(rr.gain) if rr.gain is not None else None)
            entry["gain"] = gain
            if not rr.verification.passed:
                entry["kind"] = "reject"
                round_log["attempts"].append(entry)
                continue
            entry["kind"] = "support"
            receipts += 1
            if gain is not None and gain < -MATERIAL:
                entry["harm"] = True
            # ---- 立即写 Episode（正常在线语义：passed 行动写回）----
            ep = tll.write_target_episode(
                domain=DOMAIN, op=str(chosen),
                episode_id_suffix=f"_pilot2_r{r_idx}a{attempt + 1}",
                program_steps=[{"op": op, "params": dict(pr)}
                               for op, pr in steps],
                support_gain=gain if gain is not None else 0.0,
                delayed_gain=None,
                support_context=dict(resolver.window_context(
                    tgt_values, origin, PERIOD)))
            memory.append(ep)
            entry["episode_id"] = ep.episode_id
            entry["relation"] = ep.relation
            # ---- delayed 打开（同一冻结 Workflow 的后续窗口；原位更新）----
            rd = executor.evaluate(steps, origin + HORIZON)
            gain_d = (float(rd.gain) if rd.gain is not None else None)
            entry["delayed_gain"] = gain_d
            for i_e, e in enumerate(memory):
                if getattr(e, "episode_id", "") == ep.episode_id:
                    memory[i_e] = tll.update_delayed_status(
                        e, gain_d if gain_d is not None else 0.0,
                        delayed_context=dict(resolver.window_context(
                            tgt_values, origin + HORIZON, PERIOD)))
                    entry["relation"] = memory[i_e].relation
                    break
            # ---- 第一个自然 fault（执行 gain < −M；记录不中断）----
            if first_fault is None and gain is not None and gain < -MATERIAL:
                first_fault = {"round": r_idx, "origin": origin,
                               "op": str(chosen), "gain": gain,
                               "steps": steps,
                               "delayed_gain": gain_d,
                               "episode_id": ep.episode_id}
            round_log["attempts"].append(entry)
            if receipts >= 2:
                break

        shared_explored = list(backend._explored)
        shared_deprioritized = list(backend._deprioritized)
        round_log["receipt_count"] = receipts
        round_log["memory_after"] = [
            {"episode_id": getattr(e, "episode_id", "?"),
             "relation": getattr(e, "relation", "?")} for e in memory]
        rounds_log.append(round_log)
        _checkpoint(round_i + 1)  # 每轮落盘（被杀可恢复）

        # ---- 跨轮行为变化检测（写回前状态 = 上一轮 chosen/pool）----
        cur_state = {"chosen": round_log["attempts"][-1].get("chosen")
                     if round_log["attempts"] else None,
                     "pool": round_log["attempts"][-1].get("pool")
                     if round_log["attempts"] else None}
        if prev_round_state is not None and prev_round_state != cur_state:
            behavior_changed = True
        prev_round_state = cur_state
        print(f"== round {r_idx} @{origin}: "
              f"{[(a.get('kind'), a.get('chosen'), round(a['gain'], 4) if a.get('gain') is not None else None) for a in round_log['attempts']]} "
              f"ctrl_flip={round_log.get('ctrl_remove_last', {}).get('flip_vs_prev')}",
              flush=True)

    if len(rounds_log) < len(ROUNDS):
        print(f"== incomplete ({len(rounds_log)}/{len(ROUNDS)} rounds) — "
              f"run with --resume to continue")
        return 0

    # ---- 轨迹结束：first fault 反事实（P3 触发判定，不调用 Slow 链）----
    fault_checks: dict[str, Any] = {}
    if first_fault is not None:
        op = str(first_fault["op"])
        fault_checks["op"] = op
        fault_checks["gain_action"] = first_fault["gain"]
        others: dict[str, float] = {}
        for o in OPS_ALL:
            if o == op:
                continue
            r = executor.evaluate(((o, {}),), first_fault["origin"])
            g = (float(r.gain) if r.gain is not None else None)
            if g is not None:
                others[o] = g
        best = max(others.items(), key=lambda kv: kv[1])
        fault_checks["best_other"] = best
        headroom = (best[1] >= MATERIAL
                    and (best[1] - first_fault["gain"]) >= MATERIAL)
        fault_checks["headroom"] = bool(headroom)
        fault_checks["b_single_op_negative"] = bool(
            first_fault["gain"] < -MATERIAL and best[1] < MATERIAL)
        print(f"== first fault: {op} gain={first_fault['gain']} "
              f"best_other={best} headroom={headroom}")

    # ---- verdict 判定（预注册 + 归因审查修正）----
    # 归因修正 1（ctrl）：remove-last flip 仅在 ctrl 有候选时算数（见上）。
    # 归因修正 2（harm）：NEGATIVE_TRANSFER 需要 harm 发生在 Memory 影响
    # 之后（transfer 语义）——轮 1 探测 harm（memory 写入前/同轮早期探测）
    # 是预算内探测风险，不是 transfer。
    all_abstained = all(
        a.get("kind") in ("abstain",) or a.get("chosen") is None
        for rd in rounds_log for a in rd["attempts"]
        if a.get("kind") != "exception")
    any_action = any(a.get("kind") == "support" for rd in rounds_log
                     for a in rd["attempts"])
    material_any = any(
        (a.get("gain") is not None and abs(a["gain"]) >= MATERIAL)
        for rd in rounds_log for a in rd["attempts"])
    harm_after_memory_round = any(
        a.get("harm") and rd["round"] >= 2
        for rd in rounds_log for a in rd["attempts"])
    memory_written = bool(memory)

    if not any_action or not material_any:
        verdict = "AGENT_PROGRAM_SUPPLY_STALLED" if all_abstained \
            else "NO_ACTIONABLE_FEEDBACK"
    elif behavior_changed and remove_ctrl_flipped:
        verdict = ("NEGATIVE_TRANSFER" if harm_after_memory_round
                   else "INCONCLUSIVE_LLM_VARIANCE")
    elif behavior_changed and not remove_ctrl_flipped:
        verdict = "INCONCLUSIVE_LLM_VARIANCE"
    else:
        verdict = "MEMORY_WRITTEN_NOT_USED" if memory_written \
            else "NO_ACTIONABLE_FEEDBACK"

    print(f"== behavior_changed={behavior_changed} "
          f"ctrl_flip={remove_ctrl_flipped} "
          f"harm_after_memory={harm_after_memory_round} "
          f"memory={len(memory)}")
    print(f"== verdict: {verdict}")

    # supply 状态（用户裁决 P2-V2 解释规则）：每轮 pool 是否有合法候选
    # （供应正常 vs 空转）；全 abstain + supply_ok → Agent 安全选择而非
    # supply stall
    supply_ok = all(
        any(c.startswith("cand_") for c in (a.get("pool") or []))
        for rd in rounds_log for a in rd["attempts"])
    if not any_action and all_abstained and supply_ok:
        interpretation = ("agent_safe_abstain（supply 正常、LLM 基于 "
                          "Context 稳定 abstain——非 supply stall）")
    elif not any_action and all_abstained and not supply_ok:
        interpretation = "supply_stall（供应空池导致 abstain）"
    elif behavior_changed and remove_ctrl_flipped:
        interpretation = "memory_attributed_behavior_change"
    elif behavior_changed:
        interpretation = ("llm_variance_or_mechanism（无法归因——"
                          "LLM 不稳定）")
    else:
        interpretation = "no_behavior_change"

    out = root / REPORT_OUT_REL.with_name(
        f"{REPORT_OUT_REL.stem}{args.report_suffix}{REPORT_OUT_REL.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-true-normal-entry-agentic-online-pilot"
                         f"{args.report_suffix}",
        "dataset": domain, "cohort_offset": offset,
        "actionable_context_check": _actionable_points,
        "rounds": ROUNDS, "budget_per_round": BUDGET,
        "rounds_log": rounds_log,
        "first_fault": (None if first_fault is None else
                        {k: v for k, v in first_fault.items()
                         if k != "steps"}),
        "fault_checks": fault_checks,
        "behavior_changed": behavior_changed,
        "remove_ctrl_flipped": remove_ctrl_flipped,
        "harm_after_memory_round": harm_after_memory_round,
        "supply_ok": supply_ok,
        "interpretation": interpretation,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    if ckpt_path.exists():
        ckpt_path.unlink()  # 终局后清理 checkpoint
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
