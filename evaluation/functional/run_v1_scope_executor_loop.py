"""V1 纵向切片：ScopeExecutor 接入真实方法链（零 LLM，2026-08-08）。

审查裁决（十二）：机械正控 PASS 记为 MECHANISM PASS；纵向切片只做一件事：
消费 result.program → 最小 scope executor → 同一组件完成 verifier/Support/
写回/delayed → 不重叠后续 origin 跑 A5/A3 同预算比较。

审查裁决（十三）：A5_WORSE 比较无效（4 个 P0），本次修复 Runner 控制语义，
不调 radius、不跑 LLM：
  P0-1 A5/A3 并非只有 Memory 不同 → **两臂用同一 explore=True Agent，
        唯一初始差异 = Source Memory**（A5=[seed]，A3=[]）；
  P0-2 Support 没有立即写回 → **每次 Support receipt 后立即写入该臂 Memory，
        再进行下一次 probe**（Action → Support → 写 Episode → 下一次 probe
        读更新 Memory）；
  P0-3 历史种子 Episode 被 delayed 覆盖 → **delayed 只更新本轮新建 Episode，
        不修改 seed**；Episode ID 加 origin 后缀（gefcom_target_winsorize_origin928），
        避免同算子跨轮覆盖（普通后缀，不需要 Hash）；
  P0-4 Round 2 不是时间上的下一轮 → **delayed 后在 928 的动作标注为
        counterfactual replay**（真实纵向结论需 1024 之后空间的链）。

公平重跑判定（裁决原文）：如果 A5 仍多一次 harm，才能确认 Source prior
发生负迁移（候选）；此时也不应事后调 q75，而应调查 winsorize 的
Program-specific Observation。outlier_iqr @928/976 已提供替代 Program headroom。

用法：
  python evaluation/functional/run_v1_scope_executor_loop.py
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
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

HORIZON = 48
PERIOD = 24
TARGET_DOMAIN = "gefcom"
MAX_TARGET_PROBES = 2  # 同预算（与跨域闭环一致）
SLICE_SUPPORT = 928      # 不重叠后续 origin（种子切片 832/880 之后）
SLICE_DELAYED = 976
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_scope_executor_loop_report.json")
MATERIAL = core.MATERIAL_THRESHOLD
SEED_REPORT = Path("artifacts/functional/e2/w1_scope_alignment_report.json")


def build_seed_memory(
    root: Path,
    values: Mapping[str, Any],
    config: Mapping[str, object],
) -> list[Any]:
    """种子 Memory：已验证暴露正控（winsorize @832→880 双正）——从
    scope_alignment 报告读取数值构造 Episode（不重跑、不伪造）。ID 带
    origin 后缀（历史切片，与后续 928 写回区分）。"""
    from run_v1_target_local_loop import update_delayed_status, write_target_episode

    report = json.loads((root / SEED_REPORT).read_text(encoding="utf-8"))
    control = report["part_c_closed_loop"]["control"]
    op = str(control["operator"])
    origin = int(control["origin"])
    assert op == "winsorize" and origin == 832, "种子切片与预期不符"
    params = wiring.contract_params(op, PERIOD)
    ep = write_target_episode(
        domain=TARGET_DOMAIN, op=op,
        episode_id_suffix=f"_origin{origin}",
        program_steps=[{"op": op, "params": dict(params)}],
        support_gain=float(control["support_gain"]), delayed_gain=None,
        support_context=resolver.window_context(values, origin, PERIOD))
    ep = update_delayed_status(
        ep, float(control["delayed_gain"]),
        delayed_context=resolver.window_context(values, origin + HORIZON, PERIOD))
    print(f"== seed episode: {op} @{origin} support={control['support_gain']:.5f} "
          f"delayed={control['delayed_gain']:.5f} "
          f"status={ep.local_status} relation={ep.relation} id={ep.episode_id}")
    return [ep]


def probe_arm(
    root: Path,
    executor: ScopeExecutor,
    values: Mapping[str, Any],
    config: Mapping[str, object],
    origin: int,
    memory: Sequence[Any],
    *,
    explore_operators: Sequence[str],
    domain: str = TARGET_DOMAIN,
    period: int = PERIOD,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any]]:
    """同预算探测（公平控制语义，审查裁决 十三 P0-1/P0-2）：

    - **同一 explore=True Agent**（两臂同策略，唯一初始差异 = 传入的 Memory）；
    - **每次 Support receipt 后立即写回该臂 Memory**，再进行下一次 probe
      （Action → Support → 写 Episode → 下一次 prepare 读更新 Memory）；
    - stop-on-first-positive；预算 MAX_TARGET_PROBES；
    - Episode ID 带 origin 后缀（P0-3：同算子跨轮不覆盖）。
    返回 (probes, retrieval, 更新后的该臂 Memory)。"""
    from run_v1_target_local_loop import write_target_episode

    series = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    observed = dict(resolver.window_context(values, origin, period))
    observed["bound_period"] = float(period)
    backend = wiring.DeterministicStrategyBackend(
        explore=True, operators=tuple(explore_operators))
    probes: list[dict[str, Any]] = []
    retrieval: list[dict[str, Any]] = []
    arm_memory: list[Any] = list(memory)  # 该臂 Memory（探测间立即更新）
    for _ in range(MAX_TARGET_PROBES):
        r_values = series[:origin]
        request = PreparationRequest(
            "scope-executor-loop",
            r_values,
            forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                                  metric=MetricSpec("sMASE", "lower_is_better")),
            dict(observed),
        )
        core_agent = TTHAAgentCore(
            backend, LocalPublicToolGateway(r_values, task_kind="forecast"))
        result, trace = TTHAFastAgent(core_agent).prepare(
            request, h0, experience_episodes=tuple(arm_memory))
        chosen = trace.chosen_candidate_id
        retrieval.append({
            "probe_index": len(probes),
            "chosen": chosen,
            "compilation": trace.compilation_status,
            "retrieved_memory_ids": list(trace.retrieved_memory_ids),
            "reference1": "Reference 1" in (
                wiring.DeterministicStrategyBackend.extract_instruction(
                    backend.requests[-1].messages) if backend.requests else ""),
        })
        if chosen == "identity" or chosen not in trace.candidate_program_steps:
            break  # abstain（无 Reference 且探索尽 / 无候选）
        if any(p["chosen"] == chosen for p in probes):
            break  # 同候选重复探测无信息（防御；立即写回后预期不触发）
        steps = tuple(trace.candidate_program_steps[chosen])
        receipt = executor.evaluate(steps, origin)
        probes.append({
            "chosen": chosen,
            "steps": steps,
            "gain": (float(receipt.gain) if receipt.gain is not None else None),
            "per_view_gain": receipt.per_view_gain,
            "behavior_point_count": receipt.behavior_point_count,
            "verification_passed": receipt.verification.passed,
            "checked_windows": receipt.verification.checked_windows,
            "rejected_windows": receipt.verification.rejected_windows,
            "error": receipt.error,
        })
        if receipt.gain is None:
            continue  # 窗口 verifier 拒 / 仪器失败：不写 Episode，不消耗正控判定
        # P0-2：立即写回（Action → Support → 写 Episode → 下一次 probe 读更新 Memory）
        for op, params in steps:
            arm_memory.append(write_target_episode(
                domain=domain, op=op,
                episode_id_suffix=f"_origin{origin}",
                program_steps=[{"op": op, "params": dict(params)}],
                support_gain=float(receipt.gain), delayed_gain=None,
                support_context=resolver.window_context(values, origin, period)))
        if receipt.gain >= MATERIAL:
            break  # stop-on-first-positive
    return probes, retrieval, arm_memory


def main() -> int:
    root = PROJECT_ROOT
    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    m = core.MATERIAL_THRESHOLD
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    seed = build_seed_memory(root, values, config)
    print(f"== slice: support @{SLICE_SUPPORT} delayed @{SLICE_DELAYED} "
          f"（不重叠后续 origin；种子 @832/880 之后）")

    # 可行动算子（供给层同源——A3/A5 探索序一致）
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _actionable_operators, _allowed_operators,
    )
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    request0 = PreparationRequest(
        "scope-executor-loop",
        series0[:SLICE_SUPPORT],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )
    features = extract_public_features(series0[:SLICE_SUPPORT], task_kind="forecast")
    view = resolve_harness_view(h0, features, role="fast")
    actionable = _actionable_operators(request0, series0[:SLICE_SUPPORT], view,
                                       _allowed_operators(request0))
    print(f"== actionable (supply): {sorted(actionable)}")

    # P0-1：同一 explore=True Agent，唯一初始差异 = Memory（A5=[seed]，A3=[]）
    a5_probes, a5_retrieval, a5_memory = probe_arm(
        root, executor, values, config, SLICE_SUPPORT, seed,
        explore_operators=actionable)
    a3_probes, a3_retrieval, a3_memory = probe_arm(
        root, executor, values, config, SLICE_SUPPORT, (),
        explore_operators=actionable)

    def summarize(probes: list[dict[str, Any]]) -> dict[str, Any]:
        gains = [p["gain"] for p in probes if p["gain"] is not None]
        probed = [p["chosen"] for p in probes if p["gain"] is not None]
        rejected = [p["chosen"] for p in probes if p["gain"] is None]
        return {
            "probe_order": probed,
            "support_gains": [round(float(g), 6) for g in gains],
            "instrument_rejected": rejected,
            "harm": sum(1 for g in gains if g < -m),
            "harm_magnitude": round(sum(-g for g in gains if g < -m), 6),
            "first_positive_probe": next(
                (i + 1 for i, g in enumerate(gains) if g >= m), None),
        }

    a5_sum = summarize(a5_probes)
    a3_sum = summarize(a3_probes)

    # P0-3：delayed 只更新本轮新建 Episode（seed 与更早历史不动）
    from run_v1_target_local_loop import update_delayed_status
    a5_new_start = len(seed)
    a3_new_start = 0
    delayed: dict[str, dict[str, Any]] = {}
    for probes, arm_memory, new_start, arm_name in (
            (a5_probes, a5_memory, a5_new_start, "A5"),
            (a3_probes, a3_memory, a3_new_start, "A3")):
        # 索引循环：update_delayed_status 返回新对象，必须写回列表元素
        for i in range(new_start, len(arm_memory)):
            ep = arm_memory[i]
            if ep.workflow_signature == "identity":
                continue
            steps = tuple((s["op"], s["params"]) for s in ep.context_summary
                          ["program_geometry"]["program_steps"])
            receipt = executor.evaluate(steps, SLICE_DELAYED)
            dg = receipt.gain
            if dg is not None:
                arm_memory[i] = update_delayed_status(
                    ep, float(dg),
                    delayed_context=resolver.window_context(
                        values, SLICE_DELAYED, PERIOD))
        delayed[arm_name] = {
            ep.workflow_signature: {
                "episode_id": ep.episode_id,
                "delayed_gain": ep.delayed_response.get("gain"),
                "local_status": ep.local_status,
                "relation": ep.relation,
            }
            for ep in arm_memory[new_start:]
            if ep.workflow_signature != "identity"
        }
    # 完整性检查：seed 未被 delayed 覆盖（P0-3）
    seed_delayed_gain = float(seed[0].delayed_response.get("gain") or 0.0)
    seed_preserved = abs(seed_delayed_gain - 0.5109757360087217) < 1e-9

    # P0-4：delayed 后 @928 的动作 = counterfactual replay（非在线下一轮）
    replay: dict[str, Any] = {}
    backend_r = wiring.DeterministicStrategyBackend(
        explore=True, operators=tuple(actionable))
    r_values = series0[:SLICE_SUPPORT]
    observed_r = dict(resolver.window_context(values, SLICE_SUPPORT, PERIOD))
    observed_r["bound_period"] = float(PERIOD)
    request_r = PreparationRequest(
        "scope-executor-loop-replay",
        r_values,
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed_r),
    )
    _result_r, trace_r = TTHAFastAgent(
        TTHAAgentCore(backend_r,
                      LocalPublicToolGateway(r_values, task_kind="forecast"))
    ).prepare(request_r, h0, experience_episodes=tuple(a5_memory))
    replay = {
        "nature": "counterfactual_replay",
        "note": "delayed 打开后在同一 support origin 928 的 prepare——不是时间上"
                "的下一轮；真实纵向结论需 1024 之后空间的链",
        "chosen": trace_r.chosen_candidate_id,
        "retrieved_memory_ids": list(trace_r.retrieved_memory_ids),
        "reference1": "Reference 1" in (
            wiring.DeterministicStrategyBackend.extract_instruction(
                backend_r.requests[-1].messages) if backend_r.requests else ""),
    }

    print(f"\n== A5: {a5_sum}")
    print(f"== A3: {a3_sum}")
    print(f"== delayed: A5={delayed.get('A5')} A3={delayed.get('A3')}")
    print(f"== seed preserved (delayed 未覆盖): {seed_preserved}")
    print(f"== replay: {replay}")

    # 机制判定（P0-2 立即写回 + P0-3 卫生）
    loop_checks: dict[str, bool] = {
        "consumes_result_program": bool(a5_probes or a3_probes)
        and all("steps" in p for p in (*a5_probes, *a3_probes)),
        "window_verifier_035": all(
            p["verification_passed"] for p in (*a5_probes, *a3_probes)
            if p["gain"] is not None),
        "support_receipt": any(
            p["gain"] is not None for p in (*a5_probes, *a3_probes)),
        "immediate_writeback": (
            len(a5_memory) - len(seed)
            == sum(1 for p in a5_probes if p["gain"] is not None)
            and len(a3_memory) == sum(1 for p in a3_probes if p["gain"] is not None)),
        "episode_id_distinct": all(
            ep.episode_id.endswith(f"_origin{SLICE_SUPPORT}")
            for ep in (*a5_memory[len(seed):], *a3_memory)
            if ep.workflow_signature != "identity"),
        "delayed_only_new_episodes": bool(seed_preserved),
        "delayed_recomputed": any(
            ep.delayed_response.get("evaluated")
            for ep in (*a5_memory[a5_new_start:], *a3_memory[a3_new_start:])),
        # 同预算内：probe1 渲染种子 Reference 1 → 立即写回后 probe2 响应冲突
        # 聚合（winsorize 不再提案）——Memory 更新驱动下一次行动
        "memory_drives_next_probe": bool(
            len(a5_retrieval) >= 2
            and a5_retrieval[0]["reference1"]
            and not a5_retrieval[1]["reference1"]),
    }
    loop_complete = all(loop_checks.values())

    # A5/A3 比较（同一 Agent、同预算；唯一差异 = 初始 Memory）
    if a5_sum["harm"] > a3_sum["harm"]:
        comparison = "A5_NEGATIVE_TRANSFER_CANDIDATE"
    elif a5_sum["harm"] < a3_sum["harm"]:
        comparison = "A5_POSITIVE_TRANSFER_CANDIDATE"
    else:
        comparison = "NO_DIFFERENCE_ATTRIBUTABLE_TO_SOURCE"
    verdict = (f"SCOPE_EXECUTOR_MECHANISM_"
               f"{'PASS' if loop_complete else 'PARTIAL'}_{comparison}")
    print(f"\n== loop checks: {loop_checks}")
    print(f"== verdict: {verdict}")
    print("== 口径：机制（同一 explore=True Agent、立即写回、delayed 卫生、")
    print("   ID 后缀、replay 标注）+ 公平 A5/A3 适配数据；负迁移判定仅是")
    print("   候选（单切片证据），不调 q75——需 Program-specific Observation")
    print("   调查（winsorize 尾部比例/极值拓扑/趋势端点裁剪风险）")
    print("   + outlier_iqr @928/976 已提供替代 Program headroom")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-scope-executor-loop",
            "domain": TARGET_DOMAIN,
            "slice": {"support": SLICE_SUPPORT, "delayed": SLICE_DELAYED},
            "agent_control": {"explore": True, "note": "两臂同一 Agent，"
                              "唯一初始差异 = Source Memory（P0-1 修复）"},
            "seed_episode": {"operator": "winsorize", "origin": 832,
                             "episode_id": seed[0].episode_id},
            "a5": a5_sum,
            "a3": a3_sum,
            "a5_retrieval": a5_retrieval,
            "a3_retrieval": a3_retrieval,
            "a5_delayed": delayed.get("A5"),
            "a3_delayed": delayed.get("A3"),
            "seed_preserved": seed_preserved,
            "replay": replay,
            "loop_checks": loop_checks,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
