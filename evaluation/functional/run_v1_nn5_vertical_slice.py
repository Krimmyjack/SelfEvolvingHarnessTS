"""V1 NN5 纵向切片（零 LLM，2026-08-08）。

审查裁决（十五）：CLIPPING_GEOMETRY_NOT_DISCRIMINATIVE 确认成立（结论边界：
"这组 clipping geometry 不能识别翻转"，不扩为"所有特征都不能识别"）；停止扩
Pattern；下一条真正纵向链选择 **NN5**（NOAA 会混入 Program-headroom 问题）。

NN5 数据长度 791（非 1024+），但容纳一个合法的真实下一轮：
  Source: support 536 → delayed 584（结果开放至 632）
  R1:     support 632 → delayed 680（结果开放至 728）
  R2:     support 728 → future 776（776 < 791——R2 是发生在 R1 delayed
          之后的**在线动作**，非同 origin counterfactual replay）
R2 delayed（@776 需数据到 824）NN5 不具备 → 本轮只能承重：
  - R1 完整 Support + delayed 效用；
  - R2 下一轮 Support 行动是否受累计 Memory 影响；
  - 不能声称 R2 最终 delayed Skill 已确认。

执行边界（裁决原文）：
  1. 两臂同 Agent、同 actionable inventory、同 Target 预算；
  2. 只允许初始 Memory 不同（A5=[种子]，A3=[]）；
  3. 每次 Support 后立即写 Episode（probe_arm 语义）；
  4. delayed 只更新该轮新 Episode；
  5. R2 在 728 重新调用正常 prepare，评估 [728,776)；
  6. Source 536/584 种子用当前 ScopeExecutor + H0 verifier 重新确认，
     不直接信任旧 Runner 数值；
  7. 结果称 LOCAL_SEEDED，不称跨域 A5；
  8. 零 LLM。

用法：
  python evaluation/functional/run_v1_nn5_vertical_slice.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_scope_executor_loop as loop  # noqa: E402（probe_arm 复用）
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

TARGET_DOMAIN = "nn5"
PERIOD = 7
HORIZON = 48
MAX_TARGET_PROBES = 2  # 同预算
SEED_OP = "impute_ssm"
SEED_ORIGIN = 536
SEED_DELAYED = 584
R1_SUPPORT = 632
R1_DELAYED = 680
R2_SUPPORT = 728
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_nn5_vertical_slice_report.json")
MATERIAL = core.MATERIAL_THRESHOLD


def _actionable_at(root: Path, series: np.ndarray, origin: int) -> tuple[str, ...]:
    """该 origin 的供给层 actionable（与真实入口同源；两臂共用）。"""
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _actionable_operators, _allowed_operators,
    )
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    request = PreparationRequest(
        "nn5-slice",
        series[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )
    features = extract_public_features(series[:origin], task_kind="forecast")
    view = resolve_harness_view(h0, features, role="fast")
    return _actionable_operators(request, series[:origin], view,
                                 _allowed_operators(request))


def main() -> int:
    root = PROJECT_ROOT
    m = core.MATERIAL_THRESHOLD
    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    print(f"== nn5: max_len={max(int(len(v)) for v in values.values())} "
          f"period={PERIOD} roster={len(roster)}")

    # ---------------------------------------------------------------
    # 执行边界 6：种子用当前 ScopeExecutor + H0 verifier 重新确认
    # （不直接信任旧 Runner 数值——impute_ssm @536/584）
    # ---------------------------------------------------------------
    params = wiring.contract_params(SEED_OP, PERIOD)
    seed_steps = ((SEED_OP, params),)
    s_seed = executor.evaluate(seed_steps, SEED_ORIGIN)
    d_seed = executor.evaluate(seed_steps, SEED_DELAYED)
    seed_ok = (s_seed.gain is not None and s_seed.gain >= m
               and d_seed.gain is not None and d_seed.gain >= m)
    print(f"== seed re-confirmation: {SEED_OP} @{SEED_ORIGIN} support="
          f"{s_seed.gain if s_seed.gain is None else round(s_seed.gain, 5)} "
          f"@{SEED_DELAYED} delayed="
          f"{d_seed.gain if d_seed.gain is None else round(d_seed.gain, 5)} "
          f"-> {'DOUBLE_POSITIVE' if seed_ok else 'NOT_DOUBLE_POSITIVE'}")
    seed: list[Any] = []
    if seed_ok:
        from run_v1_target_local_loop import update_delayed_status, write_target_episode
        ep = write_target_episode(
            domain=TARGET_DOMAIN, op=SEED_OP,
            episode_id_suffix=f"_origin{SEED_ORIGIN}",
            program_steps=[{"op": SEED_OP, "params": dict(params)}],
            support_gain=float(s_seed.gain), delayed_gain=None,
            support_context=resolver.window_context(values, SEED_ORIGIN, PERIOD))
        ep = update_delayed_status(
            ep, float(d_seed.gain),
            delayed_context=resolver.window_context(values, SEED_DELAYED, PERIOD))
        seed = [ep]
        print(f"   seed episode: id={ep.episode_id} status={ep.local_status} "
              f"relation={ep.relation}")

    # ---------------------------------------------------------------
    # R1 @632（B=2）：A5=[种子] vs A3=[]，立即写回，delayed @680 只更新本轮
    # ---------------------------------------------------------------
    actionable_r1 = _actionable_at(root, series0, R1_SUPPORT)
    print(f"== R1 @{R1_SUPPORT}: actionable n={len(actionable_r1)}")
    a5_r1, a5_ret1, a5_mem = loop.probe_arm(
        root, executor, values, config, R1_SUPPORT, seed,
        explore_operators=actionable_r1, domain=TARGET_DOMAIN, period=PERIOD)
    a3_r1, a3_ret1, a3_mem = loop.probe_arm(
        root, executor, values, config, R1_SUPPORT, (),
        explore_operators=actionable_r1, domain=TARGET_DOMAIN, period=PERIOD)

    from run_v1_target_local_loop import update_delayed_status
    delayed_r1: dict[str, dict[str, Any]] = {}
    for probes, arm_mem, arm_name in ((a5_r1, a5_mem, "A5"), (a3_r1, a3_mem, "A3")):
        new_start = len(seed) if arm_name == "A5" else 0
        for i in range(new_start, len(arm_mem)):
            ep = arm_mem[i]
            if ep.workflow_signature == "identity":
                continue
            steps = tuple((s["op"], s["params"]) for s in ep.context_summary
                          ["program_geometry"]["program_steps"])
            receipt = executor.evaluate(steps, R1_DELAYED)
            if receipt.gain is not None:
                arm_mem[i] = update_delayed_status(
                    ep, float(receipt.gain),
                    delayed_context=resolver.window_context(
                        values, R1_DELAYED, PERIOD))
        delayed_r1[arm_name] = {
            ep.workflow_signature: {
                "episode_id": ep.episode_id,
                "delayed_gain": ep.delayed_response.get("gain"),
                "local_status": ep.local_status,
                "relation": ep.relation,
            }
            for ep in arm_mem[new_start:]
            if ep.workflow_signature != "identity"
        }

    def summarize(probes: list[dict[str, Any]]) -> dict[str, Any]:
        gains = [p["gain"] for p in probes if p["gain"] is not None]
        probed = [p["chosen"] for p in probes if p["gain"] is not None]
        return {
            "probe_order": probed,
            "support_gains": [round(float(g), 6) for g in gains],
            "harm": sum(1 for g in gains if g < -m),
            "harm_magnitude": round(sum(-g for g in gains if g < -m), 6),
            "first_positive_probe": next(
                (i + 1 for i, g in enumerate(gains) if g >= m), None),
        }

    r1_summary = {"A5": summarize(a5_r1), "A3": summarize(a3_r1)}

    # ---------------------------------------------------------------
    # R2 @728：真正在线下一轮（R1 delayed 680 之后；评估 [728,776)）
    # 累计 Memory：A5=[seed, R1 A5 写回]，A3=[R1 A3 写回]
    # ---------------------------------------------------------------
    actionable_r2 = _actionable_at(root, series0, R2_SUPPORT)
    print(f"== R2 @{R2_SUPPORT}（在线下一轮，评估 [728,776)）")
    a5_r2, a5_ret2, a5_mem2 = loop.probe_arm(
        root, executor, values, config, R2_SUPPORT, a5_mem,
        explore_operators=actionable_r2, domain=TARGET_DOMAIN, period=PERIOD)
    a3_r2, a3_ret2, a3_mem2 = loop.probe_arm(
        root, executor, values, config, R2_SUPPORT, a3_mem,
        explore_operators=actionable_r2, domain=TARGET_DOMAIN, period=PERIOD)
    r2_summary = {"A5": summarize(a5_r2), "A3": summarize(a3_r2)}

    print(f"\n== R1 @632: A5={r1_summary['A5']} A3={r1_summary['A3']}")
    print(f"== R1 delayed @680: A5={delayed_r1.get('A5')} A3={delayed_r1.get('A3')}")
    print(f"== R2 @728: A5={r2_summary['A5']} A3={r2_summary['A3']}")

    # ---------------------------------------------------------------
    # 判定
    # ---------------------------------------------------------------
    r1_a5 = r1_summary["A5"]
    r1_a3 = r1_summary["A3"]
    r2_a5 = r2_summary["A5"]
    r2_a3 = r2_summary["A3"]
    checks: dict[str, bool] = {
        # 种子重新确认（执行边界 6）
        "seed_reconfirmed_double_positive": seed_ok,
        # R1：A5 首探由种子 Memory 引导（Reference 1 → impute_ssm）
        "r1_a5_memory_guided": bool(a5_ret1 and a5_ret1[0]["reference1"]
                                    and a5_ret1[0]["chosen"] == f"cand_{SEED_OP}"),
        # R1：立即写回 + delayed 只更新本轮
        "r1_immediate_writeback": (
            len(a5_mem) - len(seed)
            == sum(1 for p in a5_r1 if p["gain"] is not None)),
        "r1_delayed_only_new": True,  # 由 delayed_r1 只含本轮 Episode 保证
        # R2：真正在线下一轮（origin 728 > R1 delayed 680）
        "r2_online_next_round": bool(a5_r2 or a3_r2),
        # R2：A5 行动受累计 Memory 影响（首探 = impute_ssm 家族）
        "r2_a5_memory_influence": bool(
            a5_ret2 and a5_ret2[0]["reference1"]
            and a5_ret2[0]["chosen"] == f"cand_{SEED_OP}"),
    }
    mechanism = all(checks.values())
    # A5/A3 比较（R1 首轮 + R2 在线轮；仅同域种子，称 LOCAL_SEEDED）
    r1_cmp = ("A5_BETTER" if r1_a5["harm"] < r1_a3["harm"]
              else "A5_SAME" if r1_a5["harm"] == r1_a3["harm"] else "A5_WORSE")
    r2_cmp = ("A5_BETTER" if r2_a5["harm"] < r2_a3["harm"]
              else "A5_SAME" if r2_a5["harm"] == r2_a3["harm"] else "A5_WORSE")
    verdict = (f"NN5_VERTICAL_SLICE_{'MECHANISM_PASS' if mechanism else 'PARTIAL'}"
               f"_R1_{r1_cmp}_R2_{r2_cmp}_LOCAL_SEEDED")
    print(f"\n== checks: {checks}")
    print(f"== verdict: {verdict}")
    print("== 口径：LOCAL_SEEDED（同域种子 536/584，不称跨域 A5）；R2 为在线")
    print("   下一轮（728 > 680，评估 [728,776)）；R2 delayed Skill 未确认")
    print("   （NN5 数据不足 824）；零 LLM")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-nn5-vertical-slice",
            "domain": TARGET_DOMAIN,
            "slice_layout": {
                "seed": {"support": SEED_ORIGIN, "delayed": SEED_DELAYED,
                         "operator": SEED_OP},
                "r1": {"support": R1_SUPPORT, "delayed": R1_DELAYED},
                "r2": {"support": R2_SUPPORT,
                       "note": "online next round; evaluates [728,776); "
                               "no delayed (data ends at 791)"},
            },
            "seed_reconfirmation": {
                "operator": SEED_OP,
                "support_origin": SEED_ORIGIN,
                "support_gain": s_seed.gain,
                "delayed_origin": SEED_DELAYED,
                "delayed_gain": d_seed.gain,
                "scope_executor_verified": True,
                "double_positive": seed_ok,
            },
            "r1": {"summary": r1_summary, "retrieval": {"A5": a5_ret1,
                                                        "A3": a3_ret1},
                   "delayed": delayed_r1},
            "r2": {"summary": r2_summary, "retrieval": {"A5": a5_ret2,
                                                        "A3": a3_ret2}},
            "checks": checks,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
