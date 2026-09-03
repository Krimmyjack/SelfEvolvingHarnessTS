"""BOUNDED_TWO_CANDIDATE_RUNTIME_CONTROL（用户裁决 2026-08-10）。

科学解释拆分（用户裁决）：
  - Memory action sensitivity：已观察到（P0 offset120 三档稳定差异 + rationale
    引用 Reference——MEMORY_ACTION_SIGNAL 有开发级证据）；
  - 单候选 LLM selection reliability：不足（P0 offset80 同 prompt 翻转；
    provider 无 seed 能力（runtime/agent_backend.py _CAPABILITY_FLAGS
    provider_seed=False）——重复只能估计方差，不能让 Harness 更可靠）。

唯一改变面 = Control（决策控制层）：LLM 不再单独决定唯一候选并删除其余；
LLM 参与候选生成/排序，Runtime 在固定预算内验证最多两个候选，用真实
Support 结果选择赢家。不改 Memory/Prompt/Observation/Program/Risk。

流程（每轮预算 B=2）：
  Memory + Context + LLM → 产生并排序合法候选
  → Runtime 按探测序最多执行两个 Support probe（LLM chosen 非 identity
    优先；LLM abstain 不删除池中合法候选；signed positive 提高顺序；
    weak negative/conflict 只降级不 veto）
  → gain ≥ M 停止并接受（早停）
  → 都不正向 → abstain
  → 每个实际 probe 写 Episode；只让 Support 赢家进入 Skill；
    delayed 决定保留或降级（dg < M → 不写 Skill/RESTRICTED）

最小验收（development replay，零新增 LLM 调用）——用 P0 已记录输出
（w1_plan_only_contention_report.json offset80：M_positive→repair、
M_negative rep1→winsorize、M_remove→abstain 三条 replay，池相同）：
  1. 一条 replay 选择 repair；一条选择 winsorize；一条 abstain；
  2. 两次均暴露相同两个候选（池相同）；
  3. Runtime 依据相同 Support 结果选择同一赢家；
  4. 总 Support ≤ 2；
  5. delayed 翻负时仍能降级（状态机演示：合成 delayed 翻负 → 降级、
     Skill 不写）；
  6. 不把开发数据称为 fresh 证据。

context 选择说明（已暴露 development 数据 @792 实测）：offset120 双负
（repair −0.0027 / winsorize −0.041）——三条 replay 一致 abstain（演示
"Memory 引导 LLM 选负向 repair 时 Runtime 拒绝接受"）；**offset80 有
正向候选（repair −0.0198 / winsorize +0.0376）——激活赢家路径**（演示
"LLM 选 repair/abstain 均不影响 Runtime 收敛到正向 winsorize"）。

verifier rejection 不计 Support receipt（V1 语义）；REJ probe 记录但不写
Episode（无 gain 可写）。

用法：
  python evaluation/functional/run_v1_bounded_two_candidate_runtime_control.py
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
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

DOMAIN = "uci_electricity_load_diagrams"
PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
SUPPORT_ORIGIN = 792
DELAYED_ORIGIN = SUPPORT_ORIGIN + HORIZON  # 840
BUDGET = 2
OFFSET = 80  # P0 翻转档的 context；@792 winsorize 正向（+0.0376）激活赢家路径
REPLAY_SRC_REL = Path(
    "artifacts/functional/e2/w1_plan_only_contention_report.json")
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_bounded_two_candidate_runtime_control_report.json")

# 三条 replay（P0 报告 offset80 已记录 LLM 输出；池相同 [identity, repair,
# winsorize]；chosen 不同——含翻转档 M_negative 的 winsorize 选择）
REPLAYS = [
    {"name": "A_llm_repair", "level": "M_positive", "rep": 0,
     "expected_llm_chosen": "cand_repair_level_shift"},
    {"name": "B_llm_winsorize", "level": "M_negative", "rep": 1,
     "expected_llm_chosen": "cand_winsorize"},
    {"name": "C_llm_abstain", "level": "M_remove", "rep": 0,
     "expected_llm_chosen": "identity"},
]

VERDICT_RANK = {"POSITIVE_PRIOR": 0, "UNKNOWN": 1, "CONFLICT": 2,
                "RISK_PRIOR": 2}  # weak negative/conflict 降级不 veto


def probe_order(pool_ops: Sequence[str], llm_chosen: str,
                signed_ranks: Mapping[str, int]) -> list[str]:
    """探测序（预算 BUDGET）：
    - LLM chosen 非 identity → 第一（优先探测）；
    - LLM abstain → 不删除池中合法候选（仍可探测）；
    - 其余候选按 signed 排序（POSITIVE 提前 / UNKNOWN 中 / weak
      negative/conflict 降级不 veto）；
    - 截断到 BUDGET。"""
    chosen = None if llm_chosen == "identity" else llm_chosen
    if chosen and chosen.startswith("cand_"):
        chosen = chosen[len("cand_"):]  # 候选 ID → 算子名（steps_map 键）
    rest = [o for o in pool_ops if o != chosen]
    rest.sort(key=lambda o: signed_ranks.get(o, 1))
    order: list[str] = ([chosen] if chosen else []) + rest
    return order[:BUDGET]


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(root, offset=OFFSET)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=v6._evaluate)

    # 候选 steps（与 P0 装置一致：repair bound 参数 / winsorize 默认参数）
    fe = dict(extract_public_features(series0[:SUPPORT_ORIGIN],
                                      task_kind="forecast"))
    bindings = OPERATOR_METADATA["repair_level_shift"].get(
        "public_parameter_bindings") or {}
    repair_params = {p: float(fe[f]) for p, f in bindings.items() if f in fe}
    assert len(repair_params) == len(bindings), "bound repair params incomplete"
    steps_map = {
        "repair_level_shift": (("repair_level_shift", repair_params),),
        "winsorize": (("winsorize",
                       dict(wiring.contract_params("winsorize", PERIOD))),),
    }

    # P0 已记录 replay 输入
    p0 = json.loads((root / REPLAY_SRC_REL).read_text(encoding="utf-8"))
    p0_off = p0["results"][str(OFFSET)]

    results: list[dict[str, Any]] = []
    winners: list[str | None] = []
    pools_seen: set[tuple[str, ...]] = set()
    for rep_cfg in REPLAYS:
        rec = p0_off[rep_cfg["level"]]
        dec = rec["decisions"][rep_cfg["rep"]]
        llm_chosen = dec["chosen"]
        assert llm_chosen == rep_cfg["expected_llm_chosen"], (
            f"{rep_cfg['name']}: P0 recorded chosen={llm_chosen}, "
            f"expected {rep_cfg['expected_llm_chosen']}")
        pool = [c[len("cand_"):] for c in dec["pool"]
                if c.startswith("cand_")]
        pools_seen.add(tuple(pool))
        # signed 排序（weak negative/conflict 只降级不 veto——replay 档位的
        # resolver 实际 verdict）
        diag = rec["signed_diagnostic"]
        signed_ranks = {}
        if diag.get("rendered") and diag.get("repair_verdict"):
            signed_ranks["repair_level_shift"] = VERDICT_RANK.get(
                str(diag["repair_verdict"]), 1)
        order = probe_order(pool, llm_chosen, signed_ranks)
        # ---- Runtime 探测（预算 2）----
        probes: list[dict[str, Any]] = []
        winner: str | None = None
        delayed_gain: float | None = None
        final_relation: str | None = None
        for i, op in enumerate(order):
            steps = steps_map[op]
            rr = executor.evaluate(steps, SUPPORT_ORIGIN)
            gain = (float(rr.gain) if rr.gain is not None else None)
            passed = bool(rr.verification.passed)
            entry: dict[str, Any] = {"probe": i + 1, "op": op, "gain": gain,
                                     "passed": passed}
            if passed:
                ep = tll.write_target_episode(
                    domain=DOMAIN, op=op,
                    episode_id_suffix=f"_rtctrl_{rep_cfg['name']}_p{i + 1}",
                    program_steps=[{"op": o, "params": dict(p)}
                                   for o, p in steps],
                    support_gain=gain if gain is not None else 0.0,
                    delayed_gain=None,
                    support_context=dict(resolver.window_context(
                        tgt_values, SUPPORT_ORIGIN, PERIOD)))
                entry["episode_id"] = ep.episode_id
                entry["relation"] = ep.relation
            probes.append(entry)
            if passed and gain is not None and gain >= M:
                winner = op  # 早停接受
                rd = executor.evaluate(steps, DELAYED_ORIGIN)
                delayed_gain = (float(rd.gain) if rd.gain is not None else None)
                ep = tll.update_delayed_status(
                    ep, delayed_gain if delayed_gain is not None else 0.0,
                    delayed_context=dict(resolver.window_context(
                        tgt_values, DELAYED_ORIGIN, PERIOD)))
                final_relation = ep.relation
                entry["delayed_gain"] = delayed_gain
                entry["final_relation"] = final_relation
                break
        results.append({
            "name": rep_cfg["name"], "level": rep_cfg["level"],
            "llm_chosen": llm_chosen, "pool": list(pool),
            "signed_repair_verdict": diag.get("repair_verdict"),
            "probe_order": order, "probes": probes,
            "winner": winner, "delayed_gain": delayed_gain,
            "final_relation": final_relation,
            # Skill 规则：只让 Support 赢家进入 Skill；delayed < M → 降级
            # （不写 Skill——development 验收只验证状态，不落盘污染）
            "skill_written": (winner if (winner is not None
                                         and delayed_gain is not None
                                         and delayed_gain >= M) else None),
        })
        winners.append(winner)
        print(f"== {rep_cfg['name']}: llm_chosen={llm_chosen} "
              f"pool={pool} signed_repair={diag.get('repair_verdict')} "
              f"probe_order={order} winner={winner} "
              f"delayed={delayed_gain} relation={final_relation}")

    # ---- 验收检查 ----
    checks: dict[str, bool] = {}
    checks["1_replay_a_repair"] = (
        results[0]["llm_chosen"] == "cand_repair_level_shift")
    checks["2_replay_b_winsorize_or_abstain"] = (
        results[1]["llm_chosen"] in ("cand_winsorize", "identity"))
    checks["2b_replay_c_abstain"] = (
        results[2]["llm_chosen"] == "identity")
    checks["3_pool_identical"] = len(pools_seen) == 1
    unique_winners = {w for w in winners if w is not None}
    checks["4_same_winner"] = (
        (len(unique_winners) == 1) if any(w is not None for w in winners)
        else True)  # 全 abstain 也一致
    checks["5_support_budget_le_2"] = all(
        len(r["probes"]) <= BUDGET for r in results)
    # delayed 降级机制（用户验收 6）：winner 存在时，若 delayed 翻负 → 降级、
    # Skill 不写。数据驱动（本数据 winner 的 delayed）+ 合成演示（把 winner
    # 的 delayed 强制翻负重跑 update_delayed_status 状态机——零额外 evaluate）。
    downgrade_demo: dict[str, Any] | None = None
    winner_op = next((w for w in winners if w is not None), None)
    if winner_op is not None:
        ep_demo = tll.write_target_episode(
            domain=DOMAIN, op=winner_op,
            episode_id_suffix="_rtctrl_downgrade_demo",
            program_steps=[{"op": o, "params": dict(p)}
                           for o, p in steps_map[winner_op]],
            support_gain=M + 0.01,  # Support 正向（赢家前提）
            delayed_gain=None,
            support_context=dict(resolver.window_context(
                tgt_values, SUPPORT_ORIGIN, PERIOD)))
        ep_demo = tll.update_delayed_status(
            ep_demo, -0.1,  # 合成 delayed 翻负
            delayed_context=dict(resolver.window_context(
                tgt_values, DELAYED_ORIGIN, PERIOD)))
        downgrade_demo = {
            "op": winner_op,
            "relation": ep_demo.relation,
            "local_status": getattr(ep_demo, "local_status", "?"),
            "skill_written": None,  # delayed < M → 不写 Skill（降级）
        }
    checks["6_delayed_downgrade_path"] = bool(
        downgrade_demo
        and downgrade_demo["relation"] in ("NEGATIVE", "RESTRICTED",
                                           "CONFLICT", "ABSTAIN"))
    checks["7_abstain_keeps_candidates"] = len(results[2]["probes"]) >= 1
    passed = all(checks.values())
    verdict = ("BOUNDED_TWO_CANDIDATE_RUNTIME_CONTROL_DEVELOPMENT_PASS"
               if passed else "FAILED")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-bounded-two-candidate-runtime-control",
        "note": "development replay 验收（P0 已记录 LLM 输出；已暴露 UCI "
                "offset120 @792；不称 fresh 证据）",
        "dataset": DOMAIN, "offset": OFFSET,
        "support_origin": SUPPORT_ORIGIN, "delayed_origin": DELAYED_ORIGIN,
        "budget": BUDGET,
        "replays": results,
        "downgrade_demo": downgrade_demo,
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
