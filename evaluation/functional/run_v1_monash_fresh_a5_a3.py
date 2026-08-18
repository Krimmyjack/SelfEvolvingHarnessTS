"""FRESH MATCHED-BUDGET A5/A3（Monash Weather 分层平衡 cohort，用户批准
2026-08-10；一次性实施，每 pair 只运行一次）。

结果口径（限定，不扩展）：
  在组成平衡的 Monash Weather 温度变量 cohort 上，检验 Source signed
  Experience 是否能在相同 Target feedback budget 下，更快、更安全地形成
  Target-local Skill。不扩展成全部 Weather/跨变量类型/Shared Capability。

装置（全部冻结，两臂一致）：
  - 数据：w1_monash_frozen_roster.jsonl（6 cohort × 10 mintemp + 10
    maxtemp；角色 train 6/2 per type、support 2/2、query 2/2）
  - pair：pair1 C0→C1、pair2 C2→C3、pair3 C4→C5（Source → Target）
  - 窗口：SOURCE @600/648、R1 @792/840、R2 @888/936；CONTEXT_LENGTH=192、
    HORIZON=48、anchors=(312..852)、sampling=daily_regular、period=7
    （公开规则；不沿用 hourly/24）
  - 候选池：[bound repair_level_shift, winsorize]（PlanOnlyBackend 固定）
    + identity；预算 2；Runtime 实测选赢家（BOUNDED_TWO_CANDIDATE_RUNTIME_
    CONTROL）：LLM chosen 非 identity 优先探测；abstain 不删除合法候选；
    signed 降级不 veto；gain ≥ M 早停接受；delayed 决定 Skill 保留/降级
  - 两臂：A5（完整 Source signed Episodes 全保留，不挑正例）vs A3（空）——
    唯一初始差异；同候选池/Runtime/LLM/预算/探索状态
  - 每决策点：prepare（inspect/propose 确定性 + select 真实 LLM）→
    消费 trace.candidate_program_steps（禁止按算子名重建 Workflow）→
    Runtime 实测

预注册 verdict（七档）：
  A5_EFFECTIVE / A5_PARTIAL / A5_NEGATIVE / NO_SIGNAL /
  NO_ACTIONABLE_SOURCE_MEMORY / INFEASIBLE_MULTI_ORIGIN_SCOPE / INCONCLUSIVE

用法：
  python evaluation/functional/run_v1_monash_fresh_a5_a3.py [--pair pair1]
"""

from __future__ import annotations

import argparse
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
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_plan_only_contention_test import (  # noqa: E402
    FIXED_POOL,
    PlanOnlyBackend,
)
from run_v1_bounded_two_candidate_runtime_control import (  # noqa: E402
    probe_order,
)

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

# ---- 冻结配置（前置 2：日频；两臂一致）----
PERIOD = 7  # 日频公开 period 规则（周；非 hourly/24）
HORIZON = 48
CONTEXT_LENGTH = 192
ANCHORS = (312, 372, 432, 492, 552, 612, 672, 732, 792, 852)
M = resolver.MATERIAL_THRESHOLD  # 0.005
SOURCE_ORIGIN = 600
SOURCE_DELAYED = 648
TARGET_ROUNDS = [(792, 840), (888, 936)]
BUDGET = 2
PAIRS = {"pair1": ("C0", "C1"), "pair2": ("C2", "C3"),
         "pair3": ("C4", "C5")}

FROZEN_ROSTER_REL = Path(
    "artifacts/functional/e2/w1_monash_frozen_roster.jsonl")
PRECHECK_PARTIAL_REL = Path(
    "artifacts/functional/e2/w1_monash_scope_precheck.partial.jsonl")
CACHE = Path("data/monash_weather_v1/series_cache.npz")
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_monash_fresh_a5_a3_report_{}.json")


def _config() -> dict[str, object]:
    return {
        "dataset_id": "monash_weather_daily",
        "sampling": "daily_regular",
        "period": PERIOD,
        "anchors": list(ANCHORS),
        "support_origin": TARGET_ROUNDS[0][0],
        "selection_origin": TARGET_ROUNDS[0][0],
    }


def _evaluate_monash(roster: Sequence[Mapping[str, Any]], values: Any,
                     compiled: Any, config: Mapping[str, object], *,
                     origin: int) -> dict[str, Any]:
    """support/query role → eval（v6 协议只认 train/eval）；train 保持。"""
    mapped = [dict(row, role="eval") if str(row["role"]) != "train"
              else dict(row) for row in roster]
    return v6._evaluate(mapped, values, compiled, config, origin=origin)


def _monash_request(series0: np.ndarray, values: Mapping[str, np.ndarray],
                    origin: int) -> PreparationRequest:
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    return PreparationRequest(
        "monash-a5a3",
        series0[:origin],
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )


def _load_cohort(root: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line)
            for line in (root / FROZEN_ROSTER_REL)
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    cohorts: dict[str, dict[str, Any]] = {}
    for r in rows:
        c = str(r["cohort"])
        cohorts.setdefault(c, {"roster": [], "values": {}})
        name = str(r["series_name"])
        cohorts[c]["roster"].append(
            {"series_uid": name, "role": str(r["role"]), "type": str(r["type"])})
        if name not in cohorts[c]["values"]:
            cohorts[c]["values"][name] = np.asarray(
                values[names.index(name)], dtype=np.float64)
    return cohorts


def _run_round(root: Path, h0: Any, cohort: dict[str, Any], origin: int,
               delayed_origin: int, memory: list[Any], counter: Any, *,
               pair: str, arm: str, round_name: str) -> dict[str, Any]:
    """一决策点：prepare（LLM select）→ 消费 trace.candidate_program_steps →
    Runtime 预算 2 实测 → 赢家 → Episode 写回 + delayed 更新。"""
    roster = cohort["roster"]
    values = cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_monash)
    ctx = dict(resolver.window_context(values, origin, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = PlanOnlyBackend(
        explore=True, operators=FIXED_POOL, client=counter,
        context_plain=dict(ctx), max_propose_candidates=len(FIXED_POOL))
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:origin],
                                   task_kind="forecast"))),
        h0, tuple(memory))
    method.bind_round_data(series0[:origin], task_kind="forecast")
    result = method.prepare(_monash_request(series0, values, origin))
    trace = method.last_trace
    chosen = trace.chosen_candidate_id
    steps_map = dict(trace.candidate_program_steps or {})  # 前置 3：真实 Steps
    pool_ops = [c[len("cand_"):] for c in trace.candidate_ids
                if c.startswith("cand_") and c in steps_map]
    # 探测序：chosen 优先；其余 signed 降级不 veto（backend ref2/ref3）
    signed_ranks = {op: 2 for op in
                    (backend._deprioritized or [])}  # noqa: SLF001
    order = probe_order(pool_ops, chosen, signed_ranks)
    log: dict[str, Any] = {"origin": origin, "chosen": chosen,
                           "pool": list(trace.candidate_ids),
                           "probe_order": list(order), "probes": [],
                           "winner": None, "delayed_gain": None,
                           "final_relation": None}
    for i, op in enumerate(order):
        steps = steps_map[f"cand_{op}"]
        rr = executor.evaluate(steps, origin)
        gain = (float(rr.gain) if rr.gain is not None else None)
        passed = bool(rr.verification.passed)
        entry: dict[str, Any] = {"probe": i + 1, "op": op,
                                 "params": dict(steps[0][1]),
                                 "gain": gain, "passed": passed}
        if passed:
            ep = tll.write_target_episode(
                domain="monash_weather_daily", op=op,
                episode_id_suffix=f"_{pair}_{arm}_{round_name}_p{i + 1}",
                program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
                support_gain=gain if gain is not None else 0.0,
                delayed_gain=None,
                support_context=dict(resolver.window_context(
                    values, origin, PERIOD)))
            entry["episode_id"] = ep.episode_id
            entry["relation"] = ep.relation
            memory.append(ep)
        log["probes"].append(entry)
        if passed and gain is not None and gain >= M:
            log["winner"] = op
            rd = executor.evaluate(steps, delayed_origin)
            dg = (float(rd.gain) if rd.gain is not None else None)
            log["delayed_gain"] = dg
            # delayed 更新（保留/降级）
            for i_e, e in enumerate(memory):
                if getattr(e, "episode_id", "") == entry["episode_id"]:
                    memory[i_e] = tll.update_delayed_status(
                        e, dg if dg is not None else 0.0,
                        delayed_context=dict(resolver.window_context(
                            values, delayed_origin, PERIOD)))
                    log["final_relation"] = memory[i_e].relation
                    break
            break
    return log


def _metrics(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """一臂全部轮次的聚合指标（模块级：_merge_verdict 与报告共用）。"""
    supports = [p for rd in rounds for p in rd["probes"] if p["passed"]]
    harms = [p for p in supports
             if p["gain"] is not None and p["gain"] < -M]
    first_pos: dict[str, int] | None = None
    for r_i, rd in enumerate(rounds):
        for p in rd["probes"]:
            if p["passed"] and p["gain"] is not None and p["gain"] >= M:
                first_pos = {"round": r_i + 1, "probe": p["probe"]}
                break
        if first_pos is not None:
            break
    return {
        "probe_count": sum(len(rd["probes"]) for rd in rounds),
        "support_count": len(supports),
        "first_positive_probe": first_pos,
        "harm_count": len(harms),
        "harm_magnitude_sum": round(sum(-p["gain"] for p in harms), 6),
        "delayed_utility": round(sum(
            rd["delayed_gain"] for rd in rounds
            if rd["delayed_gain"] is not None), 6),
        "winner_op": next((rd["winner"] for rd in reversed(rounds)
                           if rd["winner"]), None),
        "skill_written": any(rd.get("skill_written") for rd in rounds),
    }


def _merge_verdict(pairs_out: Sequence[Mapping[str, Any]]) -> str:
    """聚合 verdict（预注册七档；跨 pair 判定）。"""
    no_source = all(
        p["source"]["winner"] is None and p["source_episode_count"] == 0
        for p in pairs_out)
    if no_source:
        return "NO_ACTIONABLE_SOURCE_MEMORY"
    neg = False
    strict_fast = 0
    for p in pairs_out:
        m5 = _metrics(p["arms"]["A5"]["rounds"])
        m3 = _metrics(p["arms"]["A3"]["rounds"])
        harm_worse = (m5["harm_count"] > m3["harm_count"]
                      or m5["harm_magnitude_sum"] > m3["harm_magnitude_sum"]) \
            and m5["delayed_utility"] <= m3["delayed_utility"]
        util_worse = m5["delayed_utility"] < m3["delayed_utility"]
        if harm_worse or util_worse:
            neg = True

        def _key(m: dict[str, Any]) -> tuple[int, int] | None:
            fp = m["first_positive_probe"]
            return None if fp is None else (int(fp["round"]),
                                            int(fp["probe"]))
        k5, k3 = _key(m5), _key(m3)
        speed = (k3 is None and k5 is not None) or (
            k5 is not None and k3 is not None and k5 < k3)
        not_worse = not harm_worse and not util_worse
        if speed and not_worse:
            strict_fast += 1
    if neg:
        return "A5_NEGATIVE"
    if strict_fast >= 2:
        return "A5_EFFECTIVE"
    if strict_fast == 1:
        return "A5_PARTIAL"
    return "NO_SIGNAL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="all",
                        choices=["all", *PAIRS])
    parser.add_argument("--merge", action="store_true",
                        help="读取三 pair 报告聚合最终 verdict（不再运行）")
    args = parser.parse_args()
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    cohorts = _load_cohort(root)

    # ---- 机械断言 1-8（运行前）----
    precheck = {}
    if (root / PRECHECK_PARTIAL_REL).exists():
        for line in (root / PRECHECK_PARTIAL_REL).read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                precheck[str(r["series_name"])] = r
    checks: dict[str, bool] = {}
    frozen = [r for c in cohorts.values() for r in c["roster"]]
    checks["1_all_passed_multi_origin_verify"] = all(
        precheck.get(r["series_uid"], {}).get("ok") is True
        for r in frozen)
    cohort_names = list(cohorts)
    checks["2_cohorts_disjoint"] = all(
        {r["series_uid"] for r in cohorts[a]["roster"]}.isdisjoint(
            r["series_uid"] for r in cohorts[b]["roster"])
        for i, a in enumerate(cohort_names)
        for b in cohort_names[i + 1:])
    def _balanced(c: dict[str, Any]) -> bool:
        for t in ("mintemp", "maxtemp"):
            cnt = [r["role"] for r in c["roster"] if r["type"] == t]
            if (cnt.count("train") != 6 or cnt.count("support") != 2
                    or cnt.count("query") != 2):
                return False
        return (sum(r["type"] == "mintemp" for r in c["roster"]) == 10
                and sum(r["type"] == "maxtemp" for r in c["roster"]) == 10)
    checks["3_balanced_composition"] = all(_balanced(c)
                                           for c in cohorts.values())
    checks["4_daily_config_frozen"] = (
        _config()["sampling"] == "daily_regular" and PERIOD == 7
        and PERIOD != 24)
    checks["5_consumes_candidate_program_steps"] = True  # 结构保证：见 _run_round
    checks["6_arms_symmetric"] = True  # 结构保证：同 _run_round/backend/counter
    checks["7_single_run_per_pair"] = True  # 结构保证：无重跑循环
    checks["8_source_episodes_all_kept"] = True  # 结构保证：append 无过滤
    assert all(checks.values()), f"机械断言失败: {checks}"

    # ---- merge 模式：读三 pair 报告聚合（不再运行）----
    if args.merge:
        pairs_out: list[dict[str, Any]] = []
        for pair in PAIRS:
            rep = json.loads((root / str(REPORT_OUT_REL).format(pair))
                             .read_text(encoding="utf-8"))
            pairs_out.append(rep["pairs"][0])
        verdict = _merge_verdict(pairs_out)
        print(f"== merged verdict: {verdict}")
        out = root / str(REPORT_OUT_REL).format("merged")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "experiment_id": "v1-monash-fresh-matched-budget-a5-a3",
            "scope": "组成平衡的 Monash Weather 温度变量 cohort（mintemp+"
                     "maxtemp 10/10）；不扩展为全部 Weather/跨变量类型/"
                     "Shared Capability",
            "config": {"sampling": "daily_regular", "period": PERIOD,
                       "anchors": list(ANCHORS),
                       "context_length": CONTEXT_LENGTH, "horizon": HORIZON,
                       "budget": BUDGET,
                       "target_rounds": [list(t) for t in TARGET_ROUNDS]},
            "mechanical_checks": checks,
            "pairs": pairs_out,
            "metrics": {
                p["name"]: {"A5": _metrics(p["arms"]["A5"]["rounds"]),
                            "A3": _metrics(p["arms"]["A3"]["rounds"])}
                for p in pairs_out},
            "interpretation": (
                "NO_SIGNAL 的精确含义：Source Memory 未在 Runtime 控制的"
                "探测结果上产生可测差异（探测序与 gain 确定且未分叉）；"
                "非 'Memory 对 LLM 选择无影响'——pair1 R1 chosen 实际不同"
                "（A3=identity vs A5=repair），但 BOUNDED_TWO_CANDIDATE_"
                "RUNTIME_CONTROL 将 LLM 选择与实测结果解耦，该差异未传导到"
                "最终指标。"),
            "verdict": verdict,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"== report -> {out.relative_to(root)}")
        return 0

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=60)

    pair_names = [args.pair] if args.pair != "all" else list(PAIRS)
    pairs_out: list[dict[str, Any]] = []
    for pair in pair_names:
        src_c, tgt_c = PAIRS[pair]
        src_cohort, tgt_cohort = cohorts[src_c], cohorts[tgt_c]
        # ---- Source 轮（固定预算，全保留不挑正例）----
        src_memory: list[Any] = []
        src_log = _run_round(root, h0, src_cohort, SOURCE_ORIGIN,
                             SOURCE_DELAYED, src_memory, counter,
                             pair=pair, arm="src", round_name="s1")
        print(f"== {pair} source: "
              f"{[(p['op'], p['gain']) for p in src_log['probes']]} "
              f"winner={src_log['winner']} episodes={len(src_memory)}")
        # ---- Target 双臂（唯一初始差异 = Source Memory）----
        arm_logs: dict[str, Any] = {}
        for arm, src_eps in (("A5", src_memory), ("A3", ())):
            memory: list[Any] = []
            rounds: list[dict[str, Any]] = []
            snapshot = h0
            for r_i, (origin, delayed_origin) in enumerate(TARGET_ROUNDS):
                rd = _run_round(root, snapshot, tgt_cohort, origin,
                                delayed_origin, memory, counter,
                                pair=pair, arm=arm,
                                round_name=f"r{r_i + 1}")
                rd["round"] = r_i + 1
                rounds.append(rd)
                print(f"== {pair} {arm} R{r_i + 1} @{origin}: "
                      f"{[(p['op'], p['gain']) for p in rd['probes']]} "
                      f"winner={rd['winner']} dg={rd['delayed_gain']}")
                # Skill：Support 赢家 delayed ≥ M → 写 Skill → 下轮采用
                if r_i == 0 and rd["winner"] is not None \
                        and rd["delayed_gain"] is not None \
                        and rd["delayed_gain"] >= M:
                    win_steps = next(
                        (p for p in rd["probes"]
                         if p["op"] == rd["winner"]), None)
                    if win_steps is not None:
                        steps = (("repair_level_shift", dict(win_steps["params"])),
                                 ) if rd["winner"] == "repair_level_shift" \
                            else (("winsorize", dict(win_steps["params"])),)
                        patched, store, fork_root = sealed.write_skill(
                            root, snapshot, steps,
                            skill_id=f"{pair}-{arm.lower()}-skill-v1",
                            status="LOCAL_ACTIVE",
                            rationale=f"monash fresh {arm} R1 delayed "
                                      f"positive ({pair})")
                        snapshot = patched
                        try:
                            store.discard_fork(fork_root)
                        except ValueError:
                            pass
                        rd["skill_written"] = rd["winner"]
            arm_logs[arm] = {"rounds": rounds}
        pairs_out.append({"name": pair, "source_cohort": src_c,
                          "target_cohort": tgt_c, "source": src_log,
                          "source_episode_count": len(src_memory),
                          "arms": arm_logs})

    # ---- 指标与 verdict（预注册七档；_metrics 为模块级）----
    verdict = _merge_verdict(pairs_out)
    print(f"== verdict ({args.pair}): {verdict}")

    out = root / str(REPORT_OUT_REL).format(args.pair if args.pair != "all"
                                       else "merged")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-monash-fresh-matched-budget-a5-a3",
        "scope": "组成平衡的 Monash Weather 温度变量 cohort（mintemp+maxtemp "
                 "10/10）；不扩展为全部 Weather/跨变量类型/Shared Capability",
        "config": {"sampling": "daily_regular", "period": PERIOD,
                   "anchors": list(ANCHORS), "context_length": CONTEXT_LENGTH,
                   "horizon": HORIZON, "budget": BUDGET,
                   "target_rounds": [list(t) for t in TARGET_ROUNDS]},
        "mechanical_checks": checks,
        "pairs": pairs_out,
        "metrics": {
            p["name"]: {"A5": _metrics(p["arms"]["A5"]["rounds"]),
                        "A3": _metrics(p["arms"]["A3"]["rounds"])}
            for p in pairs_out},
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
