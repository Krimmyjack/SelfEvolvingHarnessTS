"""E2_SOURCE_MEMORY_TWO_SLOT（用户裁决 2026-08-12）。

已暴露数据、零 LLM——三臂比较 Source Memory 双槽机制：
  A5-hard     ：Source Reference 可独占候选供应（当前行为——ref1 短路）
  A5-two-slot ：一个 Source prior 槽 + 一个当前 Context 探索槽
                 （reserve_exploration_slot=True——仅一个 Control）
  A3          ：无 Source Memory

只验证五项（用户裁决）：
  1. Source 经验不能删除探索槽（two-slot 池含探索候选；hard 池被独占）；
  2. Source 正例最多优先一个 trial（winsorize 只作第一 probe——后续
     探测不重复 Source 算子）；
  3. Source 负例/冲突只能降级、不能封杀（signswap NEGATIVE——算子仍在
     池中但排序靠后——耗尽 UNKNOWN 后才尝试）；
  4. Target 反馈能覆盖 Source 排序（KDD T117：R1 @888 Source 正例优先
     → Target winsorize −0.143 失败 → R2 @984 winsorize 降级）；
  5. 三臂 Support 预算完全相同（budget=2）。

装置（全部已暴露）：traffic @792（sealed 正例——winsorize Source
Episode）；KDD T117 @888→@984（Target 覆盖）。

通过（五项全过）→ MEMORY_TWO_SLOT_CONTROL_PASS——才进入 E3 fresh
A5-two-slot vs A3。

用法：
  python evaluation/functional/run_v1_e2_memory_two_slot.py
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
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config as _kdd_config,
    _evaluate_kdd,
    _request as _kdd_request,
)
from run_v1_operational_self_evolution_loop import _traffic_setup  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
M = resolver.MATERIAL_THRESHOLD
BUDGET = 2
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_e2_memory_two_slot_report.json"


def _source_episode(values: Mapping[str, Any], *, sign: float) -> Any:
    """traffic winsorize Source Episode（sign=+1 双正 / sign=-1 NEGATIVE
    对照——同候选同 Context 经验符号交换）。"""
    first_uid = list(values)[0]
    s = np.asarray(values[first_uid])
    ep = tll.write_target_episode(
        domain="monash:traffic_hourly", op="winsorize",
        episode_id_suffix=f"_e2_src_{'+' if sign > 0 else '-'}",
        delayed_gain=None,
        program_steps=[{"op": "winsorize", "params": {}}],
        support_gain=0.4 * sign,
        support_context=dict(resolver.window_context(values, 792, PERIOD)))
    return tll.update_delayed_status(
        ep, 0.03 * sign,
        delayed_context=dict(resolver.window_context(values, 840, PERIOD)))


def _run_traffic(root: Path, *, memory: tuple, reserve: bool,
                 rounds: int = 1) -> Any:
    """一轮或多轮（rounds=2 用于负例"降级不封杀"的两轮验证——UNKNOWN
    耗尽后 deprioritized 回退——winsorize 在第二轮进池）。"""
    roster, vals, series0, executor = _traffic_setup(root)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    out: list[Any] = []
    # 跨轮复用同一 method/backend（explore 状态累积——负例"降级→回退"
    # 依赖跨轮状态：R1 探索 denoise_median → R2 UNKNOWN 耗尽 → 回退
    # winsorize）；run_online_round 内部 bind_round_data 换 gateway。
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("denoise_median", "winsorize"),
                                  max_propose_candidates=3, force_pool=True,
                                  reserve_exploration_slot=reserve),
        LocalPublicToolGateway(series0[:792], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), h0, memory)
    for ri in range(rounds):
        origin = 792 + ri * 96
        r = run_online_round(
            method, executor, sealed._request(series0, vals, origin), vals,
            origin=origin, slow_agent=None, controller=None, store=None,
            card_builder=lambda e: {}, round_name=f"e2_r{ri + 1}",
            budget=BUDGET, allow_slow=False,
            domain="monash:traffic_hourly", period=24)
        out.append({"r": r,
                    "pool": list(method.last_trace.candidate_ids or ()),
                    "probes": [(p["candidate_id"], p.get("gain"))
                               for p in r.actual_probed_programs]})
    return out[0] if rounds == 1 else out


def _kdd_target_coverage(root: Path) -> dict[str, Any]:
    """Target 反馈覆盖 Source 排序：R1 @888 winsorize 正向 Source 优先 →
    Target −0.143 失败 → R2 @984 winsorize 降级。"""
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in rows}
    series0 = vals[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, vals, _kdd_config(),
                             evaluate_fn=_evaluate_kdd)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    src = _source_episode(vals, sign=1.0)
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=("winsorize",),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:888], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), h0, (src,))
    r1 = run_online_round(
        method, executor, _kdd_request(series0, vals, 888), vals,
        origin=888, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name="cov_r1", budget=BUDGET,
        allow_slow=False, domain="kdd_cup_2018", period=24)
    # delayed 到达后更新（成对证据——signed 判定需要 support+delayed
    # 双负才渲染 CONFLICT 降级——单 support 负向判定 UNKNOWN）
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: PLC0415
        open_delayed,
    )
    open_delayed(r1, executor)
    r1_probe_op = (r1.actual_probed_programs[0]["candidate_id"]
                   if r1.actual_probed_programs else None)
    # R2 @984：memory 含 Source 正例 **和 R1 的 Target 负向 Episode**——
    # Target 反馈必须能覆盖 Source 排序（R1 winsorize −0.143 失败 →
    # winsorize 渲染为 Reference 2/3 降级）。注意从 method 取 delayed
    # 已更新的版本（open_delayed 替换 method._experience_episodes——
    # r1._episodes 持有旧引用）。
    r1_target_ep = (method._experience_episodes[-1]  # noqa: SLF001
                    if getattr(method, "_experience_episodes", None)
                    else None)
    core2 = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=("winsorize",),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:984], task_kind="forecast"))
    method2 = TTHAMethod(sealed.TTHAFastAgent(core2), h0,
                         (src,) + ((r1_target_ep,)
                                   if r1_target_ep is not None else ()))
    r2 = run_online_round(
        method2, executor, _kdd_request(series0, vals, 984), vals,
        origin=984, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name="cov_r2", budget=BUDGET,
        allow_slow=False, domain="kdd_cup_2018", period=24)
    t2 = method2.last_trace
    # Target 覆盖 Source 排序 = verdict 层验证：R1（memory 只 src）winsorize
    # POSITIVE_PRIOR（Source 正例优先生效）→ R2（memory 加 Target 负向）
    # winsorize 不再 POSITIVE_PRIOR（被中和/降级——UNKNOWN/RISK——Source
    # 排序被 Target 证据覆盖）。
    from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: PLC0415
        resolve_order,
    )
    _obs888 = dict(getattr(_kdd_request(series0, vals, 888),
                           "observed_pattern_spec", {}) or {})
    _obs984 = dict(getattr(_kdd_request(series0, vals, 984),
                           "observed_pattern_spec", {}) or {})
    qc888 = {k: float(v) for k, v in _obs888.items()
             if str(k).startswith(("recent.", "change."))
             and isinstance(v, (int, float))}
    qc984 = {k: float(v) for k, v in _obs984.items()
             if str(k).startswith(("recent.", "change."))
             and isinstance(v, (int, float))}
    _order1, _s1 = resolve_order(
        query_context=qc888, episodes=(src,), operators=("winsorize",),
        material_threshold=M, task_consumer_key="forecast|ridge|sMASE",
        allowed_operators=("winsorize",))
    _order2, _s2 = resolve_order(
        query_context=qc984, episodes=(src, r1_target_ep),
        operators=("winsorize",),
        material_threshold=M, task_consumer_key="forecast|ridge|sMASE",
        allowed_operators=("winsorize",))
    v1 = (_s1.get("per_op", {}).get("winsorize", {}).get("verdict"))
    v2 = (_s2.get("per_op", {}).get("winsorize", {}).get("verdict"))
    return {"r1_probe_op": r1_probe_op, "r1_gain": (
        r1.actual_probed_programs[0].get("gain")
        if r1.actual_probed_programs else None),
        "r2_pool": list(t2.candidate_ids or ()),
        "r2_chosen": t2.chosen_candidate_id,
        "r2_probes": [(p["candidate_id"], p.get("gain"))
                      for p in r2.actual_probed_programs],
        "source_verdict_r1": v1,
        "source_verdict_r2": v2,
        "target_overrides": bool(
            v1 == "POSITIVE_PRIOR" and v2 != "POSITIVE_PRIOR")}


def main() -> int:
    root = PROJECT_ROOT
    src_pos = _source_episode(_traffic_setup(root)[1], sign=1.0)
    src_neg = _source_episode(_traffic_setup(root)[1], sign=-1.0)
    hard = _run_traffic(root, memory=(src_pos,), reserve=False)
    two = _run_traffic(root, memory=(src_pos,), reserve=True)
    a3 = _run_traffic(root, memory=(), reserve=False)
    # 负例"降级不封杀"：两轮——R1 denoise_median 先探（winsorize 降级）
    # → R2 UNKNOWN 耗尽 → deprioritized 回退 → winsorize 进池（不硬排除）
    neg_r1, neg_r2 = _run_traffic(root, memory=(src_neg,), reserve=True,
                                  rounds=2)
    cov = _kdd_target_coverage(root)

    def _ops(pool: list) -> list[str]:
        return [c[len("cand_"):] for c in pool
                if c.startswith("cand_") and not c.startswith("cand_skill_")]

    checks: dict[str, bool] = {
        # 1. Source 不能删除探索槽
        "C1_exploration_slot_preserved": bool(
            "denoise_median" in _ops(two["pool"])
            and "denoise_median" not in _ops(hard["pool"])),
        # 2. Source 正例最多优先一个 trial
        "C2_source_priority_limited": bool(
            two["probes"][0][0] == "cand_winsorize"
            and len([p for p in two["probes"]
                     if p[0] == "cand_winsorize"]) == 1),
        # 3. 负例只能降级不能封杀（R1 winsorize 被降级不在池首；
        #    R2 UNKNOWN 耗尽后 deprioritized 回退——winsorize 进池——
        #    不硬排除）
        "C3_negative_only_degrades": bool(
            (neg_r1["probes"][0][0] != "cand_winsorize")
            and ("winsorize" in _ops(neg_r2["pool"]))),
        # 4. Target 反馈覆盖 Source 排序（KDD：R1 winsorize 失败（Source
        #    正例曾把 winsorize 排第一）→ R2 verdict 层：winsorize 从
        #    POSITIVE_PRIOR 变为非 POSITIVE_PRIOR（Target 负向证据中和
        #    Source 正例——覆盖排序）
        "C4_target_overrides_source": bool(
            cov["r1_probe_op"] == "cand_winsorize"
            and cov["r1_gain"] is not None
            and float(cov["r1_gain"]) < -M
            and cov["target_overrides"]),
        # 5. 三臂预算相同
        "C5_budget_identical": bool(
            hard["r"].target_support_receipts_used <= BUDGET
            and two["r"].target_support_receipts_used <= BUDGET
            and a3["r"].target_support_receipts_used <= BUDGET
            and neg_r1["r"].target_support_receipts_used <= BUDGET),
    }
    verdict = "MEMORY_TWO_SLOT_CONTROL_PASS" if all(checks.values()) \
        else "MEMORY_TWO_SLOT_CONTROL_FAILED"
    print(f"== hard pool: {hard['pool']} probes={hard['probes']}")
    print(f"== two  pool: {two['pool']} probes={two['probes']}")
    print(f"== a3   pool: {a3['pool']} probes={a3['probes']}")
    print(f"== neg  R1 pool: {neg_r1['pool']} probes={neg_r1['probes']}")
    print(f"== neg  R2 pool: {neg_r2['pool']} probes={neg_r2['probes']}")
    print(f"== cov: {json.dumps(cov)}")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-e2-memory-two-slot",
        "note": "E2 Source Memory 双槽（已暴露数据/零 LLM——三臂比较；"
                "仅一个 Control：reserve_exploration_slot）",
        "arms": {
            "A5_hard": {"pool": hard["pool"], "probes": hard["probes"],
                        "receipts": hard["r"].target_support_receipts_used},
            "A5_two_slot": {"pool": two["pool"], "probes": two["probes"],
                            "receipts": two["r"].target_support_receipts_used},
            "A3": {"pool": a3["pool"], "probes": a3["probes"],
                   "receipts": a3["r"].target_support_receipts_used},
            "A5_two_slot_negative_source": {
                "r1": {"pool": neg_r1["pool"], "probes": neg_r1["probes"],
                       "receipts": neg_r1["r"].target_support_receipts_used},
                "r2": {"pool": neg_r2["pool"], "probes": neg_r2["probes"],
                       "receipts": neg_r2["r"].target_support_receipts_used}},
        },
        "target_coverage": cov,
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
