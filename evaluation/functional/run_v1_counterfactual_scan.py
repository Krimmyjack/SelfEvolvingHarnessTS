"""P0-1：两步 Workflow leave-one-step-out 反事实 headroom 扫描（零 LLM）。

可选模式：`--from-log` —— 从上次截断的 stdout 日志解析已收集候选（跳过
慢算子块），只做排序 + verify top-N + hit 选择（见 main_from_log）。

审查裁决（二十一）P0 核心：
  "零 LLM 扫描现有合法两步 Workflow，自动找到第一个满足以下条件的
   exposed development 案例：
   - 完整 Workflow 在 Target Support 负向或冲突；
   - 至少一个 leave-one-step-out 版本正向；
   - 所有 Workflow 都满足 H0 verifier；
   - 存在独立 delayed 窗口。"

本脚本是 Runner 侧的确定性扫描（LLM 不可见，之后构造反事实反馈时排除
delayed/正确答案/first fault 标注）。找到的案例供 P0-2/P0-3 使用。

口径：
  - 算子池 = canonical 中非 changes_target_space（v6 forbidden 口径一致）、
    非 external_region（repair_level_shift 需要显式公开区间，不在"可组合
    局部算子"范围）的算子；
  - 参数 = contract_params（public_parameter_schema 默认值/最小合法值，
    period 绑定 24）；
  - 评估 = ScopeExecutor（v6._evaluate 协议，cohort Ridge，baseline 缓存）；
    扫描阶段用无 verify 的快速评估（fast_gain，同执行语义），命中候选后用
    完整窗口级 verifier（H0 0.35）确认——"所有 Workflow 都满足 H0"；
  - 负向或冲突 = gain < MATERIAL_THRESHOLD（0.005）；正向 = gain ≥ 0.005；
  - delayed = origin + 48（独立窗口，数据允许即评估）；
  - 案例选择（Runner 端，不暴露给 LLM）：优先级 ① 全过 H0 verifier
    ② 正 ablation 在 delayed 保持正（无负迁移）③ 正 ablation support
    gain 最大。delayed 数据仅用于选案例，不进 LLM 反馈。

用法：
  python evaluation/functional/run_v1_counterfactual_scan.py
"""

from __future__ import annotations

import json
import re
import sys
import time
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

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
MATERIAL = core.MATERIAL_THRESHOLD  # 0.005
ORIGINS = (928, 832, 880, 976, 784, 736)  # 928 优先：已知冲突切片
VERIFY_TOPK = 6  # 完整 H0 verifier 只对排序后 top 候选执行（昂贵）
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_counterfactual_scan_report.json")

# 非 forbidden（v6 口径：changes_target_space=True 禁止）且非 external_region 的局部算子
POOL = tuple(
    name for name in OPERATOR_NAMES
    if OPERATOR_METADATA[name].get("changes_target_space") is not True
    and OPERATOR_METADATA[name].get("targeting_mode") != "external_region"
)
_PARAM_CACHE: dict[str, dict[str, object]] = {}
_PERIOD_CACHE: dict[str, int] = {}
ACTIVE_PERIOD = PERIOD  # main 按 domain 覆盖（nn5 → 7）


def _params(op: str, period: int | None = None) -> dict[str, object]:
    period = ACTIVE_PERIOD if period is None else period
    key = f"{op}@{period}"
    if key not in _PARAM_CACHE:
        _PARAM_CACHE[key] = wiring.contract_params(op, period)
    return _PARAM_CACHE[key]


def fast_gain(executor: ScopeExecutor, steps: Sequence[tuple[str, Mapping[str, object]]],
              origin: int) -> float | None:
    """无 verifier 快速评估（与 ScopeExecutor.evaluate 同执行语义；
    扫描用；命中候选后再走完整窗口 verifier）。仪器失败返回 None。"""
    try:
        baseline = executor._baseline(origin)
        result = executor._evaluate(
            executor.roster, executor.values, executor._compiled(steps),
            executor.config, origin=origin)
        return float(baseline["mean_smase"] - result["mean_smase"])
    except Exception:
        return None


def verified(executor: ScopeExecutor, steps: Sequence[tuple[str, Mapping[str, object]]],
             origin: int) -> bool:
    v = executor.verify(steps, origin)
    return v.passed


def _op_stats(executor: ScopeExecutor, origin: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for op in POOL:
        out[op] = fast_gain(executor, ((op, dict(_params(op))),), origin)
    return out


def main_from_log() -> int:
    """从截断的 stdout 日志重建候选集 → 排序 → verify top-N → hit 选择。

    日志行格式（脚本 print 的 cand 行）：
      cand: A->B support AB=... A=... B=... delayed AB=... dA=... dB=...
    候选数据完整（含 delayed）；origin 恒为 928（当前扫描起点）。
    """
    root = PROJECT_ROOT
    log = root / "artifacts/functional/e2/w1_counterfactual_scan_stdout.log"
    lines = log.read_text(encoding="utf-8").splitlines()
    candidates: list[dict[str, Any]] = []
    origin = 928
    for line in lines:
        m = re.search(r"cand: (\S+)->(\S+) support AB=(-?[\d.]+) A=(\S+) "
                      r"B=(\S+) delayed AB=(\S+) dA=(\S+) "
                      r"dB=(\S+)", line)
        if not m:
            continue
        a, b, ab, ga, gb, d_ab, d_a, d_b = m.groups()

        def _num(v: str) -> float | None:
            return None if v == "None" else float(v)

        candidates.append({
            "origin": origin,
            "step_a": a,
            "step_b": b,
            "support": {"identity": None,
                        "A_only": _num(ga), "B_only": _num(gb),
                        "A_to_B": _num(ab)},
            "all_verified": False,
            "delayed_origin": origin + HORIZON,
            "delayed": {"A_to_B": _num(d_ab),
                        "A_only": _num(d_a), "B_only": _num(d_b)},
        })
    if not candidates:
        print("== no candidates parsed from log")
        return 1
    print(f"== parsed {len(candidates)} candidates from log (origin {origin})")

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)

    def pick_rank(c: dict[str, Any]) -> tuple[int, float]:
        sup = c["support"]
        positive_ablation_gain = max(
            (g for g in (sup["A_only"], sup["B_only"])
             if g is not None), default=-1e9)
        delayed_gain = max(
            (g for g in (c["delayed"]["A_only"], c["delayed"]["B_only"])
             if g is not None and g >= MATERIAL), default=-1e9)
        return (0 if delayed_gain >= MATERIAL else 1,
                -float(positive_ablation_gain))

    ordered = sorted(candidates, key=pick_rank)
    hit: dict[str, Any] | None = None
    for c in ordered[:VERIFY_TOPK]:
        a, b = c["step_a"], c["step_b"]
        steps_ab = ((a, dict(_params(a))), (b, dict(_params(b))))
        all_verified = bool(
            verified(executor, steps_ab, origin)
            and verified(executor, ((a, dict(_params(a))),), origin)
            and verified(executor, ((b, dict(_params(b))),), origin))
        c["all_verified"] = all_verified
        print(f"   verify top: {a}->{b} all_verified={all_verified}")
        if all_verified:
            hit = c
            break

    report = {
        "experiment_id": "v1-counterfactual-scan",
        "domain": TARGET_DOMAIN,
        "material_threshold": MATERIAL,
        "pool": list(POOL),
        "origins": [origin],
        "origins_by_origin": {
            str(origin): {"single_positive": {}, "candidates": candidates,
                          "hit": hit},
        },
        "first_hit": hit,
        "note": ("candidate set parsed from truncated stdout log (impute_ar "
                 "block skipped — slow operator); hit chosen from collected "
                 "prefix; coverage not exhaustive over the full pair space"),
    }
    verdict = ("COUNTERFACTUAL_HEADROOM_FOUND" if hit is not None
               else "COUNTERFACTUAL_HEADROOM_NONE")
    report["verdict"] = verdict
    print(f"== verdict: {verdict}")
    if hit is not None:
        print(f"== HIT @{hit['origin']}: {hit['step_a']}->{hit['step_b']}")
        print(f"   support: {hit['support']}")
        print(f"   delayed: {hit['delayed']}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


def main() -> int:
    root = PROJECT_ROOT
    origins = ORIGINS
    domain = TARGET_DOMAIN
    period = PERIOD
    origin_override: int | None = None
    for arg in sys.argv[1:]:
        if arg.startswith("--origin="):
            origin_override = int(arg.split("=", 1)[1])
            print(f"== single-origin scan: {origin_override}")
        elif arg.startswith("--domain="):
            domain = arg.split("=", 1)[1]
            print(f"== domain: {domain}")
    if domain == "nn5":
        origins = (origin_override if origin_override is not None else 632,)
        period = 7
    _PERIOD_CACHE[domain] = period
    global ACTIVE_PERIOD
    ACTIVE_PERIOD = period
    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    max_len = max(int(len(v)) for v in values.values())
    print(f"== {TARGET_DOMAIN}: max_len={max_len} period={PERIOD} "
          f"roster={len(roster)} pool={len(POOL)}")
    print(f"== pool: {', '.join(POOL)}")

    report: dict[str, Any] = {
        "experiment_id": "v1-counterfactual-scan",
        "domain": domain,
        "material_threshold": MATERIAL,
        "pool": list(POOL),
        "origins": list(origins),
        "origins_by_origin": {},
    }
    first_hit: dict[str, Any] | None = None
    t0 = time.time()
    for origin in origins:
        if origin + HORIZON > max_len:
            print(f"== origin {origin}: no data for delayed {origin + HORIZON} "
                  f"(max_len={max_len}) — skip")
            continue
        print(f"== origin {origin}: single-op table ...")
        singles = _op_stats(executor, origin)
        positive = [op for op, g in singles.items()
                    if g is not None and g >= MATERIAL]
        negative = [op for op, g in singles.items()
                    if g is not None and g < MATERIAL]
        print(f"   single positive n={len(positive)}: "
              f"{[(op, round(singles[op], 5)) for op in positive]}")
        print(f"   single negative/conflict n={len(negative)}")
        if not positive:
            report["origins_by_origin"][str(origin)] = {
                "single_positive": [], "candidates": [], "hit": None}
            continue

        # 有序对 (A, B)：leave-one-out 至少一个正的必要条件 = A 或 B 单算子正
        # 候选收集只做 fast_gain（support + delayed）；完整 H0 verifier 只对
        # 排序后的 top 候选执行（verify 昂贵：逐窗口 run_pipeline）。
        pair_list = [(a, b) for a in POOL for b in POOL if a != b
                     and (singles.get(a, -1e9) >= MATERIAL
                          or singles.get(b, -1e9) >= MATERIAL)]
        candidates: list[dict[str, Any]] = []
        for a, b in pair_list:
            steps_ab = ((a, dict(_params(a))), (b, dict(_params(b))))
            g_ab = fast_gain(executor, steps_ab, origin)
            # 负向或冲突 = gain ≤ 0（负向 = < 0；抵消 = == 0，单步正被组合吞掉）。
            # 弱正（0 < gain < MATERIAL）不是负向/冲突，排除。
            if g_ab is None or g_ab > 0.0:
                continue
            # 确认至少一个 leave-one-out 正向（快路径：单算子表已含）
            gain_a = singles.get(a)
            gain_b = singles.get(b)
            ab_gains = {
                "identity": None,  # identity 基线由 fast_gain 语义隐含
                "A_only": gain_a,
                "B_only": gain_b,
                "A_to_B": g_ab,
            }
            # 独立 delayed 窗口（Runner 端选择用；不暴露给 LLM）。
            # 只对**正 ablation** 计算其 delayed（对应 Patch 的延迟表现），
            # 负/中性 ablation 的 delayed 无选择价值。
            delayed_origin = origin + HORIZON
            d_ab = fast_gain(executor, steps_ab, delayed_origin)
            d_a = (fast_gain(executor, ((a, dict(_params(a))),), delayed_origin)
                   if gain_a is not None and gain_a >= MATERIAL else None)
            d_b = (fast_gain(executor, ((b, dict(_params(b))),), delayed_origin)
                   if gain_b is not None and gain_b >= MATERIAL else None)
            candidates.append({
                "origin": origin,
                "step_a": a,
                "step_b": b,
                "support": ab_gains,
                "all_verified": False,  # 占位：排序后对 top 候选确认
                "delayed_origin": delayed_origin,
                "delayed": {
                    "A_to_B": d_ab,
                    "A_only": d_a,
                    "B_only": d_b,
                },
            })
            print(f"   cand: {a}->{b} support AB={round(g_ab, 5)} "
                  f"A={None if gain_a is None else round(gain_a, 5)} "
                  f"B={None if gain_b is None else round(gain_b, 5)} "
                  f"delayed AB={None if d_ab is None else round(d_ab, 5)} "
                  f"dA={None if d_a is None else round(d_a, 5)} "
                  f"dB={None if d_b is None else round(d_b, 5)}")

        # 选择：优先级 ① 正 ablation delayed 无负迁移（delayed ≥ MATERIAL）
        # ② 正 ablation support gain 最大（verify 昂贵：只对 top 候选确认）
        def pick_rank(c: dict[str, Any]) -> tuple[int, float]:
            sup = c["support"]
            positive_ablation_gain = max(
                (g for g in (sup["A_only"], sup["B_only"])
                 if g is not None), default=-1e9)
            delayed_gain = max(
                (g for g in (c["delayed"]["A_only"], c["delayed"]["B_only"])
                 if g is not None and g >= MATERIAL), default=-1e9)
            return (0 if delayed_gain >= MATERIAL else 1,
                    -float(positive_ablation_gain))

        ordered = sorted(candidates, key=pick_rank)
        hit: dict[str, Any] | None = None
        for c in ordered[:VERIFY_TOPK]:
            a, b = c["step_a"], c["step_b"]
            steps_ab = ((a, dict(_params(a))), (b, dict(_params(b))))
            all_verified = bool(
                verified(executor, steps_ab, origin)
                and verified(executor, ((a, dict(_params(a))),), origin)
                and verified(executor, ((b, dict(_params(b))),), origin))
            c["all_verified"] = all_verified
            print(f"   verify top: {a}->{b} all_verified={all_verified}")
            if all_verified:
                hit = c
                break
        report["origins_by_origin"][str(origin)] = {
            "single_positive": {
                op: singles[op] for op in positive},
            "candidates": candidates,
            "hit": hit,
        }
        if hit is not None and first_hit is None:
            first_hit = hit
            print(f"\n== HIT @{origin}: {hit['step_a']}->{hit['step_b']}")
            print(f"   support: {hit['support']}")
            print(f"   delayed: {hit['delayed']}")
            print("   (首个全过 H0 verifier 的 leave-one-out headroom 案例)")
            print("== 找到第一个合法案例——停止扫描（P0：'找到第一个满足条件'）")
            break

    elapsed = time.time() - t0
    report["first_hit"] = first_hit
    report["elapsed_seconds"] = round(elapsed, 1)

    verdict = ("COUNTERFACTUAL_HEADROOM_FOUND" if first_hit is not None
               else "COUNTERFACTUAL_HEADROOM_NONE")
    print(f"\n== verdict: {verdict} (elapsed {elapsed:.0f}s)")
    if first_hit is not None:
        print("== 下一步：P0-2 构造反事实反馈（delayed 不暴露给 LLM）")

    out = root / REPORT_OUT_REL
    if len(origins) == 1:
        suffix = f"_{domain}_{origins[0]}" if domain != TARGET_DOMAIN \
            else f"_{origins[0]}"
        out = out.with_name(f"{out.stem}{suffix}{out.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--from-log":
        raise SystemExit(main_from_log())
    raise SystemExit(main())
