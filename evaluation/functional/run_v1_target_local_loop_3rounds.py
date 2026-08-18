"""V1 纵向闭环：3 轮 Target-local Experience 累积（零 LLM，2026-08-08）。

用户裁决下一步：3 轮纵向闭环——互不重叠时间片、A5/A3 等预算，
直接验证项目核心主张：**Target-local Experience 是否随运行积累并持续减少试错**。

切片链（独立性判据：求值窗口 [origin, origin+48) 与所有更早窗口不交叠——
support 须位于前一轮 delayed 终点之后；delayed 未开时须位于前一轮 support 终点之后）。
源先验（A5 独有）取更早的互不重叠窗口，绝不复用 R1 自身窗口（否则 A5 排序含 oracle 成分）：
- gefcom（max_len 1024）：源 (640, 688) → R1 (736, 784) → R2 (832, 880) → R3 (928, 976)，三轮 delayed 全开
- nn5（max_len 791）：源 (536, 584) → R1 (632, 680) → R2 (728, None)（R2 delayed 776+48=824 > 791 不可开）

每轮（A5/A3 等预算 B=2、stop-on-first-positive、stable_gain 排序键）：
- A5 池 = Source prior + 累积本地；A3 池 = 累积本地（各自独立，不跨臂）
- 探测 → 写 Episode → delayed 打开（若可）→ 更新状态 → 累积

产出：每轮 harm/首次正向/探测顺序——效果曲线（逐轮变化）。

用法：
  python evaluation/functional/run_v1_target_local_loop_3rounds.py [--domain gefcom]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_target_local_loop as loop  # noqa: E402
import run_v1_target_local_radius_premise as radius  # noqa: E402 (window_context 单一来源)
import run_v1_signed_radius as resolver  # noqa: E402 (signed relation-aware radius)

MAX_TARGET_PROBES = 2
HORIZON = 48
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_target_local_loop_3rounds_report.json")

# 切片链：(support_origin, delayed_origin|None)；delayed None = 边界不可开
# 独立性：窗口 [origin, origin+HORIZON) 与所有更早窗口不交叠。
SLICES = {
    "gefcom": [(736, 784), (832, 880), (928, 976)],
    "nn5": [(632, 680), (728, None)],
}
# 源先验（A5 独有）：更早的互不重叠窗口。
# gefcom 原 A5/A3 实验 source=(832,880)——若 R1 从 736 起则 source 须前移，
# 取 (640,688)：其窗口 [640,688)/[688,736) 与全部轮次窗口不重叠。
SOURCE_ORIGINS = {
    "gefcom": (640, 688),
    "nn5": (536, 584),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 3-round target-local closed loop")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom", choices=tuple(SLICES))
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain
    slices = SLICES[domain]

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    # Source prior（A5 独有）
    from run_w2_operator_scan import _default_params
    source_episodes, _ = v1.build_source_memory(
        domain=domain, roster=roster, values=values, config=config,
        operators=sorted(n for n in v6.OPERATOR_NAMES
                         if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                         and n not in core.CTS_EXCLUDED),
        source_support_origin=SOURCE_ORIGINS[domain][0],
        source_delayed_origin=SOURCE_ORIGINS[domain][1],
        baseline_cache=baseline_cache,
        context_fn=lambda o: radius.window_context(values, o, period),  # recent/change Context
    )
    print(f"== {domain}: source episodes={len(source_episodes)}, slices={slices}")

    def probe_at(order: Sequence[str], origin: int) -> dict[str, Any]:
        gains: list[float] = []
        probed: list[str] = []
        harm = 0
        for op in order:
            if len(probed) >= MAX_TARGET_PROBES:
                break
            compiled = v1.make_compiled(op, _default_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            if g is None:
                continue
            probed.append(op)
            gains.append(g)
            if g < -core.MATERIAL_THRESHOLD:
                harm += 1
            if g >= core.MATERIAL_THRESHOLD:
                break
        return {"probe_order": probed, "support_gains": gains, "harm": harm}

    a5_local: list[Any] = []
    a3_local: list[Any] = []
    rounds: list[dict[str, Any]] = []

    for round_idx, (ts, td) in enumerate(slices):
        label = f"R{round_idx + 1}"
        # signed relation-aware 半径排序（A5/A3 各自池；历史 < 3 → 弱参考 = 半径∞）。
        # 按 Workflow 聚合：同一 Workflow 每轮只出现一次（修复重复探测）。
        f_support = radius.window_context(values, ts, period)
        default_order = sorted(n for n in v6.OPERATOR_NAMES
                               if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                               and n not in core.CTS_EXCLUDED)
        a5_order, a5_signed = resolver.resolve_order(
            query_context=f_support, episodes=source_episodes + a5_local,
            operators=default_order, material_threshold=core.MATERIAL_THRESHOLD)
        a3_order, a3_signed = resolver.resolve_order(
            query_context=f_support, episodes=a3_local,
            operators=default_order, material_threshold=core.MATERIAL_THRESHOLD)
        a5_r = probe_at(a5_order, ts)
        a3_r = probe_at(a3_order, ts)
        for arm_name, arm_signed in (("A5", a5_signed), ("A3", a3_signed)):
            first = a5_r if arm_name == "A5" else a3_r
            if first["probe_order"]:
                op0 = first["probe_order"][0]
                d = arm_signed["per_op"].get(op0, {})
                print(f"[{label}] {arm_name} first-probe {op0}: signed={d.get('verdict')} "
                      f"pos={[(e['episode_id'], e.get('distance')) for e in d.get('pos_evidence', [])]} "
                      f"neg={[(e['episode_id'], e.get('distance')) for e in d.get('neg_evidence', [])]} "
                      f"mode={arm_signed['summary']['radius_mode']}")

        # 写 Episode（各自手臂）；本轮写入起点记录在 starts（修复 3/4 用）
        starts = {"a5": len(a5_local), "a3": len(a3_local)}
        for arm_local, arm_r, arm_name in ((a5_local, a5_r, "A5"), (a3_local, a3_r, "A3")):
            for op, g in zip(arm_r["probe_order"], arm_r["support_gains"]):
                arm_local.append(loop.write_target_episode(
                    domain=domain, op=op,
                    program_steps=[{"op": op, "params": dict(_default_params(op, period))}],
                    support_gain=g, delayed_gain=None, support_context=f_support))
            if not arm_r["probe_order"]:
                arm_local.append(loop.write_abstain_episode(domain=domain, reason=f"{arm_name}_no_valid_plan"))

        # delayed 打开（若该轮可开）→ 只更新本轮写出的 Episode（修复 3/4：
        # 历史 Episode 的 delayed 不被后续轮次覆盖——每个 delayed 只对应其写入轮）；
        # delayed Context 一并写入 Episode（修复 2/4）
        if td is not None:
            f_delayed = radius.window_context(values, td, period)
            for key, arm_local in (("a5", a5_local), ("a3", a3_local)):
                new_local = []
                for i, ep in enumerate(arm_local):
                    if i < starts[key] or ep.workflow_signature == "identity":
                        new_local.append(ep)
                        continue
                    compiled = loop.compiled_from_episode(ep, period)
                    dg = v1.gain_at(roster, values, config, compiled, td, baseline_cache)
                    new_local.append(
                        loop.update_delayed_status(ep, dg, delayed_context=f_delayed)
                        if dg is not None else ep)
                arm_local[:] = new_local
        else:
            print(f"[{label}] delayed not opened (boundary slice {ts})")

        # 审查发现 4：本轮写入 Episode 的 delayed/relation/status（承重指标）
        def round_episode_summary(arm_local: list[Any], start: int) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for ep in arm_local[start:]:
                if ep.workflow_signature == "identity":
                    continue
                out[ep.workflow_signature] = {
                    "delayed_gain": ep.delayed_response.get("gain"),
                    "local_status": ep.local_status,
                    "relation": ep.relation,
                }
            return out

        rounds.append({
            "round": label,
            "slice": {"support": ts, "delayed": td},
            "a5": a5_r,
            "a3": a3_r,
            "a5_signed": a5_signed,
            "a3_signed": a3_signed,
            "a5_local_count": len(a5_local),
            "a3_local_count": len(a3_local),
            "a5_delayed": round_episode_summary(a5_local, starts["a5"]),
            "a3_delayed": round_episode_summary(a3_local, starts["a3"]),
            "a5_first_positive_probe": next(
                (i + 1 for i, g in enumerate(a5_r["support_gains"])
                 if g >= core.MATERIAL_THRESHOLD), None),
            "a3_first_positive_probe": next(
                (i + 1 for i, g in enumerate(a3_r["support_gains"])
                 if g >= core.MATERIAL_THRESHOLD), None),
            "a5_harm_magnitude": round(sum(
                -g for g in a5_r["support_gains"] if g < -core.MATERIAL_THRESHOLD), 6),
            "a3_harm_magnitude": round(sum(
                -g for g in a3_r["support_gains"] if g < -core.MATERIAL_THRESHOLD), 6),
        })
        print(f"[{label}] A5: {a5_r} (local={len(a5_local)}) | "
              f"A3: {a3_r} (local={len(a3_local)})")

    # 效果曲线：逐轮 harm
    a5_harm_curve = [r["a5"]["harm"] for r in rounds]
    a3_harm_curve = [r["a3"]["harm"] for r in rounds]
    a5_first_curve = [r["a5"]["probe_order"][:1] for r in rounds]
    print(f"\n== A5 harm curve: {a5_harm_curve}")
    print(f"== A3 harm curve: {a3_harm_curve}")
    print(f"== A5 first-probe curve: {a5_first_curve}")

    # 断言（核心主张：本地经验累积 → 试错不增/下降）
    monotone_no_increase = all(a5_harm_curve[i + 1] <= a5_harm_curve[i]
                               for i in range(len(a5_harm_curve) - 1))
    strict_decrease = len(a5_harm_curve) >= 2 and a5_harm_curve[-1] < a5_harm_curve[0]
    harm_floor_zero = a5_harm_curve[-1] == 0

    # 独立性断言（布尔，收尾②同款）：源先验 + 逐轮窗口全部不重叠
    prev_end = SOURCE_ORIGINS[domain][1] + HORIZON
    max_len = max(int(len(v)) for v in values.values())
    slice_independent = True
    for ts, td in slices:
        if ts < prev_end:
            slice_independent = False
        if td is not None:
            if td < ts + HORIZON or td + HORIZON > max_len:
                slice_independent = False
            prev_end = td + HORIZON
        else:
            prev_end = ts + HORIZON

    # 本地经验池随轮次只增不减（两臂各自）；且存在 delayed 验证过的 Episode。
    # 注：不等价于 A5 池 ≥ A3 池——stop-on-first-positive 下更快命中正向的臂
    # 探测更少、Episode 更少，那是成功而非失败。
    counts_a5 = [r["a5_local_count"] for r in rounds]
    counts_a3 = [r["a3_local_count"] for r in rounds]
    counts_non_decreasing = (
        all(counts_a5[i + 1] >= counts_a5[i] for i in range(len(counts_a5) - 1))
        and all(counts_a3[i + 1] >= counts_a3[i] for i in range(len(counts_a3) - 1))
    )
    delayed_verified = any(ep.evidence_level == "DELAYED" for ep in a5_local + a3_local)

    checks = {
        "harm_non_increasing": monotone_no_increase,
        "harm_decreased_over_rounds": strict_decrease,
        "harm_floor_zero": harm_floor_zero,
        "experience_reduction_claim": strict_decrease or harm_floor_zero,
        "a3_control_no_false_gain": a3_harm_curve[-1] >= a5_harm_curve[-1],
        "local_accumulated": counts_non_decreasing,
        "delayed_verified_accumulated": delayed_verified,
        "slice_independent": slice_independent,
    }
    all_pass = all(checks.values())
    print(f"\n== checks: {checks}")

    # 主比较（审查发现 1：不绕开 A5/A3 逐轮轨迹比较——A5 自身单调不能包装成胜出）
    round_cmp: list[str] = []
    for r in rounds:
        a5h, a3h = r["a5"]["harm"], r["a3"]["harm"]
        round_cmp.append("better" if a5h < a3h else ("same" if a5h == a3h else "worse"))
    worse_rounds = round_cmp.count("worse")
    a5_mag = round(sum(r["a5_harm_magnitude"] for r in rounds), 6)
    a3_mag = round(sum(r["a3_harm_magnitude"] for r in rounds), 6)
    if worse_rounds == 0 and a5_mag <= a3_mag:
        verdict = "PASS"  # A5 每轮不差于 A3 且 harm 幅度不超
    elif a5_mag <= a3_mag:
        verdict = "MIXED"  # 有轮次更差但幅度更小（轻微失败换取灾难性负迁移消除）
    else:
        verdict = "FAIL"
    print(f"== A5 vs A3 per-round: {round_cmp} (worse={worse_rounds})")
    print(f"== harm magnitude: A5={a5_mag} A3={a3_mag}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL.with_name(f"{REPORT_OUT_REL.stem}_{domain}{REPORT_OUT_REL.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-target-local-loop-3rounds",
            "domain": domain,
            "slices": [{"support": ts, "delayed": td} for ts, td in slices],
            "rounds": rounds,
            "a5_harm_curve": a5_harm_curve,
            "a3_harm_curve": a3_harm_curve,
            "checks": checks,
            "round_comparison_a5_vs_a3": round_cmp,
            "a5_total_harm_magnitude": a5_mag,
            "a3_total_harm_magnitude": a3_mag,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
