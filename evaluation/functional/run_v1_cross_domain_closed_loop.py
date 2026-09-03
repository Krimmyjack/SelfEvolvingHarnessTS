"""V1 跨域完整链闭环（实验级，零 LLM，2026-08-08）。

审查裁决：NO_APPLICABLE_SOURCE_MEMORY 自动触发现有 Slow Path，输出接回
Target Support → Episode → delayed → 下一轮 Fast Path。Source 超半径经验
= Slow Path"类比材料"（不获得执行权，最终由 Target Support 实测决定）。

审查者裁决（2026-08-08 二轮）：R3 失败近因 = Slow Path 触发条件过宽覆盖了
resolver 的安全序（R3 resolver 已正确判 denoise_stl 为 RISK_PRIOR 排最后）。
批准 (a) 保守模式为主修复、(c) 触发条件修正为对照复跑。

--mode 三变体（各为单行为改动，确定性可复现）：
  baseline     Slow Path 触发 = 无 POSITIVE_PRIOR 且记忆非空 → success family 覆盖
  conservative 触发 = 本地（target 域）存在 delayed 强负（四态 RESTRICTED 类）
               → source 类比材料全部降级：候选序 = 本地正验证（降序）→ 字母序其余；
               无本地候选 → abstain 兜底；无风险信号时行为同 baseline（R1/R2 零退化）
  trigger_fix  触发 = resolver 全 UNKNOWN（无任何适用证据）→ success family；
               有 CONFLICT/RISK 适用证据 → 走 resolver 安全序（R3 反事实验证）

报告含 delayed 明细与 per-op 判定（P3 修复）：每轮 a5/a3_delayed、
a5_slow_mode、首探算子的 resolver 判定。

用法：
  python evaluation/functional/run_v1_cross_domain_closed_loop.py [--mode baseline|conservative|trigger_fix]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import run_v1_target_local_loop as loop  # noqa: E402
import signed_radius as resolver  # noqa: E402

HORIZON = 48
MAX_TARGET_PROBES = 2
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_cross_domain_closed_loop_report.json")

SOURCE_DOMAIN = "noaa"
SOURCE_ORIGINS = (832, 880)
TARGET_DOMAIN = "gefcom"
TARGET_SLICES = [(736, 784), (832, 880), (928, 976)]


def build_noaa_source_memory(root: Path) -> list[Any]:
    config = dict(v6.DATASET_CONFIGS[SOURCE_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    episodes, _ = v1.build_source_memory(
        domain=SOURCE_DOMAIN, roster=roster, values=values, config=config,
        operators=sorted(n for n in v6.OPERATOR_NAMES
                         if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                         and n not in core.CTS_EXCLUDED),
        source_support_origin=SOURCE_ORIGINS[0], source_delayed_origin=SOURCE_ORIGINS[1],
        baseline_cache=baseline_cache,
        context_fn=lambda o: resolver.window_context(values, o, period),
    )
    return episodes


def local_verified_risk(memory: Sequence[Any], m: float) -> set[str]:
    """本地（target 域）delayed 已验证为负的算子（四态 RESTRICTED/CONFLICT 类）。

    审查裁决（2026-08-08 三轮）：与 update_delayed_status 四态状态机对齐——
    delayed_gain < m（含中性偏低，非仅 < -m）判为负（RESTRICTED）。
    """
    return {
        ep.workflow_signature for ep in memory
        if getattr(ep, "domain_namespace", None) == TARGET_DOMAIN
        and isinstance(ep.delayed_response.get("gain"), (int, float))
        and ep.delayed_response.get("evaluated") is True
        and ep.delayed_response["gain"] < m
    }


def success_family(memory: Sequence[Any], m: float) -> list[tuple[str, float]]:
    """Slow Path 类比材料：成功 family = 任一窗口 gain ≥ M 的算子（按最大正 gain 降序）。

    本地已验证风险排除（本地 delayed 强负优先于 Source 正窗口——跨域首跑发现）。
    """
    verified_risk = local_verified_risk(memory, m)
    best: dict[str, float] = {}
    for ep in memory:
        op = ep.workflow_signature
        if op == "identity" or op in verified_risk:
            continue
        gains = [g for g in (ep.support_response.get("gain"), ep.delayed_response.get("gain"))
                 if isinstance(g, (int, float))]
        pos = [g for g in gains if g >= m]
        if pos:
            best[op] = max(best.get(op, 0.0), max(pos))
    return sorted(best.items(), key=lambda t: -t[1])


def local_positive_family(memory: Sequence[Any], m: float) -> list[tuple[str, float]]:
    """本地（target 域）正验证算子（support 或 delayed ≥ M 且非已验证风险），降序。"""
    verified_risk = local_verified_risk(memory, m)
    best: dict[str, float] = {}
    for ep in memory:
        op = ep.workflow_signature
        if op == "identity" or op in verified_risk:
            continue
        if getattr(ep, "domain_namespace", None) != TARGET_DOMAIN:
            continue
        gains = [g for g in (ep.support_response.get("gain"), ep.delayed_response.get("gain"))
                 if isinstance(g, (int, float))]
        pos = [g for g in gains if g >= m]
        if pos:
            best[op] = max(best.get(op, 0.0), max(pos))
    return sorted(best.items(), key=lambda t: -t[1])


def slow_path_order(memory: Sequence[Any], operators: Sequence[str], m: float) -> list[str]:
    """类比推理序：成功 family（值得检查，正 gain 降序，本地已验证风险排除）
    → 其余字母序（需验证，不排除）。"""
    success = success_family(memory, m)
    success_ops = {op for op, _ in success}
    rest = [op for op in sorted(operators) if op not in success_ops]
    return [op for op, _ in success] + rest


def conservative_order(memory: Sequence[Any], operators: Sequence[str], m: float) -> list[str]:
    """保守模式序：source 类比材料全部降级——本地正验证（降序）→ 其余字母序。
    无本地正验证 → 纯字母序（与 A3 收敛）；abstain 由 probe 空结果兜底。"""
    local_pos = local_positive_family(memory, m)
    local_ops = {op for op, _ in local_pos}
    rest = [op for op in sorted(operators) if op not in local_ops]
    return [op for op, _ in local_pos] + rest


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 cross-domain closed loop")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mode", default="conservative",
                        choices=("baseline", "conservative", "trigger_fix"))
    args = parser.parse_args()
    root = args.root.resolve()
    mode = args.mode
    m = core.MATERIAL_THRESHOLD

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    operators = sorted(n for n in v6.OPERATOR_NAMES
                       if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                       and n not in core.CTS_EXCLUDED)

    noaa_source = build_noaa_source_memory(root)
    print(f"== source memory: {SOURCE_DOMAIN} n={len(noaa_source)} mode={mode}")

    from run_w2_operator_scan import _default_params

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
            if g < -m:
                harm += 1
            if g >= m:
                break
        return {"probe_order": probed, "support_gains": gains, "harm": harm}

    a5_local: list[Any] = []
    a3_local: list[Any] = []
    rounds: list[dict[str, Any]] = []

    for round_idx, (ts, td) in enumerate(TARGET_SLICES):
        label = f"R{round_idx + 1}"
        f_support = resolver.window_context(values, ts, period)
        memory = noaa_source + a5_local

        a5_order, a5_signed = resolver.resolve_order(
            query_context=f_support, episodes=memory, operators=operators,
            material_threshold=m)
        a5_counts = a5_signed["summary"]["verdict_counts"]
        a5_slow_mode = "resolver"
        if mode == "baseline":
            if a5_counts[resolver.POSITIVE_PRIOR] == 0 and len(memory) > 0:
                a5_order = slow_path_order(memory, operators, m)
                a5_slow_mode = "analogy"
        elif mode == "conservative":
            if local_verified_risk(memory, m):
                a5_order = conservative_order(memory, operators, m)
                a5_slow_mode = "conservative"
            elif a5_counts[resolver.POSITIVE_PRIOR] == 0 and len(memory) > 0:
                a5_order = slow_path_order(memory, operators, m)
                a5_slow_mode = "analogy"
        elif mode == "trigger_fix":
            if a5_counts[resolver.UNKNOWN] == len(operators) and len(memory) > 0:
                a5_order = slow_path_order(memory, operators, m)
                a5_slow_mode = "analogy"
        a5_r = probe_at(a5_order, ts)

        # A3：无 Source——默认序（R1）+ 本地检索（R2+），同预算
        if round_idx == 0:
            a3_order = list(operators)
        else:
            a3_order, _ = resolver.resolve_order(
                query_context=f_support, episodes=a3_local,
                operators=operators, material_threshold=m)
        a3_r = probe_at(a3_order, ts)

        # 写 Episode（各自臂）
        starts = {"a5": len(a5_local), "a3": len(a3_local)}
        for arm_local, arm_r, arm_name in ((a5_local, a5_r, "A5"), (a3_local, a3_r, "A3")):
            for op, g in zip(arm_r["probe_order"], arm_r["support_gains"]):
                arm_local.append(loop.write_target_episode(
                    domain=TARGET_DOMAIN, op=op,
                    program_steps=[{"op": op, "params": dict(_default_params(op, period))}],
                    support_gain=g, delayed_gain=None, support_context=f_support))
            if not arm_r["probe_order"]:
                arm_local.append(loop.write_abstain_episode(
                    domain=TARGET_DOMAIN, reason=f"{arm_name}_no_valid_plan"))

        # delayed 打开（只更新本轮 Episode）
        if td is not None:
            f_delayed = resolver.window_context(values, td, period)
            for key, arm_local_ in (("a5", a5_local), ("a3", a3_local)):
                new_local = []
                for i, ep in enumerate(arm_local_):
                    if i < starts[key] or ep.workflow_signature == "identity":
                        new_local.append(ep)
                        continue
                    compiled = loop.compiled_from_episode(ep, period)
                    dg = v1.gain_at(roster, values, config, compiled, td, baseline_cache)
                    new_local.append(loop.update_delayed_status(ep, dg, delayed_context=f_delayed)
                                     if dg is not None else ep)
                arm_local_[:] = new_local
        else:
            print(f"[{label}] delayed not opened (boundary {ts})")

        def round_delayed_summary(arm_local: list[Any], start: int) -> dict[str, Any]:
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
            "a5": a5_r, "a3": a3_r,
            "a5_slow_mode": a5_slow_mode,
            "a5_resolver_counts": a5_counts,
            # radius 校准证据（审查裁决 2026-08-08 三轮：证明 radius 由真实独立
            # Context 校准——n_hist 按去重特征向量/origin 计，非 operator/kind 重复计）
            "a5_radius_calibration": {
                "n_historical_contexts": a5_signed["summary"]["n_historical_contexts"],
                "radius_mode": a5_signed["summary"]["radius_mode"],
                "delta_q75": a5_signed["summary"]["delta_q75"],
            },
            "a5_delayed": round_delayed_summary(a5_local, starts["a5"]),
            "a3_delayed": round_delayed_summary(a3_local, starts["a3"]),
            "a5_first_positive_probe": next(
                (i + 1 for i, g in enumerate(a5_r["support_gains"]) if g >= m), None),
            "a3_first_positive_probe": next(
                (i + 1 for i, g in enumerate(a3_r["support_gains"]) if g >= m), None),
            "a5_harm_magnitude": round(sum(-g for g in a5_r["support_gains"] if g < -m), 6),
            "a3_harm_magnitude": round(sum(-g for g in a3_r["support_gains"] if g < -m), 6),
            "a5_local_count": len(a5_local), "a3_local_count": len(a3_local),
        })
        print(f"[{label}] A5 slow={a5_slow_mode} {a5_r} | A3 {a3_r}")

    a5_harm = [r["a5"]["harm"] for r in rounds]
    a3_harm = [r["a3"]["harm"] for r in rounds]
    round_cmp = []
    for a5h, a3h in zip(a5_harm, a3_harm):
        round_cmp.append("better" if a5h < a3h else ("same" if a5h == a3h else "worse"))
    a5_mag = round(sum(r["a5_harm_magnitude"] for r in rounds), 6)
    a3_mag = round(sum(r["a3_harm_magnitude"] for r in rounds), 6)
    savgol_found_a5 = any(
        op == "denoise_savgol" and g >= m
        for r in rounds for op, g in zip(r["a5"]["probe_order"], r["a5"]["support_gains"]))
    savgol_found_a3 = any(
        op == "denoise_savgol" and g >= m
        for r in rounds for op, g in zip(r["a3"]["probe_order"], r["a3"]["support_gains"]))
    worse_rounds = round_cmp.count("worse")

    if worse_rounds == 0 and a5_mag <= a3_mag:
        verdict = "CROSS_DOMAIN_CONTROL_PATH_MECHANISM_PASS"
    elif a5_mag <= a3_mag:
        verdict = "CROSS_DOMAIN_MIXED"
    else:
        verdict = "CROSS_DOMAIN_FAIL"

    print(f"\n== [{mode}] A5 harm: {a5_harm} | A3 harm: {a3_harm} | per-round: {round_cmp}")
    print(f"== [{mode}] harm magnitude: A5={a5_mag} A3={a3_mag}")
    print(f"== [{mode}] savgol headroom: A5={savgol_found_a5} A3={savgol_found_a3}")
    print(f"== [{mode}] first-positive: A5={[r['a5_first_positive_probe'] for r in rounds]} "
          f"A3={[r['a3_first_positive_probe'] for r in rounds]}")
    print(f"== [{mode}] slow modes: {[r['a5_slow_mode'] for r in rounds]}")
    print(f"== [{mode}] radius calibration: "
          f"{[(r['a5_radius_calibration']['n_historical_contexts'], r['a5_radius_calibration']['radius_mode']) for r in rounds]}")
    print(f"== [{mode}] verdict: {verdict}")

    out = root / REPORT_OUT_REL.with_name(
        f"{REPORT_OUT_REL.stem}_{mode}{REPORT_OUT_REL.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-cross-domain-closed-loop",
            "mode": mode,
            "source_domain": SOURCE_DOMAIN, "target_domain": TARGET_DOMAIN,
            "rounds": rounds,
            "a5_harm_curve": a5_harm, "a3_harm_curve": a3_harm,
            "round_comparison": round_cmp,
            "a5_total_harm_magnitude": a5_mag, "a3_total_harm_magnitude": a3_mag,
            "savgol_headroom_found": {"a5": savgol_found_a5, "a3": savgol_found_a3},
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== [{mode}] report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
