"""run_s2_transfer.py — ★v4 S2 前向迁移曲线（读 S1 的 forward_transfer JSONL → 曲线+判据+图）。

承 S1_Implementation_Plan §B.5 / Refactor_Continual_TaskReadiness_v4 S2（P0）。S1 跑完三 mode
（run_stream_s1.py 落 forward_transfer_{scratch,frozen,updating}.jsonl）后跑本脚本，产出：
  • 终端表：per-(mode, k) 的 ttr(med/max) / readiness@budget / ready_frac / reval_demote。
  • headline 判据：C(updating, memory-on) vs A(scratch, memory-off) 前向迁移是否成立 +
    三 bootstrap 分解 B−A(记忆价值)/C−B(继续更新价值) + 负迁移护栏。
  • 图（matplotlib 可用时）：time_to_readiness(k) 与 readiness@budget(k) 双面板，C/A(/B) 曲线。
  • s2_transfer.json：曲线点 + 判据（机读，喂论文表/图）。

用法：
  python -m SelfEvolvingHarnessTS.run_s2_transfer --in-dir runs/s1_demo
  python -m SelfEvolvingHarnessTS.run_s2_transfer --in-dir runs/s1 --no-plot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# Windows 控制台默认 GBK 编不出 −/→/⚠ 等字符 → 强制 utf-8，避免 print 崩 UnicodeEncodeError。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from .slow_path.forward_transfer import (
    load_transfer_log, build_curves, forward_transfer_verdict, DomainPoint,
    MEMORY_ON, MEMORY_OFF, FROZEN,
)

MODES = ("updating", "frozen", "scratch")


def _load(in_dir: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for mode in MODES:
        path = os.path.join(in_dir, f"forward_transfer_{mode}.jsonl")
        if os.path.exists(path):
            out[mode] = load_transfer_log(path)
    if not out:
        raise FileNotFoundError(f"{in_dir} 下没有 forward_transfer_*.jsonl（先跑 run_stream_s1.py）")
    return out


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_table(curves: Dict[str, List[DomainPoint]]) -> None:
    print("\n=== S2 前向迁移曲线（per mode × domain k）===")
    print(f"{'mode':<9}{'k':>3} {'domain':<12}{'ver':>4}{'cells':>6}"
          f"{'ttr_med':>9}{'ttr_max':>8}{'readi@bud':>11}{'ready%':>8}{'demote':>7}")
    for mode in ("updating", "frozen", "scratch"):
        for p in curves.get(mode, []):
            print(f"{mode:<9}{p.k:>3} {p.domain:<12}{p.harness_version:>4}{p.n_cells:>6}"
                  f"{_fmt(p.ttr_median):>9}{_fmt(p.ttr_max):>8}{_fmt(p.readiness_median):>11}"
                  f"{p.ready_frac*100:>7.0f}%{p.n_reval_demote:>7}")


def print_verdict(verdict: dict) -> None:
    print("\n=== S2 headline 判据（C=updating/memory-on vs A=scratch/memory-off）===")
    print(f"  共享 domain: {verdict['shared_domains']}  (有限 readiness 差分点={verdict['n_finite_readiness_deltas']})")
    print(f"  mean readiness(C−A) = {_fmt(verdict['mean_readiness_C_minus_A'])}  "
          f"(>0 ⇒ 记忆助益就绪度)")
    print(f"  mean ttr_gain(A−C)  = {_fmt(verdict['mean_ttr_gain_A_minus_C'])}  "
          f"(>0 ⇒ C 更快达标)")
    print(f"  memory_helps        = {verdict['memory_helps']}")
    print(f"  no_degradation      = {verdict['no_degradation']}  (C 末域 readiness 不显著低于首域)")
    print(f"  discriminative      = {verdict['discriminative']}  (C/A 有非平凡分离信号)")
    print(f"  → forward_transfer_supported = {verdict['forward_transfer_supported']}  "
          f"(None=无分离信号不可结论)")
    print(f"  负迁移护栏 fired = {verdict['neg_transfer_guardrail_fired']}  "
          f"(C 累计 reval_demote={verdict['total_reval_demote_C']})")
    if verdict["per_k"]:
        print("\n  三 bootstrap 分解（readiness@budget 代理；ΔPerf 双报告留 S4）：")
        print(f"    {'k':>3} {'domain':<12}{'B−A(记忆)':>12}{'C−B(更新)':>12}")
        for r in verdict["per_k"]:
            print(f"    {r['k']:>3} {r['domain']:<12}"
                  f"{_fmt(r['memory_value_B_minus_A']):>12}{_fmt(r['update_value_C_minus_B']):>12}")
    note = []
    if verdict["n_finite_readiness_deltas"] == 0:
        note.append("⚠ 无有限 readiness 差分（cell 全饱和/无 headroom）→ 仅 demo；真信号需 LLM×真实域跑。")
    elif not verdict["discriminative"]:
        note.append("⚠ C/A 差分全在容忍带内（饱和平局，无分离信号）→ supported 置 None；"
                    "真信号需更难域/更紧预算（让 from-scratch 来不及达标）+ LLM×真实域。")
    if verdict["no_degradation"] is None:
        note.append("⚠ C 有限 readiness 点 <2 → 无法判退化（需 ≥2 个非退化 domain）。")
    for n in note:
        print(f"  {n}")


def plot(curves: Dict[str, List[DomainPoint]], out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    style = {MEMORY_ON: ("C(updating)", "o-", "#1f77b4"),
             FROZEN: ("B(frozen)", "s--", "#7f7f7f"),
             MEMORY_OFF: ("A(scratch)", "^-", "#d62728")}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, (label, ls, color) in style.items():
        pts = curves.get(mode, [])
        if not pts:
            continue
        ks = [p.k for p in pts]
        ax1.plot(ks, [p.ttr_median if p.ttr_median is not None else float("nan") for p in pts],
                 ls, color=color, label=label)
        ax2.plot(ks, [p.readiness_median if p.readiness_median is not None else float("nan") for p in pts],
                 ls, color=color, label=label)
    ax1.set_title("time-to-readiness(k)  (lower=faster)")
    ax1.set_xlabel("domain index k"); ax1.set_ylabel("median rounds to readiness")
    ax2.set_title("readiness@budget(k)  (higher=more ready)")
    ax2.set_xlabel("domain index k"); ax2.set_ylabel("median readiness")
    ax2.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.5)
    for ax in (ax1, ax2):
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("S2 forward transfer: memory-on (C) vs memory-off (A)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="S2 前向迁移曲线分析（读 S1 JSONL）")
    ap.add_argument("--in-dir", default="runs/s1", help="含 forward_transfer_*.jsonl 的目录")
    ap.add_argument("--out", default="", help="输出 json（默认 <in-dir>/s2_transfer.json）")
    ap.add_argument("--no-plot", action="store_true", help="跳过 matplotlib 出图")
    ap.add_argument("--tol", type=float, default=0.05, help="退化/助益容忍带")
    args = ap.parse_args()

    logs = _load(args.in_dir)
    print(f"[S2] modes loaded: {sorted(logs)} from {args.in_dir}")
    curves = build_curves(logs)
    verdict = forward_transfer_verdict(curves, tol=args.tol)

    print_table(curves)
    print_verdict(verdict)

    out_json = args.out or os.path.join(args.in_dir, "s2_transfer.json")
    payload = {"curves": {m: [p.to_dict() for p in pts] for m, pts in curves.items()},
               "verdict": verdict}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[S2] machine-readable → {out_json}")

    if not args.no_plot:
        out_png = os.path.join(args.in_dir, "s2_transfer.png")
        if plot(curves, out_png):
            print(f"[S2] figure → {out_png}")
        else:
            print("[S2] matplotlib 不可用，跳过出图（--no-plot 可静默）")


if __name__ == "__main__":
    main()
