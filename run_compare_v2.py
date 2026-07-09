"""run_compare_v2.py — 汇总 S1 run 目录：官方(judge-aware) S2 判据 + relative-to-raw damage 交叉验证。

用法：python -m SelfEvolvingHarnessTS.run_compare_v2 runs/s1_flash_v2 runs/s1_pro_chronos_v2 ...
每个目录读 forward_transfer_{updating,frozen,scratch}.jsonl。
  • 官方判据：forward_transfer.analyze（readiness 已由 deploy_stream 用 judge-aware 锚算好）。
  • damage：j_cur−j_raw（越低越好，0=恢复到 raw），按 cell 比 C(updating)/A(scratch)/B(frozen)；
    对 FM 判官（readiness 可能仍因两参照同值而 nan）提供直读信号。
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .slow_path.forward_transfer import load_transfer_log, analyze, UPDATING, FROZEN, SCRATCH

MODES = (UPDATING, FROZEN, SCRATCH)
TAG = {UPDATING: "C(updating)", FROZEN: "B(frozen)", SCRATCH: "A(scratch)"}


def _load(run_dir: str) -> Dict[str, List[dict]]:
    out = {}
    for m in MODES:
        p = os.path.join(run_dir, f"forward_transfer_{m}.jsonl")
        if os.path.exists(p):
            out[m] = load_transfer_log(p)
    return out


def _damage_by_cell(rows: List[dict]) -> Dict[str, float]:
    """(cell) → mean(j_cur − j_raw) over k（damage；越低越好）。"""
    acc = defaultdict(list)
    for r in rows:
        jr, jc = r.get("j_raw"), r.get("j_cur")
        if jr is not None and jc is not None and math.isfinite(jr) and math.isfinite(jc):
            acc[f"{r.get('k')}:{r.get('cell')}"].append(jc - jr)
    return {c: float(np.mean(v)) for c, v in acc.items() if v}


def report(run_dir: str) -> None:
    logs = _load(run_dir)
    print(f"\n{'='*78}\n### {run_dir}\n{'='*78}")
    if UPDATING not in logs or SCRATCH not in logs:
        print("  (缺 updating/scratch 日志，跳过)"); return

    res = analyze(logs)
    v = res["verdict"]
    print("\n-- 官方 S2 判据（judge-aware readiness）--")
    print(f"  forward_transfer_supported = {v['forward_transfer_supported']}  "
          f"discriminative={v['discriminative']}")
    print(f"  mean readiness C−A = {v['mean_readiness_C_minus_A']}   "
          f"mean ttr_gain A−C = {v['mean_ttr_gain_A_minus_C']}")
    print(f"  memory_helps={v['memory_helps']}  no_degradation={v['no_degradation']}  "
          f"neg_transfer_guardrail_fired={v['neg_transfer_guardrail_fired']} "
          f"(demote_C={v['total_reval_demote_C']})")
    print("  per-domain readiness (C / A / C−A) + 三bootstrap (B−A / C−B):")
    for pk in v["per_k"]:
        def f(x): return f"{x:+.3f}" if isinstance(x, (int, float)) and math.isfinite(x) else str(x)
        print(f"    k={pk['k']} {pk['domain']:<16} C={f(pk['readiness_C'])} A={f(pk['readiness_A'])} "
              f"C−A={f(pk['readiness_C_minus_A'])}  B−A={f(pk['memory_value_B_minus_A'])} "
              f"C−B={f(pk['update_value_C_minus_B'])}  demote_C={pk['n_reval_demote_C']}")

    # relative-to-raw damage 交叉验证（对 readiness 仍 nan 的 cell 也有信号）
    dmg = {m: _damage_by_cell(logs[m]) for m in MODES if m in logs}
    cells = sorted(set().union(*[set(d) for d in dmg.values()]))
    print("\n-- relative-to-raw damage  j_cur−j_raw（越低越好，<0=胜过raw）--")
    cA, cC, cB = [], [], []
    for cell in cells:
        a = dmg.get(SCRATCH, {}).get(cell); c = dmg.get(UPDATING, {}).get(cell); b = dmg.get(FROZEN, {}).get(cell)
        def f(x): return f"{x:+.3f}" if x is not None else "  --  "
        better = ""
        if a is not None and c is not None and abs(c - a) > 1e-4:
            better = "C<A✓" if c < a else "A<C✗"
        print(f"    {cell:<28} C={f(c)} A={f(a)} B={f(b)}  {better}")
        if a is not None: cA.append(a)
        if c is not None: cC.append(c)
        if b is not None: cB.append(b)
    if cC and cA:
        print(f"  mean damage  C={np.mean(cC):+.3f}  A={np.mean(cA):+.3f}  "
              f"B={np.mean(cB):+.3f}" if cB else f"  mean damage  C={np.mean(cC):+.3f}  A={np.mean(cA):+.3f}")
        print(f"  → memory {'helps' if np.mean(cC) < np.mean(cA) else 'hurts'} "
              f"(C−A damage = {np.mean(cC)-np.mean(cA):+.3f})")


def main():
    dirs = sys.argv[1:] or ["runs/s1_flash_v2", "runs/s1_pro_v2",
                            "runs/s1_flash_chronos_v2", "runs/s1_pro_chronos_v2"]
    for d in dirs:
        if os.path.isdir(d):
            report(d)
        else:
            print(f"\n(跳过缺失目录 {d})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
