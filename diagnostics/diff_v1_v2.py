# -*- coding: utf-8 -*-
"""diagnostics/diff_v1_v2.py — v1↔v2 响应矩阵逐动作 diff + 供给 headroom 定量（A-29 补录用）。

只读冻结矩阵（results/E1_1_operator_pool_v1 与 E1_1_v2）、零新实验。产出：
①逐动作 bit 级一致性检查（翻案唯一归因链）；②逐 cell×action 均值对照；
③supply headroom：v_none→L1(cell-best)/L2/per-series oracle（A-30e 命名口径：
processing_gain=相对最低处理基线的附加处理价值；routing_gain=L0→L1；详 decision.md 补录）。
运行：项目根（Agent/）下 `python -m SelfEvolvingHarnessTS.diagnostics.diff_v1_v2 [v1_dir v2_dir]`
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
RESULTS = Path(__file__).resolve().parents[1] / "results"
D1 = Path(sys.argv[1]) if len(sys.argv) > 2 else RESULTS / "E1_1_operator_pool_v1"
D2 = Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "E1_1_v2"

v1 = pd.read_csv(D1 / "response_matrix.csv")
v2 = pd.read_csv(D2 / "response_matrix.csv")
print(f"A={D1.name} actions:", sorted(v1["action"].unique()))
print(f"B={D2.name} actions:", sorted(v2["action"].unique()))
print("uids:", v1["uid"].nunique(), "/", v2["uid"].nunique(),
      " same:", set(v1["uid"]) == set(v2["uid"]))

w1 = v1.pivot_table(index=["cell", "origin", "uid"], columns="action", values="oof_nrmse")
w2 = v2.pivot_table(index=["cell", "origin", "uid"], columns="action", values="oof_nrmse")
shared = sorted(set(w1.columns) & set(w2.columns))
j = w1[shared].join(w2[shared], lsuffix="_v1", rsuffix="_v2", how="inner")
print("joined series:", len(j))

print("\n== per-action identity check (same uid, A vs B) ==")
for a in shared:
    d = (j[a + "_v1"] - j[a + "_v2"]).abs()
    print(f"  {a:18s} max|diff|={d.max():.6g}  n_changed(>1e-9)={(d > 1e-9).sum()}/{len(d)}")

print("\n== per-cell x action mean oof_nrmse (A vs B) ==")
p1 = v1.pivot_table(index="cell", columns="action", values="oof_nrmse", aggfunc="mean")
p2 = v2.pivot_table(index="cell", columns="action", values="oof_nrmse", aggfunc="mean")
for cell in p1.index:
    print(f"\n[{cell}]")
    alla = sorted(set(p1.columns) | set(p2.columns))
    rows = []
    for a in alla:
        m1 = p1.loc[cell, a] if a in p1.columns else np.nan
        m2 = p2.loc[cell, a] if a in p2.columns else np.nan
        rows.append((a, m1, m2))
    for a, m1, m2 in sorted(rows, key=lambda r: (r[2] if np.isfinite(r[2]) else 9e9)):
        dtxt = f"{m2 - m1:+8.4f}" if np.isfinite(m1) and np.isfinite(m2) else "     n/a"
        t1 = f"{m1:7.4f}" if np.isfinite(m1) else "  n/a  "
        print(f"  {a:18s} A={t1}  B={m2:7.4f}  delta={dtxt}")

print("\n== SUPPLY/PROCESSING HEADROOM (B): v_none vs oracles ==")
res = []
for cell, s2 in v2.groupby("cell"):
    pw = s2.pivot_table(index=["origin", "uid"], columns="action", values="oof_nrmse")
    mnone = pw["v_none"].mean()
    cellmeans = pw.mean()
    L1a, L1 = cellmeans.idxmin(), cellmeans.min()
    l2num, n = 0.0, 0
    for og, g in pw.groupby(level="origin"):
        l2num += g.mean().min() * len(g)
        n += len(g)
    L2 = l2num / n
    per_series = pw.min(axis=1).mean()
    res.append((cell, mnone, L1, L1a, L2, per_series))
    print(f"  [{cell}] v_none={mnone:.4f}  L1({L1a})={L1:.4f}  L2={L2:.4f}  per-series-oracle={per_series:.4f}")
    print(f"      none->L1={mnone - L1:+.4f}  none->L2={mnone - L2:+.4f}  none->per-series={mnone - per_series:+.4f}")
mn = np.mean([r[1] for r in res]); l1 = np.mean([r[2] for r in res]); ps = np.mean([r[5] for r in res])
print(f"\n  MEAN(cells): v_none={mn:.4f}  L1={l1:.4f}  per-series-oracle={ps:.4f}")
print(f"  processing_gain  v_none -> L1(cell-best) = {mn - l1:+.4f}   (相对最低处理基线的附加处理价值)")
print(f"  pool ceiling     v_none -> per-series    = {mn - ps:+.4f}   (池内上界，含选择噪声)")

print("\n== global single-action means (B) ==")
print(v2.pivot_table(index="action", values="oof_nrmse", aggfunc="mean").sort_values("oof_nrmse").to_string())
