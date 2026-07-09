# -*- coding: utf-8 -*-
"""diagnostics/boundary_diag.py — S0.7-8 边界伪影定量诊断（A-31b，评审第十一轮）。

复现"零填充 vs symmetric 镜像"对 v_median 的影响：
1) 单元级：MA(mode='same') / medfilt 零填充端点失真复算；
2) 语料级：v_median 两种边界语义的 processed-history 末端失真 + OOF nRMSE 逐 uid 配对差。

⚠ 历史注记：本脚本首跑于 2026-07-03（当时 `denoise_median` 仍是零填充 medfilt），实测
   v_median 均值 OOF 1.4755(零填充)→1.3131(reflect)，坐实 S0.7-8。S0.7-8 修复合入后，
   代码库默认已是 symmetric——本脚本改为显式对比"人为零填充版"与"当前修复版"，
   结论可随时复查。不写任何 results/。
运行：项目根（Agent/）下 `python -m SelfEvolvingHarnessTS.diagnostics.boundary_diag`
"""
import io
import sys
import time

import numpy as np
import scipy.signal as sps
from scipy.ndimage import median_filter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from SelfEvolvingHarnessTS.run_variance_decomp import (            # noqa: E402
    build_corpus, assign_cells, build_cell_cache, _oof_losses, FrozenProbe)
from SelfEvolvingHarnessTS.run_main_table import fixed_harness_variants  # noqa: E402
from SelfEvolvingHarnessTS.fast_path.pipeline import process as fast_process  # noqa: E402
from SelfEvolvingHarnessTS.operators import s1_denoise as _den    # noqa: E402
from SelfEvolvingHarnessTS.operators._common import moving_average  # noqa: E402

# ── 1) 单元级复算 ────────────────────────────────────────────────
print("== 1) 端点失真单元复算 ==")
const = np.ones(400)
print("  旧 MA(np.convolve mode='same') on const-1, endpoint:",
      {w: round(float(np.convolve(const, np.ones(w) / w, mode='same')[-1]), 3) for w in (9, 15, 25)})
print("  修复 MA(symmetric) on const-1, endpoint:",
      {w: round(float(moving_average(const, w)[-1]), 3) for w in (9, 15, 25)})
trend = np.linspace(1.0, 2.0, 400)
print("  旧 medfilt 零填充 on 1→2 trend, last value:",
      {w: round(float(sps.medfilt(trend, w)[-1]), 4) for w in (5, 9, 25)})
print("  修复 median(symmetric) on 1→2 trend, last value:",
      {w: round(float(_den.denoise_median(trend, window=w)[-1]), 4) for w in (5, 9, 25)})

# ── 2) 语料级：人为零填充版 vs 当前修复版 ───────────────────────
_fixed_median = _den.denoise_median


def _zero_pad_median(x, window: int = 5, **_):
    """复刻 S0.7-8 修复前的行为（scipy.signal.medfilt 零填充），供对照。"""
    from SelfEvolvingHarnessTS.operators._common import as_1d, interp_nan
    y = interp_nan(as_1d(x))
    w = max(1, int(window))
    if w <= 1:
        return y
    return sps.medfilt(y, kernel_size=w if w % 2 == 1 else w + 1)


def _swap_median(fn):
    """同时换掉 s1_denoise 与 registry/TOOL_REGISTRY 引用（executor 走 TOOL_REGISTRY）。"""
    from SelfEvolvingHarnessTS.operators import registry as reg
    _den.denoise_median = fn
    reg.TOOL_REGISTRY["denoise_median"] = fn


t0 = time.time()
corpus = build_corpus(20)                       # 与 v2 相同：320 series
cells, _snr = assign_cells(corpus)
vm = fixed_harness_variants("forecast")
fp = FrozenProbe()

print("\n== 2) v_median 零填充(旧) vs symmetric(现)：末端失真 + OOF 配对差 ==")
print(f"{'cell':26s} {'tail|Δ|last2':>12s} {'mid|Δ|':>8s} {'旧zero':>7s} {'现sym':>7s} {'Δ(旧-现)':>10s} {'v_none':>7s}")
agg = []
for cell, series in sorted(cells.items()):
    tail_d, mid_d = [], []
    for rs in series:
        _swap_median(_zero_pad_median)
        cur = np.asarray(fast_process(rs.history, "forecast", vm["v_median"], store=None)[1], float)
        _swap_median(_fixed_median)
        fix = np.asarray(fast_process(rs.history, "forecast", vm["v_median"], store=None)[1], float)
        if cur.shape == fix.shape and np.all(np.isfinite(cur)) and np.all(np.isfinite(fix)):
            tail_d.append(float(np.mean(np.abs(cur[-2:] - fix[-2:]))))
            m = slice(len(cur) // 4, 3 * len(cur) // 4)
            mid_d.append(float(np.mean(np.abs(cur[m] - fix[m]))))
    _swap_median(_zero_pad_median)
    ac_cur, uids_cur, _ = build_cell_cache(fp, series, {"v_median": vm["v_median"], "v_none": vm["v_none"]})
    _swap_median(_fixed_median)
    ac_fix, uids_fix, _ = build_cell_cache(fp, series, {"v_median": vm["v_median"]})
    common = sorted(set(uids_cur) & set(uids_fix))
    fold = {u: i % 5 for i, u in enumerate(common)}
    l_cur = _oof_losses(ac_cur["v_median"], common, fold, 5)
    l_fix = _oof_losses(ac_fix["v_median"], common, fold, 5)
    l_non = _oof_losses(ac_cur["v_none"], common, fold, 5)
    d = np.array([l_cur[u] - l_fix[u] for u in common])
    mc, mf, mn = (float(np.mean([l[u] for u in common])) for l in (l_cur, l_fix, l_non))
    se = float(np.std(d, ddof=1) / np.sqrt(len(d)))
    print(f"{cell:26s} {np.mean(tail_d):12.4f} {np.mean(mid_d):8.5f} {mc:7.4f} {mf:7.4f} "
          f"{mc - mf:+10.4f} {mn:7.4f}   (paired SE {se:.4f})")
    agg.append((mc, mf, float(np.mean(d))))

print(f"\n  MEAN(cells): 旧zero={np.mean([a[0] for a in agg]):.4f}  现sym={np.mean([a[1] for a in agg]):.4f}  "
      f"Δ={np.mean([a[2] for a in agg]):+.4f}")
print("  参考锚（2026-07-03 首跑）：旧 1.4755 / 修复 1.3131 / Δ +0.1624，中段 bit 级一致（纯边界效应）。")
print(f"[{time.time() - t0:.1f}s]")
