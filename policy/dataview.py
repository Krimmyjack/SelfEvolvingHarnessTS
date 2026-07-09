"""policy/dataview.py — DataView：LLM composer 的 tool-mediated、history-only 信息面
（第三十三轮 DataView 决定；prereg_skill_slice.md §4）。

修复 round-32 发现的"compose_llm 不见原序列"：LLM 臂不 raw 直塞，而是经标准化视图看数据。
边界：构造 API 只收 history（assert size=CUT）；future/clean/L_test/grounded outcome
物理不可达；块级构造（无流历史→决策与排列位置无关）。

视图：核心 {structure, mask, skills, policy} + 可请求 {window, period, decomp}。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..conditioning.period import robust_period_diag
from ..run_variance_decomp import CUT
from .skills import skill_cards_text

CORE_VIEWS = ("structure", "mask", "skills", "policy")
REQUESTABLE_VIEWS = ("window", "period", "decomp")
_XP = ("period", "trend_strength", "seasonal_strength", "acf1",
       "stationarity_adf", "spectral_entropy", "lumpiness", "outlier_density")


def _interp(h: np.ndarray) -> np.ndarray:
    """感知专用线性插值（P1a 语义：只为看结构，不产出数据）。"""
    h = np.asarray(h, float).ravel()
    m = np.isfinite(h)
    if m.all():
        return h.copy()
    if not m.any():
        return np.zeros_like(h)
    idx = np.arange(h.size)
    out = h.copy()
    out[~m] = np.interp(idx[~m], idx[m], h[m])
    return out


def _gap_runs(h: np.ndarray) -> List[int]:
    m = ~np.isfinite(np.asarray(h, float).ravel())
    runs, cur = [], 0
    for v in m:
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _rep_uids(rows: List[dict]) -> List[str]:
    """3 个代表 uid：SNR 最低 / 中位 / 最高（确定性）。"""
    srt = sorted(rows, key=lambda r: (r["snr"], r["uid"]))
    return [srt[0]["uid"], srt[len(srt) // 2]["uid"], srt[-1]["uid"]]


def build_block_views(rows: List[dict], hist_of: Dict[str, np.ndarray],
                      frozen_picks: List[str], abstains: List[bool]) -> Dict[str, str]:
    """块级全视图（core+requestable 一次构建，何时给 LLM 由 composer 分段控制）。

    rows=gym 行（不含 L_test 的使用——只读 snr/miss_rate/X_p/cell/uid）；
    hist_of=代表 uid 的观测史（assert=CUT，G-B 同款泄漏守卫）。
    """
    for h in hist_of.values():
        assert np.asarray(h).size == CUT, f"DataView 只接受判官口径观测史 CUT={CUT}"
    reps = _rep_uids(rows)
    X = {f: np.array([r["X_p"][i] for r in rows], float) for i, f in enumerate(_XP)}
    snr = np.array([r["snr"] for r in rows], float)
    miss = np.array([r["miss_rate"] for r in rows], float)
    cells: Dict[str, int] = {}
    for r in rows:
        cells[r["cell"]] = cells.get(r["cell"], 0) + 1

    views: Dict[str, str] = {}
    views["structure"] = (
        f"block n={len(rows)}; cell bins: "
        + ", ".join(f"{k}×{v}" for k, v in sorted(cells.items())) + "\n"
        + f"SNR dB median={np.median(snr):.1f} [q10 {np.quantile(snr,0.1):.1f}, "
          f"q90 {np.quantile(snr,0.9):.1f}]; missing median={np.median(miss):.2f} "
          f"max={miss.max():.2f}\n"
        + "; ".join(f"{f} med={np.median(X[f]):.2f}"
                    for f in ("period", "trend_strength", "seasonal_strength",
                              "acf1", "spectral_entropy", "outlier_density")))
    runs_all = [g for u in reps for g in _gap_runs(hist_of[u])]
    views["mask"] = (
        f"missing_rate: median={np.median(miss):.2f}, n(miss>0)={int((miss>0).sum())}/{len(rows)}\n"
        + (f"gap runs (3 rep series): n={len(runs_all)}, max_len={max(runs_all)}, "
           f"mean_len={np.mean(runs_all):.1f}" if runs_all else "gap runs: none (fully observed)"))
    views["skills"] = skill_cards_text()
    hist_p, tot = {}, max(1, len(frozen_picks))
    for a in frozen_picks:
        hist_p[a] = hist_p.get(a, 0) + 1
    ab_txt = f"abstain_rate={np.mean(abstains):.2f}" if abstains else "abstain info: n/a"
    views["policy"] = ("frozen incumbent picks: "
                       + ", ".join(f"{a}×{c}" for a, c in sorted(hist_p.items(), key=lambda kv: -kv[1]))
                       + f"; {ab_txt}")

    win_lines = []
    for u in reps:
        h = _interp(hist_of[u])[-96:]
        mu, sd = h.mean(), h.std() or 1.0
        z = np.round((h[::2] - mu) / sd, 2)
        win_lines.append(f"{u.split(':', 2)[2]}: {z.tolist()}")
    views["window"] = ("last-96 window, z-scored, downsampled x2 (48 vals), 3 rep series "
                       "(low/mid/high SNR):\n" + "\n".join(win_lines))

    per_lines = []
    for u in reps:
        h = _interp(hist_of[u])
        d = robust_period_diag(h)
        f = np.abs(np.fft.rfft(h - h.mean()))
        f[0] = 0.0
        top = np.argsort(f)[-3:][::-1]
        peaks = [f"P≈{h.size/b:.0f}" for b in top if b > 0]
        per_lines.append(f"{u.split(':', 2)[2]}: detected={d['period']:.0f} "
                         f"(cand={d['cand_period']:.0f}, peak_ratio={d['peak_ratio']:.1f}, "
                         f"acf@peak={d['acf_at_peak']:.2f}); rfft top-3: {', '.join(peaks)}")
    views["period"] = "\n".join(per_lines)

    dec_lines = []
    for u in reps:
        h = _interp(hist_of[u])
        t = np.arange(h.size)
        slope = np.polyfit(t, h, 1)[0] * 100
        p = int(robust_period_diag(h)["period"]) or 24
        detr = h - np.polyval(np.polyfit(t, h, 1), t)
        phase = np.array([detr[i::p].mean() for i in range(p)])
        seas_amp = float(phase.max() - phase.min())
        resid = detr - np.tile(phase, h.size // p + 1)[:h.size]
        dec_lines.append(f"{u.split(':', 2)[2]}: trend_slope={slope:+.3f}/100steps, "
                         f"seasonal_amp@P{p}={seas_amp:.2f}, resid_std={resid.std():.2f}")
    views["decomp"] = "\n".join(dec_lines)
    return views


# ════════════════════════════ v2：可靠性标注 + featurizer（prereg_skill_slice_v2 §3/§4）════════════════════════════
def build_block_views_v2(rows: List[dict], hist_of: Dict[str, np.ndarray],
                         frozen_picks: List[str], abstains: List[bool]) -> Dict[str, str]:
    """v1 全视图 + structure 的确定性可靠性标注（类级修复：派生读数≠真值）。

    锁定规则（prereg v2 §3）：P0 块中位 seasonal_strength < 0.15 而 ≥2/3 代表序列
    robust 检出周期（robust_period_diag period>0，其内部判据=peak_ratio≥3 ∧ acf≥0.2）
    → structure 追加低可靠标注行。全部由 history 计算。
    """
    views = build_block_views(rows, hist_of, frozen_picks, abstains)
    ss = np.median([r["X_p"][2] for r in rows])
    reps = _rep_uids(rows)
    n_det = sum(1 for u in reps if robust_period_diag(_interp(hist_of[u]))["period"] > 0)
    if ss < 0.15 and n_det * 3 >= 2 * len(reps):
        views["structure"] += ("\n[低可靠] P0 季节读数（seasonal_strength≈0）与 robust 周期检测"
                               f"（{n_det}/{len(reps)} 代表序列检出显著周期）冲突——P0 的季节/周期"
                               "估计器在趋势/组合结构下失真，以 [period]/[decomp] 视图为准。")
    return views


def featurize_uid_v2(r: dict, hist: np.ndarray):
    """B+ 臂特征（P0 超集 + v2 促升证据；history-only，assert=CUT）→ (x_d[2], x_p[17])。"""
    h = np.asarray(hist, float).ravel()
    assert h.size == CUT, f"featurize_uid_v2 只接受判官口径观测史 CUT={CUT}"
    hi = _interp(h)
    d = robust_period_diag(hi)
    t = np.arange(hi.size)
    coef = np.polyfit(t, hi, 1)
    slope = coef[0] * 100
    p = int(d["period"]) or 24
    detr = hi - np.polyval(coef, t)
    phase = np.array([detr[i::p].mean() for i in range(p)])
    seas_amp = float(phase.max() - phase.min())
    resid_std = float((detr - np.tile(phase, hi.size // p + 1)[:hi.size]).std())
    runs = _gap_runs(h)
    x_d = np.array([r["snr"], r["miss_rate"]], float)
    x_p = np.array([*r["X_p"],
                    d["period"], d["cand_period"], d["peak_ratio"], d["acf_at_peak"],
                    slope, seas_amp, resid_std,
                    (max(runs) / h.size) if runs else 0.0,
                    len(runs) / max(1, h.size / 100)], float)
    return x_d, x_p
