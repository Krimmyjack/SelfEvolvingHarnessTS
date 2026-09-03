"""run_pattern_features.py — PATTERN 实验特征层。

协议见 artifacts/functional/e2/_drafts/pattern_protocol.json（FROZEN_BEFORE_ANY_FIT）。
所有特征只使用 series[:origin]。零 LLM，不计算 gain。
"""
from __future__ import annotations
import numpy as np

PERIOD = 24
RECENT = 192
EPS = 1e-12
RATE_FLOOR = 1e-3          # x_routl_by_goutl 的分母下限（1/1000 的率）


def _fin(x):
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def _mad(x):
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))))


def _decomp(x, period=PERIOD):
    """轻量 STL 替身：中心移动平均趋势 + 逐相位中位季节 + 余项。确定性、无外部依赖。"""
    n = x.size
    if n < 2 * period:
        return np.zeros(n), np.zeros(n), x - x.mean() if n else x
    w = period if period % 2 == 1 else period + 1
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    trend = np.convolve(xp, np.ones(w) / w, mode="valid")[:n]
    detr = x - trend
    prof = np.array([np.median(detr[p::period]) if detr[p::period].size else 0.0
                     for p in range(period)])
    prof = prof - prof.mean()
    seas = np.tile(prof, n // period + 1)[:n]
    return trend, seas, x - trend - seas


def _strengths(x):
    t, s, r = _decomp(x)
    vr = float(np.var(r))
    st = max(0.0, 1.0 - vr / (float(np.var(t + r)) + EPS))
    ss = max(0.0, 1.0 - vr / (float(np.var(s + r)) + EPS))
    return st, ss


def _spectral_entropy(x):
    if x.size < 4:
        return 0.0
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    p = p[1:]
    tot = p.sum()
    if tot <= EPS:
        return 0.0
    p = p / tot
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum() / np.log2(p.size)) if p.size > 1 else 0.0


def _acf1(x):
    if x.size < 3:
        return 0.0
    xc = x - x.mean()
    d = float((xc * xc).sum())
    return float((xc[:-1] * xc[1:]).sum() / d) if d > EPS else 0.0


def _outliers(x):
    """robust-z 超 3 的比例 / 最大 z / 位置重心（越接近 1 越靠窗口末尾）。"""
    if x.size == 0:
        return 0.0, 0.0, 0.0
    med = np.median(x)
    sc = 1.4826 * _mad(x)
    if sc <= EPS:
        return 0.0, 0.0, 0.0
    z = np.abs(x - med) / sc
    hit = z > 3.0
    rate = float(hit.mean())
    mx = float(z.max())
    if hit.any():
        pos = float(np.mean(np.flatnonzero(hit) / max(1, x.size - 1)))
    else:
        pos = 0.0
    return rate, mx, pos


def _seasonal_profile(x, period=PERIOD):
    _, s, _ = _decomp(x, period)
    return s[:period] if s.size >= period else np.zeros(period)


def _block(x, prefix):
    """G/R 通用块。"""
    x = _fin(x)
    st, ss = _strengths(x)
    rate, mx, pos = _outliers(x)
    sd = float(np.std(x)) if x.size else 0.0
    q75, q25 = (np.percentile(x, [75, 25]) if x.size else (0.0, 0.0))
    kurt = 0.0
    if x.size > 3 and sd > EPS:
        kurt = float(np.mean(((x - x.mean()) / sd) ** 4) - 3.0)
    return {
        prefix + "trend_strength": st,
        prefix + "seasonality_strength": ss,
        prefix + "spectral_entropy": _spectral_entropy(x),
        prefix + "acf1": _acf1(x),
        prefix + "outlier_rate": rate,
        prefix + "outlier_max_z": mx,
        prefix + "outlier_recency_weight": pos,
        prefix + "iqr_over_std": float((q75 - q25) / (sd + EPS)),
        prefix + "kurtosis": kurt,
    }


def pattern_features(raw, origin: int) -> dict[str, float]:
    """全部预注册特征。只读 raw[:origin]。"""
    raw = np.asarray(raw, dtype=np.float64)
    hist_all = raw[:origin]
    rec = raw[origin - RECENT:origin]
    hist = raw[:origin - RECENT]

    g = _block(hist_all, "g_")
    r = _block(rec, "r_")
    h = _block(hist, "h_")

    ha, rf, hf = _fin(hist_all), _fin(rec), _fin(hist)
    g_var_ratio = 0.0
    if ha.size >= 4:
        half = ha.size // 2
        v1, v2 = float(np.var(ha[:half])), float(np.var(ha[half:]))
        g_var_ratio = float(v2 / (v1 + EPS))

    mad_h, mad_a = _mad(hf), _mad(ha)
    r_level_z = float((np.median(rf) - np.median(ha)) / (mad_a + EPS)) if rf.size and ha.size else 0.0
    r_scale_ratio = float(_mad(rf) / (mad_a + EPS))
    d_level_shift = float((np.median(rf) - np.median(hf)) / (mad_h + EPS)) if rf.size and hf.size else 0.0
    d_scale_shift = float(_mad(rf) / (mad_h + EPS))

    pr, ph = _seasonal_profile(rf), _seasonal_profile(hf)
    if pr.std() > EPS and ph.std() > EPS:
        shape_drift = float(1.0 - np.corrcoef(pr, ph)[0, 1])
    else:
        shape_drift = 1.0
    # r_period_shape_corr：最近一个完整周期 vs 历史相位中位形状
    last = rf[-PERIOD:] if rf.size >= PERIOD else np.zeros(PERIOD)
    if last.std() > EPS and ph.std() > EPS:
        shape_corr = float(np.corrcoef(last - last.mean(), ph)[0, 1])
    else:
        shape_corr = 0.0

    out = {}
    for k in ("trend_strength", "seasonality_strength", "spectral_entropy", "acf1",
              "outlier_rate", "iqr_over_std", "kurtosis"):
        out["g_" + k] = g["g_" + k]
    out["g_var_ratio_halves"] = g_var_ratio
    for k in ("trend_strength", "seasonality_strength", "spectral_entropy", "acf1",
              "outlier_rate", "outlier_max_z", "outlier_recency_weight"):
        out["r_" + k] = r["r_" + k]
    out["r_period_shape_corr"] = shape_corr
    out["r_level_z"] = r_level_z
    out["r_scale_ratio"] = r_scale_ratio
    for k in ("trend_strength", "seasonality_strength", "spectral_entropy",
              "acf1", "outlier_rate"):
        out["d_" + k] = r["r_" + k] - h["h_" + k]
    out["d_level_shift"] = d_level_shift
    out["d_scale_shift"] = d_scale_shift
    out["d_seasonal_shape_drift"] = shape_drift
    out["x_gseas_by_dshape"] = out["g_seasonality_strength"] * shape_drift
    out["x_gtrend_by_dlevel"] = out["g_trend_strength"] * d_level_shift
    out["x_routl_by_goutl"] = out["r_outlier_rate"] / max(out["g_outlier_rate"], RATE_FLOOR)
    return out


FEATURE_NAMES = tuple(sorted(pattern_features(
    np.sin(np.arange(1024) * 0.26) * 10 + np.arange(1024) * 0.01, 600).keys()))
