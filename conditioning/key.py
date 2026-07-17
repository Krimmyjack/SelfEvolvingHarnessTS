"""conditioning/key.py — conditioning_key 构造（plan.md §3.1，R3 索引语言）。

输出 conditioning_key = {pattern{struct_feats(10), quality_profile}, task{...}}。
cell_id 由 conditioning/binning.py（Phase 1）从 struct_feats 派生，本文件不产 cell_id。

struct_feats 10 维（Implementation_Design §1 / TIME 7-core + 扩展）：
  period, trend_strength, seasonal_strength, SNR(dB), acf1,
  stationarity_adf(p), spectral_entropy, lumpiness, outlier_density, missing_rate

依赖：numpy（必需）。statsmodels（ADF/STL，可选，缺失则用 numpy 代理）。
所有特征对 NaN 鲁棒（先按缺失率记录，再在非缺失子序列上算）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from . import thresholds as TH

try:                                   # ADF 单位根检验（可选）
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _HAS_ADF = True
except Exception:                      # pragma: no cover
    _adfuller = None
    _HAS_ADF = False

STRUCT_FEAT_NAMES: List[str] = [
    "period", "trend_strength", "seasonal_strength", "SNR", "acf1",
    "stationarity_adf", "spectral_entropy", "lumpiness", "outlier_density", "missing_rate",
]
assert len(STRUCT_FEAT_NAMES) == TH.STRUCT_FEATS_DIM, "struct_feats 维度须与 config 一致"


# ════════════════════════════ 工具 ════════════════════════════
def _as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("empty series")
    return a


def _acf(x: np.ndarray, lag: int) -> float:
    if x.size <= lag:
        return 0.0
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


# A0 第一步（2026-07-05）：实现逐字迁入共享模块 conditioning/period.py（bit 级不变），
# 此处保留原名 `_dominant_period` —— P0 契约（PatternSpec.period_estimator_id 指
# "conditioning/key.py:_dominant_period" = legacy_fft_v0）继续成立。P1 用 robust_v1/top-k。
from .period import dominant_period_fft_v0 as _dominant_period  # noqa: E402


# ════════════════════════════ struct_feats（10 维）════════════════════════════
def struct_feats(series, *, period_hint: Optional[int] = None) -> Dict[str, float]:
    raw = _as_1d(series)
    mask = ~np.isnan(raw)
    missing_rate = float(1.0 - mask.mean())
    x = raw[mask]
    if x.size < 4:                          # 退化：几乎全缺
        feats = {k: 0.0 for k in STRUCT_FEAT_NAMES}
        feats["missing_rate"] = missing_rate
        feats["period"] = 1.0
        return feats

    n = x.size
    t = np.arange(n, dtype=float)

    # period + 频谱
    period, power = _dominant_period(x)

    # trend_strength = 线性趋势的 R²（去趋势前后方差比）
    var_x = float(np.var(x))
    if var_x <= 1e-12:
        trend_strength = 0.0
        fit = np.full(n, x.mean())
        detrended = x - fit
    else:
        coef = np.polyfit(t, x, 1)
        fit = np.polyval(coef, t)
        detrended = x - fit
        trend_strength = float(np.clip(1.0 - np.var(detrended) / var_x, 0.0, 1.0))

    # seasonal_strength = 去趋势序列在主周期处的 |ACF|
    p = period_hint if period_hint else int(round(period))
    if 2 <= p < n:
        seasonal_strength = float(np.clip(abs(_acf(detrended, p)), 0.0, 1.0))
    else:
        seasonal_strength = 0.0

    # SNR(dB)：去趋势 + 去主季节后的**稳健**残差噪声（MAD，抗 5σ 离群）vs 结构信号功率。
    # （旧 MA-11 估计被离群主导、把周期信号当噪声 → 范围窄且为负；此版分离 高斯噪声/离群/周期信号。）
    fftd = np.fft.rfft(detrended)
    if fftd.size > 1:
        k_keep = min(3, fftd.size - 1)                     # 取前 K 主频（含基频+谐波）→ 残差才是纯噪声
        top = np.argsort(np.abs(fftd[1:]))[-k_keep:] + 1
        seasonal_fft = np.zeros_like(fftd)
        seasonal_fft[top] = fftd[top]
        seasonal_comp = np.fft.irfft(seasonal_fft, n)
    else:
        seasonal_comp = np.zeros(n)
    resid = detrended - seasonal_comp
    mad = float(np.median(np.abs(resid - np.median(resid))))
    noise_var = (1.4826 * mad) ** 2                        # 稳健噪声方差（离群被中位数忽略）
    signal_var = float(np.var(fit + seasonal_comp))
    if noise_var <= 1e-12:
        snr_db = 60.0
    elif signal_var <= 1e-12:
        snr_db = -60.0
    else:
        snr_db = float(np.clip(10.0 * np.log10(signal_var / noise_var), -60.0, 60.0))

    acf1 = float(np.clip(_acf(x, 1), -1.0, 1.0))

    # ADF p-value（非平稳→大）：有 statsmodels 用真值，否则用一阶差分方差比代理
    stationarity_adf = _adf_pvalue(x)

    # spectral_entropy：功率谱归一熵 ∈ [0,1]
    p_norm = power[power > 0]
    if p_norm.size <= 1:
        spectral_entropy = 0.0
    else:
        ent = -np.sum(p_norm * np.log(p_norm))
        spectral_entropy = float(np.clip(ent / np.log(p_norm.size), 0.0, 1.0))

    # lumpiness：分块方差的方差（非平稳/异方差指标）
    lumpiness = _lumpiness(x)

    # outlier_density：稳健 MAD 判定
    outlier_density = _outlier_density(x)

    return {
        "period": float(period),
        "trend_strength": trend_strength,
        "seasonal_strength": seasonal_strength,
        "SNR": snr_db,
        "acf1": acf1,
        "stationarity_adf": stationarity_adf,
        "spectral_entropy": spectral_entropy,
        "lumpiness": lumpiness,
        "outlier_density": outlier_density,
        "missing_rate": missing_rate,
    }


def _adf_pvalue(x: np.ndarray) -> float:
    if _HAS_ADF and x.size >= 12:
        try:
            return float(np.clip(_adfuller(x, autolag="AIC")[1], 0.0, 1.0))
        except Exception:
            pass
    # 代理：一阶差分显著降方差 → 偏平稳（p 小）。映射到 [0,1]。
    if x.size < 3:
        return 1.0
    v0 = float(np.var(x))
    v1 = float(np.var(np.diff(x)))
    if v0 <= 1e-12:
        return 0.0
    ratio = v1 / v0                          # 平稳序列差分后方差↑或≈；随机游走差分后方差↓
    return float(np.clip(1.0 - min(ratio, 1.0), 0.0, 1.0))


def _lumpiness(x: np.ndarray, n_tiles: int = 10) -> float:
    if x.size < n_tiles * 2:
        return 0.0
    tiles = np.array_split(x, n_tiles)
    tile_vars = np.array([np.var(t) for t in tiles if t.size > 1])
    if tile_vars.size < 2:
        return 0.0
    scale = float(np.mean(tile_vars)) + 1e-12
    return float(np.var(tile_vars) / (scale ** 2))     # 无量纲化


def _outlier_density(x: np.ndarray, k: Optional[float] = None) -> float:
    k = TH.OUTLIER_MAD_K if k is None else k
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 1e-12:
        return 0.0
    z = np.abs(x - med) / (1.4826 * mad)               # 1.4826 → 一致正态估计
    return float(np.mean(z > k))


# ════════════════════════════ quality_profile ════════════════════════════
def quality_profile(series, feats: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    f = feats if feats is not None else struct_feats(series)
    has_missing = f["missing_rate"] > 0.0
    has_outlier = f["outlier_density"] > 0.01
    has_noise = f["SNR"] < TH.SNR_DB_NOISY
    has_drift = (f["trend_strength"] > TH.TREND_STRENGTH_DRIFT) or \
                (f["stationarity_adf"] > TH.ADF_NONSTATIONARY_P)
    problem_types = {
        "has_missing": has_missing, "has_outlier": has_outlier,
        "has_noise": has_noise, "has_drift": has_drift,
    }
    urgency = float(np.clip(
        0.4 * f["missing_rate"] + 0.3 * f["outlier_density"]
        + 0.2 * (1.0 if has_noise else 0.0) + 0.1 * (1.0 if has_drift else 0.0),
        0.0, 1.0))
    return {
        "problem_types": problem_types,
        "urgency": urgency,
        "violation_profile": {"type": None, "positions": []},   # ConstraintMiner（Phase 1）填充
    }


# ════════════════════════════ conditioning_key ════════════════════════════
def build_conditioning_key(series, task_type: str,
                           task_spec: Optional[Dict[str, Any]] = None,
                           *, period_hint: Optional[int] = None) -> Dict[str, Any]:
    """组装 conditioning_key（不含 cell_id —— 由 binning.py 在 Phase 1 派生）。"""
    feats = struct_feats(series, period_hint=period_hint)
    qp = quality_profile(series, feats)
    spec = task_spec or {}
    return {
        "pattern": {"struct_feats": feats, "quality_profile": qp},
        "task": {
            "type": task_type,
            "sensitivity": spec.get("sensitivity", {"preserve": [], "suppress": []}),
            "output_form": spec.get("output_form"),
            "readiness_eval": spec.get("readiness_eval"),
        },
    }
