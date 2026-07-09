"""operators/s1_impute.py — S1 缺失插补。Phase 0 numpy 实现。

S0.7 Operator Integrity：impute_* 契约 = **只更新缺失位置、保留已观测值**（imputer 语义）。
旧 impute_fft 违反此契约（对全序列低通重构 → 改动已观测点，实为 linear-impute + FFT 去噪）已修复。
"""
from __future__ import annotations

import numpy as np

from ._common import as_1d, interp_nan, moving_average
from ._provenance import record


def impute_linear(x, **_) -> np.ndarray:
    return interp_nan(as_1d(x))


def impute_fft(x, cutoff_ratio: float = 0.1, **_) -> np.ndarray:
    """谱插补：**仅用低通重建填补缺失位置，保留所有已观测值**（S0.7 修复观测保持契约）。

    旧版对整段做低通重构 → 改动非缺失观测（实为 linear-impute + FFT 低通去噪，非 imputation）。
    修复：无缺失 → 恒等；否则 linear 先验 + 低通重建，仅写回缺失掩码位置。全缺失/过短 → 显式回退 linear。
    """
    y = as_1d(x).astype(float)
    m = np.isnan(y)
    if not m.any():
        record("impute_fft", "impute_fft", "no_missing_identity")
        return y                                              # 无缺失 → 完全不变
    n = y.size
    if n < 8 or m.all():
        record("impute_fft", "impute_linear", "too_short_or_all_missing")
        return interp_nan(y)
    base = interp_nan(y)                                      # 线性先验（供谱重建）
    f = np.fft.rfft(base - base.mean())
    keep = max(1, int(len(f) * cutoff_ratio))
    f[keep:] = 0.0
    recon = np.fft.irfft(f, n=n) + base.mean()
    out = y.copy()
    out[m] = recon[m]                                         # 只写回缺失位置，保留已观测
    record("impute_fft", "impute_fft", "")
    return out


def impute_ema(x, alpha: float = 0.3, **_) -> np.ndarray:
    """指数平滑（EMA）前向填补。S0.7-6 正名：旧名 `impute_kalman` 系误名（实现是 EMA 非 Kalman），
    保留为兼容 alias（registry.ALIASES）——旧 artifact/模板按旧名重放不破坏。"""
    y = as_1d(x).copy()
    m = np.isnan(y)
    if not m.any():
        return y
    base = interp_nan(y)
    s = base[0]
    for i in range(y.size):
        s = alpha * base[i] + (1 - alpha) * s
        if m[i]:
            y[i] = s
    return y


impute_kalman = impute_ema             # S0.7-6 兼容引用（勿新增使用；registry 层统一走 ALIASES）


def period_complete(x, period: int = 0, **_) -> np.ndarray:
    """用同相位前一周期值填补缺失；无周期信息则退化为线性插补。"""
    y = as_1d(x).copy()
    m = np.isnan(y)
    if not m.any() or period < 2:
        return interp_nan(y)
    for i in np.where(m)[0]:
        j = i - period
        if j >= 0 and not np.isnan(y[j]):
            y[i] = y[j]
    return interp_nan(y)
