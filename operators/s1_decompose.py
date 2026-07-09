"""operators/s1_decompose.py — S1 分解（返回去噪重建信号，保持 series→series 契约）。"""
from __future__ import annotations

import numpy as np

from ._common import as_1d, interp_nan, moving_average
from .s1_denoise import denoise_stl, _guess_period


def stl_decompose(x, period: int = 0, **_) -> np.ndarray:
    """返回 trend+seasonal（= 去残差），与 denoise_stl 同语义但归类于分解。"""
    return denoise_stl(x, period=period)


def fft_decompose(x, keep_ratio: float = 0.15, **_) -> np.ndarray:
    """低通重建：保留低频主成分（趋势+主季节），抑制高频。"""
    y = interp_nan(as_1d(x))
    n = y.size
    if n < 8:
        return y
    mean = y.mean()
    f = np.fft.rfft(y - mean)
    keep = max(1, int(len(f) * keep_ratio))
    f[keep:] = 0.0
    return np.fft.irfft(f, n=n) + mean


def smooth_ema(x, alpha: float = 0.3, **_) -> np.ndarray:
    """一阶指数平滑（EMA）。S0.7-6 正名：旧名 `kalman_filter` 系误名（实现是 EMA 非 Kalman），
    保留为兼容 alias（registry.ALIASES）。"""
    y = interp_nan(as_1d(x))
    out = np.empty_like(y)
    s = y[0]
    for i in range(y.size):
        s = alpha * y[i] + (1 - alpha) * s
        out[i] = s
    return out


kalman_filter = smooth_ema             # S0.7-6 兼容引用（勿新增使用；registry 层统一走 ALIASES）
