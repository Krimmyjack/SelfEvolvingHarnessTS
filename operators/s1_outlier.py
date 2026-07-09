"""operators/s1_outlier.py — S1 离群处理（裁剪/收缩，保持长度）。"""
from __future__ import annotations

import numpy as np

from ._common import as_1d, interp_nan


def winsorize(x, limits: float = 0.05, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    lo, hi = np.quantile(y, [limits, 1.0 - limits])
    return np.clip(y, lo, hi)


def outlier_iqr(x, k: float = 1.5, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = q3 - q1
    return np.clip(y, q1 - k * iqr, q3 + k * iqr)


def outlier_mad(x, k: float = 3.5, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad <= 1e-12:
        return y
    scale = 1.4826 * mad
    return np.clip(y, med - k * scale, med + k * scale)
