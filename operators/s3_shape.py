"""operators/s3_shape.py — S3 任务成形。归一类保持 1D；窗口/特征类改变形状，Phase 0 不支持。"""
from __future__ import annotations

import numpy as np

from ._common import as_1d, interp_nan, ShapeChangingNotSupported


def znorm(x, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    mu, sd = y.mean(), y.std()
    return (y - mu) / sd if sd > 1e-12 else y - mu


def minmax_norm(x, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    lo, hi = y.min(), y.max()
    return (y - lo) / (hi - lo) if (hi - lo) > 1e-12 else np.zeros_like(y)


def sliding_window(x, **_):
    raise ShapeChangingNotSupported("sliding_window 改变形状，Phase 0 1D 流水线不支持")


def lag_features(x, **_):
    raise ShapeChangingNotSupported("lag_features 改变形状，Phase 0 1D 流水线不支持")


def spectral_features(x, **_):
    raise ShapeChangingNotSupported("spectral_features 改变形状，Phase 0 1D 流水线不支持")
