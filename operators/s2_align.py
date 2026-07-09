"""operators/s2_align.py — S2 时间规整。Phase 0 无时间戳，resample 为恒等占位；fill_gaps=插补。"""
from __future__ import annotations

import numpy as np

from ._common import as_1d, interp_nan


def resample_uniform(x, **_) -> np.ndarray:
    """Phase 0：假设已等间隔，恒等返回（接入时间戳后再实现真正重采样）。"""
    return as_1d(x).copy()


def fill_gaps(x, **_) -> np.ndarray:
    """填补缺口（与 impute_linear 同效，归类于对齐阶段）。"""
    return interp_nan(as_1d(x))
