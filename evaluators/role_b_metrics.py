"""evaluators/role_b_metrics.py — Role B per-sample 免训练指标（plan.md 不变量 #2）。

**只 log、用于信用分配定位，绝不 gate**（per-sample 质量分不做任何决策）。
慢路径 mining 用这些做"嫌疑算子"定位（如 anomaly cell 上 spike_preservation 低 → 平滑算子是元凶）。
"""
from __future__ import annotations

import numpy as np


def smoothness(x) -> float:
    """lag-1 差分 std（越低越平滑）。"""
    return float(np.std(np.diff(np.asarray(x, float))))


def spike_preservation(x, w: int = 11) -> float:
    """最大局部残差 / std（越高 = spike 越被保留）。"""
    from scipy.ndimage import median_filter
    x = np.asarray(x, float)
    res = x - median_filter(x, size=w, mode="nearest")
    return float(np.max(np.abs(res)) / (np.std(x) + 1e-9))


def fidelity(original, ready) -> float:
    """与原序列在非缺失位的 Pearson 相关（越高 = 越保真）。"""
    o, a = np.asarray(original, float).ravel(), np.asarray(ready, float).ravel()
    if o.size != a.size:
        return float("nan")
    m = ~np.isnan(o) & ~np.isnan(a)
    if m.sum() < 2 or o[m].std() < 1e-12 or a[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(o[m], a[m])[0, 1])
