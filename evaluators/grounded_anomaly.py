"""evaluators/grounded_anomaly.py — §2.E.2 anomaly grounded：注入异常 + 召回。

确定性快速 spike 检测器（top-frac 局部残差），recall@injected；val_loss = 1 - recall（越低越好）。
平滑型 harness 会削弱 spike 的局部残差 → 掉出 top-k → recall 崩（forecast↔anomaly 冲突的度量）。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import AnomalySample, ANOM_FRAC, ANOM_TOL


def _local_residual(x: np.ndarray, w: int = 11) -> np.ndarray:
    from scipy.ndimage import median_filter
    x = np.asarray(x, float)
    return x - median_filter(x, size=w, mode="nearest")


def detect(ready: np.ndarray, frac: float = ANOM_FRAC) -> set:
    """flag top-frac 点（按 |x - rolling_median|）。确定性。"""
    res = np.abs(_local_residual(np.asarray(ready, float)))
    k = max(1, int(round(frac * res.size)))
    return set(np.argsort(res)[-k:].tolist())


def anomaly_recall(batch: List[AnomalySample], seed: int = 0) -> float:
    recalls = []
    for s in batch:
        flagged = detect(s.ready)
        hits = sum(1 for pos in s.positions
                   if any((pos + d) in flagged for d in range(-ANOM_TOL, ANOM_TOL + 1)))
        recalls.append(hits / max(1, len(s.positions)))
    return float(np.mean(recalls)) if recalls else float("nan")


def anomaly_grounded(batch: List[AnomalySample], seed: int = 0) -> float:
    """val_loss = 1 - recall（越低越好，跨任务可比）。"""
    r = anomaly_recall(batch, seed)
    return float("nan") if np.isnan(r) else 1.0 - r
