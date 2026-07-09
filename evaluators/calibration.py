"""evaluators/calibration.py — proxy↔grounded Spearman 校准门（plan.md R7，τ=0.4）。

在一个 cell 上，对 ≥5 个 harness 变体用 proxy 和 grounded 双评分 → Spearman 秩相关。
proxy 在该 cell **可用 iff** Spearman ≥ τ（否则该 cell 的 proxy 预筛不可信，慢路径直接上 grounded）。
无证不用——这是把"proxy 已知不可靠"做成机械门的落点。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.stats import spearmanr

from ..config.thresholds import TAU_PROXY


@dataclass
class CalibrationResult:
    spearman: float
    p_value: float
    usable: bool
    tau: float
    n: int


def spearman_gate(proxy_losses: Sequence[float], grounded_losses: Sequence[float],
                  tau: float = TAU_PROXY) -> CalibrationResult:
    """同一组变体上的 (proxy, grounded) 配对 → Spearman；usable = ρ ≥ τ。

    两者都是"越低越好"，正相关即同向。NaN 配对剔除；有效点 < 3 → 不可用。
    """
    p = np.asarray(proxy_losses, float)
    g = np.asarray(grounded_losses, float)
    m = np.isfinite(p) & np.isfinite(g)
    if m.sum() < 3:
        return CalibrationResult(float("nan"), float("nan"), False, tau, int(m.sum()))
    rho, pv = spearmanr(p[m], g[m])
    rho = float(rho) if np.isfinite(rho) else float("nan")
    usable = bool(np.isfinite(rho) and rho >= tau)
    return CalibrationResult(rho, float(pv), usable, tau, int(m.sum()))
