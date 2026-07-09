"""evaluators/readiness.py — ★v4 S1 前向迁移度量（S1_Implementation_Plan §B.5；judge-aware 锚）。

readiness = (J_worst − J_cur) / (J_worst − J_floor)，J = grounded val_loss（**越低越好**），其中
两个参照 J_raw（不处理）与 J_min_ref（最小健康 harness）取
  • J_floor = min(J_raw, J_min_ref)  = 两参照中**可达的更优者**（readiness 的目标锚）；
  • J_worst = max(J_raw, J_min_ref)  = 两参照中更差者（readiness=0 的锚）。
readiness=1 ⇔ J_cur 追平更优参照；>1 ⇔ 超越；<0 ⇔ 比更差参照还糟（真损伤）。

为何用 min/max 而非固定 J_raw 作锚（**judge-aware**）：
  • 弱判官（frozen-probe，清洗有 headroom）：J_min_ref<J_raw → J_floor=J_min_ref、J_worst=J_raw，
    **退化为旧式** (J_raw−J_cur)/(J_raw−J_min_ref)，语义不变（readiness=1 ⇔ 追平健康基线）。
  • 基础模型判官（Chronos：在 RAW 退化序列上反而更准，naive 清洗造伪影伤它）：J_min_ref>J_raw →
    参照翻向 raw，readiness=1 ⇔ harness 学会**不伤/撤销有害清洗**追平 raw。旧式分母 J_raw−J_min_ref<0
    会把这条（唯一能显正向迁移的）线全判 nan；新式分母 |J_raw−J_min_ref|>0 → 可解释。
分母 = |J_raw − J_min_ref|，仅当两参照无差（无可观测尺度）→ nan。
time-to-readiness = 首个 readiness ≥ READINESS_THRESHOLD 的 round。
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np


def readiness_score(j_raw: float, j_cur: float, j_min_ref: float, *, eps: float = 1e-9) -> float:
    """三个 grounded val_loss → readiness 分（judge-aware：锚 = 两参照中可达更优者）。

    分母 = |J_raw − J_min_ref|；两参照无差（denom ≤ eps）→ nan（该 cell 无可观测尺度）。
    """
    if not (math.isfinite(j_raw) and math.isfinite(j_cur) and math.isfinite(j_min_ref)):
        return float("nan")
    j_floor = min(j_raw, j_min_ref)
    j_worst = max(j_raw, j_min_ref)
    denom = j_worst - j_floor
    if denom <= eps:
        return float("nan")
    return float((j_worst - j_cur) / denom)


def is_ready(j_raw: float, j_cur: float, j_min_ref: float, threshold: float) -> bool:
    r = readiness_score(j_raw, j_cur, j_min_ref)
    return bool(math.isfinite(r) and r >= threshold)


def aggregate_time_to_readiness(per_cell: List[Optional[int]]) -> dict:
    """per-cell time-to-readiness（round；未达=None）→ per-domain 聚合。

    **同报 median 与 max**（S1_Plan §B.5）：headline 用 median（对单个病态 cell 鲁棒），
    max 作 worst-case 旁证。median 只取有限值；若某 cell 永不达标(None)，max=None（worst-case 未就绪）。
    """
    finite = [t for t in per_cell if t is not None]
    median = float(np.median(finite)) if finite else None
    max_v = None if any(t is None for t in per_cell) else (max(per_cell) if per_cell else None)
    return {"median": median, "max": max_v,
            "n_ready": len(finite), "n_cells": len(per_cell)}
