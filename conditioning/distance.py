"""conditioning/distance.py — d = α·d_struct + (1-α)·d_quality（plan.md §2/R3，供 memory 检索）。

复用我们的索引语言（10 维 struct_feats + quality_profile），不另造 raw-series embedding。
d_struct = 尺度归一后的 struct_feats 欧氏距离；d_quality = problem_types 差异 + urgency 差。
similarity = 1/(1+d) ∈ (0,1]。
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from ..config.thresholds import ALPHA_DISTANCE
from .key import STRUCT_FEAT_NAMES

# 各维粗尺度（把量纲不同的特征压到可比量级；period/SNR 大，其余多在 [0,1]）
_SCALE = {
    "period": 50.0, "trend_strength": 1.0, "seasonal_strength": 1.0, "SNR": 30.0, "acf1": 1.0,
    "stationarity_adf": 1.0, "spectral_entropy": 1.0, "lumpiness": 1.0,
    "outlier_density": 0.2, "missing_rate": 0.3,
}


def _vec(feats: Dict[str, float]) -> np.ndarray:
    return np.array([float(feats.get(k, 0.0)) / _SCALE[k] for k in STRUCT_FEAT_NAMES])


def d_struct(fa: Dict[str, float], fb: Dict[str, float]) -> float:
    """归一化欧氏距离（除以 √dim 使典型值 ~[0,1]）。"""
    return float(np.linalg.norm(_vec(fa) - _vec(fb)) / math.sqrt(len(STRUCT_FEAT_NAMES)))


def d_quality(qa: Dict[str, Any], qb: Dict[str, Any]) -> float:
    pa, pb = qa.get("problem_types", {}), qb.get("problem_types", {})
    keys = set(pa) | set(pb)
    diff = sum(bool(pa.get(k)) != bool(pb.get(k)) for k in keys) / max(1, len(keys))
    durg = abs(float(qa.get("urgency", 0.0)) - float(qb.get("urgency", 0.0)))
    return float(0.7 * diff + 0.3 * durg)


def distance(key_a: Dict[str, Any], key_b: Dict[str, Any], alpha: float = ALPHA_DISTANCE) -> float:
    pa, pb = key_a["pattern"], key_b["pattern"]
    ds = d_struct(pa["struct_feats"], pb["struct_feats"])
    dq = d_quality(pa["quality_profile"], pb["quality_profile"])
    return float(alpha * ds + (1.0 - alpha) * dq)


def similarity(key_a: Dict[str, Any], key_b: Dict[str, Any], alpha: float = ALPHA_DISTANCE) -> float:
    return 1.0 / (1.0 + distance(key_a, key_b, alpha))
