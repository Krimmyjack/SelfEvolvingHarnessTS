"""evaluators/base.py — Evaluator 接口 + ready-batch 契约 + 协议常量（plan.md §3.4/R5/R9）。

契约（与 harness 解耦）：evaluator 吃**已就绪批**（fast_path 产出的 ready_artifact + 任务真值），
不吃 (harness, raw)。validator（Phase 1）负责跑 fast_path 把候选 harness 变成 ready 批，再喂这里。

统一裁决量 = val_loss，**越低越好**（forecast nRMSE / anomaly 1-recall / classify CE），便于跨任务比较。
两层：layer="proxy"（Role A 轻量预筛，Layer1）/ "grounded"（真下游模型，Layer2 唯一 accept 裁决）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ── 评估协议常量（结构参数，非校准阈值）──────────────────────────────────
L_WIN = 48           # forecast 直接多步：滞后窗口
H_FORECAST = 48      # forecast 视野
STRIDE = 8           # 窗口步长
WIN_CLF = 64         # classification 分窗长度
STRIDE_CLF = 32
ANOM_FRAC = 0.03     # anomaly 检测器 top-frac
ANOM_TOL = 2         # recall@injected 容差


# ── ready-batch 样本契约 ────────────────────────────────────────────────────
@dataclass
class ForecastSample:
    ready: np.ndarray              # harness 输出的就绪历史
    future: np.ndarray             # 真未来（留出，clean）
    obs_scale: float = 1.0         # 归一化尺度（clean std）
    period: int = 24


@dataclass
class AnomalySample:
    ready: np.ndarray              # harness 输出（应保异常信号）
    positions: List[int] = field(default_factory=list)


@dataclass
class ClassifySample:
    ready: np.ndarray
    label: int = 0


_TASK_ALIASES = {"anomaly": "anomaly_detection", "classify": "classification"}


class Evaluator:
    """按 task_type 多态 + 按 layer 分两层。evaluate(ready_batch, layer) -> val_loss。

    实现函数住在 role_a_proxy / grounded_* / frozen_probe；此处惰性 import 做派发，避免循环依赖。
    """

    def __init__(self, task_type: str):
        self.task_type = _TASK_ALIASES.get(task_type, task_type)

    def evaluate(self, ready_batch, layer: str = "grounded", seed: int = 0, **kw) -> float:
        if layer not in ("proxy", "grounded"):
            raise ValueError(f"layer ∈ {{proxy, grounded}}, got {layer!r}")
        t = self.task_type
        if t == "forecast":
            from . import role_a_proxy, grounded_forecast
            return (role_a_proxy.forecast_proxy(ready_batch, seed=seed) if layer == "proxy"
                    else grounded_forecast.forecast_grounded(ready_batch, seed=seed, **kw))
        if t == "anomaly_detection":
            from . import role_a_proxy, grounded_anomaly
            return (role_a_proxy.anomaly_proxy(ready_batch, seed=seed) if layer == "proxy"
                    else grounded_anomaly.anomaly_grounded(ready_batch, seed=seed))
        if t == "classification":
            from . import role_a_proxy, grounded_classify
            return (role_a_proxy.classify_proxy(ready_batch, seed=seed) if layer == "proxy"
                    else grounded_classify.classify_grounded(ready_batch, seed=seed, **kw))
        raise ValueError(f"unknown task_type {t!r}")


def get_evaluator(task_type: str) -> Evaluator:
    return Evaluator(task_type)
