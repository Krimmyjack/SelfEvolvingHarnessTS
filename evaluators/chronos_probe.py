"""evaluators/chronos_probe.py — ★ R10：真 TS foundation encoder（Chronos）作 grounded forecast 判官。

plan.md A.2/A.4 既定的"接预训权重"落地：用 Amazon Chronos-Bolt 零样本直接预测器替代本地 frozen-LSTM。
Chronos-Bolt = 分位数回归（**确定性**，无采样 → σ_A≈0，满足 frozen-probe/R10 确定性要求），单次前向、
跨域预训 → 在真实强季节序列上稳破 seasonal-naive 底（本地 LSTM 的天花板）。

接口与 grounded_forecast 解耦：forecast_grounded(substrate="chronos") 委派到 chronos_forecast(batch)。
模型惰性单例；首次调用下载权重（HF 直连 huggingface.co）。换模型用 set_chronos_model()。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import H_FORECAST

MODEL_ID = "amazon/chronos-bolt-small"     # 确定性、~48M、快；可换 base/chronos-2 提精度
_PIPE = None
_PIPE_ID = None


def set_chronos_model(model_id: str) -> None:
    """切换 Chronos 模型 id（下次 get_chronos 重载）。"""
    global MODEL_ID, _PIPE, _PIPE_ID
    MODEL_ID = model_id
    if _PIPE_ID != model_id:
        _PIPE = None


def get_chronos():
    """惰性单例：加载 Chronos-Bolt pipeline 到 DEVICE。"""
    global _PIPE, _PIPE_ID
    if _PIPE is not None and _PIPE_ID == MODEL_ID:
        return _PIPE
    import torch
    from chronos import BaseChronosPipeline
    from ._torch_models import DEVICE
    _PIPE = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map=DEVICE, torch_dtype=torch.float32)
    _PIPE_ID = MODEL_ID
    return _PIPE


def _fillna(x: np.ndarray) -> np.ndarray:
    """线性插补 NaN（端点最近值）→ 给 Chronos 干净上下文（ready 若未被 harness 插补也鲁棒）。"""
    x = np.asarray(x, dtype=float).ravel()
    if np.all(np.isfinite(x)):
        return x
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if good.sum() < 2:
        return np.nan_to_num(x, nan=float(np.nanmean(x)) if good.any() else 0.0)
    return np.interp(idx, idx[good], x[good])


def chronos_forecast(batch, seed: int = 0, context_len: int = 512) -> float:
    """Chronos-Bolt 零样本直接多步：context=ready 历史 → 预测 H → obs-normalized RMSE（越低越好）。确定性。"""
    import torch
    pipe = get_chronos()
    contexts, valid = [], []
    for s in batch:
        hh = _fillna(s.ready)
        if hh.size < 8:
            continue
        contexts.append(torch.tensor(hh[-context_len:], dtype=torch.float32))
        valid.append(s)
    if not contexts:
        return float("nan")

    # predict_quantiles → (quantiles[B,H,Q], mean[B,H])；用 mean 作点预测（确定性）
    _, mean = pipe.predict_quantiles(contexts, prediction_length=H_FORECAST, quantile_levels=[0.5])
    preds = mean.detach().cpu().numpy()

    errs = []
    for pred, s in zip(preds, valid):
        fut = np.asarray(s.future, dtype=float).ravel()
        h = min(len(pred), len(fut))
        if h == 0:
            continue
        rmse = float(np.sqrt(np.mean((pred[:h] - fut[:h]) ** 2)))
        errs.append(rmse / (s.obs_scale + 1e-9))
    return float(np.mean(errs)) if errs else float("nan")
