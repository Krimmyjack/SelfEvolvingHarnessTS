"""evaluators/report_target.py — ★ 独立报告器（主表 ΔPerf 专用，**绝不**当 in-loop 判官）。

守 Experiment_Design_Final §★.2「判官↔报告器分离」：报告器模型集与 grounded 判官
(frozen_probe / chronos_probe) **不相交** → readiness 增益非"优化-评测同模型"的循环自证。

与判官的区别：
  • from-scratch target（每 batch 重训）→ σ_A>0 → 多 seed 平均（判官 frozen/chronos 确定性，S=1）。
  • 只在**最终报告期**跑（final-test split），不进进化环路。
指标（AegisTS §6.1.3 模板，higher=better）：forecast perf=exp(−nRMSE)；anomaly perf=recall；
classify perf=accuracy。ΔPerf = perf(ready) − perf(raw)。
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .base import L_WIN, H_FORECAST, ForecastSample, AnomalySample, ClassifySample

# 报告器候选；调用方须挑**与所用判官不相交**的（judge=chronos→用 lstm/dlinear；judge=frozen-LSTM→用 chronos/dlinear）
FORECAST_TARGETS = ("lstm_scratch", "dlinear_scratch", "chronos")
CLASSIFY_TARGETS = ("inception", "rocket")   # judge=rocket → 独立报告器=inception；rocket 仅作确定性交叉参照
_JUDGE_MODELS = {
    "frozen": {"lstm_scratch"}, "chronos": {"chronos"},        # forecast 判官↔同源 target
    "rocket": {"rocket"}, "inception": {"inception"},          # classify 判官↔同源 target
}


def disjoint_targets(judge: str, requested: Sequence[str]) -> List[str]:
    """剔除与判官同源的 target（守分离不变量）。judge='chronos'→去 chronos；'rocket'→去 rocket。"""
    banned = _JUDGE_MODELS.get(judge, set())
    return [t for t in requested if t not in banned]


# ════════════════════════════ forecast ════════════════════════════
def _forecast_perf(batch: List[ForecastSample], target: str, seed: int) -> float:
    """train target on ready 历史窗 → 预测各序列末窗 → obs-norm nRMSE → perf=exp(−nRMSE)。"""
    from .grounded_forecast import build_windows
    from .chronos_probe import _fillna          # 线性填补 NaN：raw-with-missing 才可训练（已插补的 processed 批为 no-op）
    ready = [_fillna(s.ready) for s in batch]
    X, Y = build_windows(ready)
    if X is None or len(X) < 10:
        return float("nan")

    if target == "chronos":
        from .chronos_probe import chronos_forecast
        nrmse = chronos_forecast(batch, seed=seed)
        return float(np.exp(-nrmse)) if np.isfinite(nrmse) else float("nan")

    from . import _torch_models as tm
    tm.seed_all(seed)
    if target == "lstm_scratch":
        model = tm.train_forecaster(tm.LSTMForecaster(L_WIN, H_FORECAST), X, Y, epochs=120)
    elif target == "dlinear_scratch":
        model = tm.train_forecaster(tm.DLinear(L_WIN, H_FORECAST), X, Y, epochs=120)
    else:
        raise ValueError(f"forecast target ∈ {FORECAST_TARGETS}, got {target!r}")
    predict = lambda w: tm.forecast_predict(model, w.reshape(1, -1)).ravel()

    errs = []
    for hh, s in zip(ready, batch):
        if not np.all(np.isfinite(hh)) or hh.size < L_WIN:
            continue
        pred = predict(hh[-L_WIN:])
        fut = np.asarray(s.future, float).ravel()
        h = min(len(pred), len(fut))
        rmse = float(np.sqrt(np.mean((pred[:h] - fut[:h]) ** 2)))
        errs.append(rmse / (s.obs_scale + 1e-9))
    if not errs:
        return float("nan")
    return float(np.exp(-np.mean(errs)))


# ════════════════════════════ anomaly ════════════════════════════
# NOTE: anomaly 当前唯一可用报告器 = 判官同款（top-k）→ **尚未满足分离**（待接 AEDCNN，见 P2）。
def _anomaly_perf(batch: List[AnomalySample], target: str, seed: int) -> float:
    from .grounded_anomaly import anomaly_recall          # TODO: 换 AEDCNN 以满足分离
    return anomaly_recall(batch, seed)


# ════════════════════════════ classify ════════════════════════════
# 分离已满足：in-loop 判官=ROCKET（确定性，set_classify_substrate('rocket')）；报告器**显式**用
# InceptionLite from-scratch（target='inception'）→ 与判官不相交。target='rocket' 仅作确定性交叉参照。
def _classify_perf(batch: List[ClassifySample], target: str, seed: int) -> float:
    from .chronos_probe import _fillna                     # raw(含 miss)有 NaN → 线性填补后才能窗化/训练
    filled = [ClassifySample(_fillna(np.asarray(s.ready, float)), s.label) for s in batch]
    if target in ("inception", "lstm_scratch", ""):        # 默认/历史别名 → 独立报告器 InceptionLite
        from .grounded_classify import classify_inception
        ce = classify_inception(filled, seed=seed)
    elif target == "rocket":                               # 判官同款（仅交叉参照，非 headline 独立报告）
        from .rocket_probe import classify_grounded_rocket
        ce = classify_grounded_rocket(filled, seed=seed)
    else:
        raise ValueError(f"classify target ∈ {CLASSIFY_TARGETS}, got {target!r}")
    return float(np.exp(-ce)) if np.isfinite(ce) else float("nan")


_TASK_ALIASES = {"anomaly": "anomaly_detection", "classify": "classification"}


def report_perf(batch, task: str, target: str = "lstm_scratch", seed: int = 0) -> float:
    """单 target 单 seed 的 perf（higher=better）。"""
    task = _TASK_ALIASES.get(task, task)
    if task == "forecast":
        return _forecast_perf(batch, target, seed)
    if task == "anomaly_detection":
        return _anomaly_perf(batch, target, seed)
    if task == "classification":
        return _classify_perf(batch, target, seed)
    raise ValueError(f"unknown task {task!r}")


def perf_multi(batch, task: str, targets: Sequence[str], seeds: Sequence[int]) -> Dict[str, Tuple[float, float]]:
    """多 target × 多 seed → {target: (perf_mean, perf_std)}。from-scratch 随机 → 多 seed 压噪。"""
    out: Dict[str, Tuple[float, float]] = {}
    for t in targets:
        vals = [report_perf(batch, task, t, s) for s in seeds]
        vals = [v for v in vals if np.isfinite(v)]
        out[t] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
    return out


def delta_perf(ready_batch, raw_batch, task: str,
               targets: Sequence[str] = ("lstm_scratch", "dlinear_scratch"),
               seeds: Sequence[int] = (0, 1)) -> Dict[str, Tuple[float, float]]:
    """ΔPerf = perf(ready) − perf(raw)，多 target × 多 seed。返回 {target: (delta_mean, delta_std)}。

    多 target 方向一致 → readiness 增益稳健（非判官特异）。守分离：targets 应经 disjoint_targets() 过滤。
    """
    p_ready = perf_multi(ready_batch, task, targets, seeds)
    p_raw = perf_multi(raw_batch, task, targets, seeds)
    out: Dict[str, Tuple[float, float]] = {}
    for t in targets:
        dm = p_ready[t][0] - p_raw[t][0]
        ds = float(np.sqrt(p_ready[t][1] ** 2 + p_raw[t][1] ** 2))
        out[t] = (dm, ds)
    return out
