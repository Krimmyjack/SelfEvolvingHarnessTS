"""evaluators/grounded_forecast.py — §2.E.1 forecast grounded：训练+回测 nRMSE（floor=seasonal_naive）。

默认 substrate="frozen"（冻结 LSTM 编码器 + Ridge 头，σ_A≈0，grounded 默认底座，R10）；
substrate="scratch"（从头训 LSTMForecaster，σ_A 由 seed 体现）仅留作消融。
val_loss = mean obs-normalized RMSE（越低越好）。需 conda `project` 环境（torch）。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .base import ForecastSample, L_WIN, H_FORECAST, STRIDE


def _ready(batch) -> List[np.ndarray]:
    return [np.asarray(s.ready, dtype=float).ravel() for s in batch]


def build_windows(ready_histories, l_win=L_WIN, h=H_FORECAST, stride=STRIDE
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从就绪历史构 (X[l_win], Y[h]) 直接多步对，严格留在历史内（不泄漏真未来）。跳过含 NaN 的序列。"""
    X, Y = [], []
    span = l_win + h
    for hh in ready_histories:
        hh = np.asarray(hh, float).ravel()
        if not np.all(np.isfinite(hh)) or hh.size < span:
            continue
        for start in range(0, hh.size - span + 1, stride):
            X.append(hh[start:start + l_win])
            Y.append(hh[start + l_win:start + span])
    if not X:
        return None, None
    return np.asarray(X), np.asarray(Y)


# 预测目标模式（全局可切，默认 'raw' 保持既有行为/合成测试不变）：
#   'raw'           = 直接回归未来值（与历史一致，旧默认）。
#   'seasonal_resid'= 回归季节残差 y[t]-y[t-period]，预测时加回 seasonal-naive 基线。
#   'ensemble'      = 学习权重 w∈[0,1] 收缩混合 (frozen forecaster, seasonal-naive)：训练窗上闭式解 w*，
#                     test 出 w·frozen+(1-w)·snaive → **构造性 ≥ naive 底**（w→0 退化为 naive），
#                     强季节 cell 不再劣于 naive、趋势 cell 仍由 frozen 取胜。森林组合是标准做法，非作弊。
_TARGET = "raw"
_SUBSTRATE = "frozen"     # 全局默认 substrate：'frozen'(本地LSTM+Ridge) | 'chronos'(真 foundation) | 'scratch'(消融)


def set_forecast_target(mode: str) -> None:
    """全局切换 grounded forecast 目标（'raw'|'seasonal_resid'|'ensemble'）。一次设置整条 grounded
    链路生效，对 evaluator/validator 接口零侵入；默认 'raw'。"""
    global _TARGET
    if mode not in ("raw", "seasonal_resid", "ensemble"):
        raise ValueError(f"target ∈ {{raw, seasonal_resid, ensemble}}, got {mode!r}")
    _TARGET = mode


def set_forecast_substrate(mode: str) -> None:
    """全局切换 grounded forecast substrate（'frozen'|'chronos'|'scratch'）。chronos=真 TS foundation
    判官（plan A.2/A.4）。对 evaluator/validator 接口零侵入；默认 'frozen' 保持既有行为与测试不变。"""
    global _SUBSTRATE
    if mode not in ("frozen", "chronos", "scratch"):
        raise ValueError(f"substrate ∈ {{frozen, chronos, scratch}}, got {mode!r}")
    _SUBSTRATE = mode


def _snaive(win: np.ndarray, period: int, h: int) -> np.ndarray:
    """从窗口末尾取最近 period 段循环外推 h 步（窗口级 seasonal-naive）。"""
    p = max(2, min(int(period), win.size))
    return np.array([win[win.size - p + (i % p)] for i in range(h)])


def _build_windows_full(ready, periods, l_win=L_WIN, h=H_FORECAST, stride=STRIDE):
    """训练窗 (X[l_win], Y_raw[h], SN[h]=窗口级 seasonal-naive)；逐序列保留 period。供 raw/resid/ensemble 复用。"""
    X, Y, SN = [], [], []
    span = l_win + h
    for hh, p in zip(ready, periods):
        hh = np.asarray(hh, float).ravel()
        if not np.all(np.isfinite(hh)) or hh.size < span:
            continue
        for start in range(0, hh.size - span + 1, stride):
            x = hh[start:start + l_win]
            X.append(x)
            Y.append(hh[start + l_win:start + span])
            SN.append(_snaive(x, p, h))
    if not X:
        return None, None, None
    return np.asarray(X), np.asarray(Y), np.asarray(SN)


def _oof_frozen_preds(X: np.ndarray, Y: np.ndarray, folds: int = 2) -> np.ndarray:
    """2-fold 样本外 frozen 预测（Ridge 头闭式、廉价）：每折在补集上拟合头、在本折预测 → 诚实泛化估计。

    供 ensemble 的 blend 权用，避免 in-sample 过拟合假象。小批退化（折太小）回落整体 in-sample。
    """
    from .frozen_probe import FrozenProbe
    n = len(X)
    fold = np.arange(n) % folds
    if n < 2 * folds or min(np.bincount(fold, minlength=folds)) < 2:
        return FrozenProbe().fit(X, Y).predict(X)
    F = np.zeros_like(np.asarray(Y, dtype=float))
    for f in range(folds):
        tr, te = (fold != f), (fold == f)
        F[te] = FrozenProbe().fit(X[tr], Y[tr]).predict(X[te])
    return F


def _blend_weight(F: np.ndarray, SN: np.ndarray, Y: np.ndarray) -> float:
    """闭式最优收缩权 w*∈[0,1] 最小化 ‖w·F+(1-w)·SN − Y‖²（A=F−SN, B=SN−Y → w=−<A,B>/<A,A>）。"""
    A = (F - SN).ravel()
    B = (SN - Y).ravel()
    denom = float(np.dot(A, A))
    if denom <= 1e-12:
        return 0.0                                 # frozen 与 naive 无差异 → 退化为 naive
    return float(np.clip(-np.dot(A, B) / denom, 0.0, 1.0))


def forecast_grounded(batch: List[ForecastSample], seed: int = 0,
                      substrate: Optional[str] = None, target: Optional[str] = None) -> float:
    substrate = substrate or _SUBSTRATE
    if substrate == "chronos":                       # 真 foundation 判官：直接零样本预测，旁路 windows/probe
        from .chronos_probe import chronos_forecast
        return chronos_forecast(batch, seed=seed)
    target = target or _TARGET
    ready = _ready(batch)
    X, Y, SN = _build_windows_full(ready, [s.period for s in batch])
    if X is None or len(X) < 10:
        return float("nan")
    Y_fit = (Y - SN) if target == "seasonal_resid" else Y      # resid 学残差，其余学原始

    if substrate == "frozen":
        from .frozen_probe import FrozenProbe        # 冻结 LSTM 编码器 + Ridge → 确定性，seed 不影响
        fp = FrozenProbe().fit(X, Y_fit)
        predict = lambda w: fp.predict(w.reshape(1, -1)).ravel()
        # ensemble 权重必须用**样本外**(OOF) frozen 预测估计 —— in-sample 预测会高估泛化、推高 w 致跌破 naive
        F_oof = _oof_frozen_preds(X, Y) if target == "ensemble" else None
    elif substrate == "scratch":
        from . import _torch_models as tm            # 从头训 LSTM（σ_A 消融）
        tm.seed_all(seed)
        model = tm.train_forecaster(tm.LSTMForecaster(L_WIN, H_FORECAST), X, Y_fit, epochs=120)
        predict = lambda w: tm.forecast_predict(model, w.reshape(1, -1)).ravel()
        F_oof = tm.forecast_predict(model, X) if target == "ensemble" else None   # scratch 仅消融，免 OOF
    else:
        raise ValueError(f"substrate ∈ {{frozen, scratch}}, got {substrate!r}")

    w_blend = _blend_weight(F_oof, SN, Y) if target == "ensemble" else None

    errs = []
    for hh, s in zip(ready, batch):
        if not np.all(np.isfinite(hh)) or hh.size < L_WIN:
            continue
        win = hh[-L_WIN:]
        out = predict(win)
        fut = np.asarray(s.future)
        sn = _snaive(win, s.period, len(fut))
        h = min(len(out), len(fut), len(sn))
        if target == "seasonal_resid":
            pred = out[:h] + sn[:h]
        elif target == "ensemble":
            pred = w_blend * out[:h] + (1.0 - w_blend) * sn[:h]
        else:
            pred = out[:h]
        rmse = float(np.sqrt(np.mean((pred[:h] - fut[:h]) ** 2)))
        errs.append(rmse / (s.obs_scale + 1e-9))
    return float(np.mean(errs)) if errs else float("nan")


def seasonal_naive_floor(batch: List[ForecastSample]) -> float:
    """任务自然下限：y[t] = y[t-period]（用就绪历史外推）。供"可改进性"触发参照。"""
    errs = []
    for s in batch:
        hist = np.asarray(s.ready, float).ravel()
        p = max(2, int(s.period))
        if hist.size < p:
            continue
        pred = np.array([hist[-p + (i % p)] for i in range(len(s.future))])
        rmse = float(np.sqrt(np.mean((pred - np.asarray(s.future)) ** 2)))
        errs.append(rmse / (s.obs_scale + 1e-9))
    return float(np.mean(errs)) if errs else float("nan")
