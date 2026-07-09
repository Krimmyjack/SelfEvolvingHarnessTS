"""evaluators/role_a_proxy.py — Role A 轻量代理（Layer1 预筛，plan.md R7）。

廉价、与 grounded 同向但不裁决；只负向预筛"明显退化"的候选（不变量 #1：proxy 带歪只浪费验证轮）。
  forecast : Ridge 直接多步（DLinear 式）→ nRMSE
  anomaly  : spike 突出度缺失（无模型）→ 1 - prominence
  classify : 窗特征 + LogisticRegression → CV 交叉熵
全部 val_loss 越低越好，便于与 grounded 做 Spearman 校准（calibration.py）。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import (ForecastSample, AnomalySample, ClassifySample,
                   L_WIN, H_FORECAST, STRIDE, WIN_CLF, STRIDE_CLF)
from .grounded_forecast import build_windows
from .grounded_anomaly import _local_residual
from .grounded_classify import _windowize, _fold_by_series


# ── forecast proxy：DLinear 直接多步（torch，轻量）────────────────────────
def forecast_proxy(batch: List[ForecastSample], seed: int = 0) -> float:
    from . import _torch_models as tm
    ready = [np.asarray(s.ready, float).ravel() for s in batch]
    X, Y = build_windows(ready)
    if X is None or len(X) < 10:
        return float("nan")
    tm.seed_all(seed)
    model = tm.train_forecaster(tm.DLinear(L_WIN, H_FORECAST), X, Y, epochs=60)
    errs = []
    for hh, s in zip(ready, batch):
        if not np.all(np.isfinite(hh)) or hh.size < L_WIN:
            continue
        pred = tm.forecast_predict(model, hh[-L_WIN:].reshape(1, -1)).ravel()
        h = min(len(pred), len(s.future))
        rmse = float(np.sqrt(np.mean((pred[:h] - np.asarray(s.future)[:h]) ** 2)))
        errs.append(rmse / (s.obs_scale + 1e-9))
    return float(np.mean(errs)) if errs else float("nan")


# ── anomaly proxy：spike 突出度缺失（无模型，廉价）────────────────────────
def anomaly_proxy(batch: List[AnomalySample], seed: int = 0) -> float:
    deficits = []
    for s in batch:
        ready = np.asarray(s.ready, float).ravel()
        if not s.positions:
            continue
        z = (ready - ready.mean()) / (ready.std() + 1e-9)
        res = np.abs(_local_residual(z, 11))
        bg = np.percentile(res, 97) + 1e-9                 # top 背景残差
        prominence = float(np.mean(res[s.positions]) / bg)  # 注入位 spike 相对背景的突出度
        deficits.append(1.0 - float(np.clip(prominence, 0.0, 1.0)))
    return float(np.mean(deficits)) if deficits else float("nan")


# ── classify proxy：窗特征 + LogReg ───────────────────────────────────────
def _clf_features(win: np.ndarray) -> List[float]:
    w = np.asarray(win, float)
    spec = np.abs(np.fft.rfft(w - w.mean()))
    peak = int(np.argmax(spec[1:]) + 1) if spec.size > 1 else 0
    return [w.mean(), w.std(), float(np.mean(np.abs(np.diff(w)))),
            float(peak), float(spec.max() / (spec.sum() + 1e-9)),
            float(np.percentile(w, 90) - np.percentile(w, 10))]


def classify_proxy(batch: List[ClassifySample], seed: int = 0, n_folds: int = 5) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss

    labels = np.array([s.label for s in batch])
    if labels.size == 0:
        return float("nan")
    n_classes = int(labels.max() + 1)
    fold_of = _fold_by_series(labels, n_classes, n_folds, seed)

    ces = []
    for f in range(n_folds):
        Xtr, ytr, Xva, yva = [], [], [], []
        for i, s in enumerate(batch):
            for w, y in _windowize(s.ready, s.label):
                feat = _clf_features(w)
                (Xva if fold_of[i] == f else Xtr).append(feat)
                (yva if fold_of[i] == f else ytr).append(y)
        if not Xva or not Xtr or len(set(ytr)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=500).fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xva))
        ces.append(log_loss(yva, proba, labels=list(range(n_classes))))
    return float(np.mean(ces)) if ces else float("nan")
