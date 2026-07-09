"""evaluators/rocket_probe.py — ★ classify 的确定性 frozen+probe 类比（ROCKET-lite + LogReg）。

动机（BUILD §5 结论 #3 + §6 技术债）：classify 现判官 InceptionLite 是 from-scratch 训练 →
σ>0，正是 frozen+probe 当初要消的训练噪声（"长跑少跑 classify"）。本模块把 classify 判官也做成
**确定性**：随机卷积核（seed 固定即冻结，不训练）→ ppv+max 池化特征 → LogisticRegression 线性头。
随机特征 + 线性头 ⇒ 给定 seed 完全确定（σ=0），与 forecast 的 frozen-LSTM-probe 同哲学；且作为
in-loop 判官，与 from-scratch InceptionLite **报告器**天然不相交（守判官↔报告器分离）。

ROCKET = RandOm Convolutional KErnel Transform（Dempster 2020）的轻量复刻：无 sktime/aeon 依赖
（本环境未装），纯 numpy 卷积 + sklearn LogReg。窗化/分折复用 grounded_classify（同协议、可比）。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import ClassifySample
from .grounded_classify import _windowize, _fold_by_series

N_KERNELS = 256


# ════════════════════════════ 随机卷积核（seed 冻结）════════════════════════════
def _make_kernels(n_kernels: int, seed: int, max_len: int):
    rng = np.random.default_rng(seed)
    kernels = []
    for _ in range(n_kernels):
        l = int(rng.choice([7, 9, 11]))
        w = rng.standard_normal(l)
        w -= w.mean()                                   # ROCKET：核去均值
        b = float(rng.uniform(-1.0, 1.0))
        max_exp = int(np.log2(max(1, (max_len - 1) // (l - 1)))) if max_len > l else 0
        d = int(2 ** rng.integers(0, max_exp + 1))      # 膨胀 ∈ {1,2,4,...}
        kernels.append((w, b, d))
    return kernels


def _apply_kernel(X: np.ndarray, w: np.ndarray, b: float, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """valid 膨胀卷积 → (max, ppv) 两特征（每行一标量）。X:(n,L)。"""
    l = len(w)
    n_out = X.shape[1] - (l - 1) * d
    if n_out <= 0:                                       # 感受野超过序列长 → 退化为整窗一个位置
        z = (X * w[: X.shape[1]].sum() if l >= X.shape[1] else X[:, :1] * 0.0) + b
        return z.max(1), (z > 0).mean(1)
    out = np.zeros((X.shape[0], n_out), dtype=float)
    for j in range(l):
        out += w[j] * X[:, j * d: j * d + n_out]
    z = out + b
    return z.max(axis=1), (z > 0).mean(axis=1)


def rocket_transform(X: np.ndarray, kernels) -> np.ndarray:
    """X:(n,L) → 特征 (n, 2*K)。确定性（核固定）。"""
    feats = []
    for w, b, d in kernels:
        mx, pv = _apply_kernel(X, w, b, d)
        feats.append(mx); feats.append(pv)
    return np.stack(feats, axis=1)


# ════════════════════════════ grounded classify（确定性）════════════════════════════
def classify_grounded_rocket(batch: List[ClassifySample], seed: int = 0, n_folds: int = 5) -> float:
    """按序列分层 CV 的 ROCKET+LogReg 交叉熵（越低越好）。确定性：σ=0（核 seed 固定 + LogReg lbfgs）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss

    labels = np.array([s.label for s in batch])
    if labels.size == 0:
        return float("nan")
    n_classes = int(labels.max() + 1)
    fold_of = _fold_by_series(labels, n_classes, n_folds, seed)

    # 预窗化（用一致的 WIN_CLF/STRIDE_CLF）+ 记录每窗所属序列折
    win_max = max((np.asarray(s.ready, float).size for s in batch), default=0)
    kernels = _make_kernels(N_KERNELS, seed, max_len=min(64, win_max))

    ces = []
    for f in range(n_folds):
        Xtr_w, ytr, Xva_w, yva = [], [], [], []
        for i, s in enumerate(batch):
            for w, y in _windowize(s.ready, s.label):
                (Xva_w if fold_of[i] == f else Xtr_w).append(w)
                (yva if fold_of[i] == f else ytr).append(y)
        if not Xva_w or not Xtr_w or len(set(ytr)) < 2:
            continue
        Ftr = rocket_transform(np.asarray(Xtr_w, float), kernels)
        Fva = rocket_transform(np.asarray(Xva_w, float), kernels)
        sc = StandardScaler().fit(Ftr)
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Ftr), ytr)
        proba = clf.predict_proba(sc.transform(Fva))
        ces.append(log_loss(yva, proba, labels=list(range(n_classes))))
    return float(np.mean(ces)) if ces else float("nan")
