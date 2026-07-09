"""evaluators/grounded_classify.py — §2.E.3 classify grounded：分窗 + 按序列分层 CV 交叉熵。

grounded = InceptionTime-lite（torch，需 `project` 环境），分窗上训练（CV by series）；
val_loss = mean CV cross-entropy（越低越好）。注：plan.md A.4/E3c 指出 clf 是结构性弱点。
"""
from __future__ import annotations

from typing import List

import numpy as np

from .base import ClassifySample, WIN_CLF, STRIDE_CLF


def _windowize(series_ready: np.ndarray, label: int):
    s = np.asarray(series_ready, float)
    return [(s[st:st + WIN_CLF], label) for st in range(0, s.size - WIN_CLF + 1, STRIDE_CLF)]


def _fold_by_series(labels: np.ndarray, n_classes: int, n_folds: int, seed: int) -> np.ndarray:
    """按序列分层折分配（同一序列的窗不跨折，防泄漏）。"""
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(labels), dtype=int)
    idx = np.arange(len(labels))
    for c in range(n_classes):
        ci = idx[labels == c]
        rng.shuffle(ci)
        for j, i in enumerate(ci):
            fold_of[i] = j % n_folds
    return fold_of


# in-loop 判官底座：'inception'（默认，from-scratch InceptionLite，σ>0）/ 'rocket'（确定性 ROCKET+LogReg）。
# 默认保持 inception → 既有 evaluators/slow_path 测试行为不变；实验里 set_classify_substrate('rocket')
# 把判官换成确定性版（同时让 report_target 的 InceptionLite 报告器与判官天然不相交）。
_CLASSIFY_SUBSTRATE = "inception"


def set_classify_substrate(name: str) -> None:
    """全局切换 classify grounded 判官底座（对 Evaluator/validator 零侵入）。name ∈ {inception, rocket}。"""
    global _CLASSIFY_SUBSTRATE
    if name not in ("inception", "rocket"):
        raise ValueError(f"classify substrate ∈ {{inception, rocket}}, got {name!r}")
    _CLASSIFY_SUBSTRATE = name


def get_classify_substrate() -> str:
    return _CLASSIFY_SUBSTRATE


def classify_inception(batch: List[ClassifySample], seed: int = 0, n_folds: int = 5) -> float:
    """InceptionTime-lite CV 交叉熵（from-scratch，σ>0）。report_target 显式调本函数作独立报告器。"""
    from . import _torch_models as tm

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
                (Xva if fold_of[i] == f else Xtr).append(w)
                (yva if fold_of[i] == f else ytr).append(y)
        if not Xva or not Xtr or len(set(ytr)) < 2:
            continue
        tm.seed_all(seed)
        model = tm.train_classifier(tm.InceptionLite(n_classes), Xtr, ytr, epochs=60)
        ces.append(tm.classifier_ce(model, Xva, yva))
    return float(np.mean(ces)) if ces else float("nan")


def classify_grounded(batch: List[ClassifySample], seed: int = 0, n_folds: int = 5) -> float:
    """in-loop 判官派发（按 _CLASSIFY_SUBSTRATE）。注意：报告器必须显式调 classify_inception，
    **不要**走本派发，否则 substrate=rocket 时报告器=判官 → 循环自证。"""
    if _CLASSIFY_SUBSTRATE == "rocket":
        from .rocket_probe import classify_grounded_rocket
        return classify_grounded_rocket(batch, seed=seed, n_folds=n_folds)
    return classify_inception(batch, seed=seed, n_folds=n_folds)
