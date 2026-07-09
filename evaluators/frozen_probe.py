"""evaluators/frozen_probe.py — ★ R10：frozen encoder + linear probe（grounded 默认底座）。

E2c 已证：frozen+probe 把 σ_A 压到 ~0（相对 from-scratch 训练 12× 降），ε 落到 0.030——grounded
评估的**必要**条件（否则慢路径被训练噪声淹没）。

实现 = E2c 范式：LSTM 编码器在留出合成集上**预训一次 + 冻结**（缓存到 _artifacts/），每次评估只在冻结
特征上拟合 **Ridge 闭式头**（确定性 → σ_A≈0，对应 E2c 的 _ridge_probe 参照）。接真 foundation
encoder（TimesFM/Moirai/Chronos）时只换 get_frozen_encoder()，对外接口不变。
"""
from __future__ import annotations

import pathlib

import numpy as np
from sklearn.linear_model import Ridge

from .base import L_WIN, H_FORECAST

HIDDEN = 64
_ART = pathlib.Path(__file__).resolve().parent / "_artifacts"
_ENCODER = None          # 进程内单例（冻结编码器）


def _pretrain_encoder():
    """在留出合成集（P1/P2/P3 干净历史，seed 500+）上训 LSTM，返回其 .lstm 编码器（学域内时序结构）。"""
    from . import _torch_models as tm
    from .grounded_forecast import build_windows
    from ..data import make_forecast_batch

    tm.seed_all(42)
    batch = []
    for pat in ("P1", "P2", "P3"):
        batch += make_forecast_batch(pat, 15, seed0=500)
    ready = [s.clean_history for s in batch]
    X, Y = build_windows(ready)
    model = tm.LSTMForecaster(L_WIN, H_FORECAST, hidden=HIDDEN)
    tm.train_forecaster(model, X, Y, epochs=120)
    return model.lstm


def get_frozen_encoder():
    """惰性单例：加载缓存的冻结 LSTM 编码器；无缓存则预训一次并保存。"""
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    from . import _torch_models as tm
    import torch

    _ART.mkdir(exist_ok=True)
    path = _ART / f"frozen_lstm_h{HIDDEN}.pt"
    if path.exists():
        enc = tm.nn.LSTM(1, HIDDEN, num_layers=1, batch_first=True).to(tm.DEVICE)
        enc.load_state_dict(torch.load(path, map_location=tm.DEVICE))
    else:
        enc = _pretrain_encoder().to(tm.DEVICE)
        torch.save(enc.state_dict(), path)
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    _ENCODER = enc
    return enc


def set_frozen_encoder(encoder) -> None:
    """覆盖进程内冻结编码器单例（E2：用真实留出集预训的编码器跑 grounded forecast）。

    grounded_forecast.FrozenProbe() 默认取该单例 → 一次设置即可让整个 grounded 链路换编码器，
    对 evaluator/validator 接口零侵入。传 None 复位（下次回落到合成预训缓存）。
    """
    global _ENCODER
    _ENCODER = encoder


def load_frozen_encoder(path, *, hidden: int = HIDDEN):
    """从缓存 state_dict 加载冻结 LSTM 编码器（供已预训好的真实编码器复用，免每跑重训）。"""
    from . import _torch_models as tm
    import torch
    enc = tm.nn.LSTM(1, hidden, num_layers=1, batch_first=True).to(tm.DEVICE)
    enc.load_state_dict(torch.load(path, map_location=tm.DEVICE))
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    return enc


def pretrain_encoder_real(histories, *, epochs: int = 120, cache_path=None, hidden: int = HIDDEN):
    """在**真实留出**历史窗口上预训 LSTM 编码器并冻结返回（leave-signal-out，防泄漏 → 公平 E2）。

    histories: List[1-D array]（真实 z-score 序列的 clean_history，来自 encoder-pretrain 划分，与 eval 不相交）。
    """
    from . import _torch_models as tm
    from .grounded_forecast import build_windows
    import torch

    tm.seed_all(42)
    X, Y = build_windows([np.asarray(h, float).ravel() for h in histories])
    if X is None or len(X) < 10:
        raise ValueError(f"真实预训窗口不足（got {0 if X is None else len(X)}）——增大 encoder-pretrain 集或降 frac")
    model = tm.LSTMForecaster(L_WIN, H_FORECAST, hidden=hidden)
    tm.train_forecaster(model, X, Y, epochs=epochs)
    enc = model.lstm
    for p in enc.parameters():
        p.requires_grad = False
    enc.eval()
    if cache_path is not None:
        torch.save(enc.state_dict(), cache_path)
    return enc


class FrozenProbe:
    """冻结 LSTM 编码器特征 + Ridge 头。确定性、多输出（forecast H 维）。σ_A≈0。"""

    def __init__(self, encoder=None, ridge_alpha: float = 1.0):
        self.encoder = encoder if encoder is not None else get_frozen_encoder()
        self.alpha = ridge_alpha
        self.head = None

    def transform(self, windows) -> np.ndarray:
        from . import _torch_models as tm
        return tm.lstm_encode(self.encoder, np.asarray(windows, dtype=np.float32))

    def fit(self, X_windows, Y) -> "FrozenProbe":
        self.head = Ridge(alpha=self.alpha)
        self.head.fit(self.transform(X_windows), np.asarray(Y, dtype=float))
        return self

    def predict(self, X_windows) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("FrozenProbe not fitted")
        return self.head.predict(self.transform(X_windows))
