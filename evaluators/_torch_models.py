"""evaluators/_torch_models.py — torch 下游模型（移植自 E0 探索 e0_torch_eval.py）。

需 conda `project` 环境（torch + CUDA）。本模块在各 evaluator 内**惰性 import**，故 evaluators 包
在无 torch 环境仍可 import（只有实际评估调用才需要 torch）。
  forecast grounded = LSTMForecaster；proxy = DLinear；frozen 底座 = LSTM 编码器（预训+冻结）。
  classification grounded = InceptionLite。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================== forecasting ===============================
class DLinear(nn.Module):
    """series 分解（移动平均趋势）+ 双分量线性，直接多步。Role-A 轻量代理预测器。"""

    def __init__(self, L: int, H: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.lin_trend = nn.Linear(L, H)
        self.lin_season = nn.Linear(L, H)

    def forward(self, x):  # (B, L)
        xp = F.pad(x.unsqueeze(1), (self.kernel // 2, self.kernel // 2), mode="replicate")
        trend = F.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)[:, : x.size(1)]
        season = x - trend
        return self.lin_trend(trend) + self.lin_season(season)


class LSTMForecaster(nn.Module):
    """LSTM 编码器 -> 线性头，直接多步。grounded 预测器；.lstm 即可冻结作 frozen 编码器。"""

    def __init__(self, L: int, H: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):  # (B, L)
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :])


def train_forecaster(model, X, Y, epochs: int, lr: float = 1e-2, bs: int = 256):
    model.to(DEVICE).train()
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(np.asarray(Y), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.mse_loss(model(Xt[idx]), Yt[idx]).backward()
            opt.step()
    return model


def forecast_predict(model, windows):
    """windows: (n, L) numpy → (n, H) numpy。"""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(np.asarray(windows), dtype=torch.float32, device=DEVICE)
        return model(Xt).cpu().numpy()


def lstm_encode(encoder, windows):
    """frozen LSTM 编码器：windows (n, L) → 末隐状态特征 (n, hidden)。"""
    encoder.eval()
    with torch.no_grad():
        Xt = torch.tensor(np.asarray(windows, dtype=np.float32), device=DEVICE)
        out, _ = encoder(Xt.unsqueeze(-1))
        return out[:, -1, :].cpu().numpy()


# ============================== classification ============================
class InceptionLite(nn.Module):
    """InceptionTime-lite：2 个多核 1D 卷积块 + 全局平均池化 + 线性。grounded 分类器。"""

    def __init__(self, n_classes: int, ch: int = 32):
        super().__init__()
        self.b1 = nn.ModuleList([nn.Conv1d(1, ch, k, padding=k // 2) for k in (3, 9, 19)])
        self.bn1 = nn.BatchNorm1d(ch * 3)
        self.b2 = nn.ModuleList([nn.Conv1d(ch * 3, ch, k, padding=k // 2) for k in (3, 9, 19)])
        self.bn2 = nn.BatchNorm1d(ch * 3)
        self.head = nn.Linear(ch * 3, n_classes)

    def forward(self, x):  # (B, W)
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(torch.cat([c(x) for c in self.b1], dim=1)))
        x = F.relu(self.bn2(torch.cat([c(x) for c in self.b2], dim=1)))
        return self.head(x.mean(dim=-1))


def train_classifier(model, X, Y, epochs: int = 60, lr: float = 3e-3, bs: int = 128):
    model.to(DEVICE).train()
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32, device=DEVICE)
    Yt = torch.tensor(np.asarray(Y), dtype=torch.long, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xt)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(Xt[idx]), Yt[idx]).backward()
            opt.step()
    return model


def classifier_ce(model, Xva, Yva) -> float:
    model.eval()
    with torch.no_grad():
        xv = torch.tensor(np.asarray(Xva), dtype=torch.float32, device=DEVICE)
        yv = torch.tensor(np.asarray(Yva), dtype=torch.long, device=DEVICE)
        return float(F.cross_entropy(model(xv), yv).item())
