from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinear(nn.Module):
    def __init__(self, L: int, H: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.lin_trend = nn.Linear(L, H)
        self.lin_season = nn.Linear(L, H)

    def forward(self, x):
        xp = F.pad(x.unsqueeze(1), (self.kernel // 2, self.kernel // 2), mode="replicate")
        trend = F.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)[:, : x.size(1)]
        season = x - trend
        return self.lin_trend(trend) + self.lin_season(season)


class LSTMForecaster(nn.Module):
    def __init__(self, L: int, H: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :])


__all__ = ["DLinear", "LSTMForecaster"]
