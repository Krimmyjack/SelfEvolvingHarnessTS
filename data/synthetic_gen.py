"""data/synthetic_gen.py — 合成 (clean, degraded, label) 三元组（plan.md §2，移植自 E0-min 探索）。

x = trend + seasonal + micro-noise，再退化（加噪 + 离群 + 缺失）。三任务标签：
  forecast : 留出 clean 末 H 段为真未来；history = degraded[:N-H]
  anomaly  : 在尾段注入 K 个已知异常（spike/level_shift/plateau），记录位置
  classify : 类别 = 季节周期（低频可学但 SNR-limited；噪声/spike 为类内 nuisance）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

LENGTH = 512
H_FORECAST = 48          # 预测视野（真未来留出）
K_ANOM = 5               # 每条注入异常数
ANOM_TOL = 2             # recall@injected 的检测容差（±样本）

PATTERNS = ("P1", "P2", "P3")
PATTERN_PARAMS = {
    "P1": dict(period=24, seasonal_amp=1.0, trend_slope=0.0,   noise=0.5, miss_rate=0.02, miss_kind="scattered", outlier_rate=0.03, outlier_mag=5.0),
    "P2": dict(period=24, seasonal_amp=0.3, trend_slope=0.01,  noise=1.0, miss_rate=0.05, miss_kind="scattered", outlier_rate=0.01, outlier_mag=5.0),
    "P3": dict(period=24, seasonal_amp=0.8, trend_slope=0.005, noise=0.3, miss_rate=0.10, miss_kind="block",     outlier_rate=0.01, outlier_mag=5.0),
    # 网格预设（跨 SNR×missing 4 格，均带离群 → forecast 在降级 harness 下都有 winsorize 头部空间）
    "G_hi_full": dict(period=24, seasonal_amp=1.0, trend_slope=0.005, noise=0.05, miss_rate=0.00, miss_kind="scattered", outlier_rate=0.03, outlier_mag=5.0),
    "G_hi_miss": dict(period=24, seasonal_amp=1.0, trend_slope=0.005, noise=0.05, miss_rate=0.05, miss_kind="scattered", outlier_rate=0.03, outlier_mag=5.0),
    "G_lo_full": dict(period=24, seasonal_amp=1.0, trend_slope=0.005, noise=0.70, miss_rate=0.00, miss_kind="scattered", outlier_rate=0.03, outlier_mag=5.0),
    "G_lo_miss": dict(period=24, seasonal_amp=1.0, trend_slope=0.005, noise=0.70, miss_rate=0.05, miss_kind="scattered", outlier_rate=0.03, outlier_mag=5.0),
}

CLF_PERIODS = (18, 24, 30)


@dataclass
class RawSeries:
    pattern: str
    task: str
    seed: int
    period: int
    obs_scale: float
    clean: np.ndarray
    degraded: np.ndarray
    history: Optional[np.ndarray] = None         # forecast 输入（harness 喂这个）
    clean_history: Optional[np.ndarray] = None
    future: Optional[np.ndarray] = None          # forecast 真未来（clean[-H:]）
    anomaly_input: Optional[np.ndarray] = None   # anomaly 输入（含注入异常）
    anomaly_positions: List[int] = field(default_factory=list)
    label: Optional[int] = None                  # classify 类别
    origin: str = ""                             # 数据集/网格预设名（S0.2 分层用）
    series_uid: str = ""                         # 基底序列身份（S0.2 分组用：同基底副本共享）


def _make_clean(p: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(LENGTH)
    trend = p["trend_slope"] * t
    seasonal = p["seasonal_amp"] * (np.sin(2 * np.pi * t / p["period"])
                                    + 0.3 * np.sin(2 * np.pi * t / (p["period"] / 2.0)))
    return trend + seasonal + rng.normal(0, 0.02, LENGTH)


def _degrade(clean: np.ndarray, p: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 10_000)
    x = clean.astype(float).copy()
    n = x.size
    base_std = float(np.std(clean)) or 1.0
    if p["noise"] > 0:
        x = x + rng.normal(0, p["noise"], n)
    n_out = int(round(p["outlier_rate"] * n))
    if n_out > 0:
        idx = rng.choice(n, size=n_out, replace=False)
        x[idx] += rng.choice([-1.0, 1.0], size=n_out) * p["outlier_mag"] * base_std
    n_miss = int(round(p["miss_rate"] * n))
    if n_miss > 0:
        if p["miss_kind"] == "block":
            start = int(rng.integers(0, max(1, n - n_miss)))
            x[start:start + n_miss] = np.nan
        else:
            x[rng.choice(n, size=n_miss, replace=False)] = np.nan
    return x


def _inject_anomalies(series: np.ndarray, base_std: float, seed: int):
    rng = np.random.default_rng(seed + 20_000)
    x = series.astype(float).copy()
    n = x.size
    seg_lo, seg_hi = int(n * 0.78), n - 4
    positions = sorted(rng.choice(np.arange(seg_lo, seg_hi), size=K_ANOM, replace=False).tolist())
    for i, pos in enumerate(positions):
        kind = ("spike", "spike", "spike", "level_shift", "plateau")[i % 5]
        base = x[pos] if not np.isnan(x[pos]) else float(np.nanmedian(x))
        if kind == "spike":
            x[pos] = base + (1.0 if rng.random() < 0.5 else -1.0) * 6.0 * base_std
        elif kind == "level_shift":
            x[pos:pos + 3] = base + 4.0 * base_std
        else:
            x[pos:pos + 3] = base
    return x, positions


def make_forecast_series(pattern: str, seed: int) -> RawSeries:
    p = PATTERN_PARAMS[pattern]
    clean = _make_clean(p, seed)
    degraded = _degrade(clean, p, seed)
    cut = LENGTH - H_FORECAST
    return RawSeries(pattern, "forecast", seed, p["period"],
                     obs_scale=float(np.std(clean[cut:])) or 1.0,
                     clean=clean, degraded=degraded,
                     history=degraded[:cut].copy(), clean_history=clean[:cut].copy(),
                     future=clean[cut:].copy(),
                     origin=pattern, series_uid=f"{pattern}:forecast:{seed}")


def make_anomaly_series(pattern: str, seed: int) -> RawSeries:
    p = PATTERN_PARAMS[pattern]
    clean = _make_clean(p, seed)
    degraded = _degrade(clean, p, seed)
    base_std = float(np.std(clean)) or 1.0
    anom, pos = _inject_anomalies(degraded, base_std, seed)
    return RawSeries(pattern, "anomaly_detection", seed, p["period"], base_std,
                     clean=clean, degraded=degraded, anomaly_input=anom, anomaly_positions=pos,
                     origin=pattern, series_uid=f"{pattern}:anomaly:{seed}")


def make_classify_series(cls: int, seed: int) -> RawSeries:
    period = CLF_PERIODS[cls]
    rng = np.random.default_rng(seed)
    t = np.arange(LENGTH)
    clean = 0.003 * t + 0.8 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.02, LENGTH)
    base_std = float(np.std(clean)) or 1.0
    x = clean + rng.normal(0, 0.5, LENGTH)
    n_out = int(round(0.04 * LENGTH))
    x[rng.choice(LENGTH, size=n_out, replace=False)] += rng.choice([-1.0, 1.0], size=n_out) * 8.0 * base_std
    x[rng.choice(LENGTH, size=int(round(0.03 * LENGTH)), replace=False)] = np.nan
    return RawSeries("clf", "classification", seed, period, base_std,
                     clean=clean, degraded=x, label=cls,
                     origin=f"clf{cls}", series_uid=f"clf{cls}:{seed}")


def make_forecast_batch(pattern: str, n: int = 20, seed0: int = 0) -> List[RawSeries]:
    return [make_forecast_series(pattern, seed0 + i) for i in range(n)]


def make_anomaly_batch(pattern: str, n: int = 20, seed0: int = 0) -> List[RawSeries]:
    return [make_anomaly_series(pattern, seed0 + i) for i in range(n)]


def make_classify_batch(n_per_class: int = 10, seed0: int = 0) -> List[RawSeries]:
    return [make_classify_series(cls, seed0 + cls * 100 + i)
            for cls in range(len(CLF_PERIODS)) for i in range(n_per_class)]
