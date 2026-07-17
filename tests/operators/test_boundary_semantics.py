"""tests/test_boundary_semantics.py — S0.7-8 边界语义回归测试（A-31b，评审第十一轮）。

背景：旧 `moving_average`（np.convolve mode="same"）与旧 `denoise_median`（scipy.signal.medfilt）
都是零填充——末端 (w−1)/2 点被拉向 0，而 forecasting 恰用末窗做编码输入。诊断实测该缺陷把
v_median 均值 OOF 从 1.3131 压到 1.4755（见 results/E1_1_v2/decision.md 追录 2）。

守卫内容：①常数序列全位置保持；②线性趋势端点偏差有界（远小于零填充失真）；③长度保持；
④短序列/window≥n 不崩且语义合理；⑤scipy 与 numpy 回退路径逐点一致；⑥BOUNDARY_MODES 指纹存在。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.operators import s1_denoise as den
from SelfEvolvingHarnessTS.operators._common import (
    BOUNDARY_MODES, moving_average, sliding_median_symmetric)

N = 400
WINDOWS = [5, 9, 15, 25]


# ── ① 常数序列：全位置保持（零填充版在端点会掉到 ~w/2w） ─────────────────
@pytest.mark.parametrize("w", WINDOWS)
def test_ma_constant_preserved_all_positions(w):
    x = np.ones(N)
    out = moving_average(x, w)
    assert out.shape == (N,)
    assert np.allclose(out, 1.0), f"MA(w={w}) 常数序列端点失真 max|Δ|={np.abs(out-1).max():.3f}（零填充残留？）"


@pytest.mark.parametrize("w", WINDOWS)
def test_median_constant_preserved_all_positions(w):
    x = np.ones(N)
    out = den.denoise_median(x, window=w)
    assert out.shape == (N,)
    assert np.allclose(out, 1.0), f"median(w={w}) 常数序列端点失真（零填充残留？）"


# ── ② 线性趋势端点偏差有界 ────────────────────────────────────────────────
@pytest.mark.parametrize("w", WINDOWS)
def test_ma_linear_trend_endpoint_bounded(w):
    x = np.linspace(1.0, 2.0, N)                 # 斜率 slope=1/(N-1)
    out = moving_average(x, w)
    slope = 1.0 / (N - 1)
    # symmetric 镜像下端点偏差 ≤ slope*w/2（镜像把趋势折回）；零填充版偏差 ~x[-1]*(w//2)/w ≈ 0.5·x[-1]
    assert abs(out[-1] - x[-1]) <= slope * w, f"MA(w={w}) 端点偏差 {abs(out[-1]-x[-1]):.4f} 超镜像界"
    assert abs(out[0] - x[0]) <= slope * w


@pytest.mark.parametrize("w", WINDOWS)
def test_median_linear_trend_endpoint_bounded(w):
    x = np.linspace(1.0, 2.0, N)
    out = den.denoise_median(x, window=w)
    slope = 1.0 / (N - 1)
    # symmetric 镜像中值端点偏差 ≈ slope*w/4；旧零填充 w=25 时把 2.0 拉到 ~1.97（偏差≈12·slope）
    assert abs(out[-1] - x[-1]) <= slope * w, f"median(w={w}) 端点偏差 {abs(out[-1]-x[-1]):.4f} 超镜像界"
    assert abs(out[0] - x[0]) <= slope * w


# ── ③④ 长度保持 / 短序列 / window≥n ──────────────────────────────────────
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("w", [3, 5, 9, 25])
def test_short_series_and_large_window(n, w):
    x = np.linspace(0.0, 1.0, n) if n > 1 else np.array([1.0])
    ma = moving_average(x, w)
    md = den.denoise_median(x, window=w)
    assert ma.shape == (n,) and md.shape == (n,)
    assert np.all(np.isfinite(ma)) and np.all(np.isfinite(md))
    # 值域不越界（均值/中值都是凸组合/序统计量，symmetric 镜像不引入池外值）
    assert ma.min() >= x.min() - 1e-12 and ma.max() <= x.max() + 1e-12
    assert md.min() >= x.min() - 1e-12 and md.max() <= x.max() + 1e-12


def test_even_window_rounded_up():
    x = np.linspace(0.0, 1.0, 50)
    assert np.allclose(den.denoise_median(x, window=4), den.denoise_median(x, window=5))


# ── ⑤ scipy 路径 与 numpy 回退路径 逐点一致 ───────────────────────────────
@pytest.mark.parametrize("w", [5, 9, 25])
def test_median_scipy_numpy_paths_identical(w, monkeypatch):
    rng = np.random.default_rng(7)
    x = np.sin(2 * np.pi * np.arange(N) / 24) + rng.normal(0, 0.3, N)
    out_scipy = den.denoise_median(x, window=w)
    monkeypatch.setattr(den, "_HAS_SCIPY", False)
    out_numpy = den.denoise_median(x, window=w)
    assert np.allclose(out_scipy, out_numpy), f"median(w={w}) scipy/numpy 回退边界语义不一致"


@pytest.mark.parametrize("w", [5, 9, 25])
def test_sliding_median_symmetric_matches_ndimage(w):
    from scipy.ndimage import median_filter
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 200)
    assert np.allclose(sliding_median_symmetric(x, w), median_filter(x, size=w, mode="reflect"))


# ── ⑥ 边界语义指纹（落 provenance 用）──────────────────────────────────────
def test_boundary_modes_fingerprint():
    for op in ("moving_average", "denoise_median", "denoise_savgol", "denoise_wavelet"):
        assert op in BOUNDARY_MODES
    assert BOUNDARY_MODES["denoise_median"] == "symmetric"
    assert BOUNDARY_MODES["denoise_savgol"] == "interp"


# ── 回归锚：旧零填充行为不得复活（直接检测末端塌陷特征） ─────────────────
def test_no_zero_padding_collapse_regression():
    x = np.full(N, 5.0)                          # 远离 0 的常数：零填充端点会塌向 0
    for w in (9, 15, 25):
        assert abs(moving_average(x, w)[-1] - 5.0) < 1e-9
        assert abs(den.denoise_median(x, window=w)[-1] - 5.0) < 1e-9
