"""operators/_common.py — 算子公共工具 + 契约。

Phase 0 算子契约：`fn(x: np.ndarray, **params) -> np.ndarray`，输入/输出均为 1D float 数组。
S1/S2/归一类算子保持长度（series→series）；改变形状的算子（sliding_window/lag/spectral）
在 Phase 0 1D pipeline 不支持，统一抛 ShapeChangingNotSupported（执行被 Sandbox Gate 拦）。
"""
from __future__ import annotations

import numpy as np

try:
    import scipy.signal as _sps        # noqa
    import scipy.ndimage as _ndi       # noqa
    _HAS_SCIPY = True
except Exception:                       # pragma: no cover
    _sps = None
    _ndi = None
    _HAS_SCIPY = False


# S0.7-8 边界语义（单一真源，落进实验 provenance）：平滑类算子统一 symmetric 镜像端点，
# 禁零填充（np.convolve mode="same" / scipy.signal.medfilt 会把末端拉向 0，而 forecasting
# 恰用末窗做编码输入——诊断实测 v_median 均值 OOF 1.4755→1.3131 全部来自边界）。
# 术语：这里的 "symmetric" = np.pad(mode="symmetric") = scipy.ndimage(mode="reflect")（边界值参与镜像）。
BOUNDARY_MODES = {
    "moving_average": "symmetric",     # 底层 helper
    "smooth_ma": "symmetric",          # 注册算子 = moving_average 的 NaN-safe 包装（F0 剂量维）
    "denoise_median": "symmetric",
    "denoise_savgol": "interp",        # scipy 端点多项式拟合（显式固定，非默认漂移）
    "denoise_wavelet": "symmetric",    # pywt symmetric（显式固定）
    "hampel_filter": "symmetric",      # E-3.3 R3：滚动中值/MAD 两个窗口同一 symmetric 语义
}

# 稳健尺度常数：MAD → σ（正态一致性因子）。与 benchmark/corruption.robust_scale 同一口径，
# 故"1 个 robust σ 的跳变"在算子侧与腐蚀侧含义一致（两处独立定义，不互相 import：
# operators/ 不得依赖 benchmark/——算子库是被 benchmark 度量的对象，不能反向耦合）。
MAD_TO_SIGMA = 1.4826


class ShapeChangingNotSupported(NotImplementedError):
    """形状改变算子在 Phase 0 1D 流水线不支持。"""


def as_1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("empty series")
    return a


def interp_nan(x: np.ndarray) -> np.ndarray:
    """线性插补 NaN（首尾用最近值钳制）。全 NaN → 全 0。"""
    x = x.copy()
    m = np.isnan(x)
    if not m.any():
        return x
    if m.all():
        return np.zeros_like(x)
    idx = np.arange(x.size)
    x[m] = np.interp(idx[m], idx[~m], x[~m])
    return x


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """滑动均值（S0.7-8：symmetric 镜像 padding + valid 卷积；window>n 钳到 n）。"""
    window = max(1, int(window))
    n = x.size
    if window <= 1 or n <= 1:
        return x.copy()
    window = min(window, n)
    pad_l = (window - 1) // 2
    pad_r = window - 1 - pad_l
    xp = np.pad(x, (pad_l, pad_r), mode="symmetric")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")


def sliding_median_symmetric(y: np.ndarray, w: int) -> np.ndarray:
    """numpy 滑动中值，symmetric 镜像边界——与 `scipy.ndimage.median_filter(mode="reflect")`
    逐点一致（奇数 w），供 scipy 缺失回退，保证两条路径同边界语义（S0.7-8）。"""
    half = (w - 1) // 2
    yp = np.pad(y, (half, half), mode="symmetric")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(yp, w), axis=1)


def robust_sigma(y: np.ndarray) -> float:
    """1.4826 × MAD —— 稳健尺度估计。全常数序列 MAD=0 → 回退 std → 回退 0.0（调用方须自行
    处理零尺度：**零尺度下任何"偏离多少个 σ"的判据都无定义，必须弃权而非除零**）。"""
    a = np.asarray(y, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    mad = float(np.median(np.abs(a - np.median(a))))
    scale = MAD_TO_SIGMA * mad
    if scale <= 0.0:
        scale = float(np.std(a))
    return max(scale, 0.0)


def odd_window(w: int, n: int) -> int:
    """把窗口钳成 ≤n 的奇数（中值/MAD 类算子要求奇窗，否则中位数落在两点之间失定义）。
    返回 <3 表示"窗口不可用"，调用方应恒等返回。"""
    w = max(1, int(w))
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return w


def sliding_mad_symmetric(y: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """滚动中值与滚动 MAD（同一 symmetric 边界，同一窗口）——Hampel 判据的两个输入。

    返回 (med, mad)：med[i] = median(窗口 i)，mad[i] = median(|y − med| 在窗口 i 内)。
    **注意 mad 是对偏差序列再做一次滚动中值**（Hampel 原始定义），不是"窗口内 |y−med[i]| 的
    中位数"的近似——两者在窗口内 med 不变时等价，边界处因镜像而略有差异，这里统一走镜像。"""
    med = sliding_median_symmetric(y, w)
    return med, sliding_median_symmetric(np.abs(y - med), w)
