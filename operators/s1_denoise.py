"""operators/s1_denoise.py — S1 去噪。scipy 可用则用 Savitzky-Golay/中值，否则 numpy 回退。

S0.7 Operator Integrity：所有静默回退改为**显式记录**（_provenance.record），且 _guess_period 修复
（detrend + 周期范围 + 峰值显著性 + 无周期状态），wavelet 换 AegisTS 验证过的写法（有界 level +
symmetric mode + 仅修脏点），杜绝旧版病态（整段 periodization 重构 → nRMSE≈2×identity）。
"""
from __future__ import annotations

import math

import numpy as np

from ._common import as_1d, interp_nan, moving_average, sliding_median_symmetric, _HAS_SCIPY, _sps, _ndi
from ._provenance import record


def denoise_savgol(x, window: int = 11, order: int = 3, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    n = y.size
    w = min(int(window), n if n % 2 == 1 else n - 1)
    if w < 3:
        return y
    if w % 2 == 0:
        w += 1
    if _HAS_SCIPY and w <= n:
        # S0.7-8：端点显式固定 mode="interp"（多项式端点拟合，无零填充；防 scipy 默认漂移）
        return _sps.savgol_filter(y, window_length=w, polyorder=min(order, w - 1), mode="interp")
    return moving_average(y, w)        # 回退（moving_average 已是 symmetric 边界）


def smooth_ma(x, window: int = 5, **_) -> np.ndarray:
    """滑动均值平滑（NaN-safe 包装 `_common.moving_average`；S0.7-8 symmetric 边界）。

    E-3.3 Family 0 剂量维专用的**独立注册算子**（MA 此前只是 helper，A-32/评审第十二轮要求
    带契约入池）。window 由模板 `params_override` 注入（剂量扫描 9/15/25）；边界语义与
    `moving_average` 一致（symmetric 镜像 + valid 卷积，无零填充），落 BOUNDARY_MODES。"""
    y = interp_nan(as_1d(x))
    return moving_average(y, int(window))


def denoise_median(x, window: int = 5, strength: float = 1.0, **_) -> np.ndarray:
    """滑动中值（S0.7-8 边界修复：symmetric 镜像边界，弃 `scipy.signal.medfilt` 零填充）。

    旧版 medfilt 零填充把末端 (w−1)/2 点拉向 0，而 forecasting 恰用末窗做编码输入——
    诊断实测该缺陷压低 v_median 均值 OOF 1.4755→1.3131（详 results/E1_1_v2/decision.md 追录 2）。
    scipy 路径 = `ndimage.median_filter(mode="reflect")`；numpy 回退 = 同 padding 语义的滑动中值
    （两路径逐点一致，S0.7-8 语义一致性测试守卫）。偶数窗上调为奇数；window≥n 钳到最大奇数≤n。
    """
    numeric_strength = float(strength)
    if not math.isfinite(numeric_strength) or not 0.0 <= numeric_strength <= 1.0:
        raise ValueError("denoise_median strength must be finite and in [0, 1]")
    y = interp_nan(as_1d(x))
    n = y.size
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w <= 1 or n <= 1:
        return y
    if _HAS_SCIPY:
        repaired = _ndi.median_filter(y, size=w, mode="reflect")
    else:
        repaired = sliding_median_symmetric(y, w)
    if numeric_strength == 1.0:
        return repaired
    return y + numeric_strength * (repaired - y)


def denoise_wavelet(x, wavelet: str = "db4", level=None, **_) -> np.ndarray:
    """小波去噪（移植 AegisTS `wavelet_denoise_repair` 的有界 VisuShrink 写法）。

    旧版病态根因：`mode="periodization"` + 无界 `wavedec` 全段重构 → 过收缩，nRMSE≈2×identity。
    修复：①有界 level=min(3, dwt_max_level)；②默认 symmetric mode；③保护近似分量（趋势 coeffs[0]）；
    ④VisuShrink 软阈值重构。作为 **denoise 动作**返回平滑重构（非仅修脏点——那是 outlier repair 语义）。
    pywt 缺失/异常 → 显式记录回退 savgol（不静默伪装成 wavelet）。
    """
    y = interp_nan(as_1d(x))
    n = y.size
    try:
        import pywt
        max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
        lv = min(3, max_level) if level is None else min(int(level), max_level)
        if lv < 1:
            record("denoise_wavelet", "denoise_savgol", "insufficient_length")
            return denoise_savgol(y)
        coeffs = pywt.wavedec(y, wavelet, level=lv, mode="symmetric")  # S0.7-8 显式固定（非旧版 periodization）
        detail = coeffs[-1]
        mad = max(float(np.median(np.abs(detail - np.median(detail)))), 1e-6)
        uthresh = (mad / 0.6745) * np.sqrt(2 * np.log(max(n, 2)))
        new = [coeffs[0]] + [pywt.threshold(c, uthresh, mode="soft") for c in coeffs[1:]]  # 保护趋势分量
        rec = np.asarray(pywt.waverec(new, wavelet, mode="symmetric"))[:n]  # VisuShrink 去噪重构（有界 level → 不过收缩）
        record("denoise_wavelet", "denoise_wavelet", "")
        return rec
    except Exception as e:
        record("denoise_wavelet", "denoise_savgol", f"exception:{type(e).__name__}")
        return denoise_savgol(y)


def _coerce_period(period) -> int:
    """period 归一化为 int：'auto'/None/非数/非正 → 0（= 触发自动猜测）；浮点取整。
    防止 'auto' 等字符串撞上 `period >= 2` 比较抛 TypeError 被 except 吞成 savgol 回退。"""
    if isinstance(period, bool) or not isinstance(period, (int, float)):
        return 0
    return int(period) if period > 0 else 0


def denoise_stl(x, period=0, **_) -> np.ndarray:
    """STL 去噪：返回 trend+seasonal（剔除残差）。**无显著周期 / statsmodels 不可用 / 异常 → 显式记录回退 savgol**。
    period 接受 int 或 'auto'/None（任意非正/非数 → 自动估计周期；无显著周期返回 0）。"""
    y = interp_nan(as_1d(x))
    p_req = _coerce_period(period)
    p = p_req if p_req >= 2 else _guess_period(y)
    if p < 2:                                                # 无显著季节 → STL 无意义，显式回退（不静默伪装 STL）
        record("denoise_stl", "denoise_savgol", "no_significant_seasonality")
        return denoise_savgol(y)
    if p >= y.size // 2:
        record("denoise_stl", "denoise_savgol", "period_too_large")
        return denoise_savgol(y)
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(y, period=p, robust=True).fit()
        record("denoise_stl", "denoise_stl", "")
        return np.asarray(res.trend + res.seasonal, dtype=float)
    except Exception as e:
        record("denoise_stl", "denoise_savgol", f"exception:{type(e).__name__}")
        return denoise_savgol(y)


# A0 第一步（2026-07-05）：S0.7 修复版 `_guess_period`（含 `_acf`）逐字迁入共享模块
# conditioning/period.py = robust_v1（bit 级不变；s1_decompose 的 `from .s1_denoise import
# _guess_period` 经此别名继续成立）。D1 的双估计器分叉自此显式、单一定义点。
from ..conditioning.period import guess_period_robust_v1 as _guess_period  # noqa: E402
