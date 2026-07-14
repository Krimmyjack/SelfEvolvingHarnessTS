"""operators/s1_outlier.py — S1 离群处理（裁剪/收缩/点式替换，保持长度）。

两族机制，别混为一谈：
  **全局阈值裁剪**（winsorize / outlier_iqr / outlier_mad）：用**整条序列**的分位数或 MAD
  定一个全局上下界，把所有越界点钳回界上。对平稳序列有效；对**有趋势或强季节**的序列，
  界是全序列口径的，会把趋势的两端和季节峰谷整片削掉——它分不清"离群"和"本来就该在那儿的高点"。

  **局部自适应点式替换**（hampel_filter，E-3.3 R3，2026-07-14 新增）：阈值由**滚动窗口内**
  的中值与 MAD 决定，只有偏离局部中值超过 n_sigmas 个局部 σ 的点被替换成局部中值，其余点
  **逐位不动**。趋势/季节被滚动中值跟住，故不会被误杀。

这个区分不是修辞：`benchmark/programs.py` 的冻结排除清单把 outlier_iqr/outlier_mad 判为
"与 winsorize 机制冗余"而排除在池外——那是对的，它们仨确实是同一族。hampel 是**另一族**，
所以它值得占一个新的动作名额。
"""
from __future__ import annotations

import numpy as np

from ._common import MAD_TO_SIGMA, as_1d, interp_nan, odd_window, sliding_mad_symmetric
from ._provenance import record


def winsorize(x, limits: float = 0.05, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    lo, hi = np.quantile(y, [limits, 1.0 - limits])
    return np.clip(y, lo, hi)


def outlier_iqr(x, k: float = 1.5, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = q3 - q1
    return np.clip(y, q1 - k * iqr, q3 + k * iqr)


def outlier_mad(x, k: float = 3.5, **_) -> np.ndarray:
    y = interp_nan(as_1d(x))
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad <= 1e-12:
        return y
    scale = 1.4826 * mad
    return np.clip(y, med - k * scale, med + k * scale)


def hampel_filter(x, window: int = 7, n_sigmas: float = 3.0, **_) -> np.ndarray:
    """Hampel 滤波：滚动 MAD 判据 + **仅替换命中点**（点式，非整段平滑、非全局裁剪）。

    判据：|y[i] − med[i]| > n_sigmas × 1.4826 × mad[i]，其中 med/mad 是以 i 为中心、
    宽 `window` 的滚动中值与滚动 MAD（symmetric 镜像边界，与库内其余中值类算子同语义，
    落 `_common.BOUNDARY_MODES`）。命中点替换为 **med[i]**（局部中值），未命中点逐位不动。

    **默认参数 = 文献默认**（Hampel 经典形式：半宽 k=3 → window=2k+1=7，阈值 3σ；亦即
    MATLAB `hampel(x)` 的默认）。**刻意不做择优扫参**：本算子是否胜过 winsorize 是 benchmark
    要回答的经验问题，先把它调到能赢再拿去量，就是把尺子当靶子。参数敏感性已扫过并披露在
    operators/CHANGELOG.md；任何**冻结 preset**（若它将来进 benchmark 池）必须在
    Support-A discovery 上另行标定，不得沿用这里的合成数默认。

    合成校验（noise σ=0.5、period=24、benchmark spike 腐蚀 dose .03）：clean 序列上的附带
    损伤 0.068 < winsorize 的 0.088（winsorize 已在冻结池里），spike 通道 0.223 < winsorize
    的 0.301——两个轴上都不比一个现任池成员差。窗口越宽损伤越大（w=11 时 0.116）：**窗口相对
    季节曲率太宽时，波峰处局部 MAD 变小而偏差变大 → 系统性误杀季节峰**。这是 Hampel 的已知
    弱点，不是实现 bug；窗口保持在文献默认的窄值是对它的直接防御。

    **零 MAD 守卫**：局部窗口恒定时 mad=0 → 任何非零偏差都会被判为无穷多个 σ → 整片误杀
    （阶梯状/量化序列上尤其致命）。故 mad ≤ 0 的位置**一律不判**（弃权，不替换）。

    契约：destructive=True（改动已观测点取值）、preserves_observed=False、
    allowed_tasks 排除 anomaly——它删的正是 anomaly 要检的 spike。
    与 winsorize/outlier_iqr/outlier_mad 的机制区分见模块 docstring。
    """
    y = interp_nan(as_1d(x))
    n = y.size
    w = odd_window(window, n)
    if w < 3:
        record("hampel_filter", "hampel_filter", "window_too_small_identity")
        return y

    med, mad = sliding_mad_symmetric(y, w)
    sigma = MAD_TO_SIGMA * mad
    hit = (sigma > 0.0) & (np.abs(y - med) > float(n_sigmas) * sigma)
    if not hit.any():
        record("hampel_filter", "hampel_filter", "no_outlier_identity")
        return y

    out = y.copy()
    out[hit] = med[hit]
    record("hampel_filter", "hampel_filter", f"replaced_{int(hit.sum())}_of_{n}")
    return out
