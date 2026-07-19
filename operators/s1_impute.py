"""operators/s1_impute.py — S1 缺失插补。

S0.7 Operator Integrity：impute_* 契约 = **只更新缺失位置、保留已观测值**（imputer 语义）。
旧 impute_fft 违反此契约（对全序列低通重构 → 改动已观测点，实为 linear-impute + FFT 去噪）已修复。

════════════════════ 两族插补机制（E-3.3 R2，2026-07-14） ════════════════════

**复制/插值族**（impute_linear / impute_fft / impute_ema / period_complete）：填进去的值是
邻近观测的某种加权组合——线性内插、低通重建、指数前向、同相位前一周期。它们不对序列的
生成过程建模，只对"洞附近长什么样"做插值。整个 v0.1/v0.2 池里的插补全是这一族。

**模型预测族**（impute_ssm / impute_ar，本次新增）：先对序列的**生成过程**拟合一个模型
（状态空间 / 自回归），再用模型对缺失位置做预测。机制上与插值族可区分——这才是它们值得
各占一个动作名额的理由（`benchmark/programs.py` 的冻结排除清单正是按"机制冗余"排除算子的：
impute_ema 被判为与 forward_fill 冗余、impute_fft 与 seasonal_fill 冗余）。

⚠️ **命名红线**：新算子**不得**叫 `impute_kalman`。那个名字在本仓库的历史 trace 里已经指过
`impute_ema`（一个误名，见 registry.ALIASES），复用它会污染 provenance——同一个名字在旧记录
里指 EMA、在新记录里指状态空间，任何跨版本的算子身份审计都会被这个名字骗过去。旧 alias
保留并标记 deprecated，旧 artifact 重放不破坏。
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from ._common import as_1d, interp_nan, moving_average
from ._provenance import record
from ..conditioning.period import guess_period_robust_v1 as _guess_period


def impute_linear(x, strength: float = 1.0, **_) -> np.ndarray:
    """Linear interpolation with an optional, canonical monotonic dose.

    ``strength=1`` is the historical operator.  Lower strengths still fill every
    missing point, but interpolate between the finite-series median (the neutral
    anchor) and the full linear repair.  Making this dose part of the operator is
    what lets a fixed ProbeAPI arm and an Agent PROGRAM execute byte-identical
    transformations; the probe must not perform a hidden blend after execution.
    """

    numeric_strength = float(strength)
    if not math.isfinite(numeric_strength) or not 0.0 <= numeric_strength <= 1.0:
        raise ValueError("impute_linear strength must be finite and in [0, 1]")
    raw = as_1d(x).astype(float)
    missing = np.isnan(raw)
    repaired = interp_nan(raw)
    if numeric_strength == 1.0 or not missing.any():
        return repaired
    finite = raw[np.isfinite(raw)]
    anchor = float(np.median(finite)) if finite.size else 0.0
    output = raw.copy()
    output[missing] = anchor + numeric_strength * (repaired[missing] - anchor)
    return output


def impute_fft(x, cutoff_ratio: float = 0.1, **_) -> np.ndarray:
    """谱插补：**仅用低通重建填补缺失位置，保留所有已观测值**（S0.7 修复观测保持契约）。

    旧版对整段做低通重构 → 改动非缺失观测（实为 linear-impute + FFT 低通去噪，非 imputation）。
    修复：无缺失 → 恒等；否则 linear 先验 + 低通重建，仅写回缺失掩码位置。全缺失/过短 → 显式回退 linear。
    """
    y = as_1d(x).astype(float)
    m = np.isnan(y)
    if not m.any():
        record("impute_fft", "impute_fft", "no_missing_identity")
        return y                                              # 无缺失 → 完全不变
    n = y.size
    if n < 8 or m.all():
        record("impute_fft", "impute_linear", "too_short_or_all_missing")
        return interp_nan(y)
    base = interp_nan(y)                                      # 线性先验（供谱重建）
    f = np.fft.rfft(base - base.mean())
    keep = max(1, int(len(f) * cutoff_ratio))
    f[keep:] = 0.0
    recon = np.fft.irfft(f, n=n) + base.mean()
    out = y.copy()
    out[m] = recon[m]                                         # 只写回缺失位置，保留已观测
    record("impute_fft", "impute_fft", "")
    return out


def impute_ema(x, alpha: float = 0.3, **_) -> np.ndarray:
    """指数平滑（EMA）前向填补。S0.7-6 正名：旧名 `impute_kalman` 系误名（实现是 EMA 非 Kalman），
    保留为兼容 alias（registry.ALIASES）——旧 artifact/模板按旧名重放不破坏。"""
    y = as_1d(x).copy()
    m = np.isnan(y)
    if not m.any():
        return y
    base = interp_nan(y)
    s = base[0]
    for i in range(y.size):
        s = alpha * base[i] + (1 - alpha) * s
        if m[i]:
            y[i] = s
    return y


impute_kalman = impute_ema             # S0.7-6 兼容引用（勿新增使用；registry 层统一走 ALIASES）


def period_complete(x, period: int = 0, **_) -> np.ndarray:
    """用同相位前一周期值填补缺失；无周期信息则退化为线性插补。"""
    y = as_1d(x).copy()
    m = np.isnan(y)
    if not m.any() or period < 2:
        return interp_nan(y)
    for i in np.where(m)[0]:
        j = i - period
        if j >= 0 and not np.isnan(y[j]):
            y[i] = y[j]
    return interp_nan(y)


# ════════════════════════ 模型预测族（E-3.3 R2，2026-07-14） ════════════════════════

_SSM_MIN_OBSERVED = 8            # 观测点少于这个数，MLE 无从谈起
_SSM_MAX_ITER = 50               # 冻结：lltrend+seasonal 在 50 步内不收敛且慢一倍，故取 local level
_AR_MIN_ROWS_PER_PARAM = 3       # 设计矩阵行数须 ≥ 3×参数数，否则 AR 系数是在拟合噪声
_AR_DEFAULT_ORDER = 8            # 无季节时的阶数地板（有季节时抬到 period，见 impute_ar）


def impute_ssm(x, period: int = 0, **_) -> np.ndarray:
    """状态空间（结构时序模型）平滑插补 —— **模型预测族**，非插值族。

    模型（冻结）：`UnobservedComponents(level="local level", seasonal=period,
    concentrate_scale=True)`，MLE（lbfgs，maxiter=50）→ Kalman **平滑**（非滤波）→ 取
    `smoother_results.smoothed_forecasts`（= E[y_t | 全部数据]，双向信息）→ **只写回缺失位置**。

    选型依据（实测，400 点合成序列，噪声 σ=0.3，period=24）：
        local level + seasonal   7.5s  收敛    插补 MAE 0.213
        local linear trend + seasonal 15.7s  50 步不收敛  MAE 0.198
        local level（无季节）    0.1s  收敛    MAE 0.968 ＝ 线性插补，**一点没赚**
    → 价值全在季节分量上；local linear trend 多花一倍时间换 7% MAE、还不收敛 → 不取。
    ⚠️ **成本**：~7.5s/条序列，比 impute_linear 慢约 4 个数量级。这个数字是它能否进
    benchmark 冻结池的决定性输入（见 operators/CHANGELOG.md 的成本披露），务必先算总账。

    ⚠️ **不得整段返回 smoothed_forecasts**：平滑信号在**已观测点上也不等于观测值**（它顺带
    去了噪）。整段返回 = 一个伪装成 imputer 的 denoiser——这正是 impute_fft 犯过、S0.7 修过
    的同一个错。本函数 `out = y.copy(); out[missing] = smoothed[missing]`，preserves_observed
    由构造保证，测试守。

    依赖：statsmodels **硬依赖**。缺失 → `ImportError`（fail-loud）。**绝不静默降级到 EMA**——
    那会让台账里记着"跑了状态空间"、实际跑的是指数平滑（`impute_kalman` 误名事故的原型）。
    """
    y = as_1d(x).astype(float)
    m = np.isnan(y)
    if not m.any():
        record("impute_ssm", "impute_ssm", "no_missing_identity")
        return y

    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents
    except ImportError as exc:                        # fail-loud：不许回退到任何别的算子
        raise ImportError(
            "impute_ssm 需要 statsmodels（契约 requires_dependency='statsmodels'，"
            "fallback_policy='none'）。缺失时**硬失败**，不静默降级——降级会伪造算子身份。"
        ) from exc

    n = y.size
    observed = int((~m).sum())
    if observed < _SSM_MIN_OBSERVED or n < 2 * _SSM_MIN_OBSERVED:
        record("impute_ssm", "impute_linear", "too_few_observations")   # 退化输入：显式记账
        return interp_nan(y)

    p = int(period) if period and period >= 2 else _guess_period(interp_nan(y))
    seasonal = p if (p >= 2 and n >= 3 * p) else None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")           # 收敛警告由 provenance 记账，不靠 stderr
            model = UnobservedComponents(
                y, level="local level", seasonal=seasonal, concentrate_scale=True)
            res = model.fit(disp=False, maxiter=_SSM_MAX_ITER)
        smoothed = np.asarray(res.smoother_results.smoothed_forecasts, dtype=float).ravel()
    except Exception as e:
        record("impute_ssm", "impute_linear", f"exception:{type(e).__name__}")
        return interp_nan(y)

    if smoothed.size != n or not np.isfinite(smoothed[m]).all():
        record("impute_ssm", "impute_linear", "smoother_produced_non_finite")
        return interp_nan(y)

    converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))
    record("impute_ssm", "impute_ssm", "" if converged else "mle_not_converged")

    out = y.copy()
    out[m] = smoothed[m]                              # **只写缺失位置**（imputer 契约）
    return out


def _ar_fit(y: np.ndarray, order: int) -> np.ndarray | None:
    """在**全观测**的滞后窗口上最小二乘拟合 AR(order)+截距。行数不足 → None。

    只用 target 与全部 order 个滞后都被观测到的行——**不拿插补值当训练数据**，否则 AR 系数
    是在拟合 interp_nan 画出来的直线，模型预测族就退化成了插值族的一个包装。"""
    n = y.size
    if n <= order + 1:
        return None
    rows = np.lib.stride_tricks.sliding_window_view(y, order + 1)     # (n-order, order+1)
    good = np.isfinite(rows).all(axis=1)
    if int(good.sum()) < _AR_MIN_ROWS_PER_PARAM * (order + 1):
        return None
    block = rows[good]
    lags = block[:, :-1][:, ::-1]                     # 列 j = 滞后 j+1（最近的滞后在前）
    target = block[:, -1]
    design = np.column_stack([np.ones(lags.shape[0]), lags])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef if np.isfinite(coef).all() else None


def _ar_forward(y: np.ndarray, coef: np.ndarray, order: int) -> np.ndarray:
    """从左向右递推预测缺失位置（用得到的观测/已预测值）。上下文不足的位置留 NaN。"""
    out = y.copy()
    for i in np.flatnonzero(np.isnan(y)):
        if i < order:
            continue                                  # 左侧上下文不够 → 交给反向通道
        ctx = out[i - order:i][::-1]                  # 最近的滞后在前，与 _ar_fit 的列序一致
        if not np.isfinite(ctx).all():
            continue
        out[i] = float(coef[0] + coef[1:] @ ctx)
    return out


def impute_ar(x, order: int = 0, period: int = 0, **_) -> np.ndarray:
    """双向 AR 插补 —— **模型预测族**，纯 numpy（无第三方依赖）。

    机制：在全观测的滞后窗口上最小二乘拟合 AR(order)（正向一个、把序列翻转再拟合一个反向的），
    正向递推与反向递推各填一遍，在每个缺口内**按到两端的距离线性加权融合**（离左端近就更信
    正向预测，离右端近就更信反向）。只有一侧可用（序列开头/结尾的缺口）→ 用那一侧。两侧都
    够不着（上下文全缺）→ 显式记账回退线性插补。

    **阶数必须够到季节滞后**（`order=0` = 自动：max(8, period)，上限 n//4）。这不是调参偏好，
    是机制的死活：AR(p) 只能看见 p 步以内的历史，p < period 时它**根本看不见季节周期**，于是
    退化成一个短记忆模型。实测（period=24、30 点 block 缺口的季节序列，插补 MAE）：

        linear=2.00   AR(8)=1.33   AR(16)=0.40   AR(24)=0.34   AR(48)=0.34

    AR(8) 只把线性插补的误差砍掉三分之一，AR(24) 砍掉 83%。period 由调用方传入（benchmark 池
    可用 `period_param` 注入冻结频率）；未传则走 A0 共享周期模块估计——**不另起周期估计器**。

    与 impute_ema 的区分：EMA 是**单向、无参数、指数加权的复制**；这里是**双向、拟合出来的
    自回归模型的预测**——AR 系数由数据决定，能表达周期性/均值回复，EMA 不能。

    契约：preserves_observed=True（只写缺失位置）、requires_dependency=None、
    fallback_policy 仅在退化输入下触发且**逐次记账**（`_provenance`），无静默回退。
    """
    y = as_1d(x).astype(float)
    m = np.isnan(y)
    if not m.any():
        record("impute_ar", "impute_ar", "no_missing_identity")
        return y
    if m.all():
        record("impute_ar", "impute_linear", "all_missing")
        return interp_nan(y)

    n = y.size
    if order and int(order) >= 1:
        want = int(order)
    else:                                             # 自动：阶数必须够到季节滞后（见 docstring）
        p_seas = int(period) if period and period >= 2 else _guess_period(interp_nan(y))
        want = max(_AR_DEFAULT_ORDER, p_seas) if p_seas >= 2 else _AR_DEFAULT_ORDER
    p = max(1, min(want, (n - 1) // 4))               # 阶数不得超过序列长度的 1/4（否则拟合噪声）
    fwd_coef = _ar_fit(y, p)
    bwd_coef = _ar_fit(y[::-1], p)
    if fwd_coef is None and bwd_coef is None:
        record("impute_ar", "impute_linear", "insufficient_complete_lag_windows")
        return interp_nan(y)

    fwd = _ar_forward(y, fwd_coef, p) if fwd_coef is not None else np.full(n, np.nan)
    bwd = (_ar_forward(y[::-1], bwd_coef, p)[::-1] if bwd_coef is not None
           else np.full(n, np.nan))

    out = y.copy()
    fallback = interp_nan(y)
    unreachable = 0
    idx = np.flatnonzero(m)
    for gap in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):    # 逐个极大缺口
        if not gap.size:
            continue
        length = gap.size
        for k, i in enumerate(gap):
            f, b = fwd[i], bwd[i]
            f_ok, b_ok = np.isfinite(f), np.isfinite(b)
            if f_ok and b_ok:
                w = (k + 1.0) / (length + 1.0)        # 缺口内位置 → 反向权重
                out[i] = (1.0 - w) * f + w * b
            elif f_ok:
                out[i] = f
            elif b_ok:
                out[i] = b
            else:
                out[i] = fallback[i]
                unreachable += 1

    if not np.isfinite(out).all():                    # 递推发散（|φ|>1 可以炸）→ 整条退回线性
        record("impute_ar", "impute_linear", "recursion_diverged")
        return fallback
    if unreachable:
        record("impute_ar", "impute_ar", f"linear_fallback_on_{unreachable}_unreachable_points")
    else:
        record("impute_ar", "impute_ar", "")
    return out
