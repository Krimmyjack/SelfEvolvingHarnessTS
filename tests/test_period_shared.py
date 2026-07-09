"""tests/test_period_shared.py — A0 第一步守卫：共享周期模块 bit 等价 + 分叉显式化。

A0 前提（评审第二十三轮）：只有算子输出保持逐位一致，周期模块重构才能当成纯 Pattern
特征升级——否则旧响应矩阵/cached loss 全部作废。本 suite 用**内联逐字拷贝的旧实现**
（重构前 key.py `_dominant_period` / s1_denoise `_guess_period`）在语料上对照共享模块：
  ① legacy_fft_v0 与旧感知端逐位一致（period 与归一化功率谱）；
  ② robust_v1 与旧算子端逐位一致（整数周期，含无周期=0 情形）；
  ③ 接线身份：key._dominant_period / s1_denoise._guess_period 就是共享模块函数本体；
  ④ D1 分叉证据固化：trend+season 上 legacy 被趋势劫持、robust 找回真周期；
  ⑤ top_k_periods（P1 新增）：多周期恢复 + 谐波去重 + 不进任何算子路径。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.conditioning import period as shared
from SelfEvolvingHarnessTS.conditioning import key as key_mod
from SelfEvolvingHarnessTS.operators import s1_denoise


# ══ 旧实现内联拷贝（重构前逐字快照，勿改）══════════════════════════════════
def _old_dominant_period(x: np.ndarray) -> tuple:
    n = x.size
    if n < 8:
        return 1.0, np.ones(1)
    xd = x - x.mean()
    fft = np.fft.rfft(xd)
    power = (np.abs(fft) ** 2)[1:]
    if power.size == 0 or power.sum() <= 1e-12:
        return 1.0, np.array([1.0])
    freqs = np.fft.rfftfreq(n)[1:]
    k = int(np.argmax(power))
    f = freqs[k]
    period = float(1.0 / f) if f > 0 else 1.0
    period = min(period, float(n))
    return period, power / power.sum()


def _old_acf(resid: np.ndarray, lag: int) -> float:
    n = resid.size
    if lag <= 0 or lag >= n:
        return 0.0
    v = float(np.dot(resid, resid))
    return float(np.dot(resid[:-lag], resid[lag:]) / v) if v > 0 else 0.0


def _old_guess_period(y: np.ndarray, pmin: int = 4, pmax: int = 0,
                      min_peak_ratio: float = 3.0, acf_min: float = 0.2) -> int:
    n = y.size
    pmax = pmax if pmax >= pmin else n // 3
    if n < 2 * pmin or pmax < pmin:
        return 0
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    resid = resid - resid.mean()
    if not np.any(np.abs(resid) > 1e-12):
        return 0
    power = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n)
    band = (freqs >= 1.0 / pmax) & (freqs <= 1.0 / pmin)
    if not band.any():
        return 0
    k = int(np.argmax(np.where(band, power, 0.0)))
    med = float(np.median(power[band]))
    if med <= 0 or power[k] <= min_peak_ratio * med or freqs[k] <= 0:
        return 0
    period = int(round(1.0 / freqs[k]))
    if not (pmin <= period <= pmax):
        return 0
    if _old_acf(resid, period) < acf_min:
        return 0
    return period


def _corpus():
    """覆盖：季节/趋势/趋势+季节/白噪/常数/短序列/大幅离群，多长度。"""
    rng = np.random.default_rng(11)
    out = []
    for n in (64, 192, 480):
        t = np.arange(n, dtype=float)
        out += [
            np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(n),
            0.05 * t + 0.2 * rng.standard_normal(n),
            0.05 * t + np.sin(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(n),
            rng.standard_normal(n),
            np.full(n, 3.14),
        ]
    spiky = np.sin(2 * np.pi * np.arange(192) / 24)
    spiky[13] = 40.0
    out += [spiky, np.arange(6, dtype=float)]
    return out


# ── ①② bit 等价 ────────────────────────────────────────────────────────────
def test_legacy_fft_v0_bit_identical_to_old_perception():
    for x in _corpus():
        p_old, pw_old = _old_dominant_period(x)
        p_new, pw_new = shared.dominant_period_fft_v0(x)
        assert p_old == p_new                       # 浮点逐位相等（同一运算序列）
        assert np.array_equal(pw_old, pw_new)


def test_robust_v1_bit_identical_to_old_operator():
    for x in _corpus():
        assert _old_guess_period(x) == shared.guess_period_robust_v1(x)
    # 参数化路径同样一致
    y = np.sin(2 * np.pi * np.arange(480) / 24)
    assert (_old_guess_period(y, pmin=6, min_peak_ratio=5.0)
            == shared.guess_period_robust_v1(y, pmin=6, min_peak_ratio=5.0))


# ── ③ 接线身份（消费者用的就是共享本体，不是又一份拷贝）───────────────────
def test_consumers_wired_to_shared_module():
    assert key_mod._dominant_period is shared.dominant_period_fft_v0
    assert s1_denoise._guess_period is shared.guess_period_robust_v1
    from SelfEvolvingHarnessTS.operators.s1_decompose import _guess_period as gp_decomp
    assert gp_decomp is shared.guess_period_robust_v1
    assert set(shared.ESTIMATOR_IDS) == {"legacy_fft_v0", "robust_v1"}


def test_struct_feats_period_still_legacy():
    """P0 冻结：感知端 period 仍是 legacy（≠robust）——分叉显式保留，不许"顺手统一"。"""
    t = np.arange(480, dtype=float)
    y = 0.05 * t + np.sin(2 * np.pi * t / 24) + 0.05 * np.random.default_rng(0).standard_normal(480)
    feats = key_mod.struct_feats(y)
    assert feats["period"] == shared.dominant_period_fft_v0(y)[0]


# ── ④ D1 分叉证据固化 ──────────────────────────────────────────────────────
def test_d1_divergence_trend_hijacks_legacy_not_robust():
    """S_both 混叠机械成因的最小复现：趋势+季节 → legacy 谱峰被低频趋势劫持（period≈序列长），
    robust 去趋势后找回真周期 24。此测试是 D1 证据，也是 P1 特征升级的动机锚。"""
    n = 480
    t = np.arange(n, dtype=float)
    y = 0.05 * t + np.sin(2 * np.pi * t / 24)
    p_legacy = shared.dominant_period_fft_v0(y)[0]
    p_robust = shared.guess_period_robust_v1(y)
    assert p_legacy > n / 3, f"legacy 应被趋势劫持出大伪周期，得 {p_legacy}"
    assert 22 <= p_robust <= 26, f"robust 应找回 ~24，得 {p_robust}"


# ── ⑤ top-k 多周期（P1 专用新增）──────────────────────────────────────────
def test_top_k_periods_multiseason():
    n = 1008
    t = np.arange(n, dtype=float)
    y = np.sin(2 * np.pi * t / 24) + 0.6 * np.sin(2 * np.pi * t / 168) \
        + 0.05 * np.random.default_rng(5).standard_normal(n)
    ps = shared.top_k_periods(y, k=3)
    assert any(22 <= p <= 26 for p in ps), ps
    assert any(150 <= p <= 186 for p in ps), ps
    # 谐波去重：24 的整数倍（48/72/96…）不应重复出现
    base = [p for p in ps if 22 <= p <= 26]
    assert all(not (44 <= p <= 100) for p in ps if p not in base + [168]), ps


def test_top_k_no_period_on_noise_and_estimate_dispatch():
    rng = np.random.default_rng(9)
    assert shared.top_k_periods(rng.standard_normal(400), k=3) == []
    y = np.sin(2 * np.pi * np.arange(480) / 24)
    assert shared.estimate_period(y, "robust_v1") == float(shared.guess_period_robust_v1(y))
    assert shared.estimate_period(y, "legacy_fft_v0") == shared.dominant_period_fft_v0(y)[0]
    with pytest.raises(KeyError):
        shared.estimate_period(y, "nonexistent")
