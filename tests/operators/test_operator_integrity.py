"""tests/test_operator_integrity.py — S0.7 Operator Integrity Gate 语义测试。

不只测"不崩溃"，还测算子**身份真实**（评审要求）：长度/有限/确定性、imputer 保持已观测、
周期估计恢复已知周期且纯趋势/白噪返回 0、fallback 可见、无解释的两动作全量相同报警。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.operators import s1_denoise as den
from SelfEvolvingHarnessTS.operators import s1_impute as imp
from SelfEvolvingHarnessTS.operators import s1_structural as structural
from SelfEvolvingHarnessTS.operators import _provenance as prov

RNG = np.random.default_rng(0)
N = 464
T = np.arange(N, dtype=float)


def _season(noise=0.1, seed=1):
    return np.sin(2 * np.pi * T / 24) + np.random.default_rng(seed).normal(0, noise, N)


def _trend(noise=0.1, seed=2):
    return 0.02 * T + np.random.default_rng(seed).normal(0, noise, N)


IMPUTERS = [imp.impute_linear, imp.impute_fft, imp.impute_kalman, imp.period_complete]
DENOISERS = [den.denoise_savgol, den.denoise_median, den.denoise_stl, den.denoise_wavelet]


# ── 基本契约：长度 / 有限 / 确定性 ─────────────────────────────────────────
@pytest.mark.parametrize("fn", IMPUTERS + DENOISERS)
def test_length_finite_deterministic(fn):
    x = _season()
    x[[10, 50, 100]] = np.nan
    a = fn(x)
    b = fn(x)
    assert a.shape == (N,), f"{fn.__name__} 改变长度"
    assert np.all(np.isfinite(a)), f"{fn.__name__} 输出非有限"
    assert np.allclose(a, b), f"{fn.__name__} 非确定性"


# ── imputer 契约：只改缺失位置、保留已观测 ────────────────────────────────
@pytest.mark.parametrize("fn", IMPUTERS)
def test_imputer_preserves_observed(fn):
    x = _season()
    miss = [10, 50, 100, 200]
    x[miss] = np.nan
    obs = ~np.isnan(x)
    out = fn(x)
    assert np.allclose(out[obs], x[obs]), f"{fn.__name__} 改动了已观测值（违反 imputer 契约）"
    assert np.all(np.isfinite(out[~obs])), f"{fn.__name__} 未填补缺失"


def test_impute_fft_no_missing_is_identity():
    x = _season()
    assert np.allclose(imp.impute_fft(x), x), "impute_fft 无缺失时应恒等（不得低通改动已观测）"


# ── 周期估计：恢复已知周期，纯趋势/白噪返回 0 ─────────────────────────────
def test_guess_period_recovers_seasonal():
    assert den._guess_period(_season(noise=0.1)) == 24


def test_guess_period_zero_on_pure_trend():
    assert den._guess_period(_trend(noise=0.1)) == 0, "纯趋势不应估出伪周期（旧 bug）"


def test_guess_period_zero_on_white_noise():
    assert den._guess_period(RNG.normal(0, 1, N)) == 0


# ── fallback 可见：STL 在无季节时显式记录回退，不静默伪装 STL ──────────────
def test_stl_records_fallback_on_trend():
    prov.start_recording()
    den.denoise_stl(_trend())      # 纯趋势 → 应回退 savgol 并记录原因
    den.denoise_stl(_season())     # 有季节 → 真 STL
    prov.stop_recording()
    led = prov.get_ledger()
    trend_rec = led[0]
    assert trend_rec["effective"] == "denoise_savgol" and "seasonal" in trend_rec["reason"], \
        f"STL 未显式记录无季节回退: {trend_rec}"
    assert led[1]["effective"] == "denoise_stl" and led[1]["reason"] == "", "有季节时应真跑 STL"


# ── wavelet 非病态且非静默重复 savgol ────────────────────────────────────
def test_wavelet_is_genuine_denoiser():
    pytest.importorskip("pywt")
    both = np.sin(2 * np.pi * T / 24) + 0.012 * T
    both = both / np.std(both)
    noisy = both + RNG.normal(0, 0.5, N)

    def nrmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    w = den.denoise_wavelet(noisy)
    s = den.denoise_savgol(noisy)
    assert not np.allclose(w, s), "wavelet 与 savgol 全量相同（静默回退/重复列）"
    assert nrmse(w, both) < nrmse(noisy, both), "wavelet 应改善（旧版病态 ≈2×identity）"


# ── 无解释的动作全量相同报警：4 个 denoise 两两不得完全相同 ─────────────────
def test_denoisers_are_semantically_distinct():
    x = _season(noise=0.4, seed=7)
    outs = {f.__name__: f(x) for f in DENOISERS}
    names = list(outs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not np.allclose(outs[names[i]], outs[names[j]]), \
                f"{names[i]} 与 {names[j]} 全量相同 → 动作身份不真实（静默回退嫌疑）"


def test_dependency_fingerprint_records_versions():
    fp = prov.dependency_fingerprint()
    assert "numpy" in fp and "pywt" in fp


def test_level_repair_accepts_complete_deployment_observable_binding():
    clean = np.sin(2 * np.pi * np.arange(120, dtype=float) / 24.0)
    corrupt = clean.copy()
    corrupt[36:84] += 2.0
    repaired = structural.repair_level_shift(
        corrupt,
        region_start_fraction=36 / 120,
        region_end_fraction=84 / 120,
        estimated_offset=2.0,
    )
    np.testing.assert_allclose(repaired, clean, atol=1e-12)
    with pytest.raises(ValueError):
        structural.repair_level_shift(
            corrupt,
            region_start_fraction=36 / 120,
        )
