"""tests/test_p1a.py — P1a 提取器实验使能守卫（Stage 2.1 第一臂）。

这些不是边缘守卫（moratorium 内合法）：提取器若错，第一张表就是垃圾。守：
  ① diag 与 robust_v1 判决一致（单一定义点无漂移）；
  ② D1 修复兑现：trend+season 上 P0 period 被劫持、P1a 找回 ~24；
  ③ D2 修复兑现：缺失不压缩时间轴——30% 缺失下 period 仍恢复，且特征对缺失布局稳定
    （P0 压缩轴会把周期错频）；
  ④ 缺失拓扑特征正确（构造已知 gap 布局）；
  ⑤ C 通道语义：强季节高 c_peak_sig、白噪低；coverage=最长观测段/n；
  ⑥ PatternSpec P1a：向量顺序=契约、bit 级一致、P0 身份不受影响。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.conditioning import period as shared
from SelfEvolvingHarnessTS.conditioning.key import struct_feats
from SelfEvolvingHarnessTS.conditioning.p1a import (P1A_ALL_FEATS, P1A_C_FEATS, P1A_D_FEATS,
                                                    P1A_P_FEATS, p1a_feats, p1a_vectors)
from SelfEvolvingHarnessTS.policy import pattern_spec_p0, pattern_spec_p1a


def _mk(n=480, noise=0.1, trend=0.05, period=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return trend * t + np.sin(2 * np.pi * t / period) + noise * rng.standard_normal(n)


# ── ① 判决一致（防 diag/v1 漂移）────────────────────────────────────────────
def test_diag_period_matches_robust_v1():
    rng = np.random.default_rng(3)
    corpus = [_mk(seed=s) for s in range(4)] + [rng.standard_normal(300), np.full(64, 2.0),
                                                np.arange(50, dtype=float)]
    for y in corpus:
        assert shared.robust_period_diag(y)["period"] == shared.guess_period_robust_v1(y)
    y = _mk(seed=9)
    assert (shared.robust_period_diag(y, pmin=6, min_peak_ratio=5.0)["period"]
            == shared.guess_period_robust_v1(y, pmin=6, min_peak_ratio=5.0))


# ── ② D1 修复兑现 ───────────────────────────────────────────────────────────
def test_d1_fixed_trend_plus_season():
    y = _mk(noise=0.05)
    p0 = struct_feats(y)
    p1 = p1a_feats(y)
    assert p0["period"] > 160, "P0 应被趋势劫持（D1 证据前提）"
    assert 22 <= p1["period"] <= 26, f"P1a 应找回 ~24，得 {p1['period']}"
    assert p1["seasonal_strength"] > 0.3, "P1a 在真周期处应测得显著季节强度"
    assert p0["seasonal_strength"] < p1["seasonal_strength"]


# ── ③ D2 修复兑现：缺失不压缩时间轴 ──────────────────────────────────────────
def test_d2_period_survives_missing():
    y = _mk(noise=0.05, seed=1)
    rng = np.random.default_rng(7)
    ym = y.copy()
    ym[rng.choice(y.size, int(0.3 * y.size), replace=False)] = np.nan
    p1 = p1a_feats(ym)
    assert 22 <= p1["period"] <= 26, f"30% 缺失下 P1a period 应仍 ~24，得 {p1['period']}"
    assert p1["missing_rate"] == pytest.approx(0.3, abs=0.01)


def test_d2_feature_stability_across_missing_layouts():
    """同一 clean 信号、两种缺失布局 → P1a 核心结构特征应接近（P0 压缩轴则漂移）。"""
    y = _mk(noise=0.05, seed=2)
    out = []
    for s in (11, 12):
        rng = np.random.default_rng(s)
        ym = y.copy()
        ym[rng.choice(y.size, int(0.25 * y.size), replace=False)] = np.nan
        out.append(p1a_feats(ym))
    a, b = out
    assert a["period"] == b["period"]
    assert abs(a["trend_strength"] - b["trend_strength"]) < 0.05
    assert abs(a["seasonal_strength"] - b["seasonal_strength"]) < 0.1


# ── ④ 缺失拓扑 ──────────────────────────────────────────────────────────────
def test_gap_topology_features():
    y = _mk(n=200, trend=0.0, seed=4)
    ym = y.copy()
    ym[50:70] = np.nan                                   # 20 块状
    ym[100] = np.nan                                     # 1 孤立
    ym[150:155] = np.nan                                 # 5 块状
    f = p1a_feats(ym)
    assert f["missing_rate"] == pytest.approx(26 / 200)
    assert f["max_gap_frac"] == pytest.approx(20 / 200)
    assert f["gap_run_mean_frac"] == pytest.approx((26 / 3) / 200)
    assert f["c_obs_coverage"] == pytest.approx(50 / 200)   # 最长观测段 = 0..49（50 点）
    # 同缺失率、不同拓扑（全孤立点）→ 拓扑特征必须区分
    ym2 = y.copy()
    rng = np.random.default_rng(5)
    ym2[rng.choice(np.arange(0, 200, 2), 26, replace=False)] = np.nan
    f2 = p1a_feats(ym2)
    assert f2["missing_rate"] == pytest.approx(26 / 200)
    assert f2["max_gap_frac"] < f["max_gap_frac"]


# ── ⑤ C 通道语义 ────────────────────────────────────────────────────────────
def test_confidence_channel_semantics():
    strong = p1a_feats(_mk(noise=0.05, trend=0.0, seed=6))
    noise = p1a_feats(np.random.default_rng(8).standard_normal(480))
    assert strong["c_peak_sig"] > 0.7 > noise["c_peak_sig"]
    assert strong["c_acf_confirm"] > 0.5
    assert strong["c_obs_coverage"] == 1.0
    assert noise["period"] == 0.0 and noise["seasonal_strength"] == 0.0


# ── ⑥ PatternSpec P1a 契约 ─────────────────────────────────────────────────
def test_pattern_spec_p1a_contract():
    s = pattern_spec_p1a()
    assert s.version == "P1a"
    assert s.feature_names == tuple(P1A_ALL_FEATS)
    assert s.d_feats == tuple(P1A_D_FEATS) and s.p_feats == tuple(P1A_P_FEATS)
    assert s.confidence_schema["features"] == list(P1A_C_FEATS)
    assert "robust_v1" in s.period_estimator_id
    y = _mk(seed=10)
    v1, v2 = s.features_vector(y), s.features_vector(y)
    assert np.array_equal(v1, v2) and v1.shape == (16,) and np.all(np.isfinite(v1))
    vec = p1a_vectors(y)
    assert np.array_equal(np.concatenate([vec["d"], vec["p"], vec["c"]]), v1)
    # P0 身份不受 P1a 加入影响（禁止原地改 P0 的硬钉）
    assert pattern_spec_p0().config_sha() == "e4f10d11128e943a"
    assert s.config_sha() != pattern_spec_p0().config_sha()
