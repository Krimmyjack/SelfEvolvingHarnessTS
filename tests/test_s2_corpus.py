"""tests/test_s2_corpus.py — S2 语料生成器实验使能守卫（语料错=下游全部作废）。

守：①确定性（同 uid → bit 级同序列）；②冻结族逐字复用 v1；③新族结构成立
（intermittent 零膨胀 / hetero 波动聚簇 / regime 分段 / multiseason 双周期可恢复）；
④miss-topology 语义正确（block=单连续段、burst=多短簇、rate 准确）；
⑤split 确定性 70/30、dev∩holdout=∅、holdout 不物化。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.run_variance_decomp import _clean_signal
from SelfEvolvingHarnessTS.s2_corpus import (DEV_J, S2_DEG_GRID, S2_FAMILIES, _miss_indices,
                                             build_s2_dev, make_series, s2_clean, s2_split,
                                             sd_of, uid_of)


def test_determinism_bit_identical():
    a = make_series("S_hetero", "n_hi_block_lo", 3)
    b = make_series("S_hetero", "n_hi_block_lo", 3)
    assert np.array_equal(a.clean, b.clean)
    assert np.array_equal(a.degraded, b.degraded, equal_nan=True)
    assert a.series_uid == "S2:S_hetero:n_hi_block_lo:3"


def test_frozen_families_verbatim_v1():
    for fam in ("S_season", "S_trend", "S_both", "S_ar"):
        sd = sd_of(uid_of(fam, "n_hi_full", 0))
        assert np.array_equal(s2_clean(fam, sd), _clean_signal(fam, sd))


def test_new_family_structures():
    from SelfEvolvingHarnessTS.conditioning.period import top_k_periods
    # intermittent：零膨胀（多数点为 0）
    x = s2_clean("S_intermittent", 100)
    assert np.mean(x == 0.0) > 0.5
    # hetero：波动聚簇 → |x| 的 lag-1 自相关显著为正
    y = s2_clean("S_hetero", 101)
    a = np.abs(y) - np.abs(y).mean()
    r = float(a[:-1] @ a[1:] / (a @ a))
    assert r > 0.1, f"波动聚簇缺失 r={r}"
    # regime：分段方差 —— 全局方差显著大于差分方差（水平跳变主导）
    z = s2_clean("S_regime", 102)
    assert np.var(z) > 3 * np.var(np.diff(z))
    # multiseason：16 与 128 双周期可恢复（周期对三重约束见 s2_corpus 模块 doc）
    m = s2_clean("S_multiseason", 103)
    ps = top_k_periods(m, k=3)
    assert any(15 <= p <= 17 for p in ps), ps
    assert any(115 <= p <= 141 for p in ps), ps


def test_miss_topology_semantics():
    rng = np.random.default_rng(0)
    n = 464
    blk = _miss_indices(n, 56, "block", rng)
    assert blk.size == 56 and np.all(np.diff(blk) == 1)          # 单连续段
    rng = np.random.default_rng(1)
    bst = np.sort(_miss_indices(n, 56, "burst", rng))
    runs = np.split(bst, np.where(np.diff(bst) > 1)[0] + 1)
    assert len(runs) >= 3 and all(len(r) >= 2 for r in runs)     # 多短簇
    rnd = _miss_indices(n, 56, "random", np.random.default_rng(2))
    assert rnd.size == 56 and len(set(rnd.tolist())) == 56
    # rate 落地：series 级缺失率与网格一致
    rs = make_series("S_season", "n_lo_rand_hi", 1)
    assert np.isnan(rs.degraded).mean() == pytest.approx(0.12, abs=0.005)
    rs0 = make_series("S_season", "n_lo_full", 1)
    assert not np.isnan(rs0.degraded).any()


def test_split_deterministic_70_30_and_no_holdout_materialized():
    dev1, hold1 = s2_split()
    dev2, hold2 = s2_split()
    assert dev1 == dev2 and hold1 == hold2                       # 确定性
    n_strata = len(S2_FAMILIES) * len(S2_DEG_GRID)
    assert len(dev1) == n_strata * 7 and len(hold1) == n_strata * 3
    dev_uids = {uid_of(f, d, j) for f, d, j in dev1}
    assert not (dev_uids & set(hold1))                           # 不交
    corpus = build_s2_dev()
    assert {rs.series_uid for rs in corpus} == dev_uids          # holdout 一条都没物化
    assert len(corpus) == len(dev1)
