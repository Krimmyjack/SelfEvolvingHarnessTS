"""tests/test_p6_metrics.py — P6 冻结效应度量词汇表 + signature 计算器（prereg §0/§4）。

运行：D:\\Anaconda_envs\\envs\\project\\python.exe -m pytest SelfEvolvingHarnessTS/tests/test_p6_metrics.py -q
（cwd = C:\\Users\\辉\\Desktop\\Agent）

全部确定性：无文件 IO、无网络；bootstrap 只用显式 seed 的 default_rng。
边界用例一律用二进制精确值（0.25/0.5/1.25 …）钉死 ≥/</> 的严格性，不受浮点毛刺影响。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.p6.metrics import (
    ACTIVATION_PRIORITY,
    DEFAULT_BOOTSTRAP_B,
    EFFECT_CLASS_TOL,
    S2_MEAN_CLASSES_THRESHOLD,
    activate,
    cluster_bootstrap_means,
    cluster_lcb90,
    effect_classes,
    gain,
    gain_from_batch_delta,
    harm,
    normalized_headroom,
    regret,
    s1_selector,
    s2_supply,
    s3_scope_harm,
)


# ============================== A. 词汇表恒等式（prereg §0） ==============================
def test_vocabulary_identities():
    # gain(H→e) = loss_H − loss_e
    assert gain(1.5, 1.25) == 0.25
    assert gain(1.0, 1.5) == -0.5
    # batch_delta = loss_new − loss_old = −gain ⇒ gain_from_batch_delta(d) = −d
    assert gain_from_batch_delta(0.25) == -0.25
    assert gain_from_batch_delta(-0.25) == 0.25
    # harm = −gain；恒等式 harm(gain_from_batch_delta(d)) == d
    assert harm(0.25) == -0.25
    for d in (-1.5, -0.25, 0.0, 0.25, 2.0):
        assert harm(gain_from_batch_delta(d)) == d
        assert gain_from_batch_delta(d) == -d
    # 判官两侧 loss：gain(old→new) 与 batch_delta = loss_new − loss_old 互为相反数
    lo, ln = 1.5, 1.25
    assert gain(lo, ln) == gain_from_batch_delta(ln - lo) == 0.25
    assert harm(gain(lo, ln)) == ln - lo


def test_regret_semantics_and_caller_error():
    assert regret(1.25, 1.0) == 0.25
    assert regret(1.0, 1.0) == 0.0
    assert regret(1.0, 1.0 + 5e-13) == 0.0            # ≤1e-12 浮点毛刺截为 0
    with pytest.raises(ValueError):                   # loss_pool_min 不是池最小 → 口径错
        regret(1.0, 1.25)


def test_vocabulary_rejects_non_finite():
    for fn, args in ((gain, (np.nan, 1.0)), (gain, (1.0, np.inf)),
                     (gain_from_batch_delta, (np.nan,)), (harm, (np.inf,)),
                     (regret, (np.nan, 0.0)), (regret, (1.0, np.nan))):
        with pytest.raises(ValueError):
            fn(*args)


# ============================== B. 行为等价类（prereg §4） ==============================
def test_effect_classes_distinct_and_ordering():
    out = effect_classes({"c": 2.0, "a": 0.0, "b": 1.0})
    assert out == [{"a"}, {"b"}, {"c"}]               # 按类内最小 loss 升序
    assert effect_classes({}) == []
    assert effect_classes({"x": 1.0, "y": 1.0}) == [{"x", "y"}]   # 精确相等必并类


def test_effect_classes_tie_exact_boundary():
    tol = EFFECT_CLASS_TOL                            # 1e-9
    # 差恰 == tol → 同类（exact-tie 按 ≤ 处理）；差 = 2·tol → 分开
    assert effect_classes({"a": 0.0, "b": 1e-9}, tol=tol) == [{"a", "b"}]
    assert effect_classes({"a": 0.0, "b": 2e-9}, tol=tol) == [{"a"}, {"b"}]
    # union-find 链式传递：a~b、b~c ⇒ 全并，即使 |a−c| = 2·tol > tol
    assert effect_classes({"a": 0.0, "b": 1e-9, "c": 2e-9}, tol=tol) == [{"a", "b", "c"}]
    # 断链：中间空隙 > tol 处分裂
    assert effect_classes({"a": 0.0, "b": 1e-9, "c": 5e-9, "d": 6e-9}, tol=tol) == [
        {"a", "b"}, {"c", "d"}]


def test_effect_classes_rejects_bad_input():
    with pytest.raises(ValueError):
        effect_classes({"a": float("nan")})
    with pytest.raises(ValueError):
        effect_classes({"a": 1.0}, tol=-1e-9)


# ============================== C. 聚类 bootstrap（prereg §4 冻结契约） ==============================
def test_cluster_bootstrap_deterministic():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    cids = ["a", "a", "b", "b", "c", "c"]
    r1 = cluster_lcb90(vals, cids, 500, seed=123)
    r2 = cluster_lcb90(vals, cids, 500, seed=123)
    assert r1 == r2                                    # 同 seed → bit 级同结果
    # 换 seed → 抽取流必变（LCB 分位在 3 簇的离散均值空间上可能撞值，故看整条均值分布）
    m123 = cluster_bootstrap_means(vals, cids, 500, seed=123)
    m124 = cluster_bootstrap_means(vals, cids, 500, seed=124)
    assert np.array_equal(m123, cluster_bootstrap_means(vals, cids, 500, seed=123))
    assert not np.array_equal(m123, m124)
    m = cluster_bootstrap_means(vals, cids, 37, seed=5)
    assert m.shape == (37,)


def test_cluster_bootstrap_manual_small_sample_reference():
    """手工对照：复现冻结抽取契约（单次 rng.integers(0, n_clusters, size=(b, n_clusters))），
    用纯 Python 循环独立算 replicate 均值与 5% 分位，与实现 bit 级一致。"""
    values = [0.0, 0.0, 1.0, 1.0]
    clusters = ["A", "A", "B", "B"]                    # 首次出现次序：A→0、B→1
    b, seed = 64, 7
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, 2, size=(b, 2))
    cvals = {0: [0.0, 0.0], 1: [1.0, 1.0]}
    means = []
    for row in draw:
        pooled = []
        for j in row:
            pooled.extend(cvals[int(j)])
        means.append(sum(pooled) / len(pooled))
    got_means = cluster_bootstrap_means(values, clusters, b, seed=seed)
    assert np.array_equal(got_means, np.asarray(means, dtype=float))
    expected_lcb = float(np.quantile(np.asarray(means, dtype=float), 0.05, method="linear"))
    assert cluster_lcb90(values, clusters, b, seed=seed) == expected_lcb


def test_cluster_bootstrap_cluster_semantics_same_cluster_drawn_together():
    """同 cluster 同抽：cluster 内值恒同抽 ⇒ replicate 均值只能落在整簇组合集合里
    （episode 级独立重采样能产出的中间值不可能出现）。"""
    means = cluster_bootstrap_means([5.0, 5.0, 5.0, 9.0, 9.0, 9.0],
                                    ["A", "A", "A", "B", "B", "B"], 400, seed=11)
    assert set(means.tolist()) <= {5.0, 7.0, 9.0}
    # 不等簇大小（等权 episode 口径：Σ和/Σ计数）
    means2 = cluster_bootstrap_means([0.0, 0.0, 0.0, 12.0], ["A", "A", "A", "B"], 400, seed=12)
    assert set(means2.tolist()) <= {0.0, 3.0, 12.0}
    # 单 cluster → 每个 replicate 都是全体均值 ⇒ LCB == 均值（精确）
    assert cluster_lcb90([3.5, 4.5], ["only", "only"], 200, seed=3) == 4.0


def test_cluster_bootstrap_validation():
    with pytest.raises(ValueError):
        cluster_lcb90([], [], 10, seed=1)                          # 空
    with pytest.raises(ValueError):
        cluster_lcb90([1.0, 2.0], ["a"], 10, seed=1)               # 长度不一致
    with pytest.raises(ValueError):
        cluster_lcb90([1.0, float("nan")], ["a", "b"], 10, seed=1)  # 非有限
    with pytest.raises(ValueError):
        cluster_lcb90([1.0], ["a"], 0, seed=1)                     # b < 1


# ============================== D. S1 / S2 / S3 fired 边界 ==============================
def test_s1_selector_fired_and_boundaries():
    clusters4 = ["c0", "c1", "c2", "c3"]
    # 全部 regret = 0.25（二进制精确）：mean == eps（≥ 含边界）且 LCB > 0 → fired
    pe = [{"loss_chosen": 1.25, "loss_pool_min": 1.0}] * 4
    res = s1_selector(pe, clusters4, eps=0.25, b=200, seed=42)
    assert res["regret_mean"] == 0.25 and res["lcb90"] == 0.25
    assert res["fired"] is True
    # eps 抬高 → mean < eps → 不触发（哪怕 LCB > 0）
    assert s1_selector(pe, clusters4, eps=0.3, b=200, seed=42)["fired"] is False
    # 零 regret → mean 0 < eps 且 LCB == 0 → 不触发
    pe0 = [{"loss_chosen": 1.0, "loss_pool_min": 1.0}] * 4
    res0 = s1_selector(pe0, clusters4, eps=0.25, b=200, seed=42)
    assert res0 == {"regret_mean": 0.0, "lcb90": 0.0, "fired": False}


def test_s1_selector_lcb_gate_blocks_despite_mean():
    # 两簇：A 全零、B 全 1 → mean 0.5 ≥ eps=0.5（边界含），但 P(抽到 AA)=0.25 ≫ 5%
    # ⇒ LCB90 == 0（不 > 0）→ 不触发（聚类 CI 门独立于均值门）
    pe = ([{"loss_chosen": 1.0, "loss_pool_min": 1.0}] * 2
          + [{"loss_chosen": 2.0, "loss_pool_min": 1.0}] * 2)
    clusters = ["A", "A", "B", "B"]
    res = s1_selector(pe, clusters, eps=0.5, b=2000, seed=9)
    assert res["regret_mean"] == 0.5
    assert res["lcb90"] == 0.0
    assert res["fired"] is False


def test_s1_selector_validation():
    with pytest.raises(ValueError):
        s1_selector([], [], eps=0.1, seed=1)
    with pytest.raises(ValueError):
        s1_selector([{"loss_chosen": 1.0, "loss_pool_min": 1.0}], ["a", "b"], eps=0.1, seed=1)
    with pytest.raises(ValueError):                    # 池最小口径错（负 regret）→ raise
        s1_selector([{"loss_chosen": 1.0, "loss_pool_min": 1.5}], ["a"], eps=0.1, seed=1)


def test_s2_supply_fired_branches_and_boundaries():
    assert S2_MEAN_CLASSES_THRESHOLD == 2.0
    # 分支①：mean_classes < 2.0（严格）
    res = s2_supply([1, 2], pool_ceiling_gain=1.0, det_ceiling_gain=0.0, eps=0.25)
    assert res["mean_classes"] == 1.5 and res["fired"] is True
    # 边界：mean == 2.0 → 不触发（且 gap 分支也不触发）
    res2 = s2_supply([2, 2], pool_ceiling_gain=0.0, det_ceiling_gain=0.0, eps=0.25)
    assert res2 == {"mean_classes": 2.0, "ceiling_gap": 0.0, "fired": False}
    # 分支②：ceiling_gap < −eps（严格）——池上限劣于 det 阶梯超 ε
    res3 = s2_supply([3, 3], pool_ceiling_gain=0.0, det_ceiling_gain=0.5, eps=0.25)
    assert res3["ceiling_gap"] == -0.5 and res3["fired"] is True
    # 边界：gap == −eps → 不触发
    res4 = s2_supply([3, 3], pool_ceiling_gain=0.0, det_ceiling_gain=0.25, eps=0.25)
    assert res4["ceiling_gap"] == -0.25 and res4["fired"] is False
    # 校验：空 / 类数 < 1 → raise
    with pytest.raises(ValueError):
        s2_supply([], 0.0, 0.0, eps=0.1)
    with pytest.raises(ValueError):
        s2_supply([0, 2], 0.0, 0.0, eps=0.1)


def test_s3_scope_harm_fired_and_min_series_gate():
    # 合格 cohort（5 series）恒定 harm 0.25（gain −0.25）→ LCB == 0.25
    res = s3_scope_harm({"bad": [-0.25] * 5}, delta_safe=0.125, seed=77)
    assert res["worst_cohort"] == "bad"
    assert res["harm_lcb90"] == 0.25
    assert res["fired"] is True
    assert res["per_cohort"]["bad"] == {"n_series": 5, "harm_lcb90": 0.25}
    # 边界：LCB == delta_safe → 不触发（严格 >）
    res_eq = s3_scope_harm({"bad": [-0.25] * 5}, delta_safe=0.25, seed=77)
    assert res_eq["fired"] is False
    # min_series 门：4 series 的巨大 harm cohort 不评估
    res_tiny = s3_scope_harm({"tiny": [-9.0] * 4}, delta_safe=0.125, seed=77)
    assert res_tiny == {"worst_cohort": None, "harm_lcb90": None, "fired": False,
                        "per_cohort": {}}
    # worst = 多 cohort 里 harm LCB 最大者；ineligible 不进 per_cohort
    res_m = s3_scope_harm({"mild": [-0.125] * 5, "bad": [-0.25] * 5, "tiny": [-9.0] * 4},
                          delta_safe=0.1875, seed=77)
    assert res_m["worst_cohort"] == "bad" and res_m["fired"] is True
    assert set(res_m["per_cohort"]) == {"mild", "bad"}
    with pytest.raises(ValueError):
        s3_scope_harm({"x": [-0.1] * 5}, delta_safe=0.1, min_series=0, seed=1)


def test_s3_cohort_seed_stability():
    """cohort 级 seed 由 sha256(f"{seed}|{cohort_id}") 派生：增删其他 cohort
    不改变本 cohort 的 LCB（bootstrap 流 cohort-稳定）。"""
    gains_x = [-0.3, -0.2, -0.25, -0.1, -0.4, -0.15]
    only = s3_scope_harm({"X": gains_x}, delta_safe=0.05, seed=101)
    mixed = s3_scope_harm({"X": gains_x, "aaa": [-0.1] * 5, "zzz": [-1.0] * 3},
                          delta_safe=0.05, seed=101)
    assert only["per_cohort"]["X"] == mixed["per_cohort"]["X"]


# ============================== E. headroom 归一化 + 激活优先序 ==============================
def test_normalized_headroom_formula_and_guards():
    assert normalized_headroom(0.75, 0.5) == 0.5       # (obs−thr)/thr，二进制精确
    assert normalized_headroom(0.5, 0.5) == 0.0
    assert normalized_headroom(0.25, 0.5) == -0.5
    # 负阈值：公式原样（方向义务在调用方）——(−0.75 − (−0.5))/(−0.5) = 0.5，二进制精确
    assert normalized_headroom(-0.75, -0.5) == 0.5
    with pytest.raises(ValueError):
        normalized_headroom(1.0, 0.0)                  # 除零
    with pytest.raises(ValueError):
        normalized_headroom(float("nan"), 1.0)


def test_activate_priority_and_abstain():
    assert ACTIVATION_PRIORITY == ("S1", "S3", "S2")
    # 全不过线 → None（= abstain）；空输入也 abstain
    assert activate({}) is None
    assert activate({"S1": {"fired": False}, "S2": {"fired": False}}) is None
    # 单族触发
    assert activate({"S2": {"fired": True, "headroom": 0.5}}) == "S2"
    # headroom 最大者胜（优先序只用于并列）
    assert activate({"S1": {"fired": True, "headroom": 0.2},
                     "S2": {"fired": True, "headroom": 0.5}}) == "S2"
    # 并列：S1 > S3 > S2
    assert activate({"S1": {"fired": True, "headroom": 0.5},
                     "S2": {"fired": True, "headroom": 0.5}}) == "S1"
    assert activate({"S3": {"fired": True, "headroom": 0.5},
                     "S2": {"fired": True, "headroom": 0.5}}) == "S3"
    assert activate({"S1": {"fired": True, "headroom": 0.5},
                     "S3": {"fired": True, "headroom": 0.5}}) == "S1"
    # 未触发族可缺 headroom；触发族缺/坏 headroom → raise
    assert activate({"S1": {"fired": False},
                     "S3": {"fired": True, "headroom": 0.25}}) == "S3"
    with pytest.raises(ValueError):
        activate({"S1": {"fired": True}})
    with pytest.raises(ValueError):
        activate({"S1": {"fired": True, "headroom": float("nan")}})
    with pytest.raises(ValueError):
        activate({"S9": {"fired": True, "headroom": 1.0}})


def test_default_bootstrap_b_frozen():
    assert DEFAULT_BOOTSTRAP_B == 2000                 # prereg §4：B=2000
