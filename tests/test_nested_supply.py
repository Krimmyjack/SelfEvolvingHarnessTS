"""tests/test_nested_supply.py — nested Δ_supply 防泄漏守卫（A-31a，评审第十二轮清单）。

全过是启动正式 E-3.3 的硬前置：
  ①每 outer fold 的 fit/test uid 严格不交且并集覆盖全体；
  ②inner 动作选择只读 outer-train：扰动 outer-test 的 future/PhiTest 不改变所选动作；
  ③两池使用完全相同的 outer folds（paired 前提）；
  ④纯噪声/重复动作扩充不产生稳定正 Δ_supply（winner's-curse 免疫）；
  ⑤真实更优动作可被检出（功效对照，防"防泄漏防到没功效"）。
合成缓存不依赖 torch：直接构造 PhiX/Y/PhiTest/future/obs（与 `_cache_one` 同字段契约）。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.nested_supply import (
    delta_supply, delta_supply_grouped, inner_select, make_folds,
    nested_pool_losses, _inner_select_w)

D, H, NW = 8, 6, 30          # 特征维 / 预测步长 / 每 uid 训练窗数
N_UID = 80


def _make_caches(n_uid=N_UID, actions=("a0", "a1", "a2"), good=(), seed=0):
    """合成 uid×action 缓存。good 中的动作特征可线性预测 future（loss≈0），其余为纯噪声。
    非 good 动作的 (PhiX,Y,PhiTest,future) 全部独立随机 → 动作间无真实差异。"""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, (D, H))
    uids = [f"u{i:03d}" for i in range(n_uid)]
    out = {}
    for a in actions:
        d = {}
        for u in uids:
            PhiX = rng.normal(0, 1, (NW, D))
            PhiTest = rng.normal(0, 1, (1, D))
            if a in good:
                Y = PhiX @ W + rng.normal(0, 0.01, (NW, H))
                future = (PhiTest @ W).ravel() + rng.normal(0, 0.01, H)
            else:
                Y = rng.normal(0, 1, (NW, H))
                future = rng.normal(0, 1, H)
            d[u] = dict(PhiX=PhiX, Y=Y, PhiTest=PhiTest, future=future, obs=1.0)
        out[a] = d
    return out, uids


# ── ① fit/test 严格不交、覆盖全体 ─────────────────────────────────────────
def test_outer_folds_disjoint_and_cover():
    caches, uids = _make_caches()
    _, picks = nested_pool_losses(caches, ["a0", "a1"], uids, outer_k=5, inner_k=3)
    seen_test = []
    for p in picks:
        tr, te = set(p["train_uids"]), set(p["test_uids"])
        assert not (tr & te), "outer fold 的 fit 与 test uid 相交（泄漏）"
        assert tr | te == set(uids)
        seen_test.extend(p["test_uids"])
    assert sorted(seen_test) == sorted(uids), "outer-test 未无重复覆盖全体 uid"


def test_folds_deterministic():
    uids = [f"u{i}" for i in range(50)]
    assert make_folds(uids, 5, seed=1) == make_folds(list(reversed(uids)), 5, seed=1)
    assert make_folds(uids, 5, seed=1) != make_folds(uids, 5, seed=2)


# ── ② inner 选择只读 outer-train：污染 outer-test 不改选择 ────────────────
def test_selection_immune_to_test_corruption():
    caches, uids = _make_caches(actions=("a0", "a1", "a2", "a3"), seed=3)
    pool = ["a0", "a1", "a2", "a3"]
    _, picks_ref = nested_pool_losses(caches, pool, uids, outer_k=5, inner_k=3)
    for p in picks_ref:
        te = p["test_uids"]
        corrupted = {a: dict(d) for a, d in caches.items()}
        for a in pool:                                   # 大幅污染该 fold 的 outer-test
            for u in te:
                c = dict(corrupted[a][u])
                c["future"] = c["future"] + 1e6
                c["PhiTest"] = c["PhiTest"] * -50.0
                corrupted[a][u] = c
        _, picks_new = nested_pool_losses(corrupted, pool, uids, outer_k=5, inner_k=3)
        sel_new = {q["outer_fold"]: q["selected"] for q in picks_new}
        assert sel_new[p["outer_fold"]] == p["selected"], (
            f"outer fold {p['outer_fold']}：污染 outer-test 改变了所选动作 → 选择读到了 test（泄漏）")


def test_inner_select_never_touches_held_out_uids():
    caches, uids = _make_caches(seed=4)
    train = uids[:60]
    probe = {a: {u: caches[a][u] for u in train} for a in caches}   # 物理上只给 outer-train
    best, means = inner_select(probe, ["a0", "a1", "a2"], train, inner_k=3, seed=9)
    assert best in ("a0", "a1", "a2") and len(means) == 3            # 不需要 held-out 即可完成选择


# ── ③ 两池共用同一 outer folds ────────────────────────────────────────────
def test_pools_share_outer_folds():
    caches, uids = _make_caches(actions=("a0", "a1", "a2", "a3"), seed=5)
    res = delta_supply(caches, ["a0", "a1"], ["a0", "a1", "a2", "a3"], uids,
                       outer_k=5, inner_k=3, n_boot=50)
    fb = {p["outer_fold"]: sorted(p["test_uids"]) for p in res["picks_base"]}
    fe = {p["outer_fold"]: sorted(p["test_uids"]) for p in res["picks_expanded"]}
    assert fb == fe, "base 与 expanded 池 outer folds 不一致（paired 前提破坏）"


# ── ④ 纯噪声/重复动作扩充：Δ_supply 不得稳定为正（winner's-curse 免疫） ──
def test_null_expansion_no_positive_delta():
    """base=3 噪声动作；expanded=再加 10 个噪声动作。in-sample oracle 会机械增益
    （A-30a 实测 +0.05~+0.16），nested held-out 必须≈0 且 CI 覆盖 0。"""
    acts = tuple(f"a{i}" for i in range(13))
    caches, uids = _make_caches(actions=acts, seed=6)
    res = delta_supply(caches, list(acts[:3]), list(acts), uids,
                       outer_k=5, inner_k=3, n_boot=300)
    assert res["ci_lo"] <= 0.0, f"纯噪声扩充给出 CI 下界>0 的假阳性: {res['ci_lo']:.4f}"
    assert abs(res["delta_mean"]) < 0.15, f"纯噪声扩充 Δ 均值异常大: {res['delta_mean']:.4f}"


def test_duplicate_expansion_no_positive_delta():
    """expanded=base 的逐字节重复列 ×3 → Δ_supply 必须恰为 0（同折同选同头）。"""
    caches, uids = _make_caches(actions=("a0", "a1"), seed=7)
    dup = dict(caches)
    for i in range(3):
        dup[f"dup{i}"] = caches["a0"]
    res = delta_supply(dup, ["a0", "a1"], ["a0", "a1", "dup0", "dup1", "dup2"], uids,
                       outer_k=5, inner_k=3, n_boot=50)
    assert abs(res["delta_mean"]) < 1e-9, "重复动作扩充产生了非零 Δ_supply"


# ── ⑤ 功效对照：真实更优动作可被检出 ─────────────────────────────────────
def test_real_improvement_detected():
    acts = ("a0", "a1", "a2", "good")
    caches, uids = _make_caches(actions=acts, good=("good",), seed=8)
    res = delta_supply(caches, ["a0", "a1", "a2"], list(acts), uids,
                       outer_k=5, inner_k=3, n_boot=300)
    assert res["delta_mean"] > 0.3, f"真实更优动作未被检出: Δ={res['delta_mean']:.4f}"
    assert res["ci_lo"] > 0.0, "真实改进的 CI 下界应>0"
    picked = {p["selected"] for p in res["picks_expanded"]}
    assert picked == {"good"}, f"expanded 池应在全部 outer fold 选中 good，实选 {picked}"


# ══════════════════════════════════════════════════════════════════════════
# A-33c：full-refit group bootstrap（正式判决 CI）——同样的防泄漏/功效守卫
# ══════════════════════════════════════════════════════════════════════════
def test_grouped_inner_select_ignores_held_out_corruption():
    """`_inner_select_w` 只迭代 train_uids：污染非 train uid 的缓存不得改变所选动作。"""
    caches, uids = _make_caches(actions=("a0", "a1", "a2", "good"), good=("good",), seed=11)
    train = uids[:55]
    mult = {u: 1.0 for u in uids}
    clean = _inner_select_w(caches, ["a0", "a1", "a2", "good"], train, mult, inner_k=3, seed=13)
    corrupted = {a: dict(d) for a, d in caches.items()}
    for a in corrupted:                                          # 大幅污染 held-out（非 train）uid
        for u in uids[55:]:
            c = dict(corrupted[a][u]); c["future"] = c["future"] + 1e6; corrupted[a][u] = c
    dirty = _inner_select_w(corrupted, ["a0", "a1", "a2", "good"], train, mult, inner_k=3, seed=13)
    assert clean == dirty == "good", f"held-out 污染改变了加权 inner 选择（泄漏）: {clean}→{dirty}"


def test_grouped_null_expansion_no_positive_delta():
    """纯噪声扩充：full-refit group bootstrap 的 CI 更宽（多方差源）→ 下界必须≤0。"""
    acts = tuple(f"a{i}" for i in range(13))
    caches, uids = _make_caches(actions=acts, seed=14)
    res = delta_supply_grouped(caches, list(acts[:3]), list(acts), uids,
                               outer_k=5, inner_k=3, n_boot=200)
    assert res["method"] == "grouped_full_refit_bootstrap"
    assert res["ci_lo"] <= 0.0, f"纯噪声扩充 grouped CI 下界>0（假阳性）: {res['ci_lo']:.4f}"
    assert abs(res["delta_mean"]) < 0.15, f"纯噪声扩充 Δ 均值异常大: {res['delta_mean']:.4f}"


def test_grouped_duplicate_expansion_zero_delta():
    """expanded=base 逐字节重复列：两池每 replicate 同折同选同头 → Δ 恒为 0。"""
    caches, uids = _make_caches(actions=("a0", "a1"), seed=15)
    dup = dict(caches)
    for i in range(3):
        dup[f"dup{i}"] = caches["a0"]
    res = delta_supply_grouped(dup, ["a0", "a1"], ["a0", "a1", "dup0", "dup1", "dup2"], uids,
                               outer_k=5, inner_k=3, n_boot=40)
    assert abs(res["delta_mean"]) < 1e-9, f"重复列扩充产生非零 Δ_supply: {res['delta_mean']}"
    assert res["ci_lo"] <= 1e-9 <= res["ci_hi"] + 1e-9


def test_grouped_real_improvement_detected():
    """功效对照：真实更优动作在 group bootstrap（更保守 CI）下仍应 Δ>0 且下界>0，
    且 expanded 池的 bootstrap 选择频率被 good 主导（防'防泄漏防到没功效'）。"""
    acts = ("a0", "a1", "a2", "good")
    caches, uids = _make_caches(actions=acts, good=("good",), seed=16)
    res = delta_supply_grouped(caches, ["a0", "a1", "a2"], list(acts), uids,
                               outer_k=5, inner_k=3, n_boot=200)
    assert res["delta_mean"] > 0.3, f"真实更优动作未被检出: Δ={res['delta_mean']:.4f}"
    assert res["ci_lo"] > 0.0, f"真实改进 grouped CI 下界应>0: {res['ci_lo']:.4f}"
    assert res["frac_boot_positive"] > 0.95, "真实改进应几乎每个 replicate 都为正"
    assert res["expanded_pick_freq"].get("good", 0) > 0.9, (
        f"expanded 选择频率应被 good 主导: {res['expanded_pick_freq']}")
