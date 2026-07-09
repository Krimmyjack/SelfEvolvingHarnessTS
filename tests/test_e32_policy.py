"""E-3.2 骨架（e32_policy）toy 测试 — A-37。

toy 构造：结构决定最优动作（S_trend 嗜 a_heavy、S_season 抗 a_heavy）、D 特征无信息
→ 预期 P/D+P 臂胜 Global/D-only、oracle_struct 为上界、abstain 机制按 κ 单调。
不触碰真实语料（正式跑等 A-31e + 协议冻结）。
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from SelfEvolvingHarnessTS.e32_policy import (FALLBACK_ACTION, GBDTArm, LookupArm, PolicyData,
                                              evaluate, key_cell, lodo_evaluate, make_arms,
                                              verdict_d32e)

ACTIONS = ["v_median", "a_light", "a_heavy"]     # v_median = incumbent/fallback（冻结）


def _toy(seed: int = 0, n_per: int = 40, heavy_mode: str = "structured") -> PolicyData:
    """heavy_mode: structured=季节 1.6/趋势 0.5/noisy 掷币；all_good=a_heavy 恒 0.3（测不 abstain）。"""
    rng = np.random.default_rng(seed)
    uids, cells, origins = [], [], []
    L_rows, Xp_rows, Xd_rows = [], [], []
    for origin in ("S_season", "S_trend", "S_noisy"):
        for j in range(n_per):
            uid = f"{origin}:{j}"
            e = rng.normal(0, 0.02, 3)
            v_med, a_light = 1.0 + e[0], 0.95 + e[1]
            if heavy_mode == "all_good":
                a_heavy = 0.30 + e[2]
            elif origin == "S_season":
                a_heavy = 1.60 + e[2]
            elif origin == "S_trend":
                a_heavy = 0.50 + e[2]
            else:                                  # S_noisy：per-uid 掷币，特征不可预测
                # 独立随机源——严禁与 cell 的 j%2 同键，否则 cell 意外携带掷币信息（D 变有信息）
                a_heavy = (0.40 if rng.random() < 0.5 else 1.90) + e[2]
            L_rows.append([v_med, a_light, a_heavy])
            p0 = (1.0 if origin == "S_season" else 0.0) + rng.normal(0, 0.05)
            p1 = (1.0 if origin == "S_trend" else 0.0) + rng.normal(0, 0.05)
            p2 = rng.uniform(0, 1)
            Xp_rows.append([p0, p1, p2])
            Xd_rows.append(rng.normal(0, 1, 2))    # D 无信息
            uids.append(uid)
            origins.append(origin)
            cells.append("c0" if j % 2 == 0 else "c1")
    return PolicyData(uids=uids, actions=list(ACTIONS), L=np.array(L_rows),
                      X_d=np.array(Xd_rows), X_p=np.array(Xp_rows),
                      cell=np.array(cells), origin=np.array(origins))


@pytest.fixture(scope="module")
def toy():
    return _toy(seed=0)


@pytest.fixture(scope="module")
def res(toy):
    return evaluate(toy, n_splits=5, seed=0)


def test_fallback_must_be_in_pool():
    d = _toy(seed=1, n_per=8)
    with pytest.raises(AssertionError):
        PolicyData(uids=d.uids, actions=["a_light", "a_heavy", "x"], L=d.L,
                   X_d=d.X_d, X_p=d.X_p, cell=d.cell, origin=d.origin)


def test_every_uid_covered_once(toy, res):
    for name in make_arms():
        assert (res[name]["picks"] >= 0).all()


def test_dp_and_p_beat_global_and_dlookup(res):
    """P 承重、D 无信息 → D+P 与 P-only 应胜 Global 与 D-only（margin 0.05 ≫ toy 噪声）。"""
    assert res["dp_gbdt"]["mean_regret"] < res["global"]["mean_regret"] - 0.05
    assert res["dp_gbdt"]["mean_regret"] < res["d_lookup"]["mean_regret"] - 0.05
    assert res["p_gbdt"]["mean_regret"] < res["global"]["mean_regret"] - 0.05
    # D 无信息 → D-only（表/GBDT）≈ Global
    assert abs(res["d_lookup"]["mean_regret"] - res["global"]["mean_regret"]) < 0.03
    assert abs(res["d_gbdt"]["mean_regret"] - res["global"]["mean_regret"]) < 0.05


def test_oracle_struct_is_upper_bound(res):
    assert res["oracle_struct"]["mean_regret"] <= res["dp_gbdt"]["mean_regret"] + 0.02


def test_abstained_rows_get_fallback(toy):
    tr = np.arange(0, toy.n, 2)
    te = np.arange(1, toy.n, 2)
    arm = GBDTArm(("d", "p"), abstain=True, seed=0, kappa=5.0).fit(toy, tr)
    p, a = arm.picks(toy, te)
    assert a.any(), "κ=5 下应有 abstain 触发"
    assert (p[a] == toy.fallback_idx).all()


def test_abstain_monotone_in_kappa(toy):
    tr = np.arange(0, toy.n, 2)
    te = np.arange(1, toy.n, 2)
    rates = []
    for kappa in (0.0, 1.0, 5.0):
        arm = GBDTArm(("d", "p"), abstain=True, seed=0, kappa=kappa).fit(toy, tr)
        _, a = arm.picks(toy, te)
        rates.append(a.mean())
    assert rates[0] == 0.0                      # κ=0 → 永不 abstain
    assert rates[0] <= rates[1] <= rates[2]


def test_no_abstain_when_advantage_huge():
    d = _toy(seed=2, heavy_mode="all_good")     # a_heavy 恒 0.3，确定性大优势
    tr = np.arange(0, d.n, 2)
    te = np.arange(1, d.n, 2)
    arm = GBDTArm(("d", "p"), abstain=True, seed=0, kappa=1.0).fit(d, tr)
    p, a = arm.picks(d, te)
    assert a.mean() < 0.05
    assert (p == d.actions.index("a_heavy")).mean() > 0.9


def test_no_leak_test_losses_dont_affect_picks(toy):
    """策略只准用 train 的 L 与 test 的 X：污染 test 行的 L 不得改变任何臂的 picks。"""
    tr = np.arange(0, toy.n, 2)
    te = np.arange(1, toy.n, 2)
    doctored = dataclasses.replace(toy, L=toy.L.copy())
    doctored.L[te] = doctored.L[te] * 10 + 7.0
    for name, mk in make_arms().items():
        p1, _ = mk().fit(toy, tr).picks(toy, te)
        p2, _ = mk().fit(doctored, tr).picks(doctored, te)
        assert np.array_equal(p1, p2), f"{name} 的 picks 受 test loss 污染影响"


def test_deterministic(toy):
    r1 = evaluate(toy, n_splits=5, seed=0)
    r2 = evaluate(toy, n_splits=5, seed=0)
    for name in r1:
        assert r1[name]["mean_regret"] == r2[name]["mean_regret"]


def test_lodo_smoke(toy):
    out = lodo_evaluate(toy, group_field="origin", seed=0)
    assert set(out) == {"S_season", "S_trend", "S_noisy"}
    for g, res_g in out.items():
        assert "oracle_struct" not in res_g     # 留一结构下诊断臂无意义，应排除
        for name, r in res_g.items():
            assert np.isfinite(r["mean_regret"]) and np.isfinite(r["mean_delta_vs_incumbent"])


def test_verdict_shape_and_core_criteria(res):
    v = verdict_d32e(res)
    assert set(v) >= {"i_dp_beats_global", "ii_dp_beats_dlookup", "iii_season_worst_lcb_ok",
                      "v_abstain_not_worse", "vi_dp_beats_continuous_d"}
    assert v["i_dp_beats_global"] is True       # toy 构造保证
    assert v["ii_dp_beats_dlookup"] is True
    assert v["vi_dp_beats_continuous_d"] is True
