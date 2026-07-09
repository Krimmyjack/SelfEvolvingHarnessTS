"""E-3.2 嵌入式评估器（e32_nested）守卫测试 — A-39⑥（评审第十五轮，全过才允许正式跑）。

fake caches：PhiX=常数、Y=y_a（动作恒值）→ Ridge 头预测 ≈ y_a → loss(u,a)=|y_a − fut(u)|
——可控 loss 矩阵 + 真实标签生成路径（与 toy 直接给 L 不同，能测到边界错配）。
"""
from __future__ import annotations

import copy
import dataclasses
import json

import numpy as np
import pytest

from SelfEvolvingHarnessTS.e32_nested import (aggregate_records, run_policy_folds,
                                              stratified_folds)
from SelfEvolvingHarnessTS.e32_policy import (GBDTArm, LookupArm, evaluate, key_cell,
                                              key_cell_origin, residualized_perm_test,
                                              snr_strata)

ACTIONS = ["v_median", "a_light", "a_heavy"]
Y_OF = {"v_median": 1.0, "a_light": 0.95, "a_heavy": 0.5}


def _fake_cache(y_a: float, fut: float, n_w: int = 8):
    # H=2：sklearn Ridge 对单列 y 会 ravel 成 1 维预测（真实缓存 H=48 不受影响），故用 2 列
    return dict(PhiX=np.ones((n_w, 1)), Y=np.full((n_w, 2), y_a),
                PhiTest=np.ones((1, 1)), future=np.array([fut, fut], float), obs=1.0)


def _cells_data(seed: int = 0, n_per: int = 10):
    """2 cell × 2 origin × n_per uid。S_trend fut≈0.5（嗜 a_heavy）、S_season fut≈1.0（嗜 v_median）。"""
    rng = np.random.default_rng(seed)
    out = {}
    for cid in ("cellA", "cellB"):
        uids, origin_of, feats_of, true_d = [], {}, {}, {}
        caches = {a: {} for a in ACTIONS}
        for origin, fut_base in (("S_trend", 0.5), ("S_season", 1.0)):
            for j in range(n_per):
                u = f"{cid}:{origin}:{j}"
                fut = fut_base + 0.02 * (j - n_per / 2) / n_per
                for a in ACTIONS:
                    caches[a][u] = _fake_cache(Y_OF[a], fut)
                uids.append(u)
                origin_of[u] = origin
                feats_of[u] = {"SNR": float(rng.normal(5, 1)), "missing_rate": 0.0,
                               "seasonal_strength": 1.0 if origin == "S_season" else 0.0,
                               "trend_strength": 1.0 if origin == "S_trend" else 0.0}
                true_d[u] = (0.1, 0.0)
        out[cid] = dict(action_caches=caches, uids=uids, origin_of=origin_of,
                        feats_of=feats_of, true_d=true_d)
    return out


def _arms(seed: int = 0):
    return {"global": lambda: LookupArm(None), "d_lookup": lambda: LookupArm(key_cell),
            "dp_gbdt": lambda: GBDTArm(("d", "p"), seed=seed),
            "dp_abstain": lambda: GBDTArm(("d", "p"), abstain=True, seed=seed),
            "oracle_struct": lambda: LookupArm(key_cell_origin)}


def _single_fold(cd):
    all_uids = sorted(u for c in cd.values() for u in c["uids"])
    te = [u for u in all_uids if u.endswith(("7", "8", "9"))]
    tr = [u for u in all_uids if u not in set(te)]
    return [("f0", tr, te)]


@pytest.fixture(scope="module")
def cd():
    return _cells_data()


def test_stratified_folds_cover_and_balance(cd):
    uids = [u for c in cd.values() for u in c["uids"]]
    strat = {u: f"{cid}|{c['origin_of'][u]}" for cid, c in cd.items() for u in c["uids"]}
    fold_of = stratified_folds(uids, strat, 5, seed=0)
    assert set(fold_of) == set(uids)
    sizes = np.bincount([fold_of[u] for u in uids], minlength=5)
    assert sizes.max() - sizes.min() <= 4                      # 4 层 × round-robin 余数


def test_labels_route_structure_and_coverage(cd):
    res = run_policy_folds(cd, ACTIONS, _arms(), _single_fold(cd), seed=0, verbose=False)
    recs = res["records"]
    assert {r["uid"] for r in recs} == set(_single_fold(cd)[0][2])
    agg = aggregate_records(recs, ACTIONS, list(_arms()))
    # oracle_struct 应把 trend→a_heavy / season→v_median 学出（loss 结构由 fake caches 决定）
    for r in recs:
        want = "a_heavy" if r["origin"] == "S_trend" else "v_median"
        assert r["arms"]["oracle_struct"]["pick"] == want
    assert agg["oracle_struct"]["mean_regret"] < agg["global"]["mean_regret"] - 0.05


def test_poison_test_future_does_not_change_picks(cd):
    """守卫①：outer-test 的 future 只进评估标签，不得影响任何臂的 picks。"""
    folds = _single_fold(cd)
    te = set(folds[0][2])
    base = run_policy_folds(cd, ACTIONS, _arms(), folds, seed=0, verbose=False)
    cd2 = copy.deepcopy(cd)
    for c in cd2.values():
        for a in ACTIONS:
            for u in te & set(c["uids"]):
                c["action_caches"][a][u]["future"] = c["action_caches"][a][u]["future"] * 10 + 5
    poisoned = run_policy_folds(cd2, ACTIONS, _arms(), folds, seed=0, verbose=False)
    for rb, rp in zip(base["records"], poisoned["records"]):
        assert rb["arms"] == rp["arms"], f"{rb['uid']} picks 被 test future 污染"
    assert any(rb["L_test"] != rp["L_test"] for rb, rp in zip(base["records"], poisoned["records"]))


def test_test_uid_never_enters_head_training(cd):
    """守卫②③：NaN 污染 outer-test 的训练窗（PhiX/Y）后——train 标签 bit 级不变且全有限。
    若任何头用了 test uid 的窗，NaN 会传染到系数 → 标签变 NaN。"""
    folds = _single_fold(cd)
    te = set(folds[0][2])
    base = run_policy_folds(cd, ACTIONS, _arms(), folds, seed=0, verbose=False)
    cd2 = copy.deepcopy(cd)
    for c in cd2.values():
        for a in ACTIONS:
            for u in te & set(c["uids"]):
                c["action_caches"][a][u]["PhiX"] = np.full_like(c["action_caches"][a][u]["PhiX"], np.nan)
                c["action_caches"][a][u]["Y"] = np.full_like(c["action_caches"][a][u]["Y"], np.nan)
    poisoned = run_policy_folds(cd2, ACTIONS, _arms(), folds, seed=0, verbose=False)
    lt_b = base["fold_details"][0]["L_train"]
    lt_p = poisoned["fold_details"][0]["L_train"]
    assert lt_b == lt_p, "train 标签受 test uid 训练窗影响（头训练边界泄漏）"
    assert all(np.isfinite(v) for d in lt_p.values() for v in d.values())
    for rb, rp in zip(base["records"], poisoned["records"]):
        assert rb["arms"] == rp["arms"]
        assert all(np.isfinite(v) for v in rp["L_test"].values())


def test_origin_permutation_invariant_for_learned_arms(cd):
    """守卫④：origin 只进 oracle 臂/分层——置换 origin 不得改变 learned 臂的 picks。"""
    folds = _single_fold(cd)
    base = run_policy_folds(cd, ACTIONS, _arms(), folds, seed=0, verbose=False)
    cd2 = copy.deepcopy(cd)
    rng = np.random.default_rng(3)
    for c in cd2.values():
        us = list(c["origin_of"])
        vals = [c["origin_of"][u] for u in us]
        for u, v in zip(us, [vals[i] for i in rng.permutation(len(vals))]):
            c["origin_of"][u] = v
    perm = run_policy_folds(cd2, ACTIONS, _arms(), folds, seed=0, verbose=False)
    for rb, rp in zip(base["records"], perm["records"]):
        for n in ("global", "d_lookup", "dp_gbdt", "dp_abstain"):
            assert rb["arms"][n] == rp["arms"][n], f"learned 臂 {n} 读到了 origin"


def test_resume_bit_identical(cd, tmp_path):
    """守卫⑥：per-fold checkpoint 续跑与一次跑 bit 级一致。"""
    uids = sorted(u for c in cd.values() for u in c["uids"])
    strat = {u: f"{cid}|{c['origin_of'][u]}" for cid, c in cd.items() for u in c["uids"]}
    fold_of = stratified_folds(uids, strat, 3, seed=1)
    folds = [(f"f{f}", [u for u in uids if fold_of[u] != f],
              [u for u in uids if fold_of[u] == f]) for f in range(3)]
    one = run_policy_folds(cd, ACTIONS, _arms(), folds, seed=1,
                           ckpt_dir=tmp_path / "oneshot", verbose=False)
    interrupted = run_policy_folds(cd, ACTIONS, _arms(), folds, seed=1,
                                   ckpt_dir=tmp_path / "resume", _stop_after=1, verbose=False)
    assert len(interrupted["records"]) < len(one["records"])
    resumed = run_policy_folds(cd, ACTIONS, _arms(), folds, seed=1,
                               ckpt_dir=tmp_path / "resume", verbose=False)
    assert json.dumps(resumed["records"], sort_keys=True) == json.dumps(one["records"], sort_keys=True)


# ══════════════════════════════════════════════════════════════════════════
# 守卫⑤：residualized 置换检验正/负对照（PolicyData 级，标签直接给定）
# ══════════════════════════════════════════════════════════════════════════
def _perm_toy(seed: int, informative: bool):
    from SelfEvolvingHarnessTS.tests.test_e32_policy import _toy
    data = _toy(seed=seed, n_per=24)
    if not informative:
        rng = np.random.default_rng(seed + 99)
        data = dataclasses.replace(data, X_p=rng.normal(0, 1, data.X_p.shape))
    arms = {"d_gbdt": lambda: GBDTArm(("d",), seed=0), "dp_gbdt": lambda: GBDTArm(("d", "p"), seed=0)}

    def stat(X_p):
        d2 = dataclasses.replace(data, X_p=X_p)
        r = evaluate(d2, n_splits=4, seed=0, arms=arms)
        return r["d_gbdt"]["mean_regret"] - r["dp_gbdt"]["mean_regret"]

    strata = snr_strata(data.X_d[:, 0], data.cell, n_bins=3)
    return stat, data.X_p, strata


def test_perm_positive_control():
    stat, X_p, strata = _perm_toy(seed=0, informative=True)
    r = residualized_perm_test(stat, X_p, strata, n_perm=19, seed=0)
    assert r["T_obs"] > 0.05                                   # P 真有增量
    assert r["p"] <= 0.10                                      # 19 perm 下最小 p=0.05


def test_perm_negative_control():
    stat, X_p, strata = _perm_toy(seed=1, informative=False)
    r = residualized_perm_test(stat, X_p, strata, n_perm=19, seed=1)
    assert r["p"] >= 0.15                                      # 纯噪声 P 不得显著


def test_perm_resume_matches_oneshot():
    stat, X_p, strata = _perm_toy(seed=0, informative=True)
    full = residualized_perm_test(stat, X_p, strata, n_perm=9, seed=0)
    part = residualized_perm_test(stat, X_p, strata, n_perm=4, seed=0)
    res = residualized_perm_test(stat, X_p, strata, n_perm=9, seed=0, done_nulls=part["nulls"])
    assert res["nulls"] == full["nulls"] and res["p"] == full["p"]
