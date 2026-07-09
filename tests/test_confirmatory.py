"""tests/test_confirmatory.py — A-41⑥ confirmatory 守卫（九项 + 序列化/等价性补充）。

全过才允许一次性打开 seeds 20–39。toy 数据不触碰任何 holdout；
`build_corpus_range(0,20)` 等价性测试只读 dev namespace。
"""
from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import SelfEvolvingHarnessTS.confirmatory_corpus as cc
import SelfEvolvingHarnessTS.confirmatory_freeze as cf
import SelfEvolvingHarnessTS.run_confirmatory as rc
from SelfEvolvingHarnessTS.e32_nested import _policy_data
from SelfEvolvingHarnessTS.e32_policy import GBDTArm, LookupArm, key_cell
from SelfEvolvingHarnessTS.run_variance_decomp import _det_seed, build_corpus


# ══════════════════════════════════════════════════════════════ toy 构件
def _fake_cache(y_a: float, fut: float, n_w: int = 8) -> dict:
    return dict(PhiX=np.ones((n_w, 1)), Y=np.full((n_w, 2), y_a),
                PhiTest=np.ones((1, 1)), future=np.array([fut, fut]), obs=1.0)


def _toy_cells(n_per: int = 12) -> tuple:
    """2 cell × 2 动作：x_good 在 cellB 明显更好，v_median 在 cellA 更好。"""
    actions = ["v_median", "x_good"]
    cells = {}
    rng = np.random.default_rng(7)
    for ci, cid in enumerate(("cellA", "cellB")):
        uids = [f"{cid}:u{i}" for i in range(n_per)]
        caches = {a: {} for a in actions}
        for i, u in enumerate(uids):
            fut = 1.0 + 0.05 * rng.standard_normal()
            good = cid == "cellB"
            caches["v_median"][u] = _fake_cache(fut + (0.4 if good else 0.05), fut)
            caches["x_good"][u] = _fake_cache(fut + (0.05 if good else 0.4), fut)
        cells[cid] = dict(
            action_caches=caches, uids=uids,
            origin_of={u: ("S_season" if i % 2 == 0 else "S_trend") for i, u in enumerate(uids)},
            feats_of={u: {"SNR": float(ci * 3 + (i % 4)), "missing_rate": 0.0,
                          "period": float(i % 5)} for i, u in enumerate(uids)},
            true_d={u: (0.1, 0.0) for u in uids})
    return cells, actions


def _toy_L_of(cells, actions):
    L = {}
    for cd in cells.values():
        for u in cd["uids"]:
            L[u] = {}
            for a in actions:
                c = cd["action_caches"][a][u]
                L[u][a] = float(abs(c["Y"][0, 0] - c["future"][0]))
    return L


# ══════════════════════════════════════════════════════════════ 守卫③：freeze 先于 holdout
def test_holdout_gate_requires_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "FREEZE_PATH", tmp_path / "absent.json")
    with pytest.raises(SystemExit):
        cc.build_confirmatory_base()
    with pytest.raises(SystemExit):
        cc.generate_a38c()
    with pytest.raises(SystemExit):
        cc.build_confirmatory_corpus()
    ok = tmp_path / "freeze.json"
    ok.write_text(json.dumps({"config_sha": "x"}), "utf-8")
    monkeypatch.setattr(cc, "FREEZE_PATH", ok)
    assert cc._require_freeze()["config_sha"] == "x"          # 正分支不读 holdout


def test_runner_gate_requires_freeze_and_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--stage", "lt"])
    with pytest.raises(SystemExit):                            # 非 smoke 缺 --open-holdout
        rc.main()
    monkeypatch.setattr(rc, "FREEZE_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(sys, "argv", ["prog", "--stage", "lt", "--open-holdout"])
    with pytest.raises(SystemExit):                            # freeze 缺失
        rc.main()


# ══════════════════════════════════════════════════════════════ 基底等价性（不触 holdout）
def test_corpus_range_equivalence_dev():
    a = build_corpus(20)
    b = cc.build_corpus_range(0, 20)
    assert [x.series_uid for x in a] == [x.series_uid for x in b]
    assert [x.seed for x in a] == [x.seed for x in b]
    assert np.array_equal(a[0].history, b[0].history, equal_nan=True)
    assert np.array_equal(a[-1].future, b[-1].future)


def test_a38c_namespace_disjoint():
    for s in ("S_season", "S_ar"):
        for cid in ("forecast|snrHigh|full", "forecast|snrLow|miss"):
            for k in (0, 7, 123):
                assert (_det_seed(s, "A38C", cid, k) % 2_000_000
                        != _det_seed(s, "A31e", cid, k) % 2_000_000)
    assert ":A38C:" not in "S_season:d_hi_full:25"             # 基底 uid 与补样 uid 可区分


# ══════════════════════════════════════════════════════════════ 守卫④：A38C 零 loss（结构性）
def test_a38c_structurally_loss_free():
    src = inspect.getsource(cc)
    for banned in ("nested_supply", "_fit_head", "Ridge", "frozen_probe", "FrozenProbe",
                   "_oof_losses", "L_test"):
        assert banned not in src, f"confirmatory_corpus 不得触及评估头/loss：{banned}"


# ══════════════════════════════════════════════════════════════ 守卫①+roundtrip：序列化
def test_serialize_sha_and_roundtrip(tmp_path):
    cells, actions = _toy_cells()
    L_of = _toy_L_of(cells, actions)
    order = sorted(u for cd in cells.values() for u in cd["uids"])
    data = _policy_data(cells, actions, order, L_of)
    tr = np.arange(data.n)
    fitted = {"global": LookupArm(None).fit(data, tr),
              "d_lookup": LookupArm(key_cell).fit(data, tr),
              "dp_abstain": GBDTArm(("d", "p"), abstain=True, seed=3).fit(data, tr)}
    path = tmp_path / "arms.joblib"
    sha = cf.serialize_arms(fitted, data, seed=3, path=path)
    blob = cf.load_frozen_arms(path, verify_sha=sha)           # SHA 一致 → 通过
    idx = np.arange(data.n)
    for name in fitted:
        p0, a0 = fitted[name].picks(data, idx)
        p1, a1 = blob["arms"][name].picks(data, idx)
        assert np.array_equal(p0, p1) and np.array_equal(a0, a1), f"roundtrip picks 不一致：{name}"
    with pytest.raises(SystemExit):                            # SHA 篡改 → 拒载
        cf.load_frozen_arms(path, verify_sha="0" * 64)


# ══════════════════════════════════════════════════════════════ 守卫②：locked transfer 永不 fit + picks 只读特征
def test_frozen_eval_never_fits_and_ignores_labels(monkeypatch):
    cells, actions = _toy_cells()
    L_of = _toy_L_of(cells, actions)
    order = sorted(u for cd in cells.values() for u in cd["uids"])
    data = _policy_data(cells, actions, order, L_of)
    tr = np.arange(data.n)
    arms = {"global": LookupArm(None).fit(data, tr),
            "d_lookup": LookupArm(key_cell).fit(data, tr),
            "dp_abstain": GBDTArm(("d", "p"), abstain=True, seed=3).fit(data, tr)}

    def _boom(self, *a, **k):
        raise AssertionError("A-41②：locked-transfer 路径调用了 fit")
    monkeypatch.setattr(GBDTArm, "fit", _boom)
    monkeypatch.setattr(LookupArm, "fit", _boom)
    recs = rc.frozen_eval(cells, actions, arms, L_of)          # 不得触发 fit
    assert len(recs) == data.n
    poisoned = {u: {a: v + 100.0 for a, v in d.items()} for u, d in L_of.items()}
    recs2 = rc.frozen_eval(cells, actions, arms, poisoned)     # L 污染 → picks/abstain 不变
    for r1, r2 in zip(recs, recs2):
        assert r1["arms"] == r2["arms"], "picks 读了标签（守卫失败）"


# ══════════════════════════════════════════════════════════════ 守卫⑦：full-refit checkpoint/resume ≡ 一次跑
def test_full_refit_resume_identical(tmp_path):
    cells, actions = _toy_cells()
    picks_of = {"dp_abstain": {}, "global": {}}
    for cd in cells.values():
        for u in cd["uids"]:
            picks_of["dp_abstain"][u] = "x_good" if u.startswith("cellB") else "v_median"
            picks_of["global"][u] = "v_median"
    comparisons = ["dp_abstain_vs_global"]
    ck = tmp_path / "ck.json"
    rc.full_refit_bootstrap(cells, actions, picks_of, comparisons, n_boot=3, seed=11,
                            k=3, ckpt_path=ck, progress=0)
    resumed = rc.full_refit_bootstrap(cells, actions, picks_of, comparisons, n_boot=6, seed=11,
                                      k=3, ckpt_path=ck, progress=0)
    fresh = rc.full_refit_bootstrap(cells, actions, picks_of, comparisons, n_boot=6, seed=11,
                                    k=3, ckpt_path=None, progress=0)
    assert resumed["comparisons"] == fresh["comparisons"]
    assert resumed["subgroup_dp_delta_q05"] == fresh["subgroup_dp_delta_q05"]
    assert resumed["comparisons"]["dp_abstain_vs_global"]["boot_mean"] < 0   # 功效方向对照


# ══════════════════════════════════════════════════════════════ 守卫⑤：reporter 无静默回退
def test_reporter_raises_on_bad_input():
    from SelfEvolvingHarnessTS.confirmatory_reporter import _per_series_nrmse_dlinear
    corpus_by_uid = {"u": SimpleNamespace(future=np.ones(2), obs_scale=1.0)}
    with pytest.raises(RuntimeError):                          # 窗不足 → 显式 raise 而非跳过
        _per_series_nrmse_dlinear([np.ones(6)], [dict(uid="u")], corpus_by_uid, seed=0)


# ══════════════════════════════════════════════════════════════ 守卫⑥：provenance 含模型身份/版本
def test_reporter_provenance_fields():
    from SelfEvolvingHarnessTS.confirmatory_reporter import _provenance
    prov = _provenance()
    for k in ("numpy", "sklearn", "torch", "chronos_model", "L_WIN", "H_FORECAST", "dlinear"):
        assert k in prov


# ══════════════════════════════════════════════════════════════ 守卫⑧⑨：目录/聚合分离
def test_dirs_and_aggregations_separated():
    assert rc.LT_DIR_NAME != rc.REPL_DIR_NAME
    assert rc._out_root(True) != rc._out_root(False)
    records = [dict(uid=f"u{i}", cell=("cellA" if i < 2 else "cellB"), origin="S_trend",
                    L_test={"v_median": 1.0 + i, "x_good": 0.5},
                    arms={"g": dict(pick="x_good", abstain=False)}) for i in range(4)]
    ce = rc.cell_equal_stats(records, ["v_median", "x_good"], ["g"])
    assert set(ce["g"]["per_cell"]) == {"cellA", "cellB"}      # 原分布聚合之外单列 cell-equal
    assert ce["g"]["cell_equal_mean_regret"] == pytest.approx(0.0)   # x_good 即 per-uid oracle


# ══════════════════════════════════════════════════════════════ freeze 内容完整性（A-41①字段清单）
def test_freeze_writer_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(cf, "RESULTS_CONF", tmp_path)
    monkeypatch.setattr(cf, "FREEZE_PATH", tmp_path / "confirmatory_freeze.json")
    fr = cf.write_freeze(router_sha="f" * 64, train_n=480)
    for k in ("actions", "p_feats", "d_feats", "gbdt_params", "kappa", "fallback", "router",
              "holdout", "a38c", "scope", "estimands", "measurement", "statistics", "reporter",
              "criteria", "one_shot", "versions", "code_fingerprint", "config_sha"):
        assert k in fr, f"freeze 缺字段 {k}"
    assert fr["router"]["sha256"] == "f" * 64
    assert fr["holdout"]["j_range"] == [20, 39]
    assert fr["a38c"]["n_target"] == 40
    assert "band_split_snr_db" in fr["a38c"] and fr["a38c"]["band_split_snr_db"], "带界必须预锁"
    assert "MISSING" not in set(fr["code_fingerprint"].values())
    assert (tmp_path / "confirmatory_freeze.json").exists()
