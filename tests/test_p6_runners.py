"""tests/test_p6_runners.py — P6 三执行件机械单测（materializer / c0_runner / u_runner）。

运行：D:\\Anaconda_envs\\envs\\project\\python.exe -m pytest SelfEvolvingHarnessTS/tests/test_p6_runners.py -q
（cwd = C:\\Users\\辉\\Desktop\\Agent，加 --basetemp 指到 scratchpad）

红线：全合成/注入——不联网、不读真实数据、不 import torch（真 trainer 工厂不被调用）、
无 LLM/git。文件 IO 只发生在 pytest tmp_path。
边界判据用可精确构造的值钉死 ≥/</≤ 的严格性。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from SelfEvolvingHarnessTS.data.load_real import FORECAST_PRESETS, _zscore
from SelfEvolvingHarnessTS.data.synthetic_gen import H_FORECAST
from SelfEvolvingHarnessTS.p6 import u_runner as u_mod
from SelfEvolvingHarnessTS.p6.c0_runner import (
    C0_PRESETS,
    DELTA_SAFE_MULTIPLIER,
    EPSILON_RATE,
    JUDGE_CFG_FROZEN,
    P6Episode,
    P6TechnicalStop,
    P_GATE_PROGRAM_IDS,
    RAW_PROGRAM_ID,
    build_episodes,
    compute_cutpoints,
    degradation_seed,
    evaluate_identity_gate,
    P6FrozenParamError,
    judge_ingest,
    paired_judge_fit,
    run_c0_formal,
    run_c0_unfrozen,
    spearman_rho,
)
from SelfEvolvingHarnessTS.p6.fast_path import P6PairingError
from SelfEvolvingHarnessTS.p6.harness_state import (
    P6HarnessState,
    SamplerSpec,
    SelectorSpec,
    default_state,
)
from SelfEvolvingHarnessTS.p6.judge_closed_form import (
    HORIZON,
    SeriesView,
    fit_domain,
    fit_domain_rebuild,
)
from SelfEvolvingHarnessTS.p6.materializer import (
    MIN_SERIES_LEN,
    P6TechnicalAbort,
    compute_materialization_sha,
    content_sha,
    load_materialization,
    load_materialization_bound,
    materialize,
    validate_materialization,
    write_materialization,
)
from SelfEvolvingHarnessTS.p6.split_manifest import (
    P6StateError,
    SequentialGate,
    build_manifest,
    u_sort_key,
    virgin_sort_key,
)
from SelfEvolvingHarnessTS.p6.u_runner import (
    direction_of,
    run_u_eval_formal,
    run_u_eval_unfrozen,
    train_gain_descriptor,
)
from SelfEvolvingHarnessTS.p6.loaders import BoundUEpisodes
from SelfEvolvingHarnessTS.p6.final_packet import load_final_packet

RULE_SHA = "ab" * 32


# ════════════════════════════ 共用合成 fixtures（无真实数据） ════════════════════════════
def _arr(i: int, length: int = 200) -> np.ndarray:
    rng = np.random.default_rng(1000 + i)
    t = np.arange(length, dtype=float)
    return np.sin(2 * np.pi * t / 24.0) + 0.1 * rng.standard_normal(length)


def _universe(n: int, length: int = 200, prefix: str = "item"):
    return [(f"{prefix}{i:03d}", _arr(i, length)) for i in range(n)]


def _u_spec(n: int = 5, exclusions=(), config: str = "traffic_hourly") -> dict:
    return {"block": "U", "hash_prefix": "p6u", "config": config, "n": n,
            "exclusions": sorted(str(x) for x in exclusions), "rule": "test-rule"}


def _v_spec(block: str, configs=("cfgA",), per_config_n: int = 3, exclusions=None) -> dict:
    excl = exclusions or {}
    return {
        "block": block,
        "hash_prefix": "p6v1" if block == "V1" else "p6v2",
        "configs": list(configs),
        "per_config_n": per_config_n,
        "exclusions": {c: sorted(excl.get(c, [])) for c in configs},
        "rule": "test-rule",
    }


def _mk_series_list(n_series: int, config: str, length: int = 288):
    out = []
    for i in range(n_series):
        t = np.arange(length, dtype=float)
        clean = (np.sin(2 * np.pi * t / 24.0) + 0.3 * np.sin(2 * np.pi * t / 12.0)
                 + 0.002 * t) * (1.5 + i)
        out.append((config, f"s{i}", clean))
    return out


def _degrade_stub(clean: np.ndarray, preset: str, seed: int) -> np.ndarray:
    """注入式退化：确定性（由 seed 驱动）；hi/lo 控噪声、miss 控 NaN、恒注入大离群。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(clean, float).copy()
    x = x + rng.normal(0.0, 0.05 if "hi" in preset else 0.5, x.size)
    k = max(4, x.size // 40)
    idx = rng.choice(x.size, size=k, replace=False)
    x[idx] = x[idx] + 6.0 * rng.choice([-1.0, 1.0], size=k)
    if "miss" in preset:
        gaps = rng.choice(x.size - HORIZON, size=max(3, x.size // 30), replace=False)
        x[gaps] = np.nan
    return x


def _cf_mimic_trainer(views, seed):
    """stub Adam：返回闭式判官自身的 per-episode losses（→ identity gate 应 PASS）。"""
    return fit_domain(list(views), **JUDGE_CFG_FROZEN).per_series_rmse


def _fp_stub(history):
    h = np.asarray(history, float)
    return {"snr": float(np.nanvar(h)), "missing_rate": float(np.mean(~np.isfinite(h)))}


# ══════════════════════════════════════════════════════════════════════════
# A. materializer
# ══════════════════════════════════════════════════════════════════════════
def test_materializer_u_sort_quota_exclusion():
    uni = _universe(12)
    excl = ["item001", "item005"]
    sm = materialize(_u_spec(n=5, exclusions=excl), uni, RULE_SHA)

    cands = sorted(set(i for i, _ in uni) - set(excl))
    cands.sort(key=lambda i: (u_sort_key(i), i))
    assert sm.item_ids() == cands[:5]                       # hash 升序取前 quota
    assert sm.uids() == [f"traffic_hourly:{i}" for i in cands[:5]]
    by_id = dict(uni)
    for it in sm.items:
        arr = np.ascontiguousarray(np.asarray(by_id[it["item_id"]], np.float64).ravel())
        assert it["content_sha"] == hashlib.sha256(arr.tobytes()).hexdigest()
        assert it["length"] == arr.size >= MIN_SERIES_LEN
        assert it["finite_ok"] is True
    assert sm.payload["rule_manifest_sha"] == RULE_SHA      # 链回规则 manifest
    assert len(sm.materialization_sha) == 64
    assert sm.payload["universe_size"] == 12
    assert sm.payload["n_candidates_after_exclusion"] == 10


def test_materializer_v1_v2_and_v2_excludes_v1():
    uni = _universe(12, prefix="v")
    legacy = ["v000", "v001"]
    v1 = materialize(_v_spec("V1", exclusions={"cfgA": legacy}), uni, RULE_SHA, config="cfgA")
    exp1 = sorted(set(i for i, _ in uni) - set(legacy))
    exp1.sort(key=lambda i: (virgin_sort_key("p6v1", "cfgA", i), i))
    assert v1.item_ids() == exp1[:3]

    v2 = materialize(_v_spec("V2", exclusions={"cfgA": legacy}), uni, RULE_SHA,
                     config="cfgA", v1_selected=v1.item_ids())
    exp2 = sorted(set(i for i, _ in uni) - set(legacy) - set(v1.item_ids()))
    exp2.sort(key=lambda i: (virgin_sort_key("p6v2", "cfgA", i), i))
    assert v2.item_ids() == exp2[:3]
    assert not set(v1.item_ids()) & set(v2.item_ids())      # V2 已排除 V1 已选
    assert v2.payload["v1_selected"] == sorted(v1.item_ids())
    assert set(v1.item_ids()) <= set(v2.payload["exclusions_applied"])


def test_materializer_interface_misuse_valueerror():
    uni = _universe(8)
    with pytest.raises(ValueError):                          # V1 不得传 v1_selected
        materialize(_v_spec("V1"), uni, RULE_SHA, config="cfgA", v1_selected=["x"])
    with pytest.raises(ValueError):                          # V2 必须传 v1_selected
        materialize(_v_spec("V2"), uni, RULE_SHA, config="cfgA")
    with pytest.raises(ValueError):                          # V 规格必须给 config
        materialize(_v_spec("V1"), uni, RULE_SHA)
    with pytest.raises(ValueError):                          # config 不在规格里
        materialize(_v_spec("V1"), uni, RULE_SHA, config="nope")
    with pytest.raises(ValueError):                          # U 不得传 v1_selected
        materialize(_u_spec(), uni, RULE_SHA, v1_selected=["x"])
    with pytest.raises(ValueError):                          # 未知块
        materialize({"block": "D1", "hash_prefix": "p6u"}, uni, RULE_SHA)
    with pytest.raises(ValueError):                          # rule sha 非 64-hex
        materialize(_u_spec(), uni, "deadbeef")


def test_materializer_abort_insufficient_quota():
    uni = _universe(6)
    with pytest.raises(P6TechnicalAbort, match="候选不足"):
        materialize(_u_spec(n=5, exclusions=["item000", "item001"]), uni, RULE_SHA)


def test_materializer_abort_duplicate_item_id():
    uni = _universe(5) + [("item002", _arr(99))]
    with pytest.raises(P6TechnicalAbort, match="重复"):
        materialize(_u_spec(n=3), uni, RULE_SHA)


def test_materializer_abort_nonfinite_selected_but_not_unselected():
    # 选中条目非有限 → abort（不预过滤顶替）
    uni = _universe(5)
    bad = uni[0][1].copy()
    bad[10] = np.nan
    uni[0] = (uni[0][0], bad)
    with pytest.raises(P6TechnicalAbort, match="非有限"):
        materialize(_u_spec(n=5), uni, RULE_SHA)             # quota=全体 → 必选中坏条目

    # 未被选中的坏条目不触发 abort（冻结语义：检查作用在选中条目上）
    uni2 = _universe(8)
    order = sorted((i for i, _ in uni2), key=lambda x: (u_sort_key(x), x))
    loser = order[-1]                                        # hash 序最末 → quota=3 必不选中
    uni2 = [(iid, (np.full(200, np.nan) if iid == loser else arr)) for iid, arr in uni2]
    sm = materialize(_u_spec(n=3), uni2, RULE_SHA)
    assert loser not in sm.item_ids()


def test_materializer_abort_too_short():
    uni = _universe(4) + [("shorty", _arr(50, length=MIN_SERIES_LEN - 1))]
    with pytest.raises(P6TechnicalAbort, match="长度"):
        materialize(_u_spec(n=5), uni, RULE_SHA)


def test_materializer_roundtrip_sha_and_tamper(tmp_path):
    sm = materialize(_u_spec(n=4), _universe(10), RULE_SHA)
    path = tmp_path / "sealed_u.json"
    sha = write_materialization(sm, path)
    loaded = load_materialization(path)
    assert loaded.materialization_sha == sha == sm.materialization_sha   # 往返 sha 不变
    assert loaded.payload == sm.payload

    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["items"][0]["length"] += 1                            # 篡改内容（不改内嵌 sha）
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(P6TechnicalAbort, match="hash 漂移"):
        load_materialization(path)

    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["items"][0]["length"] -= 1                            # 还原后换一种攻击：重排 + 重算 sha
    doc["items"][0], doc["items"][1] = doc["items"][1], doc["items"][0]
    doc["materialization_sha"] = compute_materialization_sha(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(P6TechnicalAbort, match="次序"):       # 冻结 hash 排序重建抓到
        load_materialization(path)


# ══════════════════════════════════════════════════════════════════════════
# B. c0_runner
# ══════════════════════════════════════════════════════════════════════════
def test_presets_and_horizon_frozen_against_sources():
    assert C0_PRESETS == tuple(FORECAST_PRESETS)              # load_real 网格单一真源
    assert HORIZON == H_FORECAST == 48                        # 判官 H 与数据语义一致


def test_degradation_seed_rule():
    s1 = degradation_seed("nn5_daily", "T7", "G_hi_full")
    digest = hashlib.sha256(b"p6deg|nn5_daily|T7|G_hi_full").hexdigest()
    assert s1 == int(digest[:8], 16)                          # prereg §2 字面规则
    assert s1 == degradation_seed("nn5_daily", "T7", "G_hi_full")   # 确定性
    assert len({degradation_seed("nn5_daily", "T7", p) for p in C0_PRESETS}) == 4


def test_build_episodes_structure_and_split():
    series = _mk_series_list(2, "dom0")
    eps = build_episodes(series, _degrade_stub)
    assert len(eps) == 2 * 4
    assert {e.preset for e in eps} == set(C0_PRESETS)
    e = next(x for x in eps if x.uid == "dom0:s0:G_lo_miss")
    assert e.series_uid == "dom0:s0"
    clean_z = _zscore(np.asarray(series[0][2], float))
    cut = clean_z.size - HORIZON
    np.testing.assert_array_equal(e.future, clean_z[cut:])    # future = z 后 clean 末 48
    expected_deg = _degrade_stub(clean_z, "G_lo_miss",
                                 degradation_seed("dom0", "s0", "G_lo_miss"))
    np.testing.assert_array_equal(e.history, expected_deg[:cut])
    assert np.isnan(e.history).any()                          # miss preset 真有 NaN
    # standardize=False：future = 原尺度 clean 末 48
    eps_raw = build_episodes(series, _degrade_stub, standardize=False)
    e2 = next(x for x in eps_raw if x.uid == "dom0:s0:G_hi_full")
    np.testing.assert_array_equal(e2.future, np.asarray(series[0][2], float)[cut:])


def test_build_episodes_aborts():
    with pytest.raises(P6TechnicalAbort, match="长度"):
        build_episodes([("d", "x", np.ones(100))], _degrade_stub)
    bad = np.ones(300)
    bad[5] = np.nan
    with pytest.raises(P6TechnicalAbort, match="非有限"):
        build_episodes([("d", "x", bad)], _degrade_stub)
    with pytest.raises(P6TechnicalAbort, match="保长"):
        build_episodes([("d", "x", _mk_series_list(1, "d")[0][2])],
                       lambda c, p, s: c[:-1])


def test_judge_ingest_fill_and_abort():
    x = np.arange(10, dtype=float)
    out = judge_ingest(x)
    np.testing.assert_array_equal(out, x)
    assert out is not x                                       # 副本
    y = x.copy()
    y[3] = np.nan
    y[7] = np.inf                                             # ±inf 同按缺失处理
    filled = judge_ingest(y)
    assert np.all(np.isfinite(filled))
    assert filled[3] == pytest.approx(3.0) and filled[7] == pytest.approx(7.0)
    lead = x.copy()
    lead[0] = np.nan                                          # 首端 → 最近值钳制
    assert judge_ingest(lead)[0] == 1.0
    with pytest.raises(P6TechnicalAbort):
        judge_ingest(np.full(8, np.nan))


def _views_for_fit(n: int = 3, length: int = 288):
    out = []
    for i in range(n):
        rng = np.random.default_rng(7 + i)
        t = np.arange(length, dtype=float)
        x = np.sin(2 * np.pi * t / 24.0) + 0.1 * rng.standard_normal(length)
        out.append(SeriesView(uid=f"u{i}", history=x[:-HORIZON], future=x[-HORIZON:]))
    return out


def test_paired_judge_fit_pass_and_tamper_abort():
    views = _views_for_fit()
    fit = paired_judge_fit(views)                             # 正常双路 → 通过并返回主路
    assert np.isfinite(fit.utility)

    def w_tampered(vs, **cfg):
        rb = fit_domain_rebuild(vs, **cfg)
        rb.W = rb.W + 1e-3                                    # W 相对差超 1e-6 → 辅助判据
        return rb

    with pytest.raises(P6TechnicalAbort, match="辅助判据"):
        paired_judge_fit(views, rebuild_fn=w_tampered)

    def loss_tampered(vs, **cfg):
        rb = fit_domain_rebuild(vs, **cfg)
        rb.per_series_rmse = rb.per_series_rmse + 5e-9        # loss 差超 1e-9 → 主判据
        return rb

    with pytest.raises(P6TechnicalAbort, match="主判据"):
        paired_judge_fit(views, rebuild_fn=loss_tampered)


def test_spearman_rho_pins():
    assert spearman_rho([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_rho([1.0, 1.0, 2.0], [5.0, 5.0, 9.0]) == pytest.approx(1.0)  # ties 均秩
    assert spearman_rho([1.0, 1.0, 1.0], [1, 2, 3]) == 0.0    # 退化 → 冻结 0.0
    with pytest.raises(ValueError):
        spearman_rho([1.0, np.nan], [1.0, 2.0])


def _table(domain_losses):
    """{d: {g: [losses]}} 便捷构造。"""
    return {d: {g: np.asarray(v, float) for g, v in row.items()}
            for d, row in domain_losses.items()}


def test_gate_criterion1_boundary():
    # 二进制精确边界：adam raw=10.0 → tol=0.1·10.0（IEEE754 下恰为 1.0）；
    # d0 diff = 11.0−10.0 = 1.0 == tol → 过（≤ 含边界）；d1 diff = 1.5 > 1.0 → 不过。
    progs, raw = ("raw", "a"), "raw"
    cf = _table({"d0": {"raw": [11.0] * 4, "a": [1.0] * 4},
                 "d1": {"raw": [11.5] * 4, "a": [1.0] * 4}})
    ad = _table({"d0": {"raw": [10.0] * 4, "a": [0.5] * 4},
                 "d1": {"raw": [10.0] * 4, "a": [0.5] * 4}})
    presets = {"d0": ["p"] * 4, "d1": ["p"] * 4}
    out = evaluate_identity_gate(cf, ad, presets, eps=0.5, programs=progs, raw_id=raw)
    c1 = out["criterion1_raw_level"]
    assert 0.1 * 10.0 == 1.0                                  # 边界构造前提（IEEE754）
    assert c1["per_domain"]["d0"]["abs_diff"] == 1.0 == c1["per_domain"]["d0"]["tol"]
    assert c1["per_domain"]["d0"]["pass"] is True             # diff == tol → 过（≤ 含边界）
    assert c1["per_domain"]["d1"]["pass"] is False            # 1.5 > 1.0
    assert c1["pass"] is False                                # per-domain 全过才过


def test_gate_criterion2_spearman_median():
    progs, raw = ("raw", "a", "b", "c"), "raw"
    agree_cf = {"raw": [1.0] * 2, "a": [0.9] * 2, "b": [0.8] * 2, "c": [0.7] * 2}
    agree_ad = {"raw": [1.0] * 2, "a": [0.85] * 2, "b": [0.75] * 2, "c": [0.6] * 2}
    invert_ad = {"raw": [1.0] * 2, "a": [1.3] * 2, "b": [1.4] * 2, "c": [1.5] * 2}
    presets = {d: ["p"] * 2 for d in ("d0", "d1", "d2", "d3")}

    cf = _table({d: agree_cf for d in presets})
    ad_all = _table({d: agree_ad for d in presets})
    out = evaluate_identity_gate(cf, ad_all, presets, eps=0.4, programs=progs, raw_id=raw)
    assert out["criterion2_spearman"]["median_rho"] == pytest.approx(1.0)
    assert out["criterion2_spearman"]["pass"] is True

    ad_half = _table({"d0": agree_ad, "d1": agree_ad, "d2": invert_ad, "d3": invert_ad})
    out2 = evaluate_identity_gate(cf, ad_half, presets, eps=0.4, programs=progs, raw_id=raw)
    assert out2["criterion2_spearman"]["per_domain_rho"]["d2"] == pytest.approx(-1.0)
    assert out2["criterion2_spearman"]["median_rho"] == pytest.approx(0.0)
    assert out2["criterion2_spearman"]["pass"] is False       # 0 < 0.7


def test_gate_criterion3_top1_top2_rule_and_boundary():
    progs, raw = ("raw", "a", "b"), "raw"
    eps = 0.4                                                 # ε/2 = 0.2
    # 5 episodes：agree/agree/disagree/disagree/agree → 3/5 = 0.6（≥0.6 边界过）
    cf_rows = [[1.0, 0.5, 0.9],    # adam top1 == cf top1
               [1.0, 0.5, 0.6],    # adam top1 = cf top2，cf 差 0.1 ≤ 0.2
               [1.0, 0.5, 0.8],    # adam top1 = cf top2，cf 差 0.3 > 0.2 → 否
               [1.0, 0.5, 0.6],    # adam top1 = raw ∉ cf top2 → 否
               [1.0, 0.5, 0.9]]
    ad_rows = [[1.0, 0.4, 0.9],
               [1.0, 0.7, 0.6],
               [1.0, 0.9, 0.6],
               [0.4, 1.0, 1.1],
               [1.0, 0.4, 0.9]]
    cf = _table({"d0": {g: [r[j] for r in cf_rows] for j, g in enumerate(progs)}})
    ad = _table({"d0": {g: [r[j] for r in ad_rows] for j, g in enumerate(progs)}})
    presets = {"d0": ["p"] * 5}
    out = evaluate_identity_gate(cf, ad, presets, eps=eps, programs=progs, raw_id=raw)
    c3 = out["criterion3_prime_top1"]
    assert (c3["n_episodes"], c3["n_agree"]) == (5, 3)
    assert c3["rate"] == pytest.approx(0.6) and c3["pass"] is True

    cf4 = _table({"d0": {g: [r[j] for r in cf_rows[:4]] for j, g in enumerate(progs)}})
    ad4 = _table({"d0": {g: [r[j] for r in ad_rows[:4]] for j, g in enumerate(progs)}})
    out2 = evaluate_identity_gate(cf4, ad4, {"d0": ["p"] * 4}, eps=eps,
                                  programs=progs, raw_id=raw)
    assert out2["criterion3_prime_top1"]["rate"] == pytest.approx(0.5)
    assert out2["criterion3_prime_top1"]["pass"] is False


def _crit4_tables(gain_map, eps=0.4):
    """gain_map: {(preset, prog): (gain_cf, gain_adam)}；raw loss 恒 1.0，两 episode/preset。"""
    progs = ("raw", "a", "b")
    presets_order = ["p", "p", "q", "q"]
    cf_row = {g: [] for g in progs}
    ad_row = {g: [] for g in progs}
    for pre in presets_order:
        for g in progs:
            g_cf, g_ad = (0.0, 0.0) if g == "raw" else gain_map[(pre, g)]
            cf_row[g].append(1.0 - g_cf)                      # loss = raw − gain
            ad_row[g].append(1.0 - g_ad)
    return (_table({"d0": cf_row}), _table({"d0": ad_row}),
            {"d0": presets_order}, progs)


def test_gate_criterion4_sign_rate_boundary():
    # 4 个带外程序对，3 同号 → rate 0.75（≥0.75 边界过）；raw 两对 gain=0 → 带内剔除
    cf, ad, presets, progs = _crit4_tables({
        ("p", "a"): (0.2, 0.2), ("p", "b"): (0.2, 0.3),
        ("q", "a"): (-0.2, -0.15), ("q", "b"): (0.2, -0.2),
    })
    out = evaluate_identity_gate(cf, ad, presets, eps=0.4, programs=progs, raw_id="raw")
    c4 = out["criterion4_preset_sign"]
    assert (c4["n_pairs_total"], c4["n_excluded_near_zero"]) == (6, 2)
    assert (c4["n_evaluated"], c4["n_agree"]) == (4, 3)
    assert c4["rate"] == pytest.approx(0.75) and c4["pass"] is True


def test_gate_criterion4_near_zero_band_strictness_and_empty():
    # band = ε/4 = 0.5/4 = 0.125（二进制精确），严格 <：|gain| == 0.125 保留；任一侧 < 0.125 剔除
    # 全部 gain 用 2 的负幂 → loss = 1.0 − g 与 gain 重建都是精确浮点
    cf, ad, presets, progs = _crit4_tables({
        ("p", "a"): (0.125, 0.25),    # cf 恰在带界 → 保留、同号
        ("p", "b"): (0.0625, 0.5),    # cf 侧落带 → 剔除
        ("q", "a"): (0.5, 0.0625),    # adam 侧落带 → 剔除
        ("q", "b"): (-0.25, -0.25),   # 保留、同号
    })
    out = evaluate_identity_gate(cf, ad, presets, eps=0.5, programs=progs, raw_id="raw")
    c4 = out["criterion4_preset_sign"]
    assert (c4["n_evaluated"], c4["n_agree"]) == (2, 2)
    assert c4["n_excluded_near_zero"] == 4                    # raw×2 + 两个近零对
    assert c4["pass"] is True

    cf0, ad0, presets0, progs0 = _crit4_tables({
        (pre, g): (0.01, 0.01) for pre in ("p", "q") for g in ("a", "b")
    })
    out0 = evaluate_identity_gate(cf0, ad0, presets0, eps=0.4, programs=progs0, raw_id="raw")
    c40 = out0["criterion4_preset_sign"]
    assert c40["n_evaluated"] == 0 and c40["rate"] is None
    assert c40["pass"] is False                               # 分母空 → 无法建立一致 → FAIL


def test_gate_input_validation():
    cf = _table({"d0": {"raw": [1.0], "a": [0.9]}})
    ad = _table({"d0": {"raw": [1.0], "a": [0.9]}})
    with pytest.raises(ValueError):                           # ε 必须先算好且为正
        evaluate_identity_gate(cf, ad, {"d0": ["p"]}, eps=0.0, programs=("raw", "a"))
    with pytest.raises(ValueError):                           # 缺程序
        evaluate_identity_gate(cf, ad, {"d0": ["p"]}, eps=0.1, programs=("raw", "a", "b"))
    with pytest.raises(ValueError):                           # raw 必须在清单内
        evaluate_identity_gate(cf, ad, {"d0": ["p"]}, eps=0.1, programs=("a",), raw_id="raw")


@pytest.fixture(scope="module")
def c0_episodes():
    return build_episodes(_mk_series_list(2, "dom0") + _mk_series_list(2, "dom1"),
                          _degrade_stub)


def test_run_c0_smoke_pass_epsilon_cutpoints_costs(c0_episodes, tmp_path):
    out_path = tmp_path / "C0_FREEZE.json"
    rec = run_c0_unfrozen(c0_episodes, _cf_mimic_trainer, _fp_stub, out_path=out_path)

    # gate：stub Adam ≡ 闭式 → 四判据全过
    assert rec["identity_gate"]["pass"] is True
    assert rec["identity_gate"]["criterion2_spearman"]["median_rho"] == pytest.approx(1.0)
    assert rec["identity_gate"]["criterion3_prime_top1"]["rate"] == pytest.approx(1.0)
    assert rec["identity_gate"]["criterion4_preset_sign"]["n_evaluated"] > 0

    # ε/δ 算术（§3.2）：域等权 J_raw、ε=0.02J、δ=2.5ε
    j_manual = np.mean([rec["per_domain_utilities"]["closed_form"][d][RAW_PROGRAM_ID]
                        for d in rec["domains"]])
    assert rec["j_raw_c0"] == pytest.approx(j_manual)
    assert rec["epsilon"] == EPSILON_RATE * rec["j_raw_c0"]
    assert rec["delta_safe"] == DELTA_SAFE_MULTIPLIER * rec["epsilon"]

    # cutpoints（§3.3）：与 np.quantile 四分位逐位一致
    snr_vals = [_fp_stub(ep.history)["snr"] for ep in c0_episodes]
    miss_vals = [_fp_stub(ep.history)["missing_rate"] for ep in c0_episodes]
    assert rec["p0_cutpoints"]["snr"] == [float(v) for v in
                                          np.quantile(snr_vals, [0.25, 0.5, 0.75])]
    assert rec["p0_cutpoints"]["missing_rate"] == [float(v) for v in
                                                   np.quantile(miss_vals, [0.25, 0.5, 0.75])]

    # 成本账：2 域 × 8 程序闭式；×3 seeds Adam
    assert rec["costs"]["closed_form_fits"] == 2 * len(P_GATE_PROGRAM_IDS)
    assert rec["costs"]["adam_fits"] == 2 * len(P_GATE_PROGRAM_IDS) * 3
    assert rec["costs"]["wall_clock_seconds"] >= 0.0

    # 落盘 + record_sha
    doc = json.loads(out_path.read_text(encoding="utf-8"))
    assert doc["identity_gate"]["pass"] is True and len(doc["record_sha"]) == 64


def test_run_c0_gate_fail_records_then_technical_stop(c0_episodes, tmp_path):
    def adversarial(views, seed):
        # 跨程序共用常数 K=5.0 反射：程序均值 m_g → 5−m_g，gain_adam = −gain_cf
        # → ρ = −1（②崩）；水准/argmin/符号同崩（①③′④）。
        base = fit_domain(list(views), **JUDGE_CFG_FROZEN).per_series_rmse
        return 5.0 - base

    out_path = tmp_path / "C0_FREEZE_fail.json"
    with pytest.raises(P6TechnicalStop) as exc:
        run_c0_unfrozen(c0_episodes, adversarial, _fp_stub, out_path=out_path)
    rec = exc.value.record
    assert rec is not None and rec["identity_gate"]["pass"] is False
    assert rec["identity_gate"]["criterion2_spearman"]["pass"] is False
    assert rec["identity_gate"]["criterion2_spearman"]["median_rho"] == pytest.approx(-1.0)
    doc = json.loads(out_path.read_text(encoding="utf-8"))    # FAIL 也先写盘（记录后 stop）
    assert doc["identity_gate"]["pass"] is False


def test_run_c0_adam_nan_abort(c0_episodes):
    def nan_trainer(views, seed):
        out = np.ones(len(views))
        out[0] = np.nan
        return out

    with pytest.raises(P6TechnicalAbort, match="NaN"):
        run_c0_unfrozen(c0_episodes, nan_trainer, _fp_stub)


def test_run_c0_timeout_abort(c0_episodes):
    import time as _t

    def slow_trainer(views, seed):
        _t.sleep(0.01)
        return np.ones(len(views))

    with pytest.raises(P6TechnicalAbort, match="超时"):
        run_c0_unfrozen(c0_episodes, slow_trainer, _fp_stub, adam_fit_timeout_seconds=0.0)


def test_run_c0_dual_path_tamper_abort(c0_episodes):
    def w_tampered(vs, **cfg):
        rb = fit_domain_rebuild(vs, **cfg)
        rb.W = rb.W + 1e-3
        return rb

    with pytest.raises(P6TechnicalAbort, match="对拍超限"):
        run_c0_unfrozen(c0_episodes, _cf_mimic_trainer, _fp_stub, rebuild_fn=w_tampered)


def test_compute_cutpoints_validation(c0_episodes):
    with pytest.raises(ValueError, match="缺键"):
        compute_cutpoints(c0_episodes, lambda h: {"snr": 1.0})
    with pytest.raises(P6TechnicalAbort, match="非有限"):
        compute_cutpoints(c0_episodes, lambda h: {"snr": np.nan, "missing_rate": 0.0})


# ══════════════════════════════════════════════════════════════════════════
# B'. F2：manifest-bound loader + content_sha 可复算校验（finding 32）
# ══════════════════════════════════════════════════════════════════════════
def test_f2_content_sha_recompute_and_manifest_bound_loader():
    manifest = _mini_manifest_u(u_n=3, u_excluded=("Z1", "Z2"))
    universe = _u_universe(6)
    spec = manifest.payload["virgin_specs"]["U"]
    sm = materialize(spec, universe, manifest.manifest_sha)
    cfg = spec["config"]
    series_by_uid = {f"{cfg}:{iid}": arr for iid, arr in universe}

    # 可复算校验：给数据即逐条 content_sha 重算比对（口径 = float64 字节 sha256）
    validate_materialization(sm, series_by_uid)
    for it in sm.items:
        assert it["content_sha"] == content_sha(series_by_uid[it["uid"]])

    # manifest-bound loader 全绑定通过
    bound = load_materialization_bound(
        sm, manifest, series_by_uid=series_by_uid,
        expected_materialization_sha=sm.materialization_sha)
    assert bound.materialization_sha == sm.materialization_sha

    # 篡改数据 → content_sha 复算不一致 → abort
    tampered = dict(series_by_uid)
    tampered[sm.uids()[0]] = tampered[sm.uids()[0]] + 1.0
    with pytest.raises(P6TechnicalAbort, match="content_sha 复算"):
        load_materialization_bound(sm, manifest, series_by_uid=tampered)

    # rule_manifest_sha 必须 == 传入 selection manifest 内容 sha（不再只是 64-hex）
    other = _mini_manifest_u(u_n=3, u_excluded=("Z1", "Z2", "Z3"))     # 不同排除 → 不同 sha
    assert other.manifest_sha != manifest.manifest_sha
    with pytest.raises(P6TechnicalAbort, match="rule_manifest_sha"):
        load_materialization_bound(sm, other)

    # expected_materialization_sha（precommit 绑定）漂移 → abort
    with pytest.raises(P6TechnicalAbort, match="materialization_sha 与绑定值"):
        load_materialization_bound(sm, manifest, expected_materialization_sha="dead" * 16)


def test_f2_bound_loader_from_path_roundtrip(tmp_path):
    manifest = _mini_manifest_u(u_n=3)
    universe = _u_universe(6)
    spec = manifest.payload["virgin_specs"]["U"]
    sm = materialize(spec, universe, manifest.manifest_sha)
    path = tmp_path / "u_mat.json"
    write_materialization(sm, path)
    # 从路径加载并绑定核验（内嵌 sha + rule 绑定）
    bound = load_materialization_bound(path, manifest,
                                       expected_materialization_sha=sm.materialization_sha)
    assert bound.item_ids() == sm.item_ids()


# ══════════════════════════════════════════════════════════════════════════
# C. u_runner
# ══════════════════════════════════════════════════════════════════════════
class _GateStub:
    def __init__(self, allow: bool):
        self.allow = allow
        self.calls = []                     # can_open 查询日志
        self.opened = []                    # open_block 记账日志（F5）

    def can_open(self, block):
        self.calls.append(block)
        return self.allow

    def open_block(self, block, bindings=None):
        self.opened.append((block, dict(bindings or {})))
        return f"evt_{block}"


_U_BINDINGS = {"materialization_sha": "m" * 64}     # F5：U open 必含 materialization sha


# ── F2：manifest-bound loader / content_sha 复算 ─────────────────────────────
def _mini_manifest_u(u_n=3, u_excluded=("Z1", "Z2")):
    big = ("nn5_daily", "fred_md", "tourism_monthly", "covid_deaths")
    single = ("us_births", "saugeenday", "sunspot")
    rows = [{"config": c, "item_id": f"T{i}", "series_uid": f"{c}:T{i}",
             "exposure_class": "confirmed_exposed"} for c in big for i in range(1, 21)]
    rows += [{"config": c, "item_id": "T1", "series_uid": f"{c}:T1",
              "exposure_class": "confirmed_exposed"} for c in single]
    return build_manifest(rows, list(u_excluded), u_n=u_n)


def _u_universe(n=6):
    return [(f"H{i}", np.arange(160, dtype=float) + float(i)) for i in range(n)]


def _mini_manifest():
    """F5 真实 gate 用最小 manifest（与 test_p6_cycle 同构；U 排除任意 24 条）。"""
    big = ("nn5_daily", "fred_md", "tourism_monthly", "covid_deaths")
    single = ("us_births", "saugeenday", "sunspot")
    rows = [{"config": c, "item_id": f"T{i}", "series_uid": f"{c}:T{i}",
             "exposure_class": "confirmed_exposed"} for c in big for i in range(1, 21)]
    rows += [{"config": c, "item_id": "T1", "series_uid": f"{c}:T1",
              "exposure_class": "confirmed_exposed"} for c in single]
    return build_manifest(rows, [f"T{i}" for i in range(1, 25)])


class _CountTrainer:
    def __init__(self, bump: float = 0.0):
        self.calls = []
        self.bump = bump

    def __call__(self, views, seed):
        self.calls.append((tuple(v.uid for v in views), int(seed)))
        return np.array(
            [abs(float(np.mean(np.asarray(v.history)))) + 0.5 + self.bump + 0.001 * seed
             for v in views], dtype=float)


@pytest.fixture(scope="module")
def u_episodes():
    return build_episodes(_mk_series_list(3, "traffic_hourly"), _degrade_stub)


def _u_states():
    h0 = default_state()
    final = P6HarnessState(
        version="v1",
        selector=SelectorSpec("weighted_features", {"proxy_score": 1.0, "n_steps": -0.05}),
        sampler=SamplerSpec(allocation={"det": 3, "random": 5, "llm": 0}, expected_total=8),
        risk_rules=(),
    )
    return h0, final


U_JUDGE_CFG = {"delta_safe": 0.05}


class _ULoader:
    """零参 U loader（G3）：返回 BoundUEpisodes，记调用次数。"""
    def __init__(self, bound):
        self.bound = bound
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.bound


def _u_bound(u_episodes, config="traffic_hourly"):
    """从 u_episodes 构造一致的 BoundUEpisodes + bindings（materialization 覆盖其 series_uid、
    config 一致、content_sha 自洽）。返回 (bound, bindings)。"""
    big = ("nn5_daily", "fred_md", "tourism_monthly", "covid_deaths")
    single = ("us_births", "saugeenday", "sunspot")
    rows = [{"config": c, "item_id": f"T{i}", "series_uid": f"{c}:T{i}",
             "exposure_class": "confirmed_exposed"} for c in big for i in range(1, 21)]
    rows += [{"config": c, "item_id": "T1", "series_uid": f"{c}:T1",
              "exposure_class": "confirmed_exposed"} for c in single]
    item_ids = sorted({ep.series_uid.split(":", 1)[1] for ep in u_episodes})
    manifest = build_manifest(rows, ["Z1", "Z2"], u_config=config, u_n=len(item_ids))
    universe = [(iid, np.arange(160, dtype=float) + float(j)) for j, iid in enumerate(item_ids)]
    sm = materialize(manifest.payload["virgin_specs"]["U"], universe, manifest.manifest_sha)
    series_by_uid = {f"{config}:{iid}": arr for iid, arr in universe}
    bound = BoundUEpisodes(tuple(u_episodes), sm, series_by_uid)
    return bound, {"materialization_sha": sm.materialization_sha}


def test_u_can_open_false_raises_and_gate_untouched(u_episodes):
    h0, final = _u_states()
    gate = _GateStub(allow=False)
    adam, lstm = _CountTrainer(), _CountTrainer(0.1)
    bound, binds = _u_bound(u_episodes)
    loader = _ULoader(bound)
    with pytest.raises(P6StateError, match="can_open"):
        run_u_eval_unfrozen(loader, h0, final, U_JUDGE_CFG, adam, lstm, gate, bindings=binds)
    assert gate.calls == ["U"]                                # 只查询、不记账
    assert gate.opened == [] and loader.calls == 0            # can_open False → 未开箱、未加载
    assert adam.calls == [] and lstm.calls == []              # 拦截在任何拟合之前


def test_u_full_report_disclosure_and_trainer_call_counts(u_episodes):
    h0, final = _u_states()
    gate = _GateStub(allow=True)
    adam, lstm = _CountTrainer(), _CountTrainer(0.1)
    bound, binds = _u_bound(u_episodes)
    loader = _ULoader(bound)
    rep = run_u_eval_unfrozen(loader, h0, final, U_JUDGE_CFG, adam, lstm, gate, bootstrap_b=200,
                              bindings=binds)

    assert loader.calls == 1                                  # 延迟加载：open 之后恰调一次
    # trainer 注入调用次数 = 3 seeds × 2 臂（每模型）
    assert len(adam.calls) == 6 and len(lstm.calls) == 6
    assert [s for _, s in adam.calls] == [0, 0, 1, 1, 2, 2]   # 每 seed 两臂配对
    assert rep["costs"]["adam_fits"] == 6 and rep["costs"]["lstm_fits"] == 6
    assert rep["costs"]["closed_form_fits"] == 2

    # 三效应披露完整性（train/context/joint × overall + 4 preset + 方向 + LCB）
    assert set(rep["judge_effects"]) == {"train", "context", "joint"}
    for block in rep["judge_effects"].values():
        assert {"overall_gain", "lcb90", "direction", "per_preset"} <= set(block)
        assert set(block["per_preset"]) == set(C0_PRESETS)
        for p in block["per_preset"].values():
            assert {"gain", "lcb90", "n", "direction"} <= set(p)
            assert p["n"] == 3                                 # 3 底层 series/preset

    # 结果表配套：per-episode 全披露、pool_stats（realized pool size 落账）、Adam/LSTM 数字
    assert len(rep["per_episode"]) == len(u_episodes)
    assert {"loss_00", "loss_10", "loss_01", "loss_11",
            "train_gain", "context_gain", "joint_gain"} <= set(rep["per_episode"][0])
    for arm in ("h0", "final"):
        stats = rep["arms"][arm]["pool_stats"]
        assert set(stats) == {ep.uid for ep in u_episodes}
        assert all("realized_pool_size" in s for s in stats.values())
    assert len(rep["adam"]["per_seed"]) == 3 and len(rep["lstm"]["per_seed"]) == 3
    assert set(rep["adam"]["per_preset_gain"]) == set(C0_PRESETS)

    # 非门：报告无任何 verdict；描述子字段齐备
    assert "verdict" not in rep and "非门" in rep["non_gate_note"]
    d = rep["success_descriptor"]
    assert {"train_gain", "train_gain_lcb90", "delta_safe",
            "non_harm", "direction_positive", "success"} <= set(d)
    assert d["delta_safe"] == 0.05
    assert rep["protocol"]["state_h0_sha"] != rep["protocol"]["state_final_sha"]
    assert rep["provenance"]["entrypoint"] == "run_u_eval_unfrozen"
    assert rep["provenance"]["materialization_sha"] == binds["materialization_sha"]
    assert gate.calls == ["U"]
    # G4：open bindings 含 materialization_sha + entrypoint（台账记账）
    assert gate.opened[0][0] == "U"
    assert gate.opened[0][1]["materialization_sha"] == binds["materialization_sha"]
    assert gate.opened[0][1]["entrypoint"] == "run_u_eval_unfrozen"


def test_u_identical_states_three_effects_exact_zero(u_episodes):
    h0 = default_state()
    bound, binds = _u_bound(u_episodes)
    rep = run_u_eval_unfrozen(_ULoader(bound), h0, h0, U_JUDGE_CFG, _CountTrainer(),
                              _CountTrainer(), _GateStub(True), bootstrap_b=100, bindings=binds)
    for name in ("train", "context", "joint"):
        assert rep["judge_effects"][name]["overall_gain"] == 0.0     # 恒等替换严格 0
        assert rep["judge_effects"][name]["direction"] == "zero"
    assert all(row["train_gain"] == 0.0 for row in rep["per_episode"])
    d = rep["success_descriptor"]
    assert d["train_gain_lcb90"] == 0.0
    assert d["non_harm"] is True                              # 0 ≥ −δ
    assert d["direction_positive"] is False                   # 0 不是正向（严格 >）
    assert d["success"] is False


def test_u_missing_delta_safe_and_k_mismatch(u_episodes):
    h0, final = _u_states()
    bound, binds = _u_bound(u_episodes)
    with pytest.raises(ValueError, match="delta_safe"):       # delta_safe 检查在 open 之前
        run_u_eval_unfrozen(_ULoader(bound), h0, final, {}, _CountTrainer(), _CountTrainer(),
                            _GateStub(True), bindings=binds)
    k7 = P6HarnessState(
        version="v0", selector=SelectorSpec(),
        sampler=SamplerSpec(allocation={"det": 3, "random": 4, "llm": 0}, expected_total=7),
    )
    with pytest.raises(P6PairingError, match="K 不一致"):        # K 检查在 open 之前
        run_u_eval_unfrozen(_ULoader(bound), h0, k7, U_JUDGE_CFG, _CountTrainer(),
                            _CountTrainer(), _GateStub(True), bindings=binds)


def test_u_prepared_failure_abort(u_episodes, monkeypatch):
    h0, final = _u_states()
    bound, binds = _u_bound(u_episodes)
    monkeypatch.setattr(u_mod, "prepared_artifact", lambda chosen, series: None)
    with pytest.raises(P6TechnicalAbort, match="prepared"):
        run_u_eval_unfrozen(_ULoader(bound), h0, final, U_JUDGE_CFG, _CountTrainer(),
                            _CountTrainer(), _GateStub(True), bindings=binds)


def test_f5_empty_bindings_rejected(u_episodes):
    """F5/finding 35：空 / 缺 materialization_sha 的 bindings → 拒绝开箱（不读 U）。"""
    h0, final = _u_states()
    bound, _binds = _u_bound(u_episodes)
    for bad in (None, {}, {"materialization_sha": ""}, {"materialization_sha": "   "},
                {"other": "x"}):
        gate = _GateStub(True)
        loader = _ULoader(bound)
        with pytest.raises(P6StateError, match="materialization_sha"):
            run_u_eval_unfrozen(loader, h0, final, U_JUDGE_CFG, _CountTrainer(),
                                _CountTrainer(), gate, bootstrap_b=50, bindings=bad)
        assert gate.opened == [] and loader.calls == 0        # 未开箱、未加载（无重复窥视风险）


def test_f5_real_gate_one_shot_no_repeat_peek(u_episodes, tmp_path):
    """F5/finding 35：真实台账下 U 一次性 open——runner 自己开箱、绑 materialization sha；
    第二次调用因 gate 已开箱而被拒（can_open False）→ 无法重复窥视。"""
    manifest = _mini_manifest()
    ledger = tmp_path / "ledger"
    ledger.mkdir()                                            # canonical 台账落此目录
    h0, final = _u_states()
    bound, binds = _u_bound(u_episodes)
    with SequentialGate(manifest, ledger) as gate:
        gate.record_cycle_terminal(1, "abstain")              # 驱动到 cycle2 terminal
        gate.record_cycle_terminal(2, "abstain")
        assert gate.can_open("U") is True
        tip_before = gate.chain_tip
        rep = run_u_eval_unfrozen(_ULoader(bound), h0, final, U_JUDGE_CFG, _CountTrainer(),
                                  _CountTrainer(), gate, bootstrap_b=50, bindings=binds)
        assert "verdict" not in rep                           # 仍非门
        assert gate.state("U") == "open"                      # runner 已开箱
        assert gate.can_open("U") is False                    # 一次性：不能再开
        assert gate.chain_tip != tip_before                   # open 事件已记账（hash 链推进）
        # open 事件绑定 materialization sha（读回台账 pending open）
        assert gate.pending_open("U")["bindings"]["materialization_sha"] == binds["materialization_sha"]
        # 第二次调用被拒 → 无法重复窥视
        with pytest.raises(P6StateError, match="can_open"):
            run_u_eval_unfrozen(_ULoader(bound), h0, final, U_JUDGE_CFG, _CountTrainer(),
                                _CountTrainer(), gate, bootstrap_b=50, bindings=binds)


def test_train_gain_descriptor_logic_and_boundaries():
    clusters = ["s0", "s0", "s1", "s1"]
    good = train_gain_descriptor([0.1] * 4, clusters, 0.1, 0.05, b=100)
    assert good["success"] is True and good["non_harm"] is True
    assert good["direction_positive"] is True
    assert good["train_gain_lcb90"] == pytest.approx(0.1)     # 常数 → bootstrap 均值恒等

    harm = train_gain_descriptor([-0.1] * 4, clusters, -0.1, 0.05, b=100)
    assert harm["non_harm"] is False and harm["success"] is False

    edge = train_gain_descriptor([-0.05] * 4, clusters, -0.05, 0.05, b=100)
    assert edge["train_gain_lcb90"] == -0.05
    assert edge["non_harm"] is True                           # LCB ≥ −δ 含边界
    assert edge["direction_positive"] is False and edge["success"] is False

    with pytest.raises(ValueError):
        train_gain_descriptor([0.1], ["s0"], 0.1, -1.0, b=10)


# ══════════════════════════════════════════════════════════════════════════
# G 波：正式入口收口（codex 三轮复审 finding 32/34/35/36 最小再送审条件）
# ══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def c0_episodes_64():
    """冻结结构 C0（16 series × 4 preset = 64 episode，4 域）——run_c0_formal 正例。"""
    series = []
    for d in ("dom0", "dom1", "dom2", "dom3"):
        series += _mk_series_list(4, d)
    return build_episodes(series, _degrade_stub)


def test_g3_u_loader_must_return_bound_u(u_episodes, tmp_path):
    """G3/finding 35：U loader 返回非 BoundUEpisodes（裸 list）→ P6TechnicalAbort；
    U 已 open（事故如实留台账，特性非缺陷）。"""
    manifest = _mini_manifest()
    ledger = tmp_path / "led"
    ledger.mkdir()
    h0, final = _u_states()
    _bound, binds = _u_bound(u_episodes)
    with SequentialGate(manifest, ledger) as gate:
        gate.record_cycle_terminal(1, "abstain")
        gate.record_cycle_terminal(2, "abstain")
        with pytest.raises(P6TechnicalAbort, match="BoundUEpisodes"):
            run_u_eval_unfrozen(lambda: list(u_episodes), h0, final, U_JUDGE_CFG,
                                _CountTrainer(), _CountTrainer(), gate, bindings=binds)
        assert gate.state("U") == "open"                     # U 已 open（事故留台账）


def test_g3_old_episodes_u_signature_gone():
    """G3/finding 35：旧 run_u_eval（episodes_u 预加载签名）不再存在。"""
    assert not hasattr(u_mod, "run_u_eval")


def test_g3_u_materialization_sha_mismatch_aborts(u_episodes):
    """G3：loaded materialization 实际 sha ≠ open 绑定值 → abort。"""
    bound, _binds = _u_bound(u_episodes)
    h0, final = _u_states()
    with pytest.raises(P6TechnicalAbort, match="物化实际 sha"):
        run_u_eval_unfrozen(_ULoader(bound), h0, final, U_JUDGE_CFG, _CountTrainer(),
                            _CountTrainer(), _GateStub(True),
                            bindings={"materialization_sha": "d" * 64})


def test_g4_formal_asserts_include_timeout(c0_episodes_64):
    """G4/finding 36：三入口冻结断言均含 trainer 超时=900（本轮补入）。"""
    from SelfEvolvingHarnessTS.p6.c0_runner import assert_c0_frozen_params
    from SelfEvolvingHarnessTS.p6.cycle_runner import assert_cycle_frozen_params
    with pytest.raises(P6FrozenParamError, match="900"):
        assert_cycle_frozen_params((0, 1, 2), 2000, 20260712, 1, 8, 120.0)
    with pytest.raises(P6FrozenParamError, match="900"):
        u_mod.assert_u_frozen_params((0, 1, 2), 2000, 20260714, 8, 120.0)
    with pytest.raises(P6FrozenParamError, match="900"):
        assert_c0_frozen_params(c0_episodes_64, (0, 1, 2), 120.0)


def test_g4_c0_formal_is_gated_and_records_entrypoint(c0_episodes_64, tmp_path):
    """G4/finding 36：run_c0_formal 对非冻结结构 raise；冻结结构下记 entrypoint + digest。"""
    small = build_episodes(_mk_series_list(2, "dom0"), _degrade_stub)   # 8 episode ≠ 64
    with pytest.raises(P6FrozenParamError, match="episodes"):
        run_c0_formal(small, _cf_mimic_trainer, _fp_stub)
    rec = run_c0_formal(c0_episodes_64, _cf_mimic_trainer, _fp_stub)
    assert rec["provenance"]["entrypoint"] == "run_c0_formal"
    assert len(rec["provenance"]["frozen_literals_digest"]) == 64


def test_g4_u_formal_asserts_and_records_entrypoint(u_episodes, tmp_path):
    """G4/finding 36：run_u_eval_formal 断言冻结字面量（漂移 raise、open 前）+ 记 entrypoint。"""
    manifest = _mini_manifest()
    ledger = tmp_path / "l"
    ledger.mkdir()
    h0, final = _u_states()
    bound, binds = _u_bound(u_episodes)
    with SequentialGate(manifest, ledger) as gate:
        gate.record_cycle_terminal(1, "abstain")
        gate.record_cycle_terminal(2, "abstain")
        with pytest.raises(P6FrozenParamError, match="bootstrap_b"):
            run_u_eval_formal(_ULoader(bound), h0, final, U_JUDGE_CFG, _CountTrainer(),
                              _CountTrainer(), gate, bindings=binds, bootstrap_b=100)
        assert gate.state("U") == "sealed"                   # 漂移在 open 之前 → 未开箱
        rep = run_u_eval_formal(_ULoader(bound), h0, final, U_JUDGE_CFG, _CountTrainer(),
                                _CountTrainer(), gate, bindings=binds)
        assert rep["provenance"]["entrypoint"] == "run_u_eval_formal"
        assert len(rep["provenance"]["frozen_literals_digest"]) == 64
        assert gate.pending_open("U")["bindings"]["entrypoint"] == "run_u_eval_formal"


def test_g5_formal_u_writes_final_packet(u_episodes, tmp_path):
    """G5/backlog 41：run_u_eval_formal verdict 落账后外锚 final packet；篡改可检。"""
    manifest = _mini_manifest()
    ledger = tmp_path / "l"
    ledger.mkdir()
    h0, final = _u_states()
    bound, binds = _u_bound(u_episodes)
    packet_path = tmp_path / "final_packet.json"
    with SequentialGate(manifest, ledger) as gate:
        gate.record_cycle_terminal(1, "abstain")
        gate.record_cycle_terminal(2, "abstain")
        rep = run_u_eval_formal(
            _ULoader(bound), h0, final, U_JUDGE_CFG, _CountTrainer(), _CountTrainer(), gate,
            bindings=binds, final_packet_path=packet_path, freeze_shas={"prereg": "a" * 64},
            selection_manifest_sha=manifest.manifest_sha, claim_branch="B-null",
        )
        tip = gate.chain_tip
    assert rep["final_packet"]["path"] == str(packet_path)
    loaded = load_final_packet(packet_path)
    assert loaded["claim_branch"] == "B-null"
    assert loaded["materialization_sha"] == binds["materialization_sha"]
    assert loaded["selection_manifest_sha"] == manifest.manifest_sha
    assert loaded["ledger_chain_tip"] == tip                 # 外锚台账 chain_tip
    # 篡改 → packet_sha256 校验失败
    doc = json.loads(packet_path.read_text(encoding="utf-8"))
    doc["claim_branch"] = "B-strong"
    packet_path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="packet_sha256"):
        load_final_packet(packet_path)


def test_direction_of_pins():
    assert direction_of(0.25) == "positive"
    assert direction_of(-0.25) == "negative"
    assert direction_of(0.0) == "zero"
