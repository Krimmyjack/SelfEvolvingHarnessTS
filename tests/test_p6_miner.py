"""tests/test_p6_miner.py — P6 冻结 miner（p6/miner.py；prereg §4「miner = 冻结代码」）。

运行：D:\\Anaconda_envs\\envs\\project\\python.exe -m pytest SelfEvolvingHarnessTS/tests/test_p6_miner.py -q
（cwd = C:\\Users\\辉\\Desktop\\Agent，--basetemp 指向 scratchpad）

覆盖（全合成 fixture、全确定性、无 RNG/IO/网络）：
  Selector：(a) 手工小样本 ridge 对照；(b) 秩目标手工值 + 秩方向写死（gain 大者秩大）+
  ties 均秩；(c) 软阈值公式逐坐标对照 + z-standardize 常量必须取自 full-D；fold 划分公式
  稳定；12 位舍入与 −0.0 归一；退化去重（<3 合法）；单 fold → (c) 缺席；非 KNOWN 特征忽略。
  Sampler：三字面配方；负配额不可用；全不可用 → []；random_params/expected_total 原样保留。
  Risk：(a) bin/preset 两种 cohort 形；(b) 算子族映射与第一命中；(c) ≥80% 整数边界
  （恰 80% 满足）、ALLOWLIST_ORDER 扫描序、中段 bin 双条件、bin 左闭（== cutpoint 归上位）；
  三配方全部产出**单规则、when=条件列表**（同规则 AND，prereg §4 冻结——不再有 _and{i}
  拆分与 rule_groups/conjunction_note），每条可被 compile_proposal 编译并 apply_edit；
  不可用路径与坏 evidence。
  通用：≤3、确定性重放、tie-break 键、provenance 可复算且稳定、family 别名 S1/S2/S3。
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from SelfEvolvingHarnessTS.p6.edit_surfaces import (
    RiskRulePatch,
    SamplerPatch,
    SelectorPatch,
    compile_proposal,
)
from SelfEvolvingHarnessTS.p6.harness_state import (
    H0_ALLOCATION,
    KNOWN_FEATURES,
    P0_FEATURE_ALLOWLIST,
    P6HarnessState,
    P_FEATS_FROZEN,
    SamplerSpec,
    apply_edit,
    canonical_json,
    default_state,
)
from SelfEvolvingHarnessTS.p6.miner import (
    ALLOWLIST_ORDER,
    FAMILY_ALIASES,
    FAMILY_RECIPES,
    MinedCandidate,
    OP_FAMILY_MAP,
    OP_FAMILY_MEMBERS,
    candidate_sort_key,
    fold_of,
    mine,
)

H0 = default_state()


# ============================== fixture 工具（全确定性） ==============================
def _row(uid: str, proxy: float, gain: float, mf: float = 0.5, **extra) -> dict:
    feats = {"proxy_score": proxy, "n_steps": 2.0, "has_guard": 0.0,
             "modified_fraction": mf, "exec_ok": 1.0}      # exec_ok = 真实池会出现的非 KNOWN 键
    feats.update(extra)
    return {"episode_uid": uid, "features": feats, "train_gain": gain}


def _uid_in_fold(fold: int, skip: int = 0) -> str:
    """确定性地找一个落在指定 fold 的 uid（用与 miner 相同的冻结公式验证）。"""
    found = 0
    for i in range(200):
        u = f"u{i}"
        if fold_of(u) == fold:
            if found == skip:
                return u
            found += 1
    raise AssertionError("unreachable")


# 手工小样本（(a)/(b) 闭式可算）：两 episode、proxy [1,3]、gain [0,2]；
# 其余特征常量 → z 列全 0 → 权重恰 0；proxy 列 z=[-1,1,-1,1]（mu=2, sd=1）。
_HAND_ROWS = {"rows": [_row("e0", 1.0, 0.0), _row("e0", 3.0, 2.0),
                       _row("e1", 1.0, 0.0), _row("e1", 3.0, 2.0)]}

# (c) 用跨 fold 构造：proxy-gain 斜率两 fold 一致（存活）、mf-gain 斜率翻号（收缩到 0）；
# 两 fold 的 proxy 均值差距大（0/2 vs 10/12）→ per-fold 标准化常量与 full-D 显著不同。
_U0 = _uid_in_fold(0)
_U1 = _uid_in_fold(1)
_CROSS_ROWS = {"rows": [
    _row(_U0, 0.0, 0.0, mf=0.0), _row(_U0, 2.0, 2.0, mf=1.0),
    _row(_U1, 10.0, 0.2, mf=1.0), _row(_U1, 12.0, 1.8, mf=0.0),
]}

_RISK_COHORT_BIN = {"cohort_id": "p0:snr:q0", "bin": {"feature": "snr", "lo": None, "hi": 1.5}}
_C0_BINS = {"snr": (1.0, 2.0, 3.0), "missing_rate": (0.1, 0.2, 0.3)}


def _risk_ev(cohort=None, sha="a" * 16, ops=("impute_ema", "smooth_ma", "winsorize"),
             fps=None) -> dict:
    if fps is None:   # 默认：snr 全 bin0（(c) 可用）、missing_rate 分散（不合格）
        fps = [{"snr": 0.5, "missing_rate": 0.02 + 0.1 * i} for i in range(5)]
    return {"cohort": dict(cohort or _RISK_COHORT_BIN), "accused_sha": sha,
            "accused_ops": list(ops), "fingerprints": fps}


def _weights(cand: MinedCandidate) -> dict:
    return cand.proposal_dict["new_selector"]["weights"]


def _by_recipe(cands) -> dict:
    return {c.recipe_id: c for c in cands}


def _ref_ridge(Z, y):
    """测试侧独立参考实现：ridge α=1、截距不受罚、截距不返回。"""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.hstack([Z, np.ones((len(y), 1))])
    A = X.T @ X
    for j in range(Z.shape[1]):
        A[j, j] += 1.0
    return np.linalg.solve(A, X.T @ y)[: Z.shape[1]]


def _ref_standardize(X, mu=None, sd=None):
    X = np.asarray(X, dtype=float)
    if mu is None:
        mu = X.mean(axis=0)
        sd = np.sqrt(np.maximum(X.var(axis=0), 1e-12))
    return (X - mu) / sd, mu, sd


# ============================== 通用 ==============================
def test_allowlist_order_frozen():
    assert ALLOWLIST_ORDER == ("snr", "missing_rate") + P_FEATS_FROZEN
    assert frozenset(ALLOWLIST_ORDER) == P0_FEATURE_ALLOWLIST
    assert len(ALLOWLIST_ORDER) == len(P0_FEATURE_ALLOWLIST)


def test_fold_assignment_formula_stable():
    for u in ["u0", "u17", "alpha", "序列-7", "", "m4_hourly|item12|snrLow"]:
        expect = int(hashlib.sha256(u.encode("utf-8")).hexdigest(), 16) % 2
        assert fold_of(u) == expect
        assert fold_of(u) == fold_of(u)          # 重放稳定
    assert {fold_of(f"u{i}") for i in range(30)} == {0, 1}   # 两 fold 都可达


def test_unknown_family_and_bad_state_raise():
    with pytest.raises(ValueError, match="family"):
        mine("bogus", None, H0)
    with pytest.raises(ValueError, match="P6HarnessState"):
        mine("sampler", None, state="not-a-state")


def test_family_aliases_s1_s2_s3():
    assert FAMILY_ALIASES == {"S1": "selector", "S2": "sampler", "S3": "risk"}
    trip = lambda cs: [(c.recipe_id, c.candidate_sha, c.provenance_digest) for c in cs]
    assert trip(mine("S1", _HAND_ROWS, H0)) == trip(mine("selector", _HAND_ROWS, H0))
    assert trip(mine("S2", None, H0)) == trip(mine("sampler", None, H0))
    assert trip(mine("S3", _risk_ev(), H0, c0_bins=_C0_BINS)) == trip(
        mine("risk", _risk_ev(), H0, c0_bins=_C0_BINS)
    )


def test_at_most_three_and_deterministic_replay():
    calls = [("selector", _CROSS_ROWS, {}), ("sampler", None, {}),
             ("risk", _risk_ev(), {"c0_bins": _C0_BINS})]
    for fam, ev, kw in calls:
        a = mine(fam, ev, H0, **kw)
        b = mine(fam, ev, H0, **kw)
        assert 1 <= len(a) <= 3
        assert [(c.recipe_id, c.candidate_sha, c.provenance_digest) for c in a] == [
            (c.recipe_id, c.candidate_sha, c.provenance_digest) for c in b
        ]


def test_tie_break_canonical_sha_ascending():
    c1 = MinedCandidate(({"kind": "x", "v": 1},), "selector_b", {}, "d1")
    c2 = MinedCandidate(({"kind": "x", "v": 2},), "selector_b", {}, "d2")
    lo, hi = sorted([c2, c1], key=candidate_sort_key)
    assert lo.candidate_sha < hi.candidate_sha            # 同配方序 → sha 升序
    a = MinedCandidate(({"kind": "zzz", "v": 3},), "selector_a", {}, "d3")
    assert sorted([c1, a], key=candidate_sort_key)[0] is a   # 配方序优先于 sha
    assert c1.candidate_sha == hashlib.sha256(
        canonical_json([{"kind": "x", "v": 1}]).encode("utf-8")
    ).hexdigest()[:16]                                    # sha 口径 = canonical(list(proposals))
    with pytest.raises(ValueError, match="recipe_id"):
        candidate_sort_key(MinedCandidate(({"kind": "x"},), "nope", {}, "d"))


def test_provenance_digest_recomputable_stable_distinct():
    for fam, ev, kw in [("selector", _HAND_ROWS, {}), ("sampler", None, {}),
                        ("risk", _risk_ev(), {"c0_bins": _C0_BINS})]:
        cands = mine(fam, ev, H0, **kw)
        digests = set()
        for c in cands:
            assert c.provenance_digest == hashlib.sha256(
                canonical_json(c.provenance).encode("utf-8")
            ).hexdigest()                                 # 摘要可复算
            assert c.provenance["recipe_id"] == c.recipe_id
            assert c.provenance["schema"] == "p6-miner/1"
            digests.add(c.provenance_digest)
        assert len(digests) == len(cands)                 # 配方间两两不同


def test_proposal_dict_property_singleton_vs_bundle():
    cands = _by_recipe(mine("risk", _risk_ev(), H0, c0_bins=_C0_BINS))
    assert isinstance(cands["risk_a"].proposal_dict, dict)          # 单提案 OK
    assert isinstance(cands["risk_c"].proposal_dict, dict)          # 合取也是单规则 → OK
    with pytest.raises(ValueError, match="proposal_dicts"):
        _ = cands["risk_b"].proposal_dict                           # 多成员族 bundle 拒单取


# ============================== SelectorPatch 族 ==============================
def test_selector_a_hand_ridge():
    cands = _by_recipe(mine("selector", _HAND_ROWS, H0))
    w = _weights(cands["selector_a"])
    assert set(w) == set(KNOWN_FEATURES)                  # 全集固定特征
    # 闭式：w = Σzy/(Σz²+α) = 4/(4+1) = 0.8；常量列 z=0 → 权重恰 0
    assert abs(w["proxy_score"] - 0.8) < 1e-12
    assert w["n_steps"] == 0.0 and w["has_guard"] == 0.0 and w["modified_fraction"] == 0.0
    sel = cands["selector_a"].proposal_dict["new_selector"]
    assert sel["kind"] == "weighted_features"


def test_selector_b_rank_target_hand():
    cands = _by_recipe(mine("selector", _HAND_ROWS, H0))
    w = _weights(cands["selector_b"])
    # episode 内秩 [1,2]（gain 大者秩大）→ w = Σz·rank/(Σz²+1) = 2/5 = 0.4
    assert abs(w["proxy_score"] - 0.4) < 1e-12
    assert w["modified_fraction"] == 0.0


def test_selector_b_rank_direction_frozen():
    # 单 episode：gain 与 proxy 反向 → 秩与 proxy 反向 → 权重为负（方向写死：gain 大者秩大）
    ev_neg = {"rows": [_row("solo", 1.0, 5.0), _row("solo", 3.0, 1.0)]}
    w_neg = _weights(_by_recipe(mine("selector", ev_neg, H0))["selector_b"])
    assert abs(w_neg["proxy_score"] - (-1.0 / 3.0)) < 1e-12
    ev_pos = {"rows": [_row("solo", 1.0, 1.0), _row("solo", 3.0, 5.0)]}
    w_pos = _weights(_by_recipe(mine("selector", ev_pos, H0))["selector_b"])
    assert abs(w_pos["proxy_score"] - (1.0 / 3.0)) < 1e-12


def test_selector_b_ties_take_average_rank():
    # e0 全 tie（秩 1.5/1.5 均秩）、e1 不 tie（秩 1/2）→ w_b = Σz·rank/(Σz²+1) = 1/5 = 0.2；
    # 若 ties 错用序次 1/2 → y_rank=[1,2,1,2] → 0.4（且与 (a) 撞车被去重）——0.2 钉死均秩。
    ev = {"rows": [_row("e0", 1.0, 7.0), _row("e0", 3.0, 7.0),
                   _row("e1", 1.0, 0.0), _row("e1", 3.0, 2.0)]}
    cands = _by_recipe(mine("selector", ev, H0))
    assert "selector_b" in cands
    assert abs(_weights(cands["selector_b"])["proxy_score"] - 0.2) < 1e-12


def test_selector_c_soft_threshold_per_coordinate():
    cands = _by_recipe(mine("selector", _CROSS_ROWS, H0))
    rows = _CROSS_ROWS["rows"]
    X = np.array([[r["features"][f] for f in KNOWN_FEATURES] for r in rows])
    y = np.array([r["train_gain"] for r in rows])
    Z, _, _ = _ref_standardize(X)                          # full-D 常量
    folds = np.array([fold_of(r["episode_uid"]) for r in rows])
    w_full = _ref_ridge(Z, y)
    w0 = _ref_ridge(Z[folds == 0], y[folds == 0])
    w1 = _ref_ridge(Z[folds == 1], y[folds == 1])
    sigma = np.abs(w0 - w1) / np.sqrt(2.0)
    wc_ref = np.sign(w_full) * np.maximum(0.0, np.abs(w_full) - sigma)
    got = _weights(cands["selector_c"])
    for j, f in enumerate(KNOWN_FEATURES):                 # 逐坐标核对（round 12 后一致）
        assert got[f] == pytest.approx(round(float(wc_ref[j]), 12), abs=1e-12), f
    # 构造保证：fold 间翻号坐标被收缩到恰 0，稳定坐标存活
    assert got["modified_fraction"] == 0.0
    assert got["proxy_score"] > 0.0
    # (a) 也对照 full-D refit
    got_a = _weights(cands["selector_a"])
    for j, f in enumerate(KNOWN_FEATURES):
        assert got_a[f] == pytest.approx(round(float(w_full[j]), 12), abs=1e-12), f


def test_selector_zstd_constants_from_full_d():
    """(c) 的 fold 拟合必须用 full-D 标准化常量——per-fold 常量的替代实现须给出不同结果。"""
    rows = _CROSS_ROWS["rows"]
    X = np.array([[r["features"][f] for f in KNOWN_FEATURES] for r in rows])
    y = np.array([r["train_gain"] for r in rows])
    folds = np.array([fold_of(r["episode_uid"]) for r in rows])
    Z_full, _, _ = _ref_standardize(X)
    w_full = _ref_ridge(Z_full, y)
    # 备择（错误）实现：每 fold 各自标准化
    wc_alt_parts = []
    for f in (0, 1):
        Zf, _, _ = _ref_standardize(X[folds == f])
        wc_alt_parts.append(_ref_ridge(Zf, y[folds == f]))
    sigma_alt = np.abs(wc_alt_parts[0] - wc_alt_parts[1]) / np.sqrt(2.0)
    wc_alt = np.sign(w_full) * np.maximum(0.0, np.abs(w_full) - sigma_alt)
    # 正确实现：fold 拟合同用 full-D 常量
    w0 = _ref_ridge(Z_full[folds == 0], y[folds == 0])
    w1 = _ref_ridge(Z_full[folds == 1], y[folds == 1])
    sigma = np.abs(w0 - w1) / np.sqrt(2.0)
    wc = np.sign(w_full) * np.maximum(0.0, np.abs(w_full) - sigma)
    assert np.max(np.abs(wc_alt - wc)) > 1e-6              # 两种口径在本 fixture 上可区分
    got = _weights(_by_recipe(mine("selector", _CROSS_ROWS, H0))["selector_c"])
    for j, f in enumerate(KNOWN_FEATURES):
        assert got[f] == pytest.approx(round(float(wc[j]), 12), abs=1e-12)
        # 且不等于 per-fold 口径（在可区分坐标上）
    j_proxy = KNOWN_FEATURES.index("proxy_score")
    assert abs(got["proxy_score"] - round(float(wc_alt[j_proxy]), 12)) > 1e-9


def test_selector_c_unavailable_when_single_fold():
    u_a, u_b = _uid_in_fold(0), _uid_in_fold(0, skip=1)    # 两 uid 同 fold
    ev = {"rows": [_row(u_a, 1.0, 0.0), _row(u_a, 3.0, 2.0),
                   _row(u_b, 1.0, 0.5), _row(u_b, 3.0, 1.5)]}
    cands = mine("selector", ev, H0)
    assert [c.recipe_id for c in cands] == ["selector_a", "selector_b"]   # (c) 缺席合法


def test_selector_weights_rounded_12_and_no_negative_zero():
    import math as _math

    # 长小数 fixture：12 位舍入不动点
    ev = {"rows": [_row("e0", 1.0, 1.0 / 3.0), _row("e0", 2.0, 2.0 / 7.0),
                   _row("e1", 3.0, -1.0 / 3.0), _row("e1", 5.0, 1.0 / 9.0)]}
    # 翻号 fixture：w_full[mf] < 0 且 fold 全收缩 → (c) 的 mf 原生为 −0.0，须归一为 +0.0
    ev_flip = {"rows": [
        _row(_U0, 0.0, 0.0, mf=1.0), _row(_U0, 2.0, 2.0, mf=0.0),
        _row(_U1, 10.0, 0.2, mf=0.0), _row(_U1, 12.0, 1.8, mf=1.0),
    ]}
    for fixture in (ev, ev_flip):
        for c in mine("selector", fixture, H0):
            for f, w in _weights(c).items():
                assert w == round(w, 12), (c.recipe_id, f)         # 已是 12 位舍入不动点
                assert not (w == 0.0 and _math.copysign(1.0, w) < 0.0), (
                    c.recipe_id, f)                                # 无 −0.0（JSON/sha 稳定）
    flip = _by_recipe(mine("selector", ev_flip, H0))
    assert _weights(flip["selector_a"])["modified_fraction"] == pytest.approx(-0.08)
    wc_mf = _weights(flip["selector_c"])["modified_fraction"]      # sign(−)·0 → 归一后 +0.0
    assert wc_mf == 0.0 and _math.copysign(1.0, wc_mf) > 0.0


def test_selector_degenerate_dedup_lt3_legal():
    # 全零 gain → 三配方权重向量全同（0 向量）→ 去重为 1（keep 配方序首位）
    ev = {"rows": [_row(_U0, 1.0, 0.0), _row(_U0, 3.0, 0.0),
                   _row(_U1, 1.0, 0.0), _row(_U1, 3.0, 0.0)]}
    cands = mine("selector", ev, H0)
    assert len(cands) == 1
    assert cands[0].recipe_id == "selector_a"
    assert all(v == 0.0 for v in _weights(cands[0]).values())


def test_selector_order_compile_apply():
    cands = mine("selector", _CROSS_ROWS, H0)
    assert [c.recipe_id for c in cands] == ["selector_a", "selector_b", "selector_c"]
    for c in cands:
        op = compile_proposal(c.proposal_dict)
        assert isinstance(op, SelectorPatch)
        st = apply_edit(H0, op)                            # 全链可编译可应用
        assert st.selector.kind == "weighted_features"
        assert st.selector.weights == _weights(c)


def test_selector_bad_evidence_raises():
    with pytest.raises(ValueError, match="rows"):
        mine("selector", {"rows": []}, H0)
    with pytest.raises(ValueError, match="rows"):
        mine("selector", None, H0)
    with pytest.raises(ValueError, match="train_gain"):
        mine("selector", {"rows": [_row("e0", 1.0, float("nan"))]}, H0)
    with pytest.raises(ValueError, match="features"):
        mine("selector", {"rows": [_row("e0", float("inf"), 1.0)]}, H0)


def test_selector_ignores_unknown_feature_keys():
    ev_extra = {"rows": [_row("e0", 1.0, 0.0, junk_feature=99.0),
                         _row("e0", 3.0, 2.0, junk_feature=-7.0),
                         _row("e1", 1.0, 0.0), _row("e1", 3.0, 2.0)]}
    a = [(c.recipe_id, _weights(c)) for c in mine("selector", ev_extra, H0)]
    b = [(c.recipe_id, _weights(c)) for c in mine("selector", _HAND_ROWS, H0)]
    assert a == b                                          # 非 KNOWN 键（junk/exec_ok）不进设计矩阵


# ============================== SamplerPatch 族 ==============================
def test_sampler_literal_recipes_from_h0():
    st = replace(H0, sampler=SamplerSpec(allocation=dict(H0_ALLOCATION),
                                         expected_total=8,
                                         random_params={"windows": [5, 9]}))
    cands = mine("sampler", None, st)
    assert [c.recipe_id for c in cands] == ["sampler_a", "sampler_b", "sampler_c"]
    got = {c.recipe_id: c.proposal_dict["new_sampler"] for c in cands}
    assert got["sampler_a"]["allocation"] == {"det": 2, "random": 6, "llm": 0}
    assert got["sampler_b"]["allocation"] == {"det": 3, "random": 3, "llm": 2}
    assert got["sampler_c"]["allocation"] == {"det": 1, "random": 7, "llm": 0}
    for c in cands:
        ns = c.proposal_dict["new_sampler"]
        assert ns["expected_total"] == 8                   # 总 K 冻结
        assert ns["random_params"] == {"windows": [5, 9]}  # 原样保留，不夹带第二处改动
        op = compile_proposal(c.proposal_dict)
        assert isinstance(op, SamplerPatch)
        st2 = apply_edit(st, op)
        assert st2.sampler.allocation == ns["allocation"]


def test_sampler_negative_quota_unavailable():
    st = replace(H0, sampler=SamplerSpec(allocation={"det": 0, "random": 2, "llm": 6},
                                         expected_total=8))
    cands = mine("sampler", None, st)
    assert [c.recipe_id for c in cands] == ["sampler_b"]   # a: det−1<0；c: det−2<0
    assert cands[0].proposal_dict["new_sampler"]["allocation"] == {"det": 0, "random": 0, "llm": 8}


def test_sampler_all_unavailable_returns_empty():
    st = replace(H0, sampler=SamplerSpec(allocation={"det": 0, "random": 1, "llm": 7},
                                         expected_total=8))
    assert mine("sampler", None, st) == []                 # 上层按 abstain 处理


# ============================== RiskRulePatch 族 ==============================
def test_risk_a_bin_cohort_single_condition():
    cands = _by_recipe(mine("risk", _risk_ev(), H0, c0_bins=_C0_BINS))
    cand = cands["risk_a"]
    rule = cand.proposal_dict["add_rule"]
    assert rule["when"] == [{"feature": "snr", "op": "<", "value": 1.5}]   # 原生列表形
    assert rule["then"] == {"action": "ban", "target": "a" * 16}
    assert rule["rule_id"] == f"risk_a_p0:snr:q0_{'a' * 16}"
    assert cand.provenance["semantics"] == {}              # 简化：无 rule_groups/conjunction_note
    op = compile_proposal(cand.proposal_dict)
    assert isinstance(op, RiskRulePatch)
    st = apply_edit(H0, op)
    assert len(st.risk_rules) == 1


def test_risk_a_preset_membership_scope():
    """preset cohort → 单原子 preset 成员资格 scope（F7/finding 37）：ban 程序 sha @ (preset==名)；
    对 fingerprint["preset"] 求成员判定，取代旧 C0 中位数半平面近似。"""
    cohort = {"cohort_id": "preset:snrLow|miss", "preset": "snrLow|miss"}
    cands = _by_recipe(mine("risk", _risk_ev(cohort=cohort), H0))
    cand = cands["risk_a"]
    assert len(cand.proposal_dicts) == 1
    rule = cand.proposal_dict["add_rule"]
    assert rule["rule_id"] == f"risk_a_preset:snrLow|miss_{'a' * 16}"
    assert rule["when"] == [{"feature": "preset", "op": "==", "value": "snrLow|miss"}]
    assert cand.provenance["semantics"] == {}
    op = compile_proposal(cand.proposal_dict)
    assert isinstance(op, RiskRulePatch)
    st = apply_edit(H0, op)
    assert len(st.risk_rules) == 1
    # 成员资格运行语义：只有该 preset 触发；其它 preset / 缺 preset 键均不触发（保守）
    r = st.risk_rules[0]
    assert r.matches({"preset": "snrLow|miss"})
    assert not r.matches({"preset": "snrHigh|full"})
    assert not r.matches({"snr": 1.0, "missing_rate": 0.2})     # 无 preset 键 → False


def test_risk_b_family_map_first_hit_and_members():
    assert OP_FAMILY_MAP == {"denoise_median": "denoiser", "denoise_savgol": "denoiser",
                             "smooth_ma": "denoiser", "winsorize": "outlier",
                             "impute_linear": "imputer"}
    # 第一命中 = op 列表序：impute_ema 不在映射 → smooth_ma → denoiser（3 成员 ban）
    cand = _by_recipe(mine("risk", _risk_ev(), H0))["risk_b"]
    targets = [p["add_rule"]["then"]["target"] for p in cand.proposal_dicts]
    assert tuple(targets) == OP_FAMILY_MEMBERS["denoiser"]
    # 每成员各一条**单规则**，when 全部 = 同一 scope 条件列表（规则间并集 = 族语义本身）
    for p in cand.proposal_dicts:
        assert p["add_rule"]["when"] == [{"feature": "snr", "op": "<", "value": 1.5}]
    assert cand.provenance["semantics"] == {"op_family": "denoiser"}   # 无 rule_groups
    st = H0
    for p in cand.proposal_dicts:
        st = apply_edit(st, compile_proposal(p))
    assert len(st.risk_rules) == 3
    # 第一命中在 winsorize 之前拿到 impute_linear → imputer 单成员
    cand2 = _by_recipe(mine("risk", _risk_ev(ops=("impute_linear", "winsorize")), H0))["risk_b"]
    assert [p["add_rule"]["then"]["target"] for p in cand2.proposal_dicts] == ["impute_linear"]


def test_risk_b_unavailable_when_no_family_hit():
    cands = mine("risk", _risk_ev(ops=("impute_ema", "outlier_iqr")), H0, c0_bins=_C0_BINS)
    assert [c.recipe_id for c in cands] == ["risk_a", "risk_c"]


def test_risk_c_dominant_bin_allowlist_order_scan():
    # snr 分散（首特征不合格）→ 扫描继续 → missing_rate 全 bin0 合格
    fps = [{"snr": 0.5 + i, "missing_rate": 0.05} for i in range(5)]   # snr 跨 4 bin
    cand = _by_recipe(mine("risk", _risk_ev(fps=fps), H0, c0_bins=_C0_BINS))["risk_c"]
    assert len(cand.proposal_dicts) == 1                   # 合取在同一规则 when 内
    rule = cand.proposal_dict["add_rule"]
    assert rule["when"] == [{"feature": "snr", "op": "<", "value": 1.5},           # (a) scope
                            {"feature": "missing_rate", "op": "<", "value": 0.1}]  # + 合取 bin
    assert rule["rule_id"] == f"risk_c_p0:snr:q0_{'a' * 16}"       # 无 _and 后缀
    assert cand.provenance["semantics"] == {
        "dominant_bin": {"feature": "missing_rate", "bin_index": 0}
    }                                                      # 无 rule_groups/conjunction_note


def test_risk_c_80pct_boundary_geq_frozen():
    def fps_with(k_in_bin0: int) -> list:
        # 10 个 episode：k 个 snr∈bin0，其余 bin2；missing_rate 轮转 4 bin（永不合格）
        return [{"snr": 0.5 if i < k_in_bin0 else 2.5,
                 "missing_rate": 0.05 + 0.1 * (i % 4)} for i in range(10)]

    got = mine("risk", _risk_ev(fps=fps_with(8)), H0, c0_bins=_C0_BINS)
    assert "risk_c" in {c.recipe_id for c in got}          # 恰 80%（8/10）满足（写死 ≥）
    got = mine("risk", _risk_ev(fps=fps_with(7)), H0, c0_bins=_C0_BINS)
    assert "risk_c" not in {c.recipe_id for c in got}      # 70% 不满足


def test_risk_c_middle_bin_two_conditions_left_closed():
    # snr 全 == 1.0 == q1 → bisect_right → bin1（左闭右开）→ bin 贡献 2 条件
    # → 单规则 when 共 3 个原子条件（scope + bin 两原子，同规则 AND）
    fps = [{"snr": 1.0, "missing_rate": 0.05 + 0.1 * (i % 4)} for i in range(5)]
    cand = _by_recipe(mine("risk", _risk_ev(fps=fps), H0, c0_bins=_C0_BINS))["risk_c"]
    assert len(cand.proposal_dicts) == 1
    rule = cand.proposal_dict["add_rule"]
    assert rule["when"] == [{"feature": "snr", "op": "<", "value": 1.5},
                            {"feature": "snr", "op": ">=", "value": 1.0},
                            {"feature": "snr", "op": "<", "value": 2.0}]
    st = apply_edit(H0, compile_proposal(cand.proposal_dict))
    assert len(st.risk_rules) == 1
    # 三原子 AND 的运行语义哨兵：只有 [1.0, 1.5) 触发（bin 外两侧都不触发）
    r = st.risk_rules[0]
    assert r.matches({"snr": 1.2})
    assert not r.matches({"snr": 0.5}) and not r.matches({"snr": 1.7})


def test_risk_c_unavailable_paths():
    # 无 c0_bins → (c) 缺席
    got = {c.recipe_id for c in mine("risk", _risk_ev(), H0)}
    assert got == {"risk_a", "risk_b"}
    # 空 fingerprints → (c) 缺席
    got = {c.recipe_id for c in mine("risk", _risk_ev(fps=[]), H0, c0_bins=_C0_BINS)}
    assert got == {"risk_a", "risk_b"}
    # cutpoints 缺该特征 + 其余特征不合格 → (c) 缺席
    fps = [{"snr": 0.5 + i, "missing_rate": 0.05 + 0.1 * (i % 4)} for i in range(5)]
    got = {c.recipe_id for c in mine("risk", _risk_ev(fps=fps), H0,
                                     c0_bins={"missing_rate": (0.1, 0.2, 0.3)})}
    assert got == {"risk_a", "risk_b"}


def test_risk_whole_family_unavailable_without_scope():
    # bin cohort 两边 lo/hi 皆 None → 空 scope（无法作用域化）→ 整族不可用。
    cohort = {"cohort_id": "p0:snr:all", "bin": {"feature": "snr", "lo": None, "hi": None}}
    assert mine("risk", _risk_ev(cohort=cohort), H0, c0_bins=_C0_BINS) == []
    # 注：preset cohort（F7）恒产出成员资格 scope（非空），不存在"空 preset scope 不可用"分支。


def test_risk_bad_evidence_raises():
    bad_feat = {"cohort_id": "c", "bin": {"feature": "loss", "lo": 0.0, "hi": 1.0}}
    with pytest.raises(ValueError, match="allowlist"):
        mine("risk", _risk_ev(cohort=bad_feat), H0)
    both = {"cohort_id": "c", "bin": {"feature": "snr", "lo": 0.0, "hi": 1.0}, "preset": "p"}
    with pytest.raises(ValueError, match="恰含"):
        mine("risk", _risk_ev(cohort=both), H0)
    with pytest.raises(ValueError, match="accused_sha"):
        mine("risk", _risk_ev(sha=""), H0)
    with pytest.raises(ValueError, match="cutpoints"):
        mine("risk", _risk_ev(), H0, c0_bins={"snr": (1.0, 2.0)})
    # preset cohort（F7）：preset 名必须是非空 str（空/缺 → 响亮 raise）
    bad_preset = {"cohort_id": "c", "preset": ""}
    with pytest.raises(ValueError, match="preset"):
        mine("risk", _risk_ev(cohort=bad_preset), H0)


def test_risk_rules_pass_state_validation_end_to_end():
    """全部配方的每条规则都是单规则、when=非空条件列表，过 RiskRuleSpec spec 级校验 +
    fast_path 可用的 state 构造（1 提案 = 1 规则）。"""
    for c in mine("risk", _risk_ev(), H0, c0_bins=_C0_BINS):
        st = H0
        for p in c.proposal_dicts:
            w = p["add_rule"]["when"]
            assert isinstance(w, list) and len(w) >= 1     # 原生列表形（单规则携带全部合取）
            assert all(set(cond) == {"feature", "op", "value"} for cond in w)
            st = apply_edit(st, compile_proposal(p))
        assert isinstance(st, P6HarnessState)
        assert len(st.risk_rules) == len(c.proposal_dicts)


def test_recipe_registry_frozen():
    assert FAMILY_RECIPES == {
        "selector": ("selector_a", "selector_b", "selector_c"),
        "sampler": ("sampler_a", "sampler_b", "sampler_c"),
        "risk": ("risk_a", "risk_b", "risk_c"),
    }
