"""tests/test_p6_surfaces.py — P6 harness 状态 + 三 edit surface + fast path 配对语义（toy-only）。

运行：D:\\Anaconda_envs\\envs\\project\\python.exe -m pytest SelfEvolvingHarnessTS/tests/test_p6_surfaces.py -q
（cwd = C:\\Users\\辉\\Desktop\\Agent）

对齐 prereg_p6 §3.3/§4：H0 = 字面量 {det:3, random:5, llm:0}（K=8 请求 slot 预算）；
det 阶梯 3 程序、超配如实短缺不回填；realized unique pool size 逐 episode 落账；
RiskRule when = 条件列表（**同规则内 AND**，单条件 dict 向后兼容、canonical 统一列表形、
空列表 raise；bin scope 两原子同规则合取——bin 外 episode 绝不被 ban 的回归反例在
test_bin_scope_out_of_bin_episodes_never_banned）；每条件 feature 冻结 allowlist
（{"snr","missing_rate"} ∪ P_FEATS，与 e32_policy 逐项核对）；paired_risk_run 作用域外
episode prepared artifact 字节级校验（门④，含合取 bin scope 变体）。

全部确定性：只用固定 rng 的合成 toy 数据；不读 results/ 与 data/；无网络、无 LLM
（llm_supplier 只测注入的本地确定性 callable）。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from SelfEvolvingHarnessTS.p6.edit_surfaces import (
    RiskRulePatch,
    SamplerPatch,
    SelectorPatch,
    compile_proposal,
)
from SelfEvolvingHarnessTS.p6.fast_path import (
    FAILED_PROXY,
    Candidate,
    FastPathResult,
    P6PairingError,
    apply_risk,
    det_ladder,
    generate_candidates,
    generate_candidates_with_stats,
    make_candidate,
    paired_risk_run,
    paired_sampler_run,
    paired_selector_run,
    prepared_artifact,
    program_sha,
    proxy_score,
    random_grammar_sampler,
    run_fast_path,
    select,
    toy_fingerprint,
)
from SelfEvolvingHarnessTS.p6.harness_state import (
    H0_ALLOCATION,
    H0_EXPECTED_TOTAL_K,
    P0_FEATURE_ALLOWLIST,
    P6EditError,
    P6HarnessState,
    P_FEATS_FROZEN,
    RiskRuleSpec,
    SamplerSpec,
    SelectorSpec,
    apply_edit,
    default_state,
)

K = 8   # prereg §4：H0 请求 slot 预算（冻结字面量）


# ============================== toy 数据（固定种子，合成） ==============================
def _views() -> dict:
    out = {}
    for i in range(3):
        t = np.arange(160, dtype=float)
        rng = np.random.default_rng(100 + i)
        x = np.sin(2.0 * np.pi * t / 24.0 + 0.7 * i) + 0.3 * i + rng.normal(0.0, 0.05, t.size)
        if i == 1:
            x[10:14] = np.nan          # u1 带缺失，走 imputer 真路径
        out[f"u{i}"] = x
    return out


def _rule(rule_id: str = "r1", feature: str = "missing_rate", op: str = ">=",
          value: float = 0.0, target: str = "winsorize") -> RiskRuleSpec:
    """默认规则 missing_rate ≥ 0.0：在任何 toy 指纹上都触发（原 n>10 的 allowlist 版）。
    单条件 dict 形——同时兼作向后兼容形态的常驻覆盖。"""
    return RiskRuleSpec(rule_id=rule_id,
                        when={"feature": feature, "op": op, "value": value},
                        then={"action": "ban", "target": target})


def _rule_and(rule_id: str, conds, target: str = "winsorize") -> RiskRuleSpec:
    """条件列表形（同规则 AND）——when 原生 schema（prereg §4 冻结）。"""
    return RiskRuleSpec(rule_id=rule_id, when=[dict(c) for c in conds],
                        then={"action": "ban", "target": target})


def _weighted() -> SelectorSpec:
    return SelectorSpec(kind="weighted_features", weights={"n_steps": 1.0})


# ============================== A. state sha / 版本链 ==============================
def test_default_state_deterministic_sha_and_h0_definition():
    s1, s2 = default_state(K), default_state(K)
    assert s1.sha() == s2.sha()
    assert len(s1.sha()) == 16 and all(c in "0123456789abcdef" for c in s1.sha())
    # H0 定义（prereg §4 代码字面量，非均分公式）：proxy_rank + {det:3, random:5, llm:0} + v0
    assert s1.version == "v0" and s1.parent_sha is None and s1.edit_log == ()
    assert s1.selector.kind == "proxy_rank"
    assert s1.sampler.allocation == {"det": 3, "random": 5, "llm": 0} == H0_ALLOCATION
    assert s1.sampler.expected_total == K == H0_EXPECTED_TOTAL_K == 8
    assert s1.risk_rules == ()
    # 无参调用 = 同一 H0（默认 expected_total_K=8）
    assert default_state().sha() == s1.sha()
    # sha 只覆盖语义组件：改 provenance 字段（version）不改 sha
    assert replace(s1, version="vX").sha() == s1.sha()
    # expected_total 冻结校验：K ≠ 8 一律 ValueError（均分公式已删除，不再随 K 缩放）
    with pytest.raises(ValueError):
        default_state(6)
    with pytest.raises(ValueError):
        default_state(0)


def test_apply_edit_version_parent_chain_and_edit_log():
    s0 = default_state(K)
    s1 = apply_edit(s0, SelectorPatch(_weighted()))
    s2 = apply_edit(s1, RiskRulePatch(_rule("chain_r")))
    assert (s1.version, s2.version) == ("v0.e1", "v0.e2")
    assert s1.parent_sha == s0.sha() and s2.parent_sha == s1.sha()
    assert len(s1.edit_log) == 1 and len(s2.edit_log) == 2
    for entry, st in ((s1.edit_log[-1], s1), (s2.edit_log[-1], s2)):
        assert entry["applied"] is True
        assert entry["new_sha"] == st.sha()
        assert "kind" in entry["op"]
    assert s2.edit_log[0] == s1.edit_log[0]          # append-only
    # 不可变：s0 未被改动
    assert s0.version == "v0" and s0.edit_log == () and s0.risk_rules == ()


def test_sha_excluding_components():
    sA = default_state(K)
    sB = apply_edit(sA, SelectorPatch(_weighted()))
    assert sA.sha() != sB.sha()
    assert sA.sha_excluding("selector") == sB.sha_excluding("selector")
    assert sA.sha_excluding("sampler") != sB.sha_excluding("sampler")   # selector 仍在内
    with pytest.raises(ValueError):
        sA.sha_excluding("nope")


# ============================== B. 三 patch：validate/apply/非法拒绝 ==============================
def test_selector_patch_apply_and_reject_unknown_kind_and_feature():
    s0 = default_state(K)
    s1 = apply_edit(s0, SelectorPatch(_weighted()))
    assert s1.selector == _weighted()
    bad_kind = SelectorPatch(SelectorSpec(kind="nope"))
    assert isinstance(bad_kind.validate(s0), str)
    with pytest.raises(P6EditError):
        apply_edit(s0, bad_kind)
    bad_feat = SelectorPatch(SelectorSpec(kind="weighted_features", weights={"zzz": 1.0}))
    assert isinstance(bad_feat.validate(s0), str)
    with pytest.raises(P6EditError):
        apply_edit(s0, bad_feat)
    # 直接构造非法 state 也 raise（构造即校验）
    with pytest.raises(ValueError):
        P6HarnessState(version="v0", selector=SelectorSpec(kind="nope"), sampler=s0.sampler)


def test_sampler_patch_total_k_frozen():
    s0 = default_state(K)
    ok = SamplerPatch(SamplerSpec(allocation={"det": 1, "random": 7, "llm": 0}, expected_total=K))
    s1 = apply_edit(s0, ok)
    assert s1.sampler.total() == K and s1.sampler.allocation["random"] == 7
    # Σ allocation ≠ expected_total → 拒
    drift = SamplerPatch(SamplerSpec(allocation={"det": 1, "random": 1, "llm": 0}, expected_total=K))
    with pytest.raises(P6EditError):
        apply_edit(s0, drift)
    # 改 expected_total 本身（自洽但预算漂移）→ 拒
    regrow = SamplerPatch(SamplerSpec(allocation={"det": 5, "random": 5, "llm": 0}, expected_total=10))
    assert "漂移" in regrow.validate(s0)
    with pytest.raises(P6EditError):
        apply_edit(s0, regrow)
    # 未知供给器名 / 缺显式三键 → 拒
    with pytest.raises(P6EditError):
        apply_edit(s0, SamplerPatch(SamplerSpec(
            allocation={"det": 8, "random": 0, "llm": 0, "web": 0}, expected_total=K)))
    with pytest.raises(P6EditError):
        apply_edit(s0, SamplerPatch(SamplerSpec(allocation={"det": 8}, expected_total=K)))


def test_risk_rule_patch_duplicate_and_malformed():
    s0 = default_state(K)
    s1 = apply_edit(s0, RiskRulePatch(_rule("dup")))
    assert len(s1.risk_rules) == 1
    with pytest.raises(P6EditError):
        apply_edit(s1, RiskRulePatch(_rule("dup", target="smooth_ma")))   # 重复 rule_id
    with pytest.raises(P6EditError):
        apply_edit(s0, RiskRulePatch(_rule("bad_op", op="~")))            # 未知比较 op
    bad_action = RiskRuleSpec(rule_id="bad_act",
                              when={"feature": "missing_rate", "op": ">", "value": 1.0},
                              then={"action": "allow", "target": "x"})
    with pytest.raises(P6EditError):
        apply_edit(s0, RiskRulePatch(bad_action))


def test_p0_allowlist_matches_e32_policy():
    """核对义务（prereg §3.3）：harness_state 冻结字面量 P_FEATS_FROZEN 与
    e32_policy.P_FEATS 逐项一致（含次序）。e32_policy 不入 P6 freeze 清单且模块级
    引 sklearn，故运行时不 import（harness_state stdlib-only 红线）——漂移由本测试拦截。"""
    from SelfEvolvingHarnessTS.e32_policy import P_FEATS

    assert P_FEATS_FROZEN == tuple(P_FEATS)
    assert P0_FEATURE_ALLOWLIST == frozenset({"snr", "missing_rate"}) | frozenset(P_FEATS)
    assert len(P0_FEATURE_ALLOWLIST) == 10
    # fast_path 再导出的是同一对象（单一来源）
    from SelfEvolvingHarnessTS.p6 import fast_path as fp_mod

    assert fp_mod.P0_FEATURE_ALLOWLIST is P0_FEATURE_ALLOWLIST


def test_risk_rule_feature_allowlist_enforced():
    """非 allowlist 特征（outcome/response/series id/domain id/旧 toy 键）三处全拒：
    ①spec 级 validate raise ②apply_edit（RiskRulePatch）raise ③state 构造期 raise；
    ④apply_risk 第二道闸对被走私进 state 的规则也 raise（纵深防御）。"""
    s0 = default_state(K)
    for feat in ("loss", "train_gain", "batch_delta",           # outcome / judge response
                 "uid", "series_uid", "domain_id",              # series id / domain id
                 "proxy_score", "n", "missing_frac", "mean"):   # 候选特征 / 旧 toy 键
        bad = RiskRuleSpec(rule_id=f"bad_{feat}",
                           when={"feature": feat, "op": ">", "value": 0.0},
                           then={"action": "ban", "target": "winsorize"})
        with pytest.raises(ValueError):
            bad.validate()
        with pytest.raises(P6EditError):
            apply_edit(s0, RiskRulePatch(bad))
        with pytest.raises(ValueError):
            P6HarnessState(version="v0", selector=s0.selector, sampler=s0.sampler,
                           risk_rules=(bad,))
    # allowlist 内特征全部可过 spec 校验
    for feat in sorted(P0_FEATURE_ALLOWLIST):
        _rule(f"ok_{feat}", feature=feat, op=">", value=0.0).validate()
    # apply_risk 第二道闸：绕过构造期校验走私坏规则 → 求值前 raise（即使规则不触发）
    smuggled = RiskRuleSpec(rule_id="smuggled",
                            when={"feature": "proxy_score", "op": ">", "value": 1e9},
                            then={"action": "ban", "target": "winsorize"})
    s_bad = default_state(K)
    object.__setattr__(s_bad, "risk_rules", (smuggled,))        # 测试专用后门
    pool = generate_candidates("u0", s_bad, K)
    with pytest.raises(ValueError):
        apply_risk(pool, {"proxy_score": 0.0}, s_bad)


def test_to_dict_from_dict_roundtrip_and_compile_dispatch():
    ops = [
        SelectorPatch(_weighted()),
        SamplerPatch(SamplerSpec(allocation={"det": 2, "random": 4, "llm": 2}, expected_total=K,
                                 random_params={"windows": [5, 9]})),
        RiskRulePatch(_rule("rt")),
    ]
    for op in ops:
        d = op.to_dict()
        op2 = compile_proposal(d)
        assert type(op2) is type(op) and op2 == op          # 往返恒等
        assert op2.to_dict() == d
    assert compile_proposal({"kind": "unknown_thing"}) is None
    assert compile_proposal({}) is None


# ============================== C. fast path：供给/去重/确定性 ==============================
def test_det_ladder_fixed_three_programs():
    ladder = det_ladder()
    assert len(ladder) == 3
    assert ladder[0].program_steps == (("impute_linear", {}),)
    assert [c.op_names() for c in ladder] == [
        ("impute_linear",),
        ("impute_linear", "winsorize", "denoise_savgol"),
        ("impute_linear", "denoise_median"),
    ]
    assert all(c.source == "det" for c in ladder)
    assert len({c.sha for c in ladder}) == 3
    assert ladder[0].sha == program_sha([("impute_linear", {})])
    assert [c.sha for c in det_ladder()] == [c.sha for c in ladder]      # 两次调用恒等


def test_random_grammar_sampler_deterministic_and_grammar_shape():
    a = random_grammar_sampler("u0", "streamsha", 5)
    b = random_grammar_sampler("u0", "streamsha", 5)
    assert [c.sha for c in a] == [c.sha for c in b]
    assert [c.program_steps for c in a] == [c.program_steps for c in b]
    assert len(a) == 5 and len({c.sha for c in a}) == 5                  # 采样内去重
    for c in a:
        assert c.source == "random"
        assert c.op_names()[0] in ("impute_linear", "impute_ema")        # imputer 必选
        for op, params in c.program_steps[1:]:
            if "window" in params:
                assert params["window"] in (5, 9, 15, 25)
    # uid / state_sha 任一变 → 流变
    assert [c.sha for c in random_grammar_sampler("u1", "streamsha", 5)] != [c.sha for c in a]
    assert [c.sha for c in random_grammar_sampler("u0", "other", 5)] != [c.sha for c in a]
    # 文法覆盖：windows 网格收窄后只出现指定窗口
    n = random_grammar_sampler("u0", "streamsha", 8, {"windows": [15]})
    for c in n:
        for _op, params in c.program_steps:
            if "window" in params:
                assert params["window"] == 15


def test_generate_candidates_allocation_dedup_budget_guard():
    s = default_state(K)
    pool = generate_candidates("u0", s, K)
    assert 0 < len(pool) <= K
    assert len({c.sha for c in pool}) == len(pool)                       # 效果去重按 sha
    srcs = [c.source for c in pool]
    assert srcs.count("det") <= 3 and srcs.count("random") <= 5 and "llm" not in srcs
    with pytest.raises(ValueError):
        generate_candidates("u0", s, K + 1)                              # 预算完整性


def test_generate_candidates_with_stats_and_det_shortfall():
    """prereg §4 K/slot 语义：K=请求 slot 预算；det 阶梯 3 程序超配如实短缺不回填；
    realized unique pool size（跨 supplier sha 去重后）逐 episode 落账。"""
    s = default_state(K)
    pool, stats = generate_candidates_with_stats("u0", s, K)
    assert [c.sha for c in pool] == [c.sha for c in generate_candidates("u0", s, K)]  # 委托一致
    assert stats["uid"] == "u0" and stats["requested_K"] == K
    assert stats["allocation"] == {"det": 3, "random": 5, "llm": 0}
    assert stats["supplied"] == {"det": 3, "random": 5, "llm": 0}
    assert stats["det_shortfall"] == 0
    assert stats["realized_pool_size"] == len(pool)
    assert stats["pre_dedup_size"] - stats["realized_pool_size"] == stats["dedup_removed"] >= 0

    # det 分配 4 > 阶梯 3 → 如实短缺 1，random 不补位（仍只给自己的 4 个名额）
    s4 = apply_edit(s, SamplerPatch(
        SamplerSpec(allocation={"det": 4, "random": 4, "llm": 0}, expected_total=K)))
    pool4, st4 = generate_candidates_with_stats("u0", s4, K)
    assert st4["supplied"]["det"] == 3 and st4["det_shortfall"] == 1
    assert st4["supplied"]["random"] <= 4                                # 不回填
    assert st4["pre_dedup_size"] == 3 + st4["supplied"]["random"] <= 7   # 8 slot 实供 <8
    assert st4["realized_pool_size"] == len(pool4) <= 7
    assert sum(1 for c in pool4 if c.source == "random") <= 4

    # 全 det：8 slot 只有 3 程序阶梯 → realized pool = 3（短缺 5，不回填）
    s8 = apply_edit(s, SamplerPatch(
        SamplerSpec(allocation={"det": 8, "random": 0, "llm": 0}, expected_total=K)))
    pool8, st8 = generate_candidates_with_stats("u0", s8, K)
    assert [c.sha for c in pool8] == [c.sha for c in det_ladder()]
    assert st8["det_shortfall"] == 5 and st8["realized_pool_size"] == 3


def test_llm_supplier_injection_default_zero_and_sha_recomputed():
    s = apply_edit(default_state(K), SamplerPatch(
        SamplerSpec(allocation={"det": 2, "random": 4, "llm": 2}, expected_total=K)))
    # 默认 None → llm 贡献 0
    pool0 = generate_candidates("u0", s, K)
    assert all(c.source != "llm" for c in pool0) and len(pool0) <= 6
    # 注入确定性 callable（含伪造 sha 的 Candidate → sha 必须被重算）。
    # window=7 不在文法网格 (5,9,15,25) 内 → 不可能与 random 供给撞 sha 被去重。
    steps = (("impute_ema", {}), ("smooth_ma", {"window": 7}))
    spoof = Candidate(program_steps=steps, source="llm", features={}, sha="deadbeefdeadbeef")

    def supplier(uid, state, n):
        return [spoof][:n]

    pool1, st1 = generate_candidates_with_stats("u0", s, K, llm_supplier=supplier)
    llm_cands = [c for c in pool1 if c.source == "llm"]
    assert len(llm_cands) == 1
    assert llm_cands[0].sha == program_sha(steps) != "deadbeefdeadbeef"
    assert st1["supplied"]["llm"] == 1                                   # 名额 2、实供 1：如实短缺


def test_run_fast_path_deterministic_bit_level_and_pool_stats():
    views = _views()
    s = default_state(K)
    r1 = run_fast_path(views, s, K)
    r2 = run_fast_path(views, default_state(K), K)      # 全新构造的等价 state
    assert isinstance(r1, FastPathResult) and isinstance(r1, dict)      # 向后兼容 dict 语义
    assert set(r1) == set(views)
    for uid in views:
        c1, c2 = r1[uid], r2[uid]
        assert c1 is not None and c2 is not None
        assert c1.sha == c2.sha
        assert c1.features == c2.features               # bit 级（float 精确相等）
        # proxy_score 复算与 enriched 特征一致
        assert proxy_score(c1, views[uid]) == c1.features["proxy_score"]
    # 新增字段：realized pool size 逐 episode 落账（prereg §4/§6）
    assert set(r1.pool_stats) == set(views)
    for uid in views:
        st = r1.pool_stats[uid]
        assert st["realized_pool_size"] >= st["kept_pool_size"] >= 1
        assert st["abstained"] is False and st["n_banned"] == 0
        _, st_gen = generate_candidates_with_stats(uid, s, K)
        assert {k: st[k] for k in st_gen} == st_gen     # 与候选生成接口逐字段一致
    assert r1.pool_stats == r2.pool_stats               # 落账本身确定性


def test_proxy_score_toy_semantics_and_failure_sentinel():
    x = _views()["u0"]
    identity = make_candidate([("impute_linear", {})], "det")
    smooth = make_candidate([("impute_linear", {}), ("smooth_ma", {"window": 15})], "det")
    assert proxy_score(identity, x) == 1.0               # 无缺失 → 恒等 → RMSD=0
    assert proxy_score(smooth, x) < 1.0
    broken = make_candidate([("no_such_op", {})], "det")
    assert proxy_score(broken, x) == FAILED_PROXY


def test_prepared_artifact_semantics():
    x = _views()["u0"]
    # abstain（None）→ 原序列 float64 ravel 副本（不动数据；改副本不伤原序列）
    art0 = prepared_artifact(None, x)
    assert art0 is not None and np.array_equal(art0, np.asarray(x, float).ravel())
    art0[0] = 999.0
    assert x[0] != 999.0
    # 成功执行 → float64 产物；失败 → None 哨兵
    ident = make_candidate([("impute_linear", {})], "det")
    art1 = prepared_artifact(ident, x)
    assert art1 is not None and art1.dtype == np.float64 and art1.shape == (x.size,)
    broken = make_candidate([("no_such_op", {})], "det")
    assert prepared_artifact(broken, x) is None


def test_risk_rule_matches_and_apply_risk():
    views = _views()
    fp0, fp1 = toy_fingerprint(views["u0"]), toy_fingerprint(views["u1"])
    assert set(fp0) == set(fp1) == {"snr", "missing_rate"}   # 只发 allowlist 键（prereg §3.3）
    assert fp0["missing_rate"] == 0.0 and fp1["missing_rate"] > 0.0      # u1 才有缺失
    assert fp0["snr"] > 0.0
    rule = _rule("miss", feature="missing_rate", op=">", value=0.0, target="winsorize")
    assert not rule.matches(fp0) and rule.matches(fp1)
    # allowlist 内但指纹缺失的特征（P_FEATS 之 acf1）→ 不触发（确定性 False）
    assert not _rule("acf", feature="acf1", op=">", value=-1e9).matches(fp0)
    s = apply_edit(default_state(K), RiskRulePatch(rule))
    pool = generate_candidates("u1", s, K)
    kept, banned = apply_risk(pool, fp1, s)
    assert {b["sha"] for b in banned} == {c.sha for c in pool if "winsorize" in c.op_names()}
    assert all("winsorize" not in c.op_names() for c in kept)
    assert all(b["rule_id"] == "miss" for b in banned)
    kept0, banned0 = apply_risk(pool, fp0, s)            # u0 指纹不触发 → 全保留
    assert banned0 == [] and [c.sha for c in kept0] == [c.sha for c in pool]


# ============================== C2. when 条件列表 = 同规则 AND（prereg §4 冻结语义） ==============================
def test_risk_rule_when_list_and_semantics():
    """when 条件列表 = 同规则 AND：**全部**条件成立才触发；只满足其一不触发；
    任一条件的 feature 缺失 → 该条件不成立（保守方向，不 ban）。"""
    rule = _rule_and("and2", [{"feature": "snr", "op": ">=", "value": 1.0},
                              {"feature": "missing_rate", "op": ">", "value": 0.1}])
    rule.validate()
    assert rule.matches({"snr": 2.0, "missing_rate": 0.5})          # 两条件都满足 → 触发
    assert not rule.matches({"snr": 0.5, "missing_rate": 0.5})      # 只满足第二个 → 不触发
    assert not rule.matches({"snr": 2.0, "missing_rate": 0.0})      # 只满足第一个 → 不触发
    assert not rule.matches({"snr": 0.5, "missing_rate": 0.0})
    assert not rule.matches({"snr": 2.0})                           # 缺特征条件不成立（保守）
    assert not rule.matches({})
    # 端到端：apply_risk 只在两条件同时成立的指纹上 ban
    s = apply_edit(default_state(K), RiskRulePatch(rule))
    pool = generate_candidates("u0", s, K)
    with_winsor = {c.sha for c in pool if "winsorize" in c.op_names()}
    assert with_winsor                                              # det 阶梯 2 号程序保证非空
    _, banned_hit = apply_risk(pool, {"snr": 2.0, "missing_rate": 0.5}, s)
    assert {b["sha"] for b in banned_hit} == with_winsor
    for fp in ({"snr": 0.5, "missing_rate": 0.5}, {"snr": 2.0, "missing_rate": 0.0},
               {"snr": 2.0}, {}):
        kept, banned = apply_risk(pool, fp, s)
        assert banned == [] and [c.sha for c in kept] == [c.sha for c in pool]


def test_bin_scope_out_of_bin_episodes_never_banned():
    """回归测试（本缺陷的构造性反例）：bin scope lo≤f<hi = 同一规则 when 内两原子合取。
    旧实现把两原子拆成两条独立规则 → apply_risk 规则间并集(OR)下任何取值都命中至少
    一边（f<lo 命中 "<hi"、f≥hi 命中 "≥lo"）→ bin ban 实际等价全局 ban。修复后：
    bin 外（上下两侧）episode 绝不被 ban；仅 bin 内被 ban；左闭右开边界成立。"""
    lo, hi = 1.0, 2.0
    bin_rule = _rule_and("bin", [{"feature": "snr", "op": ">=", "value": lo},
                                 {"feature": "snr", "op": "<", "value": hi}])
    s = apply_edit(default_state(K), RiskRulePatch(bin_rule))
    pool = generate_candidates("u0", s, K)
    with_winsor = {c.sha for c in pool if "winsorize" in c.op_names()}
    assert with_winsor
    below, inside, above = {"snr": 0.5}, {"snr": 1.5}, {"snr": 5.0}
    # bin 内：正常 ban（正控制）
    _, banned_in = apply_risk(pool, inside, s)
    assert {b["sha"] for b in banned_in} == with_winsor
    # bin 外两侧：绝不 ban（旧 OR 拆分下 below 命中 "<hi"、above 命中 "≥lo" → 全被 ban）
    for fp in (below, above):
        kept, banned = apply_risk(pool, fp, s)
        assert banned == []
        assert [c.sha for c in kept] == [c.sha for c in pool]
    # 左闭右开：f == lo 触发、f == hi 不触发
    _, banned_lo = apply_risk(pool, {"snr": lo}, s)
    assert {b["sha"] for b in banned_lo} == with_winsor
    _, banned_hi = apply_risk(pool, {"snr": hi}, s)
    assert banned_hi == []
    # 对照（文档化被禁止的近似）：同两原子拆成两条独立规则 → 规则间并集使 bin 外两侧
    # 也全被 ban——正是 prereg §4 when 语义段禁止拆多规则的原因。
    s_split = apply_edit(default_state(K),
                         RiskRulePatch(_rule("split_lo", feature="snr", op=">=", value=lo)))
    s_split = apply_edit(s_split,
                         RiskRulePatch(_rule("split_hi", feature="snr", op="<", value=hi)))
    for fp in (below, above):
        _, banned_split = apply_risk(pool, fp, s_split)
        assert {b["sha"] for b in banned_split} == with_winsor      # 并集语义 = 全局 ban
    # 同一 fingerprint 下合取规则（修复版）与拆分版行为可区分 → 缺陷不可能悄悄回归
    _, banned_fixed = apply_risk(pool, below, s)
    assert banned_fixed == []


def test_risk_rule_single_condition_backcompat_and_canonical_sha():
    """单条件 dict（向后兼容）规范化为单元素列表；canonical 序列化（to_dict/sha）统一
    列表形 → dict 形与 [dict] 形构造的规则、装入 state 后的语义 sha 完全一致。"""
    d_form = _rule("bc", feature="snr", op=">", value=1.0)
    l_form = RiskRuleSpec(rule_id="bc",
                          when=[{"feature": "snr", "op": ">", "value": 1.0}],
                          then={"action": "ban", "target": "winsorize"})
    assert d_form.when == ({"feature": "snr", "op": ">", "value": 1.0},)   # 规范化为元组
    assert d_form == l_form
    d = d_form.to_dict()
    assert isinstance(d["when"], list) and len(d["when"]) == 1             # canonical = 列表形
    assert d == l_form.to_dict()
    # from_dict 双形态往返：legacy 单条件 dict 与新列表形都回到同一规则
    legacy = {"rule_id": "bc", "when": {"feature": "snr", "op": ">", "value": 1.0},
              "then": {"action": "ban", "target": "winsorize"}}
    assert RiskRuleSpec.from_dict(legacy) == d_form
    assert RiskRuleSpec.from_dict(d) == d_form
    # state 语义身份：两种构造形状 → 同一 sha（canonical 列表形是唯一序列化口径）
    sA = apply_edit(default_state(K), RiskRulePatch(d_form))
    sB = apply_edit(default_state(K), RiskRulePatch(l_form))
    assert sA.sha() == sB.sha()
    # 单条件匹配语义与修复前一致
    assert d_form.matches({"snr": 2.0}) and not d_form.matches({"snr": 0.5})
    assert not d_form.matches({})                                          # 缺特征不触发


def test_risk_rule_empty_when_raises_everywhere():
    """空条件列表 = 无条件全局 ban，全链路拒绝：validate raise / apply_edit 拒 /
    state 构造拒 / 走私进 state 后 apply_risk 第二道闸 raise / matches 兜底 False。"""
    empty = RiskRuleSpec(rule_id="empty", when=[],
                         then={"action": "ban", "target": "winsorize"})
    assert empty.when == ()
    with pytest.raises(ValueError, match="不能为空"):
        empty.validate()
    s0 = default_state(K)
    with pytest.raises(P6EditError):
        apply_edit(s0, RiskRulePatch(empty))
    with pytest.raises(ValueError):
        P6HarnessState(version="v0", selector=s0.selector, sampler=s0.sampler,
                       risk_rules=(empty,))
    assert not empty.matches({"snr": 1.0})               # 空合取不数学化为"真"（不 ban）
    s_bad = default_state(K)
    object.__setattr__(s_bad, "risk_rules", (empty,))    # 测试专用后门
    pool = generate_candidates("u0", s_bad, K)
    with pytest.raises(ValueError, match="为空"):
        apply_risk(pool, {"snr": 1.0}, s_bad)
    # when=None 同样在 validate 处拒
    with pytest.raises(ValueError, match="不能为空"):
        RiskRuleSpec(rule_id="none", when=None,
                     then={"action": "ban", "target": "x"}).validate()
    # 非 dict 条件元素 → 构造期响亮拒绝
    with pytest.raises(ValueError, match="条件 dict"):
        RiskRuleSpec(rule_id="junk", when=[42], then={"action": "ban", "target": "x"})


def test_risk_rule_allowlist_enforced_per_condition_in_list():
    """列表形 when 的 allowlist **逐条**生效：任一条件 feature 违规即拒
    （validate + apply_edit + apply_risk 第二道闸；首条合法也救不了）。"""
    bad = RiskRuleSpec(rule_id="bad2",
                       when=[{"feature": "snr", "op": ">", "value": 0.0},
                             {"feature": "loss", "op": ">", "value": 0.0}],  # 第二条违规
                       then={"action": "ban", "target": "winsorize"})
    with pytest.raises(ValueError, match="allowlist"):
        bad.validate()
    with pytest.raises(P6EditError):
        apply_edit(default_state(K), RiskRulePatch(bad))
    s_bad = default_state(K)
    object.__setattr__(s_bad, "risk_rules", (bad,))      # 走私 → 第二道闸逐条件拦
    pool = generate_candidates("u0", s_bad, K)
    with pytest.raises(ValueError, match="allowlist"):
        apply_risk(pool, {"snr": -1.0, "loss": -1.0}, s_bad)   # 不触发也拦
    # 全 allowlist 条件列表可过 spec 校验
    _rule_and("ok2", [{"feature": "snr", "op": ">", "value": 0.0},
                      {"feature": "acf1", "op": "<", "value": 0.5}]).validate()


# ============================== D. 配对语义（capability matrix 第 4 项） ==============================
def test_paired_selector_pools_identical_chosen_differ():
    views = _views()
    sA = default_state(K)                                # proxy_rank
    sB = apply_edit(sA, SelectorPatch(_weighted()))      # 只差 selector（n_steps 线性打分）
    out = paired_selector_run(views, sA, sB, K)
    assert set(out["A"]) == set(views)
    for uid in views:
        assert len(out["pool_shas"][uid]) > 0
        cA, cB = out["A"][uid], out["B"][uid]
        assert cA is not None and cB is not None
        # 构造保证不同：proxy_rank 选恒等程序（proxy=1.0、1 步）；n_steps 打分选最长程序（≥3 步）
        assert cA.sha != cB.sha
        assert cA.features["n_steps"] < cB.features["n_steps"]


def test_paired_sampler_same_k_different_pools():
    views = _views()
    sA = default_state(K)                                # det3/random5
    sB = apply_edit(sA, SamplerPatch(
        SamplerSpec(allocation={"det": 1, "random": 7, "llm": 0}, expected_total=K)))
    out = paired_sampler_run(views, sA, sB, K)
    for uid in views:
        shas_a = {c.sha for c in out["A"]["pools"][uid]}
        shas_b = {c.sha for c in out["B"]["pools"][uid]}
        assert shas_a != shas_b                          # 池组成不同（被测变量）
        assert det_ladder()[1].sha in shas_a and det_ladder()[1].sha not in shas_b
    # 总 K 不同的两侧 → raise（须绕过 sha 检查：sampler 本身被排除，K 差异只能靠显式断言）
    sC = P6HarnessState(version="v0", selector=sA.selector,
                        sampler=SamplerSpec(allocation={"det": 2, "random": 2, "llm": 0},
                                            expected_total=4))
    with pytest.raises(P6PairingError):
        paired_sampler_run(views, sA, sC, K)


def test_paired_risk_out_of_scope_unchanged():
    views = _views()
    sA = default_state(K)
    sB = apply_edit(sA, RiskRulePatch(_rule("ban_winsor", target="winsorize")))  # 全 uid 触发
    out = paired_risk_run(views, sA, sB, K)
    for uid in views:
        banned_b = out["B"]["banned"][uid]
        assert len(banned_b) > 0                          # det 阶梯 2 号程序含 winsorize
        assert out["A"]["banned"][uid] == ()
        # 非 scope（不含 winsorize）候选：两侧 kept 序列完全一致
        keep_a = [c.sha for c in out["A"]["kept"][uid] if "winsorize" not in c.op_names()]
        keep_b = [c.sha for c in out["B"]["kept"][uid] if "winsorize" not in c.op_names()]
        assert keep_a == keep_b
        assert all("winsorize" not in c.op_names() for c in out["B"]["kept"][uid])
    # 规则在全部 episode 触发且命中 → 无作用域外 episode，字节级校验清单为空
    assert out["out_of_scope_verified"] == ()


def test_paired_risk_byte_level_verifies_out_of_scope_episodes():
    """prereg §4 门④：作用域外每个 episode 的两臂最终 prepared artifact 字节级一致
    （不只 kept-mask）。规则经 fingerprints 参数只对 u1 触发 → u0/u2 是作用域外 episode。"""
    views = _views()
    sA = default_state(K)
    sB = apply_edit(sA, RiskRulePatch(
        _rule("miss_hi", feature="missing_rate", op=">", value=0.5, target="winsorize")))
    fps = {"u1": {"missing_rate": 1.0}}                  # u0/u2 用默认 toy 指纹（0.0 → 不触发）
    out = paired_risk_run(views, sA, sB, K, fingerprints=fps)
    assert out["out_of_scope_verified"] == ("u0", "u2")
    assert len(out["B"]["banned"]["u1"]) > 0
    assert out["B"]["banned"]["u0"] == () and out["B"]["banned"]["u2"] == ()
    for uid in ("u0", "u2"):                             # 作用域外：两臂 chosen 也一致
        assert out["A"]["chosen"][uid].sha == out["B"]["chosen"][uid].sha


def test_paired_risk_conjunctive_bin_scope_out_of_scope_byte_verified():
    """门④ 在新 AND 语义下的作用域判定：合取 bin scope 规则只在 bin 内 uid 触发；
    bin 外（上下两侧）uid = 作用域外 episode → banned 空 + 两臂 prepared artifact
    字节级校验通过 + chosen 一致。旧 OR 拆分下 u1/u2 各命中一边原子 → 三个 uid 全
    in-scope、out_of_scope_verified 为空——本测试钉死修复后的作用域。"""
    views = _views()
    sA = default_state(K)
    bin_rule = _rule_and("bin_pair", [{"feature": "snr", "op": ">=", "value": 1.0},
                                      {"feature": "snr", "op": "<", "value": 2.0}])
    sB = apply_edit(sA, RiskRulePatch(bin_rule))
    fps = {"u0": {"snr": 1.5}, "u1": {"snr": 0.5}, "u2": {"snr": 5.0}}  # 内 / 下侧 / 上侧
    out = paired_risk_run(views, sA, sB, K, fingerprints=fps)
    assert len(out["B"]["banned"]["u0"]) > 0             # bin 内正常 ban（det 2 号含 winsorize）
    assert out["B"]["banned"]["u1"] == () and out["B"]["banned"]["u2"] == ()
    assert out["A"]["banned"]["u0"] == ()                # A 臂无规则
    assert out["out_of_scope_verified"] == ("u1", "u2")  # 作用域外字节级校验通过清单
    for uid in ("u1", "u2"):
        assert out["A"]["chosen"][uid].sha == out["B"]["chosen"][uid].sha


def test_paired_risk_artifact_byte_mismatch_raises(monkeypatch):
    """门④ 的 raise 路径：篡改 prepared_artifact 使 B 臂产物偏移 → 字节级不一致必须 raise。
    （不变量下两臂 artifact 按构造相等，违规只能来自执行面被篡改——正是端到端校验要抓的。）"""
    import SelfEvolvingHarnessTS.p6.fast_path as fp_mod

    views = _views()
    sA = default_state(K)
    sB = apply_edit(sA, RiskRulePatch(
        _rule("never", feature="missing_rate", op=">", value=0.5)))     # toy 指纹上永不触发
    real = fp_mod.prepared_artifact
    calls = {"n": 0}

    def tampered(chosen, series):
        calls["n"] += 1
        art = real(chosen, series)
        if calls["n"] % 2 == 0 and art is not None:      # 每 uid 第二次调用 = B 臂
            art = art + 1.0
        return art

    monkeypatch.setattr(fp_mod, "prepared_artifact", tampered)
    with pytest.raises(P6PairingError):
        fp_mod.paired_risk_run(views, sA, sB, K)


def test_abstain_when_all_candidates_banned():
    views = _views()
    s = apply_edit(default_state(K), SamplerPatch(
        SamplerSpec(allocation={"det": 8, "random": 0, "llm": 0}, expected_total=K)))
    s = apply_edit(s, RiskRulePatch(_rule("ban_all", target="impute_linear")))
    r = run_fast_path(views, s, K)                       # det 阶梯全程序以 impute_linear 开头
    assert all(r[uid] is None for uid in views)
    for uid in views:                                    # 落账同时可见：短缺 + 全禁 + abstain
        st = r.pool_stats[uid]
        assert st["det_shortfall"] == 5 and st["realized_pool_size"] == 3
        assert st["kept_pool_size"] == 0 and st["abstained"] is True
    assert select([], s) is None


def test_fingerprints_param_scopes_risk_per_uid():
    views = _views()
    s = apply_edit(default_state(K), SamplerPatch(
        SamplerSpec(allocation={"det": 8, "random": 0, "llm": 0}, expected_total=K)))
    s = apply_edit(s, RiskRulePatch(_rule("ban_all", target="impute_linear")))
    r = run_fast_path(views, s, K, fingerprints={"u0": {}})   # u0 指纹缺 missing_rate → 不触发
    assert r["u0"] is not None
    assert r["u1"] is None and r["u2"] is None


def test_paired_tamper_raises():
    views = _views()
    sA = default_state(K)
    sB = apply_edit(apply_edit(sA, SelectorPatch(_weighted())), SamplerPatch(
        SamplerSpec(allocation={"det": 1, "random": 7, "llm": 0}, expected_total=K)))
    with pytest.raises(P6PairingError):                   # selector 配对里偷偷改了 sampler
        paired_selector_run(views, sA, sB, K)
    with pytest.raises(P6PairingError):                   # sampler 配对里偷偷改了 selector
        paired_sampler_run(views, sA, sB, K)
    with pytest.raises(P6PairingError):                   # risk 配对里 selector+sampler 都不同
        paired_risk_run(views, sA, sB, K)
