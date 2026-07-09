"""tests/test_edits.py — Track B0 编辑面守卫：scope 纪律、覆盖/passthrough、不可变+版本化、
冻结资产守卫、Memory 泄漏守卫、§13.4 焊接证明（RiskAwareRouterPolicy 是 drop-in RouterPolicy
且覆盖目标恒 ∈ menu v1 → overlay 编译不 KeyError）。全无网络/无 torch/无 joblib（mock 冻结臂）。"""
from dataclasses import replace

import pytest

from SelfEvolvingHarnessTS.policy.action_spec import action_menu_v1
from SelfEvolvingHarnessTS.policy.router_policy import RouterPolicy, RoutingDecision
from SelfEvolvingHarnessTS.policy.risk_policy import (RiskAwareRouterPolicy, RiskPolicy, RiskRule,
                                                      risk_policy_distilled, cell_bins)
from SelfEvolvingHarnessTS.policy.edits import (AddRiskRule, RetireRiskRule, MemoryWrite,
                                                ProposePatternSpecEdit, apply_edits, compile_bundle,
                                                bundle_v0, DeterministicProposer, LLMProposer)


class _MockBase(RouterPolicy):
    def __init__(self, action="v_none", abstain=False):
        self._a, self._ab = action, abstain

    def predict(self, key, menu, model_menu=None):
        return RoutingDecision(action_id=self._a, abstained=self._ab,
                               fallback_action="v_median", provenance={"arm": "mock"})


def _key(struct=None, cell="forecast|snrLow|miss"):
    return {"task": {"type": "forecast"}, "pattern": {"struct_feats": struct or {}}, "cell_id": cell}


def _force(action_in, action_out, feat_val=0.5):
    rule = RiskRule("r1", when={"feats": [{"name": "seasonal_strength", "op": ">=", "value": 0.3}]},
                    then={"op": "force", "action": action_out}, scope="region:seasonal>=0.3")
    pol = RiskAwareRouterPolicy(_MockBase(action_in), RiskPolicy("v1", (rule,)),
                                base_menu=action_menu_v1())
    return pol.predict(_key({"seasonal_strength": feat_val}), action_menu_v1())


# ── scope 纪律 ──
def test_riskrule_requires_scope():
    r = RiskRule("r", when={"feats": []}, then={"op": "force", "action": "v_median"}, scope="")
    assert r.validate() is not None and "scope" in r.validate()


def test_force_action_must_be_in_pool():
    r = RiskRule("r", when={}, then={"op": "force", "action": "not_a_real_action"}, scope="s")
    assert r.validate() is not None                       # 编辑期就挡住 overlay KeyError


def test_ban_requires_base_action_in():
    r = RiskRule("r", when={"feats": []}, then={"op": "ban", "action": "v_median"}, scope="s")
    assert r.validate() is not None


# ── RiskPolicy.apply 确定性 + 日志 ──
def test_riskpolicy_apply_deterministic_and_logs():
    pol = RiskPolicy("v", (RiskRule("r", {"feats": [{"name": "acf1", "op": ">", "value": 0.5}]},
                                    {"op": "force", "action": "v_stl"}, "s"),))
    o1 = pol.apply("v_none", False, {"acf1": 0.9}, "forecast|snrHigh|full")
    o2 = pol.apply("v_none", False, {"acf1": 0.9}, "forecast|snrHigh|full")
    assert o1 == o2 and o1[0] == "v_stl" and o1[2] == ["r"]
    # 缺特征 → 失败安全不触发
    o3 = pol.apply("v_none", False, {}, "forecast|snrHigh|full")
    assert o3[0] == "v_none" and o3[2] == []


# ── 覆盖 / passthrough ──
def test_force_override_and_passthrough():
    d = _force("v_none", "v_median")
    assert d.action_id == "v_median" and d.provenance["risk_policy"]["fired"] == ["r1"]
    assert d.provenance["risk_policy"]["base_action"] == "v_none"
    d2 = _force("v_none", "v_median", feat_val=0.1)        # 不满足 region → passthrough
    assert d2.action_id == "v_none" and d2.provenance["risk_policy"]["fired"] == []
    assert d2.provenance["arm"] == "mock"                  # base provenance 保留


def test_ban_downgrade_and_scope_miss():
    rule = RiskRule("b1", when={"feats": [{"name": "seasonal_strength", "op": ">=", "value": 0.3}],
                                "base_action_in": ["f0_median_w25"]},
                    then={"op": "ban", "action": "v_median"}, scope="region:x")
    rp = RiskPolicy("v", (rule,))
    pol = RiskAwareRouterPolicy(_MockBase("f0_median_w25"), rp, base_menu=action_menu_v1())
    assert pol.predict(_key({"seasonal_strength": 0.5}), action_menu_v1()).action_id == "v_median"
    pol2 = RiskAwareRouterPolicy(_MockBase("v_stl"), rp, base_menu=action_menu_v1())   # pick 不在被禁集
    assert pol2.predict(_key({"seasonal_strength": 0.5}), action_menu_v1()).action_id == "v_stl"


def test_abstain_rule():
    rule = RiskRule("a1", when={"cell": {"snr": "low"}}, then={"op": "abstain", "action": None},
                    scope="worst_group:snrLow")
    pol = RiskAwareRouterPolicy(_MockBase("v_median"), RiskPolicy("v", (rule,)),
                                base_menu=action_menu_v1())
    d = pol.predict(_key({}, cell="forecast|snrLow|miss"), action_menu_v1())
    assert d.abstained is True and d.provenance["risk_policy"]["resolution"] == "abstain"


def test_cell_bins_parse():
    assert cell_bins("forecast|snrLow|miss") == {"snr": "low", "miss": "some"}
    assert cell_bins("forecast|snrHigh|full") == {"snr": "high", "miss": "none"}
    assert cell_bins("") == {"snr": None, "miss": None}


# ── apply_edits 不可变 + 版本化 ──
def test_apply_edits_immutable_and_versioned():
    b0 = bundle_v0()
    rule = RiskRule("r1", when={"feats": [{"name": "acf1", "op": ">", "value": 0.5}]},
                    then={"op": "force", "action": "v_stl"}, scope="s")
    b1, log = apply_edits(b0, [AddRiskRule(rule)])
    assert log[0]["applied"] is True and b1.sha() != b0.sha()
    assert len(b0.risk.rules) == 0 and len(b1.risk.rules) == 1     # 入参不变
    assert b1.parent_sha == b0.sha()
    b2, _ = apply_edits(b1, [RetireRiskRule("r1")])
    assert len(b2.risk.rules) == 0
    # 重复 rule_id 拒绝
    _, log3 = apply_edits(b1, [AddRiskRule(rule)])
    assert log3[0]["applied"] is False and "重复" in log3[0]["reason"]


# ── 冻结资产守卫 ──
def test_frozen_pattern_spec_guard():
    b0 = bundle_v0()
    _, log = apply_edits(b0, [ProposePatternSpecEdit("P0", {"x": 1})])
    assert log[0]["applied"] is False and "P0" in log[0]["reason"]
    b2, log2 = apply_edits(b0, [ProposePatternSpecEdit("P1z", {"add_feat": "foo"})])
    assert log2[0]["applied"] is True
    assert b2.pattern_spec_version == "P0" and len(b2.pattern_spec_proposals) == 1   # 仍 live P0
    with pytest.raises(NotImplementedError):
        compile_bundle(replace(b2, pattern_spec_version="P1z"), _MockBase())


# ── Memory 泄漏守卫 ──
def test_memory_write_leakage_guard():
    ok = {"pattern_region": "seasonal_strength>=0.3", "action": "v_median",
          "grounded_utility": 0.1, "utility_ci": [0.05, 0.15], "scope": "region:x",
          "source": "proposer:enum"}
    _, log = apply_edits(bundle_v0(), [MemoryWrite(ok)])
    assert log[0]["applied"] is True
    bad = dict(ok, future=[1, 2, 3])
    _, log2 = apply_edits(bundle_v0(), [MemoryWrite(bad)])
    assert log2[0]["applied"] is False and "泄漏" in log2[0]["reason"]
    _, log3 = apply_edits(bundle_v0(), [MemoryWrite({"action": "v_median"})])   # 缺字段
    assert log3[0]["applied"] is False


# ── §13.4 焊接证明 ──
def test_weld_dropin_routerpolicy_and_menu_safe():
    d = _force("v_none", "f0_median_w25")
    assert d.action_id == "f0_median_w25"
    menu = action_menu_v1()
    assert d.action_id in menu                             # overlay program_from_spec 不 KeyError
    rule = RiskRule("r1", when={"feats": [{"name": "seasonal_strength", "op": ">=", "value": 0.3}]},
                    then={"op": "force", "action": "f0_median_w25"}, scope="s")
    pol = compile_bundle(replace(bundle_v0(), risk=RiskPolicy("v", (rule,))), _MockBase("v_none"),
                         base_menu=menu)
    assert isinstance(pol, RouterPolicy)                   # drop-in → overlay(router=pol)
    # 全池覆盖目标恒 ∈ menu v1（焊接安全性）
    for aid in ["v_none", "v_median", "f0_median_w9", "f0_median_w15", "f0_median_w25",
                "v_savgol", "v_stl", "v_wavelet", "v_winsor", "v_winsor_savgol"]:
        assert aid in menu


# ── distilled 基线可执行（散文→可执行）──
def test_distilled_policy_executes():
    rp = risk_policy_distilled()
    assert rp.sha() and len(rp.rules) >= 2
    # 强季节 + 重 median → 降档（F0 蒸馏规则）
    out = rp.apply("f0_median_w25", False, {"seasonal_strength": 0.6}, "forecast|snrHigh|full")
    assert out[0] == "v_median" and out[2]


# ── Proposer 接口 ──
def test_deterministic_proposer_and_llm_stub():
    ops = DeterministicProposer().propose({"grid": [
        {"feat": "seasonal_strength", "op": ">=", "value": 0.3,
         "ban_actions": ["f0_median_w25"], "replacement": "v_median"}]})
    assert len(ops) == 1 and isinstance(ops[0], AddRiskRule)
    _, log = apply_edits(bundle_v0(), ops)
    assert log[0]["applied"] is True
    with pytest.raises(NotImplementedError):
        LLMProposer(llm=None).propose({})
