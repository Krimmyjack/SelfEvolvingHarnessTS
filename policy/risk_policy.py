"""policy/risk_policy.py — 可执行风险策略层 + 部署路径焊接（Track B0，§13.2 / §13.4）。

现状（B0 前）：风险知识只以 **散文** 存在于 SkillSpec.risk，LLM 读它、确定性 router 不消费它；
慢路径 proposer 只能改被 overlay 绕过的 L1/L2 模板 → **知识断路**（§13.4）。

本模块把风险知识变成**可执行、版本化、带作用域**的规则，并提供 `RiskAwareRouterPolicy`——
它 **包装** 冻结臂（绝不改其权重），对其 pick 施加 scoped 覆盖规则，且**本身就是 RouterPolicy**，
因此作为 `routed_process_overlay(router=…)` 传入 = 部署路径真正消费编辑后的知识（§13.4 焊接）。

红线：
  · 冻结臂 load-only，覆盖发生在其**外层**（wrapper），config_sha 不变；
  · 覆盖目标动作必须 ∈ 冻结动作池（→ 保证 ∈ menu v1，overlay 编译不 KeyError）；
  · 规则必带 scope（作用域纪律，§13.2：防"对 S_season 真、对 S_both 误外推"全域裸用）；
  · 默认策略 = **空（pass-through）** = 现任纯冻结 router；蒸馏规则可用但不默认激活
    （F0 已证蒸馏规则会过泛化——激活是决策，非默认）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .router_policy import RoutingDecision, RouterPolicy
from .skills import SKILLS_V1
from .action_spec import ActionMenu

# 冻结动作池（SkillSpec 单射覆盖的 10 动作）——覆盖目标必须 ∈ 此集
POOL_ACTIONS: frozenset = frozenset(aid for s in SKILLS_V1.values() for aid in s.actions.values())
_OPS = ("force", "ban", "abstain")


def cell_bins(cell_id: str) -> Dict[str, Optional[str]]:
    """"forecast|snrLow|miss" → {snr: low|high, miss: none|some}。空/畸形 → 全 None（通配失败安全）。"""
    parts = (cell_id or "").split("|")
    if len(parts) < 3:
        return {"snr": None, "miss": None}
    return {"snr": "low" if "snrLow" in parts[1] else ("high" if "snrHigh" in parts[1] else None),
            "miss": "none" if parts[2] == "full" else ("some" if parts[2] else None)}


_CMP = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, "<": lambda a, b: a < b}


@dataclass(frozen=True)
class RiskRule:
    """一条 scoped 覆盖规则。

    when  触发条件（全部子条件须成立；缺特征 → 不触发，失败安全）：
      cell            {"snr": "low|high|None", "miss": "none|some|None"}（None=通配）
      feats           [{"name","op(>=|<=|>|<)","value"}, …]（对 struct_feats 逐条 AND）
      base_action_in  [action_id, …]：仅当冻结臂 pick ∈ 此集才触发（如"重 median 族"）
    then  覆盖动作：
      op=force    override pick → action（须 ∈ POOL_ACTIONS）
      op=ban      若 base pick ∈ base_action_in → override → action（替代动作，须 ∈ POOL）
      op=abstain  置 abstained=True、pick → fallback（action=None）
    scope 作用域标签（审计/纪律；须非空，如 "region:seasonal_strength>=0.3" / "worst_group:snrLow"）
    provenance {source: "distilled:F0"|"proposer:llm"|"proposer:enum", evidence, …}
    """
    rule_id: str
    when: Dict[str, Any]
    then: Dict[str, Any]
    scope: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def matches(self, struct: Dict[str, float], bins: Dict[str, Optional[str]],
                base_action: str) -> bool:
        w = self.when
        cell = w.get("cell")
        if cell:
            for k in ("snr", "miss"):
                if cell.get(k) is not None and bins.get(k) != cell[k]:
                    return False
        if w.get("base_action_in") is not None and base_action not in w["base_action_in"]:
            return False
        for pred in w.get("feats", []):
            v = struct.get(pred["name"])
            if v is None:                                    # 缺特征 → 不触发（失败安全）
                return False
            if not _CMP[pred["op"]](float(v), float(pred["value"])):
                return False
        return True

    def validate(self) -> Optional[str]:
        if not self.rule_id:
            return "rule_id 空"
        if not self.scope:
            return "规则必带 scope（作用域纪律，§13.2）"
        op = self.then.get("op")
        if op not in _OPS:
            return f"then.op 非法：{op!r}（须 ∈ {_OPS}）"
        act = self.then.get("action")
        if op in ("force", "ban"):
            if act not in POOL_ACTIONS:
                return f"then.action {act!r} 不 ∈ 冻结动作池（overlay 将 KeyError）"
            if op == "ban" and not self.when.get("base_action_in"):
                return "ban 规则须指定 base_action_in（被禁的 pick 集）"
        for pred in self.when.get("feats", []):
            if pred.get("op") not in _CMP or "name" not in pred or "value" not in pred:
                return f"feats 谓词畸形：{pred!r}"
        return None


@dataclass(frozen=True)
class RiskPolicy:
    version: str
    rules: Tuple[RiskRule, ...] = ()

    def sha(self) -> str:
        payload = [dict(rule_id=r.rule_id, when=r.when, then=r.then, scope=r.scope,
                        provenance=r.provenance) for r in self.rules]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def apply(self, base_action: str, base_abstain: bool, struct: Dict[str, float],
              cell_id: str) -> Tuple[str, bool, List[str], str]:
        """→ (action, abstained, fired_rule_ids, fallback_hint)。规则按序，首个触发的具体覆盖胜。"""
        bins = cell_bins(cell_id)
        for r in self.rules:
            if not r.matches(struct, bins, base_action):
                continue
            op = r.then["op"]
            if op == "abstain":
                return base_action, True, [r.rule_id], "abstain"
            if op == "ban":
                if base_action in (r.when.get("base_action_in") or []):
                    return r.then["action"], base_abstain, [r.rule_id], "ban"
                continue                                     # 未命中被禁集 → 该规则不改变
            if op == "force":
                return r.then["action"], base_abstain, [r.rule_id], "force"
        return base_action, base_abstain, [], "passthrough"


def risk_policy_v0() -> RiskPolicy:
    """B0 默认 incumbent = 空 pass-through = 现任纯冻结 router（不静默激活任何蒸馏规则）。"""
    return RiskPolicy(version="risk_v0_empty", rules=())


def risk_policy_distilled() -> RiskPolicy:
    """蒸馏基线（可选，非默认）：把 F0/S0.7 散文风险规则变可执行——供 proposer 种子与对照。
    阈值均为 **手设未标定**（Q5：跨域不可直接迁移；scope 记 region 观测谓词非族真标签）。"""
    HEAVY_MEDIAN = ["f0_median_w9", "f0_median_w15", "f0_median_w25"]
    return RiskPolicy(version="risk_v0_distilled", rules=(
        RiskRule(
            rule_id="F0_season_heavy_median_downgrade",
            when={"feats": [{"name": "seasonal_strength", "op": ">=", "value": 0.3}],
                  "base_action_in": HEAVY_MEDIAN},
            then={"op": "ban", "action": "v_median"},        # 重 median → 轻档 w5
            scope="region:seasonal_strength>=0.3",
            provenance={"source": "distilled:F0",
                        "evidence": "强季节族 median w9 起受伤、w25 重伤（窗≈周期抹平季节）",
                        "calibration": "手设阈值，未 nested 标定（Q5 迁移性未证）"}),
        RiskRule(
            rule_id="S07_stl_needs_period_evidence",
            when={"feats": [{"name": "seasonal_strength", "op": "<", "value": 0.15}],
                  "base_action_in": ["v_stl"]},
            then={"op": "ban", "action": "v_median"},        # 无周期证据用 stl → 退回稳健平滑
            scope="region:seasonal_strength<0.15",
            provenance={"source": "distilled:S0.7",
                        "evidence": "周期误检时 stl 退化为 garbage-period 激进平滑",
                        "calibration": "手设阈值，未 nested 标定"}),
    ))


class RiskAwareRouterPolicy(RouterPolicy):
    """冻结臂 + RiskPolicy 的**编译产物**；drop-in RouterPolicy → 传入 overlay 即焊接 §13.4。

    base_menu：冻结臂做兼容性核验用的原始菜单（默认 = 传入 predict 的 menu；当菜单被扩展时
    应显式给 v1 菜单，使冻结臂版本兼容核验不被 proposer 菜单编辑破坏——B0 暂只用 v1）。
    """

    def __init__(self, base: RouterPolicy, risk: RiskPolicy, base_menu: Optional[ActionMenu] = None):
        self.base = base
        self.risk = risk
        self.base_menu = base_menu

    def predict(self, conditioning_key: Dict[str, Any], action_menu: ActionMenu,
                model_menu: Optional[List[str]] = None) -> RoutingDecision:
        base_dec = self.base.predict(conditioning_key, self.base_menu or action_menu, model_menu)
        struct = dict(conditioning_key.get("pattern", {}).get("struct_feats", {}) or {})
        cell = str(conditioning_key.get("cell_id") or "")
        action, abstain, fired, hint = self.risk.apply(
            base_dec.action_id, base_dec.abstained, struct, cell)
        # 失败安全：覆盖目标须 ∈ 部署 menu，否则回退 base（overlay 编译不 KeyError）
        override_ok = fired and (action in action_menu or hint == "abstain")
        if fired and not override_ok:
            action, abstain, fired, hint = (base_dec.action_id, base_dec.abstained, [],
                                            "override_out_of_menu")
        prov = dict(base_dec.provenance)
        prov["risk_policy"] = {"version": self.risk.version, "sha": self.risk.sha(),
                               "base_action": base_dec.action_id, "fired": list(fired),
                               "resolution": hint}
        return RoutingDecision(action_id=action, abstained=abstain,
                               fallback_action=base_dec.fallback_action, provenance=prov)
