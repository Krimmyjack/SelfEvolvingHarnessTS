"""policy/edits.py — 最小真实结构编辑面 + Proposer 接口（Track B0，§13.2 / §13.4）。

现状（B0 前）：proposer 只能改 L1/L2 静态模板 / L4 配置——全在被 overlay 绕过的路径上（§13.4
知识断路）。本模块开放**部署路径真正消费**的编辑面，且每次编辑产出**新版本化对象**（绝不原地
改冻结资产）：

  PolicyBundle   可编辑部署知识束：RiskPolicy（over v1 菜单的 scoped 覆盖规则）+ PatternSpec 版本
                 （P0 冻结；非-P0 = 提案，需提取器，B0 外）+ Memory 写入（staged，D7-lite 校验）。
  EditOp         类型化编辑：validate(bundle)→拒因|None，apply(bundle)→新 bundle。
  apply_edits    逐 op 校验→应用，版本/parent_sha 逐步戳记；不可变（绝不改入参）。
  compile_bundle PolicyBundle + 冻结臂 → RiskAwareRouterPolicy（= overlay 消费面，§13.4 焊接）。
  Proposer       propose(context)→[EditOp]。DeterministicProposer=枚举基线（§13.2 arm①）；
                 LLMProposer=B1 主擂台占位（天花板模型，per-family mining → EditOp）。

红线：config_sha 不变（P0 提取器不动）；冻结臂 load-only（编辑在 wrapper 外层）；风险规则必带
scope；Memory 写入禁 future/label/utility 泄漏字段。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

from .risk_policy import (RiskAwareRouterPolicy, RiskPolicy, RiskRule, POOL_ACTIONS,
                          risk_policy_v0)
from .router_policy import RouterPolicy
from .action_spec import ActionMenu

# Memory 写入泄漏黑名单（D7-lite；完整 D7 schema = Track C1）
_MEM_LEAK_KEYS = frozenset({"future", "label", "target", "holdout", "l_test", "test_loss", "oracle"})
_MEM_REQUIRED = ("pattern_region", "action", "grounded_utility", "utility_ci", "scope", "source")


# ════════════════════════════ 可编辑状态束 ════════════════════════════
@dataclass(frozen=True)
class PolicyBundle:
    version: str
    risk: RiskPolicy
    pattern_spec_version: str = "P0"                    # 非-P0 = 需提取器（B0 外），仅记提案
    pattern_spec_proposals: Tuple[dict, ...] = ()
    memory_writes: Tuple[dict, ...] = ()
    menu_version: str = "v1"                            # v1 已含全 median 剂量网格；菜单编辑 B0 缓办
    parent_sha: Optional[str] = None

    def sha(self) -> str:
        payload = dict(version=self.version, risk=self.risk.sha(),
                       pattern_spec_version=self.pattern_spec_version,
                       pattern_spec_proposals=list(self.pattern_spec_proposals),
                       memory_writes=list(self.memory_writes), menu_version=self.menu_version)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def bundle_v0() -> PolicyBundle:
    """B0 incumbent 束 = 空 RiskPolicy + P0 + 无 memory 写入 = 现任纯冻结部署。"""
    return PolicyBundle(version="bundle_v0", risk=risk_policy_v0())


# ════════════════════════════ 编辑算子 ════════════════════════════
class EditOp:
    kind = "edit"

    def validate(self, b: PolicyBundle) -> Optional[str]:
        raise NotImplementedError

    def apply(self, b: PolicyBundle) -> PolicyBundle:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class AddRiskRule(EditOp):
    kind = "add_risk_rule"
    rule: RiskRule

    def validate(self, b: PolicyBundle) -> Optional[str]:
        reason = self.rule.validate()
        if reason:
            return reason
        if any(r.rule_id == self.rule.rule_id for r in b.risk.rules):
            return f"rule_id 重复：{self.rule.rule_id!r}"
        return None

    def apply(self, b: PolicyBundle) -> PolicyBundle:
        new_risk = RiskPolicy(version=f"{b.risk.version}.r{len(b.risk.rules) + 1}",
                              rules=b.risk.rules + (self.rule,))
        return replace(b, risk=new_risk)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "rule_id": self.rule.rule_id, "scope": self.rule.scope,
                "then": self.rule.then}


@dataclass(frozen=True)
class RetireRiskRule(EditOp):
    kind = "retire_risk_rule"
    rule_id: str

    def validate(self, b: PolicyBundle) -> Optional[str]:
        if not any(r.rule_id == self.rule_id for r in b.risk.rules):
            return f"rule_id 不存在：{self.rule_id!r}"
        return None

    def apply(self, b: PolicyBundle) -> PolicyBundle:
        kept = tuple(r for r in b.risk.rules if r.rule_id != self.rule_id)
        new_risk = RiskPolicy(version=f"{b.risk.version}.retire", rules=kept)
        return replace(b, risk=new_risk)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "rule_id": self.rule_id}


@dataclass(frozen=True)
class MemoryWrite(EditOp):
    """Memory-knowledge 写入口（landing on EvidenceStore 消费面）。D7-lite 校验；完整 D7=C1。"""
    kind = "memory_write"
    evidence: Dict[str, Any]

    def validate(self, b: PolicyBundle) -> Optional[str]:
        missing = [k for k in _MEM_REQUIRED if k not in self.evidence]
        if missing:
            return f"Memory 记录缺必备字段：{missing}"
        leak = [k for k in self.evidence if k.lower() in _MEM_LEAK_KEYS]
        if leak:
            return f"Memory 记录含泄漏字段（禁 future/label/utility 原值）：{leak}"
        if self.evidence.get("action") not in POOL_ACTIONS:
            return f"action {self.evidence.get('action')!r} 不 ∈ 冻结动作池"
        return None

    def apply(self, b: PolicyBundle) -> PolicyBundle:
        return replace(b, memory_writes=b.memory_writes + (dict(self.evidence),))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "pattern_region": self.evidence.get("pattern_region"),
                "action": self.evidence.get("action")}


@dataclass(frozen=True)
class ProposePatternSpecEdit(EditOp):
    """PatternSpec 结构提案（self-evolving PatternSpec 终态地基）。P0 禁原地改；非-P0 需提取器
    → B0 只**记录提案**（deferred），不置为 live spec（pattern_spec_version 仍 P0）。"""
    kind = "propose_pattern_spec"
    new_version: str
    spec_delta: Dict[str, Any]

    def validate(self, b: PolicyBundle) -> Optional[str]:
        if self.new_version == "P0":
            return "禁止原地改 P0（config_sha 钉死）——新特征须新版本名"
        if not self.spec_delta:
            return "spec_delta 空"
        return None

    def apply(self, b: PolicyBundle) -> PolicyBundle:
        prop = {"new_version": self.new_version, "spec_delta": self.spec_delta,
                "status": "deferred_needs_extractor"}
        return replace(b, pattern_spec_proposals=b.pattern_spec_proposals + (prop,))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "new_version": self.new_version}


# ════════════════════════════ 应用 + 编译 ════════════════════════════
def apply_edits(bundle: PolicyBundle, ops: List[EditOp]) -> Tuple[PolicyBundle, List[dict]]:
    """逐 op 校验→应用；拒绝的 op 记 log 跳过。不可变：绝不改 bundle 入参。"""
    cur = bundle
    log: List[dict] = []
    n_applied = 0
    for op in ops:
        reason = op.validate(cur)
        if reason:
            log.append({"op": op.to_dict(), "applied": False, "reason": reason})
            continue
        nxt = op.apply(cur)
        n_applied += 1
        nxt = replace(nxt, version=f"{bundle.version}.e{n_applied}", parent_sha=cur.sha())
        log.append({"op": op.to_dict(), "applied": True,
                    "new_version": nxt.version, "new_sha": nxt.sha()})
        cur = nxt
    return cur, log


def compile_bundle(bundle: PolicyBundle, base_router: RouterPolicy,
                   base_menu: Optional[ActionMenu] = None) -> RiskAwareRouterPolicy:
    """PolicyBundle + 冻结臂 → RiskAwareRouterPolicy（drop-in RouterPolicy → overlay 消费面）。
    §13.4 焊接点：编辑后的风险知识经此进入 routed_process_overlay(router=…)。"""
    if bundle.pattern_spec_version != "P0":
        raise NotImplementedError(
            "非-P0 PatternSpec 需注册提取器（self-evolving PatternSpec 终态，B0 外）")
    return RiskAwareRouterPolicy(base_router, bundle.risk, base_menu=base_menu)


# ════════════════════════════ Proposer 接口（B1 主擂台） ════════════════════════════
class Proposer:
    """propose(context)→[EditOp]。context = per-family mining 报告 + 响应面摘要 + 现任束。"""

    def propose(self, context: Dict[str, Any]) -> List[EditOp]:
        raise NotImplementedError


class DeterministicProposer(Proposer):
    """枚举/搜索基线（§13.2 arm①，负对照的确定性对手）：从 context['grid'] 的
    (region 谓词 × ban 剂量) 组合确定性枚举 scoped AddRiskRule——无 LLM、无经验。"""

    def propose(self, context: Dict[str, Any]) -> List[EditOp]:
        ops: List[EditOp] = []
        grid = context.get("grid", [])
        for i, g in enumerate(grid):
            rule = RiskRule(
                rule_id=f"enum_{i}_{g['feat']}_{g['op']}_{g['value']}",
                when={"feats": [{"name": g["feat"], "op": g["op"], "value": g["value"]}],
                      "base_action_in": list(g["ban_actions"])},
                then={"op": "ban", "action": g["replacement"]},
                scope=f"region:{g['feat']}{g['op']}{g['value']}",
                provenance={"source": "proposer:enum"})
            ops.append(AddRiskRule(rule))
        return ops


class LLMProposer(Proposer):
    """B1 主擂台占位。天花板模型读 **per-family 分解** mining 报告（禁聚合，§13.2 纪律）→ 提
    EditOp；写入的风险规则须带 per-family/worst-group 作用域。B0 只留 hook，B1 预注册后接。"""

    def __init__(self, llm, prompt_version: str = "unset"):
        self.llm = llm
        self.prompt_version = prompt_version

    def propose(self, context: Dict[str, Any]) -> List[EditOp]:
        raise NotImplementedError(
            "B1：接天花板模型 + 认真 prompt（受控变量）；per-family mining 报告 → EditOp；"
            "作用域裁决防过泛化。见 prereg_proposer.md（待建）")
