"""p6/edit_surfaces.py — P6 三个可进化 edit surface（typed proposal + compiler）。

capability matrix 第 1/2 项：类型化提案（EditOp）与编译器（compile_proposal）。
三个 EditOp，各对应 P6HarnessState 的一个语义组件：

  SelectorPatch(new_selector)   整体替换 selector（拒未知 kind / 未知特征名）；
  SamplerPatch(new_sampler)     整体替换 sampler（**拒改变总 K**：新 spec 自身 Σ=expected_total
                                且 expected_total 必须 == 现任 state 的 expected_total，预算冻结）；
  RiskRulePatch(add_rule)       追加一条风险规则（拒重复 rule_id；规则本身 spec 级校验）。

契约（与 harness_state 的两套 validate 约定一致）：
  validate(state) -> Optional[str]   不合法返回拒因字符串（None = 通过）；spec 级 ValueError
                                     在此被捕获并降格为拒因（apply_edit 再升格为 P6EditError）。
  apply(state) -> P6HarnessState     纯组件替换（dataclasses.replace；不动 version/parent_sha/
                                     edit_log——版本化由 harness_state.apply_edit 统一负责）。
  to_dict() / from_dict(d)           typed dict 往返（from_dict(op.to_dict()) == op）。

compile_proposal(proposal_dict) -> Optional[EditOp]：按 "kind" 字段分发编译；
**未知 kind 返回 None**（对齐 legacy promotion 语义：不可编译的提案不晋级、不 raise）。
已知 kind 但 payload 缺键 → raise（KeyError：坏 payload 必须响亮）。

红线：只 import 本包 harness_state（不 import 不修改 legacy policy/edits.py）；stdlib only。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional

from .harness_state import (
    P6HarnessState,
    RiskRuleSpec,
    SamplerSpec,
    SelectorSpec,
)

__all__ = [
    "EDIT_KINDS",
    "P6EditOp",
    "RiskRulePatch",
    "SamplerPatch",
    "SelectorPatch",
    "compile_proposal",
]


class P6EditOp:
    """EditOp 抽象基：validate(state)->Optional[str]、apply(state)->new state、to_dict/from_dict。"""

    kind = "p6_edit"

    def validate(self, state: P6HarnessState) -> Optional[str]:
        raise NotImplementedError

    def apply(self, state: P6HarnessState) -> P6HarnessState:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "P6EditOp":
        raise NotImplementedError


def _spec_reason(spec: Any) -> Optional[str]:
    """spec 级 validate()（raise 契约）→ 拒因字符串（EditOp 契约）。"""
    try:
        spec.validate()
    except ValueError as e:
        return str(e)
    return None


# ════════════════════════════ SelectorPatch ════════════════════════════
@dataclass(frozen=True)
class SelectorPatch(P6EditOp):
    """整体替换 selector。拒：未知 kind、未知特征名（经 SelectorSpec.validate）。"""

    new_selector: SelectorSpec
    kind = "selector_patch"

    def validate(self, state: P6HarnessState) -> Optional[str]:
        return _spec_reason(self.new_selector)

    def apply(self, state: P6HarnessState) -> P6HarnessState:
        return replace(state, selector=self.new_selector)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "new_selector": self.new_selector.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SelectorPatch":
        return cls(new_selector=SelectorSpec.from_dict(d["new_selector"]))


# ════════════════════════════ SamplerPatch ════════════════════════════
@dataclass(frozen=True)
class SamplerPatch(P6EditOp):
    """整体替换 sampler。拒改变总 K：新 spec 自身 Σ allocation == 其 expected_total（spec 级），
    且新 expected_total == 现任 state.sampler.expected_total（总预算是跨编辑冻结常量）。"""

    new_sampler: SamplerSpec
    kind = "sampler_patch"

    def validate(self, state: P6HarnessState) -> Optional[str]:
        reason = _spec_reason(self.new_sampler)
        if reason is not None:
            return reason
        if self.new_sampler.expected_total != state.sampler.expected_total:
            return (
                f"总 K 漂移被拒：expected_total {state.sampler.expected_total} → "
                f"{self.new_sampler.expected_total}（总预算是冻结常量，编辑只能重分配）"
            )
        return None

    def apply(self, state: P6HarnessState) -> P6HarnessState:
        return replace(state, sampler=self.new_sampler)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "new_sampler": self.new_sampler.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SamplerPatch":
        return cls(new_sampler=SamplerSpec.from_dict(d["new_sampler"]))


# ════════════════════════════ RiskRulePatch ════════════════════════════
@dataclass(frozen=True)
class RiskRulePatch(P6EditOp):
    """追加一条风险规则。拒：规则 spec 级不合法、rule_id 与现任 state 重复。"""

    add_rule: RiskRuleSpec
    kind = "risk_rule_patch"

    def validate(self, state: P6HarnessState) -> Optional[str]:
        reason = _spec_reason(self.add_rule)
        if reason is not None:
            return reason
        if any(r.rule_id == self.add_rule.rule_id for r in state.risk_rules):
            return f"重复 rule_id：{self.add_rule.rule_id!r}"
        return None

    def apply(self, state: P6HarnessState) -> P6HarnessState:
        return replace(state, risk_rules=state.risk_rules + (self.add_rule,))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "add_rule": self.add_rule.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RiskRulePatch":
        return cls(add_rule=RiskRuleSpec.from_dict(d["add_rule"]))


# ════════════════════════════ compiler ════════════════════════════
_EDIT_CLASSES = (SelectorPatch, SamplerPatch, RiskRulePatch)
EDIT_KINDS: Dict[str, type] = {c.kind: c for c in _EDIT_CLASSES}


def compile_proposal(proposal_dict: Mapping[str, Any]) -> Optional[P6EditOp]:
    """typed dict → EditOp（按 "kind" 分发）。未知 kind → None（legacy promotion 语义）；
    已知 kind 缺 payload 键 → raise KeyError（坏提案必须响亮）。"""
    cls = EDIT_KINDS.get(proposal_dict.get("kind"))
    if cls is None:
        return None
    return cls.from_dict(proposal_dict)
