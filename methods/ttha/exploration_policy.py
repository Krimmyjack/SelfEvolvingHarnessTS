"""Stage 3 pilot surface: exploration/allocation policy (Part 0 参数化).

冻结中立:DEFAULT 精确复现参数化前行为;非 DEFAULT 只能由 pilot runner 显式
安装(install_policy),运行结束必须 reset_policy()。

G3 硬边界:双门、容量门、harm 阈、越权守卫、隔离守卫、阶梯 v2 门槛、Scope
匹配语义均不在本面内;本模块不得被上述任何路径读取。合法域 = Random-legal-edit
臂采样空间(s3_part0_wiring_audit)。probe 总预算不属于本面(不可增减)。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

LEGAL_DOMAINS: dict[str, tuple] = {
    "skill_slot_merge_rule": (
        "draft_does_not_displace_agent",  # default
        "supply_then_agent",
        "agent_then_supply",
        "interleave_one_each",
    ),
    "supply_reserved_probe_slots": (0, 1),
    "probe_order_rule": (
        "chosen_first_then_pool",  # default
        "pool_as_built",
        "supply_first_then_agent",
        "agent_first_then_supply",
    ),
    "first_positive_stop": (True, False),
    "winner_compare_rule": (
        "first_positive_in_probe_order",  # default
        "max_support_gain_among_probed_positive",
    ),
    "tie_break_rule": ("probe_order", "prefer_self_proposed", "prefer_supplied"),
    "agent_proposals_kept": (1, 2),
    "displacement_margin": (0.0, 0.01, 0.05),
}


@dataclass(frozen=True)
class ExplorationPolicy:
    skill_slot_merge_rule: str = "draft_does_not_displace_agent"
    supply_reserved_probe_slots: int = 0
    probe_order_rule: str = "chosen_first_then_pool"
    first_positive_stop: bool = True
    winner_compare_rule: str = "first_positive_in_probe_order"
    tie_break_rule: str = "probe_order"
    agent_proposals_kept: int = 1
    displacement_margin: float = 0.0

    def validate(self) -> "ExplorationPolicy":
        for field, domain in LEGAL_DOMAINS.items():
            value = getattr(self, field)
            if value not in domain:
                raise ValueError(
                    "illegal policy value %s=%r (legal: %r)"
                    % (field, value, domain))
        return self

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = ExplorationPolicy()
_active: ExplorationPolicy = DEFAULT


def active_policy() -> ExplorationPolicy:
    return _active


def install_policy(policy: ExplorationPolicy) -> ExplorationPolicy:
    global _active
    _active = policy.validate()
    return _active


def reset_policy() -> ExplorationPolicy:
    global _active
    _active = DEFAULT
    return _active


def is_supplied_candidate(candidate_id: object) -> bool:
    """机械供给候选(cand_skill_*)判别;与 runner 侧 _candidate_sources 同口径。"""
    return str(candidate_id).startswith("cand_skill_")
