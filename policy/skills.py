"""policy/skills.py — SkillSpec v1：ActionSpec 的参数放开层（Component Plan §12，
prereg_skill_slice.md §3）。

Tool/Skill/Program 三层的 Skill 位：SkillSpec = 冻结动作池上的**受约束处理策略**——
参数从定值放开为范围 + 适用条件 + 风险规则 + fallback + 版本。7 skill 恰好单射覆盖
10 动作（无新算子——供给不变，变的是抽象层）。

风险规则只含**已发表阶段结论的蒸馏知识**（F0 / S0.7 / classify C1——项目账本内的
确证事实），不含任何当前 gym 的 grounded outcome（prereg §3 红线）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SKILLS_VERSION = "skills_v1"


@dataclass(frozen=True)
class SkillSpec:
    name: str
    actions: Dict[str, str]          # param_key("‒"=无参) → action_id
    applicability: str               # 适用条件（描述性，供 composer/文档）
    risk: str                        # 风险规则（蒸馏知识，来源标注）
    fallback: str = "v_median"       # 编译失败时的池内回退动作
    version: str = SKILLS_VERSION

    @property
    def param_values(self) -> List[str]:
        return [k for k in self.actions if k != "-"]


SKILLS_V1: Dict[str, SkillSpec] = {s.name: s for s in [
    SkillSpec("identity", {"-": "v_none"},
              applicability="序列已基本就绪（低噪、无缺失主导问题）；或任何处理都有害的结构",
              risk="缺失/离群未处理即喂给下游"),
    SkillSpec("median_smooth",
              {"5": "v_median", "9": "f0_median_w9", "15": "f0_median_w15", "25": "f0_median_w25"},
              applicability="噪声/离群主导；窗宽随噪声增大（w=5 轻剂量为默认安全档）",
              risk="窗宽接近周期时抹平季节性（F0 确证：强季节族在 w9 起受伤、w25 重伤——"
                   "重剂量仅限高噪且无显著季节结构）"),
    SkillSpec("savgol_smooth", {"-": "v_savgol"},
              applicability="平滑同时保多项式局部形状",
              risk="改形平滑：跨任务符号翻转先例（classify C1：助 forecast 伤 classify）"),
    SkillSpec("stl_deseason", {"-": "v_stl"},
              applicability="显著且被证据确认的周期结构下去噪（先分解后处理残差）",
              risk="周期误检时退化为 garbage-period 激进平滑（S0.7 确证）——须有周期证据才用"),
    SkillSpec("wavelet_denoise", {"-": "v_wavelet"},
              applicability="多尺度结构去噪；对非平稳纹理较稳",
              risk="强周期下可能钝化峰谷"),
    SkillSpec("winsorize", {"-": "v_winsor"},
              applicability="离群主导、其余结构应保留原样",
              risk="重尾为真实信号（如间歇需求 burst）时截掉信息"),
    SkillSpec("winsor_savgol", {"-": "v_winsor_savgol"},
              applicability="离群+噪声并存的组合处理",
              risk="双重改形，风险叠加（savgol 符号翻转先例同样适用）"),
]}

_ACTION_TO_SKILL: Dict[str, Tuple[str, str]] = {
    aid: (s.name, pk) for s in SKILLS_V1.values() for pk, aid in s.actions.items()}


def skills_sha() -> str:
    payload = {n: dict(actions=s.actions, applicability=s.applicability, risk=s.risk,
                       fallback=s.fallback, version=s.version)
               for n, s in sorted(SKILLS_V1.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:16]


def compile_skill(skill: str, param) -> Optional[str]:
    """(skill, param) → action_id。param snap 到最近可用剂量；skill 非法 → None（调用方回退）。"""
    s = SKILLS_V1.get(str(skill))
    if s is None:
        return None
    if not s.param_values:
        return s.actions["-"]
    try:
        want = float(param)
    except (TypeError, ValueError):
        return s.actions[s.param_values[0]]                  # 无/坏参数 → 最轻剂量
    best = min(s.param_values, key=lambda k: abs(float(k) - want))
    return s.actions[best]


def action_to_skill(action_id: str) -> Tuple[str, str]:
    """action → (skill, param_key)。B 臂 wrapper 守卫用（往返恒等）。"""
    return _ACTION_TO_SKILL[action_id]


def skill_cards_text() -> str:
    """7 张 SkillSpec 卡（DataView skills 视图）。"""
    lines = []
    for s in SKILLS_V1.values():
        pv = f" | params w∈{{{','.join(s.param_values)}}}" if s.param_values else ""
        lines.append(f"- {s.name}{pv}\n    适用: {s.applicability}\n    风险: {s.risk}")
    return "\n".join(lines)
