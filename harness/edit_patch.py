"""harness/edit_patch.py — EditPatch 契约（plan.md §3.2，已定版）。

proposer ↔ merger 的唯一契约。一次编辑 = 一个 layer 上的一个 op，改一个 path 的值。
要改多处 → 多个 EditPatch（最小编辑原则）。

无 Scope 字段 —— scope 由 editable_surfaces[path] 声明/解析，proposer 无权自报影响范围
（守不变量 #1：proposer 不判自己的影响范围）。

寻址规约见 editable_surfaces.parse_path：
  leaf:         l2.active_operators.outlier_iqr   → op=set value=False
  list_scalar:  l1.constraints                    → op=add/remove 标量
  named_object: l2.task_templates::forecast_default → op=set/add/remove 整个对象
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict

# ── 原子类型（运行期用元组做白名单校验，注解保留 Literal 供阅读）──────────────
LAYERS = ("L1", "L2", "L3", "L4")
OPS = ("set", "add", "remove")
SOURCE_TYPES = ("failure", "strength")   # 修弱点 / 护强项
WRITERS = ("step", "consolidator")


@dataclass
class Manifest:
    """可证伪契约（plan.md §3.2(a)）。

    只有 `target_failure_id` 是 load-bearing（FK → failure/strength_signatures：
    去重 / 信用分配 / 链回 report）。其余三项是 ADVISORY 自然语言——validator **不**
    采信其量化数字，量化阈值永远是我们校准的 ε。
    """
    target_failure_id: str                  # ★机器 FK
    target_failure_desc: str = ""           # NL，供 prompt + 审计
    expected_effect: str = ""               # NL，方向性，ADVISORY
    ablation_hint: str = ""                 # NL，ADVISORY（Phase2+ 消融）
    regression_risk: str = ""               # NL，ADVISORY（可提示 held-out(b) 优先子集）

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Manifest":
        fields = ("target_failure_id", "target_failure_desc",
                  "expected_effect", "ablation_hint", "regression_risk")
        return cls(**{k: d.get(k, "") for k in fields})


@dataclass
class EditPatch:
    """proposer↔merger 唯一契约（plan.md §3.2(a)）。"""
    # —— 机械编辑（proposer 必填）——
    edited_layer: str                       # "L1".."L4"
    op: str                                 # "set" | "add" | "remove"
    path: str                               # 见寻址规约
    value: Any                              # op=remove 可为 None；类型/引用完整性由 validate() 校验
    manifest: Manifest
    source_type: str = "failure"            # "failure"(step) | "strength"(consolidator)
    # —— 元数据（schedule/merger 写，proposer 留空）——
    writer: str = "step"                    # "step" | "consolidator"
    cell_id: str = ""                       # 来源 cell（held-out(a) 取样源 + 溯源）；consolidator 可为 "*"
    harness_ver: int = 0                    # 基于哪个版本提出（兼 batch_id 角色 + 确定性 replay 锚）
    proposal_rank: int = 0                  # 仅 step：K 候选内 schedule 排名；consolidator 恒 0

    def __post_init__(self) -> None:
        if isinstance(self.manifest, dict):
            self.manifest = Manifest.from_dict(self.manifest)
        if self.edited_layer not in LAYERS:
            raise ValueError(f"edited_layer ∈ {LAYERS}, got {self.edited_layer!r}")
        if self.op not in OPS:
            raise ValueError(f"op ∈ {OPS}, got {self.op!r}")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"source_type ∈ {SOURCE_TYPES}, got {self.source_type!r}")
        if self.writer not in WRITERS:
            raise ValueError(f"writer ∈ {WRITERS}, got {self.writer!r}")
        if not isinstance(self.manifest, Manifest):
            raise TypeError("manifest must be Manifest or dict")

    @property
    def layer_key(self) -> str:
        """path 首段（小写 l1..l4），用于落到 EDITABLE_SURFACES。"""
        head = self.path.split("::", 1)[0]
        return head.split(".", 1)[0]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-round-trip 友好（审计/回滚日志）。value 若为 dataclass 会被递归展开。"""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditPatch":
        # 只取已知字段——LLM 常额外塞 reasoning/explanation 等键，否则 cls(**d) 会 TypeError 把候选丢掉
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        d["manifest"] = Manifest.from_dict(d.get("manifest", {}) or {})
        return cls(**d)
