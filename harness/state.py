"""harness/state.py — HarnessState：被进化的唯一对象（plan.md §3.2(e) / R1/R4）。

持有 L1–L4 + 全局单调 version + 有序 patch_log。
- apply_edit：validate → 机械变更被寻址的 surface → 记日志（不自动 +version；version 由
  merger 在 accept 后显式 bump，对应 plan.md「merger 串行、版本单调」）。
- snapshot/restore：整份 config 深拷贝做回滚（config 非权重，极便宜）；evidence_store 是共享
  后端，不进快照（detach-reattach）。
- replay：MINIMAL + 有序 accepted patch ⇒ 确定性复现 harness（审计/复现的规范路径）。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import editable_surfaces as ev
from .edit_patch import EditPatch
from .layers import (
    L1Instructions, L2Skills, L3Memory, L4Verification,
    minimal_l1, minimal_l2, minimal_l3, minimal_l4,
)


class EditRejected(Exception):
    """validate 不通过时由 apply_edit(raise_on_reject=True) 抛出。"""
    def __init__(self, reason: str, patch: Optional[EditPatch] = None):
        super().__init__(reason)
        self.reason = reason
        self.patch = patch


@dataclass
class Snapshot:
    l1: L1Instructions
    l2: L2Skills
    l3: L3Memory
    l4: L4Verification
    version: int
    patch_log_len: int


@dataclass
class HarnessState:
    l1: L1Instructions
    l2: L2Skills
    l3: L3Memory
    l4: L4Verification
    version: int = 0
    patch_log: List[EditPatch] = field(default_factory=list)

    # ── 构造 ──────────────────────────────────────────────────────────────
    @classmethod
    def from_minimal(cls) -> "HarnessState":
        return cls(minimal_l1(), minimal_l2(), minimal_l3(), minimal_l4())

    @classmethod
    def replay(cls, patches: List[EditPatch]) -> "HarnessState":
        """从 MINIMAL 起按序重放 accepted patch（确定性复现）。"""
        h = cls.from_minimal()
        for p in patches:
            h.apply_edit(p, raise_on_reject=True)
        return h

    # ── 校验 / 应用 ───────────────────────────────────────────────────────
    def validate(self, patch: EditPatch) -> ev.ValidationResult:
        return ev.validate(patch, self)

    def apply_edit(self, patch: EditPatch, *, do_validate: bool = True,
                   raise_on_reject: bool = True, log: bool = True) -> ev.ValidationResult:
        """校验并应用一个 EditPatch。返回 ValidationResult；被拒时按 raise_on_reject 抛错或静默返回。"""
        res = ev.validate(patch, self) if do_validate else ev.ValidationResult(True, "skipped")
        if not res.ok:
            if raise_on_reject:
                raise EditRejected(res.reason, patch)
            return res
        self._apply_mutation(patch)
        if log:
            self.patch_log.append(patch)
        return res

    def _apply_mutation(self, patch: EditPatch) -> None:
        pp = ev.parse_path(patch.path)
        surface = ev.EDITABLE_SURFACES[pp.surface_key]
        container = getattr(getattr(self, pp.layer), pp.field)
        addr = surface.addressing

        if addr == "leaf":
            key = pp.selector
            if isinstance(container, dict):
                if patch.op == "set":
                    container[key] = patch.value
                elif patch.op == "remove":
                    container.pop(key, None)
            else:  # dataclass 属性
                if patch.op == "set":
                    setattr(container, key, patch.value)

        elif addr == "list_scalar":
            if patch.op == "add":
                if patch.value not in container:
                    container.append(patch.value)          # add-if-absent（幂等）
            elif patch.op == "remove":
                if patch.value in container:
                    container.remove(patch.value)

        elif addr == "named_object":
            name = pp.selector
            if isinstance(container, dict):                # dict-backed（key=name）
                if patch.op in ("set", "add"):
                    container[name] = patch.value
                elif patch.op == "remove":
                    container.pop(name, None)
            else:                                          # list-backed（按 name_field 定位）
                idx = _find_named(container, surface.name_field, name)
                if patch.op == "set":
                    if idx is None:
                        container.append(patch.value)
                    else:
                        container[idx] = patch.value
                elif patch.op == "add":
                    if idx is None:
                        container.append(patch.value)      # add-if-absent
                elif patch.op == "remove":
                    if idx is not None:
                        container.pop(idx)

    # ── 版本（merger 显式调用）────────────────────────────────────────────
    def bump_version(self, n: int = 1) -> int:
        self.version += n
        return self.version

    # ── 快照 / 回滚（merger 试装候选用）──────────────────────────────────
    def snapshot(self) -> Snapshot:
        store = self.l3.evidence_store
        self.l3.evidence_store = None                      # 共享后端不进快照
        try:
            snap = Snapshot(
                l1=copy.deepcopy(self.l1), l2=copy.deepcopy(self.l2),
                l3=copy.deepcopy(self.l3), l4=copy.deepcopy(self.l4),
                version=self.version, patch_log_len=len(self.patch_log),
            )
        finally:
            self.l3.evidence_store = store
        return snap

    def restore(self, snap: Snapshot) -> None:
        store = self.l3.evidence_store                     # 保留当前共享后端
        self.l1 = copy.deepcopy(snap.l1)
        self.l2 = copy.deepcopy(snap.l2)
        self.l3 = copy.deepcopy(snap.l3)
        self.l4 = copy.deepcopy(snap.l4)
        self.l3.evidence_store = store
        self.version = snap.version
        del self.patch_log[snap.patch_log_len:]

    # ── 序列化（审计用；权威复现走 replay）──────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        store = self.l3.evidence_store
        self.l3.evidence_store = None
        try:
            layers = {k: dataclasses.asdict(getattr(self, k)) for k in ("l1", "l2", "l3", "l4")}
        finally:
            self.l3.evidence_store = store
        return {"version": self.version, "layers": layers,
                "patch_log": [p.to_dict() for p in self.patch_log]}

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def _find_named(container: list, name_field: str, name: str) -> Optional[int]:
    for i, item in enumerate(container):
        val = item.get(name_field) if isinstance(item, dict) else getattr(item, name_field, None)
        if val == name:
            return i
    return None
