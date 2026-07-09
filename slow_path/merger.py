"""slow_path/merger.py — R4：合入 accepted 候选 + 版本+1 + 缓存 val_loss + ★跨 cell 整固。

合并流由 evolve 串行驱动（rank→逐候选重验→过门即合）；merger.apply 做二次防御校验 + 应用 + bump。
consolidate（epoch 末，第三时间尺度）：把 must-preserve 的 Strength 写入 L3 受保护区——这是
**唯一**能写受保护面的路径（consolidator writer），防遗忘/护承重墙。
"""
from __future__ import annotations

from typing import List, Optional

from ..harness.edit_patch import EditPatch, Manifest
from ..harness.layers import StrengthSignatureStats


class Merger:
    def apply(self, harness, patch: EditPatch, *, cached_val_loss: Optional[float] = None,
              store=None) -> int:
        """二次防御校验（apply_edit 内含 validate）+ 应用 + 版本+1 + 缓存 val_loss。"""
        harness.apply_edit(patch, raise_on_reject=True)
        v = harness.bump_version()
        if store is not None and cached_val_loss is not None:
            store.set_cached_val_loss(f"{patch.cell_id}@v{v}", cached_val_loss)
        return v

    def consolidate(self, harness, strength_reports, epoch: int) -> int:
        """把 must_preserve 的 Strength 写入 L3 受保护区（consolidator patch）。返回写入数。"""
        n = 0
        for sr in strength_reports:
            if not getattr(sr, "must_preserve", False):
                continue
            sig = StrengthSignatureStats(
                signature_id=sr.cell_id, cell_id=sr.cell_id, win_margin=float(sr.margin),
                support=1, must_preserve=True, promoted_in_version=harness.version)
            patch = EditPatch(
                edited_layer="L3", op="set",
                path=f"l3.strength_signatures::{sr.cell_id}", value=sig,
                manifest=Manifest(target_failure_id=sr.cell_id,
                                  target_failure_desc=f"cell {sr.cell_id} beats floor by {sr.margin:.3f}",
                                  expected_effect="preserve winning harness region for this cell"),
                source_type="strength", writer="consolidator", cell_id=sr.cell_id,
                harness_ver=harness.version)
            harness.apply_edit(patch, raise_on_reject=True)
            n += 1
        if n:
            harness.bump_version()
        return n
