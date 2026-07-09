"""slow_path/proposer.py — R7：同一 fixed LLM 出 K 个互异 EditPatch（只提议，不裁决）。

输入 = 当前 harness 可编辑面摘要 + weakness_report + strength（保留约束）+ 被拒编辑日志；
输出 = K 个候选 EditPatch（JSON 解析 → EditPatch → editable_surfaces 机械校验，丢非法）。
temperature>0 + nonce 采 K 个互异。Phase 1 偏好 leaf/list_scalar 简单编辑（JSON 鲁棒）。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..config.thresholds import K_CANDIDATES
from ..harness.edit_patch import EditPatch, Manifest
from ..harness.editable_surfaces import EDITABLE_SURFACES, validate as surface_validate
from ..llm import get_client, extract_json

_SYSTEM = (
    "You are a harness optimizer for a time-series data-readiness system. "
    "You propose ONE minimal edit (an EditPatch) to the harness to fix a specific weakness on one (pattern,task) cell. "
    "You never change model weights — you only edit the harness config. Output STRICT JSON only, no prose."
)

_SURFACE_HINT = """\
Editable surfaces you may target (path : meaning : value type):
- l2.active_operators.<op>     : enable/disable an operator GLOBALLY (all cells) : bool  (op ∈ operator_registry)
- l2.operator_defaults.<op>    : default params of an operator        : object
- l2.task_templates::<name>    : a CELL-SCOPED cleaning pipeline ONLY for THIS cell (op=set) : PipelineTemplate
      value = {"name":"<unique>","applies_to":{"task_type":"<task>","pattern_conditions":{"pattern_bin":"<THIS_CELL_BIN>"}},
               "stages":[{"stage":"denoise","preferred_ops":["denoise_savgol"],"banned_ops":["winsorize"]}]}
      Prefer this when an operator helps/hurts ONLY in this cell — it specializes WITHOUT affecting other cells.
- l1.constraints               : behavioral rule list (op=add/remove) : string
- l1.task_prompts.<task>       : per-task instruction                 : string (task ∈ forecast|anomaly_detection|classification)
- l4.gate_config.<field>       : gate threshold                       : number/bool
Addressing: leaf 'lN.field.key' (op=set); list 'l1.constraints' (op=add/remove a string); named 'l2.task_templates::name' (op=set).
Do NOT target read-only or protected surfaces (operator_registry, strength_signatures, grounded_evaluators)."""

_SCHEMA = """\
Return ONE JSON object:
{"edited_layer":"L1|L2|L3|L4","op":"set|add|remove","path":"<surface path>","value":<value>,
 "manifest":{"target_failure_id":"<id>","target_failure_desc":"<what failed>",
             "expected_effect":"<directional, e.g. 'lower forecast nRMSE'>",
             "ablation_hint":"<how to ablate>","regression_risk":"<known risk>"}}"""


def _weakness_text(w) -> str:
    attr = getattr(w, "op_attribution", {}) or {}
    prefer = attr.get("prefer", [])
    avoid = attr.get("avoid", [])
    attr_line = ""
    if prefer or avoid:
        # 历史实测：该 cell 上各算子的 outcome-calibrated 价值（+helps/−hurts，N=样本数）
        attr_line = (f"\nOPERATOR VALUE IN THIS CELL (from past trials; + helps, − hurts, N=evidence):\n"
                     f"  PREFER: {[(op, v) for op, v, n in prefer]}\n"
                     f"  AVOID : {[(op, v) for op, v, n in avoid]}")
    return (f"cell={w.cell_id} task={w.task} current_val_loss={w.current_val_loss:.4f} "
            f"floor={w.floor:.4f} gap={w.gap:.4f} improvable={w.improvable}\n"
            f"failure_signatures={dict(w.failure_signatures)}\n"
            f"suspicious_operators={w.suspicious_operators}{attr_line}")


def _active_ops(harness) -> Dict[str, bool]:
    return dict(harness.l2.active_operators)


class Proposer:
    def __init__(self, llm=None, k: int = K_CANDIDATES, model: str = "flash",
                 temperature: float = 0.7):
        self.llm = llm if llm is not None else get_client(model, temperature=temperature,
                                                          cache_name="proposer")
        self.k = k

    def _build_user(self, harness, weakness, strength, rejection_log) -> str:
        preserve = [getattr(s, "cell_id", "") for s in (strength or []) if getattr(s, "must_preserve", False)]
        rej = "; ".join(f"{p.get('path')}({reason})" for p, reason in (rejection_log or [])[-6:])
        pattern_bin = weakness.cell_id.split("|", 1)[1] if "|" in weakness.cell_id else ""
        return (f"{_SURFACE_HINT}\n\nTHIS_CELL_BIN = {pattern_bin!r}  (use it verbatim in pattern_conditions for cell-scoped templates)\n"
                f"Current active_operators: {json.dumps(_active_ops(harness))}\n\n"
                f"WEAKNESS:\n{_weakness_text(weakness)}\n\n"
                f"PRESERVE (do not degrade these cells): {preserve}\n"
                f"ALREADY-REJECTED (do not repeat): {rej or 'none'}\n\n"
                f"{_SCHEMA}\nPropose ONE minimal edit that should reduce val_loss on this cell. "
                f"If an operator helps/hurts only here, prefer a cell-scoped l2.task_templates edit over a global toggle.")

    def _parse(self, text: str, harness, cell_id: str, rank: int) -> Optional[EditPatch]:
        obj = extract_json(text)
        if not isinstance(obj, dict):
            return None
        # 模板编辑：value dict → PipelineTemplate（否则 surface_validate 的 isinstance 会判失败）
        if isinstance(obj.get("path"), str) and "task_templates" in obj["path"] and isinstance(obj.get("value"), dict):
            from ..harness.layers import PipelineTemplate
            obj = dict(obj)
            obj["value"] = PipelineTemplate.from_dict(obj["value"])
        try:
            patch = EditPatch.from_dict(obj)
        except Exception:
            return None
        if not patch.manifest.target_failure_id:
            patch.manifest.target_failure_id = f"weakness::{cell_id}"
        patch.cell_id = cell_id
        patch.harness_ver = harness.version
        patch.proposal_rank = rank
        if not surface_validate(patch, harness).ok:
            return None
        return patch

    def propose(self, harness, weakness, strength=None, rejection_log=None) -> List[EditPatch]:
        user = self._build_user(harness, weakness, strength, rejection_log)
        out: List[EditPatch] = []
        seen_paths = set()
        for i in range(self.k):
            try:
                txt = self.llm(_SYSTEM, user, nonce=i)
            except Exception:
                continue
            patch = self._parse(txt, harness, weakness.cell_id, rank=len(out))
            if patch is not None and (patch.path, str(patch.value)) not in seen_paths:
                seen_paths.add((patch.path, str(patch.value)))
                out.append(patch)
        return out
