"""conditioning/router.py — 纯函数贴 (pattern_bin, task_type) 标签（plan.md §2，无 learned router）。"""
from __future__ import annotations

from typing import Any, Dict

from . import binning


def route(struct_feats: Dict[str, float], task_type: str) -> Dict[str, Any]:
    pb = binning.pattern_bin(struct_feats)
    return {"cell_id": f"{task_type}|{pb}", "pattern_bin": pb, "task_type": task_type}
