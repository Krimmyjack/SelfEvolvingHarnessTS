"""fast_path/perceive.py — 为单条输入产 conditioning_key（+cell_id）。

调 conditioning/ 计算 struct_feats + quality_profile，再用冻结网格派生 cell_id/pattern_bin。
task_spec 缺省时从 harness.L1.task_sensitivity 取该 task 的 preserve/suppress。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..conditioning import build_conditioning_key, binning


def perceive(x, task_type: str, harness, task_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if task_spec is None:
        task_spec = {"sensitivity": harness.l1.task_sensitivity.get(task_type, {"preserve": [], "suppress": []})}
    key = build_conditioning_key(x, task_type, task_spec)
    feats = key["pattern"]["struct_feats"]
    key["cell_id"] = binning.bin(feats, task_type)
    key["pattern_bin"] = binning.pattern_bin(feats)
    return key
