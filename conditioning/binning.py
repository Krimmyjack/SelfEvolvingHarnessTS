"""conditioning/binning.py — 冻结网格 bin(struct_feats, task) → cell_id（plan.md §3.1 / R3）。

Phase 0 minimal 决策轴 = SNR(2 档) × missing(2 档) → 4 个 pattern_bin；× 3 task = 12 cells。
网格是**冻结**的（阈值在 config 固定），与到达顺序无关——同样的 struct_feats 永远落同一 cell。
Phase 1 可加去噪/分解/插补三轴细化（plan.md 一级 3 轴 → 8 cells/task）。
"""
from __future__ import annotations

from typing import Dict, Tuple

from ..config import thresholds as TH


def pattern_bin(struct_feats: Dict[str, float]) -> str:
    """仅由结构特征决定的 pattern 网格坐标（不含 task）。"""
    snr = struct_feats.get("SNR", 0.0)
    miss = struct_feats.get("missing_rate", 0.0)
    snr_b = "snrLow" if snr < TH.BIN_SNR_SPLIT_DB else "snrHigh"
    miss_b = "miss" if miss > TH.BIN_MISSING_ANY else "full"
    return f"{snr_b}|{miss_b}"


def bin(struct_feats: Dict[str, float], task_type: str) -> str:
    """cell_id = task | pattern_bin（与到达顺序无关）。"""
    return f"{task_type}|{pattern_bin(struct_feats)}"


def split_cell_id(cell_id: str) -> Tuple[str, str]:
    """cell_id → (task_type, pattern_bin)。"""
    task, _, pb = cell_id.partition("|")
    return task, pb
