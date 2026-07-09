"""conditioning/ — R3：全系统索引语言（快路径建键 / 慢路径 group-by / 记忆检索共用）。

Phase 0 已落地：key.py（conditioning_key 构造 = struct_feats + quality_profile + task）。
Phase 1 待补：binning.py（冻结网格 → cell_id）、distance.py、router.py。
"""
from .key import (
    STRUCT_FEAT_NAMES, struct_feats, quality_profile, build_conditioning_key,
)
from . import binning
from .binning import bin as bin_cell, pattern_bin, split_cell_id
from .router import route
from .distance import distance, similarity, d_struct, d_quality

__all__ = [
    "STRUCT_FEAT_NAMES", "struct_feats", "quality_profile", "build_conditioning_key",
    "binning", "bin_cell", "pattern_bin", "split_cell_id", "route",
    "distance", "similarity", "d_struct", "d_quality",
]
