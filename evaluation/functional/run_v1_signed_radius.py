"""V1 signed radius resolver 薄包装（2026-08-08）。

实现已迁移至 methods/ttha/signed_radius.py（方法层单一来源，供 fast_agent 真实入口
与 evaluation 实验共用）。本模块保留文件名以兼容既有实验脚本的 import：
  import run_v1_signed_radius as resolver
"""

from signed_radius import (  # noqa: F401,E402
    CONFLICT,
    CONTEXT_PREFIXES,
    DELTA_QUANTILE,
    MATERIAL_THRESHOLD,
    MIN_HISTORICAL_CONTEXTS,
    POSITIVE_PRIOR,
    RISK_PRIOR,
    UNKNOWN,
    WINDOW_LENGTH,
    episode_evidence,
    render_signed_instruction,
    resolve_order,
    window_context,
)

__all__ = [
    "CONFLICT",
    "CONTEXT_PREFIXES",
    "DELTA_QUANTILE",
    "MATERIAL_THRESHOLD",
    "MIN_HISTORICAL_CONTEXTS",
    "POSITIVE_PRIOR",
    "RISK_PRIOR",
    "UNKNOWN",
    "WINDOW_LENGTH",
    "episode_evidence",
    "render_signed_instruction",
    "resolve_order",
    "window_context",
]
