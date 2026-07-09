"""operators/ — R1：只读基础算子库（infrastructure）。

算子代码是只读基础设施；L2 只编排（active/banned/默认参/模板），不改算子实现。
registry.py 是单一真源：TOOL_REGISTRY(callable) + OPERATOR_METADATA(给 harness.L2)。
"""
from .registry import (
    OPERATOR_SPECS, OPERATOR_METADATA, OPERATOR_NAMES, TOOL_REGISTRY, get_operator,
)
from ._common import ShapeChangingNotSupported

__all__ = [
    "OPERATOR_SPECS", "OPERATOR_METADATA", "OPERATOR_NAMES", "TOOL_REGISTRY",
    "get_operator", "ShapeChangingNotSupported",
]
