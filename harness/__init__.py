"""harness/ — R1/R4：被进化的唯一对象。

依赖链：edit_patch（纯契约）→ layers（四层 dataclass）→ editable_surfaces
（按 layers 校验）→ state（持有四层 + apply/snapshot/replay）。
"""
from .edit_patch import EditPatch, Manifest
from .editable_surfaces import EDITABLE_SURFACES, Surface, ValidationResult, validate, parse_path
from .layers import (
    L1Instructions, L2Skills, L3Memory, L4Verification,
    PipelineTemplate, StageDef, MiddlewareDef, OperatorConfig, EvaluatorSpec,
    GateConfig, RetrievalConfig, ShrinkageConfig,
    FailureSignatureStats, StrengthSignatureStats, TASK_TYPES,
    minimal_l1, minimal_l2, minimal_l3, minimal_l4,
)
from .state import HarnessState, Snapshot, EditRejected

__all__ = [
    "EditPatch", "Manifest",
    "EDITABLE_SURFACES", "Surface", "ValidationResult", "validate", "parse_path",
    "L1Instructions", "L2Skills", "L3Memory", "L4Verification",
    "PipelineTemplate", "StageDef", "MiddlewareDef", "OperatorConfig", "EvaluatorSpec",
    "GateConfig", "RetrievalConfig", "ShrinkageConfig",
    "FailureSignatureStats", "StrengthSignatureStats", "TASK_TYPES",
    "minimal_l1", "minimal_l2", "minimal_l3", "minimal_l4",
    "HarnessState", "Snapshot", "EditRejected",
]
