"""Stable contracts shared by methods, runtime, and evaluation."""

from .method import (
    ExecutionReceipt,
    Method,
    PreparationRequest,
    PreparationResult,
    PreparationStatus,
    PreparedSeries,
)
from .program import Program, ProgramStep
from .task import (
    LABEL_AVAILABILITY,
    TARGET_SEMANTICS_BY_TASK,
    TASK_TYPES,
    MetricSpec,
    TaskSpec,
    anomaly_task_spec_v1,
    classification_task_spec_v1,
    forecast_task_spec_v1,
)

__all__ = [
    "ExecutionReceipt",
    "LABEL_AVAILABILITY",
    "Method",
    "PreparationRequest",
    "PreparationResult",
    "PreparationStatus",
    "PreparedSeries",
    "Program",
    "ProgramStep",
    "TARGET_SEMANTICS_BY_TASK",
    "TASK_TYPES",
    "MetricSpec",
    "TaskSpec",
    "anomaly_task_spec_v1",
    "classification_task_spec_v1",
    "forecast_task_spec_v1",
]
