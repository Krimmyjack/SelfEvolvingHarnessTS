"""Stable contracts shared by methods, runtime, and evaluation."""

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
    "LABEL_AVAILABILITY",
    "TARGET_SEMANTICS_BY_TASK",
    "TASK_TYPES",
    "MetricSpec",
    "TaskSpec",
    "anomaly_task_spec_v1",
    "classification_task_spec_v1",
    "forecast_task_spec_v1",
]
