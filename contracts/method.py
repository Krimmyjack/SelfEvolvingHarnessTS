from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

import numpy as np

from .program import Program
from .task import TaskSpec


def _array_copy(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = array.copy()
    result.setflags(write=False)
    return result


def _uid(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("series_uid must be a canonical non-empty string")
    return value


class PreparationStatus(str, Enum):
    PREPARED = "prepared"
    ABSTAINED = "abstained"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionReceipt:
    ok: bool
    error: str = ""
    trace: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class PreparationRequest:
    series_uid: str
    values: np.ndarray
    task_spec: TaskSpec
    observed_pattern_spec: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_uid", _uid(self.series_uid))
        object.__setattr__(self, "values", _array_copy(self.values, "values"))
        object.__setattr__(self, "observed_pattern_spec", dict(self.observed_pattern_spec))
        if not isinstance(self.task_spec, TaskSpec):
            raise TypeError("task_spec must be TaskSpec")


@dataclass(frozen=True)
class PreparedSeries:
    series_uid: str
    values: np.ndarray
    operators: tuple[str, ...]
    units: str = "original_units"

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_uid", _uid(self.series_uid))
        object.__setattr__(self, "values", _array_copy(self.values, "values"))
        object.__setattr__(self, "operators", tuple(self.operators))
        if self.units != "original_units":
            raise ValueError("units must be original_units")


@dataclass(frozen=True)
class PreparationResult:
    status: PreparationStatus
    prepared: PreparedSeries | None
    program: Program | None
    receipt: ExecutionReceipt

    def __post_init__(self) -> None:
        if self.status is PreparationStatus.FAILED and self.prepared is not None:
            raise ValueError("FAILED result cannot carry a prepared series")
        if self.status in {PreparationStatus.PREPARED, PreparationStatus.ABSTAINED} and self.prepared is None:
            raise ValueError(f"{self.status.name} result requires a prepared series")


class Method(Protocol):
    method_id: str

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        ...


__all__ = [
    "ExecutionReceipt",
    "Method",
    "PreparationRequest",
    "PreparationResult",
    "PreparationStatus",
    "PreparedSeries",
]
