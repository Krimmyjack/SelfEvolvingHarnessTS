from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.runtime.public_features import (
    extract_public_features as _extract_base_features,
)

if TYPE_CHECKING:
    from .agent_core import AgentRole


_FORBIDDEN_PUBLIC_NAMES = frozenset(
    {
        "clean",
        "injection_type",
        "injection_indices",
        "candidate_j",
        "j",
        "absolute_u",
        "r_private",
        "private_receipt",
        "filesystem_path",
    }
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _probe_direction(values: object) -> str:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return "unknown"
    deltas: list[float] = []
    for item in values:
        candidate = item.get("delta") if isinstance(item, Mapping) else item
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            value = float(candidate)
            if math.isfinite(value):
                deltas.append(value)
    if not deltas:
        return "unknown"
    positive = any(value > 1e-9 for value in deltas)
    negative = any(value < -1e-9 for value in deltas)
    if positive and negative:
        return "overdose_collapse"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "flat"


def extract_public_features(
    values: object,
    *,
    task_kind: str,
    fixed_probe_panel: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    base = _extract_base_features(values, task_kind=task_kind)
    panel = fixed_probe_panel or {}
    features = {
        **dict(base.mapping),
        "imputation_probe_direction": _probe_direction(panel.get("imputation", ())),
        "clipping_probe_direction": _probe_direction(panel.get("clipping", ())),
        "denoising_probe_direction": _probe_direction(panel.get("denoising", ())),
        "level_probe_direction": _probe_direction(panel.get("level_correction", ())),
    }
    return _freeze_json(features)


@dataclass(frozen=True)
class PublicToolReceipt:
    tool_name: str
    arguments: Mapping[str, object]
    public_result: Mapping[str, object]
    context_sha: str
    receipt_sha: str
    ok: bool = True

    @classmethod
    def create(
        cls,
        *,
        tool_name: str,
        arguments: Mapping[str, object],
        public_result: Mapping[str, object],
        context_sha: str,
        ok: bool = True,
    ) -> "PublicToolReceipt":
        payload = {
            "schema_version": "public-tool-receipt/1",
            "tool_name": tool_name,
            "arguments": _plain(arguments),
            "public_result": _plain(public_result),
            "context_sha": context_sha,
            "ok": ok,
        }
        return cls(
            tool_name=tool_name,
            arguments=_freeze_json(arguments),
            public_result=_freeze_json(public_result),
            context_sha=context_sha,
            receipt_sha=canonical_sha256(payload),
            ok=ok,
        )


class PublicToolGateway(Protocol):
    @property
    def context_sha(self) -> str:
        raise NotImplementedError

    def schemas_for(
        self,
        *,
        role: "AgentRole | str",
        stage: str,
    ) -> tuple[Mapping[str, object], ...]:
        raise NotImplementedError

    def call(self, name: str, arguments: Mapping[str, object]) -> PublicToolReceipt:
        raise NotImplementedError


class LocalPublicToolGateway:
    def __init__(
        self,
        values: object,
        *,
        task_kind: str,
        fixed_probe_panel: Mapping[str, object] | None = None,
    ) -> None:
        self._values = np.asarray(values, dtype=np.float64).ravel().copy()
        self._values.setflags(write=False)
        self._task_kind = task_kind
        self._panel = _freeze_json(fixed_probe_panel or {})
        self.public_features = extract_public_features(
            self._values,
            task_kind=task_kind,
            fixed_probe_panel=fixed_probe_panel,
        )
        serial_values = [float(value) if math.isfinite(float(value)) else None for value in self._values]
        self._context_sha = canonical_sha256(
            {
                "schema_version": "public-tool-context/1",
                "task_kind": task_kind,
                "values": serial_values,
                "fixed_probe_panel": _plain(self._panel),
            }
        )

    @property
    def context_sha(self) -> str:
        return self._context_sha

    def verify_context(
        self,
        values: object,
        *,
        task_kind: str,
        fixed_probe_panel: Mapping[str, object] | None = None,
    ) -> bool:
        candidate = LocalPublicToolGateway(
            values,
            task_kind=task_kind,
            fixed_probe_panel=fixed_probe_panel,
        )
        return candidate.context_sha == self.context_sha

    def schemas_for(
        self,
        *,
        role: "AgentRole | str",
        stage: str,
    ) -> tuple[Mapping[str, object], ...]:
        if str(role) not in {"fast", "AgentRole.FAST"} or stage not in {"inspect", "propose", "select"}:
            return ()
        schemas: list[Mapping[str, object]] = [
            {
                "name": "summarize_series",
                "description": "Return the immutable deployment-visible feature summary.",
                "input_schema": {"type": "object", "additionalProperties": False},
            },
            {
                "name": "localize_regions",
                "description": "Return the public estimated region fractions.",
                "input_schema": {"type": "object", "additionalProperties": False},
            },
        ]
        if self._panel:
            schemas.append(
                {
                    "name": "read_fixed_probe_panel",
                    "description": "Return the already-computed fixed public probe panel.",
                    "input_schema": {"type": "object", "additionalProperties": False},
                }
            )
        return tuple(_freeze_json(schema) for schema in schemas)

    def call(self, name: str, arguments: Mapping[str, object]) -> PublicToolReceipt:
        if not isinstance(arguments, Mapping) or arguments:
            raise PermissionError("public M0 tools accept no free-form arguments")
        if name == "summarize_series":
            result = {"features": _plain(self.public_features)}
        elif name == "localize_regions":
            result = {
                "estimated_region_start_fraction": self.public_features[
                    "estimated_region_start_fraction"
                ],
                "estimated_region_end_fraction": self.public_features[
                    "estimated_region_end_fraction"
                ],
            }
        elif name == "read_fixed_probe_panel" and self._panel:
            result = {"fixed_probe_panel": _plain(self._panel)}
        else:
            raise PermissionError(f"undeclared public tool: {name}")
        if any(key.lower() in _FORBIDDEN_PUBLIC_NAMES for key in result):
            raise PermissionError("private field cannot cross the public tool wall")
        return PublicToolReceipt.create(
            tool_name=name,
            arguments=arguments,
            public_result=result,
            context_sha=self.context_sha,
        )


def _observed_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs for one boolean missing mask."""

    padded = np.concatenate(([False], mask.astype(bool, copy=False), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop)) for start, stop in edges.reshape(-1, 2)]


def _robust_center_scale(values: np.ndarray) -> tuple[float | None, float | None]:
    observed = values[np.isfinite(values)]
    if observed.size == 0:
        return None, None
    center = float(np.median(observed))
    scale = float(1.4826 * np.median(np.abs(observed - center)))
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(observed))
    if not math.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return center, scale


def _observed_lag_pairs(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= values.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    left = values[:-lag]
    right = values[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    return left[valid], right[valid]


def _window_summary(windows: Sequence[np.ndarray], *, calendar_period: int) -> dict[str, object]:
    centers: list[float] = []
    scales: list[float] = []
    acfs: list[float] = []
    seasonal_residuals: list[float] = []
    missing_runs: list[tuple[int, int]] = []
    observed_points = 0
    total_points = 0
    for window in windows:
        mask = ~np.isfinite(window)
        missing_runs.extend(_observed_runs(mask))
        observed_points += int((~mask).sum())
        total_points += int(window.size)
        center, scale = _robust_center_scale(window)
        if center is not None and scale is not None:
            centers.append(center)
            scales.append(scale)
        left, right = _observed_lag_pairs(window, calendar_period)
        if left.size >= 3:
            left_centered = left - float(np.mean(left))
            right_centered = right - float(np.mean(right))
            denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
            if denominator > 0.0:
                acfs.append(float(np.dot(left_centered, right_centered) / denominator))
            if scale is not None:
                seasonal_residuals.append(float(np.median(np.abs(right - left)) / scale))
    run_lengths = [stop - start for start, stop in missing_runs]
    return {
        "coverage": float(observed_points / total_points) if total_points else 0.0,
        "missing_run_count": len(missing_runs),
        "maximum_missing_run_length": max(run_lengths, default=0),
        "median_robust_center": float(np.median(centers)) if centers else None,
        "median_robust_scale": float(np.median(scales)) if scales else None,
        "median_acf_at_calendar_period": float(np.median(acfs)) if acfs else None,
        "median_normalized_seasonal_residual": (
            float(np.median(seasonal_residuals)) if seasonal_residuals else None
        ),
    }


def _numeric_change(recent: Mapping[str, object], early: Mapping[str, object]) -> dict[str, object]:
    changes: dict[str, object] = {}
    for key in early:
        early_value = early[key]
        recent_value = recent[key]
        if (
            isinstance(early_value, (int, float))
            and not isinstance(early_value, bool)
            and isinstance(recent_value, (int, float))
            and not isinstance(recent_value, bool)
        ):
            changes[f"{key}_delta"] = float(recent_value) - float(early_value)
        else:
            changes[f"{key}_delta"] = None
    return changes


class CohortHistoryPublicToolGateway:
    """One deploy-visible, argument-free comparison of two historical windows.

    Values must already be cut at the decision/Support boundary.  The gateway
    never accepts identifiers, paths, arbitrary intervals, clean references, or
    downstream outcomes; it returns cohort aggregates only.
    """

    def __init__(
        self,
        series_values: Sequence[object],
        *,
        calendar_period: int,
        window_length: int = 192,
    ) -> None:
        if (
            isinstance(calendar_period, bool)
            or not isinstance(calendar_period, int)
            or calendar_period < 1
        ):
            raise ValueError("calendar_period must be a positive integer")
        if (
            isinstance(window_length, bool)
            or not isinstance(window_length, int)
            or window_length < 2
        ):
            raise ValueError("window_length must be an integer of at least two")
        arrays = tuple(np.asarray(values, dtype=np.float64).ravel().copy() for values in series_values)
        if not arrays or any(values.size < 2 * window_length for values in arrays):
            raise ValueError("each history series requires two complete fixed windows")
        for values in arrays:
            values.setflags(write=False)
        self._series_values = arrays
        self._calendar_period = calendar_period
        self._window_length = window_length
        serial_values = [
            [float(value) if math.isfinite(float(value)) else None for value in values]
            for values in arrays
        ]
        self._context_sha = canonical_sha256(
            {
                "schema_version": "cohort-history-public-tool-context/1",
                "calendar_period": calendar_period,
                "window_length": window_length,
                "series_values": serial_values,
            }
        )

    @property
    def context_sha(self) -> str:
        return self._context_sha

    def schemas_for(
        self,
        *,
        role: "AgentRole | str",
        stage: str,
    ) -> tuple[Mapping[str, object], ...]:
        if str(role) not in {"fast", "AgentRole.FAST"} or stage != "observe":
            return ()
        return (
            _freeze_json(
                {
                    "name": "compare_history_windows",
                    "description": (
                        "Compare the fixed earlier and recent deploy-visible history "
                        "windows using cohort aggregates only."
                    ),
                    "input_schema": {"type": "object", "additionalProperties": False},
                }
            ),
        )

    def call(self, name: str, arguments: Mapping[str, object]) -> PublicToolReceipt:
        if name != "compare_history_windows":
            raise PermissionError(f"undeclared public tool: {name}")
        if not isinstance(arguments, Mapping) or arguments:
            raise PermissionError("compare_history_windows accepts no arguments")
        length = self._window_length
        early = tuple(values[-2 * length : -length] for values in self._series_values)
        recent = tuple(values[-length:] for values in self._series_values)
        early_summary = _window_summary(early, calendar_period=self._calendar_period)
        recent_summary = _window_summary(recent, calendar_period=self._calendar_period)
        result = {
            "window_length": length,
            "calendar_period": self._calendar_period,
            "series_count": len(self._series_values),
            "early": early_summary,
            "recent": recent_summary,
            "early_to_recent_change": _numeric_change(recent_summary, early_summary),
        }
        return PublicToolReceipt.create(
            tool_name=name,
            arguments=arguments,
            public_result=result,
            context_sha=self.context_sha,
        )


__all__ = [
    "CohortHistoryPublicToolGateway",
    "LocalPublicToolGateway",
    "PublicToolGateway",
    "PublicToolReceipt",
    "extract_public_features",
]
