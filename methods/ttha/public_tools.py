from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, TYPE_CHECKING

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256

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


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    median = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - median)))
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.std(finite))
    return max(scale, 0.0) if math.isfinite(scale) else 0.0


def _dominant_period(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size < 12:
        return None
    centered = finite - np.mean(finite)
    energy = float(np.dot(centered, centered))
    if energy <= 0:
        return None
    max_lag = min(48, finite.size // 3)
    if max_lag < 2:
        return None
    scores = [
        float(np.dot(centered[:-lag], centered[lag:]))
        / math.sqrt(
            max(float(np.dot(centered[:-lag], centered[:-lag])), 1e-12)
            * max(float(np.dot(centered[lag:], centered[lag:])), 1e-12)
        )
        for lag in range(2, max_lag + 1)
    ]
    return float(int(np.argmax(scores)) + 2)


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
    array = np.asarray(values, dtype=np.float64).ravel()
    if array.size == 0:
        raise ValueError("public feature extraction requires a non-empty series")
    missing = ~np.isfinite(array)
    finite = array[np.isfinite(array)]
    median = float(np.median(finite)) if finite.size else 0.0
    scale = _robust_scale(array)
    filled = array.copy()
    filled[missing] = median
    robust_z = np.abs(filled - median) / scale if scale > 0 else np.zeros_like(filled)
    local_peak = float(np.max(robust_z)) if robust_z.size else 0.0
    adjacent = np.abs(np.diff(filled))
    level_score = float(np.max(adjacent) / scale) if adjacent.size and scale > 0 else 0.0
    affected = np.flatnonzero(missing)
    if affected.size == 0:
        affected = np.flatnonzero(robust_z >= 3.5)
    if affected.size == 0 and adjacent.size and level_score >= 3.5:
        jump = int(np.argmax(adjacent)) + 1
        affected = np.arange(jump, array.size)
    if affected.size:
        region_start = float(affected[0] / array.size)
        region_end = float((affected[-1] + 1) / array.size)
    else:
        region_start = 0.0
        region_end = 1.0
    midpoint = array.size // 2
    first_period = _dominant_period(filled[:midpoint])
    second_period = _dominant_period(filled[midpoint:])
    if first_period is None or second_period is None:
        period_change = 0.0
    else:
        period_change = abs(second_period - first_period) / max(first_period, second_period, 1.0)
    panel = fixed_probe_panel or {}
    features = {
        "task_kind": task_kind,
        "missing_fraction": float(np.mean(missing)),
        "longest_missing_run_fraction": float(_longest_true_run(missing) / array.size),
        "local_robust_z_peak": local_peak,
        "estimated_region_start_fraction": region_start,
        "estimated_region_end_fraction": region_end,
        "level_excursion_score": level_score,
        "period_change_score": float(period_change),
        "period_repair_available": False,
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


__all__ = [
    "LocalPublicToolGateway",
    "PublicToolGateway",
    "PublicToolReceipt",
    "extract_public_features",
]
