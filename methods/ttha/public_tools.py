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


__all__ = [
    "LocalPublicToolGateway",
    "PublicToolGateway",
    "PublicToolReceipt",
    "extract_public_features",
]
