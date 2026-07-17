from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProgramStep:
    op: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, op: str, params: Mapping[str, Any] | None = None) -> "ProgramStep":
        if not isinstance(op, str) or not op or op != op.strip():
            raise ValueError("ProgramStep.op must be a canonical non-empty string")
        return cls(op=op, params=tuple(sorted(dict(params or {}).items())))

    def execution_pair(self) -> tuple[str, dict[str, Any]]:
        return self.op, dict(self.params)


@dataclass(frozen=True)
class Program:
    steps: tuple[ProgramStep, ...]
    source: str

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        source: str,
    ) -> "Program":
        if not isinstance(source, str) or not source or source != source.strip():
            raise ValueError("Program.source must be a canonical non-empty string")
        normalized = tuple(ProgramStep.from_mapping(op, params) for op, params in steps)
        if not normalized:
            raise ValueError("Program must contain at least one step")
        return cls(normalized, source)

    def execution_steps(self) -> list[tuple[str, dict[str, Any]]]:
        return [step.execution_pair() for step in self.steps]

    def sha(self) -> str:
        payload = [[step.op, dict(step.params)] for step in self.steps]
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


__all__ = ["Program", "ProgramStep"]
