from __future__ import annotations

from collections.abc import Callable

from SelfEvolvingHarnessTS.contracts.harness import HarnessSnapshot
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest, PreparationResult
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

from .fast_agent import TTHAFastAgent


class TTHAMethod:
    method_id = "ttha_m0"

    def __init__(
        self,
        fast_agent: TTHAFastAgent,
        snapshot: HarnessSnapshot | Callable[[], HarnessSnapshot],
    ) -> None:
        self.fast_agent = fast_agent
        self._snapshot = snapshot
        self.last_trace: DecisionTrace | None = None

    def _active_snapshot(self) -> HarnessSnapshot:
        return self._snapshot() if callable(self._snapshot) else self._snapshot

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        result, trace = self.fast_agent.prepare(request, self._active_snapshot())
        self.last_trace = trace
        return result


__all__ = ["TTHAMethod"]
