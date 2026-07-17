from __future__ import annotations

from ...contracts.method import (
    ExecutionReceipt,
    PreparationRequest,
    PreparationResult,
    PreparationStatus,
    PreparedSeries,
)
from ...contracts.program import Program
from ...runtime.fast_path import execute_candidate, run_fast_path
from .config import HRefState, default_state


class HRefV02Method:
    method_id = "h_ref_v02"

    def __init__(self, state: HRefState | None = None) -> None:
        self._state = default_state() if state is None else state

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        budget = self._state.sampler.expected_total
        choices = run_fast_path({request.series_uid: request.values}, self._state, budget)
        choice = choices[request.series_uid]
        if choice is None:
            prepared = PreparedSeries(request.series_uid, request.values, (), "original_units")
            return PreparationResult(
                PreparationStatus.ABSTAINED,
                prepared,
                None,
                ExecutionReceipt(ok=True),
            )

        execution = execute_candidate(choice, request.values)
        receipt = ExecutionReceipt(
            ok=execution.ok,
            error=execution.error,
            trace=tuple(dict(row) for row in execution.trace),
        )
        program = Program.from_steps(choice.program_steps, source=choice.source)
        if not execution.ok or execution.artifact is None:
            return PreparationResult(PreparationStatus.FAILED, None, program, receipt)

        prepared = PreparedSeries(
            request.series_uid,
            execution.artifact,
            choice.op_names(),
            "original_units",
        )
        return PreparationResult(PreparationStatus.PREPARED, prepared, program, receipt)


__all__ = ["HRefV02Method"]
