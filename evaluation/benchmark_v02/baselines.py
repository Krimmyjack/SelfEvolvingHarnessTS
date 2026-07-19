"""Public baselines and runner-privileged oracle diagnostics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ...contracts.task import TaskSpec
from ...operators.registry import OPERATOR_METADATA, canonicalize
from ._frozen_reference.fast_path import (
    prepared_artifact,
    run_legacy_reference_batch,
)
from .method_api import MethodSeriesView, PreparedSeries


class BaselineProtocolError(ValueError):
    """A baseline attempted to use an unapproved selection or action surface."""


@dataclass(frozen=True)
class ProgramLoss:
    split_role: str
    cell_id: str
    program_id: str
    uid: str
    loss: float

    def __post_init__(self) -> None:
        strings = (self.split_role, self.cell_id, self.program_id, self.uid)
        if any(not isinstance(item, str) or not item or item != item.strip() for item in strings):
            raise ValueError("program-loss identifiers must be canonical non-empty strings")
        if not np.isfinite(self.loss):
            raise ValueError("program loss must be finite")


@dataclass(frozen=True)
class SelectedProgram:
    program_id: str
    selection_role: str
    mean_loss: float


class RawBaseline:
    """Method no-op; benchmark canonical ingestion still runs downstream."""

    method_id = "raw"

    def prepare(
        self,
        series_view: MethodSeriesView,
        task_spec: TaskSpec,
        observed_pattern_spec: Mapping[str, float],
    ) -> PreparedSeries:
        del task_spec, observed_pattern_spec
        return PreparedSeries(
            series_uid=series_view.series_uid,
            values=series_view.degraded_inner_train,
            operators=(),
            units="original_units",
        )


def _coverage_and_means(rows: Sequence[ProgramLoss]) -> dict[str, dict[str, float]]:
    by_cell_program: dict[tuple[str, str], list[ProgramLoss]] = defaultdict(list)
    for row in rows:
        by_cell_program[(row.cell_id, row.program_id)].append(row)
    cells = sorted({row.cell_id for row in rows})
    programs = sorted({row.program_id for row in rows})
    if not cells or not programs:
        raise BaselineProtocolError("selection requires non-empty program losses")
    output: dict[str, dict[str, float]] = {cell: {} for cell in cells}
    for cell in cells:
        uid_reference: set[str] | None = None
        for program in programs:
            group = by_cell_program.get((cell, program), [])
            uids = [row.uid for row in group]
            if not group or len(uids) != len(set(uids)):
                raise BaselineProtocolError("program pool has missing or duplicate uid coverage")
            uid_set = set(uids)
            if uid_reference is None:
                uid_reference = uid_set
            elif uid_set != uid_reference:
                raise BaselineProtocolError("program pool must have identical uid coverage")
            output[cell][program] = float(np.mean([row.loss for row in group]))
    return output


def select_best_fixed(support_a_losses: Sequence[ProgramLoss]) -> SelectedProgram:
    rows = list(support_a_losses)
    if not rows or any(row.split_role != "support_a" for row in rows):
        raise BaselineProtocolError("best-fixed selection consumes Support-A only")
    cell_means = _coverage_and_means(rows)
    programs = sorted(next(iter(cell_means.values())))
    macro = {
        program: float(np.mean([cell_means[cell][program] for cell in sorted(cell_means)]))
        for program in programs
    }
    winner = min(programs, key=lambda program: (macro[program], program))
    return SelectedProgram(winner, "support_a", macro[winner])


def _best_program_per_cell(rows: Sequence[ProgramLoss]) -> dict[str, str]:
    means = _coverage_and_means(rows)
    return {
        cell: min(values, key=lambda program: (values[program], program))
        for cell, values in means.items()
    }


def _validate_query_rows(rows: Sequence[ProgramLoss]) -> None:
    if not rows or any(row.split_role not in {"dev_query", "final_query"} for row in rows):
        raise BaselineProtocolError("oracle evaluation requires Dev-Query or Final-Query rows")
    if len({row.split_role for row in rows}) != 1:
        raise BaselineProtocolError("oracle query rows cannot mix split roles")


def oracle_transfer(
    support_a_losses: Sequence[ProgramLoss],
    query_losses: Sequence[ProgramLoss],
) -> list[ProgramLoss]:
    support = list(support_a_losses)
    query = list(query_losses)
    if not support or any(row.split_role != "support_a" for row in support):
        raise BaselineProtocolError("oracle_transfer selects on Support-A only")
    _validate_query_rows(query)
    mapping = _best_program_per_cell(support)
    query_cells = {row.cell_id for row in query}
    if query_cells - set(mapping):
        raise BaselineProtocolError("Support-A lacks a mapping for a query cell")
    selected = [row for row in query if row.program_id == mapping[row.cell_id]]
    if not selected:
        raise BaselineProtocolError("selected transfer programs are absent from query rows")
    return selected


def oracle_insample(query_losses: Sequence[ProgramLoss]) -> list[ProgramLoss]:
    query = list(query_losses)
    _validate_query_rows(query)
    mapping = _best_program_per_cell(query)
    return [row for row in query if row.program_id == mapping[row.cell_id]]


class OracleTransfer:
    """Runner privilege marker. Deliberately does not implement Method.prepare."""

    diagnostic_id = "oracle_transfer"

    def evaluate(self, support_a_losses, query_losses):
        return oracle_transfer(support_a_losses, query_losses)


class OracleInSample:
    """Runner privilege marker. Deliberately does not implement Method.prepare."""

    diagnostic_id = "oracle_insample"

    def evaluate(self, query_losses):
        return oracle_insample(query_losses)


class LegacyReferenceBaseline:
    """Benchmark-only wrapper around the retired P6 det+random reference."""

    method_id = "h_ref"

    def __init__(
        self,
        *,
        state: Any,
        budget: int,
        run_path: Callable[..., Mapping[str, Any]] = run_legacy_reference_batch,
        materialize_choice: Callable[[Any, np.ndarray], np.ndarray | None] = prepared_artifact,
    ) -> None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError("legacy reference budget must be a positive integer")
        expected = getattr(getattr(state, "sampler", None), "expected_total", budget)
        if int(expected) != budget:
            raise BaselineProtocolError("legacy reference budget disagrees with frozen state")
        self._state = state
        self._budget = budget
        self._run_path = run_path
        self._materialize_choice = materialize_choice

    @property
    def budget(self) -> int:
        return self._budget

    def prepare(
        self,
        series_view: MethodSeriesView,
        task_spec: TaskSpec,
        observed_pattern_spec: Mapping[str, float],
    ) -> PreparedSeries:
        del task_spec, observed_pattern_spec
        values = series_view.degraded_inner_train
        choices = self._run_path(
            {series_view.series_uid: values}, self._state, self._budget
        )
        choice = choices.get(series_view.series_uid)
        operators = tuple(choice.op_names()) if choice is not None else ()
        for requested in operators:
            metadata = OPERATOR_METADATA.get(canonicalize(requested))
            if metadata is None:
                raise BaselineProtocolError(
                    f"legacy reference selected unknown operator {requested!r}"
                )
            if bool(metadata.get("changes_target_space")):
                raise BaselineProtocolError(
                    "legacy reference selected a forbidden target-space operator"
                )
        artifact = self._materialize_choice(choice, values)
        if artifact is None:
            raise BaselineProtocolError("legacy reference candidate execution failed")
        return PreparedSeries(
            series_uid=series_view.series_uid,
            values=np.asarray(artifact),
            operators=operators,
            units="original_units",
        )


__all__ = [
    "BaselineProtocolError",
    "LegacyReferenceBaseline",
    "OracleInSample",
    "OracleTransfer",
    "ProgramLoss",
    "RawBaseline",
    "SelectedProgram",
    "oracle_insample",
    "oracle_transfer",
    "select_best_fixed",
]
