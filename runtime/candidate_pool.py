from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.program import Program

from .errors import ExecutionError, ProtocolViolation
from .executor import run_pipeline


class ProtocolChoiceError(ProtocolViolation):
    """The Agent did not choose exactly one member of the runtime-owned pool."""


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[Candidate, ...]
    requested_k: int

    def __post_init__(self) -> None:
        if isinstance(self.requested_k, bool) or not isinstance(self.requested_k, int):
            raise ValueError("requested_k must be an integer")
        if self.requested_k < 1:
            raise ValueError("requested_k must include identity")
        if not self.candidates or self.candidates[0].kind is not CandidateKind.IDENTITY:
            raise ValueError("CandidatePool must begin with runtime identity")
        if sum(candidate.kind is CandidateKind.IDENTITY for candidate in self.candidates) != 1:
            raise ValueError("CandidatePool must contain exactly one identity")
        if len(self.candidates) > self.requested_k:
            raise ValueError("CandidatePool exceeds requested_k")
        ids = self.ids
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")

    @classmethod
    def build(
        cls,
        programs: Iterable[Candidate],
        *,
        total_k: int,
    ) -> "CandidatePool":
        if isinstance(total_k, bool) or not isinstance(total_k, int) or total_k < 1:
            raise ValueError("total_k must include identity")
        merged = [Candidate.identity()]
        seen_program_sha: set[str] = set()
        seen_candidate_ids = {"identity"}
        for candidate in programs:
            if len(merged) == total_k:
                break
            if candidate.kind is not CandidateKind.PROGRAM:
                raise ValueError("suppliers may only submit PROGRAM candidates")
            if candidate.program is None:
                raise ValueError("PROGRAM candidate requires Program")
            program_sha = candidate.program.sha()
            if program_sha in seen_program_sha:
                continue
            if candidate.candidate_id in seen_candidate_ids:
                raise ValueError("candidate IDs must be unique")
            seen_program_sha.add(program_sha)
            seen_candidate_ids.add(candidate.candidate_id)
            merged.append(candidate)
            if len(merged) == total_k:
                break
        return cls(tuple(merged), total_k)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    def apply_risk(self, keep: Callable[[Candidate], bool]) -> "CandidatePool":
        kept = tuple(
            candidate
            for candidate in self.candidates
            if candidate.kind is CandidateKind.IDENTITY or keep(candidate)
        )
        return CandidatePool(kept, self.requested_k)

    def require_choice(self, chosen_candidate_id: str) -> Candidate:
        if (
            not isinstance(chosen_candidate_id, str)
            or not chosen_candidate_id
            or chosen_candidate_id != chosen_candidate_id.strip()
        ):
            raise ProtocolChoiceError("chosen_candidate_id must name one candidate")
        for candidate in self.candidates:
            if candidate.candidate_id == chosen_candidate_id:
                return candidate
        raise ProtocolChoiceError(
            f"chosen_candidate_id {chosen_candidate_id!r} is not in the candidate pool"
        )


def _immutable_float64(values: object) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True).ravel()
    array.setflags(write=False)
    return array


def execute_selected(
    candidate: Candidate,
    values: object,
) -> tuple[np.ndarray, Program | None]:
    if candidate.kind is CandidateKind.IDENTITY:
        return _immutable_float64(values), None
    if candidate.program is None:
        raise ExecutionError("PROGRAM candidate has no Program")
    result = run_pipeline(
        candidate.program.execution_steps(),
        values,
        source=candidate.source,
    )
    if not result.ok or result.artifact is None:
        raise ExecutionError(result.error or "candidate execution failed")
    return _immutable_float64(result.artifact), candidate.program


def effect_equivalent_to_identity(
    raw: object,
    prepared: object,
    tolerance: float | None = None,
) -> bool:
    left = np.asarray(raw)
    right = np.asarray(prepared)
    if left.shape != right.shape:
        return False
    if tolerance is None:
        return left.dtype == right.dtype and left.tobytes(order="C") == right.tobytes(order="C")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError("tolerance must be a non-negative finite number")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be a non-negative finite number")
    try:
        return bool(
            np.allclose(
                left,
                right,
                atol=float(tolerance),
                rtol=0.0,
                equal_nan=True,
            )
        )
    except (TypeError, ValueError):
        return False


__all__ = [
    "CandidatePool",
    "ProtocolChoiceError",
    "effect_equivalent_to_identity",
    "execute_selected",
]
