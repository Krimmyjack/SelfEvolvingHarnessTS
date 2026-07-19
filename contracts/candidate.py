from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .program import Program


class CandidateKind(str, Enum):
    IDENTITY = "identity"
    PROGRAM = "program"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: CandidateKind
    program: Program | None
    source: str
    downstream_model_id: str = "fixed:m0"

    def __post_init__(self) -> None:
        if not self.candidate_id or self.candidate_id != self.candidate_id.strip():
            raise ValueError("candidate_id must be canonical")
        if not self.source or self.source != self.source.strip():
            raise ValueError("source must be canonical")
        if not self.downstream_model_id or self.downstream_model_id != self.downstream_model_id.strip():
            raise ValueError("downstream_model_id must be canonical")
        if self.kind is CandidateKind.IDENTITY:
            if self.candidate_id != "identity" or self.program is not None or self.source != "runtime":
                raise ValueError("identity is runtime-owned and program-free")
        elif self.kind is CandidateKind.PROGRAM:
            if self.program is None:
                raise ValueError("PROGRAM candidate requires Program")
            if self.candidate_id == "identity":
                raise ValueError("PROGRAM candidate cannot use identity id")
        else:
            raise ValueError("unknown CandidateKind")

    @classmethod
    def identity(cls) -> "Candidate":
        return cls("identity", CandidateKind.IDENTITY, None, "runtime")

    @classmethod
    def program_candidate(
        cls,
        candidate_id: str,
        program: Program,
        *,
        source: str,
        downstream_model_id: str = "fixed:m0",
    ) -> "Candidate":
        return cls(candidate_id, CandidateKind.PROGRAM, program, source, downstream_model_id)


__all__ = ["Candidate", "CandidateKind"]
