from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256
from SelfEvolvingHarnessTS.operators.registry import operator_targeting_mode

from .candidate_pool import effect_equivalent_to_identity
from .executor import run_pipeline


RECEIPT_STATUSES = ("valid", "warning", "rejected")
REJECTION_CODES = (
    "",
    "OPERATOR_NOT_ALLOWED",
    "EXECUTION_FAILED",
    "SHAPE_MISMATCH",
    "NONFINITE_OUTPUT",
    "MODIFICATION_FRACTION_EXCEEDED",
    "OUTSIDE_SCOPE_MODIFICATION",
)
WARNING_CODES = ("EFFECT_EQUIVALENT_TO_IDENTITY",)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(nested) for key, nested in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


def _modified_indices(raw: np.ndarray, prepared: np.ndarray) -> tuple[int, ...]:
    if raw.shape != prepared.shape:
        return tuple(range(max(raw.size, prepared.size)))
    equal = np.equal(raw, prepared) | (np.isnan(raw) & np.isnan(prepared))
    return tuple(int(index) for index in np.flatnonzero(~equal))


def _contiguous_regions(indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not indices:
        return ()
    regions: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            regions.append((start, previous + 1))
            start = index
        previous = index
    regions.append((start, previous + 1))
    return tuple(regions)


def _region_fractions(
    regions: Sequence[tuple[int, int]], denominator: int
) -> tuple[tuple[float, float], ...]:
    safe = max(denominator, 1)
    return tuple(
        (round(start / safe, 6), round(end / safe, 6)) for start, end in regions
    )


def _inside_regions(index: int, regions: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= index < end for start, end in regions)


@dataclass(frozen=True)
class CandidateVerificationReceipt:
    candidate_id: str
    candidate_kind: str
    status: str
    program_sha: str | None
    operator_legality_ok: bool
    compilation_ok: bool
    execution_ok: bool
    shape_preserved: bool
    finite_output: bool
    effect_equivalent_to_identity: bool
    modified_fraction: float
    modified_region_fractions: tuple[tuple[float, float], ...]
    outside_inspected_region_modified: bool
    warnings: tuple[str, ...] = ()
    rejection_code: str = ""
    schema_version: str = "candidate-verification-receipt/1"

    def __post_init__(self) -> None:
        if self.schema_version != "candidate-verification-receipt/1":
            raise ValueError("unsupported CandidateVerificationReceipt revision")
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if self.candidate_kind not in {kind.value for kind in CandidateKind}:
            raise ValueError("candidate_kind is unsupported")
        if self.status not in RECEIPT_STATUSES:
            raise ValueError("receipt status is unsupported")
        if self.rejection_code not in REJECTION_CODES:
            raise ValueError("rejection_code is unsupported")
        if self.status == "rejected" and not self.rejection_code:
            raise ValueError("rejected receipt requires rejection_code")
        if self.status != "rejected" and self.rejection_code:
            raise ValueError("non-rejected receipt cannot carry rejection_code")
        if len(self.warnings) != len(set(self.warnings)) or any(
            warning not in WARNING_CODES for warning in self.warnings
        ):
            raise ValueError("warnings must be unique closed-vocabulary values")
        if self.status == "warning" and not self.warnings:
            raise ValueError("warning receipt requires at least one warning")
        if not math.isfinite(float(self.modified_fraction)) or not (
            0.0 <= float(self.modified_fraction) <= 1.0
        ):
            raise ValueError("modified_fraction must be finite in [0, 1]")
        for start, end in self.modified_region_fractions:
            if not (0.0 <= start < end <= 1.0):
                raise ValueError("modified regions must be normalized non-empty intervals")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "status": self.status,
            "program_sha": self.program_sha,
            "operator_legality_ok": self.operator_legality_ok,
            "compilation_ok": self.compilation_ok,
            "execution_ok": self.execution_ok,
            "shape_preserved": self.shape_preserved,
            "finite_output": self.finite_output,
            "effect_equivalent_to_identity": self.effect_equivalent_to_identity,
            "modified_fraction": float(self.modified_fraction),
            "modified_region_fractions": [
                [start, end] for start, end in self.modified_region_fractions
            ],
            "outside_inspected_region_modified": self.outside_inspected_region_modified,
            "warnings": list(self.warnings),
            "rejection_code": self.rejection_code,
        }

    @property
    def receipt_sha(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class CandidateExecutionArtifact:
    candidate: Candidate
    receipt: CandidateVerificationReceipt
    prepared_values: np.ndarray | None
    execution_trace: tuple[Mapping[str, object], ...] = field(default_factory=tuple)
    modified_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.candidate.candidate_id != self.receipt.candidate_id:
            raise ValueError("candidate and receipt IDs must match")
        if self.prepared_values is not None:
            values = np.asarray(self.prepared_values, dtype=np.float64).copy()
            values.setflags(write=False)
            object.__setattr__(self, "prepared_values", values)
        object.__setattr__(
            self,
            "execution_trace",
            tuple(_freeze_json(row) for row in self.execution_trace),
        )

    @property
    def selectable(self) -> bool:
        return self.receipt.status != "rejected"


def verify_candidate(
    candidate: Candidate,
    raw_values: object,
    *,
    allowed_operators: Sequence[str],
    inspected_regions: tuple[tuple[int, int], ...] = (),
    maximum_modified_fraction: float = 1.0,
    preserve_outside_inspected_region: bool = False,
    require_finite_output: bool = True,
) -> CandidateExecutionArtifact:
    raw = np.asarray(raw_values, dtype=np.float64).ravel()
    allowed = set(allowed_operators)

    if candidate.kind is CandidateKind.IDENTITY:
        output = raw.copy()
        receipt = CandidateVerificationReceipt(
            candidate_id=candidate.candidate_id,
            candidate_kind=candidate.kind.value,
            status="valid",
            program_sha=None,
            operator_legality_ok=True,
            compilation_ok=True,
            execution_ok=True,
            shape_preserved=True,
            finite_output=bool(np.isfinite(output).all()),
            effect_equivalent_to_identity=True,
            modified_fraction=0.0,
            modified_region_fractions=(),
            outside_inspected_region_modified=False,
        )
        return CandidateExecutionArtifact(candidate, receipt, output)

    assert candidate.program is not None
    steps = candidate.program.execution_steps()
    legal = all(operator_id in allowed for operator_id, _params in steps)
    if not legal:
        receipt = CandidateVerificationReceipt(
            candidate_id=candidate.candidate_id,
            candidate_kind=candidate.kind.value,
            status="rejected",
            program_sha=candidate.program.sha(),
            operator_legality_ok=False,
            compilation_ok=True,
            execution_ok=False,
            shape_preserved=False,
            finite_output=False,
            effect_equivalent_to_identity=False,
            modified_fraction=0.0,
            modified_region_fractions=(),
            outside_inspected_region_modified=False,
            rejection_code="OPERATOR_NOT_ALLOWED",
        )
        return CandidateExecutionArtifact(candidate, receipt, None)

    execution = run_pipeline(steps, raw, source=candidate.source)
    trace = tuple(dict(row) for row in execution.trace)
    if not execution.ok or execution.artifact is None:
        receipt = CandidateVerificationReceipt(
            candidate_id=candidate.candidate_id,
            candidate_kind=candidate.kind.value,
            status="rejected",
            program_sha=candidate.program.sha(),
            operator_legality_ok=True,
            compilation_ok=True,
            execution_ok=False,
            shape_preserved=False,
            finite_output=False,
            effect_equivalent_to_identity=False,
            modified_fraction=0.0,
            modified_region_fractions=(),
            outside_inspected_region_modified=False,
            rejection_code="EXECUTION_FAILED",
        )
        return CandidateExecutionArtifact(candidate, receipt, None, trace)

    output = np.asarray(execution.artifact, dtype=np.float64).ravel()
    shape_ok = output.shape == raw.shape
    if not shape_ok:
        receipt = CandidateVerificationReceipt(
            candidate_id=candidate.candidate_id,
            candidate_kind=candidate.kind.value,
            status="rejected",
            program_sha=candidate.program.sha(),
            operator_legality_ok=True,
            compilation_ok=True,
            execution_ok=True,
            shape_preserved=False,
            finite_output=bool(np.isfinite(output).all()),
            effect_equivalent_to_identity=False,
            modified_fraction=1.0,
            modified_region_fractions=((0.0, 1.0),),
            outside_inspected_region_modified=bool(inspected_regions),
            rejection_code="SHAPE_MISMATCH",
        )
        return CandidateExecutionArtifact(candidate, receipt, output, trace)

    modified = _modified_indices(raw, output)
    modified_fraction = len(modified) / max(raw.size, 1)
    normalized_regions = _region_fractions(_contiguous_regions(modified), raw.size)
    targeting_modes = {
        operator_targeting_mode(operator_id) for operator_id, _params in steps
    }
    intrinsically_targeted = targeting_modes == {"intrinsic"}
    outside = bool(
        inspected_regions
        and not intrinsically_targeted
        and any(not _inside_regions(index, inspected_regions) for index in modified)
    )
    finite = bool(np.isfinite(output).all())
    equivalent = effect_equivalent_to_identity(raw, output)

    rejection_code = ""
    if require_finite_output and not finite:
        rejection_code = "NONFINITE_OUTPUT"
    elif modified_fraction > float(maximum_modified_fraction):
        rejection_code = "MODIFICATION_FRACTION_EXCEEDED"
    elif preserve_outside_inspected_region and outside:
        rejection_code = "OUTSIDE_SCOPE_MODIFICATION"

    warnings = ("EFFECT_EQUIVALENT_TO_IDENTITY",) if equivalent else ()
    status = "rejected" if rejection_code else ("warning" if warnings else "valid")
    receipt = CandidateVerificationReceipt(
        candidate_id=candidate.candidate_id,
        candidate_kind=candidate.kind.value,
        status=status,
        program_sha=candidate.program.sha(),
        operator_legality_ok=True,
        compilation_ok=True,
        execution_ok=True,
        shape_preserved=True,
        finite_output=finite,
        effect_equivalent_to_identity=equivalent,
        modified_fraction=modified_fraction,
        modified_region_fractions=normalized_regions,
        outside_inspected_region_modified=outside,
        warnings=warnings if not rejection_code else (),
        rejection_code=rejection_code,
    )
    return CandidateExecutionArtifact(
        candidate,
        receipt,
        output,
        trace,
        modified,
    )


__all__ = [
    "CandidateExecutionArtifact",
    "CandidateVerificationReceipt",
    "RECEIPT_STATUSES",
    "REJECTION_CODES",
    "WARNING_CODES",
    "verify_candidate",
]
