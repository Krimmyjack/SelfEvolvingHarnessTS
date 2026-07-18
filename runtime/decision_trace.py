from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import canonical_sha256


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(nested) for key, nested in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested) for nested in value)
    return value


@dataclass(frozen=True)
class DecisionTrace:
    case_id: str
    public_observation_ids: tuple[str, ...]
    inspected_regions: tuple[tuple[int, int], ...]
    tool_calls: tuple[Mapping[str, object], ...]
    retrieved_skill_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    applicability_matches: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_program_shas: tuple[str | None, ...]
    chosen_candidate_id: str
    compilation_status: str
    execution_status: str
    modified_indices: tuple[int, ...]
    verification_actions: tuple[str, ...]
    effect_equivalent_to_identity: bool
    series_length: int | None = None
    # Grader-only execution material. It is never included in the normalized
    # behavior signature or any Agent-facing payload.
    candidate_program_steps: Mapping[
        str, tuple[tuple[str, Mapping[str, object]], ...]
    ] = field(default_factory=dict)
    agent_cache_hit_flags: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("DecisionTrace case_id must be non-empty")
        if len(self.candidate_ids) != len(self.candidate_program_shas):
            raise ValueError("candidate IDs and program SHAs must align")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("DecisionTrace candidate IDs must be unique")
        if not set(self.candidate_program_steps).issubset(set(self.candidate_ids)):
            raise ValueError("candidate program material must name a supplied candidate")
        if not isinstance(self.chosen_candidate_id, str):
            raise ValueError("chosen_candidate_id must be a string")
        for start, end in self.inspected_regions:
            if isinstance(start, bool) or isinstance(end, bool) or start < 0 or end <= start:
                raise ValueError("inspected regions must be non-empty non-negative intervals")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.modified_indices
        ):
            raise ValueError("modified indices must be non-negative integers")
        if tuple(sorted(set(self.modified_indices))) != self.modified_indices:
            raise ValueError("modified indices must be sorted and unique")
        if self.series_length is not None:
            if (
                isinstance(self.series_length, bool)
                or not isinstance(self.series_length, int)
                or self.series_length < 1
            ):
                raise ValueError("series_length must be a positive integer")
            if any(end > self.series_length for _, end in self.inspected_regions):
                raise ValueError("inspected region exceeds series length")
            if self.modified_indices and self.modified_indices[-1] >= self.series_length:
                raise ValueError("modified index exceeds series length")
        if not self.compilation_status or not self.execution_status:
            raise ValueError("compile and execution status must be non-empty")
        object.__setattr__(
            self,
            "tool_calls",
            tuple(_freeze_json(call) for call in self.tool_calls),
        )
        object.__setattr__(
            self,
            "candidate_program_steps",
            _freeze_json(self.candidate_program_steps),
        )


def _series_denominator(trace: DecisionTrace) -> int:
    if trace.series_length is not None:
        return trace.series_length
    endpoints = [end for _, end in trace.inspected_regions]
    if trace.modified_indices:
        endpoints.append(trace.modified_indices[-1] + 1)
    return max(endpoints, default=1)


def _region_fractions(
    regions: Sequence[tuple[int, int]],
    denominator: int,
) -> list[list[float]]:
    return [
        [round(start / denominator, 6), round(end / denominator, 6)]
        for start, end in regions
    ]


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


@dataclass(frozen=True)
class BehaviorSignature:
    normalized_behavior: Mapping[str, object]
    behavior_signature_sha: str

    @classmethod
    def from_trace(cls, trace: DecisionTrace) -> "BehaviorSignature":
        denominator = _series_denominator(trace)
        tool_names: list[str] = []
        for call in trace.tool_calls:
            name = call.get("tool_name", call.get("name"))
            if isinstance(name, str):
                tool_names.append(name)
        normalized = {
            "schema_version": "behavior-signature/1",
            "inspected_region_fractions": _region_fractions(
                trace.inspected_regions, denominator
            ),
            "tool_names": tool_names,
            "retrieved_skill_ids": list(trace.retrieved_skill_ids),
            "retrieved_memory_ids": list(trace.retrieved_memory_ids),
            "candidate_program_shas": list(trace.candidate_program_shas),
            "chosen_candidate_id": trace.chosen_candidate_id,
            "compilation_status": trace.compilation_status,
            "execution_status": trace.execution_status,
            "modified_region_fractions": _region_fractions(
                _contiguous_regions(trace.modified_indices), denominator
            ),
            "verification_actions": list(trace.verification_actions),
            "effect_equivalent_to_identity": trace.effect_equivalent_to_identity,
        }
        return cls(
            normalized_behavior=_freeze_json(normalized),
            behavior_signature_sha=canonical_sha256(normalized),
        )

    @property
    def sha(self) -> str:
        return self.behavior_signature_sha


__all__ = ["BehaviorSignature", "DecisionTrace"]
