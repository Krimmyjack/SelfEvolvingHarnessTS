from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.contracts.harness import HarnessSnapshot
from SelfEvolvingHarnessTS.contracts.method import (
    ExecutionReceipt,
    PreparationRequest,
    PreparationResult,
    PreparationStatus,
    PreparedSeries,
)
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.runtime.candidate_pool import (
    CandidatePool,
    ProtocolChoiceError,
    effect_equivalent_to_identity,
)
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace
from SelfEvolvingHarnessTS.runtime.executor import ExecutionResult, run_pipeline

from .agent_core import AgentProtocolError, AgentRole, AgentStageResult, TTHAAgentCore
from .public_tools import extract_public_features
from .retrieval import EffectiveHarnessView, resolve_harness_view


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def _allowed_operators(request: PreparationRequest) -> tuple[str, ...]:
    allowed: list[str] = []
    for name in OPERATOR_NAMES:
        metadata = OPERATOR_METADATA[name]
        if request.task_spec.task_type not in metadata["allowed_tasks"]:
            continue
        if metadata.get("shape_changing"):
            continue
        if request.task_spec.is_op_forbidden(name):
            continue
        allowed.append(name)
    return tuple(allowed)


def _operator_public_contract(name: str) -> dict[str, object]:
    metadata = OPERATOR_METADATA[name]
    return {
        "name": name,
        "category": metadata["category"],
        "allowed_tasks": list(metadata["allowed_tasks"]),
        "destructive": metadata["destructive"],
        "preserves_observed": metadata["preserves_observed"],
        "changes_target_space": metadata["changes_target_space"],
        "requires_dependency": metadata["requires_dependency"],
        "dependency_policy": metadata["dependency_policy"],
    }


def _compile_candidates(
    payload: Mapping[str, object],
    request: PreparationRequest,
) -> tuple[Candidate, ...]:
    allowed = set(_allowed_operators(request))
    candidates: list[Candidate] = []
    for candidate_payload in payload["candidates"]:
        candidate_id = candidate_payload["candidate_id"]
        if candidate_id == "identity":
            raise AgentProtocolError("Agent cannot supply runtime identity")
        steps: list[tuple[str, Mapping[str, object]]] = []
        for step in candidate_payload["steps"]:
            op = step["op"]
            params = step["params"]
            if op not in allowed:
                raise AgentProtocolError(f"operator is not allowed for task: {op}")
            canonical_json_bytes(params)
            steps.append((op, params))
        program = Program.from_steps(steps, source="agent")
        candidates.append(
            Candidate.program_candidate(candidate_id, program, source="agent")
        )
    return tuple(candidates)


def _regions_from_fractions(
    fractions: Sequence[Sequence[float]],
    length: int,
) -> tuple[tuple[int, int], ...]:
    regions: list[tuple[int, int]] = []
    for start_fraction, end_fraction in fractions:
        if float(end_fraction) <= float(start_fraction):
            raise AgentProtocolError("inspected region end must be after start")
        start = min(length - 1, max(0, int(math.floor(float(start_fraction) * length))))
        end = min(length, max(start + 1, int(math.ceil(float(end_fraction) * length))))
        regions.append((start, end))
    return tuple(regions)


def _modified_indices(raw: np.ndarray, prepared: np.ndarray) -> tuple[int, ...]:
    if raw.shape != prepared.shape:
        return tuple(range(max(raw.size, prepared.size)))
    equal = np.equal(raw, prepared) | (np.isnan(raw) & np.isnan(prepared))
    return tuple(int(index) for index in np.flatnonzero(~equal))


def _inside_regions(index: int, regions: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= index < end for start, end in regions)


def _risk_allows(
    candidate: Candidate,
    values: np.ndarray,
    view: EffectiveHarnessView,
    inspected_regions: tuple[tuple[int, int], ...],
) -> bool:
    if candidate.kind is CandidateKind.IDENTITY:
        return True
    assert candidate.program is not None
    execution = run_pipeline(
        candidate.program.execution_steps(), values, source=candidate.source
    )
    if not execution.ok or execution.artifact is None or execution.artifact.shape != values.shape:
        return False
    modified = _modified_indices(values, execution.artifact)
    verification = view.controls.get("verification", {})
    if not isinstance(verification, Mapping):
        return False
    maximum = float(verification.get("max_modified_fraction", 1.0))
    if len(modified) / max(values.size, 1) > maximum:
        return False
    if verification.get("preserve_outside_candidate_region") is True and inspected_regions:
        if any(not _inside_regions(index, inspected_regions) for index in modified):
            return False
    return True


class TTHAFastAgent:
    def __init__(self, core: TTHAAgentCore):
        self.core = core

    def _trace(
        self,
        *,
        request: PreparationRequest,
        view: EffectiveHarnessView,
        stages: Sequence[AgentStageResult],
        inspected_regions: tuple[tuple[int, int], ...],
        pool: CandidatePool | None,
        chosen_candidate_id: str,
        compilation_status: str,
        execution_status: str,
        modified_indices: tuple[int, ...],
        verification_actions: tuple[str, ...],
        identity_equivalent: bool,
    ) -> DecisionTrace:
        tool_calls = tuple(
            {
                "tool_name": receipt.tool_name,
                "arguments": _plain(receipt.arguments),
                "public_result": _plain(receipt.public_result),
                "receipt_sha": receipt.receipt_sha,
            }
            for stage in stages
            for receipt in stage.tool_receipts
        )
        observation_ids = tuple(
            request_hash
            for stage in stages
            for request_hash in stage.request_hashes
        )
        candidates = pool.candidates if pool is not None else (Candidate.identity(),)
        candidate_program_steps = {
            candidate.candidate_id: tuple(
                (op, params) for op, params in candidate.program.execution_steps()
            )
            for candidate in candidates
            if candidate.program is not None
        }
        cache_hit_flags = tuple(
            bool(
                getattr(stage.response.cache_receipt, "hit", False)
                if stage.response.cache_receipt is not None
                else False
            )
            for stage in stages
            for _request_hash in stage.request_hashes
        )
        return DecisionTrace(
            case_id=request.series_uid,
            public_observation_ids=observation_ids,
            inspected_regions=inspected_regions,
            tool_calls=tool_calls,
            retrieved_skill_ids=view.skill_ids,
            retrieved_memory_ids=view.memory_ids,
            applicability_matches=tuple(
                entry_id for entry_id in (*view.skill_ids, *view.memory_ids)
            ),
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            candidate_program_shas=tuple(
                candidate.program.sha() if candidate.program is not None else None
                for candidate in candidates
            ),
            chosen_candidate_id=chosen_candidate_id,
            compilation_status=compilation_status,
            execution_status=execution_status,
            modified_indices=modified_indices,
            verification_actions=verification_actions,
            effect_equivalent_to_identity=identity_equivalent,
            series_length=request.values.size,
            candidate_program_steps=candidate_program_steps,
            agent_cache_hit_flags=cache_hit_flags,
        )

    def prepare(
        self,
        request: PreparationRequest,
        snapshot: HarnessSnapshot,
        *,
        fixed_probe_panel: Mapping[str, object] | None = None,
    ) -> tuple[PreparationResult, DecisionTrace]:
        verifier = getattr(self.core.tools, "verify_context", None)
        if verifier is not None and not verifier(
            request.values,
            task_kind=request.task_spec.task_type,
            fixed_probe_panel=fixed_probe_panel,
        ):
            raise ValueError("public tool context does not match preparation request")
        features = extract_public_features(
            request.values,
            task_kind=request.task_spec.task_type,
            fixed_probe_panel=fixed_probe_panel,
        )
        view = resolve_harness_view(snapshot, features, role="fast")
        stages: list[AgentStageResult] = []
        inspected_regions: tuple[tuple[int, int], ...] = ()
        pool: CandidatePool | None = None
        chosen_id = ""
        verification_actions: tuple[str, ...] = ()
        compilation_status = "not_started"
        try:
            inspect = self.core.run_stage(
                role=AgentRole.FAST,
                stage="inspect",
                case_id=request.series_uid,
                public_input={
                    "task": request.task_spec.to_dict(),
                    "features": _plain(features),
                    "fixed_probe_panel": _plain(fixed_probe_panel or {}),
                },
                harness_view=view,
                output_schema_name="fast_inspect_v1",
                output_schema=self.core.load_stage_schema("fast_inspect_v1"),
                source_snapshot_sha=snapshot.runtime_bundle_sha,
            )
            stages.append(inspect)
            inspected_regions = _regions_from_fractions(
                inspect.payload["inspected_region_fractions"], request.values.size
            )
            allowed = _allowed_operators(request)
            propose = self.core.run_stage(
                role=AgentRole.FAST,
                stage="propose",
                case_id=request.series_uid,
                public_input={
                    "features": _plain(features),
                    "inspection": _plain(inspect.payload),
                    "allowed_operator_contracts": [
                        _operator_public_contract(name) for name in allowed
                    ],
                },
                harness_view=view,
                output_schema_name="fast_propose_v1",
                output_schema=self.core.load_stage_schema("fast_propose_v1"),
                source_snapshot_sha=snapshot.runtime_bundle_sha,
            )
            stages.append(propose)
            supplied = _compile_candidates(propose.payload, request)
            total_k = int(snapshot.candidate_policy["total_k"])
            pool = CandidatePool.build(supplied, total_k=total_k)
            pool = pool.apply_risk(
                lambda candidate: _risk_allows(
                    candidate, request.values, view, inspected_regions
                )
            )
            compilation_status = "ok"
            public_candidates = [
                {
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind.value,
                    "program_sha": candidate.program.sha() if candidate.program else None,
                    "steps": (
                        [
                            {"op": op, "params": params}
                            for op, params in candidate.program.execution_steps()
                        ]
                        if candidate.program
                        else []
                    ),
                }
                for candidate in pool.candidates
            ]
            select = self.core.run_stage(
                role=AgentRole.FAST,
                stage="select",
                case_id=request.series_uid,
                public_input={
                    "features": _plain(features),
                    "inspection": _plain(inspect.payload),
                    "candidates": public_candidates,
                },
                harness_view=view,
                output_schema_name="fast_select_v1",
                output_schema=self.core.load_stage_schema("fast_select_v1"),
                source_snapshot_sha=snapshot.runtime_bundle_sha,
            )
            stages.append(select)
            chosen_id = select.payload["chosen_candidate_id"]
            verification_actions = tuple(select.payload["verification_actions"])
            chosen = pool.require_choice(chosen_id)
        except (AgentProtocolError, ProtocolChoiceError, ValueError, TypeError) as exc:
            trace = self._trace(
                request=request,
                view=view,
                stages=stages,
                inspected_regions=inspected_regions,
                pool=pool,
                chosen_candidate_id=chosen_id,
                compilation_status="failed" if compilation_status != "ok" else compilation_status,
                execution_status="not_started",
                modified_indices=(),
                verification_actions=verification_actions,
                identity_equivalent=False,
            )
            return (
                PreparationResult(
                    PreparationStatus.FAILED,
                    None,
                    None,
                    ExecutionReceipt(ok=False, error=f"AgentProtocolError: {exc}"),
                ),
                trace,
            )
        if chosen.kind is CandidateKind.IDENTITY:
            prepared_values = request.values.copy()
            prepared = PreparedSeries(
                request.series_uid, prepared_values, (), "original_units"
            )
            trace = self._trace(
                request=request,
                view=view,
                stages=stages,
                inspected_regions=inspected_regions,
                pool=pool,
                chosen_candidate_id=chosen_id,
                compilation_status=compilation_status,
                execution_status="ok",
                modified_indices=(),
                verification_actions=verification_actions,
                identity_equivalent=True,
            )
            return (
                PreparationResult(
                    PreparationStatus.ABSTAINED,
                    prepared,
                    None,
                    ExecutionReceipt(ok=True),
                ),
                trace,
            )
        assert chosen.program is not None
        execution: ExecutionResult = run_pipeline(
            chosen.program.execution_steps(), request.values, source=chosen.source
        )
        receipt = ExecutionReceipt(
            ok=execution.ok,
            error=execution.error,
            trace=tuple(dict(row) for row in execution.trace),
        )
        if not execution.ok or execution.artifact is None:
            trace = self._trace(
                request=request,
                view=view,
                stages=stages,
                inspected_regions=inspected_regions,
                pool=pool,
                chosen_candidate_id=chosen_id,
                compilation_status=compilation_status,
                execution_status="failed",
                modified_indices=(),
                verification_actions=verification_actions,
                identity_equivalent=False,
            )
            return (
                PreparationResult(
                    PreparationStatus.FAILED, None, chosen.program, receipt
                ),
                trace,
            )
        modified = _modified_indices(request.values, execution.artifact)
        equivalent = effect_equivalent_to_identity(request.values, execution.artifact)
        prepared = PreparedSeries(
            request.series_uid,
            execution.artifact,
            tuple(step.op for step in chosen.program.steps),
            "original_units",
        )
        trace = self._trace(
            request=request,
            view=view,
            stages=stages,
            inspected_regions=inspected_regions,
            pool=pool,
            chosen_candidate_id=chosen_id,
            compilation_status=compilation_status,
            execution_status="ok",
            modified_indices=modified,
            verification_actions=verification_actions,
            identity_equivalent=equivalent,
        )
        return (
            PreparationResult(
                PreparationStatus.PREPARED, prepared, chosen.program, receipt
            ),
            trace,
        )


__all__ = ["TTHAFastAgent"]
