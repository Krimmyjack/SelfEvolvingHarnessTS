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
from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA,
    OPERATOR_NAMES,
    operator_targeting_mode,
)
from SelfEvolvingHarnessTS.runtime.candidate_pool import (
    CandidatePool,
    ProtocolChoiceError,
)
from SelfEvolvingHarnessTS.runtime.candidate_verification import (
    CandidateExecutionArtifact,
    verify_candidate,
)
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

from .agent_core import (
    AgentProtocolError,
    AgentRole,
    AgentStageResult,
    StagePostValidationError,
    TTHAAgentCore,
)
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


def public_operator_contract(name: str) -> dict[str, object]:
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
        "public_parameter_bindings": dict(
            metadata.get("public_parameter_bindings", {})
        ),
        "public_parameter_schema": _plain(
            metadata.get("public_parameter_schema") or {"type": "object"}
        ),
        "targeting_mode": operator_targeting_mode(name),
    }


def public_operator_contracts_for_task(task_kind: str) -> tuple[dict[str, object], ...]:
    """Return the deployment-safe operator menu from the registry single source."""

    return tuple(
        public_operator_contract(name)
        for name in OPERATOR_NAMES
        if task_kind in OPERATOR_METADATA[name]["allowed_tasks"]
        and not OPERATOR_METADATA[name].get("shape_changing")
    )


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


def _validate_public_parameter_bindings(
    payload: Mapping[str, object],
    public_features: Mapping[str, object],
    fixed_probe_panel: Mapping[str, object] | None = None,
) -> None:
    fixed_step_signatures: set[bytes] = set()
    contracts = (fixed_probe_panel or {}).get("probe_contracts", {})
    probes = contracts.get("probes", {}) if isinstance(contracts, Mapping) else {}
    if isinstance(probes, Mapping):
        for probe in probes.values():
            if not isinstance(probe, Mapping):
                continue
            arms = probe.get("arms", ())
            if not isinstance(arms, Sequence) or isinstance(
                arms, (str, bytes, bytearray)
            ):
                continue
            for arm in arms:
                if not isinstance(arm, Mapping):
                    continue
                steps = arm.get("current_context_program_steps", ())
                if not isinstance(steps, Sequence) or isinstance(
                    steps, (str, bytes, bytearray)
                ):
                    continue
                for step in steps:
                    if isinstance(step, Mapping):
                        fixed_step_signatures.add(canonical_json_bytes(step))
    for candidate in payload["candidates"]:
        for step in candidate["steps"]:
            operator_name = step["op"]
            bindings = OPERATOR_METADATA[operator_name].get(
                "public_parameter_bindings", {}
            )
            if not bindings:
                continue
            if canonical_json_bytes(step) in fixed_step_signatures:
                continue
            params = step["params"]
            expected_keys = set(bindings)
            if set(params) != expected_keys:
                raise StagePostValidationError(
                    "PUBLIC_PARAMETER_BINDING_INVALID",
                    (
                        f"{operator_name} params must contain exactly the canonical "
                        f"keys {sorted(expected_keys)} declared in its public parameter "
                        "bindings."
                    ),
                    retryable=True,
                )
            mismatched = [
                parameter
                for parameter, feature in bindings.items()
                if feature not in public_features
                or params[parameter] != public_features[feature]
            ]
            if mismatched:
                raise StagePostValidationError(
                    "PUBLIC_PARAMETER_BINDING_INVALID",
                    (
                        f"{operator_name} bound parameter values must exactly equal "
                        "their deployment-visible feature values from the declared "
                        f"mapping; mismatched keys: {sorted(mismatched)}."
                    ),
                    retryable=True,
                )


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


def _verification_limits(
    request: PreparationRequest,
    view: EffectiveHarnessView,
) -> tuple[float, bool]:
    verification = view.controls.get("verification", {})
    if not isinstance(verification, Mapping):
        return 0.0, True
    maxima = [float(verification.get("max_modified_fraction", 1.0))]
    if request.task_context is not None:
        maxima.append(
            float(
                request.task_context.deployment_constraints.maximum_modified_fraction
            )
        )
    preserve_outside = verification.get("preserve_outside_candidate_region") is True
    for skill in view.skills:
        guards = skill.risk_guards
        if not isinstance(guards, Mapping):
            continue
        skill_maximum = guards.get("max_modified_fraction")
        if (
            isinstance(skill_maximum, (int, float))
            and not isinstance(skill_maximum, bool)
            and math.isfinite(float(skill_maximum))
        ):
            maxima.append(float(skill_maximum))
        preserve_outside = preserve_outside or (
            guards.get("preserve_outside_candidate_region") is True
        )
    return min(maxima), preserve_outside


def _task_binding(
    request: PreparationRequest, *, legacy_inspect_stage: bool = False
) -> dict[str, object]:
    if request.task_context is None:
        return {"task": request.task_spec.to_dict()} if legacy_inspect_stage else {}
    return {
        "task": request.task_spec.to_dict(),
        "task_context": request.task_context.to_dict(),
        "task_context_sha": request.task_context.sha(),
    }


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
        supplied_noop_candidate_ids: tuple[str, ...],
        candidate_artifacts: Mapping[str, CandidateExecutionArtifact],
        rejection_receipts: tuple[Mapping[str, object], ...],
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
            supplied_noop_candidate_ids=supplied_noop_candidate_ids,
            candidate_program_steps=candidate_program_steps,
            agent_cache_hit_flags=cache_hit_flags,
            task_context_sha=(request.task_context.sha() if request.task_context else ""),
            run_context_sha=(
                request.run_dependency_binding.sha()
                if request.run_dependency_binding
                else ""
            ),
            selectable_candidate_ids=tuple(
                candidate.candidate_id for candidate in candidates
            ),
            candidate_receipt_shas={
                candidate_id: artifact.receipt.receipt_sha
                for candidate_id, artifact in candidate_artifacts.items()
                if request.task_context is not None
                and candidate_id in {candidate.candidate_id for candidate in candidates}
            },
            rejection_receipts=(
                rejection_receipts if request.task_context is not None else ()
            ),
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
        supplied_noop_candidate_ids: tuple[str, ...] = ()
        candidate_artifacts: dict[str, CandidateExecutionArtifact] = {}
        rejection_receipts: tuple[Mapping[str, object], ...] = ()
        chosen_artifact: CandidateExecutionArtifact | None = None
        compilation_status = "not_started"
        task_context_sha = request.task_context.sha() if request.task_context else ""
        run_context_sha = (
            request.run_dependency_binding.sha()
            if request.run_dependency_binding
            else ""
        )
        try:
            inspect = self.core.run_stage(
                role=AgentRole.FAST,
                stage="inspect",
                case_id=request.series_uid,
                public_input={
                    **_task_binding(request, legacy_inspect_stage=True),
                    "features": _plain(features),
                    "fixed_probe_panel": _plain(fixed_probe_panel or {}),
                },
                harness_view=view,
                output_schema_name="fast_inspect_v1",
                output_schema=self.core.load_stage_schema("fast_inspect_v1"),
                source_snapshot_sha=snapshot.runtime_bundle_sha,
                task_context_sha=task_context_sha,
                run_context_sha=run_context_sha,
                validation_retries=1,
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
                    **_task_binding(request),
                    "features": _plain(features),
                    "inspection": _plain(inspect.payload),
                    "fixed_probe_panel": _plain(fixed_probe_panel or {}),
                    "allowed_operator_contracts": [
                        public_operator_contract(name) for name in allowed
                    ],
                },
                harness_view=view,
                output_schema_name="fast_propose_v1",
                output_schema=self.core.load_stage_schema("fast_propose_v1"),
                source_snapshot_sha=snapshot.runtime_bundle_sha,
                task_context_sha=task_context_sha,
                run_context_sha=run_context_sha,
                validation_retries=1,
                post_validator=lambda payload: _validate_public_parameter_bindings(
                    payload, features, fixed_probe_panel
                ),
            )
            stages.append(propose)
            supplied = _compile_candidates(propose.payload, request)
            total_k = int(snapshot.candidate_policy["total_k"])
            if request.task_context is not None:
                total_k = min(
                    total_k,
                    request.task_context.deployment_constraints.maximum_candidates,
                )
            pool = CandidatePool.build(supplied, total_k=total_k)
            maximum_modified_fraction, preserve_outside = _verification_limits(
                request, view
            )
            verified = tuple(
                verify_candidate(
                    candidate,
                    request.values,
                    allowed_operators=allowed,
                    inspected_regions=inspected_regions,
                    maximum_modified_fraction=maximum_modified_fraction,
                    preserve_outside_inspected_region=preserve_outside,
                    require_finite_output=request.task_context is not None,
                )
                for candidate in pool.candidates
            )
            candidate_artifacts = {
                artifact.candidate.candidate_id: artifact for artifact in verified
            }
            supplied_noop_candidate_ids = tuple(
                artifact.candidate.candidate_id
                for artifact in verified
                if artifact.candidate.kind is CandidateKind.PROGRAM
                and artifact.receipt.effect_equivalent_to_identity
            )
            rejection_receipts = tuple(
                artifact.receipt.to_dict()
                for artifact in verified
                if not artifact.selectable
            )
            pool = CandidatePool(
                tuple(
                    artifact.candidate for artifact in verified if artifact.selectable
                ),
                total_k,
            )
            compilation_status = "ok"
            public_candidates = []
            for candidate in pool.candidates:
                candidate_payload = {
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
                if request.task_context is not None:
                    artifact = candidate_artifacts[candidate.candidate_id]
                    candidate_payload["verification_receipt"] = (
                        artifact.receipt.to_dict()
                    )
                    candidate_payload["verification_receipt_sha"] = (
                        artifact.receipt.receipt_sha
                    )
                public_candidates.append(candidate_payload)
            select = self.core.run_stage(
                role=AgentRole.FAST,
                stage="select",
                case_id=request.series_uid,
                public_input={
                    **_task_binding(request),
                    "features": _plain(features),
                    "inspection": _plain(inspect.payload),
                    "fixed_probe_panel": _plain(fixed_probe_panel or {}),
                    "candidates": public_candidates,
                },
                harness_view=view,
                output_schema_name="fast_select_v1",
                output_schema=self.core.load_stage_schema("fast_select_v1"),
                source_snapshot_sha=snapshot.runtime_bundle_sha,
                task_context_sha=task_context_sha,
                run_context_sha=run_context_sha,
                validation_retries=1,
            )
            stages.append(select)
            chosen_id = select.payload["chosen_candidate_id"]
            verification_actions = tuple(select.payload["verification_actions"])
            chosen = pool.require_choice(chosen_id)
            chosen_artifact = candidate_artifacts[chosen.candidate_id]
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
                supplied_noop_candidate_ids=supplied_noop_candidate_ids,
                candidate_artifacts=candidate_artifacts,
                rejection_receipts=rejection_receipts,
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
                supplied_noop_candidate_ids=supplied_noop_candidate_ids,
                candidate_artifacts=candidate_artifacts,
                rejection_receipts=rejection_receipts,
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
        assert chosen_artifact is not None and chosen_artifact.prepared_values is not None
        receipt = ExecutionReceipt(
            ok=True,
            trace=tuple(dict(row) for row in chosen_artifact.execution_trace),
        )
        modified = chosen_artifact.modified_indices
        equivalent = chosen_artifact.receipt.effect_equivalent_to_identity
        prepared = PreparedSeries(
            request.series_uid,
            chosen_artifact.prepared_values,
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
            supplied_noop_candidate_ids=supplied_noop_candidate_ids,
            candidate_artifacts=candidate_artifacts,
            rejection_receipts=rejection_receipts,
        )
        return (
            PreparationResult(
                PreparationStatus.PREPARED, prepared, chosen.program, receipt
            ),
            trace,
        )


__all__ = [
    "TTHAFastAgent",
    "public_operator_contract",
    "public_operator_contracts_for_task",
]
