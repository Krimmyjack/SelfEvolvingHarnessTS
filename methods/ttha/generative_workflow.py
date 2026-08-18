"""Minimal catalog-free Workflow generation from the public operator registry.

The module owns only the generative control path.  An LLM proposes ordinary
JSON, an external Support adapter executes/scores the resulting Candidate, and
an unpromoted Skill draft is derived from that Action--Response trace.  Query
outcomes, promotion, persistence, and domain-specific Workflow templates stay
outside this module.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import public_operator_contract
from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA,
    OPERATOR_NAMES,
    TOOL_REGISTRY,
)


IDENTITY = "IDENTITY"
EXECUTABLE = "EXECUTABLE"
UNAVAILABLE = "UNAVAILABLE"

_SEALED_FIELD = re.compile(
    r"(?:^|_)(?:dataset_?id|oracle|outcome|query(?:_?future|_?values?)?|utility)(?:$|_)",
    re.IGNORECASE,
)

Proposer = Callable[[Mapping[str, object]], Mapping[str, object]]
SupportCallback = Callable[["CompiledWorkflow"], Mapping[str, object]]
MemoryWriter = Callable[[Mapping[str, object]], object]


@dataclass(frozen=True)
class CompiledWorkflow:
    """One LLM-generated Workflow compiled against the public inventory."""

    candidate: Candidate
    requested_observations: tuple[str, ...]
    template_steps: tuple[Mapping[str, object], ...]
    fallback: str = IDENTITY
    experience_use: tuple[str, ...] = ()


class CandidateCompilationError(ValueError):
    """A single generated candidate was invalid but the LLM protocol is intact."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _plain_json(value: object, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc


def _reject_sealed_fields(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(key)).strip("_")
            if _SEALED_FIELD.search(normalized):
                raise ValueError(f"sealed Query/outcome field is forbidden at {path}.{key}")
            _reject_sealed_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_sealed_fields(nested, path=f"{path}[{index}]")


def _context_value(public_context: Mapping[str, object], path: str) -> object:
    current: object = public_context
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise KeyError(path)
        current = current[segment]
    return current


def _canonical_context_path(path: str) -> str:
    """Normalize the one public namespace prefix accepted by the JSON contract."""

    prefix = "public_context."
    relative = path[len(prefix) :] if path.startswith(prefix) else path
    if (
        not relative
        or relative.startswith(prefix)
        or any(not segment for segment in relative.split("."))
    ):
        raise ValueError("binding must be a non-empty relative public Context path")
    return relative


def _dependency_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _runtime_operator_summary(name: str) -> dict[str, object]:
    """Extract public effect/parameter hints from the registered callable."""

    operator = TOOL_REGISTRY[name]
    doc = inspect.getdoc(operator) or ""
    first_paragraph = " ".join(doc.split("\n\n", 1)[0].split())
    parameters: list[dict[str, object]] = []
    try:
        signature = inspect.signature(operator)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        for parameter in signature.parameters.values():
            if parameter.name == "x" or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            row: dict[str, object] = {
                "name": parameter.name,
                "required": parameter.default is inspect.Parameter.empty,
            }
            if parameter.default is not inspect.Parameter.empty:
                try:
                    row["default"] = _plain_json(
                        parameter.default, name=f"{name}.{parameter.name} default"
                    )
                except ValueError:
                    row["default_type"] = type(parameter.default).__name__
            parameters.append(row)
    return {
        "effect": first_paragraph[:280],
        "runtime_parameters": parameters,
    }


def build_public_operator_inventory(
    task_kind: str,
    public_context: Mapping[str, object],
    *,
    forbidden_operators: Sequence[str] = (),
) -> tuple[dict[str, object], ...]:
    """Expose every canonical operator and explain why each can or cannot run."""

    if not isinstance(task_kind, str) or not task_kind:
        raise ValueError("task_kind must be a non-empty string")
    context = _plain_json(public_context, name="public_context")
    if not isinstance(context, dict):
        raise ValueError("public_context must be an object")
    _reject_sealed_fields(context, path="public_context")
    forbidden = set(forbidden_operators)
    unknown_forbidden = forbidden - set(OPERATOR_NAMES)
    if unknown_forbidden:
        raise ValueError(
            "forbidden_operators contains unknown canonical operators: "
            + ", ".join(sorted(unknown_forbidden))
        )

    inventory: list[dict[str, object]] = []
    for name in OPERATOR_NAMES:
        metadata = OPERATOR_METADATA[name]
        reasons: list[str] = []
        if task_kind not in metadata["allowed_tasks"]:
            reasons.append(f"TASK_NOT_ALLOWED:{task_kind}")
        if name in forbidden:
            reasons.append("TASK_FORBIDDEN")
        if metadata.get("shape_changing"):
            reasons.append("SHAPE_CHANGING_RUNTIME_UNSUPPORTED")
        if metadata.get("changes_target_space"):
            reasons.append("CHANGES_TARGET_SPACE_RUNTIME_UNSUPPORTED")
        bindings = metadata.get("public_parameter_bindings", {})
        missing_bindings = [
            path
            for path in bindings.values()
            if not _has_context_value(context, str(path))
        ]
        if missing_bindings:
            reasons.append("MISSING_PUBLIC_BINDINGS:" + ",".join(missing_bindings))
        dependency = metadata.get("requires_dependency")
        if (
            isinstance(dependency, str)
            and metadata.get("dependency_policy") == "hard_fail"
            and not _dependency_available(dependency)
        ):
            reasons.append(f"DEPENDENCY_UNAVAILABLE:{dependency}")

        row = public_operator_contract(name)
        row.update(
            {
                **_runtime_operator_summary(name),
                "shape_changing": bool(metadata.get("shape_changing")),
                "availability": UNAVAILABLE if reasons else EXECUTABLE,
                "reason": ";".join(reasons) if reasons else "AVAILABLE",
            }
        )
        inventory.append(row)
    return tuple(inventory)


def _has_context_value(public_context: Mapping[str, object], path: str) -> bool:
    try:
        _context_value(public_context, path)
    except KeyError:
        return False
    return True


def _validate_schema_value(value: object, schema: Mapping[str, object], *, path: str) -> None:
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "null": value is None,
        None: True,
    }.get(expected_type, False)
    if not type_ok:
        raise ValueError(f"{path} does not match parameter schema type {expected_type!r}")
    if expected_type == "number" and not math.isfinite(float(value)):  # type: ignore[arg-type]
        raise ValueError(f"{path} must be finite")
    if "enum" in schema and value not in schema["enum"]:  # type: ignore[operator]
        raise ValueError(f"{path} is outside the parameter enum")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:  # type: ignore[operator]
            raise ValueError(f"{path} is below the parameter minimum")
        if "maximum" in schema and value > schema["maximum"]:  # type: ignore[operator]
            raise ValueError(f"{path} is above the parameter maximum")
    if expected_type == "object":
        assert isinstance(value, Mapping)
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path} has an invalid public parameter schema")
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise ValueError(f"{path} has an invalid required-parameter declaration")
        missing = set(required) - set(value)
        if missing:
            raise ValueError(f"{path} is missing parameters: {sorted(missing)}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise ValueError(f"{path} contains undeclared parameters: {sorted(extra)}")
        for key, nested in value.items():
            nested_schema = properties.get(key)
            if isinstance(nested_schema, Mapping):
                _validate_schema_value(nested, nested_schema, path=f"{path}.{key}")
    if expected_type == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, nested in enumerate(value):
                _validate_schema_value(nested, item_schema, path=f"{path}[{index}]")


def compile_workflow_proposal(
    proposal: Mapping[str, object],
    inventory: Sequence[Mapping[str, object]],
    public_context: Mapping[str, object],
    *,
    generation: int,
) -> CompiledWorkflow:
    """Compile one catalog-free JSON proposal into the existing Program/Candidate."""

    if generation < 1:
        raise ValueError("generation must be positive")
    payload = _plain_json(proposal, name="workflow proposal")
    context = _plain_json(public_context, name="public_context")
    if not isinstance(payload, dict) or not isinstance(context, dict):
        raise ValueError("proposal and public_context must be objects")
    _reject_sealed_fields(payload, path="proposal")
    allowed_fields = {"decision", "steps", "requested_observations", "fallback", "experience_use"}
    if set(payload) - allowed_fields:
        raise ValueError("workflow proposal contains unsupported fields")
    if "experience_use" in payload and not isinstance(payload["experience_use"], list):
        raise ValueError("experience_use must be a list of episode IDs")
    if payload.get("decision") != "PROPOSE":
        raise ValueError("workflow proposal decision must be PROPOSE")
    if payload.get("fallback") != IDENTITY:
        raise ValueError("generated Workflow must preserve IDENTITY fallback")

    inventory_by_name = {str(row.get("name")): row for row in inventory}
    if set(inventory_by_name) != set(OPERATOR_NAMES):
        raise ValueError("operator inventory must contain every canonical operator exactly once")
    steps_payload = payload.get("steps")
    if not isinstance(steps_payload, list) or not 1 <= len(steps_payload) <= 4:
        raise ValueError("generated Workflow requires one to four operator steps")
    steps: list[tuple[str, Mapping[str, object]]] = []
    template_steps: list[dict[str, object]] = []
    for index, raw_step in enumerate(steps_payload):
        if not isinstance(raw_step, dict) or not {"op", "params"} <= set(raw_step):
            raise ValueError("each generated step requires op and params")
        if set(raw_step) - {"op", "params", "bindings"}:
            raise ValueError("generated step contains unsupported fields")
        op = raw_step["op"]
        params = raw_step["params"]
        if not isinstance(op, str) or op not in inventory_by_name:
            raise CandidateCompilationError(
                "UNKNOWN_OPERATOR", f"unknown canonical operator at steps[{index}]"
            )
        contract = inventory_by_name[op]
        if contract.get("availability") != EXECUTABLE:
            raise CandidateCompilationError(
                "OPERATOR_UNAVAILABLE",
                f"operator {op!r} is unavailable: {contract.get('reason', 'UNKNOWN')}"
            )
        if not isinstance(params, dict):
            raise CandidateCompilationError(
                "PARAMETERS_INVALID", f"steps[{index}].params must be an object"
            )
        proposed_bindings = raw_step.get("bindings", {})
        if (
            not isinstance(proposed_bindings, dict)
            or not all(
                isinstance(parameter, str)
                and parameter
                and isinstance(path, str)
                and path
                for parameter, path in proposed_bindings.items()
            )
        ):
            raise CandidateCompilationError(
                "BINDINGS_INVALID", f"steps[{index}].bindings must map parameters to Context paths"
            )
        try:
            proposed_bindings = {
                parameter: _canonical_context_path(context_path)
                for parameter, context_path in proposed_bindings.items()
            }
        except ValueError as exc:
            raise CandidateCompilationError(
                "BINDINGS_INVALID", f"steps[{index}].bindings {exc}"
            ) from exc
        overlap = set(params) & set(proposed_bindings)
        if overlap:
            raise CandidateCompilationError(
                "BINDINGS_INVALID",
                f"steps[{index}] repeats bound parameters as constants: {sorted(overlap)}",
            )
        resolved_params = copy.deepcopy(params)
        try:
            for parameter, context_path in proposed_bindings.items():
                resolved_params[parameter] = copy.deepcopy(
                    _context_value(context, context_path)
                )
        except KeyError as exc:
            raise CandidateCompilationError(
                "BINDING_PATH_UNAVAILABLE",
                f"steps[{index}] binding path is absent from public Context: {exc.args[0]}",
            ) from exc
        parameter_schema = contract.get("public_parameter_schema", {"type": "object"})
        if not isinstance(parameter_schema, Mapping):
            raise ValueError(f"operator {op!r} has an invalid public parameter schema")
        try:
            _validate_schema_value(
                resolved_params, parameter_schema, path=f"steps[{index}].params"
            )
        except ValueError as exc:
            raise CandidateCompilationError("PARAMETERS_INVALID", str(exc)) from exc
        declared_bindings = contract.get("public_parameter_bindings", {})
        if not isinstance(declared_bindings, Mapping):
            raise ValueError(f"operator {op!r} has invalid public bindings")
        for parameter, context_path in declared_bindings.items():
            canonical_path = _canonical_context_path(str(context_path))
            if proposed_bindings.get(parameter) != canonical_path:
                raise CandidateCompilationError(
                    "REQUIRED_BINDING_MISSING",
                    f"operator {op!r} parameter {parameter!r} must use declared public "
                    f"binding {canonical_path!r}",
                )
        steps.append((op, resolved_params))
        template_steps.append(
            {
                "op": op,
                "params": copy.deepcopy(params),
                "bindings": copy.deepcopy(proposed_bindings),
            }
        )

    observations = payload.get("requested_observations", [])
    if (
        not isinstance(observations, list)
        or len(observations) > 8
        or len(set(observations)) != len(observations)
        or not all(isinstance(value, str) and value.strip() == value and value for value in observations)
    ):
        raise ValueError("requested_observations must be at most eight unique canonical strings")
    program = Program.from_steps(steps, source="llm_generated")
    candidate = Candidate.program_candidate(
        f"generated-workflow-{generation}", program, source="llm_generated"
    )
    exp_use = payload.get("experience_use") or []
    exp_use = tuple(str(x) for x in exp_use if isinstance(x, str))
    return CompiledWorkflow(candidate, tuple(observations), tuple(template_steps), experience_use=exp_use)


def _proposal_schema() -> dict[str, object]:
    return {
        "decision": "PROPOSE | ABSTAIN",
        "steps": [
            {
                "op": "canonical operator name",
                "params": {"fixed_parameter": "JSON value"},
                "bindings": {
                    "portable_parameter": (
                        "relative dotted path; optional public_context. prefix accepted"
                    )
                },
            }
        ],
        "requested_observations": ["public observation id"],
        "fallback": IDENTITY,
        "constraints": {"minimum_steps": 1, "maximum_steps": 4},
        "experience_use": [
            "optional: episode IDs from experience_contrast_pack that this "
            "proposal used, modified or avoided (omit if none)"
        ],
    }


def resolve_generated_acquisition_lifecycle(
    source_generation_results: Sequence[Mapping[str, object]],
    slow_path_result: Mapping[str, object],
    confirmation_result: Mapping[str, object] | None,
    *,
    memory_writer: MemoryWriter | None = None,
) -> dict[str, object]:
    """Resolve generated acquisition evidence without caller-owned method choices.

    The controller consumes completed traces only.  Local singleton responses can
    establish proposal credit, while promotion depends on full-policy replays and
    an explicit confirmation result.  Persistence remains an injected callback.
    """

    checks: dict[str, object] = {
        "empty_capability_memory": False,
        "program_from_llm_trace": False,
        "patch_compilation_valid": False,
        "patch_non_dead": False,
        "singleton_credit_is_proposal_only": False,
        "full_scoped_retrain_is_policy_evidence": False,
        "positive_in_scope_policy_environment_count": 0,
        "confirmation_passed": False,
        "confirmation_harm": None,
    }
    resolution_scope: dict[str, object] | None = None
    contextual_episode: dict[str, object] | None = None

    def terminal(status: str, reason_code: str) -> dict[str, object]:
        return {
            "status": status,
            "reason_code": reason_code,
            "rejected_capability_version": (
                copy.deepcopy(resolution_scope)
                if status.startswith("REJECTED")
                else None
            ),
            "contextual_episode": copy.deepcopy(contextual_episode),
            "operator_blacklisted": False,
            "program_family_closed": False,
            "memory_write_authorized": False,
            "memory_write_count": 0,
            "generated_skill_card": None,
            "checks": copy.deepcopy(checks),
        }

    if (
        not isinstance(source_generation_results, Sequence)
        or isinstance(source_generation_results, (str, bytes))
        or len(source_generation_results) < 2
        or not all(isinstance(row, Mapping) for row in source_generation_results)
    ):
        return terminal("ABSTAINED", "SOURCE_GENERATION_EVIDENCE_INCOMPLETE")
    checks["empty_capability_memory"] = all(
        report.get("capability_memory_entry_count") == 0
        for report in source_generation_results
    )
    if not checks["empty_capability_memory"]:
        return terminal("ABSTAINED", "GENERATION_DID_NOT_START_FROM_EMPTY_MEMORY")

    proposal = slow_path_result.get("scope_proposal")
    if not isinstance(proposal, Mapping):
        return terminal("ABSTAINED", "TYPED_PATCH_MISSING")
    if proposal.get("decision") == "ABSTAIN":
        return terminal("ABSTAINED", "SLOW_PATH_ABSTAINED")
    target_op = proposal.get("program_op")
    if not isinstance(target_op, str) or not target_op:
        return terminal("ABSTAINED", "TYPED_PATCH_TARGET_MISSING")
    resolution_scope = {"program_operator": target_op, "typed_scope": None}
    discovered_op = slow_path_result.get(
        "common_program_discovered_from_generation_traces"
    )
    if discovered_op is not None and discovered_op != target_op:
        return terminal("ABSTAINED", "TYPED_PATCH_TARGET_TRACE_MISMATCH")

    generated_traces: list[dict[str, object]] = []
    for report in source_generation_results:
        llm = report.get("llm")
        proposals = report.get("generation_proposals")
        if (
            not isinstance(llm, Mapping)
            or llm.get("api_integrated") is not True
            or not isinstance(llm.get("generation_api_call_count"), int)
            or int(llm["generation_api_call_count"]) <= 0
            or not isinstance(proposals, Sequence)
            or isinstance(proposals, (str, bytes))
        ):
            return terminal("ABSTAINED", "LLM_GENERATION_TRACE_MISSING")
        generation_calls = llm.get("generation_calls")
        call_stages = {
            str(row.get("stage"))
            for row in generation_calls
            if isinstance(row, Mapping) and isinstance(row.get("stage"), str)
        } if isinstance(generation_calls, Sequence) and not isinstance(
            generation_calls, (str, bytes)
        ) else set()
        matching: list[Mapping[str, object]] = []
        for row in proposals:
            if not isinstance(row, Mapping):
                continue
            steps = row.get("compiled_program_steps")
            if (
                isinstance(steps, Sequence)
                and not isinstance(steps, (str, bytes))
                and any(
                    isinstance(step, Mapping) and step.get("op") == target_op
                    for step in steps
                )
                and isinstance(row.get("candidate_id"), str)
                and row.get("stage") in call_stages
            ):
                matching.append(row)
        if len(matching) != 1:
            return terminal("ABSTAINED", "GENERATED_PROGRAM_PROVENANCE_AMBIGUOUS")
        row = matching[0]
        generated_traces.append(
            {
                "candidate_id": row["candidate_id"],
                "stage": row["stage"],
                "workflow_steps": copy.deepcopy(row.get("workflow_steps")),
                "compiled_program_steps": copy.deepcopy(
                    row.get("compiled_program_steps")
                ),
            }
        )
    checks["program_from_llm_trace"] = True

    checks["patch_compilation_valid"] = (
        slow_path_result.get("compilation") == "VALID"
        and isinstance(slow_path_result.get("compiled_conditions"), Sequence)
        and not isinstance(slow_path_result.get("compiled_conditions"), (str, bytes))
        and bool(slow_path_result.get("compiled_conditions"))
    )
    if not checks["patch_compilation_valid"]:
        return terminal("REJECTED", "TYPED_PATCH_COMPILATION_INVALID")
    resolution_scope["typed_scope"] = copy.deepcopy(
        list(slow_path_result["compiled_conditions"])  # type: ignore[arg-type]
    )
    checks["patch_non_dead"] = slow_path_result.get("dead_patch") is False
    if not checks["patch_non_dead"]:
        return terminal("REJECTED", "TYPED_PATCH_DEAD")

    dossier = slow_path_result.get("scope_dossier_sent_to_proposer")
    singleton_responses: list[Mapping[str, object]] = []
    if isinstance(dossier, Sequence) and not isinstance(dossier, (str, bytes)):
        for environment in dossier:
            episodes = environment.get("episodes") if isinstance(environment, Mapping) else None
            if isinstance(episodes, Sequence) and not isinstance(episodes, (str, bytes)):
                singleton_responses.extend(
                    response
                    for episode in episodes
                    if isinstance(episode, Mapping)
                    for response in [episode.get("support_exact_singleton_response")]
                    if isinstance(response, Mapping)
                )
    checks["singleton_credit_is_proposal_only"] = bool(singleton_responses) and all(
        row.get("credit_level") == "PROPOSAL_ONLY_LOCAL_ACTION_EPISODE"
        for row in singleton_responses
    )
    if not checks["singleton_credit_is_proposal_only"]:
        return terminal("ABSTAINED", "LOCAL_ACTION_CREDIT_SEMANTICS_INVALID")

    semantics = slow_path_result.get("evidence_semantics")
    checks["full_scoped_retrain_is_policy_evidence"] = bool(
        isinstance(semantics, Mapping)
        and semantics.get("local_singleton") == "proposal_credit_only"
        and semantics.get("full_scoped_retrain") == "policy_evidence"
    )
    if not checks["full_scoped_retrain_is_policy_evidence"]:
        return terminal("ABSTAINED", "FULL_POLICY_EVIDENCE_SEMANTICS_INVALID")
    if slow_path_result.get("risk_patch_replay_passed") is False:
        return terminal("REJECTED", "RISK_PATCH_REPLAY_NOT_PASSED")

    positive_environments: list[str] = []
    policy_episodes: list[dict[str, object]] = []
    replays = slow_path_result.get("policy_replays")
    if isinstance(replays, Sequence) and not isinstance(replays, (str, bytes)):
        for replay in replays:
            if not isinstance(replay, Mapping):
                continue
            scoped = replay.get("scoped_program")
            selection = scoped.get("selection") if isinstance(scoped, Mapping) else None
            support = scoped.get("support") if isinstance(scoped, Mapping) else None
            eligible = replay.get("eligible_count")
            selection_gain = (
                selection.get("gain_vs_identity")
                if isinstance(selection, Mapping)
                else None
            )
            selection_behavior = (
                selection.get("behavior_point_count")
                if isinstance(selection, Mapping)
                else None
            )
            positive_in_scope = bool(
                isinstance(eligible, int)
                and eligible > 0
                and isinstance(selection_gain, (int, float))
                and not isinstance(selection_gain, bool)
                and math.isfinite(float(selection_gain))
                and float(selection_gain) > 0.0
                and isinstance(selection_behavior, int)
                and selection_behavior > 0
            )
            if isinstance(replay.get("environment"), str):
                policy_episodes.append(
                    {
                        "environment": replay["environment"],
                        "relation": (
                            "POSITIVE_IN_SCOPE"
                            if positive_in_scope
                            else "OUT_OF_SCOPE_ABSTENTION"
                            if eligible == 0
                            else "NON_POSITIVE_IN_SCOPE"
                        ),
                        "eligible_count": eligible,
                        "selection_gain": selection_gain,
                        "behavior_point_count": selection_behavior,
                    }
                )
            if (
                isinstance(selection, Mapping)
                and isinstance(support, Mapping)
                and isinstance(replay.get("eligible_count"), int)
                and int(replay["eligible_count"]) > 0
                and isinstance(replay.get("training_series_count"), int)
                and int(replay["training_series_count"]) > 0
                and isinstance(selection.get("gain_vs_identity"), (int, float))
                and not isinstance(selection.get("gain_vs_identity"), bool)
                and math.isfinite(float(selection["gain_vs_identity"]))
                and float(selection["gain_vs_identity"]) > 0.0
                and isinstance(selection.get("behavior_point_count"), int)
                and int(selection["behavior_point_count"]) > 0
                and isinstance(replay.get("environment"), str)
            ):
                positive_environments.append(str(replay["environment"]))
    positive_environments = sorted(set(positive_environments))
    checks["positive_in_scope_policy_environment_count"] = len(positive_environments)

    if not isinstance(confirmation_result, Mapping):
        return terminal("ABSTAINED", "CONFIRMATION_REQUIRED")
    embedded_confirmation = slow_path_result.get("confirmation")
    if (
        isinstance(embedded_confirmation, Mapping)
        and dict(embedded_confirmation) != dict(confirmation_result)
    ):
        return terminal("ABSTAINED", "CONFIRMATION_TRACE_MISMATCH")
    selection = confirmation_result.get("selection")
    gain = selection.get("gain_vs_identity") if isinstance(selection, Mapping) else None
    behavior = (
        selection.get("behavior_point_count") if isinstance(selection, Mapping) else None
    )
    explicit_harm = confirmation_result.get("harm") is True
    harm_count = confirmation_result.get("harm_count", 0)
    numeric_harm = (
        not isinstance(gain, (int, float))
        or isinstance(gain, bool)
        or not math.isfinite(float(gain))
        or float(gain) <= 0.0
    )
    counted_harm = (
        isinstance(harm_count, (int, float))
        and not isinstance(harm_count, bool)
        and float(harm_count) > 0.0
    )
    confirmation_harm = bool(explicit_harm or counted_harm or numeric_harm)
    checks["confirmation_harm"] = confirmation_harm
    checks["confirmation_passed"] = bool(
        confirmation_result.get("passed") is True
        and not confirmation_harm
        and isinstance(behavior, int)
        and behavior > 0
    )
    contextual_episode = {
        "capability_version": copy.deepcopy(resolution_scope),
        "source_full_policy_episodes": policy_episodes,
        "confirmation_episode": {
            "environment": confirmation_result.get("environment"),
            "relation": (
                "POSITIVE"
                if checks["confirmation_passed"]
                else "CONFLICT"
                if positive_environments
                else "NEGATIVE"
            ),
            "passed": confirmation_result.get("passed"),
            "selection_gain": gain,
            "behavior_point_count": behavior,
            "harm": confirmation_harm,
        },
    }
    if not checks["confirmation_passed"]:
        return terminal("REJECTED_AFTER_CONFIRMATION", "CONFIRMATION_FAILED_OR_HARMFUL")
    if len(positive_environments) < 2:
        return terminal(
            "REJECTED_AFTER_CONFIRMATION",
            "INSUFFICIENT_FULL_POLICY_ENVIRONMENTS",
        )

    skill_card = {
        "capability_id": f"generated-scope-{target_op}",
        "status": "ACTIVE",
        "provenance": "LLM_GENERATED_TRACE_AND_CONFIRMED_TYPED_PATCH",
        "program": {
            "operator": target_op,
            "source_traces": generated_traces,
        },
        "applicability": {
            "all": copy.deepcopy(list(slow_path_result["compiled_conditions"])),
        },
        "evidence": {
            "positive_full_policy_environments": positive_environments,
            "confirmation": copy.deepcopy(dict(confirmation_result)),
            "contextual_episode": copy.deepcopy(contextual_episode),
        },
    }
    write_count = 0
    if memory_writer is not None:
        memory_writer(copy.deepcopy(skill_card))
        write_count = 1
    return {
        "status": "PROMOTED",
        "reason_code": "CONFIRMED_GENERATED_CAPABILITY",
        "rejected_capability_version": None,
        "contextual_episode": copy.deepcopy(contextual_episode),
        "operator_blacklisted": False,
        "program_family_closed": False,
        "memory_write_authorized": True,
        "memory_write_count": write_count,
        "generated_skill_card": skill_card,
        "checks": copy.deepcopy(checks),
    }


def _support_trace(
    stage: str,
    compiled: CompiledWorkflow,
    support_callback: SupportCallback,
    public_context: Mapping[str, object],
) -> dict[str, object]:
    response = _plain_json(support_callback(compiled), name="support response")
    if not isinstance(response, dict) or not isinstance(response.get("accepted"), bool):
        raise ValueError("support response requires a boolean accepted field")
    _reject_sealed_fields(response, path="support_response")
    assert compiled.candidate.program is not None
    return {
        "stage": stage,
        "candidate_id": compiled.candidate.candidate_id,
        "public_context": copy.deepcopy(dict(public_context)),
        "action": {
            "workflow_steps": copy.deepcopy(list(compiled.template_steps)),
            "program_steps": [
                {"op": op, "params": copy.deepcopy(params)}
                for op, params in compiled.candidate.program.execution_steps()
            ],
            "requested_observations": list(compiled.requested_observations),
            "fallback": compiled.fallback,
            "experience_use": list(compiled.experience_use),
        },
        "support_response": response,
    }


def _compilation_trace(
    stage: str,
    proposal: Mapping[str, object],
    error: CandidateCompilationError,
    public_context: Mapping[str, object],
) -> dict[str, object]:
    """Turn one invalid candidate into feedback for the next generation."""

    return {
        "stage": stage,
        "candidate_id": None,
        "public_context": copy.deepcopy(dict(public_context)),
        "action": {"proposed_workflow": _plain_json(proposal, name="workflow proposal")},
        "support_response": {
            "accepted": False,
            "feedback_type": "COMPILATION_ERROR",
            "error_code": error.code,
            "message": str(error),
        },
    }


def _proposer_abstain_trace(
    stage: str,
    proposal: Mapping[str, object],
    public_context: Mapping[str, object],
) -> dict[str, object]:
    raw = _plain_json(proposal, name="proposer abstention")
    _reject_sealed_fields(raw, path="proposal")
    return {
        "stage": stage,
        "candidate_id": None,
        "public_context": copy.deepcopy(dict(public_context)),
        "action": {"proposed_workflow": raw},
        "support_response": {
            "accepted": False,
            "feedback_type": "PROPOSER_ABSTAINED",
            "error_code": None,
            "message": "proposer returned ABSTAIN",
        },
    }


def _program_ast(compiled: CompiledWorkflow) -> object:
    assert compiled.candidate.program is not None
    return _plain_json(
        {
            "program_steps": compiled.candidate.program.execution_steps(),
            "template_steps": list(compiled.template_steps),
        },
        name="compiled Program AST",
    )


def _compile_generated_candidate(
    proposal: Mapping[str, object],
    inventory: Sequence[Mapping[str, object]],
    public_context: Mapping[str, object],
    *,
    generation: int,
) -> CompiledWorkflow:
    """Classify one model-produced schema error as candidate feedback."""

    try:
        return compile_workflow_proposal(
            proposal, inventory, public_context, generation=generation
        )
    except CandidateCompilationError:
        raise
    except ValueError as exc:
        raise CandidateCompilationError("PROPOSAL_INVALID", str(exc)) from exc


def _skill_from_trace(
    accepted_trace: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    action = accepted_trace["action"]
    assert isinstance(action, Mapping)
    candidate_id = str(accepted_trace["candidate_id"])
    return {
        "capability_id": f"generated-capability-{candidate_id.rsplit('-', 1)[-1]}",
        "status": "CANDIDATE",
        "task_context": copy.deepcopy(accepted_trace["public_context"]),
        "applicability_seed": {
            "source_public_context": copy.deepcopy(accepted_trace["public_context"]),
            "generalization_status": "UNCONFIRMED",
        },
        "program": {
            "steps": copy.deepcopy(action["program_steps"]),
            "source_candidate_id": candidate_id,
        },
        "program_template": {
            "steps": copy.deepcopy(action["workflow_steps"]),
        },
        "requested_observations": copy.deepcopy(action["requested_observations"]),
        "control": {"fallback": action["fallback"]},
        "history": copy.deepcopy(list(history)),
    }


def run_two_round_generation(
    public_context: Mapping[str, object],
    task_kind: str,
    initial_proposer: Proposer,
    revision_proposer: Proposer,
    support_callback: SupportCallback,
    *,
    capability_memory: Sequence[Mapping[str, object]] = (),
    forbidden_operators: Sequence[str] = (),
) -> dict[str, object]:
    """Run initial generation and one feedback-conditioned revision.

    The function is deliberately Support-only: it rejects sealed Query/outcome
    fields, creates no promoted Skill, and has no caller-supplied Workflow or
    Skill catalog.
    """

    if capability_memory:
        raise ValueError("this acquisition slice requires empty Capability Memory")
    context = _plain_json(public_context, name="public_context")
    if not isinstance(context, dict):
        raise ValueError("public_context must be an object")
    _reject_sealed_fields(context, path="public_context")
    inventory = build_public_operator_inventory(
        task_kind, context, forbidden_operators=forbidden_operators
    )
    common = {
        "public_context": context,
        "operator_inventory": copy.deepcopy(list(inventory)),
        "capability_memory": [],
        "workflow_schema": _proposal_schema(),
        "information_wall": {
            "support_only": True,
            "query_outcome_forbidden": True,
            "promotion_external": True,
        },
        "exploration_policy": {
            "maximum_generation_budget": 2,
            "revision_is_complete_replacement": True,
            "revision_must_change_program_ast_when_exploration_required": True,
            "exploration_required_after": [
                "COMPILATION_ERROR",
                "ZERO_BEHAVIOR",
                "NON_POSITIVE_SUPPORT_WITH_UNTRIED_EXECUTABLE_OPERATOR",
            ],
            "abstain_allowed_only_if": [
                "EXPLAINED_LEGAL_RISK",
                "NO_EXECUTABLE_ALTERNATIVE",
            ],
            "harness_selects_operator": False,
        },
    }
    traces: list[dict[str, object]] = []
    compiled_candidates: list[CompiledWorkflow] = []

    initial_payload = {"stage": "INITIAL", **copy.deepcopy(common)}
    initial = initial_proposer(initial_payload)
    if not isinstance(initial, Mapping):
        raise ValueError("initial proposer must return an object")
    if initial.get("decision") == "ABSTAIN":
        traces.append(_proposer_abstain_trace("INITIAL", initial, context))
        return _terminal_result(
            "ABSTAIN", "INITIAL_PROPOSER_ABSTAINED", inventory, traces, ()
        )
    initial_compiled: CompiledWorkflow | None = None
    exploration_required = False
    try:
        compiled = _compile_generated_candidate(
            initial, inventory, context, generation=1
        )
    except CandidateCompilationError as exc:
        traces.append(_compilation_trace("INITIAL", initial, exc, context))
        exploration_required = any(
            row.get("availability") == EXECUTABLE for row in inventory
        )
    else:
        initial_compiled = compiled
        compiled_candidates.append(compiled)
        traces.append(_support_trace("INITIAL", compiled, support_callback, context))
        response = traces[-1]["support_response"]
        assert isinstance(response, Mapping)
        tried = {step.op for step in compiled.candidate.program.steps}  # type: ignore[union-attr]
        untried = {
            str(row["name"])
            for row in inventory
            if row.get("availability") == EXECUTABLE and row.get("name") not in tried
        }
        exploration_required = bool(
            untried
            and (
                response.get("accepted") is not True
                or response.get("behavior") == "ZERO_BEHAVIOR"
            )
        )

    if traces[-1]["support_response"]["accepted"] is True:  # type: ignore[index]
        assert initial_compiled is not None
        return {
            "status": "CANDIDATE",
            "reason_code": "SUPPORT_ACCEPTED",
            "operator_inventory": copy.deepcopy(list(inventory)),
            "action_response_trace": copy.deepcopy(traces),
            "final_candidate": initial_compiled.candidate,
            "skill_draft": _skill_from_trace(traces[-1], traces),
        }

    revision_payload = {
        "stage": "REVISION",
        **copy.deepcopy(common),
        "initial_trace": copy.deepcopy(traces[0]),
        "exploration_required": exploration_required,
        "instruction": "Revise the typed Workflow using only the observed Support response.",
    }
    revision = revision_proposer(revision_payload)
    if not isinstance(revision, Mapping):
        raise ValueError("revision proposer must return an object")
    if revision.get("decision") == "ABSTAIN":
        traces.append(_proposer_abstain_trace("REVISION", revision, context))
        if not any(
            trace["support_response"]["accepted"] is True  # type: ignore[index]
            for trace in traces[:-1]
        ):
            return _terminal_result(
                "ABSTAIN",
                "REVISION_PROPOSER_ABSTAINED",
                inventory,
                traces,
                compiled_candidates,
            )
    else:
        try:
            revised = _compile_generated_candidate(
                revision, inventory, context, generation=2
            )
            if (
                exploration_required
                and initial_compiled is not None
                and _program_ast(revised) == _program_ast(initial_compiled)
            ):
                raise CandidateCompilationError(
                    "REVISION_PROGRAM_UNCHANGED",
                    "revision must replace the failed initial Program with a different AST",
                )
        except CandidateCompilationError as exc:
            traces.append(_compilation_trace("REVISION", revision, exc, context))
        else:
            compiled_candidates.append(revised)
            traces.append(_support_trace("REVISION", revised, support_callback, context))

    accepted = [trace for trace in traces if trace["support_response"]["accepted"] is True]  # type: ignore[index]
    if not accepted:
        return _terminal_result(
            "REJECTED", "NO_POSITIVE_SUPPORT", inventory, traces, compiled_candidates
        )
    selected_trace = accepted[-1]
    selected_id = selected_trace["candidate_id"]
    selected = next(
        item for item in compiled_candidates if item.candidate.candidate_id == selected_id
    )
    return {
        "status": "CANDIDATE",
        "reason_code": "SUPPORT_ACCEPTED",
        "operator_inventory": copy.deepcopy(list(inventory)),
        "action_response_trace": copy.deepcopy(traces),
        "final_candidate": selected.candidate,
        "skill_draft": _skill_from_trace(selected_trace, traces),
    }


def _terminal_result(
    status: str,
    reason_code: str,
    inventory: Sequence[Mapping[str, object]],
    traces: Sequence[Mapping[str, object]],
    compiled_candidates: Sequence[CompiledWorkflow],
) -> dict[str, object]:
    return {
        "status": status,
        "reason_code": reason_code,
        "operator_inventory": copy.deepcopy(list(inventory)),
        "action_response_trace": copy.deepcopy(list(traces)),
        "final_candidate": None,
        "skill_draft": None,
        "compiled_candidate_ids": [item.candidate.candidate_id for item in compiled_candidates],
    }


__all__ = [
    "CompiledWorkflow",
    "EXECUTABLE",
    "IDENTITY",
    "UNAVAILABLE",
    "build_public_operator_inventory",
    "compile_workflow_proposal",
    "run_two_round_generation",
]
