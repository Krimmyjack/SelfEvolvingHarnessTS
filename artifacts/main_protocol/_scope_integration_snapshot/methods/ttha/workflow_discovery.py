"""Bounded Workflow discovery from public TS Context and caller-owned catalogs.

The planner may select and bind existing typed Workflow templates.  It cannot
invent code, inspect outcomes, promote a Skill, or mutate the catalogs.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping, Sequence


IDENTITY = "IDENTITY"
_FORBIDDEN_FIELD = re.compile(
    r"(?:^|_)(?:dataset_?id|loss|oracle|outcome|query_?future|utility)(?:$|_)",
    re.IGNORECASE,
)
_FORBIDDEN_REFERENCE = re.compile(
    r"\b(?:dataset[_ -]?id|loss|oracle|outcome|query[_ -]?future|utility)\b",
    re.IGNORECASE,
)

Planner = Callable[[Mapping[str, object]], Mapping[str, object]]


def _plain_copy(value: object, *, name: str) -> object:
    """Copy a JSON-shaped value and reject runtime objects/callables."""

    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc


def _reject_private_fields(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            normalized = re.sub(r"[^a-zA-Z0-9]+", "_", key_text).strip("_")
            if _FORBIDDEN_FIELD.search(normalized):
                raise ValueError(f"private/outcome field is forbidden at {path}.{key}")
            _reject_private_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_private_fields(nested, path=f"{path}[{index}]")
    elif isinstance(value, str) and _FORBIDDEN_REFERENCE.search(value):
        raise ValueError(f"private/outcome reference is forbidden at {path}")


def _normalize_workflow_catalog(
    workflow_catalog: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    if isinstance(workflow_catalog, (str, bytes)) or not workflow_catalog:
        raise ValueError("workflow_catalog must contain typed Workflow templates")
    public_rows: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    for raw in workflow_catalog:
        if not isinstance(raw, Mapping):
            raise ValueError("Workflow catalog entries must be objects")
        row = _plain_copy(raw, name="workflow_catalog entry")
        assert isinstance(row, dict)
        _reject_private_fields(row, path="workflow_catalog")
        workflow_id = row.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            raise ValueError("Workflow catalog entry requires workflow_id")
        if workflow_id == IDENTITY or workflow_id in by_id:
            raise ValueError("Workflow catalog ids must be unique and exclude IDENTITY")
        bindings = row.get("public_parameter_bindings", {})
        if not isinstance(bindings, dict) or not all(
            isinstance(key, str) and key and isinstance(path, str) and path
            for key, path in bindings.items()
        ):
            raise ValueError(
                "public_parameter_bindings must map parameter ids to Context paths"
            )
        # The planner sees only the declarative, JSON-shaped template.  No
        # executor or implementation object is accepted by _plain_copy.
        public_rows.append(row)
        by_id[workflow_id] = row
    return public_rows, by_id


def _normalize_observation_catalog(
    observation_catalog: Sequence[str | Mapping[str, object]],
) -> tuple[list[object], set[str]]:
    if isinstance(observation_catalog, (str, bytes)):
        raise ValueError("observation_catalog must be a sequence")
    public_rows: list[object] = []
    observation_ids: set[str] = set()
    for raw in observation_catalog:
        row = _plain_copy(raw, name="observation_catalog entry")
        _reject_private_fields(row, path="observation_catalog")
        if isinstance(row, str):
            observation_id = row
        elif isinstance(row, dict):
            observation_id = row.get("observation_id")
        else:
            raise ValueError("Observation catalog entries must be ids or objects")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("Observation catalog entry requires observation_id")
        if observation_id in observation_ids:
            raise ValueError("Observation catalog ids must be unique")
        observation_ids.add(observation_id)
        public_rows.append(row)
    return public_rows, observation_ids


def _context_value(public_context: Mapping[str, object], path: str) -> object:
    current: object = public_context
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise ValueError(f"public Context does not provide binding path {path}")
        current = current[segment]
    return current


def _abstain(reason_code: str) -> dict[str, object]:
    return {
        "decision": "ABSTAIN",
        "workflow_supply": [],
        "compiled_workflows": [],
        "probe_order": [],
        "requested_observations": [],
        "fallback": IDENTITY,
        "reason_code": reason_code,
        "candidate_status": "NOT_CREATED",
    }


def compile_workflow_proposal(
    proposal: Mapping[str, object],
    public_context: Mapping[str, object],
    workflow_catalog: Sequence[Mapping[str, object]],
    observation_catalog: Sequence[str | Mapping[str, object]],
    *,
    max_candidates: int = 3,
) -> dict[str, object]:
    """Validate and compile one planner proposal into an executable supply."""

    if max_candidates < 2:
        raise ValueError("max_candidates must allow at least two Workflows")
    context = _plain_copy(public_context, name="public_context")
    if not isinstance(context, dict):
        raise ValueError("public_context must be an object")
    _reject_private_fields(context, path="public_context")
    _, workflows_by_id = _normalize_workflow_catalog(workflow_catalog)
    _, observation_ids = _normalize_observation_catalog(observation_catalog)
    compiled_proposal = _plain_copy(proposal, name="planner proposal")
    if not isinstance(compiled_proposal, dict):
        raise ValueError("planner proposal must be an object")
    _reject_private_fields(compiled_proposal, path="proposal")

    decision = compiled_proposal.get("decision")
    if decision == "ABSTAIN":
        return _abstain("PLANNER_ABSTAIN")
    if decision != "PROPOSE":
        raise ValueError("planner decision must be PROPOSE or ABSTAIN")
    if compiled_proposal.get("fallback") != IDENTITY:
        raise ValueError("Workflow proposal must preserve IDENTITY fallback")

    selected = compiled_proposal.get("selected_workflows")
    if not isinstance(selected, list) or not 2 <= len(selected) <= max_candidates:
        raise ValueError("proposal must select two to max_candidates Workflows")
    compiled_workflows: list[dict[str, object]] = []
    selected_ids: list[str] = []
    for selected_row in selected:
        if not isinstance(selected_row, dict):
            raise ValueError("selected Workflow must be an object")
        workflow_id = selected_row.get("workflow_id")
        if not isinstance(workflow_id, str) or workflow_id not in workflows_by_id:
            raise ValueError("proposal selected a Workflow outside the catalog")
        if workflow_id in selected_ids:
            raise ValueError("proposal selected a Workflow more than once")
        declared = workflows_by_id[workflow_id].get(
            "public_parameter_bindings", {}
        )
        assert isinstance(declared, dict)
        proposed_bindings = selected_row.get("bindings", {})
        if not isinstance(proposed_bindings, dict) or set(proposed_bindings) != set(
            declared
        ):
            raise ValueError("Workflow bindings do not match the catalog declaration")
        expected_bindings = {
            parameter: copy.deepcopy(_context_value(context, path))
            for parameter, path in declared.items()
        }
        if proposed_bindings != expected_bindings:
            raise ValueError("Workflow binding value does not match public Context")
        selected_ids.append(workflow_id)
        compiled_workflows.append(
            {
                "workflow_id": workflow_id,
                "bindings": copy.deepcopy(expected_bindings),
            }
        )

    probe_order = compiled_proposal.get("probe_order")
    if (
        not isinstance(probe_order, list)
        or len(probe_order) != len(selected_ids)
        or set(probe_order) != set(selected_ids)
        or len(set(probe_order)) != len(probe_order)
    ):
        raise ValueError("probe_order must contain every selected Workflow once")
    requested = compiled_proposal.get("requested_observations", [])
    if (
        not isinstance(requested, list)
        or len(set(requested)) != len(requested)
        or not all(
            isinstance(value, str) and value in observation_ids for value in requested
        )
    ):
        raise ValueError("requested observations must come from observation_catalog")

    return {
        "decision": "PROPOSE",
        "workflow_supply": list(selected_ids),
        "compiled_workflows": compiled_workflows,
        "probe_order": list(probe_order),
        "requested_observations": list(requested),
        "fallback": IDENTITY,
        "reason_code": "VALID_CATALOG_PROPOSAL",
        "candidate_status": "DISCOVERED_NOT_EVALUATED",
    }


def discover_workflow_supply(
    public_context: Mapping[str, object],
    workflow_catalog: Sequence[Mapping[str, object]],
    observation_catalog: Sequence[str | Mapping[str, object]],
    planner: Planner,
    *,
    max_candidates: int = 3,
) -> dict[str, object]:
    """Ask a planner once, then compile or safely abstain.

    The planner receives a detached public payload.  Invalid inputs, invented
    Workflows, binding mismatches, and planner failures all fail closed without
    creating or promoting a Skill.
    """

    try:
        context = _plain_copy(public_context, name="public_context")
        if not isinstance(context, dict):
            raise ValueError("public_context must be an object")
        _reject_private_fields(context, path="public_context")
        public_workflows, _ = _normalize_workflow_catalog(workflow_catalog)
        public_observations, _ = _normalize_observation_catalog(observation_catalog)
        if max_candidates < 2:
            raise ValueError("max_candidates must allow at least two Workflows")
        planner_input = {
            "public_context": context,
            "workflow_catalog": public_workflows,
            "observation_catalog": public_observations,
            "constraints": {
                "max_candidates": max_candidates,
                "fallback": IDENTITY,
                "catalog_only": True,
                "outcome_fields_forbidden": True,
            },
        }
        proposal = planner(copy.deepcopy(planner_input))
        if not isinstance(proposal, Mapping):
            raise ValueError("planner must return an object")
        return compile_workflow_proposal(
            proposal,
            context,
            public_workflows,
            public_observations,
            max_candidates=max_candidates,
        )
    except Exception as exc:  # fail closed at the LLM/planner boundary
        return _abstain(type(exc).__name__.upper())


__all__ = ["compile_workflow_proposal", "discover_workflow_supply"]
