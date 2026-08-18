"""Lean PolicyEpisode-driven Skill acquisition for the TS Harness.

This module deliberately owns only reusable behavior already exercised by the
natural Forecasting vertical slice.  It uses plain dictionaries and callbacks;
promotion, Consumer evaluation, persistence, and LLM calls remain outside.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from itertools import permutations
from pathlib import Path
from statistics import fmean
from typing import Any


IDENTITY = "IDENTITY"
CANDIDATE = "CANDIDATE"
_EXECUTABLE_STATUSES = frozenset(
    {"CROSS_DATASET_SUPPORTED", "SOURCE_PROMOTED", "PROMOTED"}
)
_PATCH_OPERATIONS = frozenset(
    {"ADD_OBSERVATION", "PATCH_CONTROL", "COMPOSE_WORKFLOW", "RESTRICT_SCOPE"}
)
_FORBIDDEN_PATCH_KEYS = frozenset(
    {
        "consumer",
        "metric",
        "memory",
        "memory_schema",
        "program",
        "program_schema",
        "program_supply",
        "task_context",
        "workflow_supply",
    }
)


ProbeCallback = Callable[[str], Mapping[str, object]]
SupportProbeCallback = Callable[[str], Mapping[str, object]]
PromotionCallback = Callable[[Mapping[str, object]], Mapping[str, object]]
CyclePromotionCallback = Callable[
    [Mapping[str, object], Sequence[Mapping[str, object]]], Mapping[str, object]
]
SourceWorkflowEvaluationCallback = Callable[
    [Mapping[str, object], str, Mapping[str, object]], Mapping[str, object]
]
FailurePatchProposalCallback = Callable[
    [Mapping[str, object]], Mapping[str, object]
]
FailurePatchReplayCallback = Callable[
    [Mapping[str, object]], Mapping[str, object]
]


def validate_workflow_supply(workflow_supply: Sequence[str]) -> tuple[str, ...]:
    """Return one bounded, duplicate-free Workflow supply."""

    if isinstance(workflow_supply, (str, bytes)):
        raise ValueError("workflow_supply must be a sequence of Workflow ids")
    workflow_ids = tuple(str(value) for value in workflow_supply)
    if len(workflow_ids) < 2:
        raise ValueError("at least two Workflows are required")
    if any(not value or value == IDENTITY for value in workflow_ids):
        raise ValueError("Workflow ids must be non-empty and exclude IDENTITY")
    if len(set(workflow_ids)) != len(workflow_ids):
        raise ValueError("workflow_supply contains duplicates")
    return workflow_ids


def collect_source_policy_episodes(
    source_contexts: Sequence[Mapping[str, object]],
    compiled_workflows: Sequence[Mapping[str, object]],
    evaluate_workflow: SourceWorkflowEvaluationCallback,
) -> list[dict[str, object]]:
    """Execute a bounded discovered supply into completed Source episodes.

    The Harness owns iteration and the PolicyEpisode shape.  Dataset loading,
    Program execution, and Consumer evaluation remain a caller adapter.  This
    function is for completed Source/development contexts; Target Query remains
    outside the callback used for current-Support confirmation.
    """

    workflow_rows = [copy.deepcopy(dict(row)) for row in compiled_workflows]
    workflow_ids = validate_workflow_supply(
        tuple(str(row.get("workflow_id", "")) for row in workflow_rows)
    )
    for row in workflow_rows:
        bindings = row.get("bindings", {})
        if not isinstance(bindings, Mapping):
            raise ValueError("compiled Workflow bindings must be an object")
    if not source_contexts:
        raise ValueError("at least one completed Source context is required")

    episodes: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for raw_context in source_contexts:
        case_id = raw_context.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise ValueError("Source contexts require unique non-empty case_id")
        seen_case_ids.add(case_id)
        responses: dict[str, dict[str, object]] = {}
        for workflow_id, compiled in zip(workflow_ids, workflow_rows):
            bindings = dict(compiled.get("bindings", {}))
            response = evaluate_workflow(
                copy.deepcopy(dict(raw_context)),
                workflow_id,
                copy.deepcopy(bindings),
            )
            if not isinstance(response, Mapping):
                raise ValueError("Source Workflow evaluator must return an object")
            if "support_gain" not in response or "query_gain" not in response:
                raise ValueError(
                    "completed Source response requires support_gain and query_gain"
                )
            support_gain = float(response["support_gain"])
            query_gain = float(response["query_gain"])
            if not math.isfinite(support_gain) or not math.isfinite(query_gain):
                raise ValueError("Source Workflow gains must be finite")
            normalized = copy.deepcopy(dict(response))
            normalized.update(
                {
                    "workflow_id": workflow_id,
                    "support_gain": support_gain,
                    "query_gain": query_gain,
                    "bindings": bindings,
                }
            )
            responses[workflow_id] = normalized
        episodes.append({"source_case_id": case_id, "workflows": responses})
    return episodes


def _episode_workflows(
    episode: Mapping[str, object], workflow_ids: tuple[str, ...] | None = None
) -> dict[str, Mapping[str, object]]:
    workflows = episode.get("workflows")
    if not isinstance(workflows, Mapping):
        raise ValueError("PolicyEpisode requires a workflows object")
    normalized: dict[str, Mapping[str, object]] = {}
    for key, value in workflows.items():
        workflow_id = str(key)
        if not isinstance(value, Mapping):
            raise ValueError(f"Workflow {workflow_id} response must be an object")
        if value.get("workflow_id", workflow_id) != workflow_id:
            raise ValueError(f"Workflow {workflow_id} response id does not match")
        if "support_gain" not in value or "query_gain" not in value:
            raise ValueError(
                f"Workflow {workflow_id} requires support_gain and query_gain"
            )
        float(value["support_gain"])
        float(value["query_gain"])
        normalized[workflow_id] = value
    inferred = validate_workflow_supply(tuple(normalized))
    if workflow_ids is not None:
        missing = set(workflow_ids) - set(inferred)
        if missing:
            raise ValueError(
                "PolicyEpisode is missing supplied Workflows: "
                + ", ".join(sorted(missing))
            )
        # Completed Source episodes may contain a richer menu than the bounded
        # supply discovered for the current task.  Compile evidence only for the
        # discovered subset instead of requiring the whole historical menu.
        normalized = {
            workflow_id: normalized[workflow_id] for workflow_id in workflow_ids
        }
    return normalized


def compile_probe_order_from_historical_episode(
    historical_episode: Mapping[str, object],
    workflow_supply: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Order probes using only a completed, legally visible PolicyEpisode.

    Historical response is an observation, not a current utility certificate;
    execution still requires current Support confirmation.
    """

    expected = (
        validate_workflow_supply(workflow_supply)
        if workflow_supply is not None
        else None
    )
    workflows = historical_episode.get("workflows")
    if not isinstance(workflows, Mapping) or len(workflows) < 2:
        raise ValueError("historical PolicyEpisode requires at least two Workflows")
    normalized: dict[str, Mapping[str, object]] = {}
    for key, value in workflows.items():
        workflow_id = str(key)
        if not isinstance(value, Mapping) or "support_gain" not in value:
            raise ValueError("historical Workflow response requires support_gain")
        if value.get("workflow_id", workflow_id) != workflow_id:
            raise ValueError("historical Workflow response id does not match")
        float(value["support_gain"])
        normalized[workflow_id] = value
    if expected is not None and set(normalized) != set(expected):
        raise ValueError("historical PolicyEpisode does not match Workflow supply")
    return tuple(
        sorted(
            normalized,
            key=lambda workflow_id: (
                -float(normalized[workflow_id]["support_gain"]),
                workflow_id,
            ),
        )
    )


def plan_support_only(
    workflow_supply: Sequence[str],
    probe_order: Sequence[str],
    probe: SupportProbeCallback,
    *,
    control: str,
) -> dict[str, object]:
    """Plan from current Support without exposing delayed downstream outcome.

    The callback contract is deliberately narrow: it must return exactly one
    finite ``support_gain``.  Query/future fields are rejected rather than
    ignored, so the returned planning trace cannot accidentally become an
    evaluator result.  Delayed outcomes are attached later by
    :func:`attach_delayed_outcomes`.
    """

    workflow_ids = validate_workflow_supply(workflow_supply)
    order = tuple(str(value) for value in probe_order)
    if len(order) != len(workflow_ids) or set(order) != set(workflow_ids):
        raise ValueError("probe_order must contain every supplied Workflow once")
    if control not in {"stop_on_first_positive", "keep_best_support_so_far"}:
        raise ValueError("unsupported Support planning control")

    planning_trace: list[dict[str, object]] = [
        {
            "budget": 0,
            "selected_workflow": IDENTITY,
            "abstained": True,
            "terminal": False,
        }
    ]
    observations: list[dict[str, object]] = []
    selected = IDENTITY
    selected_support_gain = 0.0
    terminal = False

    for budget, workflow_id in enumerate(order, 1):
        if not terminal:
            response = probe(workflow_id)
            if not isinstance(response, Mapping):
                raise ValueError("Support probe callback must return an object")
            if set(response) != {"support_gain"}:
                raise ValueError(
                    "Support probe response must contain only support_gain"
                )
            support_gain = float(response["support_gain"])
            if not math.isfinite(support_gain):
                raise ValueError("Support gain must be finite")
            observations.append(
                {
                    "workflow_id": workflow_id,
                    "support_gain": support_gain,
                }
            )
            if control == "stop_on_first_positive" and support_gain > 0.0:
                selected = workflow_id
                terminal = True
            elif (
                control == "keep_best_support_so_far"
                and support_gain > selected_support_gain
            ):
                selected = workflow_id
                selected_support_gain = support_gain
        planning_trace.append(
            {
                "budget": budget,
                "selected_workflow": selected,
                "abstained": selected == IDENTITY,
                "terminal": terminal,
            }
        )

    return {
        "selected_workflow": selected,
        "abstained": selected == IDENTITY,
        "probed_workflows": [row["workflow_id"] for row in observations],
        "support_observations": observations,
        "support_planning_trace": planning_trace,
        "control": control,
    }


def attach_delayed_outcomes(
    support_plan: Mapping[str, object],
    delayed_query_gains: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate a frozen Support plan after downstream outcomes become visible.

    Missing or non-finite outcomes raise before an AUC is produced.  This keeps
    an unavailable delayed label from being silently interpreted as zero gain.
    """

    if not isinstance(support_plan, Mapping):
        raise ValueError("support_plan must be an object")
    trace = support_plan.get("support_planning_trace")
    observations = support_plan.get("support_observations")
    if not isinstance(trace, Sequence) or isinstance(trace, (str, bytes)):
        raise ValueError("support_plan requires support_planning_trace")
    if not isinstance(observations, Sequence) or isinstance(
        observations, (str, bytes)
    ):
        raise ValueError("support_plan requires support_observations")
    if not isinstance(delayed_query_gains, Mapping):
        raise ValueError("delayed_query_gains must be an object")

    normalized_gains: dict[str, float] = {}
    for raw in observations:
        if not isinstance(raw, Mapping):
            raise ValueError("Support observations must be objects")
        workflow_id = raw.get("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id:
            raise ValueError("Support observation requires workflow_id")
        if workflow_id not in delayed_query_gains:
            raise ValueError(
                f"delayed outcome is unavailable for probed Workflow {workflow_id}"
            )
        gain = float(delayed_query_gains[workflow_id])
        if not math.isfinite(gain):
            raise ValueError(
                f"delayed outcome must be finite for Workflow {workflow_id}"
            )
        normalized_gains[workflow_id] = gain

    curve: list[dict[str, object]] = []
    for expected_budget, raw in enumerate(trace):
        if not isinstance(raw, Mapping) or int(raw.get("budget", -1)) != expected_budget:
            raise ValueError("Support planning budgets must be contiguous from zero")
        workflow_id = raw.get("selected_workflow")
        if not isinstance(workflow_id, str):
            raise ValueError("Support planning trace requires selected_workflow")
        if workflow_id == IDENTITY:
            gain = 0.0
        elif workflow_id in normalized_gains:
            gain = normalized_gains[workflow_id]
        else:
            raise ValueError(
                f"delayed outcome is unavailable for selected Workflow {workflow_id}"
            )
        evaluated = copy.deepcopy(dict(raw))
        evaluated["fixed_query_gain"] = gain
        curve.append(evaluated)

    result = copy.deepcopy(dict(support_plan))
    result.pop("support_planning_trace", None)
    result.pop("control", None)
    result["support_observations"] = [
        {
            **copy.deepcopy(dict(raw)),
            "fixed_query_gain": normalized_gains[str(raw["workflow_id"])],
        }
        for raw in observations
    ]
    result["adaptation_curve"] = curve
    result["adaptation_auc"] = policy_adaptation_auc(curve)
    return result


def _execute_with_legacy_probe(
    workflow_supply: Sequence[str],
    probe_order: Sequence[str],
    probe: ProbeCallback,
    *,
    control: str,
) -> dict[str, object]:
    """Compatibility adapter for callers that still return Support + Query."""

    delayed: dict[str, float] = {}

    def support_probe(workflow_id: str) -> dict[str, float]:
        response = probe(workflow_id)
        if not isinstance(response, Mapping) or "support_gain" not in response:
            raise ValueError("probe response requires support_gain")
        delayed[workflow_id] = float(
            response.get("fixed_query_gain", response.get("query_gain", 0.0))
        )
        return {"support_gain": float(response["support_gain"])}

    plan = plan_support_only(
        workflow_supply, probe_order, support_probe, control=control
    )
    return attach_delayed_outcomes(plan, delayed)


def execute_stop_on_first_positive(
    workflow_supply: Sequence[str],
    probe_order: Sequence[str],
    probe: ProbeCallback,
) -> dict[str, object]:
    """Legacy one-call Support planning plus immediate outcome evaluation."""

    return _execute_with_legacy_probe(
        workflow_supply,
        probe_order,
        probe,
        control="stop_on_first_positive",
    )


def execute_keep_best_support_so_far(
    workflow_supply: Sequence[str],
    probe_order: Sequence[str],
    probe: ProbeCallback,
) -> dict[str, object]:
    """Legacy pre-evolution control with immediate outcome evaluation."""

    return _execute_with_legacy_probe(
        workflow_supply,
        probe_order,
        probe,
        control="keep_best_support_so_far",
    )


def policy_adaptation_auc(curve: Sequence[Mapping[str, object]]) -> float:
    """Trapezoidal fixed-query adaptation AUC over contiguous budgets."""

    if len(curve) < 2:
        raise ValueError("adaptation curve requires at least two budget points")
    budgets = [int(row["budget"]) for row in curve]
    if budgets != list(range(len(curve))):
        raise ValueError("adaptation curve budgets must be contiguous from zero")
    gains = [float(row["fixed_query_gain"]) for row in curve]
    return sum(
        0.5 * (gains[index] + gains[index + 1])
        for index in range(len(gains) - 1)
    ) / float(len(gains) - 1)


def build_policy_failure_dossier(
    failure_cases: Sequence[Mapping[str, object]],
    *,
    allowed_observations: Sequence[str],
    allowed_controls: Sequence[str],
    allowed_compositions: Sequence[Mapping[str, object]] = (),
    allowed_scopes: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Compress exposed Policy failures into categorical first-fault evidence.

    Numeric responses are used locally to diagnose behavior but are not copied to
    the returned LLM-facing Dossier.  Fault types are added only after a natural
    failure family has measured the corresponding behavior.
    """

    observations = tuple(str(value) for value in allowed_observations)
    controls = tuple(str(value) for value in allowed_controls)
    if not observations or len(set(observations)) != len(observations):
        raise ValueError("allowed_observations must be non-empty and unique")
    if not controls or len(set(controls)) != len(controls):
        raise ValueError("allowed_controls must be non-empty and unique")
    compositions: list[dict[str, object]] = []
    for value in allowed_compositions:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("allowed_compositions must contain non-empty objects")
        normalized = copy.deepcopy(dict(value))
        if normalized in compositions:
            raise ValueError("allowed_compositions must be unique")
        compositions.append(normalized)
    scopes: list[dict[str, object]] = []
    for value in allowed_scopes:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("allowed_scopes must contain non-empty objects")
        normalized = copy.deepcopy(dict(value))
        if normalized in scopes:
            raise ValueError("allowed_scopes must be unique")
        scopes.append(normalized)
    if not failure_cases:
        raise ValueError("failure_cases are required")

    observation_fault_count = 0
    overwrite_fault_count = 0
    support_query_harm_count = 0
    cohort_topology_scope_fault_count = 0
    for case in failure_cases:
        transport_replays = case.get("support_to_query_replays")
        if transport_replays is not None:
            if not isinstance(transport_replays, Sequence) or isinstance(
                transport_replays, (str, bytes)
            ):
                raise ValueError("support_to_query_replays must be a sequence")
            for replay in transport_replays:
                if not isinstance(replay, Mapping):
                    raise ValueError("Support-to-Query replay must be an object")
                selected_program = str(replay.get("selected_program", ""))
                support_gains = replay.get("support_gains")
                if not selected_program or not isinstance(support_gains, Mapping):
                    raise ValueError(
                        "Support-to-Query replay lacks selection or Support gains"
                    )
                if selected_program.startswith("IDENTITY"):
                    continue
                if selected_program not in support_gains:
                    raise ValueError("selected Program lacks a Support response")
                support_gain = float(support_gains[selected_program])
                query_gain = float(replay["query_gain"])
                if not math.isfinite(support_gain) or not math.isfinite(query_gain):
                    raise ValueError("Support-to-Query gains must be finite")
                harmful_transport = support_gain > 0.0 and query_gain < 0.0
                source_topology = str(case.get("source_cohort_topology", ""))
                target_topology = str(case.get("target_cohort_topology", ""))
                topology_mismatch = bool(
                    source_topology
                    and target_topology
                    and source_topology != target_topology
                )
                if harmful_transport and topology_mismatch:
                    cohort_topology_scope_fault_count += 1
                else:
                    support_query_harm_count += int(harmful_transport)
            continue

        curve = case.get("candidate_curve")
        responses = case.get("workflow_responses")
        order = case.get("candidate_probe_order")
        if (
            not isinstance(curve, Sequence)
            or isinstance(curve, (str, bytes))
            or not isinstance(responses, Mapping)
            or not isinstance(order, Sequence)
            or isinstance(order, (str, bytes))
        ):
            raise ValueError("failure case lacks curve, responses, or probe order")
        candidate_auc = policy_adaptation_auc(curve)
        ordered_ids = tuple(str(value) for value in order)
        if set(ordered_ids) != set(str(value) for value in responses):
            raise ValueError("failure response set does not match probe order")

        comparison = case.get("comparison_adaptation_auc")
        if comparison is not None and candidate_auc < float(comparison):
            delayed_positive = False
            for index, workflow_id in enumerate(ordered_ids):
                response = responses[workflow_id]
                if not isinstance(response, Mapping):
                    raise ValueError("failure Workflow response must be an object")
                if (
                    index > 0
                    and float(response["support_gain"]) > 0.0
                    and float(response["query_gain"]) > 0.0
                ):
                    delayed_positive = True
            observation_fault_count += int(delayed_positive)

        query_gains = [float(row["fixed_query_gain"]) for row in curve]
        overwritten = any(
            earlier > 0.0 and later < 0.0
            for index, earlier in enumerate(query_gains)
            for later in query_gains[index + 1 :]
        )
        overwrite_fault_count += int(overwritten)

    faults: list[dict[str, object]] = []
    if observation_fault_count:
        faults.append(
            {
                "surface": "observation",
                "code": "GLOBAL_WORKFLOW_ORDER_NOT_TARGET_CONTEXTUALIZED",
                "observed_behavior": (
                    "a positive Workflow is delayed and the candidate adapts more "
                    "slowly than the comparison policy"
                ),
            }
        )
    if overwrite_fault_count:
        faults.append(
            {
                "surface": "harness_update_policy",
                "code": "CONFIRMED_POSITIVE_WORKFLOW_OVERWRITTEN",
                "observed_behavior": (
                    "continued probing replaces an earlier positive Workflow with "
                    "a harmful later choice"
                ),
            }
        )
    if support_query_harm_count:
        if not compositions:
            raise ValueError(
                "a diagnosed Support-to-Query fault requires a bounded composition"
            )
        faults.append(
            {
                "surface": "workflow_composition",
                "code": "ONE_SUPPORT_PROBE_CAN_FALSELY_CONFIRM",
                "observed_behavior": (
                    "a Program selected by a positive current-Support response is "
                    "harmful on the paired Query cohort"
                ),
            }
        )
    if cohort_topology_scope_fault_count:
        if not scopes:
            raise ValueError(
                "a diagnosed cohort-topology scope fault requires a bounded scope"
            )
        faults.append(
            {
                "surface": "applicability",
                "code": "SOURCE_SCOPE_OMITS_COHORT_TOPOLOGY",
                "observed_behavior": (
                    "Source evidence comes from a different cohort topology and a "
                    "positive Support confirmation is harmful on the Target Query"
                ),
            }
        )
    if not faults:
        raise ValueError("no supported first fault was diagnosed")

    return {
        "dossier_id": (
            "policy_episode_first_faults_v3"
            if cohort_topology_scope_fault_count
            else (
                "policy_episode_first_faults_v2"
                if support_query_harm_count
                else "policy_episode_first_faults_v1"
            )
        ),
        "categorical_first_faults": faults,
        "failure_case_count": len(failure_cases),
        "fault_support": {
            "observation_fault_case_count": observation_fault_count,
            "overwrite_fault_case_count": overwrite_fault_count,
            "support_to_query_harmful_replay_count": support_query_harm_count,
            "cohort_topology_scope_fault_count": cohort_topology_scope_fault_count,
        },
        "allowed_patch_values": {
            "ADD_OBSERVATION": list(observations),
            "PATCH_CONTROL": list(controls),
            "COMPOSE_WORKFLOW": compositions,
            "RESTRICT_SCOPE": scopes,
        },
        "forbidden_changes": [
            "workflow_supply",
            "program_supply",
            "consumer",
            "metric",
            "memory_schema",
            "query_visibility",
        ],
        "privacy": {
            "raw_time_series_included": False,
            "dataset_identity_included": False,
            "effect_magnitudes_included": False,
        },
    }


def validate_failure_driven_patch(
    proposal: Mapping[str, object],
    candidate: Mapping[str, object],
    dossier: Mapping[str, object],
) -> dict[str, object]:
    """Bind one LLM patch to the diagnosed faults and supplied patch values."""

    faults = dossier.get("categorical_first_faults")
    allowed = dossier.get("allowed_patch_values")
    if not isinstance(faults, Sequence) or not isinstance(allowed, Mapping):
        raise ValueError("invalid Failure Dossier")
    required_operations = {
        "observation": "ADD_OBSERVATION",
        "harness_update_policy": "PATCH_CONTROL",
        "workflow_composition": "COMPOSE_WORKFLOW",
        "applicability": "RESTRICT_SCOPE",
    }
    required = {
        required_operations[str(row["surface"])]
        for row in faults
        if isinstance(row, Mapping) and str(row.get("surface")) in required_operations
    }
    operations = proposal.get("operations")
    if not isinstance(operations, list) or len(operations) != len(required):
        raise ValueError("patch must contain exactly one operation per first fault")
    seen: set[str] = set()
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("patch operation must be an object")
        op = str(operation.get("operation"))
        target = str(operation.get("target_surface"))
        value = operation.get("value")
        if op not in required or op in seen:
            raise ValueError("patch operation does not match diagnosed first faults")
        expected_target = {
            "ADD_OBSERVATION": "observation",
            "PATCH_CONTROL": "harness_update_policy",
            "COMPOSE_WORKFLOW": "workflow",
            "RESTRICT_SCOPE": "applicability",
        }[op]
        if target != expected_target:
            raise ValueError("patch target does not match diagnosed surface")
        choices = allowed.get(op)
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ValueError("Failure Dossier lacks allowed patch values")
        if value not in choices:
            raise ValueError("patch value is outside the Failure Dossier catalog")
        seen.add(op)
    if seen != required:
        raise ValueError("patch does not cover every diagnosed first fault")
    return validate_typed_patch(proposal, candidate)


def _episode_curve_for_order(
    episode: Mapping[str, object], order: tuple[str, ...]
) -> list[dict[str, object]]:
    workflows = _episode_workflows(episode, order)
    responses = iter(workflows[workflow_id] for workflow_id in order)
    return execute_stop_on_first_positive(
        order,
        order,
        lambda _workflow_id: next(responses),
    )["adaptation_curve"]  # type: ignore[return-value]


def compile_source_workflow_prior(
    source_policy_episodes: Sequence[Mapping[str, object]],
    workflow_supply: Sequence[str],
) -> dict[str, object]:
    """Compile a probe-order prior from complete Source PolicyEpisodes."""

    workflow_ids = validate_workflow_supply(workflow_supply)
    if not source_policy_episodes:
        raise ValueError("at least one Source PolicyEpisode is required")
    for episode in source_policy_episodes:
        _episode_workflows(episode, workflow_ids)

    order_scores: list[tuple[float, tuple[str, ...]]] = []
    for order in permutations(workflow_ids):
        aucs = [
            policy_adaptation_auc(_episode_curve_for_order(episode, order))
            for episode in source_policy_episodes
        ]
        order_scores.append((fmean(aucs), order))
    score, order = max(order_scores, key=lambda row: (row[0], row[1]))
    return {
        "workflow_order": list(order),
        "source_policy_episode_count": len(source_policy_episodes),
        "mean_source_adaptation_auc": float(score),
        "candidate_order_count": len(order_scores),
    }


def build_candidate_skill(
    capability_memory: Sequence[Mapping[str, object]],
    source_policy_episodes: Sequence[Mapping[str, object]],
    *,
    capability_id: str,
    task_context: Mapping[str, object],
    workflow_supply: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build a CANDIDATE card from Memory and Source PolicyEpisodes.

    This function intentionally cannot promote the card.  A separate external
    intervention/promotion result must decide any final status.
    """

    if not capability_id:
        raise ValueError("capability_id is required")
    for existing in capability_memory:
        if existing.get("capability_id") == capability_id:
            raise ValueError("capability_id already exists in Capability Memory")
    if not isinstance(task_context, Mapping) or not task_context:
        raise ValueError("task_context is required")
    if not source_policy_episodes:
        raise ValueError("Source PolicyEpisodes are required")
    first_workflows = _episode_workflows(source_policy_episodes[0])
    supplied = validate_workflow_supply(
        workflow_supply if workflow_supply is not None else tuple(first_workflows)
    )
    prior = compile_source_workflow_prior(source_policy_episodes, supplied)
    candidate = {
        "capability_id": capability_id,
        "status": CANDIDATE,
        "task_context": copy.deepcopy(dict(task_context)),
        "workflow_supply": list(supplied),
        "observation": {
            "type": "source_policy_episode_workflow_prior",
            "use": "order current-Support Workflow probes",
            "utility_claim": "prior only; current Support must confirm utility",
        },
        "source_prior": prior,
        "control": {
            "type": "keep_best_support_so_far",
            "confirmation": "current Support exact grouped gain > 0",
            "selection": "largest positive current-Support gain seen so far",
            "fallback": IDENTITY,
        },
        "risk": {
            "abstain_if_no_positive_confirmation": True,
            "do_not_use_query_future_for_ordering_or_confirmation": True,
            "do_not_allow_later_probe_to_overwrite_confirmed_workflow": False,
        },
        "evidence": {
            "source_policy_episode_count": len(source_policy_episodes),
            "promotion_status": "NOT_EVALUATED",
        },
    }
    return validate_skill_card(candidate)


def apply_promotion_result(
    candidate: Mapping[str, object], promotion: PromotionCallback
) -> dict[str, object]:
    """Apply an explicit external promotion result without inferring support."""

    validated = validate_skill_card(candidate)
    if validated["status"] != CANDIDATE:
        raise ValueError("only a CANDIDATE card can receive a promotion result")
    result = promotion(copy.deepcopy(validated))
    if not isinstance(result, Mapping):
        raise ValueError("promotion callback must return an object")
    status = result.get("status")
    if status not in _EXECUTABLE_STATUSES | {"REJECTED", "RESTRICTED"}:
        raise ValueError("promotion callback returned an unsupported final status")
    resolved = copy.deepcopy(validated)
    resolved["status"] = status
    resolved["promotion_result"] = copy.deepcopy(dict(result))
    return validate_skill_card(resolved)


def run_skill_acquisition_cycle(
    capability_memory: Sequence[Mapping[str, object]],
    source_policy_episodes: Sequence[Mapping[str, object]],
    validation_cases: Sequence[Mapping[str, object]],
    *,
    capability_id: str,
    task_context: Mapping[str, object],
    workflow_supply: Sequence[str],
    typed_patch: Mapping[str, object],
    promotion: CyclePromotionCallback,
) -> dict[str, object]:
    """Run one bounded Candidate -> patch -> replay -> promotion cycle.

    Dataset loading, Consumer fits, proxy screening, and the promotion criterion
    remain caller-owned adapters.  This function owns only the reusable Harness
    control flow, so a new natural task can supply episodes and callbacks without
    copying the acquisition logic into another experiment runner.
    """

    if not validation_cases:
        raise ValueError("at least one validation case is required")
    candidate = build_candidate_skill(
        capability_memory,
        source_policy_episodes,
        capability_id=capability_id,
        task_context=task_context,
        workflow_supply=workflow_supply,
    )
    patched = apply_typed_patch(candidate, typed_patch)
    replays: list[dict[str, object]] = []
    seen_case_ids: set[str] = set()
    for raw_case in validation_cases:
        case_id = raw_case.get("case_id")
        workflows = raw_case.get("current_workflows")
        if not isinstance(case_id, str) or not case_id or case_id in seen_case_ids:
            raise ValueError("validation cases require unique non-empty case_id")
        if not isinstance(workflows, Mapping):
            raise ValueError("validation case requires current_workflows")
        historical_episode = raw_case.get("historical_episode")
        if historical_episode is not None and not isinstance(
            historical_episode, Mapping
        ):
            raise ValueError("historical_episode must be an object")
        seen_case_ids.add(case_id)
        replay = execute_skill_card(
            patched,
            lambda workflow_id, rows=workflows: rows[workflow_id],
            historical_episode=historical_episode,
            allow_candidate_replay=True,
        )
        replays.append({"case_id": case_id, "replay": replay})

    promotion_result = promotion(copy.deepcopy(patched), copy.deepcopy(replays))
    resolved = apply_promotion_result(
        patched, lambda _candidate: promotion_result
    )
    return {
        "candidate_before_patch": candidate,
        "candidate_after_patch": patched,
        "validation_replays": replays,
        "resolved_skill": resolved,
    }


def validate_skill_card(skill: Mapping[str, object]) -> dict[str, object]:
    """Validate the small executable Skill-card contract."""

    if not isinstance(skill, Mapping):
        raise ValueError("Skill card must be an object")
    card = copy.deepcopy(dict(skill))
    if not isinstance(card.get("capability_id"), str) or not card["capability_id"]:
        raise ValueError("Skill card requires capability_id")
    if not isinstance(card.get("status"), str) or not card["status"]:
        raise ValueError("Skill card requires status")
    supply = validate_workflow_supply(card.get("workflow_supply", ()))
    card["workflow_supply"] = list(supply)
    if not isinstance(card.get("task_context"), Mapping):
        raise ValueError("Skill card requires task_context")
    if not isinstance(card.get("observation"), Mapping):
        raise ValueError("Skill card requires observation")
    control = card.get("control")
    if not isinstance(control, Mapping):
        raise ValueError("Skill card requires control")
    if control.get("type") not in {
        "keep_best_support_so_far",
        "stop_on_first_positive",
    }:
        raise ValueError("unsupported Skill control")
    if control.get("fallback") != IDENTITY:
        raise ValueError("Skill control must preserve IDENTITY fallback")
    risk = card.get("risk")
    if not isinstance(risk, Mapping) or not all(
        risk.get(key) is True
        for key in (
            "abstain_if_no_positive_confirmation",
            "do_not_use_query_future_for_ordering_or_confirmation",
        )
    ):
        raise ValueError("Skill card requires the frozen Support/risk contract")
    overwrite_guard = risk.get(
        "do_not_allow_later_probe_to_overwrite_confirmed_workflow"
    )
    if not isinstance(overwrite_guard, bool):
        raise ValueError("Skill card requires an explicit overwrite-risk state")
    if control.get("type") == "stop_on_first_positive" and not overwrite_guard:
        raise ValueError("stop-on-first-positive requires the overwrite guard")
    prior = card.get("source_prior")
    if prior is not None:
        if not isinstance(prior, Mapping):
            raise ValueError("source_prior must be an object")
        order = tuple(str(value) for value in prior.get("workflow_order", ()))
        if len(order) != len(supply) or set(order) != set(supply):
            raise ValueError("source_prior does not match Workflow supply")
    applicability = card.get("applicability")
    if applicability is not None:
        if not isinstance(applicability, Mapping):
            raise ValueError("Skill applicability must be an object")
        cohort_topology = applicability.get("cohort_topology")
        if not isinstance(cohort_topology, str) or not cohort_topology:
            raise ValueError("Skill applicability requires cohort_topology")
        if applicability.get("on_mismatch") != "ABSTAIN":
            raise ValueError("Skill applicability mismatch must ABSTAIN")
    return card


def load_skill_card(path: str | Path) -> dict[str, object]:
    """Load and validate one ordinary JSON Skill card."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Skill-card JSON must contain an object")
    return validate_skill_card(payload)


def read_active_skill_cards(
    skill_cards: Sequence[Mapping[str, object]],
    *,
    state_updates: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Return only admitted cards for Fast-Path planning.

    Rejected and provisional cards remain available to the caller as Slow-Path
    evidence, but are never exposed as executable actions.  Full Program-specific
    validation still occurs when an admitted card reaches its executor.
    """

    updates: dict[str, dict[str, object]] = {}
    for raw in state_updates:
        if not isinstance(raw, Mapping):
            raise ValueError("Skill state updates must be objects")
        capability_id = raw.get("capability_id")
        status = raw.get("status")
        if (
            not isinstance(capability_id, str)
            or not capability_id
            or capability_id in updates
        ):
            raise ValueError("Skill state updates require unique capability ids")
        if (
            not isinstance(status, str)
            or not status
            or status in _EXECUTABLE_STATUSES
        ):
            raise ValueError("State updates may restrict, but never promote, a Skill")
        updates[capability_id] = copy.deepcopy(dict(raw))

    active: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw in skill_cards:
        if not isinstance(raw, Mapping):
            raise ValueError("Skill Memory entries must be objects")
        capability_id = raw.get("capability_id")
        status = raw.get("status")
        if (
            not isinstance(capability_id, str)
            or not capability_id
            or capability_id in seen_ids
        ):
            raise ValueError("Skill Memory requires unique non-empty capability ids")
        if not isinstance(status, str) or not status:
            raise ValueError("Skill Memory entries require status")
        seen_ids.add(capability_id)
        card = copy.deepcopy(dict(raw))
        if capability_id in updates:
            card["status"] = updates[capability_id]["status"]
            card["state_update"] = updates[capability_id]
        if card["status"] in _EXECUTABLE_STATUSES:
            active.append(card)
    unknown = set(updates) - seen_ids
    if unknown:
        raise ValueError("Skill state update references an unavailable capability")
    return active


def plan_skill_card_support_only(
    skill: Mapping[str, object],
    probe: SupportProbeCallback,
    *,
    historical_episode: Mapping[str, object] | None = None,
    allow_candidate_replay: bool = False,
    execution_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a Skill plan using only legally visible current-Support feedback."""

    card = validate_skill_card(skill)
    status = str(card["status"])
    if status not in _EXECUTABLE_STATUSES and not (
        allow_candidate_replay and status == CANDIDATE
    ):
        raise ValueError("Skill status is not executable")
    if "workflow_composition" in card:
        raise ValueError(
            "COMPOSE_WORKFLOW is a candidate composition; an external replay "
            "compiler must compile it before execution"
        )
    supply = tuple(str(value) for value in card["workflow_supply"])
    applicability = card.get("applicability")
    if isinstance(applicability, Mapping):
        actual_topology = (
            execution_context.get("cohort_topology")
            if isinstance(execution_context, Mapping)
            else None
        )
        required_topology = applicability["cohort_topology"]
        if actual_topology != required_topology:
            trace = [
                {
                    "budget": budget,
                    "selected_workflow": IDENTITY,
                    "abstained": True,
                    "terminal": True,
                }
                for budget in range(len(supply) + 1)
            ]
            return {
                "selected_workflow": IDENTITY,
                "abstained": True,
                "probed_workflows": [],
                "support_observations": [],
                "support_planning_trace": trace,
                "control": str(card["control"]["type"]),  # type: ignore[index]
                "capability_id": card["capability_id"],
                "status": status,
                "probe_order": [],
                "applicability_matched": False,
                "scope_reason": "COHORT_TOPOLOGY_MISMATCH_OR_UNAVAILABLE",
            }
    observation_type = card["observation"].get("type")  # type: ignore[union-attr]
    if observation_type == "phase_aligned_historical_policy_episode":
        if historical_episode is None:
            raise ValueError("historical PolicyEpisode is required by this Skill")
        order = compile_probe_order_from_historical_episode(
            historical_episode, supply
        )
    elif observation_type == "source_policy_episode_workflow_prior":
        prior = card.get("source_prior")
        if not isinstance(prior, Mapping):
            raise ValueError("Skill requires source_prior")
        order = tuple(str(value) for value in prior["workflow_order"])
    else:
        raise ValueError("unsupported Skill observation")
    result = plan_support_only(
        supply,
        order,
        probe,
        control=str(card["control"]["type"]),  # type: ignore[index]
    )
    result["capability_id"] = card["capability_id"]
    result["status"] = status
    result["probe_order"] = list(order)
    if isinstance(applicability, Mapping):
        result["applicability_matched"] = True
    return result


def execute_skill_card(
    skill: Mapping[str, object],
    probe: ProbeCallback,
    *,
    historical_episode: Mapping[str, object] | None = None,
    allow_candidate_replay: bool = False,
    execution_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Legacy Skill execution with immediate delayed-outcome availability.

    New Fast-Path callers should use :func:`plan_skill_card_support_only` and
    let an evaluator call :func:`attach_delayed_outcomes` only when forecasting
    outcomes have arrived.
    """

    delayed: dict[str, float] = {}

    def support_probe(workflow_id: str) -> dict[str, float]:
        response = probe(workflow_id)
        if not isinstance(response, Mapping) or "support_gain" not in response:
            raise ValueError("probe response requires support_gain")
        delayed[workflow_id] = float(
            response.get("fixed_query_gain", response.get("query_gain", 0.0))
        )
        return {"support_gain": float(response["support_gain"])}

    plan = plan_skill_card_support_only(
        skill,
        support_probe,
        historical_episode=historical_episode,
        allow_candidate_replay=allow_candidate_replay,
        execution_context=execution_context,
    )
    return attach_delayed_outcomes(plan, delayed)


def _reject_forbidden_patch_content(value: object, path: str = "patch") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_PATCH_KEYS:
                raise ValueError(f"typed patch cannot modify {normalized} at {path}")
            _reject_forbidden_patch_content(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_forbidden_patch_content(nested, f"{path}[{index}]")


def validate_typed_patch(
    patch: Mapping[str, object], skill: Mapping[str, object]
) -> dict[str, object]:
    """Validate one bounded Harness patch against an existing Skill card."""

    card = validate_skill_card(skill)
    if not isinstance(patch, Mapping):
        raise ValueError("typed patch must be an object")
    normalized = copy.deepcopy(dict(patch))
    operations = normalized.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("typed patch requires operations")
    supply = set(str(value) for value in card["workflow_supply"])
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("typed patch operation must be an object")
        op = operation.get("operation")
        target = operation.get("target_surface")
        if op not in _PATCH_OPERATIONS:
            raise ValueError(f"unsupported typed patch operation: {op}")
        allowed_targets = {
            "ADD_OBSERVATION": {"observation"},
            "PATCH_CONTROL": {"control", "harness_update_policy"},
            "COMPOSE_WORKFLOW": {"workflow", "control", "harness_update_policy"},
            "RESTRICT_SCOPE": {"applicability"},
        }[str(op)]
        if target not in allowed_targets:
            raise ValueError(f"{op} cannot modify surface {target}")
        value = operation.get("value")
        _reject_forbidden_patch_content(value, path=f"operation.{op}.value")
        if op == "COMPOSE_WORKFLOW" and isinstance(value, Mapping):
            referenced = value.get("workflow_order", value.get("workflow_ids"))
            if referenced is not None:
                if isinstance(referenced, (str, bytes)) or not isinstance(
                    referenced, Sequence
                ):
                    raise ValueError("composed Workflow references must be a sequence")
                if not set(str(item) for item in referenced).issubset(supply):
                    raise ValueError("typed patch cannot introduce a new Workflow")
        if op == "RESTRICT_SCOPE":
            if not isinstance(value, Mapping):
                raise ValueError("RESTRICT_SCOPE value must be an object")
            if set(value) != {"cohort_topology", "on_mismatch"}:
                raise ValueError("RESTRICT_SCOPE has unsupported applicability fields")
            if not isinstance(value.get("cohort_topology"), str) or not value.get(
                "cohort_topology"
            ):
                raise ValueError("RESTRICT_SCOPE requires cohort_topology")
            if value.get("on_mismatch") != "ABSTAIN":
                raise ValueError("RESTRICT_SCOPE mismatch must ABSTAIN")
    _reject_forbidden_patch_content(
        {key: value for key, value in normalized.items() if key != "operations"}
    )
    return normalized


def apply_typed_patch(
    skill: Mapping[str, object], patch: Mapping[str, object]
) -> dict[str, object]:
    """Apply a validated patch without changing evidence or promotion status."""

    card = validate_skill_card(skill)
    normalized = validate_typed_patch(patch, card)
    updated = copy.deepcopy(card)
    for operation in normalized["operations"]:
        op = operation["operation"]
        value = copy.deepcopy(operation.get("value"))
        if op == "ADD_OBSERVATION":
            updated["observation"] = (
                {"type": value} if isinstance(value, str) else value
            )
        elif op == "PATCH_CONTROL":
            control = {"type": value} if isinstance(value, str) else value
            if not isinstance(control, Mapping):
                raise ValueError("PATCH_CONTROL value must be a string or object")
            merged = copy.deepcopy(dict(updated["control"]))
            merged.update(dict(control))
            if merged.get("type") == "stop_on_first_positive":
                merged.pop("selection", None)
                merged["fallback"] = IDENTITY
                merged["confirmation"] = "current Support exact grouped gain > 0"
                merged["continue_when"] = "current Support exact grouped gain <= 0"
                updated["risk"] = {
                    **dict(updated["risk"]),
                    "do_not_allow_later_probe_to_overwrite_confirmed_workflow": True,
                }
            updated["control"] = merged
        elif op == "COMPOSE_WORKFLOW":
            if not isinstance(value, Mapping):
                raise ValueError("COMPOSE_WORKFLOW value must be an object")
            # This is intentionally declarative.  The current lean runtime does
            # not invent composition semantics; an experiment-specific replay
            # compiler must validate and compile it before execution.
            updated["workflow_composition"] = dict(value)
        elif op == "RESTRICT_SCOPE":
            if not isinstance(value, Mapping):
                raise ValueError("RESTRICT_SCOPE value must be an object")
            updated["applicability"] = dict(value)
    # A patch is a proposed behavior change, never promotion evidence.
    updated["status"] = CANDIDATE
    updated["applied_patch_id"] = normalized.get("patch_id", "unnamed_patch")
    return validate_skill_card(updated)


def run_failure_driven_update_cycle(
    candidate: Mapping[str, object],
    failure_cases: Sequence[Mapping[str, object]],
    *,
    allowed_observations: Sequence[str],
    allowed_controls: Sequence[str],
    allowed_compositions: Sequence[Mapping[str, object]] = (),
    allowed_scopes: Sequence[Mapping[str, object]] = (),
    propose_patch: FailurePatchProposalCallback,
    replay_patch: FailurePatchReplayCallback,
    resolve_patch: CyclePromotionCallback,
) -> dict[str, object]:
    """Run one bounded failure -> patch -> replay -> resolution Slow Path.

    The Harness owns categorical diagnosis and patch boundaries.  LLM/API access,
    Program-specific replay and the acceptance estimand stay behind callbacks.
    A proposal or a successful compilation can never promote itself.
    """

    card = validate_skill_card(candidate)
    if card["status"] != CANDIDATE:
        raise ValueError("failure-driven update requires a CANDIDATE Skill")
    dossier = build_policy_failure_dossier(
        failure_cases,
        allowed_observations=allowed_observations,
        allowed_controls=allowed_controls,
        allowed_compositions=allowed_compositions,
        allowed_scopes=allowed_scopes,
    )
    proposal = propose_patch(copy.deepcopy(dossier))
    if not isinstance(proposal, Mapping):
        raise ValueError("patch proposer must return one object")
    normalized_patch = validate_failure_driven_patch(proposal, card, dossier)
    patched = apply_typed_patch(card, normalized_patch)
    replay = replay_patch(copy.deepcopy(patched))
    if not isinstance(replay, Mapping):
        raise ValueError("patch replay must return one object")
    resolution = resolve_patch(copy.deepcopy(patched), [copy.deepcopy(replay)])
    if not isinstance(resolution, Mapping):
        raise ValueError("patch resolver must return one object")
    resolved = apply_promotion_result(patched, lambda _candidate: resolution)
    return {
        "failure_dossier": dossier,
        "candidate_before_patch": card,
        "typed_patch": normalized_patch,
        "candidate_after_patch": patched,
        "replay": copy.deepcopy(dict(replay)),
        "resolved_skill": resolved,
    }


__all__ = [
    "CANDIDATE",
    "IDENTITY",
    "apply_promotion_result",
    "apply_typed_patch",
    "attach_delayed_outcomes",
    "build_candidate_skill",
    "build_policy_failure_dossier",
    "collect_source_policy_episodes",
    "compile_probe_order_from_historical_episode",
    "compile_source_workflow_prior",
    "execute_skill_card",
    "execute_keep_best_support_so_far",
    "execute_stop_on_first_positive",
    "load_skill_card",
    "policy_adaptation_auc",
    "plan_skill_card_support_only",
    "plan_support_only",
    "read_active_skill_cards",
    "run_failure_driven_update_cycle",
    "run_skill_acquisition_cycle",
    "validate_failure_driven_patch",
    "validate_skill_card",
    "validate_typed_patch",
    "validate_workflow_supply",
]
