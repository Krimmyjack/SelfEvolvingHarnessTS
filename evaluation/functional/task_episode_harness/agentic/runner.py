"""G1: one command that runs the whole core Pipeline end to end.

Frozen design §13/G1 lists ten things that had never been connected in one
place.  They connect here:

1. Context-conditioned retrieval  -- ``resolve_harness_view`` on the Task's own
   projection, plus the arm-local Skill and Experience retrieval;
2. General / Specific / Experience -- the resolved Harness view (instruction,
   bootstrap and capability Skills, ``candidate_policy``) together with the
   arm-local Cards and the positive / negative / conflict Episode contrast;
3. FastAgent inspect / propose / select -- the frozen stage contracts driven by
   ``TTHAAgentCore``'s real multi-round tool loop;
4. bounded Workspace tools        -- :mod:`.gateway`, one call at a time, on a
   series the Agent names, capped per Task;
5. Runtime per-unit dispatch      -- :mod:`.dispatch`, parameter ownership
   enforced before any action unit runs;
6. Support / execute / abstain    -- the frozen paired evaluator, unchanged;
7. Episode / Local Skill lifecycle-- ``_make_episode`` and ``_lifecycle``;
8. deterministic attribution      -- :func:`attribute_first_fault`;
9. Slow single-surface edit       -- the existing ``EditController`` route, run
   only when attribution names an editable surface;
10. replay back to the Fast Path  -- the patched snapshot re-enters (3).

Cost is kept in four separate columns for the whole run (§6.3): Workspace tool
calls, LLM calls, real Support probes, and charged probe cost.  The last is a
budget penalty and is never reported as a probe count.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[4]
import sys

for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from SelfEvolvingHarnessTS.contracts.program_supply import (
    route_program_supply_fault,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    AgentTransportError,
    AgictoChatCompletionsBackend,
    BudgetedAgentBackend,
)

from evaluation.functional.task_episode_harness import g1
from evaluation.functional.task_episode_harness.e1 import (
    B,
    HORIZON,
    MATERIAL_THRESHOLD,
    _ArmState,
    _frozen_task_roster,
    _inventory_rows,
    _lifecycle,
    _load_kdd_roster,
    _make_episode,
    _retrieve_target_local_skills,
    _skill_ids,
    _source_card_from_report,
    _source_bundle_from_report,
    _source_prior_for_task,
    _sync_memory,
    _workflow_signature,
)
from evaluation.functional.task_episode_harness.normal_flow import (
    NF_BASE_URL,
    NF_MODEL,
)
from evaluation.functional.task_episode_harness.runner import _mapped_roster
from evaluation.functional.task_episode_harness.skill_evolution import (
    TASK_CONSUMER_KEY,
    _probe_compiled,
)

from .dispatch import exploration_concentration
from .fast_path import (
    STOP_ABSTAIN,
    STOP_INSTRUMENT,
    STOP_NO_DRAFT,
    STOP_REQUEST_OBSERVATION,
    _plain,
    run_agentic_fast_path,
)
from .gateway import CohortScopePublicToolGateway

PROTOCOL_VERSION = "g1_agentic_pipeline_v1"
STATE_REL = ".g1_pipeline_state"
REPORT_REL = (
    PROJECT_ROOT / "artifacts/functional/e2" / "g1_agentic_pipeline_report.json"
)
COLD_ARM = "A3"
WARM_ARM = "A5"
WORKSPACE_TOOL_BUDGET = 6
# inspect + propose + one select per probed candidate, plus tool rounds and one
# schema retry each.  A hard ceiling, not a target.
LLM_CALL_BUDGET_PER_ARM_TASK = 24


# ----------------------------------------------------------------- cohorts
def load_cohort(repo_root: Path, name: str) -> dict[str, Any]:
    """Already-exposed development cohorts only.  Nothing sealed is reachable.

    ``e31`` is kept selectable for focused replay but is not the default: at
    the frozen Task-roster origins all eight of its eval series collapse to the
    scale floor, so the Judge cannot measure a Task on it.  That is the
    recorded substrate fact, not a reason to move an origin.
    """
    if name == "e31":
        roster, values, _selected = _load_kdd_roster(
            repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e31.jsonl"
        )
        train = [r["series_uid"] for r in roster if r["role"] == "train"]
        evaluation = [r["series_uid"] for r in roster if r["role"] == "eval"]
    elif name == "T233":
        roster, values = g1._a5a3_cohort(repo_root)
        train = list(g1.A5A3_COHORT_TRAIN)
        evaluation = list(g1.A5A3_COHORT_EVAL)
    elif name == "weather":
        cohort = g1.freeze_weather_cohort(repo_root)
        roster, values = cohort["roster"], cohort["values"]
        train = [str(uid) for uid in cohort["train"]]
        evaluation = [str(uid) for uid in cohort["eval"]]
    else:
        raise ValueError(f"unknown development cohort: {name!r}")
    return {
        "name": name,
        "roster": roster,
        "mapped_roster": _mapped_roster(roster),
        "values": values,
        "train_uids": train,
        "eval_uids": evaluation,
        "exposure": "already exposed development data; not fresh",
    }


class _RetryingTransport:
    """Bounded retry around a flaky relay, and nothing else.

    The second live nine-Task run died at Task three on an APIConnectionError.
    The backend already classifies transient relay errors and raises
    AgentTransportError for them; nobody was retrying.  Retries are counted
    separately from stage requests so LLM cost stays honest -- a retried
    request is one stage decision and more than one API call.
    """

    def __init__(self, delegate: Any, *, attempts: int = 3,
                 backoff_seconds: float = 2.0) -> None:
        self.delegate = delegate
        self.attempts = int(attempts)
        self.backoff_seconds = float(backoff_seconds)
        self.transport_retries = 0

    def complete(self, request: Any) -> Any:
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                return self.delegate.complete(request)
            except AgentTransportError as exc:
                last = exc
                self.transport_retries += 1
                if attempt + 1 < self.attempts:
                    time.sleep(self.backoff_seconds * (attempt + 1))
        raise last  # type: ignore[misc]


def _default_backend_factory(maximum_calls: int) -> BudgetedAgentBackend:
    import os

    api_key = next(
        (
            os.environ.get(name, "").strip()
            for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
            if os.environ.get(name, "").strip()
        ),
        None,
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    return BudgetedAgentBackend(
        _RetryingTransport(
            AgictoChatCompletionsBackend(
                api_key=api_key, base_url=NF_BASE_URL, timeout_seconds=240
            )
        ),
        maximum_calls=maximum_calls,
    )


# ----------------------------------------------------------- knowledge inlet
def _experience_contrast(
    arm_state: _ArmState,
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Positive / negative / conflict Episode contrast, arm-local only.

    §11: an Episode is a fact about one Action--Response pair.  It is returned
    as evidence, never as an instruction, and A3 never sees A5's.
    """
    positive = [row for row in arm_state.memories if row.get("relation") == "POSITIVE"]
    negative = [row for row in arm_state.memories if row.get("relation") == "NEGATIVE"]
    signatures_positive = {row.get("workflow") for row in positive}
    signatures_negative = {row.get("workflow") for row in negative}
    conflicting = sorted(
        str(signature)
        for signature in signatures_positive & signatures_negative
        if signature
    )
    return {
        "positive_episodes": [dict(row) for row in positive[-limit:]],
        "negative_episodes": [dict(row) for row in negative[-limit:]],
        "conflicting_workflow_signatures": conflicting,
        "note": (
            "Episodes are observed Action-Response facts from this arm's own "
            "history. They are evidence, not instructions, and they never "
            "replace a Target Support probe."
        ),
    }


def _initial_context(
    *,
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    cohort: Mapping[str, Any],
    workspace_tool_budget: int,
) -> dict[str, Any]:
    """What the Agent knows before it observes anything.

    Deliberately carries no per-series numbers: §12 records "Context is still
    mostly precompressed by the Runtime into a representative-series summary"
    as one of the un-connected places, and a Task-level summary handed over
    for free would make "the tool result changed the decision" unfalsifiable.
    """
    scope = sorted(public_context.get("scope_series_uids") or ())
    return {
        "task": TASK_CONSUMER_KEY,
        "task_episode_id": str(task_spec["task_episode_id"]),
        "objective": (
            "Choose a Typed Workflow that improves the downstream forecast "
            "Consumer on this Task's scoped training series, or abstain."
        ),
        "task_kind": public_context["task_kind"],
        "metric": {"name": "sMASE", "direction": "lower_is_better",
                   "reported_as": "macro gain over identity"},
        "horizon": int(task_spec.get("horizon") or HORIZON),
        "observation_cutoff": int(public_context["observation_cutoff"]),
        "cohort": {
            "dataset": cohort["name"],
            "training_series_count": len(cohort["train_uids"]),
            "evaluation_series_count": len(cohort["eval_uids"]),
            "exposure": cohort["exposure"],
        },
        "scope": {
            "selector_feature": public_context["scope_feature"],
            "selector_bin": public_context["scope_bin"],
            "series_uids": scope,
            "series_count": len(scope),
            "note": (
                "The Program you propose runs on every series in this Scope, "
                "on 240-point training windows, not on one representative "
                "series and not on the full public prefix."
            ),
        },
        "workspace_tool_budget": int(workspace_tool_budget),
        "information_wall": (
            "Workspace tools read the public prefix only. No future value, no "
            "delayed Outcome and no Consumer utility is reachable through them."
        ),
    }


# ------------------------------------------------------------- one arm, one Task
def _run_arm(
    *,
    repo_root: Path,
    arm_state: _ArmState,
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    cohort: Mapping[str, Any],
    config: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    source_prior: Mapping[str, Any] | None,
    workspace_tool_budget: int,
    backend_factory: Callable[[int], Any],
) -> dict[str, Any]:
    arm = arm_state.arm
    scope = frozenset(public_context["scope_series_uids"])
    cutoff = int(public_context["observation_cutoff"])
    support_origins = tuple(int(o) for o in task_spec["support_origins"])
    delayed_origins = tuple(int(o) for o in task_spec["delayed_origins"])
    values = cohort["values"]
    mapped_roster = cohort["mapped_roster"]
    eval_uids = list(cohort["eval_uids"])

    gateway = CohortScopePublicToolGateway(
        {uid: np.asarray(values[uid], dtype=np.float64)[:cutoff] for uid in scope},
        task_kind=str(public_context["task_kind"]),
        observation_cutoff=cutoff,
        maximum_calls=int(workspace_tool_budget),
    )
    backend = backend_factory(LLM_CALL_BUDGET_PER_ARM_TASK)
    core = TTHAAgentCore(
        backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL
    )
    harness_view = resolve_harness_view(
        arm_state.active_snapshot,
        dict(public_context["task_fast_features"]),
        role="fast",
    )
    local_skills = _retrieve_target_local_skills(
        arm_state.active_snapshot, public_context, arm=arm
    )
    retrieved = {
        "general": {
            "carrier": "resolved Harness view (instruction, Skills, controls)",
            "candidate_policy": _plain(
                harness_view.controls.get("candidate_policy") or {}
            ),
            "retrieved_skill_ids": list(harness_view.skill_ids),
        },
        "specific_target_local_cards": [
            dict(row) for row in local_skills
            if row.get("retrieved_in_current_context")
        ],
        "specific_non_matching_cards_as_evidence_template": [
            dict(row) for row in local_skills
            if not row.get("retrieved_in_current_context")
        ],
        "experience": _experience_contrast(arm_state),
        "source_prior": (
            _plain(source_prior) if source_prior is not None else None
        ),
    }
    initial_context = _initial_context(
        task_spec=task_spec,
        public_context=public_context,
        cohort=cohort,
        workspace_tool_budget=workspace_tool_budget,
    )

    def support_probe(compiled: Any) -> Mapping[str, Any]:
        return _probe_compiled(
            mapped_roster, values, dict(config), support_origins,
            eval_uids, compiled, scope,
        )

    trace = run_agentic_fast_path(
        core=core,
        case_id=f"{arm}_{task_spec['task_episode_id']}",
        harness_view=harness_view,
        initial_context=initial_context,
        retrieved=retrieved,
        inventory=inventory,
        probe_budget=B,
        material_threshold=MATERIAL_THRESHOLD,
        support_probe=support_probe,
    )

    # ---- Episode + Local Skill lifecycle --------------------------------
    winner = None
    winner_compiled = None
    stop_reason = trace.stop_reason
    instrument_unreadable = trace.instrument_unreadable
    lifecycle: dict[str, Any] = {
        "method_event": {"stage": "no_winner"},
        "delayed_event": {"stage": "no_winner"},
    }
    active_before = _skill_ids(arm_state.active_snapshot, local_only=True)
    active_after = list(active_before)
    delayed_probe = None

    probed_rows = [row for row in trace.probes if row.get("status") == "PROBED"]
    for row in probed_rows:
        compiled_row = next(
            entry for entry in trace.compiled
            if entry.get("candidate_id") == row["candidate_id"]
            and entry.get("status") == "COMPILED"
        )
        episode = _make_episode(
            arm=arm,
            task_episode_id=str(task_spec["task_episode_id"]),
            attempt_index=int(row["attempt_index"]),
            compiled=compiled_row["workflow"],
            workflow_signature=_workflow_signature(
                compiled_row["workflow"].candidate.program.execution_steps()
            ),
            scope=scope,
            probe=row["support"],
            support_origins=support_origins,
            public_context=public_context,
        )
        row["episode_id"] = episode.episode_id
        _sync_memory(arm_state.memories, episode)
        arm_state.episodes.append(episode)
        if row["candidate_id"] == trace.chosen_candidate_id:
            winner = episode
            winner_compiled = compiled_row["workflow"]

    if winner is not None and winner_compiled is not None:
        try:
            (
                method_event,
                delayed_event,
                updated,
                delayed_probe,
                active_state,
            ) = _lifecycle(
                repo_root=repo_root,
                arm=arm,
                arm_state=arm_state,
                winner=winner,
                compiled=winner_compiled,
                workflow_signature=winner.workflow_signature,
                scope=scope,
                values=values,
                mapped_roster=mapped_roster,
                config=dict(config),
                eval_uids=eval_uids,
                delayed_origins=delayed_origins,
                public_context=public_context,
            )
            arm_state.active_snapshot = active_state["snapshot"]
            lifecycle = {
                "method_event": method_event,
                "delayed_event": delayed_event,
                "reused_existing_skill": bool(
                    active_state.get("reused_existing_skill")
                ),
                "local_skill_ids_before": active_state["local_skill_ids_before"],
                "local_skill_ids_after": active_state["local_skill_ids_after"],
            }
            active_after = list(active_state["local_skill_ids_after"])
            winner = updated
            for index, episode in enumerate(arm_state.episodes):
                if episode.episode_id == winner.episode_id:
                    arm_state.episodes[index] = winner
                    break
            _sync_memory(arm_state.memories, winner)
            arm_state.active_skill_ids = active_after
        except Exception as exc:  # noqa: BLE001
            instrument_unreadable = True
            stop_reason = STOP_INSTRUMENT
            lifecycle = {
                "method_event": {
                    "stage": "instrument_unreadable",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "delayed_event": {"stage": "instrument_unreadable"},
            }
            winner = None

    local_active = bool(
        winner is not None
        and str(getattr(winner, "local_status", "")) == "LOCAL_ACTIVE"
    )
    real_probe_count = len(probed_rows)
    charged_probe_cost = real_probe_count if local_active else B + 1

    return {
        "arm": arm,
        "stop_reason": stop_reason,
        "chosen_candidate_id": trace.chosen_candidate_id,
        "protocol_error": trace.protocol_error,
        "protocol_error_output": trace.protocol_error_output,
        "infrastructure_error": trace.infrastructure_error,
        "stages": trace.stages,
        "tool_observations": trace.tool_observations,
        "proposals": trace.proposals,
        "compiled": [
            {key: value for key, value in row.items() if key != "workflow"}
            for row in trace.compiled
        ],
        "parameter_ownership_audits": trace.ownership_audits,
        "probes": [
            {key: value for key, value in row.items() if key != "support"}
            for row in trace.probes
        ],
        "select_rounds": trace.select_rounds,
        "lifecycle": lifecycle,
        "delayed": delayed_probe,
        "winner": (
            {
                "episode_id": winner.episode_id,
                "workflow": winner.workflow_signature,
                "local_status": winner.local_status,
                "delayed_gain": winner.delayed_response.get("gain"),
                "delayed_gain_over_se": winner.delayed_response.get(
                    "gain_over_se"
                ),
            }
            if winner is not None else None
        ),
        "retrieved_knowledge_summary": {
            "retrieved_skill_ids": list(harness_view.skill_ids),
            "retrieved_memory_ids": list(harness_view.memory_ids),
            "target_local_card_count": len(
                retrieved["specific_target_local_cards"]
            ),
            "positive_episode_count": len(
                retrieved["experience"]["positive_episodes"]
            ),
            "negative_episode_count": len(
                retrieved["experience"]["negative_episodes"]
            ),
            "source_prior_matched": source_prior is not None,
        },
        "active_local_skill_ids_before": active_before,
        "active_local_skill_ids_after": active_after,
        "cost": {
            "workspace_tools": gateway.accounting(),
            "llm": {
                "calls": int(getattr(backend, "calls", 0)),
                "prompt_tokens": int(getattr(backend, "prompt_tokens", 0)),
                "completion_tokens": int(getattr(backend, "completion_tokens", 0)),
                "returned_models": sorted(getattr(backend, "returned_models", ())),
            "transport_retries": int(
                getattr(getattr(backend, "delegate", None),
                        "transport_retries", 0)
            ),
            },
            "target_support": {
                "real_support_probe_count": real_probe_count,
                "charged_probe_cost": charged_probe_cost,
                "charged_is_a_budget_penalty_not_a_probe_count": True,
            },
        },
        "metrics": {
            "real_support_probe_count": real_probe_count,
            "charged_probe_cost": charged_probe_cost,
            "workspace_tool_calls": gateway.calls,
            "llm_calls": int(getattr(backend, "calls", 0)),
            "harmful_probe_count": sum(
                1 for row in probed_rows
                if float(row["support_gain"]) < -MATERIAL_THRESHOLD
            ),
            "cumulative_support_harm": float(
                sum(
                    -float(row["support_gain"]) for row in probed_rows
                    if float(row["support_gain"]) < -MATERIAL_THRESHOLD
                )
            ),
            "task_local_active": int(local_active),
            "task_delayed_utility": (
                winner.delayed_response.get("gain")
                if winner is not None
                and winner.delayed_response.get("evaluated")
                else None
            ),
            "abstention": int(
                stop_reason in {STOP_ABSTAIN, STOP_REQUEST_OBSERVATION}
            ),
            "instrument_unreadable": int(instrument_unreadable),
            # Infrastructure, not the Agent and not the Judge.  Excluded from
            # every behavioural readout: it is not an abstention, not a
            # decision, and never evidence for a Harness edit.
            "infrastructure_failed": int(trace.infrastructure_failed),
            "distinct_series_observed": gateway.accounting()[
                "distinct_series_observed"
            ],
        },
    }


# ---------------------------------------------------- deterministic attribution
def attribute_first_fault(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Name the run's first fault from the trace, with no LLM in the loop.

    §10.2 order.  Mechanical exits first: an evaluator that could not measure,
    then a protocol slip.  Both leave through deterministic exits and are never
    handed to Slow.  Only then does a *method* question arise, and §10.1 is
    explicit that a negative or conflict Experience is enough to trigger
    diagnosis -- other Tasks succeeding does not retire it.  A single bad
    result still authorizes nothing on its own; that is the Slow stage's own
    per-clause evidence rule, not this router's job.

    The public cause vocabulary stays the frozen four: CONTEXT_GAP,
    WORKFLOW_GAP, DECISION_GAP, NO_ACTIONABLE_EVIDENCE.
    """
    def arms(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [row[arm] for arm in (COLD_ARM, WARM_ARM) if arm in row]

    all_arms = [arm_row for row in rows for arm_row in arms(row)]
    tool_calls = sum(
        int(r["metrics"].get("workspace_tool_calls", 0)) for r in all_arms
    )

    unreadable = [
        row["task_episode_id"] for row in rows
        for arm_row in arms(row)
        if int(arm_row["metrics"].get("instrument_unreadable", 0))
    ]
    if unreadable:
        return {
            "first_fault": "INSTRUMENT_UNREADABLE",
            "cause": "MECHANICAL_EXIT",
            "layer": "MECHANICAL",
            "editable": False,
            "task_ids": sorted(set(unreadable)),
            "note": (
                "A Task whose Support or delayed evaluator could not measure "
                "the candidate is excluded, never recorded as a tie, and never "
                "handed to Slow."
            ),
            "workspace_tool_calls": tool_calls,
        }
    infrastructure = [
        {"task_episode_id": row["task_episode_id"], "arm": arm_row["arm"],
         "stop_reason": arm_row["stop_reason"],
         "error": arm_row.get("infrastructure_error")}
        for row in rows for arm_row in arms(row)
        if int(arm_row["metrics"].get("infrastructure_failed", 0))
    ]
    usable = [
        arm_row for row in rows for arm_row in arms(row)
        if not int(arm_row["metrics"].get("infrastructure_failed", 0))
    ]
    if infrastructure and not usable:
        return {
            "first_fault": "INFRASTRUCTURE_FAILED",
            "cause": "MECHANICAL_EXIT",
            "layer": "INFRASTRUCTURE",
            "editable": False,
            "occurrences": infrastructure,
            "note": (
                "Every arm-Task ended on the relay, not on a decision. There "
                "is no behaviour in this run to attribute."
            ),
            "workspace_tool_calls": tool_calls,
        }
    protocol = [
        {"task_episode_id": row["task_episode_id"], "arm": arm_row["arm"],
         "error": arm_row["protocol_error"]}
        for row in rows for arm_row in arms(row)
        if arm_row.get("protocol_error")
    ]
    if protocol:
        return {
            "first_fault": "AGENT_PROTOCOL_ERROR",
            "cause": "MECHANICAL_EXIT",
            "layer": "MECHANICAL",
            "editable": False,
            "occurrences": protocol,
            "workspace_tool_calls": tool_calls,
        }

    # ---- method layer -----------------------------------------------------
    negative_probes = [
        {"task_episode_id": row["task_episode_id"], "arm": arm_row["arm"],
         "program": [str(step["op"]) for step in probe["steps"]],
         "support_gain": probe["support_gain"]}
        for row in rows for arm_row in arms(row)
        for probe in arm_row["probes"]
        if probe.get("status") == "PROBED"
        and not probe.get("meets_material_threshold")
    ]
    barren_arms = [
        {"task_episode_id": row["task_episode_id"], "arm": arm_row["arm"],
         "stop_reason": arm_row["stop_reason"],
         "real_support_probe_count": arm_row["metrics"][
             "real_support_probe_count"],
         "charged_probe_cost": arm_row["metrics"]["charged_probe_cost"]}
        for row in rows for arm_row in arms(row)
        if not int(arm_row["metrics"]["task_local_active"])
        and not int(arm_row["metrics"].get("infrastructure_failed", 0))
    ]
    requested_observation = [
        {"task_episode_id": row["task_episode_id"], "arm": arm_row["arm"]}
        for row in rows for arm_row in arms(row)
        if arm_row["stop_reason"] == STOP_REQUEST_OBSERVATION
    ]
    # An abstention that followed an envelope retry and produced no proposal
    # is protocol degradation.  It is reported, but it is not evidence of a
    # decision fault and must not be what routes the run to a Surface edit.
    degraded = {
        (entry["task_episode_id"], entry["arm"])
        for entry in protocol_quality(rows)[
            "abstentions_that_followed_an_envelope_retry_with_no_proposal"
        ]
    }
    barren_arms = [
        entry for entry in barren_arms
        if (entry["task_episode_id"], entry["arm"]) not in degraded
    ]
    no_candidate = [
        {"task_episode_id": row["task_episode_id"], "arm": arm_row["arm"]}
        for row in rows for arm_row in arms(row)
        if not arm_row["proposals"]
    ]
    compiled_any = any(
        entry.get("status") == "COMPILED"
        for r in all_arms for entry in r["compiled"]
    )
    wasted_probes = sum(
        row["real_support_probe_count"] for row in barren_arms
    )

    requested_observation = [
        entry for entry in requested_observation
        if (entry["task_episode_id"], entry["arm"]) not in {
            (row["task_episode_id"], row["arm"]) for row in infrastructure
        }
    ]
    if not (negative_probes or barren_arms or requested_observation):
        return {
            "first_fault": "NONE_BLOCKING",
            "cause": "NO_ACTIONABLE_EVIDENCE",
            "layer": "METHOD",
            "editable": False,
            "note": (
                "Every arm-Task reached a Target-local ACTIVE Skill and no "
                "probe fell below the material threshold, so there is no "
                "negative or conflict Experience to diagnose."
            ),
            "workspace_tool_calls": tool_calls,
        }

    if requested_observation and not compiled_any:
        cause, first_fault = "CONTEXT_GAP", "REQUESTED_OBSERVATION_ABSENT"
    elif no_candidate and not compiled_any:
        cause, first_fault = "WORKFLOW_GAP", "NO_COMPILABLE_PROGRAM"
    else:
        cause, first_fault = "DECISION_GAP", "SUPPORT_SPENT_WITHOUT_A_DRAFT"

    route = route_program_supply_fault(
        expressibility_status=(
            "PROVEN_EXPRESSIBLE" if compiled_any else "EXPRESSIBILITY_UNKNOWN"
        ),
        expressibility_cause="",
        capability_skill_exists=True,
        skill_retrieved=True,
        constrained_proposal_succeeds=compiled_any,
        context_resolved_decision_fault=bool(no_candidate) and compiled_any,
    )
    return {
        "first_fault": first_fault,
        "cause": cause,
        "layer": "METHOD",
        "repair_scope": "GENERAL" if cause == "DECISION_GAP" else "SPECIFIC",
        "editable": route[1] == "EDITABLE_M0",
        "program_supply_route": {"fault_family": route[0], "actionability": route[1],
                                 "surfaces": list(route[2])},
        "negative_probe_count": len(negative_probes),
        "negative_probes": negative_probes,
        "arm_tasks_without_a_local_active_skill": barren_arms,
        "support_probes_spent_without_a_draft": wasted_probes,
        "arm_tasks_requesting_an_absent_observation": requested_observation,
        "arm_tasks_with_no_proposal": no_candidate,
        "arm_tasks_excluded_as_protocol_degradation": sorted(degraded),
        "arm_tasks_excluded_as_infrastructure_failure": infrastructure,
        "workspace_tool_calls": tool_calls,
        "note": (
            "One bad result is enough to open the diagnosis and is not enough "
            "to authorize a General clause; the per-clause evidence threshold "
            "is applied at the Slow stage, on distinct Task counts."
        ),
    }


def protocol_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How much of the run was the model failing to format, not deciding.

    The stage loop already retries one malformed envelope with static
    feedback, and a retry that lands on a schema-valid but *empty* payload
    ends the Task in ABSTAIN.  That is protocol degradation wearing an
    Agent decision's clothes, and the frozen reporting rule is that a
    mechanical failure must never be read as behaviour.  So it is counted
    here and named, rather than being silently folded into the abstention
    rate.  It is a readout, not a Gate: the affected Task stays in the run
    and stays visible.
    """
    stage_count = 0
    retried_stages: list[dict[str, Any]] = []
    suspect: list[dict[str, Any]] = []
    for row in rows:
        for arm in (COLD_ARM, WARM_ARM):
            if arm not in row:
                continue
            arm_row = row[arm]
            for stage in arm_row["stages"]:
                stage_count += 1
                if int(stage.get("validation_retry_count") or 0) <= 0:
                    continue
                retried_stages.append({
                    "task_episode_id": row["task_episode_id"],
                    "arm": arm,
                    "stage": stage["stage"],
                    "validation_error_codes": list(
                        stage.get("validation_error_codes") or ()
                    ),
                })
            # Precision correction (2026-08-19): the first version flagged an
            # arm-Task whenever *any* stage had retried and the proposal came
            # back empty.  That over-counts -- an inspect-stage retry that
            # recovered says nothing about why propose returned nothing, and
            # several such arms went on to pass the draft gate.  The
            # defensible claim is narrower: the propose stage itself failed
            # validation and then returned an empty candidate list, i.e. the
            # Agent gave up on a validator rather than on the Task.
            propose_retry = next(
                (
                    stage for stage in arm_row["stages"]
                    if stage["stage"] == "propose"
                    and int(stage.get("validation_retry_count") or 0) > 0
                ),
                None,
            )
            if (
                propose_retry is not None
                and not arm_row["proposals"]
                and arm_row["stop_reason"] == STOP_ABSTAIN
            ):
                suspect.append({
                    "task_episode_id": row["task_episode_id"],
                    "arm": arm,
                    "stop_reason": arm_row["stop_reason"],
                    "propose_validation_error_codes": list(
                        propose_retry.get("validation_error_codes") or ()
                    ),
                })

    return {
        "stage_call_count": stage_count,
        "stages_needing_an_envelope_retry": len(retried_stages),
        "envelope_retry_rate": (
            len(retried_stages) / stage_count if stage_count else None
        ),
        "retried_stages": retried_stages,
        "abstentions_that_followed_an_envelope_retry_with_no_proposal": suspect,
        "role": (
            "reporting integrity readout; an abstention listed here is "
            "protocol degradation, not a considered decision, and must not be "
            "counted as one. Never a Gate."
        ),
    }


# ------------------------------------------------- Slow single-surface edit
def _census_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """This run's own Action-Response facts, in the census input shape.

    One entry per probed candidate per arm.  ``_program_evidence_census`` then
    de-duplicates to distinct Task Episodes, which is the unit of evidence; the
    per-arm attempts stay visible only as the diagnostic attempt_count.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("skipped"):
            continue
        condition = bool(row.get(g1.G1_CONDITION_FEATURE, False))
        for arm in (COLD_ARM, WARM_ARM):
            if arm not in row:
                continue
            for probe in row[arm]["probes"]:
                if probe.get("status") != "PROBED":
                    continue
                out.append({
                    "task_episode_id": row["task_episode_id"],
                    "arm": arm,
                    "program": [str(step["op"]) for step in probe["steps"]],
                    "support_gain": float(probe["support_gain"]),
                    "gain_readable": True,
                    g1.G1_CONDITION_FEATURE: condition,
                })
    return out


def run_slow_and_replay(
    *,
    repo_root: Path,
    rows: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
    config: Mapping[str, Any],
    specs: Sequence[Mapping[str, Any]],
    state_rel: str,
    workspace_tool_budget: int,
    backend_factory: Callable[[int], Any],
    first_fault: Mapping[str, Any],
) -> dict[str, Any]:
    """One Slow edit on one authorized Surface, then a same-session replay.

    §10.4: Slow never approves itself.  The deterministic compiler validates
    the edit, and a paired replay run *in this same session* decides whether it
    survives.  Historical numbers are never used as the comparison arm.
    """
    census_input = _census_rows(rows)
    census = g1._program_evidence_census(census_input)
    supported = [
        cell for cell in census
        if int(cell["distinct_task_count"])
        >= g1.GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
    ]
    trigger = {
        "first_fault": first_fault.get("first_fault"),
        "editable_surface": bool(first_fault.get("editable")),
        "census_cell_count": len(census),
        "cells_meeting_general_threshold": len(supported),
    }
    if not first_fault.get("editable"):
        return {
            "verdict": "G1_SLOW_NO_ACTIONABLE",
            "trigger": trigger,
            "evidence_census": census,
            "note": (
                "Attribution named no editable Harness surface, so the Slow "
                "path is not entered. An explicit NO_ACTIONABLE is the correct "
                "outcome, not a reason to widen the repair scope."
            ),
            "llm_api_call_count": 0,
        }

    attribution = {
        "evidence_census": census,
        "evidence_census_contract": {
            "unit_of_evidence": "distinct_task_count",
            "attempt_count_role": "diagnostic_only",
            "no_relation_filter": True,
            "no_program_filter": True,
            "source": (
                "this run's own Fast Path probes; both arms of the same Task "
                "Episode share one frozen Outcome cell, so attempts double "
                "count and only distinct Tasks are evidence"
            ),
        },
    }
    llm_counter = [0]
    patch = g1.run_g1_guidance_patch(
        repo_root, attribution, llm_counter,
        store_root=repo_root / state_rel / "slow" / "snapshots",
    )
    if patch.get("verdict") != "G1_GUIDANCE_PATCH_APPLIED":
        return {
            "verdict": "G1_SLOW_PATCH_NOT_APPLIED",
            "trigger": trigger,
            "evidence_census": census,
            "patch": patch,
            "llm_api_call_count": llm_counter[0],
        }

    # ---- paired same-session replay -------------------------------------
    base_snapshot = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    patched_store = SnapshotStore(repo_root / state_rel / "slow" / "snapshots")
    patched_snapshot = compile_snapshot(
        patched_store.root / str(patch["patched_runtime_bundle_sha"]),
        verify_lock=False,
    )
    replay_rows: list[dict[str, Any]] = []
    arm_snapshots = {"BASE": base_snapshot, "PATCHED": patched_snapshot}
    replay_states: dict[str, _ArmState] = {}
    for label, snapshot in arm_snapshots.items():
        store = SnapshotStore(state_rel_path(repo_root, state_rel, label))
        store.materialize(snapshot)
        store.set_active(snapshot.runtime_bundle_sha)
        replay_states[label] = _ArmState(
            arm=label, memories=[], episodes=[], store=store,
            active_snapshot=snapshot,
            active_skill_ids=_skill_ids(snapshot, local_only=True),
        )
    for spec in specs:
        task_id = str(spec["task_episode_id"])
        context = g1._w3_context_for(
            repo_root, state_rel, task_id, int(spec["support_origins"][0]),
            cohort["values"], cohort["train_uids"],
        )
        if not (context.get("scope_series_uids") or ()):
            continue
        inventory = _inventory_rows(context)
        # AB/BA, not AB always.  The first replay ran BASE then PATCHED on
        # every Task, so anything that depends on being second in the pair --
        # a relay that degrades within a pair, a model that formats worse on
        # the later call -- lands entirely on PATCHED.  It did: across two
        # cohorts the PATCHED arm hit AGENT_PROTOCOL_ERROR five times out of
        # eighteen and the BASE arm zero.  Order is alternated by the Task's
        # own frozen arm_order so a systematic second-position effect cannot
        # be read as a patch effect.
        labels = (
            ("BASE", "PATCHED") if spec["arm_order"] == "A3_A5"
            else ("PATCHED", "BASE")
        )
        entry: dict[str, Any] = {
            "task_episode_id": task_id,
            "replay_arm_order": list(labels),
        }
        for label in labels:
            arm_row = _run_arm(
                repo_root=repo_root,
                arm_state=replay_states[label],
                task_spec=spec,
                public_context=context,
                cohort=cohort,
                config=config,
                inventory=inventory,
                source_prior=None,
                workspace_tool_budget=workspace_tool_budget,
                backend_factory=backend_factory,
            )
            entry[label] = {
                "stop_reason": arm_row["stop_reason"],
                "protocol_error": arm_row.get("protocol_error"),
                "protocol_error_output": arm_row.get("protocol_error_output"),
                "infrastructure_error": arm_row.get("infrastructure_error"),
                "mechanical_exit": bool(
                    arm_row.get("protocol_error")
                    or int(arm_row["metrics"].get("infrastructure_failed", 0))
                    or int(arm_row["metrics"].get("instrument_unreadable", 0))
                ),
                "proposed_operator_structures": [
                    [str(step["op"]) for step in candidate.get("steps") or ()]
                    for candidate in arm_row["proposals"]
                ],
                "first_probe_program": next(
                    (
                        [str(step["op"]) for step in probe["steps"]]
                        for probe in arm_row["probes"]
                        if probe.get("status") == "PROBED"
                    ),
                    None,
                ),
                "guidance_reaching_the_agent": (
                    arm_row["retrieved_knowledge_summary"]
                ),
                "metrics": arm_row["metrics"],
            }
            print(
                "G1_REPLAY %s %s stop=%s first=%s"
                % (task_id, label, entry[label]["stop_reason"],
                   entry[label]["first_probe_program"]),
                flush=True,
            )
        # A pair is comparable only when both sides produced behaviour.  An
        # arm that left through a mechanical exit -- a protocol fault, the
        # relay, an unreadable Judge -- did not decide anything, and reading
        # "it abstained, now it errors" as "the patch changed behaviour" would
        # let a formatting failure certify a Harness edit.  The first run of
        # this replay did exactly that on three of nine Tasks.
        entry["comparable"] = not (
            entry["BASE"]["mechanical_exit"]
            or entry["PATCHED"]["mechanical_exit"]
        )
        entry["proposal_changed"] = entry["comparable"] and (
            entry["BASE"]["proposed_operator_structures"]
            != entry["PATCHED"]["proposed_operator_structures"]
        )
        entry["first_probe_changed"] = entry["comparable"] and (
            entry["BASE"]["first_probe_program"]
            != entry["PATCHED"]["first_probe_program"]
        )
        entry["decision_changed"] = entry["comparable"] and (
            entry["BASE"]["stop_reason"] != entry["PATCHED"]["stop_reason"]
        )
        replay_rows.append(entry)

    comparable = [row for row in replay_rows if row["comparable"]]
    changed = [
        row["task_episode_id"] for row in comparable
        if row["proposal_changed"] or row["first_probe_changed"]
        or row["decision_changed"]
    ]
    excluded = [
        {
            "task_episode_id": row["task_episode_id"],
            "arms": [
                label for label in ("BASE", "PATCHED")
                if row[label]["mechanical_exit"]
            ],
        }
        for row in replay_rows if not row["comparable"]
    ]
    return {
        "verdict": (
            "G1_SLOW_PATCH_CHANGES_NEXT_ROUND_BEHAVIOUR" if changed
            else "G1_SLOW_PATCH_APPLIED_BUT_BEHAVIOUR_UNCHANGED"
        ),
        "trigger": trigger,
        "evidence_census": census,
        "patch": {
            key: value for key, value in patch.items()
            if key not in {"slow_payload"}
        },
        "replay_protocol": (
            "paired, same session, same Tasks, same cohort, same tool and "
            "Support budgets; the only difference between the two arms is the "
            "candidate_policy.proposal_guidance surface"
        ),
        "replay_rows": replay_rows,
        "comparable_task_count": len(comparable),
        "tasks_excluded_from_comparison": excluded,
        "tasks_with_changed_behaviour": changed,
        "llm_api_call_count": llm_counter[0],
    }


def state_rel_path(repo_root: Path, state_rel: str, label: str) -> Path:
    return repo_root / state_rel / "replay" / label / "snapshots"


# ------------------------------------------------------------------- driver
def run_g1_pipeline(
    *,
    cohort_name: str = "T233",
    task_count: int = 3,
    workspace_tool_budget: int = WORKSPACE_TOOL_BUDGET,
    state_rel: str = STATE_REL,
    report_path: Path = REPORT_REL,
    backend_factory: Callable[[int], Any] | None = None,
    write_report: bool = True,
    run_slow: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = PROJECT_ROOT
    backend_factory = backend_factory or _default_backend_factory
    from run_v1_kdd2018_natural_slow_update import _config

    config = dict(_config())
    cohort = load_cohort(repo_root, cohort_name)
    specs = list(_frozen_task_roster()[:task_count])

    # ---- substrate preflight, before any Outcome opens -------------------
    eval_pre = g1.eval_substrate_preflight(
        cohort["values"], cohort["eval_uids"], specs
    )
    train_pre = g1.train_substrate_preflight(
        cohort["values"], cohort["train_uids"],
        [int(a) for a in config["anchors"]],
    )
    if not (eval_pre["pass"] and train_pre["pass"]):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "verdict": "G1_SUBSTRATE_INVALID",
            "cohort": cohort_name,
            "eval_substrate_preflight": eval_pre,
            "train_substrate_preflight": train_pre,
            "wall_seconds": time.perf_counter() - started,
        }

    state_root = repo_root / state_rel
    base_snapshot = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    arm_states: dict[str, _ArmState] = {}
    for arm in (COLD_ARM, WARM_ARM):
        store = SnapshotStore(state_root / arm / "snapshots")
        store.materialize(base_snapshot)
        store.set_active(base_snapshot.runtime_bundle_sha)
        arm_states[arm] = _ArmState(
            arm=arm, memories=[], episodes=[], store=store,
            active_snapshot=base_snapshot,
            active_skill_ids=_skill_ids(base_snapshot, local_only=True),
        )

    # A5's warm start: the frozen Source Card and Source evidence, offered
    # through the Runtime Scope matcher rather than assumed to apply.
    source_prior: dict[str, Any] | None = None
    try:
        legacy_report = json.loads(
            (repo_root / "artifacts/functional/e2"
             / "w1_task_episode_harness_report.json").read_text(encoding="utf-8")
        )
        source_prior = {
            "source_card": _source_card_from_report(legacy_report),
            "source_evidence": _source_bundle_from_report(legacy_report),
        }
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        source_prior = None

    rows: list[dict[str, Any]] = []
    for spec in specs:
        task_id = str(spec["task_episode_id"])
        cutoff = int(spec["support_origins"][0])
        context = g1._w3_context_for(
            repo_root, state_rel, task_id, cutoff,
            cohort["values"], cohort["train_uids"],
        )
        scope = list(context.get("scope_series_uids") or ())
        if not scope:
            rows.append({"task_episode_id": task_id, "skipped": "EMPTY_SCOPE"})
            print("G1_TASK_SKIP %s empty scope" % task_id, flush=True)
            continue
        inventory = _inventory_rows(context)
        matched_prior = (
            _source_prior_for_task(source_prior, context)
            if source_prior is not None else None
        )
        order = [(COLD_ARM, None), (WARM_ARM, matched_prior)]
        if spec["arm_order"] == "A5_A3":
            order = list(reversed(order))
        print(
            "G1_TASK_START %s cohort=%s scope=%d source_matched=%s"
            % (task_id, cohort_name, len(scope), matched_prior is not None),
            flush=True,
        )
        row: dict[str, Any] = {
            "task_episode_id": task_id,
            "arm_order": spec["arm_order"],
            "observation_cutoff": cutoff,
            "support_origins": list(spec["support_origins"]),
            "delayed_origins": list(spec["delayed_origins"]),
            "scope_series_uids": scope,
            "task_signature": dict(context["task_signature"]),
            g1.G1_CONDITION_FEATURE: bool(
                (context.get("task_fast_features") or {}).get(
                    g1.G1_CONDITION_FEATURE, False
                )
            ),
            "source_prior_retrieval": {
                "runtime_matcher": "evaluate_applicability",
                "matched": matched_prior is not None,
            },
        }
        for arm, prior in order:
            row[arm] = _run_arm(
                repo_root=repo_root,
                arm_state=arm_states[arm],
                task_spec=spec,
                public_context=context,
                cohort=cohort,
                config=config,
                inventory=inventory,
                source_prior=prior,
                workspace_tool_budget=workspace_tool_budget,
                backend_factory=backend_factory,
            )
            print(
                "G1_ARM_DONE %s %s stop=%s tools=%d llm=%d probes=%d active=%d"
                % (task_id, arm, row[arm]["stop_reason"],
                   row[arm]["metrics"]["workspace_tool_calls"],
                   row[arm]["metrics"]["llm_calls"],
                   row[arm]["metrics"]["real_support_probe_count"],
                   row[arm]["metrics"]["task_local_active"]),
                flush=True,
            )
        rows.append(row)
        # Checkpoint after every Task.  The first live nine-Task run died at
        # Task six and took every completed Task with it, because the report
        # was only written at the end.  Support probes are the one
        # non-renewable cost in this Pipeline; losing their record to an
        # unrelated crash is not acceptable.
        if write_report:
            _write_partial(report_path, {
                "protocol_version": PROTOCOL_VERSION,
                "verdict": "G1_PIPELINE_IN_PROGRESS",
                "cohort": cohort_name,
                "development_replay": True,
                "tasks_completed": len(rows),
                "tasks_planned": len(specs),
                "eval_substrate_preflight": eval_pre,
                "train_substrate_preflight": train_pre,
                "rows": rows,
            })

    scored = [row for row in rows if COLD_ARM in row or WARM_ARM in row]
    executable_names = tuple(
        str(entry["name"]) for entry in _inventory_rows(
            g1._w3_context_for(
                repo_root, state_rel, str(specs[0]["task_episode_id"]),
                int(specs[0]["support_origins"][0]),
                cohort["values"], cohort["train_uids"],
            )
        ) if str(entry.get("availability")) == "EXECUTABLE"
    )
    concentration = {}
    for arm in (COLD_ARM, WARM_ARM):
        programs = [
            [(step["op"], step["params"]) for step in entry["steps"]]
            for row in scored if arm in row
            for entry in row[arm]["probes"]
            if entry.get("status") == "PROBED"
        ]
        concentration[arm] = exploration_concentration(
            programs, executable_operator_names=executable_names
        )

    cost = {}
    for arm in (COLD_ARM, WARM_ARM):
        arm_rows = [row[arm] for row in scored if arm in row]
        cost[arm] = {
            "workspace_tool_calls": sum(
                r["metrics"]["workspace_tool_calls"] for r in arm_rows
            ),
            "llm_calls": sum(r["metrics"]["llm_calls"] for r in arm_rows),
            "llm_prompt_tokens": sum(r["cost"]["llm"]["prompt_tokens"] for r in arm_rows),
            "llm_completion_tokens": sum(
                r["cost"]["llm"]["completion_tokens"] for r in arm_rows
            ),
            "real_support_probe_count": sum(
                r["metrics"]["real_support_probe_count"] for r in arm_rows
            ),
            "charged_probe_cost": sum(
                r["metrics"]["charged_probe_cost"] for r in arm_rows
            ),
            "task_local_active_count": sum(
                r["metrics"]["task_local_active"] for r in arm_rows
            ),
            "abstention_count": sum(r["metrics"]["abstention"] for r in arm_rows),
            "instrument_unreadable_count": sum(
                r["metrics"]["instrument_unreadable"] for r in arm_rows
            ),
            "infrastructure_failed_count": sum(
                r["metrics"].get("infrastructure_failed", 0) for r in arm_rows
            ),
            "transport_retries": sum(
                r["cost"]["llm"].get("transport_retries", 0) for r in arm_rows
            ),
        }

    attribution = attribute_first_fault(scored)
    slow = None
    if run_slow:
        slow = run_slow_and_replay(
            repo_root=repo_root,
            rows=scored,
            cohort=cohort,
            config=config,
            specs=specs,
            state_rel=state_rel,
            workspace_tool_budget=workspace_tool_budget,
            backend_factory=backend_factory,
            first_fault=attribution,
        )
    closure = _closure_criteria(scored, slow=slow)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": (
            "G1_PIPELINE_CLOSURE_COMPLETE" if closure["all_pass"]
            else "G1_PIPELINE_CLOSURE_INCOMPLETE"
        ),
        "cohort": cohort_name,
        "exposure": cohort["exposure"],
        "development_replay": True,
        "task_count": len(scored),
        "arms": {
            COLD_ARM: "cold start: no Source prior, base Harness guidance",
            WARM_ARM: (
                "warm start: Source Card and Source evidence offered through "
                "the Runtime Scope matcher"
            ),
        },
        "shared_between_arms": [
            "Workspace tools", "tool-call bound", "operator inventory",
            "Target Support budget", "Runtime", "Judge", "stage contracts",
        ],
        "eval_substrate_preflight": eval_pre,
        "train_substrate_preflight": train_pre,
        "closure_criteria": closure,
        "first_fault": attribution,
        "slow_and_replay": slow,
        "exploration_concentration": concentration,
        "protocol_quality": protocol_quality(scored),
        "cost_by_arm": cost,
        "cost_columns_are_separate": (
            "Workspace tool calls, LLM calls, real Support probes and charged "
            "probe cost are four separate columns; charged cost is a budget "
            "penalty, never a probe count"
        ),
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    if write_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
    return result


def _write_partial(report_path: Path, payload: Mapping[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _closure_criteria(
    rows: Sequence[Mapping[str, Any]],
    *,
    slow: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The nine §13/G1 completion criteria, each read off the trace."""
    def arm_rows() -> list[Mapping[str, Any]]:
        return [
            row[arm] for row in rows for arm in (COLD_ARM, WARM_ARM)
            if arm in row
        ]

    all_arms = arm_rows()
    tool_calls = sum(
        int(r["metrics"]["workspace_tool_calls"]) for r in all_arms
    )
    # §13/G1 accepts either a same-input contemporaneous contrast or a
    # deterministic reference Trace.  A bare feature citation is neither: it
    # shows the Agent read something, not that a candidate depended on it.
    # The chain below is the deterministic one and it is machine-checked end
    # to end -- a proposed candidate names a hypothesis, that hypothesis cites
    # public feature names, and those names exist only because a tool call
    # returned them.  The contemporaneous contrast is kept alongside as the
    # weaker corroborating readout, never as the criterion by itself.
    reference_chains: list[dict[str, Any]] = []
    for row in rows:
        for arm in (COLD_ARM, WARM_ARM):
            if arm not in row:
                continue
            arm_row = row[arm]
            served: set[str] = set()
            for observation in arm_row["tool_observations"]:
                if not observation.get("ok"):
                    continue
                result = observation.get("public_result") or {}
                features = result.get("features")
                if isinstance(features, Mapping):
                    served.update(str(key) for key in features)
                served.update(
                    str(key) for key in result if key.startswith("estimated_")
                )
            inspect = next(
                (s for s in arm_row["stages"] if s["stage"] == "inspect"), None
            )
            hypotheses = {
                str(h.get("hypothesis_id")): [
                    str(f) for f in (h.get("evidence_features") or ())
                ]
                for h in ((inspect or {}).get("payload") or {}).get(
                    "pattern_hypotheses"
                ) or ()
                if isinstance(h, Mapping) and h.get("hypothesis_id")
            }
            for candidate in arm_row["proposals"]:
                hypothesis_id = candidate.get("addresses_hypothesis_id")
                if not hypothesis_id or hypothesis_id not in hypotheses:
                    continue
                grounded = [f for f in hypotheses[hypothesis_id] if f in served]
                if not grounded:
                    continue
                reference_chains.append({
                    "task_episode_id": row["task_episode_id"],
                    "arm": arm,
                    "observed_series": sorted({
                        str(o["arguments"].get("series_uid"))
                        for o in arm_row["tool_observations"] if o.get("ok")
                    }),
                    "hypothesis_id": hypothesis_id,
                    "features_served_by_a_tool_call": grounded,
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate_operators": [
                        str(step["op"]) for step in candidate.get("steps") or ()
                    ],
                })

    tool_effect: list[dict[str, Any]] = []
    for row in rows:
        if COLD_ARM not in row or WARM_ARM not in row:
            continue
        observed = {
            arm: sorted({
                str(obs["arguments"].get("series_uid"))
                for obs in row[arm]["tool_observations"] if obs.get("ok")
            })
            for arm in (COLD_ARM, WARM_ARM)
        }
        proposed = {
            arm: [
                [str(step["op"]) for step in entry.get("steps") or ()]
                for entry in row[arm]["proposals"]
            ]
            for arm in (COLD_ARM, WARM_ARM)
        }
        if observed[COLD_ARM] != observed[WARM_ARM] or (
            proposed[COLD_ARM] != proposed[WARM_ARM]
        ):
            tool_effect.append({
                "task_episode_id": row["task_episode_id"],
                "observed_series_by_arm": observed,
                "proposed_operator_structures_by_arm": proposed,
            })

    excluded = [
        {"arm": r["arm"], "stop_reason": r["stop_reason"]}
        for r in all_arms
        if int(r["metrics"].get("instrument_unreadable", 0))
        or int(r["metrics"].get("infrastructure_failed", 0))
    ]
    # An excluded arm-Task is not behaviour, so it cannot be the negative
    # Experience that satisfies the criterion below.
    negative_arms = [
        {"arm": r["arm"], "stop_reason": r["stop_reason"]}
        for r in all_arms
        if r["stop_reason"] in {STOP_ABSTAIN, STOP_NO_DRAFT,
                                STOP_REQUEST_OBSERVATION}
        and not int(r["metrics"].get("instrument_unreadable", 0))
        and not int(r["metrics"].get("infrastructure_failed", 0))
    ]
    criteria = {
        "one_command_runs_the_whole_loop": True,
        "agent_called_a_workspace_tool": tool_calls > 0,
        "tool_result_changed_a_later_decision": bool(reference_chains),
        "runtime_generated_and_executed_a_typed_workflow": any(
            entry.get("status") == "PROBED" for r in all_arms
            for entry in r["probes"]
        ),
        "episode_written": any(
            entry.get("episode_id") for r in all_arms for entry in r["probes"]
        ),
        "positive_result_forms_or_reuses_a_target_local_skill": any(
            int(r["metrics"]["task_local_active"]) for r in all_arms
        ),
        # Three-valued on purpose.  When no arm-Task ended negatively there is
        # nothing to route, and calling that "satisfied" would be claiming a
        # check that never ran.  It stays un-passed until a real negative
        # appears, rather than being weakened so a scripted run can close.
        "negative_or_conflict_reaches_slow_or_explicit_no_actionable": (
            "NOT_EXERCISED" if not negative_arms
            else True if slow is not None
            and str(slow.get("verdict", "")).startswith("G1_SLOW")
            else "NOT_EXERCISED"
        ),
        # False means the real failure: a patch was applied and the replay
        # showed it changed nothing.  A Slow stage that declined to write a
        # clause it could not support is the evidence rule working, so it
        # leaves the criterion untested rather than failed.
        "slow_patch_changes_next_round_behaviour": (
            "NOT_EXERCISED"
            if slow is None or slow.get("verdict") in {
                "G1_SLOW_NO_ACTIONABLE", "G1_SLOW_PATCH_NOT_APPLIED",
            }
            else slow.get("verdict")
            == "G1_SLOW_PATCH_CHANGES_NEXT_ROUND_BEHAVIOUR"
        ),
        # Every arm-Task carries both flags, and a flagged Task is excluded
        # from the behavioural readouts rather than counted as a tie.  An
        # unreadable Judge and a failed relay are different exits and are
        # recorded as different exits.
        "instrument_failures_excluded_not_tied": all(
            "instrument_unreadable" in r["metrics"]
            and "infrastructure_failed" in r["metrics"]
            for r in all_arms
        ),
        "real_and_charged_cost_reported_separately": all(
            "real_support_probe_count" in r["metrics"]
            and "charged_probe_cost" in r["metrics"]
            for r in all_arms
        ),
    }
    return {
        **criteria,
        "all_pass": all(value is True for value in criteria.values()),
        "unexercised_criteria": sorted(
            name for name, value in criteria.items() if value == "NOT_EXERCISED"
        ),
        "negative_arm_tasks": negative_arms,
        "excluded_arm_tasks": excluded,
        "deterministic_reference_chains": reference_chains,
        "contemporaneous_contrast": tool_effect,
        "workspace_tool_call_total": tool_calls,
        "slow_verdict": (slow or {}).get("verdict"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="T233",
                        choices=("e31", "T233", "weather"))
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--tool-budget", type=int, default=WORKSPACE_TOOL_BUDGET)
    parser.add_argument("--no-slow", action="store_true",
                        help="stop after the Fast Path closure")
    args = parser.parse_args(argv)
    result = run_g1_pipeline(
        cohort_name=args.cohort,
        task_count=args.tasks,
        workspace_tool_budget=args.tool_budget,
        run_slow=not args.no_slow,
    )
    print(json.dumps(
        {key: value for key, value in result.items() if key != "rows"},
        indent=2, ensure_ascii=False, default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
