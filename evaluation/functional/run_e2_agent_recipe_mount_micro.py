"""agent-recipe-mount micro: does the Fast Agent route to the batch recipe,
and does a prior Experience entry cut the cost of adapting the same batch again?

The batch recipe (``run_batch_composition_headroom.make_batch_recipe`` with
``adoption_rule_version="v2"``) already produces a delayed-non-negative adopted
plan on all six cohort x Consumer cells.  It is an offline script: the Agent
does not know it exists.  This runner mounts it as a Workspace tool the Fast
Agent may call, and then runs a three-episode single-arm micro that measures
three behaviours:

* **E1** traffic x pooled, first sight -- does the Agent call the tool at all,
  and does it adopt what came back?
* **E2** traffic x pooled again, same batch, same Consumer -- does it reuse the
  Experience entry E1 wrote instead of paying for the search a second time?
* **E3** traffic x per_channel, same data, different Consumer structure -- does
  it notice the cell changed and search again, rather than reusing E1's plan?

The single Harness change surface is the Fast Agent's Workspace tool supply:
``CohortScopePublicToolGateway`` gained an optional ``batch_recipe_binding``.
With no binding -- every existing run -- the class is byte-identical: same two
tools, same stages, same ``context_sha`` payload.  Nothing else in the Harness
is touched: ``OBSERVABLE_FEATURES``, the feature context, the Judge, the
Metric, the Operator DSL, the Source Skill and the Slow path are all unchanged,
and the recipe module itself is imported, never modified.

**Not authorization evidence.**  No Skill is written, no TRY right is granted,
no Episode is promoted, no Fast/Slow update runs, no snapshot pointer moves.
Every Experience entry this run writes carries
``provenance="batch_recipe_tool_v2_engineering"`` hard-coded in its payload: it
is a tool-mediated engineering measurement, and no later Skill-authorization
audit may count it as UNGUIDED exploration evidence.

**Information wall, stated plainly.**  The two pre-existing Workspace tools are
built from ``values[uid][:cutoff]`` and can see nothing at or after the Support
origin.  ``batch_recipe`` is not like that: the recipe reads the delayed
development origins and uses them inside its own adoption gate, so its result
carries delayed-window numbers into the Fast Agent's context.  That is why this
run writes no Skill and tags its Experience entries the way it does, and why
the tool must not be bound in any Task Episode run that produces authorization
evidence.  The tool's own description says so to the Agent as well.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import run_batch_composition_headroom as bch  # noqa: E402
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

from evaluation.functional.task_episode_harness.agentic.gateway import (  # noqa: E402
    CohortScopePublicToolGateway,
)
from evaluation.functional.task_episode_harness.agentic.runner import (  # noqa: E402
    _default_backend_factory,
)
from evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
    NF_MODEL,
)
from evaluation.functional.task_episode_harness.runner import (  # noqa: E402
    _compiled,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentProtocolError,
    AgentRole,
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_DELAYED,
    RELATION_ABSTAIN,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    SignedEpisodeRetriever,
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentCallBudgetExceeded,
    AgentTransportError,
)

PROTOCOL_VERSION = "agent_recipe_mount_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "agent_recipe_mount_v1.json"
OUT_MD = E2 / "agent_recipe_mount_v1.md"

# Hard-coded into every Experience entry this run writes.  These entries are
# tool-mediated engineering measurements; no later Skill-authorization audit
# may count them as UNGUIDED exploration evidence.
EXPERIENCE_PROVENANCE = "batch_recipe_tool_v2_engineering"
ADOPTION_RULE_VERSION = "v2"

LLM_CALL_BUDGET_TOTAL = 30
LLM_CALL_BUDGET_PER_EPISODE = 10
VALIDATION_RETRIES = 2
WORKSPACE_TOOL_BUDGET = 6
STAGE = "batch_plan"
TASK_INDEX = 0
MICRO_COHORT = "traffic"

# Fixed before the first LLM call; quoted verbatim into the artifact.
PRE_REGISTERED = {
    "behaviour_labels_are_read_off_the_transcript": (
        "tool_called is true iff a successful batch_recipe receipt exists in "
        "that episode's stage; reused is true iff the payload decision is "
        "REUSE_PRIOR_EXPERIENCE and it cites an episode_id that exists. "
        "Neither label is read out of the Agent's prose."
    ),
    "verdict_rules_first_match_wins": [
        "PROTOCOL_NOISE_BLOCKS_READOUT: E1 produced no payload, or two or "
        "more episodes produced no payload",
        "AGENT_IGNORES_TOOL: E1 made no successful batch_recipe call",
        "REUSES_WRONGLY_ACROSS_CELL: E3 reused an entry measured on a "
        "different cell and made no batch_recipe call of its own",
        "AGENT_ROUTES_AND_REUSES: E1 called and adopted, E2 reused without "
        "calling, E3 called again",
        "ROUTES_NO_REUSE: anything else that still routed to the tool",
    ],
    "circuit_breaker": (
        "the micro stops after E1 and reports if E1 produced no payload or "
        "made no successful batch_recipe call; E2 and E3 are not run"
    ),
    "budget": {
        "llm_calls_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_per_episode": LLM_CALL_BUDGET_PER_EPISODE,
        "validation_retries_per_stage": VALIDATION_RETRIES,
        "workspace_tool_calls_per_episode": WORKSPACE_TOOL_BUDGET,
    },
    "experience_relation_rule": (
        "POSITIVE if the plan's offline support and delayed gains are both "
        "positive; CONFLICT if they disagree in sign; NEGATIVE if both are "
        "non-positive and the plan treats something; ABSTAIN if the plan is "
        "identity"
    ),
}

# (episode_id, cohort, consumer_variant, request framing shown to the Agent)
EPISODE_PLAN: tuple[tuple[str, str, str, str], ...] = (
    (
        "E1",
        MICRO_COHORT,
        bch.CONSUMER_POOLED,
        "First processing request for this batch. Nothing has been recorded "
        "about it before now.",
    ),
    (
        "E2",
        MICRO_COHORT,
        bch.CONSUMER_POOLED,
        "Second processing request. It is the same batch of data as the "
        "previous request and the downstream Consumer structure is unchanged.",
    ),
    (
        "E3",
        MICRO_COHORT,
        bch.CONSUMER_PER_CHANNEL,
        "Same batch of data as the previous requests, but the downstream "
        "Consumer structure has changed: instead of one model fitted on the "
        "stacked windows of all training channels, each training channel now "
        "fits its own model and every evaluation channel is predicted by the "
        "equal-weight mean of those channel-wise models.",
    ),
)


def _cell_key(cohort: str, consumer_variant: str) -> str:
    return "batch:%s|consumer:%s" % (cohort, consumer_variant)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ------------------------------------------------------- rendered tool result
def _compact_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    """The recipe payload, rendered small enough to live in a Context.

    Adopted plan, the three-row comparison, the harm account and the gate
    outcome.  No threshold and no rule parameter is exposed as anything the
    Agent could set: the rule version is a label, and the tool takes no
    argument other than which batch and which Consumer structure.
    """
    plan = recipe["adopted_plan"]
    comparison = recipe["comparison"]
    harm = recipe["harm_account"]
    best_full = str(comparison["best_full_batch_program"])
    excluded = [str(uid) for uid in plan["excluded_series"]]
    return {
        "tool": "batch_recipe",
        "adoption_rule_version": str(recipe["adoption_rule_version"]),
        "deterministic": True,
        "llm_api_call_count": 0,
        "cohort": str(recipe["cohort"]),
        "consumer_variant": str(recipe["consumer_variant"]),
        "task_episode_id": str(recipe["task_episode_id"]),
        "support_origins": [int(o) for o in recipe["support_origins"]],
        "delayed_origins": [int(o) for o in recipe["delayed_origins"]],
        "train_series": [str(uid) for uid in recipe["train_series"]],
        "eval_series_count": len(recipe["eval_series"]),
        "adopted_plan": {
            "kind": str(plan["kind"]),
            "program": str(plan["program"]),
            "excluded_series": excluded,
            "excluded_series_count": len(excluded),
            "treated_series_count": int(plan["treated_series_count"]),
            "how_to_apply": str(plan["how_to_apply"]),
        },
        "comparison_rows": [
            {
                "plan": "adopted (%s, %d reverted)" % (plan["program"], len(excluded)),
                "support_aggregate_gain": float(comparison["support"]["adopted"]),
                "delayed_aggregate_gain": float(comparison["delayed"]["adopted"]),
            },
            {
                "plan": "best full batch (%s)" % best_full,
                "support_aggregate_gain": float(
                    comparison["support"]["best_full_batch"]
                ),
                "delayed_aggregate_gain": float(
                    comparison["delayed"]["best_full_batch"]
                ),
            },
            {
                "plan": "identity (treat nothing)",
                "support_aggregate_gain": 0.0,
                "delayed_aggregate_gain": 0.0,
            },
        ],
        "harm_account_evaluation_series_worse_than_identity": {
            key: {
                "harmed_series_count": int(
                    harm[key]["harmed_eval_series_count"]
                ),
                "total_harm": float(harm[key]["harmed_eval_series_total_harm"]),
            }
            for key in ("adopted", "best_full_batch", "identity")
        },
        "gate": {
            "adoption_path": str(recipe["adoption_path"]),
            "delayed_stability_bar": recipe.get("delayed_stability_bar"),
            "trace": [
                {
                    "program": str(row["program"]),
                    "excluded_series": [str(u) for u in row["excluded_series"]],
                    "support_aggregate_gain": float(
                        row["support_aggregate_gain"]
                    ),
                    "delayed_aggregate_gain": float(
                        row["delayed_aggregate_gain"]
                    ),
                    "delayed_bar": row.get("delayed_bar"),
                    "stability_check": str(row["stability_check"]),
                }
                for row in recipe["adoption_trace"]
            ],
        },
        "plan_validity_scope": (
            "this plan was measured on cohort=%s with consumer_variant=%s; it "
            "is a plan for that pair, and a different Consumer structure is a "
            "different batch problem with its own answer"
            % (recipe["cohort"], recipe["consumer_variant"])
        ),
        "caveat": str(recipe["caveat"]),
        "not_authorization_evidence": str(recipe["not_authorization_evidence"]),
    }


TOOL_DESCRIPTION = (
    "Run the frozen batch data-processing recipe on one already-exposed "
    "development batch and return the plan it adopts. The recipe scans a fixed "
    "program menu at full batch, runs a greedy harm-ordered exclusion mask "
    "search on the two best programs with a real Consumer retrain behind every "
    "single step, then applies a frozen delayed stability gate "
    "(adoption_rule_version v2). It is deterministic and makes no LLM call of "
    "its own, and it has no threshold or rule knob you can set: the only "
    "arguments are which batch and which Consumer structure. It costs one "
    "Workspace tool call and it is by far the most expensive tool here -- it "
    "retrains the downstream Consumer many times over. A plan it returns was "
    "measured for exactly one (cohort, consumer_variant) pair and says nothing "
    "about another pair. Information wall: unlike the two series tools, this "
    "result carries numbers from the delayed window, which the recipe uses "
    "inside its own adoption gate."
)


class BatchRecipeBinding:
    """What the gateway routes ``batch_recipe`` to.

    Owns the recipe call, the run-local result cache and the cost accounting.
    Writes no file: ``make_batch_recipe`` returns its payload and only the
    recipe module's own CLI ever writes an artifact, so no frozen stem under
    ``artifacts/functional/e2`` is touched by a tool call.
    """

    description = TOOL_DESCRIPTION

    def __init__(
        self,
        *,
        cohort_choices: Sequence[str],
        consumer_variant_choices: Sequence[str],
        task_index: int = TASK_INDEX,
        adoption_rule_version: str = ADOPTION_RULE_VERSION,
    ) -> None:
        self.cohort_choices = tuple(str(name) for name in cohort_choices)
        self.consumer_variant_choices = tuple(
            str(name) for name in consumer_variant_choices
        )
        self._task_index = int(task_index)
        self._rule_version = str(adoption_rule_version)
        self.identity = {
            "schema_version": "batch-recipe-tool-binding/1",
            "callable": "run_batch_composition_headroom.make_batch_recipe",
            "adoption_rule_version": self._rule_version,
            "task_index": self._task_index,
            "cohort_choices": list(self.cohort_choices),
            "consumer_variant_choices": list(self.consumer_variant_choices),
            "writes_no_artifact": True,
        }
        self._summaries: dict[tuple[str, str], dict[str, Any]] = {}
        self.full_recipes: dict[tuple[str, str], dict[str, Any]] = {}
        self.search_executions = 0
        self.call_log: list[dict[str, Any]] = []
        self.search_seconds = 0.0

    def run(self, *, cohort: str, consumer_variant: str) -> Mapping[str, Any]:
        key = (str(cohort), str(consumer_variant))
        cache_hit = key in self._summaries
        started = time.perf_counter()
        if not cache_hit:
            recipe = bch.make_batch_recipe(
                key[0],
                task_index=self._task_index,
                consumer_variant=key[1],
                adoption_rule_version=self._rule_version,
            )
            self.full_recipes[key] = dict(recipe)
            self._summaries[key] = _compact_recipe(recipe)
            self.search_executions += 1
            self.search_seconds += time.perf_counter() - started
        # The cache is a run-local cost saver only.  It is deliberately not
        # visible in the returned result: two calls on the same cell must look
        # identical to the Agent, or the transcript would stop being a clean
        # reading of what it chose to do.
        self.call_log.append({
            "cohort": key[0],
            "consumer_variant": key[1],
            "run_local_cache_hit": cache_hit,
            "wall_seconds": time.perf_counter() - started,
        })
        return dict(self._summaries[key])

    def accounting(self) -> dict[str, Any]:
        return {
            "tool_calls_routed": len(self.call_log),
            "recipe_searches_executed": self.search_executions,
            "recipe_search_seconds": self.search_seconds,
            "cache_hits": sum(
                1 for row in self.call_log if row["run_local_cache_hit"]
            ),
            "call_log": [dict(row) for row in self.call_log],
            "writes_no_artifact": True,
        }


# ------------------------------------------------- offline readout of a plan
class OfflinePlanEvaluator:
    """Score whatever plan the Agent ended up with, on the same mechanism.

    Nothing here is shown to the Agent.  It exists so the report can state the
    delayed gain of the plan that was actually adopted -- including a plan the
    Agent invented or mis-copied -- using the identical executor, Consumer
    variant, windows and gain definition the recipe itself uses.
    """

    def __init__(self) -> None:
        self._cells: dict[tuple[str, str], dict[str, Any]] = {}

    def _cell(self, cohort: str, consumer_variant: str) -> dict[str, Any]:
        key = (cohort, consumer_variant)
        cached = self._cells.get(key)
        if cached is not None:
            return cached
        config = dict(_config())
        loaded = bch.load_cohort(PROJECT_ROOT, cohort)
        roster = loaded["mapped_roster"]
        values = loaded["values"]
        train_uids = [str(uid) for uid in loaded["train_uids"]]
        eval_uids = [str(uid) for uid in loaded["eval_uids"]]
        _spec, support, delayed = bch._task_windows(cohort, TASK_INDEX)
        identity_support = bch._evaluate_variant(
            roster, values, None, config, support, None, consumer_variant,
        )
        identity_delayed = bch._evaluate_variant(
            roster, values, None, config, delayed, None, consumer_variant,
        )
        cell = {
            "config": config,
            "roster": roster,
            "values": values,
            "train_uids": train_uids,
            "eval_uids": eval_uids,
            "support": support,
            "delayed": delayed,
            "identity_support": identity_support,
            "identity_delayed": identity_delayed,
            "compiled": {},
            "exposure": loaded["exposure"],
        }
        self._cells[key] = cell
        return cell

    def train_uids(self, cohort: str, consumer_variant: str) -> list[str]:
        return list(self._cell(cohort, consumer_variant)["train_uids"])

    def evaluate(
        self,
        *,
        cohort: str,
        consumer_variant: str,
        program: str,
        excluded_series: Sequence[str],
    ) -> dict[str, Any]:
        cell = self._cell(cohort, consumer_variant)
        excluded = {str(uid) for uid in excluded_series}
        unknown = sorted(excluded - set(cell["train_uids"]))
        if program == bch.IDENTITY:
            assignment = {uid: None for uid in cell["train_uids"]}
        else:
            compiled = cell["compiled"].get(program)
            if compiled is None:
                compiled = _compiled(program, name="arm_%s" % program)
                cell["compiled"][program] = compiled
            assignment = {
                uid: (None if uid in excluded else compiled)
                for uid in cell["train_uids"]
            }

        def rows(origins: tuple[int, ...]) -> list[Mapping[str, Any]]:
            return [
                bch._evaluate_assignment(
                    cell["roster"], cell["values"], assignment, cell["config"],
                    origin=origin, consumer_variant=consumer_variant,
                )
                for origin in origins
            ]

        support = bch._gain_rows(
            cell["identity_support"], rows(cell["support"]), cell["eval_uids"]
        )
        delayed = bch._gain_rows(
            cell["identity_delayed"], rows(cell["delayed"]), cell["eval_uids"]
        )
        return {
            "program": program,
            "excluded_series": sorted(excluded),
            "unknown_series_ignored": unknown,
            "support_aggregate_gain": float(support["aggregate_gain"]),
            "delayed_aggregate_gain": float(delayed["aggregate_gain"]),
            "support_harmed_eval_series_count": int(
                support["harmed_eval_series_count"]
            ),
            "delayed_harmed_eval_series_count": int(
                delayed["harmed_eval_series_count"]
            ),
            "mechanism": (
                "run_batch_composition_headroom._evaluate_assignment + "
                "_gain_rows on the same Task windows and Consumer variant the "
                "recipe used; identity baseline recomputed once per cell"
            ),
        }


# --------------------------------------------------------------- stage wiring
BATCH_PLAN_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "batch-plan/1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision", "program", "excluded_series", "reason", "experience_use",
    ],
    "properties": {
        "decision": {
            "enum": [
                "ADOPT_TOOL_RESULT",
                "REUSE_PRIOR_EXPERIENCE",
                "IDENTITY_NO_TREATMENT",
            ]
        },
        "program": {"enum": list(bch.PROGRAM_MENU)},
        "excluded_series": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
            "maxItems": 24,
        },
        "reason": {"type": "string"},
        "experience_use": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8,
        },
    },
}

STAGE_NOTE = (
    "Return exactly one batch_plan payload for this batch. `program` and "
    "`excluded_series` are the plan itself: apply `program` to every training "
    "series except the ones listed, then retrain the Consumer once; "
    "`program`:'identity' with an empty list means treat nothing. `decision` "
    "records where the plan came from -- ADOPT_TOOL_RESULT if you took it from "
    "a batch_recipe result you obtained in this stage, REUSE_PRIOR_EXPERIENCE "
    "if you took it from a prior_experience entry (put that entry's "
    "episode_id in `experience_use`), IDENTITY_NO_TREATMENT if you leave the "
    "batch untreated. `reason` is one or two sentences on why, in public "
    "terms. Request one tool at a time and stop after each request; when you "
    "have what you need, return the payload."
)


def _experience_row(episode: Any, *, current_cell: str) -> dict[str, Any]:
    """One Experience entry as the Agent sees it.

    ``task_consumer_key`` is carried verbatim so the Agent can compare it with
    the cell key of the request in front of it.  The runner does not pre-judge
    the comparison for it.
    """
    row = episode.to_dict()
    return {
        "episode_id": row["episode_id"],
        "task_consumer_key": row["task_consumer_key"],
        "domain_namespace": row["domain_namespace"],
        "context_summary": row["context_summary"],
        "workflow_signature": row["workflow_signature"],
        "plan": {
            "program": row["support_response"].get("program"),
            "excluded_series": row["support_response"].get(
                "excluded_series", []
            ),
        },
        "support_aggregate_gain": row["support_response"].get("gain"),
        "delayed_aggregate_gain": row["delayed_response"].get("gain"),
        "relation": row["relation"],
        "evidence_level": row["evidence_level"],
        "local_status": row["local_status"],
        "provenance": row["support_response"].get("provenance"),
    }


def _public_input(
    *,
    episode_id: str,
    cohort: str,
    consumer_variant: str,
    request_note: str,
    spec: Mapping[str, Any],
    support: Sequence[int],
    delayed: Sequence[int],
    train_uids: Sequence[str],
    eval_count: int,
    exposure: str,
    experience_rows: Sequence[Mapping[str, Any]],
    pack: Mapping[str, Any],
    tool_budget: int,
) -> dict[str, Any]:
    return {
        "schema_version": "batch-plan-input/1",
        "episode_id": episode_id,
        "request": request_note,
        "batch": {
            "cohort": cohort,
            "consumer_variant": consumer_variant,
            "cell_key": _cell_key(cohort, consumer_variant),
            "task_episode_id": str(spec["task_episode_id"]),
            "training_series": [str(uid) for uid in train_uids],
            "evaluation_series_count": int(eval_count),
            "support_origins": [int(o) for o in support],
            "delayed_origins": [int(o) for o in delayed],
            "observation_cutoff": int(support[0]),
            "exposure": exposure,
        },
        "program_menu": list(bch.PROGRAM_MENU),
        "prior_experience": {
            "entries": [dict(row) for row in experience_rows],
            "signed_pack_for_this_cell": dict(pack),
            "how_to_read": (
                "Every entry is an Experience episode this run recorded "
                "earlier. `task_consumer_key` is the batch-and-Consumer cell "
                "the entry was measured on; `plan` is what was applied there "
                "and the two gains are what it produced. The signed pack is "
                "the deterministic retrieval restricted to the current cell "
                "key and may be empty."
            ),
        },
        "workspace_tool_budget": int(tool_budget),
        "stage_note": STAGE_NOTE,
    }


def _make_post_validator(
    *,
    gateway: CohortScopePublicToolGateway,
    known_episode_ids: Sequence[str],
    train_uids: Sequence[str],
):
    """Grounding checks only.  It never asks the Agent to choose differently.

    Three rules, each about the payload being an honest record of what
    happened: a REUSE decision must cite an entry that exists, an
    ADOPT_TOOL_RESULT decision must follow an actual successful tool call, and
    a plan may only name series that are in this batch.  Whether the adopted
    plan equals the tool's plan is measured afterwards and is deliberately not
    enforced -- a mismatch is behaviour worth reading, not a violation.
    """
    known = {str(item) for item in known_episode_ids}
    batch = {str(uid) for uid in train_uids}

    def validate(payload: Mapping[str, Any]) -> None:
        decision = str(payload["decision"])
        cited = [str(item) for item in payload.get("experience_use", ())]
        excluded = [str(item) for item in payload.get("excluded_series", ())]
        outside = sorted(set(excluded) - batch)
        if outside:
            raise StagePostValidationError(
                "SERIES_OUTSIDE_BATCH",
                "excluded_series names series that are not training series of "
                "this batch: %s" % outside,
                retryable=True,
            )
        if decision == "REUSE_PRIOR_EXPERIENCE":
            unknown = sorted(set(cited) - known)
            if not cited or unknown:
                raise StagePostValidationError(
                    "EXPERIENCE_CITATION_UNGROUNDED",
                    "a REUSE_PRIOR_EXPERIENCE decision must cite at least one "
                    "episode_id from prior_experience.entries; unknown ids: %s"
                    % (unknown or "none cited"),
                    retryable=True,
                )
        if decision == "ADOPT_TOOL_RESULT":
            called = any(
                row.get("tool_name") == "batch_recipe" and row.get("ok")
                for row in gateway.call_log
            )
            if not called:
                raise StagePostValidationError(
                    "TOOL_RESULT_NOT_OBTAINED",
                    "an ADOPT_TOOL_RESULT decision requires a successful "
                    "batch_recipe call in this stage; none was made",
                    retryable=True,
                )
        if decision == "IDENTITY_NO_TREATMENT" and (
            str(payload["program"]) != bch.IDENTITY or excluded
        ):
            raise StagePostValidationError(
                "IDENTITY_PLAN_INCONSISTENT",
                "an IDENTITY_NO_TREATMENT decision must carry "
                "program='identity' and an empty excluded_series",
                retryable=True,
            )

    return validate


def _experience_entry(
    *,
    episode_id: str,
    cohort: str,
    consumer_variant: str,
    program: str,
    excluded_series: Sequence[str],
    offline: Mapping[str, Any],
    train_count: int,
    eval_count: int,
    plan_source: str,
) -> Any:
    """One Experience episode, written through the existing episode mechanism.

    No new store: the episodes live in this run's list and in this artifact,
    exactly as ``_ArmState.episodes`` holds them inside a Task Episode run.
    ``provenance`` is hard-coded on both responses and repeated in
    ``evidence_refs``; the ``counts_as_unguided_exploration`` flag is written
    false so a later authorization audit cannot mistake a tool-mediated
    measurement for an unguided probe.
    """
    support_gain = float(offline["support_aggregate_gain"])
    delayed_gain = float(offline["delayed_aggregate_gain"])
    excluded = [str(uid) for uid in excluded_series]
    if program == bch.IDENTITY:
        relation = RELATION_ABSTAIN
    elif support_gain > 0.0 and delayed_gain > 0.0:
        relation = RELATION_POSITIVE
    elif (support_gain > 0.0) != (delayed_gain > 0.0):
        relation = RELATION_CONFLICT
    else:
        relation = RELATION_NEGATIVE
    audit = {
        "provenance": EXPERIENCE_PROVENANCE,
        "counts_as_unguided_exploration": False,
        "audit_note": (
            "tool-mediated engineering measurement produced by the mounted "
            "batch_recipe tool; not authorization evidence and not an "
            "unguided probe"
        ),
    }
    return build_episode(
        episode_id=episode_id,
        task_consumer_key=_cell_key(cohort, consumer_variant),
        domain_namespace=cohort,
        context_summary={
            "cohort": {
                "cohort_name": cohort,
                "training_series_count": int(train_count),
                "evaluation_series_count": int(eval_count),
            },
            "local_pattern": {"consumer_variant": consumer_variant},
            "program_geometry": {
                "excluded_count": len(excluded),
                "treated_count": int(train_count) - len(excluded),
            },
        },
        workflow_signature=workflow_signature_of(
            () if program == bch.IDENTITY else ({"op": program},)
        ),
        support_response={
            "gain": support_gain,
            "window": "support",
            "program": program,
            "excluded_series": excluded,
            "plan_source": plan_source,
            "harmed_eval_series_count": int(
                offline["support_harmed_eval_series_count"]
            ),
            **audit,
        },
        delayed_response={
            "gain": delayed_gain,
            "window": "delayed",
            "window_role": (
                "already-exposed development origins; the recipe's own "
                "adoption gate reads this window, so it is in-selection for "
                "any plan the tool produced"
            ),
            "harmed_eval_series_count": int(
                offline["delayed_harmed_eval_series_count"]
            ),
            **audit,
        },
        relation=relation,
        evidence_level=EVIDENCE_DELAYED,
        local_status=STATUS_EPISODE_ONLY,
        evidence_refs=(EXPERIENCE_PROVENANCE,),
    )


# ------------------------------------------------------------------- episode
def _retrieval_context(cohort: str, consumer_variant: str) -> dict[str, Any]:
    return {
        "cohort": {"cohort_name": cohort},
        "local_pattern": {"consumer_variant": consumer_variant},
    }


def _run_episode(
    *,
    episode_id: str,
    cohort: str,
    consumer_variant: str,
    request_note: str,
    binding: BatchRecipeBinding,
    evaluator: OfflinePlanEvaluator,
    episodes: list[Any],
    snapshot: Any,
    llm_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cell_key = _cell_key(cohort, consumer_variant)
    cell = evaluator._cell(cohort, consumer_variant)
    spec, support, delayed = bch._task_windows(cohort, TASK_INDEX)
    cutoff = int(support[0])
    train_uids = list(cell["train_uids"])

    gateway = CohortScopePublicToolGateway(
        {
            uid: np.asarray(cell["values"][uid], dtype=np.float64)[:cutoff]
            for uid in train_uids
        },
        task_kind="forecast",
        observation_cutoff=cutoff,
        maximum_calls=WORKSPACE_TOOL_BUDGET,
        batch_recipe_binding=binding,
    )
    pack = SignedEpisodeRetriever(
        tuple(episodes), task_consumer_key=cell_key
    ).retrieve(_retrieval_context(cohort, consumer_variant), cohort)
    experience_rows = [
        _experience_row(episode, current_cell=cell_key) for episode in episodes
    ]
    public_input = _public_input(
        episode_id=episode_id,
        cohort=cohort,
        consumer_variant=consumer_variant,
        request_note=request_note,
        spec=spec,
        support=support,
        delayed=delayed,
        train_uids=train_uids,
        eval_count=len(cell["eval_uids"]),
        exposure=str(cell["exposure"]),
        experience_rows=experience_rows,
        pack=pack.to_dict(),
        tool_budget=WORKSPACE_TOOL_BUDGET,
    )

    backend = _default_backend_factory(int(llm_budget))
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)
    harness_view = resolve_harness_view(snapshot, {}, role="fast")
    validator = _make_post_validator(
        gateway=gateway,
        known_episode_ids=[row["episode_id"] for row in experience_rows],
        train_uids=train_uids,
    )

    payload: Mapping[str, Any] | None = None
    result: Any = None
    protocol_error: str | None = None
    infrastructure_error: str | None = None
    try:
        result = core.run_stage(
            role=AgentRole.FAST,
            stage=STAGE,
            case_id="BRM_%s" % episode_id,
            public_input=public_input,
            harness_view=harness_view,
            output_schema_name="batch_plan_v1",
            output_schema=BATCH_PLAN_SCHEMA,
            source_snapshot_sha=harness_view.effective_harness_view_sha,
            validation_retries=VALIDATION_RETRIES,
            post_validator=validator,
        )
        payload = dict(result.payload)
    except (AgentProtocolError, StagePostValidationError, PermissionError) as exc:
        protocol_error = "%s: %s" % (type(exc).__name__, exc)
    except (AgentTransportError, AgentCallBudgetExceeded) as exc:
        infrastructure_error = "%s: %s" % (type(exc).__name__, exc)

    receipts = [
        {
            "tool_name": receipt.tool_name,
            "arguments": _plain(receipt.arguments),
            "ok": bool(receipt.ok),
            "receipt_sha": receipt.receipt_sha,
        }
        for receipt in (result.tool_receipts if result is not None else ())
    ]
    recipe_receipts = [
        row for row in receipts if row["tool_name"] == "batch_recipe" and row["ok"]
    ]
    tool_called = bool(recipe_receipts)
    recipe_for_this_cell = binding.full_recipes.get((cohort, consumer_variant))
    tool_plan = None
    if tool_called and recipe_for_this_cell is not None:
        adopted = recipe_for_this_cell["adopted_plan"]
        tool_plan = {
            "program": str(adopted["program"]),
            "excluded_series": sorted(
                str(uid) for uid in adopted["excluded_series"]
            ),
        }
    return {
        "episode_id": episode_id,
        "cohort": cohort,
        "consumer_variant": consumer_variant,
        "cell_key": cell_key,
        "request": request_note,
        "payload": _plain(payload) if payload is not None else None,
        "protocol_error": protocol_error,
        "infrastructure_error": infrastructure_error,
        "llm_calls": int(backend.calls),
        "llm_call_budget_for_this_episode": int(llm_budget),
        "validation_retry_count": (
            int(result.validation_retry_count) if result is not None else None
        ),
        "validation_error_codes": (
            list(result.validation_error_codes) if result is not None else []
        ),
        "first_pass_valid": (
            bool(result.first_pass_valid) if result is not None else False
        ),
        "tool_receipts": receipts,
        "tool_called": tool_called,
        "tool_plan_for_this_cell": tool_plan,
        "workspace_tool_accounting": gateway.accounting(),
        "experience_entries_visible": len(experience_rows),
        "signed_pack_for_this_cell": pack.to_dict(),
        "wall_seconds": time.perf_counter() - started,
    }


# ------------------------------------------------------------------ labelling
def _label_episode(
    record: Mapping[str, Any], episodes: Sequence[Any]
) -> dict[str, Any]:
    """Behaviour read off the transcript, by the pre-registered rules."""
    by_id = {
        str(episode.episode_id): str(episode.task_consumer_key)
        for episode in episodes
    }
    payload = record["payload"]
    if payload is None:
        return {
            "behaviour": "NO_PAYLOAD",
            "decision": None,
            "reused": False,
            "reused_across_cell": False,
            "cited_episode_ids": [],
            "cited_cells": [],
            "plan": None,
            "plan_matches_tool_plan": None,
        }
    decision = str(payload["decision"])
    cited = [str(item) for item in payload.get("experience_use", ())]
    cited_cells = [by_id.get(item, "UNKNOWN") for item in cited]
    plan = {
        "program": str(payload["program"]),
        "excluded_series": sorted(
            str(uid) for uid in payload.get("excluded_series", ())
        ),
    }
    tool_plan = record["tool_plan_for_this_cell"]
    matches = None if tool_plan is None else plan == tool_plan
    reused = decision == "REUSE_PRIOR_EXPERIENCE" and bool(cited)
    across = reused and any(
        cell != record["cell_key"] for cell in cited_cells
    )
    if record["tool_called"] and decision == "ADOPT_TOOL_RESULT":
        behaviour = (
            "TOOL_CALLED_AND_ADOPTED" if matches
            else "TOOL_CALLED_PLAN_DIVERGED"
        )
    elif record["tool_called"] and reused:
        behaviour = "TOOL_CALLED_THEN_REUSED"
    elif record["tool_called"]:
        behaviour = "TOOL_CALLED_OTHER_DECISION"
    elif reused:
        behaviour = "EXPERIENCE_REUSED_NO_TOOL_CALL"
    elif decision == "IDENTITY_NO_TREATMENT":
        behaviour = "IDENTITY_NO_TOOL_CALL"
    else:
        behaviour = "NO_TOOL_CALL_NO_REUSE"
    return {
        "behaviour": behaviour,
        "decision": decision,
        "reused": reused,
        "reused_across_cell": across,
        "cited_episode_ids": cited,
        "cited_cells": cited_cells,
        "plan": plan,
        "plan_matches_tool_plan": matches,
    }


def _verdict(
    records: Sequence[Mapping[str, Any]], stopped_reason: str | None
) -> dict[str, Any]:
    by_id = {str(row["episode_id"]): row for row in records}
    e1, e2, e3 = (by_id.get("E1"), by_id.get("E2"), by_id.get("E3"))
    no_payload = [
        str(row["episode_id"]) for row in records if row["payload"] is None
    ]
    reasons: list[str] = []
    if e1 is None:
        verdict = "PROTOCOL_NOISE_BLOCKS_READOUT"
        reasons.append("E1 did not run")
    elif e1["payload"] is None or len(no_payload) >= 2:
        verdict = "PROTOCOL_NOISE_BLOCKS_READOUT"
        reasons.append("episodes without a payload: %s" % (no_payload or "E1"))
    elif not e1["tool_called"]:
        verdict = "AGENT_IGNORES_TOOL"
        reasons.append("E1 made no successful batch_recipe call")
    elif (
        e3 is not None
        and e3["label"]["reused_across_cell"]
        and not e3["tool_called"]
    ):
        verdict = "REUSES_WRONGLY_ACROSS_CELL"
        reasons.append(
            "E3 reused %s, measured on %s, without calling the tool"
            % (e3["label"]["cited_episode_ids"], e3["label"]["cited_cells"])
        )
    elif (
        e1["label"]["behaviour"] == "TOOL_CALLED_AND_ADOPTED"
        and e2 is not None
        and e2["label"]["reused"]
        and not e2["tool_called"]
        and e3 is not None
        and e3["tool_called"]
    ):
        verdict = "AGENT_ROUTES_AND_REUSES"
        reasons.append(
            "E1 called and adopted, E2 reused without calling, E3 called again"
        )
    else:
        verdict = "ROUTES_NO_REUSE"
        reasons.append(
            "E1 routed to the tool but the reuse/re-search pattern is not the "
            "full one: E2 %s, E3 %s"
            % (
                e2["label"]["behaviour"] if e2 else "not run",
                e3["label"]["behaviour"] if e3 else "not run",
            )
        )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "stopped_early": stopped_reason,
        "episodes_run": [str(row["episode_id"]) for row in records],
        "rule": PRE_REGISTERED["verdict_rules_first_match_wins"],
    }


# --------------------------------------------------------------- orchestration
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    snapshot = compile_snapshot(
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False
    )
    binding = BatchRecipeBinding(
        cohort_choices=bch.RECIPE_V2_COHORTS,
        consumer_variant_choices=bch.CONSUMER_VARIANTS,
    )
    evaluator = OfflinePlanEvaluator()
    episodes: list[Any] = []
    records: list[dict[str, Any]] = []
    llm_used = 0
    stopped_reason: str | None = None

    for episode_id, cohort, consumer_variant, request_note in EPISODE_PLAN:
        remaining = LLM_CALL_BUDGET_TOTAL - llm_used
        if remaining < 2:
            stopped_reason = (
                "global LLM budget exhausted before %s (%d of %d used)"
                % (episode_id, llm_used, LLM_CALL_BUDGET_TOTAL)
            )
            break
        print(
            "MICRO %s %s x %s (llm used %d/%d)"
            % (episode_id, cohort, consumer_variant, llm_used,
               LLM_CALL_BUDGET_TOTAL),
            flush=True,
        )
        record = _run_episode(
            episode_id=episode_id,
            cohort=cohort,
            consumer_variant=consumer_variant,
            request_note=request_note,
            binding=binding,
            evaluator=evaluator,
            episodes=episodes,
            snapshot=snapshot,
            llm_budget=min(LLM_CALL_BUDGET_PER_EPISODE, remaining),
        )
        llm_used += int(record["llm_calls"])
        record["label"] = _label_episode(record, episodes)

        offline = None
        written: Any = None
        plan = record["label"]["plan"]
        if plan is not None:
            offline = evaluator.evaluate(
                cohort=cohort,
                consumer_variant=consumer_variant,
                program=plan["program"],
                excluded_series=plan["excluded_series"],
            )
            written = _experience_entry(
                episode_id=episode_id,
                cohort=cohort,
                consumer_variant=consumer_variant,
                program=plan["program"],
                excluded_series=plan["excluded_series"],
                offline=offline,
                train_count=len(evaluator.train_uids(cohort, consumer_variant)),
                eval_count=len(
                    evaluator._cell(cohort, consumer_variant)["eval_uids"]
                ),
                plan_source=str(record["label"]["decision"]),
            )
            episodes.append(written)
        record["adopted_plan_offline_readout"] = offline
        record["experience_written"] = (
            written.to_dict() if written is not None else None
        )
        records.append(record)
        print(
            "MICRO %s behaviour=%s tool_called=%s llm=%d plan=%s delayed=%s"
            % (
                episode_id, record["label"]["behaviour"], record["tool_called"],
                record["llm_calls"], plan,
                "n/a" if offline is None
                else "%+.6f" % offline["delayed_aggregate_gain"],
            ),
            flush=True,
        )

        if episode_id == "E1" and (
            record["payload"] is None or not record["tool_called"]
        ):
            stopped_reason = (
                "circuit breaker: E1 %s"
                % (
                    "produced no payload" if record["payload"] is None
                    else "made no successful batch_recipe call"
                )
            )
            break

    verdict = _verdict(records, stopped_reason)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "engineering demonstration that a Fast Agent can route to the "
            "mounted batch recipe and that a prior Experience entry removes "
            "the cost of re-searching the same cell"
        ),
        "not_authorization_evidence": (
            "no Skill is written, no TRY right is granted, no Episode is "
            "promoted, no Fast or Slow update runs, no snapshot pointer moves"
        ),
        "harness_change_surface": {
            "surface": "Fast Agent Workspace tool supply",
            "file": (
                "evaluation/functional/task_episode_harness/agentic/gateway.py"
            ),
            "change": (
                "CohortScopePublicToolGateway gained an optional "
                "batch_recipe_binding: one extra tool name, its description, "
                "its two bounded argument enums, one extra allowed stage and "
                "the binding identity folded into context_sha"
            ),
            "default_path_unchanged": (
                "with batch_recipe_binding=None the class serves the same two "
                "tools on the same three stages and hashes the same "
                "context_sha payload as before"
            ),
            "unchanged": [
                "OBSERVABLE_FEATURES", "feature_context_sha", "Judge",
                "Metric", "Operator DSL", "Source Skill", "Slow path",
                "run_batch_composition_headroom (imported, not modified)",
            ],
        },
        "information_wall_note": (
            "batch_recipe returns delayed-window numbers, which the two "
            "pre-existing Workspace tools cannot see. The recipe uses the "
            "delayed window inside its own adoption gate, so mounting it "
            "widens what the Fast Agent can observe. This run therefore "
            "writes no Skill and tags every Experience entry "
            "provenance=%s; the tool must not be bound in a Task Episode run "
            "that produces authorization evidence." % EXPERIENCE_PROVENANCE
        ),
        "pre_registered": PRE_REGISTERED,
        "model": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "adoption_rule_version": ADOPTION_RULE_VERSION,
        "tool_binding_identity": binding.identity,
        "tool_description_shown_to_agent": TOOL_DESCRIPTION,
        "batch_plan_schema": BATCH_PLAN_SCHEMA,
        "verdict": verdict,
        "llm_call_count": llm_used,
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "recipe_tool_accounting": binding.accounting(),
        "experience_entries": [episode.to_dict() for episode in episodes],
        "experience_provenance": EXPERIENCE_PROVENANCE,
        "episodes": records,
        "windows": {
            "cohort": MICRO_COHORT,
            "note": (
                "traffic development origins only: Support 1104/1368, delayed "
                "1800, farthest read 1848, sealed_from_index 3072 never "
                "approached"
            ),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    if dry_run:
        print(json.dumps(verdict, indent=2, ensure_ascii=False))
        return 0
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict", verdict["verdict"], flush=True)
    print("llm_calls", llm_used, flush=True)
    return 0


# --------------------------------------------------------------------- report
def _gain(value: Any) -> str:
    return "n/a" if value is None else "%+.6f" % float(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload["verdict"]
    surface = payload["harness_change_surface"]
    lines: list[str] = [
        "# agent-recipe-mount micro v1",
        "",
        "**Verdict: `%s`**" % verdict["verdict"],
        "",
        "The batch recipe was mounted as a Workspace tool the Fast Agent may "
        "call, and a three-episode single-arm micro measured whether the Agent "
        "routes to it, reuses a prior Experience entry on the same cell, and "
        "searches again when the Consumer structure changes.",
        "",
        "**Engineering demonstration, not authorization evidence.** No Skill "
        "is written, no TRY right is granted, no Episode is promoted, no Fast "
        "or Slow update runs, and no snapshot pointer moves.",
        "",
        "> **Information wall.** %s" % payload["information_wall_note"],
        "",
        "## 0. What was mounted",
        "",
        "- surface: %s (`%s`)" % (surface["surface"], surface["file"]),
        "- change: %s" % surface["change"],
        "- default path: %s" % surface["default_path_unchanged"],
        "- unchanged: %s" % ", ".join("`%s`" % item for item in surface["unchanged"]),
        "- tool arguments: `cohort` and `consumer_variant` only; no threshold "
        "and no rule parameter is exposed",
        "- the tool writes no file: `make_batch_recipe` returns its payload and "
        "only the recipe module's own CLI writes an artifact",
        "",
        "Tool description as the Agent saw it:",
        "",
        "> %s" % payload["tool_description_shown_to_agent"],
        "",
        "## 1. Verdict",
        "",
        "Rules were fixed before the first LLM call, first match wins:",
        "",
    ]
    for index, rule in enumerate(verdict["rule"], start=1):
        lines.append("%d. %s" % (index, rule))
    lines += [
        "",
        "Matched: **`%s`** -- %s." % (verdict["verdict"], "; ".join(verdict["reasons"])),
        "",
    ]
    if verdict["stopped_early"]:
        lines += ["Stopped early: %s." % verdict["stopped_early"], ""]
    lines += [
        "## 2. Episode by episode",
        "",
        "| episode | cell | behaviour | decision | plan | support | delayed | "
        "LLM calls | tool calls | retries |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in payload["episodes"]:
        label = record["label"]
        offline = record["adopted_plan_offline_readout"] or {}
        plan = label["plan"]
        plan_text = (
            "n/a" if plan is None
            else "`%s` minus %s" % (
                plan["program"],
                ", ".join(plan["excluded_series"]) or "nothing",
            )
        )
        lines.append(
            "| %s | %s | `%s` | %s | %s | %s | %s | %d | %d | %s |"
            % (
                record["episode_id"],
                record["consumer_variant"],
                label["behaviour"],
                label["decision"] or "n/a",
                plan_text,
                _gain(offline.get("support_aggregate_gain")),
                _gain(offline.get("delayed_aggregate_gain")),
                record["llm_calls"],
                len(record["tool_receipts"]),
                record["validation_retry_count"],
            )
        )
    lines += [
        "",
        "`support` and `delayed` are the aggregate gains of the plan the Agent "
        "actually returned, recomputed offline on the same executor, Consumer "
        "variant, windows and gain definition the recipe uses. They are not "
        "copied from the tool result.",
        "",
    ]
    for record in payload["episodes"]:
        label = record["label"]
        lines += [
            "### %s -- %s x %s" % (
                record["episode_id"], record["cohort"],
                record["consumer_variant"],
            ),
            "",
            "Request: %s" % record["request"],
            "",
            "- behaviour: `%s`" % label["behaviour"],
            "- tool calls: %s"
            % (
                ", ".join(
                    "`%s`%s" % (row["tool_name"], "" if row["ok"] else " (refused)")
                    for row in record["tool_receipts"]
                ) or "none"
            ),
            "- experience entries visible: %d; cited: %s"
            % (
                record["experience_entries_visible"],
                ", ".join(
                    "`%s` (%s)" % (item, cell)
                    for item, cell in zip(
                        label["cited_episode_ids"], label["cited_cells"]
                    )
                ) or "none",
            ),
            "- plan matches the tool's plan for this cell: %s"
            % label["plan_matches_tool_plan"],
            "- LLM calls: %d (budget %d), schema/post-validation retries: %s %s"
            % (
                record["llm_calls"],
                record["llm_call_budget_for_this_episode"],
                record["validation_retry_count"],
                record["validation_error_codes"] or "",
            ),
        ]
        if record["protocol_error"]:
            lines.append("- protocol error: `%s`" % record["protocol_error"])
        if record["infrastructure_error"]:
            lines.append(
                "- infrastructure error: `%s`" % record["infrastructure_error"]
            )
        if record["payload"]:
            lines.append(
                "- reason given: %s" % record["payload"].get("reason", "")
            )
        lines.append("")
    return "\n".join(lines) + "\n" + _markdown_tail(payload)


def _markdown_tail(payload: Mapping[str, Any]) -> str:
    accounting = payload["recipe_tool_accounting"]
    lines: list[str] = [
        "## 3. Experience entries written",
        "",
        "Written through the existing episode mechanism "
        "(`methods/ttha/experience_memory.build_episode`); no new store is "
        "created and no Skill is formed. Every entry carries "
        "`provenance=\"%s\"` on both its support and delayed response and in "
        "`evidence_refs`, plus `counts_as_unguided_exploration: false`: these "
        "are tool-mediated engineering measurements and a later "
        "Skill-authorization audit must not count them as UNGUIDED probes."
        % payload["experience_provenance"],
        "",
        "| episode_id | task_consumer_key | workflow_signature | plan | "
        "support | delayed | relation | local_status | provenance |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for entry in payload["experience_entries"]:
        support = entry["support_response"]
        delayed = entry["delayed_response"]
        lines.append(
            "| `%s` | `%s` | `%s` | `%s` minus %s | %s | %s | %s | %s | `%s` |"
            % (
                entry["episode_id"],
                # a cell key contains a pipe; escape it so the table survives
                str(entry["task_consumer_key"]).replace("|", r"\|"),
                entry["workflow_signature"],
                support.get("program"),
                ", ".join(support.get("excluded_series") or []) or "nothing",
                _gain(support.get("gain")),
                _gain(delayed.get("gain")),
                entry["relation"],
                entry["local_status"],
                support.get("provenance"),
            )
        )
    lines += [
        "",
        "## 4. Cost",
        "",
        "| item | value |",
        "| --- | ---: |",
        "| LLM calls, whole micro | %d (budget %d) |"
        % (payload["llm_call_count"], payload["llm_call_budget_total"]),
        "| batch_recipe tool calls routed | %d |"
        % accounting["tool_calls_routed"],
        "| recipe searches actually executed | %d |"
        % accounting["recipe_searches_executed"],
        "| run-local cache hits | %d |" % accounting["cache_hits"],
        "| recipe search wall seconds | %.1f |"
        % accounting["recipe_search_seconds"],
        "| whole micro wall seconds | %.1f |" % payload["wall_seconds"],
        "",
        "The run-local cache is a cost saver inside this process and is "
        "deliberately invisible in the tool result, so two calls on the same "
        "cell look identical to the Agent and the transcript stays a clean "
        "reading of what it chose to do.",
        "",
        "## 5. What this does not say",
        "",
        "- It does not authorize anything. No Skill, no TRY right, no "
        "promotion, no snapshot pointer move, no Fast or Slow update.",
        "- It does not claim the adopted plans generalize. The recipe's "
        "delayed window is inside its own selection, so both reported columns "
        "are in-selection for any plan the tool produced.",
        "- It does not measure reuse quality, only reuse behaviour: whether a "
        "prior entry was reused, and what the reused plan is worth on the "
        "current cell when scored offline.",
        "- Three episodes on one cohort is a mechanism demonstration, not an "
        "effect size. Nothing here is a rate.",
        "",
        "## Provenance",
        "",
        "- model: `%s` at `%s`" % (
            payload["model"]["model"], payload["model"]["base_url"],
        ),
        "- recipe: `run_batch_composition_headroom.make_batch_recipe`, "
        "adoption_rule_version `%s`, imported and not modified"
        % payload["adoption_rule_version"],
        "- windows: %s" % payload["windows"]["note"],
        "- stage: `batch_plan`, schema `batch-plan/1` declared inside this "
        "runner, so no stage-schema file was added",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the micro but print the verdict instead of writing artifacts",
    )
    args = parser.parse_args(argv)
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
