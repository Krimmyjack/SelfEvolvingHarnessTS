"""agent-recipe-mount, no-tool control arm: what does the recipe actually buy?

The tool arm (``artifacts/functional/e2/agent_recipe_mount_v1.json``) showed the
routing and reuse *behaviour*: the Fast Agent called the mounted ``batch_recipe``
tool, adopted its plan, reused the Experience entry on the same cell, and
searched again when the Consumer structure changed.  Behaviour is not quality.
This arm answers the other half: with the same three requests, the same schema
and the same prompt, but **no batch_recipe binding**, what plan does the Agent
produce from its own observation, and how does that plan score?

Three episodes, identical to the tool arm: traffic x pooled first sight, traffic
x pooled revisit, traffic x per_channel.  The tool arm is not re-run; its frozen
artifact is read for the comparison.

What is identical to the tool arm, by construction rather than by copying: the
request framings, the batch block, the program menu, the prior-experience
rendering, the Workspace tool budget and the stage note are all imported from
``run_e2_agent_recipe_mount_micro`` and reused verbatim.  The plan is scored by
the same ``OfflinePlanEvaluator`` -- the same ``_evaluate_assignment`` +
``_gain_rows`` on the same windows and Consumer variant.

Two differences, both stated in the artifact and neither of them a nudge:

1. **The tool supply.**  No binding, so the Agent has only the two observation
   tools ``summarize_series`` and ``localize_regions``.  With
   ``batch_recipe_binding=None`` the frozen gateway serves no tool at all on the
   ``batch_plan`` stage, so this runner subclasses it and looks the stage up as
   ``inspect``: the supply served is exactly the frozen ``binding=None`` supply,
   the gateway module is untouched, and no frozen contract is edited.
2. **One extra ``decision`` value.**  The tool arm's enum is
   ``ADOPT_TOOL_RESULT`` / ``REUSE_PRIOR_EXPERIENCE`` / ``IDENTITY_NO_TREATMENT``.
   With no tool and no prior entry, an Agent that wants to propose a treatment
   has no honest label left, and the post-validator would force it to identity.
   That would measure this runner's contract, not the Agent.  So the control
   schema adds ``OWN_OBSERVATION`` and the stage note gains one sentence naming
   it.  Everything else in the schema and the note is byte-identical, and the
   three tool-arm values keep their meaning, so the payloads stay comparable.
   This is the one place where the control prompt is not verbatim, and it is
   flagged in the artifact header.

Pre-registered before the first LLM call:

* per-episode outcome is read off the **delayed** aggregate gain of the plan
  each arm actually returned, recomputed offline: ``TOOL`` if the tool arm is
  ahead by more than MATERIAL_THRESHOLD, ``NOTOOL`` if the control arm is,
  ``TIE`` otherwise;
* overall verdict: ``TOOL_ARM_DOMINATES`` if every compared episode is ``TOOL``;
  ``HAND_ROLLED_COMPETITIVE`` if none is; ``MIXED_BY_EPISODE`` otherwise;
* circuit breaker: stop after E1 and report if E1 produced no payload;
* budget: at most 30 LLM calls for the arm, at most 10 per episode, at most 2
  schema/post-validation retries per stage.

Experience entries are written through the same episode mechanism with
``provenance="agent_hand_rolled_engineering"`` -- strictly distinct from the
tool arm's tag -- plus ``counts_as_unguided_exploration: false`` and the same
audit note.  No Skill, no TRY right, no promotion, no Fast or Slow update.

Data: traffic development origins only (Support 1104/1368, delayed 1800,
farthest read 1848); ``sealed_from_index=3072`` is never approached.

Run:

    python evaluation/functional/run_e2_agent_recipe_mount_notool_micro.py

Writes ``artifacts/functional/e2/agent_recipe_mount_notool_v1.json`` and ``.md``.
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
import run_e2_agent_recipe_mount_micro as toolarm  # noqa: E402

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

PROTOCOL_VERSION = "agent_recipe_mount_notool_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "agent_recipe_mount_notool_v1.json"
OUT_MD = E2 / "agent_recipe_mount_notool_v1.md"
TOOL_ARM_ARTIFACT = E2 / "agent_recipe_mount_v1.json"

# Strictly distinct from the tool arm's batch_recipe_tool_v2_engineering: these
# plans came out of the Agent's own reading of the public prefixes.
EXPERIENCE_PROVENANCE = "agent_hand_rolled_engineering"

# Everything below is the tool arm's, reused rather than restated.
LLM_CALL_BUDGET_TOTAL = toolarm.LLM_CALL_BUDGET_TOTAL
LLM_CALL_BUDGET_PER_EPISODE = toolarm.LLM_CALL_BUDGET_PER_EPISODE
VALIDATION_RETRIES = toolarm.VALIDATION_RETRIES
WORKSPACE_TOOL_BUDGET = toolarm.WORKSPACE_TOOL_BUDGET
STAGE = toolarm.STAGE
TASK_INDEX = toolarm.TASK_INDEX
MICRO_COHORT = toolarm.MICRO_COHORT
EPISODE_PLAN = toolarm.EPISODE_PLAN
MATERIAL_THRESHOLD = float(bch.MATERIAL_THRESHOLD)

# Fixed before the first LLM call; quoted verbatim into the artifact.
PRE_REGISTERED = {
    "per_episode_outcome": (
        "read off the delayed aggregate gain of the plan each arm actually "
        "returned, recomputed offline by the same evaluator: TOOL if the tool "
        "arm leads by more than MATERIAL_THRESHOLD=%.3f, NOTOOL if the control "
        "arm does, TIE otherwise" % MATERIAL_THRESHOLD
    ),
    "verdict_rules": [
        "TOOL_ARM_DOMINATES: every compared episode is TOOL",
        "HAND_ROLLED_COMPETITIVE: no compared episode is TOOL",
        "MIXED_BY_EPISODE: anything else",
    ],
    "circuit_breaker": (
        "stop after E1 and report if E1 produced no payload; E2 and E3 are "
        "not run"
    ),
    "budget": {
        "llm_calls_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_per_episode": LLM_CALL_BUDGET_PER_EPISODE,
        "validation_retries_per_stage": VALIDATION_RETRIES,
        "workspace_tool_calls_per_episode": WORKSPACE_TOOL_BUDGET,
    },
    "experience_relation_rule": toolarm.PRE_REGISTERED[
        "experience_relation_rule"
    ],
}

PROMPT_PARITY = {
    "identical_and_imported_from_the_tool_arm": [
        "request framing of E1/E2/E3 (EPISODE_PLAN)",
        "batch block, program menu, prior-experience rendering and "
        "how_to_read (_public_input)",
        "workspace tool budget",
        "stage note, except for the one appended sentence below",
        "harness view: the same h0 snapshot resolved for role=fast",
        "offline scoring: the same OfflinePlanEvaluator",
    ],
    "deliberate_differences": [
        "tool supply: no batch_recipe binding, so allowed_local_tools carries "
        "the two observation tools only",
        "decision enum gains OWN_OBSERVATION, and the stage note gains one "
        "sentence naming it, because the tool arm's three values leave an "
        "Agent with no tool and no prior entry unable to state a treatment "
        "plan honestly",
    ],
    "not_done": [
        "no hint that a recipe exists, no suggestion of a program, no "
        "suggestion to exclude any series, no threshold and no target",
    ],
}

# The tool arm's schema with exactly one enum value added.
NOTOOL_BATCH_PLAN_SCHEMA: dict[str, Any] = json.loads(
    json.dumps(toolarm.BATCH_PLAN_SCHEMA)
)
NOTOOL_BATCH_PLAN_SCHEMA["$id"] = "batch-plan/1-own-observation"
NOTOOL_BATCH_PLAN_SCHEMA["properties"]["decision"]["enum"] = [
    "ADOPT_TOOL_RESULT",
    "REUSE_PRIOR_EXPERIENCE",
    "IDENTITY_NO_TREATMENT",
    "OWN_OBSERVATION",
]
SCHEMA_NAME = "batch_plan_v1_own_observation"

NOTOOL_STAGE_NOTE = toolarm.STAGE_NOTE + (
    " Use OWN_OBSERVATION when you formed the plan yourself from what you "
    "observed in this stage."
)


class NoRecipeBatchPlanGateway(CohortScopePublicToolGateway):
    """The frozen ``binding=None`` supply, served on the batch_plan stage.

    The gateway widens its stage set only when a recipe binding is present, so
    a control arm asking for ``batch_plan`` with no binding would be served no
    tool at all -- not even the two observation tools the tool arm had.  The
    stage name is looked up as ``inspect`` instead, which is what the frozen
    class already answers with the same two tools.  Nothing else is overridden:
    the argument wall, the budget, the refusals, ``context_sha`` and
    ``accounting`` are the frozen ones, and ``batch_recipe`` remains an
    undeclared tool that the stage loop rejects before it can reach ``call``.
    """

    def schemas_for(self, *, role: Any, stage: str) -> tuple[Mapping[str, Any], ...]:
        lookup = "inspect" if str(stage) == STAGE else str(stage)
        return super().schemas_for(role=role, stage=lookup)


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
    """The tool arm's Experience entry with the provenance tag swapped.

    Same episode mechanism, same fields, same relation rule, same
    ``counts_as_unguided_exploration: false``.  Only the tag differs, so a
    later audit can tell a tool-mediated measurement from a hand-rolled one
    without reading anything else.
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
            "engineering measurement of a plan the Agent formed from the "
            "public prefixes with no recipe tool bound; not authorization "
            "evidence and not an unguided probe"
        ),
    }
    return build_episode(
        episode_id=episode_id,
        task_consumer_key=toolarm._cell_key(cohort, consumer_variant),
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
                "already-exposed development origins; scored offline after the "
                "fact and never shown to the Agent in this arm"
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
def _run_episode(
    *,
    episode_id: str,
    cohort: str,
    consumer_variant: str,
    request_note: str,
    evaluator: Any,
    episodes: list[Any],
    snapshot: Any,
    llm_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    cell_key = toolarm._cell_key(cohort, consumer_variant)
    cell = evaluator._cell(cohort, consumer_variant)
    spec, support, delayed = bch._task_windows(cohort, TASK_INDEX)
    cutoff = int(support[0])
    train_uids = list(cell["train_uids"])

    gateway = NoRecipeBatchPlanGateway(
        {
            uid: np.asarray(cell["values"][uid], dtype=np.float64)[:cutoff]
            for uid in train_uids
        },
        task_kind="forecast",
        observation_cutoff=cutoff,
        maximum_calls=WORKSPACE_TOOL_BUDGET,
    )
    pack = SignedEpisodeRetriever(
        tuple(episodes), task_consumer_key=cell_key
    ).retrieve(toolarm._retrieval_context(cohort, consumer_variant), cohort)
    experience_rows = [
        toolarm._experience_row(episode, current_cell=cell_key)
        for episode in episodes
    ]
    public_input = dict(
        toolarm._public_input(
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
    )
    # The only edited field: one appended sentence naming OWN_OBSERVATION.
    public_input["stage_note"] = NOTOOL_STAGE_NOTE

    backend = _default_backend_factory(int(llm_budget))
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)
    harness_view = resolve_harness_view(snapshot, {}, role="fast")
    validator = toolarm._make_post_validator(
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
            case_id="BRMNT_%s" % episode_id,
            public_input=public_input,
            harness_view=harness_view,
            output_schema_name=SCHEMA_NAME,
            output_schema=NOTOOL_BATCH_PLAN_SCHEMA,
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
            "arguments": toolarm._plain(receipt.arguments),
            "ok": bool(receipt.ok),
            "public_result_keys": sorted(receipt.public_result),
            "receipt_sha": receipt.receipt_sha,
        }
        for receipt in (result.tool_receipts if result is not None else ())
    ]
    return {
        "episode_id": episode_id,
        "arm": "no_tool",
        "cohort": cohort,
        "consumer_variant": consumer_variant,
        "cell_key": cell_key,
        "request": request_note,
        "payload": toolarm._plain(payload) if payload is not None else None,
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
        "observation_tool_calls": sum(1 for row in receipts if row["ok"]),
        "declared_tools": [
            str(schema["name"])
            for schema in gateway.schemas_for(role="fast", stage=STAGE)
        ],
        "workspace_tool_accounting": gateway.accounting(),
        "experience_entries_visible": len(experience_rows),
        "signed_pack_for_this_cell": pack.to_dict(),
        "wall_seconds": time.perf_counter() - started,
    }


# ------------------------------------------------------------------ labelling
def _label_episode(
    record: Mapping[str, Any], episodes: Sequence[Any]
) -> dict[str, Any]:
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
    reused = decision == "REUSE_PRIOR_EXPERIENCE" and bool(cited)
    if decision == "OWN_OBSERVATION":
        behaviour = (
            "HAND_ROLLED_IDENTITY" if plan["program"] == bch.IDENTITY
            else "HAND_ROLLED_PLAN"
        )
    elif reused:
        behaviour = "EXPERIENCE_REUSED"
    elif decision == "IDENTITY_NO_TREATMENT":
        behaviour = "IDENTITY_NO_TREATMENT"
    elif decision == "ADOPT_TOOL_RESULT":
        behaviour = "CLAIMED_TOOL_RESULT_WITHOUT_TOOL"
    else:
        behaviour = "OTHER_DECISION"
    return {
        "behaviour": behaviour,
        "decision": decision,
        "reused": reused,
        "reused_across_cell": reused and any(
            cell != record["cell_key"] for cell in cited_cells
        ),
        "cited_episode_ids": cited,
        "cited_cells": cited_cells,
        "plan": plan,
    }


def _tool_arm_rows() -> dict[str, dict[str, Any]]:
    """The frozen tool-arm artifact, indexed by episode.  Read-only."""
    data = json.loads(TOOL_ARM_ARTIFACT.read_text(encoding="utf-8"))
    rows: dict[str, dict[str, Any]] = {}
    for record in data["episodes"]:
        offline = record.get("adopted_plan_offline_readout") or {}
        rows[str(record["episode_id"])] = {
            "behaviour": record["label"]["behaviour"],
            "decision": record["label"]["decision"],
            "plan": record["label"]["plan"],
            "support_aggregate_gain": offline.get("support_aggregate_gain"),
            "delayed_aggregate_gain": offline.get("delayed_aggregate_gain"),
            "support_harmed_eval_series_count": offline.get(
                "support_harmed_eval_series_count"
            ),
            "delayed_harmed_eval_series_count": offline.get(
                "delayed_harmed_eval_series_count"
            ),
            "llm_calls": record["llm_calls"],
            "tool_calls": sum(
                1 for row in record["tool_receipts"] if row.get("ok")
            ),
            "tool_calls_refused": sum(
                1 for row in record["tool_receipts"] if not row.get("ok")
            ),
        }
    return {
        "protocol_version": data["protocol_version"],
        "verdict": data["verdict"]["verdict"],
        "llm_call_count": data["llm_call_count"],
        "episodes": rows,
    }


def _compare(
    records: Sequence[Mapping[str, Any]], tool_arm: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    outcomes: list[str] = []
    for record in records:
        episode_id = str(record["episode_id"])
        tool = tool_arm["episodes"].get(episode_id)
        offline = record.get("adopted_plan_offline_readout") or {}
        mine = offline.get("delayed_aggregate_gain")
        theirs = None if tool is None else tool.get("delayed_aggregate_gain")
        if mine is None or theirs is None:
            outcome = "UNREADABLE"
            margin = None
        else:
            margin = float(theirs) - float(mine)
            if margin > MATERIAL_THRESHOLD:
                outcome = "TOOL"
            elif margin < -MATERIAL_THRESHOLD:
                outcome = "NOTOOL"
            else:
                outcome = "TIE"
        if outcome != "UNREADABLE":
            outcomes.append(outcome)
        rows.append({
            "episode_id": episode_id,
            "cell_key": record["cell_key"],
            "tool_arm": tool,
            "no_tool_arm": {
                "behaviour": record["label"]["behaviour"],
                "decision": record["label"]["decision"],
                "plan": record["label"]["plan"],
                "support_aggregate_gain": offline.get("support_aggregate_gain"),
                "delayed_aggregate_gain": offline.get("delayed_aggregate_gain"),
                "support_harmed_eval_series_count": offline.get(
                    "support_harmed_eval_series_count"
                ),
                "delayed_harmed_eval_series_count": offline.get(
                    "delayed_harmed_eval_series_count"
                ),
                "llm_calls": record["llm_calls"],
                "tool_calls": int(record["observation_tool_calls"]),
                "tool_calls_refused": sum(
                    1 for row in record["tool_receipts"] if not row.get("ok")
                ),
            },
            "delayed_margin_tool_minus_no_tool": margin,
            "outcome": outcome,
        })
    if not outcomes:
        verdict = "MIXED_BY_EPISODE"
        reason = "no episode was comparable on the delayed column"
    elif all(item == "TOOL" for item in outcomes):
        verdict = "TOOL_ARM_DOMINATES"
        reason = "every compared episode favours the tool arm by more than %.3f" % (
            MATERIAL_THRESHOLD
        )
    elif not any(item == "TOOL" for item in outcomes):
        verdict = "HAND_ROLLED_COMPETITIVE"
        reason = (
            "no compared episode favours the tool arm by more than %.3f"
            % MATERIAL_THRESHOLD
        )
    else:
        verdict = "MIXED_BY_EPISODE"
        reason = "the compared episodes split: %s" % ", ".join(
            "%s=%s" % (row["episode_id"], row["outcome"]) for row in rows
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "outcomes": outcomes,
        "rows": rows,
        "rule": PRE_REGISTERED["verdict_rules"],
        "material_threshold": MATERIAL_THRESHOLD,
        "column_used": "delayed aggregate gain, recomputed offline for both arms",
    }


# --------------------------------------------------------------- orchestration
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    if not TOOL_ARM_ARTIFACT.is_file():
        raise SystemExit(
            "the tool arm artifact %s is missing; this control arm is only "
            "readable against it" % TOOL_ARM_ARTIFACT
        )
    tool_arm = _tool_arm_rows()
    snapshot = compile_snapshot(
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False
    )
    evaluator = toolarm.OfflinePlanEvaluator()
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
            "NOTOOL %s %s x %s (llm used %d/%d)"
            % (episode_id, cohort, consumer_variant, llm_used,
               LLM_CALL_BUDGET_TOTAL),
            flush=True,
        )
        record = _run_episode(
            episode_id=episode_id,
            cohort=cohort,
            consumer_variant=consumer_variant,
            request_note=request_note,
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
            "NOTOOL %s behaviour=%s obs_tools=%d llm=%d plan=%s delayed=%s"
            % (
                episode_id, record["label"]["behaviour"],
                record["observation_tool_calls"], record["llm_calls"], plan,
                "n/a" if offline is None
                else "%+.6f" % offline["delayed_aggregate_gain"],
            ),
            flush=True,
        )
        if episode_id == "E1" and record["payload"] is None:
            stopped_reason = "circuit breaker: E1 produced no payload"
            break

    comparison = _compare(records, tool_arm)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "arm": "no_tool control",
        "role": (
            "control arm for agent_recipe_mount_v1: the same three requests "
            "with no batch_recipe binding, to price what the mounted recipe "
            "contributes to plan quality"
        ),
        "not_authorization_evidence": (
            "no Skill is written, no TRY right is granted, no Episode is "
            "promoted, no Fast or Slow update runs, no snapshot pointer moves"
        ),
        "verdict": comparison["verdict"],
        "verdict_reason": comparison["reason"],
        "comparison": comparison,
        "tool_arm_reference": {
            "artifact": TOOL_ARM_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
            "protocol_version": tool_arm["protocol_version"],
            "verdict": tool_arm["verdict"],
            "llm_call_count": tool_arm["llm_call_count"],
            "re_run": False,
            "note": "read verbatim; the tool arm was not re-run",
        },
        "prompt_parity": PROMPT_PARITY,
        "pre_registered": PRE_REGISTERED,
        "harness_change_surface": {
            "surface": "none",
            "note": (
                "no Harness file is edited by this arm. The gateway is used "
                "through its frozen binding=None path; the runner subclasses "
                "it only to look the batch_plan stage up as inspect, which "
                "serves exactly the frozen two-tool supply"
            ),
        },
        "model": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "declared_tools_this_arm": (
            records[0]["declared_tools"] if records else []
        ),
        "batch_plan_schema": NOTOOL_BATCH_PLAN_SCHEMA,
        "stage_note": NOTOOL_STAGE_NOTE,
        "llm_call_count": llm_used,
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "stopped_early": stopped_reason,
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
        print(json.dumps(comparison, indent=2, ensure_ascii=False, default=str))
        return 0
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict", comparison["verdict"], flush=True)
    print("llm_calls", llm_used, flush=True)
    return 0


# --------------------------------------------------------------------- report
def _gain(value: Any) -> str:
    return "n/a" if value is None else "%+.6f" % float(value)


def _plan_text(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "n/a"
    return "`%s` minus %s" % (
        plan["program"], ", ".join(plan["excluded_series"]) or "nothing",
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    comparison = payload["comparison"]
    reference = payload["tool_arm_reference"]
    lines: list[str] = [
        "# agent-recipe-mount, no-tool control arm v1",
        "",
        "**Verdict: `%s`**" % payload["verdict"],
        "",
        "The tool arm measured behaviour: the Agent routed to the mounted "
        "`batch_recipe`, reused its Experience entry on the same cell and "
        "searched again when the Consumer changed. This arm prices the other "
        "half. Same three requests, same schema, same prompt, no binding: the "
        "Agent has only `summarize_series` and `localize_regions`, and every "
        "plan it returns is its own.",
        "",
        "**Engineering demonstration, not authorization evidence.** No Skill "
        "is written, no TRY right is granted, no Episode is promoted, no Fast "
        "or Slow update runs, and no snapshot pointer moves.",
        "",
        "The tool arm was **not re-run**: `%s` (`%s`, verdict `%s`, %d LLM "
        "calls) is read verbatim for the comparison."
        % (
            reference["artifact"], reference["protocol_version"],
            reference["verdict"], reference["llm_call_count"],
        ),
        "",
        "## 0. What differs from the tool arm, and what does not",
        "",
        "Identical, and imported from the tool-arm runner rather than copied:",
        "",
    ]
    for item in payload["prompt_parity"]["identical_and_imported_from_the_tool_arm"]:
        lines.append("- %s" % item)
    lines += ["", "Deliberate differences:", ""]
    for item in payload["prompt_parity"]["deliberate_differences"]:
        lines.append("- %s" % item)
    lines += [
        "",
        "Not done: %s"
        % "; ".join(payload["prompt_parity"]["not_done"]),
        "",
        "Tools declared to the Agent in this arm: %s."
        % (
            ", ".join("`%s`" % name for name in payload["declared_tools_this_arm"])
            or "none"
        ),
        "",
        "Harness change surface: **%s**. %s."
        % (
            payload["harness_change_surface"]["surface"],
            payload["harness_change_surface"]["note"],
        ),
        "",
        "## 1. Verdict",
        "",
        "Pre-registered before the first LLM call. Outcome per episode is read "
        "off the %s; `TOOL` if the tool arm leads by more than %.3f, `NOTOOL` "
        "if the control arm does, `TIE` otherwise."
        % (comparison["column_used"], comparison["material_threshold"]),
        "",
    ]
    for index, rule in enumerate(comparison["rule"], start=1):
        lines.append("%d. %s" % (index, rule))
    lines += [
        "",
        "Matched: **`%s`** -- %s." % (payload["verdict"], comparison["reason"]),
        "",
    ]
    if payload["stopped_early"]:
        lines += ["Stopped early: %s." % payload["stopped_early"], ""]
    lines += [
        "## 2. Arm comparison, episode by episode",
        "",
        "| episode | arm | behaviour | plan | support | delayed | harmed "
        "(support/delayed) | LLM | tool calls (refused) |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        for name, side in (("tool", row["tool_arm"]), ("no tool", row["no_tool_arm"])):
            if side is None:
                lines.append(
                    "| %s | %s | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                    % (row["episode_id"], name)
                )
                continue
            lines.append(
                "| %s | %s | `%s` | %s | %s | %s | %s / %s | %d | %s |"
                % (
                    row["episode_id"], name, side["behaviour"],
                    _plan_text(side["plan"]),
                    _gain(side["support_aggregate_gain"]),
                    _gain(side["delayed_aggregate_gain"]),
                    side["support_harmed_eval_series_count"],
                    side["delayed_harmed_eval_series_count"],
                    side["llm_calls"],
                    "%d (%d)" % (
                        side["tool_calls"], side.get("tool_calls_refused") or 0
                    ),
                )
            )
        lines.append(
            "| %s | **margin** | `%s` | delayed tool - no tool | | %s | | | |"
            % (row["episode_id"], row["outcome"],
               _gain(row["delayed_margin_tool_minus_no_tool"]))
        )
    lines += ["", "## 3. What the no-tool arm actually proposed", ""]
    for record in payload["episodes"]:
        label = record["label"]
        offline = record["adopted_plan_offline_readout"] or {}
        lines += [
            "### %s -- %s x %s" % (
                record["episode_id"], record["cohort"],
                record["consumer_variant"],
            ),
            "",
            "- decision: `%s` (behaviour `%s`)"
            % (label["decision"], label["behaviour"]),
            "- plan: %s" % _plan_text(label["plan"]),
            "- offline readout: support %s, delayed %s, harmed evaluation "
            "series %s (support) / %s (delayed)"
            % (
                _gain(offline.get("support_aggregate_gain")),
                _gain(offline.get("delayed_aggregate_gain")),
                offline.get("support_harmed_eval_series_count"),
                offline.get("delayed_harmed_eval_series_count"),
            ),
            "- observation tool calls: %d %s"
            % (
                record["observation_tool_calls"],
                [row["arguments"] for row in record["tool_receipts"]],
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
            "- LLM calls: %d, retries: %s %s"
            % (
                record["llm_calls"], record["validation_retry_count"],
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
    lines: list[str] = [
        "## 4. Experience entries written",
        "",
        "Same episode mechanism as the tool arm, no new store, no Skill. The "
        "provenance tag is deliberately different -- `%s` here against "
        "`batch_recipe_tool_v2_engineering` there -- so a later audit can tell "
        "a tool-mediated measurement from a hand-rolled one without reading "
        "anything else. Both carry "
        "`counts_as_unguided_exploration: false`."
        % payload["experience_provenance"],
        "",
        "| episode_id | task_consumer_key | plan | support | delayed | "
        "relation | local_status | provenance |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for entry in payload["experience_entries"]:
        support = entry["support_response"]
        delayed = entry["delayed_response"]
        lines.append(
            "| `%s` | `%s` | `%s` minus %s | %s | %s | %s | %s | `%s` |"
            % (
                entry["episode_id"],
                str(entry["task_consumer_key"]).replace("|", r"\|"),
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
        "## 5. Cost",
        "",
        "| item | this arm | tool arm |",
        "| --- | ---: | ---: |",
        "| LLM calls | %d (budget %d) | %d |"
        % (
            payload["llm_call_count"], payload["llm_call_budget_total"],
            payload["tool_arm_reference"]["llm_call_count"],
        ),
        "| recipe searches executed | 0 (no binding) | 2 |",
        "| wall seconds | %.1f | -- |" % payload["wall_seconds"],
        "",
        "## 6. What this does not say",
        "",
        "- It does not authorize anything, in either arm.",
        "- It is one cohort and three episodes on one model. A per-episode "
        "outcome is a comparison of two single draws, not a rate.",
        "- The delayed column is in-selection for the tool arm, because the "
        "recipe's own adoption gate reads it. For this arm it is out of "
        "selection: the Agent never saw a delayed number. That asymmetry is a "
        "property of the mounted tool and it favours the tool arm.",
        "- It does not measure whether a hand-rolled plan would improve with "
        "a larger Workspace tool budget or more episodes.",
        "",
        "## Provenance",
        "",
        "- model: `%s` at `%s`" % (
            payload["model"]["model"], payload["model"]["base_url"],
        ),
        "- scoring: the tool arm's `OfflinePlanEvaluator`, i.e. "
        "`run_batch_composition_headroom._evaluate_assignment` + `_gain_rows` "
        "on the same windows and Consumer variant, imported not reimplemented",
        "- windows: %s" % payload["windows"]["note"],
        "- stage: `%s`, schema `%s` declared inside this runner"
        % (payload["batch_plan_schema"]["$id"], payload["batch_plan_schema"]["$id"]),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run the arm but print the comparison instead of writing artifacts",
    )
    args = parser.parse_args(argv)
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
