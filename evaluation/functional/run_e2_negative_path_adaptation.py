"""negative-path adaptation: does the Agent learn to stop, and only there?

T233 x per_channel is an established low-headroom cell.  Its W3 full search
settles on ``identity``; the plan its W1 run adopted scores +0.0017 Support on
W2 and -0.0010 on W3, both under the 0.005 material line.  The closing question
of this line is whether an Agent that keeps failing to re-confirm a reused plan
**converges to abstention and stops spending evaluation budget**, and whether
that abstention stays local instead of generalising to a cell that has real
headroom.

Four episodes, run in order:

* **E1** the same cell on W2.  Visible experience: that cell's frozen W1 row
  only.  The W2 and W3 rows are the scoring reference and are never shown.
* **E2** the same cell on W3.  Visible: the W1 row plus everything E1 produced,
  including a failed confirmation if that is what happened.
* **E3** the same cell on W4 (``e1v2_task_04`` roster origins, quoted).  The
  runner computes W4's own full-recipe reference offline first, with no LLM.
  Visible: W1 plus E1 and E2.
* **E4**, the control: traffic x per_channel on W3, a cell whose full search
  reaches +0.387.  Visible: that cell's frozen W1 and W2 rows plus all three
  earlier episodes, so an Agent that has learned "abstain" has every chance to
  over-apply it.

Two deliberate departures from the earlier warm/cold runs, both pre-registered:

1. **within-run feedback is allowed.**  E_k's outcome enters E_{k+1}'s visible
   experience.  That is the mechanism under test -- learning from one's own
   failures -- not a leak;
2. **the adoption stage gains an identity-incumbent gate**, closing the hole the
   rotation exposed, where both arms adopted a delayed-negative plan on this
   very cell while ``identity`` sat unused in the measured list.  A plan is
   adopted only if its delayed aggregate gain is at least
   ``max(best evaluated full-batch delayed, 0)``; otherwise the episode falls
   back to identity.  The Agent never sees a delayed number; the gate is a
   Harness-side backstop and is reported separately from what the Agent chose.

Each episode the Agent picks one of three actions: re-confirm a plan it can see
in experience, which costs one evaluation and is confirmed only at Support
>= 0.005; search, which costs one evaluation per shortlisted program, at most
two, plus a free mask round on the highest-Support one; or abstain outright,
which costs nothing.  So the evaluation count is itself the behavioural signal.

Verdict rules, fixed before the first call and applied in this order:

1. ``OVERGENERALIZED_ABSTENTION`` -- E4 abstained on the high-headroom cell;
2. ``ADAPTIVE_ABSTENTION_CONVERGES`` -- E1..E3 evaluation counts are
   monotonically non-increasing, E3 ends at identity, E3 cites at least one
   earlier negative or marginal record, and E4 adopts a delayed-positive plan;
3. ``GATE_SAVES_BUT_NO_LEARNING`` -- the identity gate had to force the
   fallback in two or more episodes;
4. ``NO_BEHAVIOR_CHANGE`` -- E2 and E3 both spent the full two evaluations and
   neither abstained;
5. ``MIXED`` -- anything else, reported episode by episode without merging.

Not authorization evidence: no Skill, no TRY right, no promotion, no Fast or
Slow path.  Experience rows carry
``provenance="negative_path_engineering"``.

Run:

    python evaluation/functional/run_e2_negative_path_adaptation.py

Writes ``artifacts/functional/e2/negative_path_adaptation_v1.json`` and ``.md``.
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

import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402

from evaluation.functional.task_episode_harness.agentic.runner import (  # noqa: E402
    _default_backend_factory,
)
from evaluation.functional.task_episode_harness.e1 import (  # noqa: E402
    _frozen_task_roster,
)
from evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
    NF_MODEL,
)
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_sha256,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
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
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)

PROTOCOL_VERSION = "negative_path_adaptation_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "negative_path_adaptation_v1.json"
OUT_MD = E2 / "negative_path_adaptation_v1.md"
WINDOWS_ARTIFACT = E2 / "batch_recipe_windows_v1.json"

EXPERIENCE_PROVENANCE = "negative_path_engineering"
TREATMENTS = wvc.TREATMENTS
IDENTITY = wvc.IDENTITY

EVALUATION_BUDGET = 2
CONFIRMATION_LINE = 0.005
LLM_CALL_BUDGET_TOTAL = 30
LLM_CALL_BUDGET_PER_EPISODE = 5
VALIDATION_RETRIES = wvc.VALIDATION_RETRIES

LOW_HEADROOM_CELL = {"cohort": "T233", "consumer_variant": bch.CONSUMER_PER_CHANNEL}
CONTROL_CELL = {"cohort": "traffic", "consumer_variant": bch.CONSUMER_PER_CHANNEL}

PRE_REGISTERED = {
    "fixed_before_the_first_llm_call": True,
    "question": (
        "does the Agent converge to abstention after repeated failed "
        "re-confirmation on a low-headroom cell, and does the abstention stay "
        "local"
    ),
    "deliberate_departures_from_the_warm_cold_runs": [
        "within-run feedback is allowed: episode k's outcome is visible to "
        "episode k+1, because learning from one's own failures is the "
        "mechanism under test",
        "the adoption stage gains an identity-incumbent gate: a plan is "
        "adopted only if its delayed aggregate gain is at least "
        "max(best evaluated full-batch delayed, 0); otherwise the episode "
        "falls back to identity",
    ],
    "actions_and_their_cost": {
        "REUSE_CONFIRM": (
            "one charged evaluation: the Support of a plan named from visible "
            "experience, measured on this window. Confirmed only at Support "
            ">= %.3f" % CONFIRMATION_LINE
        ),
        "SEARCH": (
            "one charged evaluation per shortlisted program, at most %d, plus "
            "a free greedy mask round on the highest-Support one"
            % EVALUATION_BUDGET
        ),
        "ABSTAIN_IDENTITY": "no evaluation is charged and no second stage runs",
    },
    "identity_incumbent_gate": (
        "bar = max(delayed of every evaluated full-batch plan, 0.0). A plan "
        "whose delayed gain is below the bar is not adopted; the episode falls "
        "back to identity. In the reuse path no full-batch plan is evaluated, "
        "so the bar is identity at zero. The Agent never sees a delayed number"
    ),
    "verdict_rules_in_this_order": [
        "OVERGENERALIZED_ABSTENTION: E4 abstained on the high-headroom control "
        "cell",
        "ADAPTIVE_ABSTENTION_CONVERGES: E1..E3 charged-evaluation counts are "
        "monotonically non-increasing, E3 ends at identity, E3 cites at least "
        "one earlier negative or marginal record, and E4 adopts a "
        "delayed-positive plan",
        "GATE_SAVES_BUT_NO_LEARNING: the identity gate forced the fallback in "
        "two or more episodes",
        "NO_BEHAVIOR_CHANGE: E2 and E3 both spent the full %d evaluations and "
        "neither abstained" % EVALUATION_BUDGET,
        "MIXED: anything else, reported episode by episode without merging",
    ],
    "behaviour_is_read_from_the_payload": (
        "the action, the shortlist, the cited record ids and the adopted plan "
        "are read from the stage payloads and the instrument's own log; the "
        "Agent's prose is quoted in the report but never scored"
    ),
    "circuit_breaker": "stop and report if the first episode produces no "
    "stage-one payload",
    "budget": {
        "charged_evaluations_per_episode": EVALUATION_BUDGET,
        "confirmation_line_support": CONFIRMATION_LINE,
        "llm_calls_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_per_episode": LLM_CALL_BUDGET_PER_EPISODE,
        "validation_retries_per_stage": VALIDATION_RETRIES,
        "mask_round": "free, and still runs on the highest-Support shortlisted "
        "program (frozen, unchanged)",
        "delayed_readings": "free, taken after the plan is named, and never "
        "shown to the Agent",
    },
    "origins": (
        "W2 and W3 are quoted from batch_recipe_windows_v1; W4 is the frozen "
        "e1v2_task_04 roster spec, quoted; no origin is chosen here"
    ),
}


class NegativePathSearch(wvc.BudgetedSearch):
    """The inherited instrument; this run's budget label and nothing else."""

    def accounting(self) -> dict[str, Any]:
        row = super().accounting()
        row["budget"] = EVALUATION_BUDGET
        row["budget_source"] = PROTOCOL_VERSION
        return row


# ------------------------------------------------------- windows and rosters
def _windows_artifact() -> dict[str, Any]:
    return json.loads(WINDOWS_ARTIFACT.read_text(encoding="utf-8"))


def _quoted_window(
    cohort: str, consumer_variant: str, window_id: str
) -> dict[str, Any]:
    """Origins and full-search reference, quoted from the windows artifact."""
    window = wvc._target_window({
        "cohort": cohort,
        "consumer_variant": consumer_variant,
        "window_id": window_id,
    })
    window["reference_source"] = (
        WINDOWS_ARTIFACT.relative_to(PROJECT_ROOT).as_posix()
    )
    return window


def _roster_window_4() -> dict[str, Any]:
    """W4 for the low-headroom cell: the frozen e1v2_task_04 spec, quoted.

    Its full-recipe reference does not exist yet, so this runner computes it
    once with the frozen v2 recipe.  Zero LLM calls; the Agent never sees it.
    """
    spec = _frozen_task_roster()[3]
    support = [int(origin) for origin in spec["support_origins"]]
    delayed = [int(origin) for origin in spec["delayed_origins"]]
    print(
        "NPA computing the W4 reference offline (0 LLM): %s %s"
        % (support, delayed),
        flush=True,
    )
    recipe = bch.make_batch_recipe(
        LOW_HEADROOM_CELL["cohort"],
        task_index=3,
        consumer_variant=LOW_HEADROOM_CELL["consumer_variant"],
        adoption_rule_version="v2",
    )
    plan = recipe["adopted_plan"]
    return {
        "window_id": "W4",
        "support_origins": support,
        "delayed_origins": delayed,
        "origin_source": "quoted from the frozen roster",
        "origin_provenance": (
            "task_episode_harness.e1._frozen_task_roster()[3], "
            "e1v2_task_04, support and delayed origins verbatim"
        ),
        "quoted_from": "frozen e1v2 Task roster",
        "reference_plan": {
            "kind": str(plan["kind"]),
            "program": str(plan["program"]),
            "excluded_series": list(plan["excluded_series"]),
        },
        "reference_support_aggregate_gain": float(
            recipe["comparison"]["support"]["adopted"]
        ),
        "reference_delayed_aggregate_gain": float(
            recipe["comparison"]["delayed"]["adopted"]
        ),
        "reference_note": (
            "computed by this runner with the frozen v2 recipe, 0 LLM calls, "
            "because no earlier artifact covers W4 on this cell"
        ),
        "reference_adoption_path": str(recipe["adoption_path"]),
        "reference_menu_scan": {
            str(program): float(row["aggregate_gain"])
            for program, row in recipe["menu_scan"].items()
        },
        "reference_wall_seconds": float(recipe["wall_seconds"]),
    }


def _frozen_experience_rows(
    corpus: Mapping[str, Any],
    wanted: Sequence[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Frozen corpus rows, selected by (cohort, consumer_variant, window)."""
    index = {
        (str(row["cohort"]), str(row["consumer_variant"]),
         str(row["window_id"])): row
        for row in corpus["rows"]
    }
    rows: list[dict[str, Any]] = []
    for key in wanted:
        row = index.get(key)
        if row is None:
            raise RuntimeError("frozen corpus has no row for %s" % (key,))
        rows.append({**dict(row), "source": "frozen record"})
    return rows


def _run_experience_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """One earlier episode of this run, as the next episode sees it.

    The confirmation result and the gate outcome are carried explicitly: a plan
    that failed to re-confirm, or that the identity gate refused, is exactly
    the evidence this experiment is about.
    """
    plan = record["final_plan"]
    confirmation = record.get("confirmation")
    gate = record.get("identity_incumbent_gate") or {}
    return {
        "record_id": "run|%s" % record["episode_id"],
        "source": "this run",
        "episode_id": record["episode_id"],
        "cohort": record["cohort"],
        "consumer_variant": record["consumer_variant"],
        "cell_key": "batch:%s|consumer:%s"
        % (record["cohort"], record["consumer_variant"]),
        "window_id": record["window_id"],
        "support_origins": list(record["support_origins"]),
        "delayed_origins": list(record["delayed_origins"]),
        "action_taken": record["action"],
        "charged_evaluations": record["evaluations_used"],
        "reuse_confirmation": (
            None if confirmation is None else {
                "plan": confirmation["plan"],
                "support_aggregate_gain": confirmation[
                    "support_aggregate_gain"],
                "confirmation_line": CONFIRMATION_LINE,
                "confirmed": confirmation["confirmed"],
            }
        ),
        "plan_the_agent_named": record.get("agent_plan"),
        "adopted_plan": plan,
        "identity_incumbent_gate": {
            "bar": gate.get("bar"),
            "adopted_plan_passed": gate.get("passed"),
            "fell_back_to_identity": gate.get("fell_back_to_identity"),
        },
        "support_aggregate_gain": record["support"]["aggregate_gain"],
        "delayed_aggregate_gain": record["delayed"]["aggregate_gain"],
        "harmed_eval_series_count": record["support"][
            "harmed_eval_series_count"],
        "relation": record["relation"],
    }


# --------------------------------------------------------------- stage wiring
ACTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "negative-path-action/1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "action", "reuse_program", "reuse_excluded_series", "shortlist",
        "request_mask_search", "experience_use", "reason",
    ],
    "properties": {
        "action": {
            "enum": ["REUSE_CONFIRM", "SEARCH", "ABSTAIN_IDENTITY"]
        },
        "reuse_program": {"enum": list(TREATMENTS) + [""]},
        "reuse_excluded_series": {
            "type": "array", "items": {"type": "string"},
            "uniqueItems": True, "maxItems": 24,
        },
        "shortlist": {
            "type": "array", "items": {"enum": list(TREATMENTS)},
            "uniqueItems": True, "maxItems": EVALUATION_BUDGET,
        },
        "request_mask_search": {"type": "boolean"},
        "experience_use": {
            "type": "array", "items": {"type": "string"}, "maxItems": 8,
        },
        "reason": {"type": "string"},
    },
}
ADOPTION_SCHEMA = wvc.ADOPTION_SCHEMA

ACTION_NOTE = (
    "Decide how to spend this window's evaluation budget, then stop. `action` "
    "is one of three. REUSE_CONFIRM re-checks a plan you can see in "
    "`prior_experience`: name it in `reuse_program` and `reuse_excluded_series` "
    "exactly as it appears there, it costs one evaluation, and it counts as "
    "confirmed only if its Support aggregate gain on this window is at least "
    "%.3f. SEARCH names one or two programs in `shortlist`, one evaluation "
    "each; `request_mask_search` then asks for a free greedy exclusion round on "
    "whichever of them scores highest on Support. ABSTAIN_IDENTITY treats "
    "nothing and spends no evaluation. Leave the fields of the actions you are "
    "not taking empty. `experience_use` lists the `record_id`s you are relying "
    "on, if any. `reason` is one or two sentences in public terms. You will see "
    "the measured numbers next and then name the plan; the delayed window is "
    "never shown to you." % CONFIRMATION_LINE
)

ADOPTION_NOTE = (
    "The measurements are in. Name the plan to adopt: `program` and "
    "`excluded_series` must be exactly one of the entries in `measured_plans`, "
    "which is everything the instrument actually measured, `identity` "
    "included. `reason` is one or two sentences in public terms. A plan whose "
    "delayed reading turns out to be below the identity incumbent will not be "
    "adopted -- the episode falls back to identity -- and you are not shown "
    "that reading before choosing."
)


def _make_action_validator(visible_plans, visible_ids):
    plans = {
        (str(program), tuple(sorted(str(uid) for uid in excluded)))
        for program, excluded in visible_plans
    }
    known = {str(item) for item in visible_ids}

    def validate(payload: Mapping[str, Any]) -> None:
        action = str(payload["action"])
        program = str(payload["reuse_program"])
        excluded = tuple(sorted(
            str(uid) for uid in payload.get("reuse_excluded_series", ())
        ))
        shortlist = [str(item) for item in payload.get("shortlist", ())]
        cited = [str(item) for item in payload.get("experience_use", ())]
        unknown = sorted(set(cited) - known)
        if unknown:
            raise StagePostValidationError(
                "EXPERIENCE_CITATION_UNGROUNDED",
                "experience_use names record ids that are not in "
                "prior_experience: %s" % unknown,
                retryable=True,
            )
        if action == "REUSE_CONFIRM":
            if not program:
                raise StagePostValidationError(
                    "REUSE_PLAN_MISSING",
                    "REUSE_CONFIRM needs reuse_program and "
                    "reuse_excluded_series naming a plan from prior_experience",
                    retryable=True,
                )
            if (program, excluded) not in plans:
                raise StagePostValidationError(
                    "REUSE_PLAN_NOT_IN_EXPERIENCE",
                    "the named plan is not one of the plans in "
                    "prior_experience; re-confirm a plan you can actually see",
                    retryable=True,
                )
            if shortlist:
                raise StagePostValidationError(
                    "ACTION_FIELDS_INCONSISTENT",
                    "REUSE_CONFIRM spends its evaluation on the reused plan, "
                    "so shortlist must be empty",
                    retryable=True,
                )
        elif action == "SEARCH":
            if not shortlist:
                raise StagePostValidationError(
                    "SHORTLIST_EMPTY",
                    "SEARCH needs one or two programs in shortlist",
                    retryable=True,
                )
            if program or excluded:
                raise StagePostValidationError(
                    "ACTION_FIELDS_INCONSISTENT",
                    "SEARCH does not re-confirm a plan, so reuse_program must "
                    "be empty and reuse_excluded_series must be empty",
                    retryable=True,
                )
        else:
            if shortlist or program or excluded:
                raise StagePostValidationError(
                    "ACTION_FIELDS_INCONSISTENT",
                    "ABSTAIN_IDENTITY spends nothing, so shortlist, "
                    "reuse_program and reuse_excluded_series must be empty",
                    retryable=True,
                )

    return validate


def _make_adoption_validator(measured_plans: Sequence[Mapping[str, Any]]):
    allowed = {
        (str(row["program"]), tuple(sorted(row["excluded_series"])))
        for row in measured_plans
    }

    def validate(payload: Mapping[str, Any]) -> None:
        key = (
            str(payload["program"]),
            tuple(sorted(str(uid) for uid in payload.get("excluded_series", ()))),
        )
        if key not in allowed:
            raise StagePostValidationError(
                "PLAN_NOT_MEASURED",
                "adopt exactly one entry from measured_plans; %s is not one of "
                "%s" % (
                    key[0] + " minus " + (", ".join(key[1]) or "nothing"),
                    sorted(
                        "%s minus %s" % (p, ", ".join(e) or "nothing")
                        for p, e in allowed
                    ),
                ),
                retryable=True,
            )

    return validate


def _base_input(
    *,
    episode_id: str,
    cohort: str,
    consumer_variant: str,
    window: Mapping[str, Any],
    search: Any,
    observation: Sequence[Mapping[str, Any]],
    experience_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One template for every episode.

    Only ``episode_id``, ``target``, ``public_observation`` -- which is a
    function of the window -- and ``prior_experience`` move between episodes.
    Everything else is fixed, and the per-field digests are written into the
    artifact so that is checkable rather than asserted.
    """
    return {
        "schema_version": "negative-path-input/1",
        "episode_id": episode_id,
        "target": {
            "cohort": cohort,
            "consumer_variant": consumer_variant,
            "cell_key": "batch:%s|consumer:%s" % (cohort, consumer_variant),
            "window_id": str(window["window_id"]),
            "support_origins": list(search.support),
            "delayed_origins": list(search.delayed),
            "observation_cutoff": int(search.support[0]),
            "training_series": list(search.train_uids),
            "evaluation_series_count": len(search.eval_uids),
            "exposure": search.exposure,
        },
        "consumer_structure": {
            "pooled": (
                "one model fitted on the stacked windows of all training "
                "channels"
            ),
            "per_channel": (
                "each training channel fits its own model; every evaluation "
                "channel is predicted by the equal-weight mean of those "
                "channel-wise models"
            ),
        },
        "program_menu": list(TREATMENTS),
        "identity_is_always_available": True,
        "evaluation_budget": {
            "charged_evaluations": EVALUATION_BUDGET,
            "menu_size": len(TREATMENTS),
            "reuse_confirmation_costs": 1,
            "reuse_confirmation_line_support": CONFIRMATION_LINE,
            "search_costs_one_per_shortlisted_program": True,
            "mask_round": "free; runs on the highest-Support shortlisted "
            "program",
            "abstaining_costs": 0,
            "delayed_window": "read once after the plan is named and never "
            "shown to you",
            "identity_incumbent": (
                "a plan whose delayed reading is below the identity incumbent "
                "is not adopted; the episode falls back to identity"
            ),
        },
        "public_observation": {
            "rule": (
                "public features of each training series on its own public "
                "prefix values[uid][:observation_cutoff]"
            ),
            "rows": [dict(row) for row in observation],
        },
        "prior_experience": {
            "row_count": len(experience_rows),
            "how_to_read": (
                "each row is a plan that was applied somewhere and what it "
                "scored. `cell_key` names the batch and Consumer structure, "
                "`window_id` the development window. Rows with `source` = "
                "'this run' are earlier episodes of this same sequence and "
                "carry their action, whether a re-confirmation cleared the "
                "line, and whether the identity incumbent refused the plan."
            ),
            "rows": [dict(row) for row in experience_rows],
        },
    }


# ------------------------------------------------------------------- episode
def _charged_confirmation(
    search: Any, program: str, excluded: Sequence[str]
) -> dict[str, Any]:
    """Support of a named plan on this window, charged as one evaluation.

    ``support_of_plan`` is the inherited bookkeeping call and counts itself as
    an internal evaluation; a re-confirmation is a real spend, so the count is
    moved from the internal tally to the charged one rather than added twice.
    """
    gains = search.support_of_plan(program, excluded)
    search.internal_evaluations -= 1
    search.support_evaluations_charged += 1
    search.log.append({
        "kind": "reuse_confirmation",
        "program": program,
        "excluded_series": sorted(str(uid) for uid in excluded),
        "charged": True,
        "aggregate_gain": gains["aggregate_gain"],
    })
    return gains


def _measured_plan_row(
    plan_id: str,
    program: str,
    excluded: Sequence[str],
    gains: Mapping[str, Any],
    measured_by: str,
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "program": program,
        "excluded_series": sorted(str(uid) for uid in excluded),
        "support_aggregate_gain": float(gains["aggregate_gain"]),
        "harmed_evaluation_series_count": int(
            gains["harmed_eval_series_count"]
        ),
        "measured_by": measured_by,
        "full_batch": not bool(excluded),
    }


def _run_episode(
    *,
    episode_id: str,
    cohort: str,
    consumer_variant: str,
    window: Mapping[str, Any],
    experience_rows: Sequence[Mapping[str, Any]],
    snapshot: Any,
    llm_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    search = NegativePathSearch(
        cohort=cohort,
        consumer_variant=consumer_variant,
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )
    observation = wvc._observation_table(search)
    base = _base_input(
        episode_id=episode_id, cohort=cohort,
        consumer_variant=consumer_variant, window=window, search=search,
        observation=observation, experience_rows=experience_rows,
    )
    backend = _default_backend_factory(int(llm_budget))
    gateway = wvc.NoToolGateway({"episode_id": episode_id})
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)
    harness_view = resolve_harness_view(snapshot, {}, role="fast")

    visible_plans = [
        (
            str(row["adopted_plan"]["program"]),
            [str(uid) for uid in row["adopted_plan"]["excluded_series"]],
        )
        for row in experience_rows
        if row.get("adopted_plan")
    ]
    visible_plans += [
        (
            str(row["reuse_confirmation"]["plan"]["program"]),
            [
                str(uid) for uid in
                row["reuse_confirmation"]["plan"]["excluded_series"]
            ],
        )
        for row in experience_rows
        if row.get("reuse_confirmation")
    ]
    action_payload, action_info = wvc._stage(
        core,
        stage="negative_path_action",
        case_id="NPA_%s" % episode_id,
        public_input={**base, "stage_note": ACTION_NOTE},
        harness_view=harness_view,
        schema_name="negative_path_action_v1",
        schema=ACTION_SCHEMA,
        validator=_make_action_validator(
            visible_plans, [row["record_id"] for row in experience_rows]
        ),
    )
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "cohort": cohort,
        "consumer_variant": consumer_variant,
        "window_id": str(window["window_id"]),
        "support_origins": list(search.support),
        "delayed_origins": list(search.delayed),
        "experience_rows_visible": len(experience_rows),
        "experience_record_ids_visible": [
            str(row["record_id"]) for row in experience_rows
        ],
        "base_input_field_shas": {
            str(key): canonical_sha256(wvc._plain(value))
            for key, value in base.items()
        },
        "stages": [action_info],
        "action_payload": wvc._plain(action_payload),
        "action": None,
        "evaluations_used": 0,
        "confirmation": None,
        "agent_plan": None,
        "final_plan": None,
        "reference_plan": dict(window["reference_plan"]),
        "reference_delayed_aggregate_gain": float(
            window["reference_delayed_aggregate_gain"]
        ),
    }
    if action_payload is None:
        record["llm_calls"] = int(backend.calls)
        record["instrument"] = search.accounting()
        record["wall_seconds"] = time.perf_counter() - started
        return record
    return _resolve_episode(
        record=record, core=core, base=base, harness_view=harness_view,
        search=search, action_payload=action_payload, window=window,
        backend=backend, episode_id=episode_id, started=started,
    )


def _resolve_episode(
    *, record, core, base, harness_view, search, action_payload, window,
    backend, episode_id, started,
) -> dict[str, Any]:
    action = str(action_payload["action"])
    record["action"] = action
    record["action_reason"] = str(action_payload.get("reason", ""))
    record["experience_use"] = [
        str(item) for item in action_payload.get("experience_use", ())
    ]
    plans: list[dict[str, Any]] = []
    mask_result = None

    if action == "REUSE_CONFIRM":
        program = str(action_payload["reuse_program"])
        excluded = sorted(
            str(uid) for uid in action_payload["reuse_excluded_series"]
        )
        gains = _charged_confirmation(search, program, excluded)
        confirmed = bool(gains["aggregate_gain"] >= CONFIRMATION_LINE)
        record["confirmation"] = {
            "plan": {"program": program, "excluded_series": excluded},
            "support_aggregate_gain": float(gains["aggregate_gain"]),
            "confirmation_line": CONFIRMATION_LINE,
            "confirmed": confirmed,
            "harmed_eval_series_count": int(gains["harmed_eval_series_count"]),
        }
        plans.append(_measured_plan_row(
            "P1", program, excluded, gains,
            "re-confirmation of a plan from prior experience",
        ))
        print(
            "NPA %s REUSE_CONFIRM %s minus %s -> support %+.6f confirmed=%s"
            % (episode_id, program, ", ".join(excluded) or "nothing",
               gains["aggregate_gain"], confirmed),
            flush=True,
        )
    elif action == "SEARCH":
        shortlist = [str(item) for item in action_payload["shortlist"]]
        record["shortlist"] = shortlist
        record["request_mask_search"] = bool(
            action_payload["request_mask_search"]
        )
        support_results = {
            program: search.full_batch_support(program)
            for program in shortlist
        }
        record["support_results"] = support_results
        for index, program in enumerate(shortlist, start=1):
            plans.append(_measured_plan_row(
                "P%d" % index, program, [], support_results[program],
                "full-batch Support evaluation",
            ))
        if record["request_mask_search"]:
            best = max(
                shortlist,
                key=lambda program: (
                    support_results[program]["aggregate_gain"],
                    -shortlist.index(program),
                ),
            )
            mask_result = search.mask_search(best)
            record["mask_search"] = wvc._plain(mask_result)
            if mask_result["final_excluded"]:
                plans.append(_measured_plan_row(
                    "PM", best, mask_result["final_excluded"],
                    mask_result["support"], "greedy exclusion-mask round",
                ))
        print(
            "NPA %s SEARCH %s mask=%s"
            % (episode_id, shortlist, record["request_mask_search"]),
            flush=True,
        )

    identity_gains = search.support_of_plan(IDENTITY, [])
    plans.append(_measured_plan_row(
        "P0", IDENTITY, [], identity_gains,
        "the identity baseline every gain is measured against",
    ))
    record["evaluations_used"] = int(search.support_evaluations_charged)
    record["measured_plans"] = plans

    if action == "ABSTAIN_IDENTITY":
        agent_plan = {"program": IDENTITY, "excluded_series": []}
        record["adoption_payload"] = None
        record["stage_2_skipped"] = (
            "the Agent abstained at the action stage, so no adoption stage ran"
        )
    else:
        adoption_payload, adoption_info = wvc._stage(
            core,
            stage="negative_path_adoption",
            case_id="NPA_%s" % episode_id,
            public_input={
                **base,
                "stage_note": ADOPTION_NOTE,
                "action_taken": action,
                "measured_plans": [dict(row) for row in plans],
                "reuse_confirmation": record["confirmation"],
                "evaluations_spent": int(search.support_evaluations_charged),
            },
            harness_view=harness_view,
            schema_name="budgeted_adoption_v1",
            schema=ADOPTION_SCHEMA,
            validator=_make_adoption_validator(plans),
        )
        record["stages"].append(adoption_info)
        record["adoption_payload"] = wvc._plain(adoption_payload)
        if adoption_payload is None:
            record["llm_calls"] = int(backend.calls)
            record["instrument"] = search.accounting()
            record["wall_seconds"] = time.perf_counter() - started
            return record
        agent_plan = {
            "program": str(adoption_payload["program"]),
            "excluded_series": sorted(
                str(uid) for uid in adoption_payload.get("excluded_series", ())
            ),
        }
        record["adoption_reason"] = str(adoption_payload.get("reason", ""))
    record["agent_plan"] = agent_plan
    record["llm_calls"] = int(backend.calls)
    return _apply_gate(
        record=record, search=search, plans=plans, agent_plan=agent_plan,
        window=window, episode_id=episode_id, started=started,
    )


def _apply_gate(
    *, record, search, plans, agent_plan, window, episode_id, started,
) -> dict[str, Any]:
    """The identity-incumbent gate, and the episode's final numbers.

    The bar is the best delayed gain among the full-batch plans this episode
    actually evaluated, floored at identity's zero.  A plan below it is not
    adopted.  The Agent never saw any of these delayed numbers.
    """
    full_batch_delayed: dict[str, float] = {}
    for row in plans:
        if not row["full_batch"] or row["program"] == IDENTITY:
            continue
        gains = search.delayed_gate(row["program"], [])
        full_batch_delayed[row["program"]] = float(gains["aggregate_gain"])
    bar = max(list(full_batch_delayed.values()) + [0.0])
    adopted_delayed = search.delayed_gate(
        agent_plan["program"], agent_plan["excluded_series"]
    )
    passed = bool(float(adopted_delayed["aggregate_gain"]) >= bar)
    if passed:
        final_plan = dict(agent_plan)
        support = search.support_of_plan(
            final_plan["program"], final_plan["excluded_series"]
        )
        delayed = adopted_delayed
    else:
        final_plan = {"program": IDENTITY, "excluded_series": []}
        support = search.support_of_plan(IDENTITY, [])
        delayed = search.delayed_gate(IDENTITY, [])
    reference = float(window["reference_delayed_aggregate_gain"])
    support_gain = float(support["aggregate_gain"])
    delayed_gain = float(delayed["aggregate_gain"])
    if final_plan["program"] == IDENTITY:
        relation = RELATION_ABSTAIN
    elif support_gain > 0.0 and delayed_gain > 0.0:
        relation = RELATION_POSITIVE
    elif (support_gain > 0.0) != (delayed_gain > 0.0):
        relation = RELATION_CONFLICT
    else:
        relation = RELATION_NEGATIVE
    record.update({
        "identity_incumbent_gate": {
            "bar": bar,
            "full_batch_delayed_evaluated": full_batch_delayed,
            "agent_plan_delayed": float(adopted_delayed["aggregate_gain"]),
            "passed": passed,
            "fell_back_to_identity": not passed,
            "rule": (
                "adopt only if delayed >= max(best evaluated full-batch "
                "delayed, 0); the Agent never sees these numbers"
            ),
        },
        "final_plan": final_plan,
        "support": support,
        "delayed": delayed,
        "relation": relation,
        "capture_ratio": (
            delayed_gain / reference if reference else None
        ),
        "matches_reference_plan": bool(
            final_plan["program"] == str(window["reference_plan"]["program"])
            and final_plan["excluded_series"]
            == sorted(str(uid) for uid in window["reference_plan"][
                "excluded_series"])
        ),
        "instrument": search.accounting(),
        "wall_seconds": time.perf_counter() - started,
    })
    print(
        "NPA %s final %s minus %s | support %+.6f delayed %+.6f | gate bar "
        "%+.6f passed=%s | evals %d llm %d"
        % (
            episode_id, final_plan["program"],
            ", ".join(final_plan["excluded_series"]) or "nothing",
            support_gain, delayed_gain, bar, passed,
            record["evaluations_used"], record["llm_calls"],
        ),
        flush=True,
    )
    return record


def _experience_entry(record: Mapping[str, Any]) -> Any:
    """This run's own episode, through the existing episode mechanism."""
    plan = record["final_plan"]
    audit = {
        "provenance": EXPERIENCE_PROVENANCE,
        "counts_as_unguided_exploration": False,
        "audit_note": (
            "engineering measurement from the negative-path adaptation "
            "sequence; not authorization evidence and not an unguided probe. "
            "Within-run feedback is deliberate here: the mechanism under test "
            "is whether the Agent learns from its own failed re-confirmations"
        ),
    }
    return build_episode(
        episode_id=str(record["episode_id"]),
        task_consumer_key="batch:%s|consumer:%s"
        % (record["cohort"], record["consumer_variant"]),
        domain_namespace=str(record["cohort"]),
        context_summary={
            "cohort": {"cohort_name": str(record["cohort"])},
            "local_pattern": {
                "consumer_variant": str(record["consumer_variant"]),
                "window_id": str(record["window_id"]),
            },
            "program_geometry": {
                "excluded_count": len(plan["excluded_series"]),
                "evaluations_used": int(record["evaluations_used"]),
            },
        },
        workflow_signature=workflow_signature_of(
            () if plan["program"] == IDENTITY else ({"op": plan["program"]},)
        ),
        support_response={
            "gain": float(record["support"]["aggregate_gain"]),
            "window": "support",
            "program": plan["program"],
            "excluded_series": list(plan["excluded_series"]),
            "action": str(record["action"]),
            "evaluations_used": int(record["evaluations_used"]),
            "reuse_confirmed": (
                None if record.get("confirmation") is None
                else bool(record["confirmation"]["confirmed"])
            ),
            "harmed_eval_series_count": int(
                record["support"]["harmed_eval_series_count"]
            ),
            **audit,
        },
        delayed_response={
            "gain": float(record["delayed"]["aggregate_gain"]),
            "window": "delayed",
            "identity_incumbent_gate_passed": bool(
                record["identity_incumbent_gate"]["passed"]
            ),
            "fell_back_to_identity": bool(
                record["identity_incumbent_gate"]["fell_back_to_identity"]
            ),
            "window_role": (
                "read once after the plan was named; the Agent never saw it"
            ),
            "harmed_eval_series_count": int(
                record["delayed"]["harmed_eval_series_count"]
            ),
            **audit,
        },
        relation=str(record["relation"]),
        evidence_level=EVIDENCE_DELAYED,
        local_status=STATUS_EPISODE_ONLY,
        evidence_refs=(EXPERIENCE_PROVENANCE,),
    )


# ------------------------------------------------------------------ verdict
def _verdict(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["episode_id"]): row for row in records}
    e1, e2, e3, e4 = (by_id.get(key) for key in ("E1", "E2", "E3", "E4"))
    facts: dict[str, Any] = {}
    sequence = [row for row in (e1, e2, e3) if row is not None]
    evals = [int(row["evaluations_used"]) for row in sequence]
    facts["e1_e3_evaluations"] = evals
    facts["evaluations_monotonically_non_increasing"] = all(
        evals[index] >= evals[index + 1] for index in range(len(evals) - 1)
    )
    facts["e3_ends_at_identity"] = bool(
        e3 is not None and e3.get("final_plan")
        and e3["final_plan"]["program"] == IDENTITY
    )
    negative_ids = {
        "run|%s" % str(row["episode_id"])
        for row in (e1, e2) if row is not None and (
            str(row.get("relation")) in ("ABSTAIN", "NEGATIVE", "CONFLICT")
            or (row.get("confirmation") or {}).get("confirmed") is False
            or (row.get("identity_incumbent_gate") or {}).get(
                "fell_back_to_identity")
        )
    }
    cited = set(e3.get("experience_use", ())) if e3 else set()
    facts["negative_or_marginal_record_ids"] = sorted(negative_ids)
    facts["e3_cited"] = sorted(cited)
    facts["e3_cites_a_negative_or_marginal_record"] = bool(
        cited & negative_ids
    )
    facts["e4_final_program"] = (
        e4["final_plan"]["program"] if e4 and e4.get("final_plan") else None
    )
    facts["e4_abstained"] = bool(
        e4 is not None and e4.get("final_plan")
        and e4["final_plan"]["program"] == IDENTITY
    )
    facts["e4_delayed_positive"] = bool(
        e4 is not None and e4.get("delayed")
        and float(e4["delayed"]["aggregate_gain"]) > 0.0
    )
    facts["gate_fallback_count"] = sum(
        1 for row in records
        if (row.get("identity_incumbent_gate") or {}).get(
            "fell_back_to_identity")
    )
    facts["e2_e3_spent_full_budget_without_abstaining"] = bool(
        e2 is not None and e3 is not None
        and int(e2["evaluations_used"]) == EVALUATION_BUDGET
        and int(e3["evaluations_used"]) == EVALUATION_BUDGET
        and (e2.get("final_plan") or {}).get("program") != IDENTITY
        and (e3.get("final_plan") or {}).get("program") != IDENTITY
    )

    if facts["e4_abstained"]:
        verdict = "OVERGENERALIZED_ABSTENTION"
        reason = "E4 abstained on the high-headroom control cell"
    elif (
        facts["evaluations_monotonically_non_increasing"]
        and facts["e3_ends_at_identity"]
        and facts["e3_cites_a_negative_or_marginal_record"]
        and facts["e4_delayed_positive"]
    ):
        verdict = "ADAPTIVE_ABSTENTION_CONVERGES"
        reason = (
            "E1..E3 charged evaluations %s are non-increasing, E3 ended at "
            "identity citing %s, and E4 adopted a delayed-positive plan"
            % (evals, facts["e3_cited"])
        )
    elif facts["gate_fallback_count"] >= 2:
        verdict = "GATE_SAVES_BUT_NO_LEARNING"
        reason = (
            "the identity gate forced the fallback in %d episodes"
            % facts["gate_fallback_count"]
        )
    elif facts["e2_e3_spent_full_budget_without_abstaining"]:
        verdict = "NO_BEHAVIOR_CHANGE"
        reason = (
            "E2 and E3 both spent the full %d evaluations and neither "
            "abstained" % EVALUATION_BUDGET
        )
    else:
        verdict = "MIXED"
        reason = (
            "no single pre-registered pattern matched; the episodes are "
            "reported one by one"
        )
    return {"verdict": verdict, "reason": reason, "facts": facts}


def _template_parity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which prompt fields moved between episodes, from the sent digests."""
    allowed = {"episode_id", "target", "public_observation", "prior_experience"}
    shas = {
        str(row["episode_id"]): row["base_input_field_shas"]
        for row in records if row.get("base_input_field_shas")
    }
    moving: set[str] = set()
    keys: set[str] = set()
    for table in shas.values():
        keys |= set(table)
    for key in keys:
        values = {table.get(key) for table in shas.values()}
        if len(values) > 1:
            moving.add(key)
    return {
        "episodes_compared": sorted(shas),
        "fields_compared": sorted(keys),
        "fields_that_move_between_episodes": sorted(moving),
        "allowed_to_move": sorted(allowed),
        "only_allowed_fields_moved": moving.issubset(allowed),
        "note": (
            "public_observation is a function of the window, so it moves with "
            "the window identifier; every other field is fixed by the template"
        ),
    }


# --------------------------------------------------------------- orchestration
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    corpus = wvc._experience_corpus()
    low = LOW_HEADROOM_CELL
    control = CONTROL_CELL
    windows = {
        "E1": _quoted_window(low["cohort"], low["consumer_variant"], "W2"),
        "E2": _quoted_window(low["cohort"], low["consumer_variant"], "W3"),
        "E3": _roster_window_4(),
        "E4": _quoted_window(
            control["cohort"], control["consumer_variant"], "W3"
        ),
    }
    frozen_low_w1 = _frozen_experience_rows(
        corpus, [(low["cohort"], low["consumer_variant"], "W1")]
    )
    frozen_control = _frozen_experience_rows(
        corpus,
        [
            (control["cohort"], control["consumer_variant"], "W1"),
            (control["cohort"], control["consumer_variant"], "W2"),
        ],
    )
    plan = [
        ("E1", low, windows["E1"]),
        ("E2", low, windows["E2"]),
        ("E3", low, windows["E3"]),
        ("E4", control, windows["E4"]),
    ]
    snapshot = compile_snapshot(
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False
    )
    records: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    episodes: list[Any] = []
    llm_used = 0
    stopped_reason: str | None = None

    for episode_id, cell, window in plan:
        remaining = LLM_CALL_BUDGET_TOTAL - llm_used
        if remaining < 2:
            stopped_reason = (
                "global LLM budget exhausted before %s (%d of %d used)"
                % (episode_id, llm_used, LLM_CALL_BUDGET_TOTAL)
            )
            break
        visible = (
            list(frozen_control) if episode_id == "E4" else list(frozen_low_w1)
        )
        visible += list(run_rows)
        print(
            "NPA episode %s %s x %s @ %s (visible experience %d rows, llm "
            "%d/%d)"
            % (
                episode_id, cell["cohort"], cell["consumer_variant"],
                window["window_id"], len(visible), llm_used,
                LLM_CALL_BUDGET_TOTAL,
            ),
            flush=True,
        )
        record = _run_episode(
            episode_id=episode_id,
            cohort=cell["cohort"],
            consumer_variant=cell["consumer_variant"],
            window=window,
            experience_rows=visible,
            snapshot=snapshot,
            llm_budget=min(LLM_CALL_BUDGET_PER_EPISODE, remaining),
        )
        llm_used += int(record["llm_calls"])
        records.append(record)
        if record.get("final_plan") is not None:
            written = _experience_entry(record)
            episodes.append(written)
            record["experience_written"] = written.to_dict()
            run_rows.append(_run_experience_row(record))
        else:
            record["experience_written"] = None
        if episode_id == "E1" and record.get("action") is None:
            stopped_reason = (
                "circuit breaker: E1 produced no stage-one payload"
            )
            break

    verdict = _verdict(records)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "does the Agent converge to abstention after repeated failed "
            "re-confirmation on a low-headroom cell, and does the abstention "
            "stay local"
        ),
        "not_authorization_evidence": (
            "no Skill is written, no TRY right is granted, no Episode is "
            "promoted, no Fast or Slow path is entered, no snapshot pointer "
            "moves"
        ),
        "verdict": verdict["verdict"],
        "verdict_reason": verdict["reason"],
        "verdict_facts": verdict["facts"],
        "pre_registered": PRE_REGISTERED,
        "template_parity_check": _template_parity(records),
        "model": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "cells": {"low_headroom": low, "control": control},
        "windows": windows,
        "visible_experience_design": {
            "E1": "the low-headroom cell's frozen W1 row only",
            "E2": "that row plus everything E1 produced",
            "E3": "that row plus E1 and E2",
            "E4": (
                "the control cell's frozen W1 and W2 rows plus all three "
                "earlier episodes, so an over-applied abstention would show"
            ),
            "scoring_references_are_never_shown": True,
        },
        "llm_call_count": llm_used,
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "stopped_early": stopped_reason,
        "experience_entries_written": [
            episode.to_dict() for episode in episodes
        ],
        "experience_provenance": EXPERIENCE_PROVENANCE,
        "run_experience_rows": run_rows,
        "episodes": records,
        "wall_seconds": time.perf_counter() - started,
    }
    if dry_run:
        print(json.dumps(verdict, indent=2, ensure_ascii=False, default=str))
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
def _markdown_head(payload: Mapping[str, Any]) -> list[str]:
    facts = payload["verdict_facts"]
    parity = payload["template_parity_check"]
    lines = [
        "# negative-path adaptation v1",
        "",
        "**Verdict: `%s`** -- %s."
        % (payload["verdict"], payload["verdict_reason"]),
        "",
        "T233 x per_channel is an established low-headroom cell: its W3 full "
        "search settles on `identity`, and the plan its W1 run adopted scores "
        "+0.0017 Support on W2 and -0.0010 on W3, both under the 0.005 "
        "material line. This run asks whether an Agent that keeps failing to "
        "re-confirm converges to abstention and stops spending budget, and "
        "whether that abstention stays local.",
        "",
        "**Engineering effect measurement, not authorization evidence.** %s."
        % payload["not_authorization_evidence"],
        "",
        "## 0. Design and what was pre-registered",
        "",
        "Two deliberate departures from the earlier warm/cold runs, both fixed "
        "before the first call:",
        "",
    ]
    for item in payload["pre_registered"][
        "deliberate_departures_from_the_warm_cold_runs"
    ]:
        lines.append("- %s" % item)
    lines += [
        "",
        "Actions and what they cost:",
        "",
    ]
    for action, cost in payload["pre_registered"][
        "actions_and_their_cost"
    ].items():
        lines.append("- `%s`: %s" % (action, cost))
    lines += [
        "",
        "Identity-incumbent gate: %s."
        % payload["pre_registered"]["identity_incumbent_gate"],
        "",
        "Verdict rules, applied in this order:",
        "",
    ]
    for index, rule in enumerate(
        payload["pre_registered"]["verdict_rules_in_this_order"], start=1
    ):
        lines.append("%d. %s" % (index, rule))
    lines += [
        "",
        "## 1. Episodes, windows and what each one could see",
        "",
        "| episode | cell | window | support origins | delayed origins | "
        "reference plan | reference delayed | visible experience |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    design = payload["visible_experience_design"]
    for record in payload["episodes"]:
        window = payload["windows"][record["episode_id"]]
        lines.append(
            "| %s | %s x %s | %s | %s | %s | %s | %s | %s |"
            % (
                record["episode_id"], record["cohort"],
                record["consumer_variant"], record["window_id"],
                window["support_origins"], window["delayed_origins"],
                wvc._plan_text(window["reference_plan"]),
                wvc._gain(window["reference_delayed_aggregate_gain"]),
                design.get(record["episode_id"], ""),
            )
        )
    lines += [
        "",
        "W2, W3 and the control window are quoted from "
        "`batch_recipe_windows_v1`; W4 is the frozen `e1v2_task_04` roster "
        "spec and its reference was computed here by the frozen v2 recipe with "
        "0 LLM calls. Scoring references are never shown to the Agent.",
        "",
        "Template parity: fields that move between episodes are %s, all inside "
        "the allowed set %s -- **%s**. %s"
        % (
            ", ".join("`%s`" % f
                      for f in parity["fields_that_move_between_episodes"]),
            ", ".join("`%s`" % f for f in parity["allowed_to_move"]),
            parity["only_allowed_fields_moved"],
            parity["note"],
        ),
        "",
    ]
    return lines


def _markdown_body(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "## 2. Episode table",
        "",
        "| episode | window | action | evals | confirmation | plan the Agent "
        "named | gate | final plan | support | delayed | capture | relation | "
        "LLM |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: "
        "| --- | ---: |",
    ]
    for record in payload["episodes"]:
        confirmation = record.get("confirmation")
        gate = record.get("identity_incumbent_gate") or {}
        support = record.get("support") or {}
        delayed = record.get("delayed") or {}
        lines.append(
            "| %s | %s | `%s` | %d | %s | %s | %s | %s | %s | %s | %s | %s | %d |"
            % (
                record["episode_id"], record["window_id"],
                record.get("action") or "n/a",
                record["evaluations_used"],
                "n/a" if confirmation is None else "%s -> %s (%s)"
                % (
                    wvc._plan_text(confirmation["plan"]),
                    wvc._gain(confirmation["support_aggregate_gain"]),
                    "confirmed" if confirmation["confirmed"] else "**failed**",
                ),
                wvc._plan_text(record.get("agent_plan")),
                "n/a" if not gate else (
                    "pass (bar %s)" % wvc._gain(gate.get("bar"))
                    if gate.get("passed")
                    else "**fallback** (bar %s)" % wvc._gain(gate.get("bar"))
                ),
                wvc._plan_text(record.get("final_plan")),
                wvc._gain(support.get("aggregate_gain")),
                wvc._gain(delayed.get("aggregate_gain")),
                wvc._ratio(record.get("capture_ratio")),
                record.get("relation") or "n/a",
                record["llm_calls"],
            )
        )
    facts = payload["verdict_facts"]
    lines += [
        "",
        "Charged evaluations across E1..E3: %s, non-increasing: **%s**. E3 "
        "ended at identity: **%s**. E3 cited %s, of which the negative or "
        "marginal records are %s -- cites at least one: **%s**. E4 final "
        "program `%s`, delayed positive: **%s**. Identity gate forced the "
        "fallback in %d episode(s)."
        % (
            facts["e1_e3_evaluations"],
            facts["evaluations_monotonically_non_increasing"],
            facts["e3_ends_at_identity"],
            facts["e3_cited"] or "nothing",
            facts["negative_or_marginal_record_ids"] or "none",
            facts["e3_cites_a_negative_or_marginal_record"],
            facts["e4_final_program"], facts["e4_delayed_positive"],
            facts["gate_fallback_count"],
        ),
        "",
        "## 3. What each episode said",
        "",
    ]
    for record in payload["episodes"]:
        lines += [
            "### %s -- %s x %s @ %s"
            % (
                record["episode_id"], record["cohort"],
                record["consumer_variant"], record["window_id"],
            ),
            "",
            "- visible experience: %d row(s) -- %s"
            % (
                record["experience_rows_visible"],
                ", ".join(
                    "`%s`" % rid
                    for rid in record["experience_record_ids_visible"]
                ) or "none",
            ),
            "- action: `%s`; cited: %s"
            % (
                record.get("action"),
                ", ".join("`%s`" % r for r in record.get("experience_use", []))
                or "none",
            ),
            "- action reason: %s" % record.get("action_reason", ""),
        ]
        if record.get("shortlist"):
            lines.append(
                "- shortlist: %s (mask requested: %s)"
                % (
                    ", ".join("`%s`" % p for p in record["shortlist"]),
                    record.get("request_mask_search"),
                )
            )
        if record.get("confirmation"):
            confirmation = record["confirmation"]
            lines.append(
                "- re-confirmation: %s scored %s against a line of %.3f -- %s"
                % (
                    wvc._plan_text(confirmation["plan"]),
                    wvc._gain(confirmation["support_aggregate_gain"]),
                    confirmation["confirmation_line"],
                    "confirmed" if confirmation["confirmed"] else "FAILED",
                )
            )
        if record.get("adoption_reason"):
            lines.append(
                "- adoption reason: %s" % record["adoption_reason"]
            )
        if record.get("stage_2_skipped"):
            lines.append("- %s" % record["stage_2_skipped"])
        gate = record.get("identity_incumbent_gate") or {}
        if gate:
            lines.append(
                "- gate: bar %s, the named plan's delayed %s, passed %s"
                % (
                    wvc._gain(gate.get("bar")),
                    wvc._gain(gate.get("agent_plan_delayed")),
                    gate.get("passed"),
                )
            )
        lines.append(
            "- retries: %s"
            % ", ".join(
                "%s=%s%s" % (
                    info["stage"], info["validation_retry_count"],
                    info["validation_error_codes"] or "",
                )
                for info in record["stages"]
            )
        )
        lines.append("")
    return lines


def _markdown_tail(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "## 4. Experience rows this run produced",
        "",
        "Written through the existing episode mechanism with "
        "`provenance=\"%s\"`, `counts_as_unguided_exploration: false`. Unlike "
        "the earlier runs these rows **are** fed forward inside this run, "
        "which is the point: each episode sees whether the previous one's "
        "re-confirmation cleared the line and whether the identity incumbent "
        "refused its plan."
        % payload["experience_provenance"],
        "",
        "| episode | cell | window | action | confirmation | final plan | "
        "support | delayed | gate fallback | relation |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["run_experience_rows"]:
        confirmation = row.get("reuse_confirmation")
        lines.append(
            "| `%s` | `%s` | %s | `%s` | %s | %s | %s | %s | %s | %s |"
            % (
                row["episode_id"],
                str(row["cell_key"]).replace("|", r"\|"),
                row["window_id"], row["action_taken"],
                "n/a" if confirmation is None else "%s (%s)"
                % (
                    wvc._gain(confirmation["support_aggregate_gain"]),
                    "confirmed" if confirmation["confirmed"] else "failed",
                ),
                wvc._plan_text(row["adopted_plan"]),
                wvc._gain(row["support_aggregate_gain"]),
                wvc._gain(row["delayed_aggregate_gain"]),
                row["identity_incumbent_gate"]["fell_back_to_identity"],
                row["relation"],
            )
        )
    lines += [
        "",
        "## 5. What this does not say",
        "",
        "- It does not authorize anything, and abstaining is not a Skill.",
        "- Four episodes on two cells with one model. Every label here is a "
        "single draw, not a rate.",
        "- The identity gate is a Harness-side backstop that reads the delayed "
        "window. Where it fired, the episode's final plan is the gate's "
        "choice, not the Agent's, and the table separates the two.",
        "- Within-run feedback means the episodes are not independent by "
        "construction. That is the mechanism under test, not a confound to be "
        "removed, but it does mean these four rows cannot be pooled with the "
        "earlier warm/cold runs.",
        "- The low-headroom cell being low-headroom is itself an established "
        "fact from the frozen artifacts, not something this run discovered.",
        "",
        "## Provenance",
        "",
        "- model: `%s` at `%s`"
        % (payload["model"]["model"], payload["model"]["base_url"]),
        "- instrument, observation table, corpus and stage driver: imported "
        "from `run_e2_warm_vs_cold_recipe_search`, which is not modified",
        "- W4 reference: `run_batch_composition_headroom.make_batch_recipe` "
        "with `adoption_rule_version=\"v2\"`, computed here, 0 LLM calls",
        "- LLM calls: %d of %d"
        % (payload["llm_call_count"], payload["llm_call_budget_total"]),
        "- wall seconds: %.1f" % payload["wall_seconds"],
        "",
    ]
    return lines


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = (
        _markdown_head(payload) + _markdown_body(payload)
        + _markdown_tail(payload)
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run the sequence but print the verdict instead of writing",
    )
    args = parser.parse_args(argv)
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
