"""warm vs cold, rotated over all six cells: a per-target win/loss table.

``warm_vs_cold_recipe_search_v1`` ran the budgeted search on two targets and
found the warm arm far ahead on quality at equal cost -- capture 0.367 -> 0.962
on one of them -- but its pre-registered label set had no cell for "same cost,
better quality", so both targets came out ``WARM_NO_BENEFIT``.  Two targets is
also a demonstration, not a table.

This runner rotates the same experiment over all six cells and reads it with a
corrected, pre-registered criterion.  Relative to that run exactly four things
change; everything else is imported from it rather than restated:

1. **targets**: all six cells, every one of them on window **W3**, whose
   origins are quoted verbatim from ``batch_recipe_windows_v1`` and which has
   never been a target before.  Leave-one-cohort-out is unchanged: the warm arm
   sees the 18 frozen records minus every row measured on the target's own
   cohort.  Rows this rotation produces are isolated -- they never enter any
   episode's visible experience;
2. **budget**: the shortlist is capped at **2** programs, so two full-batch
   Support evaluations, plus the optional mask round and the closing delayed
   gate.  The menu still holds 7, so the budget binds harder than before;
3. **criterion**, fixed before the first call: the primary per-target readout
   is the **paired delayed difference**, warm minus cold.  Above ``+0.005`` is
   ``WARM_WINS_QUALITY``, below ``-0.005`` is ``COLD_WINS_QUALITY``, otherwise
   ``TIE``.  Cost is reported separately and never folded into the label: if the
   warm arm also used fewer evaluations without losing quality, that is recorded
   as ``WARM_ALSO_CHEAPER``.  The overall verdict is the win / loss / tie count
   over the six targets **plus the worst target**; a pooled number on its own is
   not an acceptable reading;
4. **instrument hardening**: the adoption prompt now enumerates every plan the
   instrument actually measured, so an arm cannot accidentally name an
   unmeasured mask.  The enumeration is built the same way for both arms.

Everything else is inherited verbatim from
``run_e2_warm_vs_cold_recipe_search``: the corpus builder, the observation
table, the search instrument, the shortlist and adoption validators, the
Experience writer, and the wording of both stage notes apart from the budget
number and the one appended sentence.  That module is imported, never modified.

The two arms' prompts are checked field by field before the run and the check is
written into the artifact: they may differ in ``prior_batch_experience`` and
nowhere else.

Not authorization evidence: no Skill, no TRY right, no promotion, no Fast or
Slow path.  Experience entries carry
``provenance="budgeted_search_engineering"`` and are not fed back into either
arm.

Run:

    python evaluation/functional/run_e2_warm_vs_cold_rotation.py

Writes ``artifacts/functional/e2/warm_vs_cold_rotation_v1.json`` and ``.md``.
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
from evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
    NF_MODEL,
)
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_sha256,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentRole,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)

PROTOCOL_VERSION = "warm_vs_cold_rotation_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "warm_vs_cold_rotation_v1.json"
OUT_MD = E2 / "warm_vs_cold_rotation_v1.md"
PRIOR_RUN_ARTIFACT = E2 / "warm_vs_cold_recipe_search_v1.json"

# change 2: the shortlist, and therefore the charged budget, is two.
SUPPORT_EVALUATION_BUDGET = 2
# change 3: the paired delayed difference decides the label.
QUALITY_DELTA_THRESHOLD = 0.005
WINDOW_ID = "W3"
LLM_CALL_BUDGET_TOTAL = 60
LLM_CALL_BUDGET_PER_EPISODE = 5
VALIDATION_RETRIES = wvc.VALIDATION_RETRIES
EXPERIENCE_PROVENANCE = wvc.EXPERIENCE_PROVENANCE
TREATMENTS = wvc.TREATMENTS
IDENTITY = wvc.IDENTITY

# change 1: six targets, all on W3.
TARGETS: dict[str, dict[str, Any]] = {
    "%s_%s" % (cohort, variant): {
        "target_id": "%s_%s" % (cohort, variant),
        "cohort": cohort,
        "consumer_variant": variant,
        "window_id": WINDOW_ID,
        "why": "rotation cell %s x %s on the never-targeted window %s"
        % (cohort, variant, WINDOW_ID),
    }
    for cohort in bch.RECIPE_V2_COHORTS
    for variant in bch.CONSUMER_VARIANTS
}
EPISODE_ORDER: tuple[tuple[str, str], ...] = tuple(
    (target_id, arm)
    for target_id in TARGETS
    for arm in ("cold", "warm")
)

PRE_REGISTERED = {
    "fixed_before_the_first_llm_call": True,
    "changes_from_warm_vs_cold_recipe_search_v1": [
        "targets: all six cells, all on window W3 (origins quoted from "
        "batch_recipe_windows_v1; W3 has never been a target)",
        "budget: shortlist capped at %d, so %d charged full-batch Support "
        "evaluations" % (SUPPORT_EVALUATION_BUDGET, SUPPORT_EVALUATION_BUDGET),
        "criterion: the paired delayed difference decides the label; cost is "
        "reported separately and never folded in",
        "instrument hardening: the adoption prompt enumerates every measured "
        "plan, identically for both arms",
    ],
    "primary_readout": (
        "per target, the paired delayed difference delta = warm delayed "
        "aggregate gain minus cold delayed aggregate gain, both measured by "
        "the same instrument on the same window"
    ),
    "labels_first_match_wins": [
        "WARM_WINS_QUALITY: delta > +%.3f" % QUALITY_DELTA_THRESHOLD,
        "COLD_WINS_QUALITY: delta < -%.3f" % QUALITY_DELTA_THRESHOLD,
        "TIE: otherwise",
    ],
    "cost_is_reported_separately": (
        "charged evaluations and LLM calls are reported per arm and never "
        "enter the label. WARM_ALSO_CHEAPER is recorded as a separate flag "
        "when the warm arm used strictly fewer charged evaluations and its "
        "delayed gain is not below the cold arm's"
    ),
    "overall_verdict": (
        "the win / loss / tie count over the six targets, plus the worst "
        "target by paired delta, named explicitly. A pooled mean on its own "
        "is not an acceptable reading of this run"
    ),
    "experience_isolation": (
        "the warm arm's corpus is the 18 frozen records only; every row this "
        "rotation produces is written to the artifact and to no episode's "
        "visible experience"
    ),
    "circuit_breaker": "stop and report if the first episode produces no "
    "adopted plan",
    "budget": {
        "charged_full_batch_support_evaluations": SUPPORT_EVALUATION_BUDGET,
        "menu_size": len(TREATMENTS),
        "mask_search_rounds": 1,
        "mask_search_target": "the shortlisted program with the highest "
        "full-batch Support gain (frozen, unchanged)",
        "llm_calls_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_per_episode": LLM_CALL_BUDGET_PER_EPISODE,
        "validation_retries_per_stage": VALIDATION_RETRIES,
    },
    "episode_order": ["%s %s" % (target, arm) for target, arm in EPISODE_ORDER],
    "inherited_verbatim_from": "run_e2_warm_vs_cold_recipe_search",
}

# The shortlist schema is the inherited one with the cap moved to two.
SHORTLIST_SCHEMA: dict[str, Any] = json.loads(
    json.dumps(wvc.SHORTLIST_SCHEMA)
)
SHORTLIST_SCHEMA["$id"] = "budgeted-shortlist/1-rotation"
SHORTLIST_SCHEMA["properties"]["shortlist"]["maxItems"] = (
    SUPPORT_EVALUATION_BUDGET
)
ADOPTION_SCHEMA = wvc.ADOPTION_SCHEMA

# The stage notes are the inherited strings with one substring changed and one
# sentence appended, so "otherwise verbatim" is a property of the code.
SHORTLIST_NOTE = wvc.SHORTLIST_NOTE.replace(
    "at most %d programs" % wvc.SUPPORT_EVALUATION_BUDGET,
    "at most %d programs" % SUPPORT_EVALUATION_BUDGET,
)
if SHORTLIST_NOTE == wvc.SHORTLIST_NOTE:
    raise RuntimeError("the shortlist note did not pick up the new budget")
ADOPTION_NOTE = wvc.ADOPTION_NOTE + (
    " Every plan the instrument actually measured is listed in "
    "`measured_plans`; adopt exactly one of them, naming its `program` and its "
    "`excluded_series` exactly as they appear there."
)


class RotationSearch(wvc.BudgetedSearch):
    """The inherited instrument; only the budget it reports is this run's."""

    def accounting(self) -> dict[str, Any]:
        row = super().accounting()
        row["budget"] = SUPPORT_EVALUATION_BUDGET
        row["budget_source"] = PROTOCOL_VERSION
        return row


# --------------------------------------------------------------------- inputs
def _base_input(
    *,
    target: Mapping[str, Any],
    window: Mapping[str, Any],
    search: Any,
    observation: Sequence[Mapping[str, Any]],
    experience_rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    """The inherited prompt body with this run's budget number substituted."""
    base = wvc._base_public_input(
        target=target, window=window, search=search, observation=observation,
        experience_rows=experience_rows, arm=arm,
    )
    base["evaluation_budget"] = {
        **base["evaluation_budget"],
        "full_batch_support_evaluations": SUPPORT_EVALUATION_BUDGET,
    }
    return base


def _measured_plans(
    *,
    shortlist: Sequence[str],
    support_results: Mapping[str, Mapping[str, Any]],
    mask_result: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], str]:
    """Every plan the instrument actually measured, enumerated for the Agent."""
    plans: list[dict[str, Any]] = []
    for index, program in enumerate(shortlist, start=1):
        plans.append({
            "plan_id": "P%d" % index,
            "program": program,
            "excluded_series": [],
            "support_aggregate_gain": float(
                support_results[program]["aggregate_gain"]
            ),
            "harmed_evaluation_series_count": int(
                support_results[program]["harmed_eval_series_count"]
            ),
            "measured_by": "full-batch Support evaluation",
        })
    if mask_result is None:
        note = "no mask round was requested, so no masked plan was measured"
    elif not mask_result["final_excluded"]:
        note = (
            "the mask round ran on `%s` and accepted no revert, so it produced "
            "no plan beyond the full-batch one already listed"
            % mask_result["program"]
        )
    else:
        plans.append({
            "plan_id": "PM",
            "program": str(mask_result["program"]),
            "excluded_series": list(mask_result["final_excluded"]),
            "support_aggregate_gain": float(
                mask_result["support"]["aggregate_gain"]
            ),
            "harmed_evaluation_series_count": int(
                mask_result["support"]["harmed_eval_series_count"]
            ),
            "measured_by": "greedy exclusion-mask round",
        })
        note = (
            "the mask round ran on `%s` and reverted %s"
            % (
                mask_result["program"],
                ", ".join(mask_result["final_excluded"]),
            )
        )
    plans.append({
        "plan_id": "P0",
        "program": IDENTITY,
        "excluded_series": [],
        "support_aggregate_gain": 0.0,
        "harmed_evaluation_series_count": 0,
        "measured_by": "the identity baseline every gain is measured against",
    })
    return plans, note


# ------------------------------------------------------------------- episode
def _run_episode(
    *,
    target: Mapping[str, Any],
    arm: str,
    window: Mapping[str, Any],
    corpus: Mapping[str, Any],
    snapshot: Any,
    llm_budget: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    episode_id = "%s_%s" % (target["target_id"], arm)
    search = RotationSearch(
        cohort=target["cohort"],
        consumer_variant=target["consumer_variant"],
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )
    observation = wvc._observation_table(search)
    experience_rows = wvc._visible_rows(corpus, arm, target["cohort"])
    base = _base_input(
        target=target, window=window, search=search, observation=observation,
        experience_rows=experience_rows, arm=arm,
    )
    backend = _default_backend_factory(int(llm_budget))
    gateway = wvc.NoToolGateway({"episode_id": episode_id, "arm": arm})
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)
    harness_view = resolve_harness_view(snapshot, {}, role="fast")

    shortlist_payload, shortlist_info = wvc._stage(
        core,
        stage="budgeted_shortlist",
        case_id="WVCR_%s" % episode_id,
        public_input={**base, "stage_note": SHORTLIST_NOTE},
        harness_view=harness_view,
        schema_name="budgeted_shortlist_v1_rotation",
        schema=SHORTLIST_SCHEMA,
        validator=wvc._make_shortlist_validator(),
    )
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "target_id": target["target_id"],
        "arm": arm,
        "cohort": target["cohort"],
        "consumer_variant": target["consumer_variant"],
        "window_id": window["window_id"],
        "support_origins": list(search.support),
        "delayed_origins": list(search.delayed),
        "experience_rows_visible": len(experience_rows),
        "experience_cohorts_visible": sorted(
            {str(row["cohort"]) for row in experience_rows}
        ),
        "stages": [shortlist_info],
        "base_input_field_shas": _field_shas(base),
        "shortlist_payload": wvc._plain(shortlist_payload),
        "adopted_plan": None,
        "support": None,
        "delayed": None,
        "capture_ratio": None,
        "evaluations_used": 0,
        "shortlist": [],
    }
    if shortlist_payload is None:
        record["llm_calls"] = int(backend.calls)
        record["instrument"] = search.accounting()
        record["wall_seconds"] = time.perf_counter() - started
        return record

    shortlist = [str(item) for item in shortlist_payload["shortlist"]]
    wants_mask = bool(shortlist_payload["request_mask_search"])
    support_results = {
        program: search.full_batch_support(program) for program in shortlist
    }
    best = max(
        shortlist,
        key=lambda program: (
            support_results[program]["aggregate_gain"],
            -shortlist.index(program),
        ),
    )
    mask_result = search.mask_search(best) if wants_mask else None
    plans, mask_note = _measured_plans(
        shortlist=shortlist, support_results=support_results,
        mask_result=mask_result,
    )
    print(
        "ROT %s shortlist=%s mask=%s best=%s (%+.6f)"
        % (episode_id, shortlist, wants_mask, best,
           support_results[best]["aggregate_gain"]),
        flush=True,
    )
    record.update({
        "shortlist": shortlist,
        "request_mask_search": wants_mask,
        "best_shortlisted_program": best,
        "support_results": {
            program: support_results[program] for program in shortlist
        },
        "mask_search": wvc._plain(mask_result),
        "measured_plans": plans,
        "mask_search_note": mask_note,
        "evaluations_used": int(search.support_evaluations_charged),
        "shortlist_reason": str(shortlist_payload.get("reason", "")),
    })
    return _finish_episode(
        record=record, core=core, base=base, harness_view=harness_view,
        search=search, shortlist=shortlist, support_results=support_results,
        mask_result=mask_result, plans=plans, mask_note=mask_note,
        window=window, backend=backend, episode_id=episode_id, started=started,
    )


def _finish_episode(
    *, record, core, base, harness_view, search, shortlist, support_results,
    mask_result, plans, mask_note, window, backend, episode_id, started,
) -> dict[str, Any]:
    adoption_input = {
        **base,
        "stage_note": ADOPTION_NOTE,
        "your_shortlist": list(shortlist),
        "support_results": [
            {
                "program": program,
                "full_batch_support_aggregate_gain": support_results[program][
                    "aggregate_gain"],
                "harmed_evaluation_series_count": support_results[program][
                    "harmed_eval_series_count"],
            }
            for program in shortlist
        ],
        "identity_support_aggregate_gain": 0.0,
        "mask_search_result": (
            None if mask_result is None else {
                "program": mask_result["program"],
                "reverted_series": list(mask_result["final_excluded"]),
                "support_aggregate_gain": mask_result["support"][
                    "aggregate_gain"],
                "harmed_evaluation_series_count": mask_result["support"][
                    "harmed_eval_series_count"],
                "steps": [
                    {
                        "reverted_series": row["reverted_series"],
                        "support_aggregate_gain": row["support_aggregate_gain"],
                        "decision": row["decision"],
                    }
                    for row in mask_result["steps"]
                ],
            }
        ),
        "measured_plans": [dict(row) for row in plans],
        "measured_plans_note": mask_note,
        "evaluations_spent": int(search.support_evaluations_charged),
    }
    adoption_payload, adoption_info = wvc._stage(
        core,
        stage="budgeted_adoption",
        case_id="WVCR_%s" % episode_id,
        public_input=adoption_input,
        harness_view=harness_view,
        schema_name="budgeted_adoption_v1",
        schema=ADOPTION_SCHEMA,
        validator=wvc._make_adoption_validator(
            shortlist=shortlist, mask_result=mask_result,
        ),
    )
    record["stages"].append(adoption_info)
    record["adoption_payload"] = wvc._plain(adoption_payload)
    record["llm_calls"] = int(backend.calls)
    if adoption_payload is None:
        record["instrument"] = search.accounting()
        record["wall_seconds"] = time.perf_counter() - started
        return record

    plan = {
        "program": str(adoption_payload["program"]),
        "excluded_series": sorted(
            str(uid) for uid in adoption_payload.get("excluded_series", ())
        ),
    }
    support = search.support_of_plan(plan["program"], plan["excluded_series"])
    delayed = search.delayed_gate(plan["program"], plan["excluded_series"])
    reference = float(window["reference_delayed_aggregate_gain"])
    record.update({
        "adopted_plan": plan,
        "adoption_reason": str(adoption_payload.get("reason", "")),
        "support": support,
        "delayed": delayed,
        "reference_delayed_aggregate_gain": reference,
        "reference_plan": dict(window["reference_plan"]),
        "capture_ratio": (
            float(delayed["aggregate_gain"]) / reference if reference else None
        ),
        "matches_reference_plan": bool(
            plan["program"] == str(window["reference_plan"]["program"])
            and plan["excluded_series"]
            == sorted(str(uid) for uid in window["reference_plan"][
                "excluded_series"])
        ),
        "instrument": search.accounting(),
        "wall_seconds": time.perf_counter() - started,
    })
    print(
        "ROT %s adopted %s minus %s | support %+.6f delayed %+.6f | evals %d "
        "llm %d"
        % (
            episode_id, plan["program"],
            ", ".join(plan["excluded_series"]) or "nothing",
            support["aggregate_gain"], delayed["aggregate_gain"],
            record["evaluations_used"], record["llm_calls"],
        ),
        flush=True,
    )
    return record


# ------------------------------------------------------------------ verdicts
def _target_verdict(
    cold: Mapping[str, Any] | None, warm: Mapping[str, Any] | None
) -> dict[str, Any]:
    if (
        cold is None or warm is None
        or cold.get("delayed") is None or warm.get("delayed") is None
    ):
        return {
            "label": "UNREADABLE",
            "reason": "one of the two arms produced no adopted plan",
            "paired_delayed_delta": None,
            "warm_also_cheaper": False,
        }
    cold_delayed = float(cold["delayed"]["aggregate_gain"])
    warm_delayed = float(warm["delayed"]["aggregate_gain"])
    delta = warm_delayed - cold_delayed
    if delta > QUALITY_DELTA_THRESHOLD:
        label = "WARM_WINS_QUALITY"
    elif delta < -QUALITY_DELTA_THRESHOLD:
        label = "COLD_WINS_QUALITY"
    else:
        label = "TIE"
    cold_evals = int(cold["evaluations_used"])
    warm_evals = int(warm["evaluations_used"])
    return {
        "label": label,
        "reason": (
            "paired delayed delta %+.6f (warm %+.6f - cold %+.6f) against a "
            "threshold of %.3f"
            % (delta, warm_delayed, cold_delayed, QUALITY_DELTA_THRESHOLD)
        ),
        "paired_delayed_delta": delta,
        "cold_delayed_aggregate_gain": cold_delayed,
        "warm_delayed_aggregate_gain": warm_delayed,
        "cold_support_aggregate_gain": float(
            cold["support"]["aggregate_gain"]
        ),
        "warm_support_aggregate_gain": float(
            warm["support"]["aggregate_gain"]
        ),
        "cold_capture_ratio": cold.get("capture_ratio"),
        "warm_capture_ratio": warm.get("capture_ratio"),
        "cold_evaluations_used": cold_evals,
        "warm_evaluations_used": warm_evals,
        "cold_llm_calls": int(cold["llm_calls"]),
        "warm_llm_calls": int(warm["llm_calls"]),
        "warm_also_cheaper": bool(
            warm_evals < cold_evals and warm_delayed >= cold_delayed
        ),
        "cost_is_not_part_of_the_label": True,
    }


# ------------------------------------------------------------------ analysis
def _field_shas(base: Mapping[str, Any]) -> dict[str, str]:
    """Per-field digest of the prompt body actually sent."""
    return {
        str(key): canonical_sha256(wvc._plain(value))
        for key, value in base.items()
    }


def _prompt_parity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which prompt fields differed between the two arms, per target.

    Computed from the digests of the prompt bodies that were actually sent, not
    from a rehearsal, so it is a statement about this run.  The comparison is on
    the shared stage-one body; the adoption stage necessarily differs, because
    each arm measured whatever its own shortlist named.
    """
    by_target: dict[str, dict[str, Any]] = {}
    for record in records:
        shas = record.get("base_input_field_shas")
        if not shas:
            continue
        by_target.setdefault(str(record["target_id"]), {})[
            str(record["arm"])
        ] = shas
    rows: dict[str, Any] = {}
    for target_id, arms in by_target.items():
        cold, warm = arms.get("cold"), arms.get("warm")
        if not cold or not warm:
            rows[target_id] = {"comparable": False}
            continue
        differing = sorted(
            key for key in set(cold) | set(warm)
            if cold.get(key) != warm.get(key)
        )
        rows[target_id] = {
            "comparable": True,
            "fields_that_differ": differing,
            "only_the_experience_section_differs": (
                differing == ["prior_batch_experience"]
            ),
            "fields_compared": len(set(cold) | set(warm)),
        }
    return {
        "scope": (
            "the stage-one prompt body; the adoption stage differs by "
            "construction because each arm measured its own shortlist"
        ),
        "all_targets_pass": all(
            row.get("only_the_experience_section_differs")
            for row in rows.values() if row.get("comparable")
        ),
        "per_target": rows,
    }


def _shortlist_divergence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does each arm's shortlist move with the cell, or not at all?"""
    out: dict[str, Any] = {}
    for arm in ("cold", "warm"):
        table = {
            str(record["target_id"]): tuple(record["shortlist"])
            for record in records
            if record["arm"] == arm and record.get("shortlist")
        }
        distinct: dict[tuple[str, ...], list[str]] = {}
        for target_id, shortlist in table.items():
            distinct.setdefault(shortlist, []).append(target_id)
        by_cohort: dict[str, set[tuple[str, ...]]] = {}
        by_variant: dict[str, set[tuple[str, ...]]] = {}
        for target_id, shortlist in table.items():
            target = TARGETS[target_id]
            by_cohort.setdefault(target["cohort"], set()).add(shortlist)
            by_variant.setdefault(
                target["consumer_variant"], set()
            ).add(shortlist)
        out[arm] = {
            "shortlist_per_target": {
                target_id: list(shortlist)
                for target_id, shortlist in sorted(table.items())
            },
            "distinct_shortlists": [
                {"shortlist": list(shortlist), "targets": sorted(targets)}
                for shortlist, targets in sorted(
                    distinct.items(), key=lambda item: -len(item[1])
                )
            ],
            "distinct_shortlist_count": len(distinct),
            "identical_across_all_targets": len(distinct) == 1,
            "largest_group_size": max(
                (len(targets) for targets in distinct.values()), default=0
            ),
            "varies_within_a_cohort": {
                cohort: len(values) > 1 for cohort, values in by_cohort.items()
            },
            "varies_within_a_consumer_variant": {
                variant: len(values) > 1
                for variant, values in by_variant.items()
            },
        }
    cold = out["cold"]["shortlist_per_target"]
    warm = out["warm"]["shortlist_per_target"]
    out["per_target_agreement"] = {
        target_id: {
            "cold": cold.get(target_id),
            "warm": warm.get(target_id),
            "identical": cold.get(target_id) == warm.get(target_id),
            "overlap": sorted(
                set(cold.get(target_id) or []) & set(warm.get(target_id) or [])
            ),
        }
        for target_id in sorted(set(cold) | set(warm))
    }
    return out


# --------------------------------------------------------------- orchestration
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    corpus = wvc._experience_corpus()
    corpus_sha_before = canonical_sha256(wvc._plain(corpus["rows"]))
    windows = {
        target_id: wvc._target_window(target)
        for target_id, target in TARGETS.items()
    }
    snapshot = compile_snapshot(
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False
    )
    records: list[dict[str, Any]] = []
    episodes: list[Any] = []
    llm_used = 0
    stopped_reason: str | None = None

    for target_id, arm in EPISODE_ORDER:
        remaining = LLM_CALL_BUDGET_TOTAL - llm_used
        if remaining < 2:
            stopped_reason = (
                "global LLM budget exhausted before %s %s (%d of %d used)"
                % (target_id, arm, llm_used, LLM_CALL_BUDGET_TOTAL)
            )
            break
        print(
            "ROT episode %s %s (llm used %d/%d)"
            % (target_id, arm, llm_used, LLM_CALL_BUDGET_TOTAL),
            flush=True,
        )
        record = _run_episode(
            target=TARGETS[target_id],
            arm=arm,
            window=windows[target_id],
            corpus=corpus,
            snapshot=snapshot,
            llm_budget=min(LLM_CALL_BUDGET_PER_EPISODE, remaining),
        )
        llm_used += int(record["llm_calls"])
        rows = wvc._visible_rows(corpus, arm, TARGETS[target_id]["cohort"])
        record["negative_experience_use"] = (
            wvc._negative_experience_use(record, rows) if arm == "warm" else {
                "not_applicable": "the cold arm sees no other-batch record"
            }
        )
        if record["adopted_plan"] is not None:
            written = wvc._experience_entry(record)
            episodes.append(written)
            record["experience_written"] = written.to_dict()
        else:
            record["experience_written"] = None
        records.append(record)
        if record["adopted_plan"] is None and len(records) == 1:
            stopped_reason = (
                "circuit breaker: the first episode produced no adopted plan"
            )
            break

    by_key = {(row["target_id"], row["arm"]): row for row in records}
    per_target = {}
    for target_id in TARGETS:
        if not any(row["target_id"] == target_id for row in records):
            continue
        verdict = _target_verdict(
            by_key.get((target_id, "cold")), by_key.get((target_id, "warm"))
        )
        per_target[target_id] = {
            "target": TARGETS[target_id],
            "window": windows[target_id],
            **verdict,
        }
    labels = [row["label"] for row in per_target.values()]
    counts = {
        label: labels.count(label)
        for label in ("WARM_WINS_QUALITY", "COLD_WINS_QUALITY", "TIE",
                      "UNREADABLE")
    }
    readable = {
        target_id: row for target_id, row in per_target.items()
        if row["paired_delayed_delta"] is not None
    }
    worst = (
        min(readable, key=lambda key: readable[key]["paired_delayed_delta"])
        if readable else None
    )
    corpus_sha_after = canonical_sha256(wvc._plain(corpus["rows"]))

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "warm vs cold budgeted recipe search rotated over all six cells on "
            "window W3, read with a corrected pre-registered criterion"
        ),
        "not_authorization_evidence": (
            "no Skill is written, no TRY right is granted, no Episode is "
            "promoted, no Fast or Slow path is entered, no snapshot pointer "
            "moves"
        ),
        "overall_verdict": {
            "warm_wins": counts["WARM_WINS_QUALITY"],
            "cold_wins": counts["COLD_WINS_QUALITY"],
            "ties": counts["TIE"],
            "unreadable": counts["UNREADABLE"],
            "targets_read": len(per_target),
            "worst_target": worst,
            "worst_target_paired_delta": (
                readable[worst]["paired_delayed_delta"] if worst else None
            ),
            "worst_target_label": readable[worst]["label"] if worst else None,
            "warm_also_cheaper_targets": sorted(
                target_id for target_id, row in per_target.items()
                if row.get("warm_also_cheaper")
            ),
            "reading_rule": (
                "the win / loss / tie count and the worst target together are "
                "the verdict; a pooled mean on its own is not"
            ),
        },
        "per_target": per_target,
        "pre_registered": PRE_REGISTERED,
        "prompt_parity_check": _prompt_parity(records),
        "shortlist_divergence": _shortlist_divergence(records),
        "prior_run": {
            "artifact": PRIOR_RUN_ARTIFACT.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "re_run": False,
            "re_labelled": False,
            "note": (
                "the two W2 targets of the earlier run are neither re-run nor "
                "re-labelled here; this rotation stands on its own six targets"
            ),
        },
        "experience_isolation": {
            "corpus_rows": corpus["row_count"],
            "corpus_sha_before": corpus_sha_before,
            "corpus_sha_after": corpus_sha_after,
            "corpus_unchanged_during_the_run": (
                corpus_sha_before == corpus_sha_after
            ),
            "rows_produced_by_this_rotation_fed_back": 0,
        },
        "model": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "experience_corpus": corpus,
        "target_windows": windows,
        "llm_call_count": llm_used,
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "stopped_early": stopped_reason,
        "experience_entries_written": [
            episode.to_dict() for episode in episodes
        ],
        "experience_provenance": EXPERIENCE_PROVENANCE,
        "episodes": records,
        "wall_seconds": time.perf_counter() - started,
    }
    if dry_run:
        print(json.dumps(
            payload["overall_verdict"], indent=2, ensure_ascii=False,
            default=str,
        ))
        return 0
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print(
        "warm_wins %d cold_wins %d ties %d worst %s"
        % (
            counts["WARM_WINS_QUALITY"], counts["COLD_WINS_QUALITY"],
            counts["TIE"], worst,
        ),
        flush=True,
    )
    print("llm_calls", llm_used, flush=True)
    return 0


# --------------------------------------------------------------------- report
def _markdown_head(payload: Mapping[str, Any]) -> list[str]:
    overall = payload["overall_verdict"]
    parity = payload["prompt_parity_check"]
    lines = [
        "# warm vs cold, rotated over all six cells (W3) v1",
        "",
        "**Overall: warm wins %d, cold wins %d, ties %d, over %d targets. "
        "Worst target `%s` at %s.**"
        % (
            overall["warm_wins"], overall["cold_wins"], overall["ties"],
            overall["targets_read"], overall["worst_target"],
            "n/a" if overall["worst_target_paired_delta"] is None
            else "%+.6f" % overall["worst_target_paired_delta"],
        ),
        "",
        "The earlier two-target run found the warm arm far ahead on quality at "
        "equal cost but had no label for it. This rotation runs the same "
        "budgeted search on all six cells, on a window that has never been a "
        "target, and reads it with a corrected criterion: the paired delayed "
        "difference decides, cost is reported beside it and never folded in.",
        "",
        "**Engineering effect measurement, not authorization evidence.** %s."
        % payload["not_authorization_evidence"],
        "",
        "The two W2 targets of `%s` are neither re-run nor re-labelled."
        % payload["prior_run"]["artifact"],
        "",
        "## 0. Pre-registered before the first call",
        "",
        "- primary readout: %s" % payload["pre_registered"]["primary_readout"],
        "- labels, first match wins:",
    ]
    for rule in payload["pre_registered"]["labels_first_match_wins"]:
        lines.append("  - %s" % rule)
    lines += [
        "- cost: %s" % payload["pre_registered"]["cost_is_reported_separately"],
        "- overall: %s" % payload["pre_registered"]["overall_verdict"],
        "- budget: shortlist capped at %d, so %d charged full-batch Support "
        "evaluations against a menu of %d; the mask round and the delayed gate "
        "are free, and the mask still runs on the highest-Support shortlisted "
        "program."
        % (
            SUPPORT_EVALUATION_BUDGET, SUPPORT_EVALUATION_BUDGET,
            len(TREATMENTS),
        ),
        "- experience isolation: %s"
        % payload["pre_registered"]["experience_isolation"],
        "",
        "Changes from the earlier run, and only these:",
        "",
    ]
    for change in payload["pre_registered"][
        "changes_from_warm_vs_cold_recipe_search_v1"
    ]:
        lines.append("- %s" % change)
    lines += [
        "",
        "## 1. Prompt parity, measured on this run",
        "",
        "Per-field digests of the stage-one prompt body actually sent to each "
        "arm. All targets pass: **%s**. Scope: %s."
        % (parity["all_targets_pass"], parity["scope"]),
        "",
        "| target | fields compared | fields that differ | only the experience "
        "section |",
        "| --- | ---: | --- | --- |",
    ]
    for target_id, row in parity["per_target"].items():
        if not row.get("comparable"):
            lines.append("| %s | n/a | n/a | n/a |" % target_id)
            continue
        lines.append(
            "| %s | %d | %s | %s |"
            % (
                target_id, row["fields_compared"],
                ", ".join("`%s`" % f for f in row["fields_that_differ"])
                or "none",
                row["only_the_experience_section_differs"],
            )
        )
    isolation = payload["experience_isolation"]
    lines += [
        "",
        "Experience isolation: the corpus is %d frozen rows, unchanged across "
        "the run (`%s`), and **%d** rows produced by this rotation were fed "
        "back into any arm."
        % (
            isolation["corpus_rows"],
            isolation["corpus_unchanged_during_the_run"],
            isolation["rows_produced_by_this_rotation_fed_back"],
        ),
        "",
    ]
    return lines


def _markdown_body(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "## 2. The twelve-row comparison",
        "",
        "| target | arm | shortlist | mask | evals | adopted plan | support | "
        "delayed | capture | harmed (s/d) | LLM |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in payload["episodes"]:
        support = record.get("support") or {}
        delayed = record.get("delayed") or {}
        lines.append(
            "| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s / %s | %d |"
            % (
                record["target_id"], record["arm"],
                ", ".join("`%s`" % p for p in record.get("shortlist", []))
                or "n/a",
                record.get("request_mask_search"),
                record["evaluations_used"],
                wvc._plan_text(record.get("adopted_plan")),
                wvc._gain(support.get("aggregate_gain")),
                wvc._gain(delayed.get("aggregate_gain")),
                wvc._ratio(record.get("capture_ratio")),
                support.get("harmed_eval_series_count", "n/a"),
                delayed.get("harmed_eval_series_count", "n/a"),
                record["llm_calls"],
            )
        )
    lines += [
        "",
        "`capture` is against the frozen full search's own delayed gain on the "
        "same cell and W3, quoted from `batch_recipe_windows_v1`. Neither arm "
        "ever saw a delayed number while choosing.",
        "",
        "## 3. Per-target labels",
        "",
        "| target | cold delayed | warm delayed | paired delta | label | "
        "cold evals | warm evals | warm also cheaper |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for target_id, row in payload["per_target"].items():
        lines.append(
            "| %s | %s | %s | %s | `%s` | %s | %s | %s |"
            % (
                target_id,
                wvc._gain(row.get("cold_delayed_aggregate_gain")),
                wvc._gain(row.get("warm_delayed_aggregate_gain")),
                wvc._gain(row.get("paired_delayed_delta")),
                row["label"],
                row.get("cold_evaluations_used", "n/a"),
                row.get("warm_evaluations_used", "n/a"),
                row.get("warm_also_cheaper"),
            )
        )
    overall = payload["overall_verdict"]
    lines += [
        "",
        "Counts: **warm %d / cold %d / tie %d** over %d readable targets. "
        "Worst target: **`%s`** at %s (`%s`). Targets where the warm arm was "
        "also cheaper: %s."
        % (
            overall["warm_wins"], overall["cold_wins"], overall["ties"],
            overall["targets_read"], overall["worst_target"],
            "n/a" if overall["worst_target_paired_delta"] is None
            else "%+.6f" % overall["worst_target_paired_delta"],
            overall["worst_target_label"],
            ", ".join(overall["warm_also_cheaper_targets"]) or "none",
        ),
        "",
        "## 4. Shortlist divergence",
        "",
    ]
    divergence = payload["shortlist_divergence"]
    for arm in ("cold", "warm"):
        row = divergence[arm]
        lines += [
            "### %s arm" % arm,
            "",
            "%d distinct shortlist(s) over %d targets; identical across all "
            "targets: **%s**; largest group: %d."
            % (
                row["distinct_shortlist_count"],
                len(row["shortlist_per_target"]),
                row["identical_across_all_targets"],
                row["largest_group_size"],
            ),
            "",
            "| shortlist | targets |",
            "| --- | --- |",
        ]
        for group in row["distinct_shortlists"]:
            lines.append(
                "| %s | %s |"
                % (
                    ", ".join("`%s`" % p for p in group["shortlist"]),
                    ", ".join(group["targets"]),
                )
            )
        lines += [
            "",
            "Varies within a cohort: %s. Varies within a consumer variant: %s."
            % (
                json.dumps(row["varies_within_a_cohort"], sort_keys=True),
                json.dumps(
                    row["varies_within_a_consumer_variant"], sort_keys=True
                ),
            ),
            "",
        ]
    lines += [
        "### Cold against warm, per target",
        "",
        "| target | cold shortlist | warm shortlist | identical | overlap |",
        "| --- | --- | --- | --- | --- |",
    ]
    for target_id, row in divergence["per_target_agreement"].items():
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                target_id,
                ", ".join("`%s`" % p for p in (row["cold"] or [])) or "n/a",
                ", ".join("`%s`" % p for p in (row["warm"] or [])) or "n/a",
                row["identical"],
                ", ".join("`%s`" % p for p in row["overlap"]) or "none",
            )
        )
    lines.append("")
    return lines


def _markdown_tail(payload: Mapping[str, Any]) -> list[str]:
    lines = ["## 5. Negative-experience use, warm arm only", ""]
    lines += [
        "| target | losing somewhere | losing everywhere measured | "
        "shortlisted anyway | skipped |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in payload["episodes"]:
        use = record.get("negative_experience_use") or {}
        if "not_applicable" in use:
            continue
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (
                record["target_id"],
                ", ".join("`%s`" % p for p in use[
                    "programs_seen_losing_somewhere"]) or "none",
                ", ".join("`%s`" % p for p in use[
                    "programs_seen_losing_everywhere_they_were_measured"])
                or "none",
                ", ".join("`%s`" % p for p in use[
                    "losing_everywhere_programs_shortlisted"]) or "none",
                ", ".join("`%s`" % p for p in use[
                    "losing_everywhere_programs_skipped"]) or "none",
            )
        )
    lines += ["", "## 6. Episode reasons", ""]
    for record in payload["episodes"]:
        lines += [
            "**%s** -- shortlist %s (mask %s): %s"
            % (
                record["episode_id"],
                ", ".join("`%s`" % p for p in record.get("shortlist", []))
                or "n/a",
                record.get("request_mask_search"),
                record.get("shortlist_reason", ""),
            ),
            "",
            "  adopted %s: %s"
            % (
                wvc._plan_text(record.get("adopted_plan")),
                record.get("adoption_reason", ""),
            ),
            "",
        ]
        retries = [
            "%s=%s%s" % (
                info["stage"], info["validation_retry_count"],
                info["validation_error_codes"] or "",
            )
            for info in record["stages"]
        ]
        lines += ["  retries: %s" % ", ".join(retries), ""]
    lines += [
        "## 7. Experience entries written",
        "",
        "Provenance `%s`, `counts_as_unguided_exploration: false`, and **not "
        "fed back**: the warm arm's corpus is the 18 frozen records only."
        % payload["experience_provenance"],
        "",
        "| episode | cell | plan | support | delayed | relation |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for entry in payload["experience_entries_written"]:
        support = entry["support_response"]
        delayed = entry["delayed_response"]
        lines.append(
            "| `%s` | `%s` | `%s` minus %s | %s | %s | %s |"
            % (
                entry["episode_id"],
                str(entry["task_consumer_key"]).replace("|", r"\|"),
                support.get("program"),
                ", ".join(support.get("excluded_series") or []) or "nothing",
                wvc._gain(support.get("gain")),
                wvc._gain(delayed.get("gain")),
                entry["relation"],
            )
        )
    lines += [
        "",
        "## 8. What this does not say",
        "",
        "- It does not authorize anything and it promotes no plan.",
        "- Six targets, one window, one model, one draw per cell. A per-target "
        "label is a comparison of two single runs, not a rate, which is why "
        "the count and the worst target are reported rather than a mean.",
        "- The capture denominator is the frozen full search's delayed gain, "
        "itself selected on that delayed window. Capture near 1 means 'as good "
        "as the unbudgeted search got', not 'optimal'.",
        "- The warm arm's ceiling is what the corpus holds: 18 records from "
        "three cohorts, leave-one-cohort-out.",
        "- The mask round still runs on the highest-Support shortlisted "
        "program. That frozen rule, not the experience, decides which program "
        "gets a mask at all.",
        "",
        "## Provenance",
        "",
        "- model: `%s` at `%s`"
        % (payload["model"]["model"], payload["model"]["base_url"]),
        "- instrument, corpus, validators, observation table and Experience "
        "writer: imported from `run_e2_warm_vs_cold_recipe_search`, which is "
        "not modified",
        "- windows and reference plans: quoted from `%s`"
        % payload["target_windows"][
            next(iter(payload["target_windows"]))]["quoted_from"],
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
        help="run the rotation but print the counts instead of writing",
    )
    args = parser.parse_args(argv)
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
