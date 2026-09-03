"""warm vs cold budgeted recipe search: does other-batch Experience buy a
cheaper route to the same processing plan?

This is the A5-vs-A3 question at the recipe layer.  Two arms search for a batch
processing plan under the same evaluation budget; the only difference between
them is whether they can see what other batches already produced.

The instrument is a budgeted search, deterministic and shared by both arms:

* at most **4 full-batch Support evaluations**, one per shortlisted program.
  The menu holds 7 programs, so scanning it whole would cost 7 and the budget
  binds: both arms must choose;
* at most **one round of greedy exclusion-mask search**, run on the best
  shortlisted program by Support gain.  Its internal per-series evaluations are
  the instrument's fixed cost and are not counted against the budget, exactly as
  in the frozen recipe;
* one **delayed-gate evaluation** of the adopted plan at the end.  Both arms pay
  it, it does not count against the budget, and its result is revealed only
  after the plan is fixed.

The identity baseline is instrument overhead, not a candidate: every gain is
measured against it, so it is computed once per (cell, window) and charged to
neither arm.

The Agent decides twice per episode.  First, after reading the target batch's
public observation table -- and, in the warm arm, the other-batch Experience
table -- it submits a program shortlist of at most four and says whether it
wants the mask search.  Second, after the Support numbers come back, it names
the plan to adopt.  The delayed column is never visible to it.

Two targets, four episodes, order fixed before the first call:

1. target A cold, 2. target A warm, 3. target B cold, 4. target B warm.

* **Target A**: T233 x pooled on window W2.  Full-search reference: winsorize
  at full batch, delayed +0.270423.
* **Target B**: traffic x per_channel on window W2.  Full-search reference:
  outlier_iqr with series 8 reverted, delayed +0.439241.  This one is a
  deliberately hard transfer: the other cohorts' per_channel experience points
  at denoise_median and winsorize, which is not what is true here, so an Agent
  that copies the majority vote instead of reading the target's own observation
  should lose.

The warm arm sees the 18 verified (cell x window x adopted plan x support /
delayed) records, **leave-one-cohort-out**: every row measured on the target's
own cohort is withheld.  Rows carry their cell key, their plan, both gains and,
where the source artifact recorded one, the full menu scan -- so the warm arm
also sees which programs *hurt* on other batches, for example denoise_median at
-0.635 on traffic pooled.  The cold arm sees the same prompt with that one
section reporting that no such records are available.

Nothing else differs.  Same public observation table, same menu, same budget,
same schemas, same instrument, same wording.

Not authorization evidence: no Skill is written, no TRY right is granted, no
Episode is promoted, no Fast or Slow path is entered.  Experience entries this
run writes carry ``provenance="budgeted_search_engineering"`` and are not fed
back into either arm.

Run:

    python evaluation/functional/run_e2_warm_vs_cold_recipe_search.py

Writes ``artifacts/functional/e2/warm_vs_cold_recipe_search_v1.json`` and
``.md``.
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
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_sha256,
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
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentCallBudgetExceeded,
    AgentTransportError,
)

PROTOCOL_VERSION = "warm_vs_cold_recipe_search_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "warm_vs_cold_recipe_search_v1.json"
OUT_MD = E2 / "warm_vs_cold_recipe_search_v1.md"
WINDOWS_ARTIFACT = E2 / "batch_recipe_windows_v1.json"
ALL_CELLS_ARTIFACT = E2 / "batch_recipe_v2_all_cells_v1.json"
V1_ARTIFACTS = {
    "electricity": E2 / "batch_recipe_electricity_v1.json",
    "T233": E2 / "batch_recipe_T233_v1.json",
    "traffic": E2 / "batch_recipe_traffic_v1.json",
}

EXPERIENCE_PROVENANCE = "budgeted_search_engineering"
TREATMENTS: tuple[str, ...] = tuple(bch.TREATMENTS)
IDENTITY = bch.IDENTITY

SUPPORT_EVALUATION_BUDGET = 4
MASK_SEARCH_ROUNDS = 1
LLM_CALL_BUDGET_TOTAL = 40
LLM_CALL_BUDGET_PER_EPISODE = 6
VALIDATION_RETRIES = 2
CAPTURE_QUALITY_BAR = 0.9

# Targets and arms, fixed before the first call and not reordered afterwards.
TARGETS: dict[str, dict[str, Any]] = {
    "A": {
        "target_id": "A",
        "cohort": "T233",
        "consumer_variant": bch.CONSUMER_POOLED,
        "window_id": "W2",
        "why": "a cell whose full search settles on a full-batch plan",
    },
    "B": {
        "target_id": "B",
        "cohort": "traffic",
        "consumer_variant": bch.CONSUMER_PER_CHANNEL,
        "window_id": "W2",
        "why": (
            "a deliberately hard transfer: the other cohorts' per_channel "
            "experience points at denoise_median and winsorize, while this "
            "cell's own full search settles on outlier_iqr with one series "
            "reverted"
        ),
    },
}
EPISODE_ORDER: tuple[tuple[str, str], ...] = (
    ("A", "cold"), ("A", "warm"), ("B", "cold"), ("B", "warm"),
)

OBSERVATION_FIELDS: tuple[str, ...] = (
    "missing_fraction",
    "longest_missing_run_fraction",
    "local_robust_z_peak",
    "estimated_region_start_fraction",
    "estimated_region_end_fraction",
    "level_region_fraction",
    "level_excursion_score",
    "estimated_level_offset",
    "period_change_score",
    "period_reliability",
    "post_shift_support_sufficient",
)

PRE_REGISTERED = {
    "fixed_before_the_first_llm_call": True,
    "episode_order": [
        "%s %s" % (target, arm) for target, arm in EPISODE_ORDER
    ],
    "budget": {
        "full_batch_support_evaluations": SUPPORT_EVALUATION_BUDGET,
        "mask_search_rounds": MASK_SEARCH_ROUNDS,
        "mask_search_internal_evaluations": (
            "instrument fixed cost, not counted against the budget, as in the "
            "frozen recipe"
        ),
        "identity_baseline": (
            "instrument overhead, computed once per cell and window, charged "
            "to neither arm"
        ),
        "delayed_gate_evaluation": (
            "one per episode, paid by both arms, not counted, revealed only "
            "after the plan is fixed"
        ),
        "llm_calls_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_per_episode": LLM_CALL_BUDGET_PER_EPISODE,
        "validation_retries_per_stage": VALIDATION_RETRIES,
    },
    "cost_metric": (
        "evaluations_used = the number of shortlisted programs, i.e. the "
        "number of full-batch Support evaluations the instrument had to run"
    ),
    "quality_metric": (
        "capture_ratio = the adopted plan's delayed aggregate gain divided by "
        "the delayed aggregate gain the frozen full search reached on the same "
        "cell and window, quoted verbatim from batch_recipe_windows_v1"
    ),
    "per_target_verdict_first_match_wins": [
        "WARM_HARMS_QUALITY: the warm arm's capture ratio is below the cold "
        "arm's",
        "WARM_REDUCES_COST_AT_QUALITY: the warm arm used strictly fewer "
        "evaluations and its capture ratio is at least %.2f"
        % CAPTURE_QUALITY_BAR,
        "WARM_NO_BENEFIT: anything else",
    ],
    "overall_verdict": (
        "both targets agreeing gives that verdict; otherwise MIXED, reported "
        "per target"
    ),
    "behaviour_is_read_from_the_payload": (
        "shortlist, mask request and adopted plan are read from the stage "
        "payloads and the instrument's own log, never from the Agent's prose"
    ),
    "circuit_breaker": (
        "stop and report if the first episode produces no payload"
    ),
    "experience_relation_rule": (
        "POSITIVE if the adopted plan's support and delayed gains are both "
        "positive; CONFLICT if they disagree in sign; NEGATIVE if both are "
        "non-positive and the plan treats something; ABSTAIN if the plan is "
        "identity"
    ),
}


# --------------------------------------------------------------- experience
def _experience_corpus() -> dict[str, Any]:
    """The 18 verified records, read from the three existing artifacts.

    The (cell x window x adopted plan x support / delayed) rows come from
    ``batch_recipe_windows_v1``.  Window-1 rows are enriched with the full menu
    scan recorded in ``batch_recipe_v2_all_cells_v1``, which is where the
    negative evidence lives -- a program's full-batch Support gain on a batch
    where it lost.  The three ``batch_recipe_<cohort>_v1`` artifacts are the
    v1-rule ancestors of three of those window-1 rows; their menu scans are
    checked against the v2 ones and they contribute no distinct record.
    """
    windows = json.loads(WINDOWS_ARTIFACT.read_text(encoding="utf-8"))
    all_cells = json.loads(ALL_CELLS_ARTIFACT.read_text(encoding="utf-8"))
    menu: dict[tuple[str, str], dict[str, float]] = {}
    best_full: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in all_cells["cells"]:
        recipe = cell.get("recipe")
        if recipe is None:
            continue
        key = (str(cell["cohort"]), str(cell["consumer_variant"]))
        menu[key] = {
            str(program): float(row["aggregate_gain"])
            for program, row in recipe["menu_scan"].items()
        }
        best_full[key] = {
            "program": str(recipe["comparison"]["best_full_batch_program"]),
            "support_aggregate_gain": float(
                recipe["comparison"]["support"]["best_full_batch"]
            ),
            "delayed_aggregate_gain": float(
                recipe["comparison"]["delayed"]["best_full_batch"]
            ),
        }
    v1_checks: list[dict[str, Any]] = []
    for cohort, path in V1_ARTIFACTS.items():
        if not path.is_file():
            continue
        v1 = json.loads(path.read_text(encoding="utf-8"))
        scan = {
            str(program): float(row["aggregate_gain"])
            for program, row in v1["menu_scan"].items()
        }
        reference = menu.get((cohort, bch.CONSUMER_POOLED))
        v1_checks.append({
            "artifact": path.relative_to(PROJECT_ROOT).as_posix(),
            "cohort": cohort,
            "adoption_rule_version": v1.get("adoption_rule_version", "v1"),
            "menu_scan_matches_v2_pooled": bool(reference == scan),
            "adds_a_distinct_record": False,
        })

    rows: list[dict[str, Any]] = []
    for cell in windows["cells"]:
        cohort = str(cell["cohort"])
        variant = str(cell["consumer_variant"])
        key = (cohort, variant)
        for window in cell["windows"]:
            plan = window["adopted_plan"]
            row = {
                "record_id": "%s|%s|%s" % (cohort, variant, window["window_id"]),
                "cohort": cohort,
                "consumer_variant": variant,
                "cell_key": "batch:%s|consumer:%s" % (cohort, variant),
                "window_id": str(window["window_id"]),
                "support_origins": list(window["support_origins"]),
                "delayed_origins": list(window["delayed_origins"]),
                "adopted_plan": {
                    "kind": str(plan["kind"]),
                    "program": str(plan["program"]),
                    "excluded_series": list(plan["excluded_series"]),
                },
                "support_aggregate_gain": float(
                    window["support_aggregate_gain"]
                ),
                "delayed_aggregate_gain": float(
                    window["delayed_aggregate_gain"]
                ),
                "harmed_eval_series_count": int(
                    window["harmed_eval_series_count"]
                ),
                "best_full_batch": (
                    {
                        "program": str(window["best_full_batch_program"]),
                        "support_aggregate_gain": float(
                            window["best_full_batch_support"]
                        ),
                        "delayed_aggregate_gain": float(
                            window["best_full_batch_delayed"]
                        ),
                    }
                    if "best_full_batch_program" in window
                    else best_full.get(key)
                ),
                "full_batch_menu_scan_support_gain": (
                    menu.get(key) if window["window_id"] == "W1" else None
                ),
            }
            rows.append(row)
    return {
        "rows": rows,
        "row_count": len(rows),
        "sources": {
            "records": WINDOWS_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
            "menu_scans_for_window_1": ALL_CELLS_ARTIFACT.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "v1_rule_ancestors": v1_checks,
        },
        "menu_scan_coverage": (
            "recorded for the six window-1 rows; the twelve W2/W3 rows carry "
            "their window's best full-batch program and both of its gains "
            "instead"
        ),
    }


def _visible_rows(
    corpus: Mapping[str, Any], arm: str, target_cohort: str
) -> list[dict[str, Any]]:
    """Leave-one-cohort-out for the warm arm; nothing at all for the cold arm."""
    if arm == "cold":
        return []
    return [
        dict(row) for row in corpus["rows"]
        if str(row["cohort"]) != str(target_cohort)
    ]


# ------------------------------------------------------------- the instrument
class BudgetedSearch:
    """The shared, deterministic search instrument.  No LLM, no file written.

    Everything measurable is the recipe module's: ``_evaluate_variant`` and
    ``_evaluate_assignment`` for the retrains, ``_gain_rows`` for the gain, the
    same Consumer variant and the same windows.  What this class adds is the
    budget: full-batch Support evaluations are counted, the mask search's
    internal per-series evaluations are the instrument's fixed cost, and the
    identity baseline and the delayed gate are charged to neither arm.
    """

    def __init__(
        self,
        *,
        cohort: str,
        consumer_variant: str,
        support_origins: Sequence[int],
        delayed_origins: Sequence[int],
    ) -> None:
        self.cohort = str(cohort)
        self.consumer_variant = str(consumer_variant)
        self.support = tuple(int(origin) for origin in support_origins)
        self.delayed = tuple(int(origin) for origin in delayed_origins)
        loaded = bch.load_cohort(PROJECT_ROOT, self.cohort)
        self.config = dict(_config())
        self.roster = loaded["mapped_roster"]
        self.values = loaded["values"]
        self.train_uids = [str(uid) for uid in loaded["train_uids"]]
        self.eval_uids = [str(uid) for uid in loaded["eval_uids"]]
        self.exposure = str(loaded["exposure"])
        self._compiled: dict[str, Any] = {}
        self._identity_support = bch._evaluate_variant(
            self.roster, self.values, None, self.config, self.support, None,
            self.consumer_variant,
        )
        self._identity_delayed = bch._evaluate_variant(
            self.roster, self.values, None, self.config, self.delayed, None,
            self.consumer_variant,
        )
        self.support_evaluations_charged = 0
        self.internal_evaluations = 0
        self.log: list[dict[str, Any]] = []

    def _program(self, program: str) -> Any:
        compiled = self._compiled.get(program)
        if compiled is None:
            compiled = _compiled(program, name="wvc_%s" % program)
            self._compiled[program] = compiled
        return compiled

    def _gains(
        self, rows: Sequence[Mapping[str, Any]], *, delayed: bool = False
    ) -> dict[str, Any]:
        gains = bch._gain_rows(
            self._identity_delayed if delayed else self._identity_support,
            rows,
            self.eval_uids,
        )
        return {
            "aggregate_gain": float(gains["aggregate_gain"]),
            "harmed_eval_series_count": int(gains["harmed_eval_series_count"]),
            "harmed_eval_series_total_harm": float(
                gains["harmed_eval_series_total_harm"]
            ),
            "harmed_eval_series": list(gains["harmed_eval_series"]),
            # O1 (#19 OBSERVATION_PROJECTION_GAP repair): stop projecting at
            # this interface.  ``bch._gain_rows`` already measured the
            # per-evaluation-series gain vector on the same rows; the four
            # lines above keep returning exactly what they returned before,
            # and the one line below passes the measured vector through
            # untouched so the Scope/Risk guard evaluation context can read
            # it.  Measurement semantics are unchanged: same rows in, same
            # aggregates out, one passthrough key added.
            "per_eval_series_gain": dict(gains["per_eval_series_gain"]),
        }

    def _scoped(self, program: str, scope: set[str] | None, origins):
        return bch._evaluate_variant(
            self.roster, self.values, self._program(program), self.config,
            origins, scope, self.consumer_variant,
        )

    def _masked(self, program: str, excluded: set[str], origins):
        assignment = {
            uid: (None if uid in excluded else self._program(program))
            for uid in self.train_uids
        }
        return [
            bch._evaluate_assignment(
                self.roster, self.values, assignment, self.config,
                origin=origin, consumer_variant=self.consumer_variant,
            )
            for origin in origins
        ]

    # -- the three cost classes -------------------------------------------
    def full_batch_support(self, program: str) -> dict[str, Any]:
        """One charged evaluation: this program applied to the whole batch."""
        self.support_evaluations_charged += 1
        gains = self._gains(self._scoped(program, None, self.support))
        self.log.append({
            "kind": "full_batch_support",
            "program": program,
            "charged": True,
            "aggregate_gain": gains["aggregate_gain"],
        })
        return gains

    def mask_search(self, program: str) -> dict[str, Any]:
        """One greedy exclusion round on one program.  Not charged.

        Line for line the frozen recipe's search: order the revert queue by
        ascending singleton per-series Support gain, revert one series at a
        time, keep a revert only if the measured Support aggregate improved,
        and stop at the first revert that does not.
        """
        singleton: dict[str, float] = {}
        for uid in self.train_uids:
            self.internal_evaluations += 1
            singleton[uid] = self._gains(
                self._scoped(program, {uid}, self.support)
            )["aggregate_gain"]
        order = sorted(self.train_uids, key=lambda uid: singleton[uid])
        self.internal_evaluations += 1
        current = self._gains(self._scoped(program, None, self.support))
        excluded: set[str] = set()
        steps: list[dict[str, Any]] = []
        for step, uid in enumerate(order[: int(bch.MASKED_MAX_STEPS)], start=1):
            trial = excluded | {uid}
            self.internal_evaluations += 1
            trial_gains = self._gains(self._masked(program, trial, self.support))
            improved = (
                trial_gains["aggregate_gain"] > current["aggregate_gain"]
            )
            steps.append({
                "step": step,
                "reverted_series": uid,
                "support_aggregate_gain": trial_gains["aggregate_gain"],
                "delta": (
                    trial_gains["aggregate_gain"] - current["aggregate_gain"]
                ),
                "decision": "ACCEPTED" if improved else "REJECTED_AND_STOPPED",
            })
            if not improved:
                break
            excluded = trial
            current = trial_gains
        self.log.append({
            "kind": "mask_search", "program": program, "charged": False,
            "final_excluded": sorted(excluded),
            "aggregate_gain": current["aggregate_gain"],
        })
        return {
            "program": program,
            "final_excluded": sorted(excluded),
            "support": current,
            "steps": steps,
            "revert_order": order,
            "charged": False,
        }

    def delayed_gate(
        self, program: str, excluded_series: Sequence[str]
    ) -> dict[str, Any]:
        """The delayed reading of the adopted plan.  Paid by both arms."""
        excluded = {str(uid) for uid in excluded_series}
        rows = (
            self._identity_delayed if program == IDENTITY
            else self._masked(program, excluded, self.delayed)
        )
        gains = self._gains(rows, delayed=True)
        self.log.append({
            "kind": "delayed_gate", "program": program, "charged": False,
            "excluded_series": sorted(excluded),
            "aggregate_gain": gains["aggregate_gain"],
        })
        return gains

    def support_of_plan(
        self, program: str, excluded_series: Sequence[str]
    ) -> dict[str, Any]:
        """Support reading of the adopted plan.  Bookkeeping, not charged."""
        excluded = {str(uid) for uid in excluded_series}
        if program == IDENTITY:
            return self._gains(self._identity_support)
        self.internal_evaluations += 1
        return self._gains(self._masked(program, excluded, self.support))

    def accounting(self) -> dict[str, Any]:
        return {
            "full_batch_support_evaluations_charged": (
                self.support_evaluations_charged
            ),
            "budget": SUPPORT_EVALUATION_BUDGET,
            "instrument_internal_evaluations_not_charged": (
                self.internal_evaluations
            ),
            "log": [dict(row) for row in self.log],
        }


# ------------------------------------------------------------------- prompts
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


def _observation_table(search: BudgetedSearch) -> list[dict[str, Any]]:
    """Public features of each training series on its public prefix.

    The same table is served to both arms.  It is the one-shot form of what the
    ``summarize_series`` Workspace tool serves per series, computed on
    ``values[uid][:support_origins[0]]`` through the same extractor, so no tool
    round is needed and neither arm can differ by how well it drives a tool.
    """
    cutoff = int(search.support[0])
    rows: list[dict[str, Any]] = []
    for uid in search.train_uids:
        values = np.asarray(search.values[uid], dtype=np.float64)[:cutoff]
        features = dict(extract_public_features(values, task_kind="forecast"))
        row: dict[str, Any] = {"series_uid": uid}
        for field in OBSERVATION_FIELDS:
            if field in features:
                row[field] = _plain(features[field])
        rows.append(row)
    return rows


class NoToolGateway:
    """A gateway that serves nothing.  Neither arm gets a Workspace tool."""

    def __init__(self, identity: Mapping[str, Any]) -> None:
        self._context_sha = canonical_sha256(
            {"schema_version": "warm-vs-cold-no-tool-context/1", **dict(identity)}
        )

    @property
    def context_sha(self) -> str:
        return self._context_sha

    def schemas_for(self, *, role: Any, stage: str) -> tuple[Any, ...]:
        return ()

    def call(self, name: str, arguments: Mapping[str, Any]) -> Any:
        raise PermissionError("no Workspace tool is served in this experiment")


SHORTLIST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "budgeted-shortlist/1",
    "type": "object",
    "additionalProperties": False,
    "required": ["shortlist", "request_mask_search", "reason"],
    "properties": {
        "shortlist": {
            "type": "array",
            "items": {"enum": list(TREATMENTS)},
            "minItems": 1,
            "maxItems": SUPPORT_EVALUATION_BUDGET,
            "uniqueItems": True,
        },
        "request_mask_search": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}

ADOPTION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "budgeted-adoption/1",
    "type": "object",
    "additionalProperties": False,
    "required": ["program", "excluded_series", "reason"],
    "properties": {
        "program": {"enum": list(TREATMENTS) + [IDENTITY]},
        "excluded_series": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
            "maxItems": 24,
        },
        "reason": {"type": "string"},
    },
}

SHORTLIST_NOTE = (
    "Choose which programs are worth spending the evaluation budget on. "
    "`shortlist` names at most %d programs from the menu, in the order you "
    "would try them; each one costs exactly one full-batch Support evaluation "
    "and the whole menu is %d programs, so a full scan does not fit. "
    "`request_mask_search` asks for one greedy exclusion-mask round, which is "
    "free of budget: if you request it, it is run on whichever shortlisted "
    "program scores highest on Support, it reverts training series to identity "
    "one at a time, and it keeps a revert only when the measured Support "
    "aggregate improves. `reason` is one or two sentences on why this "
    "shortlist, in public terms. You will see the Support numbers next and "
    "then choose the plan; the delayed window is never shown to you."
    % (SUPPORT_EVALUATION_BUDGET, len(TREATMENTS))
)

ADOPTION_NOTE = (
    "The Support results are in. Name the plan to adopt: `program` is one of "
    "the programs you shortlisted, or `identity` to treat nothing. "
    "`excluded_series` must be empty, or exactly the series the mask search "
    "reverted -- no other mask was ever measured, so no other mask can be "
    "adopted. `reason` is one or two sentences on why, in public terms. The "
    "delayed reading of whatever you adopt happens after this and is not shown "
    "to you first."
)


def _base_public_input(
    *,
    target: Mapping[str, Any],
    window: Mapping[str, Any],
    search: BudgetedSearch,
    observation: Sequence[Mapping[str, Any]],
    experience_rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[str, Any]:
    """The prompt body.  Identical in both arms except one section."""
    if arm == "warm":
        experience = {
            "available": True,
            "row_count": len(experience_rows),
            "withheld": (
                "every record measured on this target's own cohort (%s) is "
                "withheld from this table" % target["cohort"]
            ),
            "how_to_read": (
                "each row is a plan another batch already adopted and what it "
                "scored there. `cell_key` names the batch and the Consumer "
                "structure it was measured on, `window_id` the development "
                "window. `full_batch_menu_scan_support_gain`, where it is "
                "present, is every menu program's full-batch Support gain on "
                "that batch, so a negative entry is a program that lost there."
            ),
            "rows": [dict(row) for row in experience_rows],
        }
    else:
        experience = {
            "available": False,
            "row_count": 0,
            "withheld": "no records from other batches are available in this arm",
            "how_to_read": (
                "there is no other-batch record to read in this arm; the "
                "target's own public observation is the only evidence"
            ),
            "rows": [],
        }
    return {
        "schema_version": "budgeted-search-input/1",
        "target": {
            "cohort": target["cohort"],
            "consumer_variant": target["consumer_variant"],
            "cell_key": "batch:%s|consumer:%s"
            % (target["cohort"], target["consumer_variant"]),
            "window_id": window["window_id"],
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
            "this_target_uses": target["consumer_variant"],
        },
        "program_menu": list(TREATMENTS),
        "identity_is_always_available": True,
        "evaluation_budget": {
            "full_batch_support_evaluations": SUPPORT_EVALUATION_BUDGET,
            "menu_size": len(TREATMENTS),
            "mask_search_rounds": MASK_SEARCH_ROUNDS,
            "mask_search_costs_budget": False,
            "mask_search_target": (
                "the shortlisted program with the highest full-batch Support "
                "gain"
            ),
            "identity_baseline": "free; every gain is measured against it",
            "delayed_window": "evaluated once at the end and never shown to you",
        },
        "public_observation": {
            "rule": (
                "public features of each training series on its own public "
                "prefix values[uid][:observation_cutoff]"
            ),
            "rows": [dict(row) for row in observation],
        },
        "prior_batch_experience": experience,
    }


def _make_shortlist_validator():
    def validate(payload: Mapping[str, Any]) -> None:
        shortlist = [str(item) for item in payload["shortlist"]]
        outside = sorted(set(shortlist) - set(TREATMENTS))
        if outside:
            raise StagePostValidationError(
                "PROGRAM_OUTSIDE_MENU",
                "shortlist names programs that are not in the menu: %s"
                % outside,
                retryable=True,
            )
        if len(shortlist) != len(set(shortlist)):
            raise StagePostValidationError(
                "SHORTLIST_NOT_DISTINCT",
                "shortlist repeats a program; each entry costs one evaluation",
                retryable=True,
            )

    return validate


def _make_adoption_validator(
    *, shortlist: Sequence[str], mask_result: Mapping[str, Any] | None
):
    """The plan must be one the instrument actually measured."""
    allowed = set(str(item) for item in shortlist) | {IDENTITY}
    mask_program = None if mask_result is None else str(mask_result["program"])
    mask_set = (
        set() if mask_result is None
        else {str(uid) for uid in mask_result["final_excluded"]}
    )

    def validate(payload: Mapping[str, Any]) -> None:
        program = str(payload["program"])
        excluded = {str(uid) for uid in payload.get("excluded_series", ())}
        if program not in allowed:
            raise StagePostValidationError(
                "PROGRAM_NOT_EVALUATED",
                "program %s was not shortlisted and not evaluated; adopt one "
                "of %s or identity" % (program, sorted(allowed - {IDENTITY})),
                retryable=True,
            )
        if program == IDENTITY and excluded:
            raise StagePostValidationError(
                "IDENTITY_PLAN_INCONSISTENT",
                "identity treats nothing, so excluded_series must be empty",
                retryable=True,
            )
        if excluded and (mask_result is None or program != mask_program):
            raise StagePostValidationError(
                "MASK_NOT_MEASURED",
                "the only mask the instrument measured is the one the mask "
                "search produced%s; adopt it with its own program or adopt an "
                "empty mask"
                % (
                    "" if mask_result is None
                    else " (%s reverting %s)"
                    % (mask_program, sorted(mask_set) or "nothing")
                ),
                retryable=True,
            )
        if excluded and excluded != mask_set:
            raise StagePostValidationError(
                "MASK_NOT_MEASURED",
                "excluded_series must be exactly the mask search's result %s "
                "or empty" % (sorted(mask_set) or "(empty)"),
                retryable=True,
            )

    return validate


# ------------------------------------------------------------------- episode
def _stage(
    core: TTHAAgentCore,
    *,
    stage: str,
    case_id: str,
    public_input: Mapping[str, Any],
    harness_view: Any,
    schema_name: str,
    schema: Mapping[str, Any],
    validator: Any,
) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
    info: dict[str, Any] = {
        "stage": stage, "protocol_error": None, "infrastructure_error": None,
        "validation_retry_count": None, "validation_error_codes": [],
        "first_pass_valid": False,
    }
    try:
        result = core.run_stage(
            role=AgentRole.FAST,
            stage=stage,
            case_id=case_id,
            public_input=public_input,
            harness_view=harness_view,
            output_schema_name=schema_name,
            output_schema=schema,
            source_snapshot_sha=harness_view.effective_harness_view_sha,
            validation_retries=VALIDATION_RETRIES,
            post_validator=validator,
        )
    except (AgentProtocolError, StagePostValidationError, PermissionError) as exc:
        info["protocol_error"] = "%s: %s" % (type(exc).__name__, exc)
        return None, info
    except (AgentTransportError, AgentCallBudgetExceeded) as exc:
        info["infrastructure_error"] = "%s: %s" % (type(exc).__name__, exc)
        return None, info
    info["validation_retry_count"] = int(result.validation_retry_count)
    info["validation_error_codes"] = list(result.validation_error_codes)
    info["first_pass_valid"] = bool(result.first_pass_valid)
    return dict(result.payload), info


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
    search = BudgetedSearch(
        cohort=target["cohort"],
        consumer_variant=target["consumer_variant"],
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )
    observation = _observation_table(search)
    experience_rows = _visible_rows(corpus, arm, target["cohort"])
    base = _base_public_input(
        target=target, window=window, search=search, observation=observation,
        experience_rows=experience_rows, arm=arm,
    )
    backend = _default_backend_factory(int(llm_budget))
    gateway = NoToolGateway({"episode_id": episode_id, "arm": arm})
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)
    harness_view = resolve_harness_view(snapshot, {}, role="fast")

    shortlist_payload, shortlist_info = _stage(
        core,
        stage="budgeted_shortlist",
        case_id="WVC_%s" % episode_id,
        public_input={**base, "stage_note": SHORTLIST_NOTE},
        harness_view=harness_view,
        schema_name="budgeted_shortlist_v1",
        schema=SHORTLIST_SCHEMA,
        validator=_make_shortlist_validator(),
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
        "shortlist_payload": _plain(shortlist_payload),
        "adopted_plan": None,
        "support": None,
        "delayed": None,
        "capture_ratio": None,
        "evaluations_used": 0,
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
    print(
        "WVC %s shortlist=%s mask=%s best=%s (%+.6f)"
        % (episode_id, shortlist, wants_mask, best,
           support_results[best]["aggregate_gain"]),
        flush=True,
    )
    record["shortlist"] = shortlist
    record["request_mask_search"] = wants_mask
    record["best_shortlisted_program"] = best
    record["support_results"] = {
        program: support_results[program] for program in shortlist
    }
    record["mask_search"] = _plain(mask_result)
    record["evaluations_used"] = int(search.support_evaluations_charged)
    record["shortlist_reason"] = str(shortlist_payload.get("reason", ""))
    return _finish_episode(
        record=record, core=core, base=base, harness_view=harness_view,
        search=search, shortlist=shortlist, support_results=support_results,
        mask_result=mask_result, window=window, backend=backend,
        episode_id=episode_id, started=started,
    )


def _finish_episode(
    *, record, core, base, harness_view, search, shortlist, support_results,
    mask_result, window, backend, episode_id, started,
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
        "evaluations_spent": int(search.support_evaluations_charged),
    }
    adoption_payload, adoption_info = _stage(
        core,
        stage="budgeted_adoption",
        case_id="WVC_%s" % episode_id,
        public_input=adoption_input,
        harness_view=harness_view,
        schema_name="budgeted_adoption_v1",
        schema=ADOPTION_SCHEMA,
        validator=_make_adoption_validator(
            shortlist=shortlist, mask_result=mask_result,
        ),
    )
    record["stages"].append(adoption_info)
    record["adoption_payload"] = _plain(adoption_payload)
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
        "llm_calls": int(backend.calls),
        "wall_seconds": time.perf_counter() - started,
    })
    print(
        "WVC %s adopted %s minus %s | support %+.6f delayed %+.6f capture "
        "%.3f | evals %d llm %d"
        % (
            episode_id, plan["program"],
            ", ".join(plan["excluded_series"]) or "nothing",
            support["aggregate_gain"], delayed["aggregate_gain"],
            record["capture_ratio"] if record["capture_ratio"] is not None
            else float("nan"),
            record["evaluations_used"], record["llm_calls"],
        ),
        flush=True,
    )
    return record


def _experience_entry(record: Mapping[str, Any]) -> Any:
    """One episode's own record, through the existing episode mechanism."""
    plan = record["adopted_plan"]
    support_gain = float(record["support"]["aggregate_gain"])
    delayed_gain = float(record["delayed"]["aggregate_gain"])
    if plan["program"] == IDENTITY:
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
            "engineering measurement from a budget-constrained recipe search "
            "arm; not authorization evidence and not an unguided probe, and "
            "not fed back into either arm of this run"
        ),
    }
    return build_episode(
        episode_id=record["episode_id"],
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
            "gain": support_gain,
            "window": "support",
            "program": plan["program"],
            "excluded_series": list(plan["excluded_series"]),
            "arm": str(record["arm"]),
            "evaluations_used": int(record["evaluations_used"]),
            "harmed_eval_series_count": int(
                record["support"]["harmed_eval_series_count"]
            ),
            **audit,
        },
        delayed_response={
            "gain": delayed_gain,
            "window": "delayed",
            "capture_ratio": record["capture_ratio"],
            "window_role": (
                "revealed only after the plan was fixed; it took no part in "
                "the search"
            ),
            "harmed_eval_series_count": int(
                record["delayed"]["harmed_eval_series_count"]
            ),
            **audit,
        },
        relation=relation,
        evidence_level=EVIDENCE_DELAYED,
        local_status=STATUS_EPISODE_ONLY,
        evidence_refs=(EXPERIENCE_PROVENANCE,),
    )


# ------------------------------------------------------------------ readouts
def _target_window(target: Mapping[str, Any]) -> dict[str, Any]:
    """The target's window and its full-search reference, quoted verbatim."""
    data = json.loads(WINDOWS_ARTIFACT.read_text(encoding="utf-8"))
    cell = next(
        row for row in data["cells"]
        if str(row["cohort"]) == target["cohort"]
        and str(row["consumer_variant"]) == target["consumer_variant"]
    )
    window = next(
        row for row in cell["windows"]
        if str(row["window_id"]) == target["window_id"]
    )
    definition = next(
        row for row in data["window_definitions"][target["cohort"]]
        if str(row["window_id"]) == target["window_id"]
    )
    return {
        "window_id": str(window["window_id"]),
        "support_origins": [int(o) for o in window["support_origins"]],
        "delayed_origins": [int(o) for o in window["delayed_origins"]],
        "origin_provenance": str(definition["origin_provenance"]),
        "origin_source": str(definition["origin_source"]),
        "quoted_from": WINDOWS_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
        "reference_plan": {
            "kind": str(window["adopted_plan"]["kind"]),
            "program": str(window["adopted_plan"]["program"]),
            "excluded_series": list(window["adopted_plan"]["excluded_series"]),
        },
        "reference_support_aggregate_gain": float(
            window["support_aggregate_gain"]
        ),
        "reference_delayed_aggregate_gain": float(
            window["delayed_aggregate_gain"]
        ),
        "reference_note": (
            "the frozen full search's own answer on this cell and window: an "
            "unbudgeted 7-program menu scan plus a mask search on its top two"
        ),
    }


def _negative_experience_use(
    record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Did the warm arm act on the losing rows it was shown?

    Read from the payload and the corpus, never from the Agent's prose: which
    programs the visible menu scans show losing on another batch, and whether
    those programs made it into the shortlist.
    """
    negatives: dict[str, list[dict[str, Any]]] = {}
    positives: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scan = row.get("full_batch_menu_scan_support_gain")
        if not scan:
            continue
        for program, gain in scan.items():
            bucket = negatives if float(gain) < 0.0 else positives
            bucket.setdefault(str(program), []).append({
                "cell_key": row["cell_key"],
                "window_id": row["window_id"],
                "full_batch_support_gain": float(gain),
            })
    shortlist = {str(item) for item in record.get("shortlist", ())}
    losing = set(negatives)
    always_losing = {
        program for program in losing if program not in positives
    }
    return {
        "menu_scan_rows_visible": sum(
            1 for row in rows if row.get("full_batch_menu_scan_support_gain")
        ),
        "programs_seen_losing_somewhere": sorted(losing),
        "programs_seen_losing_everywhere_they_were_measured": sorted(
            always_losing
        ),
        "shortlisted": sorted(shortlist),
        "losing_programs_shortlisted": sorted(losing & shortlist),
        "losing_everywhere_programs_shortlisted": sorted(
            always_losing & shortlist
        ),
        "losing_everywhere_programs_skipped": sorted(always_losing - shortlist),
        "per_program_negative_rows": {
            program: rows_ for program, rows_ in sorted(negatives.items())
        },
    }


def _target_verdict(
    cold: Mapping[str, Any] | None, warm: Mapping[str, Any] | None
) -> dict[str, Any]:
    if (
        cold is None or warm is None
        or cold.get("capture_ratio") is None
        or warm.get("capture_ratio") is None
    ):
        return {
            "verdict": "UNREADABLE",
            "reason": "one of the two arms produced no adopted plan",
        }
    cold_capture = float(cold["capture_ratio"])
    warm_capture = float(warm["capture_ratio"])
    cold_evals = int(cold["evaluations_used"])
    warm_evals = int(warm["evaluations_used"])
    if warm_capture < cold_capture:
        verdict = "WARM_HARMS_QUALITY"
        reason = (
            "warm capture %.3f is below cold capture %.3f"
            % (warm_capture, cold_capture)
        )
    elif warm_evals < cold_evals and warm_capture >= CAPTURE_QUALITY_BAR:
        verdict = "WARM_REDUCES_COST_AT_QUALITY"
        reason = (
            "warm spent %d evaluations against cold's %d and captured %.3f, "
            "at or above the %.2f bar"
            % (warm_evals, cold_evals, warm_capture, CAPTURE_QUALITY_BAR)
        )
    else:
        verdict = "WARM_NO_BENEFIT"
        reason = (
            "warm spent %d evaluations against cold's %d and captured %.3f "
            "against cold's %.3f"
            % (warm_evals, cold_evals, warm_capture, cold_capture)
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "cold_capture_ratio": cold_capture,
        "warm_capture_ratio": warm_capture,
        "cold_evaluations_used": cold_evals,
        "warm_evaluations_used": warm_evals,
        "capture_quality_bar": CAPTURE_QUALITY_BAR,
    }


# --------------------------------------------------------------- orchestration
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    corpus = _experience_corpus()
    windows = {
        target_id: _target_window(target)
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
            "WVC episode %s %s (llm used %d/%d)"
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
        rows = _visible_rows(corpus, arm, TARGETS[target_id]["cohort"])
        record["negative_experience_use"] = (
            _negative_experience_use(record, rows) if arm == "warm" else {
                "not_applicable": "the cold arm sees no other-batch record"
            }
        )
        if record["adopted_plan"] is not None:
            written = _experience_entry(record)
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
    per_target = {
        target_id: {
            "target": TARGETS[target_id],
            "window": windows[target_id],
            **_target_verdict(
                by_key.get((target_id, "cold")), by_key.get((target_id, "warm"))
            ),
        }
        for target_id in TARGETS
        if any(row["target_id"] == target_id for row in records)
    }
    verdicts = {
        target_id: row["verdict"] for target_id, row in per_target.items()
    }
    distinct = set(verdicts.values())
    if not verdicts:
        overall = "UNREADABLE"
    elif len(distinct) == 1:
        overall = next(iter(distinct))
    else:
        overall = "MIXED_BY_TARGET"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "warm vs cold budgeted recipe search: does other-batch Experience "
            "buy a cheaper route to the same processing plan under a binding "
            "evaluation budget"
        ),
        "not_authorization_evidence": (
            "no Skill is written, no TRY right is granted, no Episode is "
            "promoted, no Fast or Slow path is entered, no snapshot pointer "
            "moves"
        ),
        "overall_verdict": overall,
        "per_target": per_target,
        "pre_registered": PRE_REGISTERED,
        "model": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "arms": {
            "cold": "no other-batch record is shown",
            "warm": (
                "the 18 verified records minus every row measured on the "
                "target's own cohort (leave-one-cohort-out)"
            ),
            "everything_else_identical": (
                "same public observation table, same menu, same budget, same "
                "instrument, same schemas, same wording; the prompt differs "
                "only in the prior_batch_experience section"
            ),
        },
        "experience_corpus": corpus,
        "target_windows": windows,
        "llm_call_count": llm_used,
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "stopped_early": stopped_reason,
        "experience_entries_written": [
            episode.to_dict() for episode in episodes
        ],
        "experience_provenance": EXPERIENCE_PROVENANCE,
        "experience_entries_are_not_fed_back": True,
        "episodes": records,
        "wall_seconds": time.perf_counter() - started,
    }
    if dry_run:
        print(json.dumps(per_target, indent=2, ensure_ascii=False, default=str))
        return 0
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("overall_verdict", overall, flush=True)
    print("llm_calls", llm_used, flush=True)
    return 0


# --------------------------------------------------------------------- report
def _gain(value: Any) -> str:
    return "n/a" if value is None else "%+.6f" % float(value)


def _ratio(value: Any) -> str:
    return "n/a" if value is None else "%.3f" % float(value)


def _plan_text(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "n/a"
    return "`%s` minus %s" % (
        plan["program"], ", ".join(plan.get("excluded_series") or []) or "nothing",
    )


def _markdown_head(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# warm vs cold budgeted recipe search v1",
        "",
        "**Overall verdict: `%s`**" % payload["overall_verdict"],
        "",
        "Two arms search for a batch processing plan under the same evaluation "
        "budget. The only difference between them is whether they can see what "
        "other batches already produced. This is the A5-vs-A3 question at the "
        "recipe layer.",
        "",
        "**Engineering effect measurement, not authorization evidence.** %s."
        % payload["not_authorization_evidence"],
        "",
        "## 0. The instrument and the budget",
        "",
        "| cost class | charged to the arm? |",
        "| --- | --- |",
        "| full-batch Support evaluation, one per shortlisted program | **yes, "
        "at most %d** |" % SUPPORT_EVALUATION_BUDGET,
        "| greedy exclusion-mask round on the best shortlisted program | no |",
        "| the mask round's internal per-series evaluations | no (instrument "
        "fixed cost) |",
        "| identity baseline | no (every gain is measured against it) |",
        "| delayed-gate evaluation of the adopted plan | no (both arms pay it, "
        "revealed only after the plan is fixed) |",
        "",
        "The menu holds %d programs and the budget is %d, so a full scan does "
        "not fit and both arms must choose. Everything measurable is the "
        "recipe module's `_evaluate_variant` / `_evaluate_assignment` / "
        "`_gain_rows` on the same windows and Consumer variant."
        % (len(TREATMENTS), SUPPORT_EVALUATION_BUDGET),
        "",
        "## 1. Arms",
        "",
        "- **cold**: %s" % payload["arms"]["cold"],
        "- **warm**: %s" % payload["arms"]["warm"],
        "- everything else: %s" % payload["arms"]["everything_else_identical"],
        "",
        "The corpus is %d records from `%s`, with window-1 menu scans from "
        "`%s`."
        % (
            payload["experience_corpus"]["row_count"],
            payload["experience_corpus"]["sources"]["records"],
            payload["experience_corpus"]["sources"]["menu_scans_for_window_1"],
        ),
        "",
        "## 2. Targets, quoted verbatim",
        "",
        "| target | cell | window | support origins | delayed origins | "
        "full-search reference plan | reference delayed |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for target_id, window in payload["target_windows"].items():
        target = TARGETS[target_id]
        lines.append(
            "| %s | %s x %s | %s | %s | %s | %s | %s |"
            % (
                target_id, target["cohort"], target["consumer_variant"],
                window["window_id"], window["support_origins"],
                window["delayed_origins"],
                _plan_text(window["reference_plan"]),
                _gain(window["reference_delayed_aggregate_gain"]),
            )
        )
    lines += [
        "",
        "Origins are quoted from `%s`; none was newly chosen here."
        % payload["target_windows"]["A"]["quoted_from"],
        "",
    ]
    return lines


def _markdown_body(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "## 3. The four-row comparison",
        "",
        "| target | arm | shortlist | mask asked | evals | adopted plan | "
        "support | delayed | capture | harmed (s/d) | LLM |",
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
                _plan_text(record.get("adopted_plan")),
                _gain(support.get("aggregate_gain")),
                _gain(delayed.get("aggregate_gain")),
                _ratio(record.get("capture_ratio")),
                support.get("harmed_eval_series_count", "n/a"),
                delayed.get("harmed_eval_series_count", "n/a"),
                record["llm_calls"],
            )
        )
    lines += [
        "",
        "`capture` is the adopted plan's delayed gain over the frozen full "
        "search's delayed gain on the same cell and window. The delayed column "
        "was never shown to either arm.",
        "",
        "## 4. Per-target verdicts",
        "",
        "Rules, fixed before the first call, first match wins:",
        "",
    ]
    for index, rule in enumerate(
        payload["pre_registered"]["per_target_verdict_first_match_wins"],
        start=1,
    ):
        lines.append("%d. %s" % (index, rule))
    lines += ["", "| target | verdict | why |", "| --- | --- | --- |"]
    for target_id, row in payload["per_target"].items():
        lines.append(
            "| %s (%s x %s) | `%s` | %s |"
            % (
                target_id, row["target"]["cohort"],
                row["target"]["consumer_variant"], row["verdict"],
                row["reason"],
            )
        )
    lines += [
        "",
        "Overall: **`%s`**." % payload["overall_verdict"],
        "",
        "## 5. Episode by episode",
        "",
    ]
    for record in payload["episodes"]:
        lines += [
            "### %s -- target %s, %s arm" % (
                record["episode_id"], record["target_id"], record["arm"],
            ),
            "",
            "- experience rows visible: %d (%s)"
            % (
                record["experience_rows_visible"],
                ", ".join(record["experience_cohorts_visible"]) or "none",
            ),
            "- shortlist: %s; mask search asked: %s"
            % (
                ", ".join("`%s`" % p for p in record.get("shortlist", []))
                or "n/a",
                record.get("request_mask_search"),
            ),
            "- shortlist reason: %s" % record.get("shortlist_reason", ""),
            "- Support results: %s"
            % (
                "; ".join(
                    "%s %+.6f" % (program, row["aggregate_gain"])
                    for program, row in (
                        record.get("support_results") or {}
                    ).items()
                ) or "none"
            ),
        ]
        mask = record.get("mask_search")
        if mask:
            lines.append(
                "- mask search on `%s`: reverted %s, Support %+.6f"
                % (
                    mask["program"],
                    ", ".join(mask["final_excluded"]) or "nothing",
                    mask["support"]["aggregate_gain"],
                )
            )
        lines += [
            "- adopted: %s" % _plan_text(record.get("adopted_plan")),
            "- adoption reason: %s" % record.get("adoption_reason", ""),
            "- matches the full-search reference plan: %s"
            % record.get("matches_reference_plan"),
        ]
        for info in record["stages"]:
            if info["protocol_error"] or info["infrastructure_error"]:
                lines.append(
                    "- stage `%s` error: %s"
                    % (
                        info["stage"],
                        info["protocol_error"] or info["infrastructure_error"],
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
        use = record.get("negative_experience_use") or {}
        if "not_applicable" not in use:
            lines += [
                "- losing programs visible in the menu scans: %s"
                % (", ".join("`%s`" % p for p in use[
                    "programs_seen_losing_somewhere"]) or "none"),
                "- of those, losing everywhere they were measured: %s"
                % (", ".join("`%s`" % p for p in use[
                    "programs_seen_losing_everywhere_they_were_measured"])
                   or "none"),
                "- shortlisted anyway: %s; skipped: %s"
                % (
                    ", ".join("`%s`" % p for p in use[
                        "losing_everywhere_programs_shortlisted"]) or "none",
                    ", ".join("`%s`" % p for p in use[
                        "losing_everywhere_programs_skipped"]) or "none",
                ),
            ]
        lines.append("")
    return lines


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = _markdown_head(payload) + _markdown_body(payload)
    lines += [
        "## 6. Experience entries written",
        "",
        "Written through the existing episode mechanism, "
        "`provenance=\"%s\"`, `counts_as_unguided_exploration: false`, and "
        "**not fed back into either arm** -- the warm arm's corpus is the "
        "frozen 18-record set only."
        % payload["experience_provenance"],
        "",
        "| episode | cell | plan | support | delayed | relation | provenance |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for entry in payload["experience_entries_written"]:
        support = entry["support_response"]
        delayed = entry["delayed_response"]
        lines.append(
            "| `%s` | `%s` | `%s` minus %s | %s | %s | %s | `%s` |"
            % (
                entry["episode_id"],
                str(entry["task_consumer_key"]).replace("|", r"\|"),
                support.get("program"),
                ", ".join(support.get("excluded_series") or []) or "nothing",
                _gain(support.get("gain")),
                _gain(delayed.get("gain")),
                entry["relation"],
                support.get("provenance"),
            )
        )
    lines += [
        "",
        "## 7. What this does not say",
        "",
        "- It does not authorize anything and it does not promote any plan.",
        "- Two targets and four episodes on one model is a mechanism reading, "
        "not a rate. A per-target verdict is a comparison of two single draws.",
        "- The capture denominator is the frozen full search's own delayed "
        "gain, which was selected on that same delayed window. Capture near 1 "
        "means 'as good as the unbudgeted search got', not 'optimal'.",
        "- The warm arm's advantage, if any, is bounded by what the corpus "
        "contains: 18 records from three cohorts, leave-one-cohort-out.",
        "- Neither arm ever saw a delayed number while choosing.",
        "",
        "## Provenance",
        "",
        "- model: `%s` at `%s`"
        % (payload["model"]["model"], payload["model"]["base_url"]),
        "- instrument: `run_batch_composition_headroom._evaluate_variant` / "
        "`_evaluate_assignment` / `_gain_rows`, imported and not modified",
        "- windows and reference plans: quoted from `%s`"
        % payload["target_windows"]["A"]["quoted_from"],
        "- LLM calls: %d of %d"
        % (payload["llm_call_count"], payload["llm_call_budget_total"]),
        "- wall seconds: %.1f" % payload["wall_seconds"],
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run the arms but print the verdicts instead of writing artifacts",
    )
    args = parser.parse_args(argv)
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
