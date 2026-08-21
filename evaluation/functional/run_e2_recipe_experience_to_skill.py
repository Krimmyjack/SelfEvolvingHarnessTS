"""recipe experience -> Source Skill -> Fast: does the signal survive the bridge?

The warm/cold rotation showed that a table of other-batch recipe records
improves plan quality at equal instrument cost, 4 wins / 1 loss / 1 tie over six
cells.  Two things were missing.  The signal never went through the project's
legitimate knowledge channel -- Slow compiles, a Source-derived Skill carries,
Fast reads -- it was pasted into the prompt as a table.  And the cost account
was incomplete: the mask round's internal per-series retrains were never
counted.  This run builds the bridge and closes the account.

**Part one, a deterministic Slow compiler.**  Zero LLM.  The rules are frozen in
code before any run and no clause is hand-written.  Input is the committed
frozen recipe rows; for each target every row measured on the target's own
cohort is dropped (leave-one-cohort-out).  Three rules, thresholds hard-coded:

* **R1, priority**: within the same ``consumer_variant``, a program with a
  delayed-positive adopted or full-batch record in at least two distinct
  cohorts becomes a "try first" clause, ordered by cross-cohort mean delayed;
* **R2, risk**: a program with a delayed-negative full-batch record in at least
  two distinct cohorts becomes a "deprioritize" clause;
* **R3, mask locality**: if the windows artifact's mask-stable share is at most
  one third, masks must be re-searched locally and never reused.

R1 and R2 read only rows that actually record a delayed number; a Support
reading is never substituted for a missing delayed one.  A rule with no usable
row produces nothing.  A target whose three rules all produce nothing gets
``ABSTAIN_TO_DEFAULT``, is labelled as such, and nothing is written in to fill
the gap -- in that case A5 should behave exactly like A3.

**Part two, A5 against A3** on the budgeted search instrument, with two
pre-registered instrument corrections:

* the adoption gate gains the identity incumbent;
* the bar is drawn from identity at zero plus the evaluated full-batch plans
  **whose Support is positive**.  The literal v2 rule in
  ``run_batch_composition_headroom`` was read before this was implemented and
  the finding is recorded in ``V2_GATE_SEMANTICS`` below: v2 takes the delayed
  of exactly one program, the highest-Support full-batch one, floored at zero.
  It never checks that program's Support sign.  The earlier negative-path run
  copied a wider "max over all evaluated full-batch delayed" and paid for it:
  a plan with Support -0.0023 set a bar of +0.084 and knocked out a plan that
  equalled the frozen reference.  The correction here is the narrower,
  eligibility-checked bar, and the difference from v2 is stated, not hidden.

Every Consumer retrain is counted: shortlist evaluations, the mask round's
internal per-series retrains, the identity baselines and the delayed gate.  Both
arms carry the same caps.  A5's context gains exactly one thing, the rendered
Skill card; A3 has no Source content.  Everything else in the prompt is
identical and the per-field digests are written into the artifact.

Not authorization evidence: no Skill is promoted, no TRY right is granted, no
Fast or Slow path of the real Harness runs.  The "Skill card" here is a
rendered text object this runner compiled and handed to the Agent; it is not
installed anywhere.

Run:

    python evaluation/functional/run_e2_recipe_experience_to_skill.py

Writes ``artifacts/functional/e2/recipe_skill_cards_v1.json``,
``artifacts/functional/e2/recipe_skill_bridge_v1.json`` and ``.md``.
"""
from __future__ import annotations

import argparse
import hashlib
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

from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    g3_sourcing,
)
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

PROTOCOL_VERSION = "recipe_skill_bridge_v1"
CARDS_PROTOCOL_VERSION = "recipe_skill_cards_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
CARDS_JSON = E2 / "recipe_skill_cards_v1.json"
OUT_JSON = E2 / "recipe_skill_bridge_v1.json"
OUT_MD = E2 / "recipe_skill_bridge_v1.md"
WINDOWS_ARTIFACT = E2 / "batch_recipe_windows_v1.json"
ALL_CELLS_ARTIFACT = E2 / "batch_recipe_v2_all_cells_v1.json"
V1_ARTIFACTS = {
    cohort: E2 / ("batch_recipe_%s_v1.json" % cohort)
    for cohort in ("electricity", "T233", "traffic")
}

EXPERIENCE_PROVENANCE = "skill_bridge_engineering"
TREATMENTS = wvc.TREATMENTS
IDENTITY = wvc.IDENTITY

EVALUATION_BUDGET = 2
QUALITY_DELTA_THRESHOLD = 0.005
LLM_CALL_BUDGET_TOTAL = 40
LLM_CALL_BUDGET_PER_EPISODE = 5
VALIDATION_RETRIES = wvc.VALIDATION_RETRIES

# Compiler thresholds, frozen before any run.
R1_MIN_COHORTS = 2
R2_MIN_COHORTS = 2
R3_MASK_STABLE_MAX_SHARE = 1.0 / 3.0

V2_GATE_SEMANTICS = {
    "checked_before_implementing": True,
    "source": "evaluation/functional/run_batch_composition_headroom.py",
    "what_v2_actually_does": (
        "best_full_program = ranked[0] where ranked sorts the full-batch "
        "programs by descending Support aggregate gain; bar_delayed = "
        "full_batch_delayed[best_full]; delayed_bar = max(bar_delayed, 0.0). "
        "So the bar is the delayed gain of exactly one program -- the "
        "highest-Support full-batch one -- floored at identity's zero, and the "
        "sign of that program's Support is never checked"
    ),
    "what_this_runner_does": (
        "bar = max(0.0, max delayed over the evaluated full-batch plans whose "
        "Support gain is strictly positive)"
    ),
    "difference_and_why": (
        "v2 reads one program; this reads the max over the Support-positive "
        "ones, which is never lower than identity and never lets a plan nobody "
        "could adopt set the bar. The negative-path run copied a wider 'max "
        "over all evaluated full-batch delayed' with no eligibility check: on "
        "its E3 a plan with Support -0.002312 and delayed +0.084153 set the "
        "bar and knocked out a plan that equalled the frozen W4 reference"
    ),
}

# The three targets, fixed before the first call.
TARGETS: dict[str, dict[str, Any]] = {
    "T1": {
        "target_id": "T1",
        "cohort": "T233",
        "consumer_variant": bch.CONSUMER_POOLED,
        "window_id": "e1v2_task_04",
        "task_index": 3,
        "origin_source": "quoted from the frozen roster",
        "origin_provenance": (
            "task_episode_harness.e1._frozen_task_roster()[3], e1v2_task_04, "
            "support and delayed origins verbatim. The negative-path run used "
            "the same origins on T233 x per_channel; this is the pooled cell, "
            "a different Consumer structure and therefore a different batch "
            "problem, so the reuse is legitimate and is noted here"
        ),
        "why": "a cell with real headroom under the pooled Consumer",
    },
    "T2": {
        "target_id": "T2",
        "cohort": "electricity",
        "consumer_variant": bch.CONSUMER_PER_CHANNEL,
        "window_id": "e1v2_task_04",
        "task_index": 3,
        "origin_source": "quoted from the frozen roster",
        "origin_provenance": (
            "task_episode_harness.e1._frozen_task_roster()[3], e1v2_task_04, "
            "support and delayed origins verbatim"
        ),
        "why": (
            "the hard target: this cell's own answer is denoise_median, and "
            "every denoise_median record lives on electricity rows, which "
            "leave-one-cohort-out removes. The question here is "
            "non-inferiority, not winning"
        ),
    },
    "T3": {
        "target_id": "T3",
        "cohort": "traffic",
        "consumer_variant": bch.CONSUMER_POOLED,
        "window_id": "W4_traffic_shift",
        "task_index": 0,
        "traffic_origins": (2208, 2472, 2904),
        "origin_source": "chosen",
        "origin_provenance": (
            "chosen, not quoted: the W1..W3 traffic shape and spacing shifted "
            "by a further +1104 inside the same pre-sealed development region. "
            "The sealed boundary was verified against the frozen split config "
            "before the run, not assumed"
        ),
        "why": "a high-headroom traffic cell on a window never used before",
    },
}
ARM_ORDER: tuple[tuple[str, str], ...] = tuple(
    (target_id, arm) for target_id in ("T1", "T2", "T3") for arm in ("A3", "A5")
)

PRE_REGISTERED = {
    "fixed_before_the_first_llm_call": True,
    "compiler": {
        "rules": {
            "R1_priority": (
                "within the same consumer_variant, a program with a "
                "delayed-positive adopted or full-batch record in at least %d "
                "distinct cohorts, ordered by cross-cohort mean delayed"
                % R1_MIN_COHORTS
            ),
            "R2_risk": (
                "a program with a delayed-negative full-batch record in at "
                "least %d distinct cohorts" % R2_MIN_COHORTS
            ),
            "R3_mask_locality": (
                "if the windows artifact's mask-stable share is at most %.4f, "
                "masks must be re-searched locally and never reused"
                % R3_MASK_STABLE_MAX_SHARE
            ),
        },
        "delayed_only": (
            "R1 and R2 read only rows that record a delayed number; a Support "
            "reading is never substituted for a missing delayed one"
        ),
        "empty_is_empty": (
            "a rule with no usable row produces nothing; a target whose three "
            "rules all produce nothing is ABSTAIN_TO_DEFAULT and A5 should "
            "then behave exactly like A3"
        ),
        "loco": "every row measured on the target's own cohort is dropped",
        "zero_llm": True,
    },
    "instrument_corrections": [
        "the adoption gate gains the identity incumbent",
        "the bar is identity at zero plus the evaluated full-batch plans whose "
        "Support is strictly positive; see V2_GATE_SEMANTICS for how that "
        "differs from the literal v2 rule",
    ],
    "cost_accounting": (
        "every Consumer retrain is counted: identity baselines, shortlist "
        "evaluations, the mask round's internal per-series retrains, and the "
        "delayed gate. One evaluation over three origins is three retrains"
    ),
    "arms": {
        "A3": "no Source content at all",
        "A5": (
            "exactly one addition: the rendered Skill card. The raw experience "
            "table is never shown"
        ),
    },
    "delivery_check": (
        "A5's stage-one payload must cite at least one clause id that exists "
        "on its card; otherwise the target is SKILL_NOT_DELIVERED regardless "
        "of who won"
    ),
    "per_target_label": (
        "paired delayed delta = A5 delayed minus A3 delayed; > +%.3f is "
        "A5_WINS, < -%.3f is A5_LOSES, otherwise TIE"
        % (QUALITY_DELTA_THRESHOLD, QUALITY_DELTA_THRESHOLD)
    ),
    "overall_verdict_in_this_order": [
        "SKILL_COMPILATION_INSUFFICIENT_PROVENANCE: all three targets compiled "
        "to ABSTAIN_TO_DEFAULT",
        "SKILL_LOSES_SIGNAL: at least one target is A5_LOSES; the failing "
        "clause is named",
        "SKILL_BRIDGE_DELIVERS: delivery confirmed on every non-abstaining "
        "target, no A5_LOSES, and at least one A5_WINS",
        "SKILL_BRIDGE_NO_EFFECT: added before the run because the three labels "
        "above are not exhaustive -- delivery holds, nothing loses, nothing "
        "wins",
    ],
    "cost_report_has_no_criterion": (
        "total retrains per arm-target, and the cumulative retrains each arm "
        "spent before its first delayed-positive adoption, are reported as a "
        "first measurement and decide nothing"
    ),
    "budget": {
        "charged_evaluations_per_arm_target": EVALUATION_BUDGET,
        "llm_calls_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_per_arm_target": LLM_CALL_BUDGET_PER_EPISODE,
        "validation_retries_per_stage": VALIDATION_RETRIES,
    },
    "arm_order": ["%s %s" % (target, arm) for target, arm in ARM_ORDER],
    "circuit_breaker": "stop if the first arm-target produces no payload",
    "no_cross_target_feedback": (
        "the three targets are independent; nothing this run produces is fed "
        "into another target"
    ),
}


# ------------------------------------------------------------ the compiler
def _delayed_records() -> list[dict[str, Any]]:
    """Every committed row that records a delayed number, with its provenance.

    Only rows that actually carry a delayed gain are collected.  Support-only
    rows -- the menu scans, for instance -- are deliberately left out: R1 and R2
    are delayed rules and a Support reading may not stand in for a missing
    delayed one.
    """
    rows: list[dict[str, Any]] = []

    def add(program, cohort, variant, window, delayed, kind, full_batch, prov):
        rows.append({
            "program": str(program),
            "cohort": str(cohort),
            "consumer_variant": str(variant),
            "window_id": str(window),
            "delayed_aggregate_gain": float(delayed),
            "kind": kind,
            "full_batch": bool(full_batch),
            "provenance": dict(prov),
        })

    windows = json.loads(WINDOWS_ARTIFACT.read_text(encoding="utf-8"))
    windows_name = WINDOWS_ARTIFACT.relative_to(PROJECT_ROOT).as_posix()
    for cell in windows["cells"]:
        cohort, variant = str(cell["cohort"]), str(cell["consumer_variant"])
        for window in cell["windows"]:
            key = "cells[cohort=%s,consumer_variant=%s].windows[%s]" % (
                cohort, variant, window["window_id"],
            )
            plan = window["adopted_plan"]
            add(
                plan["program"], cohort, variant, window["window_id"],
                window["delayed_aggregate_gain"], "adopted",
                not plan["excluded_series"],
                {"artifact": windows_name,
                 "key": key + ".adopted_plan + .delayed_aggregate_gain"},
            )
            if "best_full_batch_program" in window:
                add(
                    window["best_full_batch_program"], cohort, variant,
                    window["window_id"], window["best_full_batch_delayed"],
                    "full_batch", True,
                    {"artifact": windows_name,
                     "key": key + ".best_full_batch_program + "
                            ".best_full_batch_delayed"},
                )
    all_cells = json.loads(ALL_CELLS_ARTIFACT.read_text(encoding="utf-8"))
    all_name = ALL_CELLS_ARTIFACT.relative_to(PROJECT_ROOT).as_posix()
    for cell in all_cells["cells"]:
        recipe = cell.get("recipe")
        if recipe is None:
            continue
        cohort, variant = str(cell["cohort"]), str(cell["consumer_variant"])
        add(
            recipe["comparison"]["best_full_batch_program"], cohort, variant,
            "W1", recipe["comparison"]["delayed"]["best_full_batch"],
            "full_batch", True,
            {"artifact": all_name,
             "key": "cells[cohort=%s,consumer_variant=%s].recipe.comparison."
                    "best_full_batch_program + .delayed.best_full_batch"
                    % (cohort, variant)},
        )
    for cohort, path in V1_ARTIFACTS.items():
        if not path.is_file():
            continue
        v1 = json.loads(path.read_text(encoding="utf-8"))
        add(
            v1["comparison"]["best_full_batch_program"], cohort,
            bch.CONSUMER_POOLED, "W1",
            v1["comparison"]["delayed"]["best_full_batch"], "full_batch", True,
            {"artifact": path.relative_to(PROJECT_ROOT).as_posix(),
             "key": "comparison.best_full_batch_program + "
                    ".delayed.best_full_batch",
             "note": "v1 adoption rule; the menu scan on this artifact records "
                     "Support only and is therefore not used by R1 or R2"},
        )
    return rows


def _mask_stability(target_cohort: str) -> dict[str, Any]:
    """The windows artifact's mask-stable share, on the LOCO-filtered cells."""
    windows = json.loads(WINDOWS_ARTIFACT.read_text(encoding="utf-8"))
    kept = [
        cell for cell in windows["cells"]
        if str(cell["cohort"]) != target_cohort
    ]
    stable = [
        "%s x %s" % (cell["cohort"], cell["consumer_variant"])
        for cell in kept if cell["stability"]["mask_stable"]
    ]
    share = (len(stable) / len(kept)) if kept else 0.0
    all_stable = [
        "%s x %s" % (cell["cohort"], cell["consumer_variant"])
        for cell in windows["cells"] if cell["stability"]["mask_stable"]
    ]
    return {
        "loco_cells": len(kept),
        "loco_mask_stable_cells": stable,
        "loco_mask_stable_share": share,
        "all_cells": len(windows["cells"]),
        "all_mask_stable_cells": all_stable,
        "all_mask_stable_share": (
            len(all_stable) / len(windows["cells"]) if windows["cells"] else 0.0
        ),
        "threshold": R3_MASK_STABLE_MAX_SHARE,
        "provenance": {
            "artifact": WINDOWS_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
            "key": "cells[].stability.mask_stable",
        },
    }


def compile_skill_card(target: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one Source-derived Skill card.  Deterministic, 0 LLM."""
    cohort = str(target["cohort"])
    variant = str(target["consumer_variant"])
    records = _delayed_records()
    kept = [row for row in records if row["cohort"] != cohort]
    dropped = [row for row in records if row["cohort"] == cohort]

    # ---- R1 --------------------------------------------------------------
    r1_pool: dict[str, list[dict[str, Any]]] = {}
    for row in kept:
        if row["consumer_variant"] != variant:
            continue
        if row["delayed_aggregate_gain"] <= 0.0:
            continue
        if row["kind"] not in ("adopted", "full_batch"):
            continue
        if row["program"] == IDENTITY:
            continue
        r1_pool.setdefault(row["program"], []).append(row)
    r1_clauses: list[dict[str, Any]] = []
    for program, rows in r1_pool.items():
        cohorts = sorted({row["cohort"] for row in rows})
        if len(cohorts) < R1_MIN_COHORTS:
            continue
        per_cohort = {
            name: sum(
                row["delayed_aggregate_gain"] for row in rows
                if row["cohort"] == name
            ) / sum(1 for row in rows if row["cohort"] == name)
            for name in cohorts
        }
        r1_clauses.append({
            "rule": "R1",
            "program": program,
            "cohorts": cohorts,
            "cross_cohort_mean_delayed": sum(per_cohort.values()) / len(cohorts),
            "per_cohort_mean_delayed": per_cohort,
            "supporting_rows": [
                {
                    "cohort": row["cohort"], "window_id": row["window_id"],
                    "kind": row["kind"],
                    "delayed_aggregate_gain": row["delayed_aggregate_gain"],
                    "provenance": row["provenance"],
                }
                for row in rows
            ],
        })
    r1_clauses.sort(key=lambda row: -row["cross_cohort_mean_delayed"])
    for index, clause in enumerate(r1_clauses, start=1):
        clause["clause_id"] = "R1-%d" % index
        clause["text"] = (
            "Try `%s` early on this Consumer structure: it holds a "
            "delayed-positive record on %d cohorts (%s), cross-cohort mean "
            "delayed %+.6f."
            % (
                clause["program"], len(clause["cohorts"]),
                ", ".join(clause["cohorts"]),
                clause["cross_cohort_mean_delayed"],
            )
        )

    # ---- R2 --------------------------------------------------------------
    r2_pool: dict[str, list[dict[str, Any]]] = {}
    for row in kept:
        if not row["full_batch"] or row["kind"] != "full_batch":
            continue
        if row["delayed_aggregate_gain"] >= 0.0:
            continue
        if row["program"] == IDENTITY:
            continue
        r2_pool.setdefault(row["program"], []).append(row)
    r2_clauses: list[dict[str, Any]] = []
    for program, rows in sorted(r2_pool.items()):
        cohorts = sorted({row["cohort"] for row in rows})
        if len(cohorts) < R2_MIN_COHORTS:
            continue
        r2_clauses.append({
            "rule": "R2",
            "program": program,
            "cohorts": cohorts,
            "worst_delayed": min(
                row["delayed_aggregate_gain"] for row in rows
            ),
            "supporting_rows": [
                {
                    "cohort": row["cohort"], "window_id": row["window_id"],
                    "delayed_aggregate_gain": row["delayed_aggregate_gain"],
                    "provenance": row["provenance"],
                }
                for row in rows
            ],
        })
    for index, clause in enumerate(r2_clauses, start=1):
        clause["clause_id"] = "R2-%d" % index
        clause["text"] = (
            "Deprioritize `%s`: it holds a delayed-negative full-batch record "
            "on %d cohorts (%s), worst %+.6f."
            % (
                clause["program"], len(clause["cohorts"]),
                ", ".join(clause["cohorts"]), clause["worst_delayed"],
            )
        )

    # ---- R3 --------------------------------------------------------------
    stability = _mask_stability(cohort)
    r3_clauses: list[dict[str, Any]] = []
    if stability["loco_mask_stable_share"] <= R3_MASK_STABLE_MAX_SHARE:
        r3_clauses.append({
            "rule": "R3",
            "clause_id": "R3-1",
            "text": (
                "Do not reuse a historical exclusion mask. Across the "
                "leave-one-cohort-out cells only %d of %d kept their mask "
                "across windows (share %.4f, at or below the %.4f threshold), "
                "so a mask has to be re-searched on the window in front of you."
                % (
                    len(stability["loco_mask_stable_cells"]),
                    stability["loco_cells"],
                    stability["loco_mask_stable_share"],
                    R3_MASK_STABLE_MAX_SHARE,
                )
            ),
            "evidence": stability,
        })

    clauses = r1_clauses + r2_clauses + r3_clauses
    status = "ABSTAIN_TO_DEFAULT" if not clauses else "COMPILED"
    return {
        "target_id": str(target["target_id"]),
        "cohort": cohort,
        "consumer_variant": variant,
        "status": status,
        "clause_count": len(clauses),
        "clauses": clauses,
        "rules_that_produced_nothing": sorted(
            rule for rule, produced in (
                ("R1", bool(r1_clauses)), ("R2", bool(r2_clauses)),
                ("R3", bool(r3_clauses)),
            ) if not produced
        ),
        "loco": {
            "dropped_cohort": cohort,
            "rows_dropped": len(dropped),
            "rows_kept": len(kept),
        },
        "mask_stability": stability,
        "compiler": {
            "protocol_version": CARDS_PROTOCOL_VERSION,
            "zero_llm": True,
            "thresholds": {
                "R1_min_cohorts": R1_MIN_COHORTS,
                "R2_min_cohorts": R2_MIN_COHORTS,
                "R3_mask_stable_max_share": R3_MASK_STABLE_MAX_SHARE,
            },
            "delayed_only": True,
            "no_hand_written_clause": True,
        },
    }


def render_skill_card(card: Mapping[str, Any]) -> str:
    """The exact text A5 is shown.  Nothing else from the corpus reaches it."""
    if card["status"] == "ABSTAIN_TO_DEFAULT":
        return (
            "Source-derived Skill for this batch: ABSTAIN_TO_DEFAULT. The "
            "compiler found no clause that its rules could justify from "
            "other-cohort records, so it wrote none. Proceed on the target's "
            "own public observation alone."
        )
    lines = [
        "Source-derived Skill for this batch, compiled from other-cohort "
        "records only (every record measured on this cohort was withheld). "
        "Clauses are guidance, not instructions; the measurements you take on "
        "this window decide.",
        "",
    ]
    for clause in card["clauses"]:
        lines.append("- [%s] %s" % (clause["clause_id"], clause["text"]))
    return "\n".join(lines)


# ------------------------------------------------------------- the instrument
class BridgeSearch(wvc.BudgetedSearch):
    """The inherited instrument with every Consumer retrain counted.

    One evaluation over three origins is three retrains, and the mask round's
    internal per-series work is retrains too.  The earlier runs counted charged
    evaluations only, which understated the real feedback cost.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.retrains = 0
        self.retrain_log: list[dict[str, Any]] = []
        super().__init__(**kwargs)
        baseline = len(self.support) + len(self.delayed)
        self.retrains += baseline
        self.retrain_log.append({
            "what": "identity baselines", "retrains": baseline,
        })

    def _scoped(self, program: str, scope: set[str] | None, origins):
        self.retrains += len(origins)
        self.retrain_log.append({
            "what": "scoped(%s, %s)" % (
                program, "full batch" if scope is None else sorted(scope)
            ),
            "retrains": len(origins),
        })
        return super()._scoped(program, scope, origins)

    def _masked(self, program: str, excluded: set[str], origins):
        self.retrains += len(origins)
        self.retrain_log.append({
            "what": "masked(%s, minus %s)" % (
                program, sorted(excluded) or "nothing"
            ),
            "retrains": len(origins),
        })
        return super()._masked(program, excluded, origins)

    def accounting(self) -> dict[str, Any]:
        row = super().accounting()
        row["budget"] = EVALUATION_BUDGET
        row["budget_source"] = PROTOCOL_VERSION
        row["consumer_retrains_total"] = self.retrains
        row["consumer_retrain_log"] = [
            dict(entry) for entry in self.retrain_log
        ]
        row["retrain_note"] = (
            "every call that fits the Consumer is counted, including the "
            "identity baselines, the mask round's per-series work and the "
            "delayed gate"
        )
        return row


def _identity_incumbent_bar(
    search: Any, measured_plans: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The corrected bar: identity, plus Support-positive full-batch plans."""
    eligible: dict[str, float] = {}
    ineligible: dict[str, Any] = {}
    for row in measured_plans:
        if not row["full_batch"] or row["program"] == IDENTITY:
            continue
        delayed = float(
            search.delayed_gate(row["program"], [])["aggregate_gain"]
        )
        if float(row["support_aggregate_gain"]) > 0.0:
            eligible[row["program"]] = delayed
        else:
            ineligible[row["program"]] = {
                "support_aggregate_gain": row["support_aggregate_gain"],
                "delayed_aggregate_gain": delayed,
                "excluded_because": "its Support is not positive, so it is not "
                                    "a plan anyone could adopt",
            }
    bar = max(list(eligible.values()) + [0.0])
    return {
        "bar": bar,
        "eligible_full_batch_delayed": eligible,
        "ineligible_full_batch": ineligible,
        "rule": (
            "bar = max(0.0, delayed over evaluated full-batch plans with "
            "Support > 0)"
        ),
        "v2_semantics_checked": V2_GATE_SEMANTICS["what_v2_actually_does"],
    }


# --------------------------------------------------------------- stage wiring
SHORTLIST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "skill-bridge-shortlist/1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "shortlist", "request_mask_search", "skill_clause_use", "reason",
    ],
    "properties": {
        "shortlist": {
            "type": "array", "items": {"enum": list(TREATMENTS)},
            "minItems": 1, "maxItems": EVALUATION_BUDGET, "uniqueItems": True,
        },
        "request_mask_search": {"type": "boolean"},
        "skill_clause_use": {
            "type": "array", "items": {"type": "string"}, "maxItems": 8,
        },
        "reason": {"type": "string"},
    },
}
ADOPTION_SCHEMA = wvc.ADOPTION_SCHEMA

SHORTLIST_NOTE = (
    "Choose which programs are worth spending the evaluation budget on. "
    "`shortlist` names at most %d programs from the menu, in the order you "
    "would try them; each one costs one full-batch Support evaluation and the "
    "menu holds %d, so a full scan does not fit. `request_mask_search` asks for "
    "one greedy exclusion round, free of budget, run on whichever shortlisted "
    "program scores highest on Support. `skill_clause_use` lists the clause ids "
    "from `source_skill` that you actually relied on, and is empty when there "
    "are none. `reason` is one or two sentences in public terms. You will see "
    "the Support numbers next and then name the plan; the delayed window is "
    "never shown to you." % (EVALUATION_BUDGET, len(TREATMENTS))
)

ADOPTION_NOTE = (
    "The measurements are in. Name the plan to adopt: `program` and "
    "`excluded_series` must be exactly one entry of `measured_plans`, which is "
    "everything the instrument measured, `identity` included. `reason` is one "
    "or two sentences in public terms. A plan whose delayed reading turns out "
    "to be below the identity incumbent is not adopted and the target falls "
    "back to identity; you are not shown that reading before choosing."
)


def _make_shortlist_validator(clause_ids: Sequence[str]):
    known = {str(item) for item in clause_ids}

    def validate(payload: Mapping[str, Any]) -> None:
        cited = [str(item) for item in payload.get("skill_clause_use", ())]
        unknown = sorted(set(cited) - known)
        if unknown:
            raise StagePostValidationError(
                "SKILL_CLAUSE_UNGROUNDED",
                "skill_clause_use names clause ids that are not on the card "
                "you were given: %s" % unknown,
                retryable=True,
            )
        shortlist = [str(item) for item in payload["shortlist"]]
        if len(shortlist) != len(set(shortlist)):
            raise StagePostValidationError(
                "SHORTLIST_NOT_DISTINCT",
                "shortlist repeats a program; each entry costs one evaluation",
                retryable=True,
            )

    return validate


def _measured_plans(
    *, shortlist, support_results, mask_result,
) -> tuple[list[dict[str, Any]], str]:
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
            "full_batch": True,
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
            "full_batch": False,
        })
        note = "the mask round ran on `%s` and reverted %s" % (
            mask_result["program"], ", ".join(mask_result["final_excluded"]),
        )
    plans.append({
        "plan_id": "P0",
        "program": IDENTITY,
        "excluded_series": [],
        "support_aggregate_gain": 0.0,
        "harmed_evaluation_series_count": 0,
        "measured_by": "the identity baseline every gain is measured against",
        "full_batch": True,
    })
    return plans, note


def _base_input(
    *, target, window, search, observation, arm, card, card_text,
) -> dict[str, Any]:
    """One template for both arms.  A5 gains `source_skill`, A3 does not."""
    if arm == "A5":
        source_skill = {
            "available": True,
            "status": card["status"],
            "clause_ids": [
                str(clause["clause_id"]) for clause in card["clauses"]
            ],
            "card": card_text,
            "how_to_read": (
                "a Source-derived Skill compiled from other-cohort records "
                "only; every record measured on this cohort was withheld. "
                "Clauses are guidance, and the measurements you take on this "
                "window decide."
            ),
        }
    else:
        source_skill = {
            "available": False,
            "status": "NO_SOURCE_CONTENT",
            "clause_ids": [],
            "card": "",
            "how_to_read": (
                "no Source-derived content is available in this arm; the "
                "target's own public observation is the only evidence"
            ),
        }
    return {
        "schema_version": "skill-bridge-input/1",
        "arm": arm,
        "target": {
            "target_id": str(target["target_id"]),
            "cohort": str(target["cohort"]),
            "consumer_variant": str(target["consumer_variant"]),
            "cell_key": "batch:%s|consumer:%s"
            % (target["cohort"], target["consumer_variant"]),
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
            "mask_round": "free; runs on the highest-Support shortlisted "
            "program",
            "delayed_window": "read once after the plan is named and never "
            "shown to you",
            "identity_incumbent": (
                "a plan whose delayed reading is below the identity incumbent "
                "is not adopted; the target falls back to identity"
            ),
        },
        "public_observation": {
            "rule": (
                "public features of each training series on its own public "
                "prefix values[uid][:observation_cutoff]"
            ),
            "rows": [dict(row) for row in observation],
        },
        "source_skill": source_skill,
    }


# ------------------------------------------------------- windows and refs
import contextlib  # noqa: E402


@contextlib.contextmanager
def _traffic_origins(override: Sequence[int] | None):
    """Rebind the recipe module's traffic origins for one call, then restore.

    ``_TRAFFIC_SEALED_FROM_INDEX`` is deliberately not touched, so the module's
    own boundary guard stays live over the override.
    """
    if override is None:
        yield
        return
    saved = bch._TRAFFIC_DEVELOPMENT_ORIGINS
    bch._TRAFFIC_DEVELOPMENT_ORIGINS = tuple(int(item) for item in override)
    try:
        yield
    finally:
        bch._TRAFFIC_DEVELOPMENT_ORIGINS = saved


def _verify_sealed_boundary(origins: Sequence[int]) -> dict[str, Any]:
    """Check the traffic window against the frozen split config, not a memory.

    Two independent records are read: ``g3_sourcing.SEALED_FROM_INDEX``, which
    is the constant the screening code itself asserts against, and the
    ``sealed_from_index`` written into the screening artifacts.  If either is
    tighter than this window needs, the run stops rather than moving the window.
    """
    horizon = int(bch.v6.HORIZON)
    farthest = max(int(origin) for origin in origins) + horizon
    code_boundary = int(g3_sourcing.SEALED_FROM_INDEX)
    artifact_boundaries: dict[str, int] = {}
    for name in ("g3_candidate_screening_v2", "g3_candidate_screening_v3"):
        path = E2 / ("%s.json" % name)
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        value = (data.get("criteria") or {}).get("sealed_from_index")
        if value is not None:
            artifact_boundaries[name] = int(value)
    boundaries = [code_boundary] + list(artifact_boundaries.values())
    tightest = min(boundaries)
    result = {
        "origins": [int(origin) for origin in origins],
        "horizon": horizon,
        "farthest_index_read": farthest,
        "code_boundary": code_boundary,
        "code_boundary_source": (
            "evaluation/functional/task_episode_harness/agentic/g3_sourcing.py"
            "::SEALED_FROM_INDEX"
        ),
        "artifact_boundaries": artifact_boundaries,
        "tightest_boundary": tightest,
        "inside": bool(farthest <= tightest),
        "development_origins_on_record": list(g3_sourcing.DEVELOPMENT_ORIGINS),
    }
    if not result["inside"]:
        raise SystemExit(
            "STOPPED: the traffic window %s reads to index %d, at or past the "
            "tightest sealed boundary on record (%d). The window was not moved."
            % (result["origins"], farthest, tightest)
        )
    return result


def _target_window(target: Mapping[str, Any]) -> dict[str, Any]:
    """Origins plus the target's own full-recipe reference, computed here."""
    cohort = str(target["cohort"])
    variant = str(target["consumer_variant"])
    override = target.get("traffic_origins")
    sealed = None
    if override is not None:
        sealed = _verify_sealed_boundary(override)
        support = [int(override[0]), int(override[1])]
        delayed = [int(override[2])]
    else:
        spec = _frozen_task_roster()[int(target["task_index"])]
        support = [int(origin) for origin in spec["support_origins"]]
        delayed = [int(origin) for origin in spec["delayed_origins"]]
    print(
        "SKB reference for %s (%s x %s) on %s / %s -- offline, 0 LLM"
        % (target["target_id"], cohort, variant, support, delayed),
        flush=True,
    )
    with _traffic_origins(override):
        recipe = bch.make_batch_recipe(
            cohort,
            task_index=int(target["task_index"]),
            consumer_variant=variant,
            adoption_rule_version="v2",
        )
    if [int(o) for o in recipe["support_origins"]] != support:
        raise RuntimeError(
            "the reference recipe ran on %s, expected %s"
            % (recipe["support_origins"], support)
        )
    plan = recipe["adopted_plan"]
    return {
        "window_id": str(target["window_id"]),
        "support_origins": support,
        "delayed_origins": delayed,
        "farthest_index_read": max(support + delayed) + int(bch.v6.HORIZON),
        "origin_source": str(target["origin_source"]),
        "origin_provenance": str(target["origin_provenance"]),
        "sealed_boundary_check": sealed,
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
        "reference_adoption_path": str(recipe["adoption_path"]),
        "reference_menu_scan_support": {
            str(program): float(row["aggregate_gain"])
            for program, row in recipe["menu_scan"].items()
        },
        "reference_note": (
            "the frozen v2 recipe run by this runner on this window, 0 LLM "
            "calls, never shown to either arm"
        ),
        "reference_wall_seconds": float(recipe["wall_seconds"]),
    }


# ------------------------------------------------------------------- episode
def _run_arm(
    *, target, arm, window, card, card_text, snapshot, llm_budget,
) -> dict[str, Any]:
    started = time.perf_counter()
    episode_id = "%s_%s" % (target["target_id"], arm)
    search = BridgeSearch(
        cohort=str(target["cohort"]),
        consumer_variant=str(target["consumer_variant"]),
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )
    observation = wvc._observation_table(search)
    base = _base_input(
        target=target, window=window, search=search, observation=observation,
        arm=arm, card=card, card_text=card_text,
    )
    backend = _default_backend_factory(int(llm_budget))
    gateway = wvc.NoToolGateway({"episode_id": episode_id, "arm": arm})
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)
    harness_view = resolve_harness_view(snapshot, {}, role="fast")
    clause_ids = (
        [str(clause["clause_id"]) for clause in card["clauses"]]
        if arm == "A5" else []
    )
    shortlist_payload, shortlist_info = wvc._stage(
        core,
        stage="skill_bridge_shortlist",
        case_id="SKB_%s" % episode_id,
        public_input={**base, "stage_note": SHORTLIST_NOTE},
        harness_view=harness_view,
        schema_name="skill_bridge_shortlist_v1",
        schema=SHORTLIST_SCHEMA,
        validator=_make_shortlist_validator(clause_ids),
    )
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "target_id": str(target["target_id"]),
        "arm": arm,
        "cohort": str(target["cohort"]),
        "consumer_variant": str(target["consumer_variant"]),
        "window_id": str(window["window_id"]),
        "support_origins": list(search.support),
        "delayed_origins": list(search.delayed),
        "skill_card_status": str(card["status"]),
        "skill_clause_ids_available": clause_ids,
        "base_input_field_shas": {
            str(key): canonical_sha256(wvc._plain(value))
            for key, value in base.items()
        },
        "prompt_body": wvc._plain(base),
        "stages": [shortlist_info],
        "shortlist_payload": wvc._plain(shortlist_payload),
        "shortlist": [],
        "evaluations_used": 0,
        "adopted_plan": None,
        "final_plan": None,
        "support": None,
        "delayed": None,
    }
    if shortlist_payload is None:
        record["llm_calls"] = int(backend.calls)
        record["instrument"] = search.accounting()
        record["wall_seconds"] = time.perf_counter() - started
        return record

    shortlist = [str(item) for item in shortlist_payload["shortlist"]]
    wants_mask = bool(shortlist_payload["request_mask_search"])
    cited = [str(item) for item in shortlist_payload.get("skill_clause_use", ())]
    support_results = {
        program: search.full_batch_support(program) for program in shortlist
    }
    mask_result = None
    if wants_mask:
        best = max(
            shortlist,
            key=lambda program: (
                support_results[program]["aggregate_gain"],
                -shortlist.index(program),
            ),
        )
        mask_result = search.mask_search(best)
    plans, mask_note = _measured_plans(
        shortlist=shortlist, support_results=support_results,
        mask_result=mask_result,
    )
    record.update({
        "shortlist": shortlist,
        "request_mask_search": wants_mask,
        "skill_clause_use": cited,
        "shortlist_reason": str(shortlist_payload.get("reason", "")),
        "support_results": support_results,
        "mask_search": wvc._plain(mask_result),
        "measured_plans": plans,
        "measured_plans_note": mask_note,
        "evaluations_used": int(search.support_evaluations_charged),
    })
    print(
        "SKB %s shortlist=%s mask=%s cited=%s"
        % (episode_id, shortlist, wants_mask, cited),
        flush=True,
    )
    return _finish_arm(
        record=record, core=core, base=base, harness_view=harness_view,
        search=search, plans=plans, mask_note=mask_note, window=window,
        backend=backend, episode_id=episode_id, started=started,
    )


def _finish_arm(
    *, record, core, base, harness_view, search, plans, mask_note, window,
    backend, episode_id, started,
) -> dict[str, Any]:
    adoption_payload, adoption_info = wvc._stage(
        core,
        stage="skill_bridge_adoption",
        case_id="SKB_%s" % episode_id,
        public_input={
            **base,
            "stage_note": ADOPTION_NOTE,
            "your_shortlist": list(record["shortlist"]),
            "measured_plans": [dict(row) for row in plans],
            "measured_plans_note": mask_note,
            "evaluations_spent": int(record["evaluations_used"]),
        },
        harness_view=harness_view,
        schema_name="budgeted_adoption_v1",
        schema=ADOPTION_SCHEMA,
        validator=wvc._make_adoption_validator(
            shortlist=record["shortlist"],
            mask_result=record.get("mask_search"),
        ),
    )
    record["stages"].append(adoption_info)
    record["adoption_payload"] = wvc._plain(adoption_payload)
    record["llm_calls"] = int(backend.calls)
    if adoption_payload is None:
        record["instrument"] = search.accounting()
        record["wall_seconds"] = time.perf_counter() - started
        return record
    agent_plan = {
        "program": str(adoption_payload["program"]),
        "excluded_series": sorted(
            str(uid) for uid in adoption_payload.get("excluded_series", ())
        ),
    }
    gate = _identity_incumbent_bar(search, plans)
    adopted_delayed = search.delayed_gate(
        agent_plan["program"], agent_plan["excluded_series"]
    )
    passed = bool(
        float(adopted_delayed["aggregate_gain"]) >= float(gate["bar"])
    )
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
    support_gain = float(support["aggregate_gain"])
    delayed_gain = float(delayed["aggregate_gain"])
    reference = float(window["reference_delayed_aggregate_gain"])
    if final_plan["program"] == IDENTITY:
        relation = RELATION_ABSTAIN
    elif support_gain > 0.0 and delayed_gain > 0.0:
        relation = RELATION_POSITIVE
    elif (support_gain > 0.0) != (delayed_gain > 0.0):
        relation = RELATION_CONFLICT
    else:
        relation = RELATION_NEGATIVE
    record.update({
        "adopted_plan": agent_plan,
        "adoption_reason": str(adoption_payload.get("reason", "")),
        "identity_incumbent_gate": {
            **gate,
            "agent_plan_delayed": float(adopted_delayed["aggregate_gain"]),
            "passed": passed,
            "fell_back_to_identity": not passed,
        },
        "final_plan": final_plan,
        "support": support,
        "delayed": delayed,
        "relation": relation,
        "reference_delayed_aggregate_gain": reference,
        "reference_plan": dict(window["reference_plan"]),
        "capture_ratio": (delayed_gain / reference if reference else None),
        "matches_reference_plan": bool(
            final_plan["program"] == str(window["reference_plan"]["program"])
            and final_plan["excluded_series"]
            == sorted(str(uid) for uid in window["reference_plan"][
                "excluded_series"])
        ),
        "instrument": search.accounting(),
        "consumer_retrains_total": int(search.retrains),
        "llm_calls": int(backend.calls),
        "wall_seconds": time.perf_counter() - started,
    })
    print(
        "SKB %s final %s minus %s | support %+.6f delayed %+.6f | gate bar "
        "%+.6f passed=%s | evals %d retrains %d llm %d"
        % (
            episode_id, final_plan["program"],
            ", ".join(final_plan["excluded_series"]) or "nothing",
            support_gain, delayed_gain, float(gate["bar"]), passed,
            record["evaluations_used"], search.retrains, record["llm_calls"],
        ),
        flush=True,
    )
    return record


def _experience_entry(record: Mapping[str, Any]) -> Any:
    plan = record["final_plan"]
    audit = {
        "provenance": EXPERIENCE_PROVENANCE,
        "counts_as_unguided_exploration": False,
        "audit_note": (
            "engineering measurement from the Slow-compiled Skill bridge; not "
            "authorization evidence and not an unguided probe. The Skill card "
            "was compiled deterministically from committed rows and is not "
            "installed in any Harness snapshot"
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
                "arm": str(record["arm"]),
            },
            "program_geometry": {
                "excluded_count": len(plan["excluded_series"]),
                "evaluations_used": int(record["evaluations_used"]),
                "consumer_retrains": int(record["consumer_retrains_total"]),
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
            "arm": str(record["arm"]),
            "skill_clause_use": list(record.get("skill_clause_use") or []),
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
            "window_role": (
                "read once after the plan was named; neither arm saw it"
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


# ------------------------------------------------------------------ verdicts
def _target_verdict(a3, a5, card) -> dict[str, Any]:
    abstained = str(card["status"]) == "ABSTAIN_TO_DEFAULT"
    if (
        a3 is None or a5 is None
        or a3.get("delayed") is None or a5.get("delayed") is None
    ):
        return {
            "label": "UNREADABLE",
            "delivery": "NOT_APPLICABLE_ABSTAIN" if abstained else "UNREADABLE",
            "reason": "one of the two arms produced no adopted plan",
            "paired_delayed_delta": None,
        }
    a3_delayed = float(a3["delayed"]["aggregate_gain"])
    a5_delayed = float(a5["delayed"]["aggregate_gain"])
    delta = a5_delayed - a3_delayed
    if delta > QUALITY_DELTA_THRESHOLD:
        label = "A5_WINS"
    elif delta < -QUALITY_DELTA_THRESHOLD:
        label = "A5_LOSES"
    else:
        label = "TIE"
    cited = [str(item) for item in (a5.get("skill_clause_use") or [])]
    known = {str(clause["clause_id"]) for clause in card["clauses"]}
    if abstained:
        delivery = "NOT_APPLICABLE_ABSTAIN"
    elif set(cited) & known:
        delivery = "DELIVERED"
    else:
        delivery = "SKILL_NOT_DELIVERED"
    return {
        "label": label,
        "delivery": delivery,
        "clauses_cited_by_a5": cited,
        "clauses_available": sorted(known),
        "paired_delayed_delta": delta,
        "a3_delayed_aggregate_gain": a3_delayed,
        "a5_delayed_aggregate_gain": a5_delayed,
        "a3_support_aggregate_gain": float(a3["support"]["aggregate_gain"]),
        "a5_support_aggregate_gain": float(a5["support"]["aggregate_gain"]),
        "a3_capture_ratio": a3.get("capture_ratio"),
        "a5_capture_ratio": a5.get("capture_ratio"),
        "a3_consumer_retrains": int(a3["consumer_retrains_total"]),
        "a5_consumer_retrains": int(a5["consumer_retrains_total"]),
        "a3_evaluations_used": int(a3["evaluations_used"]),
        "a5_evaluations_used": int(a5["evaluations_used"]),
        "reason": (
            "paired delayed delta %+.6f (A5 %+.6f - A3 %+.6f) against a "
            "threshold of %.3f; delivery %s"
            % (delta, a5_delayed, a3_delayed, QUALITY_DELTA_THRESHOLD, delivery)
        ),
    }


def _cost_report(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """First measurement of the real feedback cost.  Decides nothing."""
    per_arm_target = {
        str(row["episode_id"]): {
            "consumer_retrains": int(row.get("consumer_retrains_total") or 0),
            "charged_evaluations": int(row["evaluations_used"]),
            "llm_calls": int(row["llm_calls"]),
            "delayed_aggregate_gain": (
                None if row.get("delayed") is None
                else float(row["delayed"]["aggregate_gain"])
            ),
        }
        for row in records
    }
    first_positive: dict[str, Any] = {}
    for arm in ("A3", "A5"):
        cumulative = 0
        reached = None
        for row in records:
            if str(row["arm"]) != arm:
                continue
            cumulative += int(row.get("consumer_retrains_total") or 0)
            delayed = row.get("delayed")
            if delayed is not None and float(delayed["aggregate_gain"]) > 0.0:
                reached = {
                    "at_episode": str(row["episode_id"]),
                    "cumulative_consumer_retrains": cumulative,
                    "delayed_aggregate_gain": float(delayed["aggregate_gain"]),
                }
                break
        first_positive[arm] = reached or {
            "at_episode": None,
            "cumulative_consumer_retrains": cumulative,
            "note": "no delayed-positive adoption in this arm",
        }
    return {
        "per_arm_target": per_arm_target,
        "arm_totals": {
            arm: sum(
                int(row.get("consumer_retrains_total") or 0)
                for row in records if str(row["arm"]) == arm
            )
            for arm in ("A3", "A5")
        },
        "cumulative_retrains_to_first_delayed_positive_adoption": first_positive,
        "has_no_criterion": True,
    }


def _prompt_parity(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_target: dict[str, dict[str, Any]] = {}
    for row in records:
        shas = row.get("base_input_field_shas")
        if shas:
            by_target.setdefault(str(row["target_id"]), {})[
                str(row["arm"])
            ] = shas
    rows: dict[str, Any] = {}
    for target_id, arms in by_target.items():
        a3, a5 = arms.get("A3"), arms.get("A5")
        if not a3 or not a5:
            rows[target_id] = {"comparable": False}
            continue
        differing = sorted(
            key for key in set(a3) | set(a5) if a3.get(key) != a5.get(key)
        )
        rows[target_id] = {
            "comparable": True,
            "fields_that_differ": differing,
            "only_source_skill_and_arm_differ": (
                differing == ["arm", "source_skill"]
            ),
            "fields_compared": len(set(a3) | set(a5)),
        }
    return {
        "scope": (
            "the stage-one prompt body; `arm` is the label field naming which "
            "arm this is, `source_skill` is the Skill card itself"
        ),
        "all_targets_pass": all(
            row.get("only_source_skill_and_arm_differ")
            for row in rows.values() if row.get("comparable")
        ),
        "per_target": rows,
    }


# --------------------------------------------------------------- orchestration
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    cards = {
        target_id: compile_skill_card(target)
        for target_id, target in TARGETS.items()
    }
    card_texts = {
        target_id: render_skill_card(card) for target_id, card in cards.items()
    }
    cards_payload = {
        "protocol_version": CARDS_PROTOCOL_VERSION,
        "role": (
            "deterministic Slow compilation of committed recipe rows into one "
            "Source-derived Skill card per target, leave-one-cohort-out"
        ),
        "llm_api_call_count": 0,
        "deterministic": True,
        "not_authorization_evidence": (
            "these cards are rendered text objects this runner compiled; none "
            "is installed in any Harness snapshot and none grants any right"
        ),
        "rules": PRE_REGISTERED["compiler"],
        "delayed_record_count": len(_delayed_records()),
        "delayed_records": _delayed_records(),
        "cards": cards,
        "rendered": card_texts,
    }
    CARDS_JSON.parent.mkdir(parents=True, exist_ok=True)
    CARDS_JSON.write_text(
        json.dumps(cards_payload, indent=2, ensure_ascii=False, default=str)
        + "\n",
        encoding="utf-8", newline="\n",
    )
    print("wrote", CARDS_JSON, flush=True)
    for target_id, card in cards.items():
        print(
            "SKB card %s: %s, %d clause(s), silent rules %s"
            % (
                target_id, card["status"], card["clause_count"],
                card["rules_that_produced_nothing"],
            ),
            flush=True,
        )
    if all(
        str(card["status"]) == "ABSTAIN_TO_DEFAULT" for card in cards.values()
    ):
        print(
            "SKB every target compiled to ABSTAIN_TO_DEFAULT; the arms are "
            "still run so the claim is measured rather than assumed",
            flush=True,
        )

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

    for target_id, arm in ARM_ORDER:
        remaining = LLM_CALL_BUDGET_TOTAL - llm_used
        if remaining < 2:
            stopped_reason = (
                "global LLM budget exhausted before %s %s (%d of %d used)"
                % (target_id, arm, llm_used, LLM_CALL_BUDGET_TOTAL)
            )
            break
        print(
            "SKB arm-target %s %s (llm %d/%d)"
            % (target_id, arm, llm_used, LLM_CALL_BUDGET_TOTAL),
            flush=True,
        )
        record = _run_arm(
            target=TARGETS[target_id],
            arm=arm,
            window=windows[target_id],
            card=cards[target_id],
            card_text=card_texts[target_id],
            snapshot=snapshot,
            llm_budget=min(LLM_CALL_BUDGET_PER_EPISODE, remaining),
        )
        llm_used += int(record["llm_calls"])
        records.append(record)
        if record.get("final_plan") is not None:
            written = _experience_entry(record)
            episodes.append(written)
            record["experience_written"] = written.to_dict()
        else:
            record["experience_written"] = None
        if len(records) == 1 and record.get("final_plan") is None:
            stopped_reason = (
                "circuit breaker: the first arm-target produced no payload"
            )
            break

    by_key = {(row["target_id"], row["arm"]): row for row in records}
    per_target = {}
    for target_id in TARGETS:
        if not any(row["target_id"] == target_id for row in records):
            continue
        per_target[target_id] = {
            "target": TARGETS[target_id],
            "window": windows[target_id],
            "skill_card_status": cards[target_id]["status"],
            "skill_clause_count": cards[target_id]["clause_count"],
            **_target_verdict(
                by_key.get((target_id, "A3")), by_key.get((target_id, "A5")),
                cards[target_id],
            ),
        }
    labels = [row["label"] for row in per_target.values()]
    deliveries = [row["delivery"] for row in per_target.values()]
    all_abstain = all(
        str(card["status"]) == "ABSTAIN_TO_DEFAULT" for card in cards.values()
    )
    losing = [
        target_id for target_id, row in per_target.items()
        if row["label"] == "A5_LOSES"
    ]
    if all_abstain:
        overall = "SKILL_COMPILATION_INSUFFICIENT_PROVENANCE"
        overall_reason = (
            "all three targets compiled to ABSTAIN_TO_DEFAULT; this "
            "implementation path is closed on the provenance available"
        )
    elif losing:
        overall = "SKILL_LOSES_SIGNAL"
        overall_reason = (
            "A5 lost on %s; the clauses it cited there were %s"
            % (
                ", ".join(losing),
                ", ".join(
                    "%s:%s" % (
                        target_id,
                        per_target[target_id]["clauses_cited_by_a5"] or "none",
                    )
                    for target_id in losing
                ),
            )
        )
    elif (
        all(
            row["delivery"] in ("DELIVERED", "NOT_APPLICABLE_ABSTAIN")
            for row in per_target.values()
        )
        and labels.count("A5_WINS") >= 1
    ):
        overall = "SKILL_BRIDGE_DELIVERS"
        overall_reason = (
            "delivery confirmed on every non-abstaining target, no A5_LOSES, "
            "and %d A5_WINS" % labels.count("A5_WINS")
        )
    else:
        overall = "SKILL_BRIDGE_NO_EFFECT"
        overall_reason = (
            "labels %s with deliveries %s: nothing lost, nothing won, or "
            "delivery failed somewhere" % (labels, deliveries)
        )
    return _write(
        payload_bits=dict(
            cards=cards, card_texts=card_texts, windows=windows,
            records=records, per_target=per_target, labels=labels,
            overall=overall, overall_reason=overall_reason,
            llm_used=llm_used, stopped_reason=stopped_reason,
            episodes=episodes, started=started,
        ),
        dry_run=dry_run,
    )


def _repo_relative(path: Path) -> str:
    """Repo-relative when it can be, the plain name when it cannot."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _write(*, payload_bits: Mapping[str, Any], dry_run: bool) -> int:
    bits = dict(payload_bits)
    records = bits["records"]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "does the recipe experience signal survive the legitimate channel: "
            "deterministic Slow compilation into a Source-derived Skill, Fast "
            "reading it, with every Consumer retrain counted"
        ),
        "not_authorization_evidence": (
            "no Skill is promoted, no TRY right is granted, no Episode is "
            "promoted, and no Fast or Slow path of the real Harness runs; the "
            "Skill card is a rendered text object, installed nowhere"
        ),
        "overall_verdict": bits["overall"],
        "overall_verdict_reason": bits["overall_reason"],
        "per_target": bits["per_target"],
        "label_counts": {
            label: bits["labels"].count(label)
            for label in ("A5_WINS", "A5_LOSES", "TIE", "UNREADABLE")
        },
        "pre_registered": PRE_REGISTERED,
        "v2_gate_semantics_check": V2_GATE_SEMANTICS,
        "prompt_parity_check": _prompt_parity(records),
        "cost_report": _cost_report(records),
        "skill_cards": bits["cards"],
        "skill_card_rendered_text": bits["card_texts"],
        "skill_cards_artifact": _repo_relative(CARDS_JSON),
        "model": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "target_windows": bits["windows"],
        "llm_call_count": bits["llm_used"],
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "stopped_early": bits["stopped_reason"],
        "experience_entries_written": [
            episode.to_dict() for episode in bits["episodes"]
        ],
        "experience_provenance": EXPERIENCE_PROVENANCE,
        "episodes": records,
        "wall_seconds": time.perf_counter() - bits["started"],
    }
    if dry_run:
        print(json.dumps(
            {
                "overall": payload["overall_verdict"],
                "per_target": {
                    key: {
                        "label": row["label"], "delivery": row["delivery"],
                        "delta": row.get("paired_delayed_delta"),
                    }
                    for key, row in payload["per_target"].items()
                },
            },
            indent=2, ensure_ascii=False, default=str,
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
    print("overall", payload["overall_verdict"], flush=True)
    print(
        "labels", json.dumps(payload["label_counts"], sort_keys=True),
        flush=True,
    )
    print("llm_calls", payload["llm_call_count"], flush=True)
    return 0


# --------------------------------------------------------------------- report
def _markdown_head(payload: Mapping[str, Any]) -> list[str]:
    parity = payload["prompt_parity_check"]
    lines = [
        "# recipe experience -> Source Skill -> Fast v1",
        "",
        "**Overall: `%s`** -- %s."
        % (payload["overall_verdict"], payload["overall_verdict_reason"]),
        "",
        "The warm/cold rotation showed a table of other-batch records helps, "
        "but it reached the Agent as a pasted table and the cost account left "
        "out the mask round's internal retrains. This run sends the same "
        "signal through a deterministic Slow compilation into a "
        "Source-derived Skill card, gives Fast only that card, and counts "
        "every Consumer retrain.",
        "",
        "**Engineering effect measurement, not authorization evidence.** %s."
        % payload["not_authorization_evidence"],
        "",
        "## 0. The v2 gate, read before it was copied",
        "",
        "- what v2 actually does: %s"
        % payload["v2_gate_semantics_check"]["what_v2_actually_does"],
        "- what this runner does: %s"
        % payload["v2_gate_semantics_check"]["what_this_runner_does"],
        "- the difference, and why: %s"
        % payload["v2_gate_semantics_check"]["difference_and_why"],
        "",
        "## 1. Compiled Skill cards",
        "",
        "| target | cell | status | clauses | rules that produced nothing |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for target_id, card in payload["skill_cards"].items():
        lines.append(
            "| %s | %s x %s | `%s` | %d | %s |"
            % (
                target_id, card["cohort"], card["consumer_variant"],
                card["status"], card["clause_count"],
                ", ".join(card["rules_that_produced_nothing"]) or "none",
            )
        )
    lines += ["", "Rendered text, exactly as A5 saw it:", ""]
    for target_id, text in payload["skill_card_rendered_text"].items():
        lines += ["**%s**" % target_id, ""]
        for line in text.splitlines():
            lines.append("> %s" % line if line else ">")
        lines.append("")
    lines += [
        "Full clause provenance -- which artifact and which key each clause "
        "came from -- is in `%s`." % payload["skill_cards_artifact"],
        "",
        "## 2. Targets and windows",
        "",
        "| target | cell | window | support | delayed | origin source | "
        "reference plan | reference delayed |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for target_id, window in payload["target_windows"].items():
        target = TARGETS[target_id]
        lines.append(
            "| %s | %s x %s | %s | %s | %s | %s | %s | %s |"
            % (
                target_id, target["cohort"], target["consumer_variant"],
                window["window_id"], window["support_origins"],
                window["delayed_origins"], window["origin_source"],
                wvc._plan_text(window["reference_plan"]),
                wvc._gain(window["reference_delayed_aggregate_gain"]),
            )
        )
    sealed = payload["target_windows"]["T3"]["sealed_boundary_check"]
    lines += [
        "",
        "T3's window is chosen, not quoted. Its sealed boundary was verified "
        "before the run against both the code constant (`%s` = %d) and the "
        "screening artifacts (%s): farthest index read %d, tightest boundary "
        "%d, inside: **%s**."
        % (
            sealed["code_boundary_source"], sealed["code_boundary"],
            json.dumps(sealed["artifact_boundaries"], sort_keys=True),
            sealed["farthest_index_read"], sealed["tightest_boundary"],
            sealed["inside"],
        ),
        "",
        "Prompt parity, per target: %s. All targets pass: **%s**."
        % (
            "; ".join(
                "%s -> %s" % (
                    target_id,
                    ", ".join("`%s`" % f for f in row.get(
                        "fields_that_differ", [])) or "none",
                )
                for target_id, row in parity["per_target"].items()
            ),
            parity["all_targets_pass"],
        ),
        "",
    ]
    return lines


def _markdown_body(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "## 3. Per-target result",
        "",
        "| target | card | delivery | clauses A5 cited | A3 delayed | "
        "A5 delayed | paired delta | label |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for target_id, row in payload["per_target"].items():
        lines.append(
            "| %s | `%s` (%d) | `%s` | %s | %s | %s | %s | `%s` |"
            % (
                target_id, row["skill_card_status"], row["skill_clause_count"],
                row["delivery"],
                ", ".join("`%s`" % c for c in row.get("clauses_cited_by_a5", []))
                or "none",
                wvc._gain(row.get("a3_delayed_aggregate_gain")),
                wvc._gain(row.get("a5_delayed_aggregate_gain")),
                wvc._gain(row.get("paired_delayed_delta")),
                row["label"],
            )
        )
    counts = payload["label_counts"]
    lines += [
        "",
        "Counts: A5_WINS %d, A5_LOSES %d, TIE %d, unreadable %d."
        % (
            counts["A5_WINS"], counts["A5_LOSES"], counts["TIE"],
            counts["UNREADABLE"],
        ),
        "",
        "## 4. The twelve-row arm table",
        "",
        "| target | arm | shortlist | mask | evals | retrains | plan named | "
        "gate | final plan | support | delayed | capture | LLM |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | "
        "---: | ---: | ---: |",
    ]
    for record in payload["episodes"]:
        gate = record.get("identity_incumbent_gate") or {}
        support = record.get("support") or {}
        delayed = record.get("delayed") or {}
        lines.append(
            "| %s | %s | %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %d |"
            % (
                record["target_id"], record["arm"],
                ", ".join("`%s`" % p for p in record.get("shortlist", []))
                or "n/a",
                record.get("request_mask_search"),
                record["evaluations_used"],
                int(record.get("consumer_retrains_total") or 0),
                wvc._plan_text(record.get("adopted_plan")),
                "n/a" if not gate else (
                    "pass (bar %s)" % wvc._gain(gate.get("bar"))
                    if gate.get("passed")
                    else "**fallback** (bar %s)" % wvc._gain(gate.get("bar"))
                ),
                wvc._plan_text(record.get("final_plan")),
                wvc._gain(support.get("aggregate_gain")),
                wvc._gain(delayed.get("aggregate_gain")),
                wvc._ratio(record.get("capture_ratio")),
                record["llm_calls"],
            )
        )
    cost = payload["cost_report"]
    lines += [
        "",
        "## 5. Cost, first measurement, no criterion attached",
        "",
        "| arm | total Consumer retrains | first delayed-positive adoption | "
        "cumulative retrains to get there |",
        "| --- | ---: | --- | ---: |",
    ]
    for arm in ("A3", "A5"):
        first = cost["cumulative_retrains_to_first_delayed_positive_adoption"][
            arm
        ]
        lines.append(
            "| %s | %d | %s | %d |"
            % (
                arm, cost["arm_totals"][arm],
                first.get("at_episode") or "never",
                first["cumulative_consumer_retrains"],
            )
        )
    lines += [
        "",
        "A retrain is one fit of the downstream Consumer. One evaluation over "
        "three origins is three retrains, and the mask round's per-series work "
        "is retrains too -- that is the part the earlier runs did not count.",
        "",
        "## 6. What each arm said",
        "",
    ]
    for record in payload["episodes"]:
        lines += [
            "**%s** -- shortlist %s (mask %s), cited %s"
            % (
                record["episode_id"],
                ", ".join("`%s`" % p for p in record.get("shortlist", []))
                or "n/a",
                record.get("request_mask_search"),
                ", ".join("`%s`" % c
                          for c in record.get("skill_clause_use") or [])
                or "none",
            ),
            "",
            "  shortlist reason: %s" % record.get("shortlist_reason", ""),
            "",
            "  adopted %s: %s"
            % (
                wvc._plan_text(record.get("adopted_plan")),
                record.get("adoption_reason", ""),
            ),
            "",
        ]
    return lines


def _markdown_tail(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "## 7. Experience rows written",
        "",
        "`provenance=\"%s\"`, `counts_as_unguided_exploration: false`, and fed "
        "into nothing: the three targets are independent and no row from this "
        "run reaches another target." % payload["experience_provenance"],
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
        "- It authorizes nothing. The Skill card is compiled text handed to a "
        "prompt; no snapshot, no Fast/Slow path and no execution right is "
        "touched.",
        "- Three targets, one draw each, one model. Every label is a "
        "comparison of two single runs.",
        "- The compiler's rules are three hand-frozen thresholds, not a "
        "learned policy, and a rule that produced nothing here says something "
        "about the provenance available, not about the mechanism.",
        "- The delayed column is out of selection for both arms -- neither "
        "saw it -- but the reference plans it is compared against were "
        "themselves selected on their own delayed windows.",
        "- The retrain count is the instrument's cost, not the Agent's: both "
        "arms are charged for whatever the instrument had to fit.",
        "",
        "## Provenance",
        "",
        "- model: `%s` at `%s`"
        % (payload["model"]["model"], payload["model"]["base_url"]),
        "- instrument, observation table and stage driver: imported from "
        "`run_e2_warm_vs_cold_recipe_search`, which is not modified",
        "- references: `run_batch_composition_headroom.make_batch_recipe` with "
        "`adoption_rule_version=\"v2\"`, run here per target, 0 LLM calls",
        "- Skill cards and every clause's provenance: `%s`"
        % payload["skill_cards_artifact"],
        "- full prompt bodies for all six arm-targets are in the JSON under "
        "`episodes[].prompt_body`",
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


# ------------------------------------------------- v2 ladder replay (0 LLM) --
# The run above answered the bridge question with one instrument defect: on a
# failed adoption gate it dropped straight to identity, which is only the
# *last* rung of the frozen v2 rule.  This section replays the adoption stage
# of the six recorded arm-targets against a faithful port of that rule.  It
# reads recipe_skill_bridge_v1.json, touches no data, fits no Consumer and
# calls no model.  v1 is left exactly as it was written.
REPLAY_PROTOCOL_VERSION = "recipe_skill_bridge_v2_replay"
REPLAY_OUT_JSON = E2 / "recipe_skill_bridge_v2_replay.json"
REPLAY_OUT_MD = E2 / "recipe_skill_bridge_v2_replay.md"

# Frozen before the replay ran.  Each rung is quoted against ADOPTION_RULE_V2
# in evaluation/functional/run_batch_composition_headroom.py and against its
# implementation in make_batch_recipe() in that same module.
REPLAY_LADDER_RULE = {
    "step_1_support_winner": (
        "the Support winner is the highest-Support plan among the full-batch "
        "plans this arm-target actually evaluated, ties broken by menu order; "
        "it is eligible only if its Support aggregate gain is strictly "
        "positive, otherwise this arm-target has no Support winner"
    ),
    "step_2_gate": (
        "the plan the Agent named is adopted only if its delayed aggregate "
        "gain is at least the bar, where bar = max(0, the Support winner's "
        "full-batch delayed aggregate gain), and bar = 0 when there is no "
        "Support winner"
    ),
    "step_3_fallback": (
        "if the named plan fails the gate, adopt the Support winner when one "
        "exists and its full-batch delayed aggregate gain is positive; "
        "otherwise adopt identity"
    ),
    "selection_is_support_only": (
        "which plan the ladder falls back to is decided by Support alone, "
        "because Support is the window a deployer can see.  Delayed is a "
        "confirmation reading: the ladder consults exactly two delayed "
        "numbers -- the Support winner's, to set the bar, and the named "
        "plan's, to confirm it.  Ranking fallback candidates by delayed, or "
        "comparing several candidates' delayed, is forbidden, and the "
        "delayed_reads tape on every arm-target records what was consulted"
    ),
}

REPLAY_V2_CORRESPONDENCE = [
    {
        "rung": "step_1_support_winner",
        "v2_source": (
            'ranked = sorted(full_support, key=lambda op: '
            '(-full_support[op]["aggregate_gain"], PROGRAM_MENU.index(op))); '
            'best_full_program = ranked[0]'
        ),
        "relation": "PORTED_WITH_ONE_DECLARED_ADDITION",
        "note": (
            "the ordering and the menu-order tie-break are the same; v2 never "
            "checks the sign of that program's Support, this replay requires "
            "Support > 0.  That addition is the correction carried forward "
            "from the negative-path run, and literal_v2_sensitivity on every "
            "arm-target reports what the unchecked reading would have adopted"
        ),
    },
    {
        "rung": "step_2_gate",
        "v2_source": (
            'ADOPTION_RULE_V2: "its delayed aggregate gain must be at least '
            'max(best full-batch delayed aggregate gain, 0) -- identity is an '
            'incumbent on the delayed window, not only the best full-batch '
            'plan"; code: delayed_bar = max(float(bar_delayed'
            '["aggregate_gain"]), 0.0)'
        ),
        "relation": "IDENTICAL",
        "note": (
            "bar = max(0, one program's full-batch delayed).  v1 read the max "
            "over every eligible full-batch plan's delayed instead, which is "
            "the forbidden comparison of several candidates' delayed"
        ),
    },
    {
        "rung": "step_3_fallback",
        "v2_source": (
            'ADOPTION_RULE_V2: "If none clears it, fall back to the best '
            'full-batch plan only if that plan\'s delayed aggregate gain is '
            'positive; otherwise fall back to identity"; code: elif '
            'bar_delayed["aggregate_gain"] > 0.0: adopted = BEST_FULL_BATCH '
            '... else: adopted = IDENTITY'
        ),
        "relation": "IDENTICAL",
        "note": (
            "this is the rung v1 skipped: it went from a failed gate straight "
            "to the else branch"
        ),
    },
    {
        "rung": "candidate set",
        "v2_source": (
            'candidates = [... for program in search["programs_searched"] if '
            'search["searches"][program]["final_excluded"]], sorted by '
            "descending Support; the first that clears the bar is adopted"
        ),
        "relation": "NARROWED_BY_DESIGN",
        "note": (
            "v2 walks its own list of masked plans.  The replay re-scores the "
            "one plan the Agent named, masked or full-batch, because the "
            "object under test is the Agent's decision, not the recipe's "
            "search.  When the named plan is itself the Support winner at "
            "full batch the gate is satisfied by equality"
        ),
    },
    {
        "rung": "full-batch pool",
        "v2_source": "full_support is built over all of TREATMENTS",
        "relation": "NARROWED_BY_BUDGET",
        "note": (
            "the arms paid for at most %d full-batch evaluations, so the pool "
            "is what each arm actually measured; nothing unmeasured is "
            "estimated" % EVALUATION_BUDGET
        ),
    },
]

REPLAY_PRE_REGISTERED = {
    "fixed_before_the_replay_ran": True,
    "what_is_replayed": (
        "only the adoption stage of the six arm-targets recorded in "
        "recipe_skill_bridge_v1.json, from numbers that run already measured"
    ),
    "why_a_replay_is_legitimate_here": (
        "the six arm-targets carry no cross-target experience feedback, and "
        "every LLM decision was taken before the gate and without seeing its "
        "result, so re-deciding the gate deterministically cannot have "
        "changed what the Agent would have said"
    ),
    "not_replayed": (
        "run_e2_negative_path_adaptation feeds each episode's outcome into "
        "the next, so a changed gate result would change later decisions; a "
        "replay there would be invalid and its labels stand as filed"
    ),
    "ladder": REPLAY_LADDER_RULE,
    "insufficient_data": (
        "an arm-target missing any number the ladder needs -- the named plan, "
        "its delayed, its Support, or a full-batch delayed for a program it "
        "evaluated -- is labelled REPLAY_INSUFFICIENT_DATA and stops the "
        "report; no number is ever estimated to fill a gap"
    ),
    "per_target_label": PRE_REGISTERED["per_target_label"],
    "overall_verdict_in_this_order": PRE_REGISTERED[
        "overall_verdict_in_this_order"
    ],
    "delivery_is_inherited": (
        "delivery was decided by what A5 cited at stage one, which the replay "
        "does not touch; the v1 delivery reading carries over verbatim"
    ),
    "cost": (
        "total Consumer retrains are a property of what was run and do not "
        "change; only the attribution of the first delayed-positive adoption "
        "is recomputed against the corrected adoptions"
    ),
    "standing": (
        "the corrected reading is filed beside the v1 verdict as the "
        "instrument-corrected reading of the bridge experiment; v1 is not "
        "overwritten and not amended"
    ),
    "zero_llm": True,
    "zero_new_consumer_retrains": True,
}


def _menu_index(program: str) -> int:
    """v2's tie-break: menu order, unknown programs last."""
    try:
        return bch.PROGRAM_MENU.index(program)
    except ValueError:
        return len(bch.PROGRAM_MENU)


def _plan_label(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "--"
    excluded = [str(uid) for uid in (plan.get("excluded_series") or [])]
    if not excluded:
        return "`%s` full batch" % plan["program"]
    return "`%s` minus %s" % (plan["program"], ", ".join(sorted(excluded)))


class _DelayedTape:
    """Records every delayed number the ladder consults, in order.

    Selection is a Support decision.  Delayed only confirms.  The tape is what
    makes that auditable rather than asserted.
    """

    def __init__(self, full_batch_delayed: Mapping[str, float]) -> None:
        self._delayed = dict(full_batch_delayed)
        self.reads: list[dict[str, Any]] = []

    def full_batch(self, program: str, role: str) -> float:
        value = float(self._delayed[program])
        self.reads.append({
            "program": program, "role": role,
            "delayed_aggregate_gain": value,
        })
        return value

    def named(self, value: float) -> float:
        self.reads.append({
            "program": "<the plan the Agent named>", "role": "confirmation",
            "delayed_aggregate_gain": float(value),
        })
        return float(value)


def _replay_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    """Everything the ladder needs, or the reason it cannot be assembled."""
    missing: list[str] = []
    gate = record.get("identity_incumbent_gate") or {}
    named = record.get("adopted_plan")
    if not named:
        missing.append("the arm named no plan")
    named_delayed = gate.get("agent_plan_delayed")
    if named_delayed is None:
        missing.append("no delayed reading for the plan the Agent named")

    # The full-batch pool is what this arm actually paid to evaluate.  Support
    # comes from the measured-plans table, delayed from the recorded gate.
    eligible = dict(gate.get("eligible_full_batch_delayed") or {})
    ineligible = dict(gate.get("ineligible_full_batch") or {})
    pool: dict[str, dict[str, float]] = {}
    for row in record.get("measured_plans") or []:
        if not row.get("full_batch") or str(row["program"]) == IDENTITY:
            continue
        program = str(row["program"])
        if program in eligible:
            delayed = float(eligible[program])
        elif program in ineligible:
            delayed = float(ineligible[program]["delayed_aggregate_gain"])
        else:
            missing.append("no full-batch delayed reading for %r" % program)
            continue
        pool[program] = {
            "support_aggregate_gain": float(row["support_aggregate_gain"]),
            "delayed_aggregate_gain": delayed,
        }
    if not pool:
        missing.append("no full-batch plan was evaluated")

    named_support = None
    if named:
        program = str(named["program"])
        excluded = [str(uid) for uid in (named.get("excluded_series") or [])]
        if program == IDENTITY:
            named_support = dict(record["support"])
        elif not excluded:
            row = (record.get("support_results") or {}).get(program)
            if row is None:
                missing.append("no full-batch Support row for %r" % program)
            else:
                named_support = dict(row)
        else:
            mask = record.get("mask_search") or {}
            same = (
                str(mask.get("program")) == program
                and sorted(
                    str(uid) for uid in (mask.get("final_excluded") or [])
                ) == sorted(excluded)
            )
            if same and mask.get("support") is not None:
                named_support = dict(mask["support"])
            else:
                missing.append(
                    "no Support row for the masked plan the Agent named"
                )
    return {
        "named_plan": named,
        "named_plan_delayed_aggregate_gain": (
            None if named_delayed is None else float(named_delayed)
        ),
        "named_plan_support": named_support,
        "full_batch_pool": pool,
        "missing": missing,
    }


def _replay_adoption(record: Mapping[str, Any]) -> dict[str, Any]:
    """Re-decide one arm-target's adoption under the faithful v2 ladder."""
    episode_id = str(record["episode_id"])
    v1_final = record.get("final_plan")
    v1_delayed = (
        None if record.get("delayed") is None
        else float(record["delayed"]["aggregate_gain"])
    )
    v1_support = (
        None if record.get("support") is None
        else float(record["support"]["aggregate_gain"])
    )
    v1_gate = record.get("identity_incumbent_gate") or {}
    inputs = _replay_inputs(record)
    if inputs["missing"]:
        return {
            "episode_id": episode_id,
            "target_id": str(record["target_id"]),
            "arm": str(record["arm"]),
            "status": "REPLAY_INSUFFICIENT_DATA",
            "missing": inputs["missing"],
            "v1_final_plan": v1_final,
            "v1_support_aggregate_gain": v1_support,
            "v1_delayed_aggregate_gain": v1_delayed,
            "consumer_retrains_total": int(
                record.get("consumer_retrains_total") or 0
            ),
            "charged_evaluations": int(record.get("evaluations_used") or 0),
            "llm_calls_in_v1": int(record.get("llm_calls") or 0),
        }

    pool = inputs["full_batch_pool"]
    supports = {
        program: row["support_aggregate_gain"] for program, row in pool.items()
    }
    ranked = sorted(supports, key=lambda op: (-supports[op], _menu_index(op)))
    top = ranked[0]
    has_winner = supports[top] > 0.0
    winner = top if has_winner else None

    tape = _DelayedTape({
        program: row["delayed_aggregate_gain"]
        for program, row in pool.items()
    })
    if winner is None:
        bar = 0.0
        bar_source = "no Support winner, so the bar is identity at zero"
    else:
        bar = max(0.0, tape.full_batch(winner, "bar"))
        bar_source = (
            "max(0, `%s` full-batch delayed), and `%s` is the Support winner"
            % (winner, winner)
        )
    named_delayed = tape.named(inputs["named_plan_delayed_aggregate_gain"])
    passed = named_delayed >= bar

    if passed:
        path = "GATE_PASS_ADOPT_NAMED"
        adopted = dict(inputs["named_plan"])
        adopted_support = dict(inputs["named_plan_support"])
        adopted_delayed = named_delayed
        path_text = (
            "the named plan cleared the bar (%+.6f >= %+.6f)"
            % (named_delayed, bar)
        )
    elif winner is not None and pool[winner]["delayed_aggregate_gain"] > 0.0:
        path = "GATE_FAIL_FALLBACK_SUPPORT_WINNER"
        adopted = {"program": winner, "excluded_series": []}
        harmed = None
        for row in record.get("measured_plans") or []:
            if str(row["program"]) == winner and row.get("full_batch"):
                harmed = row.get("harmed_evaluation_series_count")
        adopted_support = {
            "aggregate_gain": supports[winner],
            "harmed_eval_series_count": harmed,
            "harmed_eval_series_total_harm": None,
            "harmed_eval_series": None,
            "harm_account_note": (
                "v1 persisted the aggregate gains of every full-batch plan "
                "but the full harm account only of the plan it adopted; "
                "nothing is estimated to fill that in"
            ),
        }
        adopted_delayed = pool[winner]["delayed_aggregate_gain"]
        path_text = (
            "the named plan missed the bar by %+.6f, so the ladder fell back "
            "to the Support winner `%s`, whose full-batch delayed is positive"
            % (named_delayed - bar, winner)
        )
    else:
        path = "GATE_FAIL_FALLBACK_IDENTITY"
        adopted = {"program": IDENTITY, "excluded_series": []}
        adopted_support = {
            "aggregate_gain": 0.0, "harmed_eval_series_count": 0,
            "harmed_eval_series_total_harm": 0.0, "harmed_eval_series": [],
        }
        adopted_delayed = 0.0
        path_text = (
            "the named plan missed the bar by %+.6f and there is no Support "
            "winner with a positive full-batch delayed, so the ladder fell to "
            "identity" % (named_delayed - bar)
        )

    reference = record.get("reference_delayed_aggregate_gain")
    reference_plan = record.get("reference_plan") or {}
    capture = None
    if reference is not None and float(reference) > 0.0:
        capture = adopted_delayed / float(reference)
    matches_reference = (
        str(adopted["program"]) == str(reference_plan.get("program"))
        and sorted(str(uid) for uid in adopted.get("excluded_series") or [])
        == sorted(
            str(uid) for uid in reference_plan.get("excluded_series") or []
        )
    )
    changed = (
        v1_final is None
        or str(v1_final.get("program")) != str(adopted["program"])
        or sorted(str(u) for u in (v1_final.get("excluded_series") or []))
        != sorted(str(u) for u in (adopted.get("excluded_series") or []))
    )

    # What the literal v2 rule -- no Support-sign check on the bar program --
    # would have adopted, so the one declared addition is measured rather than
    # assumed.
    literal_bar = max(0.0, pool[top]["delayed_aggregate_gain"])
    if named_delayed >= literal_bar:
        literal_plan = dict(inputs["named_plan"])
        literal_delayed = named_delayed
        literal_path = "GATE_PASS_ADOPT_NAMED"
    elif pool[top]["delayed_aggregate_gain"] > 0.0:
        literal_plan = {"program": top, "excluded_series": []}
        literal_delayed = pool[top]["delayed_aggregate_gain"]
        literal_path = "GATE_FAIL_FALLBACK_BEST_FULL_BATCH"
    else:
        literal_plan = {"program": IDENTITY, "excluded_series": []}
        literal_delayed = 0.0
        literal_path = "GATE_FAIL_FALLBACK_IDENTITY"

    return {
        "episode_id": episode_id,
        "target_id": str(record["target_id"]),
        "arm": str(record["arm"]),
        "status": "REPLAYED",
        "full_batch_pool": pool,
        "support_ranking": ranked,
        "top_support_program": top,
        "support_winner": winner,
        "support_winner_note": (
            None if has_winner else
            "the highest-Support full-batch plan is %r at %+.6f, which is not "
            "positive, so nothing here is a plan a deployer could adopt"
            % (top, supports[top])
        ),
        "bar": bar,
        "bar_source": bar_source,
        "named_plan": inputs["named_plan"],
        "named_plan_delayed_aggregate_gain": named_delayed,
        "named_plan_margin": named_delayed - bar,
        "gate_passed": passed,
        "path": path,
        "path_text": path_text,
        "delayed_reads": tape.reads,
        "delayed_reads_count": len(tape.reads),
        "adopted_plan": adopted,
        "adopted_support": adopted_support,
        "adopted_support_aggregate_gain": float(
            adopted_support["aggregate_gain"]
        ),
        "adopted_delayed_aggregate_gain": float(adopted_delayed),
        "capture_ratio": capture,
        "matches_reference_plan": matches_reference,
        "reference_plan": reference_plan or None,
        "reference_delayed_aggregate_gain": reference,
        "changed_from_v1": changed,
        "v1_final_plan": v1_final,
        "v1_support_aggregate_gain": v1_support,
        "v1_delayed_aggregate_gain": v1_delayed,
        "v1_bar": v1_gate.get("bar"),
        "v1_gate_passed": v1_gate.get("passed"),
        "v1_path": (
            "the gate passed and v1 adopted the named plan"
            if v1_gate.get("passed")
            else "the gate failed and v1 dropped straight to identity"
        ),
        "v1_capture_ratio": record.get("capture_ratio"),
        "delayed_delta_vs_v1": (
            None if v1_delayed is None else adopted_delayed - v1_delayed
        ),
        "literal_v2_sensitivity": {
            "bar": literal_bar,
            "adopted_plan": literal_plan,
            "delayed_aggregate_gain": literal_delayed,
            "path": literal_path,
            "same_as_replay": (
                str(literal_plan["program"]) == str(adopted["program"])
                and sorted(
                    str(u) for u in literal_plan.get("excluded_series") or []
                ) == sorted(
                    str(u) for u in adopted.get("excluded_series") or []
                )
            ),
        },
        "consumer_retrains_total": int(
            record.get("consumer_retrains_total") or 0
        ),
        "charged_evaluations": int(record.get("evaluations_used") or 0),
        "llm_calls_in_v1": int(record.get("llm_calls") or 0),
    }


def _replay_target_verdict(
    a3: Mapping[str, Any] | None,
    a5: Mapping[str, Any] | None,
    v1_target: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        a3 is None or a5 is None
        or a3.get("status") != "REPLAYED" or a5.get("status") != "REPLAYED"
    ):
        return {
            "label": "REPLAY_INSUFFICIENT_DATA",
            "v1_label": v1_target.get("label"),
            "delivery": v1_target.get("delivery"),
            "reason": "an arm of this target could not be replayed",
            "paired_delayed_delta": None,
        }
    a3_delayed = float(a3["adopted_delayed_aggregate_gain"])
    a5_delayed = float(a5["adopted_delayed_aggregate_gain"])
    delta = a5_delayed - a3_delayed
    if delta > QUALITY_DELTA_THRESHOLD:
        label = "A5_WINS"
    elif delta < -QUALITY_DELTA_THRESHOLD:
        label = "A5_LOSES"
    else:
        label = "TIE"
    return {
        "label": label,
        "v1_label": v1_target.get("label"),
        "label_changed": label != v1_target.get("label"),
        "delivery": v1_target.get("delivery"),
        "delivery_source": (
            "inherited verbatim from v1; the replay does not touch what A5 "
            "cited at stage one"
        ),
        "clauses_cited_by_a5": v1_target.get("clauses_cited_by_a5"),
        "clauses_available": v1_target.get("clauses_available"),
        "paired_delayed_delta": delta,
        "v1_paired_delayed_delta": v1_target.get("paired_delayed_delta"),
        "a3_adopted_plan": a3["adopted_plan"],
        "a5_adopted_plan": a5["adopted_plan"],
        "a3_v1_final_plan": a3["v1_final_plan"],
        "a5_v1_final_plan": a5["v1_final_plan"],
        "a3_support_aggregate_gain": a3["adopted_support_aggregate_gain"],
        "a5_support_aggregate_gain": a5["adopted_support_aggregate_gain"],
        "a3_delayed_aggregate_gain": a3_delayed,
        "a5_delayed_aggregate_gain": a5_delayed,
        "a3_capture_ratio": a3["capture_ratio"],
        "a5_capture_ratio": a5["capture_ratio"],
        "a3_consumer_retrains": a3["consumer_retrains_total"],
        "a5_consumer_retrains": a5["consumer_retrains_total"],
        "a3_changed_from_v1": a3["changed_from_v1"],
        "a5_changed_from_v1": a5["changed_from_v1"],
        "reason": (
            "paired delayed delta %+.6f (A5 %+.6f - A3 %+.6f) against a "
            "threshold of %.3f; delivery %s inherited from v1"
            % (
                delta, a5_delayed, a3_delayed, QUALITY_DELTA_THRESHOLD,
                v1_target.get("delivery"),
            )
        ),
    }


def _replay_overall(
    per_target: Mapping[str, Mapping[str, Any]],
    cards: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """The same label ladder the bridge run pre-registered."""
    labels = [str(row["label"]) for row in per_target.values()]
    if "REPLAY_INSUFFICIENT_DATA" in labels:
        stuck = sorted(
            key for key, row in per_target.items()
            if row["label"] == "REPLAY_INSUFFICIENT_DATA"
        )
        return (
            "REPLAY_INSUFFICIENT_DATA",
            "these targets are missing numbers the ladder needs: %s"
            % ", ".join(stuck),
        )
    if all(
        str(cards[key]["status"]) == "ABSTAIN_TO_DEFAULT" for key in per_target
    ):
        return (
            "SKILL_COMPILATION_INSUFFICIENT_PROVENANCE",
            "every target compiled to ABSTAIN_TO_DEFAULT",
        )
    losers = sorted(
        key for key, row in per_target.items() if row["label"] == "A5_LOSES"
    )
    if losers:
        return (
            "SKILL_LOSES_SIGNAL",
            "A5 lost on %s; the clauses it cited there were %s"
            % (
                ", ".join(losers),
                "; ".join(
                    "%s:%s" % (key, per_target[key].get("clauses_cited_by_a5"))
                    for key in losers
                ),
            ),
        )
    undelivered = sorted(
        key for key, row in per_target.items()
        if row.get("delivery") == "SKILL_NOT_DELIVERED"
    )
    if undelivered:
        return (
            "SKILL_LOSES_SIGNAL",
            "no target lost on quality, but the card was not delivered on %s"
            % ", ".join(undelivered),
        )
    if "A5_WINS" in labels:
        return (
            "SKILL_BRIDGE_DELIVERS",
            "delivery holds on every non-abstaining target, no target is "
            "A5_LOSES, and A5 wins on %s"
            % ", ".join(
                sorted(
                    key for key, row in per_target.items()
                    if row["label"] == "A5_WINS"
                )
            ),
        )
    return (
        "SKILL_BRIDGE_NO_EFFECT",
        "delivery holds, nothing loses, and nothing wins by more than %.3f"
        % QUALITY_DELTA_THRESHOLD,
    )


def _replay_cost(
    rows: Sequence[Mapping[str, Any]], v1_cost: Mapping[str, Any],
) -> dict[str, Any]:
    """The same retrains, re-attributed against the corrected adoptions."""
    per_arm_target = {
        str(row["episode_id"]): {
            "consumer_retrains": int(row["consumer_retrains_total"]),
            "charged_evaluations": int(row["charged_evaluations"]),
            "llm_calls": int(row["llm_calls_in_v1"]),
            "delayed_aggregate_gain": (
                None if row.get("status") != "REPLAYED"
                else float(row["adopted_delayed_aggregate_gain"])
            ),
            "v1_delayed_aggregate_gain": row.get("v1_delayed_aggregate_gain"),
        }
        for row in rows
    }
    first_positive: dict[str, Any] = {}
    positive_counts: dict[str, dict[str, int]] = {}
    for arm in ("A3", "A5"):
        cumulative = 0
        reached = None
        positives = 0
        v1_positives = 0
        for row in rows:
            if str(row["arm"]) != arm:
                continue
            cumulative += int(row["consumer_retrains_total"])
            delayed = (
                None if row.get("status") != "REPLAYED"
                else float(row["adopted_delayed_aggregate_gain"])
            )
            if delayed is not None and delayed > 0.0:
                positives += 1
                if reached is None:
                    reached = {
                        "at_episode": str(row["episode_id"]),
                        "cumulative_consumer_retrains": cumulative,
                        "delayed_aggregate_gain": delayed,
                    }
            v1_delayed = row.get("v1_delayed_aggregate_gain")
            if v1_delayed is not None and float(v1_delayed) > 0.0:
                v1_positives += 1
        first_positive[arm] = reached or {
            "at_episode": None,
            "cumulative_consumer_retrains": cumulative,
            "note": "no delayed-positive adoption in this arm",
        }
        positive_counts[arm] = {
            "delayed_positive_adoptions": positives,
            "delayed_positive_adoptions_in_v1": v1_positives,
            "arm_targets": sum(1 for row in rows if str(row["arm"]) == arm),
        }
    totals = {
        arm: sum(
            int(row["consumer_retrains_total"])
            for row in rows if str(row["arm"]) == arm
        )
        for arm in ("A3", "A5")
    }
    v1_totals = dict(v1_cost.get("arm_totals") or {})
    for arm, value in totals.items():
        if int(v1_totals.get(arm, value)) != value:
            raise SystemExit(
                "the replay changed a retrain total for %s: %s vs %s"
                % (arm, v1_totals.get(arm), value)
            )
    return {
        "per_arm_target": per_arm_target,
        "arm_totals": totals,
        "arm_totals_unchanged_from_v1": True,
        "consumer_retrains_added_by_the_replay": 0,
        "llm_calls_added_by_the_replay": 0,
        "cumulative_retrains_to_first_delayed_positive_adoption": (
            first_positive
        ),
        "v1_cumulative_retrains_to_first_delayed_positive_adoption": dict(
            v1_cost.get(
                "cumulative_retrains_to_first_delayed_positive_adoption"
            ) or {}
        ),
        "delayed_positive_adoption_counts": positive_counts,
        "has_no_criterion": True,
    }


def _replay_markdown(payload: Mapping[str, Any]) -> str:
    per_target = payload["per_target"]
    rows = payload["arm_targets"]
    lines = [
        "# recipe experience -> Source Skill -> Fast, v2 ladder replay",
        "",
        "**Instrument-corrected reading: `%s`** -- %s."
        % (payload["overall_verdict"], payload["overall_verdict_reason"]),
        "",
        "**Filed beside `%s`, which read `%s`.**  That artifact is not "
        "overwritten and not amended; both readings stand, and this one "
        "carries the corrected adoption ladder."
        % (payload["v1_artifact"], payload["v1_overall_verdict"]),
        "",
        "The bridge run's adoption gate dropped to identity the moment a "
        "named plan missed the bar.  The frozen v2 rule has a rung in "
        "between: fall back to the best full-batch plan when its delayed gain "
        "is positive, and only then to identity.  That run also set the bar "
        "from the highest delayed among the eligible full-batch plans, where "
        "v2 sets it from exactly one plan, the Support winner.  This replay "
        "re-decides the adoption stage of the six recorded arm-targets from "
        "numbers that run already measured.  0 LLM calls, 0 new Consumer "
        "retrains, no data touched.",
        "",
        "**Engineering instrument correction, not authorization evidence.**  "
        "No Skill is promoted, no TRY right is granted, and no Fast or Slow "
        "path of the real Harness runs.",
        "",
        "## What changed",
        "",
        "| arm-target | old adopted | old delayed | new adopted | new delayed "
        "| path | changed |",
        "| --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        if row.get("status") != "REPLAYED":
            lines.append(
                "| `%s` | -- | -- | -- | -- | `%s` | -- |"
                % (row["episode_id"], row["status"])
            )
            continue
        lines.append(
            "| `%s` | %s | %+.6f | %s | %+.6f | `%s` | %s |"
            % (
                row["episode_id"], _plan_label(row["v1_final_plan"]),
                float(row["v1_delayed_aggregate_gain"]),
                _plan_label(row["adopted_plan"]),
                float(row["adopted_delayed_aggregate_gain"]),
                row["path"], "**yes**" if row["changed_from_v1"] else "no",
            )
        )
    lines += [
        "",
        "## Recomputed labels",
        "",
        "| target | A3 delayed | A5 delayed | paired delta | label | old "
        "label | delivery |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for key in sorted(per_target):
        row = per_target[key]
        if row.get("paired_delayed_delta") is None:
            lines.append(
                "| `%s` | -- | -- | -- | `%s` | `%s` | `%s` |"
                % (key, row["label"], row.get("v1_label"), row.get("delivery"))
            )
            continue
        lines.append(
            "| `%s` | %+.6f | %+.6f | **%+.6f** | `%s` | `%s` | `%s` |"
            % (
                key, row["a3_delayed_aggregate_gain"],
                row["a5_delayed_aggregate_gain"],
                row["paired_delayed_delta"], row["label"],
                row.get("v1_label"), row.get("delivery"),
            )
        )
    lines += [
        "",
        "Capture against each target's full-search reference: "
        + "; ".join(
            "%s A3 %s -> A5 %s"
            % (
                key,
                "--" if per_target[key].get("a3_capture_ratio") is None
                else "%.3f" % per_target[key]["a3_capture_ratio"],
                "--" if per_target[key].get("a5_capture_ratio") is None
                else "%.3f" % per_target[key]["a5_capture_ratio"],
            )
            for key in sorted(per_target)
        )
        + ".",
    ]
    cost = payload["cost_report"]
    lines += [
        "",
        "## Cost, re-attributed",
        "",
        "Total retrains are a property of what was run and do not move: A3 "
        "%d, A5 %d, %d in all.  Only which adoption counts as the first "
        "delayed-positive one is recomputed."
        % (
            cost["arm_totals"]["A3"], cost["arm_totals"]["A5"],
            cost["arm_totals"]["A3"] + cost["arm_totals"]["A5"],
        ),
        "",
        "| arm | first delayed-positive adoption | cumulative retrains | "
        "old reading |",
        "| --- | --- | ---: | --- |",
    ]
    first = cost["cumulative_retrains_to_first_delayed_positive_adoption"]
    v1_first = cost[
        "v1_cumulative_retrains_to_first_delayed_positive_adoption"
    ]
    for arm in ("A3", "A5"):
        row = first[arm]
        old = v1_first.get(arm) or {}
        lines.append(
            "| %s | `%s` | %s | `%s` at %s |"
            % (
                arm, row.get("at_episode"),
                row.get("cumulative_consumer_retrains"),
                old.get("at_episode"),
                old.get("cumulative_consumer_retrains"),
            )
        )
    counts = cost["delayed_positive_adoption_counts"]
    lines += [
        "",
        "Delayed-positive adoptions: A3 %d of %d (was %d), A5 %d of %d "
        "(was %d)."
        % (
            counts["A3"]["delayed_positive_adoptions"],
            counts["A3"]["arm_targets"],
            counts["A3"]["delayed_positive_adoptions_in_v1"],
            counts["A5"]["delayed_positive_adoptions"],
            counts["A5"]["arm_targets"],
            counts["A5"]["delayed_positive_adoptions_in_v1"],
        ),
        "",
        "## The ladder, against the frozen rule",
        "",
        "| rung | relation to v2 | note |",
        "| --- | --- | --- |",
    ]
    for item in payload["v2_correspondence"]:
        lines.append(
            "| %s | `%s` | %s |"
            % (item["rung"], item["relation"], item["note"])
        )
    lines += [
        "",
        "Selection uses Support only.  Delayed is consulted at most twice per "
        "arm-target -- once for the Support winner, to set the bar, and once "
        "for the named plan, to confirm it -- and every read is recorded on "
        "that arm-target's `delayed_reads` tape.",
        "",
        "## Per arm-target",
        "",
    ]
    for row in rows:
        lines.append("### `%s`" % row["episode_id"])
        lines.append("")
        if row.get("status") != "REPLAYED":
            lines.append(
                "`%s`: %s" % (row["status"], "; ".join(row["missing"]))
            )
            lines.append("")
            continue
        lines.append(
            "- full-batch pool: %s"
            % "; ".join(
                "`%s` support %+.6f delayed %+.6f"
                % (
                    program, item["support_aggregate_gain"],
                    item["delayed_aggregate_gain"],
                )
                for program, item in row["full_batch_pool"].items()
            )
        )
        lines.append(
            "- Support winner: %s"
            % (
                "`%s`" % row["support_winner"] if row["support_winner"]
                else "none -- %s" % row["support_winner_note"]
            )
        )
        lines.append(
            "- bar %+.6f (%s); named plan %s at delayed %+.6f, margin %+.6f"
            % (
                row["bar"], row["bar_source"], _plan_label(row["named_plan"]),
                row["named_plan_delayed_aggregate_gain"],
                row["named_plan_margin"],
            )
        )
        lines.append("- %s" % row["path_text"])
        lines.append(
            "- adopted %s at support %+.6f, delayed %+.6f%s (the old run "
            "adopted %s at delayed %+.6f under a bar of %+.6f)"
            % (
                _plan_label(row["adopted_plan"]),
                row["adopted_support_aggregate_gain"],
                row["adopted_delayed_aggregate_gain"],
                "" if row["capture_ratio"] is None
                else ", capture %.3f" % row["capture_ratio"],
                _plan_label(row["v1_final_plan"]),
                float(row["v1_delayed_aggregate_gain"]),
                float(row["v1_bar"]),
            )
        )
        lines.append(
            "- delayed numbers consulted: %d (%s)"
            % (
                row["delayed_reads_count"],
                "; ".join(
                    "%s %s %+.6f"
                    % (
                        item["role"], item["program"],
                        item["delayed_aggregate_gain"],
                    )
                    for item in row["delayed_reads"]
                ),
            )
        )
        sens = row["literal_v2_sensitivity"]
        lines.append(
            "- literal v2, without the Support-sign check, would adopt %s at "
            "%+.6f (`%s`) -- %s"
            % (
                _plan_label(sens["adopted_plan"]),
                sens["delayed_aggregate_gain"], sens["path"],
                "same" if sens["same_as_replay"] else "**different**",
            )
        )
        lines.append("")
    lines += [
        "## Standing",
        "",
        "- %s." % REPLAY_PRE_REGISTERED["standing"],
        "- %s." % REPLAY_PRE_REGISTERED["why_a_replay_is_legitimate_here"],
        "- %s." % REPLAY_PRE_REGISTERED["not_replayed"],
        "- the harm accounts of newly adopted plans are not reproduced here: "
        "the bridge run persisted the aggregate gains of every full-batch "
        "plan but the full harm account only of the plan it adopted, and this "
        "replay estimates nothing it does not have.",
        "",
    ]
    return "\n".join(lines) + "\n"


def replay_ladder(*, dry_run: bool = False) -> int:
    """Re-decide the recorded adoptions under the faithful v2 ladder."""
    started = time.perf_counter()
    if not OUT_JSON.exists():
        raise SystemExit(
            "the replay needs %s, which is not there"
            % _repo_relative(OUT_JSON)
        )
    source_bytes = OUT_JSON.read_bytes()
    v1 = json.loads(source_bytes.decode("utf-8"))
    if str(v1.get("protocol_version")) != PROTOCOL_VERSION:
        raise SystemExit(
            "unexpected protocol_version %r in %s"
            % (v1.get("protocol_version"), _repo_relative(OUT_JSON))
        )
    rows = [_replay_adoption(record) for record in v1["episodes"]]
    by_id = {str(row["episode_id"]): row for row in rows}
    per_target = {
        target_id: _replay_target_verdict(
            by_id.get("%s_A3" % target_id),
            by_id.get("%s_A5" % target_id),
            v1["per_target"][target_id],
        )
        for target_id in v1["per_target"]
    }
    overall, reason = _replay_overall(per_target, v1["skill_cards"])
    payload = {
        "protocol_version": REPLAY_PROTOCOL_VERSION,
        "role": (
            "the instrument-corrected reading of the Skill-bridge experiment: "
            "the same six recorded arm-targets, re-decided under a faithful "
            "port of the frozen v2 adoption ladder"
        ),
        "not_authorization_evidence": v1["not_authorization_evidence"],
        "overall_verdict": overall,
        "overall_verdict_reason": reason,
        "v1_overall_verdict": v1["overall_verdict"],
        "v1_overall_verdict_reason": v1["overall_verdict_reason"],
        "v1_artifact": _repo_relative(OUT_JSON),
        "v1_artifact_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "verdict_changed": overall != str(v1["overall_verdict"]),
        "per_target": per_target,
        "label_counts": {
            label: sum(
                1 for row in per_target.values() if row["label"] == label
            )
            for label in (
                "A5_WINS", "A5_LOSES", "TIE", "REPLAY_INSUFFICIENT_DATA",
            )
        },
        "v1_label_counts": v1["label_counts"],
        "arm_targets": rows,
        "changed_arm_targets": [
            str(row["episode_id"]) for row in rows
            if row.get("status") == "REPLAYED" and row["changed_from_v1"]
        ],
        "pre_registered": REPLAY_PRE_REGISTERED,
        "v2_correspondence": REPLAY_V2_CORRESPONDENCE,
        "v1_gate_as_run": v1["v2_gate_semantics_check"],
        "cost_report": _replay_cost(rows, v1["cost_report"]),
        "skill_cards_status": {
            key: card["status"] for key, card in v1["skill_cards"].items()
        },
        "skill_cards_artifact": v1["skill_cards_artifact"],
        "model": None,
        "llm_call_count": 0,
        "consumer_retrains_added": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    if OUT_JSON.read_bytes() != source_bytes:
        raise SystemExit("the replay changed its own input; refusing to write")
    if dry_run:
        print(json.dumps(
            {
                "overall": overall,
                "v1_overall": v1["overall_verdict"],
                "changed": payload["changed_arm_targets"],
                "per_target": {
                    key: {
                        "label": row["label"], "v1_label": row.get("v1_label"),
                        "delta": row.get("paired_delayed_delta"),
                    }
                    for key, row in per_target.items()
                },
            },
            indent=2, ensure_ascii=False, default=str,
        ))
        return 0
    E2.mkdir(parents=True, exist_ok=True)
    REPLAY_OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    REPLAY_OUT_MD.write_text(
        _replay_markdown(payload), encoding="utf-8", newline="\n",
    )
    print("wrote", REPLAY_OUT_JSON, flush=True)
    print("wrote", REPLAY_OUT_MD, flush=True)
    print("overall %s (was %s)" % (overall, v1["overall_verdict"]), flush=True)
    print(
        "labels", json.dumps(payload["label_counts"], sort_keys=True),
        flush=True,
    )
    print("changed", payload["changed_arm_targets"], flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cards-only", action="store_true",
        help="compile and write the Skill cards, then stop (0 LLM)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run both arms but print the verdict instead of writing",
    )
    parser.add_argument(
        "--replay-ladder", action="store_true",
        help=(
            "re-decide the recorded adoptions under a faithful port of the "
            "frozen v2 fallback ladder and write the corrected reading "
            "beside the original (0 LLM, 0 new Consumer retrains)"
        ),
    )
    args = parser.parse_args(argv)
    if args.replay_ladder:
        return replay_ladder(dry_run=bool(args.dry_run))
    if args.cards_only:
        cards = {
            target_id: compile_skill_card(target)
            for target_id, target in TARGETS.items()
        }
        for target_id, card in cards.items():
            print("=== %s: %s" % (target_id, card["status"]))
            print(render_skill_card(card))
            print()
        return 0
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
