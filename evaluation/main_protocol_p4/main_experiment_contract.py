"""The frozen main-experiment contract: Static / A3 / A5 on a scoped Harness.

Written before any arm runs, so a later runner cannot widen a budget, add an
origin or reinterpret a verdict.  Runners import these declarations;
``assert_frozen`` re-derives the parts that can be checked mechanically.

What is finally on trial here is the Harness, not a program.  Every earlier line
measured a fixed program applied globally, which is why its failures said
nothing about the project's claim: a Scope over training rows could not express
"do not treat this served series", because the served series was never treated
(``p4n``).  With the serving-side dual pipeline and the ScopeSpec in place, the
subject of the experiment is the policy the Harness actually deploys.

The geometry follows from one fact.  Anchors are frozen at ``[312 ... 852]`` and
every one clears ``anchor + 48 <= origin`` past 900, so the training corpus does
not change with the origin -- only the scoring window does.  A new **cohort** is
therefore what makes a new training condition, and that is why the Target and
the Source are new series rather than new origins.

Cohorts 1 and 2 are spent: cohort 1 chose the Phase-2 menus, cohort 2 confirmed
them once.  Neither may appear here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPLY = PROJECT_ROOT / "artifacts/main_protocol/p4s_main_experiment_supply.json"
LEDGER = PROJECT_ROOT / "artifacts/main_protocol/p4t_exposure_ledger.json"

DATA_VERSION = preflight.DATA_VERSION
CONTEXT, HORIZON = preflight.CONTEXT, preflight.HORIZON
FACES = ("support_a", "support_b")

TARGET_SLICE = (80, 120)
SOURCE_SLICE = (160, 200)

#: Target held-in.  Cohort 3 has never been fitted, so re-using origins cohorts
#: 1 and 2 were read at still gives a new corpus and a new model.
HELD_IN_ORIGINS = (1896, 2136, 2376, 2616, 2856)

#: Target held-out.  Every (Target series, origin) pair was checked against all
#: 70 artifacts this data version produced; none had been scored (``p4t``).
HELD_OUT_ORIGINS = (4056, 4296, 4536, 4776, 5016)

ANCHOR_AXIS_FROZEN = True

#: Inherited unchanged from the adjudicated split-2 operating point so the arms
#: stay comparable with P4b.  A3 and A5 receive identical vectors; A5 has no
#: budget exception, only frozen prior Skills.
PER_ARM_BUDGET = {
    "support_a_rounds": 7,
    "independent_support_b_rounds": 1,
    "probes": 24,
    "llm_calls": 6,
    "tokens": 60000,
    "accepted_updates": 1,
    "wall_seconds": 2700,
    "identical_for_a3_and_a5": True,
}

ARMS = {
    "Static": {"llm": False, "role": "reference", "scope": "none"},
    "GlobalBestFixed": {"llm": False, "role": "baseline",
                        "scope": "all_serving_series"},
    "OpenLoopTargeter": {"llm": False, "role": "baseline",
                         "note": "the diagnostic router, never the method"},
    "A3": {"llm": True, "role": "ablation",
           "what": "Target-local Scoped Harness with no prior Skills"},
    "A5": {"llm": True, "role": "full system",
           "what": "audited Source Skills, then Target held-in calibration"},
    "ParallelAtB": {"llm": True, "role": "equal-budget baseline"},
}

PRIMARY_CONTRASTS = ("A3 - Static", "A5 - A3")

#: The Fast schema has no Scope channel and this round does not add one: a field
#: no instruction mentions would be dead weight and a new snapshot difference.
#: The proven semantics are Fast proposes a Program, the Runtime initialises a
#: Scope from deployment-visible features, and Slow revises it from held-in
#: feedback -- the Harness forms and revises the Scope, which is what the claim
#: requires; it never required the Fast call to emit one in a single shot.
SCOPE_ORIGINATION = {
    "fast_proposes": "Program only",
    "runtime_initializes": "evaluation.main_protocol_p4.scope_initializer",
    "slow_revises": "atomically with the Program, via the PATCH's serving_scope",
    "fast_propose_schema_changed": False,
    "rules": (
        "the initialiser reads only deployment-visible features and may not "
        "see a UID, a cohort name or any Outcome; its family table and "
        "thresholds are frozen before the run and are identical for A3 and A5; "
        "the Runner may supply or resolve a Scope but may never hand-pick one "
        "after seeing a result; every deployment records scope_source as one of "
        "runtime_initializer / source_skill / slow_revision"
    ),
    "a_later_fast_emitted_scope_is_a_separate_version": (
        "it would need the schema, H0, the parsing chain and the snapshot lock "
        "changed together, then validation on a new unexposed Target; it is not "
        "part of this round and is not an H3"
    ),
}

#: Coverage counts *treated* series, not *assigned* ones.  A (program, scope)
#: pair can be refused after assignment -- by the window verifier, or because
#: preparing the served context flattened it -- and the series then falls back
#: to Static.  That fallback is a fail-closed abstention: correct behaviour, but
#: not coverage.  Counting it as covered would let an arm claim credit for
#: reach it never had, and would make a policy that abstains everywhere look
#: fully deployed.
COVERAGE_SEMANTICS_SUMMARY = (
    "treated / served, where treated means the series actually ran through a "
    "program pipeline; assigned-but-refused counts as abstention, not coverage"
)


def deployment_coverage(treated: int, served: int) -> float:
    """The one definition every arm uses.  Static is 0.0 by construction."""
    return round(treated / served, 4) if served else 0.0


#: Reported for the policy the Harness deploys, not for any single program.
ENDPOINTS = {
    "utility_vs_static": "mean per-series gain of the deployed policy",
    "deployment_coverage": COVERAGE_SEMANTICS_SUMMARY,
    "harmed_fraction": "<= %.2f" % bounded.BOUNDED_MAX_HARMED_FRACTION,
    "max_single_series_harm": "<= %.2f" % bounded.BOUNDED_MAX_SINGLE_SERIES_HARM,
    "skill_formation": "Skills formed, Scopes revised, causal reuse observed",
    "cost": "LLM calls, tokens, Consumer fits and wall seconds per arm",
}

#: A5 without an approved Source Skill is not A5.  Running it anyway would
#: report the ablation twice and call the second one the full system.
STOPPING_RULES = {
    "A5_TREATMENT_EMPTY": (
        "if the Source line forms zero Active Skills, stop and record it; do "
        "not run A5 as if it had a treatment"
    ),
    "held_out_stays_closed": (
        "until every arm is frozen; then Fast-only, zero feedback, zero "
        "write-back, one reading"
    ),
    "no_baseline_feedback": (
        "Static / BestFixed / Targeter results are sealed before A3 and A5 and "
        "may not be used to change them"
    ),
}

RUN_ORDER = (
    "freeze this contract",
    "run the three 0-LLM baselines and seal them",
    "configure transport explicitly; never let the client default",
    "live Scope smoke on one cell",
    "Source line: form and approve Skills naturally",
    "if Source Active Skills == 0 -> A5_TREATMENT_EMPTY and stop",
    "Target: Static / A3 / A5 / Parallel@B",
    "freeze all arms, then open held-out once",
)

BOUNDARY = {
    "llm_calls_before_transport_is_configured": 0,
    "held_out_reads_before_all_arms_frozen": 0,
    "ucr_test_outcome_reads": 0,
    "natural_final_outcome_reads": 0,
    "thresholds_changed": 0,
    "operators_added": 0,
}

#: Geometry eligibility used whether the horizon contains observed truth -- a
#: mask, never an error.  Declared here rather than discovered later.
DISCLOSED_SELECTION = (
    "cohorts and origins were screened for whether the missing-aware sMASE is "
    "defined at all; no gain, error or utility participated, and the exposure "
    "ledger reads only which cells were evaluated"
)


def cohorts() -> dict[str, list[str]]:
    """The frozen series lists, read from the audited supply artifact."""
    readable = json.loads(SUPPLY.read_text(encoding="utf-8"))["readable_uids"]
    return {
        "target": readable[TARGET_SLICE[0]:TARGET_SLICE[1]],
        "source": readable[SOURCE_SLICE[0]:SOURCE_SLICE[1]],
        "cohort_1_spent": readable[0:40],
        "cohort_2_spent": readable[40:80],
    }


def assert_frozen() -> dict[str, Any]:
    """Re-derive what can be checked, so a drifted runner fails loudly."""
    failures: list[str] = []
    groups = cohorts()
    target, source = set(groups["target"]), set(groups["source"])
    spent = set(groups["cohort_1_spent"]) | set(groups["cohort_2_spent"])
    if len(groups["target"]) != 40 or len(groups["source"]) != 40:
        failures.append("a cohort does not hold 40 series")
    if target & source:
        failures.append("Target and Source overlap")
    if (target | source) & spent:
        failures.append("a spent cohort leaked into the experiment")
    if set(HELD_IN_ORIGINS) & set(HELD_OUT_ORIGINS):
        failures.append("held-in and held-out origins overlap")
    if len(HELD_OUT_ORIGINS) != 5:
        failures.append("the held-out block is not the five frozen origins")
    if not PER_ARM_BUDGET["identical_for_a3_and_a5"]:
        failures.append("A3 and A5 budgets diverged")

    ledger_ok = False
    if LEDGER.is_file():
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
        ledger_ok = (
            ledger.get("verdict") == "ALL_PROPOSED_HELD_OUT_PAIRS_UNEXPOSED"
            and sorted(ledger.get("origins_clear") or ()) == sorted(HELD_OUT_ORIGINS)
        )
    if not ledger_ok:
        failures.append("the exposure ledger does not clear this held-out block")

    return {
        "frozen": not failures,
        "failures": failures,
        "data_version": DATA_VERSION,
        "target_slice": "readable[%d:%d]" % TARGET_SLICE,
        "source_slice": "readable[%d:%d]" % SOURCE_SLICE,
        "held_in_origins": list(HELD_IN_ORIGINS),
        "held_out_origins": list(HELD_OUT_ORIGINS),
        "arms": sorted(ARMS),
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "exposure_ledger_clears_held_out": ledger_ok,
    }


def to_dict() -> Mapping[str, Any]:
    """The contract as a receipt, for the runners to embed verbatim."""
    return {
        "stage": "P4U_MAIN_EXPERIMENT_CONTRACT",
        "data_version": DATA_VERSION,
        "subject": "the policy the Harness deploys, not a fixed program",
        "geometry": {
            "target_slice": "readable[%d:%d]" % TARGET_SLICE,
            "source_slice": "readable[%d:%d]" % SOURCE_SLICE,
            "held_in_origins": list(HELD_IN_ORIGINS),
            "held_out_origins": list(HELD_OUT_ORIGINS),
            "anchor_axis_frozen": ANCHOR_AXIS_FROZEN,
            "why_new_cohorts_not_new_origins": (
                "anchors are frozen and all clear the filter past origin 900, "
                "so the training corpus is origin-invariant; a new cohort is "
                "what makes a new training condition"
            ),
        },
        "arms": ARMS,
        "scope_origination": SCOPE_ORIGINATION,
        "per_arm_budget": PER_ARM_BUDGET,
        "primary_contrasts": list(PRIMARY_CONTRASTS),
        "endpoints": ENDPOINTS,
        "stopping_rules": STOPPING_RULES,
        "run_order": list(RUN_ORDER),
        "boundary": BOUNDARY,
        "disclosed_selection": DISCLOSED_SELECTION,
        "supply_audit": SUPPLY.name,
        "exposure_ledger": LEDGER.name,
    }
