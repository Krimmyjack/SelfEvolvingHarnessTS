"""P4U-v4 / HEC-1: the curve contract -- draft, pending sol and user.

What this version adds, and to what
-----------------------------------
v1 fixed the geometry, the arms and the budgets.  v2 added the RISK_GAP route
and the scoped ADD.  v3 removed the protocol friction, ranked the refusals and
let a Draft be restricted rather than destroyed.  All three are imported and
none is rewritten.  This one changes exactly four things:

* **the course is long enough to see a curve.**  Five units cannot show a
  machine that needs tens of units to accumulate anything; twenty-six KDD units
  in three frozen orderings can at least be read.
* **the two loops are separated.**  Selection now happens on the units already
  processed and verification on units not yet seen, which is the one change that
  answers the ``S3_EDIT_REJECTED`` / Source-line geometry where an edit was
  judged on the very unit it was derived from.
* **Slow stops guessing thresholds.**  It names a feature and a direction; a
  tool calibrates the number on frozen bins, and a shadow search records how
  much of that choice was actually the model's.
* **a failed verification is classified before it is acted on.**  Three states,
  three different available actions, because Source-v3's three losses had three
  different mechanisms and narrowing answers only one of them.

What it does not change
-----------------------
The risk thresholds (0.005 / 0.20 / 0.30), the coverage floor of 5, the window
verifier's 0.35, the operator set, the Observation vocabulary, the Consumer, the
Fast schema, the held-out block, and every earlier artifact.  ``BOUNDARY`` states
those as zeros and ``assert_frozen`` re-derives them.

Status
------
**Frozen 2026-09-03.**  sol confirmed all thirteen mainline defaults and added
six rulings (``SOL_RULINGS``); the user released the four budget envelopes
(``USER_RELEASES``).  ``assert_frozen`` re-derives mechanical drift and
``assert_launchable`` now passes for Phase S and the three Phase-T orderings and
still refuses Phase F, which needs both a supported verdict and a human seal
release.  No field changes from here; anything the run turns up is appended as an
erratum and goes to HEC-2.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import main_experiment_contract as v1
from evaluation.main_protocol_p4 import main_experiment_contract_v2 as v2
from evaluation.main_protocol_p4 import main_experiment_contract_v3 as v3
from evaluation.main_protocol_p4 import hec1_scoreability as scoreability
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import scope_initializer as initializer
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from evaluation.main_protocol_p4 import scope_repair_distance as distance
from evaluation.main_protocol_p4 import scope_threshold_tool as threshold_tool
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPLY = PROJECT_ROOT / "artifacts/main_protocol/p4ac_hec1_course_supply.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/hec1_contract.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/hec1_contract.md"

#: v4.1 = HEC-1 v1.1 (sol's six rulings of 2026-09-03 13:xx applied before any
#: scientific ordering ran; the v1 Phase S and Forward are superseded).
VERSION = "P4U-v4.1"
SUPERSEDES_VERSION = "P4U-v4"

#: sol v1.1 R-B: every scientific ordering runs from one commit with the HEC-1
#: runner files clean.  The commit id is git's own; nothing else is hashed.
#: The runner records ``code_state`` per artifact and refuses a live scientific
#: launch on a dirty runner file; the readout asserts one commit across the
#: three orderings.  A shakedown run is exempt and is excluded from the curve.
CODE_FREEZE = {
    "policy": (
        "one commit for Phase S-v1.1, Forward, Reverse and Interleaved; runner "
        "files unmodified while they run; a modification inside an ordering "
        "downgrades that ordering to shakedown"),
    "runner_files": (
        "evaluation/main_protocol_p4/run_hec1.py",
        "evaluation/main_protocol_p4/outer_loop.py",
        "evaluation/main_protocol_p4/restricted_draft.py",
        "evaluation/main_protocol_p4/scope_threshold_tool.py",
        "evaluation/main_protocol_p4/hec1_contract.py",
        "evaluation/main_protocol_p4/scoped_serving_evaluator.py",
        "evaluation/main_protocol_p4/scope_narrowing_preflight.py",
        "evaluation/main_protocol_p4/scope_initializer.py",
        "evaluation/main_protocol_p4/audit_hec1_instrument.py",
    ),
    "recorded_as": "artifact.code_state = {code_commit, runner_files_dirty}",
    "asserted_by": "audit_hec1_readout: one code_commit across the three "
                   "orderings, no dirty runner file, no shakedown artifact",
    "no_new_hash_infrastructure": True,
}

#: Phase S and Phase T did not run on the same commit, and the readout must say
#: so rather than let a reader assume one commit covers the whole course.  The
#: first Forward attempt died on the outer loop's first live Slow call
#: (``harness_view={}`` where ``core.run_stage`` reads ``.instruction``), was
#: ruled ``RUN_BLOCKED_NO_VERDICT``, and Forward restarted from unit 0 on the
#: fixed commit rather than resuming -- one ordering, one commit.  Phase S is
#: **not** re-run because it provably never executed the defective line: both
#: of its outer steps found no candidate needing a clause, so its ledger
#: records ``llm_outer = 0``.  K0's formation chain (inner Support -> delayed ->
#: authority gate) does not touch this path either.
CODE_PROVENANCE_ERRATUM = {
    "phase_s_commit": "e33f036457e481bd2e5a1eb04fd240e51d3cba00",
    "phase_t_commit": "the fix commit; recorded per artifact in code_state",
    "diff_between_them": (
        "Outer Slow harness_view/source provenance; then a per-arm-per-outer-"
        "step independent backend/core capped at 2 physical requests billed to "
        "llm_outer; then scientific STOP_TRANSPORT/403 as RunFault so an "
        "identity-only quota failure cannot pass the instrument gate"),
    "why_phase_s_is_not_re_run": (
        "Phase S never reached the changed path: ledgers.llm_outer == 0 and "
        "both outer steps recorded slow_calls == 0"),
    "why_forward_did_not_resume": (
        "one ordering runs on one commit; resuming a blocked attempt onto the "
        "fixed one would put two code versions in one curve"),
    "blocked_attempts": (
        {
            "run_root": ".hec1_runs/forward_v11_attempt1_blocked",
            "verdict": "RUN_BLOCKED_NO_VERDICT",
            "units_completed_before_the_fault": 4,
            "llm_spent": 35,
            "counted_as": "instrument overhead; enters no curve",
        },
        {
            "run_label_prefix": "v11fix_",
            "verdict": "RUN_BLOCKED_NO_VERDICT__TRANSPORT_QUOTA",
            "llm_spent": 0,
            "why": "HTTP 403 insufficient_quota became identity UnitFault and "
                   "the eight instrument checks still passed",
            "counted_as": "instrument overhead; enters no curve; do not resume",
        },
        {
            "run_label_prefix": "v11live_",
            "verdict": "RUN_BLOCKED_NO_VERDICT__OUTER_BACKEND_BUDGET_LEAK",
            "units_completed_before_the_fault": 5,
            "llm_spent": 39,
            "why": "outer Slow reused the inner cell backend already spent at 5",
            "counted_as": "instrument overhead; new Forward envelope restarts "
                          "at 500; do not resume",
        },
    ),
}
STAGE = "HEC1_CONTRACT"
DATA_VERSION = v1.DATA_VERSION

SUPERSEDES_NOTHING = (
    "v1, v2 and v3 are imported and still govern the geometry, the risk gate, "
    "the manifest assembly, the refusal ranking and the two-revision bound; no "
    "earlier artifact is rewritten and no earlier reading is reinterpreted"
)

# ---------------------------------------------------------------------------
# ratification
# ---------------------------------------------------------------------------

#: Every field the mainline filled from the handoff brief's default table.  Each
#: entry names the field, the value taken, and the clause it came from.  sol
#: confirmed all thirteen on 2026-09-03; the list is kept as the record of what
#: was confirmed rather than collapsed into a boolean.
CONFIRMED_BY_SOL: tuple[dict[str, Any], ...] = (
    {"field": "outer_loop.period_k_units", "default": 5,
     "authority": "MAINLINE_PLAN 8-7a"},
    {"field": "unit.inner_loop_immediate_slow", "default": "closed",
     "authority": "MAINLINE_PLAN 8-7b"},
    {"field": "budget.replay_fits_share_of_course_fits", "default": 1.0,
     "authority": "sol v1.1 ruling 2 (2026-09-03 13:xx): 100% of the online "
                  "arm's own projected course fits, per arm; was 0.25 under "
                  "MAINLINE_PLAN 8-7c, which the v1 Forward shakedown ran under"},
    {"field": "scope_tool.threshold_rule",
     "default": "widest feasible frozen bin edge; ties take the coarser box",
     "authority": "MAINLINE_PLAN 8-8a"},
    {"field": "scope_tool.scopefit_only_control", "default": "shadow",
     "authority": "MAINLINE_PLAN 8-8b"},
    {"field": "budget.llm_per_unit_arm", "default": 5,
     "authority": "CONTRACT_SKELETON 7; makes Forward <= 500 reachable"},
    {"field": "unit.evaluation_face_offset", "default": 144,
     "authority": "CONTRACT_SKELETON 5; scored only, never fed back"},
    {"field": "course.block_200_239_cut", "default": "A=20 / B=19",
     "authority": "p4ac; a 20/20 cell does not form from 39 series"},
    {"field": "course.target_held_in_origins", "default": 7,
     "authority": "p4ac; p4u's 5 plus 1176 and 1416, both earlier than "
                  "held-in and not time-adjacent to held-out"},
    {"field": "arms.frozen_arm_name_when_k0_empty", "default": "A3-frozen",
     "authority": "MAINLINE_PLAN 4.2 mainline note"},
    {"field": "arms.k0_empty_uses_existing_audited_cards", "default": False,
     "authority": "CONTRACT_SKELETON 4; strictly empty"},
    {"field": "phase_f.deployment", "default": "Fast-only, 0 LLM mechanical recall",
     "authority": "CONTRACT_SKELETON 10"},
    {"field": "best_safe_global.menu",
     "default": "frozen single operators plus the "
                "period_median_complete->outlier_* family",
     "authority": "CONTRACT_SKELETON 7"},
)

#: sol's six rulings of 2026-09-03, written where the code can be checked
#: against them rather than left in a chat log.
SOL_RULINGS: tuple[dict[str, Any], ...] = (
    {"ruling": "all thirteen mainline defaults confirmed",
     "enforced_by": "CONFIRMED_BY_SOL and assert_frozen"},
    {"ruling": (
        "the 20/19 cut is confirmed, but every denominator must take the "
        "actual served count at run time; a hard-coded 20 anywhere in the "
        "denominator path means the cut becomes 19/19 instead"),
     "enforced_by": "SERVED_DENOMINATOR and assert_frozen's scan"},
    {"ruling": (
        "Phase F's 0-LLM recall is confirmed, but runs only on a supported "
        "verdict and only after a separate human seal release"),
     "enforced_by": "PHASE_F and assert_launchable('phase_f')"},
    {"ruling": (
        "the Runtime calibrates the threshold and records "
        "LLM_THRESHOLD_IGNORED when Slow returns a number"),
     "enforced_by": "scope_threshold_tool.clause_from_slow"},
    {"ruling": (
        "census grouping by program is approved, but the key must carry the "
        "full operator sequence, its order and its parameters; the behaviour "
        "fingerprint may only fold aliases"),
     "enforced_by": "outer_loop._program_signature and CENSUS_KEY"},
    {"ruling": (
        "an unimplemented live arm loop is not a completion state; it must be "
        "finished and pass a 0-LLM end-to-end test before the freeze"),
     "enforced_by": "run_hec1.run_course and the end-to-end test"},
)

#: sol's final rulings of 2026-09-03 (after Fable's review), applied before the
#: first LLM call.  They amend confirmed defaults and pre-registered text; every
#: amendment is listed here rather than edited silently into the fields above.
SOL_FINAL_RULINGS: tuple[dict[str, Any], ...] = (
    {"ruling": (
        "replay fits: each online arm's cap is 100% of that arm's own projected "
        "course Consumer fits; no recent-window truncation (it would add a time-"
        "selection mechanism); replay may only screen; replay fits reported "
        "apart; the v1 Forward that ran under 0.25 is FORWARD_SHAKEDOWN"),
     "enforced_by": "REPLAY_FITS_SHARE, REPLAY_SHARE_RECORD, run_hec1.run_course"},
    {"ruling": (
        "statistics: the one-sided sign test at alpha .05 is deleted as a pass "
        "gate; HEC-1 is a descriptive development mechanism curve with the "
        "qualitative criteria in STATISTICS; exact binomial probability is "
        "reported with its floor of 1/16 stated; cohort bootstrap is description "
        "only; orderings are not independent seeds; the terminal difference is "
        "the primary endpoint, AUC / midpoint secondary"),
     "enforced_by": "STATISTICS, PREREGISTERED, audit_hec1_readout"},
    {"ruling": (
        "WAITING consumes a verification attempt: it spent a Consumer evaluation; "
        "the Draft stays WAITING under the floor and is archived at the cap"),
     "enforced_by": "LIFECYCLE['waiting'], run_hec1.run_unit_arm"},
    {"ruling": (
        "K0: a non-empty K0 must carry its store and runtime bundle and the "
        "runner refuses to fall back to h0; audit_hec1_k0_freeze runs before "
        "Forward; deployed_via reads both the candidate's source and whether "
        "the program was in the arm's Active set at unit start; a P4 pass with "
        "no online_loop event is recorded as lost_activation and never activates"),
     "enforced_by": "run_hec1.phase_s_k0 / run_course / run_unit_arm, "
                    "audit_hec1_k0_freeze, audit_hec1_instrument"},
    {"ruling": (
        "Skill taxonomy: General (procedural, h0) / Target-local Specific "
        "(Program x Scope x Task-Consumer x Evidence) / Source-derived (a "
        "Specific Skill in its cross-stage evidence role) / Shared Capability "
        "(>= 2 independent domains); General and Specific never promote into "
        "each other"),
     "enforced_by": "SKILL_TAXONOMY"},
    {"ruling": (
        "Best-Safe-Global is named 'offline in-budget comparator' only: it "
        "selects on the +144 Outcome after the fact, is not deployable and not "
        "an oracle; no prequential variant in HEC-1"),
     "enforced_by": "READOUTS['best_safe_global'], audit_hec1_best_safe_global"},
    {"ruling": (
        "Phase F is '0-LLM mechanical deployment of the frozen Skill policy', "
        "not the training-time Fast Agent reasoning again; only "
        "HEC1_EVOLUTION_SUPPORTED may apply for the seal; zero coverage reads "
        "NOT_REACHED_ON_HELD_OUT and is never a safety success"),
     "enforced_by": "PHASE_F, assert_launchable('phase_f')"},
    {"ruling": (
        "paper: Track A (curve + survival chain) is the only main hypothesis; "
        "Track B freezes the failure vocabulary only; if P1 holds and P2 fails "
        "the claim narrows to ADD / recall-driven accumulation and never to "
        "full Scope-revision evolution"),
     "enforced_by": "VERDICTS['HEC1_P1_ONLY__RECALL_ACCUMULATION']"},
)

#: What the final rulings changed relative to the 2026-09-03 01:xx freeze, so a
#: reader of the frozen receipt can see the amendment rather than infer it.
AMENDMENTS_BEFORE_LAUNCH = (
    "v1.1: the v1 Phase S (empty K0) and the v1 Forward are superseded / "
    "FORWARD_SHAKEDOWN; every scientific ordering runs from one commit with "
    "clean runner files, recorded per artifact as code_state and asserted "
    "equal across orderings by the readout",
    "budget.replay_fits_share_of_course_fits: 0.25 of all LLM arms' fits -> "
    "1.0 of the online arm's own projected course fits, per arm",
    "outer_loop.MIN_POSITIVE_UNITS_FOR_ADD: 2 -> 1 (sol v1.1 ruling 1; landed "
    "by the execution line): one positive unit opens a non-deployable Draft "
    "that must clear Support and delayed on a later independent unit",
    "census key: task_consumer_key x full typed Program x root Scope (sol v1.1 "
    "ruling 6; landed by the execution line)",
    "Phase F: additionally requires a NON-EMPTY K0 (sol v1.1 ruling 5): an "
    "empty-K0 course can support Target-local self-evolution as component "
    "evidence but never the full A5 claim, and A3 does not stand in for A5",
    "STATISTICS: sign test removed as a gate; descriptive criteria written",
    "VERDICTS: HEC1_P1_ONLY__RECALL_ACCUMULATION added",
    "LIFECYCLE.waiting: a coverage-only face consumes a verification attempt",
    "READOUTS.best_safe_global: renamed offline in-budget comparator",
    "PHASE_F: named mechanical deployment; NOT_REACHED_ON_HELD_OUT vocabulary",
    "SKILL_TAXONOMY and NAMING blocks added",
    "all amendments applied before any LLM call was made under this contract",
)

#: The four envelopes the user released on 2026-09-03.
USER_RELEASES: tuple[dict[str, Any], ...] = (
    {"field": "budget.phase_s_llm_cap", "released": 120},
    {"field": "budget.phase_t_forward_llm_cap", "released": 500},
    {"field": "budget.phase_t_reverse_llm_cap", "released": 500},
    {"field": "budget.phase_t_interleaved_llm_cap", "released": 500},
    {"field": "budget.total_llm_hard_cap", "released": 1620},
    {"field": "budget.best_safe_global_fits", "released": 1820},
)

#: Still the user's, and always will be.
PENDING_USER: tuple[dict[str, Any], ...] = (
    {"field": "phase_f.seal_opening", "proposed": "after the course readout",
     "note": "a human release, every time; never inferred from a verdict"},
)

RATIFICATION = {
    "sol_confirmed": True,
    "sol_confirmed_at": "2026-09-03",
    "rulings": [dict(row) for row in SOL_RULINGS],
    "final_rulings": [dict(row) for row in SOL_FINAL_RULINGS],
    "amendments_before_launch": list(AMENDMENTS_BEFORE_LAUNCH),
    "user_released": tuple(row["field"] for row in USER_RELEASES),
    "frozen_at": "2026-09-03",
    "autonomy_envelope": (
        "Phase S -> K0 freeze -> Forward -> Reverse -> Interleaved -> one "
        "frozen 0-LLM course readout -> stop before Phase F is opened"
    ),
    "what_never_becomes_autonomous": [
        "opening the Phase F seal: a human release, every time",
        "the verdict: the mainline writes it and sol confirms it",
    ],
    "why_a_drift_check_is_still_not_an_authorisation": (
        "assert_frozen re-derives mechanical drift and says nothing about who "
        "adjudicated the values; assert_launchable is the one that reads the "
        "ratification, and Phase F stays refused inside it"
    ),
}

#: sol's ruling 6, as the condition the runner is checked against.
AUTO_CONTINUE_CONDITIONS = (
    "the contract is frozen and the live loop passed its 0-LLM end-to-end test",
    "Phase S completed all 13 units",
    "the K0 mechanical audit passed; an empty K0 is a legal freeze too",
    "the mechanical instrument checks passed for the finished ordering, "
    "including the transport-failure backstop",
    "whether to continue reads instrument health only, never the effect's sign",
    "a fault may only be repaired as an instrument and resumed from a "
    "checkpoint; no scientific re-throw",
    "no change to the contract, thresholds, features, menu, course or prompts",
    "the readout runs exactly once and then stops, reading no Phase F held-out",
)

#: sol's ruling 2.  The served count is read from the roster at run time; this
#: constant exists so a test can assert the *absence* of a hard-coded 20 rather
#: than trusting that none was written.
SERVED_DENOMINATOR = {
    "rule": "len(roster eval rows) at the unit being run",
    "never": 20,
    "why": (
        "readable[200:239] holds 39 series, so its faces are 20 and 19.  A "
        "denominator fixed at 20 would silently understate face B's coverage "
        "and every fraction derived from it; if one is ever found the cut "
        "becomes 19/19 instead"
    ),
    "checked_by": "test_hec1_wiring and assert_frozen's source scan",
}

#: sol's ruling 5.
CENSUS_KEY = {
    "group_key": "task_consumer_key x full program signature",
    "signature_carries": ["every operator", "their order", "their parameters"],
    "behaviour_fingerprint_may_only": "fold aliases",
    "why_not_the_fingerprint_as_key": (
        "per-series gains differ from unit to unit by construction, so keying "
        "on them gives every program one group per unit and no accumulation "
        "could ever be observed"
    ),
}

# ---------------------------------------------------------------------------
# 1. identity  /  2. data and course
# ---------------------------------------------------------------------------

INHERITS = {
    "v1": "geometry, arms, per-arm budget vector, exposure ledger",
    "v2": "RISK_GAP route, scoped ADD, delayed admission",
    "v3": ("Runtime manifest assembly, fewest-exclusions refusal ranking, "
           "restricted Draft with at most two revisions"),
    "rewrites": [],
}

EVIDENCE_GRADE = {
    "phase_s": "DEVELOPMENT",
    "phase_t": "DEVELOPMENT",
    "curve_is_named": "development mechanism curve",
    "phase_f_f1": "FRESH (same family, new instance; Outcome unseen)",
    "phase_f_f2": "FRESH (new family; expected to carry safety readings only)",
}

PHASE_S_BLOCKS = ("[160:200]", "[200:239]")
PHASE_T_BLOCKS = ("[0:40]", "[40:80]", "[80:120]", "[120:160]")
ORDERINGS = ("forward", "reverse", "interleaved")

#: D1's reading, carried as a registered gap rather than as a repair.  There is
#: no deployment-time quantity in the twelve-name binnable vocabulary that
#: separates the series that get harmed on entry, so ``AGENTS`` §6 says record
#: the gap and lean on held-in feedback and abstention -- not add a feature.
OBSERVATION_GAP = {
    "source": "p4ab_routing_harm_diagnostic",
    "verdict": "NO_OUTCOME_FREE_SEPARATOR",
    "what_does_not_separate": [
        "raw/program prediction divergence (AUC 0.65, CI lower 0.52)",
        "distance to the historical safe-evidence region (0.51)",
        "program behaviour beyond historical coverage (0.60)",
        "context modification volume (0.45 / 0.40)",
    ],
    "consequence_for_hec1": (
        "no new Risk face and no new observation feature; new-entrant risk is "
        "carried by held-in probes and by evidence-bounded Scope as a "
        "candidate form only"
    ),
    "deferred_to": "HEC-3 observation face, frozen prospectively on a new cohort",
}

#: The composition self-check that came back empty, stated as a limit of the
#: course rather than discovered in the report.
COURSE_LIMITS = {
    "pattern_sparse_units": 0,
    "reading": (
        "every KDD unit has at least six series clearing z_peak >= 3 (median "
        "13.5), so the frozen initialiser's predicate barely filters anyone and "
        "all narrowing comes from Slow's clause"
    ),
    "hec1_does_not_test": (
        "silence when the pattern is absent; Epilepsy2 and the S2a clean cell "
        "already cover it"
    ),
    "statistical_power": (
        "26 units per ordering, cohort as the unit of analysis; reported as it "
        "is rather than as if 40 had been run"
    ),
}


def _supply() -> Mapping[str, Any]:
    return json.loads(SUPPLY.read_text(encoding="utf-8"))


def _units(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"block": str(row["block"]), "span": [int(row["span"][0]),
                                             int(row["span"][1])],
         "origin": int(row["origin"]),
         "exposure": list(row.get("exposure") or ())}
        for row in rows
    ]


def phase_s_units() -> list[dict[str, Any]]:
    """13 units on the two Source blocks, under the <=3816 caliber."""
    return _units(_supply()["proposals"]["phase_s_le_3816"])


def phase_t_units() -> list[dict[str, Any]]:
    """26 units on the four Phase-T blocks, under the <=3816 caliber.

    The conservative caliber is taken because it clears the 22-unit floor on its
    own (26 >= 22).  All-usable would give 30; it is not taken, and the reason
    is disclosure rather than power: origins past 3816 sit closer in time to the
    held-out block, and the course does not need them.
    """
    return _units(_supply()["proposals"]["phase_t_le_3816"])


def ordering(name: str) -> list[dict[str, Any]]:
    """One of the three frozen unit sequences, by name."""
    if name not in ORDERINGS:
        raise ValueError("ordering must be one of %s" % (ORDERINGS,))
    return _units(_supply()["proposals"]["orderings"][name])


def course() -> dict[str, Any]:
    supply = _supply()
    proposals = supply["proposals"]
    return {
        "unit_definition": "(cohort block, origin)",
        "domain": "KDD natural-gap only",
        "why_kdd_only": (
            "injecting electricity/traffic units would change the data domain, "
            "the defect mechanism and the adapter at once, and a change in the "
            "curve could not be attributed to any of them"
        ),
        "blocks": {
            "phase_s": list(PHASE_S_BLOCKS),
            "phase_t": list(PHASE_T_BLOCKS),
            "disjoint": True,
            "why_disjoint": (
                "a series K0 learned on must not reappear in the Target, or "
                "the accumulation contrast would be measuring memorisation"
            ),
        },
        "block_200_239_cut": {
            "n_series": supply["cut_200_239"]["n_series"],
            "face_a": supply["cut_200_239"]["canonical_cut"]["face_a_n"],
            "face_b": supply["cut_200_239"]["canonical_cut"]["face_b_n"],
            "rule": supply["cut_200_239"]["canonical_cut"]["rule"],
            "equal_face_alternative_not_taken": (
                supply["cut_200_239"]["equal_face_cut"] or {}).get("leftover"),
        },
        "phase_s_units": proposals["n_phase_s_le_3816"],
        "phase_t_units": proposals["n_phase_t_le_3816"],
        "phase_t_all_usable_not_taken": proposals["n_phase_t_all"],
        "orderings": {
            name: proposals["orderings"]["rules"][name] for name in ORDERINGS},
        "first_ordering_is_instrument_health_only": (
            "Forward is read for instrument completeness; whether Reverse and "
            "Interleaved run may not depend on whether Forward's effect was "
            "positive"
        ),
        "spent_dev_units": (
            "labelled DEVELOPMENT; no arm may read their historical Outcome, "
            "only deployment-visible Context and feedback from this course"
        ),
        "exposure": {
            "held_out_intersection_empty": supply["exposure_cross_check"][
                "held_out_intersection_empty"],
            "windows_checked": [0, 48, 144, 240],
            "p4t_verdict": supply["exposure_cross_check"]["p4t_verdict"],
        },
        "composition": {
            "repeat_family_jaccard_median": supply["composition_check"][
                "repeat_pattern_family"]["jaccard"]["median"],
            "unique_binned_vectors_median": supply["composition_check"][
                "within_family_heterogeneity"]["n_unique_binned_vectors"][
                    "median"],
            "pattern_sparse_units": supply["composition_check"][
                "pattern_sparse_units"]["n"],
        },
        "limits": COURSE_LIMITS,
        "supply_audit": SUPPLY.name,
    }


# ---------------------------------------------------------------------------
# 3. Consumer, program space, risk, gate authority
# ---------------------------------------------------------------------------

CONSUMER = {
    "id": "fixed:pooled-ridge-a1",
    "context": v1.CONTEXT,
    "horizon": v1.HORIZON,
    "anchors": "frozen [312 ... 852], unchanged",
    "changed_from_p4u": False,
}

PROGRAM_SPACE = {
    "operators": "the frozen P1 Common DSL, single operators",
    "compositions": "length <= 2",
    "why_compositions": (
        "AGENTS 5.1 measured it: single operators are all unstable at origin "
        "2856, while period_median_complete -> outlier_* is positive on both "
        "faces with a +0.21~+0.28 order effect.  The reason is the reading, "
        "not a wish for a larger space"
    ),
    "window_verifier_max_modified_fraction": forecast_p4.MAX_MODIFIED_FRACTION,
    "operators_added": 0,
    "counting_discipline": (
        "census and every program count deduplicate by per-series gain vector "
        "first; 396 enumerated programs contain many aliases on gappy data"
    ),
}

RISK = {
    "policy": "bounded_risk_v1",
    "material": admission_policy.MATERIAL_THRESHOLD,
    "max_harmed_fraction": bounded.BOUNDED_MAX_HARMED_FRACTION,
    "max_single_series_harm": bounded.BOUNDED_MAX_SINGLE_SERIES_HARM,
    "min_treated": distance.MIN_TREATED,
    "thresholds_changed": 0,
    "where_the_gate_sits": (
        "held-in delayed, deciding execution rights.  What is worth keeping and "
        "revising is a separate question decided by the memory admission rule, "
        "as AGENTS 4 requires"
    ),
    "on_failure": "the three-state machine decides; the gate itself does not move",
}

SCOPE_CLASS = {
    "kind": "serving_series_predicate",
    "clauses": "at most 2",
    "vocabulary": list(threshold_tool.VOCABULARY),
    "vocabulary_size": len(threshold_tool.VOCABULARY),
    "why_twelve": (
        "the public card has 21 keys and X1 adds 6 structure descriptors, but "
        "only these 12 are numeric observables with frozen bins; a threshold "
        "cannot be calibrated against a name that has no bins"
    ),
    "evidence_bounded_two_sided": "a candidate form, never a default",
    "initialiser": "evaluation.main_protocol_p4.scope_initializer, frozen table",
    "features_added": 0,
}

#: The one specification conflict D1 turned up, resolved rather than recorded.
GATE_AUTHORITY = {
    "problem": (
        "the Source-v3 artifact's round 2856 carries delayed_gate.passes=False "
        "on the coverage floor and an online_loop delayed_event of 'approved' "
        "at the same time: two gates with different calibres"
    ),
    "authority": "the P4 _gate, coverage floor included",
    "online_loop_delayed_approval": "recorded, never sufficient on its own",
    "runner_must": [
        "never call activate_approved on the online_loop event alone",
        "record gate_disagreement = {p4_gate, online_loop_event, resolved_by}",
        "assert the Active set only ever grows through the authoritative gate",
    ],
    "acceptance": (
        "a synthetic unit where online_loop approves and the P4 gate refuses "
        "leaves the Active set unchanged and records the disagreement"
    ),
}

# ---------------------------------------------------------------------------
# 4. arms
# ---------------------------------------------------------------------------

ARMS_FULL_K0 = {
    "Static": {"start": "no Harness", "write_back": False, "outer": False,
               "llm": False},
    "A5-frozen": {"start": "K0", "write_back": False, "outer": False,
                  "llm": True,
                  "note": "reset to K0 after every unit; adapts inside a unit "
                          "and takes nothing away"},
    "A5-online": {"start": "K0", "write_back": True, "outer": True,
                  "llm": True},
    "A3-online": {"start": "h0", "write_back": True, "outer": True,
                  "llm": True},
}

ARMS_EMPTY_K0 = {
    "Static": ARMS_FULL_K0["Static"],
    "A3-frozen": {"start": "h0", "write_back": False, "outer": False,
                  "llm": True,
                  "note": "the frozen contrast in the empty-K0 shape; it is "
                          "equivalent to no other arm and is the only control "
                          "criterion 1 has"},
    "A3-online": ARMS_FULL_K0["A3-online"],
}

ARMS = {
    "full_k0": ARMS_FULL_K0,
    "empty_k0": ARMS_EMPTY_K0,
    "when_k0_is_empty": (
        "A5-online would be identical to A3-online, so it is not run: paying "
        "for an equivalent arm would buy no contrast.  Criterion 3 is not "
        "scored, and criteria 1, 2 and 4 are"
    ),
    "k0_empty_uses_existing_audited_cards": False,
    "identical_across_arms": [
        "data", "feedback faces", "probe budget", "LLM budget",
        "the initialiser", "the risk gate", "the lifecycle",
    ],
    "a5_budget_exception": None,
    "controls_0_llm_offline": [
        "Best-Safe-Global baseline and advantage",
        "ScopeFit-only shadow",
    ],
}

# ---------------------------------------------------------------------------
# 5. the unit protocol and the lifecycle
# ---------------------------------------------------------------------------

FACES = {
    "support_a": "origin o",
    "delayed": "o + 48, the gate",
    "evaluation": "o + 144, scored only",
}

EVALUATION_FACE = {
    "offset": 144,
    "enters_bank": False,
    "enters_prompt": False,
    "flows_back": False,
    "why_legal": (
        "the +144 window overlaps the next origin's (o+240) Support context, "
        "and that overlap is deployment-visible data.  Its Outcome goes only to "
        "the scoring ledger, so the curve is read on a face no arm learned from"
    ),
}

UNIT_PROTOCOL = {
    "context": (
        "the face-A served series of the unit being run, deployment-visible "
        "features; the count is read from the roster and is 20 on every block "
        "except readable[200:239] face B, which holds 19"
    ),
    "served_denominator": SERVED_DENOMINATOR,
    "retrieval": (
        "Skills in K whose Scope matches are supplied under "
        "requires_target_support; the Fast-only re-encounter path is a "
        "separate route and is recorded separately"
    ),
    "fast": "at most 2 proposals, programs of at most 2 steps",
    "scope": "Runtime initialises serving_scope from defect presence",
    "probe": "Support-A, per the v1 per-arm budget",
    "admission": "bounded_risk_v1",
    "then": "delayed at o+48 -> Active, or one of the three restricted states",
    "scoring": "evaluation face at o+144, into the scoring ledger only",
    "inner_loop_immediate_slow": False,
    "why_inner_slow_is_closed": (
        "every Slow call belongs to the outer loop, so an edit is never both "
        "proposed and judged on the unit that triggered it"
    ),
}

OUTER_LOOP = {
    **outer_loop.declared_rules(),
    "period_k_units": 5,
    "arms": ["A5-online", "A3-online"],
    "steps_per_ordering": None,  # filled by budget_arithmetic()
}

LIFECYCLE = {
    "faces": FACES,
    "states": [drafts.WAITING, drafts.REVISABLE, drafts.FLAGGED],
    "priority": "FLAGGED > REVISABLE > WAITING",
    "max_revisions": drafts.MAX_REVISIONS,
    "max_verification_attempts": drafts.MAX_VERIFICATION_ATTEMPTS,
    "waiting": {
        "when": "only the coverage floor failed",
        "action": ("no revision; the Draft stays WAITING and is re-read when a "
                   "later window resolves at least MIN_TREATED series"),
        "consumes_verification_attempt": True,
        "why_it_consumes": (
            "sol final ruling §3: the face spent a real Consumer evaluation "
            "and a slot of the verification budget; a Draft cannot wait "
            "without bound and is archived at the cap"
        ),
        "closes_as": drafts.CLOSE_REASONS[drafts.WAITING],
        "why_not_a_skill_failure": (
            "a single window's treated count measures how prevalent the "
            "pattern is, not how good the Skill is"
        ),
    },
    "revisable": {
        "when": "a tail or harmed-fraction line failed and every harmed series "
                "had just entered the predicate",
        "action": "Slow may add one clause through the W2 tool chain",
    },
    "flagged": {
        "when": "the negative contribution is dominated by continuing members",
        "action": "narrowing forbidden; one re-verification unchanged",
        "closes_as": drafts.CLOSE_REASONS[drafts.FLAGGED],
        "then": "registered with the census as an Observation / Program drift "
                "signal",
    },
    "drafts_are_never_deleted": "every state keeps the evidence",
    "prior_symmetry": {
        "risk_cards": "induced over the same bins as positive cards",
        "authority": "restricts_probe -- last in the probe order, never a hard "
                     "ban",
        "revoked_when": "the same program earns a POSITIVE in a matching context",
        "context_side_coverage": "disclosed before the run, never used to filter",
    },
}

SCOPE_TOOL_CHAIN = threshold_tool.declared_rules()

# ---------------------------------------------------------------------------
# 7. budget
# ---------------------------------------------------------------------------

PER_UNIT_ARM_BUDGET = {
    **{key: value for key, value in v1.PER_ARM_BUDGET.items()
       if key != "llm_calls"},
    "llm_calls": 5,
    "why_five_not_six": (
        "v1's vector allows 6; 5 is what makes a 26-unit Forward ordering fit "
        "under 500 with the outer loop included.  It is a reduction, taken "
        "identically by every arm"
    ),
    "identical_for_every_arm": True,
}

OUTER_LLM_PER_STEP = 2
#: sol v1.1 ruling 2: each online arm's replay cap is 100% of that arm's OWN
#: projected course Consumer fits; every applicable processed cell is replayed
#: (no recency window -- it would add a time-selection mechanism); a cell
#: below the coverage floor is NOT_APPLICABLE.  Under the v1 value (0.25 of all
#: LLM arms' fits) a screen that re-scores every processed cell at 3 fits each
#: (15k fits at step k) exhausted the allowance after outer step 2, so the
#: second half of every ordering could open no Draft.
REPLAY_FITS_SHARE = 1.0

REPLAY_SHARE_RECORD = {
    "v1_forward_shakedown_ran_under": 0.25,
    "v1_formula": "share of ALL LLM arms' projected fits, one remainder",
    "v1_1_scientific_orderings_run_under": 1.0,
    "v1_1_formula": "share of the online arm's OWN projected fits, per arm",
    "why_not_mixed": (
        "the v1 Forward is FORWARD_SHAKEDOWN by sol's ruling and enters no "
        "curve; every scientific ordering runs under one commit and one value"),
}

#: The released envelopes, per phase.  A runner reads these; it does not hold its
#: own copy of a number the user approved.
LLM_CAPS = {
    "phase_s": 120,
    "phase_t_forward": 500,
    "phase_t_reverse": 500,
    "phase_t_interleaved": 500,
}
TOTAL_LLM_HARD_CAP = 1620
BEST_SAFE_GLOBAL_FIT_CAP = 1820


def budget_arithmetic() -> dict[str, Any]:
    """Where the 500 goes, computed rather than asserted."""
    units = len(phase_t_units())
    per_unit = int(PER_UNIT_ARM_BUDGET["llm_calls"])
    steps = units // int(OUTER_LOOP["period_k_units"])
    full_llm_arms = sum(1 for arm in ARMS_FULL_K0.values() if arm["llm"])
    full_online = sum(1 for arm in ARMS_FULL_K0.values() if arm["outer"])
    empty_llm_arms = sum(1 for arm in ARMS_EMPTY_K0.values() if arm["llm"])
    empty_online = sum(1 for arm in ARMS_EMPTY_K0.values() if arm["outer"])
    phase_s = len(phase_s_units())
    phase_s_steps = phase_s // int(OUTER_LOOP["period_k_units"])
    return {
        "phase_t_units": units,
        "llm_per_unit_arm": per_unit,
        "outer_steps_per_ordering": steps,
        "outer_llm_per_step": OUTER_LLM_PER_STEP,
        "forward_full_k0": (units * full_llm_arms * per_unit
                           + full_online * steps * OUTER_LLM_PER_STEP),
        "forward_empty_k0": (units * empty_llm_arms * per_unit
                            + empty_online * steps * OUTER_LLM_PER_STEP),
        "forward_hard_cap": 500,
        "phase_s_units": phase_s,
        "phase_s_outer_steps": phase_s_steps,
        "phase_s_estimate": (phase_s * per_unit
                            + phase_s_steps * OUTER_LLM_PER_STEP),
        "phase_s_cap": LLM_CAPS["phase_s"],
        "replay_fits_share_of_course_fits": REPLAY_FITS_SHARE,
        "shadow_fits": "billed separately, 0 LLM",
        "llm_caps": dict(LLM_CAPS),
        "total_llm_hard_cap": TOTAL_LLM_HARD_CAP,
        "best_safe_global_fit_cap": BEST_SAFE_GLOBAL_FIT_CAP,
        "sum_of_released_envelopes": sum(LLM_CAPS.values()),
        "reverse_and_interleaved": (
            "each released at 500, entered automatically when the "
            "mechanical instrument checks pass and never on the effect's sign"
        ),
    }


# ---------------------------------------------------------------------------
# 8. readouts, statistics, pre-registration
# ---------------------------------------------------------------------------

READOUTS = {
    "primary_figure": (
        "cumulative gain relative to Static on the evaluation face against unit "
        "index, four arms x three orderings; A5-online minus A5-frozen plotted "
        "separately"
    ),
    "secondary_figure": (
        "cumulative harm events (unit-level harmed fraction > 0.20 or single "
        "series harm > 0.30); online must not exceed frozen"
    ),
    "best_safe_global": (
        "OFFLINE IN-BUDGET COMPARATOR (sol final ruling §6): per unit, the best "
        "global program on the frozen menu that clears the risk budget, chosen "
        "on the +144 evaluation Outcome after the fact, identity when none "
        "does.  Not a deployable baseline (it reads the Outcome it is scored "
        "on) and not an oracle (a Scoped policy can beat it).  Arms are "
        "reported as an advantage over it.  No prequential variant in HEC-1"
    ),
    "true_oracle": (
        "must cover the Scoped policy (per-UID selection); offline upper bound "
        "only, behind the oracle wall, entering no arm"
    ),
    "lifecycle": [
        "cards minted", "revision success rate", "revocations",
        "survival rate", "re-encounter gain",
        "coverage (treated/served, cumulative and per window, listed apart)",
    ],
    "three_way_attribution": [
        "recall of a card frozen did not have",
        "frozen re-recommends and is refused or harmed where online has "
        "already narrowed or revoked",
        "probe slots released (the SUPPLY_STARVATION mechanism)",
    ],
    "h1_h2_h3": (
        "mechanically recorded at every verification face: harmed share among "
        "new entrants, sign-flip rate among continuing members, share of "
        "leavers that left because a feature exited the predicate"
    ),
    "slow_vs_scopefit": "agreement rate on (feature, direction) and both "
                        "re-encounter readings",
    "fast_raw_decision": ["PROPOSED", "ABSTAINED_WITH_REASON", "EMPTY_OUTPUT",
                          "MALFORMED"],
    "cost": ["llm_fast", "llm_outer", "replay_fits", "shadow_fits",
             "course_fits", "baseline_fits", "wall", "cache hit rate"],
}

STATISTICS = {
    "status": "DESCRIPTIVE development mechanism curve (sol final ruling §2)",
    "unit_of_analysis": "cohort",
    "n_cohorts": len(PHASE_T_BLOCKS),
    "why_not_origin": (
        "anchors are frozen and origin-invariant past 900, so two origins of "
        "one cohort are a time re-encounter of one training condition, not two "
        "independent samples"
    ),
    "no_significance_gate": (
        "a one-sided exact sign test on 4 cohorts has a floor of 1/16 = 0.0625 "
        "even at 4/4, so alpha 0.05 can never be reached; it is not a pass "
        "criterion.  The exact binomial probability is reported with that "
        "floor stated"
    ),
    "primary_endpoint": (
        "terminal cumulative difference D_o = sum over units of "
        "(online - frozen) evaluation-face aggregate_gain, per ordering"
    ),
    "cohort_endpoint": (
        "d_c = mean over the cohort's units of (online - frozen), averaged "
        "over the three orderings; four numbers"
    ),
    "secondary_endpoints": ["AUC of the cumulative-difference curve",
                            "difference at the midpoint unit"],
    "monotonicity": "not required; local dips are allowed",
    "qualitative_criteria_for_P1": [
        "D_o > 0 in at least 2 of 3 orderings",
        "d_c > 0 in at least 3 of 4 cohorts",
        "harm events online <= frozen in every ordering",
    ],
    "reported_only": [
        "exact binomial probability of the cohort sign pattern (floor 0.0625)",
        "cohort bootstrap percentile interval, n = 4, as uncertainty description",
        "per-unit sign counts, as correlated descriptive counts",
    ],
    "orderings_are_not_seeds": (
        "the three orderings share data, cache and units; they are drawn as "
        "three curves, never as a confidence band"
    ),
    "confirmatory_statistics": (
        "deferred to a fresh experiment with at least 8 independent cohorts or "
        "datasets (one-sided Wilcoxon signed-rank, or sign test needing >= 7 of 8)"
    ),
}

#: sol v1.1: a difference has to be **material**, not merely positive.  The
#: material line is per unit, so the course-level line is that line times the
#: number of units that can carry a point -- 0.005 x 23 = 0.115.  A terminal
#: difference of +0.02 over 23 units is 23 readings of nearly nothing, and
#: calling it evolution would be reading noise as a curve.
P1_MATERIAL_TERMINAL_DIFFERENCE = round(
    RISK["material"] * scoreability.SCOREABLE_UNITS, 6)

#: The 0-LLM control sol froze beside the arms.  Same probe budget, same risk
#: gate, no Slow and no memory: it answers "how much of the curve is the search
#: rather than the accumulation".  It is **not** part of the Harness and never
#: supplies a candidate to one.
VALIDATION_SEARCH_BASELINE = {
    "required": True,
    "llm_calls": 0,
    "budget": "the same per-unit probe budget as an arm",
    "risk_gate": "the same bounded_risk_v1 and the same coverage floor",
    "memory": "none; nothing is carried between units",
    "enters_the_harness": False,
    "why": (
        "a per-unit search under the same budget and the same gate is what "
        "separates 'the Harness accumulated something' from 'this many probes "
        "on this menu find something'.  Without it a positive curve cannot be "
        "told apart from the probe budget doing the work"
    ),
    "reported_as": "a comparator column, never an arm and never a Skill source",
}

#: sol v1.1: if Phase S is empty again, diagnose the supply exhaustively -- and
#: do not let the diagnosis become this round's treatment.
PHASE_S_EMPTY_AGAIN = {
    "then": "run an exhaustive 0-LLM supply diagnostic over the frozen menu",
    "answers": (
        "whether any program in the frozen space would have been POSITIVE on "
        "two or more Source units, which separates 'the Harness did not find "
        "it' from 'it is not there'"
    ),
    "must_not": (
        "generate this round's K0.  A card found by exhaustive enumeration was "
        "not formed by the Harness, and using it would make A5's treatment an "
        "artefact of the diagnostic rather than of accumulation"
    ),
    "k0_stays": "empty, recorded A5_TREATMENT_EMPTY",
}

PREREGISTERED = {
    "P1_evolution": {
        "claim": "online minus frozen cumulative safe utility on the evaluation "
                 "face ends **at or above %.3f** in at least 2 of 3 orderings "
                 "and is above 0 in at least 3 of 4 cohorts; harm online does "
                 "not exceed frozen; monotonic growth is not required"
                 % P1_MATERIAL_TERMINAL_DIFFERENCE,
        "material_terminal_difference": P1_MATERIAL_TERMINAL_DIFFERENCE,
        "why_a_material_line": (
            "0.005 per unit x %d scoreable units.  A terminal difference of a "
            "few hundredths over 23 units is 23 readings of nearly nothing; "
            "positive is not the same as material"
            % scoreability.SCOREABLE_UNITS),
        "first_fault_if_not": {
            "recall deploys worse than frozen searching on the spot":
                "Scope too wide -- the memory face",
            "cards minted near zero": "Fast candidate starvation -- the supply face",
            "difference positive but not significant":
                "trigger density too low -- the course face",
        },
    },
    "P2_survival": {
        "claim": "at least one Draft inside Phase T clears delayed after at "
                 "most 2 narrowings, survives, and beats frozen on the same "
                 "unit at least once in a re-encounter",
        "first_fault_if_not": {
            "still breaching the single-series line after narrowing":
                "attribute by the three-state machine first, then choose a face",
            "never reaching delayed": "Support re-probe budget",
        },
    },
    "P3_accumulation": {
        "claim": "scored only when Phase S leaves a survivor; A5-online minus "
                 "A3-online cumulative above 0",
        "first_fault_if_not": {
            "K0 Scope match near zero on the Target":
                "K0 unreachable -- report by stratum, do not change the course",
            "matched but no gain": "cross-cohort generalisation",
        },
    },
    "none_of_these_reads_as": "evolution does not work",
}

VERDICTS = {
    "HEC1_EVOLUTION_SUPPORTED": "P1 and P2 both hold",
    "HEC1_P1_ONLY__RECALL_ACCUMULATION": (
        "P1 holds and P2 fails: the curve is material but no revised Draft "
        "survived a re-encounter.  The claim is **feedback-driven "
        "Skill-library evolution** (equivalently: Skill acquisition "
        "evolution) -- cards formed from held-in feedback, recalled, and "
        "released probe slots.  Three phrasings are forbidden for this "
        "verdict (sol v1.1): **Scope-revision evolution** (no revision "
        "survived), **the complete A5 system** (that needs a non-empty K0 and "
        "P2), and **cross-domain / transfer** anything (Phase S and Phase T "
        "are the same dataset, so accumulation here is within-dataset and "
        "cross-cohort at most).  Does not qualify for the Phase F seal"),
    "P1_only_permitted_phrasings": (
        "feedback-driven Skill-library evolution",
        "Skill acquisition evolution",
    ),
    "P1_only_forbidden_phrasings": (
        "Scope-revision evolution",
        "the complete A5 system",
        "cross-domain transfer",
    ),
    "HEC1_EVOLUTION_NOT_SUPPORTED": "P1 fails, with its first fault named",
    "HEC1_INCONCLUSIVE": (
        "an ordering reached fewer than %d valid paired curve points, or the "
        "three orderings are not all in.  The floor is ceil(0.8 x %d scoreable "
        "units), not of the %d scheduled ones: three units carry no observed "
        "truth in their evaluation horizon and can never contribute a point"
        % (scoreability.MIN_PAIRED_CURVE_POINTS, scoreability.SCOREABLE_UNITS,
           scoreability.SCHEDULED_UNITS)),
    "RUN_BLOCKED_NO_VERDICT": "an instrument fault; never a scientific verdict",
    "h1_h3_annotation": ["CONSISTENT", "MIXED", "NOT_OBSERVED"],
    "P2_definition": (
        "at least one Draft in Phase T received a Scope revision, then passed "
        "the authoritative gate on a NEW unit (Active), and on a later unit was "
        "deployed by the online arm with (online - frozen) > 0 on that unit"
    ),
    "when": "at the end of the course, never at the first fault",
    "track_b_vocabulary_frozen_now": [
        "SUPPLY_STARVATION", "SCOPE_TOO_WIDE__MEMORY_FACE",
        "NEW_ENTRANT_HARM__OBSERVATION_GAP", "EFFECT_NONSTATIONARY",
        "PATTERN_NOT_REENCOUNTERED", "TRIGGER_DENSITY_TOO_LOW__COURSE_FACE",
        "REPLAY_SCREEN_ELIMINATED_ALL", "ORDERING_SENSITIVE",
    ],
}

SKILL_TAXONOMY = {
    "general_skill": {
        "what": "procedural Harness knowledge (how to inspect, propose, verify); "
                "lives in h0 / general guidance",
        "carries": "workflow text; no program, no scope",
        "execution_rights": "none; it deploys nothing",
        "lifecycle": "frozen in HEC-1 (the policy/text face is HEC-2/3)",
    },
    "target_local_specific_skill": {
        "what": "Program x Scope predicate x task_consumer_key x evidence",
        "formed_by": "held-in Support-A plus the delayed authoritative gate in "
                     "the current domain",
        "execution_rights": "deploys where its Scope matches, in its own domain "
                            "(later held-in units, then held-out)",
        "lifecycle": "Draft -> WAITING / REVISABLE / FLAGGED -> Active -> "
                     "narrow / revoke -> frozen at course end",
    },
    "source_derived_skill": {
        "what": "the same Specific Skill in its cross-stage evidence role (K0)",
        "execution_rights": "none on arrival: supplied as a probe candidate "
                            "under requires_target_support; Active in the new "
                            "domain only after that domain's own gate",
    },
    "shared_capability": {
        "what": "repeated positive and risk evidence in similar observable "
                "context across at least two independent domains",
        "status_in_project": "none exists; S2 candidate v2 is SHARED_CANDIDATE "
                             "with target_support_required",
    },
    "promotion": (
        "General and Specific never promote into each other: one is 'how to "
        "search', the other is 'what works where'.  Specific -> Source-derived "
        "is a change of role over time; Source-derived -> Shared Capability "
        "needs cross-domain evidence; nothing becomes General"
    ),
}

NAMING = {
    "curve": "development mechanism curve (exposed development data)",
    "accumulation": "cross-cohort, within-dataset accumulation (KDD 2018 "
                    "air-quality series blocks); never cross-domain until a "
                    "second dataset is a Target",
    "phase_f": "0-LLM mechanical deployment of the frozen Skill policy; not the "
               "training-time Fast Agent reasoning again",
    "best_safe_global": "offline in-budget comparator",
}

# ---------------------------------------------------------------------------
# 9. stopping rules  /  10. Phase F
# ---------------------------------------------------------------------------

UNIT_FAULTS = (
    "candidate failure",
    "window verifier rejection",
    "repeatedly malformed LLM output",
    "cell-level LLM budget exhausted",
    "ServingContextDegenerate",
)

RUN_FAULTS = (
    "BACKEND_UNAVAILABLE past the retry policy",
    "a G2 or oracle wall leak",
    "the global LLM, token or time cap",
    "a protocol or data error",
    "any held-out read",
)

STOPPING_RULES = {
    "unit_fault": "the current unit abstains to identity, is recorded, and the "
                  "course continues",
    "run_fault": "the whole run stops and records RUN_BLOCKED_NO_VERDICT",
    "unit_faults": list(UNIT_FAULTS),
    "run_faults": list(RUN_FAULTS),
    "may_not": [
        "change the contract because Forward's effect had a sign",
        "re-run a unit because it looked bad",
        "run more Source because nothing survived",
        "write an instrument fault as TREATMENT_EMPTY or NOT_SUPPORTED",
    ],
    "artifacts_never_overwritten": "a new run_label instead",
}

PHASE_F = {
    "when": "after all three orderings are frozen",
    "arms": "the terminal state of every arm (Static / A3-frozen / A3-online "
            "when K0 is empty)",
    "cohort": "readable[80:120] x held-out origins [4056, 4296, 4536, 4776, 5016]",
    "how": "0-LLM mechanical deployment of the frozen Skill policy: Scope "
           "resolution of Active Skills, deployment.  No probe, no write-back, "
           "no Slow, and no Fast Agent call -- this is the policy the course "
           "produced, not the training-time agent reasoning again",
    "then": "every arm generates all outputs, then all Outcomes are opened once",
    "arms_evaluated": "the terminal state of EVERY arm of EVERY ordering; a "
                      "terminal state is never chosen on development results",
    "context_coverage": "stratified reporting only; never a filter and never a "
                        "reason to swap a frozen origin",
    "zero_coverage_reads": "NOT_REACHED_ON_HELD_OUT for that stratum and "
                           "UNSCOREABLE_NO_COVERAGE for criterion 4; safety by "
                           "silence is never a capability success",
    "requires": [
        "the course verdict is HEC1_EVOLUTION_SUPPORTED "
        "(HEC1_P1_ONLY__RECALL_ACCUMULATION does not qualify)",
        "K0 is non-empty: the course supports the full A5 claim, not only the "
        "Target-local component (sol v1.1 ruling 5; A3 never stands in for A5)",
        "a separate human seal release, every time",
    ],
    "requires_non_empty_k0": True,
    "empty_k0_course": (
        "may continue as A3-online vs A3-frozen component evidence of "
        "Target-local self-evolution; it does not open the main Phase F"),
    "why_two_conditions": (
        "sol's ruling 3: a supported verdict is what makes the terminal state "
        "worth spending fresh data on, and the release is a human's because no "
        "verdict may authorise its own follow-up"
    ),
    "never_automatic": True,
    "held_out_reads_so_far": 0,
    # sol v1.1: all three terminal states are evaluated and the headline is
    # their macro-average.  Reporting the best ordering would be choosing the
    # result after seeing it, and the orderings are not independent replicates
    # -- they are one course in three sequences.
    "evaluates": "the terminal state of all three orderings",
    "headline": "the macro-average across the three orderings",
    "may_not": [
        "report a single ordering as the result",
        "choose which ordering to open on",
        "drop an ordering whose terminal state looks worse",
    ],
    "per_ordering_reported": "yes, beside the macro-average, never instead",
}

BOUNDARY = {
    "risk_thresholds_changed": 0,
    "coverage_floor_changed": 0,
    "window_verifier_changed": 0,
    "operators_added": 0,
    "observation_features_added": 0,
    "risk_faces_added": 0,
    "gates_added": 0,
    "shas_added": 0,
    "manifests_added": 0,
    "artifacts_overwritten": 0,
    "methods_ttha_files_changed": 0,
    "stage_schemas_added": 0,
    "held_out_reads": 0,
    "ucr_test_outcome_reads": 0,
}

FREEZE_PROCEDURE = (
    "D2 landed and the [D2] fields were filled -- done",
    "sol confirmed all 13 defaults and added 6 rulings -- done 2026-09-03",
    "the user released Phase S 120 and three orderings at 500 each -- done",
    "the live arm loop was implemented and passed its 0-LLM end-to-end test "
    "-- sol's ruling 6, done",
    "assert_frozen and assert_launchable pass; Phase F stays refused -- done",
    "the 0-LLM smoke passes -- done",
    "frozen; from here only an appended erratum, and findings go to HEC-2",
)


# ---------------------------------------------------------------------------
# receipts and checks
# ---------------------------------------------------------------------------

def to_dict() -> dict[str, Any]:
    """The contract as a receipt, for a runner to embed verbatim."""
    arithmetic = budget_arithmetic()
    return {
        "stage": STAGE,
        "version": VERSION,
        "status": "FROZEN",
        "supersedes_nothing": SUPERSEDES_NOTHING,
        "inherits": INHERITS,
        "data_version": DATA_VERSION,
        "evidence_grade": EVIDENCE_GRADE,
        "claim_and_criteria": {
            "claim": (
                "the same Harness, over a long held-in course, becomes better "
                "under risk constraint on later units than a same-start, "
                "same-feedback, same-budget system forbidden to write back; and "
                "the advantage survives freezing into unseen held-out data"
            ),
            "all_four_must_hold": [
                "the online minus frozen difference grows with the course",
                "at least one Skill survives a revision and improves on an "
                "independent re-encounter",
                "A5-online minus A3-online shows accumulation (scored only "
                "when K0 is non-empty)",
                "the advantage is still there on held-out after freezing",
            ],
            "headline": "the curve, not any single gate",
        },
        "course": course(),
        "observation_gap": OBSERVATION_GAP,
        "consumer": CONSUMER,
        "program_space": PROGRAM_SPACE,
        "risk": RISK,
        "scope_class": SCOPE_CLASS,
        "gate_authority": GATE_AUTHORITY,
        "arms": ARMS,
        "unit_protocol": UNIT_PROTOCOL,
        "evaluation_face": EVALUATION_FACE,
        "outer_loop": {**OUTER_LOOP,
                       "steps_per_ordering": arithmetic[
                           "outer_steps_per_ordering"]},
        "lifecycle": LIFECYCLE,
        "scope_tool_chain": SCOPE_TOOL_CHAIN,
        "initializer_rules": initializer.declared_rules(),
        "per_unit_arm_budget": PER_UNIT_ARM_BUDGET,
        "budget_arithmetic": arithmetic,
        "readouts": READOUTS,
        "statistics": STATISTICS,
        "preregistered": PREREGISTERED,
        "verdicts": VERDICTS,
        "skill_taxonomy": SKILL_TAXONOMY,
        "naming": NAMING,
        "code_freeze": CODE_FREEZE,
        "code_provenance_erratum": CODE_PROVENANCE_ERRATUM,
        "replay_share_record": REPLAY_SHARE_RECORD,
        "supersedes_version": SUPERSEDES_VERSION,
        "stopping_rules": STOPPING_RULES,
        "phase_f": PHASE_F,
        "boundary": BOUNDARY,
        "census_key": CENSUS_KEY,
        "served_denominator": SERVED_DENOMINATOR,
        "scoreability": scoreability.to_dict(),
        "validation_search_baseline": VALIDATION_SEARCH_BASELINE,
        "phase_s_empty_again": PHASE_S_EMPTY_AGAIN,
        "p1_material_terminal_difference": P1_MATERIAL_TERMINAL_DIFFERENCE,
        "auto_continue_conditions": list(AUTO_CONTINUE_CONDITIONS),
        "ratification": {
            **RATIFICATION,
            "confirmed_by_sol": [dict(row) for row in CONFIRMED_BY_SOL],
            "user_releases": [dict(row) for row in USER_RELEASES],
            "pending_user": [dict(row) for row in PENDING_USER],
        },
        "freeze_procedure": list(FREEZE_PROCEDURE),
        "inherited_contracts": {
            "v1": v1.to_dict(), "v3": v3.to_dict(),
        },
    }


#: The modules whose denominator path sol's ruling 2 covers, and the patterns a
#: hard-coded served count would take in them.  A source scan is cruder than a
#: type, and it is what actually catches ``/ 20`` written in a hurry at 3am.
_DENOMINATOR_SCAN_TARGETS = (
    "run_hec1.py", "outer_loop.py", "scope_threshold_tool.py",
    "audit_hec1_readout.py", "audit_hec1_best_safe_global.py",
)
_DENOMINATOR_PATTERNS = ("/ 20", "/20", "* 20", "20.0)", "len(20")


def _hardcoded_denominator_scan() -> list[str]:
    """sol's ruling 2, enforced against the source rather than trusted.

    Comments and docstrings are excluded: the ruling is about arithmetic, and a
    sentence explaining that a block holds twenty series is not a denominator.
    """
    found: list[str] = []
    here = Path(__file__).resolve().parent
    for name in _DENOMINATOR_SCAN_TARGETS:
        path = here / name
        if not path.is_file():
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            # Blank out string literals instead of skipping the whole line: a
            # denominator written on a line that also carries a message must
            # still be caught (R-H: the earlier version skipped any quoted line).
            code = re.sub(r"'[^']*'|\"[^\"]*\"", "''", code)
            for pattern in _DENOMINATOR_PATTERNS:
                if pattern in code:
                    found.append(
                        "%s:%d looks like a hard-coded served count (%r); the "
                        "denominator must come from the roster"
                        % (name, number, pattern))
    return found


def assert_frozen() -> dict[str, Any]:
    """Re-derive what can be checked mechanically, so a drifted runner fails.

    This is a drift check, not a ratification.  ``assert_launchable`` is the one
    that refuses an unadjudicated contract.
    """
    failures: list[str] = []
    inherited = v3.assert_frozen()
    if not inherited["frozen"]:
        failures.extend("v3: %s" % item for item in inherited["failures"])

    if admission_policy.MATERIAL_THRESHOLD != RISK["material"]:
        failures.append("the material line drifted")
    if bounded.BOUNDED_MAX_HARMED_FRACTION != RISK["max_harmed_fraction"]:
        failures.append("the harmed-fraction budget drifted")
    if bounded.BOUNDED_MAX_SINGLE_SERIES_HARM != RISK["max_single_series_harm"]:
        failures.append("the single-series harm budget drifted")
    if distance.MIN_TREATED != RISK["min_treated"]:
        failures.append("the coverage floor drifted")
    if (forecast_p4.MAX_MODIFIED_FRACTION
            != PROGRAM_SPACE["window_verifier_max_modified_fraction"]):
        failures.append("the window verifier's modified-fraction bound drifted")
    if narrowing.MAX_ADDED_CLAUSES != 1:
        failures.append("the per-revision clause budget drifted")
    if narrowing.MAX_TOTAL_ADDED_CLAUSES != 2:
        failures.append("the lifecycle clause budget drifted")
    if drafts.MAX_REVISIONS != LIFECYCLE["max_revisions"]:
        failures.append("the revision cap drifted")
    if drafts.MAX_VERIFICATION_ATTEMPTS != LIFECYCLE[
            "max_verification_attempts"]:
        failures.append("the verification cap drifted")
    if len(threshold_tool.VOCABULARY) != 12:
        failures.append(
            "the Scope vocabulary is %d names, not the 12 binnable numeric "
            "observables this contract declares" % len(threshold_tool.VOCABULARY))
    if threshold_tool.MIN_TREATED != distance.MIN_TREATED:
        failures.append("the tool and the gate disagree on the coverage floor")

    # The course, re-derived from the audited supply artifact rather than
    # restated here, so a contract and an audit cannot drift apart.
    try:
        phase_s, phase_t = phase_s_units(), phase_t_units()
        supply = _supply()
    except (OSError, KeyError, ValueError) as exc:
        failures.append("the course supply artifact is unreadable: %s" % exc)
        phase_s, phase_t, supply = [], [], {}

    if len(phase_t) != 26:
        failures.append("Phase T is %d units, not 26" % len(phase_t))
    if len(phase_s) != 13:
        failures.append("Phase S is %d units, not 13" % len(phase_s))
    if len(phase_t) < 22:
        failures.append("Phase T falls below the 22-unit floor")

    s_blocks = {row["block"] for row in phase_s}
    t_blocks = {row["block"] for row in phase_t}
    if s_blocks & t_blocks:
        failures.append("a block appears in both Phase S and Phase T")
    if s_blocks and s_blocks != set(PHASE_S_BLOCKS):
        failures.append("the Phase S blocks are not the declared two")
    if t_blocks and t_blocks != set(PHASE_T_BLOCKS):
        failures.append("the Phase T blocks are not the declared four")

    expected = sorted((row["block"], row["origin"]) for row in phase_t)
    for name in ORDERINGS:
        try:
            rows = ordering(name)
        except (OSError, KeyError, ValueError) as exc:
            failures.append("ordering %s is unreadable: %s" % (name, exc))
            continue
        if sorted((row["block"], row["origin"]) for row in rows) != expected:
            failures.append(
                "ordering %s is not a permutation of the Phase T units" % name)

    if supply and not supply["exposure_cross_check"][
            "held_out_intersection_empty"]:
        failures.append("a course window touches a held-out (series, origin) pair")
    if supply and supply["exposure_cross_check"]["p4t_verdict"] != (
            "ALL_PROPOSED_HELD_OUT_PAIRS_UNEXPOSED"):
        failures.append("the exposure ledger does not clear the held-out block")

    arithmetic = budget_arithmetic() if phase_t else {}
    if arithmetic and arithmetic["forward_full_k0"] > arithmetic[
            "forward_hard_cap"]:
        failures.append(
            "the Forward arithmetic is %d calls, over the declared cap of %d"
            % (arithmetic["forward_full_k0"], arithmetic["forward_hard_cap"]))
    if arithmetic and arithmetic["phase_s_estimate"] > arithmetic[
            "phase_s_cap"]:
        failures.append("the Phase S estimate exceeds its proposed cap")

    if UNIT_PROTOCOL["inner_loop_immediate_slow"]:
        failures.append("the inner-loop Slow call is not closed")

    # The three unit counts, re-derived rather than restated.  int(0.8 x 23) is
    # 18 and 18/23 is 78.3%, so the rounding is checked and not assumed.
    if scoreability.SCHEDULED_UNITS != len(phase_t or ()):
        failures.append(
            "the scoreability manifest schedules %d units and the course has %d"
            % (scoreability.SCHEDULED_UNITS, len(phase_t or ())))
    if (scoreability.SCOREABLE_UNITS
            != scoreability.SCHEDULED_UNITS - len(scoreability.UNSCOREABLE_UNITS)):
        failures.append("the scoreable count is not scheduled minus unscoreable")
    if scoreability.MIN_PAIRED_CURVE_POINTS != math.ceil(
            scoreability.COMPLETION_FRACTION * scoreability.SCOREABLE_UNITS):
        failures.append("the paired-point floor is not ceil(fraction x scoreable)")
    if (scoreability.MIN_PAIRED_CURVE_POINTS / scoreability.SCOREABLE_UNITS
            < scoreability.COMPLETION_FRACTION):
        failures.append(
            "the paired-point floor is below the declared completion fraction")
    declared_units = {(row["block"], row["origin"]) for row in (phase_t or ())}
    for block, origin in scoreability.UNSCOREABLE_UNITS:
        if (block, origin) not in declared_units:
            failures.append(
                "unscoreable unit %s x %s is not in the course" % (block, origin))
    if sum(LLM_CAPS.values()) > TOTAL_LLM_HARD_CAP:
        failures.append("the released envelopes exceed the total hard cap")
    if len(CONFIRMED_BY_SOL) != 13:
        failures.append("the confirmed-default list is no longer the 13 sol saw")
    if len(SOL_RULINGS) != 6:
        failures.append("the ruling list is no longer the 6 sol gave")
    if len(SOL_FINAL_RULINGS) != 8:
        failures.append("the final-ruling list is no longer the 8 sol gave")
    if REPLAY_FITS_SHARE != REPLAY_SHARE_RECORD[
            "v1_1_scientific_orderings_run_under"]:
        failures.append("the replay share drifted from sol's v1.1 ruling 2 (1.0 "
                        "of the online arm's own course fits)")
    if outer_loop.MIN_POSITIVE_UNITS_FOR_ADD != 1:
        failures.append("the ADD threshold is not the 1 of sol's v1.1 ruling 1")
    if not PHASE_F.get("requires_non_empty_k0"):
        failures.append("Phase F no longer requires a non-empty K0")
    if "sign test" in str(STATISTICS.get("qualitative_criteria_for_P1")):
        failures.append("a sign test crept back into the P1 criteria")
    if "HEC1_P1_ONLY__RECALL_ACCUMULATION" not in VERDICTS:
        failures.append("the narrowed P1-only verdict is missing")
    if P1_MATERIAL_TERMINAL_DIFFERENCE != round(
            RISK["material"] * scoreability.SCOREABLE_UNITS, 6):
        failures.append("the P1 material line is not material x scoreable")
    for phrase in VERDICTS["P1_only_forbidden_phrasings"]:
        if phrase in str(VERDICTS["HEC1_P1_ONLY__RECALL_ACCUMULATION"]).replace(
                "**%s**" % phrase, ""):
            failures.append("a forbidden P1-only phrasing is used as a claim")
    if VALIDATION_SEARCH_BASELINE["enters_the_harness"]:
        failures.append("the validation-search baseline entered the Harness")
    if VALIDATION_SEARCH_BASELINE["llm_calls"] != 0:
        failures.append("the validation-search baseline is not 0-LLM")
    if PHASE_S_EMPTY_AGAIN["k0_stays"] != "empty, recorded A5_TREATMENT_EMPTY":
        failures.append("the exhaustive diagnostic may not produce a K0")
    if PHASE_F.get("headline") != "the macro-average across the three orderings":
        failures.append("Phase F stopped reporting the macro-average")
    if PHASE_F["never_automatic"] is not True:
        failures.append("Phase F stopped requiring a human release")
    failures.extend(_hardcoded_denominator_scan())
    if any(value for key, value in BOUNDARY.items()):
        failures.append("a boundary counter is non-zero: %s" % sorted(
            key for key, value in BOUNDARY.items() if value))
    if "threshold" in SCOPE_TOOL_CHAIN["slow_authors"]:
        failures.append("Slow authors the threshold again")

    return {
        "frozen": not failures,
        "failures": failures,
        "version": VERSION,
        "is_ratified": bool(RATIFICATION["sol_confirmed"]),
        "v3_frozen": inherited["frozen"],
        "phase_s_units": len(phase_s),
        "phase_t_units": len(phase_t),
        "orderings": list(ORDERINGS),
        "budget_arithmetic": arithmetic,
        "what_this_does_not_check": (
            "whether the values were adjudicated; see assert_launchable"
        ),
    }


PHASE_RELEASE_FIELD = {
    "phase_s": "budget.phase_s_llm_cap",
    "phase_t_forward": "budget.phase_t_forward_llm_cap",
    "phase_t_reverse": "budget.phase_t_reverse_llm_cap",
    "phase_t_interleaved": "budget.phase_t_interleaved_llm_cap",
    "phase_f": "phase_f.seal_opening",
}


def assert_launchable(phase: str, *, verdict: str | None = None,
                      seal_released: bool = False,
                      k0_nonempty: bool = False) -> dict[str, Any]:
    """What a live phase has to satisfy before it may spend an LLM call.

    Phase F is deliberately not satisfiable by any argument a runner can pass to
    itself: it needs a supported verdict, a non-empty K0 (sol v1.1 ruling 5)
    *and* ``seal_released``, which only a human invocation sets.  Every other
    phase is satisfied by the ratification.
    """
    state = assert_frozen()
    blockers: list[str] = list(state["failures"])
    if not RATIFICATION["sol_confirmed"]:
        blockers.append("sol has not confirmed the contract")
    released = {str(name) for name in (RATIFICATION["user_released"] or ())}
    needed = PHASE_RELEASE_FIELD.get(str(phase))
    if needed is None:
        blockers.append("unknown phase %r" % phase)
    elif str(phase) == "phase_f":
        if str(verdict) != "HEC1_EVOLUTION_SUPPORTED":
            blockers.append(
                "Phase F needs a supported course verdict, not %r" % verdict)
        if not k0_nonempty:
            blockers.append(
                "Phase F needs a non-empty K0: an empty-K0 course is component "
                "evidence and A3 does not stand in for A5 (sol v1.1 ruling 5)")
        if not seal_released:
            blockers.append(
                "Phase F needs a human seal release; it is never inferred")
    elif needed not in released:
        blockers.append("the user has not released %s" % needed)
    return {
        "launchable": not blockers,
        "phase": str(phase),
        "llm_cap": LLM_CAPS.get(str(phase)),
        "blockers": blockers,
        "verdict_if_launched_anyway": "BLOCKED_ON_CONTRACT",
    }


# ---------------------------------------------------------------------------
# the draft artifacts
# ---------------------------------------------------------------------------

def _md(payload: Mapping[str, Any]) -> str:
    arithmetic = payload["budget_arithmetic"]
    course_ = payload["course"]
    lines = [
        "# HEC-1 contract (%s) -- FROZEN 2026-09-03" % VERSION,
        "",
        "Written from `docs/HEC1_CONTRACT_SKELETON_2026-09-03.md` with every "
        "`[D2]` field re-derived from `%s` at read time, so the contract and the "
        "audit cannot drift apart. sol confirmed all 13 mainline defaults and "
        "added 6 rulings; the user released four budget envelopes." % SUPPLY.name,
        "",
        "## Status",
        "",
        "| check | state |",
        "| --- | --- |",
        "| mechanical drift (`assert_frozen`) | see the JSON `assert_frozen` block |",
        "| sol ratification | **confirmed** (%d defaults, %d rulings) |" % (
            len(CONFIRMED_BY_SOL), len(SOL_RULINGS)),
        "| user budget release | **released** (%d envelopes) |" % len(
            USER_RELEASES),
        "| still human, always | %s |" % ", ".join(
            "`%s`" % row["field"] for row in PENDING_USER),
        "",
        "## Autonomy envelope",
        "",
        "> %s" % payload["ratification"]["autonomy_envelope"],
        "",
        "Never autonomous: %s." % "; ".join(
            payload["ratification"]["what_never_becomes_autonomous"]),
        "",
        "## Course",
        "",
        "| item | value |",
        "| --- | --- |",
        "| Phase S units | %s |" % course_["phase_s_units"],
        "| Phase T units | %s (all-usable %s not taken) |" % (
            course_["phase_t_units"], course_["phase_t_all_usable_not_taken"]),
        "| Phase S blocks | %s |" % ", ".join(course_["blocks"]["phase_s"]),
        "| Phase T blocks | %s |" % ", ".join(course_["blocks"]["phase_t"]),
        "| blocks disjoint | %s |" % course_["blocks"]["disjoint"],
        "| held-out intersection empty | %s |" % course_["exposure"][
            "held_out_intersection_empty"],
        "| pattern-sparse units | %s |" % course_["composition"][
            "pattern_sparse_units"],
        "",
        "HEC-1 therefore does **not** test silence when the pattern is absent: "
        "every unit has at least six series over `z_peak >= 3`, so the frozen "
        "initialiser barely filters anyone and all narrowing comes from Slow's "
        "clause. Recorded as a limit, not discovered in the report.",
        "",
        "## Budget arithmetic",
        "",
        "| item | value |",
        "| --- | --- |",
        "| LLM per unit-arm | %s |" % arithmetic["llm_per_unit_arm"],
        "| outer steps per ordering | %s (k = %s) |" % (
            arithmetic["outer_steps_per_ordering"],
            payload["outer_loop"]["period_k_units"]),
        "| Forward, full K0 | **%s** / cap %s |" % (
            arithmetic["forward_full_k0"], arithmetic["forward_hard_cap"]),
        "| Forward, empty K0 | %s |" % arithmetic["forward_empty_k0"],
        "| Phase S estimate | %s / cap %s |" % (
            arithmetic["phase_s_estimate"], arithmetic["phase_s_cap"]),
        "| released envelopes | %s (sum %s, hard cap %s) |" % (
            ", ".join("%s %s" % (key, value)
                      for key, value in sorted(arithmetic["llm_caps"].items())),
            arithmetic["sum_of_released_envelopes"],
            arithmetic["total_llm_hard_cap"]),
        "| Best-Safe-Global fits | %s |" % arithmetic[
            "best_safe_global_fit_cap"],
        "",
        "## sol's rulings",
        "",
        "| ruling | enforced by |",
        "| --- | --- |",
    ]
    for row in payload["ratification"]["rulings"]:
        lines.append("| %s | `%s` |" % (row["ruling"], row["enforced_by"]))
    lines += ["", "## Confirmed defaults", "",
              "| field | value | authority |", "| --- | --- | --- |"]
    for row in payload["ratification"]["confirmed_by_sol"]:
        lines.append("| `%s` | %s | %s |" % (
            row["field"], row["default"], row["authority"]))
    lines += ["", "## Auto-continuation conditions", ""]
    for index, condition in enumerate(payload["auto_continue_conditions"], 1):
        lines.append("%d. %s" % (index, condition))
    lines += [
        "",
        "## Boundary",
        "",
        "Every counter is zero: %s." % ", ".join(
            "`%s`" % key for key in sorted(payload["boundary"])),
        "",
        "## Freeze procedure",
        "",
    ]
    for index, step in enumerate(payload["freeze_procedure"], 1):
        lines.append("%d. %s" % (index, step))
    lines += ["", "## sol's final rulings (2026-09-03, before the first LLM call)",
              "", "| ruling | enforced by |", "| --- | --- |"]
    for row in payload["ratification"]["final_rulings"]:
        lines.append("| %s | `%s` |" % (row["ruling"], row["enforced_by"]))
    lines += ["", "### Amendments relative to the 01:xx freeze", ""]
    for item in payload["ratification"]["amendments_before_launch"]:
        lines.append("- %s" % item)
    lines += ["", "## Statistics (descriptive)", ""]
    for criterion in payload["statistics"]["qualitative_criteria_for_P1"]:
        lines.append("- P1: %s" % criterion)
    lines.append("- %s" % payload["statistics"]["no_significance_gate"])
    lines += [
        "",
        "This file now takes only an appended erratum section; no field "
        "changes, and anything the run turns up goes to HEC-2.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                       help="report the drift check without writing anything")
    args = parser.parse_args(argv)
    state = assert_frozen()
    payload = {
        **to_dict(),
        "written_at": datetime.now().astimezone().isoformat(),
        "assert_frozen": state,
        "assert_launchable": {
            phase: assert_launchable(phase) for phase in PHASE_RELEASE_FIELD},
    }
    print("version            : %s" % VERSION)
    print("mechanical drift   : %s" % ("clean" if state["frozen"] else "DRIFTED"))
    for failure in state["failures"]:
        print("  - %s" % failure)
    print("phase S / T units  : %s / %s"
          % (state["phase_s_units"], state["phase_t_units"]))
    arithmetic = state["budget_arithmetic"]
    if arithmetic:
        print("forward full / empty K0 : %s / %s <= %s"
              % (arithmetic["forward_full_k0"], arithmetic["forward_empty_k0"],
                 arithmetic["forward_hard_cap"]))
        print("phase S estimate   : %s <= %s"
              % (arithmetic["phase_s_estimate"], arithmetic["phase_s_cap"]))
    print("sol confirmed      : %s (%d defaults, %d rulings)"
          % (RATIFICATION["sol_confirmed"], len(CONFIRMED_BY_SOL),
             len(SOL_RULINGS)))
    for phase in PHASE_RELEASE_FIELD:
        verdict = assert_launchable(phase)
        print("launchable %-20s: %s%s" % (
            phase, verdict["launchable"],
            "" if verdict["launchable"] else "  <- %s" % verdict["blockers"][0]))
    if args.check:
        return 0 if state["frozen"] else 1
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    print("wrote %s" % OUT_JSON.relative_to(PROJECT_ROOT).as_posix())
    print("wrote %s" % OUT_MD.relative_to(PROJECT_ROOT).as_posix())
    return 0 if state["frozen"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
