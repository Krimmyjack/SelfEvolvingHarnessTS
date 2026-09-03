"""P4U-v3: the last Source revision -- a bounded multi-round Scope lifecycle.

v2 is imported, not copied, and no v2 artifact is rewritten.  v2 established the
mechanism and produced a null, and the null was not yet attributable.  Three
things stood between the run and an interpretable answer, and this contract
fixes exactly those three:

1. **The Slow call was spent by probe order.**  A round refusing several
   materially-positive candidates handed Slow whichever the Fast agent proposed
   first.  At origin 2136 that was ``hampel_filter``, for which the
   pre-registered oracle bound (``p4y``) had already established that *no*
   feasible one-clause narrowing exists at that origin, while the other refused
   probe there admits eleven.  Slow was handed the impossible one and cleared
   three of the four delayed lines anyway.

2. **Half the Slow calls died on manifest protocol, not on Scope.**  Fields at
   the wrong nesting level, a 64-hex dependency SHA the model had to copy, a
   required ``observable_applicability`` -- all Runtime-owned or
   Runtime-derivable, and two of them overwritten by the Runtime immediately
   afterwards anyway.

3. **The lifecycle gave one move.**  One Slow call, one clause, one delayed
   reading, then the Draft was destroyed.  AGENTS §3 permits a Target-local
   Draft to enter the next round, and §5's evidence of evolution is *local
   conflict -> bounded revision -> independent re-verification -> survival*.
   Killing the Draft at "local conflict" means the chain never reaches its
   second link, so a failure there cannot separate "Slow cannot bound the tail"
   from "one clause was not enough".

What is still not authorized
----------------------------
The risk thresholds (0.20 / 0.30), the material line, the coverage floor, the
operator set, the Observation vocabulary, the Fast schema, H0's global
``verification.rules.scope_risk_guards``, the cohorts, the origins and the
per-arm budgets.  The Source budget is unchanged at 60 calls -- removing the
friction is meant to buy method evidence out of the same spend, not more spend.

The stopping rule
-----------------
If the second revision still fails the delayed four lines, this version of the
Source line is closed and reported as a null -- and that null is then clean:
the class was ranked, the friction was removed, and the Draft had its second
move.  If it passes, one independent re-encounter must also pass before the
Skill is activated.
"""
from __future__ import annotations

from typing import Any, Mapping

from evaluation.main_protocol_p4 import main_experiment_contract as v1
from evaluation.main_protocol_p4 import main_experiment_contract_v2 as v2
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import scope_clause_agent as clause_agent
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from evaluation.main_protocol_p4 import scope_repair_distance as distance
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

VERSION = "P4U-v3"
SUPERSEDES_NOTHING = (
    "v1 and v2 are imported and still govern geometry, arms, budgets, "
    "endpoints and the risk-refusal attribution; no earlier artifact is "
    "rewritten and no earlier reading is reinterpreted"
)

#: Why v2's null could not be read, in the form the run has to answer.
WHY_V3 = {
    "the_call_was_spent_by_probe_order": (
        "origin 2136: the refused probe routed to Slow was hampel_filter, "
        "which p4y had already shown admits zero feasible one-clause "
        "narrowings at that origin; the other refused probe there admits 11"
    ),
    "half_the_calls_died_on_protocol": (
        "3 of 5 rounds ended on manifest schema errors before any Scope "
        "judgement: fields at the payload's top level, a malformed "
        "dependency_precondition_shas.observable_contract, and a required "
        "observable_applicability"
    ),
    "the_lifecycle_gave_one_move": (
        "origin 2136 cleared coverage 7/20, aggregate +0.182 and harmed "
        "fraction 0.05, and missed only single-series harm at 0.921; the "
        "Draft was then destroyed with no second revision"
    ),
}

#: (1) Slow writes one clause; the Runtime writes the manifest around it.
MANIFEST_ASSEMBLY = {
    "module": "evaluation.main_protocol_p4.scope_clause_agent",
    "slow_output_schema": "slow_scope_clause_v1",
    "slow_authors": list(clause_agent.SLOW_AUTHORED_FIELDS),
    "runtime_authors": list(clause_agent.RUNTIME_AUTHORED_FIELDS),
    "what_is_given_up": (
        "predicted_agent_behavior_change, predicted_data_effect and "
        "falsification_condition were the agent's own commitments and are now "
        "Runtime boilerplate.  They gate nothing in this protocol, but the "
        "loss is recorded here rather than left to be found in a diff"
    ),
    "what_is_not_given_up": (
        "the clause itself.  The Runtime never proposes a feature, a direction "
        "or a threshold, never repairs a clause it dislikes, and refuses "
        "rather than substitutes when a clause is unusable"
    ),
    "assembled_manifest_is_validated_against": "slow_edit_v1",
    "aggregate_negative_path_unchanged": (
        "the historical TTHASlowAgent still serves it; only the risk-refusal "
        "branch takes the clause agent"
    ),
}

#: (2) Which refusal the round's single Slow call is spent on.
CANDIDATE_SELECTION = {
    "module": "evaluation.main_protocol_p4.scope_repair_distance",
    "distance": (
        "the fewest served series that would have to be excluded before the "
        "coverage, aggregate, harmed-fraction and single-series lines all hold"
    ),
    "computed_from": "the Support-A per-series gain vector already on the refusal",
    "ties": "larger aggregate gain, then candidate id",
    "greedy_is_exact": (
        "dropping the k most-harmful series is simultaneously optimal for all "
        "four lines, so the greedy count is the minimum over all subsets"
    ),
    "does_not_use": [
        "the oracle bound p4y", "the feature the oracle selected",
        "its threshold", "the identity of any series",
    ],
    "is_a_support_window_number": (
        "p4y proves a feasible Scope exists in the Support window, not that "
        "the predicate transfers to the delayed window; this ranks local "
        "repairability and promises no repair"
    ),
    "reaches_slow": "nothing -- the ranking selects, it is never put in the card",
    "deferred_to_end_of_round_because": (
        "a choice among refusals cannot be made until every refusal is known; "
        "the aggregate-negative trigger still fires in-loop on the first "
        "material failure, bit-identically"
    ),
}

#: (3) and (4) The restricted Draft and its one further revision.
RESTRICTED_DRAFT_LIFECYCLE = {
    "module": "evaluation.main_protocol_p4.restricted_draft",
    "on_delayed_conflict": (
        "the Draft is restricted, not destroyed: kept as a Runner record, "
        "never written to the active snapshot, never retrieved, never counted "
        "as a Skill, and never used to serve a prediction"
    ),
    "held_outside_the_snapshot_because": (
        "the claim under test is how many Skills survived; a record that is "
        "not a Skill must not be able to be counted as one"
    ),
    "second_revision": {
        "where": "the next held-in origin",
        "how": (
            "the frozen program is re-supplied as a probe candidate carrying "
            "the predicate it had reached; if it is refused again on the tail "
            "budget, the ordinary risk-refusal route gives it one more clause"
        ),
        "program": "unchanged, operators and parameters alike",
        "predicate": "monotone narrowing only, at most one further clause",
        "total_added_clauses": narrowing.MAX_TOTAL_ADDED_CLAUSES,
        "counted_against": (
            "the frozen initialiser's own predicate, not the previous "
            "revision, so the bound cannot be walked past one legal step at a "
            "time"
        ),
        "coverage_floor": "unchanged at 5/20",
        "thresholds_operators_features_budget": "unchanged",
    },
    "max_revisions": drafts.MAX_REVISIONS,
    "then": (
        "a second delayed failure closes this version of the Source line; a "
        "pass goes on to one independent re-encounter, and only a Draft that "
        "clears that is activated"
    ),
    "geometry_limit_recorded": (
        "a Draft first restricted at the last held-in origin (2856) has no "
        "next origin to be revised at, and is closed as out_of_origins rather "
        "than counted as a second failure"
    ),
}

#: Sol's point 6: the mechanism is the method, so it cannot be A5-only.
FREEZE_INTO_THE_ARMS = {
    "requirement": (
        "whatever the Source line runs under must then be frozen unchanged "
        "into A3 and A5 alike"
    ),
    "why": (
        "a lifecycle available to one arm and not the other would make the "
        "A3/A5 contrast a comparison of protocols rather than of Skill "
        "inheritance, which is the only thing that contrast is for"
    ),
    "applies_to": [
        "the clause-only Slow call and the Runtime manifest assembly",
        "the risk-refusal selection rule",
        "the restricted-Draft lifecycle and its two-revision bound",
    ],
    "budgets_unchanged": True,
}

#: Kept apart on purpose, so two repairs do not share one attribution.
SEPARATE_FINDING = {
    "name": "CANDIDATE_SUPPLY_INSTABILITY",
    "observed": (
        "at origin 2856 the Fast agent proposed no candidate at all, on the "
        "origin where AGENTS 5.1 had already found period_median_complete -> "
        "outlier_* positive on both faces: Program headroom is known to exist "
        "there and no candidate was supplied"
    ),
    "first_fault_family": (
        "Program exists but no candidate -- Observation / localization / "
        "supply, a different family from the Scope/Risk conflict this "
        "contract addresses"
    ),
    "action_this_round": "recorded, not repaired",
    "why_not_now": (
        "repairing candidate supply and the Scope lifecycle in the same run "
        "would make a change in the outcome unattributable to either"
    ),
}

#: Unchanged from v2, and restated because the second revision must meet them
#: on its own reading rather than inheriting the first revision's.
DELAYED_ADMISSION = dict(v2.DELAYED_ADMISSION)

PROMOTION_GATES = (
    *v2.PROMOTION_GATES,
    "second_revision: a Draft restricted by a delayed conflict may be revised "
    "once more at the next held-in origin, and must clear the same four lines "
    "there on a delayed reading of its own",
)

ARTIFACTS = {
    "source_line": "artifacts/main_protocol/p4w3_source_line_v3.json",
    "does_not_overwrite": [
        "p4w_source_line.json", "p4w2_source_line_v2.json",
        "p4x_admission_regime.json", "p4y_oracle_scope_bound.json",
        "p4z_risk_refusal_routing.json",
    ],
    "must_persist": [
        *v2.ARTIFACTS["must_persist"],
        "the full risk-refusal ranking of every round, not only the selection",
        "every clause Slow proposed, including the ones that were unusable",
        "each restricted Draft's root predicate, every revision, and every "
        "delayed reading it failed",
    ],
}

RUN_ORDER = (
    "freeze this revision and lock the three changes as additive",
    "Source line v3: rank refusals, one clause per call, restrict rather than "
    "destroy, one further revision at the next origin",
    "a Draft that clears Support, delayed and one re-encounter is a Skill",
    "if none does -> this version of the Source line is closed as a clean null",
    "freeze the mechanism into A3 and A5 alike, then Target",
    "freeze all arms, then open held-out once",
)

BOUNDARY = {
    **v2.BOUNDARY,
    "risk_thresholds_changed": 0,
    "operators_added": 0,
    "observation_features_added": 0,
    "route_table_causes_added": 0,
    "route_table_target_classes_added": 0,
    "surfaces_added": 0,
    "stage_schemas_added": 1,
    "artifacts_overwritten": 0,
    "source_llm_budget_raised": 0,
}


def to_dict() -> Mapping[str, Any]:
    """The revision as a receipt, for the runner to embed verbatim."""
    return {
        "stage": "P4U_V3_METHOD_REVISION",
        "version": VERSION,
        "supersedes_nothing": SUPERSEDES_NOTHING,
        "v2": v2.to_dict(),
        "why_v3": WHY_V3,
        "manifest_assembly": MANIFEST_ASSEMBLY,
        "candidate_selection": CANDIDATE_SELECTION,
        "restricted_draft_lifecycle": RESTRICTED_DRAFT_LIFECYCLE,
        "freeze_into_the_arms": FREEZE_INTO_THE_ARMS,
        "separate_finding": SEPARATE_FINDING,
        "delayed_admission": DELAYED_ADMISSION,
        "promotion_gates": list(PROMOTION_GATES),
        "artifacts": ARTIFACTS,
        "run_order": list(RUN_ORDER),
        "boundary": BOUNDARY,
    }


def assert_frozen() -> dict[str, Any]:
    """Re-derive what can be checked, so a drifted runner fails loudly."""
    failures: list[str] = []
    inherited = v2.assert_frozen()
    if not inherited["frozen"]:
        failures.extend("v2: %s" % item for item in inherited["failures"])

    if narrowing.MAX_ADDED_CLAUSES != 1:
        failures.append("the per-revision clause budget drifted")
    if narrowing.MAX_TOTAL_ADDED_CLAUSES != 2:
        failures.append("the lifecycle clause budget drifted")
    if drafts.MAX_REVISIONS != 2:
        failures.append("the revision cap drifted")
    if distance.MIN_TREATED != 5:
        failures.append("the coverage floor drifted from the declared 5/20")
    if distance.MATERIAL != admission_policy.MATERIAL_THRESHOLD:
        failures.append("the material line drifted")
    if distance.MAX_HARMED != bounded.BOUNDED_MAX_HARMED_FRACTION:
        failures.append("the harmed-fraction budget drifted from P4b")
    if distance.MAX_HARM != bounded.BOUNDED_MAX_SINGLE_SERIES_HARM:
        failures.append("the single-series harm budget drifted from P4b")

    # The one field Slow still authors.  If this list ever grows to include a
    # feature, a direction or a threshold, the Runtime has started proposing
    # the predicate and the run would measure nothing.
    if list(clause_agent.SLOW_AUTHORED_FIELDS) != [
            "new_value.serving_scope.predicate[-1]"]:
        failures.append("what Slow authors drifted")

    from SelfEvolvingHarnessTS.methods.ttha.schema_contracts import (
        load_stage_schema,
    )
    try:
        schema = load_stage_schema("slow_scope_clause_v1")
    except (ValueError, OSError) as exc:
        failures.append("the clause schema is unloadable: %s" % exc)
        schema = {}
    if schema.get("required") != ["scope_clause"]:
        failures.append("the clause schema no longer requires exactly a clause")
    if schema.get("additionalProperties") is not False:
        failures.append("the clause schema stopped being closed")

    return {
        "frozen": not failures,
        "failures": failures,
        "version": VERSION,
        "v2_frozen": inherited["frozen"],
        "v1_frozen": inherited.get("v1_frozen"),
        "max_added_clauses": narrowing.MAX_ADDED_CLAUSES,
        "max_total_added_clauses": narrowing.MAX_TOTAL_ADDED_CLAUSES,
        "max_revisions": drafts.MAX_REVISIONS,
    }
