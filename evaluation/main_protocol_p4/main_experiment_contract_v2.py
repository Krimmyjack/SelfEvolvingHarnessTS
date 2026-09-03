"""P4U-v2: one additive method revision, and the gates it must pass.

v1 (``main_experiment_contract``) stands unchanged and is imported, not copied.
Everything below is what v2 adds, and it adds exactly one thing to the method:
**a materially positive candidate refused on the tail budget is a fault, and it
is routed.**

Why v1 could not produce a treatment
------------------------------------
The Source line formed zero Skills (``p4w``), and the reason was not the
proposer: seven of nine probes cleared the aggregate line -- one at +0.654 --
and every one of them was refused for tail risk.  Two defects sat underneath
that reading and both are fixed:

* ``open_delayed`` audited a scoped Skill on the **global** serving set, so the
  approval gate measured a different policy from the one being deployed;
* the admission gate computed each refusal's reason and wrote it to the probe
  row, while the fault router beneath it read only the aggregate gain.  A
  positive-but-refused candidate matched neither branch -- no winner, no Slow,
  not even ``harm_count`` -- so a round of them reported zero faults.

The second is the first fault of record: ``RISK_REFUSAL_INVISIBLE_TO_FAULT_
ROUTER``.  Fixing it exposed that no attribution reachable from the online loop
could name a Scope fault at all (``p4z``, proven by enumerating the router's
whole input space), which is what this contract authorizes a way through.

Why the attribution is RISK_GAP and not SKILL_LIBRARY_GAP
---------------------------------------------------------
The first fault is not "no Skill exists".  A deployable candidate was found and
then refused for a Scope/Risk conflict, so the attribution has to say that, or
the artifact records the wrong cause for the right repair.  ``RISK_GAP`` gains
``capability`` in its target classes for this; the change is strictly additive
and locked as such in ``tests/main_protocol/test_risk_gap_route_extension.py``.

What is *not* authorized
------------------------
The risk thresholds, the operator set, the Observation vocabulary, the Fast
schema, H0's global ``verification.rules.scope_risk_guards``, the cohorts, the
origins and the per-arm budgets are all untouched.  Slow may revise one thing:
the ``serving_scope`` of the Draft it is ADDing, monotonically.
"""
from __future__ import annotations

from typing import Any, Mapping

from evaluation.main_protocol_p4 import main_experiment_contract as v1
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import scope_narrowing_preflight as narrowing
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

VERSION = "P4U-v2"
SUPERSEDES_NOTHING = (
    "v1 is imported and still governs geometry, arms, budgets and endpoints; "
    "no v1 artifact is rewritten and no v1 reading is reinterpreted"
)

FIRST_FAULT = "RISK_REFUSAL_INVISIBLE_TO_FAULT_ROUTER"

#: The one new attribution edge.
RISK_REFUSAL_ROUTE = {
    "when": (
        "the admission gate refuses a probe whose aggregate gain clears the "
        "material line, for harmed_fraction_over_budget or "
        "single_series_harm_over_budget"
    ),
    "attributed_cause": "RISK_GAP",
    "why_not_skill_library_gap": (
        "the fault is not a missing Skill; a candidate was found and refused "
        "for a Scope/Risk conflict, and the cause code is the recorded claim"
    ),
    "surface_catalog_this_round": ("skill_library.entries/{skill_id}",),
    "operation": "ADD",
    "catalog_is_narrower_than_the_route": (
        "RISK_GAP also authorizes its risk-guard and verification surfaces; "
        "this round exposes only the Skill ADD, so a single ruling cannot "
        "quietly become licence to edit H0's global scope_risk_guards"
    ),
    "route_table_change": {
        "file": "evaluation/minipipe/feedback/fault_routes.json",
        "change": "RISK_GAP.target_classes += ['capability']",
        "additive": True,
        "newly_authorized": [
            "RISK_GAP x capability x ADD x capability",
            "RISK_GAP x capability x PATCH x capability",
        ],
        "nothing_previously_authorized_was_withdrawn": True,
    },
}

#: What Slow may and may not touch when it ADDs the Draft.
SLOW_MANDATE = {
    "adds": "one inactive Draft capability Skill, in a fork",
    "program": "frozen to the probed program, operators and parameters alike",
    "observable_applicability": (
        "fixed by the Runtime; Slow may not write it.  It decides when the "
        "Skill is retrieved, which is a different question from which served "
        "series it may treat"
    ),
    "serving_scope": (
        "the only writable field: a monotone narrowing of the probe's own "
        "Scope, adding at most %d clause" % narrowing.MAX_ADDED_CLAUSES
    ),
    "may_not_touch": [
        "operators", "risk thresholds", "Observation features",
        "the Fast schema", "H0 verification.rules.scope_risk_guards",
        "cohorts, origins, per-arm budgets",
    ],
}

#: Nothing else in the chain checks the content of a revised Scope.
REQUIRED_PREFLIGHT = {
    "module": "evaluation.main_protocol_p4.scope_narrowing_preflight",
    "why": (
        "the route table's monotone rule is a target-class gate that never "
        "sees the predicate, and RISK_GAP is not in the narrow-direction map "
        "at all.  Without this preflight a Draft could be ADDed under an "
        "authorized cause carrying a Scope that reaches further than the one "
        "it replaced"
    ),
    "structural_check_is_the_transferable_one": (
        "a subset check is cohort-local; requiring the revised clause set to "
        "be a superset of the original makes the narrowing hold on every "
        "cohort, because a conjunction with more clauses selects fewer series "
        "by construction"
    ),
    "checks": [
        "keeps every original clause",
        "adds at most one clause",
        "the added clause names a deployment-visible feature",
        "resolves to a strict subset at the origin it was derived from",
    ],
}

#: The Draft earns nothing by being created.
PROMOTION_GATES = (
    "support_reverify: the frozen program under the revised Scope is re-probed "
    "on the same Support face",
    "delayed_reresolved: at origin+48 the predicate is resolved again from "
    "that origin's own features -- never carried over as a UID list -- and the "
    "revised Scope must clear every line there",
    "re_encounter: the Skill is retrieved and deployed safely at a later "
    "origin it did not help produce",
)

#: Read on the delayed face, which is the only face that can tell a revision
#: from a memorisation of the origin it was derived from.
DELAYED_ADMISSION = {
    "min_coverage_treated_of_served": "5/20",
    "min_aggregate_gain": admission_policy.MATERIAL_THRESHOLD,
    "max_harmed_fraction": bounded.BOUNDED_MAX_HARMED_FRACTION,
    "max_single_series_harm": bounded.BOUNDED_MAX_SINGLE_SERIES_HARM,
    "thresholds_changed": 0,
    "why_a_coverage_floor": (
        "narrowing to almost nothing clears every risk line and deploys "
        "nothing; the floor is declared here, before the run, so a degenerate "
        "abstention cannot be read afterwards as a success"
    ),
    "support_is_not_an_endpoint": (
        "the Support reading is the feedback the revision was derived from, so "
        "it cannot also be the evidence that the revision generalises"
    ),
}

#: The oracle bound (``p4y``) selected predicates while looking at outcomes.
ORACLE_QUARANTINE = {
    "may_cross_into_the_run": "nothing",
    "explicitly_withheld_from_slow": [
        "the feature the oracle selected", "its threshold",
        "the identity of the excluded series", "the ranking of candidates",
    ],
    "must_not_change": [
        "the Slow prompt or any instruction text",
        "the candidate ordering or the programs probed",
        "scope_initializer's family table or thresholds",
    ],
    "what_it_was_used_for": (
        "one decision, taken before the run: whether the revision class "
        "contains a feasible Scope at all, so that a Slow failure would be "
        "attributable to Slow rather than to an impossible task"
    ),
}

ARTIFACTS = {
    "source_line": "artifacts/main_protocol/p4w2_source_line_v2.json",
    "does_not_overwrite": [
        "p4w_source_line.json", "p4x_admission_regime.json",
        "p4y_oracle_scope_bound.json", "p4z_risk_refusal_routing.json",
    ],
    "must_persist": [
        "every probe's full per-series gain vector",
        "the resolved serving series at both the support and delayed origins",
        "the narrowing preflight verdict for every proposed revision",
        "the attributed cause and the surface actually authorized",
    ],
    "why_per_series_is_mandatory": (
        "a dry run destroyed the v1 vectors and they could not be rebuilt from "
        "anything; every risk verdict in this protocol is a statement about "
        "them, so an artifact without them cannot be re-audited"
    ),
}

RUN_ORDER = (
    "freeze this revision and lock the route change as additive",
    "Source line v2: risk refusals routed, Drafts ADDed, Scopes narrowed",
    "if a Draft clears Support, delayed and one re-encounter -> it is a Skill",
    "if none does -> record which gate stopped it and stop; do not run A5",
    "freeze the Source Skills, then Target: Static / A3 / A5 / Parallel@B",
    "freeze all arms, then open held-out once",
)

BOUNDARY = {
    **v1.BOUNDARY,
    "route_table_causes_added": 0,
    "route_table_target_classes_added": 1,
    "surfaces_added": 0,
    "artifacts_overwritten": 0,
}


def to_dict() -> Mapping[str, Any]:
    """The revision as a receipt, for the runner to embed verbatim."""
    return {
        "stage": "P4U_V2_METHOD_REVISION",
        "version": VERSION,
        "supersedes_nothing": SUPERSEDES_NOTHING,
        "v1": v1.to_dict(),
        "first_fault": FIRST_FAULT,
        "risk_refusal_route": RISK_REFUSAL_ROUTE,
        "slow_mandate": SLOW_MANDATE,
        "required_preflight": REQUIRED_PREFLIGHT,
        "promotion_gates": list(PROMOTION_GATES),
        "delayed_admission": DELAYED_ADMISSION,
        "oracle_quarantine": ORACLE_QUARANTINE,
        "artifacts": ARTIFACTS,
        "run_order": list(RUN_ORDER),
        "boundary": BOUNDARY,
    }


def assert_frozen() -> dict[str, Any]:
    """Re-derive what can be checked, so a drifted runner fails loudly."""
    failures: list[str] = []
    inherited = v1.assert_frozen()
    if not inherited["frozen"]:
        failures.extend("v1: %s" % item for item in inherited["failures"])

    from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
    from SelfEvolvingHarnessTS.methods.ttha import online_loop as loop

    router = FaultRouter()
    try:
        router.authorize("RISK_GAP", target_class="capability",
                         operation="ADD", skill_kind="capability")
    except (ValueError, KeyError) as exc:
        failures.append("RISK_GAP cannot ADD a capability Skill: %s" % exc)

    if narrowing.MAX_ADDED_CLAUSES != 1:
        failures.append("the revision class widened beyond one added clause")
    if set(loop.RISK_REFUSAL_REASONS) != {
            "harmed_fraction_over_budget", "single_series_harm_over_budget"}:
        failures.append("the risk-refusal reasons drifted")
    if DELAYED_ADMISSION["max_harmed_fraction"] != bounded.BOUNDED_MAX_HARMED_FRACTION:
        failures.append("the harmed-fraction budget drifted from P4b")
    if (DELAYED_ADMISSION["max_single_series_harm"]
            != bounded.BOUNDED_MAX_SINGLE_SERIES_HARM):
        failures.append("the single-series harm budget drifted from P4b")

    return {
        "frozen": not failures,
        "failures": failures,
        "version": VERSION,
        "v1_frozen": inherited["frozen"],
        "risk_gap_may_add_a_capability_skill": "RISK_GAP cannot ADD" not in " ".join(
            failures),
        "max_added_clauses": narrowing.MAX_ADDED_CLAUSES,
    }
