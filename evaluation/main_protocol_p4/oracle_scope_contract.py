"""A0: the rules the oracle-scope feasibility bound answers to, frozen first.

The Source line refused every materially positive proposal on the tail budget
(``p4w``).  Before spending Slow budget on revising those Scopes, one question
has to be settled: *could any Scope in the class Slow is allowed to write have
passed?*  If not, a Slow failure would be uninterpretable -- indistinguishable
from a weak agent -- and the calls would be wasted.

The audit that answers it deliberately cheats: it reads the per-series Outcomes
before choosing a predicate.  That makes it an **upper bound**, never a policy,
and it is the reason this file exists separately from the audit.  A bound whose
rules are written after its results are seen bounds nothing, so the rules are
frozen here and the audit imports them.

Why narrowing is exact rather than estimated
--------------------------------------------
``scoped_serving_evaluator.scoped_evaluate`` builds its training corpus from
``role == "train"`` rows and never consults the scope; the scope only decides
which **eval** row takes the Program pipeline instead of the Raw one.  So the
Program model is scope-independent, and removing a served series from the scope
moves it to ``raw train -> raw model -> raw serve context`` -- bit-identical to
Static, gain exactly ``0.0``.  Every other series keeps the reading it already
has.  Re-scoring a subset is therefore arithmetic on the recorded per-series
gains, not a re-estimate, and it needs no model fit, no LLM call and no new
evaluation.

That property is also why the class below may only ever **narrow**.  A series
outside the original Scope was never run through the Program, so the artifact
holds no counterfactual for it; widening would require inventing a number.
``ScopeSpec.resolve`` conjoins its clauses, so "the original predicate AND one
more clause" is a subset by construction -- the grammar enforces the rule, not
just this audit.
"""
from __future__ import annotations

from typing import Any

from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import scope_spec as scopes
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

#: Only the probes the fault router should have routed and did not: materially
#: positive in aggregate, refused by the gate on a tail-risk clause.  A probe
#: refused for being a no-op is a different fault and is not in this audit.
ELIGIBLE_REFUSAL_REASONS = (
    "harmed_fraction_over_budget",
    "single_series_harm_over_budget",
)

#: The class of Scope the audit is allowed to search.  Wider than what Slow can
#: reliably find, narrower than "any subset" -- so that a negative result is a
#: statement about the class and not about the search.
REVISION_CLASS = {
    "form": "the probe's own Scope, conjoined with at most one new clause",
    "max_added_clauses": 1,
    "may_widen": False,
    "why_not": (
        "a series outside the original Scope never ran through the Program, so "
        "no counterfactual for it exists in the artifact; widening would have "
        "to invent one"
    ),
    "operators": list(scopes.OPERATORS),
    "feature_vocabulary": (
        "the numeric keys of extract_public_features on the strictly pre-origin "
        "window, i.e. exactly what scope_initializer and the live resolver read"
    ),
    "forbidden": [
        "naming a series UID, which ScopeSpec refuses at construction",
        "any feature not visible at deployment time",
        "any Outcome as a predicate input",
    ],
}

#: Every line has to hold at once.  The three risk/utility lines are the frozen
#: adjudicated ones, read from their modules rather than restated, so this file
#: cannot drift from the gate it claims to model.
FEASIBILITY_LINES = {
    "min_aggregate_gain": admission_policy.MATERIAL_THRESHOLD,
    "max_harmed_fraction": bounded.BOUNDED_MAX_HARMED_FRACTION,
    "max_single_series_harm": bounded.BOUNDED_MAX_SINGLE_SERIES_HARM,
    "min_treated_series": 5,
    "served_series": 20,
    "why_a_coverage_floor": (
        "narrowing to almost nothing passes the risk budget trivially and "
        "deploys nothing; a floor declared before the run is what stops a "
        "degenerate abstention from being read as a success"
    ),
    "thresholds_changed": 0,
}

#: The denominators, restated so the audit cannot quietly pick a kinder basis.
SCORING_BASIS = {
    "aggregate_gain": "mean over all served series, excluded ones counted as 0.0",
    "harmed_fraction": "harmed / served, matching online_loop._risk_profile",
    "harmed_series": "per-series gain < -%g" % admission_policy.MATERIAL_THRESHOLD,
    "max_single_series_harm": "-min(per-series gain), floored at 0.0",
    "coverage": "treated / served, matching the contract's coverage semantics",
}

#: The audit reads Outcomes.  Nothing it finds may re-enter the experiment.
LEAKAGE_GUARD = {
    "the_bound_is_not_a_policy": (
        "it selects its predicate with knowledge of the per-series gains; "
        "deploying it would be selection on the Outcome"
    ),
    "must_not_be_fed_back": [
        "the feature or threshold the oracle selected",
        "the identity of the excluded series",
        "the ranking of candidate clauses",
    ],
    "must_not_change": [
        "the Slow prompt or any instruction text",
        "the candidate ordering or the programs probed",
        "scope_initializer's family table or thresholds",
        "the admission thresholds",
    ],
    "what_may_be_carried_forward": (
        "one bit -- whether the frozen class contains a feasible Scope -- which "
        "decides whether Slow budget is spent at all"
    ),
}

#: Two readings, and only the first decides anything.
VERDICTS = {
    "FEASIBLE_SCOPE_EXISTS": (
        "at least one probe admits a Scope in the frozen class that clears all "
        "four lines; Slow's task is well-posed and a later Slow failure is "
        "attributable to Slow rather than to the class"
    ),
    "NO_FEASIBLE_SCOPE_IN_FROZEN_CLASS": (
        "no probe does; stop before spending LLM budget.  This does NOT by "
        "itself say the Observation vocabulary is insufficient -- the same null "
        "is produced by a Program that is simply wrong for this data, and by "
        "the Scope's coupling of two decisions (whether to clean a served "
        "context, and whether to serve that series from the Program-fitted "
        "model), which one predicate cannot separate.  T30 is the standing "
        "counter-example: harmed with zero points modified, purely by routing"
    ),
}

#: Computed alongside the verdict, and deliberately outside the deployable
#: class: the best subset of the original Scope reachable by naming series
#: directly.  It is not a candidate policy and never becomes one -- it exists so
#: that a null can be read correctly.  If even arbitrary UID-level selection
#: cannot clear the lines, the gain and the harm live in the same series and no
#: predicate over any feature vocabulary could have separated them; if it can,
#: the shortfall is in the class or the vocabulary, which are different faults.
DIAGNOSTIC_UNCONSTRAINED_BOUND = {
    "role": "interpretation only, never a verdict input",
    "deployable": False,
    "why_not": "it names series, which ScopeSpec refuses by construction",
}

BOUNDARY = {
    "llm_calls": 0,
    "new_evaluations": 0,
    "consumer_fits": 0,
    "held_out_reads": 0,
    "thresholds_changed": 0,
    "operators_added": 0,
    "artifacts_overwritten": 0,
}


def to_dict() -> dict[str, Any]:
    """The contract as a receipt, embedded verbatim by the audit."""
    return {
        "stage": "P4Y0_ORACLE_SCOPE_CONTRACT",
        "frozen_before": "any oracle result was computed",
        "question": (
            "does the class of Scope revision Slow is allowed to write contain "
            "one that would have cleared the tail budget on these probes"
        ),
        "eligible_refusal_reasons": list(ELIGIBLE_REFUSAL_REASONS),
        "revision_class": REVISION_CLASS,
        "feasibility_lines": FEASIBILITY_LINES,
        "scoring_basis": SCORING_BASIS,
        "leakage_guard": LEAKAGE_GUARD,
        "verdicts": VERDICTS,
        "diagnostic_unconstrained_bound": DIAGNOSTIC_UNCONSTRAINED_BOUND,
        "boundary": BOUNDARY,
        "exactness": (
            "scoped_evaluate builds its training corpus from role=='train' rows "
            "and never reads the scope, so the Program model is scope-"
            "independent and an excluded served series scores exactly 0.0 "
            "against Static; re-scoring a subset is arithmetic on recorded "
            "per-series gains, not a re-estimate"
        ),
    }
