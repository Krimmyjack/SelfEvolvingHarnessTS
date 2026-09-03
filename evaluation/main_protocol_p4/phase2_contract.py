"""The frozen Phase-2 contract: O0/O1 x X0/X1 on two cohorts, five origins.

Everything Phase 2 is allowed to decide is written here **before** any Outcome
on the confirmation cohort is read, so that a later script cannot quietly widen
the menu, deepen the tree or add an origin.  Downstream runners import these
declarations rather than restating them; ``assert_frozen`` re-derives the parts
that can be checked mechanically.

The geometry, and why it is this one.  P4D/P4G swept six origins over a single
training corpus -- anchors are frozen at ``[312 ... 852]`` and all of them clear
``anchor + 48 <= origin`` past 900 -- so one fitted Ridge was scored at six
forecast windows.  Changing the origin does not change the training condition;
only changing the **series** does.  Phase 2 therefore moves the cohort and holds
the anchors fixed, so that a difference is attributable to the training corpus
rather than to the time geometry.

Train/confirm separation is the load-bearing discipline:

* **cohort 1** (``[:20]`` / ``[20:40]``) is where the O1 menu, the X1 feature
  set, the Targeter and Best-Fixed are chosen.  It has already been read.
* **cohort 2** (``[40:60]`` / ``[60:80]``) is read **once**, to confirm.  No
  program may be reselected, no tree retrained and no feature added on its
  Outcome.  If that rule is broken, cohort 2 becomes another tuning set and the
  whole comparison loses its meaning.

Cohort 2 is labelled ``DEVELOPMENT_CONFIRMATION``, never fresh and never
held-out: it served as the Best-Fixed selection cell in the old P1 on the
without-missing release at origin 600, so it is an already-exposed development
pool being re-read under a different data version and different origins.
"""
from __future__ import annotations

from typing import Any, Mapping

from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight

DATA_VERSION = preflight.DATA_VERSION
CONTEXT, HORIZON = preflight.CONTEXT, preflight.HORIZON
FACES = ("support_a", "support_b")

# ---------------------------------------------------------------- geometry ---

#: Frozen before any confirmation read.  These are the origins on which *both*
#: cohorts are evaluable; 1896 is excluded because a cohort-2 series has no
#: observed truth in that horizon, and 3576-4536 are excluded as extreme-gap
#: stress points reserved for a separate experiment.
ORIGINS = (1176, 2136, 2376, 2616, 2856)

#: Origins are repeated evaluation windows inside one cohort, **not** five
#: independent training replications.  The statistical unit is cohort x face.
STATISTICAL_UNIT = "training cohort x face"

COHORTS: Mapping[str, Mapping[str, Any]] = {
    "cohort_1": {
        "slice": "readable[0:20] / readable[20:40]",
        "role": "DEVELOPMENT",
        "may_be_read_repeatedly": True,
        "chooses": ["O1 menu", "X1 feature set", "Targeter", "Best-Fixed"],
    },
    "cohort_2": {
        "slice": "readable[40:60] / readable[60:80]",
        "role": "DEVELOPMENT_CONFIRMATION",
        "may_be_read_repeatedly": False,
        "chooses": [],
        "not_fresh_because": (
            "it served as the Best-Fixed selection cell in the old P1 on the "
            "without-missing release at origin 600; it is an already-exposed "
            "development pool, re-read under a different data version"
        ),
    },
}

#: The anchor axis is deliberately not moved.  Changing cohort and anchors at
#: once would leave any effect unattributable, and a new anchor block would also
#: require rerunning the structural readability filter, which is a new geometry
#: rather than a parameter change.
ANCHOR_AXIS_FROZEN = True
ANCHOR_BLOCK_IS_A_SEPARATE_EXPERIMENT = (
    "temporal robustness, only after Phase 2 concludes"
)

# ------------------------------------------------------------ program space ---

#: O0 is the repair/completion space exactly as P4C/P4D enumerated it -- no
#: operator is added and no parameter grid is changed.
O0 = {
    "space": "audit_program_repairability.program_space(two_step=True)",
    "operators_added": 0,
    "parameter_grid_changed": False,
    "parameter_corrections": "audit_param_correction_rerun.PARAM_CORRECTIONS",
}

#: Aliases must not vote twice.  Four of the eighteen operators are no-ops on
#: gapped data, so many of the 396 enumerated programs share an effect vector;
#: menuing them separately would spend the menu budget on duplicates and would
#: let one discovery appear as several.  The rule is applied to cohort 1's
#: already-collected tensor, which is development data by construction.
O0_DEDUPLICATION = {
    "rule": (
        "distinct by the exact per-series gain signature over the frozen five "
        "origins x two faces; a program unreadable at any of those ten cells is "
        "excluded rather than imputed"
    ),
    "applied_to": "cohort 1 only",
    "distinct_programs": 68,
    "plus_the_empty_program": True,
}

#: Every arm is scored against one fixed reference so the four view rows stay
#: commensurable: the empty program under the identity view -- literally "raw".
GAIN_REFERENCE = "empty program, identity view, same origin and face"

#: O1 is O0 evaluated inside a reversible representation view.  The views are
#: exactly those the representation preflight admitted; ``reversible_scale`` is
#: excluded because the Consumer's own median/MAD standardisation cancels any
#: positive affine map exactly, which makes it a provable no-op.
O1_VIEWS = ("identity", "reversible_detrend", "reversible_seasonal_adjust",
            "reversible_difference")
O1_EXCLUDED_VIEWS = {
    "reversible_scale": (
        "positive affine in the values; erased exactly by _center_scale "
        "(median + 1.4826*MAD are affine-equivariant)"
    ),
    "temporal_grid": "changes the horizon semantics; separate accounting",
    "window_align": "changes training-sample geometry and volume; separate",
    "impute_with_missingness": "changes the Consumer input interface; separate",
}

#: The evaluator that closes the view around the Consumer.  The frozen O0 path
#: is untouched and reproduces bit-identically under the identity view.
EVALUATOR = {
    "o0_path": "run_e2_autonomous_natural_workflow_generation._evaluate",
    "o1_path": "representation_view.representation_evaluate",
    "identity_equivalence_gate": (
        "artifacts/main_protocol/p4i_representation_evaluator_preflight.json"
    ),
    "frozen_o0_evaluator_modified": False,
}

# ----------------------------------------------------------------- features ---

#: X0 is the current deployment-visible PatternCard, unchanged.
X0 = "public_tools.extract_public_features numeric fields"

#: X1 is X0 plus a small, pre-declared natural-structure set.  It is frozen
#: here as a hypothesis, not as a diagnosis: the closure document records that
#: the binding gate is maximum single-series harm and that the failure cannot be
#: uniquely attributed to features, model capacity or sample size.
X1_ADDITIONS = (
    "trend_strength",
    "seasonal_strength",
    "spectral_entropy",
    "acf_lag_1",
    "acf_at_period",
    "volatility_drift",
)

# ----------------------------------------------------------------- targeter ---

#: Identical to the P4G diagnostic so the two readings are comparable.  Depth
#: and leaf size are frozen; no search over them is authorised.
TARGETER = {
    "estimator": "sklearn.tree.DecisionTreeClassifier",
    "max_depth": 3,
    "min_samples_leaf": 2,
    "random_state": 0,
    "menu_size": 6,
    "menu_always_includes_raw": True,
    "hyperparameter_search": "NOT_AUTHORISED",
}

# ------------------------------------------------------------------ endpoint ---

#: Oracle quantities are upper-bound diagnostics only.  A run does not succeed
#: because an oracle is large.
PRIMARY_ENDPOINT = {
    "utility_vs_raw": "mean per-series gain must exceed +0.005",
    "utility_vs_best_fixed": "must exceed the frozen Best-Fixed on cohort 2",
    "deployment_coverage": "must be non-zero and is reported, not maximised",
    "harmed_fraction": "<= %.2f" % bounded.BOUNDED_MAX_HARMED_FRACTION,
    "max_single_series_harm": "<= %.2f" % bounded.BOUNDED_MAX_SINGLE_SERIES_HARM,
    "cost": "Consumer fits and wall seconds are reported for every arm",
}
ORACLE_IS_NOT_A_RESULT = (
    "per-series oracle bounds what a perfect selector could reach; it is never "
    "reported as an outcome"
)

ARMS = (
    "raw",
    "best_fixed_O0", "best_fixed_O1",
    "targeter_X0_O0", "targeter_X1_O0",
    "targeter_X0_O1", "targeter_X1_O1",
)

BOUNDARY = {
    "llm_calls": 0,
    "held_out_reads": 0,
    "ucr_test_outcome_reads": 0,
    "natural_final_outcome_reads": 0,
    "thresholds_changed": 0,
    "operators_added": 0,
}


def assert_frozen() -> dict[str, Any]:
    """Re-derive what can be checked, so a drifted runner fails loudly."""
    failures: list[str] = []
    if len(ORIGINS) != 5 or 1896 in ORIGINS:
        failures.append("origin list drifted from the frozen five")
    if TARGETER["max_depth"] != 3 or TARGETER["min_samples_leaf"] != 2:
        failures.append("Targeter structure drifted")
    if "reversible_scale" in O1_VIEWS:
        failures.append("a provably cancelled view entered the O1 menu")
    if "identity" not in O1_VIEWS:
        failures.append("O1 must contain the identity view so O0 nests inside it")
    if COHORTS["cohort_2"]["may_be_read_repeatedly"]:
        failures.append("the confirmation cohort was marked re-readable")
    if len(X1_ADDITIONS) != 6:
        failures.append("the X1 addition set drifted")
    return {
        "frozen": not failures,
        "failures": failures,
        "origins": list(ORIGINS),
        "views": list(O1_VIEWS),
        "arms": list(ARMS),
        "statistical_unit": STATISTICAL_UNIT,
        "data_version": DATA_VERSION,
    }
