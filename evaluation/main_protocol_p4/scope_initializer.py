"""Where a Scope comes from when the Fast Agent proposes only a Program.

The Fast schema has no Scope channel, and this round deliberately does not add
one: a field no instruction mentions would be dead weight and a fresh snapshot
difference.  The semantics already proven end-to-end are enough --

    Fast proposes a Program
    -> the Runtime initialises a Scope from deployment-visible features
    -> Slow revises that Scope from held-in feedback

-- and it is the Harness as a whole, not the Fast call, that the project claims
forms and revises Scopes.

The initialiser is a frozen table, not a heuristic that can drift.  Each operator
family names the defect it exists to repair, and the Scope selects the series
that actually show that defect.  A two-step program conjoins its steps' clauses,
so ``period_median_complete>winsorize`` treats the series that are both gappy and
spiky, and declines the rest.

Three constraints make it auditable rather than merely plausible:

* it reads only deployment-visible features, taken from the pre-origin context;
  no UID, no cohort name, no Outcome can reach it, which the signature enforces
  by giving it nothing else;
* the thresholds are domain constants -- 5% missing, robust-z 3.0 -- chosen
  before any Target feature was inspected, so they are not fitted to the cohort
  they will be applied to;
* an operator with no declared defect family falls back to
  ``all_serving_series`` and says so, rather than silently narrowing to nothing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evaluation.main_protocol_p4 import scope_spec as scopes

#: Operator family -> the deployment-visible evidence that its defect is present.
#: Thresholds are canonical domain lines, not values tuned on any cohort:
#: 5% is the usual materiality line for missingness, and 3.0 is the standard
#: robust-z outlier cut.
FAMILY_CLAUSES: Mapping[str, tuple[str, str, float]] = {
    # completion: only treat a series that actually has gaps
    "impute_linear": ("missing_fraction", ">=", 0.05),
    "impute_fft": ("missing_fraction", ">=", 0.05),
    "impute_ema": ("missing_fraction", ">=", 0.05),
    "impute_ar": ("missing_fraction", ">=", 0.05),
    "period_complete": ("missing_fraction", ">=", 0.05),
    "period_median_complete": ("missing_fraction", ">=", 0.05),
    # outlier / clipping: only treat a series that actually has a spike
    "outlier_iqr": ("local_robust_z_peak", ">=", 3.0),
    "outlier_mad": ("local_robust_z_peak", ">=", 3.0),
    "winsorize": ("local_robust_z_peak", ">=", 3.0),
    "hampel_filter": ("local_robust_z_peak", ">=", 3.0),
    # level repair: only treat a series showing a level excursion
    "repair_level_shift": ("level_excursion_score", ">=", 0.5),
    "repair_burst_segment": ("level_excursion_score", ">=", 0.5),
}

FALLBACK = scopes.ScopeSpec("all_serving_series")

SCOPE_SOURCES = ("runtime_initializer", "source_skill", "slow_revision")


def initialize(steps: Sequence[Mapping[str, Any]] | Sequence[tuple]) -> dict[str, Any]:
    """The Scope this Program starts with, and why.

    Returns the ScopeSpec together with its provenance, because a deployment
    that cannot say where its Scope came from cannot be audited: the same
    predicate means something different if the Runtime proposed it, a Source
    Skill carried it, or Slow revised it.
    """
    ops = []
    for step in steps:
        if isinstance(step, Mapping):
            ops.append(str(step.get("op")))
        elif isinstance(step, (tuple, list)) and step:
            ops.append(str(step[0]))
    clauses: list[scopes.Clause] = []
    unmapped: list[str] = []
    for op in ops:
        entry = FAMILY_CLAUSES.get(op)
        if entry is None:
            unmapped.append(op)
            continue
        clause = scopes.Clause(*entry)
        if clause not in clauses:
            clauses.append(clause)
    if not clauses:
        return {
            "scope": FALLBACK.to_dict(),
            "scope_source": "runtime_initializer",
            "rule": "no step declares a defect family",
            "unmapped_operators": unmapped,
            "narrowed": False,
        }
    spec = scopes.ScopeSpec("serving_series_predicate", tuple(clauses))
    return {
        "scope": spec.to_dict(),
        "scope_source": "runtime_initializer",
        "rule": "conjunction of each step's declared defect evidence",
        "description": spec.describe(),
        "unmapped_operators": unmapped,
        "narrowed": True,
    }


def resolve(scope: Mapping[str, Any],
            features: Mapping[str, Mapping[str, float]]) -> frozenset[str]:
    """Resolve a Scope against served series' own deployment-visible features."""
    return scopes.ScopeSpec.from_dict(dict(scope)).resolve(features)


def declared_rules() -> dict[str, Any]:
    """The frozen contract this initialiser answers to."""
    return {
        "reads_only_deployment_visible_features": True,
        "forbidden_inputs": ["series UID", "cohort name", "any Outcome"],
        "family_table": {
            op: {"feature": feature, "op": operator, "threshold": threshold}
            for op, (feature, operator, threshold) in FAMILY_CLAUSES.items()
        },
        "fallback": FALLBACK.to_dict(),
        "identical_for_a3_and_a5": True,
        "runner_may": ["supply a Scope", "resolve a Scope"],
        "runner_may_not": [
            "hand-pick a Scope after seeing the result",
            "vary the initialiser between arms",
        ],
        "scope_sources_recorded": list(SCOPE_SOURCES),
        "thresholds_are_domain_constants": (
            "5% missingness materiality and robust-z 3.0, fixed before any "
            "Target feature was inspected"
        ),
    }
