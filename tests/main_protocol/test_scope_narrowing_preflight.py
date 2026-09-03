"""The only thing standing between a Slow revision and a wider Scope.

Nothing else in the chain checks the content of a revised ``serving_scope``:
the route table's monotone rule is a target-class gate that never sees the
predicate, and it does not apply to ``RISK_GAP`` at all.  So these cases are
adversarial on purpose -- every way a "narrowing" can secretly widen.

0 LLM, no evaluation: predicates and feature cards only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.main_protocol_p4 import (  # noqa: E402
    scope_narrowing_preflight as preflight,
)

Z = {"feature": "local_robust_z_peak", "op": ">=", "threshold": 3.0}
GAP = {"feature": "missing_fraction", "op": ">=", "threshold": 0.05}
EXTRA = {"feature": "period_reliability", "op": ">", "threshold": 0.128}

ORIGINAL = {"scope_type": "serving_series_predicate", "predicate": [Z]}

#: Four served series: two clear the added clause, two do not.
FEATURES = {
    "s0": {"local_robust_z_peak": 9.0, "period_reliability": 0.90,
           "missing_fraction": 0.0},
    "s1": {"local_robust_z_peak": 5.0, "period_reliability": 0.50,
           "missing_fraction": 0.0},
    "s2": {"local_robust_z_peak": 4.0, "period_reliability": 0.01,
           "missing_fraction": 0.0},
    "s3": {"local_robust_z_peak": 1.0, "period_reliability": 0.99,
           "missing_fraction": 0.0},
}
AVAILABLE = sorted(FEATURES["s0"])


def _scope(*clauses):
    return {"scope_type": "serving_series_predicate", "predicate": list(clauses)}


def test_a_genuine_narrowing_is_accepted_and_says_what_it_excluded():
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(Z, EXTRA), features=FEATURES,
        available_features=AVAILABLE)
    assert verdict.accepted
    assert verdict.reason == "strictly_narrower"
    # s0,s1,s2 clear z>=3; only s0,s1 clear the added clause.
    assert (verdict.original_resolved, verdict.proposed_resolved) == (3, 2)
    assert verdict.excluded_series == 1
    assert verdict.checks["resolves_to_a_subset"]
    assert verdict.checks["excludes_at_least_one_series"]


def test_relaxing_the_original_threshold_is_refused():
    """The sneakiest widening: same feature, looser bound, looks like an edit."""
    loosened = dict(Z, threshold=2.0)
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(loosened), features=FEATURES,
        available_features=AVAILABLE)
    assert not verdict.accepted
    assert "dropped or rewritten" in verdict.reason
    assert verdict.checks["keeps_every_original_clause"] is False


def test_dropping_the_original_clause_is_refused():
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(EXTRA), features=FEATURES, available_features=AVAILABLE)
    assert not verdict.accepted
    assert verdict.checks["keeps_every_original_clause"] is False


def test_more_than_one_added_clause_is_refused():
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(Z, EXTRA, GAP), features=FEATURES,
        available_features=AVAILABLE)
    assert not verdict.accepted
    assert verdict.checks["adds_at_most_one_clause"] is False
    assert len(verdict.added_clauses) == 2


def test_an_unchanged_scope_is_refused_as_a_non_revision():
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(Z), features=FEATURES, available_features=AVAILABLE)
    assert not verdict.accepted
    assert "has not responded to the evidence" in verdict.reason


def test_a_clause_that_excludes_nobody_here_is_refused():
    """Narrower on paper, identical in effect: it does not answer the refusal."""
    inert = {"feature": "local_robust_z_peak", "op": ">=", "threshold": 3.5}
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(Z, inert), features=FEATURES,
        available_features=AVAILABLE)
    assert not verdict.accepted
    assert verdict.checks["excludes_at_least_one_series"] is False
    assert verdict.excluded_series == 0


@pytest.mark.parametrize("kind", ["none", "all_serving_series"])
def test_abstention_and_the_global_scope_are_not_narrowings(kind):
    """Treating nothing clears every risk budget; it must not enter this way."""
    verdict = preflight.validate_narrowing(
        ORIGINAL, {"scope_type": kind, "predicate": []}, features=FEATURES)
    assert not verdict.accepted
    assert "must stay a predicate" in verdict.reason


def test_a_clause_the_deployment_cannot_observe_is_refused():
    unseen = {"feature": "future_error", "op": "<=", "threshold": 1.0}
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(Z, unseen), features=FEATURES,
        available_features=AVAILABLE)
    assert not verdict.accepted
    assert "cannot observe" in verdict.reason


def test_a_clause_naming_a_series_is_refused_at_construction():
    named = {"feature": "T260", "op": ">=", "threshold": 1.0}
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(Z, named), features=FEATURES)
    assert not verdict.accepted
    assert verdict.reason.startswith("proposed_scope_is_not_a_legal_spec")
    assert "may not name a series" in verdict.reason


def test_without_features_the_verdict_does_not_claim_a_measurement():
    verdict = preflight.validate_narrowing(ORIGINAL, _scope(Z, EXTRA))
    assert verdict.accepted
    assert verdict.reason == "structurally_narrower_semantics_unchecked"
    assert verdict.original_resolved is None
    assert "resolves_to_a_subset" not in verdict.checks


def test_the_structural_rule_is_what_transfers_to_another_cohort():
    """An accepted narrowing stays a narrowing on a cohort it never saw."""
    other = {
        "t0": {"local_robust_z_peak": 99.0, "period_reliability": 0.2,
               "missing_fraction": 0.9},
        "t1": {"local_robust_z_peak": 3.1, "period_reliability": 0.5,
               "missing_fraction": 0.0},
    }
    verdict = preflight.validate_narrowing(
        ORIGINAL, _scope(Z, EXTRA), features=other, available_features=AVAILABLE)
    # It may or may not exclude anyone there, but it can never reach further.
    assert verdict.checks["resolves_to_a_subset"]


def test_a_frozen_program_passes_and_any_edit_to_it_does_not():
    steps = [{"op": "outlier_mad", "params": {"k": 3}}]
    assert preflight.validate_program_frozen(steps, list(steps)).accepted
    assert not preflight.validate_program_frozen(
        steps, [{"op": "hampel_filter", "params": {"k": 3}}]).accepted
    assert not preflight.validate_program_frozen(
        steps, [{"op": "outlier_mad", "params": {"k": 4}}]).accepted


def test_parameter_order_does_not_count_as_a_program_change():
    assert preflight.validate_program_frozen(
        [{"op": "outlier_mad", "params": {"k": 3, "w": 5}}],
        [{"op": "outlier_mad", "params": {"w": 5, "k": 3}}]).accepted
