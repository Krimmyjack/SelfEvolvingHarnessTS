"""A Scope must transfer, must resolve from features alone, and must abstain."""
from __future__ import annotations

import pytest

from evaluation.main_protocol_p4 import scope_spec as scopes

FEATURES = {
    "T1": {"missing_fraction": 0.30, "seasonal_strength": 0.80},
    "T2": {"missing_fraction": 0.05, "seasonal_strength": 0.90},
    "T3": {"missing_fraction": 0.40, "seasonal_strength": 0.10},
}
AVAILABLE = ("missing_fraction", "seasonal_strength", "trend_strength")


def _gapped_and_seasonal() -> scopes.ScopeSpec:
    return scopes.ScopeSpec(
        "serving_series_predicate",
        (scopes.Clause("missing_fraction", ">=", 0.10),
         scopes.Clause("seasonal_strength", ">=", 0.50)),
    )


def test_a_predicate_selects_only_the_series_that_satisfy_every_clause():
    assert _gapped_and_seasonal().resolve(FEATURES) == frozenset({"T1"})


def test_all_and_none_are_the_two_degenerate_scopes():
    assert scopes.ALL.resolve(FEATURES) == frozenset(FEATURES)
    assert scopes.NONE.resolve(FEATURES) == frozenset()


def test_a_scope_may_not_name_a_series():
    # A Skill that remembers UIDs cannot transfer: in the next Target those
    # names do not exist and the Scope quietly resolves to nothing.
    with pytest.raises(scopes.ScopeError, match="may not name a series"):
        scopes.Clause("T140", ">=", 0.1)


def test_a_scope_naming_an_unobservable_feature_is_refused_up_front():
    # Refused at validation rather than resolving to an empty set at run time,
    # where it would be indistinguishable from a deliberate abstention.
    spec = scopes.ScopeSpec(
        "serving_series_predicate",
        (scopes.Clause("future_error", "<=", 1.0),),
    )
    with pytest.raises(scopes.ScopeError, match="cannot observe"):
        spec.validate_against(AVAILABLE)


def test_a_valid_scope_passes_validation():
    _gapped_and_seasonal().validate_against(AVAILABLE)


def test_a_predicate_scope_needs_a_clause_and_the_others_forbid_one():
    with pytest.raises(scopes.ScopeError):
        scopes.ScopeSpec("serving_series_predicate")
    with pytest.raises(scopes.ScopeError):
        scopes.ScopeSpec("all_serving_series", (scopes.Clause("x", ">=", 1.0),))


def test_a_scope_round_trips_through_its_stored_form():
    spec = _gapped_and_seasonal()
    assert scopes.ScopeSpec.from_dict(spec.to_dict()) == spec


def test_resolution_reads_features_and_nothing_else():
    # The signature is the guarantee: resolve() is handed feature cards, so no
    # Outcome can participate even by accident.
    spec = _gapped_and_seasonal()
    assert spec.resolve({}) == frozenset()
    partial = {"T1": {"missing_fraction": 0.30}}  # missing a clause's feature
    assert spec.resolve(partial) == frozenset()


def test_the_resolved_uid_set_is_episode_evidence_not_a_skill_field():
    spec = _gapped_and_seasonal()
    resolved = spec.resolve(FEATURES)
    record = scopes.execution_record(spec, resolved, list(FEATURES))
    assert record["is_skill_field"] is False
    assert record["resolved_training_series"] == ["T1"]
    assert record["coverage"] == pytest.approx(1 / 3, abs=1e-4)
    # The stored Skill form carries the predicate, never the resolution.
    assert "resolved_training_series" not in record["scope"]


def test_describe_is_readable_enough_to_audit():
    assert _gapped_and_seasonal().describe() == (
        "missing_fraction >= 0.1 AND seasonal_strength >= 0.5"
    )
