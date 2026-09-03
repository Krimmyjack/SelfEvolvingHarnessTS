"""The bounded-risk admission rule, and the default's equivalence to the old gate.

The default has to stay bit-equivalent to the inline ``relation == "POSITIVE"``
test that granted deployment rights before P4b parameterised it, otherwise the
``A5-strict`` arm is no longer the old policy and the strict/bounded contrast
means nothing.
"""
from __future__ import annotations

import pytest

from SelfEvolvingHarnessTS.methods.ttha import admission_policy as ap
from SelfEvolvingHarnessTS.methods.ttha import exploration_policy


RELATIONS = ("POSITIVE", "CONFLICT", "NEGATIVE", "NEUTRAL", "ABSTAIN")


@pytest.fixture(autouse=True)
def _reset_policy():
    yield
    ap.reset_policy()


def test_default_is_strict_and_reproduces_the_inline_predicate():
    assert ap.active_policy() == ap.DEFAULT
    assert ap.DEFAULT.rule == ap.STRICT
    for relation in RELATIONS:
        verdict = ap.decide(
            relation=relation,
            aggregate_gain=0.3,
            per_series_gains=[0.9, -0.9],  # would matter only under bounded
        )
        assert verdict.admitted == (relation == "POSITIVE"), relation


def test_admission_is_not_in_the_random_legal_edit_sampling_space():
    # exploration_policy.LEGAL_DOMAINS is the Stage-3 Random-legal-edit arm's
    # sampling space; a harm gate must not be reachable from it.
    for field in ("rule", "max_harmed_fraction", "max_single_series_harm"):
        assert field not in exploration_policy.LEGAL_DOMAINS
    assert not hasattr(exploration_policy.ExplorationPolicy(), "rule")


def test_bounded_admits_conflict_inside_the_budget():
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20,
        max_single_series_harm=0.30))
    verdict = ap.decide(
        relation="CONFLICT", aggregate_gain=0.30,
        per_series_gains=[0.5] * 16 + [-0.1] * 4)
    assert verdict.admitted
    assert verdict.reason == "within_risk_budget"
    assert verdict.harmed_count == 4
    assert verdict.harmed_fraction == pytest.approx(0.20)
    assert verdict.max_single_series_harm == pytest.approx(0.1)


@pytest.mark.parametrize(
    "per_series, reason",
    [
        ([0.5] * 15 + [-0.1] * 5, "harmed_fraction_over_budget"),
        ([0.5] * 19 + [-0.4], "single_series_harm_over_budget"),
    ],
)
def test_bounded_rejects_outside_the_budget(per_series, reason):
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20,
        max_single_series_harm=0.30))
    verdict = ap.decide(
        relation="CONFLICT", aggregate_gain=0.30, per_series_gains=per_series)
    assert not verdict.admitted
    assert verdict.reason == reason


def test_bounded_fails_closed_without_a_per_series_reading():
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20,
        max_single_series_harm=0.30))
    verdict = ap.decide(
        relation="CONFLICT", aggregate_gain=0.30, per_series_gains=None)
    assert not verdict.admitted
    assert verdict.reason == "no_per_series_reading_fail_closed"


def test_bounded_still_requires_the_aggregate_to_clear_the_material_line():
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=1.0,
        max_single_series_harm=99.0))
    verdict = ap.decide(
        relation="NEGATIVE", aggregate_gain=-0.30, per_series_gains=[-0.3])
    assert not verdict.admitted
    assert verdict.reason == "aggregate_below_material_line"


def test_bounded_never_narrows_what_strict_admits():
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.0,
        max_single_series_harm=0.0))
    # A POSITIVE relation already means "no series materially harmed", so the
    # relaxed rule must admit everything the strict rule would, at any budget.
    assert ap.decide(
        relation="POSITIVE", aggregate_gain=0.3, per_series_gains=None).admitted


def test_illegal_policies_are_rejected_at_install():
    for bad in (
        ap.AdmissionPolicy(rule="anything_goes"),
        ap.AdmissionPolicy(rule=ap.BOUNDED_V1, max_harmed_fraction=1.5),
        ap.AdmissionPolicy(rule=ap.BOUNDED_V1, max_single_series_harm=-1.0),
    ):
        with pytest.raises(ValueError):
            ap.install_policy(bad)


def _facts(relation, aggregate, series_read, harmed, lowest):
    """The shape classify_relation hands the method layer."""
    return {
        "relation": relation,
        "aggregate_gain": aggregate,
        "series_read": series_read,
        "harmed_series_count": harmed,
        "min_per_series_gain": lowest,
    }


def test_decide_from_facts_matches_decide_under_both_rules():
    within = ([0.5] * 16 + [-0.1] * 4, _facts("CONFLICT", 0.3, 20, 4, -0.1))
    over = ([0.5] * 15 + [-0.1] * 5, _facts("CONFLICT", 0.3, 20, 5, -0.1))
    for policy in (
        ap.DEFAULT,
        ap.AdmissionPolicy(rule=ap.BOUNDED_V1, max_harmed_fraction=0.20,
                           max_single_series_harm=0.30),
    ):
        for per_series, facts in (within, over):
            direct = ap.decide(
                relation=facts["relation"],
                aggregate_gain=facts["aggregate_gain"],
                per_series_gains=per_series, policy=policy)
            from_facts = ap.decide_from_facts(facts, policy=policy)
            assert direct.admitted == from_facts.admitted
            assert direct.reason == from_facts.reason


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_aggregate_fails_closed(bad):
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=1.0, max_single_series_harm=99.0))
    direct = ap.decide(
        relation="CONFLICT", aggregate_gain=bad, per_series_gains=[0.1])
    assert not direct.admitted
    assert direct.reason == "non_finite_aggregate_fail_closed"
    from_facts = ap.decide_from_facts(_facts("CONFLICT", bad, 20, 0, 0.1))
    assert not from_facts.admitted
    assert from_facts.reason == "non_finite_aggregate_fail_closed"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_per_series_fails_closed(bad):
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=1.0, max_single_series_harm=99.0))
    # One unusable series makes the budget uncheckable: refuse the whole
    # reading rather than average over the hole.
    direct = ap.decide(
        relation="CONFLICT", aggregate_gain=0.3, per_series_gains=[0.5, bad])
    assert not direct.admitted
    assert direct.reason == "no_per_series_reading_fail_closed"
    from_facts = ap.decide_from_facts(_facts("CONFLICT", 0.3, 20, 1, bad))
    assert not from_facts.admitted
    assert from_facts.reason == "no_per_series_reading_fail_closed"


def test_decide_from_facts_fails_closed_without_a_series_count():
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=1.0, max_single_series_harm=99.0))
    for facts in (
        _facts("CONFLICT", 0.3, 0, 0, None),
        _facts("CONFLICT", 0.3, None, None, None),
        {},
    ):
        verdict = ap.decide_from_facts(facts)
        assert not verdict.admitted


def test_reset_restores_the_strict_default():
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20,
        max_single_series_harm=0.30))
    assert ap.active_policy().rule == ap.BOUNDED_V1
    assert ap.reset_policy() == ap.DEFAULT
    assert ap.active_policy().rule == ap.STRICT
