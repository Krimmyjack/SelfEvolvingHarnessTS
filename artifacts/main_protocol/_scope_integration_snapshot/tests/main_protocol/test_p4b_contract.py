"""The P4b frozen contract: origin plan, arm table, admission scoping."""
from __future__ import annotations

import pytest

from evaluation.main_protocol_p4 import p4b_contract as contract
from SelfEvolvingHarnessTS.methods.ttha import admission_policy

ALL_VIABLE = lambda origin: True  # noqa: E731


@pytest.fixture(autouse=True)
def _reset_policy():
    yield
    admission_policy.reset_policy()


@pytest.fixture
def plan():
    return contract.resolve_origins(ALL_VIABLE)


def test_a_clean_run_of_origins_is_evenly_spaced(plan):
    assert plan["held_in_spacings"] == [contract.MIN_SPACING] * 7
    assert plan["held_out_spacings"] == [contract.MIN_SPACING] * 7
    assert contract.validate_geometry(plan) == []


def test_origins_do_not_overlap_in_time(plan):
    # Minimum spacing is one context plus one horizon, so origin o reads
    # [o-192, o) and evaluates [o, o+48) without touching its neighbour.
    assert contract.MIN_SPACING == contract.CONTEXT_LENGTH + contract.HORIZON
    for key in ("held_in_origins", "held_out_origins"):
        origins = plan[key]
        for earlier, later in zip(origins, origins[1:]):
            assert later - contract.CONTEXT_LENGTH >= earlier + contract.HORIZON


def test_blocks_are_isolated_from_each_other_and_from_old_p4(plan):
    geometry = contract.geometry(plan)
    assert geometry["held_in_context_opens_at"] >= contract.OLD_P4_EVAL_END
    assert geometry["held_out_context_opens_at"] >= plan["held_in_eval_end"]
    assert not set(plan["held_in_origins"]) & set(plan["held_out_origins"])
    assert not set(plan["held_in_origins"]) & set(contract.OLD_P4_ORIGINS)
    assert not set(plan["held_out_origins"]) & set(contract.OLD_P4_ORIGINS)


def test_unevaluable_origins_are_skipped_not_shifted():
    # Some KDD context windows are flat enough that the Consumer's robust scale
    # hits its floor.  The plan steps over them and keeps the spacing rule; it
    # does not slide the whole block or accept a closer origin.
    skipped = {1416, 1656}
    plan = contract.resolve_origins(lambda origin: origin not in skipped)
    assert not skipped & set(plan["held_in_origins"])
    assert plan["held_in_origins"][0] == contract.HELD_IN_SEARCH_START
    assert all(gap >= contract.MIN_SPACING for gap in plan["held_in_spacings"])
    assert contract.validate_geometry(plan) == []


def test_a_long_degenerate_stretch_widens_a_gap_without_breaking_the_grid():
    plan = contract.resolve_origins(
        lambda origin: not (2400 <= origin <= 3600)
    )
    assert max(plan["held_in_spacings"] + plan["held_out_spacings"]) > contract.MIN_SPACING
    assert contract.validate_geometry(plan) == []
    for origin in plan["held_in_origins"] + plan["held_out_origins"]:
        assert origin % contract.ORIGIN_GRID_STEP == (
            contract.HELD_IN_SEARCH_START % contract.ORIGIN_GRID_STEP
        )


def test_resolution_fails_loudly_when_too_few_origins_are_viable():
    with pytest.raises(ValueError):
        contract.resolve_block(
            contract.HELD_IN_SEARCH_START, 8, lambda origin: origin < 2000
        )


def test_geometry_fails_when_the_roster_is_too_short(plan):
    assert contract.validate_geometry(
        plan, minimum_series_length=plan["held_out_eval_end"] - 1
    )
    assert contract.validate_geometry(
        plan, minimum_series_length=plan["held_out_eval_end"]
    ) == []


def test_both_arms_carry_state_and_share_their_start():
    # The two arms may differ only by the admission rule; anything else would
    # confound the one contrast this data can express.
    assert {arm.snapshot_source for arm in contract.ARMS} == {"shared_initial"}
    assert all(arm.carries_state for arm in contract.ARMS)
    assert {arm.bounded for arm in contract.ARMS} == {False, True}


def test_the_arm_table_is_exactly_the_gate_contrast():
    assert {arm.name for arm in contract.ARMS} == {"A5-strict", "A5-bounded"}
    primary = {arm.name for arm in contract.ARMS if arm.held_out_role == "primary"}
    assert primary == {"A5-strict", "A5-bounded"}


def test_the_accumulation_claim_is_disclaimed_in_the_contract(plan):
    # The audited Source card matches no origin in this study, so the study may
    # not be read as an accumulation experiment.
    assert contract.SOURCE_TREATMENT_ACTIVE is False
    payload = contract.contract(plan)
    assert payload["source_treatment"]["active"] is False
    assert "accumulation" in payload["not_tested_here"]
    assert contract.SOURCE_SCOPE_MATCH_MINIMUM >= 1


def test_replica_orders_are_permutations_of_held_in(plan):
    for order in contract.replica_orders(plan["held_in_origins"]).values():
        assert sorted(order) == sorted(plan["held_in_origins"])


def test_admission_scope_installs_and_always_restores():
    with contract.admission_scope("A5-bounded") as policy:
        assert policy.rule == admission_policy.BOUNDED_V1
        assert admission_policy.active_policy().rule == admission_policy.BOUNDED_V1
    assert admission_policy.active_policy() == admission_policy.DEFAULT

    with contract.admission_scope("A5-strict") as policy:
        assert policy == admission_policy.DEFAULT
    assert admission_policy.active_policy() == admission_policy.DEFAULT


def test_admission_scope_restores_even_when_the_arm_raises():
    with pytest.raises(RuntimeError):
        with contract.admission_scope("A5-bounded"):
            raise RuntimeError("arm blew up mid-cell")
    # Without this the next arm silently inherits the bounded rule and
    # A5-strict stops being the old policy.
    assert admission_policy.active_policy() == admission_policy.DEFAULT


def test_bounded_budget_matches_the_preregistered_numbers():
    assert contract.BOUNDED_POLICY.max_harmed_fraction == 0.20
    assert contract.BOUNDED_POLICY.max_single_series_harm == 0.30
    assert contract.ALLOW_SLOW is False


def test_only_held_in_spends_llm(plan):
    payload = contract.contract(plan)
    assert payload["cells"]["held_out_spends_llm"] is False
    assert payload["cells"]["global_llm_call_cap"] == (
        contract.HELD_IN_CELLS * contract.MAX_LLM_CALLS
    )
    assert payload["statistics"]["n"] == contract.ORIGINS_PER_BLOCK
    assert payload["geometry"]["outcome_read_during_selection"] is False
