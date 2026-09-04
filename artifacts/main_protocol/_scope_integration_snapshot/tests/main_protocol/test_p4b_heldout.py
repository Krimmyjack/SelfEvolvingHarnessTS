"""Held-out scoring: frozen recall, one reading, nothing flows back."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.functional import run_v1_guidance_evolution as runner
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import p4b_heldout as heldout

# A held-out origin from the resolved plan; the module-level constants were
# replaced by a screened, derived plan.
ORIGIN = 3096
WINSORIZE = [{"op": "winsorize", "params": {}}]


@pytest.fixture(scope="module")
def base_cell():
    cell, _selection, _data = forecast_p1._load_exposed_cells()
    return cell


def _state(tmp_path, *, incumbent=None):
    state = four_arms._new_state(
        snapshot=runner._h0_snapshot(),
        # Held-out never calls the Fast agent; a stub proves the path does not
        # reach for one.
        agent=SimpleNamespace(),
        store_root=tmp_path,
        tag="p4b-heldout-test",
    )
    state["incumbent"] = incumbent
    return state


def test_held_out_module_takes_no_backend_or_agent():
    # The 0-LLM property is structural, not a discipline: none of the entry
    # points can be handed a provider.
    for function in (heldout.score, heldout.row, heldout.identity_row,
                     heldout.frozen_program_row):
        names = set(inspect.signature(function).parameters)
        assert not names & {"backend", "agent", "provider", "executor"}


def test_identity_scores_exactly_zero(base_cell):
    reading = heldout.score(base_cell, ORIGIN, ())
    assert reading["delta_utility_vs_identity"] == 0.0
    assert reading["harmed_count"] == 0
    assert reading["worst_single_series_harm"] == 0.0
    assert reading["material_harm_event"] is False
    assert reading["llm_calls"] == 0


def test_a_deployed_program_is_scored_per_series(base_cell):
    reading = heldout.score(base_cell, ORIGIN, WINSORIZE)
    assert reading["series_count"] == len(reading["per_series_gain"]) == 20
    harmed = [g for g in reading["per_series_gain"] if g < -heldout.MATERIAL]
    assert reading["harmed_count"] == len(harmed)
    assert reading["harmed_fraction"] == pytest.approx(len(harmed) / 20)
    assert reading["worst_single_series_harm"] == pytest.approx(
        -min(reading["per_series_gain"])
    )
    assert reading["llm_calls"] == 0


def test_aggregate_matches_the_identity_minus_deployed_smase(base_cell):
    reading = heldout.score(base_cell, ORIGIN, WINSORIZE)
    assert reading["delta_utility_vs_identity"] == pytest.approx(
        reading["identity_smase"] - reading["deployed_smase"]
    )


def test_an_arm_with_no_state_deploys_identity(tmp_path, base_cell):
    row = heldout.row(
        arm="A5-bounded", replica="Forward", origin=ORIGIN,
        state=_state(tmp_path), base_cell=base_cell,
    )
    assert row["deploy"]["applied_steps"] == []
    assert row["delta_utility_vs_identity"] == 0.0
    assert row["feedback_taken"] is False
    assert row["state_written"] is False


def test_an_arm_standing_on_an_incumbent_deploys_it(tmp_path, base_cell):
    row = heldout.row(
        arm="A5-bounded", replica="Forward", origin=ORIGIN,
        state=_state(tmp_path, incumbent=WINSORIZE), base_cell=base_cell,
    )
    assert row["deploy"]["applied_steps"] == WINSORIZE
    assert row["deploy"]["deploy_source"] == shared_harness.DEPLOY_SOURCE_INCUMBENT
    expected = heldout.score(base_cell, ORIGIN, WINSORIZE)
    assert row["delta_utility_vs_identity"] == pytest.approx(
        expected["delta_utility_vs_identity"]
    )


def test_held_out_roles_come_from_the_frozen_contract(tmp_path, base_cell):
    for arm, role in (("A5-strict", "primary"), ("A5-bounded", "primary")):
        row = heldout.row(
            arm=arm, replica="Forward", origin=ORIGIN,
            state=_state(tmp_path), base_cell=base_cell,
        )
        assert row["held_out_role"] == role


def test_a_frozen_comparator_may_not_select_on_held_out(base_cell):
    with pytest.raises(ValueError):
        heldout.frozen_program_row(
            arm="Parallel Best-of-N@8", replica="Forward", origin=ORIGIN,
            base_cell=base_cell, applied_steps=WINSORIZE,
            selection_face="held_out",
        )
    ok = heldout.frozen_program_row(
        arm="Parallel Best-of-N@8", replica="Forward", origin=ORIGIN,
        base_cell=base_cell, applied_steps=WINSORIZE, selection_face="held_in",
    )
    assert ok["parallel_selection_face"] == "held_in"
    assert ok["deploy"]["deploy_source"] == "frozen_held_in_selection"


def test_static_reference_is_identity_everywhere(base_cell):
    row = heldout.identity_row(replica="Forward", origin=ORIGIN, base_cell=base_cell)
    assert row["arm"] == "Static"
    assert row["delta_utility_vs_identity"] == 0.0
    assert row["deploy"]["deploy_source"] == "identity"


def test_store_semantics_is_field_by_field_and_comparable(tmp_path):
    first = heldout.store_semantics(_state(tmp_path)["method"])
    again = heldout.store_semantics(_state(tmp_path / "b")["method"])
    # K0's check is equality of this view before and after held-in, so it
    # has to be stable and comparable for an unchanged store.
    assert first == again
    assert isinstance(first, tuple)
