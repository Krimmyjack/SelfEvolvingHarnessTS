"""Held-in phase: state threading, per-series logging, gated write-back."""
from __future__ import annotations

import inspect

import pytest

from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import p4b_heldin as heldin


def test_store_delta_reports_an_unchanged_store():
    view = (("skill_a", 1, "body", "{}", "{}"),)
    result = heldin.store_delta(view, view)
    assert result["identical"] is True
    assert result["changed_entries"] == []


@pytest.mark.parametrize(
    "after",
    [
        (("skill_a", 2, "body", "{}", "{}"),),            # revision moved
        (("skill_a", 1, "other", "{}", "{}"),),           # body rewritten
        (("skill_a", 1, "body", '{"x":1}', "{}"),),       # scope changed
        (("skill_a", 1, "body", "{}", '{"g":1}'),),       # risk guards changed
    ],
    ids=["revision", "body", "applicability", "risk_guards"],
)
def test_store_delta_names_what_moved(after):
    before = (("skill_a", 1, "body", "{}", "{}"),)
    result = heldin.store_delta(before, after)
    assert result["identical"] is False
    assert result["changed_entries"]


def test_store_delta_catches_an_added_entry():
    before = (("skill_a", 1, "body", "{}", "{}"),)
    after = before + (("skill_b", 1, "body", "{}", "{}"),)
    result = heldin.store_delta(before, after)
    assert result["identical"] is False
    assert result["entries_before"] == 1
    assert result["entries_after"] == 2


def test_writeback_is_gated_when_every_store_move_was_paid_for():
    rows = [
        {"arm": "A5-bounded", "replica": "Forward", "origin": 1176,
         "store_semantics_changed": True, "approved_skill_id": "sk_1",
         "incumbent_changed": True,
         "probes": [{"admission": {"admitted": True}}]},
        {"arm": "A5-strict", "replica": "Forward", "origin": 1176,
         "store_semantics_changed": False, "approved_skill_id": None,
         "incumbent_changed": False,
         "probes": [{"admission": {"admitted": False}}]},
    ]
    result = heldin.gated_writeback_check(rows)
    assert result["identical"] is True
    assert result["verdict"] == "WRITEBACK_GATED"
    assert result["cells_checked"] == 2


def test_writeback_check_catches_a_store_move_the_gate_never_authorised():
    # A Skill-level change on a cell where nothing was admitted means write-back
    # found a path around the admission gate, which voids the contrast.
    rows = [
        {"arm": "A5-strict", "replica": "Forward", "origin": 1176,
         "store_semantics_changed": True, "approved_skill_id": None,
         "incumbent_changed": False,
         "probes": [{"admission": {"admitted": False}}]},
    ]
    result = heldin.gated_writeback_check(rows)
    assert result["identical"] is False
    assert result["verdict"] == "LEAKAGE_SUSPECTED"
    assert result["ungated_writes"][0]["origin"] == 1176


def test_run_cell_does_not_take_a_backend():
    # The arm's backend lives behind its state's agent; a second handle here
    # would let a cell quietly open its own budget scope.
    assert "backend" not in inspect.signature(heldin.run_cell).parameters
    assert "backend" in inspect.signature(heldin.new_state).parameters


def test_held_in_uses_the_frozen_contract_budget_and_no_slow():
    source = inspect.getsource(heldin.run_cell)
    assert "contract.MAX_SUPPORT_A" in source
    assert "allow_slow=contract.ALLOW_SLOW" in source
    assert contract.ALLOW_SLOW is False


def test_probe_rows_keep_the_admission_verdict_and_do_not_call_passed_a_verdict():
    class _Result:
        actual_probed_programs = [
            {"candidate_id": "c1", "kind": "probe", "gain": 0.3, "passed": True,
             "admission": {"admitted": False, "reason": "harmed_fraction_over_budget"}},
            {"candidate_id": "c2", "kind": "verifier_rejected", "gain": None,
             "passed": False},
        ]

    rows = heldin._probe_rows(_Result(), [1.0] * 20)
    assert rows[0]["admission"]["admitted"] is False
    # "passed" means the probe ran legally; the qualifying decision is separate.
    assert rows[0]["probe_executed_legally"] is True
    assert "passed" not in rows[0]
    assert rows[1]["admission"] is None
