"""The four HEC-1 wirings, each checked where it could go wrong quietly.

Adversarial on purpose, because every one of these has a failure mode that looks
like success:

* a contract whose drift check passes can still be unratified, and a runner that
  read the first as the second would spend a budget nobody released;
* a threshold tool that quietly widened past the risk budget would produce legal
  clauses and unsafe deployments;
* a three-state machine that let a FLAGGED Draft narrow would spend the revision
  budget repairing the wrong surface;
* an outer loop that could write the active set would grant execution rights
  with no gate at all.

0 LLM and 0 Consumer fits: predicates, feature cards and gain vectors only.  The
fit-touching part of the delivery gate lives in ``run_hec1 --smoke``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.main_protocol_p4 import hec1_contract as contract  # noqa: E402
from evaluation.main_protocol_p4 import outer_loop  # noqa: E402
from evaluation.main_protocol_p4 import restricted_draft as drafts  # noqa: E402
from evaluation.main_protocol_p4 import run_hec1 as runner  # noqa: E402
from evaluation.main_protocol_p4 import scope_spec as scopes  # noqa: E402
from evaluation.main_protocol_p4 import (  # noqa: E402
    scope_threshold_tool as tool,
)

MATERIAL = contract.RISK["material"]

Z = "local_robust_z_peak"


def _rows(pairs, feature=Z):
    return [{"features": {feature: value}, "gain": gain}
            for value, gain in pairs]


# ---------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------

def test_the_contract_has_no_mechanical_drift():
    state = contract.assert_frozen()
    assert state["frozen"], state["failures"]
    assert (state["phase_s_units"], state["phase_t_units"]) == (13, 26)


def test_sols_final_rulings_are_in_the_contract_and_the_sign_test_is_gone():
    """sol final ruling 2026-09-03: eight rulings, the replay share at 1.0 of
    the arm's own fits, no significance gate, the narrowed P1-only verdict."""
    assert len(contract.SOL_FINAL_RULINGS) == 8
    assert contract.VERSION == "P4U-v4.1"
    assert contract.REPLAY_FITS_SHARE == 1.0
    assert contract.REPLAY_SHARE_RECORD["v1_forward_shakedown_ran_under"] == 0.25
    assert contract.PHASE_F["requires_non_empty_k0"] is True
    assert contract.CODE_FREEZE["no_new_hash_infrastructure"] is True
    assert "HEC1_P1_ONLY__RECALL_ACCUMULATION" in contract.VERDICTS
    assert "sign test" not in " ".join(
        contract.STATISTICS["qualitative_criteria_for_P1"]).lower()
    assert contract.STATISTICS["status"].startswith("DESCRIPTIVE")
    assert contract.LIFECYCLE["waiting"]["consumes_verification_attempt"] is True
    assert contract.READOUTS["best_safe_global"].startswith(
        "OFFLINE IN-BUDGET COMPARATOR")
    assert "mechanical deployment" in contract.PHASE_F["how"]
    assert "HEC1_P1_ONLY__RECALL_ACCUMULATION" in contract.PHASE_F["requires"][0]
    assert contract.SKILL_TAXONOMY["promotion"].startswith(
        "General and Specific never promote")
    confirmed = {row["field"]: row["default"] for row in contract.CONFIRMED_BY_SOL}
    assert (confirmed["budget.replay_fits_share_of_course_fits"]
            == contract.REPLAY_FITS_SHARE)
    assert len(contract.CONFIRMED_BY_SOL) == 13
    receipt = contract.to_dict()
    assert receipt["ratification"]["final_rulings"]
    assert receipt["skill_taxonomy"] and receipt["naming"]


def test_the_readout_reports_a_sign_pattern_without_a_significance_field():
    from evaluation.main_protocol_p4 import audit_hec1_readout as readout

    pattern = readout._sign_pattern([0.1, 0.2, 0.05, 0.3], unit_name="cohort")
    assert pattern["positive"] == 4 and pattern["trials"] == 4
    assert pattern["exact_binomial_probability"] == 0.0625
    assert pattern["floor_at_all_positive"] == 0.0625
    assert "significant" not in pattern and "alpha" not in pattern
    assert pattern["role"] == "DESCRIPTIVE"
    boot = readout._bootstrap_mean([0.1, 0.2, 0.05, 0.3])
    assert boot["n"] == 4 and boot["role"] == "DESCRIPTIVE"
    assert boot["interval_90"][0] <= boot["mean"] <= boot["interval_90"][1]


def test_the_readout_keeps_a_resumed_course_and_lists_what_it_superseded(
        tmp_path, monkeypatch):
    from evaluation.main_protocol_p4 import audit_hec1_readout as readout

    monkeypatch.setattr(readout, "ARTIFACTS", tmp_path)
    stamp = {"mode": "scientific",
             "code_state": {"code_commit": "abc123", "runner_files_clean": True}}
    blocked = {"status": "BLOCKED", "offline": False, "ordering": "forward",
               **stamp}
    complete = {"status": "COMPLETE", "offline": False, "ordering": "forward",
                "units": [], "cells": [], "lifecycle": {}, **stamp}
    (tmp_path / "hec1_course_v11_forward_live.json").write_text(
        json.dumps(blocked), encoding="utf-8")
    (tmp_path / "hec1_course_v11_forward_live.resumed.json").write_text(
        json.dumps(complete), encoding="utf-8")
    (tmp_path / "hec1_course_hec1_forward_live.json").write_text(
        json.dumps(complete), encoding="utf-8")          # the old default label
    (tmp_path / "hec1_course_forward_offline.json").write_text(
        json.dumps({**complete, "offline": True}), encoding="utf-8")
    # The v1 shakedown: bare label, no mode / code_state stamp.
    (tmp_path / "hec1_course_forward_live.json").write_text(
        json.dumps({"status": "COMPLETE", "offline": False,
                    "ordering": "forward", "units": [], "cells": []}),
        encoding="utf-8")
    kept, rejected = readout._live_courses("v11_")
    assert [path.name for path in kept] == [
        "hec1_course_v11_forward_live.resumed.json"]
    reasons = {row["artifact"]: row["why"] for row in rejected}
    assert "status is BLOCKED" in reasons["hec1_course_v11_forward_live.json"]
    # ``hec1_`` is a prefix like any other; it is simply not the chain asked for.
    assert "prefix" in reasons["hec1_course_hec1_forward_live.json"]
    assert "hec1_course_forward_offline.json" in reasons
    # With the v11_ chain requested, the bare-label shakedown is out on prefix.
    assert "prefix" in reasons["hec1_course_forward_live.json"]
    # Without a prefix the bare-label shakedown is still excluded by its
    # missing stamp, and an explicit shakedown stamp is excluded by name.
    (tmp_path / "hec1_course_reverse_live.json").write_text(
        json.dumps({**complete, "ordering": "reverse", "mode": "shakedown"}),
        encoding="utf-8")
    kept, rejected = readout._live_courses()
    reasons = {row["artifact"]: row["why"] for row in rejected}
    assert "shakedown" in reasons["hec1_course_reverse_live.json"]
    assert "pre-v1.1" in reasons["hec1_course_forward_live.json"]
    # An offline course under a scientific-looking label is refused by its flag.
    (tmp_path / "hec1_course_v11_interleaved_live.json").write_text(
        json.dumps({**complete, "ordering": "interleaved", "offline": True}),
        encoding="utf-8")
    _, rejected = readout._live_courses("v11_")
    reasons = {row["artifact"]: row["why"] for row in rejected}
    assert "offline" in reasons["hec1_course_v11_interleaved_live.json"]


def test_a_scientific_live_launch_is_refused_on_dirty_runner_files(monkeypatch):
    """sol v1.1 R-B, enforced before any LLM call: the runner refuses a live
    scientific course unless the HEC-1 runner files are committed and clean.
    A shakedown is exempt and is stamped so the readout excludes it."""
    monkeypatch.setattr(runner, "code_state", lambda: {
        "code_commit": "abc123", "runner_files_dirty": ["run_hec1.py"],
        "runner_files_clean": False, "git_available": True})
    report = runner.run_course(
        phase="phase_t_forward", ordering_name="forward",
        units=contract.ordering("forward"), run_label="pytest_dirty_refusal",
        offline=False, limit=1)
    assert report["status"] == "BLOCKED_ON_CODE_FREEZE"
    assert report["mode"] == "scientific"
    assert report["llm_calls"] == 0 and report["consumer_fits"] == 0
    assert "cells" not in report


def test_the_runner_default_label_is_what_the_readout_whitelists():
    from evaluation.main_protocol_p4 import audit_hec1_readout as readout

    for ordering in contract.ORDERINGS:
        assert "%s_live" % ordering in readout.LIVE_LABELS


def test_the_released_phases_are_launchable_and_phase_f_is_not():
    """The ratification opened four phases and left the fifth to a human."""
    assert contract.assert_frozen()["frozen"]
    for phase in ("phase_s", "phase_t_forward", "phase_t_reverse",
                  "phase_t_interleaved"):
        verdict = contract.assert_launchable(phase)
        assert verdict["launchable"], verdict["blockers"]
        assert verdict["llm_cap"] == contract.LLM_CAPS[phase]
    refused = contract.assert_launchable("phase_f")
    assert not refused["launchable"]
    assert refused["verdict_if_launched_anyway"] == "BLOCKED_ON_CONTRACT"


def test_the_ratification_records_what_was_adjudicated_rather_than_a_boolean():
    assert len(contract.CONFIRMED_BY_SOL) == 13
    assert len(contract.SOL_RULINGS) == 6
    assert len(contract.AUTO_CONTINUE_CONDITIONS) == 8
    assert sum(contract.LLM_CAPS.values()) <= contract.TOTAL_LLM_HARD_CAP
    for row in contract.SOL_RULINGS:
        assert row["enforced_by"], "a ruling with no enforcement point"


def test_every_ordering_is_a_permutation_of_the_same_units():
    units = sorted((row["block"], row["origin"])
                   for row in contract.phase_t_units())
    for name in contract.ORDERINGS:
        assert sorted((row["block"], row["origin"])
                      for row in contract.ordering(name)) == units


def test_phase_s_and_phase_t_never_share_a_block():
    """A series K0 learned on must not reappear in the Target."""
    s_blocks = {row["block"] for row in contract.phase_s_units()}
    t_blocks = {row["block"] for row in contract.phase_t_units()}
    assert not s_blocks & t_blocks


def test_the_forward_budget_arithmetic_fits_under_its_own_cap():
    arithmetic = contract.budget_arithmetic()
    assert arithmetic["forward_full_k0"] <= arithmetic["forward_hard_cap"]
    assert arithmetic["forward_empty_k0"] < arithmetic["forward_full_k0"]
    assert arithmetic["phase_s_estimate"] <= arithmetic["phase_s_cap"]


# ---------------------------------------------------------------------------
# scoreability: three numbers that are not the same number
# ---------------------------------------------------------------------------

def test_the_three_unit_counts_are_distinct_and_the_floor_uses_ceil():
    """26 scheduled, 23 scoreable, 19 paired points -- and 19, not 18.

    ``int(0.8 * 23)`` is 18, and 18/23 is 78.3%, which is not the 80% the
    contract declares.  The rounding is the whole assertion.
    """
    from evaluation.main_protocol_p4 import hec1_scoreability as scoreability

    assert scoreability.SCHEDULED_UNITS == 26
    assert scoreability.SCOREABLE_UNITS == 23
    assert len(scoreability.UNSCOREABLE_UNITS) == 3
    assert scoreability.MIN_PAIRED_CURVE_POINTS == 19
    assert int(0.8 * scoreability.SCOREABLE_UNITS) == 18  # the trap
    assert (scoreability.MIN_PAIRED_CURVE_POINTS
            / scoreability.SCOREABLE_UNITS) >= scoreability.COMPLETION_FRACTION


def test_the_frozen_scoreability_list_matches_a_fresh_derivation():
    """Declaration and derivation must agree, or the run stops.

    Reads the artifact the 0-fit preflight wrote rather than re-deriving here,
    so this test fails if either side drifts.
    """
    from evaluation.main_protocol_p4 import hec1_scoreability as scoreability

    path = (ROOT / "artifacts/main_protocol/hec1_evaluability.json")
    if not path.is_file():
        pytest.skip("the evaluability preflight has not been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verdict = scoreability.verify_against(payload)
    assert verdict["passed"], verdict


def test_an_unscoreable_unit_contributes_no_curve_point_rather_than_a_zero():
    """A missing evaluation must not read as a tie.

    ``_gain`` returns 0.0 for a missing reading, so an unfiltered curve would
    take three fabricated ties -- each one diluting the sign pattern and
    flattening the cumulative difference.
    """
    from evaluation.main_protocol_p4 import hec1_scoreability as scoreability

    rows = [
        {"unit": {"block": "[0:40]", "origin": 1176},
         "arms": {"A3-online": {"aggregate_gain": 0.5},
                  "A3-frozen": {"aggregate_gain": 0.2}}},
        {"unit": {"block": "[0:40]", "origin": 2856},        # unscoreable
         "arms": {"A3-online": None, "A3-frozen": None}},
        {"unit": {"block": "[0:40]", "origin": 1896},        # half-scored
         "arms": {"A3-online": {"aggregate_gain": 0.1}, "A3-frozen": {}}},
    ]
    paired = scoreability.paired_curve_points(rows, "A3-online", "A3-frozen")
    assert [row["unit"]["origin"] for row in paired] == [1176]
    assert not scoreability.unit_is_scoreable({"block": "[0:40]",
                                               "origin": 2856})
    assert scoreability.unit_is_scoreable({"block": "[0:40]", "origin": 1176})


def test_the_readout_counts_paired_points_not_units_run(tmp_path, monkeypatch):
    """The completion floor is 19 paired points, never 21 of 26 scheduled."""
    from evaluation.main_protocol_p4 import audit_hec1_readout as readout
    from evaluation.main_protocol_p4 import hec1_scoreability as scoreability

    monkeypatch.setattr(readout, "ARTIFACTS", tmp_path)
    units = [
        {"unit": {"block": "[0:40]", "origin": 1000 + index},
         "arms": {"A3-online": {"aggregate_gain": 0.2},
                  "A3-frozen": {"aggregate_gain": 0.1}}}
        for index in range(20)
    ]
    units += [
        {"unit": {"block": "[0:40]", "origin": 2856},
         "arms": {"A3-online": None, "A3-frozen": None}},
    ]
    course = {"status": "COMPLETE", "offline": False, "ordering": "forward",
              "units": units, "cells": [], "lifecycle": {}}
    path = tmp_path / "hec1_course_forward_live.json"
    path.write_text(json.dumps(course), encoding="utf-8")
    row = readout.read_ordering(path)
    assert row["units_run"] == 21
    assert row["paired_curve_points"] == 20
    assert row["min_paired_curve_points"] == scoreability.MIN_PAIRED_CURVE_POINTS
    assert row["meets_completion_floor"] is True
    # 21 units ran; ceil(0.8 * 26) = 21 would have demanded a 21st paired
    # point that cannot exist, and the curve would never be readable.
    assert len(row["paired_differences"]) == 20


def test_no_boundary_counter_is_non_zero():
    assert not [key for key, value in contract.BOUNDARY.items() if value]


# ---------------------------------------------------------------------------
# W2: the threshold tool
# ---------------------------------------------------------------------------

def test_the_widest_feasible_edge_is_taken_and_ties_go_to_the_coarser_box():
    rows = _rows([(9.0, 0.5), (8.0, 0.4), (7.0, 0.3), (6.5, 0.2), (6.2, 0.1),
                  (2.0, -0.9), (1.0, -0.8)])
    result = tool.calibrate(feature=Z, direction=">=", rows=rows)
    # >= 3 and >= 6 select the same five rows here, so the tie resolves to the
    # lower edge: the coarser box is the one the bins describe next window.
    assert (result["threshold"], result["treated"]) == (3.0, 5)
    assert result["tie_break"] == "coarser_box"


def test_a_harmful_row_inside_the_wider_box_forces_the_narrower_edge():
    rows = _rows([(9.0, 0.5), (8.0, 0.4), (7.0, 0.3), (6.5, 0.2), (6.2, 0.1),
                  (4.0, -0.9), (2.0, -0.9), (1.0, -0.8)])
    result = tool.calibrate(feature=Z, direction=">=", rows=rows)
    assert result["threshold"] == 6.0
    assert result["tie_break"] == "unique_widest"


def test_no_feasible_edge_is_a_refusal_not_a_widening():
    with pytest.raises(tool.NoFeasibleThreshold) as caught:
        tool.calibrate(feature=Z, direction=">=", rows=_rows([(9.0, -1.0)]))
    payload = caught.value.to_dict()
    assert payload["outcome"] == "NO_FEASIBLE_THRESHOLD"
    assert payload["candidates_tried"]


def test_a_threshold_from_slow_is_ignored_and_recorded():
    rows = _rows([(9.0, 0.5), (8.0, 0.4), (7.0, 0.3), (6.5, 0.2), (6.2, 0.1),
                  (2.0, -0.9), (1.0, -0.8)])
    result = tool.clause_from_slow(
        {"scope_clause": {"feature": Z, "op": ">=", "threshold": 4.25},
         "rationale": "spiky series"},
        rows=rows)
    assert tool.LLM_THRESHOLD_IGNORED in result["notes"]
    assert result["slow_threshold_as_returned"] == 4.25
    assert result["threshold"] == 3.0
    assert result["clause"]["threshold"] == 3.0


def test_a_feature_outside_the_frozen_vocabulary_is_refused():
    with pytest.raises(scopes.ScopeError):
        tool.calibrate(feature="series_uid", direction=">=",
                       rows=_rows([(9.0, 0.5)]))
    with pytest.raises(scopes.ScopeError):
        tool.calibrate(feature="period_evidence_status", direction=">=",
                       rows=_rows([(9.0, 0.5)]))


def test_the_shadow_is_recorded_and_says_it_is_not_deployable():
    rows = _rows([(9.0, 0.5), (8.0, 0.4), (7.0, 0.3), (6.5, 0.2), (6.2, 0.1),
                  (2.0, -0.9), (1.0, -0.8)])
    shadow = tool.best_stump(rows=rows)
    assert shadow["outcome"] == "BEST_STUMP"
    assert shadow["deployable"] is False
    assert shadow["feature"] in tool.VOCABULARY


def test_calibrate_and_best_stump_agree_on_feasibility():
    """Restricted to one feature, the shadow can only find what calibrate can.

    Two code paths deciding "does this clause clear the budget" is exactly the
    setup that produced the Source-v3 two-gate conflict, so the agreement is
    checked rather than assumed.
    """
    rows = _rows([(9.0, 0.5), (8.0, 0.4), (7.0, 0.3), (6.5, 0.2), (6.2, 0.1),
                  (2.0, -0.9), (1.0, -0.8)])
    calibrated = tool.calibrate(feature=Z, direction=">=", rows=rows)
    feasible_edges = {row["threshold"] for row in calibrated["candidates_tried"]
                      if row["feasible"]}
    stump = tool.best_stump(rows=rows, vocabulary=(Z,))
    assert stump["outcome"] == "BEST_STUMP"
    assert stump["direction"] == ">="
    assert stump["threshold"] in feasible_edges

    hopeless = _rows([(9.0, -1.0), (8.0, -1.0)])
    with pytest.raises(tool.NoFeasibleThreshold):
        tool.calibrate(feature=Z, direction=">=", rows=hopeless)
    assert tool.best_stump(rows=hopeless, vocabulary=(Z,))["outcome"] == (
        "NO_FEASIBLE_STUMP")


def test_the_tool_does_not_author_the_feature_list():
    rules = tool.declared_rules()
    assert rules["slow_authors"] == ["feature", "direction", "rationale"]
    assert rules["tool_authors"] == ["threshold"]
    assert "choose the feature" in rules["tool_may_not"]
    assert len(tool.VOCABULARY) == 12


# ---------------------------------------------------------------------------
# W3: the three-state machine
# ---------------------------------------------------------------------------

def _draft(ledger, origin=1896):
    predicate = [{"feature": Z, "op": ">=", "threshold": 3.0}]
    scope = {"scope_type": "serving_series_predicate", "predicate": predicate}
    return ledger.restrict(
        program_steps=(("outlier_mad", {}),), root_scope=scope,
        current_scope=scope, origin=origin,
        delayed_reading={"delayed_origin": origin + 48, "lines": {}})


def test_the_three_source_v3_windows_classify_as_sol_read_them():
    expected = runner.EXPECTED_STATES
    cases = runner._source_v3_replay_cases()
    assert len(cases) == 3
    for case in cases:
        ledger = drafts.DraftLedger()
        draft = _draft(ledger, case["restricted_at_origin"])
        ledger.record_verification(
            draft, window=case["window"],
            failed_lines=case["failed_lines"],
            per_series_gain=case["per_series_gain"],
            treated_prev=case["treated_prev"],
            treated_now=case["treated_now"], material=MATERIAL)
        assert draft.state == expected[case["restricted_at_origin"]], case


def test_a_flagged_draft_refuses_a_clause_request():
    ledger = drafts.DraftLedger()
    draft = _draft(ledger)
    ledger.record_verification(
        draft, window=2136, failed_lines=["single_series_harm"],
        per_series_gain={"a": -0.9, "b": 0.1},
        treated_prev=["a", "b"], treated_now=["a", "b"], material=MATERIAL)
    assert draft.state == drafts.FLAGGED
    assert not draft.may_add_clause()
    with pytest.raises(ValueError):
        ledger.record_revision(draft, origin=2136,
                               new_scope=dict(draft.current_scope),
                               preflight=None, support=None)


def test_a_waiting_reverification_does_not_spend_a_revision_or_an_attempt():
    ledger = drafts.DraftLedger()
    draft = _draft(ledger, 2376)
    before = draft.revisions
    ledger.record_verification(
        draft, window=2616, failed_lines=["coverage_floor"],
        per_series_gain={}, treated_prev=[], treated_now=[],
        material=MATERIAL, consumes_attempt=False)
    assert draft.state == drafts.WAITING
    assert draft.verification_attempts == 0
    assert draft.revisions == before


def test_the_verification_cap_archives_rather_than_deletes():
    ledger = drafts.DraftLedger()
    draft = _draft(ledger)
    for window in (2136, 2376, 2616):
        ledger.record_verification(
            draft, window=window, failed_lines=["single_series_harm"],
            per_series_gain={"a": -0.9, "b": -0.8, "c": 0.4},
            treated_prev=["c"], treated_now=["a", "b", "c"],
            material=MATERIAL)
    assert draft.verification_attempts == drafts.MAX_VERIFICATION_ATTEMPTS
    assert draft.closed is not None
    assert draft in ledger.drafts  # evidence is kept
    assert len(draft.history) == 3


def test_flagged_outranks_revisable_even_when_a_later_window_looks_revisable():
    ledger = drafts.DraftLedger()
    draft = _draft(ledger)
    ledger.record_verification(
        draft, window=2136, failed_lines=["single_series_harm"],
        per_series_gain={"a": -0.9, "b": 0.4},
        treated_prev=["a", "b"], treated_now=["a", "b"], material=MATERIAL)
    assert draft.state == drafts.FLAGGED
    ledger.record_verification(
        draft, window=2376, failed_lines=["single_series_harm"],
        per_series_gain={"x": -0.9, "b": 0.4},
        treated_prev=["b"], treated_now=["x", "b"], material=MATERIAL)
    assert draft.state == drafts.FLAGGED


def test_a_passing_reading_classifies_as_no_state():
    verdict = drafts.classify_failure(
        failed_lines=[], per_series_gain={"a": 0.4}, treated_prev=["a"],
        treated_now=["a"], material=MATERIAL)
    assert verdict["state"] is None


# ---------------------------------------------------------------------------
# W1: the outer loop
# ---------------------------------------------------------------------------

def _bank_row(origin, gains, op="outlier_mad"):
    return {
        "unit": {"block": "[0:40]", "origin": origin},
        "task_consumer_key": "forecast|pooled-ridge-a1|sMASE",
        "program_steps": [{"op": op, "params": {}}],
        "features": {uid: {Z: 9.0} for uid in gains},
        "per_series_gain": dict(gains),
    }


SAFE_BANK = [
    _bank_row(1176, {"a": 0.4, "b": 0.3, "c": 0.20, "d": 0.1, "e": 0.1}),
    _bank_row(1896, {"a": 0.4, "b": 0.3, "c": 0.25, "d": 0.1, "e": 0.1}),
]


def _replay(msh):
    def replay(*, steps, scope):
        return {"cells": [{"unit": {"block": "[0:40]", "origin": 1176},
                           "treated": 5, "aggregate_gain": 0.22,
                           "harmed_fraction": 0.0,
                           "max_single_series_harm": msh}],
                "fits": 2}
    return replay


def test_an_empty_bank_costs_nothing():
    ledger = drafts.DraftLedger()
    step = outer_loop.consolidate(bank=[], ledger=ledger, k_index=0)
    assert (step.slow_calls, step.replay_fits) == (0, 0)
    assert step.empty_reason == "the bank is empty"
    assert not step.drafts_opened
    assert not ledger.resupplied_programs()


def test_a_candidate_that_breached_an_already_processed_cell_is_eliminated():
    ledger = drafts.DraftLedger()
    step = outer_loop.consolidate(bank=SAFE_BANK, ledger=ledger, k_index=1,
                                  replay=_replay(0.91))
    assert [row["outcome"] for row in step.candidates] == [
        outer_loop.REPLAY_SCREEN_REJECTED]
    assert "single_series_harm" in step.rejected[0]["violations"][0]["violated"]
    assert not step.drafts_opened
    assert not ledger.resupplied_programs()


def test_a_survivor_becomes_a_resupplied_draft_and_nothing_more():
    ledger = drafts.DraftLedger()
    step = outer_loop.consolidate(bank=SAFE_BANK, ledger=ledger, k_index=1,
                                  replay=_replay(0.02))
    assert len(step.drafts_opened) == 1
    assert list(ledger.resupplied_programs()) == step.drafts_opened
    draft = ledger.by_id(step.drafts_opened[0])
    assert draft.deployable is False
    assert draft.state is None
    assert step.to_dict()["wrote_active"] is False


def test_one_positive_unit_mints_a_draft_and_never_a_deployable_card():
    """sol v1.1: the n=1 guard is forward verification, not refusing to look.

    Two was the mainline's own guard and it conflicted with ladder v2; it was
    also what made Phase S-v1 return an empty K0 without exercising the
    mechanism at all.  The replacement has to be checked at the place it now
    lives: one positive unit produces a candidate, and what that candidate
    produces is a Draft with no deployment rights.
    """
    ledger = drafts.DraftLedger()
    step = outer_loop.consolidate(bank=SAFE_BANK[:1], ledger=ledger, k_index=1,
                                  replay=_replay(0.02))
    assert outer_loop.MIN_POSITIVE_UNITS_FOR_ADD == 1
    assert len(step.drafts_opened) == 1
    draft = ledger.by_id(step.drafts_opened[0])
    assert draft.deployable is False
    assert draft.state is None          # nothing has verified it yet
    assert step.to_dict()["wrote_active"] is False


def test_the_census_key_separates_one_program_under_two_root_scopes():
    """sol v1.1: root Scope is part of the key.

    Same program, same Task/Consumer, two different starting predicates: they
    treat two different sets of series, so merging them would report one
    deployment's evidence as the other's.
    """
    def row(origin, scope, gains):
        return {**_bank_row(origin, gains), "serving_scope": scope}

    wide = {"scope_type": "serving_series_predicate",
            "predicate": [{"feature": Z, "op": ">=", "threshold": 3.0}]}
    narrow = {"scope_type": "serving_series_predicate",
              "predicate": [{"feature": Z, "op": ">=", "threshold": 6.0}]}
    bank = [row(1176, wide, {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1, "e": 0.1}),
            row(1896, narrow, {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6,
                               "e": 0.5})]
    groups = outer_loop.census(bank)
    assert len(groups) == 2
    assert len({group["root_scope_signature"] for group in groups}) == 2
    assert len({group["census_key"] for group in groups}) == 2
    # Each group keeps the predicate its own evidence was gathered under.
    for group in groups:
        assert group["root_scope"] in (wide, narrow)


def test_an_add_candidate_carries_the_root_scope_its_evidence_came_from():
    scope = {"scope_type": "serving_series_predicate",
             "predicate": [{"feature": Z, "op": ">=", "threshold": 6.0}]}
    bank = [{**_bank_row(1176, {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1,
                                "e": 0.1}), "serving_scope": scope}]
    ledger = drafts.DraftLedger()
    step = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=1,
                                  replay=_replay(0.02))
    assert step.drafts_opened
    draft = ledger.by_id(step.drafts_opened[0])
    # Not re-derived from the initialiser, which would have produced z >= 3.
    assert draft.current_scope == scope
    assert draft.root_scope == scope


def test_fault_type_does_not_enter_the_census_key():
    """It is stratified evidence, not identity."""
    left = {**_bank_row(1176, {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1,
                               "e": 0.1}), "fault_code": "RISK_REFUSAL"}
    right = {**_bank_row(1896, {"a": 0.4, "b": 0.3, "c": 0.25, "d": 0.1,
                                "e": 0.1}), "fault_code": "AGGREGATE_NEGATIVE"}
    groups = outer_loop.census([left, right])
    assert len(groups) == 1
    assert groups[0]["unit_count"] == 2


def test_programs_with_the_same_per_series_effect_collapse_to_one_candidate():
    bank = SAFE_BANK + [
        _bank_row(1176, {"a": 0.4, "b": 0.3, "c": 0.20, "d": 0.1, "e": 0.1},
                  op="winsorize"),
        _bank_row(1896, {"a": 0.4, "b": 0.3, "c": 0.25, "d": 0.1, "e": 0.1},
                  op="winsorize"),
    ]
    groups = outer_loop.census(bank)
    assert len(groups) == 1
    assert groups[0]["aliases"] == ["winsorize({})"]
    assert groups[0]["positive_units"] == 2


def test_a_flagged_draft_is_a_drift_signal_not_a_revision_candidate():
    ledger = drafts.DraftLedger()
    draft = _draft(ledger)
    ledger.record_verification(
        draft, window=2136, failed_lines=["single_series_harm"],
        per_series_gain={"a": -0.9, "b": 0.4},
        treated_prev=["a", "b"], treated_now=["a", "b"], material=MATERIAL)
    step = outer_loop.consolidate(bank=SAFE_BANK, ledger=ledger, k_index=1,
                                  replay=_replay(0.02))
    signals = [row["signal"] for row in step.drift_signals]
    assert "EFFECT_NONSTATIONARY_CANDIDATE" in signals
    assert not any(row["kind"] == "REVISE" for row in step.candidates)


def _revisable_draft(ledger):
    """A Draft the runner restricted (revisions=0) and one window put in
    REVISABLE: the harm sat entirely on new entrants."""
    predicate = [{"feature": Z, "op": ">=", "threshold": 3.0}]
    scope = {"scope_type": "serving_series_predicate", "predicate": predicate}
    draft = ledger.restrict(
        program_steps=(("outlier_mad", {}),), root_scope=scope,
        current_scope=scope, origin=1896,
        delayed_reading={"delayed_origin": 1944, "lines": {}}, revisions=0)
    ledger.record_verification(
        draft, window=1944, failed_lines=["single_series_harm"],
        per_series_gain={"a": 0.4, "b": 0.3, "x": -0.9},
        treated_prev=["a", "b"], treated_now=["a", "b", "x"], material=MATERIAL)
    assert draft.state == drafts.REVISABLE
    return draft


def _revise_bank():
    """Rows for the same program where the harmed series is the one with the
    largest ``Z``: a ``<=`` clause on ``Z`` can leave it outside."""
    rows = []
    for origin in (1176, 1896):
        row = _bank_row(origin, {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1,
                                 "e": 0.1, "x": -0.9})
        row["features"] = {"a": {Z: 3.5}, "b": {Z: 3.5}, "c": {Z: 4.0},
                           "d": {Z: 4.5}, "e": {Z: 5.0}, "x": {Z: 9.0}}
        rows.append(row)
    return rows


def _slow_names_z_below():
    def slow(*, candidate, rejected):
        return {"scope_clause": {"feature": Z, "op": "<=", "threshold": 99.0},
                "rationale": "test: exclude the spikiest series"}
    return slow


def test_a_revise_candidate_revises_the_existing_draft_in_place():
    """The lifecycle bound is per Draft, so the clause must land on the Draft
    that failed -- not on a new shell beside it with fresh counters."""
    ledger = drafts.DraftLedger()
    draft = _revisable_draft(ledger)
    before = (len(ledger.drafts), draft.revisions, draft.verification_attempts)
    step = outer_loop.consolidate(
        bank=_revise_bank(), ledger=ledger, k_index=1,
        slow=_slow_names_z_below(), replay=_replay(0.02))
    revise = [row for row in step.candidates if row["kind"] == "REVISE"]
    assert revise and revise[0]["outcome"] == "DRAFT_REVISED", step.candidates
    assert step.drafts_revised == [draft.draft_id]
    assert len(ledger.drafts) == before[0], "a REVISE opened a new shell"
    assert draft.revisions == before[1] + 1
    assert draft.verification_attempts == before[2]
    assert len(draft.current_scope["predicate"]) == 2
    assert draft.history[-1]["event"] == "revised_by_outer_loop"
    # Still owed a verification, so still resupplied -- under the revised scope.
    scopes_now = ledger.resupplied_scopes_for_verification()
    assert list(scopes_now) == [draft.draft_id]
    assert len(scopes_now[draft.draft_id]["predicate"]) == 2
    # And not proposed again before a new unit has read the revised predicate.
    again = outer_loop.consolidate(
        bank=_revise_bank(), ledger=ledger, k_index=2,
        slow=_slow_names_z_below(), replay=_replay(0.02))
    assert not any(row["kind"] == "REVISE" for row in again.candidates)
    assert draft.revisions == before[1] + 1


def test_resupply_by_verification_keeps_a_fully_revised_draft_until_it_is_read():
    """``open_drafts`` (v3) drops a Draft at the revision cap; HEC-1's resupply
    keeps it while it still owes a verification, and drops it when closed."""
    ledger = drafts.DraftLedger()
    draft = _revisable_draft(ledger)
    scope = dict(draft.current_scope)
    for _ in range(drafts.MAX_REVISIONS):
        clause = {"feature": Z, "op": "<=", "threshold": 9.0 - draft.revisions}
        scope = {"scope_type": "serving_series_predicate",
                 "predicate": list(scope["predicate"]) + [clause]}
        ledger.record_revision(draft, origin=2136, new_scope=scope,
                               preflight=None, support=None)
    assert not draft.may_revise()
    assert draft.draft_id not in ledger.resupplied_programs()          # v3 view
    assert draft.draft_id in ledger.resupplied_programs_for_verification()
    ledger.close(draft, "TEST_CLOSED")
    assert draft.draft_id not in ledger.resupplied_programs_for_verification()


def test_a_replay_cell_below_the_coverage_floor_is_neither_pass_nor_fail():
    """H3 as a screen: a processed unit where the narrowed predicate resolves
    to almost nobody reads as Static (exact zeros), and must not eliminate the
    candidate as 'not material'."""
    def replay(*, steps, scope):
        return {"cells": [
            {"unit": {"block": "[0:40]", "origin": 1176}, "treated": 5,
             "aggregate_gain": 0.22, "harmed_fraction": 0.0,
             "max_single_series_harm": 0.02},
            {"unit": {"block": "[0:40]", "origin": 1896}, "treated": 2,
             "aggregate_gain": 0.0, "harmed_fraction": 0.0,
             "max_single_series_harm": 0.0},
            {"unit": {"block": "[0:40]", "origin": 2136},
             "unusable": "SERVING_CONTEXT_DEGENERATE", "aggregate_gain": None},
        ], "fits": 3}
    verdict = outer_loop.screen({"program_steps": [{"op": "outlier_mad"}]},
                                None, replay=replay)
    assert verdict["passed"] is True
    assert verdict["cells_applicable"] == 1
    assert len(verdict["cells_not_applicable"]) == 2
    assert verdict["violations"] == []


def test_a_candidate_with_no_applicable_replay_cell_does_not_pass():
    def replay(*, steps, scope):
        return {"cells": [
            {"unit": {"block": "[0:40]", "origin": 1896}, "treated": 1,
             "aggregate_gain": 0.0, "harmed_fraction": 0.0,
             "max_single_series_harm": 0.0}], "fits": 1}
    verdict = outer_loop.screen({"program_steps": [{"op": "outlier_mad"}]},
                                None, replay=replay)
    assert verdict["passed"] is False
    assert verdict["reason"] == outer_loop.NOT_APPLICABLE


def test_an_applicable_cell_that_breaches_still_eliminates():
    def replay(*, steps, scope):
        return {"cells": [
            {"unit": {"block": "[0:40]", "origin": 1176}, "treated": 5,
             "aggregate_gain": 0.22, "harmed_fraction": 0.0,
             "max_single_series_harm": 0.02},
            {"unit": {"block": "[0:40]", "origin": 1896}, "treated": 6,
             "aggregate_gain": 0.10, "harmed_fraction": 0.0,
             "max_single_series_harm": 0.91},
        ], "fits": 2}
    verdict = outer_loop.screen({"program_steps": [{"op": "outlier_mad"}]},
                                None, replay=replay)
    assert verdict["passed"] is False
    assert verdict["reason"] == outer_loop.REPLAY_SCREEN_REJECTED


def test_the_outer_loop_is_deterministic_over_the_same_bank():
    first = outer_loop.consolidate(bank=SAFE_BANK, ledger=drafts.DraftLedger(),
                                    k_index=1, replay=_replay(0.02))
    second = outer_loop.consolidate(bank=list(reversed(SAFE_BANK)),
                                     ledger=drafts.DraftLedger(), k_index=1,
                                     replay=_replay(0.02))
    assert ([row["program_signature"] for row in first.candidates]
            == [row["program_signature"] for row in second.candidates])
    assert first.groups == second.groups


# ---------------------------------------------------------------------------
# W4: the single authoritative gate, and the Fast decision record
# ---------------------------------------------------------------------------

PASSING = {"treated": 8, "aggregate_gain": 0.20, "harmed_fraction": 0.05,
           "max_single_series_harm": 0.10}
THIN = {"treated": 2, "aggregate_gain": 0.20, "harmed_fraction": 0.0,
        "max_single_series_harm": 0.0}


def test_the_coverage_floor_belongs_to_the_authoritative_gate():
    """The exact shape of the Source-v3 round-2856 conflict."""
    gate = runner.authoritative_gate(THIN)
    assert not gate["passes"]
    assert gate["failed_lines"] == ["coverage_floor"]
    resolution = runner.resolve_gate_disagreement(gate, {"stage": "approved"})
    assert resolution["resolved_by"] == "p4_gate"
    assert resolution["disagree"] is True
    assert resolution["may_activate"] is False


def test_agreement_is_recorded_as_agreement():
    gate = runner.authoritative_gate(PASSING)
    assert gate["passes"]
    resolution = runner.resolve_gate_disagreement(gate, {"stage": "approved"})
    assert resolution["disagree"] is False
    assert resolution["may_activate"] is True


def test_the_online_loop_alone_can_never_activate():
    gate = runner.authoritative_gate(THIN)
    for event in ({"stage": "approved"}, {"stage": "rejected"}, None):
        assert not runner.resolve_gate_disagreement(gate, event)["may_activate"]


def test_a_zero_candidate_round_says_which_of_four_things_happened():
    assert runner.classify_fast_decision({"a": 1}, ["cand_1"])["decision"] == (
        "PROPOSED")
    assert runner.classify_fast_decision(
        {}, [], no_proposal_reason="no actionable defect")["decision"] == (
            "ABSTAINED_WITH_REASON")
    assert runner.classify_fast_decision(None, [])["decision"] == "EMPTY_OUTPUT"
    assert runner.classify_fast_decision({}, [])["decision"] == "MALFORMED"


def test_the_budget_guard_refuses_before_the_backend():
    ledgers = runner.Ledgers()
    guard = runner.BudgetGuard(ordering_cap=3, per_unit_arm_cap=2,
                                ledgers=ledgers)
    guard.open_cell()
    for _ in range(2):
        guard.reserve(kind="fast", where={"unit": 1})
        guard.spend(kind="fast")
    with pytest.raises(runner.UnitFault):
        guard.reserve(kind="fast", where={"unit": 1})
    assert ledgers.llm_total() == 2  # the refused call was never billed
    guard.reserve(kind="outer", where={"step": 1})
    guard.spend(kind="outer")
    with pytest.raises(runner.RunFault):
        guard.reserve(kind="outer", where={"step": 1})
    assert ledgers.llm_total() == 3


def test_a_unit_fault_and_a_run_fault_are_different_types():
    assert not issubclass(runner.UnitFault, runner.RunFault)
    assert not issubclass(runner.RunFault, runner.UnitFault)


def test_h_readings_count_turnover_without_fitting_anything():
    reading = runner.h_readings(
        window=2856, treated_prev=["b", "c"], treated_now=["a", "b"],
        per_series_gain={"a": -0.9, "b": 0.4})
    assert reading["attribution"]["new_entrant"] == ["a"]
    assert reading["attribution"]["continuing"] == ["b"]
    assert reading["attribution"]["left"] == ["c"]
    assert reading["H1_new_entrant_share_of_harm"] == 1.0
    assert reading["H3_treated"] == 2


def test_the_course_refuses_phase_f_before_building_anything(tmp_path):
    """The refusal comes before the machinery, so it cannot spend on the way."""
    report = runner.run_course(
        phase="phase_f", ordering_name="forward",
        units=contract.ordering("forward"), run_label="pytest_refusal",
        offline=True, limit=1)
    assert report["status"] == "BLOCKED_ON_CONTRACT"
    assert report["llm_calls"] == 0
    assert report["consumer_fits"] == 0
    assert any("seal release" in blocker or "verdict" in blocker
               for blocker in report["why"])
