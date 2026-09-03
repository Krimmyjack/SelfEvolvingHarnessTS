"""The HEC-1 arm loop, driven end to end on real data at 0 LLM.

Why this exists rather than a fixture
-------------------------------------
This line has paid for the other kind twice: a fixture that did not match
production (#42b's eight-cell smoke missed the whole positive branch), and a
verdict written from a run that never reached its own gates.  So nothing here is
mocked except the model's answers.  The real ``TTHAMethod``, the real
``run_online_round``, the real scoped serving evaluator, the real risk gate, the
real Draft ledger and the real outer loop all run on real KDD units; the Fast
Path answers come from ``SealedProbeBackend`` and the outer Slow call from a
scripted callable, and both are billed as zero because neither touches a relay.

What that buys is coverage of the six things sol required, on the code that will
actually run the course:

* the outer loop fires on schedule and its Draft comes out of a replay screen;
* the authoritative gate is the only thing that can activate, and a disagreement
  with the lifecycle event is recorded rather than reconciled;
* every denominator is the served count read from the roster, including on the
  39-series block where it is 19 rather than 20;
* the frozen arm is rebuilt from its start snapshot at every unit, so it cannot
  carry a Skill forward even by accident;
* the evaluation face is scored and never enters a bank;
* ``--resume`` replays the checkpoints and spends nothing.

One real defect is asserted as a defect: the +144 evaluation face is not always
evaluable, and the unit then contributes no curve point **for every arm equally**.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

from evaluation.main_protocol_p4 import hec1_contract as contract  # noqa: E402
from evaluation.main_protocol_p4 import restricted_draft as drafts  # noqa: E402
from evaluation.main_protocol_p4 import run_hec1 as runner  # noqa: E402

#: Five units reaches the first outer step at k=5 and still runs in seconds.
UNITS = 5


@pytest.fixture(scope="module")
def course(tmp_path_factory):
    label = "pytest_e2e_%s" % tmp_path_factory.mktemp("hec1").name
    report = runner.run_course(
        phase="phase_t_forward", ordering_name="forward",
        units=contract.ordering("forward"), run_label=label,
        offline=True, limit=UNITS)
    assert report["status"] == "COMPLETE", report.get("run_fault")
    return report


@pytest.fixture(scope="module")
def resumed(course):
    return runner.run_course(
        phase="phase_t_forward", ordering_name="forward",
        units=contract.ordering("forward"), run_label=course["run_label"],
        offline=True, limit=UNITS, resume=True)


def _cells(course, arm=None):
    return [row for row in course["cells"]
            if arm is None or row["arm"] == arm]


# ---------------------------------------------------------------------------
# it ran at all, and it ran for free
# ---------------------------------------------------------------------------

def test_the_course_completed_every_unit_for_every_arm(course):
    assert course["units_completed"] == UNITS
    assert set(course["arms"]) == {"Static", "A3-frozen", "A3-online"}
    for arm in course["arms"]:
        assert len(_cells(course, arm)) == UNITS


def test_no_llm_call_was_billed_and_the_scripted_ones_are_counted_apart(course):
    assert course["ledgers"]["llm_total"] == 0
    assert course["ledgers"]["llm_fast"] == 0
    assert course["ledgers"]["llm_outer"] == 0
    # A scripted call that billed would make "0 LLM" a claim the artifact
    # contradicts; one that vanished would make "nothing ran" indistinguishable.
    assert course["budget_guard"]["scripted_calls_not_billed"] >= 1


def test_the_harness_actually_probed_and_deployed_rather_than_abstaining(course):
    """A hollow pass is the failure mode this assertion exists for."""
    probed = [row for row in _cells(course)
              if row.get("probes") and len(row["probes"]) > 0]
    assert probed, "no arm ran a single probe; the test would be vacuous"
    deployed = [row for row in _cells(course) if row.get("deployed")]
    assert deployed, "no arm deployed anything; the gates were never reached"


# ---------------------------------------------------------------------------
# the outer loop
# ---------------------------------------------------------------------------

def test_the_outer_loop_fires_on_schedule_and_only_for_online_arms(course):
    steps = course["outer_steps"]
    assert steps, "the outer loop never fired at k=%s" % contract.OUTER_LOOP[
        "period_k_units"]
    assert {step["arm"] for step in steps} == {"A3-online"}
    assert all(step["k_index"] == 1 for step in steps)


def test_the_outer_step_screened_on_replay_and_produced_only_a_draft(course):
    step = course["outer_steps"][0]
    assert step["wrote_active"] is False
    if step["drafts_opened"]:
        assert step["replay_fits"] > 0, (
            "a Draft was opened without any replay screen having run")
        lifecycle = course["lifecycle"]["A3-online"]
        opened = {draft["draft_id"] for draft in lifecycle["drafts"]}
        assert set(step["drafts_opened"]) <= opened
        for draft in lifecycle["drafts"]:
            assert draft["deployable"] is False


def test_a_calibrated_clause_came_from_a_frozen_bin_edge_and_dropped_slows_number(
        course):
    calibrations = [row["calibration"] for row in
                    course["outer_steps"][0]["candidates"]
                    if (row.get("calibration") or {}).get("outcome")
                    == "CALIBRATED"]
    if not calibrations:
        pytest.skip("no candidate needed a clause on this five-unit course")
    calibration = calibrations[0]
    assert calibration["threshold_is_a_frozen_bin_edge"] is True
    assert calibration["threshold"] in calibration["bin_edges"]
    # The scripted Slow deliberately returns 4.25, which is not a bin edge.
    assert calibration["slow_threshold_as_returned"] == 4.25
    assert "LLM_THRESHOLD_IGNORED" in calibration["notes"]


def test_the_scopefit_shadow_is_recorded_beside_the_slow_choice(course):
    records = course["shadow_records"]
    if not records:
        pytest.skip("no Slow proposal on this five-unit course")
    for record in records:
        assert record["shadow"]["deployable"] is False
        assert record["slow"]["feature"]
        assert "agree" in record


def test_the_replay_allowance_is_a_share_of_the_projected_course_fits(course):
    allowance = course["replay_fit_allowance"]
    assert allowance["share"] == contract.REPLAY_FITS_SHARE == 1.0
    # sol v1.1 ruling 2: each online arm's cap is the share of that arm's OWN
    # projected course fits -- never a slice of a pool that includes the frozen
    # arm's fits, and never one remainder two arms share.  The v1 Forward that
    # ran under 0.25 of all LLM arms' fits is FORWARD_SHAKEDOWN.
    assert allowance["allowance_per_online_arm"] == int(
        allowance["share"] * allowance["projected_fits_per_llm_arm"])
    online = [arm for arm in course["arms"] if arm.endswith("-online")]
    assert set(allowance["spent_by_arm"]) == set(online)
    assert (allowance["allowance"]
            == allowance["allowance_per_online_arm"] * len(online))
    assert allowance["within"] is True
    assert allowance["record"]["v1_forward_shakedown_ran_under"] == 0.25
    for spent in allowance["spent_by_arm"].values():
        assert spent <= allowance["allowance_per_online_arm"]
    # Replay fits are their own ledger line, never folded into course fits.
    ledgers = course["ledgers"]
    assert ledgers["replay_fits"] == allowance["spent"]
    assert "replay_fits" in ledgers and "course_fits" in ledgers


# ---------------------------------------------------------------------------
# the one authoritative gate
# ---------------------------------------------------------------------------

def test_only_the_authoritative_gate_can_activate(course):
    for row in _cells(course):
        disagreement = row.get("gate_disagreement")
        if disagreement is None:
            assert not row.get("activated")
            continue
        assert disagreement["resolved_by"] == "p4_gate"
        if not disagreement["may_activate"]:
            assert not row.get("activated"), (
                "a cell activated while the authoritative gate refused")


def test_a_refusing_gate_never_produced_an_activation_and_was_classified(course):
    refused = [row for row in _cells(course)
               if (row.get("delayed") or {}).get("gate")
               and not row["delayed"]["gate"]["passes"]]
    for row in refused:
        assert row["activated"] is False
        if row.get("restricted_state"):
            assert row["restricted_state"] in (
                drafts.WAITING, drafts.REVISABLE, drafts.FLAGGED)


def test_the_frozen_arm_never_activates_even_when_the_gate_passes(course):
    passing = [row for row in _cells(course, "A3-frozen")
               if (row.get("delayed") or {}).get("gate", {}).get("passes")]
    assert passing, "the frozen arm never cleared a gate; assertion is vacuous"
    assert all(row["activated"] is False for row in passing)


def test_a_gate_disagreement_is_recorded_rather_than_reconciled(course):
    """The exact Source-v3 round-2856 shape, if it recurs here."""
    disagreements = [row for row in _cells(course)
                     if (row.get("gate_disagreement") or {}).get("disagree")]
    for row in disagreements:
        assert row["gate_disagreement"]["resolved_by"] == "p4_gate"
        assert not row["activated"] or row["gate_disagreement"]["may_activate"]


# ---------------------------------------------------------------------------
# the denominator (sol's ruling 2)
# ---------------------------------------------------------------------------

def test_every_coverage_denominator_is_the_served_count_from_the_roster(course):
    for row in _cells(course):
        assert row["served_denominator_source"] == "roster eval rows"
        reading = row.get("evaluation")
        if reading is None:
            continue
        assert reading["served"] == row["served"]
        expected = (round(reading["treated"] / reading["served"], 4)
                    if reading["served"] else 0.0)
        assert reading["coverage"] == expected


def test_the_thirty_nine_series_block_serves_nineteen_on_its_second_face():
    """readable[200:239] is why a denominator fixed at 20 would be wrong."""
    uids = runner.block_uids([200, 239])
    assert len(uids) == 39
    cell, _variant = runner.baselines._cell(uids)
    assert len(cell.support_a) == 20
    assert len(cell.support_b) == 19
    face_a = [row for row in cell.roster("support_a") if row["role"] == "eval"]
    face_b = [row for row in cell.roster("support_b") if row["role"] == "eval"]
    assert (len(face_a), len(face_b)) == (20, 19)


def test_a_phase_s_unit_on_that_block_reads_its_own_served_count():
    unit = next(row for row in contract.phase_s_units()
                if row["block"] == "[200:239]")
    ctx = runner.UnitContext(unit)
    assert len(ctx.eval_uids) == len(
        [row for row in ctx.roster if row["role"] == "eval"])
    assert len(ctx.eval_uids) == 20  # face A of a 39-series block


def test_the_source_scan_finds_no_hard_coded_served_count():
    assert contract._hardcoded_denominator_scan() == []


# ---------------------------------------------------------------------------
# the frozen reset
# ---------------------------------------------------------------------------

def test_the_frozen_arm_is_rebuilt_at_every_unit_after_the_first(course):
    rows = sorted(_cells(course, "A3-frozen"), key=lambda r: r["position"])
    assert rows[0]["reset"]["reset"] is False        # nothing to reset yet
    assert all(row["reset"]["reset"] is True for row in rows[1:])


def test_the_online_arm_keeps_its_store_and_the_frozen_arm_keeps_no_bank(course):
    frozen = sorted(_cells(course, "A3-frozen"), key=lambda r: r["position"])
    online = sorted(_cells(course, "A3-online"), key=lambda r: r["position"])
    assert all((row.get("bank_rows_after") or 0) == 0 for row in frozen)
    banks = [row.get("bank_rows_after") or 0 for row in online]
    assert max(banks) > 0, "the online arm accumulated nothing to learn from"
    assert banks == sorted(banks), "the online bank shrank"
    assert all(row["reset"]["reset"] is False for row in online)


def test_the_frozen_arm_carries_no_active_skill_across_units(course):
    assert course["active_skill_ids"]["A3-frozen"] == []


def test_open_online_drafts_are_closed_by_reason_at_the_end_of_the_course(course):
    """A Draft still open when the units run out is archived by the ledger's
    own reason (WAITING -> PATTERN_NOT_REENCOUNTERED, never verified ->
    NEVER_VERIFIED, ...), so the lifecycle table cannot report an archived
    Draft as a live one.  Frozen arms have nothing to close."""
    closures = course["course_end_closures"]
    assert set(closures) == {"A3-online"}
    legal = set(drafts.CLOSE_REASONS.values()) | {"OUT_OF_UNITS", "NEVER_VERIFIED"}
    for row in closures["A3-online"]:
        assert row["closed"] in legal, row
    for draft in course["lifecycle"]["A3-online"]["drafts"]:
        assert draft["closed"] is not None, draft["draft_id"]
    assert course["lifecycle"]["A3-online"]["open"] == []


# ---------------------------------------------------------------------------
# the K0 handoff
# ---------------------------------------------------------------------------

def test_a_k0_receipt_names_the_snapshot_phase_t_compiles_from(tmp_path, monkeypatch):
    """The first receipt carried only skill ids, and ``run_course`` then fell
    back to h0 for both A5 arms.  The receipt now names the snapshot."""
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    run_root = tmp_path / ".hec1_runs" / "phase_s_x"
    store = run_root / runner.PHASE_S_ARM / "store_online"
    (store / "abc123").mkdir(parents=True)
    (run_root / runner.PHASE_S_ARM / "active.json").write_text(
        json.dumps({"runtime_bundle_sha": "abc123"}), encoding="utf-8")
    report = {"run_root": ".hec1_runs/phase_s_x", "run_label": "phase_s_x",
              "units_completed": 13,
              "active_skill_ids": {"A5-online": ["fast_winner_x"]}}
    receipt = runner.phase_s_k0(report)
    assert receipt["empty"] is False
    assert receipt["runtime_bundle_sha"] == "abc123"
    assert receipt["store_root"] == ".hec1_runs/phase_s_x/A5-online/store_online"
    assert receipt["snapshot_resolved"] is True

    empty = runner.phase_s_k0({**report, "active_skill_ids": {"A5-online": []}})
    assert empty["empty"] is True
    assert empty["store_root"] is None and empty["runtime_bundle_sha"] is None

    (run_root / runner.PHASE_S_ARM / "active.json").unlink()
    unresolved = runner.phase_s_k0(report)
    assert unresolved["snapshot_resolved"] is False
    assert "k0_fault" in unresolved


def test_phase_t_refuses_a_non_empty_k0_it_cannot_compile():
    """Fail closed: never start the A5 arms from h0 under a K0 label."""
    report = runner.run_course(
        phase="phase_t_forward", ordering_name="forward",
        units=contract.ordering("forward"), run_label="pytest_k0_unresolved",
        offline=True, limit=1,
        k0={"active_skill_ids": ["fast_winner_forecast_ridge_smase_outlier_mad"]})
    assert report["status"] == "BLOCKED"
    assert report["verdict"] == "RUN_BLOCKED_NO_VERDICT"
    assert report["run_fault"].startswith("K0_SNAPSHOT_UNRESOLVED")
    assert report["consumer_fits"] == 0 and report["llm_calls"] == 0
    assert "cells" not in report, "an arm ran before the K0 check"


def test_deployed_via_reads_both_the_candidate_source_and_the_active_set(course):
    """sol final ruling SS4: two facts, recorded apart, and the label derived
    from both rather than from the candidate id alone."""
    legal = {"identity", "recalled_skill", "resupplied_draft",
             "searched_active_program", "searched_this_unit"}
    for row in _cells(course):
        if row["arm"] == "Static":
            continue
        assert row["deployed_via"] in legal, row["deployed_via"]
        assert isinstance(row["winner_from_skill_candidate"], bool)
        assert isinstance(row["program_in_active_set_at_start"], bool)
        assert isinstance(row["active_program_signatures_at_start"], dict)
        if row["deployed_via"] == "searched_active_program":
            assert row["program_in_active_set_at_start"] is True
        if row["deployed_via"] == "recalled_skill":
            assert row["winner_from_skill_candidate"] is True
        assert isinstance(row.get("lost_activation", False), bool)
    # The map only ever grows in the online arm, and a minted card maps its
    # own program: the unit after an activation must see it in the set.
    online = sorted(_cells(course, "A3-online"), key=lambda r: r["position"])
    for earlier, later in zip(online, online[1:]):
        if earlier.get("activated") and earlier.get("skills_minted_this_unit"):
            assert later["active_program_signatures_at_start"], (
                "an activation left the next unit's active-program map empty")


def test_a_lost_activation_is_recorded_and_never_activates(course):
    for row in _cells(course):
        if row.get("lost_activation"):
            assert row["activated"] is False
            assert row["gate_disagreement"]["may_activate"] is True
            assert "lost_activation_why" in row


@pytest.fixture(scope="module")
def phase_s_course(tmp_path_factory):
    label = "pytest_s3_%s" % tmp_path_factory.mktemp("hec1s").name
    report = runner.run_course(
        phase="phase_s", ordering_name="phase_s",
        units=contract.phase_s_units(), run_label=label, offline=True, limit=3)
    assert report["status"] == "COMPLETE", report.get("run_fault")
    return report


def test_the_k0_freeze_audit_passes_on_what_phase_s_actually_produced(phase_s_course):
    from evaluation.main_protocol_p4 import audit_hec1_k0_freeze as k0_audit

    receipt = runner.phase_s_k0(phase_s_course)
    payload = k0_audit.audit(phase_s_course, receipt)
    assert payload["checks_total"] == len(k0_audit.CHECK_NAMES)
    assert payload["passed"], [row for row in payload["checks"]
                               if not row["passed"]]
    assert payload["k0_empty"] == receipt["empty"]
    arm_set = [row for row in payload["checks"]
               if row["check"] == "arm_set_for_phase_t"][0]
    if receipt["empty"]:
        assert arm_set["arms"] == ["Static", "A3-frozen", "A3-online"]
        assert arm_set["criterion_3_scored"] is False
    else:
        assert arm_set["arms"] == ["Static", "A5-frozen", "A5-online",
                                   "A3-online"]
        assert receipt["snapshot_resolved"] is True


def test_the_k0_freeze_audit_fails_a_receipt_that_disagrees_with_its_course(
        phase_s_course):
    from evaluation.main_protocol_p4 import audit_hec1_k0_freeze as k0_audit

    receipt = runner.phase_s_k0(phase_s_course)
    forged = {**receipt, "active_skill_ids": ["fast_winner_forged"],
              "empty": False, "store_root": None, "runtime_bundle_sha": None,
              "snapshot_resolved": False}
    payload = k0_audit.audit(phase_s_course, forged)
    assert payload["passed"] is False
    failed = {row["check"] for row in payload["checks"] if not row["passed"]}
    assert {"receipt_matches_course", "snapshot_resolves"} <= failed


def test_the_frozen_arm_is_never_resupplied_a_draft_from_an_earlier_unit(course):
    """The leak the first offline course had: the store was rebuilt, the Draft
    ledger was not, and ``resupplied_draft_1`` from unit 1896 kept feeding the
    frozen arm's Fast Path for the rest of the course.  Anything resupplied to
    a frozen arm is memory across units, whatever it is called."""
    frozen = sorted(_cells(course, "A3-frozen"), key=lambda r: r["position"])
    online = sorted(_cells(course, "A3-online"), key=lambda r: r["position"])
    assert all(not row.get("resupplied_candidate_ids") for row in frozen), [
        (row["position"], row.get("resupplied_candidate_ids")) for row in frozen]
    # The same course must show the online arm *being* resupplied, or the
    # frozen assertion above is vacuous: nothing was ever there to leak.
    assert any(row.get("resupplied_candidate_ids") for row in online), (
        "the online arm was never resupplied a Draft; the frozen check proves "
        "nothing on this course")
    dropped = [row["reset"].get("dropped_drafts") for row in frozen[1:]]
    assert all(value is not None for value in dropped)
    assert any(value > 0 for value in dropped), (
        "no frozen rebuild ever had a Draft to drop; the reset path was not "
        "exercised against a non-empty ledger")


# ---------------------------------------------------------------------------
# the evaluation face
# ---------------------------------------------------------------------------

def test_the_evaluation_face_is_one_hundred_and_forty_four_steps_on(course):
    for row in _cells(course):
        reading = row.get("evaluation")
        if reading is None:
            continue
        assert reading["origin"] == row["unit"]["origin"] + 144


def test_the_evaluation_face_never_enters_a_bank(course):
    for row in _cells(course):
        assert row.get("evaluation_face_enters_bank") is False
    evaluation_origins = {row["unit"]["origin"] + 144 for row in _cells(course)}
    for arm_rows in course["lifecycle"].values():
        for draft in arm_rows["drafts"]:
            for entry in draft.get("history") or ():
                assert entry.get("window") not in evaluation_origins, (
                    "an evaluation-face window reached the Draft lifecycle")


def test_a_unit_whose_evaluation_face_has_no_truth_drops_out_for_every_arm(course):
    """A property of the data, so it must not hit one arm and spare another."""
    unreadable: dict[int, set[str]] = {}
    for row in _cells(course):
        if row.get("evaluation") is None:
            unreadable.setdefault(row["unit"]["origin"], set()).add(row["arm"])
    for origin, arms in unreadable.items():
        assert arms == set(course["arms"]), (
            "origin %d lost its evaluation face for %s but not for the rest"
            % (origin, sorted(arms)))


def test_static_is_exactly_static_on_the_evaluation_face(course):
    for row in _cells(course, "Static"):
        reading = row.get("evaluation")
        if reading is None:
            continue
        assert reading["aggregate_gain"] == 0.0
        assert reading["treated"] == 0
        assert reading["mean_smase"] == reading["static_mean_smase"]


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_resume_replays_every_checkpoint_and_spends_nothing(course, resumed):
    assert resumed["status"] == "COMPLETE"
    assert resumed["units_completed"] == course["units_completed"]
    assert all(row.get("resumed") for row in resumed["cells"])
    assert resumed["ledgers"]["course_fits"] == 0
    assert resumed["ledgers"]["llm_total"] == 0


def test_resume_reproduces_the_same_readings(course, resumed):
    def index(report):
        return {(row["arm"], row["position"]):
                (row.get("evaluation") or {}).get("aggregate_gain")
                for row in report["cells"]}

    assert index(resumed) == index(course)


def test_a_heartbeat_was_written_where_a_monitor_can_find_it(course):
    path = runner._run_root(course["run_label"]) / "heartbeat.json"
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pid"] > 0
    assert payload["phase"] == "phase_t_forward"


# ---------------------------------------------------------------------------
# the refusals that must survive the freeze
# ---------------------------------------------------------------------------

def test_phase_f_is_still_refused_to_a_runner_that_asks_itself():
    assert not contract.assert_launchable("phase_f")["launchable"]
    assert not contract.assert_launchable(
        "phase_f", verdict="HEC1_EVOLUTION_SUPPORTED")["launchable"]
    assert not contract.assert_launchable(
        "phase_f", verdict="HEC1_INCONCLUSIVE", seal_released=True)["launchable"]
    # sol v1.1 ruling 5: a supported verdict plus a human seal is still not
    # enough on an empty K0 -- A3 does not stand in for A5.
    empty_k0 = contract.assert_launchable(
        "phase_f", verdict="HEC1_EVOLUTION_SUPPORTED", seal_released=True)
    assert not empty_k0["launchable"]
    assert any("non-empty K0" in blocker for blocker in empty_k0["blockers"])
    assert not contract.assert_launchable(
        "phase_f", verdict="HEC1_P1_ONLY__RECALL_ACCUMULATION",
        seal_released=True, k0_nonempty=True)["launchable"]
    assert contract.assert_launchable(
        "phase_f", verdict="HEC1_EVOLUTION_SUPPORTED",
        seal_released=True, k0_nonempty=True)["launchable"]


def test_the_offline_transport_can_never_reach_a_relay():
    assert runner.OFFLINE_BASE_URL.endswith("/v1")
    assert ".invalid/" in runner.OFFLINE_BASE_URL
