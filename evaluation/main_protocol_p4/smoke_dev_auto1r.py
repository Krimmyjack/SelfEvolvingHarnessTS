"""DEV-AUTO-1R preflight: the comparator, the arm symmetry, the Fast budget.

Targeted checks only -- DEV-AUTO-1R 3.2 says to test the repairs, not to re-run
the two arms.  Zero LLM calls and zero Consumer fits: everything here is either
arithmetic on synthetic rows, an assertion about the code that would run, or a
read of a receipt already on disk.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_auto1r_repair as X
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import scope_threshold_tool as tool

ART = base.ROOT / "artifacts" / "main_protocol"


def _cell(gain, *, harmed=0.0, worst=0.0, treated=10, unusable=None):
    if unusable is not None:
        return {"aggregate_gain": None, "unusable": unusable}
    return {"aggregate_gain": gain, "harmed_fraction": harmed,
            "max_single_series_harm": worst, "treated": treated, "served": 20}


def _units(n: int) -> list[dict[str, Any]]:
    return [{"position": i, "unit": {"block": "[0:40]", "origin": 1000 + i}}
            for i in range(n)]


def the_comparator_keeps_three_states_apart() -> dict[str, Any]:
    cases = {
        "a real reading": (_cell(0.2), rev.READ),
        "the window verifier refused": (
            _cell(None, unusable="WINDOW_VERIFIER_REJECTED at origin 1176"),
            rev.RAW_FALLBACK),
        "the served context flattened": (
            _cell(None, unusable="SERVING_CONTEXT_DEGENERATE: preparing ..."),
            rev.RAW_FALLBACK),
        "no entry was ever stored": ({"status": "NO_STORED_READING"},
                                     rev.UNKNOWN),
        "the face could not be evaluated": ({"status": "FaceNotEvaluable"},
                                            rev.UNKNOWN),
        "nothing at all": (None, rev.UNKNOWN),
        "a new fault class defaults to unknown": (
            _cell(None, unusable="SOMETHING_NEW_HAPPENED"), rev.UNKNOWN),
    }
    wrong = {name: rev.classify_reading(value)["state"]
             for name, (value, want) in cases.items()
             if rev.classify_reading(value)["state"] != want}
    fallbacks = [rev.classify_reading(value)
                 for value, want in cases.values() if want == rev.RAW_FALLBACK]
    return {
        "check": "the_comparator_keeps_three_states_apart",
        "passed": not wrong and all(row["aggregate_gain"] == 0.0
                                    for row in fallbacks),
        "misclassified": wrong,
        "a_raw_fallback_scores": [row["aggregate_gain"] for row in fallbacks],
        "an_unknown_scores": rev.classify_reading(None)["aggregate_gain"],
    }


def one_denominator_for_both_sides() -> dict[str, Any]:
    """The position-16 shape: 7 readable of 15, against a parent readable on 15."""
    units = _units(15)
    candidate = {"pos%d" % i: rev.classify_reading(
        _cell(0.30) if i < 7
        else _cell(None, unusable="WINDOW_VERIFIER_REJECTED at origin 1"))
        for i in range(15)}
    reference = {"pos%d" % i: rev.classify_reading(_cell(0.20))
                 for i in range(15)}
    out = rev.compare_on_a_common_denominator(
        units=units, candidate=candidate, reference=reference,
        face="support_face", candidate_label="candidate",
        reference_label="parent", comparison_kind="test")
    mean = out["mean_aggregate_gain"]
    readable_only = round(0.30, 6)
    return {
        "check": "one_denominator_for_both_sides",
        "passed": (out["units_scored_on_both_sides"] == 15
                   and mean["candidate"] == round(7 * 0.30 / 15, 6)
                   and mean["reference"] == 0.20
                   and mean["candidate_beats_the_reference_by_material"] is False
                   and out["verdict_is_complete"]),
        "over_the_declared_units": mean["candidate"],
        "over_only_its_readable_cells_the_old_way": readable_only,
        "the_reference": mean["reference"],
        "which_verdict_flips": ("the old denominator makes this candidate beat "
                               "the parent; the common one does not"),
    }


def an_unknown_cell_withholds_the_verdict() -> dict[str, Any]:
    units = _units(4)
    candidate = {"pos0": rev.classify_reading(_cell(0.5)),
                 "pos1": rev.classify_reading(_cell(0.5)),
                 "pos2": rev.classify_reading(_cell(0.5)),
                 "pos3": rev.classify_reading({"status": "NO_STORED_READING"})}
    reference = {"pos%d" % i: rev.classify_reading(_cell(0.1))
                 for i in range(4)}
    out = rev.compare_on_a_common_denominator(
        units=units, candidate=candidate, reference=reference,
        face="support_face", candidate_label="candidate",
        reference_label="parent", comparison_kind="test")
    frame_b = rev.frame_b_admits(out)
    return {
        "check": "an_unknown_cell_withholds_the_verdict",
        "passed": (out["units_scored_on_both_sides"] == 3
                   and out["verdict_is_complete"] is False
                   and out["unknown_cells"]["candidate"] == ["pos3"]
                   and frame_b["admits"] is None),
        "counted_over": out["mean_aggregate_gain"]["counted_over"],
        "frame_b_verdict": frame_b["admits"],
        "why": out["why_the_verdict_is_withheld"],
        "and_it_was_not_scored_zero": (
            out["per_unit"][3]["candidate"]["aggregate_gain"] is None),
    }


def per_unit_risk_is_not_the_cross_unit_worst() -> dict[str, Any]:
    """A candidate can hold the worst cell and still be worse on another one."""
    units = _units(2)
    candidate = {"pos0": rev.classify_reading(_cell(0.5, worst=0.10)),
                 "pos1": rev.classify_reading(_cell(0.5, worst=0.25))}
    reference = {"pos0": rev.classify_reading(_cell(0.1, worst=0.30)),
                 "pos1": rev.classify_reading(_cell(0.1, worst=0.05))}
    out = rev.compare_on_a_common_denominator(
        units=units, candidate=candidate, reference=reference,
        face="support_face", candidate_label="candidate",
        reference_label="parent", comparison_kind="test")
    worst = out["cross_unit_worst"]["single_series_harm"]
    per_unit = out["per_unit_risk"]
    return {
        "check": "per_unit_risk_is_not_the_cross_unit_worst",
        "passed": (worst["candidate_worse"] is False
                   and per_unit["units_where_the_candidate_is_worse"][
                       "single_series_harm"] == ["pos1"]
                   and per_unit[
                       "cross_unit_worst_hides_a_per_unit_regression"][
                           "single_series_harm"] is True),
        "cross_unit_worst": worst,
        "units_where_it_is_worse": per_unit[
            "units_where_the_candidate_is_worse"],
    }


def the_control_freezes_only_the_skill() -> dict[str, Any]:
    source = inspect.getsource(R.build)
    frozen_spec = 'FROZEN_ARM: runner.ArmSpec(FROZEN_ARM, "k0", True, False, True)'
    cell_source = inspect.getsource(R.run_cell)
    findings = {
        "the_control_carries_its_state_across_units": frozen_spec in source,
        "no_per_cell_reseed_in_the_course_loop": (
            "seed=None" in source and "_seed_arm(a, state" not in source),
        "both_arms_are_seeded_once_at_the_same_entry_state":
            source.count("_seed_arm(arm, state, fresh_ledger=True)") == 1,
        "the_revision_channel_is_named_to_one_arm":
            "arms[REVISING_ARM], position=row[\"position\"]" in source,
        "the_invariant_is_checked_not_assumed":
            "the_control_never_revised_its_card" in source,
        "the_cell_still_takes_a_seed_hook_but_nobody_passes_one": (
            "seed: Any = None" in cell_source),
    }
    # The three places ``write_back`` gates behaviour, now live for both arms.
    gated = [line.strip() for line in cell_source.splitlines()
             if "arm.spec.write_back" in line]
    return {
        "check": "the_control_freezes_only_the_skill",
        "passed": all(findings.values()) and len(gated) == 2,
        "findings": findings,
        "write_back_gates_in_the_cell": gated,
        "what_those_gates_now_allow_the_control": [
            "activating an approved Skill after the delayed gate passes",
            "extending its own bank and processed list",
        ],
    }


def fast_budget_is_the_package_allowance() -> dict[str, Any]:
    ledgers = runner.Ledgers()
    guard = runner.BudgetGuard(ordering_cap=1000, per_unit_arm_cap=5,
                               ledgers=ledgers)
    # The cell rewrites the cap; the guard must honour the rewritten number.
    guard.open_cell()
    guard.per_unit_arm_cap = 9
    allowed = 0
    refused: str | None = None
    for _ in range(12):
        try:
            guard.reserve(kind="fast", where={"unit": "t"})
        except runner.UnitFault as exc:
            refused = str(exc)[:80]
            break
        guard.spend(kind="fast", calls=1)
        allowed += 1

    budget = R.PackageBudget()
    room_at_start = budget.llm_room(R.FROZEN_ARM)
    budget.spend_llm(R.FROZEN_ARM, 30)
    after_thirty = budget.llm_room(R.FROZEN_ARM)
    other_arm_untouched = budget.llm_room(R.REVISING_ARM)

    cell_source = inspect.getsource(R.run_cell)
    build_source = inspect.getsource(R.build)
    findings = {
        "the_guard_honours_a_rewritten_cap": allowed == 9 and refused is not None,
        "the_cap_is_rewritten_right_after_open_cell": (
            "guard.open_cell()" in cell_source
            and "guard.per_unit_arm_cap = int(budget.llm_room(" in cell_source),
        "the_backend_no_longer_holds_the_five":
            "_default_backend_factory(PER_ARM_LLM)" in build_source,
        "the_allowance_is_symmetric": room_at_start == other_arm_untouched,
        "spending_is_charged_to_one_arm_only": (
            after_thirty == room_at_start - 30
            and other_arm_untouched == room_at_start),
        "the_hec1_contract_is_untouched":
            int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]) == 5,
        "the_other_stops_are_still_there": (
            "WALL_CLOCK_CEILING_REACHED" in inspect.getsource(
                R.PackageBudget.stop_reason)
            and "FIT_CEILING_REACHED" in inspect.getsource(
                R.PackageBudget.stop_reason)),
    }
    return {
        "check": "fast_budget_is_the_package_allowance",
        "passed": all(findings.values()),
        "findings": findings,
        "calls_allowed_under_a_cap_of_nine": allowed,
        "refusal": refused,
        "per_arm_allowance": R.PER_ARM_LLM,
        "package_ceiling": R.MAX_LLM,
    }


def the_released_admission_rule_is_installed() -> dict[str, Any]:
    """DEV-AUTO-1 ran Fast under the module default, which is the strict gate.

    The strict rule admits a probe only when no series is harmed at all, and
    it refuses with ``relation_not_positive`` -- a reason that is not in
    ``RISK_REFUSAL_REASONS``, so a refused probe never reached the risk-refusal
    ledger and never became the evidence the revision channel consumes.
    """
    from evaluation.main_protocol_p4 import p4b_contract as bounded
    from SelfEvolvingHarnessTS.methods.ttha import admission_policy as ap
    source = inspect.getsource(R.build)
    findings = {
        "the_dev_runner_installs_a_policy":
            'machinery["admission_policy"].install_policy(' in source,
        "and_it_is_the_released_one": "bounded.BOUNDED_POLICY" in source,
        "the_released_one_is_the_bounded_rule":
            bounded.BOUNDED_POLICY.rule == ap.BOUNDED_V1,
        "with_the_released_constants": (
            bounded.BOUNDED_POLICY.max_harmed_fraction
            == contract.RISK["max_harmed_fraction"]
            and bounded.BOUNDED_POLICY.max_single_series_harm
            == contract.RISK["max_single_series_harm"]),
        "no_new_constant_was_minted":
            bounded.BOUNDED_POLICY.to_dict() != ap.DEFAULT.to_dict(),
    }
    return {
        "check": "the_released_admission_rule_is_installed",
        "passed": all(findings.values()),
        "findings": findings,
        "module_default_that_run2_used": ap.DEFAULT.to_dict(),
        "installed_from_now_on": bounded.BOUNDED_POLICY.to_dict(),
        "note": ("run2 was NOT re-run; its numbers still carry the strict "
                 "rule and are labelled as such"),
    }


def the_three_ways_a_cell_stops_are_counted_apart() -> dict[str, Any]:
    class _Result:
        risk_refusals = [{"candidate_id": "c1"}, {"candidate_id": "c2"}]

    truncated = R._classify_cell_stop(
        {"faults": [{"why": "this unit-arm's LLM budget of 5 is spent; the "
                            "cell abstains to identity"}],
         "llm_cap_in_force_this_cell": 5, "llm_calls_this_cell": 5}, None)
    abstained = R._classify_cell_stop(
        {"faults": [], "fast_decision": {"decision": "ABSTAINED_WITH_REASON"},
         "llm_cap_in_force_this_cell": 200, "llm_calls_this_cell": 1},
        _Result())
    findings = {
        "a_truncated_cell_is_marked_truncated":
            truncated["llm_budget_truncated_this_cell"] is True,
        "and_is_not_called_an_abstention":
            truncated["model_abstained_with_a_reason"] is False,
        "an_abstention_is_marked_an_abstention":
            abstained["model_abstained_with_a_reason"] is True,
        "and_is_not_called_a_truncation":
            abstained["llm_budget_truncated_this_cell"] is False,
        "risk_refusals_are_their_own_count":
            abstained["risk_refusals_this_cell"] == 2,
    }
    return {"check": "the_three_ways_a_cell_stops_are_counted_apart",
            "passed": all(findings.values()), "findings": findings,
            "truncated": truncated, "abstained": abstained}


def nothing_in_the_risk_or_scope_instrument_moved() -> dict[str, Any]:
    risk = dict(contract.RISK)
    findings = {
        "material": risk["material"] == 0.005,
        "max_harmed_fraction": risk["max_harmed_fraction"] == 0.20,
        "max_single_series_harm": risk["max_single_series_harm"] == 0.30,
        "min_treated": risk["min_treated"] == 5,
        # The grid is NOT one tuple: six features carry (0, 1, 3, 6), four
        # carry (0, .01, .05, .2), and two carry their own.  DEV-AUTO-1 said
        # all eight Scope failures came from "the fraction-type features
        # keeping the default (0,1,3,6)"; that description of the instrument
        # was wrong, so the check pins the grid it actually has.
        "the_frozen_bin_grid": tool.frozen_bins() == {
            "missing_fraction": (0.0, 0.01, 0.05, 0.2),
            "longest_missing_run_fraction": (0.0, 0.01, 0.05, 0.2),
            "local_robust_z_peak": (0.0, 1.0, 3.0, 6.0),
            "estimated_region_start_fraction": (0.0, 0.01, 0.05, 0.2),
            "estimated_region_end_fraction": (0.0, 0.01, 0.05, 0.2),
            "level_region_fraction": (0.0, 1.0, 3.0, 6.0),
            "level_region_end_fraction": (0.0, 1.0, 3.0, 6.0),
            "outlier_region_end_fraction": (0.0, 1.0, 3.0, 6.0),
            "level_excursion_score": (0.0, 1.0, 3.0, 6.0),
            "estimated_level_offset": (0.0, 1.0, 3.0, 6.0),
            "period_change_score": (0.0, 0.1, 0.25, 0.5),
            "period_reliability": (0.0, 0.25, 0.5, 0.75),
        },
        "no_new_support_coverage_floor": (
            "min_treated" not in inspect.getsource(X._support_admits)),
    }
    return {"check": "nothing_in_the_risk_or_scope_instrument_moved",
            "passed": all(findings.values()), "findings": findings,
            "risk": risk,
            "bins": {name: list(edges)
                     for name, edges in tool.frozen_bins().items()}}


def the_evaluation_face_is_not_reachable_from_this_package() -> dict[str, Any]:
    # Scanned over the code, not the prose: the module's own disclaimer says
    # the +144 face was not read, and a scan that matched its own denial would
    # be checking nothing.
    code = chr(10).join(
        line for line in inspect.getsource(X).splitlines()
        if not line.lstrip().startswith(("#", '"', "'"))
        and "what_this_is_not" not in line
        and "not a claim about" not in line
        and "TARGET_HELD_IN unit -- none was read" not in line)
    banned = [name for name in ("EVALUATION_OFFSET", "evaluation_origin",
                                "_evaluate_face", "+ 144", "+144")
              if name in code]
    faces = sorted({X.SUPPORT_FACE, X.DELAYED_FACE})
    from evaluation.main_protocol_p4 import (
        run_m_r0k_scope_workflow_development as m_r0k)
    origins_available = sorted(inspect.getsource(m_r0k._faces_for).count(name)
                               for name in ("support_face", "delayed_face"))
    return {
        "check": "the_evaluation_face_is_not_reachable_from_this_package",
        "passed": (not banned and faces == ["delayed_face", "support_face"]
                   and origins_available == [1, 1]),
        "banned_names_found_in_code": banned,
        "faces_this_package_reads": faces,
        "the_face_helper_offers_only_these_two": True,
    }


CHECKS: tuple[Callable[[], dict[str, Any]], ...] = (
    the_comparator_keeps_three_states_apart,
    one_denominator_for_both_sides,
    an_unknown_cell_withholds_the_verdict,
    per_unit_risk_is_not_the_cross_unit_worst,
    the_control_freezes_only_the_skill,
    fast_budget_is_the_package_allowance,
    the_released_admission_rule_is_installed,
    the_three_ways_a_cell_stops_are_counted_apart,
    nothing_in_the_risk_or_scope_instrument_moved,
    the_evaluation_face_is_not_reachable_from_this_package,
)


def run() -> dict[str, Any]:
    rows = []
    for check in CHECKS:
        try:
            rows.append(check())
        except Exception as exc:  # noqa: BLE001 - a fault is a failed check
            rows.append({"check": check.__name__, "passed": False,
                         "fault": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
    return {"stage": "DEV_AUTO1R_SMOKE",
            "passed": all(row["passed"] for row in rows),
            "llm_calls": 0, "consumer_fits": 0, "checks": rows}


def main(argv=None) -> int:
    result = run()
    print("smoke: %s | llm %d | fits %d"
          % ("PASS" if result["passed"] else "FAIL",
             result["llm_calls"], result["consumer_fits"]))
    for row in result["checks"]:
        print("   %s %s" % ("ok " if row["passed"] else "FAIL", row["check"]))
        if not row["passed"]:
            print("       %s" % json.dumps(
                {k: v for k, v in row.items() if k not in ("check", "passed")},
                ensure_ascii=False, default=str)[:600])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
