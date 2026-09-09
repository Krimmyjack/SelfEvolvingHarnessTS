"""DEV-AUTO-2 preflight: the Frame C wiring, before any live call.

Small and targeted, as the package asks: it checks the things this package
newly wires -- the comparator in the live step, Frame C's verdict split, the
declared order actually being applied, the production supply path, and the
boundary -- and nothing else.  Zero LLM calls, zero Consumer fits.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_auto2_frame_c as D2
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import scope_threshold_tool as tool


def _cell(position, gain, *, harmed=0.0, worst=0.0, treated=10, unusable=None):
    unit = {"block": "[0:40]", "origin": 1000 + position}
    if unusable is not None:
        return {"unit": unit, "aggregate_gain": None, "unusable": unusable}
    return {"unit": unit, "aggregate_gain": gain, "harmed_fraction": harmed,
            "max_single_series_harm": worst, "treated": treated, "served": 20}


def _index(n):
    return {("[0:40]", 1000 + i): i for i in range(n)}


def _by_position(n):
    return {i: {"position": i, "unit": {"block": "[0:40]", "origin": 1000 + i}}
            for i in range(n)}


def the_comparator_is_in_the_live_step() -> dict[str, Any]:
    """The step's comparison goes through the common-denominator function."""
    source = inspect.getsource(D2.revision_step)
    candidate = [_cell(i, 0.30) if i < 3
                 else _cell(i, None,
                            unusable="WINDOW_VERIFIER_REJECTED at origin 1")
                 for i in range(6)]
    parent = [_cell(i, 0.20) for i in range(6)]
    out = D2.compare_to_the_parent(candidate, parent, index=_index(6),
                                   by_position=_by_position(6),
                                   candidate_label="c")
    findings = {
        "the_step_keeps_the_parents_cells": "parent_cells = list(" in source,
        "and_passes_them_to_the_comparator": "parent_cells=parent_cells" in source,
        "one_declared_unit_set": out["units_declared"] == 6,
        "the_refused_cells_stay_in_the_denominator": (
            out["mean_aggregate_gain"]["candidate"] == round(3 * 0.30 / 6, 6)),
        "three_states_survive": out["candidate_cell_states"] == {
            rev.READ: 3, rev.RAW_FALLBACK: 3, rev.UNKNOWN: 0},
    }
    return {"check": "the_comparator_is_in_the_live_step",
            "passed": all(findings.values()), "findings": findings,
            "comparison": out["mean_aggregate_gain"]}


def an_unknown_cell_still_withholds_the_verdict() -> dict[str, Any]:
    candidate = [_cell(0, 0.5), _cell(1, 0.5)]
    parent = [_cell(0, 0.1), _cell(1, 0.1), _cell(2, 0.1)]
    out = D2.compare_to_the_parent(candidate, parent, index=_index(3),
                                   by_position=_by_position(3),
                                   candidate_label="c")
    return {
        "check": "an_unknown_cell_still_withholds_the_verdict",
        "passed": (out["units_declared"] == 3
                   and out["units_scored_on_both_sides"] == 2
                   and out["verdict_is_complete"] is False
                   and out["unknown_cells"]["candidate"] == ["pos2"]),
        "unknown": out["unknown_cells"],
        "counted_over": out["mean_aggregate_gain"]["counted_over"],
    }


def frame_c_refuses_legality_and_only_that() -> dict[str, Any]:
    """Legality and calibration still gate; the replay verdict does not."""
    shared = set(D2.SHARED_REFUSALS)
    demoted = set(D2.DEMOTED_TO_EVIDENCE)
    # Read the outcome names off the code rather than restating them, so a new
    # outcome added upstream fails this check instead of quietly queueing.
    import re
    every = set(re.findall(r'"outcome": "([A-Z_]+)"',
                           inspect.getsource(rev.evaluate_proposal)))
    every.add("ACCEPTED_FOR_VERIFICATION")
    # ``evaluate_proposal`` forwards the calibration tool's own outcome
    # verbatim when it is not CALIBRATED, so those names are outcomes too.
    forwarded = set(re.findall(r'"outcome": "([A-Z_]+)"',
                               inspect.getsource(tool.clause_from_slow)))
    forwarded |= {"NO_FEASIBLE_THRESHOLD"}   # scope_threshold_tool line 97
    forwarded.discard("CALIBRATED")
    every |= forwarded
    source = inspect.getsource(D2.evaluate_frame_c)
    findings = {
        "the_two_sets_do_not_overlap": not (shared & demoted),
        "together_they_cover_every_outcome_the_evaluator_returns":
            every <= (shared | demoted),
        "and_every_forwarded_calibration_outcome_is_a_refusal":
            forwarded <= shared,
        "an_unlisted_outcome_is_not_silently_queued": (
            "if outcome not in DEMOTED_TO_EVIDENCE" in source),
        "the_absolute_verdict_is_still_recorded":
            "the_absolute_screen_would_have_said" in source,
        "calibration_is_still_the_existing_tool":
            "tool.clause_from_slow" in inspect.getsource(rev.evaluate_proposal),
    }
    return {"check": "frame_c_refuses_legality_and_only_that",
            "passed": all(findings.values()), "findings": findings,
            "shared_refusals": sorted(shared),
            "demoted_to_evidence": sorted(demoted),
            "outcomes_the_evaluator_can_return": sorted(every)}


def the_declared_order_is_the_order_used() -> dict[str, Any]:
    """Ordering uses only arrived evidence, and the step sorts by it."""
    def verdict(diff, raw):
        return {"common_denominator_comparison": {
            "mean_aggregate_gain": {"difference": diff},
            "candidate_cell_states": {rev.READ: 5 - raw,
                                      rev.RAW_FALLBACK: raw, rev.UNKNOWN: 0}}}

    rows = [verdict(0.01, 0), verdict(0.20, 4), verdict(0.20, 0),
            verdict(None, 0)]
    order = [rows.index(r) for r in sorted(rows, key=D2.ordering_key)]
    source = inspect.getsource(D2.revision_step)
    key_source = inspect.getsource(D2.ordering_key)
    findings = {
        "higher_utility_first_then_fewer_unusable_cells": order == [2, 1, 0, 3],
        "the_step_actually_sorts_by_it": "queued.sort(key=ordering_key)" in source,
        "and_opens_drafts_in_that_order": (
            source.index("queued.sort(key=ordering_key)")
            < source.index("open_restricted(")),
        "the_key_reads_no_future": not any(
            name in key_source for name in
            ("delayed", "later", "future", "evaluation", "face_origin")),
        "the_rule_is_published_to_slow":
            "queue_order" in inspect.getsource(D2.build_feedback_frame_c),
    }
    return {"check": "the_declared_order_is_the_order_used",
            "passed": all(findings.values()), "findings": findings,
            "sorted_indices": order}


def the_runner_supplies_but_does_not_select() -> dict[str, Any]:
    """A queued candidate is offered through the production resupply path."""
    ledger = drafts.DraftLedger()
    steps = (("outlier_iqr", {}),)
    root = {"scope_type": "serving_series_predicate", "predicate": []}
    draft = ledger.open_restricted(program_steps=steps, root_scope=root,
                                   current_scope=root, origin=1176,
                                   census_key="k1",
                                   provenance={"opened_by": "smoke"})
    offered = ledger.resupplied_programs_for_verification()
    reopened = None
    try:
        ledger.open_restricted(program_steps=steps, root_scope=root,
                               current_scope=root, origin=1200,
                               census_key="k1", provenance={})
    except ValueError as exc:
        reopened = str(exc)[:80]

    step_source = inspect.getsource(D2.revision_step)
    cell_source = inspect.getsource(R.run_cell)
    build_source = inspect.getsource(D2.build)
    findings = {
        "a_queued_candidate_gets_a_rights_less_draft": (
            draft.state is None and draft.verification_attempts == 0),
        "and_is_offered_to_later_units": offered == {draft.draft_id: steps},
        "the_cell_passes_the_resupply_to_fast":
            "resupplied_programs=resupplied" in cell_source,
        "the_runner_never_forces_a_winner": not any(
            name in build_source for name in
            ("force_winner", "_winner_steps =", "deploy(")),
        "the_same_lineage_cannot_be_reopened": reopened is not None,
        "promotion_needs_fast_to_have_deployed_it": (
            'label_now in pending' in build_source
            and 'delayed_gate") or {}).get("passes")' in build_source),
        "the_step_opens_drafts_through_the_ledger":
            "arm.draft_ledger.open_restricted(" in step_source,
        "verification_counting_is_the_production_call":
            "record_verification(" in cell_source,
    }
    return {"check": "the_runner_supplies_but_does_not_select",
            "passed": all(findings.values()), "findings": findings,
            "refusal_when_reopened": reopened,
            "lifecycle_bounds": {"max_revisions": drafts.MAX_REVISIONS,
                                 "max_verification_attempts":
                                     drafts.MAX_VERIFICATION_ATTEMPTS}}


def the_repaired_wiring_is_still_in_place() -> dict[str, Any]:
    source = inspect.getsource(D2.build)
    findings = {
        "both_arms_carry_their_state": source.count(
            'runner.ArmSpec(FROZEN_ARM, "k0", True, False, True)') == 1,
        "no_per_cell_reseed": "seed=None" in source and "_seed_arm(a," not in source,
        "seeded_once_each": source.count(
            "R._seed_arm(arm, state, fresh_ledger=True)") == 1,
        "the_released_admission_rule_is_installed":
            'machinery["admission_policy"].install_policy(' in source
            and "bounded.BOUNDED_POLICY" in source,
        "the_fast_budget_is_the_package_allowance":
            "per_unit_arm_cap=R.PER_ARM_LLM" in source
            and "_default_backend_factory(R.PER_ARM_LLM)" in source,
        "the_hec1_contract_is_untouched":
            int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]) == 5,
        "only_the_revising_arm_gets_the_channel":
            source.count("revision_step(") == 1
            and "arms[REVISING_ARM], position=row" in source,
    }
    return {"check": "the_repaired_wiring_is_still_in_place",
            "passed": all(findings.values()), "findings": findings,
            "per_arm_llm": R.PER_ARM_LLM,
            "admission": bounded.BOUNDED_POLICY.to_dict()}


def the_arm_can_activate_the_snapshot_it_starts_on() -> dict[str, Any]:
    """``activate_approved`` must not die on the ordinary confirmed deployment.

    An arm's store is created empty and its start snapshot is compiled from a
    directory outside it, so ``set_active`` refuses the very bundle the arm is
    running until the store has been given it.
    """
    source = inspect.getsource(D2.build)
    findings = {
        "the_start_snapshot_is_materialized_into_the_arms_store":
            "arm._store.materialize(arm.active_snapshot())" in source,
        "before_the_course_loop": (
            source.index("arm._store.materialize(") < source.index("for row in after")),
        "and_the_build_tag_is_the_carrying_arms":
            'arm._build("online")' in source,
        "activation_still_needs_an_approved_delayed_event":
            'get("stage") != "approved"' in inspect.getsource(
                D2.v1runner._machinery()["online_loop"].activate_approved),
    }
    return {"check": "the_arm_can_activate_the_snapshot_it_starts_on",
            "passed": all(findings.values()), "findings": findings,
            "why": ("materializing grants nothing: the snapshot made "
                    "activatable is the one the arm already runs")}


def the_write_back_manifest_matches_the_contract() -> dict[str, Any]:
    """The promotion PATCH must be constructible before a course reaches one.

    DEV-AUTO-1 never promoted anything, so this manifest was never built and
    two shape errors sat in it: a ``CONTENT_SHA``/``sha256`` precondition the
    contract refuses for a PATCH, and a bare-string ``minimal_patch`` that
    ``EditController._apply`` cannot read.  Constructed here from the real
    contract class, with a real digest shape, so it cannot rot again.
    """
    from SelfEvolvingHarnessTS.contracts.harness import (EditManifest,
                                                         EditOperation)
    digest = "a" * 64
    source = inspect.getsource(rev.apply_workflow_revision)
    manifest = EditManifest(
        edit_id="smoke", base_harness_sha=digest,
        target_pattern_id="dev-auto1-revision",
        target_surface_id="skill_library.entries/%s.body" % R.K0_SKILL_ID,
        operation=EditOperation.PATCH,
        surface_precondition={"kind": "SHA", "sha": digest},
        dependency_precondition_shas={},
        minimal_patch={"value": "Frozen program steps: []"},
        patch_id="dev-auto1-workflow-revision",
        predicted_agent_behavior_change=("retrieve_skill:x",),
        predicted_data_effect=("local_improvement",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",))
    findings = {
        "the_contract_accepts_this_shape": manifest.operation
            is EditOperation.PATCH,
        "the_writer_uses_the_sha_precondition":
            '"kind": "SHA", "sha": precondition' in source,
        "and_the_value_wrapped_patch":
            'minimal_patch={"value": body}' in source,
        "the_apply_reads_that_key": 'minimal_patch or {})["value"]' in
            inspect.getsource(D2.v1runner._machinery()["EditController"]._apply),
        "a_failed_promotion_is_recorded_not_fatal":
            source.index("manifest = EditManifest(")
            > source.index("try:"),
        "the_route_is_still_the_authorised_one":
            rev.WORKFLOW_CAUSE == "OUTCOME_GAP",
    }
    return {"check": "the_write_back_manifest_matches_the_contract",
            "passed": all(findings.values()), "findings": findings}


def nothing_in_the_instrument_or_the_boundary_moved() -> dict[str, Any]:
    risk = dict(contract.RISK)
    module = inspect.getsource(D2)
    code = "\n".join(line for line in module.splitlines()
                     if not line.lstrip().startswith(("#", '"', "'"))
                     and "no sealed data" not in line
                     and "TARGET_HELD_IN" not in line)
    findings = {
        "risk_lines": (risk["material"] == 0.005
                       and risk["max_harmed_fraction"] == 0.20
                       and risk["max_single_series_harm"] == 0.30
                       and risk["min_treated"] == 5),
        "the_scope_grid": tool.frozen_bins() == {
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
        "the_evaluation_face_is_unreachable": not any(
            name in code for name in ("EVALUATION_OFFSET", "evaluation_origin",
                                      "_evaluate_face", "+ 144", "+144")),
    }
    return {"check": "nothing_in_the_instrument_or_the_boundary_moved",
            "passed": all(findings.values()), "findings": findings,
            "risk": risk}


def no_known_winner_reaches_the_model() -> dict[str, Any]:
    """DEV-AUTO-1R's result must not be handed back to the proposer."""
    seen = "\n".join([
        rev.REVISION_ROLE_INSTRUCTION,
        D2.FRAME_C_RULE, D2.CANDIDATE_ORDER,
        inspect.getsource(D2.build_feedback_frame_c),
        inspect.getsource(rev.build_feedback),
    ])
    winners = ["outlier_iqr({})>winsorize({})", "outlier_iqr>winsorize",
               "winsorize", "hampel_filter", "outlier_mad>winsorize",
               "DEV-AUTO-1R", "0.174513", "0.170298"]
    found = [name for name in winners if name in seen]
    table = rev.legal_operator_table()
    return {
        "check": "no_known_winner_reaches_the_model",
        "passed": not found and len(table) > 20,
        "leaked": found,
        "the_operator_table_is_the_whole_registry": len(table),
        "note": ("the operator table names every legal operator, which is not "
                 "a hint; what must not appear is a program, a score or a "
                 "finding from the earlier packages"),
    }


CHECKS: tuple[Callable[[], dict[str, Any]], ...] = (
    the_comparator_is_in_the_live_step,
    an_unknown_cell_still_withholds_the_verdict,
    frame_c_refuses_legality_and_only_that,
    the_declared_order_is_the_order_used,
    the_runner_supplies_but_does_not_select,
    the_repaired_wiring_is_still_in_place,
    the_arm_can_activate_the_snapshot_it_starts_on,
    the_write_back_manifest_matches_the_contract,
    nothing_in_the_instrument_or_the_boundary_moved,
    no_known_winner_reaches_the_model,
)


def run() -> dict[str, Any]:
    rows = []
    for check in CHECKS:
        try:
            rows.append(check())
        except Exception as exc:  # noqa: BLE001 - a fault is a failed check
            rows.append({"check": check.__name__, "passed": False,
                         "fault": "%s: %s" % (type(exc).__name__,
                                              str(exc)[:400])})
    return {"stage": "DEV_AUTO2_SMOKE",
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
                ensure_ascii=False, default=str)[:700])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
