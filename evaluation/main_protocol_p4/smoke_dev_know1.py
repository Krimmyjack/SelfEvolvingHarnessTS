"""DEV-KNOW-1 preflight: the retention behaviours, then the fork's symmetry.

Two kinds of check, kept apart on purpose.

*   ``dev_know1_retention`` runs the two behaviours the work order asks to be
    checked -- a child card coexists with its parent, and revoking the child
    leaves the parent -- against the real store and compiler.  They are
    executed, not asserted from source.
*   Everything else here is about the contrast itself: that the two test arms
    differ in one readable thing and in nothing else, that neither carries a
    candidate pool or the formation segment's Episodes, and that no card is
    patched anywhere in this package.

Zero LLM calls, zero Consumer fits.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from evaluation.main_protocol_p4 import dev_know1_retention as keep
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_dev_know1_two_arms as K1
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import smoke_dev_auto2_frame_c as smoke2


def the_two_retention_behaviours_hold() -> dict[str, Any]:
    """The work order's two checks, run against the real machinery."""
    result = keep.run()
    return {"check": "the_two_retention_behaviours_hold",
            "passed": bool(result["passed"]),
            "checks": {row["check"]: row["passed"]
                       for row in result["checks"]},
            "detail": [row for row in result["checks"] if not row["passed"]]
                      or "all pass"}


def the_segments_are_registered_and_do_not_overlap() -> dict[str, Any]:
    """The split is the course's, and it is fixed before anything is scored."""
    from evaluation.main_protocol_p4 import audit_m_r0_reachability as base

    doc = base.load(R.ORDERING)
    population, excluded = R._population(doc)
    after = [row["position"] for row in population
             if row["side"] == "after_the_boundary"]
    held = sorted(row["position"] for row in excluded
                  if row["why"].startswith("held for the Target"))
    formation = list(K1.FORMATION_POSITIONS)
    test = list(K1.TEST_POSITIONS)
    findings = {
        "the_formation_segment_is_u5_to_u15": formation == list(range(5, 16)),
        "the_test_segment_is_the_five_named_units":
            test == [16, 17, 23, 24, 25],
        "they_do_not_overlap": not set(formation) & set(test),
        "together_they_are_exactly_the_units_after_the_boundary":
            sorted(formation + test) == sorted(after),
        "u18_to_u22_are_excluded_by_exposure_name": held == [18, 19, 20, 21, 22],
        "no_test_unit_is_held_for_the_target": not set(test) & set(held),
    }
    return {"check": "the_segments_are_registered_and_do_not_overlap",
            "passed": all(findings.values()), "findings": findings,
            "formation": formation, "test": test,
            "units_after_the_boundary": after,
            "excluded_by_exposure": held}


def the_control_arm_is_the_boundary_minus_what_was_formed() -> dict[str, Any]:
    """A is built by removal from the boundary, never from the entry state."""
    source = inspect.getsource(K1._fork)
    findings = {
        "the_accumulated_arm_starts_at_the_boundary":
            "start = end_snapshot" in source,
        "the_control_arm_starts_at_the_boundary_minus_the_formed_cards":
            "keep.remove_cards(" in source and "skill_ids=formed" in source,
        "formed_is_the_difference_against_the_entry_library":
            "set(end_cards) - set(entry_cards)" in source,
        "a_card_lost_during_formation_is_recorded":
            "lost = sorted(set(entry_cards) - set(end_cards))" in source
            and "lost_during_the_formation_segment" in source,
        "and_it_is_not_revived_for_the_control_arm": (
            # both arms' snapshots derive from the boundary, and the entry
            # snapshot is not even reachable inside the fork: it takes the set
            # of entry card *ids*, never the entry snapshot object
            'start = removal["snapshot"]' in source
            and "snapshot=end_snapshot" in source
            and "entry_cards" in inspect.signature(K1._fork).parameters
            and not any("snapshot" in name for name
                        in inspect.signature(K1._fork).parameters)),
        "the_removal_is_the_revocation_round_trip": all(
            fragment in inspect.getsource(keep.remove_cards)
            for fragment in ("store.fork(", "path.unlink()",
                             "compile_snapshot(", "store.materialize(")),
        "and_it_is_the_same_four_steps_the_revocation_performs": all(
            fragment in inspect.getsource(revocation())
            for fragment in ("store.fork(", "target.unlink()",
                             "compile_snapshot(", "store.materialize(")),
        "the_removal_does_not_move_the_active_pointer":
            "set_active" not in inspect.getsource(keep.remove_cards),
    }
    return {"check": "the_control_arm_is_the_boundary_minus_what_was_formed",
            "passed": all(findings.values()), "findings": findings}


def revocation() -> Any:
    from SelfEvolvingHarnessTS.methods.ttha import online_loop  # noqa: PLC0415
    return online_loop.revoke_deployed_skill


def neither_arm_carries_a_candidate_pool_or_the_episodes() -> dict[str, Any]:
    """The only channel from formation to test is a Skill card."""
    source = inspect.getsource(K1._fork)
    build_source = inspect.getsource(runner.Arm._build)
    findings = {
        "every_open_draft_is_closed_at_the_fork":
            "formation.draft_ledger.close(draft, keep.FORK_CLOSURE)" in source,
        "it_is_closed_once_in_the_ledger_both_arms_copy_from":
            source.index("close(draft, keep.FORK_CLOSURE)")
            < source.index("copy.deepcopy(formation.draft_ledger)"),
        "closing_removes_it_from_the_verification_resupply":
            "may_verify" in inspect.getsource(
                drafts.DraftLedger.verifiable_drafts)
            and not drafts.RestrictedDraft(
                draft_id="x", program_steps=(), root_scope={},
                current_scope={}, revisions=0, created_at_origin=0,
                closed=drafts.CLOSE_REASONS[drafts.WAITING]).may_verify(),
        "a_closed_draft_keeps_its_counters":
            "draft.closed = str(reason)" in inspect.getsource(
                drafts.DraftLedger.close),
        "the_method_is_rebuilt_with_an_empty_experience_list":
            'self.m["TTHAMethod"](' in build_source
            and "self.start_snapshot, ())" in build_source,
        "the_arms_are_built_through_that_same_path":
            'arm._build("online")' in source,
        "no_hidden_field_hands_a_formation_program_across":
            "resupplied" not in source,
    }
    return {"check": "neither_arm_carries_a_candidate_pool_or_the_episodes",
            "passed": all(findings.values()), "findings": findings}


def the_runtime_constraints_are_carried_by_both_arms() -> dict[str, Any]:
    """Records are carried, not deleted; a permission is never restored."""
    source = inspect.getsource(K1._fork)
    findings = {
        "the_ledger_is_deep_copied_to_each_arm":
            source.count("copy.deepcopy(formation.draft_ledger)") == 1
            and "for name in TEST_ARMS" in source,
        "the_bank_and_the_processed_units_are_carried":
            "arm.bank = [dict(row) for row in formation.bank]" in source
            and "arm.processed = list(formation.processed)" in source,
        "no_record_is_deleted_anywhere_in_the_fork":
            "del " not in source and ".remove(" not in source,
        "only_the_control_arm_drops_the_formed_cards_ids":
            "dropped = set(formed) if name == ARM_OLD else set()" in source,
        "a_closed_lineage_cannot_be_reopened_by_either_arm":
            "already has a Draft" in inspect.getsource(
                drafts.DraftLedger.open_restricted),
    }
    return {"check": "the_runtime_constraints_are_carried_by_both_arms",
            "passed": all(findings.values()), "findings": findings}


def the_control_arm_is_not_an_identity_control() -> dict[str, Any]:
    """A keeps search, verification, minting and the whole outer loop."""
    fork_source = inspect.getsource(K1._fork)
    segment_source = inspect.getsource(K1._run_segment)
    findings = {
        "both_arms_have_the_same_spec_apart_from_the_name":
            'runner.ArmSpec(name, "k0", True, False, True)' in fork_source
            and fork_source.count("runner.ArmSpec(") == 1,
        "the_outer_step_runs_for_every_arm_in_the_order":
            "for name in order:" in segment_source
            and "revision_step(" in segment_source,
        "no_arm_is_excluded_from_the_outer_step":
            "SUPPLY_ARMS" not in segment_source,
        "both_arms_write_back_so_both_can_mint": (
            'runner.ArmSpec(name, "k0", True, False, True)' in fork_source),
        "the_two_test_arms_are_the_declared_pair":
            K1.TEST_ARMS == (K1.ARM_OLD, K1.ARM_ACC),
        "each_group_runs_them_in_a_different_order":
            len({tuple(plan["test_arm_order"])
                 for plan in K1.GROUPS.values()}) == len(K1.GROUPS),
    }
    return {"check": "the_control_arm_is_not_an_identity_control",
            "passed": all(findings.values()), "findings": findings,
            "test_arm_orders": {g: list(p["test_arm_order"])
                                for g, p in K1.GROUPS.items()}}


def no_card_is_patched_anywhere_in_this_package() -> dict[str, Any]:
    """Retention here is additive, and the source says so with no exception."""
    module = inspect.getsource(K1)
    findings = {
        "the_workflow_patch_writer_is_never_called":
            "apply_workflow_revision" not in module,
        "the_verification_event_has_no_write_back_branch":
            "writes_back" not in inspect.getsource(K1.verification_event),
        "the_event_records_the_mint_instead":
            "minted_a_card_of_its_own" in inspect.getsource(
                K1.verification_event),
        "the_mint_is_the_ordinary_lifecycle":
            "activate_approved" in inspect.getsource(R.run_cell),
        "the_boundary_block_declares_zero_patches":
            '"cards_patched": 0' in module,
    }
    return {"check": "no_card_is_patched_anywhere_in_this_package",
            "passed": all(findings.values()), "findings": findings}


def a_missing_card_is_reported_not_substituted() -> dict[str, Any]:
    """The ancestor fallback exists upstream and is never reached from here."""
    wrapper = inspect.getsource(K1.revision_step)
    upstream = inspect.getsource(R.current_skill_steps)
    findings = {
        "the_upstream_fallback_really_does_substitute_the_ancestor":
            "return tuple(ANCESTOR_PROGRAM)" in upstream,
        "this_package_reads_the_card_first":
            "keep.current_card(" in wrapper,
        "and_stops_when_there_is_none":
            '"NO_CURRENT_CARD"' in wrapper,
        "the_refusal_comes_before_the_delegation":
            wrapper.index("NO_CURRENT_CARD") < wrapper.index(
                "D2.revision_step("),
        "the_absent_report_names_what_is_actually_there":
            "capability_cards_now" in wrapper,
        "and_the_reading_itself_offers_no_ancestor": _absent_reads_absent(),
    }
    return {"check": "a_missing_card_is_reported_not_substituted",
            "passed": all(findings.values()), "findings": findings}


def the_transport_is_the_one_the_previous_package_used() -> dict[str, Any]:
    """A silent model switch would make the comparison a different experiment.

    The host and model come from ``M0_AGENT_*``; when those are unset the
    module default takes over, which is a *different* relay and a different
    model.  The first live attempt of this package hit exactly that: the
    fallback relay answered ``insufficient_quota`` under a model DEV-AUTO-2 and
    DEV-AUTO-3 never used.  The expected target is read from DEV-AUTO-3's own
    receipt rather than written here, so nothing hardcodes a host.
    """
    from evaluation.main_protocol_p4 import run_source_line as v1runner

    import os  # noqa: PLC0415

    receipt = (base_artifact("dev_auto3_three_arms__g1.json") or {}).get(
        "transport") or {}
    live = v1runner._machinery()["agentic"].live_transport()
    findings = {
        "a_previous_receipt_names_a_transport": bool(receipt.get("model")),
        "the_base_url_matches": receipt.get("base_url") == live.get("base_url"),
        "the_model_matches": receipt.get("model") == live.get("model"),
        "it_comes_from_the_environment_not_a_module_default":
            live.get("source") == "M0_AGENT_*",
    }
    # A deliberate transport change is allowed; a silent one is not.  Setting
    # DEV_KNOW1_TRANSPORT_NOTE to a reason acknowledges the divergence, and the
    # reason is carried into the run's artifact, so a reader can never mistake
    # a differently-modelled run for a comparable one.
    note = (os.environ.get("DEV_KNOW1_TRANSPORT_NOTE") or "").strip()
    matched = all(findings.values())
    return {"check": "the_transport_is_the_one_the_previous_package_used",
            "passed": matched or bool(note),
            "matches_the_previous_package": matched,
            "findings": findings,
            "expected": receipt, "live": live,
            "acknowledged_divergence": note or None,
            "what_the_note_means": (
                None if matched else
                "this run is NOT comparable to the previous package's numbers; "
                "it is a different model and must be reported as one"),
            "how_to_fix": ("set M0_AGENT_BASE_URL and M0_AGENT_MODEL to the "
                           "values in the previous package's receipt, and put "
                           "that provider's key in CPA_API_KEY; or set "
                           "DEV_KNOW1_TRANSPORT_NOTE to say why the change is "
                           "deliberate")}


def base_artifact(name: str) -> dict[str, Any] | None:
    from evaluation.main_protocol_p4 import audit_m_r0_reachability as base

    path = base.ROOT / "artifacts" / "main_protocol" / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class _EmptyLibrary:
    skills = ()


def _absent_reads_absent() -> bool:
    """Asked about a card that is not there, the reading says so and stops."""
    report = keep.current_card(_EmptyLibrary(), K1.K0_SKILL_ID)
    return (report["present"] is False and report["steps"] is None
            and report["program"] is None
            and report["capability_cards_present"] == []
            and "ancestor" in str(report.get("why") or "").lower())


def the_allowances_are_equal_and_the_ceilings_hold() -> dict[str, Any]:
    """One package budget, six equal arm-runs, monotone phase ceilings."""
    phases = len(K1.GROUPS) * 2
    probe = K1.Budget()
    probe.spend_fits("x", 120)
    probe.open_phase("formation", K1.FORMATION_FIT_ALLOWANCE)
    moved_with_the_spend = probe.soft_fit_ceiling == 120 +         K1.FORMATION_FIT_ALLOWANCE
    probe.spend_fits("x", K1.MAX_FITS)
    probe.open_phase("test", K1.TEST_FIT_ALLOWANCE)
    findings = {
        "six_arm_runs_share_the_model_allowance":
            K1.PER_ARM_LLM == K1.MAX_LLM // 6,
        "the_package_ceilings_are_the_authorised_ones":
            (K1.MAX_FITS, K1.MAX_LLM, K1.MAX_WALL_SECONDS)
            == (2000, 1000, 6 * 3600),
        "every_phase_gets_the_same_room":
            K1.FORMATION_FIT_ALLOWANCE == K1.TEST_FIT_ALLOWANCE,
        "the_phase_allowances_add_up_to_the_package_ceiling":
            phases * K1.FORMATION_FIT_ALLOWANCE == K1.MAX_FITS,
        "a_phase_ceiling_is_its_allowance_on_top_of_what_is_spent":
            moved_with_the_spend,
        "and_it_never_passes_the_package_ceiling":
            probe.soft_fit_ceiling == K1.MAX_FITS,
        "no_group_has_a_different_plan_except_the_arm_order":
            all(set(plan) == {"seed", "test_arm_order"}
                for plan in K1.GROUPS.values()),
        "no_per_cell_call_cap_is_reinstated":
            "llm_cap_is_the_package_allowance" in inspect.getsource(R.run_cell),
        "the_budget_keys_separate_the_groups":
            '"%s/%s" % (self.group, arm)' in inspect.getsource(
                K1.Budget._key),
    }
    return {"check": "the_allowances_are_equal_and_the_ceilings_hold",
            "passed": all(findings.values()), "findings": findings,
            "phase_fit_allowance": K1.FORMATION_FIT_ALLOWANCE,
            "phases": phases, "per_arm_llm": K1.PER_ARM_LLM}


def the_inherited_readings_cannot_leak_a_discovery() -> dict[str, Any]:
    """Both arms inherit the same prefix cache; neither feeds the other."""
    source = inspect.getsource(K1._copy_cache)
    fork_source = inspect.getsource(K1._fork)
    findings = {
        "the_copy_is_from_the_formation_arm_only":
            "_copy_cache(formation.replay_cache, arm.replay_cache)"
            in fork_source,
        "it_is_done_for_both_arms_identically":
            fork_source.count("_copy_cache(") == 1
            and "for name in TEST_ARMS" in fork_source,
        "the_key_is_rewritten_to_the_reading_arm":
            "(target.arm,) + tuple(key)[1:]" in source,
        "each_arm_still_owns_its_own_cache_object":
            "self.replay_cache = ReplayPredictionCache(spec.name)"
            in inspect.getsource(runner.Arm.__init__),
        "the_counters_are_not_copied": "physical_fits" not in source,
    }
    return {"check": "the_inherited_readings_cannot_leak_a_discovery",
            "passed": all(findings.values()), "findings": findings}


CHECKS: tuple[Callable[[], dict[str, Any]], ...] = (
    the_two_retention_behaviours_hold,
    the_segments_are_registered_and_do_not_overlap,
    the_control_arm_is_the_boundary_minus_what_was_formed,
    neither_arm_carries_a_candidate_pool_or_the_episodes,
    the_runtime_constraints_are_carried_by_both_arms,
    the_control_arm_is_not_an_identity_control,
    no_card_is_patched_anywhere_in_this_package,
    a_missing_card_is_reported_not_substituted,
    the_allowances_are_equal_and_the_ceilings_hold,
    the_inherited_readings_cannot_leak_a_discovery,
    the_transport_is_the_one_the_previous_package_used,
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
    inherited = smoke2.run()
    rows.append({"check": "the_dev_auto2_frame_c_preflight_still_passes",
                 "passed": bool(inherited["passed"]),
                 "checks": [r["check"] for r in inherited["checks"]
                            if not r["passed"]] or "all pass"})
    return {"stage": "DEV_KNOW1_SMOKE",
            "passed": all(row["passed"] for row in rows),
            "llm_calls": 0, "consumer_fits": 0, "checks": rows}


def main(argv=None) -> int:
    result = run()
    print("smoke: %s | llm %d | fits %d"
          % ("PASS" if result["passed"] else "FAIL",
             result["llm_calls"], result["consumer_fits"]))
    for row in result["checks"]:
        print("   %s %s" % ("ok  " if row["passed"] else "FAIL", row["check"]))
        if not row["passed"]:
            print("       %s" % json.dumps(
                {k: v for k, v in row.items() if k not in ("check", "passed")},
                ensure_ascii=False, default=str)[:900])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
