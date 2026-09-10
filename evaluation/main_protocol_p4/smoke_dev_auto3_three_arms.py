"""DEV-AUTO-3 preflight: only the newly added paths.

Three things are new in this package and nothing else is re-tested here -- the
DEV-AUTO-2 preflight already covers the Frame C wiring and is re-used whole:

* Draft ids after a ledger is restored;
* arm B supplies and verifies but never PATCHes the target card;
* arm C PATCHes.

Zero LLM calls, zero Consumer fits.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto2_frame_c as D2
from evaluation.main_protocol_p4 import run_dev_auto3_three_arms as D3
from evaluation.main_protocol_p4 import smoke_dev_auto2_frame_c as smoke2

ROOT_SCOPE = {"scope_type": "serving_series_predicate", "predicate": []}


def _restored_ledger() -> drafts.DraftLedger:
    """A ledger in the shape the entry-state reconstruction leaves it.

    Drafts appended directly, ``_minted`` never advanced -- which is exactly
    what ``live._state_at_k1`` produces (measured: one Draft, ``_minted == 0``).
    """
    ledger = drafts.DraftLedger()
    draft = drafts.RestrictedDraft(
        draft_id="resupplied_draft_1",
        program_steps=(("outlier_mad", {}),),
        root_scope=dict(ROOT_SCOPE), current_scope=dict(ROOT_SCOPE),
        revisions=1, created_at_origin=1176)
    draft.verification_attempts = 2
    draft.state = "REVISABLE"
    ledger.drafts.append(draft)
    return ledger


def a_restored_ledger_does_not_mint_a_used_id() -> dict[str, Any]:
    ledger = _restored_ledger()
    before = {d.draft_id: (d.state, d.verification_attempts, d.revisions,
                           d.closed) for d in ledger.drafts}
    minted_counter_before = ledger._minted
    new = ledger.open_restricted(
        program_steps=(("winsorize", {}),), root_scope=ROOT_SCOPE,
        current_scope=ROOT_SCOPE, origin=1200, census_key="k-new",
        provenance={"opened_by": "smoke"})
    second = ledger.open_restricted(
        program_steps=(("hampel_filter", {}),), root_scope=ROOT_SCOPE,
        current_scope=ROOT_SCOPE, origin=1224, census_key="k-new-2",
        provenance={"opened_by": "smoke"})
    after = {d.draft_id: (d.state, d.verification_attempts, d.revisions,
                          d.closed) for d in ledger.drafts}
    ids = [d.draft_id for d in ledger.drafts]
    findings = {
        "the_restored_id_was_not_reissued": new.draft_id != "resupplied_draft_1",
        "and_neither_was_it_the_second_time":
            second.draft_id not in ("resupplied_draft_1", new.draft_id),
        "every_id_is_unique": len(ids) == len(set(ids)),
        "the_restored_draft_keeps_its_identity_and_counters":
            before["resupplied_draft_1"] == after["resupplied_draft_1"],
        "no_existing_draft_was_renumbered":
            set(before) <= set(after),
        "by_id_is_unambiguous": all(
            sum(1 for d in ledger.drafts if d.draft_id == i) == 1 for i in ids),
        "the_counter_started_behind": minted_counter_before == 0,
    }
    return {"check": "a_restored_ledger_does_not_mint_a_used_id",
            "passed": all(findings.values()), "findings": findings,
            "ids": ids}


def the_live_entry_state_is_the_shape_this_fixes() -> dict[str, Any]:
    """The real reconstruction, not a mock: it really does arrive like that."""
    from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
    from evaluation.main_protocol_p4 import (
        run_m_r0d_forward_k1_outer_step as live)
    from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
    ledger = live._state_at_k1(base.load(R.ORDERING))["ledger"]
    ids = [d.draft_id for d in ledger.drafts]
    return {
        "check": "the_live_entry_state_is_the_shape_this_fixes",
        "passed": bool(ids) and ledger._minted < len(ids),
        "entry_state_draft_ids": ids,
        "minted_counter": ledger._minted,
        "why": ("the counter is behind the ids already present, which is the "
                "condition the mint now skips past"),
    }


def b_supplies_and_verifies_but_never_patches() -> dict[str, Any]:
    source = inspect.getsource(D3.build)
    event_source = inspect.getsource(D3._verification_event)
    findings = {
        "b_is_a_supply_arm": D3.ARM_B in D3.SUPPLY_ARMS,
        "b_is_not_an_updating_arm": D3.ARM_B not in D3.UPDATING_ARMS,
        "the_revision_step_runs_for_both_supply_arms":
            "if name not in SUPPLY_ARMS or budget.stop_reason()" in source,
        "the_write_back_is_gated_on_the_updating_arms":
            "writes_back=name in UPDATING_ARMS" in source,
        "and_only_that_branch_patches":
            event_source.index("if writes_back:")
            < event_source.index("rev.apply_workflow_revision("),
        "b_still_records_the_verification_event":
            "VERIFIED_BUT_NOT_WRITTEN_BACK" in event_source,
        "b_still_closes_the_draft_at_the_same_moment":
            event_source.count("arm.draft_ledger.close(draft, closure)") == 1,
        "control_gets_no_revision_step": D3.ARM_A not in D3.SUPPLY_ARMS,
        "control_is_not_otherwise_frozen": (
            source.count('runner.ArmSpec(name, "k0", True, False, True)') == 1
            and "seed=None" in source),
    }
    return {"check": "b_supplies_and_verifies_but_never_patches",
            "passed": all(findings.values()), "findings": findings,
            "supply_arms": list(D3.SUPPLY_ARMS),
            "updating_arms": list(D3.UPDATING_ARMS)}


def c_patches_through_the_authorised_route() -> dict[str, Any]:
    writer = inspect.getsource(rev.apply_workflow_revision)
    event_source = inspect.getsource(D3._verification_event)
    findings = {
        "c_is_an_updating_arm": D3.ARM_C in D3.UPDATING_ARMS,
        "the_route_is_outcome_gap": rev.WORKFLOW_CAUSE == "OUTCOME_GAP",
        "the_surface_is_the_k0_body":
            'surface_id = "skill_library.entries/%s.body"' in writer,
        "it_goes_through_apply_to_fork": "controller.apply_to_fork(" in writer,
        "and_reads_the_compiled_card_back":
            "verify_frozen_patch_program(" in writer,
        "the_snapshot_and_store_follow_the_patch": (
            "arm._method._snapshot = applied" in event_source
            and "arm._store.set_active(" in event_source),
    }
    return {"check": "c_patches_through_the_authorised_route",
            "passed": all(findings.values()), "findings": findings}


def the_two_supply_arms_share_no_rule_and_no_state() -> dict[str, Any]:
    source = inspect.getsource(D3.build)
    findings = {
        "one_proposer_one_frame_one_order": (
            "D2.revision_step(" in source
            and source.count("D2.revision_step(") == 1),
        "history_is_per_arm": "history_by_arm[name]" in source,
        "attempts_are_per_arm": "attempts_by_arm[name]" in source,
        "pending_is_per_arm": "pending_by_arm[name]" in source,
        "events_are_per_arm": "events_by_arm[name]" in source,
        "each_arm_has_its_own_ledger_and_cache": (
            source.count("R._seed_arm(arm, state, fresh_ledger=True)") == 1
            and "for name in (ARM_A, ARM_B, ARM_C)" in source),
        "the_case_id_separates_the_sessions": "case_prefix=" in source,
        "the_allowance_is_the_same_for_every_arm":
            D3.PER_ARM_LLM == D3.MAX_LLM // 3,
        "no_call_is_made_to_level_the_cost":
            "no call is made to level the arms" in inspect.getsource(
                D3.Budget.to_dict),
    }
    return {"check": "the_two_supply_arms_share_no_rule_and_no_state",
            "passed": all(findings.values()), "findings": findings,
            "per_arm_llm": D3.PER_ARM_LLM, "ceilings":
                {"fits": D3.MAX_FITS, "llm": D3.MAX_LLM,
                 "wall_seconds": D3.MAX_WALL_SECONDS}}


def the_groups_are_registered_before_the_runs() -> dict[str, Any]:
    orders = {g: tuple(plan["arm_order"]) for g, plan in D3.GROUPS.items()}
    seeds = {g: plan["seed"] for g, plan in D3.GROUPS.items()}
    firsts = {order[0] for order in orders.values()}
    return {
        "check": "the_groups_are_registered_before_the_runs",
        "passed": (len(set(orders.values())) == len(orders)
                   and len(set(seeds.values())) == len(seeds)
                   and len(firsts) == len(orders)
                   and all(set(o) == {D3.ARM_A, D3.ARM_B, D3.ARM_C}
                           for o in orders.values())),
        "arm_orders": {g: list(o) for g, o in orders.items()},
        "seeds": seeds,
        "why_the_order_differs": ("so the cold-cache cost does not always fall "
                                  "on the same arm"),
    }


def when_the_candidate_formed_is_recorded() -> dict[str, Any]:
    event_source = inspect.getsource(D3._verification_event)
    knowledge_source = inspect.getsource(D3._knowledge_state)
    build_source = inspect.getsource(D3.build)
    findings = {
        "the_parent_at_proposal_is_carried":
            "parent_program_at_proposal" in event_source
            and "entry.setdefault(\"parent_program_at_proposal\"" in build_source,
        "the_card_at_proposal_is_carried":
            "card_body_at_proposal" in event_source,
        "the_card_before_and_after_the_event":
            "card_body_before_the_event" in event_source
            and "card_body_after_the_event" in event_source,
        "the_four_positions": all(
            name in event_source or name in build_source
            for name in ("proposed_at_position", "verified_at_position",
                         "first_later_use")),
        "whether_the_call_read_an_earlier_update":
            "this_call_reads_an_updated_card" in knowledge_source,
        "no_new_hash_system": "sha" not in knowledge_source.lower(),
    }
    return {"check": "when_the_candidate_formed_is_recorded",
            "passed": all(findings.values()), "findings": findings}


CHECKS: tuple[Callable[[], dict[str, Any]], ...] = (
    a_restored_ledger_does_not_mint_a_used_id,
    the_live_entry_state_is_the_shape_this_fixes,
    b_supplies_and_verifies_but_never_patches,
    c_patches_through_the_authorised_route,
    the_two_supply_arms_share_no_rule_and_no_state,
    the_groups_are_registered_before_the_runs,
    when_the_candidate_formed_is_recorded,
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
    # The Frame C wiring is unchanged; re-use DEV-AUTO-2's preflight whole
    # rather than restating it.
    inherited = smoke2.run()
    rows.append({"check": "the_dev_auto2_frame_c_preflight_still_passes",
                 "passed": bool(inherited["passed"]),
                 "checks": [r["check"] for r in inherited["checks"]
                            if not r["passed"]] or "all pass"})
    return {"stage": "DEV_AUTO3_SMOKE",
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
