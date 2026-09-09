"""Zero-fit, zero-LLM preflight for DEV-AUTO-1.

Nine things have to be true before a Consumer is fitted or a model is called,
and every one of them can be checked without doing either:

1. **Population and boundary.**  The forward / A5-online ordering holds 26
   cells; five carry ``TARGET_HELD_IN`` and are excluded *by exposure name*,
   leaving 21 development units, five before the k1 boundary and sixteen after.
2. **Entry state.**  M-R0d's reconstruction at k1 produces the five processed
   contexts, their bank rows, the Draft ledger as it stood and K0's Active
   lineage -- and both arms start from a deep copy of it, so neither begins
   ahead of the other.
3. **The Skill under revision.**  The K0 card is readable, its frozen body
   parses (with the Fast consumer's own parser) to the ancestor program, and
   its serving scope is the ancestor predicate.
4. **The edit is authorised by an existing route.**  ``OUTCOME_GAP`` authorises
   ``PATCH`` of target class ``capability``; the body surface resolves; and
   ``RISK_GAP`` is confirmed *not* to authorise it, so the package is using the
   route that exists rather than the one that would be convenient.
5. **The proposal schema.**  A well-formed Scope proposal and a well-formed
   Workflow proposal validate; a third kind, an over-long program and an extra
   field are refused.
6. **The feedback is not truncated.**  Every unit given to the builder comes
   back out, with both faces and the earlier attempts.
7. **The evaluator refuses without spending.**  An illegal operator and a
   feature outside the frozen vocabulary are both refused before the replay
   callable is ever entered -- the callable used here raises if touched.
8. **The ceilings refuse before the spend.**
9. **The evaluation face is not reachable.**  The runner's source never names
   the +144 offset or ``_evaluate_face``.

Run:  python -m evaluation.main_protocol_p4.smoke_dev_auto1_skill_revision
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_auto1_revision as rev
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as m_r0k

RUNNER_SOURCE = Path(__file__).resolve().parent / "run_dev_auto1_skill_revision.py"


class _NeverCalled(RuntimeError):
    """The replay callable was entered on a path that must refuse first."""


def _no_replay(**_kwargs: Any) -> dict[str, Any]:
    raise _NeverCalled("the evaluator reached a Consumer replay it should have "
                       "refused before")


def check_population() -> dict[str, Any]:
    from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as run

    doc = base.load(run.ORDERING)
    kept, excluded = run._population(doc)
    before = [r for r in kept if r["side"] == "before_the_boundary"]
    after = [r for r in kept if r["side"] == "after_the_boundary"]
    cells = base.cells(doc, run.SOURCE_ARM)
    return {
        "check": "population_and_boundary",
        "passed": (len(cells) == 26 and len(kept) == 21
                   and len(excluded) == 5 and len(before) == 5
                   and len(after) == 16),
        "cells_in_the_ordering": len(cells),
        "development_units": len(kept),
        "excluded": [{"position": r["position"], "why": r["why"]}
                     for r in excluded],
        "excluded_by": "exposure name, never by a property of a reading",
        "positions_before_the_boundary": [r["position"] for r in before],
        "positions_after_the_boundary": [r["position"] for r in after],
        "course_order_preserved": [r["position"] for r in after]
        == sorted(r["position"] for r in after),
    }


def check_entry_state() -> dict[str, Any]:
    doc = base.load(live.ORDERING)
    state = live._state_at_k1(doc)
    copied = copy.deepcopy(state["ledger"])
    same = ([d.to_dict() for d in copied.drafts]
            == [d.to_dict() for d in state["ledger"].drafts])
    return {
        "check": "entry_state_is_the_m_r0d_reconstruction",
        "passed": bool(state["processed"] and state["bank"] and same),
        "processed_units": len(state["processed"]),
        "bank_rows": len(state["bank"]),
        "held_lineage_keys": state["held"],
        "drafts": state["drafts_before"],
        "a_deep_copy_reproduces_the_ledger": same,
        "why_a_deep_copy": ("both arms start from the same reconstruction; "
                            "sharing one ledger object would let one arm's "
                            "revision appear in the other's state"),
    }


def check_skill_under_revision() -> dict[str, Any]:
    from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as run
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        _parse_frozen_steps,
    )
    from evaluation.main_protocol_p4 import run_source_line as v1runner

    doc = base.load(run.ORDERING)
    state = live._state_at_k1(doc)
    path, source = run._resolve_k0_snapshot(doc, state["k0"])
    if path is None:
        return {"check": "the_skill_under_revision", "passed": False,
                "why": "no readable K0 snapshot"}
    machinery = v1runner._machinery()
    snapshot = machinery["compile_snapshot"](path, verify_lock=False)
    card = next((s for s in snapshot.skills
                 if s.skill_id == run.K0_SKILL_ID), None)
    if card is None:
        return {"check": "the_skill_under_revision", "passed": False,
                "why": "the K0 card is not in the snapshot"}
    parsed = _parse_frozen_steps(card.body)
    steps = tuple((str(op), dict(params)) for op, params in (parsed or ()))
    # The card's scope comes back as frozen mappings inside a tuple; compared
    # after the same canonicalisation both sides get, so the check is about the
    # predicate and not about which container the compiler happened to use.
    scope = json.loads(json.dumps(drafts._plain(
        getattr(card, "serving_scope", None) or {}), sort_keys=True,
        default=str))
    wanted = json.loads(json.dumps(run.ANCESTOR_SCOPE, sort_keys=True))
    return {
        "check": "the_skill_under_revision",
        "passed": (steps == tuple(run.ANCESTOR_PROGRAM) and scope == wanted),
        "snapshot": {"path": str(path.relative_to(base.ROOT)),
                     "resolved_from": source,
                     "runtime_bundle_sha": snapshot.runtime_bundle_sha},
        "skill_id": card.skill_id,
        "frozen_body_parses_to": rev.program_label(steps),
        "ancestor_program": rev.program_label(run.ANCESTOR_PROGRAM),
        "serving_scope": scope,
        "parser": "fast_agent._parse_frozen_steps, the Fast consumer's own",
    }


def check_edit_route() -> dict[str, Any]:
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: PLC0415
        SurfaceRegistry,
    )
    from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import (  # noqa: PLC0415
        FaultRouter,
    )
    from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as run

    surface_id = "skill_library.entries/%s.body" % run.K0_SKILL_ID
    registry, router = SurfaceRegistry(), FaultRouter()
    resolved = registry.resolve(surface_id)
    authorised, why = True, None
    try:
        router.authorize(rev.WORKFLOW_CAUSE,
                         target_class=resolved.definition.target_class,
                         operation="PATCH", skill_kind="capability",
                         target_surface_id=surface_id)
    except Exception as exc:  # noqa: BLE001
        authorised, why = False, str(exc)
    risk_gap_refused = False
    try:
        router.authorize("RISK_GAP", target_class="capability",
                         operation="PATCH", skill_kind="capability",
                         target_surface_id=surface_id)
    except Exception:  # noqa: BLE001 - refusal is the expected answer
        risk_gap_refused = True
    return {
        "check": "the_workflow_edit_uses_an_existing_authorised_route",
        "passed": bool(authorised and risk_gap_refused),
        "surface": surface_id,
        "target_class": resolved.definition.target_class,
        "allowed_operations": list(resolved.definition.allowed_operations),
        "cause": rev.WORKFLOW_CAUSE,
        "authorised": authorised,
        "why_not": why,
        "risk_gap_is_refused_for_this_edit": risk_gap_refused,
        "routes_added": 0, "surfaces_added": 0,
    }


def _validates(payload: Any) -> tuple[bool, str | None]:
    from SelfEvolvingHarnessTS.methods.ttha.schema_contracts import (  # noqa: PLC0415
        validate_local_schema,
    )
    try:
        validate_local_schema(payload, rev.PROPOSAL_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]
    return True, None


def check_proposal_schema() -> dict[str, Any]:
    good_scope = {"proposals": [{
        "kind": "scope_clause",
        "scope_clause": {"feature": "missing_fraction", "op": ">="},
        "what_changes": "narrows the scope",
        "expected_change": "drops the damaged series"}]}
    good_workflow = {"proposals": [{
        "kind": "workflow_program",
        "workflow_program": [{"op": "period_median_complete",
                              "params": {"period": 24}},
                             {"op": "outlier_mad", "params": {}}],
        "what_changes": "imputes before clipping",
        "expected_change": "the gappy series stop being clipped into noise"}]}
    empty = {"proposals": []}
    bad_kind = {"proposals": [{"kind": "retrain", "what_changes": "x",
                               "expected_change": "y"}]}
    too_long = {"proposals": [{
        "kind": "workflow_program",
        "workflow_program": [{"op": "a"}, {"op": "b"}, {"op": "c"}],
        "what_changes": "x", "expected_change": "y"}]}
    extra = {"proposals": [], "threshold": 0.5}
    kind_without_payload = {"proposals": [{
        "kind": "workflow_program", "what_changes": "x",
        "expected_change": "y"}]}
    mixed = {"proposals": [{
        "kind": "scope_clause",
        "scope_clause": {"feature": "missing_fraction", "op": ">="},
        "workflow_program": [{"op": "outlier_iqr"}],
        "what_changes": "x", "expected_change": "y"}]}
    rows = {
        "a_scope_proposal": _validates(good_scope),
        "a_workflow_proposal": _validates(good_workflow),
        "an_empty_proposal_list": _validates(empty),
        "an_unknown_kind": _validates(bad_kind),
        "a_three_operator_program": _validates(too_long),
        "an_extra_top_level_field": _validates(extra),
        "a_kind_with_no_payload": _validates(kind_without_payload),
        "both_payloads_at_once": _validates(mixed),
    }
    passed = (rows["a_scope_proposal"][0] and rows["a_workflow_proposal"][0]
              and rows["an_empty_proposal_list"][0]
              and not rows["an_unknown_kind"][0]
              and not rows["a_three_operator_program"][0]
              and not rows["an_extra_top_level_field"][0]
              and not rows["a_kind_with_no_payload"][0]
              and not rows["both_payloads_at_once"][0])
    return {
        "check": "the_proposal_schema_admits_both_edits_and_nothing_else",
        "passed": bool(passed),
        "results": {key: {"valid": value[0], "why_not": value[1]}
                    for key, value in rows.items()},
        "schema_is_passed_inline": True,
        "why_inline": ("every file under methods/ttha/schemas is a dependency "
                       "SHA of every snapshot lock; adding one would rotate "
                       "K0's runtime bundle and this package would no longer "
                       "start from the K0 the course started from"),
        "schema_files_added": 0,
    }


def check_feedback_is_whole() -> dict[str, Any]:
    history = [{
        "position": index, "unit": {"block": "[0:40]", "origin": 1000 + index},
        "deployed_label": "outlier_mad({})",
        "deployed_scope": dict(m_r0k.ANCESTOR_SCOPE),
        "support_reading": {"treated": 6, "served": 20, "aggregate_gain": 0.1,
                            "per_series_gain": {"a": 0.4, "b": -0.3, "c": 0.0},
                            "max_single_series_harm": 0.3},
        "delayed_reading": {"treated": 6, "served": 20, "aggregate_gain": -0.02,
                            "per_series_gain": {"a": 0.1, "b": -0.9, "c": 0.0},
                            "max_single_series_harm": 0.9},
        "delayed_gate": {"passes": False, "failed_lines": ["single_series_harm"]},
    } for index in range(100)]
    attempts = [{"kind": "scope_clause", "outcome": "NO_FEASIBLE_THRESHOLD"}]
    feedback = rev.build_feedback(
        skill_id="s", program_steps=m_r0k.ANCESTOR_PROGRAM,
        serving_scope=m_r0k.ANCESTOR_SCOPE, history=history,
        attempts=attempts, vocabulary=contract.SCOPE_CLASS["vocabulary"])
    first = feedback["record_of_this_skill"][0]
    passed = (feedback["units_sent"] == len(history)
              and feedback["nothing_was_truncated"]
              and first["support_face"]["series_that_gained"] == ["a"]
              and first["delayed_face"]["series_that_lost"] == ["b"]
              and first["delayed_face"]["gate_failed_lines"]
              == ["single_series_harm"]
              and len(feedback["earlier_attempts_and_what_they_did"]) == 1
              and len(feedback["what_a_workflow_program_may_use"]) > 1)
    return {
        "check": "the_feedback_is_whole",
        "passed": bool(passed),
        "units_available": feedback["units_available"],
        "units_sent": feedback["units_sent"],
        "nothing_was_truncated": feedback["nothing_was_truncated"],
        "both_faces_present": ["support_face", "delayed_face"],
        "gained_lost_unchanged_are_named": {
            "gained": first["support_face"]["series_that_gained"],
            "lost": first["delayed_face"]["series_that_lost"],
            "unchanged": first["support_face"]["series_unchanged"]},
        "earlier_attempts_carried": len(
            feedback["earlier_attempts_and_what_they_did"]),
        "operators_offered": len(feedback["what_a_workflow_program_may_use"]),
        "what_the_production_prompt_did": ("rows[:60] of one face with no unit "
                                           "or series identity, and one line "
                                           "of 'direction not feasible' for an "
                                           "earlier failure"),
    }


def check_evaluator_refuses_without_spending() -> dict[str, Any]:
    parent_summary = {"mean_aggregate_gain_over_the_served_population": 0.25}
    illegal = rev.evaluate_proposal(
        {"kind": "workflow_program",
         "workflow_program": [{"op": "not_an_operator", "params": {}}],
         "what_changes": "x", "expected_change": "y"},
        parent_steps=m_r0k.ANCESTOR_PROGRAM, parent_scope=m_r0k.ANCESTOR_SCOPE,
        root_scope=m_r0k.ANCESTOR_SCOPE, bank_rows=[], replay=_no_replay,
        parent_summary=parent_summary)
    unchanged = rev.evaluate_proposal(
        {"kind": "workflow_program",
         "workflow_program": [{"op": "outlier_mad", "params": {}}],
         "what_changes": "x", "expected_change": "y"},
        parent_steps=m_r0k.ANCESTOR_PROGRAM, parent_scope=m_r0k.ANCESTOR_SCOPE,
        root_scope=m_r0k.ANCESTOR_SCOPE, bank_rows=[], replay=_no_replay,
        parent_summary=parent_summary)
    off_vocabulary = rev.evaluate_proposal(
        {"kind": "scope_clause",
         "scope_clause": {"feature": "series_uid", "op": ">="},
         "what_changes": "x", "expected_change": "y"},
        parent_steps=m_r0k.ANCESTOR_PROGRAM, parent_scope=m_r0k.ANCESTOR_SCOPE,
        root_scope=m_r0k.ANCESTOR_SCOPE, bank_rows=[], replay=_no_replay,
        parent_summary=parent_summary)
    return {
        "check": "the_evaluator_refuses_before_it_spends",
        "passed": (illegal["outcome"] == "ILLEGAL_PROGRAM"
                   and unchanged["outcome"] == "NO_CHANGE"
                   and off_vocabulary["outcome"] == "SLOW_CLAUSE_UNUSABLE"),
        "an_operator_outside_the_registry": {
            "outcome": illegal["outcome"], "why": illegal.get("why")},
        "the_parent_program_proposed_again": {
            "outcome": unchanged["outcome"], "why": unchanged.get("why")},
        "a_feature_outside_the_frozen_vocabulary": {
            "outcome": off_vocabulary["outcome"],
            "why": off_vocabulary.get("why")},
        "the_replay_callable_raises_if_entered": True,
        "consumer_fits_spent": 0,
    }


def check_ceilings() -> dict[str, Any]:
    from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as run

    budget = run.PackageBudget()
    budget.spend_fits("a", run.MAX_FITS - 1)
    room_for_one = budget.room_for(1)
    room_for_two = budget.room_for(2)
    budget.spend_fits("a", 1)
    fit_stop = budget.stop_reason()
    llm_budget = run.PackageBudget()
    llm_budget.spend_llm("a", run.MAX_LLM)
    llm_stop = llm_budget.stop_reason()
    return {
        "check": "the_package_ceilings_refuse_before_the_spend",
        "passed": (room_for_one and not room_for_two
                   and fit_stop == "FIT_CEILING_REACHED"
                   and llm_stop == "LLM_CEILING_REACHED"),
        "fit_ceiling": run.MAX_FITS, "llm_ceiling": run.MAX_LLM,
        "wall_ceiling_seconds": run.MAX_WALL_SECONDS,
        "one_more_fit_fits": room_for_one,
        "two_more_do_not": not room_for_two,
        "stop_reasons": {"fits": fit_stop, "llm": llm_stop},
        "there_is_no_per_candidate_llm_cap": True,
    }


def check_no_evaluation_face() -> dict[str, Any]:
    source = RUNNER_SOURCE.read_text(encoding="utf-8")
    names = ["_evaluate_face", "EVALUATION_OFFSET", "evaluation_origin"]
    hits = {name: source.count(name) for name in names}
    docstring_only = source.count("evaluation-face call") + source.count(
        "+144 evaluation face")
    return {
        "check": "the_evaluation_face_is_not_reachable_from_this_runner",
        "passed": all(count == 0 for count in hits.values()),
        "occurrences": hits,
        "mentions_in_prose_only": docstring_only,
        "why": ("run_unit_arm scores the +144 face on every cell; this package "
                "is authorised not to read it, so the call is absent rather "
                "than guarded"),
    }


def run() -> dict[str, Any]:
    checks = [check_population(), check_entry_state(),
              check_skill_under_revision(), check_edit_route(),
              check_proposal_schema(), check_feedback_is_whole(),
              check_evaluator_refuses_without_spending(), check_ceilings(),
              check_no_evaluation_face()]
    return {"smoke": "DEV_AUTO1_PREFLIGHT", "llm_calls": 0, "consumer_fits": 0,
            "passed": all(row["passed"] for row in checks), "checks": checks}


def main(argv: list[str] | None = None) -> int:
    report = run()
    print(json.dumps(drafts._plain(report), ensure_ascii=False, indent=1,
                     default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
