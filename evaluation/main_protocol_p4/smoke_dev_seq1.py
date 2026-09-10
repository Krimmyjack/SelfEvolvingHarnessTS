"""DEV-SEQ-1 preflight: the four behaviours the work order asks to be checked.

    1  the binding does not cross wires -- every request is bound to its own
       sequence, with its own values, observed pattern and features;
    2  different programs execute correctly -- a heterogeneous assignment is
       scored bit-for-bit as each sequence's own program, an untreated
       sequence is bit-identical to Static, and a sequence's reading does not
       change when other sequences are treated alongside it;
    3  grouping keeps positives as well as negatives, and leaves a lone
       failure ungrouped rather than inventing a group for it;
    4  a knowledge update can enter the next Fast -- the guidance card is
       retrieved for a matching sequence, withheld from a non-matching one,
       supplies no candidate, and a proposal that tries to be a program
       instead of guidance is refused.

Plus the boundary conditions this package must not quietly change: the
registered segments, the transport, and the core files.

Zero LLM calls.  Checks 2 spends real Consumer fits -- it is an execution
check, so it executes -- and reports how many; the runner adds them to the
package ledger.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Callable

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq1_knowledge as know
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_source_line as v1runner
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as m_r0k

#: The working-tree state of the method chain before this package started.
#: The package claims to add no core edit; this is what that claim is checked
#: against rather than asserted.
CORE_BASELINE = (" M methods/ttha/online_loop.py",
                 " M methods/ttha/scope_executor.py")

PROGRAM_A = m_r0k.ANCESTOR_PROGRAM
PROGRAM_B = m_r0k.WORKFLOW_CANDIDATES["W1_hampel_filter"]

_STATE: dict[str, Any] = {}


def _machinery() -> dict[str, Any]:
    if "machinery" not in _STATE:
        _STATE["machinery"] = v1runner._machinery()
    return _STATE["machinery"]


def _unit(position: int) -> Any:
    key = "unit_%d" % position
    if key not in _STATE:
        doc = base.load(R.ORDERING)
        population, _excluded = R._population(doc)
        row = next(r for r in population if r["position"] == int(position))
        _STATE[key] = R.opp._unit_ctx(row["unit"])
    return _STATE[key]


def _start_snapshot() -> Any:
    if "snapshot" not in _STATE:
        from evaluation.main_protocol_p4 import (
            run_m_r0d_forward_k1_outer_step as live,
        )
        doc = base.load(R.ORDERING)
        state = live._state_at_k1(doc)
        directory, _source = R._resolve_k0_snapshot(doc, state["k0"])
        _STATE["snapshot"] = _machinery()["compile_snapshot"](
            directory, verify_lock=False)
    return _STATE["snapshot"]


# ---------------------------------------------------------------------------

def the_segments_are_registered_and_do_not_overlap() -> dict[str, Any]:
    from evaluation.main_protocol_p4 import run_dev_seq1 as run

    doc = base.load(R.ORDERING)
    population, excluded = R._population(doc)
    exposed = {row["position"] for row in population}
    held = {row["position"] for row in excluded}
    formation = set(run.FORMATION_POSITIONS)
    construction = set(run.CONSTRUCTION_POSITIONS)
    followup = set(run.FOLLOWUP_POSITIONS)
    origins = {p: int(_unit(p).origin)
               for p in sorted(formation | followup)}
    construction_windows = {(origins[p], origins[p] + ps.DELAYED)
                            for p in construction}
    followup_windows = {(origins[p], origins[p] + ps.DELAYED)
                        for p in followup}
    withheld = int(origins[run.WITHHELD_POSITION])
    findings = {
        "every_unit_is_an_exposed_development_unit":
            (formation | followup) <= exposed,
        "no_target_held_in_unit_is_touched": not (formation | followup) & held,
        "construction_is_inside_formation": construction <= formation,
        "the_withheld_unit_is_not_in_the_construction_set":
            run.WITHHELD_POSITION not in construction,
        "formation_and_followup_do_not_overlap": not formation & followup,
        "the_withheld_window_is_after_the_construction_windows":
            all(withheld > high for _low, high in construction_windows),
        "the_followup_windows_are_after_the_withheld_window":
            all(low > withheld + ps.DELAYED for low, _high in followup_windows),
    }
    return {"check": "the_segments_are_registered_and_do_not_overlap",
            "passed": all(findings.values()), "findings": findings,
            "origins": origins, "consumer_fits": 0}


def every_request_is_bound_to_its_own_sequence() -> dict[str, Any]:
    """The correction itself, executed: one request per sequence, not one per
    unit built from the block's first eval series."""
    machinery = _machinery()
    ctx = _unit(7)
    origin = int(ctx.origin)
    population = ps.decision_uids(ctx, size=10)
    requests = {uid: ps.sequence_request(machinery, ctx, uid, origin=origin,
                                         domain="devseq1")
                for uid in population}
    features = {uid: ps.sequence_features(machinery, ctx, uid, origin=origin)
                for uid in population}
    representative = np.asarray(ctx.at.observation_block, dtype=np.float64)
    representative_uid = str(ctx.at.support_a[0])
    first_eval = np.asarray(ctx.at.values[representative_uid],
                            dtype=np.float64)[:origin]
    cohort_request = machinery["runner"]._a5_request(
        ctx.at.observation_block, ctx.at.values, origin, "devseq1")

    bound = all(str(requests[uid].series_uid) == uid for uid in population)
    values_match = all(
        np.array_equal(np.asarray(requests[uid].values, dtype=np.float64),
                       np.asarray(ctx.at.values[uid],
                                  dtype=np.float64)[:origin],
                       equal_nan=True)
        for uid in population)
    no_crosstalk = all(
        not np.array_equal(np.asarray(requests[a].values, dtype=np.float64),
                           np.asarray(requests[b].values, dtype=np.float64),
                           equal_nan=True)
        for a in population for b in population if a < b)
    observed = {uid: json.dumps(
        {k: round(float(v), 9) for k, v in requests[uid].observed_pattern_spec.items()},
        sort_keys=True) for uid in population}
    feature_json = {uid: json.dumps(
        {k: (round(float(v), 9) if isinstance(v, (int, float))
             and not isinstance(v, bool) else v)
         for k, v in features[uid].items()}, sort_keys=True)
        for uid in population}
    findings = {
        "each_request_carries_its_own_uid": bound,
        "each_request_carries_its_own_values": values_match,
        "no_two_sequences_share_a_values_array": no_crosstalk,
        "each_sequence_has_its_own_observed_pattern":
            len(set(observed.values())) == len(population),
        "each_sequence_has_its_own_public_features":
            len(set(feature_json.values())) > 1,
        "the_task_and_consumer_are_the_same_for_all":
            len({str(requests[uid].task_spec) for uid in population}) == 1,
        "the_old_shape_really_was_one_representative_series":
            bool(np.array_equal(representative[:origin], first_eval,
                                equal_nan=True)),
        "the_representative_is_one_member_of_the_population":
            representative_uid in population,
        "the_old_request_named_a_cohort_pseudo_uid":
            str(cohort_request.series_uid) not in population,
    }
    return {"check": "every_request_is_bound_to_its_own_sequence",
            "passed": all(findings.values()), "findings": findings,
            "population": population,
            "the_series_the_old_shape_spoke_for": representative_uid,
            "distinct_feature_cards": len(set(feature_json.values())),
            "consumer_fits": 0}


def different_programs_execute_on_their_own_data() -> dict[str, Any]:
    """A heterogeneous assignment, executed and cross-checked bit for bit."""
    ctx = _unit(7)
    origin = int(ctx.origin)
    population = ps.decision_uids(ctx, size=3)
    cache = runner.ReplayPredictionCache("smoke")
    readings = ps.UnitReadings(cache=cache, ctx=ctx, executor=ctx.executor)
    assignment = {population[0]: PROGRAM_A,
                  population[1]: PROGRAM_B,
                  population[2]: ()}
    composed, _fits = readings.compose(assignment, origin, population)

    alone: dict[str, float] = {}
    for uid, steps in assignment.items():
        if not steps:
            continue
        reading, _spent = readings.reading(steps, origin, (uid,))
        alone[uid] = float(reading["per_series_gain"][uid])

    exact = all(composed["per_series_gain"][uid] == round(alone[uid], 6)
                for uid in alone)
    identity_uid = population[2]
    findings = {
        "the_assignment_used_more_than_one_program":
            composed["distinct_programs"] == 2,
        "each_treated_sequence_is_scored_under_its_own_program": exact,
        "the_untreated_sequence_scores_exactly_zero":
            composed["per_series_gain"][identity_uid] == 0.0,
        "the_two_programs_are_not_the_same_number":
            len({round(value, 9) for value in alone.values()}) == 2,
        "the_treated_count_is_the_deployed_count": composed["treated"] == 2,
    }
    return {"check": "different_programs_execute_on_their_own_data",
            "passed": all(findings.values()), "findings": findings,
            "per_series_gain": composed["per_series_gain"],
            "read_alone": {uid: round(value, 6) for uid, value in alone.items()},
            "consumer_fits": int(cache.physical_fits),
            "_cache": cache}


def the_reading_does_not_depend_on_co_treatment(
        previous: dict[str, Any]) -> dict[str, Any]:
    """One sequence's number must not move when a neighbour is treated too."""
    ctx = _unit(7)
    origin = int(ctx.origin)
    cache = previous.get("_cache") or runner.ReplayPredictionCache("smoke")
    readings = ps.UnitReadings(cache=cache, ctx=ctx, executor=ctx.executor)
    population = ps.decision_uids(ctx, size=4)
    before = int(cache.physical_fits)
    single, _s = readings.reading(PROGRAM_A, origin, (population[0],))
    together, _t = readings.reading(PROGRAM_A, origin, population)
    spent = int(cache.physical_fits) - before
    findings = {
        "the_same_series_reads_the_same_number":
            single["per_series_gain"][population[0]]
            == together["per_series_gain"][population[0]],
        "co_treatment_cost_no_extra_fit": spent == 0,
        "the_neighbours_did_move": any(
            together["per_series_gain"][uid] != single["per_series_gain"][uid]
            for uid in population[1:]),
    }
    return {"check": "the_reading_does_not_depend_on_co_treatment",
            "passed": all(findings.values()), "findings": findings,
            "alone": single["per_series_gain"][population[0]],
            "with_neighbours": together["per_series_gain"][population[0]],
            "consumer_fits": spent}


def an_untreated_sequence_is_bit_identical_to_static() -> dict[str, Any]:
    """The 0.0 an abstaining sequence scores is measured, not filled in."""
    ctx = _unit(7)
    origin = int(ctx.origin)
    at = ctx.cell_at(origin)
    roster = at.roster(runner.FACE)
    config = runner.forecast_p4._config(origin)
    executor = runner._executor(roster, at.values, config)
    compiled = executor._compiled(tuple(PROGRAM_A))
    static = scoped.scoped_evaluate(roster, at.values, None, config,
                                    origin=origin)
    empty_scope = scoped.scoped_evaluate(roster, at.values, compiled, config,
                                         origin=origin, scope=frozenset())
    findings = {
        "an_empty_scope_reproduces_static_exactly":
            list(static["per_view_smase"]) == list(
                empty_scope["per_view_smase"]),
        "an_empty_scope_never_ran_the_program_pipeline":
            empty_scope["program_pipeline_used"] is False,
    }
    return {"check": "an_untreated_sequence_is_bit_identical_to_static",
            "passed": all(findings.values()), "findings": findings,
            "consumer_fits": int(static["consumer_fits"])
                             + int(empty_scope["consumer_fits"])}


def _episode(uid: str, steps, gain: float, origin: int, suffix: str) -> Any:
    from SelfEvolvingHarnessTS.methods.ttha import online_loop
    return online_loop._write_target_episode(
        domain="devseq1", op=steps[0][0] if steps else "identity",
        program_steps=[{"op": op, "params": dict(params)}
                       for op, params in steps],
        support_gain=float(gain), support_context={},
        episode_id_suffix=suffix, per_view_gain=[float(gain)],
        support_origin=int(origin), task_spec=None, series_uids=[uid],
        consumer_id="ridge")


def grouping_keeps_the_successes_and_leaves_singletons_alone() -> dict[str, Any]:
    """Grouping on real Episodes written by the real writer.  No LLM, no fits."""
    episodes = [
        _episode("S1", PROGRAM_A, -0.20, 1176, "_a1"),
        _episode("S2", PROGRAM_A, -0.15, 1176, "_a2"),
        _episode("S3", PROGRAM_A, +0.30, 1176, "_a3"),
        _episode("S4", PROGRAM_B, -0.12, 1176, "_b1"),
    ]
    features = {"S1": {"local_robust_z_peak": 5.0, "task_kind": "forecast"},
                "S2": {"local_robust_z_peak": 4.2, "task_kind": "forecast"},
                "S3": {"local_robust_z_peak": 1.1, "task_kind": "forecast"},
                "S4": {"local_robust_z_peak": 3.4, "task_kind": "forecast"}}
    out = know.group_failures(episodes=episodes, features=features)
    groups = out["groups"]
    group = groups[0] if groups else {}
    findings = {
        "the_two_similar_failures_formed_one_group":
            len(groups) == 1 and group.get("member_count") == 2,
        "the_group_kept_its_matched_success":
            bool((group.get("contrast_cases") or {}).get("positive")),
        "the_lone_failure_stayed_an_episode":
            [row["series_uid"] for row in out["ungrouped_failures"]] == ["S4"],
        "comparability_was_checked_first":
            out["comparability"]["comparable"] is True,
        "the_group_recorded_the_visible_pattern_spread":
            bool(group.get("visible_pattern_spread")),
        "the_group_carries_a_symptom_not_only_a_sign":
            bool(group.get("symptoms")),
        "the_fault_choice_is_evidence_masked":
            isinstance(group.get("fault", {}).get("selectable_fault_types"),
                       list),
    }
    return {"check": "grouping_keeps_the_successes_and_leaves_singletons_alone",
            "passed": all(findings.values()), "findings": findings,
            "group_count": len(groups),
            "ungrouped": [row["series_uid"] for row in out["ungrouped_failures"]],
            "consumer_fits": 0}


def _guidance_manifest(snapshot: Any, *, skill_id: str, body: str,
                       applicability: dict[str, Any],
                       guards: dict[str, Any] | None = None,
                       serving_scope: dict[str, Any] | None = None) -> Any:
    from SelfEvolvingHarnessTS.contracts.harness import (
        EditManifest,
        EditOperation,
    )
    value: dict[str, Any] = {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id,
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": dict(applicability),
        "allowed_tools": [],
        "risk_guards": dict(guards or {
            "explicit_choice_required": True,
            "observable_applicability_only": True,
            "preserve_outside_candidate_region": True,
            "single_surface_only": True,
            "requires_target_support": True}),
    }
    if serving_scope:
        value["serving_scope"] = dict(serving_scope)
    return EditManifest(
        edit_id=skill_id,
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id="devseq1-smoke",
        target_surface_id="skill_library.entries/%s" % skill_id,
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value=value,
        observable_applicability=dict(applicability),
        patch_id=None,
        predicted_agent_behavior_change=("retrieve_skill:%s" % skill_id,),
        predicted_data_effect=("changed_candidate_construction",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_change_in_what_fast_proposes",),
    )


def a_knowledge_update_can_enter_the_next_fast() -> dict[str, Any]:
    """Written through the real controller, read back through real retrieval."""
    from SelfEvolvingHarnessTS.contracts.harness import SkillKind
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _skill_frozen_candidates,
    )
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

    machinery = _machinery()
    snapshot = _start_snapshot()
    store = machinery["SnapshotStore"](
        base.ROOT / "_scratch" / "dev_seq1" / "smoke_store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    store.materialize(snapshot)
    parent_bodies = {s.skill_id: s.body for s in snapshot.skills}

    applicability = {"all": [{"feature": "task_kind", "op": "==",
                              "value": "forecast"},
                             {"feature": "local_robust_z_peak", "op": ">=",
                              "value": 3.0}]}
    manifest = _guidance_manifest(
        snapshot, skill_id="devseq1_guidance_smoke",
        body=("Guidance, not a program: on a sequence whose local robust "
              "z peak is high, inspect the peak's width before proposing a "
              "clipping operator, and prefer a narrower repair when the peak "
              "is one point wide."),
        applicability=applicability)
    applied = know.apply_update(controller=controller, store=store,
                                snapshot=snapshot, manifest=manifest)
    findings: dict[str, Any] = {"the_guidance_card_compiled":
                                bool(applied.get("applied"))}
    detail: dict[str, Any] = {"apply": {k: v for k, v in applied.items()
                                        if k != "snapshot"}}
    if applied.get("applied"):
        updated = applied["snapshot"]
        matching = {"task_kind": "forecast", "local_robust_z_peak": 6.0}
        non_matching = {"task_kind": "forecast", "local_robust_z_peak": 0.4}
        view_in = resolve_harness_view(updated, matching, role="fast")
        view_out = resolve_harness_view(updated, non_matching, role="fast")
        findings.update({
            "a_matching_sequence_retrieves_it":
                "devseq1_guidance_smoke" in view_in.skill_ids,
            "a_non_matching_sequence_does_not":
                "devseq1_guidance_smoke" not in view_out.skill_ids,
            "the_parent_card_is_still_retrieved":
                any(sid.startswith("fast_winner_") for sid in view_in.skill_ids),
            "the_parent_card_body_is_unchanged": all(
                s.body == parent_bodies[s.skill_id]
                for s in updated.skills if s.skill_id in parent_bodies),
            "the_library_grew_by_exactly_one":
                len(updated.skills) == len(snapshot.skills) + 1,
            "guidance_supplies_no_candidate": all(
                not str(getattr(c, "source", "")).endswith(
                    "devseq1_guidance_smoke")
                for c in _skill_frozen_candidates(view_in, matching)),
            "the_body_reaches_the_prompt_surface": any(
                "Guidance, not a program" in s.body for s in view_in.skills),
            "it_is_a_capability_entry": any(
                s.skill_id == "devseq1_guidance_smoke"
                and s.skill_kind is SkillKind.CAPABILITY
                for s in updated.skills),
            "it_requires_target_support": any(
                s.skill_id == "devseq1_guidance_smoke"
                and (s.risk_guards or {}).get("requires_target_support") is True
                for s in updated.skills),
        })
        detail["retrieved_for_a_matching_sequence"] = list(view_in.skill_ids)
        detail["retrieved_for_a_non_matching_sequence"] = list(view_out.skill_ids)

    refusals: dict[str, str] = {}
    for name, kwargs in (
        ("a_frozen_program_body_is_refused",
         {"body": 'Frozen program steps: [{"op": "outlier_mad", "params": {}}]'}),
        ("candidate_supply_authority_is_refused",
         {"body": "guidance",
          "guards": {"authority": {"supplies_candidates": True}}}),
        ("a_serving_scope_on_guidance_is_refused",
         {"body": "guidance",
          "serving_scope": {"scope_type": "serving_series_predicate"}}),
    ):
        candidate = _guidance_manifest(
            snapshot, skill_id="devseq1_guidance_refused",
            applicability=applicability, **kwargs)
        try:
            know.guidance_preflight(candidate)
            refusals[name] = "NOT_REFUSED"
        except know.GuidanceRefused as exc:
            refusals[name] = str(exc)[:120]
        findings[name] = refusals[name] != "NOT_REFUSED"
    detail["refusals"] = refusals
    return {"check": "a_knowledge_update_can_enter_the_next_fast",
            "passed": all(bool(value) for value in findings.values()),
            "findings": findings, "detail": detail, "consumer_fits": 0}


def the_transport_is_the_one_the_previous_package_used() -> dict[str, Any]:
    """A silent model or relay switch would make this a different experiment."""
    import os

    path = base.ROOT / "artifacts" / "main_protocol" / "dev_know1_two_arms__g1.json"
    receipt = {}
    if path.is_file():
        receipt = (json.loads(path.read_text(encoding="utf-8")) or {}).get(
            "transport") or {}
    live = _machinery()["agentic"].live_transport()
    findings = {
        "a_previous_receipt_names_a_transport": bool(receipt.get("model")),
        "the_base_url_matches": receipt.get("base_url") == live.get("base_url"),
        "the_model_matches": receipt.get("model") == live.get("model"),
        "it_comes_from_the_environment_not_a_module_default":
            live.get("source") == "M0_AGENT_*",
    }
    note = (os.environ.get("DEV_SEQ1_TRANSPORT_NOTE") or "").strip()
    matched = all(findings.values())
    return {"check": "the_transport_is_the_one_the_previous_package_used",
            "passed": matched or bool(note),
            "matches_the_previous_package": matched,
            "findings": findings, "expected": receipt, "live": live,
            "acknowledged_divergence": note or None,
            "consumer_fits": 0}


def this_package_edits_no_core_file() -> dict[str, Any]:
    """The method chain is read, not rewritten.  Checked, not asserted."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "methods/", "contracts/",
             "runtime/", "evaluation/minipipe/"],
            cwd=str(base.ROOT), capture_output=True, text=True, timeout=120)
        lines = tuple(line for line in out.stdout.splitlines() if line.strip())
    except Exception as exc:  # noqa: BLE001 - recorded, never assumed clean
        return {"check": "this_package_edits_no_core_file", "passed": False,
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:200]),
                "consumer_fits": 0}
    return {"check": "this_package_edits_no_core_file",
            "passed": lines == CORE_BASELINE,
            "baseline": list(CORE_BASELINE), "now": list(lines),
            "what_the_baseline_is": (
                "the two working-tree edits that predate this package "
                "(2026-09-05); anything else means a core file was touched"),
            "consumer_fits": 0}


CHECKS: tuple[Callable[[], dict[str, Any]], ...] = (
    the_segments_are_registered_and_do_not_overlap,
    every_request_is_bound_to_its_own_sequence,
    different_programs_execute_on_their_own_data,
    an_untreated_sequence_is_bit_identical_to_static,
    grouping_keeps_the_successes_and_leaves_singletons_alone,
    a_knowledge_update_can_enter_the_next_fast,
    the_transport_is_the_one_the_previous_package_used,
    this_package_edits_no_core_file,
)


def run() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    execution_row: dict[str, Any] | None = None
    for check in CHECKS:
        try:
            row = check()
        except Exception as exc:  # noqa: BLE001 - a failed check is a result
            row = {"check": check.__name__, "passed": False,
                   "why": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
                   "consumer_fits": 0}
        if check is different_programs_execute_on_their_own_data:
            execution_row = row
        rows.append(row)
        if execution_row is not None and check is (
                different_programs_execute_on_their_own_data):
            try:
                rows.append(the_reading_does_not_depend_on_co_treatment(
                    execution_row))
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "check": "the_reading_does_not_depend_on_co_treatment",
                    "passed": False,
                    "why": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
                    "consumer_fits": 0})
    for row in rows:
        row.pop("_cache", None)
    return {
        "smoke": "DEV_SEQ1",
        "passed": all(bool(row.get("passed")) for row in rows),
        "checks": rows,
        "consumer_fits": sum(int(row.get("consumer_fits") or 0) for row in rows),
        "llm_calls": 0,
    }


def main(argv: Any = None) -> int:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
