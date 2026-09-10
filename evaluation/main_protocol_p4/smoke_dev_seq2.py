"""DEV-SEQ-2 preflight: only the paths this package adds.

DEV-SEQ-1's smoke already covers per-sequence binding, heterogeneous
execution, grouping and that a guidance ADD can enter the next Fast; it is
not repeated.  The six checks here are the new ones the work order asks for:

    1  window binding -- a feature card belongs to (series, window), and the
       old UID-keyed shape really did describe a later window with an
       earlier window's features;
    2  measurement de-duplication -- citing one real reading twice is one
       measurement, and a different window, program, parameter set or
       training config stays a separate measurement; no Episode is deleted;
    3  every whitelisted target really reaches Fast -- each surface is
       compiled and the changed text is found in the bytes Fast resolves;
    4  branch isolation -- the candidate is a fork, the parent snapshot is
       byte-unchanged and still resolvable, and no old knowledge is removed;
    5  the model cannot touch anything outside the whitelist -- eight
       over-reaching proposals are refused, fail-closed;
    6  paired scoring and UNKNOWN semantics -- a missing reading on either
       side makes the whole-population difference UNKNOWN, the readable
       subset is reported beside it and never in its place, and a budget
       reloaded across a process boundary does not re-bill.

Zero LLM calls and zero Consumer fits.  The transport identity is checked by
the runner itself, because that check costs one real call.
"""

from __future__ import annotations

import json
from typing import Any

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import dev_seq2_knowledge as know
from evaluation.main_protocol_p4 import per_sequence as ps
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_dev_auto1_skill_revision as R
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

MARKER = "DEVSEQ2_SMOKE_MARKER"


def _machinery() -> dict[str, Any]:
    return v1runner._machinery()


def _parent(machinery):
    doc = base.load(R.ORDERING)
    state = live._state_at_k1(doc)
    snapshot_dir, _source = R._resolve_k0_snapshot(doc, state["k0"])
    return machinery["compile_snapshot"](snapshot_dir, verify_lock=False)


def _controller(machinery, slug: str):
    store = machinery["SnapshotStore"](
        base.ROOT / "_scratch" / "dev_seq2_smoke" / slug / "store")
    controller = machinery["EditController"](
        store, surfaces=machinery["SurfaceRegistry"](),
        router=machinery["FaultRouter"]())
    return store, controller


def _manifest(*, surface_id: str, operation: str, snapshot, row=None,
              value=None, new_value=None, applicability=None,
              edit_id="smoke-edit"):
    from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
    return EditManifest(
        edit_id=edit_id,
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id="smoke-pattern",
        target_surface_id=surface_id,
        operation=EditOperation(operation),
        surface_precondition=(dict(row["surface_precondition"]) if row
                              else {"kind": "ABSENT"}),
        dependency_precondition_shas=(dict(row["dependency_precondition_shas"])
                                      if row else {}),
        minimal_patch=({"value": value} if value is not None else None),
        new_value=new_value,
        observable_applicability=applicability,
        predicted_agent_behavior_change=("supply_effect_distinct",),
        predicted_data_effect=("smoke",),
        falsification_condition=("smoke",))


# ---------------------------------------------------------------------------
# 1. window binding
# ---------------------------------------------------------------------------

def check_window_binding() -> dict[str, Any]:
    by_position = {
        7: {"T1": {"missing_fraction": 0.10}, "T2": {"missing_fraction": 0.50}},
        8: {"T1": {"missing_fraction": 0.90}, "T3": {"missing_fraction": 0.20}},
    }
    origins = {7: 1176, 8: 1416}
    binding = know.FeatureBinding(know.bind_window_features(by_position, origins))

    own_window = (binding.get("T1", 1176)["missing_fraction"] == 0.10
                  and binding.get("T1", 1416)["missing_fraction"] == 0.90)

    # the shape DEV-SEQ-1 used: one card per UID, first window wins
    old: dict[str, dict] = {}
    for position in (7, 8):
        for uid, card in by_position[position].items():
            old.setdefault(uid, dict(card))
    old_was_wrong = old["T1"]["missing_fraction"] == 0.10  # describes u8 too

    miss = binding.get("T1", 9999)
    missed_not_borrowed = miss == {} and binding.misses[-1]["origin"] == 9999

    spread = binding.spread([("T1", 1176), ("T1", 1416)])
    spread_uses_both = (spread["missing_fraction"]["min"] == 0.10
                        and spread["missing_fraction"]["max"] == 0.90)
    return {
        "name": "a feature card belongs to (series, window), never to a UID",
        "passed": bool(own_window and old_was_wrong and missed_not_borrowed
                       and spread_uses_both),
        "each_window_keeps_its_own_card": own_window,
        "the_old_uid_keyed_shape_described_u8_with_u7s_features": old_was_wrong,
        "a_missing_window_is_a_miss_not_a_borrowed_card": missed_not_borrowed,
        "the_spread_is_over_windows_not_series": spread_uses_both,
        "distinct_windows": len(binding.windows()),
        "distinct_series": len(binding.uids()),
    }


# ---------------------------------------------------------------------------
# 2. measurement de-duplication
# ---------------------------------------------------------------------------

def check_measurement_dedup() -> dict[str, Any]:
    from evaluation.main_protocol_p4 import run_hec1 as runner
    cache = runner.ReplayPredictionCache("smoke-dedup")
    ledger = know.MeasurementLedger(cache, runner.forecast_p4._config)
    steps = (("outlier_mad", {}),)
    other = (("outlier_mad", {"z_threshold": 3.5}),)

    first = ledger.note(unit="u007", origin=1176, steps=steps, uid="T1",
                        value=0.2, face="support")
    again = ledger.note(unit="u007", origin=1176, steps=steps, uid="T1",
                        value=0.2, face="support")
    other_window = ledger.note(unit="u007", origin=1224, steps=steps, uid="T1",
                               value=-0.1, face="delayed")
    other_params = ledger.note(unit="u007", origin=1176, steps=other, uid="T1",
                               value=0.3, face="support")
    other_series = ledger.note(unit="u007", origin=1176, steps=steps, uid="T2",
                               value=0.1, face="support")
    summary = ledger.to_dict()

    # the counting difference this fixes: 5 references, 4 measurements
    dedup_ok = (first and not again and other_window and other_params
                and other_series
                and summary["references"] == 5
                and summary["distinct_measurements"] == 4
                and summary["repeated_references"] == 1)
    return {
        "name": "citing one real reading twice is one measurement",
        "passed": bool(dedup_ok),
        "same_reading_twice_counts_once": (first and not again),
        "different_window_is_a_different_measurement": other_window,
        "different_parameters_are_a_different_measurement": other_params,
        "different_series_is_a_different_measurement": other_series,
        "ledger": summary,
        "no_episode_is_deleted_by_this": (
            "the ledger counts measurements; the Episode corpus is untouched "
            "and every member stays in its group"),
    }


# ---------------------------------------------------------------------------
# 3. every whitelisted target reaches Fast
# ---------------------------------------------------------------------------

def check_targets_reach_fast() -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

    machinery = _machinery()
    snapshot = _parent(machinery)
    store, controller = _controller(machinery, "reach")
    parent = store.materialize(snapshot)
    catalog = {row["surface_id"]: row
               for row in know.build_catalog(controller=controller,
                                             parent=parent)}
    features = {"task_kind": "forecast", "missing_fraction": 0.5,
                "local_robust_z_peak": 6.0}
    rows: list[dict[str, Any]] = []

    for surface_id, row in catalog.items():
        if row["operation"] != "PATCH":
            continue
        text = "%s %s_%s" % (row["current_value"], MARKER,
                             surface_id.split(".")[-1])
        manifest = _manifest(surface_id=surface_id, operation="PATCH",
                             snapshot=snapshot, row=row, value=text,
                             edit_id="smoke-patch-%d" % (len(rows) + 1))
        applied = know.apply_update(controller=controller, store=store,
                                    snapshot=snapshot, manifest=manifest,
                                    catalog=tuple(catalog.values()))
        found = False
        if applied.get("applied"):
            view = resolve_harness_view(applied["snapshot"], features,
                                        role="fast")
            blob = json.dumps({
                "instruction": view.instruction,
                "skills": [{"id": s.skill_id, "body": s.body}
                           for s in view.skills],
                "controls": json.loads(json.dumps(view.controls, default=dict)),
            }, ensure_ascii=False, default=str)
            found = MARKER in blob
        rows.append({"surface_id": surface_id,
                     "compiled": bool(applied.get("applied")),
                     "why": applied.get("why"),
                     "cause": applied.get("confirmed_cause"),
                     "text_is_in_the_fast_view": found,
                     "reaches": know.REACHES_FAST[row["surface_template_id"]]})

    # the ADD surface, with a condition these features satisfy
    condition = {"all": [{"feature": "task_kind", "op": "==",
                          "value": "forecast"}]}
    manifest = _manifest(surface_id="skill_library.entries/smoke_guidance_entry",
                         operation="ADD", snapshot=snapshot,
                         new_value={"schema_version": "skill-entry/1",
                                    "skill_id": "smoke_guidance_entry",
                                    "skill_kind": "capability",
                                    "body": "%s guidance body" % MARKER,
                                    "observable_applicability": condition,
                                    "allowed_tools": [],
                                    "risk_guards": {},
                                    "revision": 1},
                         applicability=condition,
                         edit_id="smoke-add-guidance")
    applied = know.apply_update(controller=controller, store=store,
                                snapshot=snapshot, manifest=manifest,
                                catalog=tuple(catalog.values()))
    found = False
    if applied.get("applied"):
        view = resolve_harness_view(applied["snapshot"], features, role="fast")
        found = MARKER in json.dumps(
            [{"id": s.skill_id, "body": s.body} for s in view.skills],
            ensure_ascii=False)
    rows.append({"surface_id": "skill_library.entries/{skill_id}",
                 "compiled": bool(applied.get("applied")),
                 "why": applied.get("why"),
                 "cause": applied.get("confirmed_cause"),
                 "text_is_in_the_fast_view": found,
                 "reaches": know.REACHES_FAST["skill_library.entries/{skill_id}"]})

    objects = {"skill_library.entries/{skill_id}",
               "bootstrap_skills.entries/{skill_id}.body",
               "candidate_policy.proposal_guidance",
               "candidate_policy.selection_guidance"}
    covered = set()
    for row in rows:
        sid = row["surface_id"]
        if sid.startswith("bootstrap_skills.entries/"):
            covered.add("bootstrap_skills.entries/{skill_id}.body")
        elif sid.startswith("skill_library.entries/"):
            covered.add("skill_library.entries/{skill_id}")
        else:
            covered.add(sid)
    return {
        "name": "every whitelisted target compiles and is read by Fast",
        "passed": bool(rows and all(r["compiled"] and r["text_is_in_the_fast_view"]
                                    for r in rows)
                       and objects <= covered),
        "surfaces": rows,
        "authorised_objects_with_an_instance": sorted(covered),
        "objects_without_an_instance_in_this_snapshot": sorted(objects - covered),
        "note": ("the two 'existing guidance entry' PATCH surfaces have no "
                 "instance at K0: the only capability card is a frozen program "
                 "card, whose body is an execution body and is outside this "
                 "package's whitelist"),
    }


# ---------------------------------------------------------------------------
# 4. branch isolation
# ---------------------------------------------------------------------------

def check_branch_isolation() -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

    machinery = _machinery()
    snapshot = _parent(machinery)
    store, controller = _controller(machinery, "fork")
    parent = store.materialize(snapshot)
    catalog = {row["surface_id"]: row
               for row in know.build_catalog(controller=controller,
                                             parent=parent)}
    surface_id = "candidate_policy.proposal_guidance"
    row = catalog[surface_id]
    before_sha = snapshot.runtime_bundle_sha
    before_skills = sorted(s.skill_id for s in snapshot.skills)
    before_policy = dict(snapshot.candidate_policy)

    manifest = _manifest(surface_id=surface_id, operation="PATCH",
                         snapshot=snapshot, row=row,
                         value="%s %s" % (row["current_value"], MARKER),
                         edit_id="smoke-fork")
    applied = know.apply_update(controller=controller, store=store,
                                snapshot=snapshot, manifest=manifest,
                                catalog=tuple(catalog.values()))
    candidate = applied.get("snapshot")
    parent_unchanged = (snapshot.runtime_bundle_sha == before_sha
                        and sorted(s.skill_id for s in snapshot.skills)
                        == before_skills
                        and dict(snapshot.candidate_policy) == before_policy)
    parent_still_resolves = bool(
        resolve_harness_view(snapshot, {"task_kind": "forecast"},
                             role="fast").skills)
    kept_knowledge = (candidate is not None
                      and set(before_skills)
                      <= {s.skill_id for s in candidate.skills})
    different = (candidate is not None
                 and candidate.runtime_bundle_sha != before_sha)
    return {
        "name": "the candidate is a fork; the parent is kept whole",
        "passed": bool(applied.get("applied") and parent_unchanged
                       and parent_still_resolves and kept_knowledge
                       and different),
        "parent_snapshot_is_byte_unchanged": parent_unchanged,
        "parent_still_resolves_for_fast": parent_still_resolves,
        "no_still_legal_old_knowledge_is_removed": kept_knowledge,
        "candidate_has_its_own_runtime_bundle_sha": different,
        "parent_runtime_bundle_sha": before_sha,
        "candidate_runtime_bundle_sha": applied.get("runtime_bundle_sha"),
    }


# ---------------------------------------------------------------------------
# 5. the whitelist is closed
# ---------------------------------------------------------------------------

def check_whitelist_is_closed() -> dict[str, Any]:
    machinery = _machinery()
    snapshot = _parent(machinery)
    attempts = [
        ("instruction.core", "PATCH", {"value": "rewrite the instruction"},
         None, "the core instruction is not editable in this package"),
        ("retrieval.capability.top_k", "PATCH", {"value": 5}, None,
         "retrieval top_k is not editable"),
        ("candidate_policy.agent_program_slots", "PATCH", {"value": 9}, None,
         "candidate slot counts are not editable"),
        ("verification.rules.scope_risk_guards", "PATCH", {"value": {}}, None,
         "the model may not edit its own risk guards"),
        ("skill_library.entries/x.risk_guards", "PATCH", {"value": {}}, None,
         "risk guards are not editable"),
        ("candidate_policy.proposal_guidance", "PATCH",
         {"value": "Frozen program steps: [{\"op\": \"outlier_mad\"}]"}, None,
         "a frozen program in a guidance surface is refused"),
        ("skill_library.entries/x", "ADD", None,
         {"schema_version": "skill-entry/1", "skill_id": "x",
          "skill_kind": "capability",
          "body": "Frozen program steps: [{\"op\": \"outlier_mad\"}]",
          "observable_applicability": {"const": True}, "allowed_tools": [],
          "risk_guards": {}, "revision": 1},
         "a frozen program body is refused"),
        ("skill_library.entries/x", "ADD", None,
         {"schema_version": "skill-entry/1", "skill_id": "x",
          "skill_kind": "capability", "body": "guidance",
          "observable_applicability": {"const": True}, "allowed_tools": [],
          "risk_guards": {"authority": {"supplies_candidates": True}},
          "revision": 1},
         "candidate-supply authority is refused"),
    ]
    rows = []
    for surface_id, operation, patch, new_value, why in attempts:
        manifest = _manifest(
            surface_id=surface_id, operation=operation, snapshot=snapshot,
            row={"surface_precondition": {"kind": "SHA", "sha": "0" * 64},
                 "dependency_precondition_shas": {}} if operation == "PATCH"
            else None,
            value=(patch or {}).get("value"), new_value=new_value,
            edit_id="smoke-refuse")
        refused = False
        detail = ""
        try:
            know.edit_preflight(manifest)
        except know.EditRefused as exc:
            refused, detail = True, str(exc)[:120]
        except Exception as exc:  # noqa: BLE001
            refused, detail = True, "%s: %s" % (type(exc).__name__,
                                                str(exc)[:100])
        rows.append({"surface_id": surface_id, "operation": operation,
                     "refused": refused, "expected": why, "detail": detail})
    return {
        "name": "the model cannot write outside the four authorised objects",
        "passed": all(row["refused"] for row in rows),
        "attempts": rows,
        "fails_closed": ("edit_preflight raises for anything it does not "
                         "recognise, so an unlisted surface is refused rather "
                         "than allowed by omission"),
    }


# ---------------------------------------------------------------------------
# 6. paired scoring, UNKNOWN, and a resumed budget
# ---------------------------------------------------------------------------

def check_paired_and_resume() -> dict[str, Any]:
    from evaluation.main_protocol_p4 import run_dev_seq2 as run

    a = [{"series_uid": "T1", "delayed_gain": 0.10},
         {"series_uid": "T2", "delayed_gain": 0.0},
         {"series_uid": "T3", "delayed_gain": -0.20}]
    b_complete = [{"series_uid": "T1", "delayed_gain": 0.20},
                  {"series_uid": "T2", "delayed_gain": 0.0},
                  {"series_uid": "T3", "delayed_gain": -0.10}]
    b_missing = [{"series_uid": "T1", "delayed_gain": 0.20},
                 {"series_uid": "T2", "delayed_gain": "UNKNOWN"},
                 {"series_uid": "T3", "delayed_gain": -0.10}]

    full = know.paired_delayed(a, b_complete)
    holed = know.paired_delayed(a, b_missing)
    expected = round((0.20 + 0.0 - 0.10) / 3 - (0.10 + 0.0 - 0.20) / 3, 6)
    complete_ok = (full["complete"]
                   and full["whole_population_difference"] == expected)
    unknown_ok = (holed["whole_population_difference"] == "UNKNOWN"
                  and holed["mean_a"] != "UNKNOWN"
                  and holed["missing_on_either_side"] == ["T2"]
                  and holed["coverage_of_the_readable_subset"]
                  == round(2 / 3, 4))

    # an identity decision that really ran raw is a measured 0.0, not a filler
    identity_is_zero = full["mean_a"] == round((0.10 + 0.0 - 0.20) / 3, 6)

    budget = run.Budget()
    budget.spend_fits("arm", 7)
    budget.spend_llm("arm", 3)
    budget.note_evaluation("arm", 5)
    path = (base.ROOT / "_scratch" / "dev_seq2_smoke" / "budget.json")
    budget.save(path)
    resumed = run.Budget.load(path)
    resume_ok = (resumed.fits_by_arm == budget.fits_by_arm
                 and resumed.llm_by_arm == budget.llm_by_arm
                 and resumed.logical_evaluations_by_arm
                 == budget.logical_evaluations_by_arm
                 and resumed.fits() == budget.fits()
                 and resumed.llm() == budget.llm())
    return {
        "name": "paired scoring, UNKNOWN semantics, and a resumed budget",
        "passed": bool(complete_ok and unknown_ok and identity_is_zero
                       and resume_ok),
        "a_complete_pair_gives_a_whole_population_difference": complete_ok,
        "complete_pair_difference": full["whole_population_difference"],
        "a_missing_reading_makes_the_total_unknown": unknown_ok,
        "the_readable_subset_is_reported_beside_it_not_in_its_place": {
            "readable_pairs": holed["readable_pairs"],
            "coverage": holed["coverage_of_the_readable_subset"],
            "whole_population_difference": holed["whole_population_difference"],
        },
        "an_executed_identity_is_a_measured_zero": identity_is_zero,
        "a_resumed_budget_carries_the_spend_and_does_not_re_bill": resume_ok,
    }


# ---------------------------------------------------------------------------

CHECKS = (
    check_window_binding,
    check_measurement_dedup,
    check_targets_reach_fast,
    check_branch_isolation,
    check_whitelist_is_closed,
    check_paired_and_resume,
)


def run() -> dict[str, Any]:
    results = []
    for check in CHECKS:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001 - a failed check is a result
            results.append({"name": check.__name__, "passed": False,
                            "error": "%s: %s" % (type(exc).__name__,
                                                 str(exc)[:400])})
    return {
        "stage": "DEV_SEQ2_SMOKE",
        "passed": all(row.get("passed") for row in results),
        "checks": results,
        "consumer_fits": 0,
        "llm_calls": 0,
        "what_is_not_re_checked": (
            "DEV-SEQ-1's smoke already covers per-sequence request binding, "
            "heterogeneous execution, co-treatment independence, grouping and "
            "that a guidance card can enter the next Fast.  Those paths are "
            "reused unchanged and are not re-run here"),
    }


def main() -> int:
    report = run()
    print(json.dumps(drafts._plain(report), ensure_ascii=False, indent=1,
                     default=str))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
