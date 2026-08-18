"""G1: Context-resolved Negative Experience -> General Decision Guidance.

Frozen authority:

* docs/BOUNDED_SKILL_CARD_ATTRIBUTION_DESIGN_2026-08-18.md rev2.0 sections 3,
  4.2, 6.2 and 7;
* docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md rev2.2 sections 6
  and 10.

This module implements exactly one end-to-end Harness behavior and nothing
else:

    complete Episode + post_shift_support_sufficient
    -> Runtime deterministic DECISION_GAP / GENERAL attribution (zero LLM)
    -> PROPOSAL_CONTROL_GAP route
    -> at most one Slow call: PATCH candidate_policy.proposal_guidance or
       ABSTAIN
    -> E1 proposal payload really consumes the deployed surface
    -> exposed false/true Context replay
    -> only if both replay sides pass: task17..27 paired development

Three deliberate boundaries:

1. The Runtime computes the Cause.  The Slow Agent receives a complete,
   de-duplicated evidence census plus a one-entry Surface catalog and may only
   PATCH that one surface or ABSTAIN; it never reports a Cause and never
   approves its own edit.  Planner ruling 2026-08-18: the census carries no
   relation filter and no program filter, evidence is counted in
   ``distinct_task_count``, and every *active* guidance clause must
   independently meet the General repeated-evidence threshold.  rev1 of this
   module violated that with a positive-only "conflict" slot; the resulting
   fresh regression is recorded as IMPLEMENTATION_MISMATCH /
   RUNTIME_CONTRAST_SAMPLING_BIAS, not as a guidance-mechanism result.
2. The two G1 arms differ **only** in ``candidate_policy.proposal_guidance``.
   Both run with ``source_prior=None`` and both read their guidance from their
   own active Harness snapshot, so the paired readout has a single variable.
   This is not the E1-v2 A3/A5 Source-prior contrast; the arm labels are kept
   only so the frozen ``arm_order`` alternation still applies.
3. No E1-v2 row, verdict or protocol object is rewritten.  G1 uses its own
   state root and appends one report section.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.functional.task_episode_harness.e0b import (
    C1_POST_SHIFT_SUPPORT_FEATURE,
    _augment_context_with_c1_feature,
)
from evaluation.functional.task_episode_harness.e1 import (
    AVAILABLE_TASK_COUNT,
    B,
    HORIZON,
    MATERIAL_THRESHOLD,
    NF_BASE_URL,
    NF_MODEL,
    _ArmState,
    _e1_slow_call,
    _frozen_task_roster,
    _load_kdd_roster,
    _mapped_roster,
    _preflight_context_cache_from_disk,
    _run_arm,
    _skill_ids,
)
from evaluation.functional.task_episode_harness.runner import REPORT_REL
from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
from SelfEvolvingHarnessTS.contracts.program_supply import (
    route_program_supply_fault,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    EditControllerError,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import _resolve_apply_manifest

PROTOCOL_VERSION = "g1_general_proposal_guidance_v1"
G1_STATE_REL = ".g1_state"
G1_SURFACE = "candidate_policy.proposal_guidance"
G1_CAUSE = "PROPOSAL_CONTROL_GAP"
G1_REPAIR_SCOPE = "GENERAL"
G1_EDIT_ID = "g1_proposal_guidance_patch"
G1_PATTERN_ID = "g1_general_decision_proposal_control"

# The one Program mechanism under attribution: the bare single-surface program.
# Combined programs that merely contain the operator are a different mechanism
# and are reported separately as the conflict control.
G1_MECHANISM_PROGRAM = ("repair_level_shift",)
G1_CONDITION_FEATURE = C1_POST_SHIFT_SUPPORT_FEATURE

# Design rev2.0 section 4.2 item 1, counted in distinct Task Episodes.  Planner
# ruling 2026-08-18 extends it from the GENERAL write permission to every
# individual active guidance clause.
GENERAL_EVIDENCE_MIN_DISTINCT_TASKS = 2

# Plan rev2.2 section 10 item 5.  task13 is a false-Context Episode whose bare
# mechanism gain is inside the material threshold, so it is neither a harm nor
# a positive control; it stays in the attribution census as a disclosed
# non-harmful false-Context row but is not part of the replay roster.
REPLAY_TASK_IDS = (
    "e1v2_task_07",
    "e1v2_task_10",
    "e1v2_task_11",
    "e1v2_task_12",
    "e1v2_task_14",
    "e1v2_task_15",
    "e1v2_task_16",
)
FRESH_TASK_IDS = tuple(f"e1v2_task_{index:02d}" for index in range(17, 28))

BASE_ARM = "A3"
PATCHED_ARM = "A5"


# --------------------------------------------------------------- G1-A: cause


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


def _attempt_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every already-open E1-v2 Support attempt, with its C1 Context bit.

    Zero LLM, zero new Outcome: this only re-reads Episodes that E1-v2 already
    opened and derives ``post_shift_support_sufficient`` with the frozen C1
    rule from the already-cached public Context.
    """
    rows: list[dict[str, Any]] = []
    for task_row in (report.get("e1_v2") or {}).get("rows") or []:
        context = _augment_context_with_c1_feature(
            task_row.get("public_context") or {}
        )
        condition = bool(
            (context.get("task_fast_features") or {}).get(
                G1_CONDITION_FEATURE, False
            )
        )
        signature = dict(context.get("task_signature") or {})
        for arm in (BASE_ARM, PATCHED_ARM):
            arm_row = task_row.get(arm) or {}
            for probe in arm_row.get("probes") or []:
                gain = probe.get("support_gain")
                program = tuple(
                    str(step["op"])
                    for step in (probe.get("compiled_steps") or [])
                )
                rows.append({
                    "task_episode_id": str(task_row.get("task_episode_id")),
                    "arm": arm,
                    "attempt_index": probe.get("attempt_index"),
                    "program": list(program),
                    "is_mechanism": program == G1_MECHANISM_PROGRAM,
                    "contains_mechanism_operator": (
                        G1_MECHANISM_PROGRAM[0] in program
                    ),
                    "support_gain": gain,
                    "gain_readable": isinstance(gain, (int, float)),
                    "task_signature": signature,
                    G1_CONDITION_FEATURE: condition,
                    "support_origins": list(
                        task_row.get("support_origins") or []
                    ),
                    "arm_stop_reason": str(arm_row.get("stop_reason") or ""),
                })
    return rows


def _relation(gain: Any) -> str:
    if not isinstance(gain, (int, float)):
        return "UNREADABLE"
    if gain >= MATERIAL_THRESHOLD:
        return "POSITIVE"
    if gain < -MATERIAL_THRESHOLD:
        return "NEGATIVE"
    return "IMMATERIAL"


def _program_evidence_census(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Complete, de-duplicated evidence census.  No one-sided filter.

    Every ``canonical program x public Context condition x relation`` cell that
    exists in the already-open Episodes is emitted -- both relations of the
    same program in the same Context, and programs that do not carry the
    mechanism operator at all, because an "prefer a different family" clause is
    an active recommendation and needs its own evidence.

    ``distinct_task_count`` is the unit of evidence.  ``attempt_count`` is
    diagnostic only: A3 and A5 probe the same Task Episode over the same frozen
    Outcome cell, so attempts double count.
    """
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not row["gain_readable"] or not row["program"]:
            continue
        key = (
            tuple(row["program"]),
            bool(row[G1_CONDITION_FEATURE]),
            _relation(row["support_gain"]),
        )
        cell = cells.setdefault(
            key, {"task_ids": set(), "attempt_count": 0}
        )
        cell["task_ids"].add(row["task_episode_id"])
        cell["attempt_count"] += 1
    census: list[dict[str, Any]] = []
    for key in sorted(
        cells,
        key=lambda item: (
            G1_MECHANISM_PROGRAM[0] not in item[0],
            len(item[0]),
            item[0],
            not item[1],
            item[2],
        ),
    ):
        program, condition, relation = key
        cell = cells[key]
        census.append({
            "canonical_program": list(program),
            "contains_mechanism_operator": (
                G1_MECHANISM_PROGRAM[0] in program
            ),
            G1_CONDITION_FEATURE: condition,
            "support_relation": relation,
            "distinct_task_count": len(cell["task_ids"]),
            "distinct_task_episode_ids": sorted(cell["task_ids"]),
            "attempt_count": cell["attempt_count"],
        })
    return census


def run_g1_attribution(report: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic Runtime attribution over already-open Episodes.

    No LLM, no new Outcome, no threshold search: the Context condition, the
    material threshold and the harm decomposition are all already frozen.
    """
    c1 = report.get("c1_observation_diagnosis") or {}
    utility_readable = bool(
        c1.get("verdict") == "C1_OBSERVATION_FOUND" and c1.get("harm_real")
    )
    rows = _attempt_rows(report)
    mechanism = [row for row in rows if row["is_mechanism"]]
    instrument_valid = bool(mechanism) and all(
        row["gain_readable"] for row in mechanism
    )

    true_side = [row for row in mechanism if row[G1_CONDITION_FEATURE]]
    false_side = [row for row in mechanism if not row[G1_CONDITION_FEATURE]]
    separator_valid = bool(
        instrument_valid
        and true_side
        and false_side
        and all(_relation(row["support_gain"]) == "POSITIVE" for row in true_side)
        and all(_relation(row["support_gain"]) != "POSITIVE" for row in false_side)
    )

    harmful = [
        row for row in false_side
        if _relation(row["support_gain"]) == "NEGATIVE"
    ]
    harmful_task_ids = sorted({row["task_episode_id"] for row in harmful})
    positive_task_ids = sorted({row["task_episode_id"] for row in true_side})
    immaterial_false_task_ids = sorted(
        {
            row["task_episode_id"] for row in false_side
            if _relation(row["support_gain"]) == "IMMATERIAL"
        }
    )

    # Outcome identity is not reused: every counted Task Episode owns a
    # disjoint Support block triple.
    origins_by_task = {
        row["task_episode_id"]: tuple(row["support_origins"])
        for row in harmful
    }
    all_origins = [
        origin for origins in origins_by_task.values() for origin in origins
    ]
    outcome_identity_not_reused = len(all_origins) == len(set(all_origins))

    # Was the harmful probe avoidable?  Design rev2.0 section 3 step 3 asks for
    # a *repeated avoidable* proposal/probe, not for a particular ordinal
    # position, so the gate is: every arm run that probed the harmful mechanism
    # ended without promoting any draft, therefore the probe bought nothing and
    # was pure recoverable cost.  The ordinal position is kept as a disclosed
    # statistic only (7 of 8 harmful probes were the lead proposal; the eighth
    # followed a combined program on the same Task).
    lead_proposal_count = sum(
        1 for row in harmful if row["attempt_index"] == 0
    )
    abstention_reached = bool(harmful) and all(
        row["arm_stop_reason"] in {"AGENT_ABSTAIN", "NO_DRAFT_IN_BUDGET"}
        for row in harmful
    )

    # Was a legal alternative already inside the generable set in the false
    # Context?  Yes if any non-mechanism program was compiled and probed there.
    harmful_keys = {(row["task_episode_id"], row["arm"]) for row in harmful}
    alternatives_in_false_context = [
        row for row in rows
        if not row["is_mechanism"]
        and not row[G1_CONDITION_FEATURE]
        and row["gain_readable"]
    ]
    alternative_workflows_reachable = bool(alternatives_in_false_context)
    effective_alternative_found_in_harmful_runs = any(
        _relation(row["support_gain"]) == "POSITIVE"
        for row in alternatives_in_false_context
        if (row["task_episode_id"], row["arm"]) in harmful_keys
    )

    # Complete evidence census.  The superseded rev1 built a "conflict" slot by
    # keeping only the POSITIVE rows of mechanism-bearing alternative programs;
    # that hid the four NEGATIVE rows of the same program in the same Context
    # and let the Slow Agent draw a locally correct but globally wrong
    # conclusion.  Planner ruling 2026-08-18: first fault
    # IMPLEMENTATION_MISMATCH / RUNTIME_CONTRAST_SAMPLING_BIAS.  The census now
    # emits every cell with no relation filter and no program filter.
    evidence_census = _program_evidence_census(rows)

    # ---- bounded first-fault ladder (design rev2.0 section 3) ----
    if not (instrument_valid and utility_readable):
        cause = "NO_ACTIONABLE_EVIDENCE"
        ladder_stop = "instrument_or_utility_unreadable"
    elif not separator_valid:
        cause = "CONTEXT_GAP"
        ladder_stop = "public_context_cannot_separate_opposite_outcomes"
    elif not (alternative_workflows_reachable and abstention_reached):
        cause = "WORKFLOW_GAP"
        ladder_stop = "correct_behavior_outside_generable_set"
    elif (
        len(harmful_task_ids) >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
        and abstention_reached
        and not effective_alternative_found_in_harmful_runs
    ):
        cause = "DECISION_GAP"
        ladder_stop = "avoidable_repeated_probe_under_resolved_context"
    else:
        cause = "NO_ACTIONABLE_EVIDENCE"
        ladder_stop = "no_repeated_avoidable_decision_fault"

    # ---- GENERAL escalation gate (design rev2.0 section 4.2) ----
    general_gate = {
        "distinct_task_episode_count": len(harmful_task_ids),
        "distinct_task_episode_count_at_least_min": (
            len(harmful_task_ids) >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
        ),
        "evidence_unit": "distinct_task_count",
        "minimum_distinct_task_count": GENERAL_EVIDENCE_MIN_DISTINCT_TASKS,
        "single_program_mechanism": list(G1_MECHANISM_PROGRAM),
        "single_first_fault": cause,
        "shared_expressible_context_condition": {
            "feature": G1_CONDITION_FEATURE,
            "value": False,
        },
        "outcome_identity_not_reused": outcome_identity_not_reused,
        "opposite_context_positive_control_count": len(positive_task_ids),
        "opposite_context_positive_control_present": bool(positive_task_ids),
        "counted_after_context_resolution": separator_valid,
    }
    general_gate["pass"] = bool(
        cause == "DECISION_GAP"
        and general_gate["distinct_task_episode_count_at_least_min"]
        and general_gate["outcome_identity_not_reused"]
        and general_gate["opposite_context_positive_control_present"]
        and general_gate["counted_after_context_resolution"]
    )

    # ---- route side (design rev2.0 section 7 item 2) ----
    route_facts = {
        "expressibility_status": "PROVEN_EXPRESSIBLE",
        "expressibility_cause": None,
        "capability_skill_exists": True,
        "skill_retrieved": False,
        "constrained_proposal_succeeds": None,
    }
    route_before = route_program_supply_fault(**route_facts)
    route_after = route_program_supply_fault(
        **route_facts,
        context_resolved_decision_fault=general_gate["pass"],
    )
    route_reaches_surface = route_after == (
        G1_CAUSE, "EDITABLE_M0", (G1_SURFACE,)
    )

    actionable = bool(general_gate["pass"] and route_reaches_surface)
    return {
        "zero_llm": True,
        "zero_new_outcome": True,
        "verdict": (
            "G1_CAUSE_CONFIRMED" if actionable else "G1_CAUSE_NOT_ACTIONABLE"
        ),
        "cause": cause,
        "repair_scope": G1_REPAIR_SCOPE if actionable else "NONE",
        "ladder_stop": ladder_stop,
        "runtime_facts": {
            "instrument_valid": instrument_valid,
            "utility_readable": utility_readable,
            "utility_readable_source": "c1_observation_diagnosis.harm_real",
            "mechanism_attempt_count": len(mechanism),
            "context_condition_feature": G1_CONDITION_FEATURE,
            "separator_valid": separator_valid,
            "true_context_attempt_count": len(true_side),
            "false_context_attempt_count": len(false_side),
            "harmful_task_episode_ids": harmful_task_ids,
            "positive_control_task_episode_ids": positive_task_ids,
            "immaterial_false_context_task_episode_ids": (
                immaterial_false_task_ids
            ),
            "harmful_probe_count_total": len(harmful),
            "harmful_probe_count_by_arm": {
                arm: sum(1 for row in harmful if row["arm"] == arm)
                for arm in (BASE_ARM, PATCHED_ARM)
            },
            "harmful_probes_that_were_lead_proposal": lead_proposal_count,
            "abstention_reached_in_every_harmful_run": abstention_reached,
            "alternative_workflows_reachable_in_false_context": (
                alternative_workflows_reachable
            ),
            "effective_alternative_found_in_harmful_runs": (
                effective_alternative_found_in_harmful_runs
            ),
            "outcome_identity_not_reused": outcome_identity_not_reused,
            "cumulative_avoidable_support_harm_both_arms": float(
                sum(-float(row["support_gain"]) for row in harmful)
            ),
            "cumulative_avoidable_support_harm_by_arm": {
                arm: float(
                    sum(
                        -float(row["support_gain"]) for row in harmful
                        if row["arm"] == arm
                    )
                )
                for arm in (BASE_ARM, PATCHED_ARM)
            },
        },
        "general_gate": general_gate,
        "route": {
            "route_fields": route_facts,
            "before_repair": list(route_before),
            "after_repair": list(route_after),
            "repair_was_required": (
                route_before[1] != "EDITABLE_M0"
                and route_after[1] == "EDITABLE_M0"
            ),
            "reaches_authorized_surface": route_reaches_surface,
        },
        "evidence_census": evidence_census,
        "evidence_census_contract": {
            "unit_of_evidence": "distinct_task_count",
            "attempt_count_role": "diagnostic_only",
            "no_relation_filter": True,
            "no_program_filter": True,
            "note": (
                "A3 and A5 probe the same Task Episode and the same frozen "
                "Outcome cell, so attempt_count double counts evidence. "
                "General authority and every guidance clause are counted in "
                "distinct_task_count."
            ),
        },
        "superseded_contrast_note": (
            "rev1 of this module emitted a positive-only 'conflict' slot "
            "(g1.py, _program_evidence_census replaces it). Planner ruling "
            "2026-08-18 records that as the load-bearing implementation fault: "
            "IMPLEMENTATION_MISMATCH / RUNTIME_CONTRAST_SAMPLING_BIAS."
        ),
        "attempt_census": rows,
    }


# ------------------------------------------------------- G1-B: Slow guidance


_G1_SLOW_SYSTEM = (
    "You are the Slow Harness update stage. The Runtime has already attributed "
    "the first fault and authorized exactly one Harness surface for this call; "
    "you may not change the Cause, widen the repair scope, or approve your own "
    "edit. A deterministic compiler validates the edit and a paired replay "
    "decides whether it survives. "
    "You receive the currently deployed proposal guidance text and a complete "
    "Runtime-computed evidence census: every canonical program, every public "
    "Context condition and every relation that appears in the open Episodes, "
    "with no cell filtered out. You do not receive trajectories or utility "
    "numbers, so do not invent thresholds. "
    "Evidence is counted in distinct_task_count; attempt_count is diagnostic "
    "only and must not be used as evidence weight. "
    "Per-clause evidence rule: a single opposite-relation cell is enough to "
    "stop you from writing a global ban, but it never authorizes a new active "
    "recommendation or an exception combination. Every clause that actively "
    "tells the proposal stage to do something must independently cite a census "
    "cell with distinct_task_count >= 2; if only one Task supports it, say "
    "nothing about it. "
    "Write the full replacement text for candidate_policy.proposal_guidance. "
    "It is deployed to every Task, so it must stay conditional on the public "
    "observable features named in the contrast, must preserve the existing "
    "policy intent, and must not name a task id, a dataset, or a private "
    "field. "
    "Return JSON only: {'decision':'PATCH','new_guidance':'...'} or "
    "{'decision':'ABSTAIN','reason':'...'}."
)


def _slow_guidance_payload(
    attribution: Mapping[str, Any],
    current_guidance: str,
) -> dict[str, Any]:
    return {
        "attributed_cause": G1_CAUSE,
        "repair_scope": G1_REPAIR_SCOPE,
        "surface_catalog": [
            {
                "surface_id": G1_SURFACE,
                "target_class": "proposal_control",
                "surface_type": "text",
                "allowed_operations": ["PATCH"],
            }
        ],
        "current_guidance": current_guidance,
        "program_mechanism": list(G1_MECHANISM_PROGRAM),
        "public_context_condition_feature": G1_CONDITION_FEATURE,
        "evidence_census": attribution["evidence_census"],
        "evidence_census_contract": attribution["evidence_census_contract"],
        "active_clause_evidence_threshold": {
            "unit": "distinct_task_count",
            "minimum": GENERAL_EVIDENCE_MIN_DISTINCT_TASKS,
            "rule": (
                "one opposite-relation cell may block a global ban but may "
                "not authorize a new active recommendation or exception "
                "combination; every active clause must independently meet the "
                "General repeated-evidence threshold"
            ),
        },
        "note": (
            "Every operator in the inventory stays legal in every Context; "
            "this surface orders and conditions proposals, it does not remove "
            "capability."
        ),
    }


def _apply_guidance_patch(
    controller: EditController,
    store: SnapshotStore,
    snapshot: Any,
    new_guidance: str,
) -> Any:
    parent = store.materialize(snapshot)
    sha = controller.surface_precondition_sha(parent, G1_SURFACE)
    manifest = EditManifest(
        edit_id=G1_EDIT_ID,
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id=G1_PATTERN_ID,
        target_surface_id=G1_SURFACE,
        operation=EditOperation.PATCH,
        surface_precondition={"kind": "SHA", "sha": sha},
        dependency_precondition_shas={},
        minimal_patch={"value": new_guidance},
        new_value=None,
        observable_applicability=None,
        predicted_agent_behavior_change=("supply_effect_distinct",),
        predicted_data_effect=("candidate_supply_change",),
        automatically_selected_risk_cases=(),
        falsification_condition=("candidate_behavior_unchanged",),
        patch_id=None,
    )
    return controller.apply_to_fork(
        parent,
        _resolve_apply_manifest(manifest, snapshot),
        confirmed_cause=G1_CAUSE,
    )


def run_g1_guidance_patch(
    repo_root: Path,
    attribution: Mapping[str, Any],
    llm_counter: list[int],
) -> dict[str, Any]:
    """At most one Slow call, at most one authorized Surface, no retry."""
    h0 = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    base_guidance = str(
        dict(h0.candidate_policy).get("proposal_guidance") or ""
    )
    payload = _slow_guidance_payload(attribution, base_guidance)
    try:
        response = _e1_slow_call([
            {"role": "system", "content": _G1_SLOW_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
        llm_counter[0] += 1
    except (RuntimeError, ValueError) as exc:
        return {
            "verdict": "G1_GUIDANCE_SUPPLY_FAILED",
            "stage": "slow_call",
            "error": f"{type(exc).__name__}: {exc}",
            "slow_payload": payload,
            "base_guidance": base_guidance,
            "no_retry_attempted": True,
        }
    decision = str(response.get("decision") or "")
    new_guidance = str(response.get("new_guidance") or "").strip()
    if decision != "PATCH" or not new_guidance:
        return {
            "verdict": "G1_GUIDANCE_SUPPLY_FAILED",
            "stage": "slow_decision",
            "slow_payload": payload,
            "slow_response": response,
            "base_guidance": base_guidance,
            "no_retry_attempted": True,
        }

    store = SnapshotStore(repo_root / G1_STATE_REL / PATCHED_ARM / "snapshots")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    store.materialize(h0)
    try:
        receipt = _apply_guidance_patch(controller, store, h0, new_guidance)
    except (EditControllerError, ValueError, TypeError) as exc:
        return {
            "verdict": "G1_GUIDANCE_SUPPLY_FAILED",
            "stage": "edit_controller",
            "error": f"{type(exc).__name__}: {exc}",
            "slow_payload": payload,
            "slow_response": response,
            "base_guidance": base_guidance,
            "proposed_guidance": new_guidance,
            "no_retry_attempted": True,
        }
    patched = receipt.candidate_snapshot
    store.set_active(patched.runtime_bundle_sha)

    before = dict(h0.candidate_policy)
    after = dict(patched.snapshot.candidate_policy)
    changed_keys = sorted(
        key for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    )
    single_surface_diff = changed_keys == ["proposal_guidance"]
    skills_unchanged = _skill_ids(h0) == _skill_ids(patched.snapshot)
    return {
        "verdict": (
            "G1_GUIDANCE_PATCH_APPLIED"
            if single_surface_diff and skills_unchanged
            else "G1_GUIDANCE_SUPPLY_FAILED"
        ),
        "stage": "applied",
        "slow_payload": payload,
        "slow_response": response,
        "base_guidance": base_guidance,
        "patched_guidance": str(after.get("proposal_guidance") or ""),
        "changed_candidate_policy_keys": changed_keys,
        "single_surface_diff": single_surface_diff,
        "skill_library_unchanged": skills_unchanged,
        "receipt": {
            "edit_id": receipt.edit_id,
            "target_surface_id": receipt.target_surface_id,
            "confirmed_cause": receipt.confirmed_cause,
            "parent_harness_content_sha": receipt.parent_harness_content_sha,
            "candidate_harness_content_sha": (
                receipt.candidate_harness_content_sha
            ),
            "source_surfaces_changed": list(receipt.source_surfaces_changed),
        },
        "patched_runtime_bundle_sha": patched.runtime_bundle_sha,
        "no_retry_attempted": True,
    }


# ------------------------------------------------------------ G1-C: paired


def _g1_arm_state(repo_root: Path, arm: str, snapshot: Any) -> _ArmState:
    store = SnapshotStore(repo_root / G1_STATE_REL / arm / "snapshots")
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    return _ArmState(
        arm=arm,
        memories=[],
        episodes=[],
        store=store,
        active_snapshot=snapshot,
        active_skill_ids=_skill_ids(snapshot, local_only=True),
    )


def _mechanism_first_probe(arm_row: Mapping[str, Any]) -> dict[str, Any]:
    for probe in arm_row.get("probes") or []:
        if probe.get("attempt_index") != 0:
            continue
        program = tuple(
            str(step["op"]) for step in (probe.get("compiled_steps") or [])
        )
        return {
            "program": list(program),
            "is_mechanism": program == G1_MECHANISM_PROGRAM,
            "support_gain": probe.get("support_gain"),
        }
    return {"program": [], "is_mechanism": False, "support_gain": None}


def _arm_mechanism_stats(arm_row: Mapping[str, Any]) -> dict[str, Any]:
    mechanism_probes = 0
    other_probes = 0
    other_effective = 0
    for probe in arm_row.get("probes") or []:
        if not isinstance(probe.get("support_gain"), (int, float)):
            continue
        program = tuple(
            str(step["op"]) for step in (probe.get("compiled_steps") or [])
        )
        if program == G1_MECHANISM_PROGRAM:
            mechanism_probes += 1
        else:
            other_probes += 1
            if float(probe["support_gain"]) >= MATERIAL_THRESHOLD:
                other_effective += 1
    return {
        "mechanism_probe_count": mechanism_probes,
        "other_workflow_probe_count": other_probes,
        "other_effective_workflow_count": other_effective,
    }


def _run_paired_roster(
    *,
    repo_root: Path,
    task_ids: Sequence[str],
    base_snapshot: Any,
    patched_snapshot: Any,
    context_cache: Mapping[str, dict[str, Any]],
    values: Mapping[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: Mapping[str, Any],
    eval_uids: list[str],
    llm_counter: list[int],
    label: str,
) -> list[dict[str, Any]]:
    from evaluation.functional.task_episode_harness.e1 import _inventory_rows

    roster = {
        str(spec["task_episode_id"]): spec
        for spec in _frozen_task_roster(AVAILABLE_TASK_COUNT)
    }
    arm_states = {
        BASE_ARM: _g1_arm_state(repo_root, BASE_ARM, base_snapshot),
        PATCHED_ARM: _g1_arm_state(repo_root, PATCHED_ARM, patched_snapshot),
    }
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        spec = roster[task_id]
        context = _augment_context_with_c1_feature(context_cache[task_id])
        inventory = _inventory_rows(context)
        condition = bool(
            (context.get("task_fast_features") or {}).get(
                G1_CONDITION_FEATURE, False
            )
        )
        order = [BASE_ARM, PATCHED_ARM]
        if spec["arm_order"] == "A5_A3":
            order = list(reversed(order))
        print(f"G1_{label}_START {task_id} {G1_CONDITION_FEATURE}={condition}",
              flush=True)
        arm_rows: dict[str, Any] = {}
        for arm in order:
            arm_rows[arm] = _run_arm(
                repo_root=repo_root,
                arm_state=arm_states[arm],
                task_spec=spec,
                public_context=context,
                source_prior=None,
                inventory=inventory,
                values=values,
                mapped_roster=mapped_roster,
                config=config,
                eval_uids=eval_uids,
                llm_counter=llm_counter,
                consume_proposal_guidance=True,
            )
        row = {
            "task_episode_id": task_id,
            "support_origins": list(spec["support_origins"]),
            "delayed_origins": list(spec["delayed_origins"]),
            "arm_order": spec["arm_order"],
            G1_CONDITION_FEATURE: condition,
            "task_signature": dict(context["task_signature"]),
        }
        for arm in (BASE_ARM, PATCHED_ARM):
            arm_row = arm_rows[arm]
            row[arm] = {
                "stop_reason": arm_row["stop_reason"],
                "initial_decision": arm_row["initial"]["decision"],
                # A transient Slow protocol error also produces ABSTAIN, so the
                # audit must be able to tell a real abstention from an
                # instrument break.
                "initial_protocol_error": arm_row["initial"].get("error"),
                "initial_reason": arm_row["initial"].get("reason"),
                "proposal_guidance_consumed": (
                    arm_row["proposal_guidance_consumed"]
                ),
                "first_proposal": _mechanism_first_probe(arm_row),
                "mechanism_stats": _arm_mechanism_stats(arm_row),
                "metrics": arm_row["metrics"],
                "probes": arm_row["probes"],
                "winner": arm_row["winner"],
            }
        rows.append(row)
        print(
            f"G1_{label}_DONE {task_id} "
            f"{BASE_ARM}={row[BASE_ARM]['stop_reason']}/"
            f"lead_mech={row[BASE_ARM]['first_proposal']['is_mechanism']} "
            f"{PATCHED_ARM}={row[PATCHED_ARM]['stop_reason']}/"
            f"lead_mech={row[PATCHED_ARM]['first_proposal']['is_mechanism']}",
            flush=True,
        )
    return rows


def _side_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition: bool,
) -> dict[str, Any]:
    side = [row for row in rows if bool(row[G1_CONDITION_FEATURE]) is condition]
    out: dict[str, Any] = {
        "task_episode_ids": [row["task_episode_id"] for row in side],
        "task_count": len(side),
    }
    for arm in (BASE_ARM, PATCHED_ARM):
        out[arm] = {
            "proposal_protocol_error_count": sum(
                1 for row in side if row[arm].get("initial_protocol_error")
            ),
            "mechanism_lead_proposal_count": sum(
                1 for row in side if row[arm]["first_proposal"]["is_mechanism"]
            ),
            "mechanism_probe_count": sum(
                row[arm]["mechanism_stats"]["mechanism_probe_count"]
                for row in side
            ),
            "other_workflow_probe_count": sum(
                row[arm]["mechanism_stats"]["other_workflow_probe_count"]
                for row in side
            ),
            "other_effective_workflow_count": sum(
                row[arm]["mechanism_stats"]["other_effective_workflow_count"]
                for row in side
            ),
            # Planner ruling 2026-08-18: task_probe_cost applies the frozen
            # B+1 penalty to any Task that ends without a LOCAL_ACTIVE Skill,
            # so it is a charged cost, not a count of Support cells opened.
            "real_support_probe_count": sum(
                sum(
                    1 for probe in row[arm]["probes"]
                    if isinstance(probe.get("support_gain"), (int, float))
                )
                for row in side
            ),
            "charged_probe_cost": sum(
                int(row[arm]["metrics"]["task_probe_cost"]) for row in side
            ),
            "harmful_probe_count": sum(
                int(row[arm]["metrics"]["harmful_probe_count"]) for row in side
            ),
            "cumulative_support_harm": float(
                sum(
                    float(row[arm]["metrics"]["cumulative_support_harm"])
                    for row in side
                )
            ),
            "abstention_count": sum(
                int(row[arm]["metrics"]["abstention"]) for row in side
            ),
            "local_active_count": sum(
                int(row[arm]["metrics"]["task_local_active"]) for row in side
            ),
        }
    return out


def _replay_verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    false_side = _side_summary(rows, condition=False)
    true_side = _side_summary(rows, condition=True)
    # False side: the known-harmful bare mechanism must stop being the
    # unconditional lead proposal, and probe / harm must not increase.
    false_improved = bool(
        false_side["task_count"] > 0
        and false_side[PATCHED_ARM]["mechanism_lead_proposal_count"]
        < false_side[BASE_ARM]["mechanism_lead_proposal_count"]
        and false_side[PATCHED_ARM]["harmful_probe_count"]
        <= false_side[BASE_ARM]["harmful_probe_count"]
        and false_side[PATCHED_ARM]["cumulative_support_harm"]
        <= false_side[BASE_ARM]["cumulative_support_harm"]
    )
    # True side: proposal eligibility must survive.  Global suppression is
    # rejected the moment the patched arm stops proposing the mechanism in the
    # positive Context.
    true_preserved = bool(
        true_side["task_count"] > 0
        and true_side[PATCHED_ARM]["mechanism_probe_count"] > 0
        and true_side[PATCHED_ARM]["local_active_count"]
        >= true_side[BASE_ARM]["local_active_count"]
    )
    # Mechanical break stops the read: a Slow protocol error also lands as
    # ABSTAIN, so a broken instrument must never be scored as a behavior.
    protocol_errors = sum(
        side[arm]["proposal_protocol_error_count"]
        for side in (false_side, true_side)
        for arm in (BASE_ARM, PATCHED_ARM)
    )
    if protocol_errors:
        verdict = "G1_REPLAY_INSTRUMENT_BROKEN"
    elif false_improved and true_preserved:
        verdict = "G1_GENERAL_GUIDANCE_REPLAY_PASS"
    elif not false_improved and true_preserved:
        verdict = "G1_GUIDANCE_INERT"
    else:
        verdict = "G1_GLOBAL_SUPPRESSION_REJECTED"
    return {
        "verdict": verdict,
        # Planner ruling 2026-08-18: this replay is a wiring and behavior
        # check, not a utility validation.
        "replay_scope": "WIRING_AND_BEHAVIOR_ONLY_NOT_UTILITY_VALIDATION",
        "true_side_check_limitation": (
            "the true-side check asks whether the mechanism is still proposed. "
            "A patched guidance that actively mandates the mechanism in the "
            "true Context therefore cannot fail it by construction, so this "
            "replay proves 'no global removal of the operator' and does not "
            "prove 'no over-mandating of one configuration'. The exposed "
            "roster carries no true-Context Task on which the bare mechanism "
            "is the wrong configuration, so that control does not exist here."
        ),
        "false_context_side": false_side,
        "true_context_side": true_side,
        "false_side_improved": false_improved,
        "true_side_eligibility_preserved": true_preserved,
        "proposal_protocol_error_count": protocol_errors,
    }


# ------------------------------------------------- derived stage decomposition


def _winner_lifecycle_class(arm_row: Mapping[str, Any]) -> str:
    """Why a Task did or did not end with a LOCAL_ACTIVE Skill.

    ``ADD_COLLISION_APPLY_FAILED`` is a pre-existing E1-v2 instrument outcome
    (it occurs 13 times in the frozen E1-v2 rows): the machine skill_id is
    derived from the workflow signature but ``_existing_local_skill`` matches on
    frozen steps *including bound params*, so the same workflow re-won with
    different Context-bound params neither reuses nor ADDs.  The delayed window
    is then never opened, which is not the same event as a delayed rejection.
    It is reported, not repaired, under the frozen mechanical-break discipline.
    """
    winner = arm_row.get("winner")
    if winner is None:
        return "NO_WINNER"
    if str(winner.get("local_status")) == "LOCAL_ACTIVE":
        return "LOCAL_ACTIVE"
    if winner.get("delayed_gain") is None:
        return "ADD_COLLISION_APPLY_FAILED"
    return "DELAYED_REJECTED"


def derive_stage_decomposition(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Deterministic re-read of stored rows.  No LLM, no new Outcome.

    It separates the two things the headline metrics conflate:

    * ``real_support_probe_count`` -- Support cells actually opened, versus
      ``charged_probe_cost``, which applies the frozen ``B + 1`` penalty to any
      Task that ended without a LOCAL_ACTIVE Skill;
    * the reason each Task failed to end LOCAL_ACTIVE.
    """
    out: dict[str, Any] = {"per_task": [], "totals": {}}
    totals = {
        arm: {
            "real_support_probe_count": 0,
            "charged_probe_cost": 0,
            "local_active_count": 0,
            "add_collision_apply_failed_count": 0,
            "delayed_rejected_count": 0,
            "no_winner_count": 0,
            "proposal_stage_abstain_count": 0,
        }
        for arm in (BASE_ARM, PATCHED_ARM)
    }
    for row in rows:
        entry: dict[str, Any] = {
            "task_episode_id": row["task_episode_id"],
            G1_CONDITION_FEATURE: row[G1_CONDITION_FEATURE],
        }
        for arm in (BASE_ARM, PATCHED_ARM):
            arm_row = row[arm]
            real = sum(
                1 for probe in arm_row["probes"]
                if isinstance(probe.get("support_gain"), (int, float))
            )
            klass = _winner_lifecycle_class(arm_row)
            entry[arm] = {
                "real_support_probe_count": real,
                "charged_probe_cost": int(
                    arm_row["metrics"]["task_probe_cost"]
                ),
                "lifecycle_class": klass,
                "cumulative_support_harm": float(
                    arm_row["metrics"]["cumulative_support_harm"]
                ),
                "lead_program": arm_row["first_proposal"]["program"],
            }
            bucket = totals[arm]
            bucket["real_support_probe_count"] += real
            bucket["charged_probe_cost"] += int(
                arm_row["metrics"]["task_probe_cost"]
            )
            if klass == "LOCAL_ACTIVE":
                bucket["local_active_count"] += 1
            elif klass == "ADD_COLLISION_APPLY_FAILED":
                bucket["add_collision_apply_failed_count"] += 1
            elif klass == "DELAYED_REJECTED":
                bucket["delayed_rejected_count"] += 1
            else:
                bucket["no_winner_count"] += 1
            if real == 0:
                bucket["proposal_stage_abstain_count"] += 1
        out["per_task"].append(entry)
    out["totals"] = totals
    out["note"] = (
        "charged_probe_cost applies the frozen B+1 penalty to any Task without "
        "a LOCAL_ACTIVE Skill, so it is not a count of Support cells opened; "
        "real_support_probe_count is. ADD_COLLISION_APPLY_FAILED is a "
        "pre-existing E1-v2 instrument outcome, not a delayed rejection."
    )
    return out


# ------------------------------------------------------------------ driver


def run_g1(
    report_path: Path = REPORT_REL,
    *,
    attribution_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )

    def _finish(result: dict[str, Any]) -> dict[str, Any]:
        result["protocol_version"] = PROTOCOL_VERSION
        result["wall_seconds"] = time.perf_counter() - started
        result["boundary"] = {
            "g1_only": True,
            "e1_v3_not_started": True,
            "e2_card_patch_not_started": True,
            "sealed_confirmation_opened": False,
            "other_causes_not_implemented": True,
            "forbidden_operators_not_wired": True,
            "ordering_card_not_wired": True,
            "negative_card_not_implemented": True,
            "fault_routes_not_rewritten": True,
            "single_authorized_surface": G1_SURFACE,
        }
        if "g1_general_proposal_guidance" not in report:
            report["historical_verdict_before_g1"] = report.get("verdict")
        report["g1_general_proposal_guidance"] = result
        report["phase"] = "g1_general_proposal_guidance"
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    attribution = run_g1_attribution(report)
    if attribution["verdict"] != "G1_CAUSE_CONFIRMED" or attribution_only:
        return _finish({
            "verdict": attribution["verdict"],
            "stage_reached": "G1A_attribution",
            "attribution": attribution,
            "llm_api_call_count": 0,
        })

    # One G1 run owns its state root outright.  A pre-existing root means a
    # previous run already spent Slow calls and probes here; per the frozen
    # mechanical-break discipline this is reported, never silently re-run.
    state_root = repo_root / G1_STATE_REL
    if state_root.exists():
        return _finish({
            "verdict": "G1_STATE_CONTAMINATED",
            "stage_reached": "G1B_guidance_patch",
            "state_root": str(state_root),
            "attribution": attribution,
            "llm_api_call_count": 0,
        })

    llm_counter = [0]
    patch = run_g1_guidance_patch(repo_root, attribution, llm_counter)
    if patch["verdict"] != "G1_GUIDANCE_PATCH_APPLIED":
        return _finish({
            "verdict": patch["verdict"],
            "stage_reached": "G1B_guidance_patch",
            "attribution": attribution,
            "guidance_patch": patch,
            "llm_api_call_count": llm_counter[0],
        })

    # ---- shared development substrate ----
    roster, values, _selected = _load_kdd_roster(
        repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e31.jsonl"
    )
    mapped_roster = _mapped_roster(roster)
    eval_uids = [row["series_uid"] for row in mapped_roster if row["role"] == "eval"]
    from run_v1_kdd2018_natural_slow_update import _config

    config = dict(_config())
    context_cache = _preflight_context_cache_from_disk(
        repo_root, _frozen_task_roster(AVAILABLE_TASK_COUNT)
    )
    missing = [
        task_id for task_id in REPLAY_TASK_IDS + FRESH_TASK_IDS
        if task_id not in context_cache
    ]
    if missing:
        return _finish({
            "verdict": "G1_CONTEXT_CACHE_INCOMPLETE",
            "stage_reached": "G1C_replay",
            "missing_task_ids": missing,
            "attribution": attribution,
            "guidance_patch": patch,
            "llm_api_call_count": llm_counter[0],
        })

    base_snapshot = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    patched_store = SnapshotStore(
        repo_root / G1_STATE_REL / PATCHED_ARM / "snapshots"
    )
    patched_snapshot = compile_snapshot(
        patched_store.root / str(patch["patched_runtime_bundle_sha"]),
        verify_lock=False,
    )

    preregistration = {
        "protocol_version": PROTOCOL_VERSION,
        "single_variable": G1_SURFACE,
        "arms": {
            BASE_ARM: "base candidate_policy.proposal_guidance (h0)",
            PATCHED_ARM: "Slow-patched candidate_policy.proposal_guidance",
        },
        "source_prior_in_both_arms": None,
        "replay_task_ids": list(REPLAY_TASK_IDS),
        "fresh_task_ids": list(FRESH_TASK_IDS),
        "horizon": HORIZON,
        "B": B,
        "material_threshold": MATERIAL_THRESHOLD,
        "llm_settings": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "replay_pass_rule": (
            "false Context: patched arm leads with the bare mechanism strictly "
            "less often and never increases harmful probes or cumulative "
            "Support harm; true Context: the mechanism keeps proposal "
            "eligibility and LOCAL_ACTIVE outcomes do not regress"
        ),
        "fresh_reads_only_negative_side": (
            "every fresh Task is post_shift_support_sufficient=false, so the "
            "fresh stage measures negative-side cost and safety only"
        ),
    }

    replay_rows = _run_paired_roster(
        repo_root=repo_root,
        task_ids=REPLAY_TASK_IDS,
        base_snapshot=base_snapshot,
        patched_snapshot=patched_snapshot,
        context_cache=context_cache,
        values=values,
        mapped_roster=mapped_roster,
        config=config,
        eval_uids=eval_uids,
        llm_counter=llm_counter,
        label="REPLAY",
    )
    replay = _replay_verdict(replay_rows)
    replay["stage_decomposition"] = derive_stage_decomposition(replay_rows)
    if replay["verdict"] != "G1_GENERAL_GUIDANCE_REPLAY_PASS":
        return _finish({
            "verdict": replay["verdict"],
            "stage_reached": "G1C_exposed_replay",
            "attribution": attribution,
            "guidance_patch": patch,
            "preregistration": preregistration,
            "exposed_replay": {**replay, "rows": replay_rows},
            "fresh_paired_development": None,
            "fresh_stage_not_started_reason": (
                "exposed replay did not pass on both sides"
            ),
            "llm_api_call_count": llm_counter[0],
        })

    fresh_rows = _run_paired_roster(
        repo_root=repo_root,
        task_ids=FRESH_TASK_IDS,
        base_snapshot=base_snapshot,
        patched_snapshot=patched_snapshot,
        context_cache=context_cache,
        values=values,
        mapped_roster=mapped_roster,
        config=config,
        eval_uids=eval_uids,
        llm_counter=llm_counter,
        label="FRESH",
    )
    fresh_false = _side_summary(fresh_rows, condition=False)
    fresh_true = _side_summary(fresh_rows, condition=True)
    missed_effective_workflow = (
        fresh_false[PATCHED_ARM]["other_effective_workflow_count"]
        < fresh_false[BASE_ARM]["other_effective_workflow_count"]
    )
    fresh_regression = bool(
        missed_effective_workflow
        or fresh_false[PATCHED_ARM]["cumulative_support_harm"]
        > fresh_false[BASE_ARM]["cumulative_support_harm"]
        or fresh_false[PATCHED_ARM]["harmful_probe_count"]
        > fresh_false[BASE_ARM]["harmful_probe_count"]
    )
    return _finish({
        "verdict": (
            "G1_FRESH_REGRESSION_DEV"
            if fresh_regression
            else "G1_GENERAL_GUIDANCE_REPLAY_PASS"
        ),
        "mechanism_verdict": "G1_END_TO_END_GUIDANCE_WIRING_PASS",
        "claim_status": "INCONCLUSIVE" if fresh_regression else "PROVISIONAL",
        "stage_reached": "G1D_fresh_paired_development",
        "attribution": attribution,
        "guidance_patch": patch,
        "preregistration": preregistration,
        "exposed_replay": {**replay, "rows": replay_rows},
        "fresh_paired_development": {
            "false_context_side": fresh_false,
            "true_context_side": fresh_true,
            "all_fresh_tasks_false_context": fresh_true["task_count"] == 0,
            "missed_other_effective_workflow": missed_effective_workflow,
            "reads_negative_side_only": True,
            "not_a_full_a5_warm_start": True,
            "stage_decomposition": derive_stage_decomposition(fresh_rows),
            "rows": fresh_rows,
        },
        "llm_api_call_count": llm_counter[0],
    })


__all__ = [
    "PROTOCOL_VERSION",
    "derive_stage_decomposition",
    "REPLAY_TASK_IDS",
    "FRESH_TASK_IDS",
    "run_g1",
    "run_g1_attribution",
    "run_g1_guidance_patch",
]
