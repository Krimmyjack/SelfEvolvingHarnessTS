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


# --------------------------------------------------------------- W2 / W3


W2_STATE_REL = ".w2_state"
W3_STATE_REL = ".w3_state"
# Fresh KDD cohort: 20 of the 59 series never touched by any experiment,
# taken numerically ascending -- deterministic and outcome-blind.
W3_COHORT_TRAIN = tuple(f"T{index}" for index in range(211, 223))
W3_COHORT_EVAL = tuple(f"T{index}" for index in range(223, 231))
W3_N0 = 12
W3_MAX_N = 19


def _exposed_attempt_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every already-exposed KDD development Support attempt, both stages.

    Pools the frozen E1-v2 rows with the G1 replay and fresh rows.  All are
    exposed KDD development Episodes, and the relation of a probe is a measured
    Outcome that does not depend on which guidance caused the program to be
    proposed.  The source split is reported so selection bias in *coverage*
    stays visible.
    """
    rows = _attempt_rows(report)
    for row in rows:
        row["evidence_source"] = "e1_v2"
    g1 = report.get("g1_general_proposal_guidance") or {}
    for key in ("exposed_replay", "fresh_paired_development"):
        stage = g1.get(key) or {}
        for task_row in stage.get("rows") or []:
            condition = bool(task_row.get(G1_CONDITION_FEATURE))
            for arm in (BASE_ARM, PATCHED_ARM):
                arm_row = task_row.get(arm) or {}
                for probe in arm_row.get("probes") or []:
                    gain = probe.get("support_gain")
                    program = tuple(
                        str(step["op"])
                        for step in (probe.get("compiled_steps") or [])
                    )
                    rows.append({
                        "task_episode_id": (
                            "g1_" + key + "_" + str(task_row["task_episode_id"])
                        ),
                        "arm": arm,
                        "attempt_index": probe.get("attempt_index"),
                        "program": list(program),
                        "is_mechanism": program == G1_MECHANISM_PROGRAM,
                        "contains_mechanism_operator": (
                            G1_MECHANISM_PROGRAM[0] in program
                        ),
                        "support_gain": gain,
                        "gain_readable": isinstance(gain, (int, float)),
                        "task_signature": dict(
                            task_row.get("task_signature") or {}
                        ),
                        G1_CONDITION_FEATURE: condition,
                        "support_origins": list(
                            task_row.get("support_origins") or []
                        ),
                        "arm_stop_reason": str(arm_row.get("stop_reason") or ""),
                        "evidence_source": "g1_" + key,
                    })
    return rows


def _unsupported_exception_flags(
    guidance: str,
    census: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Detect the one known failure mode without building a semantic parser.

    A census cell that is POSITIVE in the false Context, carries the mechanism
    operator, and rests on fewer than
    ``GENERAL_EVIDENCE_MIN_DISTINCT_TASKS`` distinct Tasks is a single-Task
    exception.  If the deployed text names every operator of such a cell
    together, the guidance is very likely authorizing that exception, which the
    per-clause evidence rule forbids.  This is a lexical flag that stops the
    run for a human ruling; it is not a Gate and not a parser.
    """
    lowered = guidance.lower()
    flags: list[dict[str, Any]] = []
    for cell in census:
        if cell["support_relation"] != "POSITIVE":
            continue
        if cell[G1_CONDITION_FEATURE]:
            continue
        if not cell["contains_mechanism_operator"]:
            continue
        if cell["distinct_task_count"] >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS:
            continue
        program = [str(op) for op in cell["canonical_program"]]
        if len(program) > 1 and all(op.lower() in lowered for op in program):
            flags.append({
                "canonical_program": program,
                "distinct_task_count": cell["distinct_task_count"],
                "required_minimum": GENERAL_EVIDENCE_MIN_DISTINCT_TASKS,
                "reason": (
                    "the deployed guidance names every operator of a "
                    "single-Task false-Context POSITIVE cell together"
                ),
            })
    return flags


def run_w2_guidance_freeze(report_path: Path = REPORT_REL) -> dict[str, Any]:
    """W2: rebuild the complete census and re-freeze the General guidance.

    Exactly one Slow call, exactly one authorized Surface, no retry, no manual
    rewriting.
    """
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    rows = _exposed_attempt_rows(report)
    census = _program_evidence_census(rows)
    from collections import Counter

    source_split = Counter(row["evidence_source"] for row in rows)
    attribution = {
        "evidence_census": census,
        "evidence_census_contract": {
            "unit_of_evidence": "distinct_task_count",
            "attempt_count_role": "diagnostic_only",
            "no_relation_filter": True,
            "no_program_filter": True,
        },
    }
    base_result: dict[str, Any] = {
        "protocol_version": "w2_guidance_freeze_v1",
        "evidence_scope": "already-exposed KDD development Episodes only",
        "evidence_source_split": dict(source_split),
        "evidence_census": census,
        "general_evidence_min_distinct_tasks": (
            GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
        ),
        "zero_new_outcome": True,
    }

    h0 = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    base_guidance = str(dict(h0.candidate_policy).get("proposal_guidance") or "")
    payload = _slow_guidance_payload(attribution, base_guidance)
    llm_calls = 0
    try:
        response = _e1_slow_call([
            {"role": "system", "content": _G1_SLOW_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
        llm_calls = 1
    except (RuntimeError, ValueError) as exc:
        return {**base_result, "verdict": "W2_GUIDANCE_SUPPLY_FAILED",
                "stage": "slow_call",
                "error": type(exc).__name__ + ": " + str(exc),
                "llm_api_call_count": 0, "no_retry_attempted": True,
                "wall_seconds": time.perf_counter() - started}

    decision = str(response.get("decision") or "")
    new_guidance = str(response.get("new_guidance") or "").strip()
    if decision != "PATCH" or not new_guidance:
        return {**base_result, "verdict": "W2_GUIDANCE_SUPPLY_FAILED",
                "stage": "slow_decision", "slow_response": response,
                "llm_api_call_count": llm_calls, "no_retry_attempted": True,
                "wall_seconds": time.perf_counter() - started}

    flags = _unsupported_exception_flags(new_guidance, census)
    if flags:
        return {**base_result, "verdict": "W2_UNSUPPORTED_EXCEPTION_CLAUSE",
                "stage": "per_clause_evidence_check",
                "slow_response": response, "proposed_guidance": new_guidance,
                "unsupported_exception_flags": flags,
                "llm_api_call_count": llm_calls,
                "no_retry_attempted": True, "no_manual_rewrite": True,
                "wall_seconds": time.perf_counter() - started}

    store = SnapshotStore(repo_root / W2_STATE_REL / "snapshots")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    store.materialize(h0)
    try:
        receipt = _apply_guidance_patch(controller, store, h0, new_guidance)
    except (EditControllerError, ValueError, TypeError) as exc:
        return {**base_result, "verdict": "W2_GUIDANCE_SUPPLY_FAILED",
                "stage": "edit_controller",
                "error": type(exc).__name__ + ": " + str(exc),
                "slow_response": response, "proposed_guidance": new_guidance,
                "llm_api_call_count": llm_calls, "no_retry_attempted": True,
                "wall_seconds": time.perf_counter() - started}

    patched = receipt.candidate_snapshot
    store.set_active(patched.runtime_bundle_sha)
    before = dict(h0.candidate_policy)
    after = dict(patched.snapshot.candidate_policy)
    changed = sorted(
        key for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    )
    single_surface = bool(
        changed == ["proposal_guidance"]
        and list(receipt.source_surfaces_changed) == [G1_SURFACE]
        and _skill_ids(h0) == _skill_ids(patched.snapshot)
        and dict(h0.retrieval) == dict(patched.snapshot.retrieval)
        and dict(h0.verification) == dict(patched.snapshot.verification)
        and h0.instruction == patched.snapshot.instruction
    )
    return {
        **base_result,
        "verdict": ("W2_GUIDANCE_FROZEN" if single_surface
                    else "W2_MULTI_SURFACE_MODIFICATION"),
        "stage": "applied",
        "slow_payload": payload,
        "slow_response": response,
        "base_guidance": base_guidance,
        "frozen_guidance": str(after.get("proposal_guidance") or ""),
        "changed_candidate_policy_keys": changed,
        "single_surface_diff": single_surface,
        "receipt": {
            "edit_id": receipt.edit_id,
            "target_surface_id": receipt.target_surface_id,
            "confirmed_cause": receipt.confirmed_cause,
            "source_surfaces_changed": list(receipt.source_surfaces_changed),
        },
        "frozen_runtime_bundle_sha": patched.runtime_bundle_sha,
        "unsupported_exception_flags": [],
        "llm_api_call_count": llm_calls,
        "no_retry_attempted": True,
        "wall_seconds": time.perf_counter() - started,
    }


def _load_w3_cohort(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the fresh-cohort roster straight from the KDD series cache.

    No manifest file and no registry: the cohort rule is frozen in
    ``W3_COHORT_TRAIN`` / ``W3_COHORT_EVAL`` -- 20 of the 59 KDD series that no
    experiment has ever referenced, taken numerically ascending.
    """
    import numpy as np

    cache = np.load(
        repo_root / "data/kdd2018/series_cache.npz", allow_pickle=True
    )
    names = [str(name) for name in cache["names"]]
    cohort = list(W3_COHORT_TRAIN) + list(W3_COHORT_EVAL)
    missing = [uid for uid in cohort if uid not in names]
    if missing:
        raise ValueError("fresh cohort series absent from cache: %r" % missing)
    values = {
        uid: np.asarray(cache["values"][names.index(uid)], dtype=np.float64)
        for uid in cohort
    }
    roster = (
        [{"series_uid": uid, "role": "train"} for uid in W3_COHORT_TRAIN]
        + [{"series_uid": uid, "role": "eval"} for uid in W3_COHORT_EVAL]
    )
    return roster, values


def _w3_context_for(
    repo_root: Path,
    state_rel: str,
    task_id: str,
    cutoff: int,
    values: Mapping[str, Any],
    train_uids: Sequence[str],
) -> dict[str, Any]:
    """Public Context for one Task of an explicit cohort, cached on disk."""
    from evaluation.functional.task_episode_harness.public_context import (
        build_task_public_context,
    )

    cache_root = repo_root / state_rel / "contexts"
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / (task_id + ".json")
    if path.is_file():
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if int(row.get("observation_cutoff") or -1) == cutoff:
                return row
        except (OSError, json.JSONDecodeError):
            pass
    context = _augment_context_with_c1_feature(
        build_task_public_context(
            values, list(train_uids), observation_cutoff=cutoff
        )
    )
    path.write_text(
        json.dumps(context, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return context


def _w3_context(
    repo_root: Path,
    task_id: str,
    cutoff: int,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Public Context for one fresh-cohort Task, cached on disk.

    Outcome-blind by construction: ``build_task_public_context`` only slices
    ``values[uid][:cutoff]``.
    """
    from evaluation.functional.task_episode_harness.public_context import (
        build_task_public_context,
    )

    cache_root = repo_root / W3_STATE_REL / "contexts"
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / (task_id + ".json")
    if path.is_file():
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if int(row.get("observation_cutoff") or -1) == cutoff:
                return row
        except (OSError, json.JSONDecodeError):
            pass
    context = build_task_public_context(
        values, list(W3_COHORT_TRAIN), observation_cutoff=cutoff
    )
    context = _augment_context_with_c1_feature(context)
    path.write_text(
        json.dumps(context, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return context


def run_w3(
    report_path: Path = REPORT_REL,
    *,
    frozen_guidance: str | None = None,
) -> dict[str, Any]:
    """W3: paired development on the fresh KDD cohort.

    Single variable: ``candidate_policy.proposal_guidance``.  Base reads the h0
    text, Patched reads the W2-frozen text; both arms read it from their own
    active Harness snapshot, both run with ``source_prior=None``, identical
    probe budget, inventory, Consumer, Metric and LLM settings, and fully
    isolated Experience / Skill stores.
    """
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    w2 = report.get("w2_guidance_freeze") or {}
    guidance = frozen_guidance or str(w2.get("frozen_guidance") or "")
    if w2.get("verdict") != "W2_GUIDANCE_FROZEN" or not guidance:
        return {"verdict": "W3_GUIDANCE_NOT_FROZEN",
                "w2_verdict": w2.get("verdict"),
                "llm_api_call_count": 0,
                "wall_seconds": time.perf_counter() - started}

    state_root = repo_root / W3_STATE_REL
    if (state_root / BASE_ARM).exists() or (state_root / PATCHED_ARM).exists():
        return {"verdict": "W3_STATE_CONTAMINATED",
                "state_root": str(state_root),
                "note": ("Outcomes on this cohort may already be open; a "
                         "re-run after a guidance change is forbidden"),
                "llm_api_call_count": 0,
                "wall_seconds": time.perf_counter() - started}

    from evaluation.functional.task_episode_harness.e1 import _inventory_rows
    from run_v1_kdd2018_natural_slow_update import _config

    roster, values = _load_w3_cohort(repo_root)
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    config = dict(_config())
    specs = {
        str(spec["task_episode_id"]): spec
        for spec in _frozen_task_roster(AVAILABLE_TASK_COUNT)
    }

    base_snapshot = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    patched_store = SnapshotStore(repo_root / W2_STATE_REL / "snapshots")
    patched_snapshot = compile_snapshot(
        patched_store.root / str(w2["frozen_runtime_bundle_sha"]),
        verify_lock=False,
    )
    arm_states = {
        BASE_ARM: _g1_arm_state(repo_root, BASE_ARM, base_snapshot),
        PATCHED_ARM: _g1_arm_state(repo_root, PATCHED_ARM, patched_snapshot),
    }
    # The stores live under .g1_state by default; W3 owns its own root.
    for arm, snapshot in ((BASE_ARM, base_snapshot),
                          (PATCHED_ARM, patched_snapshot)):
        store = SnapshotStore(repo_root / W3_STATE_REL / arm / "snapshots")
        store.materialize(snapshot)
        store.set_active(snapshot.runtime_bundle_sha)
        arm_states[arm] = _ArmState(
            arm=arm, memories=[], episodes=[], store=store,
            active_snapshot=snapshot,
            active_skill_ids=_skill_ids(snapshot, local_only=True),
        )

    preregistration = {
        "protocol_version": "w3_fresh_cohort_paired_development_v1",
        "single_variable": G1_SURFACE,
        "cohort_rule": (
            "20 KDD series never referenced by any experiment, numerically "
            "ascending, first 12 train / next 8 eval"
        ),
        "train_series": list(W3_COHORT_TRAIN),
        "eval_series": list(W3_COHORT_EVAL),
        "N0": W3_N0,
        "max_N": W3_MAX_N,
        "horizon": HORIZON,
        "B": B,
        "material_threshold": MATERIAL_THRESHOLD,
        "llm_settings": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "arms": {
            BASE_ARM: "h0 base candidate_policy.proposal_guidance",
            PATCHED_ARM: "W2-frozen candidate_policy.proposal_guidance",
        },
        "source_prior_in_both_arms": None,
        "base_guidance": str(w2.get("base_guidance") or ""),
        "frozen_guidance": guidance,
        "primary_readouts": [
            "real_support_probe_count", "harmful_probe_count",
            "cumulative_support_harm", "true/false Context stratification",
            "missed other effective Workflow",
        ],
        "auxiliary_readouts": ["charged_probe_cost"],
        "one_extension_only": True,
        "tasks_20_to_27_outcome_unopened": True,
    }

    llm_counter = [0]
    rows: list[dict[str, Any]] = []

    def run_block(task_ids: Sequence[str], label: str) -> None:
        for task_id in task_ids:
            spec = specs[task_id]
            context = _w3_context(
                repo_root, task_id, int(spec["support_origins"][0]), values
            )
            inventory = _inventory_rows(context)
            condition = bool(
                (context.get("task_fast_features") or {}).get(
                    G1_CONDITION_FEATURE, False
                )
            )
            order = [BASE_ARM, PATCHED_ARM]
            if spec["arm_order"] == "A5_A3":
                order = list(reversed(order))
            print("W3_%s_START %s %s=%s" % (
                label, task_id, G1_CONDITION_FEATURE, condition), flush=True)
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
            row: dict[str, Any] = {
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
                    "initial_protocol_error": arm_row["initial"].get("error"),
                    "proposal_guidance_consumed": (
                        arm_row["proposal_guidance_consumed"]
                    ),
                    "first_proposal": _mechanism_first_probe(arm_row),
                    "mechanism_stats": _arm_mechanism_stats(arm_row),
                    "metrics": arm_row["metrics"],
                    "probes": arm_row["probes"],
                    "winner": arm_row["winner"],
                    "lifecycle": arm_row["lifecycle"],
                    "target_memories_after": arm_row["target_memories_after"],
                    "active_local_skill_ids_after": (
                        arm_row["active_local_skill_ids_after"]
                    ),
                }
            rows.append(row)
            print("W3_%s_DONE %s %s=%s/lead_mech=%s %s=%s/lead_mech=%s" % (
                label, task_id,
                BASE_ARM, row[BASE_ARM]["stop_reason"],
                row[BASE_ARM]["first_proposal"]["is_mechanism"],
                PATCHED_ARM, row[PATCHED_ARM]["stop_reason"],
                row[PATCHED_ARM]["first_proposal"]["is_mechanism"]), flush=True)

    n0_ids = ["e1v2_task_%02d" % index for index in range(1, W3_N0 + 1)]
    run_block(n0_ids, "N0")

    from evaluation.functional.task_episode_harness.e1 import (
        _paired_summary, _sample_plan,
    )

    summary = _paired_summary(rows)
    plan = _sample_plan(summary, available_task_count=W3_MAX_N)
    extension_used = 0
    n_final = min(int(plan["N_final"]), W3_MAX_N)
    if n_final > len(rows):
        extension_ids = [
            "e1v2_task_%02d" % index for index in range(len(rows) + 1, n_final + 1)
        ]
        extension_used = len(extension_ids)
        run_block(extension_ids, "EXT")
        summary = _paired_summary(rows)

    false_side = _side_summary(rows, condition=False)
    true_side = _side_summary(rows, condition=True)
    decomposition = derive_stage_decomposition(rows)
    missed = (
        false_side[PATCHED_ARM]["other_effective_workflow_count"]
        < false_side[BASE_ARM]["other_effective_workflow_count"]
    )
    result = {
        "protocol_version": "w3_fresh_cohort_paired_development_v1",
        "verdict": "W3_PAIRED_DEVELOPMENT_COMPLETE",
        "preregistration": preregistration,
        "sample_plan": {**plan, "capped_N_final": n_final,
                        "extension_tasks_run": extension_used,
                        "one_extension_only": True},
        "paired_summary": summary,
        "false_context_side": false_side,
        "true_context_side": true_side,
        "stage_decomposition": decomposition,
        "missed_other_effective_workflow": missed,
        "rows": rows,
        "llm_api_call_count": llm_counter[0],
        "boundary": {
            "tasks_20_to_27_outcome_unopened": len(rows) < 20,
            "guidance_frozen_before_outcome": True,
            "no_rerun_after_guidance_change": True,
            "e2_not_started": True,
            "sealed_confirmation_opened": False,
            "weather_not_started": True,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    return result


# ------------------------------------------- validity repairs (2026-08-18)


# The h0 default proposal policy.  Evidence produced while consuming exactly
# this text carries no evolved General clause, so it is UNGUIDED for the
# purpose of authorizing a new active clause.
PROVENANCE_UNGUIDED = "UNGUIDED"
PROVENANCE_CONDITIONED = "GUIDANCE_CONDITIONED"


def eval_substrate_preflight(
    values: Mapping[str, Any],
    eval_uids: Sequence[str],
    specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fix 1: can the Judge even run on this roster?  Zero Outcome.

    W0 asked whether a Task's public Context is expressible, which is a
    property of the TRAIN series.  It never asked whether the Task is
    measurable, which is a property of the EVAL series: ``_evaluate`` refuses
    an evaluation context whose 192-point window collapses to the scale floor.
    Every stage of this project ran without that check, so E1-v2 task_01/02/06
    and W3 task_06/07 recorded instrument failure as Agent behaviour.

    This reads only the public prefix window ``raw[origin - 192:origin]``; the
    truth window is never touched, so it opens no Outcome.
    """
    import numpy as np
    import run_e2_autonomous_natural_workflow_generation as v6

    context_length = int(v6.CONTEXT_LENGTH)
    origins = sorted({
        int(origin)
        for spec in specs
        for role in ("support", "delayed")
        for origin in spec[f"{role}_origins"]
    })
    per_series: dict[str, Any] = {}
    for uid in eval_uids:
        raw = np.asarray(values[str(uid)], dtype=np.float64)
        hits: list[int] = []
        for origin in origins:
            if origin - context_length < 0 or origin > raw.size:
                continue
            prepared = v6._linear_integrity(raw[origin - context_length:origin])
            _center, _scale, method = v6._center_scale(np, prepared)
            if method == "scale_floor_fallback":
                hits.append(origin)
        per_series[str(uid)] = {
            "floor_hit_origin_count": len(hits),
            "floor_hit_origins": hits[:12],
            "clean": not hits,
        }
    dirty = sorted(uid for uid, row in per_series.items() if not row["clean"])
    return {
        "check": "eval_substrate_scale_floor",
        "zero_new_outcome": True,
        "context_length": context_length,
        "origin_count": len(origins),
        "eval_series": [str(uid) for uid in eval_uids],
        "per_series": per_series,
        "floor_hitting_series": dirty,
        "pass": not dirty,
        "note": (
            "a floor-hitting eval series must be rejected before any Outcome "
            "is opened; this confirms the Judge can execute, it does not "
            "change the Judge"
        ),
    }


def _guidance_provenance(consumed: Any, base_guidance: str) -> str:
    """Fix 3: was this attempt's proposal conditioned on an evolved clause?"""
    if consumed is None:
        return PROVENANCE_UNGUIDED
    text = str(consumed).strip()
    if not text or text == str(base_guidance).strip():
        return PROVENANCE_UNGUIDED
    return PROVENANCE_CONDITIONED


def _provenance_attempt_rows(
    report: Mapping[str, Any],
    base_guidance: str,
) -> list[dict[str, Any]]:
    """Exposed KDD attempts, each tagged UNGUIDED or GUIDANCE_CONDITIONED."""
    rows = _exposed_attempt_rows(report)
    by_arm: dict[tuple[str, str], str] = {}
    g1 = report.get("g1_general_proposal_guidance") or {}
    for key in ("exposed_replay", "fresh_paired_development"):
        for task_row in (g1.get(key) or {}).get("rows") or []:
            for arm in (BASE_ARM, PATCHED_ARM):
                arm_row = task_row.get(arm) or {}
                by_arm[("g1_" + key + "_" + str(task_row["task_episode_id"]), arm)] = (
                    _guidance_provenance(
                        arm_row.get("proposal_guidance_consumed"), base_guidance
                    )
                )
    for row in rows:
        if row["evidence_source"] == "e1_v2":
            # E1-v2 predates consume_proposal_guidance entirely.
            row["guidance_provenance"] = PROVENANCE_UNGUIDED
        else:
            row["guidance_provenance"] = by_arm.get(
                (row["task_episode_id"], row["arm"]), PROVENANCE_CONDITIONED
            )
    return rows


def reaudit_frozen_guidance(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Mechanical provenance re-audit of the frozen W2 text.  No Slow call.

    Every *active* clause must hold at least
    ``GENERAL_EVIDENCE_MIN_DISTINCT_TASKS`` distinct UNGUIDED Tasks.
    GUIDANCE_CONDITIONED evidence may confirm, contradict or withdraw an
    existing clause but can never authorize one, otherwise a clause proves
    itself through the proposals it caused.

    The frozen text names exactly one operator and one Context feature, so the
    clause set is enumerated lexically -- no semantic parser and no new Gate.
    """
    w2 = report.get("w2_guidance_freeze") or {}
    guidance = str(w2.get("frozen_guidance") or "")
    base_guidance = str(w2.get("base_guidance") or "")
    rows = _provenance_attempt_rows(report, base_guidance)
    unguided = [
        row for row in rows
        if row["guidance_provenance"] == PROVENANCE_UNGUIDED
    ]
    conditioned = [
        row for row in rows
        if row["guidance_provenance"] == PROVENANCE_CONDITIONED
    ]
    unguided_census = _program_evidence_census(unguided)
    full_census = _program_evidence_census(rows)

    def support(program: Sequence[str], condition: bool, relation: str,
                census: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        for cell in census:
            if (
                list(cell["canonical_program"]) == list(program)
                and bool(cell[G1_CONDITION_FEATURE]) is condition
                and cell["support_relation"] == relation
            ):
                return {
                    "distinct_task_count": cell["distinct_task_count"],
                    "distinct_task_episode_ids": cell["distinct_task_episode_ids"],
                }
        return {"distinct_task_count": 0, "distinct_task_episode_ids": []}

    lowered = guidance.lower()
    mechanism = G1_MECHANISM_PROGRAM[0]
    clauses = []
    if mechanism in lowered and "true" in lowered:
        cell = support([mechanism], True, "POSITIVE", unguided_census)
        clauses.append({
            "clause": (
                "when %s is true, prioritize %s"
                % (G1_CONDITION_FEATURE, mechanism)
            ),
            "clause_kind": "ACTIVE_RECOMMENDATION",
            "required_unguided_distinct_tasks": (
                GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            ),
            "unguided_support": cell,
            "satisfied": (
                cell["distinct_task_count"]
                >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            ),
        })
    if mechanism in lowered and "false" in lowered:
        cell = support([mechanism], False, "NEGATIVE", unguided_census)
        clauses.append({
            "clause": (
                "when %s is false, do not make %s the default"
                % (G1_CONDITION_FEATURE, mechanism)
            ),
            "clause_kind": "ACTIVE_DEPRIORITIZATION",
            "required_unguided_distinct_tasks": (
                GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            ),
            "unguided_support": cell,
            "satisfied": (
                cell["distinct_task_count"]
                >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            ),
        })
    if "prohibition" in lowered or "blanket" in lowered:
        cell = support([mechanism], True, "POSITIVE", unguided_census)
        clauses.append({
            "clause": "this is not a blanket prohibition",
            "clause_kind": "NON_BAN_RESERVATION",
            "required_unguided_distinct_tasks": 1,
            "unguided_support": cell,
            "satisfied": cell["distinct_task_count"] >= 1,
            "note": (
                "a single opposite-relation cell may block a global ban; this "
                "clause authorizes nothing, so it needs one Task, not two"
            ),
        })
    # Any pairing or exception clause naming a second operator must clear the
    # same bar on UNGUIDED evidence alone.
    for cell in full_census:
        program = [str(op) for op in cell["canonical_program"]]
        if len(program) < 2 or not cell["contains_mechanism_operator"]:
            continue
        if cell["support_relation"] != "POSITIVE" or cell[G1_CONDITION_FEATURE]:
            continue
        if not all(op.lower() in lowered for op in program):
            continue
        unguided_cell = support(program, False, "POSITIVE", unguided_census)
        clauses.append({
            "clause": "exception combination " + "+".join(program),
            "clause_kind": "ACTIVE_EXCEPTION_COMBINATION",
            "required_unguided_distinct_tasks": (
                GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            ),
            "pooled_support": cell["distinct_task_count"],
            "unguided_support": unguided_cell,
            "satisfied": (
                unguided_cell["distinct_task_count"]
                >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            ),
        })

    all_satisfied = bool(clauses) and all(row["satisfied"] for row in clauses)
    return {
        "check": "general_evidence_provenance_reaudit",
        "zero_llm": True,
        "zero_new_outcome": True,
        "no_slow_call": True,
        "frozen_guidance": guidance,
        "base_guidance": base_guidance,
        "provenance_split": {
            PROVENANCE_UNGUIDED: len(unguided),
            PROVENANCE_CONDITIONED: len(conditioned),
        },
        "unguided_census": unguided_census,
        "pooled_census": full_census,
        "clauses": clauses,
        "verdict": (
            "W2_GUIDANCE_PROVENANCE_CONFIRMED" if all_satisfied
            else "W2_GUIDANCE_PROVENANCE_UNSUPPORTED"
        ),
        "authorization_rule": (
            "a new active clause may only be authorized by UNGUIDED distinct "
            "Tasks; GUIDANCE_CONDITIONED evidence may confirm, contradict or "
            "withdraw a clause but never authorize one"
        ),
    }


# ------------------------------------- A5 vs A3 on a fresh natural cohort


A5A3_STATE_REL = ".a5a3_natural_state"
A5A3_COHORT_TRAIN = (
    "T233", "T234", "T235", "T236", "T239", "T240",
    "T241", "T244", "T246", "T247", "T254", "T256",
)
A5A3_COHORT_EVAL = (
    "T257", "T259", "T260", "T261", "T262", "T264", "T265", "T266",
)
A5A3_N0 = 12
A5A3_MAX_N = 19


def _a5a3_cohort(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    cache = np.load(
        repo_root / "data/kdd2018/series_cache.npz", allow_pickle=True
    )
    names = [str(name) for name in cache["names"]]
    cohort = list(A5A3_COHORT_TRAIN) + list(A5A3_COHORT_EVAL)
    values = {
        uid: np.asarray(cache["values"][names.index(uid)], dtype=np.float64)
        for uid in cohort
    }
    roster = (
        [{"series_uid": uid, "role": "train"} for uid in A5A3_COHORT_TRAIN]
        + [{"series_uid": uid, "role": "eval"} for uid in A5A3_COHORT_EVAL]
    )
    return roster, values


def run_a5a3_natural(report_path: Path = REPORT_REL) -> dict[str, Any]:
    """A5 vs A3 paired development on the remaining natural KDD cohort.

    Estimand (corrected before any Outcome was opened): under the natural
    Target Task distribution, is the accumulated A5 package faster and safer
    than a cold A3 start?  The two-sided G1 gate is deliberately dropped -- it
    was a sub-experiment condition, not a requirement of this question.

    A5 is offered the whole warm-start package (Source Card + Source evidence,
    and the provenance-confirmed General guidance).  The Runtime Scope matcher
    decides whether it may enter; the run records that decision per Task rather
    than assuming it.  A3 starts cold with the h0 base guidance.

    Non-negotiables kept: eval substrate preflight before any Outcome, fully
    isolated per-arm Experience and Skill stores, identical probe budget,
    inventory, Consumer, Metric and LLM settings, unexposed Outcomes, and
    instrument failure that stops rather than masquerading as behaviour.
    """
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = json.loads(report_path.read_text(encoding="utf-8"))

    w2 = report.get("w2_guidance_freeze") or {}
    repairs = report.get("v1_validity_repairs") or {}
    provenance = (repairs.get("fix_3_evidence_provenance") or {}).get("reaudit") or {}
    guidance = str(w2.get("frozen_guidance") or "")
    if provenance.get("verdict") != "W2_GUIDANCE_PROVENANCE_CONFIRMED" or not guidance:
        return {"verdict": "A5A3_GUIDANCE_NOT_CONFIRMED",
                "provenance_verdict": provenance.get("verdict"),
                "llm_api_call_count": 0,
                "wall_seconds": time.perf_counter() - started}

    state_root = repo_root / A5A3_STATE_REL
    if (state_root / BASE_ARM).exists() or (state_root / PATCHED_ARM).exists():
        return {"verdict": "A5A3_STATE_CONTAMINATED",
                "state_root": str(state_root),
                "llm_api_call_count": 0,
                "wall_seconds": time.perf_counter() - started}

    from evaluation.functional.task_episode_harness.e1 import (
        _inventory_rows, _source_bundle_from_report, _source_card_from_report,
        _runtime_source_applicability, _source_prior_for_task,
        _paired_summary, _sample_plan,
    )
    from run_v1_kdd2018_natural_slow_update import _config

    roster, values = _a5a3_cohort(repo_root)
    mapped_roster = _mapped_roster(roster)
    eval_uids = [r["series_uid"] for r in mapped_roster if r["role"] == "eval"]
    config = dict(_config())
    specs_all = _frozen_task_roster(AVAILABLE_TASK_COUNT)[:A5A3_MAX_N]
    specs = {str(spec["task_episode_id"]): spec for spec in specs_all}

    # Non-negotiable 1: the Judge must be able to run before any Outcome opens.
    preflight = eval_substrate_preflight(values, eval_uids, specs_all)
    if not preflight["pass"]:
        return {"verdict": "A5A3_EVAL_SUBSTRATE_INVALID",
                "eval_substrate_preflight": preflight,
                "llm_api_call_count": 0,
                "wall_seconds": time.perf_counter() - started}

    source_card = _source_card_from_report(report)
    source_bundle = _source_bundle_from_report(report)
    a5_source_prior = {
        "source_card": source_card, "source_evidence": source_bundle,
    }

    base_snapshot = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    patched_store = SnapshotStore(repo_root / W2_STATE_REL / "snapshots")
    patched_snapshot = compile_snapshot(
        patched_store.root / str(w2["frozen_runtime_bundle_sha"]),
        verify_lock=False,
    )
    arm_states: dict[str, _ArmState] = {}
    for arm, snapshot in ((BASE_ARM, base_snapshot),
                          (PATCHED_ARM, patched_snapshot)):
        store = SnapshotStore(state_root / arm / "snapshots")
        store.materialize(snapshot)
        store.set_active(snapshot.runtime_bundle_sha)
        arm_states[arm] = _ArmState(
            arm=arm, memories=[], episodes=[], store=store,
            active_snapshot=snapshot,
            active_skill_ids=_skill_ids(snapshot, local_only=True),
        )

    preregistration = {
        "protocol_version": "a5a3_natural_cohort_development_v1",
        "estimand": (
            "under the natural Target Task distribution, is the accumulated A5 "
            "package faster and safer than a cold A3 start"
        ),
        "dropped_claim": (
            "true-side preservation cannot be separately confirmed on this "
            "cohort; the two-sided G1 gate was a sub-experiment condition and "
            "is deliberately not applied here"
        ),
        "retained_claims": [
            "overall paired adaptation speed", "Support harm safety",
            "delayed utility",
        ],
        "cohort_train": list(A5A3_COHORT_TRAIN),
        "cohort_eval": list(A5A3_COHORT_EVAL),
        "eval_substrate_preflight": preflight,
        "arms": {
            BASE_ARM: "cold start: no Source prior, h0 base proposal guidance",
            PATCHED_ARM: (
                "warm start: Source Card + Source evidence offered through the "
                "Runtime Scope matcher, plus the provenance-confirmed General "
                "guidance"
            ),
        },
        "source_card_applicability": _runtime_source_applicability(source_card),
        "N0": A5A3_N0, "max_N": A5A3_MAX_N,
        "horizon": HORIZON, "B": B,
        "material_threshold": MATERIAL_THRESHOLD,
        "llm_settings": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "base_guidance": str(w2.get("base_guidance") or ""),
        "frozen_guidance": guidance,
        "primary_readouts": [
            "real_support_probe_count", "harmful_probe_count",
            "cumulative_support_harm", "task_local_active",
            "delayed utility",
        ],
        "auxiliary_readouts": ["charged_probe_cost",
                               "post_shift_support_sufficient stratification"],
    }

    llm_counter = [0]
    rows: list[dict[str, Any]] = []

    def run_block(task_ids: Sequence[str], label: str) -> None:
        for task_id in task_ids:
            spec = specs[task_id]
            context = _w3_context_for(
                repo_root, A5A3_STATE_REL, task_id,
                int(spec["support_origins"][0]), values, A5A3_COHORT_TRAIN
            )
            inventory = _inventory_rows(context)
            condition = bool(
                (context.get("task_fast_features") or {}).get(
                    G1_CONDITION_FEATURE, False
                )
            )
            matched_prior = _source_prior_for_task(a5_source_prior, context)
            order = [(BASE_ARM, None), (PATCHED_ARM, matched_prior)]
            if spec["arm_order"] == "A5_A3":
                order = list(reversed(order))
            print("A5A3_%s_START %s %s=%s source_matched=%s" % (
                label, task_id, G1_CONDITION_FEATURE, condition,
                matched_prior is not None), flush=True)
            arm_rows: dict[str, Any] = {}
            for arm, prior in order:
                arm_rows[arm] = _run_arm(
                    repo_root=repo_root, arm_state=arm_states[arm],
                    task_spec=spec, public_context=context,
                    source_prior=prior, inventory=inventory, values=values,
                    mapped_roster=mapped_roster, config=config,
                    eval_uids=eval_uids, llm_counter=llm_counter,
                    consume_proposal_guidance=True,
                )
            row: dict[str, Any] = {
                "task_episode_id": task_id,
                "support_origins": list(spec["support_origins"]),
                "delayed_origins": list(spec["delayed_origins"]),
                "arm_order": spec["arm_order"],
                G1_CONDITION_FEATURE: condition,
                "task_signature": dict(context["task_signature"]),
                "source_prior_retrieval": {
                    "runtime_matcher": "evaluate_applicability",
                    "matched": matched_prior is not None,
                },
            }
            for arm in (BASE_ARM, PATCHED_ARM):
                arm_row = arm_rows[arm]
                row[arm] = {
                    "stop_reason": arm_row["stop_reason"],
                    "initial_decision": arm_row["initial"]["decision"],
                    "initial_protocol_error": arm_row["initial"].get("error"),
                    "proposal_guidance_consumed": (
                        arm_row["proposal_guidance_consumed"]
                    ),
                    "first_proposal": _mechanism_first_probe(arm_row),
                    "mechanism_stats": _arm_mechanism_stats(arm_row),
                    "metrics": arm_row["metrics"],
                    "probes": arm_row["probes"],
                    "winner": arm_row["winner"],
                    "lifecycle": arm_row["lifecycle"],
                    "target_memories_after": arm_row["target_memories_after"],
                    "active_local_skill_ids_after": (
                        arm_row["active_local_skill_ids_after"]
                    ),
                }
            rows.append(row)
            print("A5A3_%s_DONE %s A3=%s A5=%s" % (
                label, task_id, row[BASE_ARM]["stop_reason"],
                row[PATCHED_ARM]["stop_reason"]), flush=True)

    run_block(["e1v2_task_%02d" % i for i in range(1, A5A3_N0 + 1)], "N0")
    summary = _paired_summary(rows)
    plan = _sample_plan(summary, available_task_count=A5A3_MAX_N)
    n_final = min(int(plan["N_final"]), A5A3_MAX_N)
    extension = 0
    if n_final > len(rows):
        extension_ids = [
            "e1v2_task_%02d" % i for i in range(len(rows) + 1, n_final + 1)
        ]
        extension = len(extension_ids)
        run_block(extension_ids, "EXT")
        summary = _paired_summary(rows)

    unreadable = [
        row["task_episode_id"] for row in rows
        if any(int(row[arm]["metrics"].get("instrument_unreadable", 0))
               for arm in (BASE_ARM, PATCHED_ARM))
    ]
    return {
        "protocol_version": "a5a3_natural_cohort_development_v1",
        "verdict": "A5A3_NATURAL_DEVELOPMENT_COMPLETE",
        "preregistration": preregistration,
        "sample_plan": {**plan, "capped_N_final": n_final,
                        "extension_tasks_run": extension},
        "paired_summary": summary,
        "false_context_side": _side_summary(rows, condition=False),
        "true_context_side": _side_summary(rows, condition=True),
        "stage_decomposition": derive_stage_decomposition(rows),
        "source_prior_matched_task_count": sum(
            1 for row in rows if row["source_prior_retrieval"]["matched"]
        ),
        "instrument_unreadable_task_ids": unreadable,
        "rows": rows,
        "llm_api_call_count": llm_counter[0],
        "boundary": {
            "sealed_confirmation_opened": False, "e2_not_started": True,
            "weather_not_started": True,
            "cutoff_geometry_unchanged": True,
            "cohort_not_enumerated_for_balance": True,
        },
        "wall_seconds": time.perf_counter() - started,
    }


# --------------------------------------------- guidance v2: cross-cohort conflict


V2_STATE_REL = ".v2_state"
COHORT_E31 = "e31"
COHORT_T233 = "T233"


def _cohort_attempt_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Exposed attempts tagged with the cohort that produced them.

    The cohort tag is an evidence-provenance label, exactly like the UNGUIDED /
    GUIDANCE_CONDITIONED split.  It is never a matching feature and never
    enters an applicability predicate.
    """
    base_guidance = str(
        ((report.get("w2_guidance_freeze") or {}).get("base_guidance")) or ""
    )
    rows = _provenance_attempt_rows(report, base_guidance)
    for row in rows:
        row["cohort"] = COHORT_E31
    a5a3 = report.get("a5a3_natural_development") or {}
    for task_row in a5a3.get("rows") or []:
        condition = bool(task_row.get(G1_CONDITION_FEATURE))
        for arm in (BASE_ARM, PATCHED_ARM):
            arm_row = task_row.get(arm) or {}
            provenance = _guidance_provenance(
                arm_row.get("proposal_guidance_consumed"), base_guidance
            )
            for probe in arm_row.get("probes") or []:
                gain = probe.get("support_gain")
                program = tuple(
                    str(step["op"])
                    for step in (probe.get("compiled_steps") or [])
                )
                rows.append({
                    "task_episode_id": (
                        "a5a3_" + str(task_row["task_episode_id"])
                    ),
                    "arm": arm,
                    "attempt_index": probe.get("attempt_index"),
                    "program": list(program),
                    "is_mechanism": program == G1_MECHANISM_PROGRAM,
                    "contains_mechanism_operator": (
                        G1_MECHANISM_PROGRAM[0] in program
                    ),
                    "support_gain": gain,
                    "gain_readable": isinstance(gain, (int, float)),
                    "task_signature": dict(task_row.get("task_signature") or {}),
                    G1_CONDITION_FEATURE: condition,
                    "support_origins": list(task_row.get("support_origins") or []),
                    "arm_stop_reason": str(arm_row.get("stop_reason") or ""),
                    "evidence_source": "a5a3_natural",
                    "guidance_provenance": provenance,
                    "cohort": COHORT_T233,
                })
    return rows


def _cross_cohort_census(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Complete census with one extra grouping key: the producing cohort."""
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if not row["gain_readable"] or not row["program"]:
            continue
        key = (
            tuple(row["program"]),
            bool(row[G1_CONDITION_FEATURE]),
            str(row["cohort"]),
            _relation(row["support_gain"]),
        )
        cell = cells.setdefault(key, {"task_ids": set(), "attempt_count": 0})
        cell["task_ids"].add(row["task_episode_id"])
        cell["attempt_count"] += 1
    census = []
    for key in sorted(
        cells,
        key=lambda item: (
            G1_MECHANISM_PROGRAM[0] not in item[0],
            len(item[0]), item[0], not item[1], item[2], item[3],
        ),
    ):
        program, condition, cohort, relation = key
        cell = cells[key]
        census.append({
            "canonical_program": list(program),
            G1_CONDITION_FEATURE: condition,
            "cohort": cohort,
            "support_relation": relation,
            "distinct_task_count": len(cell["task_ids"]),
            "attempt_count": cell["attempt_count"],
        })
    return census


def attribute_cross_cohort_conflict(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Deterministic Runtime attribution of the cross-cohort Utility flip.

    Zero LLM, zero new Outcome.  The bounded first-fault ladder is applied with
    one reading stated explicitly rather than assumed:

    Step 1 asks whether the fault requires a *new* Context condition before it
    can be repaired.  It does not.  The observed fault is that guidance v1
    makes an unconditional active recommendation on single-cohort evidence; the
    correct behaviour -- not mandating, and deferring to other public evidence
    or Target Support -- is already available to the proposal layer and needs
    no new feature.  So this is not CONTEXT_GAP.  A CONTEXT_GAP reading would
    require inventing a discriminator between the two cohorts, which is exactly
    what the ruling forbids.
    """
    rows = _cohort_attempt_rows(report)
    census = _cross_cohort_census(rows)
    mechanism = [
        row for row in rows
        if row["is_mechanism"] and row["gain_readable"]
        and bool(row[G1_CONDITION_FEATURE]) is True
    ]

    def side(cohort: str, relation: str) -> dict[str, Any]:
        ids = sorted({
            row["task_episode_id"] for row in mechanism
            if row["cohort"] == cohort
            and _relation(row["support_gain"]) == relation
        })
        return {"distinct_task_count": len(ids), "task_episode_ids": ids}

    e31_pos = side(COHORT_E31, "POSITIVE")
    e31_neg = side(COHORT_E31, "NEGATIVE")
    t233_pos = side(COHORT_T233, "POSITIVE")
    t233_neg = side(COHORT_T233, "NEGATIVE")

    conflict_confirmed = bool(
        e31_pos["distinct_task_count"] >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
        and t233_neg["distinct_task_count"] >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
        and e31_neg["distinct_task_count"] == 0
        and t233_pos["distinct_task_count"] == 0
    )
    instrument_valid = all(row["gain_readable"] for row in mechanism)
    unreadable = (
        report.get("a5a3_natural_development") or {}
    ).get("instrument_unreadable_task_ids") or []
    correct_behaviour_available = True  # abstaining from an active mandate

    if not (instrument_valid and not unreadable):
        cause, stop = "NO_ACTIONABLE_EVIDENCE", "instrument_unreadable"
    elif not conflict_confirmed:
        cause, stop = "NO_ACTIONABLE_EVIDENCE", "conflict_not_repeated"
    elif correct_behaviour_available:
        cause, stop = "DECISION_GAP", (
            "repeated avoidable harm from an unconditional active clause whose "
            "supporting evidence exists in one cohort only"
        )
    else:
        cause, stop = "WORKFLOW_GAP", "correct_behaviour_unavailable"

    route_facts = {
        "expressibility_status": "PROVEN_EXPRESSIBLE",
        "expressibility_cause": None,
        "capability_skill_exists": True,
        "skill_retrieved": False,
        "constrained_proposal_succeeds": None,
    }
    route = route_program_supply_fault(
        **route_facts,
        context_resolved_decision_fault=(cause == "DECISION_GAP"),
    )
    return {
        "check": "cross_cohort_conflict_attribution",
        "zero_llm": True, "zero_new_outcome": True,
        "cause": cause,
        "repair_scope": G1_REPAIR_SCOPE if cause == "DECISION_GAP" else "NONE",
        "ladder_stop": stop,
        "conflict": {
            "canonical_program": list(G1_MECHANISM_PROGRAM),
            "condition": G1_CONDITION_FEATURE + " == true",
            COHORT_E31: {"positive": e31_pos, "negative": e31_neg},
            COHORT_T233: {"positive": t233_pos, "negative": t233_neg},
            "confirmed": conflict_confirmed,
        },
        "instrument_unreadable_task_ids": list(unreadable),
        "cross_cohort_census": census,
        "route": {"fields": route_facts, "result": list(route)},
        "authorized_surface": G1_SURFACE,
        "verdict": (
            "CROSS_COHORT_CONFLICT_CONFIRMED" if cause == "DECISION_GAP"
            else "CROSS_COHORT_CONFLICT_NOT_ACTIONABLE"
        ),
    }


_V2_SLOW_SYSTEM = (
    "You are the Slow Harness update stage. The Runtime has attributed the "
    "first fault and authorized exactly one Harness surface, "
    "candidate_policy.proposal_guidance. You may not change the Cause, widen "
    "the scope, or approve your own edit. "
    "A previously deployed version of this guidance is now in conflict with "
    "new evidence. You receive that deployed text, a complete de-duplicated "
    "evidence census grouped by canonical program, public Context condition "
    "and producing cohort, and no trajectories or utility numbers -- so do not "
    "invent thresholds. Evidence is counted in distinct_task_count. "
    "Exactly one kind of revision is authorized: keep the clauses whose "
    "supporting evidence is consistent across every cohort, and revoke or "
    "downgrade any active recommendation whose support exists in one cohort "
    "only. A downgraded clause must say that the condition is not sufficient "
    "on its own to prioritize a Workflow and that other public evidence or "
    "Target Support is required; it must not become a prohibition. "
    "You may not introduce a new observable feature, a new numeric threshold, "
    "a new operator, or any statement about Programs, Risk or the Judge. "
    "Write the full replacement text. Return JSON only: "
    "{'decision':'PATCH','new_guidance':'...'} or "
    "{'decision':'ABSTAIN','reason':'...'}."
)


def _v2_clause_audit(
    guidance: str,
    census: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Mechanical check that v2 kept the consistent clause and dropped the other.

    No parser: the frozen vocabulary of this slice is one operator and one
    Context feature, so the two directional clauses are enumerated lexically.
    A clause is authorized only if its supporting cell is consistent in EVERY
    cohort that observed it.
    """
    lowered = guidance.lower()
    mechanism = G1_MECHANISM_PROGRAM[0]

    def cells(condition: bool) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for cell in census:
            if (
                list(cell["canonical_program"]) != [mechanism]
                or bool(cell[G1_CONDITION_FEATURE]) is not condition
            ):
                continue
            out.setdefault(str(cell["cohort"]), {})[
                str(cell["support_relation"])
            ] = int(cell["distinct_task_count"])
        return out

    true_cells, false_cells = cells(True), cells(False)

    # Clause kinds do not share an evidence bar.  The frozen principle is
    # already asymmetric -- "a single opposite cell may block a global ban but
    # never authorizes a new active recommendation" -- because authorizing an
    # act removes optionality while weakening one preserves it.  Applied here:
    #
    #   ACTIVE_RECOMMENDATION ("prioritize X when C") must be positive in every
    #   cohort that observed C, with no negatives anywhere;
    #   DEPRIORITIZATION ("do not default to X when C") needs repeated harm in
    #   at least one cohort and must not be contradicted by a cohort where the
    #   same cell is uniformly positive.
    def mandate_supported(group: Mapping[str, Mapping[str, int]]) -> bool:
        seen = [c for c in group.values() if sum(c.values())]
        return bool(seen) and all(
            c.get("POSITIVE", 0) > 0 and c.get("NEGATIVE", 0) == 0
            for c in seen
        )

    def deprioritization_supported(
        group: Mapping[str, Mapping[str, int]],
    ) -> bool:
        harm = any(
            c.get("NEGATIVE", 0) >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            for c in group.values()
        )
        contradicted = any(
            c.get("POSITIVE", 0) >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            and c.get("NEGATIVE", 0) == 0
            for c in group.values()
        )
        return bool(harm and not contradicted)

    true_consistent = mandate_supported(true_cells)
    false_consistent = deprioritization_supported(false_cells)

    mentions_true_priority = (
        mechanism in lowered
        and any(word in lowered for word in ("prioritize", "prioritise",
                                             "prefer", "first choice"))
    )
    downgraded = any(
        phrase in lowered for phrase in (
            "not sufficient", "not enough", "on its own", "alone",
            "additional public evidence", "other public evidence",
            "target support", "does not by itself", "not by itself",
        )
    )
    keeps_false_clause = mechanism in lowered and (
        "false" in lowered or "not sufficient" in lowered
    )
    return {
        "true_side_evidence_consistent_across_cohorts": true_consistent,
        "false_side_evidence_consistent_across_cohorts": false_consistent,
        "true_side_cells": true_cells,
        "false_side_cells": false_cells,
        "text_still_asserts_true_side_priority": mentions_true_priority,
        "text_downgrades_true_side": downgraded,
        "text_keeps_false_side_clause": keeps_false_clause,
        "pass": bool(
            keeps_false_clause
            and false_consistent
            and (not true_consistent)
            and (downgraded or not mentions_true_priority)
        ),
    }


def run_guidance_v2(report_path: Path = REPORT_REL) -> dict[str, Any]:
    """One Slow PATCH producing guidance v2.  v1 is preserved, never edited."""
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    w2 = report.get("w2_guidance_freeze") or {}
    v1 = str(w2.get("frozen_guidance") or "")
    base_guidance = str(w2.get("base_guidance") or "")

    attribution = attribute_cross_cohort_conflict(report)
    result: dict[str, Any] = {
        "protocol_version": "guidance_v2_cross_cohort_repair_v1",
        "attribution": attribution,
        "guidance_v1_preserved": v1,
        "base_guidance": base_guidance,
        "no_retry_attempted": True,
        "no_kdd_validation_of_v2": True,
    }
    if attribution["verdict"] != "CROSS_COHORT_CONFLICT_CONFIRMED":
        return {**result, "verdict": "V2_CAUSE_NOT_ACTIONABLE",
                "llm_api_call_count": 0,
                "wall_seconds": time.perf_counter() - started}

    payload = {
        "attributed_cause": attribution["cause"],
        "repair_scope": attribution["repair_scope"],
        "surface_catalog": [{
            "surface_id": G1_SURFACE, "target_class": "proposal_control",
            "surface_type": "text", "allowed_operations": ["PATCH"],
        }],
        "deployed_guidance": v1,
        "conflict": attribution["conflict"],
        "evidence_census": attribution["cross_cohort_census"],
        "evidence_census_contract": {
            "unit_of_evidence": "distinct_task_count",
            "attempt_count_role": "diagnostic_only",
            "cohort_is_provenance_only": (
                "the cohort label groups evidence; it is never a matching "
                "feature and must not appear in the guidance text"
            ),
        },
        "authorized_revision": (
            "keep clauses consistent across every cohort; revoke or downgrade "
            "any active recommendation supported by a single cohort"
        ),
    }
    try:
        response = _e1_slow_call([
            {"role": "system", "content": _V2_SLOW_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])
    except (RuntimeError, ValueError) as exc:
        return {**result, "verdict": "V2_GUIDANCE_SUPPLY_FAILED",
                "stage": "slow_call",
                "error": type(exc).__name__ + ": " + str(exc),
                "llm_api_call_count": 0,
                "wall_seconds": time.perf_counter() - started}
    result["slow_payload"] = payload
    result["slow_response"] = response
    v2 = str(response.get("new_guidance") or "").strip()
    if str(response.get("decision") or "") != "PATCH" or not v2:
        return {**result, "verdict": "V2_GUIDANCE_SUPPLY_FAILED",
                "stage": "slow_decision", "llm_api_call_count": 1,
                "wall_seconds": time.perf_counter() - started}

    clause_audit = _v2_clause_audit(v2, attribution["cross_cohort_census"])
    result["clause_audit"] = clause_audit
    result["proposed_guidance_v2"] = v2
    if not clause_audit["pass"]:
        return {**result, "verdict": "V2_CLAUSE_AUDIT_FAILED",
                "llm_api_call_count": 1,
                "wall_seconds": time.perf_counter() - started}

    h0 = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(repo_root / V2_STATE_REL / "snapshots")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    store.materialize(h0)
    try:
        receipt = _apply_guidance_patch(controller, store, h0, v2)
    except (EditControllerError, ValueError, TypeError) as exc:
        return {**result, "verdict": "V2_GUIDANCE_SUPPLY_FAILED",
                "stage": "edit_controller",
                "error": type(exc).__name__ + ": " + str(exc),
                "llm_api_call_count": 1,
                "wall_seconds": time.perf_counter() - started}
    patched = receipt.candidate_snapshot
    store.set_active(patched.runtime_bundle_sha)
    before, after = dict(h0.candidate_policy), dict(patched.snapshot.candidate_policy)
    changed = sorted(
        k for k in set(before) | set(after) if before.get(k) != after.get(k)
    )
    single = bool(
        changed == ["proposal_guidance"]
        and list(receipt.source_surfaces_changed) == [G1_SURFACE]
        and _skill_ids(h0) == _skill_ids(patched.snapshot)
        and h0.instruction == patched.snapshot.instruction
    )
    return {
        **result,
        "verdict": "V2_GUIDANCE_FROZEN" if single else "V2_MULTI_SURFACE_MODIFICATION",
        "guidance_v2": str(after.get("proposal_guidance") or ""),
        "changed_candidate_policy_keys": changed,
        "single_surface_diff": single,
        "v2_runtime_bundle_sha": patched.runtime_bundle_sha,
        "llm_api_call_count": 1,
        "wall_seconds": time.perf_counter() - started,
    }


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
    "attribute_cross_cohort_conflict",
    "run_guidance_v2",
    "run_a5a3_natural",
    "eval_substrate_preflight",
    "reaudit_frozen_guidance",
    "run_w3",
    "run_w2_guidance_freeze",
    "derive_stage_decomposition",
    "REPLAY_TASK_IDS",
    "FRESH_TASK_IDS",
    "run_g1",
    "run_g1_attribution",
    "run_g1_guidance_patch",
]
