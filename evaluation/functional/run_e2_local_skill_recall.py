"""Persist #11 Target-local Skills and recall them on frozen task_06.

This is the lifecycle-only continuation of ``skill_store_integration_v1``
and ``lifecycle_probe_v1``.  It changes neither the recipe/compiler nor the
Fast prompt/adoption instrument.  Three #11 ``LOCAL_ACTIVE`` Episodes are
written through the real ``handle_fast_winner`` path into four isolated arm x
target stores, then the stores are resolved naturally for ``e1v2_task_06``.

The direct recall candidate still pays current-window Support confirmation and
the unchanged v2 delayed ladder.  A miss, failed confirmation, or abstention is
reported rather than repaired.  All state writes stay under
``_scratch/skill_store/local_lifecycle_v1``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_recipe_experience_to_skill as bridge  # noqa: E402
import run_e2_skill_store_integration as ssi  # noqa: E402
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402

from evaluation.functional.task_episode_harness import e1 as e1mod  # noqa: E402
from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_json_bytes,
)
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
    load_learned_skill_entry,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_SUPPORT,
    RELATION_ABSTAIN,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    build_episode,
    episode_from_dict,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _parse_frozen_steps,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: E402
    _resolve_apply_manifest,
)

PROTOCOL_VERSION = "local_lifecycle_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "local_skill_recall_v1.json"
OUT_MD = E2 / "local_skill_recall_v1.md"
INTEGRATION_JSON = E2 / "skill_store_integration_v1.json"
PROBE_JSON = E2 / "lifecycle_probe_v1.json"
STORE_ROOT = PROJECT_ROOT / "_scratch" / "skill_store" / PROTOCOL_VERSION
H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"

TARGET_IDS = ("T1", "T2")
ARM_ORDER = (("T1", "A3"), ("T1", "A5"), ("T2", "A3"), ("T2", "A5"))
TASK_INDEX = 5
TASK_ID = "e1v2_task_06"
MATERIAL_THRESHOLD = 0.005
REUSE_HARM_THRESHOLD = -0.005
LLM_CALL_BUDGET_TOTAL = 30
LLM_CALL_BUDGET_PER_EPISODE = 5
EXPERIENCE_PROVENANCE = "local_skill_recall"

LOCAL_PLAN = {
    ("T1", "A5"): {"program": "outlier_iqr", "excluded_series": []},
    ("T2", "A5"): {"program": "outlier_iqr", "excluded_series": []},
    ("T2", "A3"): {
        "program": "repair_level_shift",
        "excluded_series": ["0", "1", "3", "10", "11"],
    },
}
LOCAL_SOURCE = {
    ("T1", "A5"): "T1_A5",
    ("T2", "A5"): "T2_A5",
    ("T2", "A3"): "T2_A3",
}

PRE_REGISTERED: dict[str, Any] = {
    "fixed_before_first_llm_call": True,
    "one_changed_surface": "Target-local Skill persistence and recall lifecycle",
    "frozen": [
        "Source Guidance card bytes from skill_store_integration_v1",
        "recipe compiler and program menu",
        "ADOPTION_RULE_V2 semantics",
        "Metric, Consumer, Support budget and #10 prompt templates",
    ],
    "stores": {
        "A5": "the target's frozen Source Guidance card plus its own #11 ACTIVE local Skill",
        "A3_T2": "its own #11 ACTIVE local Skill only",
        "A3_T1": "no learned Skill; natural from-scratch control",
    },
    "next_window": (
        "task_episode_harness.e1._frozen_task_roster()[5], required to be "
        "e1v2_task_06; support and delayed origins copied verbatim"
    ),
    "reuse": (
        "a naturally retrieved local Skill supplies exactly its frozen plan; "
        "the plan must first reach +0.005 on current Support and then pass the "
        "unchanged v2 delayed ladder. Failure is an honest abstention"
    ),
    "masked_candidate_v2_correspondence": (
        "a directly supplied masked candidate is the only measured plan, not "
        "a full-batch shortlist winner; therefore ADOPTION_RULE_V2 gives it "
        "the identity bar at zero. A direct full-batch candidate is its own "
        "Support winner and sets max(0, its delayed) as the bar"
    ),
    "cost": (
        "the primary evaluation count is total Consumer retrains, matching "
        "#10: identity baselines, direct confirmation or shortlist, mask "
        "internals, Support bookkeeping and every delayed read all count"
    ),
    "experience": {
        "provenance": EXPERIENCE_PROVENANCE,
        "counts_as_unguided_exploration": False,
    },
    "per_cell_verdicts": [
        "RECALLED_REUSED_CHEAPER",
        "RECALLED_REUSED_NOT_CHEAPER",
        "RECALLED_NOT_REUSED",
        "RECALL_MISS",
        "NO_LOCAL_SKILL",
    ],
    "quality_guard": "a reused plan with delayed < -0.005 is also REUSE_HARMFUL",
    "overall": (
        "LOCAL_LIFECYCLE_CLOSES iff A5 has no RECALL_MISS on T1/T2, at "
        "least one A5 cell is RECALLED_REUSED_*, and none is REUSE_HARMFUL"
    ),
    "llm_budget": LLM_CALL_BUDGET_TOTAL,
    "circuit_breaker": "stop if the first episode produces no LLM payload",
}

ABORTED_PREFLIGHT_NOTE = (
    "Before the first LLM call, an initial 0-LLM preflight implementation "
    "instantiated BridgeSearch while resolving task_06. BridgeSearch starts "
    "identity-baseline computation in __init__; the process was manually "
    "interrupted before it returned any metric. No value entered selection, "
    "no artifact was written, and no method/store decision used it. The "
    "corrected preflight below reads only values[:support_cutoff] and performs "
    "zero Consumer retrains. The final run rebuilds all stores and accounts "
    "every completed evaluation from scratch."
)


class _Receipt:
    def __init__(self, gain: float) -> None:
        self.gain = float(gain)
        self.verification = type("Verification", (), {"passed": True})()


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    integration = json.loads(INTEGRATION_JSON.read_text(encoding="utf-8"))
    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    if integration.get("protocol_version") != "skill_store_integration_v1":
        raise SystemExit("unexpected #10 protocol_version")
    if probe.get("protocol_version") != "lifecycle_probe_v1":
        raise SystemExit("unexpected #11 protocol_version")
    return integration, probe


def _task06_window() -> dict[str, Any]:
    spec = dict(e1mod._frozen_task_roster()[TASK_INDEX])
    if spec.get("task_episode_id") != TASK_ID:
        raise SystemExit(
            "frozen roster index 5 is %r, expected %s"
            % (spec.get("task_episode_id"), TASK_ID)
        )
    return {
        "window_id": TASK_ID,
        "support_origins": [int(value) for value in spec["support_origins"]],
        "delayed_origins": [int(value) for value in spec["delayed_origins"]],
        "horizon": int(spec["horizon"]),
        "arm_order": str(spec["arm_order"]),
        "origin_source": "quoted from the frozen roster",
        "origin_provenance": (
            "task_episode_harness.e1._frozen_task_roster()[5], "
            "e1v2_task_06, support and delayed origins verbatim"
        ),
        # #10's runner records but never exposes these two reference fields.
        # This slice does not spend a hidden full-menu scan, so identity is the
        # neutral placeholder and the fields are removed from the final row.
        "reference_plan": {"program": bridge.IDENTITY, "excluded_series": []},
        "reference_delayed_aggregate_gain": 0.0,
    }


def _source_guidance(
    integration: Mapping[str, Any], target_id: str,
) -> dict[str, Any]:
    registration = integration["registration"]
    payload = dict(registration["card_payloads"][target_id])
    payload["risk_guards"] = dict(
        registration["card_risk_guards"][target_id]
    )
    return payload


def _probe_source(
    probe: Mapping[str, Any], draft_id: str,
) -> dict[str, Any]:
    for row in probe["drafts"]:
        if row.get("draft_id") != draft_id:
            continue
        transition = row.get("transition") or {}
        episode = transition.get("updated_episode")
        if (
            row.get("outcome") != "TRANSITIONED"
            or not episode
            or episode.get("local_status") != STATUS_LOCAL_ACTIVE
            or episode.get("evidence_level") != "DELAYED"
        ):
            raise SystemExit("#11 source %s is not ACTIVE/DELAYED" % draft_id)
        return dict(row)
    raise SystemExit("#11 source %s is missing" % draft_id)


def _register_guidance(
    *, store: SnapshotStore, snapshot: Any, payload: Mapping[str, Any], slot: str,
) -> Any:
    entry = load_learned_skill_entry(dict(payload))
    parent = store.materialize(snapshot)
    fork = store.fork(parent, "local-lifecycle-guidance-%s" % slot)
    try:
        path = fork / "skills" / "learned" / (entry.skill_id + ".json")
        path.write_bytes(canonical_json_bytes(dict(payload)) + b"\n")
        candidate = compile_snapshot(fork, verify_lock=False)
        materialized = store.materialize(
            candidate, parent_sha=snapshot.runtime_bundle_sha
        )
    finally:
        store.discard_fork(fork)
    store.set_active(materialized.runtime_bundle_sha)
    return materialized.snapshot


def _patch_lifecycle_metadata(
    *, store: SnapshotStore, snapshot: Any, skill_id: str,
    plan: Mapping[str, Any], source: Mapping[str, Any], slot: str,
) -> tuple[Any, dict[str, Any]]:
    skill = next(
        (item for item in snapshot.skills if item.skill_id == skill_id), None
    )
    if skill is None:
        raise ValueError("handle_fast_winner did not write %s" % skill_id)
    guards = dict(skill.risk_guards or {})
    after = source["transition"]["after"]
    probe = source["probe"]
    guards.update({
        "local_status": str(after["local_status"]),
        "evidence_level": str(after["evidence_level"]),
        "evidence_refs": [
            "artifacts/functional/e2/lifecycle_probe_v1.json",
            "draft_id:%s" % source["draft_id"],
        ],
        "source_episode_id": str(source["episode_id"]),
        "activation_probe_window": str(
            source["probe_window"]["probe_window_id"]
        ),
        "activation_probe_origins": list(probe["origins"]),
        "activation_probe_gain": float(probe["macro_gain"]),
        "activation_probe_se_block": float(probe["se_block"]),
        "activation_probe_gain_over_se": probe["gain_over_se"],
        "frozen_plan": {
            "program": str(plan["program"]),
            "excluded_series": sorted(
                str(uid) for uid in plan["excluded_series"]
            ),
        },
        "provenance": "target_local_skill",
        "current_task_support_confirmation_required": True,
    })
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    parent = store.materialize(snapshot)
    surface = "skill_library.entries/%s.risk_guards" % skill_id
    manifest = EditManifest(
        edit_id="record_local_active_%s" % slot,
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id="local-lifecycle-v1",
        target_surface_id=surface,
        operation=EditOperation.PATCH,
        surface_precondition={
            "kind": "SHA",
            "sha": controller.surface_precondition_sha(parent, surface),
        },
        dependency_precondition_shas={},
        minimal_patch={"value": guards},
        new_value=None,
        observable_applicability=None,
        predicted_agent_behavior_change=("retrieve_skill:%s" % skill_id,),
        predicted_data_effect=("local_improvement",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=None,
    )
    receipt = controller.apply_to_fork(
        parent,
        _resolve_apply_manifest(manifest, snapshot),
        confirmed_cause="RISK_GAP",
    )
    updated = receipt.candidate_snapshot.snapshot
    store.set_active(updated.runtime_bundle_sha)
    return updated, guards


def _persist_local_skill(
    *, store: SnapshotStore, snapshot: Any, plan: Mapping[str, Any],
    source: Mapping[str, Any], slot: str,
) -> tuple[Any, dict[str, Any]]:
    steps = ((str(plan["program"]), {}),)
    source_episode = episode_from_dict(
        dict(source["transition"]["updated_episode"])
    )
    context = dict(source_episode.context_summary)
    geometry = dict(context.get("program_geometry") or {})
    geometry.update({
        "program_steps": [{"op": steps[0][0], "params": {}}],
        "frozen_plan_scope": {
            "excluded_series": sorted(
                str(uid) for uid in plan["excluded_series"]
            )
        },
    })
    context["program_geometry"] = geometry
    episode = dataclasses.replace(
        source_episode,
        workflow_signature=e1mod._v2_workflow_signature(steps),
        context_summary=context,
    )
    method = TTHAMethod(
        e1mod._FastAgentStub(), snapshot, experience_episodes=(episode,)
    )
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    card = {
        "pattern_id": "local-lifecycle-v1-%s" % slot,
        "failure_family": "target_local_skill_persistence_gap",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {
            "steps": [{"op": steps[0][0], "params": {}}]
        },
    }
    event = method.handle_fast_winner(
        episode,
        steps,
        controller=controller,
        store=store,
        card=card,
        evaluator=lambda _steps, _mode: _Receipt(
            float(episode.support_response["gain"])
        ),
        fast_features={"task_kind": "forecast"},
        support_gain=float(episode.support_response["gain"]),
        confirmed_cause="SKILL_LIBRARY_GAP",
    )
    if event.get("stage") != "pending":
        raise ValueError("handle_fast_winner returned %r" % event)
    delayed_event = method.handle_feedback_delayed(
        lambda _steps, _mode: _Receipt(
            float(episode.delayed_response["gain"])
        ),
        episode_id=episode.episode_id,
    )
    if delayed_event.get("stage") != "approved":
        raise ValueError("cached #11 delayed evidence was not approved: %r" % delayed_event)
    snapshot = method._active_snapshot()
    store.set_active(snapshot.runtime_bundle_sha)
    skill_id = "fast_winner_%s" % episode.workflow_signature
    snapshot, guards = _patch_lifecycle_metadata(
        store=store,
        snapshot=snapshot,
        skill_id=skill_id,
        plan=plan,
        source=source,
        slot=slot,
    )
    stored = next(skill for skill in snapshot.skills if skill.skill_id == skill_id)
    frozen = _parse_frozen_steps(stored.body)
    if frozen != steps:
        raise ValueError("stored frozen steps do not match the #11 plan")
    return snapshot, {
        "skill_id": skill_id,
        "handle_fast_winner": event,
        "cached_delayed_approval": delayed_event,
        "cached_evidence_only_no_remeasurement": True,
        "frozen_steps": [{"op": op, "params": dict(params)} for op, params in frozen],
        "frozen_plan": dict(guards["frozen_plan"]),
        "lifecycle": {
            "local_status": guards["local_status"],
            "evidence_level": guards["evidence_level"],
            "evidence_refs": list(guards["evidence_refs"]),
            "activation_probe_window": guards["activation_probe_window"],
            "activation_probe_origins": list(guards["activation_probe_origins"]),
            "activation_probe_gain": guards["activation_probe_gain"],
            "activation_probe_se_block": guards["activation_probe_se_block"],
            "activation_probe_gain_over_se": guards[
                "activation_probe_gain_over_se"
            ],
        },
    }


def _build_stores(
    integration: Mapping[str, Any], probe: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if STORE_ROOT.exists():
        shutil.rmtree(STORE_ROOT)
    stores: dict[str, dict[str, Any]] = {}
    for target_id, arm in ARM_ORDER:
        slot = "%s_%s" % (arm.lower(), target_id.lower())
        root = STORE_ROOT / slot / "snapshots"
        store = SnapshotStore(root)
        snapshot = compile_snapshot(H0_ROOT, verify_lock=False)
        store.materialize(snapshot)
        store.set_active(snapshot.runtime_bundle_sha)
        row: dict[str, Any] = {
            "slot": slot,
            "store_root": _repo_rel(root),
            "status": "BUILDING",
            "source_guidance": None,
            "local_skill": None,
        }
        try:
            if arm == "A5":
                guidance = _source_guidance(integration, target_id)
                snapshot = _register_guidance(
                    store=store,
                    snapshot=snapshot,
                    payload=guidance,
                    slot=slot,
                )
                row["source_guidance"] = {
                    "skill_id": str(guidance["skill_id"]),
                    "loaded_verbatim_from": (
                        "artifacts/functional/e2/skill_store_integration_v1.json::"
                        "registration.card_payloads/card_risk_guards.%s" % target_id
                    ),
                }
            source_id = LOCAL_SOURCE.get((target_id, arm))
            if source_id is not None:
                source = _probe_source(probe, source_id)
                snapshot, local = _persist_local_skill(
                    store=store,
                    snapshot=snapshot,
                    plan=LOCAL_PLAN[(target_id, arm)],
                    source=source,
                    slot=slot,
                )
                row["local_skill"] = local
            row.update({
                "status": "REGISTERED",
                "skill_ids": [skill.skill_id for skill in snapshot.skills],
                "active_pointer": json.loads(
                    store.active_path.read_text(encoding="utf-8")
                ),
            })
        except Exception as exc:  # noqa: BLE001
            row.update({
                "status": "SCHEMA_BLOCKED",
                "blocked_at_interface": "skill-entry/1 -> compiler/EditController lifecycle path",
                "blocked_reason": "%s: %s" % (type(exc).__name__, exc),
            })
        row["_snapshot"] = snapshot
        row["_store"] = store
        stores[slot] = row
        print(
            "LSR store %-5s %-15s local=%s guidance=%s"
            % (
                slot,
                row["status"],
                (row.get("local_skill") or {}).get("skill_id") or "-",
                (row.get("source_guidance") or {}).get("skill_id") or "-",
            ),
            flush=True,
        )
        if row["status"] == "SCHEMA_BLOCKED":
            break
    return stores


def _retrieval(
    snapshot: Any, search: Any, expected_local_id: str | None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    context = ssi._public_features(search)
    view = resolve_harness_view(
        snapshot, dict(context["features"]), role="fast"
    )
    local = next(
        (
            skill for skill in view.skills
            if skill.skill_id == expected_local_id
        ),
        None,
    )
    local_row = None
    if local is not None:
        local_row = {
            "skill_id": local.skill_id,
            "body": str(local.body),
            "observable_applicability": wvc._plain(
                local.observable_applicability
            ),
            "risk_guards": wvc._plain(local.risk_guards or {}),
            "frozen_steps": (
                None if _parse_frozen_steps(local.body) is None else [
                    {"op": op, "params": dict(params)}
                    for op, params in _parse_frozen_steps(local.body) or ()
                ]
            ),
        }
    guidance = [
        skill.skill_id for skill in view.skills
        if skill.skill_id.startswith("recipe_batch_guidance_")
    ]
    return view, {
        "expected_local_skill_id": expected_local_id,
        "local_skill_hit": bool(local) if expected_local_id else None,
        "resolved_skill_ids": list(view.skill_ids),
        "resolved_memory_ids": list(view.memory_ids),
        "source_guidance_hits": guidance,
        "local_skill": local_row,
        "context": {
            key: value for key, value in context.items()
            if key != "per_series"
        },
    }, context


def _confirm_support(
    search: Any, plan: Mapping[str, Any],
) -> dict[str, Any]:
    program = str(plan["program"])
    excluded = sorted(str(uid) for uid in plan["excluded_series"])
    if not excluded:
        return search.full_batch_support(program)
    search.support_evaluations_charged += 1
    rows = search._masked(program, set(excluded), search.support)
    gains = search._gains(rows)
    search.log.append({
        "kind": "local_skill_direct_support_confirmation",
        "program": program,
        "excluded_series": excluded,
        "charged": True,
        "aggregate_gain": gains["aggregate_gain"],
    })
    return gains


def _direct_recall(
    *, target: Mapping[str, Any], arm: str, window: Mapping[str, Any],
    snapshot: Any, expected_skill_id: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    search = bridge.BridgeSearch(
        cohort=str(target["cohort"]),
        consumer_variant=str(target["consumer_variant"]),
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )
    _view, retrieval, _context = _retrieval(
        snapshot, search, expected_skill_id
    )
    skill = retrieval.get("local_skill")
    if not skill:
        return {
            "episode_id": "%s_%s_%s" % (TASK_ID, target["target_id"], arm),
            "target_id": target["target_id"],
            "arm": arm,
            "cohort": target["cohort"],
            "consumer_variant": target["consumer_variant"],
            "window_id": window["window_id"],
            "support_origins": list(window["support_origins"]),
            "delayed_origins": list(window["delayed_origins"]),
            "retrieval": retrieval,
            "recall_payload": None,
            "final_plan": None,
            "support": None,
            "delayed": None,
            "llm_calls": 0,
            "evaluations_used": 0,
            "consumer_retrains_total": int(search.retrains),
            "instrument": search.accounting(),
            "wall_seconds": time.perf_counter() - started,
        }
    guards = dict(skill["risk_guards"])
    plan = dict(guards.get("frozen_plan") or {})
    frozen_steps = list(skill.get("frozen_steps") or [])
    if (
        len(frozen_steps) != 1
        or frozen_steps[0].get("op") != plan.get("program")
        or dict(frozen_steps[0].get("params") or {})
        or plan.get("program") not in bridge.TREATMENTS
    ):
        raise ValueError("retrieved local Skill does not carry one valid frozen recipe plan")
    plan = {
        "program": str(plan["program"]),
        "excluded_series": sorted(
            str(uid) for uid in plan.get("excluded_series") or ()
        ),
    }
    unknown = sorted(set(plan["excluded_series"]) - set(search.train_uids))
    if unknown:
        raise ValueError("frozen plan excludes unknown training series %s" % unknown)
    support = _confirm_support(search, plan)
    support_gain = float(support["aggregate_gain"])
    material = support_gain >= MATERIAL_THRESHOLD
    recall_payload = {
        "source": "naturally_retrieved_target_local_skill",
        "skill_id": expected_skill_id,
        "frozen_steps": frozen_steps,
        "frozen_plan": plan,
        "current_support_confirmation": {
            "aggregate_gain": support_gain,
            "material_threshold": MATERIAL_THRESHOLD,
            "passed": material,
        },
    }
    if material:
        measured = [{
            "kind": "FULL_BATCH" if not plan["excluded_series"] else "MASKED_PLAN",
            "program": plan["program"],
            "excluded_series": list(plan["excluded_series"]),
            "full_batch": not plan["excluded_series"],
            "support_aggregate_gain": support_gain,
        }]
        ladder = ssi._ladder(search, plans=measured, named=plan)
        final_plan = dict(ladder["final_plan"])
        final_support = dict(ladder["support"])
        delayed = dict(ladder["delayed"])
        adoption = {
            key: value for key, value in ladder.items()
            if key not in {"support", "delayed"}
        }
    else:
        final_plan = {"program": bridge.IDENTITY, "excluded_series": []}
        final_support = dict(search.support_of_plan(bridge.IDENTITY, []))
        delayed = dict(search.delayed_gate(bridge.IDENTITY, []))
        adoption = {
            "path": "SUPPORT_CONFIRMATION_FAILED_ABSTAIN",
            "path_text": (
                "the frozen local plan missed the +0.005 current-window "
                "Support line; this episode abstained without forcing reuse"
            ),
            "bar": 0.0,
            "gate_passed": False,
            "final_plan": dict(final_plan),
        }
    reused = bool(
        final_plan["program"] == plan["program"]
        and sorted(final_plan["excluded_series"])
        == sorted(plan["excluded_series"])
    )
    delayed_gain = float(delayed["aggregate_gain"])
    if final_plan["program"] == bridge.IDENTITY:
        relation = RELATION_ABSTAIN
    elif float(final_support["aggregate_gain"]) > 0 and delayed_gain > 0:
        relation = RELATION_POSITIVE
    elif (float(final_support["aggregate_gain"]) > 0) != (delayed_gain > 0):
        relation = RELATION_CONFLICT
    else:
        relation = RELATION_NEGATIVE
    return {
        "episode_id": "%s_%s_%s" % (TASK_ID, target["target_id"], arm),
        "target_id": target["target_id"],
        "arm": arm,
        "cohort": target["cohort"],
        "consumer_variant": target["consumer_variant"],
        "window_id": window["window_id"],
        "support_origins": list(window["support_origins"]),
        "delayed_origins": list(window["delayed_origins"]),
        "retrieval": retrieval,
        "recall_payload": recall_payload,
        "recall_candidate_plan": plan,
        "reuse_adopted": reused,
        "support_confirmation_passed": material,
        "adoption_ladder": adoption,
        "final_plan": final_plan,
        "support": final_support,
        "delayed": delayed,
        "relation": relation,
        "llm_calls": 0,
        "evaluations_used": int(search.support_evaluations_charged),
        "consumer_retrains_total": int(search.retrains),
        "instrument": search.accounting(),
        "wall_seconds": time.perf_counter() - started,
    }


def _search_episode(
    *, target: Mapping[str, Any], arm: str, window: Mapping[str, Any],
    slot: Mapping[str, Any], expected_skill_id: str | None, llm_budget: int,
) -> dict[str, Any]:
    record = ssi._run_arm(
        target=target,
        arm=arm,
        window=window,
        slot=slot,
        expected_skill_id=expected_skill_id,
        llm_budget=llm_budget,
    )
    record["episode_id"] = "%s_%s_%s" % (TASK_ID, target["target_id"], arm)
    record["reuse_adopted"] = False
    record.pop("reference_delayed_aggregate_gain", None)
    record.pop("reference_plan", None)
    record.pop("capture_ratio", None)
    record.pop("matches_reference_plan", None)
    record.pop("public_input_sha256", None)
    record.pop("base_input_field_shas", None)
    return record


def _experience(record: Mapping[str, Any]) -> dict[str, Any] | None:
    plan = record.get("final_plan")
    support = record.get("support")
    delayed = record.get("delayed")
    if not plan or support is None or delayed is None:
        return None
    non_identity = str(plan["program"]) != bridge.IDENTITY
    support_gain = float(support["aggregate_gain"])
    delayed_gain = float(delayed["aggregate_gain"])
    if not non_identity:
        relation = RELATION_ABSTAIN
    elif support_gain > 0 and delayed_gain > 0:
        relation = RELATION_POSITIVE
    elif (support_gain > 0) != (delayed_gain > 0):
        relation = RELATION_CONFLICT
    else:
        relation = RELATION_NEGATIVE
    status = (
        STATUS_LOCAL_DRAFT
        if non_identity and support_gain >= MATERIAL_THRESHOLD and delayed_gain > 0
        else STATUS_EPISODE_ONLY
    )
    audit = {
        "provenance": EXPERIENCE_PROVENANCE,
        "counts_as_unguided_exploration": False,
    }
    episode = build_episode(
        episode_id="lsr_%s_%s_%s" % (
            str(record["target_id"]).lower(),
            str(record["arm"]).lower(),
            TASK_ID,
        ),
        task_consumer_key="batch:%s|consumer:%s" % (
            record["cohort"], record["consumer_variant"]
        ),
        domain_namespace=str(record["cohort"]),
        context_summary={
            "task_episode_id": TASK_ID,
            "arm": str(record["arm"]),
            "cohort": {"cohort_name": str(record["cohort"])},
            "local_pattern": {
                "consumer_variant": str(record["consumer_variant"]),
                "retrieved_local_skill_id": (
                    (record.get("retrieval") or {}).get(
                        "expected_local_skill_id"
                    )
                ),
                "local_skill_hit": (
                    (record.get("retrieval") or {}).get("local_skill_hit")
                ),
            },
            "program_geometry": {
                "program": str(plan["program"]),
                "excluded_series": list(plan["excluded_series"]),
                "consumer_retrains": int(record["consumer_retrains_total"]),
            },
        },
        workflow_signature=workflow_signature_of(
            () if not non_identity else ({"op": str(plan["program"])},)
        ),
        support_response={
            "gain": support_gain,
            "accepted": support_gain >= MATERIAL_THRESHOLD,
            "block_origins": list(record["support_origins"]),
            "program": str(plan["program"]),
            "excluded_series": list(plan["excluded_series"]),
            **audit,
        },
        delayed_response={
            "evaluated": True,
            "gain": delayed_gain,
            "se_block": None,
            "gain_over_se": None,
            "block_origins": list(record["delayed_origins"]),
            "took_part_in_selection": True,
            **audit,
        },
        relation=relation,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=status,
        evidence_refs=(EXPERIENCE_PROVENANCE, PROTOCOL_VERSION),
    )
    return episode.to_dict()


def _cell_verdict(
    record: Mapping[str, Any], task04_cost: int, had_local_skill: bool,
) -> dict[str, Any]:
    retrieval = record.get("retrieval") or {}
    hit = retrieval.get("local_skill_hit")
    reused = bool(record.get("reuse_adopted"))
    delayed = record.get("delayed") or {}
    delayed_gain = (
        None if delayed.get("aggregate_gain") is None
        else float(delayed["aggregate_gain"])
    )
    cost = int(record.get("consumer_retrains_total") or 0)
    if not had_local_skill:
        primary = "NO_LOCAL_SKILL"
    elif hit is not True:
        primary = "RECALL_MISS"
    elif not reused:
        primary = "RECALLED_NOT_REUSED"
    elif cost < task04_cost:
        primary = "RECALLED_REUSED_CHEAPER"
    else:
        primary = "RECALLED_REUSED_NOT_CHEAPER"
    harmful = bool(
        reused and delayed_gain is not None
        and delayed_gain < REUSE_HARM_THRESHOLD
    )
    return {
        "primary": primary,
        "also": ["REUSE_HARMFUL"] if harmful else [],
        "retrieval_hit": hit,
        "reuse_adopted": reused,
        "reuse_harmful": harmful,
        "task04_evaluation_count": task04_cost,
        "task06_evaluation_count": cost,
        "task06_minus_task04": cost - task04_cost,
        "cost_unit": "Consumer retrains",
    }


def _public_store_rows(stores: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        slot: {
            key: value for key, value in row.items()
            if not key.startswith("_")
        }
        for slot, row in stores.items()
    }


def _overall(cells: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    a5 = [cells["A5_T1"], cells["A5_T2"]]
    no_miss = all(row["verdict"]["primary"] != "RECALL_MISS" for row in a5)
    reused = any(
        row["verdict"]["primary"].startswith("RECALLED_REUSED_")
        for row in a5
    )
    no_harm = not any(row["verdict"]["reuse_harmful"] for row in a5)
    if no_miss and reused and no_harm:
        return (
            "LOCAL_LIFECYCLE_CLOSES",
            "A5 had zero RECALL_MISS across T1/T2, reused at least one local "
            "Skill, and no reused plan crossed the -0.005 harm guard",
        )
    failures = []
    if not no_miss:
        failures.append("A5_RECALL_MISS")
    if not reused:
        failures.append("A5_NO_REUSE")
    if not no_harm:
        failures.append("REUSE_HARMFUL")
    return " + ".join(failures), "; ".join(failures)


def run(*, preflight_only: bool = False) -> int:
    started = time.perf_counter()
    integration, probe = _load_inputs()
    window = _task06_window()
    stores = _build_stores(integration, probe)
    blocked = [
        row for row in stores.values() if row["status"] == "SCHEMA_BLOCKED"
    ]
    if blocked:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "overall_verdict": "SCHEMA_BLOCKED",
            "overall_verdict_reason": blocked[0]["blocked_reason"],
            "pre_registered": PRE_REGISTERED,
            "task06_window": window,
            "stores": _public_store_rows(stores),
            "arm_targets": [],
            "cells": {},
            "learning_curve": [],
            "llm_call_count": 0,
            "stopped_early": "store persistence stopped at the refusing interface",
            "wall_seconds": time.perf_counter() - started,
        }
        if preflight_only:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 2
        _write_outputs(payload)
        return 2

    if preflight_only:
        checks = {}
        for target_id, arm in ARM_ORDER:
            slot = stores["%s_%s" % (arm.lower(), target_id.lower())]
            loaded = ssi.bch.load_cohort(
                PROJECT_ROOT, str(bridge.TARGETS[target_id]["cohort"])
            )
            public_only = type("PublicOnlyContext", (), {
                "train_uids": [str(uid) for uid in loaded["train_uids"]],
                "support": tuple(window["support_origins"]),
                "values": loaded["values"],
            })()
            context = ssi._public_features(public_only)
            view = resolve_harness_view(
                slot["_snapshot"], dict(context["features"]), role="fast"
            )
            local = slot.get("local_skill") or {}
            local_id = local.get("skill_id")
            checks["%s_%s" % (arm, target_id)] = {
                "local_skill_hit": (
                    any(skill.skill_id == local_id for skill in view.skills)
                    if local_id else None
                ),
                "resolved_skill_ids": list(view.skill_ids),
                "consumer_retrains": 0,
                "observation_cutoff": int(context["observation_cutoff"]),
            }
        print(json.dumps({
            "status": "PREFLIGHT_OK",
            "task06_window": window,
            "retrieval": checks,
            "llm_calls": 0,
            "aborted_preflight_note": ABORTED_PREFLIGHT_NOTE,
        }, indent=2, ensure_ascii=False))
        return 0

    task04 = {
        (str(row["target_id"]), str(row["arm"])): int(
            row["consumer_retrains_total"]
        )
        for row in integration["arm_targets"]
        if row["target_id"] in TARGET_IDS
    }
    records: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    llm_used = 0
    stopped: str | None = None
    for target_id, arm in ARM_ORDER:
        target = dict(bridge.TARGETS[target_id])
        target["window_id"] = TASK_ID
        target["task_index"] = TASK_INDEX
        slot = stores["%s_%s" % (arm.lower(), target_id.lower())]
        local = slot.get("local_skill") or {}
        local_id = local.get("skill_id")
        if local_id:
            try:
                record = _direct_recall(
                    target=target,
                    arm=arm,
                    window=window,
                    snapshot=slot["_snapshot"],
                    expected_skill_id=str(local_id),
                )
            except Exception as exc:  # noqa: BLE001
                stopped = "SCHEMA_BLOCKED during recall: %s: %s" % (
                    type(exc).__name__, exc
                )
                break
            if record["retrieval"]["local_skill_hit"] is not True:
                expected = (
                    ssi.SKILL_ID[target_id] if arm == "A5" else None
                )
                remaining = LLM_CALL_BUDGET_TOTAL - llm_used
                record = _search_episode(
                    target=target,
                    arm=arm,
                    window=window,
                    slot=slot,
                    expected_skill_id=expected,
                    llm_budget=min(LLM_CALL_BUDGET_PER_EPISODE, remaining),
                )
                record["retrieval"]["expected_local_skill_id"] = local_id
                record["retrieval"]["local_skill_hit"] = False
        else:
            remaining = LLM_CALL_BUDGET_TOTAL - llm_used
            record = _search_episode(
                target=target,
                arm=arm,
                window=window,
                slot=slot,
                expected_skill_id=None,
                llm_budget=min(LLM_CALL_BUDGET_PER_EPISODE, remaining),
            )
            record["retrieval"]["expected_local_skill_id"] = None
            record["retrieval"]["local_skill_hit"] = None
        llm_used += int(record.get("llm_calls") or 0)
        if llm_used > LLM_CALL_BUDGET_TOTAL:
            raise RuntimeError("LLM call budget exceeded")
        experience = _experience(record)
        record["experience"] = experience
        record["stored_skill_lifecycle_before"] = (
            (local.get("lifecycle") or {}).get("local_status")
        )
        record["stored_skill_lifecycle_after"] = (
            (local.get("lifecycle") or {}).get("local_status")
        )
        record["lifecycle_note"] = (
            "the task_06 delayed reading took part in this episode's v2 "
            "adoption gate, so it records an Experience but does not "
            "retroactively replace #11's out-of-selection ACTIVE evidence"
        )
        verdict = _cell_verdict(
            record,
            task04_cost=task04[(target_id, arm)],
            had_local_skill=bool(local_id),
        )
        key = "%s_%s" % (arm, target_id)
        cells[key] = {
            "target_id": target_id,
            "arm": arm,
            "verdict": verdict,
            "retrieval": record.get("retrieval"),
            "support_confirmation": (
                (record.get("recall_payload") or {}).get(
                    "current_support_confirmation"
                )
            ),
            "adopted_plan": record.get("final_plan"),
            "support_aggregate_gain": (
                (record.get("support") or {}).get("aggregate_gain")
            ),
            "delayed_aggregate_gain": (
                (record.get("delayed") or {}).get("aggregate_gain")
            ),
            "task04_evaluation_count": task04[(target_id, arm)],
            "task06_evaluation_count": int(
                record.get("consumer_retrains_total") or 0
            ),
            "charged_support_evaluations_task06": int(
                record.get("evaluations_used") or 0
            ),
            "stored_skill_lifecycle": {
                "before": record["stored_skill_lifecycle_before"],
                "after": record["stored_skill_lifecycle_after"],
                "task06_experience_status": (
                    (experience or {}).get("local_status")
                ),
            },
        }
        records.append(record)
        print(
            "LSR %s %s hit=%s reused=%s support=%s delayed=%s cost=%d/%d llm=%d"
            % (
                target_id,
                arm,
                verdict["retrieval_hit"],
                verdict["reuse_adopted"],
                cells[key]["support_aggregate_gain"],
                cells[key]["delayed_aggregate_gain"],
                cells[key]["task06_evaluation_count"],
                cells[key]["task04_evaluation_count"],
                int(record.get("llm_calls") or 0),
            ),
            flush=True,
        )
        if len(records) == 1 and (
            record.get("shortlist_payload") is None
            or record.get("adoption_payload") is None
        ):
            stopped = "the first episode produced no complete LLM payload"
            break

    if stopped and len(cells) < len(ARM_ORDER):
        overall, reason = "STOPPED_EARLY", stopped
    else:
        overall, reason = _overall(cells)
    learning_curve = [
        {
            "arm": arm,
            "target_id": target_id,
            "task04_evaluation_count": task04[(target_id, arm)],
            "task06_evaluation_count": cells["%s_%s" % (arm, target_id)][
                "task06_evaluation_count"
            ],
            "delta": (
                cells["%s_%s" % (arm, target_id)]["task06_evaluation_count"]
                - task04[(target_id, arm)]
            ),
            "unit": "Consumer retrains",
        }
        for target_id, arm in ARM_ORDER
        if "%s_%s" % (arm, target_id) in cells
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "persist #11 ACTIVE Target-local Skills through handle_fast_winner "
            "and measure natural recall/reuse on frozen task_06"
        ),
        "overall_verdict": overall,
        "overall_verdict_reason": reason,
        "pre_registered": PRE_REGISTERED,
        "task06_window": {
            key: value for key, value in window.items()
            if not key.startswith("reference_")
        },
        "source_artifacts_read_only": [
            _repo_rel(INTEGRATION_JSON), _repo_rel(PROBE_JSON)
        ],
        "aborted_preflight_note": ABORTED_PREFLIGHT_NOTE,
        "stores": _public_store_rows(stores),
        "cells": cells,
        "learning_curve": learning_curve,
        "cost_definition": PRE_REGISTERED["cost"],
        "experience_provenance": EXPERIENCE_PROVENANCE,
        "experience_counts_as_unguided_exploration": False,
        "experience_entries_written": [
            record["experience"] for record in records
            if record.get("experience") is not None
        ],
        "llm_call_count": llm_used,
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
        "stopped_early": stopped,
        "arm_targets": records,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_outputs(payload)
    return 0 if overall == "LOCAL_LIFECYCLE_CLOSES" else 1


def _plan_label(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "--"
    excluded = [str(uid) for uid in plan.get("excluded_series") or ()]
    return "%s%s" % (
        plan["program"],
        " full batch" if not excluded else " minus " + ", ".join(excluded),
    )


def _fmt(value: Any) -> str:
    return "--" if value is None else "%+.6f" % float(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Target-local Skill persistence and task_06 recall",
        "",
        "**Overall: `%s`** -- %s" % (
            payload["overall_verdict"], payload["overall_verdict_reason"]
        ),
        "",
        "Only the lifecycle surface changed. Source Guidance, recipe/compiler, "
        "program menu, Consumer/Metric/Support budget, #10 prompt templates and "
        "ADOPTION_RULE_V2 stayed frozen. task_06 is quoted verbatim from the "
        "frozen E1-v2 roster.",
        "",
        "## Per arm-target",
        "",
        "| cell | verdict | local retrieval | support confirmation | adopted | support | delayed | task_04 cost | task_06 cost | lifecycle |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in ("A3_T1", "A5_T1", "A3_T2", "A5_T2"):
        row = payload.get("cells", {}).get(key)
        if not row:
            continue
        retrieval = row.get("retrieval") or {}
        confirm = row.get("support_confirmation") or {}
        life = row.get("stored_skill_lifecycle") or {}
        verdict = row["verdict"]["primary"]
        if row["verdict"].get("also"):
            verdict += " + " + " + ".join(row["verdict"]["also"])
        lines.append(
            "| `%s` | `%s` | %s (`%s`) | %s | `%s` | %s | %s | %d | %d | %s -> %s; episode %s |"
            % (
                key,
                verdict,
                retrieval.get("local_skill_hit"),
                retrieval.get("expected_local_skill_id") or "--",
                _fmt(confirm.get("aggregate_gain")),
                _plan_label(row.get("adopted_plan")),
                _fmt(row.get("support_aggregate_gain")),
                _fmt(row.get("delayed_aggregate_gain")),
                row["task04_evaluation_count"],
                row["task06_evaluation_count"],
                life.get("before") or "--",
                life.get("after") or "--",
                life.get("task06_experience_status") or "--",
            )
        )
    lines.extend([
        "",
        "Costs are total Consumer retrains, not shortlist counts: identity "
        "baselines, direct confirmation/shortlist, mask internals, Support "
        "bookkeeping and delayed reads are all included.",
        "",
        "## Learning curve",
        "",
        "| arm | target | task_04 | task_06 | delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in payload.get("learning_curve", ()):
        lines.append(
            "| `%s` | `%s` | %d | %d | %+d |"
            % (
                row["arm"], row["target_id"],
                row["task04_evaluation_count"],
                row["task06_evaluation_count"], row["delta"],
            )
        )
    lines.extend([
        "",
        "## Persistence evidence",
        "",
    ])
    for slot, row in payload.get("stores", {}).items():
        local = row.get("local_skill") or {}
        guidance = row.get("source_guidance") or {}
        life = local.get("lifecycle") or {}
        lines.append(
            "- `%s`: `%s`; Guidance `%s`; local `%s`; status `%s`, evidence "
            "`%s`, #11 probe `%s` gain %s."
            % (
                slot,
                row.get("status"),
                guidance.get("skill_id") or "--",
                local.get("skill_id") or "--",
                life.get("local_status") or "--",
                life.get("evidence_level") or "--",
                life.get("activation_probe_window") or "--",
                _fmt(life.get("activation_probe_gain")),
            )
        )
    lines.extend([
        "",
        "## Provenance and stopping conditions",
        "",
        "Every task_06 Experience records `provenance=local_skill_recall` and "
        "`counts_as_unguided_exploration=false`. The #11 probe is referenced "
        "read-only and was not remeasured during persistence.",
        "",
        "LLM calls: %d / %d." % (
            payload.get("llm_call_count", 0), payload.get("llm_call_budget", 0)
        ),
    ])
    if payload.get("stopped_early"):
        lines.extend(["", "Stopped: %s" % payload["stopped_early"]])
    else:
        lines.extend([
            "",
            "No unresolved ambiguity stopped the run. The task_06 delayed "
            "reading participated in its own adoption gate, so it did not "
            "replace #11's independent task_05 activation evidence.",
        ])
    return "\n".join(lines) + "\n"


def _write_outputs(payload: Mapping[str, Any]) -> None:
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("overall", payload["overall_verdict"], flush=True)
    print("llm_calls", payload.get("llm_call_count", 0), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="persist and naturally resolve stores with 0 LLM, then stop",
    )
    args = parser.parse_args(argv)
    return run(preflight_only=bool(args.preflight_only))


if __name__ == "__main__":
    raise SystemExit(main())
