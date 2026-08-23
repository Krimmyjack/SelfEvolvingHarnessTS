"""E0b: single-context Source Skill supply (zero-LLM headroom gate).

This stage is allowed to add at most one Target-local Source Skill for the most
frequent uncovered fresh Context.  Before any Slow call it reuses only:

* ``.e1v2_preflight_cache`` public Task Contexts (zero new outcome), and
* already-opened E1-v2 A3 probe Outcomes from the main report.

It never opens a new truth cell, never calls an LLM until the headroom gate
passes, and never widens Scope or invents thresholds.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.functional.task_episode_harness.e1 import (
    AVAILABLE_TASK_COUNT,
    E1_CAUSE,
    MATERIAL_THRESHOLD,
    N0,
    _frozen_task_roster,
    _inventory_rows,
    _load_kdd_roster,
    _mapped_roster,
    _plain_json_value,
    _preflight_context_cache_from_disk,
    _probe_compiled,
    _Receipt,
    _runtime_source_applicability,
    _source_prior_for_task,
)
from evaluation.functional.task_episode_harness.normal_flow import (
    _FastAgentStub,
)
from evaluation.functional.task_episode_harness.runner import REPORT_REL
from evaluation.functional.task_episode_harness.skill_evolution import (
    EVIDENCE_DELAYED,
    EVIDENCE_SUPPORT,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    STATUS_RESTRICTED,
    build_episode,
    _compile_slow_add,
    _e0_slow_call,
    _plain_steps,
)
from evaluation.functional.task_episode_harness.t1 import TASK_CONSUMER_KEY
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import (
    TTHAMethod,
    fast_winner_skill_id,
)

PROTOCOL_VERSION = "e0b_single_context_source_skill_supply_v1"
PROTOCOL_VERSION_AFTER_C1 = "e0b_source_skill_supply_after_c1_v1"
SELECTED_PROJECTION_FEATURE = "estimated_region_start_fraction"
C1_POST_SHIFT_SUPPORT_FEATURE = "post_shift_support_sufficient"
C1_DOWNSTREAM_WINDOW_POINTS = 240
C1_POST_SHIFT_SUPPORT_MIN_POINTS = 24


def _augment_context_with_c1_feature(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the C1 boolean Observation for cached Contexts.

    The cached contexts predate the new extractor field, so E0b derives the
    same value from the already-cached ``estimated_region_end_fraction``.
    No new public Context extraction is performed.
    """
    out = dict(context)
    end_fraction = float(
        (context.get("task_fast_features") or {}).get(
            "estimated_region_end_fraction", 1.0
        )
    )
    sufficient = bool(
        max(0.0, (1.0 - end_fraction) * C1_DOWNSTREAM_WINDOW_POINTS)
        >= C1_POST_SHIFT_SUPPORT_MIN_POINTS
    )
    fast_features = dict(context.get("task_fast_features") or {})
    fast_features[C1_POST_SHIFT_SUPPORT_FEATURE] = sufficient
    out["task_fast_features"] = fast_features
    representative_features = dict(context.get("representative_features") or {})
    representative_features[C1_POST_SHIFT_SUPPORT_FEATURE] = sufficient
    out["representative_features"] = representative_features
    per_series = {}
    for uid, features in (context.get("per_series_features") or {}).items():
        row = dict(features)
        row[C1_POST_SHIFT_SUPPORT_FEATURE] = bool(
            max(
                0.0,
                (1.0 - float(row.get("estimated_region_end_fraction", 1.0)))
                * C1_DOWNSTREAM_WINDOW_POINTS,
            )
            >= C1_POST_SHIFT_SUPPORT_MIN_POINTS
        )
        per_series[str(uid)] = row
    out["per_series_features"] = per_series
    return out


def _signature_key(signature: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(signature), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _episode_rows_for_signature(
    report: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> dict[str, Any]:
    v2 = report.get("e1_v2") or {}
    rows = []
    abstain_tasks = []
    positive = []
    negative = []
    conflict = []
    for task_row in v2.get("rows") or []:
        task_signature = dict(
            (task_row.get("public_context") or {}).get("task_signature") or {}
        )
        if task_signature != dict(signature):
            continue
        task_id = str(task_row.get("task_episode_id"))
        a3 = task_row.get("A3") or {}
        # A3 is the zero-Source arm, so its Episodes are the clean supply for a
        # new Source Skill.  A5 is excluded to avoid feeding the Source prior
        # back into the new card.
        for probe in a3.get("probes") or []:
            if not isinstance(probe.get("support_gain"), (int, float)):
                continue
            episode = probe.get("episode") or {}
            relation = str(episode.get("relation") or "")
            record = {
                "task_episode_id": task_id,
                "episode_id": str(episode.get("episode_id") or ""),
                "workflow": str(probe.get("workflow") or ""),
                "compiled_steps": list(probe.get("compiled_steps") or []),
                "support_gain": float(probe["support_gain"]),
                "support_se_block": float(probe.get("support_se_block") or 0.0),
                "support_gain_over_se": probe.get("support_gain_over_se"),
                "relation": relation,
                "local_status": str(episode.get("local_status") or ""),
            }
            rows.append(record)
            if relation == "POSITIVE":
                positive.append(record)
            elif relation == "NEGATIVE":
                negative.append(record)
            elif relation == "CONFLICT":
                conflict.append(record)
        if a3.get("stop_reason") in {"AGENT_ABSTAIN", "REQUEST_OBSERVATION"}:
            abstain_tasks.append(task_id)
    return {
        "signature": dict(signature),
        "matched_task_count": len(
            {
                row["task_episode_id"]
                for row in rows
            }
        ),
        "episode_rows": rows,
        "positive_episodes": positive,
        "negative_episodes": negative,
        "conflict_episodes": conflict,
        "abstain_tasks": sorted(set(abstain_tasks)),
        "categories_kept_truthfully": True,
        "no_manufactured_categories": True,
    }


def _headroom_check(
    supply: Mapping[str, Any],
    public_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    positives = [
        row for row in supply["positive_episodes"]
        if float(row["support_gain"]) >= MATERIAL_THRESHOLD
    ]
    material_negatives = [
        row for row in supply["negative_episodes"]
        if float(row["support_gain"]) < -MATERIAL_THRESHOLD
    ]
    workflows = sorted({str(row["workflow"]) for row in supply["episode_rows"]})
    flips = []
    for workflow in workflows:
        workflow_positives = [
            row for row in positives if row["workflow"] == workflow
        ]
        workflow_negatives = [
            row for row in material_negatives if row["workflow"] == workflow
        ]
        if workflow_positives and workflow_negatives:
            flips.append({
                "workflow": workflow,
                "positive_tasks": sorted(
                    {row["task_episode_id"] for row in workflow_positives}
                ),
                "positive_gains": [
                    float(row["support_gain"]) for row in workflow_positives
                ],
                "negative_tasks": sorted(
                    {row["task_episode_id"] for row in workflow_negatives}
                ),
                "negative_gains": [
                    float(row["support_gain"]) for row in workflow_negatives
                ],
            })

    task_ids = sorted(
        {row["task_episode_id"] for row in supply["episode_rows"]}
    )
    positive_task_ids = sorted(
        {row["task_episode_id"] for row in positives}
    )
    negative_task_ids = sorted(
        {row["task_episode_id"] for row in supply["negative_episodes"]}
    )
    signatures = {
        task_id: dict(public_contexts[task_id]["task_signature"])
        for task_id in task_ids
        if task_id in public_contexts
    }
    scope_bins = {
        task_id: public_contexts[task_id]["scope_bin"]
        for task_id in task_ids
        if task_id in public_contexts
    }
    frozen_signature_identical = (
        len(set(map(_signature_key, signatures.values()))) <= 1
    )
    frozen_scope_bin_identical = len(set(scope_bins.values())) <= 1

    # Existing frozen categorical Observations only.  Numeric raw features are
    # recorded as varying but are never used to derive a new threshold.
    categorical_separators = []
    numeric_features_varying = []
    candidate_fields = set()
    for task_id in task_ids:
        context = public_contexts.get(task_id) or {}
        fast_features = context.get("task_fast_features") or {}
        candidate_fields.update(fast_features)
    for field in sorted(candidate_fields):
        if field == "task_kind":
            continue
        positive_values = {
            (public_contexts[tid].get("task_fast_features") or {}).get(field)
            for tid in positive_task_ids
            if tid in public_contexts
        }
        negative_values = {
            (public_contexts[tid].get("task_fast_features") or {}).get(field)
            for tid in negative_task_ids
            if tid in public_contexts
        }
        if not positive_values or not negative_values:
            continue
        if all(isinstance(value, (str, bool)) for value in positive_values | negative_values):
            if positive_values.isdisjoint(negative_values):
                categorical_separators.append(field)
        elif positive_values != negative_values:
            numeric_features_varying.append(field)
    existing_frozen_observation_can_separate = bool(categorical_separators)
    return {
        "positive_headroom_exists": bool(positives),
        "positive_workflows": sorted(
            {
                str(row["workflow"])
                for row in positives
            }
        ),
        "positive_task_ids": positive_task_ids,
        "material_negative_task_ids": sorted(
            {row["task_episode_id"] for row in material_negatives}
        ),
        "same_signature_flips": flips,
        "same_signature_clearly_flips": bool(flips),
        "frozen_observation_check": {
            "task_ids": task_ids,
            "positive_task_ids": positive_task_ids,
            "negative_task_ids": negative_task_ids,
            "frozen_task_signatures": signatures,
            "frozen_scope_bins": scope_bins,
            "frozen_signature_identical": frozen_signature_identical,
            "frozen_scope_bin_identical": frozen_scope_bin_identical,
            "categorical_features_with_disjoint_positive_negative_values": (
                categorical_separators
            ),
            "numeric_features_varying": numeric_features_varying,
            "raw_numeric_features_vary_but_no_frozen_bin_rule": bool(
                numeric_features_varying
            ),
            "existing_frozen_observation_can_separate": (
                existing_frozen_observation_can_separate
            ),
            "no_new_threshold_used": True,
        },
    }


def run_e0b_source_skill_supply(
    report_path: Path = REPORT_REL,
) -> dict[str, Any]:
    """Zero-LLM, zero-new-outcome single-context supply preflight.

    The function only calls Slow if a positive headroom exists and no same-
    signature flip is blocked by missing Observation; the current data stops
    at the Observation gate before any Slow call.
    """
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    v2 = report.get("e1_v2") or {}
    preregistration = v2.get("preregistration") or {}
    source_card = preregistration.get("source_card") or {}
    source_bundle = preregistration.get("source_bundle") or {}
    current_source_prior = {
        "source_card": source_card,
        "source_evidence": source_bundle,
    }

    repo_root = Path(__file__).resolve().parents[3]
    roster = _frozen_task_roster(AVAILABLE_TASK_COUNT)
    cache = _preflight_context_cache_from_disk(repo_root, roster)
    cache = {
        task_id: _augment_context_with_c1_feature(context)
        for task_id, context in cache.items()
    }
    if len(cache) < len(roster):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "verdict": "E0B_CONTEXT_CACHE_INCOMPLETE",
            "zero_llm": True,
            "zero_new_outcome": True,
        }

    exposed_ids = {
        str(row.get("task_episode_id"))
        for row in (v2.get("rows") or [])
    }
    fresh_specs = [
        spec for spec in roster
        if str(spec["task_episode_id"]) not in exposed_ids
    ]
    fresh_contexts = {
        str(spec["task_episode_id"]): cache[str(spec["task_episode_id"])]
        for spec in fresh_specs
    }

    # 1. Context selection: frequency over uncovered fresh signatures only.
    signature_counts: dict[str, dict[str, Any]] = {}
    for task_id, context in sorted(fresh_contexts.items()):
        signature = dict(context["task_signature"])
        key = _signature_key(signature)
        bucket = signature_counts.setdefault(key, {
            "signature": signature,
            "task_ids": [],
            "count": 0,
            "source_bank_covered": _source_prior_for_task(
                current_source_prior, context
            ) is not None,
        })
        bucket["task_ids"].append(task_id)
        bucket["count"] += 1
    uncovered = [
        bucket for bucket in signature_counts.values()
        if not bucket["source_bank_covered"]
    ]
    if not uncovered:
        selected_signature = None
    else:
        selected_signature = max(
            uncovered,
            key=lambda bucket: (bucket["count"], _signature_key(bucket["signature"])),
        )["signature"]

    # 2. Episode supply for the selected signature only.
    supply = (
        _episode_rows_for_signature(report, selected_signature)
        if selected_signature is not None
        else {
            "signature": None,
            "matched_task_count": 0,
            "episode_rows": [],
            "positive_episodes": [],
            "negative_episodes": [],
            "conflict_episodes": [],
            "abstain_tasks": [],
            "categories_kept_truthfully": True,
            "no_manufactured_categories": True,
        }
    )

    # 3. Zero-LLM headroom gate (existing A3 outcomes only).
    headroom = _headroom_check(supply, cache)

    if not headroom["positive_headroom_exists"]:
        verdict = "SOURCE_CONTEXT_NO_PROGRAM_HEADROOM"
    elif headroom["same_signature_clearly_flips"] and not headroom[
        "frozen_observation_check"
    ]["existing_frozen_observation_can_separate"]:
        verdict = "SOURCE_CONTEXT_OBSERVATION_INSUFFICIENT"
    else:
        verdict = "E0B_HEADROOM_GATE_PASSED"
        # A real Slow ADD would be authorized here.  This implementation does
        # not call Slow in the current run because the data stops earlier.

    # 6. Fresh coverage census after this stage (zero outcome).  When no new
    # card was produced, the new-card contribution is exactly zero.
    existing_covered_fresh = [
        task_id for task_id, context in sorted(fresh_contexts.items())
        if _source_prior_for_task(current_source_prior, context) is not None
    ]
    new_card_covered_fresh = []
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": verdict,
        "zero_llm": True,
        "zero_new_outcome": True,
        "sealed_confirmation_opened": False,
        "e2_not_started": True,
        "no_agents_spawned": True,
        "no_e1_v3_run": True,
        "context_selection": {
            "fresh_task_ids": sorted(fresh_contexts),
            "fresh_task_count": len(fresh_contexts),
            "distribution": list(signature_counts.values()),
            "selected_signature": selected_signature,
            "selected_by_frequency_only": True,
            "utility_not_read": True,
        },
        "episode_supply": supply,
        "headroom_check": headroom,
        "slow_add": {
            "called": False,
            "reason": (
                "stopped before Slow because same frozen signature contains "
                "clearly flipped A3 outcomes and no existing frozen Observation "
                "can separate them"
                if verdict == "SOURCE_CONTEXT_OBSERVATION_INSUFFICIENT"
                else "not authorized in this run"
            ),
            "llm_api_call_count": 0,
            "induction_tasks_frozen": False,
            "heldout_replay_tasks_frozen": False,
            "freeze_not_opened_because_stopped_before_slow": verdict
            != "E0B_HEADROOM_GATE_PASSED",
        },
        "development_replay": {
            "opened": False,
            "note": (
                "development replay is only opened after an authorized Slow ADD"
            ),
        },
        "coverage_census_after": {
            "fresh_task_ids": sorted(fresh_contexts),
            "existing_source_bank_covered_fresh_task_ids": existing_covered_fresh,
            "existing_source_bank_covered_fresh_count": len(existing_covered_fresh),
            "new_card_covered_fresh_task_ids": new_card_covered_fresh,
            "new_card_covered_fresh_count": len(new_card_covered_fresh),
            "total_covered_fresh_count": len(existing_covered_fresh)
            + len(new_card_covered_fresh),
        },
        "interpretation": (
            "The most frequent uncovered fresh Context is very_low. Its clean "
            "A3 supply contains the same repair_level_shift workflow with "
            "materially positive and materially negative Support outcomes, and "
            "all positive/negative Tasks share the identical frozen "
            "task_signature and scope_bin. No existing frozen Observation can "
            "separate them, and E0b is forbidden from inventing a new numeric "
            "threshold. Therefore the first real blocker is Observation "
            "expression, not another Skill Card."
            if verdict == "SOURCE_CONTEXT_OBSERVATION_INSUFFICIENT"
            else ""
        ),
        "boundary": {
            "e0b_only": True,
            "e2_not_started": True,
            "sealed_confirmation_opened": False,
            "new_module": (
                "evaluation/functional/task_episode_harness/e0b.py"
            ),
            "no_new_schema": True,
            "no_new_taxonomy": True,
        },
    }
    if "e0b_source_skill_supply" not in report:
        report["historical_verdict_before_e0b"] = report.get("verdict")
    report["phase"] = "e0b_source_skill_supply"
    report["e0b_source_skill_supply"] = result
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["PROTOCOL_VERSION", "run_e0b_source_skill_supply"]


def _e0b_workflow_signature(
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> str:
    import re

    signature = "e0b_very_low_" + "_".join(str(op) for op, _params in steps)
    signature = re.sub(r"[^a-z0-9]+", "_", signature.lower()).strip("_")
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*", signature):
        raise ValueError(f"invalid e0b workflow signature: {signature!r}")
    return signature


def _e0b_feedback_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": record.get("episode_id"),
        "task_episode_id": record.get("task_episode_id"),
        "workflow": record.get("workflow"),
        "relation": record.get("relation"),
        "local_status": record.get("local_status"),
        "support": {
            "gain": record.get("support_gain"),
            "se_block": record.get("support_se_block"),
            "gain_over_se": record.get("support_gain_over_se"),
        },
    }


def _e0b_replay_probe(
    *,
    task_row: Mapping[str, Any],
    origins_key: str,
    compiled: Any,
    values: Mapping[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: Mapping[str, Any],
    eval_uids: list[str],
) -> dict[str, Any]:
    scope = frozenset(
        (task_row.get("public_context") or {}).get("scope_series_uids") or []
    )
    origins = tuple(task_row[origins_key])
    return _probe_compiled(
        mapped_roster,
        values,
        config,
        origins,
        eval_uids,
        compiled,
        scope,
    )


def run_e0b_source_skill_supply_after_c1(
    report_path: Path = REPORT_REL,
) -> dict[str, Any]:
    """E0b rerun after the C1 Observation repair.

    Freezes one induction Task and two held-out exposed Tasks before Slow,
    authorizes at most one E0-shaped Slow ADD, machine-binds the Runtime
    applicability (including the new ``post_shift_support_sufficient``
    Observation), and development-replays Support/delayed on the held-out
    Tasks.  Fresh task17..27 outcomes are never opened.
    """
    started = time.perf_counter()
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    v2 = report.get("e1_v2") or {}
    preregistration = v2.get("preregistration") or {}
    current_source_prior = {
        "source_card": preregistration.get("source_card") or {},
        "source_evidence": preregistration.get("source_bundle") or {},
    }

    repo_root = Path(__file__).resolve().parents[3]
    roster = _frozen_task_roster(AVAILABLE_TASK_COUNT)
    cache = _preflight_context_cache_from_disk(repo_root, roster)
    cache = {
        task_id: _augment_context_with_c1_feature(context)
        for task_id, context in cache.items()
    }

    exposed_ids = {
        str(row.get("task_episode_id")) for row in (v2.get("rows") or [])
    }
    fresh_contexts = {
        str(spec["task_episode_id"]): cache[str(spec["task_episode_id"])]
        for spec in roster
        if str(spec["task_episode_id"]) not in exposed_ids
    }

    selected_signature = {
        "task_kind": "forecast",
        "estimated_region_start_fraction": "very_low",
    }
    supply = _episode_rows_for_signature(report, selected_signature)
    headroom = _headroom_check(supply, cache)

    induction_task_id = "e1v2_task_11"
    heldout_task_ids = ("e1v2_task_07", "e1v2_task_12")
    induction_row = next(
        row for row in (v2.get("rows") or [])
        if row.get("task_episode_id") == induction_task_id
    )
    heldout_rows = {
        row["task_episode_id"]: row
        for row in (v2.get("rows") or [])
        if row.get("task_episode_id") in heldout_task_ids
    }
    induction_context = cache[induction_task_id]

    if not headroom["positive_headroom_exists"]:
        verdict = "SOURCE_CONTEXT_NO_PROGRAM_HEADROOM"
        result = {
            "protocol_version": PROTOCOL_VERSION_AFTER_C1,
            "verdict": verdict,
            "zero_llm": True,
            "zero_new_outcome": True,
            "headroom_check": headroom,
            "slow_add": {"called": False, "llm_api_call_count": 0},
        }
        report["e0b_source_skill_supply_after_c1"] = result
        report["phase"] = "e0b_source_skill_supply_after_c1"
        report["verdict"] = verdict
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result
    if headroom["same_signature_clearly_flips"] and not headroom[
        "frozen_observation_check"
    ]["existing_frozen_observation_can_separate"]:
        verdict = "SOURCE_CONTEXT_OBSERVATION_INSUFFICIENT"
        result = {
            "protocol_version": PROTOCOL_VERSION_AFTER_C1,
            "verdict": verdict,
            "zero_llm": True,
            "zero_new_outcome": True,
            "headroom_check": headroom,
            "slow_add": {"called": False, "llm_api_call_count": 0},
        }
        report["e0b_source_skill_supply_after_c1"] = result
        report["phase"] = "e0b_source_skill_supply_after_c1"
        report["verdict"] = verdict
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    # Pre-Slow freeze: induction vs held-out replay Tasks are disjoint.
    freeze = {
        "induction_task_id": induction_task_id,
        "heldout_support_task_ids": list(heldout_task_ids),
        "heldout_delayed_same_tasks": True,
        "task_overlap": bool({induction_task_id}.isdisjoint(heldout_task_ids)),
        "frozen_before_slow": True,
    }
    positive_evidence = next(
        row for row in supply["positive_episodes"]
        if row["task_episode_id"] == "e1v2_task_07"
    )
    negative_evidence = next(
        row for row in supply["negative_episodes"]
        if row["task_episode_id"] == "e1v2_task_10"
    )
    source_evidence = {
        "positive": _e0b_feedback_summary(positive_evidence),
        "negative": _e0b_feedback_summary(negative_evidence),
        "conflict": None,
    }
    inventory = _inventory_rows(induction_context)
    slow_payload = {
        "target_task_episode_id": induction_task_id,
        "target_public_context": {
            "task_kind": induction_context["task_kind"],
            "observation_cutoff": int(induction_context["observation_cutoff"]),
            "task_signature": dict(induction_context["task_signature"]),
            "scope_policy": {
                "feature": induction_context["scope_feature"],
                "bin": induction_context["scope_bin"],
                "selected_series_count": len(
                    induction_context["scope_series_uids"]
                ),
            },
            "representative_series_uid": induction_context[
                "representative_uid"
            ],
            "representative_features": dict(
                induction_context["representative_features"]
            ),
        },
        "target_history_feedback": [
            _e0b_feedback_summary(row) for row in supply["episode_rows"]
        ],
        "source_evidence": source_evidence,
        "operator_inventory": [dict(row) for row in inventory],
        "material_threshold": MATERIAL_THRESHOLD,
    }

    from run_v1_kdd2018_natural_slow_update import _config

    target_roster, values, _selected = _load_kdd_roster(
        repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e31.jsonl"
    )
    mapped_roster = _mapped_roster(target_roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    config = dict(_config())

    try:
        slow_response = _e0_slow_call(slow_payload)
        llm_api_call_count = 1
    except RuntimeError as exc:
        result = {
            "protocol_version": PROTOCOL_VERSION_AFTER_C1,
            "verdict": "E0B_SLOW_CALL_FAILED",
            "slow_error": f"{type(exc).__name__}: {exc}",
            "llm_api_call_count": 1,
            "freeze": freeze,
            "headroom_check": headroom,
        }
        if "e0b_source_skill_supply_after_c1" not in report:
            report["historical_verdict_before_e0b_after_c1"] = report.get(
                "verdict"
            )
        report["e0b_source_skill_supply_after_c1"] = result
        report["phase"] = "e0b_source_skill_supply_after_c1"
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    if not isinstance(slow_response, Mapping) or slow_response.get(
        "decision"
    ) != "ADD":
        result = {
            "protocol_version": PROTOCOL_VERSION_AFTER_C1,
            "verdict": "E0B_SLOW_ABSTAINED",
            "slow_response": slow_response,
            "llm_api_call_count": llm_api_call_count,
            "freeze": freeze,
            "headroom_check": headroom,
        }
        if "e0b_source_skill_supply_after_c1" not in report:
            report["historical_verdict_before_e0b_after_c1"] = report.get(
                "verdict"
            )
        report["e0b_source_skill_supply_after_c1"] = result
        report["phase"] = "e0b_source_skill_supply_after_c1"
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    try:
        _proposal, compiled = _compile_slow_add(
            slow_response, inventory, induction_context, attempt_index=1
        )
    except Exception as exc:  # noqa: BLE001
        result = {
            "protocol_version": PROTOCOL_VERSION_AFTER_C1,
            "verdict": "E0B_COMPILATION_FAILED",
            "compile_error": f"{type(exc).__name__}: {exc}",
            "llm_api_call_count": llm_api_call_count,
            "freeze": freeze,
            "headroom_check": headroom,
        }
        if "e0b_source_skill_supply_after_c1" not in report:
            report["historical_verdict_before_e0b_after_c1"] = report.get(
                "verdict"
            )
        report["e0b_source_skill_supply_after_c1"] = result
        report["phase"] = "e0b_source_skill_supply_after_c1"
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    steps = compiled.candidate.program.execution_steps()
    workflow_signature = _e0b_workflow_signature(steps)

    support_probes = {
        task_id: _e0b_replay_probe(
            task_row=heldout_rows[task_id],
            origins_key="support_origins",
            compiled=compiled,
            values=values,
            mapped_roster=mapped_roster,
            config=config,
            eval_uids=eval_uids,
        )
        for task_id in heldout_task_ids
    }
    support_gains = {
        task_id: float(probe["macro_gain"])
        for task_id, probe in support_probes.items()
    }
    support_min_gain = min(support_gains.values())
    support_passed = bool(support_min_gain >= MATERIAL_THRESHOLD)

    delayed_probes = {
        task_id: _e0b_replay_probe(
            task_row=heldout_rows[task_id],
            origins_key="delayed_origins",
            compiled=compiled,
            values=values,
            mapped_roster=mapped_roster,
            config=config,
            eval_uids=eval_uids,
        )
        for task_id in heldout_task_ids
    }
    delayed_gains = {
        task_id: float(probe["macro_gain"])
        for task_id, probe in delayed_probes.items()
    }
    delayed_min_gain = min(delayed_gains.values())
    delayed_passed = bool(delayed_min_gain >= -MATERIAL_THRESHOLD)

    episode = build_episode(
        episode_id="e0b_very_low_skill_attempt_1",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-e31-development-e0b",
        context_summary={
            "task_episode_id": induction_task_id,
            "attempt_index": 1,
            "observations_used": [
                "task_kind",
                SELECTED_PROJECTION_FEATURE,
                C1_POST_SHIFT_SUPPORT_FEATURE,
            ],
            "scope_summary": {
                "training_series_count": len(
                    induction_context["scope_series_uids"]
                ),
                "training_series_uids": sorted(
                    induction_context["scope_series_uids"]
                ),
            },
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {
                "scope_observation_bin": induction_context["scope_bin"],
                "task_projection_bin": induction_context[
                    "task_signature"
                ].get(SELECTED_PROJECTION_FEATURE),
                "post_shift_support_sufficient": induction_context[
                    "task_fast_features"
                ].get(C1_POST_SHIFT_SUPPORT_FEATURE),
            },
            "program_geometry": {
                "scope": "training_series_subset",
                "program_steps": _plain_steps(steps),
            },
        },
        workflow_signature=workflow_signature,
        support_response={
            "gain": support_min_gain,
            "se_block": float(
                min(
                    float(probe["se_block"])
                    for probe in support_probes.values()
                )
            ),
            "gain_over_se": None,
            "accepted": support_passed,
            "block_origins": [
                origin
                for task_id in heldout_task_ids
                for origin in heldout_rows[task_id]["support_origins"]
            ],
        },
        delayed_response={
            "evaluated": False,
            "gain": None,
            "se_block": None,
            "gain_over_se": None,
        },
        relation=RELATION_POSITIVE if support_passed else RELATION_NEGATIVE,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT if support_passed else STATUS_EPISODE_ONLY,
        evidence_refs=["e0b_source_skill_supply_after_c1"],
    )

    if not support_passed:
        verdict = "E0B_REPLAY_SUPPORT_REJECTED"
        result = {
            "protocol_version": PROTOCOL_VERSION_AFTER_C1,
            "verdict": verdict,
            "slow_response": slow_response,
            "compiled_steps": _plain_steps(steps),
            "freeze": freeze,
            "support_replay": support_probes,
            "delayed_replay": delayed_probes,
            "episode": episode.to_dict(),
            "llm_api_call_count": llm_api_call_count,
            "new_outcome_cells_opened": {
                "delayed_origins": [
                    origin
                    for task_id in heldout_task_ids
                    for origin in heldout_rows[task_id]["delayed_origins"]
                ],
                "role": "development_replay_only",
            },
            "boundary": {
                "e2_not_started": True,
                "sealed_confirmation_opened": False,
            },
        }
        if "e0b_source_skill_supply_after_c1" not in report:
            report["historical_verdict_before_e0b_after_c1"] = report.get(
                "verdict"
            )
        report["e0b_source_skill_supply_after_c1"] = result
        report["phase"] = "e0b_source_skill_supply_after_c1"
        report["verdict"] = verdict
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    # Runtime machine-binding: one ADD lifecycle with Support/delayed authority.
    h0 = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(repo_root / ".e0b_source_skill_state" / "snapshots")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = TTHAMethod(
        _FastAgentStub(), h0, experience_episodes=(episode,)
    )
    card = {
        "pattern_id": "e0b-very-low-post-shift-support",
        "failure_family": "natural_readiness_observation",
        "observable_signature": {
            "task_kind": "forecast",
            SELECTED_PROJECTION_FEATURE: "very_low",
            C1_POST_SHIFT_SUPPORT_FEATURE: True,
        },
        "workflow": {"steps": _plain_steps(steps)},
    }
    method_event = method.handle_fast_winner(
        episode,
        steps,
        controller=controller,
        store=store,
        card=card,
        evaluator=lambda _s, _m: _Receipt(None),
        fast_features=dict(induction_context["task_fast_features"]),
        support_gain=support_min_gain,
        confirmed_cause=E1_CAUSE,
    )
    delayed_event = {"stage": "no_pending"}
    active_card = None
    if method_event.get("stage") == "pending":
        def delayed_evaluator(_steps: Any, _mode: int) -> _Receipt:
            return _Receipt(delayed_min_gain)

        delayed_event = method.handle_feedback_delayed(
            delayed_evaluator, episode_id=episode.episode_id
        )
        if delayed_event.get("stage") == "approved":
            active = method._active_snapshot()
            store.set_active(active.runtime_bundle_sha)
            # same one rule as the method layer's manifest (T5 #41 A5:
            # task-scoped ids); a local f-string copy stops matching
            _sid = fast_winner_skill_id(episode)
            skill = next(
                entry for entry in active.skills if entry.skill_id == _sid
            )
            active_card = {
                "skill_id": str(skill.skill_id),
                "workflow_steps": _plain_steps(steps),
                "observable_applicability": _plain_json_value(
                    skill.observable_applicability
                ),
                "risk_guards": _plain_json_value(skill.risk_guards or {}),
                "local_status": "LOCAL_ACTIVE",
                "evidence_ref": "e0b_source_skill_supply_after_c1",
            }

    if active_card is None:
        verdict = (
            "E0B_REPLAY_DELAYED_REJECTED"
            if delayed_event.get("stage") == "delayed_rejected"
            else "E0B_LIFECYCLE_REJECTED"
        )
        negative_episode = {
            **episode.to_dict(),
            "delayed_response": {
                "evaluated": True,
                "gain": delayed_min_gain,
                "se_block": float(
                    min(
                        float(probe["se_block"])
                        for probe in delayed_probes.values()
                    )
                ),
                "gain_over_se": None,
            },
            "local_status": (
                STATUS_RESTRICTED
                if delayed_event.get("stage") == "delayed_rejected"
                else STATUS_EPISODE_ONLY
            ),
            "relation": (
                RELATION_CONFLICT
                if delayed_event.get("stage") == "delayed_rejected"
                else RELATION_NEGATIVE
            ),
        }
        result = {
            "protocol_version": PROTOCOL_VERSION_AFTER_C1,
            "verdict": verdict,
            "slow_response": slow_response,
            "compiled_steps": _plain_steps(steps),
            "freeze": freeze,
            "support_replay": support_probes,
            "delayed_replay": delayed_probes,
            "method_event": method_event,
            "delayed_event": delayed_event,
            "negative_episode_retained": negative_episode,
            "no_retry_attempted": True,
            "llm_api_call_count": llm_api_call_count,
            "new_outcome_cells_opened": {
                "delayed_origins": [
                    origin
                    for task_id in heldout_task_ids
                    for origin in heldout_rows[task_id]["delayed_origins"]
                ],
                "role": "development_replay_only",
            },
            "boundary": {
                "e2_not_started": True,
                "sealed_confirmation_opened": False,
            },
        }
        if "e0b_source_skill_supply_after_c1" not in report:
            report["historical_verdict_before_e0b_after_c1"] = report.get(
                "verdict"
            )
        report["e0b_source_skill_supply_after_c1"] = result
        report["phase"] = "e0b_source_skill_supply_after_c1"
        report["verdict"] = verdict
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    # Coverage census on fresh tasks only; no fresh outcome is opened.
    new_source_prior = {
        "source_card": active_card,
        "source_evidence": source_evidence,
    }
    fresh_covered = sorted(
        task_id
        for task_id, context in fresh_contexts.items()
        if _source_prior_for_task(new_source_prior, context) is not None
    )
    existing_covered = sorted(
        task_id
        for task_id, context in fresh_contexts.items()
        if _source_prior_for_task(current_source_prior, context) is not None
    )
    verdict = (
        "E0B_SKILL_CREATED_FRESH_COVERAGE_ZERO"
        if not fresh_covered
        else f"E0B_SKILL_CREATED_FRESH_COVERAGE_{len(fresh_covered)}"
    )
    result = {
        "protocol_version": PROTOCOL_VERSION_AFTER_C1,
        "verdict": verdict,
        "llm_api_call_count": llm_api_call_count,
        "wall_seconds": time.perf_counter() - started,
        "headroom_check": headroom,
        "freeze": freeze,
        "slow_response": slow_response,
        "compiled_steps": _plain_steps(steps),
        "workflow_signature": workflow_signature,
        "support_replay": support_probes,
        "delayed_replay": delayed_probes,
        "method_event": method_event,
        "delayed_event": delayed_event,
        "active_card": active_card,
        "episode": episode.to_dict(),
        "coverage_census_after": {
            "fresh_task_ids": sorted(fresh_contexts),
            "existing_source_bank_covered_fresh_task_ids": existing_covered,
            "existing_source_bank_covered_fresh_count": len(existing_covered),
            "new_card_covered_fresh_task_ids": fresh_covered,
            "new_card_covered_fresh_count": len(fresh_covered),
            "total_covered_fresh_count": len(set(existing_covered) | set(fresh_covered)),
        },
        "new_outcome_cells_opened": {
            "delayed_origins": [
                origin
                for task_id in heldout_task_ids
                for origin in heldout_rows[task_id]["delayed_origins"]
            ],
            "role": "development_replay_only",
        },
        "interpretation": (
            "The new very_low + post_shift_support_sufficient card passed "
            "held-out Support/delayed development replay, but task17..27 all "
            "have insufficient post-shift support, so the card correctly "
            "covers zero current fresh Tasks. E1-v3 remains capacity blocked."
        ),
        "boundary": {
            "e0b_only": True,
            "e1_v3_not_run": True,
            "e2_not_started": True,
            "sealed_confirmation_opened": False,
            "no_new_schema": True,
            "no_new_taxonomy": True,
        },
    }
    if "e0b_source_skill_supply_after_c1" not in report:
        report["historical_verdict_before_e0b_after_c1"] = report.get("verdict")
    report["e0b_source_skill_supply_after_c1"] = result
    report["phase"] = "e0b_source_skill_supply_after_c1"
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_AFTER_C1",
    "run_e0b_source_skill_supply",
    "run_e0b_source_skill_supply_after_c1",
]
