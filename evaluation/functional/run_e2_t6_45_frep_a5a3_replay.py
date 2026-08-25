"""#45-Frep -- A5 vs A3 development replay of the forecasting forward chain.

Not a fresh confirmation.  This driver re-runs the *already exposed* NOAA 2024
development partition through the in-service forecasting runner
``run_e2_fresh_confirmation`` at current HEAD, to answer one question: does the
full forward chain

    Source Skill -> Target held-in calibration -> Target-local Skill
    -> freeze -> held-out Fast-only deployment -> one-shot scoring

still walk end to end after the #42i/#42k/#42l multi-task wiring surgery?

Every episode, prompt, schema, ladder, gate, Consumer, metric, store and
lifecycle call is the in-service one: this file imports them and supplies the
windows, the run-id-isolated store root and the budget.  No download, no
2025 read, no ``beyond_17520`` read, no new candidate, no new gate.

The held-out block is the latest triple window that fits entirely inside the
exposed 2024 partition, so the protocol *shape* (time split, held-out tail) is
the original one while the data stays development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_fresh_confirmation as fc  # noqa: E402

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "t6_45_frep_a5a3_replay.json"
OUT_MD = E2 / "t6_45_frep_a5a3_replay.md"
PROTOCOL_VERSION = "t6_45_frep_a5a3_replay_v1"
# #45-Frep-b: the symmetric zero-feedback deployment re-adjudication
OUT_JSON_B = E2 / "t6_45_frep_b_symmetric_deploy.json"
OUT_MD_B = E2 / "t6_45_frep_b_symmetric_deploy.md"
PROTOCOL_VERSION_B = "t6_45_frep_b_symmetric_deploy_v1"

# ---- the binding, fixed here before any number of this run was read --------
# held-in: the original protocol's own adaptation block, verbatim
HELD_IN_TASK_A_S = int(fc.TASK_A_S)          # 1104
HELD_IN_PROBE_ORIGINS = tuple(fc.PROBE_ORIGINS)  # 1440 / 1488 / 1536
HELD_IN_TASK_B_S = int(fc.TASK_B_S)          # 1800
# held-out: the latest triple window that fits inside [0, 8760); one backup,
# exactly one window-stride (6 * 48) earlier, used only if the missing gate
# fails.  8472 + 288 = 8760 = the last exposed development index.
HELD_OUT_S = 8472
HELD_OUT_BACKUP_S = 8184
# budget: the original configuration's numbers, unchanged
LLM_CALL_BUDGET_TOTAL = int(fc.LLM_CALL_BUDGET_TOTAL)        # 40
LLM_CALL_BUDGET_PER_EPISODE = int(fc.LLM_CALL_BUDGET_PER_EPISODE)  # 5
LLM_CAP_TASK_BOOK = 200

FRESH_BASELINE = E2 / "fresh_confirmation_v1.json"


# --------------------------------------------------------------- provenance
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _instrument_drift() -> dict[str, Any]:
    """How far HEAD has moved from the surface FRESH_A5_DELIVERS was read on."""
    baseline = json.loads(FRESH_BASELINE.read_text(encoding="utf-8"))
    before = dict(baseline["frozen_surface_before"])
    drift: list[dict[str, Any]] = []
    for rel, recorded in before.items():
        path = PROJECT_ROOT / rel
        now = _sha256(path) if path.is_file() else None
        if now != recorded:
            drift.append({
                "path": rel,
                "sha_at_fresh_confirmation": recorded,
                "sha_at_head": now,
            })
    return {
        "baseline_artifact": "artifacts/functional/e2/fresh_confirmation_v1.json",
        "files": len(before),
        "drifted": len(drift),
        "drift": drift,
        "note": (
            "the original run's own concurrent-write guard compares before/after "
            "*within* one run; this table instead compares HEAD against the "
            "surface the FRESH_A5_DELIVERS reading was taken on, which is the "
            "regression question #45-Frep asks"
        ),
    }


# ------------------------------------------------------------- the binding
def _binding(run_id: str, roster: Mapping[str, Any]) -> dict[str, Any]:
    baseline = json.loads(FRESH_BASELINE.read_text(encoding="utf-8"))
    pre = baseline["pre_registered"]
    return {
        "run_id": run_id,
        "role": (
            "development replay of the forecasting A5 vs A3 forward chain on "
            "current HEAD; a chain regression check, not a capability reading"
        ),
        "archaeology": {
            "original_verdict": baseline["overall_verdict"],
            "original_artifact": (
                "artifacts/functional/e2/fresh_confirmation_v1.json"
            ),
            "original_adjudication": (
                "artifacts/functional/e2/fresh_confirmation_v1_adjudication.md"
            ),
            "original_runner": "evaluation/functional/run_e2_fresh_confirmation.py",
            "original_first_positive_cost_pooled": dict(
                baseline["cells"]["pooled"]["first_positive_cost"]
            ),
            "original_total_cost_pooled": dict(
                baseline["cells"]["pooled"]["total_cost"]
            ),
            "original_llm_calls": int(baseline["llm_call_count"]),
            "original_llm_budget": int(baseline["llm_call_budget"]),
            "original_consumer_retrains_total": int(
                baseline["consumer_retrains_total"]
            ),
            "original_source_skill_ids": [
                str(fc.SKILL_ID[variant]) for variant in fc.CONSUMERS
            ],
            "original_source_skill_provenance": str(pre["stage_1"]),
            "original_protocol_shape": {
                "stage_0": str(pre["stage_0"]),
                "stage_2_held_in": str(pre["stage_2"]),
                "stage_4_held_out": str(pre["stage_4"]),
                "first_positive_cost": str(pre["first_positive_cost"]),
            },
        },
        "entry": {
            "reused_runner": "evaluation/functional/run_e2_fresh_confirmation.py",
            "reused_callables": [
                "stage_0", "stage_1", "stage_2", "stage_4",
                "_missing_gate", "_cell_verdict", "_overall",
                "_first_positive", "_arm_summary", "_window",
                "_assert_window_syntax", "_cohort_artifact",
                "_readability_criteria", "_missing_cap", "_load_development",
                "_cohort_payload", "Budget",
            ],
            "this_driver": (
                "evaluation/functional/run_e2_t6_45_frep_a5a3_replay.py"
            ),
            "driver_supplies_only": [
                "the held-out window start inside the exposed 2024 partition",
                "the run-id isolated store root",
                "the freeze receipt and the deploy binding assertion",
                "the chain verdict mapping",
            ],
            "no_new_platform": (
                "no Consumer, program, prompt, schema, gate, threshold or "
                "candidate was added; stage 1.5 and stage 3 (the 2025 HEAD "
                "check and the 2025 download) are not called at all"
            ),
            "precedent": (
                "run_e2_t6_forecasting_compat_0b.py already imports this "
                "runner's stage callables as the in-service forecasting entry"
            ),
        },
        "target": {
            "cohort": fc.COHORT_NAME,
            "cohort_artifact": "artifacts/functional/e2/noaa_fresh_cohort_v2.json",
            "train_uids": list(roster["train"]),
            "eval_uids": list(roster["eval"]),
            "series_count": len(roster["train"]) + len(roster["eval"]),
            "partition": "development_2024, index [0, 8760)",
            "exposure": (
                "EXPOSED development.  fresh_confirmation_v1's exposure ledger "
                "already records development_2024 outcome=EXPOSED.  This replay "
                "therefore produces no fresh-generalisation evidence and is "
                "labelled INFRASTRUCTURE/MECHANISM only."
            ),
            "sealed_not_touched": [
                "confirmation_2025 [8760, 17520) -- not read, not downloaded",
                "beyond_17520 -- not read",
                "Yahoo S5 A1 (all 67 rows) -- not read",
            ],
        },
        "source_skill_overlap": {
            "source_skill_ids": [
                str(fc.SKILL_ID[variant]) for variant in fc.CONSUMERS
            ],
            "derivation_set": (
                "every frozen delayed evidence row of the recipe line's source "
                "cohorts, compiled by the frozen recipe compiler "
                "(artifacts/functional/e2/recipe_skill_cards_v1.json)"
            ),
            "overlap_with_target_series": "none",
            "why": (
                "the NOAA fresh cohort contributes no evidence row to the card, "
                "so leave-one-cohort-out drops nothing -- quoted from "
                "fresh_confirmation_v1 pre_registered.stage_1"
            ),
            "residual_overlap_surface_disclosed": (
                "the *time* axis overlaps in the weaker sense that both held-in "
                "and held-out blocks are 2024 blocks of the same 16 stations, "
                "and both were already opened by the original run.  Series "
                "identity does not overlap the Source derivation set."
            ),
        },
        "protocol_shape": {
            "same_as_original": True,
            "held_in": {
                "rounds": 2,
                "round_1": "task_A full-price episode at s=%d" % HELD_IN_TASK_A_S,
                "out_of_selection_probe": list(HELD_IN_PROBE_ORIGINS),
                "round_2": "task_B recall-or-research at s=%d" % HELD_IN_TASK_B_S,
                "verbatim_from_original": True,
            },
            "freeze": (
                "the store's active pointer after held-in is the frozen state; "
                "the deploy stage forms no Skill and the pointer is re-read "
                "after deployment and compared byte for byte"
            ),
            "held_out": {
                "split": "time split, held-out tail",
                "start": HELD_OUT_S,
                "backup_start": HELD_OUT_BACKUP_S,
                "rule": (
                    "the latest triple window whose farthest index (s + 288) is "
                    "still <= 8760, i.e. the last block of the exposed "
                    "development partition; one backup exactly one stride "
                    "earlier, used only if the missing gate fails"
                ),
                "difference_from_original": (
                    "the original held-out block was task_C at s=9864 in the "
                    "2025 confirmation year.  2025 is sealed for this task "
                    "book, so the tail of 2024 is used instead.  Same window "
                    "syntax, same horizon, same gate, same scoring; "
                    "development data, therefore replay only."
                ),
                "scoring": "one Fast-only deployment, scored once, no feedback",
            },
        },
        "budget": {
            "llm_call_budget_total_both_arms": LLM_CALL_BUDGET_TOTAL,
            "llm_call_budget_per_episode": LLM_CALL_BUDGET_PER_EPISODE,
            "aligned_with_original": True,
            "task_book_cap": LLM_CAP_TASK_BOOK,
            "within_task_book_cap": LLM_CALL_BUDGET_TOTAL <= LLM_CAP_TASK_BOOK,
            "arm_symmetry": (
                "identical per-episode cap and identical window/roster/Consumer "
                "for both arms; the total pool is shared and consumed in cell "
                "order, exactly as in the original configuration, so the "
                "per-arm spend is reported rather than pre-partitioned"
            ),
            "forecast_retrain_accounting": (
                "Consumer retrains, the original cost unit: every call that "
                "fits the Consumer counts, including identity baselines, the "
                "mask round's per-series work and the delayed gate"
            ),
            "consumer_retrains_are_not_llm_capped": True,
        },
        "stop_rules": {
            "CHAIN_BROKEN(entry)": "the in-service entry cannot be imported or run",
            "CHAIN_BROKEN(stage)": (
                "any mechanical lifecycle failure: no Skill written, freeze "
                "failure, deploy binding assertion failure, or LLM schema "
                "refusals that empty an episode"
            ),
            "no_reroll": "one run only; a failed stage is reported, not retried",
        },
    }


# ------------------------------- #45-Frep-b: the symmetric deploy semantics
# The mainline voided the #45-Frep held-out terminal comparison on the two
# defects that run self-reported.  Both are repaired here, on this driver's
# deployment path only; the in-service runner body is not edited.
#
#   F1  the inherited stage-4 episode consulted the *held-out* delayed reading
#       as an adoption gate (ladder roles `bar` and `confirmation`), which is
#       `open_delayed` inside held-out.  Repaired by never calling the ladder:
#       the deployed Workflow is fixed from frozen state with zero outcome
#       reads, and the single delayed opening that follows is the evaluator's
#       one-shot scoring, which decides nothing.
#
#   F2  an arm with no ACTIVE Skill was routed into a full-price search on the
#       scored block, so the arm with less accumulated knowledge got more
#       search exactly where it was measured.  Repaired by giving both arms the
#       same recall-only deployment: no shortlist, no candidate evaluation, no
#       mask round, no Support confirmation, no LLM.
DEPLOY_RULE: dict[str, Any] = {
    "name": "symmetric_frozen_state_fast_only_deployment",
    "decision_inputs": [
        "the frozen Harness snapshot recompiled from disk",
        "deployment-visible public features of the training series on their "
        "own public prefix, cut at the first held-out support origin",
        "the frozen held-in ledger of the same arm",
    ],
    "decision_reads_no_outcome": True,
    "arm_rule": {
        "has_applicable_active_skill": (
            "Fast-only recall: deploy the Workflow the frozen Skill governs, "
            "read off risk_guards.frozen_plan / frozen_steps"
        ),
        "no_applicable_active_skill": (
            "deploy the standing incumbent of the frozen ledger, i.e. the "
            "final_plan of the last held-in round; identity if that is "
            "identity or absent"
        ),
    },
    "forbidden_on_the_scored_block": [
        "shortlist or any LLM call",
        "candidate evaluation, full-batch Support evaluation, mask round",
        "Support confirmation used as an adoption gate",
        "the v2 adoption ladder and its delayed bar/confirmation reads",
        "any adoption driven by a reading taken on the scored block",
        "Skill formation, approval, limitation or revocation",
    ],
    "scoring": (
        "one delayed_gate opening of the already-fixed applied bytes, on the "
        "three held-out delayed origins, sMASE gain against the identity "
        "baseline; this is the external evaluator's one-shot read and feeds "
        "nothing back"
    ),
    "cost_symmetry_claim": (
        "every arm pays the instrument's identity baselines plus one scoring "
        "read; no arm pays a search"
    ),
}
DEPLOY_SOURCE_ACTIVE = "FROZEN_ACTIVE_SKILL_RECALL"
DEPLOY_SOURCE_INCUMBENT = "FROZEN_LEDGER_INCUMBENT"
DEPLOY_SOURCE_IDENTITY = "FROZEN_LEDGER_NO_INCUMBENT_IDENTITY"


def _frozen_plan_of_skill(skill: Mapping[str, Any],
                          train_uids: Sequence[str]) -> dict[str, Any]:
    """The Workflow a frozen ACTIVE Skill governs.  Validation only, no reads.

    Same validation the in-service ``_direct_recall`` applies before it would
    reuse a recalled plan; the Support confirmation that followed it there is
    exactly what F2 removes.
    """
    guards = dict(skill["risk_guards"])
    frozen = dict(guards.get("frozen_plan") or {})
    steps = list(skill.get("frozen_steps") or [])
    if (
        len(steps) != 1
        or steps[0].get("op") != frozen.get("program")
        or dict(steps[0].get("params") or {})
        or frozen.get("program") not in fc.TREATMENTS
    ):
        raise ValueError("the frozen Skill carries no single valid Workflow")
    plan = {
        "program": str(frozen["program"]),
        "excluded_series": sorted(
            str(uid) for uid in frozen.get("excluded_series") or ()
        ),
    }
    unknown = sorted(set(plan["excluded_series"]) - set(train_uids))
    if unknown:
        raise ValueError("the frozen plan excludes unknown series %s" % unknown)
    return plan


def _frozen_ledger_incumbent(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The Workflow this arm was standing on when it froze.

    The last held-in round's ``final_plan``.  Identity if that round stood on
    identity: the rule does not search backwards for an older positive, since
    that would deploy something the arm was no longer running.
    """
    rounds = [name for name in ("task_B", "task_A") if cell.get(name)]
    last = str(rounds[0]) if rounds else None
    record = (cell.get(last) or {}) if last else {}
    plan = record.get("final_plan") or None
    if not plan or str(plan.get("program")) == fc.IDENTITY:
        return {
            "plan": {"program": fc.IDENTITY, "excluded_series": []},
            "source": DEPLOY_SOURCE_IDENTITY,
            "from_round": last,
            "why": (
                "the last held-in round stood on identity, so the frozen "
                "ledger carries no incumbent Workflow"
            ),
        }
    return {
        "plan": {
            "program": str(plan["program"]),
            "excluded_series": sorted(
                str(uid) for uid in plan.get("excluded_series") or ()
            ),
        },
        "source": DEPLOY_SOURCE_INCUMBENT,
        "from_round": last,
        "why": (
            "the last held-in round adopted this Workflow, so it is the "
            "standing incumbent of the frozen state"
        ),
    }


def _frozen_deploy_decision(
    *, snapshot: Any, search: Any, variant: str, arm: str,
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    """Fix the deployed Workflow from frozen state.  Zero outcome reads."""
    card_id = fc.SKILL_ID[variant] if arm == "A5" else None
    local_id = (cell.get("promotion") or {}).get("retrievable_skill_id")
    _view, retrieval, _context = fc._retrieval(
        snapshot, search, card_id, local_id)
    skill = retrieval.get("local_skill")
    if local_id and skill:
        plan = _frozen_plan_of_skill(skill, search.train_uids)
        return {
            "plan": plan,
            "source": DEPLOY_SOURCE_ACTIVE,
            "active_skill_id": str(local_id),
            "recall_hit": True,
            "lifecycle_at_recall": {
                key: dict(skill["risk_guards"]).get(key)
                for key in (
                    "local_status", "evidence_level",
                    "activation_probe_window", "activation_probe_gain",
                )
            },
            "retrieval": retrieval,
            "why": "Fast-only recall of the frozen ACTIVE Target-local Skill",
        }
    fallback = _frozen_ledger_incumbent(cell)
    fallback.update({
        "active_skill_id": local_id,
        "recall_hit": False if local_id else None,
        "recall_miss": bool(local_id),
        "retrieval": retrieval,
    })
    return fallback


def deploy_fast_only_symmetric(
    payload: Mapping[str, Any], adaptation: Mapping[str, Any],
    window: Mapping[str, Any], *, stores: Mapping[str, Any],
) -> dict[str, Any]:
    """Both arms deploy from frozen state only, then are scored once."""
    cells: dict[str, Any] = {}
    for variant in fc.CONSUMERS:
        target = fc._target(variant)
        for arm in fc.ARMS:
            slot_key = "%s_%s" % (arm.lower(), variant)
            slot = stores[slot_key]
            cell = adaptation["cells"][slot_key]
            search = fc.FreshSearch(
                payload=payload, consumer_variant=variant,
                support_origins=window["support_origins"],
                delayed_origins=window["delayed_origins"],
            )
            baseline_retrains = int(search.retrains)
            decision = _frozen_deploy_decision(
                snapshot=slot["_snapshot"], search=search, variant=variant,
                arm=arm, cell=cell,
            )
            # F1/F2 proof, taken before the single scoring read: deciding what
            # to deploy consumed no reading of the scored block at all.
            pre_scoring = {
                "reads_logged_before_scoring": len(search.log),
                "charged_support_evaluations": int(
                    search.support_evaluations_charged),
                "instrument_internal_evaluations": int(
                    search.internal_evaluations),
                "consumer_retrains_before_scoring": int(search.retrains),
                "identity_baseline_retrains": baseline_retrains,
                "decision_read_nothing_on_the_scored_block": bool(
                    len(search.log) == 0
                    and int(search.support_evaluations_charged) == 0
                    and int(search.internal_evaluations) == 0
                    and int(search.retrains) == baseline_retrains
                ),
            }
            plan = dict(decision["plan"])
            scored = search.delayed_gate(
                plan["program"], plan["excluded_series"])
            delayed_reads = [
                row for row in search.log
                if str(row.get("kind")) == "delayed_gate"
            ]
            record = {
                "episode_id": "deploy_%s_%s" % (variant, arm),
                "arm": arm,
                "consumer_variant": variant,
                "target_id": str(target["target_id"]),
                "window_id": str(window["window_id"]),
                "support_origins": list(search.support),
                "delayed_origins": list(search.delayed),
                "store_slot": slot_key,
                "frozen_runtime_bundle_sha": slot.get("runtime_bundle_sha"),
                "mode": "FROZEN_FAST_ONLY_DEPLOY",
                "deploy_source": str(decision["source"]),
                "deploy_why": str(decision["why"]),
                "active_skill_id": decision.get("active_skill_id"),
                "recall_hit": decision.get("recall_hit"),
                "recall_miss": bool(decision.get("recall_miss")),
                "incumbent_from_round": decision.get("from_round"),
                "lifecycle_at_recall": decision.get("lifecycle_at_recall"),
                "retrieval": decision.get("retrieval"),
                "applied_plan": plan,
                "final_plan": plan,
                "delayed": dict(scored),
                "support": None,
                "support_note": (
                    "not measured: a Support reading on the scored block would "
                    "be a feedback read, and nothing here needs one"
                ),
                "pre_scoring_purity": pre_scoring,
                "delayed_openings": len(delayed_reads),
                "llm_calls": 0,
                "new_skill_formed": False,
                "slow_agent": "off",
                "instrument": search.accounting(),
                "consumer_retrains_total": int(search.retrains),
            }
            cells[slot_key] = {
                "consumer_variant": variant,
                "arm": arm,
                "record": record,
                "consumer_retrains": int(search.retrains),
                "llm_calls": 0,
            }
            print(
                "FREPB %-18s %-34s applied %-20s delayed %+.6f harm %d "
                "retrains %d llm 0"
                % (
                    slot_key, decision["source"], plan["program"],
                    float(scored["aggregate_gain"]),
                    int(scored["harmed_eval_series_count"]),
                    int(search.retrains),
                ),
                flush=True,
            )
    return {
        "ran": True,
        "rule": dict(DEPLOY_RULE),
        "window": {
            key: value for key, value in window.items()
            if not str(key).startswith("reference_")
        },
        "slow_agent": "off",
        "cells": cells,
        "consumer_retrains": sum(
            int(row["consumer_retrains"]) for row in cells.values()),
        "llm_calls": 0,
    }


def _deploy_purity(deployment: Mapping[str, Any]) -> dict[str, Any]:
    """F1 and F2 discharged, per arm, from the instrument's own counters."""
    rows: dict[str, Any] = {}
    for slot_key, cell in (deployment.get("cells") or {}).items():
        record = cell["record"]
        pre = record["pre_scoring_purity"]
        rows[slot_key] = {
            "arm": record["arm"],
            "consumer_variant": record["consumer_variant"],
            "F1_decision_read_no_outcome": bool(
                pre["decision_read_nothing_on_the_scored_block"]),
            "F1_delayed_openings_total": int(record["delayed_openings"]),
            "F1_delayed_openings_used_for_adoption": 0,
            "F2_charged_support_evaluations": int(
                pre["charged_support_evaluations"]),
            "F2_candidate_evaluations": int(
                pre["instrument_internal_evaluations"]),
            "F2_llm_calls": int(record["llm_calls"]),
            "F2_deploy_consumer_retrains": int(cell["consumer_retrains"]),
            "new_skill_formed": bool(record["new_skill_formed"]),
        }
        rows[slot_key]["ok"] = bool(
            rows[slot_key]["F1_decision_read_no_outcome"]
            and rows[slot_key]["F1_delayed_openings_total"] == 1
            and rows[slot_key]["F2_charged_support_evaluations"] == 0
            and rows[slot_key]["F2_candidate_evaluations"] == 0
            and rows[slot_key]["F2_llm_calls"] == 0
            and not rows[slot_key]["new_skill_formed"]
        )
    costs = sorted({row["F2_deploy_consumer_retrains"] for row in rows.values()})
    return {
        "slots": rows,
        "all_pure": all(row["ok"] for row in rows.values()) and bool(rows),
        "deploy_cost_values": costs,
        "deploy_cost_identical_across_arms": len(costs) == 1,
        "deploy_cost_spread": (0 if len(costs) < 2 else costs[-1] - costs[0]),
        "cost_symmetry_note": (
            "each arm pays 3 identity support-baseline retrains + 3 identity "
            "delayed-baseline retrains, both computed by the instrument's own "
            "constructor and used by no decision, plus 3 retrains for the one "
            "scoring read of a non-identity applied plan (0 for identity, "
            "which reuses the cached identity baseline)"
        ),
    }


# ------------------------------------------------------- freeze and deploy
def _freeze_receipt(stores: Mapping[str, Any],
                    adaptation: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for slot_key, slot in stores.items():
        snapshot = slot.get("_snapshot")
        store = slot.get("_store")
        active = None
        if store is not None and store.active_path.is_file():
            active = json.loads(store.active_path.read_text(encoding="utf-8"))
        cell = (adaptation.get("cells") or {}).get(slot_key) or {}
        promotion = cell.get("promotion") or {}
        rows[slot_key] = {
            "arm": slot.get("arm"),
            "consumer_variant": slot.get("consumer_variant"),
            "store_root": slot.get("store_root"),
            "frozen_runtime_bundle_sha": (
                None if snapshot is None else snapshot.runtime_bundle_sha
            ),
            "frozen_harness_content_sha": (
                None if snapshot is None else snapshot.harness_content_sha
            ),
            "frozen_skill_ids": (
                None if snapshot is None
                else [skill.skill_id for skill in snapshot.skills]
            ),
            "active_pointer": active,
            "target_local_skill_id": promotion.get("retrievable_skill_id"),
            "target_local_promoted": bool(promotion.get("promoted")),
        }
    written = [
        key for key, row in rows.items()
        if row["frozen_runtime_bundle_sha"] and row["active_pointer"]
    ]
    return {
        "slots": rows,
        "slots_with_frozen_snapshot_on_disk": sorted(written),
        "all_slots_frozen": len(written) == len(rows) and bool(rows),
    }


def _freeze_unchanged(before: Mapping[str, Any],
                      stores: Mapping[str, Any]) -> dict[str, Any]:
    """The deploy stage may not move the frozen pointer."""
    rows: dict[str, Any] = {}
    for slot_key, slot in stores.items():
        snapshot = slot.get("_snapshot")
        store = slot.get("_store")
        active = None
        if store is not None and store.active_path.is_file():
            active = json.loads(store.active_path.read_text(encoding="utf-8"))
        was = before["slots"][slot_key]
        now_sha = None if snapshot is None else snapshot.runtime_bundle_sha
        now_ids = (
            None if snapshot is None
            else [skill.skill_id for skill in snapshot.skills]
        )
        rows[slot_key] = {
            "runtime_bundle_sha_unchanged": now_sha == was[
                "frozen_runtime_bundle_sha"],
            "skill_ids_unchanged": now_ids == was["frozen_skill_ids"],
            "active_pointer_unchanged": active == was["active_pointer"],
            "runtime_bundle_sha_after": now_sha,
        }
        rows[slot_key]["ok"] = bool(
            rows[slot_key]["runtime_bundle_sha_unchanged"]
            and rows[slot_key]["skill_ids_unchanged"]
            and rows[slot_key]["active_pointer_unchanged"]
        )
    return {
        "slots": rows,
        "all_unchanged": all(row["ok"] for row in rows.values()) and bool(rows),
        "rule": (
            "held-out deployment adds, modifies, approves, limits or revokes no "
            "Skill, so every frozen sha and the active pointer must be byte "
            "identical after scoring"
        ),
    }


def _deploy_binding(deployment: Mapping[str, Any]) -> dict[str, Any]:
    """The #42g-b 0b assertion, forecasting side.

    The scored number must be the reading of the bytes the deploy episode
    actually applied.  Free: the search's own log already records which
    ``(program, excluded_series)`` produced which delayed aggregate.
    """
    rows: dict[str, Any] = {}
    for slot_key, cell in (deployment.get("cells") or {}).items():
        record = cell.get("record") or {}
        plan = record.get("final_plan") or None
        delayed = record.get("delayed") or {}
        log = ((record.get("instrument") or {}).get("log")) or []
        scored = delayed.get("aggregate_gain")
        applied_program = None if plan is None else str(plan["program"])
        applied_excluded = (
            None if plan is None
            else sorted(str(uid) for uid in plan["excluded_series"])
        )
        matches = [
            row for row in log
            if str(row.get("kind")) == "delayed_gate"
            and str(row.get("program")) == applied_program
            and sorted(str(u) for u in (row.get("excluded_series") or []))
            == (applied_excluded or [])
        ]
        exact = [
            row for row in matches
            if scored is not None
            and row.get("aggregate_gain") is not None
            and abs(float(row["aggregate_gain"]) - float(scored)) <= 1e-12
        ]
        rows[slot_key] = {
            "arm": cell.get("arm"),
            "consumer_variant": cell.get("consumer_variant"),
            "mode": record.get("mode"),
            "applied_program": applied_program,
            "applied_excluded_series": applied_excluded,
            "scored_delayed_aggregate_gain": scored,
            "delayed_gate_reads_on_applied_bytes": len(matches),
            "delayed_gate_reads_matching_scored_value": len(exact),
            "new_skill_formed": bool(record.get("new_skill_formed")),
            "llm_calls": int(record.get("llm_calls") or 0),
            "ok": bool(plan is not None and exact and not record.get(
                "new_skill_formed")),
        }
    return {
        "slots": rows,
        "all_bound": all(row["ok"] for row in rows.values()) and bool(rows),
        "assertion": (
            "scored_program == deploy.applied_program and the scored aggregate "
            "is the delayed_gate reading of exactly those bytes; no Skill was "
            "formed by the scored episode"
        ),
        "cost": "0 extra Consumer retrains; read off the search's own log",
    }


# ------------------------------------------------------------- lifecycle
def _lifecycle_receipt(adaptation: Mapping[str, Any],
                       deployment: Mapping[str, Any]) -> dict[str, Any]:
    formed = approved = promoted = blocked = 0
    per_slot: dict[str, Any] = {}
    for slot_key, cell in (adaptation.get("cells") or {}).items():
        draft = cell.get("draft") or {}
        promotion = cell.get("promotion") or {}
        formed += 1 if draft.get("written") else 0
        approved += 1 if promotion.get("store_approved") else 0
        promoted += 1 if promotion.get("promoted") else 0
        blocked += 1 if (draft.get("written")
                         and not promotion.get("promoted")) else 0
        per_slot[slot_key] = {
            "arm": cell.get("arm"),
            "consumer_variant": cell.get("consumer_variant"),
            "draft_written": bool(draft.get("written")),
            "draft_skill_id": draft.get("skill_id"),
            "draft_reason": draft.get("reason"),
            "handle_fast_winner_stage": (
                (draft.get("handle_fast_winner") or {}).get("stage")
            ),
            "store_approved": bool(promotion.get("store_approved")),
            "promoted_to_local_active": bool(promotion.get("promoted")),
            "promotion_reason": promotion.get("reason"),
            "probe_gain": promotion.get("probe_gain"),
            "target_local_skill_id": promotion.get("retrievable_skill_id"),
            "held_in_round_2_mode": (cell.get("task_B") or {}).get("mode"),
            "held_in_round_2_reuse_adopted": (
                (cell.get("task_B") or {}).get("reuse_adopted")
            ),
        }
    abstained = 0
    for slot_key, cell in (deployment.get("cells") or {}).items():
        record = cell.get("record") or {}
        ladder = record.get("adoption_ladder") or {}
        was_abstain = str(ladder.get("path")) == (
            "SUPPORT_CONFIRMATION_FAILED_ABSTAIN")
        abstained += 1 if was_abstain else 0
        per_slot.setdefault(slot_key, {})["deploy_mode"] = record.get("mode")
        per_slot[slot_key]["deploy_abstained_on_reuse"] = bool(was_abstain)
        per_slot[slot_key]["deploy_reuse_adopted"] = record.get("reuse_adopted")
    return {
        "skills_formed_drafts_written": formed,
        "store_approvals": approved,
        "promotions_to_local_active": promoted,
        "drafts_written_but_not_promoted": blocked,
        "revocations": 0,
        "revocation_note": (
            "the original protocol has no revocation trigger inside two held-in "
            "rounds; a Draft that misses the promotion bar simply stays a "
            "Draft.  Reported as 0 observed, not as 0 possible."
        ),
        "deploy_abstentions_on_reuse": abstained,
        "per_slot": per_slot,
        "update_path": dict(fc.UPDATE_PATH),
    }


# --------------------------------------------------------------- readouts
def _pair_table(cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, cell in cells.items():
        for arm in ("A5", "A3"):
            summary = cell[arm]
            first = summary["first_positive"]
            rows.append({
                "consumer_variant": variant,
                "arm": arm,
                "first_positive_reached": bool(first["reached"]),
                "first_positive_at_step": first.get("at_step"),
                "first_positive_cost_consumer_retrains": first.get(
                    "cumulative_consumer_retrains"),
                "first_positive_cost_llm_calls": summary.get(
                    "first_positive_llm_calls"),
                "first_positive_candidate_executions": summary.get(
                    "first_positive_candidate_executions"),
                "total_consumer_retrains": summary["total_consumer_retrains"],
                "held_in_consumer_retrains": summary[
                    "adaptation_consumer_retrains"],
                "held_out_consumer_retrains": summary[
                    "confirmation_consumer_retrains"],
                "total_llm_calls": summary["llm_calls"],
                "held_in_delayed_utility": summary.get("held_in_delayed_utility"),
                "held_in_harm_count": summary.get("held_in_harm_count"),
                "held_out_plan": summary["task_c_plan"],
                "held_out_mode": summary["task_c_mode"],
                "held_out_delayed_utility": summary["task_c_delayed"],
                "held_out_support": summary["task_c_support"],
                "held_out_harm_count": summary["task_c_harm_count"],
                "held_out_harm_series": summary["task_c_harm_series"],
                "held_out_harm_total": summary["task_c_harm_total"],
                "target_local_skill": summary["local_skill"],
                "promoted": summary["promoted"],
            })
    return rows


def _augment_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Add the readouts the task book asks for that the original omits."""
    trace = list(summary["trace"])
    llm = 0
    executions = 0
    reached = False
    held_in_delayed: list[float] = []
    held_in_harm = 0
    for row in trace:
        llm += int(row.get("llm_calls") or 0)
        plan = row.get("adopted_plan")
        if plan is not None:
            executions += 1
        if str(row.get("step")) in ("task_A", "probe", "task_B"):
            gain = row.get("delayed_aggregate_gain")
            if gain is not None:
                held_in_delayed.append(float(gain))
            held_in_harm += int(row.get("harmed_eval_series_count") or 0)
        if row.get("adopted_delayed_positive"):
            reached = True
            break
    summary["first_positive_llm_calls"] = llm if reached else None
    summary["first_positive_candidate_executions"] = (
        executions if reached else None)
    summary["first_positive_llm_calls_note"] = (
        "cumulative LLM calls in the same chronological order as the retrain "
        "cost, up to and including the first delayed-positive adoption"
    )
    summary["held_in_delayed_utility"] = (
        sum(held_in_delayed) if held_in_delayed else None)
    summary["held_in_delayed_utility_rows"] = held_in_delayed
    summary["held_in_harm_count"] = held_in_harm
    return summary


def _chain_verdict(cells: Mapping[str, Any], raw: Mapping[str, Any],
                   freeze_ok: bool, binding_ok: bool,
                   lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    unreadable = [
        variant for variant, cell in cells.items()
        if cell["verdict"] == "CELL_UNREADABLE"
    ]
    if unreadable:
        return {
            "verdict": "INSTRUMENT_UNREADABLE",
            "reason": "no readable held-out delayed number for %s" % unreadable,
            "raw_instrument_verdict": raw["verdict"],
        }
    if not freeze_ok:
        return {
            "verdict": "CHAIN_BROKEN(freeze)",
            "reason": "the frozen snapshot moved across the deployment",
            "raw_instrument_verdict": raw["verdict"],
        }
    if not binding_ok:
        return {
            "verdict": "CHAIN_BROKEN(deploy_binding)",
            "reason": "the scored bytes are not the applied bytes",
            "raw_instrument_verdict": raw["verdict"],
        }
    primary = cells[bch.CONSUMER_POOLED]
    fp5 = primary["first_positive_cost"]["A5"]
    fp3 = primary["first_positive_cost"]["A3"]
    cheaper = bool(fp5 is not None and (fp3 is None or int(fp5) < int(fp3)))
    better = bool(primary["task_c_delayed_difference"] > 0.0)
    advantage = cheaper or better
    directional = []
    for variant, cell in cells.items():
        directional.append({
            "consumer_variant": variant,
            "first_positive_cost_A5_cheaper": bool(
                cell["first_positive_cost"]["A5_cheaper"]),
            "held_out_delayed_difference": cell["task_c_delayed_difference"],
            "raw_cell_verdict": cell["verdict"],
        })
    return {
        "verdict": (
            "CHAIN_REPRODUCED" if advantage else "CHAIN_OK_A5_NO_ADVANTAGE"
        ),
        "reason": (
            "both arms walked Source Skill -> held-in calibration -> "
            "Target-local Skill -> freeze -> held-out Fast-only scoring; on "
            "the primary pooled cell A5 first-positive cost %s vs %s (cheaper=%s) "
            "and held-out delayed difference %+.6f (better=%s)"
            % (fp5, fp3, cheaper,
               float(primary["task_c_delayed_difference"]), better)
        ),
        "primary_cell": bch.CONSUMER_POOLED,
        "directional": directional,
        "raw_instrument_verdict": raw["verdict"],
        "raw_instrument_reason": raw["reason"],
        "lifecycle_complete": {
            "drafts_written": lifecycle["skills_formed_drafts_written"],
            "promotions": lifecycle["promotions_to_local_active"],
        },
        "evidence_class": (
            "INFRASTRUCTURE/MECHANISM.  Development data, already-exposed "
            "outcomes: this closes nothing and opens nothing about capability."
        ),
    }


# ------------------------------------------------------------------- the run
def run(run_id: str) -> int:
    started = time.perf_counter()
    print("PY", sys.executable, flush=True)
    print("RUN_ID", run_id, flush=True)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "task_book": "#45-Frep",
        "generated_at_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_executable": sys.executable,
    }

    # ---- entry viability ------------------------------------------------
    try:
        window_syntax = fc._assert_window_syntax()
        artifact = fc._cohort_artifact()
        criteria = fc._readability_criteria(artifact)
        cap = fc._missing_cap(artifact)
        health = artifact["step_2_health_check_v2"]
        train = [str(uid) for uid in health["confirmation_roster"]]
        evaluation = [str(uid) for uid in health["substitutes"]]
    except Exception as exc:  # noqa: BLE001
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(entry)",
                "reason": "%s: %s" % (type(exc).__name__, exc),
            },
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)

    roster = {"train": train, "eval": evaluation}
    payload["binding"] = _binding(run_id, roster)
    payload["instrument_drift_vs_fresh_confirmation"] = _instrument_drift()
    payload["window_syntax_check"] = window_syntax

    held_in = {
        "task_A": fc.WINDOWS["task_A"],
        "task_B": fc.WINDOWS["task_B"],
        "probe": dict(fc.PROBE_WINDOW),
    }
    held_out = fc._window(
        HELD_OUT_S, "frep45_held_out",
        note="held-out tail inside the exposed 2024 development partition",
    )
    held_out_backup = fc._window(
        HELD_OUT_BACKUP_S, "frep45_held_out_backup",
        note="the one backup, one stride earlier, still inside 2024",
    )
    payload["windows"] = {
        "held_in": {
            name: {k: v for k, v in row.items()
                   if not str(k).startswith("reference_")}
            for name, row in held_in.items()
        },
        "held_out": {k: v for k, v in held_out.items()
                     if not str(k).startswith("reference_")},
        "held_out_backup": {k: v for k, v in held_out_backup.items()
                            if not str(k).startswith("reference_")},
        "no_overlap": {
            "held_in_farthest_index": max(
                int(held_in["task_A"]["farthest_index"]),
                int(held_in["task_B"]["farthest_index"]),
                int(held_in["probe"]["farthest_index"]),
            ),
            "held_out_first_context_index": int(
                min(held_out["support_origins"]) - fc.CONTEXT_LENGTH),
            "disjoint": bool(
                int(min(held_out["support_origins"]) - fc.CONTEXT_LENGTH)
                > max(
                    int(held_in["task_A"]["farthest_index"]),
                    int(held_in["task_B"]["farthest_index"]),
                    int(held_in["probe"]["farthest_index"]),
                )
            ),
        },
        "sealed_index_guard": {
            "held_out_farthest_index": int(held_out["farthest_index"]),
            "development_hours": int(fc.DEVELOPMENT_HOURS),
            "stays_inside_2024": int(
                held_out["farthest_index"]) <= int(fc.DEVELOPMENT_HOURS),
            "confirmation_2025_read": False,
            "beyond_17520_read": False,
        },
    }
    if not payload["windows"]["sealed_index_guard"]["stays_inside_2024"]:
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(entry)",
                "reason": "the held-out window would cross into sealed 2025",
            },
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)

    # ---- development values, 2024 only ---------------------------------
    development = fc._load_development(train + evaluation)
    dev_payload = fc._cohort_payload(train, evaluation, development["values"])
    payload["cohort"] = {
        "name": fc.COHORT_NAME,
        "train_uids": list(dev_payload["train_uids"]),
        "eval_uids": list(dev_payload["eval_uids"]),
        "development_records": {
            uid: {key: row[key] for key in (
                "length", "n_finite_development", "missing_rate_development")}
            for uid, row in development["records"].items()
        },
    }

    # ---- stage 0: is the Judge still readable ---------------------------
    stage0 = fc.stage_0(dev_payload, criteria)
    payload["stage_0_readability"] = fc._public(stage0)
    if not stage0["pass"]:
        payload.update({
            "verdict": {
                "verdict": "INSTRUMENT_UNREADABLE",
                "reason": "the identity Judge is not readable on this cohort",
            },
            "consumer_retrains_total": int(stage0["consumer_retrains"]),
            "llm_call_count": 0,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)

    # ---- run-id isolated store root ------------------------------------
    store_root = PROJECT_ROOT / "_scratch" / "skill_store" / "t6_45_frep" / run_id
    fc.STORE_ROOT = store_root
    payload["state_isolation"] = {
        "store_root": store_root.relative_to(PROJECT_ROOT).as_posix(),
        "run_id": run_id,
        "shared_with_other_runs": False,
    }

    # ---- stage 1: the frozen Source-derived Skill and the four stores ---
    try:
        stage1 = fc.stage_1()
    except Exception as exc:  # noqa: BLE001
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(source_skill)",
                "reason": "%s: %s" % (type(exc).__name__, exc),
            },
            "consumer_retrains_total": int(stage0["consumer_retrains"]),
            "llm_call_count": 0,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)
    fc.STORES.clear()
    fc.STORES.update(stage1["stores"])
    payload["stage_1_source_skill"] = fc._public(stage1)
    if stage1["verdict"] != "REGISTERED":
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(source_skill)",
                "reason": "a Source-derived Guidance card would not register: "
                          "%s" % stage1.get("blocked"),
            },
            "consumer_retrains_total": int(stage0["consumer_retrains"]),
            "llm_call_count": 0,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)

    # ---- stage 2: held-in calibration, both arms -----------------------
    budget = fc.Budget(LLM_CALL_BUDGET_TOTAL)
    try:
        stage2 = fc.stage_2(dev_payload, budget)
    except Exception as exc:  # noqa: BLE001
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(held_in)",
                "reason": "%s: %s" % (type(exc).__name__, exc),
            },
            "consumer_retrains_total": int(stage0["consumer_retrains"]),
            "llm_call_count": budget.used,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)
    payload["stage_2_held_in"] = fc._public(stage2)
    if stage2.get("stopped_early"):
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(held_in)",
                "reason": str(stage2["stopped_early"]),
            },
            "consumer_retrains_total": int(
                stage0["consumer_retrains"] + stage2["consumer_retrains"]),
            "llm_call_count": budget.used,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)

    # ---- freeze --------------------------------------------------------
    freeze = _freeze_receipt(fc.STORES, stage2)
    payload["freeze"] = freeze
    if not freeze["all_slots_frozen"]:
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(freeze)",
                "reason": "at least one arm has no frozen snapshot on disk",
            },
            "consumer_retrains_total": int(
                stage0["consumer_retrains"] + stage2["consumer_retrains"]),
            "llm_call_count": budget.used,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)

    # ---- the held-out missing gate -------------------------------------
    gate = fc._missing_gate(
        development["values"], dev_payload["eval_uids"], held_out, cap)
    gate_backup = None
    window = held_out
    if not gate["pass"]:
        gate_backup = fc._missing_gate(
            development["values"], dev_payload["eval_uids"],
            held_out_backup, cap)
        window = held_out_backup
    payload["held_out_missing_gate"] = {
        "primary": gate,
        "backup": gate_backup,
        "window_used": str(window["window_id"]),
        "backup_used": gate_backup is not None,
    }
    if gate_backup is not None and not gate_backup["pass"]:
        payload.update({
            "verdict": {
                "verdict": "INSTRUMENT_UNREADABLE",
                "reason": "neither held-out window clears the missing gate",
            },
            "consumer_retrains_total": int(
                stage0["consumer_retrains"] + stage2["consumer_retrains"]),
            "llm_call_count": budget.used,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)

    # ---- stage 4: held-out Fast-only deployment, scored once -----------
    # #45-Frep-b: the in-service ``fc.stage_4`` is no longer called here.  It
    # carried the F1 delayed-adoption ladder and the F2 asymmetric full-price
    # search; ``deploy_fast_only_symmetric`` is the repaired deployment path.
    try:
        stage4 = deploy_fast_only_symmetric(
            dev_payload, stage2, window, stores=fc.STORES)
    except Exception as exc:  # noqa: BLE001
        payload.update({
            "verdict": {
                "verdict": "CHAIN_BROKEN(deploy)",
                "reason": "%s: %s" % (type(exc).__name__, exc),
            },
            "consumer_retrains_total": int(
                stage0["consumer_retrains"] + stage2["consumer_retrains"]),
            "llm_call_count": budget.used,
            "wall_seconds": time.perf_counter() - started,
        })
        return _write(payload)
    payload["stage_4_held_out"] = fc._public(stage4)
    payload["deploy_purity"] = _deploy_purity(stage4)

    binding = _deploy_binding(stage4)
    payload["deploy_binding_assertion_0b"] = binding
    unchanged = _freeze_unchanged(freeze, fc.STORES)
    payload["freeze_unchanged_after_deploy"] = unchanged

    # ---- readouts ------------------------------------------------------
    cells = {
        variant: fc._cell_verdict(variant, stage2, stage4)
        for variant in fc.CONSUMERS
    }
    for cell in cells.values():
        _augment_summary(cell["A5"])
        _augment_summary(cell["A3"])
    raw = fc._overall(cells)
    lifecycle = _lifecycle_receipt(stage2, stage4)
    payload["lifecycle_receipt"] = lifecycle
    payload["cells"] = fc._public(cells)
    payload["paired_readouts"] = _pair_table(cells)
    payload["raw_instrument_overall"] = raw
    payload["verdict"] = _chain_verdict(
        cells, raw, bool(unchanged["all_unchanged"]),
        bool(binding["all_bound"]), lifecycle,
    )
    payload["budget_ledger"] = {
        "llm_call_budget_total": LLM_CALL_BUDGET_TOTAL,
        "llm_calls_used": int(budget.used),
        "llm_calls_left": int(budget.left),
        "task_book_cap": LLM_CAP_TASK_BOOK,
        "within_task_book_cap": int(budget.used) <= LLM_CAP_TASK_BOOK,
        "llm_by_stage": {
            "stage_0": 0,
            "stage_1": int(stage1["llm_calls"]),
            "stage_2_held_in": int(stage2["llm_calls"]),
            "stage_4_held_out": int(stage4["llm_calls"]),
        },
        "llm_by_arm": {
            arm: sum(
                int(cells[variant][arm]["llm_calls"])
                for variant in fc.CONSUMERS
            )
            for arm in ("A5", "A3")
        },
        "consumer_retrains_total": int(
            stage0["consumer_retrains"] + stage2["consumer_retrains"]
            + stage4["consumer_retrains"]),
        "consumer_retrains_by_stage": {
            "stage_0": int(stage0["consumer_retrains"]),
            "stage_2_held_in": int(stage2["consumer_retrains"]),
            "stage_4_held_out": int(stage4["consumer_retrains"]),
        },
        "consumer_retrains_by_arm": {
            arm: sum(
                int(cells[variant][arm]["total_consumer_retrains"])
                for variant in fc.CONSUMERS
            )
            for arm in ("A5", "A3")
        },
        "downloads": 0,
        "sealed_reads": 0,
    }
    payload["exposure_ledger"] = {
        "cohort": "noaa_global_hourly_fresh_v1",
        "roster": dict(roster),
        "partitions": {
            "development_2024": {
                "index": [0, int(fc.DEVELOPMENT_HOURS)],
                "instance": "SEEN",
                "outcome": "EXPOSED",
                "opened_by": (
                    "already EXPOSED by fresh_confirmation_v1; this replay "
                    "re-read it, including the 2024 tail block used as held-out"
                ),
            },
            "confirmation_2025": {
                "index": [int(fc.DEVELOPMENT_HOURS), int(fc.CONFIRMATION_END)],
                "instance": "SEEN",
                "outcome": "EXPOSED",
                "opened_by": (
                    "fresh_confirmation_v1 stage 4, not this run; this run read "
                    "zero bytes of it"
                ),
            },
            "beyond_17520": {
                "index": [int(fc.CONFIRMATION_END), None],
                "instance": "SEALED",
                "outcome": "SEALED",
                "opened_by": "nothing; this run read zero bytes of it",
            },
        },
        "yahoo_s5_a1": "not read by this run, at all",
    }
    payload["wall_seconds"] = time.perf_counter() - started
    return _write(payload)


# ------------------------- #45-Frep-b: re-adjudicate the terminal readout
def _load_frozen_stores(run_id: str,
                        freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen the #45-Frep frozen snapshots and verify them byte for byte."""
    root = PROJECT_ROOT / "_scratch" / "skill_store" / "t6_45_frep" / run_id
    stores: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    for variant in fc.CONSUMERS:
        for arm in fc.ARMS:
            slot_key = "%s_%s" % (arm.lower(), variant)
            was = (freeze.get("slots") or {}).get(slot_key) or {}
            row: dict[str, Any] = {
                "slot": slot_key,
                "expected_runtime_bundle_sha": was.get(
                    "frozen_runtime_bundle_sha"),
                "expected_skill_ids": was.get("frozen_skill_ids"),
            }
            snapshots = root / slot_key / "snapshots"
            store = fc.SnapshotStore(snapshots)
            if not store.active_path.is_file():
                row.update({"ok": False, "reason": "no active pointer on disk"})
                checks[slot_key] = row
                continue
            active = json.loads(store.active_path.read_text(encoding="utf-8"))
            sha = str(active.get("runtime_bundle_sha") or "")
            row["active_runtime_bundle_sha"] = sha
            tree = snapshots / sha
            if not tree.is_dir():
                row.update({"ok": False,
                            "reason": "the active bundle is not materialized"})
                checks[slot_key] = row
                continue
            try:
                snapshot = fc.compile_snapshot(tree, verify_lock=False)
            except Exception as exc:  # noqa: BLE001
                row.update({"ok": False,
                            "reason": "%s: %s" % (type(exc).__name__, exc)})
                checks[slot_key] = row
                continue
            row.update({
                "recompiled_runtime_bundle_sha": snapshot.runtime_bundle_sha,
                "recompiled_harness_content_sha": snapshot.harness_content_sha,
                "recompiled_skill_ids": [s.skill_id for s in snapshot.skills],
                "recompile_reproduces_active_sha": (
                    snapshot.runtime_bundle_sha == sha),
                "active_matches_frep_report": (
                    sha == row["expected_runtime_bundle_sha"]),
                "skill_ids_match_frep_report": (
                    [s.skill_id for s in snapshot.skills]
                    == list(row["expected_skill_ids"] or [])),
            })
            row["ok"] = bool(
                row["recompile_reproduces_active_sha"]
                and row["active_matches_frep_report"]
                and row["skill_ids_match_frep_report"]
            )
            checks[slot_key] = row
            stores[slot_key] = {
                "slot": slot_key,
                "arm": arm,
                "consumer_variant": variant,
                "runtime_bundle_sha": snapshot.runtime_bundle_sha,
                "store_root": snapshots.relative_to(PROJECT_ROOT).as_posix(),
                "_snapshot": snapshot,
                "_store": store,
            }
    return {
        "store_root": (
            root.relative_to(PROJECT_ROOT).as_posix()
            if root.exists() else str(root)
        ),
        "checks": checks,
        "all_ok": (
            len(stores) == len(fc.CONSUMERS) * len(fc.ARMS)
            and all(row.get("ok") for row in checks.values())
        ),
        "method": (
            "each arm's active bundle is recompiled from its materialized tree "
            "and must reproduce its own runtime_bundle_sha, match the sha the "
            "#45-Frep report published, and carry the same Skill ids"
        ),
        "_stores": stores,
    }


def _reverify_from_disk(run_id: str,
                        before: Mapping[str, Any]) -> dict[str, Any]:
    """After deployment, recompile every arm from disk again and compare.

    Re-reads the tree rather than the in-memory objects, so a deployment that
    had written anything into a store would show up here.
    """
    root = PROJECT_ROOT / "_scratch" / "skill_store" / "t6_45_frep" / run_id
    rows: dict[str, Any] = {}
    for slot_key, was in before.items():
        store = fc.SnapshotStore(root / slot_key / "snapshots")
        active = json.loads(store.active_path.read_text(encoding="utf-8"))
        sha = str(active.get("runtime_bundle_sha") or "")
        snapshot = fc.compile_snapshot(
            store.root / sha, verify_lock=False)
        ids = [skill.skill_id for skill in snapshot.skills]
        rows[slot_key] = {
            "active_pointer_unchanged": sha == was.get(
                "active_runtime_bundle_sha"),
            "runtime_bundle_sha_unchanged": (
                snapshot.runtime_bundle_sha == was.get(
                    "recompiled_runtime_bundle_sha")),
            "skill_ids_unchanged": ids == list(
                was.get("recompiled_skill_ids") or []),
            "runtime_bundle_sha_after": snapshot.runtime_bundle_sha,
        }
        rows[slot_key]["ok"] = bool(
            rows[slot_key]["active_pointer_unchanged"]
            and rows[slot_key]["runtime_bundle_sha_unchanged"]
            and rows[slot_key]["skill_ids_unchanged"]
        )
    return {
        "slots": rows,
        "all_unchanged": all(row["ok"] for row in rows.values()) and bool(rows),
        "method": "recompiled from the on-disk tree after deployment",
        "rule": (
            "held-out deployment adds, modifies, approves, limits or revokes "
            "no Skill, so every sha and pointer must be identical afterwards"
        ),
    }


def _held_in_ledger(frep: Mapping[str, Any]) -> dict[str, Any]:
    """The #45-Frep held-in readouts, reused unchanged and re-verified."""
    rows: dict[str, Any] = {}
    for variant in fc.CONSUMERS:
        for arm in fc.ARMS:
            summary = frep["cells"][variant][arm]
            held_in = [
                row for row in summary["trace"]
                if str(row.get("step")) in ("task_A", "probe", "task_B")
            ]
            recomputed = fc._first_positive(held_in)
            published = summary["first_positive"]
            rows["%s_%s" % (arm.lower(), variant)] = {
                "arm": arm,
                "consumer_variant": variant,
                "held_in_trace_rows": len(held_in),
                "first_positive_published": published,
                "first_positive_from_held_in_only": recomputed,
                "unchanged_by_dropping_the_voided_deploy_row": bool(
                    published.get("reached") == recomputed.get("reached")
                    and published.get("cumulative_consumer_retrains")
                    == recomputed.get("cumulative_consumer_retrains")
                ),
                "held_in_consumer_retrains": summary[
                    "adaptation_consumer_retrains"],
                "held_in_llm_calls": sum(
                    int(row.get("llm_calls") or 0) for row in held_in),
                "first_positive_llm_calls": summary.get(
                    "first_positive_llm_calls"),
                "held_in_delayed_utility": summary.get(
                    "held_in_delayed_utility"),
                "held_in_harm_count": summary.get("held_in_harm_count"),
                "target_local_skill": summary.get("local_skill"),
                "promoted": summary.get("promoted"),
            }
    return {
        "source": "artifacts/functional/e2/t6_45_frep_a5a3_replay.json",
        "reused_unchanged": True,
        "slots": rows,
        "all_first_positive_invariant": all(
            row["unchanged_by_dropping_the_voided_deploy_row"]
            for row in rows.values()
        ),
        "why_invariant": (
            "every arm reached its first delayed-positive adoption inside "
            "held-in, so the voided deployment row never entered the "
            "first-positive cost in the first place"
        ),
    }


def _voided_because(variant: str, frep: Mapping[str, Any],
                    voided: Mapping[str, Any]) -> dict[str, Any]:
    """Which of F1 and F2 actually bit in this cell, read off the artifact.

    F1 bit if the voided deployment episode consulted any delayed reading of
    the scored block while choosing.  F2 bit if the two arms were not given
    the same deployment treatment.
    """
    delayed_reads = 0
    for arm in ("A5", "A3"):
        slot = "%s_%s" % (arm.lower(), variant)
        record = ((frep.get("stage_4_held_out") or {}).get("cells") or {}).get(
            slot, {}).get("record") or {}
        ladder = record.get("adoption_ladder") or {}
        reads = ladder.get("delayed_reads") or ladder.get("reads") or []
        delayed_reads += len(reads)
    f1 = delayed_reads > 0
    f2 = bool(
        str(voided["A5_deploy_mode"]) != str(voided["A3_deploy_mode"])
        or int(voided["A5_deploy_retrains"] or 0)
        != int(voided["A3_deploy_retrains"] or 0)
    )
    reasons = []
    if f1:
        reasons.append(
            "F1: the voided deployment took %d delayed reading(s) of the "
            "scored block while choosing what to deploy" % delayed_reads)
    if f2:
        reasons.append(
            "F2: the two arms were not given the same deployment treatment "
            "(%s at %s retrains vs %s at %s retrains)" % (
                voided["A5_deploy_mode"], voided["A5_deploy_retrains"],
                voided["A3_deploy_mode"], voided["A3_deploy_retrains"]))
    if not reasons:
        reasons.append(
            "neither defect bit in this cell; it is re-scored only so that "
            "both cells are read under one protocol")
    return {
        "F1_applies": f1,
        "F2_applies": f2,
        "voided_delayed_reads_on_scored_block": delayed_reads,
        "text": "; ".join(reasons),
    }


def _readjudicated_cells(ledger: Mapping[str, Any],
                         deployment: Mapping[str, Any],
                         frep: Mapping[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for variant in fc.CONSUMERS:
        arms: dict[str, Any] = {}
        for arm in fc.ARMS:
            slot_key = "%s_%s" % (arm.lower(), variant)
            held_in = ledger["slots"][slot_key]
            record = deployment["cells"][slot_key]["record"]
            delayed = record["delayed"]
            arms[arm] = {
                "arm": arm,
                "consumer_variant": variant,
                "first_positive_cost_consumer_retrains": held_in[
                    "first_positive_from_held_in_only"].get(
                        "cumulative_consumer_retrains"),
                "first_positive_at_step": held_in[
                    "first_positive_from_held_in_only"].get("at_step"),
                "first_positive_llm_calls": held_in[
                    "first_positive_llm_calls"],
                "held_in_consumer_retrains": held_in[
                    "held_in_consumer_retrains"],
                "held_in_llm_calls": held_in["held_in_llm_calls"],
                "held_in_delayed_utility": held_in["held_in_delayed_utility"],
                "held_in_harm_count": held_in["held_in_harm_count"],
                "deploy_consumer_retrains": int(
                    deployment["cells"][slot_key]["consumer_retrains"]),
                "deploy_llm_calls": 0,
                "total_consumer_retrains": int(
                    held_in["held_in_consumer_retrains"]
                    + deployment["cells"][slot_key]["consumer_retrains"]),
                "total_llm_calls": int(held_in["held_in_llm_calls"]),
                "deploy_source": record["deploy_source"],
                "held_out_plan": dict(record["applied_plan"]),
                "held_out_terminal_utility": float(delayed["aggregate_gain"]),
                "held_out_harm_count": int(delayed["harmed_eval_series_count"]),
                "held_out_harm_series": list(delayed["harmed_eval_series"]),
                "held_out_harm_total": float(
                    delayed["harmed_eval_series_total_harm"]),
                "target_local_skill": held_in["target_local_skill"],
                "promoted": held_in["promoted"],
            }
        a5, a3 = arms["A5"], arms["A3"]
        delta = float(a5["held_out_terminal_utility"]) - float(
            a3["held_out_terminal_utility"])
        fp5 = a5["first_positive_cost_consumer_retrains"]
        fp3 = a3["first_positive_cost_consumer_retrains"]
        voided_cell = frep["cells"][variant]
        voided = {
            "held_out_delayed_difference": voided_cell[
                "task_c_delayed_difference"],
            "A5_terminal": voided_cell["A5"]["task_c_delayed"],
            "A3_terminal": voided_cell["A3"]["task_c_delayed"],
            "A5_deploy_mode": voided_cell["A5"]["task_c_mode"],
            "A3_deploy_mode": voided_cell["A3"]["task_c_mode"],
            "A5_deploy_retrains": voided_cell["A5"][
                "confirmation_consumer_retrains"],
            "A3_deploy_retrains": voided_cell["A3"][
                "confirmation_consumer_retrains"],
            "raw_cell_verdict": voided_cell["verdict"],
        }
        attribution = _voided_because(variant, frep, voided)
        voided.update({
            "voided_because": attribution["text"],
            "F1_applies": attribution["F1_applies"],
            "F2_applies": attribution["F2_applies"],
            "voided_delayed_reads_on_scored_block": attribution[
                "voided_delayed_reads_on_scored_block"],
        })
        cells[variant] = {
            "consumer_variant": variant,
            "A5": a5,
            "A3": a3,
            "held_out_terminal_difference": delta,
            "held_out_terminal_direction": (
                "A5_HIGHER" if delta > fc.MATERIAL_THRESHOLD
                else "A3_HIGHER" if delta < -fc.MATERIAL_THRESHOLD
                else "WITHIN_MATERIAL_BAND"
            ),
            "material_threshold": fc.MATERIAL_THRESHOLD,
            "first_positive_cost": {
                "A5": fp5, "A3": fp3,
                "A5_cheaper": bool(
                    fp5 is not None and (fp3 is None or int(fp5) < int(fp3))),
            },
            "harm": {
                "A5": a5["held_out_harm_count"],
                "A3": a3["held_out_harm_count"],
            },
            "same_applied_workflow": bool(
                str(a5["held_out_plan"]["program"])
                == str(a3["held_out_plan"]["program"])
                and a5["held_out_plan"]["excluded_series"]
                == a3["held_out_plan"]["excluded_series"]
            ),
            "voided_frep_reading": voided,
        }
    return cells


def _frep_b_verdict(cells: Mapping[str, Any], purity: Mapping[str, Any],
                    binding: Mapping[str, Any],
                    ledger: Mapping[str, Any]) -> dict[str, Any]:
    unreadable = [
        variant for variant, cell in cells.items()
        for arm in ("A5", "A3")
        if cell[arm]["held_out_terminal_utility"] is None
    ]
    if unreadable:
        return {
            "verdict": "INSTRUMENT_UNREADABLE",
            "reason": "no readable terminal utility for %s" % unreadable,
        }
    if not purity["all_pure"]:
        return {
            "verdict": "INSTRUMENT_UNREADABLE",
            "reason": (
                "the repaired deployment did not discharge F1/F2 on every "
                "arm: %s" % [k for k, v in purity["slots"].items()
                             if not v["ok"]]
            ),
        }
    if not purity["deploy_cost_identical_across_arms"]:
        return {
            "verdict": "INSTRUMENT_UNREADABLE",
            "reason": (
                "deployment cost is still not symmetric across arms: %s"
                % purity["deploy_cost_values"]
            ),
        }
    if not binding["all_bound"]:
        return {
            "verdict": "INSTRUMENT_UNREADABLE",
            "reason": "the scored bytes are not the applied bytes",
        }
    primary = cells[bch.CONSUMER_POOLED]
    return {
        "verdict": "A5A3_TERMINAL_READJUDICATED",
        "reason": (
            "both arms deployed from frozen state only, with zero delayed "
            "openings used for adoption, zero candidate evaluations and "
            "identical deployment cost (%s retrains each). On the primary "
            "pooled cell the held-out terminal utility is A5 %+.6f vs A3 "
            "%+.6f (difference %+.6f, %s); first-positive cost %s vs %s is "
            "carried over from the #45-Frep held-in ledger unchanged."
            % (
                purity["deploy_cost_values"][0],
                float(primary["A5"]["held_out_terminal_utility"]),
                float(primary["A3"]["held_out_terminal_utility"]),
                float(primary["held_out_terminal_difference"]),
                primary["held_out_terminal_direction"],
                primary["first_positive_cost"]["A5"],
                primary["first_positive_cost"]["A3"],
            )
        ),
        "primary_cell": bch.CONSUMER_POOLED,
        "per_cell": {
            variant: {
                "held_out_terminal_difference": cell[
                    "held_out_terminal_difference"],
                "held_out_terminal_direction": cell[
                    "held_out_terminal_direction"],
                "same_applied_workflow": cell["same_applied_workflow"],
                "first_positive_cost_A5_cheaper": cell[
                    "first_positive_cost"]["A5_cheaper"],
                "harm": cell["harm"],
            }
            for variant, cell in cells.items()
        },
        "first_positive_cost_invariant": bool(
            ledger["all_first_positive_invariant"]),
        "no_positive_bar_was_imposed": True,
        "evidence_class": (
            "INFRASTRUCTURE/MECHANISM.  Development data, already-exposed "
            "outcomes, one paired draw: this re-adjudicates the #45-Frep "
            "terminal comparison under a clean deployment protocol and closes "
            "nothing about capability."
        ),
    }


def run_b(from_run_id: str) -> int:
    started = time.perf_counter()
    print("PY", sys.executable, flush=True)
    print("FROM_RUN_ID", from_run_id, flush=True)

    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION_B,
        "task_book": "#45-Frep-b",
        "role": (
            "repair the two self-reported deployment defects F1 and F2 and "
            "re-adjudicate the A5 vs A3 held-out terminal utility on the "
            "#45-Frep frozen snapshots; held-in is not re-run"
        ),
        "generated_at_wall": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_executable": sys.executable,
        "scope_pin": {
            "fixes_in_scope": ["F1", "F2"],
            "held_in_re_run": False,
            "in_service_runner_edited": False,
            "fix_location": (
                "evaluation/functional/run_e2_t6_45_frep_a5a3_replay.py, the "
                "deployment path only"
            ),
            "no_other_debt_touched": True,
        },
        "deploy_rule": dict(DEPLOY_RULE),
    }

    if not OUT_JSON.is_file():
        payload.update({
            "verdict": {
                "verdict": "SNAPSHOT_UNAVAILABLE",
                "reason": "the #45-Frep artifact is missing",
            },
            "wall_seconds": time.perf_counter() - started,
        })
        return _write_b(payload)
    frep = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    payload["frep_a_source"] = {
        "artifact": "artifacts/functional/e2/t6_45_frep_a5a3_replay.json",
        "protocol_version": frep.get("protocol_version"),
        "chain_verdict": (frep.get("verdict") or {}).get("verdict"),
        "run_id": (frep.get("binding") or {}).get("run_id"),
        "held_in_llm_calls": (frep.get("budget_ledger") or {}).get(
            "llm_by_stage", {}).get("stage_2_held_in"),
    }

    reopened = _load_frozen_stores(from_run_id, frep.get("freeze") or {})
    stores = reopened.pop("_stores")
    payload["frozen_snapshot_verification"] = reopened
    if not reopened["all_ok"]:
        payload.update({
            "verdict": {
                "verdict": "SNAPSHOT_UNAVAILABLE",
                "reason": (
                    "the #45-Frep frozen snapshots are missing or do not "
                    "verify: %s" % [
                        k for k, v in reopened["checks"].items()
                        if not v.get("ok")]
                ),
                "held_in_not_re_run": True,
            },
            "wall_seconds": time.perf_counter() - started,
        })
        return _write_b(payload)

    artifact = fc._cohort_artifact()
    cap = fc._missing_cap(artifact)
    health = artifact["step_2_health_check_v2"]
    train = [str(uid) for uid in health["confirmation_roster"]]
    evaluation = [str(uid) for uid in health["substitutes"]]
    development = fc._load_development(train + evaluation)
    dev_payload = fc._cohort_payload(train, evaluation, development["values"])

    window_id = str(
        (frep.get("held_out_missing_gate") or {}).get("window_used") or "")
    start = (HELD_OUT_BACKUP_S if window_id.endswith("backup")
             else HELD_OUT_S)
    window = fc._window(
        start, window_id or "frep45_held_out",
        note="held-out tail inside the exposed 2024 development partition",
    )
    gate = fc._missing_gate(
        development["values"], dev_payload["eval_uids"], window, cap)
    payload["held_out"] = {
        "window": {k: v for k, v in window.items()
                   if not str(k).startswith("reference_")},
        "same_window_as_frep": window_id,
        "missing_gate": gate,
        "stays_inside_2024": int(
            window["farthest_index"]) <= int(fc.DEVELOPMENT_HOURS),
        "confirmation_2025_read": False,
        "beyond_17520_read": False,
        "scoring": (
            "three delayed origins, sMASE gain against the identity baseline "
            "-- the original readout unit, unchanged"
        ),
    }
    if not gate["pass"] or not payload["held_out"]["stays_inside_2024"]:
        payload.update({
            "verdict": {
                "verdict": "INSTRUMENT_UNREADABLE",
                "reason": "the held-out window no longer clears its own gate",
            },
            "wall_seconds": time.perf_counter() - started,
        })
        return _write_b(payload)

    ledger = _held_in_ledger(frep)
    payload["held_in_ledger_reused"] = ledger

    try:
        deployment = deploy_fast_only_symmetric(
            dev_payload, frep["stage_2_held_in"], window, stores=stores)
    except Exception as exc:  # noqa: BLE001
        payload.update({
            "verdict": {
                "verdict": "INSTRUMENT_UNREADABLE",
                "reason": "the repaired deployment raised %s: %s"
                          % (type(exc).__name__, exc),
            },
            "wall_seconds": time.perf_counter() - started,
        })
        return _write_b(payload)
    payload["symmetric_deployment"] = fc._public(deployment)
    purity = _deploy_purity(deployment)
    payload["deploy_purity_F1_F2"] = purity
    binding = _deploy_binding(deployment)
    payload["deploy_binding_assertion_0b"] = binding
    payload["freeze_unchanged_after_deploy"] = _reverify_from_disk(
        from_run_id, reopened["checks"])

    cells = _readjudicated_cells(ledger, deployment, frep)
    payload["cells"] = cells
    payload["paired_readouts"] = [
        {
            "consumer_variant": variant,
            "arm": arm,
            "first_positive_cost_consumer_retrains": cell[arm][
                "first_positive_cost_consumer_retrains"],
            "first_positive_llm_calls": cell[arm]["first_positive_llm_calls"],
            "total_consumer_retrains": cell[arm]["total_consumer_retrains"],
            "total_llm_calls": cell[arm]["total_llm_calls"],
            "deploy_consumer_retrains": cell[arm]["deploy_consumer_retrains"],
            "deploy_source": cell[arm]["deploy_source"],
            "held_out_plan": cell[arm]["held_out_plan"],
            "held_out_terminal_utility": cell[arm][
                "held_out_terminal_utility"],
            "held_out_harm_count": cell[arm]["held_out_harm_count"],
        }
        for variant, cell in cells.items() for arm in ("A5", "A3")
    ]
    payload["verdict"] = _frep_b_verdict(cells, purity, binding, ledger)
    payload["budget_ledger"] = {
        "this_book_llm_calls": 0,
        "this_book_llm_note": (
            "the repaired deployment is recall-only, so it spends no LLM call "
            "at all; the increment over #45-Frep is exactly 0"
        ),
        "this_book_consumer_retrains": int(deployment["consumer_retrains"]),
        "this_book_consumer_retrains_by_arm": {
            arm: sum(
                int(cells[variant][arm]["deploy_consumer_retrains"])
                for variant in fc.CONSUMERS
            )
            for arm in ("A5", "A3")
        },
        "carried_over_held_in_llm_calls": (frep.get("budget_ledger") or {}).get(
            "llm_by_stage", {}).get("stage_2_held_in"),
        "carried_over_held_in_consumer_retrains": (
            frep.get("budget_ledger") or {}).get(
                "consumer_retrains_by_stage", {}).get("stage_2_held_in"),
        "downloads": 0,
        "sealed_reads": 0,
        "held_in_re_runs": 0,
    }
    payload["exposure_ledger"] = {
        "cohort": "noaa_global_hourly_fresh_v1",
        "roster": {"train": train, "eval": evaluation},
        "development_2024": (
            "already EXPOSED before #45-Frep; this book re-read the same 2024 "
            "tail block and nothing else"
        ),
        "confirmation_2025": "zero bytes read by this book",
        "beyond_17520": "zero bytes read by this book",
        "yahoo_s5_a1": "zero bytes read by this book",
    }
    payload["wall_seconds"] = time.perf_counter() - started
    return _write_b(payload)


# ------------------------------------------------------------------ report
def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return "%+.6f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _plan_label(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "--"
    excluded = list(plan.get("excluded_series") or [])
    if not excluded:
        return "`%s` full batch" % plan["program"]
    return "`%s` minus %s" % (plan["program"], ", ".join(excluded))


def _markdown(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    raw = payload.get("raw_instrument_overall") or {}
    lines = [
        "# #45-Frep: A5 vs A3 forecasting development replay",
        "",
        "**Chain verdict: `%s`**" % verdict.get("verdict"),
        "",
        str(verdict.get("reason") or ""),
        "",
    ]
    if raw:
        lines += [
            "**Raw A5-vs-A3 instrument verdict on the same readings: `%s`** "
            "(pooled `%s`, per_channel `%s`)." % (
                raw.get("verdict"), raw.get("pooled"), raw.get("per_channel")),
            "",
            "The two verdicts answer different questions and must be quoted "
            "together. The chain verdict asks whether the forward chain still "
            "walks and whether A5 keeps a directional edge on *either* "
            "primary readout; the original pre-registered A5-vs-A3 clauses ask "
            "for a joint cost-and-terminal-utility win, and on this replay "
            "they are not met.",
            "",
        ]
    lines += [
        "Development data, already-exposed outcomes. This is a chain "
        "regression check on current HEAD, not capability evidence.",
        "",
    ]
    binding = payload.get("binding") or {}
    if binding:
        arch = binding.get("archaeology") or {}
        lines += [
            "## Binding",
            "",
            "- run id: `%s`" % binding.get("run_id"),
            "- reused entry: `%s`" % (binding.get("entry") or {}).get(
                "reused_runner"),
            "- original verdict being replayed: `%s`, first-positive cost "
            "%s vs %s (pooled)" % (
                arch.get("original_verdict"),
                (arch.get("original_first_positive_cost_pooled") or {}).get("A5"),
                (arch.get("original_first_positive_cost_pooled") or {}).get("A3"),
            ),
            "- Target: %s, %d series, partition %s" % (
                (binding.get("target") or {}).get("cohort"),
                int((binding.get("target") or {}).get("series_count") or 0),
                (binding.get("target") or {}).get("partition"),
            ),
            "- Source Skill overlap with Target series: `%s`" % (
                (binding.get("source_skill_overlap") or {}).get(
                    "overlap_with_target_series")),
            "- LLM budget both arms: %s (task-book cap %s)" % (
                (binding.get("budget") or {}).get(
                    "llm_call_budget_total_both_arms"),
                (binding.get("budget") or {}).get("task_book_cap"),
            ),
            "",
        ]
    windows = payload.get("windows") or {}
    if windows:
        held_out = windows.get("held_out") or {}
        lines += [
            "## Windows",
            "",
            "| block | window | support origins | delayed origins | farthest |",
            "| --- | --- | --- | --- | ---: |",
        ]
        for name, row in (windows.get("held_in") or {}).items():
            lines.append("| held-in %s | `%s` | %s | %s | %s |" % (
                name, row.get("window_id"),
                row.get("support_origins") or row.get("origins"),
                row.get("delayed_origins") or "--",
                row.get("farthest_index"),
            ))
        lines.append("| held-out | `%s` | %s | %s | %s |" % (
            held_out.get("window_id"), held_out.get("support_origins"),
            held_out.get("delayed_origins"), held_out.get("farthest_index"),
        ))
        guard = windows.get("sealed_index_guard") or {}
        lines += [
            "",
            "held-in / held-out disjoint: `%s`; held-out farthest %s <= 8760: "
            "`%s`; 2025 bytes read: `%s`" % (
                (windows.get("no_overlap") or {}).get("disjoint"),
                guard.get("held_out_farthest_index"),
                guard.get("stays_inside_2024"),
                guard.get("confirmation_2025_read"),
            ),
            "",
        ]
    drift = payload.get("instrument_drift_vs_fresh_confirmation") or {}
    if drift:
        lines += [
            "## Instrument drift since FRESH_A5_DELIVERS",
            "",
            "%s of %s frozen-surface files moved between the original reading "
            "and HEAD." % (drift.get("drifted"), drift.get("files")),
            "",
        ]
        for row in drift.get("drift") or []:
            lines.append("- `%s`: %s -> %s" % (
                row["path"], str(row["sha_at_fresh_confirmation"])[:12],
                str(row["sha_at_head"] or "missing")[:12],
            ))
        lines.append("")
    rows = payload.get("paired_readouts") or []
    if rows:
        lines += [
            "## Paired readouts",
            "",
            "| cell | arm | first-pos cost (retrains) | first-pos LLM | "
            "total retrains | total LLM | held-in delayed | held-in harm | "
            "held-out plan | held-out delayed | held-out harm |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | "
            "---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| `%s` | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    row["consumer_variant"], row["arm"],
                    row["first_positive_cost_consumer_retrains"],
                    row["first_positive_cost_llm_calls"],
                    row["total_consumer_retrains"], row["total_llm_calls"],
                    _fmt(row["held_in_delayed_utility"]),
                    row["held_in_harm_count"],
                    _plan_label(row["held_out_plan"]),
                    _fmt(row["held_out_delayed_utility"]),
                    row["held_out_harm_count"],
                )
            )
        lines.append("")
    cells = payload.get("cells") or {}
    if cells:
        lines += [
            "## Per cell (raw instrument clauses, unchanged from the original)",
            "",
            "| cell | raw verdict | first-pos A5/A3 | total A5/A3 | "
            "held-out delayed diff | harm A5/A3 |",
            "| --- | --- | --- | --- | ---: | --- |",
        ]
        for variant, cell in cells.items():
            lines.append("| `%s` | `%s` | %s / %s | %s / %s | %s | %s / %s |" % (
                variant, cell["verdict"],
                cell["first_positive_cost"]["A5"],
                cell["first_positive_cost"]["A3"],
                cell["total_cost"]["A5"], cell["total_cost"]["A3"],
                _fmt(cell["task_c_delayed_difference"]),
                cell["harm"]["A5"], cell["harm"]["A3"],
            ))
        lines.append("")
    life = payload.get("lifecycle_receipt") or {}
    if life:
        lines += [
            "## Lifecycle receipts",
            "",
            "- Skills formed (Drafts written): %s" % life.get(
                "skills_formed_drafts_written"),
            "- Store approvals: %s" % life.get("store_approvals"),
            "- Promotions to LOCAL_ACTIVE: %s" % life.get(
                "promotions_to_local_active"),
            "- Drafts written but not promoted: %s" % life.get(
                "drafts_written_but_not_promoted"),
            "- Revocations observed: %s" % life.get("revocations"),
            "- Deploy abstentions on reuse: %s" % life.get(
                "deploy_abstentions_on_reuse"),
            "",
        ]
    freeze = payload.get("freeze_unchanged_after_deploy") or {}
    binding_row = payload.get("deploy_binding_assertion_0b") or {}
    if freeze or binding_row:
        lines += [
            "## Freeze and deploy binding",
            "",
            "- frozen snapshot unchanged across deployment: `%s`" % freeze.get(
                "all_unchanged"),
            "- scored bytes == applied bytes on every arm: `%s`" % (
                binding_row.get("all_bound")),
            "",
        ]
    ledger = payload.get("budget_ledger") or {}
    if ledger:
        lines += [
            "## Budget",
            "",
            "- LLM calls: %s / %s (task-book cap %s)" % (
                ledger.get("llm_calls_used"),
                ledger.get("llm_call_budget_total"),
                ledger.get("task_book_cap"),
            ),
            "- LLM by arm: %s" % ledger.get("llm_by_arm"),
            "- Consumer retrains: %s" % ledger.get("consumer_retrains_total"),
            "- Consumer retrains by arm: %s" % ledger.get(
                "consumer_retrains_by_arm"),
            "- Downloads: %s; sealed reads: %s" % (
                ledger.get("downloads"), ledger.get("sealed_reads")),
            "",
        ]
    lines += [
        "## Wall",
        "",
        "%.1f seconds." % float(payload.get("wall_seconds") or 0.0),
        "",
    ]
    note = payload.get("post_run_annotation") or {}
    if note:
        lines += [
            "## Post-run annotation (0 evaluation)",
            "",
            "%s. Changes no measured number: `%s`." % (
                note.get("written"), note.get("changes_no_measured_number")),
            "",
        ]
        for row in note.get("readings") or []:
            lines += ["### %s" % row["title"], "", str(row["text"]), ""]
        findings = note.get("out_of_book_findings") or []
        if findings:
            lines += ["### Out-of-book findings (reported, not fixed)", ""]
            for row in findings:
                lines.append("- **%s** -- %s" % (row["id"], row["text"]))
            lines.append("")
        obligations = note.get("obligation_self_report") or {}
        if obligations:
            lines += ["### Obligation self-report", ""]
            for key, value in obligations.items():
                lines.append("- %s: `%s`" % (key, value))
            lines.append("")
    return "\n".join(lines) + "\n"


def _write(payload: Mapping[str, Any]) -> int:
    body = fc._public(payload)
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(body), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    verdict = (body.get("verdict") or {}).get("verdict")
    print("verdict", verdict, flush=True)
    print("llm", (body.get("budget_ledger") or {}).get("llm_calls_used"),
          flush=True)
    print("retrains",
          (body.get("budget_ledger") or {}).get("consumer_retrains_total"),
          flush=True)
    return 0 if verdict in (
        "CHAIN_REPRODUCED", "CHAIN_OK_A5_NO_ADVANTAGE") else 1


def _markdown_b(payload: Mapping[str, Any]) -> str:
    verdict = payload.get("verdict") or {}
    lines = [
        "# #45-Frep-b: symmetric zero-feedback deployment, terminal re-adjudication",
        "",
        "**Verdict: `%s`**" % verdict.get("verdict"),
        "",
        str(verdict.get("reason") or ""),
        "",
        "Development data, already-exposed outcomes, one paired draw. "
        "Held-in was not re-run: the #45-Frep frozen snapshots and held-in "
        "ledger are reused unchanged.",
        "",
    ]
    rule = payload.get("deploy_rule") or {}
    if rule:
        lines += [
            "## The repaired deployment semantics",
            "",
            "**F1** -- the deployed Workflow is fixed from frozen state before "
            "any outcome is touched. The inherited v2 adoption ladder, with "
            "its held-out delayed `bar` and `confirmation` reads, is never "
            "called. Exactly one delayed opening happens per arm, after the "
            "bytes are already fixed, and it is the evaluator's one-shot "
            "scoring.",
            "",
            "**F2** -- both arms deploy recall-only. An arm with an applicable "
            "ACTIVE Skill deploys the Workflow that Skill governs; an arm "
            "without one deploys the standing incumbent of its frozen ledger "
            "(the last held-in round's adopted plan), or identity if none "
            "stands. Forbidden on the scored block: %s."
            % "; ".join(rule.get("forbidden_on_the_scored_block") or []),
            "",
        ]
    verify = payload.get("frozen_snapshot_verification") or {}
    if verify.get("checks"):
        lines += [
            "## Frozen snapshot verification",
            "",
            "| slot | active sha | recompiles to itself | matches #45-Frep | "
            "skill ids match |",
            "| --- | --- | --- | --- | --- |",
        ]
        for slot, row in verify["checks"].items():
            lines.append("| `%s` | `%s` | `%s` | `%s` | `%s` |" % (
                slot, str(row.get("active_runtime_bundle_sha") or "--")[:16],
                row.get("recompile_reproduces_active_sha"),
                row.get("active_matches_frep_report"),
                row.get("skill_ids_match_frep_report"),
            ))
        lines.append("")
    purity = payload.get("deploy_purity_F1_F2") or {}
    if purity.get("slots"):
        lines += [
            "## Four-arm deployment ledger (F1/F2 discharge and cost symmetry)",
            "",
            "| slot | delayed openings | used for adoption | charged Support "
            "evals | candidate evals | LLM | deploy retrains | pure |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for slot, row in purity["slots"].items():
            lines.append("| `%s` | %s | %s | %s | %s | %s | %s | `%s` |" % (
                slot, row["F1_delayed_openings_total"],
                row["F1_delayed_openings_used_for_adoption"],
                row["F2_charged_support_evaluations"],
                row["F2_candidate_evaluations"], row["F2_llm_calls"],
                row["F2_deploy_consumer_retrains"], row["ok"],
            ))
        lines += [
            "",
            "Deployment cost is identical across all four arms: `%s` "
            "(values %s, spread %s). %s" % (
                purity.get("deploy_cost_identical_across_arms"),
                purity.get("deploy_cost_values"),
                purity.get("deploy_cost_spread"),
                purity.get("cost_symmetry_note"),
            ),
            "",
        ]
    rows = payload.get("paired_readouts") or []
    if rows:
        lines += [
            "## Paired four-readout table",
            "",
            "| cell | arm | first-pos cost (retrains) | first-pos LLM | "
            "total LLM/fit | deploy source | applied Workflow | "
            "held-out terminal utility | harm |",
            "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                "| `%s` | `%s` | %s | %s | %s LLM / %s fit | `%s` | %s | %s | %s |"
                % (
                    row["consumer_variant"], row["arm"],
                    row["first_positive_cost_consumer_retrains"],
                    row["first_positive_llm_calls"],
                    row["total_llm_calls"], row["total_consumer_retrains"],
                    row["deploy_source"],
                    _plan_label(row["held_out_plan"]),
                    _fmt(row["held_out_terminal_utility"]),
                    row["held_out_harm_count"],
                )
            )
        lines.append("")
    cells = payload.get("cells") or {}
    if cells:
        lines += [
            "## Re-adjudicated terminal comparison, against the voided reading",
            "",
            "| cell | A5 terminal | A3 terminal | difference | direction | "
            "same Workflow | voided #45-Frep difference |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: |",
        ]
        for variant, cell in cells.items():
            voided = cell["voided_frep_reading"]
            lines.append("| `%s` | %s | %s | %s | `%s` | `%s` | %s |" % (
                variant,
                _fmt(cell["A5"]["held_out_terminal_utility"]),
                _fmt(cell["A3"]["held_out_terminal_utility"]),
                _fmt(cell["held_out_terminal_difference"]),
                cell["held_out_terminal_direction"],
                cell["same_applied_workflow"],
                _fmt(voided["held_out_delayed_difference"]),
            ))
        lines += [
            "",
            "### Why the voided reading differed",
            "",
        ]
        for variant, cell in cells.items():
            voided = cell["voided_frep_reading"]
            lines.append(
                "- `%s`: under the defective protocol A5 deployed via `%s` for "
                "%s at %s retrains and A3 via `%s` for %s at %s retrains, raw "
                "cell verdict `%s`. Voided because %s." % (
                    variant, voided["A5_deploy_mode"],
                    _fmt(voided["A5_terminal"]), voided["A5_deploy_retrains"],
                    voided["A3_deploy_mode"], _fmt(voided["A3_terminal"]),
                    voided["A3_deploy_retrains"], voided["raw_cell_verdict"],
                    voided["voided_because"],
                )
            )
        lines.append("")
    ledger = payload.get("held_in_ledger_reused") or {}
    if ledger:
        lines += [
            "## Held-in carry-over",
            "",
            "First-positive cost is unchanged from #45-Frep on every arm: "
            "`%s`. %s" % (
                ledger.get("all_first_positive_invariant"),
                ledger.get("why_invariant"),
            ),
            "",
        ]
    budget = payload.get("budget_ledger") or {}
    if budget:
        lines += [
            "## Budget",
            "",
            "- this book's LLM calls: **%s** (%s)" % (
                budget.get("this_book_llm_calls"),
                budget.get("this_book_llm_note")),
            "- this book's Consumer retrains: %s, by arm %s" % (
                budget.get("this_book_consumer_retrains"),
                budget.get("this_book_consumer_retrains_by_arm")),
            "- carried over from #45-Frep held-in: %s LLM / %s retrains" % (
                budget.get("carried_over_held_in_llm_calls"),
                budget.get("carried_over_held_in_consumer_retrains")),
            "- downloads: %s; sealed reads: %s; held-in re-runs: %s" % (
                budget.get("downloads"), budget.get("sealed_reads"),
                budget.get("held_in_re_runs")),
            "",
        ]
    lines += [
        "## Freeze and binding",
        "",
        "- frozen snapshot unchanged across deployment: `%s`" % (
            (payload.get("freeze_unchanged_after_deploy") or {}).get(
                "all_unchanged")),
        "- scored bytes == applied bytes on every arm: `%s`" % (
            (payload.get("deploy_binding_assertion_0b") or {}).get(
                "all_bound")),
        "",
        "## Wall",
        "",
        "%.1f seconds." % float(payload.get("wall_seconds") or 0.0),
    ]
    note = payload.get("post_run_annotation") or {}
    if note:
        lines += [
            "",
            "## Post-run annotation (0 evaluation)",
            "",
            "%s. Changes no measured number: `%s`." % (
                note.get("written"), note.get("changes_no_measured_number")),
            "",
        ]
        for row in note.get("readings") or []:
            lines += ["### %s" % row["title"], "", str(row["text"]), ""]
        findings = note.get("out_of_book_findings") or []
        if findings:
            lines += ["### Out-of-book findings (reported, not fixed)", ""]
            for row in findings:
                lines.append("- **%s** -- %s" % (row["id"], row["text"]))
            lines.append("")
        obligations = note.get("obligation_self_report") or {}
        if obligations:
            lines += ["### Obligation self-report", ""]
            for key, value in obligations.items():
                lines.append("- %s: `%s`" % (key, value))
            lines.append("")
    return "\n".join(lines) + "\n"


def _write_b(payload: Mapping[str, Any]) -> int:
    body = fc._public(payload)
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON_B.write_text(
        json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD_B.write_text(_markdown_b(body), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON_B, flush=True)
    print("wrote", OUT_MD_B, flush=True)
    verdict = (body.get("verdict") or {}).get("verdict")
    print("verdict", verdict, flush=True)
    print("this-book llm",
          (body.get("budget_ledger") or {}).get("this_book_llm_calls"),
          flush=True)
    print("this-book retrains",
          (body.get("budget_ledger") or {}).get("this_book_consumer_retrains"),
          flush=True)
    return 0 if verdict == "A5A3_TERMINAL_READJUDICATED" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="frep45_r1")
    parser.add_argument(
        "--frep-b", action="store_true",
        help="#45-Frep-b: re-deploy and re-score only, on the frozen "
             "snapshots of --from-run-id; held-in is not re-run",
    )
    parser.add_argument("--from-run-id", default="frep45_r1")
    args = parser.parse_args(argv)
    if args.frep_b:
        return run_b(str(args.from_run_id))
    return run(str(args.run_id))


if __name__ == "__main__":
    raise SystemExit(main())
