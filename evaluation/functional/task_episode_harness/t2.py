"""T2: six pre-registered RECIPE_TRANSFER_ONLY Source Tasks (zero LLM).

All six tasks share the K1 base series pool with the T1 Target task, so the
bank is explicitly development-only recipe-level reuse, never cross-dataset or
cross-series transfer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.injection import (
    inject_label_touched_corpus,
)
from evaluation.functional.task_episode_harness.runner import (
    DELAYED_ORIGINS,
    INJECTION_AMPLITUDE,
    INJECTION_COUNT,
    MATERIAL_THRESHOLD,
    REPORT_REL,
    SUPPORT_ORIGINS,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import (
    T1_MAX_PROBES,
    T1_SCOPE_BIN,
    T1_SCOPE_FEATURE,
    TASK_CONSUMER_KEY,
    _public_scope_proposal,
    _task_probe,
    _update_episode_delayed,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_SUPPORT,
    RELATION_ABSTAIN,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_DRAFT,
    build_episode,
)

ALL_TRAIN = (
    "T117", "T118", "T119", "T12", "T120", "T121",
    "T122", "T123", "T124", "T125", "T126", "T127",
)
# Frozen before outcomes.  Six recipes only; no post-hoc additions allowed.
T2_RECIPES = {
    "t2s01": {"faulty": ("T117", "T118", "T119", "T12", "T120", "T121"), "seed": 7},
    "t2s02": {"faulty": ("T122", "T123", "T124", "T125", "T126", "T127"), "seed": 7},
    "t2s03": {"faulty": ("T117", "T119", "T12", "T121", "T123", "T125"), "seed": 7},
    "t2s04": {"faulty": ("T118", "T12", "T121", "T123", "T125", "T127"), "seed": 7},
    "t2s05": {"faulty": ("T117", "T118", "T119", "T12", "T120", "T121"), "seed": 11},
    "t2s06": {"faulty": ("T122", "T123", "T124", "T125", "T126", "T127"), "seed": 23},
}
BASE_SERIES_OVERLAP_WITH_TARGET = True
TRANSFER_SCOPE = "RECIPE_TRANSFER_ONLY"


def _make_source_episode(
    *,
    recipe_id: str,
    attempt_index: int,
    program: str,
    scope: frozenset[str],
    observations: dict[str, Any],
    probe: dict[str, Any],
) -> Any:
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    return build_episode(
        episode_id=f"t2_source_{recipe_id}_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-recipe-transfer-development",
        context_summary={
            "task_episode_id": f"t2-source-{recipe_id}",
            "attempt_index": attempt_index,
            "observations_used": [T1_SCOPE_FEATURE],
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "base_series_overlap_with_target": BASE_SERIES_OVERLAP_WITH_TARGET,
            "transfer_scope": TRANSFER_SCOPE,
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {
                "scope_observation_bin": T1_SCOPE_BIN,
                "scope_observation_mean_z": float(np.mean([
                    float(observations[uid]["local_robust_z_peak"])
                    for uid in scope
                ])) if scope else 0.0,
            },
            "program_geometry": {
                "scope": "training_series_subset",
                "program_steps": [{"op": program, "params": {}}],
            },
        },
        workflow_signature=program,
        support_response={
            "gain": gain,
            "se_block": float(probe["se_block"]),
            "gain_over_se": probe["gain_over_se"],
            "accepted": positive,
            "block_origins": list(SUPPORT_ORIGINS),
        },
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=RELATION_POSITIVE if positive else RELATION_NEGATIVE,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT if positive else STATUS_EPISODE_ONLY,
        evidence_refs=["task_episode_harness_t2"],
    )


def _make_abstain_episode(*, recipe_id: str, reason: str) -> Any:
    return build_episode(
        episode_id=f"t2_source_{recipe_id}_abstain",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-recipe-transfer-development",
        context_summary={
            "task_episode_id": f"t2-source-{recipe_id}",
            "attempt_index": 0,
            "observations_used": [T1_SCOPE_FEATURE],
            "scope_summary": {
                "training_series_count": 0,
                "training_series_uids": [],
            },
            "base_series_overlap_with_target": BASE_SERIES_OVERLAP_WITH_TARGET,
            "transfer_scope": TRANSFER_SCOPE,
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {},
            "program_geometry": {"scope": "none", "program_steps": []},
        },
        workflow_signature="identity",
        support_response={"gain": None, "se_block": None,
                          "gain_over_se": None, "accepted": False},
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=RELATION_ABSTAIN,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_EPISODE_ONLY,
        evidence_refs=["task_episode_harness_t2", reason],
    )


def _run_one_source_task(
    *,
    recipe_id: str,
    faulty: tuple[str, ...],
    seed: int,
    roster: list[dict[str, Any]],
    mapped_roster: list[dict[str, Any]],
    values: dict[str, Any],
    config: dict[str, Any],
    eval_uids: list[str],
    train_uids: list[str],
) -> dict[str, Any]:
    clean = tuple(uid for uid in ALL_TRAIN if uid not in faulty)
    injected, ground_truth = inject_label_touched_corpus(
        values,
        faulty_series=faulty,
        clean_series=clean,
        amplitude=INJECTION_AMPLITUDE,
        count=INJECTION_COUNT,
        seed=seed,
    )
    agent = _public_scope_proposal(injected, train_uids)
    scope = agent["scope"]
    attempts = []
    winner = None
    winner_probe = None
    if not scope:
        episode = _make_abstain_episode(
            recipe_id=recipe_id, reason="no_high_bin_scope"
        )
        attempts.append({
            "attempt_index": 0,
            "program": "identity",
            "scope": [],
            "support": None,
            "episode": episode.to_dict(),
        })
    else:
        for attempt_index, program in enumerate(
            agent["program_order"][:T1_MAX_PROBES]
        ):
            probe = _task_probe(
                mapped_roster,
                injected,
                config,
                SUPPORT_ORIGINS,
                eval_uids,
                program,
                scope,
            )
            episode = _make_source_episode(
                recipe_id=recipe_id,
                attempt_index=attempt_index,
                program=program,
                scope=scope,
                observations=agent["observations"],
                probe=probe,
            )
            attempts.append({
                "attempt_index": attempt_index,
                "program": program,
                "scope": sorted(scope),
                "support": probe,
                "episode": episode.to_dict(),
            })
            if probe["macro_gain"] >= MATERIAL_THRESHOLD:
                winner = episode
                winner_probe = probe
                break

    delayed: dict[str, Any] | None = None
    if winner is not None and winner_probe is not None:
        delayed_probe = _task_probe(
            mapped_roster,
            injected,
            config,
            DELAYED_ORIGINS,
            eval_uids,
            winner.workflow_signature,
            scope,
        )
        delayed = delayed_probe
        updated_winner = _update_episode_delayed(
            winner,
            float(delayed_probe["macro_gain"]),
            delayed_se_block=float(delayed_probe["se_block"]),
            delayed_gain_over_se=delayed_probe["gain_over_se"],
        )
        for attempt in attempts:
            if attempt["episode"]["episode_id"] == winner.episode_id:
                attempt["episode"] = updated_winner.to_dict()
        winner = updated_winner

    final_episodes = [attempt["episode"] for attempt in attempts]
    return {
        "recipe_id": recipe_id,
        "private_oracle_scope": list(faulty),
        "injection_seed": seed,
        "ground_truth": ground_truth,
        "agent_scope": sorted(scope),
        "attempts": attempts,
        "winner": (
            {
                "episode_id": winner.episode_id,
                "workflow": winner.workflow_signature,
            }
            if winner is not None else None
        ),
        "delayed": delayed,
        "final_episodes": final_episodes,
        "relations": [ep["relation"] for ep in final_episodes],
    }


def run_t2(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    cohort = _load_cohort(repo_root)
    roster = cohort["roster"]
    values = cohort["values"]
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]

    tasks = []
    all_episodes = []
    for recipe_id, recipe in T2_RECIPES.items():
        task = _run_one_source_task(
            recipe_id=recipe_id,
            faulty=recipe["faulty"],
            seed=recipe["seed"],
            roster=roster,
            mapped_roster=mapped_roster,
            values=values,
            config=config,
            eval_uids=eval_uids,
            train_uids=train_uids,
        )
        task["private_audit_only"] = {
            "oracle_scope": task.pop("private_oracle_scope"),
            "ground_truth": task.pop("ground_truth"),
            "injection_seed": task.pop("injection_seed"),
        }
        tasks.append(task)
        all_episodes.extend(task["final_episodes"])

    counts = {
        "positive_count": sum(1 for ep in all_episodes if ep["relation"] == "POSITIVE"),
        "negative_count": sum(1 for ep in all_episodes if ep["relation"] == "NEGATIVE"),
        "conflict_count": sum(1 for ep in all_episodes if ep["relation"] == "CONFLICT"),
        "abstain_count": sum(1 for ep in all_episodes if ep["relation"] == "ABSTAIN"),
    }
    sufficient = bool(
        counts["positive_count"] >= 1
        and counts["negative_count"] >= 1
        and counts["conflict_count"] >= 1
    )
    verdict = (
        "SIGNED_SOURCE_BANK_READY"
        if sufficient
        else "SIGNED_SOURCE_BANK_INSUFFICIENT"
    )
    t2 = {
        "recipes_frozen": True,
        "recipe_count": len(T2_RECIPES),
        "transfer_scope": TRANSFER_SCOPE,
        "base_series_overlap_with_target": BASE_SERIES_OVERLAP_WITH_TARGET,
        "bank_label": (
            "POSITIVE_ONLY_RECIPE_BANK"
            if counts["positive_count"] > 0
            and counts["negative_count"] == 0
            and counts["conflict_count"] == 0
            else "SIGNED_SOURCE_BANK"
        ),
        "narrow_claim": (
            "this bank can only support a positive-experience warm-start "
            "claim; it cannot support the full signed-memory or safety claim"
        ),
        "tasks": tasks,
        "source_bank": {
            "episodes": all_episodes,
            **counts,
        },
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - started,
        "llm_api_call_count": 0,
        "slow_api_call_count": 0,
    }

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {}
    report["phase"] = "T2"
    report["t2"] = t2
    report["source_bank"] = t2["source_bank"]
    report["verdict"] = verdict
    report["mechanical_checks"] = dict(
        report.get("mechanical_checks") or {},
        t2_llm_calls=0,
        t2_slow_calls=0,
        t2_source_recipes_frozen_before_outcomes=True,
        t2_recipe_transfer_only_marked=True,
    )
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return t2
