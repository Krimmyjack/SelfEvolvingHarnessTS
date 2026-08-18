"""T2b: add one fault/program family (gap -> impute_ema) to the Source bank.

Minimal extension authorized by the A5_A3_ARM_DISTINCTION_INERT result:

* keep forecast/Ridge/sMASE, Gate, Memory, Episode and LLM unchanged;
* keep impulsive-outlier -> outlier_mad;
* add exactly one new context family (missing gap) and one existing
  zero-LLM-positive-control-confirmed workflow (impute_ema);
* actual attempts all write Episodes; no artificial negative labels.
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
    inject_gap_corpus,
)
from evaluation.functional.task_episode_harness.runner import (
    DELAYED_ORIGINS,
    MATERIAL_THRESHOLD,
    REPORT_REL,
    SUPPORT_ORIGINS,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t1 import (
    T1_MAX_PROBES,
    T1_OBSERVATION_CUTOFF,
    TASK_CONSUMER_KEY,
    _task_probe,
    _update_episode_delayed,
)
from evaluation.functional.task_episode_harness.t2 import (
    ALL_TRAIN,
    BASE_SERIES_OVERLAP_WITH_TARGET,
    TRANSFER_SCOPE,
)
from SelfEvolvingHarnessTS.contracts.observables import observable_numeric_bin
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_SUPPORT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_DRAFT,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    extract_public_features,
)

T2B_GAP_RECIPES = {
    "t2b01": {"faulty": ("T117", "T118", "T119", "T12", "T120", "T121"), "seed": 11},
    "t2b02": {"faulty": ("T122", "T123", "T124", "T125", "T126", "T127"), "seed": 11},
    "t2b03": {"faulty": ("T117", "T119", "T12", "T121", "T123", "T125"), "seed": 17},
}
T2B_GAP_COUNT = 80
T2B_GAP_WORKFLOW = "impute_ema"
T2B_OUTLIER_WORKFLOW = "outlier_mad"
T2B_PROGRAM_ORDER = (T2B_GAP_WORKFLOW, T2B_OUTLIER_WORKFLOW)
T2B_CONTEXT_CLASS = "missing_gap"


def _gap_scope_proposal(values: dict[str, Any], train_uids: list[str]) -> dict[str, Any]:
    observations = {}
    selected = set()
    for uid in train_uids:
        prefix = np.asarray(values[uid], dtype=np.float64)[:T1_OBSERVATION_CUTOFF]
        features = dict(extract_public_features(prefix, task_kind="forecast"))
        missing_fraction = float(features["missing_fraction"])
        missing_bin = observable_numeric_bin("missing_fraction", missing_fraction)
        observations[uid] = {
            "missing_fraction": missing_fraction,
            "missing_fraction_bin": missing_bin,
            "longest_missing_run_fraction": float(
                features["longest_missing_run_fraction"]
            ),
        }
        if missing_bin != "zero":
            selected.add(uid)
    return {
        "scope": frozenset(selected),
        "observations": observations,
        "rule": "missing_fraction bin != zero",
        "context_class": T2B_CONTEXT_CLASS,
    }


def _make_gap_episode(
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
        episode_id=f"t2b_{recipe_id}_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace="kdd2018-recipe-transfer-development",
        context_summary={
            "task_episode_id": f"t2b-source-{recipe_id}",
            "attempt_index": attempt_index,
            "observations_used": ["missing_fraction"],
            "context_class": T2B_CONTEXT_CLASS,
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
                "scope_observation_bin": "nonzero_missing_fraction",
                "scope_observation_mean_missing_fraction": float(np.mean([
                    float(observations[uid]["missing_fraction"])
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
        evidence_refs=["task_episode_harness_t2b"],
    )


def run_t2b(report_path: Path = REPORT_REL) -> dict[str, Any]:
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
    gap_episodes = []
    for recipe_id, recipe in T2B_GAP_RECIPES.items():
        clean = tuple(uid for uid in ALL_TRAIN if uid not in recipe["faulty"])
        injected, ground_truth = inject_gap_corpus(
            values,
            faulty_series=recipe["faulty"],
            clean_series=clean,
            count=T2B_GAP_COUNT,
            seed=recipe["seed"],
        )
        agent = _gap_scope_proposal(injected, train_uids)
        scope = agent["scope"]
        attempts = []
        winner = None
        for attempt_index, program in enumerate(T2B_PROGRAM_ORDER[:T1_MAX_PROBES]):
            probe = _task_probe(
                mapped_roster,
                injected,
                config,
                SUPPORT_ORIGINS,
                eval_uids,
                program,
                scope,
            )
            episode = _make_gap_episode(
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
                break
        delayed = None
        if winner is not None:
            delayed = _task_probe(
                mapped_roster,
                injected,
                config,
                DELAYED_ORIGINS,
                eval_uids,
                winner.workflow_signature,
                scope,
            )
            updated = _update_episode_delayed(
                winner,
                float(delayed["macro_gain"]),
                delayed_se_block=float(delayed["se_block"]),
                delayed_gain_over_se=delayed["gain_over_se"],
            )
            for attempt in attempts:
                if attempt["episode"]["episode_id"] == winner.episode_id:
                    attempt["episode"] = updated.to_dict()
            winner = updated
        task = {
            "recipe_id": recipe_id,
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
            "private_audit_only": {
                "oracle_scope": list(recipe["faulty"]),
                "ground_truth": ground_truth,
                "injection_seed": recipe["seed"],
                "gap_count": T2B_GAP_COUNT,
            },
        }
        tasks.append(task)
        gap_episodes.extend(attempt["episode"] for attempt in attempts)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    old_episodes = (report.get("source_bank") or {}).get("episodes") or []
    for ep in old_episodes:
        if "context_class" not in ep.get("context_summary", {}):
            ep["context_summary"]["context_class"] = "impulsive_outlier"
    merged = old_episodes + gap_episodes

    # New sufficiency condition: at least two observable Context classes with
    # different effective workflows.
    effective_pairs = {
        (
            ep.get("context_summary", {}).get("context_class"),
            ep.get("workflow_signature"),
        )
        for ep in merged
        if ep.get("relation") == "POSITIVE"
    }
    sufficient = len(effective_pairs) >= 2
    counts = {
        "positive_count": sum(1 for ep in merged if ep["relation"] == "POSITIVE"),
        "negative_count": sum(1 for ep in merged if ep["relation"] == "NEGATIVE"),
        "conflict_count": sum(1 for ep in merged if ep["relation"] == "CONFLICT"),
        "abstain_count": sum(1 for ep in merged if ep["relation"] == "ABSTAIN"),
    }
    verdict = (
        "CONTEXT_VARIED_EPISODES_RECORDED"
        if sufficient
        else "CONTEXT_VARIED_SOURCE_BANK_INSUFFICIENT"
    )
    t2b = {
        "added_family": T2B_CONTEXT_CLASS,
        "added_workflow": T2B_GAP_WORKFLOW,
        "program_pool": [T2B_OUTLIER_WORKFLOW, T2B_GAP_WORKFLOW],
        "tasks": tasks,
        "effective_context_workflow_pairs": sorted(effective_pairs),
        "sufficient_two_contexts": sufficient,
        "gap_utility": "GAP_UTILITY_UNCERTAIN",
        "verdict": verdict,
        "wall_seconds": time.perf_counter() - started,
        "llm_api_call_count": 0,
    }
    report["phase"] = "T2b"
    report["t2b"] = t2b
    report["source_bank"] = {
        "episodes": merged,
        "bank_label": (
            "POSITIVE_ONLY_RECIPE_BANK"
            if counts["negative_count"] == 0 and counts["conflict_count"] == 0
            else "SIGNED_SOURCE_BANK"
        ),
        **counts,
    }
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return t2b
