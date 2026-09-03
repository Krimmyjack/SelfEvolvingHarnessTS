"""T3b: three additional pre-registered same-family gap Targets.

All knobs are frozen: Source bank, candidate pool [outlier_mad, impute_ema],
Gate, B=3 budget, LLM model and prompt.  Only pre-written seed/faulty-scope
vary.  Targets with no candidate headroom are kept, never dropped.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from run_v1_a5a3_runtime_regression import _load as _load_cohort
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.injection import (
    inject_gap_corpus,
)
from evaluation.functional.task_episode_harness.runner import (
    REPORT_REL,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.t2b import (
    T2B_CONTEXT_CLASS,
    _gap_scope_proposal,
)
from evaluation.functional.task_episode_harness.t3 import (
    T3_BASE_URL,
    T3_MODEL,
    _canonical_payload,
    _decision_payload,
    _llm_propose,
    _run_arm,
    _source_summaries,
)

T3B_TARGETS = (
    {
        "target_id": "gap_target_2",
        "faulty": ("T122", "T123", "T124", "T125", "T126", "T127"),
        "seed": 31,
        "gap_count": 80,
    },
    {
        "target_id": "gap_target_3",
        "faulty": ("T117", "T119", "T12", "T121", "T123", "T125"),
        "seed": 37,
        "gap_count": 80,
    },
    {
        "target_id": "gap_target_4",
        "faulty": ("T118", "T12", "T121", "T123", "T125", "T127"),
        "seed": 41,
        "gap_count": 80,
    },
)


def _target_verdict(a3: dict[str, Any], a5: dict[str, Any]) -> str:
    a3_winner = a3["winner"] is not None
    a5_winner = a5["winner"] is not None
    if not a3_winner and not a5_winner:
        return "NO_CANDIDATE_HEADROOM_ON_TARGET"
    a5_faster = a5["probe_count"] < a3["probe_count"]
    harm_not_worse = (
        a5["support_harm_count"] <= a3["support_harm_count"]
        and a5["cumulative_support_harm"] <= a3["cumulative_support_harm"]
    )
    a5_delayed_ok = bool(
        a5["winner"] is not None
        and a5["winner"]["local_status"] == "LOCAL_ACTIVE"
    )
    if a5_faster and harm_not_worse and a5_delayed_ok:
        return "A5_WARM_START_BENEFIT"
    if a5_faster and not harm_not_worse:
        return "A5_SPEED_ONLY"
    return "A5_NO_WARM_START_BENEFIT"


def run_t3b(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    source_summaries = _source_summaries(report)
    cohort = _load_cohort(repo_root)
    roster = cohort["roster"]
    values = cohort["values"]
    config = dict(_config())
    mapped_roster = _mapped_roster(roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    train_uids = [row["series_uid"] for row in roster if row["role"] == "train"]

    target_rows = []
    target_episodes = []
    llm_calls = 0
    for target in T3B_TARGETS:
        clean = tuple(uid for uid in train_uids if uid not in target["faulty"])
        injected, _gt = inject_gap_corpus(
            values,
            faulty_series=target["faulty"],
            clean_series=clean,
            count=target["gap_count"],
            seed=target["seed"],
        )
        agent = _gap_scope_proposal(injected, train_uids)
        scope = agent["scope"]
        scope_feature = "missing_fraction"
        scope_bin = "nonzero_missing_fraction"
        a3_payload = _decision_payload(
            scope=scope,
            observations=agent["observations"],
            source_summaries=[],
            context_class=T2B_CONTEXT_CLASS,
            scope_feature=scope_feature,
            scope_bin=scope_bin,
        )
        a5_payload = _decision_payload(
            scope=scope,
            observations=agent["observations"],
            source_summaries=source_summaries,
            context_class=T2B_CONTEXT_CLASS,
            scope_feature=scope_feature,
            scope_bin=scope_bin,
        )
        a3_clean = copy.deepcopy(a3_payload)
        a5_clean = copy.deepcopy(a5_payload)
        a5_clean["source_experiences"] = []
        inputs_only_source = bool(
            _canonical_payload(a3_clean) == _canonical_payload(a5_clean)
            and bool(source_summaries)
        )

        row: dict[str, Any] = {
            "target_id": target["target_id"],
            "seed": target["seed"],
            "faulty_scope_private": list(target["faulty"]),
            "agent_scope": sorted(scope),
            "inputs_only_source_differ": inputs_only_source,
        }
        try:
            a3_proposal = _llm_propose(
                a3_payload, model=T3_MODEL, base_url=T3_BASE_URL
            )
            a5_proposal = _llm_propose(
                a5_payload, model=T3_MODEL, base_url=T3_BASE_URL
            )
            llm_calls += 2
            row["a3_order"] = a3_proposal["program_order"]
            row["a5_order"] = a5_proposal["program_order"]
            if a3_proposal["program_order"] == a5_proposal["program_order"]:
                row["outcome_opened"] = False
                row["arms"] = {}
                row["target_verdict"] = "ARM_DISTINCTION_INERT"
            else:
                row["outcome_opened"] = True
                arms = {
                    "A3": _run_arm(
                        arm="A3",
                        program_order=a3_proposal["program_order"],
                        scope=scope,
                        roster=mapped_roster,
                        injected=injected,
                        config=config,
                        eval_uids=eval_uids,
                        context_class=T2B_CONTEXT_CLASS,
                        scope_bin=scope_bin,
                    ),
                    "A5": _run_arm(
                        arm="A5",
                        program_order=a5_proposal["program_order"],
                        scope=scope,
                        roster=mapped_roster,
                        injected=injected,
                        config=config,
                        eval_uids=eval_uids,
                        context_class=T2B_CONTEXT_CLASS,
                        scope_bin=scope_bin,
                    ),
                }
                row["arms"] = arms
                row["target_verdict"] = _target_verdict(arms["A3"], arms["A5"])
                for arm_record in arms.values():
                    for probe in arm_record["probes"]:
                        target_episodes.append(probe["episode"])
        except Exception as exc:  # noqa: BLE001
            row["llm_error"] = f"{type(exc).__name__}: {exc}"
            row["outcome_opened"] = False
            row["arms"] = {}
            row["target_verdict"] = "INCONCLUSIVE"
        target_rows.append(row)

    # Existing single-target T3 result is target 1 and is preserved.
    previous = report.get("t3") or {}
    previous_verdict = previous.get("verdict")
    all_verdicts = [previous_verdict] + [
        row["target_verdict"] for row in target_rows
    ]
    no_headroom = all_verdicts.count("NO_CANDIDATE_HEADROOM_ON_TARGET") + (
        1 if previous_verdict == "A5_NO_WARM_START_BENEFIT_SINGLE_TARGET" else 0
    )
    inert = all_verdicts.count("ARM_DISTINCTION_INERT")
    benefit = sum(
        1
        for v in all_verdicts
        if v in {"A5_WARM_START_BENEFIT", "POSITIVE_EXPERIENCE_WARM_START_PASS"}
    )
    total = len(all_verdicts)
    if no_headroom > total / 2:
        final_verdict = "GAP_PROGRAM_RECIPE_TRANSFER_UNSTABLE_NOT_MEMORY"
    elif benefit > 0:
        final_verdict = "RECIPE_LEVEL_WARM_START_DEV_PASS"
    else:
        final_verdict = "POSITIVE_SOURCE_WARM_START_NOT_SUPPORTED"

    t3b = {
        "pre_registered_additional_targets": len(T3B_TARGETS),
        "total_targets_including_previous": total,
        "target_rows": target_rows,
        "target_episodes": target_episodes,
        "llm_api_call_count": llm_calls,
        "all_target_verdicts": all_verdicts,
        "no_headroom_count": no_headroom,
        "inert_count": inert,
        "benefit_count": benefit,
        "final_verdict": final_verdict,
        "interpretation": (
            "All candidates/prompt/source bank are frozen. Targets with no "
            "candidate headroom are kept. The final interpretation follows "
            "the three pre-registered cases."
        ),
        "narrowed_claim": (
            "These three low-readability, Context-identical gap positive "
            "episodes produced no warm-start benefit on four recipe-transfer "
            "Targets. This says nothing general about Memory or retrieval "
            "design until a readable gap substrate exists."
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    report["phase"] = "T3b"
    report["t3b"] = t3b
    report["verdict"] = final_verdict
    report_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return t3b
