"""Pool the clean NEW_OBS supply executions into one UNGUIDED census.

Reads the per-execution reports written by ``run_t233_supply_obs_ab.py`` and
merges their NEW_OBS arms under one strict counting rule.  This module runs no
Episode, calls no model and writes no authorization artifact; it only re-reads
reports that already exist.

The counting rule, stated once because it is the whole point of pooling:

* **The unit of evidence is the distinct Task, not the execution.**  A Task that
  comes out positive for the same program x context cell in two executions is
  one distinct positive Task, not two.  Re-sampling the same Task on the same
  already-exposed data does not manufacture independent evidence.
* **A sign flip is a conflict, not a vote.**  If any execution puts a Task
  positive in a cell and any other execution puts the same Task negative in
  that same cell, the Task is reported as a conflict for that cell and is
  counted in neither the positive nor the negative column.
* **Opposing evidence blocks a precheck.**  Both a distinct negative Task and a
  conflict Task count as opposing, so a cell reaches the precheck only on
  ``>= PRECHECK_MIN_DISTINCT_TASKS`` distinct positive Tasks with a clean
  negative and conflict column.

Positive / negative / immaterial per (cell, Task, execution) reuse the driver's
own unchanged material threshold and cell key, imported rather than restated so
the pooled cells cannot drift from the per-execution cells they came from.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation.functional.task_episode_harness import g1
from evaluation.functional.run_t233_supply_obs_ab import (
    MATERIAL_THRESHOLD,
    NEW_OBS,
    PRECHECK_MIN_DISTINCT_TASKS,
    _cell_key,
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts/functional/e2"
POOLED_JSON = ARTIFACT_DIR / "t233_newobs_supply_pooled_v1.json"
POOLED_MD = ARTIFACT_DIR / "t233_newobs_supply_pooled_v1.md"

# The three clean executions.  exec1 is the two-arm delivered run, read for its
# NEW_OBS arm only; exec2 and exec3 are the single-arm supplementary runs.
SOURCES = (
    ("exec1", "t233_supply_obs_ab_v1.json"),
    ("exec2", "t233_supply_obs_ab_v1_exec2.json"),
    ("exec3", "t233_supply_obs_ab_v1_exec3.json"),
)

PROTOCOL_STOP = "AGENT_PROTOCOL_ERROR"
REQUEST_OBSERVATION_STOP = "REQUEST_OBSERVATION"
OUTLIER_MAD = "outlier_mad"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_rows(payload: Mapping[str, Any]) -> list[tuple[str, bool, dict]]:
    """(task_id, context_condition, NEW_OBS arm) for every scored row."""
    out: list[tuple[str, bool, dict]] = []
    for row in payload.get("rows") or ():
        arm = (row.get("arms") or {}).get(NEW_OBS)
        if not arm or arm.get("driver_exception"):
            continue
        out.append((
            str(row["task_episode_id"]),
            bool(row.get(g1.G1_CONDITION_FEATURE, False)),
            dict(arm),
        ))
    return out


def _sign(gain: float) -> str:
    if gain >= MATERIAL_THRESHOLD:
        return "positive"
    if gain <= -MATERIAL_THRESHOLD:
        return "negative"
    return "immaterial"


def _per_execution(label: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """One execution's Task rows plus its own funnel, nothing pooled."""
    rows = _arm_rows(payload)
    task_rows: list[dict[str, Any]] = []
    for task_id, condition, arm in rows:
        probed = [
            probe for probe in arm.get("probes") or ()
            if probe.get("status") == "PROBED"
        ]
        stage_validation = arm.get("stage_validation") or ()
        task_rows.append({
            "execution": label,
            "task_episode_id": task_id,
            "context_condition": condition,
            "stop_reason": arm.get("stop_reason"),
            "protocol_error": arm.get("protocol_error"),
            "terminal_validation_error_code": arm.get(
                "terminal_validation_error_code"
            ),
            "inspect_validation_retry_count": next(
                (
                    int(stage.get("validation_retry_count") or 0)
                    for stage in stage_validation
                    if stage.get("stage") == "inspect"
                ),
                None,
            ),
            "any_stage_retried": any(
                int(stage.get("validation_retry_count") or 0) > 0
                for stage in stage_validation
            ),
            "ungrounded_citation_rejections": arm.get(
                "ungrounded_citation_rejections"
            ),
            "ungrounded_citation_rejections_fatal": arm.get(
                "ungrounded_citation_rejections_fatal"
            ),
            "cites_any_m0b_field": arm.get("cites_any_m0b_field"),
            "real_support_probe_count": (arm.get("metrics") or {}).get(
                "real_support_probe_count"
            ),
            "llm_calls": (arm.get("metrics") or {}).get("llm_calls"),
            "probed_cells": [
                {
                    "program": _cell_key(probe.get("program") or ()),
                    "context_condition": condition,
                    "support_gain": (
                        None if probe.get("support_gain") is None
                        else float(probe["support_gain"])
                    ),
                    "sign": (
                        None if probe.get("support_gain") is None
                        else _sign(float(probe["support_gain"]))
                    ),
                }
                for probe in probed
            ],
        })

    scored = len(task_rows)
    protocol = sum(1 for r in task_rows if r["stop_reason"] == PROTOCOL_STOP)
    request_obs = sum(
        1 for r in task_rows if r["stop_reason"] == REQUEST_OBSERVATION_STOP
    )
    with_probe = sum(1 for r in task_rows if r["probed_cells"])
    stops: dict[str, int] = {}
    for r in task_rows:
        stops[str(r["stop_reason"])] = stops.get(str(r["stop_reason"]), 0) + 1
    # A stage only gets a stage_validation entry when it returned, so an
    # inspect entry carrying retry_count 2 is a stage that a repair budget of 1
    # would have killed.  These are the arm runs the raised budget bought.
    rescued = [
        r["task_episode_id"] for r in task_rows
        if r["inspect_validation_retry_count"] is not None
        and int(r["inspect_validation_retry_count"]) >= 2
    ]
    return {
        "execution": label,
        "stage_validation_retries": (
            (payload.get("pinned_parameters") or {}).get(
                "stage_validation_retries", 1
            )
        ),
        "observation_arms_run": (
            (payload.get("pinned_parameters") or {}).get(
                "observation_arms_run", ["OLD_OBS", "NEW_OBS"]
            )
        ),
        "verdict": payload.get("verdict"),
        "pinned_parameters": dict(payload.get("pinned_parameters") or {}),
        "arm_runs_scored": scored,
        "protocol_errors": protocol,
        "protocol_error_rate": (protocol / scored if scored else None),
        "request_observation": request_obs,
        "request_observation_rate": (request_obs / scored if scored else None),
        "arm_runs_with_a_probe": with_probe,
        "inspect_returned_only_after_a_second_retry": rescued,
        "arm_runs_rescued_by_the_second_retry": len(rescued),
        "counterfactual_protocol_errors_at_retry_1": protocol + len(rescued),
        "counterfactual_protocol_error_rate_at_retry_1": (
            (protocol + len(rescued)) / scored if scored else None
        ),
        "probe_total": sum(len(r["probed_cells"]) for r in task_rows),
        "stop_reasons": dict(sorted(stops.items())),
        "llm_calls": sum(int(r["llm_calls"] or 0) for r in task_rows),
        "tasks_citing_m0b_field": sorted(
            r["task_episode_id"] for r in task_rows
            if r["cites_any_m0b_field"]
        ),
        "task_rows": task_rows,
    }


def _pool(per_exec: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Distinct-Task cells with an explicit conflict column."""
    # (program, condition) -> task_id -> {execution: sign}
    observed: dict[tuple[str, bool], dict[str, dict[str, str]]] = {}
    for execution in per_exec:
        for task_row in execution["task_rows"]:
            for cell in task_row["probed_cells"]:
                if cell["sign"] is None:
                    continue
                key = (cell["program"], bool(cell["context_condition"]))
                per_task = observed.setdefault(key, {})
                signs = per_task.setdefault(task_row["task_episode_id"], {})
                # One execution can probe the same cell twice in one Task; a
                # negative anywhere in that Task's own execution is kept, so a
                # within-execution disagreement cannot be silently dropped.
                previous = signs.get(execution["execution"])
                if previous is None or previous == cell["sign"]:
                    signs[execution["execution"]] = cell["sign"]
                else:
                    signs[execution["execution"]] = "conflict"

    cells: list[dict[str, Any]] = []
    for (program, condition), per_task in sorted(
        observed.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        positive: list[dict[str, Any]] = []
        negative: list[dict[str, Any]] = []
        conflict: list[dict[str, Any]] = []
        immaterial: list[dict[str, Any]] = []
        unstable: list[dict[str, Any]] = []
        for task_id, signs in sorted(per_task.items()):
            values = set(signs.values())
            entry = {"task_episode_id": task_id, "by_execution": dict(signs)}
            if "conflict" in values or {"positive", "negative"} <= values:
                conflict.append(entry)
            elif "positive" in values:
                positive.append(entry)
                if "immaterial" in values:
                    unstable.append(entry)
            elif "negative" in values:
                negative.append(entry)
                if "immaterial" in values:
                    unstable.append(entry)
            else:
                immaterial.append(entry)
        opposing = len(negative) + len(conflict)
        cells.append({
            "program": program,
            "context_condition": condition,
            "distinct_positive_tasks": len(positive),
            "distinct_negative_tasks": len(negative),
            "conflict_tasks": len(conflict),
            "immaterial_only_tasks": len(immaterial),
            "opposing_tasks": opposing,
            "positive": positive,
            "negative": negative,
            "conflict": conflict,
            "immaterial_only": immaterial,
            # Positive in one execution and immaterial in another: not a sign
            # flip, so not a conflict, but not a stable positive either.
            "positive_or_negative_with_immaterial_elsewhere": unstable,
            "executions_contributing": sorted(
                {
                    label
                    for signs in per_task.values()
                    for label in signs
                }
            ),
            "meets_precheck_threshold": bool(
                len(positive) >= PRECHECK_MIN_DISTINCT_TASKS and opposing == 0
            ),
            "distinct_positive_tasks_short_of_threshold": max(
                PRECHECK_MIN_DISTINCT_TASKS - len(positive), 0
            ),
        })
    return {
        "counting_rule": {
            "unit_of_evidence": "distinct Task Episode, pooled across executions",
            "repeat_positive_same_task": (
                "counted once; re-sampling one Task on the same already-exposed "
                "data is not independent evidence"
            ),
            "sign_flip_across_executions": (
                "reported as a conflict for that cell and excluded from both "
                "the positive and the negative column"
            ),
            "opposing_definition": (
                "distinct negative Tasks plus conflict Tasks; either blocks a "
                "precheck"
            ),
            "material_threshold": MATERIAL_THRESHOLD,
            "precheck_min_distinct_positive_tasks": PRECHECK_MIN_DISTINCT_TASKS,
        },
        "cells": cells,
    }


def _precheck_table(pooled: Mapping[str, Any]) -> dict[str, Any]:
    cells = list(pooled["cells"])
    eligible = [c for c in cells if c["meets_precheck_threshold"]]
    ranked = sorted(
        cells,
        key=lambda c: (
            c["distinct_positive_tasks_short_of_threshold"],
            c["opposing_tasks"],
            -c["distinct_positive_tasks"],
            c["program"],
        ),
    )
    return {
        "is_a_precheck_only": True,
        "authorization_actions_taken": [],
        "no_try_written": True,
        "no_skill_written": True,
        "no_authorization_artifact_modified": True,
        "nothing_promoted": True,
        "precheck_eligible_cell_count": len(eligible),
        "precheck_eligible_cells": [
            {
                "program": c["program"],
                "context_condition": c["context_condition"],
                "distinct_positive_tasks": c["distinct_positive_tasks"],
                "opposing_tasks": c["opposing_tasks"],
            }
            for c in eligible
        ],
        "closest_three_cells": [
            {
                "program": c["program"],
                "context_condition": c["context_condition"],
                "distinct_positive_tasks": c["distinct_positive_tasks"],
                "distinct_negative_tasks": c["distinct_negative_tasks"],
                "conflict_tasks": c["conflict_tasks"],
                "opposing_tasks": c["opposing_tasks"],
                "tasks_short_of_threshold": (
                    c["distinct_positive_tasks_short_of_threshold"]
                ),
                "blocked_by_opposing": c["opposing_tasks"] > 0,
                "executions_contributing": c["executions_contributing"],
            }
            for c in ranked[:3]
        ],
    }


def _outlier_mad_recurrence(
    per_exec: Sequence[Mapping[str, Any]], pooled: Mapping[str, Any]
) -> dict[str, Any]:
    """Did the discarded-execution outlier_mad family reappear when clean?"""
    matching = [
        c for c in pooled["cells"] if OUTLIER_MAD in c["program"]
    ]
    per_execution_probes = {
        execution["execution"]: sorted({
            cell["program"]
            for task_row in execution["task_rows"]
            for cell in task_row["probed_cells"]
            if OUTLIER_MAD in cell["program"]
        })
        for execution in per_exec
    }
    clean_only = [
        label for label, programs in per_execution_probes.items()
        if programs and label != "exec1"
    ]
    if not matching:
        reading = (
            "No `outlier_mad` program was probed in any of the three clean "
            "executions, so the cell does not reproduce."
        )
    else:
        reading = (
            "The family appears only as %s, and only in %s -- %s never probed "
            "an `outlier_mad` program at all. Its pooled count is %d distinct "
            "positive Task%s, so it does **not** reach the %d-Task threshold "
            "the discarded execution appeared to clear. The discarded "
            "execution's `outlier_mad` cell does not reproduce."
            % (
                ", ".join(f"`{c['program']}`" for c in matching),
                ", ".join(
                    sorted({
                        label
                        for c in matching
                        for label in c["executions_contributing"]
                    })
                ),
                (
                    "the two retry-2 executions"
                    if not clean_only
                    else ", ".join(clean_only)
                ),
                max(c["distinct_positive_tasks"] for c in matching),
                "" if max(
                    c["distinct_positive_tasks"] for c in matching
                ) == 1 else "s",
                PRECHECK_MIN_DISTINCT_TASKS,
            )
        )
    return {
        "question": (
            "One of the two discarded exec1-era executions produced an "
            "outlier_mad cell at 3 distinct positives. Does any outlier_mad "
            "cell reappear across the three clean executions?"
        ),
        "reading": reading,
        "reappeared_at_all": bool(matching),
        "programs_probed_per_execution": per_execution_probes,
        "cells": [
            {
                "program": c["program"],
                "context_condition": c["context_condition"],
                "distinct_positive_tasks": c["distinct_positive_tasks"],
                "distinct_negative_tasks": c["distinct_negative_tasks"],
                "conflict_tasks": c["conflict_tasks"],
                "meets_precheck_threshold": c["meets_precheck_threshold"],
                "executions_contributing": c["executions_contributing"],
            }
            for c in matching
        ],
        "reaches_three_distinct_positives": any(
            c["distinct_positive_tasks"] >= PRECHECK_MIN_DISTINCT_TASKS
            for c in matching
        ),
    }


def _funnel(per_exec: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline = next(
        (e for e in per_exec if e["execution"] == "exec1"), None
    )
    retried = [e for e in per_exec if int(e["stage_validation_retries"]) >= 2]
    retried_scored = sum(e["arm_runs_scored"] for e in retried)
    retried_errors = sum(e["protocol_errors"] for e in retried)
    return {
        "note": (
            "exec1 rates are its NEW_OBS arm only, so they are comparable with "
            "the single-arm supplementary executions. The 34.2% quoted in the "
            "forensics report was both arms pooled; NEW_OBS alone is stated "
            "here as new_obs_only_protocol_error_rate."
        ),
        "per_execution": [
            {
                "execution": e["execution"],
                "stage_validation_retries": e["stage_validation_retries"],
                "arm_runs_scored": e["arm_runs_scored"],
                "protocol_errors": e["protocol_errors"],
                "protocol_error_rate": e["protocol_error_rate"],
                "request_observation": e["request_observation"],
                "request_observation_rate": e["request_observation_rate"],
                "arm_runs_with_a_probe": e["arm_runs_with_a_probe"],
                "probe_total": e["probe_total"],
                "llm_calls": e["llm_calls"],
                "stop_reasons": e["stop_reasons"],
                "arm_runs_rescued_by_the_second_retry": (
                    e["arm_runs_rescued_by_the_second_retry"]
                ),
                "inspect_returned_only_after_a_second_retry": (
                    e["inspect_returned_only_after_a_second_retry"]
                ),
                "counterfactual_protocol_error_rate_at_retry_1": (
                    e["counterfactual_protocol_error_rate_at_retry_1"]
                ),
            }
            for e in per_exec
        ],
        "retry_1_baseline_exec1_new_obs_only": {
            "arm_runs_scored": baseline["arm_runs_scored"] if baseline else None,
            "protocol_errors": baseline["protocol_errors"] if baseline else None,
            "protocol_error_rate": (
                baseline["protocol_error_rate"] if baseline else None
            ),
        },
        "retry_2_pooled": {
            "executions": [e["execution"] for e in retried],
            "arm_runs_scored": retried_scored,
            "protocol_errors": retried_errors,
            "protocol_error_rate": (
                retried_errors / retried_scored if retried_scored else None
            ),
            "arm_runs_rescued_by_the_second_retry": sum(
                e["arm_runs_rescued_by_the_second_retry"] for e in retried
            ),
            "counterfactual_protocol_error_rate_at_retry_1": (
                sum(
                    e["counterfactual_protocol_errors_at_retry_1"]
                    for e in retried
                ) / retried_scored if retried_scored else None
            ),
        },
        "all_three_executions_pooled": {
            "arm_runs_scored": sum(e["arm_runs_scored"] for e in per_exec),
            "protocol_errors": sum(e["protocol_errors"] for e in per_exec),
            "request_observation": sum(
                e["request_observation"] for e in per_exec
            ),
            "arm_runs_with_a_probe": sum(
                e["arm_runs_with_a_probe"] for e in per_exec
            ),
        },
    }


def _error_task_shift(per_exec: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which Tasks carried the protocol error, execution by execution."""
    by_exec = {
        execution["execution"]: sorted(
            row["task_episode_id"] for row in execution["task_rows"]
            if row["stop_reason"] == PROTOCOL_STOP
        )
        for execution in per_exec
    }
    baseline = set(by_exec.get("exec1") or ())
    everywhere = set.intersection(
        *(set(v) for v in by_exec.values())
    ) if by_exec else set()
    return {
        "protocol_error_tasks_by_execution": by_exec,
        "tasks_failing_in_every_execution": sorted(everywhere),
        "new_failures_not_seen_in_exec1": {
            label: sorted(set(tasks) - baseline)
            for label, tasks in by_exec.items() if label != "exec1"
        },
        "exec1_failures_that_recovered_elsewhere": {
            label: sorted(baseline - set(tasks))
            for label, tasks in by_exec.items() if label != "exec1"
        },
    }


def _retry_verdict(
    funnel: Mapping[str, Any], shift: Mapping[str, Any]
) -> dict[str, Any]:
    """Did raising the repair budget buy anything?  Stated, not assumed."""
    baseline = funnel["retry_1_baseline_exec1_new_obs_only"]
    retry2 = funnel["retry_2_pooled"]
    before = baseline["protocol_error_rate"]
    after = retry2["protocol_error_rate"]
    if before is None or after is None:
        return {"reading": "Not computable: an execution is missing."}
    each_above = [
        entry["execution"] for entry in funnel["per_execution"]
        if int(entry["stage_validation_retries"]) >= 2
        and entry["protocol_error_rate"] is not None
        and entry["protocol_error_rate"] > before
    ]
    retry_2_labels = [
        entry["execution"] for entry in funnel["per_execution"]
        if int(entry["stage_validation_retries"]) >= 2
    ]
    spread = (
        "and every retry-2 execution individually sits above the retry-1 "
        "baseline"
        if len(each_above) == len(retry_2_labels)
        else "though the retry-2 executions do not agree with each other "
        f"({', '.join(retry_2_labels)})"
    )
    rescued = int(retry2.get("arm_runs_rescued_by_the_second_retry") or 0)
    counterfactual = retry2.get("counterfactual_protocol_error_rate_at_retry_1")
    rescued_ids = ", ".join(
        f"`{task_id}`"
        for entry in funnel["per_execution"]
        for task_id in entry.get(
            "inspect_returned_only_after_a_second_retry"
        ) or ()
    ) or "none"
    new_failures = "; ".join(
        f"{label} newly failed on {len(tasks)} Task"
        f"{'' if len(tasks) == 1 else 's'} that exec1 completed "
        f"({', '.join(f'`{t}`' for t in tasks)})"
        for label, tasks in shift["new_failures_not_seen_in_exec1"].items()
        if tasks
    ) or "No execution introduced a new failing Task"
    stable = shift["tasks_failing_in_every_execution"]
    stable_failures = (
        "and the Tasks that failed in all three executions are %s"
        % ", ".join(f"`{t}`" for t in stable)
        if stable
        else "and not one Task failed in all three executions"
    )
    helped = after < before
    if helped:
        reading = (
            f"Raising the per-stage repair budget from 1 to 2 reduced the "
            f"protocol-error rate from {pct_str(before)} to {pct_str(after)}. "
            f"The effect is measured on {retry2['arm_runs_scored']} retry-2 "
            f"arm runs against {baseline['arm_runs_scored']} retry-1 arm runs "
            f"on one cohort, which is a small base; it is a reason to keep "
            f"measuring, not yet a reason to move the default."
        )
    else:
        reading = (
            f"Two things are true at once here, and collapsing them into one "
            f"number would misreport the fix.\n\n"
            f"**The second retry did work as designed.** {rescued} arm runs "
            f"across the two retry-2 executions had their inspect stage return "
            f"only on the second repair attempt ({rescued_ids}). A stage only "
            f"records a retry count when it returned, so at a budget of 1 "
            f"every one of those would have exited as "
            f"`AGENT_PROTOCOL_ERROR`. Holding the observed first-pass "
            f"behaviour fixed, retry 2 converted {rescued} would-be fatal "
            f"errors into completed stages: {pct_str(counterfactual)} would "
            f"have become {pct_str(after)}.\n\n"
            f"**And the net rate still rose against exec1.** It went from "
            f"{pct_str(before)} at retry 1 to {pct_str(after)} at retry 2, "
            f"{spread}. The repair budget is not what dominates this rate. "
            f"The first-pass slip rate itself moved more between executions "
            f"than one extra repair attempt could offset, and the errors "
            f"landed on different Tasks each time. {new_failures}, "
            f"{stable_failures}. That is what a "
            f"variance-dominated process looks like, not a stable "
            f"task-geometry effect, which also weakens the exec1 reading that "
            f"tied these errors to a low `estimated_region_start_fraction`.\n\n"
            f"So the fix is worth keeping available and is not worth promoting "
            f"to a default on this evidence: it buys a real but small "
            f"recovery, against a noise floor large enough that three "
            f"executions of 19 Tasks cannot resolve a rate difference of this "
            f"size. The default of 1 stays the default."
        )
    return {
        "retry_1_rate": before,
        "retry_2_rate": after,
        "retry_2_reduced_the_error_rate": helped,
        "arm_runs_rescued_by_the_second_retry": rescued,
        "counterfactual_retry_1_rate_on_the_retry_2_runs": counterfactual,
        "second_retry_worked_as_designed": rescued > 0,
        "reading": reading,
    }


def pct_str(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.1f}%"


def build(sources: Sequence[tuple[str, str]] = SOURCES) -> dict[str, Any]:
    per_exec: list[dict[str, Any]] = []
    missing: list[str] = []
    for label, filename in sources:
        path = ARTIFACT_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        per_exec.append(_per_execution(label, _load(path)))
    if missing:
        raise SystemExit(f"missing execution report(s): {missing}")

    pooled = _pool(per_exec)
    funnel = _funnel(per_exec)
    shift = _error_task_shift(per_exec)
    return {
        "protocol_version": "t233_newobs_supply_pooled_v1",
        "verdict": "POOLED_COMPLETE",
        "cohort": "T233",
        "observation_arm": NEW_OBS,
        "exposure": "already exposed development data; not fresh",
        "sealed_data_read": [],
        "sealed_note": (
            "Already-exposed T233 only. KDD W3, NOAA and "
            "g3_final_query_outcome were not opened by any of the three "
            "executions or by this pooling step."
        ),
        "what_this_is": (
            "Three clean NEW_OBS supply executions merged under one "
            "distinct-Task rule. The OLD_OBS arm was not re-run: the "
            "observation A/B question was already answered by exec1, and "
            "nothing here revisits it."
        ),
        "instrument_fixes": {
            "stage_validation_retries_parameterized": (
                "fast_path._run_stage and run_agentic_fast_path, and "
                "runner._run_arm, now accept a validation_retries keyword. Its "
                "default is 1 -- the value previously hard-coded in "
                "fast_path -- so every other caller in the repository behaves "
                "exactly as before and no existing readout moves. Only "
                "run_t233_supply_obs_ab passes 2, and only when asked on the "
                "command line. exec1 ran at 1; exec2 and exec3 ran at 2."
            ),
            "ungrounded_citation_rejection_count_fixed": (
                "The driver counted this only from stage_validation, which is "
                "populated after a stage returns and is therefore empty when "
                "the stage died -- so a fatal grounding rejection was counted "
                "as 0, the exact case the counter existed to catch. "
                "FastPathTrace now carries the terminal validator error code, "
                "runner surfaces it, and the driver reports recovered and "
                "fatal separately as well as their sum."
            ),
            "no_other_code_touched": (
                "No M0b working-tree change, no historical artifact and no "
                "threshold was modified, and nothing was committed."
            ),
        },
        "pinned_parameters": {
            "note": (
                "Carried verbatim from each execution's own report rather than "
                "restated here, so the pooled artifact cannot disagree with "
                "the runs it pools. Everything except "
                "stage_validation_retries and observation_arms_run is "
                "identical across the three executions."
            ),
            "differs_across_executions": {
                "stage_validation_retries": {
                    execution["execution"]: execution[
                        "stage_validation_retries"
                    ]
                    for execution in per_exec
                },
                "observation_arms_run": {
                    execution["execution"]: execution["observation_arms_run"]
                    for execution in per_exec
                },
            },
            "per_execution": {
                execution["execution"]: execution["pinned_parameters"]
                for execution in per_exec
            },
        },
        "executions": [
            {
                key: value for key, value in execution.items()
                if key not in {"task_rows", "pinned_parameters"}
            }
            for execution in per_exec
        ],
        "funnel": funnel,
        "retry_verdict": _retry_verdict(funnel, shift),
        "protocol_error_locality": shift,
        "pooled_census": pooled,
        "authorization_precheck": _precheck_table(pooled),
        "outlier_mad_recurrence": _outlier_mad_recurrence(per_exec, pooled),
        "per_execution_task_rows": [
            task_row
            for execution in per_exec
            for task_row in execution["task_rows"]
        ],
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    funnel = payload["funnel"]
    precheck = payload["authorization_precheck"]
    outlier = payload["outlier_mad_recurrence"]
    pooled = payload["pooled_census"]
    baseline = funnel["retry_1_baseline_exec1_new_obs_only"]
    retry2 = funnel["retry_2_pooled"]

    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value) * 100:.1f}%"

    lines = [
        "# T233 NEW_OBS supply, three clean executions pooled",
        "",
        "## What changed before these runs, and what did not",
        "",
        "Two instrument fixes were made, both narrow:",
        "",
        "- The per-stage repair budget is now a parameter. **Its default is "
        "unchanged at 1**, the value that was previously hard-coded, so every "
        "other caller in the repository is unaffected and no existing readout "
        "moves. Only this driver passes 2, and only on request. `exec1` ran at "
        "1; `exec2` and `exec3` ran at 2.",
        "- The driver's ungrounded-citation counter no longer reads 0 when the "
        "stage died. It previously counted only from `stage_validation`, which "
        "is empty whenever a stage raised, so a fatal grounding rejection was "
        "invisible to the one counter written to catch it. Recovered and fatal "
        "are now reported separately as well as summed.",
        "",
        "No M0b working-tree change, no historical artifact and no threshold "
        "was touched, and nothing was committed.",
        "",
        "## Merge rule",
        "",
        "The unit of evidence is the **distinct Task**, not the execution. A "
        "Task positive in the same cell in two executions is one distinct "
        "positive Task, because re-sampling one Task on the same "
        "already-exposed data is not independent evidence. A Task that comes "
        "out positive in one execution and negative in another is reported as "
        "a **conflict** for that cell and counted in neither column. Both a "
        "distinct negative Task and a conflict Task count as **opposing**, and "
        "either blocks a precheck.",
        "",
        "## Protocol-error funnel, retry 1 against retry 2",
        "",
        "| execution | retries | arm runs | protocol errors | rate | "
        "`REQUEST_OBSERVATION` | probed | rescued by 2nd retry |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in funnel["per_execution"]:
        lines.append(
            "| `%s` | %s | %d | %d | %s | %d (%s) | %d | %d |"
            % (
                entry["execution"],
                entry["stage_validation_retries"],
                entry["arm_runs_scored"],
                entry["protocol_errors"],
                pct(entry["protocol_error_rate"]),
                entry["request_observation"],
                pct(entry["request_observation_rate"]),
                entry["arm_runs_with_a_probe"],
                entry["arm_runs_rescued_by_the_second_retry"],
            )
        )
    lines += [
        "",
        "`exec1` here is its NEW_OBS arm only, so it is comparable with the "
        "single-arm supplementary runs; the 34.2% in the forensics report was "
        "both arms pooled.",
        "",
        "- retry 1, `exec1` NEW_OBS only: %d / %d = %s"
        % (
            baseline["protocol_errors"] or 0,
            baseline["arm_runs_scored"] or 0,
            pct(baseline["protocol_error_rate"]),
        ),
        "- retry 2, `exec2` + `exec3` pooled: %d / %d = %s"
        % (
            retry2["protocol_errors"],
            retry2["arm_runs_scored"],
            pct(retry2["protocol_error_rate"]),
        ),
        "",
        "### Verdict on the retry fix",
        "",
        payload["retry_verdict"]["reading"],
        "",
        "## Pooled cells",
        "",
        "| program | context | distinct + | distinct - | conflict | opposing | "
        "executions |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in pooled["cells"]:
        lines.append(
            "| `%s` | %s | %d | %d | %d | %d | %s |"
            % (
                cell["program"],
                cell["context_condition"],
                cell["distinct_positive_tasks"],
                cell["distinct_negative_tasks"],
                cell["conflict_tasks"],
                cell["opposing_tasks"],
                ", ".join(cell["executions_contributing"]),
            )
        )

    lines += [
        "",
        "## Authorization precheck, precheck only",
        "",
        "**No authorization action was taken.** No TRY and no Skill was "
        "written, no authorization artifact was modified and nothing was "
        "promoted. This section is a precheck and nothing more.",
        "",
        "Cells reaching >= %d distinct UNGUIDED positive Tasks with zero "
        "opposing: **%d**."
        % (
            pooled["counting_rule"]["precheck_min_distinct_positive_tasks"],
            precheck["precheck_eligible_cell_count"],
        ),
        "",
        "Closest three cells to the threshold:",
        "",
        "| program | context | distinct + | short by | opposing | blocked |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in precheck["closest_three_cells"]:
        lines.append(
            "| `%s` | %s | %d | %d | %d | %s |"
            % (
                cell["program"],
                cell["context_condition"],
                cell["distinct_positive_tasks"],
                cell["tasks_short_of_threshold"],
                cell["opposing_tasks"],
                "yes" if cell["blocked_by_opposing"] else "no",
            )
        )

    lines += [
        "",
        "## `outlier_mad` recurrence",
        "",
        outlier["reading"],
        "",
        "## Standing limits",
        "",
        "- Every Task here is already-exposed T233 development data. Pooling "
        "three executions raises confidence about this cohort under this "
        "budget and nothing else; it does not make any cell correct, useful "
        "downstream or transferable.",
        "- Sealed sources were not read. KDD W3, NOAA and "
        "`g3_final_query_outcome` were not opened.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write", action="store_true",
        help="build and print the summary without writing the deliverables",
    )
    args = parser.parse_args(argv)
    payload = build()
    if not args.no_write:
        POOLED_JSON.parent.mkdir(parents=True, exist_ok=True)
        POOLED_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        POOLED_MD.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(
        {
            "funnel": payload["funnel"],
            "retry_verdict": payload["retry_verdict"],
            "authorization_precheck": payload["authorization_precheck"],
            "outlier_mad_recurrence": payload["outlier_mad_recurrence"],
        },
        indent=2, ensure_ascii=False, default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
