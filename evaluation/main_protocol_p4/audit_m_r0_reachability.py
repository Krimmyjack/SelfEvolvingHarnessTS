"""M-R0: why HEC-1 never entered "revise an already-Active Skill", and what it cost.

Read-only.  0 LLM calls, 0 Consumer fits, 0 held-out reads, 0 sealed reads, no
threshold and no existing artifact is touched.  Everything here is recomputed
from three frozen course artifacts plus the K0 receipt, and every count that is
*derived* rather than *recorded* says so in the output.

Run:  python -m evaluation.main_protocol_p4.audit_m_r0_reachability

What it establishes, and how it proves it
-----------------------------------------
The census that feeds ``ADD`` and ``NARROW`` reads a bank row's relation from
``row["relation"] or row["admission"]`` (``outer_loop.py:209``).  The runner
fills those two fields from ``probe["admission"]["relation"]`` and
``probe["admission"]["reason"]`` (``run_hec1.py:1849-1850``), but
``AdmissionVerdict`` (``methods/ttha/admission_policy.py:69-82``) has **no**
``relation`` field -- so the first is always ``None`` and the second is a
*reason* string from a different vocabulary than the ``POSITIVE / CONFLICT /
NEGATIVE`` the census counts (``outer_loop.py:363-365``).

The bank is rebuilt here from the recorded probe rows and the census's own
grouping is replayed twice: once with the as-run vocabulary (which must
reproduce the artifacts' ``relation_counts`` exactly, and does -- that is the
validation) and once with the relation left absent so ``_relation`` falls
through to its own gain-based definition.  The difference between the two is
the reachability answer.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import outer_loop

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts" / "main_protocol"
ORDERINGS = ("forward", "reverse", "interleaved")
COURSE = {name: ART / ("hec1_course_v11p0_%s_live.json" % name)
          for name in ORDERINGS}
K0_RECEIPT = ART / "hec1_k0_phase_s_v11_live.json"
RUN_COMMIT = "d690850be1cb70d1da5f7034dbb097f42ba42e36"

#: The runner files whose behaviour this audit explains.  Their bytes at HEAD
#: must equal their bytes at the run commit, or no historical statement made
#: from the current checkout is admissible.
VERSIONED_FILES = (
    "evaluation/main_protocol_p4/run_hec1.py",
    "evaluation/main_protocol_p4/outer_loop.py",
    "evaluation/main_protocol_p4/restricted_draft.py",
    "evaluation/main_protocol_p4/hec1_contract.py",
    "evaluation/main_protocol_p4/scope_threshold_tool.py",
    "evaluation/main_protocol_p4/scoped_serving_evaluator.py",
    "methods/ttha/online_loop.py",
    "methods/ttha/admission_policy.py",
    "methods/ttha/scope_executor.py",
)

MATERIAL = 0.005          # hec1_contract.RISK["material"]
MIN_TREATED = 5           # hec1_contract.RISK["min_treated"]
PER_CELL_LLM_CAP = 5      # hec1_contract.PER_UNIT_ARM_BUDGET["llm_calls"]:794
OUTER_LLM_PER_STEP = 2    # hec1_contract.py:803
PERIOD_K_UNITS = 5        # hec1_contract.OUTER_LOOP["period_k_units"]:736
TASK_CONSUMER_KEY = "forecast|pooled-ridge-a1|sMASE"
ONLINE_ARMS = ("A5-online", "A3-online")
LLM_ARMS = ("A5-frozen", "A5-online", "A3-online")


# --------------------------------------------------------------------------
# version discipline
# --------------------------------------------------------------------------

def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), cwd=str(ROOT), capture_output=True,
                             text=True, check=False)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def version_check() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    rows = []
    for path in VERSIONED_FILES:
        diff = _git("diff", RUN_COMMIT, "--", path)
        worktree = _git("status", "--porcelain", "--", path)
        rows.append({
            "path": path,
            "identical_to_run_commit": (diff == "") if diff is not None else None,
            "worktree_clean": (worktree == "") if worktree is not None else None,
        })
    return {
        "run_commit_claimed_by_artifacts": RUN_COMMIT,
        "head_commit": head,
        "files": rows,
        "all_identical": all(row["identical_to_run_commit"] for row in rows),
        "why": (
            "historical behaviour may only be explained with the code that "
            "produced it; where HEAD differs the current file is not used"
        ),
    }


# --------------------------------------------------------------------------
# loading and small helpers
# --------------------------------------------------------------------------

def load(name: str) -> dict[str, Any]:
    return json.loads(COURSE[name].read_text(encoding="utf-8"))


def prog(steps: Any) -> str:
    return ">".join(str(step["op"]) for step in (steps or ()))


def unit_id(unit: dict[str, Any]) -> str:
    return "%s@%d" % (unit["block"], int(unit["origin"]))


def cells(doc: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    return sorted((cell for cell in doc["cells"] if cell["arm"] == arm),
                  key=lambda cell: cell["position"])


# --------------------------------------------------------------------------
# 1. the bank, rebuilt exactly as ``_bank_rows_from_round`` builds it
# --------------------------------------------------------------------------

def bank_rows_for_cell(cell: dict[str, Any]) -> list[dict[str, Any]]:
    """run_hec1.py:1830-1855, minus the fields the census never reads.

    ``relation`` and ``admission`` are copied with the same two lookups the
    runner uses, which is the whole point: the first key does not exist on an
    ``AdmissionVerdict`` and so is always ``None``.
    """
    rows = []
    for probe in cell.get("probes") or ():
        if probe.get("kind") != "probe":
            continue
        gains = list(probe.get("per_series_gain") or ())
        if len(gains) != int(cell["served"]):
            continue
        admission = probe.get("admission") or {}
        rows.append({
            "unit": cell["unit"],
            "program_steps": probe.get("program_steps"),
            "serving_scope": probe.get("serving_scope"),
            "relation": admission.get("relation"),
            "admission": admission.get("reason"),
            "gains": [float(value) for value in gains],
        })
    return rows


def relation_as_run(row: dict[str, Any]) -> str:
    """outer_loop._relation's first branch, verbatim (outer_loop.py:209-211)."""
    return str(row.get("relation") or row.get("admission") or "").upper()


def relation_from_gains(row: dict[str, Any]) -> str:
    """outer_loop._relation's fallback (outer_loop.py:212-220).

    Unreachable in the run, because the branch above always returns a non-empty
    string.  Replayed here to show what the same census would have counted with
    the module's own definition of the three relations.
    """
    gains = row["gains"]
    if not gains:
        return "UNKNOWN"
    aggregate = sum(gains) / len(gains)
    harmed = any(value < -MATERIAL for value in gains)
    if aggregate >= MATERIAL:
        return "CONFLICT" if harmed else "POSITIVE"
    return "NEGATIVE"


def census_replay(rows: list[dict[str, Any]], relation_fn) -> dict[str, dict]:
    """The census's grouping and counting, for one bank.

    Alias merging is deliberately omitted: every recorded step reports
    ``aliases: []``, so no merge happened in the run, and reproducing the merge
    machinery would add a code path this audit cannot validate against the
    artifact.  ``aliases_recorded_empty`` carries the premise into the output.
    """
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = outer_loop.census_key(TASK_CONSUMER_KEY, row["program_steps"],
                                    row["serving_scope"])
        bucket = groups.setdefault(key, {"census_key": key, "relations": []})
        bucket["relations"].append(relation_fn(row))
    for bucket in groups.values():
        counts = Counter(bucket["relations"])
        bucket["relation_counts"] = dict(sorted(counts.items()))
        bucket["unit_count"] = len(bucket["relations"])
        bucket["positive_units"] = counts.get("POSITIVE", 0)
        bucket["adverse_units"] = (counts.get("CONFLICT", 0)
                                   + counts.get("NEGATIVE", 0))
        bucket.pop("relations")
    return groups


# --------------------------------------------------------------------------
# 2. reachability per outer step
# --------------------------------------------------------------------------

def reachability(doc: dict[str, Any], ordering: str,
                 k0_lineage_keys: list[str]) -> list[dict[str, Any]]:
    """For every recorded outer step: replay the census, and say what blocked."""
    out = []
    for arm in ONLINE_ARMS:
        arm_cells = cells(doc, arm)
        recorded_steps = [step for step in doc["outer_steps"]
                          if step["arm"] == arm]
        held: set[str] = set(k0_lineage_keys) if arm.startswith("A5") else set()
        held_at: dict[int, set[str]] = {}
        bank: list[dict[str, Any]] = []
        bank_at: dict[int, list[dict[str, Any]]] = {}
        for cell in arm_cells:
            position = int(cell["position"])
            if cell.get("lineage_key"):
                held.add(str(cell["lineage_key"]))
            bank.extend(bank_rows_for_cell(cell))
            if (position + 1) % PERIOD_K_UNITS == 0:
                k = (position + 1) // PERIOD_K_UNITS
                held_at[k] = set(held)
                bank_at[k] = list(bank)
        for step in recorded_steps:
            k = int(step["k_index"])
            rows = bank_at.get(k, [])
            as_run = census_replay(rows, relation_as_run)
            counterfactual = census_replay(rows, relation_from_gains)
            recorded_groups = {group["census_key"]: group
                               for group in step["groups"]}
            reproduces = (
                set(as_run) == set(recorded_groups)
                and all(as_run[key]["relation_counts"] == group["relation_counts"]
                        and as_run[key]["unit_count"] == group["unit_count"]
                        and as_run[key]["positive_units"] == group["positive_units"]
                        and as_run[key]["adverse_units"] == group["adverse_units"]
                        for key, group in recorded_groups.items()))
            group_rows = []
            for key, group in sorted(counterfactual.items()):
                held_here = key in held_at.get(k, set())
                group_rows.append({
                    "census_key": key,
                    "key_in_held": held_here,
                    "as_run_relation_counts": as_run[key]["relation_counts"],
                    "as_run_positive_units": as_run[key]["positive_units"],
                    "as_run_adverse_units": as_run[key]["adverse_units"],
                    "counterfactual_relation_counts": group["relation_counts"],
                    "counterfactual_positive_units": group["positive_units"],
                    "counterfactual_adverse_units": group["adverse_units"],
                    "ADD_as_run": (as_run[key]["positive_units"] >= 1
                                   and not held_here),
                    "ADD_counterfactual": (group["positive_units"] >= 1
                                           and not held_here),
                    "NARROW_as_run": (as_run[key]["adverse_units"] >= 2
                                      and held_here),
                    "NARROW_counterfactual": (group["adverse_units"] >= 2
                                              and held_here),
                })
            out.append({
                "ordering": ordering,
                "arm": arm,
                "k_index": k,
                "units_in_bank_recorded": step["units_in_bank"],
                "bank_rows_rebuilt": len(rows),
                "recorded_empty_reason": step["empty_reason"],
                "recorded_candidate_kinds": [candidate["kind"]
                                             for candidate in step["candidates"]],
                "recorded_slow_calls": step["slow_calls"],
                "recorded_replay_fits": step["replay_fits"],
                "recorded_drift_signals": len(step["drift_signals"]),
                "recorded_revocation_recommendations": len(
                    step["revocation_recommendations"]),
                "aliases_recorded_empty": all(not group.get("aliases")
                                              for group in step["groups"]),
                "census_replay_reproduces_artifact": reproduces,
                "held_lineage_keys": sorted(held_at.get(k, set())),
                "groups": group_rows,
            })
    return out


# --------------------------------------------------------------------------
# 3. Active lineages, deployment rights, drafts, situations
# --------------------------------------------------------------------------

def lineages(doc: dict[str, Any], ordering: str) -> list[dict[str, Any]]:
    out = []
    for arm in LLM_ARMS:
        arm_cells = cells(doc, arm)
        if not arm_cells:
            continue
        start = arm_cells[0].get("active_program_signatures_at_start") or {}
        known: dict[str, dict[str, Any]] = {}
        for signature, skill_id in start.items():
            known[signature] = {
                "program_signature": signature, "skill_id": skill_id,
                "card_origin": "K0 (seeded at course start)",
                "activated_at_position": None, "readings": []}
        for cell in arm_cells:
            for signature, skill_id in (
                    cell.get("active_program_signatures_at_start") or {}).items():
                known.setdefault(signature, {
                    "program_signature": signature, "skill_id": skill_id,
                    "card_origin": "activated during the course",
                    "activated_at_position": None, "readings": []})
            if cell.get("activated") and cell.get("deployed"):
                signature = "%s({})" % prog(cell["deployed"])
                entry = known.setdefault(signature, {
                    "program_signature": signature,
                    "skill_id": (cell.get("skills_minted_this_unit")
                                 or [None])[0],
                    "card_origin": "activated during the course",
                    "activated_at_position": None, "readings": []})
                if entry["activated_at_position"] is None:
                    entry["activated_at_position"] = int(cell["position"])
                    entry["activated_at_unit"] = unit_id(cell["unit"])
                    entry["lineage_key"] = cell.get("lineage_key")
                    entry["skills_minted_this_unit"] = (
                        cell.get("skills_minted_this_unit") or [])
            deployed, delayed = cell.get("deployed"), cell.get("delayed")
            if not deployed or not delayed:
                continue
            signature = "%s({})" % prog(deployed)
            if signature not in known:
                continue
            held_at_start = signature in (
                cell.get("active_program_signatures_at_start") or {})
            gate = delayed["gate"]
            known[signature]["readings"].append({
                "position": int(cell["position"]),
                "unit": unit_id(cell["unit"]),
                "delayed_origin": delayed["origin"],
                "card_held_deployment_rights_at_cell_start": held_at_start,
                "deployed_via": cell.get("deployed_via"),
                "treated": delayed["treated"],
                "passes": gate["passes"],
                "failed_lines": gate["failed_lines"],
                "class": ("PASS" if gate["passes"]
                          else "COVERAGE_FLOOR_ONLY"
                          if gate["failed_lines"] == ["coverage_floor"]
                          else "RISK_LINE_ADVERSE"),
                "restricted_state_written_by_this_cell": cell.get(
                    "restricted_state"),
            })
        for entry in known.values():
            reads = entry["readings"]
            held = [row for row in reads
                    if row["card_held_deployment_rights_at_cell_start"]]
            entry["ordering"] = ordering
            entry["arm"] = arm
            entry["redeployments_while_holding_rights"] = len(held)
            entry["adverse_while_active_risk_lines"] = sum(
                1 for row in held if row["class"] == "RISK_LINE_ADVERSE")
            entry["adverse_while_active_coverage_floor_only"] = sum(
                1 for row in held if row["class"] == "COVERAGE_FLOOR_ONLY")
            entry["passes_while_active"] = sum(
                1 for row in held if row["class"] == "PASS")
            entry["reached_MIN_ADVERSE_UNITS_FOR_NARROWING_on_delayed_faces"] = (
                entry["adverse_while_active_risk_lines"] >= 2)
            entry["but_delayed_faces_never_enter_the_bank"] = (
                "_bank_rows_from_round keeps probe rows only "
                "(run_hec1.py:1836); the delayed gate is an authority, not "
                "evidence, so no adverse delayed reading is ever visible to "
                "the census that decides NARROW")
            out.append(entry)
    return out


def deployment_rights(doc: dict[str, Any], ordering: str) -> list[dict[str, Any]]:
    rows = []
    for arm in LLM_ARMS:
        counts = [(int(cell["position"]),
                   len(cell.get("snapshot_skill_ids_at_start") or ()))
                  for cell in cells(doc, arm)
                  if "snapshot_skill_ids_at_start" in cell]
        drops = [{"from_position": counts[index - 1][0],
                  "to_position": counts[index][0],
                  "from": counts[index - 1][1], "to": counts[index][1]}
                 for index in range(1, len(counts))
                 if counts[index][1] < counts[index - 1][1]]
        rows.append({
            "ordering": ordering, "arm": arm,
            "snapshot_skill_counts_by_position": [n for _, n in counts],
            "decreases": drops,
            "any_card_lost_deployment_rights": bool(drops),
            "revocation_recommendations_recorded": sum(
                len(step["revocation_recommendations"])
                for step in doc["outer_steps"] if step["arm"] == arm),
            "drafts_dropped_by_the_per_unit_rebuild": sum(
                int((cell.get("reset") or {}).get("dropped_drafts") or 0)
                for cell in cells(doc, arm)),
        })
    return rows


def draft_reconstruction(doc: dict[str, Any],
                         ordering: str) -> list[dict[str, Any]]:
    """Every Draft, and which cell produced each of its verification faces."""
    by_window: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in doc["cells"]:
        delayed = cell.get("delayed")
        if delayed and cell.get("deployed"):
            by_window[(cell["arm"], int(delayed["origin"]))].append(cell)
    out = []
    for arm, ledger in doc["lifecycle"].items():
        for draft in ledger["drafts"]:
            events = []
            for entry in draft["history"]:
                window = int(entry["window"])
                pool = by_window[(arm, window)]
                matches = [cell for cell in pool
                           if (cell.get("delayed") or {}).get("gate", {}).get(
                               "failed_lines") == entry["failed_lines"]]
                cell = matches[0] if len(matches) == 1 else None
                events.append({
                    "event": entry["event"],
                    "window": window,
                    "failed_lines": entry["failed_lines"],
                    "classified_state": entry["classified_state"],
                    "consumed_attempt": entry["consumed_attempt"],
                    "verification_attempts_after": entry["verification_attempts"],
                    "state_after": entry["state_after"],
                    "cell_position": (int(cell["position"]) if cell else None),
                    "cell_unit": (unit_id(cell["unit"]) if cell else None),
                    "cell_program": (prog(cell["deployed"]) if cell else None),
                    "cell_deployed_via": (cell.get("deployed_via")
                                          if cell else None),
                    "program_matches_draft": (
                        None if cell is None else
                        prog(cell["deployed"]) == prog(draft["program_steps"])),
                    "card_held_deployment_rights_at_cell_start": (
                        None if cell is None else
                        ("%s({})" % prog(cell["deployed"]))
                        in (cell.get("active_program_signatures_at_start") or {})),
                    "cell_match": ("unique" if cell is not None
                                   else "AMBIGUOUS_%d_candidates" % len(pool)),
                })
            first = events[0] if events else {}
            out.append({
                "ordering": ordering, "arm": arm,
                "draft_id": draft["draft_id"],
                "program_at_creation": prog(draft["program_steps"]),
                "root_scope": draft["root_scope"],
                "current_scope": draft["current_scope"],
                "scope_ever_revised": draft["root_scope"] != draft["current_scope"],
                "created_at_origin": draft["created_at_origin"],
                "created_by_cell_position": first.get("cell_position"),
                "created_by_cell_unit": first.get("cell_unit"),
                "created_from_a_card_that_held_deployment_rights": first.get(
                    "card_held_deployment_rights_at_cell_start"),
                "state": draft["state"],
                "revisions": draft["revisions"],
                "verification_attempts": draft["verification_attempts"],
                "closed": draft["closed"],
                "census_key": draft["census_key"],
                "census_key_is_none": draft["census_key"] is None,
                "events": events,
                "cross_program_events": [
                    event for event in events
                    if event["program_matches_draft"] is False],
            })
    return out


def situations(docs: dict[str, dict]) -> dict[str, Any]:
    """Every delayed-gate failure, keyed by its full identity."""
    table: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for ordering, doc in docs.items():
        for cell in doc["cells"]:
            delayed, deployed = cell.get("delayed"), cell.get("deployed")
            if not delayed or not deployed or delayed["gate"]["passes"]:
                continue
            scope = cell.get("deployed_serving_scope") or {}
            key = (cell["unit"]["block"], int(cell["unit"]["origin"]),
                   int(delayed["origin"]), prog(deployed),
                   json.dumps(scope, sort_keys=True))
            table[key].append({
                "ordering": ordering, "arm": cell["arm"],
                "position": int(cell["position"]),
                "failed_lines": delayed["gate"]["failed_lines"],
                "treated": delayed["treated"], "served": delayed["served"],
                "aggregate_gain": delayed["aggregate_gain"],
                "harmed_fraction": delayed["harmed_fraction"],
                "max_single_series_harm": delayed["max_single_series_harm"],
                "deployed_via": cell.get("deployed_via"),
                "card_held_deployment_rights_at_cell_start": (
                    ("%s({})" % prog(deployed))
                    in (cell.get("active_program_signatures_at_start") or {})),
                "restricted_state_written_by_this_cell": cell.get(
                    "restricted_state"),
            })
    # how many later evaluable units each ordering still has after a position
    later_evaluable: dict[str, dict[int, int]] = {}
    for ordering, doc in docs.items():
        scoreable = sorted({int(cell["position"]) for cell in doc["cells"]
                            if cell.get("evaluation")})
        later_evaluable[ordering] = {
            position: sum(1 for other in scoreable if other > position)
            for position in range(26)}

    rows = []
    for key, instances in sorted(table.items()):
        block, origin, delayed_origin, program, scope = key
        line_sets = {tuple(row["failed_lines"]) for row in instances}
        coverage_only = line_sets == {("coverage_floor",)}
        readings = {(row["treated"], row["aggregate_gain"],
                     row["harmed_fraction"], row["max_single_series_harm"])
                    for row in instances}
        per_ordering = {}
        for row in instances:
            best = per_ordering.get(row["ordering"])
            n_later = later_evaluable[row["ordering"]][row["position"]]
            if best is None or n_later > best["later_evaluable_units"]:
                per_ordering[row["ordering"]] = {
                    "first_position": row["position"],
                    "later_evaluable_units": n_later,
                    "arms": sorted({other["arm"] for other in instances
                                    if other["ordering"] == row["ordering"]}),
                }
        rows.append({
            "situation_id": "%s|o%d|d%d|%s" % (block, origin, delayed_origin,
                                               program),
            "cohort_block": block, "origin": origin,
            "delayed_origin": delayed_origin, "program": program,
            "serving_scope": json.loads(scope),
            "failed_line_sets": sorted("+".join(item) for item in line_sets),
            "treated_on_the_delayed_face": sorted(
                {row["treated"] for row in instances}),
            "min_treated_line": MIN_TREATED,
            "coverage_floor_only": coverage_only,
            "a_workflow_edit_can_move_this_failure": not coverage_only,
            "why": (
                "the coverage floor is decided by resolving the predicate "
                "against the raw window, which no Workflow edit changes while "
                "the Scope is held fixed; every candidate would fail the same "
                "line, so this situation is RULE_FOUND_NO_FIX by construction"
                if coverage_only else
                "the failing lines are risk lines, which a different program "
                "on the same resolved series can in principle move"),
            "reading_is_identical_across_all_instances": len(readings) == 1,
            "instance_count": len(instances),
            "orderings": sorted({row["ordering"] for row in instances}),
            "arms": sorted({row["arm"] for row in instances}),
            "per_ordering": per_ordering,
            "tier": ("TIER_2_ACTIVE_CARD"
                     if any(row["card_held_deployment_rights_at_cell_start"]
                            for row in instances)
                     else "TIER_1_NO_ACTIVE_CARD"),
            "instances": instances,
        })

    unique = len(rows)
    instances_total = sum(row["instance_count"] for row in rows)
    repairable = [row for row in rows
                  if row["a_workflow_edit_can_move_this_failure"]]
    return {
        "identity_fields": ["cohort_block", "origin", "delayed_origin",
                            "program", "serving_scope"],
        "unique_situations": unique,
        "instances_total": instances_total,
        "repeated_observations": instances_total - unique,
        "repeated_observations_note": (
            "every instance of one situation carries a byte-identical delayed "
            "reading; the repetition is the same deterministic evaluation "
            "recomputed per arm and per ordering, not new evidence"),
        "situations_where_a_workflow_edit_can_move_the_failure": len(repairable),
        "situation_by_ordering_pairs_total": sum(
            len(row["per_ordering"]) for row in rows),
        "situation_by_ordering_pairs_repairable": sum(
            len(row["per_ordering"]) for row in repairable),
        "situation_by_ordering_pairs_repairable_with_2plus_later_units": sum(
            1 for row in repairable for entry in row["per_ordering"].values()
            if entry["later_evaluable_units"] >= 2),
        "situations": rows,
    }


# --------------------------------------------------------------------------
# 4. billing
# --------------------------------------------------------------------------

EXHAUSTED_MARK = "AgentCallBudgetExceeded: Agent call budget exhausted at 5"


def billing(docs: dict[str, dict]) -> dict[str, Any]:
    per_ordering, totals = [], Counter()
    for ordering, doc in docs.items():
        recorded, exhausted, cells_ge1 = 0, 0, 0
        for cell in doc["cells"]:
            if cell["arm"] == "Static":
                continue
            if "llm_calls_this_cell" in cell:
                count = int(cell["llm_calls_this_cell"])
                recorded += count
                cells_ge1 += 1 if count >= 1 else 0
            elif any(EXHAUSTED_MARK in str(fault.get("why"))
                     for fault in (cell.get("faults") or ())):
                exhausted += 1
        outer_physical = sum(int(step["slow_calls"])
                             for step in doc["outer_steps"])
        ledger = doc["ledgers"]
        row = {
            "ordering": ordering,
            "billed_llm_fast": ledger["llm_fast"],
            "billed_llm_outer": ledger["llm_outer"],
            "billed_llm_total": ledger["llm_total"],
            "physical_fast_recorded_on_cells": recorded,
            "cells_with_at_least_one_recorded_call": cells_ge1,
            "under_bill_from_spent_minus_one": recorded - ledger["llm_fast"],
            "cells_that_hit_the_per_cell_cap": exhausted,
            "physical_calls_in_capped_cells_DERIVED": exhausted * PER_CELL_LLM_CAP,
            "physical_outer": outer_physical,
            "physical_total_DERIVED": (recorded + exhausted * PER_CELL_LLM_CAP
                                       + outer_physical),
            "ordering_cap": doc["budget_guard"]["ordering_cap"],
            "blocked_before_backend": doc["budget_guard"][
                "blocked_before_backend"],
            "scripted_calls_not_billed": doc["budget_guard"][
                "scripted_calls_not_billed"],
        }
        row["difference"] = row["physical_total_DERIVED"] - row["billed_llm_total"]
        row["physical_total_within_ordering_cap"] = (
            row["physical_total_DERIVED"] <= row["ordering_cap"])
        per_ordering.append(row)
        for key in ("billed_llm_total", "physical_total_DERIVED", "difference",
                    "cells_with_at_least_one_recorded_call",
                    "cells_that_hit_the_per_cell_cap",
                    "physical_calls_in_capped_cells_DERIVED", "physical_outer",
                    "under_bill_from_spent_minus_one"):
            totals[key] += row[key]
    return {
        "categories_distinguished": [
            "attempted (reserve)", "physically sent", "returned successfully",
            "refused before the backend", "retried", "billed to the ledger",
        ],
        "per_ordering": per_ordering,
        "totals": dict(totals),
        "decomposition_of_the_difference": {
            "spend_calls_equals_spent_minus_one": {
                "code": "run_hec1.py:1970",
                "calls": totals["under_bill_from_spent_minus_one"],
                "equals_cells_with_at_least_one_call": (
                    totals["under_bill_from_spent_minus_one"]
                    == totals["cells_with_at_least_one_recorded_call"]),
                "why": (
                    "BudgetGuard.reserve (run_hec1.py:205-217) checks the caps "
                    "and increments no counter, so the comment's 'the reserve "
                    "took one' is not true of any ledger; subtracting it drops "
                    "exactly one billed call from every cell that made at "
                    "least one"),
            },
            "cells_that_hit_the_per_cell_cap": {
                "code": ("runtime/agent_backend.py:450-455 raises on the "
                         "request after the 5th; run_hec1.py:1954-1966 catches "
                         "it and returns before guard.spend at :1970"),
                "cells": totals["cells_that_hit_the_per_cell_cap"],
                "calls_DERIVED": totals["physical_calls_in_capped_cells_DERIVED"],
                "derivation": (
                    "BudgetedAgentBackend raises when calls >= maximum_calls, "
                    "so exactly 5 requests had been sent; the per-cell counter "
                    "is not persisted for these cells, so 5 is read off the "
                    "cap the backend refused at rather than measured"),
            },
            "refused_before_the_backend": {
                "recorded": "budget_guard.blocked_before_backend",
                "count": sum(len(row["blocked_before_backend"])
                             for row in per_ordering),
                "why": "no ordering came near the 500-call ordering cap",
            },
            "retries": {
                "inner": ("schema-correction retries inside a cell are ordinary "
                          "backend calls and are inside llm_calls_this_cell"),
                "outer": ("_clause_for retries by calling Slow again; each "
                          "retry is one physical request and is billed through "
                          "_MeteredOuterBackend (run_hec1.py:279-298), which "
                          "is why llm_outer is exact"),
            },
        },
        "cross_check_against_closeout_doc": {
            "doc": "docs/HEC1_CLOSEOUT_DESIGN_RULING_2026-09-04.md:43",
            "claimed_physical": 1088, "claimed_billed": 666,
            "claimed_difference": 422,
            "recomputed_physical": totals["physical_total_DERIVED"],
            "recomputed_billed": totals["billed_llm_total"],
            "recomputed_difference": totals["difference"],
            "agrees": (totals["physical_total_DERIVED"] == 1088
                       and totals["billed_llm_total"] == 666
                       and totals["difference"] == 422),
        },
    }


def fits(docs: dict[str, dict]) -> dict[str, Any]:
    rows = []
    for ordering, doc in docs.items():
        probes, rejected, units_with_probes = 0, 0, set()
        for cell in doc["cells"]:
            kinds = [probe.get("kind") for probe in (cell.get("probes") or ())]
            count = sum(1 for kind in kinds if kind == "probe")
            rejected += sum(1 for kind in kinds if kind == "verifier_rejected")
            probes += count
            if count:
                units_with_probes.add(int(cell["position"]))
        ledger = doc["ledgers"]
        rows.append({
            "ordering": ordering,
            "recorded_course_fits_delayed_plus_evaluation": ledger["course_fits"],
            "recorded_replay_fits": ledger["replay_fits"],
            "recorded_shadow_fits": ledger["shadow_fits"],
            "recorded_baseline_fits": ledger["baseline_fits"],
            "support_probes_with_a_receipt": probes,
            "support_probes_rejected_by_the_window_verifier": rejected,
            "units_with_at_least_one_probe": len(units_with_probes),
            "support_probe_fits_DERIVED": 2 * probes + len(units_with_probes),
        })
    return {
        "per_ordering": rows,
        "totals": {
            "recorded_course_fits": sum(
                row["recorded_course_fits_delayed_plus_evaluation"]
                for row in rows),
            "support_probe_fits_DERIVED": sum(
                row["support_probe_fits_DERIVED"] for row in rows),
            "recorded_replay_fits": sum(row["recorded_replay_fits"]
                                        for row in rows),
            "recorded_shadow_fits": sum(row["recorded_shadow_fits"]
                                        for row in rows),
        },
        "derivation_of_the_support_line": (
            "scoped_evaluate charges 1 fit for the raw design and 1 more when "
            "a non-empty Scope makes the program pipeline run "
            "(scoped_serving_evaluator.py:200-210); every recorded probe "
            "resolved a non-empty Scope, so 2 each.  ScopeExecutor._baseline "
            "(scope_executor.py:330-337) is cached per (executor, origin) and "
            "the executor is built once per unit and shared by all four arms "
            "(run_hec1.py:343), so 1 more per unit that took any probe."),
        "why_it_is_derived_and_not_recorded": (
            "SupportReceipt carries no consumer_fits field, and "
            "ledgers.course_fits is incremented only at run_hec1.py:2032 "
            "(delayed) and :2186 (evaluation); the Support probe face is "
            "therefore absent from every recorded fit total"),
        "shadow_and_replay": (
            "best_stump is a deterministic search over bank rows and fits "
            "nothing, so shadow_fits = 0 is structural; replay_fits = 0 "
            "because no candidate ever reached the replay screen"),
        "offline_counterfactual_reuse": {
            "hec1_best_safe_global_v11p0": 1046,
            "hec1_validation_search_v11p0": 899,
            "hec1_validation_transfer_v11p0": 68,
            "total": 2013,
            "source": "each artifact's own cache / accounting block",
            "what_they_are": (
                "replays of cells the course already processed; they are "
                "offline counterfactual reuse, never new held-in feedback"),
        },
    }


# --------------------------------------------------------------------------
# 5. the three REVISE attempts
# --------------------------------------------------------------------------

def revise_attempts(docs: dict[str, dict]) -> list[dict[str, Any]]:
    out = []
    for ordering, doc in docs.items():
        for step in doc["outer_steps"]:
            for candidate in step["candidates"]:
                shadows = step["shadow_records"]
                out.append({
                    "ordering": ordering, "arm": step["arm"],
                    "k_index": step["k_index"],
                    "kind": candidate["kind"],
                    "program_signature": candidate["program_signature"],
                    "why": candidate["why"],
                    "recorded_outcome": candidate["outcome"],
                    "physical_slow_calls_this_step": step["slow_calls"],
                    "outer_llm_per_step": OUTER_LLM_PER_STEP,
                    "slow_proposals_that_reached_the_tool": len(shadows),
                    "slow_feature_and_direction": [row["slow"]
                                                   for row in shadows],
                    "shadow_outcomes": [(row["shadow"] or {}).get("outcome")
                                        for row in shadows],
                    "shadow_feasible_counts": [
                        (row["shadow"] or {}).get("feasible_count")
                        for row in shadows],
                    "calls_that_produced_a_calibrated_clause": 0,
                    "per_attempt_tool_outcome": (
                        "UNKNOWN: only the candidate's final outcome and the "
                        "shadow are persisted; whether an attempt ended as "
                        "NO_FEASIBLE_THRESHOLD or SLOW_CLAUSE_UNUSABLE cannot "
                        "be recovered from the artifact"),
                })
    return out


# --------------------------------------------------------------------------
# 5b. skill-id collisions across arms
# --------------------------------------------------------------------------

def skill_id_collisions(docs: dict[str, dict],
                        k0_ids: list[str]) -> dict[str, Any]:
    """Same skill_id in two arms is a naming collision, not a shared lineage."""
    seen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ordering, doc in docs.items():
        for arm in LLM_ARMS:
            arm_cells = cells(doc, arm)
            empty_prefix = [int(cell["position"]) for cell in arm_cells
                            if not (cell.get("snapshot_skill_ids_at_start") or ())]
            for cell in arm_cells:
                for skill_id in (cell.get("skills_minted_this_unit") or ()):
                    seen[str(skill_id)].append({
                        "ordering": ordering, "arm": arm,
                        "minted_at_position": int(cell["position"]),
                        "unit": unit_id(cell["unit"]),
                        "arm_snapshot_was_empty_before_position": (
                            max(empty_prefix) + 1 if empty_prefix else 0),
                    })
    rows = []
    for skill_id, mints in sorted(seen.items()):
        rows.append({
            "skill_id": skill_id,
            "also_the_id_of_a_K0_card": skill_id in k0_ids,
            "minted_independently": mints,
            "distinct_arms": sorted({row["arm"] for row in mints}),
            "verdict": (
                "NAMING_COLLISION_NOT_LINEAGE_IDENTITY" if len(mints) > 1
                or skill_id in k0_ids else "SINGLE_MINT"),
        })
    return {
        "why_ids_collide": (
            "_fast_winner_skill_id (methods/ttha/method.py:261-279) is a "
            "deterministic, hash-free function of (task_type, "
            "downstream_model_class, metric, operator).  HEC-1 holds all three "
            "of those fixed, so any arm that learns outlier_mad mints the same "
            "string.  An identical id across arms is therefore a name "
            "collision by construction and is never evidence of K0 leakage or "
            "of a shared lineage."),
        "independent_evidence_against_leakage": [
            "A3 arms start from h0: their snapshot_skill_ids_at_start is empty "
            "on every position before their own first activation",
            "the mint is recorded per cell in skills_minted_this_unit",
            "artifacts/main_protocol/hec1_k0_freeze_phase_s_v11_live.json's "
            "a5_a3_isolation check passed with leaks_into_non_k0_arms = []",
        ],
        "ids": rows,
    }


# --------------------------------------------------------------------------
# 5c. Appendix D zero-fit budget precheck
# --------------------------------------------------------------------------

#: Appendix D.2's finite edit set for a one-step ``outlier_*`` program, before
#: any deduplication.  Counted from the plan's own table, not invented here.
EDIT_SET = {
    "same_class_replacement": 3,
    "prefix_impute_step": 7,
    "order_swap_one_step_program": 0,
    "parameter_change_only_when_the_root_is_hampel_filter": 0,
}
E_UPPER_PRE_DEDUP = sum(EDIT_SET.values())
FITS_PER_SCORED_FACE = 3
FACES_PER_INSTANCE = 2
VERIFICATION_UNITS = 2
TOP_K = 2


def appendix_d_precheck(sits: dict[str, Any],
                        docs: dict[str, dict]) -> dict[str, Any]:
    """A zero-fit budget check on the plan's Appendix D.  No rule is decided."""
    repairable = [row for row in sits["situations"]
                  if row["a_workflow_edit_can_move_this_failure"]]
    usable_pairs = [(row, ordering, entry)
                    for row in repairable
                    for ordering, entry in row["per_ordering"].items()
                    if entry["later_evaluable_units"] >= VERIFICATION_UNITS]
    n_usable = len(usable_pairs)
    n_repairable_pairs = sum(len(row["per_ordering"]) for row in repairable)
    n_coverage_pairs = (sits["situation_by_ordering_pairs_total"]
                        - n_repairable_pairs)

    def bill(instances: int, edits: int) -> dict[str, int]:
        selection = instances * edits * FACES_PER_INSTANCE * FITS_PER_SCORED_FACE
        verification = (instances * TOP_K * VERIFICATION_UNITS
                        * FACES_PER_INSTANCE * FITS_PER_SCORED_FACE)
        control = (instances * VERIFICATION_UNITS * FACES_PER_INSTANCE
                   * FITS_PER_SCORED_FACE)
        stratification = (instances * TOP_K * VERIFICATION_UNITS
                          * FITS_PER_SCORED_FACE)
        exhaustive = (instances * (edits - TOP_K) * VERIFICATION_UNITS
                      * FACES_PER_INSTANCE * FITS_PER_SCORED_FACE)
        return {
            "selection_probes": selection,
            "verification_top_2": verification,
            "original_program_control_upper_bound": control,
            "post_hoc_plus144_stratification": stratification,
            "mandatory_upper_bound": (selection + verification + control
                                      + stratification),
            "mandatory_lower_bound_if_every_control_is_already_recorded": (
                selection + verification + stratification),
            "optional_post_hoc_exhaustive": exhaustive,
        }

    # how much of the "original programme control" is really already recorded
    control_available = 0
    control_needed = 0
    for row, ordering, entry in usable_pairs:
        doc = docs[ordering]
        scoreable = sorted({int(cell["position"]) for cell in doc["cells"]
                            if cell.get("evaluation")})
        later = [p for p in scoreable if p > entry["first_position"]][:2]
        for position in later:
            control_needed += 1
            if any(int(cell["position"]) == position
                   and cell.get("deployed")
                   and prog(cell["deployed"]) == row["program"]
                   and cell.get("delayed")
                   for cell in doc["cells"]):
                control_available += 1

    return {
        "scope": ("a zero-fit arithmetic check of "
                  "docs/NEXT_EXPERIMENT_EXECUTION_PLAN_2026-09-05.md Appendix "
                  "D.  It decides no experimental rule and approves no budget"),
        "finite_candidate_set": {
            "per_situation_breakdown": EDIT_SET,
            "upper_bound_before_any_deduplication": E_UPPER_PRE_DEDUP,
            "roots_observed_in_the_natural_cases": sorted(
                {row["program"] for row in sits["situations"]}),
            "no_observed_root_is_hampel_filter": all(
                row["program"] != "hampel_filter"
                for row in sits["situations"]),
            "why_E_equals_8_is_not_a_budget_guarantee": (
                "Appendix D.2 deduplicates by per-series gain vector and D.5 "
                "then prices selection at E = 8.  The gain vector is only "
                "known after the candidate has been evaluated, so the "
                "deduplicated count is an outcome of the spend, not a bound on "
                "it.  The only pre-fit bound is the enumerated set: E = %d."
                % E_UPPER_PRE_DEDUP),
        },
        "instance_count": {
            "plan_assumption": "12 = 4 situations x 3 orderings",
            "measured_situation_by_ordering_pairs": sits[
                "situation_by_ordering_pairs_total"],
            "of_which_coverage_floor_only": n_coverage_pairs,
            "of_which_a_workflow_edit_could_move": n_repairable_pairs,
            "of_which_also_have_at_least_2_later_evaluable_units": n_usable,
            "recommended_denominator_for_the_mandatory_bill": n_usable,
        },
        "coverage_floor_situations_are_unbuyable": {
            "situation_by_ordering_pairs": n_coverage_pairs,
            "wasted_selection_fits_if_bought_anyway": (
                n_coverage_pairs * E_UPPER_PRE_DEDUP * FACES_PER_INSTANCE
                * FITS_PER_SCORED_FACE),
            "why": (
                "their only failing line is the coverage floor, decided by "
                "resolving the fixed predicate against the raw window; no "
                "Workflow edit changes it, so D.3 rule (1) cannot be satisfied "
                "by any candidate and the outcome is RULE_FOUND_NO_FIX before "
                "the first fit"),
        },
        "recomputed_bill": {
            "at_plan_numbers_E8_x12": bill(12, 8),
            "at_measured_denominator_E10": bill(n_usable, E_UPPER_PRE_DEDUP),
            "at_all_repairable_pairs_E10": bill(n_repairable_pairs,
                                                E_UPPER_PRE_DEDUP),
            "assumptions": {
                "fits_per_scored_face": FITS_PER_SCORED_FACE,
                "faces_per_instance": FACES_PER_INSTANCE,
                "verification_units": VERIFICATION_UNITS,
                "top_k": TOP_K,
                "cache_saving_not_applied": (
                    "the plan's 'about 2/3 after caching' is left out on "
                    "purpose: a saving is not a bound"),
            },
        },
        "original_program_control": {
            "plan_text": "most already recorded -> 0",
            "later_unit_faces_needed": control_needed,
            "later_unit_faces_where_the_same_program_was_actually_deployed":
                control_available,
            "why_it_matters": (
                "the original programme's reading is recorded on the *conflict* "
                "unit, which D.4 does not use; the comparison needs it on the "
                "two later verification units, where the arm usually deployed "
                "something else or abstained"),
        },
        "tier_2_is_not_additive": {
            "tier_2_instances_all_arms": sum(
                1 for row in sits["situations"] for inst in row["instances"]
                if inst["card_held_deployment_rights_at_cell_start"]),
            "tier_2_instances_online_arms": sum(
                1 for row in sits["situations"] for inst in row["instances"]
                if inst["card_held_deployment_rights_at_cell_start"]
                and inst["arm"].endswith("online")),
            "tier_2_instances_online_arms_risk_lines_only": sum(
                1 for row in sits["situations"] for inst in row["instances"]
                if inst["card_held_deployment_rights_at_cell_start"]
                and inst["arm"].endswith("online")
                and not row["coverage_floor_only"]),
            "situations_that_are_tier_2": sum(
                1 for row in sits["situations"]
                if row["tier"] == "TIER_2_ACTIVE_CARD"),
            "situations_total": sits["unique_situations"],
            "why": (
                "Appendix D.5 prices Tier-2 as an addition of 90-100 fits per "
                "Active conflict case.  Measured, 7 of the 8 situations are "
                "already Tier-2: the same (cohort, origin, program, Scope) "
                "windows carry both labels.  A second full selection sweep "
                "would re-buy readings the Tier-1 sweep already bought.  The "
                "genuinely additional Tier-2 cost is the ancestor shadow "
                "contrast (v1 re-scored on the same faces), not a fresh "
                "candidate sweep"),
        },
        "contradictions_for_fable_to_reconcile": [
            {"id": "D-1", "where": "Appendix A row 'cost' vs Appendix C.9 vs "
                                   "Appendix D.5",
             "issue": "Appendix A still prices M-W at '300-540 fits' after C.9 "
                      "withdrew that number and D.5 replaced it with "
                      "1050-1150; the table was not updated"},
            {"id": "D-2", "where": "Appendix D.2 vs D.5",
             "issue": "D.5 budgets at the post-deduplication estimate E = 8 "
                      "while D.2's deduplication rule can only be applied "
                      "after the fits are spent; the pre-fit bound is E = 10"},
            {"id": "D-3", "where": "Appendix D.1 situation table",
             "issue": "the four situations are deduplicated by origin alone; "
                      "the recorded identity (cohort block x origin x delayed "
                      "origin x program x Scope) gives 8.  Origin 1176 covers "
                      "two different cohorts with opposite readings "
                      "(block [120:160]: 15 treated, worst harm 0.485; block "
                      "[80:120]: 3 treated, coverage floor only); the "
                      "([80:120], 2376 -> 2424) situation is absent from the "
                      "table entirely; and 2616 -> 2664 carries outlier_mad in "
                      "six instances as well as the single outlier_iqr the "
                      "table names"},
            {"id": "D-4", "where": "Appendix D.1 '<= 12 case instances'",
             "issue": "measured 20 (situation x ordering); 13 after removing "
                      "the coverage-floor-only ones; 11 after also requiring "
                      "two later evaluable units"},
            {"id": "D-5", "where": "section 3 Gate 0 vs Appendix A M-W stop "
                                   "condition vs Appendix D.4",
             "issue": "Gate 0 demands >= 8 scoreable windows per read cell, "
                      "D.4 gives each (situation x ordering) exactly 2 later "
                      "units, and Appendix A instead stops M-W at a half-width "
                      "of 0.15.  Three adequacy rules for one measurement"},
            {"id": "D-6", "where": "Appendix D.4 'same as the live path' vs "
                                   "D.5 '3 fits per scored face'",
             "issue": "the live Support face is ScopeExecutor.evaluate -- "
                      "scoped_evaluate at 2 fits plus a per-unit cached "
                      "baseline from forecast_runtime._evaluate -- while 3 "
                      "fits per face is _policy_reading, whose reference is a "
                      "scoped_evaluate call instead.  Which definition M-W "
                      "uses has to be fixed before any delta is comparable"},
            {"id": "D-7", "where": "Appendix D.5 Tier-2 line",
             "issue": "Tier-2 is priced as additive, but it labels the same "
                      "windows Tier-1 already buys; only the ancestor shadow "
                      "contrast is genuinely extra"},
            {"id": "D-8", "where": "section 1 status table",
             "issue": "'none of the 11 Drafts came from a once-Active Skill' "
                      "is contradicted by the record: 8 of 11 were created by "
                      "a delayed failure of a program the arm held an Active "
                      "card for at that cell's start"},
        ],
    }


# --------------------------------------------------------------------------
# 6. assembly
# --------------------------------------------------------------------------

def build() -> dict[str, Any]:
    docs = {name: load(name) for name in ORDERINGS}
    k0 = json.loads(K0_RECEIPT.read_text(encoding="utf-8"))
    k0_keys = list(k0.get("lineage_keys") or ())

    reach: list[dict[str, Any]] = []
    lin: list[dict[str, Any]] = []
    rights: list[dict[str, Any]] = []
    drafts_all: list[dict[str, Any]] = []
    for name, doc in docs.items():
        reach.extend(reachability(doc, name, k0_keys))
        lin.extend(lineages(doc, name))
        rights.extend(deployment_rights(doc, name))
        drafts_all.extend(draft_reconstruction(doc, name))

    sits = situations(docs)
    narrow_cf = sum(1 for step in reach for group in step["groups"]
                    if group["NARROW_counterfactual"])
    add_cf = sum(1 for step in reach for group in step["groups"]
                 if group["ADD_counterfactual"])
    steps_with_narrow_cf = sum(
        1 for step in reach
        if any(group["NARROW_counterfactual"] for group in step["groups"]))

    return {
        "stage": "M_R0_REVISION_REACHABILITY_AUDIT",
        "written_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "evidence_grade": ("INSTRUMENT / MECHANISM -- a diagnosis of why a "
                           "branch was unreachable; no capability claim"),
        "boundary": {
            "llm_calls": 0, "consumer_fits": 0, "held_out_reads": 0,
            "sealed_reads": 0, "thresholds_changed": 0, "code_changed": 0,
            "existing_artifacts_written": 0,
            "new_sha_or_hash_introduced": 0,
        },
        "version_discipline": version_check(),
        "inputs": {
            "course_artifacts": [str(path.relative_to(ROOT))
                                 for path in COURSE.values()],
            "k0_receipt": str(K0_RECEIPT.relative_to(ROOT)),
            "k0_lineage_keys": k0_keys,
        },
        "headline": {
            "outer_steps": len(reach),
            "steps_with_no_candidate_recorded": sum(
                1 for step in reach if step["recorded_empty_reason"]),
            "census_replay_reproduces_every_recorded_group": all(
                step["census_replay_reproduces_artifact"] for step in reach),
            "ADD_candidates_recorded": sum(
                step["recorded_candidate_kinds"].count("ADD") for step in reach),
            "NARROW_candidates_recorded": sum(
                step["recorded_candidate_kinds"].count("NARROW")
                for step in reach),
            "REVISE_candidates_recorded": sum(
                step["recorded_candidate_kinds"].count("REVISE")
                for step in reach),
            "REVOKE_candidates_recorded": sum(
                step["recorded_candidate_kinds"].count("REVOKE")
                for step in reach),
            "ADD_candidates_under_counterfactual_relation": add_cf,
            "NARROW_candidates_under_counterfactual_relation": narrow_cf,
            "outer_steps_that_would_have_proposed_NARROW": steps_with_narrow_cf,
            "unique_failure_situations": sits["unique_situations"],
            "failure_instances": sits["instances_total"],
            "drafts": len(drafts_all),
            "drafts_created_from_a_card_holding_deployment_rights": sum(
                1 for draft in drafts_all
                if draft["created_from_a_card_that_held_deployment_rights"]),
        },
        "blocking_condition": {
            "name": "CENSUS_RELATION_VOCABULARY_MISMATCH",
            "statement": (
                "the census's positive_units and adverse_units are counted "
                "over the strings POSITIVE / CONFLICT / NEGATIVE, but every "
                "bank row carries an AdmissionVerdict *reason* from a "
                "different vocabulary, so both counters are identically zero "
                "in all 30 outer steps"),
            "code_chain": [
                "methods/ttha/admission_policy.py:69-82 -- AdmissionVerdict "
                "has admitted / rule / reason / aggregate_gain / series_count "
                "/ harmed_count / harmed_fraction / max_single_series_harm, "
                "and no 'relation' field",
                "methods/ttha/online_loop.py:891-894 -- that verdict's "
                "to_dict() becomes probe['admission']",
                "run_hec1.py:1849 -- row['relation'] = "
                "probe['admission']['relation'], a key that does not exist, so "
                "always None",
                "run_hec1.py:1850 -- row['admission'] = "
                "probe['admission']['reason'], i.e. relation_positive / "
                "within_risk_budget / harmed_fraction_over_budget / "
                "single_series_harm_over_budget / aggregate_below_material_line",
                "outer_loop.py:209-211 -- _relation returns that reason "
                "upcased and never reaches its own gain-based fallback at "
                ":212-220",
                "outer_loop.py:363-365 -- positive_units counts 'POSITIVE'; "
                "adverse_units counts 'CONFLICT' + 'NEGATIVE'; neither string "
                "is ever produced",
                "outer_loop.py:416-417 -- ADD needs positive_units >= 1",
                "outer_loop.py:445-446 -- NARROW needs adverse_units >= 2 and "
                "key in held",
            ],
            "conditions_that_were_NOT_binding": [
                "key in held -- the K0 lineage key in the receipt is seeded "
                "into both A5 arms at course start (run_hec1.py:2513-2515) and "
                "is byte-identical to the census key the bank produces "
                "(forecast|pooled-ridge-a1|sMASE|outlier_mad({})@"
                "serving_series_predicate[local_robust_z_peak>=3]), so the "
                "held test would have passed at every step",
                "loss of deployment rights -- no arm's snapshot ever shrinks, "
                "no revocation is ever recommended, and propose_candidates "
                "emits ADD / NARROW / REVISE only, so the REVOKE branch at "
                "outer_loop.py:732 is unreachable in this protocol",
                "the outer LLM budget -- it bound only the three REVISE "
                "candidates the REVISABLE-Draft path did produce; it was never "
                "reached by an ADD or a NARROW",
            ],
            "second_independent_block_on_NARROW": {
                "name": "ADVERSE_EVIDENCE_NEVER_ENTERS_THE_BANK",
                "statement": (
                    "even with the vocabulary repaired, the adverse readings a "
                    "human would call the conflict are delayed-face gate "
                    "failures, and _bank_rows_from_round keeps probe rows only "
                    "(run_hec1.py:1836).  The delayed face is an authority, "
                    "never evidence, so the census counts Support-probe "
                    "outcomes and can only ever see the conflicts that were "
                    "already visible at Support"),
                "consequence": (
                    "the counterfactual NARROW counts in this report are what "
                    "the *Support* evidence would have produced; they are not "
                    "the same object as the delayed-face conflicts the "
                    "lifecycle recorded"),
            },
        },
        "reachability_by_outer_step": reach,
        "active_lineages": lin,
        "deployment_rights": rights,
        "drafts": drafts_all,
        "revise_attempts": revise_attempts(docs),
        "slow_proposal_cost": {
            "question": ("how many physical calls one complete structural "
                         "proposal costs (plan section 4.3)"),
            "outer_steps_that_reached_Slow": 3,
            "physical_calls_spent_there": 6,
            "attempts_that_reached_the_threshold_tool": 5,
            "attempts_that_returned_a_calibrated_clause": 0,
            "measured_cost_of_one_complete_structural_proposal": (
                "NOT MEASURED.  No proposal in the whole course reached "
                "CALIBRATED, so the course contains no completed structural "
                "proposal to price.  The plan must not infer 'four calls is "
                "enough' from the fact that two were exhausted"),
            "what_is_measured": (
                "one attempt costs exactly one physical call "
                "(outer_loop.py:643-645); a step stops at outer_llm_per_step = "
                "2 (outer_loop.py:638, hec1_contract.py:803)"),
            "internal_budget_contradiction": (
                "OuterBudget.retries_per_candidate = 2 (outer_loop.py:91) "
                "allows three attempts, but outer_llm_per_step = 2 "
                "(outer_loop.py:87) stops the loop after two.  The third "
                "attempt is unreachable by construction, so the retry "
                "allowance as written can never be spent"),
        },
        "situations": sits,
        "skill_id_collisions": skill_id_collisions(
            docs, list(k0.get("active_skill_ids") or ())),
        "billing": billing(docs),
        "fits": fits(docs),
        "appendix_d_zero_fit_precheck": appendix_d_precheck(sits, docs),
        "open_unknowns": [
            {"id": "U-1",
             "question": "per-attempt tool outcome inside an outer step",
             "status": "UNKNOWN",
             "why": ("only the candidate's final outcome and the shadow are "
                     "persisted; whether attempt 1 of forward k1 ended as "
                     "NO_FEASIBLE_THRESHOLD or SLOW_CLAUSE_UNUSABLE cannot be "
                     "recovered from the artifact")},
            {"id": "U-2",
             "question": "exact physical call count inside a capped cell",
             "status": "DERIVED, NOT RECORDED",
             "why": ("the counter is not persisted for the 47 cells that "
                     "raised AgentCallBudgetExceeded; 5 is read off the cap "
                     "the backend refused at.  If any of those cells made "
                     "fewer than 5 relay calls the 1088 figure is an upper "
                     "bound, and the lower bound is 853 (billed 666 plus the "
                     "187 spent-minus-one calls)")},
            {"id": "U-3",
             "question": "Support probe consumer fits",
             "status": "DERIVED, NOT RECORDED",
             "why": ("SupportReceipt has no consumer_fits field; 641 is "
                     "computed from scoped_evaluate's fit rule and the cached "
                     "per-unit baseline, not read off a receipt")},
            {"id": "U-4",
             "question": ("whether ScopeExecutor._baseline and "
                          "_policy_reading's static reference are numerically "
                          "the same reference"),
             "status": "UNKNOWN",
             "why": ("both evaluate the identity program, but through "
                     "forecast_runtime._evaluate and scoped_evaluate "
                     "respectively; no artifact compares them, and M-W's delta "
                     "definition depends on which one it adopts")},
            {"id": "U-5",
             "question": ("whether the census would still have produced a "
                          "usable NARROW after the vocabulary is repaired"),
             "status": "OUT OF SCOPE FOR THIS AUDIT",
             "why": ("the counterfactual here only shows the candidate would "
                     "have been proposed.  What Slow would then have returned, "
                     "and whether the threshold tool would have calibrated it, "
                     "is not decidable from a course that never asked")},
        ],
        "how_to_recompute": [
            "python -m evaluation.main_protocol_p4.audit_m_r0_reachability",
            "git diff d690850 -- evaluation/main_protocol_p4/outer_loop.py "
            "evaluation/main_protocol_p4/run_hec1.py   # must be empty",
        ],
    }


def main() -> int:
    report = build()
    out = ART / "m_r0_reachability.json"
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8")).get("stage")
        except (OSError, ValueError):
            existing = None
        if existing != report["stage"]:
            sys.stderr.write(
                "refusing to overwrite %s: it was written by %r, not by this "
                "audit\n" % (out, existing))
            return 2
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("wrote %s" % out)
    for key, value in report["headline"].items():
        print("  %-56s %s" % (key, value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
