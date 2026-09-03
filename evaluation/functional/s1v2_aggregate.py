"""S1-v2 independent ITT aggregator (PREP-1).

Read-only.  Does not import or write the in-flight forward runner.
Inputs: the frozen course plus each run's artifact and/or checkpoint.
Outputs: per-arm unit tables, a knowledge-formation timeline, frozen
material-gate arithmetic, and the pre-registered verdict map.

  python evaluation/functional/s1v2_aggregate.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
FREEZE_R2 = E2 / "s1v2_course_freeze_r2.json"

ARMS = ("Static", "A3-reset", "K0-fixed", "A5-online")
ARM_A5 = "A5-online"
ARM_A3 = "A3-reset"
ARM_K0 = "K0-fixed"
QUALITY_TOL = -0.005
HELD_OUT_THRESHOLD = 0.005


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _freeze(path: Path | None = None) -> dict[str, Any]:
    return _load_json(path or FREEZE_R2)


def _run_inputs(run_id: str) -> dict[str, Any]:
    """Prefer a finished artifact; fall back to the live checkpoint."""
    artifact = E2 / ("s1v2_forward_run%s.json" % run_id)
    checkpoint = E2 / ("s1v2_forward_run%s.checkpoint.json" % run_id)
    source = None
    payload: dict[str, Any] = {}
    if artifact.is_file():
        payload = _load_json(artifact)
        source = artifact
    elif checkpoint.is_file():
        payload = _load_json(checkpoint)
        source = checkpoint
    return {
        "run_id": run_id,
        "source": (source.relative_to(PROJECT_ROOT).as_posix()
                   if source else None),
        "source_kind": ("artifact" if artifact.is_file()
                        else "checkpoint" if checkpoint.is_file()
                        else "missing"),
        "payload": payload,
    }


def _course_index(freeze: Mapping[str, Any]) -> dict[str, Any]:
    course = list(freeze.get("course") or [])
    beneficiaries = [row for row in course
                     if str(row.get("role", "")).startswith("beneficiary")]
    producers = [row for row in course
                 if str(row.get("role", "")).startswith("producer")]
    return {
        "course": course,
        "n_units": len(course),
        "beneficiaries": beneficiaries,
        "beneficiary_ids": [row["unit_id"] for row in beneficiaries],
        "producers": producers,
        "convertible_unit_ids": [row["unit_id"] for row in beneficiaries],
        "expected_card_after": 3,
        "expected_first_fork": 4,
    }


def _is_partial(rows: Sequence[Mapping[str, Any]],
                freeze: Mapping[str, Any]) -> dict[str, Any]:
    course = list(freeze.get("course") or [])
    expected = {(int(unit["position"]), arm)
                for unit in course for arm in ARMS}
    have = {(int(row["position"]), str(row["arm"])) for row in rows}
    missing = sorted("%s/%s" % (pos, arm) for pos, arm in (expected - have))
    complete_positions = sorted({
        int(unit["position"]) for unit in course
        if all((int(unit["position"]), arm) in have for arm in ARMS)
    })
    return {
        "partial": bool(missing),
        "n_rows": len(rows),
        "n_expected": len(expected),
        "complete_positions": complete_positions,
        "missing": missing[:24],
        "missing_total": len(missing),
    }


def _arm_rows(rows: Sequence[Mapping[str, Any]], arm: str
              ) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("arm") == arm]


def _unit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    recall = dict(row.get("heldout_recall_by_class") or {})
    applied = list(row.get("applied_ops") or [])
    utility = float(row.get("heldout_utility") or 0.0)
    harm = bool(row.get("harm_event"))
    wrong = bool(applied) and (harm or utility < 0.0)
    seconds = float(row.get("seconds") or 0.0)
    fit_wall = row.get("fit_wall_seconds")
    if fit_wall is None:
        fit_wall = None
    return {
        "position": row.get("position"),
        "unit_id": row.get("unit_id"),
        "role": row.get("role"),
        "arm": row.get("arm"),
        "applied_ops": applied or ["identity"],
        "heldout_utility": utility,
        "regret": float(row.get("regret") or 0.0),
        "heldout_recall_by_class": recall,
        "worst_class_delta": float(row.get("worst_class_delta") or 0.0),
        "harm_event": harm,
        "wrong_promotion": wrong,
        "llm_calls": int(row.get("llm_calls") or 0),
        "probes": int(row.get("probes") or 0),
        "consumer_fits": int(row.get("consumer_fits") or 0),
        "unit_seconds": seconds,
        "fit_wall_seconds": fit_wall,
        "fit_wall_note": (
            "checkpoint has no separate consumer-fit wall clock; "
            "unit_seconds includes LLM" if fit_wall is None else "recorded"),
        "supply_candidates_in_pool": int(
            row.get("supply_candidates_in_pool") or 0),
        "supply_probed": int(row.get("supply_probed") or 0),
        "approved_skill_ids": list(row.get("approved_skill_ids") or []),
    }


def per_unit_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_unit_row(row) for row in rows]


def _time_to_threshold(rows: Sequence[Mapping[str, Any]],
                       arm: str) -> dict[str, Any]:
    cumulative_fits = 0
    cumulative_seconds = 0.0
    for row in sorted(_arm_rows(rows, arm),
                      key=lambda item: int(item.get("position") or 0)):
        cumulative_fits += int(row.get("consumer_fits") or 0)
        cumulative_seconds += float(row.get("seconds") or 0.0)
        if float(row.get("heldout_utility") or 0.0) >= HELD_OUT_THRESHOLD:
            return {
                "reached": True,
                "arm": arm,
                "first_position": row.get("position"),
                "first_unit_id": row.get("unit_id"),
                "threshold": HELD_OUT_THRESHOLD,
                "cumulative_fits": cumulative_fits,
                "cumulative_unit_seconds": round(cumulative_seconds, 2),
                "note": "threshold = held-out utility >= 0.005; seconds include LLM",
            }
    return {
        "reached": False, "arm": arm, "threshold": HELD_OUT_THRESHOLD,
        "cumulative_fits": cumulative_fits,
        "cumulative_unit_seconds": round(cumulative_seconds, 2),
    }


def _arm_summary(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    arm_rows = _arm_rows(rows, arm)
    utilities = [float(row.get("heldout_utility") or 0.0) for row in arm_rows]
    regrets = [float(row.get("regret") or 0.0) for row in arm_rows]
    worst = [float(row.get("worst_class_delta") or 0.0) for row in arm_rows]
    return {
        "arm": arm,
        "units": len(arm_rows),
        "cumulative_regret": sum(regrets),
        "mean_heldout_utility": (sum(utilities) / len(utilities)
                                 if utilities else 0.0),
        "worst_class_min": min(worst) if worst else 0.0,
        "harm_events": sum(1 for row in arm_rows if row.get("harm_event")),
        "wrong_promotions": sum(1 for row in arm_rows
                                if _unit_row(row)["wrong_promotion"]),
        "llm": sum(int(row.get("llm_calls") or 0) for row in arm_rows),
        "probes": sum(int(row.get("probes") or 0) for row in arm_rows),
        "consumer_fits": sum(int(row.get("consumer_fits") or 0)
                             for row in arm_rows),
        "unit_seconds": sum(float(row.get("seconds") or 0.0)
                            for row in arm_rows),
        "time_to_threshold": _time_to_threshold(rows, arm),
    }


def _producer_episodes(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _arm_rows(rows, ARM_A5):
        if not str(row.get("role", "")).startswith("producer"):
            continue
        for record in row.get("rounds") or []:
            for episode in record.get("episodes") or []:
                out.append({
                    "unit_id": row.get("unit_id"),
                    "position": row.get("position"),
                    "round": record.get("round"),
                    "program": episode.get("workflow_signature"),
                    "relation": episode.get("relation"),
                    "local_status": episode.get("local_status"),
                    "support_gain": episode.get("support_gain"),
                    "delayed_gain": episode.get("delayed_gain"),
                    "earned": (str(episode.get("relation")) == "POSITIVE"
                               and str(episode.get("local_status"))
                               == "LOCAL_ACTIVE"),
                })
    return out


def knowledge_timeline(rows: Sequence[Mapping[str, Any]],
                       events: Sequence[Mapping[str, Any]],
                       freeze: Mapping[str, Any]) -> dict[str, Any]:
    scope = dict((freeze.get("producers_scope_v1") or {}).get(
        "pattern_intersection") or {})
    cards = []
    for event in events:
        cards.append({
            "after_position": event.get("after_position"),
            "after_unit": event.get("after_unit"),
            "card_compiled": bool(event.get("card_compiled")),
            "installed": bool(event.get("installed")),
            "skill_id": event.get("skill_id") or event.get("card_id"),
            "leaf_count": (len(scope) if event.get("card_compiled")
                           else 0),
            "expected_leaf_count_if_compiled": len(scope),
            "supply_rows": event.get("supply_rows"),
            "withheld_because": event.get("withheld_because"),
            "audit": event.get("audit"),
        })
    beneficiaries = {
        row["unit_id"] for row in freeze.get("course") or []
        if str(row.get("role", "")).startswith("beneficiary")
    }
    injects = []
    for row in _arm_rows(rows, ARM_A5):
        if row.get("unit_id") not in beneficiaries:
            continue
        entered = int(row.get("supply_candidates_in_pool") or 0) > 0
        probed = int(row.get("supply_probed") or 0) > 0
        converted = bool(entered and row.get("applied_ops")
                         and float(row.get("heldout_utility") or 0.0) > 0)
        injects.append({
            "position": row.get("position"),
            "unit_id": row.get("unit_id"),
            "role": row.get("role"),
            "entered_pool": entered,
            "probed": probed,
            "dual_gate_converted": converted,
            "deployed_ops": list(row.get("applied_ops") or []) or ["identity"],
            "heldout_utility": float(row.get("heldout_utility") or 0.0),
        })
    fork = first_fork(rows)
    return {
        "producer_episodes": _producer_episodes(rows),
        "boundary_cards": cards,
        "beneficiary_inject_and_conversion": injects,
        "first_fork_a5_vs_k0": fork,
        "expected_card_boundary": "Slow after position 3",
        "expected_first_fork_position": 4,
    }


def first_fork(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_pos: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_pos.setdefault(int(row["position"]), {})[str(row["arm"])] = row
    for position in sorted(by_pos):
        a5 = by_pos[position].get(ARM_A5)
        k0 = by_pos[position].get(ARM_K0)
        if a5 is None or k0 is None:
            continue
        diffs = []
        if list(a5.get("applied_ops") or []) != list(k0.get("applied_ops") or []):
            diffs.append("applied_ops")
        if int(a5.get("supply_candidates_in_pool") or 0) != int(
                k0.get("supply_candidates_in_pool") or 0):
            diffs.append("supply_candidates_in_pool")
        if abs(float(a5.get("heldout_utility") or 0)
               - float(k0.get("heldout_utility") or 0)) > 1e-12:
            diffs.append("heldout_utility")
        if a5.get("retrieved_skill_ids") != k0.get("retrieved_skill_ids"):
            a5_ids = []
            k0_ids = []
            for record in a5.get("rounds") or []:
                a5_ids.extend(record.get("retrieved_skill_ids") or [])
            for record in k0.get("rounds") or []:
                k0_ids.extend(record.get("retrieved_skill_ids") or [])
            if sorted(a5_ids) != sorted(k0_ids):
                diffs.append("retrieved_skill_ids")
        if diffs:
            return {
                "position": position,
                "unit_id": a5.get("unit_id"),
                "diff_fields": diffs,
                "a5_applied": list(a5.get("applied_ops") or []) or ["identity"],
                "k0_applied": list(k0.get("applied_ops") or []) or ["identity"],
            }
    return {"position": None, "diff_fields": [],
            "note": "no A5-online vs K0-fixed divergence on completed pairs"}


def material_gates(rows: Sequence[Mapping[str, Any]],
                   freeze: Mapping[str, Any]) -> dict[str, Any]:
    ruling = dict((freeze.get("rulings") or {}).get("b_regret_gate") or {})
    delta = float(ruling.get("delta_material", freeze.get("delta_material") or 0))
    parts = dict(ruling.get("parts") or {})
    if not parts:
        parts = {
            row["unit_id"]: float(row["material_line"])
            for row in freeze.get("beneficiaries") or []
        }
    index = _course_index(freeze)
    convertible = list(index["convertible_unit_ids"])
    a5 = _arm_summary(rows, ARM_A5)
    a3 = _arm_summary(rows, ARM_A3)
    k0 = _arm_summary(rows, ARM_K0)
    n_conv = max(len(convertible), 1)
    probe_saved = a3["probes"] - a5["probes"]
    return {
        "source": "s1v2_course_freeze_r2.json rulings.b_regret_gate",
        "regret_gate_definition": ruling.get(
            "definition",
            "sum of the two beneficiaries' half-protocol material lines"),
        "delta_material": delta,
        "parts": parts,
        "cost_gate_definition": ruling.get(
            "cost_gate_unchanged",
            "convertible units average >= 1 probe saved"),
        "n_convertible_units": len(convertible),
        "convertible_unit_ids": convertible,
        "regret_gap_vs_a3": a3["cumulative_regret"] - a5["cumulative_regret"],
        "regret_gap_vs_k0": k0["cumulative_regret"] - a5["cumulative_regret"],
        "regret_gate_vs_a3": (
            a3["cumulative_regret"] - a5["cumulative_regret"]) >= delta,
        "regret_gate_vs_k0": (
            k0["cumulative_regret"] - a5["cumulative_regret"]) >= delta,
        "probe_gap_vs_a3": probe_saved,
        "probe_saved_per_convertible": probe_saved / n_conv,
        "cost_gate": probe_saved >= len(convertible),
        "quality_gap_vs_a3": (a5["mean_heldout_utility"]
                              - a3["mean_heldout_utility"]),
        "quality_gap_vs_k0": (a5["mean_heldout_utility"]
                              - k0["mean_heldout_utility"]),
        "non_inferior": (
            a5["mean_heldout_utility"] - a3["mean_heldout_utility"]
            >= QUALITY_TOL
            and a5["mean_heldout_utility"] - k0["mean_heldout_utility"]
            >= QUALITY_TOL
            and a5["worst_class_min"] >= min(
                a3["worst_class_min"], k0["worst_class_min"]) + QUALITY_TOL
            and a5["harm_events"] <= min(a3["harm_events"], k0["harm_events"])
        ),
    }


def itt_funnel(rows: Sequence[Mapping[str, Any]],
               freeze: Mapping[str, Any]) -> dict[str, Any]:
    """ITT: Scope-qualified inject miss is an A5 failure.

    Conditional conversion given successful inject is a secondary readout.
    """
    matched = {
        row["unit_id"] for row in freeze.get("beneficiaries") or []
        if row.get("machine_match_producer_scope")
    }
    if not matched:
        matched = {
            row["unit_id"] for row in freeze.get("course") or []
            if str(row.get("role", "")).startswith("beneficiary")
        }
    a5 = [row for row in _arm_rows(rows, ARM_A5)
          if row.get("unit_id") in matched]
    scoped = len(a5)
    injected = [row for row in a5
                if int(row.get("supply_candidates_in_pool") or 0) > 0]
    inject_miss = scoped - len(injected)
    converted = [row for row in injected
                 if row.get("applied_ops")
                 and float(row.get("heldout_utility") or 0.0) > 0]
    return {
        "analysis": "ITT",
        "scope_qualified_beneficiary_rows": scoped,
        "injected": len(injected),
        "inject_miss_counts_as_a5_failure": inject_miss,
        "itt_conversion": len(converted),
        "conditional_conversion_rate": (
            len(converted) / len(injected) if injected else None),
        "conditional_note": (
            "conditional rate is secondary and is not the main verdict"
        ),
        "scope_qualified_ids": sorted(matched),
    }


def _any_fast_knowledge(rows: Sequence[Mapping[str, Any]],
                        events: Sequence[Mapping[str, Any]]) -> bool:
    if any(event.get("card_compiled") or event.get("installed")
           for event in events):
        return True
    return any(int(row.get("supply_candidates_in_pool") or 0) > 0
               for row in rows)


def verdict_one(rows: Sequence[Mapping[str, Any]],
                events: Sequence[Mapping[str, Any]],
                freeze: Mapping[str, Any],
                *, partial: bool) -> dict[str, Any]:
    gates = material_gates(rows, freeze)
    itt = itt_funnel(rows, freeze)
    cards = [event for event in events
             if event.get("card_compiled") or event.get("installed")]
    knowledge = _any_fast_knowledge(rows, events)
    injected = int(itt["injected"])
    attributable = bool(cards and injected)
    withheld = next((event.get("withheld_because") for event in events
                     if event.get("withheld_because")), None)
    first_fault = None
    if not knowledge:
        first_fault = withheld or "no_fast_visible_self_produced_knowledge"
        return {
            "verdict": "TREATMENT_EMPTY",
            "subtype": None,
            "first_fault": first_fault,
            "partial": partial,
            "reason": (
                "no Fast-visible self-produced knowledge: no supply card "
                "compiled and no supplied candidate entered a pool"
                + (" (partial table; not a finished-course freeze)"
                   if partial else "")
            ),
            "gates": gates, "itt": itt,
        }
    if not attributable and knowledge:
        subtype = "PRIOR_ONLY" if not cards else "NO_TRANSFER"
        first_fault = (
            "advantage_or_knowledge_without_in_course_card"
            if subtype == "PRIOR_ONLY"
            else "knowledge_produced_but_no_beneficiary_inject"
        )
        return {
            "verdict": "NO_EVOLUTION_SIGNAL",
            "subtype": subtype,
            "first_fault": first_fault,
            "partial": partial,
            "reason": (
                "knowledge-like activity is not an in-course transfer "
                "chain (card -> beneficiary inject -> later-unit change)"
            ),
            "gates": gates, "itt": itt,
        }
    a5 = _arm_summary(rows, ARM_A5)
    a3 = _arm_summary(rows, ARM_A3)
    if (gates["quality_gap_vs_a3"] < QUALITY_TOL
            or a5["harm_events"] > a3["harm_events"]):
        return {
            "verdict": "NO_EVOLUTION_SIGNAL",
            "subtype": "NEGATIVE_TRANSFER",
            "first_fault": "a5_worse_than_a3_on_quality_or_harm",
            "partial": partial,
            "reason": "A5-online is worse than A3-reset on quality or harm",
            "gates": gates, "itt": itt,
        }
    if gates["non_inferior"] and (
            (gates["regret_gate_vs_a3"] and gates["regret_gate_vs_k0"])
            or gates["cost_gate"]) and attributable:
        return {
            "verdict": "S1V2_FORWARD_SIGNAL",
            "subtype": None,
            "first_fault": None,
            "partial": partial,
            "reason": (
                "A5-online is non-inferior and clears a material gate; "
                "the advantage traces to an in-course compiled card "
                "that reached a beneficiary"
            ),
            "gates": gates, "itt": itt,
        }
    return {
        "verdict": "NO_EVOLUTION_SIGNAL",
        "subtype": "NO_TRANSFER",
        "first_fault": "card_reached_or_existed_but_gates_not_cleared",
        "partial": partial,
        "reason": "in-course knowledge did not clear regret or cost gates",
        "gates": gates, "itt": itt,
    }


def merge_verdicts(v1: Mapping[str, Any],
                   v2: Mapping[str, Any] | None) -> dict[str, Any]:
    if v2 is None:
        return {
            "verdict": v1.get("verdict"),
            "subtype": v1.get("subtype"),
            "partial": True,
            "reason": "only one forward run is available",
            "runs": [v1],
        }
    names = (v1.get("verdict"), v2.get("verdict"))
    if names == ("S1V2_FORWARD_SIGNAL", "S1V2_FORWARD_SIGNAL"):
        return {
            "verdict": "S1V2_FORWARD_SIGNAL",
            "subtype": None,
            "partial": bool(v1.get("partial") or v2.get("partial")),
            "reason": "both forward runs return S1V2_FORWARD_SIGNAL",
            "runs": [v1, v2],
        }
    if "TREATMENT_EMPTY" in names:
        return {
            "verdict": "TREATMENT_EMPTY",
            "subtype": None,
            "partial": True,
            "reason": "at least one run is TREATMENT_EMPTY",
            "runs": [v1, v2],
        }
    return {
        "verdict": "NO_EVOLUTION_SIGNAL",
        "subtype": v1.get("subtype") or v2.get("subtype"),
        "partial": bool(v1.get("partial") or v2.get("partial")),
        "reason": "the two forward runs do not jointly confirm SIGNAL",
        "runs": [v1, v2],
    }


def aggregate_run(run_id: str, freeze: Mapping[str, Any]) -> dict[str, Any]:
    bundle = _run_inputs(run_id)
    payload = bundle["payload"]
    rows = list(payload.get("rows") or [])
    events = list(payload.get("events") or payload.get("supply_events") or [])
    completeness = _is_partial(rows, freeze)
    verdict = verdict_one(rows, events, freeze,
                          partial=completeness["partial"])
    return {
        "run_id": run_id,
        "source": bundle["source"],
        "source_kind": bundle["source_kind"],
        "partial": completeness["partial"],
        "completeness": completeness,
        "ledger": payload.get("ledger"),
        "per_unit": per_unit_table(rows),
        "arm_summaries": {arm: _arm_summary(rows, arm) for arm in ARMS},
        "knowledge_timeline": knowledge_timeline(rows, events, freeze),
        "material_gates": material_gates(rows, freeze),
        "itt": itt_funnel(rows, freeze),
        "verdict": verdict,
    }


def aggregate(*, freeze_path: Path | None = None) -> dict[str, Any]:
    freeze = _freeze(freeze_path)
    run1 = aggregate_run("1", freeze)
    run2 = aggregate_run("2", freeze)
    have_run2 = run2["source_kind"] != "missing"
    merged = merge_verdicts(
        run1["verdict"], run2["verdict"] if have_run2 else None)
    return {
        "protocol": "s1v2_aggregate_itt_v1",
        "freeze": FREEZE_R2.relative_to(PROJECT_ROOT).as_posix(),
        "freeze_verdict": (freeze.get("verdict") or {}).get("verdict"),
        "delta_material": freeze.get("delta_material"),
        "course": [
            {"position": row["position"], "role": row["role"],
             "unit_id": row["unit_id"]}
            for row in freeze.get("course") or []
        ],
        "run1": run1,
        "run2": run2 if have_run2 else {
            "run_id": "2", "source_kind": "missing", "partial": True,
            "verdict": None,
        },
        "merged_verdict": merged,
        "label": "partial" if (run1["partial"] or not have_run2) else "complete",
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    run1 = payload["run1"]
    merged = payload["merged_verdict"]
    lines = [
        "# S1-v2 ITT aggregator",
        "",
        "label: **%s**  freeze: `%s`" % (
            payload["label"], payload["freeze"]),
        "",
        "run1 source: `%s` (%s)" % (run1.get("source"), run1.get("source_kind")),
        "",
        "**run1 %s**%s" % (
            run1["verdict"]["verdict"],
            (" / %s" % run1["verdict"]["subtype"]
             if run1["verdict"].get("subtype") else "")),
        "",
        run1["verdict"].get("reason", ""),
        "",
        "merged: **%s** — %s" % (merged.get("verdict"), merged.get("reason")),
        "",
        "## Per-unit sample (run1, completed rows)",
        "",
        "| # | role | unit | arm | deployed | held-out | regret | worst | "
        "harm | probes | LLM | fits | supply |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in run1.get("per_unit") or []:
        lines.append(
            "| %s | %s | %s | %s | `%s` | %+.4f | %+.4f | %+.4f | %s | "
            "%s | %s | %s | %s |" % (
                row["position"], row["role"],
                str(row["unit_id"]).split("__")[0], row["arm"],
                ",".join(row["applied_ops"]),
                row["heldout_utility"], row["regret"],
                row["worst_class_delta"], row["harm_event"],
                row["probes"], row["llm_calls"], row["consumer_fits"],
                row["supply_candidates_in_pool"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="S1-v2 independent ITT aggregator")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    args = parser.parse_args()
    payload = aggregate()
    text = json.dumps(payload, indent=1, ensure_ascii=False, default=str)
    if args.json_out:
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "label": payload["label"],
        "run1_source": payload["run1"].get("source"),
        "run1_partial": payload["run1"].get("partial"),
        "run1_verdict": payload["run1"]["verdict"]["verdict"],
        "run1_subtype": payload["run1"]["verdict"].get("subtype"),
        "run1_first_fault": payload["run1"]["verdict"].get("first_fault"),
        "merged_verdict": payload["merged_verdict"].get("verdict"),
        "n_run1_rows": payload["run1"]["completeness"]["n_rows"],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
