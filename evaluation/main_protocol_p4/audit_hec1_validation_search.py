"""0-LLM validation-search control for the completed HEC-1 course.

This is a post-course comparator, never a Harness arm.  On each exposed
development unit it deterministically evaluates at most the frozen per-unit
probe budget on Support-A, under the same Runtime Scope initialiser and the
same four-line P4 gate, selects the best safe candidate, and redeploys that
unchanged Program+Scope specification on the +144 evaluation face.  It carries
nothing between units and writes nothing to an Episode bank or Skill library.

The candidate order is not chosen from these outcomes.  It is the pre-existing
order returned by ``audit_hec1_best_safe_global.menu``; identity is the free
fallback and the first ``PER_UNIT_ARM_BUDGET['probes']`` non-identity entries
are the bounded search list.  This makes the truncation mechanical and visible.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import audit_hec1_best_safe_global as bsg
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import hec1_scoreability as scoreability
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import scope_initializer as initializer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEM = "hec1_validation_search_v11p0"
COURSE_FILES = {
    name: PROJECT_ROOT / (
        "artifacts/main_protocol/hec1_course_v11p0_%s_live.json" % name)
    for name in contract.ORDERINGS
}


def _steps(program: Mapping[str, Any]) -> tuple[tuple[str, dict], ...]:
    return tuple((str(step["op"]), dict(step["params"]))
                 for step in program["steps"])


def _unit_key(unit: Mapping[str, Any]) -> tuple[str, int]:
    return str(unit["block"]), int(unit["origin"])


def frozen_candidates() -> list[dict[str, Any]]:
    """The mechanically truncated, pre-existing frozen menu."""
    budget = int(contract.PER_UNIT_ARM_BUDGET["probes"])
    non_identity = [dict(program) for program in bsg.menu()
                    if program.get("steps")]
    selected = non_identity[:budget]
    if len(selected) != budget:
        raise RuntimeError(
            "the frozen menu supplies %d non-identity candidates, not %d"
            % (len(selected), budget))
    return selected


def _compact(reading: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "treated": int(reading["treated"]),
        "served": int(reading["served"]),
        "aggregate_gain": float(reading["aggregate_gain"]),
        "harmed_fraction": float(reading["harmed_fraction"]),
        "max_single_series_harm": float(
            reading["max_single_series_harm"]),
    }


def _identity_evaluation(origin: int, served: int) -> dict[str, Any]:
    return {
        "origin": int(origin),
        "treated": 0,
        "served": int(served),
        "aggregate_gain": 0.0,
        "harmed_fraction": 0.0,
        "max_single_series_harm": 0.0,
        "identity": True,
        "gate": runner.authoritative_gate({
            "treated": 0,
            "aggregate_gain": 0.0,
            "harmed_fraction": 0.0,
            "max_single_series_harm": 0.0,
        }),
    }


def evaluate_unit(
    unit: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    support_cache: runner.ReplayPredictionCache,
    evaluation_cache: runner.ReplayPredictionCache,
) -> dict[str, Any]:
    ctx = runner.UnitContext(unit)
    support_rows: list[dict[str, Any]] = []
    for rank, program in enumerate(candidates):
        steps = _steps(program)
        initial = initializer.initialize(program["steps"])
        scope = dict(initial["scope"])
        resolved = ctx.resolve(scope, ctx.origin)
        row: dict[str, Any] = {
            "rank": rank + 1,
            "program_id": str(program["program_id"]),
            "steps": [dict(step) for step in program["steps"]],
            "scope": scope,
            "scope_source": initial["scope_source"],
            "resolved_count": len(resolved),
        }
        try:
            reading = support_cache.reading(
                ctx, ctx.origin, steps, frozenset(resolved))
        except runner.UnitFault as exc:
            row.update({"usable": False, "refusal": str(exc)[:200],
                        "gate": None})
        else:
            gate = runner.authoritative_gate(reading)
            row.update({"usable": True, "support": _compact(reading),
                        "gate": gate})
        support_rows.append(row)

    feasible = [row for row in support_rows
                if row.get("usable") and (row.get("gate") or {}).get("passes")]
    selected = (sorted(
        feasible,
        key=lambda row: (-float(row["support"]["aggregate_gain"]),
                         int(row["rank"]), str(row["program_id"])),
    )[0] if feasible else None)

    evaluation_origin = ctx.face_origin(runner.EVALUATION_OFFSET)
    if not scoreability.unit_is_scoreable(unit):
        evaluation = None
        evaluation_refusal = scoreability.UNSCOREABLE_REASON
    elif selected is None:
        evaluation = _identity_evaluation(evaluation_origin, len(ctx.eval_uids))
        evaluation_refusal = None
    else:
        selected_scope = dict(selected["scope"])
        resolved = ctx.resolve(selected_scope, evaluation_origin)
        try:
            reading = evaluation_cache.reading(
                ctx, evaluation_origin, _steps(selected), frozenset(resolved))
        except runner.UnitFault as exc:
            # Match the course's fail-closed evaluation semantics: an illegal
            # deployment contributes Static, while the refusal remains visible.
            evaluation = _identity_evaluation(
                evaluation_origin, len(ctx.eval_uids))
            evaluation_refusal = str(exc)[:200]
        else:
            evaluation = {
                "origin": evaluation_origin,
                **_compact(reading),
                "identity": False,
                "gate": runner.authoritative_gate(reading),
            }
            evaluation_refusal = None

    return {
        "unit": ctx.unit,
        "scoreable": scoreability.unit_is_scoreable(unit),
        "candidate_budget": len(candidates),
        "usable_candidates": sum(bool(row.get("usable"))
                                 for row in support_rows),
        "safe_support_candidates": len(feasible),
        "selected": ({
            key: selected[key] for key in
            ("rank", "program_id", "steps", "scope", "resolved_count",
             "support", "gate")
        } if selected is not None else {
            "program_id": "identity",
            "why": "no candidate cleared the Support four-line gate",
        }),
        "support_candidates": support_rows,
        "evaluation": evaluation,
        "evaluation_refusal": evaluation_refusal,
    }


def _load_courses() -> tuple[dict[str, dict[str, Any]], str]:
    courses: dict[str, dict[str, Any]] = {}
    commits = set()
    for name, path in COURSE_FILES.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("status") != "COMPLETE"
                or payload.get("mode") != "scientific"
                or payload.get("offline") is not False
                or payload.get("run_fault") is not None
                or payload.get("transport_failed") is not False):
            raise RuntimeError("course artifact is not a complete scientific run: %s"
                               % path)
        commits.add(str((payload.get("code_state") or {}).get("code_commit")))
        courses[name] = payload
    if len(commits) != 1:
        raise RuntimeError("the three course artifacts do not share one commit")
    return courses, commits.pop()


def _cell_map(course: Mapping[str, Any], arm: str) -> dict[tuple[str, int], Any]:
    return {
        _unit_key(cell["unit"]): cell.get("evaluation")
        for cell in course.get("cells") or () if cell.get("arm") == arm
    }


def _comparisons(units: Sequence[Mapping[str, Any]],
                 courses: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    validation = {
        _unit_key(row["unit"]): row.get("evaluation") for row in units
    }
    output: dict[str, Any] = {}
    for ordering, course in courses.items():
        rows = contract.ordering(ordering)
        per_arm: dict[str, Any] = {}
        for arm in ("A5-online", "A5-frozen", "A3-online"):
            actual = _cell_map(course, arm)
            paired = []
            for unit in rows:
                key = _unit_key(unit)
                left, right = actual.get(key), validation.get(key)
                if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                    continue
                paired.append(float(left["aggregate_gain"])
                              - float(right["aggregate_gain"]))
            per_arm[arm] = {
                "paired_points": len(paired),
                "arm_minus_validation_search_terminal": round(sum(paired), 6),
                "unit_wins_ties_losses": {
                    "wins": sum(value > 0 for value in paired),
                    "ties": sum(value == 0 for value in paired),
                    "losses": sum(value < 0 for value in paired),
                },
            }
        output[ordering] = per_arm
    return output


def build(limit: int) -> dict[str, Any]:
    started = time.time()
    frozen = contract.assert_frozen()
    if not frozen["frozen"]:
        raise RuntimeError("HEC-1 contract drifted: %s" % frozen["failures"])
    courses, course_commit = _load_courses()
    candidates = frozen_candidates()
    planned = contract.ordering("forward")[:max(0, int(limit))]
    support_cache = runner.ReplayPredictionCache("validation-search-support")
    evaluation_cache = runner.ReplayPredictionCache(
        "validation-search-evaluation")
    units = []
    for index, unit in enumerate(planned, start=1):
        units.append(evaluate_unit(
            unit, candidates, support_cache, evaluation_cache))
        print("  validation unit %d/%d: %s x %s" % (
            index, len(planned), unit["block"], unit["origin"]), flush=True)

    scoreable = [row for row in units if row["evaluation"] is not None]
    selected_nonidentity = [row for row in units
                            if row["selected"]["program_id"] != "identity"]
    eval_nonidentity = [row for row in scoreable
                        if not row["evaluation"]["identity"]]
    eval_positive = [row for row in scoreable
                     if float(row["evaluation"]["aggregate_gain"])
                     >= contract.RISK["material"]]
    eval_safe = [row for row in scoreable
                 if (row["evaluation"].get("gate") or {}).get("passes")]
    fits = int(support_cache.physical_fits + evaluation_cache.physical_fits)
    return {
        "stage": "HEC1_VALIDATION_SEARCH",
        "status": "COMPLETE" if units else "PLAN_ONLY_NOTHING_EVALUATED",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "data_version": contract.DATA_VERSION,
        "evidence_grade": "EXPOSED_DEVELOPMENT_0_LLM_COMPARATOR",
        "course_commit": course_commit,
        "definition": dict(contract.VALIDATION_SEARCH_BASELINE),
        "selection": {
            "face": "Support-A at origin o",
            "rule": "maximum aggregate gain among candidates passing the P4 four-line gate; frozen rank then program id break ties",
            "deployment": "same Program and same root Scope specification, re-resolved at o+144",
            "delayed_face": "unused because this comparator has no lifecycle or memory to authorize",
            "identity_fallback": True,
        },
        "menu": {
            "source": "audit_hec1_best_safe_global.menu (pre-existing order)",
            "available_nonidentity": len([p for p in bsg.menu() if p["steps"]]),
            "candidate_budget": int(contract.PER_UNIT_ARM_BUDGET["probes"]),
            "evaluated_program_ids": [p["program_id"] for p in candidates],
            "truncation": "first candidate_budget non-identity entries; no outcome enters ordering",
        },
        "units_available": len(contract.ordering("forward")),
        "units_considered": len(units),
        "units_scoreable": len(scoreable),
        "summary": {
            "units_with_any_safe_support_candidate": len(selected_nonidentity),
            "units_falling_back_to_identity_on_support": (
                len(units) - len(selected_nonidentity)),
            "scoreable_units_deploying_nonidentity": len(eval_nonidentity),
            "scoreable_units_with_material_eval_gain": len(eval_positive),
            "scoreable_units_clearing_eval_gate": len(eval_safe),
            "cumulative_evaluation_gain": round(sum(
                float(row["evaluation"]["aggregate_gain"])
                for row in scoreable), 6),
            "evaluation_harm_events": sum(
                (float(row["evaluation"]["harmed_fraction"])
                 > contract.RISK["max_harmed_fraction"])
                or (float(row["evaluation"]["max_single_series_harm"])
                    > contract.RISK["max_single_series_harm"])
                for row in scoreable),
            "selected_program_histogram": {
                name: sum(row["selected"]["program_id"] == name for row in units)
                for name in sorted({row["selected"]["program_id"] for row in units})
            },
        },
        "comparison_to_course_arms": _comparisons(units, courses),
        "units": units,
        "accounting": {
            "support": support_cache.to_dict(),
            "evaluation": evaluation_cache.to_dict(),
            "consumer_fits": fits,
            "llm_calls": 0,
        },
        "boundary": {
            "held_out_reads": 0,
            "episode_bank_reads": 0,
            "episode_bank_writes": 0,
            "skills_created": 0,
            "enters_any_arm": False,
            "thresholds_changed": 0,
        },
        "wall_seconds": round(time.time() - started, 1),
    }


def _md(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# HEC-1 0-LLM validation-search",
        "",
        "Post-course exposed-development comparator; it enters no Harness and opens no held-out Outcome.",
        "",
        "| item | value |",
        "| --- | ---: |",
        "| units considered / scoreable | %s / %s |" % (
            payload["units_considered"], payload["units_scoreable"]),
        "| candidates per unit | %s |" % payload["menu"]["candidate_budget"],
        "| any safe Support candidate | %s |" % s[
            "units_with_any_safe_support_candidate"],
        "| Support identity fallback | %s |" % s[
            "units_falling_back_to_identity_on_support"],
        "| material gain at evaluation | %s |" % s[
            "scoreable_units_with_material_eval_gain"],
        "| clears evaluation gate | %s |" % s[
            "scoreable_units_clearing_eval_gate"],
        "| cumulative evaluation gain | %+.6f |" % s[
            "cumulative_evaluation_gain"],
        "| Consumer fits | %s |" % payload["accounting"]["consumer_fits"],
        "| LLM calls | 0 |",
        "",
        "Selection histogram: `%s`." % json.dumps(
            s["selected_program_histogram"], ensure_ascii=False, sort_keys=True),
        "",
        "| unit | selected on Support | safe candidates | evaluation gain | evaluation gate |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["units"]:
        evaluation = row.get("evaluation")
        if evaluation is None:
            gain, gate = "—", "UNSCOREABLE"
        else:
            gain = "%+.6f" % float(evaluation["aggregate_gain"])
            gate = "PASS" if (evaluation.get("gate") or {}).get("passes") else "FAIL"
        lines.append("| %s x %s | `%s` | %s | %s | %s |" % (
            row["unit"]["block"], row["unit"]["origin"],
            row["selected"]["program_id"], row["safe_support_candidates"],
            gain, gate))
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=int, default=0,
                        help="units to evaluate; zero prints the plan only")
    parser.add_argument("--output-stem", default=DEFAULT_STEM)
    args = parser.parse_args(argv)
    payload = build(args.units)
    if args.units <= 0:
        print("candidate budget : %d" % payload["menu"]["candidate_budget"])
        print("units available : %d" % payload["units_available"])
        print("status          : %s" % payload["status"])
        return 0
    out_json = PROJECT_ROOT / (
        "artifacts/main_protocol/%s.json" % args.output_stem)
    out_md = PROJECT_ROOT / (
        "artifacts/main_protocol/%s.md" % args.output_stem)
    if out_json.exists() or out_md.exists():
        raise FileExistsError("refusing to overwrite %s / %s" % (out_json, out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    out_md.write_text(_md(payload), encoding="utf-8")
    print("consumer fits   : %d" % payload["accounting"]["consumer_fits"])
    print("cumulative gain : %+.6f" % payload["summary"]["cumulative_evaluation_gain"])
    print("wrote %s" % out_json.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
