"""Post-hoc transfer audit for Support-safe validation-search candidates.

The validation-search baseline selects one candidate on Support.  This audit
redeploys *every* candidate that cleared that frozen Support gate on the +144
evaluation face.  It distinguishes two mechanisms without changing the
baseline: the selector chose the wrong Support-safe candidate, or no
Support-safe candidate retained its safety across the window shift.

This is outcome-reading mechanism analysis, not a deployable policy.  It is
0-LLM, enters no Harness, and never reads Phase F.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import hec1_scoreability as scoreability
from evaluation.main_protocol_p4 import run_hec1 as runner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT / "artifacts/main_protocol/hec1_validation_search_v11p0.json")
DEFAULT_STEM = "hec1_validation_transfer_v11p0"


def _key(unit: Mapping[str, Any]) -> tuple[str, int]:
    return str(unit["block"]), int(unit["origin"])


def _steps(row: Mapping[str, Any]) -> tuple[tuple[str, dict], ...]:
    return tuple((str(step["op"]), dict(step["params"]))
                 for step in row["steps"])


def _compact(reading: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "treated": int(reading["treated"]),
        "served": int(reading["served"]),
        "aggregate_gain": float(reading["aggregate_gain"]),
        "harmed_fraction": float(reading["harmed_fraction"]),
        "max_single_series_harm": float(
            reading["max_single_series_harm"]),
    }


def build(source: Path) -> dict[str, Any]:
    started = time.time()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (payload.get("stage") != "HEC1_VALIDATION_SEARCH"
            or payload.get("status") != "COMPLETE"
            or int((payload.get("accounting") or {}).get(
                "llm_calls", -1)) != 0):
        raise RuntimeError("input is not a completed 0-LLM validation-search")
    if not contract.assert_frozen()["frozen"]:
        raise RuntimeError("HEC-1 contract drifted")

    by_key = {_key(unit): unit for unit in contract.ordering("forward")}
    cache = runner.ReplayPredictionCache("validation-transfer-evaluation")
    units: list[dict[str, Any]] = []
    for source_row in payload["units"]:
        unit = by_key[_key(source_row["unit"])]
        if not scoreability.unit_is_scoreable(unit):
            units.append({
                "unit": dict(unit), "scoreable": False,
                "why_unscoreable": scoreability.UNSCOREABLE_REASON,
                "support_safe_count": int(source_row["safe_support_candidates"]),
                "candidates": [],
            })
            continue
        ctx = runner.UnitContext(unit)
        evaluation_origin = ctx.face_origin(runner.EVALUATION_OFFSET)
        candidates = []
        for candidate in source_row["support_candidates"]:
            if not (candidate.get("gate") or {}).get("passes"):
                continue
            resolved = ctx.resolve(candidate["scope"], evaluation_origin)
            row: dict[str, Any] = {
                "program_id": candidate["program_id"],
                "support_rank": int(candidate["rank"]),
                "selected_on_support": (
                    candidate["program_id"]
                    == source_row["selected"]["program_id"]),
                "support": dict(candidate["support"]),
                "scope": dict(candidate["scope"]),
                "evaluation_resolved_count": len(resolved),
            }
            try:
                reading = cache.reading(
                    ctx, evaluation_origin, _steps(candidate),
                    frozenset(resolved))
            except runner.UnitFault as exc:
                row.update({"usable": False, "refusal": str(exc)[:200],
                            "evaluation": None, "evaluation_gate": None})
            else:
                gate = runner.authoritative_gate(reading)
                row.update({"usable": True, "evaluation": _compact(reading),
                            "evaluation_gate": gate})
            candidates.append(row)
        safe = [row for row in candidates
                if (row.get("evaluation_gate") or {}).get("passes")]
        selected = next((row for row in candidates
                         if row["selected_on_support"]), None)
        selected_safe = bool(
            selected and (selected.get("evaluation_gate") or {}).get("passes"))
        units.append({
            "unit": dict(unit),
            "scoreable": True,
            "support_safe_count": int(source_row["safe_support_candidates"]),
            "evaluation_safe_count": len(safe),
            "selected_retains_safety": selected_safe,
            "alternative_rescues_selected": bool(safe and not selected_safe),
            "no_support_safe_candidate_retains_safety": bool(
                candidates and not safe),
            "best_evaluation_safe_candidate": (
                max(safe, key=lambda row: (
                    float(row["evaluation"]["aggregate_gain"]),
                    -int(row["support_rank"]))) ["program_id"]
                if safe else None),
            "candidates": candidates,
        })

    with_support = [row for row in units
                    if row.get("scoreable") and row["support_safe_count"] > 0]
    pairs = [candidate for row in units for candidate in row["candidates"]]
    retained = [row for row in pairs
                if (row.get("evaluation_gate") or {}).get("passes")]
    return {
        "stage": "HEC1_VALIDATION_TRANSFER_DIAGNOSTIC",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "data_version": contract.DATA_VERSION,
        "source_artifact": source.relative_to(PROJECT_ROOT).as_posix(),
        "course_commit": payload["course_commit"],
        "oracle_banner": (
            "POST_HOC_OUTCOME_DIAGNOSTIC: evaluation outcomes are used only "
            "to attribute failure; no choice here is deployable or enters an arm"
        ),
        "summary": {
            "scoreable_units_with_support_safe_candidates": len(with_support),
            "support_safe_candidate_pairs_redeployed": len(pairs),
            "candidate_pairs_retaining_evaluation_safety": len(retained),
            "units_where_selected_retains_safety": sum(
                row["selected_retains_safety"] for row in with_support),
            "units_rescuable_by_another_support_safe_candidate": sum(
                row["alternative_rescues_selected"] for row in with_support),
            "units_where_no_support_safe_candidate_retains_safety": sum(
                row["no_support_safe_candidate_retains_safety"]
                for row in with_support),
        },
        "units": units,
        "accounting": {**cache.to_dict(), "llm_calls": 0},
        "boundary": {
            "held_out_reads": 0,
            "enters_any_arm": False,
            "episode_bank_reads": 0,
            "episode_bank_writes": 0,
            "thresholds_changed": 0,
        },
        "wall_seconds": round(time.time() - started, 1),
    }


def _md(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# HEC-1 validation transfer diagnostic",
        "", payload["oracle_banner"], "",
        "| item | value |", "| --- | ---: |",
        "| scoreable units with a Support-safe candidate | %s |" % s[
            "scoreable_units_with_support_safe_candidates"],
        "| Support-safe candidate pairs redeployed | %s |" % s[
            "support_safe_candidate_pairs_redeployed"],
        "| candidate pairs retaining safety | %s |" % s[
            "candidate_pairs_retaining_evaluation_safety"],
        "| selected candidate retains safety | %s |" % s[
            "units_where_selected_retains_safety"],
        "| another Support-safe candidate would rescue | %s |" % s[
            "units_rescuable_by_another_support_safe_candidate"],
        "| no Support-safe candidate retains safety | %s |" % s[
            "units_where_no_support_safe_candidate_retains_safety"],
        "| Consumer fits | %s |" % payload["accounting"]["physical_fits"],
        "| LLM calls | 0 |", "",
        "| unit | Support-safe | evaluation-safe | selected stable | alternative rescue |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["units"]:
        if not row["scoreable"]:
            continue
        lines.append("| %s x %s | %s | %s | %s | %s |" % (
            row["unit"]["block"], row["unit"]["origin"],
            row["support_safe_count"], row["evaluation_safe_count"],
            row["selected_retains_safety"],
            row["alternative_rescues_selected"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", default=DEFAULT_STEM)
    args = parser.parse_args(argv)
    payload = build(args.input.resolve())
    out_json = PROJECT_ROOT / (
        "artifacts/main_protocol/%s.json" % args.output_stem)
    out_md = PROJECT_ROOT / (
        "artifacts/main_protocol/%s.md" % args.output_stem)
    if out_json.exists() or out_md.exists():
        raise FileExistsError("refusing to overwrite %s / %s" % (out_json, out_md))
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    out_md.write_text(_md(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print("consumer fits : %d" % payload["accounting"]["physical_fits"])
    print("wrote %s" % out_json.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
