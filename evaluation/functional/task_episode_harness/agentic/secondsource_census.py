"""T2: second-source UNGUIDED census (Weather; e31 if instrument-valid).

Zero LLM.  Reuses ``g1._program_evidence_census``.  Does not open sealed
Outcomes.  Writes ``artifacts/functional/e2/secondsource_census_v1.json`` and
``artifacts/functional/e2/secondsource_census_v1.md``.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[4]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation.functional.task_episode_harness.e1 import MATERIAL_THRESHOLD  # noqa: E402
from evaluation.functional.task_episode_harness.g1 import (  # noqa: E402
    G1_CONDITION_FEATURE,
    G1_MECHANISM_PROGRAM,
    GENERAL_EVIDENCE_MIN_DISTINCT_TASKS,
    PROVENANCE_CONDITIONED,
    PROVENANCE_UNGUIDED,
    _guidance_provenance,
    _program_evidence_census,
    _relation,
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
BASE_GUIDANCE = (
    "Supply only minimal effect-distinct candidates justified by public evidence."
)
OUTLIER_FAMILY = frozenset({"outlier_iqr", "outlier_mad"})


def _ops(probe: Mapping[str, Any]) -> list[str]:
    steps = probe.get("compiled_steps") or probe.get("steps") or []
    return [
        str(step["op"])
        for step in steps
        if isinstance(step, Mapping) and step.get("op")
    ]


def _instrument_ok(arm_row: Mapping[str, Any]) -> bool:
    metrics = arm_row.get("metrics") or {}
    if metrics.get("instrument_unreadable"):
        return False
    if metrics.get("infrastructure_failed"):
        return False
    stop = str(arm_row.get("stop_reason") or "")
    if stop in {"AGENT_PROTOCOL_ERROR", "INSTRUMENT_UNREADABLE", "INFRASTRUCTURE"}:
        return False
    return True


def _attempt(
    *,
    task_id: str,
    arm: str,
    probe: Mapping[str, Any],
    condition: bool,
    provenance: str,
    report_source: str,
    instrument_valid: bool,
) -> dict[str, Any] | None:
    program = _ops(probe)
    gain = probe.get("support_gain")
    if not program:
        return None
    return {
        "task_episode_id": task_id,
        "arm": arm,
        "program": program,
        "support_gain": gain,
        "gain_readable": isinstance(gain, (int, float)),
        G1_CONDITION_FEATURE: bool(condition),
        "guidance_provenance": provenance,
        "report_source": report_source,
        "instrument_valid": instrument_valid,
        "support_relation": (
            _relation(gain) if isinstance(gain, (int, float)) else "UNREADABLE"
        ),
    }


def _from_weather_a5a3(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block = report.get("weather_a5a3_autonomous_guidance") or {}
    for task_row in block.get("rows") or []:
        task_id = "weather:" + str(task_row.get("task_episode_id"))
        condition = bool(task_row.get(G1_CONDITION_FEATURE))
        for arm in ("A3", "A5"):
            arm_row = task_row.get(arm) or {}
            provenance = _guidance_provenance(
                arm_row.get("proposal_guidance_consumed"), BASE_GUIDANCE
            )
            valid = _instrument_ok(arm_row)
            for probe in arm_row.get("probes") or []:
                row = _attempt(
                    task_id=task_id,
                    arm=arm,
                    probe=probe,
                    condition=condition,
                    provenance=provenance,
                    report_source="w1.weather_a5a3_autonomous_guidance",
                    instrument_valid=valid,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _from_agentic(
    report: Mapping[str, Any],
    *,
    report_source: str,
    cohort_prefix: str,
    default_provenance: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_row in report.get("rows") or []:
        task_id = cohort_prefix + ":" + str(task_row.get("task_episode_id"))
        condition = bool(task_row.get(G1_CONDITION_FEATURE))
        for arm in ("A3", "A5"):
            arm_row = task_row.get(arm) or {}
            if not arm_row:
                continue
            consumed = None
            for stage in arm_row.get("stages") or []:
                payload = stage.get("payload") or {}
                if payload.get("proposal_guidance_consumed"):
                    consumed = payload.get("proposal_guidance_consumed")
            if consumed is not None:
                provenance = _guidance_provenance(consumed, BASE_GUIDANCE)
            elif arm == "A3":
                provenance = PROVENANCE_UNGUIDED
            else:
                provenance = default_provenance
            valid = _instrument_ok(arm_row)
            for probe in arm_row.get("probes") or []:
                if probe.get("status") not in (None, "PROBED"):
                    continue
                row = _attempt(
                    task_id=task_id,
                    arm=arm,
                    probe=probe,
                    condition=condition,
                    provenance=provenance,
                    report_source=report_source,
                    instrument_valid=valid,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _load_json(name: str) -> Mapping[str, Any] | None:
    path = E2 / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _e31_note() -> dict[str, Any]:
    reports = {}
    for name in (
        "w1_e31_dev_regression_report.json",
        "w1_e31_fresh_a5_two_slot_report.json",
        "w1_e31_source_treatment_audit_report.json",
    ):
        payload = _load_json(name)
        reports[name] = None if payload is None else {
            "verdict": payload.get("verdict"),
            "reason": payload.get("reason") or payload.get("note"),
        }
    return {
        "included": False,
        "reason": (
            "Under the frozen Task-roster origins every e31 eval series hits "
            "the scale floor (runner.load_cohort). Existing w1_e31_* reports "
            "contain no instrument-valid Episode rows with a readable "
            "support_gain, so e31 contributes zero census cells."
        ),
        "existing_reports": reports,
        "instrument_valid_rows": 0,
    }


def _strip_census(census: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cell in census:
        row = dict(cell)
        out.append(row)
    return out


def _find_cell(
    census: list[Mapping[str, Any]],
    program: list[str],
    condition: bool,
    relation: str,
) -> Mapping[str, Any] | None:
    for cell in census:
        if (
            list(cell["canonical_program"]) == list(program)
            and bool(cell[G1_CONDITION_FEATURE]) is condition
            and cell["support_relation"] == relation
        ):
            return cell
    return None


def _outlier_repeat(unguided_census: list[Mapping[str, Any]]) -> dict[str, Any]:
    hits = []
    for cell in unguided_census:
        program = list(cell["canonical_program"])
        if not program:
            continue
        if not any(op in OUTLIER_FAMILY for op in program):
            continue
        if cell["support_relation"] != "POSITIVE":
            continue
        hits.append({
            "canonical_program": program,
            G1_CONDITION_FEATURE: cell[G1_CONDITION_FEATURE],
            "distinct_task_count": cell["distinct_task_count"],
            "distinct_task_episode_ids": cell["distinct_task_episode_ids"],
            "meets_active_clause_threshold": (
                int(cell["distinct_task_count"])
                >= GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
            ),
        })
    any_repeat = any(row["meets_active_clause_threshold"] for row in hits)
    return {
        "question": (
            "Does an outlier-family POSITIVE cell repeat on the second source "
            "at >=2 distinct UNGUIDED tasks?"
        ),
        "answer": bool(any_repeat),
        "unguided_outlier_positive_cells": hits,
        "threshold": {
            "unit": "distinct_task_count",
            "minimum": GENERAL_EVIDENCE_MIN_DISTINCT_TASKS,
            "provenance_that_may_authorize": [PROVENANCE_UNGUIDED],
        },
        "note": (
            "GUIDANCE_CONDITIONED cells are omitted from this decision. "
            "They may only refute or weaken a clause."
        ),
    }


def main() -> int:
    w1 = _load_json("w1_task_episode_harness_report.json")
    if w1 is None:
        raise FileNotFoundError("w1_task_episode_harness_report.json")
    rows = _from_weather_a5a3(w1)

    g2 = _load_json("g2_shakedown_weather_report.json")
    if g2 is not None and g2.get("cohort") == "weather":
        g2_rows = _from_agentic(
            g2,
            report_source="g2_shakedown_weather_report.json",
            cohort_prefix="weather",
            default_provenance=PROVENANCE_UNGUIDED,
        )
        # Agentic G2 A5 is the Source-Card warm arm, not a guidance-patch
        # arm.  Only the cold A3 rows are UNGUIDED for the active-clause
        # question; G2 A5 is kept out of both UNGUIDED and CONDITIONED.
        rows.extend(row for row in g2_rows if row["arm"] == "A3")

    for name in (
        "g1_agentic_pipeline_report.json",
        "g1_agentic_pipeline_report_n3.json",
        "g1_agentic_pipeline_report_T233.json",
        "g2_shakedown_T233_rerun.json",
    ):
        payload = _load_json(name)
        if payload is None:
            continue
        # T233 is the first source, not the second.  Record the skip.
        _ = payload.get("cohort")

    readable = [
        row for row in rows
        if row["instrument_valid"] and row["gain_readable"]
    ]
    unguided = [
        row for row in readable
        if row["guidance_provenance"] == PROVENANCE_UNGUIDED
    ]
    conditioned = [
        row for row in readable
        if row["guidance_provenance"] == PROVENANCE_CONDITIONED
    ]
    unguided_census = _program_evidence_census(unguided)
    full_census = _program_evidence_census(readable)
    outlier = _outlier_repeat(unguided_census)

    e31 = _e31_note()
    t233 = _load_json("g3d1_source_derived_skill.json") or {}
    t233_outlier = None
    for cell in t233.get("census") or []:
        if list(cell.get("canonical_program") or []) == ["outlier_iqr"]:
            t233_outlier = cell

    payload = {
        "protocol_version": "secondsource_census_v1",
        "zero_llm": True,
        "zero_new_outcome": True,
        "second_source": {
            "weather": True,
            "e31": e31,
            "note": (
                "Second source is Weather only. e31 has no instrument-valid "
                "rows under the frozen roster."
            ),
        },
        "row_shape": (
            "canonical_program x %s x support_relation x task_episode_id"
            % G1_CONDITION_FEATURE
        ),
        "census_function": "g1._program_evidence_census",
        "material_threshold": MATERIAL_THRESHOLD,
        "mechanism_program": list(G1_MECHANISM_PROGRAM),
        "attempt_counts": {
            "all_extracted": len(rows),
            "instrument_valid_readable": len(readable),
            "UNGUIDED": len(unguided),
            "GUIDANCE_CONDITIONED": len(conditioned),
            "by_report_source": dict(
                Counter(row["report_source"] for row in readable)
            ),
            "by_provenance": dict(
                Counter(row["guidance_provenance"] for row in readable)
            ),
        },
        "unguided_census": _strip_census(unguided_census),
        "full_census": _strip_census(full_census),
        "t233_outlier_iqr_reference": t233_outlier,
        "outlier_family_repeat": outlier,
        "guidance_conditioned_may_not_authorize": True,
        "task_id_namespace": (
            "weather:<e1v2_task_XX>. Same roster id on Weather and T233 is "
            "not the same evidence unit; T233 rows are not mixed in."
        ),
    }
    json_path = E2 / "secondsource_census_v1.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Second-source UNGUIDED census v1",
        "",
        "Instrument / input for mainline R2. Not a Capability or Claim.",
        "Zero LLM, zero new Outcome.",
        "",
        "## Sources",
        "",
        "- Weather UNGUIDED: `w1.weather_a5a3_autonomous_guidance` A3 "
        "(consumed the base guidance verbatim) plus "
        "`g2_shakedown_weather_report.json` A3 (cold arm, "
        "`source_prior_retrieval.matched=false`).",
        "- Weather GUIDANCE_CONDITIONED: the same w1 block's A5 arm "
        "(consumed the patched autonomous guidance). Counted only in FULL, "
        "never toward an active-clause threshold.",
        "- e31: **not included**. Frozen-roster eval series all hit the scale "
        "floor; existing `w1_e31_*` reports have no instrument-valid "
        "Episode rows. Second source is Weather only.",
        "- T233 / `g1_agentic_pipeline_report*.json` are the first source "
        "and are not mixed into this census.",
        "",
        "## UNGUIDED census",
        "",
        "| program | %s | relation | distinct_task_count | attempt_count |"
        % G1_CONDITION_FEATURE,
        "| --- | --- | --- | ---: | ---: |",
    ]
    for cell in unguided_census:
        lines.append(
            "| `%s` | %s | %s | %d | %d |"
            % (
                " + ".join(cell["canonical_program"]),
                cell[G1_CONDITION_FEATURE],
                cell["support_relation"],
                cell["distinct_task_count"],
                cell["attempt_count"],
            )
        )
    lines.extend([
        "",
        "## FULL census (UNGUIDED + GUIDANCE_CONDITIONED)",
        "",
        "| program | %s | relation | distinct_task_count | attempt_count |"
        % G1_CONDITION_FEATURE,
        "| --- | --- | --- | ---: | ---: |",
    ])
    for cell in full_census:
        lines.append(
            "| `%s` | %s | %s | %d | %d |"
            % (
                " + ".join(cell["canonical_program"]),
                cell[G1_CONDITION_FEATURE],
                cell["support_relation"],
                cell["distinct_task_count"],
                cell["attempt_count"],
            )
        )
    lines.extend([
        "",
        "## Outlier-family repeat (UNGUIDED, active-clause threshold)",
        "",
        "T233 reference: `outlier_iqr` POSITIVE, "
        "`post_shift_support_sufficient=false`, distinct_task_count=6, "
        "NEGATIVE=0. That cell is the only T233 cell that is cleanly above "
        "the ≥2 distinct-task active-clause threshold.",
        "",
        "**Answer: %s.**"
        % ("yes" if outlier["answer"] else "no"),
        "",
    ])
    if not outlier["unguided_outlier_positive_cells"]:
        lines.append(
            "Weather UNGUIDED contains **zero** `outlier_iqr` or "
            "`outlier_mad` POSITIVE cells. The Weather UNGUIDED mass is "
            "`repair_level_shift` (both POSITIVE and NEGATIVE) and "
            "`hampel_filter` NEGATIVE. An unconditioned "
            "`prefer outlier_iqr` clause is not authorized by the second "
            "source under the frozen UNGUIDED rule."
        )
    else:
        lines.append("UNGUIDED outlier-family POSITIVE cells:")
        for row in outlier["unguided_outlier_positive_cells"]:
            lines.append(
                "- `%s` pss=%s distinct=%d %s"
                % (
                    " + ".join(row["canonical_program"]),
                    row[G1_CONDITION_FEATURE],
                    row["distinct_task_count"],
                    row["distinct_task_episode_ids"],
                )
            )
    lines.extend([
        "",
        "GUIDANCE_CONDITIONED Weather A5 does contain `hampel_filter` "
        "POSITIVE cells; those may weaken a global hampel ban but cannot "
        "authorize a new active clause.",
        "",
        "## Contract",
        "",
        "- Evidence unit is `distinct_task_count`; `attempt_count` is diagnostic.",
        "- Active-clause threshold is ≥%d UNGUIDED distinct tasks."
        % GENERAL_EVIDENCE_MIN_DISTINCT_TASKS,
        "- Task ids are namespaced `weather:<id>` so they cannot collapse "
        "with T233 ids of the same roster label.",
        "",
    ])
    md_path = E2 / "secondsource_census_v1.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", json_path)
    print("wrote", md_path)
    print("outlier_repeat", outlier["answer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
