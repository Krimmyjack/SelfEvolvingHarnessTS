"""S2a iv: per-series CONFLICT decomposition + role assembly.

0 LLM. Candidate programs = oracle-program dedup of the 11 two_x cells.
CONFLICT / harm bar = in-service classify_relation + signed_radius M=0.005.
No add/drop from readings. Fit increment counts toward the 300 cap.
"""
from __future__ import annotations

import argparse
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

import numpy as np  # noqa: E402

import run_e2_s1_curriculum_four_arms as s1  # noqa: E402
import run_e2_s1a_curriculum_oracle_audit as s1a  # noqa: E402
import run_e2_s2a_electricity_sweep as elec  # noqa: E402
import run_e2_s2a_forecast_oracle as traffic  # noqa: E402
from contracts.observables import OBSERVABLE_FEATURES, observable_numeric_bin  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    CLASSIFICATION_MATERIAL_THRESHOLD,
    classify_relation,
)
from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: E402
    MATERIAL_THRESHOLD,
)
from SelfEvolvingHarnessTS.runtime.public_features import (  # noqa: E402
    extract_public_features,
)

PROTOCOL = "s2a_iv_decomposition_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s2a_oracle"
OUT_JSON = E2 / "s2a_iv_decomposition.json"
OUT_MD = E2 / "s2a_iv_decomposition.md"
COURSE_JSON = E2 / "s2a_course_frozen.json"
COURSE_MD = E2 / "s2a_course_frozen.md"
QUAL_JSON = E2 / "s2a_host_ready.json"
QUAL_MD = E2 / "s2a_host_ready.md"
PRIOR_FITS = 180
FIT_INCREMENT_CAP = 80
HARM_BAR = float(MATERIAL_THRESHOLD)
assert HARM_BAR == float(CLASSIFICATION_MATERIAL_THRESHOLD) == 0.005
SEMANTICS = {
    "classify_relation": (
        "SelfEvolvingHarnessTS/methods/ttha/experience_memory.py:411-471"
    ),
    "conflict_rule": (
        "experience_memory.py:398-444: agg >= +t and some per-series < -t "
        "-> CONFLICT; t defaults to CLASSIFICATION_MATERIAL_THRESHOLD"
    ),
    "material_threshold": (
        "signed_radius.py:40 MATERIAL_THRESHOLD=0.005; "
        "experience_memory.py:408 CLASSIFICATION_MATERIAL_THRESHOLD=0.005"
    ),
    "pattern_family_axis": (
        "forecast extractor runtime/public_features.py:265-331; "
        "binned by contracts.observables.observable_numeric_bin; "
        "pattern keys = s1a PATTERN_KEYS "
        "(run_e2_s1a_curriculum_oracle_audit.py:124-138)"
    ),
    "scope_five_axes": (
        "run_e2_s1_curriculum_four_arms.py:_scope_v1_admits:1713-1741 "
        "and _scope_v1_of:1648-1682"
    ),
}


def _impulse_oracles() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(ORACLE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("family") == "impulsive_outlier":
            rows.append(payload)
    return rows


def _score_face_safe(values, train_uids, eval_uids, compiled, origin):
    try:
        row = traffic._score_face(values, train_uids, eval_uids, compiled, origin)
        row["instrument"] = "ok"
        row["eval_uids"] = list(eval_uids)
        return row
    except RuntimeError as exc:
        if "scale floor" not in str(exc):
            raise
        return {
            "mean_smase": float("nan"),
            "per_view_smase": [],
            "n_eval": len(eval_uids),
            "eval_uids": list(eval_uids),
            "instrument": "scale_floor",
            "instrument_error": str(exc),
        }


def _per_series_gains(identity_face, candidate_face) -> dict[str, float]:
    if identity_face["instrument"] != "ok" or candidate_face["instrument"] != "ok":
        return {}
    uids = list(identity_face["eval_uids"])
    id_smase = list(identity_face["per_view_smase"])
    cand_smase = list(candidate_face["per_view_smase"])
    if len(uids) != len(id_smase) or len(uids) != len(cand_smase):
        raise RuntimeError("per-view sMASE not aligned with eval uids")
    return {uid: float(id_v) - float(cand_v)
            for uid, id_v, cand_v in zip(uids, id_smase, cand_smase)}


def _pattern_view(series: np.ndarray) -> dict[str, Any]:
    raw = dict(extract_public_features(series, task_kind="forecast").mapping)
    binned: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in OBSERVABLE_FEATURES:
            continue
        kind = OBSERVABLE_FEATURES[key]
        if (kind == "number" and isinstance(value, (int, float))
                and not isinstance(value, bool)):
            binned[key] = observable_numeric_bin(key, float(value))
        else:
            binned[key] = value
    return {key: binned[key] for key in s1a.PATTERN_KEYS if key in binned}


def _load_pools() -> dict[str, dict[str, np.ndarray]]:
    t_names, t_pool = traffic._load_pool()
    e_names, e_pool, _meta = elec._load_pool()
    return {
        "monash:traffic_hourly": {n: t_pool[n] for n in t_names},
        "uci_electricity_tsl321": {n: e_pool[n] for n in e_names},
        "_traffic_names": list(t_names),
    }


def _decompose_cell(cell: Mapping[str, Any], program: str,
                    pool: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], int]:
    injected = traffic._inject(cell, pool)
    fits = 0
    compiled = traffic._compiled(program)
    id_support = _score_face_safe(
        injected, cell["delayed"], cell["support"], None, traffic.ORIGIN_HELDIN)
    id_delayed = _score_face_safe(
        injected, cell["support"], cell["delayed"], None, traffic.ORIGIN_HELDIN)
    cand_support = _score_face_safe(
        injected, cell["delayed"], cell["support"], compiled, traffic.ORIGIN_HELDIN)
    cand_delayed = _score_face_safe(
        injected, cell["support"], cell["delayed"], compiled, traffic.ORIGIN_HELDIN)
    fits += 4
    support_gains = _per_series_gains(id_support, cand_support)
    delayed_gains = _per_series_gains(id_delayed, cand_delayed)
    all_gains = {**{("support:" + k): v for k, v in support_gains.items()},
                 **{("delayed:" + k): v for k, v in delayed_gains.items()}}
    support_agg = (None if not support_gains
                   else float(np.mean(list(support_gains.values()))))
    delayed_agg = (None if not delayed_gains
                   else float(np.mean(list(delayed_gains.values()))))
    if support_agg is None or delayed_agg is None:
        pooled = None
    else:
        pooled = min(support_agg, delayed_agg)
    facts = classify_relation(
        aggregate_gain=pooled,
        per_series_gains=all_gains or None,
        is_identity=False,
        consumer_id=traffic.CONSUMER_ID,
        material_threshold=HARM_BAR,
    )
    support_facts = classify_relation(
        aggregate_gain=support_agg, per_series_gains=support_gains or None,
        consumer_id=traffic.CONSUMER_ID, material_threshold=HARM_BAR)
    delayed_facts = classify_relation(
        aggregate_gain=delayed_agg, per_series_gains=delayed_gains or None,
        consumer_id=traffic.CONSUMER_ID, material_threshold=HARM_BAR)
    first_uid = list(cell["support"])[0]
    view = _pattern_view(np.asarray(injected[first_uid][:traffic.ORIGIN_HELDIN],
                                    dtype=np.float64))
    return {
        "unit_id": cell["unit_id"],
        "dataset": cell.get("dataset"),
        "program": program,
        "instrument": "ok" if all_gains else "scale_floor",
        "support_aggregate": support_agg,
        "delayed_aggregate": delayed_agg,
        "pooled_gain": pooled,
        "pooled_positive": bool(pooled is not None and pooled >= HARM_BAR),
        "n_harmed": int(facts["harmed_series_count"]),
        "harmed_series": list(facts["harmed_series"]),
        "min_per_series_gain": facts["min_per_series_gain"],
        "relation": facts["relation"],
        "classification_basis": facts["classification_basis"],
        "support_relation": support_facts["relation"],
        "delayed_relation": delayed_facts["relation"],
        "per_series_gains": {k: float(v) for k, v in all_gains.items()},
        "pattern_view": view,
        "oracle_program": cell.get("oracle_program"),
        "heldin_headroom_g0": cell.get("heldin_headroom"),
        "two_x": bool(cell.get("two_x")),
    }, fits


def _scope_of(producer_view: Mapping[str, Any], program: str) -> dict[str, Any]:
    return {
        "task_kind": traffic.TASK_KIND,
        "consumer_id": traffic.CONSUMER_ID,
        "metric": traffic.METRIC,
        "pattern_intersection": dict(producer_view),
        "program_geometry": [program],
        "supporting_units": [],
    }


def _four_way(row: Mapping[str, Any], scope_verdict: Mapping[str, Any],
              program: str) -> dict[str, Any]:
    same_program = row["program"] == program
    scope_match = bool(scope_verdict.get("admits"))
    pooled_pos = bool(row.get("pooled_positive"))
    harm = int(row.get("n_harmed") or 0) >= 1
    hit = bool(scope_match and same_program and pooled_pos and harm)
    return {
        "scope_match": scope_match,
        "scope_why": scope_verdict.get("why"),
        "same_program": same_program,
        "pooled_positive": pooled_pos,
        "harm_over_bar": harm,
        "four_conjunction": hit,
    }


def _assemble(program: str, rows: Sequence[Mapping[str, Any]],
              traffic_names: Sequence[str]) -> dict[str, Any]:
    two_x = [r for r in rows if r.get("two_x")]
    producer = max(two_x, key=lambda r: (float(r["heldin_headroom_g0"]),
                                         r["unit_id"]))
    strong = [r for r in two_x if r["unit_id"] != producer["unit_id"]]
    if not strong:
        return {"ready": False, "reasons": ["no_distinct_strong_beneficiary"],
                "course": []}
    strong_b = max(strong, key=lambda r: (float(r["heldin_headroom_g0"]),
                                          r["unit_id"]))
    hits = [r for r in rows if r["four_conjunction"]]
    if not hits:
        return {"ready": False, "reasons": ["no_four_conjunction_hit"],
                "course": []}
    conflict = sorted(hits, key=lambda r: r["unit_id"])[0]
    leftover_start = traffic.CELL_WIDTH * (traffic.N_IMPULSE_CELLS + traffic.N_GAP_CELLS)
    clean_chunk = list(traffic_names[leftover_start:leftover_start + traffic.CELL_WIDTH])
    if len(clean_chunk) < traffic.CELL_WIDTH:
        return {"ready": False, "reasons": ["clean_identity_cell_pool_short"],
                "course": []}
    clean_id = "traffic_clean_identity_00"
    course = [
        {"role": "producer", "unit_id": producer["unit_id"],
         "program_family": program},
        {"role": "clean_identity", "unit_id": clean_id,
         "sol_name": "无缺陷条件下的 identity 场",
         "note": "clean condition cell; not a natural identity field"},
        {"role": "boundary_compile", "unit_id": producer["unit_id"],
         "note": "ladder v2 compile from producer strong positive; not a distinct cell"},
        {"role": "strong_beneficiary", "unit_id": strong_b["unit_id"]},
        {"role": "conflict", "unit_id": conflict["unit_id"],
         "program": program},
        {"role": "conflict_reencounter", "unit_id": conflict["unit_id"],
         "note": "mechanism probe; readout listed separately; not in first-line regret"},
        {"role": "gap_out_of_family_guard", "unit_id": "traffic_gap_00"},
    ]
    clean_cell = {
        "unit_id": clean_id,
        "dataset": "monash:traffic_hourly",
        "family": "clean",
        "sol_name": "无缺陷条件下的 identity 场",
        "train": clean_chunk[:traffic.N_TRAIN],
        "support": clean_chunk[:traffic.N_FACE],
        "delayed": clean_chunk[traffic.N_FACE:traffic.N_TRAIN],
        "heldout": clean_chunk[traffic.N_TRAIN:],
        "injection": "none",
    }
    return {
        "ready": True,
        "reasons": [],
        "selected_program": program,
        "producer_unit": producer["unit_id"],
        "conflict_unit": conflict["unit_id"],
        "strong_unit": strong_b["unit_id"],
        "course": course,
        "clean_cell": clean_cell,
        "post_hoc_rejudge": (
            "If A5 live learns a program Q != P, re-judge the assigned "
            "conflict cell's already-computed four-conjunction for Q. "
            "If Q was not decomposed in Part 1 or the conjunction fails, "
            "mark R2 untested this run. Do not reassign roles live."
        ),
    }


def run() -> int:
    started = time.time()
    oracles = _impulse_oracles()
    if len(oracles) != 11:
        raise RuntimeError("expected 11 impulse cells, got %d" % len(oracles))
    programs = sorted({str(o["oracle_program"]) for o in oracles})
    pools = _load_pools()
    s1.bind_curriculum_identity(
        task_kind=traffic.TASK_KIND,
        consumer_id=traffic.CONSUMER_ID,
        metric=traffic.METRIC)
    fits = 0
    by_program: dict[str, list[dict[str, Any]]] = {}
    try:
        for program in programs:
            rows = []
            for cell in oracles:
                pool = pools[cell["dataset"]]
                row, used = _decompose_cell(cell, program, pool)
                fits += used
                if fits > FIT_INCREMENT_CAP:
                    raise RuntimeError(
                        "fit increment %d exceeded cap %d" % (fits, FIT_INCREMENT_CAP))
                rows.append(row)
                print("IV %s @ %s rel=%s pooled=%s harmed=%s" % (
                    program, cell["unit_id"], row["relation"],
                    row["pooled_gain"], row["n_harmed"]), flush=True)
            producer = max(rows, key=lambda r: (float(r["heldin_headroom_g0"]),
                                                r["unit_id"]))
            scope = _scope_of(producer["pattern_view"], program)
            for row in rows:
                verdict = s1._scope_v1_admits(scope, row["pattern_view"])
                row["scope"] = scope
                row.update(_four_way(row, verdict, program))
            by_program[program] = rows
    finally:
        s1.bind_curriculum_identity()

    eligible = [p for p, rows in by_program.items()
                if any(r["four_conjunction"] for r in rows)]
    assembly = None
    status = "S2A_CONFLICT_FIELD_UNAVAILABLE"
    if eligible:
        def _p_key(p: str) -> tuple:
            rows = by_program[p]
            top = max(rows, key=lambda r: (float(r["heldin_headroom_g0"]),
                                           r["unit_id"]))
            return (float(top["heldin_headroom_g0"]), p)
        chosen = max(eligible, key=_p_key)
        assembly = _assemble(chosen, by_program[chosen],
                             pools["_traffic_names"])
        if assembly["ready"]:
            status = "S2_HOST_READY"
        else:
            status = "S2A_CONFLICT_FIELD_UNAVAILABLE"
    elapsed = round(time.time() - started, 1)
    table = []
    for program, rows in by_program.items():
        for row in rows:
            table.append({
                "program": program,
                "unit_id": row["unit_id"],
                "scope_match": row["scope_match"],
                "same_program": row["same_program"],
                "pooled_positive": row["pooled_positive"],
                "harm_over_bar": row["harm_over_bar"],
                "four_conjunction": row["four_conjunction"],
                "relation": row["relation"],
                "pooled_gain": row["pooled_gain"],
                "n_harmed": row["n_harmed"],
                "min_per_series_gain": row["min_per_series_gain"],
                "scope_why": row["scope_why"],
            })
    payload = {
        "protocol": PROTOCOL,
        "status": status,
        "semantics": SEMANTICS,
        "harm_bar": HARM_BAR,
        "candidate_programs": programs,
        "n_impulse_cells": len(oracles),
        "eligible_programs": eligible,
        "assembly": assembly,
        "four_conjunction_table": table,
        "cells": {p: rows for p, rows in by_program.items()},
        "fits_this_part": fits,
        "fits_prior": PRIOR_FITS,
        "fits_total": PRIOR_FITS + fits,
        "llm": 0,
        "elapsed_s": elapsed,
        "expand_needed": status != "S2_HOST_READY",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    lines = [
        "# S2a iv per-series decomposition",
        "",
        "**status: %s**" % status,
        "",
        "candidate programs: %s" % (", ".join(programs) or "none"),
        "eligible programs (four-conjunction hit): %s"
        % (", ".join(eligible) or "none"),
        "harm bar M=%s  fits this part=%d  prior=%d  total=%d  elapsed_s=%s"
        % (HARM_BAR, fits, PRIOR_FITS, PRIOR_FITS + fits, elapsed),
        "",
        "Semantics: classify_relation %s; threshold %s; pattern %s."
        % (SEMANTICS["classify_relation"], SEMANTICS["material_threshold"],
           SEMANTICS["pattern_family_axis"]),
        "",
        "## Four-conjunction table",
        "",
        "| P | cell | scope | same_P | pooled+ | harm | 4AND | relation | pooled | n_harm | min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in table:
        pg = "n/a" if row["pooled_gain"] is None else "%.4f" % row["pooled_gain"]
        mn = "n/a" if row["min_per_series_gain"] is None else "%.4f" % row["min_per_series_gain"]
        lines.append(
            "| `%s` | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (row["program"], row["unit_id"], row["scope_match"],
               row["same_program"], row["pooled_positive"], row["harm_over_bar"],
               row["four_conjunction"], row["relation"], pg, row["n_harmed"], mn))
    if assembly and assembly.get("course"):
        lines += ["", "## Frozen course", ""]
        for item in assembly["course"]:
            lines.append("- %s: `%s`" % (item["role"], item["unit_id"]))
        COURSE_JSON.write_text(json.dumps({
            "protocol": PROTOCOL,
            "condition": traffic.CONDITION,
            "task_kind": traffic.TASK_KIND,
            "consumer_id": traffic.CONSUMER_ID,
            "metric": traffic.METRIC,
            "material_line": traffic._material_line(traffic.N_FACE),
            "delta_material_note": (
                "sum of beneficiary-unit half-split material lines; filled after live"),
            "selected_program": assembly["selected_program"],
            "course": assembly["course"],
            "clean_cell": assembly["clean_cell"],
            "post_hoc_rejudge": assembly["post_hoc_rejudge"],
            "oracle_isolation": traffic.ORACLE_BANNER,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        COURSE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        QUAL_JSON.write_text(json.dumps({
            "protocol": PROTOCOL,
            "S2_HOST_READY": True,
            "status": status,
            "course": assembly["course"],
            "selected_program": assembly["selected_program"],
            "conflict_unit": assembly["conflict_unit"],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        QUAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status, flush=True)
    return 0 if status == "S2_HOST_READY" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.run:
        return run()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
