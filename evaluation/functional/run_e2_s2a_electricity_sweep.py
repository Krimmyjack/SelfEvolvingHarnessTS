"""S2a G0 second-source sweep: UCI/TSL electricity, isomorphic to traffic.

0 LLM. Mechanical recut once; no add/drop from readings.
In-service file is TSL electricity.csv (321 numeric channels, UCI family).
Registry 370 x 1024 npy is not the live electricity loader and cannot host
traffic-isomorphic origins 1104/1800. Cell width 60 => at most 5 complete
impulse cells; leftover 21 unused. Gap guard reuses traffic_gap_00.
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

import run_e2_s2a_forecast_oracle as traffic  # noqa: E402
from evaluation.functional.task_episode_harness.agentic import g3_sourcing  # noqa: E402

PROTOCOL = "s2a_electricity_sweep_v1"
N_IMPULSE_CELLS = 5
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s2a_oracle"
SWEEP_JSON = E2 / "s2a_g0_electricity_sweep.json"
SWEEP_MD = E2 / "s2a_g0_electricity_sweep.md"
COURSE_JSON = E2 / "s2a_course_frozen.json"
COURSE_MD = E2 / "s2a_course_frozen.md"
QUAL_JSON = E2 / "s2a_host_ready.json"
QUAL_MD = E2 / "s2a_host_ready.md"
TRAFFIC_USED = 105


def _electricity_csv_path() -> Path:
    candidates = (
        Path(r"C:/Users/辉/desktop/agent/shared_tsq_datasets")
        / "electricity/electricity.csv",
        Path("/mnt/c/Users/辉/desktop/agent/shared_tsq_datasets/electricity/electricity.csv"),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("electricity.csv not found under shared_tsq_datasets")


def _load_pool() -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    names, values = g3_sourcing.load_csv_columns(
        _electricity_csv_path(), max_columns=400, max_rows=2500)
    usable = []
    for name in names:
        series = np.asarray(values[name], dtype=np.float64)
        if series.size < traffic.ORIGIN_HELDOUT + 48:
            continue
        usable.append(name)
    meta = {
        "source_path": str(_electricity_csv_path()),
        "n_numeric_loaded": len(names),
        "n_usable_for_origins": len(usable),
        "series_length": int(np.asarray(values[names[0]]).size) if names else 0,
        "pool_note": (
            "TSL electricity.csv is the in-service UCI-family loader "
            "(321 numeric channels). Registry 370x1024 cannot host "
            "isomorphic origins 1104/1800. Pre-declared: 5 impulse cells "
            "x 60 = 300; leftover unused; gap reused from traffic."
        ),
    }
    need = traffic.CELL_WIDTH * N_IMPULSE_CELLS
    if len(usable) < need:
        raise RuntimeError(
            "electricity pool too small for the frozen recut: "
            "usable=%d need=%d" % (len(usable), need))
    return usable, {name: np.asarray(values[name], dtype=np.float64)
                    for name in usable}, meta


def _score_face_safe(values: Mapping[str, np.ndarray],
                     train_uids: Sequence[str],
                     eval_uids: Sequence[str],
                     compiled: Any,
                     origin: int) -> dict[str, Any]:
    """Same Consumer as traffic; scale-floor is instrument-invalid, not a recut."""
    try:
        row = traffic._score_face(values, train_uids, eval_uids, compiled, origin)
        row["instrument"] = "ok"
        return row
    except RuntimeError as exc:
        text = str(exc)
        if "scale floor" not in text:
            raise
        return {
            "mean_smase": float("nan"),
            "per_view_smase": [],
            "n_eval": len(eval_uids),
            "instrument": "scale_floor",
            "instrument_error": text,
        }


def _oracle_cell(cell: Mapping[str, Any],
                 injected: Mapping[str, np.ndarray]) -> dict[str, Any]:
    material = float(cell["material_line"])
    rows = []
    id_support = _score_face_safe(injected, cell["delayed"], cell["support"],
                                  None, traffic.ORIGIN_HELDIN)
    id_delayed = _score_face_safe(injected, cell["support"], cell["delayed"],
                                  None, traffic.ORIGIN_HELDIN)
    id_heldout = _score_face_safe(injected, cell["train"], cell["heldout"],
                                  None, traffic.ORIGIN_HELDOUT)
    identity_ok = all(face["instrument"] == "ok"
                      for face in (id_support, id_delayed, id_heldout))
    for op in traffic.MENU:
        compiled = traffic._compiled(op)
        if op == "identity":
            support, delayed, heldout = id_support, id_delayed, id_heldout
        else:
            support = _score_face_safe(injected, cell["delayed"], cell["support"],
                                       compiled, traffic.ORIGIN_HELDIN)
            delayed = _score_face_safe(injected, cell["support"], cell["delayed"],
                                       compiled, traffic.ORIGIN_HELDIN)
            heldout = _score_face_safe(injected, cell["train"], cell["heldout"],
                                       compiled, traffic.ORIGIN_HELDOUT)
        faces_ok = all(face["instrument"] == "ok"
                       for face in (support, delayed, heldout))
        if not identity_ok or not faces_ok:
            rows.append({
                "program": op,
                "support_gain": None,
                "delayed_gain": None,
                "heldout_gain": None,
                "support_relation": "INSTRUMENT_INVALID",
                "delayed_relation": "INSTRUMENT_INVALID",
                "heldin_headroom": None,
                "learnable": False,
                "two_x": False,
                "near_line": False,
                "instrument": "scale_floor",
                "identity_ok": identity_ok,
                "faces_ok": faces_ok,
            })
            continue
        support_gain = id_support["mean_smase"] - support["mean_smase"]
        delayed_gain = id_delayed["mean_smase"] - delayed["mean_smase"]
        heldout_gain = id_heldout["mean_smase"] - heldout["mean_smase"]
        facts = traffic.classify_relation(
            aggregate_gain=min(support_gain, delayed_gain),
            per_series_gains=None,
            is_identity=(op == "identity"),
            consumer_id=traffic.CONSUMER_ID,
            material_threshold=material,
        )
        support_rel = traffic.classify_relation(
            aggregate_gain=support_gain, is_identity=(op == "identity"),
            consumer_id=traffic.CONSUMER_ID, material_threshold=material)
        delayed_rel = traffic.classify_relation(
            aggregate_gain=delayed_gain, is_identity=(op == "identity"),
            consumer_id=traffic.CONSUMER_ID, material_threshold=material)
        learnable = (
            op != "identity"
            and support_rel["relation"] == "POSITIVE"
            and delayed_rel["relation"] == "POSITIVE"
        )
        headroom = min(support_gain, delayed_gain)
        rows.append({
            "program": op,
            "support_gain": support_gain,
            "delayed_gain": delayed_gain,
            "heldout_gain": heldout_gain,
            "support_relation": support_rel["relation"],
            "delayed_relation": delayed_rel["relation"],
            "heldin_headroom": headroom,
            "learnable": learnable,
            "two_x": bool(learnable and headroom >= float(cell["two_x_line"])),
            "near_line": bool(learnable and material <= headroom < float(cell["two_x_line"])),
            "instrument": "ok",
            "identity_smase": {
                "support": id_support["mean_smase"],
                "delayed": id_delayed["mean_smase"],
                "heldout": id_heldout["mean_smase"],
            },
            "candidate_smase": {
                "support": support["mean_smase"],
                "delayed": delayed["mean_smase"],
                "heldout": heldout["mean_smase"],
            },
            "classification_basis": facts["classification_basis"],
        })
    learnable_rows = [r for r in rows if r["learnable"]]
    if learnable_rows:
        best = max(learnable_rows, key=lambda r: r["heldin_headroom"])
        oracle_set = [best["program"]]
        primary = best
        learnability = "LEARNABLE"
    elif identity_ok:
        oracle_set = ["identity"]
        primary = next(r for r in rows if r["program"] == "identity")
        learnability = "IDENTITY"
    else:
        oracle_set = []
        primary = {"heldin_headroom": None, "two_x": False, "near_line": False}
        learnability = "INSTRUMENT_INVALID"
    return {
        **cell,
        "banner": traffic.ORACLE_BANNER,
        "condition": traffic.CONDITION,
        "task_kind": traffic.TASK_KIND,
        "consumer_id": traffic.CONSUMER_ID,
        "metric": traffic.METRIC,
        "origins": {"heldin": traffic.ORIGIN_HELDIN,
                    "heldout": traffic.ORIGIN_HELDOUT},
        "programs": rows,
        "oracle_set": oracle_set,
        "oracle_program": oracle_set[0] if oracle_set else None,
        "learnability": learnability,
        "heldin_headroom": primary.get("heldin_headroom"),
        "two_x": bool(primary.get("two_x")),
        "near_line": bool(primary.get("near_line")),
    }


def _recut(names: Sequence[str]):
    cells = []
    cursor = 0
    leftover = list(names[traffic.CELL_WIDTH * N_IMPULSE_CELLS:])
    for index in range(N_IMPULSE_CELLS):
        chunk = list(names[cursor:cursor + traffic.CELL_WIDTH])
        cursor += traffic.CELL_WIDTH
        train = chunk[:traffic.N_TRAIN]
        heldout = chunk[traffic.N_TRAIN:]
        support = train[:traffic.N_FACE]
        delayed = train[traffic.N_FACE:]
        cells.append({
            "unit_id": "electricity_impulsive_outlier_%02d" % index,
            "dataset": "uci_electricity_tsl321",
            "family": "impulsive_outlier",
            "train": train,
            "support": support,
            "delayed": delayed,
            "heldout": heldout,
            "n_train": len(train),
            "n_half": traffic.N_FACE,
            "material_line": traffic._material_line(traffic.N_FACE),
            "two_x_line": 2.0 * traffic._material_line(traffic.N_FACE),
        })
    return cells, leftover


def _load_traffic_oracles() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(ORACLE_DIR.glob("traffic_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _roles(oracles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    impulse = [o for o in oracles if o["family"] == "impulsive_outlier"]
    gap = [o for o in oracles if o["family"] == "gap"]
    producers = [o for o in impulse
                 if o["learnability"] == "LEARNABLE" and o["two_x"]]
    strong = list(producers)
    weak = [o for o in impulse
            if o["learnability"] == "LEARNABLE" and o["near_line"]]
    identity = [o for o in impulse if o["learnability"] == "IDENTITY"]
    return {
        "impulse": impulse,
        "gap": gap,
        "producers": producers,
        "strong": strong,
        "weak": weak,
        "identity": identity,
    }


def _headroom_row(o: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": o["unit_id"],
        "dataset": o.get("dataset"),
        "family": o["family"],
        "learnability": o["learnability"],
        "oracle_program": o.get("oracle_program"),
        "heldin_headroom": o.get("heldin_headroom"),
        "two_x": bool(o.get("two_x")),
        "near_line": bool(o.get("near_line")),
    }


def _assemble_course(merged: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    reasons = []
    if len(merged["producers"]) < 1:
        reasons.append("no_producer_learnable_with_2x_margin")
    if len(merged["strong"]) < 1:
        reasons.append("no_strong_beneficiary")
    if len(merged["weak"]) < 1:
        reasons.append("no_near_line_weak_beneficiary")
    if len(merged["identity"]) < 1:
        reasons.append("no_identity_field")
    if len(merged["gap"]) < 1:
        reasons.append("gap_guard_cell_missing")
    if reasons:
        return [], reasons
    producer = merged["producers"][0]
    strong_b = next(
        (o for o in merged["strong"] if o["unit_id"] != producer["unit_id"]),
        None)
    if strong_b is None:
        return [], reasons + ["strong_beneficiary_not_distinct_from_producer"]
    weak_b = merged["weak"][0]
    ident = merged["identity"][0]
    guard = merged["gap"][0]
    course = [
        {"role": "producer", "unit_id": producer["unit_id"]},
        {"role": "identity", "unit_id": ident["unit_id"]},
        {"role": "boundary_compile", "unit_id": producer["unit_id"],
         "note": "ladder v2 compile from producer strong positive; not a distinct cell"},
        {"role": "strong_beneficiary", "unit_id": strong_b["unit_id"]},
        {"role": "near_line_conflict", "unit_id": weak_b["unit_id"]},
        {"role": "conflict_reencounter", "unit_id": weak_b["unit_id"],
         "note": "mechanism probe; readout listed separately; not in first-line regret"},
        {"role": "gap_out_of_family_guard", "unit_id": guard["unit_id"]},
    ]
    return course, []


def _write_course(course: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "protocol": PROTOCOL,
        "condition": traffic.CONDITION,
        "task_kind": traffic.TASK_KIND,
        "consumer_id": traffic.CONSUMER_ID,
        "metric": traffic.METRIC,
        "material_line": traffic._material_line(traffic.N_FACE),
        "delta_material_note": (
            "sum of beneficiary-unit half-split material lines; filled after live"),
        "course": list(course),
        "order": [
            "producer", "identity", "boundary_compile", "strong_beneficiary",
            "near_line_conflict", "conflict_reencounter",
            "gap_out_of_family_guard",
        ],
        "oracle_isolation": traffic.ORACLE_BANNER,
        "gap_role_note": (
            "gap is Scope-mismatch out-of-family guard, not identity; "
            "LEARNABLE on the gap cell does not disqualify it"
        ),
    }
    COURSE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                           encoding="utf-8")


def run() -> int:
    started = time.time()
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    names, pool, meta = _load_pool()
    cells, leftover = _recut(names)
    oracles = []
    fits = 0
    for cell in cells:
        injected = traffic._inject(cell, pool)
        oracle = _oracle_cell(cell, injected)
        fits += len(traffic.MENU) * 3
        path = ORACLE_DIR / ("%s.json" % cell["unit_id"])
        path.write_text(json.dumps(oracle, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        oracles.append(oracle)
        print("ORACLE %s %s headroom=%s two_x=%s near=%s" % (
            cell["unit_id"], oracle["learnability"],
            oracle["heldin_headroom"], oracle["two_x"],
            oracle["near_line"]), flush=True)

    traffic_oracles = _load_traffic_oracles()
    elec_roles = _roles(oracles)
    traffic_roles = _roles(traffic_oracles)
    merged_oracles = list(traffic_oracles) + list(oracles)
    merged_roles = _roles(merged_oracles)
    course, reasons = _assemble_course(merged_roles)
    elec_has_weak = len(elec_roles["weak"]) >= 1
    elec_has_identity = len(elec_roles["identity"]) >= 1
    ready = not reasons
    if not ready and not elec_has_weak and not elec_has_identity:
        status = "S2_HOST_READY_FAIL_BOTH_SOURCES"
    elif not ready:
        status = "S2_HOST_READY_FAIL"
    else:
        status = "S2_HOST_READY"

    elapsed = round(time.time() - started, 1)
    payload = {
        "protocol": PROTOCOL,
        "status": status,
        "S2_HOST_READY": ready,
        "reasons": reasons,
        "pool": meta,
        "predeclared_cut": {
            "n_impulse_cells": N_IMPULSE_CELLS,
            "n_gap_cells_this_source": 0,
            "cell_width": traffic.CELL_WIDTH,
            "leftover_unused": leftover,
            "n_leftover": len(leftover),
            "gap_reuse": "traffic_gap_00",
            "origins": {"heldin": traffic.ORIGIN_HELDIN,
                        "heldout": traffic.ORIGIN_HELDOUT},
            "injection": "impulsive_outlier via injection.py unmodified",
        },
        "electricity": {
            "n_impulse": len(elec_roles["impulse"]),
            "n_producer": len(elec_roles["producers"]),
            "n_strong": len(elec_roles["strong"]),
            "n_weak": len(elec_roles["weak"]),
            "n_identity": len(elec_roles["identity"]),
            "cells": [_headroom_row(o) for o in oracles],
        },
        "traffic": {
            "n_impulse": len(traffic_roles["impulse"]),
            "n_producer": len(traffic_roles["producers"]),
            "n_strong": len(traffic_roles["strong"]),
            "n_weak": len(traffic_roles["weak"]),
            "n_identity": len(traffic_roles["identity"]),
            "n_gap": len(traffic_roles["gap"]),
            "cells": [_headroom_row(o) for o in traffic_oracles],
        },
        "merged": {
            "n_impulse": len(merged_roles["impulse"]),
            "n_producer": len(merged_roles["producers"]),
            "n_strong": len(merged_roles["strong"]),
            "n_weak": len(merged_roles["weak"]),
            "n_identity": len(merged_roles["identity"]),
            "n_gap": len(merged_roles["gap"]),
            "headroom_table": [_headroom_row(o) for o in merged_oracles],
            "course": course,
        },
        "part_a_green": True,
        "classification_146": True,
        "fits_this_sweep": fits,
        "fits_prior_traffic": TRAFFIC_USED,
        "fits_total": fits + TRAFFIC_USED,
        "llm": 0,
        "elapsed_s": elapsed,
    }
    SWEEP_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")

    lines = [
        "# S2a G0 electricity sweep",
        "",
        "**status: %s**" % status,
        "",
        "reasons: %s" % (", ".join(reasons) or "none"),
        "electricity impulse: %d  producer: %d  strong: %d  weak: %d  identity: %d"
        % (len(elec_roles["impulse"]), len(elec_roles["producers"]),
           len(elec_roles["strong"]), len(elec_roles["weak"]),
           len(elec_roles["identity"])),
        "merged impulse: %d  producer: %d  strong: %d  weak: %d  identity: %d  gap: %d"
        % (len(merged_roles["impulse"]), len(merged_roles["producers"]),
           len(merged_roles["strong"]), len(merged_roles["weak"]),
           len(merged_roles["identity"]), len(merged_roles["gap"])),
        "fits this sweep: %d  prior traffic: %d  total: %d  elapsed_s: %s"
        % (fits, TRAFFIC_USED, fits + TRAFFIC_USED, elapsed),
        "",
        "## Pre-declared cut",
        "",
        meta["pool_note"],
        "usable=%d leftover=%d gap_reuse=traffic_gap_00 origins=1104/1800"
        % (meta["n_usable_for_origins"], len(leftover)),
        "",
        "## Merged headroom table",
        "",
        "| unit | learnability | oracle | headroom | two_x | near_line |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in payload["merged"]["headroom_table"]:
        hr = row["heldin_headroom"]
        hr_s = "n/a" if hr is None else "%.4f" % float(hr)
        lines.append("| `%s` | %s | %s | %s | %s | %s |" % (
            row["unit_id"], row["learnability"], row["oracle_program"],
            hr_s, row["two_x"], row["near_line"]))
    lines += [
        "",
        "Oracle files live under `artifacts/functional/e2/s2a_oracle/` "
        "and must not enter any arm prompt, store, or retrieval view.",
        "",
    ]
    if course:
        lines.append("## Frozen course")
        for row in course:
            lines.append("- %s: `%s`" % (row["role"], row["unit_id"]))
        _write_course(course)
        host = {
            "protocol": PROTOCOL,
            "S2_HOST_READY": True,
            "reasons": [],
            "course": course,
            "source": "merged traffic + electricity",
            "fits_total": fits + TRAFFIC_USED,
            "elapsed_s": elapsed,
        }
        QUAL_JSON.write_text(json.dumps(host, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        QUAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        COURSE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        host = {
            "protocol": PROTOCOL,
            "S2_HOST_READY": False,
            "reasons": reasons,
            "status": status,
            "fits_total": fits + TRAFFIC_USED,
            "elapsed_s": elapsed,
        }
        QUAL_JSON.write_text(json.dumps(host, ensure_ascii=False, indent=1),
                             encoding="utf-8")
        QUAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        COURSE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    SWEEP_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status, flush=True)
    return 0 if ready else 1


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
