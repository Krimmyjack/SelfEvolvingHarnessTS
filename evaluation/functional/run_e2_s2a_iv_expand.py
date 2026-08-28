"""S2a iv expand: metr_la / nn5_daily recut + Part 1 rerun.

0 LLM. Pre-declared from on-disk clean_base, not from readings.
metr_la official 1024 origins = sealed R1/R2 792/888.
nn5 on-disk n < CELL_WIDTH is a structural skip, reported not hunted.
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
import run_e2_s2a_forecast_oracle as traffic  # noqa: E402
import run_e2_s2a_iv_decomposition as iv  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    classify_relation,
)

REGISTRY = (PROJECT_ROOT / "artifacts" / "frozen" / "benchmark_v02"
            / "series_registry.jsonl")
CLEAN_BASE = PROJECT_ROOT / "data" / "benchmark_v0_2" / "clean_base"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s2a_oracle"
OUT_JSON = E2 / "s2a_iv_decomposition.json"
OUT_MD = E2 / "s2a_iv_decomposition.md"
COURSE_JSON = E2 / "s2a_course_frozen.json"
COURSE_MD = E2 / "s2a_course_frozen.md"
QUAL_JSON = E2 / "s2a_host_ready.json"
QUAL_MD = E2 / "s2a_host_ready.md"

METR_HELDIN = 792   # run_v1_sealed_a5_a3.py:74 R1_ORIGIN
METR_HELDOUT = 888  # run_v1_sealed_a5_a3.py:76 R2_ORIGIN
MENU = traffic.MENU


def _registry_rows(dataset_id: str) -> list[dict[str, Any]]:
    rows = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset_id") == dataset_id:
            rows.append(row)
    return rows


def _clean_index(needed: set[str]) -> dict[str, Path]:
    found = {}
    for rec in CLEAN_BASE.glob("*/record.json"):
        uid = json.loads(rec.read_text(encoding="utf-8")).get("series_uid")
        if uid in needed and (rec.parent / "values.npy").is_file():
            found[str(uid)] = rec.parent
    return found


def _load_dataset(dataset_id: str, min_length: int
                  ) -> tuple[list[str], dict[str, np.ndarray], dict[str, Any]]:
    rows = _registry_rows(dataset_id)
    needed = {str(r["series_uid"]) for r in rows}
    index = _clean_index(needed)
    names = []
    pool = {}
    for row in rows:
        uid = str(row["series_uid"])
        if uid not in index:
            continue
        series = np.asarray(np.load(index[uid] / "values.npy",
                                    allow_pickle=False), dtype=np.float64)
        if series.size < min_length:
            continue
        names.append(uid)
        pool[uid] = series
    meta = {
        "dataset_id": dataset_id,
        "registry_n": len(rows),
        "on_disk_n": len(index),
        "usable_n": len(names),
        "min_length": min_length,
    }
    return names, pool, meta


def _recut(dataset: str, names: Sequence[str], n_cells: int,
           family: str = "impulsive_outlier") -> list[dict[str, Any]]:
    cells = []
    cursor = 0
    for index in range(n_cells):
        chunk = list(names[cursor:cursor + traffic.CELL_WIDTH])
        cursor += traffic.CELL_WIDTH
        train = chunk[:traffic.N_TRAIN]
        cells.append({
            "unit_id": "%s_%s_%02d" % (dataset.replace(":", "_").replace("/", "_"),
                                       family, index),
            "dataset": dataset,
            "family": family,
            "train": train,
            "support": train[:traffic.N_FACE],
            "delayed": train[traffic.N_FACE:],
            "heldout": chunk[traffic.N_TRAIN:],
            "n_train": len(train),
            "n_half": traffic.N_FACE,
            "material_line": traffic._material_line(traffic.N_FACE),
            "two_x_line": 2.0 * traffic._material_line(traffic.N_FACE),
        })
    return cells


def _score_custom(values, train_uids, eval_uids, compiled, origin):
    return iv._score_face_safe(values, train_uids, eval_uids, compiled, origin)


def _oracle_and_decompose(cell: Mapping[str, Any],
                          pool: Mapping[str, np.ndarray],
                          heldin: int, heldout: int
                          ) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    injected = traffic._inject(cell, pool)
    fits = 0
    id_support = _score_custom(injected, cell["delayed"], cell["support"],
                               None, heldin)
    id_delayed = _score_custom(injected, cell["support"], cell["delayed"],
                               None, heldin)
    fits += 2
    program_rows = []
    iv_rows = []
    for op in MENU:
        compiled = traffic._compiled(op)
        if op == "identity":
            support, delayed = id_support, id_delayed
        else:
            support = _score_custom(injected, cell["delayed"], cell["support"],
                                    compiled, heldin)
            delayed = _score_custom(injected, cell["support"], cell["delayed"],
                                    compiled, heldin)
            fits += 2
        if (id_support["instrument"] != "ok" or support["instrument"] != "ok"
                or delayed["instrument"] != "ok"):
            program_rows.append({
                "program": op, "learnable": False, "two_x": False,
                "instrument": "scale_floor",
            })
            continue
        support_gains = iv._per_series_gains(id_support, support)
        delayed_gains = iv._per_series_gains(id_delayed, delayed)
        support_agg = float(np.mean(list(support_gains.values())))
        delayed_agg = float(np.mean(list(delayed_gains.values())))
        pooled = min(support_agg, delayed_agg)
        support_rel = classify_relation(
            aggregate_gain=support_agg, is_identity=(op == "identity"),
            consumer_id=traffic.CONSUMER_ID,
            material_threshold=float(cell["material_line"]))
        delayed_rel = classify_relation(
            aggregate_gain=delayed_agg, is_identity=(op == "identity"),
            consumer_id=traffic.CONSUMER_ID,
            material_threshold=float(cell["material_line"]))
        learnable = (
            op != "identity"
            and support_rel["relation"] == "POSITIVE"
            and delayed_rel["relation"] == "POSITIVE"
        )
        program_rows.append({
            "program": op,
            "support_gain": support_agg,
            "delayed_gain": delayed_agg,
            "heldin_headroom": pooled,
            "learnable": learnable,
            "two_x": bool(learnable and pooled >= float(cell["two_x_line"])),
        })
        if op != "identity":
            all_gains = {**{("support:" + k): v for k, v in support_gains.items()},
                         **{("delayed:" + k): v for k, v in delayed_gains.items()}}
            facts = classify_relation(
                aggregate_gain=pooled, per_series_gains=all_gains,
                consumer_id=traffic.CONSUMER_ID, material_threshold=iv.HARM_BAR)
            first_uid = list(cell["support"])[0]
            view = iv._pattern_view(
                np.asarray(injected[first_uid][:heldin], dtype=np.float64))
            iv_rows.append({
                "unit_id": cell["unit_id"],
                "dataset": cell["dataset"],
                "program": op,
                "pooled_gain": pooled,
                "pooled_positive": bool(pooled >= iv.HARM_BAR),
                "n_harmed": int(facts["harmed_series_count"]),
                "harmed_series": list(facts["harmed_series"]),
                "min_per_series_gain": facts["min_per_series_gain"],
                "relation": facts["relation"],
                "classification_basis": facts["classification_basis"],
                "pattern_view": view,
                "per_series_gains": {k: float(v) for k, v in all_gains.items()},
                "two_x": bool(learnable and pooled >= float(cell["two_x_line"])),
                "heldin_headroom_g0": pooled,
                "origins": {"heldin": heldin, "heldout": heldout},
            })
    learnable = [r for r in program_rows if r.get("learnable")]
    if learnable:
        best = max(learnable, key=lambda r: r["heldin_headroom"])
        oracle = {
            **cell,
            "oracle_program": best["program"],
            "learnability": "LEARNABLE",
            "heldin_headroom": best["heldin_headroom"],
            "two_x": bool(best["two_x"]),
            "programs": program_rows,
            "origins": {"heldin": heldin, "heldout": heldout},
            "banner": traffic.ORACLE_BANNER,
        }
    else:
        oracle = {
            **cell,
            "oracle_program": "identity",
            "learnability": "IDENTITY",
            "heldin_headroom": 0.0,
            "two_x": False,
            "programs": program_rows,
            "origins": {"heldin": heldin, "heldout": heldout},
            "banner": traffic.ORACLE_BANNER,
        }
    for row in iv_rows:
        row["oracle_program"] = oracle["oracle_program"]
        row["two_x"] = bool(oracle["two_x"]) if row["program"] == oracle["oracle_program"] else row["two_x"]
    return oracle, iv_rows, fits


def run() -> int:
    started = time.time()
    prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    metr_names, metr_pool, metr_meta = _load_dataset(
        "metr_la", METR_HELDOUT + 48)
    nn5_names, nn5_pool, nn5_meta = _load_dataset(
        "monash:nn5_daily", 600 + 48)
    metr_cells = _recut("metr_la", metr_names,
                        len(metr_names) // traffic.CELL_WIDTH)
    nn5_n = len(nn5_names) // traffic.CELL_WIDTH
    nn5_note = (
        "nn5 on-disk usable=%d < CELL_WIDTH=%d; structural skip"
        % (nn5_meta["usable_n"], traffic.CELL_WIDTH)
        if nn5_n == 0 else "nn5 cells cut"
    )
    s1.bind_curriculum_identity(
        task_kind=traffic.TASK_KIND,
        consumer_id=traffic.CONSUMER_ID,
        metric=traffic.METRIC)
    fits = 0
    new_oracles = []
    new_iv = []
    try:
        for cell in metr_cells:
            oracle, rows, used = _oracle_and_decompose(
                cell, metr_pool, METR_HELDIN, METR_HELDOUT)
            fits += used
            path = ORACLE_DIR / ("%s.json" % cell["unit_id"])
            path.write_text(json.dumps(oracle, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            new_oracles.append(oracle)
            new_iv.extend(rows)
            print("EXPAND %s %s headroom=%s two_x=%s" % (
                cell["unit_id"], oracle["learnability"],
                oracle["heldin_headroom"], oracle["two_x"]), flush=True)
    finally:
        s1.bind_curriculum_identity()

    # Candidate programs = oracle dedup of all two_x cells (old + new).
    old_two_x = []
    for rows in (prior.get("cells") or {}).values():
        old_two_x.extend(r for r in rows if r.get("two_x")
                         and r.get("program") == r.get("oracle_program"))
    new_two_x_oracles = [o for o in new_oracles if o.get("two_x")]
    programs = sorted({
        *[r["oracle_program"] for r in old_two_x],
        *[o["oracle_program"] for o in new_two_x_oracles],
    })
    s1.bind_curriculum_identity(
        task_kind=traffic.TASK_KIND,
        consumer_id=traffic.CONSUMER_ID,
        metric=traffic.METRIC)
    try:
        table = list(prior.get("four_conjunction_table") or [])
        eligible = []
        by_program = dict(prior.get("cells") or {})
        for program in programs:
            rows_for_p = [r for r in new_iv if r["program"] == program]
            if not rows_for_p:
                continue
            producer_candidates = [
                r for r in (by_program.get(program) or []) + rows_for_p
                if r.get("two_x") or r.get("oracle_program") == program
            ]
            if not producer_candidates:
                producer_candidates = rows_for_p
            producer = max(
                producer_candidates,
                key=lambda r: (float(r.get("heldin_headroom_g0") or 0.0),
                               r["unit_id"]))
            scope = iv._scope_of(producer["pattern_view"], program)
            for row in rows_for_p:
                verdict = s1._scope_v1_admits(scope, row["pattern_view"])
                row["scope"] = scope
                row.update(iv._four_way(row, verdict, program))
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
                    "expand": True,
                })
            by_program.setdefault(program, []).extend(rows_for_p)
            if any(r["four_conjunction"] for r in by_program[program]):
                eligible.append(program)
    finally:
        s1.bind_curriculum_identity()

    assembly = None
    status = "S2A_CONFLICT_FIELD_UNAVAILABLE"
    if eligible:
        def _p_key(p: str) -> tuple:
            rows = by_program[p]
            top = max(rows, key=lambda r: (float(r.get("heldin_headroom_g0") or 0),
                                           r["unit_id"]))
            return (float(top.get("heldin_headroom_g0") or 0), p)
        chosen = max(eligible, key=_p_key)
        t_names, _t_pool = traffic._load_pool()
        assembly = iv._assemble(chosen, by_program[chosen], t_names)
        if assembly["ready"]:
            status = "S2_HOST_READY"

    elapsed = round(time.time() - started, 1)
    prior_fits = int(prior.get("fits_total") or 224)
    payload = {
        "protocol": "s2a_iv_decomposition_v1_expand",
        "status": status,
        "prior_status": prior.get("status"),
        "semantics": iv.SEMANTICS,
        "harm_bar": iv.HARM_BAR,
        "expand": {
            "metr_la": {**metr_meta, "n_cells": len(metr_cells),
                        "origins": {"heldin": METR_HELDIN, "heldout": METR_HELDOUT},
                        "origin_cite": "run_v1_sealed_a5_a3.py:74-76"},
            "nn5": {**nn5_meta, "n_cells": nn5_n, "note": nn5_note},
        },
        "candidate_programs": programs,
        "eligible_programs": eligible,
        "assembly": assembly,
        "four_conjunction_table": table,
        "new_oracles": [{k: o.get(k) for k in
                         ("unit_id", "learnability", "oracle_program",
                          "heldin_headroom", "two_x")} for o in new_oracles],
        "fits_this_expand": fits,
        "fits_prior": prior_fits,
        "fits_total": prior_fits + fits,
        "llm": 0,
        "elapsed_s": elapsed,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    lines = [
        "# S2a iv decomposition (after metr_la/nn5 expand)",
        "",
        "**status: %s**" % status,
        "",
        "metr_la registry=%d on_disk=%d usable=%d cells=%d origins=792/888"
        % (metr_meta["registry_n"], metr_meta["on_disk_n"],
           metr_meta["usable_n"], len(metr_cells)),
        nn5_note,
        "candidate programs: %s" % (", ".join(programs) or "none"),
        "eligible: %s" % (", ".join(eligible) or "none"),
        "fits expand=%d prior=%d total=%d elapsed_s=%s"
        % (fits, prior_fits, prior_fits + fits, elapsed),
        "",
        "## Four-conjunction table (union)",
        "",
        "| P | cell | scope | same_P | pooled+ | harm | 4AND | relation | pooled | n_harm | min |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in table:
        pg = "n/a" if row["pooled_gain"] is None else "%.4f" % float(row["pooled_gain"])
        mn = ("n/a" if row["min_per_series_gain"] is None
              else "%.4f" % float(row["min_per_series_gain"]))
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
            "protocol": payload["protocol"],
            "selected_program": assembly["selected_program"],
            "course": assembly["course"],
            "clean_cell": assembly["clean_cell"],
            "post_hoc_rejudge": assembly["post_hoc_rejudge"],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        COURSE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        QUAL_JSON.write_text(json.dumps({
            "S2_HOST_READY": True, "status": status,
            "course": assembly["course"],
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
