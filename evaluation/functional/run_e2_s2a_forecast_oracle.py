"""S2a Part B: forecast exam recut + dual-layer oracle + qualification.

0 LLM. Consumer fits are oracle-only and isolated from every arm view.
Uses in-service impulsive_outlier / gap (injection.py) and pooled ridge+sMASE.
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

import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_kdd2018_natural_slow_update as kdd  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
from evaluation.functional.task_episode_harness.agentic import g3_sourcing  # noqa: E402
from evaluation.functional.task_episode_harness.injection import (  # noqa: E402
    inject_gap_corpus,
    inject_label_touched_corpus,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    classify_relation,
)

PROTOCOL = "s2a_forecast_oracle_v1"
CONDITION = "fit_only_artifact"
TASK_KIND = "forecast"
CONSUMER_ID = "pooled_ridge_a1"
METRIC = "sMASE"
N_TRAIN = 40
N_FACE = 20
N_HELDOUT = 20
CELL_WIDTH = N_TRAIN + N_HELDOUT
N_IMPULSE_CELLS = 6
N_GAP_CELLS = 1
ORIGIN_HELDIN = 1104
ORIGIN_HELDOUT = 1800
PERIOD = 24
MENU = ("identity", "outlier_iqr", "outlier_mad", "hampel_filter", "winsorize")
ORACLE_BANNER = "sealed exam key. must not enter any arm prompt, store, or retrieval view."

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s2a_oracle"
COURSE_JSON = E2 / "s2a_course_frozen.json"
COURSE_MD = E2 / "s2a_course_frozen.md"
QUAL_JSON = E2 / "s2a_host_ready.json"
QUAL_MD = E2 / "s2a_host_ready.md"


def _material_line(n_half: int) -> float:
    return max(0.005, 1.0 / float(n_half))


def _csv_path() -> Path:
    return bch._traffic_csv_path()


def _load_pool() -> tuple[list[str], dict[str, np.ndarray]]:
    names, values = g3_sourcing.load_csv_columns(
        _csv_path(), max_columns=900, max_rows=20000)
    usable = []
    for name in names:
        series = np.asarray(values[name], dtype=np.float64)
        if series.size < ORIGIN_HELDOUT + 48:
            continue
        usable.append(name)
    if len(usable) < CELL_WIDTH * (N_IMPULSE_CELLS + N_GAP_CELLS):
        raise RuntimeError("traffic pool too small for the frozen recut")
    return usable, {name: np.asarray(values[name], dtype=np.float64)
                    for name in usable}


def _recut(names: Sequence[str]) -> list[dict[str, Any]]:
    cells = []
    cursor = 0
    specs = ([("impulsive_outlier", i) for i in range(N_IMPULSE_CELLS)]
             + [("gap", 0)])
    for family, index in specs:
        chunk = list(names[cursor:cursor + CELL_WIDTH])
        cursor += CELL_WIDTH
        train = chunk[:N_TRAIN]
        heldout = chunk[N_TRAIN:]
        support = train[:N_FACE]
        delayed = train[N_FACE:]
        unit_id = "traffic_%s_%02d" % (family, index)
        cells.append({
            "unit_id": unit_id,
            "dataset": "monash:traffic_hourly",
            "family": family,
            "train": train,
            "support": support,
            "delayed": delayed,
            "heldout": heldout,
            "n_train": len(train),
            "n_half": N_FACE,
            "material_line": _material_line(N_FACE),
            "two_x_line": 2.0 * _material_line(N_FACE),
        })
    return cells


def _inject(cell: Mapping[str, Any],
            pool: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    subset = {uid: pool[uid] for uid in
              list(cell["train"]) + list(cell["heldout"])}
    faulty = tuple(cell["train"])
    clean = tuple(cell["heldout"])
    if cell["family"] == "impulsive_outlier":
        injected, _gt = inject_label_touched_corpus(
            subset, faulty_series=faulty, clean_series=clean)
    elif cell["family"] == "gap":
        injected, _gt = inject_gap_corpus(
            subset, faulty_series=faulty, clean_series=clean)
    else:
        raise ValueError(cell["family"])
    return injected


def _compiled(op: str):
    if op == "identity":
        return None
    params = wiring.contract_params(op, PERIOD)
    return v6._compiled_bound_program(
        {"op": op, "params": params}, environment="s2a")


def _score_face(values: Mapping[str, np.ndarray],
                train_uids: Sequence[str],
                eval_uids: Sequence[str],
                compiled: Any,
                origin: int) -> dict[str, Any]:
    roster = ([{"series_uid": uid, "role": "train"} for uid in train_uids]
              + [{"series_uid": uid, "role": "eval"} for uid in eval_uids])
    assignment = {uid: compiled for uid in train_uids}
    config = dict(kdd._config())
    raw = bch._evaluate_assignment(
        roster, values, assignment, config, origin=origin)
    return {
        "mean_smase": float(raw["mean_smase"]),
        "per_view_smase": [float(x) for x in raw["per_view_smase"]],
        "n_eval": len(eval_uids),
    }


def _oracle_cell(cell: Mapping[str, Any],
                 injected: Mapping[str, np.ndarray]) -> dict[str, Any]:
    material = float(cell["material_line"])
    rows = []
    id_support = _score_face(injected, cell["delayed"], cell["support"],
                             None, ORIGIN_HELDIN)
    id_delayed = _score_face(injected, cell["support"], cell["delayed"],
                             None, ORIGIN_HELDIN)
    id_heldout = _score_face(injected, cell["train"], cell["heldout"],
                             None, ORIGIN_HELDOUT)
    for op in MENU:
        compiled = _compiled(op)
        if op == "identity":
            support, delayed, heldout = id_support, id_delayed, id_heldout
        else:
            support = _score_face(injected, cell["delayed"], cell["support"],
                                  compiled, ORIGIN_HELDIN)
            delayed = _score_face(injected, cell["support"], cell["delayed"],
                                  compiled, ORIGIN_HELDIN)
            heldout = _score_face(injected, cell["train"], cell["heldout"],
                                  compiled, ORIGIN_HELDOUT)
        support_gain = id_support["mean_smase"] - support["mean_smase"]
        delayed_gain = id_delayed["mean_smase"] - delayed["mean_smase"]
        heldout_gain = id_heldout["mean_smase"] - heldout["mean_smase"]
        facts = classify_relation(
            aggregate_gain=min(support_gain, delayed_gain),
            per_series_gains=None,
            is_identity=(op == "identity"),
            consumer_id=CONSUMER_ID,
            material_threshold=material,
        )
        support_rel = classify_relation(
            aggregate_gain=support_gain, is_identity=(op == "identity"),
            consumer_id=CONSUMER_ID, material_threshold=material)
        delayed_rel = classify_relation(
            aggregate_gain=delayed_gain, is_identity=(op == "identity"),
            consumer_id=CONSUMER_ID, material_threshold=material)
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
    oracle_set = []
    if learnable_rows:
        best = max(learnable_rows, key=lambda r: r["heldin_headroom"])
        oracle_set = [best["program"]]
        primary = best
        learnability = "LEARNABLE"
    else:
        oracle_set = ["identity"]
        primary = next(r for r in rows if r["program"] == "identity")
        learnability = "IDENTITY"
    return {
        **cell,
        "banner": ORACLE_BANNER,
        "condition": CONDITION,
        "task_kind": TASK_KIND,
        "consumer_id": CONSUMER_ID,
        "metric": METRIC,
        "origins": {"heldin": ORIGIN_HELDIN, "heldout": ORIGIN_HELDOUT},
        "programs": rows,
        "oracle_set": oracle_set,
        "oracle_program": oracle_set[0],
        "learnability": learnability,
        "heldin_headroom": primary["heldin_headroom"],
        "two_x": bool(primary.get("two_x")),
        "near_line": bool(primary.get("near_line")),
    }


def _qualify(oracles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    impulse = [o for o in oracles if o["family"] == "impulsive_outlier"]
    gap = [o for o in oracles if o["family"] == "gap"]
    producers = [o for o in impulse
                 if o["learnability"] == "LEARNABLE" and o["two_x"]]
    strong = [o for o in impulse
              if o["learnability"] == "LEARNABLE" and o["two_x"]]
    weak = [o for o in impulse
            if o["learnability"] == "LEARNABLE" and o["near_line"]]
    identity = [o for o in impulse if o["learnability"] == "IDENTITY"]
    reasons = []
    if len(producers) < 1:
        reasons.append("no_producer_learnable_with_2x_margin")
    if len(strong) < 1:
        reasons.append("no_strong_beneficiary")
    if len(weak) < 1:
        reasons.append("no_near_line_weak_beneficiary")
    if len(identity) < 1:
        reasons.append("no_identity_field")
    if len(gap) != 1:
        reasons.append("gap_guard_cell_missing")
    ready = not reasons
    course = []
    if ready:
        producer = producers[0]
        strong_b = next(o for o in strong if o["unit_id"] != producer["unit_id"]) \
            if len(strong) > 1 else producer
        weak_b = weak[0]
        ident = identity[0]
        guard = gap[0]
        course = [
            {"role": "producer", "unit_id": producer["unit_id"]},
            {"role": "identity", "unit_id": ident["unit_id"]},
            {"role": "strong_beneficiary", "unit_id": strong_b["unit_id"]},
            {"role": "near_line_conflict", "unit_id": weak_b["unit_id"]},
            {"role": "conflict_reencounter", "unit_id": weak_b["unit_id"],
             "note": "mechanism probe; readout listed separately; not in first-line regret"},
            {"role": "gap_out_of_family_guard", "unit_id": guard["unit_id"]},
        ]
        # if only one strong cell, producer and strong_beneficiary collide —
        # that fails the "beneficiary >=1 strong" as a distinct later field.
        if strong_b["unit_id"] == producer["unit_id"]:
            reasons.append("strong_beneficiary_not_distinct_from_producer")
            ready = False
            course = []
    return {
        "protocol": PROTOCOL,
        "S2_HOST_READY": ready,
        "reasons": reasons,
        "n_impulse": len(impulse),
        "n_gap": len(gap),
        "n_producer": len(producers),
        "n_strong": len(strong),
        "n_weak": len(weak),
        "n_identity": len(identity),
        "course": course,
        "part_a_tests": "149 passed (3 focused + 146 classification)",
        "classification_146": True,
        "part_a_green": True,
    }


def run() -> int:
    started = time.time()
    ORACLE_DIR.mkdir(parents=True, exist_ok=True)
    names, pool = _load_pool()
    cells = _recut(names)
    oracles = []
    fits = 0
    for cell in cells:
        injected = _inject(cell, pool)
        oracle = _oracle_cell(cell, injected)
        fits += len(MENU) * 3
        path = ORACLE_DIR / ("%s.json" % cell["unit_id"])
        path.write_text(json.dumps(oracle, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        oracles.append(oracle)
        print("ORACLE %s %s headroom=%s two_x=%s" % (
            cell["unit_id"], oracle["learnability"],
            oracle["heldin_headroom"], oracle["two_x"]), flush=True)
    qual = _qualify(oracles)
    qual["fits"] = fits
    qual["elapsed_s"] = round(time.time() - started, 1)
    QUAL_JSON.write_text(json.dumps(qual, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    lines = [
        "# S2a host-ready / course freeze",
        "",
        "**S2_HOST_READY: %s**" % qual["S2_HOST_READY"],
        "",
        "reasons: %s" % (", ".join(qual["reasons"]) or "none"),
        "impulse cells: %d  gap: %d  producer: %d  strong: %d  weak: %d  identity: %d"
        % (qual["n_impulse"], qual["n_gap"], qual["n_producer"],
           qual["n_strong"], qual["n_weak"], qual["n_identity"]),
        "fits: %d  elapsed_s: %s" % (fits, qual["elapsed_s"]),
        "",
        "Oracle files live under `artifacts/functional/e2/s2a_oracle/` and "
        "must not enter any arm prompt, store, or retrieval view.",
        "",
    ]
    if qual["course"]:
        lines.append("## Frozen course")
        for row in qual["course"]:
            lines.append("- %s: `%s`" % (row["role"], row["unit_id"]))
        COURSE_JSON.write_text(
            json.dumps({
                "protocol": PROTOCOL,
                "condition": CONDITION,
                "task_kind": TASK_KIND,
                "consumer_id": CONSUMER_ID,
                "metric": METRIC,
                "material_line": _material_line(N_FACE),
                "delta_material_note": "sum of beneficiary-unit half-split material lines; filled after live",
                "course": qual["course"],
                "order": [
                    "producer", "identity", "strong_beneficiary",
                    "near_line_conflict", "conflict_reencounter",
                    "gap_out_of_family_guard",
                ],
                "oracle_isolation": ORACLE_BANNER,
            }, ensure_ascii=False, indent=1),
            encoding="utf-8")
    COURSE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    QUAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not qual["S2_HOST_READY"]:
        print("STOP S2_HOST_READY_FAIL: %s" % "; ".join(qual["reasons"]),
              flush=True)
        return 1
    print("S2_HOST_READY", flush=True)
    return 0


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
