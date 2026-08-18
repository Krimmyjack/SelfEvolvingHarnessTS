"""run_grid0_observations.py — GRID0 第 7 步：在打开 utility 之前计算 F1/F2/F3。

信息墙：只使用 series[:origin] 与钉住训练窗口 dry-run 掩码；不调用
ScopeExecutor.evaluate / nsu._evaluate_kdd；不读取 gain。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [PROJECT_ROOT, PROJECT_ROOT / "evaluation" / "functional",
          PROJECT_ROOT / "methods" / "ttha"]:
    sys.path.insert(0, str(p))

from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features  # noqa: E402
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate  # noqa: E402

import run_grid0_census as gc  # noqa: E402

CHECKPOINT_REL = gc.CHECKPOINT_REL
PINNED_ANCHORS = gc.PINNED_ANCHORS
ORIGINS = gc.ORIGINS
CONTEXT = gc.CONTEXT
HORIZON = gc.HORIZON
MAX_MODIFIED_FRACTION = gc.MAX_MODIFIED_FRACTION

NUMERIC_F1 = [
    "missing_fraction", "longest_missing_run_fraction",
    "local_robust_z_peak", "estimated_region_start_fraction",
    "estimated_region_end_fraction", "level_excursion_score",
    "estimated_level_offset", "period_change_score", "period_reliability",
]
BOOL_F1 = ["period_repair_available"]
CAT_F1 = ["task_kind", "period_evidence_status"]


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def _f1_vector(feat: Mapping[str, Any]) -> list[float]:
    v: list[float] = []
    for k in NUMERIC_F1:
        try:
            v.append(float(feat.get(k, 0.0)))
        except Exception:
            v.append(0.0)
    for k in BOOL_F1:
        v.append(1.0 if bool(feat.get(k)) else 0.0)
    for k in CAT_F1:
        val = str(feat.get(k))
        # 只对出现在当前 roster 中的类别做简单 one-hot；未知类别记 0
        v.append(1.0 if val == "forecast" else 0.0)
        v.append(1.0 if val == "OK" else 0.0)
    return v


def _f2_for_series(raw: np.ndarray) -> dict[str, Any]:
    cand = gc._candidate()
    per_window = []
    modified_fraction_mean = 0.0
    modified_fraction_max = 0.0
    modified_in_target_share = 0.0
    modified_run_count_norm = 0.0
    modified_amplitude_ratio = 0.0
    acting_windows = 0
    for a in PINNED_ANCHORS:
        w = np.asarray(raw[a - CONTEXT:a + HORIZON], dtype=np.float64)
        art = verify_candidate(
            cand, w,
            allowed_operators=("outlier_mad",),
            inspected_regions=((0, int(w.size)),),
            maximum_modified_fraction=MAX_MODIFIED_FRACTION,
            preserve_outside_inspected_region=True,
            require_finite_output=False,
        )
        idx = np.asarray(art.modified_indices, dtype=np.int64)
        nmod = int(idx.size)
        mf = float(art.receipt.modified_fraction)
        modified_fraction_mean += mf / len(PINNED_ANCHORS)
        modified_fraction_max = max(modified_fraction_max, mf)
        if nmod > 0:
            acting_windows += 1
            target_share = float(np.mean(idx >= 192))
            runs = 1 + int(np.sum(np.diff(np.sort(idx)) > 1)) if nmod > 1 else 1
            modified_in_target_share += target_share / len(PINNED_ANCHORS)
            modified_run_count_norm += (runs / max(1, nmod)) / len(PINNED_ANCHORS)
            prepared = np.asarray(art.prepared_values, dtype=np.float64)
            amp = float(np.mean(np.abs(prepared[idx] - w[idx])) / (_mad(w) + 1e-12))
            modified_amplitude_ratio += amp / len(PINNED_ANCHORS)
        per_window.append({
            "anchor": a, "modified_points": nmod, "modified_fraction": mf,
            "selectable": art.selectable, "rejection_code": art.receipt.rejection_code,
        })
    return {
        "modified_fraction_mean": modified_fraction_mean,
        "modified_fraction_max": modified_fraction_max,
        "modified_in_target_share": modified_in_target_share,
        "modified_run_count_norm": modified_run_count_norm,
        "modified_amplitude_ratio": modified_amplitude_ratio,
        "acting_window_share": acting_windows / len(PINNED_ANCHORS),
        "per_window": per_window,
    }


def _tail_shift(raw: np.ndarray, origin: int) -> float:
    prefix = raw[:origin]
    tail = prefix[origin - HORIZON:origin]
    before = prefix[:origin - HORIZON]
    if before.size < 2 or tail.size < 2:
        return 0.0
    return float(abs(np.median(tail) - np.median(before)) / (_mad(prefix) + 1e-12))


def main() -> int:
    if not CHECKPOINT_REL.exists():
        raise SystemExit("grid0_checkpoint.json 不存在——先跑 census")
    report = json.loads(CHECKPOINT_REL.read_text(encoding="utf-8"))
    census = report.get("census")
    if not census:
        raise SystemExit("checkpoint 缺少 census")

    kdd = {s["entity_id"]: s for s in gc._load_kdd_series()}
    reg, _ = gc._load_registry_series()
    reg_by_ds = {ds: {s["entity_id"]: s for s in reg if s["dataset_id"] == ds}
                 for ds in gc.COHORT_B_ORDER}

    cohorts = []
    for name in census["cohort_A"]["selected"]:
        cohorts.append({"dataset": "kdd2018", "entity_id": name,
                        "uid": name, "raw": kdd[name]["raw"]})
    b_sel = census["cohort_B"][census["cohort_B"]["dataset_selected"]]
    for item in b_sel["selected_detail"]:
        name = item["series"]
        ent = reg_by_ds[census["cohort_B"]["dataset_selected"]][name]
        cohorts.append({"dataset": census["cohort_B"]["dataset_selected"],
                        "entity_id": name, "uid": ent["series_uid"],
                        "raw": ent["raw"]})

    # 每 series F1（按 origin）与 F2（仅一次，钉住 anchors 使其 origin 不变）
    per_series = {}
    for c in cohorts:
        f1s = {}
        f3_tail = {}
        for o in ORIGINS:
            feat = dict(extract_public_features(c["raw"][:o], task_kind="forecast"))
            f1s[str(o)] = feat
            f3_tail[str(o)] = _tail_shift(c["raw"], o)
        per_series[c["entity_id"]] = {
            "cohort": "A" if c["dataset"] == "kdd2018" else "B",
            "dataset": c["dataset"],
            "entity_id": c["entity_id"],
            "series_uid": c["uid"],
            "f1": f1s,
            "f2": _f2_for_series(c["raw"]),
            "f3_tail_shift": f3_tail,
        }
        print(f"obs per-series {c['entity_id']}", flush=True)

    # F3 cohort LOO 聚合（每 cohort × origin）
    by_cohort: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for ent, d in per_series.items():
        by_cohort[d["cohort"]].append(d)

    for cohort in ("A", "B"):
        members = by_cohort[cohort]
        for d in members:
            f3 = {}
            for o in ORIGINS:
                other = [m for m in members if m is not d]
                f2_acting = [m["f2"]["acting_window_share"] > 0 for m in other]
                f3[str(o)] = {
                    "cohort_acting_series_fraction": float(np.mean(f2_acting)) if other else 0.0,
                    "cohort_mean_modified_fraction": float(np.mean([m["f2"]["modified_fraction_mean"] for m in other])) if other else 0.0,
                    "cohort_tail_shift_deviation": float(
                        d["f3_tail_shift"][str(o)]
                        - np.median([m["f3_tail_shift"][str(o)] for m in other])
                    ) if other else 0.0,
                    "cohort_mean_z_peak": float(np.mean([
                        m["f1"][str(o)]["local_robust_z_peak"] for m in other])) if other else 0.0,
                }
            d["f3"] = f3

    # 展平成 cells，方便后续统计
    cells = []
    for d in per_series.values():
        for o in ORIGINS:
            cells.append({
                "cohort": d["cohort"], "dataset": d["dataset"],
                "series": d["entity_id"], "series_uid": d["series_uid"],
                "origin": o,
                "f1": d["f1"][str(o)],
                "f1_vector": _f1_vector(d["f1"][str(o)]),
                "f2": d["f2"],
                "f3": d["f3"][str(o)],
            })

    report["observations"] = {
        "step": "grid0 step 7 observations (pre-utility, zero gain)",
        "generated_by": "evaluation/functional/run_grid0_observations.py",
        "n_series": len(per_series),
        "n_cells": len(cells),
        "feature_encoding": {
            "f1_numeric": NUMERIC_F1, "f1_bool": BOOL_F1,
            "f1_cat_onehot": CAT_F1,
            "f1_vector_len": len(_f1_vector(next(iter(per_series.values()))["f1"]["600"])),
            "f2_fields": ["modified_fraction_mean", "modified_fraction_max",
                          "modified_in_target_share", "modified_run_count_norm",
                          "modified_amplitude_ratio", "acting_window_share"],
            "f3_fields": ["cohort_acting_series_fraction",
                          "cohort_mean_modified_fraction",
                          "cohort_tail_shift_deviation", "cohort_mean_z_peak"],
        },
        "per_series": {k: {kk: vv for kk, vv in v.items() if kk in
                           ("cohort", "dataset", "entity_id", "series_uid",
                            "f2", "f3_tail_shift", "f3")}
                       for k, v in per_series.items()},
        "cells": cells,
    }
    # 不把原始 F1 dict 写两遍；cells 已含
    CHECKPOINT_REL.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          default=str) + "\n", encoding="utf-8")
    print(f"observations done: series={len(per_series)} cells={len(cells)}")
    print("info wall: no evaluate() called, no gain read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
