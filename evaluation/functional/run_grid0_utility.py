"""run_grid0_utility.py — GRID0 第 8 步：打开 Consumer utility。

前置：checkpoint.observations 已完成（信息墙顺序检查）。逐 cell 跑
outlier_mad vs identity，记录 gain；可断点续跑（已存在且 gain 非 None 的
cell 跳过）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [PROJECT_ROOT, PROJECT_ROOT / "evaluation" / "functional",
          PROJECT_ROOT / "methods" / "ttha"]:
    sys.path.insert(0, str(p))

import run_v1_kdd2018_natural_slow_update as nsu  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

import run_grid0_census as gc  # noqa: E402
import run_grid0_observations as go  # noqa: E402

CHECKPOINT_REL = gc.CHECKPOINT_REL
ORIGINS = gc.ORIGINS
PINNED_ANCHORS = gc.PINNED_ANCHORS


def _config(dataset: str, origin: int) -> dict:
    return {
        "dataset_id": dataset,
        "sampling": "hourly_regular",
        "period": 24,
        "anchors": list(PINNED_ANCHORS),
        "support_origin": origin,
        "selection_origin": origin,
    }


def _series_value(obs: dict, raw: np.ndarray):
    return raw


def main() -> int:
    if not CHECKPOINT_REL.exists():
        raise SystemExit("checkpoint 不存在")
    report = json.loads(CHECKPOINT_REL.read_text(encoding="utf-8"))
    obs = report.get("observations")
    if not obs or not obs.get("cells"):
        raise SystemExit("observations 不存在——顺序违规，拒绝打开 utility")
    if len(obs["cells"]) != 210:
        raise SystemExit(f"observations cells 数量异常 {len(obs['cells'])}，拒绝运行")

    kdd = {s["entity_id"]: s["raw"] for s in gc._load_kdd_series()}
    reg, _ = gc._load_registry_series()
    reg_by_ds = {ds: {s["entity_id"]: s for s in reg if s["dataset_id"] == ds}
                 for ds in gc.COHORT_B_ORDER}
    b_dataset = report["census"]["cohort_B"]["dataset_selected"]

    cells = obs["cells"]
    done = {(c["series"], c["origin"]) for c in (report.get("cells") or [])}
    out = []
    t_start = time.perf_counter()
    for i, c in enumerate(cells):
        key = (c["series"], c["origin"])
        if key in done:
            continue
        dataset = c["dataset"]
        if dataset == "kdd2018":
            raw = kdd[c["series"]]
        else:
            raw = reg_by_ds[b_dataset][c["series"]]["raw"]
        roster = [{"series_uid": c["series_uid"], "role": "train"},
                  {"series_uid": c["series_uid"], "role": "eval"}]
        values = {c["series_uid"]: raw}
        ex = ScopeExecutor(roster, values, _config(dataset, c["origin"]),
                           evaluate_fn=nsu._evaluate_kdd)
        row = {"cohort": c["cohort"], "dataset": dataset, "series": c["series"],
               "series_uid": c["series_uid"], "origin": c["origin"]}
        try:
            rr = ex.evaluate((("outlier_mad", {}),), c["origin"])
            row.update({
                "gain": float(rr.gain) if rr.gain is not None else None,
                "verification_passed": bool(rr.verification.passed),
                "checked_windows": int(rr.verification.checked_windows),
                "behavior_point_count": int(rr.behavior_point_count),
                "error": rr.error,
            })
        except Exception as exc:  # noqa: BLE001 —— 机械故障记录后停
            row.update({"gain": None, "verification_passed": None,
                        "behavior_point_count": None,
                        "error": f"{type(exc).__name__}: {exc}"})
        out.append(row)
        if (i + 1) % 25 == 0:
            print(f"utility {i+1}/{len(cells)} elapsed={time.perf_counter()-t_start:.1f}s", flush=True)

    if out:
        report["cells"] = (report.get("cells") or []) + out
        CHECKPOINT_REL.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                              default=str) + "\n", encoding="utf-8")
    all_cells = report.get("cells") or []
    print(f"utility done: total_cells={len(all_cells)} newly_computed={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
