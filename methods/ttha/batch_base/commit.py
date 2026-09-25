"""Delivery freeze and the later label stages. Order is enforced by files on disk:

  commit(...)       -> <job>/commit.json           (Fast's delivery; Runtime never re-selects)
  open_c_b(...)     requires commit.json           -> <job>/c_b_scores.json  (delayed check for the boundary Slow)
  freeze_e(...)     requires c_b_scores.json       -> <job>/predictions_e/*.npz, e_frozen.json (inputs only)
  score_e(...)      requires e_frozen.json         -> <job>/e_scores.json     (external evaluation only)

Each stage opens the job with only its own rows (context.STAGE_ROWS). E scores are never returned
to a controller function; they are written for the external report.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import context, data, train


def _cells(job_dir: Path) -> dict:
    out = {}
    for p in sorted((job_dir / "cells").glob("*.json")) if (job_dir / "cells").exists() else []:
        r = context.read_json(p)
        if r.get("status") == "OK":
            out[r["cell_id"]] = r
    return out


def commit(run_dir, dataset: str, job: str, material_id: str, reason: str, delivery_seed: int, stop_reason: str = "", extra: dict | None = None) -> dict:
    js = context.resolve_job(dataset, job)
    job_dir = Path(run_dir) / js.job_id
    p = job_dir / "commit.json"
    if p.exists():
        raise RuntimeError("job %s already committed; a delivery cannot be changed" % js.job_id)
    cells = _cells(job_dir)
    fitted = {c["material_id"] for c in cells.values()}
    if material_id not in fitted:
        raise ValueError("commit refused: material %s has no completed fit in this job" % material_id)
    cid = train.cell_id(js.job_id, material_id, delivery_seed)
    if cid not in cells:
        raise ValueError("commit refused: delivery seed %d has no OK cell for %s" % (delivery_seed, material_id))
    rec = {"job_id": js.job_id, "material_id": material_id, "delivery_seed": delivery_seed, "delivery_cell": cid,
           "reason": str(reason)[:1000], "stop_reason": stop_reason, "committed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
           "fitted_materials_at_commit": sorted(fitted), "extra": extra or {}}
    context.write_json(p, rec)
    return rec


def open_c_b(run_dir, dataset: str, job: str, *, roster=None) -> dict:
    """Score C_B for every OK cell of the job. Refuses to run before commit.json exists."""
    run_dir = Path(run_dir)
    if not (run_dir / context.resolve_job(dataset, job).job_id / "commit.json").exists():
        raise PermissionError("c_b prerequisite missing before loading any labels")
    ctx = context.open_job(dataset, job, run_dir, stage="c_b", roster=roster)
    if not (ctx.job_dir / "commit.json").exists():
        raise PermissionError("C_B is opened only after the job's delivery is committed")
    wins = data.build_windows(ctx.slice, ctx.job.c_b, ctx.scaler, with_truth=True)
    y_true = np.stack([w.y_true_raw for w in wins], axis=1)
    out = {"job_id": ctx.job.job_id, "opened_at_local": time.strftime("%Y-%m-%d %H:%M:%S"), "rows_read": list(ctx.job.c_b_rows), "cells": {}}
    for cid, c in _cells(ctx.job_dir).items():
        model = train.load_model(c["model_path"])
        pred = train.predict_windows(model, wins, ctx.scaler)
        out["cells"][cid] = {"material_id": c["material_id"], "model_seed": c["model_seed"], "c_b": data.score_predictions(pred, y_true, ctx.scaler.mean, ctx.scaler.scale, ctx.mase)}
        c["scores"]["c_b"] = out["cells"][cid]["c_b"]
        context.write_json(ctx.job_dir / "cells" / (cid + ".json"), c)
    context.write_json(ctx.job_dir / "c_b_scores.json", out)
    return out


def freeze_e(run_dir, dataset: str, job: str, *, roster=None) -> dict:
    run_dir = Path(run_dir)
    if not (run_dir / context.resolve_job(dataset, job).job_id / "c_b_scores.json").exists():
        raise PermissionError("e_input prerequisite missing before loading any labels")
    ctx = context.open_job(dataset, job, run_dir, stage="e_input", roster=roster)
    if not (ctx.job_dir / "c_b_scores.json").exists():
        raise PermissionError("E predictions are frozen only after C_B has been opened for the job")
    (ctx.job_dir / "predictions_e").mkdir(exist_ok=True)
    wins = data.build_windows(ctx.slice, ctx.job.e, ctx.scaler, with_truth=False)
    index = {"job_id": ctx.job.job_id, "started_local": time.strftime("%Y-%m-%d %H:%M:%S"), "rows_read": list(ctx.job.e_input_rows), "models": {}}
    for cid, c in _cells(ctx.job_dir).items():
        model = train.load_model(c["model_path"])
        pred = train.predict_windows(model, wins, ctx.scaler)
        p = ctx.job_dir / "predictions_e" / (cid + ".npz")
        np.savez(p, pred_raw=pred, origins=np.array(ctx.job.e))
        index["models"][cid] = {"path": str(p), "frozen_at_local": time.strftime("%Y-%m-%d %H:%M:%S")}
    index["frozen_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    context.write_json(ctx.job_dir / "e_frozen.json", index)
    return index


def require_cohort_frozen(run_dir, job: str, cohort_root) -> None:
    """Check W's whole-run barrier before opening any evaluation labels."""
    run_dir, cohort_root = Path(run_dir).resolve(), Path(cohort_root).resolve()
    barrier = cohort_root / "all_e_predictions_frozen.json"
    finished = cohort_root / "execution_finished.json"
    if not barrier.exists() or not finished.exists():
        raise PermissionError("whole-run E freeze barrier is missing")
    receipt = context.read_json(finished)
    eligible = {}
    for branch, branch_job in receipt["branches"]:
        branch = Path(branch).resolve()
        if branch.parent != cohort_root:
            raise PermissionError("foreign branch in E cohort")
        if (branch / branch_job / "c_b_scores.json").exists():
            eligible[branch] = branch_job
    listed = [Path(x).resolve() for x in context.read_json(barrier)["branches"]]
    if not listed or len(set(listed)) != len(listed) or set(listed) != set(eligible):
        raise PermissionError("E freeze cohort differs from completed branches")
    if eligible.get(run_dir) != job:
        raise PermissionError("this branch/job is not in the E freeze cohort")
    for branch, branch_job in eligible.items():
        frozen = branch / branch_job / "e_frozen.json"
        if not frozen.exists():
            raise PermissionError("a peer branch has not frozen E predictions")
        models = context.read_json(frozen).get("models", {})
        if not models or any(not Path(m["path"]).is_file() for m in models.values()):
            raise PermissionError("frozen E prediction files are incomplete")


def score_e(run_dir, dataset: str, job: str, *, cohort_root=None, roster=None) -> dict:
    run_dir = Path(run_dir)
    # Existing W run layout remains guarded even for a direct Python call.
    if cohort_root is None and (run_dir.parent / "experiment_started.json").exists():
        cohort_root = run_dir.parent
    if cohort_root is not None:
        require_cohort_frozen(run_dir, job, cohort_root)
    if not (run_dir / context.resolve_job(dataset, job).job_id / "e_frozen.json").exists():
        raise PermissionError("e_target prerequisite missing before loading any labels")
    ctx = context.open_job(dataset, job, run_dir, stage="e_target", roster=roster)
    fz_path = ctx.job_dir / "e_frozen.json"
    if not fz_path.exists():
        raise PermissionError("E is scored only after all E predictions are frozen")
    fz = context.read_json(fz_path)
    y_true = data.read_targets(ctx.slice, ctx.job.e)
    out = {"job_id": ctx.job.job_id, "scored_at_local": time.strftime("%Y-%m-%d %H:%M:%S"), "e_frozen_at": fz["frozen_at_local"], "rows_read": list(ctx.job.e_target_rows), "cells": {}}
    cells = _cells(ctx.job_dir)
    for cid, m in fz["models"].items():
        z = np.load(m["path"])
        if not np.array_equal(z["origins"], np.array(ctx.job.e)):
            raise RuntimeError("frozen E origins differ from the job table")
        c = cells[cid]
        out["cells"][cid] = {"material_id": c["material_id"], "model_seed": c["model_seed"], "e": data.score_predictions(z["pred_raw"], y_true, ctx.scaler.mean, ctx.scaler.scale, ctx.mase)}
    context.write_json(ctx.job_dir / "e_scores.json", out)
    return out
