"""Shared-Consumer fit and C_A scoring. `MLP` / `train_arm` are the a94 package's model_mlp.py
with the configuration passed in explicitly (same seed discipline: torch.manual_seed(model_seed)
before construction, AdamW, pooled MSE, parent 0.5 + child 0.5, fixed (2000, 64) parent-batch
index stream from spec.batch_seed). One fit = one subprocess (`python -m methods.ttha.batch_base.train
--dataset .. --job .. --run-dir .. --material .. --seed ..`), which opens the job at the `evaluate`
stage: rows [t-672, t+96) only, so C_B / E rows are never in the worker's memory.
`evaluate()` is the controller-facing wrapper: runs the fits it is asked for (skipping cells that
already exist), charges the ledger, returns the cell records.
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import context, data, materials, spec

PYTHON = sys.executable
PER_FIT_TIMEOUT_S = 300
TRAIN_DEADLINE_S = 270


# ----------------------------------------------------------------------------- model (torch imported lazily)
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_model():
    torch, nn = _torch()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(spec.L, spec.HIDDEN1), nn.ReLU(), nn.Linear(spec.HIDDEN1, spec.HIDDEN2), nn.ReLU(), nn.Linear(spec.HIDDEN2, spec.H))

        def forward(self, x):
            return self.net(x)

    return MLP


def device():
    torch, _ = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainResult:
    loss_curve: list = field(default_factory=list)
    final_train_loss: float = 0.0
    n_updates_done: int = 0
    seconds: float = 0.0
    first_step_loss: float | None = None
    first_step_grad_norm: float | None = None
    first_step_param_delta_norm: float | None = None


def batch_indices(model_seed: int, n_pool: int = spec.N_POOL, n_updates: int | None = None) -> np.ndarray:
    # n_pool: actual pooled parent count of a profile with a data-dependent legal window set; default = the frozen 32x433 pool.
    # n_updates (DEV-DATA-READINESS-TRAINING-LENGTH): a shorter run takes the first rows of the unchanged full stream, so its
    # batches are exactly the prefix of the 2000-update trajectory; None = the full stream (old behaviour).
    rng = np.random.RandomState(spec.batch_seed(model_seed))
    full = rng.randint(0, n_pool, size=(spec.N_UPDATES, spec.PARENT_BATCH))
    return full if n_updates is None else full[:check_n_updates(n_updates)]


def check_n_updates(n_updates) -> int:
    n = int(n_updates)
    if n != n_updates or not 1 <= n <= spec.N_UPDATES:
        raise ValueError("n_updates must be an integer in [1, %d]" % spec.N_UPDATES)
    return n


def train_arm(X_parent_flat, y_parent_flat, child_flat, batch_idx, model_seed: int, timeout_seconds: float | None = None, record_every: int = 100,
              n_updates: int | None = None, checkpoints=(), on_checkpoint=None):
    """n_updates None = spec.N_UPDATES (old behaviour). `on_checkpoint(step, model, batch_loss, seconds)` runs right after the
    opt.step() that completes update `step` for every step in `checkpoints`; it must only copy state (no RNG, optimizer or
    parameter change), so the trajectory is identical with or without checkpoints."""
    steps = spec.N_UPDATES if n_updates is None else check_n_updates(n_updates)
    if len(batch_idx) < steps:
        raise ValueError("batch index stream shorter than the requested updates")
    marks = {int(c) for c in checkpoints}
    if any(not 1 <= c < steps for c in marks) or (marks and on_checkpoint is None):
        raise ValueError("checkpoints must lie in [1, n_updates) and need a callback")
    torch, _ = _torch()
    dev = device()
    torch.manual_seed(model_seed)
    model = make_model()().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=spec.LR, weight_decay=spec.WEIGHT_DECAY)
    Xp = torch.as_tensor(X_parent_flat, dtype=torch.float32, device=dev)
    Yp = torch.as_tensor(y_parent_flat, dtype=torch.float32, device=dev)
    has_child = child_flat is not None
    if has_child:
        Xc = torch.as_tensor(child_flat[0], dtype=torch.float32, device=dev)
        Yc = torch.as_tensor(child_flat[1], dtype=torch.float32, device=dev)
    res = TrainResult()
    t0 = time.time()
    deadline = None if timeout_seconds is None else (t0 + float(timeout_seconds))

    def pooled(pred, target):
        return ((pred - target) ** 2).mean(dim=-1).mean()

    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            raise RuntimeError("fit exceeded %.0fs after %d updates" % (timeout_seconds, step))
        idx = torch.as_tensor(batch_idx[step], dtype=torch.long, device=dev)
        opt.zero_grad(set_to_none=True)
        loss_p = pooled(model(Xp[idx]), Yp[idx])
        if has_child:
            loss = 0.5 * loss_p + 0.5 * pooled(model(Xc[idx]), Yc[idx])
        else:
            loss = loss_p
        loss.backward()
        if step == 0:
            grad_sq = sum(float((p.grad ** 2).sum().item()) for p in model.parameters() if p.grad is not None)
            res.first_step_loss = float(loss.item())
            res.first_step_grad_norm = grad_sq ** 0.5
        pre = [p.detach().clone() for p in model.parameters()] if step == 0 else None
        opt.step()
        if step == 0:
            delta_sq = sum(float(((p.detach() - q).float() ** 2).sum().item()) for p, q in zip(model.parameters(), pre))
            res.first_step_param_delta_norm = delta_sq ** 0.5
        if step % record_every == 0 or step == steps - 1:
            res.loss_curve.append((step, float(loss.item())))
        res.n_updates_done = step + 1
        if step + 1 in marks:
            on_checkpoint(step + 1, model, float(loss.item()), time.time() - t0)
    res.final_train_loss = res.loss_curve[-1][1]
    res.seconds = time.time() - t0
    return model, res


def predict(model, X_flat: np.ndarray) -> np.ndarray:
    torch, _ = _torch()
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(X_flat, dtype=torch.float32, device=device())
        return model(x).cpu().numpy()


def predict_windows(model, windows, scaler: data.Scaler) -> np.ndarray:
    n_entities = scaler.mean.shape[0]
    pred_raw = np.empty((n_entities, len(windows), spec.H))
    for oi, w in enumerate(windows):
        pred_norm = predict(model, w.X_norm)
        pred_raw[:, oi, :] = pred_norm * scaler.scale[:, None] + scaler.mean[:, None]
    return pred_raw


def save_model(model, path) -> None:
    torch, _ = _torch()
    torch.save(model.state_dict(), path)


def load_model(path):
    torch, _ = _torch()
    m = make_model()().to(device())
    m.load_state_dict(torch.load(path, map_location=device()))
    m.eval()
    return m


# ----------------------------------------------------------------------------- one fit (subprocess entry)
def cell_id(job_id: str, material_id: str, seed: int) -> str:
    return "%s__%s__s%d" % (job_id, material_id, seed)


def fit_one(dataset: str, job: str, run_dir: str, material_id: str, seed: int, train_deadline: float = TRAIN_DEADLINE_S, feedback: bool = True, roster=None) -> dict:
    t0 = time.time()
    ctx = context.open_job(dataset, job, run_dir, stage="evaluate" if feedback else "material", roster=roster)       # rows [t-672, t+96): no C_B / E row exists in memory
    index = context.read_json(ctx.job_dir / "materials" / "index.json")
    if material_id not in index:
        raise RuntimeError("unknown material %s for %s" % (material_id, ctx.job.job_id))
    ref = materials.MaterialRef(**index[material_id])
    P = ctx.parents
    Xp, Yp = P.X_norm.reshape(-1, spec.L), P.y_norm.reshape(-1, spec.H)
    child = materials.load_child(ref)
    bidx = batch_indices(seed)
    model, tres = train_arm(Xp, Yp, child, bidx, model_seed=seed, timeout_seconds=train_deadline)
    cid = cell_id(ctx.job.job_id, material_id, seed)
    (ctx.job_dir / "runs").mkdir(exist_ok=True)
    (ctx.job_dir / "predictions_c_a").mkdir(exist_ok=True)
    (ctx.job_dir / "cells").mkdir(exist_ok=True)
    mp = ctx.job_dir / "runs" / (cid + ".pt")
    save_model(model, mp)
    score = None
    if feedback:
        wins = data.build_windows(ctx.slice, ctx.job.c_a, ctx.scaler, with_truth=True)
        pred = predict_windows(model, wins, ctx.scaler)
        y_true = np.stack([w.y_true_raw for w in wins], axis=1)
        score = data.score_predictions(pred, y_true, ctx.scaler.mean, ctx.scaler.scale, ctx.mase)
        np.savez(ctx.job_dir / "predictions_c_a" / (cid + ".npz"), pred_raw=pred, origins=np.array(ctx.job.c_a))
    import torch
    rec = {"cell_id": cid, "status": "OK", "job_id": ctx.job.job_id, "dataset": dataset, "material_id": material_id, "material_alias_of": ref.alias_of,
           "model_seed": seed, "batch_seed": spec.batch_seed(seed), "child_view": child is not None, "n_identity_entities": ref.n_identity,
           "roster": list(ctx.job.roster), "roster_explicit": ctx.job.roster_override is not None,
           "rows_read": list(ctx.job.evaluate_rows if feedback else ctx.job.material_rows), "device": str(device()), "torch": torch.__version__, "python": sys.version.split()[0], "hostname": platform.node(),
           "model_path": str(mp), "train": {"seconds": tres.seconds, "n_updates_done": tres.n_updates_done, "final_train_loss": tres.final_train_loss,
                                           "first_step_loss": tres.first_step_loss, "first_step_grad_norm": tres.first_step_grad_norm,
                                           "first_step_param_delta_norm": tres.first_step_param_delta_norm, "loss_curve": tres.loss_curve},
           "scores": {"c_a": score} if feedback else {}, "seconds_total": time.time() - t0}
    context.write_json(ctx.job_dir / "cells" / (cid + ".json"), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--material", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-deadline", type=float, default=TRAIN_DEADLINE_S)
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--roster", default=None, help="comma-separated explicit 32-entity population; omitted = dataset default")
    a = ap.parse_args()
    rec = fit_one(a.dataset, a.job, a.run_dir, a.material, a.seed, a.train_deadline, feedback=not a.no_feedback,
                  roster=None if a.roster is None else a.roster.split(","))
    print("CELL_OK %s c_a=%.4f train=%.1fs" % (rec["cell_id"], rec["scores"].get("c_a", {}).get("normalized_mse_macro", float("nan")), rec["train"]["seconds"]))


# ----------------------------------------------------------------------------- controller-facing wrapper
def evaluate(ctx: context.JobContext, ref: materials.MaterialRef, seeds, ledger, repo_root: Path, log=print) -> list:
    """Fit `ref` under each seed in its own subprocess (skipping existing OK cells), charge the ledger,
    and return the cell records (C_A scores only)."""
    out = []
    for seed in seeds:
        cid = cell_id(ctx.job.job_id, ref.material_id, seed)
        cell_path = ctx.job_dir / "cells" / (cid + ".json")
        if cell_path.exists() and context.read_json(cell_path).get("status") == "OK":
            out.append(context.read_json(cell_path))
            ledger.note_cache(cid)
            continue
        if ref.alias_of:                                   # same job, same assignment, same seed -> the alias's fit is this fit
            src = ctx.job_dir / "cells" / (cell_id(ctx.job.job_id, ref.alias_of, seed) + ".json")
            if src.exists() and context.read_json(src).get("status") == "OK":
                rec = context.read_json(src)
                rec = {**rec, "cell_id": cid, "material_id": ref.material_id, "material_alias_of": ref.alias_of, "source": "alias:" + rec["cell_id"]}
                context.write_json(cell_path, rec)
                out.append(rec)
                ledger.note_cache(cid)
                log("fit %s ALIAS of %s" % (cid, rec["source"]))
                continue
        ledger.check_fit()
        t0 = time.time()
        args = [PYTHON, "-m", "methods.ttha.batch_base.train", "--dataset", ctx.job.dataset, "--job", ctx.job.job_id,
                "--run-dir", str(ctx.run_dir), "--material", ref.material_id, "--seed", str(seed)]
        if ctx.job.roster_override is not None:
            args += ["--roster", ",".join(ctx.job.roster)]
        try:
            pr = subprocess.run(args, cwd=str(repo_root), capture_output=True, text=True, timeout=PER_FIT_TIMEOUT_S)
            rc, so, se = pr.returncode, pr.stdout, pr.stderr
        except subprocess.TimeoutExpired as ex:
            rc, so, se = -999, (ex.stdout or ""), (ex.stderr or "") + "\nTIMEOUT"
        ok = rc == 0 and cell_path.exists() and context.read_json(cell_path).get("status") == "OK"
        ledger.charge_fit(cid, ok, time.time() - t0, "" if ok else (se or so)[-600:])
        log("fit %s %s %.1fs" % (cid, "OK" if ok else "FAILED", time.time() - t0))
        if ok:
            out.append(context.read_json(cell_path))
        else:
            out.append({"cell_id": cid, "status": "FAILED", "detail": (se or so)[-600:]})
    return out


if __name__ == "__main__":
    main()
