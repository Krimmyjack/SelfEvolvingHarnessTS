"""DEV-CLS-AUG-WIRING (docs/DEV_CLS_AUG_WIRING_TASK_2026-09-25.md): classification-augmentation wiring and metadata inventory.
0 LLM, TRAIN members only (no UCR TEST member is opened), no card learning, no AutoDA joint training, no final evaluation.

  --smoke      parser / normalization / split / FCN shape / determinism / augmentation checks (short fits, 3 epochs)
  --ops        AutoDA official transforms: enabled state, formula, numeric behaviour vs strength (on GunPoint TRAIN, z-normalized)
  --timing     synthetic FCN step-time microbenchmark over the candidate lengths (no dataset, no fit)
  --precheck   6 full fits: {GunPoint, PowerCons} TRAIN x {None, Jitter, WindowSliceWarp} x seed 0, 2000 epochs, 1 GPU lane
  --inventory  UCR metadata filter -> download missing official zips -> parse TRAIN only -> per-class gate -> source groups -> families
Output _scratch/dev_cls_aug_wiring/.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import math
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "_scratch" / "dev_cls_aug_wiring"
LOCAL = REPO / "data" / "ucr_task_context"
DL = REPO / "data" / "ucr_cls_aug"
META = OUT / "meta"
UCR_SUMMARY_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv"
ZIP_URL = "https://timeseriesclassification.com/aeon-toolkit/%s.zip"

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, str(REPO / "methods" / "ttha" / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cad = _load("cls_aug_data")                    # numpy only (inventory runs without torch)


class _Lazy:
    mod = None

    def __getattr__(self, item):
        if _Lazy.mod is None:
            _Lazy.mod = _load("cls_aug")
        return getattr(_Lazy.mod, item)


ca = _Lazy()                                   # torch part, imported on first use

WIRING = ("GunPoint", "PowerCons")
SEED = 0
P_AUG = 0.5                                   # wiring-only choice, not frozen
ARMS = (("None", None, 0.0), ("Jitter", 0.03, P_AUG), ("WindowSliceWarp", 0.2, P_AUG))   # wiring-only strengths, not frozen
SEALED = {"CatsDogs": "D2_sealed (sol 2026-08-26: remain sealed)", "Epilepsy2": "D3_reserve sealed; TEST read by the 08-28 capstone"}

# Conservative same-source groups (shared raw recordings / same subjects or study); every member must sit on the same side.
SOURCE_GROUPS = {
    "G_hand_xray": (["DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "MiddlePhalanxOutlineAgeGroup",
                     "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect",
                     "ProximalPhalanxTW", "PhalangesOutlinesCorrect", "HandOutlines"],
                    "outlines extracted from the same hand radiograph collection (Davis); Phalanges* is the union of the three *Correct sets"),
    "G_cricket": (["CricketX", "CricketY", "CricketZ"], "x/y/z axes of the same accelerometer recordings"),
    "G_gunpoint": (["GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung"],
                   "the 2018 variants re-use the original GunPoint recordings; GunPoint is a wiring dataset"),
    "G_uwave": (["UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ"], "same gestures, axes / concatenation"),
    "G_worms": (["Worms", "WormsTwoClass"], "same worm motion recordings, 5-class vs 2-class labelling"),
    "G_ford": (["FordA", "FordB"], "same engine-noise problem, two acquisition conditions"),
    "G_freezer": (["FreezerRegularTrain", "FreezerSmallTrain"], "same freezer power recordings, different train sizes"),
    "G_nifecg": (["NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2"], "two thorax leads of the same recordings"),
    "G_eog": (["EOGHorizontalSignal", "EOGVerticalSignal"], "two channels of the same recordings"),
    "G_semg": (["SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2"], "same sEMG recordings, three labellings"),
    "G_uk_devices": (["Computers", "ElectricDevices", "LargeKitchenAppliances", "RefrigerationDevices", "ScreenType", "SmallKitchenAppliances"],
                     "same UK household electricity study (Powering the Nation); ElectricDevices is drawn from it"),
    "G_mixedshapes": (["MixedShapesRegularTrain", "MixedShapesSmallTrain"], "same shapes, different train sizes"),
    "G_faces": (["FaceAll", "FacesUCR"], "same face-profile outlines"),
    "G_powercons": (["PowerCons"], "wiring dataset"),
}
GROUP_OF = {m: g for g, (ms, _) in SOURCE_GROUPS.items() for m in ms}


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    tmp.replace(path)


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_norm(name: str):
    rec = ca.read_ucr_train(LOCAL / (name + ".zip"), name)
    return rec, ca.znorm(rec["values"])


def mem_snapshot() -> dict:
    import psutil
    import torch
    p = psutil.Process()
    mi = p.memory_info()
    out = {"host_rss_mb": mi.rss / 2 ** 20, "host_peak_wset_mb": getattr(mi, "peak_wset", 0) / 2 ** 20}
    if torch.cuda.is_available():
        out["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
        out["cuda_max_reserved_mb"] = torch.cuda.max_memory_reserved() / 2 ** 20
    return out


# ----------------------------------------------------------------------------------------------------------------------------- smoke
def smoke() -> dict:
    import torch
    dev = device()
    res, ok = {}, True

    def check(name, cond, detail=None):
        nonlocal ok
        res[name] = {"pass": bool(cond), "detail": detail}
        ok = ok and bool(cond)

    rec, Z = load_norm("GunPoint")
    raw = rec["values"]
    check("parser_train_only", rec["member"].lower().endswith("_train.ts"), rec["member"])
    check("parser_shape", raw.shape == (50, 150) and rec["univariate"] and rec["equal_length"] and not rec["has_nan"],
          {"shape": list(raw.shape), "classes": rec["class_counts"]})
    check("parser_raw_not_normalized", float(np.abs(raw.std(axis=1) - 1).max()) > 1e-3, float(np.abs(raw.std(axis=1) - 1).max()))
    check("znorm", float(np.abs(Z.mean(axis=1)).max()) < 1e-9 and float(np.abs(Z.std(axis=1) - 1).max()) < 1e-9)
    sp = ca.split_fit_feedback(rec["classes"])
    allidx = np.sort(np.concatenate([sp["fit"], sp["feedback"]]))
    check("split_disjoint_exhaustive", np.array_equal(allidx, np.arange(rec["n"])) and not set(sp["fit"]) & set(sp["feedback"]), sp["per_class"])
    sp2 = ca.split_fit_feedback(rec["classes"])
    check("split_deterministic", np.array_equal(sp["fit"], sp2["fit"]))
    m = ca.FCN(2).to(dev)
    x = torch.randn(4, 1, 150, device=dev)
    check("fcn_shape", tuple(m(x).shape) == (4, 2))
    nparam = sum(p.numel() for p in m.parameters())
    check("fcn_params", nparam == 1152 + 256 + 164096 + 512 + 98432 + 256 + 258, nparam)
    # determinism: two short fits with the same seed give identical weights (with and without augmentation)
    Xf, yf = Z[sp["fit"]], rec["classes"][sp["fit"]]
    for arm, s, p in (("None", None, 0.0), ("WindowSliceWarp", 0.2, P_AUG)):
        op = ca.AutoDAOp(arm, s, 150, dev) if s is not None else None
        a = ca.fit_fcn(Xf, yf, 2, seed=7, device=dev, aug=op, p_aug=p, epochs=3)
        b = ca.fit_fcn(Xf, yf, 2, seed=7, device=dev, aug=op, p_aug=p, epochs=3)
        same = all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"])
        check("determinism_" + arm, same and a["loss_curve"] == b["loss_curve"], a["aug"])
    # augmentation acts, keeps shape, never touches labels; evaluation path has no augmentation hook
    xb = torch.as_tensor(Z[:16], dtype=torch.float32, device=dev).unsqueeze(1)
    for name, s in (("Jitter", 0.03), ("WindowSliceWarp", 0.2), ("Scale", 0.5)):
        out = ca.AutoDAOp(name, s, 150, dev)(xb)
        diff = float((out - xb).abs().max())
        if name == "Scale":
            check("scale_strength_0.5_is_identity", diff == 0.0, diff)
        else:
            check("aug_changes_" + name, diff > 0, diff)
    import inspect
    check("predict_has_no_aug_hook", "aug" not in inspect.signature(ca.predict).parameters)
    res["_all_pass"] = ok
    wjson(OUT / "smoke.json", res)
    return res


# ----------------------------------------------------------------------------------------------------------------------------- ops table
OPS_FORMULA = {
    "Raw": "identity",
    "Jitter": "x + N(0,1) * s, independent per point (s = noise std in the units of x)",
    "Scale": "x * (1 + U(0,1) * (2s - 1)), independent per point: s = 0.5 is identity, s < 0.5 shrinks, s > 0.5 inflates (per-point "
             "multiplicative noise, not a global amplitude scaling)",
    "MagnitudeWarp": "x + s * (sin(f t + phi) - x), t = linspace(0, 2pi, T), f, phi ~ N(0,1) per series: interpolation toward a random "
                     "sinusoid (s = 1 replaces the series by the sinusoid), not a smooth multiplicative warp",
    "FreqWarp": "rFFT magnitude/phase resynthesized on a warped frequency axis; alpha = clamp(1 + 0.1 s N(0,1), 0.9, 1.1); audio constants "
                "(sample rate 16 kHz, Fhi 4.8 kHz) hard-coded; memory O(T^2) per series",
    "WindowSliceWarp": "crop: start ~ floor(floor(sT/2) * U), keep T - floor(sT/2) (+1) points, linear-interpolate back to T "
                       "(window slicing; keeps at least 1 - s/2 of the series)",
    "IAAFT": "s * IAAFT surrogate (20 iterations, amplitude spectrum + value distribution kept, phases randomized) + (1 - s) * x",
    "DRC": "dynamic range compression: values inside [-s, s] are halved, values outside unchanged (threshold = strength)",
    "Perm": "segment permutation; the released code references undefined names (seg, torch.random) and raises",
    "Rotation": "3-D rotation of 3 channels; raises for < 3 channels and returns zeros for the rotated channels (rotation never written)",
    "Downsampling": "keep every k-th point, k = int(2 + 3s), linear-interpolate back; deterministic (no random draw)",
    "Resampling": "strength unused; deterministic: x[0:T-1] repeated, first T points re-interpolated to T (drops the last point, "
                  "appends x[0])",
    "Slice": "deterministic: mean of linearly stretched windows of width T(0.5 + 0.5s), step width(0.2 + 0.8s)",
    "TimeWarp": "6 knots at linspace(0, T-1) scaled by N(1, s) each, bicubic-interpolated, rescaled so the last knot maps to T-1, clamped, "
                "then linear grid_sample: a time map that is NOT guaranteed monotone",
}
STRENGTHS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def ops_table() -> dict:
    import torch
    dev = device()
    mod = ca.autoda_transforms()
    enabled = [c.__name__ for c in mod.AVAILABLE_TRANSFORMS]
    rec, Z = load_norm("GunPoint")
    T = Z.shape[1]
    xb = torch.as_tensor(Z, dtype=torch.float32, device=dev).unsqueeze(1)
    ramp = torch.linspace(0, 1, T, device=dev).view(1, 1, T).repeat(64, 1, 1)
    table = {}
    for name in OPS_FORMULA:
        row = {"enabled_in_AVAILABLE_TRANSFORMS": name in enabled, "formula": OPS_FORMULA[name], "by_strength": {}}
        for s in STRENGTHS:
            try:
                torch.manual_seed(0)
                op = ca.AutoDAOp(name, s, T, dev)
                a = op(xb)
                b = op(xb)
                d = (a - xb)
                cell = {"rms_change": float(d.pow(2).mean().sqrt()), "fraction_points_changed": float((d.abs() > 1e-6).float().mean()),
                        "identity": bool(torch.equal(a, xb)), "random": not bool(torch.equal(a, b)),
                        "out_std": float(a.std(dim=-1).mean())}
                if name in ("WindowSliceWarp", "TimeWarp", "Slice", "Downsampling", "Resampling"):
                    r = op(ramp)
                    cell["time_map_non_monotone_fraction"] = float((r.diff(dim=-1) < -1e-6).any(dim=-1).float().mean())
                row["by_strength"][str(s)] = cell
            except Exception as exc:                          # noqa: BLE001 - record the official code's behaviour
                row["by_strength"][str(s)] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
        table[name] = row
    out = {"source": str(ca.AUTODA_TRANSFORMS.relative_to(REPO)), "commit": ca.AUTODA_COMMIT, "AVAILABLE_TRANSFORMS": enabled,
           "probe": "GunPoint TRAIN (50 series, T=150), per-series z-normalized; rms in z units; time map from a 0..1 ramp", "ops": table}
    wjson(OUT / "ops_table.json", out)
    return out


# ----------------------------------------------------------------------------------------------------------------------------- timing
def timing() -> dict:
    import torch
    import torch.nn.functional as F
    dev = device()
    ca.set_determinism()
    rows = []
    for T in (46, 80, 150, 300, 500, 1024, 1500, 2709):
        for bs in (3, 16):
            torch.manual_seed(0)
            torch.cuda.reset_peak_memory_stats() if dev.type == "cuda" else None
            m = ca.FCN(4).to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3, eps=1e-7)
            x = torch.randn(bs, 1, T, device=dev)
            y = torch.randint(0, 4, (bs,), device=dev)
            for _ in range(20):
                opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0, n = time.time(), 200
            for _ in range(n):
                opt.zero_grad(); F.cross_entropy(m(x), y).backward(); opt.step()
            if dev.type == "cuda":
                torch.cuda.synchronize()
            rows.append({"T": T, "batch": bs, "ms_per_step": 1000 * (time.time() - t0) / n, **mem_snapshot()})
    wjson(OUT / "timing.json", {"note": "synthetic random inputs, no dataset, no fit; plain step without augmentation", "rows": rows})
    return rows


# ----------------------------------------------------------------------------------------------------------------------------- precheck
ARM_BY_NAME = {a[0]: a for a in ARMS}


def run_fit(ds: str, arm: str, seed: int, outdir: Path, tag: str = "") -> dict:
    """One wiring fit on the fit part of a wiring dataset's TRAIN, scored on its feedback part (never augmented)."""
    import os
    import torch
    dev = device()
    path = outdir / ("%s__%s__s%d%s.json" % (ds, arm, seed, ("__" + tag) if tag else ""))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rec, Z = load_norm(ds)
    sp = ca.split_fit_feedback(rec["classes"])
    Xf, yf = Z[sp["fit"]], rec["classes"][sp["fit"]]
    Xv, yv = Z[sp["feedback"]], rec["classes"][sp["feedback"]]
    C = len(rec["label_names"])
    _, s, p = ARM_BY_NAME[arm]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    op = ca.AutoDAOp(arm, s, Z.shape[1], dev) if s is not None else None
    Xv_before = Xv.copy()
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    r = ca.fit_fcn(Xf, yf, C, seed=seed, device=dev, aug=op, p_aug=p)
    pred_v = ca.predict(r["state"], C, Xv, dev)
    pred_f = ca.predict(r["state"], C, Xf, dev)
    curve = r.pop("loss_curve")
    r.pop("state")
    out = {"dataset": ds, "arm": arm, "strength": s, "p_aug": p, "seed": seed, "tag": tag, "n_fit": int(len(yf)), "n_feedback": int(len(yv)),
           "T": int(Z.shape[1]), "n_classes": C, "split_per_class": sp["per_class"], "split_gate_fit20_fb10": sp["gate_fit20_fb10"],
           "started": started, "wall_seconds": time.time() - t0, "train": r, "loss_curve": curve, "loss_first": curve[0],
           "fit_clean_metrics_best_model": ca.metrics(yf, pred_f, C), "feedback_metrics": ca.metrics(yv, pred_v, C),
           "feedback_predictions": pred_v.tolist(), "feedback_inputs_unchanged": bool(np.array_equal(Xv, Xv_before)),
           "augmentation": op.describe() if op else None, "host": os.uname().nodename if hasattr(os, "uname") else "windows",
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "torch": torch.__version__, **mem_snapshot()}
    wjson(path, out)
    print("[fit] %s %s s%d%s F1 %.4f  %.0fs  best_ep %d" % (ds, arm, seed, (" " + tag) if tag else "", out["feedback_metrics"]["macro_f1"],
                                                         out["wall_seconds"], r["best_epoch"]), flush=True)
    return out


def precheck() -> list:
    done = [run_fit(ds, arm, SEED, OUT / "precheck") for ds in WIRING for arm, _, _ in ARMS]
    wjson(OUT / "precheck_summary.json", {"fits": done})
    return done


def launch(jobs: list, gpus: list, lanes: int, log_dir: Path, threads: int = 2) -> list:
    """Run worker command lines, at most `lanes` concurrent processes per listed GPU (CUDA_VISIBLE_DEVICES pinned per process)."""
    import os
    import subprocess
    log_dir.mkdir(parents=True, exist_ok=True)
    free = [g for g in gpus for _ in range(lanes)]
    running, rec, pending = [], [], list(jobs)
    t_all = time.time()
    while pending or running:
        while pending and free:
            g = free.pop(0)
            name, argv = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            lf = open(log_dir / (name + ".log"), "w")
            pr = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + argv, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
            running.append((pr, g, name, time.time(), lf))
        time.sleep(1)
        for item in list(running):
            pr, g, name, t0, lf = item
            if pr.poll() is not None:
                lf.close()
                running.remove(item)
                free.append(g)
                rec.append({"job": name, "gpu": g, "rc": pr.returncode, "seconds": time.time() - t0, "end_offset": time.time() - t_all})
                print("[launch] %s gpu%s rc=%s %.0fs" % (name, g, pr.returncode, time.time() - t0), flush=True)
    return rec


def parallel_precheck(gpus: list, lanes: int, seeds: list, dup: bool) -> dict:
    tag_dir = OUT / "server_precheck"
    jobs = []
    for seed in seeds:
        for ds in WIRING:
            for arm, _, _ in ARMS:
                jobs.append(("%s__%s__s%d" % (ds, arm, seed), ["--fit-one", ds, arm, str(seed), "--outdir", str(tag_dir)]))
    if dup:
        for ds in WIRING:
            for arm, _, _ in ARMS:
                jobs.append(("%s__%s__s0__dup" % (ds, arm), ["--fit-one", ds, arm, "0", "--outdir", str(tag_dir), "--tag", "dup"]))
    t0 = time.time()
    rec = launch(jobs, gpus, lanes, OUT / "logs" / "server_precheck")
    out = {"gpus": gpus, "lanes_per_gpu": lanes, "seeds": seeds, "dup": dup, "wall_seconds": time.time() - t0, "jobs": rec}
    wjson(OUT / "server_precheck_launch.json", out)
    return out


def synth_one(n: int, T: int, epochs: int, arm: str, outpath: Path) -> dict:
    """Throughput probe on random data (no dataset): the real fit loop for `epochs` epochs."""
    import torch
    dev = device()
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, T))
    y = rng.integers(0, 4, n)
    _, s, p = ARM_BY_NAME[arm]
    op = ca.AutoDAOp(arm, s, T, dev) if s is not None else None
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = ca.fit_fcn(ca.znorm(X), y, 4, seed=0, device=dev, aug=op, p_aug=p, epochs=epochs)
    secs = time.time() - t0
    out = {"n": n, "T": T, "epochs": epochs, "arm": arm, "batch": r["batch_size"], "steps": r["steps"], "seconds": secs,
           "ms_per_step": 1000 * secs / r["steps"], **mem_snapshot()}
    wjson(outpath, out)
    return out


SCALE_SHAPES = {"small": (120, 144, 60), "large": (1000, 1024, 6)}      # (n, T, epochs): ~PowerCons fit part; ~StarLightCurves


def scale_test(gpu: int, lane_list: list, arm: str) -> list:
    res = []
    for shape, (n, T, ep) in SCALE_SHAPES.items():
        for L in lane_list:
            d = OUT / "scale" / ("%s_%s_L%d" % (shape, arm, L))
            jobs = [("%s_%s_L%d_%d" % (shape, arm, L, i), ["--synth-one", str(n), str(T), str(ep), arm, str(d / ("p%d.json" % i))])
                    for i in range(L)]
            t0 = time.time()
            launch(jobs, [gpu], L, OUT / "logs" / "scale")
            wall = time.time() - t0
            per = [json.loads((d / ("p%d.json" % i)).read_text(encoding="utf-8")) for i in range(L)]
            steps = sum(q["steps"] for q in per)
            row = {"shape": shape, "n": n, "T": T, "arm": arm, "lanes": L, "wall_seconds": wall,
                   "mean_ms_per_step": float(np.mean([q["ms_per_step"] for q in per])),
                   "throughput_steps_per_s": steps / max(1e-9, max(q["seconds"] for q in per)),
                   "cuda_max_allocated_mb": max(q.get("cuda_max_allocated_mb", 0) for q in per), "host_rss_mb": max(q["host_rss_mb"] for q in per)}
            res.append(row)
            print("[scale] %s L=%d  %.2f ms/step  %.0f steps/s total" % (shape, L, row["mean_ms_per_step"], row["throughput_steps_per_s"]), flush=True)
    wjson(OUT / "scale.json", {"gpu": gpu, "arm": arm, "note": "random data, real fit loop, concurrent processes on one GPU", "rows": res})
    return res


# ----------------------------------------------------------------------------------------------------------------------------- inventory
def fetch(url: str, dest: Path) -> dict:
    t0 = time.time()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    return {"url": url, "bytes": dest.stat().st_size, "seconds": time.time() - t0,
            "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def inventory(download: bool = True, archive: Path | None = None) -> dict:
    """archive = the official UCRArchive_2018.zip: every candidate is read from it (single canonical source, no download);
    local timeseriesclassification.com .ts copies are only cross-checked against it."""
    META.mkdir(parents=True, exist_ok=True)
    summ = META / "UCR_DataSummary_2018.csv"
    if not summ.exists():
        fetch(UCR_SUMMARY_URL, summ)
    rows = list(csv.reader(io.open(summ, encoding="utf-8-sig")))
    hdr = [h.strip() for h in rows[0]]
    meta = [dict(zip(hdr, [c.strip() for c in r])) for r in rows[1:]]
    status_path = OUT / ("inventory_archive.json" if archive else "inventory.json")
    prev = {} if archive else (json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {})
    cand = {}
    for d in meta:
        name, tr, cl, L = d["Name"], int(d["Train"]), int(d["Class"]), d["Length"]
        entry = {"type": d["Type"], "train": tr, "test": int(d["Test"]), "classes": cl, "length_meta": L, "group": GROUP_OF.get(name)}
        if not (tr >= 100 and L.isdigit() and tr / cl >= 30):
            continue
        if name in SEALED:
            entry["status"] = "EXCLUDED_SEALED"
        elif GROUP_OF.get(name) in ("G_gunpoint", "G_powercons"):
            entry["status"] = "EXCLUDED_WIRING_RELATED"
        cand[name] = entry
    for name, e in cand.items():
        if e.get("status"):
            continue
        old = prev.get("datasets", {}).get(name, {})
        if old.get("status") in ("QUALIFIED", "FAILED_GATE", "FAILED_FORMAT"):
            cand[name] = old
            continue
        zp = archive if archive else LOCAL / (name + ".zip")
        if not zp.exists():
            zp = DL / (name + ".zip")
            if not zp.exists():
                if not download:
                    e["status"] = "NOT_DOWNLOADED"
                    continue
                try:
                    e["download"] = fetch(ZIP_URL % name, zp)
                    print("[inventory] downloaded %s %.1f MB %.0fs" % (name, e["download"]["bytes"] / 2 ** 20, e["download"]["seconds"]), flush=True)
                except Exception as exc:                      # noqa: BLE001
                    e["status"] = "DOWNLOAD_FAILED"
                    e["error"] = "%s: %s" % (type(exc).__name__, exc)
                    wjson(status_path, {"datasets": cand})
                    continue
        e["zip"] = str(zp.relative_to(REPO))
        try:
            rec = cad.read_ucr_train(zp, name)
        except Exception as exc:                              # noqa: BLE001
            e["status"] = "FAILED_FORMAT"
            e["error"] = "%s: %s" % (type(exc).__name__, exc)
            wjson(status_path, {"datasets": cand})
            continue
        sp = cad.split_fit_feedback(rec["classes"]) if rec["equal_length"] else None
        if archive and (LOCAL / (name + ".zip")).exists() and rec["equal_length"]:
            try:
                loc = cad.read_ucr_train(LOCAL / (name + ".zip"), name)
                same_shape = loc["values"] is not None and loc["values"].shape == rec["values"].shape
                e["crosscheck_local_ts"] = {"same_shape": bool(same_shape), "same_label_counts": sorted(loc["class_counts"]) == sorted(rec["class_counts"]),
                                            "max_abs_diff_znorm": float(np.abs(cad.znorm(loc["values"]) - cad.znorm(rec["values"])).max()) if same_shape else None}
            except Exception as exc:                          # noqa: BLE001
                e["crosscheck_local_ts"] = {"error": str(exc)[:200]}
        e.update({"train_member": rec["member"], "n_train_parsed": rec["n"], "length": rec["length"], "equal_length": rec["equal_length"],
                  "univariate": rec["univariate"], "has_nan": rec["has_nan"], "class_counts": rec["class_counts"], "min_class": rec["min_class"],
                  "split_per_class": sp["per_class"] if sp else None})
        reasons = []
        if not (rec["univariate"] and rec["equal_length"]):
            reasons.append("not univariate equal-length")
        if rec["has_nan"]:
            reasons.append("missing values")
        if rec["n"] < 100:
            reasons.append("TRAIN < 100")
        if rec["min_class"] < 30:
            reasons.append("smallest class %d < 30" % rec["min_class"])
        if sp and not sp["gate_fit20_fb10"]:
            reasons.append("2:1 split misses fit>=20 / feedback>=10")
        e["status"] = "QUALIFIED" if not reasons else "FAILED_GATE"
        e["fail_reasons"] = reasons
        wjson(status_path, {"datasets": cand})
    fam = {}
    for name, e in cand.items():
        f = fam.setdefault(e["type"], {"qualified": [], "failed": [], "excluded": [], "other": []})
        st = e.get("status")
        key = {"QUALIFIED": "qualified", "FAILED_GATE": "failed", "FAILED_FORMAT": "failed"}.get(st, "excluded" if st and st.startswith("EXCLUDED") else "other")
        f[key].append(name)
    for t, f in fam.items():
        groups = {}
        for n in f["qualified"]:
            groups.setdefault(cand[n]["group"] or ("solo:" + n), []).append(n)
        f["independent_sources"] = len(groups)
        f["groups"] = groups
    out = {"filter": "UCR 2018 DataSummary: Train >= 100, numeric Length, Train/Class >= 30 (necessary); then TRAIN parse: univariate, "
                     "equal length, no NaN, TRAIN >= 100, every class >= 30 (2:1 split -> fit >= 20, feedback >= 10)",
           "source_groups": {g: {"members": ms, "reason": why} for g, (ms, why) in SOURCE_GROUPS.items()}, "sealed": SEALED,
           "datasets": cand, "families": fam}
    wjson(status_path, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for flag in ("smoke", "ops", "timing", "precheck", "inventory"):
        ap.add_argument("--" + flag, action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--fit-one", nargs=3, metavar=("DS", "ARM", "SEED"))
    ap.add_argument("--outdir", default=str(OUT / "precheck"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--lanes", type=int, default=1)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--dup", action="store_true")
    ap.add_argument("--synth-one", nargs=5, metavar=("N", "T", "EPOCHS", "ARM", "OUT"))
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--scale-lanes", default="1,4,8,12")
    ap.add_argument("--scale-arm", default="Jitter")
    ap.add_argument("--archive", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if a.smoke:
            r = smoke()
            print(json.dumps({k: v["pass"] for k, v in r.items() if k != "_all_pass"}, indent=1), "\nALL PASS:", r["_all_pass"])
        if a.ops:
            ops_table()
            print("ops table written")
        if a.timing:
            for r in timing():
                print("T=%5d bs=%2d  %.2f ms/step" % (r["T"], r["batch"], r["ms_per_step"]))
        if a.precheck:
            precheck()
        if a.fit_one:
            run_fit(a.fit_one[0], a.fit_one[1], int(a.fit_one[2]), Path(a.outdir), a.tag)
        if a.parallel:
            parallel_precheck([int(g) for g in a.gpus.split(",")], a.lanes, [int(x) for x in a.seeds.split(",")], a.dup)
        if a.synth_one:
            n, T, ep, arm, op_ = a.synth_one
            synth_one(int(n), int(T), int(ep), arm, Path(op_))
        if a.scale:
            scale_test(int(a.gpus.split(",")[0]), [int(x) for x in a.scale_lanes.split(",")], a.scale_arm)
        if a.inventory:
            inv = inventory(download=not a.no_download, archive=Path(a.archive).resolve() if a.archive else None)
            for t, f in sorted(inv["families"].items()):
                print("%-12s qualified %2d  independent %2d  failed %2d  excluded %2d  other %2d" % (
                    t, len(f["qualified"]), f["independent_sources"], len(f["failed"]), len(f["excluded"]), len(f["other"])))
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
