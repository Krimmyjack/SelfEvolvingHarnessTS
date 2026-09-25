"""DEV-CLS-AUG-OFFLINE-SKILL (docs/DEV_CLS_AUG_OFFLINE_SKILL_TASK_2026-09-25.md): cross-dataset classification transfer of the offline
experience-learning mechanism. This file holds the 0-LLM part: roster freeze, TRAIN extraction, the frozen program grid and its GPU launcher.

  --roster            freeze docs/CLS_AUG_ROSTER_V1.json from the inventory (representatives and roles by frozen seed / exposure rule)
  --prep              extract TRAIN arrays of the 24 representatives from UCRArchive_2018.zip (TEST members are never opened)
  --plan              job list = Source grid (8 datasets x 11 programs x 3 seeds, one stratified 2:1 TRAIN split) + Test None arm +
                      Test frozen-random arm (full TRAIN) + every job later appended with --enqueue (Select / Test deliveries)
  --enqueue FILE      append jobs [{kind, ds, prog, seed}] (deduplicated against existing jobs)
  --grid              launcher: idle GPUs only (no other user's process on the card, re-checked every 30 s), N lanes per GPU, longest first
  --status            progress summary
  --worker fb|full    one fit (internal): fb = fit part -> feedback part of the frozen 2:1 split; full = official TRAIN, model saved
Test models are trained on the full official TRAIN and saved; TEST is read only after every arm's delivery is frozen (not in this file).
Output _scratch/dev_cls_aug_offline_skill/.
"""
from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("CLS_AUG_OUT", str(REPO / "_scratch" / "dev_cls_aug_offline_skill")))
SMOKE_EPOCHS = int(os.environ.get("CLS_AUG_EPOCHS", "0"))   # smoke only; 0 = the frozen 2000
ROSTER = REPO / "docs" / "CLS_AUG_ROSTER_V1.json"
INVENTORY = REPO / "_scratch" / "dev_cls_aug_wiring" / "inventory_archive.json"
ARCHIVE = REPO / "data" / "ucr_archive_2018" / "UCRArchive_2018.zip"
ROSTER_SEED = 2026092502
SEEDS = (0, 1, 2)
RANDOM_ARM_SEED = 2026092503
# Device per dataset (decided 13:20 before any fit of these datasets existed, by shape only): short-series / small datasets train on the
# server CPU, the rest on idle GPUs. Every fit of one dataset (all arms, programs, seeds) and its TEST inference use the same device class.
CPU_DATASETS = {"ECG200", "SyntheticControl"}
# 14:05 revision (user: use the other user's GPUs lightly, move CPU work to GPU): only the two datasets that already had CPU results stay on
# the CPU; Crop, Yoga, SmoothSubspace, Ham, Freezer (no finished fit) move to the GPU class (all six cards are the same RTX 5880 model; same-seed
# fits were bit-identical across cards in the wiring run).           # program draw for the random arm; separate from training seeds

# Official UCR TEST sets opened by earlier lines (artifacts/functional/e2/*_report.json with *_opened_once = true, dataset_evidence;
# evaluation/functional/run_e2_capstone_epilepsy2.py; _scratch/convert_ucr_ts_to_txt_zip.py for BinaryHeartbeat).
EXPOSED_TEST = {
    "source_task_context_label_evidence_witness_report.json": ["Coffee", "ECG200", "FordA", "GunPoint"],
    "source_action_credit_candidate_ordering_report.json": ["DistalPhalanxOutlineCorrect", "Earthquakes", "ShapeletSim", "Wine"],
    "source_integrated_context_harness_evolution_report.json": ["Computers", "HandOutlines", "PowerCons", "SemgHandGenderCh2", "WormsTwoClass", "Yoga"],
    "source_task_risk_action_credit_transfer_report.json": ["BeetleFly", "ECGFiveDays", "TwoLeadECG", "Wafer"],
    "source_task_risk_confirmation_adaptation_report.json": ["FreezerSmallTrain", "Ham", "Herring", "Strawberry"],
    "run_e2_capstone_epilepsy2.py": ["Epilepsy2"],
    "convert_ucr_ts_to_txt_zip.py": ["BinaryHeartbeat"],
}
EXPOSED = {n for v in EXPOSED_TEST.values() for n in v}


def wjson(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    tmp.replace(path)


def rjson(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, str(REPO / "methods" / "ttha" / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------------------------------------- roster
def make_roster() -> dict:
    inv = rjson(INVENTORY)
    groups_all = {g: set(v["members"]) for g, v in inv["source_groups"].items()}
    src = {}
    for n, e in sorted(inv["datasets"].items()):
        if e.get("status") != "QUALIFIED":
            continue
        g = e.get("group") or ("solo:" + n)
        src.setdefault(g, {"type": e["type"], "qualified_members": []})["qualified_members"].append(n)
    rng = np.random.default_rng(ROSTER_SEED)
    rows = []
    for g in sorted(src):
        s = src[g]
        members = sorted(s["qualified_members"])
        rep = members[int(rng.integers(len(members)))] if len(members) > 1 else members[0]
        all_members = sorted(groups_all.get(g, set(members)))
        hit = sorted(set(all_members) & EXPOSED)
        e = inv["datasets"][rep]
        rows.append({"source": g, "type": s["type"], "representative": rep, "qualified_members": members, "group_members_all": all_members,
                     "test_exposed_members": hit, "exposure": "TEST_OPENED_IN_EARLIER_LINE" if hit else "CLEAN",
                     "n_train": e["n_train_parsed"], "length": e["length"], "n_classes": len(e["class_counts"]), "class_counts": e["class_counts"]})
    clean = [r for r in rows if r["exposure"] == "CLEAN"]
    dirty = [r for r in rows if r["exposure"] != "CLEAN"]
    order = rng.permutation(len(dirty))
    for r in clean:
        r["role"] = "Test"
    for i, j in enumerate(order):
        dirty[j]["role"] = "Source" if i < 8 else "Select"
    out = {"version": "CLS_AUG_ROSTER_V1", "frozen": time.strftime("%Y-%m-%d %H:%M"), "seed": ROSTER_SEED,
           "rules": {"eligibility": "UCR 2018 univariate, equal length, no missing values, TRAIN >= 100, every class >= 30 (inventory_archive.json)",
                     "one_representative_per_source": "seeded draw among the qualified members of each conservative same-source group",
                     "roles": "EXECUTOR DEFAULT (option a, pending Planner review): the 12 sources with no official TEST ever opened -> Test; the 12 "
                              "sources with an opened TEST (any group member) -> 8 Source / 4 Select by seeded permutation",
                     "known_shift": "all 12 Test sources are multi-class (3-42 classes); the Source/Select sources are mostly binary"},
           "exposed_test_provenance": EXPOSED_TEST, "sources": rows,
           "counts": {k: sum(1 for r in rows if r["role"] == k) for k in ("Source", "Select", "Test")}}
    if ROSTER.exists():
        raise SystemExit("roster already frozen: %s" % ROSTER)
    wjson(ROSTER, out)
    return out


# ----------------------------------------------------------------------------------------------------------------------------- data
def prep() -> dict:
    cad = _load("cls_aug_data")
    ros = rjson(ROSTER)
    rec = {}
    for r in ros["sources"]:
        name = r["representative"]
        p = OUT / "data" / (name + "_TRAIN.npz")
        d = cad.read_ucr_train(ARCHIVE, name)
        assert d["member"].endswith("_TRAIN.tsv") and d["equal_length"] and not d["has_nan"] and d["min_class"] >= 30, name
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, values=d["values"], classes=d["classes"], label_names=np.array(d["label_names"]))
        rec[name] = {"role": r["role"], "n": d["n"], "length": d["length"], "class_counts": d["class_counts"], "member": d["member"]}
    wjson(OUT / "data" / "prep.json", {"archive": str(ARCHIVE.relative_to(REPO)), "note": "TRAIN members only", "datasets": rec})
    return rec


def load_train(name: str):
    z = np.load(OUT / "data" / (name + "_TRAIN.npz"))
    return z["values"], z["classes"], [str(x) for x in z["label_names"]]


# ----------------------------------------------------------------------------------------------------------------------------- plan
def est_minutes(n: int, T: int, aug: bool) -> float:
    bs = max(1, int(min(n / 10, 16)))
    ms = 2.0 + 0.6 * max(0.0, (T - 1500) / 1200) if bs >= 16 else 2.0
    return math.ceil(n / bs) * 2000 * ms / 60000 * (2.0 if aug else 1.0)


def job_path(kind, ds, prog, seed, fold=None) -> Path:
    return OUT / kind / ds / ("%s__s%d.json" % (prog, seed))


def job_name(j) -> str:
    return "%s__%s__%s__s%d" % (j["kind"], j["ds"], j["prog"], j["seed"])


def random_arm() -> dict:
    """Frozen before any outcome: one program per Test dataset, uniform over the 11 programs (None included)."""
    path = OUT / "random_arm.json"
    if path.exists():
        return rjson(path)
    ca = _load("cls_aug")
    rng = np.random.default_rng(RANDOM_ARM_SEED)
    test = sorted(r["representative"] for r in rjson(ROSTER)["sources"] if r["role"] == "Test")
    out = {"seed": RANDOM_ARM_SEED, "rule": "one uniform draw over the 11 programs per Test dataset, same program for the 3 training seeds",
           "frozen": time.strftime("%F %T"), "draws": {ds: ca.PROGRAMS[int(rng.integers(len(ca.PROGRAMS)))] for ds in test}}
    wjson(path, out)
    return out


def _job(kind, r, prog, seed) -> dict:
    n = int(r["n_train"]) - (sum(c // 3 for c in r["class_counts"]) if kind == "fb" else 0)
    return {"kind": kind, "ds": r["representative"], "prog": prog, "seed": seed, "fold": None, "role": r["role"],
            "device": "cpu" if r["representative"] in CPU_DATASETS else "gpu", "T": int(r["length"]), "est_min": est_minutes(n, r["length"], prog != "None")}


def plan() -> list:
    ca = _load("cls_aug")
    ros = {r["representative"]: r for r in rjson(ROSTER)["sources"]}
    ra = random_arm()["draws"]
    jobs, seen = [], set()

    def add(j):
        k = job_name(j)
        if k not in seen:
            seen.add(k)
            jobs.append(j)
    for ds, r in ros.items():
        for seed in SEEDS:
            if r["role"] == "Source":
                for prog in ca.PROGRAMS:
                    add(_job("fb", r, prog, seed))
            elif r["role"] == "Test":
                add(_job("full", r, "None", seed))
                add(_job("full", r, ra[ds], seed))
    extra = OUT / "jobs_extra.json"
    for e in (rjson(extra) if extra.exists() else []):
        add(_job(e["kind"], ros[e["ds"]], e["prog"], int(e["seed"])))
    jobs.sort(key=lambda j: ({"Source": 0, "Select": 1, "Test": 2}[j["role"]], -j["est_min"]))
    wjson(OUT / "plan.json", {"programs": list(ca.PROGRAMS), "seeds": list(SEEDS), "order": "Source, Select, Test; longest first within a role", "split": "cls_aug_data.split_fit_feedback (seed 2026092500)",
                              "n_jobs": len(jobs), "est_lane_hours": sum(j["est_min"] for j in jobs) / 60, "jobs": jobs})
    return jobs


def enqueue(path: str) -> int:
    ros = {r["representative"]: r for r in rjson(ROSTER)["sources"]}
    extra = OUT / "jobs_extra.json"
    cur = rjson(extra) if extra.exists() else []
    keys = {(e["kind"], e["ds"], e["prog"], int(e["seed"])) for e in cur}
    n = 0
    for e in rjson(path):
        k = (e["kind"], e["ds"], e["prog"], int(e["seed"]))
        assert e["ds"] in ros and e["kind"] in ("fb", "full") and (e["kind"] == "fb") == (ros[e["ds"]]["role"] != "Test"), e
        if k not in keys:
            keys.add(k)
            cur.append({"kind": k[0], "ds": k[1], "prog": k[2], "seed": k[3]})
            n += 1
    wjson(extra, cur)
    return n


# ----------------------------------------------------------------------------------------------------------------------------- worker
def worker(kind: str, ds: str, prog: str, seed: int, fold: int | None) -> None:
    import torch
    ca = _load("cls_aug")
    out = job_path(kind, ds, prog, seed, fold)
    if out.exists():
        return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, classes, names = load_train(ds)
    Z = ca.znorm(raw)
    C = len(names)
    P = ca.Program(prog, Z.shape[1], dev)
    p = 0.0 if P.is_identity else ca.P_PROGRAM
    ep = {"epochs": SMOKE_EPOCHS} if SMOKE_EPOCHS else {}
    t0 = time.time()
    if kind == "fb":
        sp = ca.split_fit_feedback(classes)
        tr, te = sp["fit"], sp["feedback"]
        r = ca.fit_fcn(Z[tr], classes[tr], C, seed=seed, device=dev, aug=None if P.is_identity else P, p_aug=p, **ep)
        pred = ca.predict(r["state"], C, Z[te], dev)
        extra = {"split_seed": sp["seed"], "feedback_index": te.tolist(), "feedback_pred": pred.tolist(), "feedback_true": classes[te].tolist(),
                 "feedback_metrics": ca.metrics(classes[te], pred, C)}
    else:
        r = ca.fit_fcn(Z, classes, C, seed=seed, device=dev, aug=None if P.is_identity else P, p_aug=p, **ep)
        mp = OUT / "models" / ds / ("%s__s%d.pt" % (prog, seed))
        mp.parent.mkdir(parents=True, exist_ok=True)
        torch.save(r["state"], mp)
        extra = {"model": str(mp), "n_classes": C, "label_names": names}
    curve = r.pop("loss_curve")
    r.pop("state")
    wjson(out, {"kind": kind, "ds": ds, "prog": prog, "seed": seed, "fold": fold, "program": P.describe(), "n_train": int(len(Z) if kind == "full" else len(tr)),
                "T": int(Z.shape[1]), "wall_seconds": time.time() - t0, "train": r, "loss_first": curve[0], "loss_every_100": curve[::100],
                "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch": torch.__version__, **extra})


# ----------------------------------------------------------------------------------------------------------------------------- launcher
def gpu_state(me: str) -> dict:
    import psutil
    idx = {}
    for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30).stdout.strip().splitlines():
        i, u, m = [x.strip() for x in line.split(",")]
        idx[u] = {"index": int(i), "mem_mb": int(m), "foreign": []}
    for line in subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=30).stdout.strip().splitlines():
        if not line.strip():
            continue
        u, pid = [x.strip() for x in line.split(",")]
        try:
            owner = psutil.Process(int(pid)).username()
        except Exception:                                           # noqa: BLE001 - unknown owner counts as foreign
            owner = "?"
        if owner != me and u in idx:
            idx[u]["foreign"].append({"pid": int(pid), "user": owner})
    return {v["index"]: v for v in idx.values()}


class _Adopted:
    """A worker started by an earlier launcher: polled through psutil; its return code is unknown (success = result file exists)."""

    def __init__(self, proc):
        self.proc = proc
        self.returncode = 0

    def poll(self):
        try:
            import psutil
            return None if self.proc.is_running() and self.proc.status() != psutil.STATUS_ZOMBIE else 0
        except Exception:                                           # noqa: BLE001
            return 0


class _NullLog:
    def close(self):
        pass


def adopt_running(device: str = "gpu") -> list:
    import psutil
    out = []
    for pr in psutil.process_iter(["pid", "cmdline"]):
        cl = pr.info["cmdline"] or []
        if "--worker" in cl and any(x.endswith("cls_aug_offline_skill.py") for x in cl):
            i = cl.index("--worker")
            kind, ds, prog, seed, fold = cl[i + 1:i + 6]
            try:
                cvd = pr.environ().get("CUDA_VISIBLE_DEVICES", "")
            except Exception:                                       # noqa: BLE001
                cvd = "?"
            if (device == "cpu") != (cvd == ""):
                continue
            g = "cpu" if cvd == "" else (int(cvd) if cvd.isdigit() else -1)
            j = {"kind": kind, "ds": ds, "prog": prog, "seed": int(seed), "fold": None}
            out.append((_Adopted(pr), g, job_name(j), time.time(), _NullLog(), j))
    return out


def launch(gpus: list, lanes: int, threads: int = 2, device: str = "gpu", share_lanes: int = 0, light_T: int = 320) -> None:
    """A GPU that carries another user's process gets at most share_lanes of our workers, and only light jobs (series length <= light_T),
    longest first; idle GPUs get `lanes` workers in the plan order (Source, Select, Test)."""
    me = getpass.getuser()
    adopted = adopt_running(device)
    busy = {a[2] for a in adopted}
    jobs = [j for j in rjson(OUT / "plan.json")["jobs"] if not job_path(j["kind"], j["ds"], j["prog"], j["seed"], j["fold"]).exists()
            and job_name(j) not in busy and j.get("device", "gpu") == device]
    status_file = OUT / ("grid_status.json" if device == "gpu" else "grid_status_cpu.json")
    if device == "cpu":
        gpus = ["cpu"]
        jobs.sort(key=lambda j: -j["est_min"])          # CPU pool: pure longest-first (Crop Test fits are the longest pole of the package)
    print("[grid] adopted %d running workers; %d jobs to dispatch" % (len(adopted), len(jobs)), flush=True)
    logd = OUT / "logs" / "workers"
    logd.mkdir(parents=True, exist_ok=True)
    running, done, failed = list(adopted), 0, []
    allowed, last_check, t0 = set(), 0.0, time.time()
    attempts = {}
    while jobs or running:
        if time.time() - last_check > 30:
            st = gpu_state(me) if device == "gpu" else {}
            cap = ({g: (lanes if not st[g]["foreign"] else share_lanes) for g in gpus if g in st} if device == "gpu" else {"cpu": lanes})
            allowed = {g for g, c in cap.items() if c > 0}
            last_check = time.time()
            wjson(status_file, {"time": time.strftime("%F %T"), "elapsed_min": (time.time() - t0) / 60, "pending": len(jobs),
                                             "running": [{"job": n, "gpu": g, "min": (time.time() - s) / 60} for _, g, n, s, _, _ in running],
                                             "done_this_run": done, "failed": failed, "allowed_gpus": sorted(allowed),
                                             "gpu_foreign": {g: st[g]["foreign"] for g in st if st[g]["foreign"]}, "device": device,
                                             "lane_cap": {str(k): v for k, v in cap.items()}})
        load = {g: sum(1 for _, gg, *_ in running if gg == g) for g in gpus}
        blocked = set()
        while jobs:
            free = [g for g in sorted(allowed, key=str) if load.get(g, 0) < cap.get(g, 0) and g not in blocked]
            if not free:
                break
            g = min(free, key=lambda x: load[x])
            if device == "gpu" and st.get(g, {}).get("foreign"):
                light = [x for x in jobs if x.get("T", 10 ** 9) <= light_T]
                if not light:
                    blocked.add(g)
                    continue
                j = max(light, key=lambda x: x["est_min"])
                jobs.remove(j)
            else:
                j = jobs.pop(0)
            name = job_name(j)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if g == "cpu" else str(g), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            lf = open(logd / (name + ".log"), "w")
            argv = [sys.executable, str(Path(__file__).resolve()), "--worker", j["kind"], j["ds"], j["prog"], str(j["seed"]),
                    str(-1 if j["fold"] is None else j["fold"])]
            running.append((subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)), g, name, time.time(), lf, j))
            load[g] += 1
        time.sleep(2)
        for item in list(running):
            pr, g, name, s, lf, j = item
            if pr.poll() is None:
                continue
            lf.close()
            running.remove(item)
            ok = pr.returncode == 0 and job_path(j["kind"], j["ds"], j["prog"], j["seed"], j["fold"]).exists()
            if ok:
                done += 1
            else:
                attempts[name] = attempts.get(name, 0) + 1
                if attempts[name] <= 2:
                    jobs.insert(0, j)
                else:
                    failed.append(name)
                print("[grid] %s rc=%s attempt %d" % (name, pr.returncode, attempts[name]), flush=True)
    wjson(status_file, {"time": time.strftime("%F %T"), "elapsed_min": (time.time() - t0) / 60, "pending": 0, "running": [],
                        "done_this_run": done, "failed": failed, "finished": True, "device": device})
    print("[grid] finished: done %d failed %d" % (done, len(failed)), flush=True)


def score_test() -> dict:
    """The one-shot TEST opening: only after freeze/deliveries.json exists and every delivered (dataset, program, seed) model is on disk."""
    import zipfile
    import torch
    ca = _load("cls_aug")
    fr = rjson(OUT / "freeze" / "deliveries.json")
    need = {(ds, p, s) for ds, d in fr["deliveries"].items() for p in set(d.values()) for s in SEEDS}
    missing = [k for k in need if not (OUT / "models" / k[0] / ("%s__s%d.pt" % (k[1], k[2]))).exists()]
    if missing:
        raise SystemExit("refusing to open TEST: %d delivered models missing, e.g. %s" % (len(missing), missing[:3]))
    marker = OUT / "test_opened.json"
    wjson(marker, {"opened": time.strftime("%F %T"), "datasets": sorted(fr["deliveries"]), "freeze_file_frozen": fr["frozen_local"]})
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for ds in sorted(fr["deliveries"]):
            _, _, names = load_train(ds)
            raw = z.read("UCRArchive_2018/%s/%s_TEST.tsv" % (ds, ds), pwd=b"someone").decode("utf-8").strip().splitlines()
            rows = [line.split("\t") for line in raw if line.strip()]
            y = np.array([names.index(r[0].strip()) for r in rows])
            X = ca.znorm(np.array([[float(v) for v in r[1:]] for r in rows]))
            out[ds] = {"n_test": int(len(y))}
            dv = torch.device("cpu") if ds in CPU_DATASETS else dev
            for prog in sorted({p for p in fr["deliveries"][ds].values()}):
                out[ds][prog] = {}
                for s in SEEDS:
                    st = torch.load(OUT / "models" / ds / ("%s__s%d.pt" % (prog, s)), map_location=dv)
                    pred = ca.predict(st, len(names), X, dv)
                    out[ds][prog][str(s)] = ca.metrics(y, pred, len(names))
            print("[test] %s scored (%d programs)" % (ds, len(out[ds]) - 1), flush=True)
    wjson(OUT / "test_scores.json", {"scored": time.strftime("%F %T"), "archive_member": "UCRArchive_2018/<ds>/<ds>_TEST.tsv", "scores": out})
    return out


def status() -> None:
    pl = rjson(OUT / "plan.json")["jobs"]
    fin = [j for j in pl if job_path(j["kind"], j["ds"], j["prog"], j["seed"], j["fold"]).exists()]
    by = {}
    for j in pl:
        k = (j["role"], j["kind"])
        by.setdefault(k, [0, 0, 0.0, 0.0])
        by[k][1] += 1
        by[k][3] += j["est_min"]
    for j in fin:
        k = (j["role"], j["kind"])
        by[k][0] += 1
        by[k][2] += j["est_min"]
    for k, v in sorted(by.items()):
        print("%-7s %-4s %4d / %4d jobs  (est %.0f / %.0f lane-min)" % (k[0], k[1], v[0], v[1], v[2], v[3]))
    st = OUT / "grid_status.json"
    if st.exists():
        s = rjson(st)
        print("launcher:", s.get("time"), "elapsed %.0f min" % s.get("elapsed_min", 0), "running", len(s.get("running", [])),
              "allowed", s.get("allowed_gpus"), "foreign", s.get("gpu_foreign"), "failed", s.get("failed"))


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in ("roster", "prep", "plan", "grid", "status", "score_test"):
        ap.add_argument("--" + f.replace("_", "-"), action="store_true")
    ap.add_argument("--worker", nargs=5, metavar=("KIND", "DS", "PROG", "SEED", "FOLD"))
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--enqueue", default="")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--grid-cpu", action="store_true")
    ap.add_argument("--cpu-procs", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--share-lanes", type=int, default=0)
    ap.add_argument("--light-T", type=int, default=320)
    a = ap.parse_args()
    try:
        if a.roster:
            r = make_roster()
            for row in r["sources"]:
                print("%-7s %-10s %-26s %-30s n=%-5d T=%-5d C=%-3d %s" % (row["role"], row["type"], row["source"], row["representative"], row["n_train"],
                                                                        row["length"], row["n_classes"], row["exposure"]))
        if a.prep:
            print(json.dumps({k: v["n"] for k, v in prep().items()}))
        if a.plan:
            jobs = plan()
            print("jobs", len(jobs), "est lane-hours %.1f" % (sum(j["est_min"] for j in jobs) / 60))
        if a.enqueue:
            print("enqueued", enqueue(a.enqueue))
        if a.grid:
            launch([int(g) for g in a.gpus.split(",")], a.lanes, share_lanes=a.share_lanes, light_T=a.light_T)
        if a.grid_cpu:
            launch([], a.cpu_procs, threads=a.threads, device="cpu")
        if a.status:
            status()
        if a.score_test:
            score_test()
        if a.worker:
            k, ds, prog, seed, fold = a.worker
            worker(k, ds, prog, int(seed), None if int(fold) < 0 else int(fold))
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
