"""Data-readiness profile of the batch construction base (docs/DEV_DATA_READINESS_DOMAIN_SKILL_V1_TASK_2026-09-17.md §2-§4, §7).

Same geometry (L=192, H=48, T=672, C_A/C_B/E origins), the same shared MLP / train_arm / seed discipline and the same
stage row permissions (context.STAGE_ROWS) as the augmentation profile, with natural missing values kept:

  data      RD01 = KDD Cup 2018 with-missing TSF (32 frozen series), RD02 = Beijing multi-site PM2.5 (12 stations);
            a stage converts only the rows it is authorized for into numbers
  parents   stride-1 windows inside T; legal iff X has >= 32 finite values and y is fully finite; common to every arm
  scaler    per-entity mean/std (ddof=0) of the finite raw T values, floor 1e-6; identical for every arm
  material  one closed preparation program per entity, applied to each legal X window ONLY (never to y, never to a
            pre-filled T); the empty program is Baseline-Linear (existing impute_linear strength=1)
  training  processed X + raw finite y in ONE shared MLP (train.train_arm, no child view); pool = actual legal count
  serving   the same frozen per-entity program on the 192 inputs before each origin
  scoring   missing-aware normalized MSE on observed future cells only; an entity below 25% coverage makes the block
            NOT_SCORABLE (macro null); a shadow score feeds the Baseline-Linear inputs to the same model

Operators are executed through the existing runtime.executor.run_pipeline on the registry callables. No LLM, no hashing
and no selection rule lives here. The augmentation profile (spec/data/context/materials) is untouched.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import itertools
import json
import math
import platform
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import commit as _commit
from . import context, data, policy, spec, train

REPO = Path(__file__).resolve().parents[3]
PROFILE = "data_readiness"
EXPOSURE = "EXPOSED_DEVELOPMENT"
L, H, TRAIN_SPAN = spec.L, spec.H, spec.TRAIN_SPAN
N_STARTS = spec.N_PARENTS                     # 433 stride-1 window starts inside T
MIN_X_FINITE = 32                             # legal parent: >= 32 finite X values and a fully finite y
MIN_T_FINITE = 336                            # eligibility: >= 336/672 finite T values per entity
MIN_LEGAL_PARENTS = 32
COVERAGE_MIN = 0.25                           # per entity, per block: observed cells / prediction cells
FIT_MODULE = "methods.ttha.batch_base.readiness"

# --------------------------------------------------------------------------- datasets (EXPOSED_DEVELOPMENT, read-only)
RD01_ROSTER = ["T173", "T175", "T177", "T179", "T180", "T187", "T189", "T19", "T191", "T192", "T193", "T195", "T197", "T198", "T199", "T201",
               "T203", "T204", "T205", "T207", "T209", "T21", "T210", "T211", "T212", "T213", "T214", "T215", "T216", "T218", "T220", "T221"]
RD02_ROSTER = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan",
               "Wanliu", "Wanshouxigong"]
PACKAGE_DATA = REPO / "_scratch" / "dev_data_readiness_domain_skill_v1" / "data"
DATASETS = {
    "RD01": {"kind": "tsf_zip", "path": REPO / "data" / "kdd2018" / "raw" / "kdd_cup_2018_dataset_with_missing_values.zip",
             "missing_token": "?", "roster": RD01_ROSTER,
             "roster_rule": "UIDs sorted as strings, candidates from index 80; every planned T: >=336/672 finite, finite std>1e-6, "
                            ">=32 legal parents; first 32 in sort order"},
    "RD02": {"kind": "prsa_csv", "path": REPO / "_scratch" / "train3_preflight" / "data" / "prsa" / "PRSA_Data_20130301-20170228",
             "archive": REPO / "data" / "benchmark_v0" / "raw" / "beijing_multisite" / "beijing_multi_site_air_quality.zip",
             "file": "PRSA_Data_%s_20130301-20170228.csv", "field": "PM2.5", "missing_token": "NA", "first_time": (2013, 3, 1, 0),
             "roster": RD02_ROSTER, "roster_rule": "all 12 PM2.5 stations sorted by station name"},
}
# RD01B: a new fixed cohort of the same KDD with-missing file (DEV-DATA-READINESS-PROFILE-TRANSFER §2.1); RD01 and its jobs are unchanged.
RD01B_ROSTER = ["T172", "T178", "T190", "T196", "T202", "T219", "T22", "T222", "T223", "T224", "T225", "T226", "T227", "T228", "T229", "T23",
                "T230", "T233", "T234", "T235", "T236", "T239", "T24", "T240", "T241", "T243", "T244", "T246", "T247", "T25", "T254", "T256"]
DATASETS["RD01B"] = {**DATASETS["RD01"], "roster": RD01B_ROSTER,
                     "roster_rule": "UIDs sorted as strings, candidates from index 80, excluding the 32 RD01 UIDs; length >= 9744; every planned T "
                                    "(4560, 5760, 6960, 8160, 9360): >=336/672 finite, finite std>1e-6, >=32 legal parents; first 32 in sort order"}
JOB_T = {"S1": 3360, "S2": 4560, "V1": 5760, "T1": 6960, "T2": 8160}
# Later test batches registered explicitly per dataset (DEV-DATA-READINESS-DOMAIN-WORKFLOW-EVOLVE-R2 §2.3); JOB_T above is unchanged.
# STP1: the post-freeze check batch of DEV-DATA-READINESS-TRAINING-LENGTH §3.2.
# L1-L4: the four unscored 2015 follow-up batches of DEV-TEMPO-AUG-WORKFLOW-SKILL §2 (RD02 only; T legality and block coverage
# were checked mask-only in _scratch/dev_tempo_aug_source_alignment/data_availability/availability.json).
# L5-L7: the three reserved 2015 batches of DEV-TEMPO-AUG-WORKFLOW-LEARNING-LOOP §2 (L7's E ends exactly at the sealed row 24864).
EXTRA_JOB_T = {"RD02": {"T3": 9360, "T4": 10560, "Q1": 11760, "Q2": 12960, "STP1": 14160, "L1": 15360, "L2": 16560, "L3": 18960, "L4": 20160,
                        "L5": 21360, "L6": 22560, "L7": 24480},
               "RD01B": {"S1": 4560, "S2": 5760, "V1": 6960, "Q1": 8160, "Q2": 9360, "STP1": 10320}}
SOURCE_NAMES = ("kdd", "beijing", "london", "prsa", "rd01", "rd02", "electricity", "traffic") + tuple(s.lower() for s in RD02_ROSTER)
_JOB_RE = re.compile(r"^(RD0[12]|RDX)_(S1|S2|V1|T1|T2)$")
_EXTRA_JOB_RE = re.compile(r"^(RD02|RD01B)_(S1|S2|V1|T3|T4|Q1|Q2|STP1|L1|L2|L3|L4|L5|L6|L7)$")
_JOB_TOKEN = re.compile(r"RD0\d|RDX|(?<![A-Za-z0-9])(?:S1|S2|V1|T1|T2|T3|T4|Q1|Q2|STP1|L1|L2|L3|L4|L5|L6|L7)(?![A-Za-z0-9])")

CONSUMER = {**spec.CONSUMER,
            "loss": "pooled MSE on normalized targets over all legal parent windows of the batch (no parent/child view mixing)",
            "normalization": "per-entity mean/std (ddof=0) of the finite raw T values; identical for every plan; never re-estimated from prepared arrays",
            "metric": "missing-aware normalized MSE: per entity over the observed origin x horizon cells of a block, then equal-weight entity mean"}


# --------------------------------------------------------------------------- closed action table (task §4)
OPS = {
    "impute_linear": {"strength": [1.0]},
    "impute_ema": {"alpha": [0.1, 0.3]},
    "impute_fft": {"cutoff_ratio": [0.1, 0.2]},
    "period_complete": {"period": [24, 168]},
    "period_median_complete": {"period": [24], "cycles": [3], "min_donors": [2]},
    "hampel_filter": {"window": [5, 7], "n_sigmas": [3.0, 4.0]},
    "outlier_iqr": {"k": [1.5, 3.0]},
    "outlier_mad": {"k": [3.5, 5.0]},
    "denoise_median": {"window": [3, 5], "strength": [0.5, 1.0]},
    "denoise_savgol": {"window": [5, 11], "order": [2]},
}
INT_PARAMS = frozenset({"period", "cycles", "min_donors", "window", "order"})
IMPUTERS = frozenset({"impute_linear", "impute_ema", "impute_fft", "period_complete", "period_median_complete"})
MAX_STEPS, MAX_RULES = 3, 8
OP_SEMANTICS = {
    "impute_linear": "Fills each missing X value by linear interpolation between the nearest observed values; leading/trailing gaps take the nearest observed value. Observed values unchanged. This is also the minimum completion of every plan.",
    "impute_ema": "Missing positions receive an exponential moving average (weight alpha on the current point) run forward over a linearly pre-filled copy of the window. Observed values unchanged.",
    "impute_fft": "Missing positions receive a low-pass reconstruction (first cutoff_ratio of rFFT bins kept) of a linearly pre-filled copy. Observed values unchanged; a window without gaps is unchanged.",
    "period_complete": "Walking forward, a missing position takes the value `period` hours earlier if that position holds a value (an earlier fill can be reused); the rest is linearly interpolated. With period=168 only the last 24 of the 192 inputs can have a donor inside the window.",
    "period_median_complete": "A missing position takes the median of the ORIGINALLY observed values 24, 48 and 72 hours earlier inside the same 192-point window when at least 2 exist; otherwise that point is linearly interpolated (recorded fallback). Observed values unchanged; the first 48 inputs can never have 2 donors.",
    "hampel_filter": "Linear pre-fill, then a centered rolling median/MAD (odd window, mirrored boundary); points with |x - median| > n_sigmas x 1.4826 x MAD are replaced by the local median; zero-MAD neighbourhoods are never flagged. Changes observed values.",
    "outlier_iqr": "Linear pre-fill, then every point is clipped to [Q1 - k IQR, Q3 + k IQR] of that 192-point window. Changes observed values beyond the bounds.",
    "outlier_mad": "Linear pre-fill, then every point is clipped to median +/- k x 1.4826 x MAD of that window; MAD ~ 0 leaves the window unchanged. Changes observed values beyond the bounds.",
    "denoise_median": "Linear pre-fill, then a centered moving median (odd window, reflect boundary); strength blends x + strength x (median - x). Changes observed values.",
    "denoise_savgol": "Linear pre-fill, then a Savitzky-Golay filter (window, polynomial order 2, polynomial end fits) when scipy is present, else a moving average (recorded dependency fallback). Changes observed values.",
}
CANONICAL_RULE = ("Programs are canonicalized before materials are built: every impute step after the first step is removed (the input is "
                  "already gap-free, so it is an exact no-op) and a first impute_linear(strength=1) is removed (every later operator and the "
                  "final minimum completion apply the same linear fill). Plans with identical canonical assignments are the same material "
                  "(alias; no second fit).")


def _operator_contracts() -> dict:
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA
    keys = ("category", "destructive", "preserves_observed", "fallback_policy", "requires_dependency", "dependency_policy")
    return {name: {k: OPERATOR_METADATA[name][k] for k in keys} for name in OPS}


def action_table() -> dict:
    contracts = _operator_contracts()
    return {"max_steps_per_program": MAX_STEPS, "max_rules": MAX_RULES, "empty_program": "Baseline-Linear (minimum linear completion only)",
            "canonicalization": CANONICAL_RULE,
            "ops": {name: {"parameters": OPS[name], "registry_contract": contracts[name], "semantics": OP_SEMANTICS[name]} for name in OPS},
            "execution": "Each step runs on ONE 192-point raw X window through the existing executor; the parent's y is never passed. "
                         "After the program: a residual gap is linearly completed (recorded); a window with no observed value at all is "
                         "filled with the entity's finite-T mean (recorded; serving only, a legal training window has >=32 observed values); "
                         "Inf, a length change or an operator error rejects the plan instead of passing as identity."}


# --------------------------------------------------------------------------- loading (authorized rows only)
_TSF_CACHE: dict = {}


def _tsf_series(path: Path, names) -> dict:
    key = (str(path), tuple(names))
    if key not in _TSF_CACHE:
        want, out = set(names), {}
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            if len(members) != 1:
                raise RuntimeError("unexpected TSF archive layout")
            with archive.open(members[0]) as handle:
                in_data = False
                for line in io.TextIOWrapper(handle, encoding="utf-8"):
                    if not in_data:
                        in_data = line.strip().lower() == "@data"
                        continue
                    name = line.split(":", 1)[0]
                    if name in want:
                        fields = line.rstrip().split(":")
                        out[name] = {"start": fields[4], "tokens": fields[-1].split(",")}   # text only; numbers are made per slice
        if set(out) != want:
            raise RuntimeError("roster series missing from the archive: %s" % sorted(want - set(out)))
        _TSF_CACHE[key] = out
    return _TSF_CACHE[key]


def _prsa_dir(ds: dict) -> Path:
    if ds["path"].exists():
        return ds["path"]
    target = PACKAGE_DATA / "prsa" / ds["path"].name
    if not target.exists():                    # same-source local archive (nested zip); no download
        with zipfile.ZipFile(ds["archive"]) as outer:
            inner = next(n for n in outer.namelist() if n.lower().endswith(".zip"))
            with zipfile.ZipFile(io.BytesIO(outer.read(inner))) as z:
                for station in ds["roster"]:
                    member = next(n for n in z.namelist() if n.endswith(ds["file"] % station))
                    target.mkdir(parents=True, exist_ok=True)
                    (target / (ds["file"] % station)).write_bytes(z.read(member))
    return target


@dataclass
class RSlice:
    dataset: str
    row_start: int
    row_end: int
    values: np.ndarray          # (rows, N) raw float64, NaN = missing
    hours: np.ndarray           # (rows, N) clock hour of each row per entity
    roster: list

    def rows(self, a: int, b: int) -> np.ndarray:
        if a < self.row_start or b > self.row_end or a >= b:
            raise PermissionError("row access [%d,%d) outside loaded slice [%d,%d)" % (a, b, self.row_start, self.row_end))
        return self.values[a - self.row_start: b - self.row_start]

    def hours_of(self, a: int, b: int) -> np.ndarray:
        if a < self.row_start or b > self.row_end or a >= b:
            raise PermissionError("timestamp access outside loaded slice")
        return self.hours[a - self.row_start: b - self.row_start]


def load_slice(dataset: str, row_start: int, row_end: int) -> RSlice:
    ds = DATASETS[dataset]
    if row_start < 0 or row_start >= row_end:
        raise ValueError("bad slice")
    n, roster = row_end - row_start, list(ds["roster"])
    values, hours = np.empty((n, len(roster))), np.empty((n, len(roster)), dtype=np.int64)
    if ds["kind"] == "tsf_zip":
        series = _tsf_series(ds["path"], roster)
        for e, name in enumerate(roster):
            toks = series[name]["tokens"]
            if len(toks) < row_end:
                raise RuntimeError("series shorter than the authorized slice")
            values[:, e] = [np.nan if tok == ds["missing_token"] else float(tok) for tok in toks[row_start:row_end]]
            start = _dt.datetime.strptime(series[name]["start"], "%Y-%m-%d %H-%M-%S")
            hours[:, e] = [(start + _dt.timedelta(hours=r)).hour for r in range(row_start, row_end)]
    elif ds["kind"] == "prsa_csv":
        base, folder = _dt.datetime(*ds["first_time"]), _prsa_dir(ds)
        for e, station in enumerate(roster):
            with open(folder / (ds["file"] % station), newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                iy, im, iday, ih, iv, ist = (header.index(k) for k in ("year", "month", "day", "hour", ds["field"], "station"))
                got = 0
                for i, row in enumerate(itertools.islice(reader, row_end)):
                    if i < row_start:
                        continue
                    ts = _dt.datetime(int(row[iy]), int(row[im]), int(row[iday]), int(row[ih]))
                    if ts != base + _dt.timedelta(hours=i) or row[ist] != station:
                        raise RuntimeError("time/station binding broken at absolute row %d of %s" % (i, station))
                    tok = row[iv]
                    values[i - row_start, e] = np.nan if tok == ds["missing_token"] else float(tok)
                    hours[i - row_start, e] = ts.hour
                    got += 1
                if got != n:
                    raise RuntimeError("station file shorter than the authorized slice")
    elif ds["kind"] == "synthetic":                # smoke only (registered in-process); never a real run
        values[:], hours[:] = ds["values"][row_start:row_end], ds["hours"][row_start:row_end]
    else:
        raise ValueError("unknown dataset kind")
    if np.isinf(values).any():
        raise RuntimeError("infinite raw values")
    return RSlice(dataset, row_start, row_end, values, hours, roster)


# --------------------------------------------------------------------------- jobs, scaler, legal parents
@dataclass(frozen=True)
class Job:
    dataset: str
    job_id: str
    t: int
    roster_override: None = None               # the population is frozen per dataset (no override path)
    train_range: tuple = field(init=False)
    c_a: tuple = field(init=False)
    c_b: tuple = field(init=False)
    e: tuple = field(init=False)
    material_rows: tuple = field(init=False)
    evaluate_rows: tuple = field(init=False)
    c_b_rows: tuple = field(init=False)
    e_input_rows: tuple = field(init=False)
    e_target_rows: tuple = field(init=False)

    def __post_init__(self):
        t = self.t
        for k, v in (("train_range", (t - TRAIN_SPAN, t)), ("c_a", (t, t + 48)), ("c_b", (t + 96, t + 144)),
                     ("e", (t + 192, t + 240, t + 288, t + 336))):
            object.__setattr__(self, k, v)
        object.__setattr__(self, "material_rows", (t - TRAIN_SPAN, t))
        object.__setattr__(self, "evaluate_rows", (t - TRAIN_SPAN, max(self.c_a) + H))
        object.__setattr__(self, "c_b_rows", (min(self.c_b) - L, max(self.c_b) + H))
        object.__setattr__(self, "e_input_rows", (min(self.e) - L, max(self.e)))
        object.__setattr__(self, "e_target_rows", (min(self.e), max(self.e) + H))
        assert self.evaluate_rows[1] <= min(self.c_b)

    @property
    def roster(self) -> list:
        return list(DATASETS[self.dataset]["roster"])


def resolve_job(dataset: str, job, roster=None) -> Job:
    if isinstance(job, Job):
        return job
    m = _JOB_RE.match(str(job))
    t = JOB_T[m.group(2)] if m else None
    if m is None and (m := _EXTRA_JOB_RE.match(str(job))):
        t = EXTRA_JOB_T.get(m.group(1), {}).get(m.group(2))
    if not m or m.group(1) != dataset or t is None:
        raise ValueError("readiness job id must be <dataset>_<S1|S2|V1|T1|T2> (or a registered later batch) of %s: %r" % (dataset, job))
    if roster is not None and list(roster) != DATASETS[dataset]["roster"]:
        raise ValueError("the readiness population is frozen per dataset")
    return Job(dataset, str(job), t)


def compute_scaler(seg: np.ndarray) -> data.Scaler:
    fin = np.isfinite(seg)
    mean = np.array([seg[fin[:, e], e].mean() if fin[:, e].any() else np.nan for e in range(seg.shape[1])])
    std = np.array([seg[fin[:, e], e].std(ddof=0) if fin[:, e].any() else np.nan for e in range(seg.shape[1])])
    floor_hits = int((~(std > spec.ZERO_SCALE_FLOOR)).sum())
    return data.Scaler(mean=mean, std=std, scale=np.maximum(np.nan_to_num(std, nan=0.0), spec.ZERO_SCALE_FLOOR), floor_hits=floor_hits)


@dataclass
class Legal:
    X_raw: np.ndarray           # (P, 192) raw inputs with NaN
    y_raw: np.ndarray           # (P, 48) raw finite targets
    ent: np.ndarray             # (P,) entity index
    k: np.ndarray               # (P,) window start offset inside T
    counts: np.ndarray          # (N,)

    @property
    def n(self) -> int:
        return int(self.ent.size)


def legal_parents(seg: np.ndarray) -> Legal:
    """Fixed flattening order: entity ascending, window start ascending."""
    from numpy.lib.stride_tricks import sliding_window_view
    Xs, ys, ents, ks, counts = [], [], [], [], []
    for e in range(seg.shape[1]):
        col = seg[:, e]
        cs = np.concatenate([[0], np.cumsum(np.isfinite(col))])
        starts = np.arange(N_STARTS)
        ok = ((cs[starts + L] - cs[starts]) >= MIN_X_FINITE) & ((cs[starts + L + H] - cs[starts + L]) == H)
        win = sliding_window_view(col, L + H)[:N_STARTS][ok]
        Xs.append(win[:, :L])
        ys.append(win[:, L:])
        ents.append(np.full(int(ok.sum()), e, dtype=np.int64))
        ks.append(starts[ok].astype(np.int64))
        counts.append(int(ok.sum()))
    return Legal(np.concatenate(Xs).copy(), np.concatenate(ys).copy(), np.concatenate(ents), np.concatenate(ks), np.asarray(counts))


# --------------------------------------------------------------------------- observations (T only, missing-aware)
FIELDS = ("missing_fraction", "longest_gap", "head_gap", "tail_gap", "finite_mean", "finite_std", "lag24_corr", "lag168_corr",
          "constant_run_fraction", "robust_deviation_fraction", "robust_z_max", "recent168_level_shift", "recent168_volatility_ratio",
          "n_legal_parents")
FIELD_DEFINITIONS = {
    "missing_fraction": "missing T rows / 672",
    "longest_gap": "longest run of consecutive missing T rows (hours)",
    "head_gap": "missing rows at the start of T before the first observed value",
    "tail_gap": "missing rows at the end of T after the last observed value (the most recent hours)",
    "finite_mean": "mean of the observed T values (raw units)",
    "finite_std": "std (ddof=0) of the observed T values (raw units); also the frozen scaler",
    "lag24_corr": "Pearson correlation of (x[i], x[i+24]) over pairs where both are observed; null when fewer than 32 pairs or zero variance",
    "lag168_corr": "same with lag 168",
    "constant_run_fraction": "fraction of observed T values lying in a run of >=3 consecutive, observed, exactly equal values",
    "robust_deviation_fraction": "fraction of observed T values with |x - median| > 3.5 x 1.4826 x MAD; null when MAD = 0",
    "robust_z_max": "max |x - median| / (1.4826 x MAD) over observed T values; null when MAD = 0",
    "recent168_level_shift": "(mean of observed values in the last 168 T rows - mean of observed values in the first 504) / finite_std; null when either part has < 32 observed values",
    "recent168_volatility_ratio": "std of observed values in the last 168 T rows / std of observed values in the first 504; null when either part has < 32 observed values or the second std is 0",
    "n_legal_parents": "number of legal training windows of the entity (X >= 32 observed values and a fully observed y)",
}
PUBLIC_NOTE = ("No field diagnoses an anomaly: a large robust deviation can be a real event. Every field is computed from this batch's T "
               "only; null means not computable and is never replaced by 0.")


def _runs(mask: np.ndarray) -> list:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _lag_corr(x: np.ndarray, lag: int):
    a, b = x[:-lag], x[lag:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 32 or a[m].std() == 0 or b[m].std() == 0:
        return None
    r = float(np.corrcoef(a[m], b[m])[0, 1])
    return r if math.isfinite(r) else None


def entity_observation(x: np.ndarray, n_legal: int) -> dict:
    fin = np.isfinite(x)
    v = x[fin]
    gaps = _runs(~fin)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    eq = fin[1:] & fin[:-1] & (x[1:] == x[:-1])
    const_pts = sum((b - a) + 1 for a, b in _runs(eq) if (b - a) >= 2)
    rec, pri = x[-168:], x[:-168]
    rf, pf = rec[np.isfinite(rec)], pri[np.isfinite(pri)]
    std = float(v.std(ddof=0))
    shift = vol = None
    if rf.size >= 32 and pf.size >= 32:
        shift = float((rf.mean() - pf.mean()) / max(std, spec.ZERO_SCALE_FLOOR))
        vol = float(rf.std() / pf.std()) if pf.std() > 0 else None
    return {"missing_fraction": float(1.0 - fin.mean()), "longest_gap": int(max((b - a for a, b in gaps), default=0)),
            "head_gap": int(gaps[0][1] if gaps and gaps[0][0] == 0 else 0),
            "tail_gap": int(gaps[-1][1] - gaps[-1][0] if gaps and gaps[-1][1] == x.size else 0),
            "finite_mean": float(v.mean()), "finite_std": std, "lag24_corr": _lag_corr(x, 24), "lag168_corr": _lag_corr(x, 168),
            "constant_run_fraction": float(const_pts / v.size),
            "robust_deviation_fraction": None if mad == 0 else float((np.abs(v - med) > 3.5 * 1.4826 * mad).mean()),
            "robust_z_max": None if mad == 0 else float(np.abs(v - med).max() / (1.4826 * mad)),
            "recent168_level_shift": shift, "recent168_volatility_ratio": vol, "n_legal_parents": int(n_legal)}


def _r(x, nd=4):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, list):
        return [_r(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    return x


def overview(sl: RSlice, job: Job, scaler: data.Scaler, legal: Legal) -> dict:
    seg = sl.rows(*job.train_range)
    rows = []
    for e in range(seg.shape[1]):
        obs = _r(entity_observation(seg[:, e], legal.counts[e]))
        obs["entity"] = "entity_%d" % e
        rows.append(obs)
    summary = {}
    for f in FIELDS:
        vals = np.array([r[f] for r in rows if r.get(f) is not None], dtype=np.float64)
        summary[f] = (_r({"min": float(vals.min()), "p25": float(np.percentile(vals, 25)), "median": float(np.median(vals)),
                          "p75": float(np.percentile(vals, 75)), "max": float(vals.max())}) | {"n": int(vals.size)} if vals.size else None)
    hours0 = sl.hours_of(job.train_range[0], job.train_range[0] + 1)[0]
    return {"job_id": job.job_id, "dataset_exposure": EXPOSURE, "n_entities": len(rows), "train_rows": list(job.train_range),
            "geometry": {"L": L, "H": H, "T": TRAIN_SPAN, "window_starts_per_entity": N_STARTS,
                         "legal_parent_rule": "X has >= 32 observed values and y (48) is fully observed; the same legal set for every plan",
                         "legal_parents_total": legal.n, "c_a_origins_relative_to_t": [0, 48], "c_a_block_hours": 96},
            "consumer": CONSUMER, "actions": action_table(), "fields": list(FIELDS), "field_definitions": FIELD_DEFINITIONS,
            "field_note": PUBLIC_NOTE, "entities": rows, "summary": summary,
            "time_binding": {"clock_hour_at_first_T_row": [int(h) for h in hours0],
                             "note": "absolute row indices are hourly positions of each original series; entities may start at different clock hours"},
            "source": {"rows_read": list(job.train_range), "note": "T-only; no C/E row was read to build this table"}}


INSPECT_KINDS = ("gaps", "segment", "daily_means", "hour_profile")
MAX_INSPECT_ENTITIES = 8
MAX_SEGMENT_ROWS = 336


def inspect_data(sl: RSlice, job: Job, scaler: data.Scaler, entity_indices, kind: str = "gaps", sub_range=None) -> dict:
    lo, hi = job.train_range
    if sub_range is not None:
        a, b = int(sub_range[0]), int(sub_range[1])
        if a < lo or b > hi or a >= b:
            raise PermissionError("inspect_data sub_range must lie inside T=[%d,%d)" % (lo, hi))
    else:
        a, b = (hi - 168, hi) if kind == "segment" else (lo, hi)
    if kind == "segment" and b - a > MAX_SEGMENT_ROWS:
        raise ValueError("segment is limited to %d rows" % MAX_SEGMENT_ROWS)
    seg, hrs = sl.rows(a, b), sl.hours_of(a, b)
    out = {"kind": kind, "rows_read": [a, b], "units": "normalized by the frozen per-entity T scaler; null = missing", "entities": {}}
    for e in [int(i) for i in entity_indices]:
        x = seg[:, e]
        z = (x - scaler.mean[e]) / scaler.scale[e]
        fin = np.isfinite(x)
        if kind == "gaps":
            gaps = _runs(~fin)
            rec = {"missing": int((~fin).sum()), "gaps": [[a + s, a + t, t - s] for s, t in gaps][:60],
                   "n_gaps": len(gaps), "gap_format": "[absolute_start, absolute_end_exclusive, hours]"}
        elif kind == "segment":
            rec = {"start_row": a, "clock_hour_at_start": int(hrs[0, e]), "values": [round(float(v), 3) if f else None for v, f in zip(z, fin)]}
        elif kind == "daily_means":
            rec = [{"start_row": a + d, "mean": round(float(z[d:d + 24][fin[d:d + 24]].mean()), 3) if fin[d:d + 24].any() else None,
                    "missing": int((~fin[d:d + 24]).sum())} for d in range(0, b - a, 24)]
        elif kind == "hour_profile":
            rec = [{"hour": h, "mean": round(float(z[(hrs[:, e] == h) & fin].mean()), 3) if ((hrs[:, e] == h) & fin).any() else None,
                    "observed": int(((hrs[:, e] == h) & fin).sum())} for h in range(24)]
        else:
            raise ValueError("unknown inspect kind %r" % kind)
        out["entities"]["entity_%d" % e] = rec
    return out


# --------------------------------------------------------------------------- plan language (closed DSL)
class PolicyError(ValueError):
    pass


def validate_steps(steps) -> list:
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise PolicyError("steps must be a list of at most %d steps" % MAX_STEPS)
    out = []
    for st in steps:
        if not isinstance(st, dict) or "op" not in st:
            raise PolicyError("each step must be an object with an 'op'")
        op = st["op"]
        if op not in OPS:
            raise PolicyError("unknown op %r; legal ops: %s" % (op, sorted(OPS)))
        extra = set(st) - set(OPS[op]) - {"op"}
        if extra:
            raise PolicyError("step %s has unknown parameters %s" % (op, sorted(extra)))
        clean = {"op": op}
        for k, allowed in OPS[op].items():
            if k not in st:
                if len(allowed) == 1:              # a single legal value may be omitted
                    clean[k] = int(allowed[0]) if k in INT_PARAMS else float(allowed[0])
                    continue
                raise PolicyError("step %s missing parameter %r (legal: %s)" % (op, k, allowed))
            v = st[k]
            match = [x for x in allowed if not isinstance(v, bool) and isinstance(v, (int, float)) and abs(float(v) - float(x)) < 1e-12]
            if not match:
                raise PolicyError("step %s parameter %s=%r not in %s" % (op, k, v, allowed))
            clean[k] = int(match[0]) if k in INT_PARAMS else float(match[0])
        out.append(clean)
    return out


def validate_predicate(p) -> None:
    if not isinstance(p, dict):
        raise PolicyError("predicate must be an object")
    if set(p) == {"const"}:
        if not isinstance(p["const"], bool):
            raise PolicyError("const must be a boolean")
        return
    if set(p) == {"not"}:
        return validate_predicate(p["not"])
    if set(p) in ({"all"}, {"any"}):
        k = next(iter(p))
        if not isinstance(p[k], list) or not p[k]:
            raise PolicyError("%s must be a non-empty list" % k)
        for q in p[k]:
            validate_predicate(q)
        return
    if set(p) != {"feature", "op", "value"}:
        raise PolicyError("predicate leaf must contain exactly feature, op, value (got %s)" % sorted(p))
    if p["feature"] not in FIELDS:
        raise PolicyError("unknown feature %r; legal: %s" % (p["feature"], list(FIELDS)))
    if p["op"] not in spec.PREDICATE_OPS:
        raise PolicyError("unknown predicate op %r" % p["op"])
    v = p["value"]
    if isinstance(v, dict):
        if set(v) != {"quantile"} or isinstance(v["quantile"], bool) or not isinstance(v["quantile"], (int, float)) or not 0.0 <= float(v["quantile"]) <= 1.0:
            raise PolicyError("value object must be {'quantile': q in [0,1]}")
    elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise PolicyError("value must be a finite number or {'quantile': q}")


def validate_text(text: str, where: str = "text", allow_entity_ids: bool = True) -> None:
    low = str(text or "").lower()
    for bad in SOURCE_NAMES:
        if bad in low:
            raise PolicyError("%s contains a source identifier" % where)
    if _JOB_TOKEN.search(str(text or "")):
        raise PolicyError("%s contains a job identifier" % where)
    if not allow_entity_ids and re.search(r"(?<![A-Za-z0-9_])(uid|entity_\d+|series_\d+|T\d{1,3})(?![A-Za-z0-9_])", str(text or ""), re.I):
        raise PolicyError("%s contains an entity/UID-shaped token" % where)
    if re.search(r"```|<code>|import |def |exec\(|eval\(", str(text or "")):
        raise PolicyError("%s looks like code" % where)


def validate_policy(pol) -> dict:
    if not isinstance(pol, dict) or "default" not in pol:
        raise PolicyError("policy must be an object with a 'default'")
    if set(pol) - {"default", "rules", "rationale", "observation_fields_used"}:
        raise PolicyError("unknown policy keys %s" % sorted(set(pol) - {"default", "rules", "rationale", "observation_fields_used"}))
    if not isinstance(pol["default"], dict) or set(pol["default"]) != {"steps"}:
        raise PolicyError("default must be {'steps': [...]}")
    clean = {"default": {"steps": validate_steps(pol["default"]["steps"])}, "rules": []}
    rules = pol.get("rules", []) or []
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        raise PolicyError("rules must be a list of at most %d rules" % MAX_RULES)
    for r in rules:
        if not isinstance(r, dict) or set(r) != {"when", "steps"}:
            raise PolicyError("each rule must be {'when': predicate, 'steps': [...]}")
        validate_predicate(r["when"])
        clean["rules"].append({"when": r["when"], "steps": validate_steps(r["steps"])})
    rationale = str(pol.get("rationale", ""))[:500]
    validate_text(rationale, "rationale")
    clean["rationale"] = rationale
    clean["observation_fields_used"] = [f for f in (pol.get("observation_fields_used") or []) if f in FIELDS]
    return clean


def canonical_steps(steps: list) -> list:
    out, gap_free = [], False
    for st in steps:
        if st["op"] in IMPUTERS:
            if gap_free:
                continue                           # exact no-op on a gap-free window
            gap_free = True
            if st["op"] == "impute_linear":
                continue                           # identical to the internal / final linear completion
            out.append(dict(st))
        else:
            gap_free = True
            out.append(dict(st))
    return out


def compile_policy(pol: dict, table: list) -> dict:
    clean = validate_policy(pol)
    resolved, requested, assignment, rule_index, n_unknown = {}, [], [], [], 0
    for row in table:
        chosen, ri = clean["default"]["steps"], -1
        for i, r in enumerate(clean["rules"]):
            v = policy._eval(r["when"], row, table, resolved)        # existing three-valued evaluator; UNKNOWN never fires
            if v is None:
                n_unknown += 1
            if v is True:
                chosen, ri = r["steps"], i
                break
        requested.append(chosen)
        assignment.append(canonical_steps(chosen))
        rule_index.append(ri)
    return {"policy": clean, "assignment": assignment, "requested_assignment": requested, "rule_index": rule_index,
            "resolved_thresholds": resolved, "n_unknown": n_unknown}


def material_spec(compiled: dict) -> dict:
    programs, idx = [], []
    for steps in compiled["assignment"]:
        if steps not in programs:
            programs.append(steps)
        idx.append(programs.index(steps))
    return {"policy": compiled["policy"], "programs": programs, "entity_program": idx, "rule_index": compiled["rule_index"],
            "resolved_thresholds": compiled["resolved_thresholds"], "n_unknown": compiled["n_unknown"],
            "canonicalized": compiled["assignment"] != compiled["requested_assignment"]}


def uniform_policy(steps: list, rationale: str) -> dict:
    return {"default": {"steps": steps}, "rules": [], "rationale": rationale, "observation_fields_used": []}


BASELINE_LINEAR = uniform_policy([], "Baseline-Linear: keep every observed value, minimum linear completion of input gaps")
FIXED_SEASONAL = uniform_policy([{"op": "period_median_complete", "period": 24, "cycles": 3, "min_donors": 2}],
                                "Fixed-Seasonal: uniform period-24 median completion from up to 3 prior cycles (>=2 donors)")
RANDOM_LENGTH_P = (0.5, 0.35, 0.15)
RANDOM_FIELDS = FIELDS


def random_policy(seed: int) -> dict:
    """Frozen random supply (task §6): each default / exception program has 1/2/3 steps with p=.5/.35/.15, operators and their
    public parameter values equiprobable; 0 or 1 exception rule (p=.5) on an equiprobable public field at the batch median,
    op '>=' or '<' equiprobable. Drawn once per logical candidate; never redrawn by score."""
    rng = np.random.RandomState(seed)
    names = list(OPS)

    def program():
        n = int(rng.choice([1, 2, 3], p=list(RANDOM_LENGTH_P)))
        steps = []
        for _ in range(n):
            op = names[int(rng.randint(len(names)))]
            st = {"op": op}
            for k, allowed in OPS[op].items():
                st[k] = allowed[int(rng.randint(len(allowed)))]
            steps.append(st)
        return steps
    default = program()
    rules = []
    if int(rng.randint(2)) == 1:
        f = RANDOM_FIELDS[int(rng.randint(len(RANDOM_FIELDS)))]
        op = (">=", "<")[int(rng.randint(2))]
        rules.append({"when": {"feature": f, "op": op, "value": {"quantile": 0.5}}, "steps": program()})
    return {"default": {"steps": default}, "rules": rules, "rationale": "random plan (control arm)", "observation_fields_used": []}


# --------------------------------------------------------------------------- program execution (existing executor)
class ProgramExecutionError(RuntimeError):
    pass


def apply_program(x_raw: np.ndarray, steps: list, fill_mean: float) -> tuple:
    from SelfEvolvingHarnessTS.operators.s1_impute import impute_linear
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline
    x = np.asarray(x_raw, dtype=np.float64)
    rec = {"all_missing_mean_fill": 0, "residual_nan_linear_fill": 0}
    if not np.isfinite(x).any():
        rec["all_missing_mean_fill"] = 1
        return np.full_like(x, float(fill_mean)), rec
    out = x.copy()
    if steps:
        res = run_pipeline([(s["op"], {k: v for k, v in s.items() if k != "op"}) for s in steps], x, source=PROFILE)
        if not res.ok or res.artifact is None:
            raise ProgramExecutionError("OPERATOR_FAILED: %s" % res.error[:200])
        out = np.asarray(res.artifact, dtype=np.float64).ravel()
    if out.shape != x.shape:
        raise ProgramExecutionError("LENGTH_CHANGED")
    if np.isinf(out).any():
        raise ProgramExecutionError("INF_OUTPUT")
    if np.isnan(out).any():
        if steps:
            rec["residual_nan_linear_fill"] = 1
        out = impute_linear(out, strength=1.0)
    if not np.isfinite(out).all():
        raise ProgramExecutionError("NONFINITE_AFTER_COMPLETION")
    return out, rec


def serve_inputs(sl: RSlice, origins, assignment: list, scaler: data.Scaler) -> tuple:
    N = len(assignment)
    Xs = np.empty((N, len(origins), L))
    rec = {"windows": 0, "windows_with_missing": 0, "missing_input_points": 0, "all_missing_mean_fill": 0, "residual_nan_linear_fill": 0}
    for oi, o in enumerate(origins):
        win = sl.rows(o - L, o)
        for e in range(N):
            x = win[:, e]
            out, r = apply_program(x, assignment[e], scaler.mean[e])
            Xs[e, oi] = (out - scaler.mean[e]) / scaler.scale[e]
            miss = int(np.isnan(x).sum())
            rec["windows"] += 1
            rec["windows_with_missing"] += int(miss > 0)
            rec["missing_input_points"] += miss
            rec["all_missing_mean_fill"] += r["all_missing_mean_fill"]
            rec["residual_nan_linear_fill"] += r["residual_nan_linear_fill"]
    return Xs, rec


# --------------------------------------------------------------------------- context
@dataclass
class ReadinessContext:
    job: Job
    stage: str
    run_dir: Path
    slice: RSlice
    scaler: data.Scaler
    legal: Legal | None
    _overview: dict | None = field(default=None, repr=False)
    fit_module = FIT_MODULE                    # rt.fit launches this worker (same CLI as the augmentation worker)

    @property
    def job_dir(self) -> Path:
        d = self.run_dir / self.job.job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def overview(self) -> dict:
        if self.legal is None:
            raise PermissionError("overview needs the T stage")
        if self._overview is None:
            p = self.job_dir / "overview.json"
            if p.exists():
                self._overview = context.read_json(p)
            else:
                self._overview = overview(self.slice, self.job, self.scaler, self.legal)
                context.write_json(p, self._overview)
        return self._overview

    def inspect_data(self, entity_indices, kind="gaps", sub_range=None) -> dict:
        return inspect_data(self.slice, self.job, self.scaler, entity_indices, kind, sub_range)


def open_job(dataset: str, job, run_dir, stage: str = "material", roster=None) -> ReadinessContext:
    js = resolve_job(dataset, job, roster)
    lo, hi = getattr(js, context.STAGE_ROWS[stage])
    run_dir = Path(run_dir)
    jd = run_dir / js.job_id
    sc_path = jd / "scaler.npz"
    sl = load_slice(dataset, lo, hi)
    if stage in ("material", "evaluate"):
        seg = sl.rows(*js.train_range)
        fin = np.isfinite(seg)
        scaler = compute_scaler(seg)
        legal = legal_parents(seg)
        bad = [e for e in range(seg.shape[1]) if fin[:, e].sum() < MIN_T_FINITE or not scaler.std[e] > spec.ZERO_SCALE_FLOOR or legal.counts[e] < MIN_LEGAL_PARENTS]
        if bad:
            raise RuntimeError("ELIGIBILITY_FAIL entities %s of %s" % (bad, js.job_id))
        jd.mkdir(parents=True, exist_ok=True)
        if sc_path.exists():
            with np.load(sc_path) as old:
                _check_binding(old, js)
                if not (np.array_equal(old["mean"], scaler.mean) and np.array_equal(old["scale"], scaler.scale)
                        and np.array_equal(old["legal_ent"], legal.ent) and np.array_equal(old["legal_k"], legal.k)):
                    raise RuntimeError("scaler / legal parent drift within run for %s" % js.job_id)
        else:
            np.savez(sc_path, profile=np.array(PROFILE), dataset=np.array(js.dataset), t=js.t, roster=np.array(js.roster), mean=scaler.mean,
                     std=scaler.std, scale=scaler.scale, legal_ent=legal.ent, legal_k=legal.k, legal_counts=legal.counts)
        return ReadinessContext(js, stage, run_dir, sl, scaler, legal)
    if not sc_path.exists():
        raise RuntimeError("frozen scaler missing for %s; open the material stage first" % js.job_id)
    with np.load(sc_path) as sc:
        _check_binding(sc, js)
        scaler = data.Scaler(mean=sc["mean"].copy(), std=sc["std"].copy(), scale=sc["scale"].copy(), floor_hits=0)
    return ReadinessContext(js, stage, run_dir, sl, scaler, None)


def _check_binding(saved, js: Job) -> None:
    if str(saved["profile"]) != PROFILE or str(saved["dataset"]) != js.dataset or int(saved["t"]) != js.t or [str(x) for x in saved["roster"]] != js.roster:
        raise RuntimeError("profile/dataset/cut/population drift within run for %s" % js.job_id)


# --------------------------------------------------------------------------- materials
@dataclass
class MaterialRef:
    material_id: str
    job_id: str
    path: str                   # npz with X (P, 192): processed inputs of the legal parents, normalized
    key: str
    assignment: list            # canonical per-entity programs
    alias_of: str | None
    summary_path: str


def _index_path(job_dir: Path) -> Path:
    return job_dir / "materials" / "index.json"


def load_index(job_dir: Path) -> dict:
    p = _index_path(job_dir)
    return context.read_json(p) if p.exists() else {}


def _reason_kind(reason: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", reason or "")]
    return re.sub(r"\d+", "#", reason or "") or "as_declared", (nums[0] if nums else 0)


def build(ctx: ReadinessContext, assignment: list, material_id: str) -> MaterialRef:
    from SelfEvolvingHarnessTS.operators import _provenance as prov
    from SelfEvolvingHarnessTS.operators.s1_impute import impute_linear
    if ctx.legal is None:
        raise PermissionError("build_material needs the T (material) stage")
    lg, sc = ctx.legal, ctx.scaler
    if len(assignment) != len(lg.counts):
        raise ValueError("assignment must cover all %d entities" % len(lg.counts))
    key = json.dumps(assignment, sort_keys=True, separators=(",", ":"))
    index = load_index(ctx.job_dir)
    if material_id in index:
        raise ValueError("material id exists")
    for rec in index.values():
        if rec["key"] == key:
            ref = MaterialRef(material_id, ctx.job.job_id, rec["path"], key, assignment, rec["alias_of"] or rec["material_id"], rec["summary_path"])
            index[material_id] = asdict(ref)
            context.write_json(_index_path(ctx.job_dir), index)
            return ref
    X = np.empty_like(lg.X_raw)
    entities = []
    try:
        for e in range(len(lg.counts)):
            prov.start_recording()
            acc = {"windows": 0, "windows_with_missing": 0, "filled_points": 0, "observed_points": 0, "observed_changed_points": 0,
                   "residual_nan_linear_fill": 0, "all_missing_mean_fill": 0}
            sq_obs = sq_fill = shift = ratio = 0.0
            n_ratio = 0
            fallbacks = {}
            for p in np.flatnonzero(lg.ent == e):
                raw = lg.X_raw[p]
                n0 = len(prov._LEDGER)
                out, r = apply_program(raw, assignment[e], sc.mean[e])
                for row in prov._LEDGER[n0:]:
                    kind, pts = _reason_kind(row["reason"])
                    k = "%s->%s:%s" % (row["requested"], row["effective"], kind)
                    fb = fallbacks.setdefault(k, {"windows": 0, "points": 0})
                    fb["windows"] += 1
                    fb["points"] += pts
                X[p] = (out - sc.mean[e]) / sc.scale[e]
                miss, obs = np.isnan(raw), np.isfinite(raw)
                bl = impute_linear(raw, strength=1.0)
                acc["windows"] += 1
                acc["windows_with_missing"] += int(miss.any())
                acc["filled_points"] += int(miss.sum())
                acc["observed_points"] += int(obs.sum())
                acc["observed_changed_points"] += int((out[obs] != raw[obs]).sum())
                acc["residual_nan_linear_fill"] += r["residual_nan_linear_fill"]
                sq_obs += float((((out[obs] - raw[obs]) / sc.scale[e]) ** 2).sum())
                sq_fill += float((((out[miss] - bl[miss]) / sc.scale[e]) ** 2).sum())
                shift += float((out.mean() - bl.mean()) / sc.scale[e])
                if bl.std() > 0:
                    ratio += float(out.std() / bl.std())
                    n_ratio += 1
            entities.append({**acc, "observed_changed_fraction": acc["observed_changed_points"] / max(acc["observed_points"], 1),
                             "observed_change_rms_norm": math.sqrt(sq_obs / max(acc["observed_points"], 1)),
                             "fill_difference_from_linear_rms_norm": math.sqrt(sq_fill / acc["filled_points"]) if acc["filled_points"] else 0.0,
                             "window_mean_shift_vs_linear_norm": shift / max(acc["windows"], 1),
                             "window_std_ratio_vs_linear": ratio / n_ratio if n_ratio else None,
                             "operator_records": fallbacks, "steps": assignment[e]})
    finally:
        prov.stop_recording()
    if not np.isfinite(X).all():
        raise ProgramExecutionError("nonfinite material")
    mdir = ctx.job_dir / "materials"
    mdir.mkdir(parents=True, exist_ok=True)
    npz = mdir / (material_id + ".npz")
    np.savez(npz, X=X)
    sp = mdir / (material_id + "__summary.json")
    context.write_json(sp, {"material_id": material_id, "job_id": ctx.job.job_id, "legal_parents": lg.n, "entities": entities})
    ref = MaterialRef(material_id, ctx.job.job_id, str(npz), key, assignment, None, str(sp))
    index[material_id] = asdict(ref)
    context.write_json(_index_path(ctx.job_dir), index)
    return ref


def inspect_material(ctx: ReadinessContext, ref: MaterialRef, entity_index=None, window: str = "most_missing") -> dict:
    from SelfEvolvingHarnessTS.operators.s1_impute import impute_linear
    if ctx.legal is None:
        raise PermissionError("inspect_material needs the T stage")
    summ = context.read_json(ref.summary_path)
    brief = ("windows", "windows_with_missing", "filled_points", "observed_changed_fraction", "observed_change_rms_norm",
             "fill_difference_from_linear_rms_norm", "window_mean_shift_vs_linear_norm", "window_std_ratio_vs_linear", "residual_nan_linear_fill")
    rows = {"entity_%d" % e: _r({k: r[k] for k in brief}) for e, r in enumerate(summ["entities"])}
    tot = {k: int(sum(r[k] for r in summ["entities"])) for k in ("windows", "windows_with_missing", "filled_points", "observed_points",
                                                                 "observed_changed_points", "residual_nan_linear_fill")}
    tot["observed_changed_fraction"] = round(tot["observed_changed_points"] / max(tot["observed_points"], 1), 6)
    records = {}
    for r in summ["entities"]:
        for k, v in r["operator_records"].items():
            agg = records.setdefault(k, {"windows": 0, "points": 0})
            agg["windows"] += v["windows"]
            agg["points"] += v["points"]
    tot["operator_records"] = records
    out = {"material_id": ref.material_id, "alias_of": ref.alias_of, "batch_totals": tot, "entities": rows,
           "field_note": "norm = divided by the frozen T scale; *_vs_linear compares each prepared window with its Baseline-Linear version; "
                         "operator_records = executor/operator provenance per requested->effective operator and reason (points = count named in the reason)",
           "training_targets": "Raw observed y of each legal parent window; not part of the material and never passed to an operator.",
           "material_holds": "processed X only (%d legal windows x %d inputs); the legal window set and scaler are the same for every plan" % (summ["legal_parents"], L),
           "interpretation": "Material diagnostics (what changed), not a predictor of downstream utility."}
    if entity_index is not None:
        lg, sc, e = ctx.legal, ctx.scaler, int(entity_index)
        idx = np.flatnonzero(lg.ent == e)
        p = int(idx[-1] if window == "last" else idx[int(np.argmax(np.isnan(lg.X_raw[idx]).sum(1)))])
        with np.load(ref.path) as z:
            proc = z["X"][p]
        raw = lg.X_raw[p]
        z0 = (raw - sc.mean[e]) / sc.scale[e]
        out["window"] = {"entity": "entity_%d" % e, "choice": window, "absolute_start_row": int(ctx.job.train_range[0] + lg.k[p]),
                         "entity_operator_records": summ["entities"][e]["operator_records"],
                         "raw": [round(float(v), 3) if np.isfinite(v) else None for v in z0],
                         "baseline_linear": [round(float(v), 3) for v in (impute_linear(raw, strength=1.0) - sc.mean[e]) / sc.scale[e]],
                         "prepared": [round(float(v), 3) for v in proc], "units": "normalized by the frozen T scaler"}
    return out


# --------------------------------------------------------------------------- scoring (missing-aware)
def read_truth(sl: RSlice, origins) -> np.ndarray:
    return np.stack([sl.rows(o, o + H).T for o in origins], axis=1)          # (N, O, H), NaN = unobserved


def predict_inputs(model, Xs: np.ndarray, scaler: data.Scaler) -> np.ndarray:
    pred = np.empty((Xs.shape[0], Xs.shape[1], H))
    for oi in range(Xs.shape[1]):
        pred[:, oi, :] = train.predict(model, Xs[:, oi, :]) * scaler.scale[:, None] + scaler.mean[:, None]
    return pred


def score_block(pred_raw: np.ndarray, y_raw: np.ndarray, scaler: data.Scaler) -> dict:
    N, O, Hh = y_raw.shape
    obs = np.isfinite(y_raw)
    n_nonfinite = int((~np.isfinite(pred_raw)).sum())
    cells = O * Hh
    n_obs = obs.sum(axis=(1, 2))
    coverage = n_obs / cells
    y0 = np.where(obs, y_raw, 0.0)
    err = np.where(obs, ((pred_raw - y0) / scaler.scale[:, None, None]) ** 2, 0.0)
    aerr = np.where(obs, np.abs(pred_raw - y0), 0.0)
    sse_eo = err.sum(axis=2)
    ok_e = (n_obs > 0) & np.isfinite(sse_eo).all(axis=1)
    per_entity = [float(sse_eo[e].sum() / n_obs[e]) if ok_e[e] else None for e in range(N)]
    per_entity_mae = [float(aerr[e].sum() / n_obs[e]) if ok_e[e] else None for e in range(N)]
    per_origin_entity = [[float(sse_eo[e, o] * O / n_obs[e]) if ok_e[e] else None for e in range(N)] for o in range(O)]
    not_scorable = [e for e in range(N) if coverage[e] < COVERAGE_MIN]
    status = "NONFINITE_PREDICTIONS" if n_nonfinite else ("NOT_SCORABLE" if not_scorable else "SCORABLE")
    ok = status == "SCORABLE"
    return {"status": status, "normalized_mse_macro": float(np.mean(per_entity)) if ok else None,
            "mae_raw_macro": float(np.mean(per_entity_mae)) if ok else None,
            "n_entities": N, "n_origins": O, "n_nonfinite_predictions": n_nonfinite,
            "per_origin_entity_normalized_mse": per_origin_entity, "per_entity_normalized_mse": per_entity, "per_entity_mae_raw": per_entity_mae,
            "coverage": {"cells_per_entity": cells, "observed_cells_per_entity": [int(x) for x in n_obs], "min_fraction_rule": COVERAGE_MIN,
                         "min_observed_fraction": float(coverage.min()), "not_scorable_entities": not_scorable,
                         "observed_cells_total": int(n_obs.sum())},
            "semantics": "Observed future cells only (common mask, never chosen per model). per_origin_entity = SSE of that origin's observed "
                         "cells x n_origins / observed cells of the block, so its origin mean is the entity block MSE (= per-origin MSE "
                         "when fully observed). Says nothing about unobserved true values."}


# --------------------------------------------------------------------------- fit worker (subprocess entry, same CLI as train.py)
def training_arrays(ctx: ReadinessContext, ref: MaterialRef) -> tuple:
    """(X processed & normalized, y raw & normalized) over the legal parents in the fixed order; y never depends on the plan."""
    lg, sc = ctx.legal, ctx.scaler
    with np.load(ref.path) as z:
        X = z["X"].copy()
    if X.shape != lg.X_raw.shape:
        raise RuntimeError("material does not cover the legal parent set")
    y = (lg.y_raw - sc.mean[lg.ent][:, None]) / sc.scale[lg.ent][:, None]
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise RuntimeError("non-finite training arrays")
    return X, y


def consumer_record(n_updates: int) -> dict:
    """The Consumer a cell was really trained with (explicit training-length path only)."""
    return {"arch": spec.CONSUMER["arch"], "optimizer": spec.CONSUMER["optimizer"], "lr": spec.LR, "weight_decay": spec.WEIGHT_DECAY,
            "parent_batch": spec.PARENT_BATCH, "n_updates": int(n_updates), "loss": CONSUMER["loss"], "normalization": CONSUMER["normalization"],
            "metric": CONSUMER["metric"], "batch_stream": "first n_updates rows of the unchanged 2000-row parent-batch stream of the model seed"}


def length_cell_id(job_id: str, material_id: str, seed: int, n_updates: int) -> str:
    return "%s__u%d" % (train.cell_id(job_id, material_id, seed), int(n_updates))


def cell_trained_updates(rec: dict) -> int:
    """Training length of a cell record; old-path records carry no consumer field and ran the full stream."""
    return int((rec.get("consumer") or {}).get("n_updates", spec.N_UPDATES))


def fit_one(dataset: str, job: str, run_dir: str, material_id: str, seed: int, train_deadline: float = train.TRAIN_DEADLINE_S,
            feedback: bool = True, n_updates: int | None = None, checkpoints=(), score_checkpoints=None) -> dict:
    """n_updates None: the old path (2000 updates, old cell id and record). An explicit n_updates writes cell `<id>__u<n>` with a
    consumer record; each checkpoint c < n of the same trajectory is saved as runs/<id>__u<c>.pt, and gets its own cell (C_A
    when feedback) only if c is in score_checkpoints (None = all). A cell of another training length is never overwritten."""
    t0 = time.time()
    explicit = n_updates is not None
    n = train.check_n_updates(n_updates) if explicit else spec.N_UPDATES
    marks = sorted({int(c) for c in checkpoints})
    if marks and not explicit:
        raise ValueError("checkpoints need an explicit n_updates")
    scored = set(marks) if score_checkpoints is None else {int(c) for c in score_checkpoints}
    if not scored <= set(marks):
        raise ValueError("score_checkpoints must be saved checkpoints")
    ctx = open_job(dataset, job, run_dir, stage="evaluate" if feedback else "material")   # evaluate: rows [t-672, t+96) only
    index = load_index(ctx.job_dir)
    if material_id not in index:
        raise RuntimeError("unknown material %s for %s" % (material_id, ctx.job.job_id))
    ref = MaterialRef(**index[material_id])
    lg, sc = ctx.legal, ctx.scaler
    for d in ("runs", "predictions_c_a", "cells"):
        (ctx.job_dir / d).mkdir(exist_ok=True)
    cid = length_cell_id(ctx.job.job_id, material_id, seed, n) if explicit else train.cell_id(ctx.job.job_id, material_id, seed)
    for c in [n] + marks:
        p = ctx.job_dir / "cells" / ((length_cell_id(ctx.job.job_id, material_id, seed, c) if explicit else cid) + ".json")
        if p.exists() and cell_trained_updates(context.read_json(p)) != c:
            raise RuntimeError("cell %s exists with another training length; refusing to overwrite" % p.name)
    X, y = training_arrays(ctx, ref)
    states = {}

    def keep(step, model, batch_loss, seconds):
        states[step] = ({k: v.detach().clone() for k, v in model.state_dict().items()}, batch_loss, seconds)

    if explicit:
        bidx = train.batch_indices(seed, n_pool=lg.n, n_updates=n)
        model, tres = train.train_arm(X, y, None, bidx, model_seed=seed, timeout_seconds=train_deadline,
                                      n_updates=n, checkpoints=marks, on_checkpoint=keep if marks else None)
    else:                                          # the old call, unchanged (existing smokes substitute train_arm with this signature)
        bidx = train.batch_indices(seed, n_pool=lg.n)
        model, tres = train.train_arm(X, y, None, bidx, model_seed=seed, timeout_seconds=train_deadline)
    import torch
    base = {"status": "OK", "profile": PROFILE, "job_id": ctx.job.job_id, "dataset": dataset, "material_id": material_id,
            "material_alias_of": ref.alias_of, "model_seed": seed, "batch_seed": spec.batch_seed(seed), "child_view": False,
            "pool_size": lg.n, "legal_parents_per_entity": [int(c) for c in lg.counts], "roster": list(ctx.job.roster), "roster_explicit": False,
            "rows_read": list(ctx.job.evaluate_rows if feedback else ctx.job.material_rows), "device": str(train.device()), "torch": torch.__version__,
            "python": sys.version.split()[0], "hostname": platform.node()}
    serve_c_a = serve_inputs(ctx.slice, ctx.job.c_a, ref.assignment, sc) if feedback else None
    truth_c_a = read_truth(ctx.slice, ctx.job.c_a) if feedback else None

    def score(m, cell):
        if not feedback:
            return None
        pred = predict_inputs(m, serve_c_a[0], sc)
        np.savez(ctx.job_dir / "predictions_c_a" / (cell + ".npz"), pred_raw=pred, origins=np.array(ctx.job.c_a))
        return score_block(pred, truth_c_a, sc)

    saved = []
    for c in marks:
        ccid = length_cell_id(ctx.job.job_id, material_id, seed, c)
        state, batch_loss, seconds = states[c]
        cm = train.make_model()().to(train.device())
        cm.load_state_dict(state)
        cmp_ = ctx.job_dir / "runs" / (ccid + ".pt")
        train.save_model(cm, cmp_)
        saved.append({"n_updates": c, "model_path": str(cmp_), "train_seconds_at_checkpoint": seconds, "cell_recorded": c in scored})
        if c in scored:
            context.write_json(ctx.job_dir / "cells" / (ccid + ".json"), {
                "cell_id": ccid, **base, "model_path": str(cmp_), "consumer": consumer_record(c), "physical_fit": False,
                "trajectory": {"checkpoint_of": cid, "trajectory_updates": n},
                "train": {"seconds": seconds, "n_updates_done": c, "final_train_loss": batch_loss,
                          "first_step_loss": tres.first_step_loss, "first_step_grad_norm": tres.first_step_grad_norm,
                          "first_step_param_delta_norm": tres.first_step_param_delta_norm, "loss_curve": [p for p in tres.loss_curve if p[0] < c]},
                "serve_inputs_c_a": serve_c_a[1] if feedback else None, "scores": {"c_a": score(cm, ccid)} if feedback else {},
                "seconds_total": time.time() - t0})
    mp = ctx.job_dir / "runs" / (cid + ".pt")
    train.save_model(model, mp)
    rec = {"cell_id": cid, **base, "model_path": str(mp),
           "train": {"seconds": tres.seconds, "n_updates_done": tres.n_updates_done, "final_train_loss": tres.final_train_loss,
                     "first_step_loss": tres.first_step_loss, "first_step_grad_norm": tres.first_step_grad_norm,
                     "first_step_param_delta_norm": tres.first_step_param_delta_norm, "loss_curve": tres.loss_curve},
           "serve_inputs_c_a": serve_c_a[1] if feedback else None, "scores": {"c_a": score(model, cid)} if feedback else {},
           "seconds_total": time.time() - t0}
    if explicit:
        rec.update({"consumer": consumer_record(n), "physical_fit": True, "checkpoints_saved": saved})
    context.write_json(ctx.job_dir / "cells" / (cid + ".json"), rec)
    return rec


# --------------------------------------------------------------------------- delivery and label stages
def commit(run_dir, dataset: str, job: str, material_id: str, reason: str, delivery_seed: int, stop_reason: str = "", extra: dict | None = None) -> dict:
    js = resolve_job(dataset, job)
    job_dir = Path(run_dir) / js.job_id
    p = job_dir / "commit.json"
    if p.exists():
        raise RuntimeError("job %s already committed; a delivery cannot be changed" % js.job_id)
    cells = _commit._cells(job_dir)
    fitted = {c["material_id"] for c in cells.values()}
    if material_id not in fitted:
        raise ValueError("commit refused: material %s has no completed fit in this job" % material_id)
    cid = train.cell_id(js.job_id, material_id, delivery_seed)
    if cid not in cells:
        raise ValueError("commit refused: delivery seed %d has no OK cell for %s" % (delivery_seed, material_id))
    rec = {"job_id": js.job_id, "profile": PROFILE, "material_id": material_id, "delivery_seed": delivery_seed, "delivery_cell": cid,
           "reason": str(reason)[:1000], "stop_reason": stop_reason, "committed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
           "fitted_materials_at_commit": sorted(fitted), "extra": extra or {}}
    context.write_json(p, rec)
    return rec


def _stage_predictions(ctx: ReadinessContext, origins) -> dict:
    """{cell_id: (cell, pred_main, pred_shadow, serve_record)} for every OK cell; the shadow feeds the Baseline-Linear inputs."""
    index = load_index(ctx.job_dir)
    N = len(ctx.job.roster)
    bl_inputs, _ = serve_inputs(ctx.slice, origins, [[] for _ in range(N)], ctx.scaler)
    by_key, out = {}, {}
    for cid, c in _commit._cells(ctx.job_dir).items():
        ref = index[c["material_id"]]
        if ref["key"] not in by_key:
            by_key[ref["key"]] = serve_inputs(ctx.slice, origins, ref["assignment"], ctx.scaler)
        Xs, serve = by_key[ref["key"]]
        model = train.load_model(c["model_path"])
        out[cid] = (c, predict_inputs(model, Xs, ctx.scaler), predict_inputs(model, bl_inputs, ctx.scaler), serve)
    return out


C_B_PREREQUISITES = ("commit.json", "consumer_frozen.json")    # consumer_frozen.json: agent-free Consumer study (TRAINING-LENGTH §5)


def open_c_b(run_dir, dataset: str, job: str, prerequisite: str = "commit.json") -> dict:
    run_dir = Path(run_dir)
    js = resolve_job(dataset, job)
    if prerequisite not in C_B_PREREQUISITES:
        raise ValueError("unknown C_B prerequisite")
    if not (run_dir / js.job_id / prerequisite).exists():
        raise PermissionError("c_b prerequisite missing before loading any labels")
    ctx = open_job(dataset, job, run_dir, stage="c_b")
    y_true = read_truth(ctx.slice, js.c_b)
    out = {"job_id": js.job_id, "profile": PROFILE, "opened_at_local": time.strftime("%Y-%m-%d %H:%M:%S"), "rows_read": list(js.c_b_rows), "cells": {}}
    for cid, (c, pred, shadow, serve) in _stage_predictions(ctx, js.c_b).items():
        s, sh = score_block(pred, y_true, ctx.scaler), score_block(shadow, y_true, ctx.scaler)
        out["cells"][cid] = {"material_id": c["material_id"], "model_seed": c["model_seed"], "c_b": s, "c_b_shadow_baseline_inputs": sh, "serve_inputs": serve}
        c["scores"]["c_b"], c["scores"]["c_b_shadow_baseline_inputs"] = s, sh
        context.write_json(ctx.job_dir / "cells" / (cid + ".json"), c)
    context.write_json(ctx.job_dir / "c_b_scores.json", out)
    return out


def freeze_e(run_dir, dataset: str, job: str) -> dict:
    run_dir = Path(run_dir)
    js = resolve_job(dataset, job)
    if not (run_dir / js.job_id / "c_b_scores.json").exists():
        raise PermissionError("e_input prerequisite missing before loading any labels")
    ctx = open_job(dataset, job, run_dir, stage="e_input")
    (ctx.job_dir / "predictions_e").mkdir(exist_ok=True)
    index = {"job_id": js.job_id, "profile": PROFILE, "started_local": time.strftime("%Y-%m-%d %H:%M:%S"), "rows_read": list(js.e_input_rows), "models": {}}
    for cid, (c, pred, shadow, serve) in _stage_predictions(ctx, js.e).items():
        p = ctx.job_dir / "predictions_e" / (cid + ".npz")
        np.savez(p, pred_raw=pred, pred_raw_shadow_baseline_inputs=shadow, origins=np.array(js.e))
        index["models"][cid] = {"path": str(p), "serve_inputs": serve, "frozen_at_local": time.strftime("%Y-%m-%d %H:%M:%S")}
    index["frozen_at_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    context.write_json(ctx.job_dir / "e_frozen.json", index)
    return index


def score_e(run_dir, dataset: str, job: str, *, cohort_root=None) -> dict:
    run_dir = Path(run_dir)
    js = resolve_job(dataset, job)
    if cohort_root is None and (run_dir.parent / "experiment_started.json").exists():
        cohort_root = run_dir.parent
    if cohort_root is not None:
        _commit.require_cohort_frozen(run_dir, js.job_id, cohort_root)      # existing whole-run E barrier, unchanged
    if not (run_dir / js.job_id / "e_frozen.json").exists():
        raise PermissionError("e_target prerequisite missing before loading any labels")
    ctx = open_job(dataset, job, run_dir, stage="e_target")
    fz = context.read_json(ctx.job_dir / "e_frozen.json")
    y_true = read_truth(ctx.slice, js.e)
    cells = _commit._cells(ctx.job_dir)
    out = {"job_id": js.job_id, "profile": PROFILE, "scored_at_local": time.strftime("%Y-%m-%d %H:%M:%S"), "e_frozen_at": fz["frozen_at_local"],
           "rows_read": list(js.e_target_rows), "cells": {}}
    for cid, m in fz["models"].items():
        with np.load(m["path"]) as z:
            if not np.array_equal(z["origins"], np.array(js.e)):
                raise RuntimeError("frozen E origins differ from the job table")
            main, shadow = z["pred_raw"].copy(), z["pred_raw_shadow_baseline_inputs"].copy()
        out["cells"][cid] = {"material_id": cells[cid]["material_id"], "model_seed": cells[cid]["model_seed"],
                             "e": score_block(main, y_true, ctx.scaler), "e_shadow_baseline_inputs": score_block(shadow, y_true, ctx.scaler)}
    context.write_json(ctx.job_dir / "e_scores.json", out)
    return out


def _init_openmp_before_torch() -> None:
    """Windows base Anaconda: scipy/MKL's OpenMP runtime must initialize before torch loads its own copy, otherwise the first
    MKL-threaded operator in a worker (e.g. denoise_savgol while serving inputs) aborts with OMP Error #15. Numerically neutral:
    identical trained weights were checked under both orders (DEV-DATA-READINESS-PROFILE-TRANSFER amendment 1)."""
    from scipy.signal import savgol_filter
    savgol_filter(np.zeros(5), 5, 2)


def main() -> None:
    _init_openmp_before_torch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["c_b", "freeze_e", "score_e"])
    ap.add_argument("--output", type=Path)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--job", required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--material")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--train-deadline", type=float, default=train.TRAIN_DEADLINE_S)
    ap.add_argument("--no-feedback", action="store_true")
    ap.add_argument("--roster", default=None)
    ap.add_argument("--n-updates", type=int, default=None, help="explicit training length (default: the old 2000-update path)")
    ap.add_argument("--checkpoints", default="", help="comma-separated update counts saved from the same trajectory")
    ap.add_argument("--score-checkpoints", default=None, help="comma-separated subset of --checkpoints that get cells (default: all)")
    ap.add_argument("--prerequisite", default="commit.json", choices=C_B_PREREQUISITES)
    a = ap.parse_args()
    if a.roster is not None:
        raise SystemExit("the readiness population is frozen per dataset")
    if a.stage:
        root = a.output.resolve()
        if a.stage == "c_b":
            open_c_b(root, a.dataset, a.job, prerequisite=a.prerequisite)
        else:
            {"freeze_e": freeze_e, "score_e": score_e}[a.stage](root, a.dataset, a.job)
        return
    ints = lambda s: [int(x) for x in s.split(",") if x.strip()]
    rec = fit_one(a.dataset, a.job, a.run_dir, a.material, a.seed, a.train_deadline, feedback=not a.no_feedback, n_updates=a.n_updates,
                  checkpoints=ints(a.checkpoints), score_checkpoints=None if a.score_checkpoints is None else ints(a.score_checkpoints))
    c_a = rec["scores"].get("c_a") or {}
    print("CELL_OK %s c_a=%s train=%.1fs" % (rec["cell_id"], c_a.get("normalized_mse_macro"), rec["train"]["seconds"]))


if __name__ == "__main__":
    main()
