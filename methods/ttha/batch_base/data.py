"""Row-bounded CSV access, scaler, parent pairs, evaluation windows (numerics identical to
_scratch/ts_aug_donor_pattern_probe/dp_data.py, itself a copy of the night package's tf_data.py and
the a94 package's data_h1.py: csv.reader -> float64, string-sorted roster, ddof=0 scaler with floor,
stride-1 parents, seasonal-168 MASE denominators). A Slice only ever holds the rows it was asked for;
any access outside raises, which is how stage permissions are enforced physically.
"""
from __future__ import annotations

import csv
import itertools
import datetime as _dt
from dataclasses import dataclass

import numpy as np

from . import spec


@dataclass
class Slice:
    dataset: str
    row_start: int
    row_end: int
    values: np.ndarray          # (rows, n_cols_all) float64
    columns_all: list
    roster: list
    roster_cols: np.ndarray
    dates: list

    def rows(self, a: int, b: int) -> np.ndarray:
        if a < self.row_start or b > self.row_end or a >= b:
            raise PermissionError("row access [%d,%d) outside loaded slice [%d,%d)" % (a, b, self.row_start, self.row_end))
        return self.values[a - self.row_start: b - self.row_start][:, self.roster_cols]

    def hour_of_row(self, r: int) -> int:
        if r < self.row_start or r >= self.row_end:
            raise PermissionError("timestamp access row %d outside loaded slice" % r)
        return _dt.datetime.strptime(self.dates[r - self.row_start], "%Y-%m-%d %H:%M:%S").hour


def load_slice(dataset: str, row_start: int, row_end: int, roster=None) -> Slice:
    ds = spec.DATASETS[dataset]
    if row_start < 0 or row_end > ds["total_hours"] or row_start >= row_end:
        raise ValueError("bad slice [%d,%d) for %s" % (row_start, row_end, dataset))
    dates, rows = [], []
    with open(ds["path"], newline="") as f:
        r = csv.reader(f)
        header = next(r)
        columns_all = header[1:]
        for i, row in enumerate(itertools.islice(r, row_end)):
            if i < row_start:
                continue
            if i >= row_end:
                break
            dates.append(row[0])
            rows.append(row[1:])
    if len(rows) != row_end - row_start:
        raise RuntimeError("slice [%d,%d): only %d rows available" % (row_start, row_end, len(rows)))
    values = np.asarray(rows, dtype=np.float64)
    prev = None
    for i, s in enumerate(dates):
        t = _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        if prev is not None and (t - prev).total_seconds() != 3600.0:
            raise RuntimeError("non-hourly gap inside slice at absolute row %d" % (row_start + i))
        prev = t
    if roster is None:
        roster = sorted(columns_all)[: spec.ROSTER_SIZE]
        if roster != ds["roster"]:
            raise RuntimeError("roster from CSV header differs from the frozen roster for %s" % dataset)
    else:
        roster = list(roster)
        if len(roster) != spec.ROSTER_SIZE or len(set(roster)) != spec.ROSTER_SIZE or any(n not in columns_all for n in roster):
            raise RuntimeError("explicit roster must name %d distinct columns of %s" % (spec.ROSTER_SIZE, dataset))
    name_to_idx = {name: i for i, name in enumerate(columns_all)}
    roster_cols = np.asarray([name_to_idx[n] for n in roster], dtype=np.int64)
    sl = Slice(dataset=dataset, row_start=row_start, row_end=row_end, values=values, columns_all=columns_all,
               roster=roster, roster_cols=roster_cols, dates=dates)
    if not np.isfinite(values[:, roster_cols]).all():
        raise RuntimeError("non-finite raw values in roster columns within slice [%d,%d)" % (row_start, row_end))
    return sl


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray
    scale: np.ndarray
    floor_hits: int


def compute_scaler(sl: Slice, t: int) -> Scaler:
    seg = sl.rows(t - spec.TRAIN_SPAN, t)
    mean = seg.mean(axis=0)
    std = seg.std(axis=0, ddof=0)
    floor_hits = int((std <= spec.ZERO_SCALE_FLOOR).sum())
    scale = np.maximum(std, spec.ZERO_SCALE_FLOOR)
    return Scaler(mean=mean, std=std, scale=scale, floor_hits=floor_hits)


@dataclass
class Parents:
    X_norm: np.ndarray          # (32, 433, 192)
    y_norm: np.ndarray          # (32, 433, 48)
    entity_names: list
    anchors: np.ndarray
    start_rows: np.ndarray
    hours: np.ndarray           # clock hour of each parent's start row
    n_parents: int


def build_parents(sl: Slice, t: int, scaler: Scaler) -> Parents:
    n_entities, n_parents = len(sl.roster), spec.N_PARENTS
    X = np.empty((n_entities, n_parents, spec.L))
    Y = np.empty((n_entities, n_parents, spec.H))
    anchors = np.empty(n_parents, dtype=np.int64)
    starts = np.empty(n_parents, dtype=np.int64)
    hours = np.empty(n_parents, dtype=np.int64)
    seg_start = t - spec.TRAIN_SPAN
    for k in range(n_parents):
        s = seg_start + k
        starts[k] = s
        anchors[k] = s + spec.L
        hours[k] = sl.hour_of_row(s)
        window = sl.rows(s, s + spec.L + spec.H)
        X[:, k, :] = window[: spec.L].T
        Y[:, k, :] = window[spec.L:].T
    mean, scale = scaler.mean[:, None, None], scaler.scale[:, None, None]
    X_norm, Y_norm = (X - mean) / scale, (Y - mean) / scale
    if not (np.isfinite(X_norm).all() and np.isfinite(Y_norm).all()):
        raise RuntimeError("non-finite normalized parent pairs at t=%d" % t)
    if anchors[-1] + spec.H != t:
        raise RuntimeError("last parent target must end exactly at t")
    return Parents(X_norm=X_norm, y_norm=Y_norm, entity_names=sl.roster, anchors=anchors, start_rows=starts, hours=hours, n_parents=n_parents)


@dataclass
class Window:
    origin: int
    X_norm: np.ndarray
    y_true_raw: np.ndarray | None


def build_windows(sl: Slice, origins, scaler: Scaler, with_truth: bool) -> list:
    out = []
    mean, scale = scaler.mean[:, None], scaler.scale[:, None]
    for o in origins:
        x_raw = sl.rows(o - spec.L, o).T
        y_raw = sl.rows(o, o + spec.H).T if with_truth else None
        out.append(Window(origin=o, X_norm=(x_raw - mean) / scale, y_true_raw=y_raw))
    return out


def read_targets(sl: Slice, origins) -> np.ndarray:
    out = np.empty((len(sl.roster), len(origins), spec.H))
    for oi, o in enumerate(origins):
        out[:, oi, :] = sl.rows(o, o + spec.H).T
    return out


def mase_denominators(sl: Slice, t: int) -> np.ndarray:
    seg = sl.rows(t - spec.TRAIN_SPAN, t)
    s = spec.MASE_SEASON
    diffs = np.abs(seg[s:] - seg[:-s])
    denom = diffs.mean(axis=0)
    return np.where(denom <= 0, np.nan, denom)


def score_predictions(pred_raw: np.ndarray, y_true_raw: np.ndarray, mean: np.ndarray, scale: np.ndarray, mase_denom: np.ndarray) -> dict:
    """Identical to _scratch/ts_aug_source_grounding_pilot/score_h1.py."""
    finite_pred = np.isfinite(pred_raw)
    n_nonfinite_pred = int((~finite_pred).sum())
    y_norm = (y_true_raw - mean[:, None, None]) / scale[:, None, None]
    pred_norm = (pred_raw - mean[:, None, None]) / scale[:, None, None]
    mse_norm_eo = ((pred_norm - y_norm) ** 2).mean(axis=-1)
    mae_raw_eo = np.abs(pred_raw - y_true_raw).mean(axis=-1)
    mse_norm_e = np.nanmean(mse_norm_eo, axis=1)
    mae_raw_e = np.nanmean(mae_raw_eo, axis=1)
    mase_e = mae_raw_e / mase_denom
    n_mase_undefined = int(np.isnan(mase_denom).sum())
    return {
        "n_entities": pred_raw.shape[0], "n_origins": pred_raw.shape[1],
        "n_nonfinite_predictions": n_nonfinite_pred,
        "normalized_mse_macro": float(np.nanmean(mse_norm_e)),
        "mae_raw_macro": float(np.nanmean(mae_raw_e)),
        "mase_macro": float(np.nanmean(mase_e)) if n_mase_undefined < len(mase_denom) else None,
        "n_mase_undefined_entities": n_mase_undefined,
        "per_origin_entity_normalized_mse": mse_norm_eo.T.tolist(),
        "per_entity_normalized_mse": mse_norm_e.tolist(),
        "per_entity_mae_raw": mae_raw_e.tolist(),
        "per_entity_mase": [None if np.isnan(v) else float(v) for v in mase_e],
        "per_origin_normalized_mse_mean": np.nanmean(mse_norm_eo, axis=0).tolist(),
    }
