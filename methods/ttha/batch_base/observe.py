"""Deployment-legal observations, built ONLY from a job's own T segment. `overview` returns the
per-entity table (the a94 package's 13 fields from observations.py, the probe's pattern correlations,
and the mechanism package's T-only neighbourhood descriptors) plus batch-level quantiles.
`inspect_data` returns raw T-only views on request. Nothing here reads a C or E row; nothing here
returns a recommendation. Field names are the public list spec.OBS_FIELDS, which policy predicates
may reference.
"""
from __future__ import annotations

import numpy as np

from . import data, spec

EXCL = 48
K_NN = 5


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def entity_observation(seg_raw: np.ndarray, mean: float, std: float) -> dict:
    """Verbatim numerics of _scratch/ts_aug_source_grounding_pilot/observations.py."""
    n = len(seg_raw)
    head, tail = seg_raw[:168], seg_raw[-168:]
    std_seg = (seg_raw - mean) / max(std, spec.ZERO_SCALE_FLOOR)
    idx = np.arange(n)
    slope = float(np.polyfit(idx, std_seg, 1)[0]) * 100.0
    last168_std = float(tail.std(ddof=0))
    full_std = float(seg_raw.std(ddof=0))
    lag24 = _corr(seg_raw[24:], seg_raw[:-24]) if n > 24 else 0.0
    lag168 = _corr(seg_raw[168:], seg_raw[:-168]) if n > 168 else 0.0
    freq = np.fft.rfft(seg_raw - seg_raw.mean())
    energy = np.abs(freq) ** 2
    energy_no_dc = energy[1:]
    total = energy_no_dc.sum()
    top5_share = float(np.sort(energy_no_dc)[-5:].sum() / total) if total > 0 else 0.0
    return {
        "mean": float(mean), "std": float(std), "missing_count": 0,
        "head168_mean": float(head.mean()), "tail168_mean": float(tail.mean()),
        "standardized_trend_per_100h": slope,
        "last168_std": last168_std, "full_T_std": full_std,
        "last168_std_over_full_T_std": (last168_std / full_std) if full_std > 0 else None,
        "lag24_corr": lag24, "lag168_corr": lag168,
        "nondc_spectrum_top5_energy_share": top5_share,
        "standardized_abs_p95": float(np.percentile(np.abs(std_seg), 95)),
        "standardized_abs_max": float(np.abs(std_seg).max()),
        "n_parent_pairs": spec.N_PARENTS,
    }


def _detrend(x: np.ndarray) -> np.ndarray:
    A = np.stack([np.ones(x.shape[0]), np.arange(x.shape[0], dtype=np.float64)], axis=1)
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ beta


def pattern_corrs(z: np.ndarray, half: int = 336, lag: int = 24, eps: float = 1e-8) -> tuple:
    """(r_head, r_tail) of the probe package: de-trended lag-24 Pearson on each half of the normalized T."""
    out = []
    for part in (z[:half], z[half:]):
        res = _detrend(part)
        a, b = res[:-lag], res[lag:]
        if a.std() < eps or b.std() < eps:
            out.append(None)
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        out.append(r if np.isfinite(r) else None)
    return tuple(out)


def neighbourhood(X_e: np.ndarray, Y_e: np.ndarray) -> tuple:
    """Per-entity means of the mechanism package's T-only descriptor for real parents: nearest
    real-input RMS distance (excluding |l-i|<48) and RMS deviation of the parent's own target from
    the mean target of its 5 nearest real inputs."""
    n = X_e.shape[0]
    idx = np.arange(n)
    d2 = (X_e ** 2).sum(-1)[:, None] + (X_e ** 2).sum(-1)[None, :] - 2.0 * X_e @ X_e.T
    D = np.sqrt(np.maximum(d2, 0.0) / X_e.shape[-1])
    D = np.where(np.abs(idx[:, None] - idx[None, :]) < EXCL, np.inf, D)
    part = np.argpartition(D, K_NN - 1, axis=1)[:, :K_NN]
    dk = np.take_along_axis(D, part, axis=1)
    d1 = dk.min(1)
    ydev = np.sqrt(((Y_e - Y_e[part].mean(1)) ** 2).mean(-1))
    return float(d1.mean()), float(ydev.mean())


def overview(sl: data.Slice, job: spec.JobSpec, scaler: data.Scaler, parents: data.Parents) -> dict:
    """Per-entity table (rows named entity_0..entity_31; real names never exposed to agents) + batch quantiles."""
    seg = sl.rows(job.train_range[0], job.train_range[1])            # (672, 32) raw
    rows = []
    for e in range(len(parents.entity_names)):
        obs = entity_observation(seg[:, e], float(scaler.mean[e]), float(scaler.std[e]))
        z = (seg[:, e] - scaler.mean[e]) / scaler.scale[e]
        obs["r_head"], obs["r_tail"] = pattern_corrs(z)
        obs["r_gap"] = (abs(obs["r_head"] - obs["r_tail"]) if (obs["r_head"] is not None and obs["r_tail"] is not None) else None)
        obs["nn1_input_distance"], obs["nn5_target_deviation"] = neighbourhood(parents.X_norm[e], parents.y_norm[e])
        obs["entity"] = "entity_%d" % e
        rows.append(obs)
    summary = {}
    for f in spec.OBS_FIELDS:
        vals = np.array([r[f] for r in rows if r.get(f) is not None], dtype=np.float64)
        summary[f] = ({"min": float(vals.min()), "p25": float(np.percentile(vals, 25)), "median": float(np.median(vals)),
                       "p75": float(np.percentile(vals, 75)), "max": float(vals.max()), "n": int(vals.size)} if vals.size else None)
    return {"job_id": job.job_id, "dataset_exposure": spec.EXPOSURE, "n_entities": len(rows), "train_rows": list(job.train_range),
            "n_parent_pairs_per_entity": spec.N_PARENTS, "consumer": spec.CONSUMER, "actions": spec.ACTIONS,
            "fields": spec.OBS_FIELDS, "entities": rows, "summary": summary,
            "source": {"rows_read": list(job.train_range), "note": "T-only; no C/E row was read to build this table"}}


def inspect_data(sl: data.Slice, job: spec.JobSpec, scaler: data.Scaler, entity_indices, kind: str = "hour_profile", sub_range=None) -> dict:
    """T-only views. kind: 'hour_profile' (24 clock-hour means, normalized), 'daily_means' (28 daily means, normalized),
    'segment' (raw normalized values over an optional sub-range of T, capped at 672 points per entity)."""
    lo, hi = job.train_range
    if sub_range is not None:
        a, b = int(sub_range[0]), int(sub_range[1])
        if a < lo or b > hi or a >= b:
            raise PermissionError("inspect_data sub_range must lie inside T=[%d,%d)" % (lo, hi))
    else:
        a, b = lo, hi
    seg = sl.rows(a, b)
    out = {"kind": kind, "rows_read": [a, b], "entities": {}}
    for e in [int(i) for i in entity_indices]:
        z = (seg[:, e] - scaler.mean[e]) / scaler.scale[e]
        if kind == "hour_profile":
            hours = np.array([sl.hour_of_row(r) for r in range(a, b)])
            out["entities"]["entity_%d" % e] = [float(z[hours == h].mean()) if (hours == h).any() else None for h in range(24)]
        elif kind == "daily_means":
            n = (b - a) // 24
            out["entities"]["entity_%d" % e] = z[: n * 24].reshape(n, 24).mean(1).round(4).tolist()
        elif kind == "segment":
            out["entities"]["entity_%d" % e] = z.round(4).tolist()
        else:
            raise ValueError("unknown inspect kind %r" % kind)
    return out
