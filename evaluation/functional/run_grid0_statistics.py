"""run_grid0_statistics.py — GRID0 第 9 步：A1–A8 统计。只出数，不裁定。

统计量定义见主报告 grid0_protocol.analysis_plan_P1A。P1B 裁定不在此脚本内。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in [PROJECT_ROOT, PROJECT_ROOT / "evaluation" / "functional",
          PROJECT_ROOT / "methods" / "ttha"]:
    sys.path.insert(0, str(p))

import run_grid0_census as gc  # noqa: E402
import run_grid0_observations as go  # noqa: E402

CHECKPOINT_REL = gc.CHECKPOINT_REL
M = 0.005
F2_FIELDS = ["modified_fraction_mean", "modified_fraction_max",
             "modified_in_target_share", "modified_run_count_norm",
             "modified_amplitude_ratio", "acting_window_share"]
F3_FIELDS = ["cohort_acting_series_fraction", "cohort_mean_modified_fraction",
             "cohort_tail_shift_deviation", "cohort_mean_z_peak"]


def _onehot(vals: Sequence[str]) -> np.ndarray:
    uniq = sorted(set(vals))
    return np.array([[1.0 if v == u else 0.0 for u in uniq] for v in vals])


def _ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    return beta, yhat, float(np.sum((y - yhat) ** 2))


def _ridge_fit_predict(Xtr, ytr, Xte, alpha=1.0):
    Xtr = np.asarray(Xtr, dtype=np.float64)
    ytr = np.asarray(ytr, dtype=np.float64)
    Xte = np.asarray(Xte, dtype=np.float64)
    mu = np.mean(Xtr, axis=0)
    sd = np.std(Xtr, axis=0)
    keep = sd > 1e-12
    if not np.any(keep):
        Xc = np.zeros_like(Xtr)
        Xtc = np.zeros_like(Xte)
    else:
        Xc = (Xtr - mu) / np.where(sd > 1e-12, sd, 1.0)
        Xtc = (Xte - mu) / np.where(sd > 1e-12, sd, 1.0)
        Xc = Xc[:, keep]
        Xtc = Xtc[:, keep]
    ym = float(np.mean(ytr))
    yc = ytr - ym
    n, p = Xc.shape
    if p == 0:
        beta = np.zeros(0)
    else:
        A = Xc.T @ Xc + alpha * np.eye(p)
        beta = np.linalg.solve(A, Xc.T @ yc)
    return Xtc @ beta + ym


def _macro_mse_by_series(pred: np.ndarray, actual: np.ndarray,
                         series: Sequence[str]) -> dict[str, float]:
    ms: dict[str, list[float]] = {}
    for p, a, s in zip(pred, actual, series):
        ms.setdefault(str(s), []).append(float((p - a) ** 2))
    return {s: float(np.mean(v)) for s, v in ms.items()}


def _skill(macro_model: float, macro_base: float) -> float:
    if macro_base <= 0:
        return float("nan")
    return float(1.0 - macro_model / macro_base)


def main() -> int:
    report = json.loads(CHECKPOINT_REL.read_text(encoding="utf-8"))
    obs = report.get("observations")
    cells = report.get("cells")
    if not obs or not cells or len(cells) != 210:
        raise SystemExit("checkpoint 不完整，拒绝统计")

    cell_gain = {str(c["series"]) + "|" + str(c["origin"]): c.get("gain")
                 for c in cells}
    obs_cells = obs["cells"]
    assert len(obs_cells) == 210

    rows = []
    for oc in obs_cells:
        g = cell_gain.get(str(oc["series"]) + "|" + str(oc["origin"]))
        if g is None:
            raise SystemExit("存在未完成 utility cell，拒绝统计")
        rows.append({
            "cohort": oc["cohort"], "series": oc["series"], "origin": int(oc["origin"]),
            "gain": float(g),
            "f1": np.asarray(oc["f1_vector"], dtype=np.float64),
            "f2": np.asarray([float(oc["f2"][k]) for k in F2_FIELDS],
                             dtype=np.float64),
            "f3": np.asarray([float(oc["f3"][k]) for k in F3_FIELDS],
                             dtype=np.float64),
        })

    gains = np.array([r["gain"] for r in rows])
    cohorts = [r["cohort"] for r in rows]
    series = [r["series"] for r in rows]
    origins = [r["origin"] for r in rows]
    n = len(rows)
    y_c = gains - float(np.mean(gains))
    SST = float(np.sum(y_c ** 2))

    # ---------------- A1 variance components ----------------
    X_cohort = _onehot(cohorts)
    X_series = _onehot(series)
    X_origin = _onehot([str(o) for o in origins])
    _, yh_cohort, _ = _ols_fit(X_cohort, y_c)
    SS_cohort = float(np.sum(yh_cohort ** 2))
    _, yh_series, _ = _ols_fit(X_series, y_c)
    SS_series = float(np.sum(yh_series ** 2))
    _, yh_full, SSE = _ols_fit(np.column_stack([X_series, X_origin]), y_c)
    SS_full = float(np.sum(yh_full ** 2))
    SS_series_within_cohort = SS_series - SS_cohort
    SS_origin_after_series = SS_full - SS_series
    if SSE < 0:
        SSE = 0.0
    df_total = n - 1
    df_resid = n - 40
    r2_raw = float(SS_full / SST) if SST > 0 else float("nan")
    r2_adj = float(1.0 - (SSE / df_resid) / (SST / df_total)) if SST > 0 else float("nan")
    A1 = {
        "method": "y=gain; 分解顺序 cohort -> series(within cohort) -> origin(after series) -> residual; 全因子空间=onehot(series)+onehot(origin), rank=40",
        "n": n, "SST": SST, "SSE": SSE,
        "ss_cohort": SS_cohort, "share_cohort": SS_cohort / SST if SST else None,
        "ss_series_within_cohort": SS_series_within_cohort,
        "share_series_within_cohort": SS_series_within_cohort / SST if SST else None,
        "ss_origin_after_series": SS_origin_after_series,
        "share_origin_after_series": SS_origin_after_series / SST if SST else None,
        "ss_residual": SSE, "share_residual": SSE / SST if SST else None,
        "r2_raw": r2_raw, "r2_adj": r2_adj,
    }

    def features(r: dict[str, Any], spec: str) -> np.ndarray:
        if spec == "F1":
            return r["f1"]
        if spec == "F12":
            return np.concatenate([r["f1"], r["f2"]])
        if spec == "F123":
            return np.concatenate([r["f1"], r["f2"], r["f3"]])
        raise ValueError(spec)

    def loo_series_skill(spec: str, subset: list[int] | None = None):
        idx = list(range(n)) if subset is None else subset
        preds, acts, sers = [], [], []
        for test_s in sorted({series[i] for i in idx}):
            test_idx = [i for i in idx if series[i] == test_s]
            train_idx = [i for i in idx if series[i] != test_s]
            Xtr = np.array([features(rows[i], spec) for i in train_idx])
            ytr = np.array([gains[i] for i in train_idx])
            Xte = np.array([features(rows[i], spec) for i in test_idx])
            pred = _ridge_fit_predict(Xtr, ytr, Xte)
            preds.extend(pred.tolist())
            acts.extend([gains[i] for i in test_idx])
            sers.extend([series[i] for i in test_idx])
        preds = np.array(preds)
        acts = np.array(acts)
        m_model = _macro_mse_by_series(preds, acts, sers)
        baseline = float(np.mean([gains[i] for i in idx]))
        m_base = _macro_mse_by_series(np.full_like(preds, baseline), acts, sers)
        mm = float(np.mean(list(m_model.values())))
        mb = float(np.mean(list(m_base.values())))
        return {"macro_mse_model": mm, "macro_mse_baseline": mb,
                "skill": _skill(mm, mb)}

    def loo_origin_skill(spec: str):
        preds, acts, sers = [], [], []
        for test_o in sorted({origins[i] for i in range(n)}):
            test_idx = [i for i in range(n) if origins[i] == test_o]
            train_idx = [i for i in range(n) if origins[i] != test_o]
            Xtr = np.array([features(rows[i], spec) for i in train_idx])
            ytr = np.array([gains[i] for i in train_idx])
            Xte = np.array([features(rows[i], spec) for i in test_idx])
            pred = _ridge_fit_predict(Xtr, ytr, Xte)
            preds.extend(pred.tolist())
            acts.extend([gains[i] for i in test_idx])
            sers.extend([series[i] for i in test_idx])
        preds = np.array(preds)
        acts = np.array(acts)
        m_model = _macro_mse_by_series(preds, acts, sers)
        baseline = float(np.mean(gains))
        m_base = _macro_mse_by_series(np.full_like(preds, baseline), acts, sers)
        return {"macro_mse_model": float(np.mean(list(m_model.values()))),
                "macro_mse_baseline": float(np.mean(list(m_base.values()))),
                "skill": _skill(float(np.mean(list(m_model.values()))),
                                float(np.mean(list(m_base.values()))))}

    A2 = {"method": "LOO-series, features=F1, ridge alpha=1, series macro MSE",
          **loo_series_skill("F1")}
    A3 = {"method": "LOO-origin, features=F1, ridge alpha=1, series macro MSE",
          **loo_origin_skill("F1")}

    # ---------------- A4 blocked permutation ----------------
    sign_map = np.array([1 if g >= M else (-1 if g < -M else 0) for g in gains])
    sign_mat = sign_map.reshape(35, 6)
    def n_consistent(mat: np.ndarray) -> int:
        c = 0
        for j in range(mat.shape[1]):
            col = mat[:, j]
            nz = col[col != 0]
            if nz.size >= 1 and np.all(nz == nz[0]):
                c += 1
        return c
    observed = n_consistent(sign_mat)
    n_perm = 5000
    rng = np.random.default_rng(20260815)
    exceed = 0
    for _ in range(n_perm):
        perm = sign_mat.copy()
        for i in range(perm.shape[0]):
            perm[i] = rng.permutation(perm[i])
        if n_consistent(perm) >= observed:
            exceed += 1
    A4 = {"method": "blocked permutation within series, preserve per-series sign marginals",
          "observed_consistent_origins": observed,
          "permutations": n_perm,
          "perm_count_ge_observed": exceed,
          "p_upper": (exceed + 1) / (n_perm + 1)}

    # ---------------- A5 per-series sign consistency ----------------
    per_series_sign = []
    n_all_same = 0
    for s in sorted(set(series)):
        vals = np.array([gains[i] for i in range(n) if series[i] == s])
        signs = np.array([1 if v >= M else (-1 if v < -M else 0) for v in vals])
        nz = signs[signs != 0]
        same = bool(nz.size > 0 and np.all(nz == nz[0]))
        if same:
            n_all_same += 1
        per_series_sign.append({
            "series": s, "positive": int(np.sum(signs == 1)),
            "negative": int(np.sum(signs == -1)),
            "neutral": int(np.sum(signs == 0)),
            "all_material_same_sign": same,
        })
    A5 = {"n_series": len(set(series)),
          "n_series_all_material_same_sign": n_all_same,
          "per_series": per_series_sign}

    # ---------------- A6 cross-cohort transfer ----------------
    a_idx = [i for i in range(n) if cohorts[i] == "A"]
    b_idx = [i for i in range(n) if cohorts[i] == "B"]
    def cross_cohort(train_idx, test_idx, spec):
        Xtr = np.array([features(rows[i], spec) for i in train_idx])
        ytr = np.array([gains[i] for i in train_idx])
        Xte = np.array([features(rows[i], spec) for i in test_idx])
        pred = _ridge_fit_predict(Xtr, ytr, Xte)
        act = np.array([gains[i] for i in test_idx])
        ser = [series[i] for i in test_idx]
        m_model = _macro_mse_by_series(pred, act, ser)
        baseline = float(np.mean(ytr))
        m_base = _macro_mse_by_series(np.full_like(pred, baseline), act, ser)
        mm = float(np.mean(list(m_model.values())))
        mb = float(np.mean(list(m_base.values())))
        return {"macro_mse_model": mm, "macro_mse_baseline": mb,
                "skill": _skill(mm, mb)}
    A6 = {
        "method": "train one cohort -> test other cohort; ridge alpha=1; series macro MSE; report F1 and F1+F2",
        "A_to_B": {"F1": cross_cohort(a_idx, b_idx, "F1"),
                   "F12": cross_cohort(a_idx, b_idx, "F12")},
        "B_to_A": {"F1": cross_cohort(b_idx, a_idx, "F1"),
                   "F12": cross_cohort(b_idx, a_idx, "F12")},
    }

    # ---------------- A7 F3 increment within cohort LOO-series ----------------
    A7 = {}
    for cohort in ("A", "B"):
        idx = [i for i in range(n) if cohorts[i] == cohort]
        s12 = loo_series_skill("F12", idx)
        s123 = loo_series_skill("F123", idx)
        A7[cohort] = {"skill_F12": s12["skill"], "skill_F123": s123["skill"],
                      "F3_increment": (float(s123["skill"]) - float(s12["skill"])
                                       if math.isfinite(float(s12["skill"])) else None)}

    # ---------------- A8 cohort-specific origin term after F1+F2 ----------------
    # 用 ridge alpha=1 拟合 F1+F2 + origin onehot；再加 cohort×origin 交互，
    # 报告 delta r2。这里以所有 series 等权的 cell 级 SSE 近似；留作诊断。
    X_f12 = np.array([features(r, "F12") for r in rows])
    X_origin_onehot = _onehot([str(o) for o in origins])
    inter_cols = []
    for c in ("A", "B"):
        for o in sorted(set(origins)):
            inter_cols.append(np.asarray([
                1.0 if (cohorts[i] == c and origins[i] == o) else 0.0
                for i in range(n)]))
    inter = np.column_stack(inter_cols)
    X_base = np.column_stack([X_f12, X_origin_onehot])
    X_full = np.column_stack([X_base, inter])
    pred_base = _ridge_fit_predict(X_base, gains, X_base)
    pred_full = _ridge_fit_predict(X_full, gains, X_full)
    sse_base = float(np.sum((gains - pred_base) ** 2))
    sse_full = float(np.sum((gains - pred_full) ** 2))
    A8 = {
        "method": "ridge alpha=1; base=F1+F2+origin; full=base+cohort*origin interaction; cell-level SSE diagnostic",
        "sse_base": sse_base, "sse_full": sse_full,
        "delta_r2": float((sse_base - sse_full) / SST) if SST else None,
    }

    report["statistics"] = {
        "step": "grid0 step 9 P1A statistics (numbers only, no verdict)",
        "generated_by": "evaluation/functional/run_grid0_statistics.py",
        "A1": A1, "A2": A2, "A3": A3, "A4": A4, "A5": A5,
        "A6": A6, "A7": A7, "A8": A8,
    }
    CHECKPOINT_REL.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report["statistics"][k] for k in
                      ["A1", "A2", "A3", "A4", "A6", "A7", "A8"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
