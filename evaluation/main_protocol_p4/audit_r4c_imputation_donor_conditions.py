"""R4C: do the *action conditions* of period-median imputation explain when it pays?

W2 is ``period_median_complete -> outlier_mad`` and the ancestor is ``outlier_mad``
alone; ``outlier_mad`` begins with ``interp_nan`` (linear fill), so the per-series
difference ``d = g(W2) - g(ANCESTOR)`` is exactly the marginal effect of replacing
linear fill by period-median fill at the points where the operator finds enough
same-phase donors (``operators/s1_impute.py::period_median_complete``: donors are
``raw[i - k*period]`` for ``k = 1..cycles`` inside the window, ``>= min_donors``
of them observed, else the point falls back to ``interp_nan``).

This script asks whether quantities that describe *what the operator actually
did* -- how many gap points were period-filled, how consistent the donors were,
how far the period fill sits from the linear fill, on the serving context and on
the unit's training corpus separately -- (i) are consistent with the mechanism,
(ii) track |d|, (iii) separate the sign of d out of fold, and (iv) improve the
whole-population utility of the decision "use W2 or keep the incumbent".

Nothing here fits anything.  Per-series gains come from the m_r0k prediction
store; raw with-missing KDD2018 is read strictly below the held-out frontier; the
operator functions are called, not re-implemented (the point flags are a
re-derivation that is cross-checked against the operator's own output).

Definitions, thresholds and verdict rules are frozen in
``docs/R4C_IMPUTATION_DONOR_CONDITIONS_TASKBOOK_2026-09-07.md``.

Run: ``python -m evaluation.main_protocol_p4.audit_r4c_imputation_donor_conditions``
0 Consumer fits, 0 LLM calls, 0 held-out reads, 0 horizon reads.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from contracts import observables
from evaluation.main_protocol_p4 import audit_r4a_pattern_identifiability as r4a
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_hec1 as hec1
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as smoke
from SelfEvolvingHarnessTS.methods.ttha import public_tools
from SelfEvolvingHarnessTS.operators._common import interp_nan
from SelfEvolvingHarnessTS.operators.s1_impute import period_median_complete

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = PROJECT_ROOT / "_scratch/m_r0k_prediction_store.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/r4c_imputation_donor_conditions.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/r4c_imputation_donor_conditions.md"

CONTEXT = int(scoped.CONTEXT_LENGTH)
HORIZON = int(scoped.HORIZON)
DELAYED_OFFSET = 48
PMC = dict(smoke.PMC_PARAMS)            # period / cycles / min_donors, as W2 ran
CHILD = "W2_pmc_then_outlier_mad"
PARENT = "ANCESTOR"
FACES = ("support_face", "delayed_face")

# --- frozen thresholds (taskbook 3; set before any number was seen) ---
SEVERE_MARGINAL_HARM = 0.30
CONSISTENCY_PASS = 0.95
MATERIAL = 0.005
EXACT_ZERO = 1e-9
QUANTILES = (1.0 / 3.0, 0.5, 2.0 / 3.0)

SERVING_FEATURES = (
    "srv_gap_points",
    "srv_period_filled_points",
    "srv_period_filled_frac",
    "srv_donor_dispersion",
    "srv_fill_divergence",
    "srv_tail48_period_filled",
)
TRAINING_FEATURES = (
    "trn_gap_points",
    "trn_period_filled_points",
    "trn_period_filled_frac",
    "trn_fill_divergence",
    "trn_series_touched",
)
ALL_FEATURES = SERVING_FEATURES + TRAINING_FEATURES
TARGETS = ("d_positive", "d_severe_harm")


# ---------------------------------------------------------------------------
# 1. the operator's own point flags (re-derived, then cross-checked)
# ---------------------------------------------------------------------------


def pmc_point_flags(arr: np.ndarray, *, period: int, cycles: int,
                    min_donors: int) -> dict[str, Any]:
    """Which NaN points ``period_median_complete`` fills from donors, and how.

    Mirrors ``operators/s1_impute.py`` lines 149-166: donors are read from the
    immutable input array at ``i - k*period`` for ``k = 1..cycles`` when that
    index is inside the array and finite; ``>= min_donors`` donors -> median of
    donors, else the ``interp_nan`` value at that point.
    """
    raw = np.asarray(arr, dtype=np.float64)
    missing = np.isnan(raw)
    fallback = interp_nan(raw.copy())
    period_idx: list[int] = []
    fallback_idx: list[int] = []
    medians: list[float] = []
    donor_disp: list[float] = []
    divergence: list[float] = []
    for i in np.flatnonzero(missing):
        donors = [
            raw[i - k * period]
            for k in range(1, cycles + 1)
            if i - k * period >= 0 and np.isfinite(raw[i - k * period])
        ]
        if len(donors) >= min_donors:
            med = float(np.median(donors))
            period_idx.append(int(i))
            medians.append(med)
            donor_disp.append(float(np.median(np.abs(np.asarray(donors) - med))))
            divergence.append(abs(med - float(fallback[i])))
        else:
            fallback_idx.append(int(i))
    return {
        "n_missing": int(missing.sum()),
        "period_idx": period_idx,
        "fallback_idx": fallback_idx,
        "medians": medians,
        "donor_disp": donor_disp,
        "divergence": divergence,
        "fallback": fallback,
    }


def cross_check_flags(arr: np.ndarray, flags: Mapping[str, Any]) -> bool:
    """The re-derived flags must reproduce the operator's output exactly."""
    out = period_median_complete(np.asarray(arr, dtype=np.float64), **PMC)
    if flags["period_idx"]:
        if not np.allclose(out[flags["period_idx"]], flags["medians"],
                           rtol=0, atol=1e-12):
            return False
    if flags["fallback_idx"]:
        if not np.allclose(out[flags["fallback_idx"]],
                           flags["fallback"][flags["fallback_idx"]],
                           rtol=0, atol=1e-12):
            return False
    observed = np.isfinite(np.asarray(arr, dtype=np.float64))
    return bool(np.array_equal(out[observed], np.asarray(arr, dtype=np.float64)[observed]))


def _robust_scale(window: np.ndarray) -> float:
    center, scale = public_tools._robust_center_scale(window)
    if center is None or scale is None:
        return float("nan")
    return max(float(scale), observables.PUBLIC_ROBUST_Z_MAD_FLOOR)


def serving_observables(context: np.ndarray) -> tuple[dict[str, float], bool]:
    flags = pmc_point_flags(context, **PMC)
    ok = cross_check_flags(context, flags)
    scale = _robust_scale(context)
    n_missing = flags["n_missing"]
    n_period = len(flags["period_idx"])
    out: dict[str, float] = {
        "srv_gap_points": float(n_missing),
        "srv_period_filled_points": float(n_period),
        "srv_period_filled_frac": float(n_period / n_missing) if n_missing else 0.0,
        "srv_tail48_period_filled": float(
            sum(1 for i in flags["period_idx"] if i >= context.size - 48)),
    }
    if n_period and math.isfinite(scale) and scale > 0:
        out["srv_donor_dispersion"] = float(np.median(flags["donor_disp"]) / scale)
        out["srv_fill_divergence"] = float(np.mean(flags["divergence"]) / scale)
    else:
        out["srv_donor_dispersion"] = float("nan")
        out["srv_fill_divergence"] = float("nan")
    return out, ok


def training_observables(windows: Sequence[np.ndarray],
                         window_uids: Sequence[str]) -> tuple[dict[str, float], int, int]:
    gap = period = 0
    divergences: list[float] = []
    touched: set[str] = set()
    checks_failed = 0
    for window, uid in zip(windows, window_uids):
        flags = pmc_point_flags(window, **PMC)
        if not cross_check_flags(window, flags):
            checks_failed += 1
        gap += flags["n_missing"]
        n_period = len(flags["period_idx"])
        period += n_period
        if n_period:
            touched.add(uid)
            scale = _robust_scale(window[:CONTEXT])
            if math.isfinite(scale) and scale > 0:
                divergences.extend(float(v) / scale for v in flags["divergence"])
    out = {
        "trn_gap_points": float(gap),
        "trn_period_filled_points": float(period),
        "trn_period_filled_frac": float(period / gap) if gap else 0.0,
        "trn_fill_divergence": float(np.mean(divergences)) if divergences else float("nan"),
        "trn_series_touched": float(len(touched)),
    }
    return out, len(windows), checks_failed


# ---------------------------------------------------------------------------
# 2. rank statistics (reused from R4A where possible)
# ---------------------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return None
    rx = r4a._average_ranks(x[keep])
    ry = r4a._average_ranks(y[keep])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 6)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# 3. rows
# ---------------------------------------------------------------------------


def build_pairs(store: Mapping[str, Any], position_block: Mapping[int, str]
                ) -> tuple[list[dict], dict]:
    """One row per (position, face, uid) present for both W2 and ANCESTOR."""
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in store.values():
        if entry["program"] not in (CHILD, PARENT):
            continue
        by_key[(int(entry["position"]), str(entry["face"]), str(entry["program"]))] = entry
    rows: list[dict] = []
    skipped = {"verifier_failed": 0, "degenerate_uid_rows": 0, "unpaired_faces": 0}
    for (position, face, program), child in sorted(by_key.items()):
        if program != CHILD:
            continue
        parent = by_key.get((position, face, PARENT))
        if parent is None:
            skipped["unpaired_faces"] += 1
            continue
        if not (child.get("verifier_passed") and parent.get("verifier_passed")):
            skipped["verifier_failed"] += 1
            continue
        if list(child["eval_uids"]) != list(parent["eval_uids"]):
            raise AssertionError("eval_uids differ between W2 and ANCESTOR at %s" % ((position, face),))
        degenerate = set(child.get("degenerate_uids") or ()) | set(parent.get("degenerate_uids") or ())
        c_raw = np.asarray(child["raw_per_view"], dtype=np.float64)
        c_prog = np.asarray(child["program_per_view"], dtype=np.float64)
        p_raw = np.asarray(parent["raw_per_view"], dtype=np.float64)
        p_prog = np.asarray(parent["program_per_view"], dtype=np.float64)
        if not np.allclose(c_raw, p_raw, rtol=0, atol=1e-12):
            raise AssertionError("raw_per_view differs between W2 and ANCESTOR at %s" % ((position, face),))
        for index, uid in enumerate(child["eval_uids"]):
            uid = str(uid)
            if uid in degenerate:
                skipped["degenerate_uid_rows"] += 1
                continue
            g_child = float(c_raw[index] - c_prog[index])
            g_parent = float(p_raw[index] - p_prog[index])
            d = g_child - g_parent
            rows.append({
                "position": position, "face": face, "origin": int(child["origin"]),
                "block": position_block[position], "uid": uid,
                "g_w2": g_child, "g_anc": g_parent, "d": d, "abs_d": abs(d),
                "d_positive": 1 if d > 0 else 0,
                "d_severe_harm": 1 if -d > SEVERE_MARGINAL_HARM else 0,
                "d_exact_zero": 1 if abs(d) < EXACT_ZERO else 0,
            })
    return rows, skipped


def _col(rows: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    return np.array([float(r.get(name, np.nan)) if r.get(name) is not None else np.nan
                     for r in rows], dtype=np.float64)


# ---------------------------------------------------------------------------
# 4. readouts
# ---------------------------------------------------------------------------


def consistency_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both_zero = [r for r in rows if r["srv_period_filled_points"] == 0
                 and r["trn_period_filled_points"] == 0]
    srv_zero = [r for r in rows if r["srv_period_filled_points"] == 0]
    srv_pos = [r for r in rows if r["srv_period_filled_points"] > 0]
    nonzero_d = [r for r in rows if r["d_exact_zero"] == 0]
    explained = [r for r in nonzero_d
                 if r["srv_period_filled_points"] > 0 or r["trn_period_filled_points"] > 0]

    def _rates(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        d = _col(group, "d")
        return {
            "n": len(group),
            "mean_d": float(d.mean()),
            "mean_abs_d": float(np.abs(d).mean()),
            "d_positive_rate": float(np.mean([r["d_positive"] for r in group])),
            "d_severe_harm_rate": float(np.mean([r["d_severe_harm"] for r in group])),
            "d_exact_zero_rate": float(np.mean([r["d_exact_zero"] for r in group])),
        }

    share_zero_in_both_zero = (float(np.mean([r["d_exact_zero"] for r in both_zero]))
                               if both_zero else None)
    share_explained = float(len(explained) / len(nonzero_d)) if nonzero_d else None
    if both_zero:
        verdict = ("ACTION_CONDITION_CONSISTENT"
                   if share_zero_in_both_zero >= CONSISTENCY_PASS
                   else "ACTION_CONDITION_INCONSISTENT")
        basis = "rows with no period-filled point on either side: share with |d| < 1e-9"
    else:
        verdict = ("ACTION_CONDITION_CONSISTENT"
                   if (share_explained or 0.0) >= CONSISTENCY_PASS
                   else "ACTION_CONDITION_INCONSISTENT")
        basis = ("no row has zero period-filled points on both sides (every unit's "
                 "training corpus has some); the check degenerates to: rows with "
                 "|d| > 1e-9 that have a period-filled point on at least one side")
    return {
        "rows": len(rows),
        "rows_no_period_fill_on_either_side": len(both_zero),
        "share_exact_zero_d_among_those": share_zero_in_both_zero,
        "rows_with_nonzero_d": len(nonzero_d),
        "share_of_nonzero_d_with_a_period_fill_somewhere": share_explained,
        "rows_exact_zero_d_total": int(sum(r["d_exact_zero"] for r in rows)),
        "serving_untouched": _rates(srv_zero),
        "serving_period_filled": _rates(srv_pos),
        "verdict": verdict,
        "verdict_basis": basis,
    }


def strength_readout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    abs_d = _col(rows, "abs_d")
    for feature in ("srv_fill_divergence", "srv_period_filled_points",
                    "srv_donor_dispersion", "trn_fill_divergence",
                    "trn_period_filled_frac"):
        x = _col(rows, feature)
        out[feature] = {"pooled": spearman(x, abs_d)}
        for face in FACES:
            keep = np.array([r["face"] == face for r in rows])
            out[feature][face] = spearman(x[keep], abs_d[keep])
    return out


def lodo_auc(rows: Sequence[Mapping[str, Any]], feature: str, target: str,
             blocks: Sequence[str]) -> dict[str, Any]:
    x = _col(rows, feature)
    y = np.array([int(r[target]) for r in rows])
    b = np.array([r["block"] for r in rows])
    folds = {}
    for block in blocks:
        test = b == block
        train = ~test
        if not test.any() or not train.any():
            continue
        train_auc = r4a.auc(x[train], y[train])
        if train_auc is None:
            continue
        direction = 1.0 if train_auc >= 0.5 else -1.0
        test_auc = r4a.auc(direction * x[test], y[test])
        if test_auc is None:
            continue
        folds[block] = {"train_auc_raw": train_auc, "direction": direction,
                        "test_auc_oriented": test_auc, "n_test": int(test.sum()),
                        "n_test_positive": int(y[test].sum())}
    means = [f["test_auc_oriented"] for f in folds.values()]
    return {
        "pooled_auc_raw": r4a.auc(x, y),
        "folds": folds,
        "lodo_mean": float(np.mean(means)) if means else None,
        "lodo_min": float(np.min(means)) if means else None,
        "n_folds": len(folds),
    }


def _utility(rows: Sequence[Mapping[str, Any]], use_w2: np.ndarray) -> dict[str, float]:
    g_w2 = _col(rows, "g_w2")
    g_anc = _col(rows, "g_anc")
    realised = np.where(use_w2, g_w2, g_anc)
    return {
        "utility": float(realised.mean()),
        "harmed": int((realised < 0).sum()),
        "worst_single_series": float(realised.min()),
        "w2_share": float(use_w2.mean()),
    }


def decision_readout(rows: Sequence[Mapping[str, Any]], feature: str,
                     blocks: Sequence[str]) -> dict[str, Any]:
    """Rule: use W2 iff feature <op> threshold (else the incumbent ANCESTOR).

    Thresholds are candidate quantiles of the feature on the training folds
    only; the (direction, quantile) pair is chosen on the training folds by
    utility over the incumbent and read once on the test fold.  Rows without the
    feature never select W2.
    """
    x = _col(rows, feature)
    b = np.array([r["block"] for r in rows])
    folds: dict[str, Any] = {}
    for block in blocks:
        test = b == block
        train = ~test
        train_rows = [r for r, t in zip(rows, train) if t]
        test_rows = [r for r, t in zip(rows, test) if t]
        xt = x[train]
        finite = xt[np.isfinite(xt)]
        if finite.size < 5 or not test_rows:
            continue
        best = None
        for q in QUANTILES:
            tau = float(np.quantile(finite, q))
            for op in (">=", "<="):
                sel = (xt >= tau) if op == ">=" else (xt <= tau)
                sel = sel & np.isfinite(xt)
                u = _utility(train_rows, sel)["utility"]
                anc = _utility(train_rows, np.zeros(len(train_rows), dtype=bool))["utility"]
                score = u - anc
                if best is None or score > best[0]:
                    best = (score, op, tau, q)
        _score, op, tau, q = best
        xs = x[test]
        sel = ((xs >= tau) if op == ">=" else (xs <= tau)) & np.isfinite(xs)
        rule = _utility(test_rows, sel)
        always_w2 = _utility(test_rows, np.ones(len(test_rows), dtype=bool))
        always_anc = _utility(test_rows, np.zeros(len(test_rows), dtype=bool))
        oracle = _utility(test_rows, _col(test_rows, "g_w2") > _col(test_rows, "g_anc"))
        folds[block] = {
            "rule": "use W2 iff %s %s %.6f (train quantile %.3f)" % (feature, op, tau, q),
            "rule_utility": rule, "always_w2": always_w2, "always_anc": always_anc,
            "oracle_per_series_UPPER_BOUND_NOT_DEPLOYABLE": oracle,
            "rule_minus_always_anc": rule["utility"] - always_anc["utility"],
            "rule_minus_always_w2": rule["utility"] - always_w2["utility"],
            "harmed_minus_always_anc": rule["harmed"] - always_anc["harmed"],
            "worst_minus_always_anc": rule["worst_single_series"] - always_anc["worst_single_series"],
        }
    if not folds:
        return {"folds": {}, "verdict": "NOT_EVALUABLE"}
    d_anc = [f["rule_minus_always_anc"] for f in folds.values()]
    d_w2 = [f["rule_minus_always_w2"] for f in folds.values()]
    h_anc = [f["harmed_minus_always_anc"] for f in folds.values()]
    beats_both = sum(1 for a, w in zip(d_anc, d_w2) if a > 0 and w > 0)
    beats_w2 = sum(1 for w in d_w2 if w > 0)
    n = len(folds)
    mean_anc = float(np.mean(d_anc))
    if (beats_both >= max(3, n - 1) and mean_anc >= MATERIAL
            and float(np.mean(h_anc)) <= 0):
        verdict = "VALUE_CONDITION_INFORMATIVE"
    elif beats_w2 >= max(3, n - 1) or beats_both >= 2:
        verdict = "VALUE_CONDITION_WEAK"
    else:
        verdict = "VALUE_CONDITION_UNINFORMATIVE"
    return {
        "folds": folds,
        "mean_rule_minus_always_anc": mean_anc,
        "mean_rule_minus_always_w2": float(np.mean(d_w2)),
        "mean_harmed_minus_always_anc": float(np.mean(h_anc)),
        "folds_beating_both": beats_both,
        "folds_beating_always_w2": beats_w2,
        "n_folds": n,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 5. run
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    store = json.loads(STORE.read_text(encoding="utf-8"))
    forward = contract.ordering("forward")
    variant = preflight.load_variant()
    nan_count = int(sum(int(np.isnan(s).sum()) for s in variant.values()))
    if nan_count <= 0:
        raise AssertionError("loaded variant has no NaN; wrong data identity")
    reader = r4a.Reader(variant)

    positions = sorted({int(e["position"]) for e in store.values()
                        if e["program"] in (CHILD, PARENT)})
    position_block = {p: str(forward[p]["block"]) for p in positions}
    blocks = sorted(set(position_block.values()))

    rows, skipped = build_pairs(store, position_block)

    # --- serving observables per (origin, uid) ---
    serving: dict[tuple[int, str], dict[str, float]] = {}
    serving_checks = {"contexts": 0, "cross_check_failed": 0}
    for row in rows:
        key = (row["origin"], row["uid"])
        if key in serving:
            continue
        context = reader.window(row["uid"], row["origin"] - CONTEXT, row["origin"])
        obs, ok = serving_observables(context)
        serving[key] = obs
        serving_checks["contexts"] += 1
        serving_checks["cross_check_failed"] += 0 if ok else 1

    # --- training observables per (position, face) ---
    training: dict[tuple[int, str], dict[str, Any]] = {}
    training_checks = {"units": 0, "windows": 0, "cross_check_failed": 0}
    cells: dict[str, Any] = {}
    train_geometry: dict[str, Any] = {}
    for row in rows:
        key = (row["position"], row["face"])
        if key in training:
            continue
        unit = forward[row["position"]]
        uids = hec1.block_uids(unit["span"])
        block = str(unit["block"])
        if block not in cells:
            cells[block], _v = baselines._cell(uids)
        origin = int(row["origin"])
        config = forecast_p4._config(origin)
        at = forecast_p4._cell_at(cells[block], origin)
        roster = at.roster(hec1.FACE)
        train_uids = [str(r["series_uid"]) for r in roster if r["role"] == "train"]
        eval_uids = [str(r["series_uid"]) for r in roster if r["role"] == "eval"]
        if set(eval_uids) != {r["uid"] for r in rows if r["position"] == row["position"]
                              and r["face"] == row["face"]} | set(
                                  str(u) for u in (store["%s|%d|%s" % (CHILD, row["position"], row["face"])]
                                                   .get("degenerate_uids") or ())):
            # served uids must be the eval half of the block
            raise AssertionError("eval roster does not match store uids at %s" % (key,))
        windows, window_uids = [], []
        for uid in train_uids:
            raw = np.asarray(at.values[uid], dtype=np.float64)
            for anchor in config["anchors"]:
                anchor = int(anchor)
                if anchor + HORIZON > origin:
                    continue
                reader.account_external(uid, anchor + HORIZON)
                windows.append(raw[anchor - CONTEXT:anchor + HORIZON])
                window_uids.append(uid)
        # the evaluator's own function must agree on the window set
        ref = scoped._training_windows(roster, at.values, config, origin)
        if len(ref) != len(windows) or not all(
                np.array_equal(a, b, equal_nan=True) for a, b in zip(ref, windows)):
            raise AssertionError("training windows differ from scoped._training_windows at %s" % (key,))
        obs, n_windows, failed = training_observables(windows, window_uids)
        obs["trn_n_windows"] = float(n_windows)
        obs["trn_n_series"] = float(len(train_uids))
        training[key] = obs
        training_checks["units"] += 1
        training_checks["windows"] += n_windows
        training_checks["cross_check_failed"] += failed
        train_geometry.setdefault(block, {
            "train_uids": train_uids, "n_train_series": len(train_uids),
            "anchors_all": [int(a) for a in config["anchors"]],
        })

    for row in rows:
        row.update(serving[(row["origin"], row["uid"])])
        row.update(training[(row["position"], row["face"])])

    # --- 3.1 ---
    consistency = consistency_check(rows)
    # --- 3.2 ---
    strength = strength_readout(rows)
    # --- 3.3 --- (serving features only where the serving side acted)
    acted = [r for r in rows if r["srv_period_filled_points"] > 0]
    direction: dict[str, Any] = {"rows_used_serving_features": len(acted),
                                 "rows_used_training_features": len(rows), "features": {}}
    for feature in ALL_FEATURES:
        subset = acted if feature.startswith("srv_") else rows
        direction["features"][feature] = {
            target: lodo_auc(subset, feature, target, blocks) for target in TARGETS
        }
    # --- 3.4 --- (whole served population; rows without the feature keep the incumbent)
    decision = {feature: decision_readout(rows, feature, blocks) for feature in ALL_FEATURES}
    best_feature = None
    for feature, reading in decision.items():
        if "mean_rule_minus_always_anc" not in reading:
            continue
        if best_feature is None or reading["mean_rule_minus_always_anc"] > decision[best_feature]["mean_rule_minus_always_anc"]:
            best_feature = feature
    rank = {"VALUE_CONDITION_INFORMATIVE": 3, "VALUE_CONDITION_WEAK": 2,
            "VALUE_CONDITION_UNINFORMATIVE": 1, "NOT_EVALUABLE": 0}
    value_verdict = max((r["verdict"] for r in decision.values()), key=lambda v: rank[v])
    # --- 3.5 --- unit level
    unit_rows: dict[tuple[int, str], list[float]] = {}
    for r in rows:
        unit_rows.setdefault((r["position"], r["face"]), []).append(r["d"])
    unit_mean_d = np.array([float(np.mean(v)) for v in unit_rows.values()])
    unit_level = {}
    for feature in ("trn_period_filled_frac", "trn_fill_divergence",
                    "trn_period_filled_points", "trn_series_touched"):
        xs = np.array([training[k][feature] for k in unit_rows])
        unit_level[feature] = {"spearman_with_unit_mean_d": spearman(xs, unit_mean_d),
                               "n_units": int(len(unit_rows))}
    # descriptive tables
    per_unit = [{
        "position": k[0], "face": k[1], "block": position_block[k[0]],
        "mean_d": float(np.mean(v)), "n": len(v),
        **{f: training[k][f] for f in TRAINING_FEATURES + ("trn_n_windows",)},
    } for k, v in sorted(unit_rows.items())]

    payload = {
        "task": "R4C_IMPUTATION_DONOR_CONDITIONS",
        "generated_at": datetime.now().astimezone().isoformat(),
        "evidence_class": "MECHANISM / NEGATIVE-capable; development; not a capability claim",
        "question": ("do the action conditions of period-median imputation explain "
                     "when replacing linear fill pays, and do they improve the "
                     "use-W2-or-keep-incumbent decision out of fold?"),
        "boundary": {
            "consumer_fits": 0, "llm_calls": 0, "held_out_reads": reader.held_out_reads,
            "horizon_reads": 0, "max_time_index_read": reader.max_index_read,
            "held_out_frontier": r4a.MAX_ALLOWED_INDEX, "raw_series_reads": reader.reads,
            "existing_files_edited": 0, "prediction_store_written": False, "new_sha_or_hash": 0,
        },
        "provenance": {
            "store": str(STORE.relative_to(PROJECT_ROOT)), "loader": "preflight_natural_gap_variant.load_variant",
            "nan_count": nan_count, "data_version": preflight.DATA_VERSION,
            "child": CHILD, "parent": PARENT,
            "child_steps": [list(s) for s in smoke.WORKFLOW_CANDIDATES[CHILD]],
            "parent_steps": [list(s) for s in smoke.ANCESTOR_PROGRAM],
            "pmc_params": PMC, "pmc_params_source": "smoke_m_r0k_scope_workflow.PMC_PARAMS",
            "operator_source": {
                "period_median_complete": "operators/s1_impute.py::period_median_complete (donors raw[i-k*period], k=1..cycles, >= min_donors -> median; else interp_nan)",
                "outlier_mad": "operators/s1_outlier.py::outlier_mad (interp_nan first, then global MAD clip k=3.5)",
                "d_is_marginal_effect_of": "period-median fill replacing linear fill at donor-sufficient gap points (then identical MAD clip); second-order: the clip bounds move with the filled values",
            },
            "geometry": {"context": CONTEXT, "horizon": HORIZON, "delayed_offset": DELAYED_OFFSET,
                         "training_window": "raw[anchor-192 : anchor+48] for role=train series, anchor+48 <= origin (scoped_serving_evaluator._training_windows, cross-checked)",
                         "served_half": "block_uids(span)[:20] (support_a); training half block_uids(span)[20:40] (support_b)",
                         "per_block": train_geometry},
            "position_block": position_block, "blocks": blocks,
            "skipped": skipped, "serving_cross_check": serving_checks,
            "training_cross_check": training_checks,
        },
        "definitions": {
            "d": "g(W2) - g(ANCESTOR) per (position, face, uid); g = raw_loss - program_loss",
            "d_positive": "d > 0", "d_severe_harm": "-d > 0.30", "exact_zero": "|d| < 1e-9",
            "serving_features": list(SERVING_FEATURES), "training_features": list(TRAINING_FEATURES),
            "frozen": {"consistency_pass": CONSISTENCY_PASS, "material": MATERIAL, "quantiles": list(QUANTILES)},
        },
        "rows": len(rows),
        "readout_3_1_consistency": consistency,
        "readout_3_2_strength_spearman_with_abs_d": strength,
        "readout_3_3_direction_lodo_auc": direction,
        "readout_3_4_decision": decision,
        "readout_3_4_best_feature": best_feature,
        "readout_3_5_unit_level": unit_level,
        "per_unit": per_unit,
        "verdicts": {"action_condition": consistency["verdict"], "value_condition": value_verdict},
    }
    return _round(payload)


def _md(p: Mapping[str, Any]) -> str:
    lines = ["# R4C imputation donor conditions -- machine reading", "",
             "Evidence class: %s" % p["evidence_class"], "",
             "Boundary: fits=%d, llm=%d, held_out_reads=%d, horizon_reads=%d, max_time_index_read=%s (frontier %d)."
             % (p["boundary"]["consumer_fits"], p["boundary"]["llm_calls"], p["boundary"]["held_out_reads"],
                p["boundary"]["horizon_reads"], p["boundary"]["max_time_index_read"], p["boundary"]["held_out_frontier"]),
             "Rows: %d paired (position, face, uid); skipped %s; serving cross-check failed %d/%d; training cross-check failed %d windows of %d."
             % (p["rows"], p["provenance"]["skipped"], p["provenance"]["serving_cross_check"]["cross_check_failed"],
                p["provenance"]["serving_cross_check"]["contexts"], p["provenance"]["training_cross_check"]["cross_check_failed"],
                p["provenance"]["training_cross_check"]["windows"]),
             "", "## Verdicts", "",
             "- action condition: **%s**" % p["verdicts"]["action_condition"],
             "- value condition: **%s** (best feature by mean rule-minus-incumbent: %s)"
             % (p["verdicts"]["value_condition"], p["readout_3_4_best_feature"]), ""]
    c = p["readout_3_1_consistency"]
    lines += ["## 3.1 consistency", "",
              "- rows with no period fill on either side: %d; share |d|<1e-9 among them: %s"
              % (c["rows_no_period_fill_on_either_side"], c["share_exact_zero_d_among_those"]),
              "- rows with |d|>1e-9: %d; share with a period fill somewhere: %s"
              % (c["rows_with_nonzero_d"], c["share_of_nonzero_d_with_a_period_fill_somewhere"]),
              "- rows with |d|<1e-9 in total: %d" % c["rows_exact_zero_d_total"],
              "- basis: %s" % c["verdict_basis"], "",
              "| group | n | mean d | mean abs d | d>0 rate | d<-0.30 rate | exact-zero rate |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("serving_untouched", "serving_period_filled"):
        g = c[name]
        if g.get("n"):
            lines.append("| %s | %d | %s | %s | %s | %s | %s |" % (name, g["n"], g["mean_d"], g["mean_abs_d"],
                                                                  g["d_positive_rate"], g["d_severe_harm_rate"], g["d_exact_zero_rate"]))
    lines += ["", "## 3.2 Spearman with |d|", "", "| feature | pooled | support | delayed |", "| --- | ---: | ---: | ---: |"]
    for f, v in p["readout_3_2_strength_spearman_with_abs_d"].items():
        lines.append("| %s | %s | %s | %s |" % (f, v["pooled"], v["support_face"], v["delayed_face"]))
    lines += ["", "## 3.3 LODO AUC (oriented on training folds)", "",
              "| feature | target | pooled raw | lodo mean | lodo min | folds |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for f, targets in p["readout_3_3_direction_lodo_auc"]["features"].items():
        for t, r in targets.items():
            folds = ", ".join("%s=%s" % (k, v["test_auc_oriented"]) for k, v in r["folds"].items())
            lines.append("| %s | %s | %s | %s | %s | %s |" % (f, t, r["pooled_auc_raw"], r["lodo_mean"], r["lodo_min"], folds))
    lines += ["", "## 3.4 decision: use W2 iff rule, else incumbent (whole served population)", "",
              "| feature | mean rule-ANC | mean rule-W2 | mean harmed-ANC | folds beating both | verdict |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for f, r in p["readout_3_4_decision"].items():
        lines.append("| %s | %s | %s | %s | %s/%s | %s |" % (
            f, r.get("mean_rule_minus_always_anc"), r.get("mean_rule_minus_always_w2"),
            r.get("mean_harmed_minus_always_anc"), r.get("folds_beating_both"), r.get("n_folds"), r["verdict"]))
    lines += ["", "## 3.5 unit level (Spearman with unit mean d, 42 units)", ""]
    for f, r in p["readout_3_5_unit_level"].items():
        lines.append("- %s: %s" % (f, r["spearman_with_unit_mean_d"]))
    lines += ["", "## per unit", "", "| position | face | block | mean d | trn gap | trn period-filled | frac | divergence | series touched | windows |",
              "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for u in p["per_unit"]:
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            u["position"], u["face"], u["block"], u["mean_d"], u["trn_gap_points"], u["trn_period_filled_points"],
            u["trn_period_filled_frac"], u["trn_fill_divergence"], u["trn_series_touched"], u["trn_n_windows"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = run()
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    print(json.dumps({"verdicts": payload["verdicts"], "rows": payload["rows"],
                      "boundary": payload["boundary"],
                      "consistency": {k: v for k, v in payload["readout_3_1_consistency"].items()
                                      if k not in ("serving_untouched", "serving_period_filled")},
                      "best_feature": payload["readout_3_4_best_feature"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
