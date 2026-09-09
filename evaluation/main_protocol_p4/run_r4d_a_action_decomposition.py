"""R4D-A: three-cell action decomposition of a Program's effect (pooled Ridge).

The question this answers
------------------------
Under the pooled Consumer a Scope that names one served series does two things
at once: it swaps the *model* that series is served from (the program model was
fitted on a prepared training corpus that every served series shares), and it
swaps that series' own *serving context* (the program is applied to it).  R4A /
astra 2.1 showed the second channel is often a no-op -- many served windows come
out of the program bit-identical to linear integrity -- while the harm is still
there.  So the two channels have to be measured apart.

Three cells per (program P, served instance i), all on the same truth::

    L_rr(i)   raw model      / raw context        (= Static, the library's raw_per_view)
    L_pr(i)   P model        / raw context        (only the shared model changed)
    L_pp(i)   P model        / P-prepared context (= the library's program_per_view)

    route_i = L_pr - L_rr    the shared-model channel
    ctx_i   = L_pp - L_pr    this series' own context channel
    total_i = L_pp - L_rr    = route_i + ctx_i,  and total_i = -g_i in the
                             library's sign convention (g = raw - program)

Why this costs 12 fits and not 126
----------------------------------
R4C established the training geometry with data: anchors are frozen at
[312...852], the retention rule is ``anchor + 48 <= origin``, and the smallest
origin in the store is 1176, so every anchor passes at every origin and the
training half of a block does not move.  One block therefore has exactly one raw
model and one model per program, and all of that block's faces are predictions
out of those models.  ``_serve`` fits once and then predicts every context it is
handed, and the frozen Ridge solve reads only ``x_train``/``targets``/
``weights``, so handing it the raw contexts and the prepared contexts in a
single call returns ``L_pr`` and ``L_pp`` from one physical fit.  That is
3 fits per block (raw, ANCESTOR, W2) x 4 blocks = 12, and the ledger below
refuses at 24.

Nothing is re-implemented.  The training windows come from
``scoped_serving_evaluator._training_windows``, the designs from ``_design``,
the fit-and-predict from ``_serve``, the preparation from ``_prepare``, the
programs from ``smoke_m_r0k_scope_workflow`` compiled through
``ScopeExecutor._compiled`` exactly as ``run_hec1`` compiles them, and the
metric is ``smase`` against ``seasonal_scale`` with the same missing-aware mask
``scoped_evaluate`` uses.  ``L_rr`` and ``L_pp`` are reconciled value-by-value
against the prediction store; if they do not reproduce it, the route/ctx
readings are labelled ``UNRELIABLE`` rather than reported.

Run: ``python -m evaluation.main_protocol_p4.run_r4d_a_action_decomposition``
0 LLM calls, 0 held-out reads, 12 physical Ridge fits (hard cap 24).
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p4 import audit_r4a_pattern_identifiability as r4a
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_hec1 as hec1
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as smoke
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
    seasonal_scale,
    smase,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE = PROJECT_ROOT / "_scratch/m_r0k_prediction_store.json"
THREE_CELL_STORE = PROJECT_ROOT / "_scratch/r4d_a_three_cell_store.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/r4d_a_action_decomposition.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/r4d_a_action_decomposition.md"

CONTEXT = int(scoped.CONTEXT_LENGTH)
HORIZON = int(scoped.HORIZON)

#: Physical Ridge fits.  ``FIT_TARGET`` is the estimate written before spending
#: (4 blocks x [raw, ANCESTOR, W2]); ``FIT_HARD_CAP`` is the refusal line from
#: the request's budget correction.  Every ``_serve`` call is one fit.
FIT_TARGET = 12
FIT_HARD_CAP = 24

#: Frozen by the request (section 4.1) before any number was seen.
SEVERE_HARM = 0.30
ROUTE_SHARE_DOMINANT = 0.6
SEVERE_ROUTE_DOMINANT = 0.6
RECONCILE_TOL = 1e-9

FACES = ("support_face", "delayed_face")
ANC = "ANCESTOR"
W2 = "W2_pmc_then_outlier_mad"
PROGRAM_STEPS: dict[str, tuple] = {
    ANC: tuple(smoke.ANCESTOR_PROGRAM),
    W2: tuple(smoke.WORKFLOW_CANDIDATES[W2]),
}
PROGRAMS = (ANC, W2)

#: D5's reading, quoted for the side-by-side the request asks for.  Not recomputed.
D5_REFERENCE = {
    "verdict": "ROUTE_DOMINANT",
    "windows": 11,
    "route_share": 0.73,
    "severe_route_most_negative": "8/10",
    "covered_by_this_package": False,
}


class FitCeiling(RuntimeError):
    """The hard cap would be crossed.  Nothing is spent."""


class FitLedger:
    """One counter, charged before every physical fit."""

    def __init__(self, cap: int = FIT_HARD_CAP) -> None:
        self.cap = int(cap)
        self.n = 0
        self.by_model: dict[str, int] = {}

    def charge(self, label: str) -> None:
        if self.n + 1 > self.cap:
            raise FitCeiling(
                "a %s fit would take the package past %d physical fits (spent %d)"
                % (label, self.cap, self.n))
        self.n += 1
        self.by_model[label] = self.by_model.get(label, 0) + 1


class ReadLedger:
    """Max time index touched, checked against the held-out frontier."""

    def __init__(self) -> None:
        self.max_index_read = -1
        self.reads = 0
        self.held_out_reads = 0

    def account(self, uid: str, stop: int) -> None:
        last = int(stop) - 1
        self.reads += 1
        if last >= r4a.MAX_ALLOWED_INDEX:
            self.held_out_reads += 1
            raise AssertionError(
                "read of %s reaches index %d, at or past the held-out frontier %d"
                % (uid, last, r4a.MAX_ALLOWED_INDEX))
        self.max_index_read = max(self.max_index_read, last)


# ---------------------------------------------------------------------------
# 1. the evaluator's own surfaces, replayed without editing it
# ---------------------------------------------------------------------------


def _viewed_context(prepared: np.ndarray, view: Any) -> tuple[np.ndarray, str]:
    """``_serve``'s first three lines, so a scale-floor context can be found
    before it makes the whole batch raise."""
    params = view.fit(prepared)
    viewed = view.forward(prepared, params, start=0)
    _center, _scale, method = forecast_runtime._center_scale(np, viewed)
    return viewed, str(method)


def _evaluable(prepared: np.ndarray, view: Any) -> bool:
    _viewed, method = _viewed_context(prepared, view)
    return method != "scale_floor_fallback"


def _loss(truth: np.ndarray, prediction: np.ndarray, scale: float) -> float:
    """``scoped_evaluate``'s per-series loss, unchanged."""
    observed = np.isfinite(truth)
    if not observed.any():
        raise RuntimeError("evaluation future contains no observed truth")
    return float(smase(truth[observed], prediction[observed], scale=scale))


# ---------------------------------------------------------------------------
# 2. rank / summary statistics
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
        if not math.isfinite(value):
            return None
        # Rounding is for readability only.  A reconciliation delta of 2e-16 is
        # the whole point of the reconciliation, and round(2e-16, 9) is 0.0, so
        # anything already smaller than the display precision is left alone.
        if value != 0.0 and abs(value) < 1e-6:
            return value
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _round(float(value))
    return value


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "share_positive": float(np.mean(arr > 0.0)),
        "share_severe_harm": float(np.mean(arr > SEVERE_HARM)),
    }


# ---------------------------------------------------------------------------
# 3. one block: three models, every face of that block predicted out of them
# ---------------------------------------------------------------------------


def _block_faces(store: Mapping[str, Any], forward: Sequence[Mapping[str, Any]]
                 ) -> dict[str, list[dict[str, Any]]]:
    """(position, face, origin) grouped by block, from the store's own keys."""
    faces: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in store.values():
        if entry["program"] not in PROGRAMS:
            continue
        position, face = int(entry["position"]), str(entry["face"])
        unit = forward[position]
        row = {
            "position": position,
            "face": face,
            "origin": int(entry["origin"]),
            "block": str(unit["block"]),
            "span": [int(unit["span"][0]), int(unit["span"][1])],
            "eval_uids": [str(uid) for uid in entry["eval_uids"]],
        }
        seen = faces.get((position, face))
        if seen is None:
            faces[(position, face)] = row
        elif seen["origin"] != row["origin"] or seen["eval_uids"] != row["eval_uids"]:
            raise AssertionError(
                "the two programs disagree about origin/eval_uids at %s"
                % ((position, face),))
    by_block: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(faces.values(), key=lambda r: (r["position"], r["face"])):
        by_block.setdefault(row["block"], []).append(row)
    return by_block


def run_block(block: str, faces: Sequence[Mapping[str, Any]], view: Any,
              fits: FitLedger, reads: ReadLedger) -> dict[str, Any]:
    """Three models for this block, then every (face, uid) predicted from all
    three.  Exactly three ``_serve`` calls."""
    span = faces[0]["span"]
    if any(list(row["span"]) != list(span) for row in faces):
        raise AssertionError("block %s carries more than one span" % block)
    uids = hec1.block_uids(span)
    cell, _variant = baselines._cell(uids)

    origins = sorted({int(row["origin"]) for row in faces})
    ref_origin = origins[0]
    ref_config = forecast_p4._config(ref_origin)
    ref_at = forecast_p4._cell_at(cell, ref_origin)
    ref_roster = ref_at.roster(hec1.FACE)
    ref_windows = scoped._training_windows(
        ref_roster, ref_at.values, ref_config, ref_origin)
    ref_train_uids = [str(r["series_uid"]) for r in ref_roster
                      if r["role"] == "train"]
    ref_eval_uids = [str(r["series_uid"]) for r in ref_roster
                     if r["role"] == "eval"]
    for row in faces:
        if row["eval_uids"] != ref_eval_uids:
            raise AssertionError(
                "store eval_uids differ from the roster's eval half at %s"
                % ((row["position"], row["face"]),))
    for uid in ref_train_uids:
        for anchor in ref_config["anchors"]:
            if int(anchor) + HORIZON <= ref_origin:
                reads.account(uid, int(anchor) + HORIZON)

    # R4C's structural claim, re-checked here value by value rather than cited.
    invariance = {"reference_origin": ref_origin,
                  "n_windows": len(ref_windows),
                  "n_train_series": len(ref_train_uids),
                  "anchors_all": [int(a) for a in ref_config["anchors"]],
                  "anchors_retained": [int(a) for a in ref_config["anchors"]
                                       if int(a) + HORIZON <= ref_origin],
                  "origins_checked": origins,
                  "identical_window_sets": True,
                  "identical_train_series": True}
    for origin in origins[1:]:
        config = forecast_p4._config(origin)
        at = forecast_p4._cell_at(cell, origin)
        roster = at.roster(hec1.FACE)
        windows = scoped._training_windows(roster, at.values, config, origin)
        train_uids = [str(r["series_uid"]) for r in roster if r["role"] == "train"]
        if train_uids != ref_train_uids:
            invariance["identical_train_series"] = False
        if len(windows) != len(ref_windows) or not all(
                np.array_equal(a, b, equal_nan=True)
                for a, b in zip(windows, ref_windows)):
            invariance["identical_window_sets"] = False
    if not (invariance["identical_window_sets"]
            and invariance["identical_train_series"]):
        raise AssertionError(
            "block %s does not share one training corpus across its origins; "
            "the one-fit-per-block design is void here" % block)

    executor = ScopeExecutor(
        ref_roster, ref_at.values, ref_config,
        evaluate_fn=views.forecast_runtime._evaluate,
        max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)
    compiled = {label: executor._compiled(PROGRAM_STEPS[label])
                for label in PROGRAMS}

    raw_only = [forecast_runtime._linear_integrity(w) for w in ref_windows]
    designs = {"raw": scoped._design(raw_only, view)}
    train_moved = {}
    for label in PROGRAMS:
        prepared, moved = [], 0
        for window in ref_windows:
            out, points, _trace = scoped._prepare(window, compiled[label])
            prepared.append(out)
            moved += int(points)
        designs[label] = scoped._design(prepared, view)
        train_moved[label] = moved

    # --- every served instance of this block, on all three surfaces ---
    instances: list[dict[str, Any]] = []
    for row in faces:
        origin = int(row["origin"])
        config = forecast_p4._config(origin)
        for uid in ref_eval_uids:
            raw = np.asarray(cell.values[uid], dtype=np.float64)
            reads.account(uid, origin + HORIZON)
            window = raw[origin - CONTEXT:origin]
            raw_ctx = forecast_runtime._linear_integrity(window)
            truth = raw[origin:origin + HORIZON]
            scale = float(seasonal_scale(
                raw[:origin], np.isfinite(raw[:origin]),
                period=int(config["period"]), min_pairs=32))
            item: dict[str, Any] = {
                "block": block, "position": int(row["position"]),
                "face": str(row["face"]), "origin": origin, "uid": uid,
                "truth": truth, "scale": scale,
                "raw_ctx": raw_ctx,
                "raw_ctx_evaluable": _evaluable(raw_ctx, view),
            }
            for label in PROGRAMS:
                served, moved, _trace = scoped._prepare(window, compiled[label])
                item["ctx_%s" % label] = served
                item["moved_%s" % label] = int(moved)
                item["unmodified_%s" % label] = bool(
                    np.array_equal(served, raw_ctx, equal_nan=True))
                item["ctx_evaluable_%s" % label] = _evaluable(served, view)
            instances.append(item)

    # --- fit 1: the raw model, predicting every raw context ---
    raw_index = [i for i, it in enumerate(instances) if it["raw_ctx_evaluable"]]
    if raw_index:
        fits.charge("%s|raw" % block)
        raw_pred = scoped._serve(
            [instances[i]["raw_ctx"] for i in raw_index], view, *designs["raw"])
        for slot, i in enumerate(raw_index):
            instances[i]["L_rr"] = _loss(
                instances[i]["truth"], raw_pred[slot], instances[i]["scale"])

    # --- fits 2 and 3: one program model each, raw and prepared contexts in
    #     the same call, because the fit does not depend on the contexts ---
    for label in PROGRAMS:
        pr_index = [i for i, it in enumerate(instances) if it["raw_ctx_evaluable"]]
        pp_index = [i for i, it in enumerate(instances)
                    if it["ctx_evaluable_%s" % label]]
        batch = ([instances[i]["raw_ctx"] for i in pr_index]
                 + [instances[i]["ctx_%s" % label] for i in pp_index])
        if not batch:
            continue
        fits.charge("%s|%s" % (block, label))
        pred = scoped._serve(batch, view, *designs[label])
        for slot, i in enumerate(pr_index):
            instances[i]["L_pr_%s" % label] = _loss(
                instances[i]["truth"], pred[slot], instances[i]["scale"])
        offset = len(pr_index)
        for slot, i in enumerate(pp_index):
            instances[i]["L_pp_%s" % label] = _loss(
                instances[i]["truth"], pred[offset + slot],
                instances[i]["scale"])

    for item in instances:
        for key in ("truth", "raw_ctx", "ctx_%s" % ANC, "ctx_%s" % W2):
            item.pop(key, None)
    return {"block": block, "instances": instances, "invariance": invariance,
            "train_behavior_points": train_moved,
            "eval_uids": ref_eval_uids, "train_uids": ref_train_uids,
            "faces": [{"position": r["position"], "face": r["face"],
                       "origin": r["origin"]} for r in faces]}


# ---------------------------------------------------------------------------
# 4. rows, reconciliation, readouts
# ---------------------------------------------------------------------------


def build_rows(blocks: Sequence[Mapping[str, Any]], store: Mapping[str, Any]
               ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in blocks:
        for item in payload["instances"]:
            for label in PROGRAMS:
                key = "%s|%d|%s" % (label, item["position"], item["face"])
                entry = store[key]
                index = [str(u) for u in entry["eval_uids"]].index(item["uid"])
                l_rr = item.get("L_rr")
                l_pr = item.get("L_pr_%s" % label)
                l_pp = item.get("L_pp_%s" % label)
                evaluable = None not in (l_rr, l_pr, l_pp)
                row = {
                    "program": label,
                    "block": item["block"],
                    "position": item["position"],
                    "face": item["face"],
                    "origin": item["origin"],
                    "uid": item["uid"],
                    "status": "OK" if evaluable else "NOT_EVALUABLE",
                    "not_evaluable_because": None if evaluable else {
                        "raw_ctx_scale_floor": not item["raw_ctx_evaluable"],
                        "prepared_ctx_scale_floor":
                            not item["ctx_evaluable_%s" % label],
                    },
                    "L_rr": l_rr, "L_pr": l_pr, "L_pp": l_pp,
                    "store_raw_per_view": float(entry["raw_per_view"][index]),
                    "store_program_per_view":
                        float(entry["program_per_view"][index]),
                    "serving_moved_points": item["moved_%s" % label],
                    "serving_unmodified": item["unmodified_%s" % label],
                }
                if evaluable:
                    row["route"] = float(l_pr - l_rr)
                    row["ctx"] = float(l_pp - l_pr)
                    row["total"] = float(l_pp - l_rr)
                    row["store_total"] = float(
                        row["store_program_per_view"] - row["store_raw_per_view"])
                    row["severe_harm"] = bool(row["total"] > SEVERE_HARM)
                else:
                    row["route"] = row["ctx"] = row["total"] = None
                    row["store_total"] = float(
                        row["store_program_per_view"] - row["store_raw_per_view"])
                    row["severe_harm"] = None
                rows.append(row)
    return rows


def reconcile(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``L_rr`` must reproduce ``raw_per_view`` and ``L_pp`` ``program_per_view``."""
    out: dict[str, Any] = {"tolerance": RECONCILE_TOL, "per_program": {}}
    worst_rr = worst_pp = 0.0
    matched_rr = matched_pp = compared = 0
    offenders: list[dict[str, Any]] = []
    for label in PROGRAMS:
        subset = [r for r in rows if r["program"] == label and r["status"] == "OK"]
        d_rr = np.array([abs(r["L_rr"] - r["store_raw_per_view"]) for r in subset])
        d_pp = np.array([abs(r["L_pp"] - r["store_program_per_view"]) for r in subset])
        exact_rr = np.array([r["L_rr"] == r["store_raw_per_view"] for r in subset])
        exact_pp = np.array([r["L_pp"] == r["store_program_per_view"] for r in subset])
        ok_rr = int((d_rr < RECONCILE_TOL).sum())
        ok_pp = int((d_pp < RECONCILE_TOL).sum())
        out["per_program"][label] = {
            "n": len(subset),
            "L_rr_reproduced": ok_rr,
            "L_rr_reproduction_rate": (ok_rr / len(subset)) if subset else None,
            "L_rr_bit_identical": int(exact_rr.sum()) if subset else 0,
            "L_rr_max_abs_delta": float(d_rr.max()) if subset else None,
            "L_pp_reproduced": ok_pp,
            "L_pp_reproduction_rate": (ok_pp / len(subset)) if subset else None,
            "L_pp_bit_identical": int(exact_pp.sum()) if subset else 0,
            "L_pp_max_abs_delta": float(d_pp.max()) if subset else None,
        }
        compared += len(subset)
        matched_rr += ok_rr
        matched_pp += ok_pp
        if subset:
            worst_rr = max(worst_rr, float(d_rr.max()))
            worst_pp = max(worst_pp, float(d_pp.max()))
        for row, a, b in zip(subset, d_rr, d_pp):
            if a >= RECONCILE_TOL or b >= RECONCILE_TOL:
                offenders.append({
                    "program": label, "position": row["position"],
                    "face": row["face"], "uid": row["uid"],
                    "L_rr": row["L_rr"], "store_raw": row["store_raw_per_view"],
                    "delta_rr": float(a),
                    "L_pp": row["L_pp"],
                    "store_program": row["store_program_per_view"],
                    "delta_pp": float(b),
                })
    passed = bool(compared and matched_rr == compared and matched_pp == compared)
    out.update({
        "compared_rows": compared,
        "L_rr_reproduction_rate_overall":
            (matched_rr / compared) if compared else None,
        "L_pp_reproduction_rate_overall":
            (matched_pp / compared) if compared else None,
        "max_abs_delta_L_rr": worst_rr,
        "max_abs_delta_L_pp": worst_pp,
        "offenders": offenders[:40],
        "n_offenders": len(offenders),
        "passed": passed,
        "consequence": ("route / ctx readings are reported as registered"
                        if passed else
                        "route / ctx readings are labelled UNRELIABLE"),
    })
    return out


def decomposition_readout(rows: Sequence[Mapping[str, Any]], *, reliable: bool
                          ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in PROGRAMS:
        for scope, keep in (("support_face", lambda r: r["face"] == "support_face"),
                            ("delayed_face", lambda r: r["face"] == "delayed_face"),
                            ("both_faces", lambda r: True)):
            subset = [r for r in rows if r["program"] == label
                      and r["status"] == "OK" and keep(r)]
            if not subset:
                out["%s|%s" % (label, scope)] = {"n": 0,
                                                 "verdict": "NOT_EVALUABLE"}
                continue
            route = np.array([r["route"] for r in subset], dtype=np.float64)
            ctxv = np.array([r["ctx"] for r in subset], dtype=np.float64)
            total = np.array([r["total"] for r in subset], dtype=np.float64)
            denom = float(np.abs(route).sum() + np.abs(ctxv).sum())
            share = float(np.abs(route).sum() / denom) if denom > 0 else None
            severe = total > SEVERE_HARM
            n_severe = int(severe.sum())
            # "route is the most negative component" in gain terms == route is
            # the largest positive contribution to the loss.
            route_worst = int((route[severe] > ctxv[severe]).sum()) if n_severe else 0
            severe_share = (route_worst / n_severe) if n_severe else None
            if share is None or severe_share is None:
                verdict = "NOT_EVALUABLE"
            elif (share >= ROUTE_SHARE_DOMINANT
                  and severe_share >= SEVERE_ROUTE_DOMINANT):
                verdict = "ROUTE_DOMINANT"
            elif ((1.0 - share) >= ROUTE_SHARE_DOMINANT
                  and (1.0 - severe_share) >= SEVERE_ROUTE_DOMINANT):
                verdict = "CTX_DOMINANT"
            else:
                verdict = "MIXED"
            out["%s|%s" % (label, scope)] = {
                "program": label, "scope": scope, "n": len(subset),
                "route_algebraic_share": share,
                "ctx_algebraic_share": (1.0 - share) if share is not None else None,
                "sum_abs_route": float(np.abs(route).sum()),
                "sum_abs_ctx": float(np.abs(ctxv).sum()),
                "n_severe_harm": n_severe,
                "severe_harm_rate": float(severe.mean()),
                "severe_route_most_negative": route_worst,
                "severe_route_most_negative_share": severe_share,
                "route_mean": float(route.mean()),
                "route_median": float(np.median(route)),
                "ctx_mean": float(ctxv.mean()),
                "ctx_median": float(np.median(ctxv)),
                "total_mean": float(total.mean()),
                "total_median": float(np.median(total)),
                "route_sign_agrees_with_total": float(
                    np.mean(np.sign(route) == np.sign(total))),
                "ctx_sign_agrees_with_total": float(
                    np.mean(np.sign(ctxv) == np.sign(total))),
                "route_ctx_same_sign_rate": float(
                    np.mean(np.sign(route) == np.sign(ctxv))),
                "spearman_route_total": spearman(route, total),
                "spearman_ctx_total": spearman(ctxv, total),
                "verdict": verdict if reliable else "UNRELIABLE",
                "verdict_if_reconciled": verdict,
            }
    return out


def unmodified_readout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """astra 2.1 recomputed: served windows the Program leaves value-identical."""
    out: dict[str, Any] = {
        "definition": ("_apply_program(context, compiled) equals "
                       "_linear_integrity(context) value by value, i.e. "
                       "_prepare's moved count is 0 on the 192-step served window"),
        "per_program_face": {},
    }
    for label in PROGRAMS:
        for scope, keep in (("support_face", lambda r: r["face"] == "support_face"),
                            ("delayed_face", lambda r: r["face"] == "delayed_face"),
                            ("both_faces", lambda r: True)):
            subset = [r for r in rows if r["program"] == label and keep(r)]
            ok = [r for r in subset if r["status"] == "OK"]
            unmod = [r for r in ok if r["serving_unmodified"]]
            modified = [r for r in ok if not r["serving_unmodified"]]
            severe_all = [r for r in ok if r["severe_harm"]]
            out["per_program_face"]["%s|%s" % (label, scope)] = {
                "n_instances": len(subset),
                "n_evaluable": len(ok),
                "n_serving_unmodified": len(unmod),
                "share_serving_unmodified": (len(unmod) / len(ok)) if ok else None,
                "total_distribution_unmodified": _distribution(
                    [r["total"] for r in unmod]),
                "total_distribution_modified": _distribution(
                    [r["total"] for r in modified]),
                "severe_harm_instances": len(severe_all),
                "severe_harm_carried_by_unmodified": sum(
                    1 for r in unmod if r["severe_harm"]),
                "share_of_severe_harm_on_unmodified_windows": (
                    sum(1 for r in unmod if r["severe_harm"]) / len(severe_all)
                    if severe_all else None),
                "ctx_exactly_zero_on_unmodified": sum(
                    1 for r in unmod if abs(r["ctx"]) < RECONCILE_TOL),
            }
    return out


# ---------------------------------------------------------------------------
# 5. run
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    store = json.loads(STORE.read_text(encoding="utf-8"))
    forward = contract.ordering("forward")
    by_block = _block_faces(store, forward)

    n_blocks = len(by_block)
    estimate = 3 * n_blocks
    if estimate > FIT_HARD_CAP:
        raise FitCeiling(
            "the design estimates %d physical fits, past the hard cap %d"
            % (estimate, FIT_HARD_CAP))

    view = views.IdentityView()
    fits = FitLedger()
    reads = ReadLedger()
    blocks = [run_block(block, faces, view, fits, reads)
              for block, faces in sorted(by_block.items())]

    rows = build_rows(blocks, store)
    reconciliation = reconcile(rows)
    reliable = bool(reconciliation["passed"])
    decomposition = decomposition_readout(rows, reliable=reliable)
    unmodified = unmodified_readout(rows)

    variant = preflight.load_variant()
    nan_count = int(sum(int(np.isnan(s).sum()) for s in variant.values()))
    if nan_count <= 0:
        raise AssertionError("loaded variant has no NaN; wrong data identity")

    n_not_evaluable = sum(1 for r in rows if r["status"] != "OK")
    payload = {
        "task": "R4D_A_ACTION_DECOMPOSITION",
        "generated_at": datetime.now().astimezone().isoformat(),
        "evidence_class": ("MECHANISM / INSTRUMENT; development only; not a "
                           "capability or generalisation claim"),
        "question": ("under pooled Ridge, how much of a Program's per-series "
                     "effect comes from the shared training-corpus model swap "
                     "(route) and how much from that series' own serving "
                     "context being prepared (ctx)?"),
        "boundary": {
            "physical_fits": fits.n,
            "physical_fits_hard_cap": FIT_HARD_CAP,
            "physical_fits_target": FIT_TARGET,
            "physical_fits_by_model": dict(sorted(fits.by_model.items())),
            "fit_counting_rule": "one _serve call = one physical Ridge fit",
            "physical_fits_scope": (
                "this run only; the ledger cannot see earlier runs, so a "
                "package that replays the script must add the runs up itself "
                "when reporting against the cap"),
            "llm_calls": 0,
            "held_out_reads": reads.held_out_reads,
            "held_out_frontier": int(r4a.MAX_ALLOWED_INDEX),
            "held_out_origins": list(r4a.HELD_OUT_ORIGINS),
            "max_time_index_read": reads.max_index_read,
            "raw_series_window_reads": reads.reads,
            "existing_files_edited": 0,
            "m_r0k_prediction_store_written": False,
            "new_prediction_store": str(
                THREE_CELL_STORE.relative_to(PROJECT_ROOT).as_posix()),
            "new_sha_or_hash": 0,
            "git_commits": 0,
            "sub_agents_spawned": 0,
        },
        "provenance": {
            "store": str(STORE.relative_to(PROJECT_ROOT).as_posix()),
            "store_entries": len(store),
            "loader": "preflight_natural_gap_variant.load_variant",
            "data_version": preflight.DATA_VERSION,
            "nan_count": nan_count,
            "programs": {
                label: [list(step) for step in PROGRAM_STEPS[label]]
                for label in PROGRAMS
            },
            "program_source": {
                ANC: "smoke_m_r0k_scope_workflow.ANCESTOR_PROGRAM",
                W2: "smoke_m_r0k_scope_workflow.WORKFLOW_CANDIDATES['%s']" % W2,
            },
            "compilation_path": (
                "ScopeExecutor(roster, at.values, config, "
                "evaluate_fn=representation_view.forecast_runtime._evaluate, "
                "max_modified_fraction=run_forecast_p4_performance."
                "MAX_MODIFIED_FRACTION)._compiled(steps) -- the same call "
                "run_hec1._executor / ReplayPredictionCache._build used to "
                "produce the store (run_hec1.py:432-437, 612)"),
            "reused_functions": [
                "scoped_serving_evaluator._training_windows",
                "scoped_serving_evaluator._prepare",
                "scoped_serving_evaluator._design",
                "scoped_serving_evaluator._serve",
                "run_e2_autonomous_natural_workflow_generation._linear_integrity",
                "run_e2_autonomous_natural_workflow_generation._apply_program",
                "run_e2_autonomous_natural_workflow_generation._center_scale",
                "benchmark_v02.metrics.seasonal_scale",
                "benchmark_v02.metrics.smase",
            ],
            "served_context_handling": (
                "identical to scoped_evaluate: the Program is applied to the "
                "192-step window raw[origin-192:origin] alone and nothing is "
                "truncated from a longer window "
                "(scoped_serving_evaluator.py:169-186)"),
            "metric": (
                "smase(truth[observed], prediction[observed], scale=seasonal_"
                "scale(raw[:origin], isfinite, period=config['period'], "
                "min_pairs=32)); truth = raw[origin:origin+48]; observed = "
                "isfinite(truth) -- scoped_evaluate.py:187-222 unchanged"),
            "view": view.name,
            "one_fit_per_block_basis": (
                "R4C section 4: anchors frozen at [312...852], retention "
                "anchor+48 <= origin, smallest store origin 1176, so a block's "
                "training corpus is origin-invariant.  Re-checked here value "
                "by value per block (see block_invariance)."),
            "batching_is_exact": (
                "_serve fits once from x_train/targets/weights and then "
                "predicts every context handed to it; "
                "_exact_weighted_ridge_prediction's solve never reads x_eval "
                "(run_e2_cross_series_curation.py:2826-2831), so putting the "
                "raw and the prepared contexts in one call yields L_pr and "
                "L_pp from a single physical fit"),
            "block_invariance": {p["block"]: p["invariance"] for p in blocks},
            "train_behavior_points": {p["block"]: p["train_behavior_points"]
                                      for p in blocks},
            "faces_per_block": {p["block"]: p["faces"] for p in blocks},
        },
        "definitions": {
            "L_rr": "raw model / raw context (= Static = store raw_per_view)",
            "L_pr": "program model / raw context",
            "L_pp": "program model / program-prepared context (= store program_per_view)",
            "route": "L_pr - L_rr, the shared-model channel",
            "ctx": "L_pp - L_pr, this series' own context channel",
            "total": "L_pp - L_rr = route + ctx; equals -g in the store's sign "
                     "convention (g = raw_per_view - program_per_view)",
            "severe_harm": "total > %.2f (store convention: g < -%.2f)"
                           % (SEVERE_HARM, SEVERE_HARM),
            "route_is_most_negative_component": (
                "in gain terms the most negative component; for the loss that "
                "is the largest positive one, i.e. route > ctx"),
            "frozen": {
                "severe_harm": SEVERE_HARM,
                "route_share_dominant": ROUTE_SHARE_DOMINANT,
                "severe_route_dominant": SEVERE_ROUTE_DOMINANT,
                "reconcile_tolerance": RECONCILE_TOL,
            },
        },
        "population": {
            "blocks": sorted(by_block),
            "faces": sum(len(v) for v in by_block.values()),
            "rows": len(rows),
            "rows_per_program": len(rows) // len(PROGRAMS),
            "not_evaluable_rows": n_not_evaluable,
            "faces_per_block_count": {b: len(v) for b, v in sorted(by_block.items())},
        },
        "reconciliation": reconciliation,
        "decomposition": decomposition,
        "serving_window_unmodified": unmodified,
        "d5_reference": D5_REFERENCE,
        "verdicts": {
            key: value.get("verdict")
            for key, value in decomposition.items()
        },
        "reliable": reliable,
    }
    return _round(payload), rows


def three_cell_store(rows: Sequence[Mapping[str, Any]],
                     payload: Mapping[str, Any]) -> dict[str, Any]:
    """Per-series three-cell losses, as a new prediction store."""
    entries: dict[str, Any] = {}
    for row in rows:
        key = "%s|%d|%s" % (row["program"], row["position"], row["face"])
        entry = entries.setdefault(key, {
            "program": row["program"], "position": row["position"],
            "face": row["face"], "origin": row["origin"], "block": row["block"],
            "eval_uids": [], "L_rr": [], "L_pr": [], "L_pp": [],
            "route": [], "ctx": [], "total": [],
            "status": [], "serving_moved_points": [], "serving_unmodified": [],
            "store_raw_per_view": [], "store_program_per_view": [],
        })
        entry["eval_uids"].append(row["uid"])
        for field in ("L_rr", "L_pr", "L_pp", "route", "ctx", "total",
                      "status", "serving_moved_points", "serving_unmodified",
                      "store_raw_per_view", "store_program_per_view"):
            entry[field].append(row[field])
    return {
        "task": "R4D_A_THREE_CELL_LOSSES",
        "generated_at": payload["generated_at"],
        "note": ("a new store; _scratch/m_r0k_prediction_store.json is not "
                 "written.  L_rr / L_pp reconcile with that store's "
                 "raw_per_view / program_per_view -- see the reconciliation "
                 "block of artifacts/main_protocol/r4d_a_action_decomposition.json"),
        "definitions": payload["definitions"],
        "boundary": payload["boundary"],
        "entries": entries,
    }


def _md(p: Mapping[str, Any]) -> str:
    b = p["boundary"]
    rec = p["reconciliation"]
    lines = [
        "# R4D-A three-cell action decomposition -- machine reading", "",
        "Evidence class: %s" % p["evidence_class"], "",
        "Boundary: physical_fits=%d / hard cap %d (target %d), llm=%d, "
        "held_out_reads=%d, max_time_index_read=%s (frontier %d), "
        "existing files edited=%d, m_r0k store written=%s."
        % (b["physical_fits"], b["physical_fits_hard_cap"],
           b["physical_fits_target"], b["llm_calls"], b["held_out_reads"],
           b["max_time_index_read"], b["held_out_frontier"],
           b["existing_files_edited"], b["m_r0k_prediction_store_written"]),
        "",
        "Population: %d blocks, %d faces, %d rows (%d per program), "
        "%d NOT_EVALUABLE."
        % (len(p["population"]["blocks"]), p["population"]["faces"],
           p["population"]["rows"], p["population"]["rows_per_program"],
           p["population"]["not_evaluable_rows"]),
        "", "## Reconciliation against the prediction store", "",
        "- rows compared: %d; L_rr reproduction rate %s; L_pp reproduction "
        "rate %s (tolerance |delta| < %s)"
        % (rec["compared_rows"], rec["L_rr_reproduction_rate_overall"],
           rec["L_pp_reproduction_rate_overall"], rec["tolerance"]),
        "- max |delta|: L_rr %s, L_pp %s; offenders %d"
        % (rec["max_abs_delta_L_rr"], rec["max_abs_delta_L_pp"],
           rec["n_offenders"]),
        "- %s" % rec["consequence"], "",
        "| program | n | L_rr ok | bit-identical | max d | L_pp ok | "
        "bit-identical | max d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, row in rec["per_program"].items():
        lines.append("| %s | %d | %d | %d | %s | %d | %d | %s |" % (
            label, row["n"], row["L_rr_reproduced"], row["L_rr_bit_identical"],
            row["L_rr_max_abs_delta"], row["L_pp_reproduced"],
            row["L_pp_bit_identical"], row["L_pp_max_abs_delta"]))
    lines += [
        "", "## Three-cell decomposition", "",
        "| program | scope | n | route share | severe (total>0.30) | "
        "route worst share | route mean | route med | ctx mean | ctx med | "
        "rho(route,total) | rho(ctx,total) | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | --- |",
    ]
    for key, row in p["decomposition"].items():
        if not row.get("n"):
            lines.append("| %s | | 0 | | | | | | | | | | %s |"
                         % (key, row.get("verdict")))
            continue
        lines.append("| %s | %s | %d | %s | %d | %s | %s | %s | %s | %s | %s | %s | **%s** |" % (
            row["program"], row["scope"], row["n"], row["route_algebraic_share"],
            row["n_severe_harm"], row["severe_route_most_negative_share"],
            row["route_mean"], row["route_median"], row["ctx_mean"],
            row["ctx_median"], row["spearman_route_total"],
            row["spearman_ctx_total"], row["verdict"]))
    lines += [
        "", "Sign agreement (share of instances whose component sign equals "
        "total's):", "",
        "| program | scope | route vs total | ctx vs total | route vs ctx |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for key, row in p["decomposition"].items():
        if not row.get("n"):
            continue
        lines.append("| %s | %s | %s | %s | %s |" % (
            row["program"], row["scope"], row["route_sign_agrees_with_total"],
            row["ctx_sign_agrees_with_total"], row["route_ctx_same_sign_rate"]))
    d5 = p["d5_reference"]
    lines += [
        "", "D5 (11 windows, quoted, not recomputed, not covered by this "
        "package): %s, route share %s, severe route-worst %s."
        % (d5["verdict"], d5["route_share"], d5["severe_route_most_negative"]),
        "", "## Serving window unmodified by the Program (astra 2.1 recomputed)",
        "", "%s" % p["serving_window_unmodified"]["definition"], "",
        "| program | scope | evaluable | unmodified | share | total mean "
        "(unmod) | total median (unmod) | total>0.30 (unmod) | severe total | "
        "severe share on unmod | ctx==0 on unmod |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, row in p["serving_window_unmodified"]["per_program_face"].items():
        dist = row["total_distribution_unmodified"]
        label, scope = key.split("|")
        lines.append("| %s | %s | %d | %d | %s | %s | %s | %s | %d | %s | %d |" % (
            label, scope, row["n_evaluable"], row["n_serving_unmodified"],
            row["share_serving_unmodified"], dist.get("mean"),
            dist.get("median"), dist.get("share_severe_harm"),
            row["severe_harm_instances"],
            row["share_of_severe_harm_on_unmodified_windows"],
            row["ctx_exactly_zero_on_unmodified"]))
    lines += ["", "## Block training-corpus invariance (R4C section 4, re-checked)", ""]
    for block, inv in p["provenance"]["block_invariance"].items():
        lines.append(
            "- %s: %d windows from %d train series, anchors retained %d/%d, "
            "origins checked %d, identical window sets %s, identical train "
            "series %s"
            % (block, inv["n_windows"], inv["n_train_series"],
               len(inv["anchors_retained"]), len(inv["anchors_all"]),
               len(inv["origins_checked"]), inv["identical_window_sets"],
               inv["identical_train_series"]))
    lines += ["", "## Fits", "", json.dumps(b, ensure_ascii=False), ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    payload, rows = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    THREE_CELL_STORE.parent.mkdir(parents=True, exist_ok=True)
    # Written unrounded: a prediction store whose losses are rounded cannot be
    # reconciled against m_r0k's at the 1e-9 tolerance this package used.
    THREE_CELL_STORE.write_text(
        json.dumps(three_cell_store(rows, payload),
                   ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "boundary": payload["boundary"],
        "reconciliation": {k: v for k, v in payload["reconciliation"].items()
                           if k not in ("offenders", "per_program")},
        "verdicts": payload["verdicts"],
        "decomposition": {
            k: {n: v[n] for n in (
                "n", "route_algebraic_share", "n_severe_harm",
                "severe_route_most_negative_share", "route_mean", "ctx_mean",
                "spearman_route_total", "spearman_ctx_total", "verdict")
                if n in v}
            for k, v in payload["decomposition"].items()},
    }, ensure_ascii=False, indent=1))
    print("wrote %s" % OUT_JSON.relative_to(PROJECT_ROOT).as_posix())
    print("wrote %s" % OUT_MD.relative_to(PROJECT_ROOT).as_posix())
    print("wrote %s" % THREE_CELL_STORE.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
