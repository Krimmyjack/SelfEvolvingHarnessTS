"""D1: routing-harm 0-LLM diagnostic on Source-v3 exposed windows.

One audit script, one artifact.  Does not modify existing files, does not
touch held-out / [80:120] / UCR TEST, does not change thresholds.

Ridge fits hard-capped at 400.  Predictions for S1 are taken from the same
2-fit path as scoped_evaluate (copied here because the evaluator does not
return raw_prediction / program_prediction).  Per-channel Ridge is a
minimal in-script variant; scoped_serving_evaluator.py is not edited.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p4 import main_experiment_contract as contract
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import smoke_live_scope as smoke
from evaluation.main_protocol_p4.scope_spec import ScopeSpec
from evaluation.main_protocol_p4.scoped_serving_evaluator import (
    CONTEXT_LENGTH,
    HORIZON,
    ServingContextDegenerate,
    _design,
    _prepare,
    _serve,
    _training_windows,
)
from SelfEvolvingHarnessTS.contracts.observables import (
    OBSERVABLE_FEATURES,
    OBSERVABLE_NUMERIC_BIN_LABELS,
    observable_numeric_bin,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
    seasonal_scale,
    smase,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ART = PROJECT_ROOT / (
    "artifacts/main_protocol/p4w3b_source_line_v3_clean_post_fix_replicate_1.json"
)
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/p4ab_routing_harm_diagnostic.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/p4ab_routing_harm_diagnostic.md"

FIT_CAP = 400
MATERIAL = 0.005
SEVERE = 0.30
CONTEXT = CONTEXT_LENGTH
BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20260903
PERIOD = 24
MIN_PAIRS = 32

NUMERIC_FEATURES = tuple(
    name for name, kind in OBSERVABLE_FEATURES.items() if kind == "number"
)
BIN_INDEX = {label: i for i, label in enumerate(OBSERVABLE_NUMERIC_BIN_LABELS)}


class FitCap(RuntimeError):
    """Hard stop: Ridge fit ledger hit 400."""


class Ledger:
    def __init__(self, cap: int = FIT_CAP) -> None:
        self.cap = cap
        self.n = 0
        self.pooled = 0
        self.per_channel = 0

    def charge(self, n: int, *, bucket: str) -> None:
        if n <= 0:
            return
        if self.n + n > self.cap:
            raise FitCap("consumer_fits %d + %d exceeds %d" % (self.n, n, self.cap))
        self.n += n
        if bucket == "pooled":
            self.pooled += n
        elif bucket == "per_channel":
            self.per_channel += n


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("%s is not an object" % path)
    return payload


def _steps(program: Any) -> tuple[tuple[str, dict], ...]:
    rows = program or ()
    return tuple((str(step["op"]), dict(step.get("params") or {})) for step in rows)


def _cards(variant: Mapping[str, Any], uids: list[str], origin: int
           ) -> dict[str, dict[str, float]]:
    return smoke._feature_cards(variant, uids, int(origin))


def _bin_index_vec(card: Mapping[str, Any]) -> np.ndarray:
    out = []
    for name in NUMERIC_FEATURES:
        value = card.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            out.append(float("nan"))
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            out.append(float("nan"))
            continue
        try:
            label = observable_numeric_bin(name, numeric)
        except ValueError:
            out.append(float("nan"))
            continue
        out.append(float(BIN_INDEX[label]))
    return np.asarray(out, dtype=np.float64)


def _metric_scale(raw: np.ndarray, origin: int) -> float:
    prefix = np.asarray(raw[:origin], dtype=np.float64)
    return float(seasonal_scale(
        prefix, np.isfinite(prefix), period=PERIOD, min_pairs=MIN_PAIRS))


def _behavior(raw: np.ndarray, origin: int, compiled: Any, scale: float
              ) -> dict[str, float]:
    window = np.asarray(raw[origin - CONTEXT:origin], dtype=np.float64)
    prepared, moved, _trace = _prepare(window, compiled)
    baseline = forecast_runtime._linear_integrity(window)
    delta = np.abs(np.asarray(prepared, dtype=np.float64) - baseline)
    last48 = int(np.count_nonzero(
        ~np.isclose(prepared[-48:], baseline[-48:], equal_nan=True)))
    fraction = float(moved) / float(CONTEXT)
    if moved <= 0 or not math.isfinite(scale) or scale <= 0:
        magnitude = 0.0
    else:
        magnitude = float(np.nansum(delta)) / (float(moved) * float(scale))
    return {
        "moved": float(moved),
        "mod_fraction": fraction,
        "mod_magnitude": magnitude,
        "last48_moved": float(last48),
        "vector": (fraction, magnitude, float(last48)),
    }


def scoped_evaluate_capture(
    roster, values, compiled, config, *, origin: int, scope, ledger: Ledger,
    view=None,
) -> dict[str, Any]:
    """Same 2-fit path as scoped_evaluate, also returning the predictions.

    Copied rather than editing scoped_serving_evaluator.py.
    """
    view = view or views.IdentityView()
    eval_rows = [row for row in roster if row["role"] == "eval"]
    eval_uids = [str(row["series_uid"]) for row in eval_rows]
    selected = {str(uid) for uid in scope}
    in_scope = np.array([uid in selected for uid in eval_uids], dtype=bool)

    raw_windows = _training_windows(roster, values, config, int(origin))
    behavior, steps_run = 0, []
    prepared_windows = []
    for window in raw_windows:
        prepared, moved, trace = _prepare(window, compiled)
        behavior += moved
        steps_run.extend(trace)
        prepared_windows.append(prepared)
    raw_only = [forecast_runtime._linear_integrity(w) for w in raw_windows]

    raw_contexts, prepared_contexts, truths, metric_scales = [], [], [], []
    degenerate: list[str] = []
    for uid in eval_uids:
        raw = np.asarray(values[uid], dtype=np.float64)
        window = raw[int(origin) - CONTEXT_LENGTH:int(origin)]
        raw_context = forecast_runtime._linear_integrity(window)
        raw_contexts.append(raw_context)
        if compiled is None:
            prepared_contexts.append(raw_context)
        else:
            served, _moved, _trace = _prepare(window, compiled)
            if uid in selected:
                _c, _s, method = forecast_runtime._center_scale(np, served)
                if method == "scale_floor_fallback":
                    degenerate.append(uid)
            prepared_contexts.append(served)
        truths.append(raw[int(origin):int(origin) + HORIZON])
        metric_scales.append(
            seasonal_scale(raw[:int(origin)], np.isfinite(raw[:int(origin)]),
                           period=int(config["period"]), min_pairs=32)
        )
    if degenerate:
        raise ServingContextDegenerate(
            "preparing the served context flattened %d scoped series"
            % len(degenerate))

    raw_design = _design(raw_only, view)
    ledger.charge(1, bucket="pooled")
    raw_prediction = _serve(raw_contexts, view, *raw_design)
    if compiled is None or not in_scope.any():
        program_prediction = None
        prediction = raw_prediction
    else:
        program_design = _design(prepared_windows, view)
        ledger.charge(1, bucket="pooled")
        program_prediction = _serve(prepared_contexts, view, *program_design)
        prediction = np.where(in_scope[:, None], program_prediction, raw_prediction)

    losses, raw_losses = [], []
    for index, (truth, scale) in enumerate(zip(truths, metric_scales)):
        observed = np.isfinite(truth)
        if not observed.any():
            raise RuntimeError("evaluation future contains no observed truth")
        losses.append(smase(truth[observed], prediction[index][observed], scale=scale))
        raw_losses.append(
            smase(truth[observed], raw_prediction[index][observed], scale=scale))
    return {
        "eval_uids": eval_uids,
        "per_view_smase": [float(v) for v in losses],
        "static_per_view_smase": [float(v) for v in raw_losses],
        "raw_prediction": raw_prediction,
        "program_prediction": program_prediction,
        "prediction": prediction,
        "metric_scales": [float(s) for s in metric_scales],
        "consumer_fits": 1 if program_prediction is None else 2,
        "behavior_point_count": behavior,
    }


def per_channel_one(
    uid: str, values: Mapping[str, Any], compiled: Any, config: Mapping[str, Any],
    origin: int, ledger: Ledger, view=None,
) -> dict[str, Any]:
    view = view or views.IdentityView()
    raw = np.asarray(values[uid], dtype=np.float64)
    windows = []
    for anchor in config["anchors"]:
        anchor = int(anchor)
        if anchor + HORIZON > origin:
            continue
        windows.append(raw[anchor - CONTEXT_LENGTH:anchor + HORIZON])
    if not windows:
        raise RuntimeError("no per-channel training windows for %s @ %s" % (uid, origin))
    raw_only = [forecast_runtime._linear_integrity(w) for w in windows]
    prepared = [_prepare(w, compiled)[0] for w in windows]
    raw_ctx = forecast_runtime._linear_integrity(raw[origin - CONTEXT:origin])
    prep_ctx, moved, _trace = _prepare(raw[origin - CONTEXT:origin], compiled)
    raw_design = _design(raw_only, view)
    ledger.charge(1, bucket="per_channel")
    raw_pred = _serve([raw_ctx], view, *raw_design)[0]
    prog_design = _design(prepared, view)
    ledger.charge(1, bucket="per_channel")
    prog_pred = _serve([prep_ctx], view, *prog_design)[0]
    truth = raw[origin:origin + HORIZON]
    scale = _metric_scale(raw, origin)
    observed = np.isfinite(truth)
    raw_loss = smase(truth[observed], raw_pred[observed], scale=scale)
    prog_loss = smase(truth[observed], prog_pred[observed], scale=scale)
    return {
        "gain": float(raw_loss - prog_loss),
        "moved": int(moved),
        "raw_smase": float(raw_loss),
        "program_smase": float(prog_loss),
        "fits": 2,
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = labels.astype(bool)
    if pos.sum() == 0 or (~pos).sum() == 0:
        return None
    finite = np.isfinite(scores)
    if finite[pos].sum() == 0 or finite[~pos].sum() == 0:
        return None
    s = scores[finite]
    y = pos[finite]
    order = np.argsort(s)
    y = y[order]
    n_pos = float(y.sum())
    n_neg = float((~y).sum())
    ranks = np.empty(y.size, dtype=np.float64)
    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and s[order][j + 1] == s[order][i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[i:j + 1] = avg
        i = j + 1
    sum_pos = float(ranks[y].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _bootstrap_auc(rows: list[dict[str, Any]], score_key: str, rng: np.random.Generator
                   ) -> dict[str, Any]:
    empty = {
        "auc": None, "ci95": [None, None], "n_bootstrap": 0,
        "verdict": "DOES_NOT_SEPARATE",
    }
    if not rows:
        return empty
    uids = sorted({row["uid"] for row in rows})
    if not uids:
        return empty
    by_uid: dict[str, list[dict[str, Any]]] = {uid: [] for uid in uids}
    for row in rows:
        by_uid[row["uid"]].append(row)
    point_scores = np.asarray([row[score_key] for row in rows], dtype=np.float64)
    point_labels = np.asarray([row["harmed_material"] for row in rows], dtype=bool)
    point = _auc(point_scores, point_labels)
    samples = []
    for _ in range(BOOTSTRAP):
        draw = rng.choice(uids, size=len(uids), replace=True)
        bundled = []
        for uid in draw:
            bundled.extend(by_uid[uid])
        scores = np.asarray([row[score_key] for row in bundled], dtype=np.float64)
        labels = np.asarray([row["harmed_material"] for row in bundled], dtype=bool)
        value = _auc(scores, labels)
        if value is not None:
            samples.append(value)
    if not samples or point is None:
        return {
            "auc": point, "ci95": [None, None], "n_bootstrap": len(samples),
            "verdict": "DOES_NOT_SEPARATE",
        }
    lo, hi = np.percentile(samples, [2.5, 97.5])
    separates = bool(point >= 0.75 and float(lo) > 0.5)
    return {
        "auc": round(float(point), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "n_bootstrap": len(samples),
        "verdict": "SEPARATES" if separates else "DOES_NOT_SEPARATE",
    }


def _oor(vector: tuple[float, float, float], mins: np.ndarray, maxs: np.ndarray
         ) -> tuple[bool, float]:
    v = np.asarray(vector, dtype=np.float64)
    width = np.maximum(maxs - mins, 0.0)
    below = np.clip(mins - v, 0.0, None)
    above = np.clip(v - maxs, 0.0, None)
    excess = below + above
    dists = []
    for i, extra in enumerate(excess):
        if extra <= 0:
            dists.append(0.0)
        elif width[i] <= 0:
            dists.append(1.0)
        else:
            dists.append(float(extra / width[i]))
    beh_dist = float(max(dists) if dists else 0.0)
    return bool(beh_dist > 0), beh_dist


def _nearest_l1(vec: np.ndarray, cloud: list[np.ndarray]) -> float | None:
    if not cloud:
        return None
    best = None
    n = float(vec.size)
    for other in cloud:
        delta = np.abs(vec - other)
        if not np.isfinite(delta).all() or not np.isfinite(vec).all():
            continue
        dist = float(delta.sum() / n)
        if best is None or dist < best:
            best = dist
    return best


def _windows_from_artifact(art: dict[str, Any], support_a: list[str]
                           ) -> list[dict[str, Any]]:
    out = []
    for rnd in art["rounds"]:
        origin = int(rnd["origin"])
        program = rnd.get("winner_program")
        scope = rnd.get("winner_serving_scope")
        if not program or not scope:
            continue
        probe = None
        for row in rnd.get("probes") or ():
            if row.get("kind") == "probe" and row.get("per_series_gain"):
                probe = row
                break
        delayed = rnd.get("delayed_gate")
        reenc = rnd.get("re_encounter_gate")
        if delayed:
            out.append({
                "kind": "delayed",
                "draft_origin": origin,
                "read_origin": int(delayed["read_origin"]),
                "program": program,
                "predicate": scope,
                "gain_vector": list(delayed["per_series_gain"]),
                "probe": probe,
                "previous_kind": "support_probe",
            })
        if reenc:
            out.append({
                "kind": "re_encounter",
                "draft_origin": origin,
                "read_origin": int(reenc["read_origin"]),
                "program": program,
                "predicate": scope,
                "gain_vector": list(reenc["per_series_gain"]),
                "probe": probe,
                "previous_kind": "delayed",
                "previous_gain_vector": list(delayed["per_series_gain"]) if delayed else None,
            })
    for row in out:
        gains = row["gain_vector"]
        s_nonzero = [support_a[i] for i, g in enumerate(gains) if float(g) != 0.0]
        row["s_nonzero"] = s_nonzero
        row["gains"] = {support_a[i]: float(g) for i, g in enumerate(gains)}
        if row["probe"] is not None:
            pg = row["probe"]["per_series_gain"]
            row["probe_s"] = [support_a[i] for i, g in enumerate(pg) if float(g) != 0.0]
            row["probe_gains"] = {support_a[i]: float(g) for i, g in enumerate(pg)}
            row["probe_origin"] = int(row["draft_origin"])
        else:
            row["probe_s"] = []
            row["probe_gains"] = {}
            row["probe_origin"] = int(row["draft_origin"])
        if row["kind"] == "re_encounter" and row.get("previous_gain_vector"):
            prev = row["previous_gain_vector"]
            row["previous_s"] = [
                support_a[i] for i, g in enumerate(prev) if float(g) != 0.0]
        elif row["kind"] == "delayed":
            row["previous_s"] = list(row["probe_s"])
        else:
            row["previous_s"] = []
    return out


def build() -> dict[str, Any]:
    art = _load(SOURCE_ART)
    groups = contract.cohorts()
    cell, variant = baselines._cell(groups["source"])
    support_a = list(cell.support_a)
    ledger = Ledger()
    windows_spec = _windows_from_artifact(art, support_a)
    view = views.IdentityView()

    # Precompute evidence clouds keyed by (draft_origin, kind)
    evidence: dict[tuple[int, str], dict[str, Any]] = {}
    series_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    counterfactual_windows: list[dict[str, Any]] = []
    stopped = None

    try:
        for spec in windows_spec:
            origin = int(spec["read_origin"])
            draft_origin = int(spec["draft_origin"])
            at = forecast_p4._cell_at(cell, origin)
            config = forecast_p4._config(origin)
            roster = at.roster("support_a")
            eval_uids = [str(row["series_uid"]) for row in roster if row["role"] == "eval"]
            if eval_uids != support_a:
                raise RuntimeError("eval uid order drifted from support_a")
            cards = _cards(variant, support_a, origin)
            resolved = set(ScopeSpec.from_dict(spec["predicate"]).resolve(cards))
            s_nonzero = set(spec["s_nonzero"])
            s = set(resolved)
            s_diff = {
                "in_predicate_not_nonzero": sorted(s - s_nonzero),
                "in_nonzero_not_predicate": sorted(s_nonzero - s),
            }
            previous = set(spec.get("previous_s") or ())
            steps = _steps(spec["program"])
            executor = ScopeExecutor(
                roster, at.values, config,
                evaluate_fn=views.forecast_runtime._evaluate,
                max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)
            compiled = executor._compiled(steps)

            captured = scoped_evaluate_capture(
                roster, at.values, compiled, config, origin=origin, scope=s,
                ledger=ledger, view=view)
            uid_index = {uid: i for i, uid in enumerate(captured["eval_uids"])}

            # Evidence E
            e_uids: list[str] = []
            e_bins: list[np.ndarray] = []
            e_beh: list[tuple[float, float, float]] = []
            if spec["kind"] == "delayed":
                support_origin = int(spec["probe_origin"])
                support_cards = _cards(variant, support_a, support_origin)
                support_resolved = set(
                    ScopeSpec.from_dict(spec["predicate"]).resolve(support_cards))
                for uid in sorted(support_resolved):
                    gain = spec["probe_gains"].get(uid, 0.0)
                    if gain >= -MATERIAL:
                        e_uids.append(uid)
                        e_bins.append(_bin_index_vec(support_cards[uid]))
                        raw = np.asarray(variant[uid], dtype=np.float64)
                        scale = _metric_scale(raw, support_origin)
                        beh = _behavior(raw, support_origin, compiled, scale)
                        e_beh.append(beh["vector"])
            else:
                delayed_key = (draft_origin, "delayed")
                prior = evidence.get(delayed_key, {})
                e_uids.extend(prior.get("uids") or ())
                e_bins.extend(list(prior.get("bins") or ()))
                e_beh.extend(list(prior.get("beh") or ()))
                delayed_spec = next(
                    w for w in windows_spec
                    if w["draft_origin"] == draft_origin and w["kind"] == "delayed"
                )
                delayed_origin = int(delayed_spec["read_origin"])
                delayed_cards = _cards(variant, support_a, delayed_origin)
                for uid in delayed_spec["s_nonzero"]:
                    if delayed_spec["gains"][uid] >= -MATERIAL:
                        if uid not in e_uids:
                            e_uids.append(uid)
                            e_bins.append(_bin_index_vec(delayed_cards[uid]))
                            raw = np.asarray(variant[uid], dtype=np.float64)
                            scale = _metric_scale(raw, delayed_origin)
                            beh = _behavior(raw, delayed_origin, compiled, scale)
                            e_beh.append(beh["vector"])
            evidence[(draft_origin, spec["kind"])] = {
                "uids": e_uids, "bins": e_bins, "beh": e_beh,
            }
            if e_beh:
                beh_arr = np.asarray(e_beh, dtype=np.float64)
                beh_min = beh_arr.min(axis=0)
                beh_max = beh_arr.max(axis=0)
            else:
                beh_min = beh_max = None

            pc_gains: dict[str, float | None] = {}
            pc_harmed_m = 0
            pc_harmed_s = 0
            pc_new_harmed = 0
            pooled_new_harmed = 0
            msh_pc = 0.0
            msh_pooled = 0.0
            for uid in sorted(s):
                idx = uid_index[uid]
                gain = float(spec["gains"].get(uid, 0.0))
                raw = np.asarray(variant[uid], dtype=np.float64)
                scale = _metric_scale(raw, origin)
                beh = _behavior(raw, origin, compiled, scale)
                raw_pred = captured["raw_prediction"][idx]
                prog_pred = captured["program_prediction"][idx]
                div = float(np.mean(np.abs(prog_pred - raw_pred)) / scale)
                dist = _nearest_l1(_bin_index_vec(cards[uid]), e_bins)
                if beh_min is None:
                    beh_oor, beh_dist = None, None
                else:
                    beh_oor, beh_dist = _oor(beh["vector"], beh_min, beh_max)
                attr = "CONTINUING" if uid in previous else "NEW_ENTRANT"
                pc = per_channel_one(
                    uid, at.values, compiled, config, origin, ledger, view=view)
                pc_gains[uid] = pc["gain"]
                harmed_m = bool(gain < -MATERIAL)
                harmed_s = bool(gain < -SEVERE)
                if pc["gain"] < -MATERIAL:
                    pc_harmed_m += 1
                if pc["gain"] < -SEVERE:
                    pc_harmed_s += 1
                msh_pc = max(msh_pc, max(0.0, -pc["gain"]))
                msh_pooled = max(msh_pooled, max(0.0, -gain))
                if attr == "NEW_ENTRANT" and harmed_m:
                    pooled_new_harmed += 1
                if attr == "NEW_ENTRANT" and pc["gain"] < -MATERIAL:
                    pc_new_harmed += 1
                series_rows.append({
                    "uid": uid,
                    "window": origin,
                    "kind": spec["kind"],
                    "draft_origin": draft_origin,
                    "attribution": attr,
                    "gain": gain,
                    "harmed_material": harmed_m,
                    "harmed_severe": harmed_s,
                    "div": div,
                    "dist": dist,
                    "beh_oor": beh_oor,
                    "beh_dist": beh_dist,
                    "mod_fraction": beh["mod_fraction"],
                    "mod_magnitude": beh["mod_magnitude"],
                    "moved": int(beh["moved"]),
                    "gain_per_channel": pc["gain"],
                    "last48_moved": beh["last48_moved"],
                })

            window_rows.append({
                "origin": draft_origin,
                "read_origin": origin,
                "kind": spec["kind"],
                "program": spec["program"],
                "predicate": spec["predicate"],
                "S": sorted(s),
                "S_size": len(s),
                "E_size": len(e_uids),
                "E": sorted(e_uids),
                "s_diff_vs_nonzero": s_diff,
                "n_harmed_material": sum(
                    1 for uid in s if spec["gains"].get(uid, 0.0) < -MATERIAL),
                "n_harmed_severe": sum(
                    1 for uid in s if spec["gains"].get(uid, 0.0) < -SEVERE),
            })
            if spec["kind"] == "re_encounter":
                counterfactual_windows.append({
                    "read_origin": origin,
                    "draft_origin": draft_origin,
                    "pooled": {
                        "n_S": len(s),
                        "n_harmed_material": sum(
                            1 for uid in s if spec["gains"].get(uid, 0.0) < -MATERIAL),
                        "n_harmed_severe": sum(
                            1 for uid in s if spec["gains"].get(uid, 0.0) < -SEVERE),
                        "n_new_entrant_harmed_material": pooled_new_harmed,
                        "msh": msh_pooled,
                    },
                    "per_channel": {
                        "n_harmed_material": pc_harmed_m,
                        "n_harmed_severe": pc_harmed_s,
                        "n_new_entrant_harmed_material": pc_new_harmed,
                        "msh": msh_pc,
                    },
                    "new_entrant_harmed_halved": (
                        pc_new_harmed <= pooled_new_harmed / 2.0
                        if pooled_new_harmed else pc_new_harmed == 0
                    ),
                    "msh_not_up": msh_pc <= msh_pooled + 1e-12,
                })
    except FitCap as exc:
        stopped = str(exc)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc = {}
    for key, name in (
        ("div", "S1_div"),
        ("dist", "S2_dist"),
        ("beh_dist", "S3_beh_dist"),
        ("mod_fraction", "S4_mod_fraction"),
        ("mod_magnitude", "S4_mod_magnitude"),
    ):
        usable = [row for row in series_rows if row.get(key) is not None
                  and (isinstance(row[key], (int, float)) and math.isfinite(float(row[key])))]
        auc[name] = _bootstrap_auc(usable, key, rng)
        auc[name]["n"] = len(usable)
        auc[name]["n_harmed_material"] = sum(1 for row in usable if row["harmed_material"])

    severe_rows = [row for row in series_rows if row["harmed_severe"]]
    n_severe = len(severe_rows)
    n_routing = sum(1 for row in severe_rows if int(row["moved"]) <= 1)
    routing_share = (n_routing / n_severe) if n_severe else None
    routing_verdict = (
        "ROUTING_HARM_CONFIRMED" if routing_share is not None and routing_share >= 0.5
        else "ROUTING_HARM_NOT_DOMINANT"
    )

    lower_tail = 0
    higher_tail = 0
    for row in counterfactual_windows:
        halved = row["new_entrant_harmed_halved"] and row["msh_not_up"]
        reverse = (
            row["per_channel"]["n_new_entrant_harmed_material"]
            >= 2 * row["pooled"]["n_new_entrant_harmed_material"]
            and row["pooled"]["n_new_entrant_harmed_material"] > 0
            and row["per_channel"]["msh"] >= row["pooled"]["msh"] - 1e-12
        )
        row["lower_tail_hit"] = bool(halved)
        row["higher_tail_hit"] = bool(reverse)
        lower_tail += int(halved)
        higher_tail += int(reverse)
    if lower_tail >= 2:
        cf_verdict = "PER_CHANNEL_LOWER_TAIL"
    elif higher_tail >= 2:
        cf_verdict = "PER_CHANNEL_HIGHER_TAIL"
    else:
        cf_verdict = "NO_CLEAR_DIFFERENCE"

    n_harmed = sum(1 for row in series_rows if row["harmed_material"])
    main_done = stopped is None and len(window_rows) == len(windows_spec)
    candidate_signals = [
        name for name, payload in auc.items()
        if name.startswith("S") and name.split("_")[0] in {"S1", "S2", "S3"}
        and payload.get("verdict") == "SEPARATES"
    ]
    # S1/S2/S3 are the outcome-free candidates; S4 is the rejected control.
    if (not main_done) or n_harmed < 6:
        verdict = "INCONCLUSIVE_SAMPLE"
    elif candidate_signals:
        verdict = "RISK_FACE_CANDIDATE_IDENTIFIED"
    else:
        verdict = "NO_OUTCOME_FREE_SEPARATOR"

    hit_table = []
    for name in candidate_signals:
        key = {"S1_div": "div", "S2_dist": "dist", "S3_beh_dist": "beh_dist"}[name]
        harmed = [row[key] for row in series_rows if row["harmed_material"]
                  and row[key] is not None and math.isfinite(float(row[key]))]
        safe = [row[key] for row in series_rows if not row["harmed_material"]
                and row[key] is not None and math.isfinite(float(row[key]))]
        if not harmed or not safe:
            continue
        # Youden-style single threshold on the score (higher = more harmed).
        grid = sorted(set(harmed + safe))
        best = None
        for thr in grid:
            tp = sum(1 for v in harmed if v >= thr)
            tn = sum(1 for v in safe if v < thr)
            tpr = tp / len(harmed)
            tnr = tn / len(safe)
            youden = tpr + tnr - 1.0
            if best is None or youden > best[0]:
                best = (youden, thr, tp, tn, tpr, tnr)
        hit_table.append({
            "signal": name,
            "threshold": None if best is None else best[1],
            "youden": None if best is None else round(best[0], 4),
            "tp": None if best is None else best[2],
            "tn": None if best is None else best[3],
            "tpr": None if best is None else round(best[4], 4),
            "tnr": None if best is None else round(best[5], 4),
        })

    report = {
        "stage": "P4AB_ROUTING_HARM_DIAGNOSTIC",
        "status": "COMPLETE" if main_done else "STOPPED_ON_FIT_CAP",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_ONLY_DIAGNOSTIC",
        "data_version": art.get("data_version") or contract.DATA_VERSION,
        "sources": {
            "source_line": str(SOURCE_ART.relative_to(PROJECT_ROOT).as_posix()),
        },
        "windows": window_rows,
        "series_rows": series_rows,
        "auc": auc,
        "routing_harm_check": {
            "n_harmed_severe": n_severe,
            "n_moved_le_1": n_routing,
            "share": routing_share,
            "verdict": routing_verdict,
            "rows": [
                {"uid": r["uid"], "window": r["window"], "gain": r["gain"],
                 "moved": r["moved"], "attribution": r["attribution"]}
                for r in severe_rows
            ],
        },
        "counterfactual": {
            "windows": counterfactual_windows,
            "n_reencounter_lower_tail": lower_tail,
            "n_reencounter_higher_tail": higher_tail,
            "verdict": cf_verdict,
        },
        "verdict": verdict,
        "separating_signals": candidate_signals,
        "single_threshold_hits": hit_table,
        "n_harmed_material": n_harmed,
        "main_windows_completed": main_done,
        "stopped": stopped,
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": ledger.n,
            "consumer_fits_pooled": ledger.pooled,
            "consumer_fits_per_channel": ledger.per_channel,
            "held_out_reads": 0,
            "thresholds_changed": 0,
            "operators_added": 0,
            "artifacts_overwritten": 0,
            "fit_cap": FIT_CAP,
        },
        "deviations": [
            {
                "what": "S1 prediction capture",
                "why": (
                    "scoped_evaluate computes raw_prediction/program_prediction "
                    "but does not return them; this audit copies the 2-fit path "
                    "in-script and does not edit scoped_serving_evaluator.py"
                ),
            },
            {
                "what": "22-d binned features",
                "why": (
                    "this checkout's numeric observable vocabulary is %d names; "
                    "S2 uses those frozen bins"
                    % len(NUMERIC_FEATURES)
                ),
            },
            {
                "what": "optional Support-A secondary window set",
                "why": "not run; main window set completed first and is the registered reading",
            },
        ],
        "spec_tensions": [
            {
                "what": "scoped_evaluate return dict omits predictions",
                "reading": "captured in-script; evaluator left untouched",
            },
            {
                "what": "round 2856 delayed_gate.passes=False vs delayed_event approved",
                "reading": "this audit uses delayed_gate per_series_gain, not delayed_event",
            },
        ],
        "releases": "NONE",
    }
    return report


def _md(report: dict[str, Any]) -> str:
    lines = [
        "# p4ab routing-harm diagnostic",
        "",
        "0 LLM. Ridge fits %d / %d (pooled %d, per-channel %d)."
        % (report["boundary"]["consumer_fits"], FIT_CAP,
           report["boundary"]["consumer_fits_pooled"],
           report["boundary"]["consumer_fits_per_channel"]),
        "",
        "## (1) Per-window S / E / harm",
        "",
        "| kind | origin | read | |S| | |E| | harmed_m | harmed_s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for w in report["windows"]:
        lines.append("| %s | %s | %s | %d | %d | %d | %d |" % (
            w["kind"], w["origin"], w["read_origin"], w["S_size"], w["E_size"],
            w["n_harmed_material"], w["n_harmed_severe"]))
    lines += ["", "## (2) AUC and harmed_severe", ""]
    for name, payload in report["auc"].items():
        lines.append("- %s: AUC=%s CI=%s **%s** (n=%s, harmed=%s)" % (
            name, payload.get("auc"), payload.get("ci95"), payload.get("verdict"),
            payload.get("n"), payload.get("n_harmed_material")))
    lines.append("")
    for row in report["routing_harm_check"]["rows"]:
        lines.append("- severe %s @%s gain=%s moved=%s %s" % (
            row["uid"], row["window"], round(row["gain"], 4), row["moved"],
            row["attribution"]))
    lines += [
        "",
        "routing_harm_check: **%s** (moved<=1 share=%s)"
        % (report["routing_harm_check"]["verdict"],
           report["routing_harm_check"]["share"]),
        "",
        "## (3) pooled vs per-channel (re-encounter)",
        "",
    ]
    for w in report["counterfactual"]["windows"]:
        lines.append(
            "- %s pooled new-harmed=%s msh=%s | per-channel new-harmed=%s msh=%s"
            % (w["read_origin"],
               w["pooled"]["n_new_entrant_harmed_material"],
               round(w["pooled"]["msh"], 4),
               w["per_channel"]["n_new_entrant_harmed_material"],
               round(w["per_channel"]["msh"], 4)))
    lines += [
        "",
        "## (4) Verdicts",
        "",
        "- separation/total: **%s**" % report["verdict"],
        "- routing: **%s**" % report["routing_harm_check"]["verdict"],
        "- counterfactual: **%s**" % report["counterfactual"]["verdict"],
        "",
        "## (5) Fits",
        "",
        json.dumps(report["boundary"], ensure_ascii=False),
        "",
        "AUC is in-sample separation on already-exposed windows, not a claim "
        "that harm is predictable at deployment.",
        "",
        "## (6) Deviations",
        "",
    ]
    for row in report.get("deviations") or ():
        lines.append("- %s: %s" % (row["what"], row["why"]))
    lines += ["", "## (7) Spec tensions", ""]
    for row in report.get("spec_tensions") or ():
        lines.append("- %s — %s" % (row["what"], row["reading"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(_md(report), encoding="utf-8")
    print("verdict              : %s" % report["verdict"])
    print("routing              : %s" % report["routing_harm_check"]["verdict"])
    print("counterfactual       : %s" % report["counterfactual"]["verdict"])
    print("fits pooled/pc/total : %s / %s / %s" % (
        report["boundary"]["consumer_fits_pooled"],
        report["boundary"]["consumer_fits_per_channel"],
        report["boundary"]["consumer_fits"]))
    print("n series_rows        : %d" % len(report["series_rows"]))
    print("n harmed_material    : %s" % report["n_harmed_material"])
    print("wrote %s" % OUT_JSON.relative_to(PROJECT_ROOT).as_posix())
    print("wrote %s" % OUT_MD.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
