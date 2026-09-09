"""R4D-B: the same three-cell decomposition under a per-channel Consumer.

The only variable
-----------------
R4D-A fitted one pooled Ridge per (block, program) on the training half of the
block and found the shared-model channel (``route``) carrying 70-80% of every
program's effect.  Here each *served* series gets its own models, fitted on its
own anchored training windows ``raw_i[anchor-192 : anchor+48]`` -- the same
anchors, the same retention rule ``anchor + 48 <= origin``, the same Ridge, the
same metric, the same programs applied to the same 240-step training windows and
192-step serving contexts.  Nothing else moves.

Under per-channel the ``route`` channel changes meaning: it is no longer "the
model everybody shares was refitted on other series' prepared rows" but "this
series' own model was refitted on its own prepared rows".  The readouts are
therefore reported side by side with the pooled numbers from R4D-A rather than
merged with them.

Why this costs 240 fits and not 2520
------------------------------------
With the anchors frozen at [312...852] and every origin in the store >= 1176,
a served series' own ten training windows do not depend on the origin either
(R4C section 4, re-checked here for every series).  One series therefore has one
raw model and one model per program, and every one of its faces is predicted out
of those.  ``_serve`` fits once and predicts every context it is handed, and the
frozen Ridge solve never reads ``x_eval``, so raw and prepared contexts share a
fit: 3 fits per series x 80 series = 240, refused at 260.

Run: ``python -m evaluation.main_protocol_p4.run_r4d_b_perchannel_three_cell``
0 LLM calls, 0 held-out reads, 240 physical Ridge fits (hard cap 260).
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
from evaluation.main_protocol_p4 import audit_cross_fitted_targeting as targeting
from evaluation.main_protocol_p4 import audit_r4a_pattern_identifiability as r4a
from evaluation.main_protocol_p4 import audit_r4c_imputation_donor_conditions as r4c
from evaluation.main_protocol_p4 import audit_r4e_channel_identifiability as r4e
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_hec1 as hec1
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import run_r4d_a_action_decomposition as r4d
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import seasonal_scale
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POOLED_STORE = PROJECT_ROOT / "_scratch/r4d_a_three_cell_store.json"
PC_STORE = PROJECT_ROOT / "_scratch/r4d_b_perchannel_three_cell_store.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/r4d_b_perchannel_three_cell.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/r4d_b_perchannel_three_cell.md"

CONTEXT = r4d.CONTEXT
HORIZON = r4d.HORIZON
PROGRAMS = r4d.PROGRAMS
ANC, W2 = r4d.ANC, r4d.W2
SEVERE = r4d.SEVERE_HARM
FACES = r4d.FACES
CHANNELS = ("route", "ctx", "total")

FIT_TARGET = 240
FIT_HARD_CAP = 260

# --- frozen before any number was seen (request 4.2 / R4A-b vocabulary) ---
SERIES_UID_SHARE = 0.30
SERIES_SPEARMAN = 0.50
COHORT_POSITION_SHARE = 0.50
NOISE_SPEARMAN = 0.30
# --- harm-shape rule, frozen here: a shift is a shift of at least this much ---
SHAPE_DELTA = 0.05


# ---------------------------------------------------------------------------
# 1. per-series models: three fits per served series
# ---------------------------------------------------------------------------


def _own_windows(raw: np.ndarray, anchors: Sequence[int], origin: int
                 ) -> tuple[list[np.ndarray], list[int]]:
    windows, kept = [], []
    for anchor in anchors:
        anchor = int(anchor)
        if anchor + HORIZON > origin:
            continue
        windows.append(raw[anchor - CONTEXT:anchor + HORIZON])
        kept.append(anchor)
    return windows, kept


def run_block(block: str, faces: Sequence[Mapping[str, Any]], view: Any,
              fits: r4d.FitLedger, reads: r4d.ReadLedger) -> dict[str, Any]:
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
    eval_uids = [str(r["series_uid"]) for r in ref_roster if r["role"] == "eval"]
    for row in faces:
        if row["eval_uids"] != eval_uids:
            raise AssertionError("store eval_uids differ from the roster's eval half at %s"
                                 % ((row["position"], row["face"]),))
    executor = ScopeExecutor(
        ref_roster, ref_at.values, ref_config,
        evaluate_fn=views.forecast_runtime._evaluate,
        max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)
    compiled = {label: executor._compiled(r4d.PROGRAM_STEPS[label]) for label in PROGRAMS}
    anchors = [int(a) for a in ref_config["anchors"]]

    instances: list[dict[str, Any]] = []
    per_series: list[dict[str, Any]] = []
    for uid in eval_uids:
        raw = np.asarray(cell.values[uid], dtype=np.float64)
        windows, kept = _own_windows(raw, anchors, ref_origin)
        for anchor in kept:
            reads.account(uid, anchor + HORIZON)
        # the one-fit-per-series design is void if the series' own window set
        # moved between this block's origins; checked, not assumed
        invariant = True
        for origin in origins[1:]:
            other, kept_other = _own_windows(raw, anchors, origin)
            if kept_other != kept or not all(
                    np.array_equal(a, b, equal_nan=True) for a, b in zip(other, windows)):
                invariant = False
        if not invariant:
            raise AssertionError("series %s does not keep one training window set across "
                                 "the origins of block %s" % (uid, block))
        record: dict[str, Any] = {"uid": uid, "block": block, "n_windows": len(windows),
                                  "anchors_retained": kept, "fittable": {}, "train_moved": {}}
        # designs: raw + one per program; a scale-floor training context makes the
        # series NOT_FITTABLE for that model and costs nothing
        designs: dict[str, Any] = {}
        try:
            designs["raw"] = scoped._design(
                [forecast_runtime._linear_integrity(w) for w in windows], view)
            record["fittable"]["raw"] = True
        except RuntimeError as exc:
            record["fittable"]["raw"] = False
            record["not_fittable_because"] = str(exc)
        for label in PROGRAMS:
            prepared, moved = [], 0
            for window in windows:
                out, points, _trace = scoped._prepare(window, compiled[label])
                prepared.append(out)
                moved += int(points)
            record["train_moved"][label] = moved
            try:
                designs[label] = scoped._design(prepared, view)
                record["fittable"][label] = True
            except RuntimeError as exc:
                record["fittable"][label] = False
                record.setdefault("not_fittable_because", str(exc))
        # this series' served instances, on every face of the block
        mine: list[dict[str, Any]] = []
        for row in faces:
            origin = int(row["origin"])
            config = forecast_p4._config(origin)
            reads.account(uid, origin + HORIZON)
            window = raw[origin - CONTEXT:origin]
            raw_ctx = forecast_runtime._linear_integrity(window)
            truth = raw[origin:origin + HORIZON]
            scale = float(seasonal_scale(raw[:origin], np.isfinite(raw[:origin]),
                                         period=int(config["period"]), min_pairs=32))
            item: dict[str, Any] = {
                "block": block, "position": int(row["position"]), "face": str(row["face"]),
                "origin": origin, "uid": uid, "truth": truth, "scale": scale,
                "raw_ctx": raw_ctx, "raw_ctx_evaluable": r4d._evaluable(raw_ctx, view),
            }
            for label in PROGRAMS:
                served, moved, _trace = scoped._prepare(window, compiled[label])
                item["ctx_%s" % label] = served
                item["moved_%s" % label] = int(moved)
                item["unmodified_%s" % label] = bool(np.array_equal(served, raw_ctx, equal_nan=True))
                item["ctx_evaluable_%s" % label] = r4d._evaluable(served, view)
            mine.append(item)
        # fit 1: own raw model
        if record["fittable"].get("raw"):
            idx = [i for i, it in enumerate(mine) if it["raw_ctx_evaluable"]]
            if idx:
                fits.charge("%s|raw" % uid)
                pred = scoped._serve([mine[i]["raw_ctx"] for i in idx], view, *designs["raw"])
                for slot, i in enumerate(idx):
                    mine[i]["L_rr"] = r4d._loss(mine[i]["truth"], pred[slot], mine[i]["scale"])
        # fits 2 and 3: own program models, raw and prepared contexts in one call
        for label in PROGRAMS:
            if not record["fittable"].get(label):
                continue
            pr_idx = [i for i, it in enumerate(mine) if it["raw_ctx_evaluable"]]
            pp_idx = [i for i, it in enumerate(mine) if it["ctx_evaluable_%s" % label]]
            batch = ([mine[i]["raw_ctx"] for i in pr_idx]
                     + [mine[i]["ctx_%s" % label] for i in pp_idx])
            if not batch:
                continue
            fits.charge("%s|%s" % (uid, label))
            pred = scoped._serve(batch, view, *designs[label])
            for slot, i in enumerate(pr_idx):
                mine[i]["L_pr_%s" % label] = r4d._loss(mine[i]["truth"], pred[slot], mine[i]["scale"])
            off = len(pr_idx)
            for slot, i in enumerate(pp_idx):
                mine[i]["L_pp_%s" % label] = r4d._loss(mine[i]["truth"], pred[off + slot], mine[i]["scale"])
        for item in mine:
            for key in ("truth", "raw_ctx", "ctx_%s" % ANC, "ctx_%s" % W2):
                item.pop(key, None)
        instances.extend(mine)
        per_series.append(record)
    return {"block": block, "instances": instances, "per_series": per_series,
            "eval_uids": eval_uids, "anchors": anchors,
            "faces": [{"position": r["position"], "face": r["face"], "origin": r["origin"]}
                      for r in faces]}


# ---------------------------------------------------------------------------
# 2. rows: per-channel cells next to the pooled cells of R4D-A
# ---------------------------------------------------------------------------


def build_rows(blocks: Sequence[Mapping[str, Any]], pooled: Mapping[str, Any]
               ) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in blocks:
        for item in payload["instances"]:
            for label in PROGRAMS:
                key = "%s|%d|%s" % (label, item["position"], item["face"])
                entry = pooled[key]
                index = [str(u) for u in entry["eval_uids"]].index(item["uid"])
                l_rr, l_pr, l_pp = (item.get("L_rr"), item.get("L_pr_%s" % label),
                                    item.get("L_pp_%s" % label))
                ok = None not in (l_rr, l_pr, l_pp)
                row = {
                    "program": label, "block": item["block"], "position": item["position"],
                    "face": item["face"], "origin": item["origin"], "uid": item["uid"],
                    "status": "OK" if ok else "NOT_EVALUABLE",
                    "L_rr": l_rr, "L_pr": l_pr, "L_pp": l_pp,
                    "serving_moved_points": item["moved_%s" % label],
                    "serving_unmodified": item["unmodified_%s" % label],
                    "pooled_L_rr": float(entry["L_rr"][index]),
                    "pooled_L_pr": float(entry["L_pr"][index]),
                    "pooled_L_pp": float(entry["L_pp"][index]),
                    "pooled_status": str(entry["status"][index]),
                }
                row["pooled_route"] = row["pooled_L_pr"] - row["pooled_L_rr"]
                row["pooled_ctx"] = row["pooled_L_pp"] - row["pooled_L_pr"]
                row["pooled_total"] = row["pooled_L_pp"] - row["pooled_L_rr"]
                if ok:
                    row["route"] = float(l_pr - l_rr)
                    row["ctx"] = float(l_pp - l_pr)
                    row["total"] = float(l_pp - l_rr)
                    row["severe_harm"] = bool(row["total"] > SEVERE)
                else:
                    row["route"] = row["ctx"] = row["total"] = None
                    row["severe_harm"] = None
                rows.append(row)
    return rows


def pooled_rows(pooled: Mapping[str, Any]) -> list[dict[str, Any]]:
    """R4D-A's cells in the row shape the shared readouts read."""
    rows = []
    for entry in pooled.values():
        if entry["program"] not in PROGRAMS:
            continue
        for i, uid in enumerate(entry["eval_uids"]):
            rows.append({
                "program": entry["program"], "block": entry["block"],
                "position": int(entry["position"]), "face": entry["face"],
                "origin": int(entry["origin"]), "uid": str(uid),
                "status": str(entry["status"][i]),
                "route": float(entry["route"][i]), "ctx": float(entry["ctx"][i]),
                "total": float(entry["total"][i]),
                "severe_harm": bool(float(entry["total"][i]) > SEVERE),
                "serving_unmodified": bool(entry["serving_unmodified"][i]),
            })
    return rows


# ---------------------------------------------------------------------------
# 3. readouts
# ---------------------------------------------------------------------------


def baseline_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """per-channel Static against pooled Static, series by series (request 4.5)."""
    out: dict[str, Any] = {}
    seen: set[tuple[int, str, str]] = set()
    base = []
    for r in rows:
        key = (r["position"], r["face"], r["uid"])
        if key in seen or r["status"] != "OK":
            continue
        seen.add(key)
        base.append(r)
    for scope, keep in (("support_face", lambda r: r["face"] == "support_face"),
                        ("delayed_face", lambda r: r["face"] == "delayed_face"),
                        ("both_faces", lambda r: True)):
        sub = [r for r in base if keep(r)]
        pc = np.array([r["L_rr"] for r in sub]); po = np.array([r["pooled_L_rr"] for r in sub])
        d = pc - po
        out[scope] = {
            "n": len(sub), "pc_static_mean": float(pc.mean()), "pooled_static_mean": float(po.mean()),
            "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
            "share_pc_worse": float(np.mean(d > 0)), "delta_p10": float(np.quantile(d, 0.1)),
            "delta_p90": float(np.quantile(d, 0.9)), "spearman_pc_vs_pooled": r4d.spearman(pc, po),
        }
    both = out["both_faces"]
    both["consumer_choice_in_scope"] = bool(abs(both["delta_median"]) <= 0.10)
    both["note"] = ("|median delta| <= 0.10 keeps 'which Consumer' inside the readable range; "
                    "otherwise only the conditioning question is read")
    return out


def _anova_share(values: np.ndarray, groups: np.ndarray) -> tuple[float | None, float | None]:
    if values.size < 3:
        return None, None
    grand = values.mean()
    sst = float(((values - grand) ** 2).sum())
    if sst <= 0:
        return None, None
    ssb = 0.0
    levels = 0
    for g in np.unique(groups):
        m = groups == g
        levels += 1
        ssb += m.sum() * float((values[m].mean() - grand) ** 2)
    return ssb / sst, (levels - 1) / (values.size - 1)


def persistence(rows: Sequence[Mapping[str, Any]], label: str, channel: str = "total"
                ) -> dict[str, Any]:
    """R4A-b's readout on one program: cross-face rank persistence and variance shares."""
    ok = [r for r in rows if r["program"] == label and r["status"] == "OK"
          and r.get(channel) is not None]
    by_pos: dict[int, dict[str, dict[str, float]]] = {}
    for r in ok:
        by_pos.setdefault(r["position"], {}).setdefault(r["face"], {})[r["uid"]] = float(r[channel])
    rhos = []
    for pos, faces in sorted(by_pos.items()):
        if "support_face" not in faces or "delayed_face" not in faces:
            continue
        common = sorted(set(faces["support_face"]) & set(faces["delayed_face"]))
        if len(common) < 3:
            continue
        rho = r4d.spearman(np.array([faces["support_face"][u] for u in common]),
                           np.array([faces["delayed_face"][u] for u in common]))
        if rho is not None:
            rhos.append(rho)
    out: dict[str, Any] = {
        "channel": channel, "positions_paired": len(rhos),
        "cross_face_spearman_median": float(np.median(rhos)) if rhos else None,
        "cross_face_spearman_q25": float(np.quantile(rhos, 0.25)) if rhos else None,
        "cross_face_spearman_q75": float(np.quantile(rhos, 0.75)) if rhos else None,
        "per_face": {},
    }
    for face in FACES:
        sub = [r for r in ok if r["face"] == face]
        v = np.array([r[channel] for r in sub]); pos = np.array([r["position"] for r in sub])
        pos_share, pos_null = _anova_share(v, pos)
        uid_shares, weights, nulls = [], [], []
        for block in sorted({r["block"] for r in sub}):
            bs = [r for r in sub if r["block"] == block]
            bv = np.array([r[channel] for r in bs]); bu = np.array([r["uid"] for r in bs])
            share, null = _anova_share(bv, bu)
            if share is not None:
                uid_shares.append(share); weights.append(len(bs)); nulls.append(null)
        w = np.array(weights, dtype=float)
        uid_share = float(np.average(uid_shares, weights=w)) if uid_shares else None
        uid_null = float(np.average(nulls, weights=w)) if nulls else None
        med = out["cross_face_spearman_median"]
        if uid_share is None or pos_share is None or med is None:
            verdict = "NOT_EVALUABLE"
        elif uid_share >= SERIES_UID_SHARE and med >= SERIES_SPEARMAN:
            verdict = "SERIES_LEVEL_CONDITION"
        elif pos_share >= COHORT_POSITION_SHARE and uid_share < SERIES_UID_SHARE:
            verdict = "COHORT_LEVEL_CONDITION"
        elif med < NOISE_SPEARMAN and uid_share < SERIES_UID_SHARE and pos_share < COHORT_POSITION_SHARE:
            verdict = "MOSTLY_NOISE"
        else:
            verdict = "MIXED"
        out["per_face"][face] = {
            "n": len(sub), "position_share": pos_share, "position_null": pos_null,
            "uid_share_weighted": uid_share, "uid_null_weighted": uid_null,
            "residual_share": (1.0 - pos_share - uid_share) if None not in (pos_share, uid_share) else None,
            "verdict": verdict,
        }
    return out


def harm_shape(rows_pc: Sequence[Mapping[str, Any]], rows_pooled: Sequence[Mapping[str, Any]]
               ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label in PROGRAMS:
        for face in FACES:
            a = np.array([r["total"] for r in rows_pc if r["program"] == label and r["face"] == face
                          and r["status"] == "OK"])
            b = np.array([r["total"] for r in rows_pooled if r["program"] == label and r["face"] == face
                          and r["status"] == "OK"])
            cell = {
                "pc": {"n": int(a.size), "aggregate_gain": float(-a.mean()), "harmed_fraction": float(np.mean(a > 0)),
                       "severe_fraction": float(np.mean(a > SEVERE)), "max_single_series_harm": float(a.max())},
                "pooled": {"n": int(b.size), "aggregate_gain": float(-b.mean()), "harmed_fraction": float(np.mean(b > 0)),
                           "severe_fraction": float(np.mean(b > SEVERE)), "max_single_series_harm": float(b.max())},
            }
            d_msh = cell["pc"]["max_single_series_harm"] - cell["pooled"]["max_single_series_harm"]
            d_hf = cell["pc"]["harmed_fraction"] - cell["pooled"]["harmed_fraction"]
            d_gain = cell["pc"]["aggregate_gain"] - cell["pooled"]["aggregate_gain"]
            cell["delta"] = {"max_single_series_harm": d_msh, "harmed_fraction": d_hf, "aggregate_gain": d_gain}
            cell["shape_verdict"] = ("CONSUMER_SHAPES_HARM" if (abs(d_msh) >= SHAPE_DELTA or abs(d_hf) >= SHAPE_DELTA)
                                     else "NO_SHAPE_CHANGE")
            cell["gain_verdict"] = "GAIN_RETAINED" if d_gain >= -SHAPE_DELTA else "GAIN_LOST"
            out["%s|%s" % (label, face)] = cell
    return out


def feature_cards(entries: Mapping[str, Any], variant: Mapping[str, np.ndarray],
                  reader: r4a.Reader) -> dict[tuple[int, str], dict[str, float]]:
    """R4E's card construction, unchanged (21 R4A quantities + 6 R4C quantities)."""
    cells: dict[tuple[int, str], tuple[int, list[str]]] = {}
    for entry in entries.values():
        if str(entry["program"]) not in PROGRAMS:
            continue
        cells.setdefault((int(entry["position"]), str(entry["face"])),
                         (int(entry["origin"]), [str(u) for u in entry["eval_uids"]]))
    cards: dict[tuple[int, str], dict[str, float]] = {}
    for _key, (origin, uids) in sorted(cells.items()):
        matrix, names = targeting.series_features(variant, uids, origin)
        for uid in uids:
            reader.account_external(uid, origin)
        for index, uid in enumerate(uids):
            if (origin, uid) in cards:
                continue
            context = reader.window(uid, origin - CONTEXT, origin)
            card = {name: float(matrix[index][column]) for column, name in enumerate(names)
                    if name in r4a.OBSERVABLE_NUMERIC}
            card.update({k: v for k, v in r4a.mechanism_features(context).items()
                         if k in r4a.VISIBLE_FEATURES})
            serving, _ok = r4c.serving_observables(context)
            card.update(serving)
            cards[(origin, uid)] = card
    return cards


def _ok_only(entries: Mapping[str, Any]) -> dict[str, Any]:
    """Drop NOT_EVALUABLE instances before handing a store to R4E's readers."""
    out: dict[str, Any] = {}
    for key, entry in entries.items():
        keep = [i for i, s in enumerate(entry["status"]) if s == "OK"]
        if not keep:
            continue
        copy = dict(entry)
        for field, value in entry.items():
            if isinstance(value, list) and len(value) == len(entry["status"]):
                copy[field] = [value[i] for i in keep]
        out[key] = copy
    return out


def identifiability_and_selection(entries: Mapping[str, Any], cards: Mapping[tuple[int, str], dict]
                                  ) -> dict[str, Any]:
    """R4E's reading 1 and reading 2 on one three-cell store."""
    rows = r4e.build_rows(_ok_only(entries), cards)
    table: dict[str, Any] = {}
    for program in PROGRAMS:
        for channel in CHANNELS:
            base = [r for r in rows if r["program"] == program]
            recast = r4e.channel_rows(base, channel)
            for group in ("support_face", "delayed_face", "both_faces"):
                subset = [r for r in recast if r["face"] == group] if group in FACES else list(recast)
                present = sorted({r["block"] for r in subset})
                cell = r4e.identifiability(subset, present)
                table["%s|%s|%s" % (program, channel, group)] = {
                    "n_rows": cell.get("n_rows"), "best_feature": cell.get("best_feature"),
                    "lodo_mean": (cell.get("best_reading") or {}).get("lodo_mean_auc_oriented"),
                    "lodo_min": (cell.get("best_reading") or {}).get("lodo_min_auc_oriented"),
                    "verdict": cell.get("verdict"),
                    "full_support_informative": cell.get("features_clearing_every_fold_with_full_support"),
                }
    selection: dict[str, Any] = {}
    for program in PROGRAMS:
        base = [r for r in rows if r["program"] == program]
        for group in ("support_face", "delayed_face", "both_faces"):
            subset = [r for r in base if r["face"] == group] if group in FACES else base
            present = sorted({r["block"] for r in subset})
            sel = r4e.selection_table(subset, present)
            best = sel.get("best_reading") or {}
            selection["%s|%s" % (program, group)] = {
                "n_rows": sel.get("n_rows"),
                "verdict": sel.get("verdict"),
                "features_with_SELECTION_BEATS_FIXED": sel.get("features_with_SELECTION_BEATS_FIXED"),
                "best_feature_by_binding_margin": sel.get("best_feature_by_binding_margin"),
                "best_reading_summary": {k: v for k, v in best.items()
                                         if not isinstance(v, (dict, list))},
            }
    return {"identifiability": table, "selection": selection}


# ---------------------------------------------------------------------------
# 4. run
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    pooled_payload = json.loads(POOLED_STORE.read_text(encoding="utf-8"))
    pooled = pooled_payload["entries"]
    forward = contract.ordering("forward")
    variant = preflight.load_variant()
    nan_count = int(sum(int(np.isnan(s).sum()) for s in variant.values()))
    if nan_count <= 0:
        raise AssertionError("loaded variant has no NaN; wrong data identity")
    view = views.IdentityView()
    fits = r4d.FitLedger(cap=FIT_HARD_CAP)
    reads = r4d.ReadLedger()

    # the store is keyed like m_r0k's; reuse A's grouping on A's own entries
    by_block = r4d._block_faces(pooled, forward)
    n_series = sum(len(hec1.block_uids(faces[0]["span"])[:20]) for faces in by_block.values())
    if 3 * n_series > FIT_HARD_CAP:
        raise r4d.FitCeiling("%d series x 3 = %d fits > cap %d" % (n_series, 3 * n_series, FIT_HARD_CAP))

    blocks = [run_block(block, faces, view, fits, reads) for block, faces in sorted(by_block.items())]
    rows = build_rows(blocks, pooled)
    rows_pool = pooled_rows(pooled)

    not_fittable = [{"uid": s["uid"], "block": s["block"], "fittable": s["fittable"],
                     "why": s.get("not_fittable_because")}
                    for b in blocks for s in b["per_series"] if not all(s["fittable"].values())]
    ok_rows = [r for r in rows if r["status"] == "OK"]

    generated = datetime.now().astimezone().isoformat()
    payload: dict[str, Any] = {
        "task": "R4D_B_PERCHANNEL_THREE_CELL",
        "generated_at": generated,
        "evidence_class": "MECHANISM / INSTRUMENT; development only; not a capability or generalisation claim",
        "boundary": {
            "physical_fits": fits.n, "physical_fits_hard_cap": FIT_HARD_CAP, "physical_fits_target": FIT_TARGET,
            "physical_fits_by_model_count": len(fits.by_model), "fit_counting_rule": "one _serve call = one physical Ridge fit",
            "llm_calls": 0, "held_out_reads": reads.held_out_reads, "held_out_frontier": r4a.MAX_ALLOWED_INDEX,
            "max_time_index_read": reads.max_index_read, "raw_series_window_reads": reads.reads,
            "existing_files_edited": 0, "pooled_store_written": False, "m_r0k_store_written": False,
            "new_prediction_store": str(PC_STORE.relative_to(PROJECT_ROOT)), "new_sha_or_hash": 0,
            "git_commits": 0, "sub_agents_spawned": 0,
        },
        "provenance": {
            "consumer": "per_channel: each served series fitted on its own anchored training windows",
            "pooled_reference": str(POOLED_STORE.relative_to(PROJECT_ROOT)),
            "data_version": preflight.DATA_VERSION, "nan_count": nan_count,
            "programs": {label: [list(s) for s in r4d.PROGRAM_STEPS[label]] for label in PROGRAMS},
            "geometry": {"context": CONTEXT, "horizon": HORIZON,
                         "own_training_windows": "raw_i[anchor-192:anchor+48], anchor+48 <= origin, invariance across the block's origins asserted per series"},
            "per_block": {b["block"]: {"eval_uids": b["eval_uids"], "anchors": b["anchors"],
                                       "windows_per_series": sorted({s["n_windows"] for s in b["per_series"]}),
                                       "train_moved_points": {label: int(sum(s["train_moved"].get(label, 0) for s in b["per_series"])) for label in PROGRAMS}}
                          for b in blocks},
            "rows": len(rows), "rows_ok": len(ok_rows), "not_fittable_series": not_fittable,
            "not_evaluable_rows": len(rows) - len(ok_rows),
        },
        "definitions": {
            "L_rr": "own raw model / raw context (per-channel Static)",
            "L_pr": "own program model / raw context",
            "L_pp": "own program model / program-prepared context",
            "route": "L_pr - L_rr: this series' own model refitted on its own prepared rows (meaning differs from pooled)",
            "ctx": "L_pp - L_pr: this series' own serving context prepared",
            "total": "L_pp - L_rr = route + ctx (loss orientation; > 0.30 severe)",
        },
        "readout_1_baseline_quality": baseline_quality(rows),
        "readout_2_decomposition": {
            "per_channel": r4d.decomposition_readout(rows, reliable=True),
            "pooled_R4D_A": r4d.decomposition_readout(rows_pool, reliable=True),
        },
        "readout_2b_unmodified": {
            "per_channel": r4d.unmodified_readout(rows),
            "pooled_R4D_A": r4d.unmodified_readout(rows_pool),
        },
        "readout_3_persistence": {
            "per_channel": {label: persistence(rows, label) for label in PROGRAMS},
            "pooled_R4D_A": {label: persistence(rows_pool, label) for label in PROGRAMS},
            "per_channel_route": {label: persistence(rows, label, "route") for label in PROGRAMS},
            "per_channel_ctx": {label: persistence(rows, label, "ctx") for label in PROGRAMS},
        },
        "readout_5_harm_shape": harm_shape(rows, rows_pool),
    }
    store = r4d.three_cell_store(
        [{**r, "store_raw_per_view": r["pooled_L_rr"], "store_program_per_view": r["pooled_L_pp"]}
         for r in rows], payload)
    store["task"] = "R4D_B_PERCHANNEL_THREE_CELL_LOSSES"
    store["consumer"] = "per_channel"
    store["note"] = ("per-channel three-cell losses; store_raw_per_view / store_program_per_view carry the "
                     "POOLED (R4D-A) L_rr / L_pp of the same instance for pairing, not a reconciliation target")
    for entry in store["entries"].values():
        entry["block"] = entry.get("block")
    PC_STORE.write_text(json.dumps(r4d._round(store), ensure_ascii=False), encoding="utf-8")

    # readout 4: R4E's identifiability + selection on both stores (0 fit)
    reader = r4a.Reader(variant)
    cards = feature_cards(store["entries"], variant, reader)
    payload["readout_4_identifiability_selection"] = {
        "per_channel": identifiability_and_selection(store["entries"], cards),
        "pooled_R4D_A": identifiability_and_selection(pooled, cards),
    }
    payload["boundary"]["feature_reads_max_time_index"] = reader.max_index_read
    return r4d._round(payload)


def _md(p: Mapping[str, Any]) -> str:
    b = p["boundary"]
    lines = ["# R4D-B per-channel three-cell -- machine reading", "",
             "Evidence class: %s" % p["evidence_class"], "",
             "Boundary: physical_fits=%s / cap %s (target %s), llm=0, held_out_reads=%s, max_time_index_read=%s (frontier %s)."
             % (b["physical_fits"], b["physical_fits_hard_cap"], b["physical_fits_target"], b["held_out_reads"],
                b["max_time_index_read"], b["held_out_frontier"]),
             "Rows: %d (%d OK); not-fittable series: %d; not-evaluable rows: %d."
             % (p["provenance"]["rows"], p["provenance"]["rows_ok"], len(p["provenance"]["not_fittable_series"]),
                p["provenance"]["not_evaluable_rows"]), ""]
    lines += ["## 1. baseline quality (per-channel Static vs pooled Static)", "",
              "| scope | n | pc Static mean | pooled Static mean | delta mean | delta median | share pc worse | p10 | p90 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scope, v in p["readout_1_baseline_quality"].items():
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            scope, v["n"], v["pc_static_mean"], v["pooled_static_mean"], v["delta_mean"], v["delta_median"],
            v["share_pc_worse"], v["delta_p10"], v["delta_p90"]))
    lines += ["", "consumer_choice_in_scope: %s" % p["readout_1_baseline_quality"]["both_faces"]["consumer_choice_in_scope"], ""]
    lines += ["## 2. three-cell decomposition (pc vs pooled)", "",
              "| consumer | program | scope | n | route share | severe n | severe route-worst share | rho(route,total) | rho(ctx,total) | verdict |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for consumer in ("per_channel", "pooled_R4D_A"):
        for key, v in p["readout_2_decomposition"][consumer].items():
            if v.get("n", 0) == 0:
                continue
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                consumer, v["program"], v["scope"], v["n"], v["route_algebraic_share"], v["n_severe_harm"],
                v["severe_route_most_negative_share"], v["spearman_route_total"], v["spearman_ctx_total"], v["verdict"]))
    lines += ["", "## 3. persistence (total channel; R4A-b vocabulary)", "",
              "| consumer | program | cross-face Spearman median | face | position share | uid share (null) | residual | verdict |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |"]
    for consumer in ("per_channel", "pooled_R4D_A"):
        for label, v in p["readout_3_persistence"][consumer].items():
            for face, f in v["per_face"].items():
                lines.append("| %s | %s | %s | %s | %s | %s (%s) | %s | %s |" % (
                    consumer, label, v["cross_face_spearman_median"], face, f["position_share"],
                    f["uid_share_weighted"], f["uid_null_weighted"], f["residual_share"], f["verdict"]))
    lines += ["", "## 4. identifiability (best feature per program x channel x face group) and selection", "",
              "| consumer | cell | n | best feature | lodo mean | lodo min | verdict | full-support INFORMATIVE |",
              "| --- | --- | ---: | --- | ---: | ---: | --- | --- |"]
    for consumer in ("per_channel", "pooled_R4D_A"):
        for key, v in p["readout_4_identifiability_selection"][consumer]["identifiability"].items():
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                consumer, key, v["n_rows"], v["best_feature"], v["lodo_mean"], v["lodo_min"], v["verdict"],
                v["full_support_informative"]))
    lines += ["", "| consumer | program|group | selection verdict | detail |", "| --- | --- | --- | --- |"]
    for consumer in ("per_channel", "pooled_R4D_A"):
        for key, v in p["readout_4_identifiability_selection"][consumer]["selection"].items():
            lines.append("| %s | %s | %s | %s |" % (consumer, key, v.get("verdict"),
                                                 json.dumps({k: vv for k, vv in v.items() if k != "verdict"}, ensure_ascii=False)[:300]))
    lines += ["", "## 5. harm shape (pc vs pooled)", "",
              "| program|face | pc gain | pooled gain | pc hf | pooled hf | pc msh | pooled msh | shape | gain |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for key, v in p["readout_5_harm_shape"].items():
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            key, v["pc"]["aggregate_gain"], v["pooled"]["aggregate_gain"], v["pc"]["harmed_fraction"],
            v["pooled"]["harmed_fraction"], v["pc"]["max_single_series_harm"], v["pooled"]["max_single_series_harm"],
            v["shape_verdict"], v["gain_verdict"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = run()
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    print(json.dumps({"boundary": payload["boundary"],
                      "baseline": payload["readout_1_baseline_quality"]["both_faces"],
                      "decomposition_pc": {k: (v.get("route_algebraic_share"), v.get("verdict"))
                                           for k, v in payload["readout_2_decomposition"]["per_channel"].items()},
                      "persistence_pc": {k: v["cross_face_spearman_median"]
                                         for k, v in payload["readout_3_persistence"]["per_channel"].items()}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
