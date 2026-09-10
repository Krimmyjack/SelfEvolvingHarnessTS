"""R4F: an EXPLORATORY relationship atlas over everything already scored.

This package is different in kind from R4A-R4E.  Those were confirmatory:
thresholds frozen first, verdicts pre-written, one question per package.  This
one is hypothesis-generating.  It takes every per-series loss the project has
already paid for -- the pooled and per-channel three-cell stores and the m_r0k
store -- crosses them with the 27 deployment-visible quantities, and reports
effect sizes, directions and cross-block stability *without* gates or verdicts.
It also draws the actual series behind the largest effects, because every
conclusion so far has come from aggregates and nobody has looked at a curve.

Nothing here is evidence.  Anything that is to enter the method must be frozen
and re-tested on data this package did not read.

Run: ``python -m evaluation.main_protocol_p4.explore_r4f_relationship_atlas``
0 Consumer fits, 0 LLM calls, 0 held-out reads.
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
from evaluation.main_protocol_p4 import audit_r4e_channel_identifiability as r4e
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_hec1 as hec1
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import run_r4d_a_action_decomposition as r4d
from evaluation.main_protocol_p4 import run_r4d_b_perchannel_three_cell as r4db
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POOLED_STORE = PROJECT_ROOT / "_scratch/r4d_a_three_cell_store.json"
PC_STORE = PROJECT_ROOT / "_scratch/r4d_b_perchannel_three_cell_store.json"
M_R0K_STORE = PROJECT_ROOT / "_scratch/m_r0k_prediction_store.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/r4f_relationship_atlas.json"
OUT_MD = PROJECT_ROOT / "artifacts/main_protocol/r4f_relationship_atlas.md"
GALLERY = PROJECT_ROOT / "artifacts/main_protocol/r4f_gallery"

CONTEXT = r4d.CONTEXT
HORIZON = r4d.HORIZON
PROGRAMS = r4d.PROGRAMS
ANC, W2 = r4d.ANC, r4d.W2
CHANNELS = ("route", "ctx", "total")
FEATURES = tuple(r4e.FEATURES)
CONSUMERS = ("pooled", "per_channel")
SEVERE = r4d.SEVERE_HARM
CTX_EPS = r4e.CTX_ACTIVE_EPS
PREVIOUSLY_HEADLINED = {"local_robust_z_peak", "spike_peak_over_tail_sd",
                        "srv_period_filled_points", "missing_fraction"}
M_R0K_PROGRAMS = ("ANCESTOR", "W1_hampel_filter", "W2_pmc_then_outlier_mad",
                  "CAND_outlier_iqr({})>hampel_filter({})",
                  "CAND_outlier_iqr({})>winsorize({})")


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    return r4d.spearman(np.asarray(x, dtype=float), np.asarray(y, dtype=float))


# ---------------------------------------------------------------------------
# A. relationship atlas
# ---------------------------------------------------------------------------


def _channel_subset(rows: Sequence[Mapping[str, Any]], program: str, channel: str
                    ) -> list[Mapping[str, Any]]:
    out = [r for r in rows if r["program"] == program and r["status"] == "OK"]
    if channel == "ctx":
        out = [r for r in out if abs(float(r["ctx"])) > CTX_EPS]
    return out


def atlas(rows_by_consumer: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for consumer, rows in rows_by_consumer.items():
        blocks = sorted({r["block"] for r in rows})
        for program in PROGRAMS:
            for channel in CHANNELS:
                sub = _channel_subset(rows, program, channel)
                if len(sub) < 20:
                    continue
                y = np.array([float(r[channel]) for r in sub])
                helped = (y < 0).astype(int)
                severe = (y > SEVERE).astype(int)
                blk = np.array([r["block"] for r in sub])
                for feature in FEATURES:
                    x = np.array([float(r["features"].get(feature, np.nan)) for r in sub])
                    rho = spearman(x, y)
                    per_block = {}
                    for b in blocks:
                        m = blk == b
                        per_block[b] = spearman(x[m], y[m]) if m.sum() >= 8 else None
                    defined = [v for v in per_block.values() if v is not None]
                    same = (sum(1 for v in defined if rho is not None and np.sign(v) == np.sign(rho))
                            if rho is not None else 0)
                    med_abs = float(np.median([abs(v) for v in defined])) if defined else None
                    score = (med_abs or 0.0) * (same / max(1, len(blocks)))
                    table.append({
                        "consumer": consumer, "program": program, "channel": channel,
                        "feature": feature, "n": int(np.isfinite(x).sum()),
                        "spearman_pooled": rho, "spearman_by_block": per_block,
                        "blocks_same_sign": same, "blocks_defined": len(defined),
                        "median_abs_spearman_blocks": med_abs, "stability_score": score,
                        "auc_helped_raw": r4a.auc(x, helped), "auc_severe_raw": r4a.auc(x, severe),
                        "n_helped": int(helped.sum()), "n_severe": int(severe.sum()),
                        "previously_headlined": feature in PREVIOUSLY_HEADLINED,
                    })
    table.sort(key=lambda t: (-(t["stability_score"] or 0), -(abs(t["spearman_pooled"] or 0))))
    return table


# ---------------------------------------------------------------------------
# B. tree probe (what gets selected; never a router)
# ---------------------------------------------------------------------------


def tree_probe(rows_by_consumer: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    from sklearn.tree import DecisionTreeRegressor
    out: dict[str, Any] = {}
    for consumer, rows in rows_by_consumer.items():
        blocks = sorted({r["block"] for r in rows})
        for program in PROGRAMS:
            for channel in ("total", "ctx"):
                sub = _channel_subset(rows, program, channel)
                if len(sub) < 80:
                    continue
                X = np.array([[float(r["features"].get(f, np.nan)) for f in FEATURES] for r in sub])
                X = np.where(np.isfinite(X), X, np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0))
                X = np.where(np.isfinite(X), X, 0.0)
                y = np.array([float(r[channel]) for r in sub])
                blk = np.array([r["block"] for r in sub])

                def describe(tree: Any) -> list[dict[str, Any]]:
                    t = tree.tree_
                    nodes = []
                    for i in range(t.node_count):
                        if t.children_left[i] == -1:
                            continue
                        nodes.append({"node": int(i), "feature": FEATURES[int(t.feature[i])],
                                      "threshold": float(t.threshold[i]), "n": int(t.n_node_samples[i]),
                                      "mean": float(t.value[i][0][0]),
                                      "left_n": int(t.n_node_samples[t.children_left[i]]),
                                      "left_mean": float(t.value[t.children_left[i]][0][0]),
                                      "right_n": int(t.n_node_samples[t.children_right[i]]),
                                      "right_mean": float(t.value[t.children_right[i]][0][0])})
                    return nodes

                full = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0).fit(X, y)
                folds = {}
                used: dict[str, int] = {}
                for b in blocks:
                    m = blk != b
                    if m.sum() < 80:
                        continue
                    tr = DecisionTreeRegressor(max_depth=3, min_samples_leaf=40, random_state=0).fit(X[m], y[m])
                    nodes = describe(tr)
                    folds[b] = {"root_feature": nodes[0]["feature"] if nodes else None,
                                "root_threshold": nodes[0]["threshold"] if nodes else None,
                                "features_used": sorted({n["feature"] for n in nodes})}
                    for f in {n["feature"] for n in nodes}:
                        used[f] = used.get(f, 0) + 1
                out["%s|%s|%s" % (consumer, program, channel)] = {
                    "n": len(sub), "full_tree": describe(full),
                    "lodo_root_features": {b: f["root_feature"] for b, f in folds.items()},
                    "feature_use_count_across_folds": dict(sorted(used.items(), key=lambda kv: -kv[1])),
                }
    return out


# ---------------------------------------------------------------------------
# C. relationships between programs and between Consumers
# ---------------------------------------------------------------------------


def cross_relations(rows_by_consumer: Mapping[str, Sequence[Mapping[str, Any]]],
                    m_r0k: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key = lambda r: (r["position"], r["face"], r["uid"])
    for consumer, rows in rows_by_consumer.items():
        ok = [r for r in rows if r["status"] == "OK"]
        a = {key(r): r for r in ok if r["program"] == ANC}
        w = {key(r): r for r in ok if r["program"] == W2}
        common = sorted(set(a) & set(w))
        blocks = sorted({r["block"] for r in ok})
        pb = {}
        for b in blocks:
            ks = [k for k in common if a[k]["block"] == b]
            pb[b] = spearman([a[k]["total"] for k in ks], [w[k]["total"] for k in ks])
        out["%s|total_ANC_vs_total_W2" % consumer] = {
            "n": len(common), "spearman": spearman([a[k]["total"] for k in common], [w[k]["total"] for k in common]),
            "by_block": pb}
        for program in PROGRAMS:
            sub = [r for r in ok if r["program"] == program]
            out["%s|%s|route_vs_ctx" % (consumer, program)] = {
                "n": len(sub), "spearman": spearman([r["route"] for r in sub], [r["ctx"] for r in sub]),
                "spearman_ctx_active_only": spearman([r["route"] for r in sub if abs(r["ctx"]) > CTX_EPS],
                                                     [r["ctx"] for r in sub if abs(r["ctx"]) > CTX_EPS])}
    po = {(r["program"],) + key(r): r for r in rows_by_consumer["pooled"] if r["status"] == "OK"}
    pc = {(r["program"],) + key(r): r for r in rows_by_consumer["per_channel"] if r["status"] == "OK"}
    for program in PROGRAMS:
        ks = sorted(k for k in po if k in pc and k[0] == program)
        out["pooled_vs_per_channel|%s" % program] = {
            "n": len(ks),
            "spearman_total": spearman([po[k]["total"] for k in ks], [pc[k]["total"] for k in ks]),
            "spearman_route": spearman([po[k]["route"] for k in ks], [pc[k]["route"] for k in ks]),
            "spearman_ctx": spearman([po[k]["ctx"] for k in ks], [pc[k]["ctx"] for k in ks]),
            "spearman_static_L_rr": spearman([po[k]["L_rr"] for k in ks], [pc[k]["L_rr"] for k in ks])
            if "L_rr" in next(iter(po.values())) else None,
        }
    # m_r0k: gain per program on the same (position, face, uid)
    gains: dict[str, dict[tuple, float]] = {p: {} for p in M_R0K_PROGRAMS}
    for entry in m_r0k.values():
        p = entry["program"]
        if p not in gains or not entry.get("verifier_passed"):
            continue
        for i, uid in enumerate(entry["eval_uids"]):
            gains[p][(int(entry["position"]), entry["face"], str(uid))] = float(
                entry["raw_per_view"][i] - entry["program_per_view"][i])
    matrix: dict[str, dict[str, Any]] = {}
    for p1 in M_R0K_PROGRAMS:
        matrix[p1] = {}
        for p2 in M_R0K_PROGRAMS:
            ks = sorted(set(gains[p1]) & set(gains[p2]))
            matrix[p1][p2] = {"n": len(ks), "spearman": spearman([gains[p1][k] for k in ks], [gains[p2][k] for k in ks])
                              if len(ks) >= 10 else None}
    out["m_r0k_gain_spearman_matrix"] = matrix
    # who benefits: share of instances helped by both / neither / one (ANC vs W2, pooled)
    a = {key(r): r for r in rows_by_consumer["pooled"] if r["status"] == "OK" and r["program"] == ANC}
    w = {key(r): r for r in rows_by_consumer["pooled"] if r["status"] == "OK" and r["program"] == W2}
    ks = sorted(set(a) & set(w))
    both = sum(1 for k in ks if a[k]["total"] < 0 and w[k]["total"] < 0)
    neither = sum(1 for k in ks if a[k]["total"] >= 0 and w[k]["total"] >= 0)
    out["pooled|helped_overlap_ANC_W2"] = {"n": len(ks), "both_helped": both, "neither": neither,
                                           "only_ANC": sum(1 for k in ks if a[k]["total"] < 0 <= w[k]["total"]),
                                           "only_W2": sum(1 for k in ks if w[k]["total"] < 0 <= a[k]["total"])}
    return out


# ---------------------------------------------------------------------------
# D. gallery: the actual series behind the largest effects
# ---------------------------------------------------------------------------


def _compile_per_block(store_entries: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The production compile path, once per block; no fit is involved."""
    forward = contract.ordering("forward")
    by_block = r4d._block_faces(store_entries, forward)
    compiled: dict[str, dict[str, Any]] = {}
    for block, faces in by_block.items():
        uids = hec1.block_uids(faces[0]["span"])
        cell, _v = baselines._cell(uids)
        origin = int(faces[0]["origin"])
        config = forecast_p4._config(origin)
        at = forecast_p4._cell_at(cell, origin)
        roster = at.roster(hec1.FACE)
        executor = ScopeExecutor(roster, at.values, config,
                                 evaluate_fn=views.forecast_runtime._evaluate,
                                 max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)
        compiled[block] = {label: executor._compiled(r4d.PROGRAM_STEPS[label]) for label in PROGRAMS}
        compiled[block]["_values"] = cell.values
    return compiled


def gallery(rows_by_consumer: Mapping[str, Sequence[Mapping[str, Any]]], compiled: Mapping[str, Any],
            reader: r4a.Reader) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    GALLERY.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"figures": [], "instances": []}

    def draw(items: Sequence[Mapping[str, Any]], title: str, fname: str) -> None:
        n = len(items)
        cols = 3
        rws = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(5.2 * cols, 3.0 * rws), squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, row in enumerate(items):
            ax = axes.flat[k]
            ax.set_visible(True)
            uid, origin, block = row["uid"], int(row["origin"]), row["block"]
            raw = np.asarray(compiled[block]["_values"][uid], dtype=float)
            reader.account_external(uid, origin + HORIZON)
            ctx = raw[origin - CONTEXT:origin]
            truth = raw[origin:origin + HORIZON]
            prepared, moved, _ = scoped._prepare(ctx, compiled[block][row["program"]])
            x_ctx = np.arange(-CONTEXT, 0)
            x_fut = np.arange(0, HORIZON)
            ax.plot(x_ctx, ctx, color="#1f5fa8", lw=1.0, label="raw context")
            if moved:
                ax.plot(x_ctx, prepared, color="#e07b00", lw=0.9, ls="--", label="prepared (%d pts)" % moved)
            nan_idx = np.flatnonzero(~np.isfinite(ctx))
            if nan_idx.size:
                base = np.nanmin(ctx) if np.isfinite(ctx).any() else 0.0
                ax.plot(x_ctx[nan_idx], np.full(nan_idx.size, base), "rx", ms=3, label="NaN (%d)" % nan_idx.size)
            ax.plot(x_fut, truth, color="#777777", lw=1.0, label="truth (EXPOSED)")
            ax.axvline(0, color="k", lw=0.6)
            f = row["features"]
            ax.set_title("%s @%d | tot %.2f rt %.2f ctx %.2f | z %.1f miss %.2f" % (
                uid, origin, row["total"], row["route"], row["ctx"],
                f.get("local_robust_z_peak", float("nan")), f.get("missing_fraction", float("nan"))), fontsize=8)
            ax.tick_params(labelsize=7)
            if k == 0:
                ax.legend(fontsize=6, loc="upper left")
            manifest["instances"].append({"figure": fname, "panel": k, "uid": uid, "origin": origin,
                                          "block": block, "program": row["program"], "total": row["total"],
                                          "route": row["route"], "ctx": row["ctx"], "moved_points": int(moved),
                                          "n_nan_context": int(nan_idx.size),
                                          "z_peak": f.get("local_robust_z_peak"), "missing_fraction": f.get("missing_fraction")})
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(GALLERY / fname, dpi=110)
        plt.close(fig)
        manifest["figures"].append(fname)

    for consumer, rows in rows_by_consumer.items():
        for program in PROGRAMS:
            ok = [r for r in rows if r["status"] == "OK" and r["program"] == program]
            by_total = sorted(ok, key=lambda r: r["total"])
            draw(by_total[-6:][::-1], "%s | %s | WORST 6 by total (loss increase)" % (consumer, program),
                 "%s_%s_total_worst6.png" % (consumer, program.split("_")[0]))
            draw(by_total[:6], "%s | %s | BEST 6 by total (loss decrease)" % (consumer, program),
                 "%s_%s_total_best6.png" % (consumer, program.split("_")[0]))
            active = sorted([r for r in ok if abs(r["ctx"]) > CTX_EPS], key=lambda r: r["ctx"])
            if len(active) >= 6:
                draw(active[-3:][::-1] + active[:3], "%s | %s | ctx WORST 3 (top row) / BEST 3 (bottom)" % (consumer, program),
                     "%s_%s_ctx_worst3_best3.png" % (consumer, program.split("_")[0]))
    return manifest


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    pooled = json.loads(POOLED_STORE.read_text(encoding="utf-8"))["entries"]
    pc = json.loads(PC_STORE.read_text(encoding="utf-8"))["entries"]
    m_r0k = json.loads(M_R0K_STORE.read_text(encoding="utf-8"))
    variant = preflight.load_variant()
    if int(sum(int(np.isnan(s).sum()) for s in variant.values())) <= 0:
        raise AssertionError("wrong data identity (no NaN)")
    reader = r4a.Reader(variant)
    cards = r4db.feature_cards(pooled, variant, reader)
    rows_by_consumer = {"pooled": r4e.build_rows(r4db._ok_only(pooled), cards),
                        "per_channel": r4e.build_rows(r4db._ok_only(pc), cards)}
    # carry L_rr for the Static cross-Consumer correlation
    for consumer, entries in (("pooled", pooled), ("per_channel", pc)):
        lookup = {}
        for e in entries.values():
            for i, uid in enumerate(e["eval_uids"]):
                lookup[(e["program"], int(e["position"]), e["face"], str(uid))] = float(e["L_rr"][i])
        for r in rows_by_consumer[consumer]:
            r["L_rr"] = lookup[(r["program"], r["position"], r["face"], r["uid"])]

    table = atlas(rows_by_consumer)
    probe = tree_probe(rows_by_consumer)
    cross = cross_relations(rows_by_consumer, m_r0k)
    compiled = _compile_per_block(pooled)
    manifest = gallery(rows_by_consumer, compiled, reader)

    payload = {
        "task": "R4F_RELATIONSHIP_ATLAS",
        "generated_at": datetime.now().astimezone().isoformat(),
        "evidence_class": "EXPLORATORY / HYPOTHESIS_GENERATING; nothing here is evidence; anything entering the method must be frozen and re-tested on unread data",
        "boundary": {"consumer_fits": 0, "llm_calls": 0, "held_out_reads": reader.held_out_reads,
                     "max_time_index_read": reader.max_index_read, "held_out_frontier": r4a.MAX_ALLOWED_INDEX,
                     "truth_window_plotted": "[origin, origin+48) -- already exposed by the stores; labelled EXPOSED in every panel",
                     "existing_files_edited": 0, "new_sha_or_hash": 0},
        "inputs": {"pooled_store": str(POOLED_STORE.relative_to(PROJECT_ROOT)), "pc_store": str(PC_STORE.relative_to(PROJECT_ROOT)),
                   "m_r0k_store": str(M_R0K_STORE.relative_to(PROJECT_ROOT)),
                   "rows": {k: len(v) for k, v in rows_by_consumer.items()}, "features": list(FEATURES)},
        "A_atlas_top30": table[:30],
        "A_atlas_all": table,
        "B_tree_probe": probe,
        "C_cross_relations": cross,
        "D_gallery": manifest,
    }
    return r4e._round(payload)


def _md(p: Mapping[str, Any]) -> str:
    b = p["boundary"]
    lines = ["# R4F relationship atlas -- EXPLORATORY machine reading", "",
             "Evidence class: %s" % p["evidence_class"], "",
             "Boundary: fits=0, llm=0, held_out_reads=%s, max_time_index_read=%s (frontier %s)." % (
                 b["held_out_reads"], b["max_time_index_read"], b["held_out_frontier"]),
             "Rows: %s; features: %d." % (p["inputs"]["rows"], len(p["inputs"]["features"])), "",
             "## A. Top-30 relationships (stability score = median |Spearman| over blocks x share of blocks with the pooled sign)", "",
             "| # | consumer | program | channel | feature | n | Spearman pooled | by block | same-sign blocks | median abs | score | AUC helped (raw) | AUC severe (raw) | n severe | headlined before |",
             "| ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for i, t in enumerate(p["A_atlas_top30"], 1):
        bb = ", ".join("%s=%s" % (k.replace("[", "").replace("]", ""), v) for k, v in t["spearman_by_block"].items())
        lines.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s/%s | %s | %s | %s | %s | %s | %s |" % (
            i, t["consumer"], t["program"].split("_")[0], t["channel"], t["feature"], t["n"], t["spearman_pooled"], bb,
            t["blocks_same_sign"], t["blocks_defined"], t["median_abs_spearman_blocks"], t["stability_score"],
            t["auc_helped_raw"], t["auc_severe_raw"], t["n_severe"], "yes" if t["previously_headlined"] else "NEW"))
    lines += ["", "## B. Tree probe (depth<=3, min_leaf 40; what got selected, never a router)", ""]
    for key, v in p["B_tree_probe"].items():
        lines.append("### %s (n=%d)" % (key, v["n"]))
        lines.append("")
        lines.append("| node | feature | threshold | n | mean | left n/mean | right n/mean |")
        lines.append("| ---: | --- | ---: | ---: | ---: | --- | --- |")
        for nd in v["full_tree"]:
            lines.append("| %d | %s | %s | %d | %s | %d / %s | %d / %s |" % (
                nd["node"], nd["feature"], nd["threshold"], nd["n"], nd["mean"], nd["left_n"], nd["left_mean"],
                nd["right_n"], nd["right_mean"]))
        lines.append("")
        lines.append("LODO root features: %s; feature use across folds: %s" % (
            json.dumps(v["lodo_root_features"], ensure_ascii=False), json.dumps(v["feature_use_count_across_folds"], ensure_ascii=False)))
        lines.append("")
    lines += ["## C. Relationships between programs and Consumers", ""]
    for key, v in p["C_cross_relations"].items():
        if key == "m_r0k_gain_spearman_matrix":
            lines.append("### m_r0k per-series gain Spearman matrix")
            lines.append("")
            progs = list(v.keys())
            lines.append("| | " + " | ".join(pp.split("_")[0][:14] for pp in progs) + " |")
            lines.append("| --- |" + " ---: |" * len(progs))
            for p1 in progs:
                lines.append("| %s | " % p1.split("_")[0][:14] + " | ".join(
                    "%s (n=%s)" % (v[p1][p2]["spearman"], v[p1][p2]["n"]) for p2 in progs) + " |")
            lines.append("")
        else:
            lines.append("- **%s**: %s" % (key, json.dumps(v, ensure_ascii=False)))
    lines += ["", "## D. Gallery", "", "Figures in `artifacts/main_protocol/r4f_gallery/`: " + ", ".join(p["D_gallery"]["figures"]), ""]
    lines.append("| figure | panel | uid | origin | block | program | total | route | ctx | moved pts | NaN in ctx | z_peak | missing_fraction |")
    lines.append("| --- | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for it in p["D_gallery"]["instances"]:
        lines.append("| %s | %d | %s | %d | %s | %s | %s | %s | %s | %d | %d | %s | %s |" % (
            it["figure"], it["panel"], it["uid"], it["origin"], it["block"], it["program"].split("_")[0], it["total"],
            it["route"], it["ctx"], it["moved_points"], it["n_nan_context"], it["z_peak"], it["missing_fraction"]))
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = run()
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_MD.write_text(_md(payload), encoding="utf-8")
    print(json.dumps({"boundary": payload["boundary"], "figures": payload["D_gallery"]["figures"],
                      "top10": [(t["consumer"], t["program"].split("_")[0], t["channel"], t["feature"],
                                 t["median_abs_spearman_blocks"], t["blocks_same_sign"], t["spearman_pooled"])
                                for t in payload["A_atlas_top30"][:10]]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
