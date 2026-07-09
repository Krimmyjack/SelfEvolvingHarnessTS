"""run_family0.py — E-3.3 Family 0 剂量扫描（exploratory；A-31a/A-32/A-33，评审第十一/十二轮）。

回答 F0 的单一科学问题：**更强平滑剂量能否在 held-out 上补充当前池的供给缺口**——尤其
witness 所在的 `snrLow|full`（v2.1 池最优 v_median 1.6823，池外 v1-STL 1.5406，差 0.1417）
——且不伤害其它 cell（尤 snrHigh）。

设计（全部预注册）：
  · base_pool = operator_pool_v2.1 的 7 动作（v_none/v_median/v_savgol/v_stl/v_wavelet/
    v_winsor/v_winsor_savgol）；expanded = base + 8 个 F0 剂量动作（family0_actions.F0_DOSAGE_GRID）。
  · 每 cell：真 nested held-out Δ_supply（nested_supply）。**正式判决 CI = full-refit group
    bootstrap（delta_supply_grouped，A-33c）**；同时给单次 nested 的 test-uid CI 作参照。
  · 两池共用同一 outer folds（paired）；点估计 col_mean 仅作诊断（in-sample，不判决）。
  · 聚合：cell 等权 Δ_supply + worst-group（最差 cell 的 CI 下界）+ snrHigh harm 检查。
  · **不开 confirmatory seeds 20–39**（A-32d：F0 是否终池决定分支）。

解释边界（评审强调）：F0 成功只证 degradation/SNR 条件化剂量路由有价值（selection 单位仍
cell=SNR×missing）；Pattern→强度 须留 E-3.2。reporter 同向 + worst-group 是**正式**四门的一部分；
本 exploratory 给 worst-group，reporter 同向留 confirmatory（report_target 口径）。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_family0 [--n-seeds 20] [--n-boot 300] [--out results/E1_1_family0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .evaluators.frozen_probe import FrozenProbe
from .run_main_table import fixed_harness_variants, _VARIANT_SPECS
from .run_variance_decomp import (build_corpus, assign_cells, build_cell_cache, _oof_losses)
from .family0_actions import f0_variants, F0_DOSAGE_GRID
from .nested_supply import delta_supply_grouped, delta_supply, nested_pool_losses, DEFAULT_SEED

RESULTS = Path(__file__).resolve().parent / "results" / "E1_1_family0"
BASE_POOL = list(_VARIANT_SPECS.keys())          # v2.1 的 7 动作（单一真源）
F0_ACTIONS = [a for a, _op, _w in F0_DOSAGE_GRID]
DECISIVE_CELL = "forecast|snrLow|full"           # witness 所在


def _point_col_mean(action_caches, common_uids, actions, kfold=5):
    """诊断用 in-sample col_mean OOF nRMSE（与 v2.1 report 同口径；不判决）。"""
    fold_pt = {u: i % kfold for i, u in enumerate(common_uids)}
    out = {}
    for a in actions:
        losses = _oof_losses(action_caches[a], common_uids, fold_pt, kfold)
        vals = [losses.get(u, np.nan) for u in common_uids]
        out[a] = float(np.nanmean(vals))
    return out


def _save(report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def run(cells, snr_of, base_map, f0_map, n_boot, outer_k, inner_k, seed, out_dir, cell_filter=None):
    fp = FrozenProbe()
    variants_map = {**base_map, **f0_map}                 # 15 动作统一缓存（同 uid 交集）
    expanded = BASE_POOL + F0_ACTIONS
    # 合并已有 report.json（支持增量按 cell 跑；per-cell n_boot 权威，见各 cell 字段）
    prior_cells = {}
    existing = out_dir / "report.json"
    if existing.exists():
        try:
            prior_cells = json.loads(existing.read_text(encoding="utf-8")).get("cells", {})
        except Exception:
            prior_cells = {}
    report = {"config": dict(base_pool=BASE_POOL, f0_grid=[list(g) for g in F0_DOSAGE_GRID],
                             outer_k=outer_k, inner_k=inner_k, n_boot=n_boot, seed=seed,
                             cell_filter=sorted(cell_filter) if cell_filter else None,
                             note="E-3.3 Family 0 剂量扫描 (exploratory); base=operator_pool_v2.1; "
                                  "Δ_supply=nested held-out; 正式 CI=full-refit group bootstrap (A-33c); "
                                  "confirmatory seeds 20-39 未开; per-cell n_boot 见各 cell delta_supply_grouped.n_boot"),
              "cells": dict(prior_cells)}
    todo = [(c, s) for c, s in sorted(cells.items()) if (cell_filter is None or c in cell_filter)]
    for cid, series in todo:
        uids_all = {rs.series_uid for rs in series}
        if len(uids_all) < 6:
            report["cells"][cid] = {"skip": "LOWCONF", "uids": len(uids_all)}
            print(f"[skip] {cid} LOWCONF", flush=True); continue
        print(f"\n[cell] {cid}  caching {len(variants_map)} actions × {len(uids_all)} uids …", flush=True)
        t = time.time()
        action_caches, common_uids, _origin = build_cell_cache(fp, series, variants_map)
        n = len(common_uids)

        pt = _point_col_mean(action_caches, common_uids, expanded)
        base_best = min(BASE_POOL, key=lambda a: pt[a])
        f0_beats = sorted([a for a in F0_ACTIONS if pt[a] < pt[base_best]], key=lambda a: pt[a])

        # 真 nested held-out Δ_supply（正式 CI=grouped，参照=single）
        print(f"       nested Δ_supply (grouped B={n_boot}) …")
        dg = delta_supply_grouped(action_caches, BASE_POOL, expanded, common_uids,
                                  outer_k=outer_k, inner_k=inner_k, seed=seed, n_boot=n_boot)
        ds = delta_supply(action_caches, BASE_POOL, expanded, common_uids,
                          outer_k=outer_k, inner_k=inner_k, seed=seed, n_boot=300)
        _, picks_exp = nested_pool_losses(action_caches, expanded, common_uids, outer_k, inner_k, seed)
        det_picks = [p["selected"] for p in picks_exp]

        report["cells"][cid] = dict(
            uids=n, point_col_mean={a: pt[a] for a in expanded},
            base_best_action=base_best, base_best_point=pt[base_best],
            f0_beats_base_point=f0_beats,
            delta_supply_grouped={k: dg[k] for k in
                ("delta_mean", "ci_lo", "ci_hi", "median", "loss_base", "loss_expanded",
                 "frac_boot_positive", "expanded_pick_freq", "n_boot")},
            delta_supply_single={k: ds[k] for k in ("delta_mean", "ci_lo", "ci_hi")},
            expanded_pick_deterministic=det_picks)
        print(f"       base_best(point)={base_best} {pt[base_best]:.4f}; F0 beats(point): {f0_beats}", flush=True)
        print(f"       Δ_supply grouped={dg['delta_mean']:+.4f} CI[{dg['ci_lo']:+.4f},{dg['ci_hi']:+.4f}] "
              f"pos%={dg['frac_boot_positive']:.2f}  det_picks={det_picks}  [{time.time()-t:.1f}s]", flush=True)
        print(f"       expanded bootstrap pick_freq: {dg['expanded_pick_freq']}", flush=True)
        _save(report, out_dir)                            # 逐 cell checkpoint（被 kill 不丢已完成 cell）

    # ── 聚合 ──
    active = {c: r for c, r in report["cells"].items() if "skip" not in r}
    ce = float(np.mean([r["delta_supply_grouped"]["delta_mean"] for r in active.values()]))
    worst = min(active.items(), key=lambda kv: kv[1]["delta_supply_grouped"]["ci_lo"])
    snrhigh = {c: r for c, r in active.items() if "snrHigh" in c}
    harm_min_ci = min((r["delta_supply_grouped"]["ci_lo"] for r in snrhigh.values()), default=float("nan"))
    report["aggregate"] = dict(
        cell_equal_delta_mean=ce,
        worst_cell=worst[0], worst_cell_delta=worst[1]["delta_supply_grouped"]["delta_mean"],
        worst_cell_ci_lo=worst[1]["delta_supply_grouped"]["ci_lo"],
        snrHigh_harm_min_ci_lo=harm_min_ci,
        decisive_cell=DECISIVE_CELL,
        decisive=active.get(DECISIVE_CELL, {}).get("delta_supply_grouped"))
    return report


def main():
    ap = argparse.ArgumentParser(description="E-3.3 Family 0 剂量扫描 (exploratory)")
    ap.add_argument("--n-seeds", type=int, default=20)     # 与 v2.1 语料一致
    ap.add_argument("--n-boot", type=int, default=300)     # smoke（final=1000）
    ap.add_argument("--outer-k", type=int, default=5)
    ap.add_argument("--inner-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cells", default=None, help="逗号分隔的 cell 过滤（如 forecast|snrLow|full）；空=全部")
    ap.add_argument("--diagnose", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else RESULTS
    cell_filter = set(args.cells.split(",")) if args.cells else None

    t0 = time.time()
    corpus = build_corpus(args.n_seeds)
    cells, snr_of = assign_cells(corpus)
    print(f"== corpus {len(corpus)} series → {len(cells)} cells; base={len(BASE_POOL)} + F0={len(F0_ACTIONS)} actions ==")
    if args.diagnose:
        for c in sorted(cells):
            print(f"  {c}: {len({rs.series_uid for rs in cells[c]})} uids")
        return

    base_map = fixed_harness_variants("forecast")
    f0_map = f0_variants("forecast")
    report = run(cells, snr_of, base_map, f0_map, args.n_boot, args.outer_k, args.inner_k,
                 args.seed, out_dir, cell_filter)
    _save(report, out_dir)                                # 终写（含 aggregate）

    ag = report["aggregate"]
    print("\n" + "=" * 78 + "\n== E-3.3 FAMILY 0 (exploratory) SUMMARY ==")
    print(f"  cell-equal Δ_supply(grouped) = {ag['cell_equal_delta_mean']:+.4f}")
    dec = ag["decisive"]
    if dec:
        print(f"  DECISIVE {DECISIVE_CELL}: Δ={dec['delta_mean']:+.4f} CI[{dec['ci_lo']:+.4f},{dec['ci_hi']:+.4f}] "
              f"pos%={dec['frac_boot_positive']:.2f} pick_freq={dec['expanded_pick_freq']}")
    print(f"  worst cell {ag['worst_cell']}: Δ={ag['worst_cell_delta']:+.4f} CI_lo={ag['worst_cell_ci_lo']:+.4f}")
    print(f"  snrHigh harm (min CI_lo over snrHigh cells) = {ag['snrHigh_harm_min_ci_lo']:+.4f}")
    print(f"\n  → report.json @ {out_dir}  [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
