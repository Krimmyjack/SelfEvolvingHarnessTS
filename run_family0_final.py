"""run_family0_final.py — E-3.3 Family 0 正式收尾（评审第十三轮「强化方案 b」）。

修正 exploratory（results/E1_1_family0/）的 5 处越界，产出 final-grade 判决材料：
  #1 B=1000 final（非 smoke B=100/300）——每 cell 各一个 B=1000 job。
  #2 「无 harm」≠全局安全 → cell×origin worst-group 非劣 LCB（δ_safe=0.05）。
  #3 缺聚合 CI → cell 等权 aggregate Δ_supply CI；D-only Lookup vs global 真正计算。
  #4 median vs MA 只作点级支配（combine 记录 f0_ma_w25 选择频率）。
  #5 可重放 → 每 cell 独立文件 + combine，无增量合并、无隐式混档。

数据独立性核验（AI 第 3 点，决定聚合重采样是否需按 clean-series identity 联合）：
  build_corpus（run_variance_decomp.py）里 clean 种子 `sd=_det_seed(struct, dname, j)` **含 dname**，
  而四个 cell 恰好对应四个 dname（d_hi_full/d_hi_miss/d_lo_full/d_lo_miss）→ **cell 间 clean 种子
  互不相同、无共享 latent identity**。故 cell 分层独立重采样合法，cell 等权 aggregate 分布 =
  各 cell B=1000 boot_deltas 的**卷积**（按 replicate 配对求 cell 均值）——与单一 joint pass 同分布，
  且可并行（A-34）。（反例：若 clean 种子不含 dname、跨 cell 复用，则必须按 clean-series identity
  联合重采样，不能独立 bootstrap——此处已排除。）此核验随 combine 落盘 provenance。

用法（并行；每 cell 一个 job）：
  PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -u \
      -m SelfEvolvingHarnessTS.run_family0_final --cell "forecast|snrLow|full" --n-boot 1000
  ... 四个 cell 全完成后：
  ... -m SelfEvolvingHarnessTS.run_family0_final --combine

产物：results/E1_1_family0_final/cell_<slug>.json（×4） + report.json + decision_final.md（combine 写）。
confirmatory seeds 20–39 **不在本脚本**（A-32d：F0 是否终池仍待四门 + 用户拍板）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .evaluators.frozen_probe import FrozenProbe
from .run_main_table import fixed_harness_variants, _VARIANT_SPECS
from .run_variance_decomp import build_corpus, assign_cells, build_cell_cache, _oof_losses
from .family0_actions import f0_variants, F0_DOSAGE_GRID
from .nested_supply import (delta_supply_grouped, delta_supply, nested_pool_losses,
                            DEFAULT_SEED)

RESULTS = Path(__file__).resolve().parent / "results" / "E1_1_family0_final"
BASE_POOL = list(_VARIANT_SPECS.keys())          # v2.1 的 7 动作
F0_ACTIONS = [a for a, _op, _w in F0_DOSAGE_GRID]
EXPANDED = BASE_POOL + F0_ACTIONS
DECISIVE_CELL = "forecast|snrLow|full"
DELTA_SAFE = 0.05                                 # worst-group 非劣容忍（A-31c gate④，本脚本取值，可复核）
EPS = 0.03


def _slug(cid: str) -> str:
    return cid.replace("|", "_")


def _point_col_mean(action_caches, common_uids, actions, kfold=5):
    fold_pt = {u: i % kfold for i, u in enumerate(common_uids)}
    out = {}
    for a in actions:
        losses = _oof_losses(action_caches[a], common_uids, fold_pt, kfold)
        out[a] = float(np.nanmean([losses.get(u, np.nan) for u in common_uids]))
    return out


def _boot_lcb(vals, n_boot=2000, seed=0, q=2.5):
    """test-uid 重采样 LCB（子群安全筛，非 full-refit；子群 n 小、此处只做 worst-group 非劣屏）。"""
    v = np.asarray(vals, float)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    return float(np.percentile(boots, q)), float(v.mean())


def run_cell(cid, n_boot, outer_k, inner_k, seed, out_dir):
    fp = FrozenProbe()
    variants_map = {**fixed_harness_variants("forecast"), **f0_variants("forecast")}
    corpus = build_corpus(20)                                 # 与 v2.1 语料一致（n_seeds=20）
    cells, _snr = assign_cells(corpus)
    if cid not in cells:
        raise SystemExit(f"cell {cid} 不在语料；可用: {sorted(cells)}")
    series = cells[cid]
    uids_all = {rs.series_uid for rs in series}
    if len(uids_all) < 6:
        raise SystemExit(f"cell {cid} LOWCONF (uids={len(uids_all)})")

    print(f"[final] {cid}  caching {len(variants_map)} actions × {len(uids_all)} uids …", flush=True)
    t = time.time()
    action_caches, common_uids, origin_of = build_cell_cache(fp, series, variants_map)
    n = len(common_uids)
    print(f"        cached; common_uids={n}  [{time.time()-t:.0f}s]", flush=True)

    # ── 诊断点 col_mean（in-sample，不判决）──
    pt = _point_col_mean(action_caches, common_uids, EXPANDED)
    base_best = min(BASE_POOL, key=lambda a: pt[a])
    f0_beats = sorted([a for a in F0_ACTIONS if pt[a] < pt[base_best]], key=lambda a: pt[a])

    # ── RAW 确定性 nested held-out（与 bootstrap 均值分开报告；AI 强调 1.75→1.59 是 boot 均值不是 raw 点）──
    lb_map, picks_b = nested_pool_losses(action_caches, BASE_POOL, common_uids, outer_k, inner_k, seed)
    le_map, picks_e = nested_pool_losses(action_caches, EXPANDED, common_uids, outer_k, inner_k, seed)
    common = [u for u in common_uids if u in lb_map and u in le_map]
    raw_du = {u: lb_map[u] - le_map[u] for u in common}
    raw = dict(loss_base=float(np.mean([lb_map[u] for u in common])),
               loss_expanded=float(np.mean([le_map[u] for u in common])),
               delta=float(np.mean([raw_du[u] for u in common])),
               det_picks_base=[p["selected"] for p in picks_b],
               det_picks_expanded=[p["selected"] for p in picks_e])
    print(f"        RAW nested: base={raw['loss_base']:.4f} → expanded={raw['loss_expanded']:.4f} "
          f"(Δ_raw={raw['delta']:+.4f}); det_picks_exp={raw['det_picks_expanded']}", flush=True)

    # ── per-action held-out loss（pool=[a]，nested 强制单动作）→ D-only Lookup vs global 的原料 ──
    per_action = {}
    for a in EXPANDED:
        la, _ = nested_pool_losses(action_caches, [a], common_uids, outer_k, inner_k, seed)
        per_action[a] = float(np.mean([la[u] for u in common if u in la]))
    print(f"        per-action held-out cached ({len(per_action)} actions)", flush=True)

    # ── per-origin(cell×struct) 子群 raw Δ + test-uid LCB（worst-group 非劣屏）──
    per_origin = {}
    for org in sorted({origin_of[u] for u in common}):
        us = [u for u in common if origin_of[u] == org]
        d = [raw_du[u] for u in us]
        lcb, mean = _boot_lcb(d, seed=seed + hash(org) % 9973)
        per_origin[org] = dict(n=len(us), delta=mean, lcb=lcb)
    print(f"        per-origin(cell×struct) Δ_raw / LCB: "
          f"{ {o: (round(s['delta'],3), round(s['lcb'],3), s['n']) for o, s in per_origin.items()} }",
          flush=True)

    # ── 正式 full-refit group bootstrap（B=1000）+ 单次参照（B=300）──
    # checkpoint/resume：后台任务 ~100min 墙钟寿命上限下鲁棒化（被杀重启自动续跑，A-36）
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"ckpt_{_slug(cid)}.json"
    print(f"        grouped bootstrap B={n_boot} (full-refit, ckpt={ckpt.name})…", flush=True)
    dg = delta_supply_grouped(action_caches, BASE_POOL, EXPANDED, common_uids,
                              outer_k=outer_k, inner_k=inner_k, seed=seed, n_boot=n_boot,
                              progress=max(1, n_boot // 20), ckpt_path=ckpt, ckpt_every=25)
    ds = delta_supply(action_caches, BASE_POOL, EXPANDED, common_uids,
                      outer_k=outer_k, inner_k=inner_k, seed=seed, n_boot=300)

    rec = dict(
        config=dict(cell=cid, n_boot=n_boot, outer_k=outer_k, inner_k=inner_k, seed=seed,
                    base_pool=BASE_POOL, f0_grid=[list(g) for g in F0_DOSAGE_GRID],
                    n_seeds_corpus=20, delta_safe=DELTA_SAFE, eps=EPS),
        uids=n, point_col_mean=pt, base_best_action=base_best, base_best_point=pt[base_best],
        f0_beats_base_point=f0_beats,
        raw_nested=raw, per_action_heldout=per_action, per_origin=per_origin,
        delta_supply_grouped={k: dg[k] for k in
            ("delta_mean", "ci_lo", "ci_hi", "median", "loss_base", "loss_expanded",
             "frac_boot_positive", "expanded_pick_freq", "n_boot", "boot_deltas")},
        delta_supply_single={k: ds[k] for k in ("delta_mean", "ci_lo", "ci_hi")})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"cell_{_slug(cid)}.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[final] {cid} DONE: Δ_grouped={dg['delta_mean']:+.4f} "
          f"CI[{dg['ci_lo']:+.4f},{dg['ci_hi']:+.4f}] pos%={dg['frac_boot_positive']:.2f}  "
          f"[{time.time()-t:.0f}s] → cell_{_slug(cid)}.json", flush=True)


def combine(out_dir):
    files = sorted(out_dir.glob("cell_*.json"))
    if not files:
        raise SystemExit(f"无 cell_*.json 于 {out_dir}")
    cells = {}
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        cells[r["config"]["cell"]] = r
    cids = sorted(cells)

    # ── cell 等权 aggregate Δ_supply（各 cell 独立 boot_deltas 的卷积；见文件头独立性核验）──
    B = min(len(cells[c]["delta_supply_grouped"]["boot_deltas"]) for c in cids)
    arrs = np.array([cells[c]["delta_supply_grouped"]["boot_deltas"][:B] for c in cids])  # (n_cell, B)
    agg = arrs.mean(axis=0)                                   # cell 等权，逐 replicate
    aggregate = dict(
        method="cell_equal_convolution_of_independent_grouped_bootstraps",
        n_cells=len(cids), B=int(B),
        delta_mean=float(agg.mean()), ci_lo=float(np.percentile(agg, 2.5)),
        ci_hi=float(np.percentile(agg, 97.5)), frac_positive=float(np.mean(agg > 0)),
        gt_eps=bool(agg.mean() > EPS), ci_lo_gt_0=bool(np.percentile(agg, 2.5) > 0))

    # ── D-only Lookup vs global single（held-out）──
    # global single = 全 cell 等权 held-out loss 最优的**单一固定动作**；D-lookup = 每 cell nested 选择。
    global_loss = {a: float(np.mean([cells[c]["per_action_heldout"][a] for c in cids])) for a in EXPANDED}
    global_best = min(global_loss, key=global_loss.get)
    base_global_best = min(BASE_POOL, key=lambda a: global_loss[a])
    dlookup_loss = float(np.mean([cells[c]["raw_nested"]["loss_expanded"] for c in cids]))
    routing = dict(
        global_single_best_action=global_best, global_single_loss=global_loss[global_best],
        base_pool_global_best_action=base_global_best, base_pool_global_loss=global_loss[base_global_best],
        dlookup_loss=dlookup_loss,
        delta_route_vs_global=float(global_loss[global_best] - dlookup_loss),
        delta_route_gt_eps=bool(global_loss[global_best] - dlookup_loss > EPS),
        note="global_single=最优单一固定剂量(全 cell 同用)；D-lookup=cell(=SNR×miss)条件化选择；"
             "Δ>ε ⇒ degradation 条件化剂量路由较一刀切剂量有 held-out 价值(仍非 Pattern，见 E-3.2)")

    # ── cell×origin worst-group 非劣 ──
    subgroups = []
    for c in cids:
        for org, s in cells[c]["per_origin"].items():
            subgroups.append(dict(cell=c, origin=org, n=s["n"], delta=s["delta"], lcb=s["lcb"]))
    worst = min(subgroups, key=lambda s: s["lcb"])
    safety = dict(delta_safe=DELTA_SAFE, worst_subgroup=worst,
                  worst_lcb=worst["lcb"], pass_noninferiority=bool(worst["lcb"] > -DELTA_SAFE),
                  n_subgroups=len(subgroups), subgroups=subgroups,
                  note="子群 LCB 为 test-uid 重采样(非 full-refit)：worst-group 非劣屏；n 小 → 偏宽")

    # ── per-cell 摘要 ──
    percell = {}
    for c in cids:
        dg = cells[c]["delta_supply_grouped"]
        ds = cells[c]["delta_supply_single"]
        raw = cells[c]["raw_nested"]
        # median vs ma 支配（点级）：记录 f0_ma_* 在扩池 bootstrap 的选择频率
        pf = dg["expanded_pick_freq"]
        ma_share = float(sum(v for k, v in pf.items() if k.startswith("f0_ma_")))
        median_share = float(sum(v for k, v in pf.items() if k.startswith("f0_median_")))
        percell[c] = dict(
            uids=cells[c]["uids"],
            raw_loss_base=raw["loss_base"], raw_loss_expanded=raw["loss_expanded"], raw_delta=raw["delta"],
            boot_delta_mean=dg["delta_mean"], boot_loss_base=dg["loss_base"],
            boot_loss_expanded=dg["loss_expanded"], boot_ci=[dg["ci_lo"], dg["ci_hi"]],
            frac_boot_positive=dg["frac_boot_positive"], n_boot=dg["n_boot"],
            single_ci=[ds["ci_lo"], ds["ci_hi"]], single_delta=ds["delta_mean"],
            det_picks_expanded=raw["det_picks_expanded"],
            median_pick_share=median_share, ma_pick_share=ma_share, expanded_pick_freq=pf)

    report = dict(
        provenance=dict(
            base_pool=BASE_POOL, f0_grid=[list(g) for g in F0_DOSAGE_GRID], n_seeds_corpus=20,
            per_cell_n_boot={c: cells[c]["delta_supply_grouped"]["n_boot"] for c in cids},
            seed=cells[cids[0]]["config"]["seed"], eps=EPS, delta_safe=DELTA_SAFE,
            independence_check=("clean 种子 _det_seed(struct,dname,j) 含 dname → cell 间 disjoint → "
                                "cell 分层独立重采样合法，aggregate = 独立 per-cell bootstrap 卷积 (A-34)"),
            note="E-3.3 Family 0 final（强化方案 b）；每 cell 独立 B=1000 full-refit grouped bootstrap；"
                 "raw 确定性 nested 与 bootstrap 均值分列；confirmatory seeds 20-39 未开"),
        aggregate=aggregate, routing_d_vs_global=routing, safety_worst_group=safety, cells=percell)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 控制台判决摘要 ──
    print("\n" + "=" * 78 + "\n== E-3.3 FAMILY 0 FINAL (强化方案 b) ==")
    print(f"  cells: {cids}  per-cell n_boot={report['provenance']['per_cell_n_boot']}")
    print(f"\n  cell 等权 aggregate Δ_supply = {aggregate['delta_mean']:+.4f} "
          f"CI[{aggregate['ci_lo']:+.4f},{aggregate['ci_hi']:+.4f}] "
          f"pos%={aggregate['frac_positive']:.2f}  (>ε:{aggregate['gt_eps']} CI_lo>0:{aggregate['ci_lo_gt_0']})")
    print(f"  D-only Lookup vs global-single: global_best={routing['global_single_best_action']} "
          f"{routing['global_single_loss']:.4f} vs D-lookup {routing['dlookup_loss']:.4f} "
          f"→ Δ_route={routing['delta_route_vs_global']:+.4f} (>ε:{routing['delta_route_gt_eps']})")
    print(f"  worst-group(cell×origin): {worst['cell']}×{worst['origin']} "
          f"Δ={worst['delta']:+.4f} LCB={worst['lcb']:+.4f} (非劣>{-DELTA_SAFE}:{safety['pass_noninferiority']})")
    print("\n  per-cell (raw vs bootstrap 分列):")
    for c in cids:
        p = percell[c]
        print(f"    {c:22s} n={p['uids']:3d}  raw {p['raw_loss_base']:.3f}→{p['raw_loss_expanded']:.3f}"
              f"(Δ{p['raw_delta']:+.3f})  boot Δ{p['boot_delta_mean']:+.3f} "
              f"CI[{p['boot_ci'][0]:+.3f},{p['boot_ci'][1]:+.3f}] pos%{p['frac_boot_positive']:.2f} "
              f"medshare{p['median_pick_share']:.2f} mashare{p['ma_pick_share']:.2f}")
    print(f"\n  → report.json @ {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="E-3.3 Family 0 正式收尾（强化方案 b）")
    ap.add_argument("--cell", default=None, help="单 cell 名（如 'forecast|snrLow|full'）")
    ap.add_argument("--combine", action="store_true", help="合并 4 个 cell_*.json → report.json + 摘要")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--outer-k", type=int, default=5)
    ap.add_argument("--inner-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else RESULTS
    if args.combine:
        combine(out_dir)
    elif args.cell:
        run_cell(args.cell, args.n_boot, args.outer_k, args.inner_k, args.seed, out_dir)
    else:
        raise SystemExit("需 --cell <CID> 或 --combine")


if __name__ == "__main__":
    main()
