"""run_main_table.py — ★ 主表 Data-Readiness ΔPerf 汇编（Experiment_Design_Final §★）。

**判官↔报告器分离**：本脚本只用 `evaluators.report_target` 的**独立 target**（默认 lstm_scratch +
dlinear_scratch，与 frozen/chronos 判官不相交，经 disjoint_targets 强制）在 **final-test split**
（进化期从未触碰，batch_builder.final_test）上报告 ΔPerf = perf(ready) − perf(raw)。

processing 方法（rows）：raw（基线，ΔPerf≡0）/ minimal / degraded /（可扩展 evolved）。
每 cell 在 final_test 上：raw_eval vs 各 harness 的 processed_eval → 独立 target perf → ΔPerf。
ΔPerf>0 = 处理后数据对独立下游模型更"就绪"——即 data readiness 的直接证据（无循环自证）。

运行：PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_main_table
      [--npz data/_artifacts/monash_clean.npz] [--targets lstm_scratch,dlinear_scratch] [--judge chronos]
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from .harness import HarnessState
from .data import load_signals, build_real_corpus
from .slow_path import BatchBuilder
from .slow_path.batch_builder import make_eval_sample
from .fast_path.pipeline import process as fast_process
from .evaluators import perf_multi, disjoint_targets
from .run_real_longrun import degraded_harness, N_MIN


# 7 固定 harness 变体（算子族平滑/离群强度扫描）→ per-cell oracle = C1 上界（无单一变体全 cell 最优）
_VARIANT_SPECS = {
    "v_none":         ["impute_linear"],                              # 仅插补（零平滑，anomaly 友好）
    "v_median":       ["impute_linear", "denoise_median"],           # 轻平滑
    "v_savgol":       ["impute_linear", "denoise_savgol"],           # 中平滑
    "v_stl":          ["impute_linear", "denoise_stl"],              # STL 去噪
    "v_wavelet":      ["impute_linear", "denoise_wavelet"],          # 小波去噪
    "v_winsor":       ["impute_linear", "winsorize"],                # 离群钳制（无平滑）
    "v_winsor_savgol": ["impute_linear", "winsorize", "denoise_savgol"],  # 离群+平滑（forecast 重）
}


def fixed_harness_variants(task: str = "forecast") -> dict:
    """构造 7 个固定 harness 变体：每个 = minimal + 一个全局 task_template，stages 强制其算子链
    （template 路径无条件施加，旁路 problem-gated heuristic → 变体间可比）。"""
    from .harness.layers import PipelineTemplate
    out = {}
    for name, chain in _VARIANT_SPECS.items():
        h = HarnessState.from_minimal()
        stages = [{"stage": "s1", "preferred_ops": [op], "banned_ops": [], "params_override": {}}
                  for op in chain]
        h.l2.task_templates[name] = PipelineTemplate.from_dict(
            {"name": name, "applies_to": {"task_type": task, "pattern_conditions": None}, "stages": stages})
        out[name] = h
    return out


def _eval_batch(cs_list, harness):
    """harness=None → raw passthrough（ready=raw 退化历史）；否则 fast_path 产 ready。"""
    if harness is None:
        return [make_eval_sample(cs.raw, cs) for cs in cs_list]
    return [make_eval_sample(fast_process(cs.raw, cs.task_type, harness, store=None)[1], cs)
            for cs in cs_list]


def delta_perf_table(bb, methods: dict, task: str, targets, seeds, min_final: int = 4,
                     final_size: int = None):
    """复用核心：在 final_test split 上算每 cell × method 的 ΔPerf(vs raw)。

    methods: {name: harness|None}；返回 (cells, rows, agg)。rows=[(cell, {name: dmean_over_targets})]，
    agg={name: {target: [per-cell deltas]}}。判官↔报告器分离由调用方经 disjoint_targets 保证 targets ⟂ 判官。
    final_size: final_test split 的批大小（classify 需更大以支撑 CV；None→用 bb.n_min 默认）。
    """
    ft = lambda c: bb.final_test(c, batch_size=final_size)
    cells = sorted(c for c in bb.pools if c.startswith(task + "|") and len(ft(c)) >= min_final)
    agg = defaultdict(lambda: defaultdict(list))
    rows = []
    for c in cells:
        cs_list = ft(c)
        p_raw = perf_multi(_eval_batch(cs_list, None), task, targets, seeds)
        cellrow = {}
        for mname, h in methods.items():
            p = perf_multi(_eval_batch(cs_list, h), task, targets, seeds)
            per_t = [p[t][0] - p_raw[t][0] for t in targets
                     if np.isfinite(p[t][0]) and np.isfinite(p_raw[t][0])]
            for t in targets:
                if np.isfinite(p[t][0]) and np.isfinite(p_raw[t][0]):
                    agg[mname][t].append(p[t][0] - p_raw[t][0])
            cellrow[mname] = float(np.mean(per_t)) if per_t else float("nan")
        rows.append((c, cellrow))
    return cells, rows, agg


def print_delta_table(methods, rows, agg, targets, *, title="Data-Readiness ΔPerf (vs raw)"):
    names = list(methods)
    hdr = f"{'cell':30s}" + "".join(f"{m:>16s}" for m in names)
    print(f"\n== {title} ==  (reporter ⟂ judge, final_test split)")
    print(hdr)
    print("-" * len(hdr))
    for cell, cellrow in rows:
        print(f"{cell:30s}" + "".join(f"{('%+.3f' % cellrow[m]):>16s}" for m in names))
    print("\n  mean ΔPerf over cells (per independent target, vs raw):")
    for m in names:
        parts = []
        for t in targets:
            vals = [v for v in agg[m][t] if np.isfinite(v)]
            parts.append(f"{t}={np.mean(vals):+.3f}" if vals else f"{t}=nan")
        print(f"    {m:18s} " + "   ".join(parts))


def main():
    ap = argparse.ArgumentParser(description="主表 Data-Readiness ΔPerf（独立报告器 × final-test split）")
    ap.add_argument("--npz", default=None, help="语料 npz（默认 AdaCTS/data/monash_real.npz）")
    ap.add_argument("--task", default="forecast", help="forecast / anomaly_detection / classification")
    ap.add_argument("--targets", default=None, help="报告器 target（逗号分隔）；None→按 task 默认")
    ap.add_argument("--judge", default=None, choices=["chronos", "frozen", "rocket", "inception"],
                    help="进化所用判官（用于 disjoint 标注，确保独立报告器⟂判官）；None→按 task 默认")
    ap.add_argument("--seeds", type=int, default=2, help="from-scratch target 多 seed 平均")
    ap.add_argument("--n-per-signal", type=int, default=4, help="forecast/anomaly 每信号退化 seed 数")
    ap.add_argument("--max-signals", type=int, default=400, help="[classify] ECG5000 截断条数（控运行时）")
    ap.add_argument("--final-size", type=int, default=None,
                    help="final_test 批大小；classify 默认 40（撑住 CV），forecast 用 bb.n_min")
    args = ap.parse_args()

    is_clf = args.task in ("classification", "classify")
    task = "classification" if is_clf else args.task
    judge = args.judge or ("rocket" if is_clf else "chronos")
    default_targets = "inception,rocket" if is_clf else "lstm_scratch,dlinear_scratch"
    targets_all = [t.strip() for t in (args.targets or default_targets).split(",") if t.strip()]
    independent = disjoint_targets(judge, targets_all)        # ⟂judge → 支撑非循环 headline 主张
    if not independent:
        raise SystemExit(f"所有 target 都与 judge={judge} 同源 → 无独立报告器；换 target")
    seeds = list(range(args.seeds))
    print(f"== Main-table reporter: targets={targets_all} (independent ⟂judge={judge}: {independent})  seeds={seeds} ==")

    # ── 语料 + cell 装配（classify=ECG5000 真实标签锚；其余=Monash 真实锚）──
    if is_clf:
        from .evaluators import set_classify_substrate
        from .data import load_class_signals, build_real_classify_corpus
        set_classify_substrate("rocket")                     # in-loop 判官底座=确定性 ROCKET（⟂ inception 报告器）
        final_size = args.final_size or 40
        bb = BatchBuilder(HarnessState.from_minimal(), n_min=final_size)
        sigs = load_class_signals(max_signals=args.max_signals)
        print(f"== loaded {len(sigs)} ECG5000 class signals (len={sigs[0].clean.size}) ==")
        for rs in build_real_classify_corpus(sigs, n_per_signal=1):
            bb.add_raw_series(rs)
    else:
        final_size = args.final_size
        bb = BatchBuilder(HarnessState.from_minimal(), n_min=N_MIN)
        for rs in build_real_corpus(load_signals(args.npz), n_per_signal=args.n_per_signal, tasks=(task,)):
            bb.add_raw_series(rs)

    variants = fixed_harness_variants(task)
    methods = {**variants, "minimal": HarnessState.from_minimal(), "degraded": degraded_harness()}
    t0 = time.time()
    cells, rows, agg = delta_perf_table(bb, methods, task, targets_all, seeds, final_size=final_size)
    print(f"== {len(cells)} {task} cells with final_test≥4 (split ⟂ held_in/held_out_a) ==")
    print_delta_table(variants, rows, agg, targets_all, title="Fixed-harness variants ΔPerf (vs raw)")
    print(f"  [independent reporters ⟂ judge({judge})] = {independent}"
          + (f";  cross-ref(=judge, 非 headline): {[t for t in targets_all if t not in independent]}"
             if len(targets_all) > len(independent) else ""))

    # ── per-cell oracle（每 cell 取最优变体）+ single-best-global（全 cell 平均最优的单一变体）= C1 上界 ──
    print("\n== C1 上界：per-cell oracle vs single-best-global（仅 7 变体内）==")
    print(f"  {'cell':30s}{'oracle ΔPerf':>16s}   winner（不同 cell 不同赢家 = 没有单一 harness 全局最优）")
    for cell, cr in rows:
        cand = {v: cr[v] for v in variants if np.isfinite(cr[v])}
        if not cand:
            print(f"  {cell:30s}{'nan':>16s}")
            continue
        wv = max(cand, key=cand.get)
        print(f"  {cell:30s}{('%+.3f' % cand[wv]):>16s}   {wv}")
    glob = {v: np.nanmean([cr[v] for _, cr in rows if np.isfinite(cr[v])] or [np.nan]) for v in variants}
    single_best = max(glob, key=lambda v: (glob[v] if np.isfinite(glob[v]) else -1e9))
    oracle_mean = float(np.nanmean([max((cr[v] for v in variants if np.isfinite(cr[v])), default=np.nan)
                                    for _, cr in rows]))
    print(f"\n  single-best-global = {single_best} (mean ΔPerf {glob[single_best]:+.3f})")
    print(f"  per-cell oracle mean ΔPerf = {oracle_mean:+.3f}  (gap over single-best = "
          f"{oracle_mean - glob[single_best]:+.3f} = 条件化的价值/C1)")
    print(f"  参照: minimal mean={np.nanmean([cr['minimal'] for _, cr in rows]):+.3f}  "
          f"degraded mean={np.nanmean([cr['degraded'] for _, cr in rows]):+.3f}  "
          f"(Ours-evolved 见 run_real_longrun --report-readiness)")
    print(f"\n[{time.time() - t0:.1f}s]  ΔPerf>0 = readiness gain over raw on an independent downstream model")
    print("[separation] reporter models ⟂ in-loop judge → readiness claim is non-circular (Exp_Design §★.2)")


if __name__ == "__main__":
    main()
