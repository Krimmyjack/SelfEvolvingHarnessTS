"""run_real_longrun.py — 真实 Monash 数据上的长跑实验（plan.md P0：验 C3 跨域 + 自进化）。

两步实验（用 flag 切换；共享同一套加载/报告/进化代码）：

  Step 1  观察    `--mode diag`（免 LLM，默认）            → 编码器表现 + cell 分布 + C1 机会
          进化    `--mode evolve --start minimal`         → 真 LLM 自进化，观察 C1 分化
  Step 2  恢复    `--mode evolve --start degraded`        → 关键算子关掉 → 框架自主恢复到最优证据

报告三件套（呼应任务）：
  ① **编码器表现**：每 forecast cell 在基线 harness 下 grounded nRMSE vs seasonal_naive floor
     （冻结编码器在真实信号上是否有用：nRMSE < floor 即编码器有迁移价值）。`--encoder real` 用
     真实留出集预训编码器（leave-signal-out 防泄漏）对照零样本合成编码器。
  ② **cell 分布**：真实信号经网格退化后落入哪些 (task|snr|miss) cell、各多少样本、struct_feats 摘要。
  ③ **C1 分化 / 自进化**：每 cell 学到的 cell-scoped 模板 + OPD 算子归因；degraded 起点下的恢复轨迹。

运行：
  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_real_longrun
  ... -m SelfEvolvingHarnessTS.run_real_longrun --mode evolve --start minimal --epochs 3
  ... -m SelfEvolvingHarnessTS.run_real_longrun --mode evolve --start degraded --epochs 3
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .harness import HarnessState, EditPatch, Manifest
from .conditioning.key import struct_feats
from .data import load_signals, build_real_corpus, split_encoder_eval, FORECAST_PRESETS, ANOMALY_PRESETS
from .slow_path import BatchBuilder
from .slow_path.batch_builder import make_eval_sample
from .fast_path.pipeline import process as fast_process
from .evaluators import get_evaluator, seasonal_naive_floor

_DEGRADE_OPS = ["winsorize", "outlier_iqr", "outlier_mad"]
N_MIN = 6


# ════════════════════════════ harness 起点 ════════════════════════════
def degraded_harness() -> HarnessState:
    """关掉 forecast 关键离群算子 → 制造头部空间（Step 2 起点）。"""
    h = HarnessState.from_minimal()
    for op in _DEGRADE_OPS:
        h.apply_edit(EditPatch("L2", "set", f"l2.active_operators.{op}", False, Manifest("seed_degrade")))
    return h


# ════════════════════════════ 编码器（E1 零样本合成 / E2 真实留出预训）════════════════════════════
def setup_encoder(kind: str, all_signals, seed: int = 0, cache: str = None):
    """返回用于 build 语料的信号集合。kind=real 时换进程内冻结编码器为真实留出预训版（leave-signal-out）。

    cache 给定且存在 → 直接加载已预训编码器（split 必须与缓存时一致：同 npz/seed/frac → ev 确定性互补集，无泄漏）。
    """
    if kind == "synthetic":
        return all_signals                       # 默认编码器（合成预训缓存），全部信号入语料
    if kind == "real":
        import pathlib
        from .evaluators import pretrain_encoder_real, set_frozen_encoder, load_frozen_encoder
        from .data.synthetic_gen import H_FORECAST
        pre, ev = split_encoder_eval(all_signals, frac=0.5, seed=seed)
        print(f"[encoder=real] pretrain on {len(pre)} signals "
              f"({dict(Counter(s.config for s in pre))}), eval on {len(ev)} signals "
              f"({dict(Counter(s.config for s in ev))})")
        if cache and pathlib.Path(cache).exists():
            print(f"[encoder=real] load cached encoder {cache}")
            set_frozen_encoder(load_frozen_encoder(cache))
        else:
            hist = [s.clean[: s.clean.size - H_FORECAST] for s in pre]   # 预训用 clean 历史段（z-score 已完成）
            enc = pretrain_encoder_real(hist, epochs=120, cache_path=cache)
            if cache:
                print(f"[encoder=real] pretrained + cached → {cache}")
            set_frozen_encoder(enc)
        return ev                                # 仅在不相交的 eval 信号上评估（防泄漏）
    raise ValueError(f"--encoder ∈ {{synthetic, real}}, got {kind!r}")


# ════════════════════════════ 语料 → BatchBuilder ════════════════════════════
def build_bb(h: HarnessState, signals, *, n_per_signal: int, tasks) -> BatchBuilder:
    bb = BatchBuilder(h, n_min=N_MIN)
    for rs in build_real_corpus(signals, n_per_signal=n_per_signal, tasks=tasks):
        bb.add_raw_series(rs)
    return bb


# ════════════════════════════ 报告 ① 编码器表现 + ② cell 分布 ════════════════════════════
def _cell_metric(bb: BatchBuilder, h: HarnessState, cell_id: str, n: int):
    """对 cell 的前 n 样本：基线 harness 跑 fast_path 产 ready → grounded val_loss（forecast 另算 floor）。"""
    samples = bb.pools[cell_id][:n]
    task = samples[0].task_type
    eb = [make_eval_sample(fast_process(s.raw, s.task_type, h, store=None)[1], s) for s in samples]
    grounded = get_evaluator(task).evaluate(eb, layer="grounded")
    floor = seasonal_naive_floor(eb) if task == "forecast" else float("nan")
    return grounded, floor, len(samples)


def report_diag(bb: BatchBuilder, h: HarnessState, *, encoder_kind: str) -> Dict[str, tuple]:
    cells = sorted(bb.triggerable_cells())
    all_cells = sorted(bb.pools.keys())
    print(f"\n== ② cell distribution ({len(bb.pools)} cells, triggerable≥{2*N_MIN}: {len(cells)}) ==")
    # struct_feats 摘要（每 cell 头样本）
    print(f"   {'cell':30s}{'n':>5}{'trig':>6}  | median struct_feats (SNR/period/seas/trend/miss/outlier)")
    for c in all_cells:
        pool = bb.pools[c]
        feats = [struct_feats(s.raw) for s in pool[: min(8, len(pool))]]
        med = {k: float(np.median([f[k] for f in feats])) for k in
               ("SNR", "period", "seasonal_strength", "trend_strength", "missing_rate", "outlier_density")}
        trig = "yes" if c in cells else "—"
        print(f"   {c:30s}{len(pool):>5}{trig:>6}  | "
              f"SNR={med['SNR']:6.2f} per={med['period']:6.1f} seas={med['seasonal_strength']:.2f} "
              f"trend={med['trend_strength']:.2f} miss={med['missing_rate']:.2f} out={med['outlier_density']:.3f}")

    print(f"\n== ① encoder performance (baseline v{h.version}, encoder={encoder_kind}) ==")
    print("   forecast: grounded nRMSE vs seasonal_naive floor (nRMSE<floor → 编码器有迁移价值)")
    metrics: Dict[str, tuple] = {}
    for c in cells:
        g, fl, n = _cell_metric(bb, h, c, 2 * N_MIN)
        metrics[c] = (g, fl, n)
        if c.startswith("forecast|"):
            verdict = "OK(beats floor)" if np.isfinite(g) and np.isfinite(fl) and g < fl else "WEAK(≥floor)"
            print(f"   {c:30s} nRMSE={g:7.3f}  floor={fl:7.3f}  Δ={fl - g:+7.3f}  [{verdict}]")
        else:
            print(f"   {c:30s} 1-recall={g:7.3f}  (n={n})")
    return metrics


# ════════════════════════════ 报告 ③ C1 分化（进化后）════════════════════════════
def report_evolution(ev, h: HarnessState, base_active: Dict[str, bool], cells: List[str]) -> None:
    rounds_with_accept, accepted_paths, proposed_paths, pareto_viol = 0, [], [], 0
    print("\n== per-round trace ==")
    for rr in ev.history:
        tag = "ACCEPT" if rr.n_accepted else "······"
        print(f"  [{tag}] {rr.cell_id:28s} ep{rr.epoch} r{rr.round_idx} budget={rr.budget} "
              f"proposed={rr.n_proposed} accepted={rr.n_accepted}")
        for rs in rr.reasons:
            proposed_paths.append(rs.split(":", 1)[0])
            if "pareto_violation" in rs:
                pareto_viol += 1
            if rr.n_accepted:
                print(f"           - {rs}")
        if rr.n_accepted:
            rounds_with_accept += 1
            accepted_paths.extend(rr.accepted_paths)

    n_rounds = len(ev.history)
    print("\n== ③ evolution summary ==")
    print(f"  rounds with ≥1 accept : {rounds_with_accept}/{n_rounds} "
          f"({rounds_with_accept / max(1, n_rounds):.0%})")
    print(f"  accepted paths        : {dict(Counter(accepted_paths))}")
    print(f"  pareto violations     : {pareto_viol}")
    print(f"  frozen cells          : {[c for c, s in ev.schedules.items() if s.status == 'frozen']}")
    print(f"  final version         : v{h.version} (patch_log={len(h.patch_log)})")

    changed = {k: (base_active[k], v) for k, v in h.l2.active_operators.items() if base_active.get(k) != v}
    print(f"\n  active_operators changes vs start : {changed if changed else 'none'}")
    print(f"  cell-scoped templates             : {len(h.l2.task_templates)}")
    for name, t in h.l2.task_templates.items():
        cell = t.applies_to.get("pattern_conditions", {}).get("pattern_bin", "GLOBAL")
        ops = [op for st in t.stages for op in st.preferred_ops]
        bans = [op for st in t.stages for op in st.banned_ops]
        print(f"     {name}: cell={cell}  prefer={ops}  ban={bans}")

    print("\n  learned per-cell operator attribution (OPD 公式11):")
    for c in cells:
        summ = ev.attribution.summary(c)
        if summ["prefer"] or summ["avoid"]:
            print(f"     {c}:  prefer={[(op, round(v, 3)) for op, v, n in summ['prefer']]}  "
                  f"avoid={[(op, round(v, 3)) for op, v, n in summ['avoid']]}")


# ════════════════════════════ 主流程 ════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="真实 Monash 数据上的自进化长跑")
    ap.add_argument("--mode", choices=["diag", "evolve"], default="diag",
                    help="diag=免LLM诊断(编码器+cell分布)；evolve=真LLM自进化")
    ap.add_argument("--start", choices=["minimal", "degraded"], default="minimal",
                    help="进化起点：minimal=完整最小harness；degraded=关键算子关掉(Step 2 恢复实验)")
    ap.add_argument("--encoder", choices=["synthetic", "real"], default="synthetic",
                    help="grounded forecast 冻结编码器：synthetic=零样本合成预训(实测更强)；real=真实留出集预训")
    ap.add_argument("--forecast-target", choices=["raw", "ensemble", "seasonal_resid"], default="ensemble",
                    help="grounded forecast 目标(仅 frozen substrate)：ensemble=收缩混合 frozen⊕seasonal-naive(默认)")
    ap.add_argument("--substrate", choices=["frozen", "chronos"], default="frozen",
                    help="grounded forecast 判官：frozen=本地LSTM；chronos=真 TS foundation(零样本,破naive底,默认下不开因下载/成本)")
    ap.add_argument("--chronos-model", default="amazon/chronos-bolt-small",
                    help="chronos substrate 的 HF 模型 id（bolt-small/base，或 chronos-2）")
    ap.add_argument("--report-readiness", action="store_true", default=True,
                    help="evolve 末在 final_test split 上用独立报告器(⟂判官)出 ΔPerf 主表行(Ours-evolved)")
    ap.add_argument("--no-report-readiness", dest="report_readiness", action="store_false")
    ap.add_argument("--report-targets", default="lstm_scratch,dlinear_scratch",
                    help="独立报告器 target（逗号分隔；自动剔除与 substrate 判官同源者）")
    ap.add_argument("--tasks", default="forecast,anomaly_detection",
                    help="逗号分隔任务子集")
    ap.add_argument("--eps", type=float, default=None,
                    help="接受律 ε（默认 config EPS_NARROW=0.03；配对校准: frozen≈0.03/chronos≈0.08，见 run_calibrate_eps）")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-per-signal", type=int, default=4, help="每真实信号的退化 seed 数(扩样本)")
    ap.add_argument("--configs", default=None, help="逗号分隔 Monash config 子集(默认全部)")
    ap.add_argument("--npz", default=None, help="真实语料 npz 路径(默认 AdaCTS/data/monash_real.npz)")
    ap.add_argument("--encoder-cache", default=None, help="真实编码器缓存(.pt)：存在则加载，否则预训后写入")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    configs = [c.strip() for c in args.configs.split(",")] if args.configs else None

    from .evaluators import set_forecast_target, set_forecast_substrate
    set_forecast_target(args.forecast_target)     # 强化判官（默认 ensemble）：对 evaluator/validator 零侵入
    if args.substrate == "chronos":               # chronos 旁路 encoder/target（直接零样本预测）
        from .evaluators.chronos_probe import set_chronos_model
        set_chronos_model(args.chronos_model)
        set_forecast_substrate("chronos")
        print(f"== grounded substrate = chronos ({args.chronos_model}) ==")

    signals = load_signals(args.npz, configs=configs)
    print(f"== loaded {len(signals)} real signals ==  configs={dict(Counter(s.config for s in signals))}  "
          f"periods={dict(Counter(s.period for s in signals))}  forecast_target={args.forecast_target}")

    eval_signals = setup_encoder(args.encoder, signals, seed=args.seed, cache=args.encoder_cache)

    h = degraded_harness() if args.start == "degraded" else HarnessState.from_minimal()
    base_active = dict(h.l2.active_operators)
    bb = build_bb(h, eval_signals, n_per_signal=args.n_per_signal, tasks=tasks)
    print(f"== start harness: {args.start} v{h.version} ==  "
          f"degraded_ops_off={[op for op in _DEGRADE_OPS if not h.l2.active_operators.get(op, True)]}")

    # ── ①② 诊断（两种 mode 都先报）──
    t0 = time.time()
    metrics_before = report_diag(bb, h, encoder_kind=args.encoder)
    print(f"\n[diag done in {time.time() - t0:.1f}s]")

    if args.mode == "diag":
        print("\n[mode=diag] 免 LLM 诊断完成。要跑自进化加 --mode evolve（真 DeepSeek，有成本）。")
        return

    # ── ③ 自进化（真 LLM）──
    from .slow_path import Evolver, Proposer, Validator
    from .config.thresholds import EPS_NARROW
    from .llm import get_client
    eps = args.eps if args.eps is not None else EPS_NARROW
    cells = sorted(bb.triggerable_cells())
    client = get_client("flash", temperature=0.7, cache_name=f"real_{args.start}")
    ev = Evolver(h, bb, Proposer(llm=client, k=3), validator=Validator(eps=eps))
    print(f"\n== evolving {len(cells)} cells × {args.epochs} epochs (real LLM, ε={eps}"
          f"{' [配对校准]' if args.eps is not None else ' [默认0.03; chronos建议 --eps 0.08]'}) ==")
    t1 = time.time()
    summary = ev.run(n_epochs=args.epochs)
    print(f"[evolve done in {time.time() - t1:.1f}s]  LLM={client.stats()}")

    report_evolution(ev, h, base_active, cells)

    # ── 恢复对照（Step 2）：进化后再测一遍 in-loop 判官 nRMSE，看是否回落 ──
    print(f"\n== recovery check: forecast grounded nRMSE before → after (start={args.start}, judge=in-loop) ==")
    for c in cells:
        if not c.startswith("forecast|"):
            continue
        g_after, fl, _ = _cell_metric(bb, h, c, 2 * N_MIN)
        g_before = metrics_before[c][0]
        print(f"   {c:30s} {g_before:7.3f} → {g_after:7.3f}  (Δ={g_before - g_after:+7.3f}, floor={fl:.3f})")

    # ── ★ 主表行：独立报告器(⟂判官) × final_test split → Ours-evolved ΔPerf（非循环自证）──
    if args.report_readiness and "forecast" in tasks:
        from .run_main_table import delta_perf_table, print_delta_table   # 懒导入，避免与本模块的 import 环
        from .evaluators import disjoint_targets
        rtargets = disjoint_targets(args.substrate, [t.strip() for t in args.report_targets.split(",") if t.strip()])
        if rtargets:
            start_h = degraded_harness() if args.start == "degraded" else HarnessState.from_minimal()
            methods = {f"start[{args.start}]": start_h, "evolved": h}
            print(f"\n[main-table] reporter targets={rtargets} ⟂ judge({args.substrate}); final_test split (进化期未碰)")
            _, rows, agg = delta_perf_table(bb, methods, "forecast", rtargets, seeds=list(range(2)))
            print_delta_table(methods, rows, agg, rtargets,
                              title=f"Ours-evolved Data-Readiness ΔPerf (start={args.start})")
        else:
            print(f"\n[main-table] 跳过：所有 report-targets 与 judge({args.substrate}) 同源（无合规独立报告器）")


if __name__ == "__main__":
    main()
