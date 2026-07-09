"""run_classify_longrun.py — classify 自进化长跑（阶段B，BUILD §4.4 之后）。

接 §4.4 的 classify 端到端（ECG5000 锚 + ROCKET 确定性判官 + InceptionLite 独立报告器），
把固定 7 变体主表升级为**真 LLM 自进化**：从 minimal harness 起，proposer(DeepSeek) 逐 cell 提
cell-scoped 清洗模板，validator 用**确定性 ROCKET CE**（σ=0，set_classify_substrate('rocket')）裁决。

核心看点（classify 版 C1 自进化）：
  • 进化器是否为不同 (SNR×missing) cell 学到**不同**的轻清洗模板（denoise_median/winsorize），
    且**正确避开**会抹掉判别形态的重平滑 denoise_stl（forecast 会选它 → 跨任务分化）；
  • OPD 归因是否在 classify cell 给出与 §4.4 一致的 per-op 价值（median/winsor prefer，stl avoid）；
  • 末在 final_test split 用**独立 InceptionLite 报告器(⟂ROCKET 判官)** 出 Ours-evolved ΔPerf（非循环）。

复用进化引擎（任务无关）+ run_real_longrun.report_evolution + run_main_table.delta_perf_table。
注：mining 对 classify 无 seasonal floor → weakness.improvable=False（仅提示，不门控提议）；
   mine_strength 返回 None（无 must-preserve）→ Pareto 安全仍由 validator held_out_b 守。

运行：
  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_classify_longrun
  ... --start minimal --epochs 3 --max-signals 300
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from .harness import HarnessState
from .data import load_class_signals, build_real_classify_corpus
from .slow_path import BatchBuilder, Evolver, Proposer, Validator
from .slow_path.batch_builder import make_eval_sample
from .fast_path.pipeline import process as fast_process
from .evaluators import get_evaluator, set_classify_substrate, disjoint_targets
from .config.thresholds import EPS_NARROW
from .llm import get_client
from .run_real_longrun import degraded_harness, report_evolution

N_MIN_CLF = 20          # classify cell batch（撑 ROCKET 5-fold CV by-series + final_test）


def _clf_ce(bb: BatchBuilder, h: HarnessState, cell_id: str, n: int) -> float:
    """cell 前 n 样本：harness 跑 fast_path 产 ready → ROCKET grounded CE（越低越好）。"""
    samples = bb.pools[cell_id][:n]
    eb = [make_eval_sample(fast_process(s.raw, s.task_type, h, store=None)[1], s) for s in samples]
    return get_evaluator("classification").evaluate(eb, layer="grounded")   # substrate=rocket（已全局设）


def main():
    ap = argparse.ArgumentParser(description="classify（ECG5000）自进化长跑")
    ap.add_argument("--start", choices=["minimal", "degraded"], default="minimal",
                    help="进化起点：minimal=完整最小harness（classify 主推：从零学轻清洗）；degraded=关离群算子")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-signals", type=int, default=300, help="ECG5000 截断条数")
    ap.add_argument("--n-min", type=int, default=N_MIN_CLF, help="cell batch（split/final_test 用）")
    ap.add_argument("--eps", type=float, default=None, help="接受律 ε（默认 EPS_NARROW=0.03，作用于 ROCKET CE）")
    ap.add_argument("--k", type=int, default=3, help="proposer 每轮候选数")
    ap.add_argument("--report-readiness", action="store_true", default=True,
                    help="末在 final_test 用独立 InceptionLite 报告器(⟂ROCKET) 出 Ours-evolved ΔPerf")
    ap.add_argument("--no-report-readiness", dest="report_readiness", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_classify_substrate("rocket")                       # in-loop 判官=确定性 ROCKET（σ=0）
    eps = args.eps if args.eps is not None else EPS_NARROW

    sigs = load_class_signals(max_signals=args.max_signals, seed=args.seed)
    print(f"== loaded {len(sigs)} ECG5000 signals (len={sigs[0].clean.size}, "
          f"labels={dict(Counter(s.label for s in sigs))}) ==  judge=ROCKET(确定性) ε={eps}")

    h = degraded_harness() if args.start == "degraded" else HarnessState.from_minimal()
    base_active = dict(h.l2.active_operators)
    bb = BatchBuilder(h, n_min=args.n_min)
    for rs in build_real_classify_corpus(sigs, n_per_signal=1):
        bb.add_raw_series(rs)
    cells = sorted(bb.triggerable_cells())
    print(f"== start={args.start} v{h.version} ==  cells(triggerable≥{2*args.n_min}): {cells}")

    # ── before：每 cell 起点 grounded CE（ROCKET）──
    print("\n== before evolution: per-cell grounded CE (ROCKET judge) ==")
    ce_before = {}
    for c in cells:
        ce_before[c] = _clf_ce(bb, h, c, 2 * args.n_min)
        print(f"   {c:34s} CE={ce_before[c]:.4f}")

    # ── 自进化（真 DeepSeek）──
    client = get_client("flash", temperature=0.7, cache_name=f"clf_{args.start}")
    ev = Evolver(h, bb, Proposer(llm=client, k=args.k), validator=Validator(eps=eps))
    print(f"\n== evolving {len(cells)} classify cells × {args.epochs} epochs (real LLM) ==")
    t1 = time.time()
    ev.run(n_epochs=args.epochs)
    print(f"[evolve done in {time.time() - t1:.1f}s]  LLM={client.stats()}")

    report_evolution(ev, h, base_active, cells)

    # ── after：恢复/改进对照（ROCKET CE before→after）──
    print(f"\n== classify grounded CE before → after (start={args.start}, judge=ROCKET in-loop) ==")
    for c in cells:
        ca = _clf_ce(bb, h, c, 2 * args.n_min)
        print(f"   {c:34s} {ce_before[c]:.4f} → {ca:.4f}  (Δ={ce_before[c] - ca:+.4f})")

    # ── ★ Ours-evolved ΔPerf：独立 InceptionLite 报告器(⟂ROCKET) × final_test split（非循环）──
    if args.report_readiness:
        from .run_main_table import delta_perf_table, print_delta_table
        rtargets = disjoint_targets("rocket", ["inception", "rocket"])     # → ['inception']（独立）
        start_h = degraded_harness() if args.start == "degraded" else HarnessState.from_minimal()
        methods = {f"start[{args.start}]": start_h, "evolved": h}
        print(f"\n[main-table] reporter={rtargets} ⟂ judge(rocket); final_test split (进化期未碰)")
        _, rows, agg = delta_perf_table(bb, methods, "classification", rtargets,
                                        seeds=list(range(2)), final_size=args.n_min)
        print_delta_table(methods, rows, agg, rtargets,
                          title=f"Ours-evolved classify Data-Readiness ΔPerf (start={args.start})")


if __name__ == "__main__":
    main()
