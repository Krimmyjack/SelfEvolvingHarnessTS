"""run_stream_s1.py — ★v4 S1 流式持续适应 + 三 bootstrap 消融 编排器。

跑 A(scratch)/B(frozen)/C(updating) 三 mode 于一条 domain 序列 → 落前向迁移 JSONL + 摘要。
B 复用 C 的 per-domain checkpoint（B−A=记忆价值, C−B=继续更新价值）。

用法：
  # 自洽合成 demo（免 LLM、免下载，stub proposer）：
  python -m SelfEvolvingHarnessTS.run_stream_s1 --epochs 2
  # 真 DeepSeek proposer：
  python -m SelfEvolvingHarnessTS.run_stream_s1 --llm flash --epochs 3
  # canonical + shuffle 鲁棒性：
  python -m SelfEvolvingHarnessTS.run_stream_s1 --order-seed 42

S1_Implementation_Plan §B/§C/§D。默认合成 forecast-only stream（K=3 patterns 当 domain）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Callable, List

import numpy as np

from .harness import HarnessState, EditPatch, Manifest
from .slow_path import deploy_stream as ds
from .slow_path.deploy_stream import DomainSpec
from .data import make_forecast_batch
from .evaluators import aggregate_time_to_readiness


# ──────────────────────────── proposer 工厂 ────────────────────────────
class _StubProposer:
    """免 LLM 的确定性 proposer：toggle winsorize（合法 leaf 编辑）。供自洽 demo。"""
    def propose(self, harness, weakness, strength=None, rejection_log=None):
        p = EditPatch(edited_layer="L2", op="set", path="l2.active_operators.winsorize", value=True,
                      manifest=Manifest("t", "toggle winsorize", "expect", "abl", "risk"), source_type="failure")
        p.cell_id = weakness.cell_id
        return [p]


def make_proposer_factory(llm: str, k: int = 3) -> Callable[[], object]:
    if llm in ("", "none", None):
        return _StubProposer
    from .llm import get_client
    from .slow_path import Proposer

    def factory():
        client = get_client(llm, temperature=0.7, cache_name=f"stream_s1_{llm}")
        return Proposer(llm=client, k=k)
    return factory


# ──────────────────────────── domain 序列 ────────────────────────────
def synthetic_domains(n: int) -> List[DomainSpec]:
    """合成 forecast-only stream：每个 pattern = 一个 domain（K=3）。"""
    from .data.synthetic_gen import PATTERNS
    return [DomainSpec(p, make_forecast_batch(p, n=n, seed0=100 * i), ("forecast",))
            for i, p in enumerate(PATTERNS)]


def real_domains(npz: str, n_per_signal: int, *, min_signals: int = 5,
                 max_per_domain: int = 0, max_domains: int = 0) -> List[DomainSpec]:
    """真实 forecast stream：按 **config**（Monash 数据集名）分 domain（每个数据集=一个 domain）。

    RealSignal 携 `config`（非 source/name）→ 之前用 source 分组会塌成 K=1。只保留信号 ≥ min_signals
    的 config（够触发 cell；默认丢 1-信号的 us_births/saugeenday/sunspot）。每 domain 4 退化 preset
    （G_hi/lo × full/miss）→ 同 config 内混合质量，制造 readiness headroom。
    """
    from .data.load_real import load_signals, make_real_forecast_batch, FORECAST_PRESETS
    from collections import defaultdict
    sigs = load_signals(npz_path=npz)
    by_dom = defaultdict(list)
    for s in sigs:
        by_dom[getattr(s, "config", "real")].append(s)
    groups = [(name, g) for name, g in by_dom.items() if len(g) >= min_signals]
    groups.sort(key=lambda kv: (-len(kv[1]), kv[0]))     # canonical：信号数降序，名次稳定
    if max_domains > 0:
        groups = groups[:max_domains]
    doms = []
    for name, group in groups:
        if max_per_domain > 0:
            group = group[:max_per_domain]
        corpus = []
        for pre in FORECAST_PRESETS:
            corpus += make_real_forecast_batch(group, pre, n_per_signal=n_per_signal, seed0=0)
        doms.append(DomainSpec(str(name), corpus, ("forecast",)))
    return doms


def order_domains(domains: List[DomainSpec], order_seed: int) -> List[DomainSpec]:
    if order_seed <= 0:
        return domains                       # canonical
    rng = random.Random(order_seed)
    shuffled = list(domains); rng.shuffle(shuffled)
    return shuffled


# ──────────────────────────── 摘要 ────────────────────────────
def summarize(result: ds.StreamResult) -> List[dict]:
    rows = []
    for d in result.domains:
        ttr = [r["time_to_readiness_rounds"] for r in d.cell_logs]
        agg = aggregate_time_to_readiness(ttr)
        rd = [r["readiness_at_budget"] for r in d.cell_logs if np.isfinite(r["readiness_at_budget"])]
        rows.append({"k": d.domain_idx, "domain": d.name, "mode": d.mode,
                     "version": d.harness_version, "n_cells": len(d.cell_logs),
                     "ttr_median": agg["median"], "ttr_max": agg["max"],
                     "readiness_mean": (float(np.mean(rd)) if rd else float("nan")),
                     "n_reval_demote": d.n_reval_demote})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="S1 流式持续适应 + 三 bootstrap 消融")
    ap.add_argument("--modes", default="updating,frozen,scratch")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--min-batches", type=int, default=2)
    ap.add_argument("--n", type=int, default=40, help="合成每 domain 序列数")
    ap.add_argument("--llm", default="none", help="none(stub) | flash | pro")
    ap.add_argument("--k", type=int, default=3, help="proposer 每 cell 候选数（每个=1 LLM 调用）")
    ap.add_argument("--npz", default="", help="真实 monash npz；给定则用真实 domain 流")
    ap.add_argument("--n-per-signal", type=int, default=4)
    ap.add_argument("--max-per-domain", type=int, default=0, help="每 domain 取前 N 信号（0=全取；控时长）")
    ap.add_argument("--max-domains", type=int, default=0, help="只取前 K domain（0=全取）")
    ap.add_argument("--min-signals", type=int, default=5, help="config 信号 < 此则丢弃该 domain")
    ap.add_argument("--order-seed", type=int, default=0, help="0=canonical；>0=shuffle 鲁棒性")
    ap.add_argument("--substrate", default="frozen", help="forecast 判官底座：frozen(默认) | chronos(强判官,揭 headroom) | scratch")
    ap.add_argument("--out-dir", default="runs/s1")
    ap.add_argument("--cand-log", default="", help="S0.5 候选级 JSONL 路径（仅记 updating 持续路径；空=不记）")
    args = ap.parse_args()

    if args.substrate != "frozen":
        from .evaluators import set_forecast_substrate
        set_forecast_substrate(args.substrate)
        print(f"[S1] forecast substrate = {args.substrate}")

    os.makedirs(args.out_dir, exist_ok=True)
    domains = (real_domains(args.npz, args.n_per_signal, min_signals=args.min_signals,
                            max_per_domain=args.max_per_domain, max_domains=args.max_domains)
               if args.npz else synthetic_domains(args.n))
    domains = order_domains(domains, args.order_seed)
    print(f"[S1] domains (K={len(domains)}): {[d.name for d in domains]} | llm={args.llm} | epochs={args.epochs}")

    factory = make_proposer_factory(args.llm, k=args.k)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    cand_logger = None
    if args.cand_log:
        from .slow_path import CandidateLogger
        cand_logger = CandidateLogger(args.cand_log, run_id=f"s1_{args.llm}")
        print(f"[S1] candidate-level log (updating only) → {args.cand_log}")

    checkpoints = None
    all_summary = {}
    # 先跑 updating（产 checkpoint 供 frozen）
    for mode in sorted(modes, key=lambda m: {"updating": 0, "frozen": 1, "scratch": 2}.get(m, 9)):
        kwargs = dict(mode=mode, make_harness=HarnessState.from_minimal,
                      n_epochs_per_domain=args.epochs, min_batches=args.min_batches,
                      candidate_logger=cand_logger,
                      log_path=os.path.join(args.out_dir, f"forward_transfer_{mode}.jsonl"))
        if mode == "frozen":
            kwargs["bootstrap_checkpoints"] = checkpoints     # 可能 None（k=0 退 minimal）
        else:
            kwargs["make_proposer"] = factory
        res = ds.deploy_stream(domains, **kwargs)
        if mode == "updating":
            checkpoints = res.checkpoints()
        all_summary[mode] = summarize(res)

    # 打印 + 存摘要
    print("\n=== S1 前向迁移摘要（ttr 单位=round；readiness 越高越就绪）===")
    for mode, rows in all_summary.items():
        print(f"\n[{mode}]")
        for r in rows:
            print(f"  k={r['k']} {r['domain']:<10} ver={r['version']:<3} "
                  f"ttr(med/max)={r['ttr_median']}/{r['ttr_max']} "
                  f"readiness={r['readiness_mean']:.3f} reval_demote={r['n_reval_demote']}")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[S1] logs + summary → {args.out_dir}/")


if __name__ == "__main__":
    main()
