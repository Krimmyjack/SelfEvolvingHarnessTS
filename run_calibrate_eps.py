"""run_calibrate_eps.py — ε 按**配对**判官信号标定（修正版，plan.md B.1 / Experiment_Design §6 P0）。

接受律比较的是**同一 batch 上 cur vs cand**（确定性：chronos/frozen + heuristic-compose 无训练/采样噪声）
→ ε 该由**配对同-batch Δ** 决定，不是 batch-A vs batch-B 的无配对方差（后者含「不同序列可预测性不同」
的横截面异质，在同-batch cur−cand 中**相互抵消**，会把 ε 估得过大）。

本脚本测两类配对 Δ（每个 cell 内、每个不相交 batch 上）：
  ① **未处理 vs 处理**：Δ = val(raw) − val(minimal)  （用户视角：同 batch 注入噪声未处理 vs 处理；>0=处理有益）
  ② **真实编辑**：Δ = val(minimal) − val(degraded)  （accept 律实际面对的 cur vs cand 形态）
对每个 (cell, 对比)：effect = mean_batch(Δ)；**跨 batch σ_Δ = 该编辑效应的泛化噪声** → ε 应 ≳ σ_Δ
（held_in 的 Δ 要 > ε 才大概率在别的 batch 也 >0）。推荐 ε ≈ 跨 cell 中位 σ_Δ。

运行：PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_calibrate_eps
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np

from .harness import HarnessState
from .data import load_signals, build_real_corpus
from .slow_path import BatchBuilder
from .slow_path.batch_builder import make_eval_sample
from .fast_path.pipeline import process as fast_process
from .evaluators import get_evaluator, set_forecast_substrate
from .run_real_longrun import N_MIN, degraded_harness


def _val(samples, harness):
    """同-batch grounded val_loss。harness=None → raw passthrough（ready=未处理退化序列）。遵全局 _SUBSTRATE。"""
    task = samples[0].task_type
    if harness is None:
        eb = [make_eval_sample(s.raw, s) for s in samples]
    else:
        eb = [make_eval_sample(fast_process(s.raw, s.task_type, harness, store=None)[1], s) for s in samples]
    return get_evaluator(task).evaluate(eb, layer="grounded")


def _paired_cross_batch(bb, cell, h_ref, h_edit, bs, max_b=4):
    """每个不相交 batch 上的配对 Δ = val(h_ref) − val(h_edit)（同 batch）。返回 [Δ_b]。"""
    pool = bb.pools[cell]
    n_b = min(max_b, len(pool) // bs)
    if n_b < 2:
        return []
    out = []
    for i in range(n_b):
        b = pool[i * bs:(i + 1) * bs]
        va, ve = _val(b, h_ref), _val(b, h_edit)
        if np.isfinite(va) and np.isfinite(ve):
            out.append(va - ve)
    return out


def _unpaired_p90(bb, cell, h, bs, max_b=4):
    """（参照，过估）固定 harness 跨 batch 无配对 |Δval| 的 p90 —— 含横截面异质，accept 律里会抵消。"""
    pool = bb.pools[cell]
    n_b = min(max_b, len(pool) // bs)
    if n_b < 2:
        return float("nan")
    vals = [_val(pool[i * bs:(i + 1) * bs], h) for i in range(n_b)]
    vals = [v for v in vals if np.isfinite(v)]
    d = [abs(a - b) for a, b in itertools.combinations(vals, 2)]
    return float(np.percentile(d, 90)) if d else float("nan")


def main():
    ap = argparse.ArgumentParser(description="ε 配对标定（同-batch cur vs cand 的泛化噪声）")
    ap.add_argument("--npz", default=None)
    ap.add_argument("--task", default="forecast")
    ap.add_argument("--substrates", default="chronos,frozen")
    ap.add_argument("--n-per-signal", type=int, default=8)
    args = ap.parse_args()

    sigs = load_signals(args.npz)
    bb = BatchBuilder(HarnessState.from_minimal(), n_min=N_MIN)
    for rs in build_real_corpus(sigs, n_per_signal=args.n_per_signal, tasks=(args.task,)):
        bb.add_raw_series(rs)
    minimal, degraded = HarnessState.from_minimal(), degraded_harness()
    cells = sorted(c for c in bb.pools if c.startswith(args.task + "|"))

    # 对比集合：(名, h_ref, h_edit)。Δ>0 = h_edit 比 h_ref 更优。
    pairs = [("raw−minimal (未处理 vs 处理)", None, minimal),
             ("minimal−degraded (cur vs cand)", minimal, degraded)]

    for sub in (s.strip() for s in args.substrates.split(",") if s.strip()):
        set_forecast_substrate("chronos" if sub == "chronos" else "frozen")
        print(f"\n================ substrate={sub} ================")
        for pname, h_ref, h_edit in pairs:
            print(f"\n--- 配对 Δ = {pname}  (同 batch, 确定性) ---")
            print(f"  {'cell':30s}{'effect mean_Δ':>16s}{'σ_Δ(跨batch)':>16s}   ε_cell≳σ_Δ")
            sigmas = []
            for c in cells:
                ds = _paired_cross_batch(bb, c, h_ref, h_edit, N_MIN)
                if len(ds) < 2:
                    continue
                m, sd = float(np.mean(ds)), float(np.std(ds))
                sigmas.append(sd)
                print(f"  {c:30s}{m:>+16.3f}{sd:>16.3f}   ε_cell≈{sd:.3f}")
            if sigmas:
                rec = float(np.median(sigmas))
                print(f"  → 跨 cell σ_Δ: median={rec:.3f} max={max(sigmas):.3f}  **推荐 ε≈{rec:.3f}（中位 σ_Δ）**")

        # 参照：无配对过估
        up = [v for c in cells for v in [_unpaired_p90(bb, c, minimal, N_MIN)] if np.isfinite(v)]
        if up:
            print(f"\n  [参照] 无配对 batch-spread p90 中位={np.median(up):.3f}（含横截面异质 → 过估，accept 律里抵消）")

    print("\n[结论] ε 用**配对 σ_Δ**（同 batch cur−cand 的跨 batch 泛化噪声），非无配对 batch 方差。"
          " 把推荐值填 config/thresholds 或 run_real_longrun --eps。")


if __name__ == "__main__":
    main()
