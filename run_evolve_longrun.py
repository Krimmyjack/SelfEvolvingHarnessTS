"""真实 DeepSeek 多 cell × 多 epoch evolve 长跑（观察用，非单测）。

起点 = 降级 harness（关掉 forecast 离群算子，制造头部空间），喂 forecast(P1/P2/P3)+anomaly(P1/P2)
多 cell 数据，用真 LLM proposer(flash,K=3) 跑 3 epoch round-robin，报告：accept 率 / 编辑类型 /
冻结 / Pareto / 成本，并打印最终 harness 相对降级基线的变化。

运行：  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.run_evolve_longrun
"""
from __future__ import annotations

import time
from collections import Counter

from .harness import HarnessState, EditPatch, Manifest
from .slow_path import BatchBuilder, Evolver, Proposer
from .data import make_forecast_batch, make_anomaly_batch
from .llm import get_client

_DEGRADE_OPS = ["winsorize", "outlier_iqr", "outlier_mad"]
N_MIN = 6
N_EPOCHS = 3


def degraded_harness() -> HarnessState:
    h = HarnessState.from_minimal()
    for op in _DEGRADE_OPS:
        h.apply_edit(EditPatch("L2", "set", f"l2.active_operators.{op}", False, Manifest("seed_degrade")))
    return h


def build_bb(h) -> BatchBuilder:
    bb = BatchBuilder(h, n_min=N_MIN)
    # 网格预设 → 跨 SNR×missing 多 cell（4 forecast + 2 anomaly）
    for pat in ("G_hi_full", "G_hi_miss", "G_lo_full", "G_lo_miss"):
        for rs in make_forecast_batch(pat, 2 * N_MIN, seed0=0):
            bb.add_raw_series(rs)
    for pat in ("G_hi_miss", "G_lo_miss"):
        for rs in make_anomaly_batch(pat, 2 * N_MIN, seed0=100):
            bb.add_raw_series(rs)
    return bb


def _layer_of(path: str) -> str:
    return path.split(".", 1)[0].split("::", 1)[0]


def main():
    # 从 minimal 起：heuristic 各 forecast cell 都用 winsorize(削峰)，cell-scoped 模板可逐 cell 修复（C1 机会）
    h = HarnessState.from_minimal()
    base_active = dict(h.l2.active_operators)
    bb = build_bb(h)
    cells = bb.triggerable_cells()
    print(f"== minimal baseline v{h.version} ==")
    print(f"== {len(cells)} triggerable cells ==")
    for c in cells:
        print(f"   {c}  ({len(bb.pools[c])} samples)")

    client = get_client("flash", temperature=0.7, cache_name="longrun")
    ev = Evolver(h, bb, Proposer(llm=client, k=3))

    t0 = time.time()
    summary = ev.run(n_epochs=N_EPOCHS)
    dt = time.time() - t0

    # ── 逐轮 ──
    print("\n== per-round trace ==")
    rounds_with_accept = 0
    accepted_paths, proposed_paths, pareto_viol = [], [], 0
    for rr in ev.history:
        tag = "ACCEPT" if rr.n_accepted else "······"
        print(f"  [{tag}] {rr.cell_id:30s} ep{rr.epoch} r{rr.round_idx} budget={rr.budget} "
              f"proposed={rr.n_proposed} accepted={rr.n_accepted}")
        for rs in rr.reasons:
            print(f"           - {rs}")
            proposed_paths.append(rs.split(":", 1)[0])
            if "pareto_violation" in rs:
                pareto_viol += 1
        if rr.n_accepted:
            rounds_with_accept += 1
            accepted_paths.extend(rr.accepted_paths)

    # ── 汇总 ──
    n_rounds = len(ev.history)
    print("\n== summary ==")
    print(f"  rounds            : {n_rounds}   (with >=1 accept: {rounds_with_accept}, "
          f"rate={rounds_with_accept / max(1, n_rounds):.0%})")
    print(f"  total accepts     : {summary['total_accepts']}")
    print(f"  accepted by layer : {dict(Counter(_layer_of(p) for p in accepted_paths))}")
    print(f"  accepted paths    : {dict(Counter(accepted_paths))}")
    print(f"  proposed by layer : {dict(Counter(_layer_of(p) for p in proposed_paths))}")
    print(f"  pareto violations : {pareto_viol}")
    print(f"  frozen cells      : {summary['frozen']}")
    print(f"  final version     : v{summary['final_version']}  (patch_log={len(h.patch_log)})")
    print(f"  LLM               : {client.stats()}")
    print(f"  wall time         : {dt:.1f}s")

    # ── 最终 harness 变化 ──
    print("\n== final harness changes vs minimal baseline ==")
    changed = {k: (base_active[k], v) for k, v in h.l2.active_operators.items() if base_active.get(k) != v}
    print(f"  active_operators changes : {changed if changed else 'none'}")
    print(f"  cell-scoped templates    : {len(h.l2.task_templates)}")
    for name, t in h.l2.task_templates.items():
        bin_ = t.applies_to.get("pattern_conditions", {}).get("pattern_bin", "GLOBAL")
        ops = [op for st in t.stages for op in st.preferred_ops]
        bans = [op for st in t.stages for op in st.banned_ops]
        print(f"     {name}: cell={bin_}  prefer={ops}  ban={bans}")

    # ── 学到的 per-cell 算子归因（OPD 公式11）──
    print("\n== learned operator attribution (per cell) ==")
    for c in cells:
        summ = ev.attribution.summary(c)
        if summ["prefer"] or summ["avoid"]:
            print(f"  {c}:\n     prefer={[(op, v) for op, v, n in summ['prefer']]}\n"
                  f"     avoid ={[(op, v) for op, v, n in summ['avoid']]}")


if __name__ == "__main__":
    main()
