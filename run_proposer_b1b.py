"""run_proposer_b1b.py — Track B1b-mini：开放程序 proposer 竞技场（压缩计划 §2）。

**唯一主要开销**：新程序不在缓存 L_test → 须真实执行（ActionCompiler.to_harness → fast_process →
FrozenProbe OOF nRMSE，与池动作 L_test 同一评估口径，只是 frozen-probe 而非全 DLinear → 可承受）。

架构：
  执行器（贵，每不同程序一次）：program → {uid: oof_loss} 全语料，**按程序 SHA 落盘缓存**（三臂/重跑免费）。
  gym（便宜，在已执行损失上 cache-replay）：LOFO select→heldout，三臂 vs frozen，复用 run_proposer 逻辑。

三臂（压缩计划）：random（负对照）/ det budgeted search / LLM(gpt-5.4-mini)+memory。
天花板模型 = gpt-5.4-mini（用户 2026-07-07 拍板；**败非决定性**=非最强模型→触发升级，**胜=强正**）。
执行规模 = 全 8 族 LOFO，N=10+（用户拍板"更大更稳"）。

本文件分阶段建：`--proof` 先证一条**新程序**端到端执行出真损失（架构 de-risk），gym/正式跑随后。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_proposer_b1b --proof
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .policy.action_spec import ActionCompiler
from .policy.program_edit import ProgramSpec, is_novel, to_action_spec, validate

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "ProposerB1b"


# ════════════════════════════ 执行器 ════════════════════════════
def grouped_folds(uids: List[str], kfold: int = 5) -> Dict[str, int]:
    """按 uid 分组的确定性 K 折（round-robin over sorted uid）——池动作与新程序共用同一折 → 可比。"""
    return {u: i % kfold for i, u in enumerate(sorted(uids))}


def execute_action_losses(action_spec, series, fp, fold_of: Dict[str, int], kfold: int = 5) -> Dict[str, float]:
    """任一 ActionSpec（池动作或新程序）→ {uid: OOF nRMSE}，与 L_test 同评估器（FrozenProbe + grouped OOF）。"""
    from .run_variance_decomp import build_cell_cache, _oof_losses
    harness = ActionCompiler().to_harness(action_spec, "forecast")
    caches, common, _ = build_cell_cache(fp, series, {action_spec.action_id: harness})
    order = sorted(common)
    fo = {u: fold_of[u] for u in order if u in fold_of}
    return _oof_losses(caches[action_spec.action_id], order, fo, kfold)


# ════════════════════════════ 架构 de-risk 证明 ════════════════════════════
def proof(n_series: int = 90):
    """证一条 NOVEL 程序端到端执行出真损失（对照一条池动作，同评估器同折）。"""
    from .evaluators.frozen_probe import FrozenProbe
    from .run_main_table import _VARIANT_SPECS
    from .policy.action_spec import action_menu_v1
    from .s2_corpus import build_s2_dev
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    corpus_full = build_s2_dev()
    # 取子集（每族均匀）保证 proof 快；正式 gym 用全语料
    by_fam: Dict[str, list] = {}
    for rs in corpus_full:
        by_fam.setdefault(rs.origin, []).append(rs)
    per = max(2, n_series // max(1, len(by_fam)))
    series = [rs for fam in by_fam.values() for rs in sorted(fam, key=lambda r: r.series_uid)[:per]]
    print(f"proof 子集：{len(series)} series / {len(by_fam)} 族  [{time.time()-t0:.0f}s]", flush=True)

    fp = FrozenProbe()
    fold_of = grouped_folds([rs.series_uid for rs in series])

    # NOVEL 程序：impute_linear → winsorize → denoise_median(w9)
    #   （v_winsor_savgol = winsor→savgol；winsor→median 是新链，且带窗剂量 → 不在 15 menu/10 池）
    novel = ProgramSpec(
        steps=(("impute_linear", ()), ("winsorize", ()), ("denoise_median", (("window", 9),))),
        scope=("forecast|snrLow|full",), provenance={"source": "proof"})
    ok, why = validate(novel)
    assert ok, f"novel 程序未过 gate: {why}"
    assert is_novel(novel), "novel 程序竟等于已有动作——换一条"
    nspec = to_action_spec(novel)

    # 对照池动作 v_winsor_savgol（=impute_linear→winsorize→denoise_savgol）
    pool_id = "v_winsor_savgol"
    pspec = action_menu_v1().actions[pool_id]

    print(f"NOVEL action_id={nspec.action_id}  steps={[(s.op, dict(s.params)) for s in nspec.steps]}", flush=True)
    nl = execute_action_losses(nspec, series, fp, fold_of)
    pl = execute_action_losses(pspec, series, fp, fold_of)
    common = sorted(set(nl) & set(pl))
    nv = np.array([nl[u] for u in common])
    pv = np.array([pl[u] for u in common])

    print(f"\n── 开放程序执行 weld 证明 [{time.time()-t0:.0f}s] ──", flush=True)
    print(f"  执行成功 uid：novel={len(nl)}  pool({pool_id})={len(pl)}  common={len(common)}", flush=True)
    print(f"  novel  OOF nRMSE  mean={nv.mean():.4f}  [min {nv.min():.3f}, max {nv.max():.3f}]  finite={np.isfinite(nv).all()}", flush=True)
    print(f"  {pool_id:16s} OOF nRMSE  mean={pv.mean():.4f}  [min {pv.min():.3f}, max {pv.max():.3f}]", flush=True)
    diff = nv - pv
    print(f"  逐序列差 novel−pool：mean={diff.mean():+.4f}  #(novel更好)={int((diff<0).sum())}/{len(common)}", flush=True)
    sane = bool(np.isfinite(nv).all() and np.isfinite(pv).all()
                and 0.0 < nv.mean() < 10.0 and not np.allclose(nv, pv))
    print(f"\n  {'WELD OK' if sane else 'WELD FAIL'} — 新算子链{'编译→执行→出真损失，与池动作同评估器可比，且损失确不同' if sane else '异常'}。", flush=True)
    print(f"产物就绪路径：{OUT}  [{time.time()-t0:.0f}s]", flush=True)
    return sane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proof", action="store_true")
    ap.add_argument("--n-series", type=int, default=90)
    args = ap.parse_args()
    if args.proof:
        proof(args.n_series)
    else:
        print("用 --proof（架构 de-risk）；gym/正式跑随后建")


if __name__ == "__main__":
    main()
