"""run_block_gate.py — 块级 population 新颖性门：基建验证（非 updater v4 战役，无六门）。

背景（updater v3 FAIL 的正面副产物）：机制身份信息在 P0 特征空间（族 1-NN 0.896）而
per-uid kNN p95 团块语义用不上它——新族每个点都靠近某已见点，但**整块**（~42 uid 族同质）
在特征空间成新簇。块级两样本检验有 √n 功效增益。

门的双重角色（第三十二轮定位）：①确定性系统修 c2 的候选机制（首块新区域整块回退 frozen）；
②LLM 升级式架构（四臂 D）的触发器——"何时值得花 Agent 推理"。本脚本只验证**检测质量**：
  recall@first-encounter 块 / false-positive@recurrence 块 / per-family（尤其 S_both aliasing）
  + 反事实：v2 账本上被门覆盖的首遇 harm 份额、v2+gate 的 c2/c1 估计（静态重放近似，声明）。

检测器（无标签输入；族标签仅评估）：
  空间 = v2 同款 P0 特征 [SNR, missing_rate, X_p(8)]，z-score 由已见集定；
  统计量 = 块内各点到已见集的 NN 距离均值；
  校准 = 置换零分布（从已见集抽同尺寸子集、NN 到补集）——**无阈值旋钮**，
  预锁 α=0.01、n_perm=499、seed=20260706。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_block_gate
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .run_updater2 import DELTA_SAFE, OUT as OUT2, REC_PATH, half_of, locked_permutations

OUT = Path(__file__).resolve().parent / "results" / "Stage2" / "BlockGate"
ALPHA = 0.01
N_PERM_NULL = 499
GATE_SEED = 20260706


def p0_features(r: dict) -> List[float]:
    return [r["snr"], r["miss_rate"], *r["X_p"]]


def block_novelty_p(seen_X: np.ndarray, block_X: np.ndarray, seed: int) -> Tuple[float, float]:
    """块级两样本检验 → (obs_stat, p)。无标签输入；确定性（seed 显式）。

    stat = 块内点到已见集的 NN 距离均值（z 空间，z 由已见集定）。
    null = 从已见集不放回抽 |block| 点、NN 到其补集，重复 N_PERM_NULL 次。
    """
    mu, sd = seen_X.mean(axis=0), seen_X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Zs = (seen_X - mu) / sd
    Zb = (block_X - mu) / sd
    obs = float(np.sqrt(((Zb[:, None, :] - Zs[None, :, :]) ** 2).sum(-1)).min(axis=1).mean())
    n, m = len(Zs), len(Zb)
    if n <= m + 5:
        return obs, 1.0                                       # 已见太少，不可校准 → 不触发（保守）
    D = np.sqrt(((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERM_NULL)
    idx_all = np.arange(n)
    for k in range(N_PERM_NULL):
        S = rng.choice(n, size=m, replace=False)
        comp = np.setdiff1d(idx_all, S, assume_unique=False)
        null[k] = D[np.ix_(S, comp)].min(axis=1).mean()
    p = float((1 + np.sum(null >= obs)) / (N_PERM_NULL + 1))
    return obs, p


def main():
    t0 = time.time()
    recs = [json.loads(l) for l in REC_PATH.read_text("utf-8").splitlines() if l.strip()]
    by_uid = {r["uid"]: r for r in recs}
    blocks_uids: Dict[Tuple[str, int], List[str]] = {}
    for r in recs:
        blocks_uids.setdefault((r["origin"], half_of(r["uid"])), []).append(r["uid"])
    for k in blocks_uids:
        blocks_uids[k] = sorted(blocks_uids[k])
    perms = locked_permutations()

    rows = []                                                # (perm, bi, family, first, p, flagged)
    for pi, order in enumerate(perms):
        seen: List[str] = []
        first_seen: set = set()
        for bi, (fam, hh) in enumerate(order):
            us = blocks_uids[(fam, hh)]
            if bi > 0:
                seen_X = np.array([p0_features(by_uid[u]) for u in seen], float)
                blk_X = np.array([p0_features(by_uid[u]) for u in us], float)
                obs, p = block_novelty_p(seen_X, blk_X, seed=GATE_SEED + 100 * pi + bi)
                rows.append(dict(perm=pi, block=bi, family=fam,
                                 first=fam not in first_seen, stat=round(obs, 4), p=p,
                                 flagged=bool(p <= ALPHA)))
            first_seen.add(fam)
            seen.extend(us)
        print(f"  [perm {pi}] 15 块检验完成 [{time.time()-t0:.0f}s]", flush=True)

    firsts = [r for r in rows if r["first"]]
    recurs = [r for r in rows if not r["first"]]
    per_family: Dict[str, dict] = {}
    for r in firsts:
        d = per_family.setdefault(r["family"], dict(n=0, caught=0))
        d["n"] += 1
        d["caught"] += int(r["flagged"])
    detection = dict(
        n_first=len(firsts), n_recurrence=len(recurs),
        recall_first=float(np.mean([r["flagged"] for r in firsts])) if firsts else None,
        fpr_recurrence=float(np.mean([r["flagged"] for r in recurs])) if recurs else None,
        per_family_first={k: f"{v['caught']}/{v['n']}" for k, v in sorted(per_family.items())})

    # —— 反事实（静态重放近似，声明：不改 seen/accept 动力学）——
    # v2 账本上：被门 flag 的块若整块服务 frozen → 该块 regret := frozen_regret。
    flag_of = {(r["perm"], r["block"]): r["flagged"] for r in rows}
    harm_total = harm_covered = 0.0
    fu_harm_new, cum_new, cum_old = [], [], []
    for pi in range(len(perms)):
        led = json.loads((OUT2 / "ckpt" / f"perm{pi}_updater_v2.json").read_text("utf-8"))["ledger"]
        regs = []
        fu = []
        for b in led:
            harm = b["regret"] - b["frozen_regret"]
            flagged = flag_of.get((pi, b["block"]), False)
            if b["first"] and harm > 0:
                harm_total += harm
                if flagged:
                    harm_covered += harm
            reg_cf = b["frozen_regret"] if flagged else b["regret"]
            regs.append(reg_cf)
            if b["first"]:
                fu.append(reg_cf - b["frozen_regret"])
        cum_new.append(float(np.mean(regs)))
        cum_old.append(float(np.mean([b["regret"] for b in led])))
        fu_harm_new.append(float(np.max(fu)) if fu else 0.0)
    counterfactual = dict(
        note="静态重放近似：只改被 flag 块的服务（→frozen），不改 seen/accept 动力学（声明）",
        first_harm_coverage=float(harm_covered / harm_total) if harm_total > 0 else None,
        v2_first_unseen_harm_max_mean_before=0.198,
        v2_first_unseen_harm_max_mean_after=float(np.mean(fu_harm_new)),
        v2_cum_before=float(np.mean(cum_old)), v2_cum_after=float(np.mean(cum_new)),
        frozen_cum=0.3677, delta_safe=DELTA_SAFE)

    res = dict(config=dict(alpha=ALPHA, n_perm_null=N_PERM_NULL, seed=GATE_SEED,
                           space="P0 [SNR, missing_rate, X_p]（v2 支持域同款）",
                           role="基建验证（c2 修复候选 + LLM 升级触发器）；族标签仅评估"),
               detection=detection, counterfactual=counterfactual, rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(dict(detection=detection, counterfactual=counterfactual),
                     ensure_ascii=False, indent=1), flush=True)
    print(f"产物：{OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
