"""nested_supply.py — E-3.3 真 nested held-out Δ_supply 评估器（A-31a，评审第十一轮）。

背景：`run_variance_decomp._oof_losses` 的每个 OOF loss 出自"在全部其他 fold uid 上训练的
Ridge 头"——事后按 outer fold 重切时，动作选择与 test 评估都间接见过 outer-test 分布
（A-10/A-23 同一原则：evaluator head 会拟合 → 统计重切不能替代重拟合）。本模块实现正确程序：

  每 outer fold（grouped by series_uid，两池共用同一 folds）：
    1) inner grouped CV（只在 outer-train 内）估各动作效用 → 选 pool 内 cell-best 动作；
    2) 该动作的 Ridge 头在 **outer-train 全体** 上重拟合；
    3) 仅在 outer-test uid 上评估；
  Δ_supply = loss(base 池 train-选择) − loss(expanded 池 train-选择)，逐 uid paired，
  uid 组 bootstrap CI。>0 → 扩充池在 held-out 上真的更好。

输入是 uid×action 的特征缓存（`run_variance_decomp._cache_one` 的字段：PhiX/Y/PhiTest/
future/obs）——**只缓存最终 loss 不够**（A-31a）。本模块无 torch 依赖（编码器特征已缓存），
单 cell 语义；多 cell 聚合（cell 等权 + 按 cell 分层 bootstrap）由 E-3.3 run 脚本做。

防泄漏守卫（tests/test_nested_supply.py，全过才允许正式 E-3.3）：
  ①每 outer fold 的 fit uid 与 test uid 严格不交且覆盖全体；②inner 选择只读 outer-train
  （扰动 outer-test 的 future/PhiTest 不改变所选动作）；③两池 outer folds 完全一致；
  ④纯噪声/重复动作扩充不产生稳定正 Δ_supply；⑤真实更优动作可被检出（功效对照）。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge

DEFAULT_SEED = 20260703


def make_folds(uids: Sequence[str], k: int, seed: int = DEFAULT_SEED) -> Dict[str, int]:
    """确定性 grouped folds（uid=组；同 (uids,k,seed) → 同划分，两池共用的前提）。"""
    order = sorted(uids)
    rng = np.random.default_rng(seed)
    order = [order[i] for i in rng.permutation(len(order))]
    return {u: i % k for i, u in enumerate(order)}


def _fit_head(caches: Dict[str, dict], train_uids: Sequence[str]) -> Ridge:
    PhiX = np.concatenate([caches[u]["PhiX"] for u in train_uids])
    Y = np.concatenate([caches[u]["Y"] for u in train_uids])
    return Ridge(alpha=1.0).fit(PhiX, Y)


def _eval_uids(head: Ridge, caches: Dict[str, dict], uids: Sequence[str]) -> Dict[str, float]:
    PhiTest = np.vstack([caches[u]["PhiTest"] for u in uids])
    preds = head.predict(PhiTest)
    H = preds.shape[1]
    futs = np.vstack([caches[u]["future"][:H] for u in uids])
    obs = np.array([caches[u]["obs"] for u in uids])
    rmse = np.sqrt(np.mean((preds - futs) ** 2, axis=1)) / obs
    return {u: float(r) for u, r in zip(uids, rmse)}


def inner_select(action_caches: Dict[str, Dict[str, dict]], pool: Sequence[str],
                 train_uids: Sequence[str], inner_k: int, seed: int) -> Tuple[str, Dict[str, float]]:
    """outer-train 内 inner grouped CV（每 inner fold 重拟合头）→ 选 pool 内均值最优动作。
    只接触 train_uids 的缓存——outer-test 不可见（防泄漏守卫②的被测对象）。"""
    fold_of = make_folds(train_uids, inner_k, seed)
    means: Dict[str, float] = {}
    for a in pool:
        losses: Dict[str, float] = {}
        for f in range(inner_k):
            tr = [u for u in train_uids if fold_of[u] != f]
            te = [u for u in train_uids if fold_of[u] == f]
            if not tr or not te:
                continue
            head = _fit_head(action_caches[a], tr)
            losses.update(_eval_uids(head, action_caches[a], te))
        means[a] = float(np.mean([v for v in losses.values()])) if losses else float("inf")
    best = min(means, key=means.get)
    return best, means


def nested_pool_losses(action_caches: Dict[str, Dict[str, dict]], pool: Sequence[str],
                       uids: Sequence[str], outer_k: int = 5, inner_k: int = 4,
                       seed: int = DEFAULT_SEED) -> Tuple[Dict[str, float], List[dict]]:
    """整套 nested：返回 ({uid: 该池 train-选择动作的 outer-test loss}, 逐 fold 选择台账)。
    台账含 train/test uid 名单（防泄漏测试直接断言不交/覆盖）。"""
    outer = make_folds(uids, outer_k, seed)
    out: Dict[str, float] = {}
    picks: List[dict] = []
    for f in range(outer_k):
        tr = [u for u in uids if outer[u] != f]
        te = [u for u in uids if outer[u] == f]
        if not tr or not te:
            continue
        a_star, means = inner_select(action_caches, pool, tr, inner_k, seed + 7919 * (f + 1))
        head = _fit_head(action_caches[a_star], tr)        # outer-train 全体重拟合
        out.update(_eval_uids(head, action_caches[a_star], te))
        picks.append({"outer_fold": f, "selected": a_star, "inner_means": means,
                      "train_uids": list(tr), "test_uids": list(te)})
    return out, picks


def delta_supply(action_caches: Dict[str, Dict[str, dict]], base_pool: Sequence[str],
                 expanded_pool: Sequence[str], uids: Sequence[str], outer_k: int = 5,
                 inner_k: int = 4, seed: int = DEFAULT_SEED, n_boot: int = 1000) -> dict:
    """单 cell Δ_supply = loss(base 选择) − loss(expanded 选择)（>0 → 扩充池 held-out 更好）。

    同 seed → `make_folds` 确定性 → 两池 outer folds 严格一致（paired）。CI = uid 组
    bootstrap（test loss 上重采样；train 侧的头拟合方差由 outer folds 平均覆盖）。
    """
    lb, picks_b = nested_pool_losses(action_caches, base_pool, uids, outer_k, inner_k, seed)
    le, picks_e = nested_pool_losses(action_caches, expanded_pool, uids, outer_k, inner_k, seed)
    common = [u for u in uids if u in lb and u in le]
    d = np.array([lb[u] - le[u] for u in common])
    rng = np.random.default_rng(seed + 999)
    boots = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta_mean": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(len(common)), "loss_base": float(np.mean([lb[u] for u in common])),
            "loss_expanded": float(np.mean([le[u] for u in common])),
            "per_uid": {u: float(v) for u, v in zip(common, d)},
            "picks_base": picks_b, "picks_expanded": picks_e,
            "method": "single_nested_test_uid_bootstrap"}


# ══════════════════════════════════════════════════════════════════════════
# A-33c：full-refit group bootstrap Δ_supply（评审第十二轮后续意见）
# ──────────────────────────────────────────────────────────────────────────
# `delta_supply` 的 CI 只对**一次** nested run 的 per-uid 差值重采样——inner 动作选择、
# Ridge 头、outer folds 在所有 replicate 上被冻结，故仅覆盖 test-uid 抽样，低估了
#   ①inner 选择不稳定性 ②头重拟合方差 ③fold 划分方差 ④同 fold 内 uid 共享头的相关。
# 正式判决 CI 用本函数：每 replicate 对 **uid 做组重采样**（bootstrap 多重集），
# **按 uid 身份分 fold**（同一 uid 全部副本入同折 → 无 train/test 泄漏），重数经
# **Ridge sample_weight** 进入头拟合、经加权均值进入评估（训练分布≡评价分布，A-28b），
# 且每 replicate 用不同 fold seed（覆盖③）→ 完整重跑 inner 选择+outer 重拟合。
# Ridge 便宜：B=300 smoke / B=1000 final。
# ══════════════════════════════════════════════════════════════════════════
def _fit_head_w(caches: Dict[str, dict], uids: Sequence[str], mult: Dict[str, float]) -> Ridge:
    PhiX = np.concatenate([caches[u]["PhiX"] for u in uids])
    Y = np.concatenate([caches[u]["Y"] for u in uids])
    sw = np.concatenate([np.full(len(caches[u]["PhiX"]), mult[u], float) for u in uids])
    return Ridge(alpha=1.0).fit(PhiX, Y, sample_weight=sw)


def _eval_wmean(head: Ridge, caches: Dict[str, dict], uids: Sequence[str],
                mult: Dict[str, float]) -> float:
    """uids 上的**重数加权** nRMSE 均值（每 uid 计 mult 次）。"""
    PhiTest = np.vstack([caches[u]["PhiTest"] for u in uids])
    preds = head.predict(PhiTest)
    H = preds.shape[1]
    futs = np.vstack([caches[u]["future"][:H] for u in uids])
    obs = np.array([caches[u]["obs"] for u in uids])
    rmse = np.sqrt(np.mean((preds - futs) ** 2, axis=1)) / obs
    w = np.array([mult[u] for u in uids], float)
    return float(np.sum(w * rmse) / np.sum(w))


def _inner_select_w(action_caches: Dict[str, Dict[str, dict]], pool: Sequence[str],
                    train_uids: Sequence[str], mult: Dict[str, float], inner_k: int,
                    seed: int) -> str:
    fold_of = make_folds(train_uids, inner_k, seed)          # 按身份 → 同 uid 不跨 inner train/test
    best, best_loss = None, float("inf")
    for a in pool:
        num, den = 0.0, 0.0
        for f in range(inner_k):
            tr = [u for u in train_uids if fold_of[u] != f]
            te = [u for u in train_uids if fold_of[u] == f]
            if not tr or not te:
                continue
            head = _fit_head_w(action_caches[a], tr, mult)
            wsub = float(sum(mult[u] for u in te))
            num += _eval_wmean(head, action_caches[a], te, mult) * wsub
            den += wsub
        loss = num / den if den > 0 else float("inf")
        if loss < best_loss:
            best, best_loss = a, loss
    return best


def _nested_loss_w(action_caches: Dict[str, Dict[str, dict]], pool: Sequence[str],
                   uids: Sequence[str], mult: Dict[str, float], outer_k: int, inner_k: int,
                   seed: int) -> Tuple[float, List[str]]:
    outer = make_folds(uids, outer_k, seed)
    num, den = 0.0, 0.0
    picks: List[str] = []
    for f in range(outer_k):
        tr = [u for u in uids if outer[u] != f]
        te = [u for u in uids if outer[u] == f]
        if not tr or not te:
            continue
        a_star = _inner_select_w(action_caches, pool, tr, mult, inner_k, seed + 7919 * (f + 1))
        head = _fit_head_w(action_caches[a_star], tr, mult)   # outer-train 全体（重数加权）重拟合
        wsub = float(sum(mult[u] for u in te))
        num += _eval_wmean(head, action_caches[a_star], te, mult) * wsub
        den += wsub
        picks.append(a_star)
    return (num / den if den > 0 else float("inf")), picks


def delta_supply_grouped(action_caches: Dict[str, Dict[str, dict]], base_pool: Sequence[str],
                         expanded_pool: Sequence[str], uids: Sequence[str], outer_k: int = 5,
                         inner_k: int = 4, seed: int = DEFAULT_SEED, n_boot: int = 300,
                         progress: int = 0, ckpt_path=None, ckpt_every: int = 25) -> dict:
    """full-refit group bootstrap Δ_supply（A-33c，正式判决 CI）。

    每 replicate：对 uid 组重采样（多重集）→ 按身份分 fold（防泄漏）→ 两池共用同一多重集+fold
    seed（paired）→ 各自完整重跑 nested（inner 选择重数加权、outer 重拟合、加权评估）→
    Δ_b = loss_base_b − loss_expanded_b。CI = {Δ_b} 的 [2.5,97.5] 分位。

    返回含 **boot_deltas**（逐 replicate Δ 全量数组）——因各 cell 独立（clean 种子含 dname、
    互不共享 → run_family0_final 已核），cell 等权 aggregate 分布 = 各 cell boot_deltas 的卷积
    （按 replicate 配对求 cell 均值）；无需重跑单一 joint pass（A-34）。`progress>0` 时每该
    步数打印一次心跳（长跑可监控，防误判为 hang）。

    **per-replicate 独立种子**（`seed+4242+7907*(b+1)`）：replicate b 的重采样只依赖 b，不依赖
    前序 draw → 可任意子集/断点续跑复现（A-36）。**checkpoint/resume**：给 `ckpt_path` 时每
    `ckpt_every` 个 replicate 落盘 {done, deltas, lbs, les, pick_counts, n_folds_total}；重启若
    ckpt 存在则从 done 续跑（后台任务 ~100min 墙钟寿命上限下的鲁棒化，被杀最多丢 ckpt_every 个）。
    """
    import json as _json
    import time as _t
    from pathlib import Path as _Path
    uids = sorted(uids)
    N = len(uids)
    deltas, lbs, les = [], [], []
    pick_counts: Counter = Counter()
    n_folds_total = 0
    start = 0
    ckpt = _Path(ckpt_path) if ckpt_path else None
    if ckpt and ckpt.exists():
        try:
            st = _json.loads(ckpt.read_text(encoding="utf-8"))
            if st.get("seed") == seed and st.get("n_uid") == N:      # 同实验才续（防错配）
                deltas, lbs, les = st["deltas"], st["lbs"], st["les"]
                pick_counts = Counter(st["pick_counts"]); n_folds_total = st["n_folds_total"]
                start = st["done"]
                print(f"        [resume] 从 checkpoint 续跑：已完成 {start}/{n_boot}", flush=True)
        except Exception as _e:
            print(f"        [resume] checkpoint 读取失败（重头跑）: {_e}", flush=True)

    def _save_ckpt(done):
        if not ckpt:
            return
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(_json.dumps(dict(done=done, seed=seed, n_uid=N, deltas=deltas, lbs=lbs,
                       les=les, pick_counts=dict(pick_counts), n_folds_total=n_folds_total)),
                       encoding="utf-8")
        tmp.replace(ckpt)                                            # 原子替换（防写一半被杀）

    t0 = _t.time()
    for b in range(start, n_boot):
        rng_b = np.random.default_rng(seed + 4242 + 7907 * (b + 1))  # per-replicate 独立 → 可续跑复现
        samp = [uids[i] for i in rng_b.integers(0, N, N)]
        mult = Counter(samp)
        distinct = sorted(mult)
        bseed = seed + 100003 * (b + 1)                       # 每 replicate 变 fold seed（覆盖 fold 方差）
        lb, _ = _nested_loss_w(action_caches, base_pool, distinct, mult, outer_k, inner_k, bseed)
        le, pe = _nested_loss_w(action_caches, expanded_pool, distinct, mult, outer_k, inner_k, bseed)
        deltas.append(lb - le); lbs.append(lb); les.append(le)
        for p in pe:
            pick_counts[p] += 1
        n_folds_total += len(pe)
        if ckpt and (b + 1) % ckpt_every == 0:
            _save_ckpt(b + 1)
        if progress and (b + 1) % progress == 0:
            print(f"        [grouped boot] {b+1}/{n_boot}  Δ_mean_sofar={np.mean(deltas):+.4f}  "
                  f"[{_t.time()-t0:.0f}s]", flush=True)
    _save_ckpt(n_boot)
    deltas = np.array(deltas)
    denom = max(n_folds_total, 1)
    return {"delta_mean": float(deltas.mean()),
            "ci_lo": float(np.percentile(deltas, 2.5)), "ci_hi": float(np.percentile(deltas, 97.5)),
            "median": float(np.median(deltas)), "n_boot": int(n_boot), "n_uid": int(N),
            "loss_base": float(np.mean(lbs)), "loss_expanded": float(np.mean(les)),
            "frac_boot_positive": float(np.mean(deltas > 0)),
            "expanded_pick_freq": {k: pick_counts[k] / denom for k in sorted(pick_counts)},
            "boot_deltas": [float(x) for x in deltas],
            "method": "grouped_full_refit_bootstrap"}
