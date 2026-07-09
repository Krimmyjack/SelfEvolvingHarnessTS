"""run_variance_decomp.py — E-1.1R 良定性 gate + degradation/pattern 分开估计（Stage 1 首实验，修正版）。

回答外部评审的三问（"small but hard" Stage 1）：
  Q1 不同 cell 的最优动作是否确实不同？        → 良定性 gate（D-1.1a/b）+ 三层效用阶梯 CI。
  Q2 差异是稳定结构还是噪声/近平局？          → group bootstrap（**bootstrap 内重拟合判官头**，A-10）winner 频率
                                                 + **固定 top1/top2 的配对 gap CI**（可为负）+ SVD bootstrap 稳定性（D-1.1c）。
  Q3 差异是否仅由 degradation severity 解释？   → 三层分解 L0 global → **L1 degradation-only router** → L2 structure oracle，
                                                 L0→L1 与 L1→L2 增益**分别给 CI**；cell_id 只按 SNR×missing（binning.py）
                                                 ⇒ cell oracle ≡ degradation router，pattern 价值只在 cell 内部按结构分层可见。

—— E-1.1R 相对 E-1.1（预实验）的修正（外部评审第六轮，全部锁定口径重跑）——
  (1) 配对 gap bootstrap：**固定原样本 top1/top2**，每 replicate 算 util[top2]−util[top1]（允许为负），
      不再每 replicate 重排——否则 "gap CI 下界>0" 机械满足。
  (2) **bootstrap 内重拟合**（A-10）：每 replicate 重采样 series_uid → 重拟合 Ridge 头（缓存编码器特征上）→ 重评。
  (3) **按 series_uid 交集对齐**所有动作列（弃 caches[:ref_len] 截断，防错行）。
  (4) 落**完整响应矩阵**（cell,uid,origin,snr,fold,action,oof_loss）。
  (5) L0→L1、L1→L2 增益**分别 bootstrap CI**（不再从 median gap 反推 L2）。
  (6) SVD **bootstrap 稳定性 + rank-1/2 重构误差**（不止点估计）。
  (7) structure oracle 明确标为**信息上界**（同数据内选择，乐观偏差；可实现增益须 E-3.2 held-out policy regret）。
  (8) **SNR 分层置换**敏感性：在 cell 内按 SNR 分层置换 origin 标签，检验结构×动作是否只是残差 SNR。

judge=frozen forecast substrate（in-loop 判官）；判定判官响应面良定性，用判官效用正确。产物 results/E1_1/。
运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.run_variance_decomp [--diagnose] [--n-seeds 12] [--boot 1000]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import Ridge

from .harness import HarnessState
from .data.synthetic_gen import RawSeries, LENGTH, H_FORECAST
from .fast_path.perceive import perceive
from .fast_path.pipeline import process as fast_process
from .evaluators.base import L_WIN
from .evaluators.frozen_probe import FrozenProbe
from .operators import _provenance as prov
from .evaluators.grounded_forecast import _build_windows_full
from .run_main_table import fixed_harness_variants

RESULTS = Path(__file__).resolve().parent / "results" / "E1_1"
CUT = LENGTH - H_FORECAST


# ══════════════════════════════════════════════════════════════════════════
# 1. 方差匹配结构族 × 退化网格（信号功率匹配 → SNR 可比 → 尽量落同 cell）
# ══════════════════════════════════════════════════════════════════════════
def _unit(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    return (x - np.mean(x)) / (s if s > 1e-9 else 1.0)


def _clean_signal(struct: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(LENGTH, dtype=float)
    if struct == "S_season":
        sig = np.sin(2 * np.pi * t / 24) + 0.3 * np.sin(2 * np.pi * t / 12)
    elif struct == "S_trend":
        sig = 0.02 * t
    elif struct == "S_both":
        sig = np.sin(2 * np.pi * t / 24) + 0.012 * t
    elif struct == "S_ar":
        e = rng.normal(0, 1, LENGTH)
        x = np.zeros(LENGTH)
        for i in range(1, LENGTH):
            x[i] = 0.7 * x[i - 1] + e[i]
        sig = x
    else:
        raise ValueError(struct)
    return _unit(sig)


STRUCTS = ("S_season", "S_trend", "S_both", "S_ar")
DEG_GRID = OrderedDict([
    ("d_hi_full", dict(noise=0.12, miss=0.00)),
    ("d_hi_miss", dict(noise=0.12, miss=0.06)),
    ("d_lo_full", dict(noise=0.55, miss=0.00)),
    ("d_lo_miss", dict(noise=0.55, miss=0.06)),
])
OUT_RATE = 0.02


def _degrade(clean, noise, miss, out_rate, seed):
    rng = np.random.default_rng(seed + 10_000)
    x = clean.astype(float).copy()
    n = x.size
    if noise > 0:
        x = x + rng.normal(0, noise, n)
    n_out = int(round(out_rate * n))
    if n_out > 0:
        idx = rng.choice(n, size=n_out, replace=False)
        x[idx] += rng.choice([-1.0, 1.0], size=n_out) * 5.0
    n_miss = int(round(miss * n))
    if n_miss > 0:
        x[rng.choice(n, size=n_miss, replace=False)] = np.nan
    return x


def _det_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def build_corpus(n_seeds: int) -> List[RawSeries]:
    out = []
    for struct in STRUCTS:
        for dname, dp in DEG_GRID.items():
            for j in range(n_seeds):
                sd = _det_seed(struct, dname, j) % 2_000_000
                clean = _clean_signal(struct, sd)
                degraded = _degrade(clean, dp["noise"], dp["miss"], OUT_RATE, sd)
                out.append(RawSeries(
                    pattern=struct, task="forecast", seed=sd, period=24,
                    obs_scale=float(np.std(clean[CUT:])) or 1.0,
                    clean=clean, degraded=degraded,
                    history=degraded[:CUT].copy(), clean_history=clean[:CUT].copy(),
                    future=clean[CUT:].copy(),
                    origin=struct, series_uid=f"{struct}:{dname}:{j}"))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 2. 编码器特征缓存 + OOF（bootstrap 内可重拟合）
# ══════════════════════════════════════════════════════════════════════════
def _cache_one(fp: FrozenProbe, ready_hist, future, obs):
    hh = np.asarray(ready_hist, float).ravel()
    if not np.all(np.isfinite(hh)) or hh.size < L_WIN:
        return None
    X, Y, _ = _build_windows_full([hh], [24])
    if X is None or len(X) < 6:
        return None
    return dict(PhiX=fp.transform(X), Y=np.asarray(Y, float),
                PhiTest=fp.transform(hh[-L_WIN:].reshape(1, -1)),
                future=np.asarray(future, float), obs=float(obs) + 1e-9)


def _oof_losses(caches: Dict[str, dict], order_uids: List[str], fold_of: Dict[str, int],
                kfold: int, weight: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """grouped K-fold OOF（**每 fold 重拟合 Ridge 头**）：返回 {uid: OOF nRMSE}（仅 order_uids 中的 distinct uid）。

    A-28b：`weight`（uid→bootstrap 重数）非空时，训练窗按其 uid 的重数用 **Ridge sample_weight** 加权
    ——使训练分布与评价分布**同为** bootstrap 分布（否则训练只见 distinct uid、评价却带重数，头拟合方差被低估）。
    fold 仍按唯一 uid 分组（同一 uid 不跨 train/test → 无泄漏）。
    """
    out: Dict[str, float] = {}
    for f in range(kfold):
        tr = [u for u in order_uids if fold_of[u] != f]
        te = [u for u in order_uids if fold_of[u] == f]
        if not tr or not te:
            continue
        PhiX = np.concatenate([caches[u]["PhiX"] for u in tr])
        Y = np.concatenate([caches[u]["Y"] for u in tr])
        if weight is None:
            head = Ridge(alpha=1.0).fit(PhiX, Y)
        else:
            sw = np.concatenate([np.full(len(caches[u]["PhiX"]), weight[u], float) for u in tr])
            head = Ridge(alpha=1.0).fit(PhiX, Y, sample_weight=sw)
        # 向量化：一次预测全体 test uid 的末窗，批量算 per-uid nRMSE（等价旧逐条循环，仅去 Python 开销）
        PhiTest = np.vstack([caches[u]["PhiTest"] for u in te])   # (n_te, HIDDEN)
        preds = head.predict(PhiTest)                            # (n_te, H)
        H = preds.shape[1]
        futs = np.vstack([caches[u]["future"][:H] for u in te])  # (n_te, H)；future 定长 H_FORECAST
        obs = np.array([caches[u]["obs"] for u in te])
        rmse = np.sqrt(np.mean((preds - futs) ** 2, axis=1)) / obs
        for i, u in enumerate(te):
            out[u] = float(rmse[i])
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3. 点估计统计（响应矩阵 / SVD / near-tie / pattern 置换）
# ══════════════════════════════════════════════════════════════════════════
def _row_centered_svd(M: np.ndarray):
    R = np.nan_to_num(M - np.nanmean(M, axis=1, keepdims=True))
    s = np.linalg.svd(R, compute_uv=False)
    s2 = s ** 2
    tot = float(s2.sum()) or 1.0
    eff_rank = float((s2.sum() ** 2) / (np.sum(s2 ** 2) + 1e-12))
    return s2, tot, eff_rank


def svd_point(M: np.ndarray) -> dict:
    s2, tot, eff = _row_centered_svd(M)
    return dict(first_sv_var_ratio=float(s2[0] / tot), second_sv_var_ratio=float(s2[1] / tot),
                effective_rank=eff,
                recon_err_rank1=float(1 - s2[0] / tot),
                recon_err_rank2=float(1 - (s2[0] + s2[1]) / tot),
                sv_var_ratios=[float(v / tot) for v in s2[:5]])


def near_tie_fraction(M, tol=0.02):
    ties = 0
    for row in M:
        r = np.sort(row[np.isfinite(row)])
        if r.size >= 2 and (r[1] - r[0]) < tol:
            ties += 1
    return ties / M.shape[0] if M.shape[0] else float("nan")


def _struct_oracle(M: np.ndarray, org: np.ndarray, uniq) -> float:
    """L2：每 origin 各取 best action 的加权均值（同数据内选择 → **信息上界，乐观**）。"""
    num, den = 0.0, 0
    for u in uniq:
        sub = M[org == u]
        if sub.shape[0]:
            num += float(np.min(np.nanmean(sub, axis=0))) * sub.shape[0]
            den += sub.shape[0]
    return num / den if den else float(np.min(np.nanmean(M, axis=0)))


def pattern_point(M, origins, variants, snr, perm, rng) -> Optional[dict]:
    org = np.array(origins)
    uniq = sorted(set(origins))
    if len(uniq) < 2:
        return None
    l1 = float(np.min(np.nanmean(M, axis=0)))
    l2 = _struct_oracle(M, org, uniq)
    gap = l1 - l2                                              # ≥0（信息上界）

    def perm_p(permute_fn):
        cnt = 0
        for _ in range(perm):
            lbl = permute_fn()
            cnt += (l1 - _struct_oracle(M, lbl, uniq)) >= gap
        return float((cnt + 1) / (perm + 1))

    p_plain = perm_p(lambda: rng.permutation(org))
    # SNR 分层置换：按 SNR 三分位分层，只在层内打乱 origin → 控制 cell 内残差 SNR
    snr_arr = np.array(snr)
    qs = np.quantile(snr_arr, [1 / 3, 2 / 3])
    strat = np.digitize(snr_arr, qs)

    def strat_perm():
        lbl = org.copy()
        for s in np.unique(strat):
            idx = np.where(strat == s)[0]
            lbl[idx] = org[rng.permutation(idx)]
        return lbl
    p_strat = perm_p(strat_perm)

    # 两向交互方差占比
    Rc = np.nan_to_num(M - np.nanmean(M))
    resid = Rc - np.nanmean(Rc, axis=1, keepdims=True) - np.nanmean(Rc, axis=0, keepdims=True)
    inter = np.zeros_like(Rc)
    for u in uniq:
        m = org == u
        inter[m] = np.nanmean(resid[m], axis=0, keepdims=True)
    inter_ratio = float(np.sum(inter ** 2) / (float(np.sum(resid ** 2)) or 1.0))

    snr_by_origin = {u: float(np.mean(snr_arr[org == u])) for u in uniq}
    winners = {u: variants[int(np.argmin(np.nanmean(M[org == u], axis=0)))] for u in uniq}
    return dict(n_origins=len(uniq), deg_router_nrmse=l1, struct_oracle_nrmse_UPPER_BOUND=l2,
                gap_pat_upper_bound=gap, perm_p_plain=p_plain, perm_p_snr_stratified=p_strat,
                interaction_var_ratio=inter_ratio, snr_by_origin=snr_by_origin,
                per_struct_winner=winners,
                d12c_pass_stratified=bool(p_strat < 0.05 and inter_ratio > 0.10))


# ══════════════════════════════════════════════════════════════════════════
# 4. 语料落 cell + 每 cell 缓存（按 uid 交集对齐）
# ══════════════════════════════════════════════════════════════════════════
def assign_cells(corpus):
    h = HarnessState.from_minimal()
    cells: "OrderedDict[str, List[RawSeries]]" = OrderedDict()
    snr_of: Dict[str, float] = {}
    for rs in corpus:
        key = perceive(rs.history, "forecast", h)
        cells.setdefault(key["cell_id"], []).append(rs)
        snr_of[rs.series_uid] = float(key["pattern"]["struct_feats"].get("SNR", 0.0))
    return cells, snr_of


def diagnose(cells):
    print("== 语料落 cell 诊断（cell = task|SNRbin|missbin）==")
    for cid, series in sorted(cells.items()):
        by_org = defaultdict(int)
        for rs in series:
            by_org[rs.origin] += 1
        uids = len({rs.series_uid for rs in series})
        comp = " ".join(f"{k}={v}" for k, v in sorted(by_org.items()))
        print(f"  {cid:26s} n={len(series):3d} uids={uids:3d} "
              f"[{'LOWCONF' if uids < 6 else 'ok':7s}] origins: {comp}")


def build_cell_cache(fp, series, variants):
    """→ (action_caches[action]={uid:cache}, common_uids, origins{uid}, )。按所有动作成功缓存的 uid **交集**对齐。"""
    action_caches: Dict[str, Dict[str, dict]] = {}
    origin_of: Dict[str, str] = {rs.series_uid: rs.origin for rs in series}
    for a, h in variants.items():
        d = {}
        for rs in series:
            ready = fast_process(rs.history, "forecast", h, store=None)[1]
            c = _cache_one(fp, ready, rs.future, rs.obs_scale)
            if c is not None:
                d[rs.series_uid] = c
        action_caches[a] = d
    common = set.intersection(*[set(d) for d in action_caches.values()]) if action_caches else set()
    common_uids = sorted(common)                              # 显式交集对齐（防错行）
    return action_caches, common_uids, origin_of


# ══════════════════════════════════════════════════════════════════════════
# 5. 主流程：点估计 + bootstrap（内重拟合）
# ══════════════════════════════════════════════════════════════════════════
def run(cells, snr_of, variants_map, boot, perm, kfold):
    fp = FrozenProbe()
    variants = list(variants_map.keys())
    rng = np.random.default_rng(20260703)
    report = {"cells": {}, "config": dict(boot=boot, perm=perm, kfold=kfold, variants=variants,
                                          note="E-1.1 / operator_pool_v2.1 (S0.7-6/8): 算子身份修复(period/STL/wavelet/impute_fft)+边界语义修复(median/MA symmetric, A-31b)"
                                               " + A-28 bootstrap 修正; 依赖指纹+fallback台账+semantic-equivalence 守卫")}
    long_rows = []                                            # 完整响应矩阵（点估计）
    cell_data = OrderedDict()                                 # 供跨 cell 聚合 bootstrap
    prov.start_recording()                                    # S0.7：记录点估计期算子 requested/effective

    for cid, series in sorted(cells.items()):
        uids_all = {rs.series_uid for rs in series}
        if len(uids_all) < 6:
            report["cells"][cid] = {"skip": "LOWCONF", "uids": len(uids_all)}
            print(f"\n[skip] {cid} LOWCONF (uids={len(uids_all)}<6)")
            continue
        print(f"\n[cell] {cid}  building cache…")
        action_caches, common_uids, origin_of = build_cell_cache(fp, series, variants_map)
        n = len(common_uids)
        origins = [origin_of[u] for u in common_uids]
        snr = [snr_of[u] for u in common_uids]

        # ── 点估计响应矩阵（uid 对齐）──
        fold_pt = {u: i % kfold for i, u in enumerate(common_uids)}
        cols = []
        for a in variants:
            losses = _oof_losses(action_caches[a], common_uids, fold_pt, kfold)
            cols.append([losses.get(u, np.nan) for u in common_uids])
        M = np.array(cols).T                                  # (n_uid, n_action)，行=common_uids 顺序
        for i, u in enumerate(common_uids):
            for j, a in enumerate(variants):
                long_rows.append(dict(cell=cid, uid=u, origin=origins[i], snr=snr[i],
                                      fold=fold_pt[u], action=a, oof_nrmse=float(M[i, j])))

        col_mean = np.nanmean(M, axis=0)
        order = np.argsort(col_mean)
        top1, top2 = int(order[0]), int(order[1])
        sv = svd_point(M)
        nt = near_tie_fraction(M)
        pat = pattern_point(M, origins, variants, snr, perm, rng)

        # S0.7 semantic-equivalence 守卫：任何两动作全量相同 → 报警（静默回退/重复列嫌疑，如旧 wavelet==savgol）
        dup = []
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                both = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
                if both.any() and np.allclose(M[both, i], M[both, j]):
                    dup.append([variants[i], variants[j]])
        if dup:
            print(f"   ⚠ SEMANTIC-DUP: {dup}（动作身份不真实）")

        cell_data[cid] = dict(action_caches=action_caches, common_uids=common_uids,
                              origins=np.array(origins),
                              origin_map={u: origins[i] for i, u in enumerate(common_uids)},
                              top1=top1, top2=top2)
        report["cells"][cid] = dict(uids=n, n_rows=n,
                                    col_mean_nrmse={v: float(col_mean[j]) for j, v in enumerate(variants)},
                                    point_top1=variants[top1], point_top2=variants[top2],
                                    svd=sv, near_tie_frac=nt, pattern=pat, semantic_dup=dup)
        print(f"   point winner={variants[top1]} ({col_mean[top1]:.3f}) vs {variants[top2]} ({col_mean[top2]:.3f})")
        print(f"   SVD first_sv={sv['first_sv_var_ratio']:.2f} eff_rank={sv['effective_rank']:.2f} "
              f"recon_err@rank2={sv['recon_err_rank2']:.2f} near_tie={nt:.2f}")
        if pat:
            print(f"   [pattern] deg_router={pat['deg_router_nrmse']:.3f} "
                  f"struct_oracle(UB)={pat['struct_oracle_nrmse_UPPER_BOUND']:.3f} "
                  f"gap_UB={pat['gap_pat_upper_bound']:+.3f} p_plain={pat['perm_p_plain']:.3f} "
                  f"p_snrStrat={pat['perm_p_snr_stratified']:.3f} inter={pat['interaction_var_ratio']:.2f}")
            print(f"      per-origin SNR: {pat['snr_by_origin']}  winners: {pat['per_struct_winner']}")

    prov.stop_recording()                                    # S0.7：落算子 provenance（依赖版本 + fallback 台账）
    from .operators._common import BOUNDARY_MODES            # S0.7-8：边界语义随产物落盘
    report["provenance"] = dict(dependency=prov.dependency_fingerprint(), fallbacks=prov.fallback_summary(),
                                boundary_modes=dict(BOUNDARY_MODES))
    fbs = report["provenance"]["fallbacks"]
    print(f"\n[provenance] deps={report['provenance']['dependency']}")
    print(f"[provenance] operator fallbacks (requested→effective×count): "
          f"{ {k: v for k, v in fbs.items() if list(v) != [k]} }")

    # ── 全局 bootstrap（外层 B，内层每 cell 重采样+重拟合；聚合 L0/L1/L2）──
    # A-28a/b：L1 与 L2 **同一 bootstrap 分布同一权重**（origin 权重用 samp 重数、非原始计数）
    #   → 数学不变量 L0≥L1≥L2 逐 replicate 成立（断言守）；训练按重数加权（sample_weight）。
    from collections import Counter
    TOL = 1e-9
    active = list(cell_data)
    print(f"\n== bootstrap: B={boot}, refit-in-bootstrap (multiplicity-weighted) over {len(active)} cells ==")
    b_gap = {c: [] for c in active}
    b_hit = {c: 0 for c in active}
    b_first_sv = {c: [] for c in active}
    L0s, L1s, L2s = [], [], []
    for b in range(boot):
        util_cell, l1_cell, l2_cell = {}, {}, {}
        for c in active:
            cd = cell_data[c]
            caches_by_action = cd["action_caches"]
            uids = cd["common_uids"]
            omap = cd["origin_map"]
            n = len(uids)
            samp_uids = [uids[i] for i in rng.integers(0, n, size=n)]   # 重采样多重集
            mult = Counter(samp_uids)                                   # uid → 本 replicate 重数
            distinct = list(mult.keys())
            fperm = rng.permutation(len(distinct))
            fold_of = {distinct[i]: int(fperm[i]) % kfold for i in range(len(distinct))}
            wu = np.array([mult[u] for u in distinct], float)           # distinct uid 权重 = 重数
            W = float(wu.sum())
            # per action：训练按重数加权（sample_weight），返回 distinct uid 的 OOF loss
            loss_by_a = {a: _oof_losses(caches_by_action[a], distinct, fold_of, kfold, weight=mult)
                         for a in variants}

            def wmean(a, sub_uids, sub_w):
                v = np.array([loss_by_a[a].get(u, np.nan) for u in sub_uids])
                m = np.isfinite(v)
                return float(np.sum(sub_w[m] * v[m]) / np.sum(sub_w[m])) if m.any() and np.sum(sub_w[m]) > 0 else np.nan

            util = np.array([wmean(a, distinct, wu) for a in variants])  # cell 级重数加权效用
            util_cell[c] = util
            l1 = float(np.nanmin(util))
            # L2：每 origin 各取 best action，**用同一 bootstrap 重数权重**（修 A-28a）
            l2, wsum = 0.0, 0.0
            for uo in set(omap[u] for u in distinct):
                sub = [u for u in distinct if omap[u] == uo]
                sw = np.array([mult[u] for u in sub], float)
                per = np.array([wmean(a, sub, sw) for a in variants])
                if np.isfinite(per).any():
                    l2 += float(sw.sum()) * float(np.nanmin(per))
                    wsum += float(sw.sum())
            l2 = l2 / wsum if wsum > 0 else l1
            assert l2 <= l1 + TOL, f"L2>L1 违反不变量 @ {c}: l1={l1} l2={l2}"  # 同样本同权重 → L2(结构oracle)≤L1
            l1_cell[c], l2_cell[c] = l1, l2
            # 固定 top1/top2 配对 gap（可为负）+ winner 命中 + SVD（多重集行）
            t1, t2 = cd["top1"], cd["top2"]
            b_gap[c].append(float(util[t2] - util[t1]))
            if int(np.nanargmin(util)) == t1:
                b_hit[c] += 1
            Mb = np.array([[loss_by_a[a].get(u, np.nan) for a in variants] for u in samp_uids])
            s2, tot, _ = _row_centered_svd(Mb)
            b_first_sv[c].append(float(s2[0] / tot))
        # 聚合三层（cell 等权）
        glob = np.nanmean(np.vstack([util_cell[c] for c in active]), axis=0)
        L0 = float(np.nanmin(glob))
        L1 = float(np.mean([l1_cell[c] for c in active]))
        L2 = float(np.mean([l2_cell[c] for c in active]))
        assert L0 >= L1 - TOL and L1 >= L2 - TOL, f"三层不变量违反: {L0} {L1} {L2}"
        L0s.append(L0); L1s.append(L1); L2s.append(L2)

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    for c in active:
        gaps = np.array(b_gap[c])
        report["cells"][c]["good_definedness"] = dict(
            point_top1=report["cells"][c]["point_top1"],
            win_rate=b_hit[c] / boot,
            gap_point=float(np.mean(gaps)), gap_ci=ci(gaps),
            well_defined=bool(b_hit[c] / boot >= 0.90 and np.percentile(gaps, 2.5) > 0))
        report["cells"][c]["svd"]["first_sv_ci"] = ci(np.array(b_first_sv[c]))

    L0s, L1s, L2s = map(np.array, (L0s, L1s, L2s))
    deg_gain, pat_gain = L0s - L1s, L1s - L2s     # 修 A-28 后 pat_gain≥0 逐 replicate（不变量）
    report["decomposition"] = dict(
        L0_global_oracle=[float(L0s.mean()), *ci(L0s)], L1_deg_router_oracle=[float(L1s.mean()), *ci(L1s)],
        L2_struct_oracle_UPPER_BOUND=[float(L2s.mean()), *ci(L2s)],
        gain_deg_L0_L1=[float(deg_gain.mean()), *ci(deg_gain)],
        gain_pat_L1_L2_UPPER_BOUND=[float(pat_gain.mean()), *ci(pat_gain)],
        # share 仅**描述性**（A-28c）：pat 近零、上界乐观、近零量转百分比不稳定 → 不作 headline
        deg_share_pct_DESCRIPTIVE=float(100 * deg_gain.mean() / (deg_gain.mean() + pat_gain.mean() + 1e-12)),
        pat_share_pct_DESCRIPTIVE=float(100 * pat_gain.mean() / (deg_gain.mean() + pat_gain.mean() + 1e-12)),
        note=("L0/L1 是 oracle 选择值(在数据上选 cell-best action)——可实现 degradation router 仍须 E-3.2 held-out Lookup 验; "
              "L2 是 in-sample structure oracle 信息上界; pat 增益修不变量后≥0 但可实现值须 E-3.2 held-out policy regret"))
    return report, long_rows


def summarize(report):
    cells = {c: r for c, r in report["cells"].items() if "skip" not in r}
    n = len(cells)
    n_wd = sum(1 for r in cells.values() if r["good_definedness"]["well_defined"])
    frac = n_wd / n if n else 0.0
    first_svs = [r["svd"]["first_sv_var_ratio"] for r in cells.values()]
    eff = [r["svd"]["effective_rank"] for r in cells.values()]
    pats = [r["pattern"] for r in cells.values() if r.get("pattern")]
    return dict(n_cells=n, n_well_defined=n_wd, frac_well_defined=frac,
                d11b_verdict=("function" if frac >= 0.5 else "subset-only" if frac >= 0.3 else "plateau"),
                median_first_sv=float(np.median(first_svs)), median_eff_rank=float(np.median(eff)),
                d11c_single_axis=bool(np.median(first_svs) > 0.90),
                n_d12c_stratified_pass=int(sum(p["d12c_pass_stratified"] for p in pats)),
                n_pattern_cells=len(pats))


def main():
    ap = argparse.ArgumentParser(description="E-1.1R2 良定性 gate + degradation/pattern 分开估计（operator_pool_v2）")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=1000)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--out", default=None, help="产物目录（默认 results/E1_1；v2 传 results/E1_1_v2 保留 v1）")
    ap.add_argument("--drop-wavelet", action="store_true", help="剔除 wavelet（复现 v1 的 6 动作口径）")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else RESULTS

    t0 = time.time()
    corpus = build_corpus(args.n_seeds)
    cells, snr_of = assign_cells(corpus)
    print(f"== corpus: {len(corpus)} forecast series, {len(STRUCTS)}×{len(DEG_GRID)}×{args.n_seeds} → {len(cells)} cells ==")
    diagnose(cells)
    if args.diagnose:
        print(f"\n[diagnose only] {time.time()-t0:.1f}s")
        return

    variants_map = fixed_harness_variants("forecast")
    if args.drop_wavelet:                                     # v1 复现用（病态 wavelet）；v2 默认保留修好的 wavelet
        variants_map.pop("v_wavelet", None)
    # operator_pool_v2（S0.7）：wavelet 已修（有界 level+symmetric+VisuShrink，nRMSE 0.204<savgol），
    #   _guess_period 已修（去趋势+ACF→纯趋势返回 0，STL 不再静默伪装），impute_fft 观测保持。
    #   → 7 个**语义真实**动作；semantic-equivalence 守卫在响应矩阵后报警任何全量相同列。
    print(f"\n== variants ({len(variants_map)}): {list(variants_map)}  [operator_pool_v2, S0.7] ==")
    report, long_rows = run(cells, snr_of, variants_map, args.boot, args.perm, args.kfold)
    report["summary"] = summarize(report)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out_dir / "response_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cell", "uid", "origin", "snr", "fold", "action", "oof_nrmse"])
        w.writeheader()
        w.writerows(long_rows)

    s = report["summary"]
    d = report["decomposition"]
    print("\n" + "=" * 78 + "\n== E-1.1R2 SUMMARY ==")
    print(f"  non-LOWCONF cells: {s['n_cells']}  well-defined: {s['n_well_defined']} "
          f"({s['frac_well_defined']:.0%}) → D-1.1b: {s['d11b_verdict']}")
    for cid, r in sorted(report["cells"].items()):
        if "skip" in r:
            continue
        g = r["good_definedness"]
        print(f"    {cid:24s} win={r['point_top1']:16s} win_rate={g['win_rate']:.2f} "
              f"gap={g['gap_point']:+.3f} CI[{g['gap_ci'][0]:+.3f},{g['gap_ci'][1]:+.3f}] wd={g['well_defined']}")
    print(f"  median first-SV {s['median_first_sv']:.2f} eff_rank {s['median_eff_rank']:.2f} "
          f"→ single-axis: {s['d11c_single_axis']}")
    print(f"\n  三层 ORACLE 分解 (nRMSE, cell 等权, bootstrap CI; L0/L1/L2 均为数据内 oracle 选择):")
    print(f"    L0 global-oracle {d['L0_global_oracle'][0]:.4f}  CI[{d['L0_global_oracle'][1]:.4f},{d['L0_global_oracle'][2]:.4f}]")
    print(f"    L1 deg-router-oracle {d['L1_deg_router_oracle'][0]:.4f}  CI[{d['L1_deg_router_oracle'][1]:.4f},{d['L1_deg_router_oracle'][2]:.4f}]")
    print(f"    L2 struct-oracle(UB) {d['L2_struct_oracle_UPPER_BOUND'][0]:.4f}  "
          f"CI[{d['L2_struct_oracle_UPPER_BOUND'][1]:.4f},{d['L2_struct_oracle_UPPER_BOUND'][2]:.4f}]")
    print(f"    gain L0→L1 (deg-conditioning oracle) {d['gain_deg_L0_L1'][0]:+.4f} "
          f"CI[{d['gain_deg_L0_L1'][1]:+.4f},{d['gain_deg_L0_L1'][2]:+.4f}]")
    print(f"    gain L1→L2 (pattern, in-sample UB)   {d['gain_pat_L1_L2_UPPER_BOUND'][0]:+.4f} "
          f"CI[{d['gain_pat_L1_L2_UPPER_BOUND'][1]:+.4f},{d['gain_pat_L1_L2_UPPER_BOUND'][2]:+.4f}]")
    print(f"    [descriptive only] deg {d['deg_share_pct_DESCRIPTIVE']:.1f}% / pattern(UB) {d['pat_share_pct_DESCRIPTIVE']:.1f}%  "
          f"— 不作 headline（pat 近零+上界乐观）")
    print(f"    可实现价值须 E-3.2 held-out（degradation Lookup vs full-pattern GBDT policy regret）")
    print(f"  structure×action (SNR-stratified perm) pass: {s['n_d12c_stratified_pass']}/{s['n_pattern_cells']} cells")
    print(f"  semantic-dup cells: {[c for c,r in report['cells'].items() if 'skip' not in r and r.get('semantic_dup')]}")
    print(f"\n  → report.json + response_matrix.csv @ {out_dir}\n[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
