"""tensor_runner.py — 2.3 三模型效用张量最小 runner（协议 v3=frozen_full，张量 gate 已合格）。

张量槽 U(domain, action, model) → per-series nRMSE（域为独立性单位，v2 勘误口径）：
  seasonal_naive      解析基线：processed 历史最后 period(24) 段平铺至 H=48；
  dlinear_pooled      **within-domain pooled**：域内全部序列的 processed train 段汇总构窗 →
                      训练一个共享 DLinear（epochs=120 × seeds S=5 平均）→ 逐序列末窗评估
                      —— = report_target.py 实际口径 = "用处理好的数据训练下游模型"；
  chronos_bolt_small  zero-shot 逐序列（确定性 σ_A=0）。
（评估循环在本模块自实现：chronos_probe/report_target 在 confirmatory 冻结指纹清单内不可改；
 只 import 其 get_chronos/_fillna/torch 工具，不动文件。）

域 = 合成 S2 family（8 域，series=该族 dev 全 uid）；真实 3 域（协议 v3 real_domains_frozen）
为后续波（--real 门控，本 runner 先合成）。动作 = core 10（协议 roles.core_pool）。
缓存：results/Stage2/tensor_cache/{domain}__{action}__{model}.json（存在即跳过 → resume；
per-slot cache_key = sha256(uid|action|menu_sha|model|train_cfg_sha) 按协议记录）。
失败策略（协议 failure_policy）：非有限 → 该槽记 missing 并计数，无静默剔除。

运行：  smoke（单域验成本）  python -m SelfEvolvingHarnessTS.tensor_runner --domain S_season
        全量 pilot           python -m SelfEvolvingHarnessTS.tensor_runner --all
        判读（两个决策数字） python -m SelfEvolvingHarnessTS.tensor_runner --analyze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .e32_policy import PRUNED_POOL_CORE
from .evaluators.base import H_FORECAST, L_WIN
from .evaluators.chronos_probe import _fillna, get_chronos
from .evaluators.grounded_forecast import build_windows
from .fast_path.pipeline import process as fast_process
from .run_e32 import _variant_map
from .s2_corpus import S2_FAMILIES, build_s2_dev

CACHE = Path(__file__).resolve().parent / "results" / "Stage2" / "tensor_cache"
MODELS = ("seasonal_naive", "dlinear_pooled", "chronos_bolt_small")
DL_EPOCHS, DL_SEEDS = 120, (0, 1, 2, 3, 4)
SNAIVE_PERIOD = 24
EPS, DELTA_SAFE, INTER_MIN, DOM_COVER = 0.03, 0.05, 0.15, 0.90
TRAIN_CFG = dict(dlinear=dict(epochs=DL_EPOCHS, seeds=list(DL_SEEDS), pooled="within-domain"),
                 snaive=dict(period=SNAIVE_PERIOD), chronos=dict(mode="zero-shot"))
_CFG_SHA = hashlib.sha256(json.dumps(TRAIN_CFG, sort_keys=True).encode()).hexdigest()[:16]


def _nrmse(pred: np.ndarray, fut: np.ndarray, obs: float) -> float:
    h = min(len(pred), len(fut))
    return float(np.sqrt(np.mean((pred[:h] - fut[:h]) ** 2)) / (obs + 1e-9))


def _slot_eval(model: str, processed: Dict[str, np.ndarray], series_of: Dict[str, object]) -> Dict[str, float]:
    """→ {uid: nRMSE}（非有限/失败 → 缺席该键=missing，调用方计数）。"""
    out: Dict[str, float] = {}
    uids = sorted(processed)
    if model == "seasonal_naive":
        for u in uids:
            hh = _fillna(processed[u])
            if hh.size < SNAIVE_PERIOD:
                continue
            reps = int(np.ceil(H_FORECAST / SNAIVE_PERIOD))
            pred = np.tile(hh[-SNAIVE_PERIOD:], reps)[:H_FORECAST]
            v = _nrmse(pred, np.asarray(series_of[u].future, float), series_of[u].obs_scale)
            if np.isfinite(v):
                out[u] = v
        return out
    if model == "dlinear_pooled":
        from .evaluators import _torch_models as tm
        ready = {u: _fillna(processed[u]) for u in uids}
        X, Y = build_windows([ready[u] for u in uids])
        if X is None or len(X) < 10:
            return out
        acc: Dict[str, List[float]] = {u: [] for u in uids}
        for sd in DL_SEEDS:                                    # 域内共享模型 × S seeds
            tm.seed_all(sd)
            model_t = tm.train_forecaster(tm.DLinear(L_WIN, H_FORECAST), X, Y, epochs=DL_EPOCHS)
            for u in uids:
                hh = ready[u]
                if hh.size < L_WIN or not np.all(np.isfinite(hh)):
                    continue
                pred = tm.forecast_predict(model_t, hh[-L_WIN:].reshape(1, -1)).ravel()
                v = _nrmse(pred, np.asarray(series_of[u].future, float), series_of[u].obs_scale)
                if np.isfinite(v):
                    acc[u].append(v)
        for u, vs in acc.items():
            if len(vs) == len(DL_SEEDS):                       # 任一 seed 失败 → missing（不半计）
                out[u] = float(np.mean(vs))
        return out
    if model == "chronos_bolt_small":
        import torch
        pipe = get_chronos()
        ctx, valid = [], []
        for u in uids:
            hh = _fillna(processed[u])
            if hh.size >= 8:
                ctx.append(torch.tensor(hh[-512:], dtype=torch.float32))
                valid.append(u)
        if not ctx:
            return out
        _, mean = pipe.predict_quantiles(ctx, prediction_length=H_FORECAST, quantile_levels=[0.5])
        preds = mean.detach().cpu().numpy()
        for pred, u in zip(preds, valid):
            v = _nrmse(np.asarray(pred, float).ravel(),
                       np.asarray(series_of[u].future, float), series_of[u].obs_scale)
            if np.isfinite(v):
                out[u] = v
        return out
    raise ValueError(model)


def run_domain(domain: str, actions: List[str], models=MODELS) -> None:
    from .policy import action_menu_v1
    menu_sha = action_menu_v1().sha256
    corpus = [rs for rs in build_s2_dev() if rs.origin == domain]
    assert corpus, f"未知域 {domain}"
    series_of = {rs.series_uid: rs for rs in corpus}
    variants = _variant_map(actions)
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"== 域 {domain}：n={len(corpus)} × {len(actions)} 动作 × {len(models)} 模型 ==", flush=True)
    for aid in actions:
        t0 = time.time()
        processed = {rs.series_uid: fast_process(rs.history, "forecast", variants[aid], store=None)[1]
                     for rs in corpus}
        for model in models:
            f = CACHE / f"{domain}__{aid}__{model}.json"
            if f.exists():
                continue
            t1 = time.time()
            per = _slot_eval(model, processed, series_of)
            doc = dict(domain=domain, action=aid, model=model, menu_sha=menu_sha,
                       train_cfg_sha=_CFG_SHA, n=len(per), n_missing=len(corpus) - len(per),
                       wallclock_s=round(time.time() - t1, 1),
                       cache_keys={u: hashlib.sha256(
                           f"{u}|{aid}|{menu_sha}|{model}|{_CFG_SHA}".encode()).hexdigest()[:16]
                           for u in list(per)[:1]},            # 键规则样例（全量键可再生）
                       per_series={u: round(v, 6) for u, v in per.items()})
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(doc, ensure_ascii=False), "utf-8")
            tmp.replace(f)
            print(f"  [{aid:16s}×{model:18s}] n={len(per):3d} miss={doc['n_missing']} "
                  f"[{doc['wallclock_s']}s]", flush=True)
        print(f"  [{aid}] 动作完成 [{time.time()-t0:.0f}s]", flush=True)


# ════════════════════════════ 判读：两个决策数字 ════════════════════════════
def analyze(actions: List[str], models=MODELS) -> dict:
    rows = []                                                  # (uid, domain, a, m, loss)
    for f in sorted(CACHE.glob("*__*.json")):
        doc = json.loads(f.read_text("utf-8"))
        for u, v in doc["per_series"].items():
            rows.append((u, doc["domain"], doc["action"], doc["model"], float(v)))
    uids = sorted({r[0] for r in rows})
    slot = {(r[0], r[2], r[3]): r[4] for r in rows}
    dom_of = {r[0]: r[1] for r in rows}
    full = [u for u in uids if all((u, a, m) in slot for a in actions for m in models)]
    print(f"完整 uid：{len(full)}/{len(uids)}（缺槽 uid 从判读剔除并报告——no silent caps）", flush=True)
    # —— per-series 标准化 + two-way 分解 ——
    Y = np.array([[[slot[(u, a, m)] for m in models] for a in actions] for u in full])  # (n,A,M)
    mu = Y.mean(axis=(1, 2), keepdims=True)
    sd = Y.std(axis=(1, 2), keepdims=True)
    sd[sd < 1e-12] = 1.0
    Z = (Y - mu) / sd
    g = Z.mean()
    ef_a = Z.mean(axis=(0, 2)) - g                             # action 主效应
    ef_m = Z.mean(axis=(0, 1)) - g                             # model 主效应
    cell = Z.mean(axis=0)                                      # (A,M)
    inter = cell - ef_a[:, None] - ef_m[None, :] - g
    n = Z.shape[0]
    ss_a = n * len(models) * float((ef_a ** 2).sum())
    ss_m = n * len(actions) * float((ef_m ** 2).sum())
    ss_i = n * float((inter ** 2).sum())
    ss_tot = float(((Z - g) ** 2).sum())
    share = ss_i / max(ss_a + ss_m + ss_i, 1e-12)              # 交互占系统性方差份额
    share_total = ss_i / max(ss_tot, 1e-12)
    # —— dominance：最强单一 (model, action) 对 ——
    oracle = Y.reshape(n, -1).min(axis=1)
    best, best_cover = None, -1.0
    for ai, a in enumerate(actions):
        for mi, m in enumerate(models):
            cover = float((Y[:, ai, mi] <= oracle + EPS).mean())
            if cover > best_cover:
                best, best_cover = (a, m), cover
    ai, mi = actions.index(best[0]), models.index(best[1])
    delta = Y[:, ai, mi] - oracle
    worst = {}
    doms = sorted({dom_of[u] for u in full})
    for d in doms:
        mask = np.array([dom_of[u] == d for u in full])
        dd = delta[mask]
        se = float(dd.std(ddof=1) / np.sqrt(len(dd))) if len(dd) > 1 else float("inf")
        worst[d] = dict(n=int(mask.sum()), mean=float(dd.mean()),
                        lcb=float(-(dd.mean() + 1.645 * se)))  # Δ 是亏损：安全侧看 −(mean+1.645se)>−δ
    worst_lcb = min(v["lcb"] for v in worst.values())
    out = dict(n_full=len(full), interaction_share_systematic=round(share, 4),
               interaction_share_total=round(share_total, 4),
               interaction_rule=f"≥{INTER_MIN} → L5 联合 (a,m)",
               dominance=dict(pair=best, eps_optimal_coverage=round(best_cover, 4),
                              cover_rule=f"≥{DOM_COVER}", worst_group_lcb=round(worst_lcb, 4),
                              safety_rule=f">−{DELTA_SAFE}", by_domain=worst),
               branch=("L5_joint" if share >= INTER_MIN else
                       "collapse_dominant" if best_cover >= DOM_COVER and worst_lcb > -DELTA_SAFE else
                       "sequential"))
    (CACHE.parent / "tensor_pilot_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", help="单域（smoke）")
    ap.add_argument("--all", action="store_true", help="全部 8 合成域")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    actions = list(PRUNED_POOL_CORE)
    if args.analyze:
        analyze(actions)
        return
    domains = list(S2_FAMILIES) if args.all else ([args.domain] if args.domain else [])
    if not domains:
        raise SystemExit("给 --domain <族>（smoke）或 --all 或 --analyze")
    for d in domains:
        run_domain(d, actions)


if __name__ == "__main__":
    main()
