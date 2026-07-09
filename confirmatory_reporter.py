"""confirmatory_reporter.py — confirmatory 独立报告器 panel（A-40⑥/A-41⑤，门 C6）。

panel（冻结）= {dlinear_scratch(S=5), chronos}——与 in-loop 判官 frozen_probe 不相交
（report_target 分离不变量）。对三策略批 {dp_abstain, global, d_lookup}（picks 读自
locked-transfer records，**不重算策略**）：per-uid 应用选中动作 → ready 批 → per-series
obs-norm nRMSE → perf=exp(−mean nRMSE)（report_target 冻结口径）。

不确定性分型（A-41⑤）：
  dlinear_scratch  S=5 训练种子——per-series nRMSE 先跨种子平均，再 uid 配对 bootstrap；
                   种子间离散（per-seed batch perf 的 std）另报。B×S 全量重训不做（预锁不可行）。
  chronos          零样本确定性、无可重拟合头 → grouped paired-uid bootstrap。

门 C6：两报告器各自 perf(dp)>perf(global) 且 perf(dp)>perf(d_lookup)（点方向）；且四个配对
per-series nRMSE 差（dp−base）的 bootstrap CI **无显著反向**（反向=dp 显著更差=ci_lo>0）。
season/trend 子群方向同报（防"均值确认、season 重伤"）。

无静默回退（守卫⑤）：任何 series 级非有限值直接 raise；模型身份/版本入 provenance（守卫⑥）。

运行：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe \
        -m SelfEvolvingHarnessTS.confirmatory_reporter [--smoke-dev] [--scope primary_no_Sar]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .confirmatory_freeze import (FREEZE_PATH, PAIRED_BOOT_B, REPORTER_POLICIES, REPORTER_SEEDS,
                                  RESULTS_CONF)
from .e32_policy import PRUNED_POOL_CORE
from .evaluators.base import H_FORECAST, L_WIN
from .fast_path.pipeline import process as fast_process
from .run_confirmatory import LT_DIR_NAME, _out_root, load_corpus_and_manifest
from .run_e32 import _variant_map

CHUNK = 64


def _ready_batches(records: List[dict], corpus_by_uid: Dict[str, object],
                   policies=REPORTER_POLICIES) -> Dict[str, List[np.ndarray]]:
    """per policy 的 ready 列表（与 records 顺序对齐）；(uid, action) 级缓存避免跨策略重算。"""
    from .evaluators.chronos_probe import _fillna
    variants = _variant_map(list(PRUNED_POOL_CORE))
    cache: Dict[tuple, np.ndarray] = {}
    out = {p: [] for p in policies}
    for r in records:
        rs = corpus_by_uid[r["uid"]]
        for p in policies:
            a = r["arms"][p]["pick"]
            key = (r["uid"], a)
            if key not in cache:
                ready = fast_process(rs.history, "forecast", variants[a], store=None)[1]
                cache[key] = _fillna(np.asarray(ready, float))
            out[p].append(cache[key])
    return out


def _per_series_nrmse_dlinear(ready: List[np.ndarray], records: List[dict],
                              corpus_by_uid: Dict[str, object], seed: int) -> np.ndarray:
    from .evaluators import _torch_models as tm
    from .evaluators.grounded_forecast import build_windows
    X, Y = build_windows(ready)
    if X is None or len(X) < 10:
        raise RuntimeError("reporter 守卫⑤：dlinear 训练窗不足（无静默回退）")
    tm.seed_all(seed)
    model = tm.train_forecaster(tm.DLinear(L_WIN, H_FORECAST), X, Y, epochs=120)
    out = []
    for hh, r in zip(ready, records):
        if not np.all(np.isfinite(hh)) or hh.size < L_WIN:
            raise RuntimeError(f"reporter 守卫⑤：uid={r['uid']} ready 非法（无静默回退）")
        pred = tm.forecast_predict(model, hh[-L_WIN:].reshape(1, -1)).ravel()
        fut = np.asarray(corpus_by_uid[r["uid"]].future, float).ravel()
        h = min(len(pred), len(fut))
        rmse = float(np.sqrt(np.mean((pred[:h] - fut[:h]) ** 2)))
        v = rmse / (corpus_by_uid[r["uid"]].obs_scale + 1e-9)
        if not np.isfinite(v):
            raise RuntimeError(f"reporter 守卫⑤：uid={r['uid']} nRMSE 非有限")
        out.append(v)
    return np.array(out)


def _per_series_nrmse_chronos(ready: List[np.ndarray], records: List[dict],
                              corpus_by_uid: Dict[str, object]) -> np.ndarray:
    import torch
    from .evaluators.chronos_probe import get_chronos
    pipe = get_chronos()
    out = np.full(len(ready), np.nan)
    for lo in range(0, len(ready), CHUNK):
        chunk = ready[lo:lo + CHUNK]
        contexts = [torch.tensor(h[-512:], dtype=torch.float32) for h in chunk]
        _, mean = pipe.predict_quantiles(contexts, prediction_length=H_FORECAST,
                                         quantile_levels=[0.5])
        preds = mean.detach().cpu().numpy()
        for i, pred in enumerate(preds):
            r = records[lo + i]
            fut = np.asarray(corpus_by_uid[r["uid"]].future, float).ravel()
            h = min(len(pred), len(fut))
            rmse = float(np.sqrt(np.mean((pred[:h] - fut[:h]) ** 2)))
            out[lo + i] = rmse / (corpus_by_uid[r["uid"]].obs_scale + 1e-9)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("reporter 守卫⑤：chronos nRMSE 含非有限值")
    return out


def _paired_ci(diff: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 424244)
    boots = np.array([float(np.mean(diff[rng.integers(0, len(diff), len(diff))]))
                      for _ in range(n_boot)])
    return dict(mean=float(diff.mean()), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), n=int(len(diff)))


def _provenance() -> dict:
    import sklearn
    out = dict(numpy=np.__version__, sklearn=sklearn.__version__,
               dlinear="evaluators._torch_models.DLinear(L_WIN,H) epochs=120",
               L_WIN=int(L_WIN), H_FORECAST=int(H_FORECAST))
    try:
        import torch
        out["torch"] = torch.__version__
    except Exception:
        out["torch"] = None
    try:
        from importlib.metadata import version
        from .evaluators.chronos_probe import MODEL_ID
        out["chronos_model"] = MODEL_ID
        out["chronos_forecasting"] = version("chronos-forecasting")
    except Exception:
        out["chronos_model"] = None
    return out


def run_panel(scope: str, smoke_dev: bool, seed: int = 20260705, limit: int = 0) -> dict:
    root = _out_root(smoke_dev) / LT_DIR_NAME
    rec_path = root / f"records_locked_{scope}.jsonl"
    records = [json.loads(l) for l in rec_path.read_text("utf-8").splitlines() if l.strip()]
    if limit:
        assert smoke_dev, "--limit 仅限 smoke（正式 panel 口径冻结为全量，A-40⑥）"
        records = records[::max(1, len(records) // limit)][:limit]
    corpus, _ = load_corpus_and_manifest(smoke_dev)
    corpus_by_uid = {rs.series_uid: rs for rs in corpus}
    print(f"reporter panel：scope={scope} n={len(records)}（smoke_dev={smoke_dev}）", flush=True)
    ready = _ready_batches(records, corpus_by_uid)

    nrmse: Dict[str, Dict[str, np.ndarray]] = {"dlinear_scratch": {}, "chronos": {}}
    seed_spread: Dict[str, dict] = {}
    for p in REPORTER_POLICIES:
        t0 = time.time()
        per_seed = []
        for s in REPORTER_SEEDS:
            per_seed.append(_per_series_nrmse_dlinear(ready[p], records, corpus_by_uid, s))
            print(f"  [dlinear] policy={p} seed={s} 完成 [{time.time()-t0:.0f}s]", flush=True)
        arr = np.stack(per_seed)                              # (S, n)
        nrmse["dlinear_scratch"][p] = arr.mean(axis=0)        # per-series 跨种子平均
        seed_spread[p] = dict(per_seed_batch_perf=[float(np.exp(-a.mean())) for a in per_seed],
                              batch_perf_std=float(np.std([np.exp(-a.mean()) for a in per_seed])))
        t0 = time.time()
        nrmse["chronos"][p] = _per_series_nrmse_chronos(ready[p], records, corpus_by_uid)
        print(f"  [chronos] policy={p} 完成 [{time.time()-t0:.0f}s]", flush=True)

    origins = np.array([r["origin"] for r in records])
    result = dict(scope=scope, smoke_dev=smoke_dev, n=len(records),
                  policies=list(REPORTER_POLICIES), provenance=_provenance(),
                  dlinear_seed_spread=seed_spread, reporters={})
    gate_dirs, gate_reversals = [], []
    for rep in ("dlinear_scratch", "chronos"):
        block = dict(perf={p: float(np.exp(-nrmse[rep][p].mean())) for p in REPORTER_POLICIES},
                     comparisons={}, subgroup_direction={})
        for base in ("global", "d_lookup"):
            diff = nrmse[rep]["dp_abstain"] - nrmse[rep][base]      # 负 = dp 更好（nRMSE 更低）
            ci = _paired_ci(diff, PAIRED_BOOT_B, seed)
            direction_ok = block["perf"]["dp_abstain"] > block["perf"][base]
            reversal = bool(ci["ci_lo"] > 0)                        # dp 显著更差
            block["comparisons"][f"dp_abstain_vs_{base}"] = dict(
                nrmse_diff_ci=ci, direction_ok=bool(direction_ok), significant_reversal=reversal)
            gate_dirs.append(bool(direction_ok))
            gate_reversals.append(reversal)
            for org in ("S_season", "S_trend"):
                m = origins == org
                if m.any():
                    block["subgroup_direction"][f"{org}_vs_{base}"] = dict(
                        mean_nrmse_diff=float(diff[m].mean()),
                        dp_better=bool(diff[m].mean() < 0), n=int(m.sum()))
        result["reporters"][rep] = block
    result["gate"] = dict(
        passed=bool(all(gate_dirs) and not any(gate_reversals)),
        directions_ok=gate_dirs, significant_reversals=gate_reversals,
        rule="两报告器 perf(dp)>perf(global)∧>perf(d_lookup) 且无显著反向（freeze C6）")
    out_path = root / f"reporter_panel_{scope}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")
    print(f"gate C6: {result['gate']}", flush=True)
    print(f"→ {out_path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="primary_no_Sar", choices=["primary_no_Sar", "all_data"])
    ap.add_argument("--smoke-dev", action="store_true")
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="仅 smoke：子采样 uid 数")
    args = ap.parse_args()
    if not args.smoke_dev and not args.open_holdout:
        raise SystemExit("门禁：非 smoke 读取 holdout，必须显式 --open-holdout。")
    if not args.smoke_dev and not FREEZE_PATH.exists():
        raise SystemExit("A-41 门禁：freeze 未落盘。")
    if not args.smoke_dev and args.limit:
        raise SystemExit("--limit 仅限 smoke（正式 panel 冻结为全量）。")
    run_panel(args.scope, args.smoke_dev, limit=args.limit)


if __name__ == "__main__":
    main()
