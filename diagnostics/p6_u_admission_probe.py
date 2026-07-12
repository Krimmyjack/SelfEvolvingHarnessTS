# -*- coding: utf-8 -*-
"""diagnostics/p6_u_admission_probe.py — P6 预注册前 U 域准入探针（只读诊断，非实验本体）。

候选 unseen-domain U = Monash `electricity_hourly` / `traffic_hourly`（本项目此前从未下载）。
冻结判官协议：L_WIN=48 / H=48 / MIN_LEN=144（SelfEvolvingHarnessTS/evaluators/base.py 与
data/load_real.py:43）。hourly 数据同时有 24 与 168 两个周期而回看窗只有 48 —— 本探针判断
"主导可用周期是否 ≤48"，据此给出准入结论（不建议、也不裁量修改 L/H）。

五项报告（每 config）：
  1. 可用性：加载数 / 长度≥MIN_LEN 数 / 长度分布；
  2. period 估计行为：legacy_fft_v0（= conditioning/key.py struct_feats 感知端，P0 冻结契约）
     与 robust_v1（= 算子端 S0.7 修复版）在 last-1024 裁剪上的返回值分桶（≈24 / ≈168 / 其他）；
  3. ACF 强度：per-series z-score 后 lag-24 / lag-168 自相关（公式镜像 key.py `_acf`）；
  4. 48-lookback 判官能力：numpy 闭式 ridge-DLinear（镜像 evaluators/_torch_models.py DLinear
     语义：kernel=25 replicate padding 滑动平均取 trend，season=x−trend，φ=[trend;season] 96 维
     + 截距列；λ∈{1e-3,1.0} 两档），pool 探针序列 history=x[:-48] 滑窗（stride 4）拟合共享 W，
     用 x[:-48] 末 48 点预测 x[-48:]；对照 seasonal-naive-24（tile 末 24 点×2）与
     seasonal-naive-168（pred[t]=x[n-48-168+t]）；报告 nRMSE 均值比与逐序列胜率；
  5. 截断影响：同批 item_id 以 max_len=4096 重载（仅此诊断用），比较两估计器在
     last-4096 vs last-1024 上的分桶是否变化。

探针采样纪律：seed=20260710，对每 config 从按 item_id 排序后的可用序列（长度≥144）中抽 24 条；
落盘被探针 uid 清单 + 每条序列 sha256（NaN 线性填充后、z-score 前的 float64 字节）——
这些 uid 之后必须从最终 U 集排除（JSON 键 `excluded_uids_for_final_U`）。

准入判则（写入 JSON verdict 与 MD 结论）：
  · 多数探针序列"主导可用周期 ≤48"（robust period≈24，或 |acf24|>|acf168|）→ PASS
    （headline-U 候选）；
  · 多数序列 168 主导（robust period≈168 且 |acf168|≥|acf24|）→ STRESS
    （降级为 compound-seasonality stress test 候选）；
  · 皆非 → AMBIGUOUS。

红线：只写 results/Stage2/P6Probes/{u_admission_electricity_hourly.json,
u_admission_traffic_hourly.json, u_admission_report.md}；不改任何现有文件；除 HF parquet
下载外无网络调用；确定性（固定 seed、闭式解、加载顺序 = 排序 shard × parquet 行序）。

运行（项目根 Agent/ 下）：
  D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.diagnostics.p6_u_admission_probe
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:            # 支持按路径直跑（等价于项目根 -m 运行）
    sys.path.insert(0, str(_REPO_ROOT))

from AdaCTS.data.load_monash import load_config_series                      # noqa: E402
from SelfEvolvingHarnessTS.conditioning.period import (                     # noqa: E402
    dominant_period_fft_v0, guess_period_robust_v1,
)
from SelfEvolvingHarnessTS.evaluators.base import L_WIN, H_FORECAST         # noqa: E402

# ── 协议常量（冻结，只读引用）─────────────────────────────────────────────
MIN_LEN = (L_WIN + H_FORECAST) + H_FORECAST      # = data/load_real.py:43 的 96 + H_FORECAST
assert (L_WIN, H_FORECAST, MIN_LEN) == (48, 48, 144), "判官协议常量漂移——探针前提失效"

CONFIGS = ("electricity_hourly", "traffic_hourly")
SEED = 20260710                                  # 探针采样纪律（任务书钉死）
N_PROBE = 24
MAX_SERIES = 60
LOAD_MIN_LEN = 64                                # load_config_series 入参（≥144 过滤在其后）
MAX_LEN_MAIN = 1024
MAX_LEN_TRUNC = 4096                             # 仅第 5 项截断诊断用
STRIDE_PROBE = 4                                 # 判官滑窗步长（探针协议，非 evaluators.STRIDE）
KERNEL = 25                                      # DLinear 移动平均核（_torch_models.DLinear 默认）
LAMBDAS = (1e-3, 1.0)                            # ridge 两档
TOL_REL = 0.10                                   # period 分桶相对容差（FFT 网格 1024/k 取不到整 24/168）
OUT_DIR = _REPO_ROOT / "SelfEvolvingHarnessTS" / "results" / "Stage2" / "P6Probes"


# ═══════════════════════ 工具（镜像项目语义，纯 numpy）═══════════════════════
def _sha256_f64(x: np.ndarray) -> str:
    """NaN 填充后、z-score 前的 float64 字节指纹（load_config_series 返回值即该状态）。"""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def _zscore(x: np.ndarray) -> np.ndarray:
    """镜像 data/load_real.py `_zscore`（mean/std，std 护栏 1e-9）。"""
    x = np.asarray(x, dtype=float)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    return (x - m) / (s if s > 1e-9 else 1.0)


def _acf(x: np.ndarray, lag: int) -> float:
    """镜像 conditioning/key.py `_acf`（全序列去均值点积比）。"""
    if x.size <= lag:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def _bucket(p: float) -> str:
    """period 分桶：≈24 / ≈168 /（legacy 1.0 或 robust 0 =）none / other。相对容差 TOL_REL。"""
    if p is None or p < 2.0:
        return "none"
    if abs(p - 24.0) <= TOL_REL * 24.0:
        return "24"
    if abs(p - 168.0) <= TOL_REL * 168.0:
        return "168"
    return "other"


def _dlinear_phi(X: np.ndarray) -> np.ndarray:
    """镜像 _torch_models.DLinear 分解：replicate pad kernel//2 两侧 + 窗 25 滑动均值 = trend，
    season = X − trend；φ = [trend; season; 1] → (n, 2*L_WIN+1)。纯线性、确定性。"""
    X = np.asarray(X, dtype=float)
    pad = KERNEL // 2
    Xp = np.pad(X, ((0, 0), (pad, pad)), mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(Xp, KERNEL, axis=1)   # (n, L, KERNEL)
    trend = sw.mean(axis=2)[:, : X.shape[1]]
    season = X - trend
    ones = np.ones((X.shape[0], 1))
    return np.concatenate([trend, season, ones], axis=1)


def _fit_ridge(Phi: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """闭式 ridge：W = (ΦᵀΦ + λD)⁻¹ΦᵀY，D=I 但截距列不惩罚（sklearn fit_intercept 惯例）。
    [trend;season] 96 维由 48 维窗线性生成（秩 ≤49）——正是 ridge 处理该结构性共线的场景。"""
    d = Phi.shape[1]
    D = np.eye(d)
    D[-1, -1] = 0.0
    return np.linalg.solve(Phi.T @ Phi + lam * D, Phi.T @ Y)


def _judge_report(probe: list) -> dict:
    """第 4 项：48-lookback 闭式 ridge-DLinear vs seasonal-naive-24/168。
    协议：per-series z-score → pool 各序列 history=x[:-H] 的 (48→48) 滑窗（stride 4）拟合共享 W
    → 用 history 末 48 点预测 x[-H:] → per-series nRMSE（z 尺度下 obs_scale=1，nRMSE=RMSE）。"""
    Z = [_zscore(x) for _, x in probe]
    span = L_WIN + H_FORECAST
    Xs, Ys = [], []
    for z in Z:
        hist = z[:-H_FORECAST]
        for s in range(0, hist.size - span + 1, STRIDE_PROBE):
            Xs.append(hist[s:s + L_WIN])
            Ys.append(hist[s + L_WIN:s + span])
    X = np.asarray(Xs)
    Y = np.asarray(Ys)
    Phi = _dlinear_phi(X)
    Ws = {lam: _fit_ridge(Phi, Y, lam) for lam in LAMBDAS}

    per_series: list[dict] = []
    for (iid, _), z in zip(probe, Z):
        n = z.size
        hist, fut = z[:-H_FORECAST], z[-H_FORECAST:]
        rec: dict = {"item_id": iid}
        phi_last = _dlinear_phi(hist[-L_WIN:].reshape(1, -1))
        for lam in LAMBDAS:
            pred = (phi_last @ Ws[lam]).ravel()
            rec[f"nrmse_judge_lam{lam:g}"] = float(np.sqrt(np.mean((pred - fut) ** 2)))
        sn24 = np.tile(hist[-24:], 2)[:H_FORECAST]              # tile 末 24 点两次
        rec["nrmse_snaive24"] = float(np.sqrt(np.mean((sn24 - fut) ** 2)))
        if n >= H_FORECAST + 168:                               # pred[t] = x[n-48-168+t], t=0..47
            sn168 = z[n - H_FORECAST - 168: n - 168]
            rec["nrmse_snaive168"] = float(np.sqrt(np.mean((sn168 - fut) ** 2)))
        else:
            rec["nrmse_snaive168"] = None
        per_series.append(rec)

    def _mean(key):
        v = [r[key] for r in per_series if r[key] is not None]
        return float(np.mean(v)) if v else None

    def _median(key):
        v = [r[key] for r in per_series if r[key] is not None]
        return float(np.median(v)) if v else None

    summary: dict = {"n_windows_pooled": int(len(X)),
                     "protocol": {"stride": STRIDE_PROBE, "kernel": KERNEL,
                                  "lambdas": list(LAMBDAS), "phi_dim": 2 * L_WIN,
                                  "intercept": "unpenalized column",
                                  "zscore": "per-series, full loaded array",
                                  "nrmse": "RMSE on per-series z scale (obs_scale=1)"}}
    for lam in LAMBDAS:
        jk = f"nrmse_judge_lam{lam:g}"
        s = {"mean_nrmse_judge": _mean(jk), "median_nrmse_judge": _median(jk)}
        for base, bk in (("snaive24", "nrmse_snaive24"), ("snaive168", "nrmse_snaive168")):
            pairs = [(r[jk], r[bk]) for r in per_series if r[bk] is not None]
            s[f"mean_nrmse_{base}"] = _mean(bk)
            s[f"ratio_mean_judge_over_{base}"] = (s["mean_nrmse_judge"] / s[f"mean_nrmse_{base}"]
                                                  if pairs else None)
            s[f"winrate_judge_vs_{base}"] = (float(np.mean([j < b for j, b in pairs]))
                                             if pairs else None)
            s[f"n_compared_{base}"] = len(pairs)
        summary[f"lam{lam:g}"] = s
    return {"summary": summary, "per_series": per_series}


# ═══════════════════════ 单 config 探针 ═══════════════════════
def probe_config(config: str) -> dict:
    out: dict = {"config": config, "seed": SEED, "n_probe_target": N_PROBE,
                 "protocol_constants": {"L_WIN": L_WIN, "H": H_FORECAST, "MIN_LEN": MIN_LEN,
                                        "max_series": MAX_SERIES, "load_min_len": LOAD_MIN_LEN,
                                        "max_len_main": MAX_LEN_MAIN, "max_len_trunc": MAX_LEN_TRUNC,
                                        "period_bucket_rel_tol": TOL_REL},
                 "estimators": {"legacy_fft_v0": "conditioning/key.py struct_feats 感知端（P0 冻结契约）",
                                "robust_v1": "conditioning/period.py guess_period_robust_v1（S0.7 修复版）"}}
    t0 = time.time()
    try:
        series = load_config_series(config, split="test", max_series=MAX_SERIES,
                                    min_len=LOAD_MIN_LEN, max_len=MAX_LEN_MAIN)
    except Exception as exc:                     # noqa: BLE001 —— 红线：下载失败如实记录并继续
        out["status"] = "LOAD_FAILED"
        out["error"] = f"{type(exc).__name__}: {exc}"
        if "no parquet shards" in str(exc):      # 根因诊断：pinned parquet rev 是否根本没有该 config
            try:
                from huggingface_hub import HfApi
                from AdaCTS.data.load_monash import REPO_ID, PARQUET_REV
                files = HfApi().list_repo_files(REPO_ID, repo_type="dataset", revision=PARQUET_REV)
                cfgs = sorted({f.split("/")[0] for f in files if "/" in f})
                out["diagnosis"] = {
                    "config_present_in_parquet_rev": config in cfgs,
                    "n_configs_in_parquet_rev": len(cfgs),
                    "similar_configs": [c for c in cfgs if any(tok in c for tok in config.split("_"))],
                    "implication": "该 config 无法经项目 pinned loader（monash_tsf refs/convert/parquet）"
                                   "获取——在不改数据路径的前提下不可作为 U 候选",
                }
            except Exception as exc2:            # noqa: BLE001
                out["diagnosis"] = {"error": f"{type(exc2).__name__}: {exc2}"}
        return out
    lens_all = [int(x.size) for _, x in series]

    # ① 可用性
    eligible = sorted([(iid, x) for iid, x in series if x.size >= MIN_LEN], key=lambda t: t[0])
    lens_el = [int(x.size) for _, x in eligible]
    out["report1_availability"] = {
        "n_loaded": len(series), "n_len_ge_min_len": len(eligible),
        "length_loaded": {"min": min(lens_all) if lens_all else None,
                          "median": float(np.median(lens_all)) if lens_all else None,
                          "max": max(lens_all) if lens_all else None},
        "length_eligible": {"min": min(lens_el) if lens_el else None,
                            "median": float(np.median(lens_el)) if lens_el else None,
                            "max": max(lens_el) if lens_el else None},
    }
    if len(eligible) < 15:
        out["status"] = "INSUFFICIENT_SERIES"
        out["note"] = f"可用序列 {len(eligible)} < 15（红线判据），如实记录"
        if not eligible:
            return out
    # 探针采样：seed 固定，从 item_id 排序后的 eligible 抽 N_PROBE（不放回，选中索引升序落盘）
    rng = np.random.default_rng(SEED)
    k = min(N_PROBE, len(eligible))
    pick = sorted(rng.choice(len(eligible), size=k, replace=False).tolist())
    probe = [eligible[i] for i in pick]
    out["probe_manifest"] = [{"item_id": iid, "length": int(x.size), "sha256_f64": _sha256_f64(x)}
                             for iid, x in probe]
    out["excluded_uids_for_final_U"] = [iid for iid, _ in probe]

    # ② period 估计行为（last-1024 = 加载即裁剪状态）
    per2 = []
    for iid, x in probe:
        leg = float(dominant_period_fft_v0(x)[0])
        rob = int(guess_period_robust_v1(x))
        per2.append({"item_id": iid, "legacy_fft_v0": leg, "legacy_bucket": _bucket(leg),
                     "robust_v1": rob, "robust_bucket": _bucket(float(rob))})
    def _share(rows, key, val):
        return float(np.mean([r[key] == val for r in rows]))
    out["report2_period"] = {
        "per_series": per2,
        "legacy_fft_v0_share": {b: _share(per2, "legacy_bucket", b) for b in ("24", "168", "none", "other")},
        "robust_v1_share": {b: _share(per2, "robust_bucket", b) for b in ("24", "168", "none", "other")},
        "note": "分桶相对容差 10%（n=1024 的 FFT 网格 1024/k 取不到整 24/168：如 23.81/170.67）",
    }

    # ③ ACF 强度（z-score 后 lag-24/168）
    per3 = []
    for iid, x in probe:
        z = _zscore(x)
        a24, a168 = _acf(z, 24), _acf(z, 168)
        per3.append({"item_id": iid, "acf24": a24, "acf168": a168,
                     "abs24_gt_abs168": bool(abs(a24) > abs(a168))})
    out["report3_acf"] = {
        "per_series": per3,
        "acf24": {"mean": float(np.mean([r["acf24"] for r in per3])),
                  "median": float(np.median([r["acf24"] for r in per3]))},
        "acf168": {"mean": float(np.mean([r["acf168"] for r in per3])),
                   "median": float(np.median([r["acf168"] for r in per3]))},
        "share_abs_acf24_gt_abs_acf168": float(np.mean([r["abs24_gt_abs168"] for r in per3])),
    }

    # ④ 48-lookback 判官能力
    out["report4_judge48"] = _judge_report(probe)

    # ⑤ 截断影响（同批 item_id，max_len=4096 重载；仅此诊断用）
    try:
        series4k = load_config_series(config, split="test", max_series=MAX_SERIES,
                                      min_len=LOAD_MIN_LEN, max_len=MAX_LEN_TRUNC)
        by_id = {iid: x for iid, x in series4k}
        per5, missing, tail_mismatch = [], [], 0
        for iid, x1k in probe:
            x4k = by_id.get(iid)
            if x4k is None:
                missing.append(iid)
                continue
            if x4k.size >= x1k.size and not np.array_equal(x4k[-x1k.size:], x1k):
                tail_mismatch += 1               # NaN 线性填充在不同裁剪边界可产生差异 → 记录
            leg1, leg4 = float(dominant_period_fft_v0(x1k)[0]), float(dominant_period_fft_v0(x4k)[0])
            rob1, rob4 = int(guess_period_robust_v1(x1k)), int(guess_period_robust_v1(x4k))
            per5.append({"item_id": iid, "len_4096": int(x4k.size),
                         "legacy_1024": leg1, "legacy_4096": leg4,
                         "legacy_bucket_changed": _bucket(leg1) != _bucket(leg4),
                         "robust_1024": rob1, "robust_4096": rob4,
                         "robust_bucket_changed": _bucket(float(rob1)) != _bucket(float(rob4))})
        out["report5_truncation"] = {
            "per_series": per5, "n_matched": len(per5), "missing_item_ids": missing,
            "tail1024_mismatch_count": tail_mismatch,
            "legacy_bucket_change_share": (float(np.mean([r["legacy_bucket_changed"] for r in per5]))
                                           if per5 else None),
            "robust_bucket_change_share": (float(np.mean([r["robust_bucket_changed"] for r in per5]))
                                           if per5 else None),
        }
    except Exception as exc:                     # noqa: BLE001
        out["report5_truncation"] = {"status": "LOAD_FAILED", "error": f"{type(exc).__name__}: {exc}"}

    # 准入判决（判则见模块 docstring；不裁量 L/H）
    rob_bucket = {r["item_id"]: r["robust_bucket"] for r in per2}
    acf_dom24 = {r["item_id"]: r["abs24_gt_abs168"] for r in per3}
    acf_abs = {r["item_id"]: (abs(r["acf24"]), abs(r["acf168"])) for r in per3}
    flags_le48 = [(rob_bucket[iid] == "24") or acf_dom24[iid] for iid, _ in probe]
    flags_168 = [(rob_bucket[iid] == "168") and (acf_abs[iid][1] >= acf_abs[iid][0])
                 for iid, _ in probe]
    share_le48, share_168 = float(np.mean(flags_le48)), float(np.mean(flags_168))
    verdict = ("PASS_HEADLINE_U" if share_le48 > 0.5
               else "STRESS_COMPOUND_SEASONALITY" if share_168 > 0.5 else "AMBIGUOUS")
    out["verdict"] = {
        "rule": "PASS: 多数序列 robust period≈24 或 |acf24|>|acf168|；STRESS: 多数 robust≈168 且 |acf168|≥|acf24|",
        "share_dominant_period_le48": share_le48,
        "share_168_dominant": share_168,
        "verdict": verdict,
        "note": "不建议修改 L/H（非本探针裁量范围）",
    }
    out.setdefault("status", "OK")
    out["elapsed_sec"] = round(time.time() - t0, 1)
    return out


# ═══════════════════════ 汇总 MD ═══════════════════════
def _fmt(v, nd=4):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_report_md(results: dict, path: pathlib.Path) -> None:
    L = ["# P6 U 域准入探针报告（u_admission）", "",
         f"- 日期：2026-07-10；seed={SEED}；探针 n={N_PROBE}/config（item_id 排序后抽样，uid 落盘并排除出最终 U 集）",
         f"- 冻结判官协议：L_WIN={L_WIN}, H={H_FORECAST}, MIN_LEN={MIN_LEN}（evaluators/base.py, data/load_real.py:43）",
         "- 性质：只读准入探针，**不属于 P6 实验本体**；不裁量修改 L/H。", ""]
    for cfg, r in results.items():
        L += [f"## {cfg}", ""]
        if r.get("status") == "LOAD_FAILED":
            L += [f"**下载失败**：{r.get('error')}", ""]
            continue
        a = r["report1_availability"]
        L += [f"**① 可用性** 加载 {a['n_loaded']}，长度≥{MIN_LEN} 共 {a['n_len_ge_min_len']}；"
              f"长度 min/median/max = {a['length_loaded']['min']}/{_fmt(a['length_loaded']['median'],0)}/"
              f"{a['length_loaded']['max']}"]
        if r.get("status") == "INSUFFICIENT_SERIES":
            L += ["", f"**可用序列不足**：{r.get('note')}", ""]
            if "report2_period" not in r:
                continue
        p = r["report2_period"]
        L += ["", "**② period 估计**（分桶容差 ±10%）", "",
              "| estimator | ≈24 | ≈168 | none | other |", "|---|---|---|---|---|",
              "| legacy_fft_v0 (P0 感知端) | " + " | ".join(
                  _fmt(p["legacy_fft_v0_share"][b], 2) for b in ("24", "168", "none", "other")) + " |",
              "| robust_v1 (算子端) | " + " | ".join(
                  _fmt(p["robust_v1_share"][b], 2) for b in ("24", "168", "none", "other")) + " |"]
        c = r["report3_acf"]
        L += ["", f"**③ ACF** acf24 mean/median = {_fmt(c['acf24']['mean'])}/{_fmt(c['acf24']['median'])}；"
              f"acf168 = {_fmt(c['acf168']['mean'])}/{_fmt(c['acf168']['median'])}；"
              f"|acf24|>|acf168| 占比 = {_fmt(c['share_abs_acf24_gt_abs_acf168'], 2)}"]
        j = r["report4_judge48"]["summary"]
        L += ["", f"**④ 48-lookback 判官**（pool {j['n_windows_pooled']} 窗，stride {STRIDE_PROBE}）", "",
              "| λ | judge mean nRMSE | snaive24 mean | judge/sn24 | 胜率 vs sn24 | snaive168 mean | judge/sn168 | 胜率 vs sn168 |",
              "|---|---|---|---|---|---|---|---|"]
        for lam in LAMBDAS:
            s = j[f"lam{lam:g}"]
            L += [f"| {lam:g} | {_fmt(s['mean_nrmse_judge'])} | {_fmt(s['mean_nrmse_snaive24'])} | "
                  f"{_fmt(s['ratio_mean_judge_over_snaive24'], 3)} | {_fmt(s['winrate_judge_vs_snaive24'], 2)} | "
                  f"{_fmt(s['mean_nrmse_snaive168'])} | {_fmt(s['ratio_mean_judge_over_snaive168'], 3)} | "
                  f"{_fmt(s['winrate_judge_vs_snaive168'], 2)} |"]
        t = r["report5_truncation"]
        if t.get("status") == "LOAD_FAILED":
            L += ["", f"**⑤ 截断影响** 4096 重载失败：{t.get('error')}"]
        else:
            L += ["", f"**⑤ 截断影响**（last-4096 vs last-1024，n={t['n_matched']}）"
                  f" legacy 分桶变化占比 = {_fmt(t['legacy_bucket_change_share'], 2)}；"
                  f"robust = {_fmt(t['robust_bucket_change_share'], 2)}；"
                  f"tail-1024 字节不一致 {t['tail1024_mismatch_count']} 条（NaN 填充边界效应）"]
        v = r["verdict"]
        L += ["", f"**准入判决：`{v['verdict']}`**（主导周期≤48 占比 {_fmt(v['share_dominant_period_le48'], 2)}；"
              f"168 主导占比 {_fmt(v['share_168_dominant'], 2)}）", ""]
    # 首选推荐：PASS 者中 share_le48 高者优先；平手看 λ=1 判官 vs snaive24 胜率
    cand = []
    for cfg, r in results.items():
        if r.get("verdict", {}).get("verdict") == "PASS_HEADLINE_U":
            wr = r["report4_judge48"]["summary"]["lam1"]["winrate_judge_vs_snaive24"] or 0.0
            cand.append((r["verdict"]["share_dominant_period_le48"], wr, cfg))
    L += ["## 结论与首选推荐", ""]
    if cand:
        cand.sort(reverse=True)
        L += [f"- **首选 headline-U 候选：`{cand[0][2]}`**"
              f"（主导周期≤48 占比 {_fmt(cand[0][0], 2)}，λ=1 判官 vs snaive24 胜率 {_fmt(cand[0][1], 2)}）。"]
        for sh, wr, cfg in cand[1:]:
            L += [f"- `{cfg}` 同为 PASS（占比 {_fmt(sh, 2)}，胜率 {_fmt(wr, 2)}），列为次选。"]
    else:
        L += ["- 无 config 通过 PASS 判则——两者按各自判决处理（STRESS/AMBIGUOUS），headline-U 需另觅。"]
    for cfg, r in results.items():
        v = r.get("verdict", {}).get("verdict", r.get("status"))
        if v == "STRESS_COMPOUND_SEASONALITY":
            L += [f"- `{cfg}` 降级为 compound-seasonality stress test 候选。"]
        elif r.get("status") == "LOAD_FAILED":
            L += [f"- `{cfg}` **无法经项目 pinned loader 获取**（{r.get('error')}）——"
                  "在不改数据路径的前提下不可作为 U 候选（改路径属 P6 决策，非本探针裁量）。"]
    L += ["- 被探针 uid（各 config 24 条）见对应 JSON `excluded_uids_for_final_U`，**必须排除出最终 U 集**。",
          "- 不建议修改 L/H（不在裁量范围）。", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for cfg in CONFIGS:
        print(f"[probe] {cfg} ...")
        r = probe_config(cfg)
        results[cfg] = r
        p = OUT_DIR / f"u_admission_{cfg}.json"
        p.write_text(json.dumps(r, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        print(f"[probe] {cfg}: status={r.get('status')} verdict="
              f"{r.get('verdict', {}).get('verdict', '-')} -> {p}")
    write_report_md(results, OUT_DIR / "u_admission_report.md")
    print(f"[probe] report -> {OUT_DIR / 'u_admission_report.md'}")


if __name__ == "__main__":
    main()
