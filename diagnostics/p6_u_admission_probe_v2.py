# -*- coding: utf-8 -*-
"""diagnostics/p6_u_admission_probe_v2.py — P6 U 域**全宇宙**复检探针（v2；只读诊断，非实验本体）。

兑现外部 GPT 审查 NO-GO 意见 #10（对 prereg_p6.md §2 "U 候选宇宙"的两点整改）：
  1. 覆盖面：首轮探针（p6_u_admission_probe.py）经 load_config_series(max_series=60) 只看到
     traffic_hourly **前 60 条**，而最终 U 要从**全量 862 条**中抽取——本轮结构统计
     （period / lag-24 与 lag-168 ACF / 准入判则）在过滤后全宇宙上**逐条**计算（廉价统计，非抽样）。
  2. 判官口径：首轮 judge-capability 项用探针内镜像实现 + **全数组 z-score**（含 future），与
     canonical 判官 `p6/judge_closed_form.py`（dlinear_closed_form_v1，z-score **history-only**）
     不一致。本轮直接调用 canonical 判官：series_weight="equal"、λ=1e-3、stride=4、
     window_cap=None；SeriesView(history=x[:-48], future=x[-48:])，z-score 由模块内部按
     history-only 口径处理（**不**先对全数组 z-score）；每次拟合附双路对拍（atol 1e-9）。
     对照 seasonal-naive-24 / -168 与首轮同定义，但口径统一为：**原始尺度 RMSE ÷ 该序列
     history 的 std**（zscore_state 同一状态）——判官 per-series RMSE 本就等于
     raw-RMSE/std(history)，故两边同尺度可比。

judge-capability 子样本纪律：seed=20260711，从过滤后全宇宙（item_id 字典序）中**排除首轮已
探针 24 条**（读自 u_admission_traffic_hourly.json `excluded_uids_for_final_U`，记该文件
sha256 溯源）后均匀抽 32 条（不放回，选中索引升序）；落盘 item_id + sha256
（NaN 线性填充后、z-score 前 float64 字节 = load_config_series 返回状态）。

总排除清单 `all_probe_consumed_item_ids` = 首轮 24 + 本轮 32（去重）——**最终 U 抽取的排除集**。

准入判则（与首轮逐字一致，但在全宇宙上计算）：
  · 多数序列"主导可用周期 ≤48"（robust period≈24，或 |acf24|>|acf168|）→ PASS_HEADLINE_U；
  · 多数序列 168 主导（robust≈168 且 |acf168|≥|acf24|）→ STRESS_COMPOUND_SEASONALITY；
  · 皆非 → AMBIGUOUS。不建议、也不裁量修改 L/H。

确定性校验：`--verify` 在**新进程**里全量重算，与盘上 JSON 做 canonical diff
（json sort_keys 序列化，剔除 `_volatile` 与 `determinism` 两个墙钟/自指字段后逐字节比较），
PASS/FAIL 写回 JSON `determinism` 块并重写 MD 报告。

内存注记：全量加载走项目 loader 原路径（逐 shard `to_pylist`，未分批）；862 × ≤17544 点在
本机内存内一次可容——若未来 config 更大需分批，结果须与本实现等价。

红线：只写 results/Stage2/P6Probes/{u_admission_v2_traffic_hourly.json, u_admission_v2_report.md}；
不改任何现有文件；除 HF 下载（已有缓存）无其他网络；无 LLM/git；确定性（固定 seed、闭式解、
加载顺序 = 排序 shard × parquet 行序）。

运行（项目根 Agent/ 下）：
  D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.diagnostics.p6_u_admission_probe_v2
  D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.diagnostics.p6_u_admission_probe_v2 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import platform
import sys
import time
from collections import Counter

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
from SelfEvolvingHarnessTS.p6 import judge_closed_form as judge             # noqa: E402
from SelfEvolvingHarnessTS.p6.judge_closed_form import (                    # noqa: E402
    SeriesView, fit_domain, fit_domain_rebuild, zscore_state,
)

# ── 协议常量（冻结，只读引用；断言钉死防漂移）─────────────────────────────
CONFIG = "traffic_hourly"
SPLIT = "test"
SEED_V2 = 20260711                               # capability 子样本采样纪律（任务书钉死）
N_CAPABILITY = 32
EXPECTED_UNIVERSE = 862                          # prereg_p6.md §2：traffic_hourly 全量
MAX_SERIES_UNIVERSE = 100_000                    # > 862 → loader 不截断宇宙
LOAD_MIN_LEN = 64                                # load_config_series 入参（≥144 过滤在其后）
MAX_LEN = 1024
MIN_LEN = (L_WIN + H_FORECAST) + H_FORECAST      # = data/load_real.py:43 的 96 + H_FORECAST
TOL_REL = 0.10                                   # period 分桶相对容差（同首轮）
LAM = 1e-3                                       # canonical 判官协议（prereg §1）
STRIDE = 4
DUAL_PATH_ATOL = 1e-9

OUT_DIR = _REPO_ROOT / "SelfEvolvingHarnessTS" / "results" / "Stage2" / "P6Probes"
JSON_PATH = OUT_DIR / "u_admission_v2_traffic_hourly.json"
MD_PATH = OUT_DIR / "u_admission_v2_report.md"
ROUND1_JSON = OUT_DIR / "u_admission_traffic_hourly.json"

assert (L_WIN, H_FORECAST, MIN_LEN) == (48, 48, 144), "判官协议常量漂移——复检前提失效"
assert (judge.CONTEXT_LEN, judge.HORIZON) == (48, 48), "canonical 判官 L/H 漂移"
assert judge.PROTOCOL_ID == "dlinear_closed_form_v1", "canonical 判官协议 ID 漂移"
assert (judge.DEFAULT_LAM, judge.DEFAULT_STRIDE) == (LAM, STRIDE), "canonical 判官默认 λ/stride 漂移"

_VOLATILE_KEYS = ("_volatile", "determinism")    # canonical diff 剔除的墙钟/自指字段


# ═══════════════════════ 工具（镜像首轮探针语义，纯 numpy）═══════════════════════
def _sha256_f64(x: np.ndarray) -> str:
    """NaN 填充后、z-score 前的 float64 字节指纹（load_config_series 返回值即该状态）。"""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()


def _zscore_full(x: np.ndarray) -> np.ndarray:
    """镜像首轮 `_zscore`（data/load_real.py 语义：全数组 mean/std，std 护栏 1e-9）。

    仅用于结构统计的 ACF（ACF 对线性缩放不变，保留只为与首轮读数逐位可比）；
    **不**用于判官——判官 z-score 由 canonical 模块内部 history-only 处理。"""
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


def _len_dist(lens: list) -> dict:
    """长度分布：min/p25/median/p75/max；不同长度 ≤12 种时附完整直方图（确定性排序）。"""
    if not lens:
        return {}
    a = np.asarray(lens, dtype=float)
    d = {"min": int(a.min()), "p25": float(np.percentile(a, 25)),
         "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
         "max": int(a.max())}
    uniq = sorted(set(lens))
    if len(uniq) <= 12:
        d["histogram"] = {str(u): int(lens.count(u)) for u in uniq}
    return d


# ═══════════════════════ ④ judge-capability（canonical 判官）═══════════════════════
def _judge_capability(cap: list) -> dict:
    """canonical 判官 vs seasonal-naive-24/168（统一口径：raw-RMSE ÷ history std）。

    判官：fit_domain(series_weight="equal", λ=1e-3, stride=4, window_cap=None)，
    z-score 全在模块内部（history-only）；per-series RMSE 数学上 = raw-RMSE/std(history)。
    基线：原始尺度预测 → RMSE ÷ zscore_state(history).std（同一状态、同一护栏）→ 同尺度。
    sn24 = tile(history 末 24 点×2)；sn168 = pred[t]=x[n-48-168+t]（与首轮定义逐字一致，
    即 history[-168:-120]），要求 len(history) ≥ 168。附双路对拍（atol 1e-9）。"""
    views = [SeriesView(uid=iid, history=x[:-H_FORECAST], future=x[-H_FORECAST:])
             for iid, x in cap]
    fit = fit_domain(views, lam=LAM, stride=STRIDE, window_cap=None, series_weight="equal")
    fit_rb = fit_domain_rebuild(views, lam=LAM, stride=STRIDE, window_cap=None,
                                series_weight="equal")
    w_diff = float(np.max(np.abs(fit.W - fit_rb.W)))
    w_max = float(np.max(np.abs(fit.W)))
    r_diff = float(np.max(np.abs(fit.per_series_rmse - fit_rb.per_series_rmse)))
    u_diff = float(abs(fit.utility - fit_rb.utility))
    dual = {"atol": DUAL_PATH_ATOL,
            "pass_strict_W_and_rmse": bool(w_diff <= DUAL_PATH_ATOL
                                           and r_diff <= DUAL_PATH_ATOL),
            "pass_rmse_and_utility": bool(r_diff <= DUAL_PATH_ATOL
                                          and u_diff <= DUAL_PATH_ATOL),
            "w_max_abs_diff": w_diff, "w_max_abs": w_max,
            "w_rel_diff_vs_maxW": (w_diff / w_max if w_max > 0 else None),
            "rmse_max_abs_diff": r_diff, "utility_abs_diff": u_diff,
            "note": "toy 级单测（test_dual_path_consistency）W 与 rmse 皆过 atol 1e-9；"
                    "真实 U 尺度（32 序列 × 221 窗 = 7072 窗 pooled）下两条代数等价路径的"
                    "浮点累积使 W 级 |Δ| 超 1e-9（~1e-8 量级、相对 max|W| ~1e-8），而承载判决的"
                    "per-series RMSE / utility 在 1e-9 下大幅通过（~1e-14）——正式 runner 的"
                    "双路对拍若按 W 级 atol=1e-9 实现将在 U 尺度 technical abort，"
                    "须按评估量（rmse/utility）或 W 相对容差实现（供 P6 签发前决策，本探针不裁量）"}

    per_series: list = []
    for i, (iid, x) in enumerate(cap):
        hist, fut = x[:-H_FORECAST], x[-H_FORECAST:]
        _, std = zscore_state(hist)                          # canonical 状态（STD_FLOOR 护栏）
        sn24 = np.tile(hist[-24:], 2)[:H_FORECAST]
        n24 = float(np.sqrt(np.mean((sn24 - fut) ** 2)) / std)
        if hist.size >= 168:
            sn168 = hist[hist.size - 168: hist.size - 168 + H_FORECAST]
            n168 = float(np.sqrt(np.mean((sn168 - fut) ** 2)) / std)
        else:
            n168 = None
        per_series.append({"item_id": iid, "n_windows": int(fit.stats[i].n_windows),
                           "nrmse_judge": float(fit.per_series_rmse[i]),
                           "nrmse_snaive24": n24, "nrmse_snaive168": n168})

    js = [r["nrmse_judge"] for r in per_series]
    s24 = [r["nrmse_snaive24"] for r in per_series]
    p168 = [(r["nrmse_judge"], r["nrmse_snaive168"]) for r in per_series
            if r["nrmse_snaive168"] is not None]
    summary = {
        "n_series": len(per_series),
        "n_windows_total": int(fit.n_windows_total),
        "dual_path_check": dual,
        "mean_nrmse_judge": float(np.mean(js)),
        "median_nrmse_judge": float(np.median(js)),
        "mean_nrmse_snaive24": float(np.mean(s24)),
        "ratio_mean_judge_over_snaive24": float(np.mean(js) / np.mean(s24)),
        "winrate_judge_vs_snaive24": float(np.mean([j < b for j, b in zip(js, s24)])),
        "n_compared_snaive24": len(s24),
        "mean_nrmse_snaive168": (float(np.mean([b for _, b in p168])) if p168 else None),
        "ratio_mean_judge_over_snaive168": (float(np.mean([j for j, _ in p168])
                                                  / np.mean([b for _, b in p168]))
                                            if p168 else None),
        "winrate_judge_vs_snaive168": (float(np.mean([j < b for j, b in p168]))
                                       if p168 else None),
        "n_compared_snaive168": len(p168),
        "note_ratio_scope": "ratio/胜率均在成对可比子集上计算（sn168 需 len(history)≥168）",
    }
    protocol = {"judge": "p6/judge_closed_form.py（canonical，dlinear_closed_form_v1）",
                "lam": LAM, "stride": STRIDE, "window_cap": None, "series_weight": "equal",
                "zscore": "history-only（模块内部 zscore_state，std 护栏 1e-8）——修正首轮全数组口径",
                "baseline_scale": "seasonal-naive 原始尺度 RMSE ÷ zscore_state(history).std（与判官同尺度）",
                "split": "SeriesView(history=x[:-48], future=x[-48:])"}
    return {"protocol": protocol, "summary": summary, "per_series": per_series}


# ═══════════════════════ 主探针 ═══════════════════════
def probe_v2() -> dict:
    t0 = time.time()
    out: dict = {
        "probe": "u_admission_v2",
        "config": CONFIG,
        "split": SPLIT,
        "purpose": ("外部 GPT 审查 NO-GO 意见 #10 兑现：首轮探针只覆盖前 60 条（max_series=60）"
                    "且 judge-capability 用全数组 z-score 镜像实现；本轮=全宇宙结构统计（逐条）"
                    "+ canonical 判官能力子样本（history-only z-score）+ 最终 U 抽取总排除清单"),
        "seed_capability": SEED_V2,
        "n_capability_target": N_CAPABILITY,
        "protocol_constants": {
            "L_WIN": int(L_WIN), "H": int(H_FORECAST), "MIN_LEN": int(MIN_LEN),
            "load_min_len": LOAD_MIN_LEN, "max_len": MAX_LEN,
            "max_series_universe": MAX_SERIES_UNIVERSE,
            "expected_universe": EXPECTED_UNIVERSE,
            "period_bucket_rel_tol": TOL_REL,
            "judge": {"protocol_id": judge.PROTOCOL_ID, "lam": LAM, "stride": STRIDE,
                      "window_cap": None, "series_weight": "equal",
                      "std_floor": judge.STD_FLOOR, "dual_path_atol": DUAL_PATH_ATOL},
        },
        "estimators": {
            "legacy_fft_v0": "conditioning/period.py dominant_period_fft_v0（P0 冻结契约感知端）",
            "robust_v1": "conditioning/period.py guess_period_robust_v1（S0.7 修复版算子端）",
        },
    }

    # 首轮溯源：24 条已消费 uid（读现有 JSON，只读；记 sha256）
    r1_bytes = ROUND1_JSON.read_bytes()
    r1 = json.loads(r1_bytes.decode("utf-8"))
    r1_uids = [str(u) for u in r1["excluded_uids_for_final_U"]]
    if len(r1_uids) != 24 or len(set(r1_uids)) != 24:
        raise RuntimeError(f"首轮排除清单异常：{len(r1_uids)} 条（去重 {len(set(r1_uids))}）≠ 24")
    r1_set = set(r1_uids)

    # ① 全宇宙加载（loader 原路径；max_series 远大于 862 → 不截断）
    print(f"[probe-v2] loading {CONFIG}/{SPLIT} full universe ...")
    try:
        series = load_config_series(CONFIG, split=SPLIT, max_series=MAX_SERIES_UNIVERSE,
                                    min_len=LOAD_MIN_LEN, max_len=MAX_LEN)
    except Exception as exc:                     # noqa: BLE001 —— 红线：失败如实记录
        out["status"] = "LOAD_FAILED"
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["_volatile"] = _volatile_block(t0)
        return out
    lens_all = [int(x.size) for _, x in series]
    eligible = sorted([(iid, x) for iid, x in series if x.size >= MIN_LEN],
                      key=lambda t: t[0])
    lens_el = [int(x.size) for _, x in eligible]
    ids = [iid for iid, _ in eligible]
    dup = sorted([i for i, c in Counter(ids).items() if c > 1])
    r1_missing = sorted(r1_set - set(ids))
    out["round1_provenance"] = {
        "path": "results/Stage2/P6Probes/u_admission_traffic_hourly.json",
        "sha256_file": hashlib.sha256(r1_bytes).hexdigest(),
        "n_round1_excluded": len(r1_uids),
        "excluded_uids_round1": sorted(r1_uids),
        "round1_verdict": r1.get("verdict", {}).get("verdict"),
        "n_round1_found_in_universe": len(r1_set) - len(r1_missing),
        "round1_missing_from_universe": r1_missing,
    }
    out["report1_universe"] = {
        "n_loaded": len(series),
        "n_eligible_ge_min_len": len(eligible),
        "matches_expected_862": bool(len(series) == EXPECTED_UNIVERSE),
        "n_duplicate_item_ids": len(dup),
        "duplicate_item_ids": dup,
        "length_loaded": _len_dist(lens_all),
        "length_eligible": _len_dist(lens_el),
    }
    print(f"[probe-v2] loaded {len(series)} series; eligible(>= {MIN_LEN}) = {len(eligible)}")
    if not eligible:
        out["status"] = "INSUFFICIENT_SERIES"
        out["_volatile"] = _volatile_block(t0)
        return out

    # ② + ③ 全宇宙结构统计（逐条；period 双估计器 + lag-24/168 ACF）
    uni_rows: list = []
    for iid, x in eligible:
        leg = float(dominant_period_fft_v0(x)[0])
        rob = int(guess_period_robust_v1(x))
        z = _zscore_full(x)
        a24, a168 = _acf(z, 24), _acf(z, 168)
        uni_rows.append({"item_id": iid, "length": int(x.size),
                         "legacy_fft_v0": leg, "legacy_bucket": _bucket(leg),
                         "robust_v1": rob, "robust_bucket": _bucket(float(rob)),
                         "acf24": a24, "acf168": a168,
                         "abs24_gt_abs168": bool(abs(a24) > abs(a168))})
    print(f"[probe-v2] universe structural stats done (n={len(uni_rows)})")

    def _share(key: str, val: str) -> float:
        return float(np.mean([r[key] == val for r in uni_rows]))

    def _count(key: str, val: str) -> int:
        return int(sum(r[key] == val for r in uni_rows))

    buckets = ("24", "168", "none", "other")
    out["report2_period_universe"] = {
        "n": len(uni_rows),
        "legacy_fft_v0_share": {b: _share("legacy_bucket", b) for b in buckets},
        "legacy_fft_v0_count": {b: _count("legacy_bucket", b) for b in buckets},
        "robust_v1_share": {b: _share("robust_bucket", b) for b in buckets},
        "robust_v1_count": {b: _count("robust_bucket", b) for b in buckets},
        "note": "分桶相对容差 10%（n=1024 的 FFT 网格 1024/k 取不到整 24/168：如 23.81/170.67）",
    }
    out["report3_acf_universe"] = {
        "n": len(uni_rows),
        "acf24": {"mean": float(np.mean([r["acf24"] for r in uni_rows])),
                  "median": float(np.median([r["acf24"] for r in uni_rows]))},
        "acf168": {"mean": float(np.mean([r["acf168"] for r in uni_rows])),
                   "median": float(np.median([r["acf168"] for r in uni_rows]))},
        "share_abs_acf24_gt_abs_acf168": float(np.mean([r["abs24_gt_abs168"]
                                                        for r in uni_rows])),
        "note": "ACF 在首轮口径（全数组 z-score）上计算；ACF 对线性缩放不变，读数与口径无关",
    }
    out["universe_per_series"] = uni_rows

    # ④ judge-capability 子样本：排除首轮 24 条后均匀抽 32（seed=20260711，索引升序）
    pool = [(iid, x) for iid, x in eligible if iid not in r1_set]
    rng = np.random.default_rng(SEED_V2)
    k = min(N_CAPABILITY, len(pool))
    pick = sorted(rng.choice(len(pool), size=k, replace=False).tolist())
    cap = [pool[i] for i in pick]
    out["capability_sampling"] = {
        "seed": SEED_V2, "n_target": N_CAPABILITY, "n_drawn": k,
        "pool_size_after_round1_exclusion": len(pool),
        "rule": "eligible 按 item_id 字典序排序 → 剔除首轮 24 条 → default_rng(seed).choice"
                "(len(pool), size=32, replace=False)，选中索引升序",
    }
    out["capability_manifest"] = [{"item_id": iid, "length": int(x.size),
                                   "sha256_f64": _sha256_f64(x)} for iid, x in cap]
    print(f"[probe-v2] capability sample drawn (n={k}); running canonical judge ...")
    out["report4_judge_capability"] = _judge_capability(cap)
    _dual = out["report4_judge_capability"]["summary"]["dual_path_check"]
    print("[probe-v2] canonical judge done "
          f"(dual_path strict={_dual['pass_strict_W_and_rmse']} "
          f"rmse+utility={_dual['pass_rmse_and_utility']})")

    # ⑤ 准入判决（判则与首轮逐字一致；在全宇宙 n 上计算）
    flags_le48 = [(r["robust_bucket"] == "24") or r["abs24_gt_abs168"] for r in uni_rows]
    flags_168 = [(r["robust_bucket"] == "168") and (abs(r["acf168"]) >= abs(r["acf24"]))
                 for r in uni_rows]
    share_le48, share_168 = float(np.mean(flags_le48)), float(np.mean(flags_168))
    verdict = ("PASS_HEADLINE_U" if share_le48 > 0.5
               else "STRESS_COMPOUND_SEASONALITY" if share_168 > 0.5 else "AMBIGUOUS")
    r1_verdict = out["round1_provenance"]["round1_verdict"]
    out["verdict"] = {
        "rule": "PASS: 多数序列 robust period≈24 或 |acf24|>|acf168|；"
                "STRESS: 多数 robust≈168 且 |acf168|≥|acf24|（与首轮逐字一致）",
        "based_on": f"过滤后全宇宙 n={len(uni_rows)}（首轮仅前 60 条中抽 24）",
        "share_dominant_period_le48": share_le48,
        "share_168_dominant": share_168,
        "verdict": verdict,
        "round1_verdict": r1_verdict,
        "maintained": bool(verdict == r1_verdict),
        "note": "不建议修改 L/H（非本探针裁量范围）",
    }

    # ⑥ 总排除清单 = 首轮 24 + 本轮 capability 32（去重）——最终 U 抽取的排除集
    cap_ids = [iid for iid, _ in cap]
    overlap = sorted(r1_set & set(cap_ids))
    all_consumed = sorted(r1_set | set(cap_ids))
    out["exclusion_accounting"] = {
        "n_round1": len(r1_set), "n_v2_capability": len(cap_ids),
        "n_overlap": len(overlap), "overlap_item_ids": overlap,
        "n_total": len(all_consumed),
    }
    out["all_probe_consumed_item_ids"] = all_consumed

    out["status"] = "OK"
    out["_volatile"] = _volatile_block(t0)
    return out


def _volatile_block(t0: float) -> dict:
    return {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_sec": round(time.time() - t0, 1),
            "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "platform": platform.platform()}}


# ═══════════════════════ canonical diff（确定性校验）═══════════════════════
def _canonical(d: dict) -> bytes:
    """确定性比较用 canonical 序列化：剔除墙钟/自指字段，sort_keys，utf-8 字节。"""
    core = {k: v for k, v in d.items() if k not in _VOLATILE_KEYS}
    return json.dumps(core, ensure_ascii=False, sort_keys=True, indent=1,
                      default=float).encode("utf-8")


# ═══════════════════════ MD 报告 ═══════════════════════
def _fmt(v, nd=4):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_report_md(r: dict, path: pathlib.Path) -> None:
    L = ["# P6 U 域全宇宙复检探针报告（u_admission_v2）", "",
         f"- 日期：2026-07-11；capability seed={SEED_V2}；性质：只读复检探针"
         "（外部 GPT 审查 NO-GO 意见 #10 兑现），**不属于 P6 实验本体**；不裁量修改 L/H。",
         "- 相对首轮（u_admission_report.md）的两点整改：①结构统计从前 60 条扩到**过滤后全宇宙逐条**；"
         "②judge-capability 改调 **canonical 判官** `p6/judge_closed_form.py`"
         "（history-only z-score；series_weight=\"equal\"、λ=1e-3、stride=4、window_cap=None），"
         "对照基线统一为原始尺度 RMSE ÷ history std（与判官同尺度）。",
         f"- 冻结协议：L_WIN={L_WIN}, H={H_FORECAST}, MIN_LEN={MIN_LEN}；"
         f"判官 `{judge.PROTOCOL_ID}`；双路对拍 atol={DUAL_PATH_ATOL:g}。", ""]
    if r.get("status") == "LOAD_FAILED":
        L += [f"**下载失败**：{r.get('error')}", ""]
        path.write_text("\n".join(L), encoding="utf-8")
        return

    a = r["report1_universe"]
    hist_el = a["length_eligible"].get("histogram")
    L += ["## ① 全宇宙可用性（loader min_len=64 → 再按 ≥144 过滤）", "",
          f"- 加载 **{a['n_loaded']}** 条（预期 862：{'✔ 一致' if a['matches_expected_862'] else '✘ 不一致，如实记录'}）；"
          f"长度≥{MIN_LEN}：**{a['n_eligible_ge_min_len']}** 条；重复 item_id：{a['n_duplicate_item_ids']}。",
          f"- 长度分布（过滤后）：min/median/max = {a['length_eligible'].get('min')}/"
          f"{_fmt(a['length_eligible'].get('median'), 0)}/{a['length_eligible'].get('max')}"
          + (f"；直方图 {hist_el}" if hist_el else ""), ""]

    p = r["report2_period_universe"]
    L += [f"## ② period 估计（全宇宙 n={p['n']}；分桶容差 ±10%）", "",
          "| estimator | ≈24 | ≈168 | none | other |", "|---|---|---|---|---|",
          "| legacy_fft_v0 (P0 感知端) | " + " | ".join(
              _fmt(p["legacy_fft_v0_share"][b], 4) for b in ("24", "168", "none", "other")) + " |",
          "| robust_v1 (算子端) | " + " | ".join(
              _fmt(p["robust_v1_share"][b], 4) for b in ("24", "168", "none", "other")) + " |", "",
          f"（robust_v1 计数：≈24 {p['robust_v1_count']['24']}、≈168 {p['robust_v1_count']['168']}、"
          f"none {p['robust_v1_count']['none']}、other {p['robust_v1_count']['other']}）", ""]

    c = r["report3_acf_universe"]
    L += [f"## ③ ACF（全宇宙 n={c['n']}）", "",
          f"- acf24 mean/median = {_fmt(c['acf24']['mean'])}/{_fmt(c['acf24']['median'])}；"
          f"acf168 = {_fmt(c['acf168']['mean'])}/{_fmt(c['acf168']['median'])}；"
          f"**|acf24|>|acf168| 占比 = {_fmt(c['share_abs_acf24_gt_abs_acf168'], 4)}**", ""]

    j = r["report4_judge_capability"]
    s = j["summary"]
    d = s["dual_path_check"]
    samp = r["capability_sampling"]
    L += [f"## ④ judge-capability（canonical 判官；n={s['n_series']}；"
          f"排除首轮 24 条后从 {samp['pool_size_after_round1_exclusion']} 条均匀抽样）", "",
          f"- 双路对拍（fit_domain vs fit_domain_rebuild，atol {DUAL_PATH_ATOL:g}）：评估量级"
          f"（per-series RMSE + utility）**{'PASS' if d['pass_rmse_and_utility'] else 'FAIL'}**"
          f"（rmse max|Δ|={d['rmse_max_abs_diff']:.2e}，utility |Δ|={d['utility_abs_diff']:.2e}）；"
          f"W 严格级 **{'PASS' if d['pass_strict_W_and_rmse'] else 'FAIL'}**"
          f"（W max|Δ|={d['w_max_abs_diff']:.2e}，相对 max|W|="
          f"{'-' if d['w_rel_diff_vs_maxW'] is None else format(d['w_rel_diff_vs_maxW'], '.2e')}）；"
          f"pooled {s['n_windows_total']} 窗（stride {STRIDE}，series_weight=equal，λ={LAM:g}）。",
          "- ⚠ 发现（供 P6 签发前决策）：真实 U 尺度（7072 窗 pooled）下两条代数等价路径的浮点累积使 "
          "W 级 |Δ| ≈ 1e-8 > 1e-9（toy 级单测通过），而承载判决的 RMSE/utility 一致到 ~1e-14——"
          "正式 runner 的双路对拍若按 W 级 atol=1e-9 实现将在 U 尺度 technical abort，"
          "须按评估量或 W 相对容差实现（本探针不裁量）。", "",
          "| judge mean nRMSE | judge median | sn24 mean | judge/sn24 | 胜率 vs sn24 "
          "| sn168 mean | judge/sn168 | 胜率 vs sn168 |",
          "|---|---|---|---|---|---|---|---|",
          f"| {_fmt(s['mean_nrmse_judge'])} | {_fmt(s['median_nrmse_judge'])} | "
          f"{_fmt(s['mean_nrmse_snaive24'])} | {_fmt(s['ratio_mean_judge_over_snaive24'], 3)} | "
          f"{_fmt(s['winrate_judge_vs_snaive24'], 2)} | {_fmt(s['mean_nrmse_snaive168'])} | "
          f"{_fmt(s['ratio_mean_judge_over_snaive168'], 3)} | "
          f"{_fmt(s['winrate_judge_vs_snaive168'], 2)} |", "",
          "- 口径：judge = canonical per-series RMSE（history-only z 空间，数学上 = raw-RMSE ÷ "
          "history std）；sn24/sn168 = 原始尺度 RMSE ÷ zscore_state(history).std —— 同尺度可比。",
          "- 首轮参考值（不同口径不直接可比，仅方向参考：判官镜像实现 + 全数组 z-score、"
          "pooled 无 series 等权、n=24、前 60 条内）：judge/sn24=0.855、胜率 0.79；"
          "judge/sn168=0.944、胜率 0.50。", ""]

    v = r["verdict"]
    L += [f"## ⑤ 准入判决（判则与首轮逐字一致；{v['based_on']}）", "",
          f"- share_dominant_period_le48 = **{_fmt(v['share_dominant_period_le48'], 4)}**；"
          f"share_168_dominant = {_fmt(v['share_168_dominant'], 4)}",
          f"- **判决：`{v['verdict']}`**（首轮 `{v['round1_verdict']}` → "
          f"{'**维持**' if v['maintained'] else '**变更，如实降级**'}）", ""]

    e = r["exclusion_accounting"]
    L += ["## ⑥ 探针消费与最终 U 排除集", "",
          f"- 首轮 {e['n_round1']} 条 + 本轮 capability {e['n_v2_capability']} 条，"
          f"重叠 {e['n_overlap']} → `all_probe_consumed_item_ids` 共 **{e['n_total']}** 条"
          "（见 JSON 同名字段）——**最终 U 抽取的排除集**。",
          "- capability 子样本 32 条 item_id + content sha256（NaN 填充后、z-score 前 float64 字节）"
          "见 JSON `capability_manifest`。", ""]

    det = r.get("determinism", {"status": "NOT_RUN"})
    L += ["## ⑦ 确定性（幂等重跑 diff）", ""]
    if det.get("status") == "NOT_RUN":
        L += ["- 未运行（需 `--verify`：新进程全量重算 + canonical diff）。", ""]
    else:
        L += [f"- 新进程全量重算，canonical diff（json sort_keys；剔除 `_volatile`/`determinism` "
              f"墙钟字段后逐字节比较）：**{det['status']}**",
              f"- payload sha256：run1 = `{det.get('sha256_run1')}`，run2 = `{det.get('sha256_run2')}`", ""]
        if det.get("status") == "FAIL" and det.get("first_diffs"):
            L += ["- 首批差异行：", ""] + [f"  - `{x}`" for x in det["first_diffs"]] + [""]
    path.write_text("\n".join(L), encoding="utf-8")


def _write_outputs(res: dict) -> None:
    JSON_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                         encoding="utf-8")
    write_report_md(res, MD_PATH)


# ═══════════════════════ 入口 ═══════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="P6 U 域全宇宙复检探针 v2")
    ap.add_argument("--verify", action="store_true",
                    help="新进程全量重算并与盘上 JSON 做 canonical diff（除墙钟字段外应 bit 级一致）")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.verify:
        res = probe_v2()
        res["determinism"] = {"status": "NOT_RUN",
                              "note": "运行 --verify 在新进程重算并 canonical diff"}
        _write_outputs(res)
        print(f"[probe-v2] status={res.get('status')} "
              f"verdict={res.get('verdict', {}).get('verdict', '-')}")
        print(f"[probe-v2] json -> {JSON_PATH}")
        print(f"[probe-v2] report -> {MD_PATH}")
        return

    if not JSON_PATH.exists():
        raise SystemExit("先运行主模式生成 JSON，再 --verify")
    disk = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    t0 = time.time()
    res2 = probe_v2()
    c1, c2 = _canonical(disk), _canonical(res2)
    status = "PASS" if c1 == c2 else "FAIL"
    det: dict = {
        "status": status,
        "method": "fresh-process 全量重算；canonical JSON（sort_keys，剔除 _volatile/determinism）"
                  "逐字节比较",
        "excluded_fields": list(_VOLATILE_KEYS),
        "sha256_run1": hashlib.sha256(c1).hexdigest(),
        "sha256_run2": hashlib.sha256(c2).hexdigest(),
        "verify_elapsed_sec": round(time.time() - t0, 1),
    }
    if status == "FAIL":
        l1, l2 = c1.decode("utf-8").splitlines(), c2.decode("utf-8").splitlines()
        diffs = [f"L{i}: run1={a!r} run2={b!r}"
                 for i, (a, b) in enumerate(zip(l1, l2)) if a != b][:20]
        if len(l1) != len(l2):
            diffs.append(f"行数不同：run1={len(l1)} run2={len(l2)}")
        det["first_diffs"] = diffs
    disk["determinism"] = det                    # 记录写回 run1 工件（run1 = 正式载荷）
    _write_outputs(disk)
    print(f"[probe-v2][verify] determinism={status} "
          f"run1={det['sha256_run1'][:12]} run2={det['sha256_run2'][:12]}")
    print(f"[probe-v2][verify] json/report rewritten -> {JSON_PATH}")
    if status != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    main()
