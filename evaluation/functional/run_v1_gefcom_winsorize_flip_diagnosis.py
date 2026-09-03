"""工作包 V1：GEFCom 反例归因——winsorize 跨切片翻转归因（零 LLM，2026-08-08）。

背景：泄漏修复后 GEFCom 上 A5 首探 winsorize（Source 832/880 排序第一）在
Target 928 第一探负（-0.1636），多一次 harm probe。审查裁决（十四）修正归因：
弱引用模式（n_hist=2 < 3 → delta=None → weak_reference，非 radius 匹配）；
同域种子（GEFCom 832/880）→ LOCAL_SEEDED_MEMORY_NEGATIVE_TRANSFER_CANDIDATE。

诊断一（已有）：winsorize 翻转是否由局部缺失驱动（缺失 0 组 vs 缺失>0 组；
GEFCom 缺失是单事件二值 {0,18}，缺失组 n=1，不构成统计）。

诊断二（审查裁决 十四 新增）：**structured_clipping_geometry family**——
winsorize 与 outlier_iqr 在**相同规范 Scope（训练窗口，v6._evaluate 同构）**
下执行，度量全局阈值裁剪族的机制一致 Observation：
  - clip_fraction：被裁剪点比例（clip 后 ≠ interp 后输入的占比）；
  - clip_magnitude：平均裁剪幅度（|out−y|，MAD 归一）；
  - clip_at_edges_fraction：被裁剪点落在窗口前/后 25%（趋势端点区）的占比
    （均匀期望 0.5）；
  - clip_at_peak/trough_fraction：被裁剪点落在日型峰/谷相位 ±3h 带的占比
    （均匀期望 ≈ 7/24 ≈ 0.292）；
  - clip_upper_fraction：上边界裁剪占比（>hi vs <lo 的不对称性）。
对照：winsorize（分位数边界）vs outlier_iqr（IQR 边界）——同族不同边界绑定。

裁决标准（审查）：若几何特征能区分 winsorize 正负切片（832/880/976 正 vs
928 负），同时保留 outlier_iqr @928/976 替代 headroom → 再批准一次单独的
Observation/Scope 修改；若不能区分 → 停止扩 Pattern，记 weak-history 下
不可识别，依赖 Support 验证/abstain，转向 1024+ 空间的链。

用法：
  python evaluation/functional/run_v1_gefcom_winsorize_flip_diagnosis.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

from SelfEvolvingHarnessTS.contracts.program import Program  # noqa: E402
from SelfEvolvingHarnessTS.operators._common import as_1d, interp_nan  # noqa: E402
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline  # noqa: E402

WINDOW = 192
HORIZON = 48
# winsorize（翻转主角）+ GEFCom w2 扫描确认的翻转同类（B+C-：support 正 delayed 负）
OPERATORS = ["winsorize", "denoise_wavelet", "impute_fft", "outlier_mad"]
# structured_clipping_geometry 对照：全局阈值裁剪族（分位数 vs IQR 边界）
CLIP_PAIR = ["winsorize", "outlier_iqr"]
CLIP_ORIGINS = [832, 880, 928, 976]
PHASE_BAND = 3  # 峰/谷相位 ±3h 带
EDGE_FRACTION = 0.25
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_gefcom_winsorize_flip_report.json")


def local_missing_feature(values: Mapping[str, np.ndarray], origin: int) -> dict[str, float]:
    """局部窗口特征：每序列 [origin-WINDOW, origin) 窗口的最大缺失 run
    （复用 run_v1_nn5_local_sensitivity.py 思路：跨序列 median + max + fraction）。"""
    per_series_runs: list[int] = []
    for array in values.values():
        arr = np.asarray(array, dtype=np.float64)
        lo = max(0, origin - WINDOW)
        mask = ~np.isfinite(arr[lo:origin])
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        per_series_runs.append(best)
    return {
        "median_window_max_missing_run": float(statistics.median(per_series_runs)),
        "max_window_max_missing_run": float(max(per_series_runs)),
        "series_with_missing_fraction": float(
            sum(1 for r in per_series_runs if r > 0) / len(per_series_runs)
        ),
    }


def _training_windows(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
    origin: int,
) -> list[tuple[str, int, np.ndarray]]:
    """规范 Scope 训练窗口（与 v6._evaluate 同构：train rows × anchors，
    anchor + HORIZON > origin 跳过）。"""
    windows: list[tuple[str, int, np.ndarray]] = []
    for row in roster:
        if str(row["role"]) != "train":
            continue
        uid = str(row["series_uid"])
        raw = np.asarray(values[uid], dtype=np.float64)
        for anchor in config["anchors"]:  # type: ignore[union-attr]
            anchor = int(anchor)
            if anchor + HORIZON > origin:
                continue
            windows.append((uid, anchor, raw[anchor - WINDOW: anchor + HORIZON]))
    return windows


def clipping_geometry(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
    op: str,
    params: Mapping[str, object],
    origin: int,
) -> dict[str, Any]:
    """structured_clipping_geometry（审查裁决 十四）：规范 Scope（训练窗口）下
    执行全局阈值裁剪算子，度量裁剪比例/幅度/端点集中/季节峰谷集中/上侧不对称。
    每个窗口一个值，跨窗口取 median（与 cohort 评估同聚合层）。"""
    program = Program.from_steps([(op, dict(params))], source="flip_diagnosis")
    steps = program.execution_steps()
    per_window: dict[str, list[float]] = {
        "clip_fraction": [], "clip_magnitude": [],
        "clip_at_edges_fraction": [], "clip_at_peak_fraction": [],
        "clip_at_trough_fraction": [], "clip_upper_fraction": [],
    }
    n_windows = 0
    for _uid, _anchor, window in _training_windows(roster, values, config, origin):
        inp = as_1d(window).astype(np.float64)
        y = interp_nan(inp)  # 算子内部先插值（winsorize/outlier_iqr 同语义）
        execution = run_pipeline(steps, y, source="flip_diagnosis")
        if not execution.ok or execution.artifact is None:
            continue
        out = np.asarray(execution.artifact, dtype=np.float64).ravel()
        n_windows += 1
        clipped = ~np.isclose(out, y, equal_nan=True)
        n_clipped = int(clipped.sum())
        if n_clipped == 0:
            continue
        n = y.size
        frac = n_clipped / n
        mad = float(1.4826 * np.median(np.abs(y - np.median(y))))
        if not np.isfinite(mad) or mad <= 1e-9:
            mad = float(np.std(y)) or 1.0
        per_window["clip_fraction"].append(frac)
        per_window["clip_magnitude"].append(
            float(np.median(np.abs(out[clipped] - y[clipped])) / mad))
        # 端点区（趋势端点）：前/后 25%（均匀期望 0.5）
        edge_mask = (np.arange(n) < EDGE_FRACTION * n) | (
            np.arange(n) >= (1.0 - EDGE_FRACTION) * n)
        per_window["clip_at_edges_fraction"].append(
            float(clipped[edge_mask].sum() / n_clipped))
        # 日型峰/谷相位（period 分桶均值 → argmax/argmin；±PHASE_BAND 环带）
        period = int(config.get("period", 24))
        phases = np.arange(n) % period
        daily = np.asarray([np.median(y[phases == p]) for p in range(period)])
        peak, trough = int(np.argmax(daily)), int(np.argmin(daily))
        in_band = lambda center: (  # noqa: E731 环带
            (np.abs((phases - center + period // 2) % period - period // 2)
             <= PHASE_BAND))
        peak_band = in_band(peak)
        trough_band = in_band(trough)
        per_window["clip_at_peak_fraction"].append(
            float(clipped[peak_band].sum() / n_clipped))
        per_window["clip_at_trough_fraction"].append(
            float(clipped[trough_band].sum() / n_clipped))
        # 上侧不对称（>hi 裁剪占比）
        hi = float(np.quantile(y, 1.0 - float(dict(params).get("limits", 0.05)))) \
            if op == "winsorize" else float(
                np.quantile(y, 0.75) + float(dict(params).get("k", 1.5))
                * (np.quantile(y, 0.75) - np.quantile(y, 0.25)))
        upper = y[clipped] > hi if n_clipped else np.asarray([], dtype=bool)
        per_window["clip_upper_fraction"].append(
            float(upper.sum() / n_clipped) if n_clipped else 0.0)
    return {
        "operator": op,
        "origin": origin,
        "n_windows": n_windows,
        "median_per_window": {
            key: round(float(statistics.median(vals)), 4) if vals else None
            for key, vals in per_window.items()
        },
        "window_min_max": {
            key: [round(min(vals), 4), round(max(vals), 4)] if vals else None
            for key, vals in per_window.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 GEFCom winsorize flip diagnosis")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    domain = "gefcom"

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    max_len = max(int(len(v)) for v in values.values())
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    # origin 需 origin+2*HORIZON <= max_len（support 与 delayed 两段）；任务规定
    # 200-928 步长 48，并显式包含冻结时间线关键点 832/880/928
    # （注意：200+48k 网格不含 832/880/928，故并集补齐）
    origins = sorted(
        set(range(200, min(max_len - 2 * HORIZON, 928) + 1, HORIZON))
        | {832, 880, 928}
    )
    print(f"== {domain}: max_len={max_len}, origins={origins}")

    per_op: dict[str, list[dict[str, Any]]] = {op: [] for op in OPERATORS}
    for origin in origins:
        F = local_missing_feature(values, origin)
        for op in OPERATORS:
            compiled = v1.make_compiled(op, _default_params(op, period))
            s = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            d = v1.gain_at(roster, values, config, compiled, origin + HORIZON,
                           baseline_cache)
            if s is None or d is None:
                continue
            per_op[op].append({
                "origin": origin, "F": F, "support_gain": s, "delayed_gain": d,
            })

    results: dict[str, Any] = {}
    print("\n== 每 origin 表（support 为 s(origin)，delayed 为 s(origin+48)）")
    for op in OPERATORS:
        rows = per_op[op]
        print(f"\n  {op}:")
        for r in rows:
            print(
                f"    origin={r['origin']:4d} missing={r['F']['median_window_max_missing_run']:4.1f} "
                f"(max={r['F']['max_window_max_missing_run']:4.1f}, frac={r['F']['series_with_missing_fraction']:.3f}) "
                f"support={r['support_gain']:+.4f} delayed={r['delayed_gain']:+.4f}"
            )
        results[op] = rows

    # --- 分组对比：缺失水平 0 vs >0 ---
    print("\n== 分组对比（局部缺失 median=0 vs >0）")
    groups: dict[str, Any] = {}
    for op in OPERATORS:
        rows = per_op[op]
        g0 = [r for r in rows if r["F"]["median_window_max_missing_run"] == 0.0]
        g1 = [r for r in rows if r["F"]["median_window_max_missing_run"] > 0.0]

        def summarize(rows_g: list[dict[str, Any]], label: str) -> dict[str, Any]:
            if not rows_g:
                return {"label": label, "n": 0}
            s_vals = [r["support_gain"] for r in rows_g]
            d_vals = [r["delayed_gain"] for r in rows_g]
            flip_any = sum(1 for r in rows_g if (r["support_gain"] > 0) != (r["delayed_gain"] > 0))
            flip_pos_neg = sum(1 for r in rows_g if r["support_gain"] > 0 and r["delayed_gain"] < 0)
            both_pos = sum(1 for r in rows_g if r["support_gain"] > 0 and r["delayed_gain"] > 0)
            return {
                "label": label,
                "n": len(rows_g),
                "support_mean": round(sum(s_vals) / len(s_vals), 4),
                "support_min": round(min(s_vals), 4),
                "delayed_mean": round(sum(d_vals) / len(d_vals), 4),
                "delayed_min": round(min(d_vals), 4),
                "flip_any_fraction": round(flip_any / len(rows_g), 3),
                "flip_pos_to_neg_fraction": round(flip_pos_neg / len(rows_g), 3),
                "both_positive_fraction": round(both_pos / len(rows_g), 3),
                "support_negative_any": any(v <= 0 for v in s_vals),
                "delayed_negative_any": any(v <= 0 for v in d_vals),
            }

        g0s, g1s = summarize(g0, "missing0"), summarize(g1, "missing>0")
        groups[op] = {"missing_0": g0s, "missing_gt0": g1s}
        print(f"  {op:18s} missing0 n={g0s['n']:2d} s_mean={g0s.get('support_mean')} "
              f"s_neg_any={g0s.get('support_negative_any')} | "
              f"missing>0 n={g1s['n']:2d} s_mean={g1s.get('support_mean')} "
              f"s_neg_any={g1s.get('support_negative_any')}")

    # --- 裁决 ---
    win_rows = per_op["winsorize"]
    src = {r["origin"]: r for r in win_rows}
    diag: dict[str, Any] = {}
    r832, r880, r928 = src.get(832), src.get(880), src.get(928)
    if r832 and r880 and r928:
        # Source 经验 relation（build_source_memory 语义）：只看 s(832) 与 s(880)
        relation = ("POSITIVE" if r832["support_gain"] > 0 and r880["support_gain"] > 0
                    else "NOT_BOTH_POSITIVE")
        diag["source_832_880"] = {
            "missing": [r832["F"]["median_window_max_missing_run"],
                        r880["F"]["median_window_max_missing_run"]],
            "support": [round(r832["support_gain"], 4), round(r880["support_gain"], 4)],
            "delayed_origin_880_support": round(r880["support_gain"], 4),
            # 注意：d(880)=s(928)——Source delayed 切片自身的 delayed 评估已翻负
            "s928_as_delayed_of_880": round(r880["delayed_gain"], 4),
            "relation": relation,
        }
        diag["target_928"] = {
            "missing": r928["F"]["median_window_max_missing_run"],
            "support": round(r928["support_gain"], 4),
            "delayed": round(r928["delayed_gain"], 4),
            "flipped_vs_source": r928["support_gain"] <= 0,
        }

    # 翻转同类算子对照：928 处翻转（s>0,d<0 或 s<0,d>0）是否在缺失 0 组内同样存在
    flip_peers: dict[str, Any] = {}
    for op in OPERATORS:
        rows = per_op[op]
        by_o = {r["origin"]: r for r in rows}
        r928 = by_o.get(928)
        if r928 is None:
            continue
        flip_928 = (r928["support_gain"] > 0) != (r928["delayed_gain"] > 0)
        g0_rows = [r for r in rows if r["F"]["median_window_max_missing_run"] == 0.0]
        peer_flips = [
            {"origin": r["origin"], "support": round(r["support_gain"], 4),
             "delayed": round(r["delayed_gain"], 4)}
            for r in g0_rows if (r["support_gain"] > 0) != (r["delayed_gain"] > 0)
        ]
        flip_peers[op] = {
            "flip_at_928": flip_928,
            "support_928": round(r928["support_gain"], 4),
            "delayed_928": round(r928["delayed_gain"], 4),
            "same_flip_in_missing0_group": len(peer_flips) > 0,
            "missing0_flip_samples": peer_flips[:4],
        }

    # 关键判定：缺失 0 组内 winsorize 是否也存在 support<0 / delayed<0 / 翻转
    g0 = groups["winsorize"]["missing_0"]
    g1 = groups["winsorize"]["missing_gt0"]
    if g0["n"] == 0 or g1["n"] == 0:
        verdict = "INSUFFICIENT_GROUP_COVERAGE"
        reason = "winsorize 在缺失 0 组或缺失>0 组无有效样本"
    else:
        if g0["support_negative_any"] or g0["delayed_negative_any"] or g0["flip_any_fraction"] > 0:
            verdict = "NOT_MISSING_DRIVEN"
            reason = (
                f"winsorize 在缺失 0 的 {g0['n']} 个 origin 上已出现 "
                f"support<0({g0['support_negative_any']})/delayed<0({g0['delayed_negative_any']})/"
                f"翻转率 {g0['flip_any_fraction']}，负收益与翻转在无缺失切片同样发生，"
                f"928 的翻转不能用局部缺失 0→18 解释 → 单切片经验不跨切片确认"
            )
        elif g1["n"] == 1 and g1["support_negative_any"]:
            verdict = "CONSISTENT_WITH_MISSING_BUT_SINGLE_SAMPLE"
            reason = (
                f"缺失 0 组（n={g0['n']}）winsorize 全部 support>0 且 delayed>0 且无翻转，"
                f"仅缺失>0 的 928 翻转（support={r928['support_gain']:+.4f}）——与缺失驱动一致，"
                f"但缺失组 n=1（GEFCom 缺失是单事件二值 {0,18}），不构成统计证据，不能判定驱动"
            )
        else:
            verdict = "FLIP_WITHOUT_CLEAR_MISSING_ASSOCIATION"
            reason = f"分组结果：missing0={g0}, missing>0={g1}"

    print(f"\n== winsorize 翻转诊断")
    print(f"   Source 832/880: {diag.get('source_832_880')}")
    print(f"   Target  928:   {diag.get('target_928')}")
    print(f"   （注：d(880)=s(928)=-0.1636——winsorize 在 Source delayed 切片的"
          f" delayed 评估（即 [880,928) 切片）已为负）")
    print(f"\n== 翻转同类算子（928 翻转 vs 缺失 0 组内同类翻转）")
    for op, fp in flip_peers.items():
        print(f"   {op:18s} flip_928={fp['flip_at_928']} "
              f"928(s,d)=({fp['support_928']},{fp['delayed_928']}) "
              f"missing0 组同类翻转={fp['same_flip_in_missing0_group']} "
              f"样本={fp['missing0_flip_samples']}")
    print(f"\n== verdict: {verdict}")
    print(f"   {reason}")
    print("   （单时间线单次运行、缺失组 n=1，不构成统计显著）")

    # ------------------------------------------------------------------
    # 诊断二：structured_clipping_geometry（审查裁决 十四）
    # 规范 Scope（训练窗口）+ 相同 Evaluator 对照 winsorize vs outlier_iqr
    # ------------------------------------------------------------------
    import run_v1_a5_vs_a3 as core  # noqa: PLC0415

    m_threshold = core.MATERIAL_THRESHOLD
    clip_rows: dict[str, dict[int, dict[str, Any]]] = {}
    gains: dict[str, dict[int, float | None]] = {}
    print("\n== structured_clipping_geometry（规范 Scope 训练窗口；"
          "均匀期望：edges=0.5、峰/谷带=%.3f）" % ((2 * PHASE_BAND + 1) / period))
    for op in CLIP_PAIR:
        params = _default_params(op, period)
        clip_rows[op] = {}
        gains[op] = {}
        for origin in CLIP_ORIGINS:
            geom = clipping_geometry(roster, values, config, op, params, origin)
            clip_rows[op][origin] = geom
            g = v1.gain_at(roster, values, config, v1.make_compiled(op, params),
                           origin, baseline_cache)
            gains[op][origin] = g
            g_text = f"{g:+.4f}" if g is not None else "None"
            med = geom["median_per_window"]
            print(f"  {op:12s} @{origin} gain={g_text:>8s} "
                  f"frac={med['clip_fraction']} mag={med['clip_magnitude']} "
                  f"edges={med['clip_at_edges_fraction']} "
                  f"peak={med['clip_at_peak_fraction']} "
                  f"trough={med['clip_at_trough_fraction']} "
                  f"upper={med['clip_upper_fraction']} "
                  f"nwin={geom['n_windows']}")

    # 正负切片分组（winsorize 的 v6 规范 gain）
    win_g = gains["winsorize"]
    pos_origins = [o for o in CLIP_ORIGINS
                   if win_g[o] is not None and win_g[o] > 0]
    neg_origins = [o for o in CLIP_ORIGINS
                   if win_g[o] is not None and win_g[o] <= 0]
    features = ["clip_fraction", "clip_magnitude", "clip_at_edges_fraction",
                "clip_at_peak_fraction", "clip_at_trough_fraction",
                "clip_upper_fraction"]
    separation: dict[str, Any] = {}
    for feat in features:
        pos_vals = [clip_rows["winsorize"][o]["median_per_window"][feat]
                    for o in pos_origins
                    if clip_rows["winsorize"][o]["median_per_window"][feat]
                    is not None]
        neg_vals = {o: clip_rows["winsorize"][o]["median_per_window"][feat]
                    for o in neg_origins
                    if clip_rows["winsorize"][o]["median_per_window"][feat]
                    is not None}
        if pos_vals and neg_vals:
            lo, hi = min(pos_vals), max(pos_vals)
            separation[feat] = {
                "positive_origins": {o: clip_rows["winsorize"][o]
                                     ["median_per_window"][feat]
                                     for o in pos_origins},
                "negative_origins": neg_vals,
                "positive_range": [lo, hi],
                "negative_outside_positive_range": any(
                    v < lo or v > hi for v in neg_vals.values()),
            }
        else:
            separation[feat] = None
    iqr_g = gains["outlier_iqr"]
    iqr_headroom = any(g is not None and g >= m_threshold for g in iqr_g.values())
    iqr_headroom_origins = [o for o in CLIP_ORIGINS
                            if iqr_g[o] is not None and iqr_g[o] >= m_threshold]
    discriminating = [feat for feat, sep in separation.items()
                      if sep and sep["negative_outside_positive_range"]]

    if discriminating and iqr_headroom:
        clipping_verdict = "CLIPPING_GEOMETRY_DISCRIMINATES"
        clipping_reason = (
            f"winsorize 负切片（{neg_origins}）在特征 {discriminating} 上落在 "
            f"正切片（{pos_origins}）范围之外（可区分）；outlier_iqr 替代 headroom "
            f"保留（@{iqr_headroom_origins}）→ 可批准一次单独的 Observation/Scope "
            f"修改（候选，非统计显著：正 n={len(pos_origins)} 负 n={len(neg_origins)}）"
        )
    elif not discriminating and iqr_headroom:
        clipping_verdict = "CLIPPING_GEOMETRY_NOT_DISCRIMINATIVE"
        clipping_reason = (
            f"winsorize 负切片（{neg_origins}）无任何几何特征落在正切片范围之外"
            f"（weak-history 下不可识别）；停止扩 Pattern，依赖 Support 验证/"
            f"abstain，转向 1024+ 空间的链做真正纵向验证。"
        )
    else:
        clipping_verdict = "NO_IQR_ALTERNATIVE_HEADROOM"
        clipping_reason = (
            f"outlier_iqr 在 {CLIP_ORIGINS} 上无 support gain ≥ {m_threshold}"
            f"（替代 headroom 不成立），对照无效。"
        )
    print(f"\n== clipping verdict: {clipping_verdict}")
    print(f"   {clipping_reason}")
    print("   （正负切片 n 极小，仅定性候选；不调 q75、不接入 resolver、不换链）")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-gefcom-winsorize-flip-diagnosis",
            "domain": domain,
            "window": WINDOW,
            "feature": "median_window_max_missing_run",
            "origins": origins,
            "per_operator": {
                op: [
                    {"origin": r["origin"],
                     "F": r["F"],
                     "support_gain": round(r["support_gain"], 4),
                     "delayed_gain": round(r["delayed_gain"], 4)}
                    for r in rows
                ]
                for op, rows in per_op.items()
            },
            "group_comparison": groups,
            "winsorize_flip_diag": diag,
            "flip_peer_operators": flip_peers,
            "verdict": verdict,
            "verdict_reason": reason,
            "clipping_geometry": {
                "family": "structured_clipping_geometry",
                "scope": "training_windows_only",
                "operators": CLIP_PAIR,
                "origins": CLIP_ORIGINS,
                "per_operator": clip_rows,
                "support_gains": gains,
                "positive_origins_winsorize": pos_origins,
                "negative_origins_winsorize": neg_origins,
                "feature_separation": separation,
                "outlier_iqr_headroom_origins": iqr_headroom_origins,
                "verdict": clipping_verdict,
                "verdict_reason": clipping_reason,
            },
            "claim_ceiling": (
                "Single timeline single run, GEFCom missingness is a single binary event "
                "({0,18}), missing group n=1: not statistically significant. Clipping "
                "geometry comparison is qualitative (positive n=3, negative n=1)."
            ),
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
