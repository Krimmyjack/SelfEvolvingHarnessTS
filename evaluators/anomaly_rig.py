"""evaluators/anomaly_rig.py — P2 最小 anomaly rig（Final_Plan_CodeAgentFirst_2026-07-09 §P2）。

三件套：合成注入协议（point + contextual level-shift）+ 固定检测器（NaN 安全的
rolling-median 残差鲁棒 z-score，确定性、零训练、σ=0 判官）+ F1/AUROC 判官
（命中容差 ±ANOM_TOL=2，沿用项目惯例）。

纪律（一周封顶）：rig 是**判官不是研究对象**——检测器与阈值冻结在 DETECTOR_SPEC，
不做检测器研究；老的 grounded_anomaly.py（top-frac 召回版）保留不动，本模块是增量。
任务语义：anomaly 任务下 spike/level-shift 是**目标信号**，平滑/删改类预处理会抹掉
检测目标（registry 已物理禁 anomaly 下的 smoothing/destructive；本 rig 量化"若违约执行
会损失多少"，供 P2 动机表使用）。
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

ANOM_TOL = 2   # 命中容差（±样本），与 evaluators/base.py 的惯例一致

DETECTOR_SPEC: Dict[str, Any] = {
    "detector": "residual_zscore",
    # leave-one-out：基线=窗内**邻居**中值（不含中心点）。含自身会自吸收每点残差
    # （中心分布被压缩而尾部不变 → MAD 低估 ~27%、z 虚高、假告警成簇——rig 首轮实测教训）
    "baseline": "rolling_median_loo_nan_safe",
    # 窗设计：须 > 2×contextual 段长（抗 level-shift 吞并）且 ≪ 周期（基线跟踪季节曲线，
    # 否则整周期中值≈0 → 正弦全量进残差、MAD 膨胀、spike z 被稀释）
    "window": 11,
    "threshold": 4.0,
    # 告警尺度在 **raw 输入**上冻结标定（运营语义：阈值按进线数据定，不随清洗器输出重拟合）。
    # 若随 artifact 自标定，平滑会把残差与 MAD 一起压扁 → z 反而全面爆炸、recall 假性保持。
    "scale_calibration": "frozen_on_raw_input",
    "tol": ANOM_TOL,
    "frozen": True,   # 判官冻结：改任何一项 = 新判官身份，须在 manifest 里换名
}


# ── 注入协议 ────────────────────────────────────────────────────────────────

def _pick_positions(rng: np.random.Generator, n: int, count: int,
                    *, margin: int, min_sep: int, occupied: np.ndarray) -> List[int]:
    """从 [margin, n-margin) 里挑 count 个互不靠近、且不与 occupied 冲突的位置（确定性于 rng）。"""
    if count <= 0:
        return []
    candidates = rng.permutation(np.arange(margin, n - margin))
    picked: List[int] = []
    for pos in candidates:
        if occupied[max(0, pos - min_sep):pos + min_sep + 1].any():
            continue
        if any(abs(pos - q) < min_sep for q in picked):
            continue
        picked.append(int(pos))
        if len(picked) >= count:
            break
    return picked


def inject_anomalies(
    x_clean: np.ndarray,
    *,
    rng: np.random.Generator,
    n_points: int = 6,
    n_contextual: int = 1,
    point_amp: float = 6.0,
    shift_amp: float = 2.5,
    segment_len: int = 5,
    margin: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """在干净序列上注入 point spike 与 contextual level-shift；返回 (x, labels)。

    labels 为逐点布尔真值（segment 的每个点都标注）。注入只发生在 margin 内侧，
    point 与 segment 互不重叠（min_sep 保证）。
    """
    x = np.asarray(x_clean, dtype=float).copy()
    n = x.size
    labels = np.zeros(n, dtype=bool)
    occupied = np.zeros(n, dtype=bool)

    seg_starts = _pick_positions(rng, n - segment_len, n_contextual,
                                 margin=margin, min_sep=segment_len + 2 * ANOM_TOL + 2,
                                 occupied=occupied)
    if len(seg_starts) < n_contextual:
        raise ValueError(
            f"注入容量不足：请求 {n_contextual} 段 contextual，仅能放置 {len(seg_starts)}"
            f"（n={n}, segment_len={segment_len}, margin={margin}）——静默少注会漂移指标分母")
    for start in seg_starts:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        x[start:start + segment_len] += sign * shift_amp
        labels[start:start + segment_len] = True
        occupied[max(0, start - ANOM_TOL):start + segment_len + ANOM_TOL] = True

    points = _pick_positions(rng, n, n_points,
                             min_sep=2 * ANOM_TOL + 2, margin=margin, occupied=occupied)
    if len(points) < n_points:
        raise ValueError(
            f"注入容量不足：请求 {n_points} 个 point 异常，仅能放置 {len(points)}"
            f"（n={n}, margin={margin}）——静默少注会漂移指标分母")
    for pos in points:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        x[pos] += sign * point_amp
        labels[pos] = True
        occupied[max(0, pos - ANOM_TOL):pos + ANOM_TOL + 1] = True

    return x, labels


# ── 固定检测器（NaN 安全、确定性）───────────────────────────────────────────

def _loo_residuals(x: np.ndarray, window: int) -> np.ndarray:
    """leave-one-out 残差：x 减去窗内**邻居**中值（不含中心点，NaN 安全）。"""
    x = np.asarray(x, dtype=float).ravel()
    if window % 2 == 0:
        raise ValueError(f"window 须为奇数，得到 {window}")
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(xp, window)
    neighbors = np.delete(sw, pad, axis=1)          # 剔除中心列 → 自吸收消除
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN 窗口
        baseline = np.nanmedian(neighbors, axis=1)
    return x - baseline


def residual_zscore_scores(x: np.ndarray, *, window: int = 11, eps: float = 1e-9,
                           scale_reference: np.ndarray | None = None) -> np.ndarray:
    """|鲁棒 z| 分数。med/MAD 尺度默认从 scale_reference（=raw 输入）的残差标定并冻结；
    不提供 reference 时退化为自标定。NaN 位置分数=0。"""
    resid = _loo_residuals(x, window)
    ref_resid = resid if scale_reference is None else _loo_residuals(scale_reference, window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(ref_resid)
        mad = np.nanmedian(np.abs(ref_resid - med)) * 1.4826
    if not np.isfinite(med):
        med = 0.0
    scale = mad if np.isfinite(mad) and mad > eps else eps
    z = np.abs(resid - med) / scale
    z[~np.isfinite(z)] = 0.0
    return z


def detect_flags(x: np.ndarray, *, window: int = 11, threshold: float = 4.0,
                 scale_reference: np.ndarray | None = None) -> np.ndarray:
    return residual_zscore_scores(x, window=window, scale_reference=scale_reference) > float(threshold)


# ── 判官（容差 ±tol 的 F1 + 膨胀标签 AUROC）─────────────────────────────────

def _dilate(labels: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return labels.astype(bool)
    kernel = np.ones(2 * tol + 1)
    return np.convolve(labels.astype(float), kernel, mode="same") > 0


def _auroc(scores: np.ndarray, labels: np.ndarray, dilated_labels: np.ndarray) -> float:
    """严格正类（labels）vs 严格负类（~dilated），容差环排除——环内点既不奖励也不惩罚。"""
    from scipy.stats import rankdata
    mask = labels | ~dilated_labels
    s, p = scores[mask], labels[mask]
    n_pos, n_neg = int(p.sum()), int((~p).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)   # 平均秩处理并列（NaN 位置分数=0 常并列）
    u = float(ranks[p].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def anomaly_metrics(scores: np.ndarray, flags: np.ndarray, labels: np.ndarray,
                    *, tol: int = ANOM_TOL) -> Dict[str, float]:
    """recall = 有 flag 落在 ±tol 内的真值点占比；precision = 落在任一真值 ±tol 内的 flag 占比；
    AUROC 用 ±tol 膨胀后的标签逐点计算。"""
    scores = np.asarray(scores, dtype=float).ravel()
    flags = np.asarray(flags, dtype=bool).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    dilated_flags = _dilate(flags, tol)
    dilated_labels = _dilate(labels, tol)

    n_true = int(labels.sum())
    hits = int((labels & dilated_flags).sum())
    recall = hits / n_true if n_true else float("nan")

    n_flag = int(flags.sum())
    good_flags = int((flags & dilated_labels).sum())
    precision = good_flags / n_flag if n_flag else 0.0

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "recall": float(recall),
        "precision": float(precision),
        "F1": float(f1),
        "AUROC": float(_auroc(scores, labels, dilated_labels)),
        "n_true": n_true,
        "n_flagged": n_flag,
    }


def anomaly_readiness_eval(artifact: np.ndarray, labels: np.ndarray,
                           *, raw_reference: np.ndarray | None = None,
                           window: int | None = None, threshold: float | None = None,
                           tol: int | None = None) -> Dict[str, float]:
    """对处理后 artifact 跑冻结检测器，按原始坐标真值打分（预处理抹掉目标 → recall/F1 掉）。

    raw_reference = 系统实际收到的 raw 序列：告警尺度在它上面冻结标定（DETECTOR_SPEC
    scale_calibration）。缺省 None = 自标定退化（仅单测/探索用；动机表一律传 raw）。"""
    w = int(window if window is not None else DETECTOR_SPEC["window"])
    t = float(threshold if threshold is not None else DETECTOR_SPEC["threshold"])
    k = int(tol if tol is not None else DETECTOR_SPEC["tol"])
    scores = residual_zscore_scores(artifact, window=w, scale_reference=raw_reference)
    flags = scores > t
    return anomaly_metrics(scores, flags, np.asarray(labels, dtype=bool), tol=k)


# ── 动机切片生成器（forecast 与 anomaly 共用同一批序列 = 同数据不同任务）────────

def make_anomaly_slice(
    n_series: int,
    *,
    length: int = 256,
    horizon: int = 24,
    period: int = 24,
    seed: int = 20260709,
    n_points: int = 6,
    n_contextual: int = 1,
    miss_len: int = 12,
) -> List[Dict[str, Any]]:
    """确定性切片：seasonal 基线 + cell 化噪声 + 注入异常（仅历史段）+ 缺失（不落异常点）。

    偶数 idx = anomaly|snrHigh|full（σ=0.15，无缺失）；奇数 = anomaly|snrLow|miss（σ=0.7，
    连续缺失块）。future_clean 为无异常无噪声的干净未来（forecast 判官的真值）。
    """
    rows: List[Dict[str, Any]] = []
    total = length + horizon
    t = np.arange(total)
    for idx in range(int(n_series)):
        rng = np.random.default_rng(seed * 100_003 + idx)
        phase = rng.uniform(0, 2 * np.pi)
        clean_full = np.sin(2 * np.pi * t / period + phase)
        high = idx % 2 == 0
        sigma = 0.15 if high else 0.7
        cell = "anomaly|snrHigh|full" if high else "anomaly|snrLow|miss"

        hist_clean = clean_full[:length]
        hist_anom, labels = inject_anomalies(hist_clean, rng=rng,
                                             n_points=n_points, n_contextual=n_contextual)
        x = hist_anom + rng.normal(0.0, sigma, length)

        miss_rate = 0.0
        if not high:
            protected = _dilate(labels, ANOM_TOL + 1)
            starts = [s for s in range(8, length - miss_len - 8)
                      if not protected[s:s + miss_len].any()]
            if not starts:
                raise ValueError(
                    f"缺失块放不下：miss_len={miss_len} 在 length={length}、注入密度下无合法起点"
                    "（缺失不得覆盖异常点）——请缩小 miss_len 或降低注入密度")
            start = int(starts[int(rng.integers(0, len(starts)))])
            x[start:start + miss_len] = np.nan
            miss_rate = miss_len / length

        rows.append({
            "uid": f"p2_{idx}",
            "cell": cell,
            "x": x,
            "labels": labels,
            "future_clean": clean_full[length:],
            "period": period,
            "miss_rate": miss_rate,
            "noise_sigma": sigma,
        })
    return rows
