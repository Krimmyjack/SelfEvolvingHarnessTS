"""conditioning/period.py — 共享周期估计模块（Stage 2.1-A0 第一步，D1 修复的地基）。

D1 根因 = **两个 period estimator 漂移**：感知端（conditioning/key.py，朴素 FFT argmax，
被低频趋势劫持 → S_both 160/160 period 中位 452≈序列长）与算子端（s1_denoise，S0.7-1 已修
稳健版）各自实现。本模块把两者**逐字搬入**同一定义点，行为 bit 级不变：

  legacy_fft_v0   原 key.py `_dominant_period`——P0 冻结契约的一部分，**朴素缺陷保留是有意的**
                  （P0 的价值=E-3.2/confirmatory 训练分布的精确复现锚，禁止原地改）；
  robust_v1       原 s1_denoise `_guess_period`（S0.7 修复版：去趋势 + 候选范围 [pmin, n/3]
                  + 谱峰显著性 + ACF 确认 + 无周期返回 0）——STL 算子在用；
  top_k_periods   robust_v1 的多周期扩展（P1 特征专用，新增，不影响任何既有路径）。

三处消费者：Pattern extractor（key.py，legacy=P0；P1 将用 robust/top-k）、STL 算子
（s1_denoise/s1_decompose，robust）、period-aware 动作（period_complete 显式传参，未来可接）。
估计器 ID 进入 PatternSpec/provenance——两估计器**允许不同**（P0 冻结要求），但分叉自此
显式、有版本、单一定义点。

bit 等价守卫：tests/test_period_shared.py（内联旧实现逐字对照）。
依赖：仅 numpy（保持 operators → 本模块无循环导入）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ═══════════════════════ legacy_fft_v0（P0 感知端，缺陷冻结）═══════════════════════
def dominant_period_fft_v0(x: np.ndarray) -> tuple:
    """返回 (period, power_spectrum_normalized)。period=1 表示无显著周期。

    ⚠ 逐字来自 conditioning/key.py `_dominant_period`（2026-07-05 迁入，bit 级不变）：
    无去趋势/无显著性检验/无 ACF 确认（D1 已知缺陷）——**P0 冻结契约，禁止在此"顺手修"**；
    修复走 robust_v1/top_k_periods 且只进 PatternSpec 新版本（P1+）。"""
    n = x.size
    if n < 8:
        return 1.0, np.ones(1)
    xd = x - x.mean()
    fft = np.fft.rfft(xd)
    power = (np.abs(fft) ** 2)[1:]          # 去 DC
    if power.size == 0 or power.sum() <= 1e-12:
        return 1.0, np.array([1.0])
    freqs = np.fft.rfftfreq(n)[1:]
    k = int(np.argmax(power))
    f = freqs[k]
    period = float(1.0 / f) if f > 0 else 1.0
    period = min(period, float(n))          # 不超过序列长
    return period, power / power.sum()


# ═══════════════════════ robust_v1（算子端，S0.7 修复版）═══════════════════════
def _acf_demeaned(resid: np.ndarray, lag: int) -> float:
    """滞后 lag 的自相关（输入已去均值）。白噪 → ≈0；真周期 lag → 显著正。
    逐字来自 operators/s1_denoise `_acf`（2026-07-05 迁入，bit 级不变）。"""
    n = resid.size
    if lag <= 0 or lag >= n:
        return 0.0
    v = float(np.dot(resid, resid))
    return float(np.dot(resid[:-lag], resid[lag:]) / v) if v > 0 else 0.0


def robust_period_diag(y: np.ndarray, pmin: int = 4, pmax: int = 0,
                       min_peak_ratio: float = 3.0, acf_min: float = 0.2) -> Dict[str, float]:
    """robust_v1 的**诊断变体**（P1 C 通道专用，2026-07-05 P1a 加入）：除 period 外返回候选峰
    证据——无论判据是否通过（C 通道要看到"证据有多强"，P 通道只要"判决"）。

    period 值与 guess_period_robust_v1 **完全一致**（后者委托本函数——单一定义点，D1 教训；
    诊断读数是纯附加运算，不改变 period 的运算序列；bit 等价由 test_period_shared 内联旧实现守）。
    返回 {period, cand_period, peak_ratio, acf_at_peak}；退化输入 → 全 0。"""
    n = y.size
    pmax = pmax if pmax >= pmin else n // 3
    out: Dict[str, float] = {"period": 0, "cand_period": 0, "peak_ratio": 0.0, "acf_at_peak": 0.0}
    if n < 2 * pmin or pmax < pmin:
        return out
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T                          # 去线性趋势（防低频劫持）
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    resid = resid - resid.mean()
    if not np.any(np.abs(resid) > 1e-12):
        return out
    power = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n)
    band = (freqs >= 1.0 / pmax) & (freqs <= 1.0 / pmin)      # 只在候选周期范围找峰
    if not band.any():
        return out
    k = int(np.argmax(np.where(band, power, 0.0)))
    med = float(np.median(power[band]))
    if med <= 0 or freqs[k] <= 0:
        return out
    cand = int(round(1.0 / freqs[k]))
    out["cand_period"] = cand
    out["peak_ratio"] = float(power[k] / med)                 # 诊断读数（不影响 period 判决序列）
    if pmin <= cand <= pmax:
        # ACF 确认：白噪残差的谱峰会偶然超过 ratio，但其 ACF≈0；真周期在该 lag ACF 显著
        out["acf_at_peak"] = float(_acf_demeaned(resid, cand))
    accepted = (power[k] > min_peak_ratio * med and pmin <= cand <= pmax
                and out["acf_at_peak"] >= acf_min)
    out["period"] = cand if accepted else 0
    return out


def guess_period_robust_v1(y: np.ndarray, pmin: int = 4, pmax: int = 0,
                           min_peak_ratio: float = 3.0, acf_min: float = 0.2) -> int:
    """估计主周期；**无显著季节返回 0**。

    逐字来自 operators/s1_denoise `_guess_period`（S0.7 修复版，2026-07-05 迁入，bit 级不变）：
    ①去线性趋势后谱分析（防低频劫持）；②候选周期限 [pmin, pmax=n//3]；③谱峰显著性
    （峰功率 ≥ ratio×频带中位）；④ACF 确认（候选 lag 自相关 ≥ acf_min）；⑤都不过 → 0。
    实现自 P1a 起委托 robust_period_diag（同一运算序列取 period 位；防两份实现漂移=D1 教训）。"""
    return int(robust_period_diag(y, pmin, pmax, min_peak_ratio, acf_min)["period"])


# ═══════════════════════ top-k 多周期（P1 特征专用，新增）═══════════════════════
def top_k_periods(y: np.ndarray, k: int = 3, pmin: int = 4, pmax: int = 0,
                  min_peak_ratio: float = 3.0, acf_min: float = 0.2,
                  dedupe_rel: float = 0.15) -> List[int]:
    """robust_v1 判据下的 top-k 多周期（按谱峰功率降序；谐波/近邻去重）。

    只供 PatternSpec P1+ 的多周期特征——不进 P0、不进任何算子路径（A0 前提：算子输出
    bit 级不变，旧响应矩阵与 cached loss 才能复用）。
    去重：候选与已接受周期 p 满足 |cand−p·m|/ (p·m) < dedupe_rel（m=1..4 谐波）→ 跳过。"""
    n = y.size
    pmax = pmax if pmax >= pmin else n // 3
    if n < 2 * pmin or pmax < pmin:
        return []
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    resid = resid - resid.mean()
    if not np.any(np.abs(resid) > 1e-12):
        return []
    power = np.abs(np.fft.rfft(resid)) ** 2
    freqs = np.fft.rfftfreq(n)
    band = np.where((freqs >= 1.0 / pmax) & (freqs <= 1.0 / pmin))[0]
    if band.size == 0:
        return []
    med = float(np.median(power[band]))
    out: List[int] = []
    for idx in band[np.argsort(power[band])[::-1]]:           # 功率降序
        if len(out) >= k or med <= 0 or power[idx] <= min_peak_ratio * med:
            break                                             # 后续峰更弱 → 全部不显著
        p = int(round(1.0 / freqs[idx])) if freqs[idx] > 0 else 0
        if not (pmin <= p <= pmax) or _acf_demeaned(resid, p) < acf_min:
            continue
        dup = any(abs(p - q * m) / (q * m) < dedupe_rel
                  for q in out for m in (1, 2, 3, 4)) or \
              any(abs(q - p * m) / (p * m) < dedupe_rel
                  for q in out for m in (2, 3, 4))
        if not dup:
            out.append(p)
    return out


# ═══════════════════════ 统一入口 ═══════════════════════
ESTIMATOR_IDS: Dict[str, str] = {
    "legacy_fft_v0": "dominant_period_fft_v0（P0 冻结：朴素 FFT argmax，缺陷保留）",
    "robust_v1": "guess_period_robust_v1（S0.7 修复：detrend+范围+显著性+ACF）",
}


def estimate_period(x: np.ndarray, estimator: str = "robust_v1", **kw) -> float:
    """按估计器 ID 派发（provenance 用同一 ID 记账）。legacy 返回 float（P0 语义），
    robust 返回 int（0=无显著周期）。"""
    y = np.asarray(x, dtype=float).ravel()
    if estimator == "legacy_fft_v0":
        return dominant_period_fft_v0(y)[0]
    if estimator == "robust_v1":
        return float(guess_period_robust_v1(y, **kw))
    raise KeyError(f"未知 period estimator: {estimator!r}；可用：{sorted(ESTIMATOR_IDS)}")
