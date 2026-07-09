"""conditioning/p1a.py — PatternSpec P1a 特征提取器（Stage 2.1 第一臂，2026-07-05）。

P1a = P0 的 **D1+D2 修复版** + 缺失拓扑 D 特征 + 最小 C 通道（评审第二十五轮定案的最小集）：
  D1 修复：period 改用 robust_v1（去趋势+显著性+ACF 确认；conditioning/period.py 单一定义点）；
  D2 修复：**时间轴永不压缩**——P0 的 `x = raw[mask]` 压缩轴让 t 错位、谱错频；P1a 保持
           全长时间轴，谱/ACF 用**仅供感知的显式线性插值**（绝不触算子路径），逐点统计
           （噪声 MAD/离群/lumpiness/acf1）只在**观测点**上算。

特征分类边界（Component Plan v1.1d 冻结）：本模块只产 P/D/C；φ(P,D,a,m) 动作交互特征
（window/period、候选窗可平滑能量）**不在此处**——Router 轮（2.2）由 P 原料现算。

P 槽位与 P0 的 8 维语义对齐（隔离"修复"这一处理，避免和"换特征集"混淆）+ 1 新增：
  period                robust_v1（0=无显著周期；≠P0 的 1.0——no_period_repr 变更记入 spec）
  period_count          top_k_periods 数量（多周期词汇，P1 新增）
  trend_strength        观测点上的真时间轴线性拟合 R²
  seasonal_strength     |ACF(去趋势插值序列, period)|（无周期=0）
  acf1                  mask-aware lag-1 自相关（只用相邻两点都观测的对）
  stationarity_adf      ADF on 插值序列（语义=感知插值后的平稳性）
  spectral_entropy      去趋势插值序列的功率谱归一熵（P0 未去趋势——趋势能量会压低熵）
  lumpiness             时间轴保持的分块方差之方差（块内只用观测点）
  outlier_density       观测点稳健 MAD 判定（与 P0 同式）
D（退化画像，deploy 可得）：
  SNR                   与 P0 同式（top-K 谱剥离季节 + MAD 稳健噪声）但残差只取观测点
  missing_rate          缺失率（压缩前原索引）
  max_gap_frac          最长连续缺失段 / n
  gap_run_mean_frac     缺失段平均长度 / n（无缺失=0）
C（估计可信度，最小三维；双视图=P1b，不进本版）：
  c_peak_sig            谱峰显著性 ratio/(ratio+min_peak_ratio) ∈ [0,1)（0.5=判据线上）
  c_acf_confirm         候选周期处 ACF 原值（证据本身，无论判决通过与否）
  c_obs_coverage        最长连续**观测**段 / n（窗口可用性；区别于 missing_rate 的拓扑量）

退化规则（镜像 P0）：观测点 <4 → 全 0 + missing_rate/coverage 照记。
依赖：numpy + 复用 key.py 的 _adf_pvalue/_outlier_density（单一定义点，不复制公式）。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .key import _adf_pvalue, _outlier_density
from .period import robust_period_diag, top_k_periods

P1A_D_FEATS: Tuple[str, ...] = ("SNR", "missing_rate", "max_gap_frac", "gap_run_mean_frac")
P1A_P_FEATS: Tuple[str, ...] = ("period", "period_count", "trend_strength", "seasonal_strength",
                                "acf1", "stationarity_adf", "spectral_entropy",
                                "lumpiness", "outlier_density")
P1A_C_FEATS: Tuple[str, ...] = ("c_peak_sig", "c_acf_confirm", "c_obs_coverage")
P1A_ALL_FEATS: Tuple[str, ...] = P1A_D_FEATS + P1A_P_FEATS + P1A_C_FEATS
_MIN_PEAK_RATIO = 3.0                     # 与 robust_v1 默认一致（c_peak_sig 的判据线）


# ════════════════════════════ mask 工具 ════════════════════════════
def _runs(mask_bad: np.ndarray) -> List[int]:
    """连续 True 段的长度列表。"""
    out, cur = [], 0
    for b in mask_bad:
        if b:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
    if cur:
        out.append(cur)
    return out


def _interp_perception(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """仅供感知的显式线性插值（时间轴保持；端点最近观测值延伸）。**绝不进算子路径**。"""
    xi = raw.copy()
    t = np.arange(raw.size, dtype=float)
    xi[~mask] = np.interp(t[~mask], t[mask], raw[mask])
    return xi


def _acf1_masked(raw: np.ndarray, mask: np.ndarray) -> float:
    """lag-1 自相关，只用相邻两点都观测的对（插值会人为抬高 acf1——插值段光滑）。"""
    obs = raw[mask]
    if obs.size < 3:
        return 0.0
    mu = float(obs.mean())
    pair = mask[:-1] & mask[1:]
    if not pair.any():
        return 0.0
    a = raw[:-1][pair] - mu
    b = raw[1:][pair] - mu
    denom = float(np.sum((obs - mu) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.sum(a * b) / denom, -1.0, 1.0))


def _lumpiness_masked(raw: np.ndarray, mask: np.ndarray, n_tiles: int = 10) -> float:
    """P0 lumpiness 的时间轴保持版：按原索引分块，块内只用观测点（≥2）算方差。"""
    if int(mask.sum()) < n_tiles * 2:
        return 0.0
    tile_vars = []
    for tile_idx in np.array_split(np.arange(raw.size), n_tiles):
        v = raw[tile_idx]
        v = v[mask[tile_idx]]
        if v.size > 1:
            tile_vars.append(float(np.var(v)))
    if len(tile_vars) < 2:
        return 0.0
    tv = np.array(tile_vars)
    scale = float(tv.mean()) + 1e-12
    return float(np.var(tv) / (scale ** 2))


# ════════════════════════════ 主提取器 ════════════════════════════
def p1a_feats(series) -> Dict[str, float]:
    """P1a 全特征 dict（D+P+C，键=P1A_ALL_FEATS）。同输入 → bit 级一致（纯 numpy 确定性）。"""
    raw = np.asarray(series, dtype=float).ravel()
    if raw.size == 0:
        raise ValueError("empty series")
    n = raw.size
    mask = ~np.isnan(raw)
    missing_rate = float(1.0 - mask.mean())
    gap_runs = _runs(~mask)
    max_gap_frac = float(max(gap_runs) / n) if gap_runs else 0.0
    gap_run_mean_frac = float(np.mean(gap_runs) / n) if gap_runs else 0.0
    obs_runs = _runs(mask)
    c_obs_coverage = float(max(obs_runs) / n) if obs_runs else 0.0

    feats = {k: 0.0 for k in P1A_ALL_FEATS}
    feats.update(missing_rate=missing_rate, max_gap_frac=max_gap_frac,
                 gap_run_mean_frac=gap_run_mean_frac, c_obs_coverage=c_obs_coverage)
    if int(mask.sum()) < 4:                                  # 退化：几乎全缺（镜像 P0 规则）
        return feats

    t = np.arange(n, dtype=float)
    obs = raw[mask]
    xi = _interp_perception(raw, mask)

    # ── trend：观测点 × 真时间轴（D2 修复本体——P0 压缩轴让 t 错位）──
    var_obs = float(np.var(obs))
    if var_obs <= 1e-12:
        trend_strength = 0.0
        fit_full = np.full(n, obs.mean())
    else:
        coef = np.polyfit(t[mask], obs, 1)
        fit_full = np.polyval(coef, t)
        trend_strength = float(np.clip(1.0 - np.var(obs - fit_full[mask]) / var_obs, 0.0, 1.0))
    detr_i = xi - fit_full                                   # 感知插值 + 去趋势（谱/ACF 用）

    # ── period（D1 修复本体）+ C 证据 ──
    diag = robust_period_diag(detr_i)
    period = int(diag["period"])
    feats["period"] = float(period)                          # 0 = 无显著周期（no_period_repr 变更）
    feats["period_count"] = float(len(top_k_periods(detr_i)))
    feats["c_peak_sig"] = float(diag["peak_ratio"] / (diag["peak_ratio"] + _MIN_PEAK_RATIO))
    feats["c_acf_confirm"] = float(np.clip(diag["acf_at_peak"], -1.0, 1.0))

    # ── seasonal_strength：|ACF(detr_i, period)|（去均值后；无周期=0）──
    if period >= 2:
        d0 = detr_i - detr_i.mean()
        v = float(np.dot(d0, d0))
        ac = float(np.dot(d0[:-period], d0[period:]) / v) if v > 1e-12 and period < n else 0.0
        feats["seasonal_strength"] = float(np.clip(abs(ac), 0.0, 1.0))

    # ── SNR：P0 同式（top-K 谱剥离季节 + MAD 噪声）但残差只取观测点 ──
    fftd = np.fft.rfft(detr_i)
    if fftd.size > 1:
        k_keep = min(3, fftd.size - 1)
        top = np.argsort(np.abs(fftd[1:]))[-k_keep:] + 1
        seasonal_fft = np.zeros_like(fftd)
        seasonal_fft[top] = fftd[top]
        seasonal_comp = np.fft.irfft(seasonal_fft, n)
    else:
        seasonal_comp = np.zeros(n)
    resid_obs = (detr_i - seasonal_comp)[mask]               # 噪声只在观测点上估（插值段人为光滑）
    mad = float(np.median(np.abs(resid_obs - np.median(resid_obs))))
    noise_var = (1.4826 * mad) ** 2
    signal_var = float(np.var(fit_full + seasonal_comp))
    if noise_var <= 1e-12:
        feats["SNR"] = 60.0
    elif signal_var <= 1e-12:
        feats["SNR"] = -60.0
    else:
        feats["SNR"] = float(np.clip(10.0 * np.log10(signal_var / noise_var), -60.0, 60.0))

    # ── 其余 P 槽位 ──
    feats["acf1"] = _acf1_masked(raw, mask)
    feats["stationarity_adf"] = _adf_pvalue(xi)              # 语义=感知插值序列的 ADF
    p_norm = np.abs(np.fft.rfft(detr_i - detr_i.mean())) ** 2
    p_norm = p_norm[1:]
    p_norm = p_norm[p_norm > 0]
    if p_norm.size > 1:
        p_norm = p_norm / p_norm.sum()
        ent = -np.sum(p_norm * np.log(p_norm))
        feats["spectral_entropy"] = float(np.clip(ent / np.log(p_norm.size), 0.0, 1.0))
    feats["lumpiness"] = _lumpiness_masked(raw, mask)
    feats["outlier_density"] = _outlier_density(obs)         # 与 P0 同式（本就只依赖观测值集合）
    return feats


def p1a_vectors(series) -> Dict[str, np.ndarray]:
    """→ {"d": (4,), "p": (9,), "c": (3,)} 有序向量（重放 runner / Router 消费）。"""
    f = p1a_feats(series)
    return {"d": np.array([f[k] for k in P1A_D_FEATS], float),
            "p": np.array([f[k] for k in P1A_P_FEATS], float),
            "c": np.array([f[k] for k in P1A_C_FEATS], float)}
