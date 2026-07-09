"""data/load_real.py — 跨 domain 真实锚（plan.md §2 / 附录 A.3 / P0：验 C3 跨域）。

把 Monash TSF 的真实单变量序列当作"干净信号源"，**逐序列 z-score** 后用与合成完全相同的退化
网格（synthetic_gen 的 _degrade / _inject_anomalies + grid PATTERN_PARAMS）注入受控劣化，产出
与 synthetic_gen 同形的 RawSeries —— 因此 BatchBuilder / evaluators / slow_path 全链路零改动即可吃真数据。

设计要点（与合成长跑可比 + 真实信号是唯一变量）：
  • **真实信号 + 受控网格退化**：clean = 真实 Monash 序列（z-score），degraded 用 G_hi/G_lo × full/miss
    四预设注入 → cell 分布可控、可与 L2/L3/L4 合成长跑逐 cell 对照；唯一变化 = "真实信号结构"。
  • **per-series z-score**（必需）：真实 std 跨域差 5 个数量级（nn5≈6 vs fred≈1.5e5）；不归一则
    网格里的绝对噪声(0.05/0.70)与 5σ 离群尺度全失真，且冻结编码器（合成上 ~单位尺度预训）完全 OOD。
  • **真实 period 入 RawSeries.period**（来自 config 频率元数据）：seasonal_naive floor 用它；注意
    period **struct_feat**（FFT 主频）在强趋势真数据上会被低频趋势主导（→≈序列长），是真实诊断现象，
    但 binning 只用 SNR×missing，不受影响。

数据来源：默认读本地 `AdaCTS/data/monash_real.npz` 的 clean 数组（12 序列 / 3 域，无需联网）。
扩充更丰富语料见 `build_real_npz()`（懒加载 AdaCTS loader，需 HF 网络）。
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .synthetic_gen import (
    RawSeries, H_FORECAST, K_ANOM, PATTERN_PARAMS, _degrade, _inject_anomalies,
)

# 默认真实语料（AdaCTS 已离线准备：clean = NaN 填补后的真实单变量序列）
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_NPZ = _REPO_ROOT / "AdaCTS" / "data" / "monash_real.npz"

# forecast 退化网格（与 synthetic_gen 网格预设一致 → 跨 SNR×missing 四 cell）
FORECAST_PRESETS = ("G_hi_full", "G_hi_miss", "G_lo_full", "G_lo_miss")
# anomaly 只用带 missing 的两格（与合成长跑一致；异常注入后 forecast 离群结构保留）
ANOMALY_PRESETS = ("G_hi_miss", "G_lo_miss")
# classify 用全 4 格 → 与 forecast 同 (SNR×missing) cell 坐标，便于逐 cell 跨任务对照（C1）
CLASSIFY_PRESETS = ("G_hi_full", "G_hi_miss", "G_lo_full", "G_lo_miss")

MIN_LEN = 96 + H_FORECAST     # 至少能切一个 forecast 窗口（L_WIN+H=96）+ 留 H 真未来


@dataclass
class RealSignal:
    """一条真实单变量锚信号（已 z-score 的 clean）。"""
    config: str
    item_id: str
    period: int
    clean: np.ndarray            # z-score 后的真实序列（作为干净信号源）


@dataclass
class RealClassSignal:
    """一条真实带标签序列（z-score clean + 类标签）—— classify 锚（ECG5000）。"""
    config: str
    item_id: str
    label: int
    clean: np.ndarray


# ════════════════════════════ 加载真实信号 ════════════════════════════
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.nanmean(x))
    s = float(np.nanstd(x))
    return (x - m) / (s if s > 1e-9 else 1.0)


def load_signals(npz_path: Optional[str] = None, *, configs: Optional[List[str]] = None,
                 standardize: bool = True, min_len: int = MIN_LEN) -> List[RealSignal]:
    """读 Monash clean 序列 → List[RealSignal]（默认 z-score）。

    npz 由 AdaCTS.data.load_monash.save_real 写：clean(object array) + 同名 .meta.jsonl(config/period/item_id)。
    """
    path = pathlib.Path(npz_path) if npz_path else DEFAULT_NPZ
    if not path.exists():
        raise FileNotFoundError(
            f"真实语料缺失：{path}\n用 build_real_npz(...) 或 `python -m AdaCTS.data.load_monash` 生成。")
    with np.load(path, allow_pickle=True) as d:
        cleans = [np.asarray(a, dtype=float) for a in d["clean"]]
    meta_path = path.with_suffix(".meta.jsonl")
    metas: List[dict] = []
    if meta_path.exists():
        metas = [json.loads(ln) for ln in meta_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    out: List[RealSignal] = []
    for i, clean in enumerate(cleans):
        m = metas[i] if i < len(metas) else {}
        cfg = str(m.get("config", "unknown"))
        if configs is not None and cfg not in configs:
            continue
        clean = clean[np.isfinite(clean)] if not np.all(np.isfinite(clean)) else clean
        if clean.size < min_len:
            continue
        sig = _zscore(clean) if standardize else clean.astype(float)
        out.append(RealSignal(cfg, str(m.get("item_id", i)), int(m.get("period", 24) or 24), sig))
    if not out:
        raise ValueError(f"无可用真实信号（path={path}, configs={configs}, min_len={min_len}）")
    return out


# ════════════════════════════ RawSeries 工厂（真实 clean + 网格退化）════════════════════════════
def _forecast_from_signal(sig: RealSignal, preset: str, seed: int) -> RawSeries:
    clean = sig.clean
    degraded = _degrade(clean, PATTERN_PARAMS[preset], seed)
    cut = clean.size - H_FORECAST
    return RawSeries(
        pattern=f"{sig.config}:{preset}", task="forecast", seed=seed, period=sig.period,
        obs_scale=float(np.std(clean[cut:])) or 1.0,
        clean=clean, degraded=degraded,
        history=degraded[:cut].copy(), clean_history=clean[:cut].copy(), future=clean[cut:].copy(),
        origin=sig.config, series_uid=f"{sig.config}:{sig.item_id}",   # 同基底×多退化seed 共享 uid
    )


def _anomaly_from_signal(sig: RealSignal, preset: str, seed: int) -> RawSeries:
    clean = sig.clean
    degraded = _degrade(clean, PATTERN_PARAMS[preset], seed)
    base_std = float(np.std(clean)) or 1.0
    anom, pos = _inject_anomalies(degraded, base_std, seed)
    return RawSeries(
        pattern=f"{sig.config}:{preset}", task="anomaly_detection", seed=seed, period=sig.period,
        obs_scale=base_std, clean=clean, degraded=degraded,
        anomaly_input=anom, anomaly_positions=pos,
        origin=sig.config, series_uid=f"{sig.config}:{sig.item_id}",
    )


def make_real_forecast_batch(signals: List[RealSignal], preset: str,
                             n_per_signal: int = 4, seed0: int = 0) -> List[RawSeries]:
    """每条真实信号 × n_per_signal 个退化 seed → forecast RawSeries（同信号多退化实现 = 一批样本）。"""
    return [_forecast_from_signal(sig, preset, seed0 + si * 1000 + k)
            for si, sig in enumerate(signals) for k in range(n_per_signal)]


def make_real_anomaly_batch(signals: List[RealSignal], preset: str,
                            n_per_signal: int = 4, seed0: int = 100) -> List[RawSeries]:
    return [_anomaly_from_signal(sig, preset, seed0 + si * 1000 + k)
            for si, sig in enumerate(signals) for k in range(n_per_signal)]


def build_real_corpus(signals: Optional[List[RealSignal]] = None, *,
                      n_per_signal: int = 4, tasks=("forecast", "anomaly_detection"),
                      forecast_presets=FORECAST_PRESETS, anomaly_presets=ANOMALY_PRESETS,
                      ) -> List[RawSeries]:
    """一站式：真实信号 → 跨 (preset × task) 全量 RawSeries（喂 BatchBuilder.add_raw_series）。"""
    sigs = signals if signals is not None else load_signals()
    corpus: List[RawSeries] = []
    if "forecast" in tasks:
        for pre in forecast_presets:
            corpus += make_real_forecast_batch(sigs, pre, n_per_signal=n_per_signal, seed0=0)
    if "anomaly_detection" in tasks:
        for pre in anomaly_presets:
            corpus += make_real_anomaly_batch(sigs, pre, n_per_signal=n_per_signal, seed0=100)
    return corpus


# ════════════════════════════ classify 锚（ECG5000 真实标签 + 网格退化）════════════════════════════
def load_class_signals(*, cache_path: Optional[str] = None, top_k: int = 3,
                       cap_per_class: Optional[int] = None, max_signals: Optional[int] = None,
                       standardize: bool = True, seed: int = 0) -> List[RealClassSignal]:
    """ECG5000 → List[RealClassSignal]（已 z-score）。懒导入 load_ecg5000 避免无谓联网。

    max_signals: 截断总条数（控运行时；按已平衡/打乱顺序取前 N，保持类大致均衡）。
    """
    from .load_ecg5000 import load_ecg5000
    X, y = load_ecg5000(cache_path, top_k=top_k, cap_per_class=cap_per_class,
                        standardize=standardize, seed=seed)
    if max_signals is not None and X.shape[0] > max_signals:
        X, y = X[:max_signals], y[:max_signals]
    return [RealClassSignal("ecg5000", str(i), int(y[i]), X[i].astype(float))
            for i in range(X.shape[0])]


def _classify_from_signal(sig: RealClassSignal, preset: str, seed: int) -> RawSeries:
    clean = sig.clean
    degraded = _degrade(clean, PATTERN_PARAMS[preset], seed)
    return RawSeries(
        pattern=f"{sig.config}:{preset}", task="classification", seed=seed,
        period=PATTERN_PARAMS[preset]["period"], obs_scale=float(np.std(clean)) or 1.0,
        clean=clean, degraded=degraded, label=int(sig.label),
        origin=sig.config, series_uid=f"{sig.config}:{sig.item_id}",
    )


def make_real_classify_batch(class_signals: List[RealClassSignal], preset: str,
                             n_per_signal: int = 1, seed0: int = 200) -> List[RawSeries]:
    """每条带标签信号 × n_per_signal 退化 seed → classify RawSeries（同 preset = 同 SNR×missing cell）。"""
    return [_classify_from_signal(sig, preset, seed0 + si * 1000 + k)
            for si, sig in enumerate(class_signals) for k in range(n_per_signal)]


def build_real_classify_corpus(class_signals: Optional[List[RealClassSignal]] = None, *,
                               n_per_signal: int = 1, presets=CLASSIFY_PRESETS) -> List[RawSeries]:
    """一站式：真实带标签信号 → 跨 4 退化预设的 classify RawSeries（喂 BatchBuilder）。

    同一批信号在每个 preset 下各退化一次 → 落入不同 (SNR×missing) cell，类标签不变 →
    cell 间唯一变量 = 退化模式 → 可直接看"最优清洗强度是否随 cell 变（且与 forecast 相反）"。
    """
    sigs = class_signals if class_signals is not None else load_class_signals()
    corpus: List[RawSeries] = []
    for pre in presets:
        corpus += make_real_classify_batch(sigs, pre, n_per_signal=n_per_signal)
    return corpus


# ════════════════════════════ 编码器留出划分（防泄漏，供 E2 真实预训）════════════════════════════
def split_encoder_eval(signals: List[RealSignal], *, frac: float = 0.5,
                       seed: int = 0) -> tuple[List[RealSignal], List[RealSignal]]:
    """按信号（非样本）分 encoder-pretrain / eval 两不相交集 —— 真实编码器预训必须 leave-signal-out 防泄漏。

    在每个 config 内分层取样，保证两侧 domain 覆盖一致。
    """
    rng = np.random.default_rng(seed)
    by_cfg: Dict[str, List[RealSignal]] = {}
    for s in signals:
        by_cfg.setdefault(s.config, []).append(s)
    pre, ev = [], []
    for cfg, group in by_cfg.items():
        idx = rng.permutation(len(group))
        k = max(1, int(round(frac * len(group)))) if len(group) > 1 else 1
        for j, gi in enumerate(idx):
            (pre if j < k else ev).append(group[gi])
    if not ev:                                   # 退化（每 config 1 条）→ 不分，复用全部（仅诊断用）
        ev = list(signals)
    return pre, ev


# ════════════════════════════ 可选：联网扩充语料 ════════════════════════════
def build_real_npz(out_path: str, *, configs: Optional[List[str]] = None,
                   per_config: int = 8, missing: float = 0.0, noise: float = 0.0,
                   outliers: float = 0.0, seed: int = 0) -> str:
    """懒加载 AdaCTS loader 下载更丰富 Monash 语料并存为本格式（需 HF 网络）。

    注意：本框架自己注入退化（degraded 在 RawSeries 工厂里现造），故这里 missing/noise/outliers=0，
    只取**干净**真实序列；save_real 的 clean 字段即我们需要的信号源。
    """
    import sys
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from AdaCTS.data.load_monash import build_dataset, save_real, DEFAULT_CONFIGS
    records = build_dataset(configs or DEFAULT_CONFIGS, per_config=per_config,
                            missing=missing, noise=noise, outliers=outliers, seed=seed)
    save_real(records, out_path)
    return out_path
