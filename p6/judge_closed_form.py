"""p6/judge_closed_form.py — P6 闭式可归因判官（canonical ID: dlinear_closed_form_v1）。

toy-only 机械实现：与 evaluators/_torch_models.DLinear 语义对齐的闭式（ridge）版本，
用于两级冻结流程中的可归因判官原型。模块级只依赖 numpy；全程无 RNG、确定性。

协议规格（dlinear_closed_form_v1）
----------------------------------
- 特征映射 phi(window)：window 长 L=48（z-scored）；
  trend = k=25 的 replicate-pad(k//2 两侧) 滑动平均、stride 1、取前 L 个输出
  （与 torch 的 F.pad(mode="replicate") + F.avg_pool1d 语义一致）；
  season = window - trend；特征 = [trend(48); season(48); 1]（截距列在最后），维度 97。
- 窗口化：per-series 先 z-score（mean/std 取自该序列 history，std 下限 1e-8，状态记录）；
  滑窗 stride（默认 4）：输入 h[t:t+48]、目标 h[t+48:t+96]，全在 history 内；
  每序列窗口配额 window_cap（默认 None=全部；取时间序上最前的 window_cap 个，确定性）。
- 充分统计量：每序列 G_i=Φ_iᵀΦ_i（97×97）、C_i=Φ_iᵀY_i（97×48）。
  域拟合：G=ΣG_i+λR（R=单位阵但截距对角=0，截距不受罚），W*=np.linalg.solve(G, ΣC_i)。
- series_weight（防长序列支配训练，v4 外审要求）：None=原始 pooled（每窗等权）；
  "equal"=等权序列——每序列统计量乘 w_i = n̄/n_i（n̄=有窗序列的平均窗数），
  Σ w_i·n_i = Σ n_i 保持总权重不变（λ 语义可比）；n_windows=0 的序列 w_i=0。
  替换反事实下权重按"实际训练的组成"重算（e 与 H 同长时权重相同 → 恒等替换仍严格 0.0）。
- 评估协议：每序列 eval 输入 = prepared history 的最后 48 点（z-scored），预测 48 步，
  与该序列 future（用同一 z-score 状态标准化）算 RMSE；
  batch utility = 各序列 RMSE 的等权均值（先 per-series 后聚合）。
- 三效应替换反事实 replacement_effects：对"把序列 i 的 prepared 版本从 H 换成 e"给出
  train / context / joint 三个精确量。数值实现说明：语义上是"充分统计量减旧加新 + 重解"，
  实现取其数值精确形式——缓存每序列 (G_i, C_i) 并按原次序整体重组（仅第 i 项替换后重新求和、
  重新 solve），避免浮点 (A-B)+B ≠ A 的消去误差，使 e ≡ H（bit 级相同输入）时三效应严格为 0.0。

红线：本模块只 import numpy；不读任何真实数据/结果文件；无 RNG。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROTOCOL_ID = "dlinear_closed_form_v1"
CONTEXT_LEN = 48                     # L：输入窗口长
HORIZON = 48                         # H：预测步数
KERNEL = 25                          # 滑动平均核
PHI_DIM = 2 * CONTEXT_LEN + 1        # 97 = [trend(48); season(48); 1]
INTERCEPT_IDX = PHI_DIM - 1          # 截距列 = 最后一列（index 96）
DEFAULT_LAM = 1e-3                   # ridge λ 默认值
DEFAULT_STRIDE = 4                   # 滑窗步长默认值
STD_FLOOR = 1e-8                     # z-score std 下限
_WINDOW_TOTAL = CONTEXT_LEN + HORIZON  # 96：一个训练窗口占用的 history 长度


# ============================== 数据结构 ==============================
@dataclass
class SeriesView:
    """一条序列的原始尺度视图（z-score 在内部完成）。history/future 均为 1-D float64。"""

    uid: str
    history: np.ndarray
    future: np.ndarray

    def __post_init__(self) -> None:
        self.history = np.asarray(self.history, dtype=np.float64).ravel()
        self.future = np.asarray(self.future, dtype=np.float64).ravel()


@dataclass
class SeriesStats:
    """一条序列的充分统计量 + 评估所需状态（全部在 z-scored 空间）。"""

    uid: str
    G: np.ndarray            # (97, 97) = Φᵢᵀ Φᵢ
    C: np.ndarray            # (97, 48) = Φᵢᵀ Yᵢ
    n_windows: int
    mean: float              # z-score 状态（取自该序列 history）
    std: float
    eval_input: np.ndarray   # (48,) prepared history 最后 48 点（z-scored）
    future_norm: np.ndarray  # (48,) future 前 48 点（同一 z-score 状态）


@dataclass
class DomainFit:
    """一次域拟合的结果（W* + 逐序列评估）。"""

    protocol: str
    W: np.ndarray                    # (97, 48)
    stats: List[SeriesStats]
    lam: float
    stride: int
    window_cap: Optional[int]
    per_series_rmse: np.ndarray      # (n_series,)
    utility: float                   # 等权均值 RMSE（负向指标，越小越好）
    n_windows_total: int
    series_weight: Optional[str] = None   # None | "equal"


@dataclass(frozen=True)
class EffectPair:
    """一个反事实效应：batch 均值 RMSE 变化 + 序列 i 自身 RMSE 变化（负=改善）。"""

    batch_delta: float
    self_delta: float


@dataclass(frozen=True)
class ReplacementEffects:
    """三效应替换反事实（序列 i：H → e）。"""

    protocol: str
    index: int
    uid_h: str
    uid_e: str
    baseline_utility: float
    baseline_self_rmse: float
    train_effect: EffectPair    # 只换训练统计量 (G_i, C_i)，eval 全保持 H 版
    context_effect: EffectPair  # 模型保持 H 拟合，只换序列 i 的 eval 输入/参照为 e 版
    joint_effect: EffectPair    # 两者都换


# ============================== 特征映射 ==============================
def zscore_state(history: np.ndarray) -> Tuple[float, float]:
    """z-score 状态：mean/std 取自该序列 history（ddof=0），std 下限 STD_FLOOR。"""
    h = np.asarray(history, dtype=np.float64)
    mean = float(h.mean())
    std = float(h.std())
    if std < STD_FLOOR:
        std = STD_FLOOR
    return mean, std


def moving_average_replicate(x: np.ndarray, kernel: int = KERNEL) -> np.ndarray:
    """replicate-pad(kernel//2 两侧) 滑动平均、stride 1、取前 len(x) 个输出。

    与 torch `F.avg_pool1d(F.pad(x, (k//2, k//2), mode="replicate"), k, stride=1)[..., :L]`
    语义一致：逐窗口独立求均值（不用 cumsum，以贴近 avg_pool1d 的逐窗求和）。
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    pad = kernel // 2
    padded = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    return windows.mean(axis=-1)[: x.shape[0]]


def phi(window: np.ndarray) -> np.ndarray:
    """特征映射：window(48, z-scored) → [trend(48); season(48); 1]，维度 97。"""
    w = np.asarray(window, dtype=np.float64).ravel()
    if w.shape[0] != CONTEXT_LEN:
        raise ValueError(f"phi 需要长度 {CONTEXT_LEN} 的窗口，得到 {w.shape[0]}")
    trend = moving_average_replicate(w, KERNEL)
    out = np.empty(PHI_DIM, dtype=np.float64)
    out[:CONTEXT_LEN] = trend
    out[CONTEXT_LEN : 2 * CONTEXT_LEN] = w - trend
    out[INTERCEPT_IDX] = 1.0
    return out


# ============================== 窗口化与充分统计量 ==============================
def window_starts(
    n_hist: int, stride: int = DEFAULT_STRIDE, window_cap: Optional[int] = None
) -> List[int]:
    """训练窗口起点：t = 0, stride, 2*stride, ...，要求 t+96 ≤ n_hist（全在 history 内）。

    window_cap 非 None 时取最前的 window_cap 个（确定性配额）。
    """
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1，得到 {stride}")
    if n_hist < _WINDOW_TOTAL:
        starts: List[int] = []
    else:
        starts = list(range(0, n_hist - _WINDOW_TOTAL + 1, stride))
    if window_cap is not None:
        if window_cap < 0:
            raise ValueError(f"window_cap 必须 ≥ 0 或 None，得到 {window_cap}")
        starts = starts[:window_cap]
    return starts


@dataclass
class _Prepared:
    """内部：一条序列 z-score 后的设计矩阵与评估状态。"""

    uid: str
    mean: float
    std: float
    Phi: np.ndarray          # (n_windows, 97)
    Y: np.ndarray            # (n_windows, 48)
    eval_input: np.ndarray   # (48,)
    future_norm: np.ndarray  # (48,)


def _prepare(
    view: SeriesView, stride: int, window_cap: Optional[int]
) -> _Prepared:
    h = np.asarray(view.history, dtype=np.float64).ravel()
    f = np.asarray(view.future, dtype=np.float64).ravel()
    if h.shape[0] < CONTEXT_LEN:
        raise ValueError(
            f"series {view.uid!r}: history 长 {h.shape[0]} < {CONTEXT_LEN}，无法构造 eval 输入"
        )
    if f.shape[0] < HORIZON:
        raise ValueError(
            f"series {view.uid!r}: future 长 {f.shape[0]} < {HORIZON}"
        )
    mean, std = zscore_state(h)
    hn = (h - mean) / std
    starts = window_starts(h.shape[0], stride=stride, window_cap=window_cap)
    Phi = np.empty((len(starts), PHI_DIM), dtype=np.float64)
    Y = np.empty((len(starts), HORIZON), dtype=np.float64)
    for r, t in enumerate(starts):
        Phi[r] = phi(hn[t : t + CONTEXT_LEN])
        Y[r] = hn[t + CONTEXT_LEN : t + _WINDOW_TOTAL]
    return _Prepared(
        uid=view.uid,
        mean=mean,
        std=std,
        Phi=Phi,
        Y=Y,
        eval_input=hn[-CONTEXT_LEN:].copy(),
        future_norm=(f[:HORIZON] - mean) / std,
    )


def _stats_from_prepared(p: _Prepared) -> SeriesStats:
    return SeriesStats(
        uid=p.uid,
        G=p.Phi.T @ p.Phi,
        C=p.Phi.T @ p.Y,
        n_windows=int(p.Phi.shape[0]),
        mean=p.mean,
        std=p.std,
        eval_input=p.eval_input,
        future_norm=p.future_norm,
    )


def series_stats(
    view: SeriesView,
    stride: int = DEFAULT_STRIDE,
    window_cap: Optional[int] = None,
) -> SeriesStats:
    """一条序列的充分统计量 G_i, C_i + 评估状态。history 不足 96 时 n_windows=0（仅参与评估）。"""
    return _stats_from_prepared(_prepare(view, stride, window_cap))


# ============================== 求解 ==============================
def ridge_matrix() -> np.ndarray:
    """R = 单位阵但截距对角=0（截距不受罚）。每次返回新副本。"""
    R = np.eye(PHI_DIM, dtype=np.float64)
    R[INTERCEPT_IDX, INTERCEPT_IDX] = 0.0
    return R


def _series_weights(
    n_windows: Sequence[int], series_weight: Optional[str]
) -> np.ndarray:
    """逐序列训练权重：None → 全 1（每窗等权）；"equal" → w_i = n̄/n_i（等权序列）。

    n̄ 取有窗序列（n_i>0）的平均窗数；n_i=0 的序列权重 0（它们本就不贡献训练统计量）。
    Σ w_i·n_i = Σ n_i：总有效权重不变，λ 的相对强度与 None 模式可比。
    """
    n = np.asarray(list(n_windows), dtype=np.float64)
    if series_weight is None:
        return np.ones(n.shape[0], dtype=np.float64)
    if series_weight != "equal":
        raise ValueError(f"series_weight 须为 None 或 'equal'，得到 {series_weight!r}")
    pos = n > 0
    if not pos.any():
        raise ValueError("域内没有任何训练窗口（所有序列 history < 96 或配额为 0）")
    nbar = float(n[pos].mean())
    w = np.zeros(n.shape[0], dtype=np.float64)
    w[pos] = nbar / n[pos]
    return w


def solve_weights(
    stats: Sequence[SeriesStats],
    lam: float = DEFAULT_LAM,
    series_weight: Optional[str] = None,
) -> np.ndarray:
    """充分统计量路：G = Σ w_i·G_i + λR，W* = solve(G, Σ w_i·C_i)。

    按 stats 列表原次序逐项累加（固定求和次序 → 确定性；
    替换反事实沿用同一次序，保证 bit 级可比）。
    series_weight=None 时走原始无乘法路径（与历史行为 bit 级一致）。
    """
    if sum(s.n_windows for s in stats) == 0:
        raise ValueError("域内没有任何训练窗口（所有序列 history < 96 或配额为 0）")
    G = np.zeros((PHI_DIM, PHI_DIM), dtype=np.float64)
    C = np.zeros((PHI_DIM, HORIZON), dtype=np.float64)
    if series_weight is None:
        for s in stats:
            G = G + s.G
            C = C + s.C
    else:
        w = _series_weights([s.n_windows for s in stats], series_weight)
        for j, s in enumerate(stats):
            G = G + w[j] * s.G
            C = C + w[j] * s.C
    G = G + lam * ridge_matrix()
    return np.linalg.solve(G, C)


def solve_from_design(
    Phi: np.ndarray, Y: np.ndarray, lam: float = DEFAULT_LAM
) -> np.ndarray:
    """全量堆叠路：G = ΦᵀΦ + λR，W* = solve(G, ΦᵀY)。"""
    Phi = np.asarray(Phi, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if Phi.shape[0] == 0:
        raise ValueError("设计矩阵为空：没有任何训练窗口")
    G = Phi.T @ Phi + lam * ridge_matrix()
    return np.linalg.solve(G, Phi.T @ Y)


# ============================== 评估 ==============================
def predict(W: np.ndarray, context_norm: np.ndarray) -> np.ndarray:
    """闭式预测：yhat = phi(context) @ W*，(48,)，z-scored 空间。"""
    return phi(context_norm) @ W


def series_rmse(W: np.ndarray, stats: SeriesStats) -> float:
    """一条序列的 RMSE：eval 输入=最后 48 点，预测 48 步 vs future_norm。"""
    err = predict(W, stats.eval_input) - stats.future_norm
    return float(np.sqrt(np.mean(err * err)))


def evaluate(
    W: np.ndarray, stats: Sequence[SeriesStats]
) -> Tuple[np.ndarray, float]:
    """逐序列 RMSE + batch utility（等权均值；先 per-series 后聚合）。"""
    rmses = np.array([series_rmse(W, s) for s in stats], dtype=np.float64)
    return rmses, float(rmses.mean())


# ============================== 域拟合（双路） ==============================
def fit_domain(
    views: Sequence[SeriesView],
    lam: float = DEFAULT_LAM,
    stride: int = DEFAULT_STRIDE,
    window_cap: Optional[int] = None,
    series_weight: Optional[str] = None,
) -> DomainFit:
    """充分统计量路域拟合：逐序列 (G_i, C_i)（可选 w_i 加权）累加 + 一次 solve。"""
    stats = [series_stats(v, stride=stride, window_cap=window_cap) for v in views]
    W = solve_weights(stats, lam=lam, series_weight=series_weight)
    rmses, utility = evaluate(W, stats)
    return DomainFit(
        protocol=PROTOCOL_ID,
        W=W,
        stats=stats,
        lam=lam,
        stride=stride,
        window_cap=window_cap,
        per_series_rmse=rmses,
        utility=utility,
        n_windows_total=int(sum(s.n_windows for s in stats)),
        series_weight=series_weight,
    )


def fit_domain_rebuild(
    views: Sequence[SeriesView],
    lam: float = DEFAULT_LAM,
    stride: int = DEFAULT_STRIDE,
    window_cap: Optional[int] = None,
    series_weight: Optional[str] = None,
) -> DomainFit:
    """全量堆叠路域拟合：把所有序列的 Φ/Y 堆成大矩阵直接求解（对拍用独立实现）。

    series_weight="equal" 经行缩放实现：每序列的 Φ/Y 行乘 √w_i
    （(√w·Φ)ᵀ(√w·Φ) = w·ΦᵀΦ，与充分统计量路的加权语义一致）。
    """
    preps = [_prepare(v, stride, window_cap) for v in views]
    w = _series_weights([p.Phi.shape[0] for p in preps], series_weight)
    scaled = [
        (p.Phi * np.sqrt(w[j]), p.Y * np.sqrt(w[j])) for j, p in enumerate(preps)
    ]
    Phi_all = (
        np.concatenate([ph for ph, _ in scaled], axis=0)
        if scaled
        else np.zeros((0, PHI_DIM), dtype=np.float64)
    )
    Y_all = (
        np.concatenate([y for _, y in scaled], axis=0)
        if scaled
        else np.zeros((0, HORIZON), dtype=np.float64)
    )
    W = solve_from_design(Phi_all, Y_all, lam=lam)
    stats = [_stats_from_prepared(p) for p in preps]
    rmses, utility = evaluate(W, stats)
    return DomainFit(
        protocol=PROTOCOL_ID,
        W=W,
        stats=stats,
        lam=lam,
        stride=stride,
        window_cap=window_cap,
        per_series_rmse=rmses,
        utility=utility,
        n_windows_total=int(sum(p.Phi.shape[0] for p in preps)),
        series_weight=series_weight,
    )


# ============================== torch 权重拆解（仅产 numpy 数组） ==============================
def torch_dlinear_state(W: np.ndarray) -> Dict[str, np.ndarray]:
    """把 W* (97,48) 拆成 evaluators/_torch_models.DLinear(L=48,H=48) 的 state_dict 数组。

    闭式：yhat = trend @ Wt + season @ Ws + b，其中 Wt=W[:48], Ws=W[48:96], b=W[96]。
    torch：lin(x) = x @ weight.T + bias → lin_trend.weight = Wtᵀ (H,L)、
    lin_season.weight = Wsᵀ；截距整体并入 lin_trend.bias，lin_season.bias = 0。
    本函数只产 numpy 数组（模块不 import torch）。
    """
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (PHI_DIM, HORIZON):
        raise ValueError(f"W* 形状应为 {(PHI_DIM, HORIZON)}，得到 {W.shape}")
    return {
        "lin_trend.weight": W[:CONTEXT_LEN].T.copy(),
        "lin_trend.bias": W[INTERCEPT_IDX].copy(),
        "lin_season.weight": W[CONTEXT_LEN : 2 * CONTEXT_LEN].T.copy(),
        "lin_season.bias": np.zeros(HORIZON, dtype=np.float64),
    }


# ============================== 三效应替换反事实 ==============================
def replacement_effects(
    views_H: Sequence[SeriesView],
    i: int,
    view_e: SeriesView,
    lam: float = DEFAULT_LAM,
    stride: int = DEFAULT_STRIDE,
    window_cap: Optional[int] = None,
    series_weight: Optional[str] = None,
) -> ReplacementEffects:
    """把序列 i 的 prepared 版本从 H 换成 e 的三个精确反事实效应。

    - train_effect：只替换序列 i 的训练统计量 (G_i, C_i)；所有序列 eval 保持 H 版。
    - context_effect：模型保持 H 拟合；只替换序列 i 的 eval 输入（及其 z-score
      状态与 future 参照）为 e 版。
    - joint_effect：两者都替换。
    每个效应 = (batch 均值 RMSE 变化, 序列 i 自身 RMSE 变化)，负=改善。

    精确性：缓存逐序列 (G_i, C_i)，替换后按原次序整体重新求和并重新 solve
    （"减旧加新"的数值精确形式，无近似）；e 与 H bit 级相同时三效应严格为 0.0。
    """
    views_H = list(views_H)
    n = len(views_H)
    if not (0 <= i < n):
        raise IndexError(f"i={i} 超出范围 [0, {n})")

    stats_H = [series_stats(v, stride=stride, window_cap=window_cap) for v in views_H]
    stats_e = series_stats(view_e, stride=stride, window_cap=window_cap)
    stats_repl = list(stats_H)
    stats_repl[i] = stats_e

    # 权重按"实际训练的组成"分别重算（prepared 同长时两侧权重相同 → 恒等替换仍严格 0.0）
    W_H = solve_weights(stats_H, lam=lam, series_weight=series_weight)      # 基线模型（全 H）
    W_T = solve_weights(stats_repl, lam=lam, series_weight=series_weight)   # 训练侧替换后的模型

    rmse_base, u_base = evaluate(W_H, stats_H)
    rmse_train, u_train = evaluate(W_T, stats_H)     # train：模型换、eval 保持 H
    rmse_ctx, u_ctx = evaluate(W_H, stats_repl)      # context：模型保持 H、eval i→e
    rmse_joint, u_joint = evaluate(W_T, stats_repl)  # joint：两者都换

    return ReplacementEffects(
        protocol=PROTOCOL_ID,
        index=i,
        uid_h=views_H[i].uid,
        uid_e=view_e.uid,
        baseline_utility=u_base,
        baseline_self_rmse=float(rmse_base[i]),
        train_effect=EffectPair(
            batch_delta=u_train - u_base,
            self_delta=float(rmse_train[i] - rmse_base[i]),
        ),
        context_effect=EffectPair(
            batch_delta=u_ctx - u_base,
            self_delta=float(rmse_ctx[i] - rmse_base[i]),
        ),
        joint_effect=EffectPair(
            batch_delta=u_joint - u_base,
            self_delta=float(rmse_joint[i] - rmse_base[i]),
        ),
    )
