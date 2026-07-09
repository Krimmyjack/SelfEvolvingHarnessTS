"""e32_policy.py — E-3.2 七臂策略 + held-out policy regret 评估（骨架，A-37，2026-07-04）。

臂（6 主比较 + 1 诊断）：
  global      全局单动作（train 均值 argmin）
  d_lookup    D-only Lookup（cell=SNRbin×missbin 查表）——F0 证明其结构不安全的那条路由
  d_gbdt      D-only GBDT（连续 D=SNR+missing_rate）——判据(vi) 连续 SNR residualization 的对照臂
  p_gbdt      P-only GBDT（8 维结构特征，不含 SNR/missing_rate）
  dp_gbdt     D+P GBDT
  dp_abstain  D+P GBDT + abstain（预测优势 < κ·ensemble std → 回退 FALLBACK_ACTION）
  oracle_struct（诊断，不进主比较）cell×origin 查表 = 结构路由信息上界

主指标 = held-out policy regret（相对 per-uid oracle），非动作分类准确率（D-3.2e）。
fallback 冻结（A-37④）= `v_median`（median@5, incumbent）；κ=1 dev 默认，confirmatory 前最终冻结。

已知机制边界（写给 E-3.2 设计，2026-07-04）：ensemble-std abstain 只在**预测不确定**处触发；
若 held-out 结构与训练结构共享特征签名（aliasing，如 S_both↔S_trend）、模型**自信地外推**，
std 低 → abstain 不触发 → 判据 (v) 是经验问题而非机制保证——正式 E-3.2 须报 abstain 触发率
per subgroup，并把 aliasing 情形（S_both）单列。

**正式跑门（A-37/A-38）**：本模块仅 toy/单元测试验证；正式六臂等 A-31e 补样 + 协议冻结后跑
（不在失衡语料上看 D+P 信号，防特征/阈值/abstain 设计被污染）。origin 只进分层/审计/oracle 臂。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold

EPS = 0.03
DELTA_SAFE = 0.05
FALLBACK_ACTION = "v_median"            # A-37④ 冻结：incumbent 轻 median
KAPPA = 1.0                             # abstain 触发系数（dev 默认；confirmatory 前冻结）
N_ENSEMBLE = 5
D_FEATS = ("SNR", "missing_rate")       # 连续退化坐标（deploy 可得）
P_FEATS = ("period", "trend_strength", "seasonal_strength", "acf1",
           "stationarity_adf", "spectral_entropy", "lumpiness", "outlier_density")

# 剪枝扩充池（A-37①）：冻结的是候选集，不是 D-only 策略
PRUNED_POOL_CORE = ["v_none", "v_median", "v_savgol", "v_stl", "v_wavelet",
                    "v_winsor", "v_winsor_savgol",
                    "f0_median_w9", "f0_median_w15", "f0_median_w25"]
ABLATION_MA = ["f0_ma_w9", "f0_ma_w15", "f0_ma_w25"]   # 消融臂专用，不进核心池


@dataclass
class PolicyData:
    """一行 = 一个 uid。L[i, a] = uid i 在动作 a 下的 held-out（OOF）loss。"""
    uids: List[str]
    actions: List[str]
    L: np.ndarray                        # (n, A)
    X_d: np.ndarray                      # (n, len(D_FEATS)) 连续退化坐标
    X_p: np.ndarray                      # (n, len(P_FEATS)) 结构特征（不含 SNR/missing）
    cell: np.ndarray                     # (n,) str —— D-only Lookup 的键
    origin: np.ndarray                   # (n,) str —— 只进分层/审计/oracle 臂
    X_t: Optional[np.ndarray] = None     # (n, 2) 生成器真实 (noise, miss)（true-D 诊断臂专用，A-39③）

    def __post_init__(self):
        n = len(self.uids)
        assert self.L.shape == (n, len(self.actions))
        assert self.X_d.shape[0] == n and self.X_p.shape[0] == n
        assert FALLBACK_ACTION in self.actions, f"fallback {FALLBACK_ACTION} 必须在动作池内（A-37④）"

    @property
    def n(self) -> int:
        return len(self.uids)

    @property
    def fallback_idx(self) -> int:
        return self.actions.index(FALLBACK_ACTION)

    def X(self, feats: Sequence[str]) -> np.ndarray:
        cols = []
        if "d" in feats:
            cols.append(self.X_d)
        if "p" in feats:
            cols.append(self.X_p)
        if "t" in feats:
            assert self.X_t is not None, "true-D 诊断臂需要 X_t"
            cols.append(self.X_t)
        assert cols, "feats 至少含 'd'/'p'/'t' 之一"
        return np.hstack(cols)


# ══════════════════════════════════════════════════════════════════════════
# 臂：统一接口 fit(data, tr_idx) → picks(data, te_idx) -> (action_idx, abstained_mask)
# ══════════════════════════════════════════════════════════════════════════
class LookupArm:
    """key_fn=None → 全局单动作；否则按 key 查表（train 均值 argmin），unseen key 回退全局。"""

    def __init__(self, key_fn: Optional[Callable[[PolicyData, int], str]] = None):
        self.key_fn = key_fn

    def fit(self, data: PolicyData, tr: np.ndarray) -> "LookupArm":
        self.global_pick = int(np.argmin(data.L[tr].mean(axis=0)))
        self.table: Dict[str, int] = {}
        if self.key_fn is not None:
            keys = np.array([self.key_fn(data, i) for i in tr])
            for k in np.unique(keys):
                sub = data.L[tr[keys == k]]
                self.table[k] = int(np.argmin(sub.mean(axis=0)))
        return self

    def picks(self, data: PolicyData, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.key_fn is None:
            p = np.full(len(te), self.global_pick, int)
        else:
            p = np.array([self.table.get(self.key_fn(data, i), self.global_pick) for i in te], int)
        return p, np.zeros(len(te), bool)


def key_cell(data: PolicyData, i: int) -> str:
    return str(data.cell[i])


def key_cell_origin(data: PolicyData, i: int) -> str:
    return f"{data.cell[i]}|{data.origin[i]}"       # 诊断臂专用：origin 不进学习特征


class GBDTArm:
    """每动作一个 GBDT 回归 loss；policy = argmin 预测。abstain=True 时用 E 个成员的 ensemble：
    **成员 e 跨动作共享 random_state**（同训练子样本 → paired，A-39④），触发规则用 **paired
    advantage**：adv_e = pred_e(fallback) − pred_e(selected)，mean(adv_e) < κ·std(adv_e) → 回退
    fallback（原 std(selected) 版忽略 fallback 自身预测不稳，已废）。"""

    def __init__(self, feats: Sequence[str], abstain: bool = False, seed: int = 0,
                 n_ensemble: int = N_ENSEMBLE, kappa: float = KAPPA):
        self.feats, self.abstain, self.seed = tuple(feats), abstain, seed
        self.E = n_ensemble if abstain else 1
        self.kappa = kappa

    def _gbr(self, e: int) -> GradientBoostingRegressor:
        # random_state 只依赖成员 e（不依赖动作）→ 同成员各动作用相同子采样序列（paired ensemble）；
        # 取模压入 sklearn uint32 合法域（大 seed 如 20260704 会溢出；小 seed 行为不变）
        return GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1,
                                         subsample=0.7,
                                         random_state=(100000 + self.seed * 1000 + e * 7) % 4294967295)

    def fit(self, data: PolicyData, tr: np.ndarray) -> "GBDTArm":
        Xtr = data.X(self.feats)[tr]
        self.models = [[self._gbr(e).fit(Xtr, data.L[tr, a]) for e in range(self.E)]
                       for a in range(len(data.actions))]
        return self

    def picks(self, data: PolicyData, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xte = data.X(self.feats)[te]
        preds = np.stack([[m.predict(Xte) for m in ens] for ens in self.models])  # (A, E, n_te)
        mu = preds.mean(axis=1)                                                   # (A, n_te)
        pick = np.argmin(mu, axis=0)
        abstained = np.zeros(len(te), bool)
        if self.abstain:
            rows = np.arange(len(te))
            adv_e = preds[data.fallback_idx] - preds[pick, :, rows].T   # (E, n_te) paired advantage
            abstained = adv_e.mean(axis=0) < self.kappa * adv_e.std(axis=0)
            pick = np.where(abstained, data.fallback_idx, pick)
        return pick.astype(int), abstained


def make_arms(seed: int = 0) -> "Dict[str, Callable[[], object]]":
    """七臂工厂（每 fold 重新实例化）。oracle_struct 为诊断臂，不进主比较。"""
    return {
        "global":        lambda: LookupArm(None),
        "d_lookup":      lambda: LookupArm(key_cell),
        "d_gbdt":        lambda: GBDTArm(("d",), seed=seed),
        "p_gbdt":        lambda: GBDTArm(("p",), seed=seed),
        "dp_gbdt":       lambda: GBDTArm(("d", "p"), seed=seed),
        "dp_abstain":    lambda: GBDTArm(("d", "p"), abstain=True, seed=seed),
        "oracle_struct": lambda: LookupArm(key_cell_origin),
    }


# ══════════════════════════════════════════════════════════════════════════
# 评估：held-out policy regret + subgroup 安全（vs incumbent）+ LODO
# ══════════════════════════════════════════════════════════════════════════
def _folds(data: PolicyData, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """cell×origin 分层 K-fold（一行一 uid，无重复 → 无需 group 约束）；类太小回退普通 KFold。"""
    strat = np.array([f"{c}|{o}" for c, o in zip(data.cell, data.origin)])
    _, cnt = np.unique(strat, return_counts=True)
    idx = np.arange(data.n)
    if cnt.min() >= n_splits:
        sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(idx[tr], idx[te]) for tr, te in sk.split(idx.reshape(-1, 1), strat)]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(idx[tr], idx[te]) for tr, te in kf.split(idx)]


def _subgroup_stats(data: PolicyData, loss_pick: np.ndarray) -> Dict[str, dict]:
    """per (cell,origin)：Δ vs incumbent（正=优于 incumbent）的 mean/n/正态 95% 单侧 LCB。"""
    inc = data.fallback_idx                                        # incumbent ≡ v_median（同一动作）
    delta = data.L[:, inc] - loss_pick
    out: Dict[str, dict] = {}
    keys = np.array([f"{c}|{o}" for c, o in zip(data.cell, data.origin)])
    for k in np.unique(keys):
        d = delta[keys == k]
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        out[k] = dict(n=int(len(d)), mean=float(d.mean()),
                      lcb=float(d.mean() - 1.645 * se) if np.isfinite(se) else float("-inf"))
    return out


def _summarize(data: PolicyData, loss_pick: np.ndarray, abstained: np.ndarray) -> dict:
    oracle = data.L.min(axis=1)
    sub = _subgroup_stats(data, loss_pick)
    worst = min(sub.values(), key=lambda v: v["mean"])
    worst_lcb = min(sub.values(), key=lambda v: v["lcb"])
    return dict(mean_loss=float(loss_pick.mean()),
                mean_regret=float((loss_pick - oracle).mean()),
                worst_group_mean=float(worst["mean"]),
                worst_group_lcb=float(worst_lcb["lcb"]),
                abstain_rate=float(abstained.mean()),
                subgroups=sub)


def evaluate(data: PolicyData, n_splits: int = 5, seed: int = 0,
             arms: Optional[Dict[str, Callable[[], object]]] = None) -> Dict[str, dict]:
    """arm-in：分层 K-fold，train 上拟合策略、test 上按 pick 记 loss/regret/安全子群。"""
    arms = arms or make_arms(seed)
    picks = {name: np.full(data.n, -1, int) for name in arms}
    abst = {name: np.zeros(data.n, bool) for name in arms}
    for tr, te in _folds(data, n_splits, seed):
        for name, mk in arms.items():
            arm = mk().fit(data, tr)
            p, a = arm.picks(data, te)
            picks[name][te], abst[name][te] = p, a
    out = {}
    for name in arms:
        assert (picks[name] >= 0).all(), "每个 uid 必须恰被一个 test fold 覆盖"
        loss_pick = data.L[np.arange(data.n), picks[name]]
        out[name] = _summarize(data, loss_pick, abst[name])
        out[name]["picks"] = picks[name]
    return out


def lodo_evaluate(data: PolicyData, group_field: str = "origin", seed: int = 0,
                  arms: Optional[Dict[str, Callable[[], object]]] = None) -> Dict[str, Dict[str, dict]]:
    """leave-one-group-out。group_field="origin" = 留一**结构**（level-1，A-37⑤：不得写"跨域"）；
    真实数据接 dataset/domain 字段后同一机制升 level-2。"""
    arms = arms or make_arms(seed)
    groups = getattr(data, group_field)
    out: Dict[str, Dict[str, dict]] = {}
    for g in np.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        res = {}
        for name, mk in arms.items():
            if name == "oracle_struct" and group_field == "origin":
                continue                       # 留一结构下 oracle 表对 held-out 结构无键，无意义
            arm = mk().fit(data, tr)
            p, a = arm.picks(data, te)
            loss_pick = data.L[te, p]
            oracle = data.L[te].min(axis=1)
            inc = data.L[te, data.fallback_idx]
            res[name] = dict(mean_loss=float(loss_pick.mean()),
                             mean_regret=float((loss_pick - oracle).mean()),
                             mean_delta_vs_incumbent=float((inc - loss_pick).mean()),
                             abstain_rate=float(a.mean()), n=int(len(te)))
        out[str(g)] = res
    return out


def verdict_d32e(res: Dict[str, dict], eps: float = EPS, delta_safe: float = DELTA_SAFE,
                 trend_retention_val: Optional[float] = None,
                 perm_p: Optional[float] = None) -> dict:
    """D-3.2e 六判据（A-37③/A-39②）。(iv)/(vi 置换 p) 由正式 runner 传入；缺省 None=未算。"""
    dp, gl, dl = res["dp_gbdt"], res["global"], res["d_lookup"]
    dgb, ab = res["d_gbdt"], res["dp_abstain"]
    season_lcb = min((v["lcb"] for k, v in ab["subgroups"].items() if "S_season" in k),
                     default=float("nan"))
    return {
        "i_dp_beats_global": bool(dp["mean_regret"] < gl["mean_regret"] - eps),
        "ii_dp_beats_dlookup": bool(dp["mean_regret"] < dl["mean_regret"] - eps),
        "iii_season_worst_lcb_ok": bool(season_lcb > -delta_safe) if np.isfinite(season_lcb) else None,
        "iv_trend_retention_ok": (bool(trend_retention_val >= 0.5)
                                  if trend_retention_val is not None else None),
        "v_abstain_not_worse": bool(ab["worst_group_lcb"] >= dp["worst_group_lcb"] - 1e-12),
        "vi_dp_beats_continuous_d": bool(dp["mean_regret"] < dgb["mean_regret"] - eps),
        "vi_perm_p_ok": (bool(perm_p < 0.05) if perm_p is not None else None),
        "trend_retention": trend_retention_val, "perm_p": perm_p,
    }


# ══════════════════════════════════════════════════════════════════════════
# A-39 统计工具：paired bootstrap CI / trend 保留率 / SNR 分层置换检验
# ══════════════════════════════════════════════════════════════════════════
def paired_bootstrap_ci(vals_a: np.ndarray, vals_b: np.ndarray, n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """per-uid paired 差值 (a−b) 的 uid bootstrap CI。caveat（A-39⑤预注册）：条件于已拟合
    头/router，未含重拟合方差——confirmatory 如需升级 grouped full-refit（A-33c）。"""
    d = np.asarray(vals_a, float) - np.asarray(vals_b, float)
    rng = np.random.default_rng(seed + 424243)
    boots = np.array([float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(mean=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi),
                frac_positive=float(np.mean(boots > 0)), n=int(len(d)), n_boot=int(n_boot))


def subgroup_delta_vs_incumbent(data: PolicyData, loss_pick: np.ndarray,
                                origin_name: str) -> float:
    """某 origin 子群上 Δ=mean(L[:,incumbent] − loss_pick)（>0=优于 incumbent）。"""
    m = data.origin == origin_name
    if not m.any():
        return float("nan")
    return float((data.L[m, data.fallback_idx] - loss_pick[m]).mean())


def trend_retention(data: PolicyData, loss_pick_candidate: np.ndarray,
                    loss_pick_dlookup: np.ndarray, origin_name: str = "S_trend") -> dict:
    """判据 (iv)：候选臂在 trend 子群保留 D-only 增益的比例（安全回退不许吃光收益）。
    Δ_trend(D-only) ≤ 0 时判据不适用（无收益可保留）→ ratio=None。"""
    d_cand = subgroup_delta_vs_incumbent(data, loss_pick_candidate, origin_name)
    d_dl = subgroup_delta_vs_incumbent(data, loss_pick_dlookup, origin_name)
    ratio = (d_cand / d_dl) if (np.isfinite(d_dl) and d_dl > 1e-9) else None
    return dict(delta_candidate=d_cand, delta_dlookup=d_dl,
                retention=float(ratio) if ratio is not None else None)


def snr_strata(snr: np.ndarray, cell: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """置换层 = cell × cell 内连续 SNR 分位 bin（A-39②，n_bins=3 预注册）。"""
    strata = np.empty(len(snr), dtype=object)
    for c in np.unique(cell):
        m = cell == c
        qs = np.quantile(snr[m], np.linspace(0, 1, n_bins + 1)[1:-1]) if m.sum() >= n_bins else []
        strata[m] = [f"{c}#b{int(np.digitize(v, qs))}" for v in snr[m]]
    return strata


def residualized_perm_test(stat_fn: Callable[[np.ndarray], float], X_p: np.ndarray,
                           strata: np.ndarray, n_perm: int = 99, seed: int = 0,
                           progress: int = 0, done_nulls: Optional[List[float]] = None) -> dict:
    """判据 (vi) 置换检验：T=stat_fn(X_p)（如 regret(d_gbdt)−regret(dp_gbdt)，正值=P 有增量）；
    null=层内置换 X_p 行（D/标签/折固定）。**per-perm 独立种子** → 可断点续（done_nulls 传入已算
    null 时跳过重算，A-36 精神）。p=(1+#{T_b≥T})/(n_perm+1)。"""
    T_obs = float(stat_fn(X_p))
    nulls: List[float] = list(done_nulls or [])
    idx = np.arange(len(X_p))
    for b in range(len(nulls), n_perm):
        rng = np.random.default_rng(seed + 5555 + 131 * (b + 1))    # 只依赖 b → resume bit 级一致
        perm = idx.copy()
        for s in np.unique(strata):
            sm = np.where(strata == s)[0]
            perm[sm] = sm[rng.permutation(len(sm))]
        nulls.append(float(stat_fn(X_p[perm])))
        if progress and (b + 1) % progress == 0:
            print(f"        [perm] {b+1}/{n_perm}", flush=True)
    nulls_arr = np.array(nulls[:n_perm])
    p = float((1 + np.sum(nulls_arr >= T_obs)) / (n_perm + 1))
    return dict(T_obs=T_obs, p=p, n_perm=int(n_perm), null_mean=float(nulls_arr.mean()),
                nulls=[float(x) for x in nulls_arr])
