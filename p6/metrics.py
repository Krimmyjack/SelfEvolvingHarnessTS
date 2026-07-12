"""p6/metrics.py — P6 冻结效应度量词汇表 + failure signature 计算器（prereg §0 / §4）。

**本模块是全项目唯一合法效应词汇表**（prereg §0：「统一效应口径……runner 须经
`p6/metrics.py` 的冻结函数产出」）：任何 runner / 文档 / JSON 产出 gain、regret、harm、
batch_delta 换算，一律经本模块函数，**禁止调用方绕过本模块手写符号转换**。

词汇表（prereg §0，冻结）：
  loss                 判官 RMSE（越小越好）；
  gain(H→e)            = loss_H − loss_e（正 = 改善）→ `gain(loss_baseline, loss_edit)`；
  batch_delta          判官模块口径 = loss_new − loss_old = −gain
                       → `gain_from_batch_delta(batch_delta) = −batch_delta`；
  regret               = loss_chosen − min(loss_pool)（≥0，负值 = 调用方口径错 → raise）；
  harm                 = −gain。

failure signature（prereg §4，判官尺度、D 块，冻结定义）：
  行为等价类   两候选在一 episode 上等价 ⟺ |loss 差| ≤ tol（默认 1e-9；union-find 成类，
              exact-tie 按 ≤ 处理）→ `effect_classes`；
  S1          batch 均值 regret ≥ ε 且 series 聚类 CI90 下界 > 0 → `s1_selector`；
  S2          episode 级等价类数的全 episode 均值 < 2.0，或 池上限 gain 低于 det 阶梯
              同口径 − ε（ceiling_gap = pool − det < −ε）→ `s2_supply`；
  S3          冻结 cohort 清单内某 cohort（≥ min_series 底层 series）train_harm 的
              LCB90 > δ_safe → `s3_scope_harm`；
  激活         归一化 headroom = (observed − threshold)/threshold，取 headroom 最大的
              fired 族；并列按 S1 > S3 > S2；全部不过线 → None = abstain → `activate`。

聚类 bootstrap（prereg §4，冻结）：按 cluster（底层 series_uid，"4 preset 同抽"）重采样、
均值分布的 **5% 分位（quantile linear 插值）** = LCB90（双侧 90% CI 下端点）；
PRNG = np.random.default_rng(seed)（seed 由调用方显式传入，正式协议 = 20260711+cycle）；
冻结抽取契约 = 单次调用 rng.integers(0, n_clusters, size=(b, n_clusters))。

红线：全部纯函数；只依赖 numpy + stdlib；无全局/隐式 RNG（bootstrap 只用显式 seed 的
default_rng）；不读文件、不落盘；非有限输入一律 raise ValueError（technical-abort 风格，
不静默传播 NaN）。
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

__all__ = [
    "ACTIVATION_PRIORITY",
    "DEFAULT_BOOTSTRAP_B",
    "EFFECT_CLASS_TOL",
    "S2_MEAN_CLASSES_THRESHOLD",
    "activate",
    "cluster_bootstrap_means",
    "cluster_lcb90",
    "effect_classes",
    "gain",
    "gain_from_batch_delta",
    "harm",
    "normalized_headroom",
    "regret",
    "s1_selector",
    "s2_supply",
    "s3_scope_harm",
]

EFFECT_CLASS_TOL: float = 1e-9              # prereg §4：行为等价类 loss 容差
S2_MEAN_CLASSES_THRESHOLD: float = 2.0      # prereg §4：S2 等价类数均值阈（< 触发，严格）
ACTIVATION_PRIORITY: Tuple[str, ...] = ("S1", "S3", "S2")   # prereg §4：并列优先序
DEFAULT_BOOTSTRAP_B: int = 2000             # prereg §4：B=2000
_REGRET_FP_TOL: float = 1e-12               # regret 负值的浮点容差（超出 → 调用方口径错）


def _as_finite_float(x: Any, name: str) -> float:
    """标量 → float，非有限（NaN/±inf）→ raise ValueError（冻结度量不静默传播坏值）。"""
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限数，得到 {x!r}")
    return v


# ════════════════════════════ 词汇表（prereg §0） ════════════════════════════
def gain(loss_baseline: float, loss_edit: float) -> float:
    """gain(H→e) = loss_H − loss_e（prereg §0；正 = 改善）。

    loss_baseline = 现任 H 的判官 loss；loss_edit = 被评 edit 的判官 loss。
    调用方不得手写该减法——本函数是唯一合法出口。"""
    return _as_finite_float(loss_baseline, "loss_baseline") - _as_finite_float(
        loss_edit, "loss_edit"
    )


def gain_from_batch_delta(batch_delta: float) -> float:
    """gain = −batch_delta（prereg §0：判官模块 batch_delta = loss_new − loss_old = −gain）。

    判官（judge_closed_form 等）产出的 raw batch_delta 一律经本函数换成 gain 口径，
    文档/JSON 不再混用 raw delta；调用方禁止手写取负。"""
    return -_as_finite_float(batch_delta, "batch_delta")


def harm(gain_value: float) -> float:
    """harm = −gain（prereg §0）。S3 的 train_harm = harm(train_gain)；调用方禁止手写取负。"""
    return -_as_finite_float(gain_value, "gain_value")


def regret(loss_chosen: float, loss_pool_min: float) -> float:
    """regret = loss_chosen − min(loss_pool) ≥ 0（prereg §0）。

    chosen ∈ pool ⇒ regret 按构造非负；loss_chosen < loss_pool_min − 1e-12 说明
    调用方传的不是池内最小 loss（口径错）→ raise ValueError。浮点毛刺（≤1e-12）截为 0。"""
    lc = _as_finite_float(loss_chosen, "loss_chosen")
    lp = _as_finite_float(loss_pool_min, "loss_pool_min")
    d = lc - lp
    if d < -_REGRET_FP_TOL:
        raise ValueError(
            f"regret 为负（loss_chosen={lc} < loss_pool_min={lp}）——loss_pool_min 必须是"
            f"池内 per-episode 最小 loss（prereg §0），疑似调用方口径错误"
        )
    return max(0.0, d)


# ════════════════════════════ 行为等价类（prereg §4） ════════════════════════════
def effect_classes(
    losses: Mapping[str, float], tol: float = EFFECT_CLASS_TOL
) -> List[Set[str]]:
    """行为等价类：|loss 差| ≤ tol 的候选（candidate_sha）union-find 成类（prereg §4）。

    语义 = 完整 ≤tol 图的连通分量（union-find）：链式并类（a~b、b~c ⇒ a,b,c 同类，即使
    |loss_a − loss_c| > tol）；exact-tie 按 ≤ 处理（差恰为 tol → 同类）。
    实现：按 loss 排序后只并相邻对——排序下任意 ≤tol 边所跨的相邻差必各 ≤tol，
    连通分量与全图一致。返回类列表按类内最小 loss 升序（确定性）。"""
    t = _as_finite_float(tol, "tol")
    if t < 0.0:
        raise ValueError(f"tol 必须 ≥ 0，得到 {tol!r}")
    items: List[Tuple[float, str]] = []
    for sha, loss in losses.items():
        items.append((_as_finite_float(loss, f"losses[{sha!r}]"), str(sha)))
    if not items:
        return []
    items.sort()                                   # (loss, sha) —— 确定性
    classes: List[Set[str]] = [{items[0][1]}]
    prev_loss = items[0][0]
    for loss, sha in items[1:]:
        if loss - prev_loss <= t:                  # 相邻差 ≤ tol → 并入当前类（链式传递）
            classes[-1].add(sha)
        else:
            classes.append({sha})
        prev_loss = loss
    return classes


# ════════════════════════════ 聚类 bootstrap（prereg §4） ════════════════════════════
def cluster_bootstrap_means(
    values: Sequence[float],
    cluster_ids: Sequence[Any],
    b: int = DEFAULT_BOOTSTRAP_B,
    *,
    seed: int,
) -> np.ndarray:
    """聚类 bootstrap 的均值分布（shape=(b,)）。cluster_lcb90 的承重内核，单独暴露供披露/测试。

    冻结契约（prereg §4）：
      - 重采样单位 = cluster（同 cluster 的全部 values 一起进 replicate——"4 preset 同抽"）；
      - cluster 索引 = 按 cluster_ids 首次出现次序（0..n_clusters−1，确定性）；
      - 抽取 = **单次调用** rng.integers(0, n_clusters, size=(b, n_clusters))，
        PRNG = np.random.default_rng(seed)（seed 显式传入）；
      - replicate 统计量 = 所抽全部 episode 值的等权均值（Σ cluster 和 / Σ cluster 计数）。"""
    vals = np.asarray(list(values), dtype=float)
    ids = list(cluster_ids)
    if vals.size == 0:
        raise ValueError("values 不能为空")
    if len(ids) != vals.size:
        raise ValueError(f"values（{vals.size}）与 cluster_ids（{len(ids)}）长度不一致")
    if not np.all(np.isfinite(vals)):
        raise ValueError("values 含非有限值（NaN/inf）——technical abort 口径，不静默")
    if not isinstance(b, int) or isinstance(b, bool) or b < 1:
        raise ValueError(f"b 必须是 ≥1 的 int，得到 {b!r}")

    index: Dict[Any, int] = {}
    for cid in ids:
        if cid not in index:
            index[cid] = len(index)               # 首次出现次序（确定性）
    n_clusters = len(index)
    sums = np.zeros(n_clusters, dtype=float)
    counts = np.zeros(n_clusters, dtype=float)
    for v, cid in zip(vals, ids):
        j = index[cid]
        sums[j] += v
        counts[j] += 1.0

    rng = np.random.default_rng(seed)
    draw = rng.integers(0, n_clusters, size=(b, n_clusters))     # 冻结抽取契约
    rep_sums = sums[draw].sum(axis=1)
    rep_counts = counts[draw].sum(axis=1)
    return rep_sums / rep_counts                   # counts ≥ 1 逐 cluster ⇒ 分母恒 > 0


def cluster_lcb90(
    values: Sequence[float],
    cluster_ids: Sequence[Any],
    b: int = DEFAULT_BOOTSTRAP_B,
    *,
    seed: int,
) -> float:
    """按 cluster 重采样的均值分布的 **5% 分位（quantile linear 插值）** = LCB90
    （双侧 90% CI 下端点；prereg §4 冻结）。确定性：同 (values, cluster_ids, b, seed)
    → 同输出。抽取契约见 cluster_bootstrap_means。"""
    means = cluster_bootstrap_means(values, cluster_ids, b, seed=seed)
    return float(np.quantile(means, 0.05, method="linear"))


def _cohort_seed(seed: int, cohort_id: Any) -> int:
    """cohort 级派生 seed：sha256(f"{seed}|{cohort_id}") 前 8 字节 → int。
    确定性且 cohort-稳定（增删其他 cohort 不改变本 cohort 的抽取流；不用内建 hash——有盐）。"""
    digest = hashlib.sha256(f"{int(seed)}|{cohort_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


# ════════════════════════════ S1 / S2 / S3（prereg §4） ════════════════════════════
def s1_selector(
    per_episode: Sequence[Mapping[str, float]],
    clusters: Sequence[Any],
    eps: float,
    b: int = DEFAULT_BOOTSTRAP_B,
    *,
    seed: int,
) -> Dict[str, Any]:
    """S1（selector 失败 signature，prereg §4）：
    fired = regret_mean ≥ eps **且** 聚类 bootstrap LCB90 > 0。

    per_episode：[{loss_chosen, loss_pool_min}]（判官尺度）；clusters：逐 episode 的
    series 聚类 id（同底层 series 的 4 preset 同 id）。regret 经本模块 regret() 产出
    （负 regret 即 raise，见其 docstring）。返回 {"regret_mean","lcb90","fired"}。"""
    if len(per_episode) == 0:
        raise ValueError("per_episode 不能为空")
    if len(per_episode) != len(clusters):
        raise ValueError(
            f"per_episode（{len(per_episode)}）与 clusters（{len(clusters)}）长度不一致"
        )
    e = _as_finite_float(eps, "eps")
    regrets = [regret(ep["loss_chosen"], ep["loss_pool_min"]) for ep in per_episode]
    regret_mean = float(np.mean(regrets))
    lcb = cluster_lcb90(regrets, clusters, b, seed=seed)
    return {
        "regret_mean": regret_mean,
        "lcb90": lcb,
        "fired": bool(regret_mean >= e and lcb > 0.0),
    }


def s2_supply(
    per_episode_classes_count: Sequence[int],
    pool_ceiling_gain: float,
    det_ceiling_gain: float,
    eps: float,
) -> Dict[str, Any]:
    """S2（supply 失败 signature，prereg §4）：
    fired = mean_classes < 2.0（严格 <）**或** ceiling_gap < −eps（严格 <）。

    per_episode_classes_count：逐 episode 的行为等价类数（effect_classes 的 len，≥1）；
    pool_ceiling_gain：池上限 gain（池内 per-episode min-loss 构成的 batch 的 gain 口径）；
    det_ceiling_gain：det 阶梯同口径；ceiling_gap = pool − det。
    返回 {"mean_classes","ceiling_gap","fired"}。"""
    if len(per_episode_classes_count) == 0:
        raise ValueError("per_episode_classes_count 不能为空")
    counts = []
    for i, c in enumerate(per_episode_classes_count):
        v = _as_finite_float(c, f"per_episode_classes_count[{i}]")
        if v < 1.0:
            raise ValueError(f"等价类数必须 ≥ 1（episode {i} 得到 {c!r}）")
        counts.append(v)
    e = _as_finite_float(eps, "eps")
    mean_classes = float(np.mean(counts))
    # 注意：ceiling_gap 是两个同口径 gain 的差（gap），不是 loss→gain 符号换算，
    # 不经词汇表函数（词汇表禁的是绕过符号转换，不是普通差分）。
    ceiling_gap = _as_finite_float(pool_ceiling_gain, "pool_ceiling_gain") - _as_finite_float(
        det_ceiling_gain, "det_ceiling_gain"
    )
    return {
        "mean_classes": mean_classes,
        "ceiling_gap": ceiling_gap,
        "fired": bool(mean_classes < S2_MEAN_CLASSES_THRESHOLD or ceiling_gap < -e),
    }


def s3_scope_harm(
    cohort_gains: Mapping[Any, Sequence[float]],
    delta_safe: float,
    min_series: int = 5,
    b: int = DEFAULT_BOOTSTRAP_B,
    *,
    seed: int,
) -> Dict[str, Any]:
    """S3（scope-harm 失败 signature，prereg §4 / §3.3 冻结 cohort 清单上的 worst-case）：
    fired = 存在 cohort（≥ min_series 底层 series）使 train_harm 的 LCB90 > delta_safe（严格 >）。

    cohort_gains：{cohort_id: [per-series train_gain]}（判官尺度；gain 口径）；
    harm = harm(gain)（本模块词汇表）；每 series 视为一个 cluster（series 级 bootstrap）。
    < min_series 的 cohort **不评估**（清单内小 cohort 不承重）。cohort 级 seed 由
    sha256(f"{seed}|{cohort_id}") 派生（cohort-稳定，见 _cohort_seed）。
    返回 {"worst_cohort","harm_lcb90","fired","per_cohort"}；无合格 cohort →
    worst_cohort=None、harm_lcb90=None、fired=False。worst 并列取排序（str）靠前者。"""
    d = _as_finite_float(delta_safe, "delta_safe")
    if not isinstance(min_series, int) or isinstance(min_series, bool) or min_series < 1:
        raise ValueError(f"min_series 必须是 ≥1 的 int，得到 {min_series!r}")
    per_cohort: Dict[Any, Dict[str, Any]] = {}
    worst_id: Optional[Any] = None
    worst_lcb: Optional[float] = None
    for cid in sorted(cohort_gains, key=lambda c: str(c)):       # 确定性遍历
        gains_c = list(cohort_gains[cid])
        if len(gains_c) < min_series:
            continue
        harms = [harm(g) for g in gains_c]                        # 词汇表出口
        lcb = cluster_lcb90(
            harms, list(range(len(harms))), b, seed=_cohort_seed(seed, cid)
        )
        per_cohort[cid] = {"n_series": len(gains_c), "harm_lcb90": lcb}
        if worst_lcb is None or lcb > worst_lcb:                  # 严格 > ⇒ 并列取靠前
            worst_id, worst_lcb = cid, lcb
    return {
        "worst_cohort": worst_id,
        "harm_lcb90": worst_lcb,
        "fired": bool(worst_lcb is not None and worst_lcb > d),
        "per_cohort": per_cohort,
    }


# ════════════════════════════ headroom 与激活（prereg §4） ════════════════════════════
def normalized_headroom(observed: float, threshold: float) -> float:
    """归一化 headroom = (observed − threshold)/threshold（prereg §4 原式）。

    threshold == 0 → raise（除零无定义）。方向义务在调用方：对"低于阈值触发"的量，
    negative-threshold 形式或镜像变换由调用方按其 signature 语义给出——本函数只冻结公式。"""
    o = _as_finite_float(observed, "observed")
    t = _as_finite_float(threshold, "threshold")
    if t == 0.0:
        raise ValueError("threshold 不能为 0（归一化除零）")
    return (o - t) / t


def activate(signatures: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    """激活裁决（prereg §4）：在 fired 的 signature 族中取 **headroom 最大**者；
    并列（headroom 精确相等）按优先序 S1 > S3 > S2；无一 fired → None（= abstain）。

    signatures：{"S1"|"S2"|"S3": {"fired": bool, "headroom": float, ...}}（可只给子集；
    未给的族视为未触发）。fired 的族必须带有限 headroom，否则 raise。未知族名 → raise。"""
    unknown = sorted(set(signatures) - set(ACTIVATION_PRIORITY))
    if unknown:
        raise ValueError(f"未知 signature 族 {unknown}（可用：{sorted(ACTIVATION_PRIORITY)}）")
    best: Optional[Tuple[str, float]] = None
    for fam in ACTIVATION_PRIORITY:                # 优先序遍历 + 严格 > ⇒ 并列归先序
        sig = signatures.get(fam)
        if sig is None or not bool(sig.get("fired", False)):
            continue
        if "headroom" not in sig:
            raise ValueError(f"{fam} fired 但缺 headroom")
        h = _as_finite_float(sig["headroom"], f"{fam}.headroom")
        if best is None or h > best[1]:
            best = (fam, h)
    return best[0] if best else None
