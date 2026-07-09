"""s2_corpus.py — Stage-2 结构库 v3 语料生成器（协议 v3 的参数**单一真源**）。

八族 = 4 冻结族（S_season/S_trend/S_both/S_ar，**逐字复用** run_variance_decomp._clean_signal）
     + 4 新族（本模块定义，生成方程+小参数网格如下，variant = sd % len(grid) 确定性选取）：
  S_intermittent  零膨胀间歇需求：Bernoulli(p_event) 触发 burst（长度 ≤ burst_len），幅度
                  0.5+|N(0,1)|；**只除 std 不去均值**（零膨胀是结构本体，demean 会破坏）。
  S_hetero        GARCH(1,1) 波动聚簇：σ²_t = ω + α x²_{t-1} + β σ²_{t-1}（ω=1−α−β → 无条件方差 1）。
  S_regime        分段均值/斜率（K 段，断点 ∈ [64, L−64]），加 0.15 iid 纹理（防常数段 SNR 退化）。
  S_multiseason   sin(2πt/16) + a2·sin(2πt/128 + φ)——直击 D1 混叠/多周期词汇。周期对 (16,128)
                  的三重约束（L=512 实测定）：①both 整 rfft bin（无泄漏）；②公度比 8 → 短周期
                  分量在 lag=128 处 cos=+1，robust_v1 的单 lag ACF 确认不被拉低（**非公度对如
                  (24,128)：24 分量在 lag 128 贡献 cos(120°)=−0.5 → 真峰 ACF 0.18<0.2 被拒——
                  这是估计器真实局限，记为 P1b 改进标的**）；③128∉{16m: m≤4} 不被谐波去重误杀。

退化网格 S2_DEG_GRID（12 dname）：noise ∈ {0.12, 0.55} × miss-topology 六态
  {none, random 3%, random 12%, block 3%, block 12%, burst 6%}——**miss-topology 是第一类
  变异轴**（prereg_s2_replication.md §2：dev 旧语料 6% 均匀随机 → gap 特征不可评估的教训）。
  噪声/离群流 = default_rng(sd+10_000)（与 v1 网格同构）；缺失流独立 = default_rng(sd+20_000)。

namespace（协议 v2/§namespace 冻结方案）：uid = "S2:{family}:{dname}:{j}"，
sd = sha256(uid)[:4 LE] % 2_000_000。本波 dev_j = 0..9（j 10..19 保留给后续波，仍在 [0,19] 内）。
split：每 (family,dname) 层内 rng(_det_seed("S2split",family,dname)) 置换取前 70% → dev；
holdout uid **只记录不物化**（生成确定性 → 解锁时可再生；首读须 holdout_access_log.jsonl）。

运行（生成 dev + audit）：PYTHONIOENCODING=utf-8 PYTHONPATH=<Agent> \
  D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.s2_corpus
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .data.synthetic_gen import RawSeries, LENGTH, H_FORECAST
from .run_variance_decomp import CUT, OUT_RATE, _clean_signal, _unit

OUT_DIR = Path(__file__).resolve().parent / "results" / "Stage2"
AUDIT_PATH = OUT_DIR / "s2_corpus_audit.json"

FROZEN_V1_FAMILIES = ("S_season", "S_trend", "S_both", "S_ar")
NEW_FAMILIES = ("S_intermittent", "S_hetero", "S_regime", "S_multiseason")
S2_FAMILIES = FROZEN_V1_FAMILIES + NEW_FAMILIES

S2_FAMILY_GRID: Dict[str, List[dict]] = {          # variant = sd % len(grid)（确定性）
    "S_intermittent": [dict(p_event=0.05, burst_len=1), dict(p_event=0.15, burst_len=3)],
    "S_hetero":       [dict(alpha=0.15, beta=0.80), dict(alpha=0.30, beta=0.65)],
    "S_regime":       [dict(n_seg=3, kind="mean"), dict(n_seg=4, kind="slope")],
    "S_multiseason":  [dict(p1=16, p2=128, a2=0.6), dict(p1=16, p2=128, a2=1.0)],
}

S2_DEG_GRID = OrderedDict([                        # noise × miss-topology（第一类轴）
    ("n_hi_full",     dict(noise=0.12, miss=0.00, topo="none")),
    ("n_hi_rand_lo",  dict(noise=0.12, miss=0.03, topo="random")),
    ("n_hi_rand_hi",  dict(noise=0.12, miss=0.12, topo="random")),
    ("n_hi_block_lo", dict(noise=0.12, miss=0.03, topo="block")),
    ("n_hi_block_hi", dict(noise=0.12, miss=0.12, topo="block")),
    ("n_hi_burst",    dict(noise=0.12, miss=0.06, topo="burst")),
    ("n_lo_full",     dict(noise=0.55, miss=0.00, topo="none")),
    ("n_lo_rand_lo",  dict(noise=0.55, miss=0.03, topo="random")),
    ("n_lo_rand_hi",  dict(noise=0.55, miss=0.12, topo="random")),
    ("n_lo_block_lo", dict(noise=0.55, miss=0.03, topo="block")),
    ("n_lo_block_hi", dict(noise=0.55, miss=0.12, topo="block")),
    ("n_lo_burst",    dict(noise=0.55, miss=0.06, topo="burst")),
])

DEV_J = tuple(range(10))                           # 本波 j=0..9；10..19 保留
DEV_FRAC = 0.7
BURST_MEAN_LEN = 6                                 # burst 段平均长度（miss 拓扑）


# ════════════════════════════ 确定性种子 ════════════════════════════
def uid_of(family: str, dname: str, j: int) -> str:
    return f"S2:{family}:{dname}:{j}"


def sd_of(uid: str) -> int:
    """协议 v2 冻结方案：sd = sha256(uid) % 2_000_000（前 4 字节 little-endian）。"""
    return int.from_bytes(hashlib.sha256(uid.encode("utf-8")).digest()[:4], "little") % 2_000_000


def _det_seed(*parts) -> int:
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


# ════════════════════════════ 新族生成方程 ════════════════════════════
def s2_clean(family: str, sd: int) -> np.ndarray:
    if family in FROZEN_V1_FAMILIES:
        return _clean_signal(family, sd)           # 冻结族逐字复用（bit 级同 v1）
    rng = np.random.default_rng(sd)
    pp = S2_FAMILY_GRID[family][sd % len(S2_FAMILY_GRID[family])]
    t = np.arange(LENGTH, dtype=float)

    if family == "S_intermittent":
        x = np.zeros(LENGTH)
        i = 0
        while i < LENGTH:
            if rng.random() < pp["p_event"]:
                run = int(rng.integers(1, pp["burst_len"] + 1))
                mag = 0.5 + abs(rng.normal(0.0, 1.0))
                for k in range(min(run, LENGTH - i)):
                    x[i + k] = mag * (0.8 + 0.4 * rng.random())
                i += run
            else:
                i += 1
        s = float(np.std(x))
        return x / (s if s > 1e-9 else 1.0)        # 不去均值：零膨胀是结构本体

    if family == "S_hetero":
        a, b = pp["alpha"], pp["beta"]
        om = 1.0 - a - b
        z = rng.normal(0.0, 1.0, LENGTH)
        x = np.empty(LENGTH)
        sig2 = 1.0
        x[0] = z[0]
        for i in range(1, LENGTH):
            sig2 = om + a * x[i - 1] ** 2 + b * sig2
            x[i] = math.sqrt(sig2) * z[i]
        return _unit(x)

    if family == "S_regime":
        k = pp["n_seg"]
        bps = np.sort(rng.choice(np.arange(64, LENGTH - 64), size=k - 1, replace=False))
        x = np.empty(LENGTH)
        for seg in np.split(np.arange(LENGTH), bps):
            if pp["kind"] == "mean":
                x[seg] = rng.uniform(-1.5, 1.5)
            else:                                   # slope：段内线性 + 段基线
                x[seg] = rng.uniform(-0.5, 0.5) + rng.uniform(-0.05, 0.05) * np.arange(seg.size)
        x = x + 0.15 * rng.normal(0.0, 1.0, LENGTH)  # 纹理（防常数段 SNR 退化）
        return _unit(x)

    if family == "S_multiseason":
        x = np.sin(2 * np.pi * t / pp["p1"]) + pp["a2"] * np.sin(
            2 * np.pi * t / pp["p2"] + rng.uniform(0.0, 2.0 * np.pi))
        return _unit(x)
    raise ValueError(family)


# ════════════════════════════ 退化（miss-topology 第一类轴）════════════════════════════
def _miss_indices(n: int, n_miss: int, topo: str, rng: np.random.Generator) -> np.ndarray:
    if n_miss <= 0:
        return np.empty(0, dtype=int)
    if topo == "random":
        return rng.choice(n, size=n_miss, replace=False)
    if topo == "block":                             # 单连续块
        start = int(rng.integers(0, n - n_miss + 1))
        return np.arange(start, start + n_miss)
    if topo == "burst":                             # 多短簇（平均长 BURST_MEAN_LEN，不重叠 best-effort）
        idx: List[int] = []
        guard = 0
        while len(idx) < n_miss and guard < 200:
            guard += 1
            run = int(np.clip(rng.geometric(1.0 / BURST_MEAN_LEN), 2, 3 * BURST_MEAN_LEN))
            run = min(run, n_miss - len(idx))
            start = int(rng.integers(0, n - run + 1))
            seg = range(start, start + run)
            if not any(s in idx for s in seg):
                idx.extend(seg)
        return np.array(sorted(idx[:n_miss]), dtype=int)
    raise ValueError(topo)


def s2_degrade(clean: np.ndarray, noise: float, miss: float, topo: str, sd: int) -> np.ndarray:
    """噪声+离群流 = rng(sd+10_000)（与 v1 网格同构次序）；缺失流独立 = rng(sd+20_000)。"""
    rng = np.random.default_rng(sd + 10_000)
    x = clean.astype(float).copy()
    n = x.size
    if noise > 0:
        x = x + rng.normal(0, noise, n)
    n_out = int(round(OUT_RATE * n))
    if n_out > 0:
        idx = rng.choice(n, size=n_out, replace=False)
        x[idx] += rng.choice([-1.0, 1.0], size=n_out) * 5.0
    rng_m = np.random.default_rng(sd + 20_000)
    mi = _miss_indices(n, int(round(miss * n)), topo, rng_m)
    if mi.size:
        x[mi] = np.nan
    return x


def make_series(family: str, dname: str, j: int) -> RawSeries:
    uid = uid_of(family, dname, j)
    sd = sd_of(uid)
    dp = S2_DEG_GRID[dname]
    clean = s2_clean(family, sd)
    degraded = s2_degrade(clean, dp["noise"], dp["miss"], dp["topo"], sd)
    return RawSeries(pattern=family, task="forecast", seed=sd, period=24,
                     obs_scale=float(np.std(clean[CUT:])) or 1.0,
                     clean=clean, degraded=degraded,
                     history=degraded[:CUT].copy(), clean_history=clean[:CUT].copy(),
                     future=clean[CUT:].copy(), origin=family, series_uid=uid)


# ════════════════════════════ split（70/30，holdout 不物化）════════════════════════════
def s2_split() -> Tuple[List[Tuple[str, str, int]], List[str]]:
    """→ (dev 三元组列表, holdout uid 列表)。层内确定性置换取前 70%。"""
    dev: List[Tuple[str, str, int]] = []
    hold: List[str] = []
    for family in S2_FAMILIES:
        for dname in S2_DEG_GRID:
            js = list(DEV_J)
            rng = np.random.default_rng(_det_seed("S2split", family, dname))
            perm = [js[i] for i in rng.permutation(len(js))]
            n_dev = int(math.ceil(DEV_FRAC * len(js)))
            for j in perm[:n_dev]:
                dev.append((family, dname, j))
            for j in perm[n_dev:]:
                hold.append(uid_of(family, dname, j))
    return dev, hold


def build_s2_dev() -> List[RawSeries]:
    """物化 dev 语料（holdout uid 触碰即违规——本函数根本不生成它们）。"""
    dev, _ = s2_split()
    return [make_series(f, d, j) for f, d, j in sorted(dev)]


# ════════════════════════════ audit ════════════════════════════
def main():
    from .run_variance_decomp import assign_cells
    dev, hold = s2_split()
    corpus = build_s2_dev()
    cells, snr_of = assign_cells(corpus)
    by_cell = {cid: {} for cid in cells}
    for cid, series in cells.items():
        for rs in series:
            by_cell[cid][rs.origin] = by_cell[cid].get(rs.origin, 0) + 1
    audit = dict(
        date="2026-07-05", n_dev=len(corpus), n_holdout_reserved=len(hold),
        families=list(S2_FAMILIES), deg_grid=list(S2_DEG_GRID),
        dev_j=list(DEV_J), dev_frac=DEV_FRAC,
        family_grid={k: v for k, v in S2_FAMILY_GRID.items()},
        cells={cid: dict(n=len(series), origins=by_cell[cid]) for cid, series in sorted(cells.items())},
        snr_range=[float(min(snr_of.values())), float(max(snr_of.values()))],
        holdout_note="holdout 未物化；uid 列表可由 s2_split() 确定性再生；首读须 holdout_access_log.jsonl")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audit, ensure_ascii=False, indent=1), "utf-8")
    print(f"S2 dev 语料：{len(corpus)} uid（holdout 保留 {len(hold)}，未物化）", flush=True)
    for cid in sorted(cells):
        comp = " ".join(f"{k}={v}" for k, v in sorted(by_cell[cid].items()))
        print(f"  {cid:26s} n={len(cells[cid]):4d}  {comp}", flush=True)
    print(f"audit → {AUDIT_PATH}", flush=True)


if __name__ == "__main__":
    main()
