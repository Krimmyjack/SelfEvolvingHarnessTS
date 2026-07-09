"""slow_path/forward_transfer.py — ★v4 S2：前向迁移曲线分析（读 S1 的 forward_transfer JSONL）。

承 S1_Implementation_Plan §B.5 + Refactor_Continual_TaskReadiness_v4 S2（P0）。把 S1 落的
per-(domain k, cell, mode) JSONL 聚合成 per-(mode, k) 曲线点，产出 headline 实验 S2：

  • time_to_readiness(k)：纵轴 round（median/max over cells），mode=updating(C, memory-on)
    vs scratch(A, memory-off) 两条曲线 —— "能力随 domain 增强"的**可证伪**载体。
  • readiness@budget(k)：跑满预算后的 readiness（median over cells），同样 C vs A。
  • 三 bootstrap 分解 B−A(记忆价值) / C−B(继续更新价值)（readiness@budget 代理；
    下游 ΔPerf 双报告留 S4 的 report_target，不在 S2）。
  • 负迁移护栏：per-domain n_reval_demote（warm-start 重验降级次数）。
  • headline 判据（诚实预期）：C 在后到 domain 上 **不慢于** A（ttr ≤ / readiness ≥）**且不退化**
    → 前向迁移成立（非要求单调，承诺「≥ from-scratch 且不退化」，见 Experiment_Design_Final S2）。

纯分析、无绘图依赖（plot 在 runner 里可选）。JSONL 含 NaN（readiness 分母退化时）——json 默认
allow_nan 可解析为 float('nan')，聚合时按 finite 过滤。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..evaluators import aggregate_time_to_readiness

SCRATCH, FROZEN, UPDATING = "scratch", "frozen", "updating"
MEMORY_ON, MEMORY_OFF = UPDATING, SCRATCH       # C=memory-on, A=memory-off


# ════════════════════════════ 读日志 ════════════════════════════
def load_transfer_log(path: str) -> List[dict]:
    """读 forward_transfer_{mode}.jsonl（每行一 cell record）。空行跳过；NaN 由 json 默认解析。"""
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _finite(xs):
    return [x for x in xs if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]


# ════════════════════════════ per-(mode,k) 聚合 ════════════════════════════
@dataclass
class DomainPoint:
    """一个 (mode, domain k) 的曲线点：cells 聚合后的就绪度量。"""
    k: int
    domain: str
    mode: str
    n_cells: int
    ttr_median: Optional[float]          # round；over cells 取 median（病态 cell 鲁棒）
    ttr_max: Optional[int]               # worst-case；任一 cell 未达 → None
    n_ready: int                         # 达标 cell 数
    readiness_median: Optional[float]    # readiness@budget 的 median（只取 finite）
    ready_frac: float                    # ready=True 的 cell 占比
    n_reval_demote: int                  # 该 domain warm-start 重验降级数（负迁移护栏）
    harness_version: int

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def per_domain_points(rows: List[dict]) -> List[DomainPoint]:
    """同一 mode 的 cell records → per-domain 曲线点（按 k 升序）。"""
    by_k: Dict[int, List[dict]] = {}
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(r)

    points: List[DomainPoint] = []
    for k in sorted(by_k):
        cell_rows = by_k[k]
        ttr = [r.get("time_to_readiness_rounds") for r in cell_rows]
        agg = aggregate_time_to_readiness(ttr)
        rd = _finite([r.get("readiness_at_budget") for r in cell_rows])
        ready_flags = [bool(r.get("ready")) for r in cell_rows]
        demote = max((int(r.get("n_reval_demote_domain", 0) or 0) for r in cell_rows), default=0)
        ver = max((int(r.get("harness_version", 0) or 0) for r in cell_rows), default=0)
        points.append(DomainPoint(
            k=k, domain=str(cell_rows[0].get("domain", "")), mode=str(cell_rows[0].get("mode", "")),
            n_cells=len(cell_rows), ttr_median=agg["median"], ttr_max=agg["max"],
            n_ready=agg["n_ready"],
            readiness_median=(float(np.median(rd)) if rd else None),
            ready_frac=(float(np.mean(ready_flags)) if ready_flags else 0.0),
            n_reval_demote=demote, harness_version=ver))
    return points


def build_curves(logs_by_mode: Dict[str, List[dict]]) -> Dict[str, List[DomainPoint]]:
    """{mode: rows} → {mode: [DomainPoint...]}。"""
    return {mode: per_domain_points(rows) for mode, rows in logs_by_mode.items()}


# ════════════════════════════ headline 判据 ════════════════════════════
def _by_k(points: List[DomainPoint]) -> Dict[int, DomainPoint]:
    return {p.k: p for p in points}


def _trend_nondecreasing(vals: List[Optional[float]], tol: float = 0.05) -> Optional[bool]:
    """末点不显著低于首点（允许 tol 回弹）；不足 2 个有限点 → None（无法判定）。"""
    fin = [v for v in vals if v is not None and math.isfinite(v)]
    if len(fin) < 2:
        return None
    return bool(fin[-1] >= fin[0] - tol)


def forward_transfer_verdict(curves: Dict[str, List[DomainPoint]], *, tol: float = 0.05) -> dict:
    """C(updating, memory-on) vs A(scratch, memory-off) 前向迁移判据 + 三 bootstrap 分解 + 护栏。

    诚实预期（Experiment_Design_Final S2）：不要求单调上升，承诺「C ≥ A 且 C 不退化」。
    """
    c = _by_k(curves.get(MEMORY_ON, []))     # updating
    a = _by_k(curves.get(MEMORY_OFF, []))    # scratch
    b = _by_k(curves.get(FROZEN, []))        # frozen
    shared = sorted(set(c) & set(a))

    per_k = []
    rd_deltas, ttr_gains = [], []
    for k in shared:
        ck, ak = c[k], a[k]
        # readiness：越高越好 → C−A 为正=记忆助益
        rd_d = (ck.readiness_median - ak.readiness_median
                if (ck.readiness_median is not None and ak.readiness_median is not None) else None)
        # ttr：越低越好 → A−C 为正=C 更快达标
        ttr_g = (ak.ttr_median - ck.ttr_median
                 if (ck.ttr_median is not None and ak.ttr_median is not None) else None)
        if rd_d is not None:
            rd_deltas.append(rd_d)
        if ttr_g is not None:
            ttr_gains.append(ttr_g)
        # 三 bootstrap：B−A(记忆价值), C−B(继续更新价值)（readiness@budget 代理）
        bk = b.get(k)
        b_minus_a = (bk.readiness_median - ak.readiness_median
                     if (bk and bk.readiness_median is not None and ak.readiness_median is not None) else None)
        c_minus_b = (ck.readiness_median - bk.readiness_median
                     if (bk and bk.readiness_median is not None and ck.readiness_median is not None) else None)
        per_k.append({
            "k": k, "domain": ck.domain,
            "readiness_C": ck.readiness_median, "readiness_A": ak.readiness_median,
            "readiness_C_minus_A": rd_d,
            "ttr_C": ck.ttr_median, "ttr_A": ak.ttr_median, "ttr_gain_A_minus_C": ttr_g,
            "memory_value_B_minus_A": b_minus_a, "update_value_C_minus_B": c_minus_b,
            "n_reval_demote_C": ck.n_reval_demote,
        })

    mean_rd = float(np.mean(rd_deltas)) if rd_deltas else None
    mean_ttr_gain = float(np.mean(ttr_gains)) if ttr_gains else None
    # memory_helps：readiness 优先（C≥A 平均），无 finite readiness 则退 ttr（C 不慢于 A）
    if mean_rd is not None:
        memory_helps = bool(mean_rd >= -tol)
    elif mean_ttr_gain is not None:
        memory_helps = bool(mean_ttr_gain >= 0)
    else:
        memory_helps = None
    no_degradation = _trend_nondecreasing([c[k].readiness_median for k in sorted(c)], tol=tol)
    total_demote = sum(p.n_reval_demote for p in curves.get(MEMORY_ON, []))

    # discriminative：C vs A 是否有**非平凡**分离（否则全饱和/全 0 差 → 判据空转，不能下"成立"结论）。
    # readiness 差 > tol，或 ttr 差 ≥ 1 round，才算有信号。
    max_abs_rd = max((abs(d) for d in rd_deltas), default=0.0)
    max_abs_ttr = max((abs(g) for g in ttr_gains), default=0.0)
    discriminative = bool(max_abs_rd > tol or max_abs_ttr >= 1.0)

    supported = None
    if not discriminative:
        supported = None                    # 无分离信号 → 不可结论（饱和 demo / 域太少）
    elif memory_helps is not None and no_degradation is not None:
        supported = bool(memory_helps and no_degradation)
    elif memory_helps is not None:
        supported = bool(memory_helps)      # 单域无法判退化 → 仅据 helps

    return {
        "shared_domains": shared,
        "per_k": per_k,
        "mean_readiness_C_minus_A": mean_rd,
        "mean_ttr_gain_A_minus_C": mean_ttr_gain,
        "memory_helps": memory_helps,
        "no_degradation": no_degradation,
        "discriminative": discriminative,
        "forward_transfer_supported": supported,
        "neg_transfer_guardrail_fired": total_demote > 0,
        "total_reval_demote_C": int(total_demote),
        "n_shared_domains": len(shared),
        "n_finite_readiness_deltas": len(rd_deltas),
    }


def analyze(logs_by_mode: Dict[str, List[dict]]) -> dict:
    """一站式：curves + verdict。供 runner / 测试直接调。"""
    curves = build_curves(logs_by_mode)
    return {
        "curves": {m: [p.to_dict() for p in pts] for m, pts in curves.items()},
        "verdict": forward_transfer_verdict(curves),
    }
