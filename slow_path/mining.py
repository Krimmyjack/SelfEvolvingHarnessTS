"""slow_path/mining.py — R6：结构流 + 质量流 + 信用分配 + Strength（对偶）。

结构流：EvidenceRecord 按 failure_signature group-by（support≥min_support）。
质量流：当前 harness 的 grounded val_loss + 可改进性 gap（vs seasonal_naive floor / 任务下限）。
信用分配：Role B（spike_preservation 等）× execution_trace 算子 → suspicious_operators（提示，非因果）。
Strength（对偶）：val_loss 显著优于 floor 的 cell → must-preserve（供 merger consolidator 写受保护区）。
输出 weakness_report + strength_report 喂 proposer。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config.thresholds import MIN_SUPPORT, EPS_WIDE
from ..evaluators import get_evaluator, seasonal_naive_floor, role_b_metrics
from .batch_builder import CellSample, make_eval_sample


# S0.3/F3：EvidenceStore 读取的 harness_version 过滤臂（A-7/A-20）。
#   strict = 只当前版本（污染诊断，保守默认）；prev = ≥ version-1（样本量敏感性）；all = 不过滤。
EVIDENCE_VERSION_ARMS = ("strict", "prev", "all")


def filter_evidence_by_version(recs, harness_version: int, arm: str = "strict"):
    """按 harness_version 过滤证据（S0.3）。运行中不自动切臂——由调用方显式选臂、两臂都落盘。"""
    if arm == "all":
        return list(recs)
    if arm == "prev":
        lo = harness_version - 1
        return [r for r in recs if r.harness_version >= lo]
    # strict（默认）
    return [r for r in recs if r.harness_version == harness_version]


@dataclass
class WeaknessReport:
    cell_id: str
    task: str
    current_val_loss: float
    floor: float
    gap: float                                  # current - floor（>0 = 有改进空间）
    improvable: bool                            # gap > ε_wide
    failure_signatures: Dict[str, int] = field(default_factory=dict)
    suspicious_operators: List[str] = field(default_factory=list)
    op_attribution: Dict = field(default_factory=dict)   # {"prefer":[(op,val,n)...], "avoid":[...]}（evolve 注入）
    # S0.3：证据版本过滤前后条数（污染诊断落盘用）
    version_arm: str = "strict"
    evidence_n_raw: int = 0
    evidence_n_kept: int = 0


@dataclass
class StrengthReport:
    cell_id: str
    val_loss: float
    floor: float
    margin: float                               # floor - val_loss（>0 = 优于下限，承重墙候选）
    must_preserve: bool


def _ready_eval(samples: List[CellSample], harness):
    """ready 批只构一次：跑 fast_path → (ready_arts, eval_batch)。供 grounded + floor + 信用分配复用。"""
    from ..fast_path.pipeline import process as fp
    arts = [fp(s.raw, s.task_type, harness, store=None)[1] for s in samples]
    eb = [make_eval_sample(a, s) for a, s in zip(arts, samples)]
    return arts, eb


def _grounded_and_floor(task: str, eb):
    cur = get_evaluator(task).evaluate(eb, layer="grounded")     # 报告用，单 seed 即可（非裁决）
    floor = seasonal_naive_floor(eb) if task == "forecast" else float("nan")
    return cur, floor


def mine_weakness(cell_id: str, samples: List[CellSample], harness,
                  evidence_store=None, version_arm: str = "strict") -> WeaknessReport:
    task = samples[0].task_type
    arts, eb = _ready_eval(samples, harness)                     # 一次 fast_path
    cur, floor = _grounded_and_floor(task, eb)
    gap = (cur - floor) if np.isfinite(floor) else float("nan")
    improvable = bool(np.isfinite(gap) and gap > EPS_WIDE)

    # 结构流：从 EvidenceStore 聚合该 cell 的 failure_signature（S0.3：先按 harness_version 过滤）
    sigs: Dict[str, int] = {}
    n_raw = n_kept = 0
    if evidence_store is not None:
        recs = evidence_store.query_by_cell(cell_id)
        n_raw = len(recs)
        kept = filter_evidence_by_version(recs, harness.version, version_arm)
        n_kept = len(kept)
        c = Counter(r.verification_result.get("failure_signature")
                    for r in kept if r.verification_result.get("failure_signature"))
        sigs = {s: n for s, n in c.items() if n >= MIN_SUPPORT}

    # 信用分配（轻量）：anomaly 上 spike_preservation 低 → 平滑类算子可疑（复用已算的 ready_arts）
    suspicious: List[str] = []
    if task == "anomaly_detection":
        sp = [role_b_metrics.spike_preservation(a) for a in arts[:8]]
        if sp and float(np.median(sp)) < 1.0:
            suspicious = ["denoise_savgol", "denoise_median", "denoise_stl", "winsorize"]

    return WeaknessReport(cell_id, task, cur, floor, gap, improvable, sigs, suspicious,
                          version_arm=version_arm, evidence_n_raw=n_raw, evidence_n_kept=n_kept)


def mine_strength(cell_id: str, samples: List[CellSample], harness,
                  margin_thresh: float = EPS_WIDE) -> Optional[StrengthReport]:
    task = samples[0].task_type
    _arts, eb = _ready_eval(samples, harness)                    # 一次 fast_path
    cur, floor = _grounded_and_floor(task, eb)
    if not (np.isfinite(cur) and np.isfinite(floor)):
        return None
    margin = floor - cur
    return StrengthReport(cell_id, cur, floor, margin, must_preserve=bool(margin > margin_thresh))
