"""slow_path/validator.py — R5：Layer1 proxy 负向预筛 → Layer2 grounded 三 split + 接受律。

grounded 是唯一 accept 裁判（不变量 #1）。候选 harness 通过 snapshot→apply→eval→restore 临时构造，
绝不污染当前 harness。每次 validate 都对**传入的当前 harness**重新算 baseline——契合 §6.1「逐候选
对当前 harness 重验」决策（消除过期候选），故 evolve 串行合并时无 staleness。

接受律（plan.md §4.2 / Implementation_Design §2.C）：
  held_in 兑现（val_cand < val_cur - ε）∧ held_out(a) 不退化（≤ val_cur + ε）∧
  held_out(b) Pareto 安全（无任一 cell 退化 > ε）∧ 至少一轴改善。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config.thresholds import EPS_NARROW, TAU_PROXY, S_SEEDS
from ..evaluators import get_evaluator
from ..harness.editable_surfaces import validate as surface_validate
from ..fast_path.pipeline import process as fast_process
from .batch_builder import CellSample, make_eval_sample

# 随机 grounded substrate（训练有 seed 方差，需多 seed 平均压噪）；frozen forecast + anomaly 检测器确定性 → S=1
_STOCHASTIC_GROUNDED = {"classification"}


def grounded_val_loss(harness, samples: List[CellSample], layer: str = "grounded",
                      seed: int = 0) -> float:
    """对一批 CellSample：用给定 harness 跑 fast_path 产 ready → 组 eval batch → evaluator.val_loss。

    ready 批只构一次；随机 substrate（classify InceptionLite）的 grounded 上对 S_SEEDS 个 seed 取均值
    压训练噪声（B.2 #6 substrate-aware），否则 σ_A 可能超过 ε 让接受律被噪声主导。
    """
    if not samples:
        return float("nan")
    task = samples[0].task_type
    eb = [make_eval_sample(fast_process(s.raw, s.task_type, harness, store=None)[1], s)
          for s in samples]
    ev = get_evaluator(task)
    if layer == "grounded" and task in _STOCHASTIC_GROUNDED and S_SEEDS > 1:
        return float(np.mean([ev.evaluate(eb, layer="grounded", seed=s) for s in range(S_SEEDS)]))
    return ev.evaluate(eb, layer=layer, seed=seed)


@dataclass
class ValidationOutcome:
    accept: bool
    reason: str
    resolved_scope: Optional[str] = None
    val_in_cur: float = float("nan")
    val_in_cand: float = float("nan")
    val_a_cur: float = float("nan")
    val_a_cand: float = float("nan")
    pareto_safe: bool = True
    pareto_violator: str = ""

    def deltas(self) -> Dict[str, float]:
        return {"held_in": self.val_in_cur - self.val_in_cand,        # >0 = 改善
                "held_out_a": self.val_a_cur - self.val_a_cand}


class Validator:
    def __init__(self, eps: float = EPS_NARROW, use_proxy_prescreen: bool = False,
                 proxy_margin: float = 0.10):
        # use_proxy_prescreen 默认 OFF（「无证不用」）：proxy 仅在该 cell 经 calibration(τ) 证可信时才该启用；
        # 且 frozen+probe 让 grounded 已很廉价（forecast 上 proxy=训 DLinear 反而更贵）。calibration 接入后再开。
        self.eps = eps
        self.use_proxy = use_proxy_prescreen
        self.proxy_margin = proxy_margin     # proxy 仅负向：候选比当前差 > margin 才杀

    def _candidate_val(self, harness, patch, splits):
        """snapshot→apply→在三 split 上评候选→restore。返回 (val_in, val_a, pareto_safe, violator)。"""
        held_in, held_out_a, held_out_b = splits
        snap = harness.snapshot()
        try:
            harness.apply_edit(patch, raise_on_reject=True)
            v_in = grounded_val_loss(harness, held_in)
            v_a = grounded_val_loss(harness, held_out_a) if held_out_a else v_in
            cand_b = {c: grounded_val_loss(harness, s) for c, s in held_out_b.items()}
        finally:
            harness.restore(snap)
        return v_in, v_a, cand_b

    def validate(self, patch, harness, cell_id: str, splits) -> ValidationOutcome:
        # 形态/越界/受保护面机械校验（先于昂贵 grounded）
        res = surface_validate(patch, harness)
        if not res.ok:
            return ValidationOutcome(False, f"invalid:{res.reason}")
        held_in, held_out_a, held_out_b = splits
        if not held_in:
            return ValidationOutcome(False, "empty held_in", res.resolved_scope)

        # Layer 1：proxy 负向预筛（仅杀明显退化；proxy 不裁决 accept）
        if self.use_proxy:
            try:
                p_cur = grounded_val_loss(harness, held_in, layer="proxy")
                snap = harness.snapshot()
                try:
                    harness.apply_edit(patch, raise_on_reject=True)
                    p_cand = grounded_val_loss(harness, held_in, layer="proxy")
                finally:
                    harness.restore(snap)
                if np.isfinite(p_cur) and np.isfinite(p_cand) and p_cand - p_cur > self.proxy_margin:
                    return ValidationOutcome(False, f"proxy_prescreen_reject(Δ={p_cand - p_cur:.3f})",
                                             res.resolved_scope)
            except Exception:
                pass   # proxy 失败不阻断；grounded 是唯一裁判

        # Layer 2：grounded 三 split
        v_in_cur = grounded_val_loss(harness, held_in)
        v_a_cur = grounded_val_loss(harness, held_out_a) if held_out_a else v_in_cur
        cur_b = {c: grounded_val_loss(harness, s) for c, s in held_out_b.items()}
        v_in_cand, v_a_cand, cand_b = self._candidate_val(harness, patch, splits)

        improve_in = v_in_cand < v_in_cur - self.eps
        no_regress_a = v_a_cand <= v_a_cur + self.eps
        improve_a = v_a_cand < v_a_cur - self.eps
        pareto_safe, violator = True, ""
        for c, cur in cur_b.items():
            if np.isfinite(cur) and np.isfinite(cand_b[c]) and cand_b[c] > cur + self.eps:
                pareto_safe, violator = False, c
                break

        accept = bool(improve_in and no_regress_a and pareto_safe and (improve_in or improve_a))
        if accept:
            reason = "accept"
        elif not improve_in:
            reason = f"held_in_not_fulfilled(Δ={v_in_cur - v_in_cand:.3f}<ε={self.eps})"
        elif not no_regress_a:
            reason = f"held_out_a_regress(Δ={v_a_cur - v_a_cand:.3f})"
        else:
            reason = f"pareto_violation@{violator}"
        return ValidationOutcome(accept, reason, res.resolved_scope,
                                 v_in_cur, v_in_cand, v_a_cur, v_a_cand, pareto_safe, violator)
