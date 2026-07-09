"""fast_path/verify.py — Gate 链 + Role B（只 log）+ failure_signature（plan.md §3 / 不变量 #2/#4/#5）。

Gate 链（任一失败即停，记录首个失败为 failure_signature）：
  AST       → 程序结构合法（算子都在 TOOL_REGISTRY）
  Skill     → 每个算子 active 且未被该 task 模板 banned（进化关算子 = 物理拦截）
  Contract  → task 契约复查（registry allowed_tasks；D6：recovery/LLM/模板任何路径都不得绕过）
  Sandbox   → 执行未抛错/超时
  Blowup    → 产物有限且未超出 μ±blowup_sigma·σ（原序列分布）
  Constraint→ L1 硬约束：长度保持、无 NaN/Inf、值域 ⊆ [min-3σ, max+3σ]
Role B 是 per-sample 免训练指标，**只 log 不 gate**（不变量 #2：per-sample 质量分不做决策）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..operators.registry import TOOL_REGISTRY
from ..sandbox import ExecutionResult
from .compose import Program


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


def _orig_stats(x: np.ndarray):
    m = ~np.isnan(x)
    xv = x[m] if m.any() else np.array([0.0])
    return float(xv.min()), float(xv.max()), float(xv.mean()), float(xv.std())


def run_gates(original, exec_result: ExecutionResult, program: Program, harness,
              task_type: str, pattern_bin: str = "",
              struct_feats: Optional[Dict[str, float]] = None
              ) -> Tuple[bool, List[GateResult], Optional[str]]:
    original = np.asarray(original, dtype=float).ravel()
    gates: List[GateResult] = []

    def fail(g: GateResult):
        gates.append(g)
        return False, gates, f"{g.name}:{g.detail}"

    # ── AST ──
    unknown = [s.op for s in program.steps if s.op not in TOOL_REGISTRY]
    if unknown:
        return fail(GateResult("ast", False, f"unknown_ops={unknown}"))
    gates.append(GateResult("ast", True))

    # ── Skill（物理强制 banned/inactive；banned 与 compose 同为 cell-scoped）──
    from .compose import cell_banned_ops
    banned = cell_banned_ops(harness, task_type, pattern_bin, struct_feats)
    for s in program.steps:
        if not harness.l2.active_operators.get(s.op, False):
            return fail(GateResult("skill", False, f"inactive_op:{s.op}"))
        if s.op in banned:
            return fail(GateResult("skill", False, f"banned_op:{s.op}"))
    gates.append(GateResult("skill", True))

    # ── Contract（D6：task 契约物理复查，与 compose.is_operator_eligible 同源）──
    from ..operators.registry import OPERATOR_METADATA, canonicalize
    from .compose import _CONTRACT_TASKS
    if task_type in _CONTRACT_TASKS:
        for s in program.steps:
            allowed = OPERATOR_METADATA.get(canonicalize(s.op), {}).get("allowed_tasks")
            if allowed and task_type not in allowed:
                return fail(GateResult("contract", False, f"task_contract:{s.op}"))
    gates.append(GateResult("contract", True))

    # ── Sandbox ──
    if not exec_result.ok or exec_result.artifact is None:
        return fail(GateResult("sandbox", False, exec_result.error or "exec_failed"))
    gates.append(GateResult("sandbox", True))

    art = np.asarray(exec_result.artifact, dtype=float).ravel()
    omin, omax, omean, ostd = _orig_stats(original)

    # ── Blowup ──
    if not np.all(np.isfinite(art)):
        return fail(GateResult("blowup", False, "non_finite"))
    sigma = harness.l4.gate_config.blowup_sigma
    if ostd > 1e-12 and np.max(np.abs(art - omean)) > sigma * ostd:
        return fail(GateResult("blowup", False, f"exceeds_{sigma}sigma"))
    gates.append(GateResult("blowup", True))

    # ── Constraint（L1 硬约束）──
    if art.size != original.size:
        return fail(GateResult("constraint", False, f"len {art.size}!={original.size}"))
    if np.isnan(art).any() or np.isinf(art).any():
        return fail(GateResult("constraint", False, "nan_or_inf"))
    lo, hi = omin - 3 * ostd, omax + 3 * ostd
    if ostd > 1e-12 and (art.min() < lo or art.max() > hi):
        return fail(GateResult("constraint", False, "out_of_range"))
    gates.append(GateResult("constraint", True))

    return True, gates, None


def role_b_score(original, artifact, task_type: str) -> float:
    """免训练保真度代理（与原序列在非缺失位的 Pearson 相关）。**只 log，不 gate。**"""
    o = np.asarray(original, dtype=float).ravel()
    a = np.asarray(artifact, dtype=float).ravel()
    if a.size != o.size:
        return float("nan")
    m = ~np.isnan(o) & ~np.isnan(a)
    if m.sum() < 2:
        return float("nan")
    ov, av = o[m], a[m]
    if ov.std() < 1e-12 or av.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ov, av)[0, 1])
