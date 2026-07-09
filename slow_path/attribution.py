"""slow_path/attribution.py — outcome-calibrated 信用分配（借 OPD-Evolver 公式11）。

把 OPD 的"只在被检索任务上估值 + 置信因子 γ=1−1/√(1+N⁺)"搬到我们的设定：用 validator 实测的
held_in/held_out(a) val_loss delta（>0=改善）作为 outcome（替代 OPD 的 reward R），逐 (cell, op)
累积 op 的贡献，γ(N) 下调低样本估计。proposer 据此在该 cell 优先/规避相应算子 → 提升模板命中率。

贡献符号：候选 prefer/enable 的算子 +delta；ban/disable 的算子 -delta（禁它有用=它有害）。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _gamma(n: int) -> float:
    """置信因子 γ=1−1/√(1+N)（OPD 公式11）：N=1→0.29, 2→0.42, 3→0.50, 10→0.70。"""
    return 1.0 - 1.0 / math.sqrt(1.0 + n) if n > 0 else 0.0


def ops_credit(patch, delta: float) -> Dict[str, float]:
    """从一个 EditPatch + 其 delta(>0=改善) 抽 per-op 贡献。"""
    out: Dict[str, float] = {}
    path = patch.path
    if "task_templates" in path and hasattr(patch.value, "stages"):
        for st in patch.value.stages:
            for op in st.preferred_ops:
                out[op] = out.get(op, 0.0) + delta
            for op in st.banned_ops:
                out[op] = out.get(op, 0.0) - delta
    elif path.startswith("l2.active_operators."):
        op = path.rsplit(".", 1)[-1]
        out[op] = delta if patch.value else -delta            # enable:+ / disable:-
    elif path.startswith("l2.operator_defaults."):
        out[path.rsplit(".", 1)[-1]] = delta                  # 调参≈promote 该算子
    return out


class AttributionStore:
    """逐 (cell, op) 累积贡献；value = γ(N)·mean。跨轮累积（无 decay，Phase 2b 够用）。"""

    def __init__(self):
        self._obs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    def record(self, cell_id: str, patch, delta: float) -> None:
        if not math.isfinite(delta):
            return
        for op, c in ops_credit(patch, delta).items():
            self._obs[cell_id][op].append(c)

    def value(self, cell_id: str) -> Dict[str, Tuple[float, int]]:
        out: Dict[str, Tuple[float, int]] = {}
        for op, cs in self._obs.get(cell_id, {}).items():
            n = len(cs)
            out[op] = (_gamma(n) * (sum(cs) / n), n)
        return out

    def summary(self, cell_id: str, top: int = 3) -> Dict[str, List[Tuple[str, float, int]]]:
        """给 proposer 的精简表：top 正(prefer) + top 负(avoid)。"""
        v = self.value(cell_id)
        ranked = sorted(v.items(), key=lambda kv: kv[1][0])
        avoid = [(op, round(val, 4), n) for op, (val, n) in ranked if val < -1e-6][:top]
        prefer = [(op, round(val, 4), n) for op, (val, n) in reversed(ranked) if val > 1e-6][:top]
        return {"prefer": prefer, "avoid": avoid}
