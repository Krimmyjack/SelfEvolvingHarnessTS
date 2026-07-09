"""fast_path/ — R2 快路径（per-input）：只产证据，不改 harness。

Phase 0：PERCEIVE→COMPOSE(heuristic)→EXECUTE→VERIFY→EMIT（不含 RETRIEVE，Phase 2 补）。
"""
from .pipeline import process
from .perceive import perceive
from .compose import (compose, compose_llm, compose_recovery, is_operator_eligible,
                      usable_ops, Program, ProgramStep)
from .execute import execute
from .retrieve import Retriever
from .verify import run_gates, role_b_score, GateResult

__all__ = [
    "process", "perceive", "compose", "compose_llm", "compose_recovery", "is_operator_eligible",
    "usable_ops", "Program", "ProgramStep", "execute", "Retriever", "run_gates", "role_b_score",
    "GateResult",
]
