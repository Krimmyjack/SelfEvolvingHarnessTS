"""sandbox/ — 算子流水线的隔离执行（plan.md §2）。

Phase 0：in-process 执行 + 异常围堵（run_pipeline）。
Phase 1 硬化：subprocess 隔离 + 硬超时（Windows 无 SIGALRM，走进程级超时）。
"""
from .executor import ExecutionResult, run_pipeline

__all__ = ["ExecutionResult", "run_pipeline"]
