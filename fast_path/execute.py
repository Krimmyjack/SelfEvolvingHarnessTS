"""fast_path/execute.py — 经 sandbox 执行 Program → ready_artifact + execution_trace。"""
from __future__ import annotations

from ..sandbox import run_pipeline, ExecutionResult
from .compose import Program


def execute(program: Program, x) -> ExecutionResult:
    return run_pipeline(program.as_pairs(), x, source=program.source)
