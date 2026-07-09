"""sandbox/executor.py — 算子流水线执行（plan.md §2 / R1）。

Phase 0：in-process，逐步执行算子并围堵异常；返回 ExecutionResult（含 per-step trace，
trace 里标 source=template|llm_custom 供信用分配）。不做硬超时（Windows 无 SIGALRM），
Phase 1 换 subprocess 隔离 + 进程级超时。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..operators.registry import TOOL_REGISTRY, canonicalize
from ..operators._provenance import record as _prov_record


@dataclass
class ExecutionResult:
    artifact: Optional[np.ndarray]
    ok: bool
    error: str = ""
    trace: List[Dict[str, Any]] = field(default_factory=list)


def run_pipeline(steps: Sequence[Tuple[str, dict]], x,
                 registry: Optional[Dict] = None, source: str = "template") -> ExecutionResult:
    """按序执行 (op_name, params) 列表。任一步抛错 → ok=False，artifact=None，trace 记录失败点。"""
    registry = registry if registry is not None else TOOL_REGISTRY
    cur = np.asarray(x, dtype=float).ravel()
    trace: List[Dict[str, Any]] = []

    for name, params in steps:
        canon = canonicalize(name)                  # S0.7-6：trace 同录 requested/canonical
        rec = {"op": name, "canonical": canon, "source": source, "ok": False, "error": ""}
        if canon != name:
            _prov_record(name, canon, "compat_alias")
        fn = registry.get(name)
        if fn is None:
            rec["error"] = "op not in registry"
            trace.append(rec)
            return ExecutionResult(None, False, f"unknown op {name!r}", trace)
        try:
            out = np.asarray(fn(cur, **(params or {})), dtype=float).ravel()
            rec["ok"] = True
            cur = out
        except Exception as e:                      # 围堵任意算子异常（含 ShapeChangingNotSupported）
            rec["error"] = f"{type(e).__name__}: {e}"
            trace.append(rec)
            return ExecutionResult(None, False, rec["error"], trace)
        trace.append(rec)

    return ExecutionResult(cur, True, "", trace)
