"""evaluation/functional/parallel_eval.py — 最小并行助手（2026-08-13，
用户裁决：独立评估/LLM 格并行提速——非平台——单一函数）。

适用范围（并行安全边界）：
  - 相互独立的确定性评估（跨 series/候选——如 census、supply 搜索）；
  - 相互独立的真实 LLM 格（如 P0 的 case×arm×rep——每次调用全新
    CountingClient，只共享只读 gold/h0）。
不可用于：单轮内探测顺序（chosen-first/budget/首个正向停止=14 条
固定语义——自适应决策的一部分）。

实现：线程池（网络 I/O 型 LLM 调用）或进程池（CPU 型 numpy 评估——
Windows spawn 下任务函数必须模块级可 pickle）。结果按任务提交顺序
返回（确定性）；单任务异常不吞——以 (ok, result_or_exc) 收集。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, Sequence

DEFAULT_WORKERS = 4  # agicto 并发保守值；可调


def run_parallel(
    tasks: Sequence[Callable[[], Any]],
    workers: int = DEFAULT_WORKERS,
    *,
    use_processes: bool = False,
) -> list[tuple[bool, Any]]:
    """并行执行独立任务，按提交顺序返回 (ok, result_or_exception)。
    空任务列表 → 空结果。workers <= 1 → 顺序执行（退化为原行为——
    便于对照）。"""
    if not tasks:
        return []
    if workers <= 1:
        out: list[tuple[bool, Any]] = []
        for fn in tasks:
            try:
                out.append((True, fn()))
            except Exception as exc:  # noqa: BLE001
                out.append((False, exc))
        return out
    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with executor_cls(max_workers=workers) as pool:
        futures = [pool.submit(fn) for fn in tasks]
        out = []
        for fut in futures:
            try:
                out.append((True, fut.result()))
            except Exception as exc:  # noqa: BLE001
                out.append((False, exc))
    return out


__all__ = ["run_parallel"]
