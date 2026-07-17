from typing import TypedDict


class ExecutionTraceRecord(TypedDict):
    op: str
    canonical: str
    source: str
    ok: bool
    error: str


__all__ = ["ExecutionTraceRecord"]
