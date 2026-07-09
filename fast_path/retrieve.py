"""fast_path/retrieve.py — RETRIEVE 步（plan.md §4.1）：SuccessRetriever + FailureRetriever。

给 conditioning_key，从 L3 记忆取相似上下文的成功 program 片段（暖启动）+ failure 警告。
冷启动（memory=None / 空 store）→ 空，快路径退化为 Phase 0 行为。检索参数取 harness.L3.retrieval_config。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..memory.retrieval import MemoryIndex


class Retriever:
    def __init__(self, store=None, retrieval_config=None):
        self.cfg = retrieval_config
        alpha = getattr(retrieval_config, "alpha", 0.5)
        self.index = MemoryIndex(store, alpha) if store is not None else None

    def retrieve(self, key: Dict[str, Any]) -> Dict[str, list]:
        if self.index is None:
            return {"prior_fragments": [], "failure_warnings": []}
        k1 = getattr(self.cfg, "max_prior_fragments", 5)
        k2 = getattr(self.cfg, "max_failure_warnings", 5)
        ms = getattr(self.cfg, "min_similarity", 0.6)
        return {"prior_fragments": self.index.retrieve_success(key, k1, ms),
                "failure_warnings": self.index.retrieve_failures(key, k2)}
