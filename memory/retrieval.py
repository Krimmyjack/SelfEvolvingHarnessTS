"""memory/retrieval.py — d_struct kNN 检索（plan.md §2/R3/L3）。

在 EvidenceStore 的记录上按 conditioning-key 相似度做 kNN：
  retrieve_success → 相似上下文里 output_status=ready 的成功 program 片段（暖启动 COMPOSE）；
  retrieve_failures → 相似上下文里的 failure_signature 警告（避免重蹈）。
冷启动（store 空/None）→ 空列表。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..conditioning.distance import similarity


class MemoryIndex:
    def __init__(self, store, alpha: float = 0.5):
        self.store = store
        self.alpha = alpha

    def _all_records(self) -> List:
        if self.store is None:
            return []
        return [r for c in self.store.get_all_cells() for r in self.store.query_by_cell(c)]

    @staticmethod
    def _task_type(k: Dict[str, Any]) -> Any:
        """conditioning_key → task type（缺失容忍：None 不参与硬过滤）。"""
        return (k or {}).get("task", {}).get("type") if isinstance(k, dict) else None

    def _task_filtered_records(self, key: Dict[str, Any]) -> List:
        """F1（S0.1）：跨样本检索必须先按 task 硬过滤——forecast 经验不得进 classify 查询。
        query 或记录缺 task.type 时不过滤该条（保守，不静默丢证据）。"""
        qt = self._task_type(key)
        recs = self._all_records()
        if qt is None:
            return recs
        return [r for r in recs
                if self._task_type(r.conditioning_key) in (None, qt)]

    def retrieve_success(self, key: Dict[str, Any], k: int = 5, min_sim: float = 0.6) -> List[Dict]:
        out = []
        for r in self._task_filtered_records(key):
            if r.verification_result.get("output_status") != "ready":
                continue
            s = similarity(key, r.conditioning_key, self.alpha)
            if s >= min_sim:
                out.append((s, r))
        out.sort(key=lambda x: x[0], reverse=True)
        return [{"sim": round(s, 3), "cell": r.cell_id, "program": r.program} for s, r in out[:k]]

    def retrieve_failures(self, key: Dict[str, Any], k: int = 5) -> List[Dict]:
        scored = []
        for r in self._task_filtered_records(key):
            sig = r.verification_result.get("failure_signature")
            if sig:
                scored.append((similarity(key, r.conditioning_key, self.alpha), sig))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen, out = set(), []
        for s, sig in scored:
            if sig in seen:
                continue
            seen.add(sig)
            out.append({"sim": round(s, 3), "signature": sig})
            if len(out) >= k:
                break
        return out
