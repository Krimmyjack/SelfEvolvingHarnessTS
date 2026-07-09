"""memory/evidence_store.py — EvidenceRecord（快慢路径唯一耦合点）+ EvidenceStore 后端。

EvidenceRecord schema 见 plan.md §3.3。EvidenceStore 接口见 Implementation_Design §3.3：
write / query_by_cell / get|set_cached_val_loss / get_all_cells。Phase 0 用内存 dict-of-lists
（按 cell_id 索引）；接口不变，Phase 1+ 换 SQLite/Parquet。
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceRecord:
    conditioning_key: Dict[str, Any]
    cell_id: str
    harness_version: int
    program: Any                                  # Program 的 dict 形态（可序列化）
    execution_trace: List[Dict[str, Any]]
    verification_result: Dict[str, Any]           # {passed, gate_results, failure_signature, role_b_score, output_status}
    batch_id: str = ""
    timestamp: float = field(default_factory=time.time)
    # 2.0-⑤（v1.1c 期限=首个消费 routed 证据的实验之前——即 Harness 垂直切片，2026-07-05 兑现）：
    # 路由证据 {policy_version/artifact_sha, selected_action, predicted_utility, uncertainty,
    #           support, grounded_utility(事后回填), decision_context}。非 routed 路径 = None（旧行为不变）。
    routing: Optional[Dict[str, Any]] = None

    @property
    def output_status(self) -> str:
        return self.verification_result.get("output_status", "")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceStore:
    """Phase 0 内存实现：按 cell_id 索引的 dict-of-lists + per-cell val_loss 缓存。"""

    def __init__(self) -> None:
        self._by_cell: Dict[str, List[EvidenceRecord]] = defaultdict(list)
        self._val_loss_cache: Dict[str, float] = {}

    # —— 快路径 emit 写入 ——
    def write(self, record: EvidenceRecord) -> None:
        self._by_cell[record.cell_id].append(record)

    # —— Mining 按 cell 取证据 ——
    def query_by_cell(self, cell_id: str) -> List[EvidenceRecord]:
        return list(self._by_cell.get(cell_id, []))

    # —— Validation 缓存（§2.A②）——
    def get_cached_val_loss(self, cell_key: str) -> Optional[float]:
        return self._val_loss_cache.get(cell_key)

    def set_cached_val_loss(self, cell_key: str, val_loss: float) -> None:
        self._val_loss_cache[cell_key] = val_loss

    # —— 慢路径遍历 ready cell ——
    def get_all_cells(self) -> List[str]:
        return list(self._by_cell.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_cell.values())
