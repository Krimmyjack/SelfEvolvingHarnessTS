"""memory/ — R3/L3：证据与检索后端。

Phase 0 已落地：evidence_store.py（EvidenceRecord + 内存 dict-of-lists EvidenceStore）。
Phase 1+：retrieval.py（d_struct kNN）、signatures.py（保留策略）；后端换 SQLite/Parquet。
"""
from .evidence_store import EvidenceRecord, EvidenceStore
from .retrieval import MemoryIndex
from .signatures import SignatureStat, aggregate_failures, retain_top

__all__ = ["EvidenceRecord", "EvidenceStore", "MemoryIndex",
           "SignatureStat", "aggregate_failures", "retain_top"]

from .evidence_schema import MemoryEvidence, build_memory_evidence
