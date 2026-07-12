"""Evidence records and stores for fast/slow path coupling."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


@dataclass
class EvidenceRecord:
    conditioning_key: Dict[str, Any]
    cell_id: str
    harness_version: int
    program: Any
    execution_trace: List[Dict[str, Any]]
    verification_result: Dict[str, Any]
    batch_id: str = ""
    timestamp: float = field(default_factory=time.time)
    routing: Optional[Dict[str, Any]] = None

    @property
    def output_status(self) -> str:
        return self.verification_result.get("output_status", "")

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvidenceRecord":
        return cls(
            conditioning_key=dict(raw.get("conditioning_key") or {}),
            cell_id=str(raw.get("cell_id") or ""),
            harness_version=int(raw.get("harness_version") or 0),
            program=raw.get("program"),
            execution_trace=list(raw.get("execution_trace") or []),
            verification_result=dict(raw.get("verification_result") or {}),
            batch_id=str(raw.get("batch_id") or ""),
            timestamp=float(raw.get("timestamp") or time.time()),
            routing=dict(raw["routing"]) if isinstance(raw.get("routing"), dict) else None,
        )


class EvidenceStore:
    """EvidenceStore with the original in-memory index plus optional JSONL persistence."""

    def __init__(self, persist_path: str | Path | None = None, *, load_existing: bool = False) -> None:
        self._by_cell: Dict[str, List[EvidenceRecord]] = defaultdict(list)
        self._val_loss_cache: Dict[str, float] = {}
        self.persist_path = Path(persist_path) if persist_path is not None else None
        if load_existing and self.persist_path is not None and self.persist_path.exists():
            for record in self._read_jsonl_records(self.persist_path):
                self._write_memory_only(record)

    def _write_memory_only(self, record: EvidenceRecord) -> None:
        self._by_cell[record.cell_id].append(record)

    def write(self, record: EvidenceRecord) -> None:
        self._write_memory_only(record)
        if self.persist_path is not None:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with self.persist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
                f.write("\n")

    def query_by_cell(self, cell_id: str) -> List[EvidenceRecord]:
        return list(self._by_cell.get(cell_id, []))

    def get_cached_val_loss(self, cell_key: str) -> Optional[float]:
        return self._val_loss_cache.get(cell_key)

    def set_cached_val_loss(self, cell_key: str, val_loss: float) -> None:
        self._val_loss_cache[cell_key] = val_loss

    def get_all_cells(self) -> List[str]:
        return list(self._by_cell.keys())

    def get_all_records(self) -> List[EvidenceRecord]:
        return [record for cell in self.get_all_cells() for record in self.query_by_cell(cell)]

    def iter_records(self) -> Iterable[EvidenceRecord]:
        for cell in self.get_all_cells():
            yield from self.query_by_cell(cell)

    def save_jsonl(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for record in self.iter_records():
                f.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False))
                f.write("\n")

    @staticmethod
    def _read_jsonl_records(path: Path) -> Iterable[EvidenceRecord]:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid evidence JSONL at {path}:{line_no}: {exc}") from exc
                yield EvidenceRecord.from_dict(raw)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EvidenceStore":
        store = cls()
        for record in cls._read_jsonl_records(Path(path)):
            store._write_memory_only(record)
        return store

    def replay_contract(self) -> dict[str, Any]:
        records = list(self.iter_records())
        return {
            "schema": "evidence_store_replay_v1",
            "n_records": len(records),
            "cells": sorted(self.get_all_cells()),
            "batches": sorted({record.batch_id for record in records if record.batch_id}),
            "harness_versions": sorted({record.harness_version for record in records}),
        }

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_cell.values())
