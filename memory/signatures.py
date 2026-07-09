"""memory/signatures.py — failure/strength signature 聚合 + 保留策略（plan.md §2/L3）。

借 OPD-Evolver 的溯源思路（success/usage/last_used）做 L3 维护：聚合 EvidenceStore 里的
failure_signature（支持度 + 末见版本），保留 top-N（按支持度）。strength 由 slow_path 的
mine_strength + merger.consolidate 维护，这里只做 failure 侧的轻量聚合/淘汰。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class SignatureStat:
    signature_id: str
    support: int = 0                 # 出现次数（≈ OPD usage_count）
    last_seen_version: int = 0
    cells: int = 0                   # 出现于多少个 cell（跨 cell 普遍性）


def aggregate_failures(store) -> Dict[str, SignatureStat]:
    if store is None:
        return {}
    per_sig_cells: Dict[str, set] = {}
    counter: Counter = Counter()
    last_ver: Dict[str, int] = {}
    for cell in store.get_all_cells():
        for r in store.query_by_cell(cell):
            sig = r.verification_result.get("failure_signature")
            if not sig:
                continue
            counter[sig] += 1
            per_sig_cells.setdefault(sig, set()).add(cell)
            last_ver[sig] = max(last_ver.get(sig, 0), r.harness_version)
    return {s: SignatureStat(s, counter[s], last_ver.get(s, 0), len(per_sig_cells.get(s, ())))
            for s in counter}


def retain_top(stats: Dict[str, SignatureStat], max_entries: int = 50) -> Dict[str, SignatureStat]:
    """保留策略：按 (support, 跨cell普遍性) 取 top-N，淘汰长尾噪声签名。"""
    ranked = sorted(stats.values(), key=lambda s: (s.support, s.cells), reverse=True)
    return {s.signature_id: s for s in ranked[:max_entries]}
