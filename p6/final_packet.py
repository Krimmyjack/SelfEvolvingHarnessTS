"""p6/final_packet.py — P6 运行结束结果包最小写入器（建议 41 / prereg §4 台账"外锚"）。

prereg §4 冻结："**hash 链固有极限**：末条事件的自洽篡改裸链不可捕获——**运行结束的结果包
必须外锚 ledger chain_tip**（写入最终 VERDICT 与 freeze 追加记录）。"本模块把外锚所需最小集
{ledger chain_tip, freeze SHA 集, selection/materialization SHA, claim 分支} 写成**单个**外锚
JSON（含内嵌 packet_sha256，与 materializer/split_manifest 同款"canonical + 剔易变字段"口径），
供最终 VERDICT 与 freeze 追加记录引用比对。

红线：stdlib-only；无网络/RNG；**不产生真实 results 文件**（落盘路径由调用方传入）；不读写
任何现有文件。典型用法（调用方）：
    write_final_packet(results_path, chain_tip=gate.chain_tip,
                       manifest_sha=manifest.manifest_sha,
                       materialization_sha=sm.materialization_sha,
                       freeze_shas={f: sha256(f) ...}, claim_branch="B-weak")
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

__all__ = [
    "CLAIM_BRANCHES",
    "FINAL_PACKET_SCHEMA",
    "compute_packet_sha",
    "load_final_packet",
    "write_final_packet",
]

FINAL_PACKET_SCHEMA = "p6-final-packet/1"
#: prereg §0 冻结 claim 分支词汇表（U 转移限定词单独作为可选字段，不改分支枚举）。
CLAIM_BRANCHES = ("B-strong", "B-weak", "B-partial", "B-null")
_HEX64 = set("0123456789abcdef")
#: 顶层易变字段：不参与 packet_sha256，load 时剔除（对齐 materializer._VOLATILE_KEYS 风格）。
_VOLATILE_KEYS = frozenset({"packet_sha256", "created_at", "written_at", "timestamp", "ts"})


def _check_sha(name: str, sha: Any) -> str:
    s = str(sha)
    if len(s) != 64 or any(c not in _HEX64 for c in s):
        raise ValueError(f"{name} 必须是 64 位十六进制 sha256，got {sha!r}")
    return s


def compute_packet_sha(payload: Mapping[str, Any]) -> str:
    """canonical JSON（sort_keys、ASCII、紧凑分隔符）剔除顶层易变字段后取 sha256。"""
    clean = {k: payload[k] for k in payload if k not in _VOLATILE_KEYS}
    canonical = json.dumps(clean, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_payload(
    *,
    chain_tip: str,
    manifest_sha: str,
    materialization_sha: str,
    freeze_shas: Mapping[str, str],
    claim_branch: str,
    u_transfer: Optional[bool] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if str(claim_branch) not in CLAIM_BRANCHES:
        raise ValueError(f"claim_branch {claim_branch!r} 不在冻结分支 {CLAIM_BRANCHES}")
    fs = {str(k): _check_sha(f"freeze_shas[{k!r}]", v) for k, v in dict(freeze_shas).items()}
    if not fs:
        raise ValueError("freeze_shas 不能为空（freeze SHA 集是外锚的组成部分）")
    payload: Dict[str, Any] = {
        "schema_version": FINAL_PACKET_SCHEMA,
        "ledger_chain_tip": _check_sha("chain_tip", chain_tip),
        "selection_manifest_sha": _check_sha("manifest_sha", manifest_sha),
        "materialization_sha": _check_sha("materialization_sha", materialization_sha),
        "freeze_shas": dict(sorted(fs.items())),
        "claim_branch": str(claim_branch),
        "u_transfer_qualifier": (None if u_transfer is None else bool(u_transfer)),
    }
    if extra is not None:
        # 只接受 JSON-native 可序列化的附加字段（canonical 复制，防止夹带不可序列化对象）。
        payload["extra"] = json.loads(json.dumps(extra, sort_keys=True, ensure_ascii=True))
    return payload


def write_final_packet(
    path: Any,
    *,
    chain_tip: str,
    manifest_sha: str,
    materialization_sha: str,
    freeze_shas: Mapping[str, str],
    claim_branch: str,
    u_transfer: Optional[bool] = None,
    extra: Optional[Mapping[str, Any]] = None,
    created_at: Optional[str] = None,
) -> str:
    """写外锚结果包 JSON（含内嵌 packet_sha256），返回 packet_sha256。落盘路径由调用方给。

    freeze_shas：{冻结件名 → 64hex sha256}（prereg §7 冻结清单的实际 SHA 集）。
    claim_branch ∈ CLAIM_BRANCHES；u_transfer=True/False/None（U 转移限定词是否成立）。
    任一 sha 非 64hex / claim 非法 / freeze 空 → ValueError。
    """
    payload = _build_payload(
        chain_tip=chain_tip, manifest_sha=manifest_sha,
        materialization_sha=materialization_sha, freeze_shas=freeze_shas,
        claim_branch=claim_branch, u_transfer=u_transfer, extra=extra,
    )
    sha = compute_packet_sha(payload)
    doc = dict(payload)
    doc["packet_sha256"] = sha
    if created_at is not None:
        doc["created_at"] = str(created_at)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sha


def load_final_packet(path: Any) -> Dict[str, Any]:
    """读回外锚结果包；剔除易变字段、校验内嵌 packet_sha256（漂移/篡改 → ValueError）。"""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("final packet 顶层必须是 JSON object")
    embedded = doc.get("packet_sha256")
    payload = {k: v for k, v in doc.items() if k not in _VOLATILE_KEYS}
    recomputed = compute_packet_sha(payload)
    if embedded is None or str(embedded) != recomputed:
        raise ValueError(
            f"packet_sha256 校验失败（漂移/篡改）：内嵌 {embedded!r} != 重算 {recomputed}"
        )
    return payload
