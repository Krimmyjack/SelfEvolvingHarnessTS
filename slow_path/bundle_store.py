"""slow_path/bundle_store.py — PolicyBundle 持久化 + 版本链 + rollback（P4，Final_Plan §P4）。

纪律：artifact **不可变 append-only**——每个版本落独立 JSON（含 sha/parent_sha），禁覆盖；
`chain.json` 记 HEAD 指针 + 全事件流（save/rollback），rollback 只移 HEAD 并追加事件，
**绝不删除**已晋升版本（审计可回放）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..policy.edits import PolicyBundle
from ..policy.risk_policy import RiskPolicy, RiskRule


def bundle_to_dict(bundle: PolicyBundle) -> Dict[str, Any]:
    return {
        "version": bundle.version,
        "sha": bundle.sha(),
        "parent_sha": bundle.parent_sha,
        "menu_version": bundle.menu_version,
        "pattern_spec_version": bundle.pattern_spec_version,
        "pattern_spec_proposals": list(bundle.pattern_spec_proposals),
        "memory_writes": list(bundle.memory_writes),
        "risk": {
            "version": bundle.risk.version,
            "rules": [
                {"rule_id": r.rule_id, "when": dict(r.when), "then": dict(r.then),
                 "scope": r.scope, "provenance": dict(r.provenance)}
                for r in bundle.risk.rules
            ],
        },
    }


def bundle_from_dict(d: Dict[str, Any]) -> PolicyBundle:
    rules = tuple(
        RiskRule(rule_id=str(r["rule_id"]), when=dict(r["when"]), then=dict(r["then"]),
                 scope=str(r["scope"]), provenance=dict(r.get("provenance") or {}))
        for r in (d.get("risk") or {}).get("rules", [])
    )
    bundle = PolicyBundle(
        version=str(d["version"]),
        risk=RiskPolicy(version=str((d.get("risk") or {}).get("version", "risk_v0_empty")),
                        rules=rules),
        pattern_spec_version=str(d.get("pattern_spec_version", "P0")),
        pattern_spec_proposals=tuple(d.get("pattern_spec_proposals") or ()),
        memory_writes=tuple(d.get("memory_writes") or ()),
        menu_version=str(d.get("menu_version", "v1")),
        parent_sha=d.get("parent_sha"),
    )
    recorded = d.get("sha")
    if recorded and recorded != bundle.sha():
        raise ValueError(f"bundle {d['version']!r} sha 不一致：记录 {recorded}，重算 {bundle.sha()}"
                         "（artifact 被篡改或序列化损坏）")
    return bundle


class BundleStore:
    """目录式版本仓：{version}.json（不可变）+ chain.json（HEAD + append-only 事件流）。"""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._chain_path = self.root / "chain.json"

    def _chain(self) -> Dict[str, Any]:
        if self._chain_path.exists():
            return json.loads(self._chain_path.read_text(encoding="utf-8"))
        return {"head": None, "events": []}

    def _write_chain(self, chain: Dict[str, Any]) -> None:
        self._chain_path.write_text(json.dumps(chain, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    def _bundle_path(self, version: str) -> Path:
        return self.root / f"{version}.json"

    def save(self, bundle: PolicyBundle, meta: Optional[Dict[str, Any]] = None) -> Path:
        path = self._bundle_path(bundle.version)
        if path.exists():
            raise ValueError(f"bundle 版本已存在（artifact 不可变，禁覆盖）：{bundle.version!r}")
        payload = bundle_to_dict(bundle)
        if meta:
            payload["meta"] = dict(meta)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        chain = self._chain()
        chain["head"] = bundle.version
        chain["events"].append({"event": "save", "version": bundle.version,
                                "sha": bundle.sha(), "parent_sha": bundle.parent_sha})
        self._write_chain(chain)
        return path

    def load(self, version: str) -> PolicyBundle:
        path = self._bundle_path(version)
        if not path.exists():
            raise KeyError(f"bundle 版本不存在：{version!r}")
        return bundle_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def head(self) -> PolicyBundle:
        chain = self._chain()
        if not chain["head"]:
            raise KeyError("bundle 仓为空（无 HEAD）")
        return self.load(chain["head"])

    def rollback(self, to_version: str, *, reason: str = "") -> PolicyBundle:
        target = self.load(to_version)                        # 不存在则 fail-loud
        chain = self._chain()
        chain["events"].append({"event": "rollback", "from": chain["head"],
                                "to": to_version, "reason": reason})
        chain["head"] = to_version
        self._write_chain(chain)
        return target
