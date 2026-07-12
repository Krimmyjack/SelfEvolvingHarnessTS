"""P4 契约测试：PolicyBundle 持久化 + 版本链 + rollback（append-only、artifact 不可变）。"""
import json

import pytest

from SelfEvolvingHarnessTS.policy.edits import AddRiskRule, apply_edits, bundle_v0
from SelfEvolvingHarnessTS.policy.risk_policy import RiskRule
from SelfEvolvingHarnessTS.slow_path.bundle_store import BundleStore, bundle_from_dict, bundle_to_dict


def _rule():
    return RiskRule(
        rule_id="mined_ban_w25_snrLow",
        when={"cell": {"snr": "low"}, "base_action_in": ["f0_median_w25"]},
        then={"op": "ban", "action": "f0_median_w9"},
        scope="region:cell_snr=low",
        provenance={"source": "proposer:enum", "evidence": "held_in w9 beats w25"},
    )


def _v1():
    v0 = bundle_v0()
    v1, log = apply_edits(v0, [AddRiskRule(_rule())])
    assert log[0]["applied"]
    return v0, v1


def test_bundle_dict_roundtrip_preserves_sha():
    _, v1 = _v1()
    d = bundle_to_dict(v1)
    json.dumps(d, allow_nan=False)
    back = bundle_from_dict(d)
    assert back.sha() == v1.sha()
    assert back.version == v1.version
    assert back.risk.rules[0].rule_id == "mined_ban_w25_snrLow"


def test_store_save_load_head_and_chain(tmp_path):
    v0, v1 = _v1()
    store = BundleStore(tmp_path)
    store.save(v0, meta={"role": "incumbent"})
    assert store.head().sha() == v0.sha()
    store.save(v1, meta={"role": "promoted", "validation": {"held_out_gain": 0.5}})
    assert store.head().sha() == v1.sha()
    chain = json.loads((tmp_path / "chain.json").read_text(encoding="utf-8"))
    assert chain["head"] == v1.version
    assert [e["event"] for e in chain["events"]] == ["save", "save"]
    loaded = store.load(v0.version)
    assert loaded.sha() == v0.sha()


def test_store_rejects_version_overwrite(tmp_path):
    v0, _ = _v1()
    store = BundleStore(tmp_path)
    store.save(v0)
    with pytest.raises(ValueError, match="已存在"):
        store.save(v0)


def test_rollback_moves_head_appends_event_keeps_artifact(tmp_path):
    v0, v1 = _v1()
    store = BundleStore(tmp_path)
    store.save(v0)
    store.save(v1)
    store.rollback(v0.version, reason="demo")
    assert store.head().sha() == v0.sha()
    chain = json.loads((tmp_path / "chain.json").read_text(encoding="utf-8"))
    assert chain["events"][-1]["event"] == "rollback"
    assert chain["events"][-1]["reason"] == "demo"
    # v1 artifact 保留（append-only 审计）
    assert store.load(v1.version).sha() == v1.sha()
