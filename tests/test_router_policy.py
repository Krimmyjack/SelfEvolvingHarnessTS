"""tests/test_router_policy.py — 2.0-③ RouterPolicy 包装等价性（锚定 P0，Component Plan v1.1b）。

主判据："harness 原生（批式冻结臂）与外挂包装（RouterPolicy 逐输入）在 dev 全 uid 上
picks 逐一一致（含 abstain mask）"。特征来自 dev records（= E-3.2/confirmatory 训练分布本身），
所以这是对**包装层数学**的等价性测试；PatternSpec 提取端一致性由 test_policy_contract 守。

依赖冻结产物 frozen_arms.joblib + dev records（repo 内已有）；缺失则 skip（新 clone 不炸）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parent.parent
RECORDS = PKG / "results" / "E3_2" / "records_primary_no_Sar.jsonl"
ARMS = PKG / "results" / "E3_2_confirmatory" / "frozen_arms.joblib"

pytestmark = pytest.mark.skipif(
    not (RECORDS.exists() and ARMS.exists()),
    reason="冻结产物/dev records 缺失（新 clone）——等价性测试需要 Stage-1 冻结资产")


@pytest.fixture(scope="module")
def frozen():
    from SelfEvolvingHarnessTS.confirmatory_freeze import (load_dev_records, load_frozen_arms,
                                                           policy_data_from_records)
    blob = load_frozen_arms(ARMS)
    records = load_dev_records("primary_no_Sar")
    data = policy_data_from_records(records, list(blob["actions"]))
    return blob, records, data


def _key_from_record(r: dict) -> dict:
    """dev record 的特征 → conditioning_key shim（与训练侧同映射的逆向拼装）。"""
    from SelfEvolvingHarnessTS.e32_policy import D_FEATS, P_FEATS
    struct = {"SNR": r["snr"], "missing_rate": r["miss_rate"]}
    struct.update({k: v for k, v in zip(P_FEATS, r["X_p"])})
    assert set(D_FEATS) <= set(struct)
    return {"pattern": {"struct_feats": struct}, "task": {"type": "forecast"},
            "cell_id": r["cell"]}


# ── 主判据：dev 全 uid picks 逐一一致 ───────────────────────────────────────
@pytest.mark.parametrize("arm_name", ["dp_abstain", "global", "d_lookup"])
def test_wrapper_picks_match_direct_all_dev_uids(frozen, arm_name):
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy, action_menu_v1
    blob, records, data = frozen
    direct_picks, direct_abst = blob["arms"][arm_name].picks(data, np.arange(data.n))
    router = FrozenArmRouterPolicy.load_frozen(arm_name, path=ARMS)
    menu = action_menu_v1()
    for i, r in enumerate(records):
        d = router.predict(_key_from_record(r), menu)
        assert d.action_id == blob["actions"][int(direct_picks[i])], \
            f"uid={r['uid']} 包装 pick {d.action_id} ≠ 直调 {blob['actions'][int(direct_picks[i])]}"
        assert d.abstained == bool(direct_abst[i]), f"uid={r['uid']} abstain mask 不一致"


# ── provenance（第 (d) 层）──────────────────────────────────────────────────
def test_decision_provenance_complete(frozen):
    import hashlib
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy, action_menu_v1
    _blob, records, _data = frozen
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain", path=ARMS)
    menu = action_menu_v1()
    d = router.predict(_key_from_record(records[0]), menu)
    prov = d.provenance
    assert prov["router_artifact_sha"] == hashlib.sha256(ARMS.read_bytes()).hexdigest()
    assert prov["arm"] == "dp_abstain"
    assert prov["pattern_spec"]["version"] == "P0"
    assert len(prov["pattern_spec"]["code_sha256"]) == 64    # 提取器闭包活值（key.py+period.py）
    assert prov["action_menu"] == {"version": menu.version, "sha256": menu.sha256}
    # 评审第二十四轮：版本核验结果必须落每条决策（"放行须记录"不能是空话）
    rc = prov["runtime_check"]
    assert rc["checked"] is True and rc["allowed_mismatch"] is False and rc["mismatch"] == {}
    assert rc["recorded"]["sklearn"] is not None or rc["recorded"]["numpy"] is not None
    assert d.fallback_action == "v_median"


def test_sha_guard_matches_freeze_json(frozen):
    """confirmatory_freeze.json 在场时 load_frozen 的守卫①自动核验必须通过。"""
    from SelfEvolvingHarnessTS.confirmatory_freeze import FREEZE_PATH
    if not FREEZE_PATH.exists():
        pytest.skip("无 confirmatory_freeze.json")
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain")     # verify_sha="auto"
    sha = json.loads(FREEZE_PATH.read_text("utf-8"))["router"]["sha256"]
    assert router.artifact_sha == sha


# ── 菜单兼容性守卫 ──────────────────────────────────────────────────────────
def test_menu_missing_actions_rejected(frozen):
    from SelfEvolvingHarnessTS.policy import ActionMenu, FrozenArmRouterPolicy, action_menu_v1
    _blob, records, _data = frozen
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain", path=ARMS)
    full = action_menu_v1()
    partial = ActionMenu("v1", [s for aid, s in full.actions.items() if aid != "v_median"])
    with pytest.raises(ValueError, match="缺冻结臂动作"):
        router.predict(_key_from_record(records[0]), partial)


def test_model_menu_not_implemented(frozen):
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy, action_menu_v1
    _blob, records, _data = frozen
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain", path=ARMS)
    with pytest.raises(NotImplementedError):
        router.predict(_key_from_record(records[0]), action_menu_v1(), model_menu=["dlinear"])


# ── Step1.1-①：task scope 拒绝（冻结臂标签=forecast，其他 task 无效用语义）──
def test_task_scope_rejects_non_forecast(frozen):
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy, action_menu_v1
    _blob, records, _data = frozen
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain", path=ARMS)
    key = _key_from_record(records[0])
    key["task"] = {"type": "anomaly_detection"}
    with pytest.raises(ValueError, match="task_scope"):
        router.predict(key, action_menu_v1())


# ── Step1.1-④：运行时版本核验 + OOD support 标记 ────────────────────────────
def test_runtime_version_mismatch_fail_loud():
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy
    bad_blob = {"sklearn_version": "0.0.1", "numpy_version": "0.0.1"}
    with pytest.raises(RuntimeError, match="版本不匹配"):
        FrozenArmRouterPolicy._check_runtime_versions(bad_blob, allow=False)
    rc = FrozenArmRouterPolicy._check_runtime_versions(bad_blob, allow=True)  # 显式放行不抛…
    assert rc["allowed_mismatch"] is True and set(rc["mismatch"]) == {"sklearn", "numpy"}  # …但放行可审计
    rc0 = FrozenArmRouterPolicy._check_runtime_versions({}, allow=False)      # 无记录 → 不判
    assert rc0["mismatch"] == {} and rc0["allowed_mismatch"] is False


def test_support_marker_records_not_blocks(frozen):
    from SelfEvolvingHarnessTS.policy import FrozenArmRouterPolicy, action_menu_v1
    _blob, records, _data = frozen
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain", path=ARMS)
    menu = action_menu_v1()
    # dev 训练点本身 → 距离 0，必在支持域内
    d_in = router.predict(_key_from_record(records[0]), menu)
    sup = d_in.provenance["support"]
    assert sup["available"] and not sup["out_of_support"] and sup["distance"] <= 1e-9
    assert sup["n_train"] > 0 and "未绑定 artifact" in sup["source"]  # 挂账显式：S1/部署前须入 PolicyArtifact
    # 荒诞输入 → 标注 out_of_support，但**不拦截**（仍返回合法动作）
    far = _key_from_record(records[0])
    far["pattern"]["struct_feats"] = {k: 1e6 for k in far["pattern"]["struct_feats"]}
    d_out = router.predict(far, menu)
    assert d_out.provenance["support"]["out_of_support"] is True
    assert d_out.action_id in menu


# ── 端到端：Router 选中的 ID == 线上执行的动作 ───────────────────────────────
def test_routed_process_executes_selected_action(frozen):
    from SelfEvolvingHarnessTS.policy import (ActionCompiler, FrozenArmRouterPolicy,
                                              action_menu_v1, routed_process)
    router = FrozenArmRouterPolicy.load_frozen("dp_abstain", path=ARMS)
    menu = action_menu_v1()
    rng = np.random.default_rng(3)
    n = 192
    x = np.sin(2 * np.pi * np.arange(n) / 24) + 0.3 * rng.standard_normal(n)
    x[50:55] = np.nan
    decision, record, artifact = routed_process(x, "forecast", router, menu)
    assert decision.action_id in menu
    # 执行契约核心：record 里的程序 = 选中 ActionSpec 编译出的模板程序（非别的动作）
    assert record.program["note"] == f"tmpl:{decision.action_id}"
    spec_ops = [st.op for st in menu.actions[decision.action_id].steps]
    assert [s["op"] for s in record.program["steps"]] == spec_ops
    assert artifact.shape == x.shape and np.all(np.isfinite(artifact))
