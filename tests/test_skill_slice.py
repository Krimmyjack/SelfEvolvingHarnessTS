"""tests/test_skill_slice.py — LLM-Skill 切片守卫：registry 完整性 / 编译确定性 /
DataView 泄漏边界 / composer 纯函数（无 LLM 调用）行为。"""
import numpy as np
import pytest

from SelfEvolvingHarnessTS.e32_policy import PRUNED_POOL_CORE
from SelfEvolvingHarnessTS.policy.dataview import build_block_views
from SelfEvolvingHarnessTS.policy.skill_composer import (_bins_of, _valid_decision,
                                                         compile_block_policy)
from SelfEvolvingHarnessTS.policy.skills import (SKILLS_V1, action_to_skill, compile_skill,
                                                 skills_sha)
from SelfEvolvingHarnessTS.run_variance_decomp import CUT


def test_registry_bijective_over_pool():
    """7 skill 恰好单射覆盖 10 动作（无新算子、无重复、无遗漏）。"""
    covered = [aid for s in SKILLS_V1.values() for aid in s.actions.values()]
    assert sorted(covered) == sorted(PRUNED_POOL_CORE)
    for a in PRUNED_POOL_CORE:
        assert compile_skill(*action_to_skill(a)) == a       # 往返恒等（B 臂守卫的数学）
    assert len(skills_sha()) == 16


def test_compile_snap_and_invalid():
    assert compile_skill("median_smooth", 25) == "f0_median_w25"
    assert compile_skill("median_smooth", 11) == "f0_median_w9"     # snap 最近剂量
    assert compile_skill("median_smooth", None) == "v_median"       # 缺参 → 最轻剂量
    assert compile_skill("stl_deseason", 99) == "v_stl"             # 无参 skill 忽略参数
    assert compile_skill("no_such_skill", 5) is None


def test_decision_validation_and_compile():
    spec = {"default": {"skill": "median_smooth", "param": 5},
            "overrides": [{"when": {"snr": "low", "miss": "some"},
                           "skill": "identity", "param": None},
                          {"when": {"snr": "BAD"}, "skill": "identity"},   # 非法 when → 丢弃
                          {"when": {"snr": "high"}, "skill": "nope"}],     # 非法 skill → 丢弃
            "rationale": "x"}
    d = _valid_decision(spec)
    assert d is not None and len(d["overrides"]) == 1
    rows = [{"uid": "u1", "cell": "forecast|snrLow|miss"},
            {"uid": "u2", "cell": "forecast|snrHigh|full"}]
    pol = compile_block_policy(d, rows)
    assert pol == {"u1": "v_none", "u2": "v_median"}
    assert _valid_decision({"overrides": []}) is None               # 缺 default
    assert _valid_decision({"default": {"skill": "nope"}}) is None
    assert compile_block_policy(None, rows) is None                 # 失败 → 回退语义在调用方


def test_bins_of():
    assert _bins_of("forecast|snrLow|miss") == {"snr": "low", "miss": "some"}
    assert _bins_of("forecast|snrHigh|full") == {"snr": "high", "miss": "none"}


def test_dataview_history_only_guard():
    """view 构造只接受判官口径观测史；rows 中 L_test 不被读取。"""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(8):
        rows.append(dict(uid=f"S2:S_x:n_hi_full:{i}", snr=float(10 - i), miss_rate=0.0,
                         cell="forecast|snrHigh|full",
                         X_p=[24.0, 0.3, 0.5, 0.8, -3.0, 0.6, 0.1, 0.02]))
    from SelfEvolvingHarnessTS.policy.dataview import _rep_uids
    hist = {u: rng.normal(0, 1, CUT) for u in _rep_uids(rows)}
    views = build_block_views(rows, hist, ["v_median"] * 8, abstains=[])
    assert set(views) == {"structure", "mask", "skills", "policy",
                          "window", "period", "decomp"}
    with pytest.raises(AssertionError):
        bad = dict(hist)
        k = next(iter(bad))
        bad[k] = rng.normal(0, 1, 512)                       # 含未来长度 → 拒绝
        build_block_views(rows, bad, ["v_median"] * 8, abstains=[])
