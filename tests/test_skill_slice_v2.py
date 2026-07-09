"""tests/test_skill_slice_v2.py — 切片 v2 守卫：featurizer 泄漏边界/维度、
可靠性标注触发逻辑、verify 协议强制两段（mock llm，无网络）。"""
import numpy as np
import pytest

from SelfEvolvingHarnessTS.policy.dataview import (build_block_views_v2, featurize_uid_v2,
                                                   _rep_uids)
from SelfEvolvingHarnessTS.policy.skill_composer import decide_block_v2
from SelfEvolvingHarnessTS.run_variance_decomp import CUT


def _rows(ss: float):
    return [dict(uid=f"S2:S_x:n_hi_full:{i}", snr=float(10 - i), miss_rate=0.0,
                 cell="forecast|snrHigh|full",
                 X_p=[0.0, 0.8, ss, 0.8, -3.0, 0.6, 0.1, 0.02]) for i in range(6)]


def _seasonal_hist():
    t = np.arange(CUT)
    return np.sin(2 * np.pi * t / 24) + 0.5 * t / CUT


def test_featurizer_dims_and_leakage_guard():
    r = _rows(0.0)[0]
    h = _seasonal_hist()
    x_d, x_p = featurize_uid_v2(r, h)
    assert x_d.shape == (2,) and x_p.shape == (17,)
    x_d2, x_p2 = featurize_uid_v2(r, h)
    assert np.array_equal(x_p, x_p2)                        # 确定性
    with pytest.raises(AssertionError):
        featurize_uid_v2(r, np.zeros(512))                  # 含未来长度 → 拒绝


def test_reliability_annotation_on_conflict():
    """P0 季节读数≈0 而 robust 检出周期 → structure 追加 [低可靠] 标注；一致时不加。"""
    rows = _rows(ss=0.0)
    hist = {u: _seasonal_hist() for u in _rep_uids(rows)}
    v = build_block_views_v2(rows, hist, ["v_median"] * 6, abstains=[])
    assert "[低可靠]" in v["structure"]
    rows2 = _rows(ss=0.6)                                   # P0 自己就报告了季节 → 无冲突
    v2 = build_block_views_v2(rows2, hist, ["v_median"] * 6, abstains=[])
    assert "[低可靠]" not in v2["structure"]


def _mk_views():
    rows = _rows(0.0)
    hist = {u: _seasonal_hist() for u in _rep_uids(rows)}
    return build_block_views_v2(rows, hist, ["v_median"] * 6, abstains=[])


_DEC = '{"default": {"skill": "stl_deseason", "param": null}, "overrides": [], "rationale": "x"}'


def test_v2_mode_core_includes_robust_evidence():
    calls = []

    def mock(sys, user, nonce=0):
        calls.append(user)
        return _DEC
    d = decide_block_v2(_mk_views(), mock, "T:h0", mode="v2")
    assert d["n_calls"] == 1 and d["decision"] is not None
    assert "[period]" in calls[0] and "[decomp]" in calls[0]    # robust 证据在 core


def test_verify_mode_forces_two_stages():
    calls = []

    def mock_violating(sys, user, nonce=0):
        calls.append(user)
        return _DEC                                          # stage1 就给决策 = 违规
    d = decide_block_v2(_mk_views(), mock_violating, "T:h0", mode="verify")
    assert d["n_calls"] == 2 and d["violation"] is True
    assert set(("period", "decomp")).issubset(d["views_used"])   # 强制附 robust 证据
    assert "[period]" in calls[1] and "[decomp]" in calls[1]
    assert d["decision"] is not None

    def mock_good(sys, user, nonce=0):
        if "Do NOT decide yet" in user:
            return '{"request_views": ["period"]}'
        return _DEC
    d2 = decide_block_v2(_mk_views(), mock_good, "T:h0", mode="verify")
    assert d2["violation"] is False and d2["n_calls"] == 2
