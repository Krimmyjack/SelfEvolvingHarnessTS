"""tests/test_updater3.py — updater v3 响应签名守卫（prereg_updater3.md §3 G-B/G-C/G-D）。

G-A（结构重放）在 runner 内执行（需全量数据），本文件覆盖 API 级守卫：
泄漏边界 / 确定性 / 探针预算硬帽 / 失败保守语义 / 支持域配方。
"""
import numpy as np
import pytest

from SelfEvolvingHarnessTS.evaluators.frozen_probe import FrozenProbe
from SelfEvolvingHarnessTS.run_e32 import _variant_map
from SelfEvolvingHarnessTS.run_updater3 import (COVERAGE_FLOOR, CUT_SIG, MIN_PF_OBS, PROBES,
                                                SIG_DIM, build_support_sig, in_support_sig,
                                                probe_signature)
from SelfEvolvingHarnessTS.run_variance_decomp import CUT
from SelfEvolvingHarnessTS.s2_corpus import make_series


@pytest.fixture(scope="module")
def probe_env():
    return _variant_map(list(PROBES)), FrozenProbe()


def test_probe_budget_hard_cap():
    """G-D：4 探针动作 × 3 维签名 × 切点=CUT−48，硬帽钉死。"""
    assert len(PROBES) == 4 and PROBES[0] == "v_none"
    assert SIG_DIM == 3
    assert CUT_SIG == CUT - 48


def test_leakage_api_rejects_longer_input(probe_env):
    """G-B：签名只接受判官口径观测史（CUT=464）——含未来的全长序列被 assert 拒绝。"""
    variants, fp = probe_env
    with pytest.raises(AssertionError):
        probe_signature(np.zeros(512), variants, fp)
    with pytest.raises(AssertionError):
        probe_signature(np.zeros(CUT + 1), variants, fp)


def test_signature_deterministic_and_future_independent(probe_env):
    """G-C：同 uid 两次 bit 级一致；G-B：污染 rs.future / clean 不改变签名（物理不可达）。"""
    variants, fp = probe_env
    rs = make_series("S_season", "n_lo_rand_hi", 0)
    s1 = probe_signature(rs.history, variants, fp)
    rs.future[:] = 999.0                      # 污染真未来
    rs.clean[:] = -999.0                      # 污染 clean（labels 的原料）
    s2 = probe_signature(rs.history, variants, fp)
    assert s1 is not None and len(s1) == SIG_DIM
    assert s1 == s2


def test_pseudo_future_nan_conservative(probe_env):
    """伪未来有效观测 < MIN_PF_OBS → 签名无效（None，保守 out-of-support）。"""
    variants, fp = probe_env
    rs = make_series("S_trend", "n_hi_full", 1)
    hist = rs.history.copy()
    hist[CUT_SIG:] = np.nan
    assert probe_signature(hist, variants, fp) is None
    assert MIN_PF_OBS == 8


def test_invalid_signature_out_of_support():
    """签名 None / 支持域不可建 → in-support 恒 False（回退 frozen）。"""
    sig = {"a": [0.1, 0.2, 0.3], "b": [0.11, 0.21, 0.29], "c": None}
    rows = [{"uid": "a"}, {"uid": "b"}]
    sup = build_support_sig(rows, sig)
    assert sup is not None
    assert in_support_sig(sup, {"uid": "c"}, sig) is False          # 无效签名
    assert in_support_sig(None, {"uid": "a"}, sig) is False         # 支持域不可建
    assert build_support_sig([{"uid": "c"}], sig) is None           # 有效行 <2


def test_support_sig_same_recipe_as_v2():
    """配方同 v2（z-score kNN LOO p95）：拟合成员 in、远点 out。"""
    rng = np.random.default_rng(0)
    sig = {f"u{i}": list(rng.normal(0, 1, 3)) for i in range(40)}
    rows = [{"uid": f"u{i}"} for i in range(40)]
    sup = build_support_sig(rows, sig)
    n_in = sum(in_support_sig(sup, r, sig) for r in rows)
    assert n_in >= 36                                               # LOO p95 → ~95% 成员 in
    sig["far"] = [50.0, -50.0, 50.0]
    assert in_support_sig(sup, {"uid": "far"}, sig) is False
    assert 0.0 < COVERAGE_FLOOR < 1.0
