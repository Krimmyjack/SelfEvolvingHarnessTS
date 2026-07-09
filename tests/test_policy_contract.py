"""tests/test_policy_contract.py — 2.0-②/②b 契约层四层等价性测试（Component Plan v1.1b）。

四层（判据速查 2.0-③ 行）：
  (a) action ID：menu v1 覆盖冻结动作池（PRUNED_POOL_CORE 10 + ABLATION_MA 3），
      且与实验单一真源（_VARIANT_SPECS / F0_DOSAGE_GRID）逐一对应；
  (b) 编译 Program：ActionCompiler 产物与实验 harness 变体（fixed_harness_variants /
      f0_variants）经同一 compose 的产物逐 op+params 一致；
  (c) 执行 artifact：两侧全链 fast_path.process 产物 bit 级一致；
  (d) provenance：menu SHA 确定性 + spec 溯源字段齐全。
另守 PatternSpec P0：冻结身份、bit 级一致提取、router_features 与训练侧映射一致。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.conditioning.key import STRUCT_FEAT_NAMES, build_conditioning_key
from SelfEvolvingHarnessTS.e32_policy import ABLATION_MA, D_FEATS, P_FEATS, PRUNED_POOL_CORE
from SelfEvolvingHarnessTS.family0_actions import F0_DOSAGE_GRID, f0_variants
from SelfEvolvingHarnessTS.fast_path.compose import compose
from SelfEvolvingHarnessTS.fast_path.pipeline import process as fast_process
from SelfEvolvingHarnessTS.policy import (ActionCompiler, action_menu_v1, pattern_spec_p0)
from SelfEvolvingHarnessTS.run_main_table import _VARIANT_SPECS, fixed_harness_variants


def _series(n: int = 192) -> np.ndarray:
    rng = np.random.default_rng(7)
    x = (np.sin(2 * np.pi * np.arange(n) / 24) + 0.02 * np.arange(n)
         + 0.15 * rng.standard_normal(n))
    x[17] = 6.0
    x[60:66] = np.nan
    return x


def _key(task: str = "forecast"):
    return build_conditioning_key(_series(), task)


# ══ PatternSpec P0 ══════════════════════════════════════════════════════════
def test_pattern_spec_p0_frozen_identity():
    s = pattern_spec_p0()
    assert s.version == "P0"
    assert s.feature_names == tuple(STRUCT_FEAT_NAMES)      # feature order 是契约
    assert s.d_feats == tuple(D_FEATS) and s.p_feats == tuple(P_FEATS)
    assert s.scaler == "none" and s.confidence_schema is None
    assert "v1" in s.compatible_action_menus
    assert s.config_sha() == pattern_spec_p0().config_sha()  # 语义身份稳定
    # P0 语义身份钉死（评审第二十四轮）：任何人改动 P0 字段（"禁止原地改 P0"）在此炸响。
    # code_sha256/依赖指纹是活值不进 config_sha —— 代码/环境漂移由 protocol/provenance 记账。
    assert s.config_sha() == "e4f10d11128e943a"
    assert len(s.code_sha256) == 64 and s.dependency_fingerprint["numpy"]
    # 闭包指纹覆盖 period.py（A0 后周期实现所在文件）：改 period.py 必须反映到活值
    from SelfEvolvingHarnessTS.policy.pattern_spec import _CODE_CLOSURE
    assert {p.name for p in _CODE_CLOSURE} == {"key.py", "period.py"}


def test_pattern_spec_bit_identical_extraction():
    s = pattern_spec_p0()
    v1, v2 = s.features_vector(_series()), s.features_vector(_series())
    assert np.array_equal(v1, v2)                            # bit 级一致（2.0-② 验收）
    assert v1.shape == (10,) and np.all(np.isfinite(v1))


def test_router_features_match_training_mapping():
    """与 e32_nested._policy_data 同映射：X_d=[SNR, missing_rate]、X_p=P_FEATS 列序、缺键填 0。"""
    s = pattern_spec_p0()
    struct = {k: float(i + 1) for i, k in enumerate(STRUCT_FEAT_NAMES)}
    xd, xp = s.router_features(struct)
    assert np.array_equal(xd, [struct["SNR"], struct["missing_rate"]])
    assert np.array_equal(xp, [struct[k] for k in P_FEATS])
    xd2, _ = s.router_features({})                           # f.get(k, 0.0) 语义
    assert np.array_equal(xd2, [0.0, 0.0])


# ══ (a) action ID 覆盖 ══════════════════════════════════════════════════════
def test_menu_v1_covers_frozen_pool_and_sources():
    from SelfEvolvingHarnessTS.harness.layers import minimal_l2
    defaults = minimal_l2().operator_defaults
    menu = action_menu_v1()
    ids = set(menu.actions)
    assert ids == set(_VARIANT_SPECS) | {n for n, _o, _w in F0_DOSAGE_GRID}
    assert set(PRUNED_POOL_CORE) <= ids and set(ABLATION_MA) <= ids
    # 定义单一真源逐一对应；Step1.1-②：params = 完整 resolved（defaults ⊕ override）
    for name, chain in _VARIANT_SPECS.items():
        steps = menu.actions[name].steps
        assert [st.op for st in steps] == list(chain)
        for st in steps:
            assert dict(st.params) == dict(defaults.get(st.op, {}))
    for name, op, w in F0_DOSAGE_GRID:
        steps = menu.actions[name].steps
        assert [st.op for st in steps] == ["impute_linear", op]
        assert dict(steps[1].params) == {**defaults.get(op, {}), "window": int(w)}
        assert menu.actions[name].provenance["override_params"][1] == {"window": int(w)}


def test_menu_sha_binds_execution_semantics():
    """Step1.1-② 核心：defaults 变 → resolved params 变 → menu SHA 变（语义身份绑定）。"""
    from SelfEvolvingHarnessTS.policy import ActionMenu, ActionSpec, ActionStep
    base = action_menu_v1()
    assert base.meta["params_resolution"].startswith("resolved_full")
    assert len(base.meta["operator_defaults_sha"]) == 16
    # 同一动作集、仅一步参数不同 → SHA 必须不同
    spec = base.actions["v_median"]
    tweaked = ActionSpec(spec.action_id, (spec.steps[0],
                                          ActionStep(spec.steps[1].op, {"window": 999})),
                         spec.task_constraints, None, dict(spec.provenance))
    others = [s for aid, s in base.actions.items() if aid != "v_median"]
    m2 = ActionMenu("v1", others + [tweaked], meta=base.meta)
    assert m2.sha256 != base.sha256


def test_task_constraints_from_registry():
    menu = action_menu_v1()
    assert "anomaly_detection" in menu.actions["v_none"].task_constraints   # 仅插补
    assert "anomaly_detection" not in menu.actions["v_median"].task_constraints
    assert menu.actions["f0_median_w9"].task_constraints == ("forecast", "classification")
    assert all(a.model_constraints is None for a in menu.actions.values())  # P0：待 2.3


# ══ (b) 编译 Program 等价 ═══════════════════════════════════════════════════
def _experiment_harnesses():
    return {**fixed_harness_variants("forecast"), **f0_variants("forecast")}


def test_compiled_program_equals_experiment_variant():
    menu, comp, key = action_menu_v1(), ActionCompiler(), _key()
    exp = _experiment_harnesses()
    for aid, spec in menu.actions.items():
        p_new = comp.to_program(spec, key)
        p_exp = compose(key, exp[aid])
        assert p_new.op_names() == p_exp.op_names(), aid
        assert [s.params for s in p_new.steps] == [s.params for s in p_exp.steps], aid


# ══ (c) 执行 artifact bit 级等价 ═════════════════════════════════════════════
def test_compiled_artifact_bit_identical():
    menu, comp = action_menu_v1(), ActionCompiler()
    exp = _experiment_harnesses()
    x = _series()
    for aid, spec in menu.actions.items():
        h_new = comp.to_harness(spec, "forecast")
        _rec_n, art_new = fast_process(x, "forecast", h_new, store=None)
        _rec_e, art_exp = fast_process(x, "forecast", exp[aid], store=None)
        assert np.array_equal(art_new, art_exp), f"{aid}: artifact 不一致"


# ══ (d) provenance ══════════════════════════════════════════════════════════
def test_menu_sha_deterministic_and_provenance():
    m1, m2 = action_menu_v1(), action_menu_v1()
    assert m1.sha256 == m2.sha256 and len(m1.sha256) == 64
    assert m1.actions["v_median"].provenance["source"] == "run_main_table._VARIANT_SPECS"
    assert m1.actions["f0_ma_w25"].provenance["source"] == "family0_actions.F0_DOSAGE_GRID"
    d = m1.to_dict()
    assert d["version"] == "v1" and len(d["actions"]) == len(m1)


# ══ 契约合成（D6 × ②b × Step1.1-① fail-loud）════════════════════════════════
def test_compiler_rejects_task_violation_fail_loud():
    """Step1.1-①：违反 task_constraints 的编译直接拒绝——不再依赖 D6 静默滤步
    （那会造成 'Router 选 v_median、anomaly 实际执行≈v_none' 的语义漂移）。"""
    menu, comp = action_menu_v1(), ActionCompiler()
    key = _key("anomaly_detection")
    with pytest.raises(ValueError, match="fail-loud"):
        comp.to_program(menu.actions["v_median"], key)
    with pytest.raises(ValueError, match="fail-loud"):
        comp.to_harness(menu.actions["f0_median_w9"], "anomaly_detection")
    # 契约允许的动作（v_none=仅插补）在 anomaly 下正常编译
    prog = comp.to_program(menu.actions["v_none"], key)
    assert prog.op_names() == ["impute_linear"]


def test_menu_rejects_duplicate_ids():
    from SelfEvolvingHarnessTS.policy import ActionMenu
    menu = action_menu_v1()
    spec = menu.actions["v_none"]
    with pytest.raises(ValueError):
        ActionMenu("dup", [spec, spec])
