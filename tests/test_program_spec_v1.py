"""P0 契约测试：ProgramSpec grammar v1（policy/program_edit.py 扩展，v0 冻结不动）。

v1 相对 v0（B1b 冻结面）的质变（Final_Plan_CodeAgentFirst_2026-07-09 §P0）：
  task_type 显式（不再 forecast 写死）+ pattern_guard + invariants（保长/修改率预算）
  + fallback + risk_budget_beta。v0 的 ProgramSpec/validate/sha 语义 bit 级不变（B1b 重放依赖）。
"""
import numpy as np
import pytest

from SelfEvolvingHarnessTS.policy.program_edit import (
    ProgramSpec,
    ProgramSpecV1,
    check_execution_invariants,
    guard_matches,
    to_action_spec_v1,
    validate,
    validate_v1,
)


def _v1(**kw):
    base = dict(
        steps=(("impute_linear", ()),),
        scope=("forecast|snrLow|full",),
        task_type="forecast",
        pattern_guard=(("seasonal_strength", ">=", 0.5),),
        risk_budget_beta=0.3,
        fallback="v_raw_identity",
    )
    base.update(kw)
    return ProgramSpecV1(**base)


# ── v0 冻结守卫 ─────────────────────────────────────────────────────────────

def test_v0_spec_unchanged_by_v1_addition():
    spec = ProgramSpec(steps=(("impute_linear", ()),), scope=("cellA",))
    ok, why = validate(spec)
    assert ok, why
    assert spec.action_id.startswith("prog_")


# ── v1 结构校验 ─────────────────────────────────────────────────────────────

def test_valid_v1_spec_passes():
    ok, why = validate_v1(_v1())
    assert ok, why


def test_beta_range_enforced():
    ok, why = validate_v1(_v1(risk_budget_beta=1.5))
    assert not ok and "risk_budget_beta" in why
    ok2, why2 = validate_v1(_v1(risk_budget_beta=-0.1))
    assert not ok2


def test_explicit_budget_cannot_exceed_beta():
    ok, why = validate_v1(_v1(risk_budget_beta=0.2, max_modified_fraction=0.5))
    assert not ok and "max_modified_fraction" in why
    ok2, why2 = validate_v1(_v1(risk_budget_beta=0.2, max_modified_fraction=0.1))
    assert ok2, why2


def test_guard_feature_whitelist():
    ok, why = validate_v1(_v1(pattern_guard=(("volatility_of_vibes", ">", 0.0),)))
    assert not ok and "pattern_guard" in why


def test_guard_comparator_whitelist():
    ok, why = validate_v1(_v1(pattern_guard=(("snr", "!=", 0.0),)))
    assert not ok


def test_fallback_must_be_known_action():
    ok, why = validate_v1(_v1(fallback="v_mystery"))
    assert not ok and "fallback" in why
    ok2, _ = validate_v1(_v1(fallback="v_none"))
    assert ok2                                   # menu v1 动作合法
    ok3, _ = validate_v1(_v1(fallback="v_impute_linear"))
    assert ok3                                   # canonical 语义名合法


def test_task_aware_operator_contract_blocks_anomaly_smoothing():
    spec = _v1(
        task_type="anomaly_detection",
        steps=(("impute_linear", ()), ("denoise_median", (("window", 9),))),
        pattern_guard=(),
    )
    ok, why = validate_v1(spec)
    assert not ok and "anomaly_detection" in why


def test_anomaly_impute_only_chain_valid():
    spec = _v1(task_type="anomaly_detection", steps=(("impute_linear", ()),), pattern_guard=())
    ok, why = validate_v1(spec)
    assert ok, why


def test_unknown_task_type_rejected():
    ok, why = validate_v1(_v1(task_type="regression"))
    assert not ok and "task_type" in why


def test_v0_mechanical_rules_still_apply_in_v1():
    ok, _ = validate_v1(_v1(steps=(("denoise_median", (("window", 9),)),)))   # 首步须 imputer
    assert not ok
    ok2, _ = validate_v1(_v1(steps=(("impute_linear", ()), ("denoise_median", (("window", 7),)))))
    assert not ok2                                                            # 窗须在剂量网格
    ok3, _ = validate_v1(_v1(scope=()))
    assert not ok3                                                            # scope 纪律


def test_window_must_be_exact_int_from_grid():
    # code-review #1：int(w) 隶属判定会放行 9.0/"9"，真实执行时 medfilt 拒绝 → 烧真实执行预算
    ok, why = validate_v1(_v1(steps=(("impute_linear", ()), ("denoise_median", (("window", 9.0),)))))
    assert not ok
    ok2, _ = validate_v1(_v1(steps=(("impute_linear", ()), ("denoise_median", (("window", "9"),)))))
    assert not ok2
    ok3, _ = validate_v1(_v1(steps=(("impute_linear", ()), ("denoise_median", (("window", True),)))))
    assert not ok3


# ── 身份（SHA）─────────────────────────────────────────────────────────────


def test_sha_normalizes_default_budget_to_resolved():
    # code-review #2：max_modified_fraction=None 与显式等于 β 语义等价，身份必须相同
    a = _v1(risk_budget_beta=0.3)
    b = _v1(risk_budget_beta=0.3, max_modified_fraction=0.3)
    assert a.sha() == b.sha()
    c = _v1(risk_budget_beta=0.3, max_modified_fraction=0.1)
    assert c.sha() != a.sha()

def test_sha_sensitive_to_guard_task_beta_but_chain_sha_stable():
    a = _v1()
    b = _v1(pattern_guard=(("snr", "<=", 0.0),))
    c = _v1(task_type="classification")
    d = _v1(risk_budget_beta=0.7)
    assert len({a.sha(), b.sha(), c.sha(), d.sha()}) == 4
    assert a.chain_sha() == b.chain_sha() == d.chain_sha()
    assert a.action_id.startswith("prog1_")


# ── guard 评估 ──────────────────────────────────────────────────────────────

def test_guard_matches_pattern_summary():
    spec = _v1(pattern_guard=(("seasonal_strength", ">=", 0.5), ("snr", "<", 0.0)))
    pat = {"snr": -3.0, "missing_rate": 0.1, "struct_feats": {"seasonal_strength": 0.8}}
    assert guard_matches(spec, pat)
    pat_bad = {"snr": 5.0, "missing_rate": 0.1, "struct_feats": {"seasonal_strength": 0.8}}
    assert not guard_matches(spec, pat_bad)
    assert guard_matches(_v1(pattern_guard=()), pat)      # 空 guard = 无条件适用


def test_guard_missing_feature_fails_loud():
    spec = _v1(pattern_guard=(("seasonal_strength", ">=", 0.5),))
    with pytest.raises(KeyError):
        guard_matches(spec, {"snr": 1.0, "missing_rate": 0.0, "struct_feats": {}})


def test_guard_none_valued_feature_is_missing_not_typeerror():
    # P5-A 实测：LLM 提出带 snr guard 的程序，summary 里 snr=None → 须按缺特征 fail-loud
    #（KeyError，上游转 pattern_guard_feature_missing），不得炸 TypeError
    spec = _v1(pattern_guard=(("snr", "<", 0.0),))
    with pytest.raises(KeyError):
        guard_matches(spec, {"snr": None, "missing_rate": 0.0, "struct_feats": {}})
    spec2 = _v1(pattern_guard=(("seasonal_strength", ">=", 0.5),))
    with pytest.raises(KeyError):
        guard_matches(spec2, {"snr": 1.0, "missing_rate": 0.0,
                              "struct_feats": {"seasonal_strength": None}})


# ── 执行后不变量 ────────────────────────────────────────────────────────────

def test_invariants_nan_fill_not_counted_as_modification():
    spec = _v1(risk_budget_beta=0.2)
    x_in = np.arange(10.0)
    x_in[3] = np.nan
    x_out = np.arange(10.0)                                # NaN 被补上，观测点原样
    ok, detail = check_execution_invariants(spec, x_in, x_out)
    assert ok, detail
    assert detail["modified_fraction"] == 0.0


def test_invariants_budget_violation_caught():
    spec = _v1(risk_budget_beta=0.2)
    x_in = np.arange(10.0)
    x_out = x_in.copy()
    x_out[:3] = 99.0                                       # 3/10 观测点被改 > 0.2
    ok, detail = check_execution_invariants(spec, x_in, x_out)
    assert not ok
    assert detail["modified_fraction"] > 0.2


def test_invariants_length_change_caught():
    spec = _v1()
    ok, detail = check_execution_invariants(spec, np.arange(10.0), np.arange(9.0))
    assert not ok
    assert "preserve_length" in detail["violations"]


# ── 编译 ────────────────────────────────────────────────────────────────────

def test_to_action_spec_v1_compiles_with_resolved_params():
    spec = _v1(steps=(("impute_linear", ()), ("denoise_median", (("window", 9),))))
    a = to_action_spec_v1(spec)
    assert a.action_id == spec.action_id
    assert [s.op for s in a.steps] == ["impute_linear", "denoise_median"]
    assert a.steps[1].params["window"] == 9
    assert "forecast" in a.task_constraints
    assert a.provenance["grammar"] == "v1"
    assert a.provenance["fallback"] == "v_raw_identity"
