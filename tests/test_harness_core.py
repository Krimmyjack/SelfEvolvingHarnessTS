"""Phase 0 验证：EditPatch 契约 + editable_surfaces 校验链 + HarnessState apply/snapshot/replay
+ conditioning.key。

运行：  python -m SelfEvolvingHarnessTS.tests.test_harness_core   （cwd=Agent）
或 pytest：  python -m pytest SelfEvolvingHarnessTS/tests/test_harness_core.py -q
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.harness import (
    HarnessState, EditPatch, Manifest, EditRejected,
    PipelineTemplate, StageDef, StrengthSignatureStats, EvaluatorSpec, GateConfig,
)
from SelfEvolvingHarnessTS.conditioning import build_conditioning_key, struct_feats, STRUCT_FEAT_NAMES


def _mf(fid="f001"):
    return Manifest(target_failure_id=fid, target_failure_desc="t", expected_effect="e",
                    ablation_hint="a", regression_risk="r")


# ── 1. 冷启动 ────────────────────────────────────────────────────────────
def test_from_minimal():
    h = HarnessState.from_minimal()
    assert h.version == 0 and h.patch_log == []
    assert h.l2.active_operators["outlier_iqr"] is True
    assert "forecast" in h.l4.proxy_evaluators
    assert h.l4.grounded_evaluators["forecast"].model == "lstm_forecast"


# ── 2. leaf set（禁用一个算子）─────────────────────────────────────────────
def test_leaf_set_active_operator():
    h = HarnessState.from_minimal()
    p = EditPatch("L2", "set", "l2.active_operators.outlier_iqr", False, _mf())
    res = h.apply_edit(p)
    assert res.ok and res.resolved_scope == "global"
    assert h.l2.active_operators["outlier_iqr"] is False
    assert len(h.patch_log) == 1


# ── 3. ref 完整性：未知算子被拒（在昂贵 grounded 之前机械杀掉）──────────────
def test_ref_integrity_unknown_operator():
    h = HarnessState.from_minimal()
    p = EditPatch("L2", "set", "l2.active_operators.no_such_op", True, _mf())
    res = h.validate(p)
    assert not res.ok and "unknown operators" in res.reason


# ── 4. writer 权限：step 不可碰受保护面，consolidator 可 ─────────────────────
def test_writer_permission_protected():
    h = HarnessState.from_minimal()
    sig = StrengthSignatureStats(signature_id="s1", cell_id="P2/forecast", win_margin=0.2)
    step_patch = EditPatch("L3", "set", "l3.strength_signatures::s1", sig, _mf(),
                           source_type="strength", writer="step")
    assert not h.validate(step_patch).ok
    cons_patch = EditPatch("L3", "set", "l3.strength_signatures::s1", sig, _mf(),
                           source_type="strength", writer="consolidator")
    res = h.apply_edit(cons_patch)
    assert res.ok and h.l3.strength_signatures["s1"].win_margin == 0.2


# ── 5. named_object：scope 动态解析 + 模板内算子 ref 完整性 ──────────────────
def test_named_object_scope_and_refs():
    h = HarnessState.from_minimal()
    good = PipelineTemplate(
        name="forecast_default",
        applies_to={"task_type": "forecast", "pattern_conditions": None},
        stages=[StageDef("denoise", preferred_ops=["denoise_savgol"]),
                StageDef("shape", preferred_ops=["lag_features"], banned_ops=["znorm"])])
    p = EditPatch("L2", "set", "l2.task_templates::forecast_default", good, _mf())
    res = h.apply_edit(p)
    assert res.ok and res.resolved_scope == "global"        # pattern_conditions=None → global
    assert "forecast_default" in h.l2.task_templates

    cell_tmpl = PipelineTemplate(
        name="forecast_lowsnr",
        applies_to={"task_type": "forecast", "pattern_conditions": {"SNR": "low"}},
        stages=[StageDef("denoise", preferred_ops=["denoise_median"])])
    res2 = h.validate(EditPatch("L2", "set", "l2.task_templates::forecast_lowsnr", cell_tmpl, _mf()))
    assert res2.ok and res2.resolved_scope == "cell"        # 有 pattern_conditions → cell

    bad = PipelineTemplate(name="bad", applies_to={"task_type": "forecast"},
                           stages=[StageDef("denoise", banned_ops=["ghost_op"])])
    res3 = h.validate(EditPatch("L2", "set", "l2.task_templates::bad", bad, _mf()))
    assert not res3.ok and "unknown operators" in res3.reason


# ── 6. list_scalar add/remove（幂等）────────────────────────────────────────
def test_list_scalar():
    h = HarnessState.from_minimal()
    n0 = len(h.l1.constraints)
    rule = "Do not apply global smoothing for anomaly tasks"
    h.apply_edit(EditPatch("L1", "add", "l1.constraints", rule, _mf()))
    assert h.l1.constraints.count(rule) == 1
    h.apply_edit(EditPatch("L1", "add", "l1.constraints", rule, _mf()))   # 幂等
    assert h.l1.constraints.count(rule) == 1 and len(h.l1.constraints) == n0 + 1
    h.apply_edit(EditPatch("L1", "remove", "l1.constraints", rule, _mf()))
    assert rule not in h.l1.constraints


# ── 7. snapshot / restore ──────────────────────────────────────────────────
def test_snapshot_restore():
    h = HarnessState.from_minimal()
    h.apply_edit(EditPatch("L2", "set", "l2.active_operators.winsorize", False, _mf()))
    snap = h.snapshot()
    h.apply_edit(EditPatch("L2", "set", "l2.active_operators.znorm", False, _mf()))
    h.bump_version()
    assert h.l2.active_operators["znorm"] is False and h.version == 1
    h.restore(snap)
    assert h.l2.active_operators["znorm"] is True            # 回滚
    assert h.l2.active_operators["winsorize"] is False       # 快照时的改动保留
    assert h.version == 0 and len(h.patch_log) == 1


# ── 8. replay 等价性 ───────────────────────────────────────────────────────
def test_replay_equivalence():
    h = HarnessState.from_minimal()
    patches = [
        EditPatch("L2", "set", "l2.active_operators.outlier_iqr", False, _mf()),
        EditPatch("L1", "add", "l1.constraints", "C-new", _mf()),
        EditPatch("L4", "set", "l4.gate_config.blowup_sigma", 8.0, _mf()),
    ]
    for p in patches:
        h.apply_edit(p)
    h2 = HarnessState.replay(patches)
    assert h2.l2.active_operators["outlier_iqr"] is False
    assert "C-new" in h2.l1.constraints
    assert h2.l4.gate_config.blowup_sigma == 8.0
    assert h.to_dict()["layers"] == h2.to_dict()["layers"]


# ── 9. 越界 / 形态错误一律被拒 ──────────────────────────────────────────────
def test_rejections():
    h = HarnessState.from_minimal()
    # 只读基础设施
    assert not h.validate(EditPatch("L1", "set", "l1.system_prompt.x", "y", _mf())).ok
    # op × addressing：add 到 leaf 非法
    assert not h.validate(EditPatch("L2", "add", "l2.active_operators.znorm", True, _mf())).ok
    # 位置索引深路径被禁
    r = h.validate(EditPatch("L2", "set", "l2.task_templates.forecast[0].banned_ops", [], _mf()))
    assert not r.ok
    # 类型不符：blowup_sigma 期望 float
    assert not h.validate(EditPatch("L4", "set", "l4.gate_config.blowup_sigma", "high", _mf())).ok
    # edited_layer 与 path 不一致
    assert not h.validate(EditPatch("L1", "set", "l2.active_operators.znorm", False, _mf())).ok
    # dataclass 属性不可 remove
    assert not h.validate(EditPatch("L4", "remove", "l4.gate_config.blowup_sigma", None, _mf())).ok


# ── 9b. from_dict 忽略未知键（LLM 常塞 reasoning/explanation；不应丢候选）──
def test_from_dict_ignores_unknown_keys():
    raw = {"edited_layer": "L2", "op": "set", "path": "l2.active_operators.znorm", "value": True,
           "manifest": {"target_failure_id": "f", "extra_in_manifest": "x"},
           "reasoning": "because outliers", "explanation": "...", "proposal_rank": 2}
    p = EditPatch.from_dict(raw)
    assert p.edited_layer == "L2" and p.value is True and p.proposal_rank == 2
    assert p.manifest.target_failure_id == "f"


# ── 10. (f) 端到端示例（plan.md §3.2(f)）────────────────────────────────────
def test_plan_md_f_example():
    h = HarnessState.from_minimal()
    tmpl = PipelineTemplate(
        name="forecast_default",
        applies_to={"task_type": "forecast", "pattern_conditions": None},
        stages=[StageDef("denoise", preferred_ops=["denoise_savgol"]),
                StageDef("shape", preferred_ops=["lag_features"], banned_ops=[])])  # 去掉 standardize
    patch = EditPatch(
        edited_layer="L2", op="set", path="l2.task_templates::forecast_default", value=tmpl,
        manifest=Manifest("quality_regression_P2_forecast_001",
                          "standardize 把 val_loss 推到 2.3 vs baseline 2.0",
                          "NRMSE 方向性回落向 baseline 2.0",
                          "单独恢复 standardize 阶段，确认 val_loss 回退",
                          "P3/forecast 若依赖标准化其 NRMSE 可能升 → held-out(b) 兜"),
        source_type="failure", cell_id="P2/forecast", harness_ver=3)
    res = h.apply_edit(patch)
    assert res.ok and res.resolved_scope == "global"
    # JSON round-trip（审计日志）
    d = patch.to_dict()
    assert EditPatch.from_dict(d).path == patch.path


# ── 11. conditioning.key ───────────────────────────────────────────────────
def test_conditioning_key():
    rng = np.random.default_rng(0)
    t = np.arange(240)
    series = (np.sin(2 * np.pi * t / 24) + 0.01 * t + rng.normal(0, 0.3, t.size)).astype(float)
    series[[50, 80, 130, 170, 200]] = 20.0   # 注入离群簇（density > 1% 阈值）
    series[100:105] = np.nan                 # 注入缺失

    feats = struct_feats(series)
    assert set(feats) == set(STRUCT_FEAT_NAMES)
    assert all(np.isfinite(v) for v in feats.values())
    assert feats["missing_rate"] > 0.0 and feats["outlier_density"] > 0.0
    assert 20.0 <= feats["period"] <= 28.0          # 主周期 ≈ 24
    assert feats["trend_strength"] >= 0.0

    key = build_conditioning_key(series, "forecast")
    qp = key["pattern"]["quality_profile"]
    assert qp["problem_types"]["has_missing"] and qp["problem_types"]["has_outlier"]
    assert key["task"]["type"] == "forecast"
    assert 0.0 <= qp["urgency"] <= 1.0


# ── runner ─────────────────────────────────────────────────────────────────
def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
