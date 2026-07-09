"""Phase 2b 验证：cell-scoped task_template 机制（compose 应用 + 物理 ban + proposer 提议 + validator Pareto 安全）。

证明 C1 specialization 的落点：pattern-conditioned 模板只改本 cell 的 compose，对其他 cell 无影响
→ 天然 Pareto 安全 → 现有 validator 可接受。

运行：  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.tests.test_templates
"""
from __future__ import annotations

import numpy as np

from SelfEvolvingHarnessTS.harness import HarnessState, EditPatch, Manifest, PipelineTemplate, StageDef
from SelfEvolvingHarnessTS.harness.editable_surfaces import validate as surface_validate
from SelfEvolvingHarnessTS.fast_path import compose, execute, run_gates, Program, ProgramStep
from SelfEvolvingHarnessTS.fast_path.perceive import perceive
from SelfEvolvingHarnessTS.slow_path import BatchBuilder, Validator, Proposer, WeaknessReport
from SelfEvolvingHarnessTS.data.synthetic_gen import make_forecast_series


def _mf():
    return Manifest("t", "d", "specialize this cell", "", "")


def _key(pat, seed=0):
    h = HarnessState.from_minimal()
    rs = make_forecast_series(pat, seed)
    return perceive(rs.history, "forecast", h), rs


# ── 1. from_dict 重建 ─────────────────────────────────────────────────────
def test_template_from_dict():
    d = {"name": "x", "applies_to": {"task_type": "forecast", "pattern_conditions": {"pattern_bin": "snrLow|miss"}},
         "stages": [{"stage": "denoise", "preferred_ops": ["denoise_median"], "banned_ops": ["winsorize"]}]}
    t = PipelineTemplate.from_dict(d)
    assert t.name == "x" and isinstance(t.stages[0], StageDef)
    assert t.stages[0].preferred_ops == ["denoise_median"] and t.stages[0].banned_ops == ["winsorize"]


# ── 2. compose 用匹配该 cell 的模板；其他 cell 不受影响 ───────────────────
def test_compose_uses_matching_template():
    h = HarnessState.from_minimal()
    key, rs = _key("G_hi_full"); pb = key["pattern_bin"]
    tmpl = PipelineTemplate("fc_hi", {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb}},
                            [StageDef("impute", preferred_ops=["impute_linear"]),
                             StageDef("shape", preferred_ops=["znorm"])])
    h.apply_edit(EditPatch("L2", "set", "l2.task_templates::fc_hi", tmpl, _mf()))
    key = perceive(rs.history, "forecast", h)
    prog = compose(key, h)
    assert prog.source == "template" and prog.note == "tmpl:fc_hi"
    assert prog.op_names() == ["impute_linear", "znorm"]
    # 不同 cell（snrLow）→ 模板不匹配 → 走 heuristic（不会是这个模板）
    rs2 = make_forecast_series("G_lo_full", 0)
    prog2 = compose(perceive(rs2.history, "forecast", h), h)
    assert prog2.note != "tmpl:fc_hi"


# ── 3. cell-scoped ban：只在本 cell 禁，其他 cell 不禁 ────────────────────
def test_cell_scoped_ban():
    h = HarnessState.from_minimal()
    key, rs = _key("G_hi_miss"); pb = key["pattern_bin"]
    assert "winsorize" in compose(perceive(rs.history, "forecast", h), h).op_names()   # 基线含 winsorize
    tmpl = PipelineTemplate("ban_wz", {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb}},
                            [StageDef("outlier", banned_ops=["winsorize"])])
    h.apply_edit(EditPatch("L2", "set", "l2.task_templates::ban_wz", tmpl, _mf()))
    assert "winsorize" not in compose(perceive(rs.history, "forecast", h), h).op_names()  # 本 cell 禁掉
    rs2 = make_forecast_series("G_lo_miss", 0)                                            # 其他 cell 不受影响
    assert "winsorize" in compose(perceive(rs2.history, "forecast", h), h).op_names()


# ── 4. Skill Gate 也 cell-scoped ──────────────────────────────────────────
def test_skill_gate_cell_scoped():
    h = HarnessState.from_minimal()
    key, rs = _key("G_hi_miss"); pb = key["pattern_bin"]
    tmpl = PipelineTemplate("ban_dn", {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb}},
                            [StageDef("denoise", banned_ops=["denoise_savgol"])])
    h.apply_edit(EditPatch("L2", "set", "l2.task_templates::ban_dn", tmpl, _mf()))
    prog = Program([ProgramStep("denoise_savgol", {})], source="llm_custom")
    ex = execute(prog, rs.history)
    _p1, _g1, sig1 = run_gates(rs.history, ex, prog, h, "forecast", pb)                  # 本 cell → skill 拦
    assert sig1 and sig1.startswith("skill:") and "denoise_savgol" in sig1
    _p2, _g2, sig2 = run_gates(rs.history, ex, prog, h, "forecast", "snrLow|full")        # 别的 cell → skill 不拦
    assert sig2 is None or not sig2.startswith("skill:")


# ── 5. proposer 解析模板 JSON → 合法 EditPatch（scope=cell）───────────────
def test_proposer_parses_template():
    h = HarnessState.from_minimal()
    js = ('{"edited_layer":"L2","op":"set","path":"l2.task_templates::fc_lo",'
          '"value":{"name":"fc_lo","applies_to":{"task_type":"forecast","pattern_conditions":{"pattern_bin":"snrLow|miss"}},'
          '"stages":[{"stage":"denoise","preferred_ops":["denoise_median"],"banned_ops":["winsorize"]}]},'
          '"manifest":{"target_failure_id":"f"},"reasoning":"specialize"}')
    pr = Proposer(llm=lambda s, u, nonce=0: js, k=1)
    w = WeaknessReport("forecast|snrLow|miss", "forecast", 0.5, 0.4, 0.1, True, {}, [])
    cands = pr.propose(h, w)
    assert len(cands) == 1
    c = cands[0]
    assert c.path == "l2.task_templates::fc_lo" and isinstance(c.value, PipelineTemplate)
    res = surface_validate(c, h)
    assert res.ok and res.resolved_scope == "cell"


# ── 6. validator：cell-scoped 模板天然 Pareto 安全（不影响其他 cell）──────
def test_validator_cell_template_pareto_safe():
    h = HarnessState.from_minimal()
    bb = BatchBuilder(h, n_min=6)
    for pat in ("G_hi_full", "G_lo_full"):
        for rs in [make_forecast_series(pat, s) for s in range(12)]:
            bb.add_raw_series(rs)
    cell = next(c for c in bb.triggerable_cells() if "snrHigh" in c)
    pb = cell.split("|", 1)[1]
    # 把 winsorize(削峰) 换成 outlier_iqr(温和) —— 只对本 cell
    tmpl = PipelineTemplate("fc_hi_iqr", {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb}},
                            [StageDef("impute", preferred_ops=["impute_linear"]),
                             StageDef("outlier", preferred_ops=["outlier_iqr"]),
                             StageDef("denoise", preferred_ops=["denoise_savgol"])])
    patch = EditPatch("L2", "set", "l2.task_templates::fc_hi_iqr", tmpl, _mf(), cell_id=cell)
    out = Validator().validate(patch, h, cell, bb.splits(cell))
    assert out.resolved_scope == "cell"
    assert out.pareto_safe                       # 模板不匹配其他 cell → 它们 val_loss 不变 → Pareto 安全
    print(f"    held_in Δ={out.val_in_cur - out.val_in_cand:+.4f}  accept={out.accept}")


# ── 7. 软结构门（方向 A）：同 bin 但结构远的模板不复用，结构近的复用 ──────
def test_soft_dstruct_gate():
    from SelfEvolvingHarnessTS.config import thresholds as TH
    from SelfEvolvingHarnessTS.conditioning.distance import d_struct
    stages = [StageDef("impute", preferred_ops=["impute_linear"]),
              StageDef("shape", preferred_ops=["znorm"])]

    # 近锚 = 当前 cell 自身特征（d_struct=0<τ）→ 复用
    h = HarnessState.from_minimal()
    key, rs = _key("G_hi_full"); pb = key["pattern_bin"]; feats = key["pattern"]["struct_feats"]
    near = PipelineTemplate("fc_near",
            {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb, "struct_ref": dict(feats)}}, stages)
    h.apply_edit(EditPatch("L2", "set", "l2.task_templates::fc_near", near, _mf()))
    assert compose(perceive(rs.history, "forecast", h), h).note == "tmpl:fc_near"

    # 远锚 = 结构迥异（周期/SNR/趋势/季节大幅偏移，d_struct>τ）→ 不复用，回退 heuristic
    h2 = HarnessState.from_minimal()
    far_ref = dict(feats)
    far_ref["period"] = feats.get("period", 1.0) + 200.0
    far_ref["SNR"] = feats.get("SNR", 0.0) - 60.0
    far_ref["trend_strength"] = 1.0 - feats.get("trend_strength", 0.0)
    far_ref["seasonal_strength"] = 1.0 - feats.get("seasonal_strength", 0.0)
    far = PipelineTemplate("fc_far",
            {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb, "struct_ref": far_ref}}, stages)
    h2.apply_edit(EditPatch("L2", "set", "l2.task_templates::fc_far", far, _mf()))
    note = compose(perceive(rs.history, "forecast", h2), h2).note
    assert note != "tmpl:fc_far" and note.startswith("heuristic")        # 软门拦住 → heuristic
    # 守门确由距离决定（非阈值巧合）
    assert d_struct(dict(feats), feats) < TH.BIN_DSTRUCT_TAU <= d_struct(far_ref, feats)


# ── 8. evolve 给 accepted cell-scoped 模板补打 struct_ref；全局模板不打 ──────
def test_evolve_tags_struct_ref():
    from SelfEvolvingHarnessTS.slow_path.evolve import Evolver
    from SelfEvolvingHarnessTS.slow_path.batch_builder import cell_sample_from_raw_series
    h = HarnessState.from_minimal()
    bb = BatchBuilder(h, n_min=4)
    samples = [cell_sample_from_raw_series(make_forecast_series("G_hi_full", s)) for s in range(4)]
    cell_id = perceive(samples[0].raw, "forecast", h)["cell_id"]
    pb = cell_id.split("|", 1)[1]
    ev = Evolver(h, bb, proposer=None)

    # cell-scoped 模板（无 struct_ref）→ 合入后补打
    tmpl = PipelineTemplate("fc_tag", {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb}},
                            [StageDef("impute", preferred_ops=["impute_linear"])])
    h.apply_edit(EditPatch("L2", "set", "l2.task_templates::fc_tag", tmpl, _mf()))
    ev._tag_template_struct_ref(EditPatch("L2", "set", "l2.task_templates::fc_tag", tmpl, _mf(), cell_id=cell_id),
                                cell_id, samples)
    pc = h.l2.task_templates["fc_tag"].applies_to["pattern_conditions"]
    assert "struct_ref" in pc and len(pc["struct_ref"]) == 10

    # 全局模板（pattern_conditions=None）→ 不打标，保持全局
    g = PipelineTemplate("fc_glob", {"task_type": "forecast", "pattern_conditions": None},
                         [StageDef("impute", preferred_ops=["impute_linear"])])
    h.apply_edit(EditPatch("L2", "set", "l2.task_templates::fc_glob", g, _mf()))
    ev._tag_template_struct_ref(EditPatch("L2", "set", "l2.task_templates::fc_glob", g, _mf(), cell_id=cell_id),
                                cell_id, samples)
    assert h.l2.task_templates["fc_glob"].applies_to.get("pattern_conditions") in (None, {})


def _run_all():
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}"); passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
