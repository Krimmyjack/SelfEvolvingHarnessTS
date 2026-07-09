"""Phase 2b 验证：outcome-calibrated 信用分配（OPD 公式11）。

ops_credit 符号、γ(N) 置信加权、summary prefer/avoid，以及 evolve 中累积 → 注入 proposer。

运行：  PYTHONPATH=<Agent> D:/Anaconda_envs/envs/project/python.exe -m SelfEvolvingHarnessTS.tests.test_attribution
"""
from __future__ import annotations

import math

from SelfEvolvingHarnessTS.harness import HarnessState, EditPatch, Manifest, PipelineTemplate, StageDef
from SelfEvolvingHarnessTS.slow_path import AttributionStore, ops_credit
from SelfEvolvingHarnessTS.slow_path.attribution import _gamma


def _mf():
    return Manifest("a")


# ── 1. γ 单调 + 已知值 ────────────────────────────────────────────────────
def test_gamma():
    assert _gamma(0) == 0.0
    assert abs(_gamma(1) - (1 - 1 / math.sqrt(2))) < 1e-9
    assert _gamma(3) == 0.5
    assert _gamma(1) < _gamma(2) < _gamma(10) < 1.0


# ── 2. ops_credit 符号：prefer/enable +delta；ban/disable −delta ──────────
def test_ops_credit_signs():
    tmpl = PipelineTemplate("t", {"task_type": "forecast"},
                            [StageDef("outlier", preferred_ops=["outlier_iqr"], banned_ops=["winsorize"])])
    p = EditPatch("L2", "set", "l2.task_templates::t", tmpl, _mf())
    cr = ops_credit(p, delta=0.1)
    assert cr["outlier_iqr"] == 0.1 and cr["winsorize"] == -0.1     # prefer:+, ban:−

    enable = EditPatch("L2", "set", "l2.active_operators.znorm", True, _mf())
    assert ops_credit(enable, 0.2)["znorm"] == 0.2                  # enable:+
    disable = EditPatch("L2", "set", "l2.active_operators.znorm", False, _mf())
    assert ops_credit(disable, 0.2)["znorm"] == -0.2               # disable:−（禁它有用=它有害）


# ── 3. AttributionStore 累积 + γ 加权 value ──────────────────────────────
def test_store_value_and_summary():
    s = AttributionStore()
    cell = "forecast|snrHigh|full"
    good = EditPatch("L2", "set", "l2.active_operators.outlier_iqr", True, _mf())
    bad = EditPatch("L2", "set", "l2.active_operators.winsorize", True, _mf())
    s.record(cell, good, 0.10); s.record(cell, good, 0.20)          # outlier_iqr 两次正
    s.record(cell, bad, -0.15)                                      # winsorize 一次负
    s.record(cell, bad, float("nan"))                              # 非有限 → 跳过

    val = s.value(cell)
    assert val["outlier_iqr"][1] == 2                               # N=2
    assert abs(val["outlier_iqr"][0] - _gamma(2) * 0.15) < 1e-9     # γ(2)·mean(0.10,0.20)
    assert val["winsorize"][1] == 1 and val["winsorize"][0] < 0     # 有害

    summ = s.summary(cell)
    assert summ["prefer"][0][0] == "outlier_iqr"                    # 最有益排第一
    assert summ["avoid"][0][0] == "winsorize"


# ── 4. 集成：evolve 跑一轮后 attribution 被填充并能喂回 proposer ──────────
def test_attribution_populated_in_evolve():
    from SelfEvolvingHarnessTS.slow_path import BatchBuilder, Evolver
    from SelfEvolvingHarnessTS.data.synthetic_gen import make_forecast_series

    h = HarnessState.from_minimal()
    bb = BatchBuilder(h, n_min=6)
    for pat in ("G_hi_full", "G_lo_full"):
        for rs in [make_forecast_series(pat, s) for s in range(12)]:
            bb.add_raw_series(rs)
    cell = next(c for c in bb.triggerable_cells() if "snrHigh" in c)
    pb = cell.split("|", 1)[1]

    # stub proposer：提一个会改善的 cell-scoped 模板（winsorize→outlier_iqr）
    tmpl = PipelineTemplate("fc", {"task_type": "forecast", "pattern_conditions": {"pattern_bin": pb}},
                            [StageDef("impute", preferred_ops=["impute_linear"]),
                             StageDef("outlier", preferred_ops=["outlier_iqr"], banned_ops=["winsorize"])])

    class Stub:
        def propose(self, harness, weakness, strength=None, rejection_log=None):
            q = EditPatch.from_dict(EditPatch("L2", "set", "l2.task_templates::fc", tmpl, _mf()).to_dict())
            q.value = tmpl; q.cell_id = weakness.cell_id; q.proposal_rank = 0
            return [q]

    ev = Evolver(h, bb, Stub())
    ev.evolve_cell(cell, epoch=0)
    val = ev.attribution.value(cell)
    assert "outlier_iqr" in val and val["outlier_iqr"][1] >= 1      # 被记录
    # outlier_iqr 应为正（改善）、winsorize 为负（被 ban 且 ban 有用）
    assert val["outlier_iqr"][0] > 0 and val.get("winsorize", (0, 0))[0] < 0
    summ = ev.attribution.summary(cell)
    assert any(op == "outlier_iqr" for op, _v, _n in summ["prefer"])


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
