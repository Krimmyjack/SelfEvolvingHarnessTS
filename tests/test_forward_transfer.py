"""tests/test_forward_transfer.py — ★v4 S2 前向迁移曲线分析 + bug 修复回归。

覆盖：
  • bug 回归：stl_decompose 默认 period=0 真跑 STL（非 'auto' 字符串吞成 savgol 回退）；
    denoise_stl 对 'auto'/None/非法 period 不崩；deploy_stream scratch/updating 缺 proposer 抛 ValueError。
  • S2：per_domain_points 聚合、forward_transfer_verdict（助益/退化/分离/护栏）、NaN/file 往返。
"""
import json
import math

import numpy as np
import pytest

from SelfEvolvingHarnessTS.harness.state import HarnessState
from SelfEvolvingHarnessTS.operators import s1_denoise, s1_decompose
from SelfEvolvingHarnessTS.slow_path import deploy_stream as ds
from SelfEvolvingHarnessTS.slow_path import forward_transfer as ft


# ════════════════════════════ bug 修复回归 ════════════════════════════
def test_stl_default_param_runs_real_stl():
    """bug#1：默认 period 应为 0（自动猜测，真跑 STL），不是 'auto'（撞 >= 比较被吞成 savgol）。"""
    defaults = HarnessState.from_minimal().l2.operator_defaults
    assert defaults["stl_decompose"] == {"period": 0}
    x = np.sin(np.linspace(0, 20, 200)) + 0.1 * np.random.RandomState(0).randn(200)
    savgol = s1_denoise.denoise_savgol(x)
    out = s1_decompose.stl_decompose(x, **defaults["stl_decompose"])
    assert not np.allclose(out, savgol)          # 真 STL ≠ savgol 回退


def test_denoise_stl_tolerates_nonint_period():
    """bug#1 纵深：'auto'/None/非法 period 应被 coerce 成自动猜测，绝不抛、绝不静默退 savgol。"""
    x = np.sin(np.linspace(0, 20, 200)) + 0.1 * np.random.RandomState(1).randn(200)
    savgol = s1_denoise.denoise_savgol(x)
    auto = s1_decompose.stl_decompose(x, period="auto")
    assert np.allclose(auto, s1_decompose.stl_decompose(x, period=0))   # 'auto' == 自动猜测
    assert not np.allclose(auto, savgol)                                # 不再退 savgol
    for bad in (None, "auto", -5, 0, 1, True, 1.9):                     # 一律不崩
        s1_decompose.stl_decompose(x, period=bad)
    assert s1_denoise._coerce_period("auto") == 0
    assert s1_denoise._coerce_period(24) == 24
    assert s1_denoise._coerce_period(-3) == 0


def test_savgol_default_param_accepts_order():
    """bug#3 复核（假阳性）：denoise_savgol 包装器接受 'order' 并内部翻译成 scipy polyorder，不应崩。"""
    defaults = HarnessState.from_minimal().l2.operator_defaults
    assert "order" in defaults["denoise_savgol"]   # 包装器参数名就是 order（非 polyorder）
    x = np.sin(np.linspace(0, 20, 120)) + 0.1 * np.random.RandomState(2).randn(120)
    y = s1_denoise.denoise_savgol(x, **defaults["denoise_savgol"])
    assert y.shape == x.shape and np.all(np.isfinite(y))


@pytest.mark.parametrize("mode", ["scratch", "updating"])
def test_deploy_stream_requires_proposer(mode):
    """bug#2：scratch/updating 缺 make_proposer 应在入口抛 ValueError（非进化中 AttributeError）。"""
    doms = [ds.DomainSpec("d0", [], ("forecast",))]
    with pytest.raises(ValueError):
        ds.deploy_stream(doms, mode=mode, make_harness=HarnessState.from_minimal)


def test_deploy_stream_frozen_allows_no_proposer():
    """frozen 不进化 → 允许 proposer=None（空语料即跑通，不抛）。"""
    doms = [ds.DomainSpec("d0", [], ("forecast",))]
    res = ds.deploy_stream(doms, mode="frozen", make_harness=HarnessState.from_minimal)
    assert res.mode == "frozen"


# ════════════════════════════ S2 聚合 ════════════════════════════
def _row(k, mode, cell, ttr, readiness, ready, demote=0, ver=0, domain=None):
    return {"k": k, "domain": domain or f"D{k}", "mode": mode, "cell": cell, "task": "forecast",
            "time_to_readiness_rounds": ttr, "llm_calls_to_readiness": ttr,
            "readiness_at_budget": readiness, "ready": ready,
            "j_raw": 1.0, "j_cur": 0.5, "j_min_ref": 0.5,
            "harness_version": ver, "n_reval_demote_domain": demote}


def test_per_domain_points_aggregation():
    rows = [_row(0, "updating", "c1", 2, 0.9, True, ver=2),
            _row(0, "updating", "c2", None, float("nan"), False, ver=2)]
    pts = ft.per_domain_points(rows)
    assert len(pts) == 1
    p = pts[0]
    assert p.n_cells == 2 and p.n_ready == 1
    assert p.ttr_median == 2.0          # 只取有限值
    assert p.ttr_max is None            # 任一 cell None → worst-case 未就绪
    assert p.readiness_median == 0.9    # NaN 被过滤
    assert p.ready_frac == 0.5


def test_verdict_forward_transfer_supported():
    """C(updating) 后到域更快达标（ttr 降）且 readiness 不退 → supported=True, discriminative=True。"""
    c = [_row(0, "updating", "c", 3, 0.8, True), _row(1, "updating", "c", 1, 0.95, True)]
    a = [_row(0, "scratch", "c", 3, 0.8, True), _row(1, "scratch", "c", 3, 0.8, True)]
    b = [_row(0, "frozen", "c", 0, 0.7, True), _row(1, "frozen", "c", 0, 0.9, True)]
    v = ft.forward_transfer_verdict(ft.build_curves({"updating": c, "scratch": a, "frozen": b}))
    assert v["discriminative"] is True
    assert v["memory_helps"] is True
    assert v["no_degradation"] is True
    assert v["forward_transfer_supported"] is True
    assert v["mean_ttr_gain_A_minus_C"] == pytest.approx(1.0)   # (0 + 2)/2
    # 三 bootstrap 分解存在
    assert v["per_k"][1]["memory_value_B_minus_A"] is not None


def test_verdict_saturated_tie_is_inconclusive():
    """C 与 A 全平局（差分在容忍带内）→ discriminative=False → supported=None（不可结论）。"""
    c = [_row(0, "updating", "c", 1, 1.0, True), _row(1, "updating", "c", 1, 1.0, True)]
    a = [_row(0, "scratch", "c", 1, 1.0, True), _row(1, "scratch", "c", 1, 1.0, True)]
    v = ft.forward_transfer_verdict(ft.build_curves({"updating": c, "scratch": a}))
    assert v["discriminative"] is False
    assert v["forward_transfer_supported"] is None


def test_verdict_negative_transfer_guardrail():
    """C 域 reval_demote>0 → 护栏 fired=True。"""
    c = [_row(0, "updating", "c", 2, 0.9, True, demote=1)]
    a = [_row(0, "scratch", "c", 3, 0.7, True)]
    v = ft.forward_transfer_verdict(ft.build_curves({"updating": c, "scratch": a}))
    assert v["neg_transfer_guardrail_fired"] is True
    assert v["total_reval_demote_C"] == 1


def test_load_transfer_log_roundtrip_with_nan(tmp_path):
    """JSONL 往返：含 NaN 行可读回（json 默认 allow_nan）；空行跳过。"""
    p = tmp_path / "forward_transfer_updating.jsonl"
    lines = [_row(0, "updating", "c", 1, 1.0, True), _row(1, "updating", "c", None, float("nan"), False)]
    p.write_text("\n".join(json.dumps(r) for r in lines) + "\n\n", encoding="utf-8")
    rows = ft.load_transfer_log(str(p))
    assert len(rows) == 2
    assert math.isnan(rows[1]["readiness_at_budget"])
    pts = ft.per_domain_points(rows)
    assert pts[1].readiness_median is None     # 全 NaN → None


def test_analyze_end_to_end_on_demo_logs():
    """对真实 demo 目录（若存在）跑通 analyze，结构完整。"""
    import os
    demo = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "s1_demo")
    if not os.path.isdir(demo):
        pytest.skip("runs/s1_demo 不存在")
    logs = {}
    for m in ("updating", "frozen", "scratch"):
        fp = os.path.join(demo, f"forward_transfer_{m}.jsonl")
        if os.path.exists(fp):
            logs[m] = ft.load_transfer_log(fp)
    res = ft.analyze(logs)
    assert "curves" in res and "verdict" in res
    assert "forward_transfer_supported" in res["verdict"]
