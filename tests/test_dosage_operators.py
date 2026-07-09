"""tests/test_dosage_operators.py — E-3.3 Family 0 剂量算子 gate（评审第十二轮清单）。

新算子 `smooth_ma` 与 8 个剂量动作（median 9/15/25、savgol 21/31、MA 9/15/25）在进入 nested
供给评估前必须过：
  ①契约完整（7 字段）+ 注册/激活 + 非 anomaly（smoothing）；
  ②边界：常数逐点保持（含端点）、线性趋势端点有界（无零填充伪影）；
  ③剂量单调可分：不同窗/不同算子在含噪序列上输出不同（无意外 semantic-dup 入池）；
  ④no-defect harm：干净正弦上重窗仍有限、长度不变、值域不爆；
  ⑤task 物理过滤：anomaly 的 usable_ops 不含 smooth_ma；forecast 含；
  ⑥params_override 端到端注入：dosage_variant 经 fast_process 真的按窗施加（换窗即换输出，
    且无缺失历史上 ready ≡ moving_average(history, W)）。
"""
from __future__ import annotations

import numpy as np
import pytest

from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA, OPERATOR_NAMES, TOOL_REGISTRY, get_operator)
from SelfEvolvingHarnessTS.operators._common import moving_average, BOUNDARY_MODES
from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.fast_path.compose import usable_ops
from SelfEvolvingHarnessTS.fast_path.pipeline import process as fast_process
from SelfEvolvingHarnessTS.family0_actions import F0_DOSAGE_GRID, dosage_variant, f0_variants

_CONTRACT_FIELDS = {"allowed_tasks", "destructive", "preserves_observed", "reversible",
                    "changes_target_space", "requires_dependency", "fallback_policy"}
_MEDIAN_WINDOWS = [9, 15, 25]
_SAVGOL_WINDOWS = [21, 31]
_MA_WINDOWS = [9, 15, 25]


# ── ① smooth_ma 注册/契约/激活 ────────────────────────────────────────────
def test_smooth_ma_registered_and_contract_complete():
    assert "smooth_ma" in OPERATOR_NAMES and "smooth_ma" in TOOL_REGISTRY
    m = OPERATOR_METADATA["smooth_ma"]
    assert _CONTRACT_FIELDS <= set(m), f"smooth_ma 契约缺字段: {_CONTRACT_FIELDS - set(m)}"
    assert "smoothing" in m["tags"]
    assert m["allowed_tasks"] == ("forecast", "classification")   # 非 anomaly
    assert m["destructive"] is False and m["requires_dependency"] is None
    assert not m.get("is_alias")
    assert BOUNDARY_MODES["smooth_ma"] == "symmetric"


def test_smooth_ma_active_in_minimal_harness():
    h = HarnessState.from_minimal()
    assert h.l2.active_operators.get("smooth_ma") is True


def test_smooth_ma_nan_safe_and_length_preserving():
    fn = get_operator("smooth_ma")
    x = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])
    y = fn(x, window=5)
    assert y.shape == x.shape and np.all(np.isfinite(y))


# ── ② 边界：常数保持 + 趋势端点有界（对全部剂量 op×window） ─────────────────
def _grid_ops_windows():
    return ([("denoise_median", w) for w in _MEDIAN_WINDOWS]
            + [("denoise_savgol", w) for w in _SAVGOL_WINDOWS]
            + [("smooth_ma", w) for w in _MA_WINDOWS])


@pytest.mark.parametrize("op,w", _grid_ops_windows())
def test_constant_preserved_all_positions(op, w):
    fn = get_operator(op)
    x = np.full(64, 5.0)
    y = fn(x, window=w)
    assert np.allclose(y, 5.0, atol=1e-9), f"{op}@{w} 未逐点保持常数（端点零填充伪影？）max|Δ|={np.max(np.abs(y-5.0)):.2e}"


@pytest.mark.parametrize("op,w", _grid_ops_windows())
def test_linear_trend_endpoint_bounded(op, w):
    fn = get_operator(op)
    slope = 0.3
    x = slope * np.arange(80, dtype=float)
    y = fn(x, window=w)
    # symmetric 边界：端点偏移应 ≤ slope*window 量级（零填充会把端点拉向 0 → 偏移 ~ x[0]/x[-1]）
    assert abs(y[0] - x[0]) <= slope * w + 1e-6, f"{op}@{w} 左端点偏移过大: {y[0]:.3f} vs {x[0]:.3f}"
    assert abs(y[-1] - x[-1]) <= slope * w + 1e-6, f"{op}@{w} 右端点偏移过大: {y[-1]:.3f} vs {x[-1]:.3f}"


# ── ③ 剂量可分：不同窗/不同算子输出不同（无意外 semantic-dup） ──────────────
def test_dosage_windows_distinct():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 1, 120)) + rng.normal(0, 0.5, 120)
    for op, windows in [("denoise_median", _MEDIAN_WINDOWS), ("denoise_savgol", _SAVGOL_WINDOWS),
                        ("smooth_ma", _MA_WINDOWS)]:
        fn = get_operator(op)
        outs = [fn(x, window=w) for w in windows]
        for i in range(len(outs)):
            for j in range(i + 1, len(outs)):
                assert not np.allclose(outs[i], outs[j]), f"{op} 窗 {windows[i]}≠{windows[j]} 却输出相同"


def test_ops_at_same_window_distinct():
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 1, 120)) + rng.normal(0, 0.5, 120)
    ma = get_operator("smooth_ma")(x, window=15)
    med = get_operator("denoise_median")(x, window=15)
    assert not np.allclose(ma, med), "smooth_ma 与 denoise_median 同窗输出相同（意外重复列）"


# ── ④ no-defect harm：干净正弦上**内部**不放大方差（抓病态放大如旧 wavelet 2×identity）。
#     端点剂量质量（如 savgol@31 窗>周期时的多项式过冲）不在 gate——那正是 F0 nested 要测量、
#     并由选择器在季节 cell 里正确拒绝的信号，非隐藏伪影（详 family0_actions 文档）。
@pytest.mark.parametrize("op,w", _grid_ops_windows())
def test_no_defect_harm_interior_not_amplified(op, w):
    t = np.arange(240, dtype=float)
    x = np.sin(2 * np.pi * t / 24)
    y = get_operator(op)(x, window=w)
    assert y.shape == x.shape and np.all(np.isfinite(y))
    half = int(w)                                          # 每端排除一个整窗（端点=剂量质量，交实验）
    xi, yi = x[half:-half], y[half:-half]
    assert np.std(yi) <= np.std(xi) + 1e-9, f"{op}@{w} 内部平滑反而放大方差（病态）"


# ── ⑤ task 物理过滤 ───────────────────────────────────────────────────────
def test_anomaly_excludes_smooth_ma():
    h = HarnessState.from_minimal()
    fc = usable_ops(h, "forecast")
    an = usable_ops(h, "anomaly_detection")
    assert "smooth_ma" in fc, "forecast 应允许 smooth_ma"
    assert "smooth_ma" not in an, "anomaly 应物理禁 smooth_ma（smoothing 毁 spike）"


# ── ⑥ params_override 端到端注入（F0 runner 依赖此正确性） ─────────────────
def test_dosage_variant_injects_window_end_to_end():
    rng = np.random.default_rng(2)
    hist = np.cumsum(rng.normal(0, 1, 96)) + rng.normal(0, 0.4, 96)   # 无缺失 → impute_linear 恒等
    # 换窗即换输出
    r9 = fast_process(hist, "forecast", dosage_variant("f0_ma_w9", "smooth_ma", 9), store=None)[1]
    r25 = fast_process(hist, "forecast", dosage_variant("f0_ma_w25", "smooth_ma", 25), store=None)[1]
    assert not np.allclose(r9, r25), "换窗（9→25）未改变 ready → params_override 未注入"
    # 无缺失历史上 ready ≡ moving_average(history, 15)
    r15 = fast_process(hist, "forecast", dosage_variant("f0_ma_w15", "smooth_ma", 15), store=None)[1]
    assert np.allclose(r15, moving_average(hist, 15), atol=1e-8), "ready 未等于 smooth_ma@15 直算（管线注入不一致）"


def test_f0_variants_complete_and_ids_have_params():
    v = f0_variants("forecast")
    assert len(v) == len(F0_DOSAGE_GRID) == 8
    for name, _op, w in F0_DOSAGE_GRID:
        assert name in v and (f"w{w}" in name), f"动作 ID {name} 未含窗参数"
