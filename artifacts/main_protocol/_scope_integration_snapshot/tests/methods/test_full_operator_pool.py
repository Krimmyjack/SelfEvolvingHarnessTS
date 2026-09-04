"""FULL_OPERATOR_SKILL_CAPABILITY（2026-08-14）聚焦测试——算子池切换（无 LLM）。

锁住两臂候选供给面的机械行为：
  - pool_mode=actionable（默认，生产路径零改动）：T100@600 上 A 池 = 6 算子
    （与 P4 记录证据一致的集合：denoise_median/hampel_filter/outlier_iqr/
    outlier_mad/resample_uniform/winsorize）；
  - pool_mode=full（新路径）：_full_pool_operators = 26 canonical − 3
    shape_changing = 23（task/依赖/绑定前提过滤；不含 verifier 0.35 探测）；
    propose 面再减无缺失时的 7 个缺失处理族 = 16；
  - A ⊆ B（full 池是 actionable 池的超集）；
  - 合成缺失 Context：full 池包含 impute 族，noop 过滤器不再剔除；
  - 绑定前提：绑定特征不完整 → repair_level_shift 被剔除；
  - 依赖前提：hard_fail 依赖缺失 → 剔除；recorded_fallback → 保留；
  - pool_mode 非法值 → ValueError（在任何 stage 之前）。

装置：真实 cohort 数据（nsu._load_cohort）+ h0 snapshot（verify_lock=False，
与全部 runner 同一惯例）+ forecast_task_context_v1。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import run_v1_kdd2018_natural_slow_update as nsu  # noqa: E402

import SelfEvolvingHarnessTS.methods.ttha.fast_agent as fa  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    deployment_constraints_v1,
    forecast_task_context_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _full_pool_operators,
    _noop_ops_for_context,
    _actionable_operators,
    _allowed_operators,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)

H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"

# P4 记录证据锁定的 A 池（T100@600/T1@888 两 context 一致）
_EXPECTED_ACTIONABLE_6 = {
    "denoise_median", "hampel_filter", "outlier_iqr", "outlier_mad",
    "resample_uniform", "winsorize",
}
# 参数所有权修复（AGENTIC_SKILL_HARNESS_GLOBAL_DESIGN_2026-08-19 §7.3/§17.4）之后新增的第 7 个。
# 机械原因：_actionable_operators 对声明了 public_parameter_bindings 的算子
# 用代表序列特征构造**绑定候选**去实测 verifier；legacy 绑定在 T100@600 上
# 修改了超过 0.35 的点，因此被判不可行动。契约改为 OPERATOR_INTRINSIC 后，
# 同一算子用空参数构造，算子在窗口内部自行定位，修改分数落回 cap 之下。
# 这不是放宽 verifier——cap 未变，变的是交给 verifier 的那个候选。
_EXPECTED_ACTIONABLE_7 = _EXPECTED_ACTIONABLE_6 | {"repair_level_shift"}
_MISSING_ONLY = {
    "impute_ar", "impute_ema", "impute_fft", "impute_linear", "impute_ssm",
    "period_complete", "period_median_complete",
}
_SHAPE_CHANGING = {"sliding_window", "lag_features", "spectral_features"}


def _cohort():
    return nsu._load_cohort(PROJECT_ROOT)


def _h0():
    return compile_snapshot(H0_ROOT, verify_lock=False)


def _request(series_uid: str, origin: int, values=None):
    cohort = _cohort()
    if values is None:
        values = cohort["values"]
    series0 = np.asarray(values[series_uid], dtype=np.float64)
    request = nsu._request(series0, values, origin)
    return dataclasses.replace(
        request,
        task_context=forecast_task_context_v1(
            task_spec=request.task_spec,
            deployment_constraints=deployment_constraints_v1()),
    )


def _view(request, h0=None):
    features = extract_public_features(
        np.asarray(request.values, dtype=float), task_kind="forecast")
    return resolve_harness_view(h0 or _h0(), features, role="fast")


def _a_pool(request, h0=None) -> set[str]:
    h0 = h0 or _h0()
    actionable = set(_actionable_operators(
        request, np.asarray(request.values, dtype=float),
        _view(request, h0), _allowed_operators(request)))
    return actionable - set(_noop_ops_for_context(request))


def _b_pool(request) -> set[str]:
    """full 模式的 propose 面：_full_pool_operators 后再减 noop 前提
    （与 fast_agent.prepare 的 supply_ops → propose_ops 组装一致）。"""
    return set(_full_pool_operators(request)) - set(
        _noop_ops_for_context(request))


def _synmiss_values():
    cohort = _cohort()
    values = dict(cohort["values"])
    arr = np.asarray(values["T101"], dtype=np.float64).copy()
    arr[list(range(300, 360)) + [100, 200, 250, 450, 500]] = np.nan
    values["T101"] = arr
    return values


def test_actionable_pool_on_t100_600_after_parameter_ownership_fix() -> None:
    """A 路径 = P4 记录的 6 算子 + intrinsic 化后重新可行动的 level 修复。"""
    request = _request("T100", 600)
    assert _a_pool(request) == _EXPECTED_ACTIONABLE_7


def test_full_pool_mechanical_filters_t100_600() -> None:
    """full 路径：canonical 26 − shape 3 = 23（无缺失 Context 的 propose
    面再减缺失处理族 7 = 16）。不含 0.35 探测——全局平滑/分解/归一化族
    全部出现在暴露面。"""
    request = _request("T100", 600)
    full = set(_full_pool_operators(request))
    assert len(full) == 23, full
    assert not (full & _SHAPE_CHANGING)
    # 全局族（0.35 探测会排除的）在 full 池中出现
    assert {"denoise_stl", "smooth_ma", "stl_decompose", "fft_decompose",
            "smooth_ema", "znorm", "minmax_norm", "repair_level_shift",
            "denoise_savgol", "denoise_wavelet"} <= full
    pool = _b_pool(request)
    assert len(pool) == 16, pool
    assert not (pool & _MISSING_ONLY)


def test_full_pool_superset_of_actionable() -> None:
    request = _request("T100", 600)
    assert _a_pool(request) <= _b_pool(request)


def test_full_pool_includes_impute_on_synthetic_missingness() -> None:
    """合成缺失 Context：缺失处理族不再是确定性 no-op → 两臂都包含
    impute 族（A 经探测，B 经前提——探测在 10.8% ≤ 0.35 下通过）。
    机械事实（实测锁定）：A 池在此 Context = 原 6 中除 resample_uniform
    （NaN 序列上探测失败——合法剔除）外全部保留 + 7 缺失处理族 +
    repair_level_shift（绑定完整且 region ≤ 0.35）。"""
    values = _synmiss_values()
    request = _request("T101", 600, values=values)
    features = extract_public_features(
        np.asarray(request.values, dtype=float), task_kind="forecast")
    assert float(features["missing_fraction"]) > 0.05
    full = _full_pool_operators(request)
    assert "impute_linear" in full and "impute_fft" in full
    assert _noop_ops_for_context(request) == ()
    a_pool = _a_pool(request)
    assert _MISSING_ONLY <= a_pool
    assert "repair_level_shift" in a_pool
    assert _EXPECTED_ACTIONABLE_6 - {"resample_uniform"} <= a_pool
    assert "resample_uniform" not in a_pool  # NaN 序列探测失败（机械）
    # B 池 = 26 − 3 shape = 23（合成 Context 的 propose 面无剔除）
    assert len(_b_pool(request)) == 23


def test_no_operator_supply_depends_on_external_region_features(
    monkeypatch,
) -> None:
    """参数所有权修复（AGENTIC_SKILL_HARNESS_GLOBAL_DESIGN_2026-08-19 §7.3/§17.4）后没有算子的可用性挂在外部定位特征上。

    旧契约下 repair_level_shift 的执行前提是三个 region/offset 特征齐全，
    缺一个就被剔除。现在它在自己拿到的 action unit 内部定位，因此把这三个
    特征从公开特征里删掉，供给面不变——这正是 RUNTIME_BOUND 归零的可观察
    后果，也是"不在 intrinsic Operator 外再建一套定位器"的机械前提。
    """
    request = _request("T100", 600)
    real_extract = fa.extract_public_features

    def _no_region(values, *, task_kind):  # noqa: ANN001
        feats = real_extract(values, task_kind=task_kind)
        # extract_public_features 返回 mappingproxy（不可变）——重建 dict
        drop = {"estimated_region_start_fraction",
                "estimated_region_end_fraction",
                "estimated_level_offset"}
        return {k: v for k, v in feats.items() if k not in drop}

    baseline = set(_full_pool_operators(request))
    monkeypatch.setattr(fa, "extract_public_features", _no_region)
    full = set(_full_pool_operators(request))
    assert full == baseline
    assert "repair_level_shift" in full
    assert "outlier_mad" in full and "denoise_stl" in full


def test_full_pool_dependency_policy(monkeypatch) -> None:
    """依赖缺失：hard_fail（impute_ssm）剔除；recorded_fallback
    （denoise_savgol/denoise_stl）保留。"""
    request = _request("T100", 600)
    monkeypatch.setattr(fa, "_dependency_available", lambda dep: False)
    full = set(_full_pool_operators(request))
    assert "impute_ssm" not in full
    assert "denoise_savgol" in full
    assert "denoise_stl" in full
    assert "denoise_wavelet" in full  # recorded_fallback→denoise_savgol


def test_pool_mode_invalid_rejected() -> None:
    """非法 pool_mode 在任何 stage/core 使用之前抛 ValueError。"""
    import pytest  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: PLC0415
    request = _request("T100", 600)
    fast = TTHAFastAgent(object())  # fake core：验证先于 core 使用
    with pytest.raises(ValueError):
        fast.prepare(request, _h0(), pool_mode="bogus")
