"""tests/test_operators_e33.py — E-3.3 新算子的语义测试三件套 + 契约守卫（2026-07-14）。

四个新算子：repair_level_shift（R1）/ impute_ssm、impute_ar（R2）/ hampel_filter（R3）。

每个算子三件套（DoD）：
  ① **不该动的时候不动**（干净输入上恒等——这里一律要求**逐位恒等**，不是"近似恒等"：
     容差是给数值误差留的，不是给"改了一点点"留的）；
  ② **该动的时候动对**（注入已知缺陷后，修复误差显著下降）；
  ③ **反噬测试**（最重要的一条：在这个算子最容易帮倒忙的输入上，它必须无害或伤害有界）。

反噬测试为什么最重要：本仓库的算子事故全部是这一类——v_stl 曾经是"garbage-period 上激进平滑
碰巧赢"的伪影；impute_fft 曾经是伪装成 imputer 的 denoiser；impute_kalman 从来就不是 Kalman。
一个算子在它该赢的格子上赢，什么都不证明；它在**不该动的格子上不动**，才是它没在撒谎的证据。

注入缺陷一律调用 `benchmark.corruption.apply_corruption` 本尊——不手写"我以为的 level shift"。
算子必须对付真正的腐蚀，不是对付我对腐蚀的想象。
"""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from SelfEvolvingHarnessTS.benchmark.corruption import apply_corruption
from SelfEvolvingHarnessTS.fast_path import usable_ops
from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.operators import _provenance as prov
from SelfEvolvingHarnessTS.operators._common import robust_sigma
from SelfEvolvingHarnessTS.operators.registry import (
    ALIASES,
    OPERATOR_METADATA,
    OPERATOR_NAMES,
    canonicalize,
    get_operator,
)
from SelfEvolvingHarnessTS.operators.s1_impute import impute_ar, impute_linear, impute_ssm
from SelfEvolvingHarnessTS.operators.s1_outlier import hampel_filter, winsorize
from SelfEvolvingHarnessTS.operators.s1_structural import repair_level_shift

_NEW_OPERATORS = ("repair_level_shift", "hampel_filter", "impute_ssm", "impute_ar")
_PERIOD = 24


def _mae(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).mean())


def _seasonal(n: int = 600, noise: float = 0.5, slope: float = 0.02, seed: int = 3) -> np.ndarray:
    """趋势 + 季节 + 噪声——roster 的主频段（hourly/daily）长这样。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10.0 + slope * t + 3.0 * np.sin(2 * np.pi * t / _PERIOD) + rng.normal(0, noise, n)


# ══════════════════════ R1 repair_level_shift ══════════════════════
@pytest.mark.parametrize(
    "name,series",
    [
        ("trend+seasonal+noise", _seasonal()),
        ("pure_white_noise", np.random.default_rng(11).normal(0, 1, 500)),
        ("pure_linear_trend", np.linspace(0.0, 50.0, 500)),
        ("steep_trend_plus_noise", 0.2 * np.arange(500) + np.random.default_rng(12).normal(0, 1, 500)),
        ("strong_seasonal", 5 * np.sin(2 * np.pi * np.arange(600) / _PERIOD)),
    ],
)
def test_r1_identity_on_series_without_a_break(name, series):
    """① 无断点 → **逐位恒等**。

    容差写死为 0：本算子只在"检测到断层"时才动手，检测不到就该原样返回原数组的值。
    近似恒等（例如"顺手去了个趋势再加回来"）会引入浮点噪声，而那噪声会被下游判官当信号读。
    """
    out = repair_level_shift(series, period=_PERIOD)
    assert np.array_equal(out, series), f"{name}: 无断层输入被改动了"


def test_r1_repairs_an_injected_level_shift():
    """② 注入 benchmark 真实的 level_shift（dose .05，2σ_robust）后，修复误差显著下降。"""
    clean = _seasonal(n=500)
    damaged, repaired = [], []
    for seed in range(10):
        dirty = apply_corruption(clean, scenario="level_shift", dose=0.05, seed=seed + 11)
        fixed = repair_level_shift(dirty, period=_PERIOD)
        damaged.append(_mae(dirty, clean))
        repaired.append(_mae(fixed, clean))

    assert np.mean(repaired) < 0.25 * np.mean(damaged), (
        f"平均只消除了 {100 * (1 - np.mean(repaired) / np.mean(damaged)):.0f}% 的损伤"
    )
    # 逐 seed 都必须**至少不使情况变坏**——平均值会掩盖个别灾难，而灾难正是我们怕的。
    for i, (d, r) in enumerate(zip(damaged, repaired)):
        assert r <= d, f"seed{i}: 修复后比不修还差（{r:.4f} > {d:.4f}）"


def test_r1_reverse_test_does_not_flatten_a_trend():
    """③ **反噬测试**：纯趋势序列上不得把趋势斩平。

    这是本算子最容易犯的错：朴素的分段均值二分会把线性趋势看成一串小台阶，找出一堆假断点，
    然后把真实趋势"对齐"掉。防御是构造性的（设计矩阵含 [1, t] → 纯趋势的残差恒为 0 → 检验量
    恒为 0 → 一个断点都接受不了），不是靠阈值调出来的。斜率必须**逐位不变**。
    """
    rng = np.random.default_rng(5)
    for slope in (0.05, 0.25, 1.0, -0.4):
        trend = slope * np.arange(400) + rng.normal(0, 0.5, 400)
        out = repair_level_shift(trend, period=0)
        assert np.array_equal(out, trend), f"slope={slope}: 趋势序列被动了"


@pytest.mark.parametrize(
    "name,series",
    [
        # ⚠️ 这一组是 **Support-A discovery 回归测试**：v1 就死在这里。
        # 合成的"趋势+正弦+白噪"全绿，但真实 roster 上误触发 82%、平均损伤 1.78σ——因为那些
        # 合成序列**恰好就是设计矩阵假设的那个模型**。拿自己的假设当测试，什么也没测。
        # 下面每一条都取自真实 roster 里存在的形态：
        ("cumulative_counts_s_curve",                       # monash:covid_deaths（v1 上 69σ 重建误差）
         408.0 / (1.0 + np.exp(-0.06 * (np.arange(212) - 106.0)))),
        ("exponential_growth",
         np.exp(np.linspace(0.0, 6.0, 300))),
        ("saturating_concave",
         100.0 * (1.0 - np.exp(-np.linspace(0.0, 5.0, 300)))),
        ("weekday_weekend_hourly",                          # 真实交通/电力：周内周末电平本就不同
         (10.0 + 5.0 * np.sin(2 * np.pi * np.arange(720) / 24)
          - 3.0 * ((np.arange(720) // 24) % 7 >= 5))),
    ],
)
def test_r1_reverse_test_nonlinear_trends_are_not_chopped_into_fake_steps(name, series):
    """③c **反噬测试（真实形态）**：非线性趋势不得被读成一串假断层。

    这是 v1 的死因，也是本算子最贵的一课：**模型失配会被读成缺陷**。线性趋势基底对一条累计
    计数的 S 曲线严重失配 → 残差是系统性的 U 形 → sup-Chow 把失配读成"电平断层" → 分段对齐
    把整条曲线打成垃圾。

    修法是结构性的而非调参：趋势基底换成**连续**分段线性（hinge）——它柔到跟得住 S 曲线，
    但**连续基底无法表示跳变**，所以真断点依然留在残差里。柔性与检测力在这里不冲突。
    """
    out = repair_level_shift(series, period=24 if "hourly" in name else 7)
    damage = _mae(out, series) / max(robust_sigma(series), 1e-9)
    assert damage <= 0.05, f"{name}: 无断层的真实形态被改动了 {damage:.3f}σ"


def test_r1_reverse_test_survives_a_shift_riding_on_a_trend():
    """③b 趋势**与**断层同时存在：断层要修掉，趋势要留住。（分不开这两者的实现会在这里死。）"""
    n = 600
    t = np.arange(n)
    rng = np.random.default_rng(9)
    clean = 5.0 + 0.05 * t + 2.0 * np.sin(2 * np.pi * t / _PERIOD) + rng.normal(0, 0.4, n)
    dirty = apply_corruption(clean, scenario="level_shift", dose=0.05, seed=77)
    fixed = repair_level_shift(dirty, period=_PERIOD)

    assert _mae(fixed, clean) < 0.5 * _mae(dirty, clean)
    slope_before = np.polyfit(t, clean, 1)[0]
    slope_after = np.polyfit(t, fixed, 1)[0]
    assert abs(slope_after - slope_before) < 0.1 * abs(slope_before), "趋势被断层修复顺手削掉了"


# ══════════════════════ R3 hampel_filter ══════════════════════
def test_r3_identity_on_a_clean_series_with_no_outliers():
    """① 无离群 → 未命中的点**逐位不动**（点式替换语义：不是整段平滑）。"""
    clean = 5.0 * np.abs(np.sin(2 * np.pi * np.arange(600) / _PERIOD)) ** 4  # 尖峰季节，无噪
    assert np.array_equal(hampel_filter(clean), clean)

    flat = np.full(300, 7.0)                       # 常数序列：局部 MAD=0 → 零 MAD 守卫必须弃权
    assert np.array_equal(hampel_filter(flat), flat)


def test_r3_catches_injected_spikes_and_repairs_them():
    """② 注入 benchmark 真实的 spike（dose .03，6σ_robust）→ 命中率高 + 修复增益大。

    判据取自实测（文献默认 w=7/3σ 下：逐 seed 召回 72–89%，平均 83%；误差降到原来的 27%），
    留出余量后写死。**没有为了让判据好看去调窗口**：w=11 的召回是 94–100%，但对干净序列的
    附带损伤会从 0.015σ 涨到 0.026σ。这是一个真实的取舍，不是一个可以两头都要的旋钮；
    选文献默认、如实记录取舍，而不是选一个"在 benchmark 上好看"的值——后者就是把尺子当靶子。
    """
    clean = _seasonal()
    recalls, gains = [], []
    for seed in range(5):
        dirty = apply_corruption(clean, scenario="spike", dose=0.03, seed=seed + 31)
        fixed = hampel_filter(dirty)
        spiked = np.flatnonzero(dirty != clean)
        recalls.append(float(np.mean(fixed[spiked] != dirty[spiked])))
        gains.append(_mae(fixed, clean) / _mae(dirty, clean))

    assert np.mean(recalls) >= 0.75, f"spike 平均召回率仅 {np.mean(recalls):.0%}"
    assert min(recalls) >= 0.60, f"最差 seed 的召回率 {min(recalls):.0%} 说明算子不稳定"
    assert np.mean(gains) < 0.40, f"修复后误差仍有原来的 {np.mean(gains):.0%}"


@pytest.mark.parametrize(
    "name,series",
    [
        ("random_walk", np.cumsum(np.random.default_rng(21).normal(0, 1, 600))),
        ("heavy_noise", np.random.default_rng(22).normal(0, 5, 600)),
        ("ar1_phi_0.9", None),                     # 在测试体内构造（需递推）
    ],
)
def test_r3_reverse_test_false_kill_is_bounded_on_volatile_but_clean_series(name, series):
    """③ **反噬测试**：高波动但**无异常**的序列上，误杀有界。

    Hampel 的已知弱点：局部 MAD 在平坦处变小 → 正常的大波动会被当成离群。两条界：
      (a) 误杀比例 ≤ 10%（实测 6.5–8.5%）；
      (b) 附带损伤 ≤ 0.15 × 序列自身的 robust σ（实测 0.008–0.102σ）。
    界是 **σ 相对**的：绝对 MAE 在不同量纲的序列上没有可比性，而 roster 跨着 10kW 电表和
    万辆车流——这正是 benchmark 的 scale-relative 腐蚀所遵循的同一条原则。
    """
    if series is None:
        rng = np.random.default_rng(23)
        series = np.zeros(600)
        for i in range(1, 600):
            series[i] = 0.9 * series[i - 1] + rng.normal(0, 1)

    out = hampel_filter(series)
    false_kill = float((out != series).mean())
    damage = _mae(out, series) / robust_sigma(series)
    assert false_kill <= 0.10, f"{name}: 误杀 {false_kill:.1%}"
    assert damage <= 0.15, f"{name}: 附带损伤 {damage:.3f}σ"


def test_r3_does_less_collateral_damage_than_winsorize_on_structured_series():
    """③b 有参照物的界：在**有结构**的干净序列上，hampel 的附带损伤不超过 winsorize
    （一个已在冻结池里的算子）。

    ⚠️ 这个不等式**只在有结构的序列上成立**，测试的范围如实限定在那里。纯白噪声上它反过来：
    hampel 0.102σ vs winsorize 0.040σ——因为"局部中值"在毫无局部结构可言的序列上本来就是个
    差预测，而 winsorize 只钳两条尾巴。这是算子的真实性质，写在这里而不是藏起来；roster 里
    的序列有趋势有季节，落在不等式成立的那一侧，但这个条件是被声明的，不是被假设的。
    """
    rng = np.random.default_rng(21)
    for name, series in [
        ("seasonal", _seasonal()),
        ("random_walk", np.cumsum(rng.normal(0, 1, 600))),
    ]:
        assert _mae(hampel_filter(series), series) <= _mae(winsorize(series), series), name


# ══════════════════════ R2 impute_ssm / impute_ar ══════════════════════
@pytest.mark.parametrize("imputer", [impute_ssm, impute_ar])
def test_r2_identity_when_nothing_is_missing(imputer):
    """① 无缺失 → 逐位恒等。imputer 不是 denoiser。"""
    series = _seasonal(n=300)
    assert np.array_equal(imputer(series), series)


@pytest.mark.parametrize("imputer", [impute_ssm, impute_ar])
def test_r2_preserves_every_observed_value(imputer):
    """imputer 契约的**核心**：只写缺失位置，已观测点逐位不变。

    这不是形式主义。`impute_ssm` 底下的 Kalman 平滑器在**已观测点上也会输出一个"平滑值"**
    （它顺带去了噪），整段返回它 = 一个伪装成 imputer 的 denoiser。impute_fft 就犯过这个错
    （S0.7 修复），而它伪造的正是"插补 vs 去噪"这两个动作的身份区别——那是 benchmark 用来
    区分机制的坐标轴本身。
    """
    series = _seasonal(n=300)
    damaged = series.copy()
    damaged[50:70] = np.nan
    damaged[200] = np.nan
    missing = np.isnan(damaged)

    out = imputer(damaged, period=_PERIOD) if imputer is impute_ssm else imputer(damaged, period=_PERIOD)
    assert np.array_equal(out[~missing], damaged[~missing]), "已观测点被改动 → 这不是 imputer"
    assert np.isfinite(out).all()


@pytest.mark.parametrize("imputer", [impute_ssm, impute_ar])
def test_r2_model_predictive_imputation_beats_linear_interpolation(imputer):
    """② 在有周期结构的序列上，模型预测族必须显著胜过插值族——否则它不配占一个动作名额。

    这正是它们存在的理由：一个 30 点的 block 缺口横跨一个多周期，线性插补拉一条弦切过去，
    把季节结构整段抹平；模型知道周期，能把它续出来。
    """
    series = _seasonal(n=500, noise=0.4, slope=0.01, seed=200)
    damaged = series.copy()
    damaged[100:130] = np.nan                     # 长 block 缺口（> 一个周期）
    damaged[300:312] = np.nan
    missing = np.isnan(damaged)

    model_mae = _mae(imputer(damaged, period=_PERIOD)[missing], series[missing])
    linear_mae = _mae(impute_linear(damaged)[missing], series[missing])
    assert model_mae < 0.6 * linear_mae, (
        f"{imputer.__name__}: MAE {model_mae:.4f} 对线性插补的 {linear_mae:.4f} 没有显著优势"
    )


def test_r2_impute_ar_order_must_reach_the_seasonal_lag():
    """③ **反噬测试（AR）**：阶数够不到季节滞后时，AR 退化成短记忆模型。

    自动阶数（order=0 → max(8, period)）必须显著胜过写死的 AR(8)。这条测试守的是一个**机制**
    失效而非数值失效：AR(8) 在 period=24 的序列上**根本看不见季节周期**，它会安静地给出一个
    平庸的答案——没有异常、没有报错，只是这个算子不再是它自称的那个东西。
    """
    series = _seasonal(n=500, noise=0.4, slope=0.01, seed=201)
    damaged = series.copy()
    damaged[100:130] = np.nan
    missing = np.isnan(damaged)

    auto = _mae(impute_ar(damaged, period=_PERIOD)[missing], series[missing])
    short = _mae(impute_ar(damaged, order=8)[missing], series[missing])
    assert auto < 0.5 * short, f"自动阶数 {auto:.4f} 没能显著胜过 AR(8) 的 {short:.4f}"


def test_r2_impute_ssm_hard_fails_when_statsmodels_is_missing(monkeypatch):
    """③ **反噬测试（SSM）**：依赖缺失时**硬失败**，绝不静默降级。

    这条是整个 E-3.3 里最重要的一条测试。`impute_kalman` 这个名字在本仓库历史上指的一直是
    **EMA**——一个从来没有 Kalman 的 Kalman。那个事故的机理就是"依赖不在 → 悄悄换一个算子顶上，
    名字不变"。台账记着跑了状态空间，实际跑的是指数平滑，而 router 从这条假记录里学动作标签。
    所以 impute_ssm 的契约是 dependency_policy="hard_fail"：statsmodels 不在，它**抛异常**，
    不返回任何值。
    """
    real_import = builtins.__import__

    def no_statsmodels(name, *args, **kwargs):
        if name.startswith("statsmodels"):
            raise ImportError("simulated: statsmodels not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_statsmodels)

    damaged = _seasonal(n=200).copy()
    damaged[50:60] = np.nan
    with pytest.raises(ImportError, match="statsmodels"):
        impute_ssm(damaged, period=_PERIOD)       # 必须炸——不许返回一个 EMA 填出来的数组


def test_r2_impute_ssm_never_falls_back_to_an_exponential_smoother():
    """静态守卫：impute_ssm 的任何回退路径都不得指向 impute_ema / smooth_ema。

    退化输入（观测点过少、平滑器异常）下它可以显式记账回退到 **impute_linear**——那是一个
    诚实的、被台账记下的降级。但**永远不能**回退到指数平滑：那正好是 `impute_kalman` 误名
    所声称的东西，一旦发生，两个算子在台账里就无法区分了。
    """
    ledger_targets = set()
    prov.start_recording()
    try:
        for n_obs in (3, 5, 7):                   # 观测点少于 _SSM_MIN_OBSERVED → 走退化分支
            x = np.full(40, np.nan)
            x[:n_obs] = np.arange(n_obs, dtype=float)
            impute_ssm(x)
        for entry in prov.get_ledger():
            if entry["requested"] == "impute_ssm":
                ledger_targets.add(entry["effective"])
    finally:
        prov.stop_recording()

    assert ledger_targets, "退化输入下 impute_ssm 一次账都没记 → 静默回退"
    assert not (ledger_targets & {"impute_ema", "smooth_ema"}), (
        f"impute_ssm 回退到了指数平滑：{ledger_targets}"
    )


# ══════════════════════ 契约 / 命名 / 动作面 ══════════════════════
def test_new_operator_names_do_not_reuse_a_historical_meaning():
    """命名红线：新算子不得复用任何在历史上指过别的实现的名字。

    `impute_kalman` 在旧 trace 里指 impute_ema。真的状态空间插补必须叫 **impute_ssm**：
    同一个名字在旧记录里指 EMA、在新记录里指 Kalman，会让任何跨版本的算子身份审计失效。
    旧 alias 保留（旧 artifact 重放不破坏）但标记 deprecated。
    """
    for name in _NEW_OPERATORS:
        assert name in OPERATOR_NAMES
        assert name not in ALIASES
        assert not OPERATOR_METADATA[name].get("is_alias")

    assert canonicalize("impute_kalman") == "impute_ema"      # 旧含义不变（重放保证）
    assert get_operator("impute_ssm") is not get_operator("impute_kalman")
    for alias in ALIASES:
        assert OPERATOR_METADATA[alias]["deprecated"] is True


@pytest.mark.parametrize("name", sorted(_NEW_OPERATORS))
def test_new_operators_carry_a_complete_contract(name):
    meta = OPERATOR_METADATA[name]
    for field in ("allowed_tasks", "destructive", "preserves_observed", "reversible",
                  "changes_target_space", "requires_dependency", "fallback_policy",
                  "dependency_policy"):
        assert field in meta, f"{name} 缺契约字段 {field}"
    assert meta["changes_target_space"] is False              # 都不改目标空间 → 可进 action surface


def test_dependency_policy_distinguishes_hard_fail_from_recorded_fallback():
    """新契约字段的存在理由：denoise_stl 与 impute_ssm 的 requires_dependency 都是 statsmodels，
    但正确行为相反——前者记账回退 savgol，后者硬失败。旧契约说不出这个区别。"""
    assert OPERATOR_METADATA["impute_ssm"]["dependency_policy"] == "hard_fail"
    assert OPERATOR_METADATA["denoise_stl"]["dependency_policy"] == "recorded_fallback"
    for name in OPERATOR_NAMES:
        meta = OPERATOR_METADATA[name]
        if meta["requires_dependency"]:
            assert meta["dependency_policy"] in ("hard_fail", "recorded_fallback"), name
        else:
            assert meta["dependency_policy"] is None, name


def test_destructive_new_operators_are_physically_banned_from_anomaly():
    """repair_level_shift 与 hampel_filter 在 anomaly 下**物理禁用**。

    这是 C1（H* = f(pattern, task)）最干净的一个例证，不是保守主义：level shift 与 spike
    在异常检测里**就是要检出的信号本身**，"修"掉它等于把标签擦了。同一个算子，forecast 下
    是修复、anomaly 下是破坏——动作的好坏由 task 决定，这正是本项目的核心主张。
    """
    harness = HarnessState.from_minimal()
    anomaly_ops = set(usable_ops(harness, "anomaly_detection"))
    assert "repair_level_shift" not in anomaly_ops
    assert "hampel_filter" not in anomaly_ops

    forecast_ops = set(usable_ops(harness, "forecast"))
    assert {"repair_level_shift", "hampel_filter", "impute_ssm", "impute_ar"} <= forecast_ops
    # 插补类不 destructive → anomaly 下仍可用（anomaly 也要杀 NaN）
    assert {"impute_ssm", "impute_ar"} <= anomaly_ops
