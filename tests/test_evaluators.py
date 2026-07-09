"""Phase 1 前置验证：evaluators/ 模块（proxy/grounded 两层 + frozen_probe + calibration）。

运行：  python -m SelfEvolvingHarnessTS.tests.test_evaluators   （cwd=Agent）
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from SelfEvolvingHarnessTS.evaluators import (
    get_evaluator, ForecastSample, AnomalySample, ClassifySample,
    forecast_grounded, seasonal_naive_floor, anomaly_grounded, anomaly_recall,
    classify_grounded, forecast_proxy, anomaly_proxy, spearman_gate, FrozenProbe,
    role_b_metrics,
)
from SelfEvolvingHarnessTS.data import make_forecast_batch, make_anomaly_batch, make_classify_batch


def _fill(x):
    x = np.asarray(x, float).copy()
    m = np.isnan(x)
    if m.any():
        idx = np.arange(x.size)
        x[m] = np.interp(idx[m], idx[~m], x[~m])
    return x


# ── 1. frozen_probe 确定性：σ_A ≈ 0（grounded 默认底座的核心）──────────────
def test_frozen_probe_deterministic():
    fb = make_forecast_batch("P1", 8)
    good = [ForecastSample(s.clean_history, s.future, s.obs_scale, s.period) for s in fb]
    l0 = forecast_grounded(good, seed=0, substrate="frozen")
    l1 = forecast_grounded(good, seed=1, substrate="frozen")
    l2 = forecast_grounded(good, seed=2, substrate="frozen")
    assert np.isfinite(l0)
    assert l0 == l1 == l2                       # 冻结核 + Ridge 闭式 → seed 无关 → σ_A=0


# ── 2. grounded forecast 区分 good vs bad ready ───────────────────────────
def test_forecast_grounded_discriminates():
    rng = np.random.default_rng(0)
    fb = make_forecast_batch("P1", 10)
    good = [ForecastSample(s.clean_history, s.future, s.obs_scale, s.period) for s in fb]
    bad = [ForecastSample(s.clean_history + rng.normal(0, 3.0, s.clean_history.size),
                          s.future, s.obs_scale, s.period) for s in fb]
    g_good, g_bad = forecast_grounded(good), forecast_grounded(bad)
    assert g_good < g_bad                        # 干净就绪 → 下游学得更好 → nRMSE 更低
    floor = seasonal_naive_floor(good)
    assert np.isfinite(floor) and floor > 0


# ── 3. grounded anomaly：保 spike vs 平滑 ────────────────────────────────
def test_anomaly_grounded_smoothing_hurts():
    ab = make_anomaly_batch("P1", 10)
    preserved = [AnomalySample(_fill(s.anomaly_input), s.anomaly_positions) for s in ab]
    smoothed = [AnomalySample(median_filter(_fill(s.anomaly_input), size=21, mode="nearest"),
                              s.anomaly_positions) for s in ab]
    loss_pres, loss_smooth = anomaly_grounded(preserved), anomaly_grounded(smoothed)
    assert anomaly_recall(preserved) > anomaly_recall(smoothed)
    assert loss_pres < loss_smooth               # 平滑削 spike → recall↓ → loss↑


# ── 4. calibration：平滑强度扫描上 proxy↔grounded 同向 → usable ───────────
def test_calibration_spearman_usable():
    ab = make_anomaly_batch("P1", 12)
    proxy_losses, grounded_losses = [], []
    for w in (1, 5, 11, 21, 31):                 # 5 个"变体"（平滑窗口递增）
        ready = [(_fill(s.anomaly_input) if w == 1
                  else median_filter(_fill(s.anomaly_input), size=w, mode="nearest"))
                 for s in ab]
        samp = [AnomalySample(r, s.anomaly_positions) for r, s in zip(ready, ab)]
        proxy_losses.append(anomaly_proxy(samp))
        grounded_losses.append(anomaly_grounded(samp))
    res = spearman_gate(proxy_losses, grounded_losses)
    assert res.n == 5 and res.usable and res.spearman >= res.tau


# ── 5. grounded classify 协议跑通，CE 有效 ────────────────────────────────
def test_classify_grounded_runs():
    cb = make_classify_batch(n_per_class=6)
    samp = [ClassifySample(median_filter(_fill(s.degraded), size=5, mode="nearest"), s.label)
            for s in cb]
    ce = classify_grounded(samp, seed=0)
    assert np.isfinite(ce) and 0.0 < ce < 1.3    # 协议产出有效 CE（随机基线 ln3≈1.10）


# ── 6. get_evaluator 派发 + 两层皆出有限 val_loss + alias ─────────────────
def test_dispatch_and_layers():
    fb = make_forecast_batch("P1", 8)
    samp = [ForecastSample(s.clean_history, s.future, s.obs_scale, s.period) for s in fb]
    ev = get_evaluator("forecast")
    assert np.isfinite(ev.evaluate(samp, "proxy")) and np.isfinite(ev.evaluate(samp, "grounded"))
    assert get_evaluator("anomaly").task_type == "anomaly_detection"   # alias 归一
    assert get_evaluator("classify").task_type == "classification"
    try:
        ev.evaluate(samp, "bogus"); assert False
    except ValueError:
        pass


# ── 7. Role B per-sample 指标（log 用）：平滑降 spike_preservation ─────────
def test_role_b_metrics():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300); x[150] = 12.0
    sp_raw = role_b_metrics.spike_preservation(x)
    sp_smooth = role_b_metrics.spike_preservation(median_filter(x, size=11, mode="nearest"))
    assert sp_raw > sp_smooth
    assert role_b_metrics.smoothness(np.cumsum(rng.normal(0, 1, 200))) >= 0


# ── 8. 集成：fast_path 产 ready → evaluator 裁决（端到端打通 Phase0↔Phase1）─
def test_integration_fastpath_to_evaluator():
    from SelfEvolvingHarnessTS.harness import HarnessState
    from SelfEvolvingHarnessTS.fast_path import process
    h = HarnessState.from_minimal()
    fb = make_forecast_batch("P1", 8)
    samp = []
    for s in fb:
        _rec, art = process(s.history, "forecast", h)
        samp.append(ForecastSample(art, s.future, s.obs_scale, s.period))
    g = forecast_grounded(samp)
    assert np.isfinite(g) and g > 0


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
