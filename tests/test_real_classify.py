"""classify 端到端新增件的单测（ECG5000 锚 + ROCKET 确定性判官 + 判官↔报告器分离）。

运行：  python -m SelfEvolvingHarnessTS.tests.test_real_classify   （cwd=Agent）

绝大多数用例**免联网**（手造 RealClassSignal / 合成可分批）；仅 test_ecg_cache_smoke 在
本地缓存 `data/_artifacts/ecg5000.npz` 存在时跑（不存在则跳过，不触发下载）。
"""
from __future__ import annotations

import pathlib

import numpy as np

from SelfEvolvingHarnessTS.evaluators import (
    ClassifySample, classify_grounded, classify_inception, classify_grounded_rocket,
    set_classify_substrate, get_classify_substrate, disjoint_targets, report_perf,
)
from SelfEvolvingHarnessTS.data import (
    RealClassSignal, make_real_classify_batch, build_real_classify_corpus, CLASSIFY_PRESETS,
)
from SelfEvolvingHarnessTS.harness import HarnessState
from SelfEvolvingHarnessTS.slow_path import BatchBuilder


def _two_class_batch(n=24, seed=0):
    """易分的 2 类：类0=平滑正弦；类1=同正弦+周期尖峰（形态判别，含高频）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(140)
    out = []
    for i in range(n):
        base = np.sin(2 * np.pi * t / 30)
        if i % 2 == 0:
            x, lab = base + rng.normal(0, 0.1, 140), 0
        else:
            x = base.copy(); x[::17] += 2.5; x = x + rng.normal(0, 0.1, 140); lab = 1
        out.append(ClassifySample(x.astype(float), lab))
    return out


# ── 1. ROCKET 判官确定性：同 seed 完全一致（σ=0），且能分易分集 ──────────────
def test_rocket_deterministic_and_separates():
    b = _two_class_batch()
    r0, r0b, r1 = (classify_grounded_rocket(b, seed=0), classify_grounded_rocket(b, seed=0),
                   classify_grounded_rocket(b, seed=1))
    assert np.isfinite(r0)
    assert r0 == r0b                       # 核 seed 固定 + LogReg 确定 → σ=0
    assert r0 < 0.4                         # 易分集上 CE 应很低


# ── 2. substrate 派发开关：classify_grounded 按全局底座路由，默认 inception ──
def test_classify_substrate_switch():
    assert get_classify_substrate() == "inception"
    b = _two_class_batch()
    try:
        set_classify_substrate("rocket")
        # rocket 确定性 → 精确相等可验证路由确实走 rocket
        assert classify_grounded(b, seed=0) == classify_grounded_rocket(b, seed=0)
        set_classify_substrate("inception")
        # inception σ>0（from-scratch），只验路由到 inception 路径且出有限值
        assert np.isfinite(classify_grounded(b, seed=0))
    finally:
        set_classify_substrate("inception")   # 复位，勿污染其它测试


# ── 3. 判官↔报告器分离：judge=rocket → 独立报告器集去掉 rocket ────────────────
def test_disjoint_classify():
    assert disjoint_targets("rocket", ["inception", "rocket"]) == ["inception"]
    assert disjoint_targets("inception", ["inception", "rocket"]) == ["rocket"]


# ── 4. 报告器对含 NaN(miss) 的 ready 仍出有限 perf（_fillna 生效）────────────
def test_report_perf_handles_missing():
    b = _two_class_batch()
    for s in b[::3]:                            # 注入散点 NaN 模拟 miss cell 的 raw
        s.ready[5:9] = np.nan
    p = report_perf(b, "classification", target="inception", seed=0)
    assert np.isfinite(p) and 0.0 < p <= 1.0


# ── 5. classify 语料工厂：RealClassSignal → classify RawSeries(带 label) ──────
def test_classify_corpus_factory():
    rng = np.random.default_rng(0)
    sigs = [RealClassSignal("syn", str(i), i % 2,
                            (np.sin(2 * np.pi * np.arange(140) / 30) + rng.normal(0, 0.05, 140)))
            for i in range(20)]
    batch = make_real_classify_batch(sigs, "G_hi_full", n_per_signal=1)
    assert len(batch) == 20
    assert all(rs.task == "classification" and rs.label in (0, 1) for rs in batch)

    corpus = build_real_classify_corpus(sigs, n_per_signal=1)
    assert len(corpus) == 20 * len(CLASSIFY_PRESETS)
    bb = BatchBuilder(HarnessState.from_minimal(), n_min=4)
    cells = {bb.add_raw_series(rs) for rs in corpus}
    assert all(c.startswith("classification|") for c in cells)
    # missing 轴必由 preset 决定（full preset → full bin），SNR 轴可随信号结构
    assert any("full" in c for c in cells)


# ── 6. （可选，缓存存在才跑）ECG5000 真实锚装配到 classify cell ────────────────
def test_ecg_cache_smoke():
    from SelfEvolvingHarnessTS.data.load_ecg5000 import DEFAULT_CACHE
    if not pathlib.Path(DEFAULT_CACHE).exists():
        print("    [skip] 无 ECG5000 缓存（不触发下载）")
        return
    from SelfEvolvingHarnessTS.data import load_class_signals
    sigs = load_class_signals(max_signals=40)
    assert len(sigs) == 40 and sigs[0].clean.size > 0
    bb = BatchBuilder(HarnessState.from_minimal(), n_min=4)
    for rs in build_real_classify_corpus(sigs, n_per_signal=1):
        bb.add_raw_series(rs)
    clf_cells = [c for c in bb.pools if c.startswith("classification|")]
    assert clf_cells, "ECG 语料未落入任何 classification cell"


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
