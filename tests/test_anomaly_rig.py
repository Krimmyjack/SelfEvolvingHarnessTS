"""P2 契约测试：最小 anomaly rig（注入协议 + 固定 residual z-score 检测器 + F1/AUROC 判官）。

rig 是判官不是研究对象（一周封顶纪律）；确定性（σ=0）、NaN 安全、
"平滑毁 recall" 的符号翻转机制在单测层面直接可证。
"""
import numpy as np
from scipy.ndimage import median_filter

from SelfEvolvingHarnessTS.evaluators.anomaly_rig import (
    DETECTOR_SPEC,
    anomaly_metrics,
    anomaly_readiness_eval,
    inject_anomalies,
    make_anomaly_slice,
    residual_zscore_scores,
)


def _clean(n=256, period=24):
    return np.sin(2 * np.pi * np.arange(n) / period)


def test_injection_labels_and_amplitude():
    x, labels = inject_anomalies(_clean(), rng=np.random.default_rng(0),
                                 n_points=5, n_contextual=1, segment_len=5)
    assert labels.dtype == bool and labels.shape == x.shape
    assert labels.sum() == 5 + 5                       # 5 点异常 + 1 段×5 点 contextual
    assert np.all(np.abs(x[labels] - _clean()[labels]) > 1.0)
    assert np.array_equal(x[~labels], _clean()[~labels])


def test_detector_finds_spikes_on_noisy_series():
    rng = np.random.default_rng(1)
    base = _clean() + rng.normal(0.0, 0.15, 256)
    x, labels = inject_anomalies(base, rng=rng, n_points=6, n_contextual=0)
    m = anomaly_readiness_eval(x, labels)
    assert m["recall"] >= 0.8
    assert m["F1"] >= 0.5
    assert m["AUROC"] >= 0.9


def test_smoothing_destroys_recall():
    # 符号翻转机制本体：median 平滑抹掉 spike → 残差跌破 raw 上冻结标定的告警线 → recall 崩
    #（若让检测器随 artifact 自标定，平滑会把 MAD 一起压扁、z 反而爆炸——rig 首轮实测教训，
    #  故 raw_reference 是协议的一部分）
    rng = np.random.default_rng(2)
    base = _clean() + rng.normal(0.0, 0.15, 256)
    x, labels = inject_anomalies(base, rng=rng, n_points=6, n_contextual=0)
    before = anomaly_readiness_eval(x, labels, raw_reference=x)
    after = anomaly_readiness_eval(median_filter(x, size=9, mode="nearest"), labels, raw_reference=x)
    assert before["recall"] >= 0.8
    assert after["recall"] <= 0.2
    assert after["F1"] < before["F1"] - 0.4


def test_nan_safe_scores_and_metrics():
    x, labels = inject_anomalies(_clean(), rng=np.random.default_rng(3),
                                 n_points=4, n_contextual=0)
    x[50:60] = np.nan
    scores = residual_zscore_scores(x)
    assert np.all(np.isfinite(scores))
    assert np.all(scores[50:60] == 0.0)                # NaN 位置不可 flag
    m = anomaly_readiness_eval(x, labels)
    assert np.isfinite(m["F1"]) and np.isfinite(m["AUROC"])


def test_metrics_edge_no_flags_no_crash():
    labels = np.zeros(64, dtype=bool)
    labels[10] = True
    m = anomaly_metrics(np.zeros(64), np.zeros(64, dtype=bool), labels)
    assert m["F1"] == 0.0 and m["recall"] == 0.0 and m["precision"] == 0.0


def test_auroc_perfect_separation():
    labels = np.zeros(100, dtype=bool)
    labels[:10] = True
    scores = np.zeros(100)
    scores[:10] = 5.0
    m = anomaly_metrics(scores, scores > 1.0, labels)
    assert m["AUROC"] == 1.0
    assert m["recall"] == 1.0


def test_injection_fails_loud_when_capacity_exceeded():
    # code-review：塞不下请求数量时不得静默少注（分母漂移），须 fail-loud
    import pytest
    with pytest.raises(ValueError):
        inject_anomalies(_clean(64), rng=np.random.default_rng(0),
                         n_points=200, n_contextual=0)


def test_slice_fails_loud_when_no_missing_slot():
    import pytest
    with pytest.raises(ValueError, match="缺失块"):
        make_anomaly_slice(n_series=2, seed=1, miss_len=240)


def test_slice_deterministic_and_missing_not_on_anomalies():
    a = make_anomaly_slice(n_series=4, seed=7)
    b = make_anomaly_slice(n_series=4, seed=7)
    assert [r["uid"] for r in a] == [r["uid"] for r in b]
    assert np.array_equal(a[1]["x"], b[1]["x"], equal_nan=True)
    for row in a:
        assert row["labels"].sum() > 0
        assert len(row["future_clean"]) == 24
        assert not np.any(np.isnan(row["x"]) & row["labels"])   # 缺失不落在异常点
    cells = {row["cell"] for row in a}
    assert cells == {"anomaly|snrHigh|full", "anomaly|snrLow|miss"}
    assert DETECTOR_SPEC["detector"] == "residual_zscore"
