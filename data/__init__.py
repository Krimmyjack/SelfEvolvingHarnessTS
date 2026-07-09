"""data/ — 合成数据生成 + 退化注入 + 真实锚（plan.md §2）。

Phase 0：synthetic_gen.py 产 (clean, degraded, label) 三元组（forecast/anomaly/classify），
供 evaluators 与 slow_path 测试。Phase 1+：corruptions.py、load_real.py（跨 domain 锚，验 C3）。
"""
from .synthetic_gen import (
    RawSeries, make_forecast_batch, make_anomaly_batch, make_classify_batch, PATTERNS,
)
from .load_real import (
    RealSignal, RealClassSignal, load_signals, load_class_signals,
    build_real_corpus, build_real_classify_corpus,
    make_real_forecast_batch, make_real_anomaly_batch, make_real_classify_batch, split_encoder_eval,
    FORECAST_PRESETS, ANOMALY_PRESETS, CLASSIFY_PRESETS,
)

__all__ = [
    "RawSeries", "make_forecast_batch", "make_anomaly_batch", "make_classify_batch", "PATTERNS",
    "RealSignal", "RealClassSignal", "load_signals", "load_class_signals",
    "build_real_corpus", "build_real_classify_corpus",
    "make_real_forecast_batch", "make_real_anomaly_batch", "make_real_classify_batch",
    "split_encoder_eval", "FORECAST_PRESETS", "ANOMALY_PRESETS", "CLASSIFY_PRESETS",
]
