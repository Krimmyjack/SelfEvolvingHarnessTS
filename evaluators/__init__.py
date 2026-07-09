"""evaluators/ — R5/R9/R10：proxy + grounded 两层，三任务三协议。

裁决纪律（不变量 #1/#2）：grounded 是唯一 accept 裁判；proxy 只负向预筛；Role B 只 log 不 gate。
本机无 torch → grounded 走 sklearn + ROCKET 式 frozen+probe（E2c 必要条件的 torch-free 复刻）。

接口：get_evaluator(task).evaluate(ready_batch, layer="proxy"|"grounded") -> val_loss（越低越好）。
"""
from .base import (
    Evaluator, get_evaluator,
    ForecastSample, AnomalySample, ClassifySample,
    L_WIN, H_FORECAST, STRIDE, WIN_CLF, STRIDE_CLF, ANOM_FRAC, ANOM_TOL,
)
from .frozen_probe import (
    FrozenProbe, set_frozen_encoder, pretrain_encoder_real, get_frozen_encoder, load_frozen_encoder,
)
from .grounded_forecast import (
    forecast_grounded, seasonal_naive_floor, build_windows, set_forecast_target, set_forecast_substrate,
)
from .grounded_anomaly import anomaly_grounded, anomaly_recall, detect
from .grounded_classify import (
    classify_grounded, classify_inception, set_classify_substrate, get_classify_substrate,
)
from .rocket_probe import classify_grounded_rocket
from .role_a_proxy import forecast_proxy, anomaly_proxy, classify_proxy
from .report_target import (
    report_perf, delta_perf, perf_multi, disjoint_targets, FORECAST_TARGETS, CLASSIFY_TARGETS,
)
from . import role_b_metrics
from .calibration import spearman_gate, CalibrationResult
from .readiness import readiness_score, is_ready, aggregate_time_to_readiness

__all__ = [
    "readiness_score", "is_ready", "aggregate_time_to_readiness",
    "Evaluator", "get_evaluator",
    "ForecastSample", "AnomalySample", "ClassifySample",
    "FrozenProbe", "set_frozen_encoder", "pretrain_encoder_real", "get_frozen_encoder",
    "load_frozen_encoder",
    "forecast_grounded", "seasonal_naive_floor", "build_windows",
    "set_forecast_target", "set_forecast_substrate",
    "anomaly_grounded", "anomaly_recall", "detect",
    "classify_grounded", "classify_inception", "classify_grounded_rocket",
    "set_classify_substrate", "get_classify_substrate",
    "forecast_proxy", "anomaly_proxy", "classify_proxy",
    "report_perf", "delta_perf", "perf_multi", "disjoint_targets",
    "FORECAST_TARGETS", "CLASSIFY_TARGETS",
    "role_b_metrics", "spearman_gate", "CalibrationResult",
    "L_WIN", "H_FORECAST", "STRIDE", "WIN_CLF", "STRIDE_CLF", "ANOM_FRAC", "ANOM_TOL",
]
