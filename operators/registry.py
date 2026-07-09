"""operators/registry.py — 算子注册表（单一真源）。

OPERATOR_SPECS 同时给出 callable（TOOL_REGISTRY，给 sandbox 执行）与 metadata
（OPERATOR_METADATA，给 harness.L2.operator_registry 配置面）。harness/layers.py 从这里
取 metadata，避免名字清单两处漂移。`tags` 直接就是 L2 算子选择元数据（smoothing/destructive/
scale_change，对应 plan.md A.3 AdaCTS registry 借鉴）。

S0.7-6 Registry 契约（A-29e，评审第九轮）：
  1. **canonical 正名 + 兼容 alias**：`fill_gaps`≡`impute_linear`（重复）、`impute_kalman`/
     `kalman_filter` 实为 EMA（误名）→ canonical 名为 `impute_linear`/`impute_ema`/`smooth_ema`；
     旧 ID 全部保留在 ALIASES/TOOL_REGISTRY，旧 artifact/模板按旧名重放不破坏，执行时
     provenance 台账同录 requested/canonical（reason="compat_alias"）。
  2. **契约元数据**（每 canonical 算子）：allowed_tasks / destructive / preserves_observed /
     reversible / changes_target_space / requires_dependency / fallback_policy——新算子必须
     从第一天带契约入池（E-3.3 扩充池前置）。
  3. **task 级物理过滤**：`fast_path.usable_ops` 按 allowed_tasks 硬过滤——anomaly 物理禁
     smoothing/destructive（保 spike/changepoint），不再只靠模板 ban。
"""
from __future__ import annotations

from typing import Callable, Dict

from . import s1_impute, s1_denoise, s1_outlier, s1_decompose, s2_align, s3_shape
from ._common import BOUNDARY_MODES  # noqa: F401（S0.7-8 边界语义指纹，随 registry 一并落 provenance）
from ._provenance import record as _prov_record

_ALL_TASKS = ("forecast", "classification", "anomaly_detection")
_NON_ANOMALY = ("forecast", "classification")   # 平滑/删改类：物理禁 anomaly（毁 spike/changepoint 信号）

_CONTRACT_BASE = {
    "allowed_tasks": _ALL_TASKS,
    "destructive": False,              # 删改观测点取值（离群置换/裁剪）
    "preserves_observed": False,       # imputer 契约：只写缺失位置、保留观测
    "reversible": False,               # 可由记录的参数逆变换（如仿射归一化）
    "changes_target_space": False,     # 改变下游目标空间（预测须逆变换才能对齐原 future）
    "requires_dependency": None,       # 非 numpy 依赖（缺失时按 fallback_policy 显式回退）
    "fallback_policy": "none",         # "none" | "explicit_record→<op>" | "numpy_equivalent"
}


def _c(**kw) -> dict:
    d = dict(_CONTRACT_BASE)
    d.update(kw)
    return d


# (name, category, stage, tags, fn, shape_changing, contract) —— canonical 算子（单一真源）
OPERATOR_SPECS = [
    ("impute_linear", "impute", "s1", [], s1_impute.impute_linear, False,
     _c(preserves_observed=True)),
    ("impute_fft", "impute", "s1", [], s1_impute.impute_fft, False,
     _c(preserves_observed=True, fallback_policy="explicit_record→impute_linear")),
    ("impute_ema", "impute", "s1", [], s1_impute.impute_ema, False,
     _c(preserves_observed=True)),
    ("period_complete", "impute", "s1", [], s1_impute.period_complete, False,
     _c(preserves_observed=True)),
    ("denoise_savgol", "denoise", "s1", ["smoothing"], s1_denoise.denoise_savgol, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="scipy", fallback_policy="numpy_equivalent")),
    ("denoise_wavelet", "denoise", "s1", ["smoothing"], s1_denoise.denoise_wavelet, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="pywt",
        fallback_policy="explicit_record→denoise_savgol")),
    ("denoise_median", "denoise", "s1", ["smoothing"], s1_denoise.denoise_median, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="scipy", fallback_policy="numpy_equivalent")),
    ("smooth_ma", "denoise", "s1", ["smoothing"], s1_denoise.smooth_ma, False,
     _c(allowed_tasks=_NON_ANOMALY)),   # 纯 numpy（无依赖），symmetric 边界；F0 剂量维
    ("denoise_stl", "denoise", "s1", ["smoothing"], s1_denoise.denoise_stl, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="statsmodels",
        fallback_policy="explicit_record→denoise_savgol")),
    ("winsorize", "outlier", "s1", ["destructive"], s1_outlier.winsorize, False,
     _c(allowed_tasks=_NON_ANOMALY, destructive=True)),
    ("outlier_iqr", "outlier", "s1", ["destructive"], s1_outlier.outlier_iqr, False,
     _c(allowed_tasks=_NON_ANOMALY, destructive=True)),
    ("outlier_mad", "outlier", "s1", ["destructive"], s1_outlier.outlier_mad, False,
     _c(allowed_tasks=_NON_ANOMALY, destructive=True)),
    ("stl_decompose", "decompose", "s1", [], s1_decompose.stl_decompose, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="statsmodels",
        fallback_policy="explicit_record→denoise_savgol")),
    ("fft_decompose", "decompose", "s1", [], s1_decompose.fft_decompose, False,
     _c(allowed_tasks=_NON_ANOMALY)),
    ("smooth_ema", "decompose", "s1", ["smoothing"], s1_decompose.smooth_ema, False,
     _c(allowed_tasks=_NON_ANOMALY)),
    ("resample_uniform", "align", "s2", [], s2_align.resample_uniform, False,
     _c(preserves_observed=True)),
    ("znorm", "shape", "s3", ["scale_change"], s3_shape.znorm, False,
     _c(reversible=True, changes_target_space=True)),
    ("minmax_norm", "shape", "s3", ["scale_change"], s3_shape.minmax_norm, False,
     _c(reversible=True, changes_target_space=True)),
    ("sliding_window", "shape", "s3", [], s3_shape.sliding_window, True,
     _c(changes_target_space=True)),
    ("lag_features", "shape", "s3", [], s3_shape.lag_features, True,
     _c(changes_target_space=True)),
    ("spectral_features", "shape", "s3", [], s3_shape.spectral_features, True,
     _c(changes_target_space=True)),
]

# 旧 ID → canonical（S0.7-6；旧 artifact/模板重放依赖，只增不删）
ALIASES: Dict[str, str] = {
    "fill_gaps": "impute_linear",      # 语义重复（fill_gaps=interp_nan≡impute_linear）
    "impute_kalman": "impute_ema",     # 误名：实现是 EMA 前向填补，非 Kalman
    "kalman_filter": "smooth_ema",     # 误名：实现是一阶 EMA，非 Kalman
}

TOOL_REGISTRY: Dict[str, Callable] = {name: fn for (name, _c_, _s, _t, fn, _sc, _ct) in OPERATOR_SPECS}
for _alias, _canon in ALIASES.items():
    TOOL_REGISTRY[_alias] = TOOL_REGISTRY[_canon]

OPERATOR_METADATA: Dict[str, dict] = {
    name: {"name": name, "category": c, "stage": s, "tags": list(t), "shape_changing": sc, **ct}
    for (name, c, s, t, _fn, sc, ct) in OPERATOR_SPECS
}
for _alias, _canon in ALIASES.items():
    OPERATOR_METADATA[_alias] = {**OPERATOR_METADATA[_canon], "name": _alias,
                                 "is_alias": True, "alias_of": _canon}

OPERATOR_NAMES = tuple(name for (name, *_rest) in OPERATOR_SPECS)   # canonical only


def canonicalize(name: str) -> str:
    """旧 ID → canonical；canonical/未知名原样返回（未知名由 get_operator/executor 报错）。"""
    return ALIASES.get(name, name)


def get_operator(name: str) -> Callable:
    if name not in TOOL_REGISTRY:
        raise KeyError(f"unknown operator: {name!r}")
    canon = canonicalize(name)
    if canon != name:
        _prov_record(name, canon, "compat_alias")   # manifest 同录 requested/canonical（S0.7-6）
    return TOOL_REGISTRY[canon]
