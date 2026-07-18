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
     reversible / changes_target_space / requires_dependency / fallback_policy / dependency_policy
     ——新算子必须从第一天带契约入池（E-3.3 扩充池前置）。
  3. **task 级物理过滤**：`fast_path.usable_ops` 按 allowed_tasks 硬过滤——anomaly 物理禁
     smoothing/destructive（保 spike/changepoint），不再只靠模板 ban。

E-3.3 R1–R3（2026-07-14）新增契约字段 `dependency_policy`。动机：原契约**说不出**
"依赖缺失时该怎么办"——`denoise_stl` 与 `impute_ssm` 的 requires_dependency 都是 "statsmodels"，
但前者的正确行为是记账回退 savgol、后者的正确行为是**硬失败**（静默变成 EMA 就是 `impute_kalman`
误名事故的原型：台账记着"跑了 Kalman"、实际跑的是指数平滑）。两者在旧契约里无法区分，
于是"不许静默降级"只能靠代码注释和人的记性守——这正是本项目反复踩的"声明≠执行"。
现在它是一个可被测试机械检查的字段。
"""
from __future__ import annotations

from typing import Callable, Dict

from . import s1_impute, s1_denoise, s1_outlier, s1_decompose, s1_structural, s2_align, s3_shape
from ._common import BOUNDARY_MODES  # noqa: F401（S0.7-8 边界语义指纹，随 registry 一并落 provenance）
from ._provenance import record as _prov_record

_ALL_TASKS = ("forecast", "classification", "anomaly_detection")
_NON_ANOMALY = ("forecast", "classification")   # 平滑/删改类：物理禁 anomaly（毁 spike/changepoint 信号）

_CONTRACT_BASE = {
    "allowed_tasks": _ALL_TASKS,
    "destructive": False,              # 删改观测点取值（离群置换/裁剪/搬电平）
    "preserves_observed": False,       # imputer 契约：只写缺失位置、保留观测
    "reversible": False,               # 可由记录的参数逆变换（如仿射归一化）
    "changes_target_space": False,     # 改变下游目标空间（预测须逆变换才能对齐原 future）
    "requires_dependency": None,       # 非 numpy 依赖
    "fallback_policy": "none",         # 退化**输入**下的回退："none" | "explicit_record→<op>" | "numpy_equivalent"
    "dependency_policy": None,         # 依赖**缺失**时："hard_fail" | "recorded_fallback"；无依赖 → None
    "public_parameter_bindings": {},   # 可从部署可观察特征机械绑定的参数；空字典表示无声明
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
    # —— E-3.3 R2：模型预测族插补（机制上与上面的复制/插值族可区分）——
    ("impute_ssm", "impute", "s1", [], s1_impute.impute_ssm, False,
     _c(preserves_observed=True, requires_dependency="statsmodels",
        dependency_policy="hard_fail",                       # 缺 statsmodels → raise，**绝不降级到 EMA**
        fallback_policy="explicit_record→impute_linear")),   # 仅退化输入（观测点过少/平滑器异常）
    ("impute_ar", "impute", "s1", [], s1_impute.impute_ar, False,
     _c(preserves_observed=True,                             # 纯 numpy：无依赖
        fallback_policy="explicit_record→impute_linear")),   # 仅退化输入（滞后窗口不足/递推发散）
    ("denoise_savgol", "denoise", "s1", ["smoothing"], s1_denoise.denoise_savgol, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="scipy",
        dependency_policy="recorded_fallback", fallback_policy="numpy_equivalent")),
    ("denoise_wavelet", "denoise", "s1", ["smoothing"], s1_denoise.denoise_wavelet, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="pywt",
        dependency_policy="recorded_fallback",
        fallback_policy="explicit_record→denoise_savgol")),
    ("denoise_median", "denoise", "s1", ["smoothing"], s1_denoise.denoise_median, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="scipy",
        dependency_policy="recorded_fallback", fallback_policy="numpy_equivalent")),
    ("smooth_ma", "denoise", "s1", ["smoothing"], s1_denoise.smooth_ma, False,
     _c(allowed_tasks=_NON_ANOMALY)),   # 纯 numpy（无依赖），symmetric 边界；F0 剂量维
    ("denoise_stl", "denoise", "s1", ["smoothing"], s1_denoise.denoise_stl, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="statsmodels",
        dependency_policy="recorded_fallback",               # 与 impute_ssm 的 hard_fail 刻意不同
        fallback_policy="explicit_record→denoise_savgol")),
    ("winsorize", "outlier", "s1", ["destructive"], s1_outlier.winsorize, False,
     _c(allowed_tasks=_NON_ANOMALY, destructive=True)),
    ("outlier_iqr", "outlier", "s1", ["destructive"], s1_outlier.outlier_iqr, False,
     _c(allowed_tasks=_NON_ANOMALY, destructive=True)),
    ("outlier_mad", "outlier", "s1", ["destructive"], s1_outlier.outlier_mad, False,
     _c(allowed_tasks=_NON_ANOMALY, destructive=True)),
    # —— E-3.3 R3：局部自适应点式离群修复（与上面三个全局阈值裁剪算子机制不同）——
    ("hampel_filter", "outlier", "s1", ["destructive"], s1_outlier.hampel_filter, False,
     _c(allowed_tasks=_NON_ANOMALY, destructive=True)),      # 纯 numpy
    # —— E-3.3 R1：结构断层修复（填 benchmark 预先声明的 structural_break 能力缺口）——
    ("repair_level_shift", "structural", "s1", ["destructive"], s1_structural.repair_level_shift, False,
     _c(
         allowed_tasks=_NON_ANOMALY,
         destructive=True,
         public_parameter_bindings={
             "region_start_fraction": "estimated_region_start_fraction",
             "region_end_fraction": "estimated_region_end_fraction",
             "estimated_offset": "estimated_level_offset",
         },
     )),      # 纯 numpy（无 ruptures 依赖）
    ("stl_decompose", "decompose", "s1", [], s1_decompose.stl_decompose, False,
     _c(allowed_tasks=_NON_ANOMALY, requires_dependency="statsmodels",
        dependency_policy="recorded_fallback",
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
#
# E-3.3（2026-07-14）：三个 alias 全部标记 **deprecated**——新代码/新模板/新菜单一律不得引用。
# 它们保留的**唯一**理由是旧 artifact 可重放；OPERATOR_METADATA 里带 deprecated=True，
# 测试守新动作面（menu v2）不含任何 alias。
#
# ⚠️ 尤其 `impute_kalman`/`kalman_filter`：这两个名字在本仓库历史 trace 里指的是 EMA。
# 真正的状态空间插补叫 **impute_ssm**（E-3.3 R2 新增），**不叫 impute_kalman**——复用旧名字
# 会让"同一个名字在旧记录里指 EMA、在新记录里指 Kalman"，任何跨版本算子身份审计都会被骗过去。
ALIASES: Dict[str, str] = {
    "fill_gaps": "impute_linear",      # 语义重复（fill_gaps=interp_nan≡impute_linear）
    "impute_kalman": "impute_ema",     # 误名：实现是 EMA 前向填补，非 Kalman；真 SSM 见 impute_ssm
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
                                 "is_alias": True, "alias_of": _canon,
                                 "deprecated": True}   # 只为旧 artifact 重放而存在，新代码禁用

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
