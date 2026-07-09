"""policy/pattern_spec.py — PatternSpec（Stage 2.0-②，Component Plan v1.1b）。

P0 = 现 10 维 struct_feats 的**冻结快照**：任何特征改动之前先钉死本规格；新特征（2.1-A0 的
共享周期模块、mask-aware 等）一律以新版本（P1…）落地，**禁止原地改 P0**。

复现字段（v1.1b 必备，缺了"PatternSpec=P0"就会随环境/代码漂移）：
  feature order / missing-value 语义 / period estimator ID / no-period 表示 /
  normalization 参数 / confidence 输出 schema / 依赖指纹 + code SHA / 兼容 action-menu 版本。

P0 有两个**已知且冻结不修**的缺陷（修复=P1 的内容，见 D1/D2）：
  D1 周期估计朴素（无去趋势/无显著性/无 ACF 确认）→ S_both 被趋势劫持；
  D2 缺失值压缩时间轴（x=raw[mask] 后直接 FFT/ACF/ADF）。
把缺陷写进规格是有意的：P0 的价值是"E-3.2/confirmatory 训练分布的精确复现锚"，不是最优特征。

特征分类边界（v1.1d 冻结）：PatternSpec 只装 P（内在结构）/ D（退化）/ C（可信度）三类；
φ(P,D,a,m) 动作/模型交互特征（如 window/period 比、候选窗可平滑能量）**禁止入 spec**——
它们在 Router 决策时由 P×动作元数据现算，否则 Pattern 随 ActionMenu 漂移、失去跨动作迁移意义。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..conditioning.key import STRUCT_FEAT_NAMES, build_conditioning_key, struct_feats
from ..e32_policy import D_FEATS, P_FEATS

# 提取器**代码闭包**（评审第二十四轮：A0 后周期实现已迁 period.py，只 hash key.py 会漏掉
# 真正承载 period 数值的文件）。P0 的提取路径 = key.py（struct_feats）→ period.py（legacy_fft_v0）。
_COND_DIR = Path(__file__).resolve().parent.parent / "conditioning"
_CODE_CLOSURE = (_COND_DIR / "key.py", _COND_DIR / "period.py")


def _sha256_closure(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _dependency_fingerprint() -> Dict[str, Optional[str]]:
    """特征数值敏感的依赖：statsmodels 缺失时 ADF 走 numpy 代理 → **数值不同**，必须可核验。"""
    out: Dict[str, Optional[str]] = {"numpy": np.__version__}
    for mod in ("statsmodels", "scipy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                    # pragma: no cover
            out[mod] = None
    return out


@dataclass(frozen=True)
class PatternSpec:
    """特征提取契约。同 series + 同 spec（version+code_sha+依赖指纹一致）→ bit 级一致特征。"""
    version: str
    feature_names: Tuple[str, ...]           # 有序——顺序本身是契约的一部分
    d_feats: Tuple[str, ...]                 # Router X_d 列序（退化坐标，deploy 可得）
    p_feats: Tuple[str, ...]                 # Router X_p 列序（结构特征，不含 SNR/missing）
    scaler: str                              # 特征入模前缩放（P0="none"：GBDT 尺度不变）
    missing_semantics: str                   # 缺失值如何进入特征计算
    period_estimator_id: str                 # 周期估计器身份（D1 双估计器漂移的记账点）
    no_period_repr: str                      # 无周期/退化输入的表示约定
    missing_feature_fill: float              # 特征字典缺键时的填充（e32_nested f.get(k, 0.0) 语义）
    confidence_schema: Optional[Dict[str, Any]]   # C 通道输出 schema（P0=None：无 C 通道）
    code_sha256: str                         # 提取器代码闭包 SHA（key.py+period.py；活值，核验用）
    dependency_fingerprint: Dict[str, Optional[str]]
    compatible_action_menus: Tuple[str, ...]  # 可搭配的 ActionMenu 版本

    # ── 提取 ──────────────────────────────────────────────────────────────
    def extract(self, series, task_type: str, task_spec: Optional[dict] = None) -> Dict[str, Any]:
        """完整 conditioning_key（P0 = 现 build_conditioning_key，零改动包装）。
        P1a 暂不提供（离线 Router 实验只需 features_vector；fast path 集成待 Router 轮胜出）。"""
        self._require_p0()
        return build_conditioning_key(series, task_type, task_spec)

    def features_vector(self, series) -> np.ndarray:
        """有序特征向量（bit 级一致性测试的对象）。P0=10 维；P1a=16 维（D4+P9+C3）。"""
        if self.version == "P0":
            f = struct_feats(series)
        elif self.version == "P1a":
            from ..conditioning.p1a import p1a_feats
            f = p1a_feats(series)
        else:
            raise NotImplementedError(
                f"PatternSpec {self.version!r} 的提取器尚未落地（新版本须自带 extractor，禁止改 P0）")
        return np.array([float(f[k]) for k in self.feature_names], dtype=float)

    def router_features(self, struct: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """struct_feats dict → (X_d 行, X_p 行)，与 e32_nested._policy_data 同一映射
        （f.get(k, fill) × D_FEATS/P_FEATS 列序）→ 部署侧特征与 Router 训练分布同源。"""
        fill = self.missing_feature_fill
        xd = np.array([float(struct.get(k, fill)) for k in self.d_feats], dtype=float)
        xp = np.array([float(struct.get(k, fill)) for k in self.p_feats], dtype=float)
        return xd, xp

    # ── 身份 ──────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_sha(self) -> str:
        """语义身份 SHA（不含活值 code_sha/依赖指纹——那两者用于**核验**环境，不定义语义）。"""
        payload = {k: v for k, v in self.to_dict().items()
                   if k not in ("code_sha256", "dependency_fingerprint")}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def _require_p0(self):
        if self.version != "P0":
            raise NotImplementedError(
                f"PatternSpec {self.version!r} 的提取器尚未落地（新版本须自带 extractor，禁止改 P0）")


def pattern_spec_p0() -> PatternSpec:
    """P0 冻结快照工厂。任何字段改动 = 新版本，不是编辑本函数。"""
    return PatternSpec(
        version="P0",
        feature_names=tuple(STRUCT_FEAT_NAMES),
        d_feats=tuple(D_FEATS),
        p_feats=tuple(P_FEATS),
        scaler="none",
        missing_semantics=(
            "mask-drop：x = raw[~isnan(raw)] 压缩时间轴后计算全部特征（D2 已知缺陷，P0 冻结不修）；"
            "missing_rate 在压缩前按原索引记录；非缺失 <4 点 → 全 0 向量 + missing_rate + period=1.0"),
        period_estimator_id=(
            "fft_dominant_naive_v0：conditioning/key.py:_dominant_period（rfft 全谱 argmax，"
            "无去趋势/无显著性/无 ACF 确认——D1 已知缺陷，P0 冻结不修；上限 clip 到序列长）"),
        no_period_repr="period=1.0 表示无显著周期；seasonal_strength 相应为 0.0",
        missing_feature_fill=0.0,
        confidence_schema=None,
        code_sha256=_sha256_closure(_CODE_CLOSURE),
        dependency_fingerprint=_dependency_fingerprint(),
        compatible_action_menus=("v1",),
    )


_CODE_CLOSURE_P1A = _CODE_CLOSURE + (_COND_DIR / "p1a.py",)


def pattern_spec_p1a() -> PatternSpec:
    """P1a = D1+D2 修复 + 缺失拓扑 D + 最小 C 通道（Stage 2.1 第一臂，2026-07-05）。
    φ(P,D,a,m) 动作交互特征按 v1.1d 边界**不在本 spec**（Router 轮由 P 原料现算）。
    P0 原样冻结；本 spec 是新版本对象，字段见 conditioning/p1a.py 模块文档。"""
    from ..conditioning.p1a import P1A_ALL_FEATS, P1A_C_FEATS, P1A_D_FEATS, P1A_P_FEATS
    return PatternSpec(
        version="P1a",
        feature_names=tuple(P1A_ALL_FEATS),
        d_feats=tuple(P1A_D_FEATS),
        p_feats=tuple(P1A_P_FEATS),
        scaler="none",
        missing_semantics=(
            "mask-aware（D2 修复）：时间轴永不压缩；谱/ACF/ADF 用仅供感知的显式线性插值"
            "（端点最近观测延伸），逐点统计（噪声 MAD/离群/lumpiness/acf1）只在观测点上算；"
            "缺失拓扑（max_gap_frac/gap_run_mean_frac/c_obs_coverage）按原索引的 NaN 段计；"
            "观测 <4 点 → 全 0 向量 + missing_rate/coverage 照记"),
        period_estimator_id=(
            "robust_v1：conditioning/period.py:guess_period_robust_v1（去趋势+候选范围+谱峰"
            "显著性+ACF 确认，D1 修复）+ top_k_periods（period_count）+ robust_period_diag（C 证据）"),
        no_period_repr="period=0.0 表示无显著周期（≠P0 的 1.0）；seasonal_strength 相应为 0.0",
        missing_feature_fill=0.0,
        confidence_schema={
            "features": list(P1A_C_FEATS),
            "semantics": "c_peak_sig=ratio/(ratio+3)∈[0,1)（0.5=判据线）；c_acf_confirm=候选周期"
                         "ACF 原值（证据，无论判决）；c_obs_coverage=最长连续观测段/n",
            "usage": "本轮只作特征+诊断相关性；abstain/拦截接入=2.2-⑥"},
        code_sha256=_sha256_closure(_CODE_CLOSURE_P1A),
        dependency_fingerprint=_dependency_fingerprint(),
        compatible_action_menus=("v1",),
    )
