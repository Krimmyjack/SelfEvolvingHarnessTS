"""p6/miner.py — P6 冻结 miner（prereg §4 「miner = 冻结代码」：全部常量在码内、零自由度）。

入口 `mine(family, evidence, state, eps=None, c0_bins=None) -> List[MinedCandidate]`：
≤3 个候选、确定性次序（排序键 = `candidate_sort_key` = (族内配方序, canonical proposal
sha 升序)——sha 是并列 tie-break，prereg §4「候选并列 tie-break = canonical edit sha 升序」）。
payload 完全相同的候选去重（按配方序 keep-first；prereg「三者权重向量完全相同则去重」的
逐对一般化——同一 payload 即同一 edit，重评预算不重复消耗）。

family（冻结）："selector" / "sampler" / "risk"；另接受 metrics.activate 的输出别名
S1→selector、S2→sampler、S3→risk（FAMILY_ALIASES，免去 runner 手写映射）。
`eps` 保留在签名以稳定 API，但**任何冻结配方都不使用它**（零自由度：配方无 ε 旋钮）。

SelectorPatch 族（evidence = {"rows": [{episode_uid, features, train_gain}, ...]}）：
  公共底座：特征 = KNOWN_FEATURES 全集固定次序（缺失键按 0.0，对齐 fast_path.select；
  非 KNOWN 特征键忽略）；z-standardize 常量取自 full-D（均值/方差 ddof=0，方差下限
  VARIANCE_FLOOR=1e-12）；ridge α=1.0（截距不受罚、截距不进 SelectorSpec.weights）；
  uid 2-fold：fold = int(sha256(uid utf-8).hexdigest(), 16) % 2（`fold_of`）。
  (a) selector_a：目标 = train_gain，部署权重 = full-D refit；
  (b) selector_b：目标 = episode 内平均秩（ties 均秩；**秩升序 = gain 大者秩大**，写死），
      部署权重 = full-D refit；秩不做跨 episode 归一化（prereg 未指定 → 不加自由度）；
  (c) selector_c：坐标级保守收缩 σ̂_j = |w_j^{fold0} − w_j^{fold1}|/√2，
      w_c[j] = sign(w_full[j])·max(0, |w_full[j]| − σ̂_j)（逐坐标向零软阈值；prereg 首稿
      「LCB 打分」的唯一冻结释义）；某 fold 无样本 → (c) 不可用（<3 候选合法）。
  权重按 WEIGHT_DECIMALS=12 位小数 round（Python round，half-even）保 JSON 稳定，
  −0.0 归一为 0.0。

SamplerPatch 族（allocation 经 `state.sampler` 供给——单一事实来源，evidence 参数不读，
防止双来源漂移）：字面配方 SAMPLER_DELTAS：(a) det−1/random+1；(b) random−2/llm+2；
(c) det−2/random+2。任一新配额 <0 → 该配方不可用；全部不可用 → []（上层按 abstain）。
expected_total 与 random_params 原样保留（SamplerPatch 整体替换 sampler，不得夹带第二处改动）。

RiskRulePatch 族（evidence = {"cohort", "fingerprints", "accused_sha", "accused_ops"}；
c0_bins = {allowlist 特征: 3 个升序 C0 四分位 cutpoints}）：
  cohort 两种冻结形：
    {"cohort_id", "bin": {"feature", "lo", "hi"}}  → scope 条件 = [(f,">=",lo)]+[(f,"<",hi)]
                                                     （None 边省略；左闭右开对齐四分 bin）；
    {"cohort_id", "preset": 名}                    → preset 成员资格 scope（F7/finding 37）=
                                                     单原子 (preset,"==",名)；apply_risk/matches
                                                     对 fingerprint["preset"] 求成员判定，取代
                                                     旧 C0 中位数半平面近似（那会把 preset 误当
                                                     snr/missing 数值条件、ban 错集）。
  scope 条件为空 → 整族 []（无法作用域化）；条件 feature ∉ P0_FEATURE_ALLOWLIST∪{preset} 或
  op 非法 → raise ValueError（坏 evidence 必须响亮）。
  **when 语义（prereg §4 冻结）**：RiskRuleSpec.when 原生支持条件列表 = 同规则内 AND，
  三配方全部产出**单规则、when = 条件列表**（bin scope 的 lo/hi 两原子同规则合取；禁止
  拆多规则近似——apply_risk 逐规则独立求值会把合取退化为并集 ban，bin scope 在并集语义
  下等价全局 ban）。规则**之间**才是并集 ban（(b) 的跨成员多规则 = 族语义本身）。
  (a) risk_a：ban accused 程序 sha @ scope（单规则，when = scope 条件列表）；
  (b) risk_b：ban 算子族 @ 同 scope；族映射 OP_FAMILY_MAP 冻结在码内
      （{denoise_median, denoise_savgol, smooth_ma}→denoiser、{winsorize}→outlier、
      {impute_linear}→imputer）；以被告程序 **op 列表次序的第一个命中族**为准；
      族 = 全部成员 op **各一条单规则**（RiskRuleSpec 单 target；每条 when = 同一 scope
      条件列表），成员序 = OP_FAMILY_MEMBERS 冻结序；无命中 → (b) 不可用；
  (c) risk_c：同 (a) 加合取条件——按 ALLOWLIST_ORDER 冻结次序（= allowlist 构造序
      ("snr","missing_rate")+P_FEATS_FROZEN）取第一个满足「cohort ≥80% episode 落入其
      C0 四分位单一 bin」的特征，把该 bin 的原子条件**追加进同一规则的 when 列表**
      （bin 索引 = bisect_right(cutpoints, v)，左闭右开；中段 bin 本身贡献 2 个原子条件）；
      ≥80% 以整数形写死 **5·count ≥ 4·n**（恰 80% 满足，无浮点边界）；缺特征的 episode
      计入分母、不入任何 bin；无合格特征 / 无 fingerprint / 无 c0_bins → (c) 不可用。

MinedCandidate（frozen dataclass）：
  proposal_dicts        tuple[dict]，每个都可被 edit_surfaces.compile_proposal 编译
                        （(a)/(c) 恰 1 个；(b) 多成员族 bundle = 每成员 1 个）；
  proposal_dict         便捷 property：恰 1 个提案时返回之；bundle 上 raise ValueError
                        （防止只应用半个族）；
  recipe_id             冻结配方名（FAMILY_RECIPES）；
  provenance            {"schema","family","recipe_id","evidence_summary","semantics"}；
  provenance_digest     sha256(canonical_json(provenance)).hexdigest()
                        （= sha256(canonical(evidence 摘要+recipe_id))，摘要含族/schema
                        ——均为配方决定的常量）；
  candidate_sha         property：sha256(canonical_json(list(proposal_dicts)))[:16]
                        （tie-break 与去重键）。

红线：numpy + stdlib；无 RNG、无 IO、无网络；不修改任何输入；不 import legacy。
"""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .harness_state import (
    KNOWN_FEATURES,
    P0_FEATURE_ALLOWLIST,
    PRESET_SCOPE_FEATURE,
    PRESET_SCOPE_OP,
    P6HarnessState,
    P_FEATS_FROZEN,
    RISK_OPS,
    SUPPLIER_NAMES,
    canonical_json,
)

__all__ = [
    "ALLOWLIST_ORDER",
    "COHORT_BIN_MAJORITY_DEN",
    "COHORT_BIN_MAJORITY_NUM",
    "FAMILY_ALIASES",
    "FAMILY_RECIPES",
    "MinedCandidate",
    "OP_FAMILY_MAP",
    "OP_FAMILY_MEMBERS",
    "PROVENANCE_SCHEMA",
    "RIDGE_ALPHA",
    "SAMPLER_DELTAS",
    "VARIANCE_FLOOR",
    "WEIGHT_DECIMALS",
    "candidate_sort_key",
    "fold_of",
    "mine",
]

# ═══════════════════════ 冻结常量（全部在码内，零自由度） ═══════════════════════
PROVENANCE_SCHEMA = "p6-miner/1"
RIDGE_ALPHA: float = 1.0            # prereg §4：ridge α=1.0（截距不受罚）
VARIANCE_FLOOR: float = 1e-12       # z-standardize 方差下限（sd = sqrt(max(var, floor))）
WEIGHT_DECIMALS: int = 12           # 权重 12 位小数 round（JSON 稳定）
COHORT_BIN_MAJORITY_NUM: int = 4    # (c) ≥80% 的整数冻结形：5·count ≥ 4·n（恰 80% 满足）
COHORT_BIN_MAJORITY_DEN: int = 5

# allowlist 冻结次序 = 其构造序（harness_state：frozenset(("snr","missing_rate")+P_FEATS)）
ALLOWLIST_ORDER: Tuple[str, ...] = ("snr", "missing_rate") + P_FEATS_FROZEN
if frozenset(ALLOWLIST_ORDER) != P0_FEATURE_ALLOWLIST or len(ALLOWLIST_ORDER) != len(
    P0_FEATURE_ALLOWLIST
):
    raise AssertionError("ALLOWLIST_ORDER 与 P0_FEATURE_ALLOWLIST 漂移（冻结不变量破坏）")

# 算子族映射（prereg §4：冻结在 miner 码内；三族）
OP_FAMILY_MAP: Dict[str, str] = {
    "denoise_median": "denoiser",
    "denoise_savgol": "denoiser",
    "smooth_ma": "denoiser",
    "winsorize": "outlier",
    "impute_linear": "imputer",
}
OP_FAMILY_MEMBERS: Dict[str, Tuple[str, ...]] = {
    "denoiser": ("denoise_median", "denoise_savgol", "smooth_ma"),
    "outlier": ("winsorize",),
    "imputer": ("impute_linear",),
}

FAMILY_RECIPES: Dict[str, Tuple[str, ...]] = {
    "selector": ("selector_a", "selector_b", "selector_c"),
    "sampler": ("sampler_a", "sampler_b", "sampler_c"),
    "risk": ("risk_a", "risk_b", "risk_c"),
}
FAMILY_ALIASES: Dict[str, str] = {"S1": "selector", "S2": "sampler", "S3": "risk"}

# 字面 Sampler 配方（prereg §4）
SAMPLER_DELTAS: Dict[str, Dict[str, int]] = {
    "sampler_a": {"det": -1, "random": +1, "llm": 0},
    "sampler_b": {"det": 0, "random": -2, "llm": +2},
    "sampler_c": {"det": -2, "random": +2, "llm": 0},
}

_RECIPE_RANK: Dict[str, int] = {
    rid: i for recipes in FAMILY_RECIPES.values() for i, rid in enumerate(recipes)
}


# ═══════════════════════════ 基础工具 ═══════════════════════════
def _sha16(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


def _finite(x: Any, name: str) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"{name} 必须是有限数，得到 {x!r}")
    return v


def fold_of(uid: str) -> int:
    """uid 2-fold（冻结公式）：int(sha256(uid utf-8).hexdigest(), 16) % 2。"""
    return int(hashlib.sha256(str(uid).encode("utf-8")).hexdigest(), 16) % 2


@dataclass(frozen=True)
class MinedCandidate:
    """miner 产出的一个候选 edit（字段语义见模块 docstring）。"""

    proposal_dicts: Tuple[Dict[str, Any], ...]
    recipe_id: str
    provenance: Dict[str, Any]
    provenance_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_dicts", tuple(dict(p) for p in self.proposal_dicts)
        )
        object.__setattr__(self, "recipe_id", str(self.recipe_id))
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "provenance_digest", str(self.provenance_digest))
        if not self.proposal_dicts:
            raise ValueError("MinedCandidate 至少含 1 个 proposal_dict")

    @property
    def candidate_sha(self) -> str:
        """canonical proposal sha（tie-break/去重键）：全部提案按序 canonical JSON 的 sha16。"""
        return _sha16(list(self.proposal_dicts))

    @property
    def proposal_dict(self) -> Dict[str, Any]:
        """单提案便捷视图；bundle（合取/族多规则）上 raise——不许只应用半个合取。"""
        if len(self.proposal_dicts) != 1:
            raise ValueError(
                f"候选 {self.recipe_id!r} 含 {len(self.proposal_dicts)} 个提案"
                f"（合取/族 bundle），请使用 proposal_dicts 全量应用"
            )
        return self.proposal_dicts[0]


def candidate_sort_key(candidate: MinedCandidate) -> Tuple[int, str]:
    """确定性输出序键：(族内配方序, canonical proposal sha 升序)。未知 recipe_id → raise。"""
    if candidate.recipe_id not in _RECIPE_RANK:
        raise ValueError(f"未知 recipe_id {candidate.recipe_id!r}")
    return (_RECIPE_RANK[candidate.recipe_id], candidate.candidate_sha)


def _make_candidate(
    family: str,
    recipe_id: str,
    proposal_dicts: Sequence[Dict[str, Any]],
    evidence_summary: Dict[str, Any],
    semantics: Dict[str, Any],
) -> MinedCandidate:
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "family": family,
        "recipe_id": recipe_id,
        "evidence_summary": evidence_summary,
        "semantics": semantics,
    }
    digest = hashlib.sha256(canonical_json(provenance).encode("utf-8")).hexdigest()
    return MinedCandidate(
        proposal_dicts=tuple(proposal_dicts),
        recipe_id=recipe_id,
        provenance=provenance,
        provenance_digest=digest,
    )


# ═══════════════════════════ SelectorPatch 族 ═══════════════════════════
def _ridge_fit(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ridge（α=RIDGE_ALPHA）+ 不受罚截距；返回特征权重（截距不返回、不进 weights）。"""
    n, p = Z.shape
    X = np.concatenate([Z, np.ones((n, 1), dtype=float)], axis=1)
    A = X.T @ X
    for j in range(p):                      # 只罚特征坐标，截距（最后一列）不受罚
        A[j, j] += RIDGE_ALPHA
    w = np.linalg.solve(A, X.T @ y)
    return w[:p]


def _avg_ranks(values: Sequence[float]) -> List[float]:
    """episode 内平均秩：升序 1..m，ties 取均秩（精确相等成组）；**gain 大者秩大**（写死）。"""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0             # 位置 i..j（0 基）的均秩（1 基）
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _round_weight(v: float) -> float:
    r = round(float(v), WEIGHT_DECIMALS)
    return 0.0 if r == 0.0 else r           # −0.0 归一（JSON/sha 稳定）


def _selector_normalize(
    evidence: Any,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if not isinstance(evidence, Mapping) or not evidence.get("rows"):
        raise ValueError('selector evidence 必须含非空 "rows"')
    uids: List[str] = []
    X: List[List[float]] = []
    y: List[float] = []
    for i, row in enumerate(evidence["rows"]):
        uids.append(str(row["episode_uid"]))
        feats = row["features"]
        X.append(
            [_finite(feats.get(f, 0.0), f"rows[{i}].features[{f!r}]") for f in KNOWN_FEATURES]
        )
        y.append(_finite(row["train_gain"], f"rows[{i}].train_gain"))
    return uids, np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _mine_selector(evidence: Any) -> List[MinedCandidate]:
    uids, X, y = _selector_normalize(evidence)
    mu = X.mean(axis=0)                                  # 标准化常量取自 full-D（冻结）
    sd = np.sqrt(np.maximum(X.var(axis=0), VARIANCE_FLOOR))
    Z = (X - mu) / sd

    weights_by_recipe: List[Tuple[str, np.ndarray]] = []
    w_full = _ridge_fit(Z, y)
    weights_by_recipe.append(("selector_a", w_full))     # (a) train_gain / full-D refit

    groups: Dict[str, List[int]] = {}
    for i, u in enumerate(uids):
        groups.setdefault(u, []).append(i)
    y_rank = np.empty_like(y)
    for idxs in groups.values():
        for i, r in zip(idxs, _avg_ranks([float(y[i]) for i in idxs])):
            y_rank[i] = r
    weights_by_recipe.append(("selector_b", _ridge_fit(Z, y_rank)))   # (b) 平均秩

    folds = np.asarray([fold_of(u) for u in uids])
    if bool((folds == 0).any()) and bool((folds == 1).any()):
        w0 = _ridge_fit(Z[folds == 0], y[folds == 0])    # fold 拟合同用 full-D 标准化常量
        w1 = _ridge_fit(Z[folds == 1], y[folds == 1])
        sigma = np.abs(w0 - w1) / math.sqrt(2.0)
        wc = np.sign(w_full) * np.maximum(0.0, np.abs(w_full) - sigma)
        weights_by_recipe.append(("selector_c", wc))     # (c) 坐标级软阈值收缩
    # 某 fold 无样本 → (c) 不可用（<3 候选合法）

    rows_norm = [[u, [float(v) for v in X[i]], float(y[i])] for i, u in enumerate(uids)]
    summary = {"n_rows": len(uids), "rows_sha": _sha16(rows_norm)}
    out: List[MinedCandidate] = []
    for recipe_id, w in weights_by_recipe:
        weights = {f: _round_weight(w[j]) for j, f in enumerate(KNOWN_FEATURES)}
        proposal = {
            "kind": "selector_patch",
            "new_selector": {"kind": "weighted_features", "weights": weights},
        }
        out.append(_make_candidate("selector", recipe_id, (proposal,), summary, {}))
    return out


# ═══════════════════════════ SamplerPatch 族 ═══════════════════════════
def _mine_sampler(state: P6HarnessState) -> List[MinedCandidate]:
    alloc = {k: int(state.sampler.allocation[k]) for k in SUPPLIER_NAMES}
    expected_total = int(state.sampler.expected_total)
    random_params = dict(state.sampler.random_params)
    summary = {
        "allocation": dict(alloc),
        "expected_total": expected_total,
        "random_params": dict(random_params),
    }
    out: List[MinedCandidate] = []
    for recipe_id in FAMILY_RECIPES["sampler"]:
        delta = SAMPLER_DELTAS[recipe_id]
        new_alloc = {k: alloc[k] + delta[k] for k in SUPPLIER_NAMES}
        if any(v < 0 for v in new_alloc.values()):
            continue                                     # 负配额 → 该配方不可用
        proposal = {
            "kind": "sampler_patch",
            "new_sampler": {
                "allocation": new_alloc,
                "expected_total": expected_total,        # 总 K 冻结（Σδ=0 按构造）
                "random_params": dict(random_params),    # 原样保留：不夹带第二处改动
            },
        }
        out.append(_make_candidate("sampler", recipe_id, (proposal,), summary, {}))
    return out


# ═══════════════════════════ RiskRulePatch 族 ═══════════════════════════
def _atomic(feature: str, op: str, value: Any) -> Dict[str, Any]:
    return {"feature": str(feature), "op": str(op), "value": value}


def _validate_conditions(conds: Sequence[Mapping[str, Any]]) -> None:
    for c in conds:
        if c["feature"] == PRESET_SCOPE_FEATURE:
            # preset 成员资格 scope（F7）：op 固定 "=="、value 是非空 preset 名 str。
            if c["op"] != PRESET_SCOPE_OP:
                raise ValueError(
                    f"preset scope 条件 op 必须为 {PRESET_SCOPE_OP!r}（成员资格），得到 {c['op']!r}"
                )
            if not isinstance(c["value"], str) or not c["value"]:
                raise ValueError("preset scope 条件 value 必须是非空 preset 名 str")
            continue
        if c["feature"] not in P0_FEATURE_ALLOWLIST:
            raise ValueError(
                f"scope 条件特征 {c['feature']!r} 不在冻结 P0 allowlist"
                f"（{sorted(P0_FEATURE_ALLOWLIST)}）、也非 preset 成员资格保留维"
                f"（{PRESET_SCOPE_FEATURE!r}）"
            )
        if c["op"] not in RISK_OPS:
            raise ValueError(f"scope 条件 op {c['op']!r} 非法（可用：{RISK_OPS}）")


def _scope_conditions(cohort: Any) -> List[Dict[str, Any]]:
    """cohort 定义 → allowlist 原子条件列表（可为空 = 整族不可用）。两种冻结形见模块 docstring。"""
    if not isinstance(cohort, Mapping) or "cohort_id" not in cohort:
        raise ValueError('cohort 必须是含 "cohort_id" 的 Mapping')
    has_bin = "bin" in cohort
    has_preset = "preset" in cohort
    if has_bin == has_preset:
        raise ValueError('cohort 必须恰含 "bin"（P0 四分 bin）或 "preset"（preset 名）之一')
    conds: List[Dict[str, Any]] = []
    if has_bin:
        b = cohort["bin"]
        feature = str(b["feature"])
        lo, hi = b.get("lo"), b.get("hi")
        if lo is not None:
            conds.append(_atomic(feature, ">=", _finite(lo, "cohort.bin.lo")))
        if hi is not None:
            conds.append(_atomic(feature, "<", _finite(hi, "cohort.bin.hi")))
    else:
        # preset 成员资格 scope（F7/finding 37）：单原子 preset == 名（取代旧半平面近似）。
        preset = cohort.get("preset")
        if not isinstance(preset, str) or not preset:
            raise ValueError('preset cohort 必须含非空 str "preset"')
        conds.append(_atomic(PRESET_SCOPE_FEATURE, PRESET_SCOPE_OP, str(preset)))
    _validate_conditions(conds)
    return conds


def _single_rule(
    rule_id: str, conds: Sequence[Mapping[str, Any]], target: str
) -> Dict[str, Any]:
    """条件列表 → **单规则**：when = 条件列表 = 同规则内 AND（prereg §4 冻结语义；
    bin scope 的 lo/hi 两原子必须同规则合取，禁止拆多规则近似）。"""
    return {
        "rule_id": str(rule_id),
        "when": [dict(c) for c in conds],
        "then": {"action": "ban", "target": str(target)},
    }


def _rule_patches(rules: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{"kind": "risk_rule_patch", "add_rule": dict(r)} for r in rules]


def _bin_conditions(feature: str, bin_idx: int, cuts: Sequence[float]) -> List[Dict[str, Any]]:
    """四分位 bin → 原子条件（左闭右开；端 bin 单条件、中段 bin 两条件）。"""
    if bin_idx == 0:
        return [_atomic(feature, "<", cuts[0])]
    if bin_idx == len(cuts):
        return [_atomic(feature, ">=", cuts[-1])]
    return [
        _atomic(feature, ">=", cuts[bin_idx - 1]),
        _atomic(feature, "<", cuts[bin_idx]),
    ]


def _first_dominant_bin(
    fingerprints: Sequence[Mapping[str, Any]],
    c0_bins: Optional[Mapping[str, Sequence[float]]],
) -> Optional[Tuple[str, int, List[Dict[str, Any]]]]:
    """按 ALLOWLIST_ORDER 冻结次序找第一个「≥80% episode 落入单一 C0 四分位 bin」的特征。

    bin 索引 = bisect_right(cutpoints, v)（左闭右开：v == cutpoint 归上位 bin）；
    ≥80% 判定 = 整数形 5·count ≥ 4·n（恰 80% 满足）；缺该特征的 episode 计入分母、
    不入任何 bin。无合格特征 / 空 fingerprints / 无 c0_bins → None（(c) 不可用）。"""
    if not fingerprints or not c0_bins:
        return None
    n = len(fingerprints)
    for feature in ALLOWLIST_ORDER:
        if feature not in c0_bins:
            continue
        cuts = [_finite(c, f"c0_bins[{feature!r}]") for c in c0_bins[feature]]
        if len(cuts) != 3 or sorted(cuts) != cuts:
            raise ValueError(
                f"c0_bins[{feature!r}] 必须是 3 个升序四分位 cutpoints，得到 {cuts!r}"
            )
        counts = [0, 0, 0, 0]
        for i, fp in enumerate(fingerprints):
            if feature not in fp:
                continue
            counts[bisect_right(cuts, _finite(fp[feature], f"fingerprints[{i}][{feature!r}]"))] += 1
        best = max(range(4), key=lambda b: (counts[b], -b))       # 并列取小 bin 索引
        if COHORT_BIN_MAJORITY_DEN * counts[best] >= COHORT_BIN_MAJORITY_NUM * n:
            return feature, best, _bin_conditions(feature, best, cuts)
    return None


def _mine_risk(
    evidence: Any, c0_bins: Optional[Mapping[str, Sequence[float]]]
) -> List[MinedCandidate]:
    if not isinstance(evidence, Mapping):
        raise ValueError("risk evidence 必须是 Mapping")
    cohort = evidence["cohort"]
    cohort_id = str(cohort["cohort_id"])
    scope = _scope_conditions(cohort)
    if not scope:
        return []                                       # 无法作用域化 → 整族不可用
    accused_sha = str(evidence["accused_sha"])
    if not accused_sha:
        raise ValueError("accused_sha 不能为空")
    accused_ops = tuple(str(o) for o in (evidence.get("accused_ops") or ()))
    fingerprints = list(evidence.get("fingerprints") or ())

    summary = {
        "cohort_id": cohort_id,
        "scope_conditions": [dict(c) for c in scope],
        "accused_sha": accused_sha,
        "accused_ops": list(accused_ops),
        "n_cohort_episodes": len(fingerprints),
        "fingerprints_sha": _sha16([{str(k): v for k, v in fp.items()} for fp in fingerprints]),
        "c0_bins_sha": (
            _sha16({str(f): [float(c) for c in cuts] for f, cuts in c0_bins.items()})
            if c0_bins
            else None
        ),
    }
    out: List[MinedCandidate] = []

    # (a) ban 程序 sha @ cohort scope（单规则，when = scope 条件列表 = 同规则 AND）
    rule_a = _single_rule(f"risk_a_{cohort_id}_{accused_sha}", scope, accused_sha)
    out.append(_make_candidate("risk", "risk_a", _rule_patches([rule_a]), summary, {}))

    # (b) ban 算子族 @ 同 scope（被告 op 列表第一个命中族；无命中 → 不可用）：
    # 每成员各一条单规则（when = 同一 scope 条件列表）；规则间并集 = 族语义本身。
    family_name = next((OP_FAMILY_MAP[o] for o in accused_ops if o in OP_FAMILY_MAP), None)
    if family_name is not None:
        rules_b = [
            _single_rule(f"risk_b_{cohort_id}_{family_name}_{op}", scope, op)
            for op in OP_FAMILY_MEMBERS[family_name]
        ]
        out.append(
            _make_candidate(
                "risk", "risk_b", _rule_patches(rules_b), summary,
                {"op_family": family_name},
            )
        )

    # (c) 同 (a) + 第一个 ≥80% 单 bin 特征的合取条件（同一规则 when 内追加；无 → 不可用）
    dominant = _first_dominant_bin(fingerprints, c0_bins)
    if dominant is not None:
        feature, bin_idx, bin_conds = dominant
        rule_c = _single_rule(
            f"risk_c_{cohort_id}_{accused_sha}", list(scope) + bin_conds, accused_sha
        )
        out.append(
            _make_candidate(
                "risk", "risk_c", _rule_patches([rule_c]), summary,
                {"dominant_bin": {"feature": feature, "bin_index": bin_idx}},
            )
        )
    return out


# ═══════════════════════════ 入口 ═══════════════════════════
def mine(
    family: str,
    evidence: Any,
    state: P6HarnessState,
    eps: Optional[float] = None,
    c0_bins: Optional[Mapping[str, Sequence[float]]] = None,
) -> List[MinedCandidate]:
    """冻结 miner 入口：≤3 个 MinedCandidate，确定性次序（见模块 docstring）。

    `eps` 保留在签名（API 稳定），冻结配方一律不使用；`c0_bins` 只被 risk (c) 消费。
    payload 相同的候选去重（配方序 keep-first）；配方不可用即缺席（<3 候选合法）；
    全部不可用 → []（上层按 abstain 处理）。"""
    del eps                                             # 零自由度：配方无 ε 旋钮
    fam = FAMILY_ALIASES.get(str(family), str(family))
    if fam not in FAMILY_RECIPES:
        raise ValueError(
            f"未知 family {family!r}（可用：{sorted(FAMILY_RECIPES)} 及别名 "
            f"{sorted(FAMILY_ALIASES)}）"
        )
    if not isinstance(state, P6HarnessState):
        raise ValueError(f"state 必须是 P6HarnessState，得到 {type(state).__name__}")
    if fam == "selector":
        candidates = _mine_selector(evidence)
    elif fam == "sampler":
        candidates = _mine_sampler(state)
    else:
        candidates = _mine_risk(evidence, c0_bins)

    seen: set = set()
    deduped: List[MinedCandidate] = []
    for c in candidates:                                # 构建即配方序 → keep-first
        if c.candidate_sha in seen:
            continue
        seen.add(c.candidate_sha)
        deduped.append(c)
    return sorted(deduped, key=candidate_sort_key)
