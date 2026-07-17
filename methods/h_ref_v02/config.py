"""p6/harness_state.py — P6 harness 状态（两级冻结 toy-only 机械层的部署面状态）。

P6 的部署面 = P6 自己的 fast path；capability matrix 四项全部在 p6/ 包内闭合。
本模块定义三个可进化组件的 spec + 冻结版本化的 harness 状态容器：

  SelectorSpec   选择器：v0 两种 kind——
                 - "proxy_rank"（默认）：按候选 features["proxy_score"] 降序，tie-break 按
                   program sha 升序；
                 - "weighted_features"：对候选特征 dict 线性打分（Σ w_f · feature_f），
                   可用特征名 = KNOWN_FEATURES（proxy_score / n_steps / has_guard /
                   modified_fraction），未知特征名 validate() 时 raise ValueError。
  SamplerSpec    供给分配：{"allocation": {"det": k1, "random": k2, "llm": k3},
                 "random_params": {...}}。总 K = Σ allocation 是**冻结常量**：构造时传入
                 expected_total，validate() 强制 Σ == expected_total（防预算漂移）。
  RiskRuleSpec   轻量 scope 风险规则：{rule_id, when: 条件列表, then: {action: "ban",
                 target: program_sha 或 op 名}}。**when 原生支持条件列表 = 同规则内 AND**
                 （prereg §4 冻结：bin scope 的 lo/hi 两原子必须同规则合取；拆多规则会在
                 apply_risk 的规则间并集语义下退化为 OR ban）。两种输入形态：
                 单条件 dict {feature, op, value}（向后兼容）或条件 dict 列表；构造期一律
                 规范化为条件元组，canonical 序列化（to_dict/sha）统一为列表形。空列表
                 validate() raise。语义对齐 legacy policy/edits.AddRiskRule（scoped ban），
                 但**独立实现**，不 import 不修改 legacy。每个条件的 feature 强制
                 ∈ P0_FEATURE_ALLOWLIST（prereg §3.3 冻结：{"snr","missing_rate"} ∪ P_FEATS；
                 拒绝 outcome/response/series id/domain id 类特征），validate() 逐条 raise。
  P6HarnessState frozen dataclass（version, selector, sampler, risk_rules, parent_sha,
                 edit_log）。sha() = 语义组件（selector/sampler/risk_rules）canonical JSON
                 （sort_keys、紧凑分隔符）的 sha256[:16]——**剔除易变字段**
                 （version/parent_sha/edit_log 是 provenance，不进 sha）。
  apply_edit     validate→apply→version 递增（"v{n}.e{k}" 语义）→ parent_sha 链 →
                 edit_log 追加 {op, applied, new_version, new_sha}。校验失败 raise P6EditError
                 （单 edit 入口不做静默跳过）。
  default_state  H0 默认 state（本函数是 H0 定义的唯一权威；prereg §4 字面量
                 {det:3, random:5, llm:0}、K=8 请求 slot 预算，见其 docstring）。

约定（两套 validate 契约，勿混淆）：
  - spec 级 `SelectorSpec.validate() / SamplerSpec.validate() / RiskRuleSpec.validate()`
    —— 不合法 raise ValueError（规格原文："未知特征名 validate 时 raise"、"validate 强制
    Σ=expected_total"）。
  - edit 级 `EditOp.validate(state) -> Optional[str]`（见 edit_surfaces.py）——返回拒因字符串，
    由 apply_edit 升格为 P6EditError。

红线：本模块只依赖 stdlib；不读 results/ 与 data/；无 RNG；不修改任何现有文件。
"""
from __future__ import annotations

import hashlib
import json
import operator as _operator
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

__all__ = [
    "DET_PROGRAM_STEPS",
    "GRAMMAR_DENOISERS",
    "GRAMMAR_IMPUTERS",
    "GRAMMAR_OUTLIERS",
    "GRAMMAR_WINDOWS",
    "GUARD_OPS",
    "H0_ALLOCATION",
    "H0_EXPECTED_TOTAL_K",
    "HREF_OPERATOR_DEFAULTS",
    "HRefEditError",
    "HRefState",
    "KNOWN_FEATURES",
    "P0_FEATURE_ALLOWLIST",
    "P6EditError",
    "P6HarnessState",
    "PRESET_SCOPE_FEATURE",
    "PRESET_SCOPE_OP",
    "P_FEATS_FROZEN",
    "RISK_ACTIONS",
    "RISK_OPS",
    "RiskRuleSpec",
    "SCHEMA_VERSION",
    "SELECTOR_KINDS",
    "SUPPLIER_NAMES",
    "SamplerSpec",
    "SelectorSpec",
    "apply_edit",
    "canonical_json",
    "default_state",
]

SCHEMA_VERSION = "p6-harness-state/1"

SELECTOR_KINDS: Tuple[str, ...] = ("proxy_rank", "weighted_features")
KNOWN_FEATURES: Tuple[str, ...] = ("proxy_score", "n_steps", "has_guard", "modified_fraction")
SUPPLIER_NAMES: Tuple[str, ...] = ("det", "random", "llm")
RISK_ACTIONS: Tuple[str, ...] = ("ban",)

_RISK_OP_FNS = {
    "<": _operator.lt,
    "<=": _operator.le,
    ">": _operator.gt,
    ">=": _operator.ge,
    "==": _operator.eq,
    "!=": _operator.ne,
}
RISK_OPS: Tuple[str, ...] = tuple(sorted(_RISK_OP_FNS))

# ——— 冻结 P0 特征 allowlist（prereg §3.3：RiskRule scope 同用）———
# = {"snr","missing_rate"} ∪ P_FEATS。P_FEATS 字面量抄自
# SelfEvolvingHarnessTS/e32_policy.py::P_FEATS（2026-07-11 逐项核对）。
# 为什么抄录而非 import：①本模块 stdlib-only 红线（e32_policy 模块级引 numpy+sklearn）；
# ②e32_policy.py 不在 prereg §7 冻结清单——allowlist 内容必须钉进冻结文件本身，
#   否则冻结 sha 覆盖不到 allowlist 实际取值。
# 核对义务：tests/test_p6_surfaces.py::test_p0_allowlist_matches_e32_policy 每次套件
# 运行时与 e32_policy.P_FEATS 逐项比对（含次序），漂移即红（套件通过是 §8 签发门）。
P_FEATS_FROZEN: Tuple[str, ...] = (
    "period", "trend_strength", "seasonal_strength", "acf1",
    "stationarity_adf", "spectral_entropy", "lumpiness", "outlier_density",
)
P0_FEATURE_ALLOWLIST = frozenset(("snr", "missing_rate") + P_FEATS_FROZEN)

#: preset 成员资格 scope（prereg §4 miner RiskRulePatch (a)："scope = 该 cohort 的定义条件
#: 本身：preset 成员或 P0 bin 区间"）。preset 是**成员资格**维（episode.preset == 值），不是
#: P0 数值特征——故不进 P0_FEATURE_ALLOWLIST（保 test_p0_allowlist_matches_e32_policy 不变），
#: 而作为 RiskRule scope 的保留维单列（F7/finding 37：取代旧 C0 中位数半平面近似）。
#: 求值域 = per-uid fingerprint 的 "preset" 键；op 固定 "=="（成员判定）。
PRESET_SCOPE_FEATURE = "preset"
PRESET_SCOPE_OP = "=="

_WHEN_KEYS = frozenset({"feature", "op", "value"})
_THEN_KEYS = frozenset({"action", "target"})


class HRefEditError(RuntimeError):
    """apply_edit 校验失败（拒因见消息）。"""


def canonical_json(obj: Any) -> str:
    """canonical JSON：sort_keys + ASCII + 紧凑分隔符（sha 的唯一序列化口径）。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha16(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:16]


# ════════════════════════════ SelectorSpec ════════════════════════════
@dataclass(frozen=True)
class SelectorSpec:
    """选择器 spec。kind="proxy_rank"（默认，无 weights）或 "weighted_features"（须给 weights）。"""

    kind: str = "proxy_rank"
    weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(
            self, "weights", {str(k): float(v) for k, v in dict(self.weights or {}).items()}
        )

    def validate(self) -> None:
        """不合法 raise ValueError（spec 级契约）。"""
        if self.kind not in SELECTOR_KINDS:
            raise ValueError(f"未知 selector kind {self.kind!r}（可用：{SELECTOR_KINDS}）")
        if self.kind == "weighted_features":
            if not self.weights:
                raise ValueError("weighted_features 需要非空 weights")
            unknown = sorted(k for k in self.weights if k not in KNOWN_FEATURES)
            if unknown:
                raise ValueError(
                    f"未知特征名 {unknown}（可用：{KNOWN_FEATURES}）"
                )
        elif self.weights:
            raise ValueError("proxy_rank 不接受 weights（线性打分请用 weighted_features）")

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "weights": dict(self.weights)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SelectorSpec":
        return cls(kind=d["kind"], weights=dict(d.get("weights", {})))


# ════════════════════════════ SamplerSpec ════════════════════════════
@dataclass(frozen=True)
class SamplerSpec:
    """供给分配 spec。总 K = Σ allocation 是冻结常量（== expected_total，validate 强制）。

    allocation 必须**显式**含 det/random/llm 三键（无隐式默认——预算必须全部可见），
    值为非负 int。random_params 是 random_grammar_sampler 的参数覆盖（JSON-native 值）。
    """

    allocation: Dict[str, int]
    expected_total: int
    random_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allocation", dict(self.allocation))
        object.__setattr__(self, "random_params", dict(self.random_params or {}))

    def total(self) -> int:
        return sum(int(v) for v in self.allocation.values())

    def validate(self) -> None:
        """不合法 raise ValueError；强制 Σ allocation == expected_total（预算冻结）。"""
        if not isinstance(self.expected_total, int) or isinstance(self.expected_total, bool):
            raise ValueError(f"expected_total 必须是 int，得到 {type(self.expected_total).__name__}")
        if self.expected_total < 1:
            raise ValueError(f"expected_total 必须 ≥ 1，得到 {self.expected_total}")
        keys = set(self.allocation)
        unknown = sorted(keys - set(SUPPLIER_NAMES))
        if unknown:
            raise ValueError(f"未知供给器 {unknown}（可用：{SUPPLIER_NAMES}）")
        missing = sorted(set(SUPPLIER_NAMES) - keys)
        if missing:
            raise ValueError(f"allocation 必须显式含三键 det/random/llm，缺 {missing}")
        for k, v in self.allocation.items():
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise ValueError(f"allocation[{k!r}] 必须是非负 int，得到 {v!r}")
        if self.total() != self.expected_total:
            raise ValueError(
                f"总 K 漂移：Σ allocation = {self.total()} ≠ expected_total = "
                f"{self.expected_total}（总预算是冻结常量）"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocation": dict(self.allocation),
            "expected_total": self.expected_total,
            "random_params": dict(self.random_params),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SamplerSpec":
        return cls(
            allocation=dict(d["allocation"]),
            expected_total=d["expected_total"],
            random_params=dict(d.get("random_params", {})),
        )


# ════════════════════════════ RiskRuleSpec ════════════════════════════
def _normalize_when(when: Any) -> Tuple[Dict[str, Any], ...]:
    """when 形状规范化（内容合法性在 validate()，spec 级 raise 契约）：
    单条件 Mapping → 单元素元组（向后兼容）；list/tuple 条件序列 → 元组（同规则 AND）；
    None/空 → 空元组（validate() 处 raise）；其余形状 → ValueError（构造期响亮拒绝）。"""
    if when is None:
        return ()
    if isinstance(when, Mapping):
        return (dict(when),)
    if isinstance(when, (list, tuple)):
        conds: list = []
        for i, c in enumerate(when):
            if not isinstance(c, Mapping):
                raise ValueError(
                    f"when[{i}] 必须是条件 dict {{feature, op, value}}，得到 {type(c).__name__}"
                )
            conds.append(dict(c))
        return tuple(conds)
    raise ValueError(
        f"when 必须是单条件 dict 或条件 dict 列表，得到 {type(when).__name__}"
    )


@dataclass(frozen=True)
class RiskRuleSpec:
    """轻量 scope 风险规则（语义对齐 legacy AddRiskRule，独立实现）。

    when = 条件元组，每条 {"feature": str, "op": ∈ RISK_OPS, "value": JSON 标量}；
    构造期经 _normalize_when 规范化：单条件 dict（向后兼容）与条件列表都统一为元组。
    **匹配语义 = 同规则内 AND（prereg §4 冻结）**：matches() 要求全部条件在 per-uid
    fingerprint（toy dict）上成立；任一条件的 feature 缺失 → 该条件不成立 → 规则不触发
    （False，保守方向，确定性）。规则之间的并集（OR）ban 语义在 fast_path.apply_risk
    （那是本来正确的层）。空条件列表 validate() raise（空合取=无条件全局 ban，禁止）。
    **每个条件的 feature 强制 ∈ P0_FEATURE_ALLOWLIST**（prereg §3.3 冻结：
    {"snr","missing_rate"} ∪ P_FEATS）——outcome/judge response/series id/domain id 类
    特征一律 validate() raise；fast_path.apply_risk 处再设第二道同 allowlist 闸（纵深防御）。
    then = {"action": "ban", "target": program_sha 或 op 名}：触发时剔除命中候选
    （命中 = 候选 sha == target，或 target ∈ 候选程序的 op 名集合；求值在 fast_path.apply_risk）。
    canonical 序列化（to_dict → payload/sha）一律列表形——单条件规则的 sha 与其
    单元素列表形完全一致（sha 稳定）。
    """

    rule_id: str
    when: Tuple[Dict[str, Any], ...]
    then: Dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", str(self.rule_id))
        object.__setattr__(self, "when", _normalize_when(self.when))
        object.__setattr__(self, "then", dict(self.then or {}))

    def validate(self) -> None:
        """不合法 raise ValueError（spec 级契约）；when 逐条件校验。"""
        if not self.rule_id:
            raise ValueError("rule_id 不能为空")
        if not self.when:
            raise ValueError(
                "when 条件列表不能为空（至少 1 个 {feature, op, value} 条件；"
                "空合取 = 无条件全局 ban，禁止）"
            )
        for i, cond in enumerate(self.when):
            if set(cond) != _WHEN_KEYS:
                raise ValueError(
                    f"when[{i}] 必须恰含键 {sorted(_WHEN_KEYS)}，得到 {sorted(cond)}"
                )
            if not isinstance(cond["feature"], str) or not cond["feature"]:
                raise ValueError(f"when[{i}].feature 必须是非空 str")
            if cond["feature"] == PRESET_SCOPE_FEATURE:
                # preset 成员资格 scope（F7）：op 固定 "=="、value 是非空 preset 名 str。
                if cond["op"] != PRESET_SCOPE_OP:
                    raise ValueError(
                        f"when[{i}] preset scope 的 op 必须为 {PRESET_SCOPE_OP!r}（成员资格判定），"
                        f"得到 {cond['op']!r}"
                    )
                if not isinstance(cond["value"], str) or not cond["value"]:
                    raise ValueError(f"when[{i}] preset scope 的 value 必须是非空 preset 名 str")
                continue
            if cond["feature"] not in P0_FEATURE_ALLOWLIST:
                raise ValueError(
                    f"when[{i}].feature {cond['feature']!r} 不在冻结 P0 allowlist"
                    f"（prereg §3.3：{sorted(P0_FEATURE_ALLOWLIST)}）——outcome/response/"
                    f"series id/domain id 类特征禁止进入 RiskRule scope"
                    f"（preset 成员资格用保留维 {PRESET_SCOPE_FEATURE!r}）"
                )
            if cond["op"] not in _RISK_OP_FNS:
                raise ValueError(f"when[{i}] 未知比较 op {cond['op']!r}（可用：{RISK_OPS}）")
        if set(self.then) != _THEN_KEYS:
            raise ValueError(f"then 必须恰含键 {sorted(_THEN_KEYS)}，得到 {sorted(self.then)}")
        if self.then["action"] not in RISK_ACTIONS:
            raise ValueError(f"未知 action {self.then['action']!r}（可用：{RISK_ACTIONS}）")
        if not isinstance(self.then["target"], str) or not self.then["target"]:
            raise ValueError("then.target 必须是非空 str（program sha 或 op 名）")

    def matches(self, fingerprint: Mapping[str, Any]) -> bool:
        """同规则 AND：**全部**条件在 fingerprint 上成立才触发；任一条件的 feature 缺失
        → 该条件不成立 → False（保守方向）。空条件列表 → False（validate 已禁，兜底不 ban）。
        数值 op 期望数值特征。"""
        if not self.when:
            return False
        for cond in self.when:
            feat = cond["feature"]
            if feat not in fingerprint:
                return False
            if not _RISK_OP_FNS[cond["op"]](fingerprint[feat], cond["value"]):
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """canonical 形：when 一律条件列表（单条件也是单元素列表——sha 稳定口径）。"""
        return {
            "rule_id": self.rule_id,
            "when": [dict(c) for c in self.when],
            "then": dict(self.then),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RiskRuleSpec":
        """接受两种 when 形态（单条件 dict 向后兼容 / 条件列表）；规范化在 __post_init__。"""
        return cls(rule_id=d["rule_id"], when=d["when"], then=dict(d["then"]))


# ════════════════════════════ HRefState ════════════════════════════
_STATE_COMPONENTS: Tuple[str, ...] = ("selector", "sampler", "risk_rules")


@dataclass(frozen=True)
class HRefState:
    """P6 部署面状态（frozen）。构造即校验（__post_init__ 跑全部 spec 级 validate）。

    sha() 只覆盖语义组件 {selector, sampler, risk_rules}；version/parent_sha/edit_log
    是 provenance（易变字段），**不进 sha**——因此 sha 是"语义身份"，同组件不同版本号
    sha 相同（也使 edit_log 里记 new_sha 无循环依赖）。
    """

    version: str
    selector: SelectorSpec
    sampler: SamplerSpec
    risk_rules: Tuple[RiskRuleSpec, ...] = ()
    parent_sha: Optional[str] = None
    edit_log: Tuple[Dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_rules", tuple(self.risk_rules or ()))
        object.__setattr__(self, "edit_log", tuple(dict(e) for e in (self.edit_log or ())))
        self.selector.validate()
        self.sampler.validate()
        seen_ids = set()
        for r in self.risk_rules:
            r.validate()
            if r.rule_id in seen_ids:
                raise ValueError(f"重复 rule_id：{r.rule_id!r}")
            seen_ids.add(r.rule_id)

    # —— 语义 payload 与 sha ——
    def payload(self) -> Dict[str, Any]:
        """语义组件的 canonical payload（provenance 字段剔除）。"""
        return {
            "schema": SCHEMA_VERSION,
            "selector": self.selector.to_dict(),
            "sampler": self.sampler.to_dict(),
            "risk_rules": [r.to_dict() for r in self.risk_rules],
        }

    def sha(self) -> str:
        return _sha16(self.payload())

    def sha_excluding(self, *components: str) -> str:
        """剔除指定语义组件后的 sha（配对语义用：如 sha_excluding("selector") =
        「除 selector 外」的身份）。components ⊆ {selector, sampler, risk_rules}。"""
        p = self.payload()
        for c in components:
            if c not in _STATE_COMPONENTS:
                raise ValueError(f"未知组件 {c!r}（可用：{_STATE_COMPONENTS}）")
            p.pop(c)
        return _sha16(p)


# ════════════════════════════ apply_edit ════════════════════════════
_VERSION_RE = re.compile(r"^(?P<base>.+)\.e(?P<k>\d+)$")


def _bump_version(version: str) -> str:
    """"v{n}" → "v{n}.e1"；"v{n}.e{k}" → "v{n}.e{k+1}"（编辑计数在同一代内单调递增）。"""
    m = _VERSION_RE.match(version)
    if m:
        return f"{m.group('base')}.e{int(m.group('k')) + 1}"
    return f"{version}.e1"


def apply_edit(state: HRefState, edit: Any) -> HRefState:
    """validate→apply→版本递增→parent_sha 链→edit_log 追加。

    edit 是任何满足 EditOp 契约的对象（validate(state)->Optional[str]、apply(state)->state、
    to_dict()；见 edit_surfaces.py——鸭子类型，避免循环 import）。
    校验失败 raise P6EditError（单 edit 入口，不静默跳过、不落 rejected 日志）。
    不可变：绝不改 state 入参。
    """
    reason = edit.validate(state)
    if reason is not None:
        raise HRefEditError(f"edit 被拒（{edit.to_dict().get('kind', '?')}）：{reason}")
    applied = edit.apply(state)                     # 纯组件替换（不动 provenance 字段）
    new_sha = applied.sha()                         # sha 不含 edit_log → 无循环
    new_version = _bump_version(state.version)
    entry = {
        "op": edit.to_dict(),
        "applied": True,
        "new_version": new_version,
        "new_sha": new_sha,
    }
    return replace(
        applied,
        version=new_version,
        parent_sha=state.sha(),
        edit_log=state.edit_log + (entry,),
    )


# ════════════════════════════ H0 默认 state ════════════════════════════
H0_ALLOCATION: Dict[str, int] = {"det": 3, "random": 5, "llm": 0}   # prereg §4 代码字面量
H0_EXPECTED_TOTAL_K: int = 8                                        # Σ H0_ALLOCATION（冻结）
HREF_OPERATOR_DEFAULTS = {
    "denoise_savgol": {"window": 11, "order": 3},
    "denoise_median": {"window": 5},
    "smooth_ma": {"window": 5},
    "stl_decompose": {"period": 0},
}
GUARD_OPS = ("winsorize", "outlier_iqr", "outlier_mad")
GRAMMAR_IMPUTERS = ("impute_linear", "impute_ema")
GRAMMAR_OUTLIERS = ("winsorize", "outlier_iqr", "outlier_mad")
GRAMMAR_DENOISERS = ("denoise_median", "smooth_ma", "denoise_savgol")
GRAMMAR_WINDOWS = (5, 9, 15, 25)
DET_PROGRAM_STEPS = (
    (("impute_linear", {}),),
    (("impute_linear", {}), ("winsorize", {}), ("denoise_savgol", {})),
    (("impute_linear", {}), ("denoise_median", {"window": 9})),
)


def default_state(expected_total_K: int = H0_EXPECTED_TOTAL_K) -> HRefState:
    """H0 默认 state（本函数为 H0 定义的唯一权威；prereg §4 代码字面量，**非均分公式**）。

    定义（prereg §4 冻结）：
      - selector   = proxy_rank（proxy_score 降序，tie-break program sha 升序）；
      - sampler    = **字面量 {det: 3, random: 5, llm: 0}**（H0_ALLOCATION），
        Σ = K = 8（H0_EXPECTED_TOTAL_K）。llm = 0 使 H0 离线安全（llm_supplier
        默认 None → 贡献 0 候选，不得联网）；
      - risk_rules = ()；version = "v0"；parent_sha = None；edit_log = ()。

    K 语义（prereg §4）：**K=8 是请求 slot 预算，不是候选数承诺**——det 阶梯固定
    3 程序（det 分配 >3 时如实短缺不回填）；跨 supplier sha 去重后 realized unique
    pool 可 <8，由 fast_path 逐 episode 落账（generate_candidates_with_stats /
    run_fast_path().pool_stats）。

    expected_total 仍冻结校验：传入 expected_total_K ≠ 8 时，SamplerSpec.validate 的
    Σ allocation == expected_total 闸在构造期 raise ValueError（预算漂移一律响亮拒绝）。
    """
    return HRefState(
        version="v0",
        selector=SelectorSpec(kind="proxy_rank"),
        sampler=SamplerSpec(
            allocation=dict(H0_ALLOCATION),
            expected_total=int(expected_total_K),
        ),
        risk_rules=(),
    )


# Transitional names used only by legacy P6 imports.
P6HarnessState = HRefState
P6EditError = HRefEditError
