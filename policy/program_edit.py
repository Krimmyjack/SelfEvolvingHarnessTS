"""policy/program_edit.py — B1b-mini 开放程序编辑面（压缩计划 §2）。

B1a（run_proposer.SPACE）编辑面 = 有限 scoped risk 规则（**只在 10 个缓存池动作间重路由** →
cache-replay 可评）。B1b **本质不同**：编辑面 = **提议新程序** = 现有算子组 1–3 步有序链
（可改顺序/参数/适用条件），**新链不在缓存 L_test → 须真实执行**（这是唯一主要开销）。

程序空间（不可枚举，N=6–10 预算下 = LLM 生态位真身份检验）：
  step1（必需）= imputer ∈ {impute_linear, impute_fft, impute_ema, period_complete}
  step2/3（可选）= outlier ∈ {winsorize, outlier_iqr, outlier_mad}
                 | denoise ∈ {denoise_median, denoise_savgol, denoise_wavelet, denoise_stl, smooth_ma}
  窗算子（denoise_median/smooth_ma）带 window ∈ {5,9,15,25}；顺序有意义（denoise∘winsor ≠ winsor∘denoise）。
  scope = 适用 cell 集（在这些 cell 覆盖 frozen pick 为本程序）。

每个 ProgramSpec → ActionSpec（现有 ActionCompiler 直接可编译执行，无需新执行机）→ overlay 消费面。

**红线**：只组现有已注册算子（供给不变，变的是组合/顺序/参数）；不改 PatternSpec、不造新算子。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .action_spec import ActionSpec, ActionStep

# ── forecast 可用算子（运行时按 registry 契约核验；这里固定 curated 子集 = 池语义的预处理算子）──
IMPUTERS = ("impute_linear", "impute_fft", "impute_ema", "period_complete")
OUTLIERS = ("winsorize", "outlier_iqr", "outlier_mad")
DENOISERS = ("denoise_median", "denoise_savgol", "denoise_wavelet", "denoise_stl", "smooth_ma")
WINDOWED = {"denoise_median", "smooth_ma"}
WINDOW_GRID = (5, 9, 15, 25)
_ALL_OPS = frozenset(IMPUTERS + OUTLIERS + DENOISERS)


@dataclass(frozen=True)
class ProgramSpec:
    """一条候选程序：有序 (op, params) 链 + 适用 scope（cell 集）。不可变、SHA 身份。"""
    steps: Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...]   # ((op, (("window",9),)), ...)
    scope: Tuple[str, ...]                                       # 适用 cell_id 集（非空）
    provenance: Dict[str, Any] = field(default_factory=dict)     # {source, arm, ...}

    def sha(self) -> str:
        payload = {"steps": [[op, dict(p)] for op, p in self.steps], "scope": sorted(self.scope)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                              .encode()).hexdigest()[:12]

    @property
    def action_id(self) -> str:
        return f"prog_{self.sha()}"

    def chain_sig(self) -> Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...]:
        """仅算子链（不含 scope）的规范签名——判 novelty / 去重执行用。"""
        return tuple((op, tuple(sorted(p))) for op, p in self.steps)

    def chain_sha(self) -> str:
        """scope 无关的 **resolved** 算子链 SHA——执行器缓存键 + 去重键。用编译后 (op, window) 身份
        （非 raw steps）→ 防 bare savgol/median 默认窗静默等价池动作的 identity 伪影（Stage 1 算子身份教训）。"""
        return hashlib.sha256(json.dumps(list(resolved_sig(self)),
                                         default=str).encode()).hexdigest()[:12]


def _op_ok(op: str) -> bool:
    """registry 契约核验：canonical + 支持 forecast（fail-loud 之外的显式护栏）。"""
    from ..operators.registry import OPERATOR_METADATA, canonicalize
    meta = OPERATOR_METADATA.get(canonicalize(op))
    return meta is not None and "forecast" in meta.get("allowed_tasks", ())


def validate(spec: ProgramSpec) -> Tuple[bool, str]:
    """机械 Gate（评估前免费过滤；非法 → (False, 原因)）。"""
    if not spec.steps or len(spec.steps) > 3:
        return False, "步数须 1..3"
    ops = [op for op, _ in spec.steps]
    if ops[0] not in IMPUTERS:
        return False, "首步须为 imputer（下游要求缺失已处理）"
    for op, params in spec.steps:
        if op not in _ALL_OPS or not _op_ok(op):
            return False, f"算子 {op!r} 不在 forecast 可用集/未注册"
        if op in WINDOWED:
            w = dict(params).get("window")
            if w is None or int(w) not in WINDOW_GRID:
                return False, f"{op} 窗须 ∈ {WINDOW_GRID}"
        elif params:
            return False, f"{op} 不接受参数覆盖（本 grammar）"
    if any(ops[i] == ops[i + 1] for i in range(len(ops) - 1)):
        return False, "禁相邻重复算子"
    # 顺序契约：imputer 只在首步；outlier/denoise 在其后（denoise 可多步，顺序自由）
    if any(o in IMPUTERS for o in ops[1:]):
        return False, "imputer 只能在首步"
    if not spec.scope:
        return False, "scope 不得为空（作用域纪律）"
    return True, "ok"


def to_action_spec(spec: ProgramSpec, defaults: Optional[Mapping[str, Mapping]] = None) -> ActionSpec:
    """ProgramSpec → ActionSpec（defaults ⊕ override 完整 resolve，与 action_menu_v1 同语义 →
    ActionCompiler 直接可编译）。"""
    if defaults is None:
        from ..harness.layers import minimal_l2
        defaults = minimal_l2().operator_defaults
    from .action_spec import _task_constraints
    steps = tuple(ActionStep(op, {**defaults.get(op, {}), **dict(params)}) for op, params in spec.steps)
    return ActionSpec(action_id=spec.action_id, steps=steps,
                      task_constraints=_task_constraints([op for op, _ in spec.steps]),
                      model_constraints=None,
                      provenance={"source": "program_edit", "sha": spec.sha(),
                                  "scope": list(spec.scope), **spec.provenance})


def resolved_sig(spec: ProgramSpec) -> Tuple[Tuple[str, Any], ...]:
    """编译后（defaults ⊕ override）的 (op, window) 规范签名——novelty / dedup / executor 缓存的
    **唯一真身份**。bare denoise_savgol 默认窗=11 → 等价 v_savgol；此签名捕获它（raw steps 不能）。"""
    a = to_action_spec(spec)
    return tuple((s.op, s.params.get("window")) for s in a.steps)


def pool_resolved_sigs() -> set:
    """15 menu 动作的 **resolved** (op, window) 签名集——novelty 判定基准。"""
    from .action_spec import action_menu_v1
    return {tuple((s.op, s.params.get("window")) for s in spec.steps)
            for spec in action_menu_v1().actions.values()}


# 兼容旧名（raw 链签名；is_novel 已改用 resolved）
def pool_chain_sigs() -> set:
    return pool_resolved_sigs()


def is_novel(spec: ProgramSpec, pool_sigs: Optional[set] = None) -> bool:
    """程序是否 ≠ 任何已有 menu/池动作（按**编译后** (op, window) 身份比较；scope 无关）。"""
    if pool_sigs is None:
        pool_sigs = pool_resolved_sigs()
    return resolved_sig(spec) not in pool_sigs


# ═════════════════ ProgramSpec grammar v1（P0，Final_Plan_CodeAgentFirst_2026-07-09 §P0）═════════════════
#
# v0（上方）= B1b 冻结面，bit 级不动（重放/缓存键依赖）。v1 的质变（本地约束 C1：单纯开空间
# 救不了 code agent，扩张必须质变）：
#   task_type 显式（registry allowed_tasks 契约按任务过滤，anomaly 物理禁平滑/删改）
#   + pattern_guard（结构条件适用域，白名单特征 = P_FEATS ∪ {snr, missing_rate}）
#   + invariants（保长 + 观测点修改率预算；no_future_access 为构造性声明）
#   + fallback（guard 不满足/gate 拒绝时的显式回退动作）
#   + risk_budget_beta（β ∈ [0,1] 风险上限；修改率预算默认 = β，显式给出时不得超 β）。

_GUARD_COMPARATORS = ("<", "<=", ">", ">=", "==")


def _guard_feats() -> frozenset:
    from ..e32_policy import P_FEATS
    return frozenset(P_FEATS) | {"snr", "missing_rate"}


def _op_allows_task(op: str, task: str) -> bool:
    from ..operators.registry import OPERATOR_METADATA, canonicalize
    meta = OPERATOR_METADATA.get(canonicalize(op))
    return meta is not None and task in meta.get("allowed_tasks", ())


@lru_cache(maxsize=1)
def _known_fallbacks() -> frozenset:
    # menu v1 冻结（改动作集=新 SHA），构建结果可安全缓存——validate_v1 在 gym 循环里是热点
    from .action_semantics import V_IMPUTE_LINEAR, V_RAW_IDENTITY
    from .action_spec import action_menu_v1
    return frozenset(action_menu_v1().actions) | {V_RAW_IDENTITY, V_IMPUTE_LINEAR}


@dataclass(frozen=True)
class ProgramSpecV1:
    """grammar v1 候选程序：有序链 + task 契约 + guard + β/修改率预算 + fallback。SHA 身份。"""
    steps: Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...]
    scope: Tuple[str, ...]
    task_type: str = "forecast"
    pattern_guard: Tuple[Tuple[str, str, float], ...] = ()
    risk_budget_beta: float = 0.3
    fallback: str = "v_raw_identity"
    max_modified_fraction: Optional[float] = None     # None → 默认 = risk_budget_beta
    provenance: Dict[str, Any] = field(default_factory=dict)

    def resolved_budget(self) -> float:
        return (float(self.risk_budget_beta) if self.max_modified_fraction is None
                else float(self.max_modified_fraction))

    def invariants(self) -> Dict[str, Any]:
        return {"no_future_access": True, "preserve_length": True,
                "max_modified_fraction": self.resolved_budget()}

    def sha(self) -> str:
        payload = {"grammar": "v1",
                   "steps": [[op, dict(p)] for op, p in self.steps],
                   "scope": sorted(self.scope),
                   "task_type": self.task_type,
                   "pattern_guard": [list(g) for g in self.pattern_guard],
                   "risk_budget_beta": float(self.risk_budget_beta),
                   # 身份用 resolved 预算：None 与显式等于 β 语义等价 → 同 SHA（防重复真实执行）
                   "max_modified_fraction": self.resolved_budget(),
                   "fallback": self.fallback}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                              .encode()).hexdigest()[:12]

    @property
    def action_id(self) -> str:
        return f"prog1_{self.sha()}"

    def chain_sha(self) -> str:
        """resolved 算子链 SHA（与 v0 同语义：guard/scope/β 无关，执行去重键）。"""
        sig = tuple((op, params.get("window")) for op, params in _resolved_steps_v1(self))
        return hashlib.sha256(json.dumps(list(sig), default=str).encode()).hexdigest()[:12]


def _resolved_steps_v1(spec: ProgramSpecV1,
                       defaults: Optional[Mapping[str, Mapping]] = None
                       ) -> Tuple[Tuple[str, Dict[str, Any]], ...]:
    """defaults ⊕ override 完整 resolve（与 v0 to_action_spec / action_menu_v1 同一合成语义）。"""
    if defaults is None:
        from ..harness.layers import minimal_l2
        defaults = minimal_l2().operator_defaults
    return tuple((op, {**defaults.get(op, {}), **dict(params)}) for op, params in spec.steps)


def validate_v1(spec: ProgramSpecV1) -> Tuple[bool, str]:
    """机械 Gate v1（评估前免费过滤；非法 → (False, 原因)）。v0 机械规则全保留 + 四类新约束。"""
    from .task_spec import TASK_TYPES
    if spec.task_type not in TASK_TYPES:
        return False, f"task_type 须 ∈ {TASK_TYPES}，得到 {spec.task_type!r}"
    beta = spec.risk_budget_beta
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or not (0.0 <= float(beta) <= 1.0):
        return False, f"risk_budget_beta 须 ∈ [0,1]，得到 {beta!r}"
    if spec.max_modified_fraction is not None:
        m = spec.max_modified_fraction
        if not isinstance(m, (int, float)) or isinstance(m, bool) or not (0.0 <= float(m) <= 1.0):
            return False, f"max_modified_fraction 须 ∈ [0,1]，得到 {m!r}"
        if float(m) > float(beta) + 1e-12:
            return False, (f"max_modified_fraction={m} 不得超过 risk_budget_beta={beta}"
                           "（β 是风险上限，预算不得越权）")
    feats = _guard_feats()
    for g in spec.pattern_guard:
        if not isinstance(g, tuple) or len(g) != 3:
            return False, "pattern_guard 条目须为 (feature, cmp, value) 三元组"
        feat, cmp, val = g
        if feat not in feats:
            return False, (f"pattern_guard 特征 {feat!r} 不在白名单"
                           "（P_FEATS ∪ {snr, missing_rate}）")
        if cmp not in _GUARD_COMPARATORS:
            return False, f"pattern_guard 比较子须 ∈ {_GUARD_COMPARATORS}，得到 {cmp!r}"
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return False, f"pattern_guard 阈值须为数值，得到 {val!r}"
    if spec.fallback not in _known_fallbacks():
        return False, (f"fallback {spec.fallback!r} 不是已知动作"
                       "（menu v1 ∪ {v_raw_identity, v_impute_linear}）")
    if not spec.steps or len(spec.steps) > 3:
        return False, "步数须 1..3"
    ops = [op for op, _ in spec.steps]
    if ops[0] not in IMPUTERS:
        return False, "首步须为 imputer（下游要求缺失已处理）"
    for op, params in spec.steps:
        if op not in _ALL_OPS:
            return False, f"算子 {op!r} 不在 grammar 算子集"
        if not _op_allows_task(op, spec.task_type):
            return False, (f"算子 {op!r} 不支持 task={spec.task_type!r}"
                           "（registry allowed_tasks 契约：anomaly_detection 物理禁平滑/删改）")
        if op in WINDOWED:
            w = dict(params).get("window")
            # 须为精确 int（int(w) 隶属判定会放行 9.0/"9"，真实执行时 medfilt 拒绝→烧执行预算）
            if not isinstance(w, int) or isinstance(w, bool) or w not in WINDOW_GRID:
                return False, f"{op} 窗须为 int 且 ∈ {WINDOW_GRID}，得到 {w!r}"
        elif params:
            return False, f"{op} 不接受参数覆盖（本 grammar）"
    if any(ops[i] == ops[i + 1] for i in range(len(ops) - 1)):
        return False, "禁相邻重复算子"
    if any(o in IMPUTERS for o in ops[1:]):
        return False, "imputer 只能在首步"
    if not spec.scope:
        return False, "scope 不得为空（作用域纪律）"
    return True, "ok"


def guard_matches(spec: ProgramSpecV1, pattern_summary: Mapping[str, Any]) -> bool:
    """guard 评估。pattern_summary 形状 = EvidencePacket["pattern"]（snr/missing_rate 顶层，
    结构特征在 struct_feats）。缺特征 **fail-loud**（KeyError）——guard 不允许静默放行。"""
    struct = pattern_summary.get("struct_feats") or {}
    for feat, cmp, threshold in spec.pattern_guard:
        # None 值 = 特征不可得，与键缺失同语义（P5-A 实测：summary 带 snr=None 时曾炸 TypeError）
        if feat in ("snr", "missing_rate"):
            raw_value = pattern_summary.get(feat)
            if raw_value is None:
                raise KeyError(f"pattern_guard 特征 {feat!r} 在 pattern summary 顶层缺失/为 None（fail-loud）")
            value = float(raw_value)
        else:
            raw_value = struct.get(feat)
            if raw_value is None:
                raise KeyError(f"pattern_guard 特征 {feat!r} 在 struct_feats 缺失/为 None（fail-loud）")
            value = float(raw_value)
        ok = {"<": value < threshold, "<=": value <= threshold,
              ">": value > threshold, ">=": value >= threshold,
              "==": value == threshold}[cmp]
        if not ok:
            return False
    return True


def check_execution_invariants(spec: ProgramSpecV1, x_in, x_out) -> Tuple[bool, Dict[str, Any]]:
    """执行后不变量：preserve_length + 观测点修改率 ≤ resolved budget。

    修改率只数**原本有限**的位置——缺失填补是 imputer 本职，不计入 distortion 预算；
    观测点被改成非有限同样计为修改。no_future_access 是构造性声明（registry 算子皆为
    离线 series→series，执行器不接收 future），不在此做数值检验。
    """
    x_in = np.asarray(x_in, dtype=float).ravel()
    x_out = np.asarray(x_out, dtype=float).ravel()
    detail: Dict[str, Any] = {"n_in": int(x_in.size), "n_out": int(x_out.size),
                              "max_modified_fraction": spec.resolved_budget()}
    violations: List[str] = []
    if x_in.size != x_out.size:
        violations.append("preserve_length")
        detail["modified_fraction"] = None
        detail["violations"] = violations
        return False, detail
    finite = np.isfinite(x_in)
    n_obs = int(finite.sum())
    if n_obs == 0:
        frac = 0.0
    else:
        a, b = x_in[finite], x_out[finite]
        changed = ~np.isclose(a, b, rtol=1e-9, atol=1e-12)   # 观测点改成 NaN/inf 也算改（isclose 恒 False）
        frac = float(changed.sum()) / n_obs
    detail["modified_fraction"] = frac
    if frac > spec.resolved_budget() + 1e-12:
        violations.append("max_modified_fraction")
    detail["violations"] = violations
    return not violations, detail


def spec_v1_to_dict(spec: ProgramSpecV1) -> Dict[str, Any]:
    """canonical JSON 载体（TypedCandidate.program_spec / LLM 输出契约）。不含 provenance（非身份）。"""
    return {"grammar": "v1",
            "steps": [[op, dict(p)] for op, p in spec.steps],
            "scope": list(spec.scope),
            "task_type": spec.task_type,
            "pattern_guard": [list(g) for g in spec.pattern_guard],
            "risk_budget_beta": float(spec.risk_budget_beta),
            "max_modified_fraction": spec.max_modified_fraction,
            "fallback": spec.fallback}


def spec_v1_from_dict(d: Any) -> ProgramSpecV1:
    """严格反序列化（fail-loud ValueError）——composer/LLM 输出的唯一入口。
    只做结构合法性；深层合法性（算子契约/剂量网格/guard 白名单/β 范围）由 validate_v1 负责。"""
    if not isinstance(d, Mapping):
        raise ValueError(f"ProgramSpec v1 须为 JSON object，得到 {type(d).__name__}")
    if d.get("grammar", "v1") != "v1":
        raise ValueError(f"grammar 须为 'v1'，得到 {d.get('grammar')!r}")
    raw_steps = d.get("steps")
    if not isinstance(raw_steps, (list, tuple)) or not raw_steps:
        raise ValueError("steps 须为非空列表")
    steps: List[Tuple[str, Tuple[Tuple[str, Any], ...]]] = []
    for item in raw_steps:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str):
            raise ValueError(f"step 须为 [op, params]，得到 {item!r}")
        op, params = item
        if not isinstance(params, Mapping):
            raise ValueError(f"step params 须为 object，得到 {params!r}")
        steps.append((op, tuple(sorted(dict(params).items()))))
    scope = d.get("scope")
    if not isinstance(scope, (list, tuple)) or not all(isinstance(s, str) for s in scope):
        raise ValueError("scope 须为字符串列表")
    guard_raw = d.get("pattern_guard") or []
    if not isinstance(guard_raw, (list, tuple)):
        raise ValueError("pattern_guard 须为列表")
    guard: List[Tuple[str, str, Any]] = []
    for g in guard_raw:
        if not isinstance(g, (list, tuple)) or len(g) != 3:
            raise ValueError(f"pattern_guard 条目须为 [feature, cmp, value]，得到 {g!r}")
        guard.append((str(g[0]), str(g[1]), g[2]))
    beta = d.get("risk_budget_beta", 0.3)
    if not isinstance(beta, (int, float)) or isinstance(beta, bool):
        raise ValueError(f"risk_budget_beta 须为数值，得到 {beta!r}")
    m = d.get("max_modified_fraction")
    if m is not None and (not isinstance(m, (int, float)) or isinstance(m, bool)):
        raise ValueError(f"max_modified_fraction 须为数值或 null，得到 {m!r}")
    fallback = d.get("fallback", "v_raw_identity")
    if not isinstance(fallback, str) or not fallback:
        raise ValueError("fallback 须为非空字符串")
    return ProgramSpecV1(
        steps=tuple(steps),
        scope=tuple(str(s) for s in scope),
        task_type=str(d.get("task_type", "forecast")),
        pattern_guard=tuple(guard),
        risk_budget_beta=float(beta),
        fallback=fallback,
        max_modified_fraction=None if m is None else float(m),
    )


def is_novel_v1(spec: ProgramSpecV1, pool_sigs: Optional[set] = None) -> bool:
    """v1 程序是否 ≠ 任何冻结 menu 动作（按编译后 (op, window) resolved 身份；guard/scope/β 无关）。"""
    if pool_sigs is None:
        pool_sigs = pool_resolved_sigs()
    sig = tuple((op, params.get("window")) for op, params in _resolved_steps_v1(spec))
    return sig not in pool_sigs


def to_action_spec_v1(spec: ProgramSpecV1,
                      defaults: Optional[Mapping[str, Mapping]] = None) -> ActionSpec:
    """ProgramSpecV1 → ActionSpec（defaults ⊕ override 完整 resolve；ActionCompiler 直接可编译）。
    task_type 必须被链算子契约允许（validate_v1 已保证；此处双保险 fail-loud）。"""
    from .action_spec import _task_constraints
    resolved = _resolved_steps_v1(spec, defaults)
    steps = tuple(ActionStep(op, params) for op, params in resolved)
    constraints = _task_constraints([op for op, _ in resolved])
    if spec.task_type not in constraints:
        raise ValueError(
            f"ProgramSpecV1 task_type={spec.task_type!r} 不被链算子契约允许（{constraints}）"
            "——先过 validate_v1")
    return ActionSpec(
        action_id=spec.action_id, steps=steps,
        task_constraints=constraints, model_constraints=None,
        provenance={"source": "program_edit_v1", "grammar": "v1",
                    "sha": spec.sha(), "chain_sha": spec.chain_sha(),
                    "scope": list(spec.scope), "task_type": spec.task_type,
                    "pattern_guard": [list(g) for g in spec.pattern_guard],
                    "risk_budget_beta": float(spec.risk_budget_beta),
                    "max_modified_fraction": spec.resolved_budget(),
                    "fallback": spec.fallback, **spec.provenance})
