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
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
