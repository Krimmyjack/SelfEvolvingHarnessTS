"""policy/action_spec.py — ActionSpec / ActionMenu / ActionCompiler（Stage 2.0-②b，v1.1b）。

E-3.2 Router 输出的 action ID（v_none / v_median / f0_median_w9 / …）**不是** Operator
Registry 的原子算子，而是实验代码构造的完整 harness 变体（如 f0_median_w9 =
impute_linear → denoise_median(window=9)）。本模块把这一隐式事实变成显式契约：

  ActionSpec     动作身份：action_id + 有序 steps(op, params_override) + task/model 约束 + provenance
  ActionMenu     版本化动作集（canonical JSON SHA）——Router artifact 与菜单版本互相核验
  ActionCompiler ActionSpec + conditioning context → 可执行 Program

定义单一真源 = run_main_table._VARIANT_SPECS + family0_actions.F0_DOSAGE_GRID（menu 构建
时直接读取，禁止第二处复制清单）。编译不平行实现参数合成逻辑：to_harness 与实验构造器
（fixed_harness_variants / dosage_variant）同构（minimal + 单全局模板），to_program 走同一
fast_path.compose 模板路径（operator_defaults + params_override）→ **Program 等价由构造保证**，
tests/test_policy_contract.py 四层等价性测试守回归。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..harness import HarnessState
from ..harness.layers import PipelineTemplate
from ..fast_path.compose import Program, compose

_TASK_ORDER = ("forecast", "classification", "anomaly_detection")


@dataclass(frozen=True)
class ActionStep:
    """一步 = 算子 + **完整 resolved 参数**（Step 1.1-②，评审第二十三轮）。

    v1 首版只存 override、缺省参数编译时取 harness defaults——那使 defaults 改动后
    "menu SHA 不变而 Program 行为已变"。现在 menu 构建时就把 defaults ⊕ override 完整
    resolve 进 params → menu SHA 绑定动作的**执行语义**；且编译时 override 优先级最高，
    旧 menu 在新 defaults 下仍 bit 级重放旧行为（跨版本重放保证）。"""
    op: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"op": self.op, "params": dict(self.params)}


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    steps: Tuple[ActionStep, ...]                       # 有序
    task_constraints: Tuple[str, ...]                   # 全 steps 的 registry allowed_tasks 交集
    model_constraints: Optional[Tuple[str, ...]]        # P0=None；2.3 张量 pilot 后启用（C6）
    provenance: Dict[str, Any] = field(default_factory=dict)   # {source, menu_version}

    def to_dict(self) -> Dict[str, Any]:
        return {"action_id": self.action_id,
                "steps": [s.to_dict() for s in self.steps],
                "task_constraints": list(self.task_constraints),
                "model_constraints": (list(self.model_constraints)
                                      if self.model_constraints is not None else None),
                "provenance": dict(self.provenance)}


def _task_constraints(ops: List[str]) -> Tuple[str, ...]:
    """registry 契约交集（单一真源；alias 先 canonical 化）。"""
    from ..operators.registry import OPERATOR_METADATA, canonicalize
    allowed = set(_TASK_ORDER)
    for op in ops:
        meta = OPERATOR_METADATA.get(canonicalize(op))
        if meta is None:
            raise KeyError(f"ActionSpec 引用未注册算子: {op!r}")
        allowed &= set(meta["allowed_tasks"])
    return tuple(t for t in _TASK_ORDER if t in allowed)


class ActionMenu:
    """版本化动作集。sha256 = canonical JSON 摘要（含 meta：defaults 指纹等语义身份字段）
    ——RouterPolicy 的 provenance 与张量协议都引用它；改动作集/改语义 = 新 SHA，不原地编辑。"""

    def __init__(self, version: str, actions: List[ActionSpec], meta: Optional[Dict[str, Any]] = None):
        ids = [a.action_id for a in actions]
        if len(ids) != len(set(ids)):
            raise ValueError("ActionMenu 内 action_id 重复")
        self.version = version
        self.actions: Dict[str, ActionSpec] = {a.action_id: a for a in actions}
        self.meta = dict(meta or {})
        self.sha256 = hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "meta": self.meta,
                "actions": {aid: spec.to_dict() for aid, spec in sorted(self.actions.items())}}

    def __contains__(self, action_id: str) -> bool:
        return action_id in self.actions

    def __len__(self) -> int:
        return len(self.actions)


def action_menu_v1() -> ActionMenu:
    """Menu v1 = Stage-1 冻结动作全集：7 个 v_*（_VARIANT_SPECS）+ 8 个 f0_*（F0_DOSAGE_GRID）。
    覆盖 PRUNED_POOL_CORE(10) + ABLATION_MA(3) + savgol 剂量诊断(2)，由 tests 守。

    Step 1.1-②：构建时把 minimal_l2 的 operator_defaults **完整 resolve** 进每步 params
    （defaults ⊕ override，与 compose 模板路径同一合成语义）→ SHA 绑定执行语义；
    原始 override 保留在 per-action provenance 供审计。"""
    from ..run_main_table import _VARIANT_SPECS
    from ..family0_actions import F0_DOSAGE_GRID
    from ..harness.layers import minimal_l2
    defaults = minimal_l2().operator_defaults
    defaults_sha = hashlib.sha256(
        json.dumps(defaults, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def _resolved(op: str, override: Dict[str, Any]) -> Dict[str, Any]:
        return {**defaults.get(op, {}), **override}

    specs: List[ActionSpec] = []
    for name, chain in _VARIANT_SPECS.items():
        specs.append(ActionSpec(
            action_id=name,
            steps=tuple(ActionStep(op, _resolved(op, {})) for op in chain),
            task_constraints=_task_constraints(list(chain)),
            model_constraints=None,
            provenance={"source": "run_main_table._VARIANT_SPECS", "menu_version": "v1",
                        "override_params": [{} for _ in chain]}))
    for name, op, w in F0_DOSAGE_GRID:
        override = {"window": int(w)}
        specs.append(ActionSpec(
            action_id=name,
            steps=(ActionStep("impute_linear", _resolved("impute_linear", {})),
                   ActionStep(op, _resolved(op, override))),
            task_constraints=_task_constraints(["impute_linear", op]),
            model_constraints=None,
            provenance={"source": "family0_actions.F0_DOSAGE_GRID", "menu_version": "v1",
                        "override_params": [{}, override]}))
    return ActionMenu("v1", specs, meta={
        "params_resolution": "resolved_full（defaults ⊕ override，构建时固化）",
        "operator_defaults_sha": defaults_sha,
        "defaults_source": "harness.layers.minimal_l2().operator_defaults"})


# ════════════════════════ Menu v2（E-3.3 R5，2026-07-14） ════════════════════════
#
# 新算子进了 registry **并不等于它可用**——程序生成面才是动作真正被枚举的地方。本项目有
# 三个互不相同的枚举面，别混：
#   ① `policy.action_spec.action_menu_*`  —— Router/selector 的版本化动作集（**本节扩的是它**）；
#   ② `p6.fast_path.GRAMMAR_* / det_ladder` —— H_ref 的候选文法，**硬编码、已冻结、绝不能碰**
#      （v0.2 竞技场的每一个数字都是在这个文法下量出来的；动它 = 全部读数作废）；
#   ③ `fast_path.compose._IMPUTE/_DENOISE/_OUTLIER` —— heuristic 合成的同类降级链（旧路径，
#      本次不动：新算子经 usable_ops 已对 LLM/selector 可见，不需要挤进 heuristic 的优先级表）。
#
# **v1 必须逐位不变**（它是 P0 冻结面）。这有两条硬约束：
#   (a) 新动作只能进 v2，v1 的 actions 一个字节都不许改；
#   (b) **不得往 `harness.layers.minimal_l2().operator_defaults` 里加新算子的默认参数**——
#       该 dict 的 sha 是 menu v1 的 meta 字段，加一个键 v1 的 SHA 就变了；而且 P6 的
#       `resolve_steps` 也读它，改它会静默改掉 H_ref 的参数解析。
#   → 所以 v2 的新动作**把每个参数完整写死在 ActionSpec 里**（不依赖任何 defaults 查表）。
#       这同时也是 v1 就立下的规矩：menu SHA 必须绑定动作的**执行语义**，而不只是算子名。
#
# TTHA-0 的 prereg 将 pin 的就是 `action_menu_v2().sha256`：动作面一旦签发即冻结，中途加算子
# = 协议漂移。所以 R1–R3 必须在 prereg 签发**之前**全部入列——这就是它们排在最前面的原因。

_MENU_V2_ADDITIONS: List[Tuple[str, Tuple[Tuple[str, Dict[str, Any]], ...]]] = [
    # 模型预测族插补（机制上区别于 v1 里清一色的复制/插值族）
    # period=0 / order=0 = "自动"（走 A0 共享周期模块）——菜单里 pin 的是**这个策略**，
    # 而不是某个具体周期：动作要在全 roster 上都成立，roster 里 hourly/daily 频率并存。
    ("v_ssm", (("impute_ssm", {"period": 0}),)),
    ("v_ar", (("impute_ar", {"order": 0, "period": 0}),)),
    # 局部自适应点式离群修复（区别于 v_winsor 的全局阈值裁剪）
    ("v_hampel", (("impute_linear", {}),
                  ("hampel_filter", {"window": 7, "n_sigmas": 3.0}))),
    # 结构断层修复（填 benchmark 预先声明的 structural_break 缺口——v1 动作面在这个机制上是空集）
    ("v_levelshift", (("impute_linear", {}),
                      ("repair_level_shift", {"period": 0, "t_threshold": 3.5,
                                              "min_jump_sigma": 1.0, "min_segment": 10,
                                              "max_breaks": 5}))),
]


def action_menu_v2() -> ActionMenu:
    """Menu v2 = menu v1 全集（逐位不变）+ 4 个新机制动作（E-3.3 R1–R3 的算子）。

    v2 ⊃ v1：任何 pin 了 v1 动作 ID 的旧 artifact 在 v2 下仍能解析到**同一个** ActionSpec
    （v1 的 spec 直接取自 `action_menu_v1()`，不重新构造），故升级菜单不使旧 Router 失效。
    新动作的 params 全部显式写死（见上方注释 (b)），不查 operator_defaults。
    """
    v1 = action_menu_v1()
    specs: List[ActionSpec] = [v1.actions[aid] for aid in sorted(v1.actions)]
    for name, chain in _MENU_V2_ADDITIONS:
        if name in v1:
            raise ValueError(f"menu v2 新动作 {name!r} 与冻结的 v1 动作 ID 冲突"
                             f"（同名不同义 = provenance 污染，拒绝）")
        specs.append(ActionSpec(
            action_id=name,
            steps=tuple(ActionStep(op, dict(params)) for op, params in chain),
            task_constraints=_task_constraints([op for op, _ in chain]),
            model_constraints=None,
            provenance={"source": "policy.action_spec._MENU_V2_ADDITIONS",
                        "menu_version": "v2",
                        "override_params": [dict(params) for _, params in chain]}))
    return ActionMenu("v2", specs, meta={
        **v1.meta,
        "extends": "v1",
        "extends_sha256": v1.sha256,
        "params_resolution": (
            "v1 动作：resolved_full（沿用 v1 构建结果，逐位不变）；"
            "v2 新动作：explicit_full（参数完整写死在 spec 里，不查 operator_defaults）"),
        "new_mechanisms": ["model_predictive_imputation", "local_adaptive_outlier_repair",
                           "structural_break_repair"],
        "frozen_surfaces_untouched": ["p6.fast_path.GRAMMAR_*", "p6.fast_path.det_ladder",
                                      "harness.layers.minimal_l2().operator_defaults"]})


class ActionCompiler:
    """ActionSpec + conditioning context → 可执行对象。

    两级编译，均复用既有承重路径（不引入平行实现）：
      to_harness  与 fixed_harness_variants / dosage_variant 同构：minimal harness +
                  单条全局 PipelineTemplate（stages 逐 step，params_override=step.params
                  ——②后 params 已是完整 resolved，override 优先级最高 → 跨版本重放稳定）；
      to_program  在该 harness 上走 fast_path.compose 的模板路径 → D6 契约过滤与
                  实验/部署完全同一份代码。

    Step 1.1-①（fail-loud）：违反 task_constraints 的编译**直接拒绝**，不再依赖 D6 静默
    滤步——否则 "Router 选 v_median、anomaly 下实际执行≈v_none"：action ID 对、语义漂了。
    """

    @staticmethod
    def _check_task(spec: ActionSpec, task: str):
        if spec.task_constraints and task not in spec.task_constraints:
            raise ValueError(
                f"动作 {spec.action_id!r} 不支持 task={task!r}"
                f"（task_constraints={spec.task_constraints}）——编译拒绝（fail-loud）；"
                "静默降级会造成 action ID 正确而执行语义不同")

    def to_harness(self, spec: ActionSpec, task: str) -> HarnessState:
        self._check_task(spec, task)
        h = HarnessState.from_minimal()
        if not spec.steps:
            # strict raw（空步 spec）：不注册模板——stages=[] 的模板永不被 _best_template 匹配，
            # 注册它是"看似生效实则惰性"的死 artifact；恒等语义由 to_program 空步分支保证
            return h
        stages = [{"stage": "s1", "preferred_ops": [st.op], "banned_ops": [],
                   "params_override": dict(st.params)} for st in spec.steps]
        h.l2.task_templates[spec.action_id] = PipelineTemplate.from_dict(
            {"name": spec.action_id,
             "applies_to": {"task_type": task, "pattern_conditions": None},
             "stages": stages})
        return h

    def to_program(self, spec: ActionSpec, conditioning_key: Dict[str, Any],
                   harness: Optional[HarnessState] = None) -> Program:
        task = conditioning_key["task"]["type"]
        self._check_task(spec, task)
        if not spec.steps:
            # strict raw（P0 语义三拆）：空 steps = 恒等程序。不得走 compose——空模板不会
            # 匹配 _best_template，会静默落入 heuristic 合成插补链 = 语义漂移。
            return Program(steps=[], source="template", note=f"raw_identity:{spec.action_id}")
        h = harness if harness is not None else self.to_harness(spec, task)
        return compose(conditioning_key, h)
