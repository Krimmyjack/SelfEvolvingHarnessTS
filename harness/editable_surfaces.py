"""harness/editable_surfaces.py — 可编辑面声明 + 机械校验（plan.md §3.2(c)/(e)）。

EDITABLE_SURFACES 声明每个可改字段的 {type(value 期望类型), writer, scope, addressing,
protected, ref_domain, name_field, scope_resolver}。validate(patch, harness) 机械执行
§3.2(e) 校验链；parse_path 把 path 解析成 (layer, field, selector)，供 validate 与
state.apply_edit 共用。

注意 `type` = patch.value 的**期望类型**（不是字段容器类型）：
  list_scalar → 元素类型；leaf-into-dict → 值类型；named_object → 元素类名(str)/类；
  leaf-into-dataclass → None（从现有属性类型推断）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .edit_patch import EditPatch
from .layers import (
    PipelineTemplate, MiddlewareDef, EvaluatorSpec, StrengthSignatureStats, TASK_TYPES,
)

# named_object 的类名字符串 → 实际类（isinstance 校验用）
CLASS_REGISTRY = {
    "PipelineTemplate": PipelineTemplate,
    "MiddlewareDef": MiddlewareDef,
    "EvaluatorSpec": EvaluatorSpec,
    "StrengthSignatureStats": StrengthSignatureStats,
}

ALLOWED_OPS = {
    "leaf": {"set", "remove"},          # set 覆盖键值 / remove 删 dict 键（dataclass 属性不可 remove）
    "list_scalar": {"add", "remove"},
    "named_object": {"set", "add", "remove"},
}


@dataclass
class Surface:
    type: Any                       # value 期望类型：内建类型 / 类名 str / None(从现值推断)
    writer: str                     # "step" | "consolidator"
    scope: str                      # "cell" | "global"（静态默认）
    addressing: str                 # "leaf" | "list_scalar" | "named_object"
    protected: bool = False
    ref_domain: Optional[str] = None        # "OPERATOR_REGISTRY" | "TASK_TYPES"
    value_range: Optional[Tuple] = None      # 数值 leaf 的 [lo, hi]
    name_field: str = "name"                 # named_object 在 list-backed 时的标识字段
    scope_resolver: Optional[Any] = None     # Callable[[EditPatch], str] | None


# ── scope_resolver（task_templates 动态：有 pattern_conditions → cell；否则 global）──
def _pattern_conditions(value: Any):
    if value is None:
        return None
    applies = value.get("applies_to") if isinstance(value, dict) else getattr(value, "applies_to", None)
    if not isinstance(applies, dict):
        return None
    return applies.get("pattern_conditions")


def _templates_scope(patch: EditPatch) -> str:
    return "global" if not _pattern_conditions(patch.value) else "cell"


# ════════════════════════════ 可编辑面注册表 ════════════════════════════
EDITABLE_SURFACES = {
    # L1 ──────────────────────────────────────────────
    "l1.system_prompt":       None,                                                  # ❌ 冷启动一次性，只读
    "l1.constraints":         Surface(str, "step", "global", "list_scalar"),
    "l1.recovery_rules":      Surface(dict, "step", "global", "named_object", name_field="trigger"),
    "l1.task_prompts":        Surface(str, "step", "global", "leaf", ref_domain="TASK_TYPES"),
    "l1.task_sensitivity":    Surface(dict, "step", "global", "leaf", ref_domain="TASK_TYPES"),
    # L2 ──────────────────────────────────────────────
    "l2.operator_registry":   None,                                                  # ❌ 只读基础设施
    "l2.active_operators":    Surface(bool, "step", "global", "leaf", ref_domain="OPERATOR_REGISTRY"),
    "l2.operator_defaults":   Surface(dict, "step", "global", "leaf", ref_domain="OPERATOR_REGISTRY"),
    "l2.task_templates":      Surface("PipelineTemplate", "step", "global", "named_object",
                                      ref_domain="OPERATOR_REGISTRY", scope_resolver=_templates_scope),
    "l2.middlewares":         Surface("MiddlewareDef", "step", "global", "named_object",
                                      ref_domain="OPERATOR_REGISTRY"),
    # L3 ──────────────────────────────────────────────
    "l3.retrieval_config":    Surface(None, "step", "global", "leaf"),               # leaf-into-dataclass → 推断类型
    "l3.evidence_store":      None,                                                  # ❌ 后端只读
    "l3.failure_signatures":  None,                                                  # ⚙️ mining 维护，非 EditPatch 面
    "l3.strength_signatures": Surface("StrengthSignatureStats", "consolidator", "global",
                                      "named_object", protected=True),               # 🔒 承重墙
    # L4 ──────────────────────────────────────────────
    "l4.gate_config":         Surface(None, "step", "global", "leaf"),
    "l4.proxy_evaluators":    Surface("EvaluatorSpec", "step", "global", "named_object",
                                      ref_domain="TASK_TYPES"),
    "l4.grounded_evaluators": Surface("EvaluatorSpec", "consolidator", "global", "named_object",
                                      protected=True, ref_domain="TASK_TYPES"),       # 🔒 裁判 step 不可改
    "l4.shrinkage_config":    Surface(None, "step", "global", "leaf"),
}


# ════════════════════════════ path 解析 ════════════════════════════
@dataclass
class ParsedPath:
    layer: str                       # l1 | l2 | l3 | l4
    field: str
    selector_kind: Optional[str]     # None | "key" | "name"
    selector: Optional[str]

    @property
    def surface_key(self) -> str:
        return f"{self.layer}.{self.field}"


def parse_path(path: str) -> ParsedPath:
    """leaf 'lN.field.key' / list_scalar 'lN.field' / named_object 'lN.field::name'。
    禁位置索引深路径（>3 段或 [i] 形式）。"""
    if "[" in path or "]" in path:
        raise ValueError(f"positional index forbidden in path: {path!r}")
    if "::" in path:
        left, _, name = path.partition("::")
        parts = left.split(".")
        if len(parts) != 2 or not name:
            raise ValueError(f"named_object path must be 'lN.field::name': {path!r}")
        return ParsedPath(parts[0], parts[1], "name", name)
    parts = path.split(".")
    if len(parts) == 2:
        return ParsedPath(parts[0], parts[1], None, None)
    if len(parts) == 3:
        return ParsedPath(parts[0], parts[1], "key", parts[2])
    raise ValueError(f"path too deep / malformed (max 3 segments): {path!r}")


# ════════════════════════════ 校验结果 ════════════════════════════
@dataclass
class ValidationResult:
    ok: bool
    reason: str
    resolved_scope: Optional[str] = None     # 通过时给出（决定 held-out(b) 广度）

    def __bool__(self) -> bool:
        return self.ok


def resolve_scope(patch: EditPatch, surface: Surface) -> str:
    if surface.scope_resolver is not None:
        try:
            return surface.scope_resolver(patch)
        except Exception:
            return "global"      # 解析失败 → 保守取 global（多查 Pareto）
    return surface.scope


# ════════════════════════════ 校验链（§3.2(e)）════════════════════════════
def validate(patch: EditPatch, harness) -> ValidationResult:
    # ① path 解析
    try:
        pp = parse_path(patch.path)
    except ValueError as e:
        return ValidationResult(False, f"bad path: {e}")
    if pp.layer != patch.edited_layer.lower():
        return ValidationResult(False, f"edited_layer {patch.edited_layer} ≠ path layer {pp.layer}")
    if pp.layer not in ("l1", "l2", "l3", "l4"):
        return ValidationResult(False, f"unknown layer {pp.layer}")

    # ① path 命中 surface（None / 缺失 → 只读/越界）
    surface = EDITABLE_SURFACES.get(pp.surface_key, "MISSING")
    if surface == "MISSING":
        return ValidationResult(False, f"{pp.surface_key} is not an editable surface (read-only/unknown)")
    if surface is None:
        return ValidationResult(False, f"{pp.surface_key} is read-only infrastructure")

    # ② op × addressing 相容
    if patch.op not in ALLOWED_OPS[surface.addressing]:
        return ValidationResult(False, f"op {patch.op!r} invalid for addressing {surface.addressing!r}")
    bad_sel = _check_selector_shape(surface.addressing, pp)
    if bad_sel:
        return ValidationResult(False, bad_sel)

    # ⑥ writer 权限（先于昂贵检查；protected ⇒ consolidator-only）
    if surface.writer == "consolidator" and patch.writer != "consolidator":
        return ValidationResult(False,
            f"{pp.surface_key} is consolidator-only (protected={surface.protected}); got writer={patch.writer}")

    # 定位容器
    layer_obj = getattr(harness, pp.layer, None)
    if layer_obj is None or not hasattr(layer_obj, pp.field):
        return ValidationResult(False, f"harness has no {pp.surface_key}")
    container = getattr(layer_obj, pp.field)

    # ③ value 类型 + ④ ref 完整性 + range
    ok, reason = _check_value(patch, surface, pp, container, harness)
    if not ok:
        return ValidationResult(False, reason)

    # ⑤ scope 解析
    return ValidationResult(True, "ok", resolve_scope(patch, surface))


def _check_selector_shape(addressing: str, pp: ParsedPath) -> Optional[str]:
    if addressing == "leaf" and pp.selector_kind != "key":
        return "leaf path requires 'lN.field.key'"
    if addressing == "list_scalar" and pp.selector_kind is not None:
        return "list_scalar path must be 'lN.field' (no key/name)"
    if addressing == "named_object" and pp.selector_kind != "name":
        return "named_object path requires 'lN.field::name'"
    return None


def _type_ok(value: Any, expected: type) -> bool:
    if expected is bool:
        return isinstance(value, bool)
    if expected in (int, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, expected)


def _referenced_operators(pp: ParsedPath, value: Any) -> set:
    """收集 patch 引用到的算子名（用于 OPERATOR_REGISTRY 完整性检查）。"""
    if pp.field in ("active_operators", "operator_defaults"):
        return {pp.selector} if pp.selector else set()
    if pp.field == "task_templates":
        ops: set = set()
        stages = value.get("stages") if isinstance(value, dict) else getattr(value, "stages", [])
        for st in (stages or []):
            pref = st.get("preferred_ops") if isinstance(st, dict) else getattr(st, "preferred_ops", [])
            ban = st.get("banned_ops") if isinstance(st, dict) else getattr(st, "banned_ops", [])
            ops.update(pref or [])
            ops.update(ban or [])
        return ops
    if pp.field == "middlewares":
        comp = value.get("composed_of") if isinstance(value, dict) else getattr(value, "composed_of", [])
        return set(comp or [])
    return set()


def _check_value(patch: EditPatch, surface: Surface, pp: ParsedPath, container, harness):
    op, value = patch.op, patch.value

    # leaf-into-dataclass 不支持 remove（属性不可删；应 set 回默认值）
    if op == "remove" and surface.addressing == "leaf" and not isinstance(container, dict):
        return False, "cannot 'remove' a dataclass attribute (use 'set')"

    # ④ 引用完整性（即便 remove 也校验 key 域：保证删的是合法目标）
    if surface.ref_domain == "TASK_TYPES" and pp.selector is not None:
        if pp.selector not in TASK_TYPES:
            return False, f"task_type {pp.selector!r} not in {TASK_TYPES}"
    if surface.ref_domain == "OPERATOR_REGISTRY":
        registry = set(harness.l2.operator_registry.keys())
        referenced = _referenced_operators(pp, value) if op != "remove" else (
            {pp.selector} if pp.field in ("active_operators", "operator_defaults") and pp.selector else set())
        unknown = referenced - registry
        if unknown:
            return False, f"unknown operators (not in registry): {sorted(unknown)}"

    if op == "remove":
        return True, "ok"   # remove 不带新值，类型检查跳过

    # ③ 值类型
    if surface.addressing == "named_object":
        cls = CLASS_REGISTRY.get(surface.type) if isinstance(surface.type, str) else surface.type
        if cls is not None and not isinstance(value, cls):
            return False, f"value must be {getattr(cls, '__name__', cls)}, got {type(value).__name__}"

    elif surface.addressing == "leaf":
        if isinstance(container, dict):
            if surface.type is not None and not _type_ok(value, surface.type):
                return False, f"value type {type(value).__name__} != expected {surface.type.__name__}"
        else:  # leaf-into-dataclass：属性须存在，类型须与现值相容
            if not hasattr(container, pp.selector):
                return False, f"no attribute {pp.selector!r} on {type(container).__name__}"
            cur = getattr(container, pp.selector)
            if cur is not None and not _type_ok(value, type(cur)):
                return False, f"value type {type(value).__name__} != current {type(cur).__name__}"
            if surface.value_range and isinstance(value, (int, float)) and not isinstance(value, bool):
                lo, hi = surface.value_range
                if not (lo <= value <= hi):
                    return False, f"value {value} out of range [{lo}, {hi}]"

    elif surface.addressing == "list_scalar":
        if surface.type is not None and not _type_ok(value, surface.type):
            return False, f"list element type {type(value).__name__} != expected {surface.type.__name__}"

    return True, "ok"
