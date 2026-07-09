"""fast_path/compose.py — 程序合成（Phase 0 heuristic，无 LLM）。

输出 Program = 有序 (op, params) 步骤 + source。heuristic 按 (task, quality_profile) 决定
流水线——直接体现 C1：同一退化输入，forecast / anomaly / classification 产出不同程序：
  - forecast：插补 → (去离群) → 去噪；**绝不 standardize**（Part II：标准化是 forecast 灾难主因）
  - anomaly_detection：仅温和插补；**禁平滑、禁删离群**（保 spike/changepoint）
  - classification：插补（+可选 znorm）；不平滑（CNN 自去噪）
合成只用「可用」算子（active 且未被该 task 模板 banned）；不可用则降级到同类备选。
Phase 0+ 接 LLM compose 时，只需替换 compose() 内部，Program 契约不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import thresholds as TH
from ..conditioning.distance import d_struct

# 同类算子优先级（前者不可用则退到后者）。S0.7-6：全部 canonical 名
# （fill_gaps≡impute_linear 去重；impute_kalman→impute_ema、kalman_filter→smooth_ema 正名）
_IMPUTE = ["impute_linear", "impute_fft", "impute_ema", "period_complete"]
_DENOISE = ["denoise_savgol", "denoise_median", "denoise_stl", "fft_decompose", "smooth_ema"]
_OUTLIER = ["winsorize", "outlier_iqr", "outlier_mad"]


@dataclass
class ProgramStep:
    op: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Program:
    steps: List[ProgramStep] = field(default_factory=list)
    source: str = "template"            # "template"(heuristic) | "llm_custom"
    note: str = ""

    def as_pairs(self) -> List[Tuple[str, dict]]:
        return [(s.op, s.params) for s in self.steps]

    def op_names(self) -> List[str]:
        return [s.op for s in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        return {"steps": [{"op": s.op, "params": s.params} for s in self.steps],
                "source": self.source, "note": self.note}


def _template_matches(tmpl, task: str, pattern_bin: str,
                      struct_feats: Optional[Dict[str, float]] = None) -> bool:
    """task_type 必须相等；pattern_conditions 为空 → 全局模板。否则定 cell-scope 复用资格：

    **软结构门（方向 A，AME-TS 软先验）**：模板携带创建时的 `struct_ref`（10 维结构特征）且
    当前 struct_feats 可得 → 仅当 `d_struct(struct_ref, struct_feats) < τ` 才复用——跨/同 bin
    一致按全 10 维结构距离裁，消解粗 bin(SNR×missing) 把结构迥异 cell 误并的负迁移。
    回退：模板无 struct_ref（旧版/未标注）或当前无特征 → 旧式 `pattern_bin==` 精确匹配（向后兼容）。
    """
    if tmpl.applies_to.get("advisory"):              # S0.4：迁移重验降级的模板 → 不再强制生效（不删）
        return False
    if tmpl.applies_to.get("task_type") != task:
        return False
    pc = tmpl.applies_to.get("pattern_conditions") or {}
    if not pc:
        return True
    ref = pc.get("struct_ref")
    if ref and struct_feats:
        return d_struct(ref, struct_feats) < TH.BIN_DSTRUCT_TAU
    return pc.get("pattern_bin") == pattern_bin


def matching_templates(harness, task: str, pattern_bin: str,
                       struct_feats: Optional[Dict[str, float]] = None):
    return [t for t in harness.l2.task_templates.values()
            if _template_matches(t, task, pattern_bin, struct_feats)]


def cell_banned_ops(harness, task: str, pattern_bin: str,
                    struct_feats: Optional[Dict[str, float]] = None) -> set:
    """该 (task, cell) 下被 banned 的算子——只数匹配该 cell 的模板，故 ban 天然是 cell-scoped。
    S0.7-6：按 canonical 名归一（旧模板 ban `fill_gaps` 等旧 ID 时对 canonical 同样生效）。"""
    from ..operators.registry import canonicalize
    banned: set = set()
    for t in matching_templates(harness, task, pattern_bin, struct_feats):
        for st in t.stages:
            banned.update(canonicalize(b) for b in st.banned_ops)
    return banned


def _best_template(harness, task: str, pattern_bin: str,
                   struct_feats: Optional[Dict[str, float]] = None):
    """匹配该 cell 且有 stages 的模板，pattern-conditioned 优先于全局。"""
    cands = [t for t in matching_templates(harness, task, pattern_bin, struct_feats) if t.stages]
    if not cands:
        return None
    scoped = [t for t in cands if t.applies_to.get("pattern_conditions")]
    return (scoped or cands)[0]


_CONTRACT_TASKS = ("forecast", "classification", "anomaly_detection")


def is_operator_eligible(op: str, task: Optional[str], harness, banned) -> bool:
    """D6 统一算子资格判定（模板/heuristic/LLM/recovery/run_gates 五路共用，防再漂移）。

    资格 = canonical 化后 active ∧ 未被该 cell ban ∧ **task 契约**（registry allowed_tasks——
    anomaly 物理禁 smoothing/destructive）。task=None 或未知 task → 不做契约过滤（旧行为）。"""
    from ..operators.registry import OPERATOR_METADATA, canonicalize
    cname = canonicalize(op)
    if not harness.l2.active_operators.get(cname, False) or cname in banned:
        return False
    if task in _CONTRACT_TASKS:
        allowed = OPERATOR_METADATA.get(cname, {}).get("allowed_tasks")
        if allowed and task not in allowed:
            return False
    return True


def _first_usable(candidates, harness, banned, task: Optional[str] = None) -> str:
    """S0.7-6：候选名先 canonical 化再查资格——旧模板的 preferred_ops 用旧 ID 也能重放。
    D6：传 task 时同步执行契约过滤（compose/recovery 一律传；不传=旧行为，仅兼容旧测试）。"""
    from ..operators.registry import canonicalize
    for name in candidates:
        cname = canonicalize(name)
        if is_operator_eligible(cname, task, harness, banned):
            return cname
    return ""


def _key_struct_feats(conditioning_key: Dict[str, Any]) -> Optional[Dict[str, float]]:
    return (conditioning_key.get("pattern") or {}).get("struct_feats")


def compose(conditioning_key: Dict[str, Any], harness) -> Program:
    task = conditioning_key["task"]["type"]
    pattern_bin = conditioning_key.get("pattern_bin", "")
    feats = _key_struct_feats(conditioning_key)
    pt = conditioning_key["pattern"]["quality_profile"]["problem_types"]
    banned = cell_banned_ops(harness, task, pattern_bin, feats)

    # 1) 匹配该 cell 的模板（pattern-conditioned 优先）→ 用其 stages 驱动流水线（C1 specialization 的落点）
    tmpl = _best_template(harness, task, pattern_bin, feats)
    if tmpl is not None:
        tsteps: List[ProgramStep] = []
        for st in tmpl.stages:
            op = _first_usable(st.preferred_ops, harness, banned, task)
            if op:
                params = dict(harness.l2.operator_defaults.get(op, {}))
                params.update(st.params_override or {})
                tsteps.append(ProgramStep(op, params))
        if tsteps:
            return Program(steps=tsteps, source="template", note=f"tmpl:{tmpl.name}")

    # 2) 否则 heuristic（banned 已是 cell-scoped）
    steps: List[ProgramStep] = []

    def add(cands):
        op = _first_usable(cands, harness, banned, task)
        if op:
            steps.append(ProgramStep(op, dict(harness.l2.operator_defaults.get(op, {}))))
        return op

    if task == "forecast":
        if pt["has_missing"]:
            add(_IMPUTE)
        if pt["has_outlier"]:
            add(_OUTLIER)
        if pt["has_noise"]:
            add(_DENOISE)
        # 不加任何 standardize/znorm（forecast 保尺度）
    elif task == "anomaly_detection":
        if pt["has_missing"]:
            add(_IMPUTE)              # 仅温和插补；不平滑、不删离群（保异常信号）
    elif task == "classification":
        if pt["has_missing"]:
            add(_IMPUTE)
        if _first_usable(["znorm"], harness, banned, task):
            steps.append(ProgramStep("znorm", {}))
    else:
        if pt["has_missing"]:
            add(_IMPUTE)

    return Program(steps=steps, source="template",
                   note=f"heuristic:{task}" if steps else f"heuristic:{task}:identity")


def usable_ops(harness, task: str, pattern_bin: str = "",
               struct_feats: Optional[Dict[str, float]] = None) -> List[str]:
    """active 且未被该 cell 模板 banned 且非形状改变（Phase 0 1D pipeline）的算子。

    S0.7-6：另加 **task 级物理过滤**（registry 契约 allowed_tasks）——anomaly 物理禁
    smoothing/destructive（保 spike/changepoint），不再只靠模板 ban；alias 条目不重复出现
    （canonical 已在列表中）。未知 task 不做契约过滤（保持旧行为）。"""
    from ..operators.registry import OPERATOR_METADATA
    banned = cell_banned_ops(harness, task, pattern_bin, struct_feats)
    out: List[str] = []
    for op, on in harness.l2.active_operators.items():
        meta = OPERATOR_METADATA.get(op, {})
        if not on or meta.get("shape_changing", False) or meta.get("is_alias", False):
            continue
        if not is_operator_eligible(op, task, harness, banned):     # D6 统一资格判定
            continue
        out.append(op)
    return out


_LLM_SYS = (
    "You compose a time-series data-cleaning pipeline for a downstream task. "
    "You DO NOT see the raw series, only its structural summary. Output STRICT JSON only: "
    '{"steps":[{"op":"<name>","params":{}}]}. Use ONLY operators from the allowed list.'
)


def compose_llm(conditioning_key: Dict[str, Any], harness, prior_fragments, failure_warnings, llm) -> Program:
    """LLM 程序合成（pipeline-spec 模式）：见结构摘要 + L1 task_prompt + 可用算子 + 暖启动片段 + 失败警告，
    不见原序列。产出步骤经 usable 过滤；空/非法 → 回退 heuristic compose。仅用于部署路径（非验证环路）。"""
    task = conditioning_key["task"]["type"]
    usable = usable_ops(harness, task, conditioning_key.get("pattern_bin", ""),
                        _key_struct_feats(conditioning_key))
    if not usable or llm is None:
        return compose(conditioning_key, harness)
    feats = conditioning_key["pattern"]["struct_feats"]
    pt = conditioning_key["pattern"]["quality_profile"]["problem_types"]
    prompt = (
        f"task={task}\nL1 instruction: {harness.l1.task_prompts.get(task, '')}\n"
        f"sensitivity: {harness.l1.task_sensitivity.get(task, {})}\n"
        f"structural summary: SNR={feats.get('SNR'):.1f}dB period={feats.get('period'):.0f} "
        f"trend={feats.get('trend_strength'):.2f} problems={pt}\n"
        f"allowed operators (ordered S1→S3): {usable}\n"
        f"prior successful pipelines (warm-start): "
        f"{[ [s['op'] for s in f['program'].get('steps', [])] for f in (prior_fragments or [])][:3]}\n"
        f"failure warnings to avoid: {[w['signature'] for w in (failure_warnings or [])][:3]}\n"
        f"Compose a minimal ordered pipeline. Output JSON only."
    )
    from ..llm import extract_json
    try:
        spec = extract_json(llm(_LLM_SYS, prompt, nonce=0))
    except Exception:
        spec = None
    steps_spec = (spec.get("steps", []) if isinstance(spec, dict)
                  else spec if isinstance(spec, list) else [])
    steps: List[ProgramStep] = []
    for st in steps_spec:
        op = st.get("op") if isinstance(st, dict) else (st if isinstance(st, str) else None)
        if op in usable:
            steps.append(ProgramStep(op, (st.get("params", {}) if isinstance(st, dict) else {}) or {}))
    if not steps:
        return compose(conditioning_key, harness)        # 回退
    return Program(steps=steps, source="llm_custom", note=f"llm:{task}")


def compose_recovery(conditioning_key: Dict[str, Any], harness, failure_signature: str) -> Program:
    """安全恢复程序：温和插补（杀 NaN）+ winsorize（压爆炸/超范围），只用可用算子。

    D6 修复：recovery 同样过 task 契约——anomaly 下 winsorize/outlier_*（destructive，毁 spike）
    被物理过滤，恢复退化为仅温和插补（与 heuristic anomaly 路径一致），不再绕过 allowed_tasks。"""
    task = conditioning_key["task"]["type"]
    banned = cell_banned_ops(harness, task, conditioning_key.get("pattern_bin", ""),
                             _key_struct_feats(conditioning_key))
    steps: List[ProgramStep] = []
    imp = _first_usable(_IMPUTE, harness, banned, task)
    if imp:
        steps.append(ProgramStep(imp, {}))
    wz = _first_usable(["winsorize", "outlier_mad", "outlier_iqr"], harness, banned, task)
    if wz:
        steps.append(ProgramStep(wz, {}))
    return Program(steps=steps, source="template", note=f"recovery<-{failure_signature}")
