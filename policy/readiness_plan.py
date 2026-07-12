"""policy/readiness_plan.py — ReadinessPlan → deterministic compiler（P5-A.3 前置④，外评路线图）。

对症 P5-A/A.2 的实测瓶颈（L1 合规塌缩 + guard 校准不足）：LLM 只产出**语义意图**
（ReadinessPlan：插补策略 / 是否钳制离群 / 去噪族×剂量 / 可选 guard / β），合规由本编译器
**按构造保证**——任何结构合法的 plan 编译出的 ProgramSpecV1 必过 validate_v1（穷举 property
tests 守）；guard 只保留观测面有背书且在白名单内的谓词，丢弃项与任务契约过滤项全部留痕
（semantic 与 compliance 由此可分离计量，A.3 四指标分解的地基）。

剂量映射（冻结；与 F0 剂量网格/menu 语义一致）：
  median/ma: light→w9  medium→w15  heavy→w25
  savgol/stl/wavelet: 无窗参（resolved defaults），strength 仅记录
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from .program_edit import ProgramSpecV1, _guard_feats, spec_v1_to_dict, validate_v1
from .skill_memory_composer import TypedCandidate, _extract_json

PLAN_SCHEMA_VERSION = "v1"
_IMPUTE_MAP = {"linear": "impute_linear", "period": "period_complete",
               "fft": "impute_fft", "ema": "impute_ema"}
_DENOISE_FAMILY = {"median": "denoise_median", "ma": "smooth_ma",
                   "savgol": "denoise_savgol", "stl": "denoise_stl",
                   "wavelet": "denoise_wavelet", "none": None}
_STRENGTH_WINDOW = {"light": 9, "medium": 15, "heavy": 25}
_WINDOWED = {"denoise_median", "smooth_ma"}
_GUARD_CMPS = ("<", "<=", ">", ">=", "==")

_PLAN_SYSTEM_PROMPT = (
    "You are a frozen time-series data-readiness planner. Read ONLY the evidence packet. "
    "Emit exactly one JSON ReadinessPlan v1: {plan:'v1', task_type, impute:'linear|period|fft|ema', "
    "outlier_clip:true|false, denoise:{family:'median|ma|savgol|stl|wavelet|none', "
    "strength:'light|medium|heavy'}, guard:[[feature,cmp,value],...], risk_budget_beta, rationale}. "
    "A deterministic compiler maps your plan to an executable program and enforces all contracts; "
    "focus on semantic choices grounded in the numeric evidence. Invalid output counts as a no-op. "
    'Example: {"plan": "v1", "task_type": "forecast", "impute": "linear", "outlier_clip": false, '
    '"denoise": {"family": "median", "strength": "light"}, "guard": [["seasonal_strength", ">=", 0.3]], '
    '"risk_budget_beta": 0.3, "rationale": "seasonal series with moderate noise"}'
)

_REPAIR_SUFFIX = (
    "\n\nYour previous output was invalid ({reason}). "
    "Output ONLY one corrected ReadinessPlan v1 JSON object, nothing else."
)


@dataclass(frozen=True)
class ReadinessPlan:
    task_type: str
    impute: str
    outlier_clip: bool
    denoise_family: str
    denoise_strength: str
    guard: Tuple[Tuple[str, str, float], ...]
    risk_budget_beta: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {"plan": PLAN_SCHEMA_VERSION, "task_type": self.task_type,
                "impute": self.impute, "outlier_clip": bool(self.outlier_clip),
                "denoise": {"family": self.denoise_family, "strength": self.denoise_strength},
                "guard": [list(g) for g in self.guard],
                "risk_budget_beta": float(self.risk_budget_beta),
                "rationale": self.rationale}


def plan_from_dict(d: Any) -> ReadinessPlan:
    """严格反序列化（fail-loud ValueError）——planner/LLM 输出的唯一入口。"""
    if not isinstance(d, Mapping):
        raise ValueError(f"ReadinessPlan 须为 JSON object，得到 {type(d).__name__}")
    if d.get("plan", PLAN_SCHEMA_VERSION) != PLAN_SCHEMA_VERSION:
        raise ValueError(f"plan 版本须为 {PLAN_SCHEMA_VERSION!r}，得到 {d.get('plan')!r}")
    impute = str(d.get("impute", "linear"))
    if impute not in _IMPUTE_MAP:
        raise ValueError(f"impute 须 ∈ {sorted(_IMPUTE_MAP)}，得到 {impute!r}")
    den = d.get("denoise") or {}
    if not isinstance(den, Mapping):
        raise ValueError("denoise 须为 object")
    family = str(den.get("family", "none"))
    if family not in _DENOISE_FAMILY:
        raise ValueError(f"denoise.family 须 ∈ {sorted(_DENOISE_FAMILY)}，得到 {family!r}")
    strength = str(den.get("strength", "light"))
    if strength not in _STRENGTH_WINDOW:
        raise ValueError(f"denoise.strength 须 ∈ {sorted(_STRENGTH_WINDOW)}，得到 {strength!r}")
    guard_raw = d.get("guard") or []
    if not isinstance(guard_raw, (list, tuple)):
        raise ValueError("guard 须为列表")
    guard: List[Tuple[str, str, float]] = []
    for g in guard_raw:
        if not isinstance(g, (list, tuple)) or len(g) != 3 or str(g[1]) not in _GUARD_CMPS \
                or isinstance(g[2], bool) or not isinstance(g[2], (int, float)):
            raise ValueError(f"guard 条目须为 [feature, cmp, number]，得到 {g!r}")
        guard.append((str(g[0]), str(g[1]), float(g[2])))
    beta = d.get("risk_budget_beta", 0.3)
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError(f"risk_budget_beta 须为数值，得到 {beta!r}")
    return ReadinessPlan(
        task_type=str(d.get("task_type", "forecast")),
        impute=impute, outlier_clip=bool(d.get("outlier_clip", False)),
        denoise_family=family, denoise_strength=strength,
        guard=tuple(guard),
        risk_budget_beta=min(1.0, max(0.0, float(beta))),
        rationale=str(d.get("rationale", ""))[:400],
    )


def compile_plan(plan: ReadinessPlan, fingerprint: Mapping[str, Any]
                 ) -> Tuple[ProgramSpecV1, Dict[str, Any]]:
    """plan + 观测指纹 → (ProgramSpecV1, 编译台账)。合规按构造保证；语义损失全部留痕。"""
    from ..operators.registry import OPERATOR_METADATA, canonicalize

    task = plan.task_type if plan.task_type in ("forecast", "classification", "anomaly_detection") \
        else "forecast"
    info: Dict[str, Any] = {"dropped_steps": [], "dropped_guards": [], "task_type": task}

    def _task_ok(op: str) -> bool:
        meta = OPERATOR_METADATA.get(canonicalize(op))
        return meta is not None and task in meta.get("allowed_tasks", ())

    steps: List[Tuple[str, Tuple[Tuple[str, Any], ...]]] = [(_IMPUTE_MAP[plan.impute], ())]
    if plan.outlier_clip:
        if _task_ok("winsorize"):
            steps.append(("winsorize", ()))
        else:
            info["dropped_steps"].append("winsorize")               # 任务契约过滤，留痕
    den_op = _DENOISE_FAMILY[plan.denoise_family]
    if den_op is not None:
        if _task_ok(den_op):
            params = ((("window", _STRENGTH_WINDOW[plan.denoise_strength]),)
                      if den_op in _WINDOWED else ())
            steps.append((den_op, params))
        else:
            info["dropped_steps"].append(den_op)
    if len(steps) > 3:                                               # impute+winsor+denoise=3 上限
        dropped = steps.pop(1)
        info["dropped_steps"].append(dropped[0])

    provided = set((fingerprint.get("struct_feats") or {}).keys()) | \
        {k for k in ("snr", "missing_rate") if fingerprint.get(k) is not None}
    whitelist = _guard_feats()
    kept_guard: List[Tuple[str, str, float]] = []
    for g in plan.guard:
        if g[0] in whitelist and g[0] in provided:
            kept_guard.append(g)
        else:
            info["dropped_guards"].append(list(g))                  # 无背书/越白名单 → 丢弃留痕

    spec = ProgramSpecV1(
        steps=tuple(steps), scope=("*",), task_type=task,
        pattern_guard=tuple(kept_guard),
        risk_budget_beta=plan.risk_budget_beta,
        fallback="v_impute_linear",
        provenance={"source": "readiness_plan_compiler", "plan": plan.to_dict()},
    )
    ok, why = validate_v1(spec)
    if not ok:                                                       # 构造性保证的兜底（不应触发）
        raise AssertionError(f"compiler 违反构造性合规保证：{why}")
    return spec, info


@dataclass(frozen=True)
class PlanComposeOutcome:
    candidate: Optional[TypedCandidate]
    plan: Optional[ReadinessPlan]
    compile_info: Optional[Dict[str, Any]]
    backend: str
    api_calls: int = 0
    invalid_reason: str = ""
    raw_response: str = ""


class PlanComposer:
    """LLM（或 stub）→ ReadinessPlan → deterministic compiler → TypedCandidate。ITT 同 CA composer。"""

    def __init__(self, backend: str = "stub", llm: Optional[Callable[..., str]] = None,
                 nonce: int = 0, repair_retries: int = 0):
        if backend not in ("stub", "llm"):
            raise ValueError(f"backend 须 ∈ {{'stub','llm'}}，得到 {backend!r}")
        self.backend, self.llm = backend, llm
        self.nonce, self.repair_retries = int(nonce), int(repair_retries)
        self.total_invocations = 0
        self.total_api_calls = 0

    def compose(self, packet: Mapping[str, Any]) -> PlanComposeOutcome:
        self.total_invocations += 1
        fingerprint = packet.get("pattern") or {}
        task_type = str((packet.get("task") or {}).get("task_type")
                        or (packet.get("task") or {}).get("type") or "forecast")
        if self.backend == "stub":
            plan = ReadinessPlan(task_type=task_type, impute="linear", outlier_clip=False,
                                 denoise_family=("median" if task_type == "forecast" else "none"),
                                 denoise_strength="light", guard=(),
                                 risk_budget_beta=0.3, rationale="stub_plan_v1")
            spec, info = compile_plan(plan, fingerprint)
            return PlanComposeOutcome(self._candidate(spec), plan, info, "stub", 0, "")
        if self.llm is None:
            return PlanComposeOutcome(None, None, None, "llm", 0, "no_backend")
        base_user = json.dumps(packet, ensure_ascii=False, sort_keys=True)
        user, calls, last_reason, last_raw = base_user, 0, "", ""
        for _attempt in range(1 + self.repair_retries):
            try:
                try:
                    raw = self.llm(_PLAN_SYSTEM_PROMPT, user, nonce=self.nonce)
                except TypeError:
                    raw = self.llm(_PLAN_SYSTEM_PROMPT, user)
            except Exception as exc:
                self.total_api_calls += calls + 1
                return PlanComposeOutcome(None, None, None, "llm", calls + 1,
                                          f"llm_error:{type(exc).__name__}")
            calls += 1
            last_raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            obj = raw if isinstance(raw, Mapping) else _extract_json(str(raw))
            if obj is None:
                last_reason = "unparseable_output"
            else:
                try:
                    plan = plan_from_dict(obj)
                    spec, info = compile_plan(plan, fingerprint)
                except ValueError as exc:
                    last_reason = f"malformed_plan:{exc}"
                else:
                    self.total_api_calls += calls
                    return PlanComposeOutcome(self._candidate(spec), plan, info, "llm",
                                              calls, "", last_raw[:2000])
            user = base_user + _REPAIR_SUFFIX.format(reason=last_reason[:300])
        self.total_api_calls += calls
        return PlanComposeOutcome(None, None, None, "llm", calls, last_reason, last_raw[:2000])

    @staticmethod
    def _candidate(spec: ProgramSpecV1) -> TypedCandidate:
        return TypedCandidate(program_spec=spec_v1_to_dict(spec), rationale="readiness_plan_compiler")
