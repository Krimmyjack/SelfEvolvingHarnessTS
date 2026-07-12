"""policy/code_agent_composer.py — Frozen code agent composer（P1，Final_Plan_CodeAgentFirst §P1）。

code-agent-first 快路径的默认 composer：输入只能是 EvidencePacket（v2），输出只能是携带
ProgramSpec v1 的 TypedCandidate（action_id=None，program 候选身份由 SafetyGate 的 grammar/
guard 检查裁决）。两个后端：

  stub  确定性规则合成（no-API、CI 安全、bit 级可复现）——P1 接线验收与 P5 对照的机械基线；
  llm   缓存优先的真实 LLM（DeepSeek LLMClient；P1 测试一律用假客户端，真实调用推迟到 P3+
        且必须过磁盘缓存与预算帽）。

ITT 纪律：invalid/empty/不可解析输出 → candidate=None（api_calls 照记），上层回退 raw。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .program_edit import ProgramSpecV1, spec_v1_from_dict, spec_v1_to_dict, validate_v1
from .skill_memory_composer import TypedCandidate, _extract_json

_SYSTEM_PROMPT = (
    "You are a frozen time-series data-readiness code agent. Read ONLY the evidence packet. "
    "Emit exactly one JSON object: a ProgramSpec v1 of shape {grammar:'v1', steps:[[op,{params}],...], "
    "scope:[cell_id], task_type, pattern_guard:[[feature,cmp,value],...], risk_budget_beta, fallback}. "
    "Respect allowed_grammar (operators, window_grid as exact ints, max_steps, beta_range, guard_features). "
    "The first step must be an imputer. Consume continuous_evidence and trace_summaries as numeric "
    "evidence, never as binary verdicts. Invalid output counts as a no-op. "
    # P5-A.2 前置②：内嵌合法输出示例（P5-A 实测 ca_plain 180/180 malformed = 无示例格式塌缩）
    'Example of a valid output (adapt operators/params to the evidence, output JSON only, no prose): '
    '{"grammar": "v1", "steps": [["impute_linear", {}], ["denoise_median", {"window": 9}]], '
    '"scope": ["*"], "task_type": "forecast", "pattern_guard": [["seasonal_strength", ">=", 0.3]], '
    '"risk_budget_beta": 0.3, "fallback": "v_impute_linear"}'
)

_REPAIR_SUFFIX = (
    "\n\nYour previous output was invalid ({reason}). "
    "Output ONLY one corrected ProgramSpec v1 JSON object, nothing else."
)


@dataclass(frozen=True)
class ComposeOutcome:
    candidate: Optional[TypedCandidate]
    backend: str
    api_calls: int = 0
    cache_hit: Optional[bool] = None
    invalid_reason: str = ""
    raw_response: str = ""


class CodeAgentComposer:
    """Callable 满足 escalation.Composer 协议；compose() 返回带 ITT 记账的 ComposeOutcome。"""

    def __init__(self, backend: str = "stub",
                 llm: Optional[Callable[..., str]] = None,
                 nonce: int = 0,
                 repair_retries: int = 0):
        if backend not in ("stub", "llm"):
            raise ValueError(f"backend 须 ∈ {{'stub','llm'}}，得到 {backend!r}")
        self.backend = backend
        self.llm = llm
        self.nonce = int(nonce)
        # P5-A.2 前置②：schema 修复重试预算（默认 0 = P5-A 口径；retrial 预注册开 1）。
        # 每次重试都是真实调用、计入 api_calls；重试后仍无效 → ITT no-op。
        self.repair_retries = int(repair_retries)
        self.total_invocations = 0
        self.total_api_calls = 0

    @classmethod
    def with_deepseek(cls, model: str = "flash", temperature: float = 0.2,
                      cache_name: str = "code_agent_p1", **kw: Any) -> "CodeAgentComposer":
        from ..llm.client import LLMClient
        return cls(backend="llm",
                   llm=LLMClient(model=model, temperature=temperature, cache_name=cache_name, **kw))

    def __call__(self, packet: Mapping[str, Any]) -> Optional[TypedCandidate]:
        return self.compose(packet).candidate

    def compose(self, packet: Mapping[str, Any]) -> ComposeOutcome:
        self.total_invocations += 1
        outcome = (self._compose_stub(packet) if self.backend == "stub"
                   else self._compose_llm(packet))
        self.total_api_calls += outcome.api_calls
        return outcome

    # ── stub 后端 ───────────────────────────────────────────────────────────

    def _compose_stub(self, packet: Mapping[str, Any]) -> ComposeOutcome:
        pattern = packet.get("pattern") or {}
        task = packet.get("task") or {}
        task_type = str(task.get("task_type") or task.get("type") or "forecast")
        cell = str(pattern.get("cell") or "unscoped")
        try:
            snr = float(pattern.get("snr", 0.0))
        except (TypeError, ValueError):
            snr = 0.0
        steps = [("impute_linear", ())]
        if task_type == "forecast":
            window = 25 if snr <= -2.0 else (15 if snr <= 2.0 else 9)   # F0：窗随 SNR 单调
            steps.append(("denoise_median", (("window", window),)))
        elif task_type == "classification":
            steps.append(("denoise_median", (("window", 5),)))          # classify C1：轻平滑护形
        # anomaly_detection：仅插补（registry 物理禁平滑/删改）
        spec = ProgramSpecV1(
            steps=tuple(steps), scope=(cell,), task_type=task_type,
            pattern_guard=(), risk_budget_beta=0.3, fallback="v_impute_linear",
        )
        ok, why = validate_v1(spec)
        if not ok:
            return ComposeOutcome(None, "stub", 0, None, f"stub_invalid:{why}")
        return ComposeOutcome(self._candidate(spec, "stub_code_agent_v1"), "stub", 0, None, "")

    # ── llm 后端 ────────────────────────────────────────────────────────────

    def _compose_llm(self, packet: Mapping[str, Any]) -> ComposeOutcome:
        if self.llm is None:
            return ComposeOutcome(None, "llm", 0, None, "no_backend")
        base_user = json.dumps(packet, ensure_ascii=False, sort_keys=True)
        user = base_user
        calls = 0
        last_reason, last_raw = "", ""
        for _attempt in range(1 + self.repair_retries):
            try:
                try:
                    raw = self.llm(_SYSTEM_PROMPT, user, nonce=self.nonce)
                except TypeError:
                    raw = self.llm(_SYSTEM_PROMPT, user)
            except Exception as exc:
                return ComposeOutcome(None, "llm", calls + 1, None, f"llm_error:{type(exc).__name__}")
            calls += 1
            last_raw = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            obj = raw if isinstance(raw, Mapping) else _extract_json(str(raw))
            if obj is None:
                last_reason = "unparseable_output"
            else:
                try:
                    spec = spec_v1_from_dict(obj)
                except ValueError as exc:
                    last_reason = f"malformed_program_spec:{exc}"
                else:
                    ok, why = validate_v1(spec)
                    if ok:
                        return ComposeOutcome(self._candidate(spec, "llm_code_agent_v1"),
                                              "llm", calls, None, "", last_raw[:2000])
                    last_reason = why
            user = base_user + _REPAIR_SUFFIX.format(reason=last_reason[:300])
        return ComposeOutcome(None, "llm", calls, None, last_reason, last_raw[:2000])

    @staticmethod
    def _candidate(spec: ProgramSpecV1, rationale: str) -> TypedCandidate:
        return TypedCandidate(program_spec=spec_v1_to_dict(spec), rationale=rationale)
