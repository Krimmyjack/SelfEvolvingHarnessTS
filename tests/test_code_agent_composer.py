"""P1 契约测试：CodeAgentComposer（stub + llm 双后端，ITT，typed ProgramSpecV1 输出）。"""
import json

import pytest

from SelfEvolvingHarnessTS.policy.code_agent_composer import CodeAgentComposer
from SelfEvolvingHarnessTS.policy.evidence_packet import build_evidence_packet_v2
from SelfEvolvingHarnessTS.policy.program_edit import (
    ProgramSpecV1,
    spec_v1_from_dict,
    spec_v1_to_dict,
    validate_v1,
)
from SelfEvolvingHarnessTS.policy.skill_memory_composer import TypedCandidate
from SelfEvolvingHarnessTS.policy.skills import SKILLS_V1
from SelfEvolvingHarnessTS.policy.task_spec import anomaly_task_spec_v1

MENU_META = {"version": "v1", "sha256": "abc", "allowed_actions": ["v_none", "v_median"]}


def _packet(task_spec=None):
    record = {
        "uid": "u1",
        "cell": "forecast|snrLow|miss",
        "snr": -4.0,
        "miss_rate": 0.2,
        "X_p": [24, 0.2, 0.3, 0.4, 0, 0.6, 0.7, 0.05],
    }
    return build_evidence_packet_v2(
        record, skills=SKILLS_V1, memory_rows=None,
        action_menu_meta=MENU_META, task_spec=task_spec,
    )


# ── ProgramSpecV1 序列化（composer 的输出载体）─────────────────────────────

def test_spec_v1_dict_roundtrip_preserves_identity():
    spec = ProgramSpecV1(
        steps=(("impute_linear", ()), ("denoise_median", (("window", 9),))),
        scope=("forecast|snrLow|miss",),
        pattern_guard=(("snr", "<", 0.0),),
        risk_budget_beta=0.4,
        fallback="v_impute_linear",
    )
    d = spec_v1_to_dict(spec)
    json.dumps(d, allow_nan=False)
    back = spec_v1_from_dict(d)
    assert back.sha() == spec.sha()
    assert back.steps == spec.steps
    assert back.pattern_guard == spec.pattern_guard


def test_spec_v1_from_dict_fail_loud():
    with pytest.raises(ValueError):
        spec_v1_from_dict({"grammar": "v1", "scope": ["c"]})           # 无 steps
    with pytest.raises(ValueError):
        spec_v1_from_dict({"grammar": "v9", "steps": [["impute_linear", {}]], "scope": ["c"]})
    with pytest.raises(ValueError):
        spec_v1_from_dict("not a mapping")


# ── stub 后端（no-API，CI 安全，确定性）───────────────────────────────────

def test_stub_backend_emits_valid_forecast_program():
    outcome = CodeAgentComposer(backend="stub").compose(_packet())
    assert outcome.backend == "stub"
    assert outcome.api_calls == 0
    assert outcome.candidate is not None
    assert outcome.candidate.action_id is None                        # program 候选，非 menu 选择
    spec = spec_v1_from_dict(dict(outcome.candidate.program_spec))
    ok, why = validate_v1(spec)
    assert ok, why
    assert spec.steps[0][0] == "impute_linear"
    assert spec.task_type == "forecast"


def test_stub_backend_is_task_aware_for_anomaly():
    outcome = CodeAgentComposer(backend="stub").compose(_packet(task_spec=anomaly_task_spec_v1()))
    spec = spec_v1_from_dict(dict(outcome.candidate.program_spec))
    assert spec.task_type == "anomaly_detection"
    ok, why = validate_v1(spec)
    assert ok, why
    assert [op for op, _ in spec.steps] == ["impute_linear"]         # 平滑/删改物理禁

def test_stub_backend_deterministic():
    a = CodeAgentComposer(backend="stub").compose(_packet())
    b = CodeAgentComposer(backend="stub").compose(_packet())
    assert dict(a.candidate.program_spec) == dict(b.candidate.program_spec)


# ── llm 后端（假客户端；真实调用只在 P3+ 且必须走缓存）─────────────────────

def _valid_llm_json():
    return json.dumps({
        "grammar": "v1",
        "steps": [["impute_linear", {}]],
        "scope": ["forecast|snrLow|miss"],
        "task_type": "forecast",
        "pattern_guard": [],
        "risk_budget_beta": 0.2,
        "fallback": "v_raw_identity",
    })


def test_llm_backend_parses_valid_json():
    fake = lambda system, user, nonce=0: _valid_llm_json()
    outcome = CodeAgentComposer(backend="llm", llm=fake).compose(_packet())
    assert outcome.candidate is not None
    assert outcome.api_calls == 1
    assert outcome.invalid_reason == ""


def test_llm_backend_invalid_output_is_itt_noop():
    prose = lambda system, user, nonce=0: "I think you should smooth the series."
    outcome = CodeAgentComposer(backend="llm", llm=prose).compose(_packet())
    assert outcome.candidate is None
    assert outcome.invalid_reason
    assert outcome.api_calls == 1                                     # ITT：调用了但无效=统计在内

    bad = json.dumps({"grammar": "v1", "steps": [["denoise_median", {"window": 9}]],
                      "scope": ["c"], "task_type": "forecast", "pattern_guard": [],
                      "risk_budget_beta": 0.3, "fallback": "v_raw_identity"})
    outcome2 = CodeAgentComposer(backend="llm", llm=lambda s, u, nonce=0: bad).compose(_packet())
    assert outcome2.candidate is None
    assert "imputer" in outcome2.invalid_reason


def test_llm_backend_without_client_is_noop_not_network():
    outcome = CodeAgentComposer(backend="llm", llm=None).compose(_packet())
    assert outcome.candidate is None
    assert outcome.invalid_reason == "no_backend"
    assert outcome.api_calls == 0


def test_composer_conforms_to_escalation_composer_protocol():
    cand = CodeAgentComposer(backend="stub")(_packet())
    assert isinstance(cand, TypedCandidate)
    assert cand.program_spec


# ── P5-A.2 前置②：格式合规面（示例 + 修复重试预算）─────────────────────────

def test_system_prompt_contains_valid_exemplar():
    from SelfEvolvingHarnessTS.policy.code_agent_composer import _SYSTEM_PROMPT
    assert '"grammar"' in _SYSTEM_PROMPT and '"steps"' in _SYSTEM_PROMPT   # few-shot 示例内嵌
    assert "impute_linear" in _SYSTEM_PROMPT


def test_repair_retry_recovers_malformed_output():
    calls = []

    def flaky(system, user, nonce=0):
        calls.append(user)
        if len(calls) == 1:
            return "Sure! Here is a plan in prose, not JSON."
        return _valid_llm_json()

    outcome = CodeAgentComposer(backend="llm", llm=flaky, repair_retries=1).compose(_packet())
    assert outcome.candidate is not None
    assert outcome.api_calls == 2                            # 首次 + 一次修复，全部计入台账
    assert outcome.invalid_reason == ""
    assert "invalid" in calls[1].lower()                     # 修复请求携带失败原因


def test_repair_retry_capped_and_itt_preserved():
    prose = lambda system, user, nonce=0: "still not json"
    outcome = CodeAgentComposer(backend="llm", llm=prose, repair_retries=1).compose(_packet())
    assert outcome.candidate is None                         # 修复后仍无效 → ITT no-op
    assert outcome.api_calls == 2
    assert outcome.invalid_reason


def test_default_no_repair_preserves_p5a_semantics():
    prose = lambda system, user, nonce=0: "not json"
    outcome = CodeAgentComposer(backend="llm", llm=prose).compose(_packet())
    assert outcome.candidate is None
    assert outcome.api_calls == 1                            # repair_retries 默认 0 = P5-A 口径
