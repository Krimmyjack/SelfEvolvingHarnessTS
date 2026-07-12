"""P1 契约测试：composer-first escalation（code agent 默认上场，SafetyGate 后接，novel program 直编译）。"""
import numpy as np

from SelfEvolvingHarnessTS.policy.action_spec import action_menu_v1
from SelfEvolvingHarnessTS.policy.code_agent_composer import CodeAgentComposer
from SelfEvolvingHarnessTS.policy.escalation import (
    EscalationConfig,
    decide_fast_path,
    execute_fast_path_decision,
)

RECORD = {
    "uid": "u1",
    "cell": "forecast|snrLow|miss",
    "snr": -4.0,
    "miss_rate": 0.2,
    "X_p": [24, 0.2, 0.3, 0.4, 0, 0.6, 0.7, 0.05],
}
MENU = action_menu_v1()


def _decide(composer, **cfg_kw):
    return decide_fast_path(
        RECORD,
        action_menu_meta=MENU.to_dict(),
        composer=composer,
        config=EscalationConfig(composer_first=True, **cfg_kw),
    )


def _prog_dict(**kw):
    base = {
        "grammar": "v1",
        "steps": [["impute_linear", {}], ["denoise_median", {"window": 9}]],
        "scope": ["forecast|snrLow|miss"],
        "task_type": "forecast",
        "pattern_guard": [],
        "risk_budget_beta": 0.3,
        "fallback": "v_raw_identity",
    }
    base.update(kw)
    return base


def test_composer_first_routes_to_code_agent_program():
    d = _decide(CodeAgentComposer(backend="stub"))
    assert d.route == "code_agent"
    assert d.proposal_route == "code_agent"
    assert d.composer_called
    assert d.safety.accepted
    assert d.action_id.startswith("prog1_")
    assert d.program_action is not None
    assert d.program_action.action_id == d.action_id
    assert d.packet["schema"] == "skill_memory_evidence_packet_v2"   # composer_first → v2 输入面


def test_composer_first_program_executes_end_to_end():
    d = _decide(CodeAgentComposer(backend="stub"))
    x = np.sin(np.linspace(0, 2 * np.pi, 64))
    x[10:14] = np.nan
    executed = execute_fast_path_decision(d, RECORD, MENU, x)
    assert executed.compiled.reason == "compiled_program_spec_v1"
    assert executed.status == "executed"
    assert executed.execution_ok
    assert np.all(np.isfinite(executed.artifact))                    # 插补链补上了缺失


def test_composer_invalid_output_falls_back_raw_itt():
    d = _decide(lambda packet: None)
    assert d.composer_called
    assert d.route == "raw_fallback"
    assert "composer_no_candidate" in d.safety.reasons
    assert d.action_id == "v_none"
    assert d.program_action is None


def test_grammar_violation_rejected_by_gate():
    bad = _prog_dict(steps=[["impute_linear", {}], ["denoise_median", {"window": 7}]])  # 窗不在剂量网格
    d = _decide(lambda packet: {"ProgramSpec": bad})
    assert d.route == "raw_fallback"
    assert "program_grammar_rejected" in d.safety.reasons
    assert d.safety.serve_action_id == "v_none"


def test_pattern_guard_unsatisfied_falls_back():
    guarded = _prog_dict(pattern_guard=[["snr", ">", 5.0]])          # record snr=-4 → guard 不满足
    d = _decide(lambda packet: {"ProgramSpec": guarded})
    assert d.route == "raw_fallback"
    assert "pattern_guard_unsatisfied" in d.safety.reasons


def test_malformed_program_spec_rejected():
    d = _decide(lambda packet: {"ProgramSpec": {"grammar": "v1", "scope": []}})
    assert d.route == "raw_fallback"
    assert "invalid_program_spec" in d.safety.reasons


def test_task_forbidden_op_is_live_in_gate():
    # P2：TaskSpec.forbidden_modifications 成为 gate 活语义——packet 携带的任务禁改集
    # 能拦下 grammar 合法但任务实例禁用的程序（stub 为 forecast 产出含 denoise_median 的链）
    from SelfEvolvingHarnessTS.policy.task_spec import forecast_task_spec_v1

    spec = forecast_task_spec_v1(forbidden_modifications=("denoise_median",))
    d = decide_fast_path(
        RECORD,
        action_menu_meta=MENU.to_dict(),
        composer=CodeAgentComposer(backend="stub"),
        config=EscalationConfig(composer_first=True),
        task_spec=spec,
    )
    assert d.route == "raw_fallback"
    assert "task_forbidden_op" in d.safety.reasons
    assert d.program_action is None


def test_legacy_path_unchanged_without_composer_first():
    d = decide_fast_path(RECORD, action_menu_meta=MENU.to_dict(), composer=None)
    assert d.proposal_route in {"deterministic", "raw_fallback"}
    assert d.packet["schema"] == "skill_memory_evidence_packet_v1"   # 默认仍是 v1 输入面
    assert d.program_action is None
