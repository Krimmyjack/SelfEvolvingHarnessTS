from dataclasses import replace
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.harness import EditOperation, load_skill_entry
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest, PreparationStatus
from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse, ReplayAgentBackend


H0_ROOT = Path(__file__).resolve().parents[2] / "methods" / "ttha" / "harness" / "h0"


def _stage(stage, payload):
    return AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "stage_result",
            "stage": stage,
            "payload": payload,
        },
        raw_response={"id": f"replay-{stage}"},
    )


def _identity_responses():
    return [
        _stage(
            "inspect",
            {
                "inspected_region_fractions": [[0.0, 1.0]],
                "requested_public_tools": [],
                "uncertainty": "high",
            },
        ),
        _stage("propose", {"candidates": []}),
        _stage(
            "select",
            {
                "chosen_candidate_id": "identity",
                "verification_actions": ["public_evidence_insufficient"],
            },
        ),
    ]


def _request(values=None):
    return PreparationRequest(
        "series-1",
        np.asarray(values if values is not None else [1.0, 2.0, 3.0]),
        forecast_task_spec_v1(horizon=1),
        {},
    )


def test_fast_and_slow_paths_share_the_same_agent_core():
    core = TTHAAgentCore(
        ReplayAgentBackend(_identity_responses()),
        LocalPublicToolGateway(np.arange(8.0), task_kind="forecast"),
    )
    fast = TTHAFastAgent(core)
    slow = TTHASlowAgent(core)
    assert fast.core is core
    assert slow.core is core


def test_fast_path_explicit_identity_maps_to_abstention():
    h0 = compile_snapshot(H0_ROOT)
    core = TTHAAgentCore(
        ReplayAgentBackend(_identity_responses()),
        LocalPublicToolGateway(np.asarray([1.0, 2.0, 3.0]), task_kind="forecast"),
    )
    result, trace = TTHAFastAgent(core).prepare(_request(), h0)
    assert result.status is PreparationStatus.ABSTAINED
    np.testing.assert_array_equal(result.prepared.values, _request().values)
    assert result.program is None
    assert trace.chosen_candidate_id == "identity"


def test_out_of_scope_skill_does_not_change_effective_view():
    h0 = compile_snapshot(H0_ROOT)
    capability_skill = load_skill_entry(
        {
            "schema_version": "skill-entry/1",
            "skill_id": "very_high_peak_only_v1",
            "skill_kind": "capability",
            "revision": 1,
            "body": "Use a local public-evidence repair.",
            "observable_applicability": {
                "feature": "local_robust_z_peak",
                "op": ">=",
                "value": 999.0,
            },
            "allowed_tools": ["hampel_filter"],
            "risk_guards": {},
        }
    )
    public_features = {
        "task_kind": "forecast",
        "local_robust_z_peak": 2.0,
    }
    baseline = resolve_harness_view(h0, public_features)
    edited_snapshot = replace(h0, skills=(*h0.skills, capability_skill))
    edited = resolve_harness_view(edited_snapshot, public_features)
    assert capability_skill.skill_id not in edited.skill_ids
    assert baseline.effective_harness_view_sha == edited.effective_harness_view_sha


def test_local_tool_request_is_executed_then_same_stage_resumes():
    tool_request = AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "tool_request",
            "call_id": "call-1",
            "tool_name": "summarize_series",
            "arguments": {},
        },
        raw_response={"id": "tool-request"},
    )
    inspect_result = _stage(
        "inspect",
        {
            "inspected_region_fractions": [[0.0, 1.0]],
            "requested_public_tools": ["summarize_series"],
            "uncertainty": "low",
        },
    )
    backend = ReplayAgentBackend([tool_request, inspect_result])
    gateway = LocalPublicToolGateway(np.arange(8.0), task_kind="forecast")
    core = TTHAAgentCore(backend, gateway)
    h0 = compile_snapshot(H0_ROOT)
    view = resolve_harness_view(h0, gateway.public_features)
    result = core.run_stage(
        role="fast",
        stage="inspect",
        case_id="case-tool",
        public_input={"features": gateway.public_features},
        harness_view=view,
        output_schema_name="fast_inspect_v1",
        output_schema=core.load_stage_schema("fast_inspect_v1"),
        source_snapshot_sha=h0.runtime_bundle_sha,
    )
    assert result.payload["uncertainty"] == "low"
    assert result.tool_receipts[0].tool_name == "summarize_series"
    assert backend.call_count == 2


def test_fast_path_compiles_selects_and_executes_program():
    values = np.asarray([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    responses = [
        _stage(
            "inspect",
            {
                "inspected_region_fractions": [[0.0, 1.0]],
                "requested_public_tools": [],
                "uncertainty": "low",
            },
        ),
        _stage(
            "propose",
            {
                "candidates": [
                    {
                        "candidate_id": "agent-0",
                        "steps": [{"op": "impute_linear", "params": {}}],
                    }
                ]
            },
        ),
        _stage(
            "select",
            {
                "chosen_candidate_id": "agent-0",
                "verification_actions": ["scope_checked"],
            },
        ),
    ]
    core = TTHAAgentCore(
        ReplayAgentBackend(responses),
        LocalPublicToolGateway(values, task_kind="forecast"),
    )
    result, trace = TTHAFastAgent(core).prepare(_request(values), compile_snapshot(H0_ROOT))
    assert result.status is PreparationStatus.PREPARED
    assert np.isfinite(result.prepared.values).all()
    assert result.program.steps[0].op == "impute_linear"
    assert trace.modified_indices == (1,)
    assert trace.chosen_candidate_id == "agent-0"


def test_malformed_agent_output_is_behavior_failure():
    malformed = AgentResponse(
        transport_ok=True,
        raw_response={"id": "bad-propose"},
        assistant_text="not-json",
        parsed_envelope=None,
        parse_status="INVALID_AGENT_ENVELOPE",
    )
    responses = [_identity_responses()[0], malformed]
    values = np.asarray([1.0, 2.0, 3.0])
    core = TTHAAgentCore(
        ReplayAgentBackend(responses),
        LocalPublicToolGateway(values, task_kind="forecast"),
    )
    result, trace = TTHAFastAgent(core).prepare(_request(values), compile_snapshot(H0_ROOT))
    assert result.status is PreparationStatus.FAILED
    assert trace.compilation_status == "failed"
    assert "AgentProtocolError" in result.receipt.error


def test_slow_path_returns_untrusted_add_skill_manifest():
    h0 = compile_snapshot(H0_ROOT)
    new_skill = {
        "schema_version": "skill-entry/1",
        "skill_id": "local_outlier_repair_v1",
        "skill_kind": "capability",
        "revision": 1,
        "body": "Use a bounded local repair when public evidence supports it.",
        "observable_applicability": {
            "feature": "local_robust_z_peak",
            "op": ">=",
            "value": 5.0,
        },
        "allowed_tools": ["hampel_filter"],
        "risk_guards": {"max_modified_fraction": 0.05},
    }
    manifest_payload = {
        "edit_id": "edit-1",
        "base_harness_sha": h0.harness_content_sha,
        "target_pattern_id": "pattern-1",
        "target_surface_id": "skill_library.entries/local_outlier_repair_v1",
        "operation": "ADD",
        "surface_precondition": {"kind": "ABSENT"},
        "dependency_precondition_shas": {},
        "new_value": new_skill,
        "predicted_agent_behavior_change": ["retrieve_skill:local_outlier_repair_v1"],
        "predicted_data_effect": ["target_gain"],
        "falsification_condition": ["skill_not_retrieved"],
    }
    backend = ReplayAgentBackend(
        [_stage("edit", {"edit_manifest": manifest_payload})]
    )
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(np.arange(8.0), task_kind="forecast"),
    )
    manifest = TTHASlowAgent(core).propose_edit(
        {
            "pattern_id": "pattern-1",
            "observable_signature": {
                "task_kind": "forecast",
                "local_robust_z_peak": 7.0,
            },
        },
        [{"surface_id": "skill_library.entries/{skill_id}", "operation": "ADD"}],
        h0,
    )
    assert manifest.operation is EditOperation.ADD
    assert manifest.new_value["skill_id"] == "local_outlier_repair_v1"
