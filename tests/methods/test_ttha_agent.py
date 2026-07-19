import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from SelfEvolvingHarnessTS.contracts.harness import EditOperation, load_skill_entry
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest, PreparationStatus
from SelfEvolvingHarnessTS.contracts.run_context import RunDependencyBinding
from SelfEvolvingHarnessTS.contracts.task import (
    deployment_constraints_v1,
    forecast_task_context_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (
    AgentProtocolError,
    TTHAAgentCore,
    validate_local_schema,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse, ReplayAgentBackend
from SelfEvolvingHarnessTS.runtime import candidate_verification


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


def _f1_request(values, *, maximum_modified_fraction=0.35):
    task = forecast_task_spec_v1(horizon=1)
    context = forecast_task_context_v1(
        task_spec=task,
        deployment_constraints=deployment_constraints_v1(
            maximum_modified_fraction=maximum_modified_fraction
        ),
    )
    binding = RunDependencyBinding(
        task_context_sha=context.sha(),
        evaluator_adapter_id="forecast-chronos-v1",
        instrument_epoch="probe-instrument/3",
        corpus_epoch="f1-live-slice-v1",
        capability_bundle_sha="1" * 64,
        runtime_sha="2" * 64,
        harness_sha="3" * 64,
        code_commit="4" * 40,
        provider_id="agicto-chat-completions",
        model_id="gpt-5.5",
    )
    return PreparationRequest(
        "series-f1",
        np.asarray(values, dtype=float),
        task,
        {},
        task_context=context,
        run_dependency_binding=binding,
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


def test_slow_optimizer_can_inspect_an_out_of_scope_capability_for_retrieval_repair():
    h0 = compile_snapshot(H0_ROOT)
    capability_skill = load_skill_entry(
        {
            "schema_version": "skill-entry/1",
            "skill_id": "overly_strict_missing_v1",
            "skill_kind": "capability",
            "revision": 1,
            "body": "Use bounded missing-value repair.",
            "observable_applicability": {
                "feature": "missing_fraction",
                "op": ">",
                "value": 0.9,
            },
            "allowed_tools": ["impute_linear"],
            "risk_guards": {},
        }
    )
    snapshot = replace(h0, skills=(*h0.skills, capability_skill))
    public_features = {"task_kind": "forecast", "missing_fraction": 0.1}
    assert capability_skill.skill_id not in resolve_harness_view(
        snapshot, public_features, role="fast"
    ).skill_ids
    assert capability_skill.skill_id in resolve_harness_view(
        snapshot, public_features, role="slow"
    ).skill_ids


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


def test_live_prompt_makes_outer_agent_envelope_unambiguous():
    response = _stage(
        "inspect",
        {
            "inspected_region_fractions": [[0.0, 1.0]],
            "requested_public_tools": [],
            "uncertainty": "low",
        },
    )

    class CapturingBackend:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return response

    backend = CapturingBackend()
    gateway = LocalPublicToolGateway(np.arange(8.0), task_kind="forecast")
    h0 = compile_snapshot(H0_ROOT)
    core = TTHAAgentCore(backend, gateway)
    core.run_stage(
        role="fast",
        stage="inspect",
        case_id="case-envelope-prompt",
        public_input={"features": gateway.public_features},
        harness_view=resolve_harness_view(h0, gateway.public_features),
        output_schema_name="fast_inspect_v1",
        output_schema=core.load_stage_schema("fast_inspect_v1"),
        source_snapshot_sha=h0.runtime_bundle_sha,
    )

    request = backend.requests[0]
    assert "never return the stage payload by itself" in request.messages[0]["content"]
    prompt = json.loads(request.messages[1]["content"])
    contract = prompt["response_contract"]
    assert contract["outer_envelope_required"] is True
    assert contract["bare_stage_payload_forbidden"] is True
    assert contract["exactly_one_json_value_per_response"] is True
    assert contract["tool_request_allowed"] is True
    assert "stop the response immediately" in contract["tool_request_rule"]
    assert '"stage":"inspect"' in contract["stage_result_template"]
    assert prompt["stage_payload_schema_name"] == "fast_inspect_v1"
    assert "output_schema" not in prompt


def test_prompt_does_not_advertise_tool_requests_in_a_tool_free_stage():
    response = AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "no_proposal",
            "stage": "edit",
            "reason_code": "insufficient_public_evidence",
        },
        raw_response={"id": "replay-no-proposal"},
    )

    class CapturingBackend:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return response

    backend = CapturingBackend()
    gateway = LocalPublicToolGateway(np.arange(8.0), task_kind="forecast")
    h0 = compile_snapshot(H0_ROOT)
    core = TTHAAgentCore(backend, gateway)
    core.run_stage(
        role="slow",
        stage="edit",
        case_id="case-tool-free-prompt",
        public_input={"features": gateway.public_features},
        harness_view=resolve_harness_view(h0, gateway.public_features, role="slow"),
        output_schema_name="slow_edit_v1",
        output_schema=core.load_stage_schema("slow_edit_v1"),
        source_snapshot_sha=h0.runtime_bundle_sha,
    )

    prompt = json.loads(backend.requests[0].messages[1]["content"])
    contract = prompt["response_contract"]
    assert prompt["allowed_local_tools"] == []
    assert contract["tool_request_allowed"] is False
    assert "tool_request_template" not in contract


def test_fast_propose_and_select_receive_the_fixed_probe_contracts():
    class CapturingBackend:
        def __init__(self):
            self.requests = []
            self.responses = _identity_responses()

        def complete(self, request):
            self.requests.append(request)
            return self.responses[len(self.requests) - 1]

    fixed_panel = {
        "probe_contracts": {
            "schema_version": "fixed-probe-contracts/1",
            "contracts_sha": "1" * 64,
            "probes": {},
        }
    }
    values = np.arange(8.0)
    backend = CapturingBackend()
    gateway = LocalPublicToolGateway(
        values,
        task_kind="forecast",
        fixed_probe_panel=fixed_panel,
    )
    TTHAFastAgent(TTHAAgentCore(backend, gateway)).prepare(
        _request(values),
        compile_snapshot(H0_ROOT),
        fixed_probe_panel=fixed_panel,
    )
    assert [request.stage for request in backend.requests] == [
        "inspect",
        "propose",
        "select",
    ]
    for request in backend.requests[1:]:
        prompt = json.loads(request.messages[1]["content"])
        assert prompt["public_input"]["fixed_probe_panel"] == fixed_panel


def test_slow_edit_prompt_receives_operator_and_probe_contracts():
    response = AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "no_proposal",
            "stage": "edit",
            "reason_code": "insufficient_public_evidence",
        },
        raw_response={"id": "slow-contract-capture"},
    )

    class CapturingBackend:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return response

    backend = CapturingBackend()
    slow = TTHASlowAgent(
        TTHAAgentCore(
            backend,
            LocalPublicToolGateway(np.arange(8.0), task_kind="forecast"),
        )
    )
    operator_contracts = ({"name": "hampel_filter", "targeting_mode": "intrinsic"},)
    probe_contracts = {
        "schema_version": "fixed-probe-contracts/1",
        "contracts_sha": "2" * 64,
        "probes": {},
    }
    slow.propose_edit(
        {
            "pattern_id": "pattern-contracts",
            "observable_signature": {"task_kind": "forecast"},
        },
        [],
        compile_snapshot(H0_ROOT),
        allowed_operator_contracts=operator_contracts,
        fixed_probe_contracts=probe_contracts,
    )
    prompt = json.loads(backend.requests[0].messages[1]["content"])
    public_input = prompt["public_input"]
    assert public_input["allowed_operator_contracts"] == list(operator_contracts)
    assert public_input["fixed_probe_contracts"] == probe_contracts


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


def test_f1_task_context_and_receipts_reach_select_with_single_execution(monkeypatch):
    values = np.asarray([1.0, np.nan, 3.0, 4.0])
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
                        "candidate_id": "fill-gap",
                        "steps": [{"op": "impute_linear", "params": {}}],
                    }
                ]
            },
        ),
        _stage(
            "select",
            {
                "chosen_candidate_id": "fill-gap",
                "verification_actions": ["runtime_receipt_checked"],
            },
        ),
    ]

    class CapturingBackend:
        def __init__(self):
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return responses[len(self.requests) - 1]

    calls = 0
    original = candidate_verification.run_pipeline

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_verification, "run_pipeline", counted)
    backend = CapturingBackend()
    request = _f1_request(values)
    result, trace = TTHAFastAgent(
        TTHAAgentCore(
            backend,
            LocalPublicToolGateway(values, task_kind="forecast"),
        )
    ).prepare(request, compile_snapshot(H0_ROOT))

    assert result.status is PreparationStatus.PREPARED
    assert calls == 1
    assert {item.task_context_sha for item in backend.requests} == {
        request.task_context.sha()
    }
    assert {item.run_context_sha for item in backend.requests} == {
        request.run_dependency_binding.sha()
    }
    prompts = [json.loads(item.messages[1]["content"]) for item in backend.requests]
    assert all(
        prompt["public_input"]["task_context_sha"] == request.task_context.sha()
        for prompt in prompts
    )
    select_candidate = prompts[-1]["public_input"]["candidates"][1]
    assert "candidates" not in prompts[0]["public_input"]
    assert "candidates" not in prompts[1]["public_input"]
    assert "verification_receipt" not in json.dumps(prompts[0], sort_keys=True)
    assert "verification_receipt" not in json.dumps(prompts[1], sort_keys=True)
    assert select_candidate["verification_receipt"]["execution_ok"] is True
    assert select_candidate["verification_receipt"]["modified_fraction"] == 0.25
    assert trace.task_context_sha == request.task_context.sha()
    assert trace.run_context_sha == request.run_dependency_binding.sha()
    assert set(trace.candidate_receipt_shas) == {"identity", "fill-gap"}


def test_f1_rejected_candidate_is_not_selectable_but_remains_in_trace():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
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
                        "candidate_id": "global-winsorize",
                        "steps": [
                            {"op": "winsorize", "params": {"limits": 0.25}}
                        ],
                    }
                ]
            },
        ),
        _stage(
            "select",
            {
                "chosen_candidate_id": "identity",
                "verification_actions": ["overflow_candidate_rejected"],
            },
        ),
    ]
    request = _f1_request(values, maximum_modified_fraction=0.05)
    backend = ReplayAgentBackend(responses)
    result, trace = TTHAFastAgent(
        TTHAAgentCore(
            backend,
            LocalPublicToolGateway(values, task_kind="forecast"),
        )
    ).prepare(request, compile_snapshot(H0_ROOT))

    assert result.status is PreparationStatus.ABSTAINED
    assert trace.candidate_ids == ("identity",)
    assert trace.rejection_receipts[0]["candidate_id"] == "global-winsorize"
    assert (
        trace.rejection_receipts[0]["rejection_code"]
        == "MODIFICATION_FRACTION_EXCEEDED"
    )


def test_malformed_agent_output_is_behavior_failure():
    malformed = AgentResponse(
        transport_ok=True,
        raw_response={"id": "bad-propose"},
        assistant_text="not-json",
        parsed_envelope=None,
        parse_status="INVALID_AGENT_ENVELOPE",
    )
    responses = [_identity_responses()[0], malformed, malformed]
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
        "observable_applicability": new_skill["observable_applicability"],
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


def test_slow_path_can_return_an_explicit_no_proposal_envelope():
    h0 = compile_snapshot(H0_ROOT)
    backend = ReplayAgentBackend(
        [
            AgentResponse.valid(
                {
                    "schema_version": "agent-envelope/1",
                    "kind": "no_proposal",
                    "stage": "edit",
                    "reason_code": "no_authorized_minimal_edit",
                },
                raw_response={"id": "no-proposal"},
            )
        ]
    )
    slow = TTHASlowAgent(
        TTHAAgentCore(
            backend,
            LocalPublicToolGateway(np.arange(8.0), task_kind="forecast"),
        )
    )
    manifest = slow.propose_edit(
        {
            "pattern_id": "pattern-1",
            "observable_signature": {"task_kind": "forecast"},
        },
        [],
        h0,
    )
    assert manifest is None
    assert slow.last_no_proposal_reason == "no_authorized_minimal_edit"


def test_slow_schema_exposes_closed_recursive_applicability_with_numeric_bins():
    schema = TTHAAgentCore.load_stage_schema("slow_edit_v1")
    h0 = compile_snapshot(H0_ROOT)
    base = {
        "edit_id": "edit-period-observation",
        "base_harness_sha": h0.harness_content_sha,
        "target_pattern_id": "pattern-period",
        "target_surface_id": "bootstrap_skills.entries/inspect_and_localize.body",
        "operation": "PATCH",
        "surface_precondition": {"kind": "SHA", "sha": "0" * 64},
        "dependency_precondition_shas": {},
        "minimal_patch": {"value": "Preserve reliable public period evidence."},
        "observable_applicability": {
            "all": [
                {"feature": "period_change_score", "op": "==", "value": "high"},
                {"feature": "period_evidence_status", "op": "==", "value": "OK"},
            ]
        },
        "predicted_agent_behavior_change": ["identity_retained"],
        "predicted_data_effect": ["better_localization"],
        "falsification_condition": ["predicted_behavior_absent"],
    }
    validate_local_schema({"edit_manifest": base}, schema)

    invalid = dict(base)
    invalid["observable_applicability"] = {"period_change_score": "high"}
    with pytest.raises(AgentProtocolError):
        validate_local_schema({"edit_manifest": invalid}, schema)


def test_fast_propose_schema_forbids_identity_program_and_nonregistry_ops():
    schema = TTHAAgentCore.load_stage_schema("fast_propose_v1")
    valid = {
        "candidates": [
            {
                "candidate_id": "agent-0",
                "steps": [{"op": "impute_linear", "params": {}}],
            }
        ]
    }
    validate_local_schema(valid, schema)
    literal_identity = {
        "candidates": [
            {
                "candidate_id": "identity",
                "steps": [{"op": "impute_linear", "params": {}}],
            }
        ]
    }
    identity_operator = {
        "candidates": [
            {
                "candidate_id": "agent-0",
                "steps": [{"op": "identity", "params": {}}],
            }
        ]
    }
    with pytest.raises(AgentProtocolError):
        validate_local_schema(literal_identity, schema)
    with pytest.raises(AgentProtocolError):
        validate_local_schema(identity_operator, schema)


def test_fast_propose_schema_exposes_canonical_public_parameter_bindings():
    schema = TTHAAgentCore.load_stage_schema("fast_propose_v1")
    canonical = {
        "candidates": [
            {
                "candidate_id": "repair-level",
                "steps": [
                    {
                        "op": "repair_level_shift",
                        "params": {
                            "region_start_fraction": 0.5,
                            "region_end_fraction": 0.75,
                            "estimated_offset": 1.0,
                        },
                    }
                ],
            }
        ]
    }
    validate_local_schema(canonical, schema)
    wrong_feature_names = {
        "candidates": [
            {
                "candidate_id": "repair-level",
                "steps": [
                    {
                        "op": "repair_level_shift",
                        "params": {
                            "estimated_region_start_fraction": 0.5,
                            "estimated_region_end_fraction": 0.75,
                            "estimated_level_offset": 1.0,
                        },
                    }
                ],
            }
        ]
    }
    with pytest.raises(AgentProtocolError):
        validate_local_schema(wrong_feature_names, schema)


def test_fast_propose_schema_exposes_closed_probe_operator_parameters():
    schema = TTHAAgentCore.load_stage_schema("fast_propose_v1")
    valid = {
        "candidates": [
            {
                "candidate_id": "bounded-outlier",
                "steps": [
                    {
                        "op": "hampel_filter",
                        "params": {
                            "window": 7,
                            "n_sigmas": 8.0,
                            "global_z_min": 4.0,
                        },
                    }
                ],
            }
        ]
    }
    validate_local_schema(valid, schema)
    invalid = json.loads(json.dumps(valid))
    invalid["candidates"][0]["steps"][0]["params"]["hidden_mask"] = True
    with pytest.raises(AgentProtocolError):
        validate_local_schema(invalid, schema)


def test_fast_propose_retries_mismatched_public_parameter_values():
    index = np.arange(192, dtype=float)
    values = np.sin(2.0 * np.pi * index / 24.0)
    values[120:168] += 2.0
    gateway = LocalPublicToolGateway(values, task_kind="forecast")
    features = gateway.public_features
    correct_params = {
        "region_start_fraction": features["estimated_region_start_fraction"],
        "region_end_fraction": features["estimated_region_end_fraction"],
        "estimated_offset": features["estimated_level_offset"],
    }
    wrong_params = dict(correct_params)
    wrong_params["estimated_offset"] = float(correct_params["estimated_offset"]) + 0.1
    inspect = _stage(
        "inspect",
        {
            "inspected_region_fractions": [
                [
                    features["estimated_region_start_fraction"],
                    features["estimated_region_end_fraction"],
                ]
            ],
            "requested_public_tools": [],
            "uncertainty": "low",
        },
    )
    proposal = lambda params: _stage(
        "propose",
        {
            "candidates": [
                {
                    "candidate_id": "repair-level",
                    "steps": [{"op": "repair_level_shift", "params": params}],
                }
            ]
        },
    )
    select = _stage(
        "select",
        {
            "chosen_candidate_id": "repair-level",
            "verification_actions": ["public_parameter_binding_checked"],
        },
    )
    backend = ReplayAgentBackend(
        [inspect, proposal(wrong_params), proposal(correct_params), select]
    )
    core = TTHAAgentCore(backend, gateway)
    result, trace = TTHAFastAgent(core).prepare(
        PreparationRequest(
            "series-level",
            values,
            forecast_task_spec_v1(horizon=1),
            {},
        ),
        compile_snapshot(H0_ROOT),
    )
    assert result.status is PreparationStatus.PREPARED
    assert trace.chosen_candidate_id == "repair-level"
    assert backend.call_count == 4


def test_matched_skill_risk_guard_is_enforced_by_runtime_pool():
    index = np.arange(192, dtype=float)
    values = np.sin(2.0 * np.pi * index / 24.0)
    values[120:168] += 2.0
    gateway = LocalPublicToolGateway(values, task_kind="forecast")
    features = gateway.public_features
    h0 = compile_snapshot(H0_ROOT)
    constrained = load_skill_entry(
        {
            "schema_version": "skill-entry/1",
            "skill_id": "narrow-level-only-v1",
            "skill_kind": "capability",
            "revision": 1,
            "body": "Use only a very narrow public level repair.",
            "observable_applicability": {"const": True},
            "allowed_tools": ["repair_level_shift"],
            "risk_guards": {
                "max_modified_fraction": 0.05,
                "preserve_outside_candidate_region": True,
            },
        }
    )
    snapshot = replace(h0, skills=(*h0.skills, constrained))
    params = {
        "region_start_fraction": features["estimated_region_start_fraction"],
        "region_end_fraction": features["estimated_region_end_fraction"],
        "estimated_offset": features["estimated_level_offset"],
    }
    responses = [
        _stage(
            "inspect",
            {
                "inspected_region_fractions": [
                    [
                        features["estimated_region_start_fraction"],
                        features["estimated_region_end_fraction"],
                    ]
                ],
                "requested_public_tools": [],
                "uncertainty": "low",
            },
        ),
        _stage(
            "propose",
            {
                "candidates": [
                    {
                        "candidate_id": "repair-level",
                        "steps": [{"op": "repair_level_shift", "params": params}],
                    }
                ]
            },
        ),
        _stage(
            "select",
            {
                "chosen_candidate_id": "identity",
                "verification_actions": ["skill_scope_guard_applied"],
            },
        ),
    ]
    result, trace = TTHAFastAgent(
        TTHAAgentCore(ReplayAgentBackend(responses), gateway)
    ).prepare(
        PreparationRequest(
            "series-level-guard",
            values,
            forecast_task_spec_v1(horizon=1),
            {},
        ),
        snapshot,
    )
    assert result.status is PreparationStatus.ABSTAINED
    assert trace.candidate_ids == ("identity",)


def test_slow_stage_retries_once_with_schema_error_feedback():
    h0 = compile_snapshot(H0_ROOT)
    common = {
        "edit_id": "edit-retry",
        "base_harness_sha": h0.harness_content_sha,
        "target_pattern_id": "pattern-retry",
        "target_surface_id": "bootstrap_skills.entries/inspect_and_localize.body",
        "operation": "PATCH",
        "surface_precondition": {"kind": "SHA", "sha": "0" * 64},
        "dependency_precondition_shas": {},
        "minimal_patch": {"value": "Inspect narrow public subregions."},
        "predicted_agent_behavior_change": ["localization_iou>=0.30"],
        "predicted_data_effect": ["localization_iou_improves"],
        "falsification_condition": ["predicted_behavior_absent"],
    }
    invalid = dict(common)
    invalid["observable_applicability"] = {"period_change_score": "high"}
    corrected = dict(common)
    corrected["observable_applicability"] = {
        "feature": "period_change_score",
        "op": "==",
        "value": "high",
    }

    class CapturingBackend:
        def __init__(self):
            self.requests = []
            self.responses = [
                _stage("edit", {"edit_manifest": invalid}),
                _stage("edit", {"edit_manifest": corrected}),
            ]

        def complete(self, request):
            self.requests.append(request)
            return self.responses[len(self.requests) - 1]

    backend = CapturingBackend()
    slow = TTHASlowAgent(
        TTHAAgentCore(
            backend,
            LocalPublicToolGateway(np.arange(8.0), task_kind="forecast"),
        )
    )
    manifest = slow.propose_edit(
        {
            "pattern_id": "pattern-retry",
            "observable_signature": {"period_change_score": "high"},
        },
        [{"surface_id": "bootstrap_skills.entries/inspect_and_localize.body"}],
        h0,
    )
    assert manifest is not None
    assert manifest.observable_applicability["value"] == "high"
    assert len(backend.requests) == 2
    retry_feedback = json.loads(backend.requests[1].messages[-1]["content"])
    assert retry_feedback["schema_version"] == "stage-validation-error/2"
    assert retry_feedback["error_code"] == "STAGE_SCHEMA_INVALID"
    assert "observable_applicability" in retry_feedback["public_message"]
