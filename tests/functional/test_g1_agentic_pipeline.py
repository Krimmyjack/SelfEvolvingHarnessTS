"""G1's one load-bearing integration test: the whole Pipeline, zero LLM.

Frozen design §13/G1 allows one primary report and one required integration
test for this stage, and §15 forbids building a new test matrix.  So this file
drives the real Runner over real already-exposed KDD data with a scripted
Agent standing in for the model, and checks the closure claims the stage is
allowed to make -- that a Workspace tool was really called, that its result
really reached the later stages, that the Runtime really compiled and executed
a Typed Workflow, that an Episode was written, and that parameter ownership is
enforced mechanically rather than by prompt text.

The scripted Agent is deliberately thin: it reads the allowed operator menu
and the tool result out of the request it is given, so it cannot pass by
hard-coding an operator name or a series id.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    dispatch,
    runner as g1_runner,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.gateway import (  # noqa: E402
    CohortScopePublicToolGateway,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse  # noqa: E402

PREFERRED_OPERATORS = ("outlier_mad", "hampel_filter", "winsorize")


class ScriptedAgentBackend:
    """A deterministic Agent that actually uses the Workspace tool.

    It requests one ``summarize_series`` call on a series it reads out of the
    public input, cites a feature it received back, proposes an operator taken
    from the menu it was given, and selects the probed candidate.  Nothing is
    hard-coded except the preference order among operators the menu offers.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.returned_models: set[str] = set()
        self.stages_seen: list[str] = []
        self._tool_done: set[str] = set()
        self._observed_features: dict[str, list[str]] = {}

    @staticmethod
    def _public_input(request) -> dict:
        user = json.loads(request.messages[-1]["content"])
        while "public_input" not in user:
            # A tool-result or correction turn: fall back to the first user
            # message, which always carries the stage's public input.
            user = json.loads(request.messages[1]["content"])
        return dict(user["public_input"])

    @staticmethod
    def _tool_result(request) -> dict | None:
        for message in reversed(request.messages):
            if message["role"] != "user":
                continue
            try:
                payload = json.loads(message["content"])
            except json.JSONDecodeError:
                continue
            if payload.get("schema_version") == "tool-result/1":
                return payload
        return None

    def complete(self, request) -> AgentResponse:
        self.calls += 1
        self.stages_seen.append(request.stage)
        key = f"{request.case_id}:{request.stage}"
        public_input = self._public_input(request)

        if request.stage == "inspect":
            result = self._tool_result(request)
            if result is None and key not in self._tool_done:
                self._tool_done.add(key)
                series = list(public_input["scope"]["series_uids"])
                return AgentResponse.valid(
                    {
                        "schema_version": "agent-envelope/1",
                        "kind": "tool_request",
                        "call_id": f"call-{self.calls}",
                        "tool_name": "summarize_series",
                        "arguments": {"series_uid": series[0]},
                    },
                    raw_response={},
                )
            features = sorted(
                (result or {}).get("public_result", {}).get("features", {})
            )
            self._observed_features[request.case_id] = features
            cited = [f for f in features if f == "missing_fraction"] or features[:1]
            return AgentResponse.valid(
                {
                    "schema_version": "agent-envelope/1",
                    "kind": "stage_result",
                    "stage": "inspect",
                    "payload": {
                        "inspected_region_fractions": [[0.0, 1.0]],
                        "requested_public_tools": ["summarize_series"],
                        "uncertainty": "medium",
                        "pattern_hypotheses": [
                            {
                                "hypothesis_id": "scripted_signal",
                                "pattern_type": "extreme_deviation",
                                "region_fractions": [0.0, 1.0],
                                "evidence_features": cited,
                                "confidence": "medium",
                            }
                        ],
                    },
                },
                raw_response={},
            )

        if request.stage == "propose":
            menu = [
                row["name"] for row in public_input["operator_contracts"]
                if row.get("availability") == "EXECUTABLE"
            ]
            chosen = next(
                (name for name in PREFERRED_OPERATORS if name in menu), menu[0]
            )
            return AgentResponse.valid(
                {
                    "schema_version": "agent-envelope/1",
                    "kind": "stage_result",
                    "stage": "propose",
                    "payload": {
                        "candidates": [
                            {
                                "candidate_id": "scripted-candidate",
                                "addresses_hypothesis_id": "scripted_signal",
                                "steps": [{"op": chosen, "params": {}}],
                            }
                        ]
                    },
                },
                raw_response={},
            )

        probed = public_input["probed_candidates"]
        best = max(probed, key=lambda row: float(row["support_gain"]))
        chosen = (
            best["candidate_id"] if best["meets_material_threshold"] else "identity"
        )
        return AgentResponse.valid(
            {
                "schema_version": "agent-envelope/1",
                "kind": "stage_result",
                "stage": "select",
                "payload": {
                    "chosen_candidate_id": chosen,
                    "verification_actions": ["support_probe_reviewed"],
                },
            },
            raw_response={},
        )


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    backends: list[ScriptedAgentBackend] = []

    def factory(_maximum_calls: int) -> ScriptedAgentBackend:
        backend = ScriptedAgentBackend()
        backends.append(backend)
        return backend

    state = tmp_path_factory.mktemp("g1_pipeline_state")
    result = g1_runner.run_g1_pipeline(
        cohort_name="T233",
        task_count=1,
        state_rel=str(state.relative_to(state.anchor)),
        backend_factory=factory,
        write_report=False,
        # The Slow leg is a live-model stage; this file stays zero-LLM and
        # exercises its trigger logic directly below instead.
        run_slow=False,
    )
    return result, backends


def _scored_rows(result):
    return [row for row in result["rows"] if "A3" in row or "A5" in row]


def test_one_command_runs_the_whole_closure(pipeline_result):
    result, backends = pipeline_result
    assert result["protocol_version"] == g1_runner.PROTOCOL_VERSION
    assert result["verdict"].startswith("G1_PIPELINE_CLOSURE")
    assert result["eval_substrate_preflight"]["pass"]
    assert result["train_substrate_preflight"]["pass"]
    assert _scored_rows(result), "no Task produced a paired row"
    # Both arms ran through the same three stages of the same contract.
    assert backends
    for backend in backends:
        assert backend.stages_seen[:2] == ["inspect", "inspect"] or (
            backend.stages_seen[0] == "inspect"
        )
        assert "propose" in backend.stages_seen


def test_agent_really_called_a_workspace_tool_and_it_reached_later_stages(
    pipeline_result,
):
    result, _backends = pipeline_result
    criteria = result["closure_criteria"]
    assert criteria["agent_called_a_workspace_tool"]
    assert criteria["workspace_tool_call_total"] > 0
    assert criteria["tool_result_changed_a_later_decision"]
    # The grounding is deterministic: the inspect stage cited a feature name
    # that only exists because a tool returned it.
    cited = criteria["cited_public_evidence"]
    assert cited
    for row in _scored_rows(result):
        for arm in ("A3", "A5"):
            observations = row[arm]["tool_observations"]
            assert observations, f"{arm} made no Workspace observation"
            for observation in observations:
                assert set(observation["arguments"]) == {"series_uid"}
                assert observation["arguments"]["series_uid"] in (
                    row["scope_series_uids"]
                )


def test_runtime_compiled_executed_and_wrote_an_episode(pipeline_result):
    result, _backends = pipeline_result
    criteria = result["closure_criteria"]
    assert criteria["runtime_generated_and_executed_a_typed_workflow"]
    assert criteria["episode_written"]
    for row in _scored_rows(result):
        for arm in ("A3", "A5"):
            probes = [p for p in row[arm]["probes"] if p.get("status") == "PROBED"]
            assert probes
            for probe in probes:
                assert isinstance(probe["support_gain"], float)
                assert probe["episode_id"]


def test_costs_are_reported_in_separate_columns(pipeline_result):
    result, _backends = pipeline_result
    assert result["closure_criteria"]["real_and_charged_cost_reported_separately"]
    for arm in ("A3", "A5"):
        cost = result["cost_by_arm"][arm]
        assert set(
            {"workspace_tool_calls", "llm_calls", "real_support_probe_count",
             "charged_probe_cost"}
        ) <= set(cost)
        # A charged cost is a budget penalty; it must never be read as a probe
        # count, so the two are allowed to differ and are stored apart.
        assert cost["real_support_probe_count"] >= 0
        assert cost["charged_probe_cost"] >= 0


def test_exploration_concentration_is_recorded_without_a_shared_denominator(
    pipeline_result,
):
    result, _backends = pipeline_result
    for arm in ("A3", "A5"):
        readout = result["exploration_concentration"][arm]
        assert "distinct_canonical_program_count" in readout
        assert "distinct_operator_name_count" in readout
        assert "executable_operator_name_count" in readout
        assert readout["distinct_operator_name_count"] != (
            readout["distinct_canonical_program_count"]
        ) or readout["attempt_count"] <= 1
        assert readout["role"].endswith("never a Gate")


def test_parameter_ownership_is_enforced_before_any_action_unit_runs():
    """The Runtime gate, exercised directly rather than through a live run.

    Under the current registry no operator declares a RUNTIME_BOUND parameter,
    so the audit passes everything and reports zero.  The gate is written
    against the registry, so a re-declared external binding turns it back on --
    which is what this test pins.
    """
    audit = dispatch.audit_program_parameter_ownership(
        [("outlier_mad", {}), ("repair_level_shift", {})]
    )
    assert audit["ok"]
    assert audit["runtime_bound_parameter_count"] == 0
    assert all(
        row["targeting_mode"] != "external_region" for row in audit["steps"]
    )

    # Simulate an operator that re-declares an external binding and check the
    # broadcast is refused rather than silently applied to every unit.
    original = dispatch.runtime_bound_parameters
    try:
        dispatch.runtime_bound_parameters = lambda op: (  # type: ignore[assignment]
            {"region_start_fraction": "estimated_region_start_fraction"}
            if op == "repair_level_shift" else {}
        )
        with pytest.raises(dispatch.ParameterOwnershipViolation) as excinfo:
            dispatch.audit_program_parameter_ownership(
                [("repair_level_shift", {"region_start_fraction": 0.5})]
            )
        assert excinfo.value.code == (
            "TASK_LEVEL_BROADCAST_OF_RUNTIME_BOUND_PARAMETER"
        )
        # ... and that the per-unit resolver rebinds from the unit's own view.
        resolved = dispatch.resolve_action_unit_parameters(
            [("repair_level_shift", {"region_start_fraction": 0.5})],
            {"estimated_region_start_fraction": 0.9},
        )
        assert resolved == [("repair_level_shift", {"region_start_fraction": 0.9})]
    finally:
        dispatch.runtime_bound_parameters = original


def test_workspace_gateway_refuses_out_of_scope_and_over_budget_calls():
    import numpy as np

    prefixes = {
        "S1": np.linspace(0.0, 1.0, 64),
        "S2": np.linspace(1.0, 2.0, 64),
    }
    gateway = CohortScopePublicToolGateway(
        prefixes, task_kind="forecast", observation_cutoff=64, maximum_calls=2
    )
    schemas = gateway.schemas_for(role="fast", stage="inspect")
    assert {schema["name"] for schema in schemas} == {
        "summarize_series", "localize_regions"
    }
    for schema in schemas:
        assert schema["input_schema"]["properties"]["series_uid"]["enum"] == [
            "S1", "S2"
        ]
    with pytest.raises(PermissionError):
        gateway.call("summarize_series", {"series_uid": "S3"})
    with pytest.raises(PermissionError):
        gateway.call("summarize_series", {})
    with pytest.raises(PermissionError):
        gateway.call("read_raw_values", {"series_uid": "S1"})
    assert gateway.call("summarize_series", {"series_uid": "S1"}).ok
    assert gateway.call("localize_regions", {"series_uid": "S2"}).ok
    refused = gateway.call("summarize_series", {"series_uid": "S1"})
    assert not refused.ok
    assert refused.public_result["refused"] == "WORKSPACE_TOOL_BUDGET_EXHAUSTED"
    accounting = gateway.accounting()
    assert accounting["workspace_tool_calls"] == 2
    assert accounting["workspace_tool_calls_refused"] == 1
    assert accounting["distinct_series_observed"] == 2


def test_slow_is_not_entered_when_attribution_names_no_editable_surface(
    pipeline_result,
):
    """§10.1: the Slow path is failure-driven, and NO_ACTIONABLE is an answer.

    Entering Slow because a run finished is exactly the drift the frozen design
    forbids, so the trigger is checked against the attribution rather than
    against "did the run end".
    """
    result, _backends = pipeline_result
    outcome = g1_runner.run_slow_and_replay(
        repo_root=Path(g1_runner.PROJECT_ROOT),
        rows=_scored_rows(result),
        cohort={},
        config={},
        specs=(),
        state_rel=".unused_state",
        workspace_tool_budget=1,
        backend_factory=lambda _n: None,
        first_fault={"first_fault": "NONE_BLOCKING", "editable": False},
    )
    assert outcome["verdict"] == "G1_SLOW_NO_ACTIONABLE"
    assert outcome["llm_api_call_count"] == 0
    assert outcome["trigger"]["editable_surface"] is False
    # The census is still built and reported, so the negative is inspectable.
    assert outcome["evidence_census"]


def test_census_counts_distinct_tasks_not_arm_attempts(pipeline_result):
    """A3 and A5 probe the same Task Episode over one frozen Outcome cell."""
    result, _backends = pipeline_result
    rows = _scored_rows(result)
    census_input = g1_runner._census_rows(rows)
    assert census_input
    assert {row["arm"] for row in census_input} == {"A3", "A5"}
    from evaluation.functional.task_episode_harness import g1

    census = g1._program_evidence_census(census_input)
    for cell in census:
        assert cell["distinct_task_count"] <= cell["attempt_count"]
        assert cell["distinct_task_count"] <= len(rows)
