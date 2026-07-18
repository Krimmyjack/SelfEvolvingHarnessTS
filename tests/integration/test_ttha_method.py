from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.method_api import MethodSeriesView
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.method_compat import BenchmarkMethodAdapter
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway
from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse, ReplayAgentBackend


ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods" / "ttha" / "harness" / "h0"


def _stage(stage, payload):
    return AgentResponse.valid(
        {
            "schema_version": "agent-envelope/1",
            "kind": "stage_result",
            "stage": stage,
            "payload": payload,
        },
        raw_response={"id": f"ttha-adapter-{stage}"},
    )


def _method(values, *, program=False):
    candidates = (
        [
            {
                "candidate_id": "agent-0",
                "steps": [{"op": "impute_linear", "params": {}}],
            }
        ]
        if program
        else []
    )
    choice = "agent-0" if program else "identity"
    backend = ReplayAgentBackend(
        [
            _stage(
                "inspect",
                {
                    "inspected_region_fractions": [[0.0, 1.0]],
                    "requested_public_tools": [],
                    "uncertainty": "low",
                },
            ),
            _stage("propose", {"candidates": candidates}),
            _stage(
                "select",
                {
                    "chosen_candidate_id": choice,
                    "verification_actions": ["scope_checked"],
                },
            ),
        ]
    )
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(values, task_kind="forecast"),
    )
    return TTHAMethod(TTHAFastAgent(core), compile_snapshot(H0_ROOT))


def test_benchmark_adapter_accepts_ttha_identity():
    values = np.sin(np.arange(160, dtype=float) / 8.0)
    series = MethodSeriesView("u0", values)
    prepared = BenchmarkMethodAdapter(_method(values)).prepare(
        series, forecast_task_spec_v1(horizon=12), {}
    )
    assert prepared.series_uid == series.series_uid
    assert np.array_equal(prepared.values, values)
    assert prepared.operators == ()


def test_benchmark_adapter_accepts_ttha_program_choice():
    values = np.sin(np.arange(160, dtype=float) / 8.0)
    values[10:14] = np.nan
    series = MethodSeriesView("u1", values)
    prepared = BenchmarkMethodAdapter(_method(values, program=True)).prepare(
        series, forecast_task_spec_v1(horizon=12), {}
    )
    assert np.isfinite(prepared.values).all()
    assert prepared.operators == ("impute_linear",)

