"""P2 online abstain regression test (reviewer bug, 2026-08-16).

The first material failure currently routes through the online Program Supply
adapter, which has no expressibility evidence and no constrained-proposal
experiment.  The router must therefore return an empty catalog and the runner
must record ``abstained_by_route`` without:

- invoking Slow,
- consuming Slow replay / extra Support budget,
- raising ``UnboundLocalError``.

This test uses the same zero-LLM SealedProbeBackend / ScopeExecutor pattern as
the other functional tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner
import run_v1_sealed_a5_a3 as sealed

from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import (
    FaultRouter,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (
    ScopeExecutor,
)

DATASET = "p2_online_abstain_test"
OP = "winsorize"
ORIGIN = 400


def _series() -> np.ndarray:
    t = np.arange(1024, dtype=np.float64)
    return np.sin(t / 7.0) + 0.1 * np.sin(t / 3.0) + 5.0


def _failure_eval(roster, values, compiled, config, *, origin):
    """Baseline mean 1.0; any executed candidate mean 1.10 -> gain -0.10."""
    if compiled is None:
        return {"mean_smase": 1.0, "per_view_smase": [1.0],
                "behavior_point_count": 10}
    return {"mean_smase": 1.10, "per_view_smase": [1.10],
            "behavior_point_count": 10}


class _CountingSlowAgent:
    last_no_proposal_reason = None

    def __init__(self):
        self.calls = 0

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        self.calls += 1


@pytest.mark.parametrize("slow_options", [None, ()])
def test_material_failure_with_unknown_route_abstains_without_touching_slow(
    tmp_path, slow_options,
):
    values = {"s0": _series()}
    series0 = values["s0"]
    snapshot = runner._h0_snapshot()
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    backend = sealed.SealedProbeBackend(
        explore=True, operators=(OP,), force_pool=True
    )
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"),
    )
    method = TTHAMethod(TTHAFastAgent(core), snapshot, ())
    executor = ScopeExecutor(
        [{"series_uid": "s0", "role": "train"},
         {"series_uid": "s0", "role": "eval"}],
        values,
        {"anchors": []},
        evaluate_fn=_failure_eval,
    )
    request = runner._a5_request(series0, values, ORIGIN, DATASET)
    slow_agent = _CountingSlowAgent()

    result = run_online_round(
        method,
        executor,
        request,
        values,
        origin=ORIGIN,
        slow_agent=slow_agent,
        controller=controller,
        store=store,
        card_builder=lambda episode: {},
        round_name="p2_abstain",
        budget=2,
        allow_slow=True,
        domain=DATASET,
        period=24,
        fast_features=dict(extract_public_features(
            series0[:ORIGIN], task_kind="forecast")),
        slow_typed_patch_options=slow_options,
    )

    # No exception is the primary regression.  Then the budget/abstain
    # contract must hold exactly.
    assert result._slow_event is not None
    assert result._slow_event["stage"] == "abstained_by_route"
    assert result._slow_event["cause_code"] == "EXPRESSIBILITY_UNKNOWN"
    assert result._slow_event["actionability"] == "EVIDENCE_BACKLOG"
    assert slow_agent.calls == 0
    assert result.slow_replay_receipts_used == 0
    assert result.target_support_receipts_used == 1
    assert len(result.actual_probed_programs) == 1
    assert result.actual_probed_programs[0]["kind"] == "probe"
    assert result.actual_probed_programs[0]["gain"] == pytest.approx(-0.10)
    assert result.actual_probed_programs[0]["passed"] is True
    assert not any(
        item.get("kind") == "slow_replay"
        for item in result.actual_probed_programs
    )
    assert result.abstained is True
