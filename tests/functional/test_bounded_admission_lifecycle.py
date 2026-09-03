"""The bounded gate has to carry a candidate all the way to reuse, not just win.

Four inline ``relation == "POSITIVE"`` tests stood between a probe and a
reusable Skill: the winner, the Support-time Draft, the delayed LOCAL_ACTIVE,
and -- in the method layer -- ``handle_fast_winner``'s pending and
``handle_feedback_delayed``'s approval.  Relaxing only some of them produces the
worst outcome available: the arm deploys for the round, nothing persists, and
the next origin still has nothing to reuse.  So this walks the whole chain

    Support-A bounded CONFLICT -> winner -> LOCAL_DRAFT / pending
    -> Support-B bounded CONFLICT -> approved / LOCAL_ACTIVE
    -> next round retrieves and actually uses it

and then checks the two rejections that must survive the relaxation: a
Support-B sign reversal, and harm past the frozen budget.

0 LLM: the Fast Path runs on ``SealedProbeBackend`` and the Consumer is a stub.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402

from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import (  # noqa: E402
    FaultRouter,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha import admission_policy as ap  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    WindowVerification,
)

ORIGIN = 400
DELAYED_ORIGIN = ORIGIN + 48
ALT_OP = "outlier_mad"
DOMAIN = "bounded-admission-test"
N_SERIES = 5

# 1 of 5 series harmed = 20% exactly, worst loss 0.10: inside the frozen P4b
# budget (<=20%, <=0.30) and a CONFLICT under the strict rule.
WITHIN_BUDGET = (0.5, 0.5, 0.5, 0.5, -0.10)
# 2 of 5 = 40% harmed: over the fraction budget.
OVER_FRACTION = (0.5, 0.5, 0.5, -0.10, -0.10)
# 1 of 5 harmed but by 0.50: over the single-series budget.
OVER_MAGNITUDE = (0.9, 0.9, 0.9, 0.9, -0.50)
# Aggregate negative: the sign reversal that must still be refused.
SIGN_REVERSAL = (-0.3, -0.3, -0.3, -0.3, -0.3)


def _bounded_policy() -> ap.AdmissionPolicy:
    return ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20, max_single_series_harm=0.30
    )


@pytest.fixture(autouse=True)
def _reset_policy():
    yield
    ap.reset_policy()


def _values() -> dict[str, np.ndarray]:
    x = np.arange(1024, dtype=np.float64)
    return {
        "s%d" % i: np.sin(x / (7.0 + i)) + 0.1 * np.sin(x / 3.0) + 5.0 + i
        for i in range(N_SERIES)
    }


def _verification(op: str) -> WindowVerification:
    result = WindowVerification(
        passed=True,
        checked_windows=1,
        window_modified_flags=(True,),
        window_identity_equivalent_flags=(False,),
    )
    result._program_supply_prepared_values = (
        np.asarray([float(sum(ord(ch) for ch in op))]),
    )
    return result


class _PerSeriesExecutor:
    """Stub Consumer whose per-series split is chosen per face."""

    def __init__(self, support: tuple[float, ...], delayed: tuple[float, ...]):
        self.support = support
        self.delayed = delayed

    def verify(self, steps, origin):
        return _verification(str(steps[0][0]) if steps else "identity")

    def evaluate(self, steps, origin):
        op = str(steps[0][0]) if steps else "identity"
        if op != ALT_OP:
            per = (-0.10,) * N_SERIES
        else:
            per = self.delayed if int(origin) >= DELAYED_ORIGIN else self.support
        return SimpleNamespace(
            verification=_verification(op),
            gain=float(np.mean(per)),
            per_view_gain=tuple(float(v) for v in per),
            behavior_point_count=1,
        )


def _method(snapshot, series, backend):
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series[:ORIGIN], task_kind="forecast")
    )
    return TTHAMethod(TTHAFastAgent(core), snapshot, ())


def _round(method, executor, values, series, *, name, store=None, controller=None):
    request = runner._a5_request(series, values, ORIGIN, DOMAIN)
    features = dict(extract_public_features(series[:ORIGIN], task_kind="forecast"))
    return run_online_round(
        method,
        executor,
        request,
        values,
        origin=ORIGIN,
        slow_agent=None,
        # handle_fast_winner -- the method-layer persistence gate this test
        # exists for -- only runs when both of these are present.
        controller=controller,
        store=store,
        card_builder=lambda _episode: {},
        round_name=name,
        budget=2,
        # Fast winner -> Target-local Draft Skill; without this the method-layer
        # gate is never reached and the chain stops at the round's winner.
        allow_fast_skill=True,
        # P4b does not exercise Slow this round; the Slow Support gate is
        # therefore untouched and out of scope for this test.
        allow_slow=False,
        domain=DOMAIN,
        period=24,
        fast_features=features,
    )


def _drive(support, delayed, tmp_path, *, policy):
    values = _values()
    series = values["s0"]
    ap.install_policy(policy)
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    method = _method(
        runner._h0_snapshot(),
        series,
        sealed.SealedProbeBackend(
            explore=True, operators=(ALT_OP,), force_pool=True
        ),
    )
    executor = _PerSeriesExecutor(support, delayed)
    first = _round(method, executor, values, series, name="a_face",
                   store=store, controller=controller)
    return method, executor, store, first, values, series


def test_strict_default_refuses_the_same_conflict(tmp_path):
    _method_, _executor, _store, first, _values_, _series_ = _drive(
        WITHIN_BUDGET, WITHIN_BUDGET, tmp_path, policy=ap.DEFAULT
    )
    # Identical readings, strict rule: the CONFLICT wins nothing at all.
    assert first.winner_program is None
    assert first.abstained is True


def test_bounded_conflict_reaches_draft_then_active_then_reuse(tmp_path):
    method, executor, store, first, values, series = _drive(
        WITHIN_BUDGET, WITHIN_BUDGET, tmp_path, policy=_bounded_policy()
    )

    # A face: the in-budget CONFLICT earns the round.
    assert first.winner_program == [{"op": ALT_OP, "params": {}}]
    assert first.abstained is False
    probe = next(
        row for row in first.actual_probed_programs
        if row.get("kind") == "probe" and row.get("admission")
    )
    assert probe["admission"]["admitted"] is True
    assert probe["admission"]["reason"] == "within_risk_budget"
    assert probe["admission"]["harmed_count"] == 1

    # ... and is retained rather than left as evidence only.
    episode, _steps = first._episodes[-1]
    assert episode.relation == "CONFLICT"
    assert episode.local_status == "LOCAL_DRAFT"
    assert episode.support_response["accepted"] is True

    # Method layer: the in-budget CONFLICT forms a pending update rather than
    # being refused as "relation_conflict".
    assert first._fast_skill_event["stage"] == "pending", first._fast_skill_event
    assert first._fast_skill_event["support_admission"]["admitted"] is True

    # B face: an independent in-budget reading approves it.
    open_delayed(first, executor, delayed_origin=DELAYED_ORIGIN, store=store)
    assert first._delayed_event["stage"] == "approved", first._delayed_event
    assert first._delayed_event["delayed_admission"]["admitted"] is True
    assert first.approved_skill_id
    assert activate_approved(first, store) is True

    # Next round: really retrieved and really used, not merely in the pool.
    skill_ids = {
        skill.skill_id for skill in method._active_snapshot().skills
    }
    replay_method = _method(
        method._active_snapshot(),
        series,
        sealed.SealedProbeBackend(
            explore=False, operators=(), force_pool=True,
            prefer_skill_in_select=True,
        ),
    )
    replay = _round(replay_method, executor, values, series, name="reuse")
    trace = replay_method.last_trace
    retrieved = set(trace.retrieved_skill_ids or ())
    assert retrieved & skill_ids, (retrieved, skill_ids)
    assert replay.winner_program == [{"op": ALT_OP, "params": {}}]


@pytest.mark.parametrize(
    "delayed_face",
    [SIGN_REVERSAL, OVER_FRACTION, OVER_MAGNITUDE],
    ids=["sign_reversal", "over_fraction", "over_magnitude"],
)
def test_bounded_still_refuses_on_the_independent_face(tmp_path, delayed_face):
    _method_, executor, store, first, _values_, _series_ = _drive(
        WITHIN_BUDGET, delayed_face, tmp_path, policy=_bounded_policy()
    )
    assert first.winner_program == [{"op": ALT_OP, "params": {}}]
    open_delayed(first, executor, delayed_origin=DELAYED_ORIGIN, store=store)
    assert first._delayed_event["stage"] == "delayed_rejected", first._delayed_event
    assert first._delayed_event["delayed_admission"]["admitted"] is False
    assert not first.approved_skill_id
    assert activate_approved(first, store) is not True


@pytest.mark.parametrize(
    "support_face", [OVER_FRACTION, OVER_MAGNITUDE],
    ids=["over_fraction", "over_magnitude"],
)
def test_bounded_refuses_out_of_budget_support(tmp_path, support_face):
    _method_, _executor, _store, first, _values_, _series_ = _drive(
        support_face, WITHIN_BUDGET, tmp_path, policy=_bounded_policy()
    )
    assert first.winner_program is None
    assert first.abstained is True


def test_every_probed_candidate_carries_its_program_and_risk(tmp_path):
    """Slow can only patch what the receipt describes.

    A Patch moves Workflow, targeting, or strength -- all of which live in the
    program steps -- and it decides using the per-series risk of the failure.
    Recording those only for the winner leaves a rejected candidate as a bare
    id and an aggregate, which is not enough to repair from, and makes a
    "strict refused / bounded admitted" pair impossible to compare on program
    rather than on outcome.
    """
    _method, _executor, _store, first, _values, _series = _drive(
        WITHIN_BUDGET, WITHIN_BUDGET, tmp_path, policy=ap.DEFAULT
    )
    probes = [
        row for row in first.actual_probed_programs if row.get("kind") == "probe"
    ]
    assert probes, "the strict run still probes; it just admits nothing"
    for probe in probes:
        # Strict refused this candidate: it has no winner_program anywhere.
        assert probe["admission"]["admitted"] is False
        steps = probe["program_steps"]
        assert steps and all("op" in step and "params" in step for step in steps)
        risk = probe["risk_profile"]
        assert risk["series_read"] > 0
        assert 0.0 <= risk["harmed_fraction"] <= 1.0
        assert risk["max_single_series_harm"] >= 0.0
        assert len(probe["per_series_gain"]) == risk["series_read"]


def test_the_recorded_risk_profile_matches_the_admission_reading(tmp_path):
    # The profile is derived from the same per-series readings the gate used,
    # so a receipt cannot disagree with the decision it records.
    _method, _executor, _store, first, _values, _series = _drive(
        WITHIN_BUDGET, WITHIN_BUDGET, tmp_path, policy=_bounded_policy()
    )
    probe = next(
        row for row in first.actual_probed_programs
        if row.get("kind") == "probe" and row.get("admission", {}).get("harmed_count")
        is not None
    )
    assert probe["risk_profile"]["harmed_count"] == probe["admission"]["harmed_count"]
    assert probe["risk_profile"]["max_single_series_harm"] == pytest.approx(
        probe["admission"]["max_single_series_harm"]
    )
