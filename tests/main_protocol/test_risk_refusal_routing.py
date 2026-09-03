"""A refusal the gate computed must reach the router that can act on it.

``_admission_decide`` already derives *why* a candidate was refused -- harmed
fraction, worst single-series loss -- and the probe row already carries it.  The
fault router beneath it read only the aggregate gain, so a candidate that was
materially positive and refused on the tail budget matched neither branch: it
never became a winner, never entered the Slow Path, and never even incremented
``harm_count``.  A round made entirely of such candidates reported zero faults.

That is the class of evidence Scope revision exists to act on -- the program
works, its serving scope is too wide -- so these locks pin both halves:

* the new branch fires on exactly that class, carrying the material a Patch
  would need (program, scope, per-series gains, refusal reason);
* every historical path is untouched, including the case that matters most --
  under the library default policy the new branch is inert, because a strict
  refusal reports ``relation_not_positive`` and never a budget reason.

0 LLM: the Fast Path runs on ``SealedProbeBackend`` and the Consumer is a stub
returning a fixed per-series vector, so the gate's verdict is exact.
"""
from __future__ import annotations

import dataclasses
import json
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
from SelfEvolvingHarnessTS.methods.ttha import online_loop as loop  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    WindowVerification,
)

ORIGIN = 400
ALT_OP = "outlier_mad"
DOMAIN = "risk-refusal-routing-test"
N_SERIES = 5

#: Aggregate +0.30, one series past the 0.30 harm line: refused on the tail.
HARM_OVER_BUDGET = (0.5, 0.5, 0.5, 0.5, -0.50)
#: Aggregate +0.52, two of five harmed: refused on the 0.20 harmed fraction.
FRACTION_OVER_BUDGET = (1.0, 1.0, 1.0, -0.2, -0.2)
#: Aggregate +0.36, both budgets respected: this one earns deployment.
ADMITTED = (0.5, 0.5, 0.5, 0.5, -0.2)
#: The historical Slow entry: the aggregate itself goes materially negative.
AGGREGATE_NEGATIVE = (-0.5,) * N_SERIES
#: Exactly no behaviour: below the material line, and not a risk fault at all.
NO_OP = (0.0,) * N_SERIES


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
        passed=True, checked_windows=1,
        window_modified_flags=(True,),
        window_identity_equivalent_flags=(False,),
    )
    result._program_supply_prepared_values = (
        np.asarray([float(sum(ord(ch) for ch in op))]),
    )
    return result


class _VectorExecutor:
    """A Consumer stub whose per-series split is fixed, so the gate is exact.

    ``identity`` is a true no-op, which keeps the first probe of every round
    from being a fault in its own right and confounding the assertion.
    """

    def __init__(self, per_series) -> None:
        self.per_series = tuple(float(value) for value in per_series)

    def verify(self, steps, origin):
        return _verification(str(steps[0][0]) if steps else "identity")

    def evaluate(self, steps, origin, serving_scope=None):
        op = str(steps[0][0]) if steps else "identity"
        per = NO_OP if op != ALT_OP else self.per_series
        return SimpleNamespace(
            verification=_verification(op), gain=float(np.mean(per)),
            per_view_gain=tuple(float(value) for value in per),
            behavior_point_count=1,
        )


def _drive(tmp_path, per_series, *, policy=None, allow_slow=True):
    """One round against a Consumer with the given per-series outcome."""
    values = _values()
    series = values["s0"]
    ap.install_policy(policy if policy is not None else _bounded_policy())
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    method = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(
            sealed.SealedProbeBackend(
                explore=True, operators=(ALT_OP,), force_pool=True),
            LocalPublicToolGateway(series[:ORIGIN], task_kind="forecast"))),
        runner._h0_snapshot(), ())
    executor = _VectorExecutor(per_series)
    features = dict(extract_public_features(series[:ORIGIN], task_kind="forecast"))
    result = loop.run_online_round(
        method, executor, runner._a5_request(series, values, ORIGIN, DOMAIN),
        values, origin=ORIGIN, slow_agent=None, controller=controller,
        store=store, card_builder=lambda _episode: {},
        round_name="risk-refusal", budget=8, allow_fast_skill=True,
        allow_slow=allow_slow, domain=DOMAIN, period=24, fast_features=features,
    )
    return result


def test_a_tail_risk_refusal_now_reaches_the_fault_router(tmp_path):
    """The whole point: refused on the tail budget, and Slow is entered."""
    result = _drive(tmp_path, HARM_OVER_BUDGET)
    assert result.risk_refusal_count == 1
    assert result._slow_trigger == "risk_refusal"
    # Entering the router has to leave a receipt, whatever it decided.
    assert result._slow_event is not None
    assert result.winner_program is None


def test_the_refusal_carries_what_a_patch_would_have_to_act_on(tmp_path):
    """A fault with no program, scope or per-series reading is not actionable."""
    result = _drive(tmp_path, FRACTION_OVER_BUDGET)
    assert result.risk_refusal_count == 1
    (refusal,) = result.risk_refusals
    assert refusal["reason"] == "harmed_fraction_over_budget"
    assert refusal["aggregate_gain"] == pytest.approx(0.52)
    assert refusal["harmed_fraction"] == pytest.approx(0.40)
    assert refusal["max_single_series_harm"] == pytest.approx(0.20)
    assert [step["op"] for step in refusal["program_steps"]] == [ALT_OP]
    assert refusal["per_series_gain"] == list(FRACTION_OVER_BUDGET)
    assert refusal["episode_id"]
    # "serving_scope" is present even when this round carried none, so a reader
    # can tell "no scope" from "the field was dropped".
    assert "serving_scope" in refusal


def test_the_historical_negative_aggregate_path_is_unchanged(tmp_path):
    """The old entry still fires, still counts harm, and is labelled as itself."""
    result = _drive(tmp_path, AGGREGATE_NEGATIVE)
    assert result._slow_trigger == "aggregate_negative"
    assert result.harm_count == 1
    assert result.harm_magnitude == pytest.approx(0.5)
    assert result.risk_refusal_count == 0
    assert result.risk_refusals == []


def test_a_no_op_is_not_a_risk_fault_and_routes_nowhere(tmp_path):
    """Below the material line is a different fault; Scope cannot repair it."""
    result = _drive(tmp_path, NO_OP)
    assert result.risk_refusal_count == 0
    assert result._slow_trigger is None
    assert result.harm_count == 0


def test_an_admitted_candidate_still_wins_and_raises_no_fault(tmp_path):
    """Inside both budgets the gate grants deployment, exactly as before."""
    result = _drive(tmp_path, ADMITTED)
    assert result.winner_program is not None
    assert [step["op"] for step in result.winner_program] == [ALT_OP]
    assert result.risk_refusal_count == 0
    assert result._slow_trigger is None


def test_under_the_default_policy_the_new_branch_is_inert(tmp_path):
    """The strongest lock: nothing changes unless a runner installs BOUNDED_V1.

    Under ``strict_positive_only`` the same CONFLICT is refused for
    ``relation_not_positive``, which is not a budget reason, so the candidate
    falls through both branches exactly as it did before this change -- no
    trigger, no counter, no harm.  A runner that never installs a policy
    therefore cannot observe the new path at all.
    """
    result = _drive(tmp_path, HARM_OVER_BUDGET, policy=ap.DEFAULT)
    assert result.risk_refusal_count == 0
    assert result._slow_trigger is None
    assert result.harm_count == 0
    assert result.winner_program is None


# --------------------------------------------------------------------------
# Isolation, measured rather than asserted.
#
# The locks above name the fields they expect.  That is only as good as the
# list, and a list cannot prove a *negative* -- that nothing else moved.  These
# compare the entire RoundResult between the branch being live and the branch
# being neutralised, so a field nobody thought to check cannot drift silently.
#
# Setting RISK_REFUSAL_REASONS to () reproduces the pre-change control flow
# exactly: ``_risk_refused`` is then always False, the counter never
# increments, nothing is appended to ``_risk_contexts``, and the end-of-round
# ``_fire_risk_refusal_slow`` never runs.  It is the "before" oracle, not an
# approximation of one.
# --------------------------------------------------------------------------

#: A live object handle whose repr carries its address; not round state.
VOLATILE_FIELDS = ("_method",)

#: What the branch is allowed to touch, on the class it exists to serve.
#: ``_risk_refusal_selection`` joined the set in P4U-v3, when the Slow call
#: moved to the end of the round so that it could *choose* which refusal to
#: spend itself on: the choice, and the ranking behind it, are round state, and
#: an artifact recording only the winner could not answer whether the rule made
#: any difference to what Slow was asked.
EXPECTED_BLAST_RADIUS = {
    "risk_refusal_count", "risk_refusals", "_slow_trigger", "_slow_event",
    "_risk_refusal_selection",
}

HISTORICAL_CLASSES = {
    "admitted": ADMITTED,
    "aggregate_negative": AGGREGATE_NEGATIVE,
    "no_op": NO_OP,
}
RISK_REFUSAL_CLASSES = {
    "single_series_harm": HARM_OVER_BUDGET,
    "harmed_fraction": FRACTION_OVER_BUDGET,
}


def _observable(result) -> dict:
    """Every declared field of the round, normalised for comparison."""
    return {
        field.name: json.loads(json.dumps(
            getattr(result, field.name),
            default=lambda o: dict(o) if hasattr(o, "items") else (
                list(o) if hasattr(o, "__iter__") else repr(o))))
        for field in dataclasses.fields(loop.RoundResult)
        if field.name not in VOLATILE_FIELDS
    }


def _changed_fields(tmp_path, monkeypatch, per_series) -> set[str]:
    """Which fields move when the branch goes from neutralised to live."""
    monkeypatch.setattr(loop, "RISK_REFUSAL_REASONS", ())
    before = _observable(_drive(tmp_path / "before", per_series))
    monkeypatch.undo()
    after = _observable(_drive(tmp_path / "after", per_series))
    return {name for name, value in before.items() if after[name] != value}


@pytest.mark.parametrize("per_series", list(HISTORICAL_CLASSES.values()),
                         ids=list(HISTORICAL_CLASSES))
def test_a_round_is_reproducible_so_the_isolation_locks_mean_something(
        tmp_path, per_series):
    """Guards the locks below: if rounds drifted, comparing them would be noise."""
    first = _observable(_drive(tmp_path / "first", per_series))
    second = _observable(_drive(tmp_path / "second", per_series))
    assert first == second


@pytest.mark.parametrize("per_series", list(HISTORICAL_CLASSES.values()),
                         ids=list(HISTORICAL_CLASSES))
def test_the_historical_paths_are_bit_identical_with_the_branch_live(
        tmp_path, monkeypatch, per_series):
    """Not "the fields we checked match" -- no field of the round moves at all."""
    assert _changed_fields(tmp_path, monkeypatch, per_series) == set()


@pytest.mark.parametrize("per_series", list(RISK_REFUSAL_CLASSES.values()),
                         ids=list(RISK_REFUSAL_CLASSES))
def test_the_branch_moves_only_its_own_fields_on_the_class_it_serves(
        tmp_path, monkeypatch, per_series):
    """It must do something here -- and nothing beyond its own blast radius."""
    changed = _changed_fields(tmp_path, monkeypatch, per_series)
    assert changed == EXPECTED_BLAST_RADIUS
