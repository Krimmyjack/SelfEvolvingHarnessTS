"""sol's v1.1 addendum A/B/C, checked where each one could go wrong quietly.

**A -- lineage dedupe on the full census key.**  The failure to catch is a
closed key coming back as a fresh ADD: the new shell carries ``revisions=0`` and
``verification_attempts=0``, so a Draft that has spent its budget gets a second
one and the two-revision bound becomes decorative.

**B -- the replay prediction cache.**  A cache that is *nearly* right is worse
than none, because every screen after the first would be judged on numbers the
uncached path never produced.  So the test is bitwise equality against the
uncached reading on five Scope shapes, not a tolerance.

**C -- future-step budget reservation.**  The failure to catch is an early step
with several candidates eating the whole allowance, after which every later step
is blocked before it sees its first candidate.

0 LLM throughout.  The cache tests do real Consumer fits on two real KDD cells
because the property under test is arithmetic identity with the real evaluator;
a stub would prove nothing about the thing that could actually be wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

from evaluation.main_protocol_p4 import hec1_contract as contract  # noqa: E402
from evaluation.main_protocol_p4 import outer_loop  # noqa: E402
from evaluation.main_protocol_p4 import restricted_draft as drafts  # noqa: E402
from evaluation.main_protocol_p4 import run_hec1 as runner  # noqa: E402

Z = "local_robust_z_peak"
TASK = runner.TASK_CONSUMER_KEY
STEPS = (("outlier_mad", {}),)

WIDE = {"scope_type": "serving_series_predicate",
        "predicate": [{"feature": Z, "op": ">=", "threshold": 3.0}]}
NARROW = {"scope_type": "serving_series_predicate",
          "predicate": [{"feature": Z, "op": ">=", "threshold": 6.0}]}


def _bank_row(origin, gains, scope=None, op="outlier_mad"):
    return {
        "unit": {"block": "[0:40]", "origin": origin},
        "task_consumer_key": TASK,
        "program_steps": [{"op": op, "params": {}}],
        "serving_scope": scope,
        "features": {uid: {Z: 9.0} for uid in gains},
        "per_series_gain": dict(gains),
    }


GOOD = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1, "e": 0.1}


def _replay(msh=0.02):
    def replay(*, steps, scope):
        return {"cells": [{"unit": {"block": "[0:40]", "origin": 1176},
                           "treated": 5, "aggregate_gain": 0.22,
                           "harmed_fraction": 0.0,
                           "max_single_series_harm": msh}],
                "fits": 3}
    replay.estimated_fits_per_candidate = 3
    replay.cells = 1
    return replay


# ---------------------------------------------------------------------------
# A. lineage dedupe on the full census key
# ---------------------------------------------------------------------------

def test_one_key_appearing_repeatedly_leaves_exactly_one_lineage():
    ledger = drafts.DraftLedger()
    bank = [_bank_row(1176, GOOD, WIDE), _bank_row(1896, GOOD, WIDE),
            _bank_row(2136, GOOD, WIDE)]
    first = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=1,
                                   replay=_replay())
    assert len(first.drafts_opened) == 1
    # A later step sees the same key again and must not open a second lineage.
    second = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=2,
                                    replay=_replay())
    assert not second.drafts_opened
    assert len(ledger.drafts) == 1
    assert len(ledger.lineage_keys()) == 1


def test_one_program_under_two_root_scopes_is_two_lineages():
    ledger = drafts.DraftLedger()
    bank = [_bank_row(1176, GOOD, WIDE), _bank_row(1896, GOOD, NARROW)]
    step = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=1,
                                  replay=_replay())
    assert len(step.drafts_opened) == 2
    assert len(ledger.lineage_keys()) == 2
    roots = {runner.json.dumps(d.root_scope, sort_keys=True)
             for d in ledger.drafts}
    assert len(roots) == 2


def test_a_closed_key_cannot_reopen_under_a_fresh_shell():
    """The bypass this rule exists for: a new shell resets both counters."""
    ledger = drafts.DraftLedger()
    bank = [_bank_row(1176, GOOD, WIDE)]
    step = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=1,
                                  replay=_replay())
    draft = ledger.by_id(step.drafts_opened[0])
    ledger.close(draft, "EFFECT_NONSTATIONARY")
    assert draft.closed

    reopened = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=2,
                                      replay=_replay())
    assert not reopened.drafts_opened, "a closed key reopened"
    assert len(ledger.drafts) == 1
    # And the ledger refuses it even if the census is bypassed entirely.
    with pytest.raises(ValueError):
        ledger.open_restricted(
            program_steps=STEPS, root_scope=WIDE, current_scope=WIDE,
            origin=2, census_key=draft.census_key)


def test_held_is_read_from_the_lineage_not_from_bank_source_skill_ids():
    """A card can hold a key with no row in this window naming it."""
    ledger = drafts.DraftLedger()
    key = outer_loop.census_key(TASK, [{"op": "outlier_mad", "params": {}}],
                                WIDE)
    bank = [_bank_row(1176, GOOD, WIDE)]      # no source_skill_id anywhere
    step = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=1,
                                  replay=_replay(), held_lineage_keys=[key])
    assert not step.drafts_opened
    assert not [row for row in step.candidates if row["kind"] == "ADD"]


def test_the_census_key_is_task_program_and_root_scope():
    left = outer_loop.census_key(TASK, [{"op": "outlier_mad", "params": {}}],
                                 WIDE)
    same = outer_loop.census_key(TASK, [{"op": "outlier_mad", "params": {}}],
                                 WIDE)
    other_scope = outer_loop.census_key(
        TASK, [{"op": "outlier_mad", "params": {}}], NARROW)
    other_params = outer_loop.census_key(
        TASK, [{"op": "outlier_mad", "params": {"k": 3}}], WIDE)
    other_order = outer_loop.census_key(
        TASK, [{"op": "winsorize", "params": {}},
               {"op": "outlier_mad", "params": {}}], WIDE)
    reversed_order = outer_loop.census_key(
        TASK, [{"op": "outlier_mad", "params": {}},
               {"op": "winsorize", "params": {}}], WIDE)
    assert left == same
    assert len({left, other_scope, other_params, other_order,
                reversed_order}) == 5


# ---------------------------------------------------------------------------
# the outer Slow's own call path
# ---------------------------------------------------------------------------

HARMED = {"a": 0.4, "b": 0.3, "c": 0.2, "d": -0.9, "e": -0.8}


class _OneClauseBackend:
    """A relay that answers, so everything above the wire is the real thing.

    The mock stops at the reply: ``TTHAAgentCore.run_stage`` builds the real
    system message from the real resolved Harness view, validates against the
    real ``slow_scope_clause_v1`` schema, and only the bytes that would have
    come back over HTTP are supplied here.  That is the layer the first Forward
    attempt died above -- ``harness_view={}`` never reached a backend at all.
    """

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def complete(self, request: Any) -> Any:
        from SelfEvolvingHarnessTS.runtime.agent_backend import AgentResponse

        self.requests.append(request)
        return AgentResponse.valid(
            {"schema_version": "agent-envelope/1", "kind": "stage_result",
             "stage": "edit",
             "payload": {"scope_clause": {"feature": Z, "op": ">=",
                                          "threshold": 4.0},
                         "rationale": "test: spiky series only"}},
            raw_response={"id": "test-one-clause"})


def _real_core(backend):
    import numpy as np

    from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        LocalPublicToolGateway,
    )

    return TTHAAgentCore(
        backend,
        LocalPublicToolGateway(np.zeros(8, dtype=np.float64),
                               task_kind="forecast"),
        model="test-model", base_url=runner.OFFLINE_BASE_URL)


def _h0_snapshot():
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )

    return compile_snapshot(ROOT / "methods/ttha/harness/h0", verify_lock=False)


def test_the_outer_slow_asks_through_the_real_stage_on_a_narrow_candidate():
    """The regression the 28 end-to-end cells could not catch.

    Every offline course scripts the outer Slow, so no test had ever driven
    ``OuterSlowAgent`` itself; and Phase S's two outer steps found no candidate
    that needed a clause (``llm_outer`` 0), so the live path was first entered
    by Forward's A3-online at unit 5 -- and died there on ``harness_view={}``.
    This drives a real NARROW candidate through ``consolidate`` into the real
    ``run_stage``, with only the relay's reply supplied.
    """
    backend = _OneClauseBackend()
    snapshot = _h0_snapshot()
    guard = runner.BudgetGuard(ordering_cap=8, per_unit_arm_cap=8,
                               ledgers=runner.Ledgers())
    slow = runner.OuterSlowAgent(
        _real_core(backend), vocabulary=contract.SCOPE_CLASS["vocabulary"],
        guard=guard, snapshot=snapshot)

    ledger = drafts.DraftLedger()
    key = outer_loop.census_key(TASK, [{"op": "outlier_mad", "params": {}}],
                                WIDE)
    bank = [_bank_row(1176, HARMED, WIDE), _bank_row(1896, HARMED, WIDE)]
    step = outer_loop.consolidate(
        bank=bank, ledger=ledger, k_index=1, slow=slow, replay=_replay(),
        held_lineage_keys=[key])

    narrowing = [row for row in step.candidates if row["kind"] == "NARROW"]
    assert narrowing, "no NARROW candidate; the Slow path was never entered"
    assert step.to_dict()["slow_calls"] >= 1
    assert backend.requests, "run_stage never reached the relay"
    assert slow.calls and slow.calls[0]["returned"] == {
        "feature": Z, "op": ">=", "threshold": 4.0}


def test_the_outer_slow_resolves_a_real_harness_view_not_an_empty_dict():
    """``run_stage`` reads ``harness_view.instruction``; a dict has none."""
    backend = _OneClauseBackend()
    snapshot = _h0_snapshot()
    slow = runner.OuterSlowAgent(
        _real_core(backend), vocabulary=contract.SCOPE_CLASS["vocabulary"],
        guard=runner.BudgetGuard(ordering_cap=4, per_unit_arm_cap=4,
                                 ledgers=runner.Ledgers()),
        snapshot=snapshot)
    payload = slow(candidate={"kind": "NARROW", "rows": [],
                              "base_scope": WIDE,
                              "program_steps": [{"op": "outlier_mad",
                                                 "params": {}}]},
                   rejected=())
    assert payload["scope_clause"]["feature"] == Z
    request = backend.requests[0]
    system = "\n".join(str(message.get("content") or "")
                       for message in request.messages
                       if message.get("role") == "system")
    # The bootstrap instruction and the h0 procedural Skills are what Slow is
    # supposed to be looking at; an empty view would carry neither.
    assert system.strip(), "no system message was built"
    assert "Resolved Harness" in system
    assert "inspect_and_localize" in system
    # And the snapshot identity travels with the call.
    assert request.source_harness_snapshot_sha == snapshot.runtime_bundle_sha


def test_the_runner_hands_the_arm_snapshot_to_the_outer_slow_factory():
    """Wiring lock: the factory takes a snapshot, and the arm passes its own."""
    import inspect

    source = inspect.getsource(runner.Arm.outer_step)
    assert "self.active_snapshot()" in source
    assert "harness_view={}" not in inspect.getsource(runner.OuterSlowAgent)
    params = inspect.signature(runner.OuterSlowAgent.__init__).parameters
    assert "snapshot" in params
    # Required, not defaulted: the class cannot be constructed unarmed again.
    assert params["snapshot"].default is inspect.Parameter.empty


def test_a_resumed_arm_still_has_a_snapshot_to_ask_with():
    """Resume replays cells from checkpoints and never calls begin_unit.

    ``_method`` is then still None when the outer step runs, so the snapshot
    has to come from somewhere that exists -- the arm's start snapshot, which
    is the truthful answer for an arm that has learned nothing this process.
    """
    spec = runner.arm_specs(k0_empty=True)[-1]
    assert spec.outer, "expected the online arm"
    arm = runner.Arm(spec, root=ROOT / "_scratch" / "resume_snapshot_probe",
                     machinery={}, start_snapshot=_h0_snapshot(),
                     ledgers=runner.Ledgers(),
                     guard=runner.BudgetGuard(ordering_cap=1,
                                              per_unit_arm_cap=1,
                                              ledgers=runner.Ledgers()),
                     backend_factory=None, outer_slow_factory=None,
                     offline=True)
    assert arm._method is None
    assert arm.active_snapshot() is arm.start_snapshot
    assert arm.active_snapshot().runtime_bundle_sha


# ---------------------------------------------------------------------------
# B. the replay prediction cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cell():
    return runner.UnitContext(contract.ordering("forward")[0])


@pytest.fixture(scope="module")
def course(tmp_path_factory):
    """A short real course, so the behaviour locks run on real cells."""
    label = "pytest_v11_%s" % tmp_path_factory.mktemp("v11").name
    report = runner.run_course(
        phase="phase_t_forward", ordering_name="forward",
        units=contract.ordering("forward"), run_label=label,
        offline=True, limit=6)
    assert report["status"] == "COMPLETE", report.get("run_fault")
    return report


def _cells(course, arm=None):
    return [row for row in course["cells"]
            if arm is None or row["arm"] == arm]


def _scopes(ctx):
    """Five shapes: empty, everyone, a proper subset, and two different ones."""
    uids = list(ctx.eval_uids)
    return {
        "empty": frozenset(),
        "all": frozenset(uids),
        "proper_subset": frozenset(uids[:7]),
        "scope_a": frozenset(uids[:5]),
        "scope_b": frozenset(uids[5:12]),
    }


def test_the_cache_reproduces_the_uncached_reading_bit_for_bit(cell):
    """Five Scope shapes, exact equality -- not a tolerance."""
    cache = runner.ReplayPredictionCache("A3-online")
    for name, scope in _scopes(cell).items():
        fresh = runner._policy_reading(cell, cell.origin, STEPS, scope)
        cached = cache.reading(cell, cell.origin, STEPS, scope)
        for field in ("treated", "served", "aggregate_gain", "harmed_fraction",
                      "max_single_series_harm", "identity"):
            assert cached[field] == fresh[field], (name, field)
        assert cached["per_series_gain"] == fresh["per_series_gain"], name
        assert cached["mean_smase"] == pytest.approx(fresh["mean_smase"],
                                                     abs=0.0, rel=0.0), name
        assert cached["static_mean_smase"] == pytest.approx(
            fresh["static_mean_smase"], abs=0.0, rel=0.0), name


def test_the_cache_fits_once_and_then_charges_nothing(cell):
    cache = runner.ReplayPredictionCache("A3-online")
    scopes = _scopes(cell)
    cache.reading(cell, cell.origin, STEPS, scopes["all"])
    after_first = cache.physical_fits
    assert after_first > 0
    for scope in scopes.values():
        cache.reading(cell, cell.origin, STEPS, scope)
    assert cache.physical_fits == after_first, "a re-mask refitted the Consumer"
    assert cache.cache_hits == len(scopes)
    assert cache.logical_evaluations == len(scopes) + 1
    assert cache.to_dict()["entries"] == 1


def test_two_arms_never_share_a_cache_entry(cell):
    left = runner.ReplayPredictionCache("A3-online")
    right = runner.ReplayPredictionCache("A3-frozen")
    scope = frozenset(list(cell.eval_uids)[:6])
    a = left.reading(cell, cell.origin, STEPS, scope)
    b = right.reading(cell, cell.origin, STEPS, scope)
    assert a["per_series_gain"] == b["per_series_gain"]     # same arithmetic
    assert left.physical_fits > 0 and right.physical_fits > 0
    assert left.cache_hits == 0 and right.cache_hits == 0   # no shared entry
    assert left.key(cell.unit, cell.origin,
                    runner.forecast_p4._config(cell.origin), STEPS) != \
        right.key(cell.unit, cell.origin,
                  runner.forecast_p4._config(cell.origin), STEPS)


def test_the_cache_key_separates_face_and_program(cell):
    cache = runner.ReplayPredictionCache("A3-online")
    config = runner.forecast_p4._config(cell.origin)
    base = cache.key(cell.unit, cell.origin, config, STEPS)
    other_face = cache.key(cell.unit, cell.origin + 48, config, STEPS)
    other_program = cache.key(cell.unit, cell.origin, config,
                              (("winsorize", {}),))
    assert len({base, other_face, other_program}) == 3


def _degenerate_case():
    """A real (unit, program) whose prepared served context flattens a series.

    Searched rather than constructed: the property under test is that the
    cached path refuses exactly what the uncached path refuses, and a forged
    ``degenerate_uids`` would only test the branch, not the agreement.
    """
    programs = ((("period_median_complete", {}), ("outlier_mad", {})),
                (("winsorize", {}),), (("outlier_iqr", {}),),
                (("hampel_filter", {}),))
    for unit in contract.ordering("forward")[:6]:
        ctx = runner.UnitContext(unit)
        for steps in programs:
            probe = runner.ReplayPredictionCache("probe")
            try:
                probe.reading(ctx, ctx.origin, steps,
                              frozenset(ctx.eval_uids[:1]))
            except runner.UnitFault:
                continue
            entry = list(probe._entries.values())[0]
            if entry.get("degenerate_uids"):
                return ctx, steps, list(entry["degenerate_uids"])
    return None, None, None


def test_a_degenerate_series_is_refused_identically_cached_and_uncached(cell):
    """The one Scope-dependent thing the cache must not smooth over."""
    ctx, steps, degenerate = _degenerate_case()
    if ctx is None:
        pytest.skip("no degenerate served context in the first six units")
    cache = runner.ReplayPredictionCache("A3-online")
    reaching = frozenset(degenerate[:1])
    with pytest.raises(runner.UnitFault) as cached_error:
        cache.reading(ctx, ctx.origin, steps, reaching)
    with pytest.raises(runner.UnitFault) as fresh_error:
        runner._policy_reading(ctx, ctx.origin, steps, reaching)
    assert "DEGENERATE" in str(cached_error.value)
    assert "DEGENERATE" in str(fresh_error.value)
    # A Scope that avoids the degenerate series is legal on both paths, and
    # the readings agree bit for bit.
    avoiding = frozenset(uid for uid in ctx.eval_uids
                         if uid not in set(degenerate))
    if avoiding:
        assert (cache.reading(ctx, ctx.origin, steps, avoiding)
                ["per_series_gain"]
                == runner._policy_reading(ctx, ctx.origin, steps, avoiding)
                ["per_series_gain"])


def test_a_forged_degenerate_entry_is_still_refused(cell):
    """The branch itself, on a cell that has no real degenerate series."""
    cache = runner.ReplayPredictionCache("A3-online")
    cache.reading(cell, cell.origin, STEPS, frozenset(cell.eval_uids[:3]))
    entry = list(cache._entries.values())[0]
    entry["degenerate_uids"] = [cell.eval_uids[0]]
    with pytest.raises(runner.UnitFault) as caught:
        cache.reading(cell, cell.origin, STEPS,
                      frozenset(cell.eval_uids[:2]))
    assert "DEGENERATE" in str(caught.value)
    cache.reading(cell, cell.origin, STEPS, frozenset(cell.eval_uids[1:4]))


def test_the_three_cost_ledgers_are_kept_apart(cell):
    cache = runner.ReplayPredictionCache("A3-online")
    for scope in _scopes(cell).values():
        cache.reading(cell, cell.origin, STEPS, scope)
    payload = cache.to_dict()
    assert payload["physical_fits"] > 0
    assert payload["logical_evaluations"] == 5
    assert payload["cache_hits"] == 4
    assert payload["physical_fits"] < payload["logical_evaluations"] * 3


# ---------------------------------------------------------------------------
# C. future-step budget reservation
# ---------------------------------------------------------------------------

def test_the_reservation_holds_back_one_screen_per_future_step():
    """5 + 10 + 15 + 20 + 25 = 75 cells, converted to fits at the cache cost."""
    cells = [runner.reserve_for_future_steps(k, 5, period=5, fits_per_cell=1)
             for k in range(0, 6)]
    assert cells[0] == 75             # before step 1: all five still to come
    assert cells[1] == 70             # 10 + 15 + 20 + 25
    assert cells[4] == 25             # only step 5 remains
    assert cells[5] == 0              # nothing after the last step
    assert runner.reserve_for_future_steps(1, 0, period=5) == 0
    # The budget is denominated in fits, so the reservation must be too.
    # Subtracting cells from a fit budget reserves a fraction of what was
    # intended and starves exactly the later steps this exists to protect.
    assert runner.CACHE_FITS_PER_CELL == 2
    for k in range(0, 6):
        assert runner.reserve_for_future_steps(k, 5, period=5) == cells[k] * 2


def test_the_reservation_arithmetic_closes_for_every_step_of_a_real_course():
    """The guarantee, checked against the contract's own allowance.

    26 units, five outer steps, 156 fits per online arm.  Every step must be
    able to afford its own first screen after holding back the later ones, and
    the five screens together must fit inside the allowance.
    """
    from evaluation.main_protocol_p4 import hec1_scoreability  # noqa: F401

    units = len(contract.phase_t_units())
    period = int(contract.OUTER_LOOP["period_k_units"])
    steps = units // period
    allowance = int(contract.REPLAY_FITS_SHARE * runner.FITS_PER_SCORED_FACE
                    * runner.SCORED_FACES_PER_CELL * units)
    assert (units, period, steps, allowance) == (26, 5, 5, 156)

    spent = 0
    for k in range(1, steps + 1):
        reserved = runner.reserve_for_future_steps(k, steps, period=period)
        remaining = allowance - spent - reserved
        need = k * period * runner.CACHE_FITS_PER_CELL
        assert remaining >= need, (
            "step %d could not afford its first screen: %d available, %d needed"
            % (k, remaining, need))
        spent += need
    assert spent <= allowance, (spent, allowance)


def test_every_step_with_a_candidate_gets_at_least_one_screen():
    """The guarantee C actually makes, over a synthetic five-step course."""
    ledger = drafts.DraftLedger()
    # The real per-arm allowance for a 26-unit course, not a round number:
    # the reservation is denominated in fits and so must the budget be.
    allowance = int(contract.REPLAY_FITS_SHARE * runner.FITS_PER_SCORED_FACE
                    * runner.SCORED_FACES_PER_CELL
                    * len(contract.phase_t_units()))
    screened = []
    for k in range(1, 6):
        reserved = runner.reserve_for_future_steps(k, 5, period=5)
        remaining = allowance - sum(screened) - reserved
        # Each step sees a brand-new key, so each has a candidate to screen.
        scope = {"scope_type": "serving_series_predicate",
                 "predicate": [{"feature": Z, "op": ">=",
                                "threshold": float(k)}]}
        budget = outer_loop.OuterBudget(replay_fits_remaining=max(0, remaining))
        step = outer_loop.consolidate(
            bank=[_bank_row(1000 + k, GOOD, scope)], ledger=ledger,
            k_index=k, replay=_replay(), budget=budget)
        assert step.drafts_opened, (
            "step %d had a candidate and no screen; the reservation failed" % k)
        screened.append(step.replay_fits)
    assert len(ledger.drafts) == 5


def test_a_candidate_the_budget_could_not_reach_is_recorded_not_dropped():
    ledger = drafts.DraftLedger()
    bank = []
    for index in range(4):
        scope = {"scope_type": "serving_series_predicate",
                 "predicate": [{"feature": Z, "op": ">=",
                                "threshold": float(index + 1)}]}
        bank.append(_bank_row(1176 + index, GOOD, scope))
    budget = outer_loop.OuterBudget(replay_fits_remaining=3)  # one screen only
    step = outer_loop.consolidate(bank=bank, ledger=ledger, k_index=1,
                                  replay=_replay(), budget=budget)
    starved = [row for row in step.candidates
               if row["outcome"] == "REPLAY_FITS_BUDGET_SPENT"]
    assert starved, "no candidate was truncated; the cap did not bind"
    for row in starved:
        assert row["kind"]
        assert row["program_signature"]
        assert row["replay_estimate"] is not None
        assert row["replay_fits_remaining"] is not None
    assert step.replay_fits <= 156


# ---------------------------------------------------------------------------
# P0: the authority must be consulted before the snapshot is committed
# ---------------------------------------------------------------------------

class _Pending:
    """The smallest thing ``handle_feedback_delayed`` will act on."""

    def __init__(self, snapshot):
        self.candidate_snapshot = type("R", (), {"snapshot": snapshot})()


class _Reading:
    def __init__(self, gain):
        self.gain = gain
        self.verification = type("V", (), {"passed": True})()
        self.per_view_gain = (gain,) * 8


def _method_with_pending():
    from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod

    method = TTHAMethod.__new__(TTHAMethod)
    method._snapshot = "SNAPSHOT_BEFORE"
    method._pending_update = {
        "kind": "program",
        "steps": (("outlier_mad", {}),),
        "episode_id": "ep-1",
        "series_uids": tuple("s%d" % i for i in range(8)),
        "consumer_id": "ridge",
        "receipt": _Pending("SNAPSHOT_AFTER"),
    }
    return method


def test_a_refusing_authority_stops_the_snapshot_commit_itself():
    """The P0: the commit used to happen before any external gate was asked."""
    method = _method_with_pending()
    event = method.handle_feedback_delayed(
        lambda steps, _mode: _Reading(0.5),
        episode_id="ep-1",
        authorize=lambda _evidence: False)
    assert event["stage"] == "authority_refused"
    assert event["snapshot_updated"] is False
    # The three things the lock has to prove, at the source.
    assert method._snapshot == "SNAPSHOT_BEFORE", "the snapshot was committed"
    assert method._pending_update is None, "the pending survived a refusal"


def test_an_approving_authority_still_commits():
    method = _method_with_pending()
    event = method.handle_feedback_delayed(
        lambda steps, _mode: _Reading(0.5),
        episode_id="ep-1",
        authorize=lambda _evidence: True)
    assert event["stage"] == "approved"
    assert method._snapshot == "SNAPSHOT_AFTER"


def test_omitting_the_authorizer_reproduces_the_historical_path():
    """Additive: every existing caller must behave exactly as before."""
    method = _method_with_pending()
    event = method.handle_feedback_delayed(
        lambda steps, _mode: _Reading(0.5), episode_id="ep-1")
    assert event["stage"] == "approved"
    assert event["snapshot_updated"] is True
    assert method._snapshot == "SNAPSHOT_AFTER"


def test_the_authorizer_is_not_consulted_when_the_lifecycle_itself_refuses():
    """Order matters: a lifecycle rejection short-circuits before the gate."""
    method = _method_with_pending()
    asked = []
    event = method.handle_feedback_delayed(
        lambda steps, _mode: _Reading(-5.0),      # NEGATIVE: lifecycle refuses
        episode_id="ep-1",
        authorize=lambda evidence: asked.append(evidence) or True)
    assert event["stage"] == "delayed_rejected"
    assert not asked
    assert method._snapshot == "SNAPSHOT_BEFORE"


def test_open_delayed_threads_the_authorizer_through():
    from SelfEvolvingHarnessTS.methods.ttha import online_loop

    import inspect

    signature = online_loop.open_delayed.__code__.co_varnames
    assert "delayed_authorizer" in signature
    source = inspect.getsource(online_loop.open_delayed)
    # Every *call site* inside open_delayed must pass it, or one lifecycle
    # route (slow pending / fast-skill pending / group pending) stays
    # unguarded.  Counting the bare name would also count the prose.
    call_sites = source.count("method.handle_feedback_delayed(")
    assert call_sites == 3, call_sites
    assert source.count("authorize=delayed_authorizer") == call_sites


def test_a_refused_gate_leaves_snapshot_store_and_rights_untouched(course):
    """The behaviour lock, on the real course this time.

    For every cell whose authoritative gate refused: the snapshot's Skill set
    and the Store's active pointer are unchanged across the delayed face, the
    cell did not activate, and nothing was minted.
    """
    refused = [row for row in _cells(course)
               if (row.get("delayed") or {}).get("gate")
               and not row["delayed"]["gate"]["passes"]]
    assert refused, "no gate refused in this course; the lock is vacuous"
    for row in refused:
        state = row.get("authority_state")
        assert state is not None, row["position"]
        assert state["snapshot_unchanged"], (row["position"], state)
        assert state["store_active_unchanged"], (row["position"], state)
        assert row["activated"] is False
        assert not row.get("skills_minted_this_unit")


def test_a_refused_gate_is_followed_by_zero_retrieval_and_zero_deployment(
        course):
    """The next unit must not see the refused Skill.

    The leak's observable signature was exactly this: A3-online retrieved and
    deployed a card the P4 gate had refused, because the snapshot had already
    been committed.
    """
    for arm in course["arms"]:
        rows = sorted(_cells(course, arm), key=lambda r: r["position"])
        for earlier, later in zip(rows, rows[1:]):
            gate = (earlier.get("delayed") or {}).get("gate")
            if not gate or gate["passes"] or earlier.get("activated"):
                continue
            # Nothing entered the Active set, so nothing can be recalled from it.
            assert not earlier.get("skills_minted_this_unit")
            assert later["deployed_via"] != "recalled_skill" or (
                later.get("program_in_active_set_at_start")), (
                    "unit %d recalled a Skill that no authorised activation "
                    "produced" % later["position"])


def test_the_four_classification_rules():
    """sol's rules, each on its own input."""
    moved = {"unchanged": False, "snapshot_unchanged": False,
             "store_active_unchanged": True}
    still = {"unchanged": True, "snapshot_unchanged": True,
             "store_active_unchanged": True}
    approved = {"stage": "approved"}

    # 1. any state leak -> BYPASSED, demotes
    leak = runner.resolve_gate_disagreement(
        {"passes": False, "failed_lines": ["coverage_floor"]}, approved,
        state=moved)
    assert leak["kind"] == runner.AUTHORITY_BYPASSED
    assert leak["demotes_the_ordering"] is True

    # 2. coverage_floor only, full state unchanged -> UPHELD, disclose only
    upheld = runner.resolve_gate_disagreement(
        {"passes": False, "failed_lines": ["coverage_floor"]}, approved,
        state=still)
    assert upheld["kind"] == runner.AUTHORITY_UPHELD
    assert upheld["demotes_the_ordering"] is False

    # 3. a risk line is involved -> demotes, even with state unchanged
    for line in runner.RISK_LINES:
        risky = runner.resolve_gate_disagreement(
            {"passes": False, "failed_lines": [line]}, approved, state=still)
        assert risky["kind"] == runner.AUTHORITY_BYPASSED, line
        assert risky["demotes_the_ordering"] is True, line
    mixed = runner.resolve_gate_disagreement(
        {"passes": False, "failed_lines": ["coverage_floor", "harmed_fraction"]},
        approved, state=still)
    assert mixed["kind"] == runner.AUTHORITY_BYPASSED

    # 4. LOST_ACTIVATION -> counted, never a demotion
    lost = runner.resolve_gate_disagreement(
        {"passes": True, "failed_lines": []}, {"stage": "delayed_rejected"},
        state=still)
    assert lost["kind"] == runner.LOST_ACTIVATION
    assert lost["demotes_the_ordering"] is False


def test_a_state_leak_is_a_breach_even_when_the_gates_agree():
    cell = {"activated": False,
            "authority_state": {"unchanged": False},
            "gate_disagreement": {"may_activate": False, "kind": None}}
    assert runner.classify_authority_breach(cell) == runner.AUTHORITY_BYPASSED


# ---------------------------------------------------------------------------
# cache accounting and starting-Active-set recall
# ---------------------------------------------------------------------------

def test_the_replay_cache_counts_reach_the_course_ledger(course):
    ledgers = course["ledgers"]
    caches = course.get("replay_cache") or {}
    assert ledgers["cache_counts"] == "the replay prediction cache"
    assert ledgers["llm_prompt_cache_enabled"] is False
    expected_hits = sum(row["cache_hits"] for row in caches.values())
    assert ledgers["cache_hits"] == expected_hits
    total = ledgers["cache_hits"] + ledgers["cache_misses"]
    assert total == sum(row["logical_evaluations"] for row in caches.values())
    if total:
        assert ledgers["cache_hit_rate"] is not None


def test_the_dynamic_denominator_on_a_real_nineteen_series_face():
    """sol mechanical gate 4: lock the denominator on real 19-series data.

    A source scan for ``/ 20`` catches a literal; it cannot catch a length
    taken from the wrong roster.  ``readable[200:239]`` holds 39 series, so its
    two faces are 20 and 19, and every fraction computed on the 19-series face
    must divide by 19.  The arithmetic exercised here is the runner's own --
    the same expressions ``_policy_reading`` uses -- run over a real
    ``scoped_evaluate`` reading rather than a fixture.
    """
    import numpy as np

    from evaluation.main_protocol_p4 import run_forecast_p4_performance as p4
    from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped

    uids = runner.block_uids([200, 239])
    assert len(uids) == 39
    cell, _variant = runner.baselines._cell(uids)
    assert (len(cell.support_a), len(cell.support_b)) == (20, 19)

    origin = int(contract.phase_s_units()[-1]["origin"])
    at = p4._cell_at(cell, origin)
    config = p4._config(origin)
    # Face B is the 19-series face: roster("support_b") evaluates support_b.
    roster = at.roster("support_b")
    eval_uids = [row["series_uid"] for row in roster if row["role"] == "eval"]
    assert len(eval_uids) == 19, "face B is not the 19-series face"

    static = scoped.scoped_evaluate(roster, at.values, None, config,
                                    origin=origin)
    base = np.asarray(static["per_view_smase"], dtype=np.float64)
    assert base.size == 19, "the reading did not return 19 per-series values"

    executor = runner._executor(roster, at.values, config)
    steps = (("outlier_mad", {}),)
    if not executor.verify(steps, origin).passed:
        pytest.skip("the window verifier refuses this program on face B")
    treated = frozenset(eval_uids[:6])
    reading = scoped.scoped_evaluate(
        roster, at.values, executor._compiled(steps), config,
        origin=origin, scope=treated)
    gains = base - np.asarray(reading["per_view_smase"], dtype=np.float64)
    material = contract.RISK["material"]

    # The runner's own expressions, over 19 series.
    served = len(eval_uids)
    harmed_fraction = float((gains < -material).mean())
    coverage = round(reading["scope_size"] / served, 4)
    assert gains.size == served == 19
    assert harmed_fraction == pytest.approx(
        float((gains < -material).sum()) / 19)
    # The trap: dividing by 20 would move both numbers.  Compared at the
    # runner's own rounding, so the assertion is about the denominator and not
    # about float formatting.
    assert coverage == round(6 / 19, 4)
    assert coverage != round(6 / 20, 4)
    if (gains < -material).any():
        assert harmed_fraction != pytest.approx(
            float((gains < -material).sum()) / 20)


def test_recall_is_attributed_against_the_active_set_at_unit_start(course):
    """A program re-proposed from scratch is a search, not a recall."""
    for row in _cells(course):
        if row["arm"] == "Static":
            continue
        active_at_start = row.get("active_program_signatures_at_start")
        assert isinstance(active_at_start, dict)
        if row.get("deployed_via") == "recalled_skill":
            assert row["program_in_active_set_at_start"] is True, row["position"]
        if row.get("deployed_via") == "searched_this_unit":
            assert row["program_in_active_set_at_start"] is False, row["position"]


# ---------------------------------------------------------------------------
# contract sync
# ---------------------------------------------------------------------------

def test_p1_needs_a_material_terminal_difference_not_merely_a_positive_one():
    from evaluation.main_protocol_p4 import audit_hec1_readout as readout
    from evaluation.main_protocol_p4 import hec1_scoreability as scoreability

    assert contract.P1_MATERIAL_TERMINAL_DIFFERENCE == pytest.approx(0.115)
    assert contract.P1_MATERIAL_TERMINAL_DIFFERENCE == pytest.approx(
        contract.RISK["material"] * scoreability.SCOREABLE_UNITS)
    # A course ending +0.02 over 23 units must not count toward P1.
    rows = [{"terminal_difference": 0.02}, {"terminal_difference": 0.2},
            {"terminal_difference": 0.3}]
    line = contract.P1_MATERIAL_TERMINAL_DIFFERENCE
    assert sum(1 for r in rows if r["terminal_difference"] >= line) == 2
    assert sum(1 for r in rows if r["terminal_difference"] > 0) == 3
    assert "D_o_material_in_at_least_2_of_3" in readout.build()["criteria"]


def test_the_p1_only_verdict_may_not_be_written_as_the_forbidden_claims():
    permitted = contract.VERDICTS["P1_only_permitted_phrasings"]
    forbidden = contract.VERDICTS["P1_only_forbidden_phrasings"]
    assert "feedback-driven Skill-library evolution" in permitted
    assert "Skill acquisition evolution" in permitted
    assert set(forbidden) == {"Scope-revision evolution",
                              "the complete A5 system",
                              "cross-domain transfer"}
    text = contract.VERDICTS["HEC1_P1_ONLY__RECALL_ACCUMULATION"]
    # The verdict states the three forbidden readings as forbidden, and says
    # why each one is unavailable rather than only naming it.
    assert "forbidden" in text
    assert "**Scope-revision evolution**" in text
    assert "**the complete A5 system**" in text
    assert "cross-domain" in text and "within-dataset" in text


def test_the_validation_search_baseline_is_zero_llm_and_outside_the_harness():
    baseline = contract.VALIDATION_SEARCH_BASELINE
    assert baseline["required"] is True
    assert baseline["llm_calls"] == 0
    assert baseline["enters_the_harness"] is False
    assert "same" in baseline["budget"] and "same" in baseline["risk_gate"]


def test_an_empty_phase_s_may_not_manufacture_a_k0_from_the_diagnostic():
    rule = contract.PHASE_S_EMPTY_AGAIN
    assert rule["k0_stays"] == "empty, recorded A5_TREATMENT_EMPTY"
    assert "exhaustive" in rule["then"]
    assert "not formed by the Harness" in rule["must_not"]
    assert "generate this round's K0" in rule["must_not"]


def test_phase_f_evaluates_all_three_orderings_and_reports_the_macro_average():
    phase_f = contract.PHASE_F
    assert phase_f["headline"] == "the macro-average across the three orderings"
    assert "all three" in phase_f["evaluates"]
    assert any("single ordering" in row for row in phase_f["may_not"])
    assert any("choose which ordering" in row for row in phase_f["may_not"])


def test_the_contract_fails_closed_on_each_synced_value():
    """Every sync item is asserted, not merely stated."""
    import evaluation.main_protocol_p4.hec1_contract as module

    assert module.assert_frozen()["frozen"]
    for attribute, key, bad in (
            ("VALIDATION_SEARCH_BASELINE", "enters_the_harness", True),
            ("VALIDATION_SEARCH_BASELINE", "llm_calls", 4),
            ("PHASE_S_EMPTY_AGAIN", "k0_stays", "whatever the sweep found"),
            ("PHASE_F", "headline", "the best ordering")):
        block = getattr(module, attribute)
        original = block[key]
        block[key] = bad
        try:
            assert not module.assert_frozen()["frozen"], (attribute, key)
        finally:
            block[key] = original
    assert module.assert_frozen()["frozen"]


def test_the_reservation_adds_no_priority_order():
    """The frozen deterministic order must be untouched by the reservation."""
    def order(remaining):
        ledger = drafts.DraftLedger()
        bank = []
        for index in range(3):
            scope = {"scope_type": "serving_series_predicate",
                     "predicate": [{"feature": Z, "op": ">=",
                                    "threshold": float(index + 1)}]}
            bank.append(_bank_row(1176 + index, GOOD, scope))
        step = outer_loop.consolidate(
            bank=bank, ledger=ledger, k_index=1, replay=_replay(),
            budget=outer_loop.OuterBudget(replay_fits_remaining=remaining))
        return [row["program_signature"] + "@" + str(row.get(
            "root_scope_signature")) for row in step.candidates]

    assert order(1000) == order(3), "the budget changed the candidate order"
