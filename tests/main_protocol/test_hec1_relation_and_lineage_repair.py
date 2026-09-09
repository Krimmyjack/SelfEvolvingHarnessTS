"""The four M-R0 wiring repairs, each checked where it actually broke.

Every one of these failed silently in the live course, which is why they are
checked against the production path rather than against a convenient stub:

* the census counted no relation at all for 30 outer steps, because the probe's
  relation was read off an admission verdict that has no such field.  So the
  first test drives ``run_online_round`` -- the real Episode classifier, the
  real admission policy -- and follows the probe it produces into
  ``_bank_rows_from_round`` and ``census``.  Nothing here writes the string
  ``POSITIVE`` by hand; if it appears in a count, the classifier put it there.
* one program's refused deployment was recorded as another program's
  verification face, because the Draft was found by predicate alone.
* a closed lineage could be reopened under a fresh shell with its counters back
  at zero.
* a cell billed one fewer call than it sent, and a cell that ended in a fault
  billed none of them.

0 LLM (``SealedProbeBackend`` answers without a relay) and 0 Consumer fits (the
executor is a stub that returns per-series gains directly).
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

import run_v1_guidance_evolution as gerunner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402

from evaluation.main_protocol_p4 import outer_loop  # noqa: E402
from evaluation.main_protocol_p4 import restricted_draft as drafts  # noqa: E402
from evaluation.main_protocol_p4 import run_hec1 as runner  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha import admission_policy as ap  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha import online_loop  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    SupportReceipt,
    WindowVerification,
)

ORIGIN = 400
OP = "outlier_mad"
ALT = "winsorize"
DOMAIN = "m-r0-repair-test"
N_SERIES = 5
MATERIAL = ap.MATERIAL_THRESHOLD

#: Aggregate over the line and nothing materially harmed -> POSITIVE.
POSITIVE_GAINS = (0.5, 0.5, 0.5, 0.5, 0.5)
#: Aggregate over the line, one series materially harmed but inside the
#: bounded budget -> CONFLICT that is nonetheless admitted.  This is the pair
#: that makes "admitted == POSITIVE" a wrong shortcut.
CONFLICT_GAINS = (0.5, 0.5, 0.5, 0.5, -0.10)

Z = "local_robust_z_peak"
SCOPE_A = {"scope_type": "serving_series_predicate",
           "predicate": [{"feature": Z, "op": ">=", "threshold": 3.0}]}


# ---------------------------------------------------------------------------
# the production path: run_online_round -> bank -> census
# ---------------------------------------------------------------------------

def _values() -> dict[str, np.ndarray]:
    x = np.arange(1024, dtype=np.float64)
    return {"s%d" % i: np.sin(x / (7.0 + i)) + 0.1 * np.sin(x / 3.0) + 5.0 + i
            for i in range(N_SERIES)}


def _verification(op: str) -> WindowVerification:
    result = WindowVerification(
        passed=True, checked_windows=1, window_modified_flags=(True,),
        window_identity_equivalent_flags=(False,))
    result._program_supply_prepared_values = (
        np.asarray([float(sum(ord(ch) for ch in op))]),)
    return result


class _StubExecutor:
    """A Consumer that returns a chosen per-series split.  Zero fits."""

    def __init__(self, gains: tuple[float, ...], *, report_fits: int | None):
        self.gains = gains
        self.report_fits = report_fits

    def verify(self, steps, origin):
        return _verification(str(steps[0][0]) if steps else "identity")

    def evaluate(self, steps, origin):
        op = str(steps[0][0]) if steps else "identity"
        per = self.gains if op == OP else (-0.10,) * N_SERIES
        return SupportReceipt(
            origin=int(origin), verification=_verification(op),
            gain=float(np.mean(per)),
            per_view_gain=[float(v) for v in per],
            behavior_point_count=1,
            consumer_fits=self.report_fits)


def _round(gains, *, report_fits=None, policy=None):
    """One real Fast round on a scripted backend.  Returns its result."""
    ap.install_policy(policy or ap.DEFAULT)
    values = _values()
    series = values["s0"]
    core = TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=(OP,),
                                  force_pool=True),
        LocalPublicToolGateway(series[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), gerunner._h0_snapshot(), ())
    return run_online_round(
        method, _StubExecutor(gains, report_fits=report_fits),
        gerunner._a5_request(series, values, ORIGIN, DOMAIN),
        values, origin=ORIGIN, slow_agent=None, controller=None, store=None,
        card_builder=lambda _episode: {}, round_name="m_r0_repair",
        budget=2, allow_fast_skill=False, allow_slow=False, domain=DOMAIN,
        period=24,
        fast_features=dict(extract_public_features(series[:ORIGIN],
                                                   task_kind="forecast")))


def _ctx():
    return SimpleNamespace(
        unit={"block": "[0:40]", "span": [0, 40], "origin": ORIGIN,
              "exposure": ["SPENT_DEV"]},
        eval_uids=["s%d" % i for i in range(N_SERIES)],
        features={"s%d" % i: {Z: 4.0} for i in range(N_SERIES)})


def _probe_of(result, op=OP):
    return next(row for row in result.actual_probed_programs
                if row.get("kind") == "probe"
                and (row.get("program_steps") or [{}])[0].get("op") == op)


@pytest.fixture(autouse=True)
def _reset_policy():
    yield
    ap.reset_policy()


def test_the_episode_relation_reaches_the_probe_record_not_the_admission_reason():
    """The classifier's own word, on the probe, next to the admission reason."""
    result = _round(POSITIVE_GAINS)
    probe = _probe_of(result)
    episode, _steps = result._episodes[-1]

    # The relation on the probe is the Episode's, verbatim -- not re-derived.
    assert probe["relation"] == episode.relation
    assert probe["relation"] in outer_loop.RELATIONS
    # ... and the admission verdict still has no relation of its own, which is
    # exactly why reading one out of it produced None for the whole course.
    assert "relation" not in probe["admission"]
    assert probe["admission"]["reason"] not in outer_loop.RELATIONS


def test_admitted_is_not_positive_and_the_census_keeps_them_apart():
    """A bounded CONFLICT is admitted.  It must still count as adverse."""
    result = _round(CONFLICT_GAINS, policy=ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20,
        max_single_series_harm=0.30))
    probe = _probe_of(result)
    episode, _steps = result._episodes[-1]

    assert episode.relation == "CONFLICT"
    assert probe["relation"] == "CONFLICT"
    assert probe["admission"]["admitted"] is True          # deployable ...
    assert probe["admission"]["reason"] == "within_risk_budget"

    rows = runner._bank_rows_from_round(_ctx(), result)
    row = next(r for r in rows if r["program_steps"][0]["op"] == OP)
    assert row["relation"] == "CONFLICT"
    assert row["admitted"] is True                          # ... and adverse.
    assert outer_loop._relation(row, MATERIAL) == "CONFLICT"


def test_the_census_counts_the_relations_the_production_path_produced():
    """POSITIVE and CONFLICT, from real rounds, arriving as real counts."""
    positive = _round(POSITIVE_GAINS)
    conflict_a = _round(CONFLICT_GAINS)
    conflict_b = _round(CONFLICT_GAINS)

    ctx = _ctx()
    bank = []
    for index, result in enumerate((positive, conflict_a, conflict_b)):
        cell = SimpleNamespace(
            unit={**ctx.unit, "origin": ORIGIN + 240 * index},
            eval_uids=ctx.eval_uids, features=ctx.features)
        bank.extend(runner._bank_rows_from_round(cell, result))

    groups = outer_loop.census(bank, material=MATERIAL)
    group = next(g for g in groups
                 if g["program_signature"] == "%s({})" % OP)

    assert group["relation_counts"] == {"CONFLICT": 2, "POSITIVE": 1}
    assert group["positive_units"] == 1
    assert group["adverse_units"] == 2

    # The two gates the counts feed.  Before the repair both were unreachable
    # for every group in every step of the live course.
    candidates, _signals = outer_loop.propose_candidates(
        groups, ledger=drafts.DraftLedger(),
        held_lineage_keys=[group["census_key"]])
    assert [c["kind"] for c in candidates] == ["NARROW"]

    candidates, _signals = outer_loop.propose_candidates(
        groups, ledger=drafts.DraftLedger(), held_lineage_keys=[])
    assert [c["kind"] for c in candidates] == ["ADD"]


def test_an_admission_reason_is_no_longer_mistaken_for_a_relation():
    """The exact shape the live bank carried: reason present, relation absent."""
    row = {"relation": None, "admission": "within_risk_budget",
           "per_series_gain": {"a": 0.5, "b": 0.5, "c": -0.10}}
    # Not "WITHIN_RISK_BUDGET": the row falls through to the gain-based
    # definition, which is what the module always meant by these three words.
    assert outer_loop._relation(row, MATERIAL) == "CONFLICT"

    row = {"relation": "harmed_fraction_over_budget",
           "per_series_gain": {"a": 0.5, "b": 0.5, "c": 0.5}}
    assert outer_loop._relation(row, MATERIAL) == "POSITIVE"


def test_the_probe_record_carries_the_fit_count_the_receipt_reported():
    """The receipt's number, on the probe.  What the *cell* is billed is read
    off the executor instead -- see the D4 tests below, which is what lets a
    round that aborts still pay for the probes it ran."""
    reported = _round(POSITIVE_GAINS, report_fits=2)
    silent = _round(POSITIVE_GAINS, report_fits=None)
    assert _probe_of(reported)["consumer_fits"] == 2
    # Not zero: an evaluator that reported nothing leaves the cost unmeasured.
    assert _probe_of(silent)["consumer_fits"] is None

    blob = runner.Ledgers().to_dict()
    assert "support_fits" in blob
    assert "support_fits_unrecorded_probes" in blob


# ---------------------------------------------------------------------------
# one Scope, two programs
# ---------------------------------------------------------------------------

def _restrict(ledger, op, scope, *, origin=ORIGIN, key=None):
    return ledger.restrict(
        program_steps=((op, {}),), root_scope=scope, current_scope=scope,
        origin=origin, delayed_reading={"delayed_origin": origin + 48,
                                        "lines": {"single_series_harm": False}},
        revisions=0, census_key=key)


def test_two_programs_under_one_scope_are_two_drafts():
    ledger = drafts.DraftLedger()
    mad = _restrict(ledger, OP, SCOPE_A)
    win = _restrict(ledger, ALT, SCOPE_A)
    assert mad.draft_id != win.draft_id

    assert ledger.by_scope(SCOPE_A, program_steps=((OP, {}),)) is mad
    assert ledger.by_scope(SCOPE_A, program_steps=((ALT, {}),)) is win


def test_one_programs_refusal_does_not_spend_another_programs_attempts():
    """The live reverse ordering: a winsorize refusal closed an outlier_mad
    Draft on its third face.  Same inputs, program-aware lookup."""
    ledger = drafts.DraftLedger()
    mad = _restrict(ledger, OP, SCOPE_A)
    win = _restrict(ledger, ALT, SCOPE_A)

    gate = {"lines": {"coverage_floor": False, "aggregate": True,
                      "harmed_fraction": True, "single_series_harm": True}}
    found, refusal = runner._draft_for_refused_deployment(
        ledger, steps=((ALT, {}),), scope=SCOPE_A, origin=ORIGIN,
        delayed_origin=ORIGIN + 48, gate=gate)
    assert refusal is None and found is win

    ledger.record_verification(
        found, window=ORIGIN + 48, failed_lines=["coverage_floor"],
        per_series_gain={}, treated_prev=[], treated_now=[],
        material=MATERIAL, reading=gate)

    assert win.verification_attempts == 1
    assert mad.verification_attempts == 0, (
        "the other program's Draft must not have been charged")
    assert mad.closed is None


# ---------------------------------------------------------------------------
# one lineage, one shell
# ---------------------------------------------------------------------------

def test_restrict_refuses_a_second_shell_for_a_live_lineage():
    ledger = drafts.DraftLedger()
    _restrict(ledger, OP, SCOPE_A)
    with pytest.raises(ValueError, match="already has a Draft"):
        _restrict(ledger, OP, SCOPE_A, origin=ORIGIN + 240)


def test_a_closed_lineage_cannot_come_back_with_its_counters_at_zero():
    ledger = drafts.DraftLedger()
    first = _restrict(ledger, OP, SCOPE_A)
    for window in (448, 688, 928):
        ledger.record_verification(
            first, window=window, failed_lines=["single_series_harm"],
            per_series_gain={"a": -0.4}, treated_prev=["a"], treated_now=["a"],
            material=MATERIAL)
    assert first.verification_attempts == drafts.MAX_VERIFICATION_ATTEMPTS
    assert first.closed is not None

    with pytest.raises(ValueError, match="already has a Draft"):
        _restrict(ledger, OP, SCOPE_A, origin=1168)


def test_the_two_creation_entries_share_one_guard():
    """``restrict`` and ``open_restricted`` are one bound, not two."""
    key = outer_loop.census_key("forecast|c|m", ((OP, {}),), SCOPE_A)

    ledger = drafts.DraftLedger()
    _restrict(ledger, OP, SCOPE_A, key=key)
    assert ledger.lineage_keys() == {key}
    with pytest.raises(ValueError):
        ledger.open_restricted(
            program_steps=((OP, {}),), root_scope=SCOPE_A,
            current_scope=SCOPE_A, origin=ORIGIN, census_key=key)

    ledger = drafts.DraftLedger()
    ledger.open_restricted(
        program_steps=((OP, {}),), root_scope=SCOPE_A, current_scope=SCOPE_A,
        origin=ORIGIN, census_key=key)
    with pytest.raises(ValueError):
        _restrict(ledger, OP, SCOPE_A, key=key)


def test_the_runner_records_a_refusal_instead_of_minting_a_second_shell():
    ledger = drafts.DraftLedger()
    first = _restrict(ledger, OP, SCOPE_A)
    ledger.close(first, "EFFECT_NONSTATIONARY")

    gate = {"lines": {"coverage_floor": True, "aggregate": True,
                      "harmed_fraction": False, "single_series_harm": True}}
    draft, refusal = runner._draft_for_refused_deployment(
        ledger, steps=((OP, {}),), scope=SCOPE_A, origin=1168,
        delayed_origin=1216, gate=gate)

    assert draft is None
    assert refusal["draft_id"] == first.draft_id
    assert refusal["closed"] == "EFFECT_NONSTATIONARY"
    assert len(ledger.drafts) == 1, "no second shell was minted"


def test_a_draft_only_lineage_blocks_ADD_without_becoming_a_NARROW_target():
    """The two uses of "held" are separate, and stay separate."""
    ledger = drafts.DraftLedger()
    key = outer_loop.census_key("forecast|pooled-ridge-a1|sMASE",
                                ((OP, {}),), SCOPE_A)
    _restrict(ledger, OP, SCOPE_A, key=key)

    group = {
        "task_consumer_key": "forecast|pooled-ridge-a1|sMASE",
        "program_signature": "%s({})" % OP, "census_key": key,
        "root_scope_signature": outer_loop._root_scope_signature(SCOPE_A),
        "root_scope": SCOPE_A, "program_steps": [{"op": OP, "params": {}}],
        "rows": [], "units": [], "relation_counts": {"POSITIVE": 3,
                                                     "CONFLICT": 3},
        "unit_count": 6, "positive_units": 3, "adverse_units": 3,
        "source_skill_ids": [],
    }

    candidates, _ = outer_loop.propose_candidates(
        [group], ledger=ledger, held_lineage_keys=[])
    assert candidates == [], (
        "a key the ledger already carries is neither a fresh ADD nor, on its "
        "own, an Active card to narrow")

    candidates, _ = outer_loop.propose_candidates(
        [group], ledger=ledger, held_lineage_keys=[key])
    assert [c["kind"] for c in candidates] == ["NARROW"]


# ---------------------------------------------------------------------------
# billing
# ---------------------------------------------------------------------------

class _StubArm:
    """An arm whose backend does not meter itself -- the offline shape."""

    def __init__(self, calls: int) -> None:
        self._calls = calls

    def backend_calls(self) -> int:
        return self._calls

    def backend_billed_calls(self) -> int | None:
        return None


def _guard(cap: int = 500) -> runner.BudgetGuard:
    return runner.BudgetGuard(ordering_cap=cap, per_unit_arm_cap=5,
                              ledgers=runner.Ledgers())


def test_reserve_increments_nothing_so_every_sent_call_is_billed():
    guard = _guard()
    guard.open_cell()
    guard.reserve(kind="fast", where={"unit": "u"})
    assert guard.ledgers.llm_fast == 0 and guard.spent_this_cell == 0

    record: dict = {}
    billed = runner._bill_fast(_StubArm(5), guard, record, 0, aborted=False)
    assert billed == 5
    assert guard.ledgers.llm_fast == 5, "not four; the check reserved nothing"
    assert record["llm_calls_this_cell"] == 5
    assert "llm_calls_billed_after_fault" not in record


def test_a_cell_that_ends_in_a_fault_still_pays_for_what_it_sent():
    guard = _guard()
    guard.open_cell()
    guard.reserve(kind="fast", where={"unit": "u"})
    record: dict = {}
    billed = runner._bill_fast(_StubArm(5), guard, record, 0, aborted=True)
    assert billed == 5
    assert guard.ledgers.llm_fast == 5
    assert record["llm_calls_billed_after_fault"] == 5


def test_every_exit_from_a_cell_bills_before_it_returns():
    """A structural guard on the three exits, since driving the fault paths
    end to end needs a course and therefore Consumer fits."""
    import inspect
    source = inspect.getsource(runner.run_unit_arm)
    body = source.split("guard.open_cell()", 1)[1].split(
        "trace = getattr(arm._method", 1)[0]
    handlers = [line for line in body.splitlines()
                if line.strip().startswith("except ")]
    assert len(handlers) == 2, handlers
    assert body.count("_bill_fast(") == 4, (
        "two except handlers, one run-fault re-raise and the normal path")
    assert "max(0, spent - 1)" not in source


def test_the_outer_backend_bills_a_request_whose_delegate_raised():
    class _Boom:
        calls = 0
        maximum_calls = 2

        def complete(self, request):
            type(self).calls += 1
            raise RuntimeError("relay failed after the request went out")

    guard = _guard()
    metered = runner._MeteredOuterBackend(_Boom(), guard=guard, billable=True)
    with pytest.raises(RuntimeError):
        metered.complete({"messages": []})
    assert guard.ledgers.llm_outer == 1


def test_the_ordering_cap_still_refuses_before_the_backend_and_bills_nothing():
    guard = _guard(cap=1)
    guard.ledgers.llm_fast = 1
    with pytest.raises(runner.RunFault):
        guard.reserve(kind="fast", where={"unit": "u"})
    assert guard.ledgers.llm_total() == 1
    assert guard.blocked[-1]["reason"] == "ORDERING_LLM_CAP"


# ---------------------------------------------------------------------------
# what must not have moved
# ---------------------------------------------------------------------------

def test_the_frozen_arm_still_rebuilds_its_ledger_every_unit():
    import inspect
    source = inspect.getsource(runner.Arm.begin_unit)
    assert "self.draft_ledger = drafts.DraftLedger()" in source
    assert '"dropped_drafts": dropped_drafts' in source
    assert '"reason": "frozen arm rebuilt from its start snapshot"' in source


def test_the_delayed_gate_is_still_the_only_authority():
    import inspect
    source = inspect.getsource(runner.run_unit_arm)
    assert "delayed_authorizer=lambda _evidence: bool(gate[\"passes\"])" in source
    assert 'record["evaluation_face_enters_bank"]' not in source
    assert "evaluation_face_enters_bank" in inspect.getsource(
        runner._evaluate_face)


# ---------------------------------------------------------------------------
# D1: the two creation entries, on every way in
# ---------------------------------------------------------------------------

KEY_A = "forecast|c|m|%s({})@z3" % OP
KEY_B = "forecast|c|m|%s({})@z3-other-name" % OP


def _open_restricted(ledger, op, scope, *, key=None, origin=ORIGIN):
    return ledger.open_restricted(
        program_steps=((op, {}),), root_scope=scope, current_scope=scope,
        origin=origin, census_key=key)


def test_open_restricted_refuses_when_the_first_shell_carried_no_key():
    """Hole 1: restrict without a key, then the outer loop opens one with a
    key.  Nothing matched, so a second shell used to be minted at zero."""
    ledger = drafts.DraftLedger()
    first = _restrict(ledger, OP, SCOPE_A, key=None)
    ledger.close(first, "EFFECT_NONSTATIONARY")
    with pytest.raises(ValueError, match="already has a Draft"):
        _open_restricted(ledger, OP, SCOPE_A, key=KEY_A)
    assert len(ledger.drafts) == 1


def test_open_restricted_refuses_when_the_outer_loop_brings_no_key():
    """Hole 2: a keyed Draft exists and the outer loop arrives with
    census_key None, which the old ``if census_key`` guard skipped."""
    ledger = drafts.DraftLedger()
    _restrict(ledger, OP, SCOPE_A, key=KEY_A)
    with pytest.raises(ValueError, match="already has a Draft"):
        _open_restricted(ledger, OP, SCOPE_A, key=None)
    assert len(ledger.drafts) == 1


def test_open_restricted_refuses_a_different_name_for_the_same_identity():
    """Hole 3: same program, same root Scope, a different key string."""
    ledger = drafts.DraftLedger()
    _restrict(ledger, OP, SCOPE_A, key=KEY_A)
    with pytest.raises(ValueError, match="already has a Draft"):
        _open_restricted(ledger, OP, SCOPE_A, key=KEY_B)
    assert len(ledger.drafts) == 1


def test_a_different_program_under_one_scope_still_gets_its_own_shell():
    """The guard is on identity, not on the predicate: a genuinely different
    program that happens to share a Scope must not be blocked."""
    ledger = drafts.DraftLedger()
    _restrict(ledger, OP, SCOPE_A)
    win = _open_restricted(ledger, ALT, SCOPE_A, key=None)
    assert len(ledger.drafts) == 2
    assert win.program_steps[0][0] == ALT


# ---------------------------------------------------------------------------
# D1b: one predicate, two programs, one root question
# ---------------------------------------------------------------------------

ROOT_WIDE = {"scope_type": "serving_series_predicate",
             "predicate": [{"feature": Z, "op": ">=", "threshold": 1.0}]}
ROOT_OTHER = {"scope_type": "serving_series_predicate",
              "predicate": [{"feature": Z, "op": ">=", "threshold": 2.0}]}


def _revised(ledger, op, root):
    return ledger.restrict(
        program_steps=((op, {}),), root_scope=root, current_scope=SCOPE_A,
        origin=ORIGIN, delayed_reading={"lines": {}}, revisions=1)


def test_root_for_scope_never_answers_with_another_programs_root():
    ledger = drafts.DraftLedger()
    _revised(ledger, OP, ROOT_WIDE)
    _revised(ledger, ALT, ROOT_OTHER)

    # Ambiguous without the program: two roots share this predicate.  The
    # answer is "no root applies", never whichever was found first.
    assert ledger.root_for_scope(SCOPE_A) is None

    # With the program it is exact, and each gets its own.
    assert ledger.root_for_scope(
        SCOPE_A, program_steps=((OP, {}),)) == ROOT_WIDE
    assert ledger.root_for_scope(
        SCOPE_A, program_steps=((ALT, {}),)) == ROOT_OTHER


def test_root_for_scope_is_unchanged_when_only_one_draft_matches():
    ledger = drafts.DraftLedger()
    _revised(ledger, OP, ROOT_WIDE)
    assert ledger.root_for_scope(SCOPE_A) == ROOT_WIDE
    assert runner._VerifiableLedgerView(ledger).root_for_scope(
        SCOPE_A, program_steps=((OP, {}),)) == ROOT_WIDE


def test_two_programs_agreeing_on_one_root_is_not_ambiguous():
    ledger = drafts.DraftLedger()
    _revised(ledger, OP, ROOT_WIDE)
    _revised(ledger, ALT, ROOT_WIDE)
    # The same value whichever Draft it came from, so answering is not a guess.
    assert ledger.root_for_scope(SCOPE_A) == ROOT_WIDE


# ---------------------------------------------------------------------------
# D2: the same observation, twice
# ---------------------------------------------------------------------------

UNIT_1176 = {"block": "[0:40]", "span": [0, 40], "origin": 1176,
             "exposure": ["SPENT_DEV"]}


def _bank_row(origin, relation, gains, series=("a", "b", "c")):
    return {
        "unit": {**UNIT_1176, "origin": origin},
        "task_consumer_key": "forecast|pooled-ridge-a1|sMASE",
        "program_steps": [{"op": OP, "params": {}}],
        "serving_scope": SCOPE_A,
        "relation": relation,
        "features": {uid: {Z: 4.0} for uid in series},
        "per_series_gain": dict(zip(series, gains)),
    }


def test_an_identical_repeat_of_one_observation_is_not_a_second_unit():
    once = outer_loop.census([_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4))],
                             material=MATERIAL)[0]
    twice = outer_loop.census(
        [_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)),
         _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4))], material=MATERIAL)[0]

    assert twice["unit_count"] == once["unit_count"] == 1
    assert twice["adverse_units"] == once["adverse_units"] == 1
    assert twice["relation_counts"] == once["relation_counts"]
    assert twice["bank_rows_seen"] == 2
    assert twice["duplicate_observations_dropped"] == 1
    # ... and the threshold tool is not handed the same series twice.
    assert len(twice["rows"]) == len(once["rows"]) == 3


def test_a_repeat_cannot_reach_the_narrowing_threshold_on_its_own():
    """MIN_ADVERSE_UNITS_FOR_NARROWING is written in units.  One unit probed
    twice is one unit."""
    repeated = [_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4))] * 2
    groups = outer_loop.census(repeated, material=MATERIAL)
    key = groups[0]["census_key"]
    candidates, _ = outer_loop.propose_candidates(
        groups, ledger=drafts.DraftLedger(), held_lineage_keys=[key])
    assert candidates == []

    groups = outer_loop.census(
        repeated + [_bank_row(1896, "CONFLICT", (0.5, 0.5, -0.4))],
        material=MATERIAL)
    candidates, _ = outer_loop.propose_candidates(
        groups, ledger=drafts.DraftLedger(), held_lineage_keys=[key])
    assert [c["kind"] for c in candidates] == ["NARROW"]


def test_two_different_readings_of_one_unit_are_kept_and_flagged():
    """R1 is open, so the census must not pick one of them."""
    group = outer_loop.census(
        [_bank_row(1176, "POSITIVE", (0.5, 0.5, 0.5)),
         _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4))], material=MATERIAL)[0]
    assert group["relation_counts"] == {"CONFLICT": 1, "POSITIVE": 1}
    assert group["duplicate_observations_dropped"] == 0
    assert len(group["units_with_more_than_one_distinct_observation"]) == 1


def test_the_production_path_repeat_does_not_double_the_census():
    """The same round's probes, written into one unit twice."""
    result = _round(CONFLICT_GAINS)
    once = runner._bank_rows_from_round(_ctx(), result)
    assert once, "the round produced no bank row to repeat"
    group_once = next(g for g in outer_loop.census(once, material=MATERIAL)
                      if g["program_signature"] == "%s({})" % OP)
    group_twice = next(
        g for g in outer_loop.census(once + once, material=MATERIAL)
        if g["program_signature"] == "%s({})" % OP)
    assert group_twice["adverse_units"] == group_once["adverse_units"]
    assert group_twice["positive_units"] == group_once["positive_units"]
    assert group_twice["unit_count"] == group_once["unit_count"]
    assert group_twice["duplicate_observations_dropped"] == len(once)
    assert len(group_twice["rows"]) == len(group_once["rows"])


# ---------------------------------------------------------------------------
# D4: the ordering cap, on the real request path
# ---------------------------------------------------------------------------

def _metered_backend(guard, maximum_calls=5):
    from SelfEvolvingHarnessTS.runtime.agent_backend import (
        BudgetedAgentBackend,
    )
    inner = BudgetedAgentBackend(
        sealed.SealedProbeBackend(explore=True, operators=(OP,),
                                  force_pool=True),
        maximum_calls=maximum_calls)
    return runner._MeteredFastBackend(inner, guard=guard, billable=True)


def _run_with(backend, executor):
    values = _values()
    series = values["s0"]
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), gerunner._h0_snapshot(), ())
    return run_online_round(
        method, executor,
        gerunner._a5_request(series, values, ORIGIN, DOMAIN),
        values, origin=ORIGIN, slow_agent=None, controller=None, store=None,
        card_builder=lambda _episode: {}, round_name="m_r0_cap",
        budget=2, allow_fast_skill=False, allow_slow=False, domain=DOMAIN,
        period=24,
        fast_features=dict(extract_public_features(series[:ORIGIN],
                                                   task_kind="forecast")))


def test_with_one_call_of_headroom_exactly_one_request_reaches_the_backend():
    """499 of 500 spent: the cap has to bite at the transport, not after."""
    guard = _guard(cap=500)
    guard.ledgers.llm_fast = 499
    guard.open_cell()
    backend = _metered_backend(guard)

    with pytest.raises(runner.RunFault):
        _run_with(backend, _StubExecutor(POSITIVE_GAINS, report_fits=None))

    assert backend.calls == 1, "a second request went out past the cap"
    assert guard.ledgers.llm_fast == 500
    assert guard.ledgers.llm_total() <= guard.ordering_cap
    assert guard.blocked[-1]["reason"] == "ORDERING_LLM_CAP"


def test_the_meter_and_the_cell_reconciliation_charge_each_call_once():
    guard = _guard(cap=500)
    guard.open_cell()
    backend = _metered_backend(guard)
    arm = SimpleNamespace(backend_calls=lambda: backend.calls,
                          backend_billed_calls=lambda: backend.billed_calls)

    _run_with(backend, _StubExecutor(POSITIVE_GAINS, report_fits=None))
    sent = backend.calls
    assert sent > 0
    assert guard.ledgers.llm_fast == sent, "the meter bills what it sends"

    record = {}
    runner._bill_fast(arm, guard, record, 0, aborted=False, billed_before=0)
    assert guard.ledgers.llm_fast == sent, "reconciliation must not re-charge"
    assert record["llm_calls_this_cell"] == sent
    assert record["llm_calls_billed_by_the_backend_meter"] == sent


# ---------------------------------------------------------------------------
# D4: Support cost survives an abort, and is charged once
# ---------------------------------------------------------------------------

def _real_executor(fits):
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
    x = np.arange(1024, dtype=np.float64)
    values = {"t%d" % index: np.sin(x / (7.0 + index)) + 5.0 + index
              for index in range(4)}
    roster = [{"series_uid": "t0", "role": "train"},
              {"series_uid": "t1", "role": "train"},
              {"series_uid": "t2", "role": "eval"},
              {"series_uid": "t3", "role": "eval"}]

    def evaluate_fn(_roster, _values, compiled, _config, *, origin):
        reading = {"mean_smase": 1.0 if compiled is None else 0.8,
                   "per_view_smase": ([1.0, 1.0] if compiled is None
                                      else [0.9, 0.7]),
                   "behavior_point_count": 3}
        if fits is not None:
            reading["consumer_fits"] = fits
        return reading

    return ScopeExecutor(roster, values,
                         {"anchors": [312, 360], "period": 24},
                         evaluate_fn=evaluate_fn)


def test_support_cost_is_read_off_the_executor_not_off_a_returned_result():
    executor = _real_executor(2)
    ledgers = runner.Ledgers()
    before = runner._executor_cost(executor)

    receipt = executor.evaluate((("outlier_mad", {}),), 500)
    assert receipt.gain is not None and receipt.consumer_fits == 2

    # Billed without touching any round result -- the situation the fault
    # paths are in, where ``result`` was never bound.
    billed = runner._bill_support_fits(ledgers, executor, before)
    assert billed["support_fits_this_cell"] == 2
    assert ledgers.support_fits == 2
    assert billed["baseline_evaluations_this_cell"] == 1
    assert ledgers.baseline_evaluations == 1


def test_the_shared_baseline_is_counted_once_then_as_a_cache_hit():
    executor = _real_executor(2)
    ledgers = runner.Ledgers()

    before = runner._executor_cost(executor)
    executor.evaluate((("outlier_mad", {}),), 500)
    runner._bill_support_fits(ledgers, executor, before)

    before = runner._executor_cost(executor)      # the next arm on this unit
    executor.evaluate((("outlier_mad", {}),), 500)
    second = runner._bill_support_fits(ledgers, executor, before)

    assert second["baseline_evaluations_this_cell"] == 0
    assert second["baseline_cache_hits_this_cell"] == 1
    assert ledgers.baseline_evaluations == 1, "the model was fitted once"
    assert ledgers.baseline_cache_hits == 1
    assert ledgers.support_fits == 4
    blob = ledgers.to_dict()
    assert blob["baseline_evaluations"] == 1
    assert blob["baseline_cache_hits"] == 1


def test_an_evaluator_that_reports_no_fits_is_counted_as_unmeasured():
    executor = _real_executor(None)
    ledgers = runner.Ledgers()
    before = runner._executor_cost(executor)
    executor.evaluate((("outlier_mad", {}),), 500)
    billed = runner._bill_support_fits(ledgers, executor, before)
    assert billed["support_fits_this_cell"] == 0
    assert billed["support_evaluations_without_a_fit_count"] == 1
    assert ledgers.support_fits == 0
    assert ledgers.support_fits_unrecorded == 1


def test_billing_the_same_cell_boundary_twice_adds_nothing():
    executor = _real_executor(2)
    ledgers = runner.Ledgers()
    before = runner._executor_cost(executor)
    executor.evaluate((("outlier_mad", {}),), 500)
    runner._bill_support_fits(ledgers, executor, before)
    after = runner._executor_cost(executor)
    runner._bill_support_fits(ledgers, executor, after)
    assert ledgers.support_fits == 2, "each fit is charged once"


def test_every_exit_bills_both_the_calls_and_the_fits():
    import inspect
    source = inspect.getsource(runner.run_unit_arm)
    body = source.split("guard.open_cell()", 1)[1].split(
        "trace = getattr(arm._method", 1)[0]
    assert body.count("_bill_fast(") == 4
    assert body.count("_bill_support_fits(") == 3, (
        "the two except handlers and the run-fault re-raise")
    assert "result.actual_probed_programs))" not in source, (
        "the Support bill must not depend on a result the fault paths lack")


# ---------------------------------------------------------------------------
# 1. the real preflight: identity in, ambiguity refused
# ---------------------------------------------------------------------------

CLAUSE_2 = {"feature": "missing_fraction", "op": ">=", "threshold": 0.05}
CLAUSE_3 = {"feature": "period_reliability", "op": ">=", "threshold": 0.5}
CLAUSE_4 = {"feature": Z, "op": ">=", "threshold": 6.0}
ROOT_CLAUSE = {"feature": Z, "op": ">=", "threshold": 3.0}


def _predicate(*clauses):
    return {"scope_type": "serving_series_predicate", "predicate": list(clauses)}


#: The initialiser's own predicate.
ROOT_ONLY = _predicate(ROOT_CLAUSE)
#: Root + 2 added clauses: the lifecycle bound of MAX_TOTAL_ADDED_CLAUSES is
#: exactly spent.
AT_THE_BOUND = _predicate(ROOT_CLAUSE, CLAUSE_2, CLAUSE_3)
#: Root + 3.  One clause more than the step rule allows in total, and the only
#: thing that can see that is the root.
PAST_THE_BOUND = _predicate(ROOT_CLAUSE, CLAUSE_2, CLAUSE_3, CLAUSE_4)
#: A genuine first revision: one clause added to the initialiser's own scope.
FIRST_REVISION = _predicate(ROOT_CLAUSE, CLAUSE_4)

#: Two series clear the added z >= 6 clause and three do not, so the proposals
#: above are structurally and semantically legal narrowings.  Only the clause
#: budget stands between them and acceptance.
PF_FEATURES = {
    "s0": {Z: 9.0, "missing_fraction": 0.9, "period_reliability": 0.9},
    "s1": {Z: 9.0, "missing_fraction": 0.9, "period_reliability": 0.9},
    "s2": {Z: 4.0, "missing_fraction": 0.9, "period_reliability": 0.9},
    "s3": {Z: 4.0, "missing_fraction": 0.9, "period_reliability": 0.9},
    "s4": {Z: 4.0, "missing_fraction": 0.9, "period_reliability": 0.9},
}
PF_AVAILABLE = [Z, "missing_fraction", "period_reliability"]


def _real_preflight(ledger):
    """The preflight the runner actually injects into ``run_online_round``."""
    from evaluation.main_protocol_p4 import run_source_line_v3 as v3run
    return v3run._preflight(PF_FEATURES, PF_AVAILABLE,
                            runner._VerifiableLedgerView(ledger))


def _at_the_bound(ledger, op, root):
    """A Draft whose predicate is already two clauses past its initialiser."""
    return ledger.restrict(
        program_steps=((op, {}),), root_scope=root,
        current_scope=AT_THE_BOUND, origin=ORIGIN,
        delayed_reading={"lines": {}}, revisions=2)


def test_a_third_clause_is_refused_when_the_root_is_known():
    ledger = drafts.DraftLedger()
    _at_the_bound(ledger, OP, ROOT_ONLY)
    check = _real_preflight(ledger)

    verdict = check(AT_THE_BOUND, PAST_THE_BOUND, ORIGIN,
                    program_steps=((OP, {}),))
    assert verdict["accepted"] is False
    assert verdict["total_added_since_root"] == 3
    assert verdict["max_total_added_clauses"] == 2
    assert verdict["checks"]["within_lifecycle_clause_budget"] is False
    assert verdict["root_lookup"]["ambiguous"] is False


def test_the_same_proposal_is_accepted_when_no_root_is_supplied():
    """The hole, stated as a fact about ``validate_narrowing``: handed no root
    it never reaches the lifecycle budget, and a fourth clause reads as a
    strict narrowing.  This is why ambiguity must not answer with None."""
    from evaluation.main_protocol_p4 import scope_narrowing_preflight as pf
    verdict = pf.validate_narrowing(
        AT_THE_BOUND, PAST_THE_BOUND, features=PF_FEATURES,
        available_features=PF_AVAILABLE, root=None).to_dict()
    assert verdict["accepted"] is True
    assert verdict["total_added_since_root"] is None
    assert "within_lifecycle_clause_budget" not in verdict["checks"]


def test_an_ambiguous_root_refuses_instead_of_pricing_it_as_a_first_revision():
    """Two programs share this predicate and their roots disagree.  Before,
    the lookup answered None and the clause budget was never checked."""
    ledger = drafts.DraftLedger()
    _at_the_bound(ledger, OP, ROOT_ONLY)
    _at_the_bound(ledger, ALT, ROOT_WIDE)
    check = _real_preflight(ledger)

    verdict = check(AT_THE_BOUND, PAST_THE_BOUND, ORIGIN)
    assert verdict["accepted"] is False, (
        "an unresolvable root must refuse, not fall through to the no-root "
        "path where the clause budget is never checked")
    assert verdict["root_lookup"]["ambiguous"] is True
    assert verdict["root_lookup"]["distinct_roots"] == 2
    assert verdict["root_lookup"]["program_identity_supplied"] is False
    assert "ambiguous" in verdict["reason"]

    # Handing it the identity resolves the ambiguity -- and the answer is
    # still a refusal, now for the reason it should always have been.
    resolved = check(AT_THE_BOUND, PAST_THE_BOUND, ORIGIN,
                     program_steps=((OP, {}),))
    assert resolved["accepted"] is False
    assert resolved["root_lookup"]["ambiguous"] is False
    assert resolved["root_lookup"]["program_identity_supplied"] is True
    assert resolved["checks"]["within_lifecycle_clause_budget"] is False
    assert resolved["total_added_since_root"] == 3


def test_the_no_root_path_is_still_open_for_a_genuine_first_revision():
    """Refusing on ambiguity must not refuse a Draft that really is its own
    root -- that is the documented first-revision case."""
    ledger = drafts.DraftLedger()
    check = _real_preflight(ledger)
    verdict = check(ROOT_ONLY, FIRST_REVISION, ORIGIN,
                    program_steps=((OP, {}),))
    assert verdict["accepted"] is True
    assert verdict["root_lookup"]["matches"] == 0
    assert verdict["root_lookup"]["ambiguous"] is False


def test_the_round_hands_the_preflight_the_program_being_revised():
    """The identity reaches the real call site, not just the ledger API."""
    seen = {}

    def preflight(original, proposed, origin, *, program_steps=None):
        seen["program_steps"] = program_steps
        return {"accepted": True}

    preflight.accepts_program_steps = True
    assert online_loop._call_preflight(
        preflight, ROOT_ONLY, AT_THE_BOUND, ORIGIN, ((OP, {}),)
    ) == {"accepted": True}
    assert seen["program_steps"] == ((OP, {}),)


def test_a_preflight_that_does_not_declare_the_keyword_is_called_as_before():
    """Other lines inject three-argument preflights; they must not break."""
    calls = []

    def legacy(original, proposed, origin):
        calls.append((original, proposed, origin))
        return {"accepted": True}

    assert online_loop._call_preflight(
        legacy, ROOT_ONLY, AT_THE_BOUND, ORIGIN, ((OP, {}),)
    ) == {"accepted": True}
    assert calls == [(ROOT_ONLY, AT_THE_BOUND, ORIGIN)]


def test_the_injected_hec1_preflight_declares_that_it_takes_the_identity():
    ledger = drafts.DraftLedger()
    assert getattr(_real_preflight(ledger), "accepts_program_steps", False)


# ---------------------------------------------------------------------------
# 2. duplicate detection compares the readings, not a rounded summary
# ---------------------------------------------------------------------------

def test_readings_that_differ_below_the_fingerprints_resolution_are_two():
    """-0.3000001 and -0.2999999 round to the same six-decimal behaviour
    fingerprint.  They are not the same observation."""
    near = [_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.3000001)),
            _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.2999999))]

    # The fingerprint really does collapse them -- that is why it must not be
    # what deduplication is decided on.
    assert (outer_loop.behaviour_fingerprint(near[0]["per_series_gain"])
            == outer_loop.behaviour_fingerprint(near[1]["per_series_gain"]))
    assert (outer_loop.exact_reading(near[0]["per_series_gain"])
            != outer_loop.exact_reading(near[1]["per_series_gain"]))

    group = outer_loop.census(near, material=MATERIAL)[0]
    assert group["duplicate_observations_dropped"] == 0
    # Two observations, kept apart ...
    assert group["observations"] == 2
    assert len(group["units_with_more_than_one_distinct_observation"]) == 1
    # ... of one unit, which under R1 casts a single vote because both of its
    # readings say the same thing.
    assert group["unit_count"] == 1
    assert group["unit_votes"] == {"CONFLICT": 1}
    assert group["adverse_units"] == 1
    # Both readings stay in the evidence the threshold tool searches.
    gains = sorted(row["gain"] for row in group["rows"]
                   if row["series"] == "c")
    assert gains == [-0.3000001, -0.2999999]


def test_a_bit_for_bit_repeat_is_still_one_observation():
    same = [_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.3000001)),
            _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.3000001))]
    group = outer_loop.census(same, material=MATERIAL)[0]
    assert group["duplicate_observations_dropped"] == 1
    assert group["unit_count"] == 1
    assert group["adverse_units"] == 1


def test_the_bank_keeps_every_row_it_was_handed():
    """Deduplication is a census view; the raw record is not edited."""
    rows = [_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4))] * 3
    before = [dict(row) for row in rows]
    group = outer_loop.census(rows, material=MATERIAL)[0]
    assert rows == before
    assert group["bank_rows_seen"] == 3
    assert group["duplicate_observations_dropped"] == 2


# ---------------------------------------------------------------------------
# 3. what a Consumer-internal failure costs
# ---------------------------------------------------------------------------

def _failing_executor(*, fail_on):
    """A real ScopeExecutor whose injected Consumer raises when asked to."""
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
    x = np.arange(1024, dtype=np.float64)
    values = {"t%d" % index: np.sin(x / (7.0 + index)) + 5.0 + index
              for index in range(4)}
    roster = [{"series_uid": "t0", "role": "train"},
              {"series_uid": "t1", "role": "train"},
              {"series_uid": "t2", "role": "eval"},
              {"series_uid": "t3", "role": "eval"}]

    def evaluate_fn(_roster, _values, compiled, _config, *, origin):
        which = "baseline" if compiled is None else "candidate"
        if which == fail_on:
            raise RuntimeError("the Consumer fell over part-way through")
        return {"mean_smase": 1.0 if compiled is None else 0.8,
                "per_view_smase": ([1.0, 1.0] if compiled is None
                                   else [0.9, 0.7]),
                "behavior_point_count": 3, "consumer_fits": 2}

    return ScopeExecutor(roster, values,
                         {"anchors": [312, 360], "period": 24},
                         evaluate_fn=evaluate_fn)


def test_a_candidate_that_raises_leaves_its_cost_unmeasured_not_zero():
    executor = _failing_executor(fail_on="candidate")
    ledgers = runner.Ledgers()
    before = runner._executor_cost(executor)

    receipt = executor.evaluate((("outlier_mad", {}),), 500)
    assert receipt.gain is None and "fell over" in receipt.error
    assert receipt.consumer_fits is None

    billed = runner._bill_support_fits(ledgers, executor, before)
    # The reference model did complete before the candidate failed, so its
    # cost is known and recorded ...
    assert billed["baseline_evaluations_this_cell"] == 1
    # ... and the candidate's is not recoverable, so it is counted as such
    # rather than added to the total as a zero.
    assert billed["support_fits_this_cell"] == 0
    assert billed["support_evaluations_with_unmeasured_cost"] == 1
    assert ledgers.support_fits_unmeasured_after_error == 1
    assert ledgers.support_fits == 0

    blob = ledgers.to_dict()
    assert blob["support_fits_unmeasured_after_error"] == 1
    assert "unmeasured_cost_note" in blob


def test_a_reference_evaluation_that_raises_is_not_a_completed_one():
    executor = _failing_executor(fail_on="baseline")
    ledgers = runner.Ledgers()
    before = runner._executor_cost(executor)

    receipt = executor.evaluate((("outlier_mad", {}),), 500)
    assert receipt.gain is None

    billed = runner._bill_support_fits(ledgers, executor, before)
    assert billed["baseline_evaluations_this_cell"] == 0, (
        "an evaluation that raised part-way is not a completed reference read")
    assert billed["baseline_evaluations_failed_this_cell"] == 1
    assert ledgers.baseline_evaluations == 0
    assert ledgers.baseline_evaluations_failed == 1
    assert ledgers.to_dict()["baseline_evaluations_failed"] == 1


def test_the_ledger_does_not_claim_baseline_evaluations_is_a_fit_total():
    blob = runner.Ledgers().to_dict()
    note = blob["baseline_evaluations_is_not_a_fit_total"]
    assert "not" in note and "total of Consumer fits" in note
    assert blob["baseline_note"].startswith("the reference model")
    # The three cost states are named apart, so none of them can be read as
    # another: measured, reported-nothing, and unrecoverable.
    for key in ("support_fits", "support_fits_unrecorded_probes",
                "support_fits_unmeasured_after_error"):
        assert key in blob


def test_the_known_part_of_a_failed_cell_is_still_charged():
    """One good probe, then the Consumer falls over on the next."""
    from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
    x = np.arange(1024, dtype=np.float64)
    values = {"t%d" % index: np.sin(x / (7.0 + index)) + 5.0 + index
              for index in range(4)}
    roster = [{"series_uid": "t0", "role": "train"},
              {"series_uid": "t1", "role": "train"},
              {"series_uid": "t2", "role": "eval"},
              {"series_uid": "t3", "role": "eval"}]
    seen = {"candidates": 0}

    def evaluate_fn(_roster, _values, compiled, _config, *, origin):
        if compiled is not None:
            seen["candidates"] += 1
            if seen["candidates"] == 2:
                raise RuntimeError("the Consumer fell over on the second")
        return {"mean_smase": 1.0 if compiled is None else 0.8,
                "per_view_smase": ([1.0, 1.0] if compiled is None
                                   else [0.9, 0.7]),
                "behavior_point_count": 3, "consumer_fits": 2}

    executor = ScopeExecutor(roster, values,
                             {"anchors": [312, 360], "period": 24},
                             evaluate_fn=evaluate_fn)
    ledgers = runner.Ledgers()
    before = runner._executor_cost(executor)
    executor.evaluate((("outlier_mad", {}),), 500)
    executor.evaluate((("winsorize", {}),), 500)

    billed = runner._bill_support_fits(ledgers, executor, before)
    assert billed["support_fits_this_cell"] == 2, "the probe that finished"
    assert billed["support_evaluations_with_unmeasured_cost"] == 1
    assert ledgers.support_fits == 2
    assert ledgers.support_fits_unmeasured_after_error == 1


# ---------------------------------------------------------------------------
# R1: one unit, one vote, and only on consensus
# ---------------------------------------------------------------------------

def _votes(*rows):
    return outer_loop.census(list(rows), material=MATERIAL)[0]


def test_a_unit_that_agrees_with_itself_votes_once():
    group = _votes(_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)),
                   _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.45)))
    assert group["observations"] == 2, "both readings are on record"
    assert group["unit_count"] == 1
    assert group["unit_votes"] == {"CONFLICT": 1}
    assert group["adverse_units"] == 1
    assert group["units_without_a_consistent_relation"] == []


def test_positive_beside_conflict_in_one_unit_votes_for_neither():
    group = _votes(_bank_row(1176, "POSITIVE", (0.5, 0.5, 0.5)),
                   _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)))
    assert group["observations"] == 2
    assert group["relation_counts"] == {"CONFLICT": 1, "POSITIVE": 1}
    assert group["unit_count"] == 1
    assert group["unit_votes"] == {}
    assert group["positive_units"] == 0
    assert group["adverse_units"] == 0
    assert len(group["units_without_a_consistent_relation"]) == 1


def test_negative_beside_conflict_is_a_disagreement_not_an_adverse_unit():
    """Both are adverse relations.  The rule is consensus, not adversity:
    "the effect is gone" and "the effect still hurts someone" are different
    findings, and choosing between them is not the census's call."""
    group = _votes(_bank_row(1176, "NEGATIVE", (0.0, 0.0, 0.0)),
                   _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)))
    assert group["relation_counts"] == {"CONFLICT": 1, "NEGATIVE": 1}
    assert group["unit_votes"] == {}
    assert group["adverse_units"] == 0, (
        "an adverse-wins rule would say 1 here; it has not been approved")
    assert len(group["units_without_a_consistent_relation"]) == 1


def test_two_units_agreeing_separately_are_two_adverse_votes():
    group = _votes(_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)),
                   _bank_row(1896, "NEGATIVE", (0.0, 0.0, 0.0)))
    assert group["unit_count"] == 2
    assert group["unit_votes"] == {"CONFLICT": 1, "NEGATIVE": 1}
    assert group["adverse_units"] == 2, (
        "different units may vote differently; only a unit at odds with "
        "itself abstains")


def test_a_split_unit_cannot_carry_a_lineage_to_the_narrowing_threshold():
    """Two units, one of them split: one vote, and NARROW needs two."""
    rows = [_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)),
            _bank_row(1896, "CONFLICT", (0.5, 0.5, -0.4)),
            _bank_row(1896, "POSITIVE", (0.5, 0.5, 0.5))]
    groups = outer_loop.census(rows, material=MATERIAL)
    key = groups[0]["census_key"]
    assert groups[0]["adverse_units"] == 1
    assert groups[0]["observations"] == 3
    candidates, _ = outer_loop.propose_candidates(
        groups, ledger=drafts.DraftLedger(), held_lineage_keys=[key])
    assert candidates == []

    # Resolve the split unit into agreement and the threshold is reached.
    agreed = [_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)),
              _bank_row(1896, "CONFLICT", (0.5, 0.5, -0.4)),
              _bank_row(1896, "CONFLICT", (0.5, 0.5, -0.45))]
    groups = outer_loop.census(agreed, material=MATERIAL)
    assert groups[0]["adverse_units"] == 2
    candidates, _ = outer_loop.propose_candidates(
        groups, ledger=drafts.DraftLedger(), held_lineage_keys=[key])
    assert [c["kind"] for c in candidates] == ["NARROW"]


def test_a_split_unit_does_not_vote_for_ADD_either():
    rows = [_bank_row(1176, "POSITIVE", (0.5, 0.5, 0.5)),
            _bank_row(1176, "NEGATIVE", (0.0, 0.0, 0.0))]
    groups = outer_loop.census(rows, material=MATERIAL)
    assert groups[0]["positive_units"] == 0
    candidates, _ = outer_loop.propose_candidates(
        groups, ledger=drafts.DraftLedger(), held_lineage_keys=[])
    assert candidates == [], (
        "no best-of rule: a unit that disagrees with itself supplies no "
        "POSITIVE vote for ADD")


def test_observations_and_votes_are_reported_apart():
    group = _votes(_bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)),
                   _bank_row(1176, "CONFLICT", (0.5, 0.5, -0.4)),
                   _bank_row(1176, "POSITIVE", (0.5, 0.5, 0.5)),
                   _bank_row(1896, "CONFLICT", (0.5, 0.5, -0.4)))
    assert group["bank_rows_seen"] == 4          # what the bank handed over
    assert group["duplicate_observations_dropped"] == 1
    assert group["observations"] == 3            # after exact deduplication
    assert group["unit_count"] == 2              # distinct units
    assert group["unit_votes"] == {"CONFLICT": 1}   # only 1896 agrees
    assert group["adverse_units"] == 1
    assert len(group["units_without_a_consistent_relation"]) == 1


def _scripted_slow():
    """Slow, answering with a feature and a direction.  No relay, no model."""
    calls = []

    def slow(*, candidate, rejected):
        calls.append(candidate)
        return {"scope_clause": {"feature": Z, "op": ">=", "threshold": 99.0},
                "rationale": "test: keep only the spikiest series"}

    slow.calls = calls
    return slow


def _passing_replay():
    """A replay screen that finds every already-processed cell safe."""
    def replay(*, steps, scope):
        return {"cells": [{"unit": {"block": "[0:40]", "origin": 1176},
                           "treated": 5, "aggregate_gain": 0.22,
                           "harmed_fraction": 0.0,
                           "max_single_series_harm": 0.05}],
                "fits": 2}
    replay.estimated_fits_per_candidate = 2
    return replay

# ---------------------------------------------------------------------------
# R2: a narrowing whose lineage already has a Draft
# ---------------------------------------------------------------------------

#: Eight series: five spiky ones the program helps, three flat ones it harms.
#: Aggregate clears the material line while the harmed fraction does not, so
#: the unit reads CONFLICT -- and ``z >= 6`` is a clause the tool can actually
#: calibrate, which is what lets the folded REVISE run to record_revision.
_HELPED = ("a", "b", "c", "d", "e")
_HARMED = ("f", "g", "h")


def _adverse_row(origin):
    gains = {uid: 0.5 for uid in _HELPED}
    gains.update({uid: -0.4 for uid in _HARMED})
    features = {uid: {Z: 9.0} for uid in _HELPED}
    features.update({uid: {Z: 4.0} for uid in _HARMED})
    return {
        "unit": {**UNIT_1176, "origin": origin},
        "task_consumer_key": "forecast|pooled-ridge-a1|sMASE",
        "program_steps": [{"op": OP, "params": {}}],
        "serving_scope": SCOPE_A,
        "relation": "CONFLICT",
        "features": features,
        "per_series_gain": gains,
    }


ADVERSE_ROWS = [_adverse_row(1176), _adverse_row(1896)]


def _narrow_groups():
    return outer_loop.census(ADVERSE_ROWS, material=MATERIAL)


def _lineage_key():
    return _narrow_groups()[0]["census_key"]


def _draft_for_lineage(ledger, state, *, revisions=0, verified=True,
                       closed=None, program=OP):
    """A Draft on the NARROW lineage, put into a chosen lifecycle state."""
    draft = ledger.restrict(
        program_steps=((program, {}),), root_scope=SCOPE_A,
        current_scope=SCOPE_A, origin=ORIGIN,
        delayed_reading={"delayed_origin": ORIGIN + 48, "lines": {}},
        revisions=revisions, census_key=_lineage_key())
    draft.state = state
    if verified:
        draft.history.append({"event": "verification", "window": ORIGIN + 48,
                              "state_after": state})
    if closed:
        ledger.close(draft, closed)
    return draft


def test_without_a_draft_the_narrow_path_is_unchanged():
    ledger = drafts.DraftLedger()
    candidates, _ = outer_loop.propose_candidates(
        _narrow_groups(), ledger=ledger, held_lineage_keys=[_lineage_key()])
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "NARROW"
    assert candidates[0]["needs_clause"] is True
    assert "outcome_preset" not in candidates[0]


def test_a_revisable_draft_turns_the_narrow_into_a_revise_of_that_draft():
    ledger = drafts.DraftLedger()
    draft = _draft_for_lineage(ledger, drafts.REVISABLE)
    before = (draft.revisions, draft.verification_attempts)

    candidates, _ = outer_loop.propose_candidates(
        _narrow_groups(), ledger=ledger, held_lineage_keys=[_lineage_key()])

    assert len(candidates) == 1, "one lineage, one candidate per step"
    candidate = candidates[0]
    assert candidate["kind"] == "REVISE", "not a second NARROW beside it"
    assert candidate["draft_id"] == draft.draft_id
    # From current_scope, with the ancestry kept.
    assert candidate["base_scope"] == draft.current_scope
    assert candidate["root_scope"] == draft.root_scope
    # The adverse evidence that raised the narrowing is carried, not dropped.
    folded = candidate["evidence"]["folded_from_narrow"]
    assert folded["adverse_units"] == 2
    assert folded["unit_votes"] == {"CONFLICT": 2}
    # Proposing changes no counter.
    assert (draft.revisions, draft.verification_attempts) == before


def test_the_folded_revise_reaches_record_revision_and_only_then_counts():
    """Through ``consolidate``: one Slow clause, one revision, no new shell."""
    ledger = drafts.DraftLedger()
    draft = _draft_for_lineage(ledger, drafts.REVISABLE)
    assert draft.revisions == 0

    record = outer_loop.consolidate(
        bank=ADVERSE_ROWS, ledger=ledger, k_index=1,
        slow=_scripted_slow(), replay=_passing_replay(),
        budget=outer_loop.OuterBudget(),
        held_lineage_keys=[_lineage_key()])

    assert [row["kind"] for row in record.candidates] == ["REVISE"]
    assert record.candidates[0]["outcome"] == "DRAFT_REVISED"
    assert record.drafts_revised == [draft.draft_id]
    assert record.drafts_opened == [], "no second shell"
    assert len(ledger.drafts) == 1
    assert draft.revisions == 1, "record_revision adds one, and only there"
    assert draft.verification_attempts == 0, (
        "the later verification is what spends an attempt")
    assert draft.state == drafts.REVISABLE, "state untouched by the fold"


def test_a_flagged_lineage_is_refused_at_propose_without_calling_slow():
    ledger = drafts.DraftLedger()
    draft = _draft_for_lineage(ledger, drafts.FLAGGED)
    slow = _scripted_slow()

    record = outer_loop.consolidate(
        bank=ADVERSE_ROWS, ledger=ledger, k_index=1,
        slow=slow, replay=_passing_replay(),
        budget=outer_loop.OuterBudget(),
        held_lineage_keys=[_lineage_key()])

    assert [row["kind"] for row in record.candidates] == ["NARROW"]
    assert record.candidates[0]["outcome"] == "REVISION_TARGET_FLAGGED"
    assert record.slow_calls == 0, "Slow was never asked"
    assert record.replay_fits == 0, "the replay screen never ran"
    assert record.drafts_opened == [] and record.drafts_revised == []
    assert draft.state == drafts.FLAGGED
    assert (draft.revisions, draft.verification_attempts) == (0, 0)
    assert len(ledger.drafts) == 1


def test_a_closed_lineage_is_refused_at_propose():
    ledger = drafts.DraftLedger()
    _draft_for_lineage(ledger, drafts.REVISABLE, closed="EFFECT_NONSTATIONARY")
    record = outer_loop.consolidate(
        bank=ADVERSE_ROWS, ledger=ledger, k_index=1,
        slow=_scripted_slow(), replay=_passing_replay(),
        budget=outer_loop.OuterBudget(),
        held_lineage_keys=[_lineage_key()])
    assert record.candidates[0]["outcome"] == "LINEAGE_CLOSED"
    assert record.slow_calls == 0
    assert len(ledger.drafts) == 1


def test_a_lineage_with_no_revision_budget_left_is_refused_at_propose():
    ledger = drafts.DraftLedger()
    _draft_for_lineage(ledger, drafts.REVISABLE,
                       revisions=drafts.MAX_REVISIONS)
    record = outer_loop.consolidate(
        bank=ADVERSE_ROWS, ledger=ledger, k_index=1,
        slow=_scripted_slow(), replay=_passing_replay(),
        budget=outer_loop.OuterBudget(),
        held_lineage_keys=[_lineage_key()])
    assert record.candidates[0]["outcome"] == "REVISION_BUDGET_EXHAUSTED"
    assert record.slow_calls == 0


def test_a_lineage_awaiting_verification_of_its_last_revision_is_refused():
    ledger = drafts.DraftLedger()
    draft = _draft_for_lineage(ledger, drafts.REVISABLE, revisions=1,
                               verified=False)
    draft.history.append({"event": "revised_by_outer_loop", "revisions": 1})
    record = outer_loop.consolidate(
        bank=ADVERSE_ROWS, ledger=ledger, k_index=1,
        slow=_scripted_slow(), replay=_passing_replay(),
        budget=outer_loop.OuterBudget(),
        held_lineage_keys=[_lineage_key()])
    assert record.candidates[0]["outcome"] == (
        "AWAITING_VERIFICATION_OF_LAST_REVISION")
    assert record.slow_calls == 0


def test_one_lineage_produces_one_candidate_even_when_both_paths_apply():
    """The Draft loop would propose this lineage on its own.  The NARROW must
    not add a second candidate beside it; its evidence rides along instead."""
    ledger = drafts.DraftLedger()
    draft = ledger.restrict(
        program_steps=((OP, {}),), root_scope=SCOPE_A, current_scope=SCOPE_A,
        origin=ORIGIN,
        delayed_reading={"delayed_origin": ORIGIN + 48, "lines": {}},
        revisions=0, census_key=_lineage_key())
    draft.state = drafts.REVISABLE
    draft.history.append({"event": "verification", "window": ORIGIN + 48,
                          "state_after": drafts.REVISABLE})

    candidates, _ = outer_loop.propose_candidates(
        _narrow_groups(), ledger=ledger, held_lineage_keys=[_lineage_key()])
    assert len(candidates) == 1
    assert candidates[0]["kind"] == "REVISE"
    assert candidates[0]["draft_id"] == draft.draft_id
    ids = [row.get("draft_id") for row in candidates]
    assert len(ids) == len(set(ids)), "deduplicated by lineage"


# ---------------------------------------------------------------------------
# R2 boundary: the early lookup must use the identity the mint uses
# ---------------------------------------------------------------------------

def _keyless_draft(ledger, state, *, program=OP, root=SCOPE_A):
    """A Draft of this lineage that carries no census key.

    That is what ``restrict`` minted before it recorded one, and what the v3
    and smoke paths still mint, so the early lookup meets it in the wild.
    """
    draft = ledger.restrict(
        program_steps=((program, {}),), root_scope=root, current_scope=root,
        origin=ORIGIN, delayed_reading={"delayed_origin": ORIGIN + 48,
                                        "lines": {}},
        revisions=0)
    draft.state = state
    draft.history.append({"event": "verification", "window": ORIGIN + 48,
                          "state_after": state})
    assert draft.census_key is None
    return draft


def test_a_keyless_revisable_draft_still_produces_one_candidate():
    """Keyed on the census key alone, this lineage produced a NARROW *and* a
    REVISE -- two candidates for one lineage in one step."""
    ledger = drafts.DraftLedger()
    draft = _keyless_draft(ledger, drafts.REVISABLE)

    candidates, _ = outer_loop.propose_candidates(
        _narrow_groups(), ledger=ledger, held_lineage_keys=[_lineage_key()])

    assert [row["kind"] for row in candidates] == ["REVISE"]
    assert candidates[0]["draft_id"] == draft.draft_id
    assert candidates[0]["evidence"]["folded_from_narrow"][
        "adverse_units"] == 2


def test_a_keyless_flagged_draft_is_refused_before_slow_and_replay():
    """Keyed on the census key alone this went out as a clause-needing NARROW,
    paid for Slow and a replay screen, and was refused at the mint."""
    ledger = drafts.DraftLedger()
    _keyless_draft(ledger, drafts.FLAGGED)

    record = outer_loop.consolidate(
        bank=ADVERSE_ROWS, ledger=ledger, k_index=1,
        slow=_scripted_slow(), replay=_passing_replay(),
        budget=outer_loop.OuterBudget(),
        held_lineage_keys=[_lineage_key()])

    assert [row["kind"] for row in record.candidates] == ["NARROW"]
    assert record.candidates[0]["outcome"] == "REVISION_TARGET_FLAGGED"
    assert record.slow_calls == 0
    assert record.replay_fits == 0
    assert record.drafts_opened == []
    assert len(ledger.drafts) == 1


def test_a_draft_filed_under_a_different_key_is_still_the_same_lineage():
    ledger = drafts.DraftLedger()
    draft = ledger.restrict(
        program_steps=((OP, {}),), root_scope=SCOPE_A, current_scope=SCOPE_A,
        origin=ORIGIN, delayed_reading={"delayed_origin": ORIGIN + 48,
                                        "lines": {}},
        revisions=0, census_key="some|other|name@for-the-same-identity")
    draft.state = drafts.FLAGGED
    draft.history.append({"event": "verification", "window": ORIGIN + 48})

    candidates, _ = outer_loop.propose_candidates(
        _narrow_groups(), ledger=ledger, held_lineage_keys=[_lineage_key()])
    assert candidates[0]["outcome_preset"] == "REVISION_TARGET_FLAGGED"
    assert candidates[0]["needs_clause"] is False


def test_the_early_lookup_and_the_mint_agree_on_what_a_lineage_is():
    """Whatever the early lookup lets through, the mint must accept."""
    ledger = drafts.DraftLedger()
    _keyless_draft(ledger, drafts.REVISABLE)
    # The mint refuses this identity ...
    with pytest.raises(ValueError, match="already has a Draft"):
        ledger.open_restricted(
            program_steps=((OP, {}),), root_scope=SCOPE_A,
            current_scope=SCOPE_A, origin=ORIGIN, census_key=_lineage_key())
    # ... so the early lookup must not send a NARROW down that road.
    candidates, _ = outer_loop.propose_candidates(
        _narrow_groups(), ledger=ledger, held_lineage_keys=[_lineage_key()])
    assert all(row["kind"] != "NARROW" or row.get("outcome_preset")
               for row in candidates)


def test_a_different_program_is_not_folded_onto_this_lineage():
    """The lookup is on identity, so a Draft of another program under the same
    predicate must not capture this narrowing."""
    ledger = drafts.DraftLedger()
    _keyless_draft(ledger, drafts.FLAGGED, program=ALT)
    candidates, _ = outer_loop.propose_candidates(
        _narrow_groups(), ledger=ledger, held_lineage_keys=[_lineage_key()])
    assert [row["kind"] for row in candidates] == ["NARROW"]
    assert candidates[0].get("outcome_preset") is None
    assert candidates[0]["needs_clause"] is True


# ---------------------------------------------------------------------------
# R1 boundary: the alias merge must not erase a split unit
# ---------------------------------------------------------------------------

def _aliased_row(origin, relation, gains, op):
    row = _bank_row(origin, relation, gains)
    row["program_steps"] = [{"op": op, "params": {}}]
    return row


def test_an_alias_merge_cannot_turn_a_split_unit_into_a_vote():
    """One program reads a unit two ways, so the unit does not vote.  Merging
    an alias that read it one way must not hand that vote back."""
    split_only = [_aliased_row(1176, "CONFLICT", (0.5, 0.5, -0.4), OP),
                  _aliased_row(1176, "POSITIVE", (0.5, 0.5, 0.5), OP)]
    alone = outer_loop.census(split_only, material=MATERIAL)[0]
    assert alone["unit_votes"] == {}
    assert len(alone["units_without_a_consistent_relation"]) == 1

    # "aaa_alias" sorts first, so it -- not outlier_mad -- represents the
    # merged class; the representative's single reading used to be the only
    # one that survived.
    merged = outer_loop.census(
        split_only + [_aliased_row(1176, "POSITIVE", (0.5, 0.5, 0.5),
                                   "aaa_alias")],
        material=MATERIAL)
    assert len(merged) == 1, "the two programs did merge as aliases"
    group = merged[0]
    assert group["aliases"], "and the merge is on record"
    assert group["unit_votes"] == {}, (
        "the unit still disagrees with itself after the merge")
    assert group["positive_units"] == 0
    assert len(group["units_without_a_consistent_relation"]) == 1


def test_every_bank_row_of_a_merged_group_is_accounted_for():
    """seen = observations + dropped, with nothing quietly disappearing."""
    rows = [_aliased_row(1176, "CONFLICT", (0.5, 0.5, -0.4), OP),
            _aliased_row(1176, "POSITIVE", (0.5, 0.5, 0.5), OP),
            _aliased_row(1176, "POSITIVE", (0.5, 0.5, 0.5), "aaa_alias")]
    group = outer_loop.census(rows, material=MATERIAL)[0]
    assert group["bank_rows_seen"] == 3
    assert group["observations"] == 2
    assert group["duplicate_observations_dropped"] == 1
    assert (group["bank_rows_seen"]
            == group["observations"] + group["duplicate_observations_dropped"])


def test_an_alias_reading_a_unit_differently_is_kept_not_dropped():
    """Aliasing is decided on the six-decimal fingerprint, so two programs can
    be aliases while the numbers they recorded differ.  The merge must keep
    both readings; it is not entitled to drop one for arriving second."""
    rows = [_aliased_row(1176, "CONFLICT", (0.5, 0.5, -0.3000001), OP),
            _aliased_row(1176, "CONFLICT", (0.5, 0.5, -0.2999999),
                         "aaa_alias")]
    merged = outer_loop.census(rows, material=MATERIAL)
    assert len(merged) == 1 and merged[0]["aliases"], (
        "the fingerprint really does call these two aliases")
    group = merged[0]
    assert group["observations"] == 2
    assert group["duplicate_observations_dropped"] == 0
    # Both readings reach the evidence the threshold tool searches.
    gains = sorted(row["gain"] for row in group["rows"]
                   if row["series"] == "c")
    assert gains == [-0.3000001, -0.2999999]
    # The unit agrees with itself on the relation, so it still votes once.
    assert group["unit_votes"] == {"CONFLICT": 1}
    assert len(group["units_with_more_than_one_distinct_observation"]) == 1


def test_a_merged_group_can_still_reach_the_narrowing_threshold_honestly():
    """Two units, each unanimous across both aliases: two adverse votes."""
    rows = [_aliased_row(1176, "CONFLICT", (0.5, 0.5, -0.4), OP),
            _aliased_row(1176, "CONFLICT", (0.5, 0.5, -0.4), "aaa_alias"),
            _aliased_row(1896, "CONFLICT", (0.5, 0.5, -0.4), OP)]
    groups = outer_loop.census(rows, material=MATERIAL)
    assert groups[0]["unit_votes"] == {"CONFLICT": 2}
    assert groups[0]["adverse_units"] == 2
    assert groups[0]["duplicate_observations_dropped"] == 1
