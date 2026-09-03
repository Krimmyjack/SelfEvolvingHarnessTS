"""P4U-v3's three changes, driven rather than argued, with no LLM.

Each corresponds to one way v2's null was un-readable:

* the round's Slow call went to whichever refusal Fast proposed first, and at
  origin 2136 that was the probe the pre-registered oracle had already shown
  admits no feasible one-clause narrowing at all;
* half the Slow calls died on manifest protocol before any Scope was judged;
* a Draft that missed one of four delayed lines was destroyed, so the protocol
  never reached the second link of its own evidence chain.

The two-round test is the load-bearing one.  "The controller would keep the
Draft" is exactly the kind of claim that has already been wrong twice in this
protocol, so the restricted Draft is put back in front of a real
``run_online_round`` and the round is made to show that it probed it, ranked it
above the fresh candidate, and spent its one Slow call there.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "evaluation" / "functional", ROOT / "methods" / "ttha"):
    sys.path.insert(0, str(_path))

import run_v1_guidance_evolution as runner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402

from evaluation.main_protocol_p4 import restricted_draft as rd  # noqa: E402
from evaluation.main_protocol_p4 import run_source_line_v3 as v3run  # noqa: E402
from evaluation.main_protocol_p4 import scope_clause_agent as ca  # noqa: E402
from evaluation.main_protocol_p4 import scope_narrowing_preflight as pf  # noqa: E402
from evaluation.main_protocol_p4 import scope_repair_distance as srd  # noqa: E402
from evaluation.main_protocol_p4 import scope_spec as scopes  # noqa: E402
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import (  # noqa: E402
    FaultRouter,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha import admission_policy as ap  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha import online_loop as loop  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.exploration_policy import (  # noqa: E402
    is_supplied_candidate,
)
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

ORIGIN, NEXT_ORIGIN, ALT_OP = 400, 640, "outlier_mad"
UIDS = tuple("s%d" % index for index in range(8))

#: In-scope per-series gains.  s4 and s5 are the damage, and each narrowing
#: below removes exactly one of them: the shape the Source line actually
#: produced, where selecting on defect presence also selects what gets hurt.
BASE = (0.5, 0.5, 0.5, 0.5, -0.9, -0.4, 0.0, 0.0)

FEATURES = {
    "s0": {"gapped": 1.0, "spiky": 0.0, "noisy": 0.0, "flatline": 1.0},
    "s1": {"gapped": 1.0, "spiky": 0.0, "noisy": 0.0, "flatline": 0.0},
    "s2": {"gapped": 1.0, "spiky": 0.0, "noisy": 0.0, "flatline": 0.0},
    "s3": {"gapped": 1.0, "spiky": 0.0, "noisy": 0.0, "flatline": 0.0},
    "s4": {"gapped": 1.0, "spiky": 0.0, "noisy": 1.0, "flatline": 0.0},
    "s5": {"gapped": 1.0, "spiky": 1.0, "noisy": 0.0, "flatline": 0.0},
    "s6": {"gapped": 0.0, "spiky": 0.0, "noisy": 0.0, "flatline": 0.0},
    "s7": {"gapped": 0.0, "spiky": 0.0, "noisy": 0.0, "flatline": 0.0},
}
AVAILABLE = ["flatline", "gapped", "noisy", "spiky"]

ROOT_SCOPE = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.5),))
REVISION_1 = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.5),
                                 scopes.Clause("spiky", "<=", 0.5)))
REVISION_2 = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.5),
                                 scopes.Clause("spiky", "<=", 0.5),
                                 scopes.Clause("noisy", "<=", 0.5)))
REVISION_3 = scopes.ScopeSpec(
    "serving_series_predicate", REVISION_2.clauses + (
        scopes.Clause("flatline", "<=", 0.5),))


def _plain(value):
    """EditManifest freezes its payload, so equality needs a plain form."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _under(spec: scopes.ScopeSpec) -> tuple[float, ...]:
    selected = spec.resolve(FEATURES)
    return tuple(BASE[i] if UIDS[i] in selected else 0.0
                 for i in range(len(UIDS)))


# ============================================================ the distance ===

def test_the_distance_is_the_count_of_exclusions_that_clears_all_four_lines():
    """Root scope needs two dropped; the first revision needs one."""
    assert srd.min_exclusions_to_clear(_under(ROOT_SCOPE)) == 2
    assert srd.min_exclusions_to_clear(_under(REVISION_1)) == 1
    assert srd.min_exclusions_to_clear(_under(REVISION_2)) == 0


def test_a_vector_no_exclusion_can_rescue_is_reported_infeasible():
    """Not "zero exclusions needed" and not a crash -- an explicit None.

    This is the case the oracle bound found at origin 2136 and that v2 spent a
    Slow call on: harm spread wide enough that dropping series hits the
    coverage floor before it clears the lines.
    """
    hopeless = tuple([-0.9] * 6 + [0.5, 0.5])
    assert srd.min_exclusions_to_clear(hopeless) is srd.INFEASIBLE


def test_the_ranking_prefers_the_closer_refusal_over_the_earlier_one():
    """The rule's whole point: probe order must stop deciding this."""
    refusals = [
        {"candidate_id": "probed_first", "aggregate_gain": 0.09,
         "per_series_gain": list(_under(ROOT_SCOPE))},
        {"candidate_id": "probed_second", "aggregate_gain": 0.14,
         "per_series_gain": list(_under(REVISION_1))},
    ]
    choice = srd.select_risk_refusal(refusals)
    assert choice["selected_candidate_id"] == "probed_second"
    assert choice["selected_min_exclusions"] == 1
    assert choice["all_candidates_locally_infeasible"] is False
    assert srd.selector(refusals) == 1


def test_an_all_infeasible_round_still_selects_and_says_so():
    """A round with nothing repairable must be recorded, not silently skipped."""
    refusals = [{"candidate_id": "only", "aggregate_gain": 0.01,
                 "per_series_gain": [-0.9] * 6 + [0.5, 0.5]}]
    choice = srd.select_risk_refusal(refusals)
    assert choice["all_candidates_locally_infeasible"] is True
    assert choice["selected_candidate_id"] == "only"


def test_the_ranking_never_looks_at_a_feature_or_a_series():
    """It ranks faults; it must not be able to hint at the repair."""
    choice = srd.select_risk_refusal([
        {"candidate_id": "c", "aggregate_gain": 0.09,
         "per_series_gain": list(_under(ROOT_SCOPE))}])
    blob = json.dumps(choice)
    for name in AVAILABLE:
        assert name not in blob
    for uid in UIDS:
        assert '"%s"' % uid not in blob


# ====================================================== the lifecycle bound ===

def test_a_second_clause_is_legal_when_counted_against_the_root():
    verdict = pf.validate_narrowing(
        REVISION_1.to_dict(), REVISION_2.to_dict(),
        features=FEATURES, available_features=AVAILABLE,
        root=ROOT_SCOPE.to_dict())
    assert verdict.accepted is True
    assert verdict.total_added_since_root == 2
    assert verdict.checks["keeps_every_root_clause"] is True


def test_a_third_clause_is_refused_even_though_each_step_added_only_one():
    """The bound is on the Skill, not on the step.

    Every individual move here adds one clause and narrows monotonically, so a
    per-step check passes each of them.  Counting against the initialiser's own
    predicate is what stops the chain from being walked past one legal step at
    a time.
    """
    verdict = pf.validate_narrowing(
        REVISION_2.to_dict(), REVISION_3.to_dict(),
        features=FEATURES, available_features=AVAILABLE,
        root=ROOT_SCOPE.to_dict())
    assert verdict.accepted is False
    assert "lifecycle allows 2" in verdict.reason
    assert verdict.checks["within_lifecycle_clause_budget"] is False
    # ... and it would have been accepted without the lifecycle bound, which is
    # what makes the bound the thing doing the work rather than a restatement.
    assert pf.validate_narrowing(
        REVISION_2.to_dict(), REVISION_3.to_dict(),
        features=FEATURES, available_features=AVAILABLE).accepted is True


def test_a_revision_that_abandons_the_root_predicate_is_refused():
    drifted = scopes.ScopeSpec(
        "serving_series_predicate", (scopes.Clause("spiky", "<=", 0.5),
                                     scopes.Clause("noisy", "<=", 0.5)))
    verdict = pf.validate_narrowing(
        REVISION_1.to_dict(), drifted.to_dict(),
        features=FEATURES, available_features=AVAILABLE,
        root=ROOT_SCOPE.to_dict())
    assert verdict.accepted is False


# ========================================================== the clause agent ===

class _ClauseCore:
    """0-LLM stand-in for the agent core: returns one prepared clause."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.inputs: list[dict] = []

    @staticmethod
    def load_stage_schema(name):
        from SelfEvolvingHarnessTS.methods.ttha.schema_contracts import (
            load_stage_schema,
        )
        return load_stage_schema(name)

    def run_stage(self, **kwargs):
        self.calls += 1
        self.inputs.append(dict(kwargs["public_input"]))
        return SimpleNamespace(payload=self.payload, no_proposal_reason=None)


def _card(snapshot_pattern="p4w3-source-line"):
    return {
        "pattern_id": snapshot_pattern,
        "observable_signature": {"task_kind": "forecast"},
        "observable_applicability": {
            "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]},
        "budget": {"min_treated": 5},
        "deployment_visible_features": AVAILABLE,
        "typed_patch_options": [{"patch_id": loop.RISK_REFUSAL_PATCH_ID,
                                 "program_steps": [{"op": ALT_OP, "params": {}}]}],
        "risk_refusal": {
            "reason": "single_series_harm_over_budget",
            "aggregate_gain": 0.0875,
            "serving_scope": ROOT_SCOPE.to_dict(),
            "per_series_gain": list(_under(ROOT_SCOPE)),
        },
    }


def _catalog():
    return [{"surface_id": "skill_library.entries/{skill_id}",
             "operation": "ADD", "allowed_operations": ["ADD"],
             "surface_precondition": {"kind": "ABSENT"},
             "dependency_precondition_shas": {}}]


@pytest.fixture(scope="module")
def snapshot():
    return runner._h0_snapshot()


def test_one_clause_from_slow_becomes_a_schema_valid_manifest(snapshot):
    """The whole point of the change: no manifest field is Slow's to get wrong."""
    core = _ClauseCore({"scope_clause": {"feature": "spiky", "op": "<=",
                                         "threshold": 0.5}})
    agent = ca.ScopeClauseSlowAgent(core)
    manifest = agent.propose_edit(_card(), _catalog(), snapshot)
    assert manifest is not None
    assert manifest.operation.value == "ADD"
    assert manifest.patch_id == loop.RISK_REFUSAL_PATCH_ID
    assert _plain(manifest.new_value["serving_scope"]) == REVISION_1.to_dict()
    assert core.calls == 1


def test_the_runtime_keeps_every_original_clause_rather_than_trusting_slow(snapshot):
    """Appending, not merging: the structural narrowing holds by construction."""
    core = _ClauseCore({"scope_clause": {"feature": "noisy", "op": "<=",
                                         "threshold": 0.5}})
    manifest = ca.ScopeClauseSlowAgent(core).propose_edit(
        _card(), _catalog(), snapshot)
    predicate = _plain(manifest.new_value["serving_scope"])["predicate"]
    assert predicate[0] == {"feature": "gapped", "op": ">=", "threshold": 0.5}
    assert len(predicate) == 2


def test_a_clause_over_an_unobservable_feature_is_refused_not_repaired(snapshot):
    """Refusing is the honest failure; substituting would measure the Runtime."""
    core = _ClauseCore({"scope_clause": {"feature": "clipping_probe",
                                         "op": "<=", "threshold": 0.5}})
    agent = ca.ScopeClauseSlowAgent(core)
    assert agent.propose_edit(_card(), _catalog(), snapshot) is None
    assert "clause_unusable" in str(agent.last_no_proposal_reason)
    assert agent.proposals[-1]["outcome"] == "clause_unusable"


def test_a_non_add_authorization_is_refused_by_the_assembler_too(snapshot):
    """The route pin refuses this upstream; the assembler must not be a way past."""
    core = _ClauseCore({"scope_clause": {"feature": "spiky", "op": "<=",
                                         "threshold": 0.5}})
    catalog = [{**_catalog()[0], "operation": "PATCH",
                "allowed_operations": ["PATCH"]}]
    agent = ca.ScopeClauseSlowAgent(core)
    assert agent.propose_edit(_card(), catalog, snapshot) is None
    assert agent.last_no_proposal_reason == "authorized_operation_is_not_add"
    assert core.calls == 0, "an unauthorized surface must not cost an LLM call"


def test_no_series_identity_is_put_in_front_of_the_clause_writer(snapshot):
    core = _ClauseCore({"scope_clause": {"feature": "spiky", "op": "<=",
                                         "threshold": 0.5}})
    ca.ScopeClauseSlowAgent(core).propose_edit(_card(), _catalog(), snapshot)
    blob = json.dumps(core.inputs[0], default=str)
    for uid in UIDS:
        assert '"%s"' % uid not in blob


# ================================================= the two-round Draft path ===

def _verification(op: str) -> WindowVerification:
    result = WindowVerification(
        passed=True, checked_windows=1, window_modified_flags=(True,),
        window_identity_equivalent_flags=(False,))
    result._program_supply_prepared_values = (
        np.asarray([float(sum(ord(char) for char in op))]),)
    return result


class _ScopedExecutor:
    """Declined series take the raw pipeline, so they score exactly zero."""

    def verify(self, steps, origin):
        return _verification(str(steps[0][0]) if steps else "identity")

    def evaluate(self, steps, origin, serving_scope=None):
        op = str(steps[0][0]) if steps else "identity"
        if op != ALT_OP:
            per = (0.0,) * len(UIDS)
        elif serving_scope is None:
            per = BASE
        else:
            per = tuple(BASE[i] if UIDS[i] in serving_scope else 0.0
                        for i in range(len(UIDS)))
        return SimpleNamespace(
            verification=_verification(op), gain=float(np.mean(per)),
            per_view_gain=tuple(float(value) for value in per),
            behavior_point_count=1)


class _NoopSlow:
    """Records what it was handed and proposes nothing."""

    last_no_proposal_reason = "test_stub"

    def __init__(self) -> None:
        self.cards: list[dict] = []

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        self.cards.append(dict(card))
        return None


def _resolve(spec, _origin):
    return scopes.ScopeSpec.from_dict(dict(spec)).resolve(FEATURES)


def _drive(*, origin, ledger, base_scopes, slow):
    values = {uid: np.sin(np.arange(1400, dtype=np.float64) / (7.0 + index)) + 5.0 + index
              for index, uid in enumerate(UIDS)}
    series = values["s0"]
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20, max_single_series_harm=0.30))
    root = Path(tempfile.mkdtemp())
    store = SnapshotStore(root / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    snap = runner._h0_snapshot()
    method = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
        sealed.SealedProbeBackend(
            explore=True, operators=(ALT_OP,), force_pool=True),
        LocalPublicToolGateway(series[:origin], task_kind="forecast"))),
        snap, ())
    executor = _ScopedExecutor()
    result = loop.run_online_round(
        method, executor,
        runner._a5_request(series, values, origin, "v3-lifecycle"), values,
        origin=origin, slow_agent=slow, controller=controller, store=store,
        card_builder=lambda _episode: {
            "pattern_id": "risk-refusal",
            "failure_family": "workflow_component_negative",
            "observable_signature": {"task_kind": "forecast"},
            "workflow": {"steps": [{"op": ALT_OP, "params": {}}]}},
        round_name="v3_r", budget=12, allow_slow=True, allow_fast_skill=True,
        domain="v3-lifecycle", period=24,
        fast_features=dict(extract_public_features(
            series[:origin], task_kind="forecast")),
        candidate_scopes=v3run._MergedScopes(base_scopes, ledger),
        scope_resolver=_resolve,
        scope_revision_preflight=v3run._preflight(FEATURES, AVAILABLE, ledger),
        program_supply_verifier=executor,
        resupplied_programs=ledger.resupplied_programs(),
        risk_refusal_selector=srd.selector,
        risk_refusal_slow_agent=slow)
    return result


@pytest.fixture(scope="module")
def second_round():
    """Round two, with one Draft restricted after a failed delayed reading."""
    ledger = rd.DraftLedger()
    draft = ledger.restrict(
        program_steps=((ALT_OP, {}),),
        root_scope=ROOT_SCOPE.to_dict(),
        current_scope=REVISION_1.to_dict(),
        origin=ORIGIN,
        delayed_reading={"delayed_origin": ORIGIN + 48,
                         "lines": {"coverage_floor": True, "aggregate": True,
                                   "harmed_fraction": True,
                                   "single_series_harm": False}})
    slow = _NoopSlow()
    base = {candidate: ROOT_SCOPE.to_dict()
            for candidate in (ALT_OP, "cand_" + ALT_OP, "cand_skill_" + ALT_OP)}
    result = _drive(origin=NEXT_ORIGIN, ledger=ledger, base_scopes=base,
                    slow=slow)
    return SimpleNamespace(result=result, ledger=ledger, draft=draft, slow=slow)


def test_the_restricted_draft_is_put_back_in_front_of_the_next_round(second_round):
    assert second_round.result._resupplied_candidate_ids == [
        second_round.draft.draft_id]
    probed = {row["candidate_id"] for row in
              second_round.result.actual_probed_programs}
    assert second_round.draft.draft_id in probed


def test_it_is_probed_under_the_predicate_its_first_revision_reached(second_round):
    """Not the root.  Re-probing under the root would waste the second clause."""
    row = next(row for row in second_round.result.actual_probed_programs
               if row["candidate_id"] == second_round.draft.draft_id)
    assert row["serving_scope"] == REVISION_1.to_dict()
    assert row["per_series_gain"] == list(_under(REVISION_1))


def test_the_round_spends_its_slow_call_on_the_draft_not_the_fresh_candidate(
        second_round):
    """The selection rule doing real work, on a round that contains both."""
    selection = second_round.result._risk_refusal_selection
    assert selection["selected_candidate_id"] == second_round.draft.draft_id
    assert selection["candidates_considered"] >= 2
    assert selection["selector_injected"] is True
    assert second_round.result._slow_trigger == "risk_refusal"


def test_the_card_that_reaches_slow_carries_the_drafts_current_scope(second_round):
    assert second_round.slow.cards, "Slow must have been called"
    refusal = second_round.slow.cards[-1]["risk_refusal"]
    assert refusal["serving_scope"] == REVISION_1.to_dict()
    assert refusal["reason"] == "single_series_harm_over_budget"


def test_the_restricted_draft_never_became_a_skill(second_round):
    """It is a Runner record.  Counting it would corrupt the only claim made."""
    assert second_round.draft.deployable is False
    ledger_blob = second_round.ledger.to_dict()
    assert ledger_blob["open"] == [second_round.draft.draft_id]
    assert second_round.draft.draft_id.startswith(rd.RESUPPLY_PREFIX)
    assert not is_supplied_candidate(second_round.draft.draft_id), (
        "a resupplied Draft must not be counted as a Skill-placed candidate")


def test_a_draft_that_has_spent_both_revisions_is_no_longer_resupplied():
    ledger = rd.DraftLedger()
    draft = ledger.restrict(
        program_steps=((ALT_OP, {}),), root_scope=ROOT_SCOPE.to_dict(),
        current_scope=REVISION_1.to_dict(), origin=ORIGIN,
        delayed_reading={"lines": {"single_series_harm": False}})
    assert ledger.resupplied_programs()
    ledger.record_revision(draft, origin=NEXT_ORIGIN,
                           new_scope=REVISION_2.to_dict(),
                           preflight=None, support=None)
    assert draft.revisions == rd.MAX_REVISIONS
    assert draft.may_revise() is False
    assert ledger.resupplied_programs() == {}


def test_the_ledger_is_what_teaches_the_preflight_the_root():
    ledger = rd.DraftLedger()
    ledger.restrict(program_steps=((ALT_OP, {}),),
                    root_scope=ROOT_SCOPE.to_dict(),
                    current_scope=REVISION_1.to_dict(), origin=ORIGIN,
                    delayed_reading={"lines": {}})
    assert ledger.root_for_scope(REVISION_1.to_dict()) == ROOT_SCOPE.to_dict()
    assert ledger.root_for_scope(ROOT_SCOPE.to_dict()) is None
    check = v3run._preflight(FEATURES, AVAILABLE, ledger)
    assert check(REVISION_2.to_dict(), REVISION_3.to_dict(),
                 NEXT_ORIGIN)["accepted"] is True, (
        "REVISION_2 is not an open Draft's scope, so no root applies")
    assert check(REVISION_1.to_dict(), REVISION_2.to_dict(),
                 NEXT_ORIGIN)["total_added_since_root"] == 2


# ================================================== restriction vs promotion ===
#
# "The controller would not activate early" and "the controller would keep the
# Draft" are the same kind of claim, and the first one has already been wrong
# twice here.  So the promotion path is driven: a delayed reading that misses a
# line must restrict, a delayed reading that passes must still wait for an
# independent origin, and only both together may activate.


class _OriginExecutor(_ScopedExecutor):
    """Per-origin readings, so delayed and re-encounter can genuinely differ."""

    def __init__(self, table):
        self.table = dict(table)

    def evaluate(self, steps, origin, serving_scope=None):
        per_full = self.table.get(int(origin), BASE)
        op = str(steps[0][0]) if steps else "identity"
        if serving_scope is None:
            per = per_full
        else:
            per = tuple(per_full[i] if UIDS[i] in serving_scope else 0.0
                        for i in range(len(UIDS)))
        return SimpleNamespace(
            verification=_verification(op), gain=float(np.mean(per)),
            per_view_gain=tuple(float(value) for value in per),
            behavior_point_count=1)


class _StubLoop:
    def __init__(self) -> None:
        self.opened: list[int] = []
        self.activated = 0

    def open_delayed(self, result, executor, *, delayed_origin, store,
                     scope_resolver):
        self.opened.append(int(delayed_origin))
        result._delayed_event = {"stage": "stub"}

    def activate_approved(self, result, store):
        self.activated += 1
        return True


#: Clean on every series in scope: clears all four lines.
CLEAN = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0)
#: One series over the single-series budget, and that series is *inside*
#: REVISION_1 (s5 is not -- it is the one the first clause already excluded).
#: Three lines of four: the shape the live v2 run produced at origin 2136.
TAIL_HEAVY = (0.5, 0.5, 0.5, 0.5, -0.92, 0.5, 0.0, 0.0)

ORIGINS = [ORIGIN, NEXT_ORIGIN]


def _result(scope, *, selected="cand_" + ALT_OP, refusal_scope=None):
    return SimpleNamespace(
        _winner_steps=((ALT_OP, {}),),
        _winner_serving_scope=scope.to_dict(),
        _risk_refusal_selection={"selected_candidate_id": selected,
                                 "selected_probe_index": 0},
        risk_refusals=[{"serving_scope": (refusal_scope or ROOT_SCOPE).to_dict()}],
        _scope_revision_preflight={"accepted": True},
        _slow_event={"stage": "pending"},
        _delayed_event=None,
        delayed_serving_series=None,
        delayed_scope_reresolved=None,
        _method=SimpleNamespace(_active_snapshot=lambda: None))


def _promote(monkeypatch, *, table, scope, ledger, index=0, result=None):
    executor = _OriginExecutor(table)
    ctx = {"executor": executor, "resolve": _resolve, "origin": ORIGINS[index]}
    monkeypatch.setattr(v3run, "_context",
                        lambda m, cell, variant, origin: {
                            "executor": executor, "resolve": _resolve,
                            "origin": int(origin)})
    stub = _StubLoop()
    out = v3run._promote(
        m=None, cell=None, variant=None, loop=stub, store=None, ledger=ledger,
        ctx=ctx, result=result or _result(scope), origin=ORIGINS[index],
        origins=ORIGINS, index=index, activated=[])
    return out, stub


def test_a_delayed_miss_restricts_the_draft_instead_of_destroying_it(monkeypatch):
    ledger = rd.DraftLedger()
    out, stub = _promote(monkeypatch,
                         table={ORIGIN + 48: TAIL_HEAVY},
                         scope=REVISION_1, ledger=ledger)
    assert out["delayed_gate"]["failed_lines"] == ["single_series_harm"]
    assert out["activated"] is False
    assert stub.activated == 0
    assert out["restriction"] == "restricted_for_one_more_revision"
    assert len(ledger.open_drafts()) == 1
    draft = ledger.open_drafts()[0]
    # The root is the predicate the initialiser wrote, not the one that failed:
    # the second clause has to be counted against it, not against this.
    assert draft.root_scope == ROOT_SCOPE.to_dict()
    assert draft.current_scope == REVISION_1.to_dict()
    assert draft.revisions == 1


def test_a_second_delayed_miss_closes_the_draft_and_the_version(monkeypatch):
    ledger = rd.DraftLedger()
    draft = ledger.restrict(
        program_steps=((ALT_OP, {}),), root_scope=ROOT_SCOPE.to_dict(),
        current_scope=REVISION_1.to_dict(), origin=ORIGIN,
        delayed_reading={"lines": {"single_series_harm": False}})
    result = _result(REVISION_2, selected=draft.draft_id)
    out, stub = _promote(monkeypatch, table={NEXT_ORIGIN + 48: TAIL_HEAVY},
                         scope=REVISION_2, ledger=ledger, index=1,
                         result=result)
    assert out["activated"] is False
    assert out["restriction"] == "closed_after_second_revision"
    assert draft.closed == "second_revision_failed_delayed_gate"
    assert draft.revisions == 2
    assert ledger.resupplied_programs() == {}, (
        "a closed Draft must not come back for a third revision")


def test_a_delayed_pass_still_waits_for_an_independent_origin(monkeypatch):
    """The delayed window is where the revision was taken; it cannot self-certify."""
    ledger = rd.DraftLedger()
    out, stub = _promote(
        monkeypatch,
        table={ORIGIN + 48: CLEAN, NEXT_ORIGIN: TAIL_HEAVY},
        scope=REVISION_1, ledger=ledger)
    assert out["delayed_gate"]["passes"] is True
    assert out["re_encounter_gate"]["passes"] is False
    assert out["activated"] is False
    assert stub.activated == 0, "activation must not happen on the delayed read"


def test_only_both_readings_together_activate(monkeypatch):
    ledger = rd.DraftLedger()
    out, stub = _promote(
        monkeypatch,
        table={ORIGIN + 48: CLEAN, NEXT_ORIGIN: CLEAN},
        scope=REVISION_1, ledger=ledger)
    assert out["delayed_gate"]["passes"] is True
    assert out["re_encounter_gate"]["passes"] is True
    assert out["re_encounter_gate"]["read_origin"] == NEXT_ORIGIN
    assert out["activated"] is True
    assert stub.activated == 1
    assert ledger.open_drafts() == []


def test_the_last_origin_cannot_supply_its_own_re_encounter(monkeypatch):
    """Recorded as a geometry limit, never as a pass by default."""
    ledger = rd.DraftLedger()
    out, stub = _promote(monkeypatch, table={NEXT_ORIGIN + 48: CLEAN},
                         scope=REVISION_1, ledger=ledger, index=1)
    assert out["delayed_gate"]["passes"] is True
    assert out["re_encounter_gate"]["passes"] is False
    assert out["re_encounter_gate"]["failed_lines"] == [
        "no_independent_origin_available"]
    assert out["activated"] is False


def test_a_draft_restricted_at_the_last_origin_is_closed_not_left_open(monkeypatch):
    ledger = rd.DraftLedger()
    out, _ = _promote(monkeypatch, table={NEXT_ORIGIN + 48: TAIL_HEAVY},
                      scope=REVISION_1, ledger=ledger, index=1)
    assert out["restriction"] == "restricted_but_no_next_origin"
    assert ledger.drafts[0].closed == "out_of_origins"
    assert ledger.open_drafts() == []


def test_both_readings_use_one_definition_of_passing(monkeypatch):
    """Two notions of "passed" sharing a name is how this protocol has failed."""
    ledger = rd.DraftLedger()
    out, _ = _promote(monkeypatch,
                      table={ORIGIN + 48: CLEAN, NEXT_ORIGIN: CLEAN},
                      scope=REVISION_1, ledger=ledger)
    delayed, reenc = out["delayed_gate"], out["re_encounter_gate"]
    assert delayed["thresholds"] == reenc["thresholds"]
    assert sorted(delayed["lines"]) == sorted(reenc["lines"])
    assert delayed["scope_reresolved"] is reenc["scope_reresolved"] is True


def test_a_draft_that_is_simply_admitted_later_is_found_and_closed(monkeypatch):
    """The live path that the id-only lookup missed.

    At origin 2376 the Draft restricted at 2136 was not refused again -- it was
    re-probed under its own revised predicate and admitted outright.  With no
    refusal there is no selection, so a lookup keyed on the selected refusal
    found nothing, the Draft stayed open after it had already become a Skill,
    and it was handed a further revision six LLM calls later.
    """
    ledger = rd.DraftLedger()
    draft = ledger.restrict(
        program_steps=((ALT_OP, {}),), root_scope=ROOT_SCOPE.to_dict(),
        current_scope=REVISION_1.to_dict(), origin=ORIGIN,
        delayed_reading={"lines": {"single_series_harm": False}})
    # no refusal this round: the Draft was admitted, so nothing was selected
    result = _result(REVISION_1, selected="")
    out, stub = _promote(
        monkeypatch, table={ORIGIN + 48: CLEAN, NEXT_ORIGIN: CLEAN},
        scope=REVISION_1, ledger=ledger, result=result)
    assert out["activated"] is True
    assert draft.closed == "promoted_to_skill"
    assert ledger.open_drafts() == [], (
        "a Draft that became a Skill must not still be awaiting a revision")
    assert ledger.resupplied_programs() == {}


def test_by_scope_ignores_closed_drafts_and_frozen_shapes():
    from types import MappingProxyType as mp
    ledger = rd.DraftLedger()
    draft = ledger.restrict(
        program_steps=((ALT_OP, {}),), root_scope=ROOT_SCOPE.to_dict(),
        current_scope=REVISION_1.to_dict(), origin=ORIGIN,
        delayed_reading={"lines": {}})
    # the shape a scope has after coming back out of an applied manifest
    frozen = {"scope_type": "serving_series_predicate",
              "predicate": tuple(mp(dict(c))
                                 for c in REVISION_1.to_dict()["predicate"])}
    assert ledger.by_scope(frozen) is draft, (
        "a frozen predicate must still match the plain one it equals")
    assert ledger.by_scope(REVISION_2.to_dict()) is None
    ledger.close(draft, "done")
    assert ledger.by_scope(REVISION_1.to_dict()) is None

