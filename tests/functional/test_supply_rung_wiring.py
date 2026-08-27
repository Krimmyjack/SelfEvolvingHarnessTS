"""W-1: the supply rung of the permission ladder, wired in ``methods/ttha``.

``ordering_card.py`` has carried four authority fields since E1, and
``supplies_candidates`` was the one nothing ever read.  PS-1b froze that as
TEXT_RUNG_INERT; PS-2 then supplied a frozen program through the existing
``_skill_frozen_candidates`` channel and measured two breaks
(``artifacts/functional/e2/ps2_mechanical_supply.json``):

* in 4 of 12 runs the injected program never reached the pool.  The persisted
  records show why: ``pool == ["identity"]`` with ``proposal_count=0``,
  ``chosen=""`` and 2 LLM calls -- the ``_trace`` fallback for
  ``pool is None``.  The propose stage raised, the outer handler took the
  round, and the merge that materialises the card's program never ran because
  it lives *downstream* of a successful propose stage;
* runs 9 and 12 put the inject in the pool, probed it, and recorded Support
  +0.6364 / +0.6000 POSITIVE with delayed +0.30 and LOCAL_ACTIVE -- and still
  deployed identity, because a ``cand_skill_*`` winner is routed to
  ``deployed_existing_skill``, a branch ``open_delayed`` never adjudicated, so
  ``approved_skill_id`` stayed None and every ledger incumbent rule keys on it.

Both are same-rights defects: the supplied candidate had *fewer* rights than an
agent-authored one, not more.  Zero LLM here; no threshold, no authorization
policy and no permission class is asserted differently than before.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_v1_guidance_evolution as gerunner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha import online_loop  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentProtocolError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
    _supplies_candidates,
    _supply_rung_candidates,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor,
)

H0 = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
DOMAIN = "supply_rung_test"
ORIGIN = 400
SUPPLY_OP = "hampel_filter"
SUPPLY_SKILL_ID = "w1_supply_rung_v1"
AGENT_OPS = ("winsorize", "outlier_mad")


# ------------------------------------------------------------------ fixtures
def _series():
    t = np.arange(1024, dtype=np.float64)
    return np.sin(t / 7.0) + 0.1 * np.sin(t / 3.0) + 5.0


def _neutral_eval(roster, values, compiled, config, *, origin):
    """Every candidate neutral: these checks are about wiring, not gain."""
    return {"mean_smase": 1.0, "per_view_smase": [1.0],
            "behavior_point_count": 10}


def _supply_card(*, skill_id=SUPPLY_SKILL_ID, supplies=True,
                 requires_target_support=True):
    """A card on the supply rung.  Who may hold this flag is decided outside
    ``methods``; this is only a card that already holds it."""
    return {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id,
        "skill_kind": "capability",
        "revision": 1,
        "body": (
            "Source hypothesis for this Scope. Frozen program steps: "
            + json.dumps([{"op": SUPPLY_OP, "params": {}}])),
        "observable_applicability": {
            "feature": "task_kind", "op": "==", "value": "forecast"},
        "allowed_tools": [],
        "risk_guards": {
            "card_kind": "w1_supply_test",
            "requires_target_support": bool(requires_target_support),
            "authority": {
                "reorders_supplied_candidates": False,
                "supplies_candidates": bool(supplies),
                "suppresses_operators": False,
                "grants_execution": False,
            },
        },
    }


def _snapshot_with(card, tmp_path):
    store = SnapshotStore(tmp_path / "store")
    h0 = gerunner._h0_snapshot()
    parent = store.materialize(h0)
    fork = store.fork(parent, edit_id="install_%s" % card["skill_id"])
    learned = fork / "skills" / "learned"
    learned.mkdir(parents=True, exist_ok=True)
    (learned / ".gitkeep").unlink(missing_ok=True)
    (learned / ("%s.json" % card["skill_id"])).write_text(
        json.dumps(card, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8")
    snapshot = compile_snapshot(fork, verify_lock=False)
    store.materialize(snapshot, parent_sha=h0.runtime_bundle_sha)
    store.set_active(snapshot.runtime_bundle_sha)
    return store, snapshot


class _ProposeFailsBackend(sealed.SealedProbeBackend):
    """The PS-2 failure shape: the propose stage names an illegal operator.

    ``_compile_candidates`` raises ``AgentProtocolError`` on it, which is the
    same exception class the four lost PS-2 runs hit.
    """

    def complete(self, request):
        if getattr(request, "stage", "") == "propose":
            from SelfEvolvingHarnessTS.runtime.agent_backend import (
                AgentResponse,
            )
            payload = {
                "candidates": [{
                    "candidate_id": "illegal",
                    "steps": [{"op": "not_a_real_operator", "params": {}}],
                }],
            }
            return AgentResponse(json.dumps(payload))
        return super().complete(request)


class _InspectFailsBackend(sealed.SealedProbeBackend):
    """G-3 Part 0: the inspect stage fails.

    This is the shape the ps2p production runs left behind: two LLM calls,
    empty chosen, no agent program.  It sits one stage upstream of the W-1
    repair, so before Part 0 the supplied program died here too.
    """

    def complete(self, request):
        if getattr(request, "stage", "") == "inspect":
            from SelfEvolvingHarnessTS.runtime.agent_backend import (
                AgentResponse,
            )
            # Schema-valid envelope, region fractions outside [0,1]: the
            # inspect post-validator refuses it on both attempts.
            return AgentResponse(json.dumps({
                "inspected_region_fractions": [[1.5, 2.5]],
                "requested_public_tools": [],
                "uncertainty": "high",
            }))
        return super().complete(request)


def _round(snapshot, *, backend_cls=sealed.SealedProbeBackend,
           evaluate_fn=_neutral_eval, budget=2, store=None,
           controller=None, allow_fast_skill=False):
    values = {"s0": _series()}
    series0 = values["s0"]
    backend = backend_cls(explore=True, operators=AGENT_OPS,
                          max_propose_candidates=len(AGENT_OPS),
                          force_pool=True)
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), snapshot, ())
    executor = ScopeExecutor(
        [{"series_uid": "s0", "role": "train"},
         {"series_uid": "s0", "role": "eval"}],
        values, {"anchors": []}, evaluate_fn=evaluate_fn)
    result = online_loop.run_online_round(
        method, executor,
        gerunner._a5_request(series0, values, ORIGIN, DOMAIN),
        values, origin=ORIGIN, slow_agent=None, controller=controller,
        store=store, card_builder=gerunner._a5v2_card,
        round_name="w1_supply", budget=budget, allow_slow=False,
        domain=DOMAIN, period=24,
        fast_features=dict(extract_public_features(
            series0[:ORIGIN], task_kind="forecast")),
        allow_fast_skill=allow_fast_skill)
    return method, result


def _supply_cand_id(skill_id=SUPPLY_SKILL_ID):
    return "cand_skill_%s" % skill_id


def _pool(result):
    return list(getattr(result._method.last_trace, "candidate_ids", ()) or ())


# =========================================================== (a) decoupling
def test_a_the_supply_survives_a_propose_stage_that_raises(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    method, result = _round(snapshot, backend_cls=_ProposeFailsBackend)

    pool = _pool(result)
    assert _supply_cand_id() in pool, pool
    # The agent contributed nothing: the propose stage never reached
    # ``stages``, which is how the failure stays visible.
    assert len(method.last_trace.public_observation_ids) >= 1
    assert not [cid for cid in pool
                if cid not in ("identity", _supply_cand_id())]
    # ... and the supplied candidate is the one that got the Support trial.
    probed = [row["candidate_id"] for row in result.actual_probed_programs]
    assert probed == [_supply_cand_id()], probed


def test_a_the_same_failure_without_the_flag_still_fails(tmp_path):
    """(e) The flag is what revived: an identical card without it changes
    nothing, so the repair reads the permission rather than the shape."""
    _store, snapshot = _snapshot_with(
        _supply_card(skill_id="w1_no_supply_v1", supplies=False), tmp_path)
    _method, result = _round(snapshot, backend_cls=_ProposeFailsBackend)
    pool = _pool(result)
    assert "cand_skill_w1_no_supply_v1" not in pool, pool
    assert result.actual_probed_programs == []


def test_a_the_supply_survives_an_inspect_stage_that_raises(tmp_path):
    """G-3 Part 0.  With no region recorded the scope check is not asserted
    (``verify_candidate`` computes ``outside`` only when regions exist), so
    this needs no degraded full-window decision."""
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    method, result = _round(snapshot, backend_cls=_InspectFailsBackend)

    pool = _pool(result)
    assert _supply_cand_id() in pool, pool
    assert method.last_trace.inspected_regions == ()
    probed = [row["candidate_id"] for row in result.actual_probed_programs]
    assert probed == [_supply_cand_id()], probed


def test_a_an_inspect_failure_without_the_flag_still_fails(tmp_path):
    _store, snapshot = _snapshot_with(
        _supply_card(skill_id="w1_no_supply_v1", supplies=False), tmp_path)
    _method, result = _round(snapshot, backend_cls=_InspectFailsBackend)
    assert "cand_skill_w1_no_supply_v1" not in _pool(result)
    assert result.actual_probed_programs == []


def test_e_the_reader_is_the_authority_field_not_the_frozen_program(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    features = dict(extract_public_features(
        _series()[:ORIGIN], task_kind="forecast"))
    view = resolve_harness_view(snapshot, features, role="fast")
    card = next(s for s in view.skills if s.skill_id == SUPPLY_SKILL_ID)
    assert _supplies_candidates(card) is True
    assert [str(c.source) for c in _supply_rung_candidates(view, features)] \
        == ["skill:%s" % SUPPLY_SKILL_ID]

    _store2, plain = _snapshot_with(
        _supply_card(skill_id="w1_no_supply_v1", supplies=False),
        tmp_path / "b")
    plain_view = resolve_harness_view(plain, features, role="fast")
    other = next(s for s in plain_view.skills
                 if s.skill_id == "w1_no_supply_v1")
    assert _supplies_candidates(other) is False
    assert _supply_rung_candidates(plain_view, features) == ()


# ================================================ (b) the exploration slot
def test_b_the_agents_own_first_choice_is_probed_first(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    _method, result = _round(snapshot)

    pool = _pool(result)
    assert _supply_cand_id() in pool, pool
    agent_ids = [cid for cid in pool
                 if cid not in ("identity", _supply_cand_id())]
    assert agent_ids, "the exploration slot must survive the inject"
    order = list(result.probe_order_after_card)
    assert order[0] in agent_ids, order
    assert _supply_cand_id() in order
    assert order.index(agent_ids[0]) < order.index(_supply_cand_id())


def test_b_a_one_receipt_budget_spends_it_on_the_agent(tmp_path):
    """Conservative direction: the supply rung never displaces exploration."""
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    _method, result = _round(snapshot, budget=1)
    probed = [row["candidate_id"] for row in result.actual_probed_programs
              if row["kind"] == "probe"]
    assert probed and probed[0] != _supply_cand_id(), probed


# ============================================================= (c) the cap
def test_c_the_inject_is_inside_the_candidate_cap_not_added_to_it(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    _method, result = _round(snapshot)
    request = gerunner._a5_request(_series(), {"s0": _series()}, ORIGIN, DOMAIN)
    cap = request.task_context.deployment_constraints.maximum_candidates
    pool = _pool(result)
    assert len(pool) <= cap, (pool, cap)
    assert _supply_cand_id() in pool


# ======================================================== (d) no shortcut
class _StubMethod:
    def __init__(self):
        self.updated = []

    def _active_snapshot(self):
        # Only the revocation path reads this, and with ``store=None`` it
        # records revocation_pending rather than rewriting a snapshot.
        return types.SimpleNamespace(
            runtime_bundle_sha="stub",
            skills=(types.SimpleNamespace(skill_id=SUPPLY_SKILL_ID),))

    def update_experience_episode(self, episode):
        self.updated.append(episode)

    def handle_feedback_delayed(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("no pending Draft in these cases")


class _StubExecutor:
    def __init__(self, gain):
        self._gain = gain

    def evaluate(self, steps, origin):
        return types.SimpleNamespace(
            gain=self._gain, per_view_gain=[self._gain],
            verification=types.SimpleNamespace(passed=True))


def _supply_winner_result(*, support_gain, delayed_gain):
    """A round whose winner came from the supplied Skill."""
    steps = ((SUPPLY_OP, {}),)
    episode = online_loop._write_target_episode(
        domain=DOMAIN, op=SUPPLY_SKILL_ID,
        program_steps=[{"op": SUPPLY_OP, "params": {}}],
        support_gain=support_gain, support_context={},
        episode_id_suffix="_r1_p1", per_view_gain=[support_gain],
        support_origin=ORIGIN, task_spec=None, series_uids=("s0",))
    result = online_loop.RoundResult(round_name="w1", origin=ORIGIN)
    result._method = _StubMethod()
    result._values = {"s0": _series()}
    result._period = 24
    result._domain = DOMAIN
    result._series_uids = ("s0",)
    result._consumer_id = "ridge"
    result._episodes = [(episode, steps)]
    result._winner_steps = steps
    result._winner_candidate_id = _supply_cand_id()
    result.winner_program = [{"op": SUPPLY_OP, "params": {}}]
    result.deployed_skill_id = SUPPLY_SKILL_ID
    result._fast_skill_event = {"stage": "deployed_existing_skill",
                                "skill_id": SUPPLY_SKILL_ID}
    online_loop.open_delayed(result, _StubExecutor(delayed_gain),
                             delayed_origin=ORIGIN + 1, store=None)
    return result


def test_d_a_supplied_winner_the_delayed_gate_confirms_is_approved():
    result = _supply_winner_result(support_gain=0.40, delayed_gain=0.30)
    assert result.approved_skill_id == SUPPLY_SKILL_ID
    assert result._delayed_event["stage"] == "approved"
    assert result._delayed_event["route"] == "deployed_existing_skill"


def test_d_a_supplied_winner_the_delayed_gate_refuses_is_not_deployed():
    """Support drafted it; delayed did not confirm; the ledger must not move."""
    import run_e2_t6_cls_op_shared_harness as cls

    result = _supply_winner_result(support_gain=0.40, delayed_gain=0.0)
    assert result.approved_skill_id is None
    previous = [{"op": "identity", "params": {}}]
    assert cls._incumbent_after_delayed(result, previous) == previous


def test_d_a_delayed_harm_revokes_and_does_not_approve():
    result = _supply_winner_result(support_gain=0.40, delayed_gain=-0.40)
    assert result.approved_skill_id is None


def test_d_support_that_is_not_a_material_positive_drafts_nothing(tmp_path):
    """The supplied candidate has no Support shortcut: a neutral probe is an
    Episode and nothing more."""
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    method, result = _round(snapshot, backend_cls=_ProposeFailsBackend)
    assert result.winner_program is None
    episodes = list(method.experience_episodes)
    assert episodes, "the probe must still be recorded as evidence"
    assert all(str(ep.relation) != "POSITIVE" for ep in episodes)
    assert all(str(ep.local_status) == "EPISODE_ONLY" for ep in episodes)
    assert result.approved_skill_id is None


def test_d_the_supplied_card_never_grants_execution(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    card = next(s for s in snapshot.skills if s.skill_id == SUPPLY_SKILL_ID)
    assert card.risk_guards["authority"]["grants_execution"] is False
    assert card.allowed_tools == ()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
