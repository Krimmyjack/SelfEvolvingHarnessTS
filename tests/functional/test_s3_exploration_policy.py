"""S3-R1 focused tests for the exploration/allocation policy surface.

DEFAULT must reproduce the pre-parameterization probe contract
(agent-before-DRAFT-supply, chosen-first, first Support-POSITIVE stops).
Non-DEFAULT values are fail-closed and only live under install_policy.
G3 surfaces (dual-gate, harm threshold, delayed deployment) are not on
this surface and stay byte-identical.
"""
from __future__ import annotations

import json
import sys
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
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    CLASSIFICATION_MATERIAL_THRESHOLD,
    classify_relation,
)
from SelfEvolvingHarnessTS.methods.ttha.exploration_policy import (  # noqa: E402
    DEFAULT,
    LEGAL_DOMAINS,
    ExplorationPolicy,
    active_policy,
    install_policy,
    reset_policy,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.signed_radius import MATERIAL_THRESHOLD  # noqa: E402

DOMAIN = "s3_policy_test"
ORIGIN = 400
SUPPLY_OP = "hampel_filter"
SUPPLY_SKILL_ID = "s3_supply_rung_v1"
AGENT_OPS = ("winsorize", "outlier_mad")
# baseline mean_smase = 1.0; gain = 1.0 - candidate_smase
_OP_GAIN = {
    "winsorize": 0.02,
    "outlier_mad": 0.02,
    "hampel_filter": 0.10,
}


def _series():
    t = np.arange(1024, dtype=np.float64)
    return np.sin(t / 7.0) + 0.1 * np.sin(t / 3.0) + 5.0


def _eval_by_op(roster, values, compiled, config, *, origin):
    if compiled is None:
        smase = 1.0
    else:
        steps = compiled.candidate.program.execution_steps()
        op = steps[0][0] if steps else "identity"
        smase = 1.0 - float(_OP_GAIN.get(op, 0.0))
    return {
        "mean_smase": smase,
        "per_view_smase": [smase],
        "behavior_point_count": 10,
    }


def _supply_card(*, skill_id=SUPPLY_SKILL_ID, supplies=True):
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
            "card_kind": "s3_policy_test",
            "requires_target_support": True,
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


def _round(snapshot, *, budget=2, policy=None):
    reset_policy()
    if policy is not None:
        install_policy(policy)
    values = {"s0": _series()}
    series0 = values["s0"]
    backend = sealed.SealedProbeBackend(
        explore=True, operators=AGENT_OPS,
        max_propose_candidates=len(AGENT_OPS),
        force_pool=True)
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:ORIGIN], task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), snapshot, ())
    executor = ScopeExecutor(
        [{"series_uid": "s0", "role": "train"},
         {"series_uid": "s0", "role": "eval"}],
        values, {"anchors": []}, evaluate_fn=_eval_by_op)
    try:
        result = online_loop.run_online_round(
            method, executor,
            gerunner._a5_request(series0, values, ORIGIN, DOMAIN),
            values, origin=ORIGIN, slow_agent=None, controller=None,
            store=None, card_builder=gerunner._a5v2_card,
            round_name="s3_policy", budget=budget, allow_slow=False,
            domain=DOMAIN, period=24,
            fast_features=dict(extract_public_features(
                series0[:ORIGIN], task_kind="forecast")),
            allow_fast_skill=False)
    finally:
        reset_policy()
    return method, result


def _pool(result):
    return list(getattr(result._method.last_trace, "candidate_ids", ()) or ())


def _supply_id(skill_id=SUPPLY_SKILL_ID):
    return "cand_skill_%s" % skill_id


def _agent_ids(pool, skill_id=SUPPLY_SKILL_ID):
    return [cid for cid in pool
            if cid not in ("identity", _supply_id(skill_id))]


def _probed_ids(result):
    return [row["candidate_id"] for row in result.actual_probed_programs
            if row["kind"] == "probe"]


def _winner_ops(result):
    return [step["op"] for step in (result.winner_program or [])]


@pytest.fixture(autouse=True)
def _always_reset_policy():
    reset_policy()
    yield
    reset_policy()


# ------------------------------------------------------------------ 1 DEFAULT
def test_default_agent_before_draft_supply_chosen_first_stops(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    _method, result = _round(snapshot)
    pool = _pool(result)
    supply = _supply_id()
    agent_ids = _agent_ids(pool)
    assert supply in pool, pool
    assert agent_ids, pool
    order = list(result.probe_order_after_card)
    assert order[0] in agent_ids, order
    assert order.index(agent_ids[0]) < order.index(supply)
    probed = _probed_ids(result)
    assert probed, probed
    assert probed[0] in agent_ids
    assert supply not in probed
    assert result.target_support_receipts_used == 1
    assert _winner_ops(result)[0] in AGENT_OPS
    assert result.first_positive_support_receipt_index == 1


# -------------------------------------------------------------- 2 fail-closed
def test_illegal_policy_value_fail_closed():
    with pytest.raises(ValueError, match="illegal policy value"):
        ExplorationPolicy(probe_order_rule="not_a_rule").validate()
    with pytest.raises(ValueError, match="illegal policy value"):
        install_policy(ExplorationPolicy(supply_reserved_probe_slots=2))
    assert active_policy() == DEFAULT
    with pytest.raises(ValueError, match="illegal policy value"):
        install_policy(ExplorationPolicy(displacement_margin=0.02))
    assert active_policy() == DEFAULT


# ----------------------------------------------------- 3 install / reset / isolation
def test_install_reset_does_not_leak_across_arms(tmp_path):
    assert active_policy() == DEFAULT
    edited = ExplorationPolicy(first_positive_stop=False)
    install_policy(edited)
    assert active_policy().first_positive_stop is False
    try:
        raise RuntimeError("arm failed")
    except RuntimeError:
        reset_policy()
    assert active_policy() == DEFAULT
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    _method, result = _round(snapshot)  # no policy → DEFAULT
    assert _supply_id() not in _probed_ids(result)
    assert active_policy() == DEFAULT


# ----------------------------------------------- 4 reserved probe keeps budget=2
def test_reserved_supply_slot_keeps_total_budget_two(tmp_path):
    """Two agent candidates sit in front of the card. Budget=2 and
    supply_reserved_probe_slots=1 must skip the second agent (no extra
    receipt) so the last receipt still reaches the supplied card."""
    from types import SimpleNamespace

    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    supply = _supply_id()
    pool = ["cand_winsorize", "cand_outlier_mad", supply]
    steps_map = {
        "cand_winsorize": (("winsorize", {}),),
        "cand_outlier_mad": (("outlier_mad", {}),),
        supply: (("hampel_filter", {}),),
    }

    class _Method:
        last_trace = SimpleNamespace(
            candidate_ids=tuple(pool),
            candidate_program_steps=steps_map,
            chosen_candidate_id="cand_winsorize",
            memory_resolution_status="no_memory")

        def bind_round_data(self, *args, **kwargs):
            return None

        def prepare(self, *args, **kwargs):
            return None

        def _active_snapshot(self):
            return snapshot

        def append_experience_episode(self, episode):
            return None

    class _Exec:
        def evaluate(self, steps, origin):
            return SimpleNamespace(
                gain=0.02, per_view_gain=[0.02],
                verification=SimpleNamespace(passed=True))

    policy = ExplorationPolicy(
        probe_order_rule="pool_as_built",
        supply_reserved_probe_slots=1,
        first_positive_stop=False)
    install_policy(policy)
    series0 = _series()
    values = {"s0": series0}
    try:
        result = online_loop.run_online_round(
            _Method(), _Exec(),
            gerunner._a5_request(series0, values, ORIGIN, DOMAIN),
            values, origin=ORIGIN, slow_agent=None, controller=None,
            store=None, card_builder=gerunner._a5v2_card,
            round_name="s3_reserved", budget=2, allow_slow=False,
            domain=DOMAIN, period=24,
            fast_features=dict(extract_public_features(
                series0[:ORIGIN], task_kind="forecast")))
    finally:
        reset_policy()
    assert result.target_support_receipts_used == 2
    assert supply in _probed_ids(result)
    skipped = [row["candidate_id"] for row in result.actual_probed_programs
               if row["kind"] == "skipped_reserved_for_supply"]
    assert skipped == ["cand_outlier_mad"], result.actual_probed_programs


# ------------------------------------ 5 supply-first / max-gain / tie-break
def test_supply_first_probes_the_card_before_the_agent(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    policy = ExplorationPolicy(probe_order_rule="supply_first_then_agent")
    _method, result = _round(snapshot, policy=policy)
    order = list(result.probe_order_after_card)
    supply = _supply_id()
    agent_ids = _agent_ids(_pool(result))
    assert order[0] == supply, order
    assert all(order.index(supply) < order.index(aid) for aid in agent_ids
               if aid in order)
    assert _probed_ids(result)[0] == supply
    assert _winner_ops(result) == [SUPPLY_OP]


def test_max_gain_picks_the_higher_positive_when_stop_is_off(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    policy = ExplorationPolicy(
        probe_order_rule="agent_first_then_supply",
        first_positive_stop=False,
        winner_compare_rule="max_support_gain_among_probed_positive")
    _method, result = _round(snapshot, policy=policy)
    probed = _probed_ids(result)
    assert _supply_id() in probed
    assert any(cid in probed for cid in _agent_ids(_pool(result)))
    assert _winner_ops(result) == [SUPPLY_OP]
    assert result._winner_candidate_id == _supply_id()


def test_tie_break_prefer_supplied_on_equal_gain(tmp_path):
    _store, snapshot = _snapshot_with(_supply_card(), tmp_path)
    # Force equal POSITIVE gains so the margin treats them as a tie.
    original = dict(_OP_GAIN)
    _OP_GAIN["hampel_filter"] = 0.02
    try:
        policy = ExplorationPolicy(
            probe_order_rule="agent_first_then_supply",
            first_positive_stop=False,
            winner_compare_rule="max_support_gain_among_probed_positive",
            displacement_margin=0.05,
            tie_break_rule="prefer_supplied")
        _method, result = _round(snapshot, policy=policy)
    finally:
        _OP_GAIN.update(original)
    assert result._winner_candidate_id == _supply_id()
    assert _winner_ops(result) == [SUPPLY_OP]


# -------------------------------------------------------------- 6 G3 untouched
def test_g3_material_threshold_and_conflict_classifier_untouched():
    assert MATERIAL_THRESHOLD == 0.005
    assert CLASSIFICATION_MATERIAL_THRESHOLD == 0.005
    facts = classify_relation(
        aggregate_gain=0.10,
        per_series_gains={"s0": 0.10, "s1": -0.02},
        is_identity=False)
    assert facts["relation"] == "CONFLICT"
    positive = classify_relation(
        aggregate_gain=0.10,
        per_series_gains={"s0": 0.10},
        is_identity=False)
    assert positive["relation"] == "POSITIVE"


def test_instrument_stop_is_not_a_scientific_verdict():
    import run_e2_s3_pilot_probe_policy as s3
    gate = s3._instrument_gate(
        [], stopped="BACKEND_UNAVAILABLE", llm_edit_illegal=False)
    assert gate["candidate"] is None
    assert gate["instrument_stop"] == "BACKEND_UNAVAILABLE"
    judged = s3._judge([], llm_edit_illegal=False)
    assert judged["candidate"] == "S3_SEED_UNREPRODUCED"


def test_legal_domains_do_not_include_g3_knobs():
    forbidden = (
        "material_threshold", "harm_threshold", "maximum_candidates",
        "grants_execution", "requires_target_support", "scope",
        "delayed", "capacity")
    joined = " ".join(LEGAL_DOMAINS)
    for name in forbidden:
        assert name not in joined
    assert set(DEFAULT.to_dict()) == set(LEGAL_DOMAINS)

