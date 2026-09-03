"""The whole P4U-v2 chain, driven rather than argued, with no LLM.

    risk refusal -> RISK_GAP -> Slow ADDs an inactive Draft carrying a narrowed
    Scope -> Support re-verified under that Scope -> delayed re-resolved at
    origin+48 -> only then activated -> the stored Skill transfers as a
    predicate

Each link is asserted separately because each has already failed once in this
protocol in a way that reading the code did not reveal: the delayed gate read a
scoped Skill on the global serving set, and the Support replay did the same.
Both failures look like "Slow could not find a working revision" from the
outside, so "the controller would not activate early" is not something to take
on trust.

The harm sits **inside** the original Scope, which is the shape the Source line
actually produced: selecting on defect presence picks up the series the program
goes on to damage, so the repair has to be a narrower predicate rather than a
different program.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "evaluation" / "functional", ROOT / "methods" / "ttha"):
    sys.path.insert(0, str(_path))

import run_v1_guidance_evolution as runner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402

from evaluation.main_protocol_p4 import scope_narrowing_preflight as pf  # noqa: E402
from evaluation.main_protocol_p4 import scope_spec as scopes  # noqa: E402
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
)
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

ORIGIN, ALT_OP, N_SERIES = 400, "outlier_mad", 5
UIDS = tuple("s%d" % index for index in range(N_SERIES))
SKILL_ID = "risk_scoped_draft"

#: Aggregate +0.14 over five served series, one harmed at 0.50: inside the
#: harmed-fraction budget, over the single-series line.  s3 is in the original
#: Scope and outside the revised one -- the harm lives in the selected set.
TREATED = (0.5, 0.5, 0.2, -0.50, 0.0)
GLOBAL_GAIN = 0.14
SCOPED_GAIN = 0.20

FEATURES = {"s0": {"gapped": 1.0}, "s1": {"gapped": 0.95},
            "s2": {"gapped": 0.6}, "s3": {"gapped": 0.55},
            "s4": {"gapped": 0.0}}
ORIGINAL = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.5),))
#: Keeps the original clause and adds one: a structural narrowing.
REVISED = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.5),
                                 scopes.Clause("gapped", ">=", 0.9)))
#: Drops the original clause: reaches further, and must never be adopted.
WIDENED = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.1),))


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
            per = (0.0,) * N_SERIES
        elif serving_scope is None:
            per = TREATED
        else:
            per = tuple(TREATED[i] if UIDS[i] in serving_scope else 0.0
                        for i in range(N_SERIES))
        return SimpleNamespace(
            verification=_verification(op), gain=float(np.mean(per)),
            per_view_gain=tuple(float(value) for value in per),
            behavior_point_count=1)


class _RecordingSlow:
    """0-LLM Slow: returns one prepared manifest and remembers what it saw."""

    last_no_proposal_reason = None

    def __init__(self, manifest: EditManifest) -> None:
        self.manifest = manifest
        self.calls = 0
        self.catalogs: list[list[tuple[str, str]]] = []
        self.cards: list[dict] = []

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        self.calls += 1
        self.catalogs.append(
            [(entry["surface_id"], entry["operation"]) for entry in surface_catalog])
        self.cards.append(dict(card))
        preflight = kwargs.get("manifest_preflight")
        if preflight is not None:
            preflight(self.manifest)
        return self.manifest


def _manifest(snapshot, scope: scopes.ScopeSpec) -> EditManifest:
    return EditManifest(
        edit_id="risk-scope-revision",
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id="risk-refusal",
        target_surface_id="skill_library.entries/{skill_id}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1", "skill_id": SKILL_ID,
            "skill_kind": "capability", "revision": 1,
            "body": "runtime binds the frozen program",
            "serving_scope": scope.to_dict(),
            "observable_applicability": {
                "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]},
            "allowed_tools": [ALT_OP],
            "risk_guards": {"single_surface_only": True},
        },
        observable_applicability={
            "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]},
        predicted_agent_behavior_change=("retrieve_skill:" + SKILL_ID,),
        predicted_data_effect=("narrow_the_treated_set",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_gain",),
        patch_id=loop.RISK_REFUSAL_PATCH_ID,
    )


def _resolve(spec, _origin):
    return scopes.ScopeSpec.from_dict(dict(spec)).resolve(FEATURES)


def _preflight(original, proposed, _origin):
    return pf.validate_narrowing(
        original, proposed, features=FEATURES,
        available_features=["gapped"]).to_dict()


def _skill_ids(snapshot) -> list[str]:
    return sorted(str(skill.skill_id) for skill in snapshot.skills)


def _drive(scope: scopes.ScopeSpec):
    values = {uid: np.sin(np.arange(1024, dtype=np.float64) / (7.0 + index)) + 5.0 + index
              for index, uid in enumerate(UIDS)}
    series = values["s0"]
    ap.install_policy(ap.AdmissionPolicy(
        rule=ap.BOUNDED_V1, max_harmed_fraction=0.20, max_single_series_harm=0.30))
    root = Path(tempfile.mkdtemp())
    store = SnapshotStore(root / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    snapshot = runner._h0_snapshot()
    method = TTHAMethod(TTHAFastAgent(TTHAAgentCore(
        sealed.SealedProbeBackend(
            explore=True, operators=(ALT_OP,), force_pool=True),
        LocalPublicToolGateway(series[:ORIGIN], task_kind="forecast"))),
        snapshot, ())
    executor = _ScopedExecutor()
    slow = _RecordingSlow(_manifest(snapshot, scope))
    result = loop.run_online_round(
        method, executor,
        runner._a5_request(series, values, ORIGIN, "risk-lifecycle"), values,
        origin=ORIGIN, slow_agent=slow, controller=controller, store=store,
        card_builder=lambda _episode: {
            "pattern_id": "risk-refusal",
            "failure_family": "workflow_component_negative",
            "observable_signature": {"task_kind": "forecast"},
            "workflow": {"steps": [{"op": ALT_OP, "params": {}}]}},
        round_name="risk_lifecycle", budget=8, allow_slow=True,
        allow_fast_skill=True, domain="risk-lifecycle", period=24,
        fast_features=dict(extract_public_features(
            series[:ORIGIN], task_kind="forecast")),
        candidate_scopes={candidate: ORIGINAL.to_dict() for candidate in
                          (ALT_OP, "cand_" + ALT_OP, "cand_skill_" + ALT_OP)},
        scope_resolver=_resolve, scope_revision_preflight=_preflight,
        program_supply_verifier=executor)
    return SimpleNamespace(result=result, method=method, store=store,
                           slow=slow, executor=executor)


@pytest.fixture(scope="module")
def narrowed():
    return _drive(REVISED)


@pytest.fixture(scope="module")
def widened():
    return _drive(WIDENED)


# ------------------------------------------------------ the accepted chain ---

def test_the_refusal_reaches_slow_through_a_single_add_surface(narrowed):
    assert narrowed.result._slow_trigger == "risk_refusal"
    assert narrowed.result.risk_refusal_count == 1
    assert narrowed.slow.calls == 1
    assert narrowed.slow.catalogs == [
        [("skill_library.entries/{skill_id}", "ADD")]], (
        "RISK_GAP must expose exactly one surface, and only the ADD")


def test_slow_can_only_bind_the_probes_own_program(narrowed):
    """The freeze is structural: the whitelist holds one option, the probe's."""
    options = narrowed.slow.cards[0]["typed_patch_options"]
    assert options == [{
        "patch_id": loop.RISK_REFUSAL_PATCH_ID,
        "program_steps": [{"op": ALT_OP, "params": {}}]}]


def test_the_card_slow_receives_actually_carries_the_refusal(narrowed):
    """The evidence has to be in the card at the moment Slow is called.

    A first live run passed these facts through a holder the Runner filled
    after the round had already returned, so the card Slow actually saw held
    only the pattern id and the frozen program.  Slow proposed a Draft with no
    Scope, the delayed gate refused it, and the run looked like "Slow cannot
    revise a Scope" when Slow had never been told there was one.
    """
    card = narrowed.slow.cards[0]
    refusal = card["risk_refusal"]
    assert refusal["reason"] == "single_series_harm_over_budget"
    assert refusal["aggregate_gain"] == pytest.approx(GLOBAL_GAIN)
    assert refusal["max_single_series_harm"] == pytest.approx(0.50)
    assert refusal["serving_scope"] == ORIGINAL.to_dict()
    assert refusal["per_series_gain"] == list(TREATED)


def test_no_series_identity_reaches_the_card(narrowed):
    """Anonymity is structural: the writer never learns a UID to name."""
    blob = json.dumps(narrowed.slow.cards[0])
    for uid in UIDS:
        assert '"%s"' % uid not in blob


def test_the_narrowing_preflight_ran_and_accepted_it(narrowed):
    verdict = narrowed.result._scope_revision_preflight
    assert verdict["accepted"] is True
    assert verdict["reason"] == "strictly_narrower"
    assert verdict["added_clauses"] == [
        {"feature": "gapped", "op": ">=", "threshold": 0.9}]
    assert (verdict["original_resolved"], verdict["proposed_resolved"]) == (4, 2)


def test_support_is_re_verified_under_the_revised_scope_not_globally(narrowed):
    """The discriminating number.

    Replaying the frozen program without the revised Scope reproduces exactly
    the configuration the gate just refused, and would record +0.14.  The
    scoped replay records +0.20.  Both clear the material line, so this is the
    only place the difference is visible at all.
    """
    event = narrowed.result._slow_event or {}
    assert event.get("stage") == "pending"
    assert event.get("support_gain") == pytest.approx(SCOPED_GAIN)
    assert event.get("support_gain") != pytest.approx(GLOBAL_GAIN)


def test_the_draft_is_written_to_the_fork_and_is_not_active(narrowed):
    """Not "the controller would not activate early" -- it did not."""
    assert SKILL_ID not in _skill_ids(narrowed.method._active_snapshot())
    assert narrowed.result.pending_patch_id == loop.RISK_REFUSAL_PATCH_ID
    assert narrowed.result.winner_program is not None


def test_the_delayed_gate_re_resolves_the_revised_predicate(narrowed):
    loop.open_delayed(narrowed.result, narrowed.executor,
                      delayed_origin=ORIGIN + 48, store=narrowed.store,
                      scope_resolver=_resolve)
    assert narrowed.result.delayed_scope_reresolved is True
    assert narrowed.result.delayed_serving_series == frozenset({"s0", "s1"})
    event = narrowed.result._delayed_event or {}
    assert event.get("stage") == "approved"
    assert event.get("delayed_gain") == pytest.approx(SCOPED_GAIN)


def test_only_after_the_delayed_gate_does_the_draft_activate(narrowed):
    # Ordering matters: the delayed gate above is what makes this legal.
    assert loop.activate_approved(narrowed.result, narrowed.store) is True
    assert SKILL_ID in _skill_ids(narrowed.method._active_snapshot())


def test_the_stored_skill_carries_a_predicate_that_transfers(narrowed):
    """Re-encounter: it resolves on a cohort whose UIDs it has never seen."""
    stored = next(skill for skill in narrowed.method._active_snapshot().skills
                  if str(skill.skill_id) == SKILL_ID)
    spec = scopes.ScopeSpec.from_dict(
        json.loads(json.dumps(stored.serving_scope, default=dict)))
    assert len(spec.clauses) == 2, "the revised predicate must be what was stored"
    assert spec.resolve(FEATURES) == frozenset({"s0", "s1"})
    assert spec.resolve(
        {"x1": {"gapped": 1.0}, "x2": {"gapped": 0.6}}) == frozenset({"x1"})


# ------------------------------------------------------- the refused chain ---

def test_a_widening_is_refused_and_nothing_is_deployed(widened):
    """The safety half: an authorized cause must not carry an unsafe Scope."""
    verdict = widened.result._scope_revision_preflight
    assert verdict["accepted"] is False
    assert "dropped or rewritten" in verdict["reason"]
    event = widened.result._slow_event or {}
    assert event.get("stage") == "scope_revision_refused"
    # The whole revision is void: falling back to the original Scope would
    # deploy the configuration the tail budget just refused.
    assert widened.result.winner_program is None
    assert widened.result.pending_patch_id is None
    assert SKILL_ID not in _skill_ids(widened.method._active_snapshot())


def test_a_refused_revision_cannot_be_activated(widened):
    assert loop.activate_approved(widened.result, widened.store) is False
    assert SKILL_ID not in _skill_ids(widened.method._active_snapshot())
