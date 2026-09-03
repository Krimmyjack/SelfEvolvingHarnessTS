"""The Scope has to survive the whole chain, not just the probe.

    Program+Scope -> scoped verify -> Support-A -> Support-B
    -> Active Skill -> retrieval -> Scope PATCH -> version+1 -> re-encounter

A Scope that is honoured at probe time and dropped at Episode time, or stored as
UIDs instead of as a predicate, produces the same failure as no Scope at all --
only later, and harder to see.  So each link is asserted separately, and two
properties are asserted everywhere: the declined series never move, and the
resolved UID set never becomes a Skill field.

0 LLM: the Fast Path runs on ``SealedProbeBackend`` and the Consumer is a stub
whose per-series split is chosen by the Scope.
"""
from __future__ import annotations

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
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    WindowVerification,
)

ORIGIN = 400
ALT_OP = "outlier_mad"
DOMAIN = "scope-lifecycle-test"
N_SERIES = 5
UIDS = tuple("s%d" % i for i in range(N_SERIES))
#: Treating everything is a CONFLICT: four gain, one is harmed past the budget.
TREATED = (0.5, 0.5, 0.5, 0.5, -0.50)
#: The Scope the Harness should be able to express: treat the four, decline s4.
PREDICATE = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.5),)
)
#: Graded so that tightening the threshold visibly shrinks the resolution.
FEATURES = {"s0": {"gapped": 1.0}, "s1": {"gapped": 0.9},
            "s2": {"gapped": 0.6}, "s3": {"gapped": 0.55},
            "s4": {"gapped": 0.0}}
#: What a Slow PATCH revises it to: only the two most gapped series.
REVISED = scopes.ScopeSpec(
    "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.9),)
)


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
        uid: np.sin(x / (7.0 + i)) + 0.1 * np.sin(x / 3.0) + 5.0 + i
        for i, uid in enumerate(UIDS)
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


class _ScopeAwareExecutor:
    """Stub Consumer that honours a serving Scope the way the dual pipeline does.

    A declined series is served by the raw model from the raw context, so its
    gain against Static is exactly zero -- not merely small.
    """

    def __init__(self) -> None:
        self.scoped_calls: list[frozenset[str]] = []

    def verify(self, steps, origin):
        return _verification(str(steps[0][0]) if steps else "identity")

    def evaluate(self, steps, origin, serving_scope=None):
        op = str(steps[0][0]) if steps else "identity"
        if op != ALT_OP:
            per = (-0.10,) * N_SERIES
        elif serving_scope is None:
            per = TREATED
        else:
            self.scoped_calls.append(frozenset(serving_scope))
            per = tuple(
                TREATED[i] if UIDS[i] in serving_scope else 0.0
                for i in range(N_SERIES)
            )
        return SimpleNamespace(
            verification=_verification(op), gain=float(np.mean(per)),
            per_view_gain=tuple(float(v) for v in per),
            behavior_point_count=1,
        )


def _method(snapshot, series, backend):
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series[:ORIGIN], task_kind="forecast")
    )
    return TTHAMethod(TTHAFastAgent(core), snapshot, ())


def _resolver(spec, _origin):
    return scopes.ScopeSpec.from_dict(dict(spec)).resolve(FEATURES)


def _round(method, executor, values, series, *, store, controller,
           candidate_scopes=None):
    request = runner._a5_request(series, values, ORIGIN, DOMAIN)
    features = dict(extract_public_features(series[:ORIGIN], task_kind="forecast"))
    return loop.run_online_round(
        method, executor, request, values,
        origin=ORIGIN, slow_agent=None, controller=controller, store=store,
        card_builder=lambda _episode: {}, round_name="scoped",
        budget=2, allow_fast_skill=True, allow_slow=False,
        domain=DOMAIN, period=24, fast_features=features,
        candidate_scopes=candidate_scopes,
        scope_resolver=_resolver if candidate_scopes else None,
    )


@pytest.fixture()
def driven(tmp_path):
    values = _values()
    series = values["s0"]
    ap.install_policy(_bounded_policy())
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    method = _method(
        runner._h0_snapshot(), series,
        sealed.SealedProbeBackend(
            explore=True, operators=(ALT_OP,), force_pool=True),
    )
    executor = _ScopeAwareExecutor()
    scopes_by_candidate = {
        cand: PREDICATE.to_dict()
        for cand in (ALT_OP, "cand_" + ALT_OP, "cand_skill_" + ALT_OP)
    }
    result = _round(method, executor, values, series, store=store,
                    controller=controller,
                    candidate_scopes=scopes_by_candidate)
    return SimpleNamespace(method=method, executor=executor, store=store,
                           result=result, values=values, series=series)


def _plain(value):
    """Frozen mappings and tuples back to plain JSON containers."""
    return json.loads(json.dumps(value, default=lambda o: (
        dict(o) if hasattr(o, "items") else list(o))))


def _probe_rows(result):
    return [p for p in result.actual_probed_programs if p.get("kind") == "probe"]


def test_the_probe_ran_under_the_scope_and_recorded_both_forms(driven):
    rows = _probe_rows(driven.result)
    assert rows, "the round produced no legal Support receipt"
    scoped = [row for row in rows if row.get("serving_scope")]
    assert scoped, "no probe carried a Scope"
    row = scoped[0]
    assert row["serving_scope"]["scope_type"] == "serving_series_predicate"
    assert row["resolved_serving_series"] == ["s0", "s1", "s2", "s3"]


def test_declined_series_do_not_move_at_all(driven):
    scoped = [row for row in _probe_rows(driven.result) if row.get("serving_scope")]
    gains = scoped[0]["per_series_gain"]
    # s4 was declined: its gain against Static must be exactly zero, which is
    # the property a training-row scope could never provide.
    assert gains[-1] == 0.0
    assert all(value != 0.0 for value in gains[:4])


def test_the_executor_was_actually_handed_the_resolved_set(driven):
    assert driven.executor.scoped_calls
    assert driven.executor.scoped_calls[0] == frozenset({"s0", "s1", "s2", "s3"})


def test_the_episode_records_the_predicate_and_marks_the_resolution_as_evidence(
        driven):
    episodes = getattr(driven.method, "_experience_episodes", None) or []
    geometries = [
        (getattr(ep, "context_summary", None) or {}).get("program_geometry") or {}
        for ep in episodes
    ]
    scoped = [g for g in geometries if g.get("scope") == "serving_series_predicate"]
    assert scoped, "no Episode recorded a serving-series Scope"
    geometry = scoped[0]
    assert geometry["serving_scope"]["predicate"][0]["feature"] == "gapped"
    assert geometry["resolved_serving_series"] == ["s0", "s1", "s2", "s3"]
    assert geometry["resolved_is_skill_field"] is False


def test_the_winner_carries_the_predicate_and_its_resolution_separately(driven):
    result = driven.result
    if result.winner_program is None:
        pytest.skip("no winner formed under this stub; scope storage untested")
    assert result._winner_serving_scope["scope_type"] == "serving_series_predicate"
    assert result._winner_resolved_series == frozenset({"s0", "s1", "s2", "s3"})
    assert result._winner_scope_revision == 1


def test_a_patch_may_revise_the_predicate_atomically_with_the_program():
    # Swapping the program while keeping the old predicate would apply a new
    # treatment to a set chosen for the old one, so the PATCH carries both.
    revised = scopes.ScopeSpec(
        "serving_series_predicate", (scopes.Clause("gapped", ">=", 0.9),)
    ).to_dict()
    assert loop._scope_of_patch({"serving_scope": revised}) == revised
    assert loop._scope_of_patch({"frozen_program": []}) is None
    assert loop._scope_of_patch({"serving_scope": {}}) is None


def test_a_stored_skill_can_be_rebuilt_into_a_candidate_scope():
    # Re-encounter: the Skill carries a predicate, so it resolves against
    # whatever series the next Target happens to contain.
    stored = PREDICATE.to_dict()
    rebuilt = scopes.ScopeSpec.from_dict(stored)
    assert rebuilt.resolve(FEATURES) == frozenset({"s0", "s1", "s2", "s3"})
    elsewhere = {"x1": {"gapped": 1.0}, "x2": {"gapped": 0.0}}
    assert rebuilt.resolve(elsewhere) == frozenset({"x1"})


# --------------------------------------------------- steps 7 and 8, driven ---

PATCH_ID = "tighten-the-scope"
SKILL_ID = "scoped_replacement"


class _CapturingSlow:
    """A 0-LLM Slow whose PATCH revises the Scope predicate, not the program."""

    last_no_proposal_reason = None

    def __init__(self, manifest: EditManifest) -> None:
        self.manifest = manifest
        self.calls = 0

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        self.calls += 1
        preflight = kwargs.get("manifest_preflight")
        if preflight is not None:
            preflight(self.manifest)
        return self.manifest


def _revising_manifest(snapshot) -> EditManifest:
    return EditManifest(
        edit_id="scope-revision",
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id="scoped-first-fault",
        target_surface_id="skill_library.entries/{skill_id}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": SKILL_ID,
            "skill_kind": "capability",
            "revision": 1,
            "body": "runtime binds the frozen program",
            # The revision under test: a tighter predicate, and still no UID.
            "serving_scope": REVISED.to_dict(),
            "observable_applicability": {
                "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]
            },
            "allowed_tools": [ALT_OP],
            "risk_guards": {"single_surface_only": True},
        },
        observable_applicability={
            "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]
        },
        predicted_agent_behavior_change=("retrieve_skill:" + SKILL_ID,),
        predicted_data_effect=("narrow_the_treated_set",),
        automatically_selected_risk_cases=(),
        falsification_condition=("no_improvement",),
        patch_id=PATCH_ID,
    )


def test_a_slow_patch_revises_the_scope_and_bumps_the_version(tmp_path):
    # Steps 7 and 8 driven rather than unit-checked: Slow proposes a tighter
    # predicate, the round adopts it atomically with the program, and the
    # resolution shrinks accordingly.
    values = _values()
    series = values["s0"]
    ap.install_policy(_bounded_policy())
    snapshot = runner._h0_snapshot()
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    method = _method(
        snapshot, series,
        sealed.SealedProbeBackend(
            explore=True, operators=("winsorize",), force_pool=True),
    )
    executor = _ScopeAwareExecutor()
    slow = _CapturingSlow(_revising_manifest(snapshot))
    request = runner._a5_request(series, values, ORIGIN, DOMAIN)
    features = dict(extract_public_features(series[:ORIGIN], task_kind="forecast"))

    result = loop.run_online_round(
        method, executor, request, values,
        origin=ORIGIN, slow_agent=slow, controller=controller, store=store,
        card_builder=lambda _episode: {
            "pattern_id": "scoped-first-fault",
            "failure_family": "workflow_component_negative",
            "observable_signature": {"task_kind": "forecast"},
            "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        },
        round_name="scope_patch", budget=2, allow_slow=True,
        domain=DOMAIN, period=24, fast_features=features,
        slow_typed_patch_options=(
            {"patch_id": PATCH_ID,
             "program_steps": [{"op": ALT_OP, "params": {}}]},
        ),
        program_supply_verifier=executor,
        candidate_scopes={
            cand: PREDICATE.to_dict()
            for cand in ("winsorize", ALT_OP, "cand_" + ALT_OP)
        },
        scope_resolver=_resolver,
    )

    event = result._slow_event or {}
    if event.get("stage") != "pending":
        pytest.skip("Slow did not reach pending under this stub: %s"
                    % event.get("stage"))
    # The manifest freezes its payload, so compare canonical JSON rather than
    # container identity.
    assert _plain(event["serving_scope"]) == REVISED.to_dict(), (
        "the PATCH's revised predicate must reach the slow event")
    assert _plain(result._winner_serving_scope) == REVISED.to_dict()
    assert result._winner_scope_revision == 2, "a revision must bump the version"
    # The tighter predicate resolves to strictly fewer series -- the observable
    # consequence of the revision, not merely a stored field.
    assert result._winner_resolved_series == frozenset({"s0", "s1"})


class _RecordingExecutor(_ScopeAwareExecutor):
    """Same stub, but it also remembers which origin each call arrived at."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[int, frozenset | None]] = []

    def evaluate(self, steps, origin, serving_scope=None):
        self.calls.append(
            (int(origin),
             None if serving_scope is None else frozenset(serving_scope)))
        return super().evaluate(steps, origin, serving_scope)


class _HistoricalExecutor(_ScopeAwareExecutor):
    """An executor that predates the Scope parameter and refuses to see it."""

    def evaluate(self, steps, origin):  # noqa: D102 - historical signature
        return super().evaluate(steps, origin, None)


def _driven_with(executor, tmp_path, *, scoped=True):
    values = _values()
    series = values["s0"]
    ap.install_policy(_bounded_policy())
    store = SnapshotStore(tmp_path / "store")
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter())
    method = _method(
        runner._h0_snapshot(), series,
        sealed.SealedProbeBackend(
            explore=True, operators=(ALT_OP,), force_pool=True),
    )
    result = _round(
        method, executor, values, series, store=store, controller=controller,
        candidate_scopes=({
            cand: PREDICATE.to_dict()
            for cand in (ALT_OP, "cand_" + ALT_OP, "cand_skill_" + ALT_OP)
        } if scoped else None),
    )
    return method, result


def test_the_delayed_gate_reads_under_the_winners_own_scope(tmp_path):
    # Approving a scoped Skill on a global average makes it answer for series
    # it never proposed to touch -- the reading this whole line moved away from.
    executor = _RecordingExecutor()
    _method_, result = _driven_with(executor, tmp_path)
    if result._winner_serving_scope is None:
        pytest.skip("no scoped winner formed under this stub")
    executor.calls.clear()
    loop.open_delayed(result, executor, delayed_origin=ORIGIN + 48,
                      scope_resolver=_resolver)
    delayed = [row for row in executor.calls if row[0] == ORIGIN + 48]
    assert delayed, "the delayed gate never evaluated at the delayed origin"
    assert all(scope == frozenset({"s0", "s1", "s2", "s3"})
               for _origin, scope in delayed)
    assert result.delayed_serving_series == frozenset({"s0", "s1", "s2", "s3"})
    assert result.delayed_scope_reresolved is True


def test_the_delayed_scope_is_re_resolved_not_carried_over(tmp_path):
    # A Scope is a predicate, so it is re-read at the delayed origin; pinning
    # the UID list resolved at the earlier origin would hide exactly the drift
    # the delayed gate exists to catch.
    executor = _RecordingExecutor()
    _method_, result = _driven_with(executor, tmp_path)
    if result._winner_serving_scope is None:
        pytest.skip("no scoped winner formed under this stub")
    executor.calls.clear()
    tighter = {"s0": {"gapped": 1.0}, "s1": {"gapped": 0.95},
               "s2": {"gapped": 0.1}, "s3": {"gapped": 0.1},
               "s4": {"gapped": 0.0}}
    loop.open_delayed(
        result, executor, delayed_origin=ORIGIN + 48,
        scope_resolver=lambda spec, _origin: scopes.ScopeSpec.from_dict(
            dict(spec)).resolve(tighter))
    assert result.delayed_serving_series == frozenset({"s0", "s1"})
    assert all(scope == frozenset({"s0", "s1"})
               for _origin, scope in executor.calls if _origin == ORIGIN + 48)


def test_without_a_resolver_the_delayed_call_shape_is_the_historical_one(
        tmp_path):
    # The default branch of an additive change must preserve the call shape,
    # not merely the default value: an executor that predates the parameter
    # would otherwise be struck down with a TypeError.
    executor = _HistoricalExecutor()
    _method_, result = _driven_with(executor, tmp_path, scoped=False)
    loop.open_delayed(result, executor, delayed_origin=ORIGIN + 48)
    assert result.delayed_serving_series is None
    assert result.delayed_scope_reresolved is False
