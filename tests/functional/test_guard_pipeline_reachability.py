"""B: the guard tier of the three-way visibility policy, made reachable.

The policy itself is already frozen and is not touched here.  An experience
card with no authorized TRY and no repeated scoped RISK is Slow-only; one with
repeated scoped harm evidence reaches Fast as a structured avoid; one with
repeated scoped positive evidence may reach Fast as a TRY.  ``retrieval.py``
has read that middle tier since T1.

The s1a audits then showed it was dead code on the classification line, for
three mechanical reasons that all sit upstream of the predicate:

1. ``online_loop._write_target_episode`` never wrote
   ``context_summary.task_episode_id``, so ``risk_skill._task_of`` returned
   ``''`` for every Episode and two curriculum units collapsed into one
   counted Task;
2. ``source_skill.build_skill_payload`` never copied the deduplicated count
   onto ``risk_guards.evidence_distinct_task_count``, the one field the
   predicate reads to tell repeated evidence from prose;
3. the classification shared harness never called
   ``run_risk_skill_lifecycle``, so a harm Episode was never compiled at all.

One focused check per repair, zero LLM.  No threshold, no clause-kind rule and
no scope semantics is asserted differently here than before the repair.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_t6_cls_op_shared_harness as cls_harness  # noqa: E402
import run_v1_guidance_evolution as gerunner  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import SkillKind  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    classification_task_spec_v1,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
    _skill_frozen_candidates,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    _write_target_episode,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    _is_inert_experience_card,
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import (  # noqa: E402
    ScopeExecutor,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    risk_skill,
    source_skill,
)

H0 = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"

# Two curriculum units of the classification line, named exactly the way the
# shared harness names a cell: dataset/condition.
UNIT_A = "GunPoint/impulse_v2"
UNIT_B = "Wine/impulse_v2"

CLASSIFICATION_FEATURES = {"task_kind": "classification"}
HARM_GAIN = -0.08          # < -CLASSIFICATION_MATERIAL_THRESHOLD (0.005)


def _cls_spec():
    return classification_task_spec_v1(
        downstream_model_class="ridge-raw-plus-difference-v1")


def _harm_episode(unit: str, op: str = "outlier_mad", *, gain=HARM_GAIN):
    """One harm Episode written by the production classification writer."""
    return _write_target_episode(
        domain=unit,
        op=op,
        program_steps=[{"op": op, "params": {}}],
        support_gain=gain,
        support_context={},
        episode_id_suffix="_r1_p1",
        task_spec=_cls_spec(),
        series_uids=("heldin_observation",),
        consumer_id="ridge-raw-plus-difference-v1",
    )


# ==========================================================================
# Repair 1 -- the Episode carries the unit it happened in
# ==========================================================================
def test_a_classification_episode_carries_a_non_empty_unit_id():
    episode = _harm_episode(UNIT_A)
    assert episode.context_summary["task_episode_id"] == UNIT_A
    # The id is what the census actually reads; asserting the key alone would
    # not show the counter can see it.
    assert risk_skill._task_of(episode) == UNIT_A


def test_a_the_unit_id_changes_with_the_unit_and_not_with_the_probe():
    first = _harm_episode(UNIT_A)
    second = _harm_episode(UNIT_B)
    assert risk_skill._task_of(first) != risk_skill._task_of(second)

    # Two probes inside one unit stay one unit: the count is about the family,
    # and A3/A5 probing the same cell must not read as two Tasks.
    same_unit_again = _harm_episode(UNIT_A, op="outlier_iqr")
    assert risk_skill._task_of(same_unit_again) == risk_skill._task_of(first)


def _neutral_eval(roster, values, compiled, config, *, origin):
    """Every candidate neutral: this checks what is written, not the gain."""
    return {"mean_smase": 1.0, "per_view_smase": [1.0],
            "behavior_point_count": 10}


def _live_round(domain: str):
    """One real ``run_online_round`` with a scripted backend.  Zero LLM."""
    origin = 400
    t = np.arange(1024, dtype=np.float64)
    values = {"s0": np.sin(t / 7.0) + 0.1 * np.sin(t / 3.0) + 5.0}
    series0 = values["s0"]
    operators = ("winsorize", "outlier_mad", "hampel_filter")
    backend = sealed.SealedProbeBackend(
        explore=True, operators=operators,
        max_propose_candidates=len(operators), force_pool=True)
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(series0[:origin], task_kind="forecast"))
    method = TTHAMethod(
        TTHAFastAgent(core), gerunner._h0_snapshot(), ())
    executor = ScopeExecutor(
        [{"series_uid": "s0", "role": "train"},
         {"series_uid": "s0", "role": "eval"}],
        values, {"anchors": []}, evaluate_fn=_neutral_eval)
    run_online_round(
        method, executor,
        gerunner._a5_request(series0, values, origin, domain),
        values, origin=origin, slow_agent=None, controller=None, store=None,
        card_builder=gerunner._a5v2_card,
        round_name="guard_pipeline_%s" % domain, budget=len(operators),
        allow_slow=False, domain=domain, period=24,
        fast_features=dict(extract_public_features(
            series0[:origin], task_kind="forecast")))
    return method.experience_episodes


def test_a_the_live_online_round_writes_the_unit_id_into_the_runtime():
    """Read back out of the Runtime, not off the helper's return value."""
    for domain in ("unit_alpha", "unit_beta"):
        episodes = _live_round(domain)
        assert episodes, "the scripted round must write at least one Episode"
        assert {risk_skill._task_of(e) for e in episodes} == {domain}


# ==========================================================================
# Repair 2 -- the deduplicated count reaches the card
# ==========================================================================
def _probe(task_id: str, program: str, gain: float, condition: bool = False):
    relation = ("NEGATIVE" if gain <= -0.005
                else "POSITIVE" if gain >= 0.005 else "IMMATERIAL")
    return {
        "task_episode_id": task_id, "arm": "SOURCE", "program": program,
        "context_condition": condition, "support_gain": gain,
        "relation": relation,
        "conditioned_snapshot": False, "conditioned_served": False,
    }


def _card_from(probes):
    audit = source_skill.authorization_audit(
        probes, min_distinct_tasks=cls_harness.MIN_DISTINCT_TASKS,
        conditioning_key="conditioned_snapshot")
    return source_skill.build_skill_payload(
        _SECTIONS, skill_id=cls_harness.SOURCE_SKILL_ID,
        applicability=cls_harness.SOURCE_APPLICABILITY,
        risk_evidence=source_skill.risk_guard_rows(
            audit, condition_feature=cls_harness.CENSUS_CONDITION_KEY))


_SECTIONS = {
    "WHEN": "When task_kind == classification in a new cohort.",
    "OBSERVE": "Inspect the applicable context condition before deciding.",
    "TRY": source_skill.TRY_ABSTAIN,
    "RISK": "Repeated harm was observed for the listed program family.",
    "VERIFY": "Require this Task's own Target Support before believing it.",
    "FALLBACK": "Gather task-local Target Support rather than selecting.",
}


def test_b_two_distinct_units_of_the_same_program_count_as_two():
    card = _card_from([
        _probe(UNIT_A, "outlier_mad", HARM_GAIN),
        _probe(UNIT_B, "outlier_mad", HARM_GAIN),
    ])
    assert card["risk_guards"]["evidence_distinct_task_count"] == 2


def test_b_the_same_unit_twice_counts_as_one():
    card = _card_from([
        _probe(UNIT_A, "outlier_mad", HARM_GAIN),
        _probe(UNIT_A, "outlier_mad", -0.12),
    ])
    assert card["risk_guards"]["evidence_distinct_task_count"] == 1


def test_b_the_guard_on_the_card_is_structured_and_adds_no_prose():
    card = _card_from([
        _probe(UNIT_A, "outlier_mad", HARM_GAIN),
        _probe(UNIT_B, "outlier_mad", HARM_GAIN),
    ])
    guards = card["risk_guards"]
    assert guards["deprioritized_scoped_evidence"] == [{
        "operators": ["outlier_mad"],
        "context_scope": {
            "feature": cls_harness.CENSUS_CONDITION_KEY,
            "op": "==", "value": False,
        },
        "distinct_task_count": 2,
        "deprioritization_authorized": True,
    }]
    # The body is still exactly the six Slow-authored sections: the repair
    # added a field, not a sentence.
    assert card["body"] == "\n".join(
        "%s: %s" % (name, _SECTIONS[name]) for name in source_skill.SECTIONS)


def test_b_a_split_family_still_says_nothing():
    """The frozen clause-kind rule: a conflicted family supports neither."""
    card = _card_from([
        _probe(UNIT_A, "outlier_mad", HARM_GAIN),
        _probe(UNIT_B, "outlier_mad", HARM_GAIN),
        _probe("Ham/impulse_v2", "outlier_mad", +0.20),
    ])
    assert "evidence_distinct_task_count" not in card["risk_guards"]
    assert "deprioritized_scoped_evidence" not in card["risk_guards"]


def test_b_passing_nothing_leaves_the_pre_repair_card_untouched():
    plain = source_skill.build_skill_payload(
        _SECTIONS, skill_id=cls_harness.SOURCE_SKILL_ID,
        applicability=cls_harness.SOURCE_APPLICABILITY)
    assert set(plain["risk_guards"]) == {
        "carrier", "advises_the_proposal_stage_only",
        "never_supplies_a_candidate", "requires_target_support", "sections"}


# ==========================================================================
# Repair 3 -- the classification line compiles harm into a guard
# ==========================================================================
def _cls_state(tmp_path, units):
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = compile_snapshot(H0, verify_lock=False)
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    method = TTHAMethod(None, snapshot, ())
    for unit in units:
        method.append_experience_episode(_harm_episode(unit))
    return {"store": store, "method": method}


def test_c_two_units_of_harm_become_a_guard_the_fast_view_serves(tmp_path):
    state = _cls_state(tmp_path, (UNIT_A, UNIT_B))
    out = cls_harness._risk_lifecycle(state, arm="A5")
    assert out["candidate_families"] == ["outlier_mad"]
    assert out["risk_skill_ids"] == ["target_risk_outlier_mad"]

    snapshot = state["method"]._active_snapshot()
    view = resolve_harness_view(
        snapshot, CLASSIFICATION_FEATURES, role="fast")
    assert "target_risk_outlier_mad" in view.skill_ids

    guard = next(s for s in view.skills
                 if s.skill_id == "target_risk_outlier_mad")
    # The T1 delivery predicate: this carrier is not withheld from Fast.
    assert _is_inert_experience_card(guard) is False
    # Structured avoid, and structurally incapable of proposing.
    assert guard.skill_kind is SkillKind.SAFETY
    assert guard.allowed_tools == ()
    assert guard.risk_guards["deprioritize_only"] is True
    assert guard.risk_guards["never_supplies_a_candidate"] is True
    assert guard.risk_guards["evidence_distinct_task_count"] == 2
    assert "Frozen program steps:" not in guard.body
    assert tuple(_skill_frozen_candidates(view, CLASSIFICATION_FEATURES)) == ()


def test_c_one_unit_probed_twice_mints_nothing(tmp_path):
    """The pre-repair collapse, reproduced as the rule it is supposed to be.

    Before repair 1 every classification Episode carried the empty id, so two
    units looked exactly like this case.  The rule itself is unchanged: one
    Task is still one Task.
    """
    state = _cls_state(tmp_path, (UNIT_A, UNIT_A))
    out = cls_harness._risk_lifecycle(state, arm="A5")
    assert out["events"] == []
    assert out["risk_skill_ids"] == []
    view = resolve_harness_view(
        state["method"]._active_snapshot(), CLASSIFICATION_FEATURES,
        role="fast")
    assert not [sid for sid in view.skill_ids if sid.startswith("target_risk_")]


def test_c_the_guard_survives_into_the_next_round_of_the_same_arm(tmp_path):
    """The wiring has to move the method's own snapshot reference, not just
    the store pointer, or the next round proposes into the same dead end."""
    state = _cls_state(tmp_path, (UNIT_A, UNIT_B))
    before = state["method"]._active_snapshot().runtime_bundle_sha
    cls_harness._risk_lifecycle(state, arm="A5")
    after = state["method"]._active_snapshot()
    assert after.runtime_bundle_sha != before
    assert "target_risk_outlier_mad" in [s.skill_id for s in after.skills]


# ==========================================================================
# Reachability differential -- the r2-style re-review, zero LLM
# ==========================================================================
REAL_EPISODE_FILE = (
    PROJECT_ROOT / "artifacts/functional/e2/t6_cls_conf_dev_ecg200.json")
SYNTHETIC_SECOND_UNIT = "Wine/fit_only_artifact"


def _real_ecg200_harm_row():
    """The ECG200 ``outlier_mad`` NEGATIVE Episode this line already produced.

    Read out of the committed artifact rather than re-run: the point of the
    differential is what the *existing* evidence would have done, and re-
    probing it would spend a Consumer fit to learn nothing new.
    """
    payload = json.loads(REAL_EPISODE_FILE.read_text(encoding="utf-8"))
    for round_row in payload["rounds"]:
        for episode in round_row["episodes"]:
            if (episode["workflow_signature"] == "outlier_mad"
                    and episode["relation"] == "NEGATIVE"):
                return episode
    raise AssertionError("the ECG200 harm Episode is not in the artifact")


def _without_unit_id(episode):
    """The pre-repair Episode: identical but for the field that was missing."""
    summary = {key: value for key, value in episode.context_summary.items()
               if key != "task_episode_id"}
    return dataclasses.replace(episode, context_summary=summary)


def _two_unit_episodes():
    real = _real_ecg200_harm_row()
    first = _harm_episode(str(real["domain_namespace"]),
                          gain=float(real["support_gain"]))
    second = _harm_episode(SYNTHETIC_SECOND_UNIT, gain=-0.05)
    return real, [first, second]


def _minted_guard(episodes):
    """Run the real lifecycle on a throwaway store; return the Fast view ids."""
    with tempfile.TemporaryDirectory() as tmp:
        store = SnapshotStore(Path(tmp) / "snapshots")
        snapshot = compile_snapshot(H0, verify_lock=False)
        store.materialize(snapshot)
        store.set_active(snapshot.runtime_bundle_sha)
        method = TTHAMethod(None, snapshot, ())
        for episode in episodes:
            method.append_experience_episode(episode)
        state = {"store": store, "method": method}
        out = cls_harness._risk_lifecycle(state, arm="A5")
        view = resolve_harness_view(
            method._active_snapshot(), CLASSIFICATION_FEATURES, role="fast")
        guard = next((s for s in view.skills
                      if s.skill_id.startswith("target_risk_")), None)
        return out, list(view.skill_ids), (
            None if guard is None else {
                "skill_id": guard.skill_id,
                "skill_kind": guard.skill_kind.value,
                "allowed_tools": list(guard.allowed_tools),
                "risk_guards": dict(guard.risk_guards or {}),
                "observable_applicability": dict(
                    guard.observable_applicability),
                "supplies_a_frozen_candidate": bool(tuple(
                    _skill_frozen_candidates(view, CLASSIFICATION_FEATURES))),
            })


def _card_stage(probes, applicability):
    audit = source_skill.authorization_audit(
        probes, min_distinct_tasks=cls_harness.MIN_DISTINCT_TASKS,
        conditioning_key="conditioned_snapshot")
    rows = source_skill.risk_guard_rows(
        audit, condition_feature=cls_harness.CENSUS_CONDITION_KEY)
    from SelfEvolvingHarnessTS.contracts.harness import load_skill_entry

    before = load_skill_entry(source_skill.build_skill_payload(
        _SECTIONS, skill_id=cls_harness.SOURCE_SKILL_ID,
        applicability=applicability))
    after = load_skill_entry(source_skill.build_skill_payload(
        _SECTIONS, skill_id=cls_harness.SOURCE_SKILL_ID,
        applicability=applicability, risk_evidence=rows))
    return {
        "audit_pooled_negative": [cell["pooled_negative"] for cell in audit],
        "deprioritization_authorized": [
            cell["deprioritization_authorized"] for cell in audit],
        "guard_rows": rows,
        "before_count": (before.risk_guards or {}).get(
            "evidence_distinct_task_count"),
        "after_count": (after.risk_guards or {}).get(
            "evidence_distinct_task_count"),
        "before_inert": _is_inert_experience_card(before),
        "after_inert": _is_inert_experience_card(after),
    }


def reachability_differential():
    """The full chain, run twice: as it was, and as it is."""
    real, episodes = _two_unit_episodes()
    pre = [_without_unit_id(episode) for episode in episodes]

    pre_candidates = risk_skill.risk_candidates(
        pre, threshold=cls_harness.MATERIAL)
    post_candidates = risk_skill.risk_candidates(
        episodes, threshold=cls_harness.MATERIAL)
    post_out, post_ids, post_guard = _minted_guard(episodes)

    probes = [
        _probe(str(real["domain_namespace"]), "outlier_mad",
               float(real["support_gain"])),
        _probe(SYNTHETIC_SECOND_UNIT, "outlier_mad", -0.05),
    ]
    # A scope finer than the eligibility gate, written with a feature the
    # observable contract actually registers.  CENSUS_CONDITION_KEY is not in
    # OBSERVABLE_FEATURES, so the classification census condition cannot be an
    # applicability leaf at all -- recorded below as a structural finding.
    scoped = {"all": [
        {"feature": "task_kind", "op": "==", "value": "classification"},
        {"feature": "local_robust_z_peak", "op": "==", "value": "high"},
    ]}
    rows = [
        {
            "stage": "1. Episode carries the unit id",
            "reader": "risk_skill._task_of (risk_skill.py:72-74)",
            "before": [risk_skill._task_of(e) for e in pre],
            "after": [risk_skill._task_of(e) for e in episodes],
            "reachable_before": False, "reachable_after": True,
        },
        {
            "stage": "2. distinct harm Tasks for the family",
            "reader": "risk_skill.census (risk_skill.py:83-112)",
            "before": len(risk_skill.census(
                pre, threshold=cls_harness.MATERIAL
            )["outlier_mad"]["negative_task_ids"]),
            "after": len(risk_skill.census(
                episodes, threshold=cls_harness.MATERIAL
            )["outlier_mad"]["negative_task_ids"]),
            "reachable_before": False, "reachable_after": True,
        },
        {
            "stage": "3. guard candidate at MIN_DISTINCT=2",
            "reader": "risk_skill.risk_candidates (risk_skill.py:153-184)",
            "before": [row["family"] for row in pre_candidates],
            "after": [row["family"] for row in post_candidates],
            "reachable_before": bool(pre_candidates),
            "reachable_after": bool(post_candidates),
        },
        {
            "stage": "4. classification line compiles it",
            "reader": ("run_e2_t6_cls_op_shared_harness._risk_lifecycle -> "
                       "agentic/runner.run_risk_skill_lifecycle"),
            "before": "no call site existed",
            "after": post_out["risk_skill_ids"],
            "reachable_before": False,
            "reachable_after": bool(post_out["risk_skill_ids"]),
        },
        {
            "stage": "5. served into the Fast view",
            "reader": "retrieval.resolve_harness_view(role='fast')",
            "before": [],
            "after": [sid for sid in post_ids
                      if sid.startswith("target_risk_")],
            "reachable_before": False,
            "reachable_after": any(
                sid.startswith("target_risk_") for sid in post_ids),
        },
        {
            "stage": "6. form is a structured avoid, not a candidate",
            "reader": "risk_skill.risk_skill_payload + fast_agent."
                      "_skill_frozen_candidates",
            "before": None,
            "after": post_guard,
            "reachable_before": False,
            "reachable_after": bool(
                post_guard
                and post_guard["skill_kind"] == "safety"
                and post_guard["allowed_tools"] == []
                and not post_guard["supplies_a_frozen_candidate"]
                and post_guard["risk_guards"][
                    "evidence_distinct_task_count"] == 2),
        },
    ]
    card = {
        "as_shipped_scope": _card_stage(
            probes, cls_harness.SOURCE_APPLICABILITY),
        "with_a_scope_finer_than_the_eligibility_gate": _card_stage(
            probes, scoped),
    }
    return {
        "real_episode": real,
        "real_episode_source": str(
            REAL_EPISODE_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "synthetic_second_unit": SYNTHETIC_SECOND_UNIT,
        "safety_guard_chain": rows,
        "experience_card_chain": card,
    }


def test_the_chain_flips_from_unreachable_to_reachable_at_two_units():
    table = reachability_differential()
    assert all(not row["reachable_before"] for row in table["safety_guard_chain"])
    assert all(row["reachable_after"] for row in table["safety_guard_chain"])
    card = table["experience_card_chain"]
    # Repair 2 lands the count on the card in both scopes ...
    for branch in card.values():
        assert branch["before_count"] is None
        assert branch["after_count"] == 2
    # ... but the shipped card is scoped only to the eligibility gate, so the
    # predicate still withholds it.  Structural finding, not a repair.
    assert card["as_shipped_scope"]["after_inert"] is True
    assert card["with_a_scope_finer_than_the_eligibility_gate"][
        "after_inert"] is False


# ==========================================================================
# Artifact writer.  `python tests/functional/test_guard_pipeline_reachability.py`
# ==========================================================================
ARTIFACT_JSON = (
    PROJECT_ROOT / "artifacts/functional/e2/b_guard_pipeline_reachability.json")
ARTIFACT_MD = ARTIFACT_JSON.with_suffix(".md")

REPAIRS = [
    {
        "repair": 1,
        "site": "methods/ttha/online_loop.py:_write_target_episode",
        "change": ("context_summary gains task_episode_id = the `domain` "
                   "argument, which is the Episode's own domain_namespace"),
        "why_stable": (
            "the caller builds it from the cell identity before the round "
            "runs (classification: dataset/condition), so it is constant "
            "across probes and across r1/r2 inside one unit and changes when "
            "the unit changes"),
        "why_no_leak": (
            "it is the string already stored on the same Episode as "
            "domain_namespace, and it carries no Outcome, no delayed reading "
            "and no held-out label"),
    },
    {
        "repair": 2,
        "site": ("evaluation/functional/task_episode_harness/agentic/"
                 "source_skill.py:risk_guard_rows + build_skill_payload"),
        "change": ("risk_guards gains evidence_distinct_task_count and a "
                   "structured deprioritized_scoped_evidence row "
                   "(operators + context scope + count); the body is "
                   "byte-identical and no free text is added"),
    },
    {
        "repair": 3,
        "site": ("evaluation/functional/run_e2_t6_cls_op_shared_harness.py:"
                 "_risk_lifecycle, called from _run_round"),
        "change": ("the existing agentic/runner.run_risk_skill_lifecycle is "
                   "called on this arm's own Episodes after every round; no "
                   "new lifecycle, no new census, no new rule"),
    },
]

STRUCTURAL_FINDINGS = [
    {
        "finding": "the shipped source card is scoped only to the "
                   "eligibility gate",
        "detail": (
            "run_e2_t6_cls_op_shared_harness.SOURCE_APPLICABILITY is "
            "task_kind == classification, and retrieval._scopes_beyond_task_"
            "kind requires something finer, so repair 2 is necessary but not "
            "sufficient for the *experience card* branch of the middle tier. "
            "Narrowing that constant is a Scope change and is outside this "
            "book."),
    },
    {
        "finding": "the classification census condition is not an observable "
                   "feature",
        "detail": (
            "CENSUS_CONDITION_KEY = support_reproduces_fit_signal is absent "
            "from contracts/observables.OBSERVABLE_FEATURES, so it cannot "
            "appear in observable_applicability at all; the guard row records "
            "it inside risk_guards, which is free-form JSON. Giving the "
            "middle tier a real Context scope on this line therefore needs an "
            "observable-contract addition, not a wiring fix."),
    },
    {
        "finding": "classification Episodes carry no task_signature",
        "detail": (
            "online_loop._write_target_episode writes no "
            "context_summary.task_signature, so risk_skill.applicability_from "
            "returns {'const': True} and the minted guard is unconditioned "
            "within the arm. The frozen risk_skill rule says that is the "
            "correct reading of a family that failed under every observed "
            "Context; narrowing it means writing a signature, which is a "
            "Scope change and is outside this book."),
    },
    {
        "finding": "the guard body names the units the harm happened in",
        "detail": (
            "risk_skill.risk_skill_payload renders negative_task_ids into the "
            "body ('Tasks: ...'), so with dataset/condition ids a later unit's "
            "Fast prompt sees earlier cohort names as provenance. Applicability "
            "is still decided by observable_applicability, never by the names. "
            "Changing the body is a semantics change and is outside this book."),
    },
    {
        "finding": "tests/functional/test_skill_revocation.py does not parse "
                   "on Python 3.10",
        "detail": ("pre-existing: a multi-line f-string expression at line 166 "
                   "is 3.12+ syntax. Not touched, not caused here."),
    },
]


def _markdown(payload) -> str:
    lines = [
        "# B -- guard pipeline reachability differential",
        "",
        "Zero LLM.  Evidence grade: %s." % payload["evidence_grade"],
        "",
        "Real Episode: `%s` from `%s` (support_gain %.4f, relation %s)."
        % (payload["real_episode"]["episode_id"],
           payload["real_episode_source"],
           float(payload["real_episode"]["support_gain"]),
           payload["real_episode"]["relation"]),
        "Synthetic second unit: `%s` (same family, harm)."
        % payload["synthetic_second_unit"],
        "",
        "## The three repairs",
        "",
    ]
    for row in payload["repairs"]:
        lines.append("- **%d** `%s` -- %s" % (row["repair"], row["site"],
                                              row["change"]))
    lines += ["", "## Safety-guard chain, before vs after", "",
              "| stage | reader | before | after | before reachable | "
              "after reachable |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in payload["safety_guard_chain"]:
        before = json.dumps(row["before"], ensure_ascii=False, default=str)
        after = json.dumps(row["after"], ensure_ascii=False, default=str)
        lines.append("| %s | `%s` | %s | %s | %s | %s |" % (
            row["stage"], row["reader"],
            before if len(before) < 160 else before[:157] + "...",
            after if len(after) < 160 else after[:157] + "...",
            row["reachable_before"], row["reachable_after"]))
    lines += ["", "## Experience-card chain (repair 2)", "",
              "| card scope | count before | count after | inert before | "
              "inert after |", "| --- | --- | --- | --- | --- |"]
    for name, branch in payload["experience_card_chain"].items():
        lines.append("| %s | %s | %s | %s | %s |" % (
            name, branch["before_count"], branch["after_count"],
            branch["before_inert"], branch["after_inert"]))
    lines += ["", "## Structural findings", ""]
    for row in payload["structural_findings"]:
        lines.append("- **%s** -- %s" % (row["finding"], row["detail"]))
    lines += ["", "## Verdict", "", payload["verdict_reason"], ""]
    return "\n".join(lines)


def _write_artifact() -> int:
    import platform
    import subprocess

    table = reachability_differential()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False).stdout.strip()
    payload = {
        "protocol_version": "b_guard_pipeline_reachability_v1",
        "evidence_grade": "INFRASTRUCTURE",
        "llm_calls": 0,
        "git_head": head,
        "python": platform.python_version(),
        "generated_by": "tests/functional/test_guard_pipeline_reachability.py",
        "repairs": REPAIRS,
        "thresholds_unchanged": {
            "risk_skill.RISK_MIN_DISTINCT_TASKS": (
                risk_skill.RISK_MIN_DISTINCT_TASKS),
            "run_e2_t6_cls_op_shared_harness.MIN_DISTINCT_TASKS": (
                cls_harness.MIN_DISTINCT_TASKS),
            "run_e2_t6_cls_op_shared_harness.MATERIAL": cls_harness.MATERIAL,
        },
        **table,
        "structural_findings": STRUCTURAL_FINDINGS,
        "verdict": "GUARD_TIER_REACHABLE_AT_TWO_UNITS",
        "verdict_reason": (
            "Every stage of the safety-guard chain was unreachable before the "
            "three repairs and is reachable after them at n=2 distinct units, "
            "on one real ECG200 outlier_mad harm Episode plus one synthetic "
            "second-unit harm.  The experience-card branch now carries the "
            "count the predicate reads, but stays withheld from Fast because "
            "the shipped card's Scope is the eligibility gate itself -- a "
            "structural finding, not a repair."),
    }
    ARTIFACT_JSON.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True,
                   default=str) + "\n", encoding="utf-8")
    ARTIFACT_MD.write_text(_markdown(payload), encoding="utf-8")
    print("WROTE", ARTIFACT_JSON)
    print("WROTE", ARTIFACT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(_write_artifact())
