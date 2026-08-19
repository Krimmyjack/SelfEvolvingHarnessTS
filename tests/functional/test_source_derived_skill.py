"""R2: the Source census reaches Fast as something it can act on.

G3-D1 delivered its Source knowledge as free guidance text on
``candidate_policy.proposal_guidance``, and the electricity development run
measured the result: A5 and A3 were within noise on every readout and their
first-probe family distributions matched.  The guidance arrived; it carried
nothing actionable.

The census it came from does support something.  Counted in distinct Tasks:
``outlier_iqr`` is positive in six with no opposing cell, ``repair_level_shift``
under the same Context is five positive against six negative.  So there is one
guarded hypothesis and one warning to be had, and a genuinely split family in
between.  What was missing was a shape to say them in.

These checks cover the shape and its authority, not the content.  Whether the
hypothesis is any good is what Target Support decides; a rubric that scored
the wording would be scoring an expectation instead of evidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SelfEvolvingHarnessTS.contracts.observables import (  # noqa: E402
    OBSERVABLE_FEATURES,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _skill_frozen_candidates,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_NAMES  # noqa: E402
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    runner as g1r,
    source_skill,
)

FROZEN_CENSUS = (
    PROJECT_ROOT / "artifacts/functional/e2/g3d1_source_derived_skill.json"
)


@pytest.fixture(scope="module")
def census():
    if not FROZEN_CENSUS.is_file():
        pytest.skip("the frozen T233 Source census is not present")
    return json.loads(FROZEN_CENSUS.read_text(encoding="utf-8"))["census"]


SECTIONS_OK = {
    "WHEN": (
        "A forecast Task whose public inspection shows isolated extreme "
        "points rather than a sustained level change."
    ),
    "OBSERVE": (
        "Summarize the scoped series and localize regions before proposing; "
        "check whether post_shift_support_sufficient holds."
    ),
    "TRY": (
        "Lead with outlier_iqr when the observation shows isolated extremes, "
        "since that family carried repeated benefit with nothing against it."
    ),
    "RISK": (
        "Do not lead with repair_level_shift on prior expectation: the "
        "evidence is split on it and it has repeatedly cost a probe."
    ),
    "VERIFY": (
        "Believe nothing here until this Task's own Target Support shows a "
        "material gain; the note never substitutes for the probe."
    ),
    "FALLBACK": (
        "If the observation does not show isolated extremes, keep an "
        "effect-distinct candidate supported by what you observed, or abstain."
    ),
}


# ------------------------------------------------------------- the census view
def test_signed_summary_states_the_split_the_raw_census_hides(census):
    """One row per relation means a reader must join three to see a split."""
    rows = {
        (row["program"], row["context_condition"]): row
        for row in source_skill.signed_summary(census)
    }
    split = rows[("repair_level_shift", False)]
    assert split["positive_tasks"] == 5 and split["negative_tasks"] == 6
    assert split["verdict"] == "SPLIT"

    clean = rows[("outlier_iqr", False)]
    assert clean["negative_tasks"] == 0 and clean["positive_tasks"] == 6
    assert clean["verdict"] == "POSITIVE_NO_OPPOSING_CELL"


def test_vocabulary_is_what_the_census_mentions_and_nothing_else(census):
    vocabulary = source_skill.census_vocabulary(census)
    assert "outlier_iqr" in vocabulary["operators"]
    assert "post_shift_support_sufficient" in vocabulary["features"]
    # A real operator the census never observed stays outside it.
    assert "winsorize" not in vocabulary["operators"]


# ------------------------------------------------------------------ the audit
def _audit(sections, census):
    return source_skill.audit_sections(
        sections, census,
        operator_names=list(OPERATOR_NAMES),
        observable_features=list(OBSERVABLE_FEATURES),
        source_cohort_tokens=("T233", "kdd"),
    )


def test_a_sayable_skill_passes(census):
    assert _audit(SECTIONS_OK, census)["pass"]


@pytest.mark.parametrize(
    "mutation, failing_check",
    [
        ({"TRY": "Lead with winsorize when extremes are isolated."},
         "no_invented_operator"),
        ({"WHEN": "A forecast Task whose missing_fraction is elevated."},
         "no_invented_observable_feature"),
        ({"VERIFY": "Believe it once the gain clears 0.005."},
         "no_numeric_threshold"),
        ({"TRY": "outlier_iqr worked across the T233 cohort, so lead with it."},
         "no_source_cohort_identity_leaked"),
        ({"FALLBACK": "   "}, "all_six_sections_present"),
    ],
)
def test_the_audit_refuses_what_the_census_cannot_say(
    census, mutation, failing_check
):
    audit = _audit({**SECTIONS_OK, **mutation}, census)
    assert not audit["pass"]
    assert audit["checks"][failing_check] is False


def test_an_extra_section_is_refused(census):
    audit = _audit({**SECTIONS_OK, "NOTES": "anything"}, census)
    assert not audit["pass"]
    assert audit["extra_sections"] == ["NOTES"]


# ------------------------------------------------------- Slow decision paths
def _slow_returning(payload):
    def call(_messages):
        return payload
    return call


def test_abstain_writes_nothing_and_is_reported_as_a_result(census, tmp_path):
    out = g1r.run_source_skill_slow(
        repo_root=PROJECT_ROOT, census=census, census_provenance="T233 19 Tasks",
        store_root=tmp_path / "snapshots",
        slow_call=_slow_returning({"decision": "ABSTAIN", "reason": "split"}),
    )
    assert out["verdict"] == "R2_SLOW_ABSTAINED"
    assert "frozen_runtime_bundle_sha" not in out
    assert out["target_outcome_read"] is False


def test_a_skill_that_fails_the_audit_is_rejected_whole(census, tmp_path):
    bad = {**SECTIONS_OK, "TRY": "Lead with winsorize."}
    out = g1r.run_source_skill_slow(
        repo_root=PROJECT_ROOT, census=census, census_provenance="T233 19 Tasks",
        store_root=tmp_path / "snapshots",
        slow_call=_slow_returning({"decision": "ADD", "sections": bad}),
    )
    assert out["verdict"] == "R2_SLOW_SKILL_REJECTED"
    assert out["audit"]["invented_operators"] == ["winsorize"]
    assert "frozen_runtime_bundle_sha" not in out


@pytest.fixture()
def frozen(census, tmp_path):
    out = g1r.run_source_skill_slow(
        repo_root=PROJECT_ROOT, census=census, census_provenance="T233 19 Tasks",
        store_root=tmp_path / "snapshots",
        slow_call=_slow_returning({"decision": "ADD", "sections": SECTIONS_OK}),
    )
    assert out["verdict"] == "R2_SOURCE_SKILL_FROZEN", out
    return out


def test_the_frozen_skill_compiles_back_and_carries_all_six_sections(frozen):
    entry = frozen["skill_entry"]
    assert entry["skill_kind"] == "capability"
    for name in source_skill.SECTIONS:
        assert (name + ":") in entry["body"]
    assert set(entry["risk_guards"]["sections"]) == set(source_skill.SECTIONS)


def test_it_reaches_the_fast_agent_on_a_forecast_task(frozen, tmp_path):
    snapshot = compile_snapshot(
        Path(frozen["store_root"]) / frozen["frozen_runtime_bundle_sha"],
        verify_lock=False,
    )
    view = resolve_harness_view(
        snapshot,
        {"task_kind": "forecast", "post_shift_support_sufficient": False},
        role="fast",
    )
    assert source_skill.SOURCE_SKILL_ID in view.skill_ids


def test_it_advises_and_never_becomes_an_executable_candidate(frozen):
    """The difference from a Target-local capability Skill.

    A ``fast_winner_*`` entry carries frozen steps and is supplied to the pool
    as ``cand_skill_<id>``.  This one carries none, so it reaches the proposal
    stage as knowledge and the Agent still has to propose.
    """
    snapshot = compile_snapshot(
        Path(frozen["store_root"]) / frozen["frozen_runtime_bundle_sha"],
        verify_lock=False,
    )
    features = {"task_kind": "forecast", "post_shift_support_sufficient": False}
    view = resolve_harness_view(snapshot, features, role="fast")
    entry = frozen["skill_entry"]
    assert entry["allowed_tools"] == []
    assert "Frozen program steps:" not in entry["body"]
    supplied = [
        str(getattr(candidate, "candidate_id", candidate))
        for candidate in _skill_frozen_candidates(view, features)
    ]
    assert all(source_skill.SOURCE_SKILL_ID not in name for name in supplied)


def test_the_skill_is_frozen_before_any_target_outcome_is_read(frozen):
    """Ordering is the whole validity of the A5/A3 contrast.

    A Source Skill written after the Target Outcome was seen would be
    describing the answer, not transferring knowledge.
    """
    assert frozen["target_outcome_read"] is False
    assert frozen["llm_api_call_count"] == 1
    assert frozen["census_provenance"] == "T233 19 Tasks"
