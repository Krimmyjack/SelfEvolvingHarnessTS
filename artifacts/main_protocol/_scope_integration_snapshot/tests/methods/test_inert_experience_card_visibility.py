"""C40 first-fault: an experience card with nothing to authorize is withheld
from the Fast proposal view.

C40 (``artifacts/functional/e2/t6_cls_op_r2_three_arms.json``) served A5 the
Slow-compiled card ``source_investigation_cls_v1``.  Its TRY was the abstention
sentinel, so the execution layer correctly supplied no candidate; its RISK was
free prose that still named ``repair_level_shift``, and both A5 rounds proposed
that family and were rejected while A3 reached ``hampel_filter``
(A5 - A3 = -0.269 held-out accuracy).

Two checks, no LLM: the predicate's own boundary, and the real card.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "methods" / "ttha"),
    str(PROJECT_ROOT / "evaluation" / "functional"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pytest  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    SkillKind,
    load_skill_entry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from evaluation.functional.task_episode_harness.agentic.source_skill import (  # noqa: E402
    TRY_ABSTAIN,
    build_skill_payload,
)

H0_ROOT = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
C40_REPORT = (
    PROJECT_ROOT / "artifacts/functional/e2/t6_cls_op_r2_three_arms.json"
)
CLASSIFICATION_FEATURES = {"task_kind": "classification"}

# The exact shape the Slow stage emits, so the checks are about authorization
# and not about a hand-written card.
_SECTIONS = {
    "WHEN": "When task_kind == classification in a new cohort.",
    "OBSERVE": "Inspect task_kind and the applicable context_condition.",
    "TRY": TRY_ABSTAIN,
    "RISK": (
        "The census gives no unguided positive support for any listed "
        "program, and the lone negative result for repair_level_shift is not "
        "repeated; do not infer a transferable preference."
    ),
    "VERIFY": "Require this Task's own Target Support before believing it.",
    "FALLBACK": "Gather task-local Target Support rather than selecting.",
}
_AUTHORIZED_SECTIONS = {
    **_SECTIONS,
    "TRY": (
        "Lead with hampel_filter when the observation shows isolated extreme "
        "points, since that family carried repeated unguided benefit."
    ),
}


def _card(sections, skill_id):
    return load_skill_entry(
        build_skill_payload(
            sections,
            skill_id=skill_id,
            applicability={
                "feature": "task_kind", "op": "==", "value": "classification",
            },
        )
    )


def _view_ids(*entries, role="fast"):
    h0 = compile_snapshot(H0_ROOT, verify_lock=False)
    snapshot = dataclasses.replace(h0, skills=(*h0.skills, *entries))
    return resolve_harness_view(
        snapshot, CLASSIFICATION_FEATURES, role=role
    ).skill_ids


def test_only_the_card_with_nothing_to_authorize_leaves_the_fast_view():
    inert = _card(_SECTIONS, "source_investigation_inert_v1")
    authorized = _card(_AUTHORIZED_SECTIONS, "source_investigation_try_v1")
    bootstrap = tuple(
        skill.skill_id
        for skill in compile_snapshot(H0_ROOT, verify_lock=False).skills
        if skill.skill_kind is SkillKind.BOOTSTRAP_PROCEDURE
    )
    assert bootstrap, "h0 must carry the operator-neutral procedural Skills"

    fast = _view_ids(inert, authorized)
    assert inert.skill_id not in fast
    assert authorized.skill_id in fast
    # The procedural bootstrap Skills carry no clause to authorize and are
    # operator-neutral, so the predicate must not reach them.
    assert set(bootstrap) <= set(fast)
    assert set(bootstrap) <= set(_view_ids())

    # Withheld from the proposal view only: Slow still resolves it, so what it
    # said stays auditable and revisable.
    assert inert.skill_id in _view_ids(inert, authorized, role="slow")


def test_the_c40_source_card_is_excluded_by_the_predicate():
    if not C40_REPORT.is_file():
        pytest.skip("the C40 three-arm report is not present")
    payload = json.loads(C40_REPORT.read_text(encoding="utf-8"))
    entry = payload["part_b"]["source_skill_entry"]
    assert entry["skill_id"] == "source_investigation_cls_v1"
    assert entry["risk_guards"]["sections"]["TRY"] == TRY_ABSTAIN
    assert "repair_level_shift" in entry["risk_guards"]["sections"]["RISK"]

    card = load_skill_entry(entry)
    assert card.skill_id not in _view_ids(card)
    assert card.skill_id in _view_ids(card, role="slow")
