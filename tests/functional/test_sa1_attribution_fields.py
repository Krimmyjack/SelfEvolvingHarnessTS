"""SA-1 Part 0: the four attribution fields.

The SA-0 wiring audit found the revision loop blocked on instrumentation
rather than on mechanism: a card-supplied Episode and an agent-proposed one
are the same record, a revised card has no version stamp, "did not match" and
"matched and was refused" both read as an absent candidate, and conditioning
is derived from a unit's position rather than recorded per card.

Four pure additions, one focused test each.  Every one of them is a readout
of state the round already had; none of them is consulted by anything that
proposes, retrieves, probes or deploys.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import copy  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    load_skill_entry,
)
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    _write_target_episode,
    source_skill_of_candidate,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    source_skill as ss,
)

import run_e2_s1_curriculum_four_arms as s1  # noqa: E402

SKILL_ID = "sa1_attribution_probe_v1"


def _card(*, pattern_value: str = "high") -> dict:
    scope = {
        "task_kind": "classification",
        "consumer_id": "ridge-raw-plus-difference-v1",
        "metric": "accuracy",
        "pattern_intersection": {"local_robust_z_peak": pattern_value},
        "program_geometry": ["hampel_filter"],
    }
    return ss.build_supply_card_payload(
        skill_id=SKILL_ID, scope=scope,
        sources=[{"unit_id": "U1", "task_episode_id": "U1", "run_id": "r1",
                  "support_gain": 0.19, "delayed_gain": 0.05}])


# --------------------------------------------------------------------- (1) --
def test_a_source_skill_id_names_the_card_that_supplied_the_candidate() -> None:
    """A supplied Episode says which card supplied it; a self-proposed one
    says null.  The decode is exact rather than a guess: ``fast_agent`` mints
    ``cand_skill_<id>`` and ``source="skill:<id>"`` from the same string."""
    assert source_skill_of_candidate("cand_skill_%s" % SKILL_ID) == SKILL_ID
    assert source_skill_of_candidate("skill:%s" % SKILL_ID) == SKILL_ID
    assert source_skill_of_candidate("cand_hampel_filter_1") is None
    assert source_skill_of_candidate("identity") is None
    assert source_skill_of_candidate(None) is None

    steps = [{"op": "hampel_filter", "params": {}}]
    supplied = _write_target_episode(
        domain="U1", op="hampel_filter", program_steps=steps,
        support_gain=0.19, support_context={}, episode_id_suffix="_r1_p1",
        source_skill_id=source_skill_of_candidate("cand_skill_%s" % SKILL_ID))
    self_proposed = _write_target_episode(
        domain="U1", op="hampel_filter", program_steps=steps,
        support_gain=0.19, support_context={}, episode_id_suffix="_r1_p2",
        source_skill_id=source_skill_of_candidate("cand_hampel_filter_1"))

    assert supplied.context_summary["source_skill_id"] == SKILL_ID
    assert self_proposed.context_summary["source_skill_id"] is None
    # The collision the field exists to break: same program, same signature,
    # and before this field the two rows were indistinguishable.
    assert supplied.workflow_signature == self_proposed.workflow_signature
    # Pure addition: relation, status and evidence level are untouched.
    assert supplied.relation == self_proposed.relation
    assert supplied.local_status == self_proposed.local_status


# --------------------------------------------------------------------- (2) --
def test_b_the_revision_stamp_is_the_card_content_sha() -> None:
    """``SkillEntry.revision`` is a static authoring field -- every mint
    writes 1 -- so the version stamp is the content address instead.  A
    payload and the entry loaded from it hash to the same string, and a
    Scope narrowing moves it."""
    payload = _card()
    entry = load_skill_entry(payload)
    sha = ss.skill_content_sha(payload)

    assert sha == ss.skill_content_sha(entry)
    assert len(sha) == 64
    assert ss.skill_content_sha(_card()) == sha          # deterministic
    assert int(payload["revision"]) == 1

    narrowed = copy.deepcopy(payload)
    narrowed["observable_applicability"] = {
        "all": [payload["observable_applicability"],
                {"not": {"all": [{"feature": "period_change_score",
                                  "op": "==", "value": "zero"}]}}]}
    assert ss.skill_content_sha(narrowed) != sha
    # and the static field did not have to move for the stamp to move
    assert int(narrowed["revision"]) == int(payload["revision"])


# --------------------------------------------------------------------- (3) --
def test_c_scope_match_separates_not_matched_from_matched_and_refused() -> None:
    """Per unit, per card, the AST verdict -- which is the whole difference
    between "the Scope is too narrow" and "the card is wrong"."""
    entries = {SKILL_ID: load_skill_entry(_card())}

    admits = {"task_kind": "classification", "local_robust_z_peak": "high"}
    refuses = {"task_kind": "classification", "local_robust_z_peak": "low"}
    silent = {"task_kind": "classification"}          # leaf absent -> abstain

    assert s1._scope_match_by_skill_id(entries, admits) == {SKILL_ID: True}
    assert s1._scope_match_by_skill_id(entries, refuses) == {SKILL_ID: False}
    assert s1._scope_match_by_skill_id(entries, silent) == {SKILL_ID: False}
    assert s1._scope_match_by_skill_id({}, admits) == {}


# --------------------------------------------------------------------- (4) --
def test_d_guidance_conditioning_is_recorded_per_card_not_derived() -> None:
    """In view is what conditioning means.  Two cards installed at different
    boundaries get two answers in the same round, which the position-derived
    rule could not express."""
    other = "sa1_attribution_probe_v2"
    entries = {SKILL_ID: load_skill_entry(_card()),
               other: load_skill_entry(
                   {**_card(), "skill_id": other})}

    assert s1._guidance_conditioned_by_skill_id(entries, [SKILL_ID]) == {
        SKILL_ID: True, other: False}
    assert s1._guidance_conditioned_by_skill_id(entries, []) == {
        SKILL_ID: False, other: False}
    assert s1._guidance_conditioned_by_skill_id(entries,
                                                [SKILL_ID, other]) == {
        SKILL_ID: True, other: True}


def test_e_episode_attribution_joins_the_two_episode_side_fields() -> None:
    payload = _card()
    entries = {SKILL_ID: load_skill_entry(payload)}
    shas = {SKILL_ID: ss.skill_content_sha(payload)}
    steps = [{"op": "hampel_filter", "params": {}}]

    supplied = _write_target_episode(
        domain="U1", op="hampel_filter", program_steps=steps,
        support_gain=0.19, support_context={}, episode_id_suffix="_r1_p1",
        source_skill_id=SKILL_ID)
    self_proposed = _write_target_episode(
        domain="U1", op="hampel_filter", program_steps=steps,
        support_gain=0.19, support_context={}, episode_id_suffix="_r1_p2",
        source_skill_id=None)

    assert s1._episode_attribution(supplied, entries, shas) == {
        "source_skill_id": SKILL_ID, "source_skill_revision": shas[SKILL_ID]}
    assert s1._episode_attribution(self_proposed, entries, shas) == {
        "source_skill_id": None, "source_skill_revision": None}
