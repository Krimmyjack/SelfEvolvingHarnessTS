"""R2: turn the Source census into a Skill the Fast Agent can act on.

The G3-D1 Slow call was authorized on exactly one surface -- free guidance text
on ``candidate_policy.proposal_guidance`` -- and what came back was procedural
discipline.  The electricity development run then measured the consequence:
A5 and A3 were within noise on every readout, and A5's first-probe family
distribution was the same as A3's.  The guidance was delivered; it just never
told the Agent anything it could act on.

The census it was derived from does contain something actionable, and says so
in distinct Tasks:

    outlier_iqr          cond=False  POSITIVE   6 Tasks, no opposing cell
    hampel_filter        cond=False  POSITIVE   4 Tasks, 1 opposing
    repair_level_shift   cond=False  NEGATIVE   6 Tasks
    repair_level_shift   cond=False  POSITIVE   5 Tasks

So the Source evidence supports one guarded positive hypothesis and one risk
clause, and is genuinely conflicted about the family the Agent keeps reaching
for.  What was missing was a shape in which those can be said.

This module gives the Slow stage that shape: six named sections, and a Skill
entry to put them in.

* ``WHEN``     -- the observable Context in which this applies at all
* ``OBSERVE``  -- what to look at in the Workspace before deciding
* ``TRY``      -- the guarded positive hypothesis
* ``RISK``     -- what the Source evidence warns against
* ``VERIFY``   -- what Target Support has to show before it is believed
* ``FALLBACK`` -- what to do when OBSERVE does not support TRY

The entry is ``skill_kind=capability`` and carries no ``Frozen program steps:``
marker, so ``_skill_frozen_candidates`` produces nothing from it: it advises
the proposal stage and never bypasses it.  That is the difference from a
Target-local capability Skill, which does supply its frozen program, and from
a SAFETY Skill, which may only deprioritize.

ABSTAIN is a legitimate result.  If the Slow stage declines, or if what it
returns fails the deterministic audit, nothing is written -- the Skill is
never hand-authored to make the round produce something.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

SOURCE_SKILL_ID = "source_investigation_v1"

SECTIONS = ("WHEN", "OBSERVE", "TRY", "RISK", "VERIFY", "FALLBACK")

# Every Task in both the Source and the Target cohort is a forecast Task, so
# this is the honest scope of the whole census and nothing narrower is earned.
# The finer condition lives in WHEN, as text the Agent reads and applies --
# that is the point of a General Skill, as opposed to a retrieval gate that
# would silently withhold it.
SOURCE_APPLICABILITY: dict[str, Any] = {
    "feature": "task_kind", "op": "==", "value": "forecast",
}

SLOW_SYSTEM = (
    "You are the Slow Harness update stage. Exactly one Harness surface is "
    "authorized this call: an ADD of the skill library entry "
    f"'{SOURCE_SKILL_ID}'. You do not approve your own edit; a deterministic "
    "audit validates it and a paired replay decides whether it survives.\n"
    "You receive a complete, de-duplicated evidence census over every "
    "canonical program, public Context condition and outcome relation "
    "observed in a Source domain. No trajectories and no utility numbers are "
    "provided, so do not invent thresholds.\n"
    "Evidence rules. Evidence is counted in distinct_task_count, never in "
    "attempt_count. A statement that actively tells the proposal stage to try "
    "something needs a cohort that is positive with no opposing cell in the "
    "same Context; one opposing cell is enough to withhold it. A statement "
    "that warns against something needs repeated harm somewhere and does not "
    "need a uniformly positive cohort. A program the census is genuinely "
    "split on supports neither.\n"
    "You may not introduce a new observable feature, a new operator, a new "
    "numeric threshold, or any statement about the Judge or the Consumer. "
    "Name only operators and features that appear in the census. Write for a "
    "different domain than the one the census came from: say what to observe "
    "and what would have to hold, never that a family worked in some named "
    "cohort.\n"
    "Return JSON only. Either\n"
    "{'decision':'ADD','sections':{'WHEN':'...','OBSERVE':'...','TRY':'...',"
    "'RISK':'...','VERIFY':'...','FALLBACK':'...'},'reason':'...'}\n"
    "or {'decision':'ABSTAIN','reason':'...'}.\n"
    "Each section is one or two sentences. WHEN is the observable Context in "
    "which any of this applies. OBSERVE is what to look at in the current "
    "Workspace before deciding. TRY is the guarded positive hypothesis. RISK "
    "is what the census warns against. VERIFY is what this Task's own Target "
    "Support must show before the result is believed. FALLBACK is what to do "
    "when OBSERVE does not support TRY."
)


def census_vocabulary(census: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Operators and Context features the census actually mentions.

    The audit is a containment check against this, so a Skill can only talk
    about things the evidence talked about.
    """
    operators = {
        str(op).lower()
        for cell in census for op in (cell.get("canonical_program") or ())
    }
    features = {
        str(key).lower() for cell in census for key in cell
        if key not in {
            "canonical_program", "contains_mechanism_operator",
            "support_relation", "distinct_task_count",
            "distinct_task_episode_ids", "attempt_count",
        }
    }
    features.add("task_kind")
    return {"operators": operators, "features": features}


def signed_summary(census: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per program x Context, the signed picture in one row.

    The raw census is one row per relation, so a reader has to join three rows
    to see that a family is split.  This states it directly, which is what the
    clause-kind rules are actually about.
    """
    grouped: dict[tuple[str, Any], dict[str, Any]] = {}
    for cell in census:
        program = "+".join(str(op) for op in (cell.get("canonical_program") or ()))
        condition = next(
            (value for key, value in cell.items()
             if key not in {
                 "canonical_program", "contains_mechanism_operator",
                 "support_relation", "distinct_task_count",
                 "distinct_task_episode_ids", "attempt_count"}),
            None,
        )
        row = grouped.setdefault((program, condition), {
            "program": program, "context_condition": condition,
            "positive_tasks": 0, "negative_tasks": 0, "immaterial_tasks": 0,
        })
        relation = str(cell.get("support_relation") or "").upper()
        count = int(cell.get("distinct_task_count") or 0)
        if relation == "POSITIVE":
            row["positive_tasks"] += count
        elif relation == "NEGATIVE":
            row["negative_tasks"] += count
        else:
            row["immaterial_tasks"] += count
    out = []
    for row in grouped.values():
        positive, negative = row["positive_tasks"], row["negative_tasks"]
        row["verdict"] = (
            "SPLIT" if positive and negative
            else "POSITIVE_NO_OPPOSING_CELL" if positive
            else "NEGATIVE_ONLY" if negative
            else "IMMATERIAL_ONLY"
        )
        out.append(row)
    return sorted(out, key=lambda item: (item["program"],
                                         str(item["context_condition"])))


_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


def audit_sections(
    sections: Mapping[str, Any],
    census: Sequence[Mapping[str, Any]],
    *,
    operator_names: Sequence[str],
    observable_features: Sequence[str],
    source_cohort_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    """Deterministic containment audit.  No rubric about what it should say.

    This checks that the Skill is *sayable from the census* and is usable in a
    different domain.  It deliberately does not grade content: whether the
    hypothesis is any good is what Target Support is for, and a rubric that
    scored the wording would be scoring my own expectations.
    """
    missing = [name for name in SECTIONS
               if not str(sections.get(name) or "").strip()]
    extra = sorted(set(sections) - set(SECTIONS))
    text = " ".join(str(sections.get(name) or "") for name in SECTIONS)
    lowered = text.lower()

    vocabulary = census_vocabulary(census)
    invented_operators = sorted(
        name for name in operator_names
        if name.lower() in lowered and name.lower() not in vocabulary["operators"]
    )
    invented_features = sorted(
        name for name in observable_features
        if name.lower() in lowered and name.lower() not in vocabulary["features"]
    )
    numbers = _NUMBER.findall(text)
    leaked_cohort = sorted(
        token for token in source_cohort_tokens if token.lower() in lowered
    )
    checks = {
        "all_six_sections_present": not missing,
        "no_extra_sections": not extra,
        "no_invented_operator": not invented_operators,
        "no_invented_observable_feature": not invented_features,
        "no_numeric_threshold": not numbers,
        "no_source_cohort_identity_leaked": not leaked_cohort,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "missing_sections": missing,
        "extra_sections": extra,
        "invented_operators": invented_operators,
        "invented_observable_features": invented_features,
        "numeric_literals": numbers,
        "leaked_cohort_tokens": leaked_cohort,
    }


def build_skill_payload(sections: Mapping[str, Any]) -> dict[str, Any]:
    """The ``skill-entry/1`` value for the Source-derived General Skill.

    ``allowed_tools`` is empty and the body carries no frozen program, so
    ``_skill_frozen_candidates`` yields nothing: this Skill reaches the
    proposal stage as knowledge and never becomes an executable candidate on
    its own.  The structured sections are duplicated into ``risk_guards``
    because ``skill-entry/1`` has a closed field set and that is the only
    free-form JSON on it -- the body is what the Agent reads.
    """
    body = "\n".join(
        "%s: %s" % (name, str(sections[name]).strip()) for name in SECTIONS
    )
    return {
        "schema_version": "skill-entry/1",
        "skill_id": SOURCE_SKILL_ID,
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": dict(SOURCE_APPLICABILITY),
        "allowed_tools": [],
        "risk_guards": {
            "carrier": "source_derived_general_skill",
            "advises_the_proposal_stage_only": True,
            "never_supplies_a_candidate": True,
            "requires_target_support": True,
            "sections": {name: str(sections[name]).strip() for name in SECTIONS},
        },
    }
