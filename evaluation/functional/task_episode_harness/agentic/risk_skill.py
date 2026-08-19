"""Target-local Risk Skill: the carrier in-domain negative evidence never had.

The electricity development run made the gap concrete.  ``repair_level_shift``
was probed 14 times across 8 distinct Tasks and was negative **every** time,
for a cumulative -1.186; both arms still led with it in 7 of 9 Tasks.  Inside
one Task the Agent handles its own failure correctly -- A5 on task_05 probed
``repair_level_shift`` at -0.126, pivoted, and took ``outlier_mad`` at +0.114,
the best probe of the run -- because a probe result *is* Target Support in
that trajectory.  Across Tasks it evaporated: task_07, task_08 and task_09 all
led with ``repair_level_shift`` again.

The reason is structural, not a tuning problem.  ``_make_episode`` grades a
non-positive probe ``STATUS_EPISODE_ONLY``, and only a positive one becomes a
Target-local Skill.  Under the architecture override an Episode reaches the
Fast Agent *only* as a Skill, so an arm can accumulate success and is
constitutionally unable to learn from its own failures.

This module closes that with the narrowest thing that is still knowledge: a
``skill_kind=safety`` entry.  Three properties make it the right carrier, and
all three already existed in the contract before this module:

* ``resolve_harness_view`` retrieves SAFETY through its own branch, filtered
  by observable applicability and exempt from the capability ``top_k`` -- so
  it is Context-conditioned, not a global switch;
* ``_skill_frozen_candidates`` skips every non-CAPABILITY Skill, so a risk
  Skill is *structurally incapable* of putting a program on the table;
* it is serialized into the Fast system prompt like any other Skill.

So it can lower a family's standing and can never raise one.  That asymmetry
is deliberate and mirrors the clause-kind rule already frozen for General
guidance: a repeated harm somewhere is enough to deprioritize, and is never
enough to recommend.

What it is not: a prohibition.  The body says so, the Agent keeps the whole
Operator set, and the current Task's own Workspace evidence still overrides
it.  The claim is "do not lead with this family absent new evidence", which
is exactly what the census supports and nothing more.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# One distinct Task can be an accident of that Task; two is the smallest
# count that is about the family rather than the instance.  It matches the
# distinct-Task evidence unit used everywhere else in this harness -- attempts
# are diagnostic only, because A3 and A5 probe the same Task against one
# frozen Outcome cell.
RISK_MIN_DISTINCT_TASKS = 2

RISK_SKILL_PREFIX = "target_risk_"

# The guard the retrieval layer reads to stop recalling a disconfirmed Skill.
RESTRICTED_GUARD = "restricted_by_target_feedback"


def family_of(steps: Sequence[Mapping[str, Any]]) -> str:
    """Program family = the operator structure, parameters discarded.

    The evidence is about *which operators* keep failing here, not about one
    parameter binding, and ``repair_level_shift`` is intrinsic now anyway --
    it has no public parameters left to vary.
    """
    return "+".join(str(step["op"]) for step in steps)


def _signature(episode: Any) -> dict[str, Any]:
    summary = getattr(episode, "context_summary", None) or {}
    return dict(summary.get("task_signature") or {})


def _task_of(episode: Any) -> str:
    summary = getattr(episode, "context_summary", None) or {}
    return str(summary.get("task_episode_id") or "")


def _family_of_episode(episode: Any) -> str:
    summary = getattr(episode, "context_summary", None) or {}
    geometry = summary.get("program_geometry") or {}
    return family_of(geometry.get("program_steps") or ())


def census(episodes: Sequence[Any], *, threshold: float) -> dict[str, dict[str, Any]]:
    """Signed per-family evidence over one arm's own Episodes.

    Deterministic and zero-LLM.  Counted in distinct Tasks, and the Task
    signatures are carried along so applicability can be derived from the
    Tasks that actually produced the harm.
    """
    rows: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        family = _family_of_episode(episode)
        if not family:
            continue
        gain = float(
            (getattr(episode, "support_response", None) or {}).get("gain") or 0.0
        )
        task_id = _task_of(episode)
        row = rows.setdefault(family, {
            "family": family,
            "negative_task_ids": set(),
            "positive_task_ids": set(),
            "negative_gains": [],
            "negative_signatures": [],
        })
        if gain >= threshold:
            row["positive_task_ids"].add(task_id)
        else:
            row["negative_task_ids"].add(task_id)
            row["negative_gains"].append(gain)
            row["negative_signatures"].append(_signature(episode))
    return rows


def _shared_signature(signatures: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The Context the failures actually share, and nothing wider.

    Only keys present with an identical value in *every* failing Task survive.
    A key that varied across the failures is not evidence about Context and
    must not narrow -- or widen -- where the Skill applies.
    """
    if not signatures:
        return {}
    shared = dict(signatures[0])
    for other in signatures[1:]:
        shared = {
            key: value for key, value in shared.items()
            if key in other and other[key] == value
        }
    return shared


def applicability_from(signature: Mapping[str, Any]) -> dict[str, Any]:
    """Task signature -> observable applicability AST.

    ``task_signature`` is already a binned, closed-vocabulary projection of
    the public features, so every leaf here is Workspace-observable by
    construction.  With nothing shared the Skill still applies -- ``const``
    true -- because a family that failed under *every* observed Context is
    more general evidence, not less.  It stays a deprioritization either way.
    """
    leaves = [
        {"feature": str(key), "op": "==", "value": value}
        for key, value in sorted(signature.items())
    ]
    if not leaves:
        return {"const": True}
    if len(leaves) == 1:
        return leaves[0]
    return {"all": leaves}


def risk_candidates(
    episodes: Sequence[Any],
    *,
    threshold: float,
    min_distinct_tasks: int = RISK_MIN_DISTINCT_TASKS,
) -> list[dict[str, Any]]:
    """Families this arm's own history says not to lead with.

    Both halves of the rule matter.  ``>= min_distinct_tasks`` negative makes
    it about the family; *no* positive anywhere keeps the rule silent exactly
    where the evidence is mixed.  On T233 ``repair_level_shift`` was 6
    negative against 5 positive and would correctly produce nothing here --
    a conflicted family is a question for Slow, not a deprioritization.
    """
    out: list[dict[str, Any]] = []
    for family, row in sorted(census(episodes, threshold=threshold).items()):
        negative = row["negative_task_ids"]
        positive = row["positive_task_ids"]
        if positive or len(negative) < min_distinct_tasks:
            continue
        shared = _shared_signature(row["negative_signatures"])
        out.append({
            "family": family,
            "skill_id": RISK_SKILL_PREFIX + family.replace("+", "_"),
            "distinct_negative_task_count": len(negative),
            "negative_task_ids": sorted(negative),
            "worst_gain": min(row["negative_gains"]),
            "cumulative_gain": sum(row["negative_gains"]),
            "shared_task_signature": shared,
            "observable_applicability": applicability_from(shared),
        })
    return out


def contradicted_risk_families(
    episodes: Sequence[Any],
    *,
    threshold: float,
) -> set[str]:
    """Families that later earned a material-positive in this arm.

    A risk Skill minted before that evidence arrived is now making a claim its
    own Domain has disproved, so it must stop being recalled rather than sit
    in the snapshot outvoting fresh evidence.
    """
    return {
        family for family, row in census(episodes, threshold=threshold).items()
        if row["positive_task_ids"]
    }


def risk_skill_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """The ``skill-entry/1`` value for one risk Skill.

    ``allowed_tools`` is empty and the body carries no ``Frozen program
    steps:`` marker -- both are load-time invariants for SAFETY, and together
    they are what makes "cannot propose" a property of the entry rather than
    a promise about the caller.
    """
    tasks = ", ".join(candidate["negative_task_ids"])
    family = candidate["family"]
    return {
        "schema_version": "skill-entry/1",
        "skill_id": str(candidate["skill_id"]),
        "skill_kind": "safety",
        "revision": 1,
        "body": (
            "Target-local risk: the program family "
            + repr(str(family))
            + " was probed in "
            + str(candidate["distinct_negative_task_count"])
            + " distinct Tasks in this Domain under this Context and produced"
            " no material benefit in any of them (worst "
            + format(float(candidate["worst_gain"]), ".4f")
            + ", cumulative "
            + format(float(candidate["cumulative_gain"]), ".4f")
            + "; Tasks: " + tasks + "). "
            "Do not lead with this family here on prior expectation alone. "
            "This is a deprioritization, not a prohibition: it is still "
            "allowed, and current-Task Workspace evidence that specifically "
            "indicates it overrides this note. Prefer spending the first "
            "probe on an effect-distinct family that the current observation "
            "supports; if this family is chosen anyway, cite the current "
            "observation that justifies it. Target Support in this Task, not "
            "this note, decides the outcome."
        ),
        "observable_applicability": dict(candidate["observable_applicability"]),
        "allowed_tools": [],
        "risk_guards": {
            "deprioritize_only": True,
            "not_a_prohibition": True,
            "never_supplies_a_candidate": True,
            "overridable_by_current_observation": True,
            "evidence_distinct_task_count": int(
                candidate["distinct_negative_task_count"]),
        },
    }
