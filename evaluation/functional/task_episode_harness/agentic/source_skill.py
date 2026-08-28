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

import json
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

# The literal a Slow stage must use when no active recommendation is
# authorized.  An explicit abstention is a section; silence is not.
TRY_ABSTAIN = "NO_AUTHORIZED_ACTIVE_RECOMMENDATION"


_SLOW_SYSTEM_TEMPLATE = (
    "You are the Slow Harness update stage. Exactly one Harness surface is "
    "authorized this call: an ADD of the skill library entry "
    f"'{SOURCE_SKILL_ID}'. You do not approve your own edit; a deterministic "
    "audit validates it and a paired replay decides whether it survives.\n"
    "You receive a complete, de-duplicated evidence census over every "
    "canonical program, public Context condition and outcome relation "
    "observed in a Source domain. No trajectories and no utility numbers are "
    "provided, so do not invent thresholds.\n"
    "Evidence rules. Evidence is counted in distinct_task_count, never in "
    "attempt_count. A statement that warns against something needs repeated "
    "harm somewhere and does not need a uniformly positive cohort. A program "
    "the census is genuinely split on supports neither.\n"
    "Active recommendations are pre-authorized for you, deterministically, "
    "and the list is in the payload as authorized_try_operators. Evidence "
    "produced while a Skill was already naming a family is not independent "
    "of the Harness: it may confirm, refute or withdraw a clause and can "
    "never authorize a new active recommendation, so it has already been "
    "excluded. TRY may name only operators on that list. If the list is "
    "empty, TRY must be exactly the string "
    f"'{TRY_ABSTAIN}' and must name no operator at all -- that is a result, "
    "not a failure, and the other sections still stand on their own rules.\n"
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


def _slow_system_template(skill_id: str | None = None) -> str:
    """The frozen forecasting template, with an optional Skill id override.

    The default path returns the import-time string unchanged so T233
    artifacts keep seeing the same bytes.  An override only replaces the
    quoted Skill id; every other sentence stays put.
    """
    if not skill_id or skill_id == SOURCE_SKILL_ID:
        return _SLOW_SYSTEM_TEMPLATE
    return _SLOW_SYSTEM_TEMPLATE.replace(
        "'%s'" % SOURCE_SKILL_ID, "'%s'" % skill_id, 1)


def slow_system(authorized: Sequence[str],
                skill_id: str | None = None) -> str:
    """The Slow system prompt with this call's authorization stated in it."""
    listed = ", ".join(sorted(authorized)) if authorized else "(empty)"
    return (
        _slow_system_template(skill_id)
        + "\nauthorized_try_operators for this call: " + listed + "."
    )


# --------------------------------------------------------- provenance rules
LOCAL_SKILL_PREFIX = "fast_winner_e1v2_"
BOOTSTRAP_SKILL_IDS = frozenset({
    "build_contrastive_candidates", "inspect_and_localize",
    "select_or_identity_and_verify",
})

def _families_named_by(skill_ids: Sequence[str]) -> set[str]:
    return {
        str(sid)[len(LOCAL_SKILL_PREFIX):]
        for sid in skill_ids or ()
        if str(sid).startswith(LOCAL_SKILL_PREFIX)
    }


def _relation(gain: float, threshold: float) -> str:
    if gain >= threshold:
        return "POSITIVE"
    if gain <= -threshold:
        return "NEGATIVE"
    return "IMMATERIAL"


def provenance_labelled_probes(
    report: Mapping[str, Any],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Every Source probe, labelled by what the Fast payload held before it.

    Evidence produced while a Skill was already naming that Program family is
    not independent of the Harness.  The T233 census made the difference
    decisive rather than academic: ``outlier_iqr`` looks like six positive
    Tasks against nothing, and five of those six happened after
    ``fast_winner_e1v2_outlier_iqr`` was already in the arm's snapshot.  One
    Skill formed, then re-confirmed itself five times.

    Two readings are produced.  ``conditioned_snapshot`` counts every Skill in
    the arm's snapshot when the Task began; ``conditioned_served`` counts only
    the ones the resolved view actually served.  The first is the conservative
    one -- withholding an authorization on a doubt is the safe direction --
    and it is what :func:`authorization_audit` uses by default.
    """
    probes: list[dict[str, Any]] = []
    for row in report.get("rows") or ():
        task_id = str(row["task_episode_id"])
        condition = bool(row.get("post_shift_support_sufficient"))
        for arm in ("A3", "A5"):
            entry = row.get(arm)
            if not entry:
                continue
            summary = entry.get("retrieved_knowledge_summary") or {}
            in_snapshot = _families_named_by(
                entry.get("active_local_skill_ids_before") or ())
            served = _families_named_by([
                sid for sid in (summary.get("retrieved_skill_ids") or ())
                if sid not in BOOTSTRAP_SKILL_IDS
            ])
            prior = bool(summary.get("source_prior_matched"))
            for probe in entry.get("probes") or ():
                if probe.get("status") != "PROBED":
                    continue
                operators = [str(step["op"]) for step in probe["steps"]]
                gain = float(probe["support_gain"])
                probes.append({
                    "task_episode_id": task_id,
                    "arm": arm,
                    "program": "+".join(operators),
                    "context_condition": condition,
                    "support_gain": gain,
                    "relation": _relation(gain, threshold),
                    # A Skill naming one operator conditions any Program that
                    # contains it, so a compound stays conditioned too.
                    "conditioned_snapshot": bool(in_snapshot & set(operators)) or prior,
                    "conditioned_served": bool(served & set(operators)) or prior,
                })
    return probes


def authorization_audit(
    probes: Sequence[Mapping[str, Any]],
    *,
    min_distinct_tasks: int,
    conditioning_key: str = "conditioned_snapshot",
) -> list[dict[str, Any]]:
    """Per program x Context: what the evidence may and may not authorize.

    Three rules, applied in this order and none of them tunable here:

    * an active recommendation may be authorized only by UNGUIDED distinct
      Tasks -- conditioned evidence may confirm, refute or withdraw a clause
      and may never authorize a new one;
    * after removing any one Source Task the UNGUIDED positive count must
      still reach ``min_distinct_tasks``, so no recommendation rests on a
      single Task;
    * opposing evidence blocks from either provenance, precisely because
      conditioned evidence is allowed to refute.

    A deprioritization is graded on the other side of the frozen clause-kind
    asymmetry: repeated harm somewhere is enough, and it does not need a
    uniformly positive cohort, so provenance does not gate it.
    """
    cells: dict[tuple[str, Any], dict[str, dict[str, set[str]]]] = {}
    for probe in probes:
        key = (probe["program"], probe["context_condition"])
        cell = cells.setdefault(key, {
            "pooled": {}, "unguided": {}, "conditioned": {},
        })
        bucket = "conditioned" if probe[conditioning_key] else "unguided"
        for name in ("pooled", bucket):
            cell[name].setdefault(probe["relation"], set()).add(
                probe["task_episode_id"])

    out: list[dict[str, Any]] = []
    for (program, condition), cell in sorted(cells.items(),
                                             key=lambda kv: (kv[0][0], str(kv[0][1]))):
        unguided_positive = cell["unguided"].get("POSITIVE", set())
        pooled_negative = cell["pooled"].get("NEGATIVE", set())
        pooled_positive = cell["pooled"].get("POSITIVE", set())
        loo_minimum = (
            min(len(unguided_positive - {task}) for task in unguided_positive)
            if unguided_positive else 0
        )
        try_authorized = bool(
            unguided_positive
            and loo_minimum >= min_distinct_tasks
            and not pooled_negative
        )
        risk_authorized = bool(
            len(pooled_negative) >= min_distinct_tasks and not pooled_positive
        )
        out.append({
            "program": program,
            "context_condition": condition,
            "pooled_positive": len(pooled_positive),
            "pooled_negative": len(pooled_negative),
            "pooled_immaterial": len(cell["pooled"].get("IMMATERIAL", set())),
            "unguided_positive": len(unguided_positive),
            "unguided_negative": len(cell["unguided"].get("NEGATIVE", set())),
            "conditioned_positive": len(cell["conditioned"].get("POSITIVE", set())),
            "conditioned_negative": len(cell["conditioned"].get("NEGATIVE", set())),
            "leave_one_out_minimum_positive": loo_minimum,
            "active_try_authorized": try_authorized,
            "deprioritization_authorized": risk_authorized,
            "withheld_because": (
                None if try_authorized
                else "no_unguided_positive" if not unguided_positive
                else "opposing_evidence_in_this_context" if pooled_negative
                else "does_not_survive_leave_one_out"
            ),
        })
    return out


def authorized_try_operators(audit: Sequence[Mapping[str, Any]]) -> set[str]:
    """Operators a TRY clause may name.  Empty means TRY must abstain."""
    operators: set[str] = set()
    for cell in audit:
        if cell["active_try_authorized"]:
            operators.update(str(cell["program"]).split("+"))
    return operators


# ------------------------------------------------------- P0: the supply tier
# The permission ladder has two graded exits on the positive side and this is
# the lower one.  ``authorization_audit`` above is the TRY tier: it authorizes
# an *active recommendation*, and its leave-one-out floor means a card needs
# three unguided positive Tasks before it may name an operator to prefer.
# Nothing about it changes here.
#
# The supply tier authorizes strictly less: one candidate placed in the pool
# for the Target to verify, with ``grants_execution=false`` and
# ``requires_target_support=true``, so Support and the delayed gate keep every
# decision.  Two independent unguided positives is what that costs.  The two
# tiers share their clause vocabulary -- unguided evidence only, opposing
# evidence blocks -- and differ in exactly one parameter, the count.
# Ladder revision v2 (2026-08-28, sol proposal / user approved / main line
# entered into canon): the evidence price is set to what the permission is
# worth.  A supply-only card asks for the least authority on the ladder -- it
# places one candidate for the Target to verify, executes nothing and deploys
# nothing -- so one *strong* positive buys it: Support and delayed both
# POSITIVE on the same Episode.  Two independent unguided positives still buy
# the intersection-Scope Source card, and the TRY tier's leave-one-out floor
# in ``authorization_audit`` above is untouched.
#
# Anti-bootstrap is unchanged and is what makes the low price safe: a positive
# earned while this card was already in view is Harness-conditioned and counts
# zero, so a single-Episode card can never license its own promotion.
SUPPLY_TIER_MIN_DISTINCT_TASKS = 1

SUPPLY_CARD_KIND = "source_supply_tier/1"

# The five axes a supplied candidate is scoped on.  The first three are
# identity of the Task the evidence was earned against; the fourth is the
# deployment-visible Pattern the evidence shares; the fifth is the Program
# geometry the card carries frozen.
SUPPLY_SCOPE_AXES = ("task_kind", "consumer_id", "metric",
                     "pattern_intersection", "program_geometry")


def skill_content_sha(entry: Any) -> str:
    """A card's content address -- SA-1's version stamp.

    ``SkillEntry.revision`` is a static authoring field: every mint writes the
    literal 1 and nothing in the repository increments it (SA-0 audit A-2), so
    a revised card would still read ``revision: 1`` and no ledger could say
    which version was in view.  The store already addresses snapshots by
    content, and this is the same idea one level down: canonicalize the card
    exactly as the compiler serializes it and hash those bytes.  Nothing here
    writes to the card, so no mint site changes.
    """
    import hashlib

    from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
    from SelfEvolvingHarnessTS.contracts.harness import load_skill_entry
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        skill_entry_to_dict,
    )

    # Both shapes go through the same serializer, so the stamp a card payload
    # gets before it is installed and the stamp the installed entry gets are
    # the same string.
    loaded = load_skill_entry(dict(entry)) if isinstance(entry, Mapping) else entry
    return hashlib.sha256(
        canonical_json_bytes(skill_entry_to_dict(loaded))).hexdigest()


def _distinct(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["task_episode_id"]) for row in rows}


def supply_tier_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_distinct_tasks: int = SUPPLY_TIER_MIN_DISTINCT_TASKS,
    conditioning_key: str = "conditioned_snapshot",
) -> list[dict[str, Any]]:
    """Per Program family: may the supply tier speak, and on what evidence?

    Three rules, and each is the same clause the TRY tier already applies --
    only the count differs:

    * evidence produced while a Skill already named the family is not
      independent of the Harness, so a conditioned positive counts zero;
    * ``>= min_distinct_tasks`` *distinct* unguided positive Tasks; one Task
      is an accident of that Task, and repeating it does not make it two;
    * an unresolved opposing reading in the same family blocks, from either
      provenance.  ``authorization_audit`` blocks the TRY tier on exactly this
      and the supply tier takes the same, most conservative, reading: a family
      the evidence is split on supports neither clause.

    No leave-one-out.  That floor is what distinguishes the TRY tier, and
    lowering or raising it here would silently move the other tier's line.
    """
    families: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        family = str(row["program"])
        bucket = families.setdefault(
            family, {"positive_unguided": [], "positive_conditioned": [],
                     "negative": [], "immaterial": []})
        relation = str(row.get("relation") or "").upper()
        if relation == "POSITIVE":
            key = ("positive_conditioned" if row.get(conditioning_key)
                   else "positive_unguided")
        elif relation == "NEGATIVE":
            key = "negative"
        else:
            key = "immaterial"
        bucket[key].append(row)

    out: list[dict[str, Any]] = []
    for family, bucket in sorted(families.items()):
        unguided = _distinct(bucket["positive_unguided"])
        conditioned = _distinct(bucket["positive_conditioned"])
        negative = _distinct(bucket["negative"])
        enough = len(unguided) >= int(min_distinct_tasks)
        authorized = bool(enough and not negative)
        out.append({
            "program": family,
            "unguided_positive_tasks": sorted(unguided),
            "unguided_positive": len(unguided),
            "conditioned_positive": len(conditioned),
            "opposing_negative": len(negative),
            "opposing_negative_tasks": sorted(negative),
            "immaterial": len(_distinct(bucket["immaterial"])),
            "min_distinct_tasks": int(min_distinct_tasks),
            "supply_authorized": authorized,
            "withheld_because": (
                None if authorized
                else "opposing_evidence_in_the_same_family" if negative
                else "fewer_than_%d_distinct_unguided_positive_tasks"
                     % int(min_distinct_tasks)),
        })
    return out


# --------------------------------------------------------------------------- #
# Scope rule v2 -- the Pattern *family* axis (SA-1 Part 0.5)
# --------------------------------------------------------------------------- #
# The degenerate case of the intersection rule below is what L1 paid for: with
# one source Episode the "intersection" is that Episode's entire recorded
# Pattern view, incidental leaves included, and one such leaf
# (``period_change_score``) decided two of three non-matches.  Scope rule v2
# replaces the Pattern axis of a *supply* card with the family the evidence
# belongs to.
#
# The family definition is not invented here and is not chosen by looking at
# what would have matched.  It is the Pattern intersection S1a already used to
# decide whether a Program cluster qualifies as a family at all --
# ``run_e2_s1a_curriculum_oracle_audit._compatible_clusters``
# (``:619-652``, intersection helper ``_intersect_maps`` at ``:609-616``),
# frozen months before L1 in
# ``artifacts/functional/e2/s1a_curriculum_audit.json`` at
# ``part_b.clusters[].pattern_intersection``.  S1a's gate reads: >= 2
# independent positives sharing Task/Consumer, one Program geometry, and a
# non-empty deployment-visible Pattern intersection.  Those shared leaves are
# exactly "what this defect family looks like"; a leaf only one member carries
# is by construction not in them.
S1A_AUDIT_RELPATH = ("artifacts/functional/e2/s1a_curriculum_audit.json")
PATTERN_FAMILY_AXIS_KIND = "s1a_cluster_pattern_intersection/1"


def s1a_pattern_family_leaves(program: str, *, audit_path: Any
                              ) -> dict[str, Any] | None:
    """The frozen Pattern leaves that define ``program``'s defect family.

    ``None`` when S1a found no qualifying cluster for that Program -- a
    single-member "cluster" carries the whole member's view and is marked
    ``compatible: false`` there, so it is not a family and must not be used
    as one.  A missing definition is a stop, never an invitation to compile
    the family here.
    """
    from pathlib import Path as _Path

    audit = json.loads(_Path(audit_path).read_text(encoding="utf-8"))
    clusters = (audit.get("part_b") or {}).get("clusters") or []
    for cluster in clusters:
        if str(cluster.get("program")) != str(program):
            continue
        if not cluster.get("compatible"):
            return None
        leaves = dict(cluster.get("pattern_intersection") or {})
        return leaves or None
    return None


def five_axis_scope(rows: Sequence[Mapping[str, Any]],
                    *, pattern_family: Mapping[str, Any] | None = None
                    ) -> dict[str, Any] | None:
    """The Scope the supplying evidence actually shares, and nothing wider.

    Task identity must be identical across the sources -- a card that spans
    two Consumers is making a claim its evidence never tested.  Returns
    ``None`` when the identity axes disagree.

    The Pattern axis has two rules.  Without ``pattern_family`` it keeps only
    leaves present with the same value in *every* source, the same
    intersection rule ``risk_skill._shared_signature`` uses on the guard side;
    this is the Source-card rule and is untouched.  With ``pattern_family``
    (Scope rule v2, supply tier only) the axis *is* the family, and every
    source must agree with the family on every one of its leaves -- an
    Episode that disagrees is not a member and buys no family-wide card.
    """
    if not rows:
        return None
    first = rows[0]
    identity = {axis: first.get(axis) for axis in
                ("task_kind", "consumer_id", "metric")}
    for row in rows[1:]:
        if any(row.get(axis) != identity[axis] for axis in identity):
            return None
    programs = {str(row["program"]) for row in rows}
    if len(programs) != 1:
        return None
    if pattern_family is None:
        pattern = dict(first.get("pattern") or {})
        for row in rows[1:]:
            other = dict(row.get("pattern") or {})
            pattern = {key: value for key, value in pattern.items()
                       if key in other and other[key] == value}
        axis_kind = "source_intersection/1"
    else:
        pattern = dict(pattern_family)
        for row in rows:
            view = dict(row.get("pattern") or {})
            if any(key not in view or view[key] != value
                   for key, value in pattern.items()):
                return None
        axis_kind = PATTERN_FAMILY_AXIS_KIND
    return {
        "task_kind": identity["task_kind"],
        "consumer_id": identity["consumer_id"],
        "metric": identity["metric"],
        "pattern_intersection": pattern,
        "pattern_axis_kind": axis_kind,
        "program_geometry": list(programs.pop().split("+")),
    }


def _edit_schema_features(project_root: Any) -> frozenset[str]:
    """Leaf names an edit manifest may carry.

    ``contracts/observables.OBSERVABLE_FEATURES`` is a superset of
    ``contracts/schemas/observable_feature_v1.json``.  PS-1 hit this: dumping
    the raw intersection into ``observable_applicability`` fails shape
    validation before anything runs.  The drift itself is recorded as a
    finding elsewhere; here the compiler simply keeps the machine AST inside
    what the schema accepts and reports what it dropped.
    """
    import json as _json
    from pathlib import Path as _Path

    schema_path = (_Path(project_root) / "contracts" / "schemas"
                   / "observable_feature_v1.json")
    schema = _json.loads(schema_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for option in schema.get("oneOf") or []:
        feature = (option.get("properties") or {}).get("feature") or {}
        if "const" in feature:
            names.add(str(feature["const"]))
        names.update(str(item) for item in (feature.get("enum") or []))
    return frozenset(names)


def supply_applicability(
    scope: Mapping[str, Any],
    *,
    legal_features: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Scope -> machine AST, plus the leaves the edit schema cannot carry."""
    legal = frozenset(str(name) for name in legal_features) \
        if legal_features is not None else None
    leaves = [{"feature": "task_kind", "op": "==",
               "value": str(scope["task_kind"])}]
    dropped: list[str] = []
    for key, value in sorted(dict(scope["pattern_intersection"]).items()):
        # Q11: the intersection already carries task_kind (the identity
        # axis is copied into the Pattern view).  The leaf above is the
        # only one; historical cards are not rewritten.
        if key == "task_kind":
            continue
        if legal is not None and key not in legal:
            dropped.append(str(key))
            continue
        leaves.append({"feature": str(key), "op": "==", "value": value})
    return {"all": leaves}, dropped


def build_supply_card_payload(
    *,
    skill_id: str,
    scope: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    legal_features: Sequence[str] | None = None,
    revision: int = 1,
) -> dict[str, Any]:
    """The supply-tier card.  Mechanical template fill, no LLM.

    Every sentence below is a slot filled from the audit and the Scope.  A
    Slow stage authoring prose is what the TRY tier needs, because a TRY
    clause is an argument; a supplied candidate is not an argument, it is a
    Program plus the Scope it was earned in, and templating it removes the
    one place a model could quietly widen the claim.
    """
    pattern = dict(scope["pattern_intersection"])
    geometry = list(scope["program_geometry"])
    applicability, dropped = supply_applicability(
        scope, legal_features=legal_features)
    provenance = "; ".join(
        "%s (Support %+.4f, delayed %+.4f)"
        % (row["unit_id"], float(row["support_gain"]),
           float(row["delayed_gain"]))
        for row in sources)
    steps = [{"op": op, "params": {}} for op in geometry]
    body = "\n".join([
        "WHEN: task_kind == %s, consumer %s, metric %s, and the deployment-"
        "visible pattern reads %s."
        % (scope["task_kind"], scope["consumer_id"], scope["metric"],
           ", ".join("%s=%s" % (key, value)
                     for key, value in sorted(pattern.items()))),
        "OBSERVE: before deciding, read those same pattern features in the "
        "current Workspace and check whether what they describe is present "
        "here too.",
        "SUPPLY: one candidate is placed in this round's candidate pool for "
        "verification -- %s. It occupies a slot inside the existing candidate "
        "cap, it is not a recommendation, and it carries no right to deploy."
        % ", ".join(geometry),
        "EVIDENCE: %d independent prior domains improved under this program "
        "in the same direction -- %s. n = %d. Independent agreeing domains at "
        "this count establish a candidate worth one Target probe, not a fact."
        % (len(sources), provenance, len(sources)),
        "VERIFY: this holds here only if this Target's own held-in Support "
        "reads materially positive and the delayed feedback approves the "
        "Draft. Neither is assumed from the prior domains.",
        "FALLBACK: if Support or delayed refuses, drop the candidate and "
        "return to identity rather than retrying the program.",
        "Frozen program steps: " + json.dumps(steps, separators=(",", ":")),
    ])
    return {
        "schema_version": "skill-entry/1",
        "skill_id": str(skill_id),
        "skill_kind": "capability",
        "revision": int(revision),
        "body": body,
        "observable_applicability": applicability,
        "allowed_tools": [],
        "risk_guards": {
            "card_kind": SUPPLY_CARD_KIND,
            "authority": {
                "reorders_supplied_candidates": False,
                "supplies_candidates": True,
                "suppresses_operators": False,
                "grants_execution": False,
            },
            "requires_target_support": True,
            "execution_right": "withheld_supplies_candidate_only",
            "scope_v1": {
                "task_kind": scope["task_kind"],
                "consumer_id": scope["consumer_id"],
                "metric": scope["metric"],
                "pattern_intersection": pattern,
                # Which rule filled the Pattern axis.  A reader has to be able
                # to tell a family axis from one Episode's whole recorded view
                # without re-deriving it.
                "pattern_axis_kind": scope.get("pattern_axis_kind",
                                               "source_intersection/1"),
                "pattern_axis_provenance": scope.get(
                    "pattern_axis_provenance"),
                "program_geometry": geometry,
            },
            "pattern_leaves_dropped_as_uncontracted_for_edit_schema": dropped,
            # Q7: axes the recorded Scope names but the edit schema cannot
            # carry.  Pure addition -- matching still uses the machine AST
            # only.  Empty when every intersection leaf is contracted.
            "scope_unreachable_axes": list(dropped),
            "evidence": {
                "tier": "supply",
                "source_count": len(sources),
                "min_distinct_tasks": SUPPLY_TIER_MIN_DISTINCT_TASKS,
                "sources": [
                    {"unit_id": row["unit_id"],
                     "task_episode_id": row.get("task_episode_id"),
                     "run_id": row.get("run_id"),
                     "support_gain": row["support_gain"],
                     "delayed_gain": row["delayed_gain"],
                     # SA-1 R2 compiles an exclusion by comparing a refusing
                     # unit against the units this card was earned on, so the
                     # card has to carry their binned views rather than send
                     # a later reader back to a run artifact.
                     "pattern_view": dict(row.get("pattern") or {}),
                     "direction": "improved"}
                    for row in sources],
                "uncertainty": (
                    "n=%d agreeing unguided domains; a candidate worth one "
                    "probe, not a fact" % len(sources)),
            },
            "counting_rule": (
                "a positive earned under this card is a Target-local Skill "
                "only and counts zero toward any cross-domain authorization "
                "for this Source Skill"),
        },
    }


def compile_supply_tier(
    rows: Sequence[Mapping[str, Any]],
    *,
    skill_id: str,
    min_distinct_tasks: int = SUPPLY_TIER_MIN_DISTINCT_TASKS,
    conditioning_key: str = "conditioned_snapshot",
    legal_features: Sequence[str] | None = None,
    pattern_family: Mapping[str, Any] | None = None,
    pattern_axis_provenance: str | None = None,
) -> dict[str, Any]:
    """The supply-tier exit: Episodes in, one card or a stated refusal out.

    Deterministic end to end.  When nothing qualifies the result carries
    ``withheld_because`` and no card, which is a result rather than a
    failure -- the same shape the TRY tier uses when its audit authorizes
    nothing.
    """
    audit = supply_tier_audit(rows, min_distinct_tasks=min_distinct_tasks,
                              conditioning_key=conditioning_key)
    eligible = [row for row in audit if row["supply_authorized"]]
    if not eligible:
        return {"tier": "supply", "audit": audit, "card": None, "scope": None,
                "withheld_because": (
                    audit[0]["withheld_because"] if len(audit) == 1
                    else "no_program_family_met_the_supply_rule")}
    if len(eligible) > 1:
        # Two families qualifying at once is a Slow question, not something a
        # template should resolve by picking one.
        return {"tier": "supply", "audit": audit, "card": None, "scope": None,
                "withheld_because": "more_than_one_family_qualified"}
    family = eligible[0]
    # Sorted, so the compiled bytes are a function of the evidence and not of
    # the order it happened to be read in.
    sources = sorted(
        (row for row in rows
         if str(row["program"]) == family["program"]
         and str(row.get("relation") or "").upper() == "POSITIVE"
         and not row.get(conditioning_key)),
        key=lambda row: (str(row["task_episode_id"]), str(row.get("run_id"))))
    scope = five_axis_scope(sources, pattern_family=pattern_family)
    if scope is None:
        return {"tier": "supply", "audit": audit, "card": None, "scope": None,
                "withheld_because": (
                    "a source Episode is outside the Pattern family"
                    if pattern_family is not None
                    else "identity_axes_disagree_across_sources")}
    if pattern_axis_provenance:
        scope["pattern_axis_provenance"] = str(pattern_axis_provenance)
    if not scope["pattern_intersection"]:
        return {"tier": "supply", "audit": audit, "card": None,
                "scope": scope,
                "withheld_because": "pattern_intersection_empty"}
    card = build_supply_card_payload(
        skill_id=skill_id, scope=scope, sources=sources,
        legal_features=legal_features)
    return {"tier": "supply", "audit": audit, "scope": scope, "card": card,
            "withheld_because": None}


def risk_guard_rows(
    audit: Sequence[Mapping[str, Any]],
    *,
    condition_feature: str,
) -> list[dict[str, Any]]:
    """The harm the census can speak about, as structured guard rows.

    A projection, not a second counter and not a second rule.
    ``pooled_negative`` is already the number of distinct ``task_episode_id``
    values the harm repeated across -- the same evidence unit
    ``risk_skill._task_of`` counts in -- so nothing here recomputes it.

    One half of the frozen clause-kind rule is applied, because it decides
    whether the census may say anything at all here: a cell the census is
    split on supports neither clause, so it produces no row.  The other half,
    how many distinct Tasks make a harm repeated, is *not* applied -- the row
    reports the count it found and leaves the comparison to
    ``authorization_audit`` and to the retrieval predicate, which already own
    that number.

    Each row is operator names, the observable Context leaf the cell was
    counted under, and that count.  No prose: a Fast reader that has to parse
    a sentence to find out what is deprioritized is exactly what the inert
    card predicate was written to stop serving.
    """
    rows: list[dict[str, Any]] = []
    for cell in audit:
        if not int(cell["pooled_negative"]) or int(cell["pooled_positive"]):
            continue
        rows.append({
            "operators": sorted(set(str(cell["program"]).split("+"))),
            "context_scope": {
                "feature": str(condition_feature),
                "op": "==",
                "value": cell["context_condition"],
            },
            "distinct_task_count": int(cell["pooled_negative"]),
            "deprioritization_authorized": bool(
                cell["deprioritization_authorized"]),
        })
    return sorted(rows, key=lambda row: (row["operators"],
                                         str(row["context_scope"]["value"])))


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
    authorized_try: Sequence[str] | None = None,
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
    text = " ".join(
        "" if str(sections.get(name) or "").strip() == TRY_ABSTAIN
        else str(sections.get(name) or "")
        for name in SECTIONS
    )
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
    # The TRY clause is checked against what the provenance audit authorized,
    # not against what the pooled census appears to support.  Passing None
    # keeps the pre-authorization behaviour for callers that have no audit.
    try_text = str(sections.get("TRY") or "")
    named_in_try = sorted(
        name for name in operator_names if name.lower() in try_text.lower()
    )
    if authorized_try is None:
        try_ok, unauthorized = True, []
    elif not authorized_try:
        try_ok = try_text.strip() == TRY_ABSTAIN
        unauthorized = named_in_try
    else:
        unauthorized = [n for n in named_in_try if n not in set(authorized_try)]
        try_ok = not unauthorized and bool(named_in_try)

    checks = {
        "all_six_sections_present": not missing,
        "no_extra_sections": not extra,
        "no_invented_operator": not invented_operators,
        "no_invented_observable_feature": not invented_features,
        "no_numeric_threshold": not numbers,
        "no_source_cohort_identity_leaked": not leaked_cohort,
        "try_clause_within_authorization": try_ok,
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
        "operators_named_in_try": named_in_try,
        "unauthorized_operators_in_try": unauthorized,
        "authorized_try_operators": sorted(authorized_try or ()),
        "try_abstained": try_text.strip() == TRY_ABSTAIN,
    }


def build_skill_payload(
    sections: Mapping[str, Any],
    *,
    skill_id: str | None = None,
    applicability: Mapping[str, Any] | None = None,
    risk_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The ``skill-entry/1`` value for the Source-derived General Skill.

    ``allowed_tools`` is empty and the body carries no frozen program, so
    ``_skill_frozen_candidates`` yields nothing: this Skill reaches the
    proposal stage as knowledge and never becomes an executable candidate on
    its own.  The structured sections are duplicated into ``risk_guards``
    because ``skill-entry/1`` has a closed field set and that is the only
    free-form JSON on it -- the body is what the Agent reads.

    ``skill_id`` / ``applicability`` default to the frozen forecasting
    constants.  Passing them is how the AD wrapper names its own entry
    without rewriting this module's defaults.

    ``risk_evidence`` carries the deprioritizations the deterministic
    authorization audit already granted (see :func:`risk_guard_rows`).  The
    retrieval layer reads ``risk_guards.evidence_distinct_task_count`` to
    decide whether a card's RISK clause is repeated evidence or free prose;
    until this argument existed the compiler never wrote that field, so the
    clause was unreadable there no matter what the census said and every such
    card resolved as inert.  Passing nothing keeps the pre-existing bytes.
    """
    body = "\n".join(
        "%s: %s" % (name, str(sections[name]).strip()) for name in SECTIONS
    )
    guards: dict[str, Any] = {
        "carrier": "source_derived_general_skill",
        "advises_the_proposal_stage_only": True,
        "never_supplies_a_candidate": True,
        "requires_target_support": True,
        "sections": {name: str(sections[name]).strip() for name in SECTIONS},
    }
    rows = [dict(row) for row in (risk_evidence or ())]
    if rows:
        guards["deprioritized_scoped_evidence"] = rows
        guards["evidence_distinct_task_count"] = max(
            int(row["distinct_task_count"]) for row in rows)
    return {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id or SOURCE_SKILL_ID,
        "skill_kind": "capability",
        "revision": 1,
        "body": body,
        "observable_applicability": dict(applicability or SOURCE_APPLICABILITY),
        "allowed_tools": [],
        "risk_guards": guards,
    }
